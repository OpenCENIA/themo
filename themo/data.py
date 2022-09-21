import collections
import functools
import inspect
import operator
import os
import pathlib
import re
import types
import typing as tp
import warnings
import xml.etree.ElementTree as ET

import datasets
import joblib
import numpy as np
import numpy.typing as npt
import PIL.Image
import pyarrow as pa
import pyarrow.csv
import pytorch_lightning as pl
import requests
import torch
import tqdm
import transformers
import typing_extensions as tpx

__all__ = [
    "WITParallel",
    "LitWITParallel",
    "TARGET_FEATURES_MODEL",
    "WITTranslated",
    "LitWITTranslated",
    "LitParallel",
]

TARGET_FEATURES_MODEL = "openai/clip-vit-large-patch14"
BERT_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-uncased"

_TQDM_WIDTH = 120
_NS = types.SimpleNamespace


# TODO: name field is sometimes just a name (a str) or sometimes a full pathlib.Path,
# maybe we should fix this
class _FileMeta(tp.NamedTuple):
    name: tp.Union[str, pathlib.Path]
    size: int
    nrows: int


class _ParallelItem(tp.NamedTuple):
    source: str
    target: str
    target_features: npt.NDArray[np.float32]


class _Batch(tp.NamedTuple):
    input: transformers.BatchEncoding
    target: torch.Tensor


def _collate_wit_items(
    batch: tp.Sequence[_ParallelItem],
    tokenize: tp.Callable[[tp.Sequence[str]], transformers.BatchEncoding],
) -> _Batch:
    source_sentences, _, target_features = zip(*batch)
    return _Batch(tokenize(source_sentences), torch.tensor(np.stack(target_features)))


def build_parallel(files: tp.List[_FileMeta], langs: tp.Tuple[str, str]) -> pa.Table:
    """Builds a parallel dataset of the form (image_url, langs[0]:caption,
    langs[1]:caption) out of some .tsv files. Expect each row of the .tsv file
    to have columns (language, image_url, caption_reference_description), rest
    are ignored.

    Aditionally logs progress, and reading of files is done in row batches
    using pyarrow, for efficiency.

    Reading of tsv files is performed using constant memory, but the resulting
    image-caption tuples use variable memory since they are stored in a simple
    list. Assumes the resulting number of caption pairs is small enough to fit
    in memory.
    """
    print("Building parallel dataset")
    source_lang, target_lang = langs
    total_rows = sum(f.nrows for f in files)

    # maps image urls to caption dicts, (lang, caption) key-value pairs
    image_to_captions: tp.MutableMapping[
        str, tp.MutableMapping[str, str]
    ] = collections.defaultdict(dict)

    with tqdm.tqdm(
        desc="Processing rows",
        unit="rows",
        unit_scale=True,
        total=total_rows,
        ncols=_TQDM_WIDTH,
    ) as progress:
        for fpath, *_ in files:
            for batch in pa.csv.open_csv(
                fpath,
                parse_options=pa.csv.ParseOptions(
                    delimiter="\t", newlines_in_values=True
                ),
                convert_options=pa.csv.ConvertOptions(
                    include_columns=(
                        "language",
                        "image_url",
                        "caption_reference_description",
                    )
                ),
            ):
                for row in batch.to_pylist():
                    row_lang = row["language"]
                    caption = row["caption_reference_description"]
                    if row_lang in langs and caption:
                        image_to_captions[row["image_url"]][row_lang] = caption
                    progress.update()

    return pa.Table.from_pylist(
        [
            {
                "key": key,
                "source": captions[source_lang],
                "target": captions[target_lang],
            }
            for key, captions in image_to_captions.items()
            if set(langs).issubset(captions.keys())
        ]
    )


def compute_target_features(
    target_sentences: tp.Sequence[pa.StringScalar],
    clip_version: str,
    batch_size: int,
    num_workers: int,
) -> npt.NDArray[np.float32]:
    print("Computing target features, this might take a while")

    print(f"Loading CLIP tokenizer for {clip_version}")
    tokenizer = transformers.CLIPTokenizer.from_pretrained(clip_version)

    def preprocess(
        sentences: tp.Sequence[pa.StringScalar],
    ) -> transformers.BatchEncoding:
        sentences = [sentence.as_py() for sentence in sentences]
        return tokenizer(
            sentences,
            truncation=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt",
        )

    dloader: torch.utils.data.DataLoader[
        transformers.BatchEncoding
    ] = torch.utils.data.DataLoader(
        tp.cast(torch.utils.data.Dataset[transformers.BatchEncoding], target_sentences),
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=preprocess,
    )

    print(f"Loading CLIPTextModel version {clip_version}")
    model = transformers.CLIPTextModel.from_pretrained(clip_version)
    model = model.eval()
    model.requires_grad_(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("No GPU found, switching to CPU mode", RuntimeWarning)
    model.to(device)
    results = []
    for batch in tqdm.tqdm(
        dloader,
        desc="Computing features",
        unit="batch",
        ncols=_TQDM_WIDTH,
    ):
        features = model(**batch.to(device)).pooler_output.cpu().data.numpy()
        results.append(features)
    return np.concatenate(results)


class WITParallel(torch.utils.data.Dataset[_ParallelItem]):
    """Interface to WIT parallel dataset.

    Downloads (or expects) files in this filesystem structure:

    <datadir>
    └── wit
        ├── wit_v1.test.all-00000-of-00005.tsv.gz
        ├── wit_v1.test.all-00001-of-00005.tsv.gz
        ├── ...
        ├── wit_v1.train.all-00000-of-00010.tsv.gz
        ├── wit_v1.train.all-00001-of-00010.tsv.gz
        ├── ...
        ├── wit_v1.val.all-00000-of-00005.tsv.gz
        └── ...

    .. note::
       Even when using download=True, it will refuse to replace already
       existing files.
    """

    META = _NS(
        dataset_name="wit",
        baseurl="https://storage.googleapis.com/gresearch/wit/",
        files=_NS(
            train=[
                _FileMeta(
                    "wit_v1.train.all-00000-of-00010.tsv.gz", 2672819495, 3708026
                ),
                _FileMeta(
                    "wit_v1.train.all-00001-of-00010.tsv.gz", 2667931762, 3702075
                ),
                _FileMeta(
                    "wit_v1.train.all-00002-of-00010.tsv.gz", 2669251466, 3701785
                ),
                _FileMeta(
                    "wit_v1.train.all-00003-of-00010.tsv.gz", 2670373763, 3706924
                ),
                _FileMeta(
                    "wit_v1.train.all-00004-of-00010.tsv.gz", 2668172723, 3701161
                ),
                _FileMeta(
                    "wit_v1.train.all-00005-of-00010.tsv.gz", 2673104331, 3708106
                ),
                _FileMeta(
                    "wit_v1.train.all-00006-of-00010.tsv.gz", 2670156092, 3704684
                ),
                _FileMeta(
                    "wit_v1.train.all-00007-of-00010.tsv.gz", 2669891774, 3703736
                ),
                _FileMeta(
                    "wit_v1.train.all-00008-of-00010.tsv.gz", 2669091199, 3705646
                ),
                _FileMeta(
                    "wit_v1.train.all-00009-of-00010.tsv.gz", 2670659115, 3704243
                ),
            ],
            val=[
                _FileMeta("wit_v1.val.all-00000-of-00005.tsv.gz", 40332966, 52561),
                _FileMeta("wit_v1.val.all-00001-of-00005.tsv.gz", 40092981, 52342),
                _FileMeta("wit_v1.val.all-00002-of-00005.tsv.gz", 39643004, 51805),
                _FileMeta("wit_v1.val.all-00003-of-00005.tsv.gz", 39958628, 52096),
                _FileMeta("wit_v1.val.all-00004-of-00005.tsv.gz", 39943476, 52220),
            ],
            test=[
                _FileMeta("wit_v1.test.all-00000-of-00005.tsv.gz", 32291578, 42049),
                _FileMeta("wit_v1.test.all-00001-of-00005.tsv.gz", 32153532, 42070),
                _FileMeta("wit_v1.test.all-00002-of-00005.tsv.gz", 31963406, 42085),
                _FileMeta("wit_v1.test.all-00003-of-00005.tsv.gz", 32038958, 42058),
                _FileMeta("wit_v1.test.all-00004-of-00005.tsv.gz", 32133564, 41904),
            ],
        ),
    )

    parallel_items: pa.Table
    target_features: npt.NDArray[np.float32]

    def __init__(
        self,
        datadir: str,
        split: tpx.Literal["train", "val", "test"],
        langs: tp.Tuple[str, str] = ("es", "en"),
        download: bool = False,
        clip_version: str = TARGET_FEATURES_MODEL,
    ) -> None:
        self.datadir = datadir
        self.split = split
        self.langs = langs
        self.clip_version = clip_version

        print(f"Loading {self!r}")

        if download:
            self.download(datadir, split)

        memory = joblib.Memory(datadir, verbose=0)
        # I am not too sold on using filedata._replace here, but it gets the job done
        self.parallel_items = memory.cache(build_parallel)(
            [
                filedata._replace(
                    name=pathlib.Path(datadir) / self.META.dataset_name / filedata.name
                )
                for filedata in getattr(self.META.files, split)
            ],
            langs,
        )

        # Compute CLIP embeddings
        self.target_features = memory.cache(
            compute_target_features, ignore=["batch_size", "num_workers"]
        )(
            self.parallel_items["target"],
            clip_version=self.clip_version,
            batch_size=512,
            num_workers=4,
        )

    def __getitem__(self, key: int) -> _ParallelItem:
        source = self.parallel_items["source"][key]
        target = self.parallel_items["target"][key]
        features = self.target_features[key]
        return _ParallelItem(str(source), str(target), features)

    def __len__(self) -> int:
        return len(self.parallel_items)

    @classmethod
    def download(cls, datadir: str, split=None):
        basepath = pathlib.Path(datadir) / cls.META.dataset_name
        basepath.mkdir(exist_ok=True, parents=True)

        splits = (split,) if split else ("train", "val", "test")

        print(f"Downloading files to {basepath}, this might take a while...")
        all_files = [
            fdata for split in splits for fdata in getattr(cls.META.files, split)
        ]
        if all((basepath / filedata.name).exists() for filedata in all_files):
            print(f"All files already in {basepath}, skipping...")
            return
        for filedata in all_files:
            full_path = basepath / filedata.name
            if full_path.exists():
                print(f"found {full_path}, won't download")
                continue

            total_written = total_downloaded = 0
            with open(full_path, "wb") as outfile, requests.get(
                cls.META.baseurl + filedata.name, stream=True
            ) as response, tqdm.tqdm(
                desc=filedata.name,
                unit="b",
                unit_scale=True,
                total=filedata.size,
                ncols=_TQDM_WIDTH,
            ) as progress:
                for chunk in response.iter_content(10 * 2**20):  # 10MB
                    written = outfile.write(chunk)

                    total_downloaded += len(chunk)
                    total_written += written
                    progress.update(len(chunk))

            assert total_written == total_downloaded == filedata.size, (
                f"something doesn't match for file {filedata.name}: "
                f"total_written={total_written}, "
                f"total_downloaded={total_downloaded}, "
                f"filedata.size={filedata.size}"
            )
        print("Done!")

    def __repr__(self) -> str:
        ignore_list = ("download",)
        return (
            f"{type(self).__name__}("
            + ", ".join(
                f"{arg}={getattr(self, arg, 'failed')!r}"
                for arg in inspect.signature(self.__init__).parameters.keys()
                if arg not in ignore_list
            )
            + ")"
        )


def translate(
    sentences: tp.Sequence[pa.StringScalar],
    model_name: str,
    batch_size: int,
) -> pa.Array:
    print("Translating sentences, this could take some time...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        warnings.warn("No GPU found, switching to CPU mode", RuntimeWarning)

    print(f"Loading tokenizer for {model_name}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    print(f"Loading model {model_name}")
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()

    def collate_fn(batch: tp.Sequence[pa.StringScalar]) -> transformers.BatchEncoding:
        as_str = [str(item) for item in batch]
        return tokenizer(
            as_str,
            padding=True,
            truncation=True,
            max_length=model.config.max_length,
            return_tensors="pt",
        )

    dataloader = torch.utils.data.DataLoader(
        sentences,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
    )

    translated = []
    with torch.no_grad():
        for batch in tqdm.tqdm(
            dataloader, desc="Translating sentences", unit="batch", ncols=_TQDM_WIDTH
        ):
            batch.to(device)
            output_ids = model.generate(**batch, max_new_tokens=100)
            translated += tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    return pa.array(translated)


class WITTranslated(WITParallel):
    def __init__(
        self,
        datadir: str,
        split: tpx.Literal["train", "val", "test"],
        langs: tp.Tuple[str, str] = ("es", "en"),
        download: bool = False,
        clip_version: str = TARGET_FEATURES_MODEL,
    ) -> None:
        self.datadir = datadir
        self.split = split
        self.langs = langs
        self.clip_version = clip_version

        print(f"Loading {self!r}")

        if download:
            self.download(datadir, split)

        memory = joblib.Memory(datadir, verbose=0)
        # Load original parallel items
        # ============================
        og_parallel_items = memory.cache(build_parallel)(
            [
                filedata._replace(
                    name=pathlib.Path(datadir) / self.META.dataset_name / filedata.name
                )
                for filedata in getattr(self.META.files, split)
            ],
            langs,
        )
        keys, sources, targets = og_parallel_items.itercolumns()
        column_names = og_parallel_items.column_names

        # Prepare stuff for translation
        # =============================================
        translation_model_template = "Helsinki-NLP/opus-mt-{}-{}"
        batch_size = 64

        # Translate source sentences to target lang
        # =========================================
        source_translation = memory.cache(translate, ignore=["batch_size"])(
            sources,
            translation_model_template.format(*langs),
            batch_size=batch_size,
        )
        source_translated_table = pa.table(
            [keys, sources, source_translation], names=column_names
        )

        # Translate target sentences to source lang
        # =========================================
        target_translation = memory.cache(translate, ignore=["batch_size"])(
            targets,
            translation_model_template.format(*langs[::-1]),
            batch_size=batch_size,
        )
        target_translated_table = pa.table(
            [keys, target_translation, targets], names=column_names
        )

        # Build final table
        self.parallel_items = pa.concat_tables(
            [source_translated_table, target_translated_table]
        )
        # Compute target features using the final table
        self.target_features = memory.cache(
            compute_target_features, ignore=["batch_size", "num_workers"]
        )(
            self.parallel_items["target"],
            clip_version=self.clip_version,
            batch_size=512,
            num_workers=4,
        )


class LitWITParallel(pl.LightningDataModule):
    _dataset_cls: tp.ClassVar[type] = WITParallel

    # these attrs are set in __init__, and work mostly as hparams
    datadir: str
    batch_size: int
    max_sequence_length: int
    tokenizer_name: str
    # these attrs are set in setup method
    # splits are optional because some might not be present depending on stage
    train_split: tp.Optional[WITParallel]
    val_split: tp.Optional[WITParallel]
    test_split: tp.Optional[WITParallel]

    tokenizer: transformers.BertTokenizer
    _collate: tp.Callable[[tp.Sequence[_ParallelItem]], _Batch]

    def __init__(
        self,
        datadir: str,
        batch_size: int,
        max_sequence_length: int,
        tokenizer_name: str = BERT_MODEL_NAME,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))

        self.datadir = datadir
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer_name = tokenizer_name

    def prepare_data(self) -> None:
        self._dataset_cls.download(self.datadir)

    def setup(
        self, stage: tp.Optional[tpx.Literal["fit", "validate", "test"]] = None
    ) -> None:
        if stage in ("fit", "validate", None):
            self.val_split = self._dataset_cls(self.datadir, "val")
        if stage in ("fit", None):
            self.train_split = self._dataset_cls(self.datadir, "train")
        if stage in ("test", None):
            self.test_split = self._dataset_cls(self.datadir, "test")

        # always load tokenizer
        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.tokenizer_name)
        self._collate = functools.partial(
            _collate_wit_items,
            tokenize=functools.partial(
                self.tokenizer,
                padding="max_length",
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="pt",
            ),
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        # if we train with absurd amounts of data, possibly don't shuffle
        return torch.utils.data.DataLoader(
            dataset=self.train_split,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            dataset=self.val_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            dataset=self.test_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate,
        )


class LitWITTranslated(LitWITParallel):
    _dataset_cls = WITTranslated


# ParallelDatasets


def _collate_parallel_items(
    batch: tp.Sequence[_ParallelItem],
    tokenize: tp.Callable[[tp.Sequence[str]], transformers.BatchEncoding],
) -> _Batch:
    source_sentences, _, target_features = zip(*batch)
    return _Batch(
        tokenize(source_sentences),
        torch.tensor(np.stack(target_features).astype("float32")),
    )


def compute_huggingface_dataset_features(
    dataset: datasets.DatasetDict,
    batch_size: int = 512,
    clip_version: str = TARGET_FEATURES_MODEL,
) -> datasets.DatasetDict:

    device = "cuda"
    tokenizer = transformers.CLIPTokenizer.from_pretrained(clip_version)
    model = transformers.CLIPTextModel.from_pretrained(clip_version)
    model = model.eval()
    model.requires_grad_(False)
    model.to(device)

    def compute_features(
        target_sentence,
    ) -> npt.NDArray[np.float32]:
        tokenized = tokenizer(
            target_sentence,
            truncation=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt",
        )
        return (
            model(**tokenized.to(device)).pooler_output.cpu().numpy().astype("float32")
        )

    def feature_computer_helper(item, key) -> dict:
        return {"target_features": compute_features(item[key])}

    feature_computer = functools.partial(
        feature_computer_helper,
        key="target",
    )

    return dataset.map(feature_computer, batched=True, batch_size=batch_size)


class ParallelDataset(torch.utils.data.Dataset[_ParallelItem]):
    dataset_name: str = None

    def __init__(
        self,
        datadir: str,
        split: tpx.Literal["train", "val", "test"],
        clip_version: str = TARGET_FEATURES_MODEL,
        prepare_data: bool = False,
    ) -> None:
        self.datadir = datadir
        self.clip_version = clip_version
        self.split = split

        if prepare_data:
            self.prepare_data(datadir)

        self.parallel_items = datasets.load_from_disk(
            dataset_path=pathlib.Path(datadir) / self.dataset_name, keep_in_memory=True
        )
        self.parallel_items = self.parallel_items[split]

    @classmethod
    def prepare_data(cls, datadir: str, split: str = None) -> None:
        basepath = pathlib.Path(datadir) / cls.dataset_name
        basepath.mkdir(exist_ok=True, parents=True)

        splits = (split,) if split else ("train", "val", "test")

        print(f"Downloading files to {basepath}, this might take a while...")
        if all((basepath / split_type).exists() for split_type in splits):
            print(f"All files already in {basepath}, skipping...")
            return

        dset = cls.custom_data_preparation()
        dset.save_to_disk(basepath)

    def custom_data_preparation(self) -> None:
        pass

    def __getitem__(self, key: int) -> _ParallelItem:
        source = self.parallel_items[key]["source"]
        target = self.parallel_items[key]["target"]
        features = self.parallel_items[key]["target_features"]
        return _ParallelItem(str(source), str(target), features)

    def __len__(self) -> int:
        return len(self.parallel_items)


class TatoebaParallel(ParallelDataset):
    dataset_name: str = "tatoeba"

    @classmethod
    def custom_data_preparation(cls) -> datasets.DatasetDict:
        dset = datasets.load_dataset(cls.dataset_name, lang1="en", lang2="es")
        dset = dset.flatten()
        dset = dset.rename_column("translation.es", "source")
        dset = dset.rename_column("translation.en", "target")
        dset = compute_huggingface_dataset_features(
            dset,
            clip_version=cls.clip_version,
        )

        # wacky workaround to create a dataset without test and val
        aux_empty_dset_dict = {column: [] for column in dset["train"].column_names}
        aux_empty_dset = datasets.Dataset.from_dict(aux_empty_dset_dict)

        dset = datasets.DatasetDict(
            {
                "train": dset["train"],
                "test": aux_empty_dset,
                "val": aux_empty_dset,
            }
        )

        return dset


class OpusParallel(ParallelDataset):
    dataset_name: str = "opus100"

    @classmethod
    def custom_data_preparation(cls) -> datasets.DatasetDict:
        dset = datasets.load_dataset(cls.dataset_name, "en-es")
        dset = dset.flatten()
        dset = dset.rename_column("translation.es", "source")
        dset = dset.rename_column("translation.en", "target")
        dset = compute_huggingface_dataset_features(
            dset,
            clip_version=cls.clip_version,
        )

        dset = datasets.DatasetDict(
            {
                "train": dset["train"],
                "test": dset["test"],
                "val": dset["validation"],
            }
        )

        return dset


class TedParallel(ParallelDataset):
    dataset_name: str = "ted_talks_iwslt"

    @classmethod
    def custom_data_preparation(cls) -> datasets.DatasetDict:
        years = ["2014", "2015", "2016"]
        dsets = []
        for year in years:
            dset = datasets.load_dataset(
                cls.dataset_name,
                language_pair=("en", "es"),
                year=year,
            )
            dset = dset.flatten()
            dset = dset.rename_column("translation.es", "source")
            dset = dset.rename_column("translation.en", "target")
            dset = compute_huggingface_dataset_features(
                dset,
                clip_version=cls.clip_version,
            )
            dsets.append(dset["train"])

        dset = datasets.concatenate_datasets(dsets)

        # wacky workaround to create a dataset without test and val
        aux_empty_dset_dict = {column: [] for column in dset["train"].column_names}
        aux_empty_dset = datasets.Dataset.from_dict(aux_empty_dset_dict)

        dset = datasets.DatasetDict(
            {
                "train": dset["train"],
                "test": aux_empty_dset,
                "val": aux_empty_dset,
            }
        )

        return dset


class LitParallel(pl.LightningDataModule):
    _datasets: tp.ClassVar[tp.List[type]] = [
        TedParallel,
        OpusParallel,
        TatoebaParallel,
    ]

    # these attrs are set in __init__, and work mostly as hparams
    datadir: str
    batch_size: int
    max_sequence_length: int
    tokenizer_name: str
    # these attrs are set in setup method
    # splits are optional because some might not be present depending on stage
    train_split: tp.Optional[ParallelDataset]
    val_split: tp.Optional[ParallelDataset]
    test_split: tp.Optional[ParallelDataset]

    tokenizer: transformers.BertTokenizer
    _collate: tp.Callable[[tp.Sequence[_ParallelItem]], _Batch]

    def __init__(
        self,
        datadir: str,
        batch_size: int,
        max_sequence_length: int,
        tokenizer_name: str = BERT_MODEL_NAME,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))

        self.datadir = datadir
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer_name = tokenizer_name

    def prepare_data(self) -> None:
        for _dataset_cls in self._datasets:
            _dataset_cls.prepare_data(self.datadir)

    def setup(
        self, stage: tp.Optional[tpx.Literal["fit", "validate", "test"]] = None
    ) -> None:
        if stage in ("fit", "validate", None):
            self.val_split = torch.utils.data.ConcatDataset(
                [_dataset_cls(self.datadir, "val") for _dataset_cls in self._datasets]
            )
        if stage in ("fit", None):
            self.train_split = torch.utils.data.ConcatDataset(
                [_dataset_cls(self.datadir, "train") for _dataset_cls in self._datasets]
            )
        if stage in ("test", None):
            self.test_split = torch.utils.data.ConcatDataset(
                [_dataset_cls(self.datadir, "test") for _dataset_cls in self._datasets]
            )

        # always load tokenizer
        self.tokenizer = transformers.BertTokenizer.from_pretrained(self.tokenizer_name)
        self._collate = functools.partial(
            _collate_parallel_items,
            tokenize=functools.partial(
                self.tokenizer,
                padding="max_length",
                truncation=True,
                max_length=self.max_sequence_length,
                return_tensors="pt",
            ),
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        # if we train with absurd amounts of data, possibly don't shuffle
        return torch.utils.data.DataLoader(
            dataset=self.train_split,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            dataset=self.val_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            dataset=self.test_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self._collate,
        )


# Evaluation


class _CLIPItem(tp.NamedTuple):
    text: str
    image: PIL.Image.Image


class XTD10(torch.utils.data.Dataset[_CLIPItem]):
    name: tp.ClassVar[str] = "xtd10"

    _base_url = (
        "https://raw.githubusercontent.com/adobe-research/"  # org
        "Cross-lingual-Test-Dataset-XTD10/"  # repo
        "d77ff16148468c84287f3efcbf602a015c5e4954/"  # specific commit
        "XTD10/"  # dir inside repo
    )
    _fname_template = "test_1kcaptions_{}.txt"
    _image_names_url = _base_url + "test_image_names.txt"

    datadir: str
    fname: str

    captions: tp.List[str]
    image_names: tp.List[str]
    images: tp.List[PIL.Image.Image]

    def __init__(self, datadir: str, lang: str, split: tpx.Literal["test"] = "test"):

        self.datadir = datadir
        self.fname = self._fname_template.format(lang)
        self.lang = lang
        assert split == "test", "XTD10 is a test dataset"

        memory = joblib.Memory(datadir, verbose=0)

        @memory.cache
        def download_simple_lines(url: str) -> tp.List[str]:
            """Downloads a simple .txt file and returns the lines as a list"""
            return requests.get(url).text.splitlines()

        self.captions = download_simple_lines(self._base_url + self.fname)
        self.image_names = download_simple_lines(self._image_names_url)

        assert all(
            self._image_path_from_name(image_name).exists()
            for image_name in self.image_names
        ), (
            "Some needed COCO images are missing. "
            f"Please download COCO and put it in {datadir}/coco"
        )

        @memory.cache
        def images_from_names(image_names):
            paths = [self._image_path_from_name(name) for name in image_names]
            return [PIL.Image.open(path) for path in paths]

        # just load everything in memory, this dataset is super small
        self.images = images_from_names(self.image_names)

    def __getitem__(self, key: int) -> _CLIPItem:
        return _CLIPItem(self.captions[key], self.images[key])

    def __len__(self) -> int:
        return len(self.captions)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.datadir!r}, {self.lang!r})"

    def _image_path_from_name(self, image_name: str) -> pathlib.Path:
        """Simple helper function to determine a COCO image path given its
        name. Basically prepends the correct split dir to the path name"""
        match = re.match(r"COCO_(.*)_\d*.jpg", image_name)
        if not match:
            raise ValueError(f"Bad image_name format {image_name}")

        splitname = match.group(1)
        return pathlib.Path(self.datadir) / "mscoco" / splitname / image_name


class LitXTD10(pl.LightningDataModule):
    """LightningDataModule for the test dataset XTD10.

    >>> datamodule = LitXTD10(
    ...     datadir="data",
    ...     lang="es",
    ...     batch_size=32,
    ...     tokenizer_version="dccuchile/bert-base-spanish-wwm-uncased",
    ...     feature_extractor_version="openai/clip-vit-large-patch14"
    ... )
    >>> datamodule.setup("test")
    >>> len(datamodule.test_dataset)
    1000
    """

    def __init__(
        self,
        datadir: str,
        lang: str,
        batch_size: int,
        tokenizer_version: str,
        feature_extractor_version: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))
        self.datadir = datadir
        self.lang = lang
        self.batch_size = batch_size
        self.tokenizer_version = tokenizer_version
        self.feature_extractor_version = feature_extractor_version

    def setup(self, stage=None):
        if stage == "test":
            self.processor = transformers.VisionTextDualEncoderProcessor(
                transformers.AutoFeatureExtractor.from_pretrained(
                    self.feature_extractor_version
                ),
                transformers.AutoTokenizer.from_pretrained(self.tokenizer_version),
            )
            self.test_dataset = XTD10(self.datadir, self.lang, split="test")
        else:
            raise ValueError(
                f"this datamodule can be used only for testing, got stage={stage}"
            )

    def collate_fn(self, items: tp.Sequence[_CLIPItem]) -> transformers.BatchEncoding:
        """Returns a transformers.BatchEncoding with attributes `input_ids`,
        `token_type_ids`, `attention_mask` (from the tokenizer) and
        `pixel_values` (from the feature extractor)
        """
        # processor inputs have to be lists
        texts, images = map(list, zip(*items))

        # TODO: add extra kwargs to processor, like max_length and such
        return self.processor(
            text=texts, images=images, return_tensors="pt", padding=True
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn,
        )


class _ImageNetItem(tp.NamedTuple):
    image: PIL.Image.Image
    prompt: tp.Optional[str]
    label: tp.Optional[int]


class ImageNet(torch.utils.data.Dataset[_ImageNetItem]):
    name: tp.ClassVar[str] = "imagenet"
    prompt_prefix: str = "a picture of a "
    _lens = {"val": 50_000, "test": 100_000}

    datadir: str
    split: tpx.Literal["val", "test"]
    prompt_lang: str

    class _SynsetData(tpx.TypedDict):
        prompt: str
        class_idx: int

    synset_mapping: tp.Dict[str, _SynsetData]
    english_classes: tp.List[str]

    def __init__(
        self, datadir: str, split: tpx.Literal["val", "test"], prompt_lang: str
    ) -> None:
        assert split in (
            "val",
            "test",
        ), "For this work we don't use the train split, so it's not available"

        self.datadir = datadir
        self.split = split
        self.prompt_lang = prompt_lang

        wnids = []
        self.english_classes = []
        with open(pathlib.Path(datadir) / self.name / "LOC_synset_mapping.txt") as file:
            # NOTE: i am keeping only the first comma-separated field of the
            # synset description
            for i, line in enumerate(file):
                wnid, _, descriptions = line.partition(" ")
                first_description, *_ = (d.strip() for d in descriptions.split(","))
                wnids.append(wnid)
                self.english_classes.append(first_description)

        english_prompts = [self.prompt_prefix + name for name in self.english_classes]
        # preeetty nasty formating, not gonna lie
        translated_prompts = [
            str(prompt)
            for prompt in joblib.Memory(datadir, verbose=0).cache(
                ignore=["batch_size"]
            )(translate)(
                english_prompts,
                f"Helsinki-NLP/opus-mt-en-{prompt_lang}",
                batch_size=100,
            )
        ]
        self.synset_mapping = {}
        for i, (wnid, prompt) in enumerate(zip(wnids, translated_prompts)):
            self.synset_mapping[wnid] = {"prompt": prompt, "class_idx": i}

    @property
    def all_prompts(self) -> tp.List[str]:
        # since py37 dict insertion order is guaranteed
        # im sceptic
        syn_vals = self.synset_mapping.values()
        sorted_vals = sorted(syn_vals, key=operator.itemgetter("class_idx"))
        return [v["prompt"] for v in sorted_vals]

    def __getitem__(self, key: int) -> _ImageNetItem:
        if not (0 <= key < len(self)):
            raise IndexError
        # imagenet files are 1-indexed
        key = key + 1
        base_path = pathlib.Path(self.datadir) / self.name / "ILSVRC"
        prompt: tp.Optional[str] = None
        label: tp.Optional[int] = None
        if self.split == "val":
            # only val split has annotations
            ann_file = (
                base_path
                / "Annotations"
                / "CLS-LOC"
                / self.split
                / f"ILSVRC2012_val_{key:08}.xml"
            )
            ann_wnid = self.parse_annotation(ann_file)["object"]["name"]
            prompt = self.synset_mapping[ann_wnid]["prompt"]
            label = self.synset_mapping[ann_wnid]["class_idx"]

        image_path = (
            base_path
            / "Data"
            / "CLS-LOC"
            / self.split
            / f"ILSVRC2012_{self.split}_{key:08}.JPEG"
        )

        image = PIL.Image.open(image_path)
        return _ImageNetItem(image, prompt, label)

    def __len__(self) -> int:
        return self._lens[self.split]

    @staticmethod
    def parse_annotation(xml_path: tp.Union[str, pathlib.Path]) -> dict:
        def parse_node(node):
            if len(node) == 0:
                return node.text
            else:
                return {child.tag: parse_node(child) for child in node}

        return parse_node(ET.parse(xml_path).getroot())


class LitImageNet(pl.LightningDataModule):
    """LightningDataModule for the val and test splits of ImageNet

    >>> datamodule = LitImageNet(
    ...     datadir="data",
    ...     prompt_lang="es",
    ...     batch_size=32,
    ...     tokenizer_version="dccuchile/bert-base-spanish-wwm-uncased",
    ...     feature_extractor_version="openai/clip-vit-large-patch14",
    ... )
    >>> datamodule.setup("test")
    >>> len(datamodule.test_dataset)
    50000
    """

    def __init__(
        self,
        datadir: str,
        lang: str,
        batch_size: int,
        tokenizer_version: str,
        feature_extractor_version: str,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))

        self.datadir = datadir
        self.lang = lang
        self.batch_size = batch_size
        self.tokenizer_version = tokenizer_version
        self.feature_extractor_version = feature_extractor_version

    def setup(self, stage: str = None):
        if stage != "test":
            # possibly we can support the `predict` stage to produce prediction
            # for the imagenet competition or something
            raise ValueError("For now this datamodule can only be used for testing")

        # see https://github.com/huggingface/transformers/issues/5486
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.tokenizer_version
        )
        self.feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
            self.feature_extractor_version
        )

        # use val dataset of imagenet as test set since the actual test set
        # doesn't have labels
        self.test_dataset = ImageNet(self.datadir, split="val", prompt_lang=self.lang)

    def collate_fn(
        self, items: tp.Sequence[_ImageNetItem]
    ) -> tp.Tuple[transformers.BatchEncoding, torch.Tensor]:
        # ignore prompts, they are processed in the prompts dataloader
        images, _, targets = map(list, zip(*items))
        return (
            self.feature_extractor(images, return_tensors="pt"),
            torch.tensor(targets),
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn,
        )

    def prompts_dataloader(
        self,
    ) -> torch.utils.data.DataLoader[transformers.BatchEncoding]:
        """This is not part of pytorch_lightning's api, just my own dataloader"""
        dataset = getattr(self, "test_dataset", None) or getattr(
            self, "predict_dataset"
        )
        return torch.utils.data.DataLoader(
            dataset.all_prompts,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=functools.partial(
                self.tokenizer, return_tensors="pt", padding=True
            ),
        )
