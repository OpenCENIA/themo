import functools
import transformers
import os
import joblib
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.csv
import pytorch_lightning as pl
import collections
import requests
import inspect
import pathlib
import transformers
import torch
import typing as tp
import typing_extensions as tpx
import types
import tqdm
import warnings
import os


__all__ = ["WITParallel", "LitWitParallel", "TARGET_FEATURES_MODEL"]

TARGET_FEATURES_MODEL = "openai/clip-vit-large-patch14"

_TQDM_WIDTH = 120
_NS = types.SimpleNamespace


# TODO: name field is sometimes just a name (a str) or sometimes a full pathlib.Path,
# maybe we should fix this
class _FileMeta(tp.NamedTuple):
    name: tp.Union[str, pathlib.Path]
    size: int
    nrows: int


class _WITItem(tp.NamedTuple):
    source: str
    target: str
    target_features: npt.NDArray[np.float32]


class _Batch(tp.NamedTuple):
    input: transformers.BatchEncoding
    target: torch.Tensor


def _collate_wit_items(
    batch: tp.Sequence[_WITItem],
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

    tokenizer = transformers.CLIPTokenizer.from_pretrained(clip_version)

    def preprocess(sentences):
        sentences = [sentence.as_py() for sentence in sentences]
        return tokenizer(
            sentences,
            truncation=True,
            max_length=77,
            padding="max_length",
            return_tensors="pt",
        )

    dloader = torch.utils.data.DataLoader(
        target_sentences,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=preprocess,
    )

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


class WITParallel(torch.utils.data.Dataset[_WITItem]):
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
        # I am not too sold on using filedata._replace here, but it gets the job done
        self.parallel_items = joblib.Memory(datadir).cache(build_parallel)(
            [
                filedata._replace(
                    name=pathlib.Path(datadir) / self.META.dataset_name / filedata.name
                )
                for filedata in getattr(self.META.files, split)
            ],
            langs,
        )

        # Compute CLIP embeddings
        self.target_features = joblib.Memory(datadir).cache(compute_target_features)(
            self.parallel_items["target"],
            clip_version=self.clip_version,
            batch_size=512,
            num_workers=os.cpu_count() or 0,
        )

    def __getitem__(self, key: int) -> _WITItem:
        source = self.parallel_items["source"][key]
        target = self.parallel_items["target"][key]
        features = self.target_features[key]
        return _WITItem(str(source), str(target), features)

    def __len__(self) -> int:
        return len(self.parallel_items)

    @classmethod
    def download(cls, datadir: str, split=None):
        basepath = pathlib.Path(datadir) / cls.META.dataset_name
        basepath.mkdir(exist_ok=True, parents=True)

        splits = (split,) if split else ("train", "val", "test")

        print(f"Downloading files to {basepath}, this might take a while...")
        for split in splits:
            for filedata in getattr(cls.META.files, split):
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
                    f"somthing doesn't match for file {filedata.name}: "
                    f"total_written={total_written},"
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


class LitWITParallel(pl.LightningDataModule):
    # these attrs are set in __init__, and work mostly as hparams
    datadir: str
    batch_size: int
    max_sequence_length: int
    # these attrs are set in setup method
    # splits are optional because some might not be present depending on stage
    train_split: tp.Optional[WITParallel]
    val_split: tp.Optional[WITParallel]
    test_split: tp.Optional[WITParallel]

    tokenizer: transformers.BertTokenizer
    _collate: tp.Callable[[tp.Sequence[_WITItem]], _Batch]

    def __init__(
        self,
        datadir: str,
        batch_size: int,
        max_sequence_length: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))

        self.datadir = datadir
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length

    def prepare_data(self) -> None:
        WITParallel.download(self.datadir)

    def setup(
        self, stage: tp.Optional[tpx.Literal["fit", "validate", "test"]] = None
    ) -> None:
        if stage in ("fit", "validate", None):
            self.val_split = WITParallel(self.datadir, "val")
        if stage in ("fit", None):
            self.train_split = WITParallel(self.datadir, "train")
        if stage in ("test", None):
            self.test_split = WITParallel(self.datadir, "test")

        # always load tokenizer
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            "dccuchile/bert-base-spanish-wwm-uncased"
        )
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
        # if we train with absurd amounts of data, posibbly don't shuffle
        return torch.utils.data.DataLoader(
            dataset=self.train_split,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=os.cpu_count() or 0,
            collate_fn=self._collate,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            dataset=self.val_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() or 0,
            collate_fn=self._collate,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader[_Batch]:
        return torch.utils.data.DataLoader(
            dataset=self.test_split,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=os.cpu_count() or 0,
            collate_fn=self._collate,
        )
