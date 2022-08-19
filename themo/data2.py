import collections
import functools
import inspect
import pathlib
import types
import typing as tp
import warnings

import datasets
import joblib
import numpy as np
import numpy.typing as npt
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
    "TatoebaParallel",
    "LitFloresParallel",
    "LitOpusParallel",
    "LitTedParallel",
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


class _Batch(tp.NamedTuple):
    input: transformers.BatchEncoding
    target: torch.Tensor


class _ParallelItem(tp.NamedTuple):
    source: str
    target: str
    target_features: npt.NDArray[np.float32]


def _collate_parallel_items(
    batch: tp.Sequence[_ParallelItem],
    tokenize: tp.Callable[[tp.Sequence[str]], transformers.BatchEncoding],
) -> _Batch:
    source_sentences, _, target_features = zip(*batch)
    return _Batch(
        tokenize(source_sentences),
        torch.tensor(np.stack(target_features).astype("float32")),
    )


class ParallelDasaset(torch.utils.data.Dataset[_ParallelItem]):
    dataset_name: str = None

    def __init__(
        self,
        datadir: str,
        split: tpx.Literal["train", "val", "test"],
        clip_version: str = TARGET_FEATURES_MODEL,
        download: bool = False,
    ) -> None:
        self.datadir = datadir
        self.clip_version = clip_version
        self.split = split

        if download:
            self.download(datadir)

        self.parallel_items = datasets.load_from_disk(
            dataset_path=pathlib.Path(datadir) / self.dataset_name, keep_in_memory=True
        )
        self.parallel_items = self.parallel_items[split]

    @classmethod
    def download(cls, datadir: str, split=None):
        basepath = pathlib.Path(datadir) / cls.dataset_name
        basepath.mkdir(exist_ok=True, parents=True)

        splits = (split,) if split else ("train", "val", "test")

        print(f"Downloading files to {basepath}, this might take a while...")
        if all((basepath / split_type).exists() for split_type in splits):
            print(f"All files already in {basepath}, skipping...")
            return

        device = "cuda"
        tokenizer = transformers.CLIPTokenizer.from_pretrained(TARGET_FEATURES_MODEL)
        model = transformers.CLIPTextModel.from_pretrained(TARGET_FEATURES_MODEL)
        model = model.eval()
        model.requires_grad_(False)
        model.to(device)

        def compute_features(
            target_sentence,
        ):
            tokenized = tokenizer(
                target_sentence,
                truncation=True,
                max_length=77,
                padding="max_length",
                return_tensors="pt",
            )
            return model(**tokenized.to(device)).pooler_output.cpu().numpy()

        def feature_computer_helper(item, key):
            return {"target_features": compute_features(item[key])}

        cls.feature_computer = functools.partial(
            feature_computer_helper,
            key="target",
        )

        dset = cls.get_dataset()
        dset.save_to_disk(basepath)

    def get_dataset(self):
        pass

    def __getitem__(self, key: int) -> _ParallelItem:
        source = self.parallel_items[key]["source"]
        target = self.parallel_items[key]["target"]
        features = self.parallel_items[key]["target_features"]
        return _ParallelItem(str(source), str(target), features)

    def __len__(self) -> int:
        return len(self.parallel_items)


class FloresParallel(ParallelDasaset):
    dataset_name: str = "facebook/flores"

    def __init__(
        self,
        datadir: str,
        split: tpx.Literal["train", "val", "test"],
        clip_version: str = TARGET_FEATURES_MODEL,
        download: bool = False,
    ):
        super().__init__(datadir, split, clip_version, download)

    @classmethod
    def get_dataset(cls):
        dset_es = datasets.load_dataset(cls.dataset_name, "spa_Latn")
        dset_en = datasets.load_dataset(cls.dataset_name, "eng_Latn")

        def process_dset(dset, columns_to_rename, new_column_names):
            # cursed
            new_dset = dset.remove_columns(
                list(set(columns_to_rename) ^ set(dset["dev"].column_names))
            )
            new_dset = new_dset.rename_columns(
                dict(zip(columns_to_rename, new_column_names))
            )
            return new_dset

        es_dset = process_dset(
            dset=dset_es,
            columns_to_rename=["sentence"],
            new_column_names=["source"],
        )
        en_dset = process_dset(
            dset=dset_en,
            columns_to_rename=["sentence"],
            new_column_names=["target"],
        )
        en_dset = en_dset.map(
            cls.feature_computer,
            batched=True,
            batch_size=256,
        )

        dset = {}
        # dev: 997 items | devtest: 1012 items
        dset["train"] = datasets.concatenate_datasets(
            [es_dset["devtest"], en_dset["devtest"]], axis=1
        )
        test_val = datasets.concatenate_datasets(
            [es_dset["dev"], en_dset["dev"]], axis=1
        )
        test_val = test_val.train_test_split(test_size=0.5)
        dset["test"] = test_val["train"]
        dset["val"] = test_val["test"]

        dset = datasets.DatasetDict(dset)
        return dset


class TatoebaParallel(ParallelDasaset):
    dataset_name: str = "tatoeba"

    def __init__(
        self,
        datadir: str,
        split: tpx.Literal["train", "val", "test"],
        clip_version: str = TARGET_FEATURES_MODEL,
        download: bool = False,
    ) -> None:
        super().__init__(datadir, split, clip_version, download)

    @classmethod
    def get_dataset(cls):
        dset = datasets.load_dataset(cls.dataset_name, lang1="en", lang2="es")
        dset = dset.flatten()
        dset = dset.rename_column("translation.es", "source")
        dset = dset.rename_column("translation.en", "target")
        dset = dset.map(
            cls.feature_computer,
            batched=True,
            batch_size=512,
        )

        # 90% train, 10% test + validation
        dset_train_testvalid = dset["train"].train_test_split(
            test_size=0.1
        )
        # Split the 10% test + valid in half test, half valid
        dset_test_valid = dset_train_testvalid[
            "test"
        ].train_test_split(test_size=0.5)
        # gather everyone if you want to have a single DatasetDict
        dset = datasets.DatasetDict(
            {
                "train": dset_train_testvalid["train"],
                "test": dset_test_valid["test"],
                "val": dset_test_valid["train"],
            }
        )

        return dset


class OpusParallel(ParallelDasaset):
    dataset_name: str = "opus100"

    def __init__(
        self,
        datadir: str,
        split: tpx.Literal["train", "val", "test"],
        clip_version: str = TARGET_FEATURES_MODEL,
        download: bool = False,
    ) -> None:
        super().__init__(datadir, split, clip_version, download)

    @classmethod
    def get_dataset(cls):
        dset = datasets.load_dataset(cls.dataset_name, "en-es")
        dset = dset.flatten()
        dset = dset.rename_column("translation.es", "source")
        dset = dset.rename_column("translation.en", "target")
        dset = dset.map(
            cls.feature_computer,
            batched=True,
            batch_size=512,
        )

        dset = datasets.DatasetDict(
            {
                "train": dset["train"],
                "test": dset["test"],
                "val": dset["validation"],
            }
        )

        return dset


class TedParallel(ParallelDasaset):
    dataset_name: str = "ted_talks_iwslt"

    def __init__(
        self,
        datadir: str,
        split: tpx.Literal["train", "val", "test"],
        clip_version: str = TARGET_FEATURES_MODEL,
        download: bool = False,
    ) -> None:
        super().__init__(datadir, split, clip_version, download)

    @classmethod
    def get_dataset(cls):
        years = ['2014', '2015', '2016']
        dsets = []
        for year in years:
            dset = datasets.load_dataset(
                cls.dataset_name,
                language_pair=("en", "es"),
                year=year
            )
            dset = dset.flatten()
            dset = dset.rename_column("translation.es", "source")
            dset = dset.rename_column("translation.en", "target")
            dset = dset.map(
                cls.feature_computer,
                batched=True,
                batch_size=512,
            )
            dsets.append(dset["train"])

        dset = datasets.concatenate_datasets(dsets)

        train_testvalid = dset.train_test_split(test_size=0.1)
        test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
        dset = datasets.DatasetDict(
            {
                "train": train_testvalid["train"],
                "test": test_valid["test"],
                "val": test_valid["train"],
            }
        )

        return dset

    @classmethod
    def download(cls, datadir: str, split=None):
        basepath = pathlib.Path(datadir) / cls.dataset_name
        basepath.mkdir(exist_ok=True, parents=True)

        splits = (split,) if split else ("train", "val", "test")

        print(f"Downloading files to {basepath}, this might take a while...")
        if all((basepath / split_type).exists() for split_type in splits):
            print(f"All files already in {basepath}, skipping...")
            return

        device = "cuda"
        tokenizer = transformers.CLIPTokenizer.from_pretrained(TARGET_FEATURES_MODEL)
        model = transformers.CLIPTextModel.from_pretrained(TARGET_FEATURES_MODEL)
        model = model.eval()
        model.requires_grad_(False)
        model.to(device)

        def compute_features(
            target_sentence,
        ):
            tokenized = tokenizer(
                target_sentence,
                truncation=True,
                max_length=77,
                padding="max_length",
                return_tensors="pt",
            )
            return model(**tokenized.to(device)).pooler_output.cpu().data.numpy()

        def feature_computer_helper(item, key):
            return {"target_features": compute_features(item[key])}

        feature_computer = functools.partial(
            feature_computer_helper,
            key="target",
        )

        years = ['2014', '2015', '2016']
        dsets = []
        for year in years:
            dset = datasets.load_dataset("ted_talks_iwslt", language_pair=("en", "es"), year=year)
            dset = dset.flatten()
            dset = dset.rename_column("translation.es", "source")
            dset = dset.rename_column("translation.en", "target")
            dset = dset.map(
                feature_computer,
                batched=True,
                batch_size=512,
            )
            dsets.append(dset["train"])

        dset = datasets.concatenate_datasets(dsets)

        train_testvalid = dset.train_test_split(test_size=0.1)
        test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
        dset = datasets.DatasetDict(
            {
                "train": train_testvalid["train"],
                "test": test_valid["test"],
                "val": test_valid["train"],
            }
        )

        dset.save_to_disk(basepath)

    def __getitem__(self, key: int) -> _ParallelItem:
        source = self.parallel_items[key]["source"]
        target = self.parallel_items[key]["target"]
        features = self.parallel_items[key]["target_features"]
        return _ParallelItem(str(source), str(target), features)

    def __len__(self) -> int:
        return len(self.parallel_items)


class LitParallel(pl.LightningDataModule):
    pass

class LitTatoebaParallel(pl.LightningDataModule):
    _dataset_cls: tp.ClassVar[type] = TatoebaParallel

    # these attrs are set in __init__, and work mostly as hparams
    datadir: str
    batch_size: int
    max_sequence_length: int
    tokenizer_name: str
    # these attrs are set in setup method
    # splits are optional because some might not be present depending on stage
    train_split: tp.Optional[TatoebaParallel]
    val_split: tp.Optional[TatoebaParallel]
    test_split: tp.Optional[TatoebaParallel]

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


class LitFloresParallel(pl.LightningDataModule):
    _dataset_cls: tp.ClassVar[type] = FloresParallel

    # these attrs are set in __init__, and work mostly as hparams
    datadir: str
    batch_size: int
    max_sequence_length: int
    tokenizer_name: str
    # these attrs are set in setup method
    # splits are optional because some might not be present depending on stage
    train_split: tp.Optional[FloresParallel]
    val_split: tp.Optional[FloresParallel]
    test_split: tp.Optional[FloresParallel]

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


class LitOpusParallel(pl.LightningDataModule):
    _dataset_cls: tp.ClassVar[type] = OpusParallel

    # these attrs are set in __init__, and work mostly as hparams
    datadir: str
    batch_size: int
    max_sequence_length: int
    tokenizer_name: str
    # these attrs are set in setup method
    # splits are optional because some might not be present depending on stage
    train_split: tp.Optional[OpusParallel]
    val_split: tp.Optional[OpusParallel]
    test_split: tp.Optional[OpusParallel]

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


class LitTedParallel(pl.LightningDataModule):
    _dataset_cls: tp.ClassVar[type] = TedParallel

    # these attrs are set in __init__, and work mostly as hparams
    datadir: str
    batch_size: int
    max_sequence_length: int
    tokenizer_name: str
    # these attrs are set in setup method
    # splits are optional because some might not be present depending on stage
    train_split: tp.Optional[TedParallel]
    val_split: tp.Optional[TedParallel]
    test_split: tp.Optional[TedParallel]

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
