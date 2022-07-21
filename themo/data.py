import pyarrow as pa
import pyarrow.csv
import pytorch_lightning as pl
import collections
import requests
import inspect
import pathlib
import typing as tp
import typing_extensions as tpx
import types
import torch
import tqdm
from .text_embedders import TextEmbedder

import themo.utils as utils

__all__ = ["WITParallel", "WITParallelDataModule"]

_TQDM_WIDTH = 120
_NS = types.SimpleNamespace


# TODO: name field is sometimes just a name (a str) or sometimes a full pathlib.Path
# maybe we should fix this
_FileMeta = collections.namedtuple("_FileMeta", ["name", "size", "nrows"])


def build_parallel(files: tp.List[_FileMeta], langs: tp.Tuple[str, str], lang_to_embed: str = 'en') -> pa.Table:
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

    # instatiate text embedder
    textEncoder = TextEmbedder().cuda()

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
                "embedding": textEncoder(captions[lang_to_embed])[0].cpu().numpy()
            }
            for key, captions in image_to_captions.items()
            if set(langs).issubset(captions.keys())
        ]
    )


class WITParallel(torch.utils.data.Dataset):
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

    def __init__(
        self,
        datadir: str,
        split: tpx.Literal["train", "val", "test"],
        langs: tp.Tuple[str, str] = ("es", "en"),
        download: bool = False,
    ) -> None:
        self.datadir = datadir
        self.split = split
        self.langs = langs

        if download:
            self.download(datadir, split)

        # 1: probar que esta wea (parquet_cache), no se cae si buildparallel retorna vectores de numeros
        # 2: correr build_parallel solo, y dsp hacer el pipeline de parquet_cache sobre el resultadod e eso
        # y dsp (o en paralelo), calcular los embedding sobre lo que salgad e build_parallel 
        # reminder: 512 es el optimo del batch_size

        # I am not too sold on using filedata._replace here, but it gets the job done
        print(f"Loading {self!r}")
        self.parallel_items = utils.parquet_cache(datadir)(build_parallel)(
            [
                filedata._replace(
                    name=pathlib.Path(datadir) / self.META.dataset_name / filedata.name
                )
                for filedata in getattr(self.META.files, split)
            ],
            langs,
        )

    def __getitem__(self, key: int) -> tp.Tuple[str, str]:
        # I dont know if there is a better way to index a pyarrow.Table
        row = self.parallel_items.slice(key, 1).to_pylist()[0]
        return row["source"], row["target"], row["embedding"]

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
                    f"total_written={total_written}, total_downloaded={total_downloaded}, "
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


class WITParallelDataModule(pl.LightningDataModule):
    """WIP (work in Progress, not wit, wikipedia-based image Text)"""

    def __init__(self, datadir: str, batch_size: int) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=("datadir",))

        self.datadir = datadir
        self.batch_size = batch_size

    def prepare_data(self):
        WITParallel.download(self.datadir)
