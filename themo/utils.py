import numpy.typing as npt
import numpy as np
import typing as tp
import inspect
import pyarrow as pa
import pyarrow.parquet
import time
import hashlib
import pickle
import pathlib
import functools
import dataclasses
import collections


@dataclasses.dataclass
class _CachingTool:
    save: tp.Callable[[tp.Any, str], None]
    load: tp.Callable[[str], tp.Any]
    extension: str


def _multidict(d):
    return {
        ik: v
        for k, v in d.items()
        for ik in (
            k
            if not isinstance(k, str) and isinstance(k, collections.abc.Iterable)
            else [k]
        )
    }


_caching_tools = _multidict(
    {
        pa.Table: _CachingTool(
            pa.parquet.write_table, pa.parquet.read_table, ".parquet"
        ),
        (np.ndarray, npt.NDArray[np.float32]): _CachingTool(
            lambda obj, file: np.save(file, obj), np.load, ".npy"
        ),
    }
)


def diskcache(datadir: str):
    """Caches function based on declared return type."""
    cacheroot = pathlib.Path(datadir) / ".cache"
    cacheroot.mkdir(exist_ok=True, parents=True)

    def decorator(fun):
        fun_return_type = tp.get_type_hints(fun)["return"]
        try:
            caching_tool = _caching_tools[fun_return_type]
        except KeyError:
            raise TypeError(
                "I don't know how to cache the following"
                f" return type: {fun_return_type}"
            ) from None

        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            tic = time.perf_counter()
            key = (
                hashlib.md5(
                    pickle.dumps((args, kwargs, inspect.getsource(fun)))
                ).hexdigest()
                + caching_tool.extension
            )
            location = cacheroot / key
            if location.exists():
                print("cache hit, loading...")
                out = caching_tool.load(str(location))
            else:
                print("cache miss, computing...")
                out = fun(*args, **kwargs)
                caching_tool.save(out, str(location))

            tac = time.perf_counter()
            print(f"took {tac - tic:0.1f}s")
            return out

        return wrapper

    return decorator
