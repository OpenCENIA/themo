import inspect
import pyarrow as pa
import pyarrow.parquet
import time
import hashlib
import pickle
import pathlib
import functools


def parquet_cache(datadir: str):
    """Caches functions that return Apache Parquet tables"""
    cacheroot = pathlib.Path(datadir) / ".cache"
    cacheroot.mkdir(exist_ok=True, parents=True)

    def decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            tic = time.perf_counter()
            key = (
                hashlib.md5(
                    pickle.dumps((args, kwargs, inspect.getsource(fun)))
                ).hexdigest()
                + ".parquet"
            )
            location = cacheroot / key
            if location.exists():
                print("cache hit, loading...")
                out = pa.parquet.read_table(str(location))
            else:
                print("cache miss, computing...")
                out = fun(*args, **kwargs)
                pa.parquet.write_table(out, str(location))

            tac = time.perf_counter()
            print(f"took {tac - tic:0.1f}s")
            return out

        return wrapper

    return decorator
