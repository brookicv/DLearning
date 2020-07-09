from functools import wraps

from tqdm import tqdm as _tqdm


@wraps(_tqdm)
def tqdm(*args, **kwargs):
    with _tqdm(*args, **kwargs) as t:
        try:
            for _ in t:
                yield _
        except KeyboardInterrupt:
            t.close()
            raise KeyboardInterrupt
