"""
Temporarily limiting the BLAS thread pool while numba-parallel kernels run.

The integral kernels run on numba's OpenMP threads; the SCF interleaves them
with many small numpy BLAS/LAPACK calls (matrices of nao <= ~120). After a
call, OpenBLAS worker threads busy-wait for the next one, and so do numba's
OpenMP workers after a parallel region; with as many threads as cores the two
pools starve each other. Measured on 4 cores (ethanol/6-31G*, nao = 57): one
J/K build plus one 57 x 57 ``eigh`` takes 13 ms interleaved with 4 BLAS
threads versus 4.5 ms with one (3.8 ms + 0.3 ms in isolation). Matrices this
small gain nothing from BLAS threads, so :class:`aimd.qc.scf.SCFSolver` runs
with BLAS limited to one thread (``SCFOptions.blas_threads``).

The limit goes through the ``openblas_set_num_threads`` entry points (plain,
``64_``-suffixed and ``scipy_``-prefixed variants, as bundled with numpy and
scipy wheels) of the OpenBLAS libraries loaded in the process, found once in
/proc/self/maps (Linux); failing that, through threadpoolctl if it is
installed; otherwise :func:`limit_blas_threads` does nothing. The setting is
process-wide while the context is active, so other Python threads using BLAS
at the same time are limited too.
"""

from __future__ import annotations

import contextlib
import ctypes
import re
from typing import Callable, Iterator

_SYMBOLS = (
    ("openblas_get_num_threads", "openblas_set_num_threads"),
    ("openblas_get_num_threads64_", "openblas_set_num_threads64_"),
    ("scipy_openblas_get_num_threads", "scipy_openblas_set_num_threads"),
    ("scipy_openblas_get_num_threads64_", "scipy_openblas_set_num_threads64_"),
)

_pools: list[tuple[Callable[[], int], Callable[[int], None]]] | None = None


def _openblas_pools() -> list[tuple[Callable[[], int], Callable[[int], None]]]:
    """(get, set) thread-count functions of every loaded OpenBLAS (cached)."""
    global _pools
    if _pools is not None:
        return _pools
    pools = []
    try:
        with open("/proc/self/maps") as fh:
            paths = sorted({m.group(1) for line in fh
                            if (m := re.search(r"(/\S*openblas\S*\.so[.\d]*)\s*$", line))})
    except OSError:
        paths = []
    for path in paths:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        for get_name, set_name in _SYMBOLS:
            if hasattr(lib, get_name) and hasattr(lib, set_name):
                get, set_ = getattr(lib, get_name), getattr(lib, set_name)
                get.restype = ctypes.c_int
                get.argtypes = []
                set_.restype = None
                set_.argtypes = [ctypes.c_int]
                pools.append((get, set_))
                break
    _pools = pools
    return pools


def blas_thread_counts() -> list[int]:
    """Current thread counts of the OpenBLAS pools found (empty if none)."""
    return [int(get()) for get, _ in _openblas_pools()]


@contextlib.contextmanager
def limit_blas_threads(n: int | None = 1) -> Iterator[None]:
    """Run the block with at most ``n`` BLAS threads (None: leave BLAS alone)."""
    if n is None:
        yield
        return
    n = max(1, int(n))
    pools = _openblas_pools()
    if not pools:
        try:
            from threadpoolctl import threadpool_limits  # pylint: disable=import-outside-toplevel
        except ImportError:
            yield
            return
        with threadpool_limits(limits=n, user_api="blas"):
            yield
        return
    old = [int(get()) for get, _ in pools]
    for (_, set_), k in zip(pools, old):
        if k > n:
            set_(n)
    try:
        yield
    finally:
        for (_, set_), k in zip(pools, old):
            if k > n:
                set_(k)
