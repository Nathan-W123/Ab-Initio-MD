"""
Mean, spread and the standard error of time-correlated MD series.

Successive MD samples are correlated, so std / sqrt(n) underestimates the
error of the mean by the square root of the statistical inefficiency
g = 1 + 2 tau_int (tau_int: integrated autocorrelation time in samples).
Block averaging estimates it without modelling the correlation:

Blocking (Flyvbjerg & Petersen, J. Chem. Phys. 91, 461 (1989))
    Level 0 is the series; level j+1 averages neighbouring pairs of level j
    (dropping the oldest sample when the length is odd), which leaves the
    mean (nearly) unchanged. SE_j = sqrt(var_j / (n_j - 1)) grows with j
    until the blocks are longer than the correlation time and then stays on a
    plateau (with relative scatter 1 / sqrt(2 (n_j - 1))).

Plateau choice (Lee, Needs & Drummond, Phys. Rev. B 83, 245117 (2011); the
same bias / variance balance as Wolff's automatic window for the integrated
autocorrelation time, Comput. Phys. Commun. 156, 143 (2004))
    Blocks of B samples leave a bias SE_B^2 / SE^2 - 1 ~ -tau / B and a
    statistical error ~ sqrt(B / n); the mean squared error is smallest for
    B^3 ~ n tau^2. The level used is the smallest B with

        B^3 > 2 n (SE_B / SE_0)^4      (SE_0 = std / sqrt(n), so the ratio^2 = g).

    Measured on AR(1) series with g = 19 (n = 5e4 - 3e5): SE low by 2 +- 1 %,
    where a chi^2 test for vanishing lag-1 correlation of the block means
    (Jonsson, Phys. Rev. E 98, 043304 (2018)) stops at blocks ~7 tau long and
    is low by 3-7 % (n = 1e6 - 6.5e4).

``block_average(x, block_size=b)`` instead uses fixed blocks of b samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

# Fewer blocks than this give an SE uncertain by more than ~30 %.
_MIN_BLOCKS = 8


@dataclass
class BlockAverage:
    mean: float
    std: float                     # sample standard deviation (ddof = 1)
    sem: float                     # standard error of the mean, from blocking
    sem_error: float               # its own uncertainty, sem / sqrt(2 (n_blocks - 1))
    sem_naive: float               # std / sqrt(n): valid only for uncorrelated samples
    block_size: int                # samples per block at the chosen level
    n_blocks: int
    n_samples: int
    converged: bool                # False: no level met the criterion with >= 8 blocks
    # Every blocking level (automatic mode), finest first.
    block_sizes: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    sems: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def statistical_inefficiency(self) -> float:
        """g = (sem / sem_naive)^2 = 1 + 2 tau_int."""
        return (self.sem / self.sem_naive) ** 2 if self.sem_naive > 0 else 1.0

    @property
    def correlation_time(self) -> float:
        """Integrated autocorrelation time tau_int = (g - 1) / 2, in samples."""
        return 0.5 * (self.statistical_inefficiency - 1.0)


def _fixed_blocks(x: np.ndarray, block_size: int) -> tuple[float, int]:
    n_blocks = x.size // block_size
    if n_blocks < 2:
        raise ValueError(f"block_size {block_size} leaves fewer than 2 blocks")
    # Drop the oldest samples that do not fill a block.
    means = x[x.size - n_blocks * block_size :].reshape(n_blocks, block_size).mean(axis=1)
    return float(means.std(ddof=1) / math.sqrt(n_blocks)), n_blocks


def block_average(x: np.ndarray, block_size: int | None = None) -> BlockAverage:
    """
    Mean, standard deviation and blocked standard error of the 1-D series ``x``
    (module docstring). ``block_size`` fixes the blocks instead of choosing
    the level automatically.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.size
    if n < 4:
        raise ValueError("need at least 4 samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains nan/inf")
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    sem_naive = std / math.sqrt(n)

    if block_size is not None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        sem, n_blocks = _fixed_blocks(x, int(block_size))
        return BlockAverage(mean, std, sem, sem / math.sqrt(2 * (n_blocks - 1)), sem_naive,
                            int(block_size), n_blocks, n, True)

    sizes, lengths, sems = [], [], []
    level, size = x, 1
    while level.size >= 2:
        m = level.size
        dev = level - level.mean()
        sizes.append(size)
        lengths.append(m)
        sems.append(math.sqrt(float(dev @ dev) / (m * (m - 1))))
        if m % 2:
            level = level[1:]                       # drop the oldest sample
        level = 0.5 * (level[0::2] + level[1::2])
        size *= 2
    se0 = sems[0]
    pick = None
    for j, (b, se) in enumerate(zip(sizes, sems)):
        if se0 == 0.0 or b**3 > 2.0 * n * (se / se0) ** 4:
            pick = j
            break
    converged = pick is not None and lengths[pick] >= _MIN_BLOCKS
    if not converged:
        # Too little data for the criterion: the coarsest level that still
        # has a few blocks (flagged as not converged).
        enough = [j for j, m in enumerate(lengths) if m >= _MIN_BLOCKS]
        pick = enough[-1] if enough else 0
    sem, n_blocks = sems[pick], lengths[pick]
    return BlockAverage(
        mean=mean,
        std=std,
        sem=sem,
        sem_error=sem / math.sqrt(2 * (n_blocks - 1)),
        sem_naive=sem_naive,
        block_size=sizes[pick],
        n_blocks=n_blocks,
        n_samples=n,
        converged=converged,
        block_sizes=np.array(sizes),
        sems=np.array(sems),
    )


DEFAULT_COLUMNS = ("potential_Eh", "kinetic_Eh", "total_Eh", "temperature_K", "conserved_Eh")


def column_statistics(
    log: Mapping[str, np.ndarray] | Any,
    columns: Iterable[str] = DEFAULT_COLUMNS,
    skip: int = 0,
    block_size: int | None = None,
) -> dict[str, BlockAverage]:
    """
    :func:`block_average` of each column of an energy log, after discarding
    the first ``skip`` rows (equilibration). ``log`` is a mapping column ->
    array (``read_energy_log``) or an ``MDResult`` (anything with
    ``.column(name)``). Columns the log lacks are skipped.
    """
    out: dict[str, BlockAverage] = {}
    for name in columns:
        if hasattr(log, "column"):
            try:
                values = np.asarray(log.column(name), dtype=float)
            except KeyError:
                continue
        elif name in log:
            values = np.asarray(log[name], dtype=float)
        else:
            continue
        out[name] = block_average(values[skip:], block_size)
    return out
