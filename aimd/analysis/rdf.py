"""
Radial distribution function g_AB(r) and running coordination numbers.

For atom sets A and B (element symbols or index lists) the histogram counts
ordered pairs (i in A, j in B, i != j) with r_ij in shells [r_k, r_k + dr),
summed over frames. With n_pairs = |A| |B| - |A ∩ B| such pairs per frame,

    g(r_k) = counts_k / (n_frames * n_pairs * dV_k / V),
    dV_k = 4 pi (r_{k+1}^3 - r_k^3) / 3,

i.e. the pair density relative to n_pairs pairs spread uniformly over V. For
A = B every unordered pair is counted twice, in counts and n_pairs alike, so
the N_A (N_A - 1) normalisation is exact for an ideal gas (no finite-size
1 - 1/N bias).

Normalisation volume V
  periodic cubic box (``box=L``): V = L^3 and r_ij by the minimum-image
      convention, valid for r_max <= L / 2. An ideal gas gives g = 1.
  isolated cluster (``box=None``): there is no bulk density, so V is the
      sampled sphere, V = 4 pi r_max^3 / 3: g is the pair-distance density
      relative to a uniform distribution of the pairs over that sphere. Then
      sum_k g_k dV_k / V = fraction of pairs closer than r_max (1 when r_max
      exceeds the cluster diameter). Example: points uniform in a ball of
      radius R with r_max = 2R give g(r) = 8 (1 - 3r/(4R) + r^3/(16 R^3)).

The running coordination number n_AB(r) = mean number of B atoms (other than
the atom itself) within r of an A atom = cumulative counts / (n_frames |A|)
does not depend on V.

Lengths are in bohr (Allen & Tildesley, *Computer Simulation of Liquids*, 2nd
ed., Sec. 8.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RDFResult:
    r: np.ndarray              # bin centres, bohr, (n_bins,)
    edges: np.ndarray          # bin edges, bohr, (n_bins + 1,)
    g: np.ndarray              # g(r), (n_bins,)
    counts: np.ndarray         # ordered (i, j) pairs per shell, all frames, (n_bins,)
    coordination: np.ndarray   # n_AB at each upper bin edge, (n_bins,)
    volume: float              # normalisation volume V, bohr^3
    n_frames: int
    n_a: int
    n_b: int
    n_pairs: int               # ordered pairs (i in A, j in B, i != j) per frame
    periodic: bool


def _select(spec: str | Sequence[int], symbols: Sequence[str], n_atoms: int) -> np.ndarray:
    if isinstance(spec, str):
        idx = np.array([i for i, s in enumerate(symbols) if s.lower() == spec.lower()], dtype=int)
        if idx.size == 0:
            raise ValueError(f"no atoms of element {spec!r}")
        return idx
    idx = np.unique(np.asarray(spec, dtype=int).reshape(-1))
    if idx.size == 0 or idx.min() < 0 or idx.max() >= n_atoms:
        raise ValueError(f"atom indices must be in [0, {n_atoms - 1}]")
    return idx


def radial_distribution(
    positions: np.ndarray,
    symbols: Sequence[str] | None,
    pair: tuple[str | Sequence[int], str | Sequence[int]],
    r_max: float,
    n_bins: int = 200,
    box: float | None = None,
) -> RDFResult:
    """
    g_AB(r) over ``positions`` (n_frames, N, 3) or (N, 3), bohr.

    ``pair`` = (A, B): element symbols (matched against ``symbols``) or atom
    index lists. ``r_max`` (bohr) and ``n_bins`` set the histogram [0, r_max).
    ``box`` is the edge of a periodic cubic box (bohr); None treats the
    system as an isolated cluster (see the module docstring for both
    normalisations).
    """
    x = np.asarray(positions, dtype=float)
    if x.ndim == 2:
        x = x[None]
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("positions must have shape (n_frames, N, 3) or (N, 3)")
    n_frames, n_atoms = x.shape[:2]
    names = [] if symbols is None else [str(s) for s in symbols]   # list, tuple or array
    if any(isinstance(p, str) for p in pair) and len(names) != n_atoms:
        raise ValueError("selecting by element needs one symbol per atom")
    a = _select(pair[0], names, n_atoms)
    b = _select(pair[1], names, n_atoms)
    if r_max <= 0.0 or n_bins < 1:
        raise ValueError("r_max must be positive and n_bins >= 1")
    if box is not None:
        if box <= 0.0:
            raise ValueError("box must be positive")
        if r_max > 0.5 * box * (1.0 + 1e-12):
            raise ValueError(f"r_max = {r_max} exceeds half the box ({0.5 * box}); "
                             "minimum-image distances are incomplete beyond it")

    distinct = a[:, None] != b[None, :]                    # exclude i == j
    n_pairs = int(distinct.sum())
    if n_pairs == 0:
        raise ValueError("the selections contain no distinct pairs")
    dr = r_max / n_bins
    counts = np.zeros(n_bins, dtype=np.int64)
    chunk = max(1, int(2**21 // max(n_pairs, 1)))          # bound the (f, A, B, 3) array
    for f0 in range(0, n_frames, chunk):
        xa, xb = x[f0 : f0 + chunk, a], x[f0 : f0 + chunk, b]
        d = xb[:, None, :, :] - xa[:, :, None, :]
        if box is not None:
            d -= box * np.round(d / box)
        r = np.sqrt(np.einsum("fijk,fijk->fij", d, d))[:, distinct]
        k = np.floor(r / dr).astype(np.int64)
        counts += np.bincount(k[k < n_bins], minlength=n_bins)

    edges = np.linspace(0.0, r_max, n_bins + 1)
    shell = 4.0 * math.pi / 3.0 * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = box**3 if box is not None else 4.0 * math.pi / 3.0 * r_max**3
    g = counts / (n_frames * n_pairs * shell / volume)
    return RDFResult(
        r=0.5 * (edges[1:] + edges[:-1]),
        edges=edges,
        g=g,
        counts=counts,
        coordination=np.cumsum(counts) / (n_frames * a.size),
        volume=float(volume),
        n_frames=n_frames,
        n_a=int(a.size),
        n_b=int(b.size),
        n_pairs=n_pairs,
        periodic=box is not None,
    )
