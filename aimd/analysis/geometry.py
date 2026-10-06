"""
Internal-coordinate time series: bond lengths, bond angles and dihedrals.

All functions take positions of shape (..., N, 3) in bohr, e.g. one frame
(N, 3) or a trajectory (n_frames, N, 3), and atom index tuples of shape
(n, k) (k = 2, 3, 4); the result has shape (..., n). A single tuple gives
(...,). With ``box`` (edge of a periodic cubic box, bohr) bond vectors use the
minimum-image convention.

  bond_lengths     |r_j - r_i|, bohr
  bond_angles      angle i-j-k at the middle atom j, in [0, 180] degrees
  dihedral_angles  torsion i-j-k-l in (-180, 180] degrees, IUPAC sign
                   convention (IUPAC Gold Book, "torsion angle"; Blondel &
                   Karplus, J. Comput. Chem. 17, 1132 (1996)): looking along
                   j -> k, the angle is positive when bond i-j has to be
                   turned clockwise (by less than 180 degrees) to eclipse k-l.

``degrees=False`` returns radians. Angles that are undefined (a zero-length
bond, or collinear i-j-k / j-k-l for a dihedral) are nan.

Dihedrals are periodic: a torsion fluctuating about 180 degrees jumps between
about +179 and -179, so its plain mean and standard deviation are
meaningless. ``circular_mean`` gives the mean direction, and
``unwrap_about(phi, circular_mean(phi))`` maps the series into one turn
around it, after which linear statistics (block averages) apply as long as
the distribution is concentrated (much narrower than a full turn).
"""

from __future__ import annotations

import numpy as np


def _tuples(indices: np.ndarray | list, k: int, n_atoms: int) -> tuple[np.ndarray, bool]:
    idx = np.asarray(indices, dtype=int)
    single = idx.ndim == 1
    # Check the width before reshaping: (n, k') with k' != k must not be
    # silently regrouped into other tuples.
    if idx.ndim not in (1, 2) or idx.shape[-1] != k:
        raise ValueError(f"expected tuples of {k} atom indices, got shape {idx.shape}")
    idx = idx.reshape(-1, k)
    if idx.size and (idx.min() < 0 or idx.max() >= n_atoms):
        raise ValueError(f"atom indices must be in [0, {n_atoms - 1}]")
    return idx, single


def _vectors(x: np.ndarray, frm: np.ndarray, to: np.ndarray, box: float | None) -> np.ndarray:
    d = x[..., to, :] - x[..., frm, :]
    if box is not None:
        d = d - box * np.round(d / box)
    return d


def _finish(values: np.ndarray, single: bool, degrees: bool, angular: bool = True) -> np.ndarray:
    if angular and degrees:
        values = np.degrees(values)
    return values[..., 0] if single else values


def bond_lengths(
    positions: np.ndarray, pairs: np.ndarray | list, box: float | None = None
) -> np.ndarray:
    """Distances (bohr) between the atom pairs (i, j)."""
    x = np.asarray(positions, dtype=float)
    idx, single = _tuples(pairs, 2, x.shape[-2])
    r = np.linalg.norm(_vectors(x, idx[:, 0], idx[:, 1], box), axis=-1)
    return _finish(r, single, degrees=False, angular=False)


def bond_angles(
    positions: np.ndarray,
    triples: np.ndarray | list,
    degrees: bool = True,
    box: float | None = None,
) -> np.ndarray:
    """Angles i-j-k at atom j."""
    x = np.asarray(positions, dtype=float)
    idx, single = _tuples(triples, 3, x.shape[-2])
    u = _vectors(x, idx[:, 1], idx[:, 0], box)
    v = _vectors(x, idx[:, 1], idx[:, 2], box)
    # atan2(|u x v|, u.v) is accurate near 0 and 180 degrees, unlike arccos.
    cross = np.linalg.norm(np.cross(u, v), axis=-1)
    dot = np.einsum("...i,...i->...", u, v)
    with np.errstate(invalid="ignore"):
        theta = np.arctan2(cross, dot)
    degenerate = (np.linalg.norm(u, axis=-1) == 0.0) | (np.linalg.norm(v, axis=-1) == 0.0)
    return _finish(np.where(degenerate, np.nan, theta), single, degrees)


def dihedral_angles(
    positions: np.ndarray,
    quads: np.ndarray | list,
    degrees: bool = True,
    box: float | None = None,
) -> np.ndarray:
    """
    Torsions i-j-k-l (IUPAC sign, range (-180, 180]):

        b1 = r_j - r_i, b2 = r_k - r_j, b3 = r_l - r_k,
        phi = atan2(|b2| b1 . (b2 x b3), (b1 x b2) . (b2 x b3)).
    """
    x = np.asarray(positions, dtype=float)
    idx, single = _tuples(quads, 4, x.shape[-2])
    b1 = _vectors(x, idx[:, 0], idx[:, 1], box)
    b2 = _vectors(x, idx[:, 1], idx[:, 2], box)
    b3 = _vectors(x, idx[:, 2], idx[:, 3], box)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    b2_len = np.linalg.norm(b2, axis=-1)
    y = b2_len * np.einsum("...i,...i->...", b1, n2)
    xx = np.einsum("...i,...i->...", n1, n2)
    phi = np.arctan2(y, xx)
    phi = np.where(phi <= -np.pi, np.pi, phi)                # (-pi, pi]
    # Undefined when i-j-k or j-k-l is collinear (normal vector ~ 0).
    scale = b2_len * np.maximum(np.linalg.norm(b1, axis=-1), np.linalg.norm(b3, axis=-1))
    tiny = 1e-12 * np.maximum(scale, np.finfo(float).tiny)
    undefined = (np.linalg.norm(n1, axis=-1) <= tiny) | (np.linalg.norm(n2, axis=-1) <= tiny)
    return _finish(np.where(undefined, np.nan, phi), single, degrees)


def circular_mean(angles: np.ndarray, degrees: bool = True) -> float:
    """
    Mean direction atan2(<sin a>, <cos a>) of a series of angles, in
    (-180, 180] degrees (or (-pi, pi] radians); nan entries are ignored, and
    the result is nan when no finite angle is left or the resultant vanishes
    (e.g. a uniform distribution). Mardia & Jupp, *Directional Statistics*
    (2000), Sec. 2.2.
    """
    a = np.asarray(angles, dtype=float).ravel()
    a = a[np.isfinite(a)]
    if degrees:
        a = np.deg2rad(a)
    if a.size == 0:
        return float("nan")
    s, c = np.sin(a).mean(), np.cos(a).mean()
    if np.hypot(s, c) < 1e-12:
        return float("nan")
    m = float(np.arctan2(s, c))
    if m <= -np.pi:
        m = float(np.pi)
    return float(np.rad2deg(m)) if degrees else m


def unwrap_about(angles: np.ndarray, center: float, degrees: bool = True) -> np.ndarray:
    """
    Angles shifted by whole turns into (center - 180, center + 180] (degrees;
    (center - pi, center + pi] in radians), so that linear statistics (mean,
    variance, block averages) of a periodic series concentrated around
    ``center`` are meaningful: a torsion jittering about 180 degrees reads
    +179 / -179 in (-180, 180], but 179 / 181 about center 180.
    """
    a = np.asarray(angles, dtype=float)
    turn = 360.0 if degrees else 2.0 * np.pi
    # (a - center) mod turn in [0, turn) -> (-turn/2, turn/2]
    d = np.mod(a - center, turn)
    d = np.where(d > 0.5 * turn, d - turn, d)
    return center + d
