"""
McMurchie-Davidson building blocks: Hermite expansion coefficients E and
Hermite Coulomb integrals R.

McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978); notation follows
Helgaker, Jorgensen & Olsen, *Molecular Electronic-Structure Theory* (2000),
sec. 9.5 (E) and 9.9 (R).

Hermite expansion (one Cartesian direction). For Gaussians with exponents
a, b on centers A, B (p = a + b, P = (aA + bB)/p, mu = ab/p, X_AB = A - B):

    x_A^i x_B^j exp(-a x_A^2 - b x_B^2) = sum_{t=0}^{i+j} E^{ij}_t Lambda_t(x_P),
    Lambda_t = (d/dP_x)^t exp(-p x_P^2),

    E^{00}_0     = exp(-mu X_AB^2)
    E^{i+1,j}_t  = E^{ij}_{t-1} / (2p) + X_PA E^{ij}_t + (t+1) E^{ij}_{t+1}
    E^{i,j+1}_t  = E^{ij}_{t-1} / (2p) + X_PB E^{ij}_t + (t+1) E^{ij}_{t+1}

Overlap-type integrals use only t = 0 (int Lambda_t dx = delta_t0 sqrt(pi/p));
Coulomb-type integrals contract E with

    R^n_{000}     = (-2 alpha)^n F_n(alpha |R_PC|^2)
    R^n_{t+1,u,v} = t R^{n+1}_{t-1,u,v} + X_PC R^{n+1}_{t,u,v}     (same for u, v)

so that R_{tuv} = R^0_{tuv} = (d/dP_x)^t (d/dP_y)^u (d/dP_z)^v F_0(alpha |R_PC|^2).
Nuclear attraction: <a|1/r_C|b> = (2pi/p) sum_tuv E^{ab}_tuv R_tuv(p, P - C);
electron repulsion: (ab|cd) = 2pi^(5/2) / (pq sqrt(p+q)) sum E^{ab}_tuv
(-1)^(tau+nu+phi) E^{cd}_{tau nu phi} R_{t+tau,u+nu,v+phi}(pq/(p+q), P - Q).

Hermite index (t, u, v) is linearized in order of increasing degree t+u+v,
so the first NH(L) = (L+1)(L+2)(L+3)/6 indices are exactly those with
t+u+v <= L; HERM_IDX maps (t, u, v) back to the linear index.

Cartesian component tables (PySCF order) and their normalization factors
(see aimd.qc.basis) live here too, as module constants that numba freezes
into the compiled kernels. numba's disk cache does not track them (or the
functions here) for kernels defined in integrals.py: after editing this
file, delete aimd/qc/__pycache__ (see aimd.qc.integrals).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

from aimd.qc.basis import cartesian_components, cartesian_factor, ncart
from aimd.qc.boys import INV_ODD, boys_top

LMAX = 4                     # highest basis angular momentum supported (g)
HMAX = 4 * LMAX + 1          # highest Hermite degree needed (ERI gradients)


def nherm(L: int) -> int:
    return (L + 1) * (L + 2) * (L + 3) // 6


def _cart_tables():
    off = np.zeros(LMAX + 3, dtype=np.int64)
    lxyz, fac = [], []
    for l in range(LMAX + 2):
        off[l + 1] = off[l] + ncart(l)
        for c in cartesian_components(l):
            lxyz.append(c)
            fac.append(cartesian_factor(*c))
    return off, np.array(lxyz, dtype=np.int64), np.array(fac)


CART_OFF, CART_LXYZ, CART_FAC = _cart_tables()


def _herm_tables():
    tuv = []
    for d in range(HMAX + 1):
        for t in range(d, -1, -1):
            for u in range(d - t, -1, -1):
                tuv.append((t, u, d - t - u))
    tuv = np.array(tuv, dtype=np.int64)
    idx = -np.ones((HMAX + 2, HMAX + 2, HMAX + 2), dtype=np.int64)
    for h, (t, u, v) in enumerate(tuv):
        idx[t, u, v] = h
    n = len(tuv)
    # recursion helpers: reduce along the first nonzero direction d
    hdir = np.zeros(n, dtype=np.int64)
    hm1 = np.zeros(n, dtype=np.int64)
    hm2 = np.zeros(n, dtype=np.int64)
    hrc = np.zeros(n)
    for h in range(1, n):
        c = tuv[h].copy()
        d = 0 if c[0] > 0 else (1 if c[1] > 0 else 2)
        hdir[h] = d
        c1 = c.copy()
        c1[d] -= 1
        hm1[h] = idx[tuple(int(v) for v in c1)]
        if c[d] >= 2:
            c2 = c.copy()
            c2[d] -= 2
            hm2[h] = idx[tuple(int(v) for v in c2)]
            hrc[h] = c[d] - 1
    sign = np.where(tuv.sum(axis=1) % 2 == 0, 1.0, -1.0)
    return tuv, idx, hdir, hm1, hm2, hrc, sign


HERM_TUV, HERM_IDX, HERM_DIR, HERM_M1, HERM_M2, HERM_RC, HERM_SIGN = _herm_tables()
NHERM = np.array([nherm(L) for L in range(HMAX + 2)], dtype=np.int64)


@njit(cache=True)
def hermite_e(imax: int, jmax: int, a: float, b: float, xab: float, e00: float,
              out: np.ndarray) -> None:
    """
    Fill out[i, j, t] = E^{ij}_t for i <= imax, j <= jmax, t <= i + j (one
    Cartesian direction; xab = A_x - B_x) with E^{00}_0 = e00 (the caller
    passes exp(-mu X_AB^2) times any constant prefactor). Entries with
    t > i + j are set to zero; ``out`` must be at least
    (imax+1, jmax+1, imax+jmax+2) so the t+1 reads stay in bounds.
    """
    p = a + b
    xpa = -b * xab / p
    xpb = a * xab / p
    h = 0.5 / p
    tmax = imax + jmax + 1
    for i in range(imax + 1):
        for j in range(jmax + 1):
            for t in range(tmax + 1):
                out[i, j, t] = 0.0
    out[0, 0, 0] = e00
    for j in range(1, jmax + 1):
        for t in range(j + 1):
            v = xpb * out[0, j - 1, t] + (t + 1) * out[0, j - 1, t + 1]
            if t > 0:
                v += h * out[0, j - 1, t - 1]
            out[0, j, t] = v
    for i in range(1, imax + 1):
        for j in range(jmax + 1):
            for t in range(i + j + 1):
                v = xpa * out[i - 1, j, t] + (t + 1) * out[i - 1, j, t + 1]
                if t > 0:
                    v += h * out[i - 1, j, t - 1]
                out[i, j, t] = v


@njit(cache=True)
def hermite_r(L: int, alpha: float, x: float, y: float, z: float,
              out: np.ndarray, tmp: np.ndarray) -> None:
    """
    out[h] = R_{tuv}(alpha, (x, y, z)) for all h < NH(L). ``tmp`` is scratch
    of the same length. Levels n = L..0 alternate between the two buffers so
    that level 0 ends up in ``out``; the Boys values F_n(alpha r^2) are
    generated on the way down by the downward recursion from F_L.

    (integrals.py carries an inlined copy of this loop for the ERI kernels.)
    """
    T = alpha * (x * x + y * y + z * z)
    f = boys_top(L, T)
    ex = 0.0
    if L > 0:
        ex = math.exp(-T)
    t2 = 2.0 * T
    m2a = -2.0 * alpha
    pw = 1.0
    for _ in range(L):
        pw *= m2a
    xyz = (x, y, z)
    for n in range(L, -1, -1):
        if n < L:
            f = (t2 * f + ex) * INV_ODD[n]
        if n % 2 == 0:
            cur = out
            prv = tmp
        else:
            cur = tmp
            prv = out
        cur[0] = pw * f
        for h in range(1, NHERM[L - n]):
            cur[h] = xyz[HERM_DIR[h]] * prv[HERM_M1[h]] + HERM_RC[h] * prv[HERM_M2[h]]
        pw /= m2a
