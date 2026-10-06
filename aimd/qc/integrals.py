"""
Molecular integrals over contracted Cartesian Gaussians, and the nuclear-
derivative contractions needed for analytic SCF gradients.

Method: McMurchie-Davidson (Hermite Gaussian) scheme with the Boys function,
see McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978) and Helgaker,
Jorgensen & Olsen, *Molecular Electronic-Structure Theory* (2000), ch. 9.
Building blocks live in :mod:`aimd.qc.hermite` and :mod:`aimd.qc.boys`; all
loops are numba-jitted (``cache=True``) and work shell by shell on the flat
arrays of :class:`aimd.qc.basis.BasisSet`.

Units: Hartree atomic units throughout (bohr, hartree). AOs are the unit-
normalized Cartesian functions described in :mod:`aimd.qc.basis`.

Integrals
---------
  overlap_matrix        S_mn = <m|n>
  kinetic_matrix        T_mn = <m| -1/2 nabla^2 |n>
  nuclear_attraction_matrix
                        V_mn = -sum_C Z_C <m| 1/|r - R_C| |n>
  dipole_integrals      <m| r - O |n>, shape (3, nao, nao); the electronic
                        dipole is -sum_mn D_mn <m|r - O|n> (electron charge -1)
  eri_tensor            (mn|ls) = int m(1) n(1) |r1 - r2|^-1 l(2) s(2), full
                        (nao, nao, nao, nao) array (chemists' notation)
  jk_from_eri           J_mn = sum (mn|ls) D_ls, K_mn = sum (ml|ns) D_ls
  nuclear_repulsion_energy / _gradient, nuclear_dipole

ERI tensor size limit: the full tensor takes 8 nao^4 bytes (nao = 100: 0.8
GB, nao = 150: 4 GB), so ``eri_tensor`` refuses nao > 120 (~1.6 GB) unless
``max_nao`` is raised. The two-electron kernels loop over the basis's
contraction blocks (:mod:`aimd.qc.basis`: the shells of a general
contraction share their primitives), evaluating each primitive quartet once
and contracting it into every combination of the blocks' contractions, as
libcint does for general contractions. Only block quartets unique under the
8-fold permutational symmetry are computed, and quartets with Q_ab Q_cd below
``schwarz_threshold`` (Q_ab = max |(ab|ab)|^(1/2) over the block pair; Haser &
Ahlrichs, J. Comput. Chem. 10, 104 (1989)) are skipped; the Schwarz
inequality makes this a strict bound, |dropped (mn|ls)| < threshold.
Primitive pairs with mu |A - B|^2 > PRIM_CUTOFF (exp(-mu R^2) < 2e-22) are
dropped as well. The gradient contraction never stores any 4-index array.

Parallelism: the quartet loops (Schwarz bounds, ERI tensor, ERI gradient,
J/K) run on numba threads (``NUMBA_NUM_THREADS`` / ``numba.set_num_threads``)
over a fixed number of work chunks, so results are bitwise identical for any
thread count. Block-pair data and Schwarz bounds are cached on the BasisSet
(i.e. per geometry) and shared by eri_tensor and two_electron_gradient.

Compiled-code cache: numba's on-disk cache (``cache=True``) is keyed on the
file that defines each kernel. The kernels here embed compiled copies of
hermite.py / boys.py functions and their module-level tables, so editing
only hermite.py, boys.py or basis.py does NOT recompile them; delete
aimd/qc/__pycache__ (or touch this file) after such an edit.

Gradient contractions (all return d/dR_A, shape (natm, 3), hartree/bohr)
----------------------------------------------------------------------
Densities are AO matrices in this basis; they must be symmetric (they are
symmetrized on input). The AOs move with their atoms; densities are held
fixed, i.e. these are the "integral-derivative" parts of the gradient.

  overlap_gradient(basis, W)         d/dR [ sum_mn W_mn S_mn ]
  kinetic_gradient(basis, D)         d/dR [ sum_mn D_mn T_mn ]
  nuclear_attraction_gradient(basis, D)
                                     d/dR [ sum_mn D_mn V_mn ], including the
                                     Hellmann-Feynman (operator-center) term
  one_electron_gradient(basis, D, W) kinetic + nuclear_attraction - overlap
  two_electron_gradient(basis, D_coulomb, D_exchange_list, k_factor)
        d/dR of E2 = 1/2 sum D^c_mn D^c_ls (mn|ls)
                     - k/2 sum_s sum D^s_ml D^s_ns (mn|ls)
        RHF: D_coulomb = P, D_exchange_list = [P], k_factor = 0.5
        UHF: D_coulomb = Pa + Pb, D_exchange_list = [Pa, Pb], k_factor = 1.0

so that an SCF gradient is

    dE/dR = nuclear_repulsion_gradient + one_electron_gradient(D, W)
            + two_electron_gradient(...)

Energy-weighted density convention (the Pulay/orthonormality term):

    W_mn = sum_i n_i eps_i C_mi C_ni      (sum over occupied orbitals)

with n_i = 2 for RHF (W = 2 C_occ eps_occ C_occ^T = 1/2 P F P for the total
density P) and n_i = 1 for each UHF spin orbital (W = Wa + Wb, W_s = D_s F_s
D_s). one_electron_gradient subtracts the overlap term: it returns
d/dR[ tr(D h) ] - d/dR[ tr(W S) ].

Derivative integrals come from the Gaussian-center identity
    d/dA_x [x_A^i exp(-a x_A^2)] = 2a x_A^(i+1) exp(-a x_A^2) - i x_A^(i-1) exp(-a x_A^2)
applied to the Hermite expansions (E^{i+1,j}, E^{i-1,j}), and from
dR_tuv/dC_x = -R_{t+1,u,v} for the nuclear-attraction operator center.
Translational invariance (sum over centers = 0) supplies the remaining
center of each two-center (S, T), three-center (V) term. For each ERI shell
quartet the bra-center derivatives are contracted with the density on the
fly; the ket centers come from a second pass with bra and ket swapped,
which is skipped (via invariance) when the bra or ket pair sits on one atom.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numba import njit, prange

from aimd.qc.basis import BasisSet
from aimd.qc.boys import INV_ODD, boys_top
from aimd.qc.hermite import (
    CART_FAC, CART_LXYZ, CART_OFF, HERM_DIR, HERM_IDX, HERM_M1, HERM_M2, HERM_RC, HERM_SIGN,
    HERM_TUV, LMAX, NHERM, hermite_e, hermite_r,
)

_PI = math.pi
_TWO_PI_2_5 = 2.0 * math.pi ** 2.5
DEFAULT_SCHWARZ = 1e-12
# The Schwarz bound is not strict for derivative integrals (they carry extra
# factors ~2a); measured on ethanol/6-31G* with an RHF density, screening the
# gradient at Q Q max|Gamma| < 1e-12 / 1e-13 / 1e-14 gives max errors of
# 1.2e-10 / 9e-12 / 4e-13 Eh/bohr, at no measurable cost difference.
DEFAULT_SCHWARZ_GRAD = 1e-14
MAX_NAO_ERI = 120


def _check_lmax(basis: BasisSet) -> None:
    if basis.lmax > LMAX:
        raise ValueError(f"angular momentum l = {basis.lmax} > supported LMAX = {LMAX}")


def _sym(D: np.ndarray, nao: int, what: str) -> np.ndarray:
    D = np.asarray(D, dtype=float)
    if D.shape != (nao, nao):
        raise ValueError(f"{what} must have shape ({nao}, {nao}), got {D.shape}")
    return np.ascontiguousarray(0.5 * (D + D.T))


def _charges(basis: BasisSet, charges) -> np.ndarray:
    if charges is None:
        return basis.nuclear_charges.copy()
    q = np.asarray(charges, dtype=float).reshape(-1)
    if q.shape[0] != basis.natm:
        raise ValueError(f"expected {basis.natm} charges, got {q.shape[0]}")
    return q


# ============================================================ one-electron

@njit(cache=True)
def _kin1d(E, i, j, b):
    """1D kinetic factor -1/2 <i| d^2/dx^2 |j> from overlap factors E[i, j, 0]."""
    v = -2.0 * b * (2 * j + 1) * E[i, j, 0] + 4.0 * b * b * E[i, j + 2, 0]
    if j >= 2:
        v += j * (j - 1) * E[i, j - 2, 0]
    return -0.5 * v


@njit(cache=True)
def _overlap_type_kernel(shell_atom, shell_l, shell_prim, shell_ao, prim_exp, prim_coef,
                         positions, origin, S, T, M):
    """S, T and dipole <m|r - O|n> (lower shell triangle, then mirrored)."""
    ns = shell_l.shape[0]
    lmax = 0
    for s in range(ns):
        lmax = max(lmax, shell_l[s])
    E = np.zeros((3, lmax + 1, lmax + 3, 2 * lmax + 5))
    sq_pi = math.sqrt(_PI)
    for si in range(ns):
        li = shell_l[si]
        ai = shell_atom[si]
        oi = shell_ao[si]
        ci0 = CART_OFF[li]
        nci = CART_OFF[li + 1] - ci0
        for sj in range(si + 1):
            lj = shell_l[sj]
            aj = shell_atom[sj]
            oj = shell_ao[sj]
            cj0 = CART_OFF[lj]
            ncj = CART_OFF[lj + 1] - cj0
            for ka in range(shell_prim[si], shell_prim[si + 1]):
                a = prim_exp[ka]
                for kb in range(shell_prim[sj], shell_prim[sj + 1]):
                    b = prim_exp[kb]
                    p = a + b
                    mu = a * b / p
                    sq = sq_pi / math.sqrt(p)
                    for d in range(3):
                        xab = positions[ai, d] - positions[aj, d]
                        e00 = math.exp(-mu * xab * xab) * sq
                        if d == 0:
                            e00 *= prim_coef[ka] * prim_coef[kb]
                        hermite_e(li, lj + 2, a, b, xab, e00, E[d])
                    px = (a * positions[ai, 0] + b * positions[aj, 0]) / p - origin[0]
                    py = (a * positions[ai, 1] + b * positions[aj, 1]) / p - origin[1]
                    pz = (a * positions[ai, 2] + b * positions[aj, 2]) / p - origin[2]
                    for ia in range(nci):
                        ix = CART_LXYZ[ci0 + ia, 0]
                        iy = CART_LXYZ[ci0 + ia, 1]
                        iz = CART_LXYZ[ci0 + ia, 2]
                        fa = CART_FAC[ci0 + ia]
                        for ib in range(ncj):
                            jx = CART_LXYZ[cj0 + ib, 0]
                            jy = CART_LXYZ[cj0 + ib, 1]
                            jz = CART_LXYZ[cj0 + ib, 2]
                            f = fa * CART_FAC[cj0 + ib]
                            sx = E[0, ix, jx, 0]
                            sy = E[1, iy, jy, 0]
                            sz = E[2, iz, jz, 0]
                            kx = _kin1d(E[0], ix, jx, b)
                            ky = _kin1d(E[1], iy, jy, b)
                            kz = _kin1d(E[2], iz, jz, b)
                            m = oi + ia
                            n = oj + ib
                            S[m, n] += f * sx * sy * sz
                            T[m, n] += f * (kx * sy * sz + sx * ky * sz + sx * sy * kz)
                            M[0, m, n] += f * (E[0, ix, jx, 1] + px * sx) * sy * sz
                            M[1, m, n] += f * sx * (E[1, iy, jy, 1] + py * sy) * sz
                            M[2, m, n] += f * sx * sy * (E[2, iz, jz, 1] + pz * sz)
            if si != sj:
                for m in range(oi, oi + nci):
                    for n in range(oj, oj + ncj):
                        S[n, m] = S[m, n]
                        T[n, m] = T[m, n]
                        for d in range(3):
                            M[d, n, m] = M[d, m, n]


@njit(cache=True)
def _herm3(E, ax, ay, az, bx, by, bz, h):
    """3D Hermite coefficient E^{ab}_{tuv} for linear Hermite index h."""
    return (E[0, ax, bx, HERM_TUV[h, 0]] * E[1, ay, by, HERM_TUV[h, 1]]
            * E[2, az, bz, HERM_TUV[h, 2]])


@njit(cache=True)
def _dherm3_a(E, ax, ay, az, bx, by, bz, h, d, a):
    """
    Hermite coefficient h of d/dA_d of the Gaussian product (exponent a on
    center A), via d/dA_x g_i = 2a g_{i+1} - i g_{i-1}; E must hold i <= l_a + 1.
    """
    t = HERM_TUV[h, 0]
    u = HERM_TUV[h, 1]
    v = HERM_TUV[h, 2]
    ex = E[0, ax, bx, t]
    ey = E[1, ay, by, u]
    ez = E[2, az, bz, v]
    if d == 0:
        dx = 2.0 * a * E[0, ax + 1, bx, t]
        if ax > 0:
            dx -= ax * E[0, ax - 1, bx, t]
        return dx * ey * ez
    elif d == 1:
        dy = 2.0 * a * E[1, ay + 1, by, u]
        if ay > 0:
            dy -= ay * E[1, ay - 1, by, u]
        return ex * dy * ez
    dz = 2.0 * a * E[2, az + 1, bz, v]
    if az > 0:
        dz -= az * E[2, az - 1, bz, v]
    return ex * ey * dz


@njit(cache=True)
def _nuclear_kernel(shell_atom, shell_l, shell_prim, shell_ao, prim_exp, prim_coef,
                    positions, charges, charge_pos, V):
    """V_mn = -sum_C Z_C <m|1/r_C|n> = (2pi/p) sum_h E_h sum_C (-Z_C) R_h(p, P - C)."""
    ns = shell_l.shape[0]
    nc = charges.shape[0]
    lmax = 0
    for s in range(ns):
        lmax = max(lmax, shell_l[s])
    E = np.zeros((3, lmax + 1, lmax + 1, 2 * lmax + 2))
    nhmax = NHERM[2 * lmax]
    R = np.zeros(nhmax)
    Rt = np.zeros(nhmax)
    Rsum = np.zeros(nhmax)
    for si in range(ns):
        li = shell_l[si]
        ai = shell_atom[si]
        oi = shell_ao[si]
        ci0 = CART_OFF[li]
        nci = CART_OFF[li + 1] - ci0
        for sj in range(si + 1):
            lj = shell_l[sj]
            aj = shell_atom[sj]
            oj = shell_ao[sj]
            cj0 = CART_OFF[lj]
            ncj = CART_OFF[lj + 1] - cj0
            L = li + lj
            nh = NHERM[L]
            for ka in range(shell_prim[si], shell_prim[si + 1]):
                a = prim_exp[ka]
                for kb in range(shell_prim[sj], shell_prim[sj + 1]):
                    b = prim_exp[kb]
                    p = a + b
                    mu = a * b / p
                    for d in range(3):
                        xab = positions[ai, d] - positions[aj, d]
                        e00 = math.exp(-mu * xab * xab)
                        if d == 0:
                            e00 *= prim_coef[ka] * prim_coef[kb]
                        hermite_e(li, lj, a, b, xab, e00, E[d])
                    # P - C = (A - C) - (b/p)(A - B): exact zero when A = B = C,
                    # and no loss of digits to large absolute coordinates
                    bp = b / p
                    for h in range(nh):
                        Rsum[h] = 0.0
                    for c in range(nc):
                        X = (positions[ai, 0] - charge_pos[c, 0]) - bp * (positions[ai, 0] - positions[aj, 0])
                        Y = (positions[ai, 1] - charge_pos[c, 1]) - bp * (positions[ai, 1] - positions[aj, 1])
                        Z = (positions[ai, 2] - charge_pos[c, 2]) - bp * (positions[ai, 2] - positions[aj, 2])
                        hermite_r(L, p, X, Y, Z, R, Rt)
                        w = -charges[c] * 2.0 * _PI / p
                        for h in range(nh):
                            Rsum[h] += w * R[h]
                    for ia in range(nci):
                        ix = CART_LXYZ[ci0 + ia, 0]
                        iy = CART_LXYZ[ci0 + ia, 1]
                        iz = CART_LXYZ[ci0 + ia, 2]
                        fa = CART_FAC[ci0 + ia]
                        for ib in range(ncj):
                            jx = CART_LXYZ[cj0 + ib, 0]
                            jy = CART_LXYZ[cj0 + ib, 1]
                            jz = CART_LXYZ[cj0 + ib, 2]
                            acc = 0.0
                            for h in range(nh):
                                acc += _herm3(E, ix, iy, iz, jx, jy, jz, h) * Rsum[h]
                            V[oi + ia, oj + ib] += fa * CART_FAC[cj0 + ib] * acc
            if si != sj:
                for m in range(oi, oi + nci):
                    for n in range(oj, oj + ncj):
                        V[n, m] = V[m, n]


@njit(cache=True)
def _overlap_kinetic_grad_kernel(shell_atom, shell_l, shell_prim, shell_ao, prim_exp,
                                 prim_coef, positions, W, D, grad):
    """grad += d/dR [ sum W_mn S_mn + sum D_mn T_mn ] (W, D symmetric)."""
    ns = shell_l.shape[0]
    lmax = 0
    for s in range(ns):
        lmax = max(lmax, shell_l[s])
    E = np.zeros((3, lmax + 2, lmax + 3, 2 * lmax + 6))
    sq_pi = math.sqrt(_PI)
    S1 = np.zeros(3)
    K1 = np.zeros(3)
    dS = np.zeros(3)
    dK = np.zeros(3)
    for si in range(ns):
        li = shell_l[si]
        ai = shell_atom[si]
        oi = shell_ao[si]
        ci0 = CART_OFF[li]
        nci = CART_OFF[li + 1] - ci0
        for sj in range(si):
            aj = shell_atom[sj]
            if aj == ai:
                continue     # two-center integrals on one atom are invariant
            lj = shell_l[sj]
            oj = shell_ao[sj]
            cj0 = CART_OFF[lj]
            ncj = CART_OFF[lj + 1] - cj0
            g0 = 0.0
            g1 = 0.0
            g2 = 0.0
            for ka in range(shell_prim[si], shell_prim[si + 1]):
                a = prim_exp[ka]
                for kb in range(shell_prim[sj], shell_prim[sj + 1]):
                    b = prim_exp[kb]
                    p = a + b
                    mu = a * b / p
                    sq = sq_pi / math.sqrt(p)
                    for d in range(3):
                        xab = positions[ai, d] - positions[aj, d]
                        e00 = math.exp(-mu * xab * xab) * sq
                        if d == 0:
                            e00 *= prim_coef[ka] * prim_coef[kb]
                        hermite_e(li + 1, lj + 2, a, b, xab, e00, E[d])
                    for ia in range(nci):
                        fa = CART_FAC[ci0 + ia]
                        for ib in range(ncj):
                            f = 2.0 * fa * CART_FAC[cj0 + ib]   # 2: (m,n) and (n,m)
                            wmn = f * W[oi + ia, oj + ib]
                            dmn = f * D[oi + ia, oj + ib]
                            for d in range(3):
                                i = CART_LXYZ[ci0 + ia, d]
                                j = CART_LXYZ[cj0 + ib, d]
                                S1[d] = E[d, i, j, 0]
                                K1[d] = _kin1d(E[d], i, j, b)
                                dS[d] = 2.0 * a * E[d, i + 1, j, 0]
                                dK[d] = 2.0 * a * _kin1d(E[d], i + 1, j, b)
                                if i > 0:
                                    dS[d] -= i * E[d, i - 1, j, 0]
                                    dK[d] -= i * _kin1d(E[d], i - 1, j, b)
                            # d/dA_x of S = dSx Sy Sz; of T = dKx Sy Sz + dSx (Ky Sz + Sy Kz)
                            g0 += wmn * dS[0] * S1[1] * S1[2] + dmn * (
                                dK[0] * S1[1] * S1[2] + dS[0] * (K1[1] * S1[2] + S1[1] * K1[2]))
                            g1 += wmn * S1[0] * dS[1] * S1[2] + dmn * (
                                dK[1] * S1[0] * S1[2] + dS[1] * (K1[0] * S1[2] + S1[0] * K1[2]))
                            g2 += wmn * S1[0] * S1[1] * dS[2] + dmn * (
                                dK[2] * S1[0] * S1[1] + dS[2] * (K1[0] * S1[1] + S1[0] * K1[1]))
            grad[ai, 0] += g0
            grad[ai, 1] += g1
            grad[ai, 2] += g2
            grad[aj, 0] -= g0
            grad[aj, 1] -= g1
            grad[aj, 2] -= g2


@njit(cache=True)
def _nuclear_grad_kernel(shell_atom, shell_l, shell_prim, shell_ao, prim_exp, prim_coef,
                         positions, charges, D, grad):
    """grad += d/dR sum_mn D_mn V_mn (basis centers and operator centers C = atoms)."""
    ns = shell_l.shape[0]
    natm = charges.shape[0]
    lmax = 0
    for s in range(ns):
        lmax = max(lmax, shell_l[s])
    E = np.zeros((3, lmax + 2, lmax + 1, 2 * lmax + 3))
    nh1max = NHERM[2 * lmax + 1]
    R = np.zeros(nh1max)
    Rt = np.zeros(nh1max)
    dE = np.zeros(nh1max)
    dA = np.zeros((3, nh1max))
    for si in range(ns):
        li = shell_l[si]
        ai = shell_atom[si]
        oi = shell_ao[si]
        ci0 = CART_OFF[li]
        nci = CART_OFF[li + 1] - ci0
        for sj in range(si + 1):
            lj = shell_l[sj]
            aj = shell_atom[sj]
            oj = shell_ao[sj]
            cj0 = CART_OFF[lj]
            ncj = CART_OFF[lj + 1] - cj0
            L = li + lj
            nh = NHERM[L]
            nh1 = NHERM[L + 1]
            sym = 2.0 if si != sj else 1.0
            for ka in range(shell_prim[si], shell_prim[si + 1]):
                a = prim_exp[ka]
                for kb in range(shell_prim[sj], shell_prim[sj + 1]):
                    b = prim_exp[kb]
                    p = a + b
                    mu = a * b / p
                    for d in range(3):
                        xab = positions[ai, d] - positions[aj, d]
                        e00 = math.exp(-mu * xab * xab)
                        if d == 0:
                            e00 *= prim_coef[ka] * prim_coef[kb]
                        hermite_e(li + 1, lj, a, b, xab, e00, E[d])
                    # density-contracted Hermite vectors of the pair and of d/dA
                    for h in range(nh1):
                        dE[h] = 0.0
                        dA[0, h] = 0.0
                        dA[1, h] = 0.0
                        dA[2, h] = 0.0
                    for ia in range(nci):
                        ix = CART_LXYZ[ci0 + ia, 0]
                        iy = CART_LXYZ[ci0 + ia, 1]
                        iz = CART_LXYZ[ci0 + ia, 2]
                        fa = CART_FAC[ci0 + ia]
                        for ib in range(ncj):
                            jx = CART_LXYZ[cj0 + ib, 0]
                            jy = CART_LXYZ[cj0 + ib, 1]
                            jz = CART_LXYZ[cj0 + ib, 2]
                            w = sym * fa * CART_FAC[cj0 + ib] * D[oi + ia, oj + ib]
                            if w == 0.0:
                                continue
                            for h in range(nh):
                                dE[h] += w * _herm3(E, ix, iy, iz, jx, jy, jz, h)
                            for h in range(nh1):
                                for d in range(3):
                                    dA[d, h] += w * _dherm3_a(E, ix, iy, iz, jx, jy, jz, h, d, a)
                    bp = b / p      # P - C as in _nuclear_kernel
                    for c in range(natm):
                        X = (positions[ai, 0] - positions[c, 0]) - bp * (positions[ai, 0] - positions[aj, 0])
                        Y = (positions[ai, 1] - positions[c, 1]) - bp * (positions[ai, 1] - positions[aj, 1])
                        Z = (positions[ai, 2] - positions[c, 2]) - bp * (positions[ai, 2] - positions[aj, 2])
                        hermite_r(L + 1, p, X, Y, Z, R, Rt)
                        pref = -charges[c] * 2.0 * _PI / p
                        for d in range(3):
                            gA = 0.0
                            for h in range(nh1):
                                gA += dA[d, h] * R[h]
                            # dR_tuv/dC = -R_{tuv + e_d}
                            gC = 0.0
                            for h in range(nh):
                                t = HERM_TUV[h, 0]
                                u = HERM_TUV[h, 1]
                                v = HERM_TUV[h, 2]
                                if d == 0:
                                    t += 1
                                elif d == 1:
                                    u += 1
                                else:
                                    v += 1
                                gC -= dE[h] * R[HERM_IDX[t, u, v]]
                            gA *= pref
                            gC *= pref
                            grad[ai, d] += gA
                            grad[c, d] += gC
                            grad[aj, d] -= gA + gC


# ============================================================ block pairs

# Primitive pairs with mu |A - B|^2 > PRIM_CUTOFF are dropped from the ERI
# pair lists: their Gaussian-product prefactor exp(-mu R^2) < 2e-22 is far
# below anything the normalized coefficients/polynomial factors can lift to
# 1e-15 (cf. libcint's ``expcutoff``).
PRIM_CUTOFF = 50.0
# Fixed number of work chunks for the parallel quartet loops: chunk c owns the
# bra pairs f, f + NCHUNK, f + 2 NCHUNK, ... with f = CHUNK_FIRST[c]
# (interleaved for load balance). Partial gradients are summed per chunk in a
# fixed order, so results do not depend on the number of threads.
NCHUNK = 64
# numba's OpenMP backend gives each of T threads a contiguous block of
# NCHUNK / T chunks. With f = c, a molecule with fewer than NCHUNK / T shell
# pairs (water / STO-3G: 15) had all its work in thread 0's block, i.e. ran
# serially. The 6-bit reversal f = rev(c) gives every block the residues
# f = r (mod T) for one r per thread (T = 2, 4, 8, ...), so small and large
# molecules alike are spread over all threads.
CHUNK_FIRST = np.array([int(f"{c:06b}"[::-1], 2) for c in range(NCHUNK)], dtype=np.int64)


@njit(cache=True)
def _build_pairs(blk_atom, blk_l, blk_prim, blk_exp, blk_coef, blk_ncon, positions):
    """
    Primitive-pair data for every pair of contraction blocks (i >= j), pair
    index i(i+1)/2 + j (blocks: see aimd.qc.basis). E[k, d, i, j, t] holds
    the Hermite coefficients (without contraction coefficients) up to
    i <= l_i + 1, j <= l_j + 1 (enough for first derivatives on either
    center); C[k, ci * ncon_j + cj] = c_i[ka, ci] c_j[kb, cj] are the
    contraction-coefficient products of primitive pair k = (ka, kb).
    """
    nb = blk_l.shape[0]
    npair = nb * (nb + 1) // 2
    lmax = 0
    maxcon = 1
    for s in range(nb):
        lmax = max(lmax, blk_l[s])
        maxcon = max(maxcon, blk_ncon[s])
    pair_pp = np.zeros(npair + 1, dtype=np.int64)
    pair_blocks = np.zeros((npair, 2), dtype=np.int64)
    npp = 0
    ip = 0
    for i in range(nb):
        for j in range(i + 1):
            pair_pp[ip] = npp
            pair_blocks[ip, 0] = i
            pair_blocks[ip, 1] = j
            r2 = 0.0
            for d in range(3):
                r2 += (positions[blk_atom[i], d] - positions[blk_atom[j], d]) ** 2
            for ka in range(blk_prim[i], blk_prim[i + 1]):
                for kb in range(blk_prim[j], blk_prim[j + 1]):
                    a = blk_exp[ka]
                    b = blk_exp[kb]
                    if a * b / (a + b) * r2 <= PRIM_CUTOFF:
                        npp += 1
            ip += 1
    pair_pp[npair] = npp
    pp_ab = np.zeros((npp, 2))
    pp_P = np.zeros((npp, 3))
    pp_E = np.zeros((npp, 3, lmax + 2, lmax + 2, 2 * lmax + 4))
    pp_C = np.zeros((npp, maxcon * maxcon))
    k = 0
    for i in range(nb):
        ai = blk_atom[i]
        for j in range(i + 1):
            aj = blk_atom[j]
            ncj = blk_ncon[j]
            r2 = 0.0
            for d in range(3):
                r2 += (positions[ai, d] - positions[aj, d]) ** 2
            for ka in range(blk_prim[i], blk_prim[i + 1]):
                a = blk_exp[ka]
                for kb in range(blk_prim[j], blk_prim[j + 1]):
                    b = blk_exp[kb]
                    p = a + b
                    mu = a * b / p
                    # exact negation of the counting test above: with a NaN
                    # coordinate both comparisons are False, and "mu r2 >
                    # cutoff" would write past the npp allocated pairs
                    # (numba does no bounds checking -> heap corruption)
                    if not mu * r2 <= PRIM_CUTOFF:
                        continue
                    pp_ab[k, 0] = a
                    pp_ab[k, 1] = b
                    for ci in range(blk_ncon[i]):
                        for cj in range(ncj):
                            pp_C[k, ci * ncj + cj] = blk_coef[ka, ci] * blk_coef[kb, cj]
                    for d in range(3):
                        xab = positions[ai, d] - positions[aj, d]
                        # = (a A + b B)/p, written so that P = A exactly when A = B
                        pp_P[k, d] = positions[ai, d] - (b / p) * xab
                        e00 = math.exp(-mu * xab * xab)
                        hermite_e(blk_l[i] + 1, blk_l[j] + 1, a, b, xab, e00, pp_E[k, d])
                    k += 1
    return pair_pp, pair_blocks, pp_ab, pp_P, pp_E, pp_C


class _Pairs:
    """Geometry-dependent block-pair data and scratch sizes (cached on the BasisSet)."""

    def __init__(self, basis: BasisSet) -> None:
        (self.pair_pp, self.pair_blocks, self.pp_ab, self.pp_P, self.pp_E,
         self.pp_C) = _build_pairs(basis.blk_atom, basis.blk_l, basis.blk_prim, basis.blk_exp,
                                   basis.blk_coef, basis.blk_ncon, basis.positions)
        self.npair = self.pair_blocks.shape[0]
        npp = np.diff(self.pair_pp)
        self.maxpp = max(int(npp.max()), 1) if self.npair else 1
        # largest row count (contractions x Cartesian components) of a pair,
        # and of npp * (number of Hermite functions) for the gradient's Gk
        i, j = self.pair_blocks.T
        l, ncon = basis.blk_l, basis.blk_ncon
        nc = (l + 1) * (l + 2) // 2
        rows = ncon[i] * ncon[j] * nc[i] * nc[j]
        nh = np.array([NHERM[v] for v in (l[i] + l[j])], dtype=np.int64)
        self.maxrow = max(int(rows.max()), 1) if self.npair else 1
        self.maxpph = max(int((npp * nh).max()), 1) if self.npair else 1


def _pairs(basis: BasisSet) -> _Pairs:
    p = basis._cache.get("pairs")
    if p is None:
        p = basis._cache["pairs"] = _Pairs(basis)
    return p


# ============================================================ ERIs
#
# Block quartet (IJ|KL) in the bra pair pb = (I, J) and ket pair pk = (K, L):
# rows r = (ci ncon_J + cj) nab + ab, columns (ck ncon_L + cl) ncd + cd, with
# ab = ia ncart_J + ib and cd likewise. The primitive integrals are
#
#   (ab|cd)_prim = 2 pi^(5/2) / (p q sqrt(p + q))
#                  sum_{tuv} E^{ab}_{tuv} sum_{t'u'v'} (-1)^{t'+u'+v'} E^{cd}_{t'u'v'}
#                  R_{t+t', u+u', v+v'}(alpha, P - Q)
#
# and the contracted ones sum_{bp, kp} C_bp[cij] C_kp[ckl] (ab|cd)_prim.
# Per primitive quartet the Boys function and R_tuv are evaluated once and
# contracted into every (cij, ckl) combination: ket coefficients enter X,
# bra coefficients the final accumulation (folded into scalars when a pair
# has a single contraction, the segmented case).

@njit(cache=True)
def _ket_hermite(pk, pair_pp, pair_blocks, blk_l, pp_E, Ek):
    """Ek[kp, k, cd] = (-1)^{|k|} E^{cd}_k (component factors included) for ket pair pk."""
    sk = pair_blocks[pk, 0]
    sl = pair_blocks[pk, 1]
    lk = blk_l[sk]
    ll = blk_l[sl]
    c0 = CART_OFF[lk]
    nck = CART_OFF[lk + 1] - c0
    d0 = CART_OFF[ll]
    ncl = CART_OFF[ll + 1] - d0
    nhk = NHERM[lk + ll]
    k0 = pair_pp[pk]
    for kp in range(pair_pp[pk + 1] - k0):
        E = pp_E[k0 + kp]
        for k in range(nhk):
            for cd in range(nck * ncl):
                Ek[kp, k, cd] = 0.0
        for ic in range(nck):
            cx = CART_LXYZ[c0 + ic, 0]
            cy = CART_LXYZ[c0 + ic, 1]
            cz = CART_LXYZ[c0 + ic, 2]
            fc = CART_FAC[c0 + ic]
            for idd in range(ncl):
                dx = CART_LXYZ[d0 + idd, 0]
                dy = CART_LXYZ[d0 + idd, 1]
                dz = CART_LXYZ[d0 + idd, 2]
                f = fc * CART_FAC[d0 + idd]
                cd = ic * ncl + idd
                # nonzero only inside the box t <= cx+dx, u <= cy+dy, v <= cz+dz
                for t in range(cx + dx + 1):
                    et = f * E[0, cx, dx, t]
                    for u in range(cy + dy + 1):
                        etu = et * E[1, cy, dy, u]
                        for v in range(cz + dz + 1):
                            k = HERM_IDX[t, u, v]
                            Ek[kp, k, cd] = HERM_SIGN[k] * etu * E[2, cz, dz, v]


@njit(cache=True)
def _hidx_fill(nh, nhk, hidx):
    """hidx[h, k] = linear Hermite index of (t_h + t_k, u_h + u_k, v_h + v_k)."""
    for h in range(nh):
        for k in range(nhk):
            hidx[h, k] = HERM_IDX[HERM_TUV[h, 0] + HERM_TUV[k, 0], HERM_TUV[h, 1] + HERM_TUV[k, 1],
                                  HERM_TUV[h, 2] + HERM_TUV[k, 2]]


@njit(cache=True, inline="always")
def _pair_dims(ip, pair_blocks, blk_l, blk_ncon):
    """(Cartesian component pairs, contraction pairs) of block pair ip."""
    i = pair_blocks[ip, 0]
    j = pair_blocks[ip, 1]
    li = blk_l[i]
    lj = blk_l[j]
    return ((CART_OFF[li + 1] - CART_OFF[li]) * (CART_OFF[lj + 1] - CART_OFF[lj]),
            blk_ncon[i] * blk_ncon[j])


@njit(cache=True, inline="always")
def _ket_r_contract(bp, pk, nh, L, pair_pp, pp_ab, pp_P, pp_C, Ek, nhk, ncd, nkc, X, T, RR, hidx):
    """
    X[h, ckl ncd + cd] = sum_kp C_kp[ckl] sum_k pref R_{h+k}(alpha, P - Q) Ek[kp, k, cd]
    for bra primitive pair bp, h < nh (R up to order L), over the ket pair pk.
    """
    p = pp_ab[bp, 0] + pp_ab[bp, 1]
    for h in range(nh):
        for col in range(nkc * ncd):
            X[h, col] = 0.0
    k0 = pair_pp[pk]
    for kp in range(pair_pp[pk + 1] - k0):
        q = pp_ab[k0 + kp, 0] + pp_ab[k0 + kp, 1]
        alpha = p * q / (p + q)
        pref = _TWO_PI_2_5 / (p * q * math.sqrt(p + q))
        x = pp_P[bp, 0] - pp_P[k0 + kp, 0]
        y = pp_P[bp, 1] - pp_P[k0 + kp, 1]
        z = pp_P[bp, 2] - pp_P[k0 + kp, 2]
        # R_tuv(alpha, P - Q) into RR[0, :NH(L)]: inlined copy of
        # hermite.hermite_r (a call with array arguments costs more than
        # the work here); levels n = L..0 ping-pong between RR[0], RR[1]
        # and F_n comes from the downward recursion
        T2 = alpha * (x * x + y * y + z * z)
        f = boys_top(L, T2)
        ex = 0.0
        if L > 0:
            ex = math.exp(-T2)
        m2a = -2.0 * alpha
        pw = 1.0
        for _ in range(L):
            pw *= m2a
        for n in range(L, -1, -1):
            if n < L:
                f = (2.0 * T2 * f + ex) * INV_ODD[n]
            cur = n & 1
            prv = 1 - cur
            RR[cur, 0] = pw * f
            for h in range(1, NHERM[L - n]):
                d = HERM_DIR[h]
                xd = x if d == 0 else (y if d == 1 else z)
                RR[cur, h] = xd * RR[prv, HERM_M1[h]] + HERM_RC[h] * RR[prv, HERM_M2[h]]
            pw /= m2a
        if nkc == 1:
            # segmented ket: fold its coefficient into the scalar
            pc = pref * pp_C[k0 + kp, 0]
            for h in range(nh):
                for k in range(nhk):
                    r = pc * RR[0, hidx[h, k]]
                    if r == 0.0:
                        continue
                    for cd in range(ncd):
                        X[h, cd] += r * Ek[kp, k, cd]
        else:
            # general ket: contract over k once, then spread over contractions
            for h in range(nh):
                for cd in range(ncd):
                    T[cd] = 0.0
                for k in range(nhk):
                    r = pref * RR[0, hidx[h, k]]
                    if r == 0.0:
                        continue
                    for cd in range(ncd):
                        T[cd] += r * Ek[kp, k, cd]
                for ckl in range(nkc):
                    c = pp_C[k0 + kp, ckl]
                    if c == 0.0:
                        continue
                    o = ckl * ncd
                    for cd in range(ncd):
                        X[h, o + cd] += c * T[cd]


@njit(cache=True)
def _eri_block(pb, pk, pair_pp, pair_blocks, blk_l, blk_ncon, pp_ab, pp_P, pp_E, pp_C,
               block, Ek, X, T, W, RR, hidx):
    """block[r, s] = (IJ|KL) for bra block pair pb and ket block pair pk (layout above)."""
    si = pair_blocks[pb, 0]
    sj = pair_blocks[pb, 1]
    sk = pair_blocks[pk, 0]
    sl = pair_blocks[pk, 1]
    li = blk_l[si]
    lj = blk_l[sj]
    a0 = CART_OFF[li]
    nci = CART_OFF[li + 1] - a0
    b0 = CART_OFF[lj]
    ncj = CART_OFF[lj + 1] - b0
    nab, nbc = _pair_dims(pb, pair_blocks, blk_l, blk_ncon)
    ncd, nkc = _pair_dims(pk, pair_blocks, blk_l, blk_ncon)
    ncol = nkc * ncd
    Lab = li + lj
    L = Lab + blk_l[sk] + blk_l[sl]
    nhb = NHERM[Lab]
    nhk = NHERM[blk_l[sk] + blk_l[sl]]
    _hidx_fill(nhb, nhk, hidx)
    _ket_hermite(pk, pair_pp, pair_blocks, blk_l, pp_E, Ek)
    for r in range(nbc * nab):
        for col in range(ncol):
            block[r, col] = 0.0
    for bp in range(pair_pp[pb], pair_pp[pb + 1]):
        _ket_r_contract(bp, pk, nhb, L, pair_pp, pp_ab, pp_P, pp_C, Ek, nhk, ncd, nkc,
                        X, T, RR, hidx)
        # bra Hermite expansion: segmented bra -> straight into block (with
        # its coefficient), general bra -> W[ab, :], then into every cij
        cb = pp_C[bp, 0] if nbc == 1 else 1.0
        E = pp_E[bp]
        for ia in range(nci):
            ax = CART_LXYZ[a0 + ia, 0]
            ay = CART_LXYZ[a0 + ia, 1]
            az = CART_LXYZ[a0 + ia, 2]
            fa = cb * CART_FAC[a0 + ia]
            for ib in range(ncj):
                bx = CART_LXYZ[b0 + ib, 0]
                by = CART_LXYZ[b0 + ib, 1]
                bz = CART_LXYZ[b0 + ib, 2]
                f = fa * CART_FAC[b0 + ib]
                ab = ia * ncj + ib
                out = block[ab] if nbc == 1 else W[ab]
                if nbc != 1:
                    for col in range(ncol):
                        out[col] = 0.0
                for t in range(ax + bx + 1):
                    et = f * E[0, ax, bx, t]
                    for u in range(ay + by + 1):
                        etu = et * E[1, ay, by, u]
                        for v in range(az + bz + 1):
                            e = etu * E[2, az, bz, v]
                            h = HERM_IDX[t, u, v]
                            for col in range(ncol):
                                out[col] += e * X[h, col]
        if nbc != 1:
            for cij in range(nbc):
                c = pp_C[bp, cij]
                if c == 0.0:
                    continue
                for ab in range(nab):
                    r = cij * nab + ab
                    for col in range(ncol):
                        block[r, col] += c * W[ab, col]


@njit(cache=True)
def _scratch(lmax, maxpp, maxrow, deriv):
    """
    Work arrays sized for the largest block quartet. Uninitialized: every
    kernel writes the part it uses before reading it.
    """
    nc = (lmax + 1) * (lmax + 2) // 2
    nhp = NHERM[2 * lmax + deriv]
    nR = NHERM[4 * lmax + deriv]
    block = np.empty((maxrow, maxrow))
    Ek = np.empty((maxpp, NHERM[2 * lmax], nc * nc))
    X = np.empty((nhp, maxrow))
    T = np.empty(nc * nc)
    W = np.empty((nc * nc, maxrow))
    RR = np.empty((2, nR))
    hidx = np.empty((nhp, NHERM[2 * lmax]), dtype=np.int64)
    return block, Ek, X, T, W, RR, hidx


@njit(cache=True, parallel=True)
def _schwarz_kernel(pair_pp, pair_blocks, blk_l, blk_ncon, pp_ab, pp_P, pp_E, pp_C,
                    lmax, maxpp, maxrow, chunk_first):
    """Q[ij] = max_ab |(ab|ab)|^(1/2) over each block pair (all its contractions)."""
    npair = pair_blocks.shape[0]
    nchunk = chunk_first.shape[0]
    Q = np.zeros(npair)
    for c in prange(nchunk):
        if chunk_first[c] >= npair:
            continue
        block, Ek, X, T, W, RR, hidx = _scratch(lmax, maxpp, maxrow, 0)
        for ip in range(chunk_first[c], npair, nchunk):
            _eri_block(ip, ip, pair_pp, pair_blocks, blk_l, blk_ncon, pp_ab, pp_P, pp_E, pp_C,
                       block, Ek, X, T, W, RR, hidx)
            nab, nbc = _pair_dims(ip, pair_blocks, blk_l, blk_ncon)
            m = 0.0
            for r in range(nab * nbc):
                m = max(m, abs(block[r, r]))
            Q[ip] = math.sqrt(m)
    return Q


@njit(cache=True, parallel=True)
def _eri_tensor_kernel(pair_pp, pair_blocks, blk_l, blk_ncon, blk_ao, pp_ab, pp_P, pp_E, pp_C,
                       lmax, maxpp, maxrow, Q, threshold, chunk_first, eri):
    """
    Writes (mn|ls) and its bra/ket-internal permutations for unique block
    quartets (bra pair >= ket pair), halving diagonal (bra == ket) quartets;
    _symmetrize_pairs then adds the bra-ket transpose.
    """
    npair = pair_blocks.shape[0]
    nchunk = chunk_first.shape[0]
    for c in prange(nchunk):
        if chunk_first[c] >= npair:
            continue
        block, Ek, X, T, W, RR, hidx = _scratch(lmax, maxpp, maxrow, 0)
        for p1 in range(chunk_first[c], npair, nchunk):
            for p2 in range(p1 + 1):
                if Q[p1] * Q[p2] < threshold:
                    continue
                # the cost scales with the ket's column count: put the
                # smaller pair in the ket
                n1, c1 = _pair_dims(p1, pair_blocks, blk_l, blk_ncon)
                n2, c2 = _pair_dims(p2, pair_blocks, blk_l, blk_ncon)
                swapped = n1 * c1 < n2 * c2
                if swapped:
                    _eri_block(p2, p1, pair_pp, pair_blocks, blk_l, blk_ncon, pp_ab, pp_P, pp_E,
                               pp_C, block, Ek, X, T, W, RR, hidx)
                else:
                    _eri_block(p1, p2, pair_pp, pair_blocks, blk_l, blk_ncon, pp_ab, pp_P, pp_E,
                               pp_C, block, Ek, X, T, W, RR, hidx)
                scale = 0.5 if p1 == p2 else 1.0
                # always write rows (m n) of the outer pair p1: the rows stay in
                # cache across the inner p2 loop
                si = pair_blocks[p1, 0]
                sj = pair_blocks[p1, 1]
                sk = pair_blocks[p2, 0]
                sl = pair_blocks[p2, 1]
                nci = CART_OFF[blk_l[si] + 1] - CART_OFF[blk_l[si]]
                ncj = CART_OFF[blk_l[sj] + 1] - CART_OFF[blk_l[sj]]
                nck = CART_OFF[blk_l[sk] + 1] - CART_OFF[blk_l[sk]]
                ncl = CART_OFF[blk_l[sl] + 1] - CART_OFF[blk_l[sl]]
                nqj = blk_ncon[sj]
                nql = blk_ncon[sl]
                for qi in range(blk_ncon[si]):
                    for a in range(nci):
                        m = blk_ao[si] + qi * nci + a
                        for qj in range(nqj):
                            for b in range(ncj):
                                n = blk_ao[sj] + qj * ncj + b
                                r1 = (qi * nqj + qj) * n1 + a * ncj + b
                                for qk in range(blk_ncon[sk]):
                                    for cc in range(nck):
                                        lam = blk_ao[sk] + qk * nck + cc
                                        for ql in range(nql):
                                            for d in range(ncl):
                                                sig = blk_ao[sl] + ql * ncl + d
                                                r2 = (qk * nql + ql) * n2 + cc * ncl + d
                                                if swapped:
                                                    v = scale * block[r2, r1]
                                                else:
                                                    v = scale * block[r1, r2]
                                                eri[m, n, lam, sig] = v
                                                eri[n, m, lam, sig] = v
                                                eri[m, n, sig, lam] = v
                                                eri[n, m, sig, lam] = v


@njit(cache=True, parallel=True)
def _symmetrize_pairs(M):
    """M <- M + M^T for a square matrix, in cache-friendly 32 x 32 tiles."""
    N = M.shape[0]
    B = 32
    nb = (N + B - 1) // B
    for I in prange(nb):
        r1 = min(N, (I + 1) * B)
        for J in range(I + 1):
            c0 = J * B
            for r in range(I * B, r1):
                c1 = min(N, (J + 1) * B) if J < I else r + 1
                for c in range(c0, c1):
                    v = M[r, c] + M[c, r]
                    M[r, c] = v
                    M[c, r] = v


def schwarz_bounds(basis: BasisSet) -> np.ndarray:
    """
    Q_P = max_{ab in P} |(ab|ab)|^(1/2) per pair P = i(i+1)/2 + j of the
    basis's contraction blocks (``basis.blk_*``; cached).
    """
    _check_lmax(basis)
    Q = basis._cache.get("schwarz")
    if Q is None:
        pr = _pairs(basis)
        Q = _schwarz_kernel(pr.pair_pp, pr.pair_blocks, basis.blk_l, basis.blk_ncon, pr.pp_ab,
                            pr.pp_P, pr.pp_E, pr.pp_C, basis.lmax, pr.maxpp, pr.maxrow,
                            CHUNK_FIRST)
        basis._cache["schwarz"] = Q
    return Q


def eri_tensor(basis: BasisSet, schwarz_threshold: float = DEFAULT_SCHWARZ,
               max_nao: int = MAX_NAO_ERI) -> np.ndarray:
    """
    Full two-electron integral tensor (mn|ls), shape (nao,)*4, chemists'
    notation. Memory is 8 nao^4 bytes; see the module docstring.
    """
    _check_lmax(basis)
    n = basis.nao
    if n > max_nao:
        raise MemoryError(
            f"ERI tensor for nao = {n} needs {8 * n ** 4 / 1e9:.1f} GB; "
            f"raise max_nao (currently {max_nao}) if you really want it"
        )
    pr = _pairs(basis)
    Q = schwarz_bounds(basis)
    eri = np.zeros((n, n, n, n))
    _eri_tensor_kernel(pr.pair_pp, pr.pair_blocks, basis.blk_l, basis.blk_ncon, basis.blk_ao,
                       pr.pp_ab, pr.pp_P, pr.pp_E, pr.pp_C, basis.lmax, pr.maxpp, pr.maxrow, Q,
                       float(schwarz_threshold), CHUNK_FIRST, eri)
    _symmetrize_pairs(eri.reshape(n * n, n * n))
    return eri


@njit(cache=True, parallel=True)
def _jk_kernel(eri, D, J, K):
    """J[s] = sum_ls (mn|ls) D[s]_ls, K[s] = sum_ls (ml|ns) D[s]_ls, one pass over (mn|..)."""
    n = eri.shape[0]
    nset = D.shape[0]
    for m in prange(n):
        for nn in range(n):
            blk = eri[m, nn]          # (m nn | . .), contiguous n x n
            for s in range(nset):
                acc = 0.0
                for l in range(n):
                    for q in range(n):
                        acc += blk[l, q] * D[s, l, q]
                J[s, m, nn] = acc
                # K_ml += (m nn | l q) D_nn,q
                for l in range(n):
                    acc = 0.0
                    for q in range(n):
                        acc += blk[l, q] * D[s, nn, q]
                    K[s, m, l] += acc


def jk_from_eri(eri: np.ndarray, densities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Coulomb and exchange matrices from the full ERI tensor:
    J_mn = sum_ls (mn|ls) D_ls,  K_mn = sum_ls (ml|ns) D_ls.
    ``densities`` is (nao, nao) or a stack (nset, nao, nao); J, K have the
    same shape. Densities need not be symmetric.
    """
    D = np.asarray(densities, dtype=float)
    single = D.ndim == 2
    D3 = np.ascontiguousarray(D[None] if single else D)
    J = np.zeros_like(D3)
    K = np.zeros_like(D3)
    _jk_kernel(eri, D3, J, K)
    return (J[0], K[0]) if single else (J, K)


# ============================================================ ERI gradient

@njit(cache=True)
def _grad_pass(pb, pk, G, pair_pp, pair_blocks, blk_l, blk_ncon, pp_ab, pp_P, pp_E, pp_C,
               Ek, Gk, Gkb, Gb, Y, X, T, RR, hidx, out):
    """
    out[0, d] = sum_{rs} G[r, s] d(IJ|KL)_{rs}/dA_d and out[1, d] the same
    for B, where A, B are the centers of bra block pair pb (ket pair pk is
    held fixed; layout of G as of the ERI block).

    The bra-center derivatives use shifted Hermite expansions,
    d/dA_x -> 2a E^{i+1,j}_t - i E^{i-1,j}_t (t <= l_a + l_b + 1). Per bra
    primitive pair, G is first reduced to the effective primitive-pair
    density Gb[ab, s] = sum_cij C_bp[cij] G[(cij, ab), s] (for a general
    bra; a segmented bra folds its coefficient into a scalar). G is then
    contracted with the ket either before the R contraction (Gk, cost ~ nab
    per primitive quartet) or after it (X[h, s], cost ~ ncd x ket
    contractions), whichever side is smaller.
    """
    si = pair_blocks[pb, 0]
    sj = pair_blocks[pb, 1]
    sk = pair_blocks[pk, 0]
    sl = pair_blocks[pk, 1]
    li = blk_l[si]
    lj = blk_l[sj]
    lk = blk_l[sk]
    ll = blk_l[sl]
    a0 = CART_OFF[li]
    nci = CART_OFF[li + 1] - a0
    b0 = CART_OFF[lj]
    ncj = CART_OFF[lj + 1] - b0
    nab, nbc = _pair_dims(pb, pair_blocks, blk_l, blk_ncon)
    ncd, nkc = _pair_dims(pk, pair_blocks, blk_l, blk_ncon)
    nrow = nbc * nab
    ncol = nkc * ncd
    Lab = li + lj
    L = Lab + lk + ll + 1
    nh1 = NHERM[Lab + 1]
    nhk = NHERM[lk + ll]
    use_x = ncol < nab
    _hidx_fill(nh1, nhk, hidx)
    _ket_hermite(pk, pair_pp, pair_blocks, blk_l, pp_E, Ek)
    k0 = pair_pp[pk]
    nkp = pair_pp[pk + 1] - k0
    Gk3 = Gk[:nkp * nrow * nhk].reshape((nkp, nrow, nhk))
    Gkb3 = Gkb[:nkp * nab * nhk].reshape((nkp, nab, nhk))
    if not use_x:
        # Gk[kp, r, k] = sum_{ckl, cd} G[r, (ckl, cd)] C_kp[ckl] Ek[kp, k, cd]
        for kp in range(nkp):
            for r in range(nrow):
                for cd in range(ncd):
                    s = 0.0
                    for ckl in range(nkc):
                        s += pp_C[k0 + kp, ckl] * G[r, ckl * ncd + cd]
                    T[cd] = s
                for k in range(nhk):
                    s = 0.0
                    for cd in range(ncd):
                        s += T[cd] * Ek[kp, k, cd]
                    Gk3[kp, r, k] = s
    for d in range(3):
        out[0, d] = 0.0
        out[1, d] = 0.0
    for bp in range(pair_pp[pb], pair_pp[pb + 1]):
        a = pp_ab[bp, 0]
        b = pp_ab[bp, 1]
        p = a + b
        cb = pp_C[bp, 0] if nbc == 1 else 1.0
        if use_x:
            _ket_r_contract(bp, pk, nh1, L, pair_pp, pp_ab, pp_P, pp_C, Ek, nhk, ncd, nkc,
                            X, T, RR, hidx)
            if nbc != 1:
                for ab in range(nab):
                    for col in range(ncol):
                        Gb[ab, col] = 0.0
                for cij in range(nbc):
                    c = pp_C[bp, cij]
                    if c == 0.0:
                        continue
                    for ab in range(nab):
                        for col in range(ncol):
                            Gb[ab, col] += c * G[cij * nab + ab, col]
        else:
            if nbc != 1:
                # Gkb[kp, ab, k] = sum_cij C_bp[cij] Gk[kp, (cij, ab), k]
                for kp in range(nkp):
                    for ab in range(nab):
                        for k in range(nhk):
                            Gkb3[kp, ab, k] = 0.0
                    for cij in range(nbc):
                        c = pp_C[bp, cij]
                        if c == 0.0:
                            continue
                        for ab in range(nab):
                            for k in range(nhk):
                                Gkb3[kp, ab, k] += c * Gk3[kp, cij * nab + ab, k]
                Gsrc = Gkb3
            else:
                Gsrc = Gk3
            for ab in range(nab):
                for h in range(nh1):
                    Y[ab, h] = 0.0
            for kp in range(nkp):
                q = pp_ab[k0 + kp, 0] + pp_ab[k0 + kp, 1]
                alpha = p * q / (p + q)
                pref = _TWO_PI_2_5 / (p * q * math.sqrt(p + q))
                x = pp_P[bp, 0] - pp_P[k0 + kp, 0]
                y = pp_P[bp, 1] - pp_P[k0 + kp, 1]
                z = pp_P[bp, 2] - pp_P[k0 + kp, 2]
                # R_tuv(alpha, P - Q), as in _ket_r_contract
                T2 = alpha * (x * x + y * y + z * z)
                f = boys_top(L, T2)
                ex = 0.0
                if L > 0:
                    ex = math.exp(-T2)
                m2a = -2.0 * alpha
                pw = 1.0
                for _ in range(L):
                    pw *= m2a
                for n in range(L, -1, -1):
                    if n < L:
                        f = (2.0 * T2 * f + ex) * INV_ODD[n]
                    cur = n & 1
                    prv = 1 - cur
                    RR[cur, 0] = pw * f
                    for h in range(1, NHERM[L - n]):
                        d = HERM_DIR[h]
                        xd = x if d == 0 else (y if d == 1 else z)
                        RR[cur, h] = xd * RR[prv, HERM_M1[h]] + HERM_RC[h] * RR[prv, HERM_M2[h]]
                    pw /= m2a
                for h in range(nh1):
                    for k in range(nhk):
                        r = pref * RR[0, hidx[h, k]]
                        if r == 0.0:
                            continue
                        for ab in range(nab):
                            Y[ab, h] += r * Gsrc[kp, ab, k]
        E = pp_E[bp]
        for ia in range(nci):
            ax = CART_LXYZ[a0 + ia, 0]
            ay = CART_LXYZ[a0 + ia, 1]
            az = CART_LXYZ[a0 + ia, 2]
            fa = cb * CART_FAC[a0 + ia]
            for ib in range(ncj):
                bx = CART_LXYZ[b0 + ib, 0]
                by = CART_LXYZ[b0 + ib, 1]
                bz = CART_LXYZ[b0 + ib, 2]
                f = fa * CART_FAC[b0 + ib]
                ab = ia * ncj + ib
                grow = G[ab] if nbc == 1 else Gb[ab]
                tx = ax + bx
                ty = ay + by
                tz = az + bz
                # Hermite box of the derivative expansions: one index may
                # exceed the undifferentiated range by one
                for t in range(tx + 2):
                    ex = E[0, ax, bx, t]
                    dax = 2.0 * a * E[0, ax + 1, bx, t]
                    dbx = 2.0 * b * E[0, ax, bx + 1, t]
                    if ax > 0:
                        dax -= ax * E[0, ax - 1, bx, t]
                    if bx > 0:
                        dbx -= bx * E[0, ax, bx - 1, t]
                    for u in range(ty + 2):
                        if t > tx and u > ty:
                            break
                        ey = E[1, ay, by, u]
                        day = 2.0 * a * E[1, ay + 1, by, u]
                        dby = 2.0 * b * E[1, ay, by + 1, u]
                        if ay > 0:
                            day -= ay * E[1, ay - 1, by, u]
                        if by > 0:
                            dby -= by * E[1, ay, by - 1, u]
                        vmax = tz + 2 if (t <= tx and u <= ty) else tz + 1
                        for v in range(vmax):
                            h = HERM_IDX[t, u, v]
                            if use_x:
                                yv = 0.0
                                for col in range(ncol):
                                    yv += grow[col] * X[h, col]
                            else:
                                yv = Y[ab, h]
                            yv *= f
                            if yv == 0.0:
                                continue
                            ez = E[2, az, bz, v]
                            daz = 2.0 * a * E[2, az + 1, bz, v]
                            dbz = 2.0 * b * E[2, az, bz + 1, v]
                            if az > 0:
                                daz -= az * E[2, az - 1, bz, v]
                            if bz > 0:
                                dbz -= bz * E[2, az, bz - 1, v]
                            out[0, 0] += yv * dax * ey * ez
                            out[0, 1] += yv * ex * day * ez
                            out[0, 2] += yv * ex * ey * daz
                            out[1, 0] += yv * dbx * ey * ez
                            out[1, 1] += yv * ex * dby * ez
                            out[1, 2] += yv * ex * ey * dbz


@njit(cache=True, parallel=True)
def _eri_grad_kernel(pair_pp, pair_blocks, blk_l, blk_ncon, blk_atom, blk_ao, pp_ab, pp_P, pp_E,
                     pp_C, lmax, maxpp, maxrow, maxpph, Q, threshold, Dc, Dx, kfac, chunk_first,
                     grad_parts):
    npair = pair_blocks.shape[0]
    nchunk = chunk_first.shape[0]
    nx = Dx.shape[0]
    nc = (lmax + 1) * (lmax + 2) // 2
    for chunk in prange(nchunk):
        if chunk_first[chunk] >= npair:
            continue
        _blk, Ek, X, T, _W, RR, hidx = _scratch(lmax, maxpp, 1, 1)
        X = np.empty((NHERM[2 * lmax + 1], maxrow))
        Gk = np.empty(maxpph * maxrow)
        Gkb = np.empty(maxpp * nc * nc * NHERM[2 * lmax])
        Gb = np.empty((nc * nc, maxrow))
        Y = np.empty((nc * nc, NHERM[2 * lmax + 1]))
        G = np.empty((maxrow, maxrow))
        Gt = np.empty((maxrow, maxrow))
        gab = np.zeros((2, 3))
        gcd = np.zeros((2, 3))
        grad = grad_parts[chunk]
        for pb in range(chunk_first[chunk], npair, nchunk):
            si = pair_blocks[pb, 0]
            sj = pair_blocks[pb, 1]
            nci = CART_OFF[blk_l[si] + 1] - CART_OFF[blk_l[si]]
            ncj = CART_OFF[blk_l[sj] + 1] - CART_OFF[blk_l[sj]]
            nqi = blk_ncon[si]
            nqj = blk_ncon[sj]
            nab = nci * ncj
            aA = blk_atom[si]
            aB = blk_atom[sj]
            for pk in range(pb + 1):
                sk = pair_blocks[pk, 0]
                sl = pair_blocks[pk, 1]
                aC = blk_atom[sk]
                aD = blk_atom[sl]
                if aA == aB and aB == aC and aC == aD:
                    continue            # one-center quartet: translationally invariant
                if Q[pb] * Q[pk] == 0.0:
                    continue
                nck = CART_OFF[blk_l[sk] + 1] - CART_OFF[blk_l[sk]]
                ncl = CART_OFF[blk_l[sl] + 1] - CART_OFF[blk_l[sl]]
                nqk = blk_ncon[sk]
                nql = blk_ncon[sl]
                ncd = nck * ncl
                nrow = nqi * nqj * nab
                ncol = nqk * nql * ncd
                # 1/2 of E2 times the permutational degeneracy of the unique
                # block quartet (inside a diagonal pair I == J the loops below
                # already run over all ordered (contraction, component) pairs)
                deg = 0.5
                if si != sj:
                    deg *= 2.0
                if sk != sl:
                    deg *= 2.0
                if pb != pk:
                    deg *= 2.0
                # G = deg [Dc_mn Dc_ls - k/2 sum_s (Ds_ml Ds_ns + Ds_ms Ds_nl)]
                gmax = 0.0
                for qi in range(nqi):
                    for a in range(nci):
                        m = blk_ao[si] + qi * nci + a
                        for qj in range(nqj):
                            for b in range(ncj):
                                n = blk_ao[sj] + qj * ncj + b
                                r = (qi * nqj + qj) * nab + a * ncj + b
                                for qk in range(nqk):
                                    for c in range(nck):
                                        lam = blk_ao[sk] + qk * nck + c
                                        for ql in range(nql):
                                            for d in range(ncl):
                                                sig = blk_ao[sl] + ql * ncl + d
                                                s = (qk * nql + ql) * ncd + c * ncl + d
                                                v = Dc[m, n] * Dc[lam, sig]
                                                for x in range(nx):
                                                    v -= 0.5 * kfac * (Dx[x, m, lam] * Dx[x, n, sig]
                                                                       + Dx[x, m, sig] * Dx[x, n, lam])
                                                v *= deg
                                                G[r, s] = v
                                                Gt[s, r] = v
                                                gmax = max(gmax, abs(v))
                if Q[pb] * Q[pk] * gmax < threshold:
                    continue
                need_bra = aA != aB
                need_ket = aC != aD
                if not need_bra and not need_ket:
                    # A == B, C == D: one pass, on the pair with more columns
                    if nrow >= ncol:
                        _grad_pass(pb, pk, G, pair_pp, pair_blocks, blk_l, blk_ncon, pp_ab, pp_P,
                                   pp_E, pp_C, Ek, Gk, Gkb, Gb, Y, X, T, RR, hidx, gab)
                        for d in range(3):
                            g = gab[0, d] + gab[1, d]
                            grad[aA, d] += g
                            grad[aC, d] -= g
                    else:
                        _grad_pass(pk, pb, Gt, pair_pp, pair_blocks, blk_l, blk_ncon, pp_ab, pp_P,
                                   pp_E, pp_C, Ek, Gk, Gkb, Gb, Y, X, T, RR, hidx, gcd)
                        for d in range(3):
                            g = gcd[0, d] + gcd[1, d]
                            grad[aC, d] += g
                            grad[aA, d] -= g
                    continue
                if need_bra:
                    _grad_pass(pb, pk, G, pair_pp, pair_blocks, blk_l, blk_ncon, pp_ab, pp_P,
                               pp_E, pp_C, Ek, Gk, Gkb, Gb, Y, X, T, RR, hidx, gab)
                if need_ket:
                    _grad_pass(pk, pb, Gt, pair_pp, pair_blocks, blk_l, blk_ncon, pp_ab, pp_P,
                               pp_E, pp_C, Ek, Gk, Gkb, Gb, Y, X, T, RR, hidx, gcd)
                for d in range(3):
                    if need_bra and need_ket:
                        grad[aA, d] += gab[0, d]
                        grad[aB, d] += gab[1, d]
                        grad[aC, d] += gcd[0, d]
                        grad[aD, d] += gcd[1, d]
                    elif need_bra:          # C == D: their sum by translational invariance
                        grad[aA, d] += gab[0, d]
                        grad[aB, d] += gab[1, d]
                        grad[aC, d] -= gab[0, d] + gab[1, d]
                    else:                   # A == B
                        grad[aC, d] += gcd[0, d]
                        grad[aD, d] += gcd[1, d]
                        grad[aA, d] -= gcd[0, d] + gcd[1, d]


# ============================================================ public API

def _shell_args(basis: BasisSet):
    return (basis.shell_atom, basis.shell_l, basis.shell_prim, basis.shell_ao,
            basis.prim_exp, basis.prim_coef, basis.positions)


def _overlap_type(basis: BasisSet, origin=None):
    _check_lmax(basis)
    n = basis.nao
    S = np.zeros((n, n))
    T = np.zeros((n, n))
    M = np.zeros((3, n, n))
    o = np.zeros(3) if origin is None else np.asarray(origin, dtype=float).reshape(3)
    _overlap_type_kernel(*_shell_args(basis), o, S, T, M)
    return S, T, M


def overlap_matrix(basis: BasisSet) -> np.ndarray:
    """S_mn = <m|n>, (nao, nao)."""
    return _overlap_type(basis)[0]


def kinetic_matrix(basis: BasisSet) -> np.ndarray:
    """T_mn = <m| -1/2 nabla^2 |n> (hartree), (nao, nao)."""
    return _overlap_type(basis)[1]


def dipole_integrals(basis: BasisSet, origin: Sequence[float] | None = None) -> np.ndarray:
    """<m| r - O |n> (bohr), shape (3, nao, nao); O defaults to the coordinate origin."""
    return _overlap_type(basis, origin)[2]


def nuclear_attraction_matrix(basis: BasisSet, charges: np.ndarray | None = None) -> np.ndarray:
    """V_mn = -sum_A Z_A <m| 1/|r - R_A| |n> over the atoms of ``basis`` (hartree)."""
    _check_lmax(basis)
    q = _charges(basis, charges)
    V = np.zeros((basis.nao, basis.nao))
    _nuclear_kernel(*_shell_args(basis), q, basis.positions, V)
    return V


def one_electron_matrices(basis: BasisSet, charges: np.ndarray | None = None
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(S, T, V) in one call."""
    S, T, _ = _overlap_type(basis)
    return S, T, nuclear_attraction_matrix(basis, charges)


def nuclear_repulsion_energy(charges: np.ndarray, positions: np.ndarray) -> float:
    """E_nuc = sum_{A<B} Z_A Z_B / R_AB (hartree)."""
    q = np.asarray(charges, dtype=float).reshape(-1)
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    i, j = np.triu_indices(len(q), k=1)
    return float(np.sum(q[i] * q[j] / np.linalg.norm(x[i] - x[j], axis=1)))


def nuclear_repulsion_gradient(charges: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """dE_nuc/dR_A = -sum_B Z_A Z_B (R_A - R_B) / R_AB^3, (natm, 3)."""
    q = np.asarray(charges, dtype=float).reshape(-1)
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    i, j = np.triu_indices(len(q), k=1)
    d = x[i] - x[j]
    r = np.linalg.norm(d, axis=1)
    pair = -(q[i] * q[j] / r ** 3)[:, None] * d
    g = np.zeros_like(x)
    np.add.at(g, i, pair)
    np.add.at(g, j, -pair)
    return g


def nuclear_dipole(charges: np.ndarray, positions: np.ndarray,
                   origin: Sequence[float] | None = None) -> np.ndarray:
    """sum_A Z_A (R_A - O), (3,) in e * bohr."""
    q = np.asarray(charges, dtype=float).reshape(-1)
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    o = np.zeros(3) if origin is None else np.asarray(origin, dtype=float).reshape(3)
    return q @ (x - o)


def overlap_gradient(basis: BasisSet, W: np.ndarray) -> np.ndarray:
    """d/dR [ sum_mn W_mn S_mn ], (natm, 3)."""
    _check_lmax(basis)
    g = np.zeros((basis.natm, 3))
    W = _sym(W, basis.nao, "W")
    _overlap_kinetic_grad_kernel(*_shell_args(basis), W, np.zeros_like(W), g)
    return g


def kinetic_gradient(basis: BasisSet, D: np.ndarray) -> np.ndarray:
    """d/dR [ sum_mn D_mn T_mn ], (natm, 3)."""
    _check_lmax(basis)
    g = np.zeros((basis.natm, 3))
    D = _sym(D, basis.nao, "D")
    _overlap_kinetic_grad_kernel(*_shell_args(basis), np.zeros_like(D), D, g)
    return g


def nuclear_attraction_gradient(basis: BasisSet, D: np.ndarray,
                                charges: np.ndarray | None = None) -> np.ndarray:
    """
    d/dR [ sum_mn D_mn V_mn ], (natm, 3): basis-function centers plus the
    Hellmann-Feynman term from moving the nuclei (operator centers).
    """
    _check_lmax(basis)
    q = _charges(basis, charges)
    g = np.zeros((basis.natm, 3))
    _nuclear_grad_kernel(*_shell_args(basis), q, _sym(D, basis.nao, "D"), g)
    return g


def one_electron_gradient(basis: BasisSet, D: np.ndarray, W: np.ndarray,
                          charges: np.ndarray | None = None) -> np.ndarray:
    """
    d/dR [ sum D_mn (T + V)_mn ] - d/dR [ sum W_mn S_mn ], (natm, 3); W is the
    energy-weighted density (convention in the module docstring).
    """
    _check_lmax(basis)
    D = _sym(D, basis.nao, "D")
    W = _sym(W, basis.nao, "W")
    g = np.zeros((basis.natm, 3))
    _overlap_kinetic_grad_kernel(*_shell_args(basis), -W, D, g)
    _nuclear_grad_kernel(*_shell_args(basis), _charges(basis, charges), D, g)
    return g


def two_electron_gradient(basis: BasisSet, D_coulomb: np.ndarray,
                          D_exchange_list: Sequence[np.ndarray], k_factor: float,
                          schwarz_threshold: float = DEFAULT_SCHWARZ_GRAD) -> np.ndarray:
    """
    d/dR of E2 = 1/2 sum Dc_mn Dc_ls (mn|ls) - k/2 sum_s sum Ds_ml Ds_ns (mn|ls)
    at fixed densities, (natm, 3). Quartets with Q_ab Q_cd max|Gamma| below
    ``schwarz_threshold`` are skipped (Gamma = the quartet's two-particle
    density); see DEFAULT_SCHWARZ_GRAD for the resulting error.
    """
    _check_lmax(basis)
    n = basis.nao
    Dc = _sym(D_coulomb, n, "D_coulomb")
    Dx = np.zeros((len(D_exchange_list), n, n))
    for s, Ds in enumerate(D_exchange_list):
        Dx[s] = _sym(Ds, n, "D_exchange_list entry")
    pr = _pairs(basis)
    Q = schwarz_bounds(basis)
    parts = np.zeros((NCHUNK, basis.natm, 3))
    _eri_grad_kernel(pr.pair_pp, pr.pair_blocks, basis.blk_l, basis.blk_ncon, basis.blk_atom,
                     basis.blk_ao, pr.pp_ab, pr.pp_P, pr.pp_E, pr.pp_C, basis.lmax,
                     pr.maxpp, pr.maxrow, pr.maxpph, Q, float(schwarz_threshold), Dc, Dx,
                     float(k_factor), CHUNK_FIRST, parts)
    return parts.sum(axis=0)
