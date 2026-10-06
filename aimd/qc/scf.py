"""
Restricted (RHF) and unrestricted (UHF) Hartree-Fock SCF on the native
integrals of :mod:`aimd.qc.integrals`.

Units: Hartree atomic units throughout (bohr, hartree, e * bohr for dipoles).
AO matrices refer to the unit-normalized Cartesian AOs of
:mod:`aimd.qc.basis`.

Equations
---------
Closed-shell Roothaan-Hall (Roothaan, Rev. Mod. Phys. 23, 69 (1951); Hall,
Proc. R. Soc. A 205, 541 (1951)) and open-shell Pople-Nesbet UHF (Pople &
Nesbet, J. Chem. Phys. 22, 571 (1954)) equations F_s C_s = S C_s eps_s, with
spin densities D_s = C_s,occ C_s,occ^T (occupation 1 per spin orbital) and

    F_s = h + J[D_a + D_b] - K[D_s],      E_el = 1/2 sum_s tr D_s (h + F_s),

J_mn[D] = sum (mn|ls) D_ls, K_mn[D] = sum (ml|ns) D_ls, h = T + V. RHF is the
special case D_a = D_b = P/2 (P = total density), i.e. F = h + J[P] - K[P]/2
and E_el = 1/2 tr P (h + F). Internally both run through one loop over
``nspin`` spin densities (1 for RHF, 2 for UHF) with a spin degeneracy factor
g = 2 / nspin. The ERI tensor is stored (``eri_tensor``), so every iteration
is an O(nao^4) contraction; nao is limited by ``SCFOptions.max_nao``.

Orthogonalization
-----------------
The generalized eigenproblem is solved in an orthonormal basis X (X^T S X =
1). Overlap eigenvalues s_k below ``lindep_threshold`` (default 1e-7, the
Psi4 S_TOLERANCE convention: s^-1/2 then amplifies rounding errors by < ~3e3)
signal near-linear dependence. ``orthogonalization``:
  "auto"       symmetric (Loewdin) X = U s^-1/2 U^T if min s >= threshold,
               otherwise canonical;
  "canonical"  X = U_k s_k^-1/2 over the kept eigenvectors only, so nmo =
               nao - (number of removed combinations) (Loewdin, Adv. Quantum
               Chem. 5, 185 (1970); Szabo & Ostlund sec. 3.4.5);
  "symmetric"  always Loewdin; raises if min s < threshold.
Both span the same space when nothing is removed, so energies are identical.
In MD, a change in the number of removed combinations between geometries
would be a (tiny) discontinuity; standard basis sets without diffuse
functions never come near the threshold. With combinations removed, the
variational space is a projected one that moves with the nuclei, which the
standard energy-weighted-density (Pulay) gradient term does not account for:
measured on water / 6-31G plus diffuse s, p shells (two combinations
removed), analytic gradients differ from finite differences by 1.7e-6
Eh/bohr, in PySCF exactly as here (1.5e-9 without removal).

Initial guesses
---------------
  "sad"   (default) superposition of atomic densities (Almloef, Faegri &
          Korsell, J. Comput. Chem. 3, 385 (1982); Van Lenthe et al., J.
          Comput. Chem. 27, 926 (2006)): a block-diagonal density from
          spherically averaged, fractionally occupied, spin-restricted HF
          calculations on each neutral atom *in the molecular basis*. Each
          atomic SCF fills the ground-state configuration shell by shell
          per angular momentum (the electron counts per l follow the
          Madelung order; orbitals of one l are recognized by their 2l+1
          degeneracy, which a spherical density preserves exactly), and the
          partly filled shell is occupied evenly so the density stays
          spherical (the scheme of PySCF's ``scf.atom_hf``; H, a one-
          electron atom, gets its exact one-electron ground state instead).
          Atomic blocks depend only on the element and its shells, so they
          are computed once per process and cached. The guess is not
          idempotent and not normalized to the molecular charge; the first
          diagonalization fixes both. For UHF both spin densities start as
          P_SAD / 2: spin polarization appears at the first diagonalization
          when n_alpha != n_beta, but a singlet UHF stays spin-restricted
          (no symmetry breaking is attempted).
  "core"  eigenvectors of the core Hamiltonian h.
  "gwh"   generalized Wolfsberg-Helmholz, F_mn = K/2 S_mn (h_mm + h_nn),
          K = 1.75 (Wolfsberg & Helmholz, J. Chem. Phys. 20, 837 (1952)).
  array   an explicit AO density in the density-guess protocol of
          :mod:`aimd.backends.base`: total density (nao, nao) for RHF,
          [alpha, beta] (2, nao, nao) for UHF. For convenience a UHF run also
          accepts a total density (split evenly between the spins) and an RHF
          run a (2, nao, nao) pair (summed). The density may come from a
          different geometry (it is used as-is to build the first Fock
          matrix) and need not be idempotent (XL-BOMD auxiliary densities).
  SCFResult  its converged spin densities.

Iterations and convergence
--------------------------
Iteration k builds F_k from D_k, evaluates E_k = E[D_k] and the orthonormal-
basis commutator e_s = X^T (F_s D_s S - S D_s F_s) X for each spin density
D_s (also for RHF, D_s = P/2, so RHF and UHF thresholds mean the same: for an
idempotent D_s, e_s is unitarily equivalent to the occupied-virtual block of
F_s, i.e. the orbital gradient). The SCF has converged when

    |E_k - E_{k-1}| < conv_energy  (default 1e-10 Eh)  and
    max_s max|e_s|  < conv_commutator  (default 1e-7),

otherwise F_k is extrapolated by Pulay DIIS (Pulay, Chem. Phys. Lett. 73,
393 (1980); J. Comput. Chem. 3, 556 (1982)) over the last ``diis_space``
Fock matrices, with the alpha and beta error vectors concatenated for UHF
(one set of coefficients; see :class:`DIIS` for the numerically stable
solve). The extrapolated F is diagonalized and the lowest orbitals of each
spin are occupied (aufbau) to give D_{k+1}. DIIS is reset to the latest
vector when the commutator norm has not improved on its best value for
``diis_restart`` iterations (stalled extrapolation).
For difficult cases, a level shift sigma adds sigma (1 - D_s) to the
orthonormal-basis Fock matrix (virtual orbitals raised by sigma; Saunders &
Hillier, Int. J. Quantum Chem. 7, 699 (1973)) and ``damping`` mixes that
fraction of the old density into the new one; both act only while the
commutator norm exceeds ``stabilize_until`` and leave the converged solution
unchanged (they vanish at a fixed point where [F, D] = 0).

Difficult cases: water with both bonds stretched x1.8 converges with the
defaults from every guess. Far from equilibrium RHF has several solutions
(water x2.5-x3: the one reached depends on the guess, as in PySCF), and for
some distorted open-shell ions DIIS from the SAD guess oscillates between
hole configurations (ethanol+ / 6-31G* with 0.2 bohr random displacements;
PySCF's DIIS fails there too). Remedies, in order: ``guess="gwh"`` (24
iterations there) or ``"core"``; then ``diis=False, level_shift=1.0,
stabilize_until=1e-8`` (slow but steady: 277 iterations there). Level
shifting and damping on top of DIIS did not help in that case. There is no
second-order solver or SCF stability analysis; in MD the previous density is
the guess, which keeps the SCF on the same state from step to step.

The returned state is that of the last iteration: ``density`` = D_k,
``energy`` = E[D_k], ``fock`` = F[D_k] (all mutually consistent, which the
gradient needs), and the MOs are the canonical orbitals of F_k (one extra
diagonalization, no extra Fock build); they reproduce D_k up to
O(commutator). A run that reaches ``max_iter`` returns that state with
``converged = False`` and emits an :class:`SCFConvergenceWarning` (unless
``warn_unconverged=False``); it never raises for non-convergence.

Outputs for analytic gradients (the next step)
----------------------------------------------
``SCFResult.basis`` is the BasisSet at the run's geometry (its cached shell-
pair data and Schwarz bounds are reused by the gradient code), ``density``
the total density P, ``energy_weighted_density`` W_mn = sum_i n_i eps_i C_mi
C_ni summed over occupied spin orbitals (RHF n_i = 2: W = 1/2 P F P; UHF W =
W_a + W_b) as :func:`aimd.qc.integrals.one_electron_gradient` expects, and
``coulomb_exchange_densities()`` the (D_coulomb, D_exchange_list, k_factor)
arguments of :func:`aimd.qc.integrals.two_electron_gradient`.

Performance
-----------
Per geometry: one-electron integrals and the ERI tensor (O(nao^4) memory),
then O(nao^4) J/K contractions per iteration (numba threads). The SCF limits
BLAS to one thread while it runs (``blas_threads``, :mod:`aimd.qc.threads`),
which removes thread-pool contention with the numba kernels (2.5x faster
SCF for ethanol/6-31G*). Warm timings on 4 cores, median over fresh
geometries (default guess / previous density from 0.01 bohr away):
water/STO-3G (nao 7) 8 / 7 iterations, ~3 ms; water/6-31G* (nao 19) 11 / 9
iterations, ~7 / 6 ms; ethanol/6-31G* (nao 57) 12 / 9 iterations, ~190 /
170 ms, of which ~130 ms integrals and ~5 ms per iteration (PySCF 2.14,
same basis: 44, 53 and 261 ms). ``solver.clear_cache()`` releases the stored
ERI tensor (8 nao^4 bytes).

Properties
----------
Mulliken charges q_A = Z_A - sum_{m on A} (P S)_mm (Mulliken, J. Chem. Phys.
23, 1833 (1955)); dipole mu = -tr(P r) + sum_A Z_A R_A about the coordinate
origin (origin-dependent for ions); for UHF <S^2> = S_z (S_z + 1) + n_beta -
tr(D_a S D_b S) (Szabo & Ostlund eq. 2.271).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field, fields, replace
from typing import Any, Sequence

import numpy as np

from aimd.elements import normalize_symbol
from aimd.qc import integrals as qcint
from aimd.qc.basis import BasisSet, build_basis
from aimd.qc.threads import limit_blas_threads

LINDEP_THRESHOLD = 1e-7
GWH_K = 1.75
_GUESSES = ("sad", "core", "gwh")
_ORTHO = ("auto", "symmetric", "canonical")
_REFERENCES = {"rhf": "rhf", "restricted": "rhf", "uhf": "uhf", "unrestricted": "uhf"}


class SCFConvergenceWarning(RuntimeWarning):
    """Emitted when an SCF stops at ``max_iter`` without converging."""


# ================================================================ options

@dataclass(frozen=True)
class SCFOptions:
    """
    SCF settings (see the module docstring for their meaning).

    conv_energy       |E_k - E_{k-1}| threshold (Eh)
    conv_commutator   max |X^T (F D_s S - S D_s F) X| threshold (Eh)
    max_iter          Fock builds before giving up (result flagged unconverged)
    guess             default initial guess: "sad", "core" or "gwh"
    diis              Pulay DIIS on/off (off: plain Roothaan iterations)
    diis_space        number of stored Fock / error pairs
    diis_restart      reset DIIS after this many iterations without a new
                      best commutator norm (0 disables)
    level_shift       virtual-orbital shift (Eh) while the commutator norm
                      exceeds ``stabilize_until``
    damping           fraction of the old density kept, same condition
    stabilize_until   commutator norm below which shift and damping stop
    orthogonalization "auto", "symmetric" or "canonical"
    lindep_threshold  overlap eigenvalues below this are linear dependencies
    schwarz_threshold ERI screening threshold (strict bound on dropped ints)
    max_nao           refuse to store the ERI tensor beyond this many AOs
    warn_unconverged  emit SCFConvergenceWarning for unconverged runs
    blas_threads      BLAS threads during a run (None: leave BLAS alone);
                      see :mod:`aimd.qc.threads` for why the default is 1
    """
    conv_energy: float = 1e-10
    conv_commutator: float = 1e-7
    max_iter: int = 100
    guess: str = "sad"
    diis: bool = True
    diis_space: int = 8
    diis_restart: int = 12
    level_shift: float = 0.0
    damping: float = 0.0
    stabilize_until: float = 1e-3
    orthogonalization: str = "auto"
    lindep_threshold: float = LINDEP_THRESHOLD
    schwarz_threshold: float = qcint.DEFAULT_SCHWARZ
    max_nao: int = qcint.MAX_NAO_ERI
    warn_unconverged: bool = True
    blas_threads: int | None = 1

    def __post_init__(self) -> None:
        if not (self.conv_energy > 0.0 and self.conv_commutator > 0.0):
            raise ValueError("SCF convergence thresholds must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        if self.guess not in _GUESSES:
            raise ValueError(f"unknown guess {self.guess!r}; use one of {_GUESSES}")
        if self.diis_space < 2:
            raise ValueError("diis_space must be >= 2")
        if self.diis_restart < 0:
            raise ValueError("diis_restart must be >= 0")
        if self.level_shift < 0.0:
            raise ValueError("level_shift must be >= 0")
        if not 0.0 <= self.damping < 1.0:
            raise ValueError("damping must be in [0, 1)")
        if self.orthogonalization not in _ORTHO:
            raise ValueError(f"unknown orthogonalization {self.orthogonalization!r}; use one of {_ORTHO}")
        if not self.lindep_threshold > 0.0:
            raise ValueError("lindep_threshold must be positive")
        if self.blas_threads is not None and self.blas_threads < 1:
            raise ValueError("blas_threads must be >= 1 or None")


def electron_counts(nuclear_charge: int, charge: int, multiplicity: int,
                    reference: str = "uhf") -> tuple[int, int]:
    """
    (n_alpha, n_beta) for a molecule with total nuclear charge ``nuclear_charge``
    (sum of Z), net ``charge`` and spin ``multiplicity`` = 2S + 1, with
    n_alpha - n_beta = 2S. Raises ValueError for impossible combinations and,
    for ``reference="rhf"``, anything but a closed-shell singlet.
    """
    n = int(nuclear_charge) - int(charge)
    mult = int(multiplicity)
    if n <= 0:
        raise ValueError(f"charge {charge} leaves {n} electrons; need at least one")
    if mult < 1:
        raise ValueError(f"multiplicity must be >= 1, got {mult}")
    if (n + mult - 1) % 2:
        raise ValueError(
            f"multiplicity {mult} is impossible with {n} electrons "
            f"({'even' if n % 2 == 0 else 'odd'} electron counts need "
            f"{'odd' if n % 2 == 0 else 'even'} multiplicities)")
    if mult - 1 > n:
        raise ValueError(f"multiplicity {mult} needs at least {mult - 1} electrons, have {n}")
    na, nb = (n + mult - 1) // 2, (n - mult + 1) // 2
    ref = str(reference).strip().lower()
    if ref not in _REFERENCES:
        raise ValueError(f"unknown reference {reference!r}; use 'rhf' or 'uhf'")
    if _REFERENCES[ref] == "rhf" and na != nb:
        raise ValueError(
            f"RHF needs a closed-shell singlet; {n} electrons with multiplicity "
            f"{mult} is open-shell (use UHF)")
    return na, nb


# ======================================================= orthogonalization

def orthogonalizer(S: np.ndarray, method: str = "auto",
                   threshold: float = LINDEP_THRESHOLD) -> tuple[np.ndarray, int, float]:
    """
    X (nao, nmo) with X^T S X = 1, the number of removed near-linearly-
    dependent combinations, and the smallest overlap eigenvalue.
    """
    if method not in _ORTHO:
        raise ValueError(f"unknown orthogonalization {method!r}; use one of {_ORTHO}")
    s, U = np.linalg.eigh(S)
    smin = float(s[0])
    keep = s >= threshold
    if method == "symmetric" or (method == "auto" and keep.all()):
        if not keep.all():
            raise ValueError(
                f"overlap eigenvalue {smin:.3e} < lindep_threshold {threshold:.1e}: "
                "the basis is nearly linearly dependent; use canonical orthogonalization")
        return (U * s ** -0.5) @ U.T, 0, smin
    if not keep.any():
        raise ValueError("every overlap eigenvalue is below lindep_threshold")
    return U[:, keep] * s[keep] ** -0.5, int((~keep).sum()), smin


# ============================================================ integrals

@dataclass(frozen=True, eq=False)
class SCFIntegrals:
    """Everything the SCF needs at one geometry (all AO quantities, a.u.)."""
    basis: BasisSet
    S: np.ndarray            # overlap (nao, nao)
    H: np.ndarray            # core Hamiltonian T + V
    eri: np.ndarray          # (mn|ls), (nao,)*4
    X: np.ndarray            # orthogonalizer (nao, nmo)
    n_removed: int           # linear dependencies removed
    min_overlap_eigenvalue: float
    nuclear_repulsion: float

    @property
    def nao(self) -> int:
        return self.basis.nao

    @property
    def nmo(self) -> int:
        return self.X.shape[1]


def build_integrals(basis: BasisSet, options: SCFOptions | None = None) -> SCFIntegrals:
    """S, h, the ERI tensor and the orthogonalizer for ``basis`` at its geometry."""
    opt = options or SCFOptions()
    S, T, V = qcint.one_electron_matrices(basis)
    eri = qcint.eri_tensor(basis, schwarz_threshold=opt.schwarz_threshold, max_nao=opt.max_nao)
    X, n_removed, smin = orthogonalizer(S, opt.orthogonalization, opt.lindep_threshold)
    enuc = qcint.nuclear_repulsion_energy(basis.nuclear_charges, basis.positions)
    return SCFIntegrals(basis, S, T + V, eri, X, n_removed, smin, enuc)


# ================================================================ guesses

def _occupied_density(F: np.ndarray, X: np.ndarray, nocc: int) -> np.ndarray:
    """Spin density of the lowest ``nocc`` eigenvectors of F (AO) in the X basis."""
    _, Cp = np.linalg.eigh(X.T @ F @ X)
    C = X @ Cp[:, :nocc]
    return C @ C.T


def core_guess_matrix(ints_: SCFIntegrals) -> np.ndarray:
    """Core-Hamiltonian guess Fock matrix (= h)."""
    return ints_.H


def gwh_guess_matrix(ints_: SCFIntegrals, k: float = GWH_K) -> np.ndarray:
    """Generalized Wolfsberg-Helmholz F_mn = k/2 S_mn (h_mm + h_nn), diagonal h_mm."""
    h = np.diag(ints_.H)
    F = 0.5 * k * ints_.S * (h[:, None] + h[None, :])
    np.fill_diagonal(F, h)
    return F


# Madelung (n + l, n) filling order, used for the per-l electron counts of SAD.
_MADELUNG = sorted(((n, l) for n in range(1, 8) for l in range(n)), key=lambda t: (t[0] + t[1], t[0]))


def ground_state_l_counts(Z: int) -> list[int]:
    """
    Electrons per angular momentum l (s, p, d, f) of a neutral atom filled in
    Madelung order (the few transition-metal exceptions, e.g. Cr, Cu, are
    irrelevant for a spherically averaged guess and ignored).
    """
    counts = [0, 0, 0, 0]
    left = int(Z)
    for _n, l in _MADELUNG:
        if left <= 0:
            break
        k = min(left, 2 * (2 * l + 1))
        counts[l] += k
        left -= k
    return counts


def _degenerate_groups(eps: np.ndarray, tol: float) -> list[tuple[int, int]]:
    groups = []
    i = 0
    while i < len(eps):
        j = i + 1
        while j < len(eps) and eps[j] - eps[j - 1] < tol:
            j += 1
        groups.append((i, j))
        i = j
    return groups


def _atomic_occupations(eps: np.ndarray, l_counts: Sequence[int], tol: float = 1e-6) -> np.ndarray:
    """
    Spin-summed occupations (0..2) for ascending atomic orbital energies
    ``eps``: degenerate groups of size 2l+1 are shells of angular momentum l
    and are filled lowest-first with l_counts[l] electrons, evenly within the
    partly filled shell. Falls back to fractional aufbau over all groups if a
    group size is not 1, 3, 5, 7 or 9 (accidental degeneracy) or a shell type
    runs out.
    """
    groups = _degenerate_groups(eps, tol)
    occ = np.zeros(len(eps))
    left = list(l_counts) + [0] * (5 - len(l_counts))
    ok = True
    for i, j in groups:
        g = j - i
        if g % 2 == 0 or g > 9:
            ok = False
            break
        l = (g - 1) // 2
        k = min(left[l], 2 * g)
        occ[i:j] = k / g
        left[l] -= k
    if ok and not any(left):
        return occ
    occ[:] = 0.0                      # fractional aufbau fallback
    n = float(sum(l_counts))
    for i, j in groups:
        k = min(n, 2.0 * (j - i))
        occ[i:j] = k / (j - i)
        n -= k
    return occ


_ATOMIC_CACHE: dict[tuple, tuple[np.ndarray, float, bool]] = {}


def atomic_hf(basis_atom: BasisSet, max_iter: int = 200) -> tuple[np.ndarray, float, bool]:
    """
    (spin-summed density (nao_A, nao_A), energy (Eh), converged) of the
    neutral atom described by the one-atom basis ``basis_atom``: spherically
    averaged, fractional-occupation, spin-restricted HF (module docstring;
    the scheme of PySCF's ``scf.atom_hf``). A one-electron atom (H) gets the
    exact one-electron ground state in the basis instead, free of the
    self-interaction that the fractional-occupation formula would add.
    Results are cached by element and shell data.
    """
    if basis_atom.natm != 1:
        raise ValueError("atomic_hf needs a one-atom basis")
    key = (int(basis_atom.atomic_numbers[0]), basis_atom.shell_l.tobytes(),
           basis_atom.shell_prim.tobytes(), basis_atom.prim_exp.tobytes(),
           basis_atom.prim_coef.tobytes())
    hit = _ATOMIC_CACHE.get(key)
    if hit is not None:
        return hit[0].copy(), hit[1], hit[2]
    Z = int(basis_atom.atomic_numbers[0])
    l_counts = ground_state_l_counts(Z)
    S, T, V = qcint.one_electron_matrices(basis_atom)
    H = T + V
    X, _, _ = orthogonalizer(S, "canonical", LINDEP_THRESHOLD)
    if Z == 1:
        eps, Cp = np.linalg.eigh(X.T @ H @ X)
        c = X @ Cp[:, 0]
        result = (np.outer(c, c), float(eps[0]), True)
        _ATOMIC_CACHE[key] = result
        return result[0].copy(), result[1], True
    eri = qcint.eri_tensor(basis_atom)

    def density(Fo: np.ndarray) -> np.ndarray:
        eps, Cp = np.linalg.eigh(Fo)
        C = X @ Cp
        return (C * _atomic_occupations(eps, l_counts)) @ C.T

    D = density(X.T @ H @ X)
    diis = DIIS(8)
    e_old = np.inf
    converged = False
    for _ in range(max_iter):
        J, K = qcint.jk_from_eri(eri, D)
        F = H + J - 0.5 * K
        F = 0.5 * (F + F.T)
        e = 0.5 * float(np.vdot(D, H + F))
        err = X.T @ (F @ D @ S - S @ D @ F) @ X
        if abs(e - e_old) < 1e-11 and np.abs(err).max() < 1e-8:
            converged = True
            break
        e_old = e
        diis.push((X.T @ F @ X)[None], err[None])
        D = density(diis.extrapolate()[0])
    result = (0.5 * (D + D.T), e, converged)
    _ATOMIC_CACHE[key] = result
    return result[0].copy(), e, converged


def atomic_density(basis_atom: BasisSet) -> np.ndarray:
    """Spin-summed SAD density block of the neutral atom (see :func:`atomic_hf`)."""
    return atomic_hf(basis_atom)[0]


def atom_basis(basis: BasisSet, atom: int) -> BasisSet:
    """The shells of atom ``atom`` of ``basis`` as a one-atom basis at the origin."""
    sh = np.nonzero(basis.shell_atom == atom)[0]
    k0 = [basis.shell_prim[s] for s in sh]
    k1 = [basis.shell_prim[s + 1] for s in sh]
    prim = np.concatenate([np.arange(a, b) for a, b in zip(k0, k1)]) if len(sh) else np.zeros(0, int)
    shell_prim = np.concatenate([[0], np.cumsum([b - a for a, b in zip(k0, k1)])])
    return BasisSet(basis.name, [basis.symbols[atom]], np.zeros((1, 3)),
                    np.zeros(len(sh), dtype=np.int64), basis.shell_l[sh], shell_prim,
                    basis.prim_exp[prim], basis.prim_coef[prim])


def sad_density(basis: BasisSet) -> np.ndarray:
    """Superposition-of-atomic-densities guess (total density, (nao, nao))."""
    P = np.zeros((basis.nao, basis.nao))
    for a in range(basis.natm):
        i, j = basis.atom_ao[a]
        if j > i:
            P[i:j, i:j] = atomic_density(atom_basis(basis, a))
    return P


# =================================================================== DIIS

class DIIS:
    """
    Pulay DIIS: F = sum_i c_i F_i with the c_i minimizing |sum_i c_i e_i|
    subject to sum_i c_i = 1. Vectors are stacks (nspin, nmo, nmo); the norm
    sums over spins, so alpha and beta share coefficients.

    Eliminating the constraint with the newest vector m as anchor turns this
    into the linear least-squares problem min |e_m + sum_{i<m} c_i (e_i - e_m)|
    (c_m = 1 - sum_{i<m} c_i), solved by SVD on the error vectors themselves
    rather than through the normal equations B_ij = <e_i|e_j> (whose condition
    number is the square). Linearly dependent histories (e.g. repeated
    vectors) get the minimum-norm solution; if the coefficients still blow up
    (|c| > ``max_coefficient``), the oldest vectors are dropped.
    """

    def __init__(self, space: int = 8, max_coefficient: float = 1e4) -> None:
        self.space = int(space)
        self.max_coefficient = float(max_coefficient)
        self.focks: list[np.ndarray] = []
        self.errors: list[np.ndarray] = []

    def __len__(self) -> int:
        return len(self.focks)

    def reset(self) -> None:
        self.focks.clear()
        self.errors.clear()

    def push(self, fock: np.ndarray, error: np.ndarray) -> None:
        self.focks.append(np.array(fock, dtype=float))
        self.errors.append(np.array(error, dtype=float))
        while len(self.focks) > self.space:
            self.focks.pop(0)
            self.errors.pop(0)

    def coefficients(self) -> np.ndarray:
        """DIIS coefficients of the stored vectors (oldest first)."""
        last = self.errors[-1].ravel()
        if len(self.errors) == 1:
            return np.ones(1)
        delta = np.stack([e.ravel() - last for e in self.errors[:-1]], axis=1)
        x = np.linalg.lstsq(delta, -last, rcond=None)[0]
        return np.append(x, 1.0 - x.sum())

    def extrapolate(self) -> np.ndarray:
        """Extrapolated Fock stack."""
        while len(self.focks) > 1:
            c = self.coefficients()
            if np.all(np.isfinite(c)) and np.abs(c).max() <= self.max_coefficient:
                return sum(ci * Fi for ci, Fi in zip(c, self.focks))
            self.focks.pop(0)
            self.errors.pop(0)
        return self.focks[-1].copy()


# ================================================================ results

@dataclass(eq=False)
class SCFResult:
    """
    Converged (or flagged unconverged) SCF state at one geometry. AO arrays
    use the unit-normalized Cartesian AOs; energies in Eh.

    RHF: mo_energy (nmo,), mo_coeff (nao, nmo), mo_occ (nmo,) in {0, 2},
    fock (nao, nao). UHF: the same with a leading spin axis [alpha, beta] and
    mo_occ in {0, 1}. ``spin_densities`` is always (2, nao, nao).
    """
    reference: str
    converged: bool
    iterations: int
    energy: float
    electronic_energy: float
    nuclear_repulsion: float
    mo_energy: np.ndarray
    mo_coeff: np.ndarray
    mo_occ: np.ndarray
    spin_densities: np.ndarray
    fock: np.ndarray
    energy_weighted_density: np.ndarray
    mulliken_charges: np.ndarray
    dipole: np.ndarray
    s2: float
    n_alpha: int
    n_beta: int
    energy_change: float
    commutator_norm: float
    basis: BasisSet
    n_removed: int = 0
    guess: str = ""
    diis_restarts: int = 0
    history: list[dict[str, float]] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def unrestricted(self) -> bool:
        return self.reference == "uhf"

    @property
    def density(self) -> np.ndarray:
        """Total density P = D_alpha + D_beta, (nao, nao)."""
        return self.spin_densities[0] + self.spin_densities[1]

    @property
    def density_alpha(self) -> np.ndarray:
        return self.spin_densities[0]

    @property
    def density_beta(self) -> np.ndarray:
        return self.spin_densities[1]

    @property
    def spin_density(self) -> np.ndarray:
        """D_alpha - D_beta."""
        return self.spin_densities[0] - self.spin_densities[1]

    @property
    def guess_density(self) -> np.ndarray:
        """The density in the density-guess protocol shape: (nao, nao) RHF, (2, nao, nao) UHF."""
        return self.spin_densities.copy() if self.unrestricted else self.density

    @property
    def nmo(self) -> int:
        return int(self.mo_coeff.shape[-1])

    @property
    def s2_exact(self) -> float:
        """S (S + 1) of the pure spin state with S = (n_alpha - n_beta) / 2."""
        sz = 0.5 * (self.n_alpha - self.n_beta)
        return sz * (sz + 1.0)

    def coulomb_exchange_densities(self) -> tuple[np.ndarray, list[np.ndarray], float]:
        """(D_coulomb, D_exchange_list, k_factor) for integrals.two_electron_gradient."""
        if self.unrestricted:
            return self.density, [self.density_alpha, self.density_beta], 1.0
        P = self.density
        return P, [P], 0.5

    def __repr__(self) -> str:
        flag = "converged" if self.converged else "NOT CONVERGED"
        return (f"SCFResult({self.reference.upper()}/{self.basis.name}, E = {self.energy:.10f} Eh, "
                f"{self.iterations} iterations, {flag})")


# ================================================================= solver

# Closest allowed approach of two nuclei, bohr (PySCF's "Ill geometry" test
# in gto.mole.energy_nuc uses the same 1e-5 bohr).
MIN_NUCLEAR_SEPARATION = 1e-5


def check_nuclear_separation(positions: np.ndarray, atomic_numbers: np.ndarray) -> None:
    """
    ValueError if two nuclei (Z > 0) are closer than MIN_NUCLEAR_SEPARATION
    bohr. Coincident nuclei make Z_A Z_B / R_AB and its gradient infinite /
    NaN while the electronic integrals stay finite, so an SCF would "converge"
    to E = +inf; refuse instead, like non-finite positions (aimd.qc.basis).
    """
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    z = np.asarray(atomic_numbers)
    if len(x) < 2 or not np.all(np.isfinite(x)):
        return                       # non-finite positions: refused by BasisSet
    r = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1)
    i, j = np.triu_indices(len(x), k=1)
    bad = (r[i, j] < MIN_NUCLEAR_SEPARATION) & (z[i] > 0) & (z[j] > 0)
    if np.any(bad):
        k = int(np.argmax(bad))
        raise ValueError(
            f"atoms {i[k]} and {j[k]} are {r[i[k], j[k]]:.3g} bohr apart (coincident "
            f"nuclei, < {MIN_NUCLEAR_SEPARATION:g} bohr): the nuclear repulsion is infinite")


class SCFSolver:
    """
    Reusable RHF/UHF solver for one molecule (atoms, basis, charge, spin).
    The basis is parsed once; every :meth:`run` moves it to new positions
    (``BasisSet.with_positions``, O(1)), rebuilds the integrals and iterates
    from the requested guess. Integrals of the most recent geometry are kept,
    so repeated runs at the same positions (e.g. with different guesses)
    skip the integral step (replacing ``options`` invalidates them).

    Parameters
      symbols       element symbols
      basis         basis name ('sto-3g', '6-31G*', 'cc-pVDZ', ...) or a
                    BasisSet (its own positions are ignored)
      charge, multiplicity
      reference     "rhf", "uhf" or None (RHF for singlets, UHF otherwise)
      options       SCFOptions; keyword overrides are applied on top
    """

    def __init__(
        self,
        symbols: Sequence[str],
        basis: str | BasisSet = "sto-3g",
        charge: int = 0,
        multiplicity: int = 1,
        reference: str | None = None,
        options: SCFOptions | None = None,
        **overrides: Any,
    ) -> None:
        opt = options or SCFOptions()
        if overrides:
            known = {f.name for f in fields(SCFOptions)}
            bad = sorted(set(overrides) - known)
            if bad:
                raise TypeError(f"unknown SCF option(s): {', '.join(bad)}")
            opt = replace(opt, **overrides)
        self.options = opt
        n = len(symbols)
        if isinstance(basis, BasisSet):
            if [normalize_symbol(s) for s in symbols] != basis.symbols:
                raise ValueError("symbols do not match the atoms of the given BasisSet")
            self.basis = basis
        else:
            self.basis = build_basis(symbols, np.zeros((n, 3)), basis)
        self.symbols = list(self.basis.symbols)
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)
        if reference is None:
            ref = "rhf" if self.multiplicity == 1 else "uhf"
        else:
            key = str(reference).strip().lower()
            if key not in _REFERENCES:
                raise ValueError(f"unknown reference {reference!r}; use 'rhf' or 'uhf'")
            ref = _REFERENCES[key]
        self.reference = ref
        self.n_alpha, self.n_beta = electron_counts(
            int(self.basis.atomic_numbers.sum()), self.charge, self.multiplicity, ref)
        if self.n_alpha > self.basis.nao:
            raise ValueError(f"{self.n_alpha} alpha electrons do not fit in {self.basis.nao} AOs")
        self._ints: SCFIntegrals | None = None
        self._ints_positions: np.ndarray | None = None
        self._ints_options: SCFOptions | None = None
        self._sad: np.ndarray | None = None

    # ------------------------------------------------------------ properties
    @property
    def unrestricted(self) -> bool:
        return self.reference == "uhf"

    @property
    def nao(self) -> int:
        return self.basis.nao

    @property
    def nelectron(self) -> int:
        return self.n_alpha + self.n_beta

    @property
    def density_shape(self) -> tuple[int, ...]:
        """Shape of a density guess / ``SCFResult.guess_density``."""
        n = self.nao
        return (2, n, n) if self.unrestricted else (n, n)

    # ------------------------------------------------------------- integrals
    def integrals(self, positions: np.ndarray) -> SCFIntegrals:
        """SCF integrals at ``positions`` (bohr, (natm, 3)); cached for the last geometry."""
        x = np.array(positions, dtype=float).reshape(-1, 3)
        if x.shape[0] != self.basis.natm:
            raise ValueError(f"expected {self.basis.natm} positions, got {x.shape[0]}")
        check_nuclear_separation(x, self.basis.atomic_numbers)
        if (self._ints is None or self._ints_options is not self.options
                or not np.array_equal(x, self._ints_positions)):
            self._ints = None      # release the old ERI tensor before building the new one
            self._ints = build_integrals(self.basis.with_positions(x), self.options)
            self._ints_positions = x
            self._ints_options = self.options
        return self._ints

    def clear_cache(self) -> None:
        """Drop the cached integrals (the ERI tensor) of the last geometry."""
        self._ints = None
        self._ints_positions = None

    # ---------------------------------------------------------------- guesses
    def _nocc(self) -> tuple[int, ...]:
        return (self.n_alpha,) if not self.unrestricted else (self.n_alpha, self.n_beta)

    def initial_spin_densities(self, ints_: SCFIntegrals, guess: Any) -> tuple[np.ndarray, str]:
        """(nspin, nao, nao) starting spin densities and a label for ``guess``."""
        nspin = 2 if self.unrestricted else 1
        n = self.nao
        if isinstance(guess, SCFResult):
            if guess.basis.nao != n:
                raise ValueError("guess SCFResult has a different basis")
            D = guess.spin_densities
            return (D[:nspin].copy() if self.unrestricted else 0.5 * (D[0] + D[1])[None]), "result"
        if guess is None:
            guess = self.options.guess
        if isinstance(guess, str):
            kind = guess.strip().lower()
            if kind == "sad":
                if self._sad is None:
                    self._sad = sad_density(self.basis)
                Ds = np.repeat(0.5 * self._sad[None], nspin, axis=0)
            elif kind in ("core", "gwh"):
                F = core_guess_matrix(ints_) if kind == "core" else gwh_guess_matrix(ints_)
                Ds = np.array([_occupied_density(F, ints_.X, k) for k in self._nocc()])
            else:
                raise ValueError(f"unknown guess {guess!r}; use one of {_GUESSES} or a density")
            return Ds, kind
        D = np.array(guess, dtype=float)
        if D.shape == (n, n):
            Ds = np.repeat(0.5 * D[None], nspin, axis=0)
        elif D.shape == (2, n, n):
            Ds = D.copy() if self.unrestricted else 0.5 * (D[0] + D[1])[None]
        else:
            raise ValueError(
                f"density guess has shape {D.shape}; expected {self.density_shape} "
                f"({self.reference.upper()}) or {(n, n) if self.unrestricted else (2, n, n)}")
        if not np.all(np.isfinite(Ds)):
            raise ValueError("density guess contains non-finite values")
        return 0.5 * (Ds + Ds.transpose(0, 2, 1)), "density"

    # ------------------------------------------------------------------- SCF
    def run(self, positions: np.ndarray, guess: Any = None) -> SCFResult:
        """
        SCF at ``positions`` (bohr, (natm, 3)) starting from ``guess`` (None:
        ``options.guess``; "sad" / "core" / "gwh"; a density array; or an
        SCFResult). Unconverged runs are returned flagged, see module docstring.
        """
        return self._run(positions, guess, stacklevel=3)

    def _run(self, positions: np.ndarray, guess: Any, stacklevel: int) -> SCFResult:
        # stacklevel: frames from warnings.warn up to the public caller, so the
        # SCFConvergenceWarning points at the user's call (run or run_scf)
        t0 = time.perf_counter()
        with limit_blas_threads(self.options.blas_threads):
            I = self.integrals(positions)
            t1 = time.perf_counter()
            Ds, label = self.initial_spin_densities(I, guess)
            t2 = time.perf_counter()
            if I.nmo < self.n_alpha:
                raise ValueError(f"{self.n_alpha} alpha electrons do not fit in {I.nmo} "
                                 "linearly independent orbitals")
            state = self._iterate(I, Ds)
            t3 = time.perf_counter()
            res = self._result(I, state, label)
        res.timings = {"integrals": t1 - t0, "guess": t2 - t1, "iterations": t3 - t2,
                       "total": time.perf_counter() - t0}
        if not res.converged and self.options.warn_unconverged:
            warnings.warn(
                f"SCF not converged after {res.iterations} iterations "
                f"(dE = {res.energy_change:.2e} Eh, commutator {res.commutator_norm:.2e}); "
                "returning the last iterate flagged converged=False",
                SCFConvergenceWarning, stacklevel=stacklevel)
        return res

    def _fock(self, I: SCFIntegrals, Ds: np.ndarray) -> np.ndarray:
        g = 2.0 / Ds.shape[0]
        J, K = qcint.jk_from_eri(I.eri, Ds)
        F = I.H + g * J.sum(axis=0) - K
        return 0.5 * (F + F.transpose(0, 2, 1))

    def _iterate(self, I: SCFIntegrals, Ds: np.ndarray) -> dict[str, Any]:
        opt = self.options
        nspin = Ds.shape[0]
        g = 2.0 / nspin
        nocc = self._nocc()
        S, H, X = I.S, I.H, I.X
        XtS = X.T @ S
        diis = DIIS(opt.diis_space) if opt.diis else None
        e_prev = np.nan
        best = np.inf
        since_best = 0
        restarts = 0
        history: list[dict[str, float]] = []
        converged = False
        for it in range(1, opt.max_iter + 1):
            F = self._fock(I, Ds)
            e_el = 0.5 * g * float(sum(np.vdot(Ds[s], H + F[s]) for s in range(nspin)))
            FDS = F @ Ds @ S
            err = X.T @ (FDS - FDS.transpose(0, 2, 1)) @ X
            err_norm = float(np.abs(err).max())
            de = e_el - e_prev            # nan at the first iteration
            history.append({"energy": e_el + I.nuclear_repulsion, "energy_change": de,
                            "commutator": err_norm, "diis": len(diis) if diis else 0})
            if abs(de) < opt.conv_energy and err_norm < opt.conv_commutator:
                converged = True
                break
            if it == opt.max_iter:
                break
            Fo = X.T @ F @ X
            if diis is not None:
                if err_norm < best:
                    best, since_best = err_norm, 0
                else:
                    since_best += 1
                if opt.diis_restart and since_best >= opt.diis_restart:
                    diis.reset()
                    restarts += 1
                    best, since_best = err_norm, 0
                diis.push(Fo, err)
                Fo = diis.extrapolate()
            stabilize = err_norm > opt.stabilize_until
            if stabilize and opt.level_shift > 0.0:
                Do = XtS @ Ds @ XtS.T          # spin densities in the orthonormal basis
                Fo = Fo + opt.level_shift * (np.eye(I.nmo) - Do)
            Dn = np.empty_like(Ds)
            for s in range(nspin):
                _, Cp = np.linalg.eigh(Fo[s])
                C = X @ Cp[:, :nocc[s]]
                Dn[s] = C @ C.T
            if stabilize and opt.damping > 0.0:
                Dn = (1.0 - opt.damping) * Dn + opt.damping * Ds
            Ds = Dn
            e_prev = e_el
        return {"Ds": Ds, "F": F, "e_el": e_el, "de": de, "err": err_norm, "it": it,
                "converged": converged, "history": history, "restarts": restarts}

    def _result(self, I: SCFIntegrals, st: dict[str, Any], label: str) -> SCFResult:
        basis = I.basis
        Ds, F = st["Ds"], st["F"]
        nspin = Ds.shape[0]
        g = 2.0 / nspin
        nocc = self._nocc()
        X, S = I.X, I.S
        eps = np.empty((nspin, I.nmo))
        C = np.empty((nspin, I.nao, I.nmo))
        occ = np.zeros((nspin, I.nmo))
        W = np.zeros((I.nao, I.nao))
        for s in range(nspin):
            eps[s], Cp = np.linalg.eigh(X.T @ F[s] @ X)
            C[s] = X @ Cp
            occ[s, :nocc[s]] = g
            Co = C[s][:, :nocc[s]]
            W += g * (Co * eps[s, :nocc[s]]) @ Co.T
        spin = np.array([Ds[0], Ds[-1]])          # RHF: alpha = beta
        P = spin[0] + spin[1]
        PS = P @ S
        q = basis.nuclear_charges.copy()
        np.subtract.at(q, basis.ao_atom, np.diag(PS))
        r = qcint.dipole_integrals(basis)
        dip = -np.einsum("xmn,mn->x", r, P) + qcint.nuclear_dipole(basis.nuclear_charges, basis.positions)
        sz = 0.5 * (self.n_alpha - self.n_beta)
        if nspin == 2:
            s2 = sz * (sz + 1.0) + self.n_beta - float(np.vdot(spin[0] @ S, (spin[1] @ S).T))
        else:
            s2 = 0.0
        if nspin == 1:
            eps, C, occ, F = eps[0], C[0], occ[0], F[0]
        e_el = st["e_el"]
        return SCFResult(
            reference=self.reference, converged=st["converged"], iterations=st["it"],
            energy=e_el + I.nuclear_repulsion, electronic_energy=e_el,
            nuclear_repulsion=I.nuclear_repulsion, mo_energy=eps, mo_coeff=C, mo_occ=occ,
            spin_densities=spin, fock=F, energy_weighted_density=0.5 * (W + W.T),
            mulliken_charges=q, dipole=dip, s2=float(s2), n_alpha=self.n_alpha,
            n_beta=self.n_beta, energy_change=float(st["de"]), commutator_norm=st["err"],
            basis=basis, n_removed=I.n_removed, guess=label, diis_restarts=st["restarts"],
            history=st["history"])


def run_scf(
    symbols: Sequence[str],
    positions: np.ndarray,
    basis: str | BasisSet = "sto-3g",
    charge: int = 0,
    multiplicity: int = 1,
    reference: str | None = None,
    guess: Any = None,
    options: SCFOptions | None = None,
    **overrides: Any,
) -> SCFResult:
    """One-shot SCF: ``SCFSolver(...).run(positions, guess)``."""
    solver = SCFSolver(symbols, basis, charge=charge, multiplicity=multiplicity,
                       reference=reference, options=options, **overrides)
    return solver._run(positions, guess, stacklevel=3)
