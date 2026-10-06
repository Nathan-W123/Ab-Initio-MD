"""
Analytic nuclear gradients of restricted (RHF) and unrestricted (UHF)
Hartree-Fock energies, from an :class:`aimd.qc.scf.SCFResult`.

Units: Hartree atomic units. Gradients are dE/dR_A in hartree / bohr, shape
(natm, 3), rows in the atom order of the SCF; forces are ``-gradient``.

Energy
------
With spin densities D_a, D_b in the AO basis (RHF: D_a = D_b = P/2), total
density P = D_a + D_b, core Hamiltonian h = T + V and ERIs (mn|ls) in
chemists' notation (:mod:`aimd.qc.integrals`),

    E = E_nuc + sum_mn P_mn h_mn + 1/2 sum_mnls Gamma_mnls (mn|ls),
    Gamma_mnls = P_mn P_ls - sum_s D^s_ml D^s_ns

(RHF: Gamma = P_mn P_ls - 1/2 P_ml P_ns).

Gradient
--------
The AOs move with their nuclei, so S, h and (mn|ls) depend on the nuclear
coordinates X_A; the MO coefficients depend on them too, but their response
drops out because the SCF energy is stationary under orbital rotations
within the orthonormality constraint C_s^T S C_s = 1. What remains (Pulay,
Mol. Phys. 17, 197 (1969); Pople, Krishnan, Schlegel & Binkley, Int. J.
Quantum Chem. Symp. 13, 225 (1979); Szabo & Ostlund, *Modern Quantum
Chemistry* (1989), App. C) is

    dE/dX_A = dE_nuc/dX_A                                  nuclear repulsion
            + sum_mn P_mn d(T + V)_mn/dX_A                 one-electron, incl.
                                                           the Hellmann-Feynman
                                                           term dV/dX_A of the
                                                           operator -Z_A/|r-R_A|
            + 1/2 sum_mnls Gamma_mnls d(mn|ls)/dX_A        two-electron
            - sum_mn W_mn dS_mn/dX_A                       energy-weighted
                                                           density (Pulay) term

at fixed P, Gamma and W. Each integral-derivative contraction is evaluated
by :mod:`aimd.qc.integrals` without forming derivative-integral arrays.

Energy-weighted density (convention)
------------------------------------
W is the Lagrange multiplier of the orthonormality constraint:

    W = sum_s D_s F_s D_s             (RHF: W = 1/2 P F P),

which at convergence equals sum_s sum_{i occ} eps_si C_s,mi C_s,ni (the
"n_i eps_i C C^T" form, n_i = 2 for RHF). It is formed here from the density
and Fock matrix that the SCF returns (D_k and F[D_k] of its last iteration),
not from the canonical orbitals: for any idempotent D_s = C_s C_s^T the
occupied-occupied projection of the stationarity condition F_s C_s = S C_s
eps_s gives eps_s = C_s^T F_s C_s, i.e. W_s = D_s F_s D_s, and the only
residual is then the occupied-virtual block of F_s, the SCF convergence
measure. The canonical-orbital form, built from the eigenvectors of F[D_k]
(which differ from those spanning D_k by O(residual)), adds a second error
of the same order. Measured gradient errors against a tightly converged
reference, SCF stopped at the given commutator norm (canonical-orbital W vs
D F D): water/6-31G* 3e-6: 1.0e-5 vs 3.3e-6 Eh/bohr; NH2 doublet (UHF)
/6-31G* 4e-6: 1.3e-6 vs 3.2e-7; H2O+ /6-31G* 2e-6: 5.4e-7 vs 6.4e-8. At
convergence both forms agree to ~1e-13.

Accuracy and invariances
------------------------
The formula is the exact derivative only at a stationary SCF; otherwise the
error is first order in the orbital gradient: at most about the commutator
norm max|X^T (F D S - S D F) X| (ratios 0.03-1.0 measured on water,
ethanol, NH2 and H2O+ for commutators 1e-6..3e-4), so the default SCF
threshold 1e-7 gives <~1e-7 Eh/bohr. The energy is variational, its error
second order.

  translation  sum_A dE/dR_A = 0. Exact by construction for the nuclear-
               repulsion and one-electron terms (one center of every
               derivative integral is obtained by invariance); the four-center
               ERI quartets differentiate bra and ket separately, so the ERI
               term sums to zero to rounding (measured <= 4e-14 Eh/bohr).
  rotation     sum_A (R_A - R_c) x dE/dR_A = 0, because the energy of an
               isolated molecule is invariant under rigid rotations (every
               Cartesian shell is closed under rotation). Not built in: it
               holds only for a converged SCF and correct derivative
               integrals, which makes it an independent check.

Known limitation: with canonical orthogonalization removing near-linear
dependencies (``SCFResult.n_removed > 0``) the variational space is a
projected one that moves with the nuclei, which this standard formula does
not differentiate; see :mod:`aimd.qc.scf` (measured error 1.7e-6 Eh/bohr on
water / 6-31G plus diffuse shells, PySCF behaves the same).

Performance: the two-electron term dominates (O(N^4) derivative quartets,
numba-parallel; ethanol / 6-31G* ~0.3 s on 4 cores); it reuses the shell-pair
data and Schwarz bounds that the SCF cached on ``SCFResult.basis``. The
one-electron and nuclear-repulsion terms take ~0.5-12 ms.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from aimd.qc import integrals as qcint
from aimd.qc.scf import SCFResult
from aimd.qc.threads import limit_blas_threads


def energy_weighted_density(spin_densities: np.ndarray, fock: np.ndarray) -> np.ndarray:
    """
    W = sum_s D_s F_s D_s, (nao, nao), symmetrized (module docstring).

    ``spin_densities`` is (2, nao, nao) [alpha, beta]; ``fock`` is one
    (nao, nao) matrix shared by both spins (RHF, then W = 1/2 P F P with
    P = D_a + D_b) or a (2, nao, nao) pair (UHF).
    """
    D = np.asarray(spin_densities, dtype=float)
    F = np.asarray(fock, dtype=float)
    if D.ndim != 3 or D.shape[0] != 2 or D.shape[1] != D.shape[2]:
        raise ValueError(f"spin_densities must have shape (2, nao, nao), got {D.shape}")
    n = D.shape[1]
    if F.shape == (n, n):
        W = D[0] @ F @ D[0] + D[1] @ F @ D[1]
    elif F.shape == (2, n, n):
        W = D[0] @ F[0] @ D[0] + D[1] @ F[1] @ D[1]
    else:
        raise ValueError(f"fock must have shape ({n}, {n}) or (2, {n}, {n}), got {F.shape}")
    return 0.5 * (W + W.T)


@dataclass(frozen=True)
class GradientTerms:
    """
    The parts of an SCF gradient (each (natm, 3), Eh/bohr; module docstring):

      nuclear_repulsion   dE_nuc/dR
      kinetic             sum P dT/dR
      nuclear_attraction  sum P dV/dR, basis-function and operator (Hellmann-
                          Feynman) centers
      overlap             -sum W dS/dR (sign included)
      two_electron        1/2 sum Gamma d(mn|ls)/dR

    ``total`` is their sum; ``timings`` holds wall times in seconds.
    """
    nuclear_repulsion: np.ndarray
    kinetic: np.ndarray
    nuclear_attraction: np.ndarray
    overlap: np.ndarray
    two_electron: np.ndarray
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def one_electron(self) -> np.ndarray:
        """kinetic + nuclear_attraction + overlap (= integrals.one_electron_gradient)."""
        return self.kinetic + self.nuclear_attraction + self.overlap

    @property
    def total(self) -> np.ndarray:
        return self.nuclear_repulsion + self.one_electron + self.two_electron


def _check(result: SCFResult, require_converged: bool) -> None:
    if not isinstance(result, SCFResult):
        raise TypeError(f"expected an SCFResult, got {type(result).__name__}")
    if require_converged and not result.converged:
        raise ValueError(
            f"SCF not converged ({result.iterations} iterations, commutator "
            f"{result.commutator_norm:.2e}); its analytic gradient is not dE/dR")


def scf_gradient(
    result: SCFResult,
    schwarz_threshold: float = qcint.DEFAULT_SCHWARZ_GRAD,
    require_converged: bool = False,
    blas_threads: int | None = 1,
) -> np.ndarray:
    """
    Analytic dE/dR (hartree / bohr, (natm, 3)) of an RHF or UHF SCF result.

    ``schwarz_threshold`` screens ERI-derivative quartets (see
    :data:`aimd.qc.integrals.DEFAULT_SCHWARZ_GRAD`). An unconverged result
    gives the formula evaluated at its last iterate (error ~ commutator norm,
    module docstring) unless ``require_converged`` is set, which raises
    ValueError instead. BLAS is limited to ``blas_threads`` meanwhile (see
    :mod:`aimd.qc.threads`; None leaves it alone).
    """
    _check(result, require_converged)
    b = result.basis
    with limit_blas_threads(blas_threads):
        P = result.density
        W = energy_weighted_density(result.spin_densities, result.fock)
        g = qcint.nuclear_repulsion_gradient(b.nuclear_charges, b.positions)
        g += qcint.one_electron_gradient(b, P, W)
        g += qcint.two_electron_gradient(b, *result.coulomb_exchange_densities(),
                                         schwarz_threshold=schwarz_threshold)
    return g


def gradient_terms(
    result: SCFResult,
    schwarz_threshold: float = qcint.DEFAULT_SCHWARZ_GRAD,
    require_converged: bool = False,
    blas_threads: int | None = 1,
) -> GradientTerms:
    """
    The gradient of ``result`` split into its physical parts (one extra
    overlap/kinetic integral pass compared with :func:`scf_gradient`).
    """
    _check(result, require_converged)
    b = result.basis
    with limit_blas_threads(blas_threads):
        t0 = time.perf_counter()
        P = result.density
        W = energy_weighted_density(result.spin_densities, result.fock)
        g_nuc = qcint.nuclear_repulsion_gradient(b.nuclear_charges, b.positions)
        t1 = time.perf_counter()
        g_t = qcint.kinetic_gradient(b, P)
        g_v = qcint.nuclear_attraction_gradient(b, P)
        g_s = -qcint.overlap_gradient(b, W)
        t2 = time.perf_counter()
        g_2 = qcint.two_electron_gradient(b, *result.coulomb_exchange_densities(),
                                          schwarz_threshold=schwarz_threshold)
        t3 = time.perf_counter()
    timings = {"nuclear_and_w": t1 - t0, "one_electron": t2 - t1, "two_electron": t3 - t2,
               "total": t3 - t0}
    return GradientTerms(g_nuc, g_t, g_v, g_s, g_2, timings)


def invariance_residuals(positions: np.ndarray, gradient: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    (net force sum_A g_A, net torque sum_A (R_A - R_c) x g_A), each (3,), for
    a gradient g (Eh/bohr) at ``positions`` (bohr); R_c is the centroid, which
    makes the torque origin-independent even if the net force is not exactly
    zero. Both vanish for the exact gradient of an isolated molecule.
    """
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    g = np.asarray(gradient, dtype=float).reshape(-1, 3)
    if x.shape != g.shape:
        raise ValueError(f"positions {x.shape} and gradient {g.shape} differ in shape")
    return g.sum(axis=0), np.cross(x - x.mean(axis=0), g).sum(axis=0)
