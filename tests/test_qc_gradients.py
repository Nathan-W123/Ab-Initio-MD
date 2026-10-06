"""
Tests for aimd.qc.gradients (analytic RHF / UHF nuclear gradients).

Independent references:
  - PySCF analytic gradients (mf.nuc_grad_method()) of a PySCF SCF on the same
    Cartesian basis data (tests/qc_reference.py), started from our density so
    that both land on the same SCF solution;
  - five-point central finite differences of our own SCF energy;
  - exact symmetries: translational and rotational invariance of the energy
    (net force and torque vanish) and covariance of the gradient under rigid
    motions and atom permutations;
  - literature HF equilibrium geometries of water (Hehre, Radom, Schleyer &
    Pople, *Ab Initio Molecular Orbital Theory* (1986); NIST CCCBDB), located
    with our analytic gradient.
"""

from __future__ import annotations

import functools

import numpy as np
import pytest

import qc_reference as ref
from pyscf.grad import rhf as pyscf_grad_rhf
from aimd.qc import integrals as I
from aimd.qc import scf as S
from aimd.qc.gradients import (
    GradientTerms, energy_weighted_density, gradient_terms, invariance_residuals, scf_gradient,
)
from aimd.units import BOHR_TO_ANG

# Tight SCF: the analytic gradient error is ~ the commutator norm (see
# aimd.qc.gradients), so 1e-10 leaves it far below every tolerance here.
TIGHT = dict(conv_commutator=1e-10, conv_energy=1e-13)

# (molecule, basis, random displacement in bohr, charge, multiplicity).
# Displaced geometries have no symmetry: every gradient component is nonzero
# and nothing vanishes by accident.
RHF_CASES = [
    ("water", "sto-3g", 0.0, 0, 1),
    ("water", "6-31g*", 0.15, 0, 1),
    ("ammonia", "6-31g**", 0.1, 0, 1),
    ("hcl", "cc-pvdz", 0.1, 0, 1),
    ("ch3oh", "6-31g*", 0.1, 0, 1),
]
UHF_CASES = [
    ("nh2", "6-31g*", 0.1, 0, 2),
    ("water", "6-31g*", 0.15, 1, 2),
    ("o2", "6-31g*", 0.05, 0, 3),
    ("ch2", "cc-pvdz", 0.1, 0, 3),
]


@functools.lru_cache(maxsize=None)
def _case(name, basis, distort, charge, mult):
    """(our tight SCF result, our gradient, PySCF's gradient)."""
    sym, pos = ref.molecule(name, distort=distort, seed=7)
    res = S.run_scf(sym, pos, basis, charge=charge, multiplicity=mult, **TIGHT)
    assert res.converged
    mol = ref.pyscf_mole(res.basis, charge=charge, spin=mult - 1)
    dm0 = ref.density_to_pyscf(res.guess_density, ref.ao_scale(res.basis))
    mf = ref.run_scf(mol, unrestricted=res.unrestricted, conv_tol=1e-12, conv_tol_grad=1e-9,
                     dm0=dm0)
    assert abs(mf.e_tot - res.energy) < 1e-9          # same SCF solution
    return res, scf_gradient(res), mf.nuc_grad_method().kernel()


# ================================================================ vs PySCF

@pytest.mark.parametrize("case", RHF_CASES + UHF_CASES, ids=lambda c: f"{c[0]}-{c[1]}-m{c[4]}")
def test_gradient_matches_pyscf(case):
    res, g, g_pyscf = _case(*case)
    assert res.reference == ("rhf" if case[4] == 1 else "uhf")
    assert g.shape == (len(res.basis.symbols), 3)
    # measured <= 3.4e-11 Eh/bohr (gradients up to 0.13); the 1e-8 tolerance
    # also covers PySCF's own convergence (conv_tol_grad 1e-9)
    np.testing.assert_allclose(g, g_pyscf, rtol=0, atol=1e-8)
    assert np.abs(g).max() > 1e-3                       # not trivially zero


def test_terms_sum_to_the_gradient_and_nuclear_term_matches_pyscf():
    res, g, _ = _case("water", "6-31g*", 0.15, 0, 1)
    terms = gradient_terms(res)
    assert isinstance(terms, GradientTerms)
    np.testing.assert_allclose(terms.total, g, rtol=0, atol=1e-12)
    b = res.basis
    P = res.density
    W = 0.5 * P @ res.fock @ P
    np.testing.assert_allclose(terms.one_electron, I.one_electron_gradient(b, P, W),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(terms.overlap, -I.overlap_gradient(b, W), rtol=0, atol=1e-12)
    g_nuc = pyscf_grad_rhf.grad_nuc(ref.pyscf_mole(b))
    np.testing.assert_allclose(terms.nuclear_repulsion, g_nuc, rtol=0, atol=1e-12)
    assert set(terms.timings) >= {"one_electron", "two_electron", "total"}
    # every physical part matters at this geometry
    for part in (terms.kinetic, terms.nuclear_attraction, terms.overlap, terms.two_electron):
        assert np.abs(part).max() > 1e-2


# ================================================================ vs finite differences

def _fd5(solver, pos, guess, h):
    """Five-point central differences of the SCF energy, (natm, 3)."""
    fd = np.zeros_like(pos)
    for a in range(pos.shape[0]):
        for k in range(3):
            e = {}
            for m in (-2, -1, 1, 2):
                x = pos.copy()
                x[a, k] += m * h
                r = solver.run(x, guess=guess)
                assert r.converged
                e[m] = r.energy
            fd[a, k] = (e[-2] - 8.0 * e[-1] + 8.0 * e[1] - e[2]) / (12.0 * h)
    return fd


@pytest.mark.parametrize("name,basis,charge,mult", [
    ("water", "6-31g*", 0, 1), ("nh2", "6-31g*", 0, 2), ("ammonia", "6-31g**", 1, 2)])
def test_gradient_matches_finite_differences_of_the_energy(name, basis, charge, mult):
    sym, pos = ref.molecule(name, distort=0.15, seed=11)
    solver = S.SCFSolver(sym, basis, charge=charge, multiplicity=mult, **TIGHT)
    r0 = solver.run(pos)
    g = scf_gradient(r0)
    fd = _fd5(solver, pos, r0, h=2e-3)
    # measured 7.3e-11 (water), 6.7e-11 (NH2), 1.7e-10 (NH3+); truncation
    # O(h^4) and round-off (~1e-13 Eh / h) are both ~1e-10 at h = 2e-3 bohr
    np.testing.assert_allclose(g, fd, rtol=0, atol=3e-9)


# ================================================================ invariances

@pytest.mark.parametrize("case", RHF_CASES + UHF_CASES, ids=lambda c: f"{c[0]}-{c[1]}-m{c[4]}")
def test_translational_and_rotational_invariance(case):
    res, g, _ = _case(*case)
    force, torque = invariance_residuals(res.basis.positions, g)
    # translation: exact by construction except for the four-center ERI
    # derivatives (measured <= 3.9e-14); rotation: holds only for a correct,
    # converged gradient (measured <= 8.5e-11 Eh, gradients ~0.1 Eh/bohr)
    assert np.abs(force).max() < 1e-11
    assert np.abs(torque).max() < 1e-9


def test_invariance_residuals_detect_a_broken_gradient():
    # The same check fails for a gradient missing its energy-weighted density
    # (Pulay) term: the AO basis then "drags" the molecule.
    res, g, _ = _case("water", "6-31g*", 0.15, 0, 1)
    pulay = gradient_terms(res).overlap
    force, torque = invariance_residuals(res.basis.positions, g - pulay)
    assert np.abs(force).max() < 1e-11                  # each term is translation invariant
    assert np.abs(torque).max() > 1e-3                  # but not rotation invariant (1.6e-3)
    with pytest.raises(ValueError, match="shape"):
        invariance_residuals(np.zeros((3, 3)), np.zeros((2, 3)))


def test_gradient_is_covariant_under_rigid_motion_and_permutation():
    sym, pos = ref.molecule("ammonia", distort=0.1, seed=3)
    rng = np.random.default_rng(5)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))                      # proper rotation
    t = np.array([0.7, -1.3, 2.2])
    solver = S.SCFSolver(sym, "6-31g*", **TIGHT)
    r0 = solver.run(pos)
    g0 = scf_gradient(r0)
    r1 = solver.run(pos @ q.T + t)
    g1 = scf_gradient(r1)
    assert r1.energy == pytest.approx(r0.energy, abs=1e-10)
    # measured 9e-15 (rotation) and 1.3e-14 (permutation): the SCF iterations
    # are covariant themselves, so only rounding differs; the tolerance allows
    # for a different iteration count (gradient error ~ commutator ~ 1e-10)
    np.testing.assert_allclose(g1, g0 @ q.T, rtol=0, atol=1e-9)
    perm = [0, 3, 1, 2]                                  # relabel the H atoms
    rp = S.run_scf([sym[i] for i in perm], pos[perm], "6-31g*", **TIGHT)
    np.testing.assert_allclose(scf_gradient(rp), g0[perm], rtol=0, atol=1e-9)


# ================================================================ literature

@pytest.mark.parametrize("basis,r_ang,theta_deg", [("sto-3g", 0.989, 100.0), ("6-31g*", 0.947, 105.5)])
def test_water_equilibrium_geometry_matches_literature(basis, r_ang, theta_deg):
    """
    HF equilibrium geometries of water, r(OH) / angle: STO-3G 0.989 A / 100.0
    deg, 6-31G* (6 Cartesian d) 0.947 A / 105.5 deg (Hehre, Radom, Schleyer &
    Pople 1986; NIST CCCBDB). BFGS driven by the analytic gradient stops
    where the gradient vanishes, so a wrong gradient gives a wrong geometry.
    Measured: 0.98942 A / 100.03 deg and 0.94732 A / 105.50 deg; tolerance
    half a unit of the last quoted digit.
    """
    from scipy.optimize import minimize
    solver = S.SCFSolver(["O", "H", "H"], basis, conv_commutator=1e-9, conv_energy=1e-12)
    x0 = np.array([[0.0, 0.0, 0.2], [0.0, 1.4, -0.9], [0.1, -1.5, -0.95]])

    def energy_and_gradient(v):
        r = solver.run(v.reshape(3, 3))
        return r.energy, scf_gradient(r).ravel()

    opt = minimize(energy_and_gradient, x0.ravel(), jac=True, method="BFGS",
                   options={"gtol": 1e-7})
    assert opt.success
    x = opt.x.reshape(3, 3)
    b1, b2 = x[1] - x[0], x[2] - x[0]
    r1, r2 = np.linalg.norm(b1), np.linalg.norm(b2)
    theta = np.degrees(np.arccos(b1 @ b2 / (r1 * r2)))
    assert r1 * BOHR_TO_ANG == pytest.approx(r_ang, abs=5e-4)
    assert r2 * BOHR_TO_ANG == pytest.approx(r_ang, abs=5e-4)
    assert theta == pytest.approx(theta_deg, abs=0.05)


# ================================================================ W convention, loose SCF

def test_energy_weighted_density_conventions():
    res, _, _ = _case("water", "6-31g*", 0.15, 1, 2)          # UHF
    D, F = res.spin_densities, res.fock
    W = energy_weighted_density(D, F)
    np.testing.assert_allclose(W, D[0] @ F[0] @ D[0] + D[1] @ F[1] @ D[1], atol=1e-14)
    # at convergence equal to the canonical-orbital form sum n_i eps_i C C^T
    np.testing.assert_allclose(W, res.energy_weighted_density, rtol=0, atol=1e-9)
    rhf, _, _ = _case("water", "6-31g*", 0.15, 0, 1)
    P = rhf.density
    np.testing.assert_allclose(energy_weighted_density(rhf.spin_densities, rhf.fock),
                               0.5 * P @ rhf.fock @ P, rtol=0, atol=1e-13)
    with pytest.raises(ValueError, match="spin_densities"):
        energy_weighted_density(P, rhf.fock)
    with pytest.raises(ValueError, match="fock"):
        energy_weighted_density(rhf.spin_densities, np.zeros((3, 3)))


@pytest.mark.parametrize("name,charge,mult", [("water", 0, 1), ("nh2", 0, 2)])
def test_gradient_error_of_a_loose_scf_is_of_the_order_of_the_commutator(name, charge, mult):
    """
    Away from convergence the gradient error is first order in the orbital
    gradient. Measured, SCF stopped at commutator 3.4e-5 (water) / 9.0e-5
    (NH2): errors 9.5e-6 / 6.7e-6 Eh/bohr with W = sum D F D, and 2.9e-5 /
    3.0e-5 with W from the canonical orbitals of the last Fock matrix (the
    reason for the D F D convention, see aimd.qc.gradients).
    """
    sym, pos = ref.molecule(name, distort=0.1, seed=3)
    exact = scf_gradient(S.run_scf(sym, pos, "6-31g*", charge=charge, multiplicity=mult, **TIGHT))
    loose = S.run_scf(sym, pos, "6-31g*", charge=charge, multiplicity=mult,
                      conv_commutator=1e-4, conv_energy=1.0)
    assert 1e-6 < loose.commutator_norm < 1e-4
    err = np.abs(scf_gradient(loose) - exact).max()
    assert err < 2.0 * loose.commutator_norm
    b = loose.basis
    canonical = (I.nuclear_repulsion_gradient(b.nuclear_charges, b.positions)
                 + I.one_electron_gradient(b, loose.density, loose.energy_weighted_density)
                 + I.two_electron_gradient(b, *loose.coulomb_exchange_densities()))
    assert err < 0.5 * np.abs(canonical - exact).max()


def test_unconverged_results_are_refused_on_request():
    sym, pos = ref.molecule("water")
    with pytest.warns(S.SCFConvergenceWarning):
        res = S.run_scf(sym, pos, "sto-3g", max_iter=2)
    g = scf_gradient(res)                                # allowed by default
    assert np.all(np.isfinite(g))
    with pytest.raises(ValueError, match="not converged"):
        scf_gradient(res, require_converged=True)
    with pytest.raises(ValueError, match="not converged"):
        gradient_terms(res, require_converged=True)
    with pytest.raises(TypeError, match="SCFResult"):
        scf_gradient(res.density)


def test_parallel_work_chunks_cover_every_pair_once_and_spread_over_threads():
    """
    The ERI / ERI-gradient kernels give chunk c the block pairs CHUNK_FIRST[c]
    + k NCHUNK. numba's OpenMP backend hands each of T threads a contiguous
    block of chunks; every block must hold one residue class mod T, so that
    even a molecule with < NCHUNK / T block pairs uses all threads (with
    CHUNK_FIRST[c] = c water / STO-3G ran serially). Bitwise independence of
    the thread count is checked in test_qc_integrals and test_hf_backend.
    """
    first = I.CHUNK_FIRST
    assert sorted(first.tolist()) == list(range(I.NCHUNK))           # a permutation
    for T in (2, 4, 8, 16):
        block = I.NCHUNK // T
        residues = [set((first[t * block:(t + 1) * block] % T).tolist()) for t in range(T)]
        assert all(len(r) == 1 for r in residues)
        assert set().union(*residues) == set(range(T))
