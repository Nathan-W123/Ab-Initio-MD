"""
Tests for aimd.qc.scf (RHF / UHF on the native integrals).

Independent references:
  - PySCF RHF/UHF (Cartesian Mole with the same basis data, tests/qc_reference.py)
    for energies, <S^2>, orbital energies, densities, energy-weighted
    densities, dipoles, Mulliken charges and analytic gradients; PySCF
    restarted from our density checks that we found the same SCF solution;
    PySCF's Fock matrix and energy at our returned density check the reported
    energy, Fock matrix and convergence measure (commutator);
  - PySCF's spherically averaged atomic HF (scf.atom_hf) and Cartesian closed-
    shell atomic RHF for the SAD atomic calculations; tabulated ground-state
    electron configurations;
  - exact identities: duplicated basis functions must not change the energy,
    DIIS is exact for a linear error model, X^T S X = 1.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

import qc_reference as ref
from pyscf import gto
from pyscf import scf as pyscf_scf
from pyscf.scf import atom_hf

from aimd.qc import integrals as I
from aimd.qc import scf as S
from aimd.qc.basis import BasisSet, build_basis
from aimd.qc.basis_data import BASIS_SETS

# Tight SCF for property comparisons: with the default commutator threshold
# (1e-7) orbital energies, densities, W, dipoles, charges and gradients agree
# with PySCF to 4e-8 .. 2.5e-7 (measured on water/6-31G*, HCl/cc-pVDZ,
# formaldehyde/6-31G**, OH, CH2, H2O+); at 1e-9 they agree to <= 4e-9, so
# the 1e-7 tolerances below have a > 25x margin while still failing for
# any real error (wrong W convention, wrong occupation, missing factor).
TIGHT = dict(conv_commutator=1e-9, conv_energy=1e-12)


def _geometry(name):
    if name == "h":
        return ["H"], np.zeros((1, 3))
    return ref.molecule(name)


@functools.lru_cache(maxsize=None)
def _ours(name, basis, charge=0, mult=1, reference=None, tight=False):
    sym, pos = _geometry(name)
    return S.run_scf(sym, pos, basis, charge=charge, multiplicity=mult, reference=reference,
                     **(TIGHT if tight else {}))


@functools.lru_cache(maxsize=None)
def _pyscf(name, basis, charge=0, mult=1, unrestricted=False, tight=False):
    sym, pos = _geometry(name)
    mol = ref.pyscf_mole(build_basis(sym, pos, basis), charge=charge, spin=mult - 1)
    return ref.run_scf(mol, unrestricted=unrestricted, conv_tol_grad=1e-9 if tight else None)


def _stretched_water(factor):
    sym, pos = ref.molecule("water")
    pos = pos.copy()
    pos[1:] = pos[0] + factor * (pos[1:] - pos[0])
    return sym, pos


def _displacement(pos, size=0.01, seed=1):
    d = np.random.default_rng(seed).normal(size=pos.shape)
    return d * (size / np.linalg.norm(d, axis=1).max())


# ======================================================== RHF vs PySCF

RHF_CASES = [
    ("h2", "sto-3g"), ("h2", "cc-pvdz"),
    ("water", "sto-3g"), ("water", "6-31g"), ("water", "6-31g*"), ("water", "cc-pvdz"),
    ("ammonia", "6-31g**"), ("ch4", "6-31g*"), ("hf", "cc-pvdz"), ("n2", "6-31g*"),
    ("hcn", "6-31g"), ("formaldehyde", "6-31g**"), ("h2s", "6-31g*"), ("hcl", "cc-pvdz"),
    ("c2h4", "6-31g*"), ("ch3oh", "cc-pvdz"),
]


@pytest.mark.parametrize("name,basis", RHF_CASES)
def test_rhf_energy_matches_pyscf(name, basis):
    res = _ours(name, basis)
    mf = _pyscf(name, basis)
    assert res.converged and res.reference == "rhf"
    assert abs(res.energy - mf.e_tot) < 1e-8, res.energy - mf.e_tot
    assert abs(res.nuclear_repulsion - mf.energy_nuc()) < 1e-10
    assert res.energy == pytest.approx(res.electronic_energy + res.nuclear_repulsion, abs=1e-12)
    assert res.commutator_norm < 1e-7 and abs(res.energy_change) < 1e-10
    assert res.s2 == 0.0 and res.n_alpha == res.n_beta
    # closed shell: occupations, electron count, alpha = beta
    np.testing.assert_array_equal(res.mo_occ[: res.n_alpha], 2.0)
    assert res.mo_occ.sum() == res.n_alpha + res.n_beta
    np.testing.assert_array_equal(res.density_alpha, res.density_beta)
    S_ = I.overlap_matrix(res.basis)
    assert np.trace(res.density @ S_) == pytest.approx(res.mo_occ.sum(), abs=1e-10)


@pytest.mark.parametrize("name,basis", [("water", "6-31g*"), ("hcl", "cc-pvdz"),
                                        ("formaldehyde", "6-31g**")])
def test_rhf_orbitals_densities_and_properties_match_pyscf(name, basis):
    res = _ours(name, basis, tight=True)
    mf = _pyscf(name, basis, tight=True)
    c = ref.ao_scale(res.basis)
    np.testing.assert_allclose(res.mo_energy, mf.mo_energy, atol=1e-7, rtol=0)
    np.testing.assert_allclose(res.density, ref.density_to_ours(mf.make_rdm1(), c), atol=1e-7, rtol=0)
    np.testing.assert_allclose(res.energy_weighted_density,
                               ref.density_to_ours(ref.energy_weighted_density(mf), c), atol=1e-7, rtol=0)
    np.testing.assert_allclose(res.dipole, mf.dip_moment(unit="AU", verbose=0), atol=1e-7, rtol=0)
    np.testing.assert_allclose(res.mulliken_charges, mf.mulliken_pop(verbose=0)[1], atol=1e-7, rtol=0)
    assert abs(res.mulliken_charges.sum()) < 1e-10           # neutral molecule
    # MOs are S-orthonormal and canonical for the returned Fock matrix
    S_ = I.overlap_matrix(res.basis)
    C = res.mo_coeff
    np.testing.assert_allclose(C.T @ S_ @ C, np.eye(res.nmo), atol=1e-10)
    np.testing.assert_allclose(C.T @ res.fock @ C, np.diag(res.mo_energy), atol=1e-10)


# ======================================================== UHF vs PySCF

UHF_CASES = [
    ("h", "sto-3g", 0, 2),            # one electron, n_beta = 0
    ("oh", "6-31g*", 0, 2),           # doublet radical
    ("nh2", "6-31g**", 0, 2),
    ("ch2", "cc-pvdz", 0, 3),         # triplet
    ("n2", "6-31g*", 0, 3),
    ("o2", "6-31g*", 0, 3),           # ground-state triplet, half-filled pi* shell
    ("no", "cc-pvdz", 0, 2),          # slow (pi* orbital rotation): ~20 iterations
    ("water", "6-31g", 1, 2),         # cations
    ("water", "cc-pvdz", 1, 2),
    ("hf", "6-31g*", 1, 2),
    ("h2s", "6-31g*", 1, 2),
    ("water", "sto-3g", -1, 2),       # anion
]


@pytest.mark.parametrize("name,basis,charge,mult", UHF_CASES)
def test_uhf_energy_and_spin_contamination_match_pyscf(name, basis, charge, mult):
    res = _ours(name, basis, charge, mult)
    mf = _pyscf(name, basis, charge, mult, unrestricted=True)
    assert res.converged and res.reference == "uhf"
    assert abs(res.energy - mf.e_tot) < 1e-8, res.energy - mf.e_tot
    assert abs(res.s2 - mf.spin_square()[0]) < 1e-6
    assert res.s2 >= res.s2_exact - 1e-10                    # UHF <S^2> >= S(S+1)
    assert res.n_alpha - res.n_beta == mult - 1
    assert res.n_alpha + res.n_beta == int(res.basis.atomic_numbers.sum()) - charge
    # same state: PySCF started from our density stays at our energy
    c = ref.ao_scale(res.basis)
    dm0 = np.array([ref.density_to_pyscf(D, c) for D in res.spin_densities])
    mf2 = ref.run_scf(mf.mol, unrestricted=True, dm0=dm0)
    assert abs(mf2.e_tot - res.energy) < 1e-8


@pytest.mark.parametrize("name,basis,charge,mult", [("ch2", "cc-pvdz", 0, 3), ("water", "6-31g", 1, 2)])
def test_uhf_spin_densities_and_properties_match_pyscf(name, basis, charge, mult):
    # Non-degenerate open shells; for OH the unpaired electron may sit in either
    # pi orbital, so its spin densities are only equal up to that rotation.
    res = _ours(name, basis, charge, mult, tight=True)
    mf = _pyscf(name, basis, charge, mult, unrestricted=True, tight=True)
    c = ref.ao_scale(res.basis)
    Dp = mf.make_rdm1()
    for s in range(2):
        np.testing.assert_allclose(res.spin_densities[s], ref.density_to_ours(Dp[s], c), atol=1e-7, rtol=0)
        np.testing.assert_allclose(res.mo_energy[s], mf.mo_energy[s], atol=1e-7, rtol=0)
        np.testing.assert_array_equal(res.mo_occ[s], mf.mo_occ[s])
    np.testing.assert_allclose(res.energy_weighted_density,
                               ref.density_to_ours(ref.energy_weighted_density(mf), c), atol=1e-7, rtol=0)
    np.testing.assert_allclose(res.dipole, mf.dip_moment(unit="AU", verbose=0), atol=1e-7, rtol=0)
    np.testing.assert_allclose(res.mulliken_charges, mf.mulliken_pop(verbose=0)[1], atol=1e-7, rtol=0)
    assert res.mulliken_charges.sum() == pytest.approx(charge, abs=1e-10)


def test_oh_radical_invariant_properties_match_pyscf():
    res = _ours("oh", "6-31g*", 0, 2, tight=True)
    mf = _pyscf("oh", "6-31g*", 0, 2, unrestricted=True, tight=True)
    np.testing.assert_allclose(res.dipole, mf.dip_moment(unit="AU", verbose=0), atol=1e-7, rtol=0)
    np.testing.assert_allclose(res.mulliken_charges, mf.mulliken_pop(verbose=0)[1], atol=1e-7, rtol=0)
    np.testing.assert_allclose(res.mo_energy, mf.mo_energy, atol=1e-7, rtol=0)


def test_uhf_of_closed_shell_equals_rhf():
    rhf = _ours("water", "6-31g*")
    uhf = _ours("water", "6-31g*", 0, 1, "uhf")
    assert uhf.reference == "uhf"
    assert abs(uhf.energy - rhf.energy) < 1e-10
    assert abs(uhf.s2) < 1e-10
    np.testing.assert_allclose(uhf.density, rhf.density, atol=1e-7)
    np.testing.assert_allclose(uhf.energy_weighted_density, rhf.energy_weighted_density, atol=1e-7)


@pytest.mark.parametrize("name,basis", [("oh", "6-31g"), ("cn", "6-31g*")])
def test_closed_shell_anion_rhf_matches_pyscf(name, basis):
    res = _ours(name, basis, -1, 1)
    mf = _pyscf(name, basis, -1, 1)
    assert res.reference == "rhf" and abs(res.energy - mf.e_tot) < 1e-8
    assert res.mulliken_charges.sum() == pytest.approx(-1.0, abs=1e-10)


@pytest.mark.parametrize("name,basis,charge,mult,max_iter", [
    ("water", "6-31g*", 0, 1, 100), ("oh", "6-31g*", 0, 2, 100),
    ("water", "6-31g*", 0, 1, 4), ("ch2", "cc-pvdz", 0, 3, 4)])     # the last two unconverged
def test_reported_energy_fock_and_commutator_are_those_of_the_returned_density(
        name, basis, charge, mult, max_iter):
    # Independent check of the convergence bookkeeping: PySCF's Fock matrix and
    # energy at our returned density, and the commutator in the Loewdin basis
    # of PySCF's overlap matrix. Measured: |dE| <= 2e-13, |dF| <= 1.4e-13, and
    # the commutator norms agree to 1.2e-6 relative (rounding of F ~1e-13
    # against commutators ~1e-8).
    sym, pos = ref.molecule(name)
    r = S.run_scf(sym, pos, basis, charge=charge, multiplicity=mult, max_iter=max_iter,
                  warn_unconverged=False)
    b = r.basis
    c = ref.ao_scale(b)
    mol = ref.pyscf_mole(b, charge=charge, spin=mult - 1)
    mf = (pyscf_scf.UHF if r.unrestricted else pyscf_scf.RHF)(mol)
    if r.unrestricted:
        dm = np.array([ref.density_to_pyscf(D, c) for D in r.spin_densities])
        Ds = r.spin_densities
    else:
        dm = ref.density_to_pyscf(r.density, c)
        Ds = 0.5 * r.density[None]                 # per-spin density, as documented
    F = ref.matrix_to_ours(np.asarray(mf.get_fock(dm=dm)), c)
    assert r.energy == pytest.approx(mf.energy_tot(dm=dm), abs=1e-10)
    np.testing.assert_allclose(r.fock, F, atol=1e-10, rtol=0)
    S_ = ref.matrix_to_ours(mol.intor("int1e_ovlp_cart"), c)
    s, U = np.linalg.eigh(S_)
    X = (U * s ** -0.5) @ U.T
    F3 = F.reshape(-1, b.nao, b.nao)
    comm = max(np.abs(X @ (F3[k] @ Ds[k] @ S_ - S_ @ Ds[k] @ F3[k]) @ X).max() for k in range(len(Ds)))
    assert r.commutator_norm == pytest.approx(comm, rel=1e-4)
    assert r.converged == (comm < 1e-7 and abs(r.energy_change) < 1e-10)
    assert r.converged == (max_iter == 100)


# =========================================== outputs needed by gradients

@pytest.mark.parametrize("name,basis,charge,mult,reference", [
    ("water", "6-31g*", 0, 1, None), ("formaldehyde", "6-31g**", 0, 1, None),
    ("oh", "6-31g*", 0, 2, None), ("ch2", "cc-pvdz", 0, 3, None)])
def test_gradient_assembled_from_scf_outputs_matches_pyscf(name, basis, charge, mult, reference):
    # density, energy_weighted_density and coulomb_exchange_densities in the
    # conventions of aimd.qc.integrals give PySCF's analytic gradient.
    res = _ours(name, basis, charge, mult, reference, tight=True)
    mf = _pyscf(name, basis, charge, mult, unrestricted=res.unrestricted, tight=True)
    b = res.basis
    g = (I.nuclear_repulsion_gradient(b.nuclear_charges, b.positions)
         + I.one_electron_gradient(b, res.density, res.energy_weighted_density)
         + I.two_electron_gradient(b, *res.coulomb_exchange_densities()))
    gp = mf.nuc_grad_method().kernel()
    # measured <= 2e-9 at commutator 1e-9
    np.testing.assert_allclose(g, gp, atol=2e-8, rtol=0)
    if not res.unrestricted:
        P, F = res.density, res.fock
        np.testing.assert_allclose(res.energy_weighted_density, 0.5 * P @ F @ P, atol=1e-8)
        Dc, Dx, k = res.coulomb_exchange_densities()
        assert k == 0.5 and len(Dx) == 1 and np.array_equal(Dc, P)
    else:
        Dc, Dx, k = res.coulomb_exchange_densities()
        assert k == 1.0 and len(Dx) == 2
        W = sum(D @ Fs @ D for D, Fs in zip(res.spin_densities, res.fock))
        np.testing.assert_allclose(res.energy_weighted_density, W, atol=1e-8)


# ======================================================== SAD guess

@pytest.mark.filterwarnings("ignore:remove_linear_dep_ is deprecated:DeprecationWarning")  # PySCF-internal
@pytest.mark.parametrize("basis", ["sto-3g", "6-31g"])
def test_sad_atomic_hf_matches_pyscf_atom_hf(basis):
    # s/p-only bases: Cartesian = spherical, so PySCF's spherically averaged,
    # fractionally occupied atomic HF (same configurations) is a direct reference.
    for el in ref.ELEMENTS_H_AR:
        b = build_basis([el], [[0.0, 0.0, 0.0]], basis)
        D, e, conv = S.atomic_hf(b)
        e_ref = atom_hf.get_atm_nrhf(ref.pyscf_mole(b))[el][0]
        assert conv and abs(e - e_ref) < 1e-9, (el, e - e_ref)
        assert np.trace(D @ I.overlap_matrix(b)) == pytest.approx(b.atomic_numbers[0], abs=1e-10)


@pytest.mark.parametrize("basis", ["6-31g*", "cc-pvdz"])
def test_sad_closed_shell_atoms_match_cartesian_rhf(basis):
    # PySCF's atom_hf forces a spherical basis; closed-shell atoms (integer
    # occupations) instead compare with plain Cartesian RHF.
    for el in ["He", "Be", "Ne", "Mg", "Ar"]:
        b = build_basis([el], [[0.0, 0.0, 0.0]], basis)
        _, e, conv = S.atomic_hf(b)
        assert conv and abs(e - ref.run_scf(ref.pyscf_mole(b)).e_tot) < 1e-9, el


def test_ground_state_configurations():
    # (s, p, d, f) electron counts of tabulated ground states
    table = {1: [1, 0, 0, 0], 3: [3, 0, 0, 0], 6: [4, 2, 0, 0], 8: [4, 4, 0, 0], 11: [5, 6, 0, 0],
             17: [6, 11, 0, 0], 18: [6, 12, 0, 0], 26: [8, 12, 6, 0], 36: [8, 18, 10, 0]}
    for Z, counts in table.items():
        assert S.ground_state_l_counts(Z) == counts, Z


def test_atomic_occupations_follow_the_configuration_not_the_orbital_order():
    # Li (1s2 2s1) with a p shell below the 2s level (as can happen in early
    # atomic iterations): the configuration decides, not plain aufbau.
    eps = np.array([-2.5, -0.3, -0.3, -0.3, -0.2, 0.4, 0.4, 0.4, 0.4, 0.4])
    occ = S._atomic_occupations(eps, S.ground_state_l_counts(3))
    np.testing.assert_array_equal(occ, [2, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    # C (2p2): the partly filled p shell is occupied evenly (spherical density)
    occ = S._atomic_occupations(np.array([-11.0, -0.7, -0.4, -0.4, -0.4, 0.5]), [4, 2, 0, 0])
    np.testing.assert_allclose(occ, [2, 2, 2 / 3, 2 / 3, 2 / 3, 0])
    # an accidental 2-fold degeneracy falls back to fractional aufbau
    occ = S._atomic_occupations(np.array([-2.0, -0.5, -0.5, 0.3]), [3, 0, 0, 0])
    np.testing.assert_allclose(occ, [2, 0.5, 0.5, 0])


def test_sad_density_is_block_diagonal_superposition():
    sym, pos = ref.molecule("ethanol")
    b = build_basis(sym, pos, "6-31g*")
    P = S.sad_density(b)
    S_ = I.overlap_matrix(b)
    np.testing.assert_array_equal(P, P.T)
    for a in range(b.natm):
        i, j = b.atom_ao[a]
        assert np.trace(P[i:j, i:j] @ S_[i:j, i:j]) == pytest.approx(b.atomic_numbers[a], abs=1e-10)
        mask = np.ones(b.nao, bool)
        mask[i:j] = False
        assert not P[i:j][:, mask].any()
    # same element, same block (cached atomic calculation)
    (i0, j0), (i1, j1) = b.atom_ao[0], b.atom_ao[1]
    np.testing.assert_array_equal(P[i0:j0, i0:j0], P[i1:j1, i1:j1])


def test_atom_basis_extracts_the_atom_shells():
    sym, pos = ref.molecule("water")
    b = build_basis(sym, pos, "6-31g*")
    ab = S.atom_basis(b, 0)
    ref_b = build_basis(["O"], [[0.0, 0.0, 0.0]], "6-31g*")
    for attr in ("shell_l", "shell_prim", "prim_exp", "prim_coef", "shell_atom"):
        np.testing.assert_array_equal(getattr(ab, attr), getattr(ref_b, attr))


# ==================================================== guesses and reuse

@pytest.mark.parametrize("name,basis,charge,mult", [("water", "6-31g", 0, 1), ("ch2", "6-31g*", 0, 3)])
def test_all_guesses_converge_to_the_same_state(name, basis, charge, mult):
    sym, pos = ref.molecule(name)
    solver = S.SCFSolver(sym, basis, charge=charge, multiplicity=mult)
    energies = {}
    for guess in ["sad", "core", "gwh"]:
        r = solver.run(pos, guess=guess)
        assert r.converged and r.guess == guess
        energies[guess] = r.energy
    r0 = solver.run(pos)
    assert r0.guess == "sad"                         # the default
    rd = solver.run(pos, guess=r0.guess_density)
    assert rd.guess == "density" and rd.iterations <= 3
    rr = solver.run(pos, guess=r0)
    assert rr.guess == "result" and rr.iterations <= 3
    for e in list(energies.values()) + [rd.energy, rr.energy]:
        assert abs(e - r0.energy) < 1e-9


@pytest.mark.parametrize("name,basis,mult", [("water", "6-31g*", 1), ("ethanol", "6-31g*", 1),
                                             ("oh", "6-31g*", 2)])
def test_guess_from_displaced_geometry_saves_iterations(name, basis, mult):
    sym, pos = ref.molecule(name)
    solver = S.SCFSolver(sym, basis, multiplicity=mult)
    r0 = solver.run(pos)
    new = pos + _displacement(pos, 0.01)
    r_default = solver.run(new)
    r_reuse = solver.run(new, guess=r0.guess_density)
    assert r_default.converged and r_reuse.converged
    assert abs(r_reuse.energy - r_default.energy) < 1e-9
    # measured: water 11 -> 8, ethanol 12 -> 9, OH 12 -> 10 iterations (PySCF
    # from its default / the same reused guess at a 1e-7 orbital-gradient
    # threshold: water 10 -> 8, ethanol 13 -> 9); the gain is limited by
    # the 1e-7 commutator target, not by the guess (start: 100x smaller)
    assert r_reuse.iterations <= r_default.iterations - 2
    assert r_reuse.history[0]["commutator"] < 0.01 * r_default.history[0]["commutator"]


def test_non_idempotent_guess_converges():
    # XL-BOMD hands over auxiliary densities that are not idempotent
    sym, pos = ref.molecule("water")
    solver = S.SCFSolver(sym, "6-31g*")
    r0 = solver.run(pos)
    r1 = solver.run(pos + _displacement(pos, 0.05, seed=3))
    mix = 0.6 * r0.density + 0.4 * r1.density + 1e-3 * ref.random_symmetric(solver.nao, np.random.default_rng(0))
    r = solver.run(pos, guess=mix)
    assert r.converged and abs(r.energy - r0.energy) < 1e-9


def test_density_guess_shapes():
    sym, pos = ref.molecule("water")
    rhf = S.SCFSolver(sym, "6-31g")
    uhf = S.SCFSolver(sym, "6-31g", charge=1, multiplicity=2)
    n = rhf.nao
    assert rhf.density_shape == (n, n) and uhf.density_shape == (2, n, n)
    r = rhf.run(pos)
    ru = uhf.run(pos)
    assert r.guess_density.shape == (n, n) and ru.guess_density.shape == (2, n, n)
    np.testing.assert_array_equal(r.guess_density, r.density)
    # RHF accepts an [alpha, beta] pair (summed), UHF a total density (split)
    r2 = rhf.run(pos, guess=np.array([0.5 * r.density, 0.5 * r.density]))
    ru2 = uhf.run(pos, guess=r.density)
    assert abs(r2.energy - r.energy) < 1e-9 and abs(ru2.energy - ru.energy) < 1e-9
    # the conversions themselves (the SCF would also converge from wrong ones):
    # RHF works with the per-spin density (Da + Db) / 2, UHF splits a total P evenly
    Da, Db = ru.spin_densities
    Ds, label = rhf.initial_spin_densities(rhf.integrals(pos), np.array([Da, Db]))
    assert label == "density"
    np.testing.assert_allclose(Ds, [0.5 * (Da + Db)], atol=1e-15)
    Ds, _ = uhf.initial_spin_densities(uhf.integrals(pos), r.density)
    np.testing.assert_allclose(Ds, [0.5 * r.density, 0.5 * r.density], atol=1e-15)
    Ds, label = uhf.initial_spin_densities(uhf.integrals(pos), ru)
    assert label == "result"
    np.testing.assert_array_equal(Ds, ru.spin_densities)
    with pytest.raises(ValueError, match="density guess has shape"):
        rhf.run(pos, guess=np.zeros((n + 1, n + 1)))
    with pytest.raises(ValueError, match="non-finite"):
        rhf.run(pos, guess=np.full((n, n), np.nan))
    with pytest.raises(ValueError, match="unknown guess"):
        rhf.run(pos, guess="minao")


# ============================================ robustness and convergence

@pytest.mark.parametrize("basis", ["6-31g*", "cc-pvdz"])
def test_stretched_water_converges_with_defaults(basis):
    sym, pos = _stretched_water(1.8)
    b = build_basis(sym, pos, basis)
    mf = ref.run_scf(ref.pyscf_mole(b))
    for guess in ["sad", "core", "gwh"]:
        r = S.run_scf(sym, pos, basis, guess=guess)
        assert r.converged and abs(r.energy - mf.e_tot) < 1e-8, (guess, r.energy - mf.e_tot)


def test_level_shift_damping_and_plain_roothaan_reach_the_same_solution():
    sym, pos = _stretched_water(1.8)
    e0 = S.run_scf(sym, pos, "6-31g*").energy
    for opts in [{"level_shift": 0.5}, {"damping": 0.5}, {"level_shift": 0.3, "damping": 0.3},
                 {"level_shift": 0.5, "stabilize_until": 1e-6}, {"diis_restart": 2}]:
        r = S.run_scf(sym, pos, "6-31g*", **opts)
        assert r.converged and abs(r.energy - e0) < 1e-9, opts
    sym, pos = ref.molecule("water")
    plain = S.run_scf(sym, pos, "sto-3g", diis=False, max_iter=200)
    assert plain.converged and abs(plain.energy - _ours("water", "sto-3g").energy) < 1e-9
    assert plain.iterations > _ours("water", "sto-3g").iterations     # DIIS accelerates


def test_level_shift_and_damping_stabilize_plain_roothaan():
    # Water with bonds x1.5 / 6-31G*: plain Roothaan iterations (no DIIS)
    # converge only linearly (commutator 2.4e-3 after 60 iterations, ~0.93 per
    # step). Measured: damping 0.5 converges in 26 iterations, a level shift of
    # 0.5 Eh kept on throughout in 47. The test above only shows that they do
    # not change the solution; this one shows that they act at all.
    sym, pos = _stretched_water(1.5)
    base = dict(diis=False, max_iter=60, warn_unconverged=False)
    plain = S.run_scf(sym, pos, "6-31g*", **base)
    assert not plain.converged
    e_ref = S.run_scf(sym, pos, "6-31g*").energy
    for opts in [{"damping": 0.5}, {"level_shift": 0.5, "stabilize_until": 1e-10}]:
        r = S.run_scf(sym, pos, "6-31g*", **base, **opts)
        assert r.converged and abs(r.energy - e_ref) < 1e-9, opts
    # both act only while the commutator exceeds stabilize_until: above every
    # commutator they never act, and the iterations are plain Roothaan's
    off = S.run_scf(sym, pos, "6-31g*", **base, level_shift=0.5, damping=0.5, stabilize_until=1e3)
    assert [h["energy"] for h in off.history] == [h["energy"] for h in plain.history]


def test_diis_restart_on_stall_still_converges_to_a_stationary_point():
    # Water at 2.5x bond length / STO-3G has several RHF solutions; from the SAD
    # guess the DIIS stalls (restarts with patience 4..12: 7, 3, 2, 1, 1).
    # Whatever solution is reached must be a genuine SCF solution: PySCF
    # started from it stays there.
    sym, pos = _stretched_water(2.5)
    r = S.run_scf(sym, pos, "sto-3g", diis_restart=6)
    assert r.converged and r.diis_restarts >= 1
    c = ref.ao_scale(r.basis)
    mf = ref.run_scf(ref.pyscf_mole(r.basis), dm0=ref.density_to_pyscf(r.density, c))
    assert abs(mf.e_tot - r.energy) < 1e-8


@functools.lru_cache(maxsize=None)
def _ethanol_cation():
    """Distorted ethanol cation / 6-31G* and its UHF ground-state energy from
    PySCF's second-order (Newton) solver; PySCF's DIIS does not converge here."""
    sym, pos = ref.molecule("ethanol", distort=0.2, seed=1)
    mol = ref.pyscf_mole(build_basis(sym, pos, "6-31g*"), charge=1, spin=1)
    mf = pyscf_scf.UHF(mol).newton()
    mf.conv_tol = 1e-11
    mf.kernel()
    assert mf.converged
    return sym, pos, mf.e_tot


def test_hard_open_shell_case_is_flagged_and_the_documented_fallback_converges():
    # DIIS from the SAD guess oscillates between hole configurations (measured:
    # not converged after 100 iterations, like PySCF's DIIS); it must be
    # flagged, and the GWH guess (module docstring) reaches the ground state.
    sym, pos, e_ref = _ethanol_cation()
    solver = S.SCFSolver(sym, "6-31g*", charge=1, multiplicity=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = solver.run(pos)
    warned = any(issubclass(w.category, S.SCFConvergenceWarning) for w in caught)
    assert r.converged != warned
    r = solver.run(pos, guess="gwh")
    assert r.converged and abs(r.energy - e_ref) < 1e-8


@pytest.mark.slow
def test_level_shifted_roothaan_converges_the_hard_case():
    sym, pos, e_ref = _ethanol_cation()
    r = S.run_scf(sym, pos, "6-31g*", charge=1, multiplicity=2, diis=False, level_shift=1.0,
                  stabilize_until=1e-8, max_iter=400)
    assert r.converged and abs(r.energy - e_ref) < 1e-8


def test_unconverged_runs_are_flagged_and_warned():
    sym, pos = ref.molecule("water")
    with pytest.warns(S.SCFConvergenceWarning, match="not converged after 3 iterations"):
        r = S.run_scf(sym, pos, "6-31g*", max_iter=3)
    assert not r.converged and r.iterations == 3 and len(r.history) == 3
    assert r.commutator_norm > 1e-7
    assert "NOT CONVERGED" in repr(r)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = S.run_scf(sym, pos, "6-31g*", max_iter=3, warn_unconverged=False)
    assert not r.converged
    # the unconverged state is self-consistent: energy = E[density]
    b = r.basis
    S_, T, V = I.one_electron_matrices(b)
    J, K = I.jk_from_eri(I.eri_tensor(b), r.density)
    H = T + V
    e = 0.5 * np.vdot(r.density, 2 * H + J - 0.5 * K) + r.nuclear_repulsion
    assert e == pytest.approx(r.energy, abs=1e-10)


def test_convergence_thresholds_are_respected():
    sym, pos = ref.molecule("water")
    loose = S.run_scf(sym, pos, "6-31g*", conv_energy=1e-6, conv_commutator=1e-4)
    tight = S.run_scf(sym, pos, "6-31g*", conv_energy=1e-12, conv_commutator=1e-10)
    assert loose.commutator_norm < 1e-4 and abs(loose.energy_change) < 1e-6
    assert tight.commutator_norm < 1e-10 and abs(tight.energy_change) < 1e-12
    assert loose.iterations < tight.iterations
    assert abs(loose.energy - tight.energy) < 1e-6
    h = tight.history
    assert len(h) == tight.iterations and h[-1]["energy"] == tight.energy
    assert np.isnan(h[0]["energy_change"])                      # undefined at iteration 1
    assert h[1]["energy_change"] == pytest.approx(h[1]["energy"] - h[0]["energy"], abs=1e-12)


# ===================================================== orthogonalization

def test_orthogonalizer_methods():
    sym, pos = ref.molecule("water")
    b = build_basis(sym, pos, "cc-pvdz")
    S_ = I.overlap_matrix(b)
    for method in ["auto", "symmetric", "canonical"]:
        X, nrem, smin = S.orthogonalizer(S_, method)
        np.testing.assert_allclose(X.T @ S_ @ X, np.eye(X.shape[1]), atol=1e-12)
        assert nrem == 0 and smin == pytest.approx(np.linalg.eigvalsh(S_)[0])
    Xs, _, _ = S.orthogonalizer(S_, "symmetric")
    np.testing.assert_allclose(Xs, Xs.T, atol=1e-12)           # Loewdin X is symmetric
    e = {m: S.run_scf(sym, pos, "cc-pvdz", orthogonalization=m).energy
         for m in ["symmetric", "canonical"]}
    assert abs(e["symmetric"] - e["canonical"]) < 1e-10


def _h2_basis(exponents):
    """H2 at 1.4 bohr with uncontracted s functions of the given exponents on each H."""
    n = len(exponents)
    return BasisSet("custom", ["H", "H"], [[0, 0, 0], [0, 0, 1.4]],
                    shell_atom=np.repeat([0, 1], n), shell_l=np.zeros(2 * n, int),
                    shell_prim=np.arange(2 * n + 1), prim_exp=np.tile(exponents, 2),
                    prim_coef=np.tile([(2 * a / np.pi) ** 0.75 for a in exponents], 2))


def test_canonical_orthogonalization_removes_linear_dependencies(monkeypatch):
    # An exactly duplicated function adds nothing: same energy as without it.
    pos = [[0, 0, 0], [0, 0, 1.4]]
    e_ref = S.run_scf(["H", "H"], pos, _h2_basis([3.0, 0.5])).energy
    dup = _h2_basis([3.0, 0.5, 0.5])
    r = S.run_scf(["H", "H"], pos, dup)                     # "auto" switches to canonical
    assert r.converged and r.n_removed == 2 and r.nmo == dup.nao - 2
    assert r.mo_coeff.shape == (dup.nao, dup.nao - 2)
    assert abs(r.energy - e_ref) < 1e-10
    with pytest.raises(ValueError, match="nearly linearly dependent"):
        S.run_scf(["H", "H"], pos, dup, orthogonalization="symmetric")
    # near-duplicate exponents (overlap eigenvalues 2.4e-11, 5.7e-11 < 1e-7): same
    # answer as PySCF with canonical orthogonalization at the same threshold
    near = _h2_basis([3.0, 0.5, 0.5 * (1 + 2e-5)])
    r = S.run_scf(["H", "H"], pos, near)
    assert r.n_removed == 2
    mol = gto.M(atom=[("H", (0, 0, 0)), ("H", (0, 0, 1.4))], unit="Bohr", verbose=0,
                basis={"H": [[0, [3.0, 1.0]], [0, [0.5, 1.0]], [0, [0.5 * (1 + 2e-5), 1.0]]]})
    # PySCF >= 2.13 removes overlap eigenvalues below this module threshold
    # (canonical orthogonalization) in every SCF
    monkeypatch.setattr(pyscf_scf.hf, "overlap_zero_eigenvalue_threshold", S.LINDEP_THRESHOLD)
    monkeypatch.setattr(pyscf_scf.hf, "remove_overlap_zero_eigenvalue", True)
    mf = ref.run_scf(mol)
    assert abs(r.energy - mf.e_tot) < 1e-9


# Even-tempered diffuse s and p shells added to 6-31G on every atom of water.
_DIFFUSE = [(0, 0.03), (0, 0.01), (0, 0.004), (0, 0.0015), (1, 0.03), (1, 0.01), (1, 0.004)]


@pytest.mark.parametrize("charge,mult", [(0, 1), (1, 2)])
def test_canonical_orthogonalization_of_a_diffuse_basis_matches_pyscf(charge, mult, monkeypatch):
    # A realistic near-linear dependence (diffuse functions on neighbouring
    # atoms), for RHF and UHF: overlap eigenvalues 1.20e-8 and 2.26e-8 lie
    # below the 1e-7 threshold, the next one (2.39e-7) above, so exactly two
    # combinations go. Keeping them (a threshold 1000x looser) would shift the
    # energy by 3.6e-5 (RHF) / 5.2e-6 Eh (UHF). Measured agreement with PySCF's
    # canonical orthogonalization at the same threshold: <= 4e-13 Eh.
    sym, pos = ref.molecule("water")
    data = {s: BASIS_SETS["6-31g"][s] + [[l, [e, 1.0]] for l, e in _DIFFUSE] for s in set(sym)}
    b, mol = ref.custom_basis(sym, pos, data, charge=charge, spin=mult - 1)
    s = np.linalg.eigvalsh(I.overlap_matrix(b))
    assert 1e-10 < s[0] and s[1] < S.LINDEP_THRESHOLD < s[2]
    r = S.run_scf(sym, pos, b, charge=charge, multiplicity=mult)
    assert r.converged and r.n_removed == 2 and r.nmo == b.nao - 2
    assert r.mo_coeff.shape[-2:] == (b.nao, b.nao - 2)
    monkeypatch.setattr(pyscf_scf.hf, "overlap_zero_eigenvalue_threshold", S.LINDEP_THRESHOLD)
    monkeypatch.setattr(pyscf_scf.hf, "remove_overlap_zero_eigenvalue", True)
    mf = ref.run_scf(mol, unrestricted=r.unrestricted)
    assert np.shape(mf.mo_coeff)[-1] == r.nmo
    assert abs(r.energy - mf.e_tot) < 1e-9, r.energy - mf.e_tot
    if r.unrestricted:
        assert abs(r.s2 - mf.spin_square()[0]) < 1e-6


def test_non_finite_positions_raise_instead_of_corrupting_memory():
    # Regression: a NaN coordinate (e.g. from an MD run that blew up) used to
    # abort the whole process with heap corruption. The primitive-pair kernel
    # counted pairs with "mu r^2 <= cutoff" but filled them unless "mu r^2 >
    # cutoff"; both are False for NaN, so it wrote past its arrays.
    sym, pos = ref.molecule("water")
    for bad in (np.nan, np.inf):
        p = pos.copy()
        p[1, 0] = bad
        with pytest.raises(ValueError, match="finite"):
            S.run_scf(sym, p, "sto-3g")
        with pytest.raises(ValueError, match="finite"):
            S.SCFSolver(sym, "6-31g").run(p)
    # the kernel itself (below the BasisSet check): pairs touching the NaN atom
    # are dropped and every other pair keeps exactly its own primitive pairs
    b = build_basis(sym, pos, "sto-3g")
    args = (b.shell_atom, b.shell_l, b.shell_prim, b.prim_exp, b.prim_coef)
    pp_ok, _, ab_ok, _, _ = I._build_pairs(*args, pos)
    p = pos.copy()
    p[1, 0] = np.nan
    pp_nan, shells, ab_nan, _, _ = I._build_pairs(*args, p)
    assert pp_nan[-1] == len(ab_nan)
    for ip, (i, j) in enumerate(shells):
        n_nan = pp_nan[ip + 1] - pp_nan[ip]
        if 1 in (b.shell_atom[i], b.shell_atom[j]):
            assert n_nan == 0
        else:
            np.testing.assert_array_equal(ab_nan[pp_nan[ip]:pp_nan[ip + 1]],
                                          ab_ok[pp_ok[ip]:pp_ok[ip + 1]])


# ======================================================= solver reuse

def test_solver_reuses_parsed_basis_and_caches_integrals():
    sym, pos = ref.molecule("water")
    solver = S.SCFSolver(sym, "6-31G*")
    new = pos + _displacement(pos, 0.05)
    r1 = solver.run(pos)
    r2 = solver.run(new, guess=r1)
    fresh = S.run_scf(sym, new, "6-31g*")
    assert abs(r2.energy - fresh.energy) < 1e-10
    np.testing.assert_array_equal(r2.basis.positions, new)
    assert np.shares_memory(r2.basis.prim_exp, solver.basis.prim_exp)   # no re-parsing
    assert np.shares_memory(r1.basis.prim_coef, r2.basis.prim_coef)
    ints = solver.integrals(new)
    assert solver.integrals(new.copy()) is ints                  # same geometry: cached
    assert solver.integrals(pos) is not ints
    # deterministic: same input, bitwise the same energy
    assert solver.run(new, guess=r1).energy == r2.energy
    solver.clear_cache()
    ints = solver.integrals(new)
    assert solver.integrals(new) is ints
    solver.options = S.SCFOptions(orthogonalization="canonical")
    assert solver.integrals(new) is not ints                     # options changed


def test_solver_accepts_a_custom_basis_set():
    pos = np.array([[0, 0, 0], [0, 0, 1.4]])
    b = _h2_basis([3.0, 0.5])
    r = S.SCFSolver(["H", "H"], b).run(pos)
    mol = gto.M(atom=[("H", (0, 0, 0)), ("H", (0, 0, 1.4))], unit="Bohr", verbose=0,
                basis={"H": [[0, [3.0, 1.0]], [0, [0.5, 1.0]]]})
    assert abs(r.energy - ref.run_scf(mol).e_tot) < 1e-10
    with pytest.raises(ValueError, match="do not match"):
        S.SCFSolver(["H", "He"], b)


# ========================================================== validation

def test_electron_count_validation():
    assert S.electron_counts(10, 0, 1) == (5, 5)
    assert S.electron_counts(10, 1, 2) == (5, 4)
    assert S.electron_counts(8, 0, 3) == (5, 3)
    assert S.electron_counts(1, 0, 2) == (1, 0)
    with pytest.raises(ValueError, match="multiplicity 2 is impossible with 10 electrons"):
        S.electron_counts(10, 0, 2)
    with pytest.raises(ValueError, match="multiplicity 1 is impossible with 9 electrons"):
        S.electron_counts(10, 1, 1)
    with pytest.raises(ValueError, match="needs at least 4 electrons"):
        S.electron_counts(2, 0, 5)
    with pytest.raises(ValueError, match="leaves 0 electrons"):
        S.electron_counts(2, 2, 1)
    with pytest.raises(ValueError, match="multiplicity must be >= 1"):
        S.electron_counts(2, 0, 0)
    with pytest.raises(ValueError, match="RHF needs a closed-shell singlet"):
        S.electron_counts(8, 0, 3, "rhf")
    sym, _ = ref.molecule("oh")
    with pytest.raises(ValueError, match="impossible"):
        S.SCFSolver(sym, "sto-3g")                     # OH with multiplicity 1
    with pytest.raises(ValueError, match="RHF needs"):
        S.SCFSolver(sym, "sto-3g", multiplicity=2, reference="rhf")
    with pytest.raises(ValueError, match="unknown reference"):
        S.SCFSolver(sym, "sto-3g", multiplicity=2, reference="rohf")
    assert S.SCFSolver(sym, "sto-3g", multiplicity=2).reference == "uhf"
    he2 = BasisSet("one s each", ["He", "He"], [[0, 0, 0], [0, 0, 2]], shell_atom=[0, 1],
                   shell_l=[0, 0], shell_prim=[0, 1, 2], prim_exp=[1.0, 1.0], prim_coef=[0.7, 0.7])
    with pytest.raises(ValueError, match="3 alpha electrons do not fit in 2 AOs"):
        S.SCFSolver(["He", "He"], he2, charge=-2)


def test_option_validation():
    with pytest.raises(TypeError, match="unknown SCF option"):
        S.SCFSolver(["H", "H"], "sto-3g", conv_tol=1e-8)
    opt = S.SCFSolver(["H", "H"], "sto-3g", options=S.SCFOptions(max_iter=7), conv_energy=1e-9).options
    assert opt.max_iter == 7 and opt.conv_energy == 1e-9 and opt.guess == "sad"
    for bad in [dict(max_iter=0), dict(conv_energy=0.0), dict(guess="minao"), dict(diis_space=1),
                dict(damping=1.0), dict(level_shift=-0.1), dict(orthogonalization="lowdin"),
                dict(blas_threads=0)]:
        with pytest.raises(ValueError):
            S.SCFOptions(**bad)


def test_diis_is_exact_for_a_linear_error_model():
    rng = np.random.default_rng(4)
    F_star = ref.random_symmetric(5, rng)[None]
    G = ref.random_symmetric(5, rng)[None]
    d = S.DIIS(4)
    d.push(F_star + G, G)                 # error proportional to F - F*
    d.push(F_star - 0.5 * G, -0.5 * G)
    np.testing.assert_allclose(d.extrapolate(), F_star, atol=1e-12)
    d.push(F_star - 0.5 * G, -0.5 * G)   # duplicate -> singular system, handled
    out = d.extrapolate()
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, F_star, atol=1e-10)
    for _ in range(6):
        d.push(F_star, np.zeros_like(G))
    assert len(d) == 4                    # bounded history
    np.testing.assert_allclose(d.extrapolate(), F_star, atol=1e-12)


def test_diis_drops_the_oldest_vectors_when_the_extrapolation_blows_up():
    # Two nearly identical error vectors that are not orthogonal to their
    # difference: the least-squares coefficients are ~1e7, beyond
    # max_coefficient, so the oldest vector is dropped and the newest Fock
    # matrix is returned unchanged.
    rng = np.random.default_rng(7)
    u = ref.random_symmetric(4, rng)[None]
    w = ref.random_symmetric(4, rng)[None]
    F_old, F_new = ref.random_symmetric(4, rng)[None], ref.random_symmetric(4, rng)[None]
    d = S.DIIS(4)
    d.push(F_old, u + 1e-7 * w)
    d.push(F_new, u)
    assert np.abs(d.coefficients()).max() > d.max_coefficient
    np.testing.assert_array_equal(d.extrapolate(), F_new)
    assert len(d) == 1


def test_blas_thread_limit_is_scoped():
    from aimd.qc.threads import blas_thread_counts, limit_blas_threads
    before = blas_thread_counts()
    if not before:
        pytest.skip("no OpenBLAS thread pool found in this process")
    with limit_blas_threads(1):
        assert all(k == 1 for k in blas_thread_counts())
    assert blas_thread_counts() == before
    with limit_blas_threads(None):
        assert blas_thread_counts() == before
    with pytest.raises(RuntimeError):
        with limit_blas_threads(1):
            raise RuntimeError
    assert blas_thread_counts() == before                       # restored on error
    # results do not depend on the BLAS thread setting
    sym, pos = ref.molecule("water")
    e1 = S.run_scf(sym, pos, "6-31g*", blas_threads=1).energy
    e2 = S.run_scf(sym, pos, "6-31g*", blas_threads=None).energy
    assert abs(e1 - e2) < 1e-11
