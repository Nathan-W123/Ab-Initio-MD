"""
Native HF backend (aimd.backends.hf): agreement with the SCF solver it wraps
and with PySCF run directly (independent reference, same Cartesian basis
data), finite-difference gradients, the density-guess protocol, UHF for open
shells, unconverged-SCF handling, and NVE molecular dynamics.
"""

from __future__ import annotations

import dataclasses
import os
import warnings

import numpy as np
import pytest

import qc_reference as ref
from pyscf import lib

from aimd.backends.base import GradientResult
from aimd.backends.hf import HFBackend  # registers "hf"
from aimd.backends.registry import get_backend
from aimd.integrators import VelocityVerlet
from aimd.md import run_md
from aimd.qc import scf as S
from aimd.qc.basis import build_basis
from aimd.qc.gradients import scf_gradient
from aimd.qc.integrals import overlap_matrix
from aimd.testing import finite_difference_gradient

WATER = ["O", "H", "H"]
# Distorted water (bohr): no gradient component vanishes by symmetry.
X = np.array([
    [0.02, -0.03, 0.231098],
    [-0.05, 1.470523, -0.854392],
    [0.03, -1.410523, -0.944392],
])
TIGHT = dict(conv_tol=1e-12, conv_tol_grad=1e-9)


@pytest.fixture(autouse=True, scope="module")
def single_pyscf_thread():
    """One PySCF OpenMP thread: fast for these tiny reference calculations."""
    old = lib.num_threads()
    lib.num_threads(1)
    yield
    lib.num_threads(old)


def pyscf_reference(basis, charge=0, mult=1, unrestricted=False, symbols=WATER, x=X):
    """(energy, gradient, dipole, density in our AO normalization) from PySCF."""
    b = build_basis(symbols, x, basis)
    mol = ref.pyscf_mole(b, charge=charge, spin=mult - 1)
    mf = ref.run_scf(mol, unrestricted=unrestricted, conv_tol=1e-12, conv_tol_grad=1e-9)
    dm = mf.make_rdm1()
    dip = mf.dip_moment(mol, dm, unit="AU", verbose=0)
    return (mf.e_tot, mf.nuc_grad_method().kernel(), np.asarray(dip),
            ref.density_to_ours(np.asarray(dm), ref.ao_scale(b)), mf)


def test_registered_as_hf():
    assert get_backend("hf") is HFBackend
    assert get_backend(" HF ") is HFBackend
    assert HFBackend.supports_density_guess


# ── Same numbers as the SCF solver and as PySCF ──────────────────────────────

CASES = {
    "rhf-sto-3g": (dict(basis="sto-3g"), dict(basis="sto-3g"), "RHF", (7, 7)),
    "rhf-6-31g*": (dict(basis="6-31g*"), dict(basis="6-31g*"), "RHF", (19, 19)),
    "uhf-cation": (dict(basis="6-31g*", charge=1, multiplicity=2),
                   dict(basis="6-31g*", charge=1, mult=2, unrestricted=True), "UHF", (2, 19, 19)),
    "uhf-singlet": (dict(basis="sto-3g", reference="uhf"),
                    dict(basis="sto-3g", unrestricted=True), "UHF", (2, 7, 7)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_energy_gradient_density_dipole(case):
    kw, ref_kw, label, shape = CASES[case]
    backend = HFBackend(WATER, **TIGHT, **kw)
    res = backend.compute(X)
    assert isinstance(res, GradientResult)
    assert res.converged and res.info["scf_converged"]
    assert res.info["method"] == label and res.info["guess"] == "init"
    assert res.gradient.shape == (3, 3) and res.dipole.shape == (3,)
    assert res.density.shape == shape == backend.density_shape

    # The same solver + gradient called directly: identical numbers.
    solver = S.SCFSolver(WATER, kw["basis"], charge=kw.get("charge", 0),
                         multiplicity=kw.get("multiplicity", 1), reference=kw.get("reference"),
                         conv_energy=1e-12, conv_commutator=1e-9)
    direct = solver.run(X)
    assert res.energy == pytest.approx(direct.energy, abs=1e-13)
    np.testing.assert_allclose(res.gradient, scf_gradient(direct), rtol=0, atol=1e-13)
    np.testing.assert_allclose(res.density, direct.guess_density, rtol=0, atol=1e-13)
    assert res.info["scf_iterations"] == direct.iterations

    # PySCF on the same basis data. Measured: |dE| <= 1e-13 Eh, |dg| <=
    # 5.6e-10 Eh/bohr, |d dipole| <= 4.1e-9 e bohr, |dD| <= 2.1e-9 (both SCFs
    # stop at an orbital gradient ~1e-9).
    e_ref, g_ref, dip_ref, d_ref, _ = pyscf_reference(**ref_kw)
    assert res.energy == pytest.approx(e_ref, abs=1e-10)
    np.testing.assert_allclose(res.gradient, g_ref, rtol=0, atol=1e-8)
    np.testing.assert_allclose(res.dipole, dip_ref, rtol=0, atol=5e-8)
    np.testing.assert_allclose(res.density, d_ref, rtol=0, atol=5e-8)


def test_info_reports_energy_components_spin_and_timings():
    backend = HFBackend(WATER, basis="6-31g*", charge=1, multiplicity=2, **TIGHT)
    res = backend.compute(X)
    info = res.info
    assert info["scf_energy"] == res.energy
    assert info["electronic_energy"] + info["nuclear_repulsion"] == pytest.approx(res.energy, abs=1e-12)
    assert info["nuclear_repulsion"] > 0.0 > info["electronic_energy"]
    assert info["commutator_norm"] < 1e-9 and abs(info["energy_change"]) < 1e-12
    assert info["s2_exact"] == 0.75 and info["warnings"] == [] and info["n_removed"] == 0
    assert info["basis"] == "6-31g*" and info["reference"] == "uhf"
    assert info["mulliken_charges"].shape == (3,)
    assert info["mulliken_charges"].sum() == pytest.approx(1.0, abs=1e-10)   # cation
    t = info["timings"]
    assert set(t) == {"integrals", "guess", "scf_iterations", "scf", "gradient", "total"}
    assert all(v >= 0.0 for v in t.values())
    assert t["total"] >= t["scf"] + t["gradient"] - 1e-6
    # <S^2> of the doublet cation from PySCF (measured difference 5e-10).
    *_, mf = pyscf_reference("6-31g*", charge=1, mult=2, unrestricted=True)
    assert info["s2"] == pytest.approx(mf.spin_square()[0], abs=1e-8)
    assert info["s2"] > 0.75                                # spin contaminated
    # alpha / beta blocks hold n_alpha / n_beta electrons
    Smat = overlap_matrix(build_basis(WATER, X, "6-31g*"))
    assert np.trace(res.density[0] @ Smat) == pytest.approx(5.0, abs=1e-10)
    assert np.trace(res.density[1] @ Smat) == pytest.approx(4.0, abs=1e-10)


def test_literature_value_h2_sto3g():
    """Szabo & Ostlund, Modern Quantum Chemistry, sec. 3.5.2: H2/STO-3G at
    R = 1.4 bohr, E(HF) = -1.1167 Eh. R exceeds the STO-3G bond length (1.346
    bohr), so the atoms are pulled together along the bond."""
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    res = HFBackend(["H", "H"], **TIGHT).compute(x)
    assert abs(res.energy - (-1.1167)) < 5e-5
    assert np.allclose(res.gradient[:, :2], 0.0, atol=1e-12)
    assert res.gradient[0, 2] == pytest.approx(-res.gradient[1, 2], abs=1e-12)
    assert res.gradient[1, 2] > 0.0
    assert np.allclose(res.dipole, 0.0, atol=1e-10)


@pytest.mark.parametrize("kw", [dict(basis="sto-3g"), dict(basis="6-31g*", charge=1, multiplicity=2)],
                         ids=["rhf", "uhf-doublet"])
def test_gradient_matches_finite_differences(kw):
    backend = HFBackend(WATER, **TIGHT, **kw)
    g = backend.compute(X).gradient
    fd = finite_difference_gradient(backend, X, step=1e-4)
    # measured 1.9e-9 (RHF) and 2.4e-9 (UHF): the O(h^2) truncation error of
    # the three-point formula at h = 1e-4 bohr
    np.testing.assert_allclose(g, fd, rtol=0, atol=2e-8)
    assert np.abs(g.sum(axis=0)).max() < 1e-12             # no net force


def test_rigid_motion_covariance_and_charged_dipole():
    """
    E(Q x + t) = E(x), g(Q x + t) = g(x) Q^T; the dipole rotates, and for a
    cation also shifts by q t about the fixed origin. The second SCF starts
    from the first one's density (not rotated), so the two differ by the SCF
    convergence error: measured |dE| 9e-14 Eh, |dg| <= 2.0e-9 Eh/bohr,
    |d dipole| <= 3.9e-9 e bohr.
    """
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    t = np.array([1.3, -0.7, 2.1])
    for charge, mult in ((0, 1), (1, 2)):
        backend = HFBackend(WATER, basis="6-31g*", charge=charge, multiplicity=mult, **TIGHT)
        r0 = backend.compute(X)
        r1 = backend.compute(X @ q.T + t)
        assert r1.energy == pytest.approx(r0.energy, abs=1e-10)
        np.testing.assert_allclose(r1.gradient, r0.gradient @ q.T, rtol=0, atol=1e-8)
        np.testing.assert_allclose(r1.dipole, r0.dipole @ q.T + charge * t, rtol=0, atol=1e-8)


def test_custom_basis_set_object():
    basis = build_basis(WATER, np.zeros((3, 3)), "6-31g")
    custom = HFBackend(WATER, basis=basis, **TIGHT).compute(X)
    named = HFBackend(WATER, basis="6-31G", **TIGHT).compute(X)
    assert custom.energy == pytest.approx(named.energy, abs=1e-12)
    np.testing.assert_allclose(custom.gradient, named.gradient, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="symbols"):
        HFBackend(["O", "H", "F"], basis=basis)


# ── Density-guess protocol ───────────────────────────────────────────────────

def test_density_guess_is_used_for_the_next_call_only():
    backend = HFBackend(WATER)
    first = backend.compute(X)
    n_init = first.info["scf_iterations"]
    assert first.info["guess"] == "init" and first.info["init_guess"] == "sad" and n_init >= 5

    # The converged density as the guess: done after two Fock builds (the
    # second confirms the energy change).
    backend.set_density_guess(first.density)
    again = backend.compute(X)
    assert again.info["guess"] == "external"
    assert again.info["scf_iterations"] <= 2
    assert again.energy == pytest.approx(first.energy, abs=1e-10)
    # both stop below the commutator threshold 1e-7 (measured 3e-8 Eh/bohr)
    np.testing.assert_allclose(again.gradient, first.gradient, rtol=0, atol=3e-7)

    # Consumed: the next call falls back to the previous density.
    assert backend.compute(X).info["guess"] == "previous"

    # A poor guess (zero density = core-Hamiltonian start) is really used:
    # more iterations (measured 17 vs 2), same state.
    backend.set_density_guess(np.zeros((7, 7)))
    poor = backend.compute(X)
    assert poor.info["guess"] == "external"
    assert poor.info["scf_iterations"] > n_init
    assert poor.energy == pytest.approx(first.energy, abs=1e-9)

    # reset_guess forgets the stored density: back to the initial guess.
    backend.reset_guess()
    assert backend.compute(X).info["guess"] == "init"


def test_guess_decides_a_capped_scf():
    """With one Fock build allowed the result is that of the guess: exact
    from the converged density, visibly wrong from the SAD guess."""
    exact = HFBackend(WATER, basis="6-31g", **TIGHT).compute(X)
    capped = HFBackend(WATER, basis="6-31g", max_cycles=1, reuse_density=False)
    cold = capped.compute(X)
    assert not cold.converged
    capped.set_density_guess(exact.density)
    warm = capped.compute(X)
    assert warm.energy == pytest.approx(exact.energy, abs=1e-10)
    np.testing.assert_allclose(warm.gradient, exact.gradient, rtol=0, atol=1e-8)
    assert abs(cold.energy - exact.energy) > 1e-3
    assert capped.compute(X).info["guess"] == "init"         # reuse disabled


def test_reuse_density_saves_iterations_in_md(water):
    """
    Water/6-31G*, 0.5 fs: measured 9 SCF iterations per step with the
    previous density as guess, 11 from SAD, and positions that agree to 8e-8
    bohr after 6 steps (both SCFs converge to the same tolerance). (In
    STO-3G the SAD guess is already so good that reuse saves nothing.)
    """
    runs = {}
    for reuse in (True, False):
        s = water.copy()
        s.initialize_velocities(300.0, rng=4)
        backend = HFBackend(s.symbols, basis="6-31g*", reuse_density=reuse)
        integ = VelocityVerlet(backend, 0.5)
        infos = []
        run_md(s, integ, 6, callback=lambda rec: infos.append(dict(integ.result.info)))
        runs[reuse] = (s, infos)
    (s_on, on), (s_off, off) = runs[True], runs[False]
    assert on[0]["guess"] == "init" and all(i["guess"] == "previous" for i in on[1:])
    assert all(i["guess"] == "init" for i in off)
    for i_on, i_off in zip(on[1:], off[1:]):
        assert i_on["scf_iterations"] < i_off["scf_iterations"]
    assert np.max(np.abs(s_on.positions - s_off.positions)) < 1e-6


def test_unrestricted_guess_keeps_alpha_and_beta_apart():
    """
    Stretched H2 (4 bohr), UHF singlet: the spin-symmetric guess stays on the
    RHF solution, a broken-symmetry guess (alpha on atom A, beta on B)
    reaches the lower UHF solution, the same as PySCF from that guess, and
    swapping the alpha and beta guesses flips the spin density. Fails if the
    backend summed, averaged or reordered the spin blocks. Measured: E(RHF)
    = -0.76108, E(UHF, broken symmetry) = -0.93584 Eh.
    """
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    rhf = HFBackend(["H", "H"], **TIGHT).compute(x)
    uhf = HFBackend(["H", "H"], reference="uhf", reuse_density=False, **TIGHT)
    uhf.set_density_guess(np.stack([rhf.density / 2, rhf.density / 2]))
    sym = uhf.compute(x)
    on_a, on_b = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
    uhf.set_density_guess(np.stack([on_a, on_b]))
    ab = uhf.compute(x)
    uhf.set_density_guess(np.stack([on_b, on_a]))
    ba = uhf.compute(x)

    mol = ref.pyscf_mole(build_basis(["H", "H"], x, "sto-3g"), spin=0)
    mf = ref.run_scf(mol, unrestricted=True, conv_tol=1e-12, conv_tol_grad=1e-9,
                     dm0=np.stack([on_a, on_b]))
    assert sym.energy == pytest.approx(rhf.energy, abs=1e-10)
    assert ab.energy == pytest.approx(mf.e_tot, abs=1e-10)
    assert ba.energy == pytest.approx(mf.e_tot, abs=1e-10)
    assert ab.energy < rhf.energy - 0.1
    assert ab.info["s2"] > 0.5 and sym.info["s2"] == pytest.approx(0.0, abs=1e-10)
    spin_ab, spin_ba = ab.density[0] - ab.density[1], ba.density[0] - ba.density[1]
    assert spin_ab[0, 0] > 0.5 and spin_ab[1, 1] < -0.5
    np.testing.assert_allclose(spin_ba, -spin_ab, atol=1e-6)


def test_density_guess_shape_and_values_are_checked():
    rhf = HFBackend(WATER)
    uhf = HFBackend(WATER, charge=1, multiplicity=2)
    with pytest.raises(ValueError, match="expected"):
        rhf.set_density_guess(np.zeros((2, 7, 7)))
    with pytest.raises(ValueError, match="unrestricted"):
        uhf.set_density_guess(np.zeros((7, 7)))
    with pytest.raises(ValueError, match="non-finite"):
        rhf.set_density_guess(np.full((7, 7), np.nan))
    uhf.set_density_guess(np.zeros((2, 7, 7)))               # accepted
    assert uhf.density_shape == (2, 7, 7) and rhf.density_shape == (7, 7)


def test_returned_density_is_not_the_stored_guess():
    # Modifying a returned density must not change the next SCF's guess.
    backend = HFBackend(WATER, basis="6-31g")
    r0 = backend.compute(X)
    r0.density[:] = 0.0                                      # caller scribbles on it
    r1 = backend.compute(X)
    assert r1.info["guess"] == "previous" and r1.info["scf_iterations"] <= 2


# ── Unconverged SCF, errors ──────────────────────────────────────────────────

def test_unconverged_scf_is_flagged_not_raised(water):
    backend = HFBackend(WATER, max_cycles=2, reuse_density=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # no per-call warning either
        res = backend.compute(X)
    assert not res.converged and not res.info["scf_converged"]
    assert res.info["scf_iterations"] == 2
    assert len(res.info["warnings"]) == 1 and "not converged" in res.info["warnings"][0]
    assert np.isfinite(res.energy) and np.all(np.isfinite(res.gradient))
    assert np.all(np.isfinite(res.density)) and np.all(np.isfinite(res.dipole))

    s = water.copy()
    s.initialize_velocities(300.0, rng=1)
    with pytest.warns(RuntimeWarning, match="4 of 4 force evaluations did not converge"):
        out = run_md(s, VelocityVerlet(backend, 0.5), 3)
    assert out.unconverged_steps == [0, 1, 2, 3]


def test_unconverged_density_is_the_next_previous_guess():
    """The last iterate of an unconverged SCF seeds the next call (documented
    "previous" semantics), and that call can then converge."""
    capped = HFBackend(WATER, basis="6-31g", max_cycles=4)
    r0 = capped.compute(X)
    assert not r0.converged
    capped.solver.options = dataclasses.replace(capped.options, max_iter=100)
    r1 = capped.compute(X)
    assert r1.info["guess"] == "previous" and r1.converged and r1.info["warnings"] == []
    exact = HFBackend(WATER, basis="6-31g", **TIGHT).compute(X)
    assert r1.energy == pytest.approx(exact.energy, abs=1e-9)


def test_linear_dependence_is_reported_in_the_warnings():
    """
    Canonical orthogonalization with a threshold above the smallest overlap
    eigenvalue (0.069 for water / 6-31G at X) drops one combination; the
    backend must say that the analytic gradient is then not dE/dR. Measured
    against central differences: 1.6e-2 Eh/bohr off, so the warning matters.
    """
    backend = HFBackend(WATER, basis="6-31g", **TIGHT,
                        scf_options={"orthogonalization": "canonical", "lindep_threshold": 0.1})
    res = backend.compute(X)
    assert res.converged and res.info["n_removed"] == 1
    assert len(res.info["warnings"]) == 1 and "linearly-dependent" in res.info["warnings"][0]
    fd = finite_difference_gradient(backend, X, step=1e-4)
    assert np.abs(fd - res.gradient).max() > 1e-3


def test_invalid_arguments():
    with pytest.raises(ValueError, match="Unknown basis"):
        HFBackend(WATER, basis="no-such-basis")
    with pytest.raises(ValueError, match="multiplicity"):
        HFBackend(WATER, multiplicity=2)                      # 10 electrons
    with pytest.raises(ValueError, match="closed-shell"):
        HFBackend(WATER, reference="rhf", charge=1, multiplicity=2)
    with pytest.raises(ValueError, match="reference"):
        HFBackend(WATER, reference="rohf")
    with pytest.raises(ValueError, match="conflicts"):
        HFBackend(WATER, method="uhf", reference="rhf")
    with pytest.raises(ValueError, match="Hartree-Fock only"):
        HFBackend(WATER, method="b3lyp")
    with pytest.raises(ValueError, match="guess"):
        HFBackend(WATER, guess="minao")
    with pytest.raises(ValueError, match="positive"):
        HFBackend(WATER, conv_tol=0.0)
    with pytest.raises(ValueError, match="max_iter"):
        HFBackend(WATER, max_cycles=0)
    with pytest.raises(ValueError, match="conv_tol_grad"):
        HFBackend(WATER, scf_options={"conv_commutator": 1e-8})
    with pytest.raises(ValueError, match="scf_options"):
        HFBackend(WATER, scf_options={"not_an_option": 1})
    with pytest.raises(ValueError, match="gradient_screening"):
        HFBackend(WATER, gradient_screening=-1.0)
    with pytest.raises(ValueError, match="threads"):
        HFBackend(WATER, threads=0)
    backend = HFBackend(WATER)
    with pytest.raises(ValueError, match="shape"):
        backend.compute(X[:2])
    with pytest.raises(ValueError, match="finite"):
        backend.compute(np.full((3, 3), np.nan))
    b = HFBackend(WATER, method="uhf", scf_options={"diis_space": 6, "level_shift": 0.1})
    assert b.unrestricted and b.label == "UHF" and b.options.diis_space == 6


def test_threads_option_applies_during_compute_only(monkeypatch):
    import numba
    import aimd.backends.hf as hf_module
    seen = []

    def spy(*args, **kwargs):                                 # runs inside compute
        seen.append(numba.get_num_threads())
        return scf_gradient(*args, **kwargs)

    monkeypatch.setattr(hf_module, "scf_gradient", spy)
    before = numba.get_num_threads()
    one = HFBackend(WATER, threads=1)
    r1 = one.compute(X)
    assert seen == [1] and numba.get_num_threads() == before  # restored
    default = HFBackend(WATER).compute(X)
    assert seen == [1, before]
    if before > 1:
        # the integral kernels are bitwise independent of the thread count
        assert r1.energy == default.energy and np.array_equal(r1.gradient, default.gradient)


def test_close_releases_the_cache():
    backend = HFBackend(WATER)
    backend.compute(X)
    assert backend.solver._ints is not None and backend.last_scf is not None
    backend.close()
    assert backend.solver._ints is None and backend.last_scf is None
    assert backend.compute(X).info["guess"] == "init"        # usable after close


# ── Molecular dynamics ───────────────────────────────────────────────────────

def _nve(water, dt_fs, n_steps):
    s = water.copy()
    s.initialize_velocities(300.0, rng=7)
    L0, P0 = s.angular_momentum(), s.momentum()
    backend = HFBackend(s.symbols, basis="sto-3g", **TIGHT)
    integ = VelocityVerlet(backend, dt_fs)
    iters = []
    out = run_md(s, integ, n_steps, callback=lambda rec: iters.append(integ.result.info["scf_iterations"]))
    assert not out.unconverged_steps
    e = out.column("total_Eh")
    return (np.abs(e - e[0]).max(), np.abs(s.angular_momentum() - L0).max(),
            np.abs(s.momentum() - P0).max(), out)


def test_nve_water_conserves_energy_and_momenta(water):
    """
    NVE water, HF/STO-3G, 300 K Maxwell-Boltzmann start (seed 7), tight SCF,
    50 fs (about 6 O-H stretch periods).

    Measured max |E_tot(t) - E_tot(0)|: 9.5e-5 Eh at dt = 0.5 fs (100 steps)
    and 2.4e-5 Eh at 0.25 fs (200 steps), a ratio of 3.98: the deviation is
    the O(dt^2) oscillation of velocity Verlet's shadow energy (STO-3G puts
    the O-H stretch at ~4100 cm^-1, period 8 fs, so 0.5 fs is 1/16 of a
    period), not an inconsistency between energy and forces, which would not
    shrink with dt. A force error of 1e-5 Eh/bohr would add ~1e-5 Eh over
    the ~1.5 bohr path of an H atom, visibly breaking the dt^2 ratio. For
    scale, E_pot varies by 5e-3 Eh along the run.

    Verlet conserves the total and angular momentum exactly for forces with
    zero net force and torque, so their drift measures the translational
    (measured 6e-13) and rotational (1.1e-6 au, from the residual SCF error:
    1.3e-4 with the default 1e-7 orbital-gradient threshold) invariance of
    the gradient; |L| = 2.7 au here.
    """
    dev_05, dl_05, dp_05, out = _nve(water, 0.5, 100)
    dev_025, dl_025, dp_025, _ = _nve(water, 0.25, 200)
    assert np.ptp(out.column("potential_Eh")) > 4e-3          # it really moved
    assert dev_05 < 1.5e-4
    assert dev_025 < 4e-5
    assert 3.5 < dev_05 / dev_025 < 4.5                       # O(dt^2) Verlet error
    assert max(dp_05, dp_025) < 1e-10
    assert max(dl_05, dl_025) < 1e-5


def test_openmp_wait_policy_default_respects_the_user(monkeypatch):
    """aimd.qc defaults libgomp to passive waiting (aimd.qc.threads) unless set."""
    from aimd.qc.threads import prefer_passive_openmp_wait
    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    monkeypatch.delenv("GOMP_SPINCOUNT", raising=False)
    assert prefer_passive_openmp_wait()
    assert os.environ["OMP_WAIT_POLICY"] == "PASSIVE"
    monkeypatch.setenv("OMP_WAIT_POLICY", "active")
    assert not prefer_passive_openmp_wait() and os.environ["OMP_WAIT_POLICY"] == "active"
    monkeypatch.delenv("OMP_WAIT_POLICY")
    monkeypatch.setenv("GOMP_SPINCOUNT", "1000")
    assert not prefer_passive_openmp_wait() and "OMP_WAIT_POLICY" not in os.environ


def test_coincident_nuclei_are_refused_not_reported_as_converged():
    # Regression: O(0,0,0), H(1.8,0,0), H(1.8,0,0) returned energy=+inf, a NaN
    # gradient and converged=True (the electronic integrals stay finite, so the
    # SCF converged; only Z_A Z_B / R_AB blew up). PySCF raises "Ill geometry"
    # below 1e-5 bohr; so do the backend and the SCF driver now.
    for gap in (0.0, 5e-6):
        x = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [1.8 + gap, 0.0, 0.0]])
        with warnings.catch_warnings():
            warnings.simplefilter("error")             # no divide-by-zero on the way
            with pytest.raises(ValueError, match=r"atoms 1 and 2 .*\(coincident nuclei"):
                HFBackend(WATER, basis="sto-3g").compute(x)
            with pytest.raises(ValueError, match="coincident"):
                S.run_scf(WATER, x, "sto-3g")
    # just above the threshold the geometry is (absurd but) finite: no refusal
    x = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [1.8, 1e-4, 0.0]])
    S.check_nuclear_separation(x, np.array([8, 1, 1]))
    # a ghost (Z = 0) centre may sit on an atom, as in PySCF
    S.check_nuclear_separation(np.zeros((2, 3)), np.array([1, 0]))
