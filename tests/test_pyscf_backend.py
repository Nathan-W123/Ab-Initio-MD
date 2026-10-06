"""
PySCF backend: agreement with PySCF run directly, independent checks of the
gradients (finite differences, rigid-motion covariance, a literature value),
the dipole (finite electric field) and the density-guess protocol.
"""

import sys

import numpy as np
import pytest

pytest.importorskip("pyscf")
from pyscf import dft, gto, lib, mp, scf  # noqa: E402

from aimd.backends.pyscf_backend import PySCFBackend  # noqa: E402  (registers "pyscf")
from aimd.backends.registry import get_backend  # noqa: E402
from aimd.integrators import VelocityVerlet  # noqa: E402
from aimd.md import run_md  # noqa: E402
from aimd.testing import finite_difference_gradient  # noqa: E402

WATER = ["O", "H", "H"]
# Distorted water (bohr): no gradient component vanishes by symmetry.
X = np.array([
    [0.02, -0.03, 0.231098],
    [-0.05, 1.470523, -0.854392],
    [0.03, -1.410523, -0.944392],
])
TIGHT = dict(conv_tol=1e-12, conv_tol_grad=1e-9)


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    """One OpenMP thread: deterministic and fast for these tiny molecules."""
    old = lib.num_threads()
    lib.num_threads(1)
    yield
    lib.num_threads(old)


def pyscf_reference(method, basis="sto-3g", charge=0, spin=0, cart=False, xc=None,
                    grid_level=3, unrestricted=None):
    """The same calculation done directly with PySCF."""
    mol = gto.M(atom=[(a, c) for a, c in zip(WATER, X.tolist())], unit="Bohr",
                basis=basis, charge=charge, spin=spin, cart=cart, symmetry=False,
                verbose=0)
    unr = spin != 0 if unrestricted is None else unrestricted
    if xc is not None:
        mf = dft.UKS(mol) if unr else dft.RKS(mol)
        mf.xc = xc
        mf.grids.level = grid_level
    else:
        mf = scf.UHF(mol) if unr else scf.RHF(mol)
    mf.conv_tol = 1e-10
    mf.chkfile = None
    mf.kernel()
    if method == "mp2":
        pt = mp.UMP2(mf) if unr else mp.MP2(mf)
        pt.kernel()
        return pt.e_tot, pt.nuc_grad_method().kernel(), mf
    g = mf.nuc_grad_method()
    if xc is not None:
        g.grid_response = True
    return mf.e_tot, g.kernel(), mf


def test_registered_as_pyscf():
    assert get_backend("pyscf") is PySCFBackend
    assert get_backend("PySCF") is PySCFBackend
    assert PySCFBackend.supports_density_guess


# ── Same numbers as PySCF itself ─────────────────────────────────────────────

CASES = {
    "rhf": (dict(method="hf"), dict(method="hf"), "RHF", (7, 7)),
    "uhf": (dict(method="hf", charge=1, multiplicity=2),
            dict(method="hf", charge=1, spin=1), "UHF", (2, 7, 7)),
    "uhf-singlet": (dict(method="uhf"), dict(method="hf", unrestricted=True), "UHF", (2, 7, 7)),
    "rks": (dict(method="b3lyp"), dict(method="dft", xc="b3lyp"), "RKS(b3lyp)", (7, 7)),
    "uks": (dict(method="pbe", charge=1, multiplicity=2, grid_level=1),
            dict(method="dft", xc="pbe", charge=1, spin=1, grid_level=1), "UKS(pbe)", (2, 7, 7)),
    "mp2": (dict(method="mp2"), dict(method="mp2"), "RHF-MP2", (7, 7)),
    "ump2": (dict(method="mp2", charge=1, multiplicity=2),
             dict(method="mp2", charge=1, spin=1), "UHF-MP2", (2, 7, 7)),
    "cart-6-31g*": (dict(method="hf", basis="6-31g*", cart=True),
                    dict(method="hf", basis="6-31g*", cart=True), "RHF", (19, 19)),
    "sph-6-31g*": (dict(method="hf", basis="6-31g*"),
                   dict(method="hf", basis="6-31g*"), "RHF", (18, 18)),
}


@pytest.mark.parametrize("case", list(CASES))
def test_energy_gradient_density_dipole_equal_pyscf(case):
    kw, ref_kw, label, shape = CASES[case]
    backend = PySCFBackend(WATER, **kw)
    res = backend.compute(X)
    e_ref, g_ref, mf = pyscf_reference(**ref_kw)

    # Same code path and settings: equal up to OpenMP/BLAS summation order
    # (measured |dE| <= 4e-14 Eh, |dg| <= 8e-14 Eh/bohr, |dD| <= 9e-13).
    assert abs(res.energy - e_ref) < 1e-12
    assert np.max(np.abs(res.gradient - g_ref)) < 1e-12
    assert res.gradient.shape == (3, 3)
    assert res.density.shape == shape == backend.density_shape
    assert np.max(np.abs(res.density - mf.make_rdm1())) < 1e-11
    assert res.converged and res.info["scf_converged"]
    assert res.info["scf_iterations"] == mf.cycles
    assert res.info["method"] == label and res.info["guess"] == "init"
    if ref_kw["method"] == "mp2":
        assert res.dipole is None
        assert res.info["scf_energy"] == pytest.approx(mf.e_tot, abs=1e-12)
        assert res.info["mp2_correlation_energy"] < 0.0
    else:
        assert res.dipole.shape == (3,)
        dip = mf.dip_moment(mf.mol, mf.make_rdm1(), unit="AU", verbose=0)
        assert np.max(np.abs(res.dipole - dip)) < 1e-10


def test_literature_values_h2_sto3g():
    """Szabo & Ostlund, Modern Quantum Chemistry: H2/STO-3G at R = 1.4 bohr,
    E(HF) = -1.1167 Eh (Sec. 3.5.2) and E(2) = -0.0132 Eh (Ch. 6); the
    tolerance is half a unit in the last quoted digit."""
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    hf = PySCFBackend(["H", "H"], method="hf").compute(x)
    mp2 = PySCFBackend(["H", "H"], method="mp2").compute(x)
    assert abs(hf.energy - (-1.1167)) < 5e-5
    assert abs(mp2.info["mp2_correlation_energy"] - (-0.0132)) < 5e-5
    assert mp2.energy == pytest.approx(hf.energy + mp2.info["mp2_correlation_energy"], abs=1e-12)
    # By symmetry the force is along the bond; R = 1.4 bohr is longer than the
    # STO-3G bond length (1.346 bohr), so the atoms are pulled together.
    assert np.allclose(hf.gradient[:, :2], 0.0, atol=1e-12)
    assert hf.gradient[0, 2] == pytest.approx(-hf.gradient[1, 2], abs=1e-12)
    assert hf.gradient[1, 2] > 0.0


# ── Independent checks of the gradient ───────────────────────────────────────

@pytest.mark.parametrize("kw, tol", [
    # measured max |analytic - FD| (h = 1e-4 bohr): 1.6e-9, 1.6e-9, 1.1e-7
    (dict(method="hf"), 2e-8),
    (dict(method="hf", charge=1, multiplicity=2), 2e-8),
    # MP2: FD of the MP2 energy carries the SCF residual through the
    # non-variational correlation energy, hence the larger (measured) error.
    (dict(method="mp2"), 1e-6),
])
def test_gradient_matches_finite_differences(kw, tol):
    backend = PySCFBackend(WATER, **TIGHT, **kw)
    g = backend.compute(X).gradient
    fd = finite_difference_gradient(backend, X, step=1e-4)
    assert np.max(np.abs(g - fd)) < tol
    # Translation invariance: no net force.
    assert np.max(np.abs(g.sum(axis=0))) < 1e-10


def test_dft_gradient_includes_grid_response():
    """
    With atom-centred grids the quadrature energy depends on the grid motion.
    HF molecule, PBE/STO-3G, grid level 1 (coarse, which exaggerates the
    effect). Measured: with grid response |g - FD| = 1e-9 and |sum g| = 7e-16;
    without it 3.6e-4 and 4.2e-4.
    """
    x = np.array([[0.1, -0.2, 0.05], [0.3, 0.1, 1.75]])
    errors, net = {}, {}
    for response in (True, False):
        b = PySCFBackend(["F", "H"], method="pbe", grid_level=1,
                         grid_response=response, **TIGHT)
        g = b.compute(x).gradient
        errors[response] = np.max(np.abs(g - finite_difference_gradient(b, x, 1e-4)))
        net[response] = np.max(np.abs(g.sum(axis=0)))
    assert errors[True] < 1e-7 and net[True] < 1e-10
    assert errors[False] > 1e-5 and net[False] > 1e-5


def test_rigid_motion_and_atom_permutation_covariance():
    """
    E(Q x + t) = E(x) and g(Q x + t) = g(x) Q^T for a rotation Q: fails if
    PySCF reoriented or recentred the molecule, or if units were mixed up.
    Swapping the two H atoms swaps the gradient rows. Measured: |dE| ~ 1e-14,
    |dg| = 1.5e-10 (rotation), 7.5e-11 (permutation).
    """
    rng = np.random.default_rng(3)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))                  # proper rotation
    t = np.array([1.3, -0.7, 2.1])
    backend = PySCFBackend(WATER, **TIGHT)
    r0 = backend.compute(X)
    r1 = backend.compute(X @ q.T + t)
    assert r1.energy == pytest.approx(r0.energy, abs=1e-10)
    assert np.max(np.abs(r1.gradient - r0.gradient @ q.T)) < 1e-8
    # The dipole of a neutral molecule rotates and does not see the translation.
    assert np.max(np.abs(r1.dipole - r0.dipole @ q.T)) < 1e-8

    perm = [0, 2, 1]
    r2 = PySCFBackend(WATER, **TIGHT).compute(X[perm])
    assert r2.energy == pytest.approx(r0.energy, abs=1e-10)
    assert np.max(np.abs(r2.gradient - r0.gradient[perm])) < 1e-8


# ── Dipole and density ────────────────────────────────────────────────────────

def test_dipole_is_minus_the_electric_field_derivative():
    """
    mu = -dE/dF: add a uniform field F to PySCF's core Hamiltonian
    (electrons: +F.r, nuclei: -Z F.R) and difference the SCF energy. For HF
    the relaxed and the density dipole coincide. Measured error 5.6e-9 e bohr.
    """
    backend = PySCFBackend(WATER, **TIGHT)
    mu = backend.compute(X).dipole

    mol = gto.M(atom=[(a, c) for a, c in zip(WATER, X.tolist())], unit="Bohr",
                basis="sto-3g", verbose=0)
    r_ints = mol.intor("int1e_r")
    zr = mol.atom_charges() @ mol.atom_coords()

    def energy(field):
        mf = scf.RHF(mol)
        mf.conv_tol, mf.conv_tol_grad, mf.chkfile = 1e-12, 1e-9, None
        h = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        h = h + np.einsum("x,xij->ij", field, r_ints)
        mf.get_hcore = lambda *args: h
        return mf.kernel() - field @ zr

    f = 1e-4
    fd = np.array([-(energy(f * e) - energy(-f * e)) / (2 * f) for e in np.eye(3)])
    assert np.max(np.abs(mu - fd)) < 1e-7


def test_charged_dipole_shifts_by_charge_times_translation():
    """For a cation, mu(x + t) = mu(x) + q t about the fixed origin (measured
    error 1.1e-10)."""
    backend = PySCFBackend(WATER, charge=1, multiplicity=2, **TIGHT)
    t = np.array([0.4, -1.1, 0.7])
    mu0 = backend.compute(X).dipole
    mu1 = backend.compute(X + t).dipole
    assert np.max(np.abs(mu1 - mu0 - 1.0 * t)) < 1e-8


@pytest.mark.parametrize("kw, n_elec", [
    (dict(method="hf"), (10,)),
    (dict(method="hf", charge=1, multiplicity=2), (5, 4)),
    (dict(method="pbe", multiplicity=3, grid_level=1), (6, 4)),
])
def test_density_counts_electrons(kw, n_elec):
    backend = PySCFBackend(WATER, **kw)
    d = backend.compute(X).density
    mol = gto.M(atom=[(a, c) for a, c in zip(WATER, X.tolist())], unit="Bohr",
                basis="sto-3g", verbose=0)
    s = mol.intor("int1e_ovlp")
    blocks = d if d.ndim == 3 else d[None]
    assert len(blocks) == len(n_elec)
    for block, n in zip(blocks, n_elec):
        assert np.allclose(block, block.T, atol=1e-12)
        assert np.trace(block @ s) == pytest.approx(n, abs=1e-10)


# ── Density-guess protocol ───────────────────────────────────────────────────

def test_density_guess_is_used_for_the_next_call_only():
    backend = PySCFBackend(WATER)
    first = backend.compute(X)
    n_init = first.info["scf_iterations"]
    assert first.info["guess"] == "init" and n_init >= 5

    # The converged density as the guess: SCF done in one cycle.
    backend.set_density_guess(first.density)
    again = backend.compute(X)
    assert again.info["guess"] == "external"
    assert again.info["scf_iterations"] <= 2
    assert again.energy == pytest.approx(first.energy, abs=1e-10)
    # Both SCFs stop within conv_tol_grad = 1e-5 of the exact density;
    # measured gradient difference 2.8e-7 Eh/bohr.
    assert np.max(np.abs(again.gradient - first.gradient)) < 2e-6

    # The guess was consumed; the backend falls back to its last density.
    assert backend.compute(X).info["guess"] == "previous"

    # A deliberately poor guess (zero density = core-Hamiltonian start) is
    # actually used: more cycles than reusing the converged density, same answer.
    backend.set_density_guess(np.zeros((7, 7)))
    poor = backend.compute(X)
    assert poor.info["guess"] == "external"
    assert poor.info["scf_iterations"] > again.info["scf_iterations"] + 3
    assert poor.energy == pytest.approx(first.energy, abs=1e-9)


def test_guess_decides_a_capped_scf():
    """With one SCF cycle allowed the result is that of the guess: exact from
    the converged density, visibly wrong from the default MINAO guess."""
    exact = PySCFBackend(WATER, **TIGHT).compute(X)
    capped = PySCFBackend(WATER, max_cycle=1, reuse_density=False)
    cold = capped.compute(X)
    capped.set_density_guess(exact.density)
    warm = capped.compute(X)
    assert warm.energy == pytest.approx(exact.energy, abs=1e-10)
    assert abs(cold.energy - exact.energy) > 1e-4
    assert capped.compute(X).info["guess"] == "init"        # reuse disabled


def test_reuse_density_can_be_switched_off():
    # 6-31G: in a minimal basis the MINAO guess is already almost exact.
    on = PySCFBackend(WATER, basis="6-31g")
    off = PySCFBackend(WATER, basis="6-31g", reuse_density=False)
    for backend in (on, off):
        backend.compute(X)
    bent = X + np.array([[0.0, 0.0, 0.0], [0.0, 0.03, -0.02], [0.0, -0.01, 0.02]])
    r_on, r_off = on.compute(bent), off.compute(bent)
    assert r_on.info["guess"] == "previous" and r_off.info["guess"] == "init"
    assert r_on.info["scf_iterations"] < r_off.info["scf_iterations"]
    assert r_on.energy == pytest.approx(r_off.energy, abs=1e-9)
    on.reset_guess()
    assert on.compute(X).info["guess"] == "init"


def test_unrestricted_guess_keeps_alpha_and_beta_apart():
    """
    Stretched H2 (4 bohr), UHF singlet: the spin-symmetric guess stays on the
    RHF solution, a broken-symmetry guess (alpha on atom A, beta on B) reaches
    the lower UHF solution, the same one as PySCF started from that guess, and
    swapping the alpha and beta guesses flips the spin density. Fails if the
    backend summed, averaged or reordered the two spin blocks. Measured:
    E(RHF) = -0.76108, E(UHF, broken symmetry) = -0.93584 Eh.
    """
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    rhf = PySCFBackend(["H", "H"], **TIGHT).compute(x)
    uhf = PySCFBackend(["H", "H"], reference="uhf", reuse_density=False, **TIGHT)
    uhf.set_density_guess(np.stack([rhf.density / 2, rhf.density / 2]))
    sym = uhf.compute(x)
    on_a, on_b = np.diag([1.0, 0.0]), np.diag([0.0, 1.0])
    uhf.set_density_guess(np.stack([on_a, on_b]))
    ab = uhf.compute(x)
    uhf.set_density_guess(np.stack([on_b, on_a]))
    ba = uhf.compute(x)

    mol = gto.M(atom=[("H", (0, 0, 0)), ("H", (0, 0, 4.0))], unit="Bohr",
                basis="sto-3g", verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = TIGHT["conv_tol"], TIGHT["conv_tol_grad"]
    e_bs = mf.kernel(dm0=np.stack([on_a, on_b]))

    assert sym.energy == pytest.approx(rhf.energy, abs=1e-10)
    assert ab.energy == pytest.approx(e_bs, abs=1e-10)
    assert ba.energy == pytest.approx(e_bs, abs=1e-10)
    assert ab.energy < rhf.energy - 0.1
    spin_ab, spin_ba = ab.density[0] - ab.density[1], ba.density[0] - ba.density[1]
    assert spin_ab[0, 0] > 0.5 and spin_ab[1, 1] < -0.5
    assert np.allclose(spin_ba, -spin_ab, atol=1e-6)


def test_density_guess_shape_is_checked():
    rhf = PySCFBackend(WATER)
    uhf = PySCFBackend(WATER, charge=1, multiplicity=2)
    with pytest.raises(ValueError, match="expected"):
        rhf.set_density_guess(np.zeros((2, 7, 7)))
    with pytest.raises(ValueError, match="unrestricted"):
        uhf.set_density_guess(np.zeros((7, 7)))
    uhf.set_density_guess(np.zeros((2, 7, 7)))               # accepted


def test_velocity_verlet_reuses_the_previous_density(water):
    """
    Plain BOMD needs nothing from the integrator: the backend starts every SCF
    after the first from the previous density. Water/6-31G, 0.5 fs, orbital
    gradient converged to 1e-7: measured 8-9 cycles per step with reuse, 10
    without, and positions that agree to 1.3e-8 bohr after 6 steps. (Below an
    orbital-gradient threshold of ~1e-8 PySCF's DIIS reaches its noise floor
    and the cycle counts become erratic for any guess.)
    """
    runs = {}
    for reuse in (True, False):
        s = water.copy()
        s.initialize_velocities(300.0, rng=4)
        backend = PySCFBackend(s.symbols, basis="6-31g", reuse_density=reuse,
                               conv_tol=1e-12, conv_tol_grad=1e-7)
        integ = VelocityVerlet(backend, 0.5)
        infos = []
        run_md(s, integ, 6, callback=lambda rec: infos.append(dict(integ.result.info)))
        runs[reuse] = (s, infos)
    (s_on, on), (s_off, off) = runs[True], runs[False]
    assert on[0]["guess"] == "init" and all(i["guess"] == "previous" for i in on[1:])
    assert all(i["guess"] == "init" for i in off)
    for i_on, i_off in zip(on[1:], off[1:]):
        assert i_on["scf_iterations"] < i_off["scf_iterations"]
    # Both converge to the same tolerance: same dynamics.
    assert np.max(np.abs(s_on.positions - s_off.positions)) < 1e-7


# ── Unconverged SCF, errors ──────────────────────────────────────────────────

def test_unconverged_scf_is_flagged_not_raised(water):
    backend = PySCFBackend(WATER, max_cycle=2, reuse_density=False)
    res = backend.compute(X)
    assert not res.converged and not res.info["scf_converged"]
    assert res.info["scf_iterations"] == 2
    assert np.isfinite(res.energy) and np.all(np.isfinite(res.gradient))

    s = water.copy()
    s.initialize_velocities(300.0, rng=1)
    with pytest.warns(RuntimeWarning, match="4 of 4 force evaluations did not converge"):
        out = run_md(s, VelocityVerlet(backend, 0.5), 3)
    assert out.unconverged_steps == [0, 1, 2, 3]


@pytest.mark.filterwarnings("ignore:Basis may be available")
def test_invalid_arguments():
    with pytest.raises(ValueError, match="unknown method"):
        PySCFBackend(WATER, method="ccsd(t)")
    with pytest.raises(ValueError, match="could not build"):
        PySCFBackend(WATER, basis="no-such-basis")
    with pytest.raises(ValueError, match="could not build"):
        PySCFBackend(WATER, multiplicity=2)                  # 10 electrons
    with pytest.raises(ValueError, match="multiplicity 1"):
        PySCFBackend(WATER, method="rhf", multiplicity=3)
    with pytest.raises(ValueError, match="conflicts"):
        PySCFBackend(WATER, method="uhf", reference="rhf")
    with pytest.raises(ValueError, match="reference"):
        PySCFBackend(WATER, reference="rohf")
    with pytest.raises(ValueError, match="positive"):
        PySCFBackend(WATER, conv_tol=0.0)
    # Regression: PySCF's max_cycle = 0 gives E[guess] with a gradient of the
    # density from diagonalising F[guess] (-75.81 vs -75.94 Eh, water/6-31G).
    with pytest.raises(ValueError, match="max_cycle"):
        PySCFBackend(WATER, max_cycle=0)
    with pytest.raises(ValueError, match="no option"):
        PySCFBackend(WATER, scf_options={"not_an_option": 1}).compute(X)
    with pytest.raises(ValueError, match="expected 3 atoms"):
        PySCFBackend(WATER).compute(X[:2])
    b = PySCFBackend(WATER, method="b3lyp", reference="uhf")
    assert b.unrestricted and b.label == "UKS(b3lyp)" and b.density_shape == (2, 7, 7)


def test_user_scf_callback_is_kept():
    """Regression: the cycle counter used to replace a callback passed in
    scf_options, which was then silently never called."""
    seen = []
    backend = PySCFBackend(WATER, scf_options={"callback": lambda env: seen.append(env["cycle"])})
    res = backend.compute(X)
    assert seen == list(range(res.info["scf_iterations"])) and len(seen) >= 5


def test_threads_option_sets_pyscf_threads():
    old = lib.num_threads()
    try:
        PySCFBackend(WATER, threads=2)
        assert lib.num_threads() == 2
    finally:
        lib.num_threads(old)
    with pytest.raises(ValueError, match="threads"):
        PySCFBackend(WATER, threads=0)


def test_missing_pyscf_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyscf", None)          # import now fails
    with pytest.raises(RuntimeError, match="pip install pyscf"):
        PySCFBackend(WATER)
