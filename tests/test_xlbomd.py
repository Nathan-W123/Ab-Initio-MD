"""
XL-BOMD (Niklasson et al., J. Chem. Phys. 130, 214109 (2009)).

  - the published coefficients: moment conditions and a stable, damped
    auxiliary recursion (characteristic roots from numpy);
  - the auxiliary-density update against a companion-matrix evolution and an
    exactly tracked linear-in-time density;
  - the integrator's call sequence on a model backend whose forces do not
    depend on the guess (then XL-BOMD must be velocity Verlet, bit for bit);
  - with PySCF (water, HF/STO-3G): same dynamics as BOMD for a tight SCF, fewer
    SCF cycles, much smaller energy drift for a loose SCF (the point of the
    method), and exact restarts from checkpoints.
"""

from pathlib import Path

import numpy as np
import pytest

from aimd.backends.morse import MorseBackend
from aimd.checkpoint import load_checkpoint
from aimd.integrators import CSVR, XLBOMD, VelocityVerlet, integrator_from_state
from aimd.md import run_md
from aimd.system import MolecularSystem
from aimd.thermostats import CSVRThermostat, NoseHooverChainThermostat
from aimd.xlbomd import XL_COEFFICIENTS, AuxiliaryDensity, xl_coefficients

ORDERS = sorted(XL_COEFFICIENTS)
WATER_XYZ = Path(__file__).resolve().parent / "data" / "water_experimental.xyz"


def _recursion(k: int) -> np.ndarray:
    """a_j with e(t+dt) = sum_j a_j e(t - j dt) for a static SCF density."""
    kappa, alpha, c = xl_coefficients(k)
    a = alpha * c
    a[0] += 2.0 - kappa
    a[1] -= 1.0
    return a


# ── The coefficient table ────────────────────────────────────────────────────

@pytest.mark.parametrize("k", ORDERS)
def test_coefficients_have_the_published_moment_structure(k):
    """
    sum c_k = sum k c_k = 0 (a density constant or linear in time is not
    damped), and the odd moments, the time-irreversible part of the
    dissipation, vanish below order 2K - 3. A wrong digit in Table I breaks this.
    """
    c = XL_COEFFICIENTS[k][2]                      # Python ints: exact sums

    def moment(p):
        return sum(ck * j**p for j, ck in enumerate(c))

    assert len(c) == k + 1
    assert moment(0) == 0 and moment(1) == 0
    for p in range(3, 2 * k - 3, 2):
        assert moment(p) == 0, p
    assert moment(2 * k - 3) != 0


def test_coefficients_values_and_lookup():
    kappa, alpha, c = xl_coefficients(5)
    assert (kappa, alpha) == (1.82, 0.018)
    assert c.tolist() == [-6, 14, -8, -3, 4, -1]
    # alpha decreases with K: weaker dissipation, better reversibility.
    alphas = [xl_coefficients(k)[1] for k in ORDERS]
    assert alphas == sorted(alphas, reverse=True)
    for bad in (2, 10, 5.5):
        with pytest.raises(ValueError, match="K must be one of"):
            xl_coefficients(bad)
        with pytest.raises(ValueError, match="K must be one of"):
            AuxiliaryDensity(bad)


@pytest.mark.parametrize("k", ORDERS)
def test_auxiliary_recursion_is_stable_and_damped(k):
    """
    For a fixed SCF density D, e = P - D obeys a linear recursion whose
    characteristic roots must lie inside the unit circle (decaying
    oscillation; measured spectral radius 0.63 for K = 3 rising to 0.99 for
    K = 9), while without dissipation the Verlet oscillator (0 < kappa < 4)
    sits on the unit circle. AuxiliaryDensity must match the companion-matrix
    evolution of the same recursion.
    """
    kappa, _, _ = xl_coefficients(k)
    a = _recursion(k)
    rho = np.max(np.abs(np.roots(np.concatenate([[1.0], -a]))))
    assert rho < 1.0
    assert np.allclose(np.abs(np.roots([1.0, -(2.0 - kappa), 1.0])), 1.0)

    rng = np.random.default_rng(k)
    d = rng.normal(size=(3, 3))
    d = d + d.T
    hist = d + rng.normal(size=(k + 1, 3, 3))
    aux = AuxiliaryDensity(k)
    aux.load_state_dict({"k": k, "history": hist})
    comp = np.zeros((k + 1, k + 1))
    comp[0] = a
    comp[1:, :-1] = np.eye(k)
    e = (hist - d).reshape(k + 1, -1)
    for _ in range(80):
        aux.propagate(d)
        e = comp @ e
    got = np.stack(aux.history) - d
    assert np.allclose(got.reshape(k + 1, -1), e, rtol=0.0, atol=1e-10)


def test_density_linear_in_time_is_followed_exactly():
    """If D(t) = A + B t and the history lies on that line, P(t + dt) = D(t + dt)."""
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(2, 2, 4, 4))
    for k in ORDERS:
        aux = AuxiliaryDensity(k)
        aux.load_state_dict({"k": k, "history": np.stack([a - j * b for j in range(k + 1)])})
        for n in range(1, 25):
            p = aux.propagate(a + (n - 1) * b)        # D at the current time n - 1
            assert np.allclose(p, a + n * b, rtol=0.0, atol=1e-11)


def test_reset_and_state_round_trip():
    aux = AuxiliaryDensity(4)
    assert not aux.initialized
    with pytest.raises(RuntimeError):
        aux.propagate(np.eye(2))
    d0 = np.arange(8.0).reshape(2, 2, 2)                 # an unrestricted density
    aux.reset(d0)
    assert all(np.array_equal(p, d0) for p in aux.history) and len(aux.history) == 5
    p1 = aux.propagate(d0 + 1.0)
    assert np.array_equal(aux.current, p1)
    state = aux.state_dict()
    assert state["history"].shape == (5, 2, 2, 2)
    other = AuxiliaryDensity(4)
    other.load_state_dict(state)
    assert np.array_equal(other.propagate(d0), aux.propagate(d0))
    with pytest.raises(ValueError, match="shape"):
        aux.propagate(np.eye(2))
    with pytest.raises(ValueError, match="K=4"):
        AuxiliaryDensity(5).load_state_dict(state)
    with pytest.raises(ValueError, match="holds 4 densities, expected 5"):
        AuxiliaryDensity(4).load_state_dict({"k": 4, "history": state["history"][:-1]})


# ── The integrator on a model backend ────────────────────────────────────────

class GuessRecordingMorse(MorseBackend):
    """
    Morse forces plus a fake SCF density D(R) that does not depend on the
    guess; records the order of set_density_guess / compute calls.
    """

    supports_density_guess = True

    def __init__(self, symbols, unrestricted=False, **kw):
        super().__init__(symbols, **kw)
        self.unrestricted = unrestricted
        self.calls = []

    def density(self, positions):
        x = np.asarray(positions).ravel()[:4]
        d = np.outer(x, x) + np.eye(4)
        return np.stack([d, 0.5 * d]) if self.unrestricted else d

    def compute(self, positions):
        res = super().compute(positions)
        res.density = self.density(positions)
        self.calls.append(("compute", res.density.copy()))
        return res

    def set_density_guess(self, density):
        self.calls.append(("guess", np.array(density)))


def _h4(h4, seed=2):
    s = h4.copy()
    s.initialize_velocities(600.0, rng=seed, remove_rotation=True)
    return s


@pytest.mark.parametrize("unrestricted", [False, True])
def test_guess_sequence_and_dynamics_on_a_guess_independent_backend(h4, unrestricted):
    n, k = 25, 5
    s_xl, s_vv = _h4(h4), _h4(h4)
    backend = GuessRecordingMorse(s_xl.symbols, unrestricted=unrestricted)
    integ = XLBOMD(backend, 0.2, k=k)
    run_md(s_xl, integ, n)
    run_md(s_vv, VelocityVerlet(MorseBackend(s_vv.symbols), 0.2), n)
    # Forces do not depend on the guess: identical to velocity Verlet.
    assert np.array_equal(s_xl.positions, s_vv.positions)
    assert np.array_equal(s_xl.velocities, s_vv.velocities)

    # One initial SCF, then a guess before every force call.
    kinds = [c[0] for c in backend.calls]
    assert kinds == ["compute"] + ["guess", "compute"] * n
    scf = [c[1] for c in backend.calls if c[0] == "compute"]
    guesses = [c[1] for c in backend.calls if c[0] == "guess"]
    # Independent re-implementation of the recursion; history starts at D(t0),
    # and the guess for step m + 1 is built from the density of step m.
    kappa, alpha, c = xl_coefficients(k)
    hist = [scf[0]] * (k + 1)
    for m in range(n):
        p = (2 * hist[0] - hist[1] + kappa * (scf[m] - hist[0])
             + alpha * sum(ck * pk for ck, pk in zip(c, hist)))
        assert np.allclose(guesses[m], p, rtol=0.0, atol=1e-12), m
        hist = [p] + hist[:-1]
    assert np.allclose(guesses[0], scf[0], rtol=0.0, atol=1e-13)   # first guess = D(t0)


def test_thermostatted_xlbomd_is_thermostatted_velocity_verlet(h4):
    s_xl, s_vv = _h4(h4), _h4(h4)
    xl = XLBOMD(GuessRecordingMorse(s_xl.symbols), 0.2,
                thermostat=CSVRThermostat(300.0, tau_fs=10.0, rng=5))
    vv = CSVR(MorseBackend(s_vv.symbols), 0.2, 300.0, tau_fs=10.0, rng=5)
    r_xl = run_md(s_xl, xl, 40)
    r_vv = run_md(s_vv, vv, 40)
    assert np.array_equal(s_xl.positions, s_vv.positions)
    assert np.array_equal(r_xl.column("conserved_Eh"), r_vv.column("conserved_Eh"))
    assert xl.temperature_k == 300.0
    b = XLBOMD(GuessRecordingMorse(s_xl.symbols), 0.2, k=3, temperature_k=300.0,
               berendsen_tau_fs=20.0)
    assert b.config() == {"timestep_fs": 0.2, "k": 3, "thermostat": {
        "type": "BerendsenThermostat", "config": b.thermostat.config()}}


def test_xlbomd_rejects_backends_without_density_guesses(h4):
    with pytest.raises(ValueError, match="density guesses.*'morse'"):
        XLBOMD(MorseBackend(h4.symbols), 0.2)
    with pytest.raises(ValueError, match="K must be one of"):
        XLBOMD(GuessRecordingMorse(h4.symbols), 0.2, k=2)

    class NoDensity(GuessRecordingMorse):
        def compute(self, positions):
            res = super().compute(positions)
            res.density = None
            return res

    with pytest.raises(RuntimeError, match="no SCF density"):
        run_md(h4.copy(), XLBOMD(NoDensity(h4.symbols), 0.2), 1)


def test_xlbomd_state_is_validated_and_rebuilt(h4, tmp_path):
    s = _h4(h4)
    integ = XLBOMD(GuessRecordingMorse(s.symbols), 0.2, k=6,
                   thermostat=CSVRThermostat(300.0, tau_fs=10.0, rng=1))
    run_md(s, integ, 5, checkpoint_path=tmp_path / "x.ckpt")
    ckpt = load_checkpoint(tmp_path / "x.ckpt")
    assert ckpt.integrator_state["xl"]["history"].shape == (7, 4, 4)
    rebuilt = ckpt.make_integrator(GuessRecordingMorse(s.symbols))
    assert type(rebuilt) is XLBOMD and rebuilt.config() == integ.config()
    assert all(np.array_equal(p, q) for p, q in zip(rebuilt.aux.history, integ.aux.history))
    assert type(integrator_from_state(GuessRecordingMorse(s.symbols), integ.state_dict())) is XLBOMD

    state = integ.state_dict()
    with pytest.raises(ValueError, match="K=6"):
        XLBOMD(GuessRecordingMorse(s.symbols), 0.2, k=5,
               thermostat=CSVRThermostat(300.0, rng=1)).load_state_dict(state)
    with pytest.raises(ValueError, match="XLBOMD"):
        VelocityVerlet(MorseBackend(s.symbols), 0.2,
                       thermostat=CSVRThermostat(300.0, rng=1)).load_state_dict(state)
    # A rejected state leaves the integrator untouched.
    plain = XLBOMD(GuessRecordingMorse(s.symbols), 0.2, k=6)
    with pytest.raises(ValueError, match="thermostat"):
        plain.load_state_dict(state)
    assert plain.result is None and not plain.aux.initialized
    bad = {**state, "xl": {"k": 6, "history": state["xl"]["history"][:, :2, :2]}}
    with pytest.raises(ValueError, match="does not match"):
        XLBOMD(GuessRecordingMorse(s.symbols), 0.2, k=6,
               thermostat=CSVRThermostat(300.0, rng=1)).load_state_dict(bad)


class ShapeCheckingMorse(GuessRecordingMorse):
    """Rejects a guess of the wrong shape, as the density-guess protocol requires."""

    def set_density_guess(self, density):
        want = (2, 4, 4) if self.unrestricted else (4, 4)
        if np.shape(density) != want:
            raise ValueError(f"density guess has shape {np.shape(density)}, expected {want}")
        super().set_density_guess(density)


def test_state_rejected_late_leaves_the_integrator_untouched(h4):
    """
    Regression: the backend's check of the saved density (restricted
    checkpoint, unrestricted backend) and the thermostat's own checks (chain
    length) used to fail after ``result`` had already been replaced, leaving a
    half-loaded integrator whose next step would have used the checkpoint's
    forces at the wrong positions.
    """
    s = _h4(h4)
    xl = XLBOMD(ShapeCheckingMorse(s.symbols), 0.2)
    run_md(s, xl, 3)
    state = xl.state_dict()
    backend = ShapeCheckingMorse(s.symbols, unrestricted=True)
    other = XLBOMD(backend, 0.2)
    with pytest.raises(ValueError, match="expected"):
        other.load_state_dict(state)
    assert other.result is None and not other.aux.initialized and backend.calls == []

    nhc = VelocityVerlet(ShapeCheckingMorse(s.symbols), 0.2,
                         thermostat=NoseHooverChainThermostat(300.0, chain_length=3))
    run_md(s.copy(), nhc, 2)
    longer = VelocityVerlet(ShapeCheckingMorse(s.symbols), 0.2,
                            thermostat=NoseHooverChainThermostat(300.0, chain_length=4))
    with pytest.raises(ValueError, match="chain"):
        longer.load_state_dict(nhc.state_dict())
    assert longer.result is None and longer.backend.calls == []
    assert np.all(longer.thermostat.xi == 0.0)
    longer.load_state_dict({**nhc.state_dict(), "thermostat": longer.thermostat.state_dict()})
    assert longer.result is not None                       # a valid state still loads


# ── PySCF: the physics ───────────────────────────────────────────────────────

TIGHT = dict(conv_tol=1e-12, conv_tol_grad=1e-8)
LOOSE = dict(conv_tol=1e-5)          # conv_tol_grad = sqrt(conv_tol) = 3e-3
CATION = dict(charge=1, multiplicity=2)


@pytest.fixture(scope="module")
def pyscf_md():
    """
    run(kind, n_steps, seed=1, **backend_kw) -> (MDResult, positions per step,
    SCF cycles per step), water HF/STO-3G, dt = 0.5 fs, 300 K, cached.
    """
    pytest.importorskip("pyscf")
    from pyscf import lib

    from aimd.backends.pyscf_backend import PySCFBackend

    old = lib.num_threads()
    lib.num_threads(1)               # deterministic and fastest for 7 AOs
    cache = {}

    def run(kind, n_steps, seed=1, **kw):
        key = (kind, n_steps, seed, tuple(sorted(kw.items())))
        if key not in cache:
            charge, mult = kw.pop("charge", 0), kw.pop("multiplicity", 1)
            s = MolecularSystem.from_xyz(WATER_XYZ, charge, mult)
            s.initialize_velocities(300.0, rng=seed, remove_rotation=True)
            backend = PySCFBackend(s.symbols, charge, mult, method="hf",
                                   basis="sto-3g", **kw)
            integ = XLBOMD(backend, 0.5) if kind == "xl" else VelocityVerlet(backend, 0.5)
            pos, cycles = [], []

            def spy(rec):
                pos.append(s.positions.copy())
                cycles.append(integ.result.info["scf_iterations"])

            res = run_md(s, integ, n_steps, callback=spy)
            cache[key] = (res, np.array(pos), np.array(cycles[1:]))
        return cache[key]

    yield run
    lib.num_threads(old)


def _drift_rate(res) -> float:
    """Slope of a straight-line fit of E_tot(t), hartree / ps."""
    return float(np.polyfit(res.column("time_fs"), res.column("total_Eh"), 1)[0]) * 1000.0


def test_tight_scf_xlbomd_follows_bomd(pyscf_md):
    """
    With the SCF converged tightly the guess only changes the cycle count:
    measured over 200 steps (100 fs) max |dx| = 3.4e-8 bohr (6e-9 at step
    100), max |dE_pot| = 1.4e-10 Eh; the tolerances are ~10x the measurement.
    """
    bo, x_bo, _ = pyscf_md("bo", 200, **TIGHT)
    xl, x_xl, _ = pyscf_md("xl", 200, **TIGHT)
    assert np.max(np.abs(x_xl - x_bo)) < 3e-7
    assert np.max(np.abs(xl.column("potential_Eh") - bo.column("potential_Eh"))) < 2e-9
    assert xl.total_energy_drift == pytest.approx(bo.total_energy_drift, rel=1e-3)


@pytest.mark.parametrize("kind_kw, measured", [
    ((200, TIGHT), (8.76, 7.48)),
    ((100, dict(conv_tol=1e-8)), (4.43, 3.60)),
    ((50, CATION), (7.46, 6.54)),        # UHF, default thresholds
])
def test_xl_guess_needs_fewer_scf_cycles(pyscf_md, kind_kw, measured):
    """
    Mean SCF cycles per step, BOMD (previous density) vs XL-BOMD; measured
    values in the parameters. The XL guess is ~4x closer to the converged
    density (Frobenius |P - D| 0.004 vs 0.017, both references). Caveat, not
    tested: for the UHF cation with conv_tol_grad = 1e-8 PySCF's DIIS is near
    its noise floor; XL then took 13.7 cycles per step vs 13.5 over 200 steps
    (14.1 vs 12.6 over the first 50; counts erratic, up to 30).
    """
    n, kw = kind_kw
    _, _, c_bo = pyscf_md("bo", n, **kw)
    _, _, c_xl = pyscf_md("xl", n, **kw)
    assert c_xl.mean() <= c_bo.mean() - 0.5
    assert c_bo.mean() == pytest.approx(measured[0], abs=1.0)


@pytest.mark.parametrize("kw, measured", [
    (dict(**LOOSE), (-5.6e-3, -4.6e-5)),
    (dict(**LOOSE, **CATION), (-1.06e-2, -2.2e-4)),
])
def test_loose_scf_xlbomd_has_much_smaller_energy_drift(pyscf_md, kw, measured):
    """
    The raison d'etre of XL-BOMD. SCF converged to only 1e-5 Eh (|g| < 3e-3),
    200 steps of 0.5 fs, 300 K. BOMD restarting each SCF from the previous
    density makes a systematic force error and the total energy drifts; with
    the time-reversible XL guess it does not.

    Measured (closed shell / cation):
      drift rate, straight-line fit (Eh/ps)
        BOMD             -5.6e-3 / -1.06e-2
        XL-BOMD          -4.6e-5 / -2.2e-4
        tight-SCF BOMD   -3.8e-5 / -1.2e-4   (Verlet fluctuation, not drift)
      mean of the last 40 steps - mean of the first 40 (Eh)
        BOMD             -4.5e-4 / -8.6e-4
        XL-BOMD          -1.8e-6 / -1.9e-5
      SCF cycles per step: BOMD 2.02 / 2.53, XL-BOMD 1.99 / 2.00.
    Over seeds 1-3 at 300 and 600 K the closed-shell BOMD rate stayed within
    -4.7e-3 .. -6.4e-3 Eh/ps and |XL / BOMD| <= 0.015.
    """
    bo, _, c_bo = pyscf_md("bo", 200, **kw)
    xl, _, c_xl = pyscf_md("xl", 200, **kw)
    rate_bo, rate_xl = _drift_rate(bo), _drift_rate(xl)
    assert rate_bo == pytest.approx(measured[0], rel=0.5)    # the drift is there
    assert abs(rate_xl) < 0.1 * abs(rate_bo)
    assert abs(rate_xl) < 5.0 * abs(measured[1])
    # Same SCF effort (both ~2 cycles per step): the gain is not bought with cycles.
    assert c_xl.mean() <= c_bo.mean()
    e_bo, e_xl = bo.column("total_Eh"), xl.column("total_Eh")
    shift_bo = e_bo[-40:].mean() - e_bo[:40].mean()
    shift_xl = e_xl[-40:].mean() - e_xl[:40].mean()
    assert abs(shift_xl) < 0.1 * abs(shift_bo)


def test_loose_scf_xlbomd_energy_stays_at_the_verlet_level(pyscf_md):
    """
    The XL-BOMD total-energy fluctuation with the loose SCF is that of the
    tight-SCF run (measured max |E - E0|: 9.50e-5 vs 9.15e-5 Eh, closed
    shell), while loose-SCF BOMD has drifted to 6.1e-4 Eh after 100 fs.
    """
    tight, _, _ = pyscf_md("bo", 200, **TIGHT)
    xl, _, _ = pyscf_md("xl", 200, **LOOSE)
    bo, _, _ = pyscf_md("bo", 200, **LOOSE)
    assert xl.total_energy_drift < 1.25 * tight.total_energy_drift
    assert bo.total_energy_drift > 4.0 * tight.total_energy_drift


@pytest.mark.parametrize("case", ["rhf", "uhf", "rhf-csvr", "rhf-bomd", "rhf-bomd-noreuse"])
def test_restart_is_exact_with_pyscf(case, tmp_path):
    """
    N steps + checkpoint + restart (fresh backend and integrator) + N steps
    reproduces 2N steps to 1e-10, with a loose SCF so that the result depends
    on the guess: on the restored density history for XL-BOMD, on the saved
    density handed back to the backend for BOMD ("rhf-bomd"). Control: a
    restart without that guess information changes the positions after N
    more steps by far more (measured 1.2e-5, 1.1e-4, 3.8e-5 bohr for rhf,
    uhf, rhf-bomd).

    Regression ("rhf-bomd-noreuse"): a backend with reuse_density=False starts
    every SCF from its initial guess, so the restart must not hand it the saved
    density; it used to, and the restarted run differed by 3.1e-5 bohr. The
    control there is that old behaviour.
    """
    pytest.importorskip("pyscf")
    from pyscf import lib

    from aimd.backends.pyscf_backend import PySCFBackend

    old = lib.num_threads()
    lib.num_threads(1)
    try:
        charge, mult = (1, 2) if case == "uhf" else (0, 1)

        def start():
            s = MolecularSystem.from_xyz(WATER_XYZ, charge, mult)
            s.initialize_velocities(300.0, rng=3, remove_rotation=True)
            return s

        def backend():
            return PySCFBackend(["O", "H", "H"], charge, mult, conv_tol=1e-5,
                                reuse_density=case != "rhf-bomd-noreuse")

        def integrator(b):
            if case.startswith("rhf-bomd"):
                return VelocityVerlet(b, 0.5)
            thermo = CSVRThermostat(300.0, tau_fs=20.0, rng=8) if "csvr" in case else None
            return XLBOMD(b, 0.5, thermostat=thermo)

        n = 15
        ref_sys = start()
        ref = run_md(ref_sys, integrator(backend()), 2 * n)

        s = start()
        first = run_md(s, integrator(backend()), n, checkpoint_path=tmp_path / "xl.ckpt")
        ckpt = load_checkpoint(tmp_path / "xl.ckpt")
        if not case.startswith("rhf-bomd"):
            shape = (6, 2, 7, 7) if case == "uhf" else (6, 7, 7)
            assert ckpt.integrator_state["xl"]["history"].shape == shape
        second = run_md(ckpt.system, ckpt.make_integrator(backend()), n, restart=ckpt)

        joined = first.records + second.records[1:]
        for key in ref.records[0]:
            a = np.array([r[key] for r in joined])
            assert np.allclose(a, ref.column(key), rtol=0.0, atol=1e-10), key
        assert np.max(np.abs(ckpt.system.positions - ref_sys.positions)) <= 1e-10
        assert np.max(np.abs(ckpt.system.velocities - ref_sys.velocities)) <= 1e-10

        # Control: same checkpoint without the guess information (XL: history
        # replaced by the saved SCF density; BOMD: backend's guess dropped;
        # no-reuse BOMD: the saved density handed over as the old code did).
        b3 = backend()
        integ3 = ckpt.make_integrator(b3)
        sys3 = start()
        ckpt.restore(sys3, integ3)
        if case == "rhf-bomd":
            b3.reset_guess()
        elif case == "rhf-bomd-noreuse":
            b3.set_density_guess(integ3.result.density)
        else:
            integ3.aux.reset(integ3.result.density)
        run_md(sys3, integ3, n, start_step=n)
        assert np.max(np.abs(sys3.positions - ref_sys.positions)) > 1e-8
    finally:
        lib.num_threads(old)


def test_restricted_history_is_rejected_by_an_unrestricted_non_reusing_backend():
    """
    Regression: with reuse_density=False the restart no longer hands the saved
    density to the backend (which validated its shape), so XL-BOMD checks its
    history against the backend's ``density_shape`` itself. Before, a
    restricted checkpoint loaded into an unrestricted run and failed only in
    the next step, after the system had been moved.
    """
    pytest.importorskip("pyscf")
    from aimd.backends.pyscf_backend import PySCFBackend

    s = MolecularSystem.from_xyz(WATER_XYZ)
    s.initialize_velocities(300.0, rng=1)
    xl = XLBOMD(PySCFBackend(s.symbols, conv_tol=1e-6), 0.5)
    run_md(s, xl, 1)
    state = xl.state_dict()
    other = XLBOMD(PySCFBackend(s.symbols, reference="uhf", reuse_density=False), 0.5)
    with pytest.raises(ValueError, match="backend's"):
        other.load_state_dict(state)
    assert other.result is None and not other.aux.initialized
