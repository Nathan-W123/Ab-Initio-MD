"""
Thermostat tests against independent references.

  - CSVR: one rescaling step reproduces the exact transition law of the
    kinetic energy (scaled noncentral chi^2, scipy.stats) and of the signed
    projection on the old velocity (Gaussian);
  - NHC: the MTK chain operator matches a tight scipy ODE solution of the
    chain equations and converges at the Suzuki-Yoshida order;
  - harmonic systems: Velocity Verlet conserves the shadow energy
    (p^2 + w^2 (1 - h^2 w^2 / 4) q^2) / 2 exactly, so the thermostat
    bookkeeping (effective energy / extended Hamiltonian) can be checked to
    round-off, and the stationary averages are known in closed form;
  - canonical sampling: <K> = N_f kT / 2, Var(K) = (N_f / 2) (kT)^2 and
    position variances, with block-averaged standard errors and 5 sigma;
  - conserved-quantity drift on the anharmonic Morse cluster.
"""

import math

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import solve_ivp

from aimd.backends import get_backend
from aimd.backends.harmonic import HarmonicBackend
from aimd.integrators import CSVR, LangevinBAOAB, NoseHooverChain, VelocityVerlet
from aimd.md import run_md
from aimd.system import MolecularSystem
from aimd.thermostats import CSVRThermostat, _sy_weights, nhc_propagate
from aimd.units import FS_TO_AU_TIME, KB_AU

T = 300.0
KT = KB_AU * T


# ── Helpers ───────────────────────────────────────────────────────────────────

def block_se(x: np.ndarray, n_blocks: int = 20) -> float:
    """Standard error of the mean of a correlated series by block averaging."""
    n = len(x) // n_blocks * n_blocks
    means = x[:n].reshape(n_blocks, -1).mean(axis=1)
    return float(means.std(ddof=1) / math.sqrt(n_blocks))


def oscillators(n_atoms: int, p_min: float, p_max: float, seed: int, dt_fs: float):
    """
    Tethered H atoms, each Cartesian coordinate an independent oscillator with
    a distinct period in [p_min, p_max] fs. Starts from a thermal draw. Returns
    the system, backend, force constants k (N, 3), and the shadow force
    constants k (1 - (w h)^2 / 4) whose quadratic form Verlet conserves.
    """
    rng = np.random.default_rng(seed)
    periods = np.geomspace(p_min, p_max, 3 * n_atoms)
    rng.shuffle(periods)
    masses = MolecularSystem(["H"] * n_atoms, np.zeros((n_atoms, 3))).masses
    w = 2.0 * np.pi / (periods.reshape(n_atoms, 3) * FS_TO_AU_TIME)
    k = masses[:, None] * w**2
    h = dt_fs * FS_TO_AU_TIME
    system = MolecularSystem(
        ["H"] * n_atoms,
        rng.normal(size=(n_atoms, 3)) * np.sqrt(KT / k),
        velocities=rng.normal(size=(n_atoms, 3)) * np.sqrt(KT / masses[:, None]),
    )
    return system, HarmonicBackend(system.symbols, force_constants=k), k, k * (1 - (w * h) ** 2 / 4)


# ── CSVR: exact one-step transition law ───────────────────────────────────────

def _csvr_case(kind: str) -> MolecularSystem:
    rng = np.random.default_rng(1)
    if kind == "diatomic":          # COM + rotation removed: N_f = 1 (no chi^2 part)
        s = MolecularSystem(["H", "H"], [[0, 0, 0], [0, 0, 1.4]])
        s.initialize_velocities(900.0, rng=rng, remove_rotation=True)
    elif kind == "atom":            # free atom: N_f = 3
        s = MolecularSystem(["He"], [[0, 0, 0]], velocities=rng.normal(size=(1, 3)) * 1e-3)
    else:                           # 5 atoms, COM removed: N_f = 12
        s = MolecularSystem(["C", "H", "H", "O", "N"], rng.normal(size=(5, 3)))
        s.initialize_velocities(150.0, rng=rng)
    return s


@pytest.mark.parametrize("k0_over_kt", [0.4, 12.0], ids=["cold", "hot"])
@pytest.mark.parametrize("kind, n_f", [("diatomic", 1), ("atom", 3), ("cluster", 12)])
def test_csvr_step_reproduces_exact_kinetic_energy_law(kind, n_f, k0_over_kt):
    """
    CSVR solves the kinetic-energy process exactly: over a time t, with
    c = exp(-t/tau), the new kinetic energy is distributed as |u'|^2 / 2 for
    u' = sqrt(c) u0 + sqrt((1 - c) kT) xi (mass-weighted velocities, xi ~ N(0, I)),
    i.e. (1 - c) kT / 2 times a noncentral chi^2 with N_f DOF and noncentrality
    2 c K0 / ((1 - c) kT). The velocity is reversed (alpha < 0) exactly when
    u' points against u0, with probability Phi(-sqrt(c) |u0| / sqrt((1 - c) kT)).
    t / tau = 0.7 is far outside any small-step approximation.
    """
    s = _csvr_case(kind)
    assert s.n_dof == n_f
    s.velocities *= math.sqrt(k0_over_kt * KT / s.kinetic_energy())
    v0, k0 = s.velocities.copy(), s.kinetic_energy()
    thermo = CSVRThermostat(T, tau_fs=10.0, rng=2)
    t = 7.0 * FS_TO_AU_TIME
    c = math.exp(-0.7)
    n = 20000
    kin, alpha = np.empty(n), np.empty(n)
    for i in range(n):
        s.velocities = v0.copy()
        alpha[i] = thermo.rescale(s, t)
        kin[i] = s.kinetic_energy()
    law = stats.ncx2(df=n_f, nc=2 * c * k0 / ((1 - c) * KT), scale=(1 - c) * KT / 2)
    assert stats.kstest(kin, law.cdf).pvalue > 1e-4
    p_reverse = stats.norm.cdf(-math.sqrt(c * 2 * k0 / ((1 - c) * KT)))
    assert stats.binomtest(int(np.sum(alpha < 0)), n, p_reverse).pvalue > 1e-4
    assert np.allclose(alpha**2 * k0, kin, rtol=1e-12, atol=0.0)
    # Heat bookkeeping: the accumulated added kinetic energy.
    assert thermo.heat == pytest.approx(np.sum(kin - k0), rel=1e-9)


def test_csvr_leaves_a_system_at_rest_alone():
    s = MolecularSystem(["H", "H"], [[0, 0, 0], [0, 0, 1.4]])
    thermo = CSVRThermostat(T, tau_fs=10.0, rng=0)
    assert thermo.rescale(s, 100.0) == 1.0 and s.kinetic_energy() == 0.0


# ── NHC: chain operator against an ODE solver ─────────────────────────────────

def _nhc_ode_reference(xi0, vxi0, q, kT, n_f, ekin2, t):
    """Chain equations + velocity scale s (K = s^2 K0), DOP853 at rtol 1e-12."""
    m = len(xi0)

    def rhs(_, y):
        s, v = y[0], y[1 + m:]
        g = np.empty(m)
        g[0] = (s * s * ekin2 - n_f * kT) / q[0]
        g[1:] = (q[:-1] * v[:-1] ** 2 - kT) / q[1:]
        dv = g - np.append(v[:-1] * v[1:], 0.0)
        return np.concatenate([[-v[0] * s], v, dv])

    sol = solve_ivp(rhs, (0.0, t), np.concatenate([[1.0], xi0, vxi0]),
                    method="DOP853", rtol=1e-12, atol=1e-14)
    y = sol.y[:, -1]
    return y[0], y[1:1 + m], y[1 + m:]


def test_nhc_operator_matches_ode_and_converges_at_suzuki_yoshida_order():
    # A hot system (2K = 2.5 N_f kT) with a chain already in motion, propagated
    # over omega t = 0.6 of the thermostat period: a hard case for the splitting.
    kT, n_f, omega = 1e-3, 7, 0.01
    q = np.array([n_f * kT, kT, kT]) / omega**2
    xi0, vxi0 = np.array([0.1, -0.2, 0.3]), np.array([4e-3, -3e-3, 2e-3])
    ekin2, t = 2.5 * n_f * kT, 60.0
    s_ref, xi_ref, vxi_ref = _nhc_ode_reference(xi0, vxi0, q, kT, n_f, ekin2, t)

    def error(n_sy, n_mts):
        xi, vxi = xi0.copy(), vxi0.copy()
        s = nhc_propagate(xi, vxi, q, kT, n_f, ekin2, t, np.array(_sy_weights(n_sy)), n_mts)
        return max(abs(s - s_ref), np.abs(xi - xi_ref).max(), np.abs(vxi - vxi_ref).max() / 1e-3)

    assert error(7, 16) < 1e-6           # measured 2.5e-7
    for n_sy, order in ((1, 2), (3, 4), (7, 6)):
        observed = math.log2(error(n_sy, 8) / error(n_sy, 16))
        assert abs(observed - order) < 0.35, (n_sy, observed)


# ── Exact bookkeeping on harmonic systems via the Verlet shadow energy ────────

def _shadow_run(make, n_steps=3000, dt_fs=1.0, seed=5):
    """Max |H_shadow + thermostat term - initial| / kT over a harmonic run."""
    rng = np.random.default_rng(seed)
    n = 4
    a = rng.normal(size=(3 * n, 3 * n))
    hess = a @ a.T * (0.05 / (3 * n)) + 0.02 * np.eye(3 * n)      # dense SPD Hessian
    backend = HarmonicBackend(["H"] * n, hessian=hess, reference_positions=rng.normal(size=(n, 3)))
    s = MolecularSystem(["H"] * n, backend.reference_positions + 0.15 * rng.normal(size=(n, 3)))
    s.velocities = 1.5 * rng.normal(size=(n, 3)) * np.sqrt(KT / s.masses[:, None])
    w, modes = backend.normal_modes(s.masses)
    h = dt_fs * FS_TO_AU_TIME
    sqm = np.sqrt(np.repeat(s.masses, 3))
    shadow_hess = np.outer(sqm, sqm) * (modes @ np.diag(w**2 * (1 - (w * h) ** 2 / 4)) @ modes.T)
    integ = make(backend, dt_fs)
    values = []

    def shadow(_rec):
        dx = (s.positions - backend.reference_positions).ravel()
        values.append(0.5 * dx @ shadow_hess @ dx + s.kinetic_energy() + integ.thermostat_energy())

    res = run_md(s, integ, n_steps, callback=shadow)
    values = np.array(values)
    moved = np.abs(np.diff(res.column("conserved_Eh") - res.column("total_Eh"))).sum()
    return np.abs(values - values[0]).max() / KT, moved / KT


@pytest.mark.parametrize("make", [
    lambda b, dt: VelocityVerlet(b, dt),
    lambda b, dt: VelocityVerlet(b, dt, temperature_k=T, berendsen_tau_fs=5.0),
    lambda b, dt: CSVR(b, dt, T, tau_fs=5.0, rng=1),
], ids=["nve", "berendsen", "csvr"])
def test_rescaling_thermostats_conserve_shadow_effective_energy_exactly(make):
    """
    Verlet conserves H_shadow exactly for a harmonic surface and rescaling
    changes only K, so H_shadow - (kinetic energy added) is constant to
    round-off. A wrong sign or a missed half-step in the heat bookkeeping would
    show up at the level of the heat exchanged (tens of kT here).
    """
    err, moved = _shadow_run(make)
    assert err < 1e-10                    # measured ~2e-13 kT
    if moved:
        assert moved > 10.0


def test_nhc_extended_energy_is_exact_up_to_chain_splitting_error():
    """
    With Verlet exact in the shadow sense, the only error left in H' is the
    Suzuki-Yoshida splitting of the chain: it must vanish as the chain
    sub-steps are refined (6th order for 7 weights).
    """
    make = lambda n_mts: lambda b, dt: NoseHooverChain(
        b, dt, T, period_fs=20.0, n_mts=n_mts, n_suzuki_yoshida=7)
    coarse, moved = _shadow_run(make(4), n_steps=1000)
    fine, _ = _shadow_run(make(8), n_steps=1000)
    assert moved > 100.0                  # chain energy moved: ~870 kT
    assert fine < 2e-7                    # measured 6.2e-8 kT
    assert 45.0 < coarse / fine < 90.0    # 2x finer chain step: ideal 2^6 = 64 (measured 63.7)


def test_nhc_divergence_raises_instead_of_returning_nan():
    """A 10 fs chain period with 1 fs steps is unstable with the (1, 3) scheme."""
    s, backend, _, _ = oscillators(4, 8.0, 40.0, seed=11, dt_fs=1.0)
    s.positions *= 2.0
    s.velocities *= 0.2
    with pytest.raises(FloatingPointError, match="diverged"):
        run_md(s, NoseHooverChain(backend, 1.0, T, period_fs=10.0), 5000)


# ── Canonical sampling ────────────────────────────────────────────────────────

def _canonical_stats(make, n_steps, dt_fs=0.5, seed=1):
    """
    24 oscillators (periods 16-48 fs). Returns (ratio, SE) pairs, each ratio
    expected to be 1 in the canonical ensemble of the Verlet shadow
    Hamiltonian, which a velocity thermostat around Verlet samples exactly
    (CSVR) or to O(dt^2) (NHC):
      <K> / (N_f kT / 2),  Var(K) / ((N_f / 2) (kT)^2),
      <sum_i k_shadow,i x_i^2 / 2> / (N_f kT / 2)    (position variances).
    """
    s, backend, _, k_shadow = oscillators(8, 16.0, 48.0, seed, dt_fs)
    integ = make(backend, dt_fs)
    pot = []
    res = run_md(s, integ, n_steps,
                 callback=lambda _r: pot.append(0.5 * np.sum(k_shadow * s.positions**2)))
    burn = n_steps // 10
    kin = res.column("kinetic_Eh")[burn:]
    pot = np.array(pot)[burn:]
    kbar = s.n_dof * KT / 2
    var_ref = s.n_dof * KT**2 / 2
    dk2 = (kin - kin.mean()) ** 2
    return {
        "K": (kin.mean() / kbar, block_se(kin) / kbar),
        "VarK": (dk2.mean() / var_ref, block_se(dk2) / var_ref),
        "U": (pot.mean() / kbar, block_se(pot) / kbar),
    }


@pytest.mark.parametrize("make, max_se", [
    (lambda b, dt: CSVR(b, dt, T, tau_fs=10.0, rng=7), {"K": 0.01, "VarK": 0.035, "U": 0.01}),
    (lambda b, dt: NoseHooverChain(b, dt, T, period_fs=20.0), {"K": 0.005, "VarK": 0.03, "U": 0.005}),
], ids=["csvr", "nhc"])
def test_global_thermostats_sample_canonical_kinetic_and_potential_energy(make, max_se):
    """
    Tolerance: 5 block-averaging standard errors (20 blocks of 2700 steps, much
    longer than the measured integrated autocorrelation times of 2-18 steps).
    ``max_se`` caps the SE so the test keeps its power: measured SEs are
    CSVR 0.0064 / 0.019 / 0.0064 and NHC 0.0018 / 0.017 / 0.0021.
    (Per-coordinate equipartition is not tested here: a single global
    thermostat shares energy between uncoupled modes very slowly, which is a
    property of the method; see the Langevin test below for that check.)
    """
    for key, (ratio, se) in _canonical_stats(make, 60000).items():
        assert se < max_se[key], (key, se)
        assert abs(ratio - 1.0) < 5.0 * se, (key, ratio, se)


def test_baoab_samples_exact_harmonic_positions_per_coordinate():
    """
    BAOAB is exact in configuration for harmonic forces at any step size
    (Leimkuhler & Matthews, AMRX 2013): <k x^2> = kT per coordinate, while the
    kinetic energy is biased to <m v^2> = kT (1 - (w h)^2 / 4). With w h up to
    0.79 the bias reaches 15%, so the second check also shows the test can
    tell the two apart.
    """
    dt_fs = 1.0
    s, backend, k, _ = oscillators(8, 8.0, 40.0, seed=3, dt_fs=dt_fs)
    h = dt_fs * FS_TO_AU_TIME
    wh2 = ((k / s.masses[:, None]) * h**2 / 4).ravel()
    xs, vs = [], []
    run_md(s, LangevinBAOAB(backend, dt_fs, T, friction_per_fs=0.1, rng=3), 60000,
           callback=lambda _r: (xs.append(s.positions.ravel().copy()),
                                vs.append(s.velocities.ravel().copy())))
    x2 = np.array(xs)[5000:] ** 2 * k.ravel() / KT
    v2 = np.array(vs)[5000:] ** 2 * np.repeat(s.masses, 3) / KT
    x_mean, v_mean = x2.mean(axis=0), v2.mean(axis=0)
    x_se = np.array([block_se(c) for c in x2.T])
    v_se = np.array([block_se(c) for c in v2.T])
    assert np.all(x_se < 0.03)                              # measured ~0.02
    assert np.all(np.abs(x_mean - 1.0) < 5 * x_se)
    assert np.all(np.abs(v_mean - (1.0 - wh2)) < 5 * v_se)
    # Aggregated over coordinates the uncorrected kT is rejected decisively.
    agg_se = math.sqrt(np.sum(v_se**2)) / len(v_se)
    assert abs(v_mean.mean() - 1.0) > 5 * agg_se


def test_global_thermostats_target_reduced_dof_and_keep_constraints(h4):
    """
    With P = 0 and L = 0 the H4 cluster has N_f = 6. CSVR, NHC and Berendsen
    rescale globally, so P and L stay zero and the mean kinetic temperature
    (computed with N_f = 6) must reach the target; targeting 12 or 9 DOF would
    give 600 K or 450 K. Tolerance: 5 block SEs (measured SE 2-6 K).
    """
    morse = get_backend("morse")
    for make in (
        lambda b: CSVR(b, 0.2, T, tau_fs=20.0, rng=8),
        lambda b: NoseHooverChain(b, 0.2, T, period_fs=20.0),
        lambda b: VelocityVerlet(b, 0.2, temperature_k=T, berendsen_tau_fs=20.0),
    ):
        s = h4.copy()
        s.initialize_velocities(600.0, rng=8, remove_rotation=True)
        res = run_md(s, make(morse(s.symbols)), 15000)
        temp = res.column("temperature_K")[3000:]
        assert s.n_dof == 6
        assert abs(temp.mean() - T) < 5 * block_se(temp)
        p_scale = math.sqrt(s.masses.sum() * 2 * s.kinetic_energy())
        assert np.abs(s.momentum()).max() < 1e-11 * p_scale
        assert np.abs(s.angular_momentum()).max() < 1e-11 * 3.0 * p_scale


# ── Conserved quantities on an anharmonic surface ─────────────────────────────

def _drift(make, h4, dt_fs, t_fs, seed=3):
    s = h4.copy()
    s.initialize_velocities(600.0, rng=seed, remove_rotation=True)
    res = run_md(s, make(get_backend("morse")(s.symbols), dt_fs), int(round(t_fs / dt_fs)))
    moved = np.abs(np.diff(res.column("conserved_Eh") - res.column("total_Eh"))).sum()
    return res.conserved_energy_drift / KT, moved / KT


@pytest.mark.parametrize("make", [
    lambda b, dt: VelocityVerlet(b, dt),
    lambda b, dt: NoseHooverChain(b, dt, T, period_fs=20.0),
], ids=["nve", "nhc"])
def test_conserved_quantity_error_is_second_order(make, h4):
    """Over a fixed 40 fs window halving dt cuts the error 4x (measured 4.06, 4.03)."""
    coarse, _ = _drift(make, h4, 0.2, 40.0)
    fine, _ = _drift(make, h4, 0.1, 40.0)
    assert 3.5 < coarse / fine < 4.5


@pytest.mark.parametrize("make", [
    lambda b, dt: NoseHooverChain(b, dt, T, period_fs=20.0),
    lambda b, dt: CSVR(b, dt, T, tau_fs=20.0, rng=1),
    lambda b, dt: LangevinBAOAB(b, dt, T, friction_per_fs=0.02, rng=1),
], ids=["nhc", "csvr", "langevin"])
def test_conserved_quantity_stays_bounded_while_thermostat_works(make, h4):
    """
    2 ps at dt = 0.2 fs, starting at 600 K with a 300 K target. Measured
    max |E_cons - E_cons(0)|: NHC 0.11 kT, CSVR 0.05 kT, Langevin 0.11 kT, the
    size of the NVE energy error (0.04 kT), while the thermostats exchanged
    1200-2400 kT. Bound 0.5 kT; any bookkeeping error larger than ~2e-4 of the
    energy exchanged would break it.
    """
    drift, moved = _drift(make, h4, 0.2, 2000.0)
    assert moved > 500.0
    assert drift < 0.5
