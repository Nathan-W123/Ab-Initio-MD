import numpy as np
import pytest

from aimd.backends import get_backend
from aimd.integrators import LangevinBAOAB, VelocityVerlet
from aimd.md import run_md
from aimd.system import MolecularSystem


def _morse(system):
    return get_backend("morse")(system.symbols)


def test_nve_conserves_energy(h4):
    h4.initialize_velocities(500.0, rng=3)
    result = run_md(h4, VelocityVerlet(_morse(h4), timestep_fs=0.1), n_steps=2000)
    assert result.total_energy_drift < 1e-5


def test_nve_drift_is_second_order_in_dt(h4):
    """Halving dt should cut the energy error ~4x (Verlet is O(dt^2))."""
    drifts = []
    for dt in (0.2, 0.1):
        s = h4.copy()
        s.initialize_velocities(500.0, rng=3)
        res = run_md(s, VelocityVerlet(_morse(s), dt), n_steps=int(round(40 / dt)))
        drifts.append(res.total_energy_drift)
    assert 3.0 < drifts[0] / drifts[1] < 5.0


def test_velocity_verlet_is_time_reversible(h4):
    h4.initialize_velocities(500.0, rng=4)
    start = h4.positions.copy()
    run_md(h4, VelocityVerlet(_morse(h4), 0.1), n_steps=300)
    h4.velocities *= -1.0
    run_md(h4, VelocityVerlet(_morse(h4), 0.1), n_steps=300)
    assert np.allclose(h4.positions, start, atol=1e-8)


def test_nve_conserves_momentum(h4):
    h4.initialize_velocities(500.0, rng=5)
    run_md(h4, VelocityVerlet(_morse(h4), 0.2), n_steps=500)
    assert np.allclose(h4.masses @ h4.velocities, 0.0, atol=1e-10)


def test_berendsen_drives_temperature_to_target(h4):
    h4.initialize_velocities(50.0, rng=6)
    integ = VelocityVerlet(_morse(h4), 0.2, temperature_k=400.0, berendsen_tau_fs=10.0)
    res = run_md(h4, integ, n_steps=3000)
    assert res.column("temperature_K")[-1000:].mean() == pytest.approx(400.0, rel=0.15)


def test_langevin_samples_target_temperature(h4):
    h4.initialize_velocities(300.0, rng=7)
    integ = LangevinBAOAB(_morse(h4), 0.25, temperature_k=300.0, friction_per_fs=0.05, rng=7)
    res = run_md(h4, integ, n_steps=40000)
    # LangevinBAOAB lets the COM drift, so all 3N momenta are thermalised.
    ke = res.column("kinetic_Eh")[5000:]
    from aimd.units import KB_AU
    t_mean = 2.0 * ke.mean() / (3 * h4.n_atoms * KB_AU)
    assert t_mean == pytest.approx(300.0, rel=0.05)


def test_run_md_writes_trajectory_and_log(tmp_path, h4):
    h4.initialize_velocities(300.0, rng=8)
    traj, log = tmp_path / "t.xyz", tmp_path / "e.csv"
    run_md(h4, VelocityVerlet(_morse(h4), 0.5), 10,
           trajectory=traj, energy_log=log, write_every=5)
    frames = traj.read_text().count("step=")
    assert frames == 3                       # steps 0, 5, 10
    assert len(log.read_text().strip().splitlines()) == 4   # header + 3 rows
    last = MolecularSystem.from_xyz_string(
        "\n".join(traj.read_text().splitlines()[-6:])
    )
    assert np.allclose(last.positions, h4.positions, atol=1e-9)


# ── Conserved quantity, harmonic closed forms, driver options ─────────────────

from aimd.backends.harmonic import HarmonicBackend  # noqa: E402
from aimd.trajectory import ENERGY_COLUMNS  # noqa: E402
from aimd.units import FS_TO_AU_TIME  # noqa: E402


def test_nve_conserved_energy_is_total_energy(h4):
    h4.initialize_velocities(300.0, rng=11)
    res = run_md(h4, VelocityVerlet(_morse(h4), 0.2), n_steps=50)
    assert np.array_equal(res.column("conserved_Eh"), res.column("total_Eh"))
    assert res.conserved_energy_drift == res.total_energy_drift


def test_velocity_verlet_matches_discrete_harmonic_solution():
    """
    For x'' = -w^2 x, Velocity Verlet positions satisfy the recursion
    x_{n+1} - 2 x_n + x_{n-1} = -(w h)^2 x_n, solved exactly by
    x_n = x_0 cos(n th) + (h v_0 / sin th) sin(n th), cos th = 1 - (w h)^2 / 2,
    and v_n = (x_{n+1} - x_n) / h + h w^2 x_n / 2.
    """
    sys_ = MolecularSystem(["H"], [[0.3, -0.2, 0.1]], velocities=[[1e-4, 2e-4, -3e-4]])
    k = np.array([0.05, 0.2, 0.4])                       # Eh / bohr^2
    backend = HarmonicBackend(["H"], force_constants=k[None, :])
    dt_fs, n = 0.5, 1000
    x0, v0 = sys_.positions[0].copy(), sys_.velocities[0].copy()
    run_md(sys_, VelocityVerlet(backend, dt_fs), n_steps=n)

    h = dt_fs * FS_TO_AU_TIME
    w = np.sqrt(k / sys_.masses[0])
    th = np.arccos(1.0 - 0.5 * (w * h) ** 2)            # phase advance per step
    x = lambda j: x0 * np.cos(j * th) + h * v0 / np.sin(th) * np.sin(j * th)
    v_n = (x(n + 1) - x(n)) / h + 0.5 * h * w**2 * x(n)
    assert np.allclose(sys_.positions[0], x(n), rtol=0.0, atol=1e-12)
    assert np.allclose(sys_.velocities[0], v_n, rtol=0.0, atol=1e-14)


def test_berendsen_reproduces_original_implementation(h4):
    """The refactor into a thermostat object must not change Berendsen numerics."""
    h4.initialize_velocities(100.0, rng=12)
    ref = h4.copy()
    backend = _morse(h4)
    run_md(h4, VelocityVerlet(backend, 0.2, temperature_k=400.0, berendsen_tau_fs=10.0), 300)

    # Original loop (aimd 0.1.0): VV step, then v *= sqrt(1 + dt/tau (T0/T - 1)).
    dt, tau = 0.2 * FS_TO_AU_TIME, 10.0 * FS_TO_AU_TIME
    acc = lambda: -backend.compute(ref.positions).gradient / ref.masses[:, None]
    a = acc()
    for _ in range(300):
        ref.velocities += 0.5 * dt * a
        ref.positions += dt * ref.velocities
        a = acc()
        ref.velocities += 0.5 * dt * a
        lam2 = 1.0 + (dt / tau) * (400.0 / ref.temperature() - 1.0)
        ref.velocities *= np.sqrt(max(lam2, 0.0))
    assert np.array_equal(h4.positions, ref.positions)
    assert np.array_equal(h4.velocities, ref.velocities)


def test_nve_keeps_zero_linear_and_angular_momentum(h4):
    """Verlet conserves P and L exactly, so the reduced N_dof stays valid."""
    h4.initialize_velocities(800.0, rng=13, remove_rotation=True)
    assert h4.n_dof == 6
    run_md(h4, VelocityVerlet(_morse(h4), 0.2), n_steps=2000)
    p_scale = np.sqrt(h4.masses.sum() * 2 * h4.kinetic_energy())
    assert np.abs(h4.momentum()).max() < 1e-12 * p_scale
    assert np.abs(h4.angular_momentum()).max() < 1e-12 * p_scale * 3.0


def test_langevin_releases_com_and_rotation_constraints(h4):
    h4.initialize_velocities(300.0, rng=14, remove_rotation=True)
    run_md(h4, LangevinBAOAB(_morse(h4), 0.2, 300.0, rng=14), n_steps=2)
    assert not h4.com_removed and not h4.rotation_removed and h4.n_dof == 12


def test_energy_log_appends_conserved_column(tmp_path, h4):
    h4.initialize_velocities(300.0, rng=15)
    log = tmp_path / "e.csv"
    run_md(h4, VelocityVerlet(_morse(h4), 0.2), 3, energy_log=log)
    header = log.read_text().splitlines()[0].split(",")
    assert header == ["step", "time_fs", "potential_Eh", "kinetic_Eh", "total_Eh",
                      "temperature_K", "conserved_Eh"]
    assert header == ENERGY_COLUMNS


def test_start_step_continues_numbering_and_time(h4):
    h4.initialize_velocities(300.0, rng=16)
    res = run_md(h4, VelocityVerlet(_morse(h4), 0.25), 5, start_step=40)
    assert list(res.column("step")) == list(range(40, 46))
    assert np.array_equal(res.column("time_fs"), np.arange(40, 46) * 0.25)
    res = run_md(h4, VelocityVerlet(_morse(h4), 0.25), 2, start_step=3, start_time_fs=7.0)
    assert np.allclose(res.column("time_fs"), [7.0, 7.25, 7.5])


def test_thermostat_target_can_be_changed_between_runs(h4):
    """
    Annealing: the legacy ``temperature_k`` attribute stays settable. After a
    300 -> 900 K change the mean temperature must match 900 K within 5 block
    SEs (measured SE: 38 K for CSVR / NHC, 0.8 K for Berendsen); the SE cap
    keeps the old 300 K target more than 10 SE away.
    """
    from aimd.integrators import CSVR, NoseHooverChain
    from test_thermostats import block_se

    h4.initialize_velocities(300.0, rng=17, remove_rotation=True)
    for integ in (CSVR(_morse(h4), 0.2, 300.0, tau_fs=5.0, rng=17),
                  NoseHooverChain(_morse(h4), 0.2, 300.0, period_fs=20.0),
                  VelocityVerlet(_morse(h4), 0.2, temperature_k=300.0, berendsen_tau_fs=5.0)):
        s = h4.copy()
        run_md(s, integ, 200)
        integ.temperature_k = 900.0
        assert integ.temperature_k == 900.0
        assert integ.thermostat.kT == pytest.approx(900.0 * 3.166811563455546e-6)
        temp = run_md(s, integ, 4000).column("temperature_K")[1000:]
        assert block_se(temp) < 60.0
        assert abs(temp.mean() - 900.0) < 5 * block_se(temp)
    with pytest.raises(AttributeError):
        VelocityVerlet(_morse(h4), 0.2).temperature_k = 300.0


def test_invalid_integrator_and_driver_arguments(h4):
    from aimd.thermostats import CSVRThermostat

    b = _morse(h4)
    with pytest.raises(ValueError, match="either"):
        VelocityVerlet(b, 0.2, temperature_k=300.0, thermostat=CSVRThermostat(300.0))
    with pytest.raises(ValueError, match="both"):
        VelocityVerlet(b, 0.2, temperature_k=300.0)
    with pytest.raises(ValueError, match="checkpoint_path"):
        run_md(h4, VelocityVerlet(b, 0.2), 1, checkpoint_every=5)
