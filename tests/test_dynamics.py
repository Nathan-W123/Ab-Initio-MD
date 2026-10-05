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
