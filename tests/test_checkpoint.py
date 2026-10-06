"""
Checkpoint / restart: file round trip, safety of the format, and exact
continuation (2N steps == N steps + checkpoint + restart + N steps).
"""

import json
import warnings

import numpy as np
import pytest

from aimd.backends.morse import MorseBackend
from aimd.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from aimd.integrators import CSVR, LangevinBAOAB, NoseHooverChain, VelocityVerlet
from aimd.md import run_md
from aimd.system import MolecularSystem
from aimd.thermostats import CSVRThermostat

T = 300.0

FACTORIES = {
    "nve": lambda b, seed: VelocityVerlet(b, 0.2),
    "berendsen": lambda b, seed: VelocityVerlet(b, 0.2, temperature_k=T, berendsen_tau_fs=10.0),
    "langevin": lambda b, seed: LangevinBAOAB(b, 0.2, T, friction_per_fs=0.05, rng=seed),
    "csvr": lambda b, seed: CSVR(b, 0.2, T, tau_fs=10.0, rng=seed),
    "nhc": lambda b, seed: NoseHooverChain(b, 0.2, T, period_fs=10.0, n_mts=2),
    "vv+csvr": lambda b, seed: VelocityVerlet(
        b, 0.2, thermostat=CSVRThermostat(T, tau_fs=10.0, rng=seed)),
}


class DensityMorse(MorseBackend):
    """Morse surface that also reports a fake SCF density, dipole and info."""

    supports_density_guess = True

    def __init__(self, symbols, **kw):
        super().__init__(symbols, **kw)
        self.guesses = []

    def compute(self, positions):
        res = super().compute(positions)
        x = np.asarray(positions).ravel()[:4]
        res.density = np.outer(x, x) + np.eye(4)
        res.dipole = np.asarray(positions).sum(axis=0)
        res.info = {"scf_iterations": 7, "method": "fake", "orbital_energies": x.copy()}
        return res

    def set_density_guess(self, density):
        self.guesses.append(np.array(density))


def _start(h4, seed=1):
    s = h4.copy()
    s.initialize_velocities(600.0, rng=seed, remove_rotation=True)
    return s


# ── Exact continuation ────────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", list(FACTORIES))
def test_restart_reproduces_uninterrupted_run(kind, h4, tmp_path):
    n = 150
    ref_sys = _start(h4)
    ref = run_md(ref_sys, FACTORIES[kind](MorseBackend(ref_sys.symbols), 21), 2 * n)

    s = _start(h4)
    first = run_md(s, FACTORIES[kind](MorseBackend(s.symbols), 21), n,
                   checkpoint_path=tmp_path / "md.ckpt")
    ckpt = load_checkpoint(tmp_path / "md.ckpt")
    assert ckpt.step == n and ckpt.time_fs == n * 0.2
    # Fresh objects; a different seed proves the RNG state comes from the file.
    sys2 = ckpt.system
    integ2 = FACTORIES[kind](MorseBackend(sys2.symbols), 999)
    second = run_md(sys2, integ2, n, restart=ckpt)

    assert np.allclose(sys2.positions, ref_sys.positions, rtol=0.0, atol=1e-12)
    assert np.allclose(sys2.velocities, ref_sys.velocities, rtol=0.0, atol=1e-12)
    joined = first.records + second.records[1:]
    assert [r["step"] for r in joined] == list(range(2 * n + 1))
    for key in ref.records[0]:
        a = np.array([r[key] for r in joined])
        b = ref.column(key)
        assert np.allclose(a, b, rtol=0.0, atol=1e-12), key
    assert (sys2.com_removed, sys2.rotation_removed) == (ref_sys.com_removed, ref_sys.rotation_removed)


@pytest.mark.parametrize("kind", ["berendsen", "langevin", "csvr", "nhc", "vv+csvr"])
def test_checkpoint_rebuilds_integrator_with_same_settings(kind, h4, tmp_path):
    s = _start(h4)
    integ = FACTORIES[kind](MorseBackend(s.symbols), 5)
    run_md(s, integ, 20, checkpoint_path=tmp_path / "c.ckpt")
    ckpt = load_checkpoint(tmp_path / "c.ckpt")
    rebuilt = ckpt.make_integrator(MorseBackend(s.symbols))
    assert type(rebuilt) is type(integ)
    assert rebuilt.config() == integ.config()
    sys2 = ckpt.system
    run_md(s, integ, 30)
    run_md(sys2, rebuilt, 30)
    assert np.array_equal(sys2.positions, s.positions)
    assert np.array_equal(sys2.velocities, s.velocities)


def test_restart_appends_identical_files_after_simulated_crash(h4, tmp_path):
    """
    Continuous 2N-step run vs. a run checkpointed at step N that kept writing
    (as if it crashed later), restarted from the checkpoint: the trajectory
    and log must be byte-identical. write_every = 4 does not divide N = 30.
    """
    n = 30
    kw = dict(write_every=4)
    ref = _start(h4)
    run_md(ref, FACTORIES["csvr"](MorseBackend(ref.symbols), 3), 2 * n,
           trajectory=tmp_path / "ref.xyz", energy_log=tmp_path / "ref.csv", **kw)

    s = _start(h4)
    integ = FACTORIES["csvr"](MorseBackend(s.symbols), 3)
    traj, log, ck = tmp_path / "t.xyz", tmp_path / "e.csv", tmp_path / "md.ckpt"
    run_md(s, integ, n, trajectory=traj, energy_log=log, checkpoint_path=ck,
           checkpoint_every=10, **kw)
    # The doomed continuation writes frames 32..40 but no checkpoint.
    run_md(s, integ, 10, trajectory=traj, energy_log=log, start_step=n, **kw)
    assert "step=40 " in traj.read_text()

    ckpt = load_checkpoint(ck)
    assert ckpt.step == n
    run_md(ckpt.system, FACTORIES["csvr"](MorseBackend(s.symbols), 0), n,
           trajectory=traj, energy_log=log, restart=ckpt, **kw)
    assert traj.read_bytes() == (tmp_path / "ref.xyz").read_bytes()
    assert log.read_bytes() == (tmp_path / "ref.csv").read_bytes()


def test_checkpoint_object_can_be_restarted_from_twice(h4, tmp_path):
    """
    Regression: run_md(ckpt.system, ..., restart=ckpt) propagates ckpt.system
    in place, and a second restart from the same object used to put back
    those advanced positions together with the saved (old) forces.
    """
    s = _start(h4)
    run_md(s, FACTORIES["csvr"](MorseBackend(s.symbols), 4), 20,
           checkpoint_path=tmp_path / "c.ckpt")
    ckpt = load_checkpoint(tmp_path / "c.ckpt")
    runs = [run_md(ckpt.system, ckpt.make_integrator(MorseBackend(s.symbols)), 15,
                   restart=ckpt) for _ in range(2)]
    for key in runs[0].records[0]:
        assert np.array_equal(runs[0].column(key), runs[1].column(key)), key


def test_checkpoint_every_writes_at_multiples_and_at_the_end(h4, tmp_path):
    s = _start(h4)
    steps = []
    integ = VelocityVerlet(MorseBackend(s.symbols), 0.2)
    ck = tmp_path / "md.ckpt"

    def spy(rec):
        if ck.exists():
            steps.append((rec["step"], load_checkpoint(ck).step))

    run_md(s, integ, 23, checkpoint_path=ck, checkpoint_every=10, callback=spy)
    assert steps[0] == (11, 10) and (21, 20) in steps
    assert load_checkpoint(ck).step == 23
    assert not list(tmp_path.glob("*.tmp"))


# ── File contents and format ──────────────────────────────────────────────────

def test_checkpoint_round_trip_preserves_full_state(h4, tmp_path):
    s = _start(h4)
    backend = DensityMorse(s.symbols)
    integ = NoseHooverChain(backend, 0.2, T, period_fs=10.0)
    run_md(s, integ, 7)
    integ.result.info["not_plain_data"] = object()
    with pytest.warns(UserWarning, match="not_plain_data"):
        save_checkpoint(tmp_path / "a.ckpt", s, integ, step=7, time_fs=1.4)
    ck = load_checkpoint(tmp_path / "a.ckpt")

    assert ck.step == 7 and ck.time_fs == 1.4 and ck.backend == "morse"
    for attr in ("positions", "velocities", "masses"):
        assert np.array_equal(getattr(ck.system, attr), getattr(s, attr))
    assert ck.system.symbols == s.symbols
    assert (ck.system.charge, ck.system.multiplicity) == (s.charge, s.multiplicity)
    assert (ck.system.com_removed, ck.system.rotation_removed, ck.system.rotational_dof) == (
        True, True, 3)
    res = ck.integrator_state["result"]
    assert res["energy"] == integ.result.energy
    for key in ("gradient", "density", "dipole"):
        assert np.array_equal(res[key], getattr(integ.result, key))
    assert res["info"]["scf_iterations"] == 7 and res["info"]["method"] == "fake"
    assert np.array_equal(res["info"]["orbital_energies"], integ.result.info["orbital_energies"])
    assert "not_plain_data" not in res["info"]
    thermo = ck.integrator_state["thermostat"]
    assert np.array_equal(thermo["xi"], integ.thermostat.xi)
    assert np.array_equal(thermo["vxi"], integ.thermostat.vxi)
    assert thermo["n_dof"] == 6

    # Loading hands the saved density to the SCF backend as the next guess.
    backend2 = DensityMorse(s.symbols)
    integ2 = NoseHooverChain(backend2, 0.2, T, period_fs=10.0)
    integ2.load_state_dict(ck.integrator_state)
    assert len(backend2.guesses) == 1
    assert np.array_equal(backend2.guesses[0], integ.result.density)
    assert integ2.thermostat_energy() == integ.thermostat_energy()


def test_checkpoint_is_plain_npz_readable_without_pickle(h4, tmp_path):
    s = _start(h4)
    integ = LangevinBAOAB(MorseBackend(s.symbols), 0.2, T, rng=1)
    run_md(s, integ, 3)
    path = save_checkpoint(tmp_path / "x.ckpt", s, integ, step=3)
    assert path == tmp_path / "x.ckpt"                   # no ".npz" appended
    with np.load(path, allow_pickle=False) as npz:
        assert all(npz[k].dtype != object for k in npz.files)
        meta = json.loads(str(npz["__metadata__"]))
    assert meta["format"] == "aimd-checkpoint" and meta["step"] == 3
    assert meta["integrator"]["rng"]["bit_generator"] == "PCG64"
    assert meta["integrator"]["heat"] == integ.heat


@pytest.mark.parametrize("bitgen", ["PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64"])
def test_rng_state_survives_checkpoint(bitgen, h4, tmp_path):
    s = _start(h4)
    rng = np.random.Generator(getattr(np.random, bitgen)(42))
    integ = CSVR(MorseBackend(s.symbols), 0.2, T, tau_fs=10.0, rng=rng)
    run_md(s, integ, 5)
    save_checkpoint(tmp_path / "r.ckpt", s, integ, step=5)
    restored = CSVR(MorseBackend(s.symbols), 0.2, T, tau_fs=10.0)
    restored.load_state_dict(load_checkpoint(tmp_path / "r.ckpt").integrator_state)
    assert type(restored.thermostat.rng.bit_generator).__name__ == bitgen
    assert np.array_equal(restored.thermostat.rng.normal(size=8), integ.thermostat.rng.normal(size=8))


def test_mismatched_restarts_are_rejected(h4, tmp_path):
    s = _start(h4)
    integ = CSVR(MorseBackend(s.symbols), 0.2, T, rng=1)
    run_md(s, integ, 2)
    save_checkpoint(tmp_path / "m.ckpt", s, integ, step=2)
    ck = load_checkpoint(tmp_path / "m.ckpt")
    with pytest.raises(ValueError, match="CSVR"):
        LangevinBAOAB(MorseBackend(s.symbols), 0.2, T).load_state_dict(ck.integrator_state)
    plain_vv = {**ck.integrator_state, "integrator": "VelocityVerlet"}
    with pytest.raises(ValueError, match="thermostat"):
        VelocityVerlet(MorseBackend(s.symbols), 0.2).load_state_dict(plain_vv)
    nhc = NoseHooverChain(MorseBackend(s.symbols), 0.2, T, chain_length=3)
    run_md(s.copy(), nhc, 1)
    state = nhc.state_dict()
    with pytest.raises(ValueError, match="chain"):
        NoseHooverChain(MorseBackend(s.symbols), 0.2, T, chain_length=4).load_state_dict(state)
    water = Checkpoint(MolecularSystem(["O", "H", "H"], np.eye(3)), ck.step, ck.time_fs,
                       ck.integrator_state)
    with pytest.raises(ValueError, match="atoms"):
        run_md(s, CSVR(MorseBackend(s.symbols), 0.2, T), 1, restart=water)
    with pytest.raises(ValueError, match="either"):
        run_md(s, integ, 1, restart=ck, start_step=5)
    # A rejected restart leaves the target system untouched.
    before = s.positions.copy()
    with pytest.raises(ValueError):
        run_md(s, LangevinBAOAB(MorseBackend(s.symbols), 0.2, T), 1, restart=ck)
    assert np.array_equal(s.positions, before)


def test_non_checkpoint_files_are_rejected(tmp_path):
    np.savez(tmp_path / "plain.npz", x=np.zeros(3))
    with pytest.raises(ValueError, match="not an aimd checkpoint"):
        load_checkpoint(tmp_path / "plain.npz")
    meta = json.dumps({"format": "aimd-checkpoint", "version": 99})
    with (tmp_path / "future.ckpt").open("wb") as fh:
        np.savez(fh, __metadata__=np.array(meta))
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(tmp_path / "future.ckpt")


def test_appending_to_a_log_with_other_columns_is_refused(h4, tmp_path):
    log = tmp_path / "old.csv"
    log.write_text("step,time_fs,potential_Eh,kinetic_Eh,total_Eh,temperature_K\n0,0,0,0,0,0\n")
    s = _start(h4)
    traj = tmp_path / "t.xyz"
    run_md(s.copy(), VelocityVerlet(MorseBackend(s.symbols), 0.2), 15, trajectory=traj)
    frames = traj.read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="cannot append"):
            run_md(s, VelocityVerlet(MorseBackend(s.symbols), 0.2), 1,
                   trajectory=traj, energy_log=log, start_step=10)
    # Regression: the refused run must not have cut frames 11..15 already.
    assert traj.read_bytes() == frames


def test_partially_written_frame_and_row_are_dropped_on_resume(h4, tmp_path):
    """A crash in the middle of a write leaves half a frame / row behind."""
    n = 12
    ref = _start(h4)
    run_md(ref, FACTORIES["nhc"](MorseBackend(ref.symbols), 0), 2 * n,
           trajectory=tmp_path / "ref.xyz", energy_log=tmp_path / "ref.csv")
    s = _start(h4)
    traj, log, ck = tmp_path / "t.xyz", tmp_path / "e.csv", tmp_path / "md.ckpt"
    run_md(s, FACTORIES["nhc"](MorseBackend(s.symbols), 0), n,
           trajectory=traj, energy_log=log, checkpoint_path=ck)
    # Cut inside "step=13": the stub parses as step 1, so only the
    # completeness check can tell it from a genuine early frame / row.
    with traj.open("a") as fh:
        fh.write("4\nstep=1")
    with log.open("a") as fh:
        fh.write("1")
    ckpt = load_checkpoint(ck)
    run_md(ckpt.system, FACTORIES["nhc"](MorseBackend(s.symbols), 0), n,
           trajectory=traj, energy_log=log, restart=ckpt)
    assert ckpt.system.n_dof == 6
    assert traj.read_bytes() == (tmp_path / "ref.xyz").read_bytes()
    assert log.read_bytes() == (tmp_path / "ref.csv").read_bytes()


def test_explicit_append_of_a_fresh_run_keeps_earlier_frames(h4, tmp_path):
    traj, log = tmp_path / "t.xyz", tmp_path / "e.csv"
    for seed in (1, 2):
        s = _start(h4, seed)
        run_md(s, VelocityVerlet(MorseBackend(s.symbols), 0.2), 5,
               trajectory=traj, energy_log=log, append=seed == 2)
    assert traj.read_text().count("step=") == 12          # 2 x steps 0..5
    rows = log.read_text().strip().splitlines()
    assert rows[0].startswith("step,") and len(rows) == 13
    assert [r.split(",")[0] for r in rows[1:]] == [str(i) for i in range(6)] * 2
