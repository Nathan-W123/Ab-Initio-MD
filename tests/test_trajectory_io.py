"""
Trajectory / log output of run_md and the readers in aimd.trajectory.

  - round trip: positions (XYZ, angstrom), velocities, dipoles and energies
    read back equal the in-memory states of the run (velocities, dipoles and
    energies bit for bit; positions to the 1e-10 angstrom print precision);
  - restart-append: a run checkpointed at step N, continued past N, crashed
    and restarted from the checkpoint leaves all four files byte-identical to
    an uninterrupted run;
  - formats: velocity units are self-describing and converted on reading;
    missing dipoles become nan with one warning; refused appends touch no file;
  - readers: comment metadata, incomplete trailing frames, malformed files.
"""

import warnings

import numpy as np
import pytest

from aimd.backends.morse import MorseBackend
from aimd.checkpoint import load_checkpoint
from aimd.integrators import CSVR, VelocityVerlet
from aimd.md import run_md
from aimd.system import MolecularSystem
from aimd.trajectory import (
    DIPOLE_COLUMNS,
    ENERGY_COLUMNS,
    VELOCITY_UNITS,
    VelocityWriter,
    XYZWriter,
    parse_comment,
    read_dipole_log,
    read_energy_log,
    read_velocities,
    read_xyz,
)
from aimd.units import ANG_TO_BOHR, AU_TIME_TO_FS, BOHR_TO_ANG

DT = 0.2


class ChargedMorse(MorseBackend):
    """Morse surface with point charges q_i: dipole = sum_i q_i r_i."""

    def __init__(self, symbols, **kw):
        super().__init__(symbols, **kw)
        self.charges = np.linspace(-0.3, 0.3, len(self.symbols))

    def compute(self, positions):
        res = super().compute(positions)
        res.dipole = self.charges @ np.asarray(positions)
        return res


def _start(h4, seed=1):
    s = h4.copy()
    s.initialize_velocities(600.0, rng=seed, remove_rotation=True)
    return s


def _outputs(tmp_path, prefix=""):
    return dict(trajectory=tmp_path / f"{prefix}t.xyz",
                velocity_trajectory=tmp_path / f"{prefix}v.xyz",
                energy_log=tmp_path / f"{prefix}e.csv",
                dipole_log=tmp_path / f"{prefix}d.csv")


def test_round_trip_of_all_outputs(h4, tmp_path):
    s = _start(h4)
    integ = VelocityVerlet(ChargedMorse(s.symbols), DT)
    states = {}

    def keep(rec):
        states[rec["step"]] = (s.positions.copy(), s.velocities.copy(),
                               integ.result.dipole.copy())

    out = _outputs(tmp_path)
    res = run_md(s, integ, 20, write_every=3, callback=keep, **out)
    steps = list(range(0, 21, 3))

    traj = read_xyz(out["trajectory"])
    assert traj.symbols == ["H"] * 4 and traj.positions.shape == (7, 4, 3)
    assert list(traj.step) == steps
    assert np.allclose(traj.time_fs, np.array(steps) * DT, rtol=0.0, atol=5e-5)
    assert traj.frame_interval_fs == pytest.approx(3 * DT, rel=1e-12)
    ref_pos = np.array([states[k][0] for k in steps])
    # %16.10f angstrom: rounding <= 5e-11 angstrom ~ 1e-10 bohr.
    assert np.allclose(traj.positions, ref_pos, rtol=0.0, atol=1e-10)
    epot = res.column("potential_Eh")[steps]
    assert np.allclose([i["E_pot"] for i in traj.info], epot, rtol=0.0, atol=5e-11)
    assert np.allclose(traj.masses, s.masses)

    vel = read_velocities(out["velocity_trajectory"])
    assert np.array_equal(vel.velocities, np.array([states[k][1] for k in steps]))
    assert list(vel.step) == steps and vel.info[0]["units"] == "bohr/au_time"

    log = read_energy_log(out["energy_log"])
    assert list(log) == ENERGY_COLUMNS and log["step"].dtype == np.int64
    for key in ENERGY_COLUMNS:
        assert np.array_equal(log[key], res.column(key)[steps]), key

    dip = read_dipole_log(out["dipole_log"])
    assert list(dip.step) == steps and np.array_equal(dip.time_fs, log["time_fs"])
    assert np.array_equal(dip.dipole, np.array([states[k][2] for k in steps]))
    assert dip.frame_interval_fs == pytest.approx(3 * DT, rel=1e-12)
    # Velocities are synchronous with positions: the kinetic energy in the log
    # is that of the velocity file's frame.
    ekin = 0.5 * np.einsum("i,fij,fij->f", s.masses, vel.velocities, vel.velocities)
    assert np.allclose(ekin, log["kinetic_Eh"], rtol=1e-14, atol=0.0)


def test_positions_file_format_is_unchanged(h4, tmp_path):
    """Other tools and restart-append rely on the exact XYZ layout."""
    s = _start(h4)
    run_md(s, VelocityVerlet(MorseBackend(s.symbols), DT), 1, trajectory=tmp_path / "t.xyz")
    lines = (tmp_path / "t.xyz").read_text().splitlines()
    assert lines[0] == "4"
    assert lines[1].startswith("step=0 t=0.0000fs E_pot=") and " E_tot=" in lines[1]
    x = h4.positions[0] * BOHR_TO_ANG
    assert lines[2] == f"H   {x[0]:16.10f} {x[1]:16.10f} {x[2]:16.10f}"
    assert lines[6] == "4" and lines[7].startswith("step=1 t=0.2000fs")
    first = MolecularSystem.from_xyz_string("\n".join(lines[:6]))
    assert np.allclose(read_xyz(tmp_path / "t.xyz").positions[0], first.positions, atol=1e-15)


@pytest.mark.parametrize("write_every", [3, 4])
def test_restart_append_is_byte_identical_for_all_outputs(write_every, h4, tmp_path):
    """
    Continuous 2N steps vs N steps + checkpoint, a doomed continuation that
    writes past N without checkpointing, then a restart from the checkpoint.
    With write_every = 3 the checkpoint frame (N = 30) is in the files already
    and must not be written twice; 4 does not divide N.
    """
    n, kw = 30, dict(write_every=write_every)
    ref = _start(h4)
    ref_out = _outputs(tmp_path, "ref_")
    run_md(ref, CSVR(ChargedMorse(ref.symbols), DT, 300.0, tau_fs=10.0, rng=3), 2 * n,
           **ref_out, **kw)

    s = _start(h4)
    out = _outputs(tmp_path)
    ck = tmp_path / "md.ckpt"
    integ = CSVR(ChargedMorse(s.symbols), DT, 300.0, tau_fs=10.0, rng=3)
    run_md(s, integ, n, checkpoint_path=ck, checkpoint_every=10, **out, **kw)
    run_md(s, integ, 10, start_step=n, **out, **kw)
    assert f"step={40 // write_every * write_every} " in out["velocity_trajectory"].read_text()

    ckpt = load_checkpoint(ck)
    run_md(ckpt.system, CSVR(ChargedMorse(s.symbols), DT, 300.0, tau_fs=10.0, rng=0), n,
           restart=ckpt, **out, **kw)
    for key in out:
        assert out[key].read_bytes() == ref_out[key].read_bytes(), key

    vel = read_velocities(out["velocity_trajectory"])
    assert list(vel.step) == list(range(0, 2 * n + 1, write_every))
    assert np.array_equal(vel.velocities[-1], ref.velocities)
    assert np.array_equal(read_dipole_log(out["dipole_log"]).step, vel.step)


def test_new_output_on_restart_gets_the_starting_frame(h4, tmp_path):
    """A velocity file first requested on restart starts at the checkpoint step."""
    s = _start(h4)
    ck = tmp_path / "md.ckpt"
    run_md(s, VelocityVerlet(MorseBackend(s.symbols), DT), 10,
           trajectory=tmp_path / "t.xyz", checkpoint_path=ck)
    ckpt = load_checkpoint(ck)
    run_md(ckpt.system, VelocityVerlet(MorseBackend(s.symbols), DT), 5, restart=ckpt,
           trajectory=tmp_path / "t.xyz", velocity_trajectory=tmp_path / "v.xyz")
    assert list(read_xyz(tmp_path / "t.xyz").step) == list(range(16))
    vel = read_velocities(tmp_path / "v.xyz")
    assert list(vel.step) == list(range(10, 16))
    assert np.array_equal(vel.velocities[0], load_checkpoint(ck).system.velocities)


def test_logs_first_requested_on_restart_get_the_starting_row(h4, tmp_path):
    """
    CSV logs: a new file, and one cut back to its header (it held only rows
    past the checkpoint), both start at the checkpoint step.
    """
    s = _start(h4)
    ck = tmp_path / "md.ckpt"
    run_md(s, VelocityVerlet(ChargedMorse(s.symbols), DT), 10, checkpoint_path=ck)
    stale = tmp_path / "e.csv"
    stale.write_text(",".join(ENERGY_COLUMNS) + "\r\n" + "11," + ",".join(["0.5"] * 6) + "\r\n")
    ckpt = load_checkpoint(ck)
    run_md(ckpt.system, VelocityVerlet(ChargedMorse(s.symbols), DT), 5, restart=ckpt,
           energy_log=stale, dipole_log=tmp_path / "d.csv")
    start = load_checkpoint(ck).system
    log = read_energy_log(stale)
    assert list(log["step"]) == list(range(10, 16))
    assert log["kinetic_Eh"][0] == start.kinetic_energy()
    dip = read_dipole_log(tmp_path / "d.csv")
    assert list(dip.step) == list(range(10, 16))
    assert np.array_equal(dip.dipole[0], ChargedMorse(s.symbols).charges @ start.positions)


def test_restart_with_another_timestep_is_not_evenly_spaced(h4, tmp_path):
    """Steps stay evenly spaced but times do not: the frame interval must be refused."""
    s = _start(h4)
    out = dict(trajectory=tmp_path / "t.xyz", dipole_log=tmp_path / "d.csv")
    run_md(s, VelocityVerlet(ChargedMorse(s.symbols), DT), 5, **out)
    run_md(s, VelocityVerlet(ChargedMorse(s.symbols), 1.5 * DT), 5, start_step=5,
           start_time_fs=5 * DT, **out)
    traj, dip = read_xyz(out["trajectory"]), read_dipole_log(out["dipole_log"])
    assert list(traj.step) == list(dip.step) == list(range(11))
    assert traj.time_fs[-1] == pytest.approx(5 * DT + 5 * 1.5 * DT, abs=1e-4)
    for data in (traj, dip):
        with pytest.raises(ValueError, match="evenly spaced in time"):
            data.frame_interval_fs


def test_velocity_units_are_self_describing(h4, tmp_path):
    s = _start(h4)
    path = tmp_path / "v.xyz"
    w = VelocityWriter(path, units="angstrom/fs")
    w.write(s, "step=0 t=0.0000fs")
    w.close()
    assert "units=angstrom/fs" in path.read_text().splitlines()[1]
    # 1 bohr / au_time = 21.877 angstrom / fs.
    assert VELOCITY_UNITS["angstrom/fs"] == pytest.approx(21.876912, rel=1e-6)
    line = path.read_text().splitlines()[2].split()
    assert float(line[1]) == pytest.approx(s.velocities[0, 0] * BOHR_TO_ANG / AU_TIME_TO_FS,
                                           rel=1e-15)
    # Appending frames in another unit: each frame is converted by its own tag.
    w = VelocityWriter(path, append=True)
    w.write(s, "step=1 t=0.2000fs")
    w.close()
    vel = read_velocities(path)
    assert np.allclose(vel.velocities[0], s.velocities, rtol=1e-15, atol=0.0)
    assert np.array_equal(vel.velocities[1], s.velocities)
    # ... and the tag keeps a velocity file from being read as positions
    # (regression: read_xyz took the velocities for angstrom coordinates)
    with pytest.raises(ValueError, match=r"is a velocity file \(frame 0: units=angstrom/fs\)"):
        read_xyz(path)

    bare = tmp_path / "bare.xyz"
    bare.write_text("1\nstep=0\nH 1e-3 0 0\n")
    with pytest.raises(ValueError, match="units="):
        read_velocities(bare)
    assert read_velocities(bare, units="angstrom/fs").velocities[0, 0, 0] == pytest.approx(
        1e-3 * ANG_TO_BOHR * AU_TIME_TO_FS, rel=1e-15)
    bare.write_text("1\nunits=furlong/fortnight\nH 1 0 0\n")
    with pytest.raises(ValueError, match="unknown velocity unit"):
        read_velocities(bare)
    with pytest.raises(ValueError, match="unknown velocity unit"):
        VelocityWriter(tmp_path / "x.xyz", units="m/s")


def test_missing_dipoles_are_logged_as_nan_with_one_warning(h4, tmp_path):
    s = _start(h4)
    with pytest.warns(RuntimeWarning, match="no dipole for 6 logged") as record:
        run_md(s, VelocityVerlet(MorseBackend(s.symbols), DT), 10,
               dipole_log=tmp_path / "d.csv", write_every=2)
    assert len([w for w in record if "dipole" in str(w.message)]) == 1
    dip = read_dipole_log(tmp_path / "d.csv")
    assert list(dip.step) == [0, 2, 4, 6, 8, 10] and np.all(np.isnan(dip.dipole))
    from aimd.analysis import ir_spectrum
    with pytest.raises(ValueError, match="nan"):
        ir_spectrum(dip.dipole, dip.frame_interval_fs)


def test_refused_append_leaves_every_file_untouched(h4, tmp_path):
    s = _start(h4)
    out = _outputs(tmp_path)
    run_md(s.copy(), VelocityVerlet(ChargedMorse(s.symbols), DT), 15, **out)
    before = {k: p.read_bytes() for k, p in out.items()}
    out["dipole_log"].write_text("step,time_fs,mu_x\n0,0,0\n")
    before["dipole_log"] = out["dipole_log"].read_bytes()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="cannot append"):
            run_md(s, VelocityVerlet(ChargedMorse(s.symbols), DT), 1, start_step=10, **out)
    for key, path in out.items():
        assert path.read_bytes() == before[key], key
    with pytest.raises(ValueError, match="distinct"):
        run_md(s, VelocityVerlet(MorseBackend(s.symbols), DT), 1,
               trajectory=tmp_path / "same.xyz", velocity_trajectory=tmp_path / "same.xyz")
    # Regression: a checkpoint on an output's path replaced the open file.
    with pytest.raises(ValueError, match="distinct"):
        run_md(s, VelocityVerlet(MorseBackend(s.symbols), DT), 1,
               energy_log=tmp_path / "same.csv", checkpoint_path=tmp_path / "same.csv")
    assert not (tmp_path / "same.csv").exists()


def test_dipole_log_columns(tmp_path, h4):
    s = _start(h4)
    run_md(s, VelocityVerlet(ChargedMorse(s.symbols), DT), 2, dipole_log=tmp_path / "d.csv")
    header = (tmp_path / "d.csv").read_text().splitlines()[0].split(",")
    assert header == DIPOLE_COLUMNS == ["step", "time_fs", "dipole_x_au", "dipole_y_au",
                                        "dipole_z_au"]


# ── Readers ───────────────────────────────────────────────────────────────────

def test_parse_comment():
    info = parse_comment("step=12 t=2.4000fs E_pot=-0.1234567890 E_tot=-0.12 units=bohr/au_time x")
    assert info == {"step": 12, "time_fs": 2.4, "E_pot": -0.123456789, "E_tot": -0.12,
                    "units": "bohr/au_time"}
    assert isinstance(info["step"], int)
    assert parse_comment("water, near the HF/STO-3G minimum") == {}


def test_reader_drops_an_incomplete_last_frame(h4, tmp_path):
    s = _start(h4)
    path = tmp_path / "t.xyz"
    run_md(s, VelocityVerlet(MorseBackend(s.symbols), DT), 4, trajectory=path)
    complete = path.read_text()
    for stub in ("4\nstep=5", "4\nstep=5 t=1.0000fs\nH 0.1 0.2 0.3\nH 0.1", "4\n"):
        path.write_text(complete + stub)
        with pytest.warns(UserWarning, match="incomplete last frame"):
            traj = read_xyz(path)
        assert list(traj.step) == [0, 1, 2, 3, 4]
    path.write_text(complete + "\n\n")                       # trailing blank lines are fine
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert read_xyz(path).n_frames == 5


def test_readers_drop_a_last_line_cut_inside_a_number(h4, tmp_path):
    """
    Regression: a write cut off inside the last number leaves a line with all
    its fields but no line break; the readers returned the truncated value
    (for the velocities "...e-01" -> "...", off by a factor of 10). They now
    drop it, as restart truncation does.
    """
    s = _start(h4)
    out = _outputs(tmp_path)
    run_md(s, VelocityVerlet(ChargedMorse(s.symbols), DT), 4, **out)
    readers = dict(trajectory=read_xyz, velocity_trajectory=read_velocities,
                   energy_log=read_energy_log, dipole_log=read_dipole_log)
    for key, reader in readers.items():
        full = out[key].read_bytes()
        out[key].write_bytes(full[:-5])                  # line break + 3-4 characters
        with pytest.warns(UserWarning, match="incomplete last"):
            data = reader(out[key])
        steps = data["step"] if isinstance(data, dict) else data.step
        assert list(steps) == [0, 1, 2, 3], key
        out[key].write_bytes(full[:-1] if full.endswith(b"\r\n") else full)   # "\r" ends a row
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            data = reader(out[key])
        assert len(data["step"] if isinstance(data, dict) else data.step) == 5, key
    # A lone hand-written frame without a final line break is kept.
    one = tmp_path / "one.xyz"
    one.write_text("2\nH2\nH 0 0 0\nH 0 0 0.74")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert read_xyz(one).positions[0, 1, 2] == pytest.approx(0.74 * ANG_TO_BOHR)


def test_reader_rejects_malformed_or_inconsistent_files(tmp_path):
    path = tmp_path / "bad.xyz"
    path.write_text("2\na\nH 0 0 0\nH 0 0 1\n2\nb\nH 0 0 0\nO 0 0 1\n")
    with pytest.raises(ValueError, match="differ"):
        read_xyz(path)
    path.write_text("2\na\nH 0 0 0\nH 0 0\n2\nb\nH 0 0 0\nH 0 0 1\n")
    with pytest.raises(ValueError, match="malformed"):
        read_xyz(path)
    path.write_text("two\na\n")
    with pytest.raises(ValueError, match="atom count"):
        read_xyz(path)
    path.write_text("")
    with pytest.raises(ValueError, match="no frames"):
        read_xyz(path)
    path.write_text("1\nstep=0 t=0.0fs\nH 0 0 0\n1\nstep=0 t=0.0fs\nH 0 0 1\n")
    with pytest.raises(ValueError, match="evenly spaced"):        # two runs in one file
        read_xyz(path).frame_interval_fs


def test_reader_matches_single_frame_parser():
    from pathlib import Path

    water = Path(__file__).resolve().parent / "data" / "water_experimental.xyz"
    traj = read_xyz(water)
    ref = MolecularSystem.from_xyz(water)
    assert traj.n_frames == 1 and traj.symbols == ref.symbols
    assert np.array_equal(traj.positions[0], ref.positions)
    assert traj.step is None and traj.time_fs is None
    with pytest.raises(ValueError, match="two frames"):
        traj.frame_interval_fs


def test_explicit_append_of_fresh_runs_is_readable_but_not_evenly_spaced(h4, tmp_path):
    traj, log = tmp_path / "t.xyz", tmp_path / "e.csv"
    for seed in (1, 2):
        s = _start(h4, seed)
        run_md(s, VelocityVerlet(MorseBackend(s.symbols), DT), 3,
               trajectory=traj, velocity_trajectory=tmp_path / "v.xyz",
               energy_log=log, append=seed == 2)
    assert list(read_xyz(traj).step) == [0, 1, 2, 3] * 2
    assert list(read_velocities(tmp_path / "v.xyz").step) == [0, 1, 2, 3] * 2
    assert list(read_energy_log(log)["step"]) == [0, 1, 2, 3] * 2
    with pytest.raises(ValueError, match="evenly spaced"):
        read_velocities(tmp_path / "v.xyz").frame_interval_fs


def test_xyz_writer_append_without_truncation(h4, tmp_path):
    path = tmp_path / "t.xyz"
    for append in (False, True):
        w = XYZWriter(path, append=append)
        assert w.resumed == append
        w.write(h4, f"step={int(append)}")
        w.close()
    assert read_xyz(path).n_frames == 2


def test_csv_reader_drops_an_incomplete_last_row(h4, tmp_path):
    s = _start(h4)
    log = tmp_path / "e.csv"
    run_md(s, VelocityVerlet(MorseBackend(s.symbols), DT), 4, energy_log=log)
    complete = log.read_text()
    log.write_text(complete + "5,1.0,-0.1")                   # crash mid-row
    with pytest.warns(UserWarning, match="incomplete last row"):
        assert list(read_energy_log(log)["step"]) == [0, 1, 2, 3, 4]
    log.write_text(complete.replace("\n2,", "\n2,x,", 1))     # damage inside the file
    with pytest.raises(ValueError, match="malformed"):
        read_energy_log(log)


def test_frame_count_comes_from_the_data():
    from aimd.trajectory import XYZTrajectory

    traj = XYZTrajectory(["H"], np.zeros((3, 1, 3)))
    assert traj.n_frames == 3 and traj.n_atoms == 1 and traj.step is None
