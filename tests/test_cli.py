"""
Command-line interface: option plumbing, config files and error handling.

Fast checks on the morse / harmonic surfaces and tiny HF runs; the physics of
complete CLI runs is checked against independent references in
test_end_to_end.py.
"""

import importlib.util
import json

import numpy as np
import pytest

from aimd.backends import get_backend
from aimd.checkpoint import load_checkpoint
from aimd.cli import CLIError, _StepTimer, backend_kwargs, build_parser, check_spin_state, main
from aimd.trajectory import read_energy_log, read_xyz
from conftest import EXAMPLES

WATER = str(EXAMPLES / "water.xyz")
H4 = str(EXAMPLES / "h4_cluster.xyz")
_TIMER_CALL = _StepTimer.__call__


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    """Default output files land in a scratch directory, never in the repo."""
    monkeypatch.chdir(tmp_path)


def run(capsys, *argv):
    rc = main([str(a) for a in argv])
    out = capsys.readouterr()
    return rc, out.out, out.err


# ── aimd backends ─────────────────────────────────────────────────────────────

def test_backends_lists_every_backend_with_dependency_status(capsys):
    rc, out, _ = run(capsys, "backends")
    assert rc == 0
    rows = {line.split()[0]: line for line in out.splitlines()[1:]}
    assert set(rows) == {"harmonic", "hf", "morse", "psi4", "pyscf"}
    assert "none needed" in rows["hf"] and "none needed" in rows["morse"]
    for pkg in ("pyscf", "psi4"):
        present = importlib.util.find_spec(pkg) is not None
        assert ("(installed)" if present else "NOT installed") in rows[pkg]


# ── Option plumbing ───────────────────────────────────────────────────────────

def _args(*argv):
    return build_parser().parse_args(["run", WATER, *argv])


def test_generic_flags_map_to_each_backends_argument_names():
    args = _args("--method", "b3lyp", "--basis", "6-31g", "--threads", "2",
                 "--conv-tol", "1e-9", "--max-cycles", "50",
                 "--backend-option", "cart=true", "--backend-option", 'scf_options={"a": 1}')
    kw = backend_kwargs(args, get_backend("pyscf"))
    assert kw == {"method": "b3lyp", "basis": "6-31g", "threads": 2, "conv_tol": 1e-9,
                  "max_cycle": 50, "cart": True, "scf_options": {"a": 1}}
    args = _args("--threads", "3", "--conv-tol", "1e-8")
    assert backend_kwargs(args, get_backend("psi4")) == {"num_threads": 3, "conv_tol": 1e-8}
    assert backend_kwargs(_args("--max-cycles", "7"), get_backend("hf")) == {"max_cycles": 7}
    assert backend_kwargs(_args(), get_backend("morse")) == {}       # nothing unasked


@pytest.mark.parametrize("argv, message", [
    (["--basis", "sto-3g"], "--basis is not an option of the 'morse' backend"),
    (["--backend-option", "basis=1"], "has no option 'basis'"),
    (["--backend-option", "charge=1"], "set by aimd itself"),
    (["--backend-option", "depth"], "expects KEY=VALUE"),
])
def test_options_a_backend_does_not_take_are_rejected(argv, message):
    with pytest.raises(CLIError, match=message):
        backend_kwargs(_args("--backend", "morse", *argv), get_backend("morse"))


@pytest.mark.parametrize("symbols, charge, mult, ok", [
    (["O", "H", "H"], 0, 1, True), (["O", "H", "H"], 0, 3, True),
    (["O", "H", "H"], 0, 2, False), (["O", "H"], 0, 2, True), (["O", "H"], -1, 1, True),
    (["H"], 0, 1, False), (["H"], 1, 1, True), (["H"], 2, 1, False), (["H", "H"], 0, 5, False),
])
def test_spin_state_check_follows_electron_count_parity(symbols, charge, mult, ok):
    if ok:
        check_spin_state(symbols, charge, mult)
    else:
        with pytest.raises(CLIError):
            check_spin_state(symbols, charge, mult)


# ── Runs on model surfaces ───────────────────────────────────────────────────

def test_morse_nve_run_writes_outputs_and_summary(capsys, tmp_path):
    rc, out, _ = run(capsys, "run", H4, "--backend", "morse", "--steps", "20", "--dt", "0.2",
                     "--temperature", "100", "--seed", "1", "--print-every", "5")
    assert rc == 0
    log = read_energy_log("energies.csv")
    assert list(log["step"]) == list(range(21))
    assert read_xyz("trajectory.xyz").n_frames == 21
    # header, 5 printed steps (0, 5, ..., 20) and the summary
    assert "integrator velocity Verlet, NVE, dt = 0.2 fs" in out
    assert sum(line.split()[0].isdigit() for line in out.splitlines() if line.strip()) == 5
    drift = float(out.split("max |E - E0| = ")[1].split()[0])
    assert drift == pytest.approx(np.max(np.abs(log["conserved_Eh"] - log["conserved_Eh"][0])),
                                  rel=1e-3)
    assert log["temperature_K"][0] == pytest.approx(100.0)
    assert "(steps 1-20)" in out


def test_seed_makes_runs_reproducible_and_streams_independent(capsys):
    common = ["run", H4, "--backend", "morse", "--steps", "5", "--temperature", "200",
              "--thermostat", "langevin", "--print-every", "0"]
    for name, seed in (("a", 3), ("b", 3), ("c", 4)):
        assert run(capsys, *common, "--seed", seed, "--energies", f"{name}.csv",
                   "--trajectory", f"{name}.xyz")[0] == 0
    a, b, c = (read_energy_log(f"{n}.csv")["total_Eh"] for n in "abc")
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_outputs_can_be_disabled_and_extra_outputs_requested(capsys, tmp_path):
    with pytest.warns(RuntimeWarning, match="reported no dipole for 4 logged step"):
        rc, _, err = run(capsys, "run", H4, "--backend", "morse", "--steps", "3",
                         "--temperature", "50", "--trajectory", "none", "--energies", "none",
                         "--velocities", "v.xyz", "--dipoles", "d.csv", "--print-every", "0")
    assert rc == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["d.csv", "v.xyz"]


# ── Config files ──────────────────────────────────────────────────────────────

def test_toml_and_json_configs_equal_the_flags_and_flags_override(capsys, tmp_path):
    (tmp_path / "geo").mkdir()
    (tmp_path / "geo" / "h4.xyz").write_text(open(H4).read())
    toml = tmp_path / "geo" / "run.toml"
    toml.write_text(
        'xyz = "h4.xyz"\nbackend = "morse"\nsteps = 8\ndt = 0.25\n'
        'thermostat = "csvr"\ntemperature = 150.0\ntau = 10\nseed = 9\n'
        'remove_rotation = true\nprint-every = 0\nenergies = "toml.csv"\n'
        'trajectory = "toml.xyz"\n[backend_options]\ndepth = 0.2\n')
    cfg = json.loads(json.dumps({
        "xyz": "h4.xyz", "backend": "morse", "steps": 8, "dt": 0.25, "thermostat": "csvr",
        "temperature": 150.0, "tau": 10, "seed": 9, "remove_rotation": True,
        "print_every": 0, "energies": "json.csv", "trajectory": "json.xyz",
        "backend_options": {"depth": 0.2}}))
    (tmp_path / "geo" / "run.json").write_text(json.dumps(cfg))
    assert run(capsys, "run", "--config", toml)[0] == 0
    assert run(capsys, "run", "--config", tmp_path / "geo" / "run.json")[0] == 0
    assert run(capsys, "run", H4, "--backend", "morse", "--steps", "8", "--dt", "0.25",
               "--thermostat", "csvr", "--temperature", "150", "--tau", "10", "--seed", "9",
               "--remove-rotation", "--print-every", "0", "--backend-option", "depth=0.2",
               "--energies", "flags.csv", "--trajectory", "flags.xyz")[0] == 0
    ref = (tmp_path / "flags.csv").read_text()
    assert (tmp_path / "toml.csv").read_text() == ref
    assert (tmp_path / "json.csv").read_text() == ref
    # flags override the file (steps, and a boolean switched off)
    assert run(capsys, "run", "--config", toml, "--steps", "3", "--no-remove-rotation",
               "--energies", "short.csv")[0] == 0
    log = read_energy_log(tmp_path / "short.csv")
    assert len(log["step"]) == 4
    # same 150 K start, but with rotation kept N_dof = 3N - 3 = 9 instead of 3N - 6 = 6
    ke_rot_removed = read_energy_log(tmp_path / "toml.csv")["kinetic_Eh"][0]
    assert log["kinetic_Eh"][0] / ke_rot_removed == pytest.approx(9 / 6, rel=1e-12)


@pytest.mark.parametrize("name, text, message", [
    ("bad.toml", 'steps = 3\nstepz = 4\n', "unknown option 'stepz'"),
    ("bad.toml", 'xlbomd = "yes"\n', "must be true or false"),
    ("bad.toml", 'steps = [1, 2]\n', "must be a single value"),
    ("bad.toml", 'steps = \n', "not valid TOML"),
    ("bad.json", '[1, 2]', "must hold a table"),
    ("bad.json", '{"steps": }', "not valid JSON"),
    ("bad.toml", 'config = "other.toml"\n', "unknown option 'config'"),
    ("bad.toml", 'thermostat = "andersen"\n', "invalid choice"),
])
def test_bad_config_files_give_helpful_errors(capsys, tmp_path, name, text, message):
    (tmp_path / name).write_text(text)
    rc, _, err = run(capsys, "run", H4, "--backend", "morse", "--config", tmp_path / name)
    assert rc == 2 and message in err


@pytest.mark.parametrize("flag", ["--conf", "--confi", "--con"])
def test_abbreviated_config_flag_is_rejected_not_ignored(capsys, tmp_path, flag):
    # Regression: argparse accepted --conf as --config, but the file is
    # expanded before parsing (exact --config only), so it was silently dropped
    # and the run used the defaults
    (tmp_path / "run.toml").write_text('backend = "morse"\nsteps = 1\n')
    rc, _, err = run(capsys, "run", H4, flag, tmp_path / "run.toml", "--print-every", "0")
    assert rc == 2 and "unrecognized arguments" in err and flag in err
    assert not (tmp_path / "energies.csv").exists()
    rc, out, _ = run(capsys, "run", H4, "--config", tmp_path / "run.toml", "--print-every", "0")
    assert rc == 0 and "morse" in out


def test_missing_config_file(capsys):
    rc, _, err = run(capsys, "run", "--config", "nowhere.toml")
    assert rc == 2 and "cannot read config file nowhere.toml" in err


def test_harmonic_backend_is_centred_on_the_starting_geometry(capsys, tmp_path):
    # Regression: from the CLI the harmonic wells sat at the origin, so water
    # at 100 K ran at ~70,000 K. Centred on the start, the NVE run starts at
    # the minimum (E_pot = 0), E_tot = K(0), and since E_pot >= 0 the
    # temperature can never exceed its initial 100 K, up to the Verlet energy
    # error: (w dt)^2 / 4 ~ 3% of E for the 9 fs H periods at dt = 0.5 fs.
    rc, out, _ = run(capsys, "run", WATER, "--backend", "harmonic", "--steps", "200",
                     "--temperature", "100", "--seed", "4", "--print-every", "0",
                     "--checkpoint", "c.npz")
    assert rc == 0
    log = read_energy_log("energies.csv")
    assert log["potential_Eh"][0] == 0.0
    assert log["temperature_K"][0] == pytest.approx(100.0)
    assert log["temperature_K"].max() < 105.0
    # an explicit centre still wins: the origin gives the old, huge energy
    origin = json.dumps([0.0] * 9)
    rc, _, _ = run(capsys, "run", WATER, "--backend", "harmonic", "--steps", "1",
                   "--backend-option", f"reference_positions={origin}", "--print-every", "0",
                   "--energies", "e0.csv", "--trajectory", "t0.xyz")
    assert rc == 0 and read_energy_log("e0.csv")["potential_Eh"][0] > 1.0
    # a restart keeps the original centre: the checkpoint records the backend
    # arguments, so the XYZ file is not needed (it used to be), and the
    # continuation is the same with or without it
    rc, out, _ = run(capsys, "run", "--restart", "c.npz", "--steps", "1",
                     "--print-every", "0")
    assert rc == 0 and "backend    harmonic" in out
    row = read_energy_log("energies.csv")["potential_Eh"][-1]
    rc, out, _ = run(capsys, "run", WATER, "--restart", "c.npz", "--backend", "harmonic",
                     "--steps", "5", "--print-every", "0")
    assert rc == 0
    # the summary labels the continued segment with absolute step numbers
    assert "5 (200 -> 205)" in out and "(steps 201-205)" in out
    log = read_energy_log("energies.csv")
    assert list(log["step"][199:203]) == [199, 200, 201, 202]
    assert log["potential_Eh"][201] == row
    assert log["temperature_K"].max() < 105.0


# ── Input errors ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("argv, message", [
    (["--basis", "def2-qzvpp"], "Unknown basis set 'def2-qzvpp'"),
    (["--multiplicity", "2"], "10 electrons (even) need an odd"),
    (["--charge", "1"], "9 electrons (odd) need an even"),
    (["--thermostat", "csvr"], "--thermostat csvr needs a target --temperature"),
    (["--thermostat", "nhc"], "--thermostat nhc needs a target --temperature"),
    (["--xlbomd", "--thermostat", "langevin", "--temperature", "300"], "Langevin"),
    (["--xlbomd", "--backend", "morse"], "needs a backend that accepts SCF density"),
    (["--backend", "gaussian"], "Unknown backend 'gaussian'"),
    (["--dt", "-1"], "--dt must be a positive"),
    (["--checkpoint-every", "5"], "--checkpoint-every needs --checkpoint"),
    (["--xlbomd", "--xl-k", "12"], "K"),
    (["--method", "b3lyp"], "Hartree-Fock only"),
    (["--thermostat", "berendsen", "--temperature", "300", "--tau", "0"], "tau_fs must be"),
])
def test_bad_run_options_give_helpful_errors(capsys, argv, message):
    rc, out, err = run(capsys, "run", WATER, "--steps", "1", *argv)
    assert rc == 2, err
    assert message in err
    assert "Traceback" not in err


def test_bad_geometry_files(capsys, tmp_path):
    (tmp_path / "x.xyz").write_text("2\n\nXx 0 0 0\nH 0 0 1\n")
    rc, _, err = run(capsys, "run", tmp_path / "x.xyz", "--backend", "morse")
    assert rc == 2 and "Unknown element symbol: 'Xx'" in err
    (tmp_path / "kr.xyz").write_text("1\n\nKr 0 0 0\n")
    rc, _, err = run(capsys, "run", tmp_path / "kr.xyz")
    assert rc == 2 and "has no data for element 'Kr'" in err
    rc, _, err = run(capsys, "run", tmp_path / "missing.xyz")
    assert rc == 2 and "missing.xyz" in err
    rc, _, err = run(capsys, "run", "--steps", "1")
    assert rc == 2 and "no starting geometry" in err


@pytest.mark.skipif(importlib.util.find_spec("psi4") is not None, reason="Psi4 is installed")
def test_missing_optional_dependency_is_explained(capsys):
    rc, _, err = run(capsys, "run", WATER, "--backend", "psi4")
    assert rc == 2 and "needs Psi4" in err


# ── Restart mismatches ────────────────────────────────────────────────────────

@pytest.fixture
def water_checkpoint(capsys, tmp_path):
    rc, _, _ = run(capsys, "run", WATER, "--steps", "2", "--thermostat", "csvr",
                   "--temperature", "300", "--seed", "1", "--checkpoint", "c.npz",
                   "--print-every", "0")
    assert rc == 0
    ckpt = load_checkpoint(tmp_path / "c.npz")
    assert ckpt.step == 2 and ckpt.backend == "hf"
    return tmp_path / "c.npz"


@pytest.mark.parametrize("argv, message", [
    (["--thermostat", "nhc", "--temperature", "300"],
     "written by integrator CSVR + CSVRThermostat, but the options select "
     "NoseHooverChain + NoseHooverChainThermostat"),
    (["--thermostat", "csvr", "--temperature", "300", "--xlbomd"], "options select XLBOMD"),
    (["--thermostat", "csvr", "--temperature", "300", "--basis", "6-31g"],
     "--basis '6-31g' differs from the checkpoint's 'sto-3g'"),
    # morse, not an optional backend: without PySCF installed, --backend pyscf
    # failed earlier ("needs PySCF") and this case failed instead of testing
    # the mismatch
    (["--thermostat", "csvr", "--temperature", "300", "--backend", "morse"],
     "written by the 'hf' backend, but this run uses 'morse'"),
    (["--thermostat", "csvr", "--temperature", "300", "--charge", "2"],
     "--charge 2 differs from the checkpoint's 0"),
    (["--thermostat", "csvr", "--temperature", "300", H4], "has atoms ['H', 'H', 'H', 'H']"),
])
def test_restart_mismatch_is_explained_and_changes_nothing(capsys, water_checkpoint, argv,
                                                           message):
    before = (water_checkpoint.read_bytes(), open("energies.csv").read())
    rc, _, err = run(capsys, "run", "--restart", water_checkpoint, "--steps", "2", *argv)
    assert rc == 2 and message in err, err
    assert (water_checkpoint.read_bytes(), open("energies.csv").read()) == before


def _interrupt_at(monkeypatch, step):
    """Make the CLI's per-step callback raise Ctrl-C at ``step``."""
    from aimd import cli
    call = _TIMER_CALL                        # the original, not an earlier patch

    def interrupting(self, rec):
        call(self, rec)
        if rec["step"] == step:
            raise KeyboardInterrupt
    monkeypatch.setattr(cli._StepTimer, "__call__", interrupting)


def test_interrupt_names_only_a_checkpoint_that_exists(capsys, tmp_path, monkeypatch):
    # Regression: Ctrl-C before the first --checkpoint-every boundary printed
    # "the last checkpoint is in c.npz" although no such file existed
    argv = ["run", H4, "--backend", "morse", "--temperature", "300", "--steps", "20",
            "--checkpoint", "c.npz", "--checkpoint-every", "5", "--print-every", "0"]
    _interrupt_at(monkeypatch, 3)
    rc, _, err = run(capsys, *argv)
    assert rc == 130 and not (tmp_path / "c.npz").exists()
    assert "no checkpoint has been written to c.npz yet" in err and "last checkpoint" not in err
    _interrupt_at(monkeypatch, 7)
    rc, _, err = run(capsys, *argv)
    assert rc == 130 and load_checkpoint(tmp_path / "c.npz").step == 5
    assert "the last checkpoint (step 5, t = 2.5 fs) is in c.npz" in err
    # a stale file from an earlier run is not this run's checkpoint
    _interrupt_at(monkeypatch, 3)
    rc, _, err = run(capsys, *argv)
    assert rc == 130 and "no checkpoint has been written to c.npz yet" in err
    # ... but the file a restart started from is
    # (the stale run above rewrote energies.csv / trajectory.xyz up to step 3,
    # so they no longer continue the checkpoint: a restart into them is refused)
    _interrupt_at(monkeypatch, 6)
    rc, _, err = run(capsys, "run", "--restart", "c.npz", *argv[2:])
    assert rc == 2 and "is step 3, but the run wrote step 5 last" in err
    rc, _, err = run(capsys, "run", "--restart", "c.npz", *argv[2:],
                     "--energies", "e2.csv", "--trajectory", "t2.xyz")
    assert rc == 130 and "the last checkpoint (step 5, t = 2.5 fs) is in c.npz" in err


def test_restart_with_changed_settings_warns(capsys, water_checkpoint):
    with pytest.warns(UserWarning, match=r"integrator settings differ.*temperature_k"):
        rc, out, _ = run(capsys, "run", "--restart", water_checkpoint, "--steps", "1",
                         "--thermostat", "csvr", "--temperature", "350", "--print-every", "0")
    assert rc == 0
    assert list(read_energy_log("energies.csv")["step"]) == [0, 1, 2, 3]


def test_damaged_or_missing_restart_file(capsys, water_checkpoint, tmp_path):
    data = water_checkpoint.read_bytes()
    (tmp_path / "cut.npz").write_bytes(data[: len(data) // 2])
    rc, _, err = run(capsys, "run", "--restart", tmp_path / "cut.npz")
    assert rc == 2 and "not a valid aimd checkpoint" in err
    rc, _, err = run(capsys, "run", "--restart", tmp_path / "none.npz")
    assert rc == 2 and "restart file" in err and "not found" in err


# ── aimd analyze: input errors and the optional plot ─────────────────────────

def test_analyze_input_errors(capsys, tmp_path):
    with pytest.warns(RuntimeWarning, match="no dipole"):
        assert run(capsys, "run", H4, "--backend", "morse", "--steps", "4", "--temperature",
                   "100", "--dipoles", "d.csv", "--print-every", "0")[0] == 0
    rc, _, err = run(capsys, "analyze", "geometry", "trajectory.xyz")
    assert rc == 2 and "at least one --bond" in err
    rc, _, err = run(capsys, "analyze", "geometry", "trajectory.xyz", "--bond", "0", "9")
    assert rc == 2 and "atom indices must be in [0, 3]" in err
    rc, _, err = run(capsys, "analyze", "ir", "d.csv")       # morse reports no dipole
    assert rc == 2 and "nan dipoles" in err
    rc, _, err = run(capsys, "analyze", "stats", "energies.csv", "--columns", "foo")
    assert rc == 2 and "no column(s) ['foo']" in err
    rc, _, err = run(capsys, "analyze", "vdos", "trajectory.xyz")   # positions, not velocities
    assert rc == 2 and "velocity unit" in err
    rc, _, err = run(capsys, "analyze", "rdf", "trajectory.xyz", "--pair", "O", "H")
    assert rc == 2 and "no atoms of element 'O'" in err
    # regression: the half-box error used to show bohr values without a unit
    rc, _, err = run(capsys, "analyze", "rdf", "trajectory.xyz", "--pair", "H", "H",
                     "--box", "5", "--r-max", "6")
    assert rc == 2 and "--r-max 6 A exceeds half the box (2.5 A)" in err
    # regression: a velocity file used to be read as angstrom coordinates
    assert run(capsys, "run", H4, "--backend", "morse", "--steps", "4", "--temperature",
               "100", "--velocities", "v.xyz", "--print-every", "0")[0] == 0
    for argv in (["rdf", "v.xyz", "--pair", "H", "H"], ["geometry", "v.xyz", "--bond", "0", "1"]):
        rc, _, err = run(capsys, "analyze", *argv)
        assert rc == 2 and "v.xyz is a velocity file (frame 0: units=bohr/au_time)" in err
        assert not (tmp_path / f"{argv[0]}.csv").exists()


def test_analyze_geometry_uses_circular_statistics_for_dihedrals(capsys, tmp_path):
    # Regression: torsions near +-180 deg (+179 / -179) were averaged linearly
    # (methyl radical example: mean -19.5, std 167 deg for a ~13 deg spread).
    # Frames: H-C-C-H with the last H rotated to phi_k = 180 + 12 sin(k) deg.
    rng = np.random.default_rng(3)
    phi = 180.0 + 12.0 * np.sin(np.arange(200)) + rng.normal(0.0, 1.0, 200)
    frames = []
    for k, p in enumerate(np.deg2rad(phi)):
        xyz = [(-1.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.5, 0.0, 0.0),
               (2.5, np.cos(p), np.sin(p))]
        frames.append("4\nstep=%d t=%gfs\n" % (k, 0.5 * k)
                      + "".join(f"{s} {x:.10f} {y:.10f} {z:.10f}\n"
                                for s, (x, y, z) in zip("HCCH", xyz)))
    (tmp_path / "t.xyz").write_text("".join(frames))
    rc, out, _ = run(capsys, "analyze", "geometry", "t.xyz", "--dihedral", "0", "1", "2", "3")
    assert rc == 0
    line = next(ln for ln in out.splitlines() if "dihedral_0_1_2_3_deg" in ln).split()
    mean, std = float(line[2]), float(line[5])
    # independent reference: circular mean of exp(i phi); spread of phi itself
    ref_mean = np.degrees(np.angle(np.exp(1j * np.deg2rad(phi)).mean()))
    assert abs(abs(mean) - abs(ref_mean)) < 0.05 and abs(abs(mean) - 180.0) < 3.0
    assert std == pytest.approx(np.std(phi, ddof=1), abs=1e-3)
    assert "(circular)" in out
    # the CSV keeps the raw (-180, 180] values
    col = np.loadtxt(tmp_path / "geometry.csv", delimiter=",", skiprows=1)[:, 2]
    np.testing.assert_allclose(np.mod(col - phi + 180.0, 360.0) - 180.0, 0.0, atol=1e-6)
    assert col.min() < -170.0 and col.max() > 170.0


def test_plot_is_optional(capsys, monkeypatch, tmp_path):
    assert run(capsys, "run", H4, "--backend", "morse", "--steps", "4", "--temperature",
               "100", "--print-every", "0")[0] == 0
    monkeypatch.setitem(__import__("sys").modules, "matplotlib", None)   # not importable
    rc, out, err = run(capsys, "analyze", "rdf", "trajectory.xyz", "--pair", "H", "H",
                       "--plot", "rdf.png")
    assert rc == 0 and "matplotlib is not installed" in err
    assert (tmp_path / "rdf.csv").exists() and not (tmp_path / "rdf.png").exists()


def test_plot_is_written_with_matplotlib(capsys, tmp_path):
    pytest.importorskip("matplotlib")
    assert run(capsys, "run", H4, "--backend", "morse", "--steps", "4", "--temperature",
               "100", "--print-every", "0")[0] == 0
    rc, _, _ = run(capsys, "analyze", "stats", "energies.csv", "--plot", "e.png")
    assert rc == 0 and (tmp_path / "e.png").stat().st_size > 1000


class _FakeAxes:
    def __init__(self):
        self.lines, self.ylabel = [], None

    def plot(self, x, y, label=None, **kw):
        self.lines.append((label, np.array(y)))

    def set_ylabel(self, text):
        self.ylabel = text

    def set_xlabel(self, text):
        pass

    def legend(self, **kw):
        pass


def _fake_matplotlib(monkeypatch):
    """A recording stand-in for matplotlib (not installed in the dev environment),
    with matplotlib's own refusal of zero subplot rows."""
    import sys
    import types
    figures = []
    mpl, plt = types.ModuleType("matplotlib"), types.ModuleType("matplotlib.pyplot")
    mpl.use = lambda *a, **k: None

    def subplots(nrows, ncols, **kw):
        if nrows < 1:
            raise ValueError(f"Number of rows must be a positive integer, not {nrows}")
        axes = np.array([[_FakeAxes()] for _ in range(nrows)], dtype=object)
        fig = types.SimpleNamespace(tight_layout=lambda: None, axes=axes,
                                    savefig=lambda path, **k: open(path, "w").write("png"))
        figures.append(fig)
        return fig, axes

    plt.subplots, plt.close = subplots, lambda fig: None
    mpl.pyplot = plt
    monkeypatch.setitem(sys.modules, "matplotlib", mpl)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", plt)
    return figures


def test_stats_plot_includes_temperature_and_non_energy_columns(capsys, monkeypatch, tmp_path):
    # Regression: only *_Eh columns were plotted, so --columns temperature_K
    # called subplots(0, 1) and exited 2 after printing the statistics
    assert run(capsys, "run", H4, "--backend", "morse", "--steps", "6", "--temperature",
               "100", "--print-every", "0")[0] == 0
    figures = _fake_matplotlib(monkeypatch)
    rc, out, err = run(capsys, "analyze", "stats", "energies.csv", "--columns",
                       "temperature_K", "--plot", "t.png")
    assert rc == 0, err
    assert (tmp_path / "t.png").exists() and "wrote t.png" in out
    (ax,) = figures[-1].axes[:, 0]
    log = read_energy_log("energies.csv")
    assert ax.ylabel == "T / K" and ax.lines[0][0] == "temperature_K"
    assert np.array_equal(ax.lines[0][1], log["temperature_K"])
    # default columns: energies relative to the start in one panel, T in another
    rc, _, _ = run(capsys, "analyze", "stats", "energies.csv", "--plot", "e.png")
    assert rc == 0
    panels = {ax.ylabel: ax for ax in figures[-1].axes[:, 0]}
    assert set(panels) == {"E - E(0) / Eh", "T / K"}
    energy = dict(panels["E - E(0) / Eh"].lines)
    np.testing.assert_array_equal(energy["total_Eh"], log["total_Eh"] - log["total_Eh"][0])


def test_plot_with_nothing_to_plot_is_a_message_not_an_error(capsys, monkeypatch):
    from aimd.cli_analyze import _plot
    _fake_matplotlib(monkeypatch)
    _plot("x.png", np.arange(3.0), {}, "x", "y")
    assert "nothing to plot" in capsys.readouterr().err


def test_vdos_reads_untagged_velocities_with_units(capsys, tmp_path):
    # Regression: a velocity file without units= tags could not be analysed
    # from the CLI, and the error said "pass units=", which is no CLI option
    assert run(capsys, "run", H4, "--backend", "morse", "--steps", "40", "--temperature",
               "300", "--seed", "1", "--velocities", "v.xyz", "--print-every", "0")[0] == 0
    text = (tmp_path / "v.xyz").read_text()
    (tmp_path / "bare.xyz").write_text(text.replace(" units=bohr/au_time", ""))
    from aimd.units import AU_TIME_TO_FS, BOHR_TO_ANG
    scale = BOHR_TO_ANG / AU_TIME_TO_FS
    lines = text.replace(" units=bohr/au_time", "").splitlines()
    for i, ln in enumerate(lines):                       # the same data in angstrom/fs
        parts = ln.split()
        if len(parts) == 4:
            lines[i] = parts[0] + " " + " ".join(f"{float(v) * scale:.17e}" for v in parts[1:])
    (tmp_path / "bare_aps.xyz").write_text("\n".join(lines) + "\n")

    rc, _, err = run(capsys, "analyze", "vdos", "bare.xyz")
    assert rc == 2 and "--units" in err and "pass units=" not in err
    for name, unit, out in (("v.xyz", None, "ref.csv"), ("bare.xyz", "bohr/au_time", "a.csv"),
                            ("bare_aps.xyz", "angstrom/fs", "b.csv")):
        argv = ["analyze", "vdos", name, "-o", out] + (["--units", unit] if unit else [])
        assert run(capsys, *argv)[0] == 0
    ref = np.loadtxt(tmp_path / "ref.csv", delimiter=",", skiprows=1)
    np.testing.assert_array_equal(np.loadtxt(tmp_path / "a.csv", delimiter=",", skiprows=1), ref)
    np.testing.assert_allclose(np.loadtxt(tmp_path / "b.csv", delimiter=",", skiprows=1), ref,
                               rtol=1e-12, atol=1e-12 * np.abs(ref).max())
    # a units= tag wins over --units
    assert run(capsys, "analyze", "vacf", "v.xyz", "--units", "angstrom/fs", "-o", "c.csv")[0] == 0
    assert run(capsys, "analyze", "vacf", "v.xyz", "-o", "d.csv")[0] == 0
    assert (tmp_path / "c.csv").read_bytes() == (tmp_path / "d.csv").read_bytes()
