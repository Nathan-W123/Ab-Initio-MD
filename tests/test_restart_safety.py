"""
Restart safety: a continuation must stay on the checkpointed run's
potential-energy surface, append only to that run's own output files, keep
its output cadence, and a bad checkpoint path must fail before any compute.

Each test reproduces a defect that used to pass silently (or fail only at the
end of a run); the references are the uninterrupted runs (byte-identical
files) and the checkpoint's own state.
"""

import importlib.util
import io
import json
import zipfile

import numpy as np
import pytest

from aimd.backends.hf import HFBackend
from aimd.backends.morse import MorseBackend
from aimd.checkpoint import backend_fingerprint, load_checkpoint
from aimd.cli import main
from aimd.integrators import XLBOMD, VelocityVerlet
from aimd.md import run_md
from aimd.trajectory import read_energy_log
from conftest import EXAMPLES

WATER = str(EXAMPLES / "water.xyz")
METHANOL = str(EXAMPLES / "methanol.xyz")
H4 = str(EXAMPLES / "h4_cluster.xyz")
HAS_PYSCF = importlib.util.find_spec("pyscf") is not None


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def run(capsys, *argv):
    rc = main([str(a) for a in argv])
    out = capsys.readouterr()
    return rc, out.out, out.err


def _snapshot(tmp_path):
    return {p.name: p.read_bytes() for p in sorted(tmp_path.iterdir()) if p.is_file()}


# ── Electronic-structure settings are recorded and enforced ──────────────────

def test_backend_fingerprint_tells_same_size_bases_and_methods_apart():
    # water: 6-31G** and cc-pVDZ both have 25 Cartesian AOs, so the density
    # shape alone (the old check) cannot tell them apart
    a = HFBackend(["O", "H", "H"], basis="6-31g**")
    b = HFBackend(["O", "H", "H"], basis="cc-pvdz")
    fa, fb = backend_fingerprint(a), backend_fingerprint(b)
    assert fa["nao"] == fb["nao"] == 25 and fa["density_shape"] == fb["density_shape"]
    assert fa["basis"] != fb["basis"]
    assert backend_fingerprint(HFBackend(["O", "H", "H"], basis="6-31G**")) == fa
    assert backend_fingerprint(MorseBackend(["H", "H"])) == {"name": "morse"}


def test_run_md_refuses_a_restart_on_another_basis_of_the_same_size(water, tmp_path):
    """XL-BOMD on 6-31G** -> cc-pVDZ (both nao 25) used to be accepted, reusing
    the 6-31G** density history as the cc-pVDZ guess (E_cons jumped 3.7 mEh)."""
    s = water.copy()
    s.initialize_velocities(300.0, rng=1)
    integ = XLBOMD(HFBackend(s.symbols, basis="6-31g**"), 0.5, k=5)
    ck = tmp_path / "c.npz"
    run_md(s, integ, 2, checkpoint_path=ck)
    ckpt = load_checkpoint(ck)
    assert ckpt.backend_info["basis"] == "6-31g**"
    other = XLBOMD(HFBackend(s.symbols, basis="cc-pvdz"), 0.5, k=5)
    log = tmp_path / "e.csv"
    with pytest.raises(ValueError, match=r"basis: '6-31g\*\*' -> 'cc-pvdz'"):
        run_md(ckpt.system, other, 2, restart=ckpt, energy_log=log)
    assert not log.exists() and other.result is None         # nothing touched
    # the same basis restarts fine
    run_md(ckpt.system, XLBOMD(HFBackend(s.symbols, basis="6-31G**"), 0.5, k=5), 1,
           restart=ckpt)


def test_cli_restart_refuses_another_basis_with_the_same_number_of_aos(capsys, tmp_path):
    rc, _, _ = run(capsys, "run", WATER, "--basis", "6-31g**", "--xlbomd", "--steps", "2",
                   "--temperature", "300", "--thermostat", "csvr", "--seed", "1",
                   "--checkpoint", "c.npz", "--print-every", "0")
    assert rc == 0
    before = _snapshot(tmp_path)
    rc, _, err = run(capsys, "run", "--restart", "c.npz", "--basis", "cc-pvdz", "--xlbomd",
                     "--steps", "2", "--temperature", "300", "--thermostat", "csvr")
    assert rc == 2
    assert "--basis 'cc-pvdz' differs from the checkpoint's '6-31g**'" in err
    assert _snapshot(tmp_path) == before


def test_cli_restart_takes_omitted_backend_settings_from_the_checkpoint(capsys, tmp_path):
    """
    Leaving --backend / --basis / --backend-option off the restart used to
    fall back to the defaults (hf / sto-3g) whenever the density shape
    matched. Now they come from the checkpoint and the result is identical to
    the uninterrupted run; a changed SCF threshold is accepted with a warning,
    --threads silently.
    """
    common = ["--temperature", "300", "--thermostat", "csvr", "--seed", "2",
              "--print-every", "0"]
    setup = ["--basis", "6-31g", "--backend-option", "guess=core",
             "--conv-tol", "1e-11"]
    rc, _, _ = run(capsys, "run", WATER, *setup, "--steps", "4", *common,
                   "--energies", "ref.csv", "--trajectory", "ref.xyz")
    assert rc == 0
    rc, _, _ = run(capsys, "run", WATER, *setup, "--steps", "2", *common,
                   "--checkpoint", "c.npz")
    assert rc == 0
    assert load_checkpoint(tmp_path / "c.npz").run_info["backend_kwargs"] == {
        "basis": "6-31g", "conv_tol": 1e-11, "guess": "core"}
    rc, out, err = run(capsys, "run", "--restart", "c.npz", "--steps", "2",
                       "--threads", "1", *common)
    assert rc == 0, err
    assert "basis      6-31g (13 AOs)" in out
    assert "backend settings from the checkpoint: basis='6-31g'" in out
    assert (tmp_path / "energies.csv").read_bytes() == (tmp_path / "ref.csv").read_bytes()
    assert (tmp_path / "trajectory.xyz").read_bytes() == (tmp_path / "ref.xyz").read_bytes()

    with pytest.warns(UserWarning, match=r"SCF settings differ.*conv_tol 1e-11 -> 1e-09"):
        rc, _, _ = run(capsys, "run", "--restart", "c.npz", "--steps", "1",
                       "--conv-tol", "1e-9", *common)
    assert rc == 0
    # the reference changes the surface (and the density's meaning): refused
    before = _snapshot(tmp_path)
    rc, _, err = run(capsys, "run", "--restart", "c.npz", "--steps", "1",
                     "--reference", "uhf", *common)
    assert rc == 2 and "--reference 'uhf' differs from the checkpoint's 'auto'" in err
    rc, _, err = run(capsys, "run", "--restart", "c.npz", "--steps", "1",
                     "--backend-option", "gradient_screening=1e-3", *common)
    assert rc == 2 and "--backend-option gradient_screening 0.001 differs" in err
    assert _snapshot(tmp_path) == before


@pytest.mark.skipif(not HAS_PYSCF, reason="needs PySCF")
def test_cli_restart_without_method_stays_on_b3lyp(capsys, tmp_path):
    """The reported defect: B3LYP silently became RHF (E_cons jumped 0.40 Eh)."""
    common = ["--backend", "pyscf", "--basis", "sto-3g", "--init-temperature", "300",
              "--seed", "1", "--print-every", "0"]
    rc, _, _ = run(capsys, "run", WATER, "--method", "b3lyp", *common, "--steps", "3",
                   "--energies", "ref.csv", "--trajectory", "ref.xyz")
    assert rc == 0
    rc, _, _ = run(capsys, "run", WATER, "--method", "b3lyp", *common, "--steps", "2",
                   "--checkpoint", "c.npz")
    assert rc == 0
    before = _snapshot(tmp_path)
    rc, _, err = run(capsys, "run", "--restart", "c.npz", "--method", "hf", "--steps", "1")
    assert rc == 2 and "--method 'hf' differs from the checkpoint's 'b3lyp'" in err
    assert _snapshot(tmp_path) == before
    rc, out, err = run(capsys, "run", "--restart", "c.npz", "--steps", "1",
                       "--print-every", "0")
    assert rc == 0, err
    assert "method     RKS(b3lyp)" in out
    ref, log = read_energy_log(tmp_path / "ref.csv"), read_energy_log(tmp_path / "energies.csv")
    # The uninterrupted run, but not bitwise: PySCF's threaded DFT
    # integration differs in the last bits (measured max 8.5e-14 Eh over three
    # repeats). The defect was a 0.40 Eh jump (B3LYP -> RHF).
    assert list(log["step"]) == [0, 1, 2, 3]
    np.testing.assert_allclose(log["conserved_Eh"], ref["conserved_Eh"], rtol=0, atol=1e-10)


def test_old_checkpoints_without_settings_still_hit_the_density_check(capsys, tmp_path):
    """Checkpoints written before settings were recorded keep the shape check."""
    rc, _, _ = run(capsys, "run", WATER, "--steps", "1", "--checkpoint", "c.npz",
                   "--print-every", "0")
    assert rc == 0
    # strip the new metadata, as an older aimd wrote the file
    with zipfile.ZipFile(tmp_path / "c.npz") as z:
        members = {n: z.read(n) for n in z.namelist()}
    with np.load(tmp_path / "c.npz") as npz:
        meta = json.loads(str(npz["__metadata__"]))
    del meta["backend_info"], meta["run"]
    buf = io.BytesIO()
    np.save(buf, np.array(json.dumps(meta)))
    members["__metadata__.npy"] = buf.getvalue()
    with zipfile.ZipFile(tmp_path / "old.npz", "w") as z:
        for n, data in members.items():
            z.writestr(n, data)
    old = load_checkpoint(tmp_path / "old.npz")
    assert old.backend_info == {} and old.run_info == {} and old.step == 1
    with pytest.warns(UserWarning, match="does not record the backend settings"):
        rc, _, err = run(capsys, "run", "--restart", "old.npz", "--basis", "6-31g",
                         "--steps", "1", "--energies", "e2.csv", "--trajectory", "t2.xyz")
    assert rc == 2 and "SCF density of shape (7, 7), but this backend expects (13, 13)" in err


# ── Output files must continue the checkpointed run ──────────────────────────

def test_restart_refuses_outputs_overwritten_by_another_molecule(capsys, tmp_path):
    """The reported repro: methanol in between, then a water restart spliced
    10 water frames onto 21 methanol frames (exit 0)."""
    rc, _, _ = run(capsys, "run", WATER, "--backend", "morse", "--temperature", "300",
                   "--seed", "1", "--steps", "6", "--checkpoint", "w.npz", "--print-every", "0")
    assert rc == 0
    rc, _, _ = run(capsys, "run", METHANOL, "--backend", "morse", "--temperature", "300",
                   "--steps", "9", "--print-every", "0")
    assert rc == 0
    before = _snapshot(tmp_path)
    rc, _, err = run(capsys, "run", "--restart", "w.npz", "--temperature", "300",
                     "--seed", "1", "--steps", "3", "--print-every", "0")
    assert rc == 2 and "cannot append to" in err and "it does not continue" in err
    assert _snapshot(tmp_path) == before                   # nothing truncated


def test_restart_refuses_same_molecule_outputs_of_another_run(capsys, tmp_path):
    common = ["--backend", "morse", "--temperature", "300", "--print-every", "0"]
    rc, _, _ = run(capsys, "run", H4, *common, "--seed", "1", "--steps", "6",
                   "--checkpoint", "c.npz")
    assert rc == 0
    # same atoms, same steps, other velocities: only the values tell
    rc, _, _ = run(capsys, "run", H4, *common, "--seed", "2", "--steps", "8",
                   "--velocities", "v.xyz")
    assert rc == 0
    for out, message in (
            (["--energies", "none"], "cannot append to trajectory.xyz: its step-6 "
                                     "positions differ from the checkpoint's"),
            (["--trajectory", "none"], "cannot append to energies.csv: its step-6 "
                                       "potential_Eh = ")):
        before = _snapshot(tmp_path)
        rc, _, err = run(capsys, "run", "--restart", "c.npz", *common, "--seed", "1",
                         "--steps", "2", *out)
        assert rc == 2 and message in err, err
        assert _snapshot(tmp_path) == before
    # velocities alone
    rc, _, err = run(capsys, "run", "--restart", "c.npz", *common, "--steps", "2",
                     "--energies", "e3.csv", "--trajectory", "t3.xyz", "--velocities", "v.xyz")
    assert rc == 2 and "velocities are not the checkpoint's" in err


def test_restart_refuses_outputs_that_end_before_the_checkpoint(capsys, tmp_path):
    common = ["--backend", "morse", "--temperature", "300", "--seed", "1",
              "--print-every", "0"]
    rc, _, _ = run(capsys, "run", H4, *common, "--steps", "6", "--checkpoint", "c.npz")
    assert rc == 0
    lines = (tmp_path / "energies.csv").read_text().splitlines(keepends=True)
    (tmp_path / "energies.csv").write_text("".join(lines[:5]))      # steps 0..3 only
    rc, _, err = run(capsys, "run", "--restart", "c.npz", *common, "--steps", "2")
    assert rc == 2 and "is step 3, but the run wrote step 6 last" in err


def test_legitimate_restart_still_appends_identically(capsys, tmp_path):
    """Control: the checks accept a file that does continue the run (with all
    four outputs, write_every 2 not dividing the checkpoint step 5)."""
    common = ["--backend", "morse", "--temperature", "300", "--thermostat", "csvr",
              "--seed", "4", "--print-every", "0", "--write-every", "2"]
    outs = ["--velocities", "v.xyz", "--dipoles", "d.csv"]
    with pytest.warns(RuntimeWarning, match="no dipole"):        # morse: nan dipoles
        rc, _, _ = run(capsys, "run", H4, *common, "--steps", "9", "--energies", "ref.csv",
                       "--trajectory", "ref.xyz", "--velocities", "refv.xyz",
                       "--dipoles", "refd.csv")
    assert rc == 0
    with pytest.warns(RuntimeWarning, match="no dipole"):
        rc, _, _ = run(capsys, "run", H4, *common, *outs, "--steps", "5",
                       "--checkpoint", "c.npz")
    assert rc == 0
    with pytest.warns(RuntimeWarning, match="no dipole"):
        rc, _, err = run(capsys, "run", "--restart", "c.npz", *common, *outs, "--steps", "4")
    assert rc == 0, err
    for a, b in (("energies.csv", "ref.csv"), ("trajectory.xyz", "ref.xyz"),
                 ("v.xyz", "refv.xyz"), ("d.csv", "refd.csv")):
        assert (tmp_path / a).read_bytes() == (tmp_path / b).read_bytes(), a


def test_restart_with_another_write_every_is_refused(capsys, tmp_path):
    common = ["--backend", "morse", "--temperature", "300", "--seed", "3",
              "--print-every", "0"]
    rc, _, _ = run(capsys, "run", H4, *common, "--steps", "8", "--write-every", "4",
                   "--checkpoint", "c.npz")
    assert rc == 0
    assert load_checkpoint(tmp_path / "c.npz").run_info["write_every"] == 4
    before = _snapshot(tmp_path)
    rc, _, err = run(capsys, "run", "--restart", "c.npz", *common, "--steps", "4",
                     "--write-every", "1")
    assert rc == 2 and "write_every 1 differs from the checkpointed run's 4" in err
    assert _snapshot(tmp_path) == before
    # into new files a different cadence is fine
    rc, _, _ = run(capsys, "run", "--restart", "c.npz", *common, "--steps", "4",
                   "--write-every", "1", "--energies", "e2.csv", "--trajectory", "t2.xyz")
    assert rc == 0
    assert list(read_energy_log(tmp_path / "e2.csv")["step"]) == [8, 9, 10, 11, 12]


# ── Checkpoint path is checked before the run ────────────────────────────────

def test_unwritable_checkpoint_path_fails_before_the_first_step(capsys, tmp_path):
    rc, out, err = run(capsys, "run", H4, "--backend", "morse", "--steps", "50",
                       "--temperature", "300", "--checkpoint", "nodir/c.npz")
    assert rc == 2
    assert "cannot write checkpoint nodir/c.npz: directory nodir does not exist" in err
    assert out == "" and not (tmp_path / "energies.csv").exists()
    rc, _, err = run(capsys, "run", H4, "--backend", "morse", "--steps", "1",
                     "--checkpoint", str(tmp_path))
    assert rc == 2 and "is a directory" in err


def test_run_md_checks_the_checkpoint_path_before_computing(h4, tmp_path):
    calls = []

    class Counting(MorseBackend):
        def compute(self, positions):
            calls.append(1)
            return super().compute(positions)

    s = h4.copy()
    with pytest.raises(ValueError, match="does not exist"):
        run_md(s, VelocityVerlet(Counting(s.symbols), 0.2), 100,
               checkpoint_path=tmp_path / "missing" / "c.npz")
    assert calls == []
    # a valid path leaves no temporary file behind
    run_md(s, VelocityVerlet(Counting(s.symbols), 0.2), 1, checkpoint_path=tmp_path / "c.npz")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c.npz"]


# ── Seed ──────────────────────────────────────────────────────────────────────

def test_negative_seed_is_a_one_line_usage_error(capsys):
    rc, _, err = run(capsys, "run", H4, "--backend", "morse", "--seed", "-1",
                     "--temperature", "300", "--steps", "1")
    assert rc == 2 and "--seed: must be >= 0, got -1" in err and "Traceback" not in err

