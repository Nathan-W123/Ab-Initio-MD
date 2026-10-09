"""
``aimd view``: the HTML trajectory viewer (aimd/viewer.py).

The page's data payload is checked against the files it was built from; the
embedded JavaScript is syntax-checked with node when node is installed. (The
rendering itself was checked in headless Chromium when the viewer was written:
no console errors in light, dark and phone layouts; playback, rotation and
chart seeking work.)
"""

import json
import shutil
import subprocess

import numpy as np
import pytest

from aimd.backends.morse import MorseBackend
from aimd.cli import main
from aimd.integrators import VelocityVerlet
from aimd.md import run_md
from aimd.trajectory import read_energy_log, read_xyz
from aimd.units import BOHR_TO_ANG, HARTREE_TO_KCALMOL
from aimd.viewer import build_viewer_data, find_energy_log, render_html, write_viewer

MARKER = '<script type="application/json" id="aimd-data">'


def payload(html_text):
    return json.loads(html_text.split(MARKER, 1)[1].split("</script>", 1)[0])


@pytest.fixture
def run_files(tmp_path, h4):
    h4.initialize_velocities(300.0, rng=1)
    traj, log = tmp_path / "traj.xyz", tmp_path / "energies.csv"
    run_md(h4, VelocityVerlet(MorseBackend(h4.symbols), 0.5), 10,
           trajectory=traj, energy_log=log)
    return traj, log


def _distances(frames):
    d = frames[:, :, None, :] - frames[:, None, :, :]
    return np.linalg.norm(d, axis=-1)


def test_frames_keep_geometry_and_are_centred(run_files):
    traj, log = run_files
    data = build_viewer_data(traj, energies=log)
    ref = read_xyz(traj)
    frames = np.array(data["frames"]).reshape(ref.n_frames, ref.n_atoms, 3)
    assert data["symbols"] == ref.symbols and len(data["frames"]) == 11
    # centring is a rigid translation: interatomic distances unchanged (1e-4 A rounding)
    assert np.allclose(_distances(frames), _distances(ref.positions * BOHR_TO_ANG), atol=3e-4)
    com = np.einsum("i,fij->fj", ref.masses, frames) / ref.masses.sum()
    assert np.abs(com).max() < 1e-3
    assert data["steps"] == list(range(11))
    assert data["times"] == pytest.approx(list(np.arange(11) * 0.5))


def test_energies_are_matched_by_step_and_relative_to_their_own_start(run_files):
    traj, log = run_files
    data = build_viewer_data(traj, energies=log, every=3)
    assert data["steps"] == [0, 3, 6, 9] and data["stride"] == 3
    e, ref = data["energies"], read_energy_log(log)
    for key in ("potential_Eh", "total_Eh", "conserved_Eh"):
        assert e[key] == pytest.approx(list(ref[key][[0, 3, 6, 9]]), abs=1e-10)
        rel = e[key.replace("_Eh", "_kcal")]
        assert rel[0] == 0.0
        assert rel == pytest.approx(list((ref[key][[0, 3, 6, 9]] - ref[key][0])
                                         * HARTREE_TO_KCALMOL), abs=1e-6)
    assert e["temperature_K"] == pytest.approx(list(ref["temperature_K"][[0, 3, 6, 9]]),
                                               abs=1e-4)


def test_max_frames_raises_the_stride(run_files):
    traj, _ = run_files
    data = build_viewer_data(traj, max_frames=4)
    assert data["stride"] == 3 and data["steps"] == [0, 3, 6, 9] and data["totalFrames"] == 11
    assert build_viewer_data(traj, max_frames=None)["stride"] == 1


def test_hand_written_xyz_without_steps(tmp_path):
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nfirst\nH 0 0 0\nH 0 0 0.74\n2\nsecond\nH 0 0 0\nH 0 0 3.0\n")
    log = tmp_path / "e.csv"
    log.write_text("step,time_fs,potential_Eh,kinetic_Eh,total_Eh,temperature_K,conserved_Eh\n"
                   "0,0.0,-1.0,0.1,-0.9,300.0,-0.9\n1,0.5,-0.9,0.0,-0.9,0.0,-0.9\n")
    data = build_viewer_data(xyz, energies=log)
    assert data["steps"] is None and data["energies"]["potential_Eh"] == [-1.0, -0.9]
    assert data["times"] == [0.0, 0.5]                      # taken from the log
    log.write_text(log.read_text() + "2,1.0,-0.8,0.0,-0.8,0.0,-0.8\n")
    with pytest.warns(RuntimeWarning, match="row count"):
        assert build_viewer_data(xyz, energies=log)["energies"] is None


def test_unmatched_steps_warn(run_files, tmp_path):
    traj, _ = run_files
    other = tmp_path / "other.csv"
    other.write_text("step,time_fs,potential_Eh,kinetic_Eh,total_Eh,temperature_K,conserved_Eh\n"
                     "500,0.0,-1.0,0.1,-0.9,300.0,-0.9\n")
    with pytest.warns(RuntimeWarning, match="no row matches"):
        assert build_viewer_data(traj, energies=other)["energies"] is None


def test_energy_log_discovery(tmp_path):
    traj = tmp_path / "run.xyz"
    traj.write_text("1\n\nH 0 0 0\n")
    assert find_energy_log(traj) is None
    (tmp_path / "energies.csv").write_text("step\n0\n")
    assert find_energy_log(traj) == tmp_path / "energies.csv"
    (tmp_path / "run_energies.csv").write_text("step\n0\n")
    assert find_energy_log(traj) == tmp_path / "run_energies.csv"


def test_payload_cannot_break_out_of_the_script_element(run_files):
    traj, _ = run_files
    page = render_html(build_viewer_data(traj, title="a</script><b>x</b>"))
    assert page.count("</script>") == 2                     # data element + viewer code
    assert "<title>a&lt;/script&gt;&lt;b&gt;x&lt;/b&gt;</title>" in page
    assert payload(page)["title"] == "a</script><b>x</b>"


def test_invalid_arguments(run_files):
    traj, _ = run_files
    for kw in ({"every": 0}, {"max_frames": 0}, {"bond_tolerance": -0.1},
               {"bond_tolerance": float("nan")}):
        with pytest.raises(ValueError):
            build_viewer_data(traj, **kw)


def test_cli_view(run_files, tmp_path, capsys):
    traj, log = run_files
    assert main(["view", str(traj)]) == 0
    out = capsys.readouterr().out
    assert f"wrote {traj.with_suffix('.html')}: 11 frames, 4 atoms; energies: {log}" in out
    page = payload(traj.with_suffix(".html").read_text())
    assert page["energies"]["file"] == str(log)

    html_out = tmp_path / "movie.html"
    assert main(["view", str(traj), "-o", str(html_out), "--energies", "none",
                 "--every", "2", "--title", "H4"]) == 0
    page = payload(html_out.read_text())
    assert page["energies"] is None and page["title"] == "H4" and len(page["frames"]) == 6


def test_cli_view_errors(run_files, tmp_path, capsys):
    traj, _ = run_files
    assert main(["view", str(tmp_path / "missing.xyz")]) == 2
    assert "file not found" in capsys.readouterr().err
    vel = tmp_path / "v.xyz"
    vel.write_text("1\nstep=0 t=0fs units=bohr/au_time\nH 0 0 0\n")
    assert main(["view", str(vel)]) == 2
    assert "velocity file" in capsys.readouterr().err
    assert main(["view", str(traj), "--every", "0"]) == 2


def test_written_page_is_complete(run_files):
    traj, log = run_files
    out = write_viewer(traj, energies=log)
    page = out.read_text()
    assert page.startswith("<!doctype html>") and page.rstrip().endswith("</html>")
    assert "http://" not in page and "https://" not in page    # no network needed


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_embedded_javascript_parses(run_files, tmp_path):
    traj, log = run_files
    page = write_viewer(traj, energies=log).read_text()
    js = tmp_path / "viewer.js"
    js.write_text(page.split("<script>\n", 1)[1].rsplit("</script>", 1)[0])
    proc = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
