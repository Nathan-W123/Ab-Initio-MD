"""
End-to-end runs of the ``aimd`` command with the native HF backend, checked
against independent references:

  - the whole NVE pipeline (CLI -> backend -> integrator -> files) against the
    same run driven by PySCF forces (Cartesian basis, as aimd.qc uses);
  - XL-BOMD against plain BOMD at tight SCF convergence;
  - thermostatted runs (CSVR, Nose-Hoover chain, Berendsen, Langevin) through
    their conserved quantities;
  - restart via the CLI against an uninterrupted run (identical files);
  - the example config file against the equivalent flags;
  - ``aimd analyze`` on the produced files against direct NumPy evaluation,
    exact identities, and PySCF's analytic harmonic frequencies;
  - a UHF radical against PySCF's UHF energy.
"""

import csv

import numpy as np
import pytest

from aimd.cli import main
from aimd.trajectory import read_csv_log, read_dipole_log, read_energy_log, read_xyz
from aimd.units import BOHR_TO_ANG
from conftest import EXAMPLES

WATER = str(EXAMPLES / "water.xyz")
CH3 = str(EXAMPLES / "methyl_radical.xyz")


def cli(capsys, *argv) -> str:
    rc = main([str(a) for a in argv])
    out = capsys.readouterr()
    assert rc == 0, out.err
    return out.out


def summary_value(out: str, key: str) -> float:
    return float(out.split(key)[1].split()[0])


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# ── NVE: native HF against PySCF forces ──────────────────────────────────────

def test_water_nve_matches_the_same_run_with_pyscf_forces(capsys, tmp_path):
    """
    20 steps of HF/STO-3G water at 300 K, 0.5 fs. The PySCF run (cart=True,
    same seed hence the same initial velocities) is an independent
    implementation of the energy and forces. Both SCFs are converged to an
    orbital gradient of 1e-9: the two codes then agree to measured 6.9e-12 Eh
    and 5.9e-9 bohr over the run. (At the default 1e-7 the differences are
    1.6e-9 Eh and 1.5e-6 bohr: they scale with the SCF threshold, i.e. they
    are convergence noise amplified by the dynamics, not an error in either.)
    """
    pytest.importorskip("pyscf")
    common = ["run", WATER, "--steps", "20", "--dt", "0.5", "--temperature", "300",
              "--seed", "11", "--print-every", "0", "--conv-tol", "1e-11",
              "--conv-tol-grad", "1e-9"]
    out = cli(capsys, *common, "--trajectory", "hf.xyz", "--energies", "hf.csv")
    cli(capsys, *common, "--backend", "pyscf", "--backend-option", "cart=true",
        "--trajectory", "ps.xyz", "--energies", "ps.csv")
    hf, ps = read_energy_log("hf.csv"), read_energy_log("ps.csv")
    np.testing.assert_allclose(hf["potential_Eh"], ps["potential_Eh"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(read_xyz("hf.xyz").positions, read_xyz("ps.xyz").positions,
                               rtol=0, atol=5e-8)
    # E_pot(0): the HF/STO-3G minimum (PySCF, cart=True: -74.965901192299)
    assert hf["potential_Eh"][0] == pytest.approx(-74.965901192299, abs=1e-10)
    # NVE: E_tot conserved to velocity Verlet's O(dt^2) fluctuation
    # (measured max |E - E0| = 2.3e-5 Eh, with E_kin ~ 2.9e-3 Eh)
    drift = np.max(np.abs(hf["total_Eh"] - hf["total_Eh"][0]))
    assert drift < 5e-5
    assert summary_value(out, "max |E - E0| = ") == pytest.approx(drift, rel=1e-3)
    assert np.array_equal(hf["conserved_Eh"], hf["total_Eh"])
    assert "basis      sto-3g (7 AOs)" in out and "method     RHF" in out


def test_nve_energy_error_scales_as_dt_squared(capsys):
    """Halving dt divides the max |E_tot - E_tot(0)| over the same 10 fs by ~4."""
    drifts = []
    for dt, steps in ((0.5, 20), (0.25, 40)):
        out = cli(capsys, "run", WATER, "--steps", steps, "--dt", dt, "--temperature", "300",
                  "--seed", "11", "--print-every", "0", "--trajectory", "none",
                  "--energies", f"e{dt}.csv")
        drifts.append(summary_value(out, "max |E - E0| = "))
    assert 3.5 < drifts[0] / drifts[1] < 4.5


# ── XL-BOMD ──────────────────────────────────────────────────────────────────

def test_xlbomd_follows_bomd_and_saves_scf_iterations(capsys):
    """
    With a tight SCF (orbital gradient 1e-9) XL-BOMD (K = 5) and BOMD with
    the previous density as guess integrate the same Born-Oppenheimer
    trajectory: measured max position difference 4.0e-8 bohr after 40 steps
    of water/6-31G* (1.0e-5 bohr at the default 1e-7: the difference scales
    with the SCF threshold). The XL guess saves SCF iterations: measured mean
    11.12 vs 12.00 per step (8.2 vs 9.2 at the default threshold).
    """
    common = ["run", WATER, "--basis", "6-31g*", "--steps", "40", "--temperature", "300",
              "--seed", "5", "--print-every", "0", "--energies", "none",
              "--conv-tol-grad", "1e-9"]
    bo = cli(capsys, *common, "--trajectory", "bo.xyz")
    xl = cli(capsys, *common, "--trajectory", "xl.xyz", "--xlbomd", "--xl-k", "5")
    assert "XL-BOMD (K = 5) + velocity Verlet, NVE" in xl
    x_bo, x_xl = read_xyz("bo.xyz").positions, read_xyz("xl.xyz").positions
    assert np.max(np.abs(x_bo - x_xl)) < 2e-7
    it_bo = summary_value(bo, "SCF iterations   mean ")
    it_xl = summary_value(xl, "SCF iterations   mean ")
    assert it_xl <= it_bo - 0.5


# ── Thermostats ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("argv, cons_bound", [
    (["--thermostat", "csvr", "--tau", "10"], 1e-4),
    (["--thermostat", "nhc", "--tau", "20"], 1e-4),
    (["--thermostat", "berendsen", "--tau", "10"], 1e-4),
    (["--thermostat", "langevin", "--friction", "0.05"], 2e-4),
    (["--thermostat", "csvr", "--tau", "10", "--xlbomd"], 1e-4),
], ids=["csvr", "nhc", "berendsen", "langevin", "csvr-xlbomd"])
def test_thermostatted_runs_conserve_their_extended_energy(capsys, argv, cons_bound):
    """
    Water started at 50 K with a 600 K target, 60 steps of 0.5 fs: the
    thermostat changes E_tot by far more than the conserved quantity
    E_tot - (energy added by the thermostat) drifts. Measured max |dE_tot| /
    max |dE_cons| (Eh): csvr 5.1e-3 / 4.0e-5, nhc 1.7e-2 / 5.1e-5, berendsen
    5.8e-3 / 3.1e-5, langevin 1.6e-2 / 9.8e-5, csvr + XL-BOMD 5.1e-3 / 4.0e-5.
    The bounds are ~2x those: E_cons has velocity Verlet's O(dt^2)
    fluctuation, which grows with the temperature reached (up to ~1200 K).
    """
    out = cli(capsys, "run", WATER, "--steps", "60", "--dt", "0.5", "--temperature", "600",
              "--init-temperature", "50", "--seed", "3", "--print-every", "0",
              "--trajectory", "none", *argv)
    log = read_energy_log("energies.csv")
    d_tot = np.max(np.abs(log["total_Eh"] - log["total_Eh"][0]))
    d_cons = np.max(np.abs(log["conserved_Eh"] - log["conserved_Eh"][0]))
    assert d_cons < cons_bound
    assert d_tot > 30 * d_cons
    assert log["temperature_K"][0] == pytest.approx(50.0)
    assert summary_value(out, "max |E - E0| = ") == pytest.approx(d_cons, rel=1e-3)


# ── Restart through the CLI ──────────────────────────────────────────────────

@pytest.mark.parametrize("argv", [
    ["--thermostat", "csvr", "--temperature", "300", "--xlbomd"],
    ["--thermostat", "langevin", "--temperature", "300"],
    ["--thermostat", "nhc", "--temperature", "300", "--tau", "20"],
])
def test_restart_reproduces_the_uninterrupted_run(capsys, tmp_path, argv):
    """
    Run A: 10 steps with a checkpoint (kept as c10.npz), continued to step 15
    as if it had then crashed. Restarting from c10.npz for 10 steps must cut
    the outputs back to step 10 and give files identical, byte for byte, to
    an uninterrupted 20-step run (RNG streams, thermostat variables and the
    XL-BOMD density history all continue exactly).
    """
    common = ["run", WATER, "--seed", "8", "--print-every", "0", *argv]
    outs = ["--trajectory", "t.xyz", "--energies", "e.csv", "--velocities", "v.xyz",
            "--dipoles", "d.csv"]
    cli(capsys, *common, "--steps", "20", "--trajectory", "ref.xyz", "--energies", "ref.csv",
        "--velocities", "refv.xyz", "--dipoles", "refd.csv")
    cli(capsys, *common, "--steps", "10", *outs, "--checkpoint", "c.npz")
    (tmp_path / "c10.npz").write_bytes((tmp_path / "c.npz").read_bytes())
    cli(capsys, *common, "--steps", "5", *outs, "--restart", "c.npz", "--checkpoint", "c.npz")
    assert list(read_energy_log("e.csv")["step"]) == list(range(16))
    out = cli(capsys, *common, "--steps", "10", *outs, "--restart", "c10.npz")
    assert "restart    c10.npz: step 10" in out
    for mine, ref in (("t.xyz", "ref.xyz"), ("e.csv", "ref.csv"), ("v.xyz", "refv.xyz"),
                      ("d.csv", "refd.csv")):
        assert (tmp_path / mine).read_bytes() == (tmp_path / ref).read_bytes(), mine


# ── Config file ──────────────────────────────────────────────────────────────

def test_example_config_equals_its_flags(capsys, tmp_path):
    """examples/water_nvt.toml (shortened) and the same options as flags."""
    short = ["--steps", "6", "--basis", "sto-3g", "--checkpoint", "none",
             "--print-every", "0"]
    cli(capsys, "run", "--config", EXAMPLES / "water_nvt.toml", *short)
    cli(capsys, "run", WATER, "--backend", "hf", "--method", "hf", "--conv-tol", "1e-10",
        "--conv-tol-grad", "1e-7", "--dt", "0.5", "--thermostat", "csvr",
        "--temperature", "300", "--tau", "100", "--xlbomd", "--xl-k", "5",
        "--remove-rotation", "--seed", "2024", "--backend-option",
        'scf_options={"diis_space": 8}', "--trajectory", "flags.xyz",
        "--energies", "flags.csv", *short)
    assert (tmp_path / "water_nvt_energies.csv").read_text() == \
        (tmp_path / "flags.csv").read_text()
    assert (tmp_path / "water_nvt.xyz").read_text() == (tmp_path / "flags.xyz").read_text()
    assert (tmp_path / "water_nvt_velocities.xyz").exists()
    assert (tmp_path / "water_nvt_dipoles.csv").exists()


# ── aimd analyze on a produced trajectory ────────────────────────────────────

@pytest.fixture(scope="module")
def water_run(tmp_path_factory):
    """
    1000 steps (500 fs) of rotation-free NVE water at HF/STO-3G started at
    300 K from the minimum; positions, velocities, energies and dipoles.
    """
    d = tmp_path_factory.mktemp("water_run")
    rc = main(["run", WATER, "--steps", "1000", "--dt", "0.5", "--temperature", "300",
               "--remove-rotation", "--seed", "2", "--print-every", "0",
               "--trajectory", str(d / "t.xyz"), "--energies", str(d / "e.csv"),
               "--velocities", str(d / "v.xyz"), "--dipoles", str(d / "d.csv")])
    assert rc == 0
    return d


@pytest.fixture(scope="module")
def harmonic_wavenumbers():
    """PySCF analytic RHF Hessian at examples/water.xyz (cart=True)."""
    pytest.importorskip("pyscf")
    from pyscf import gto, scf
    from pyscf.hessian import thermo

    atoms = [(s, x) for s, x in zip(read_xyz(WATER).symbols, read_xyz(WATER).positions[0])]
    mol = gto.M(atom=atoms, unit="Bohr", basis="sto-3g", cart=True, verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    return np.sort(thermo.harmonic_analysis(mol, mf.Hessian().kernel())["freq_wavenumber"])


def test_analyze_geometry_matches_direct_evaluation(capsys, water_run):
    out = cli(capsys, "analyze", "geometry", water_run / "t.xyz", "--bond", "0", "1",
              "--bond", "0", "2", "--angle", "1", "0", "2", "-o", "g.csv")
    g = read_csv_log("g.csv")
    x = read_xyz(water_run / "t.xyz").positions * BOHR_TO_ANG
    r1, r2 = np.linalg.norm(x[:, 1] - x[:, 0], axis=1), np.linalg.norm(x[:, 2] - x[:, 0], axis=1)
    cos = np.einsum("fi,fi->f", x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]) / (r1 * r2)
    np.testing.assert_allclose(g["bond_0_1_angstrom"], r1, rtol=1e-12)
    np.testing.assert_allclose(g["bond_0_2_angstrom"], r2, rtol=1e-12)
    np.testing.assert_allclose(g["angle_1_0_2_deg"], np.degrees(np.arccos(cos)), atol=1e-9)
    assert list(g["step"]) == list(range(1001))
    assert f"mean {r1.mean():10.4f} A" in out
    # vibrating about the HF/STO-3G minimum, 0.9894 A / 100.03 deg
    assert abs(r1.mean() - 0.9894) < 0.01 and abs(g["angle_1_0_2_deg"].mean() - 100.03) < 2


def test_analyze_rdf_counts_the_two_oh_bonds(capsys, water_run):
    out = cli(capsys, "analyze", "rdf", water_run / "t.xyz", "--pair", "O", "H",
              "--r-max", "1.5", "--bins", "150")
    rdf = read_csv_log("rdf.csv")
    assert rdf["coordination"][-1] == pytest.approx(2.0, abs=1e-12)   # n_OH within 1.5 A
    r_peak = rdf["r_angstrom"][np.argmax(rdf["g_r"])]
    oh = np.linalg.norm(np.diff(read_xyz(water_run / "t.xyz").positions[:, :2], axis=1), axis=2)
    assert abs(r_peak - oh.mean() * BOHR_TO_ANG) < 0.01                # bin width 0.01 A
    assert "highest peak at r =" in out


def test_analyze_vacf_starts_at_twice_the_kinetic_energy(capsys, water_run):
    cli(capsys, "analyze", "vacf", water_run / "v.xyz", "--max-lag", "200")
    vacf = read_csv_log("vacf.csv")
    log = read_energy_log(water_run / "e.csv")
    # C(0) = <sum_i m_i v_i^2> = 2 <E_kin> over the same frames
    assert vacf["vacf"][0] == pytest.approx(2.0 * log["kinetic_Eh"].mean(), rel=1e-10)
    assert vacf["vacf_normalized"][0] == 1.0
    assert vacf["time_fs"][-1] == pytest.approx(100.0)


def test_analyze_vdos_and_ir_find_the_harmonic_bands(capsys, water_run, harmonic_wavenumbers):
    """
    PySCF harmonic wavenumbers: bend 2169.9, stretches 4139.6 / 4390.7 cm^-1.
    The 500 fs run resolves 133 cm^-1 (Hann window), which separates the bend
    but merges the two stretches into one band. Measured bend peak: VDOS
    2172.8, IR 2173.0 cm^-1 (classical anharmonic shift at ~150 K is a few
    cm^-1); the stretch band 4417 / 4415 cm^-1, between the two stretches.
    """
    bend, sym, asym = harmonic_wavenumbers
    out = cli(capsys, "analyze", "vdos", water_run / "v.xyz")
    vdos = read_csv_log("vdos.csv")
    out_ir = cli(capsys, "analyze", "ir", water_run / "d.csv", "--temperature", "300")
    ir = read_csv_log("ir.csv")
    for spec in (vdos, ir):
        w, y = spec["wavenumber_cm-1"], spec["intensity"]
        for lo, hi, ref_lo, ref_hi in ((1500, 3000, bend, bend), (3500, 5000, sym, asym)):
            sel = (w > lo) & (w < hi)
            peak = w[sel][np.argmax(y[sel])]
            assert ref_lo - 15 < peak < ref_hi + 40, (peak, ref_lo, ref_hi)
    peaks = [float(line.split()[0]) for line in out.splitlines() if "cm^-1   " in line]
    assert abs(peaks[0] - bend) < 15
    assert "km mol^-1 per cm^-1" in out_ir


def test_analyze_stats_matches_numpy(capsys, water_run):
    out = cli(capsys, "analyze", "stats", water_run / "e.csv", "--skip", "100", "-o", "s.csv")
    log = read_energy_log(water_run / "e.csv")
    with open("s.csv", newline="") as fh:
        rows = {r["column"]: r for r in csv.DictReader(fh)}
    assert set(rows) == {"potential_Eh", "kinetic_Eh", "total_Eh", "temperature_K",
                         "conserved_Eh"}
    for name, row in rows.items():
        x = log[name][100:]
        assert float(row["mean"]) == pytest.approx(x.mean(), rel=1e-12)
        assert float(row["std"]) == pytest.approx(x.std(ddof=1), rel=1e-12)
        assert int(row["n_samples"]) == 901
    assert all(float(r["sem"]) > 0.0 for r in rows.values())
    assert "901 of 1001 rows (100 skipped)" in out


# ── UHF radical ──────────────────────────────────────────────────────────────

def test_methyl_radical_uhf_run_matches_pyscf(capsys):
    """UHF/6-31G* CH3 doublet: E_pot(0) against PySCF UHF (cart=True)."""
    pyscf = pytest.importorskip("pyscf")
    out = cli(capsys, "run", CH3, "--multiplicity", "2", "--basis", "6-31g*",
              "--steps", "10", "--temperature", "300", "--seed", "1", "--print-every", "0",
              "--dipoles", "d.csv")
    assert "method     UHF" in out and "(21 AOs)" in out
    traj = read_xyz("trajectory.xyz")
    mol = pyscf.gto.M(atom=list(zip(traj.symbols, traj.positions[0])), unit="Bohr",
                      basis="6-31g*", cart=True, spin=1, verbose=0)
    e_ref = pyscf.scf.UHF(mol).run(conv_tol=1e-12).e_tot
    log = read_energy_log("energies.csv")
    assert log["potential_Eh"][0] == pytest.approx(e_ref, abs=1e-9)
    assert np.all(np.isfinite(read_dipole_log("d.csv").dipole))
    assert summary_value(out, "max |E - E0| = ") < 1e-4
