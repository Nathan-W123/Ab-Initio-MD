"""
``aimd analyze``: post-processing of the files written by ``aimd run``.

    aimd analyze rdf trajectory.xyz --pair O H --r-max 6
    aimd analyze geometry trajectory.xyz --bond 0 1 --angle 1 0 2
    aimd analyze vacf velocities.xyz
    aimd analyze vdos velocities.xyz --max-lag 400
    aimd analyze ir dipoles.csv --temperature 300
    aimd analyze stats energies.csv --skip 100

Each subcommand prints a short summary, writes a CSV (``-o``, default
``<command>.csv``; ``stats`` writes one only with ``-o``) and, with
``--plot FILE``, a figure if matplotlib is installed (an optional
dependency: without it the plot is skipped with a message).

Units at this boundary: distances in angstrom, angles in degrees, times in
fs, wavenumbers in cm^-1; the numerics are in aimd.analysis (atomic units).
Atom indices are 0-based, in file order.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aimd import analysis
from aimd.analysis.spectra import QUANTUM_CORRECTIONS, WINDOWS
from aimd.analysis.statistics import DEFAULT_COLUMNS
from aimd.trajectory import (
    VELOCITY_UNITS,
    read_dipole_log,
    read_energy_log,
    read_velocities,
    read_xyz,
)
from aimd.units import ANG_TO_BOHR, BOHR_TO_ANG


class AnalysisInputError(ValueError):
    """Bad analysis input (reported by aimd.cli as a usage error)."""


# ── Parser ────────────────────────────────────────────────────────────────────

def _common(p: argparse.ArgumentParser, default_out: str | None) -> None:
    p.add_argument("-o", "--output", default=default_out,
                   help="CSV file to write" + ("" if default_out else " (optional)"))
    p.add_argument("--plot", default=None, metavar="FILE",
                   help="also save a plot (needs matplotlib)")


def _spectral(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-lag", type=int, default=None,
                   help="correlation length in frames (default n/2); sets the "
                        "resolution 1/(c max_lag dt)")
    p.add_argument("--window", default="hann", choices=WINDOWS, help="lag window")
    p.add_argument("--zero-pad", type=int, default=4, help="zero-padding factor")
    p.add_argument("--skip", type=int, default=0, help="discard the first N frames")
    p.add_argument("--dt", type=float, default=None,
                   help="frame spacing in fs (default: read from the file)")
    p.add_argument("--max-wavenumber", type=float, default=4500.0,
                   help="upper end of the reported / plotted range, cm^-1")


def add_analyze_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    fmt = argparse.ArgumentDefaultsHelpFormatter
    a = sub.add_parser("analyze", help="analyse trajectories and logs written by 'aimd run'",
                       description=__doc__.split("\n\n")[0].strip())
    asub = a.add_subparsers(dest="analysis", required=True)

    p = asub.add_parser("rdf", help="radial distribution function g(r)", formatter_class=fmt)
    p.add_argument("trajectory", help="positions XYZ file")
    p.add_argument("--pair", nargs=2, required=True, metavar=("A", "B"),
                   help="element symbols or comma-separated 0-based atom indices")
    p.add_argument("--r-max", type=float, default=6.0, help="histogram range, angstrom")
    p.add_argument("--bins", type=int, default=200)
    p.add_argument("--box", type=float, default=None,
                   help="periodic cubic box edge, angstrom (default: isolated cluster)")
    p.add_argument("--skip", type=int, default=0, help="discard the first N frames")
    _common(p, "rdf.csv")
    p.set_defaults(func=cmd_rdf)

    p = asub.add_parser("geometry", help="bond lengths, angles, dihedrals over time",
                        formatter_class=fmt)
    p.add_argument("trajectory", help="positions XYZ file")
    p.add_argument("--bond", nargs=2, type=int, action="append", default=[],
                   metavar=("I", "J"), help="bond length I-J (angstrom); repeatable")
    p.add_argument("--angle", nargs=3, type=int, action="append", default=[],
                   metavar=("I", "J", "K"), help="angle I-J-K (degrees); repeatable")
    p.add_argument("--dihedral", nargs=4, type=int, action="append", default=[],
                   metavar=("I", "J", "K", "L"),
                   help="dihedral I-J-K-L (degrees, IUPAC sign); repeatable")
    p.add_argument("--skip", type=int, default=0,
                   help="discard the first N frames from the statistics")
    _common(p, "geometry.csv")
    p.set_defaults(func=cmd_geometry)

    for name, helptext in (("vacf", "velocity autocorrelation function"),
                           ("vdos", "vibrational density of states (VACF spectrum)")):
        p = asub.add_parser(name, help=helptext, formatter_class=fmt)
        p.add_argument("velocities", help="velocity file (aimd run --velocities)")
        p.add_argument("--atoms", type=int, nargs="+", default=None,
                       help="0-based atom indices (default: all)")
        p.add_argument("--no-mass-weight", action="store_true",
                       help="plain instead of mass-weighted velocities")
        p.add_argument("--units", default=None, choices=tuple(VELOCITY_UNITS),
                       help="velocity unit of frames whose comment line states none "
                            "(files from other programs); a units= tag wins")
        if name == "vdos":
            _spectral(p)
        else:
            p.add_argument("--max-lag", type=int, default=None,
                           help="longest lag in frames (default n - 1)")
            p.add_argument("--skip", type=int, default=0, help="discard the first N frames")
            p.add_argument("--dt", type=float, default=None,
                           help="frame spacing in fs (default: read from the file)")
        _common(p, f"{name}.csv")
        p.set_defaults(func=cmd_vacf if name == "vacf" else cmd_vdos)

    p = asub.add_parser("ir", help="infrared spectrum from the dipole log", formatter_class=fmt)
    p.add_argument("dipoles", help="dipole CSV (aimd run --dipoles)")
    p.add_argument("--temperature", type=float, default=None,
                   help="K; gives absolute intensities (km/mol per cm^-1)")
    p.add_argument("--correction", default="harmonic",
                   choices=QUANTUM_CORRECTIONS + ("none",),
                   help="quantum correction of the classical line shape")
    _spectral(p)
    _common(p, "ir.csv")
    p.set_defaults(func=cmd_ir)

    p = asub.add_parser("stats", help="means and block-averaged errors of an energy log",
                        formatter_class=fmt)
    p.add_argument("energies", help="energy CSV (aimd run --energies)")
    p.add_argument("--columns", nargs="+", default=list(DEFAULT_COLUMNS))
    p.add_argument("--skip", type=int, default=0, help="discard the first N rows")
    p.add_argument("--block-size", type=int, default=None,
                   help="fixed block length (default: automatic blocking)")
    _common(p, None)
    p.set_defaults(func=cmd_stats)
    return a


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write_csv(path: str | None, header: Sequence[str], columns: Sequence[np.ndarray]) -> None:
    if not path:
        return
    with Path(path).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for row in zip(*columns):
            w.writerow([repr(float(v)) if isinstance(v, (float, np.floating)) else v
                        for v in row])
    print(f"wrote {path}")


def _plot(path: str | None, x: np.ndarray, ys: dict[str, np.ndarray],
          xlabel: str, ylabel: str | dict[str, str]) -> None:
    """
    Line plot of the series ``ys`` against ``x`` (skipped without matplotlib).
    ``ylabel`` is one axis label, or a mapping series name -> label: series
    with different labels (units) then go into separate stacked panels.
    """
    if not path:
        return
    if not ys:
        print(f"nothing to plot; {path} not written", file=sys.stderr)
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; --plot skipped (pip install matplotlib)",
              file=sys.stderr)
        return
    labels = ylabel if isinstance(ylabel, dict) else {name: ylabel for name in ys}
    panels: dict[str, list[str]] = {}
    for name in ys:
        panels.setdefault(labels[name], []).append(name)
    fig, axes = plt.subplots(len(panels), 1, sharex=True, squeeze=False,
                             figsize=(6.4, 2.2 + 1.8 * len(panels)))
    for ax, (label, names) in zip(axes[:, 0], panels.items()):
        for name in names:
            ax.plot(x, ys[name], label=name, linewidth=1.2)
        ax.set_ylabel(label)
        if len(names) > 1:
            ax.legend(frameon=False)
    axes[-1, 0].set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def _skip(n_frames: int, skip: int, need: int = 2) -> slice:
    if skip < 0:
        raise AnalysisInputError("--skip must be >= 0")
    if n_frames - skip < need:
        raise AnalysisInputError(
            f"{n_frames} frames, {skip} skipped: need at least {need} to analyse")
    return slice(skip, None)


def _frame_dt(dt: float | None, reader: Any) -> float:
    if dt is not None:
        if not dt > 0.0:
            raise AnalysisInputError("--dt must be positive")
        return float(dt)
    return float(reader.frame_interval_fs)


def strongest_peaks(spec: analysis.Spectrum, hi: float, n: int = 5,
                    lo: float = 20.0, rel: float = 0.02) -> list[tuple[float, float]]:
    """
    Up to ``n`` local maxima of ``spec`` in (lo, hi], strongest first, that
    reach ``rel`` times the strongest local maximum there (a monotonic tail,
    e.g. of rotational motion near 0 cm^-1, is not a band), as (wavenumber,
    intensity) with the position parabolically interpolated (Spectrum.peak).
    """
    w, y = spec.wavenumber, spec.intensity
    sel = np.flatnonzero((w > lo) & (w <= hi))
    peaks = [k for k in sel if 0 < k < w.size - 1 and y[k] >= y[k - 1] and y[k] > y[k + 1]]
    if not peaks:
        return []
    top = max(y[k] for k in peaks)
    peaks = sorted((k for k in peaks if y[k] >= rel * top), key=lambda k: -y[k])
    return [(spec.peak(w[k - 1], w[k + 1]), float(y[k])) for k in peaks[:n]]


def _print_peaks(spec: analysis.Spectrum, hi: float) -> None:
    peaks = strongest_peaks(spec, hi)
    print(f"resolution {spec.resolution:.1f} cm^-1; strongest bands below {hi:g} cm^-1:")
    for nu, inten in peaks:
        print(f"  {nu:8.1f} cm^-1   {inten:.4g} {spec.unit}")
    if not peaks:
        print("  (none)")


# ── Subcommands ───────────────────────────────────────────────────────────────

def _selection(spec: str) -> str | list[int]:
    if all(part.strip().lstrip("-").isdigit() for part in spec.split(",")):
        return [int(part) for part in spec.split(",")]
    return spec


def cmd_rdf(args: argparse.Namespace) -> int:
    traj = read_xyz(args.trajectory)
    sl = _skip(traj.n_frames, args.skip, need=1)
    if not args.r_max > 0.0:
        raise AnalysisInputError("--r-max must be positive")
    # checked here, in the user's angstrom (aimd.analysis repeats the checks in bohr)
    if args.box is not None:
        if not args.box > 0.0:
            raise AnalysisInputError("--box must be positive")
        if args.r_max > 0.5 * args.box * (1.0 + 1e-12):
            raise AnalysisInputError(
                f"--r-max {args.r_max:g} A exceeds half the box ({0.5 * args.box:g} A); "
                "minimum-image distances are incomplete beyond it")
    box = None if args.box is None else args.box * ANG_TO_BOHR
    res = analysis.radial_distribution(
        traj.positions[sl], traj.symbols,
        (_selection(args.pair[0]), _selection(args.pair[1])),
        r_max=args.r_max * ANG_TO_BOHR, n_bins=args.bins, box=box)
    r = res.r * BOHR_TO_ANG
    k = int(np.argmax(res.g))
    print(f"g_{args.pair[0]}{args.pair[1]}(r) over {res.n_frames} frames, "
          f"{res.n_pairs} pairs per frame ({'periodic box' if res.periodic else 'cluster'})")
    print(f"highest peak at r = {r[k]:.3f} A (g = {res.g[k]:.3f}); "
          f"coordination number within {args.r_max:g} A: {res.coordination[-1]:.3f}")
    _write_csv(args.output, ["r_angstrom", "g_r", "coordination"], [r, res.g, res.coordination])
    _plot(args.plot, r, {"g(r)": res.g}, "r / Å", "g(r)")
    return 0


def cmd_geometry(args: argparse.Namespace) -> int:
    if not (args.bond or args.angle or args.dihedral):
        raise AnalysisInputError("give at least one --bond, --angle or --dihedral")
    traj = read_xyz(args.trajectory)
    x = traj.positions
    names, series, units = [], [], []
    try:
        for i, j in args.bond:
            names.append(f"bond_{i}_{j}_angstrom")
            series.append(analysis.bond_lengths(x, [(i, j)])[:, 0] * BOHR_TO_ANG)
            units.append("A")
        for i, j, k in args.angle:
            names.append(f"angle_{i}_{j}_{k}_deg")
            series.append(analysis.bond_angles(x, [(i, j, k)])[:, 0])
            units.append("deg")
        for i, j, k, m in args.dihedral:
            names.append(f"dihedral_{i}_{j}_{k}_{m}_deg")
            series.append(analysis.dihedral_angles(x, [(i, j, k, m)])[:, 0])
            units.append("deg")
    except IndexError as e:
        raise AnalysisInputError(f"atom index out of range ({traj.n_atoms} atoms): {e}") from e
    sl = _skip(traj.n_frames, args.skip, need=1)
    print(f"{traj.n_frames} frames; statistics over frames {args.skip}..{traj.n_frames - 1}")
    for name, y, unit in zip(names, series, units):
        ys = y[sl]
        periodic = name.startswith("dihedral_")
        if ys.size >= 2:
            if periodic:
                # (-180, 180] series near +-180 jump by 360: block-average them
                # unwrapped about the circular mean, report the mean wrapped back
                center = analysis.circular_mean(ys)
                if np.isfinite(center):
                    ys = analysis.unwrap_about(ys, center)
            ba = analysis.block_average(ys)
            mean = ba.mean
            if periodic and np.isfinite(mean):
                mean = float(analysis.unwrap_about(mean, 0.0))
            print(f"  {name:28s} mean {mean:10.4f} {unit}  std {ba.std:8.4f}  "
                  f"sem {ba.sem:.2g}" + ("  (circular)" if periodic else ""))
        else:
            print(f"  {name:28s} {ys[0]:10.4f} {unit}")
    step = traj.step if traj.step is not None else np.arange(traj.n_frames)
    time_fs = traj.time_fs if traj.time_fs is not None else np.full(traj.n_frames, np.nan)
    _write_csv(args.output, ["step", "time_fs", *names], [step, time_fs, *series])
    t = time_fs if np.all(np.isfinite(time_fs)) else step
    _plot(args.plot, t, dict(zip(names, series)),
          "time / fs" if t is time_fs else "frame",
          {n: ("length / Å" if u == "A" else "angle / deg") for n, u in zip(names, units)})
    return 0


def _velocities(args: argparse.Namespace) -> tuple[np.ndarray, float, np.ndarray | None]:
    try:
        vel = read_velocities(args.velocities, units=args.units)
    except ValueError as e:
        if "states no velocity unit" in str(e):
            raise AnalysisInputError(
                f"{args.velocities}: its frames state no velocity unit (no units= tag); "
                f"for velocities written by another program give the unit with --units "
                f"({' or '.join(VELOCITY_UNITS)}); a positions trajectory cannot be "
                "analysed as velocities") from e
        raise
    sl = _skip(vel.n_frames, args.skip, need=3)
    dt = _frame_dt(args.dt, vel)
    masses = None if args.no_mass_weight else vel.masses
    if args.atoms is not None and (min(args.atoms) < 0 or max(args.atoms) >= vel.n_atoms):
        raise AnalysisInputError(f"--atoms must be in 0..{vel.n_atoms - 1}")
    return vel.velocities[sl], dt, masses


def cmd_vacf(args: argparse.Namespace) -> int:
    v, dt, masses = _velocities(args)
    res = analysis.velocity_autocorrelation(v, dt, masses=masses, atoms=args.atoms,
                                            max_lag=args.max_lag)
    print(f"VACF over {v.shape[0]} frames ({dt:g} fs apart), lags up to "
          f"{res.time_fs[-1]:g} fs; C(0) = {res.acf[0]:.6g}"
          + (" Eh (= 2 <E_kin>)" if res.mass_weighted else " bohr^2/au_time^2"))
    _write_csv(args.output, ["time_fs", "vacf", "vacf_normalized"],
               [res.time_fs, res.acf, res.normalized])
    _plot(args.plot, res.time_fs, {"C(t)/C(0)": res.normalized}, "time / fs", "C(t) / C(0)")
    return 0


def cmd_vdos(args: argparse.Namespace) -> int:
    v, dt, masses = _velocities(args)
    spec = analysis.vibrational_dos(v, dt, masses=masses, atoms=args.atoms,
                                    max_lag=args.max_lag, window=args.window,
                                    zero_pad=args.zero_pad)
    print(f"VDOS from {v.shape[0]} frames ({dt:g} fs apart), max lag {spec.max_lag}")
    _print_peaks(spec, args.max_wavenumber)
    _write_csv(args.output, ["wavenumber_cm-1", "intensity"], [spec.wavenumber, spec.intensity])
    keep = spec.wavenumber <= args.max_wavenumber
    _plot(args.plot, spec.wavenumber[keep], {"VDOS": spec.intensity[keep]},
          "wavenumber / cm$^{-1}$", f"intensity / ({spec.unit})")
    return 0


def cmd_ir(args: argparse.Namespace) -> int:
    log = read_dipole_log(args.dipoles)
    sl = _skip(log.dipole.shape[0], args.skip, need=3)
    dipole = log.dipole[sl]
    if not np.all(np.isfinite(dipole)):
        raise AnalysisInputError(
            f"{args.dipoles} holds nan dipoles: the backend reported none (e.g. MP2)")
    dt = _frame_dt(args.dt, log)
    correction = None if args.correction == "none" else args.correction
    spec = analysis.ir_spectrum(dipole, dt, temperature_k=args.temperature,
                                quantum_correction=correction, max_lag=args.max_lag,
                                window=args.window, zero_pad=args.zero_pad)
    print(f"IR spectrum from {dipole.shape[0]} dipoles ({dt:g} fs apart), "
          f"max lag {spec.max_lag}, correction {args.correction}")
    _print_peaks(spec, args.max_wavenumber)
    _write_csv(args.output, ["wavenumber_cm-1", "intensity"], [spec.wavenumber, spec.intensity])
    keep = spec.wavenumber <= args.max_wavenumber
    _plot(args.plot, spec.wavenumber[keep], {"IR": spec.intensity[keep]},
          "wavenumber / cm$^{-1}$", f"intensity / ({spec.unit})")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    log = read_energy_log(args.energies)
    n = len(next(iter(log.values())))
    unknown = [c for c in args.columns if c not in log]
    if unknown:
        raise AnalysisInputError(f"{args.energies} has no column(s) {unknown}; "
                                 f"columns: {', '.join(log)}")
    _skip(n, args.skip, need=2)
    stats = analysis.column_statistics(log, args.columns, skip=args.skip,
                                       block_size=args.block_size)
    print(f"{n - args.skip} of {n} rows ({args.skip} skipped)")
    print(f"  {'column':16s} {'mean':>18s} {'std':>11s} {'sem':>11s} {'block':>6s}")
    rows = []
    for name, ba in stats.items():
        flag = "" if ba.converged else "  (blocking not converged: sem is a lower bound)"
        print(f"  {name:16s} {ba.mean:18.10g} {ba.std:11.4g} {ba.sem:11.4g} "
              f"{ba.block_size:6d}{flag}")
        rows.append((name, ba.mean, ba.std, ba.sem, ba.sem_naive, ba.block_size,
                     ba.n_samples, int(ba.converged)))
    if args.output:
        _write_csv(args.output,
                   ["column", "mean", "std", "sem", "sem_naive", "block_size",
                    "n_samples", "converged"],
                   list(zip(*rows)))
    if args.plot:
        t = log.get("time_fs", np.arange(n, dtype=float))
        ys, labels = stats_plot_series(log, args.columns)
        _plot(args.plot, t, ys, "time / fs", labels)
    return 0


def stats_plot_series(
    log: dict[str, np.ndarray], columns: Sequence[str]
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """
    Series and panel labels for ``analyze stats --plot``: energies (``*_Eh``)
    relative to their first value in one panel, the temperature in K in
    another, any other column as is in a panel of its own.
    """
    ys: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    for c in columns:
        if c.endswith("_Eh"):
            ys[c], labels[c] = log[c] - log[c][0], "E - E(0) / Eh"
        elif c == "temperature_K":
            ys[c], labels[c] = log[c], "T / K"
        else:
            ys[c], labels[c] = log[c], c
    return ys, labels
