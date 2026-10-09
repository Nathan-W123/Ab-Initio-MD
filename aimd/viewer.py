"""
``aimd view``: an interactive trajectory viewer as one self-contained HTML file.

    aimd view trajectory.xyz                    # -> trajectory.html
    aimd view run.xyz --energies run_e.csv -o movie.html --every 2

The page needs no installation and no network: the frames, the energy log and
a small canvas renderer are all embedded, so it opens offline in any browser
(and in sandboxed previews that block external scripts).

What it shows
  - a ball-and-stick molecule, rotated by dragging and zoomed with the wheel;
  - bonds recomputed for every frame from covalent radii (Cordero et al.,
    Dalton Trans. 2008, 2832): atoms i, j are bonded when
    r_ij < r_i + r_j + tolerance (default 0.4 angstrom, as in Open Babel), so
    bonds that break or form during the run appear and disappear;
  - the molecular fragments of each frame (connected components of the bond
    graph) as Hill formulas, e.g. "H2O" or "HO + H" after a dissociation;
  - play / pause, a frame slider, playback speed, 0-based atom index labels
    (the indices ``aimd analyze geometry`` takes);
  - if an energy log is found, the change of E_pot, E_tot and the conserved
    energy from their first shown values (kcal/mol; the conserved energy is
    then a flat line at 0 in a healthy run) and the temperature, with a cursor
    synchronised to the molecule; clicking a chart seeks.

Frames are matched to energy-log rows by step number (``step=`` in each XYZ
comment, as written by ``aimd run``); a hand-written XYZ without step numbers
is matched row by row when the counts agree. By default each frame is
translated so its centre of mass is at the origin, which keeps a drifting
molecule in view; rotation is left as it is (it is real motion).
"""

from __future__ import annotations

import argparse
import html
import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from aimd.trajectory import read_energy_log, read_xyz
from aimd.units import BOHR_TO_ANG, HARTREE_TO_KCALMOL

# Covalent radii in angstrom, Cordero et al. 2008 (C: sp3 value).
COVALENT_RADII_ANG: dict[str, float] = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71,
    "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11,
    "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76, "Sc": 1.70,
    "Ti": 1.60, "V": 1.53, "Cr": 1.39, "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24,
    "Cu": 1.32, "Zn": 1.22, "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20,
    "Kr": 1.16,
}

# Jmol / CPK element colours.
ELEMENT_COLORS: dict[str, str] = {
    "H": "#FFFFFF", "He": "#D9FFFF", "Li": "#CC80FF", "Be": "#C2FF00", "B": "#FFB5B5",
    "C": "#909090", "N": "#3050F8", "O": "#FF0D0D", "F": "#90E050", "Ne": "#B3E3F5",
    "Na": "#AB5CF2", "Mg": "#8AFF00", "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000",
    "S": "#FFFF30", "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
    "Sc": "#E6E6E6", "Ti": "#BFC2C7", "V": "#A6A6AB", "Cr": "#8A99C7", "Mn": "#9C7AC7",
    "Fe": "#E06633", "Co": "#F090A0", "Ni": "#50D050", "Cu": "#C88033", "Zn": "#7D80B0",
    "Ga": "#C28F8F", "Ge": "#668F8F", "As": "#BD80E3", "Se": "#FFA100", "Br": "#A62929",
    "Kr": "#5CB8D1",
}

DEFAULT_MAX_FRAMES = 2000
DEFAULT_BOND_TOLERANCE = 0.4      # angstrom


def _finite_or_none(values: np.ndarray, digits: int) -> list[float | None]:
    return [round(float(v), digits) if math.isfinite(v) else None for v in values]


def _align_energies(
    log: dict[str, np.ndarray], steps: np.ndarray | None, n_frames: int, source: str,
) -> dict[str, np.ndarray] | None:
    """Energy-log columns re-ordered to the frames (NaN where a frame has no row)."""
    if steps is None:
        if len(log.get("step", [])) == n_frames:
            return {k: np.asarray(v, dtype=float) for k, v in log.items()}
        warnings.warn(f"{source}: the trajectory has no step numbers and the row count "
                      f"({len(log.get('step', []))}) differs from the frame count "
                      f"({n_frames}); energies not shown", RuntimeWarning, stacklevel=3)
        return None
    if "step" not in log:
        warnings.warn(f"{source}: no 'step' column; energies not shown",
                      RuntimeWarning, stacklevel=3)
        return None
    row_of = {int(s): k for k, s in enumerate(log["step"])}
    idx = np.array([row_of.get(int(s), -1) for s in steps])
    if np.all(idx < 0):
        warnings.warn(f"{source}: no row matches a trajectory step; energies not shown",
                      RuntimeWarning, stacklevel=3)
        return None
    if np.any(idx < 0):
        warnings.warn(f"{source}: {int(np.sum(idx < 0))} of {n_frames} frames have no "
                      "energy row (gaps in the charts)", RuntimeWarning, stacklevel=3)
    out = {}
    for name, col in log.items():
        col = np.asarray(col, dtype=float)
        out[name] = np.where(idx >= 0, col[np.maximum(idx, 0)], np.nan)
    return out


def find_energy_log(trajectory: str | Path) -> Path | None:
    """``<stem>_energies.csv`` or ``energies.csv`` next to the trajectory, if present."""
    traj = Path(trajectory)
    for cand in (traj.with_name(f"{traj.stem}_energies.csv"), traj.with_name("energies.csv")):
        if cand.is_file():
            return cand
    return None


def build_viewer_data(
    trajectory: str | Path,
    energies: str | Path | None = None,
    every: int = 1,
    max_frames: int | None = DEFAULT_MAX_FRAMES,
    center: bool = True,
    bond_tolerance: float = DEFAULT_BOND_TOLERANCE,
    title: str | None = None,
) -> dict[str, Any]:
    """
    The JSON payload of the viewer page: frames in angstrom (rounded to 1e-4),
    per-atom colours and radii, step/time labels and, if ``energies`` is
    given, the matching energy-log columns.
    """
    if every < 1:
        raise ValueError("every must be >= 1")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if not (bond_tolerance >= 0.0 and math.isfinite(bond_tolerance)):
        raise ValueError("bond_tolerance must be a finite number >= 0")
    traj = read_xyz(trajectory)
    unknown = sorted({s for s in traj.symbols if s not in COVALENT_RADII_ANG})
    if unknown:
        raise ValueError(f"no covalent radius for element(s) {unknown}")

    n_all = traj.n_frames
    stride = every
    if max_frames is not None and math.ceil(n_all / stride) > max_frames:
        stride = math.ceil(n_all / max_frames)
    keep = np.arange(0, n_all, stride)

    pos = traj.positions[keep] * BOHR_TO_ANG                       # (F, N, 3)
    if center:
        m = traj.masses
        pos = pos - (np.einsum("i,fij->fj", m, pos) / m.sum())[:, None, :]
    else:
        pos = pos - pos.reshape(-1, 3).mean(axis=0)
    steps = traj.step[keep] if traj.step is not None else None
    times = traj.time_fs[keep] if traj.time_fs is not None else None

    data: dict[str, Any] = {
        "title": title or Path(trajectory).name,
        "source": str(trajectory),
        "symbols": traj.symbols,
        "colors": [ELEMENT_COLORS.get(s, "#FF1493") for s in traj.symbols],
        "radii": [COVALENT_RADII_ANG[s] for s in traj.symbols],
        "bondTolerance": float(bond_tolerance),
        "frames": np.round(pos.reshape(len(keep), -1), 4).tolist(),
        "steps": None if steps is None else [int(s) for s in steps],
        "times": None if times is None else _finite_or_none(times, 6),
        "stride": int(stride),
        "totalFrames": int(n_all),
        "energies": None,
    }

    if energies is not None:
        log = read_energy_log(energies)
        aligned = _align_energies(log, steps, n_all if steps is None else len(keep),
                                  str(energies))
        if aligned is not None and steps is None:
            aligned = {k: v[keep] for k, v in aligned.items()}
        if aligned is not None:
            # Each energy is plotted as its change from its own first value, so
            # a conserved quantity is a flat line at 0 and its drift is visible.
            series: dict[str, Any] = {"file": str(energies), "reference_Eh": {}}
            for key in ("potential_Eh", "total_Eh", "conserved_Eh"):
                if key not in aligned:
                    continue
                col = aligned[key]
                series[key] = _finite_or_none(col, 10)
                finite = col[np.isfinite(col)]
                if finite.size:
                    ref = float(finite[0])
                    series["reference_Eh"][key] = ref
                    series[key.replace("_Eh", "_kcal")] = _finite_or_none(
                        (col - ref) * HARTREE_TO_KCALMOL, 6)
            if "temperature_K" in aligned:
                series["temperature_K"] = _finite_or_none(aligned["temperature_K"], 4)
            if data["times"] is None and "time_fs" in aligned:
                data["times"] = _finite_or_none(aligned["time_fs"], 6)
            data["energies"] = series
    return data


def render_html(data: dict[str, Any]) -> str:
    """The complete page for a :func:`build_viewer_data` payload."""
    # "</" inside a <script> element would end it early: escape it in the JSON.
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    return (_TEMPLATE
            .replace("__TITLE__", html.escape(str(data["title"])))
            .replace("__DATA__", payload))


def write_viewer(
    trajectory: str | Path,
    out: str | Path | None = None,
    energies: str | Path | None = None,
    **kwargs: Any,
) -> Path:
    """Write the viewer page for ``trajectory`` (default ``<trajectory>.html``)."""
    out = Path(out) if out is not None else Path(trajectory).with_suffix(".html")
    data = build_viewer_data(trajectory, energies=energies, **kwargs)
    out.write_text(render_html(data), encoding="utf-8")
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def add_view_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    v = sub.add_parser(
        "view", help="write an interactive HTML movie of a trajectory",
        description="Write a self-contained HTML page (no network needed) that plays an "
                    "XYZ trajectory with per-frame bonds, fragments and synced energy "
                    "and temperature charts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    v.add_argument("trajectory", help="positions XYZ file written by 'aimd run'")
    v.add_argument("-o", "--output", default=None,
                   help="HTML file to write (default: the trajectory name with .html)")
    v.add_argument("--energies", default=None,
                   help="energy CSV to plot (default: <stem>_energies.csv or energies.csv "
                        "next to the trajectory, if present; 'none' to leave out)")
    v.add_argument("--every", type=int, default=1, help="keep every N-th frame")
    v.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                   help="raise the stride further if more frames than this remain "
                        "(0: no limit)")
    v.add_argument("--no-center", dest="center", action="store_false",
                   help="keep lab-frame positions (default: centre of mass at the origin)")
    v.add_argument("--bond-tolerance", type=float, default=DEFAULT_BOND_TOLERANCE,
                   help="bonded if r < r_cov,i + r_cov,j + this (angstrom)")
    v.add_argument("--title", default=None, help="page title (default: file name)")
    v.set_defaults(func=cmd_view)
    return v


def cmd_view(args: argparse.Namespace) -> int:
    energies = args.energies
    if energies is None:
        energies = find_energy_log(args.trajectory)
    elif str(energies).lower() == "none":
        energies = None
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = write_viewer(
            args.trajectory, args.output, energies=energies, every=args.every,
            max_frames=args.max_frames or None, center=args.center,
            bond_tolerance=args.bond_tolerance, title=args.title,
        )
    for w in caught:
        print(f"warning: {w.message}")
    data = json.loads(out.read_text(encoding="utf-8").split(
        '<script type="application/json" id="aimd-data">', 1)[1].split("</script>", 1)[0])
    n = len(data["frames"])
    note = (f" (stride {data['stride']} over {data['totalFrames']} frames)"
            if data["stride"] > 1 else "")
    shown = data["energies"]["file"] if data["energies"] else "none"
    print(f"wrote {out}: {n} frames{note}, {len(data['symbols'])} atoms; energies: {shown}")
    return 0


# ── Page template ─────────────────────────────────────────────────────────────

_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root {
  --bg: #f7f7f5; --panel: #ffffff; --text: #1d1d1f; --muted: #6b6b70;
  --border: #dcdcd8; --accent: #2f6fde; --grid: #e7e7e3;
  --c1: #2f6fde; --c2: #e07a1f; --c3: #2a9d5c; --c4: #c2410c;
  --stage: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #141416; --panel: #1d1d20; --text: #ececee; --muted: #9a9aa2;
    --border: #34343a; --accent: #6ea0ff; --grid: #2a2a30;
    --c1: #6ea0ff; --c2: #f0a050; --c3: #4cc38a; --c4: #f08060;
    --stage: #111114;
  }
}
:root[data-theme="dark"] {
  --bg: #141416; --panel: #1d1d20; --text: #ececee; --muted: #9a9aa2;
  --border: #34343a; --accent: #6ea0ff; --grid: #2a2a30;
  --c1: #6ea0ff; --c2: #f0a050; --c3: #4cc38a; --c4: #f08060;
  --stage: #111114;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.4 system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
main { max-width: 980px; margin: 0 auto; padding: 16px; }
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 4px 12px; margin-bottom: 10px; }
h1 { font-size: 17px; margin: 0; font-weight: 600; overflow-wrap: anywhere; }
.sub { color: var(--muted); font-size: 13px; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; }
#stage { position: relative; height: min(62vh, 520px); min-height: 260px;
  background: var(--stage); border-radius: 10px; overflow: hidden; touch-action: none; cursor: grab; }
#stage.dragging { cursor: grabbing; }
#mol { width: 100%; height: 100%; display: block; }
#hud { position: absolute; left: 10px; top: 8px; right: 10px; pointer-events: none;
  font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--muted); }
#hud b { color: var(--text); font-weight: 600; }
#frag { position: absolute; left: 10px; bottom: 8px; right: 10px; pointer-events: none;
  font: 13px/1.4 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--text); }
#frag.changed { color: var(--c4); }
.controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; padding: 10px 12px; margin-top: 10px; }
button, select { font: inherit; color: var(--text); background: var(--bg); border: 1px solid var(--border);
  border-radius: 7px; padding: 5px 10px; cursor: pointer; }
button:hover, select:hover { border-color: var(--accent); }
#play { min-width: 74px; font-weight: 600; }
#slider { flex: 1 1 220px; accent-color: var(--accent); min-width: 140px; }
label.chk { display: inline-flex; gap: 5px; align-items: center; color: var(--muted); cursor: pointer; }
#framelabel { font: 12px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--muted); min-width: 92px; text-align: right; }
.charts { display: grid; grid-template-columns: 1fr; gap: 10px; margin-top: 10px; }
@media (min-width: 760px) { .charts { grid-template-columns: 1fr 1fr; } }
.chart { padding: 8px 10px 6px; }
.chart h2 { font-size: 13px; font-weight: 600; margin: 0 0 2px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; font-size: 12px; color: var(--muted); margin-bottom: 4px; }
.legend i { display: inline-block; width: 12px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 5px; }
.chart canvas { width: 100%; height: 170px; display: block; cursor: crosshair; }
.hint { color: var(--muted); font-size: 12px; margin-top: 8px; }
.empty { color: var(--muted); padding: 12px; }
</style>
</head>
<body>
<main>
  <header>
    <h1 id="title"></h1>
    <span class="sub" id="subtitle"></span>
  </header>
  <div id="stage" class="card">
    <canvas id="mol" aria-label="molecule"></canvas>
    <div id="hud"></div>
    <div id="frag"></div>
  </div>
  <div class="controls card">
    <button id="play" title="Play / pause (space)">Play</button>
    <button id="prev" title="Previous frame (left arrow)" aria-label="previous frame">&#9664;</button>
    <button id="next" title="Next frame (right arrow)" aria-label="next frame">&#9654;</button>
    <input id="slider" type="range" min="0" value="0" step="1" aria-label="frame">
    <span id="framelabel"></span>
    <label class="chk">speed
      <select id="speed">
        <option value="5">5 fps</option><option value="10">10 fps</option>
        <option value="20" selected>20 fps</option><option value="30">30 fps</option>
        <option value="60">60 fps</option>
      </select>
    </label>
    <label class="chk"><input type="checkbox" id="loop" checked> loop</label>
    <label class="chk"><input type="checkbox" id="labels"> atom indices</label>
    <button id="reset" title="Reset rotation and zoom">Reset view</button>
  </div>
  <div class="charts" id="charts">
    <div class="chart card" id="echart">
      <h2>Energy change from the first frame, kcal/mol</h2>
      <div class="legend" id="elegend"></div>
      <canvas id="ecanvas" aria-label="energy chart"></canvas>
    </div>
    <div class="chart card" id="tchart">
      <h2>Temperature, K</h2>
      <div class="legend" id="tlegend"></div>
      <canvas id="tcanvas" aria-label="temperature chart"></canvas>
    </div>
  </div>
  <div class="hint">Drag to rotate, wheel or pinch to zoom, click a chart to jump to that time.
    Bonds are redrawn every frame from covalent radii, so breaking and forming bonds show up;
    the line at the bottom of the view lists the fragments (in red once they differ from frame 1).</div>
</main>
<script type="application/json" id="aimd-data">__DATA__</script>
<script>
(function () {
  "use strict";
  var D = JSON.parse(document.getElementById("aimd-data").textContent);
  var N = D.symbols.length, F = D.frames.length;
  var $ = function (id) { return document.getElementById(id); };
  var css = function (name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); };

  // ── Header ────────────────────────────────────────────────────────────────
  $("title").textContent = D.title;
  var counts = {};
  D.symbols.forEach(function (s) { counts[s] = (counts[s] || 0) + 1; });
  var sub = N + " atoms (" + formula(counts) + "), " + F + " frames";
  if (D.stride > 1) sub += " (stride " + D.stride + " over " + D.totalFrames + ")";
  $("subtitle").textContent = sub;

  // ── Helpers ───────────────────────────────────────────────────────────────
  function formula(c) {             // Hill order
    var els = Object.keys(c), out = "";
    var order = c.C ? ["C", "H"].filter(function (e) { return c[e]; })
                      .concat(els.filter(function (e) { return e !== "C" && e !== "H"; }).sort())
                    : els.sort();
    order.forEach(function (e) { out += e + (c[e] > 1 ? c[e] : ""); });
    return out;
  }
  function shade(hex, f) {          // f > 0 lighten towards white, f < 0 darken
    var n = parseInt(hex.slice(1), 16), r = n >> 16, g = (n >> 8) & 255, b = n & 255;
    function m(v) { return Math.round(f >= 0 ? v + (255 - v) * f : v * (1 + f)); }
    return "rgb(" + m(r) + "," + m(g) + "," + m(b) + ")";
  }
  function fmt(x, d) { return x === null || x === undefined ? "–" : x.toFixed(d); }

  // ── Bonds and fragments per frame ─────────────────────────────────────────
  function bondsOf(f) {
    var p = D.frames[f], out = [], tol = D.bondTolerance, R = D.radii;
    for (var i = 0; i < N; i++) for (var j = i + 1; j < N; j++) {
      var dx = p[3*i] - p[3*j], dy = p[3*i+1] - p[3*j+1], dz = p[3*i+2] - p[3*j+2];
      var r = Math.sqrt(dx*dx + dy*dy + dz*dz), cut = R[i] + R[j] + tol;
      if (r > 0.1 && r < cut) out.push([i, j]);
    }
    return out;
  }
  function fragmentsOf(bonds) {
    var parent = []; for (var i = 0; i < N; i++) parent.push(i);
    function find(a) { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; }
    bonds.forEach(function (b) { parent[find(b[0])] = find(b[1]); });
    var groups = {};
    for (var k = 0; k < N; k++) {
      var g = groups[find(k)] = groups[find(k)] || {};
      g[D.symbols[k]] = (g[D.symbols[k]] || 0) + 1;
    }
    var names = Object.keys(groups).map(function (g) { return formula(groups[g]); });
    names.sort(function (a, b) { return b.length - a.length || (a < b ? -1 : 1); });
    var tally = {}; names.forEach(function (n) { tally[n] = (tally[n] || 0) + 1; });
    return Object.keys(tally).map(function (n) { return (tally[n] > 1 ? tally[n] + " " : "") + n; }).join(" + ");
  }
  var bondCache = {}, fragCache = {};
  function bonds(f) { return bondCache[f] || (bondCache[f] = bondsOf(f)); }
  function frags(f) { return fragCache[f] || (fragCache[f] = fragmentsOf(bonds(f))); }

  // ── View state ────────────────────────────────────────────────────────────
  var rot = [1,0,0, 0,1,0, 0,0,1], zoom = 1, frame = 0, playing = false;
  var extent = 0.5;
  D.frames.forEach(function (p) {
    for (var i = 0; i < N; i++) {
      var r = Math.sqrt(p[3*i]*p[3*i] + p[3*i+1]*p[3*i+1] + p[3*i+2]*p[3*i+2]) + D.radii[i];
      if (r > extent) extent = r;
    }
  });
  function rotate(ax, ang) {        // pre-multiply by a rotation about a screen axis
    var c = Math.cos(ang), s = Math.sin(ang), m;
    if (ax === 0) m = [1,0,0, 0,c,-s, 0,s,c]; else m = [c,0,s, 0,1,0, -s,0,c];
    var o = [];
    for (var r = 0; r < 3; r++) for (var k = 0; k < 3; k++)
      o.push(m[3*r]*rot[k] + m[3*r+1]*rot[3+k] + m[3*r+2]*rot[6+k]);
    rot = o;
  }

  // ── Molecule renderer (canvas 2D, painter's algorithm) ────────────────────
  var mol = $("mol"), ctx = mol.getContext("2d"), dpr = 1;
  function fit(canvas, c) {
    dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w*dpr) || canvas.height !== Math.round(h*dpr)) {
      canvas.width = Math.round(w*dpr); canvas.height = Math.round(h*dpr);
    }
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    return [w, h];
  }
  function drawMolecule() {
    var wh = fit(mol, ctx), w = wh[0], h = wh[1];
    ctx.clearRect(0, 0, w, h);
    var scale = zoom * 0.48 * Math.min(w, h) / extent, cx = w / 2, cy = h / 2 + 4;
    var p = D.frames[frame], X = [], Y = [], Z = [];
    for (var i = 0; i < N; i++) {
      var x = p[3*i], y = p[3*i+1], z = p[3*i+2];
      X.push(cx + scale * (rot[0]*x + rot[1]*y + rot[2]*z));
      Y.push(cy - scale * (rot[3]*x + rot[4]*y + rot[5]*z));
      Z.push(rot[6]*x + rot[7]*y + rot[8]*z);
    }
    var items = [], bs = bonds(frame);
    bs.forEach(function (b) {
      var i = b[0], j = b[1], mx = (X[i]+X[j])/2, my = (Y[i]+Y[j])/2, mz = (Z[i]+Z[j])/2;
      items.push({z: (Z[i]+mz)/2 - 1e-3, k: 1, a: i, x0: X[i], y0: Y[i], x1: mx, y1: my});
      items.push({z: (Z[j]+mz)/2 - 1e-3, k: 1, a: j, x0: X[j], y0: Y[j], x1: mx, y1: my});
    });
    for (var a = 0; a < N; a++) items.push({z: Z[a], k: 0, a: a});
    items.sort(function (u, v) { return u.z - v.z; });
    var bw = Math.max(2, 0.16 * scale), outline = css("--stage");
    var zmin = -extent, zspan = 2 * extent, labels = [];
    items.forEach(function (it) {
      var col = D.colors[it.a], depth = (it.z - zmin) / zspan;   // 0 far .. 1 near
      if (it.k === 1) {
        ctx.lineCap = "round";
        ctx.strokeStyle = shade(col, -0.45); ctx.lineWidth = bw + 2;
        ctx.beginPath(); ctx.moveTo(it.x0, it.y0); ctx.lineTo(it.x1, it.y1); ctx.stroke();
        ctx.strokeStyle = shade(col, -0.15 + 0.2 * depth); ctx.lineWidth = bw;
        ctx.beginPath(); ctx.moveTo(it.x0, it.y0); ctx.lineTo(it.x1, it.y1); ctx.stroke();
      } else {
        var r = Math.max(3, (0.18 + 0.32 * D.radii[it.a]) * scale), x = X[it.a], y = Y[it.a];
        var g = ctx.createRadialGradient(x - 0.35*r, y - 0.35*r, 0.1*r, x, y, r);
        g.addColorStop(0, shade(col, 0.75));
        g.addColorStop(0.45, shade(col, -0.05 + 0.1 * depth));
        g.addColorStop(1, shade(col, -0.55));
        ctx.beginPath(); ctx.arc(x, y, r, 0, 2*Math.PI);
        ctx.fillStyle = g; ctx.fill();
        ctx.lineWidth = 1; ctx.strokeStyle = outline; ctx.globalAlpha = 0.35; ctx.stroke(); ctx.globalAlpha = 1;
        labels.push([it.a, x, y, r]);
      }
    });
    if ($("labels").checked) {          // last, so no atom or bond hides an index
      ctx.textAlign = "center"; ctx.textBaseline = "middle";
      labels.forEach(function (l) {
        ctx.font = "600 " + Math.max(10, Math.min(15, 0.5*l[3])) + "px ui-monospace, monospace";
        ctx.lineWidth = 3; ctx.strokeStyle = outline; ctx.strokeText(String(l[0]), l[1], l[2]);
        ctx.fillStyle = css("--text"); ctx.fillText(String(l[0]), l[1], l[2]);
      });
    }
    // HUD
    var E = D.energies, t = D.times ? D.times[frame] : null, st = D.steps ? D.steps[frame] : null;
    var hud = "frame <b>" + (frame + 1) + "</b>/" + F;
    if (st !== null) hud += " &nbsp; step <b>" + st + "</b>";
    if (t !== null && t !== undefined) hud += " &nbsp; t = <b>" + t.toFixed(2) + "</b> fs";
    if (E && E.potential_Eh && E.potential_Eh[frame] !== null) hud += "<br>E<sub>pot</sub> = <b>" + E.potential_Eh[frame].toFixed(8) + "</b> E<sub>h</sub>";
    if (E && E.temperature_K && E.temperature_K[frame] !== null) hud += " &nbsp; T = <b>" + E.temperature_K[frame].toFixed(1) + "</b> K";
    hud += " &nbsp; bonds <b>" + bs.length + "</b>";
    $("hud").innerHTML = hud;
    var fr = frags(frame);
    $("frag").textContent = fr;
    $("frag").className = fr === frags(0) ? "" : "changed";
  }

  // ── Charts ────────────────────────────────────────────────────────────────
  var E = D.energies, xs = D.times || D.frames.map(function (_, k) { return k; });
  var xlabel = D.times ? "t / fs" : "frame";
  var eSeries = [], tSeries = [];
  if (E) {
    if (E.potential_kcal) eSeries.push({name: "E_pot", v: E.potential_kcal, c: "--c1"});
    if (E.total_kcal) eSeries.push({name: "E_tot", v: E.total_kcal, c: "--c2"});
    if (E.conserved_kcal) eSeries.push({name: "E_conserved", v: E.conserved_kcal, c: "--c3"});
    if (E.temperature_K) tSeries.push({name: "T", v: E.temperature_K, c: "--c4"});
  }
  if (!eSeries.length && !tSeries.length) {
    $("charts").innerHTML = '<div class="card empty">No energy log: run <code>aimd view</code> with <code>--energies energies.csv</code> to plot energies and temperature.</div>';
  }
  function legend(id, series) {
    var el = $(id); if (!el) return;
    el.innerHTML = series.map(function (s) {
      return '<span><i style="background:' + css(s.c) + '"></i>' + s.name + "</span>";
    }).join("");
  }
  function niceTicks(lo, hi, n) {
    var span = hi - lo || 1, step = Math.pow(10, Math.floor(Math.log10(span / n)));
    [1, 2, 5, 10].some(function (m) { if (span / (step * m) <= n) { step *= m; return true; } return false; });
    var t = [], v = Math.ceil(lo / step) * step;
    for (; v <= hi + 1e-9 * span; v += step) t.push(Math.abs(v) < 1e-12 * span ? 0 : v);
    return t;
  }
  function drawChart(canvas, series) {
    if (!canvas || !series.length) return;
    var c = canvas.getContext("2d"), wh = fit(canvas, c), w = wh[0], h = wh[1];
    c.clearRect(0, 0, w, h);
    var L = 52, R = 10, T = 6, B = 24, pw = w - L - R, ph = h - T - B;
    var lo = Infinity, hi = -Infinity;
    series.forEach(function (s) { s.v.forEach(function (y) { if (y !== null) { lo = Math.min(lo, y); hi = Math.max(hi, y); } }); });
    if (!isFinite(lo)) return;
    if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
    var pad = 0.06 * (hi - lo); lo -= pad; hi += pad;
    var x0 = xs[0], x1 = xs[xs.length - 1]; if (x1 === x0) x1 = x0 + 1;
    var sx = function (x) { return L + pw * (x - x0) / (x1 - x0); };
    var sy = function (y) { return T + ph * (1 - (y - lo) / (hi - lo)); };
    c.font = "11px system-ui, sans-serif"; c.fillStyle = css("--muted"); c.strokeStyle = css("--grid"); c.lineWidth = 1;
    c.textAlign = "right"; c.textBaseline = "middle";
    niceTicks(lo, hi, 4).forEach(function (y) {
      c.beginPath(); c.moveTo(L, sy(y)); c.lineTo(L + pw, sy(y)); c.stroke();
      c.fillText(Math.abs(hi - lo) < 0.05 ? y.toExponential(1) : +y.toPrecision(4), L - 6, sy(y));
    });
    c.textAlign = "center"; c.textBaseline = "top";
    niceTicks(x0, x1, Math.max(2, Math.floor(pw / 80))).forEach(function (x) { c.fillText(+x.toPrecision(5), sx(x), T + ph + 5); });
    c.textAlign = "left"; c.fillText(xlabel, 2, T + ph + 5);
    series.forEach(function (s) {
      c.strokeStyle = css(s.c); c.lineWidth = 1.6; c.beginPath();
      var pen = false;
      s.v.forEach(function (y, k) {
        if (y === null) { pen = false; return; }
        if (pen) c.lineTo(sx(xs[k]), sy(y)); else { c.moveTo(sx(xs[k]), sy(y)); pen = true; }
      });
      c.stroke();
    });
    // cursor
    var cx = sx(xs[frame]);
    c.strokeStyle = css("--text"); c.globalAlpha = 0.55; c.lineWidth = 1;
    c.beginPath(); c.moveTo(cx, T); c.lineTo(cx, T + ph); c.stroke(); c.globalAlpha = 1;
    series.forEach(function (s) {
      var y = s.v[frame]; if (y === null) return;
      c.fillStyle = css(s.c); c.beginPath(); c.arc(cx, sy(y), 3.2, 0, 2*Math.PI); c.fill();
    });
    canvas._map = {L: L, pw: pw, x0: x0, x1: x1};
  }
  function seekFromChart(ev) {
    var m = ev.currentTarget._map; if (!m) return;
    var rect = ev.currentTarget.getBoundingClientRect();
    var x = m.x0 + (ev.clientX - rect.left - m.L) / m.pw * (m.x1 - m.x0), best = 0;
    for (var k = 1; k < F; k++) if (Math.abs(xs[k] - x) < Math.abs(xs[best] - x)) best = k;
    setFrame(best);
  }
  ["ecanvas", "tcanvas"].forEach(function (id) {
    var el = $(id); if (!el) return;
    var down = false;
    el.addEventListener("pointerdown", function (e) { down = true; el.setPointerCapture(e.pointerId); seekFromChart(e); });
    el.addEventListener("pointermove", function (e) { if (down) seekFromChart(e); });
    el.addEventListener("pointerup", function () { down = false; });
  });
  if (!eSeries.length && $("echart")) $("echart").style.display = "none";
  if (!tSeries.length && $("tchart")) $("tchart").style.display = "none";

  // ── Frame control ─────────────────────────────────────────────────────────
  var slider = $("slider");
  slider.max = String(F - 1);
  function draw() {
    drawMolecule();
    drawChart($("ecanvas"), eSeries);
    drawChart($("tcanvas"), tSeries);
    slider.value = String(frame);
    $("framelabel").textContent = (frame + 1) + " / " + F;
  }
  function setFrame(f) { frame = Math.max(0, Math.min(F - 1, f)); draw(); }
  function setPlaying(on) {
    playing = on && F > 1; $("play").textContent = playing ? "Pause" : "Play";
    if (playing) { last = performance.now(); acc = 0; requestAnimationFrame(tick); }
  }
  var last = 0, acc = 0;
  function tick(now) {
    if (!playing) return;
    acc += (now - last) / 1000 * parseFloat($("speed").value); last = now;
    if (acc >= 1) {
      var n = Math.floor(acc); acc -= n;
      if (frame + n >= F) {
        if ($("loop").checked) frame = (frame + n) % F; else { frame = F - 1; setPlaying(false); }
      } else frame += n;
      draw();
    }
    requestAnimationFrame(tick);
  }
  $("play").addEventListener("click", function () {
    if (!playing && frame === F - 1 && !$("loop").checked) frame = 0;
    setPlaying(!playing);
  });
  $("prev").addEventListener("click", function () { setPlaying(false); setFrame(frame - 1); });
  $("next").addEventListener("click", function () { setPlaying(false); setFrame(frame + 1); });
  slider.addEventListener("input", function () { setFrame(parseInt(slider.value, 10)); });
  $("labels").addEventListener("change", draw);
  $("reset").addEventListener("click", function () { rot = [1,0,0, 0,1,0, 0,0,1]; zoom = 1; draw(); });
  document.addEventListener("keydown", function (e) {
    if (e.target && (e.target.tagName === "INPUT" || e.target.tagName === "SELECT")) {
      if (e.target.type !== "range" && e.target.type !== "checkbox") return;
    }
    if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
    else if (e.key === "ArrowRight") { e.preventDefault(); setPlaying(false); setFrame(frame + 1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); setPlaying(false); setFrame(frame - 1); }
  });

  // ── Rotate (drag), zoom (wheel / pinch) ───────────────────────────────────
  var stage = $("stage"), pointers = {}, lastX = 0, lastY = 0, pinch = 0;
  stage.addEventListener("pointerdown", function (e) {
    pointers[e.pointerId] = [e.clientX, e.clientY]; stage.setPointerCapture(e.pointerId);
    lastX = e.clientX; lastY = e.clientY; stage.classList.add("dragging");
    var ids = Object.keys(pointers);
    if (ids.length === 2) { var a = pointers[ids[0]], b = pointers[ids[1]]; pinch = Math.hypot(a[0]-b[0], a[1]-b[1]); }
  });
  stage.addEventListener("pointermove", function (e) {
    if (!(e.pointerId in pointers)) return;
    pointers[e.pointerId] = [e.clientX, e.clientY];
    var ids = Object.keys(pointers);
    if (ids.length === 2) {
      var a = pointers[ids[0]], b = pointers[ids[1]], d = Math.hypot(a[0]-b[0], a[1]-b[1]);
      if (pinch > 0) zoom = Math.max(0.2, Math.min(8, zoom * d / pinch));
      pinch = d; drawMolecule(); return;
    }
    var dx = e.clientX - lastX, dy = e.clientY - lastY; lastX = e.clientX; lastY = e.clientY;
    rotate(1, dx * 0.01); rotate(0, dy * 0.01); drawMolecule();
  });
  function up(e) { delete pointers[e.pointerId]; pinch = 0; if (!Object.keys(pointers).length) stage.classList.remove("dragging"); }
  stage.addEventListener("pointerup", up); stage.addEventListener("pointercancel", up);
  stage.addEventListener("wheel", function (e) {
    e.preventDefault(); zoom = Math.max(0.2, Math.min(8, zoom * Math.exp(-e.deltaY * 0.0015))); drawMolecule();
  }, {passive: false});

  legend("elegend", eSeries); legend("tlegend", tSeries);
  window.addEventListener("resize", draw);
  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    var redo = function () { legend("elegend", eSeries); legend("tlegend", tSeries); draw(); };
    if (mq.addEventListener) mq.addEventListener("change", redo);
  }
  // A small initial tilt shows depth; frame 0 is drawn immediately.
  rotate(0, -0.35); rotate(1, 0.5);
  window.aimdViewer = {setFrame: setFrame, frames: F, frame: function () { return frame; },
                       fragments: frags, bonds: bonds, play: setPlaying};
  draw();
})();
</script>
</body>
</html>
"""
