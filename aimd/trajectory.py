"""
Trajectory and log files: the writers run_md uses, and readers for analysis.

Writers
  XYZWriter       - positions, multi-frame XYZ in angstrom (VMD, Avogadro, ASE, OVITO)
  VelocityWriter  - velocities in the same XYZ layout; every comment line
                    states the unit (``units=bohr/au_time``, or angstrom/fs)
  EnergyLogger    - CSV, one row per logged step (ENERGY_COLUMNS)
  DipoleLogger    - CSV of the backend's dipole moment (DIPOLE_COLUMNS)

File formats
  positions   ``N`` / comment / ``N`` lines ``sym x y z`` (angstrom, %16.10f).
              run_md writes the comment ``step=<n> t=<t>fs E_pot=<Eh> E_tot=<Eh>``.
  velocities  same layout, ``sym vx vy vz`` with 17 significant digits (an
              exact float64 round trip in bohr / au_time); comment
              ``step=<n> t=<t>fs units=<unit>``. The unit travels with each
              frame, so a file is self-describing even if appended to in
              another unit.
  energies    header ``step,time_fs,potential_Eh,kinetic_Eh,total_Eh,
              temperature_K,conserved_Eh``; floats written with repr (exact).
  dipoles     header ``step,time_fs,dipole_x_au,dipole_y_au,dipole_z_au``,
              electronic + nuclear dipole in e * bohr (atomic units, 1 e bohr
              = 2.5417 D); ``nan`` where the backend reported no dipole.

Readers (engine units on the way in)
  read_xyz          -> XYZTrajectory:      positions (n_frames, N, 3), bohr
  read_velocities   -> VelocityTrajectory: velocities (n_frames, N, 3), bohr / au_time
  read_energy_log   -> dict column -> array (``step`` as int)
  read_dipole_log   -> DipoleLog:          dipole (n_rows, 3), e * bohr
Comment lines are parsed into ``key=value`` metadata; ``step`` and the time
``t=<t>fs`` are exposed as arrays. A file being written (or cut by a crash)
may end in a partial frame or row: an incomplete one, or one whose last line
has no line break (it may stop inside a number), is dropped with a warning,
unless it is the file's only frame / row (a hand-written file).

Restart: the writers can append to an existing file when a run is continued
from a checkpoint. With ``truncate_after_step=s`` anything an earlier run wrote
for steps > s (for example after its last checkpoint, before it crashed) is cut
off first, so the continued file is identical to that of an uninterrupted run.
``resumed`` tells whether the file already held data after that cut.
:func:`last_kept_frame` / :func:`last_kept_row` return, without changing the
file, what that cut would keep last; run_md checks it against the checkpoint
before appending.
"""

from __future__ import annotations

import csv
import io
import math
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any, Mapping

import numpy as np

from aimd.elements import ATOMIC_MASSES_AMU, ATOMIC_NUMBERS
from aimd.system import MolecularSystem
from aimd.units import ANG_TO_BOHR, AMU_TO_AU, AU_TIME_TO_FS, BOHR_TO_ANG

ENERGY_COLUMNS = [
    "step", "time_fs", "potential_Eh", "kinetic_Eh", "total_Eh", "temperature_K",
    "conserved_Eh",
]

DIPOLE_COLUMNS = ["step", "time_fs", "dipole_x_au", "dipole_y_au", "dipole_z_au"]

# Velocity units a file may be written in: value in that unit per bohr / au_time.
VELOCITY_UNITS: dict[str, float] = {
    "bohr/au_time": 1.0,
    "angstrom/fs": BOHR_TO_ANG / AU_TIME_TO_FS,
}

_STEP_RE = re.compile(r"\bstep=(-?\d+)")
_KEY_VALUE_RE = re.compile(r"(\w+)=(\S+)")
_TIME_RE = re.compile(r"^(.*?)fs$")


# ── Restart truncation ────────────────────────────────────────────────────────

def _cut(path: Path, offset: int) -> None:
    with path.open("r+b") as fh:
        fh.truncate(offset)


def _scan_xyz(data: bytes, after_step: int) -> tuple[int, list[bytes] | None, bool]:
    """
    Walk the frames of an XYZ-layout file up to the first frame whose comment
    has step > after_step, or an incomplete trailing frame (crash mid-write).
    Returns (byte offset where the kept frames end, lines of the last kept
    frame or None, well_formed); well_formed is False if a frame header is not
    an atom count (then nothing is cut: the file is not ours to edit).
    """
    lines = data.splitlines(keepends=True)
    offset, i, last = 0, 0, None
    while i < len(lines):
        try:
            n = int(lines[i].split()[0])
        except (IndexError, ValueError):
            return offset, last, False
        frame = lines[i : i + 2 + n]
        complete = len(frame) == 2 + n and frame[-1].endswith(b"\n")
        m = _STEP_RE.search(frame[1].decode("utf-8", "replace")) if len(frame) > 1 else None
        if not complete or (m is not None and int(m.group(1)) > after_step):
            return offset, last, True
        offset += sum(len(ln) for ln in frame)
        last = frame
        i += 2 + n
    return offset, last, True


def _truncate_xyz(path: Path, after_step: int) -> None:
    """
    Drop the first frame whose comment has step > after_step, and everything
    after it; an incomplete trailing frame (crash mid-write) is dropped too.
    """
    data = path.read_bytes()
    offset, _, well_formed = _scan_xyz(data, after_step)
    if well_formed and offset < len(data):
        _cut(path, offset)


def _scan_csv(data: bytes, after_step: int) -> tuple[int, bytes | None, bool]:
    """CSV analogue of _scan_xyz: (offset, last kept data row or None, well_formed)."""
    lines = data.splitlines(keepends=True)
    offset = len(lines[0]) if lines else 0
    last = None
    for ln in lines[1:]:
        try:
            step = int(ln.split(b",", 1)[0])
        except ValueError:
            return offset, last, False
        if step > after_step or not ln.endswith(b"\n"):
            return offset, last, True
        offset += len(ln)
        last = ln
    return offset, last, True


def _truncate_csv(path: Path, after_step: int) -> None:
    """Drop the first data row with step > after_step (or a partial row) and the rest."""
    data = path.read_bytes()
    offset, _, well_formed = _scan_csv(data, after_step)
    if well_formed and offset < len(data):
        _cut(path, offset)


def last_kept_frame(
    path: str | Path, after_step: int
) -> tuple[list[str], np.ndarray, dict[str, Any]] | None:
    """
    Read-only: the last frame of an XYZ-layout file that restart truncation at
    ``after_step`` would keep, as (symbols, raw values (N, 3) in the file's
    units, parsed comment); None if no frame would be kept. ValueError for a
    file the writers cannot have produced.
    """
    path = Path(path)
    _, frame, well_formed = _scan_xyz(path.read_bytes(), after_step)
    if not well_formed:
        raise ValueError(f"{path} is not an XYZ trajectory (a frame header is not an atom count)")
    if frame is None:
        return None
    text = [ln.decode("utf-8", "replace") for ln in frame]
    try:
        body = [ln.split() for ln in text[2:]]
        symbols = [_canonical_label(p[0]) for p in body]
        values = np.array([p[1:4] for p in body], dtype=float).reshape(-1, 3)
    except (IndexError, ValueError):
        raise ValueError(f"{path}: malformed frame ({text[1].strip()!r})") from None
    return symbols, values, parse_comment(text[1])


def last_kept_row(path: str | Path, after_step: int) -> dict[str, float] | None:
    """
    Read-only: the last data row of a CSV log that restart truncation at
    ``after_step`` would keep, column -> float (``step`` as int); None if none.
    """
    path = Path(path)
    data = path.read_bytes()
    _, row, well_formed = _scan_csv(data, after_step)
    if not well_formed:
        raise ValueError(f"{path}: a data row does not start with a step number")
    if row is None:
        return None
    header = next(csv.reader(io.StringIO(data.splitlines()[0].decode())), [])
    fields_ = next(csv.reader(io.StringIO(row.decode("utf-8", "replace"))), [])
    values = _parse_row(fields_, len(header))
    if values is None:
        raise ValueError(f"{path}: malformed row {row.decode('utf-8', 'replace').strip()!r}")
    out = dict(zip(header, values))
    if "step" in out:
        out["step"] = int(out["step"])
    return out


def check_csv_columns(path: str | Path, columns: list[str]) -> bool:
    """
    True if ``path`` exists with a header, which must equal ``columns``
    (ValueError otherwise); False for a missing or empty file. Reads only.
    run_md checks every log before it opens (and truncates) any output, so a
    refused restart leaves all files untouched.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open(newline="") as fh:
        header = next(csv.reader(fh), [])
    if header != columns:
        raise ValueError(f"cannot append to {path}: columns {header} != {columns}")
    return True


# ── Writers ───────────────────────────────────────────────────────────────────

class _FrameWriter:
    """XYZ-layout frames (atom count, comment, one line per atom)."""

    def __init__(
        self,
        path: str | Path,
        append: bool = False,
        truncate_after_step: int | None = None,
    ) -> None:
        self.path = Path(path)
        if append and truncate_after_step is not None and self.path.exists():
            _truncate_xyz(self.path, truncate_after_step)
        self.resumed = append and self.path.exists() and self.path.stat().st_size > 0
        self._fh: IO[str] = self.path.open("a" if append else "w")

    def _write(self, symbols: list[str], rows: np.ndarray, comment: str, fmt: str) -> None:
        out = [f"{len(symbols)}\n{comment}\n"]
        out.extend(f"{sym:<3s} {fmt.format(*row)}\n" for sym, row in zip(symbols, rows))
        self._fh.write("".join(out))
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class XYZWriter(_FrameWriter):
    """Positions in angstrom, ``%16.10f`` (the format restart-append relies on)."""

    def write(self, system: MolecularSystem, comment: str = "") -> None:
        self._write(system.symbols, system.positions * BOHR_TO_ANG, comment,
                    "{:16.10f} {:16.10f} {:16.10f}")


class VelocityWriter(_FrameWriter):
    """
    Velocities in ``units`` (a key of VELOCITY_UNITS), 17 significant digits.
    ``units=<unit>`` is appended to every comment line.
    """

    def __init__(
        self,
        path: str | Path,
        append: bool = False,
        truncate_after_step: int | None = None,
        units: str = "bohr/au_time",
    ) -> None:
        if units not in VELOCITY_UNITS:
            raise ValueError(f"unknown velocity unit {units!r}; use one of {list(VELOCITY_UNITS)}")
        self.units = units
        self._scale = VELOCITY_UNITS[units]
        super().__init__(path, append, truncate_after_step)

    def write(self, system: MolecularSystem, comment: str = "") -> None:
        tag = f"units={self.units}"
        self._write(system.symbols, system.velocities * self._scale,
                    f"{comment} {tag}" if comment else tag,
                    "{:24.16e} {:24.16e} {:24.16e}")


class CSVLogger:
    """One CSV row per call; ``columns`` is fixed by the subclass."""

    columns: list[str] = []

    def __init__(
        self,
        path: str | Path,
        append: bool = False,
        truncate_after_step: int | None = None,
    ) -> None:
        self.path = Path(path)
        has_header = append and check_csv_columns(self.path, self.columns)
        if has_header and truncate_after_step is not None:
            _truncate_csv(self.path, truncate_after_step)
        if has_header:
            with self.path.open("rb") as fh:
                self.resumed = sum(1 for _ in fh) > 1
        else:
            self.resumed = False
        self._fh = self.path.open("a" if has_header else "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.columns)
        if not has_header:
            self._writer.writeheader()

    def write(self, record: Mapping[str, Any]) -> None:
        self._writer.writerow({k: record[k] for k in self.columns})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class EnergyLogger(CSVLogger):
    columns = ENERGY_COLUMNS


class DipoleLogger(CSVLogger):
    columns = DIPOLE_COLUMNS

    def write_dipole(self, step: int, time_fs: float, dipole: np.ndarray | None) -> bool:
        """Write one row (``nan`` if ``dipole`` is None); returns whether it had one."""
        if dipole is None:
            d = (math.nan,) * 3
        else:
            d = tuple(float(v) for v in np.asarray(dipole, dtype=float).reshape(3))
        self.write(dict(zip(self.columns, (step, time_fs, *d))))
        return dipole is not None


# ── Readers ───────────────────────────────────────────────────────────────────

def parse_comment(comment: str) -> dict[str, Any]:
    """
    ``key=value`` pairs of an XYZ comment line. Values become int or float when
    they parse as such, else stay strings; ``t=<x>fs`` is stored as
    ``time_fs`` (float).
    """
    info: dict[str, Any] = {}
    for key, raw in _KEY_VALUE_RE.findall(comment):
        if key == "t":
            m = _TIME_RE.match(raw)
            if m:
                try:
                    info["time_fs"] = float(m.group(1))
                    continue
                except ValueError:
                    pass
        value: Any = raw
        for conv in (int, float):
            try:
                value = conv(raw)
                break
            except ValueError:
                pass
        info[key] = value
    return info


def frame_interval(step: np.ndarray | None, time_fs: np.ndarray | None) -> float:
    """
    Spacing in fs of evenly spaced frames. With step numbers the steps must be
    evenly spaced and increasing; the interval is then (t_last - t_0) /
    (n - 1), which cancels most of the rounding of times printed to 1e-4 fs.
    """
    if time_fs is None or len(time_fs) < 2:
        raise ValueError("need at least two frames with times")
    t = np.asarray(time_fs, dtype=float)
    if step is not None:
        ds = np.diff(np.asarray(step))
        if ds[0] <= 0 or np.any(ds != ds[0]):
            raise ValueError("frames are not evenly spaced in step "
                             "(several runs appended to one file?)")
    dt = (t[-1] - t[0]) / (len(t) - 1)
    # Times in XYZ comments carry 4 decimals: diffs are exact to 1e-4 fs.
    if dt <= 0.0 or np.max(np.abs(np.diff(t) - dt)) > 2e-4 + 1e-6 * dt:
        raise ValueError("frames are not evenly spaced in time")
    return float(dt)


def _standard_masses(symbols: list[str]) -> np.ndarray:
    try:
        return np.array([ATOMIC_MASSES_AMU[s] * AMU_TO_AU for s in symbols])
    except KeyError as e:
        raise ValueError(f"no standard mass for atom label {e.args[0]!r}") from None


class _Frames:
    """Metadata shared by the XYZ-layout readers' results."""

    symbols: list[str]
    comments: list[str]
    info: list[dict[str, Any]]

    def _values(self) -> np.ndarray:
        raise NotImplementedError

    def _meta(self, key: str, dtype: type) -> np.ndarray | None:
        if not self.info or any(key not in d for d in self.info):
            return None
        return np.array([d[key] for d in self.info], dtype=dtype)

    @property
    def n_frames(self) -> int:
        return int(self._values().shape[0])

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def step(self) -> np.ndarray | None:
        """Step numbers from the comments, or None if any frame lacks one."""
        return self._meta("step", np.int64)

    @property
    def time_fs(self) -> np.ndarray | None:
        """Times (fs) from the ``t=<t>fs`` comments, or None if any frame lacks one."""
        return self._meta("time_fs", float)

    @property
    def frame_interval_fs(self) -> float:
        """Spacing of evenly spaced frames in fs (ValueError otherwise)."""
        return frame_interval(self.step, self.time_fs)

    @property
    def masses(self) -> np.ndarray:
        """Standard atomic masses (m_e) of the atoms; custom isotopes are not known here."""
        return _standard_masses(self.symbols)


@dataclass
class XYZTrajectory(_Frames):
    symbols: list[str]
    positions: np.ndarray                    # bohr, (n_frames, N, 3)
    comments: list[str] = field(default_factory=list)
    info: list[dict[str, Any]] = field(default_factory=list)

    def _values(self) -> np.ndarray:
        return self.positions


@dataclass
class VelocityTrajectory(_Frames):
    symbols: list[str]
    velocities: np.ndarray                   # bohr / au_time, (n_frames, N, 3)
    comments: list[str] = field(default_factory=list)
    info: list[dict[str, Any]] = field(default_factory=list)

    def _values(self) -> np.ndarray:
        return self.velocities


def _canonical_label(label: str) -> str:
    sym = label[:1].upper() + label[1:].lower()
    return sym if sym in ATOMIC_NUMBERS else label


def _cut_off_tail(text: str) -> bool:
    """
    True if the last non-blank line of ``text`` has no line break: the writers
    end every line with one, so such a line was cut off mid-write (possibly
    inside a number) - the same rule restart truncation applies.
    """
    body = text.rstrip(" \t")                  # no copy when nothing is stripped
    return bool(body) and not body.endswith(("\n", "\r"))


def _read_frames(path: str | Path) -> tuple[list[str], np.ndarray, list[str]]:
    """Symbols, raw values (n_frames, N, 3) and comments of an XYZ-layout file."""
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()
    symbols: list[str] | None = None
    rows: list[list[str]] = []
    comments: list[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():                   # blank separator / trailing lines
            i += 1
            continue
        try:
            n = int(lines[i].split()[0])
        except ValueError:
            raise ValueError(f"{path}:{i + 1}: expected an atom count, got {lines[i]!r}") from None
        end = i + 2 + n
        body = [ln.split() for ln in lines[i + 2 : end]]
        last = all(not ln.strip() for ln in lines[end:])
        # A final frame whose last line lacks its line break may end inside a
        # number. A lone frame is kept: far more likely a hand-written file.
        cut_off = last and bool(comments) and _cut_off_tail(text)
        if end > len(lines) or any(len(p) < 4 for p in body) or cut_off:
            if last:
                # The last frame, cut off by a crash or a run still writing.
                warnings.warn(f"{path}: incomplete last frame (line {i + 1}) ignored",
                              stacklevel=3)
                break
            raise ValueError(f"{path}:{i + 1}: malformed frame")
        labels = [_canonical_label(p[0]) for p in body]
        if symbols is None:
            symbols = labels
        elif labels != symbols:
            raise ValueError(f"{path}:{i + 1}: atoms differ from the first frame")
        rows.extend(p[1:4] for p in body)
        comments.append(lines[i + 1])
        i = end
    if symbols is None:
        raise ValueError(f"{path}: no frames")
    try:
        values = np.array(rows, dtype=float).reshape(len(comments), len(symbols), 3)
    except ValueError as e:
        raise ValueError(f"{path}: non-numeric coordinate ({e})") from None
    return symbols, values, comments


def read_xyz(path: str | Path) -> XYZTrajectory:
    """
    All frames of a multi-frame XYZ file (angstrom) as positions in bohr,
    shape (n_frames, N, 3). Every frame must list the same atoms. An
    incomplete last frame is skipped with a warning. A velocity file (frames
    stating a velocity unit, ``units=bohr/au_time``) is refused with a
    ValueError rather than read as coordinates.
    """
    symbols, values, comments = _read_frames(path)
    info = [parse_comment(c) for c in comments]
    for k, d in enumerate(info):
        if d.get("units") in VELOCITY_UNITS:
            raise ValueError(f"{path} is a velocity file (frame {k}: units={d['units']}), "
                             "not positions (velocities are read by read_velocities, "
                             "aimd analyze vacf / vdos)")
    return XYZTrajectory(
        symbols=symbols,
        positions=values * ANG_TO_BOHR,
        comments=comments,
        info=info,
    )


def read_velocities(path: str | Path, units: str | None = None) -> VelocityTrajectory:
    """
    Velocities written by :class:`VelocityWriter`, converted to bohr / au_time.

    Each frame's unit is read from its ``units=`` comment; ``units`` is used
    for frames without one (ValueError if neither is given).
    """
    symbols, values, comments = _read_frames(path)
    info = [parse_comment(c) for c in comments]
    scale = np.empty(len(comments))
    for k, d in enumerate(info):
        unit = d.get("units", units)
        if unit is None:
            raise ValueError(f"{path}: frame {k} states no velocity unit (no units= on its "
                             "comment line); give it with units= (aimd analyze: --units)")
        if unit not in VELOCITY_UNITS:
            raise ValueError(f"{path}: frame {k}: unknown velocity unit {unit!r}")
        scale[k] = VELOCITY_UNITS[unit]
    return VelocityTrajectory(
        symbols=symbols,
        velocities=values / scale[:, None, None],
        comments=comments,
        info=info,
    )


def _parse_row(row: list[str], width: int) -> list[float] | None:
    if len(row) != width:
        return None
    try:
        return [float(v) for v in row]
    except ValueError:
        return None


def read_csv_log(path: str | Path) -> dict[str, np.ndarray]:
    """
    Columns of a CSV log as float arrays (``step`` as int64), in file order.
    A malformed last row, or one without its line break (cut off by a crash
    or a run still writing, possibly inside a number) is skipped with a
    warning, unless it is the only row; a malformed row elsewhere is an error.
    """
    path = Path(path)
    with path.open(newline="") as fh:
        text = fh.read()
    reader = csv.reader(io.StringIO(text))
    header = next(reader, None)
    if not header:
        raise ValueError(f"{path}: empty log")
    raw = [r for r in reader if r]
    rows = [_parse_row(r, len(header)) for r in raw]
    if rows and (rows[-1] is None or (len(rows) > 1 and _cut_off_tail(text))):
        warnings.warn(f"{path}: incomplete last row ignored", stacklevel=2)
        rows.pop()
    if any(r is None for r in rows):
        raise ValueError(f"{path}: malformed row")
    data = np.array(rows, dtype=float).reshape(len(rows), len(header))
    out = {name: data[:, k] for k, name in enumerate(header)}
    if "step" in out:
        out["step"] = out["step"].astype(np.int64)
    return out


def read_energy_log(path: str | Path) -> dict[str, np.ndarray]:
    """The energy CSV written by run_md: column name -> array (ENERGY_COLUMNS)."""
    return read_csv_log(path)


@dataclass
class DipoleLog:
    step: np.ndarray                         # (n,), int
    time_fs: np.ndarray                      # (n,)
    dipole: np.ndarray                       # e * bohr, (n, 3); nan rows: no dipole

    @property
    def frame_interval_fs(self) -> float:
        return frame_interval(self.step, self.time_fs)


def read_dipole_log(path: str | Path) -> DipoleLog:
    """The dipole CSV written by run_md (``dipole_log=``)."""
    cols = read_csv_log(path)
    missing = [c for c in DIPOLE_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"{path}: not a dipole log (missing {missing})")
    return DipoleLog(
        step=cols["step"],
        time_fs=cols["time_fs"],
        dipole=np.stack([cols[c] for c in DIPOLE_COLUMNS[2:]], axis=1),
    )
