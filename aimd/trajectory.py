"""
Trajectory and energy-log output.

  XYZWriter    - multi-frame XYZ in angstrom (readable by VMD, Avogadro, ASE, OVITO)
  EnergyLogger - CSV with one row per logged step

Both can append to an existing file when a run is continued from a checkpoint.
With ``truncate_after_step=s`` anything an earlier run wrote for steps > s (for
example after its last checkpoint, before it crashed) is cut off first, so the
continued file is identical to that of an uninterrupted run.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import IO

from aimd.system import MolecularSystem
from aimd.units import BOHR_TO_ANG

ENERGY_COLUMNS = [
    "step", "time_fs", "potential_Eh", "kinetic_Eh", "total_Eh", "temperature_K",
    "conserved_Eh",
]

_STEP_RE = re.compile(r"\bstep=(-?\d+)")


def _cut(path: Path, offset: int) -> None:
    with path.open("r+b") as fh:
        fh.truncate(offset)


def _truncate_xyz(path: Path, after_step: int) -> None:
    """
    Drop the first frame whose comment has step > after_step, and everything
    after it; an incomplete trailing frame (crash mid-write) is dropped too.
    """
    with path.open("rb") as fh:
        lines = fh.read().splitlines(keepends=True)
    offset, i = 0, 0
    while i < len(lines):
        try:
            n = int(lines[i].split()[0])
        except (IndexError, ValueError):
            return                                 # not a frame header: leave as is
        frame = lines[i : i + 2 + n]
        complete = len(frame) == 2 + n and frame[-1].endswith(b"\n")
        m = _STEP_RE.search(frame[1].decode("utf-8", "replace")) if len(frame) > 1 else None
        if not complete or (m is not None and int(m.group(1)) > after_step):
            _cut(path, offset)
            return
        offset += sum(len(ln) for ln in frame)
        i += 2 + n


def _truncate_csv(path: Path, after_step: int) -> None:
    """Drop the first data row with step > after_step (or a partial row) and the rest."""
    with path.open("rb") as fh:
        lines = fh.read().splitlines(keepends=True)
    offset = len(lines[0]) if lines else 0
    for ln in lines[1:]:
        try:
            step = int(ln.split(b",", 1)[0])
        except ValueError:
            return
        if step > after_step or not ln.endswith(b"\n"):
            _cut(path, offset)
            return
        offset += len(ln)


class XYZWriter:
    def __init__(
        self,
        path: str | Path,
        append: bool = False,
        truncate_after_step: int | None = None,
    ) -> None:
        self.path = Path(path)
        if append and truncate_after_step is not None and self.path.exists():
            _truncate_xyz(self.path, truncate_after_step)
        self._fh: IO[str] = self.path.open("a" if append else "w")

    def write(self, system: MolecularSystem, comment: str = "") -> None:
        pos = system.positions * BOHR_TO_ANG
        self._fh.write(f"{system.n_atoms}\n{comment}\n")
        for sym, (x, y, z) in zip(system.symbols, pos):
            self._fh.write(f"{sym:<3s} {x:16.10f} {y:16.10f} {z:16.10f}\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class EnergyLogger:
    def __init__(
        self,
        path: str | Path,
        append: bool = False,
        truncate_after_step: int | None = None,
    ) -> None:
        self.path = Path(path)
        has_rows = append and self.path.exists() and self.path.stat().st_size > 0
        if has_rows:
            with self.path.open(newline="") as fh:
                header = next(csv.reader(fh), [])
            if header != ENERGY_COLUMNS:
                raise ValueError(
                    f"cannot append to {self.path}: columns {header} != {ENERGY_COLUMNS}"
                )
            if truncate_after_step is not None:
                _truncate_csv(self.path, truncate_after_step)
        self._fh = self.path.open("a" if has_rows else "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=ENERGY_COLUMNS)
        if not has_rows:
            self._writer.writeheader()

    def write(self, record: dict) -> None:
        self._writer.writerow({k: record[k] for k in ENERGY_COLUMNS})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
