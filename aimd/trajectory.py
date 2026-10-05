"""
Trajectory and energy-log output.

  XYZWriter    - multi-frame XYZ in angstrom (readable by VMD, Avogadro, ASE, OVITO)
  EnergyLogger - CSV with one row per logged step
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import IO

from aimd.system import MolecularSystem
from aimd.units import BOHR_TO_ANG

ENERGY_COLUMNS = [
    "step", "time_fs", "potential_Eh", "kinetic_Eh", "total_Eh", "temperature_K",
]


class XYZWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh: IO[str] = self.path.open("w")

    def write(self, system: MolecularSystem, comment: str = "") -> None:
        pos = system.positions * BOHR_TO_ANG
        self._fh.write(f"{system.n_atoms}\n{comment}\n")
        for sym, (x, y, z) in zip(system.symbols, pos):
            self._fh.write(f"{sym:<3s} {x:16.10f} {y:16.10f} {z:16.10f}\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class EnergyLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fh = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=ENERGY_COLUMNS)
        self._writer.writeheader()

    def write(self, record: dict) -> None:
        self._writer.writerow({k: record[k] for k in ENERGY_COLUMNS})
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
