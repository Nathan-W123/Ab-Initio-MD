"""
MD driver: runs an integrator for N steps and records energies and frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from aimd.integrators import Integrator
from aimd.system import MolecularSystem
from aimd.trajectory import EnergyLogger, XYZWriter


@dataclass
class MDResult:
    records: list[dict] = field(default_factory=list)

    def column(self, key: str) -> np.ndarray:
        return np.array([r[key] for r in self.records])

    @property
    def total_energy_drift(self) -> float:
        """Max |E_tot(t) - E_tot(0)| in hartree over the recorded steps."""
        e = self.column("total_Eh")
        return float(np.max(np.abs(e - e[0])))


def _record(step: int, integrator: Integrator, system: MolecularSystem) -> dict:
    epot = integrator.result.energy
    ekin = system.kinetic_energy()
    return {
        "step": step,
        "time_fs": step * integrator.timestep_fs,
        "potential_Eh": epot,
        "kinetic_Eh": ekin,
        "total_Eh": epot + ekin,
        "temperature_K": system.temperature(),
    }


def run_md(
    system: MolecularSystem,
    integrator: Integrator,
    n_steps: int,
    trajectory: str | Path | None = None,
    energy_log: str | Path | None = None,
    write_every: int = 1,
    callback: Callable[[dict], None] | None = None,
) -> MDResult:
    """
    Propagate ``system`` in place for ``n_steps`` steps.

    Step 0 (the starting geometry) is always recorded. Frames and log rows are
    written every ``write_every`` steps; ``callback`` receives every record.
    """
    if write_every < 1:
        raise ValueError("write_every must be >= 1")
    xyz = XYZWriter(trajectory) if trajectory else None
    log = EnergyLogger(energy_log) if energy_log else None
    result = MDResult()

    def emit(step: int) -> None:
        rec = _record(step, integrator, system)
        result.records.append(rec)
        if step % write_every == 0:
            if xyz:
                xyz.write(
                    system,
                    f"step={step} t={rec['time_fs']:.4f}fs "
                    f"E_pot={rec['potential_Eh']:.10f} E_tot={rec['total_Eh']:.10f}",
                )
            if log:
                log.write(rec)
        if callback:
            callback(rec)

    try:
        if integrator.result is None:
            integrator.initialize(system)
        emit(0)
        for step in range(1, n_steps + 1):
            integrator.step(system)
            emit(step)
    finally:
        if xyz:
            xyz.close()
        if log:
            log.close()
    return result
