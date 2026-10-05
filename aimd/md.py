"""
MD driver: runs an integrator for N steps and records energies and frames.

Each record holds the step, time, potential / kinetic / total energy, kinetic
temperature and the integrator's conserved quantity ``conserved_Eh`` (total
energy for NVE, extended Hamiltonian for Nose-Hoover chains, effective energy
for CSVR / Berendsen / Langevin; see aimd.integrators).

Restart: ``run_md(..., checkpoint_path=..., checkpoint_every=k)`` writes a
checkpoint every k steps and at the end. To continue, pass the loaded
checkpoint as ``restart``; step numbers and times carry on, and the trajectory
and energy log are appended to (after cutting anything written beyond the
checkpoint step), so the files match those of an uninterrupted run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from aimd.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
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

    @property
    def conserved_energy_drift(self) -> float:
        """Max |E_cons(t) - E_cons(0)| in hartree over the recorded steps."""
        e = self.column("conserved_Eh")
        return float(np.max(np.abs(e - e[0])))


def _record(
    step: int, time_fs: float, integrator: Integrator, system: MolecularSystem
) -> dict:
    epot = integrator.result.energy
    ekin = system.kinetic_energy()
    return {
        "step": step,
        "time_fs": time_fs,
        "potential_Eh": epot,
        "kinetic_Eh": ekin,
        "total_Eh": epot + ekin,
        "temperature_K": system.temperature(),
        "conserved_Eh": epot + ekin + integrator.thermostat_energy(),
    }


def run_md(
    system: MolecularSystem,
    integrator: Integrator,
    n_steps: int,
    trajectory: str | Path | None = None,
    energy_log: str | Path | None = None,
    write_every: int = 1,
    callback: Callable[[dict], None] | None = None,
    *,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    restart: Checkpoint | str | Path | None = None,
    start_step: int = 0,
    start_time_fs: float | None = None,
    append: bool | None = None,
) -> MDResult:
    """
    Propagate ``system`` in place for ``n_steps`` steps.

    The starting state is always recorded. Frames and log rows are written
    every ``write_every`` steps (counted from step 0 of the whole simulation);
    ``callback`` receives every record.

    Restart options:
      checkpoint_path, checkpoint_every
          Write a checkpoint every ``checkpoint_every`` steps (if given) and
          after the last step.
      restart
          A :class:`~aimd.checkpoint.Checkpoint` (or its path). Its state is
          copied into ``system`` and ``integrator`` (which must be of the same
          type) and the run continues from its step and time.
      start_step, start_time_fs
          Step number / time of the starting state when continuing without a
          checkpoint object (default 0 and ``start_step * dt``).
      append
          Append to existing trajectory / log files instead of overwriting.
          Default: True when continuing (``start_step > 0``). When continuing,
          anything those files hold beyond ``start_step`` is discarded first,
          and the starting frame, already written by the previous run, is not
          written again.
    """
    if write_every < 1:
        raise ValueError("write_every must be >= 1")
    if checkpoint_every is not None:
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be >= 1")
        if not checkpoint_path:
            raise ValueError("checkpoint_every needs checkpoint_path")
    if n_steps < 0:
        raise ValueError("n_steps must be non-negative")
    if restart is not None:
        if start_step != 0 or start_time_fs is not None:
            raise ValueError("give either restart or start_step/start_time_fs")
        if not isinstance(restart, Checkpoint):
            restart = load_checkpoint(restart)
        restart.restore(system, integrator)
        start_step, start_time_fs = restart.step, restart.time_fs
    if start_step < 0:
        raise ValueError("start_step must be non-negative")
    continuing = start_step > 0
    if append is None:
        append = continuing
    # Continuing into existing files: drop anything written after start_step,
    # and do not write the starting frame again.
    resume_files = append and continuing

    dt_fs = integrator.timestep_fs
    if start_time_fs is None or start_time_fs == start_step * dt_fs:
        # Same expression as an uninterrupted run: identical time stamps.
        def time_of(step: int) -> float:
            return step * dt_fs
    else:
        t0 = float(start_time_fs)

        def time_of(step: int) -> float:
            return t0 + (step - start_step) * dt_fs

    cut = start_step if resume_files else None
    xyz = XYZWriter(trajectory, append, cut) if trajectory else None
    log = EnergyLogger(energy_log, append, cut) if energy_log else None
    result = MDResult()

    def emit(step: int, write: bool = True) -> None:
        rec = _record(step, time_of(step), integrator, system)
        result.records.append(rec)
        if write and step % write_every == 0:
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

    saved_step = None

    def checkpoint(step: int) -> None:
        nonlocal saved_step
        save_checkpoint(checkpoint_path, system, integrator, step, time_of(step))
        saved_step = step

    try:
        if integrator.result is None:
            integrator.initialize(system)
        emit(start_step, write=not resume_files)
        last = start_step + n_steps
        for step in range(start_step + 1, last + 1):
            integrator.step(system)
            emit(step)
            if checkpoint_every and step % checkpoint_every == 0:
                checkpoint(step)
        if checkpoint_path and saved_step != last:
            checkpoint(last)
    finally:
        if xyz:
            xyz.close()
        if log:
            log.close()
    return result
