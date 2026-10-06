"""
MD driver: runs an integrator for N steps and records energies and frames.

Each record holds the step, time, potential / kinetic / total energy, kinetic
temperature and the integrator's conserved quantity ``conserved_Eh`` (total
energy for NVE, extended Hamiltonian for Nose-Hoover chains, effective energy
for CSVR / Berendsen / Langevin; see aimd.integrators).

Steps whose force call reported ``converged=False`` (an SCF that hit its
iteration cap) are listed in ``MDResult.unconverged_steps`` and summarised in
one warning at the end of the run; the run itself continues with those forces.

Output files (formats and readers in aimd.trajectory), every ``write_every``
steps: ``trajectory`` (positions, XYZ in angstrom), ``velocity_trajectory``
(velocities, XYZ layout in bohr / au_time with the unit on every comment
line), ``energy_log`` (CSV of the record columns) and ``dipole_log`` (CSV of
GradientResult.dipole in e * bohr; ``nan`` when the backend gives none, with
one warning at the end of the run).

Restart: ``run_md(..., checkpoint_path=..., checkpoint_every=k)`` writes a
checkpoint every k steps and at the end. To continue, pass the loaded
checkpoint as ``restart``; step numbers and times carry on, and the output
files are appended to (after cutting anything written beyond the checkpoint
step), so the files match those of an uninterrupted run.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from aimd.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from aimd.integrators import Integrator
from aimd.system import MolecularSystem
from aimd.trajectory import (
    DIPOLE_COLUMNS,
    ENERGY_COLUMNS,
    DipoleLogger,
    EnergyLogger,
    VelocityWriter,
    XYZWriter,
    check_csv_columns,
)


@dataclass
class MDResult:
    records: list[dict] = field(default_factory=list)
    # Recorded steps whose forces came from an unconverged calculation.
    unconverged_steps: list[int] = field(default_factory=list)

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
    velocity_trajectory: str | Path | None = None,
    dipole_log: str | Path | None = None,
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

    Outputs (each optional, see aimd.trajectory for the formats):
      trajectory           positions, multi-frame XYZ (angstrom)
      velocity_trajectory  velocities, XYZ layout (bohr / au_time), synchronous
                           with the positions of the same step
      energy_log           CSV with the record columns (ENERGY_COLUMNS)
      dipole_log           CSV of the backend's dipole (DIPOLE_COLUMNS, e * bohr)
    The paths, and ``checkpoint_path``, must all differ (ValueError).

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
          written again (it is written to an output that holds no data yet).
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
    # A checkpoint replaces its file, so it must not share one with an output.
    outputs = [Path(p).resolve() for p in
               (trajectory, velocity_trajectory, energy_log, dipole_log, checkpoint_path) if p]
    if len(set(outputs)) != len(outputs):
        raise ValueError("output files (and checkpoint_path) must be distinct")
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
    xyz: XYZWriter | None = None
    vel: VelocityWriter | None = None
    log: EnergyLogger | None = None
    dip: DipoleLogger | None = None
    result = MDResult()
    no_dipole = 0

    def due(out: XYZWriter | VelocityWriter | EnergyLogger | DipoleLogger | None,
            initial: bool) -> bool:
        # The starting frame of a continued run is already in a resumed file.
        return out is not None and not (initial and resume_files and out.resumed)

    def emit(step: int, initial: bool = False) -> None:
        nonlocal no_dipole
        rec = _record(step, time_of(step), integrator, system)
        result.records.append(rec)
        if not integrator.result.converged:
            result.unconverged_steps.append(step)
        if step % write_every == 0:
            stamp = f"step={step} t={rec['time_fs']:.4f}fs"
            if due(xyz, initial):
                xyz.write(
                    system,
                    f"{stamp} E_pot={rec['potential_Eh']:.10f} E_tot={rec['total_Eh']:.10f}",
                )
            if due(vel, initial):
                vel.write(system, stamp)
            if due(log, initial):
                log.write(rec)
            if due(dip, initial) and not dip.write_dipole(
                step, rec["time_fs"], integrator.result.dipole
            ):
                no_dipole += 1
        if callback:
            callback(rec)

    saved_step = None

    def checkpoint(step: int) -> None:
        nonlocal saved_step
        save_checkpoint(checkpoint_path, system, integrator, step, time_of(step))
        saved_step = step

    try:
        # Check the logs' columns before anything is opened: a refused append
        # must not have truncated any of the other files already.
        if append:
            for path, columns in ((energy_log, ENERGY_COLUMNS), (dipole_log, DIPOLE_COLUMNS)):
                if path:
                    check_csv_columns(path, columns)
        if energy_log:
            log = EnergyLogger(energy_log, append, cut)
        if dipole_log:
            dip = DipoleLogger(dipole_log, append, cut)
        if trajectory:
            xyz = XYZWriter(trajectory, append, cut)
        if velocity_trajectory:
            vel = VelocityWriter(velocity_trajectory, append, cut)
        if integrator.result is None:
            integrator.initialize(system)
        emit(start_step, initial=True)
        last = start_step + n_steps
        for step in range(start_step + 1, last + 1):
            integrator.step(system)
            emit(step)
            if checkpoint_every and step % checkpoint_every == 0:
                checkpoint(step)
        if checkpoint_path and saved_step != last:
            checkpoint(last)
        if result.unconverged_steps:
            bad = result.unconverged_steps
            warnings.warn(
                f"{len(bad)} of {len(result.records)} force evaluations did not "
                f"converge (first at step {bad[0]}); see MDResult.unconverged_steps",
                RuntimeWarning,
                stacklevel=2,
            )
        if no_dipole:
            warnings.warn(
                f"backend {getattr(integrator.backend, 'name', '')!r} reported no "
                f"dipole for {no_dipole} logged step(s); written as nan to {dipole_log}",
                RuntimeWarning,
                stacklevel=2,
            )
    finally:
        for out in (xyz, vel, log, dip):
            if out:
                out.close()
    return result
