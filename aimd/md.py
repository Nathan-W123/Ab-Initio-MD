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
step), so the files match those of an uninterrupted run. Before anything is
changed, a restart is refused (ValueError) if the backend's fingerprint
differs from the checkpoint's (another method / basis: another surface, see
aimd.checkpoint.backend_fingerprint), if ``write_every`` differs from the
checkpointed run's while files are appended to (mixed frame spacing), or if an
output file does not continue the checkpointed run: the last frame / row kept
at the checkpoint step s must be the one the run wrote last (step
s - s % write_every), list the same atoms and, when it is step s itself, hold
the checkpoint's positions / velocities / energies (to the precision of the
file format). This catches a file overwritten by another run in between (for
example the default ``trajectory.xyz``). The checkpoint path is tested for
writability before the first force call, so a bad path fails at once instead
of after the whole run.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from aimd.checkpoint import Checkpoint, check_writable, load_checkpoint, save_checkpoint
from aimd.integrators import Integrator
from aimd.system import MolecularSystem
from aimd.trajectory import (
    DIPOLE_COLUMNS,
    ENERGY_COLUMNS,
    DipoleLogger,
    EnergyLogger,
    VelocityWriter,
    VELOCITY_UNITS,
    XYZWriter,
    check_csv_columns,
    last_kept_frame,
    last_kept_row,
)
from aimd.units import BOHR_TO_ANG


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
    checkpoint_info: dict | None = None,
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
          after the last step. The path is checked for writability first.
      checkpoint_info
          Extra plain data stored in the checkpoint's ``run_info`` (next to
          ``write_every``), e.g. the backend's constructor arguments.
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
          With ``restart``, each file must continue the checkpointed run
          (module docstring); ValueError otherwise, with nothing changed.
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
    if checkpoint_path:
        # Fail now, not after the whole run when the first checkpoint is due.
        check_writable(checkpoint_path)
    if restart is not None:
        if start_step != 0 or start_time_fs is not None:
            raise ValueError("give either restart or start_step/start_time_fs")
        if not isinstance(restart, Checkpoint):
            restart = load_checkpoint(restart)
        restart.check_backend(integrator.backend)
        start_step, start_time_fs = restart.step, restart.time_fs
    if start_step < 0:
        raise ValueError("start_step must be non-negative")
    continuing = start_step > 0
    if append is None:
        append = continuing
    # Check the logs' columns before anything is changed or opened: a refused
    # append must leave the system, the integrator and every file untouched.
    if append:
        for path, columns in ((energy_log, ENERGY_COLUMNS), (dipole_log, DIPOLE_COLUMNS)):
            if path:
                check_csv_columns(path, columns)
        if continuing:
            _check_resumed_outputs(
                start_step, write_every, restart, trajectory=trajectory,
                velocity_trajectory=velocity_trajectory, energy_log=energy_log,
                dipole_log=dipole_log)
    if restart is not None:
        restart.restore(system, integrator)
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
    run_info = {**(checkpoint_info or {}), "write_every": int(write_every)}

    def checkpoint(step: int) -> None:
        nonlocal saved_step
        save_checkpoint(checkpoint_path, system, integrator, step, time_of(step),
                        run_info=run_info)
        saved_step = step

    try:
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


def _check_resumed_outputs(
    cut: int,
    write_every: int,
    restart: Checkpoint | None,
    **paths: str | Path | None,
) -> None:
    """
    Before appending to existing outputs at step ``cut``: refuse (ValueError)
    files that do not continue the run being resumed. Reads only.

    The frame / row kept last must be the last one the run wrote, step
    ``cut - cut % write_every``, with the run's atoms; at step ``cut`` itself
    its values must be the checkpoint's: positions (written to 1e-10
    angstrom), E_pot (1e-10 Eh in the XYZ comment), velocities and the energy
    log (17 significant digits / repr: agree to rounding), the dipole.
    """
    existing = {k: Path(p) for k, p in paths.items()
                if p and Path(p).exists() and Path(p).stat().st_size > 0}
    if not existing:
        return
    saved_every = None if restart is None else restart.run_info.get("write_every")
    if saved_every is not None and int(saved_every) != write_every:
        raise ValueError(
            f"write_every {write_every} differs from the checkpointed run's "
            f"{saved_every}: appending to {', '.join(map(str, existing.values()))} "
            "would mix two frame spacings (which aimd analyze ir / vdos refuse); use "
            f"write_every {saved_every}, or write to new files")
    want = cut - cut % write_every
    saved = None if restart is None else restart.saved_system
    result = None if restart is None else (restart.integrator_state.get("result") or {})

    def refuse(path: Path, why: str) -> None:
        raise ValueError(f"cannot append to {path}: {why}; it does not continue the "
                         f"checkpointed run (step {cut}). Move it away or write to "
                         "another file")

    def close(a: float, b: float, rtol: float, atol: float = 0.0) -> bool:
        return bool(np.isclose(a, b, rtol=rtol, atol=atol))

    for kind, path in existing.items():
        if kind in ("trajectory", "velocity_trajectory"):
            frame = last_kept_frame(path, cut)
            if frame is None:
                continue                     # nothing kept: written afresh from step cut
            symbols, values, info = frame
            step = info.get("step")
            if step != want:
                refuse(path, f"its last frame up to step {cut} is step {step}, but the run "
                             f"wrote step {want} last (write_every {write_every})")
            if saved is None:
                continue
            if symbols != list(saved.symbols):
                refuse(path, f"it holds atoms {symbols}, the checkpoint {saved.symbols}")
            if step != cut:
                continue
            if kind == "trajectory":
                dx = np.max(np.abs(values - saved.positions * BOHR_TO_ANG))
                epot = info.get("E_pot")
                if dx > 1e-9:
                    refuse(path, f"its step-{cut} positions differ from the checkpoint's "
                                 f"by up to {dx:.3g} angstrom")
                if result.get("energy") is not None and not (
                        isinstance(epot, float)
                        and close(epot, result["energy"], 0.0, 1e-9)):
                    refuse(path, f"its step-{cut} E_pot={epot} is not the checkpoint's "
                                 f"{result['energy']:.10f} Eh")
            else:
                scale = VELOCITY_UNITS.get(info.get("units"))
                if scale is None or not np.allclose(values / scale, saved.velocities,
                                                    rtol=1e-12, atol=1e-15):
                    refuse(path, f"its step-{cut} velocities are not the checkpoint's")
        else:
            row = last_kept_row(path, cut)
            if row is None:
                continue
            step = row.get("step")
            if step != want:
                refuse(path, f"its last row up to step {cut} is step {step}, but the run "
                             f"wrote step {want} last (write_every {write_every})")
            if restart is None or step != cut:
                continue
            if not close(row["time_fs"], restart.time_fs, 1e-12, 1e-9):
                refuse(path, f"its step-{cut} time {row['time_fs']} fs is not the "
                             f"checkpoint's {restart.time_fs} fs")
            if kind == "energy_log":
                checks = [("potential_Eh", result.get("energy")),
                          ("kinetic_Eh", saved.kinetic_energy())]
            else:
                dip = result.get("dipole")
                checks = [] if dip is None else list(zip(
                    ("dipole_x_au", "dipole_y_au", "dipole_z_au"),
                    np.asarray(dip, dtype=float).reshape(3)))
            for col, ref in checks:
                if ref is not None and not close(row[col], ref, 1e-12, 1e-14):
                    refuse(path, f"its step-{cut} {col} = {row[col]!r} is not the "
                                 f"checkpoint's {float(ref)!r}")
