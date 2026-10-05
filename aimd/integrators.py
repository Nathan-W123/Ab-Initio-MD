"""
Time integrators for Born-Oppenheimer MD.

Every integrator calls the backend exactly once per step, at the new positions,
and caches that result so the next step can reuse it.

  VelocityVerlet  - NVE; optionally with a Berendsen (weak-coupling) thermostat
  LangevinBAOAB   - NVT; BAOAB splitting of Leimkuhler & Matthews (2013),
                    which gives accurate configurational sampling at large dt
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from aimd.backends.base import ForceBackend, GradientResult
from aimd.system import MolecularSystem
from aimd.units import FS_TO_AU_TIME, KB_AU


class Integrator(ABC):
    def __init__(self, backend: ForceBackend, timestep_fs: float) -> None:
        if timestep_fs <= 0.0:
            raise ValueError("timestep_fs must be positive")
        self.backend = backend
        self.timestep_fs = float(timestep_fs)
        self.dt = self.timestep_fs * FS_TO_AU_TIME
        self.result: GradientResult | None = None

    def initialize(self, system: MolecularSystem) -> GradientResult:
        """Evaluate forces at the starting geometry."""
        self.result = self.backend.compute(system.positions)
        return self.result

    def _accel(self, system: MolecularSystem) -> np.ndarray:
        return self.result.forces / system.masses[:, None]

    @abstractmethod
    def step(self, system: MolecularSystem) -> GradientResult:
        """Advance ``system`` in place by one timestep."""


class VelocityVerlet(Integrator):
    """
    Velocity Verlet. With ``berendsen_tau_fs`` and ``temperature_k`` set,
    velocities are rescaled after each step toward the target temperature
    (useful for equilibration; it does not sample the canonical ensemble).
    """

    def __init__(
        self,
        backend: ForceBackend,
        timestep_fs: float,
        temperature_k: float | None = None,
        berendsen_tau_fs: float | None = None,
    ) -> None:
        super().__init__(backend, timestep_fs)
        if (temperature_k is None) != (berendsen_tau_fs is None):
            raise ValueError(
                "Berendsen coupling needs both temperature_k and berendsen_tau_fs"
            )
        self.temperature_k = temperature_k
        self.tau = None if berendsen_tau_fs is None else berendsen_tau_fs * FS_TO_AU_TIME

    def step(self, system: MolecularSystem) -> GradientResult:
        if self.result is None:
            self.initialize(system)
        dt = self.dt
        system.velocities += 0.5 * dt * self._accel(system)
        system.positions += dt * system.velocities
        self.result = self.backend.compute(system.positions)
        system.velocities += 0.5 * dt * self._accel(system)

        if self.tau is not None:
            t_now = system.temperature()
            if t_now > 0.0:
                lam2 = 1.0 + (dt / self.tau) * (self.temperature_k / t_now - 1.0)
                system.velocities *= np.sqrt(max(lam2, 0.0))
        return self.result


class LangevinBAOAB(Integrator):
    """
    Langevin dynamics with the BAOAB splitting:
        B: v += dt/2 * a      A: x += dt/2 * v
        O: v = c1 v + c2 sqrt(kT/m) xi
        A: x += dt/2 * v      B: v += dt/2 * a   (a at new x)
    """

    def __init__(
        self,
        backend: ForceBackend,
        timestep_fs: float,
        temperature_k: float,
        friction_per_fs: float = 0.01,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        super().__init__(backend, timestep_fs)
        if temperature_k < 0.0:
            raise ValueError("temperature_k must be non-negative")
        self.temperature_k = float(temperature_k)
        self.gamma = float(friction_per_fs) / FS_TO_AU_TIME   # 1 / au_time
        self.rng = np.random.default_rng(rng)
        self.c1 = np.exp(-self.gamma * self.dt)
        self.c2 = np.sqrt(1.0 - self.c1**2)

    def step(self, system: MolecularSystem) -> GradientResult:
        if self.result is None:
            self.initialize(system)
        dt = self.dt
        system.velocities += 0.5 * dt * self._accel(system)
        system.positions += 0.5 * dt * system.velocities

        sigma = np.sqrt(KB_AU * self.temperature_k / system.masses)[:, None]
        noise = self.rng.normal(size=system.velocities.shape)
        system.velocities = self.c1 * system.velocities + self.c2 * sigma * noise
        # The random kicks carry net momentum, so the COM is no longer fixed.
        system.com_removed = False

        system.positions += 0.5 * dt * system.velocities
        self.result = self.backend.compute(system.positions)
        system.velocities += 0.5 * dt * self._accel(system)
        return self.result
