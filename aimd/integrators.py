"""
Time integrators for Born-Oppenheimer MD.

Every integrator calls the backend exactly once per step, at the new positions,
and caches that result so the next step can reuse it.

  VelocityVerlet   - NVE, or NVT with a velocity thermostat from
                     aimd.thermostats applied as half-step operators around the
                     Verlet step (Berendsen via the legacy keyword arguments)
  CSVR             - VelocityVerlet + Bussi-Donadio-Parrinello stochastic
                     velocity rescaling (canonical)
  NoseHooverChain  - VelocityVerlet + MTK Nose-Hoover chain (canonical)
  LangevinBAOAB    - NVT; BAOAB splitting of Leimkuhler & Matthews (2013),
                     which gives accurate configurational sampling at large dt

Each integrator reports a conserved quantity, ``conserved_energy(system)``:
E_pot + E_kin for NVE, the extended Hamiltonian for Nose-Hoover chains, and the
"effective energy" E_pot + E_kin - (kinetic energy added by the thermostat) for
CSVR, Berendsen and Langevin (Bussi et al. 2007; Bussi & Parrinello, Phys. Rev.
E 75, 056707 (2007)). Its drift measures the integration error.

``state_dict()`` / ``load_state_dict()`` capture everything needed to continue a
run bit-for-bit: the cached GradientResult, RNG bit-generator states and
thermostat variables (see aimd.checkpoint).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from aimd.backends.base import ForceBackend, GradientResult
from aimd.system import MolecularSystem
from aimd.thermostats import (
    BerendsenThermostat,
    CSVRThermostat,
    NoseHooverChainThermostat,
    Thermostat,
    rng_from_state,
    rng_state,
    thermostat_from_config,
)
from aimd.units import FS_TO_AU_TIME, KB_AU


def _result_state(result: GradientResult | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "energy": float(result.energy),
        "gradient": np.array(result.gradient, dtype=float),
        "converged": bool(result.converged),
        "density": None if result.density is None else np.array(result.density),
        "dipole": None if result.dipole is None else np.array(result.dipole),
        "info": {k: (v.copy() if isinstance(v, np.ndarray) else v)
                 for k, v in result.info.items()},
    }


def _result_from_state(state: dict[str, Any] | None) -> GradientResult | None:
    if state is None:
        return None
    return GradientResult(
        energy=float(state["energy"]),
        gradient=np.array(state["gradient"], dtype=float),
        converged=bool(state["converged"]),
        density=None if state["density"] is None else np.array(state["density"]),
        dipole=None if state["dipole"] is None else np.array(state["dipole"]),
        info=dict(state.get("info") or {}),
    )


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

    # ── Conserved quantity ────────────────────────────────────────────────────

    def thermostat_energy(self) -> float:
        """Thermostat term of the conserved quantity (0 for NVE), hartree."""
        return 0.0

    def conserved_energy(self, system: MolecularSystem) -> float:
        """E_pot + E_kin + thermostat term at the current (synchronised) state."""
        if self.result is None:
            raise RuntimeError("integrator not initialised")
        return self.result.energy + system.kinetic_energy() + self.thermostat_energy()

    # ── Restart state ─────────────────────────────────────────────────────────

    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Constructor keyword arguments other than ``backend`` (JSON-able)."""

    @classmethod
    def from_config(cls, backend: ForceBackend, config: dict[str, Any]) -> "Integrator":
        """Construct a fresh integrator from :meth:`config` output."""
        return cls(backend, **config)

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to continue the run exactly (arrays are copies)."""
        return {
            "integrator": type(self).__name__,
            "config": self.config(),
            "result": _result_state(self.result),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """
        Restore the dynamical state saved by :meth:`state_dict`.

        The integrator type must match. Parameters such as the timestep or
        target temperature are those of ``self``, so they may be changed on a
        restart (the conserved quantity is then not continuous). If the backend
        supports density guesses, the saved density is passed on as the guess
        for the next SCF.
        """
        if state.get("integrator") != type(self).__name__:
            raise ValueError(
                f"cannot load {state.get('integrator')!r} state into "
                f"{type(self).__name__}"
            )
        self.result = _result_from_state(state["result"])
        # An SCF backend would have started the next step from this density.
        if (
            self.result is not None
            and self.result.density is not None
            and self.backend.supports_density_guess
        ):
            self.backend.set_density_guess(self.result.density)


class VelocityVerlet(Integrator):
    """
    Velocity Verlet, optionally with a velocity thermostat.

    ``VelocityVerlet(backend, dt)`` is NVE. With ``berendsen_tau_fs`` and
    ``temperature_k`` set, velocities are rescaled after each step toward the
    target temperature (useful for equilibration; it does not sample the
    canonical ensemble). Any :class:`aimd.thermostats.Thermostat` can be passed
    as ``thermostat`` instead.
    """

    def __init__(
        self,
        backend: ForceBackend,
        timestep_fs: float,
        temperature_k: float | None = None,
        berendsen_tau_fs: float | None = None,
        *,
        thermostat: Thermostat | None = None,
    ) -> None:
        super().__init__(backend, timestep_fs)
        if thermostat is not None and (
            temperature_k is not None or berendsen_tau_fs is not None
        ):
            raise ValueError("pass either a thermostat or the Berendsen arguments")
        if (temperature_k is None) != (berendsen_tau_fs is None):
            raise ValueError(
                "Berendsen coupling needs both temperature_k and berendsen_tau_fs"
            )
        if berendsen_tau_fs is not None:
            thermostat = BerendsenThermostat(temperature_k, berendsen_tau_fs)
        self.thermostat = thermostat

    @property
    def temperature_k(self) -> float | None:
        """Thermostat target in K (None for NVE); settable, e.g. for annealing."""
        return None if self.thermostat is None else self.thermostat.temperature_k

    @temperature_k.setter
    def temperature_k(self, value: float) -> None:
        if self.thermostat is None:
            raise AttributeError("an NVE integrator has no target temperature")
        self.thermostat.set_temperature(value)

    def initialize(self, system: MolecularSystem) -> GradientResult:
        if self.thermostat is not None:
            self.thermostat.initialize(system)
        return super().initialize(system)

    def step(self, system: MolecularSystem) -> GradientResult:
        if self.result is None:
            self.initialize(system)
        dt = self.dt
        thermo = self.thermostat
        if thermo is not None:
            thermo.begin_step(system, dt)
        system.velocities += 0.5 * dt * self._accel(system)
        system.positions += dt * system.velocities
        self.result = self.backend.compute(system.positions)
        system.velocities += 0.5 * dt * self._accel(system)
        if thermo is not None:
            thermo.end_step(system, dt)
        return self.result

    def thermostat_energy(self) -> float:
        return 0.0 if self.thermostat is None else self.thermostat.energy()

    def config(self) -> dict[str, Any]:
        cfg: dict[str, Any] = {"timestep_fs": self.timestep_fs}
        if self.thermostat is not None:
            cfg["thermostat"] = {
                "type": type(self.thermostat).__name__,
                "config": self.thermostat.config(),
            }
        return cfg

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["thermostat"] = (
            None if self.thermostat is None else self.thermostat.state_dict()
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved = state.get("thermostat")
        mine = None if self.thermostat is None else type(self.thermostat).__name__
        theirs = None if saved is None else saved.get("type")
        if mine != theirs:
            raise ValueError(f"checkpoint thermostat {theirs!r} != integrator's {mine!r}")
        super().load_state_dict(state)
        if saved is not None:
            self.thermostat.load_state_dict(saved)

    @classmethod
    def from_config(cls, backend: ForceBackend, config: dict[str, Any]) -> "VelocityVerlet":
        cfg = dict(config)
        thermo = cfg.pop("thermostat", None)
        if thermo is not None:
            cfg["thermostat"] = thermostat_from_config(thermo["type"], thermo["config"])
        return cls(backend, **cfg)


class CSVR(VelocityVerlet):
    """Velocity Verlet with Bussi-Donadio-Parrinello stochastic velocity rescaling."""

    def __init__(
        self,
        backend: ForceBackend,
        timestep_fs: float,
        temperature_k: float,
        tau_fs: float = 100.0,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        super().__init__(
            backend, timestep_fs, thermostat=CSVRThermostat(temperature_k, tau_fs, rng)
        )

    def config(self) -> dict[str, Any]:
        return {"timestep_fs": self.timestep_fs, **self.thermostat.config()}


class NoseHooverChain(VelocityVerlet):
    """Velocity Verlet with an MTK Nose-Hoover chain (see NoseHooverChainThermostat)."""

    def __init__(
        self,
        backend: ForceBackend,
        timestep_fs: float,
        temperature_k: float,
        period_fs: float = 100.0,
        chain_length: int = 3,
        n_mts: int = 1,
        n_suzuki_yoshida: int = 3,
    ) -> None:
        super().__init__(
            backend,
            timestep_fs,
            thermostat=NoseHooverChainThermostat(
                temperature_k, period_fs, chain_length, n_mts, n_suzuki_yoshida
            ),
        )

    def config(self) -> dict[str, Any]:
        return {"timestep_fs": self.timestep_fs, **self.thermostat.config()}


class LangevinBAOAB(Integrator):
    """
    Langevin dynamics with the BAOAB splitting:
        B: v += dt/2 * a      A: x += dt/2 * v
        O: v = c1 v + c2 sqrt(kT/m) xi,   c1 = exp(-gamma dt), c2 = sqrt(1 - c1^2)
        A: x += dt/2 * v      B: v += dt/2 * a   (a at new x)
    The kinetic energy changed by the O step is accumulated in ``heat``, so the
    effective energy E_pot + E_kin - heat is conserved up to integration error.
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
        if friction_per_fs < 0.0:
            raise ValueError("friction_per_fs must be non-negative")
        self.temperature_k = float(temperature_k)
        self.friction_per_fs = float(friction_per_fs)
        self.gamma = self.friction_per_fs / FS_TO_AU_TIME   # 1 / au_time
        self.rng = np.random.default_rng(rng)
        self.c1 = math.exp(-self.gamma * self.dt)
        self.c2 = math.sqrt(1.0 - self.c1**2)
        self.heat = 0.0

    def step(self, system: MolecularSystem) -> GradientResult:
        if self.result is None:
            self.initialize(system)
        dt = self.dt
        system.velocities += 0.5 * dt * self._accel(system)
        system.positions += 0.5 * dt * system.velocities

        ekin = system.kinetic_energy()
        sigma = np.sqrt(KB_AU * self.temperature_k / system.masses)[:, None]
        noise = self.rng.normal(size=system.velocities.shape)
        system.velocities = self.c1 * system.velocities + self.c2 * sigma * noise
        self.heat += system.kinetic_energy() - ekin
        # Per-atom random kicks carry net momentum and angular momentum, so all
        # 3N momenta are thermalised.
        system.com_removed = False
        system.rotation_removed = False
        system.rotational_dof = None

        system.positions += 0.5 * dt * system.velocities
        self.result = self.backend.compute(system.positions)
        system.velocities += 0.5 * dt * self._accel(system)
        return self.result

    def thermostat_energy(self) -> float:
        return -self.heat

    def config(self) -> dict[str, Any]:
        return {
            "timestep_fs": self.timestep_fs,
            "temperature_k": self.temperature_k,
            "friction_per_fs": self.friction_per_fs,
        }

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "heat": self.heat, "rng": rng_state(self.rng)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.heat = float(state["heat"])
        self.rng = rng_from_state(state["rng"])


INTEGRATORS: dict[str, type[Integrator]] = {
    cls.__name__: cls for cls in (VelocityVerlet, CSVR, NoseHooverChain, LangevinBAOAB)
}


def integrator_from_state(backend: ForceBackend, state: dict[str, Any]) -> Integrator:
    """Rebuild an integrator from ``state_dict()`` (as saved in a checkpoint)."""
    cls = INTEGRATORS.get(state.get("integrator", ""))
    if cls is None:
        raise ValueError(f"unknown integrator {state.get('integrator')!r}")
    integ = cls.from_config(backend, state["config"])
    integ.load_state_dict(state)
    return integ
