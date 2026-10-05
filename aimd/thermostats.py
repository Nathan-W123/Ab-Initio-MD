"""
Velocity thermostats applied as operators around a Velocity Verlet step.

A thermostat acts only on the velocities. :class:`aimd.integrators.VelocityVerlet`
calls ``begin_step`` before the first velocity half-kick and ``end_step`` after
the second one, which gives the symmetric Trotter splitting

    exp(iL_T dt/2) exp(iL_B dt/2) exp(iL_A dt) exp(iL_B dt/2) exp(iL_T dt/2)

for CSVR and Nose-Hoover chains. (Berendsen keeps its original single rescale
at the end of the step.) Each thermostat also reports :meth:`Thermostat.energy`,
its contribution to the conserved quantity

    E_cons = E_pot + E_kin + thermostat.energy().

  BerendsenThermostat        weak coupling (Berendsen et al., J. Chem. Phys.
                             81, 3684 (1984)); does not sample a known ensemble
  CSVRThermostat             canonical sampling through velocity rescaling
                             (Bussi, Donadio & Parrinello, J. Chem. Phys. 126,
                             014101 (2007)) with the exact kinetic-energy
                             resampling of their appendix
  NoseHooverChainThermostat  Nose-Hoover chains (Martyna, Klein & Tuckerman,
                             J. Chem. Phys. 97, 2635 (1992)) integrated with the
                             MTK scheme (Martyna, Tuckerman, Tobias & Klein,
                             Mol. Phys. 87, 1117 (1996)): multiple time steps
                             and Suzuki-Yoshida factorisation

For the rescaling thermostats E_cons is the "effective energy" of Bussi et al.:
the total energy minus the kinetic energy the thermostat has added so far. It
is constant up to integration error, so its drift measures the accuracy of the
run just as the total energy does in NVE.

All quantities are in atomic units; constructor arguments take kelvin and fs.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numba
import numpy as np

from aimd.system import MolecularSystem
from aimd.units import FS_TO_AU_TIME, KB_AU

# Bit generators whose state may be restored from a checkpoint (by name, so a
# checkpoint never makes us instantiate an arbitrary class).
_BIT_GENERATORS = ("PCG64", "PCG64DXSM", "MT19937", "Philox", "SFC64")


def rng_state(rng: np.random.Generator) -> dict[str, Any]:
    """Bit-generator state of ``rng`` (a plain dict; see ``rng_from_state``)."""
    return rng.bit_generator.state


def rng_from_state(state: dict[str, Any]) -> np.random.Generator:
    """New ``Generator`` continuing exactly from a saved bit-generator state."""
    name = state.get("bit_generator")
    if name not in _BIT_GENERATORS:
        raise ValueError(f"unsupported bit generator in saved state: {name!r}")
    bitgen = getattr(np.random, name)()
    st = dict(state)
    if name == "MT19937":
        inner = dict(st["state"])
        inner["key"] = np.asarray(inner["key"], dtype=np.uint32)
        st["state"] = inner
    elif name == "Philox":
        inner = dict(st["state"])
        inner["counter"] = np.asarray(inner["counter"], dtype=np.uint64)
        inner["key"] = np.asarray(inner["key"], dtype=np.uint64)
        st["state"] = inner
        st["buffer"] = np.asarray(st["buffer"], dtype=np.uint64)
    bitgen.state = st
    return np.random.Generator(bitgen)


class Thermostat(ABC):
    """A velocity operator applied at the start and end of each MD step."""

    def __init__(self, temperature_k: float) -> None:
        self.set_temperature(temperature_k)

    def set_temperature(self, temperature_k: float) -> None:
        """Change the target temperature (e.g. for annealing between runs)."""
        if not np.isfinite(temperature_k) or temperature_k < 0.0:
            raise ValueError("temperature_k must be finite and non-negative")
        self.temperature_k = float(temperature_k)
        self.kT = KB_AU * self.temperature_k

    def initialize(self, system: MolecularSystem) -> None:
        """Bind to ``system`` before the first step (default: nothing)."""

    def begin_step(self, system: MolecularSystem, dt: float) -> None:
        """Applied before the first velocity half-kick of a step of length dt."""

    def end_step(self, system: MolecularSystem, dt: float) -> None:
        """Applied after the second velocity half-kick of a step of length dt."""

    @abstractmethod
    def energy(self) -> float:
        """Contribution to the conserved quantity (hartree)."""

    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Constructor keyword arguments (JSON-serialisable)."""

    def state_dict(self) -> dict[str, Any]:
        return {"type": type(self).__name__, "config": self.config()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("type") != type(self).__name__:
            raise ValueError(
                f"cannot load {state.get('type')!r} state into {type(self).__name__}"
            )


class _RescalingThermostat(Thermostat):
    """Shared bookkeeping: E_cons = E_pot + E_kin - (kinetic energy added)."""

    def __init__(self, temperature_k: float) -> None:
        super().__init__(temperature_k)
        self.heat = 0.0      # kinetic energy added by the thermostat so far, Eh

    def energy(self) -> float:
        return -self.heat

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "heat": self.heat}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.heat = float(state["heat"])


class BerendsenThermostat(_RescalingThermostat):
    """
    Weak coupling: after each step scale v by lambda with
    lambda^2 = 1 + (dt / tau) (T0 / T - 1). Useful for equilibration only.
    """

    def __init__(self, temperature_k: float, tau_fs: float) -> None:
        super().__init__(temperature_k)
        if tau_fs <= 0.0:
            raise ValueError("tau_fs must be positive")
        self.tau_fs = float(tau_fs)
        self.tau = self.tau_fs * FS_TO_AU_TIME

    def end_step(self, system: MolecularSystem, dt: float) -> None:
        t_now = system.temperature()
        if t_now > 0.0:
            lam2 = max(1.0 + (dt / self.tau) * (self.temperature_k / t_now - 1.0), 0.0)
            ekin = system.kinetic_energy()
            system.velocities *= np.sqrt(lam2)
            self.heat += ekin * (lam2 - 1.0)

    def config(self) -> dict[str, Any]:
        return {"temperature_k": self.temperature_k, "tau_fs": self.tau_fs}


class CSVRThermostat(_RescalingThermostat):
    """
    Stochastic velocity rescaling (Bussi, Donadio & Parrinello 2007).

    The kinetic energy K of the N_f thermostatted DOF follows
        dK = (K_bar - K) dt / tau + 2 sqrt(K K_bar / N_f) dW / sqrt(tau),
    K_bar = N_f kT / 2, whose stationary law is the canonical Gamma(N_f/2, kT).
    Over a time t the process is solved exactly (appendix of the paper): with
    c = exp(-t / tau), R_1 ~ N(0, 1) and S ~ chi^2_{N_f - 1} (drawn as
    2 * Gamma((N_f - 1)/2), i.e. the sum of N_f - 1 squared Gaussians),

        alpha^2 = c + (1 - c) (R_1^2 + S) K_bar / (N_f K)
                    + 2 R_1 sqrt(c (1 - c) K_bar / (N_f K)),

    and v -> alpha v with sign(alpha) = sign(sqrt(c) + R_1 sqrt((1-c) K_bar / (N_f K)))
    (the component along the old velocity direction), which keeps the move
    time-reversible. It is applied for dt/2 before and after each Verlet step.

    N_f is ``system.n_dof``; a global rescaling keeps P = 0 and L = 0, so the
    COM / rotation bookkeeping of the system stays valid.
    """

    def __init__(
        self,
        temperature_k: float,
        tau_fs: float = 100.0,
        rng: np.random.Generator | int | None = None,
    ) -> None:
        super().__init__(temperature_k)
        if not tau_fs > 0.0:
            raise ValueError("tau_fs must be positive")
        self.tau_fs = float(tau_fs)
        self.tau = self.tau_fs * FS_TO_AU_TIME
        self.rng = np.random.default_rng(rng)

    def rescale(self, system: MolecularSystem, t: float) -> float:
        """Propagate the CSVR process for time ``t`` (au); returns alpha."""
        ekin = system.kinetic_energy()
        if ekin <= 0.0:
            return 1.0        # no velocity direction to rescale along
        n_f = system.n_dof
        c = math.exp(-t / self.tau)
        q = (1.0 - c) * 0.5 * self.kT / ekin      # (1 - c) K_bar / (N_f K)
        r1 = float(self.rng.standard_normal())
        s = 2.0 * float(self.rng.standard_gamma(0.5 * (n_f - 1))) if n_f > 1 else 0.0
        along = math.sqrt(c) + r1 * math.sqrt(q)
        alpha2 = along * along + q * s
        alpha = math.copysign(math.sqrt(alpha2), along)
        system.velocities *= alpha
        self.heat += ekin * (alpha2 - 1.0)
        return alpha

    def begin_step(self, system: MolecularSystem, dt: float) -> None:
        self.rescale(system, 0.5 * dt)

    def end_step(self, system: MolecularSystem, dt: float) -> None:
        self.rescale(system, 0.5 * dt)

    def config(self) -> dict[str, Any]:
        return {"temperature_k": self.temperature_k, "tau_fs": self.tau_fs}

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "rng": rng_state(self.rng)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.rng = rng_from_state(state["rng"])


# Suzuki-Yoshida weights (sum to 1) for the NHC factorisation: 3 and 5 weights
# give 4th order, 7 weights 6th order (Yoshida, Phys. Lett. A 150, 262 (1990),
# solution A, in the order tabulated by MTK96).
def _sy_weights(n: int) -> tuple[float, ...]:
    if n == 1:
        return (1.0,)
    if n == 3:
        w = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        return (w, 1.0 - 2.0 * w, w)
    if n == 5:
        w = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        return (w, w, 1.0 - 4.0 * w, w, w)
    if n == 7:
        w1, w2, w3 = 0.784513610477560, 0.235573213359357, -1.17767998417887
        w4 = 1.0 - 2.0 * (w1 + w2 + w3)
        return (w1, w2, w3, w4, w3, w2, w1)
    raise ValueError("n_suzuki_yoshida must be 1, 3, 5 or 7")


@numba.njit(cache=True)
def nhc_propagate(
    xi: np.ndarray,
    vxi: np.ndarray,
    q: np.ndarray,
    kT: float,
    n_dof: int,
    ekin2: float,
    t: float,
    weights: np.ndarray,
    n_mts: int,
) -> float:
    """
    MTK propagator for exp(iL_NHC t), the thermostat part of the Liouvillian.

    Equations of motion (v_xi_j = p_xi_j / Q_j, M = len(xi)):
        dv/dt       = -v_xi_1 v                       (particle velocities)
        d xi_j/dt   = v_xi_j
        d v_xi_1/dt = (2K - N_f kT)/Q_1 - v_xi_1 v_xi_2
        d v_xi_j/dt = (Q_{j-1} v_xi_{j-1}^2 - kT)/Q_j - v_xi_j v_xi_{j+1}
        d v_xi_M/dt = (Q_{M-1} v_xi_{M-1}^2 - kT)/Q_M
    The time t is cut into n_mts * len(weights) sub-steps h = w t / n_mts; each
    is the palindrome of MTK96 (their Fig. 2 / Tuckerman's NHCINT): half-updates
    of v_xi from the chain end down, a full scaling of v and update of xi, and
    half-updates back up. A half-update of v_xi_j over h/2 with v_xi_{j+1}
    frozen is exp(-h/4 v_xi_{j+1}) [v_xi_j += h/2 G_j] exp(-h/4 v_xi_{j+1}).
    ``xi`` and ``vxi`` (float64 arrays) are updated in place; ``ekin2`` is 2K
    before the call. Returns the factor by which v must be scaled.
    """
    m = xi.shape[0]
    g = np.empty(m)
    g[0] = (ekin2 - n_dof * kT) / q[0]
    for j in range(1, m):
        g[j] = (q[j - 1] * vxi[j - 1] ** 2 - kT) / q[j]
    scale = 1.0
    for _ in range(n_mts):
        for w in weights:
            h = w * t / n_mts
            h2 = 0.5 * h
            h4 = 0.25 * h
            vxi[m - 1] += h2 * g[m - 1]
            for j in range(m - 2, -1, -1):
                a = math.exp(-h4 * vxi[j + 1])
                vxi[j] = vxi[j] * a * a + h2 * g[j] * a
            a = math.exp(-h * vxi[0])
            scale *= a
            g[0] = (scale * scale * ekin2 - n_dof * kT) / q[0]
            for j in range(m):
                xi[j] += h * vxi[j]
            for j in range(m - 1):
                a = math.exp(-h4 * vxi[j + 1])
                vxi[j] = vxi[j] * a * a + h2 * g[j] * a
                g[j + 1] = (q[j] * vxi[j] ** 2 - kT) / q[j + 1]
            vxi[m - 1] += h2 * g[m - 1]
    return scale


class NoseHooverChainThermostat(Thermostat):
    """
    One Nose-Hoover chain of length M coupled to all N_f kinetic DOF.

    Thermostat masses from a characteristic period P (fs), omega = 2 pi / P:
        Q_1 = N_f kT / omega^2,   Q_j = kT / omega^2  (j >= 2)   (MKT92).
    The conserved extended energy is
        H' = E_pot + K + sum_j Q_j v_xi_j^2 / 2 + N_f kT xi_1 + kT sum_{j>=2} xi_j.
    N_f is taken from ``system.n_dof`` when the thermostat is first used and
    then kept fixed (it is part of H').

    The chain is integrated with ``n_mts`` sub-steps of ``n_suzuki_yoshida``
    (1, 3, 5 or 7) weights per half step. The default (1, 3) is accurate for
    periods of >~ 20 timesteps; tighter coupling needs a larger ``n_mts``.
    A single global chain relies on the system itself to share energy among
    its modes (nearly harmonic systems equipartition slowly).
    """

    def __init__(
        self,
        temperature_k: float,
        period_fs: float = 100.0,
        chain_length: int = 3,
        n_mts: int = 1,
        n_suzuki_yoshida: int = 3,
    ) -> None:
        self.n_dof: int | None = None
        super().__init__(temperature_k)
        if not period_fs > 0.0:
            raise ValueError("period_fs must be positive")
        if chain_length < 1 or n_mts < 1:
            raise ValueError("chain_length and n_mts must be >= 1")
        self.period_fs = float(period_fs)
        self.chain_length = int(chain_length)
        self.n_mts = int(n_mts)
        self.n_suzuki_yoshida = int(n_suzuki_yoshida)
        self.weights = np.array(_sy_weights(self.n_suzuki_yoshida))
        self.omega = 2.0 * math.pi / (self.period_fs * FS_TO_AU_TIME)
        self._q: np.ndarray | None = None           # cached masses once N_f is known
        self.xi = np.zeros(self.chain_length)       # thermostat positions
        self.vxi = np.zeros(self.chain_length)      # thermostat velocities, 1/au_time

    @property
    def masses(self) -> np.ndarray:
        if self.n_dof is None:
            raise RuntimeError("thermostat not initialised (N_f unknown)")
        q = np.full(self.chain_length, self.kT / self.omega**2)
        q[0] *= self.n_dof
        return q

    def set_temperature(self, temperature_k: float) -> None:
        """Change the target; the masses Q follow kT (H' is not continuous)."""
        if not temperature_k > 0.0:
            raise ValueError("Nose-Hoover chains need temperature_k > 0")
        super().set_temperature(temperature_k)
        if self.n_dof is not None:
            self._q = self.masses

    def initialize(self, system: MolecularSystem) -> None:
        if self.n_dof is None:
            self.n_dof = system.n_dof
            self._q = self.masses

    def propagate(self, system: MolecularSystem, t: float) -> None:
        """Apply exp(iL_NHC t) to the system velocities and chain."""
        self.initialize(system)
        scale = nhc_propagate(
            self.xi, self.vxi, self._q, self.kT, self.n_dof,
            2.0 * system.kinetic_energy(), t, self.weights, self.n_mts,
        )
        if not (math.isfinite(scale) and scale > 0.0 and np.all(np.isfinite(self.vxi))):
            raise FloatingPointError(
                "Nose-Hoover chain integration diverged; use a longer period_fs "
                "(>~ 20 timesteps) or more substeps (n_mts)"
            )
        system.velocities *= scale

    def begin_step(self, system: MolecularSystem, dt: float) -> None:
        self.propagate(system, 0.5 * dt)

    def end_step(self, system: MolecularSystem, dt: float) -> None:
        self.propagate(system, 0.5 * dt)

    def energy(self) -> float:
        if self.n_dof is None:
            return 0.0
        return float(
            0.5 * np.sum(self._q * self.vxi**2)
            + self.n_dof * self.kT * self.xi[0]
            + self.kT * np.sum(self.xi[1:])
        )

    def config(self) -> dict[str, Any]:
        return {
            "temperature_k": self.temperature_k,
            "period_fs": self.period_fs,
            "chain_length": self.chain_length,
            "n_mts": self.n_mts,
            "n_suzuki_yoshida": self.n_suzuki_yoshida,
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            **super().state_dict(),
            "n_dof": self.n_dof,
            "xi": self.xi.copy(),
            "vxi": self.vxi.copy(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        xi = np.asarray(state["xi"], dtype=float)
        vxi = np.asarray(state["vxi"], dtype=float)
        if xi.shape != (self.chain_length,) or vxi.shape != (self.chain_length,):
            raise ValueError(
                f"saved chain shape {xi.shape} != ({self.chain_length},)"
            )
        self.n_dof = None if state["n_dof"] is None else int(state["n_dof"])
        self._q = None if self.n_dof is None else self.masses
        self.xi = xi.copy()
        self.vxi = vxi.copy()


THERMOSTATS: dict[str, type[Thermostat]] = {
    cls.__name__: cls
    for cls in (BerendsenThermostat, CSVRThermostat, NoseHooverChainThermostat)
}


def thermostat_from_config(kind: str, config: dict[str, Any]) -> Thermostat:
    """Construct a thermostat by class name from its ``config()``."""
    cls = THERMOSTATS.get(kind)
    if cls is None:
        raise ValueError(f"unknown thermostat type {kind!r}")
    return cls(**config)


def thermostat_from_state(state: dict[str, Any]) -> Thermostat:
    """Rebuild a thermostat (config + dynamical state) from ``state_dict()``."""
    thermo = thermostat_from_config(state.get("type", ""), state["config"])
    thermo.load_state_dict(state)
    return thermo
