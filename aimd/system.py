"""
MolecularSystem: the nuclear state propagated by the integrators.

Positions are bohr, velocities bohr / atomic-time-unit, masses electron masses.
Arrays are shaped (N, 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aimd.elements import ATOMIC_MASSES_AMU, ATOMIC_NUMBERS, normalize_symbol
from aimd.units import AMU_TO_AU, ANG_TO_BOHR, KB_AU


@dataclass
class MolecularSystem:
    symbols: list[str]
    positions: np.ndarray                       # bohr, (N, 3)
    velocities: np.ndarray | None = None        # bohr / au_time, (N, 3)
    masses: np.ndarray | None = None            # m_e, (N,)
    charge: int = 0
    multiplicity: int = 1
    # True once centre-of-mass motion has been removed; used for the DOF count.
    com_removed: bool = field(default=False)

    def __post_init__(self) -> None:
        self.symbols = [normalize_symbol(s) for s in self.symbols]
        self.positions = np.array(self.positions, dtype=float).reshape(-1, 3)
        n = len(self.symbols)
        if self.positions.shape[0] != n:
            raise ValueError(
                f"{n} symbols but {self.positions.shape[0]} position rows"
            )
        if self.velocities is None:
            self.velocities = np.zeros((n, 3))
        else:
            self.velocities = np.array(self.velocities, dtype=float).reshape(n, 3)
        if self.masses is None:
            self.masses = np.array(
                [ATOMIC_MASSES_AMU[s] * AMU_TO_AU for s in self.symbols]
            )
        else:
            self.masses = np.array(self.masses, dtype=float).reshape(n)

    # ── Construction ──────────────────────────────────────────────────────────

    @classmethod
    def from_xyz(
        cls, path: str | Path, charge: int = 0, multiplicity: int = 1
    ) -> "MolecularSystem":
        """Read the first frame of an XYZ file (coordinates in angstrom)."""
        lines = Path(path).read_text().splitlines()
        return cls.from_xyz_string("\n".join(lines), charge, multiplicity)

    @classmethod
    def from_xyz_string(
        cls, text: str, charge: int = 0, multiplicity: int = 1
    ) -> "MolecularSystem":
        lines = [ln for ln in text.splitlines()]
        try:
            n = int(lines[0].split()[0])
        except (IndexError, ValueError) as e:
            raise ValueError("XYZ: first line must be the atom count") from e
        body = lines[2 : 2 + n]
        if len(body) != n:
            raise ValueError(f"XYZ: expected {n} atom lines, found {len(body)}")
        symbols, coords = [], []
        for ln in body:
            parts = ln.split()
            if len(parts) < 4:
                raise ValueError(f"XYZ: malformed atom line {ln!r}")
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
        return cls(
            symbols=symbols,
            positions=np.array(coords) * ANG_TO_BOHR,
            charge=charge,
            multiplicity=multiplicity,
        )

    # ── Derived quantities ────────────────────────────────────────────────────

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)

    @property
    def atomic_numbers(self) -> np.ndarray:
        return np.array([ATOMIC_NUMBERS[s] for s in self.symbols])

    @property
    def n_dof(self) -> int:
        """Kinetic degrees of freedom used to define the temperature."""
        dof = 3 * self.n_atoms - (3 if self.com_removed else 0)
        return max(dof, 1)

    def kinetic_energy(self) -> float:
        return 0.5 * float(np.sum(self.masses[:, None] * self.velocities**2))

    def temperature(self) -> float:
        """Instantaneous kinetic temperature in kelvin."""
        return 2.0 * self.kinetic_energy() / (self.n_dof * KB_AU)

    def center_of_mass(self) -> np.ndarray:
        return self.masses @ self.positions / self.masses.sum()

    # ── Velocity manipulation ─────────────────────────────────────────────────

    def remove_com_motion(self) -> None:
        p = self.masses @ self.velocities
        self.velocities -= p / self.masses.sum()
        self.com_removed = True

    def initialize_velocities(
        self,
        temperature_k: float,
        rng: np.random.Generator | int | None = None,
        exact: bool = True,
    ) -> None:
        """
        Draw Maxwell-Boltzmann velocities at ``temperature_k``.

        Centre-of-mass motion is removed. With ``exact=True`` the velocities are
        then rescaled so the instantaneous temperature equals the target.
        """
        rng = np.random.default_rng(rng)
        if temperature_k <= 0.0:
            self.velocities = np.zeros_like(self.positions)
            self.com_removed = True
            return
        sigma = np.sqrt(KB_AU * temperature_k / self.masses)
        self.velocities = rng.normal(size=self.positions.shape) * sigma[:, None]
        self.remove_com_motion()
        if exact:
            t_now = self.temperature()
            if t_now > 0.0:
                self.velocities *= np.sqrt(temperature_k / t_now)

    def copy(self) -> "MolecularSystem":
        return MolecularSystem(
            symbols=list(self.symbols),
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            masses=self.masses.copy(),
            charge=self.charge,
            multiplicity=self.multiplicity,
            com_removed=self.com_removed,
        )
