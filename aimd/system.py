"""
MolecularSystem: the nuclear state propagated by the integrators.

Positions are bohr, velocities bohr / atomic-time-unit, masses electron masses.
Arrays are shaped (N, 3).

Degrees of freedom
------------------
The kinetic temperature is T = 2 K / (N_dof k_B). For an isolated molecule the
total linear momentum P and the angular momentum L about the centre of mass are
conserved by translation- and rotation-invariant forces, and Velocity Verlet
conserves both exactly (they are quadratic invariants of the form p^T C q;
Hairer, Lubich & Wanner, *Geometric Numerical Integration*, Sec. IV.2).
Once they are set to zero the velocities stay in a lower-dimensional subspace:

    N_dof = 3N - 3 [P = 0] - n_rot [L = 0],
    n_rot = 3 (nonlinear), 2 (linear), 0 (single atom).

``n_rot`` is fixed when the rotation is removed, so N_dof stays constant over a
run even if an initially linear molecule bends (a thermostat needs a fixed
N_dof). For a linear polyatomic that bends while L = 0 is enforced the third
component of L also becomes a constraint, so 3N - 5 slightly overcounts; this
is the usual convention and is irrelevant for diatomics (always linear).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aimd.elements import ATOMIC_MASSES_AMU, ATOMIC_NUMBERS, normalize_symbol
from aimd.units import AMU_TO_AU, ANG_TO_BOHR, KB_AU

# A molecule is treated as linear when its smallest principal moment of inertia
# is below this fraction of the largest (~1e-3 angstrom off-axis for a 1 angstrom
# bond). The same threshold makes the inertia-tensor solve a pseudo-inverse.
LINEAR_TOL = 1e-6


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
    # True once angular momentum about the COM has been removed.
    rotation_removed: bool = field(default=False)
    # Rotational DOF removed with the angular momentum (3, 2 if linear, 0 for a
    # single atom). Fixed at removal time; None means "from the current geometry".
    rotational_dof: int | None = field(default=None)

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
        self.com_removed = bool(self.com_removed)
        self.rotation_removed = bool(self.rotation_removed)
        if self.rotation_removed and self.rotational_dof is None:
            self.rotational_dof = self.n_rotations()
        if self.rotational_dof is not None:
            self.rotational_dof = int(self.rotational_dof)
            if self.rotational_dof not in (0, 2, 3):
                raise ValueError("rotational_dof must be 0, 2 or 3")

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
        """Kinetic degrees of freedom used to define the temperature (>= 1)."""
        dof = 3 * self.n_atoms
        if self.com_removed:
            dof -= 3
        if self.rotation_removed:
            rot = self.rotational_dof
            dof -= self.n_rotations() if rot is None else rot
        return max(dof, 1)

    def kinetic_energy(self) -> float:
        return 0.5 * float(np.sum(self.masses[:, None] * self.velocities**2))

    def temperature(self) -> float:
        """Instantaneous kinetic temperature in kelvin."""
        return 2.0 * self.kinetic_energy() / (self.n_dof * KB_AU)

    def center_of_mass(self) -> np.ndarray:
        return self.masses @ self.positions / self.masses.sum()

    def momentum(self) -> np.ndarray:
        """Total linear momentum (m_e bohr / au_time), shape (3,)."""
        return self.masses @ self.velocities

    def inertia_tensor(self) -> np.ndarray:
        """Inertia tensor about the centre of mass (m_e bohr^2), shape (3, 3)."""
        r = self.positions - self.center_of_mass()
        mr2 = float(np.sum(self.masses * np.einsum("ij,ij->i", r, r)))
        return mr2 * np.eye(3) - (self.masses[:, None] * r).T @ r

    def angular_momentum(self) -> np.ndarray:
        """Angular momentum about the centre of mass, shape (3,).

        Equal to sum_i m_i (r_i - R) x (v_i - V): the COM velocity drops out
        because sum_i m_i (r_i - R) = 0.
        """
        r = self.positions - self.center_of_mass()
        return np.sum(self.masses[:, None] * np.cross(r, self.velocities), axis=0)

    def is_linear(self, tol: float = LINEAR_TOL) -> bool:
        """True for one or two atoms, or collinear atoms (to within ``tol``)."""
        if self.n_atoms <= 2:
            return True
        moments = np.linalg.eigvalsh(self.inertia_tensor())
        return bool(moments[0] <= tol * moments[-1])

    def n_rotations(self) -> int:
        """Rotational DOF of the current geometry: 0 (atom), 2 (linear) or 3."""
        if self.n_atoms == 1:
            return 0
        return 2 if self.is_linear() else 3

    # ── Velocity manipulation ─────────────────────────────────────────────────

    def remove_com_motion(self) -> None:
        p = self.masses @ self.velocities
        self.velocities -= p / self.masses.sum()
        self.com_removed = True

    def remove_angular_momentum(self) -> None:
        """
        Remove the rigid-body rotation about the centre of mass.

        Solves I w = L for the angular velocity and subtracts w x (r_i - R)
        from every velocity. Positions and the total linear momentum are
        unchanged (sum_i m_i w x (r_i - R) = 0). The inertia tensor is singular
        for linear molecules (zero moment about the axis, where L has no
        component either) and zero for a single atom, so it is inverted only on
        the subspace of non-negligible principal moments.
        """
        r = self.positions - self.center_of_mass()
        ang = np.sum(self.masses[:, None] * np.cross(r, self.velocities), axis=0)
        moments, axes = np.linalg.eigh(self.inertia_tensor())
        keep = moments > LINEAR_TOL * moments[-1] if moments[-1] > 0.0 else moments > 0.0
        inv = np.zeros(3)
        inv[keep] = 1.0 / moments[keep]
        omega = axes @ (inv * (axes.T @ ang))
        self.velocities -= np.cross(omega, r)
        self.rotation_removed = True
        self.rotational_dof = self.n_rotations()

    def initialize_velocities(
        self,
        temperature_k: float,
        rng: np.random.Generator | int | None = None,
        exact: bool = True,
        remove_rotation: bool = False,
    ) -> None:
        """
        Draw Maxwell-Boltzmann velocities at ``temperature_k``.

        Centre-of-mass motion is removed, and with ``remove_rotation=True``
        also the angular momentum (for isolated molecules). With ``exact=True``
        the velocities are then rescaled so the instantaneous temperature,
        computed with the reduced ``n_dof``, equals the target. A uniform
        rescaling keeps P = 0 and L = 0.
        """
        rng = np.random.default_rng(rng)
        if temperature_k <= 0.0:
            self.velocities = np.zeros_like(self.positions)
            self.com_removed = True
            self.rotation_removed = bool(remove_rotation)
            self.rotational_dof = self.n_rotations() if remove_rotation else None
            return
        sigma = np.sqrt(KB_AU * temperature_k / self.masses)
        self.velocities = rng.normal(size=self.positions.shape) * sigma[:, None]
        self.remove_com_motion()
        if remove_rotation:
            self.remove_angular_momentum()
        else:
            self.rotation_removed = False
            self.rotational_dof = None
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
            rotation_removed=self.rotation_removed,
            rotational_dof=self.rotational_dof,
        )
