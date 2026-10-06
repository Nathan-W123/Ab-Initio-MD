"""
Harmonic model potential with analytic gradients.

Not ab initio: an exactly solvable surface for testing integrators and
thermostats against closed-form results.

    E = E_0 + 1/2 dx^T H dx,    dx = (x - x_0).ravel()   (bohr, 3N vector)
    dE/dx = H dx

H is either a full symmetric positive semi-definite (3N, 3N) ``hessian`` or
diagonal, built from ``force_constants`` (hartree / bohr^2): a scalar, one per
atom (N,), or one per Cartesian coordinate (N, 3). ``reference_positions`` x_0
default to the origin, so atoms are tethered (no translation invariance; the
total momentum is not conserved). ``aimd run --backend harmonic`` centres the
wells on the starting geometry instead (aimd.cli), unless
``--backend-option reference_positions=[...]`` (bohr) is given.

With masses m the normal-mode angular frequencies are the square roots of the
eigenvalues of M^{-1/2} H M^{-1/2}. Useful closed forms for tests:
  - canonical position covariance: <dx dx^T> = kT H^{-1};
  - Velocity Verlet with step h conserves, for each mode q of frequency w,
    the shadow energy (p^2 + w^2 (1 - h^2 w^2 / 4) q^2) / 2 exactly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aimd.backends.base import ForceBackend, GradientResult
from aimd.backends.registry import register_backend


@register_backend
class HarmonicBackend(ForceBackend):
    name = "harmonic"
    description = "harmonic model surface (not ab initio; tests)"

    def __init__(
        self,
        symbols: Sequence[str],
        charge: int = 0,
        multiplicity: int = 1,
        force_constants: float | np.ndarray | None = None,
        hessian: np.ndarray | None = None,
        reference_positions: np.ndarray | None = None,
        energy_offset: float = 0.0,
    ) -> None:
        super().__init__(symbols, charge, multiplicity)
        n = len(self.symbols)
        if hessian is not None and force_constants is not None:
            raise ValueError("give either force_constants or hessian, not both")
        if hessian is not None:
            h = np.array(hessian, dtype=float)
            if h.shape != (3 * n, 3 * n):
                raise ValueError(f"hessian must have shape {(3 * n, 3 * n)}")
            if not np.allclose(h, h.T, rtol=0.0, atol=1e-12 * max(np.abs(h).max(), 1.0)):
                raise ValueError("hessian must be symmetric")
            self._hessian = 0.5 * (h + h.T)
            self._diag = None
        else:
            k = 0.5 if force_constants is None else force_constants
            k = np.asarray(k, dtype=float)
            if k.ndim == 0:
                kd = np.full((n, 3), float(k))
            elif k.shape == (n,):
                kd = np.repeat(k[:, None], 3, axis=1)
            elif k.shape == (n, 3):
                kd = k.copy()
            else:
                raise ValueError("force_constants must be a scalar, (N,) or (N, 3)")
            self._diag = kd.ravel()
            self._hessian = None
        if reference_positions is None:
            self.reference_positions = np.zeros((n, 3))
        else:
            self.reference_positions = np.array(reference_positions, dtype=float).reshape(n, 3)
        self.energy_offset = float(energy_offset)

    @property
    def hessian(self) -> np.ndarray:
        """The (3N, 3N) Hessian, hartree / bohr^2."""
        return np.diag(self._diag) if self._hessian is None else self._hessian

    def compute(self, positions: np.ndarray) -> GradientResult:
        dx = (np.asarray(positions, dtype=float).reshape(-1, 3)
              - self.reference_positions).ravel()
        g = self._diag * dx if self._diag is not None else self._hessian @ dx
        energy = self.energy_offset + 0.5 * float(dx @ g)
        return GradientResult(energy=energy, gradient=g.reshape(-1, 3))

    def normal_modes(self, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Angular frequencies (1 / au_time, ascending, shape (3N,)) and the
        orthonormal mass-weighted mode vectors (columns, shape (3N, 3N)).
        """
        m3 = np.repeat(np.asarray(masses, dtype=float), 3)
        mw = self.hessian / np.sqrt(np.outer(m3, m3))
        w2, modes = np.linalg.eigh(mw)
        return np.sqrt(np.clip(w2, 0.0, None)), modes
