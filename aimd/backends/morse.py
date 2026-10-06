"""
Pairwise Morse potential with analytic gradients.

Not ab initio: this is a cheap, exactly-differentiable model surface used to
test the integrators, thermostats and driver without an electronic-structure
code, and to give new backends something to compare plumbing against.

    E = sum_{i<j} D * (1 - exp(-a (r_ij - r_e)))^2 - D

Defaults are the H2 ground state (D_e = 0.1745 Eh, a = 1.0282 / bohr,
r_e = 1.4011 bohr). Every pair uses the same parameters.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from aimd.backends.base import ForceBackend, GradientResult
from aimd.backends.registry import register_backend


@register_backend
class MorseBackend(ForceBackend):
    name = "morse"
    description = "pairwise Morse model surface (not ab initio; tests, plumbing)"

    def __init__(
        self,
        symbols: Sequence[str],
        charge: int = 0,
        multiplicity: int = 1,
        depth: float = 0.1745,
        alpha: float = 1.0282,
        r_eq: float = 1.4011,
    ) -> None:
        super().__init__(symbols, charge, multiplicity)
        self.depth = float(depth)
        self.alpha = float(alpha)
        self.r_eq = float(r_eq)

    def compute(self, positions: np.ndarray) -> GradientResult:
        x = np.asarray(positions, dtype=float).reshape(-1, 3)
        n = x.shape[0]
        i, j = np.triu_indices(n, k=1)
        d = x[i] - x[j]                              # (P, 3)
        r = np.linalg.norm(d, axis=1)
        ex = np.exp(-self.alpha * (r - self.r_eq))
        energy = float(np.sum(self.depth * ((1.0 - ex) ** 2 - 1.0)))

        dE_dr = 2.0 * self.depth * self.alpha * (1.0 - ex) * ex
        pair_grad = (dE_dr / r)[:, None] * d        # dE/dx_i for each pair
        grad = np.zeros_like(x)
        np.add.at(grad, i, pair_grad)
        np.add.at(grad, j, -pair_grad)
        return GradientResult(energy=energy, gradient=grad)
