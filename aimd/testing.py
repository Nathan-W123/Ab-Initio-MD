"""
Validation helpers for backends.

``finite_difference_gradient`` is the yardstick for any new analytic gradient
(e.g. the RHF gradient to be ported from HF-SCF-Engine): it should agree with
the analytic result to ~1e-6 Eh/bohr for a well-converged SCF.
"""

from __future__ import annotations

import numpy as np

from aimd.backends.base import ForceBackend


def finite_difference_gradient(
    backend: ForceBackend, positions: np.ndarray, step: float = 1e-4
) -> np.ndarray:
    """Central-difference gradient (hartree / bohr), 6N energy evaluations."""
    x0 = np.asarray(positions, dtype=float).reshape(-1, 3)
    grad = np.zeros_like(x0)
    for a in range(x0.shape[0]):
        for k in range(3):
            xp = x0.copy()
            xm = x0.copy()
            xp[a, k] += step
            xm[a, k] -= step
            ep = backend.compute(xp).energy
            em = backend.compute(xm).energy
            grad[a, k] = (ep - em) / (2.0 * step)
    return grad


def max_gradient_error(
    backend: ForceBackend, positions: np.ndarray, step: float = 1e-4
) -> float:
    """Max abs difference between analytic and finite-difference gradients."""
    analytic = backend.compute(positions).gradient
    numeric = finite_difference_gradient(backend, positions, step)
    return float(np.max(np.abs(analytic - numeric)))
