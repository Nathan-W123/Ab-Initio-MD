"""
Extended-Lagrangian Born-Oppenheimer MD (XL-BOMD): the auxiliary density.

Why
---
In Born-Oppenheimer MD the SCF at every step starts from a guess, usually the
density of the previous step(s). An SCF that is not converged to machine
precision leaves an error that depends on that guess, and a guess extrapolated
from the past is not symmetric under time reversal. The force error is then
systematic, and the total energy drifts, often by orders of magnitude more than
the Verlet integration error, unless the SCF is converged very tightly.

XL-BOMD (Niklasson, Phys. Rev. Lett. 100, 123004 (2008)) instead propagates an
auxiliary density P with a time-reversible Verlet equation, harmonically
coupled to the SCF density D(R):

    P(t+dt) = 2 P(t) - P(t-dt) + kappa (D(t) - P(t)),     kappa = dt^2 omega^2,

and uses P(t+dt) as the SCF guess at R(t+dt). P follows D closely, and because
the guess is generated time-reversibly, the residual SCF error fluctuates
instead of accumulating: no systematic drift even with loose SCF convergence.

Dissipation
-----------
Round-off and the SCF error itself excite the auxiliary harmonic oscillator,
whose amplitude would then grow slowly. Niklasson, Steneteg, Odell, Bock,
Challacombe, Tymczak, Holmstrom, Zheng & Weber, J. Chem. Phys. 130, 214109
(2009) add a weak dissipation built from the last K+1 densities:

    P(t+dt) = 2 P(t) - P(t-dt) + kappa (D(t) - P(t))
              + alpha sum_{k=0..K} c_k P(t - k dt).

Expanding P(t - k dt) in powers of dt, the odd powers are the part that is
odd under time reversal. sum_k c_k = sum_k k c_k = 0 (a density that is
constant or linear in time is left alone), and the odd moments
sum_k c_k k^j vanish for odd j < 2K - 3, so the dissipation breaks
time-reversal symmetry only at order dt^(2K-3) (dt^7 for K = 5). Table I of
the 2009 paper (``XL_COEFFICIENTS``) gives c_k with the coupling kappa and the
dissipation strength alpha determined there for K = 3..9; K = 5 is the usual
choice. Higher K means weaker damping: the spectral radius of the
static-D recursion is 0.63 for K = 3, 0.91 for K = 5 and 0.99 for K = 9.

Practical note
--------------
XL-BOMD removes the systematic drift, not the noise of an SCF stopped by a
threshold: when the cycle count changes from step to step, so does the force
error. Measured, water, 0.5 fs, 400 K, 100 fs, conv_tol 1e-5, straight-line
drift rate of E_tot in Eh/ps (tight SCF: ~1e-5): HF/6-31G, BOMD -3.9e-3,
XL (K=5) 7e-6; PBE/6-31G, BOMD +1.3e-3, XL 1e-4 (K=5), -2.5e-4 (K=7). With
exactly 3 SCF cycles per step (tight conv_tol, max_cycle=3; such steps are
reported as unconverged), PBE gives BOMD -3.9e-3, XL -1.4e-4 (K=5),
-9e-6 (K=7). A fixed, small number of cycles is the usual practice.
For a threshold-stopped DFT SCF XL-BOMD is not guaranteed to help at all:
PBE/STO-3G, 500 K, conv_tol 1e-5, two seeds x grid levels 1 and 3 gave
XL (K=5) +1.8e-3 .. +3.0e-3 Eh/ps against BOMD +5.7e-4 .. +1.6e-3, while
with exactly 3 cycles XL gave -1.3e-4 .. -6.8e-4 against BOMD -3.1e-2.

Units and representation
------------------------
P and D are the backend's AO density matrices (restricted: total density
(nao, nao); unrestricted: (2, nao, nao)); kappa and alpha are dimensionless.
The AO basis moves with the atoms, so the AO matrix itself is a smooth
function of time and is propagated directly.
The history is initialised with the first SCF density: P(t0 - k dt) = D(t0).
The integrator that uses this (aimd.integrators.XLBOMD) is velocity Verlet for
the nuclei with the backend's forces at the new geometry, computed from the
guess P(t+dt).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# K: (kappa, alpha, (c_0, ..., c_K)), Niklasson et al., J. Chem. Phys. 130,
# 214109 (2009), Table I.
XL_COEFFICIENTS: dict[int, tuple[float, float, tuple[int, ...]]] = {
    3: (1.69, 150e-3, (-2, 3, 0, -1)),
    4: (1.75, 57e-3, (-3, 6, -2, -2, 1)),
    5: (1.82, 18e-3, (-6, 14, -8, -3, 4, -1)),
    6: (1.84, 5.5e-3, (-14, 36, -27, -2, 12, -6, 1)),
    7: (1.86, 1.6e-3, (-36, 99, -88, 11, 32, -25, 8, -1)),
    8: (1.88, 0.44e-3, (-99, 286, -286, 78, 78, -90, 42, -10, 1)),
    9: (1.89, 0.12e-3, (-286, 858, -936, 364, 168, -300, 184, -63, 12, -1)),
}


def xl_coefficients(k: int) -> tuple[float, float, np.ndarray]:
    """(kappa, alpha, c) for dissipation order ``k`` (3..9); c has K+1 entries."""
    if k not in XL_COEFFICIENTS:
        raise ValueError(
            f"XL-BOMD dissipation order K must be one of {sorted(XL_COEFFICIENTS)}, got {k!r}"
        )
    kappa, alpha, c = XL_COEFFICIENTS[k]
    return kappa, alpha, np.array(c, dtype=float)


class AuxiliaryDensity:
    """
    History [P(t), P(t-dt), ..., P(t-K dt)] of the auxiliary density and its
    dissipative Verlet update (see the module docstring).
    """

    def __init__(self, k: int = 5) -> None:
        self.kappa, self.alpha, self.c = xl_coefficients(k)   # validates k
        self.k = int(k)
        self.history: list[np.ndarray] | None = None

    @property
    def initialized(self) -> bool:
        return self.history is not None

    @property
    def current(self) -> np.ndarray:
        """P(t), the guess that produced the latest SCF density."""
        if self.history is None:
            raise RuntimeError("auxiliary density not initialised")
        return self.history[0]

    def reset(self, density: np.ndarray) -> None:
        """Start from rest: P(t - k dt) = D for k = 0..K."""
        d = np.array(density, dtype=float)
        self.history = [d.copy() for _ in range(self.k + 1)]

    def propagate(self, density: np.ndarray) -> np.ndarray:
        """
        Given the SCF density D(t) at the current geometry, return
        P(t+dt) and shift the history by one step.
        """
        if self.history is None:
            raise RuntimeError("auxiliary density not initialised")
        d = np.asarray(density, dtype=float)
        hist = self.history
        if d.shape != hist[0].shape:
            raise ValueError(
                f"SCF density shape {d.shape} != auxiliary density shape {hist[0].shape}"
            )
        dissipation = self.c[0] * hist[0]
        for ck, pk in zip(self.c[1:], hist[1:]):
            dissipation = dissipation + ck * pk
        p_next = (2.0 * hist[0] - hist[1] + self.kappa * (d - hist[0])
                  + self.alpha * dissipation)
        self.history = [p_next] + hist[:-1]
        return p_next.copy()

    # ── Restart state ────────────────────────────────────────────────────────

    def state_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "history": None if self.history is None else np.stack(self.history),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state["k"]) != self.k:
            raise ValueError(
                f"checkpoint XL-BOMD order K={state['k']} != integrator's K={self.k}"
            )
        hist = state["history"]
        if hist is None:
            self.history = None
            return
        hist = np.array(hist, dtype=float)
        if hist.shape[0] != self.k + 1:
            raise ValueError(
                f"XL-BOMD history holds {hist.shape[0]} densities, expected {self.k + 1}"
            )
        self.history = [hist[i].copy() for i in range(self.k + 1)]
