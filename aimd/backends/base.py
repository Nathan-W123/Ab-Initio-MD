"""
Force-provider interface between the MD loop and electronic-structure codes.

Modelled on Quantize's ``QuantumBackend`` (backend/base_backend.py), trimmed
to what MD needs: one energy + gradient call per step.

Adding a backend
----------------
1. Subclass :class:`ForceBackend` in ``aimd/backends/<name>.py``.
2. Set ``name = "<name>"`` (the key used by the CLI / configs).
3. Decorate the class with ``@register_backend``.
4. Implement :meth:`ForceBackend.compute`.
5. Import the module in ``aimd/backends/__init__.py`` to trigger registration.

Unit contract: ``compute`` receives positions in bohr, shape (N, 3), and
returns the energy in hartree and the gradient dE/dR in hartree / bohr,
shape (N, 3). Forces are ``-gradient``.

Density-guess protocol (SCF backends; used by XL-BOMD and guess reuse)
---------------------------------------------------------------------
A backend that sets ``supports_density_guess = True`` must:
  - return its converged AO density in ``GradientResult.density``:
    shape (nao, nao) total density for restricted references, or
    shape (2, nao, nao) [alpha, beta] for unrestricted ones;
  - accept an array of the same shape in ``set_density_guess``, to be used as
    the starting density of the *next* ``compute`` call only.
Backends that do not support it leave ``density`` as ``None``.

Optional properties: ``GradientResult.dipole`` is the electronic + nuclear
dipole moment in atomic units (e * bohr), shape (3,), or ``None``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence

import numpy as np


@dataclass
class GradientResult:
    """Energy and nuclear gradient for one geometry."""
    energy: float                       # hartree
    gradient: np.ndarray                # hartree / bohr, (N, 3)
    converged: bool = True
    density: np.ndarray | None = None   # AO density, see module docstring
    dipole: np.ndarray | None = None    # e * bohr, (3,)
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def forces(self) -> np.ndarray:
        return -self.gradient


class ForceBackend(ABC):
    """Abstract energy/gradient provider."""

    name: ClassVar[str] = ""
    supports_density_guess: ClassVar[bool] = False

    def __init__(
        self,
        symbols: Sequence[str],
        charge: int = 0,
        multiplicity: int = 1,
    ) -> None:
        self.symbols = list(symbols)
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)

    @abstractmethod
    def compute(self, positions: np.ndarray) -> GradientResult:
        """Energy and gradient at ``positions`` (bohr, shape (N, 3))."""

    def set_density_guess(self, density: np.ndarray) -> None:
        """Starting density for the next ``compute`` call (SCF backends only)."""
        raise NotImplementedError(
            f"backend '{self.name}' does not support density guesses"
        )

    def close(self) -> None:
        """Release external resources (scratch files, processes). Optional."""
