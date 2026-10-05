"""
Psi4 backend: analytic SCF/DFT/MP2 gradients via the Psi4 Python API.

Adapted from Quantize's Psi4 backend (backend/psi4/). For MD the geometry is
passed in bohr and Psi4 is told not to translate, reorient or symmetrise it,
so gradient rows line up with our atom order at every step.

TODO: reuse the previous step's orbitals as the SCF guess (GUESS READ).
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np

from aimd.backends.base import ForceBackend, GradientResult
from aimd.backends.registry import register_backend


@register_backend
class Psi4Backend(ForceBackend):
    name = "psi4"

    def __init__(
        self,
        symbols: Sequence[str],
        charge: int = 0,
        multiplicity: int = 1,
        method: str = "hf",
        basis: str = "sto-3g",
        reference: str | None = None,
        memory: str = "2 GB",
        num_threads: int = 1,
        output_file: str | None = None,
        options: dict | None = None,
    ) -> None:
        super().__init__(symbols, charge, multiplicity)
        try:
            import psi4  # pylint: disable=import-outside-toplevel
        except ImportError as e:
            raise RuntimeError(
                "The psi4 backend needs Psi4 "
                "(conda install -c conda-forge psi4)."
            ) from e
        self._psi4 = psi4
        self.method = str(method).strip()
        self.basis = str(basis).strip()

        psi4.set_memory(memory)
        psi4.set_num_threads(int(num_threads))
        psi4.core.set_output_file(output_file or f"psi4_aimd_{os.getpid()}.out", False)

        if reference is None:
            reference = "rhf" if self.multiplicity == 1 else "uhf"
        opts = {"basis": self.basis, "reference": reference, "scf_type": "pk"}
        opts.update(options or {})
        psi4.set_options(opts)

    def _molecule(self, positions: np.ndarray):
        lines = [f"{self.charge} {self.multiplicity}"]
        for sym, (x, y, z) in zip(self.symbols, positions):
            lines.append(f"{sym} {x:.12f} {y:.12f} {z:.12f}")
        lines += ["units bohr", "no_com", "no_reorient", "symmetry c1"]
        return self._psi4.geometry("\n".join(lines))

    def compute(self, positions: np.ndarray) -> GradientResult:
        psi4 = self._psi4
        x = np.asarray(positions, dtype=float).reshape(-1, 3)
        mol = self._molecule(x)

        grad, wfn = psi4.gradient(self.method, molecule=mol, return_wfn=True)
        g = np.asarray(grad.np, dtype=float)
        return GradientResult(
            energy=float(wfn.energy()),
            gradient=g.reshape(-1, 3),
            info={"method": self.method, "basis": self.basis},
        )

    def close(self) -> None:
        self._psi4.core.clean()
