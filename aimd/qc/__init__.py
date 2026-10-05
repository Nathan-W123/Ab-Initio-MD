"""
Native quantum chemistry for the HF backend: Gaussian basis sets
(:mod:`aimd.qc.basis_data`, :mod:`aimd.qc.basis`), McMurchie-Davidson
molecular integrals and their nuclear-derivative contractions
(:mod:`aimd.qc.integrals`, built on :mod:`aimd.qc.hermite` and
:mod:`aimd.qc.boys`). Hartree atomic units throughout.
"""

from aimd.qc import integrals
from aimd.qc.basis import BasisSet, build_basis
from aimd.qc.basis_data import available_basis_sets, normalize_basis_name

__all__ = [
    "BasisSet",
    "available_basis_sets",
    "build_basis",
    "integrals",
    "normalize_basis_name",
]
