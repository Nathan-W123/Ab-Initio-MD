"""
Native quantum chemistry for the HF backend: Gaussian basis sets
(:mod:`aimd.qc.basis_data`, :mod:`aimd.qc.basis`), McMurchie-Davidson
molecular integrals and their nuclear-derivative contractions
(:mod:`aimd.qc.integrals`, built on :mod:`aimd.qc.hermite` and
:mod:`aimd.qc.boys`), restricted / unrestricted Hartree-Fock SCF
(:mod:`aimd.qc.scf`) and its analytic nuclear gradients
(:mod:`aimd.qc.gradients`). Hartree atomic units throughout.
"""

from aimd.qc import integrals
from aimd.qc.basis import BasisSet, build_basis
from aimd.qc.basis_data import available_basis_sets, normalize_basis_name
from aimd.qc.gradients import GradientTerms, gradient_terms, scf_gradient
from aimd.qc.scf import SCFConvergenceWarning, SCFOptions, SCFResult, SCFSolver, run_scf

__all__ = [
    "BasisSet",
    "GradientTerms",
    "SCFConvergenceWarning",
    "SCFOptions",
    "SCFResult",
    "SCFSolver",
    "available_basis_sets",
    "build_basis",
    "gradient_terms",
    "integrals",
    "normalize_basis_name",
    "run_scf",
    "scf_gradient",
]
