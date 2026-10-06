"""
Force backends. Importing this package registers all built-in backends:

  hf        native RHF / UHF with analytic gradients (aimd.qc; needs only numba)
  pyscf     HF / DFT / MP2 through PySCF (optional dependency, imported lazily)
  psi4      HF / DFT / MP2 through Psi4 (optional dependency, imported lazily)
  morse     pairwise Morse model surface (tests, plumbing)
  harmonic  harmonic model surface (tests against closed-form results)

PySCF and Psi4 are imported only when such a backend is constructed, so
registering them needs neither installed. :func:`backend_dependencies`
reports which optional packages a backend needs and whether they are present
(``aimd backends`` prints it).
"""

from __future__ import annotations

import importlib.util

from aimd.backends.base import ForceBackend, GradientResult
from aimd.backends.registry import get_backend, list_backends, register_backend

# Registration side effects.
from aimd.backends import harmonic as _harmonic  # noqa: F401,E402
from aimd.backends import hf as _hf  # noqa: F401,E402
from aimd.backends import morse as _morse  # noqa: F401,E402
from aimd.backends import psi4_backend as _psi4  # noqa: F401,E402
from aimd.backends import pyscf_backend as _pyscf  # noqa: F401,E402


def backend_dependencies(name: str) -> dict[str, bool]:
    """
    Optional Python packages the backend ``name`` needs, mapped to whether
    each can be imported (checked with importlib.util.find_spec, without
    importing it). Empty for backends that need only aimd's own dependencies.
    """
    cls = get_backend(name)
    return {
        pkg: importlib.util.find_spec(pkg) is not None
        for pkg in getattr(cls, "requires", ())
    }


__all__ = [
    "ForceBackend",
    "GradientResult",
    "backend_dependencies",
    "get_backend",
    "list_backends",
    "register_backend",
]
