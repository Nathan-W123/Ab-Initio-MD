"""Force backends. Importing this package registers all built-in backends."""

from aimd.backends.base import ForceBackend, GradientResult
from aimd.backends.registry import get_backend, list_backends, register_backend

# Registration side effects. psi4 is imported lazily inside Psi4Backend, so
# registering it does not require Psi4 to be installed.
from aimd.backends import morse as _morse  # noqa: F401
from aimd.backends import psi4_backend as _psi4  # noqa: F401

__all__ = [
    "ForceBackend",
    "GradientResult",
    "get_backend",
    "list_backends",
    "register_backend",
]
