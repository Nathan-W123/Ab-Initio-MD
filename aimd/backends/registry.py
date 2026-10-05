"""
Backend registry: maps string names to ForceBackend subclasses.

Backends self-register when their module is imported::

    @register_backend
    class MyBackend(ForceBackend):
        name = "mybackend"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aimd.backends.base import ForceBackend

_REGISTRY: dict[str, type["ForceBackend"]] = {}


def register_backend(cls: type) -> type:
    key = str(cls.name).strip().lower()
    if not key:
        raise ValueError(f"{cls.__name__} must define a non-empty 'name'")
    _REGISTRY[key] = cls
    return cls


def get_backend(name: str) -> type["ForceBackend"]:
    key = str(name).strip().lower()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. Available backends: {list_backends()}"
        )
    return _REGISTRY[key]


def list_backends() -> list[str]:
    return sorted(_REGISTRY)
