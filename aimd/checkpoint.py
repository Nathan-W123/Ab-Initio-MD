"""
Checkpoint / restart files.

A checkpoint holds everything needed to continue a run so that N steps +
checkpoint + restart + M steps is bit-for-bit the same as N + M steps:

  - the system: symbols, positions, velocities, masses, charge, multiplicity
    and the COM / rotation constraint flags (they set N_dof);
  - the step number and simulation time;
  - the integrator's ``state_dict()``: its configuration, the cached
    GradientResult (energy, gradient, density, dipole, info), RNG bit-generator
    states and thermostat variables, including the accumulated thermostat
    energy that enters the conserved quantity, and for XL-BOMD the auxiliary
    density history (K+1 AO density matrices).

Format: a NumPy ``.npz`` archive. Every array is stored as its own ``.npy``
member (exact float64 / uint64 round trip) and the rest of the state is UTF-8
JSON in the ``__metadata__`` member, with arrays referenced as
``{"__ndarray__": "<member name>"}``. Loading uses ``allow_pickle=False`` and
only rebuilds dicts, lists, scalars and arrays, so a checkpoint cannot execute
code. Files are written to a temporary name and renamed, so an interrupted
write never clobbers the previous checkpoint.

Units are those of the engine (bohr, bohr / au_time, m_e, hartree); the time
is in fs.
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aimd.system import MolecularSystem

if TYPE_CHECKING:
    from aimd.backends.base import ForceBackend
    from aimd.integrators import Integrator

FORMAT = "aimd-checkpoint"
VERSION = 1
_META_KEY = "__metadata__"
_ARRAY_TAG = "__ndarray__"


# ── Encoding: nested plain data + arrays <-> JSON + named arrays ─────────────

def _encode(obj: Any, arrays: dict[str, np.ndarray]) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.dtype.hasobject:
            raise TypeError("object arrays cannot be checkpointed")
        name = f"a{len(arrays):04d}"
        arrays[name] = obj
        return {_ARRAY_TAG: name}
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"checkpoint dict keys must be str, got {k!r}")
            if k == _ARRAY_TAG:
                raise TypeError(f"{_ARRAY_TAG!r} is a reserved key")
            out[k] = _encode(v, arrays)
        return out
    if isinstance(obj, (list, tuple)):
        return [_encode(v, arrays) for v in obj]
    if isinstance(obj, np.generic):
        return _encode(obj.item(), arrays)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise TypeError(f"cannot checkpoint object of type {type(obj).__name__}")


def _decode(obj: Any, arrays: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {_ARRAY_TAG}:
            return np.array(arrays[obj[_ARRAY_TAG]])
        return {k: _decode(v, arrays) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode(v, arrays) for v in obj]
    return obj


def _encodable(obj: Any) -> bool:
    try:
        _encode(obj, {})
    except TypeError:
        return False
    return True


def _sanitize_info(state: dict[str, Any]) -> dict[str, Any]:
    """Drop GradientResult.info entries that are not plain data (diagnostics only)."""
    result = state.get("result")
    if not result or not result.get("info"):
        return state
    info = {}
    for k, v in result["info"].items():
        if isinstance(k, str) and k != _ARRAY_TAG and _encodable(v):
            info[k] = v
        else:
            warnings.warn(
                f"GradientResult.info[{k!r}] ({type(v).__name__}) is not "
                "checkpointable plain data and was dropped",
                stacklevel=3,
            )
    return {**state, "result": {**result, "info": info}}


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class Checkpoint:
    """
    A loaded checkpoint. ``system`` is a working copy that may be propagated
    in place (``run_md(ckpt.system, integ, n, restart=ckpt)``); :meth:`restore`
    always uses the state as it was saved, so the same checkpoint can be
    restarted from more than once.
    """

    system: MolecularSystem
    step: int
    time_fs: float
    integrator_state: dict[str, Any]
    backend: str = ""                   # name of the backend that wrote it (info)
    _saved: MolecularSystem = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # Snapshot: running from ``self.system`` must not change what a later
        # restore() puts back (the cached forces belong to these positions).
        self._saved = self.system.copy()

    def restore(self, system: MolecularSystem, integrator: "Integrator") -> None:
        """Copy the saved state into ``system`` (in place) and ``integrator``."""
        src = self._saved
        if list(system.symbols) != list(src.symbols):
            raise ValueError(
                f"checkpoint atoms {src.symbols} != system atoms {system.symbols}"
            )
        integrator.load_state_dict(self.integrator_state)   # validates first
        system.positions = src.positions.copy()
        system.velocities = src.velocities.copy()
        system.masses = src.masses.copy()
        system.charge = src.charge
        system.multiplicity = src.multiplicity
        system.com_removed = src.com_removed
        system.rotation_removed = src.rotation_removed
        system.rotational_dof = src.rotational_dof

    def make_integrator(self, backend: "ForceBackend") -> "Integrator":
        """Rebuild the saved integrator (same class and settings) on ``backend``."""
        from aimd.integrators import integrator_from_state

        return integrator_from_state(backend, self.integrator_state)


def save_checkpoint(
    path: str | Path,
    system: MolecularSystem,
    integrator: "Integrator",
    step: int,
    time_fs: float | None = None,
) -> Path:
    """Write a checkpoint of ``system`` + ``integrator`` after ``step`` steps."""
    path = Path(path)
    if time_fs is None:
        time_fs = step * integrator.timestep_fs
    meta = {
        "format": FORMAT,
        "version": VERSION,
        "step": int(step),
        "time_fs": float(time_fs),
        "backend": str(getattr(integrator.backend, "name", "")),
        "system": {
            "symbols": list(system.symbols),
            "positions": np.array(system.positions, dtype=float),
            "velocities": np.array(system.velocities, dtype=float),
            "masses": np.array(system.masses, dtype=float),
            "charge": int(system.charge),
            "multiplicity": int(system.multiplicity),
            "com_removed": bool(system.com_removed),
            "rotation_removed": bool(system.rotation_removed),
            "rotational_dof": system.rotational_dof,
        },
        "integrator": _sanitize_info(integrator.state_dict()),
    }
    arrays: dict[str, np.ndarray] = {}
    encoded = _encode(meta, arrays)
    arrays[_META_KEY] = np.array(json.dumps(encoded))
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as fh:            # a file object: savez adds no suffix
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())             # data on disk before the rename
    os.replace(tmp, path)
    return path


def load_checkpoint(path: str | Path) -> Checkpoint:
    """Read a checkpoint written by :func:`save_checkpoint` (no pickle)."""
    with np.load(Path(path), allow_pickle=False) as npz:
        if _META_KEY not in npz.files:
            raise ValueError(f"{path} is not an aimd checkpoint")
        meta = json.loads(str(npz[_META_KEY]))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not an aimd checkpoint")
        if meta.get("version") != VERSION:
            raise ValueError(f"unsupported checkpoint version {meta.get('version')}")
        data = _decode(meta, {k: npz[k] for k in npz.files if k != _META_KEY})
    s = data["system"]
    system = MolecularSystem(
        symbols=s["symbols"],
        positions=s["positions"],
        velocities=s["velocities"],
        masses=s["masses"],
        charge=s["charge"],
        multiplicity=s["multiplicity"],
        com_removed=s["com_removed"],
        rotation_removed=s["rotation_removed"],
        rotational_dof=s["rotational_dof"],
    )
    return Checkpoint(
        system=system,
        step=int(data["step"]),
        time_fs=float(data["time_fs"]),
        integrator_state=data["integrator"],
        backend=data.get("backend", ""),
    )
