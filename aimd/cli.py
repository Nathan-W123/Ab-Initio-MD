"""
Command-line interface (console script ``aimd``).

    aimd run water.xyz --backend hf --basis sto-3g --steps 200 --dt 0.5 \\
        --temperature 300 --thermostat csvr --tau 50 --seed 1
    aimd run --config examples/water_nvt.toml          # same, from a file
    aimd run water.xyz ... --checkpoint run.npz --checkpoint-every 50
    aimd run water.xyz ... --restart run.npz --steps 100  # continue
    aimd analyze ir dipoles.csv --temperature 300      # see aimd.cli_analyze
    aimd backends                                      # registered backends

Units at this boundary: XYZ in angstrom, times in fs, temperatures in K,
friction in 1/fs; energies are printed and logged in hartree.

Configuration files (``--config``, JSON or TOML) hold the ``run`` options
as keys named like the long flags (``basis``, ``conv_tol`` or
``conv-tol``, ...), a ``xyz`` key for the starting geometry and an optional
``backend_options`` table passed to the backend constructor as keyword
arguments. Flags on the command line override the file. Relative ``xyz`` and
``restart`` paths in a file are resolved against the file's directory;
output paths against the current directory.

Restart: ``--restart CKPT`` continues the run saved in a checkpoint written
with ``--checkpoint``. The options must describe the same simulation (same
thermostat and XL-BOMD choice, output cadence; run it with the same command
line or config file); step numbers continue from the checkpoint, ``--steps``
more steps are run, and the output files are appended to after cutting
anything written beyond the checkpoint step, so they end up identical to
those of an uninterrupted run. The checkpoint records the backend and its
constructor arguments (method, basis, reference, SCF settings,
``--backend-option``): flags left out on the restart are taken from it, and
a flag that changes the potential-energy surface (method, basis, reference,
other backend options) is refused; SCF thresholds may change with a warning,
``--threads`` freely. An output file that does not continue the checkpointed
run (overwritten by another run in between) is refused too (aimd.md).
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from aimd.backends import backend_dependencies, get_backend, list_backends
from aimd.checkpoint import Checkpoint, check_writable, load_checkpoint
from aimd.elements import ATOMIC_NUMBERS
from aimd.integrators import (
    CSVR,
    XLBOMD,
    Integrator,
    LangevinBAOAB,
    NoseHooverChain,
    VelocityVerlet,
)
from aimd.md import MDResult, run_md
from aimd.system import MolecularSystem
from aimd.thermostats import (
    BerendsenThermostat,
    CSVRThermostat,
    NoseHooverChainThermostat,
    Thermostat,
)

THERMOSTATS = ("nve", "berendsen", "langevin", "csvr", "nhc")

# Generic electronic-structure flags -> constructor argument, per backend
# where the name differs. A flag the backend's constructor does not take is
# an error (e.g. --basis with the morse backend).
_GENERIC_OPTIONS = ("method", "basis", "reference", "threads", "conv_tol",
                    "conv_tol_grad", "max_cycles")
_ALIASES: dict[str, dict[str, str]] = {
    "psi4": {"threads": "num_threads"},
    "pyscf": {"max_cycles": "max_cycle"},
}
# Backend constructor arguments a restart may change: numerical SCF settings
# (warning: the surface moves by about the convergence error) and
# performance settings (silently). Any other change is refused.
_RESTART_SOFT = {"conv_tol", "conv_tol_grad", "max_cycles", "max_cycle", "guess",
                 "init_guess", "reuse_density"}
_RESTART_FREE = {"threads", "num_threads", "max_memory_mb", "memory"}
# argparse dests that a config file may not set
_NOT_CONFIGURABLE = {"config", "command", "func", "xyz", "help"}


class CLIError(Exception):
    """A user-input error: printed as ``aimd: error: ...``, exit status 2."""


# ── Parser ────────────────────────────────────────────────────────────────────

def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def _add_run_parser(sub: argparse._SubParsersAction) -> argparse.ArgumentParser:
    # allow_abbrev=False: --config is expanded before parsing (_expand_config),
    # so an abbreviation such as --conf must be an error, not silently accepted
    # and ignored
    r = sub.add_parser(
        "run", help="run an MD trajectory",
        description="Born-Oppenheimer MD from an XYZ geometry (angstrom).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False,
    )
    r.add_argument("xyz", nargs="?", default=None,
                   help="starting geometry (XYZ, angstrom); optional with --restart")
    r.add_argument("--xyz", dest="xyz_option", default=None, help=argparse.SUPPRESS)
    r.add_argument("--config", default=None,
                   help="JSON or TOML file with run options (flags override it)")

    g = r.add_argument_group("electronic structure")
    g.add_argument("--backend", default=None,
                   help=f"force backend ({', '.join(list_backends())}); default hf, "
                        "or the checkpoint's with --restart")
    g.add_argument("--method", default=None,
                   help="method, e.g. hf, uhf (hf); hf, b3lyp, pbe, mp2 (pyscf, psi4)")
    g.add_argument("--basis", default=None, help="basis set (backend default: sto-3g)")
    g.add_argument("--reference", default=None, help="rhf or uhf (default: by multiplicity)")
    g.add_argument("--charge", type=int, default=None, help="molecular charge (default 0)")
    g.add_argument("--multiplicity", type=int, default=None,
                   help="spin multiplicity 2S+1 (default 1)")
    g.add_argument("--threads", type=_positive_int, default=None,
                   help="threads for the backend (numba for hf, OpenMP for pyscf)")
    g.add_argument("--conv-tol", type=float, default=None,
                   help="SCF energy convergence, Eh (backend default 1e-10)")
    g.add_argument("--conv-tol-grad", type=float, default=None,
                   help="SCF orbital-gradient convergence")
    g.add_argument("--max-cycles", type=_positive_int, default=None,
                   help="SCF iteration limit")
    g.add_argument("--backend-option", action="append", default=[], metavar="KEY=VALUE",
                   help="extra backend constructor argument (VALUE parsed as JSON "
                        "if possible), e.g. cart=true; repeatable")

    d = r.add_argument_group("dynamics")
    d.add_argument("--steps", type=_nonnegative_int, default=100,
                   help="number of MD steps (more steps, when restarting)")
    d.add_argument("--dt", type=float, default=0.5, help="timestep, fs")
    d.add_argument("--thermostat", default="nve", choices=THERMOSTATS + ("none",),
                   help="nve (none), berendsen, langevin (BAOAB), csvr (Bussi) or "
                        "nhc (Nose-Hoover chain)")
    d.add_argument("--temperature", type=float, default=None,
                   help="thermostat target, K; also the initial temperature")
    d.add_argument("--init-temperature", type=float, default=None,
                   help="temperature of the initial Maxwell-Boltzmann velocities, K "
                        "(default: --temperature, or 0 = start at rest)")
    d.add_argument("--tau", type=float, default=100.0,
                   help="coupling time (berendsen, csvr) or chain period (nhc), fs")
    d.add_argument("--friction", type=float, default=0.01, help="Langevin friction, 1/fs")
    d.add_argument("--nhc-chain-length", type=_positive_int, default=3,
                   help="Nose-Hoover chain length")
    d.add_argument("--nhc-substeps", type=_positive_int, default=1,
                   help="Nose-Hoover multiple-time-step substeps")
    d.add_argument("--xlbomd", action=argparse.BooleanOptionalAction, default=False,
                   help="extended-Lagrangian BOMD (SCF guess from an auxiliary density)")
    d.add_argument("--xl-k", type=int, default=5,
                   help="XL-BOMD dissipation order K (3..9)")
    d.add_argument("--remove-rotation", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="remove the initial angular momentum (isolated molecules)")
    d.add_argument("--seed", type=_nonnegative_int, default=None,
                   help="random seed (initial velocities, csvr / langevin noise)")

    o = r.add_argument_group("output")
    o.add_argument("--trajectory", default="trajectory.xyz",
                   help="positions, multi-frame XYZ (angstrom); 'none' to disable")
    o.add_argument("--energies", default="energies.csv",
                   help="energy log CSV; 'none' to disable")
    o.add_argument("--velocities", default=None,
                   help="velocity trajectory (XYZ layout, bohr/au_time)")
    o.add_argument("--dipoles", default=None, help="dipole log CSV (e*bohr)")
    o.add_argument("--write-every", type=_positive_int, default=1,
                   help="write output every N steps")
    o.add_argument("--print-every", type=_nonnegative_int, default=1,
                   help="print a line every N steps (0: header and summary only)")
    o.add_argument("--checkpoint", default=None, help="checkpoint file to write (.npz)")
    o.add_argument("--checkpoint-every", type=_positive_int, default=None,
                   help="write the checkpoint every N steps (and at the end)")
    o.add_argument("--restart", default=None,
                   help="continue from this checkpoint (step numbers continue, "
                        "outputs are appended to)")
    r.set_defaults(func=cmd_run)
    return r


def build_parser() -> argparse.ArgumentParser:
    from aimd.cli_analyze import add_analyze_parser

    p = argparse.ArgumentParser(prog="aimd", description="Ab initio molecular dynamics")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("backends", help="list force backends and their dependencies")
    b.set_defaults(func=cmd_backends)
    p.run_parser = _add_run_parser(sub)               # type: ignore[attr-defined]
    add_analyze_parser(sub)
    return p


# ── Config files ──────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict[str, Any]:
    """Read a JSON (.json) or TOML (.toml, or anything else) config file."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as e:
        raise CLIError(f"cannot read config file {path}: {e.strerror or e}") from e
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise CLIError(f"config file {path} is not valid JSON: {e}") from e
    else:
        try:
            import tomllib
        except ModuleNotFoundError:                    # Python 3.10
            import tomli as tomllib                    # type: ignore[no-redef]
        try:
            data = tomllib.loads(text)
        except tomllib.TOMLDecodeError as e:
            raise CLIError(f"config file {path} is not valid TOML: {e}") from e
    if not isinstance(data, dict):
        raise CLIError(f"config file {path} must hold a table / object of options")
    return data


def _option_index(parser: argparse.ArgumentParser) -> dict[str, argparse.Action]:
    """Long option name (without --, dashes) -> action, for the run parser."""
    out = {}
    for action in parser._actions:                     # argparse has no public API
        for opt in action.option_strings:
            if opt.startswith("--") and not opt.startswith("--no-"):
                out[opt[2:]] = action
    return out


def config_to_argv(
    data: dict[str, Any], run_parser: argparse.ArgumentParser, base: Path
) -> list[str]:
    """Translate config keys into ``run`` flags (placed before the real ones)."""
    index = _option_index(run_parser)
    argv: list[str] = []
    for raw_key, value in data.items():
        key = str(raw_key).strip().replace("_", "-")
        if key == "backend-options":
            if not isinstance(value, dict):
                raise CLIError("config key 'backend_options' must be a table / object")
            for k, v in value.items():
                argv += ["--backend-option", f"{k}={json.dumps(v)}"]
            continue
        action = index.get(key)
        if key in ("xyz", "restart") and value is not None:
            p = Path(str(value)).expanduser()
            value = str(p if p.is_absolute() else base / p)
            action = index["xyz"] if key == "xyz" else action
        if action is None or action.dest in _NOT_CONFIGURABLE:
            valid = sorted(k.replace("-", "_") for k, a in index.items()
                           if a.dest not in _NOT_CONFIGURABLE and a.help != argparse.SUPPRESS)
            raise CLIError(f"unknown option {raw_key!r} in the config file; "
                           f"valid keys: xyz, backend_options, {', '.join(valid)}")
        flag = "--" + key
        if isinstance(action, argparse.BooleanOptionalAction):
            if not isinstance(value, bool):
                raise CLIError(f"config option {raw_key!r} must be true or false")
            argv.append(flag if value else "--no-" + key)
        elif value is None:
            continue
        elif isinstance(value, bool):
            raise CLIError(f"config option {raw_key!r} takes a value, not a boolean")
        elif isinstance(value, (list, dict)):
            if action.dest == "backend_option" and isinstance(value, list):
                for item in value:
                    argv += [flag, str(item)]
            else:
                raise CLIError(f"config option {raw_key!r} must be a single value")
        else:
            argv += [flag, str(value)]
    return argv


def _expand_config(argv: list[str], parser: argparse.ArgumentParser) -> list[str]:
    """``run ... --config F ...`` -> ``run <flags from F> ...`` (flags win)."""
    if not argv or argv[0] != "run":
        return argv
    rest = argv[1:]
    path = None
    for i, tok in enumerate(rest):
        if tok == "--":
            break
        if tok == "--config" and i + 1 < len(rest):
            path = rest[i + 1]
        elif tok.startswith("--config="):
            path = tok.split("=", 1)[1]
    if path is None:
        return argv
    data = load_config(path)
    base = Path(path).resolve().parent
    return ["run", *config_to_argv(data, parser.run_parser, base), *rest]  # type: ignore[attr-defined]


# ── aimd backends ─────────────────────────────────────────────────────────────

def _version(pkg: str) -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version(pkg)
    except (ImportError, PackageNotFoundError):
        return "?"


def cmd_backends(args: argparse.Namespace) -> int:
    print(f"{'backend':10s} {'optional dependency':28s} description")
    for name in list_backends():
        cls = get_backend(name)
        deps = backend_dependencies(name)
        if deps:
            dep = ", ".join(
                f"{pkg} {_version(pkg)} (installed)" if ok else f"{pkg} (NOT installed)"
                for pkg, ok in deps.items()
            )
        else:
            dep = "none needed"
        summary = getattr(cls, "description", "") or (
            (cls.__doc__ or "").strip().splitlines() or [""])[0]
        print(f"{name:10s} {dep:28s} {summary}")
    return 0


# ── aimd run: building the pieces ─────────────────────────────────────────────

def _parse_backend_options(items: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise CLIError(f"--backend-option expects KEY=VALUE, got {item!r}")
        key, text = item.split("=", 1)
        key = key.strip()
        if not key.isidentifier():
            raise CLIError(f"invalid backend option name {key!r}")
        try:
            out[key] = json.loads(text)
        except json.JSONDecodeError:
            out[key] = text
    return out


def backend_kwargs(args: argparse.Namespace, cls: type) -> dict[str, Any]:
    """Constructor keyword arguments for backend class ``cls`` from the flags."""
    params = inspect.signature(cls.__init__).parameters
    takes_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    aliases = _ALIASES.get(cls.name, {})
    kwargs: dict[str, Any] = {}
    for opt in _GENERIC_OPTIONS:
        value = getattr(args, opt)
        if value is None:
            continue
        name = aliases.get(opt, opt)
        if name not in params and not takes_any:
            raise CLIError(
                f"--{opt.replace('_', '-')} is not an option of the {cls.name!r} backend")
        kwargs[name] = value
    for key, value in _parse_backend_options(args.backend_option).items():
        if key in ("symbols", "charge", "multiplicity", "self"):
            raise CLIError(f"backend option {key!r} is set by aimd itself")
        if key not in params and not takes_any:
            valid = [p for p in params if p not in ("self", "symbols", "charge", "multiplicity")]
            raise CLIError(f"the {cls.name!r} backend has no option {key!r} "
                           f"(options: {', '.join(valid)})")
        kwargs[key] = value
    return kwargs


def _normalized(value: Any) -> Any:
    """Plain, comparable form of a backend argument (names case-insensitive)."""
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, np.ndarray):
        return _normalized(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_normalized(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalized(v) for k, v in value.items()}
    return value


def restart_backend_kwargs(
    cls: type, given: dict[str, Any], saved: dict[str, Any], restart: str
) -> tuple[dict[str, Any], list[str]]:
    """
    Backend arguments for a restart: those recorded in the checkpoint
    (``saved``) overridden by the ones given now. A given value that differs
    from the recorded one (or, for an argument not recorded, from the
    constructor default) is refused if it changes the potential-energy
    surface, warned about if it is an SCF threshold (_RESTART_SOFT) and
    accepted if it only affects performance (_RESTART_FREE).
    Returns (kwargs, names taken from the checkpoint).
    """
    params = inspect.signature(cls.__init__).parameters
    out = dict(saved)
    inherited = [k for k in saved if k not in given]
    soft = []
    for key, value in given.items():
        if key in saved:
            before = saved[key]
        elif key in params and params[key].default is not inspect.Parameter.empty:
            before = params[key].default
        else:
            before = "<not set>"
        out[key] = value
        if _normalized(value) == _normalized(before) or key in _RESTART_FREE:
            continue
        if key in _RESTART_SOFT:
            soft.append(f"{key} {before!r} -> {value!r}")
            continue
        flag = (f"--{key.replace('_', '-')}" if key in _GENERIC_OPTIONS
                else f"--backend-option {key}")
        raise CLIError(
            f"{flag} {value!r} differs from the checkpoint's {before!r}: the run in "
            f"{restart} used another potential-energy surface. Leave the option out "
            "to continue with the checkpoint's setting")
    if soft:
        warnings.warn(f"SCF settings differ from the checkpoint ({'; '.join(soft)}); "
                      "the potential energy changes by about the convergence error",
                      UserWarning, stacklevel=2)
    return out, inherited


def check_spin_state(symbols: Sequence[str], charge: int, multiplicity: int) -> None:
    """Reject charge / multiplicity combinations no electron count allows."""
    n_el = sum(ATOMIC_NUMBERS[s] for s in symbols) - charge
    if multiplicity < 1:
        raise CLIError(f"multiplicity must be >= 1, got {multiplicity}")
    if n_el < 0:
        raise CLIError(f"charge {charge} leaves {n_el} electrons")
    if (n_el + multiplicity - 1) % 2:
        parity = "even" if n_el % 2 == 0 else "odd"
        need = "odd (1, 3, ...)" if n_el % 2 == 0 else "even (2, 4, ...)"
        raise CLIError(
            f"charge {charge} and multiplicity {multiplicity} are inconsistent: "
            f"{n_el} electrons ({parity}) need an {need} multiplicity")
    if multiplicity - 1 > n_el:
        raise CLIError(f"multiplicity {multiplicity} needs at least {multiplicity - 1} "
                       f"electrons, the molecule has {n_el}")


def _streams(seed: int | None) -> tuple[np.random.Generator, np.random.Generator]:
    """Independent generators for the initial velocities and thermostat noise."""
    a, b = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(a), np.random.default_rng(b)


def build_integrator(
    args: argparse.Namespace, backend: Any, rng: np.random.Generator | None
) -> Integrator:
    """The integrator / thermostat combination selected by the flags."""
    kind = "nve" if args.thermostat == "none" else args.thermostat
    temp = args.temperature
    if kind != "nve" and temp is None:
        raise CLIError(f"--thermostat {kind} needs a target --temperature (K)")
    if args.xlbomd:
        if kind == "langevin":
            raise CLIError("--xlbomd cannot be combined with Langevin dynamics; "
                           "use --thermostat csvr or nhc")
        if not getattr(backend, "supports_density_guess", False):
            raise CLIError(f"--xlbomd needs a backend that accepts SCF density "
                           f"guesses (hf, pyscf); {backend.name!r} does not")
        thermo: Thermostat | None = None
        if kind == "berendsen":
            thermo = BerendsenThermostat(temp, args.tau)
        elif kind == "csvr":
            thermo = CSVRThermostat(temp, args.tau, rng)
        elif kind == "nhc":
            thermo = NoseHooverChainThermostat(temp, args.tau, args.nhc_chain_length,
                                               args.nhc_substeps)
        return XLBOMD(backend, args.dt, k=args.xl_k, thermostat=thermo)
    if kind == "nve":
        return VelocityVerlet(backend, args.dt)
    if kind == "berendsen":
        return VelocityVerlet(backend, args.dt, temp, args.tau)
    if kind == "csvr":
        return CSVR(backend, args.dt, temp, args.tau, rng=rng)
    if kind == "nhc":
        return NoseHooverChain(backend, args.dt, temp, args.tau,
                               args.nhc_chain_length, args.nhc_substeps)
    return LangevinBAOAB(backend, args.dt, temp, args.friction, rng=rng)


def _describe_integrator(integ: Integrator) -> str:
    dt = f"dt = {integ.timestep_fs:g} fs"
    if isinstance(integ, LangevinBAOAB):
        return (f"Langevin BAOAB, T = {integ.temperature_k:g} K, friction "
                f"{integ.friction_per_fs:g}/fs, {dt}")
    name = "XL-BOMD (K = %d) + velocity Verlet" % integ.k if isinstance(integ, XLBOMD) \
        else "velocity Verlet"
    thermo = getattr(integ, "thermostat", None)
    if thermo is None:
        return f"{name}, NVE, {dt}"
    cfg = thermo.config()
    if isinstance(thermo, NoseHooverChainThermostat):
        extra = (f"period {cfg['period_fs']:g} fs, chain {cfg['chain_length']}, "
                 f"{cfg['n_mts']} substep(s)")
    else:
        extra = f"tau {cfg['tau_fs']:g} fs"
    label = {"BerendsenThermostat": "Berendsen", "CSVRThermostat": "CSVR (Bussi)",
             "NoseHooverChainThermostat": "Nose-Hoover chain"}[type(thermo).__name__]
    return f"{name} + {label}, T = {thermo.temperature_k:g} K, {extra}, {dt}"


def _backend_header(backend: Any, args: argparse.Namespace) -> list[str]:
    lines = [f"backend    {backend.name}"]
    label = getattr(backend, "label", None) or args.method
    if label:
        lines.append(f"method     {label}")
    basis = getattr(backend, "basis", None)
    basis = getattr(basis, "name", basis)
    if isinstance(basis, str):
        nao = getattr(backend, "nao", None)
        lines.append(f"basis      {basis}" + (f" ({nao} AOs)" if nao else ""))
    return lines


def _check_restart(ckpt: Checkpoint, integ: Integrator, backend: Any,
                   args: argparse.Namespace) -> None:
    """Explain a checkpoint / options mismatch before anything is changed."""
    if ckpt.backend and ckpt.backend != backend.name:
        raise CLIError(f"checkpoint {args.restart} was written by the {ckpt.backend!r} "
                       f"backend, but this run uses {backend.name!r}")
    try:
        ckpt.check_backend(backend)
    except ValueError as e:
        raise CLIError(f"checkpoint {args.restart}: {e}") from e
    state = ckpt.integrator_state
    saved = state.get("integrator")
    thermo = getattr(integ, "thermostat", None)
    mine = (type(integ).__name__, None if thermo is None else type(thermo).__name__)
    theirs = (saved, (state.get("thermostat") or {}).get("type"))
    if mine != theirs:
        def show(pair: tuple[str | None, str | None]) -> str:
            return pair[0] + (f" + {pair[1]}" if pair[1] else "")
        raise CLIError(
            f"checkpoint {args.restart} was written by integrator {show(theirs)}, but "
            f"the options select {show(mine)}; restart with the same --thermostat / "
            "--xlbomd options as the original run")
    density = (state.get("result") or {}).get("density")
    want = getattr(backend, "density_shape", None)
    if density is not None and want is not None and np.shape(density) != tuple(want):
        raise CLIError(
            f"checkpoint {args.restart} holds an SCF density of shape "
            f"{np.shape(density)}, but this backend expects {tuple(want)}: the basis "
            "or reference differs from the original run")
    mine, theirs = integ.config(), state.get("config", {})
    changed = sorted(k for k in set(mine) | set(theirs) if mine.get(k) != theirs.get(k))
    if changed:
        warnings.warn(
            f"integrator settings differ from the checkpoint ({', '.join(changed)}); "
            "the new values are used and the conserved quantity is not continuous",
            UserWarning, stacklevel=2)


def _read_xyz(path: str) -> MolecularSystem:
    try:
        return MolecularSystem.from_xyz(path)
    except ValueError as e:
        raise CLIError(f"{path}: {e}") from e


def _output(path: str | None) -> str | None:
    if path is None or str(path).strip().lower() in ("", "none"):
        return None
    return path


# ── aimd run ──────────────────────────────────────────────────────────────────

def cmd_run(args: argparse.Namespace) -> int:
    xyz = args.xyz or args.xyz_option
    if args.dt <= 0.0 or not math.isfinite(args.dt):
        raise CLIError("--dt must be a positive number of fs")
    if args.checkpoint_every is not None and not args.checkpoint:
        raise CLIError("--checkpoint-every needs --checkpoint PATH")
    if _output(args.checkpoint) is None:          # --checkpoint none: no checkpoints
        args.checkpoint = args.checkpoint_every = None
    else:
        try:                                      # before any compute, not at the end
            check_writable(args.checkpoint)
        except ValueError as e:
            raise CLIError(str(e)) from e

    ckpt: Checkpoint | None = None
    if args.restart:
        try:
            ckpt = load_checkpoint(args.restart)
        except FileNotFoundError as e:
            raise CLIError(f"restart file {args.restart} not found") from e
        except ValueError as e:
            raise CLIError(str(e)) from e
        system = ckpt.system
        if xyz:
            start = _read_xyz(xyz)
            if start.symbols != system.symbols:
                raise CLIError(f"{xyz} has atoms {start.symbols}, the checkpoint "
                               f"{system.symbols}")
        for opt in ("charge", "multiplicity"):
            given = getattr(args, opt)
            if given is not None and given != getattr(system, opt):
                raise CLIError(f"--{opt} {given} differs from the checkpoint's "
                               f"{getattr(system, opt)}")
    else:
        if not xyz:
            raise CLIError("no starting geometry: give an XYZ file (or --restart)")
        system = _read_xyz(xyz)
        system.charge = args.charge or 0
        system.multiplicity = 1 if args.multiplicity is None else args.multiplicity
    check_spin_state(system.symbols, system.charge, system.multiplicity)

    if args.backend is None:
        args.backend = (ckpt.backend if ckpt is not None and ckpt.backend else "hf")
    if ckpt is not None and ckpt.backend and ckpt.backend != args.backend:
        raise CLIError(f"checkpoint {args.restart} was written by the {ckpt.backend!r} "
                       f"backend, but this run uses {args.backend!r}")
    try:
        cls = get_backend(args.backend)
    except ValueError as e:
        raise CLIError(str(e)) from e
    kwargs = backend_kwargs(args, cls)
    inherited: list[str] = []
    saved = None if ckpt is None else ckpt.run_info.get("backend_kwargs")
    if saved is not None:
        kwargs, inherited = restart_backend_kwargs(cls, kwargs, saved, args.restart)
    elif ckpt is not None and not ckpt.backend_info:
        warnings.warn(f"checkpoint {args.restart} does not record the backend settings "
                      "(written by an older aimd): make sure the method, basis and "
                      "backend options are those of the original run", UserWarning,
                      stacklevel=2)
    _default_reference_positions(cls, kwargs, xyz)
    args.inherited_backend_options = {k: kwargs[k] for k in inherited
                                      if k != "reference_positions"}
    try:
        backend = cls(system.symbols, system.charge, system.multiplicity, **kwargs)
    except (ValueError, RuntimeError, TypeError) as e:
        raise CLIError(f"cannot set up the {cls.name!r} backend: {e}") from e

    try:
        return _run(args, system, backend, ckpt, kwargs)
    finally:
        backend.close()


def _default_reference_positions(cls: type, kwargs: dict[str, Any], xyz: str | None) -> None:
    """
    Model surfaces with a ``reference_positions`` argument (the harmonic
    backend) default to the origin, which from the CLI would tether every atom
    to (0, 0, 0) far from the molecule (water at 100 K ran at ~70,000 K). From
    the CLI the well is centred on the starting geometry instead: the XYZ file
    (also on a restart, where the checkpoint holds the current, not the
    starting, positions). An explicit --backend-option reference_positions=...
    (bohr) wins.
    """
    if "reference_positions" not in inspect.signature(cls.__init__).parameters:
        return
    if "reference_positions" in kwargs:
        return
    if xyz is None:                            # only possible with --restart
        raise CLIError(f"the {cls.name!r} backend is centred on the starting geometry: "
                       "give the original XYZ file together with --restart, or "
                       "--backend-option reference_positions=[...] (bohr)")
    kwargs["reference_positions"] = _read_xyz(xyz).positions


def _run(args: argparse.Namespace, system: MolecularSystem, backend: Any,
         ckpt: Checkpoint | None, kwargs: dict[str, Any] | None = None) -> int:
    vel_rng, noise_rng = _streams(args.seed)
    try:
        integ = build_integrator(args, backend, noise_rng)
    except ValueError as e:
        raise CLIError(str(e)) from e

    if ckpt is not None:
        _check_restart(ckpt, integ, backend, args)
    else:
        t0 = args.init_temperature if args.init_temperature is not None else args.temperature
        if t0 is not None and (t0 < 0.0 or not math.isfinite(t0)):
            raise CLIError("the initial temperature must be finite and >= 0")
        system.initialize_velocities(t0 or 0.0, rng=vel_rng,
                                     remove_rotation=args.remove_rotation)

    print("aimd run")
    print(f"system     {len(system.symbols)} atoms ({''.join(_formula(system.symbols))}), "
          f"charge {system.charge}, multiplicity {system.multiplicity}")
    for line in _backend_header(backend, args):
        print(line)
    print(f"integrator {_describe_integrator(integ)}")
    if ckpt is not None:
        print(f"restart    {args.restart}: step {ckpt.step}, t = {ckpt.time_fs:g} fs; "
              f"{args.steps} more steps")
        taken = getattr(args, "inherited_backend_options", None)
        if taken:
            print("           backend settings from the checkpoint: "
                  + ", ".join(f"{k}={v!r}" for k, v in taken.items()))
    else:
        print(f"start      T = {system.temperature():.2f} K, N_dof = {system.n_dof}"
              + (", rotation removed" if system.rotation_removed else ""))
    if isinstance(integ, LangevinBAOAB) and args.remove_rotation:
        print("note       Langevin noise re-thermalises translation and rotation")

    timer = _StepTimer(integ, args.print_every)
    outputs = dict(trajectory=_output(args.trajectory), energy_log=_output(args.energies),
                   velocity_trajectory=_output(args.velocities),
                   dipole_log=_output(args.dipoles))
    ckpt_path = _output(args.checkpoint)
    ckpt_before = _mtime(ckpt_path)
    t_start = time.perf_counter()
    try:
        result = run_md(
            system, integ, args.steps, write_every=args.write_every, callback=timer,
            checkpoint_path=_output(args.checkpoint),
            checkpoint_every=args.checkpoint_every, restart=ckpt,
            checkpoint_info={"backend_kwargs": _storable(kwargs or {})}, **outputs,
        )
    except KeyboardInterrupt:
        print("\ninterrupted" + _checkpoint_note(ckpt_path, ckpt_before, args.restart),
              file=sys.stderr)
        return 130
    except ValueError as e:
        # e.g. appending to a log with other columns, or a checkpoint that
        # does not fit the integrator
        raise CLIError(str(e)) from e
    except (FloatingPointError, RuntimeError) as e:
        print(f"aimd: error during the run: {e}", file=sys.stderr)
        return 1
    _summary(result, timer, time.perf_counter() - t_start, outputs, args)
    return 0


def _storable(kwargs: dict[str, Any]) -> dict[str, Any]:
    """The backend arguments that a checkpoint can hold (plain data / arrays)."""
    from aimd.checkpoint import _encodable

    return {k: v for k, v in kwargs.items() if _encodable(v)}


def _mtime(path: str | None) -> int | None:
    try:
        return Path(path).stat().st_mtime_ns if path else None
    except OSError:
        return None


def _checkpoint_note(path: str | None, mtime_before: int | None, restart: str | None) -> str:
    """
    After an interrupt: name the checkpoint only if this run wrote one (or it
    is the file the run restarted from), with its step, so that the advice
    to ``--restart`` from it works.
    """
    if path is None:
        return ""
    same_as_restart = restart is not None and Path(restart).resolve() == Path(path).resolve()
    if _mtime(path) is None or (_mtime(path) == mtime_before and not same_as_restart):
        return f"; no checkpoint has been written to {path} yet"
    try:
        ckpt = load_checkpoint(path)
    except Exception as e:                     # noqa: BLE001 - report it, keep the exit code
        return f"; the checkpoint {path} could not be read ({e})"
    return f"; the last checkpoint (step {ckpt.step}, t = {ckpt.time_fs:g} fs) is in {path}"


def _formula(symbols: Sequence[str]) -> list[str]:
    counts: dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    return [f"{s}{n if n > 1 else ''}" for s, n in counts.items()]


class _StepTimer:
    """run_md callback: per-step lines, wall time per step and SCF statistics."""

    def __init__(self, integ: Integrator, print_every: int) -> None:
        self.integ = integ
        self.print_every = print_every
        self.last = time.perf_counter()
        self.step_times: list[float] = []
        self.backend_times: list[float] = []
        self.scf_iterations: list[int] = []
        self.first = True

    def __call__(self, rec: dict) -> None:
        now = time.perf_counter()
        info = self.integ.result.info if self.integ.result is not None else {}
        if not self.first:
            self.step_times.append(now - self.last)
            total = (info.get("timings") or {}).get("total")
            if total is not None:
                self.backend_times.append(float(total))
            if "scf_iterations" in info:
                self.scf_iterations.append(int(info["scf_iterations"]))
        self.last = now
        if self.first and self.print_every:
            print(f"{'step':>8s} {'time/fs':>10s} {'E_pot/Eh':>16s} {'E_tot/Eh':>16s} "
                  f"{'E_cons/Eh':>16s} {'T/K':>9s} {'SCF':>4s} {'ms':>7s}")
        if self.print_every and (self.first or rec["step"] % self.print_every == 0):
            scf = info.get("scf_iterations", "-")
            if info.get("scf_converged") is False:
                scf = f"{scf}!"
            ms = "-" if self.first else f"{1e3 * self.step_times[-1]:.1f}"
            print(f"{rec['step']:8d} {rec['time_fs']:10.3f} {rec['potential_Eh']:16.8f} "
                  f"{rec['total_Eh']:16.8f} {rec['conserved_Eh']:16.8f} "
                  f"{rec['temperature_K']:9.2f} {scf!s:>4s} {ms:>7s}", flush=True)
        self.first = False


def _summary(result: MDResult, timer: _StepTimer, wall: float,
             outputs: dict[str, str | None], args: argparse.Namespace) -> None:
    n = len(result.records) - 1
    print("summary")
    if n < 1:
        print("  no steps run")
        return
    t = result.column("time_fs")
    e = result.column("conserved_Eh")
    span_ps = (t[-1] - t[0]) * 1e-3
    print(f"  steps            {n} ({result.records[0]['step']} -> "
          f"{result.records[-1]['step']}), {span_ps * 1e3:g} fs")
    line = (f"  conserved energy max |E - E0| = {result.conserved_energy_drift:.3e} Eh, "
            f"final E - E0 = {e[-1] - e[0]:+.3e} Eh")
    if span_ps >= 0.05:              # a fitted slope means little over a few periods
        line += f", fitted drift {np.polyfit(t * 1e-3, e, 1)[0]:+.3e} Eh/ps"
    print(line)
    temp = result.column("temperature_K")[1:]
    print(f"  temperature      mean {temp.mean():.2f} K, std {temp.std():.2f} K "
          f"(steps {result.records[1]['step']}-{result.records[-1]['step']})")
    if timer.scf_iterations:
        it = np.array(timer.scf_iterations)
        print(f"  SCF iterations   mean {it.mean():.2f} per step (min {it.min()}, "
              f"max {it.max()})")
    if result.unconverged_steps:
        print(f"  WARNING          {len(result.unconverged_steps)} unconverged force "
              f"evaluation(s), first at step {result.unconverged_steps[0]}")
    per_step = np.median(timer.step_times) * 1e3
    line = f"  timing           {wall:.2f} s total, {per_step:.2f} ms per step (median)"
    if timer.backend_times:
        line += f", backend {np.median(timer.backend_times) * 1e3:.2f} ms"
    print(line)
    written = [p for p in outputs.values() if p]
    if args.checkpoint:
        written.append(args.checkpoint)
    if written:
        print(f"  output           {', '.join(written)}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    tokens = list(sys.argv[1:] if argv is None else argv)
    try:
        tokens = _expand_config(tokens, parser)
        args = parser.parse_args(tokens)
        if args.command == "analyze":
            try:
                return int(args.func(args))
            except ValueError as e:                    # unreadable / unsuitable input
                raise CLIError(str(e)) from e
        return int(args.func(args))
    except CLIError as e:
        print(f"aimd: error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"aimd: error: file not found: {e.filename}", file=sys.stderr)
        return 2
    except SystemExit as e:                            # argparse: usage error or --help
        return e.code if isinstance(e.code, int) else 2


if __name__ == "__main__":
    sys.exit(main())
