"""
Command-line entry point.

    aimd run water.xyz --backend psi4 --method hf --basis sto-3g \\
        --steps 200 --dt 0.5 --temperature 300 --thermostat langevin

    aimd backends          # list registered backends
"""

from __future__ import annotations

import argparse
import sys

from aimd.backends import get_backend, list_backends
from aimd.integrators import LangevinBAOAB, VelocityVerlet
from aimd.md import run_md
from aimd.system import MolecularSystem


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aimd", description="Ab initio molecular dynamics")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("backends", help="list available force backends")

    r = sub.add_parser("run", help="run an MD trajectory")
    r.add_argument("xyz", help="starting geometry (XYZ, angstrom)")
    r.add_argument("--backend", default="psi4", help="force backend (default: psi4)")
    r.add_argument("--method", default="hf", help="electronic-structure method (psi4)")
    r.add_argument("--basis", default="sto-3g", help="basis set (psi4)")
    r.add_argument("--charge", type=int, default=0)
    r.add_argument("--multiplicity", type=int, default=1)
    r.add_argument("--threads", type=int, default=1, help="threads for the backend (psi4)")

    r.add_argument("--steps", type=int, default=100)
    r.add_argument("--dt", type=float, default=0.5, help="timestep in fs (default 0.5)")
    r.add_argument("--temperature", type=float, default=300.0,
                   help="initial / target temperature in K (default 300)")
    r.add_argument("--thermostat", choices=["none", "berendsen", "langevin"], default="none")
    r.add_argument("--tau", type=float, default=100.0,
                   help="Berendsen coupling time in fs (default 100)")
    r.add_argument("--friction", type=float, default=0.01,
                   help="Langevin friction in 1/fs (default 0.01)")
    r.add_argument("--seed", type=int, default=None, help="random seed")

    r.add_argument("--trajectory", default="trajectory.xyz")
    r.add_argument("--energies", default="energies.csv")
    r.add_argument("--write-every", type=int, default=1)
    return p


def _backend_kwargs(args: argparse.Namespace) -> dict:
    if args.backend.lower() == "psi4":
        return {"method": args.method, "basis": args.basis, "num_threads": args.threads}
    return {}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "backends":
        print("\n".join(list_backends()))
        return 0

    system = MolecularSystem.from_xyz(args.xyz, args.charge, args.multiplicity)
    system.initialize_velocities(args.temperature, rng=args.seed)

    backend = get_backend(args.backend)(
        system.symbols, args.charge, args.multiplicity, **_backend_kwargs(args)
    )
    if args.thermostat == "langevin":
        integrator = LangevinBAOAB(
            backend, args.dt, args.temperature, args.friction, rng=args.seed
        )
    elif args.thermostat == "berendsen":
        integrator = VelocityVerlet(backend, args.dt, args.temperature, args.tau)
    else:
        integrator = VelocityVerlet(backend, args.dt)

    def report(rec: dict) -> None:
        if rec["step"] % args.write_every == 0:
            print(
                f"{rec['step']:7d} {rec['time_fs']:10.3f} fs  "
                f"Epot={rec['potential_Eh']:.8f}  Etot={rec['total_Eh']:.8f}  "
                f"T={rec['temperature_K']:8.2f} K",
                flush=True,
            )

    try:
        result = run_md(
            system, integrator, args.steps,
            trajectory=args.trajectory, energy_log=args.energies,
            write_every=args.write_every, callback=report,
        )
    finally:
        backend.close()

    if args.thermostat == "none":
        print(f"Max |E_tot drift| = {result.total_energy_drift:.3e} Eh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
