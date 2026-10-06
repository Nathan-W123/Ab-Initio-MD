"""
Ab initio (Born-Oppenheimer) molecular dynamics.

Nuclei move classically on the potential-energy surface of an electronic-
structure method evaluated on the fly. The public API, in Hartree atomic
units (bohr, hartree, m_e, au_time) with fs / K / angstrom at the I/O
boundary::

    from aimd import MolecularSystem, CSVR, get_backend, run_md

    system = MolecularSystem.from_xyz("examples/water.xyz")
    system.initialize_velocities(300.0, rng=1)
    backend = get_backend("hf")(system.symbols, basis="sto-3g")
    result = run_md(system, CSVR(backend, 0.5, 300.0), 200,
                    trajectory="traj.xyz", energy_log="energies.csv")

Subpackages: :mod:`aimd.backends` (force providers), :mod:`aimd.qc` (native
Hartree-Fock: integrals, SCF, gradients), :mod:`aimd.analysis` (RDF,
geometry, spectra, statistics).
"""

from aimd import analysis
from aimd.backends import (
    ForceBackend,
    GradientResult,
    backend_dependencies,
    get_backend,
    list_backends,
    register_backend,
)
from aimd.checkpoint import Checkpoint, load_checkpoint, save_checkpoint
from aimd.integrators import (
    CSVR,
    XLBOMD,
    Integrator,
    LangevinBAOAB,
    NoseHooverChain,
    VelocityVerlet,
    integrator_from_state,
)
from aimd.md import MDResult, run_md
from aimd.system import MolecularSystem
from aimd.thermostats import (
    BerendsenThermostat,
    CSVRThermostat,
    NoseHooverChainThermostat,
    Thermostat,
)
from aimd.trajectory import (
    read_dipole_log,
    read_energy_log,
    read_velocities,
    read_xyz,
)
from aimd.xlbomd import AuxiliaryDensity

__version__ = "0.2.0"

__all__ = [
    "AuxiliaryDensity",
    "BerendsenThermostat",
    "CSVR",
    "CSVRThermostat",
    "Checkpoint",
    "ForceBackend",
    "GradientResult",
    "Integrator",
    "LangevinBAOAB",
    "MDResult",
    "MolecularSystem",
    "NoseHooverChain",
    "NoseHooverChainThermostat",
    "Thermostat",
    "VelocityVerlet",
    "XLBOMD",
    "analysis",
    "backend_dependencies",
    "get_backend",
    "integrator_from_state",
    "list_backends",
    "load_checkpoint",
    "read_dipole_log",
    "read_energy_log",
    "read_velocities",
    "read_xyz",
    "register_backend",
    "run_md",
    "save_checkpoint",
]
