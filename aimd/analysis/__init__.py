"""
Trajectory analysis: pure functions of NumPy arrays in engine units (bohr,
bohr / au_time, e * bohr, m_e, hartree), returning arrays or small dataclasses.

  rdf         radial_distribution -> RDFResult (g(r), coordination numbers;
              isolated-cluster or periodic cubic-box normalisation)
  geometry    bond_lengths, bond_angles, dihedral_angles (time series)
  spectra     autocorrelation, velocity_autocorrelation -> VACF,
              vibrational_dos / ir_spectrum -> Spectrum (cm^-1),
              frequency-unit conversions, quantum correction factors
  statistics  block_average -> BlockAverage, column_statistics

With the files written by run_md (readers in aimd.trajectory)::

    from aimd.trajectory import read_dipole_log, read_energy_log, read_velocities, read_xyz
    from aimd import analysis

    traj = read_xyz("traj.xyz")
    oh = analysis.bond_lengths(traj.positions, [(0, 1), (0, 2)])     # (n_frames, 2)
    rdf = analysis.radial_distribution(traj.positions, traj.symbols, ("O", "H"), r_max=8.0)

    vel = read_velocities("vel.xyz")
    vdos = analysis.vibrational_dos(vel.velocities, vel.frame_interval_fs, masses=vel.masses)
    dip = read_dipole_log("dipole.csv")
    ir = analysis.ir_spectrum(dip.dipole, dip.frame_interval_fs, temperature_k=300.0)

    stats = analysis.column_statistics(read_energy_log("energies.csv"), skip=1000)
"""

from aimd.analysis.geometry import bond_angles, bond_lengths, dihedral_angles
from aimd.analysis.rdf import RDFResult, radial_distribution
from aimd.analysis.spectra import (
    QUANTUM_CORRECTIONS,
    VACF,
    WINDOWS,
    Spectrum,
    angular_frequency_to_wavenumber,
    autocorrelation,
    correlation_spectrum,
    ir_spectrum,
    lag_window,
    quantum_correction_factor,
    velocity_autocorrelation,
    vibrational_dos,
    wavenumber_to_angular_frequency,
)
from aimd.analysis.statistics import BlockAverage, block_average, column_statistics

__all__ = [
    "BlockAverage",
    "QUANTUM_CORRECTIONS",
    "RDFResult",
    "Spectrum",
    "VACF",
    "WINDOWS",
    "angular_frequency_to_wavenumber",
    "autocorrelation",
    "block_average",
    "bond_angles",
    "bond_lengths",
    "column_statistics",
    "correlation_spectrum",
    "dihedral_angles",
    "ir_spectrum",
    "lag_window",
    "quantum_correction_factor",
    "radial_distribution",
    "velocity_autocorrelation",
    "vibrational_dos",
    "wavenumber_to_angular_frequency",
]
