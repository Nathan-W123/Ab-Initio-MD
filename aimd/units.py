"""
Unit conventions and conversion factors.

Everything inside the engine is in Hartree atomic units:
  - length:  bohr
  - energy:  hartree
  - mass:    electron mass (m_e)
  - time:    atomic time unit (hbar / E_h ~ 0.0242 fs)

Conversions to and from user-facing units (angstrom, amu, fs, kelvin) happen
only at the I/O boundary (XYZ files, CLI arguments, log output).
"""

# CODATA 2018
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG

AMU_TO_AU = 1822.888486209          # unified atomic mass unit -> electron masses

AU_TIME_TO_FS = 2.4188843265857e-2  # atomic time unit -> femtoseconds
FS_TO_AU_TIME = 1.0 / AU_TIME_TO_FS

KB_AU = 3.166811563455546e-6        # Boltzmann constant, hartree / kelvin

HARTREE_TO_EV = 27.211386245988
HARTREE_TO_KCALMOL = 627.5094740631
