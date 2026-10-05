"""
Element data: atomic numbers and standard atomic masses (amu) for H-Kr.

Masses are the IUPAC standard atomic weights (most-abundant-isotope mixtures),
which is what MD of natural-abundance molecules wants.
"""

from __future__ import annotations

_DATA = [
    ("H", 1.008), ("He", 4.002602),
    ("Li", 6.94), ("Be", 9.0121831), ("B", 10.81), ("C", 12.011),
    ("N", 14.007), ("O", 15.999), ("F", 18.998403163), ("Ne", 20.1797),
    ("Na", 22.98976928), ("Mg", 24.305), ("Al", 26.9815385), ("Si", 28.085),
    ("P", 30.973761998), ("S", 32.06), ("Cl", 35.45), ("Ar", 39.948),
    ("K", 39.0983), ("Ca", 40.078), ("Sc", 44.955908), ("Ti", 47.867),
    ("V", 50.9415), ("Cr", 51.9961), ("Mn", 54.938044), ("Fe", 55.845),
    ("Co", 58.933194), ("Ni", 58.6934), ("Cu", 63.546), ("Zn", 65.38),
    ("Ga", 69.723), ("Ge", 72.630), ("As", 74.921595), ("Se", 78.971),
    ("Br", 79.904), ("Kr", 83.798),
]

ATOMIC_NUMBERS: dict[str, int] = {sym: z for z, (sym, _) in enumerate(_DATA, start=1)}
ATOMIC_MASSES_AMU: dict[str, float] = {sym: m for sym, m in _DATA}


def normalize_symbol(symbol: str) -> str:
    """Return the canonical capitalisation of an element symbol ('cl' -> 'Cl')."""
    s = symbol.strip()
    sym = s[:1].upper() + s[1:].lower()
    if sym not in ATOMIC_NUMBERS:
        raise ValueError(f"Unknown element symbol: {symbol!r}")
    return sym
