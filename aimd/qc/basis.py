"""
Contracted Cartesian Gaussian basis sets laid out as flat arrays.

A contracted Cartesian Gaussian (AO) on atom A at R_A is

    phi(r) = f(lx,ly,lz) * x_A^lx y_A^ly z_A^lz * sum_k c_k exp(-a_k r_A^2),
    r_A = r - R_A,  l = lx + ly + lz,

with all (l+1)(l+2)/2 Cartesian components kept for every shell (6 d,
10 f). Pople's 6-31G* was defined with 6 Cartesian d functions; cc-pVDZ was
defined with 5 spherical ones, but for simplicity (and because the extra
s-type combination x^2+y^2+z^2 only adds variational freedom) every basis
here is Cartesian. SCF energies therefore match PySCF run with
``mol.cart = True`` (and Psi4 with ``puream false``), *not* the default
spherical calculations.

Normalization: every AO, including each Cartesian component of a d or f
shell, has unit self-overlap. The contraction coefficients ``prim_coef``
normalize the axis-aligned component x^l (they include the primitive
normalization (2a/pi)^(3/4) (4a)^(l/2) / sqrt((2l-1)!!)), and component
(lx,ly,lz) carries the extra factor

    f(lx,ly,lz) = sqrt((2l-1)!! / ((2lx-1)!! (2ly-1)!! (2lz-1)!!))

(see :data:`cartesian_factor`). PySCF instead normalizes only the radial
part of Cartesian d/f functions (e.g. <d_xx|d_xx> = 4pi/5), so its AOs differ
from ours by a constant diagonal scaling; the test helper rescales.

AO order: atoms in input order; within an atom, shells sorted by l (stable,
file order otherwise; a general contraction becomes consecutive segmented
shells in column order); within a shell, Cartesian components in PySCF order
(xx, xy, xz, yy, yz, zz). This is exactly PySCF's AO order for the same
basis data.

Flat layout (numba kernels consume these directly; all int arrays are int64):

    shell_atom[s], shell_l[s]           atom index and angular momentum
    shell_prim[s]:shell_prim[s+1]       slice of prim_exp / prim_coef
    shell_ao[s]:shell_ao[s+1]           AO slice of the shell
    prim_exp[k], prim_coef[k]           exponents (bohr^-2), coefficients
    positions[A]                        nuclear positions (bohr); a shell's
                                        center is positions[shell_atom[s]]

Moving atoms only changes ``positions``; :meth:`BasisSet.with_positions`
returns a new BasisSet sharing every other array (O(1); MD calls it every
step). Arrays are read-only so cached geometry-dependent data stay valid.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from aimd.elements import ATOMIC_NUMBERS, normalize_symbol
from aimd.qc.basis_data import get_basis_shells, normalize_basis_name

_L_LABELS = "spdfghi"


def ncart(l: int) -> int:
    """Number of Cartesian components of angular momentum l."""
    return (l + 1) * (l + 2) // 2


def cartesian_components(l: int) -> list[tuple[int, int, int]]:
    """(lx, ly, lz) of a shell in PySCF order: xx, xy, xz, yy, yz, zz for l = 2."""
    return [(lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)]


def _dfact(n: int) -> int:
    """Double factorial n!!, with (-1)!! = 0!! = 1."""
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out


def cartesian_factor(lx: int, ly: int, lz: int) -> float:
    """Factor normalizing component (lx,ly,lz) relative to the x^l component."""
    l = lx + ly + lz
    return float(np.sqrt(_dfact(2 * l - 1) / (_dfact(2 * lx - 1) * _dfact(2 * ly - 1) * _dfact(2 * lz - 1))))


def normalized_contraction(l: int, exponents: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    """
    Coefficients c_k such that x^l sum_k c_k exp(-a_k r^2) has unit norm,
    given coefficients ``coefficients`` with respect to normalized primitives.
    """
    a = np.asarray(exponents, dtype=float)
    d = np.asarray(coefficients, dtype=float)
    # norm of the primitive x^l exp(-a r^2): int x^(2l) e^(-2a r^2) d^3r
    prim_norm = (2.0 * a / np.pi) ** 0.75 * (4.0 * a) ** (0.5 * l) / np.sqrt(_dfact(2 * l - 1))
    c = d * prim_norm
    p = a[:, None] + a[None, :]
    ovlp = _dfact(2 * l - 1) * np.pi ** 1.5 / ((2.0 * p) ** l * p ** 1.5)
    return c / np.sqrt(c @ ovlp @ c)


@dataclass(frozen=True)
class Shell:
    """One segmented shell (read-only view for inspection and tests)."""
    atom: int
    l: int
    center: np.ndarray        # bohr, (3,)
    exponents: np.ndarray     # bohr^-2
    coefficients: np.ndarray  # normalized, see module docstring
    ao_offset: int

    @property
    def ncart(self) -> int:
        return ncart(self.l)


class BasisSet:
    """Cartesian Gaussian basis for a molecule; see the module docstring."""

    def __init__(
        self,
        name: str,
        symbols: Sequence[str],
        positions: np.ndarray,
        shell_atom: np.ndarray,
        shell_l: np.ndarray,
        shell_prim: np.ndarray,
        prim_exp: np.ndarray,
        prim_coef: np.ndarray,
    ) -> None:
        self.name = name
        self.symbols = [normalize_symbol(s) for s in symbols]
        self.natm = len(self.symbols)
        self.atomic_numbers = _readonly(np.array([ATOMIC_NUMBERS[s] for s in self.symbols], dtype=np.int64))
        self.shell_atom = _readonly(np.asarray(shell_atom, dtype=np.int64))
        self.shell_l = _readonly(np.asarray(shell_l, dtype=np.int64))
        self.shell_prim = _readonly(np.asarray(shell_prim, dtype=np.int64))
        self.prim_exp = _readonly(np.asarray(prim_exp, dtype=float))
        self.prim_coef = _readonly(np.asarray(prim_coef, dtype=float))
        self.nshell = len(self.shell_l)
        self.nprim = len(self.prim_exp)

        nc = (self.shell_l + 1) * (self.shell_l + 2) // 2
        self.shell_ao = _readonly(np.concatenate([[0], np.cumsum(nc)]).astype(np.int64))
        self.nao = int(self.shell_ao[-1])
        self.lmax = int(self.shell_l.max()) if self.nshell else 0

        self.ao_shell = _readonly(np.repeat(np.arange(self.nshell, dtype=np.int64), nc))
        self.ao_atom = _readonly(self.shell_atom[self.ao_shell])
        self.ao_l = _readonly(self.shell_l[self.ao_shell])
        lxyz = [c for l in self.shell_l for c in cartesian_components(int(l))]
        self.ao_lxyz = _readonly(np.array(lxyz, dtype=np.int64).reshape(-1, 3))
        # AOs of atom A are ao slice atom_ao[A, 0]:atom_ao[A, 1] (contiguous)
        rng = np.zeros((self.natm, 2), dtype=np.int64)
        for a in range(self.natm):
            idx = np.nonzero(self.ao_atom == a)[0]
            rng[a] = (idx[0], idx[-1] + 1) if len(idx) else (0, 0)
        self.atom_ao = _readonly(rng)
        self._set_positions(positions)

    # ------------------------------------------------------------------ geometry
    def _set_positions(self, positions: np.ndarray) -> None:
        x = np.array(positions, dtype=float).reshape(-1, 3)
        if x.shape[0] != self.natm:
            raise ValueError(f"expected {self.natm} positions, got {x.shape[0]}")
        self.positions = _readonly(x)
        self._cache: dict = {}   # geometry-dependent data (shell pairs, Schwarz bounds)

    def with_positions(self, positions: np.ndarray) -> "BasisSet":
        """Same basis on moved atoms (bohr, (natm, 3)); shares all other arrays."""
        new = copy.copy(self)
        new._set_positions(positions)
        return new

    @property
    def shell_centers(self) -> np.ndarray:
        return self.positions[self.shell_atom]

    @property
    def nuclear_charges(self) -> np.ndarray:
        """Z_A as floats, (natm,)."""
        return self.atomic_numbers.astype(float)

    # ------------------------------------------------------------- inspection
    def shells(self) -> list[Shell]:
        out = []
        for s in range(self.nshell):
            k0, k1 = self.shell_prim[s], self.shell_prim[s + 1]
            out.append(Shell(
                atom=int(self.shell_atom[s]), l=int(self.shell_l[s]),
                center=self.positions[self.shell_atom[s]].copy(),
                exponents=self.prim_exp[k0:k1].copy(),
                coefficients=self.prim_coef[k0:k1].copy(),
                ao_offset=int(self.shell_ao[s]),
            ))
        return out

    def ao_labels(self) -> list[str]:
        """Human-readable AO labels, e.g. '0 O dxy'."""
        labels = []
        for mu in range(self.nao):
            a = int(self.ao_atom[mu])
            lx, ly, lz = (int(v) for v in self.ao_lxyz[mu])
            comp = "x" * lx + "y" * ly + "z" * lz
            labels.append(f"{a} {self.symbols[a]} {_L_LABELS[lx + ly + lz]}{comp}")
        return labels

    def __repr__(self) -> str:
        return (f"BasisSet({self.name!r}, natm={self.natm}, nshell={self.nshell}, "
                f"nao={self.nao}, nprim={self.nprim})")


def _readonly(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


def build_basis(symbols: Sequence[str], positions: np.ndarray, basis: str) -> BasisSet:
    """
    Basis ``basis`` (e.g. 'sto-3g', '6-31G*', 'cc-pVDZ'; case-insensitive) for
    atoms ``symbols`` at ``positions`` (bohr, (natm, 3)).
    """
    name = normalize_basis_name(basis)
    shell_atom, shell_l, shell_prim, prim_exp, prim_coef = [], [], [0], [], []
    for a, sym in enumerate(symbols):
        shells = get_basis_shells(name, sym)
        # stable sort by l, as PySCF orders shells within an atom
        for l, exps, coefs in sorted(shells, key=lambda sh: sh[0]):
            for col in range(coefs.shape[1]):
                keep = coefs[:, col] != 0.0   # general contractions may pad with zeros
                e = exps[keep]
                c = normalized_contraction(l, e, coefs[keep, col])
                shell_atom.append(a)
                shell_l.append(l)
                prim_exp.extend(e)
                prim_coef.extend(c)
                shell_prim.append(len(prim_exp))
    return BasisSet(name, symbols, positions, np.array(shell_atom), np.array(shell_l),
                    np.array(shell_prim), np.array(prim_exp), np.array(prim_coef))
