"""
PySCF reference helpers shared by the native quantum-chemistry tests (aimd.qc).

PySCF is the independent reference. ``pyscf_mole`` builds a Mole from the
*same* basis data (passed as a custom basis dict, not by name) with
``cart=True``, so it has exactly our AOs in exactly our order (checked by
``assert_same_aos``). The AOs differ only by a constant factor: PySCF
normalizes the radial part of Cartesian d/f functions (e.g. <d_xx|d_xx> =
4pi/5), we normalize every component. With ``c = ao_scale(basis)``,

    phi_ours = c * phi_pyscf,
    M_ours = c_m c_n M_pyscf          (integral matrices: matrix_to_ours)
    D_ours = D_pyscf / (c_m c_n)      (densities: density_to_ours / _to_pyscf)

Geometries are in bohr. Importing this module skips the calling test module
when PySCF is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyscf")
from pyscf import gto, scf  # noqa: E402

from aimd.qc.basis import BasisSet, normalized_contraction  # noqa: E402
from aimd.qc.basis_data import BASIS_SETS  # noqa: E402
from aimd.units import ANG_TO_BOHR  # noqa: E402

ELEMENTS_H_AR = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
                 "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"]

# Realistic geometries (angstrom in the literals, converted to bohr).
_MOLECULES_ANG: dict[str, tuple[list[str], list[list[float]]]] = {
    "water": (["O", "H", "H"],
              [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]]),
    "ammonia": (["N", "H", "H", "H"],
                [[0.0, 0.0, 0.1162], [0.0, 0.9377, -0.2711],
                 [0.8121, -0.4689, -0.2711], [-0.8121, -0.4689, -0.2711]]),
    "hcl": (["H", "Cl"], [[0.0, 0.0, -1.2027], [0.0, 0.0, 0.0723]]),
    "h2s": (["S", "H", "H"],
            [[0.0, 0.0, 0.1030], [0.0, 0.9616, -0.8239], [0.0, -0.9616, -0.8239]]),
    "nh2": (["N", "H", "H"],
            [[0.0, 0.0, 0.1414], [0.0, 0.8044, -0.4949], [0.0, -0.8044, -0.4949]]),
    "ethanol": (["C", "C", "O", "H", "H", "H", "H", "H", "H"],
                [[1.1879, -0.3829, 0.0], [0.0, 0.5526, 0.0], [-1.1867, -0.2472, 0.0],
                 [-1.9237, 0.3850, 0.0], [2.0985, 0.2306, 0.0], [1.1184, -1.0093, 0.8869],
                 [1.1184, -1.0093, -0.8869], [0.0598, 1.1956, 0.8834],
                 [0.0598, 1.1956, -0.8834]]),
    # Near-experimental equilibrium structures (bond lengths within ~0.01 A), used
    # by the SCF tests; the references are computed at exactly these geometries.
    "h2": (["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.7414]]),
    "hf": (["F", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.9168]]),
    "n2": (["N", "N"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0977]]),
    "ch4": (["C", "H", "H", "H", "H"],
            [[0.0, 0.0, 0.0], [0.6276, 0.6276, 0.6276], [-0.6276, -0.6276, 0.6276],
             [-0.6276, 0.6276, -0.6276], [0.6276, -0.6276, -0.6276]]),
    "hcn": (["H", "C", "N"], [[0.0, 0.0, -1.0655], [0.0, 0.0, 0.0], [0.0, 0.0, 1.1532]]),
    "formaldehyde": (["C", "O", "H", "H"],
                     [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2033], [0.0, 0.9339, -0.5821],
                      [0.0, -0.9339, -0.5821]]),
    "oh": (["O", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.9697]]),
    "ch2": (["C", "H", "H"], [[0.0, 0.0, 0.1], [0.0, 0.9896, -0.3207], [0.0, -0.9896, -0.3207]]),
    "o2": (["O", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2075]]),
    "no": (["N", "O"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1508]]),
    "cn": (["C", "N"], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.1718]]),
    "c2h4": (["C", "C", "H", "H", "H", "H"],
             [[0.0, 0.0, 0.6695], [0.0, 0.0, -0.6695], [0.0, 0.9289, 1.2321],
              [0.0, -0.9289, 1.2321], [0.0, 0.9289, -1.2321], [0.0, -0.9289, -1.2321]]),
    "ch3oh": (["C", "O", "H", "H", "H", "H"],
              [[-0.0467, 0.6634, 0.0], [-0.0467, -0.7570, 0.0], [-1.0919, 0.9767, 0.0],
               [0.4342, 1.0732, 0.8938], [0.4342, 1.0732, -0.8938], [0.8651, -1.0861, 0.0]]),
}


def molecule(name: str, distort: float = 0.0, seed: int = 0) -> tuple[list[str], np.ndarray]:
    """
    (symbols, positions in bohr) of ``name`` (a key of _MOLECULES_ANG: water,
    ammonia, hcl, h2s, nh2, ethanol, h2, hf, n2, ch4, hcn, formaldehyde, oh,
    ch2, o2, no, cn, c2h4, ch3oh); ``distort`` > 0 adds a seeded random
    displacement (bohr).
    """
    sym, xyz = _MOLECULES_ANG[name]
    pos = np.array(xyz) * ANG_TO_BOHR
    if distort:
        pos = pos + distort * np.random.default_rng(seed).uniform(-1.0, 1.0, pos.shape)
    return list(sym), pos


def element_cluster(symbols: list[str]) -> tuple[list[str], np.ndarray]:
    """
    Non-symmetric cluster of the given atoms (bohr): distorted triangle /
    octahedron vertices ~2.8 bohr from the origin, so every pair of elements
    interacts. Used to cover all basis data with multi-center integrals.
    """
    verts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    n = len(symbols)
    rng = np.random.default_rng(len(symbols) + sum(map(ord, "".join(symbols))))
    pos = 2.8 * verts[:n] + rng.uniform(-0.3, 0.3, (n, 3))
    return list(symbols), pos


def pyscf_mole(basis: BasisSet, charge: int = 0, spin: int | None = None) -> gto.Mole:
    """Cartesian PySCF Mole with the same atoms, positions and basis data as ``basis``."""
    atom = [(s, tuple(map(float, x))) for s, x in zip(basis.symbols, basis.positions)]
    data = {s: BASIS_SETS[basis.name][s] for s in set(basis.symbols)}
    nelec = int(basis.atomic_numbers.sum()) - charge
    mol = gto.Mole()
    mol.build(atom=atom, basis=data, unit="Bohr", cart=True, charge=charge,
              spin=(nelec % 2) if spin is None else spin, verbose=0)
    return mol


def custom_basis(symbols: list[str], positions: np.ndarray, data: dict, charge: int = 0,
                 spin: int | None = None) -> tuple[BasisSet, gto.Mole]:
    """
    Our BasisSet and the matching Cartesian PySCF Mole for basis data given
    in PySCF's format, {symbol: [[l, [exp, c1, c2, ...], ...], ...]} (extra
    coefficient columns = general contraction), e.g. a standard basis from
    BASIS_SETS plus extra diffuse shells. Shells are laid out as
    ``build_basis`` lays them out (checked against PySCF's AO order).
    """
    shell_atom, shell_l, shell_prim, prim_exp, prim_coef = [], [], [0], [], []
    for a, sym in enumerate(symbols):
        for entry in sorted(data[sym], key=lambda sh: sh[0]):
            l = int(entry[0])
            rows = np.array(entry[1:], dtype=float)
            for col in range(1, rows.shape[1]):
                keep = rows[:, col] != 0.0
                exps = rows[keep, 0]
                shell_atom.append(a)
                shell_l.append(l)
                prim_exp.extend(exps)
                prim_coef.extend(normalized_contraction(l, exps, rows[keep, col]))
                shell_prim.append(len(prim_exp))
    pos = np.asarray(positions, dtype=float)
    basis = BasisSet("custom", symbols, pos, np.array(shell_atom), np.array(shell_l),
                     np.array(shell_prim), np.array(prim_exp), np.array(prim_coef))
    nelec = int(basis.atomic_numbers.sum()) - charge
    mol = gto.Mole()
    mol.build(atom=[(s, tuple(map(float, x))) for s, x in zip(symbols, pos)],
              basis={s: data[s] for s in set(symbols)}, unit="Bohr", cart=True, charge=charge,
              spin=(nelec % 2) if spin is None else spin, verbose=0)
    assert_same_aos(basis, mol)
    return basis, mol


def _dfact(n: int) -> int:
    out = 1
    while n > 1:
        out *= n
        n -= 2
    return out


def ao_scale(basis: BasisSet) -> np.ndarray:
    """
    c_m with phi_ours = c_m phi_pyscf: PySCF's Cartesian component (lx,ly,lz)
    of l >= 2 has <phi|phi> = 4pi (2lx-1)!!(2ly-1)!!(2lz-1)!! / (2l+1)!!;
    s and p functions are unit-normalized in both codes.
    """
    c = np.ones(basis.nao)
    for m, (lx, ly, lz) in enumerate(basis.ao_lxyz):
        l = lx + ly + lz
        if l >= 2:
            norm2 = 4.0 * np.pi * _dfact(2 * lx - 1) * _dfact(2 * ly - 1) * _dfact(2 * lz - 1) / _dfact(2 * l + 1)
            c[m] = 1.0 / np.sqrt(norm2)
    return c


def matrix_to_ours(M: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Scale PySCF AO integral matrices (..., nao, nao) to our normalization."""
    return M * np.outer(c, c)


def eri_to_ours(eri: np.ndarray, c: np.ndarray) -> np.ndarray:
    return eri * np.einsum("i,j,k,l->ijkl", c, c, c, c)


def density_to_pyscf(D: np.ndarray, c: np.ndarray) -> np.ndarray:
    return D * np.outer(c, c)


def density_to_ours(D: np.ndarray, c: np.ndarray) -> np.ndarray:
    return D / np.outer(c, c)


def assert_same_aos(basis: BasisSet, mol: gto.Mole) -> None:
    """Same AO count, atom, angular momentum and Cartesian component, in order."""
    assert mol.nao == basis.nao
    labels = mol.ao_labels(fmt=False)
    for m, (atom, _sym, _nl, comp) in enumerate(labels):
        assert atom == basis.ao_atom[m]
        lx, ly, lz = basis.ao_lxyz[m]
        assert comp == "x" * lx + "y" * ly + "z" * lz, (m, comp, basis.ao_lxyz[m])


def random_symmetric(n: int, rng: np.random.Generator, scale: float = 0.5) -> np.ndarray:
    A = rng.uniform(-scale, scale, (n, n))
    return A + A.T


def run_scf(mol: gto.Mole, unrestricted: bool = False, conv_tol: float = 1e-12, dm0=None,
            conv_tol_grad: float | None = None):
    """
    Converged PySCF RHF/UHF object, optionally started from the PySCF-normalized
    density ``dm0``. PySCF's orbital-gradient threshold defaults to
    sqrt(conv_tol) (1e-6 here): pass ``conv_tol_grad`` when comparing densities,
    orbitals or properties more tightly than that.
    """
    mf = (scf.UHF if unrestricted else scf.RHF)(mol)
    mf.conv_tol = conv_tol
    if conv_tol_grad is not None:
        mf.conv_tol_grad = conv_tol_grad
    mf.kernel(dm0=dm0)
    assert mf.converged
    return mf


def energy_weighted_density(mf) -> np.ndarray:
    """W = sum_i n_i eps_i C_mi C_ni (PySCF AO basis), alpha + beta for UHF."""
    occ = np.asarray(mf.mo_occ)
    e = np.asarray(mf.mo_energy)
    C = np.asarray(mf.mo_coeff)
    if C.ndim == 3:
        return sum((C[s] * (occ[s] * e[s])) @ C[s].T for s in range(2))
    return (C * (occ * e)) @ C.T
