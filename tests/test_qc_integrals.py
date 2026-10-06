"""
Tests for aimd.qc: basis-set data, basis construction, the Boys function,
one- and two-electron integrals and their nuclear-derivative contractions.

Every numerical check uses a reference independent of the code under test:
  - PySCF integrals (int1e_ovlp/kin/nuc/r, int2e) and derivative integrals
    (int1e_ipovlp/ipkin/ipnuc/iprinv, int2e_ip1) for a Cartesian Mole built
    from the same basis data (helpers in tests/qc_reference.py);
  - PySCF analytic RHF/UHF gradients, fed with PySCF's own converged densities;
  - fourth-order central finite differences of the energy expressions at
    fixed (random, symmetric) densities;
  - a 60-digit series for the Boys function.
"""

from __future__ import annotations

import math
from decimal import Decimal, localcontext

import numba
import numpy as np
import pytest
import scipy.linalg

import qc_reference as ref
from pyscf import gto
from pyscf.grad import rhf as pyscf_rhf_grad

from aimd.qc import integrals as I
from aimd.qc.basis import BasisSet, build_basis, cartesian_components, normalized_contraction
from aimd.qc.basis_data import BASIS_SETS, available_basis_sets, normalize_basis_name
from aimd.qc.boys import NMAX, X_SWITCH, boys

BASES = ["sto-3g", "6-31g", "6-31g*", "6-31g**", "cc-pvdz"]
# Element clusters covering H-Ar, three atoms each (every element pair inside
# a cluster interacts; see qc_reference.element_cluster).
CLUSTERS = [ref.ELEMENTS_H_AR[i:i + 3] for i in range(0, 18, 3)]


def _basis(name: str, basis: str, distort: float = 0.0, seed: int = 0):
    sym, pos = ref.molecule(name, distort=distort, seed=seed)
    return build_basis(sym, pos, basis)


# ---------------------------------------------------------------- basis data

@pytest.mark.parametrize("name", BASES)
def test_basis_data_identical_to_pyscf_library(name):
    # The embedded data were cross-checked against PySCF's own copies; keep
    # them bitwise identical (exponents and every contraction column).
    assert available_basis_sets() == BASES
    for el in ref.ELEMENTS_H_AR:
        assert BASIS_SETS[name][el] == gto.basis.load(name, el), (name, el)


def test_basis_names_are_case_insensitive_and_errors_are_clear():
    assert normalize_basis_name("6-31G*") == "6-31g*"
    assert normalize_basis_name(" 6-31G(d,p) ") == "6-31g**"
    assert normalize_basis_name("STO-3G") == "sto-3g"
    assert normalize_basis_name("cc-pVDZ") == "cc-pvdz"
    a = build_basis(["O", "H"], [[0, 0, 0], [0, 0, 1.8]], "6-31G*")
    b = build_basis(["o", "h"], [[0, 0, 0], [0, 0, 1.8]], "6-31g(d)")
    assert a.nao == b.nao == 17
    np.testing.assert_array_equal(a.prim_coef, b.prim_coef)
    with pytest.raises(ValueError, match="Unknown basis set 'def2-svp'"):
        build_basis(["H"], [[0, 0, 0]], "def2-svp")
    with pytest.raises(ValueError, match="no data for element 'K'"):
        build_basis(["K"], [[0, 0, 0]], "sto-3g")
    with pytest.raises(ValueError, match="Unknown element"):
        build_basis(["Xx"], [[0, 0, 0]], "sto-3g")
    with pytest.raises(ValueError, match="expected 2 positions"):
        build_basis(["H", "H"], [[0, 0, 0]], "sto-3g")


def test_basis_layout():
    b = _basis("water", "6-31g*")
    # O: 3s 2p 1d (6 Cartesian d) = 3 + 6 + 6; H: 2s each
    assert (b.nao, b.nshell, b.natm) == (19, 10, 3)
    assert list(b.shell_l) == [0, 0, 0, 1, 1, 2, 0, 0, 0, 0]
    assert list(b.shell_atom) == [0] * 6 + [1, 1, 2, 2]
    np.testing.assert_array_equal(np.diff(b.shell_ao), (b.shell_l + 1) * (b.shell_l + 2) // 2)
    np.testing.assert_array_equal(b.atom_ao, [[0, 15], [15, 17], [17, 19]])
    np.testing.assert_array_equal(b.ao_lxyz[9:15], cartesian_components(2))
    assert [tuple(c) for c in cartesian_components(2)] == [
        (2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    assert b.ao_labels()[10] == "0 O dxy"
    # O 1s has 6 primitives in 6-31G
    assert b.shell_prim[1] - b.shell_prim[0] == 6
    with pytest.raises(ValueError):
        b.positions[0, 0] = 1.0        # read-only: cached pair data stay valid


def test_with_positions_moves_basis_without_copying():
    b = _basis("water", "6-31g*")
    new = b.positions + np.array([[0.1, -0.2, 0.05], [0.0, 0.3, 0.0], [-0.1, 0.0, 0.2]])
    m = b.with_positions(new)
    assert np.shares_memory(m.prim_coef, b.prim_coef)
    assert np.shares_memory(m.shell_l, b.shell_l)
    np.testing.assert_array_equal(b.positions, _basis("water", "6-31g*").positions)
    fresh = build_basis(b.symbols, new, "6-31g*")
    np.testing.assert_array_equal(I.overlap_matrix(m), I.overlap_matrix(fresh))
    np.testing.assert_array_equal(I.eri_tensor(m), I.eri_tensor(fresh))
    # per-geometry caches are not shared
    assert not np.array_equal(I.schwarz_bounds(m), I.schwarz_bounds(b))


# ---------------------------------------------------------------- Boys function

def _boys_reference(n: int, x: float) -> float:
    """F_n(x) = exp(-x) sum_k (2x)^k / ((2n+1)(2n+3)...(2n+2k+1)), 60 digits."""
    with localcontext() as ctx:
        ctx.prec = 60
        X = Decimal(float(x))          # exact binary value
        term = Decimal(1) / (2 * n + 1)
        total = term
        k = 1
        while True:
            term = term * 2 * X / (2 * n + 2 * k + 1)
            total += term
            k += 1
            if term < total * Decimal("1e-40"):
                break
        return float((-X).exp() * total)


def test_boys_function_against_high_precision_series():
    rng = np.random.default_rng(11)
    xs = list(10 ** rng.uniform(-8, math.log10(150.0), 60))
    xs += [0.0, 0.05, 0.049999999, 1.0, X_SWITCH - 1e-9, X_SWITCH, X_SWITCH + 1e-9, 1000.0]
    worst = 0.0
    for x in xs:
        f = boys(24, x)
        for n in range(25):
            worst = max(worst, abs(f[n] / _boys_reference(n, x) - 1.0))
    # measured 3.8e-15; the table/Taylor truncation error is < 1e-19
    assert worst < 2e-14
    np.testing.assert_allclose(boys(NMAX, 0.0), 1.0 / (2 * np.arange(NMAX + 1) + 1), rtol=1e-15)
    with pytest.raises(ValueError):
        boys(NMAX + 1, 1.0)


def _boys_asymptotic(n: int, x: float) -> float:
    """
    F_n(x) = (2n-1)!! sqrt(pi) / (2^(n+1) x^(n+1/2)) to 60 digits; the
    neglected incomplete-gamma tail, relatively ~x^(n-1/2) exp(-x) / Gamma(n+1/2),
    is below 1e-370 for x >= 1000 and n <= 32.
    """
    with localcontext() as ctx:
        ctx.prec = 60
        X = Decimal(float(x))
        pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494")
        return float(math.prod(range(1, 2 * n, 2)) * pi.sqrt() / (2 ** (n + 1) * X ** n * X.sqrt()))


def test_boys_function_all_orders_and_huge_arguments():
    # Regression: boys() recursed down from F_NMAX, which underflows to 0 for
    # x > 3e10 and then zeroed every order (F_0(1e12) came out as 0).
    worst = 0.0
    for x in [1e3, 1e6, 1e10, 1e12, 1e20, 1e100]:
        f = boys(NMAX, x)
        for n in range(NMAX + 1):
            r = _boys_asymptotic(n, x)
            if r < 1e-300:                     # true value is subnormal or 0
                assert 0.0 <= f[n] < 1e-300
            else:
                worst = max(worst, abs(f[n] / r - 1.0))
    # every order up to NMAX on both branches, against the 60-digit series
    for x in [0.0, 1e-9, 0.37, 7.77, 23.45, X_SWITCH - 1e-6, X_SWITCH, 45.0, 150.0]:
        f = boys(NMAX, x)
        for n in range(NMAX + 1):
            worst = max(worst, abs(f[n] / _boys_reference(n, x) - 1.0))
    assert worst < 2e-14                       # measured 2.4e-15


# ---------------------------------------------------------------- one-electron

@pytest.mark.parametrize("basis", BASES)
@pytest.mark.parametrize("elements", CLUSTERS, ids=lambda e: "".join(e))
def test_one_electron_integrals_vs_pyscf(basis, elements):
    sym, pos = ref.element_cluster(elements)
    b = build_basis(sym, pos, basis)
    mol = ref.pyscf_mole(b)
    ref.assert_same_aos(b, mol)
    c = ref.ao_scale(b)
    S, T, V = I.one_electron_matrices(b)
    # measured over all 30 cases: diag(S) 1.1e-15, S 1.6e-15, r 4.4e-15; T and V
    # errors grow with the core values (|T| <= 160, |V| <= 320 for Ar): measured
    # |dT| <= 6e-15 max(1, |T|), |dV| <= 1e-13 max(1, |V|)
    np.testing.assert_allclose(np.diag(S), 1.0, rtol=0, atol=1e-14)   # every AO normalized
    np.testing.assert_allclose(S, ref.matrix_to_ours(mol.intor("int1e_ovlp"), c), rtol=0, atol=2e-14)
    np.testing.assert_allclose(T, ref.matrix_to_ours(mol.intor("int1e_kin"), c), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(V, ref.matrix_to_ours(mol.intor("int1e_nuc"), c), rtol=1e-12, atol=1e-12)
    origin = np.array([0.3, -0.2, 0.5])
    with mol.with_common_orig(origin):
        r = ref.matrix_to_ours(mol.intor("int1e_r"), c)
    np.testing.assert_allclose(I.dipole_integrals(b, origin), r, rtol=0, atol=5e-14)


def test_nuclear_repulsion_and_dipole_vs_pyscf():
    b = _basis("ethanol", "sto-3g", distort=0.1)
    mol = ref.pyscf_mole(b)
    Z = b.nuclear_charges
    assert I.nuclear_repulsion_energy(Z, b.positions) == pytest.approx(mol.energy_nuc(), rel=1e-14)
    np.testing.assert_allclose(I.nuclear_repulsion_gradient(Z, b.positions),
                               pyscf_rhf_grad.grad_nuc(mol), rtol=0, atol=1e-12)
    o = np.array([0.2, 0.1, -0.3])
    np.testing.assert_allclose(I.nuclear_dipole(Z, b.positions, o),
                               mol.atom_charges() @ (mol.atom_coords() - o), atol=1e-12)


# ---------------------------------------------------------------- ERIs

# every element H-Ar appears in some ERI case; clusters kept <= 45 AOs for speed
ERI_CASES = [
    ("water", "sto-3g"), ("water", "6-31g*"), ("ammonia", "cc-pvdz"), ("hcl", "6-31g**"),
    (("H", "He", "Li"), "cc-pvdz"), (("Be", "B", "C"), "6-31g**"), (("N", "O", "F", "Ne"), "6-31g"),
    (("Na", "Mg"), "6-31g*"), (("Al", "Si"), "cc-pvdz"), (("P", "S", "Cl", "Ar"), "sto-3g"),
]


@pytest.mark.parametrize("system,basis", ERI_CASES, ids=lambda v: v if isinstance(v, str) else "".join(v))
def test_eri_tensor_vs_pyscf(system, basis):
    if isinstance(system, str):
        b = _basis(system, basis, distort=0.05)
    else:
        b = build_basis(*ref.element_cluster(list(system)), basis)
    mol = ref.pyscf_mole(b)
    eri = I.eri_tensor(b)
    expect = ref.eri_to_ours(mol.intor("int2e"), ref.ao_scale(b))
    # the default Schwarz threshold 1e-12 bounds each dropped integral
    # strictly (measured 8.4e-13); computed integrals agree to 4.2e-14 (|ERI| <= 11)
    np.testing.assert_allclose(eri, expect, rtol=1e-13, atol=1e-12)


def test_schwarz_screening_is_a_strict_bound():
    # two waters 12 bohr apart: many quartets fall below the threshold
    sym, pos = ref.molecule("water")
    b = build_basis(sym + sym, np.vstack([pos, pos + [0.0, 12.0, 3.0]]), "6-31g*")
    exact = I.eri_tensor(b, schwarz_threshold=0.0)
    thr = 1e-6
    screened = I.eri_tensor(b, schwarz_threshold=thr)
    diff = np.abs(screened - exact)
    assert diff.max() < thr
    assert np.count_nonzero(diff) > 1000    # screening actually dropped quartets


def test_jk_from_eri_matches_einsum():
    b = _basis("ammonia", "6-31g*", distort=0.05)
    eri = I.eri_tensor(b)
    rng = np.random.default_rng(5)
    D = rng.normal(size=(2, b.nao, b.nao))       # not symmetric on purpose
    J, K = I.jk_from_eri(eri, D)
    np.testing.assert_allclose(J, np.einsum("ijkl,skl->sij", eri, D), atol=1e-12)
    np.testing.assert_allclose(K, np.einsum("ikjl,skl->sij", eri, D), atol=1e-12)
    J0, K0 = I.jk_from_eri(eri, D[0])
    np.testing.assert_array_equal(J0, J[0])
    np.testing.assert_array_equal(K0, K[0])


# ---------------------------------------------------------------- derivatives

def _per_atom(mol, M):
    """g[A] = sum_{m on A} sum_n M[:, m, n]."""
    g = np.zeros((mol.natm, 3))
    for a, (_s0, _s1, p0, p1) in enumerate(mol.aoslice_by_atom()):
        g[a] = M[:, p0:p1].sum(axis=(1, 2))
    return g


def _gamma(Dc, Dxs, k):
    """Symmetrized two-particle density of E2 (see integrals.two_electron_gradient)."""
    G = np.einsum("ij,kl->ijkl", Dc, Dc)
    for X in Dxs:
        G -= 0.5 * k * (np.einsum("ik,jl->ijkl", X, X) + np.einsum("il,jk->ijkl", X, X))
    return G


def _pyscf_derivative_contractions(mol, D, W):
    """
    d/dR tr(W S), tr(D T), tr(D V) from PySCF derivative integrals (D, W
    symmetric, PySCF normalization). <nabla m|n> differentiates the bra
    function w.r.t. the electron coordinate, i.e. -d/dA; the ket term doubles
    it; the operator centers add d/dC <m|1/r_C|n> (int1e_iprinv).
    """
    gS = -2 * _per_atom(mol, mol.intor("int1e_ipovlp") * W)
    gT = -2 * _per_atom(mol, mol.intor("int1e_ipkin") * D)
    gV = -2 * _per_atom(mol, mol.intor("int1e_ipnuc") * D)
    for a in range(mol.natm):
        with mol.with_rinv_at_nucleus(a):
            gV[a] -= 2 * mol.atom_charge(a) * np.einsum("xij,ij->x", mol.intor("int1e_iprinv"), D)
    return gS, gT, gV


@pytest.mark.parametrize("name,basis", [("water", "6-31g*"), ("ammonia", "cc-pvdz"), ("hcl", "6-31g**")])
def test_derivative_contractions_vs_pyscf_derivative_integrals(name, basis):
    b = _basis(name, basis, distort=0.05, seed=4)
    mol = ref.pyscf_mole(b)
    c = ref.ao_scale(b)
    rng = np.random.default_rng(7)
    D, W, Da, Db = (ref.random_symmetric(b.nao, rng) for _ in range(4))
    Dp, Wp, Dap, Dbp = (ref.density_to_pyscf(X, c) for X in (D, W, Da, Db))

    gS, gT, gV = _pyscf_derivative_contractions(mol, Dp, Wp)
    tol = dict(rtol=0, atol=5e-12)     # measured <= 3.2e-13 (values up to ~10)
    np.testing.assert_allclose(I.overlap_gradient(b, W), gS, **tol)
    np.testing.assert_allclose(I.kinetic_gradient(b, D), gT, **tol)
    np.testing.assert_allclose(I.nuclear_attraction_gradient(b, D), gV, **tol)
    np.testing.assert_allclose(I.one_electron_gradient(b, D, W), gT + gV - gS, **tol)

    ip1 = mol.intor("int2e_ip1")
    for Dc, Dxs, k, ours in [
        (Dp, [Dp], 0.5, I.two_electron_gradient(b, D, [D], 0.5)),                 # RHF-like
        (Dap + Dbp, [Dap, Dbp], 1.0, I.two_electron_gradient(b, Da + Db, [Da, Db], 1.0)),  # UHF-like
        (Dp, [], 0.0, I.two_electron_gradient(b, D, [], 0.0)),                    # Coulomb only
    ]:
        expect = -2 * _per_atom(mol, np.einsum("xijkl,ijkl->xij", ip1, _gamma(Dc, Dxs, k)))
        np.testing.assert_allclose(ours, expect, **tol)
        # translational invariance (not imposed by the 4-center passes); measured 5e-15
        np.testing.assert_allclose(ours.sum(axis=0), 0.0, atol=1e-13)


def test_f_and_g_shells_vs_pyscf():
    # No shipped basis goes beyond d, but the kernels claim LMAX = 4 (g):
    # build f and g shells (contracted, on different atoms) by hand, the
    # way build_basis does, and compare everything with PySCF.
    data = {"He": [[0, [2.0, 1.0]], [1, [3.0, 0.4], [0.7, 0.7]], [3, [2.5, 0.4], [0.9, 0.6]]],
            "H": [[0, [1.5, 1.0]], [2, [1.1, 1.0]], [4, [1.3, 1.0]]]}
    sym = ["He", "H"]
    pos = np.array([[0.1, -0.2, 0.0], [0.3, 0.9, 1.6]])
    shell_atom, shell_l, shell_prim, exps, coefs = [], [], [0], [], []
    for a, s in enumerate(sym):
        for l, *rows in data[s]:
            rows = np.array(rows)
            shell_atom.append(a)
            shell_l.append(l)
            exps.extend(rows[:, 0])
            coefs.extend(normalized_contraction(l, rows[:, 0], rows[:, 1]))
            shell_prim.append(len(exps))
    b = BasisSet("custom", sym, pos, np.array(shell_atom), np.array(shell_l), np.array(shell_prim),
                 np.array(exps), np.array(coefs))
    assert (b.lmax, b.nao) == (4, 36)
    mol = gto.M(atom=list(zip(sym, map(tuple, pos))), basis=data, unit="Bohr", cart=True, spin=1, verbose=0)
    ref.assert_same_aos(b, mol)
    c = ref.ao_scale(b)
    # measured: S 1.3e-15, T 1.1e-14, V 5.6e-14, r 3.8e-15, ERI 1.5e-14 (values <= 30);
    # gradient contractions <= 2.1e-13 (values ~6)
    tol = dict(rtol=0, atol=5e-13)
    gtol = dict(rtol=0, atol=2e-12)
    S, T, V = I.one_electron_matrices(b)
    np.testing.assert_allclose(np.diag(S), 1.0, **tol)
    np.testing.assert_allclose(S, ref.matrix_to_ours(mol.intor("int1e_ovlp"), c), **tol)
    np.testing.assert_allclose(T, ref.matrix_to_ours(mol.intor("int1e_kin"), c), **tol)
    np.testing.assert_allclose(V, ref.matrix_to_ours(mol.intor("int1e_nuc"), c), **tol)
    np.testing.assert_allclose(I.dipole_integrals(b), ref.matrix_to_ours(mol.intor("int1e_r"), c), **tol)
    np.testing.assert_allclose(I.eri_tensor(b, schwarz_threshold=0.0),
                               ref.eri_to_ours(mol.intor("int2e"), c), **tol)
    rng = np.random.default_rng(17)
    D, W = (ref.random_symmetric(b.nao, rng) for _ in range(2))
    gS, gT, gV = _pyscf_derivative_contractions(mol, ref.density_to_pyscf(D, c), ref.density_to_pyscf(W, c))
    np.testing.assert_allclose(I.overlap_gradient(b, W), gS, **gtol)
    np.testing.assert_allclose(I.kinetic_gradient(b, D), gT, **gtol)
    np.testing.assert_allclose(I.nuclear_attraction_gradient(b, D), gV, **gtol)
    Dp = ref.density_to_pyscf(D, c)
    expect = -2 * _per_atom(mol, np.einsum("xijkl,ijkl->xij", mol.intor("int2e_ip1"), _gamma(Dp, [Dp], 0.5)))
    np.testing.assert_allclose(I.two_electron_gradient(b, D, [D], 0.5, schwarz_threshold=0.0), expect, **gtol)


def test_general_contractions_form_blocks_that_share_primitives():
    # Regression (performance): every column of a general contraction used to
    # become its own segmented shell carrying all shared primitives, so the ERI
    # kernels recomputed each primitive quartet once per column combination
    # (Cl / cc-pVDZ: three 11-primitive s shells -> 81 copies of each s
    # quartet; PCl3 cc-pVDZ step 9 s vs 1.6 s now). The kernels now run over
    # contraction blocks; check that every primitive pair is stored once.
    b = _basis("hcl", "cc-pvdz")
    cl = [B for B in range(b.nblock) if b.blk_atom[B] == 1]
    # Cl: s (11 prims x 3 contr.), s (diffuse), p (7 x 2), p, d
    assert [(int(b.blk_l[B]), int(b.blk_ncon[B]), int(b.blk_prim[B + 1] - b.blk_prim[B]))
            for B in cl] == [(0, 3, 11), (0, 1, 1), (1, 2, 7), (1, 1, 1), (2, 1, 1)]
    assert b.nshell == 11 and b.nblock == 8 and b.nao == 24      # AO layout unchanged
    np.testing.assert_array_equal(b.blk_ao, b.shell_ao[b.blk_shell[:-1]])
    pr = I._pairs(b)
    seen = set()
    for ip, (i, j) in enumerate(pr.pair_blocks):
        for k in range(pr.pair_pp[ip], pr.pair_pp[ip + 1]):
            key = (int(b.blk_atom[i]), int(b.blk_l[i]), pr.pp_ab[k, 0],
                   int(b.blk_atom[j]), int(b.blk_l[j]), pr.pp_ab[k, 1])
            assert key not in seen
            seen.add(key)
    # Pople split-valence shells have disjoint exponents: one block per shell
    p = _basis("hcl", "6-31g*")
    assert p.nblock == p.nshell and np.all(p.blk_ncon == 1)


def test_general_contractions_vs_pyscf():
    # General contractions in s, p and d (zero-padded columns, an uncontracted
    # shell repeating one of the exponents -> merged into the block, and a
    # disjoint shell -> its own block) on three atoms: ERIs and the ERI
    # gradient against PySCF, covering general bras and kets of both
    # gradient contraction orders.
    data = {
        "O": [[0, [30.0, 0.2, 0.0], [6.0, 0.5, -0.2], [1.2, 0.4, 0.6]], [0, [1.2, 1.0]],
              [1, [5.0, 0.3, 0.1], [1.1, 0.6, -0.5], [0.3, 0.3, 0.9]],
              [2, [1.6, 0.7, 0.2], [0.5, 0.4, 0.8]], [2, [0.5, 1.0]], [2, [0.2, 1.0]]],
        "H": [[0, [4.0, 0.3, 0.1], [0.8, 0.7, 0.4], [0.2, 0.0, 0.8]], [1, [0.9, 1.0]]],
    }
    sym = ["O", "H", "H"]
    pos = np.array([[0.0, 0.1, -0.1], [1.5, 1.1, 0.2], [-1.4, 0.9, 0.4]])
    b, mol = ref.custom_basis(sym, pos, data)
    o = [(int(b.blk_l[B]), int(b.blk_ncon[B])) for B in range(b.nblock) if b.blk_atom[B] == 0]
    assert o == [(0, 3), (1, 2), (2, 3), (2, 1)]
    assert int(b.blk_ncon.max()) == 3 and (b.nshell, b.nblock) == (15, 8)
    c = ref.ao_scale(b)
    # measured: ERI 4.4e-14 (values <= 2.5), gradients 6.7e-13 (values <= 33)
    np.testing.assert_allclose(I.eri_tensor(b, schwarz_threshold=0.0),
                               ref.eri_to_ours(mol.intor("int2e"), c), rtol=0, atol=2e-13)
    rng = np.random.default_rng(23)
    Da, Db = (ref.random_symmetric(b.nao, rng) for _ in range(2))
    Dap, Dbp = (ref.density_to_pyscf(X, c) for X in (Da, Db))
    ip1 = mol.intor("int2e_ip1")
    for Dc, Dxs, k, ours in [
        (Dap, [Dap], 0.5, I.two_electron_gradient(b, Da, [Da], 0.5, schwarz_threshold=0.0)),
        (Dap + Dbp, [Dap, Dbp], 1.0,
         I.two_electron_gradient(b, Da + Db, [Da, Db], 1.0, schwarz_threshold=0.0)),
    ]:
        expect = -2 * _per_atom(mol, np.einsum("xijkl,ijkl->xij", ip1, _gamma(Dc, Dxs, k)))
        np.testing.assert_allclose(ours, expect, rtol=0, atol=3e-12)


def test_nuclear_attraction_with_custom_charges_vs_pyscf():
    # V and its gradient for point charges q_A (not Z_A) on the atoms:
    # V = -sum_A q_A <m|1/r_A|n>; d/dR includes the moving charges.
    b = _basis("hcl", "6-31g*", distort=0.05, seed=6)
    mol = ref.pyscf_mole(b)
    c = ref.ao_scale(b)
    q = np.array([0.7, -2.5])
    D = ref.random_symmetric(b.nao, np.random.default_rng(8))
    Dp = ref.density_to_pyscf(D, c)
    V = np.zeros((b.nao, b.nao))
    dV = np.zeros((3, b.nao, b.nao))          # <nabla m| sum_A -q_A / r_A |n>
    g = np.zeros((b.natm, 3))
    for a in range(b.natm):
        with mol.with_rinv_at_nucleus(a):
            V -= q[a] * mol.intor("int1e_rinv")
            ip = mol.intor("int1e_iprinv")
            dV -= q[a] * ip
            g[a] -= 2 * q[a] * np.einsum("xij,ij->x", ip, Dp)     # operator center
    g -= 2 * _per_atom(mol, dV * Dp)                                # basis centers
    # measured 2.1e-14 (V), 9.6e-14 (gradient, values ~5)
    np.testing.assert_allclose(I.nuclear_attraction_matrix(b, q), ref.matrix_to_ours(V, c), rtol=0, atol=5e-13)
    np.testing.assert_allclose(I.nuclear_attraction_gradient(b, D, q), g, rtol=0, atol=1e-12)


def _fd_gradient(energy, x0: np.ndarray, h: float = 1e-3) -> np.ndarray:
    """Fourth-order central differences of a vector of energies, (natm, 3, nE)."""
    out = None
    for a in range(x0.shape[0]):
        for k in range(3):
            e = []
            for s in (2, 1, -1, -2):
                x = x0.copy()
                x[a, k] += s * h
                e.append(np.asarray(energy(x)))
            d = (-e[0] + 8 * e[1] - 8 * e[2] + e[3]) / (12 * h)
            if out is None:
                out = np.zeros(x0.shape + d.shape)
            out[a, k] = d
    return out


@pytest.mark.parametrize("name,basis", [("water", "6-31g*"), ("hcl", "6-31g**")])
def test_derivative_contractions_vs_finite_differences(name, basis):
    b = _basis(name, basis, distort=0.05, seed=1)
    rng = np.random.default_rng(3)
    D, W, Da, Db = (ref.random_symmetric(b.nao, rng) for _ in range(4))

    def energies(x):
        m = b.with_positions(x)
        S, T, V = I.one_electron_matrices(m)
        eri = I.eri_tensor(m, schwarz_threshold=0.0)
        j = lambda X: np.einsum("ij,kl,ijkl->", X, X, eri)
        kx = lambda X: np.einsum("ik,jl,ijkl->", X, X, eri)
        return [np.sum(W * S), np.sum(D * T), np.sum(D * V),
                0.5 * j(D) - 0.25 * kx(D),
                0.5 * j(Da + Db) - 0.5 * (kx(Da) + kx(Db))]

    fd = _fd_gradient(energies, b.positions)
    analytic = [I.overlap_gradient(b, W), I.kinetic_gradient(b, D), I.nuclear_attraction_gradient(b, D),
                I.two_electron_gradient(b, D, [D], 0.5),
                I.two_electron_gradient(b, Da + Db, [Da, Db], 1.0)]
    for i, g in enumerate(analytic):
        # measured <= 1.3e-10 (h = 1e-3, round-off limited); gradients are O(1-10)
        np.testing.assert_allclose(g, fd[..., i], rtol=0, atol=1e-9, err_msg=f"term {i}")


@pytest.mark.parametrize("name,basis,unrestricted", [
    ("water", "6-31g*", False), ("hcl", "cc-pvdz", False), ("nh2", "6-31g**", True)])
def test_scf_gradient_assembled_from_contractions_matches_pyscf(name, basis, unrestricted):
    b = _basis(name, basis, distort=0.05, seed=2)
    c = ref.ao_scale(b)
    mf = ref.run_scf(ref.pyscf_mole(b), unrestricted=unrestricted)
    expect = mf.nuc_grad_method().kernel()
    W = ref.density_to_ours(ref.energy_weighted_density(mf), c)
    if unrestricted:
        Da, Db = (ref.density_to_ours(X, c) for X in mf.make_rdm1())
        D = Da + Db
        g2 = I.two_electron_gradient(b, D, [Da, Db], 1.0)
    else:
        D = ref.density_to_ours(mf.make_rdm1(), c)
        g2 = I.two_electron_gradient(b, D, [D], 0.5)
    g = (I.nuclear_repulsion_gradient(b.nuclear_charges, b.positions)
         + I.one_electron_gradient(b, D, W) + g2)
    # same densities on both sides, so only integral round-off remains (measured 6.1e-14)
    np.testing.assert_allclose(g, expect, rtol=0, atol=1e-12)


def test_default_gradient_screening_error_is_small():
    # The Schwarz bound is not strict for derivative integrals, so pin the
    # accuracy of the default on a molecule large enough for screening to
    # matter (the small test molecules screen almost nothing). Ethanol/6-31G*,
    # core-Hamiltonian density, error vs unscreened: measured 1.7e-13 at the
    # default 1e-14 (8.4e-12 at 1e-13, 5e-11 at 1e-12, 4e-9 at 1e-10).
    b = _basis("ethanol", "6-31g*", distort=0.05, seed=3)
    S, T, V = I.one_electron_matrices(b)
    _e, C = scipy.linalg.eigh(T + V, S)
    nocc = int(b.atomic_numbers.sum()) // 2
    P = 2.0 * C[:, :nocc] @ C[:, :nocc].T
    exact = I.two_electron_gradient(b, P, [P], 0.5, schwarz_threshold=0.0)
    err = np.abs(I.two_electron_gradient(b, P, [P], 0.5) - exact).max()
    assert 0.0 < err < 2e-12        # > 0: screening did skip quartets


def test_results_do_not_depend_on_thread_count():
    b = _basis("ammonia", "6-31g*", distort=0.05)
    rng = np.random.default_rng(9)
    D = ref.random_symmetric(b.nao, rng)
    nthreads = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        g1 = I.two_electron_gradient(b.with_positions(b.positions), D, [D], 0.5)
        e1 = I.eri_tensor(b.with_positions(b.positions))
    finally:
        numba.set_num_threads(nthreads)
    g = I.two_electron_gradient(b.with_positions(b.positions), D, [D], 0.5)
    e = I.eri_tensor(b.with_positions(b.positions))
    np.testing.assert_array_equal(g, g1)
    np.testing.assert_array_equal(e, e1)


def test_input_validation():
    b = _basis("water", "sto-3g")
    with pytest.raises(ValueError, match="shape"):
        I.kinetic_gradient(b, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="charges"):
        I.nuclear_attraction_matrix(b, charges=[8.0, 1.0])
    with pytest.raises(MemoryError):
        I.eri_tensor(b, max_nao=5)


def test_integrals_do_not_lose_digits_far_from_the_origin():
    # Regression: the product center was formed as P = (a A + b B)/p, so for
    # a molecule ~1000 bohr from the origin P - C lost ~3 digits (V changed by
    # 1.1e-11 and one-center ERIs by 8.8e-13 under the shift; a one-center
    # <s|1/r_A|p_x> on an atom at x = 1e4 bohr came out as 6e-11 instead of
    # 0). Now P - C = (A - C) - (b/p)(A - B), exact for coincident centers.
    sym, pos = ref.molecule("hcl", distort=0.05)
    b0 = build_basis(sym, pos, "6-31g**")
    D = ref.random_symmetric(b0.nao, np.random.default_rng(12))
    b = b0.with_positions(pos + 1000.0)
    # measured after the fix: V 2.2e-13 (|V| <= 250), ERI 1.6e-14, 1e/2e
    # gradients 6.3e-13 / 1.5e-13; the remaining differences come from the
    # coordinates themselves (eps * 1000 bohr ~ 1e-13 bohr in each distance)
    np.testing.assert_allclose(I.nuclear_attraction_matrix(b), I.nuclear_attraction_matrix(b0), rtol=0, atol=1e-12)
    np.testing.assert_allclose(I.eri_tensor(b), I.eri_tensor(b0), rtol=0, atol=1e-13)
    np.testing.assert_allclose(I.one_electron_gradient(b, D, D), I.one_electron_gradient(b0, D, D), rtol=0, atol=3e-12)
    np.testing.assert_allclose(I.two_electron_gradient(b, D, [D], 0.5),
                               I.two_electron_gradient(b0, D, [D], 0.5), rtol=0, atol=1e-12)
    # one-center integrals of a far atom: exact zeros by symmetry
    ar = build_basis(["Ar"], [[1e4, 0.3, 0.1]], "6-31g*")
    V = I.nuclear_attraction_matrix(ar)
    odd = ar.ao_lxyz.sum(axis=1) % 2 == 1          # s/d (even) vs p (odd) components
    assert np.abs(V[np.ix_(~odd, odd)]).max() == 0.0
    eri = I.eri_tensor(ar)
    assert np.abs(eri[np.ix_(~odd, ~odd, ~odd, odd)]).max() == 0.0
