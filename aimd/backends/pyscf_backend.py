"""
PySCF backend: analytic HF / Kohn-Sham DFT / MP2 gradients through PySCF.

PySCF (Sun et al., WIREs Comput. Mol. Sci. 8, e1340 (2018); J. Chem. Phys.
153, 024109 (2020)) is imported lazily, so registering the backend does not
require it to be installed.

Geometry handling
-----------------
One ``gto.Mole`` is built in the constructor (basis parsing, shell tables) and
moved to each new geometry with ``Mole.set_geom_``, which only rewrites the
coordinates in the integral environment. Coordinates are given in bohr
(``unit='Bohr'``) with ``symmetry=False``; PySCF then neither translates nor
reorients the molecule, so gradient rows follow our atom order. A fresh SCF
object is created at every step (cheap), so no geometry-dependent cache (ERIs,
DFT grids) can leak from one geometry to the next. No chkfile is written.

Methods
-------
``method``      reference / functional
  "hf"          RHF for singlets, UHF otherwise (override with ``reference``)
  "rhf", "uhf"  forced reference
  "mp2"         MP2 on an RHF / UHF reference, analytic relaxed gradient
  anything else a libxc functional name ("b3lyp", "pbe", "pbe0", ...):
                RKS for singlets, UKS otherwise

For DFT the gradient includes the response of the atom-centred integration
grid (``grid_response=True``, not PySCF's default), which makes it the exact
derivative of the quadrature energy; without it the force is inconsistent
with the energy at the 1e-5 Eh/bohr level and NVE runs drift.

SCF guesses and the density-guess protocol (aimd.backends.base)
----------------------------------------------------------------
The starting density of an SCF is, in order of priority: a density passed to
:meth:`PySCFBackend.set_density_guess` (used for the next ``compute`` only;
XL-BOMD), the previous converged density (``reuse_density=True``, the default:
plain BOMD with guess reuse), or PySCF's ``init_guess`` (default "minao").
``GradientResult.density`` is the final SCF density in the AO basis: the total
density (nao, nao) for restricted references, [alpha, beta] (2, nao, nao) for
unrestricted ones. For MP2 it is the reference (HF) density, which is the
quantity that the SCF guess concerns.

An SCF that does not converge within ``max_cycle`` iterations returns its last
energy and gradient with ``converged=False``; it does not raise.

Results
-------
energy (Eh) and gradient (Eh / bohr, (N, 3)); dipole: electronic + nuclear
dipole moment in atomic units (e bohr) about the coordinate origin, from the
SCF density (``None`` for MP2: its relaxed density stays inside PySCF's
gradient code, and the unrelaxed one is not dE/dF); info: ``scf_iterations``
(SCF cycles of this call, not counting PySCF's extra post-convergence
diagonalisation), ``scf_converged``, ``scf_energy``, ``guess`` ("init",
"previous" or "external"), ``method``, ``basis``, and for MP2
``mp2_correlation_energy``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from aimd.backends.base import ForceBackend, GradientResult
from aimd.backends.registry import register_backend

_HF_METHODS = {"hf": None, "scf": None, "rhf": "rhf", "uhf": "uhf"}
_MP2_METHODS = {"mp2": None, "rmp2": "rhf", "ump2": "uhf"}
_REFERENCES = {"rhf": "rhf", "restricted": "rhf", "rks": "rhf",
               "uhf": "uhf", "unrestricted": "uhf", "uks": "uhf"}


def _import_pyscf() -> dict[str, Any]:
    try:
        from pyscf import dft, gto, lib, mp, scf  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise RuntimeError(
            "The pyscf backend needs PySCF (pip install pyscf, or "
            "pip install 'aimd[pyscf]')."
        ) from e
    return {"dft": dft, "gto": gto, "lib": lib, "mp": mp, "scf": scf}


@register_backend
class PySCFBackend(ForceBackend):
    """
    HF / DFT / MP2 energies and analytic gradients from PySCF.

    Parameters (beyond symbols, charge, multiplicity)
      method         "hf", "rhf", "uhf", "mp2" or a libxc functional name
      basis          any PySCF basis name (or a PySCF basis dict)
      reference      None (by multiplicity), "rhf" or "uhf"
      cart           Cartesian (6d, 10f) instead of spherical functions
      conv_tol       SCF energy convergence (Eh); conv_tol_grad defaults to
                     sqrt(conv_tol), as in PySCF
      max_cycle      SCF iteration cap; reaching it gives converged=False
      grid_level     DFT grid level (PySCF 0-9)
      grid_response  include the grid-weight derivatives in DFT gradients
      init_guess     PySCF guess for the first SCF ("minao", "atom", "huckel", ...)
      reuse_density  start each SCF from the previous converged density
      threads        PySCF OpenMP threads (process-wide setting; None leaves it)
      max_memory_mb  PySCF memory limit (None: PySCF default)
      scf_options    extra SCF attributes, e.g. {"level_shift": 0.2, "diis_space": 12}
    """

    name = "pyscf"
    description = "HF / DFT / MP2 analytic gradients through PySCF"
    supports_density_guess = True
    requires = ("pyscf",)            # optional packages (aimd.backends.backend_dependencies)

    def __init__(
        self,
        symbols: Sequence[str],
        charge: int = 0,
        multiplicity: int = 1,
        method: str = "hf",
        basis: str | dict = "sto-3g",
        reference: str | None = None,
        cart: bool = False,
        conv_tol: float = 1e-10,
        conv_tol_grad: float | None = None,
        max_cycle: int = 100,
        grid_level: int = 3,
        grid_response: bool = True,
        init_guess: str = "minao",
        reuse_density: bool = True,
        threads: int | None = None,
        max_memory_mb: int | None = None,
        scf_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(symbols, charge, multiplicity)
        self._pyscf = _import_pyscf()
        if self.multiplicity < 1:
            raise ValueError("multiplicity must be >= 1")
        if conv_tol <= 0.0 or (conv_tol_grad is not None and conv_tol_grad <= 0.0):
            raise ValueError("SCF convergence thresholds must be positive")
        if max_cycle < 1:
            # PySCF's max_cycle = 0 returns E[guess] together with orbitals
            # (hence density and gradient) from diagonalising F[guess]:
            # an energy and a force that belong to different densities.
            raise ValueError("max_cycle must be >= 1")

        self.method, self.xc, ref = self._parse_method(method)
        if reference is not None:
            key = str(reference).strip().lower()
            if key not in _REFERENCES:
                raise ValueError(f"unknown reference {reference!r}; use 'rhf' or 'uhf'")
            if ref is not None and _REFERENCES[key] != ref:
                raise ValueError(f"method {method!r} conflicts with reference {reference!r}")
            ref = _REFERENCES[key]
        if ref is None:
            ref = "rhf" if self.multiplicity == 1 else "uhf"
        if ref == "rhf" and self.multiplicity != 1:
            raise ValueError(
                "a restricted closed-shell reference needs multiplicity 1; "
                "use reference='uhf' for open shells"
            )
        self.unrestricted = ref == "uhf"

        self.basis = basis.strip() if isinstance(basis, str) else basis
        self.cart = bool(cart)
        self.conv_tol = float(conv_tol)
        self.conv_tol_grad = None if conv_tol_grad is None else float(conv_tol_grad)
        self.max_cycle = int(max_cycle)
        self.grid_level = int(grid_level)
        self.grid_response = bool(grid_response)
        self.init_guess = str(init_guess)
        self.reuse_density = bool(reuse_density)
        self.scf_options = dict(scf_options or {})
        if threads is not None:
            if int(threads) < 1:
                raise ValueError("threads must be >= 1")
            self._pyscf["lib"].num_threads(int(threads))

        # Build the molecule once at a placeholder geometry (atoms 2 bohr
        # apart on z): the basis layout does not depend on the positions.
        n = len(self.symbols)
        placeholder = np.zeros((n, 3))
        placeholder[:, 2] = 2.0 * np.arange(n)
        mol = self._pyscf["gto"].Mole()
        mol.atom = [(s, xyz) for s, xyz in zip(self.symbols, placeholder.tolist())]
        mol.unit = "Bohr"
        mol.basis = self.basis
        mol.cart = self.cart
        mol.charge = self.charge
        mol.spin = self.multiplicity - 1
        mol.symmetry = False
        mol.verbose = 0
        mol.output = None
        if max_memory_mb is not None:
            mol.max_memory = int(max_memory_mb)
        try:
            mol.build(dump_input=False, parse_arg=False)
        except (RuntimeError, KeyError, ValueError) as e:
            raise ValueError(f"PySCF could not build the molecule: {e}") from e
        self._mol = mol
        self.nao = int(mol.nao_nr())

        self._guess: np.ndarray | None = None          # external, next call only
        self._last_density: np.ndarray | None = None   # previous result

    # ── Configuration helpers ────────────────────────────────────────────────

    def _parse_method(self, method: str) -> tuple[str, str | None, str | None]:
        """(kind, xc, forced reference) for a method string."""
        m = str(method).strip().lower()
        if m in _HF_METHODS:
            return "hf", None, _HF_METHODS[m]
        if m in _MP2_METHODS:
            return "mp2", None, _MP2_METHODS[m]
        try:
            self._pyscf["dft"].libxc.parse_xc(m)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"unknown method {method!r}: use 'hf', 'rhf', 'uhf', 'mp2' or a "
                "libxc functional name such as 'b3lyp'"
            ) from e
        return "dft", m, None

    @property
    def density_shape(self) -> tuple[int, ...]:
        """Shape of ``GradientResult.density`` and of a density guess."""
        n = self.nao
        return (2, n, n) if self.unrestricted else (n, n)

    @property
    def label(self) -> str:
        if self.method == "dft":
            return f"{'UKS' if self.unrestricted else 'RKS'}({self.xc})"
        ref = "UHF" if self.unrestricted else "RHF"
        return ref if self.method == "hf" else f"{ref}-MP2"

    def _make_scf(self, mol: Any) -> Any:
        scf, dft = self._pyscf["scf"], self._pyscf["dft"]
        if self.method == "dft":
            mf = dft.UKS(mol) if self.unrestricted else dft.RKS(mol)
            mf.xc = self.xc
            mf.grids.level = self.grid_level
        else:
            mf = scf.UHF(mol) if self.unrestricted else scf.RHF(mol)
        mf.verbose = 0
        mf.chkfile = None
        mf.conv_tol = self.conv_tol
        if self.conv_tol_grad is not None:
            mf.conv_tol_grad = self.conv_tol_grad
        mf.max_cycle = self.max_cycle
        mf.init_guess = self.init_guess
        for key, value in self.scf_options.items():
            if not hasattr(mf, key):
                raise ValueError(f"PySCF SCF object has no option {key!r}")
            setattr(mf, key, value)
        return mf

    # ── Density-guess protocol ───────────────────────────────────────────────

    def set_density_guess(self, density: np.ndarray) -> None:
        """AO density to start the next SCF from (shape ``density_shape``)."""
        d = np.array(density, dtype=float)
        if d.shape != self.density_shape:
            raise ValueError(
                f"density guess has shape {d.shape}, expected {self.density_shape} "
                f"({'unrestricted' if self.unrestricted else 'restricted'} reference)"
            )
        self._guess = d

    def reset_guess(self) -> None:
        """Forget the stored and external guesses (next SCF uses ``init_guess``)."""
        self._guess = None
        self._last_density = None

    # ── Energy and gradient ──────────────────────────────────────────────────

    def compute(self, positions: np.ndarray) -> GradientResult:
        x = np.asarray(positions, dtype=float).reshape(-1, 3)
        if x.shape[0] != len(self.symbols):
            raise ValueError(f"expected {len(self.symbols)} atoms, got {x.shape[0]}")
        mol = self._mol.set_geom_(x, unit="Bohr", symmetry=False)

        if self._guess is not None:
            dm0, guess = self._guess, "external"
        elif self.reuse_density and self._last_density is not None:
            dm0, guess = self._last_density, "previous"
        else:
            dm0, guess = None, "init"
        self._guess = None                                # one call only

        mf = self._make_scf(mol)
        cycles = [0]
        user_callback = mf.callback                       # from scf_options, if any

        def count(envs: dict) -> None:                    # once per SCF cycle
            cycles[0] += 1
            if callable(user_callback):
                user_callback(envs)

        mf.callback = count
        mf.kernel(dm0=dm0)
        scf_converged = bool(mf.converged)
        density = np.array(mf.make_rdm1(), dtype=float)   # plain array, no tags
        info: dict[str, Any] = {
            "method": self.label,
            "basis": self.basis if isinstance(self.basis, str) else "custom",
            "scf_iterations": int(cycles[0]),
            "scf_converged": scf_converged,
            "scf_energy": float(mf.e_tot),
            "guess": guess,
        }

        if self.method == "mp2":
            mp = self._pyscf["mp"]
            pt = mp.UMP2(mf) if self.unrestricted else mp.MP2(mf)
            pt.verbose = 0
            pt.kernel()
            grad_obj = pt.nuc_grad_method()
            grad_obj.verbose = 0
            gradient = grad_obj.kernel()
            energy = float(pt.e_tot)
            info["mp2_correlation_energy"] = float(pt.e_corr)
            dipole = None
        else:
            grad_obj = mf.nuc_grad_method()
            grad_obj.verbose = 0
            if self.method == "dft":
                grad_obj.grid_response = self.grid_response
            gradient = grad_obj.kernel()
            energy = float(mf.e_tot)
            dipole = np.asarray(
                mf.dip_moment(mol, density, unit="AU", verbose=0), dtype=float
            ).reshape(3)

        if self.reuse_density:
            self._last_density = density
        return GradientResult(
            energy=energy,
            gradient=np.array(gradient, dtype=float).reshape(-1, 3),
            converged=scf_converged,
            density=density.copy(),
            dipole=dipole,
            info=info,
        )

    def close(self) -> None:
        self.reset_guess()
