"""
Native Hartree-Fock backend: RHF / UHF energies and analytic gradients from
:mod:`aimd.qc` (McMurchie-Davidson integrals, :mod:`aimd.qc.scf`,
:mod:`aimd.qc.gradients`). No external quantum-chemistry program needed.

Units: positions in bohr, (N, 3); energy in hartree; gradient dE/dR in
hartree / bohr, (N, 3); dipole in e * bohr about the coordinate origin.

Method
------
``reference="auto"`` runs RHF for singlets and UHF otherwise; "rhf" / "uhf"
force the reference (UHF of a singlet starts spin-restricted and stays so
unless the guess breaks the symmetry, see :mod:`aimd.qc.scf`). Basis sets:
'sto-3g', '6-31g', '6-31g*', '6-31g**', 'cc-pvdz' (Cartesian functions,
:mod:`aimd.qc.basis`), or a custom :class:`aimd.qc.basis.BasisSet`. The
energy matches PySCF / Psi4 run with Cartesian functions (``cart=True`` /
``puream false``), not their default spherical d shells.

One :class:`aimd.qc.scf.SCFSolver` is built in the constructor (basis parsed
once, SAD atomic densities computed once and cached). Each ``compute`` moves
the basis to the new positions (O(1)), builds S, h and the ERI tensor once,
iterates the SCF, and evaluates the gradient from the same block-pair data
and Schwarz bounds (cached on the geometry's BasisSet). BLAS is held to one
thread for the whole call (:mod:`aimd.qc.threads`); the integral kernels run
on numba threads (``threads`` sets their number for the duration of each
call). :mod:`aimd.qc` defaults libgomp to passive waiting
(``OMP_WAIT_POLICY``, see :mod:`aimd.qc.threads`): with spinning workers a
step of water / STO-3G took 104 ms instead of 6 ms while another process
kept half of the cores busy. The ERI tensor of the last geometry (8 nao^4
bytes) stays cached until the next call at a different geometry or
:meth:`HFBackend.close`.

SCF guesses and the density-guess protocol (aimd.backends.base)
----------------------------------------------------------------
The starting density of an SCF is, in order of priority: a density passed to
:meth:`HFBackend.set_density_guess` (used for the next ``compute`` only,
e.g. by XL-BOMD; ``info["guess"] = "external"``), the density returned by
the previous call (``reuse_density=True``, the default; "previous"; the last
iterate if that SCF did not converge, usually still the best guess at hand),
or the initial guess ``guess`` ("auto" = "sad", or "core" / "gwh"; "init").
``GradientResult.density`` is the final AO density in the protocol shape:
the total density (nao, nao) for RHF, [alpha, beta] (2, nao, nao) for UHF.

Convergence
-----------
``conv_tol`` is the energy-change threshold (Eh) and ``conv_tol_grad`` the
orbital-gradient threshold max|X^T (F D S - S D F) X| of
:class:`aimd.qc.scf.SCFOptions` (``conv_energy`` / ``conv_commutator``);
both must hold. The gradient error is at most about the final commutator
norm (:mod:`aimd.qc.gradients`), the energy error second order.
An SCF that reaches ``max_cycles`` without converging returns the energy,
gradient, density and dipole of its last iterate (mutually consistent) with
``converged=False`` and a message in ``info["warnings"]``; it does not
raise, and no Python warning is emitted per call (:func:`aimd.md.run_md`
reports unconverged steps once per run). Positions with two nuclei closer
than 1e-5 bohr (PySCF's "Ill geometry" limit) raise ValueError rather than
return E = +inf with a NaN gradient (:func:`aimd.qc.scf.check_nuclear_separation`).
Other options of
:class:`~aimd.qc.scf.SCFOptions` (DIIS space, level shift, damping,
orthogonalization, screening) go through ``scf_options``.

Results
-------
energy, gradient, converged (the SCF flag), density (above), dipole
(electronic + nuclear, a.u.), and ``info``:

  method, basis, reference      "RHF" / "UHF", canonical basis name, "rhf" / "uhf"
  scf_converged, scf_iterations Fock builds of this call
  scf_energy                    total energy (= ``energy``), Eh
  electronic_energy, nuclear_repulsion   its two parts, Eh
  energy_change, commutator_norm         final SCF convergence measures
  s2, s2_exact                  <S^2> (UHF; 0 for RHF) and S(S+1)
  mulliken_charges              (natm,) e
  guess, init_guess             where the starting density came from
  n_removed                     linear dependencies removed (gradient not
                                exact if > 0, see aimd.qc.gradients)
  warnings                      list of messages (empty if all is well)
  timings                       seconds: scf (= integrals + guess +
                                scf_iterations + properties), gradient, total

Performance
-----------
Warm wall time per MD step (velocity Verlet, 0.5 fs, 300 K, previous
density as guess, default thresholds; 4 cores, 4 numba threads), median:

  water   / STO-3G (nao  7)   3.0 ms   8 SCF iterations
  water   / 6-31G* (nao 19)   8.8 ms   9
  ethanol / 6-31G* (nao 57)   0.47 s  10   (ERI tensor 0.12 s, SCF
                                            iterations 0.05 s, ERI
                                            gradient 0.29 s)

For comparison, the pyscf backend with the same Cartesian basis and
thresholds (4 OpenMP threads): 11 ms, 31 ms, 1.36 s. Re-measured the same
way while another process kept ~2-3 of the 4 cores busy (load average 3.4):
5.9 ms, 18 ms, 0.58 s (ethanol: ERI tensor 0.13 s, SCF iterations 0.08 s,
ERI gradient 0.34 s).

General contractions (cc-pVDZ) cost about the same as segmented bases of
the same size: the ERI kernels evaluate each primitive quartet once per
contraction block (:mod:`aimd.qc.basis`), not once per contracted shell.
One warm step at a displaced geometry (same machine, default thresholds):

  ethanol / cc-pVDZ (nao 75)   1.0 s   (was 1.4 s with per-shell kernels)
  Cl2     / 6-31G*  (nao 38)   0.18 s;  cc-pVDZ (nao 38) 0.13 s (was 0.80 s;
                                        pyscf cart 0.52 s)
  PCl3    / 6-31G*  (nao 76)   2.2 s;   cc-pVDZ (nao 76) 1.8 s  (was 9.0 s;
                                        pyscf cart 2.9 s)
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Iterator, Sequence

import numba
import numpy as np

from aimd.backends.base import ForceBackend, GradientResult
from aimd.backends.registry import register_backend
from aimd.qc import integrals as qcint
from aimd.qc.basis import BasisSet
from aimd.qc.gradients import scf_gradient
from aimd.qc.scf import SCFOptions, SCFResult, SCFSolver
from aimd.qc.threads import limit_blas_threads

_REFERENCES = {"auto": None, "rhf": "rhf", "restricted": "rhf",
               "uhf": "uhf", "unrestricted": "uhf"}
_METHODS = {"hf": None, "scf": None, "rhf": "rhf", "uhf": "uhf"}
_GUESSES = {"auto": "sad", "sad": "sad", "core": "core", "gwh": "gwh"}
# SCFOptions fields that have their own constructor argument here
_OWN_OPTIONS = {"conv_energy": "conv_tol", "conv_commutator": "conv_tol_grad",
                "max_iter": "max_cycles", "guess": "guess", "warn_unconverged": None}


@contextlib.contextmanager
def _numba_threads(n: int | None) -> Iterator[None]:
    """Run the block with ``n`` numba threads (the setting is per calling thread)."""
    if n is None:
        yield
        return
    old = numba.get_num_threads()
    numba.set_num_threads(n)
    try:
        yield
    finally:
        numba.set_num_threads(old)


@register_backend
class HFBackend(ForceBackend):
    """
    Native RHF / UHF energies and analytic gradients (module docstring).

    Parameters (beyond symbols, charge, multiplicity)
      basis          basis name ('sto-3g', '6-31g*', 'cc-pvdz', ...) or a BasisSet
      reference      "auto" (RHF for singlets, UHF otherwise), "rhf" or "uhf"
      method         "hf" (default); "rhf" / "uhf" force the reference like
                     ``reference`` (accepted for symmetry with the pyscf /
                     psi4 backends; nothing beyond HF is available)
      conv_tol       SCF energy-change threshold, Eh
      conv_tol_grad  SCF orbital-gradient (commutator) threshold
      max_cycles     Fock builds before giving up (result flagged unconverged)
      guess          initial guess of the first SCF: "auto" (= "sad"), "sad",
                     "core" or "gwh"
      reuse_density  start each SCF from the previous call's density
      gradient_screening
                     Schwarz threshold of the ERI-derivative contraction
                     (aimd.qc.integrals.DEFAULT_SCHWARZ_GRAD)
      threads        numba threads for the integral kernels during each
                     ``compute`` (restored afterwards; None: numba's current
                     setting, NUMBA_NUM_THREADS by default)
      scf_options    further SCFOptions fields, e.g. {"diis_space": 10,
                     "level_shift": 0.5, "orthogonalization": "canonical"}
    """

    name = "hf"
    description = "native RHF / UHF, analytic gradients (McMurchie-Davidson, aimd.qc)"
    supports_density_guess = True

    def __init__(
        self,
        symbols: Sequence[str],
        charge: int = 0,
        multiplicity: int = 1,
        basis: str | BasisSet = "sto-3g",
        reference: str | None = "auto",
        method: str = "hf",
        conv_tol: float = 1e-10,
        conv_tol_grad: float = 1e-7,
        max_cycles: int = 100,
        guess: str = "auto",
        reuse_density: bool = True,
        gradient_screening: float = qcint.DEFAULT_SCHWARZ_GRAD,
        threads: int | None = None,
        scf_options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(symbols, charge, multiplicity)
        ref = self._parse_reference(reference, method)
        key = str(guess).strip().lower()
        if key not in _GUESSES:
            raise ValueError(f"unknown guess {guess!r}; use one of {sorted(_GUESSES)}")
        self.init_guess = _GUESSES[key]
        if not gradient_screening >= 0.0:
            raise ValueError("gradient_screening must be >= 0")
        self.gradient_screening = float(gradient_screening)
        self.reuse_density = bool(reuse_density)

        extra = dict(scf_options or {})
        for k in extra:
            if k in _OWN_OPTIONS:
                arg = _OWN_OPTIONS[k]
                raise ValueError(f"set SCF option {k!r} through the backend argument {arg!r}"
                                 if arg else f"SCF option {k!r} is managed by the backend")
        # SCFOptions validates every value (and rejects unknown names)
        try:
            options = SCFOptions(conv_energy=float(conv_tol), conv_commutator=float(conv_tol_grad),
                                 max_iter=int(max_cycles), guess=self.init_guess,
                                 warn_unconverged=False, **extra)
        except TypeError as e:
            raise ValueError(f"invalid scf_options: {e}") from e
        self.solver = SCFSolver(self.symbols, basis, charge=self.charge,
                                multiplicity=self.multiplicity, reference=ref, options=options)
        if threads is not None and not 1 <= int(threads) <= numba.config.NUMBA_NUM_THREADS:
            raise ValueError(f"threads must be in 1..{numba.config.NUMBA_NUM_THREADS}")
        self.threads = None if threads is None else int(threads)

        self._guess: np.ndarray | None = None          # external, next call only
        self._last_density: np.ndarray | None = None   # previous result
        self.last_scf: SCFResult | None = None         # full SCF state of the last call

    # ── Configuration ────────────────────────────────────────────────────────

    def _parse_reference(self, reference: str | None, method: str) -> str | None:
        m = str(method).strip().lower()
        if m not in _METHODS:
            raise ValueError(f"the hf backend does Hartree-Fock only; unknown method {method!r} "
                             "(use 'hf', 'rhf' or 'uhf', or the pyscf backend for DFT / MP2)")
        r = "auto" if reference is None else str(reference).strip().lower()
        if r not in _REFERENCES:
            raise ValueError(f"unknown reference {reference!r}; use 'auto', 'rhf' or 'uhf'")
        ref, forced = _REFERENCES[r], _METHODS[m]
        if ref is not None and forced is not None and ref != forced:
            raise ValueError(f"method {method!r} conflicts with reference {reference!r}")
        return ref or forced

    @property
    def options(self) -> SCFOptions:
        return self.solver.options

    @property
    def unrestricted(self) -> bool:
        return self.solver.unrestricted

    @property
    def nao(self) -> int:
        return self.solver.nao

    @property
    def basis(self) -> BasisSet:
        """The parsed basis (at placeholder positions; moved per call)."""
        return self.solver.basis

    @property
    def density_shape(self) -> tuple[int, ...]:
        """Shape of ``GradientResult.density`` and of a density guess."""
        return self.solver.density_shape

    @property
    def label(self) -> str:
        return "UHF" if self.unrestricted else "RHF"

    # ── Density-guess protocol ───────────────────────────────────────────────

    def set_density_guess(self, density: np.ndarray) -> None:
        """AO density to start the next SCF from (shape ``density_shape``)."""
        d = np.array(density, dtype=float)
        if d.shape != self.density_shape:
            raise ValueError(
                f"density guess has shape {d.shape}, expected {self.density_shape} "
                f"({'unrestricted' if self.unrestricted else 'restricted'} reference)")
        if not np.all(np.isfinite(d)):
            raise ValueError("density guess contains non-finite values")
        self._guess = d

    def reset_guess(self) -> None:
        """Forget the stored and external guesses (next SCF uses ``guess``)."""
        self._guess = None
        self._last_density = None

    # ── Energy and gradient ──────────────────────────────────────────────────

    def compute(self, positions: np.ndarray) -> GradientResult:
        t0 = time.perf_counter()
        x = np.array(positions, dtype=float)
        if x.ndim != 2 or x.shape != (len(self.symbols), 3):
            raise ValueError(f"expected positions of shape ({len(self.symbols)}, 3), got {x.shape}")

        if self._guess is not None:
            guess, label = self._guess, "external"
        elif self.reuse_density and self._last_density is not None:
            guess, label = self._last_density, "previous"
        else:
            guess, label = self.init_guess, "init"
        self._guess = None                                # one call only

        with limit_blas_threads(self.options.blas_threads), _numba_threads(self.threads):
            res = self.solver.run(x, guess=guess)
            t1 = time.perf_counter()
            gradient = scf_gradient(res, schwarz_threshold=self.gradient_screening,
                                    blas_threads=None)
        t2 = time.perf_counter()

        density = res.guess_density                       # a fresh array
        if self.reuse_density:
            self._last_density = density
        self.last_scf = res

        warnings_: list[str] = []
        if not res.converged:
            warnings_.append(
                f"SCF not converged after {res.iterations} iterations (energy change "
                f"{res.energy_change:.2e} Eh, commutator {res.commutator_norm:.2e}); "
                "energy and gradient are those of the last iterate")
        if res.n_removed:
            warnings_.append(
                f"{res.n_removed} near-linearly-dependent basis combination(s) removed; "
                "the analytic gradient is then not the exact derivative of the energy")
        info: dict[str, Any] = {
            "method": self.label,
            "basis": res.basis.name,
            "reference": res.reference,
            "scf_converged": bool(res.converged),
            "scf_iterations": int(res.iterations),
            "scf_energy": float(res.energy),
            "electronic_energy": float(res.electronic_energy),
            "nuclear_repulsion": float(res.nuclear_repulsion),
            "energy_change": float(res.energy_change),
            "commutator_norm": float(res.commutator_norm),
            "s2": float(res.s2),
            "s2_exact": float(res.s2_exact),
            "mulliken_charges": res.mulliken_charges.copy(),
            "guess": label,
            "init_guess": self.init_guess,
            "n_removed": int(res.n_removed),
            "warnings": warnings_,
            "timings": {
                "integrals": float(res.timings.get("integrals", 0.0)),
                "guess": float(res.timings.get("guess", 0.0)),
                "scf_iterations": float(res.timings.get("iterations", 0.0)),
                "scf": t1 - t0,
                "gradient": t2 - t1,
                "total": time.perf_counter() - t0,
            },
        }
        return GradientResult(
            energy=float(res.energy),
            gradient=gradient,
            converged=bool(res.converged),
            density=density.copy(),
            dipole=np.array(res.dipole, dtype=float),
            info=info,
        )

    def close(self) -> None:
        """Release the cached ERI tensor and forget the stored densities."""
        self.solver.clear_cache()
        self.reset_guess()
        self.last_scf = None
