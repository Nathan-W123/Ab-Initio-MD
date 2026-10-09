# Ab-Initio-MD

Born–Oppenheimer ab initio molecular dynamics with its own Hartree–Fock
engine. Nuclei move classically on the potential-energy surface of an
electronic-structure method that is solved anew at every step. The package
`aimd` contains:

- a native RHF / UHF code: McMurchie–Davidson integrals, DIIS SCF and
  analytic gradients, compiled with numba;
- integrators: velocity Verlet and Langevin BAOAB, with Berendsen, CSVR and
  Nosé–Hoover-chain thermostats, plus extended-Lagrangian BOMD (XL-BOMD);
- exact checkpoint/restart;
- trajectory analysis: RDF, geometry, VACF/VDOS, IR spectra and block-averaged
  statistics;
- an `aimd` command line that ties these together.

PySCF and Psi4 can serve as alternative force backends.

```bash
pip install -e ".[dev]"
aimd run examples/water.xyz --basis 6-31g* --temperature 300 --steps 200 \
    --thermostat csvr --dipoles dipoles.csv --velocities velocities.xyz --seed 1
aimd analyze ir dipoles.csv --temperature 300
aimd view trajectory.xyz          # interactive HTML movie with energy charts
```

## Contents

- [Physics and methods](#physics-and-methods)
- [Layout](#layout)
- [Installation](#installation)
- [Command line](#command-line)
- [Python API](#python-api)
- [Units](#units)
- [Backends](#backends)
- [Validation](#validation)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Tests](#tests)

## Physics and methods

**Born–Oppenheimer MD.** At every step the electronic Schrödinger equation is
solved at fixed nuclear positions R. The resulting energy E(R) and its
gradient give the forces F = −∇E, and the nuclei follow Newton's equations.
This assumes the electrons stay in their ground state (no non-adiabatic
effects) and treats the nuclei classically: there is no zero-point energy and
no tunnelling. Energy is conserved only if the forces are the exact
derivative of the energy that is reported. In practice that means analytic
gradients and well-converged SCF.

**Hartree–Fock** (`aimd/qc/scf.py`). The code solves the restricted
closed-shell Roothaan–Hall equations and the unrestricted Pople–Nesbet
equations F C = S C ε:

- F = h + J[P] − K[D_σ];
- the initial guess is a superposition of atomic densities (SAD), with core
  and generalised Wolfsberg–Helmholz guesses as alternatives;
- Pulay DIIS extrapolates the Fock matrix (Chem. Phys. Lett. 73, 393
  (1980));
- the AO basis is orthogonalised symmetrically, or canonically when it is
  nearly linearly dependent;
- the SCF has converged when both the energy change and the orbital gradient
  max|Xᵀ(FDS − SDF)X| fall below their thresholds.

Basis sets are STO-3G, 6-31G, 6-31G*, 6-31G** and cc-pVDZ, for H–Ar. All of
them use **Cartesian** d functions (6d).

**Integrals** (`aimd/qc/integrals.py`, `hermite.py`, `boys.py`).

- Method: McMurchie–Davidson Hermite-Gaussian expansion with the Boys
  function (McMurchie & Davidson, J. Comput. Phys. 26, 218 (1978); Helgaker,
  Jørgensen & Olsen, ch. 9).
- Scope: overlap, kinetic, nuclear-attraction and dipole integrals, plus the
  full ERI tensor.
- Symmetry and screening: the 8-fold permutational symmetry is used, and
  Schwarz screening (Häser & Ahlrichs 1989) gives a strict bound on what it
  drops.
- Parallelism: the loops are numba kernels running on numba threads, with
  results bitwise identical for any thread count.

**Analytic gradients** (`aimd/qc/gradients.py`). The SCF energy is
stationary in the orbitals, so the gradient needs only derivative integrals
contracted with the densities:

dE/dX = dE_nuc/dX + Σ P dh/dX + ½ Σ Γ d(μν|λσ)/dX − Σ W dS/dX.

Here W is the energy-weighted density, which gives the Pulay term (Pulay,
Mol. Phys. 17, 197 (1969)). The two-electron part is integral-direct: no
derivative tensor is ever stored.

**Integrators** (`aimd/integrators.py`, `aimd/thermostats.py`).

| Integrator | What it is | Reference |
|---|---|---|
| Velocity Verlet | NVE dynamics | |
| Berendsen | weak coupling; for equilibration only | |
| CSVR | stochastic velocity rescaling | Bussi, Donadio & Parrinello, J. Chem. Phys. 126, 014101 (2007) |
| Nosé–Hoover chains | MTK integration with Suzuki–Yoshida factorisation | Martyna et al., Mol. Phys. 87, 1117 (1996) |
| Langevin BAOAB | Langevin dynamics | Leimkuhler & Matthews, Appl. Math. Res. Express 2013, 34 |

The thermostats act as velocity operators in a symmetric Trotter splitting
around the Verlet step. Every integrator reports a conserved quantity: the
total energy for NVE, the extended Hamiltonian for Nosé–Hoover chains, and the
"effective energy" (total energy minus the heat added by the thermostat) for
CSVR, Berendsen and Langevin. Its drift is the accuracy check that the CLI
prints. Centre-of-mass motion is always removed from the initial velocities.
`--remove-rotation` also removes the angular momentum, and the temperature
then uses N_dof = 3N − 6, also for linear polyatomics such as CO₂: velocity
Verlet keeps all three components of L at zero and the molecule bends at once,
so 3N − 5 would overcount (thermostats would run its internal modes ~4/3 too
hot). A diatomic has N_dof = 1.

**XL-BOMD** (`aimd/xlbomd.py`).

- A plain SCF started from an extrapolated guess leaves an error that is not
  time-reversible, so the energy drifts unless the SCF is converged very
  tightly.
- XL-BOMD instead propagates an auxiliary density with a time-reversible
  Verlet equation, P(t+dt) = 2P(t) − P(t−dt) + κ(D(t) − P(t)) + α Σ c_k P(t−k dt),
  and uses it as the SCF guess (Niklasson, Phys. Rev. Lett. 100, 123004
  (2008)).
- The weak dissipation term comes from Niklasson et al., J. Chem. Phys. 130,
  214109 (2009), with K = 3–9.
- Backends that accept density guesses support it: `hf` and `pyscf`.

**Checkpoints** (`aimd/checkpoint.py`).

- What is saved: positions, velocities, the cached forces, the bit-generator
  states of the random numbers, the thermostat variables and the XL-BOMD
  density history; a fingerprint of the backend (method label, basis name,
  number of AOs, Cartesian flag), the output cadence and, from the CLI, the
  backend's constructor arguments.
- Format: NumPy `.npz` with JSON metadata, loaded without pickle.
- Safety: a restart is refused, before anything is changed, if the backend
  fingerprint differs (another method or basis, even one with the same
  number of AOs), if `write_every` differs while files are appended to, or
  if an output file does not continue the checkpointed run (its frame/row at
  the checkpoint step must hold the checkpoint's atoms, positions,
  velocities and energies). A checkpoint path that cannot be written fails
  before the first step.
- Guarantee: "N steps + restart + M steps" writes the same files, byte for
  byte, as an uninterrupted N + M run. This is tested for the hf backend
  through the CLI, with Langevin, CSVR + XL-BOMD and NHC.

**Analysis** (`aimd/analysis/`).

- Radial distribution functions, for clusters or a cubic periodic box.
- Bond lengths, angles and dihedrals.
- Velocity autocorrelation and the vibrational density of states.
- IR spectra from the dipole-derivative autocorrelation, with optional
  quantum corrections.
- Block-averaged statistics: the error of correlated time series by the
  Flyvbjerg–Petersen blocking method.

## Layout

```
aimd/
  units.py, elements.py   atomic units, element data (H–Kr masses)
  system.py               MolecularSystem: positions, velocities, masses, DOF bookkeeping
  backends/
    base.py               ForceBackend interface + GradientResult (density-guess protocol)
    registry.py           name -> backend class (@register_backend)
    hf.py                 native RHF/UHF backend (aimd.qc)
    pyscf_backend.py      HF / DFT / MP2 through PySCF (optional)
    psi4_backend.py       HF / DFT / MP2 through Psi4 (optional, untested here)
    morse.py, harmonic.py model surfaces for tests
  qc/                     native quantum chemistry: basis sets, integrals, SCF, gradients
  integrators.py          VelocityVerlet, CSVR, NoseHooverChain, LangevinBAOAB, XLBOMD
  thermostats.py          Berendsen, CSVR, Nose-Hoover-chain velocity operators
  xlbomd.py               XL-BOMD auxiliary density and coefficient table
  md.py                   run_md driver (outputs, checkpoints, restart)
  checkpoint.py           save_checkpoint / load_checkpoint
  trajectory.py           XYZ / velocity / CSV writers and readers
  analysis/               rdf, geometry, spectra, statistics
  cli.py, cli_analyze.py  the `aimd` command
  viewer.py               `aimd view`: self-contained HTML trajectory viewer
  testing.py              finite-difference gradient check for new backends
examples/                 geometries and config files (below)
tests/                    pytest suite
```

## Installation

You need Python ≥ 3.10. The required packages are numpy, scipy and numba,
plus tomli on Python 3.10.

```bash
pip install -e .                 # engine + native HF backend
pip install -e ".[pyscf]"        # + PySCF backend
pip install -e ".[plot]"         # + matplotlib for `aimd analyze --plot`
pip install -e ".[dev]"          # + pytest and PySCF, to run the full test suite
conda install -c conda-forge psi4   # Psi4 backend (conda only)
```

`aimd backends` shows which optional backends can be used. The first run
compiles the numba kernels, which takes about 40–60 s. The compiled code is
cached in `__pycache__`, so later runs start in under a second.

## Command line

### `aimd run`

```bash
# NVE water, HF/STO-3G, 0.5 fs steps, initial velocities at 300 K
aimd run examples/water.xyz --steps 200 --dt 0.5 --temperature 300 --seed 1

# Canonical sampling: CSVR (or nhc, langevin, berendsen) at 300 K, larger basis
aimd run examples/water.xyz --basis 6-31g* --thermostat csvr --temperature 300 --tau 100

# Open-shell radical with UHF and XL-BOMD SCF guesses, rotation removed
aimd run examples/methyl_radical.xyz --multiplicity 2 --basis 6-31g* \
    --temperature 300 --xlbomd --remove-rotation --dipoles ch3_dipoles.csv

# DFT through PySCF (Cartesian basis option passed through to the backend)
aimd run examples/methanol.xyz --backend pyscf --method b3lyp --basis 6-31g* \
    --temperature 300 --backend-option cart=true
```

The options fall into four groups (see `aimd run --help`):

- **Electronic structure:**
  - `--backend`: `hf` by default; or `pyscf`, `psi4`, `morse`, `harmonic`
    (a model surface; from the CLI each atom sits in a harmonic well centred
    on its starting position, which the checkpoint records for a restart).
  - `--method`, `--basis`, `--reference rhf|uhf`, `--charge`,
    `--multiplicity`.
  - `--threads`: numba threads for `hf`, OpenMP threads for `pyscf`.
  - SCF control: `--conv-tol` (energy, Eh), `--conv-tol-grad` (orbital
    gradient), `--max-cycles`.
  - `--backend-option KEY=VALUE` passes any other constructor argument, with
    the value parsed as JSON, e.g. `scf_options={"level_shift": 0.3}`.
- **Dynamics:**
  - `--steps` and `--dt` (fs).
  - `--thermostat nve|berendsen|langevin|csvr|nhc` with `--temperature` (K).
    `--temperature` is also the initial temperature unless
    `--init-temperature` is given.
  - `--tau`: Berendsen/CSVR coupling time, or the NHC period, in fs.
  - `--friction`: Langevin, in 1/fs.
  - `--nhc-chain-length`, `--nhc-substeps`.
  - `--xlbomd` and `--xl-k`.
  - `--remove-rotation`.
  - `--seed` (an integer >= 0): seeds both the initial velocities and the
    thermostat noise, as independent streams.
- **Output:**
  - `--trajectory`: XYZ, in Å.
  - `--energies`: CSV of step, time, E_pot, E_kin, E_tot, T and E_cons.
  - `--velocities`: XYZ layout, in bohr/au_time.
  - `--dipoles`: CSV, in e·bohr.
  - `--write-every`, `--print-every`.
  - `--checkpoint PATH` and `--checkpoint-every N`.
- **Restart:** `--restart CKPT` continues a checkpointed run. Use the same
  options as the original run (the same command line or config file):
  - `--steps` then means *additional* steps, and step numbers carry on.
  - Output files are cut back to the checkpoint step and appended to, so
    they match an uninterrupted run.
  - The backend and its settings (`--method`, `--basis`, `--reference`, SCF
    thresholds, `--backend-option`) are recorded in the checkpoint: flags
    left out are taken from it. A flag that changes the potential-energy
    surface (method, basis, reference, other backend options) is refused;
    changed SCF thresholds are accepted with a warning, `--threads` freely.
  - Mismatches are reported clearly: a different thermostat or XL-BOMD
    choice, backend settings, backend, atoms, charge or `--write-every`
    (for files being appended to), and output files that another run has
    overwritten since the checkpoint (for example the default
    `trajectory.xyz` of a run in the same directory). Changed parameters such
    as the temperature are accepted with a warning.

Each run prints a header (backend, method, basis with the number of AOs, and
the integrator) and one line per step: E_pot, E_tot, the conserved quantity,
T, SCF iterations, and the wall time of the step in ms. A summary follows at
the end. This one is from `aimd run examples/water.xyz --temperature 300
--seed 2 --steps 400 --velocities v.xyz --dipoles d.csv` (HF/STO-3G, NVE):

```
summary
  steps            400 (0 -> 400), 200 fs
  conserved energy max |E - E0| = 1.911e-06 Eh, final E - E0 = +2.711e-07 Eh, fitted drift +1.359e-07 Eh/ps
  temperature      mean 292.83 K, std 4.79 K (steps 1-400)
  SCF iterations   mean 7.82 per step (min 7, max 8)
  timing           2.78 s total, 5.61 ms per step (median), backend 5.44 ms
  output           trajectory.xyz, energies.csv, v.xyz, d.csv
```

With this seed most of the kinetic energy happens to be rotational: E_pot
varies by only 1.9e-4 Eh, so the conserved-energy error is unusually small.
With `--seed 1` the vibrations are excited (T std 45 K), and the same 0.5 fs
step gives max |E − E0| = 5.0e-5 Eh over 50 steps, with the native backend and
with PySCF alike. The Validation section has longer runs. On a restart, the
temperature line gives absolute step numbers (e.g. `steps 51-70`).

Bad input stops the run with a one-line `aimd: error: ...` and exit status
2. This covers an unknown basis or element, a charge/multiplicity
combination that no electron count allows, a thermostat without
`--temperature`, XL-BOMD with Langevin or with a backend that does not accept
density guesses, a missing optional package, a negative `--seed`, a
checkpoint path in a missing directory (checked before the first step), and
a restart file that does not fit the options or the output files.

### Config files

`--config FILE` reads JSON (`.json`) or TOML (anything else; tomllib). The
keys are the long flag names, with `_` or `-`. An `xyz` key gives the
geometry, and a `backend_options` table is passed to the backend
constructor. Flags given on the command line override the file. Relative
`xyz` and `restart` paths are resolved against the file's directory.
[`examples/water_nvt.toml`](examples/water_nvt.toml) is a commented example
(CSVR + XL-BOMD, checkpoints every 200 steps):

```bash
aimd run --config examples/water_nvt.toml                 # as written
aimd run --config examples/water_nvt.toml --steps 100     # flags override
aimd run --config examples/water_nvt.toml --restart water_nvt.npz --steps 1000
```

### `aimd analyze`

These subcommands read the files that `aimd run` writes. Each one prints a
summary, writes a CSV (`-o`, default `<command>.csv`; `stats` writes one only
when `-o` is given) and, with `--plot FILE`, a figure. The figure is skipped
with a message if matplotlib is not installed. Dihedral statistics are
circular: the series is unwrapped about its circular mean before block
averaging, so a torsion fluctuating about ±180° gets a mean near ±180° and
its actual spread. A velocity file passed where positions are expected is
refused. `vacf` / `vdos` read the unit from each frame's `units=` tag;
`--units bohr/au_time|angstrom/fs` gives it for files without one (from other
programs). `stats --plot` shows the selected energies relative to their
first value, the temperature and any other column in separate panels.

```bash
aimd analyze rdf trajectory.xyz --pair O H --r-max 6          # g(r), coordination number
aimd analyze geometry trajectory.xyz --bond 0 1 --angle 1 0 2 --dihedral 0 1 2 3
aimd analyze vacf velocities.xyz --max-lag 400
aimd analyze vdos velocities.xyz                              # strongest bands in cm^-1
aimd analyze ir dipoles.csv --temperature 300 --correction harmonic
aimd analyze stats energies.csv --skip 200 -o stats.csv       # block-averaged errors
```

Distances are in Å, angles in degrees, times in fs and wavenumbers in
cm⁻¹. Atom indices are 0-based.

### `aimd view`

Writes one HTML file that plays a trajectory in any browser, offline: the
frames, the energy log and a small canvas renderer are embedded, with no
external scripts.

```bash
aimd view trajectory.xyz                         # -> trajectory.html
aimd view run.xyz --energies run_e.csv -o movie.html --every 2
```

- Ball-and-stick molecule: drag to rotate, wheel or pinch to zoom; play /
  pause (space), step (arrow keys), frame slider, playback speed, loop, and
  0-based atom index labels (the indices `aimd analyze geometry` takes).
- Bonds are recomputed for every frame from covalent radii (bonded if
  r < r_cov,i + r_cov,j + 0.4 Å, `--bond-tolerance`), so bonds that break or
  form appear and disappear. The fragments of each frame are listed as Hill
  formulas (e.g. `H2O`, then `HO + H` after a dissociation), in red once they
  differ from the first frame.
- With an energy log (`--energies`; by default `<stem>_energies.csv` or
  `energies.csv` next to the trajectory is used if present, `none` to leave
  it out): the change of E_pot, E_tot and the conserved energy from their
  first values in kcal/mol (the conserved energy is a flat line at 0 in a
  healthy run), and the temperature, with a cursor following the movie;
  clicking a chart jumps to that time. Frames are matched to log rows by
  step number.
- Each frame is translated so its centre of mass is at the origin, keeping a
  drifting molecule in view (`--no-center` for lab-frame positions).
  Trajectories longer than `--max-frames` (2000) are thinned by raising the
  stride; `--every N` thins explicitly.

### `aimd backends`

```
backend    optional dependency          description
harmonic   none needed                  harmonic model surface (not ab initio; tests)
hf         none needed                  native RHF / UHF, analytic gradients (McMurchie-Davidson, aimd.qc)
morse      none needed                  pairwise Morse model surface (not ab initio; tests, plumbing)
psi4       psi4 (NOT installed)         HF / DFT / MP2 analytic gradients through Psi4
pyscf      pyscf 2.14.0 (installed)     HF / DFT / MP2 analytic gradients through PySCF
```

## Python API

```python
from aimd import (MolecularSystem, CSVR, XLBOMD, CSVRThermostat, get_backend,
                  run_md, load_checkpoint, analysis, read_velocities)

system = MolecularSystem.from_xyz("examples/water.xyz")          # angstrom -> bohr
system.initialize_velocities(300.0, rng=1, remove_rotation=True)
backend = get_backend("hf")(system.symbols, basis="6-31g*")      # RHF (singlet)

integ = XLBOMD(backend, timestep_fs=0.5, k=5, thermostat=CSVRThermostat(300.0, 100.0, rng=2))
result = run_md(system, integ, 1000, trajectory="traj.xyz", energy_log="energies.csv",
                velocity_trajectory="vel.xyz", checkpoint_path="run.npz",
                checkpoint_every=100)
print(result.conserved_energy_drift)                  # Eh

# continue later, bit-for-bit
ckpt = load_checkpoint("run.npz")
backend = get_backend("hf")(ckpt.system.symbols, basis="6-31g*")
run_md(ckpt.system, ckpt.make_integrator(backend), 1000, restart=ckpt,
       trajectory="traj.xyz", energy_log="energies.csv", velocity_trajectory="vel.xyz")

vel = read_velocities("vel.xyz")
vdos = analysis.vibrational_dos(vel.velocities, vel.frame_interval_fs, masses=vel.masses)
print(vdos.peak(1000, 3000))                          # cm^-1
```

You can also use the native HF code directly, without any dynamics:

```python
from aimd.qc import run_scf, scf_gradient
res = run_scf(["O", "H", "H"], positions_bohr, basis="cc-pvdz")
grad = scf_gradient(res)                              # Eh/bohr, (N, 3)
```

**Adding a backend:**

1. Subclass `aimd.backends.ForceBackend`.
2. Set `name`.
3. Decorate the class with `@register_backend`.
4. Implement `compute(positions_bohr) -> GradientResult(energy, gradient)`.
5. Import the module in `aimd/backends/__init__.py`.

Check the result with `aimd.testing.max_gradient_error(backend, positions)`.
Forces that are not consistent with the energy make NVE runs drift.

## Units

Internally everything is in Hartree atomic units: bohr, hartree, electron
mass, and atomic time (≈ 0.0242 fs). Conversions happen only at the I/O
boundary:

- XYZ files are in Å.
- Timesteps, temperatures and friction are given in fs, K and 1/fs.
- Energies are logged in Eh.
- Velocity files are in bohr/au_time, with the unit stated on every frame.
- Dipoles are in e·bohr about the origin (1 e·bohr = 2.5417 D).

Backends receive positions in bohr, shape `(N, 3)`. They return the energy in
Eh and dE/dR in Eh/bohr, shape `(N, 3)`.

## Backends

| | `hf` (native) | `pyscf` | `psi4` | `morse`, `harmonic` |
|---|---|---|---|---|
| Methods | RHF, UHF | RHF/UHF, RKS/UKS (any libxc functional), MP2 | HF, DFT, MP2 (Psi4 methods) | model surfaces |
| Basis sets | STO-3G, 6-31G, 6-31G*, 6-31G**, cc-pVDZ (H–Ar, Cartesian) | any PySCF basis, spherical or Cartesian | any Psi4 basis | — |
| Gradients | analytic | analytic (DFT incl. grid response) | analytic | analytic |
| Density guess / XL-BOMD | yes | yes | no (fresh guess each step) | no |
| Dipole | yes | yes (not MP2) | no | no |
| Extra dependency | none (numba) | `pyscf` | `psi4` (conda) | none |
| Tested here | yes | yes | **no** (Psi4 not installed) | yes |

The `hf` backend is about 3–4× faster per MD step than `pyscf` for the small
systems measured below. That comes from a cached basis, density reuse and
numba kernels. PySCF is the choice for DFT, MP2, larger basis sets or spherical
d functions.

## Validation

All numbers below were measured on this repository's test machine (a
4-vCPU VM, Python 3.11, numba 0.68, PySCF 2.14.0). The tests that pin them
are listed under [Tests](#tests).

**Energies, gradients and dipoles against PySCF.** These were run with
`mol.cart=True` at randomly distorted geometries (σ = 0.05 bohr), with both
codes converged to 1e-12 Eh and an orbital gradient of 1e-9:

| System | Basis | Ref. | nao | max \|ΔE\| (Eh) | max \|Δ∇E\| (Eh/bohr) | max \|Δμ\| (e·bohr) |
|---|---|---|---|---|---|---|
| H₂O | STO-3G | RHF | 7 | 1.4e-13 | 4.9e-10 | 5.0e-9 |
| H₂O | 6-31G* | RHF | 19 | 1.4e-13 | 7.8e-11 | 9.8e-10 |
| H₂O | cc-pVDZ | RHF | 25 | 1.6e-13 | 1.7e-10 | 1.0e-9 |
| CH₃• | 6-31G* | UHF | 21 | 3.4e-13 | 5.1e-10 | 1.1e-9 |
| CH₃OH | 6-31G** | RHF | 50 | 1.8e-13 | 1.5e-10 | 7.9e-10 |
| H₂O⁺ | 6-31G | UHF | 13 | 1.7e-13 | 8.1e-11 | 3.5e-10 |

**Whole MD pipeline against PySCF.** This compares a 20-step NVE water run
through the CLI with the native backend against the same run with PySCF
forces (same seed, SCF orbital gradient 1e-9):

- energies agree to 6.9e-12 Eh;
- positions agree to 5.9e-9 bohr.

At the default orbital-gradient threshold of 1e-7 the agreement is 1.6e-9 Eh
and 1.5e-6 bohr. The difference scales with the threshold, so it is SCF
convergence noise that the dynamics amplifies.

**Geometries.** The example minima were optimised with the native gradients
(`max |∇E|` is given in each file):

| Molecule | Level | Result | Literature |
|---|---|---|---|
| H₂O | HF/STO-3G | r_OH = 0.9894 Å, θ = 100.03°, E = −74.965901 Eh | r = 0.989 Å, θ = 100.0° |
| CH₃• | UHF/6-31G* | E = −39.558992 Eh, ⟨S²⟩ = 0.7615 | |
| CH₃OH | RHF/6-31G* | E = −115.035418 Eh | |

**Harmonic frequencies and spectra.** PySCF's analytic Hessian at
`examples/water.xyz` gives harmonic wavenumbers of 2169.9, 4139.6 and
4390.7 cm⁻¹ for HF/STO-3G water. A 500 fs rotation-free NVE run analysed
with `aimd analyze` puts the bend at:

- 2172.8 cm⁻¹ in the VDOS;
- 2173.0 cm⁻¹ in the IR spectrum.

**Energy conservation (NVE, velocity Verlet)** for water/6-31G*, starting at
300 K with rotation kept:

| dt | duration | max \|E − E₀\| | fitted drift |
|---|---|---|---|
| 0.5 fs | 1 ps | 1.15e-4 Eh | +4.6e-7 Eh/ps |
| 0.25 fs | 1 ps | 2.89e-5 Eh | −2.6e-8 Eh/ps |

Halving dt divides the fluctuation by 3.99, the O(dt²) behaviour of Verlet
(also asserted in `test_nve_energy_error_scales_as_dt_squared`). There is no
systematic drift: the fitted slope over 1 ps is far below the fluctuation.

**XL-BOMD with a loose SCF.** Water/6-31G*, 1 ps at 0.5 fs, with the SCF
stopped at an energy change of 1e-6 and an orbital gradient of 1e-4:

| | max \|E − E₀\| | fitted drift | SCF iterations / step |
|---|---|---|---|
| BOMD, previous density as guess | 3.4e-3 Eh | −3.4e-3 Eh/ps | 4.55 |
| XL-BOMD (K = 5) | 2.7e-4 Eh | −1.5e-4 Eh/ps | 3.29 |

The tight-SCF reference run (first table) gives 1.15e-4 Eh. XL-BOMD cuts the
loose-SCF drift by a factor of 23 and needs fewer SCF iterations, but a
residual drift remains. When the SCF stops at a threshold and its cycle count
varies from step to step, the error is not purely time-reversible; see
`aimd/xlbomd.py`. With a tight SCF (orbital gradient 1e-9), XL-BOMD and BOMD
trajectories agree to 4e-8 bohr after 40 steps.

**Timing per MD step** (median, CLI, water and methanol at 300 K, default
thresholds, 4 threads):

| System | nao | `hf` | `pyscf` (cart) | SCF iterations |
|---|---|---|---|---|
| H₂O / STO-3G | 7 | 6.4 ms | 26.6 ms | 7.9 |
| H₂O / 6-31G* | 19 | 17.9 ms | 53.1 ms | 9.2 |
| CH₃• / 6-31G* (UHF) | 21 | 28 ms | | 9.9 |
| CH₃OH / 6-31G* | 38 | 137 ms | 383 ms | 9.4 |

Absolute timings on this shared VM vary by up to ~2× between sessions. The
`hf.py` docstring records an earlier, faster measurement (3.0 / 8.8 ms for
the two water rows). The ratio between the two backends was stable.

General contractions (cc-pVDZ) cost about as much as segmented bases of the
same size. The ERI kernels group the shells of a general contraction into one
block, so each primitive quartet is evaluated once rather than once per
pair of contracted shells. One warm step at a displaced geometry:

| System | nao | `hf` 6-31G* | `hf` cc-pVDZ | `pyscf` (cart) cc-pVDZ |
|---|---|---|---|---|
| Cl₂ | 38 | 0.18 s | 0.13 s (was 0.80 s) | 0.52 s |
| PCl₃ | 76 | 2.2 s | 1.8 s (was 9.0 s) | 2.9 s |

**Thermostats.** These are tested against closed-form canonical results on
model surfaces in `tests/test_thermostats.py`. Through the CLI, on water/STO-3G with rotation
removed (N_dof = 3, target 300 K, τ = 20 fs), single 4 ps runs gave these
means:

- CSVR, 5 seeds: 263, 275, 285, 288 and 294 K;
- NHC, 1 seed: 315 K.

The blocking errors are 6–25 K per run. Pooling the 4 CSVR seeds gives
285 ± 11 K. With only 3 degrees of freedom the canonical temperature
fluctuations are about 80 % and the energy exchange is slow, so these runs
check the thermostats only to about 5 %. On a 3-atom Morse cluster (same
N_dof), 4 × 20 ps per setting gave 293–303 K (±3–4 K) for CSVR and NHC at
dt = 0.1 and 0.4 fs.

## Limitations

- **Cartesian basis functions only (6d).** Energies match PySCF with
  `cart=True` or Psi4 with `puream false`, not their default spherical runs.
  cc-pVDZ, defined with 5d, therefore has extra functions. For Li, Be, Na and
  Mg, cc-pVDZ uses PySCF's older d exponents.
- **Memory for the ERI tensor.** The full tensor is stored, 8·nao⁴ bytes. The
  native backend refuses nao > 120 (about 1.6 GB) unless `max_nao` is raised,
  which in practice means about 6–7 heavy atoms plus hydrogens in 6-31G*
(ethanol is 57 AOs). There is
  no integral-direct J/K and no density fitting. Use the `pyscf` backend for
  larger systems.
- **Hartree–Fock only in the native code.** There is no DFT, MP2 or
  dispersion; use `pyscf` for those. A UHF singlet stays spin-restricted
  unless the guess breaks the symmetry. There is no stability analysis and
  no second-order SCF; hard open-shell cases may need `guess="gwh"` or a
  level shift via `--backend-option`.
- **Isolated molecules only.** There are no periodic boundary conditions,
  except in RDF analysis of an externally produced cubic-box trajectory.
- **Psi4 backend untested here.** Psi4 is not installed on the test machine.
  The backend is a thin wrapper, has no density-guess protocol and so cannot
  run XL-BOMD, and reports no dipole.
- **Restart of SCF backends.** Restarts are bit-for-bit for `hf` (tested).
  For other SCF codes, only the density is restored, not internal state such
  as DIIS history.
- **XL-BOMD** cannot be combined with Langevin dynamics. With an SCF stopped
  at a threshold whose cycle count changes between steps, it removes the
  systematic drift but not the noise.
- **Classical nuclei.** There is no zero-point energy or tunnelling. Use the
  quantum correction factors of `aimd analyze ir` for IR line shapes.
- **First-run compile time.** The numba kernels take about 40–60 s to compile
  on first use. Editing `aimd/qc/hermite.py`, `boys.py` or `basis.py` alone
  does not invalidate the cache: delete `aimd/qc/__pycache__` afterwards.

## Roadmap

1. Spherical-harmonic (5d/7f) basis functions, so energies match standard
   cc-pVXZ runs.
2. Integral-direct or density-fitted J/K, to lift the nao ≤ 120 limit.
3. Native Kohn–Sham DFT (grids, libxc), and a dispersion correction.
4. SCF stability analysis and a second-order (Newton) solver for difficult
   open-shell cases.
5. Psi4 backend: density-guess protocol (`GUESS READ`), dipoles, and testing
   in CI with Psi4 installed.
6. Periodic boundary conditions (Ewald for the nuclei; this requires a
   periodic electronic-structure backend).
7. Ring-polymer MD (nuclear quantum effects) and surface hopping (non-adiabatic
   dynamics).

## Tests

```bash
pytest                 # default suite (slow tests deselected)
pytest -m slow         # only the slow tests
pytest -m ""           # everything
```

On the 4-vCPU test machine, with numba's cache warm, the default suite takes
about 2 minutes. The first run compiles the kernels and takes about a minute
longer. The suite checks:

- integrals, SCF energies and gradients against PySCF, and gradients against
  finite differences;
- NVE energy conservation with O(dt²) scaling, time reversibility, and
  momentum and angular-momentum conservation;
- canonical sampling of the thermostats against closed-form results;
- that XL-BOMD follows BOMD;
- bit-for-bit restarts;
- file formats and readers;
- the analysis functions against analytic results;
- the CLI end to end (`tests/test_cli.py`, `tests/test_end_to_end.py`).

The Psi4 gradient test runs only when Psi4 is installed.
