# Ab-Initio-MD

Born–Oppenheimer ab initio molecular dynamics: nuclei move classically on a
potential-energy surface computed on the fly by an electronic-structure code.

At each step the engine asks a **force backend** for the energy and nuclear
gradient at the current geometry, then advances positions and velocities with
an integrator.

## Layout

```
aimd/
  units.py            atomic units internally; conversions at the I/O boundary
  elements.py         atomic numbers and masses (H–Kr)
  system.py           MolecularSystem: positions, velocities, masses, T, XYZ input
  backends/
    base.py           ForceBackend interface + GradientResult
    registry.py       name -> backend class (@register_backend)
    psi4_backend.py   HF / DFT / MP2 gradients through Psi4
    morse.py          analytic pairwise Morse model (tests and plumbing)
  integrators.py      VelocityVerlet (NVE, optional Berendsen), LangevinBAOAB (NVT)
  md.py               run_md driver, MDResult
  trajectory.py       multi-frame XYZ writer, CSV energy log
  testing.py          finite-difference gradient check for new backends
  cli.py              `aimd` command
```

The backend interface follows the `QuantumBackend` / registry pattern from
[Quantize](https://github.com/Nathan-W123/Quantize). The planned native HF
backend will be ported from
[HF-SCF-Engine](https://github.com/Nathan-W123/HF-SCF-Engine) (see Roadmap).

## Units

Internally everything is in Hartree atomic units: bohr, hartree, electron
mass, atomic time (≈ 0.0242 fs). XYZ files are in ångström; timesteps,
temperatures and friction are given in fs, K and 1/fs.

Backends receive positions in bohr, shape `(N, 3)`, and return the energy in
hartree and the gradient `dE/dR` in hartree/bohr, shape `(N, 3)`.

## Install

```bash
pip install -e ".[dev]"
conda install -c conda-forge psi4      # for the psi4 backend
```

## Usage

```bash
# NVE water at HF/STO-3G, 0.5 fs steps, starting at 300 K
aimd run examples/water.xyz --backend psi4 --method hf --basis sto-3g \
    --steps 200 --dt 0.5 --temperature 300 --seed 1

# NVT with a Langevin thermostat
aimd run examples/water.xyz --backend psi4 --method b3lyp --basis 6-31g* \
    --thermostat langevin --temperature 300 --friction 0.01

# No Psi4? Exercise the engine on the model surface
aimd run examples/h4_cluster.xyz --backend morse --steps 1000 --dt 0.2
```

This writes `trajectory.xyz` (open it in VMD, Avogadro or ASE) and
`energies.csv` (step, time, potential/kinetic/total energy, temperature).

From Python:

```python
from aimd import MolecularSystem, VelocityVerlet, get_backend, run_md

system = MolecularSystem.from_xyz("examples/water.xyz")
system.initialize_velocities(300.0, rng=1)
backend = get_backend("psi4")(system.symbols, method="hf", basis="sto-3g")
result = run_md(system, VelocityVerlet(backend, timestep_fs=0.5), n_steps=200,
                trajectory="water.xyz", energy_log="water.csv")
print(result.total_energy_drift)
```

### Adding a backend

```python
from aimd.backends import ForceBackend, GradientResult, register_backend

@register_backend
class MyBackend(ForceBackend):
    name = "mine"
    def compute(self, positions):          # bohr, (N, 3)
        ...
        return GradientResult(energy=e, gradient=g)   # Eh, Eh/bohr (N, 3)
```

Import the module in `aimd/backends/__init__.py`, then check it with
`aimd.testing.max_gradient_error(backend, positions)`. If forces aren't
consistent with the energy, NVE runs will drift.

## Tests

```bash
pytest
```

The tests check analytic against finite-difference gradients, NVE energy
conservation with O(dt²) error scaling, time reversibility, momentum
conservation, Berendsen/Langevin temperature control, and the trajectory and
log output. The Psi4 gradient test runs only when Psi4 is installed.

## Roadmap

1. **Native HF backend.** Port `integrals.py` and the RHF SCF loop from
   HF-SCF-Engine into a stateful backend: a cached basis, the previous
   density as the next SCF guess, and symmetry disabled.
2. **Analytic RHF gradients.** Add derivative overlap, kinetic, nuclear and
   ERI integrals and the energy-weighted-density term. Validate against
   `aimd.testing` and Psi4.
3. UHF, for bond breaking and open-shell systems.
4. SCF guess reuse in the Psi4 backend (`GUESS READ`).
5. Extended-Lagrangian BOMD (XL-BOMD) for long-time energy stability with
   loose SCF convergence.
6. Restart files, removal of rotation for isolated molecules, and analysis
   tools (radial distribution functions, vibrational spectra from velocity
   autocorrelation).
