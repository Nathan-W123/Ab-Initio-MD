"""
Analysis tools against independent references.

  - units: cm^-1 per rad/au_time vs the CODATA hartree-wavenumber relation,
    c in atomic units vs 1/alpha, and a signal of known period in fs (SI only);
  - autocorrelation vs the direct O(n^2) sum;
  - VDOS: Morse H2 at 20 K vs omega = sqrt(2 D a^2 / mu); a coupled harmonic
    system vs its prescribed normal-mode frequencies, with band areas equal to
    the mode energies; the normalisation integral equals C(0);
  - IR: synthetic dipoles with known frequencies and amplitudes; absolute band
    intensity vs the double-harmonic formula (42.256 km/mol per (D/A)^2/amu,
    rebuilt from SI definitions); quantum corrections vs c2 = hc/k; end to
    end through run_md and the dipole log with a point-charge Morse diatomic;
  - RDF: cube / rock-salt cluster shells; uniform ball closed form; periodic
    ideal gas; simple cubic lattice; normalisation identities;
  - geometry: hand-computed water; IUPAC dihedral sign from its definition;
    an independent projection formula; minimum image;
  - block averaging: AR(1) series vs the exact standard error of the mean.
"""

import math

import numpy as np
import pytest
from scipy import stats
from scipy.signal import lfilter

from aimd import analysis
from aimd.analysis import spectra
from aimd.backends.harmonic import HarmonicBackend
from aimd.backends.morse import MorseBackend
from aimd.integrators import LangevinBAOAB, VelocityVerlet
from aimd.md import run_md
from aimd.system import MolecularSystem
from aimd.trajectory import read_dipole_log, read_energy_log
from aimd.units import AMU_TO_AU, BOHR_TO_ANG, FS_TO_AU_TIME, KB_AU

# Independent constants (CODATA 2018 / SI), not taken from aimd.
HARTREE_TO_CM1 = 219474.6313632          # E_h / (h c) in cm^-1
INV_FINE_STRUCTURE = 137.035999084       # c in atomic units
C_CM_S = 2.99792458e10                   # cm / s
C2_CM_K = 6.62607015e-34 * 2.99792458e10 / 1.380649e-23   # hc/k in cm K (exact SI)


# ── Units ─────────────────────────────────────────────────────────────────────

def test_frequency_conversion_matches_codata():
    # hbar = 1: omega = 1 / au_time is an energy of 1 hartree.
    assert analysis.angular_frequency_to_wavenumber(1.0) == pytest.approx(
        HARTREE_TO_CM1, rel=1e-10)
    assert spectra.C_AU == pytest.approx(INV_FINE_STRUCTURE, rel=1e-10)
    nu = np.array([0.0, 1000.0, 4400.0])
    assert np.allclose(analysis.angular_frequency_to_wavenumber(
        analysis.wavenumber_to_angular_frequency(nu)), nu, rtol=1e-14)


def test_vdos_of_a_signal_with_known_period_in_fs():
    """v(t) = cos(2 pi t / P): the peak must sit at 1 / (c P), SI units only."""
    period_fs, dt_fs, n = 20.0, 0.25, 8000
    t_fs = np.arange(n) * dt_fs
    v = np.zeros((n, 1, 3))
    v[:, 0, 0] = np.cos(2 * np.pi * t_fs / period_fs)
    spec = analysis.vibrational_dos(v, dt_fs)
    expected = 1.0 / (C_CM_S * period_fs * 1e-15)              # 1667.8 cm^-1
    assert spec.resolution == pytest.approx(1.0 / (C_CM_S * (n // 2) * dt_fs * 1e-15))
    # Measured |error| 0.025 cm^-1 (resolution 33.4 cm^-1).
    assert abs(spec.peak() - expected) < 0.01 * spec.resolution


# ── Correlation functions ─────────────────────────────────────────────────────

def test_autocorrelation_matches_direct_sum():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(301, 4, 3)) + 0.3
    w = rng.uniform(1.0, 5.0, size=(4, 1))
    n, max_lag = x.shape[0], 120
    for subtract in (False, True):
        y = x - x.mean(axis=0) if subtract else x
        direct = np.array([
            np.sum(w * np.sum(y[: n - k] * y[k:], axis=0)) / (n - k) for k in range(max_lag + 1)
        ])
        fast = analysis.autocorrelation(x, max_lag, weights=w, subtract_mean=subtract)
        assert np.allclose(fast, direct, rtol=0.0, atol=1e-12 * direct[0])
    assert analysis.autocorrelation(x[:, 0, 0]).shape == (n,)


def test_fft_lengths_are_fast_and_long_enough():
    for n in (2, 7, 31, 97, 1000, 4097, 12345):
        m = spectra._fast_length(n)
        e = spectra._fast_length(n, even=True)
        assert m >= n and e >= n and e % 2 == 0
        for k in (m, e):
            for p in (2, 3, 5):
                while k % p == 0:
                    k //= p
            assert k == 1


def test_vacf_definitions():
    rng = np.random.default_rng(2)
    v = rng.normal(size=(200, 3, 3))
    masses = np.array([1.0, 2.0, 16.0]) * AMU_TO_AU
    vacf = analysis.velocity_autocorrelation(v, 0.5, masses=masses, max_lag=50)
    # Lag 0 of the mass-weighted VACF is <sum_i m_i v_i^2> = 2 <E_kin>.
    assert vacf.acf[0] == pytest.approx(np.mean(np.sum(masses[:, None] * v**2, axis=(1, 2))))
    assert vacf.normalized[0] == 1.0 and vacf.mass_weighted
    assert np.allclose(vacf.time_fs, 0.5 * np.arange(51))
    plain = analysis.velocity_autocorrelation(v, 0.5, atoms=[1], max_lag=50)
    assert plain.acf[0] == pytest.approx(np.mean(np.sum(v[:, 1] ** 2, axis=1)))


# ── Vibrational density of states ─────────────────────────────────────────────

def _morse_h2(dt_fs, n_steps, temperature_k, seed):
    backend = MorseBackend(["H", "H"])
    s = MolecularSystem(["H", "H"], [[0, 0, 0], [0, 0, backend.r_eq]])
    # COM and rotation removed: a pure vibration (L = 0 is conserved).
    s.initialize_velocities(temperature_k, rng=seed, remove_rotation=True)
    frames = []
    res = run_md(s, VelocityVerlet(backend, dt_fs), n_steps,
                 callback=lambda _r: frames.append(s.velocities.copy()))
    return s, backend, np.array(frames), res


def test_vdos_peak_of_morse_h2_matches_harmonic_frequency():
    dt_fs = 0.1
    s, b, v, res = _morse_h2(dt_fs, 20000, 20.0, seed=1)
    mu = s.masses[0] * s.masses[1] / s.masses.sum()
    omega = math.sqrt(2.0 * b.depth * b.alpha**2 / mu)        # k = 2 D a^2
    nu_harm = omega * HARTREE_TO_CM1
    # The model's defaults are H2's: experimental omega_e = 4401.2 cm^-1
    # (Huber & Herzberg); the Morse fit gives 4398.2.
    assert nu_harm == pytest.approx(4401.2, rel=2e-3)

    spec = analysis.vibrational_dos(v, dt_fs, masses=s.masses)
    assert spec.resolution == pytest.approx(33.36, abs=0.01)     # 1 / (c * 1 ps)
    peak = spec.peak()
    assert abs(peak - nu_harm) < spec.resolution
    # Sharper: the classical Morse frequency at the vibrational energy E,
    # omega sqrt(1 - E / D), with Verlet's phase advance arccos(1 - (w h)^2/2) / h.
    # Measured |difference| 0.012 cm^-1.
    h = dt_fs * FS_TO_AU_TIME
    e_vib = res.records[0]["total_Eh"] + b.depth
    nu_md = math.acos(1.0 - 0.5 * (omega * h) ** 2) / h * math.sqrt(1.0 - e_vib / b.depth)
    assert abs(peak - nu_md * HARTREE_TO_CM1) < 0.3

    # Normalisation: the spectrum integrates to C(0) = <sum m v^2> exactly.
    c0 = np.mean(np.sum(s.masses[:, None] * v**2, axis=(1, 2)))
    assert spec.band_area() == pytest.approx(c0, rel=1e-10)
    # ... which for a (nearly harmonic) vibration is its energy, kT / 2 here.
    assert spec.band_area(nu_harm - 200, nu_harm + 200) == pytest.approx(e_vib, rel=2e-3)


def test_vdos_of_coupled_harmonic_system_resolves_normal_modes():
    """
    Tethered O-H-H with a full Hessian built from prescribed frequencies
    600..3800 cm^-1 and random mixing: mass-weighted VDOS peaks at the normal
    modes, and each band's area is the energy of that mode (equipartition
    between kinetic and potential energy of a harmonic mode: <q'^2> = E).
    """
    rng = np.random.default_rng(7)
    s = MolecularSystem(["O", "H", "H"], np.zeros((3, 3)))
    m3 = np.repeat(s.masses, 3)
    nu = np.linspace(600.0, 3800.0, 9)
    omega = nu / HARTREE_TO_CM1
    u, _ = np.linalg.qr(rng.normal(size=(9, 9)))
    hess = np.sqrt(m3)[:, None] * (u @ np.diag(omega**2) @ u.T) * np.sqrt(m3)[None, :]
    backend = HarmonicBackend(s.symbols, hessian=hess)
    s.positions = rng.normal(size=(3, 3)) * 0.05
    s.velocities = rng.normal(size=(3, 3)) * np.sqrt(KB_AU * 300.0 / s.masses[:, None])
    q = u.T @ (np.sqrt(m3) * s.positions.ravel())
    qdot = u.T @ (np.sqrt(m3) * s.velocities.ravel())
    energy = 0.5 * (qdot**2 + omega**2 * q**2)

    dt_fs, frames = 0.1, []
    run_md(s, VelocityVerlet(backend, dt_fs), 20000,
           callback=lambda _r: frames.append(s.velocities.copy()))
    spec = analysis.vibrational_dos(np.array(frames), dt_fs, masses=s.masses)
    h = dt_fs * FS_TO_AU_TIME
    nu_verlet = np.arccos(1.0 - 0.5 * (omega * h) ** 2) / h * HARTREE_TO_CM1
    for k in range(9):
        peak = spec.peak(nu[k] - 150, nu[k] + 150)
        assert abs(peak - nu[k]) < spec.resolution
        # Measured |peak - Verlet frequency| <= 0.072 cm^-1 (resolution 33).
        assert abs(peak - nu_verlet[k]) < 0.25
        # Measured band area / mode energy within 0.76 % (finite-time beats).
        assert spec.band_area(nu[k] - 200, nu[k] + 200) == pytest.approx(energy[k], rel=0.015)
    # Partial VDOS are additive over atoms.
    parts = [analysis.vibrational_dos(np.array(frames), dt_fs, masses=s.masses, atoms=[i])
             for i in range(3)]
    assert np.allclose(sum(p.intensity for p in parts), spec.intensity,
                       rtol=0.0, atol=1e-10 * spec.intensity.max())


@pytest.mark.parametrize("window", analysis.WINDOWS)
def test_spectrum_normalisation_holds_for_every_window(window):
    rng = np.random.default_rng(4)
    acf = np.cumsum(rng.normal(size=301))[::-1] * 1e-3 + 2.0
    spec = analysis.correlation_spectrum(acf, 0.3, window=window, zero_pad=3)
    assert spec.band_area() == pytest.approx(acf[0], rel=1e-12)
    w = analysis.lag_window(window, 300)
    assert w[0] == 1.0
    if window in ("hann", "blackman"):
        assert abs(w[-1]) < 1e-15


def test_spectrum_equals_direct_cosine_sum_in_si_units():
    """
    At every returned grid point, for every window and padding factor:
    S(nu~) = 2 c dt [w_0 C_0 + 2 sum_k w_k C_k cos(2 pi c nu~ k dt)], evaluated
    directly with c in cm/s and dt in s (no aimd constants), with the window
    formulas written out and the grid ending exactly at Nyquist 1 / (2 c dt).
    Zero padding therefore only samples the same function more densely.
    """
    rng = np.random.default_rng(17)
    dt_fs, m = 0.37, 41
    acf = rng.normal(size=m + 1)
    u = np.arange(m + 1) / m
    windows = {"hann": 0.5 * (1 + np.cos(np.pi * u)),
               "blackman": 0.42 + 0.5 * np.cos(np.pi * u) + 0.08 * np.cos(2 * np.pi * u),
               "hamming": 0.54 + 0.46 * np.cos(np.pi * u),
               "none": np.ones(m + 1)}
    dt_s = dt_fs * 1e-15
    for window, w in windows.items():
        w = w / w[0]
        for zero_pad in (1, 3, 4):
            spec = analysis.correlation_spectrum(acf, dt_fs, window=window, zero_pad=zero_pad)
            phase = 2 * np.pi * C_CM_S * np.outer(spec.wavenumber, np.arange(1, m + 1)) * dt_s
            direct = 2 * C_CM_S * dt_s * (acf[0] + 2 * np.cos(phase) @ (w[1:] * acf[1:]))
            assert np.allclose(spec.intensity, direct, rtol=0.0, atol=1e-11 * np.abs(direct).max())
            assert spec.wavenumber[-1] == pytest.approx(1 / (2 * C_CM_S * dt_s), rel=1e-12)
            assert spec.wavenumber.size >= zero_pad * (m + 1)


def test_hann_window_suppresses_truncation_ripple():
    """
    A 1500 cm^-1 cosine cut off at lag M (resolution 66.7 cm^-1). 2000-3000
    cm^-1 lies 7.5-22 resolutions away, where the measured leakage relative
    to the peak is 2.3e-2 (rectangular, sinc sidelobes) and 8.5e-5 (Hann).
    """
    dt_fs, m = 0.5, 1000
    w0 = 1500.0 / HARTREE_TO_CM1
    acf = np.cos(w0 * np.arange(m + 1) * dt_fs * FS_TO_AU_TIME)
    far = lambda s: np.abs(s.intensity[(s.wavenumber > 2000) & (s.wavenumber < 3000)]).max()
    rect = analysis.correlation_spectrum(acf, dt_fs, window="none")
    hann = analysis.correlation_spectrum(acf, dt_fs, window="hann")
    assert far(rect) > 1e-2 * rect.intensity.max()
    assert far(hann) < 2e-4 * hann.intensity.max()
    # Hann line width (FWHM, half-maximum crossings interpolated linearly)
    # equals the stated resolution 1 / (c M dt); measured ratio 0.988.
    y, nu = hann.intensity, hann.wavenumber
    k = int(np.argmax(y))
    lo = k - int(np.argmax(y[k::-1] < 0.5 * y[k]))            # first point below half
    hi = k + int(np.argmax(y[k:] < 0.5 * y[k]))
    cross = lambda i, j: nu[i] + (0.5 * y[k] - y[i]) * (nu[j] - nu[i]) / (y[j] - y[i])
    assert cross(hi - 1, hi) - cross(lo, lo + 1) == pytest.approx(hann.resolution, rel=0.03)


# ── IR spectra ────────────────────────────────────────────────────────────────

def _synthetic_dipoles(nus, amps, dt_fs, n, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt_fs * FS_TO_AU_TIME
    w = np.asarray(nus) / HARTREE_TO_CM1
    u = rng.normal(size=(len(nus), 3))
    u /= np.linalg.norm(u, axis=1)[:, None]
    phase = rng.uniform(0.0, 2 * np.pi, len(nus))
    mu = np.array([0.3, -0.1, 0.7]) + np.einsum(
        "k,kt,ki->ti", np.asarray(amps), np.cos(np.outer(w, t) + phase[:, None]), u)
    return mu, w


def test_ir_peaks_and_band_areas_of_synthetic_dipole():
    nus, amps = [830.0, 1650.0, 3420.0], [0.05, 0.02, 0.01]
    mu, w = _synthetic_dipoles(nus, amps, dt_fs=0.5, n=8000)
    spec = analysis.ir_spectrum(mu, 0.5)
    assert len(spec.wavenumber) > 0 and spec.unit.startswith("(e bohr")
    for nu, a, wk in zip(nus, amps, w):
        # Measured |error| <= 0.007 cm^-1 (resolution 16.7 cm^-1).
        assert abs(spec.peak(nu - 100, nu + 100) - nu) < 0.01 * spec.resolution
        # <mu'^2> of a cos(w t) line is a^2 w^2 / 2; measured within 0.07 %.
        area = spec.band_area(nu - 8 * spec.resolution, nu + 8 * spec.resolution)
        assert area == pytest.approx(0.5 * a**2 * wk**2, rel=5e-3)
    # Nothing else in the spectrum.
    gap = (spec.wavenumber > 2200) & (spec.wavenumber < 3000)
    assert np.abs(spec.intensity[gap]).max() < 1e-4 * spec.intensity.max()


def test_finite_difference_attenuation_is_removed():
    """At dt = 1 fs a 3400 cm^-1 line loses sinc^2(w dt / 2) = 3.4 % to the difference quotient."""
    nu, a, dt_fs = 3400.0, 0.02, 1.0
    mu, w = _synthetic_dipoles([nu], [a], dt_fs, 4000)
    exact = 0.5 * a**2 * w[0] ** 2
    lo, hi = nu - 200, nu + 200
    corrected = analysis.ir_spectrum(mu, dt_fs).band_area(lo, hi)
    raw = analysis.ir_spectrum(mu, dt_fs, finite_difference_correction=False).band_area(lo, hi)
    half_phase = 0.5 * w[0] * dt_fs * FS_TO_AU_TIME
    sinc2 = (math.sin(half_phase) / half_phase) ** 2
    assert sinc2 < 0.97
    assert corrected == pytest.approx(exact, rel=2e-3)
    assert raw == pytest.approx(sinc2 * exact, rel=2e-3)


def _double_harmonic_km_mol(dmu_dq_au: float) -> float:
    """pi N_A |dmu/dQ|^2 / (3 * 4 pi eps0 * c^2) in km/mol, from SI definitions."""
    e, a0, me = 1.602176634e-19, 0.529177210903e-10, 9.1093837015e-31
    mu0_over_4pi = 1.00000000055e-7                  # = 1 / (4 pi eps0 c^2)
    dmu_dq = dmu_dq_au * e * a0 / (math.sqrt(me) * a0)
    return 6.02214076e23 * math.pi / 3.0 * mu0_over_4pi * dmu_dq**2 / 1e3


def test_double_harmonic_reference_reproduces_literature_constant():
    debye = 1e-21 / 2.99792458e8                     # C m
    amu, me, e, a0 = 1.66053906660e-27, 9.1093837015e-31, 1.602176634e-19, 0.529177210903e-10
    # (D / angstrom) / sqrt(amu) expressed in e / sqrt(m_e):
    unit = (debye / 1e-10) / math.sqrt(amu) / (e * a0 / (math.sqrt(me) * a0))
    assert _double_harmonic_km_mol(unit) == pytest.approx(42.2561, rel=1e-5)


def test_ir_absolute_intensity_of_a_thermal_harmonic_mode():
    """
    mu = mu0 + (dmu/dQ) Q(t), Q = A cos(w t) with mode energy w^2 A^2 / 2 = kT:
    with the harmonic (= classical) correction the band integrates to the
    double-harmonic intensity, temperature independent.
    """
    temperature, nu0, dmu_dq = 300.0, 1700.0, 0.013          # e / sqrt(m_e)
    w0 = nu0 / HARTREE_TO_CM1
    t = np.arange(8000) * 0.5 * FS_TO_AU_TIME
    amp = math.sqrt(2.0 * KB_AU * temperature) / w0
    mu = np.array([0.1, 0.2, 0.3]) + np.outer(dmu_dq * amp * np.cos(w0 * t + 0.3), [0.6, 0.8, 0.0])
    spec = analysis.ir_spectrum(mu, 0.5, temperature_k=temperature)
    assert spec.unit == "km mol^-1 per cm^-1"
    # Measured within 0.06 %.
    assert spec.band_area(nu0 - 300, nu0 + 300) == pytest.approx(
        _double_harmonic_km_mol(dmu_dq), rel=3e-3)
    classical = analysis.ir_spectrum(mu, 0.5, temperature_k=temperature,
                                     quantum_correction="classical")
    assert np.array_equal(classical.intensity, spec.intensity)


@pytest.mark.parametrize("kind", ["standard", "schofield"])
def test_quantum_correction_factors(kind):
    temperature = 300.0
    mu, _ = _synthetic_dipoles([900.0, 2500.0], [0.03, 0.01], 0.5, 4000)
    harm = analysis.ir_spectrum(mu, 0.5, temperature_k=temperature)
    corr = analysis.ir_spectrum(mu, 0.5, temperature_k=temperature, quantum_correction=kind)
    nu = harm.wavenumber
    x = C2_CM_K * nu[1:] / temperature                         # h c nu / kT
    expected = 2 * np.tanh(x / 2) / x if kind == "standard" else 2 * np.sinh(x / 2) / x
    sel = harm.intensity[1:] != 0.0
    assert np.allclose(corr.intensity[1:][sel] / harm.intensity[1:][sel], expected[sel], rtol=1e-8)
    r = analysis.quantum_correction_factor(np.array([0.0, 1e-9]), temperature, kind)
    assert np.allclose(r, 1.0)
    with pytest.raises(ValueError, match="temperature"):
        analysis.ir_spectrum(mu, 0.5, quantum_correction=kind)


def test_ir_rejects_missing_dipoles():
    mu = np.zeros((10, 3))
    mu[4] = np.nan
    with pytest.raises(ValueError, match="nan"):
        analysis.ir_spectrum(mu, 0.5)
    with pytest.raises(ValueError, match="unknown quantum correction"):
        analysis.quantum_correction_factor(np.ones(3), 300.0, "bogus")
    with pytest.raises(ValueError, match="unknown quantum correction"):
        analysis.ir_spectrum(np.ones((10, 3)), 0.5, quantum_correction="Harmonic")


def test_ir_rejects_bad_temperatures():
    """
    Regression: a nan temperature passed the ``<= 0`` check and returned an
    all-nan spectrum silently; None reached quantum_correction_factor as a
    TypeError.
    """
    mu, _ = _synthetic_dipoles([1000.0], [0.02], 0.5, 400)
    for temp in (0.0, -10.0, math.nan, math.inf):
        for kind in ("harmonic", "standard", "schofield"):
            with pytest.raises(ValueError, match="temperature_k"):
                analysis.ir_spectrum(mu, 0.5, temperature_k=temp, quantum_correction=kind)
        for kind in ("standard", "schofield"):
            with pytest.raises(ValueError, match="temperature_k"):
                analysis.quantum_correction_factor(np.ones(3), temp, kind)
    with pytest.raises(ValueError, match="temperature_k"):
        analysis.quantum_correction_factor(np.ones(3), None, "standard")


def test_spectra_reject_bad_frame_spacing_and_nonfinite_velocities():
    v = np.random.default_rng(18).normal(size=(50, 2, 3))
    for dt in (0.0, -0.5, math.nan, math.inf):
        with pytest.raises(ValueError, match="dt_fs"):
            analysis.vibrational_dos(v, dt)
        with pytest.raises(ValueError, match="dt_fs"):
            analysis.velocity_autocorrelation(v, dt)
        with pytest.raises(ValueError, match="dt_fs"):
            analysis.ir_spectrum(v[:, 0], dt)
    v[7, 1, 2] = np.nan
    with pytest.raises(ValueError, match="nan"):
        analysis.vibrational_dos(v, 0.5)


class PointChargeMorse(MorseBackend):
    """Morse diatomic carrying charges +q / -q: mu = q (r_0 - r_1)."""

    def __init__(self, symbols, q=0.4, **kw):
        super().__init__(symbols, **kw)
        self.q = q

    def compute(self, positions):
        res = super().compute(positions)
        x = np.asarray(positions)
        res.dipole = self.q * (x[0] - x[1])
        return res


def test_ir_spectrum_end_to_end_through_dipole_log(tmp_path):
    """
    H-F-mass Morse diatomic with point charges, run with dipole_log: the IR
    band sits at sqrt(k / mu) and integrates to the double-harmonic intensity
    of dmu/dQ = q / sqrt(mu) scaled by E_vib / kT (one molecule, NVE).
    """
    temperature, dt_fs = 30.0, 0.2
    backend = PointChargeMorse(["H", "F"])
    s = MolecularSystem(["H", "F"], [[0, 0, 0], [0, 0, backend.r_eq]])
    s.initialize_velocities(temperature, rng=5, remove_rotation=True)
    log, energies = tmp_path / "dip.csv", tmp_path / "e.csv"
    run_md(s, VelocityVerlet(backend, dt_fs), 10000, dipole_log=log, energy_log=energies,
           write_every=2)
    dip = read_dipole_log(log)
    assert dip.frame_interval_fs == pytest.approx(2 * dt_fs, rel=1e-12)
    spec = analysis.ir_spectrum(dip.dipole, dip.frame_interval_fs, temperature_k=temperature)

    mu_red = s.masses[0] * s.masses[1] / s.masses.sum()
    nu = math.sqrt(2.0 * backend.depth * backend.alpha**2 / mu_red) * HARTREE_TO_CM1
    assert abs(spec.peak(nu - 300, nu + 300) - nu) < spec.resolution
    e_vib = read_energy_log(energies)["total_Eh"][0] + backend.depth
    thermal = _double_harmonic_km_mol(backend.q / math.sqrt(mu_red))
    expected = thermal * e_vib / (KB_AU * temperature)
    # Measured within 0.44 % (Morse anharmonicity at E/D = 3e-4 is negligible);
    # the peak is 1.5 cm^-1 above sqrt(k / mu), Verlet's phase error at 0.2 fs.
    assert spec.band_area(nu - 300, nu + 300) == pytest.approx(expected, rel=0.01)


# ── Radial distribution function ──────────────────────────────────────────────

def _cube(a, symbols):
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
    return corners * a, [symbols[(i + j + k) % 2] for i, j, k in corners.astype(int)]


def test_rdf_of_a_cube_cluster():
    """8 atoms on a cube of edge a: 12 edges, 12 face and 4 body diagonals."""
    a = 2.0
    x, sym = _cube(a, ["Ar", "Ar"])
    r_max, n_bins = 2.0 * a, 40                                # dr = 0.1: no edge on a shell
    res = analysis.radial_distribution(x, sym, ("Ar", "Ar"), r_max, n_bins)
    assert res.n_pairs == 56 and not res.periodic
    shells = {a: 24, a * math.sqrt(2): 24, a * math.sqrt(3): 8}   # ordered pairs
    expected = np.zeros(n_bins)
    for r, c in shells.items():
        expected[int(r / (r_max / n_bins))] += c
    assert np.array_equal(res.counts, expected)
    # Documented cluster normalisation: V = 4 pi r_max^3 / 3.
    dv = 4 * math.pi / 3 * (res.edges[1:] ** 3 - res.edges[:-1] ** 3)
    v = 4 * math.pi / 3 * r_max**3
    assert np.allclose(res.g, expected / (56 * dv / v), rtol=1e-12)
    assert np.sum(res.g * dv) / v == pytest.approx(1.0)       # all pairs < r_max
    at = lambda r: res.coordination[int(r / (r_max / n_bins))]
    assert (at(1.05 * a), at(1.5 * a), at(1.8 * a)) == (3.0, 6.0, 7.0)


def test_rdf_selects_element_pairs():
    a = 2.5
    x, sym = _cube(a, ["Na", "Cl"])                             # rock-salt cube
    res = analysis.radial_distribution(x, sym, ("Na", "Cl"), 2.0 * a, 50)
    assert (res.n_a, res.n_b, res.n_pairs) == (4, 4, 16)
    k = lambda r: int(r / (2.0 * a / 50))
    # Each Na: 3 Cl along the edges, 1 Cl across the body diagonal.
    assert res.coordination[k(1.1 * a)] == 3.0 and res.coordination[k(1.9 * a)] == 4.0
    same = analysis.radial_distribution(x, sym, ("Na", "Na"), 2.0 * a, 50)
    assert same.n_pairs == 12
    assert same.coordination[k(1.3 * a)] == 0.0 and same.coordination[k(1.9 * a)] == 3.0
    by_index = analysis.radial_distribution(
        x, None, ([i for i, s in enumerate(sym) if s == "Na"],
                  [i for i, s in enumerate(sym) if s == "Cl"]), 2.0 * a, 50)
    assert np.array_equal(by_index.g, res.g)
    # Regression: symbols as a NumPy array raised "truth value ... ambiguous".
    assert np.array_equal(analysis.radial_distribution(
        x, np.array(sym), ("Na", "Cl"), 2.0 * a, 50).g, res.g)
    with pytest.raises(ValueError, match="element"):
        analysis.radial_distribution(x, sym, ("Na", "K"), 2.0 * a)


def test_rdf_coordination_of_unequal_sets():
    """
    Two rigid waters 10 bohr apart, O-H = 1.83 bohr: every O has 2 H within
    2.45 bohr and every H one O, so n_OH = 2 and n_HO = 1 (normalised by the
    size of the first set, |A| != |B| here).
    """
    bond, angle = 1.83, math.radians(104.5)           # radii mid-bin (dr = 0.1)
    water = np.array([[0, 0, 0], [bond, 0, 0], [bond * math.cos(angle), bond * math.sin(angle), 0]])
    x = np.concatenate([water, water + [10.0, 0, 0]])
    sym = ["O", "H", "H"] * 2
    r_max, n_bins = 4.0, 40
    k = lambda r: int(r / (r_max / n_bins))
    oh = analysis.radial_distribution(x, sym, ("O", "H"), r_max, n_bins)
    ho = analysis.radial_distribution(x, sym, ("H", "O"), r_max, n_bins)
    assert (oh.n_a, oh.n_b, oh.n_pairs) == (2, 4, 8) and ho.n_pairs == 8
    assert oh.counts[k(bond)] == 4 and oh.counts.sum() == 4     # the other water is > r_max away
    assert oh.coordination[k(2.45)] == 2.0 and ho.coordination[k(2.45)] == 1.0
    dv = 4 * math.pi / 3 * (oh.edges[1:] ** 3 - oh.edges[:-1] ** 3)
    assert oh.g[k(bond)] == pytest.approx(4 / (8 * dv[k(bond)] / oh.volume), rel=1e-12)


def _chi2_pvalue(counts, expected, ordered=True):
    """Poisson chi^2 of histogram counts; ordered pairs come in twos (variance 2 E)."""
    keep = expected > 20
    z2 = (counts[keep] - expected[keep]) ** 2 / ((2.0 if ordered else 1.0) * expected[keep])
    return stats.chi2.sf(z2.sum(), keep.sum())


def test_rdf_of_uniform_ball_matches_closed_form():
    """
    Independent uniform points in a ball of radius R: the pair-distance CDF is
    F(s) = s^3 - 9 s^4 / 16 + s^6 / 32 (s = r / R), so with r_max = 2R the
    cluster-normalised g is 8 (1 - 3s/4 + s^3/16).
    """
    rng = np.random.default_rng(5)
    radius, n_atoms, n_frames = 3.0, 20, 2000
    d = rng.normal(size=(n_frames, n_atoms, 3))
    x = d / np.linalg.norm(d, axis=2)[..., None] * radius * rng.uniform(
        size=(n_frames, n_atoms, 1)) ** (1 / 3)
    res = analysis.radial_distribution(x, ["Ar"] * n_atoms, ("Ar", "Ar"), 2 * radius, 30)
    assert res.n_pairs == n_atoms * (n_atoms - 1)
    s = res.edges / radius
    cdf = s**3 - 9 * s**4 / 16 + s**6 / 32
    expected = np.diff(cdf) * n_frames * n_atoms * (n_atoms - 1)
    assert _chi2_pvalue(res.counts, expected) > 1e-3
    # g with the documented V = 4 pi r_max^3 / 3: shell averages of the closed
    # form, within 5 sigma (ordered-pair counts have variance 2 E).
    dv = 4 * math.pi / 3 * (res.edges[1:] ** 3 - res.edges[:-1] ** 3)
    g_bin = np.diff(cdf) * res.volume / dv
    keep = expected > 100
    assert np.all(np.abs(res.g - g_bin)[keep] < 5 * g_bin[keep] * np.sqrt(2 / expected[keep]))
    # ... = the point form 8 (1 - 3s/4 + s^3/16) at the shells' r^2-weighted
    # mean radius (exact for the linear part; the cubic one leaves <= 1.1e-3).
    r1, r2 = res.edges[:-1], res.edges[1:]
    s_bar = 0.75 * (r2**4 - r1**4) / (r2**3 - r1**3) / radius
    assert np.allclose(g_bin, 8 * (1 - 0.75 * s_bar + s_bar**3 / 16), rtol=0.0, atol=2e-3)
    assert res.coordination[-1] == pytest.approx(n_atoms - 1)


def test_rdf_of_periodic_ideal_gas_is_one():
    """
    Uniform points in a periodic box: g = 1 exactly with the N (N - 1)
    normalisation. With N^2 instead, g would be low by 2 %, which these
    statistics resolve (chi^2 p-value 3e-93, against 0.016 as it is).
    """
    rng = np.random.default_rng(6)
    box, n_atoms, n_frames = 10.0, 50, 2000
    x = rng.uniform(0.0, box, size=(n_frames, n_atoms, 3))
    res = analysis.radial_distribution(x, ["He"] * n_atoms, ("He", "He"), box / 2, 25, box=box)
    dv = 4 * math.pi / 3 * (res.edges[1:] ** 3 - res.edges[:-1] ** 3)
    expected = n_frames * n_atoms * (n_atoms - 1) * dv / box**3
    assert _chi2_pvalue(res.counts, expected) > 1e-3
    # g itself (counts / expected if the normalisation is right): chi^2 vs 1.
    keep = expected > 20
    z2 = (res.g[keep] - 1.0) ** 2 / (2.0 / expected[keep])
    assert stats.chi2.sf(z2.sum(), keep.sum()) > 1e-3
    # Running coordination = (number density of the others) x sphere volume.
    r = res.edges[1:]
    assert np.allclose(res.coordination[5:], (n_atoms - 1) / box**3 * 4 * math.pi / 3 * r[5:] ** 3,
                       rtol=0.03)
    with pytest.raises(ValueError, match="half the box"):
        analysis.radial_distribution(x, None, (range(5), range(5)), 0.6 * box, box=box)


def test_rdf_of_simple_cubic_lattice_with_minimum_image():
    a, m = 2.0, 4
    grid = np.stack(np.meshgrid(*[np.arange(m)] * 3, indexing="ij"), -1).reshape(-1, 3) * a
    rng = np.random.default_rng(8)
    frames = np.stack([grid + 0.3, (grid + rng.uniform(-50, 50, size=3)) % (m * a)])
    res = analysis.radial_distribution(frames, ["Ne"] * len(grid), ("Ne", "Ne"),
                                       1.9 * a, 38, box=m * a)
    k = lambda r: int(r / (1.9 * a / 38))
    # Shells of 6, 12 and 8 neighbours at a, a sqrt 2, a sqrt 3; any rigid shift.
    assert [res.coordination[k(f * a)] for f in (1.2, 1.5, 1.8)] == [6.0, 18.0, 26.0]
    assert res.counts.sum() == 2 * 64 * 26


def test_rdf_normalisation_counts_pairs_inside_r_max():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(50, 12, 3)) * 2.0
    r_max = 4.0
    res = analysis.radial_distribution(x, ["C"] * 12, ("C", "C"), r_max, 64)
    dv = 4 * math.pi / 3 * (res.edges[1:] ** 3 - res.edges[:-1] ** 3)
    d = np.linalg.norm(x[:, :, None] - x[:, None], axis=-1)
    inside = (d[:, ~np.eye(12, dtype=bool)] < r_max).mean()
    assert np.sum(res.g * dv) / res.volume == pytest.approx(inside, rel=1e-12)


# ── Geometry ──────────────────────────────────────────────────────────────────

def test_bond_lengths_and_angles_of_water():
    oh, half_angle = 0.9572, math.radians(104.52 / 2)          # angstrom, degrees
    x = np.array([[0.0, 0.0, 0.0],
                  [0.0, oh * math.sin(half_angle), oh * math.cos(half_angle)],
                  [0.0, -oh * math.sin(half_angle), oh * math.cos(half_angle)]]) / BOHR_TO_ANG
    traj = np.stack([x, 1.1 * x, x + 3.0])
    r = analysis.bond_lengths(traj, [(0, 1), (0, 2), (1, 2)])
    assert r.shape == (3, 3)
    hh = 2 * oh * math.sin(half_angle) / BOHR_TO_ANG
    assert np.allclose(r, [[oh / BOHR_TO_ANG] * 2 + [hh],
                           [1.1 * oh / BOHR_TO_ANG] * 2 + [1.1 * hh],
                           [oh / BOHR_TO_ANG] * 2 + [hh]], rtol=1e-13)
    ang = analysis.bond_angles(traj, (1, 0, 2))
    assert ang.shape == (3,) and np.allclose(ang, 104.52, rtol=1e-12)
    assert analysis.bond_angles(x, [(1, 0, 2)], degrees=False)[0] == pytest.approx(
        math.radians(104.52), rel=1e-12)
    assert analysis.bond_angles(x, (0, 1, 2)) == pytest.approx((180 - 104.52) / 2, rel=1e-12)


def test_circular_mean_and_unwrapping_of_periodic_series():
    # von Mises sample about 179 deg (kappa = 40, ~9 deg spread): in (-180, 180]
    # it straddles the cut. Reference: scipy.stats.circmean / circstd.
    rng = np.random.default_rng(12)
    a = np.degrees(rng.vonmises(np.radians(179.0), 40.0, 5000))
    a = np.mod(a + 180.0, 360.0) - 180.0
    assert a.min() < -170.0 and a.max() > 170.0
    m = analysis.circular_mean(a)
    ref = np.degrees(stats.circmean(np.radians(a), high=np.pi, low=-np.pi))
    assert abs(np.mod(m - ref + 180.0, 360.0) - 180.0) < 1e-9
    assert abs(abs(m) - 179.0) < 0.5
    u = analysis.unwrap_about(a, m)
    assert np.all(np.abs(u - m) <= 180.0)
    turns = (u - a) / 360.0                         # whole turns only
    np.testing.assert_allclose(turns, np.round(turns), rtol=0, atol=1e-12)
    assert set(np.round(turns)) == {0.0, 1.0} or set(np.round(turns)) == {0.0, -1.0}
    # for a concentrated sample the linear spread of the unwrapped series is the
    # circular spread sqrt(-2 ln R) (to O(sigma^3)); plain np.std(a) is ~180 deg
    circ_sd = np.degrees(stats.circstd(np.radians(a)))
    assert np.std(u) == pytest.approx(circ_sd, rel=0.01) and np.std(a) > 150.0
    # radians, nan handling, and a vanishing resultant
    assert analysis.circular_mean(np.radians(a), degrees=False) == pytest.approx(np.radians(m))
    assert analysis.circular_mean(np.array([np.nan, 170.0, -170.0])) == pytest.approx(180.0)
    assert np.isnan(analysis.circular_mean(np.array([0.0, 90.0, 180.0, -90.0])))
    assert np.isnan(analysis.circular_mean(np.array([np.nan])))
    assert analysis.unwrap_about(np.array([-179.0, 179.0]), 180.0).tolist() == [181.0, 179.0]


def test_dihedral_sign_follows_iupac_definition():
    """
    i = (1,0,0), j = 0, k = (0,0,1), l = k + (cos p, sin p, 0). Looking along
    j -> k (+z), turning bond i-j by +p about +z (counter-clockwise seen from
    above, i.e. from +z looking down) is clockwise for that viewer, so the
    IUPAC torsion is +p.
    """
    for p in (60.0, -60.0, 120.0, -150.0, 179.0, 10.0):
        q = math.radians(p)
        x = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1], [math.cos(q), math.sin(q), 1.0]])
        assert analysis.dihedral_angles(x, (0, 1, 2, 3)) == pytest.approx(p, abs=1e-10)
        mirror = x * np.array([1, -1, 1])
        assert analysis.dihedral_angles(mirror, (0, 1, 2, 3)) == pytest.approx(-p, abs=1e-10)
        assert analysis.dihedral_angles(x, (3, 2, 1, 0)) == pytest.approx(p, abs=1e-10)
    trans = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1], [-1, 0, 1.0]])
    assert analysis.dihedral_angles(trans, (0, 1, 2, 3)) == 180.0    # (-180, 180]
    assert analysis.dihedral_angles(trans * [1, -1, 1], (0, 1, 2, 3)) == 180.0


def test_dihedral_matches_projection_formula():
    """Independent route: signed angle between the projections of j->i and k->l."""
    rng = np.random.default_rng(10)
    x = rng.normal(size=(200, 6, 3)) * 1.5
    quads = [(0, 1, 2, 3), (5, 4, 1, 0), (2, 3, 4, 5)]
    phi = analysis.dihedral_angles(x, quads)
    for c, (i, j, k, l) in enumerate(quads):
        axis = x[:, k] - x[:, j]
        axis /= np.linalg.norm(axis, axis=1)[:, None]
        proj = lambda v: v - np.sum(v * axis, axis=1)[:, None] * axis
        a, b = proj(x[:, i] - x[:, j]), proj(x[:, l] - x[:, k])
        ref = np.degrees(np.arctan2(np.sum(np.cross(a, b) * axis, axis=1), np.sum(a * b, axis=1)))
        assert np.allclose(phi[:, c], ref, atol=1e-9)


def test_geometry_edge_cases():
    collinear = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0.0]])
    assert np.isnan(analysis.dihedral_angles(collinear, (0, 1, 2, 3)))
    assert analysis.bond_angles(collinear, (0, 1, 2)) == pytest.approx(180.0)
    # Minimum image across a periodic box.
    x = np.array([[0.2, 0.0, 0.0], [9.9, 0.0, 0.0]])
    assert analysis.bond_lengths(x, (0, 1), box=10.0) == pytest.approx(0.3)
    assert analysis.bond_lengths(x, (0, 1)) == pytest.approx(9.7)
    with pytest.raises(ValueError, match="indices"):
        analysis.bond_lengths(x, (0, 2))
    with pytest.raises(ValueError, match="tuples"):
        analysis.bond_lengths(x, [0, 1, 1, 0])
    # Regression: tuples of the wrong width were regrouped silently, e.g.
    # [(0, 1, 2), (3, 4, 5)] measured the "bonds" 0-1, 2-3 and 4-5.
    x6 = np.random.default_rng(16).normal(size=(6, 3))
    with pytest.raises(ValueError, match="tuples of 2"):
        analysis.bond_lengths(x6, [(0, 1, 2), (3, 4, 5)])
    with pytest.raises(ValueError, match="tuples of 3"):
        analysis.bond_angles(x6, [(0, 1, 2, 3), (2, 3, 4, 5), (1, 2, 3, 4)])
    with pytest.raises(ValueError, match="tuples of 4"):
        analysis.dihedral_angles(x6, [(0, 1, 2)] * 4)
    assert analysis.bond_lengths(x6, np.zeros((0, 2), dtype=int)).shape == (0,)


# ── Block averaging ───────────────────────────────────────────────────────────

def _ar1(phi, n, rng):
    """Stationary AR(1): x_t = phi x_{t-1} + e_t, e ~ N(0, 1)."""
    x0 = rng.normal() / math.sqrt(1.0 - phi**2)
    return lfilter([1.0], [1.0, -phi], rng.normal(size=n), zi=[phi * x0])[0]


def _ar1_sem(phi, n):
    """Exact SE of the mean: (g0 / n) [1 + 2 sum_k (1 - k/n) phi^k], g0 = 1 / (1 - phi^2)."""
    k = np.arange(1, n)
    return math.sqrt((1.0 + 2.0 * np.sum((1.0 - k / n) * phi**k)) / ((1.0 - phi**2) * n))


def test_block_average_recovers_ar1_standard_error():
    """
    phi = 0.9 (statistical inefficiency 19), 40 series of 2^16 samples.
    Measured: SE / exact = 0.978 +- 0.010 on average (expected finite-block
    bias -0.9 %), single series 0.85 - 1.10, g = 18.3; the naive SE is
    0.23 x exact.
    """
    rng = np.random.default_rng(11)
    phi, n = 0.9, 2**16
    exact = _ar1_sem(phi, n)
    results = [analysis.block_average(_ar1(phi, n, rng)) for _ in range(40)]
    ratio = np.array([r.sem for r in results]) / exact
    assert abs(ratio.mean() - 1.0) < 0.04
    assert np.all((ratio > 0.75) & (ratio < 1.25))
    assert np.all([r.converged for r in results])
    assert max(r.sem_naive for r in results) < 0.3 * exact
    g = np.mean([r.statistical_inefficiency for r in results])
    assert g == pytest.approx((1 + phi) / (1 - phi), rel=0.1)
    # The reported uncertainty of the SE matches the observed scatter
    # (measured 0.061 vs 0.060; the std of 40 samples is itself +-11 %).
    scatter = ratio.std() / (np.mean([r.sem_error for r in results]) / exact)
    assert 0.7 < scatter < 1.4


def test_blocking_levels_match_direct_reblocking():
    """
    Level j of the automatic analysis is the SE of the means of blocks of 2^j
    samples, ddof = 1, with the oldest samples that do not fill a block
    dropped (n = 1003 is odd at several levels), computed here by a direct
    reshape. A 1/m^2 instead of 1/(m (m - 1)) variance, or dropping the newest
    samples, changes these values.
    """
    rng = np.random.default_rng(15)
    x = _ar1(0.6, 1003, rng) + 5.0
    res = analysis.block_average(x)
    n = x.size
    assert list(res.block_sizes) == [2**j for j in range(9)]       # down to 3 blocks
    for b, se in zip(res.block_sizes, res.sems):
        nb = n // b
        means = x[n - nb * b :].reshape(nb, b).mean(axis=1)
        assert se == pytest.approx(means.std(ddof=1) / math.sqrt(nb), rel=1e-10)
    pick = list(res.block_sizes).index(res.block_size)
    assert res.sem == res.sems[pick] and res.n_blocks == n // res.block_size
    assert res.sem == pytest.approx(analysis.block_average(x, block_size=res.block_size).sem,
                                    rel=1e-10)


def test_block_average_of_uncorrelated_data():
    rng = np.random.default_rng(12)
    results = [analysis.block_average(rng.normal(2.0, 3.0, size=10000)) for _ in range(40)]
    ratio = np.array([r.sem for r in results]) / (3.0 / 100.0)
    # Measured mean 0.997 +- 0.007.
    assert abs(ratio.mean() - 1.0) < 0.03
    assert np.mean([r.mean for r in results]) == pytest.approx(2.0, abs=0.01)
    assert np.mean([r.std for r in results]) == pytest.approx(3.0, rel=0.01)


def test_block_average_with_fixed_blocks_and_short_series():
    x = np.arange(22, dtype=float) ** 1.5
    res = analysis.block_average(x, block_size=5)
    means = x[2:].reshape(4, 5).mean(axis=1)                   # oldest 2 samples dropped
    assert res.n_blocks == 4 and res.sem == pytest.approx(means.std(ddof=1) / 2.0, rel=1e-14)
    assert res.mean == pytest.approx(x.mean()) and res.std == pytest.approx(x.std(ddof=1))
    # A random walk never decorrelates within 1000 samples: flagged.
    walk = np.cumsum(np.random.default_rng(13).normal(size=1000))
    assert not analysis.block_average(walk).converged
    with pytest.raises(ValueError, match="fewer than 2"):
        analysis.block_average(x, block_size=20)


def test_column_statistics_of_md_energies(h4, tmp_path):
    h4.initialize_velocities(300.0, rng=14)
    log = tmp_path / "e.csv"
    res = run_md(h4, LangevinBAOAB(MorseBackend(h4.symbols), 0.25, 300.0,
                                   friction_per_fs=0.05, rng=14), 3000, energy_log=log)
    from_file = analysis.column_statistics(read_energy_log(log), skip=500)
    from_result = analysis.column_statistics(res, skip=500)
    assert set(from_file) == set(from_result) == set(analysis.statistics.DEFAULT_COLUMNS)
    temp = res.column("temperature_K")[500:]
    st = from_file["temperature_K"]
    assert st.mean == pytest.approx(temp.mean()) and st.n_samples == 2501
    assert from_result["temperature_K"].sem == pytest.approx(st.sem, rel=1e-12)
    # Correlated samples: blocking gives a larger error bar than the naive one.
    assert st.sem > st.sem_naive
