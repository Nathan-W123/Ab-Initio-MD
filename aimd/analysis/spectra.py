"""
Time-correlation functions and vibrational spectra from MD trajectories.

Inputs are series sampled every ``dt_fs`` femtoseconds (the frame spacing,
``write_every * timestep``) in engine units: velocities in bohr / au_time,
dipoles in e * bohr, masses in m_e. Spectra come on a wavenumber grid in cm^-1.

Frequency units
---------------
An angular frequency omega in rad / au_time is an energy hbar omega in hartree
(hbar = 1), so

    nu~ = omega / (2 pi c) = omega * E_h / (h c) = omega * 219474.63 cm^-1,

with c = 2.99792458e10 cm/s times the atomic time unit (aimd.units) in cm per
au_time. :func:`angular_frequency_to_wavenumber` is that conversion.

Correlation functions
---------------------
    C(k dt) = 1 / (n - k) * sum_{t=0}^{n-1-k} sum_c w_c x_c(t) x_c(t + k)

is the unbiased (every lag averaged over its n - k time origins) estimator,
summed over components c (atoms x Cartesian directions) with weights w_c.
It is evaluated with FFTs zero-padded to >= 2n - 1 points, which removes the
circular wrap-around (Wiener-Khinchin), in O(n log n).

  - VACF: C_v(t) = sum_i w_i <v_i(0) . v_i(t)>, w_i = m_i (mass-weighted;
    C_v(0) = 2 <E_kin> = N_dof kT at equilibrium) or 1.
  - Dipole derivative: C_mu'(t) = <mu'(0) . mu'(t)> with mu' from forward
    differences of the dipole series.

Spectra (Blackman-Tukey estimate)
---------------------------------
The correlation function up to lag M (``max_lag``) is multiplied by a lag
window w_k with w_0 = 1 (``hann``: 0.5 (1 + cos(pi k / M)), zero at lag M),
which suppresses the truncation ripple (Gibbs sidelobes) of a cut-off
correlation function and its noisy long-lag tail, and cosine transformed:

    S(nu~) = 2 c dt [w_0 C_0 + 2 sum_{k=1}^{M} w_k C_k cos(2 pi c nu~ k dt)].

S is a one-sided spectral density per cm^-1 normalised so that

    integral_0^{nu~_Nyquist} S(nu~) d nu~ = C(0),

which the trapezoidal rule on the returned (even-length FFT) grid satisfies to
round-off. Zero padding (``zero_pad``) only interpolates the grid. The
resolution is set by the correlation length: ``resolution = 1 / (c M dt)``
cm^-1, which is also the full width at half maximum of a Hann-windowed line.

  - VDOS: S of the VACF. Mass-weighted, S / kT integrates to the number of
    thermalised degrees of freedom; for a harmonic system each normal mode
    contributes a band of area <q'_k^2> (twice its mean kinetic energy).
  - IR: S of the dipole-derivative correlation, D(nu~). The finite-difference
    derivative multiplies the spectrum by sinc^2(omega dt / 2), which is
    divided out by default. With a temperature the result is the absorption
    cross-section per molecule (isotropic average; Gaussian/atomic units,
    McQuarrie, *Statistical Mechanics*, ch. 21),

        sigma(nu~) = pi beta / (3 c^2) * R(x) * D(nu~),   x = hbar omega / kT,

    reported as N_A sigma in km mol^-1 per cm^-1, so a band's area is its
    integrated intensity in km/mol. R is the quantum correction applied to
    the classical correlation function (Ramirez, Lopez-Ciudad, Kumar & Marx,
    J. Chem. Phys. 121, 3973 (2004)), relative to the classical result:

        "harmonic" (= classical)  R = 1      [Q = x / (1 - e^-x)]
        "standard"                R = 2 tanh(x/2) / x   [Q = 2 / (1 + e^-x)]
        "schofield"               R = 2 sinh(x/2) / x   [Q = e^(x/2)]

    The harmonic correction makes the classical band of a harmonic mode with
    dipole derivative dmu/dQ integrate to the exact quantum double-harmonic
    intensity pi N_A |dmu/dQ|^2 / (3 c^2) at any temperature (42.256 km/mol per
    (D/angstrom)^2 / amu).

References: Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed.,
Sec. 8.1; Thomas, Brehm, Fligg & Kirchner, Phys. Chem. Chem. Phys. 15, 6608
(2013); Blackman & Tukey, *The Measurement of Power Spectra* (1958).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aimd.units import AU_TIME_TO_FS, BOHR_TO_ANG, FS_TO_AU_TIME, KB_AU

SPEED_OF_LIGHT_SI = 299792458.0                    # m / s (exact)
AVOGADRO = 6.02214076e23                           # 1 / mol (exact)
# Speed of light in cm / au_time and in bohr / au_time (= 1 / alpha).
C_CM_PER_AU_TIME = SPEED_OF_LIGHT_SI * 1e2 * AU_TIME_TO_FS * 1e-15
C_AU = SPEED_OF_LIGHT_SI * AU_TIME_TO_FS * 1e-15 / (BOHR_TO_ANG * 1e-10)
BOHR_TO_KM = BOHR_TO_ANG * 1e-13

WINDOWS = ("hann", "blackman", "hamming", "none")
QUANTUM_CORRECTIONS = ("harmonic", "classical", "standard", "schofield")


# ── Units ─────────────────────────────────────────────────────────────────────

def angular_frequency_to_wavenumber(omega: np.ndarray | float) -> np.ndarray | float:
    """Angular frequency (rad / au_time) -> wavenumber (cm^-1)."""
    nu = np.asarray(omega, dtype=float) / (2.0 * math.pi * C_CM_PER_AU_TIME)
    return float(nu) if nu.ndim == 0 else nu


def wavenumber_to_angular_frequency(nu_cm: np.ndarray | float) -> np.ndarray | float:
    """Wavenumber (cm^-1) -> angular frequency (rad / au_time)."""
    omega = np.asarray(nu_cm, dtype=float) * (2.0 * math.pi * C_CM_PER_AU_TIME)
    return float(omega) if omega.ndim == 0 else omega


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fast_length(n: int, even: bool = False) -> int:
    """Smallest 2^a 3^b 5^c >= n (with a >= 1 if ``even``): a fast FFT size."""
    n = max(int(n), 2)
    best = 1 << (n - 1).bit_length()
    p5 = 1
    while p5 < best:
        p35 = p5
        while p35 < best:
            q = p35
            while q < n or (even and q % 2):
                q *= 2
            best = min(best, q)
            p35 *= 3
        p5 *= 5
    return best


def _trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def lag_window(name: str, max_lag: int) -> np.ndarray:
    """Half lag window w_0..w_M (w_0 = 1) for lags 0..max_lag."""
    u = np.arange(max_lag + 1) / max(max_lag, 1)
    if name == "hann":
        w = 0.5 * (1.0 + np.cos(np.pi * u))
    elif name == "blackman":
        w = 0.42 + 0.5 * np.cos(np.pi * u) + 0.08 * np.cos(2.0 * np.pi * u)
    elif name == "hamming":
        w = 0.54 + 0.46 * np.cos(np.pi * u)
    elif name == "none":
        w = np.ones_like(u)
    else:
        raise ValueError(f"unknown window {name!r}; use one of {WINDOWS}")
    return w / w[0]                                   # exactly 1 at lag 0


def _check_dt(dt_fs: float) -> float:
    dt = float(dt_fs)
    if not (math.isfinite(dt) and dt > 0.0):
        raise ValueError(f"dt_fs must be a positive frame spacing, got {dt_fs!r}")
    return dt


def _check_temperature(temperature_k: float | None) -> float:
    t = math.nan if temperature_k is None else float(temperature_k)
    if not (math.isfinite(t) and t > 0.0):
        raise ValueError(f"temperature_k must be positive and finite, got {temperature_k!r}")
    return t


def _default_lag(n: int, max_lag: int | None) -> int:
    m = n // 2 if max_lag is None else int(max_lag)
    if not 1 <= m < n:
        raise ValueError(f"max_lag must be in [1, {n - 1}] for {n} samples, got {m}")
    return m


# ── Correlation functions ─────────────────────────────────────────────────────

def autocorrelation(
    x: np.ndarray,
    max_lag: int | None = None,
    weights: np.ndarray | None = None,
    subtract_mean: bool = False,
) -> np.ndarray:
    """
    Unbiased autocorrelation of a series sampled along axis 0.

    ``x`` has shape (n,) or (n, ...); the lag-k value is
    1/(n-k) sum_t sum_c w_c x_c(t) x_c(t+k), summed over all trailing
    components c with ``weights`` (broadcast to ``x.shape[1:]``, default 1).
    Returns lags 0..max_lag (default n - 1), shape (max_lag + 1,).
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n < 2:
        raise ValueError("need at least two samples")
    m = n - 1 if max_lag is None else int(max_lag)
    if not 0 <= m < n:
        raise ValueError(f"max_lag must be in [0, {n - 1}], got {m}")
    cols = x.reshape(n, -1)
    if subtract_mean:
        cols = cols - cols.mean(axis=0)
    w = (np.ones(cols.shape[1]) if weights is None
         else np.broadcast_to(np.asarray(weights, dtype=float), x.shape[1:]).reshape(-1))
    length = _fast_length(2 * n - 1)               # no circular overlap for lags < n
    chunk = max(1, int(2**22 // length))           # bound the complex work array
    power = np.zeros(length // 2 + 1)
    for c0 in range(0, cols.shape[1], chunk):
        f = np.fft.rfft(cols[:, c0 : c0 + chunk], n=length, axis=0)
        power += (f.real**2 + f.imag**2) @ w[c0 : c0 + chunk]
    acf = np.fft.irfft(power, n=length)[: m + 1]
    return acf / (n - np.arange(m + 1))


@dataclass
class VACF:
    """Velocity autocorrelation function sum_i w_i <v_i(0) . v_i(t)>."""
    time_fs: np.ndarray        # lags, (max_lag + 1,)
    acf: np.ndarray            # hartree (m_e bohr^2 / au_time^2) if mass-weighted,
                               # else bohr^2 / au_time^2
    mass_weighted: bool

    @property
    def normalized(self) -> np.ndarray:
        """C(t) / C(0)."""
        return self.acf / self.acf[0]


def _velocity_series(
    velocities: np.ndarray, masses: np.ndarray | None, atoms: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    v = np.asarray(velocities, dtype=float)
    if v.ndim != 3 or v.shape[2] != 3:
        raise ValueError("velocities must have shape (n_frames, N, 3)")
    if not np.all(np.isfinite(v)):
        raise ValueError("velocities contain nan/inf")
    idx = np.arange(v.shape[1]) if atoms is None else np.asarray(atoms, dtype=int).reshape(-1)
    w = None
    if masses is not None:
        m = np.asarray(masses, dtype=float).reshape(-1)
        if m.shape[0] != v.shape[1]:
            raise ValueError(f"{m.shape[0]} masses for {v.shape[1]} atoms")
        w = np.repeat(m[idx][:, None], 3, axis=1)
    return v[:, idx, :], w


def velocity_autocorrelation(
    velocities: np.ndarray,
    dt_fs: float,
    masses: np.ndarray | None = None,
    atoms: np.ndarray | list[int] | None = None,
    max_lag: int | None = None,
) -> VACF:
    """
    VACF of ``velocities`` (n_frames, N, 3), bohr / au_time, frames ``dt_fs``
    apart. With ``masses`` (N,) it is mass-weighted; ``atoms`` selects a
    subset of atoms (partial VACF). Lags 0..max_lag (default n - 1).
    """
    dt_fs = _check_dt(dt_fs)
    v, w = _velocity_series(velocities, masses, atoms)
    acf = autocorrelation(v, max_lag, weights=w)
    return VACF(time_fs=np.arange(acf.size) * dt_fs, acf=acf,
                mass_weighted=masses is not None)


# ── Spectra ───────────────────────────────────────────────────────────────────

@dataclass
class Spectrum:
    """One-sided spectral density on a wavenumber grid."""
    wavenumber: np.ndarray     # cm^-1, (n_freq,), 0 .. Nyquist
    intensity: np.ndarray      # per cm^-1, see ``unit``
    resolution: float          # cm^-1, 1 / (c * max_lag * dt)
    max_lag: int
    window: str
    unit: str

    def band_area(self, lo: float = 0.0, hi: float = math.inf) -> float:
        """Integral of the intensity over lo <= nu~ <= hi (trapezoidal rule)."""
        sel = (self.wavenumber >= lo) & (self.wavenumber <= hi)
        return _trapezoid(self.intensity[sel], self.wavenumber[sel])

    def peak(self, lo: float = 0.0, hi: float = math.inf) -> float:
        """Position (cm^-1) of the maximum within [lo, hi], parabolically interpolated."""
        sel = np.flatnonzero((self.wavenumber >= lo) & (self.wavenumber <= hi))
        if sel.size == 0:
            raise ValueError("no grid points in the requested range")
        k = sel[np.argmax(self.intensity[sel])]
        if 0 < k < self.wavenumber.size - 1:
            y0, y1, y2 = self.intensity[k - 1 : k + 2]
            den = y0 - 2.0 * y1 + y2
            if den < 0.0:
                step = self.wavenumber[1] - self.wavenumber[0]
                return float(self.wavenumber[k] + 0.5 * (y0 - y2) / den * step)
        return float(self.wavenumber[k])


def correlation_spectrum(
    acf: np.ndarray,
    dt_fs: float,
    window: str = "hann",
    zero_pad: int = 4,
    unit: str = "",
) -> Spectrum:
    """
    One-sided spectral density (per cm^-1) of a correlation function given at
    lags 0..M (``acf``), spaced ``dt_fs``: windowed cosine transform with
    integral over nu~ equal to acf[0] (see the module docstring).
    """
    dt_fs = _check_dt(dt_fs)
    c = np.asarray(acf, dtype=float).reshape(-1)
    m = c.size - 1
    if m < 1:
        raise ValueError("need at least two lags")
    if zero_pad < 1:
        raise ValueError("zero_pad must be >= 1")
    cw = c * lag_window(window, m)
    # Even length >= 2M + 2 holds the symmetric sequence C_|k| without overlap,
    # and gives a Nyquist bin, so the one-sided trapezoid sum is exact.
    length = _fast_length(int(zero_pad) * (2 * m + 2), even=True)
    seq = np.zeros(length)
    seq[: m + 1] = cw
    seq[length - m :] = cw[:0:-1]
    cosine_sum = np.fft.rfft(seq).real               # w0 C0 + 2 sum_k wk Ck cos(.)
    dt_au = dt_fs * FS_TO_AU_TIME
    nu = np.arange(length // 2 + 1) / (length * dt_au * C_CM_PER_AU_TIME)
    return Spectrum(
        wavenumber=nu,
        intensity=2.0 * C_CM_PER_AU_TIME * dt_au * cosine_sum,
        resolution=1.0 / (C_CM_PER_AU_TIME * m * dt_au),
        max_lag=m,
        window=window,
        unit=unit,
    )


def vibrational_dos(
    velocities: np.ndarray,
    dt_fs: float,
    masses: np.ndarray | None = None,
    atoms: np.ndarray | list[int] | None = None,
    max_lag: int | None = None,
    window: str = "hann",
    zero_pad: int = 4,
) -> Spectrum:
    """
    Vibrational density of states: the power spectrum of the velocities, i.e.
    the windowed Fourier transform of the VACF (see velocity_autocorrelation
    for ``masses`` / ``atoms``). ``max_lag`` (default n_frames // 2) sets the
    resolution 1 / (c max_lag dt). The band areas sum to C_v(0): with masses,
    2 <E_kin> in hartree (divide by kT for a mode count).
    """
    v, w = _velocity_series(velocities, masses, atoms)
    m = _default_lag(v.shape[0], max_lag)
    acf = autocorrelation(v, m, weights=w)
    unit = ("hartree per cm^-1" if masses is not None
            else "bohr^2 au_time^-2 per cm^-1")
    return correlation_spectrum(acf, dt_fs, window, zero_pad, unit)


def quantum_correction_factor(
    wavenumber: np.ndarray, temperature_k: float, kind: str | None = "harmonic"
) -> np.ndarray:
    """
    Factor R(x), x = hbar omega / kT, that multiplies the classical IR line
    shape for quantum correction ``kind`` (module docstring); R -> 1 as x -> 0.

    The Schofield factor grows as e^(x/2) / x: at 300 K it is ~1.5e3 at
    4000 cm^-1 and ~1e32 at the Nyquist wavenumber of 0.5 fs frames, so any
    noise or window leakage far above the physical bands (dipole noise of
    1e-9 e bohr already gives an intensity ~1e25 times the peak at Nyquist)
    dominates the corrected spectrum there. Restrict ``peak`` / ``band_area``
    to the physical range when using it.
    """
    if kind in (None, "harmonic", "classical"):
        return np.ones_like(np.asarray(wavenumber, dtype=float))
    if kind not in QUANTUM_CORRECTIONS:
        raise ValueError(f"unknown quantum correction {kind!r}; use one of {QUANTUM_CORRECTIONS}")
    temperature_k = _check_temperature(temperature_k)
    omega = wavenumber_to_angular_frequency(np.asarray(wavenumber, dtype=float))
    # Capped only so that sinh stays finite (R <= ~1e301).
    x = np.minimum(omega / (KB_AU * temperature_k), 1400.0)
    small = x < 1e-8
    xs = np.where(small, 1.0, x)
    if kind == "standard":
        r = 2.0 * np.tanh(0.5 * xs) / xs
    else:                                           # schofield
        r = 2.0 * np.sinh(0.5 * xs) / xs
    return np.where(small, 1.0, r)


def ir_spectrum(
    dipoles: np.ndarray,
    dt_fs: float,
    temperature_k: float | None = None,
    quantum_correction: str | None = "harmonic",
    max_lag: int | None = None,
    window: str = "hann",
    zero_pad: int = 4,
    finite_difference_correction: bool = True,
) -> Spectrum:
    """
    Infrared spectrum from the dipole-derivative autocorrelation.

    ``dipoles`` (n_frames, 3) in e * bohr, frames ``dt_fs`` apart (e.g.
    ``read_dipole_log(path).dipole``). mu' is taken by forward differences
    (mu_{t+1} - mu_t) / dt, whose sinc^2(omega dt / 2) attenuation is divided
    out unless ``finite_difference_correction=False``.

    Without ``temperature_k`` the result is D(nu~), the spectral density of
    <mu'(0) . mu'(t)> in (e bohr / au_time)^2 per cm^-1 (integral = <|mu'|^2>);
    only the classical / harmonic correction is then possible. With it, the
    intensity is the molar absorption cross-section in km mol^-1 per cm^-1
    (band areas in km/mol) with ``quantum_correction`` applied.

    The dipoles of run_md are taken about the origin, so for a charged
    system mu' contains Q v_com: remove the centre-of-mass motion first (as
    MolecularSystem.initialize_velocities does) or the translation shows up
    as a band at nu~ = 0.
    """
    dt_fs = _check_dt(dt_fs)
    mu = np.asarray(dipoles, dtype=float)
    mu = mu.reshape(mu.shape[0], -1)
    if not np.all(np.isfinite(mu)):
        raise ValueError("dipoles contain nan/inf (did the backend report a dipole?)")
    if mu.shape[0] < 3:
        raise ValueError("need at least three dipole samples")
    if quantum_correction is not None and quantum_correction not in QUANTUM_CORRECTIONS:
        raise ValueError(f"unknown quantum correction {quantum_correction!r}; "
                         f"use one of {QUANTUM_CORRECTIONS}")
    if temperature_k is None and quantum_correction not in (None, "harmonic", "classical"):
        raise ValueError(f"the {quantum_correction!r} correction needs temperature_k")
    if temperature_k is not None:
        temperature_k = _check_temperature(temperature_k)
    dt_au = dt_fs * FS_TO_AU_TIME
    mu_dot = np.diff(mu, axis=0) / dt_au
    m = _default_lag(mu_dot.shape[0], max_lag)
    spec = correlation_spectrum(autocorrelation(mu_dot, m), dt_fs, window, zero_pad)
    intensity = spec.intensity
    if finite_difference_correction:
        half_phase = 0.5 * wavenumber_to_angular_frequency(spec.wavenumber) * dt_au
        intensity = intensity / np.sinc(half_phase / np.pi) ** 2   # <= pi^2/4 at Nyquist
    if temperature_k is None:
        spec.intensity = intensity
        spec.unit = "(e bohr / au_time)^2 per cm^-1"
        return spec
    beta = 1.0 / (KB_AU * temperature_k)
    prefactor = math.pi * beta / (3.0 * C_AU**2) * AVOGADRO * BOHR_TO_KM
    spec.intensity = prefactor * quantum_correction_factor(
        spec.wavenumber, temperature_k, quantum_correction) * intensity
    spec.unit = "km mol^-1 per cm^-1"
    return spec
