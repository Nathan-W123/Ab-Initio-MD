"""
Boys function F_n(x) = int_0^1 t^(2n) exp(-x t^2) dt.

Every Coulomb-type Gaussian integral (nuclear attraction, electron repulsion
and their nuclear derivatives) reduces to F_0..F_nmax at one argument
x = alpha |P - Q|^2 per primitive pair/quartet, so this sits in the innermost
loop and must be both fast and accurate to ~1e-15 relative.

Method (Helgaker, Jorgensen & Olsen, *Molecular Electronic-Structure Theory*,
Wiley 2000, sec. 9.8.2):

  - x < X_SWITCH: tabulate F_n on a grid x_k = k * STEP, then take a Taylor
    expansion about the nearest grid point,
        F_n(x) = sum_k F_{n+k}(x_k) (x_k - x)^k / k!       (dF_n/dx = -F_{n+1}),
    for n = nmax only, and fill n < nmax by the (stable) downward recursion
        F_n(x) = (2x F_{n+1}(x) + exp(-x)) / (2n + 1).
    With STEP = 0.1 (|x - x_k| <= 0.05) and TAYLOR_TERMS = 10 the truncation
    error is < 0.05^10 / 10! ~ 3e-20 relative.
  - x >= X_SWITCH: F_0 = sqrt(pi/x) erf(sqrt(x)) / 2 (erf = 1 to double
    precision) and the upward recursion
        F_{n+1}(x) = ((2n + 1) F_n(x) - exp(-x)) / (2x),
    which is stable while 2n + 1 < 2x (guaranteed: nmax <= NMAX < X_SWITCH).
    For x >= X_SWITCH the downward recursion is stable as well, so callers
    can compute the highest order once (``boys_top``) and recurse downwards
    -- as long as F_nmax(x) ~ (2n-1)!! sqrt(pi) / (2^(n+1) x^(n+1/2)) is a
    normal double. It becomes subnormal (and then zero, which the downward
    recursion would propagate to every lower order) for x > 3e10 at n = 32,
    x > 2e18 at n = 17 and x > 8e32 at n = 9. The integral kernels need
    n <= 4 l_max + 1 (9 for d shells, 17 for g shells) with x = alpha R^2,
    alpha <= 3e5 bohr^-2 for the bases here, so this needs R > 1e6 bohr even
    for g shells; ``boys_into`` recurses upward from F_0 for x >= X_SWITCH
    and is accurate for any x.

The grid table is built once at import from the convergent series
    F_n(x) = exp(-x) sum_k (2x)^k / ((2n+1)(2n+3)...(2n+2k+1))
(all terms positive, so no cancellation) evaluated for the highest order and
followed by downward recursion.

Everything is dimensionless; ``x`` is in bohr^-2 * bohr^2.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

NMAX = 32                 # highest order the table supports
TAYLOR_TERMS = 10         # Taylor terms about the nearest grid point
STEP = 0.1                # grid spacing in x
X_SWITCH = 40.0           # switch to the asymptotic/upward branch here
_NGRID = int(round(X_SWITCH / STEP)) + 2
_NROWS = NMAX + TAYLOR_TERMS + 1


def _build_table() -> np.ndarray:
    """TABLE[k, n] = F_n(x_k) on the grid x_k = k * STEP, n = 0.._NROWS-1."""
    x = np.arange(_NGRID) * STEP
    top = _NROWS - 1
    # Series for the highest order: terms grow until k ~ x - n, then decay
    # geometrically; 400 terms is far past convergence for x <= 40.2.
    term = np.full_like(x, 1.0 / (2 * top + 1))
    total = term.copy()
    for k in range(1, 400):
        term = term * (2.0 * x) / (2 * top + 2 * k + 1)
        total += term
    ex = np.exp(-x)
    table = np.empty((_NROWS, _NGRID))
    table[top] = ex * total
    for n in range(top - 1, -1, -1):
        table[n] = (2.0 * x * table[n + 1] + ex) / (2 * n + 1)
    # grid-major layout: the Taylor sum reads F_nmax..F_nmax+9 at one grid
    # point, which are then contiguous in memory
    return np.ascontiguousarray(table.T)


TABLE: np.ndarray = _build_table()
TABLE.flags.writeable = False

# reciprocals, so that no division sits in a dependency chain
_INV_INT = np.array([1.0 / (m + 1) for m in range(TAYLOR_TERMS)])
INV_ODD = np.array([1.0 / (2 * n + 1) for n in range(NMAX + 1)])
_INV_STEP = 1.0 / STEP


@njit(cache=True)
def boys_top(n: int, x: float) -> float:
    """
    F_n(x) for a single order n <= NMAX.

    Scalar arguments only (the table is a module constant that numba freezes
    into the compiled code): numba increments/decrements the reference count
    of every array argument of a non-inlined call, which would cost more than
    the Boys function itself in the innermost integral loops.
    """
    if x < X_SWITCH:
        k = int(x * _INV_STEP + 0.5)
        dx = k * STEP - x
        # Taylor series sum_m F_{n+m}(x_k) dx^m / m!
        acc = 0.0
        fac = 1.0
        for m in range(TAYLOR_TERMS):
            acc += TABLE[k, n + m] * fac
            fac *= dx * _INV_INT[m]
        return acc
    # erf(sqrt(x)) rounds to 1.0 in double precision for x >= 40
    f = 0.5 * math.sqrt(math.pi / x)
    if n > 0:
        ex = math.exp(-x)
        inv2x = 0.5 / x
        for m in range(n):
            f = ((2 * m + 1) * f - ex) * inv2x
    return f


@njit(cache=True)
def boys_into(nmax: int, x: float, out: np.ndarray) -> None:
    """Write F_0(x)..F_nmax(x) into ``out[0:nmax+1]`` (nmax <= NMAX)."""
    if x >= X_SWITCH:
        # upward from F_0 (stable here, see the module docstring): recursing
        # down from F_nmax would turn an underflowed F_nmax (x > 3e10 for
        # nmax = 32) into F_n = 0 for every n
        f = 0.5 * math.sqrt(math.pi / x)
        out[0] = f
        ex = math.exp(-x)
        inv2x = 0.5 / x
        for n in range(nmax):
            f = ((2 * n + 1) * f - ex) * inv2x
            out[n + 1] = f
        return
    f = boys_top(nmax, x)
    out[nmax] = f
    if nmax > 0:
        # downward recursion; stable for all x (relative errors do not grow)
        ex = math.exp(-x)
        x2 = 2.0 * x
        for n in range(nmax - 1, -1, -1):
            f = (x2 * f + ex) * INV_ODD[n]
            out[n] = f


def boys(nmax: int, x: float) -> np.ndarray:
    """F_0(x)..F_nmax(x) as a new array (convenience wrapper for tests/tools)."""
    if not 0 <= nmax <= NMAX:
        raise ValueError(f"Boys order must be in [0, {NMAX}], got {nmax}")
    if x < 0:
        raise ValueError("Boys function argument must be >= 0")
    out = np.empty(nmax + 1)
    boys_into(nmax, float(x), out)
    return out
