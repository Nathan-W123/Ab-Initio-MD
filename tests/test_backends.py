import importlib.util

import numpy as np
import pytest

from aimd.backends import get_backend, list_backends
from aimd.testing import max_gradient_error


def test_registry_lists_builtin_backends():
    assert {"morse", "psi4"} <= set(list_backends())


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Available backends"):
        get_backend("nope")


def test_morse_diatomic_minimum_and_dissociation():
    be = get_backend("morse")(["H", "H"])
    at_min = be.compute(np.array([[0, 0, 0], [0, 0, be.r_eq]]))
    assert at_min.energy == pytest.approx(-be.depth)
    assert np.allclose(at_min.gradient, 0.0, atol=1e-12)
    far = be.compute(np.array([[0, 0, 0], [0, 0, 50.0]]))
    assert far.energy == pytest.approx(0.0, abs=1e-12)


def test_morse_gradient_matches_finite_difference(h4):
    be = get_backend("morse")(h4.symbols)
    x = h4.positions + np.random.default_rng(0).normal(scale=0.1, size=h4.positions.shape)
    assert max_gradient_error(be, x, step=1e-5) < 1e-8


def test_morse_forces_sum_to_zero(h4):
    be = get_backend("morse")(h4.symbols)
    x = h4.positions + 0.2
    x[0] += 0.3
    assert np.allclose(be.compute(x).forces.sum(axis=0), 0.0, atol=1e-12)


@pytest.mark.skipif(importlib.util.find_spec("psi4") is None, reason="psi4 not installed")
def test_psi4_hf_gradient_matches_finite_difference(water):
    be = get_backend("psi4")(water.symbols, method="hf", basis="sto-3g")
    try:
        assert max_gradient_error(be, water.positions, step=1e-3) < 1e-5
    finally:
        be.close()
