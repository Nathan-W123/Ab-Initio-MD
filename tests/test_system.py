import numpy as np
import pytest

from aimd.system import MolecularSystem
from aimd.units import AMU_TO_AU, ANG_TO_BOHR


def test_xyz_parsing_converts_to_bohr(water):
    assert water.symbols == ["O", "H", "H"]
    assert water.positions[1, 1] == pytest.approx(0.757 * ANG_TO_BOHR)
    assert water.masses[0] == pytest.approx(15.999 * AMU_TO_AU)


def test_xyz_rejects_wrong_atom_count():
    with pytest.raises(ValueError):
        MolecularSystem.from_xyz_string("3\n\nH 0 0 0\nH 0 0 1\n")


def test_unknown_element_rejected():
    with pytest.raises(ValueError):
        MolecularSystem(["Xx"], np.zeros((1, 3)))


def test_maxwell_boltzmann_hits_target_and_zero_momentum(water):
    water.initialize_velocities(300.0, rng=1)
    assert water.temperature() == pytest.approx(300.0)
    assert np.allclose(water.masses @ water.velocities, 0.0, atol=1e-12)
    assert water.n_dof == 6


def test_zero_temperature_gives_zero_velocities(water):
    water.initialize_velocities(0.0)
    assert water.kinetic_energy() == 0.0
