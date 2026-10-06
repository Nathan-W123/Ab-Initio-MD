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


# ── Rigid-body motion and degree-of-freedom accounting ────────────────────────

from aimd.units import KB_AU  # noqa: E402

CO2 = "3\nlinear CO2\nO 0 0 -1.16\nC 0 0 0\nO 0 0 1.16\n"
H2 = "2\nH2\nH 0 0 0\nH 0 0 0.74\n"
HE = "1\nHe\nHe 0 0 0\n"


def _random_velocities(system, seed):
    rng = np.random.default_rng(seed)
    system.velocities = rng.normal(size=system.positions.shape) * 1e-3
    return system


@pytest.mark.parametrize(
    "xyz, n3, after_com, after_rot, n_rot",
    [
        (None, 9, 6, 3, 3),        # water (fixture below), nonlinear
        (CO2, 9, 6, 3, 3),         # linear triatomic: L = 0 is 3 constraints once it bends
        (H2, 6, 3, 1, 2),          # diatomic: one vibration
        (HE, 3, 1, 1, 0),          # single atom: 0 DOF, clamped to 1
    ],
)
def test_dof_accounting(water, xyz, n3, after_com, after_rot, n_rot):
    s = water if xyz is None else MolecularSystem.from_xyz_string(xyz)
    _random_velocities(s, 1)
    assert s.n_dof == n3
    s.remove_com_motion()
    assert s.n_dof == after_com
    s.remove_angular_momentum()
    assert s.rotation_removed and s.rotational_dof == n_rot
    assert s.n_dof == after_rot


def test_inertia_tensor_of_diatomic_matches_closed_form():
    s = MolecularSystem(["H", "O"], [[0.3, -0.2, 0.1], [2.1, -0.2, 0.1]])
    m1, m2 = s.masses
    mu_d2 = m1 * m2 / (m1 + m2) * 1.8**2
    assert np.allclose(s.inertia_tensor(), np.diag([0.0, mu_d2, mu_d2]),
                       rtol=0.0, atol=1e-12 * mu_d2)


def test_rigid_rotation_is_removed_exactly_and_translation_kept(water):
    """v = w x (r - R) + V + internal: removal leaves V + internal, L = 0."""
    rng = np.random.default_rng(2)
    r = water.positions - water.center_of_mass()
    omega = rng.normal(size=3) * 1e-3
    vcom = rng.normal(size=3) * 1e-4
    water.velocities = np.cross(omega, r) + vcom
    start = water.positions.copy()
    water.remove_angular_momentum()
    assert np.allclose(water.velocities, vcom, rtol=0.0, atol=1e-15)
    assert np.array_equal(water.positions, start)


def test_angular_momentum_removal_kinetic_energy_and_momentum(water):
    _random_velocities(water, 3)
    p0, k0 = water.momentum(), water.kinetic_energy()
    ang = water.angular_momentum()
    # Removing the rigid rotation lowers K by exactly L.I^-1.L / 2, because the
    # remaining velocities are mass-orthogonal to every rigid rotation.
    expected_drop = 0.5 * ang @ np.linalg.solve(water.inertia_tensor(), ang)
    start = water.positions.copy()
    water.remove_angular_momentum()
    assert np.abs(water.angular_momentum()).max() < 1e-14 * np.abs(ang).max()
    assert np.allclose(water.momentum(), p0, rtol=0.0, atol=1e-15)
    assert k0 - water.kinetic_energy() == pytest.approx(expected_drop, rel=1e-10)
    assert np.array_equal(water.positions, start)


def test_linear_molecule_angular_momentum_removal_is_finite():
    s = _random_velocities(MolecularSystem.from_xyz_string(CO2), 4)
    vz = s.velocities[:, 2].copy()
    l0 = np.abs(s.angular_momentum()).max()
    s.remove_angular_momentum()
    assert np.all(np.isfinite(s.velocities))
    assert np.abs(s.angular_momentum()).max() < 1e-14 * l0
    # The singular axis is untouched: w is perpendicular to z, so w x r has no
    # z component for atoms on the z axis.
    assert np.array_equal(s.velocities[:, 2], vz)


def test_nearly_linear_molecule_uses_pseudo_inverse():
    """A 1e-6 bohr kink must not produce a huge spurious rotation."""
    s = _random_velocities(MolecularSystem.from_xyz_string(CO2), 5)
    s.positions[1, 0] += 1e-6
    assert s.is_linear()
    v0 = s.velocities.copy()
    s.remove_angular_momentum()
    assert np.abs(s.velocities - v0).max() < 10 * np.abs(v0).max()
    assert s.n_rotations() == 2 and s.rotational_dof == 3


def test_single_atom_rotation_removal_is_a_no_op():
    s = _random_velocities(MolecularSystem.from_xyz_string(HE), 6)
    v0 = s.velocities.copy()
    s.remove_angular_momentum()
    assert np.array_equal(s.velocities, v0)


def test_initialize_velocities_can_remove_rotation(water):
    water.initialize_velocities(300.0, rng=7, remove_rotation=True)
    assert water.n_dof == 3
    assert water.temperature() == pytest.approx(300.0)
    assert np.abs(water.momentum()).max() < 1e-12
    assert np.abs(water.angular_momentum()).max() < 1e-12
    # Re-drawing without the option restores the default accounting.
    water.initialize_velocities(300.0, rng=7)
    assert not water.rotation_removed and water.n_dof == 6
    assert np.abs(water.angular_momentum()).max() > 1e-3


def test_rotational_dof_is_frozen_when_a_linear_molecule_bends():
    # Regression: a linear polyatomic used to get N_dof = 3N - 5, but velocity
    # Verlet keeps all three components of L at zero and the molecule bends at
    # once, so only 3N - 6 velocity directions are accessible (dynamics check
    # in test_thermostats.py). The count is fixed at removal time.
    s = MolecularSystem.from_xyz_string(CO2)
    assert s.is_linear() and s.n_rotations() == 2
    s.initialize_velocities(300.0, rng=8, remove_rotation=True)
    assert s.n_dof == 3 and s.temperature() == pytest.approx(300.0)
    s.positions[1, 0] += 0.1                      # bend: now nonlinear
    assert not s.is_linear() and s.n_dof == 3
    c = s.copy()
    assert (c.com_removed, c.rotation_removed, c.rotational_dof) == (True, True, 3)
    # a diatomic cannot bend: 2 constraints
    h2 = MolecularSystem.from_xyz_string(H2)
    h2.initialize_velocities(300.0, rng=8, remove_rotation=True)
    assert h2.rotational_dof == 2 and h2.n_dof == 1
    # an explicitly stored count (e.g. from an old checkpoint) is honoured
    assert MolecularSystem(s.symbols, s.positions, com_removed=True, rotation_removed=True,
                           rotational_dof=2).n_dof == 4


def test_zero_temperature_with_rotation_removal(water):
    water.initialize_velocities(0.0, remove_rotation=True)
    assert water.kinetic_energy() == 0.0 and water.n_dof == 3
    assert water.temperature() == 0.0


def test_temperature_uses_reduced_dof(water):
    _random_velocities(water, 9)
    water.remove_com_motion()
    water.remove_angular_momentum()
    assert water.temperature() == pytest.approx(
        2.0 * water.kinetic_energy() / (3 * KB_AU)
    )
