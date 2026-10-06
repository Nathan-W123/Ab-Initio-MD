from pathlib import Path

import pytest

from aimd.system import MolecularSystem

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
# Test-owned inputs; several tests pin numbers measured on these geometries.
DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture
def h4() -> MolecularSystem:
    return MolecularSystem.from_xyz(EXAMPLES / "h4_cluster.xyz")


@pytest.fixture
def water() -> MolecularSystem:
    return MolecularSystem.from_xyz(DATA / "water_experimental.xyz")
