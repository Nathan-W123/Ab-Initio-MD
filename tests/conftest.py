from pathlib import Path

import pytest

from aimd.system import MolecularSystem

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture
def h4() -> MolecularSystem:
    return MolecularSystem.from_xyz(EXAMPLES / "h4_cluster.xyz")


@pytest.fixture
def water() -> MolecularSystem:
    return MolecularSystem.from_xyz(EXAMPLES / "water.xyz")
