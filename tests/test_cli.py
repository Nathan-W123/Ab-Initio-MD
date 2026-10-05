from aimd.cli import main
from conftest import EXAMPLES


def test_cli_lists_backends(capsys):
    assert main(["backends"]) == 0
    assert "morse" in capsys.readouterr().out


def test_cli_runs_morse_nve(tmp_path, capsys):
    rc = main([
        "run", str(EXAMPLES / "h4_cluster.xyz"), "--backend", "morse",
        "--steps", "20", "--dt", "0.2", "--seed", "1",
        "--trajectory", str(tmp_path / "t.xyz"),
        "--energies", str(tmp_path / "e.csv"),
    ])
    assert rc == 0
    assert "E_tot drift" in capsys.readouterr().out
    assert (tmp_path / "t.xyz").exists()
