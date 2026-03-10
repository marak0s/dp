import subprocess


def test_cli_run_awgn_small() -> None:
    cmd = [
        "python",
        "-m",
        "comm_ai.experiments.run_experiment",
        "--config",
        "src/comm_ai/config/experiments/awgn_small.yaml",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    assert "Saved run to:" in result.stdout
