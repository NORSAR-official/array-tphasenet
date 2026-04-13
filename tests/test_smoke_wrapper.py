import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.smoke
def test_wrapper_dry_run_with_default_config():
    cmd = [
        "bash",
        "scripts/reproduce_benchmark.sh",
        "--config",
        "config_1stat.yaml",
        "--dry-run",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"wrapper dry-run failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "python train.py --config" in result.stdout
    # For default config, train.py already writes test predictions, so the
    # wrapper may skip an explicit predict-test stage.
    assert (
        "python predict_on_testdata.py --config" in result.stdout
        or "Skipping predict-test: train.py already produced test predictions" in result.stdout
    )
    assert "python evaluate_on_testdata.py --config" in result.stdout
    assert "python predict_continuous.py -c" in result.stdout
    assert "python evaluate_continuous.py --config" in result.stdout
