import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.smoke
def test_preflight_benchmark_stage():
    cmd = [
        sys.executable,
        "scripts/check_inputs.py",
        "--config",
        "config_1stat.yaml",
        "--stage",
        "benchmark",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"preflight failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
