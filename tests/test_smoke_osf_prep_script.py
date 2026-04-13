import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.smoke
def test_osf_prep_script_dry_run():
    cmd = [
        "bash",
        "scripts/prepare_osf_data.sh",
        "--dry-run",
        "--merge-only",
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        f"osf prep dry-run failed\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "Running merge step for known split files..." in result.stdout
