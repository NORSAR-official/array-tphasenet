import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.smoke
def test_minimal_notebook_is_valid_json():
    notebooks = [
        REPO_ROOT / "notebooks" / "reproduce_main_results_walkthrough.ipynb",
    ]
    for notebook_path in notebooks:
        assert notebook_path.exists()
        data = json.loads(notebook_path.read_text(encoding="utf-8"))
        assert data.get("nbformat", 0) >= 4
        assert isinstance(data.get("cells"), list)
        assert len(data["cells"]) >= 3
