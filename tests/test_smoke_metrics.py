from types import SimpleNamespace

import numpy as np
import pytest

from utils import f1_score, precision, recall


@pytest.mark.smoke
def test_precision_recall_f1_toy_example():
    cfg = SimpleNamespace(data=SimpleNamespace(sampling_rate=40.0))

    y_true = np.zeros((2, 120, 1), dtype=np.float32)
    y_pred = np.zeros((2, 120, 1), dtype=np.float32)

    y_true[0, 30, 0] = 1.0
    y_true[1, 90, 0] = 1.0

    y_pred[0, 31, 0] = 1.0
    y_pred[1, 89, 0] = 1.0

    p = precision(y_true, y_pred, cfg, dt=1.0, th=0.5, livemode=True)
    r = recall(y_true, y_pred, cfg, dt=1.0, th=0.5, livemode=True)
    f1 = f1_score(y_true, y_pred, cfg, dt=1.0, th=0.5, livemode=True)

    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert p > 0.9
    assert r > 0.9
    assert f1 > 0.9
