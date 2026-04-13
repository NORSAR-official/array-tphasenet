from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf


@pytest.mark.smoke
def test_tf_model_save_load_and_predict(tmp_path: Path):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(32, 3)),
            tf.keras.layers.Conv1D(8, 3, padding="same", activation="relu"),
            tf.keras.layers.Conv1D(3, 1, padding="same", activation="softmax"),
        ]
    )

    x = np.random.rand(2, 32, 3).astype(np.float32)
    y = model(x, training=False).numpy()
    assert y.shape == (2, 32, 3)

    model_path = tmp_path / "saved_model_smoke.tf"
    model.save(str(model_path), save_format="tf")

    loaded = tf.keras.models.load_model(str(model_path), compile=False)
    pred = loaded(x, training=False).numpy()

    assert pred.shape == (2, 32, 3)
    assert np.all(np.isfinite(pred))
