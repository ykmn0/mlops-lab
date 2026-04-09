from pathlib import Path

import joblib

from train import train_and_save_model


def test_train_and_save_model_creates_artifact(tmp_path: Path):
    model_path = tmp_path / "model.pkl"

    accuracy = train_and_save_model(str(model_path))

    assert model_path.exists()
    assert 0.0 <= accuracy <= 1.0

    model = joblib.load(model_path)
    assert hasattr(model, "predict")
