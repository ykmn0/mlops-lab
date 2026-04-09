from train import train_and_save_model


def test_train_and_save_model_returns_accuracy(monkeypatch):
    class DummyModelInfo:
        registered_model_version = "1"

    class DummyModelVersion:
        version = "1"

    monkeypatch.setattr(
        "train.mlflow.sklearn.log_model",
        lambda *args, **kwargs: DummyModelInfo(),
    )
    monkeypatch.setattr(
        "train.MlflowClient.set_registered_model_alias",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "train.MlflowClient.get_model_version_by_alias",
        lambda *args, **kwargs: DummyModelVersion(),
    )

    accuracy = train_and_save_model()
    assert 0.0 <= accuracy <= 1.0
