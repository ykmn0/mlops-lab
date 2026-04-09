from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

def train_and_save_model(output_path: str = "model.pkl") -> float:
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_experiment("iris")

    with mlflow.start_run() as run:
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)

        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name="iris-model",
        )
        client = MlflowClient()
        registered_version = getattr(model_info, "registered_model_version", None)
        if not registered_version:
            raise RuntimeError(
                f"failed to resolve registered model version for run_id={run.info.run_id}"
            )
        client.set_registered_model_alias(
            name="iris-model",
            alias="champion",
            version=registered_version,
        )
        alias = client.get_model_version_by_alias(
            name="iris-model",
            alias="champion",
        )
        print(f"champion alias -> iris-model version {alias.version}")

        

    output = Path(output_path)
    joblib.dump(model, output)

    return accuracy


if __name__ == "__main__":
    accuracy = train_and_save_model("model.pkl")
    print(f"accuracy={accuracy:.4f}")