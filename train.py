from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path


def train_and_save_model(output_path: str = "model.pkl") -> float:
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    output = Path(output_path)
    joblib.dump(model, output)
    return accuracy


if __name__ == "__main__":
    accuracy = train_and_save_model("model.pkl")
    print(f"accuracy={accuracy:.4f}")
