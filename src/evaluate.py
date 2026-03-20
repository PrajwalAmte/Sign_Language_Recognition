import json
import numpy as np
import yaml
from pathlib import Path
from keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)


LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")


def main():
    params = yaml.safe_load(open("params.yaml"))
    data_cfg = params["data"]

    processed = Path(data_cfg["processed_dir"])
    test_data = np.load(processed / "test.npz")
    x_test, y_test_onehot = test_data["x"], test_data["y"]
    y_true = np.argmax(y_test_onehot, axis=1)

    model = load_model("models/smnist.h5")
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))

    report = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "classification_report": report,
    }

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    with open(metrics_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Accuracy:    {acc:.4f}")
    print(f"F1 (macro):  {f1_macro:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(classification_report(y_true, y_pred, target_names=LABELS))


if __name__ == "__main__":
    main()
