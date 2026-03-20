import sys
import json
import numpy as np
import yaml
import mlflow
from pathlib import Path
from keras.models import Sequential
from keras.layers import (
    Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization,
)
from keras.callbacks import ReduceLROnPlateau
from keras.src.legacy.preprocessing.image import ImageDataGenerator


def log(msg):
    print(msg, flush=True)


def build_model(cfg: dict) -> Sequential:
    filters = cfg["conv_filters"]
    k = cfg["kernel_size"]

    return Sequential([
        Conv2D(filters[0], (k, k), strides=1, padding="same",
               activation="relu", input_shape=(28, 28, 1)),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding="same"),

        Conv2D(filters[1], (k, k), strides=1, padding="same", activation="relu"),
        Dropout(cfg["conv_dropout_rate"]),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding="same"),

        Conv2D(filters[2], (k, k), strides=1, padding="same", activation="relu"),
        BatchNormalization(),
        MaxPool2D((2, 2), strides=2, padding="same"),

        Flatten(),
        Dense(cfg["dense_units"], activation="relu"),
        Dropout(cfg["dropout_rate"]),
        Dense(cfg["num_classes"], activation="softmax"),
    ])


def main():
    log("[1/6] Loading params...")
    params = yaml.safe_load(open("params.yaml"))
    train_cfg = params["train"]
    model_cfg = params["model"]
    aug_cfg = params["augmentation"]
    data_cfg = params["data"]

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("sign-language-recognition")

    log("[2/6] Loading data...")
    processed = Path(data_cfg["processed_dir"])
    train_data = np.load(processed / "train.npz")
    test_data = np.load(processed / "test.npz")
    x_train, y_train = train_data["x"], train_data["y"]
    x_test, y_test = test_data["x"], test_data["y"]
    log(f"       Train: {x_train.shape}, Test: {x_test.shape}")

    log("[3/6] Augmentation...")
    datagen = ImageDataGenerator(
        rotation_range=aug_cfg["rotation_range"],
        zoom_range=aug_cfg["zoom_range"],
        width_shift_range=aug_cfg["width_shift_range"],
        height_shift_range=aug_cfg["height_shift_range"],
    )
    datagen.fit(x_train)

    lr_callback = ReduceLROnPlateau(
        monitor=train_cfg["lr_reduction"]["monitor"],
        patience=train_cfg["lr_reduction"]["patience"],
        factor=train_cfg["lr_reduction"]["factor"],
        min_lr=train_cfg["lr_reduction"]["min_lr"],
    )

    log("[4/6] Building model...")
    model = build_model(model_cfg)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    log(f"[5/6] Training — {train_cfg['epochs']} epochs, batch {train_cfg['batch_size']}")
    with mlflow.start_run():
        mlflow.log_params({f"train.{k}": v for k, v in train_cfg.items() if not isinstance(v, dict)})
        mlflow.log_params({f"model.{k}": v for k, v in model_cfg.items()})

        history = model.fit(
            datagen.flow(x_train, y_train, batch_size=train_cfg["batch_size"]),
            epochs=train_cfg["epochs"],
            validation_data=(x_test, y_test),
            callbacks=[lr_callback],
            verbose=1,
        )

        best_val_acc = float(max(history.history["val_accuracy"]))
        best_val_loss = float(min(history.history["val_loss"]))
        final_train_acc = float(history.history["accuracy"][-1])

        mlflow.log_metric("val_accuracy", best_val_acc)
        mlflow.log_metric("val_loss", best_val_loss)
        mlflow.log_metric("train_accuracy", final_train_acc)

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        model.save(models_dir / "smnist.h5")
        mlflow.log_artifact(str(models_dir / "smnist.h5"))

        metrics = {
            "val_accuracy": best_val_acc,
            "val_loss": best_val_loss,
            "train_accuracy": final_train_acc,
        }
        metrics_dir = Path("metrics")
        metrics_dir.mkdir(exist_ok=True)
        with open(metrics_dir / "train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    log(f"[6/6] Done — val_accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
