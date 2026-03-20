import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from sklearn.preprocessing import LabelBinarizer


def main():
    params = yaml.safe_load(open("params.yaml"))
    data_cfg = params["data"]

    raw_train = Path(data_cfg["raw_train"])
    raw_test = Path(data_cfg["raw_test"])
    out_dir = Path(data_cfg["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    img_size = data_cfg["image_size"]

    train_df = pd.read_csv(raw_train)
    test_df = pd.read_csv(raw_test)

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    x_train = train_df.drop("label", axis=1).values / 255.0
    x_test = test_df.drop("label", axis=1).values / 255.0

    x_train = x_train.reshape(-1, img_size, img_size, 1)
    x_test = x_test.reshape(-1, img_size, img_size, 1)

    binarizer = LabelBinarizer()
    y_train = binarizer.fit_transform(y_train)
    y_test = binarizer.transform(y_test)

    np.savez(out_dir / "train.npz", x=x_train, y=y_train)
    np.savez(out_dir / "test.npz", x=x_test, y=y_test)
    np.save(out_dir / "classes.npy", binarizer.classes_)

    print(f"Train: {x_train.shape}, Test: {x_test.shape}, Classes: {len(binarizer.classes_)}")


if __name__ == "__main__":
    main()
