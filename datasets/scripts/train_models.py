"""Train + evaluate two small models on the synthetic datasets.

    pip install scikit-learn numpy
    python datasets/scripts/train_models.py

1. alert classifier  : detections.csv  features -> is_alert
2. re-ID pair linker : reid_pairs.csv   features -> same_identity
Prints test metrics. Use the real datasets (datasets/README.md) for numbers worth quoting.
"""
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

D = Path(__file__).resolve().parents[1] / "synthetic"
OBJS = ["person", "car", "truck", "bike", "bag"]


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def det_xy(rows):
    X = [[float(r["hour"]), float(r["confidence"]), float(r["dwell_sec"]), float(r["speed_px_s"]), float(r["in_restricted_zone"]),
          float(r["persons_in_frame"]), float(r["w"]) * float(r["h"]), OBJS.index(r["object_type"]), float(r["camera_id"][-1])] for r in rows]
    return np.array(X), np.array([int(r["is_alert"]) for r in rows])


def pair_xy(rows):
    X = [[float(r["time_gap_s"]), float(r["camera_hops"]), float(r["color_match"]), float(r["size_ratio"]), float(r["embedding_cosine"])] for r in rows]
    return np.array(X), np.array([int(r["same_identity"]) for r in rows])


def report(name, model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    print(f"{name}: acc={accuracy_score(yte, p):.3f} precision={precision_score(yte, p):.3f} recall={recall_score(yte, p):.3f} f1={f1_score(yte, p):.3f}")


if __name__ == "__main__":
    report("alert classifier", GradientBoostingClassifier(random_state=0), *det_xy(load(D / "train/detections.csv")), *det_xy(load(D / "test/detections.csv")))
    report("re-ID pair linker", GradientBoostingClassifier(random_state=0), *pair_xy(load(D / "train/reid_pairs.csv")), *pair_xy(load(D / "test/reid_pairs.csv")))
