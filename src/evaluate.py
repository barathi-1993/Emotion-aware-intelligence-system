#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained emotion recognition model.

Example
-------
python src/evaluate.py --data data/processed/coords.zip --model models/gradient_boosting_emotion_model.pkl
"""

from __future__ import annotations

import argparse
import pickle
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def load_coords(path: str) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
            with z.open(csv_names[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained emotion model.")
    parser.add_argument("--data", default="data/processed/coords.zip", type=str)
    parser.add_argument("--model", default="models/gradient_boosting_emotion_model.pkl", type=str)
    parser.add_argument("--output-dir", default="results/evaluation", type=str)
    parser.add_argument("--test-size", default=0.30, type=float)
    parser.add_argument("--random-state", default=1234, type=int)
    args = parser.parse_args()

    df = load_coords(args.data).dropna().reset_index(drop=True)
    X = df.drop("class", axis=1)
    y = df["class"]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "classification_report.txt", "w") as f:
        f.write(report)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=45, cmap="viridis", colorbar=True)
    plt.title("Emotion Recognition Confusion Matrix")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    print(report)
    print(f"[Evaluate] Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
