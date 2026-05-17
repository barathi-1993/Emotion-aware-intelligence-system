#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot learning curve for the Gradient Boosting model.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import ShuffleSplit, learning_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_coords(path):
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
            with z.open(csv_names[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/coords.zip")
    parser.add_argument("--output", default="results/evaluation/gb_learning_curve.png")
    args = parser.parse_args()

    df = load_coords(args.data).dropna()
    X = df.drop("class", axis=1)
    y = df["class"]

    estimator = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=1234))
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    train_sizes, train_scores, val_scores = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        train_sizes=np.linspace(0.1, 1.0, 5),
        n_jobs=1,
    )

    plt.figure(figsize=(9, 6))
    plt.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training score")
    plt.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.title("Gradient Boosting Learning Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
