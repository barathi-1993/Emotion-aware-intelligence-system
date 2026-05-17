#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train classical ML emotion classifiers on MediaPipe landmark features.

The manuscript evaluated multiple classifiers and selected Gradient Boosting
for real-time deployment because it achieved the best average accuracy.

Example
-------
python src/train.py --data data/processed/coords.csv --model-out models/gradient_boosting_emotion_model.pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def load_coords(path: str) -> pd.DataFrame:
    """Load coords.csv directly or from coords.zip."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            csv_names = [n for n in z.namelist() if n.endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError("No CSV file found inside zip.")
            with z.open(csv_names[0]) as f:
                return pd.read_csv(f)
    return pd.read_csv(path)


def build_pipelines(random_state: int = 1234):
    """Build classifier pipelines used in the paper-style comparison."""
    return {
        "lr": make_pipeline(
            StandardScaler(),
            LogisticRegression(random_state=random_state, max_iter=2000),
        ),
        "rf": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(random_state=random_state),
        ),
        "gb": make_pipeline(
            StandardScaler(),
            GradientBoostingClassifier(random_state=random_state),
        ),
        "knn": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=1),
        ),
        "svm": make_pipeline(
            StandardScaler(),
            SVC(probability=True, random_state=random_state),
        ),
        "dt": make_pipeline(
            StandardScaler(),
            DecisionTreeClassifier(random_state=random_state),
        ),
        "ridge": make_pipeline(
            StandardScaler(),
            RidgeClassifier(),
        ),
        "naive_bayes": make_pipeline(
            StandardScaler(),
            GaussianNB(),
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Train emotion recognition classifiers.")
    parser.add_argument("--data", default="data/processed/coords.zip", type=str)
    parser.add_argument("--model-out", default="models/gradient_boosting_emotion_model.pkl", type=str)
    parser.add_argument("--metrics-out", default="results/metrics.json", type=str)
    parser.add_argument("--test-size", default=0.30, type=float)
    parser.add_argument("--random-state", default=1234, type=int)
    args = parser.parse_args()

    df = load_coords(args.data)
    if "class" not in df.columns:
        raise ValueError("Input CSV must contain a 'class' column.")

    df = df.dropna().reset_index(drop=True)
    X = df.drop("class", axis=1)
    y = df["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    pipelines = build_pipelines(args.random_state)
    results = {}
    fitted = {}

    for name, model in pipelines.items():
        print(f"[Train] Fitting {name}...")
        model.fit(X_train, y_train)
        fitted[name] = model

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            "accuracy": float(acc),
            "classification_report": classification_report(y_test, y_pred, output_dict=True),
        }
        print(f"  {name}: accuracy={acc:.4f}")

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = fitted[best_name]

    model_out = Path(args.model_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump(best_model, f)

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_out, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    print(f"[Train] Best model: {best_name}")
    print(f"[Train] Saved model to {model_out}")
    print(f"[Train] Saved metrics to {metrics_out}")


if __name__ == "__main__":
    main()
