# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def metric_values(y_true, y_pred, y_score):
    values = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1 and y_score is not None:
        values["auc"] = float(roc_auc_score(y_true, y_score))
    return values


def bootstrap_ci(df: pd.DataFrame, metrics: list[str], n_boot: int, seed: int):
    required = {"true_label", "pred_label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Prediction file missing required columns: {sorted(missing)}")
    has_score = "score_malignant" in df.columns
    y_true = df["true_label"].astype(int).to_numpy()
    y_pred = df["pred_label"].astype(int).to_numpy()
    y_score = df["score_malignant"].astype(float).to_numpy() if has_score else None

    point = metric_values(y_true, y_pred, y_score)
    rng = np.random.default_rng(seed)
    samples = {metric: [] for metric in metrics if metric in point}
    n = len(df)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_true = y_true[idx]
        boot_pred = y_pred[idx]
        boot_score = y_score[idx] if y_score is not None else None
        boot_values = metric_values(boot_true, boot_pred, boot_score)
        for metric in samples:
            if metric in boot_values:
                samples[metric].append(boot_values[metric])

    rows = []
    summary = {}
    for metric in samples:
        arr = np.asarray(samples[metric], dtype=np.float64)
        lower, upper = np.percentile(arr, [2.5, 97.5])
        row = {
            "metric": metric,
            "point": float(point[metric]),
            "ci_lower": float(lower),
            "ci_upper": float(upper),
            "ci_text": f"{point[metric]:.4f} ({lower:.4f}-{upper:.4f})",
        }
        rows.append(row)
        summary[metric] = row
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description="Bootstrap 95% confidence intervals for prediction CSV files.")
    parser.add_argument("--predictions", required=True, help="CSV with true_label, pred_label, and optionally score_malignant.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics", nargs="+", default=["accuracy", "precision", "recall", "f1", "auc"])
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else pred_path.parent / "bootstrap_ci"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(pred_path)
    rows, summary = bootstrap_ci(df, args.metrics, args.n_bootstrap, args.seed)
    pd.DataFrame(rows).to_csv(output_dir / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    (output_dir / "bootstrap_ci.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
