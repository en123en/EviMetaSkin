# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score


def binary_metrics(df: pd.DataFrame) -> dict:
    y_true = df["true_label"].astype(int).to_numpy()
    y_pred = df["pred_label"].astype(int).to_numpy()
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if len(df) else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true)) > 1 else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)) if len(df) else 0.0,
        "recall": float(recall_score(y_true, y_pred, zero_division=0)) if len(df) else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)) if len(df) else 0.0,
        "specificity": float(tn / (tn + fp + 1e-8)),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def uncertainty_bins(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    work = df.copy()
    work["correct"] = work["true_label"].astype(int) == work["pred_label"].astype(int)
    work["malignant"] = work["true_label"].astype(int) == 1
    work["uncertainty_bin"] = pd.qcut(
        work["uncertainty"],
        q=bins,
        labels=[f"Q{i + 1}" for i in range(bins)],
        duplicates="drop",
    )
    rows = []
    for name, group in work.groupby("uncertainty_bin", observed=False):
        metrics = binary_metrics(group)
        metrics.update(
            {
                "uncertainty_bin": str(name),
                "uncertainty_min": float(group["uncertainty"].min()),
                "uncertainty_max": float(group["uncertainty"].max()),
                "uncertainty_mean": float(group["uncertainty"].mean()),
                "error_rate": float(1.0 - group["correct"].mean()),
                "malignant_rate": float(group["malignant"].mean()),
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def selective_prediction(df: pd.DataFrame, coverages: list[float]) -> pd.DataFrame:
    work = df.sort_values("uncertainty", ascending=True).reset_index(drop=True)
    rows = []
    total = len(work)
    for coverage in coverages:
        keep_n = max(1, int(round(total * coverage)))
        retained = work.iloc[:keep_n].copy()
        referred = work.iloc[keep_n:].copy()
        metrics = binary_metrics(retained)
        metrics.update(
            {
                "coverage": float(coverage),
                "retained_n": int(len(retained)),
                "referred_n": int(len(referred)),
                "retained_uncertainty_mean": float(retained["uncertainty"].mean()),
                "referred_uncertainty_mean": float(referred["uncertainty"].mean()) if len(referred) else 0.0,
                "referred_malignant_rate": float((referred["true_label"].astype(int) == 1).mean()) if len(referred) else 0.0,
                "referred_error_rate": float((referred["true_label"].astype(int) != referred["pred_label"].astype(int)).mean()) if len(referred) else 0.0,
            }
        )
        rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_correct_wrong(df: pd.DataFrame) -> dict:
    correct_mask = df["true_label"].astype(int) == df["pred_label"].astype(int)
    correct_unc = df.loc[correct_mask, "uncertainty"]
    wrong_unc = df.loc[~correct_mask, "uncertainty"]
    return {
        "total_n": int(len(df)),
        "correct_n": int(correct_mask.sum()),
        "wrong_n": int((~correct_mask).sum()),
        "mean_uncertainty_correct": float(correct_unc.mean()) if len(correct_unc) else 0.0,
        "mean_uncertainty_wrong": float(wrong_unc.mean()) if len(wrong_unc) else 0.0,
        "median_uncertainty_correct": float(correct_unc.median()) if len(correct_unc) else 0.0,
        "median_uncertainty_wrong": float(wrong_unc.median()) if len(wrong_unc) else 0.0,
        "uncertainty_gap_wrong_minus_correct": float(wrong_unc.mean() - correct_unc.mean()) if len(correct_unc) and len(wrong_unc) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze uncertainty reliability and referral behavior.")
    parser.add_argument("--run-dir", required=True, help="Run directory containing test_predictions.csv and test_metrics.json.")
    parser.add_argument("--bins", type=int, default=3, help="Number of uncertainty quantile bins.")
    parser.add_argument(
        "--coverages",
        default="1.0,0.95,0.9,0.85,0.8,0.7",
        help="Comma-separated retained-sample coverages for selective prediction.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    pred_path = run_dir / "test_predictions.csv"
    metrics_path = run_dir / "test_metrics.json"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)

    out_dir = run_dir / "uncertainty_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pred_path)
    required = {"true_label", "pred_label", "uncertainty"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"test_predictions.csv missing required columns: {sorted(missing)}")

    with metrics_path.open("r", encoding="utf-8") as f:
        test_metrics = json.load(f)

    coverages = [float(item.strip()) for item in args.coverages.split(",") if item.strip()]
    correct_wrong = summarize_correct_wrong(df)
    bins_df = uncertainty_bins(df, args.bins)
    selective_df = selective_prediction(df, coverages)

    bins_df.to_csv(out_dir / "uncertainty_bins.csv", index=False, encoding="utf-8-sig")
    selective_df.to_csv(out_dir / "selective_prediction.csv", index=False, encoding="utf-8-sig")
    (out_dir / "correct_wrong_uncertainty.json").write_text(
        json.dumps(correct_wrong, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report = [
        "# Uncertainty Analysis",
        "",
        "## Original Test Metrics",
        "```json",
        json.dumps(test_metrics, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Correct vs Wrong Uncertainty",
        "```json",
        json.dumps(correct_wrong, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Uncertainty Quantile Bins",
        "```text",
        bins_df.to_string(index=False),
        "```",
        "",
        "## Selective Prediction / Referral",
        "```text",
        selective_df.to_string(index=False),
        "```",
    ]
    (out_dir / "uncertainty_analysis.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved uncertainty analysis to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
