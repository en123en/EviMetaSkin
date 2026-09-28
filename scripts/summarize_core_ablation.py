# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_RUNS = [
    {
        "model": "Image + CE",
        "run": "effb0_imageonly_ce_softmax_seed2026",
        "image": True,
        "metadata_mggf": False,
        "ce": True,
        "edl": False,
    },
    {
        "model": "Image + Metadata + CE",
        "run": "effb0_gatedmeta_ce_softmax_seed2026",
        "image": True,
        "metadata_mggf": True,
        "ce": True,
        "edl": False,
    },
    {
        "model": "Image + CE + EDL",
        "run": "effb0_imageonly_ev005_seed2026",
        "image": True,
        "metadata_mggf": False,
        "ce": True,
        "edl": True,
    },
    {
        "model": "Image + Metadata + CE + EDL (Ours)",
        "run": "effb0_gatedmeta_ev005_seed2026",
        "image": True,
        "metadata_mggf": True,
        "ce": True,
        "edl": True,
    },
]


def reliability_metrics(predictions_path: Path) -> dict:
    df = pd.read_csv(predictions_path)
    required = {"true_label", "pred_label", "uncertainty"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{predictions_path} is missing columns: {sorted(missing)}")

    errors = (df["true_label"].astype(int) != df["pred_label"].astype(int)).astype(int)
    uncertainty = df["uncertainty"].astype(float)
    correct_uncertainty = uncertainty[errors == 0]
    wrong_uncertainty = uncertainty[errors == 1]
    error_auc = float(roc_auc_score(errors, uncertainty)) if errors.nunique() > 1 else None
    uncertainty_type = (
        str(df["uncertainty_type"].dropna().iloc[0])
        if "uncertainty_type" in df.columns and not df["uncertainty_type"].dropna().empty
        else "evidential_k_over_s"
    )
    return {
        "mean_uncertainty_correct": float(correct_uncertainty.mean()),
        "mean_uncertainty_wrong": float(wrong_uncertainty.mean()),
        "uncertainty_gap": float(wrong_uncertainty.mean() - correct_uncertainty.mean()),
        "error_detection_auc": error_auc,
        "uncertainty_type": uncertainty_type,
    }


def checkbox(value: bool) -> str:
    return "Yes" if value else "No"


def format_value(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def markdown_table(rows: list[dict]) -> str:
    columns = [
        ("Model", "model"),
        ("Image", "image"),
        ("Metadata/MGGF", "metadata_mggf"),
        ("CE", "ce"),
        ("EDL", "edl"),
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1", "f1"),
        ("Specificity", "specificity"),
        ("AUC", "auc"),
        ("Error-detection AUC", "error_detection_auc"),
    ]
    lines = [
        "| " + " | ".join(title for title, _ in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for _, key in columns:
            if key in {"image", "metadata_mggf", "ce", "edl"}:
                values.append(checkbox(bool(row[key])))
            elif key == "model":
                values.append(str(row[key]))
            else:
                values.append(format_value(row.get(key)))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Build the 2x2 CE/EDL core ablation table from run outputs.")
    parser.add_argument(
        "--runs-root",
        default="results/experiments/evimetaskin/binary",
        help="Directory containing the four run folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/tables/core_ablation",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in DEFAULT_RUNS:
        run_dir = runs_root / spec["run"]
        metrics_path = run_dir / "test_metrics.json"
        predictions_path = run_dir / "test_predictions.csv"
        if not metrics_path.exists() or not predictions_path.exists():
            missing = [str(path) for path in (metrics_path, predictions_path) if not path.exists()]
            raise FileNotFoundError("Missing required result files: " + ", ".join(missing))

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        reliability = reliability_metrics(predictions_path)
        row = {
            **spec,
            "accuracy": metrics.get("accuracy"),
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "specificity": metrics.get("specificity"),
            "auc": metrics.get("auc"),
            "threshold": metrics.get("threshold"),
            "prediction_mode": metrics.get(
                "prediction_mode", "evidential" if spec["edl"] else "softmax"
            ),
            **reliability,
        }
        rows.append(row)

    table_df = pd.DataFrame(rows)
    table_df.to_csv(output_dir / "table2_core_ablation.csv", index=False, encoding="utf-8-sig")
    (output_dir / "table2_core_ablation.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown = "\n".join(
        [
            "# Table 2. Core 2x2 ablation of metadata-guided fusion and evidential learning",
            "",
            markdown_table(rows),
            "",
            "Note: CE-only runs use normalized predictive entropy; EDL runs use u = K / S. "
            "Error-detection AUC is scale-independent and is therefore the preferred reliability comparison.",
        ]
    )
    (output_dir / "table2_core_ablation.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
