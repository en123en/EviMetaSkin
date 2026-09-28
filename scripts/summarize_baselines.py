# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "specificity",
    "auc",
    "threshold",
    "fps",
    "tp",
    "fp",
    "tn",
    "fn",
    "best_val_f1",
]


def collect_metrics(root: Path, task: str):
    task_root = root / task
    rows = []
    for metrics_path in sorted(task_root.glob("**/test_metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as f:
            metrics = json.load(f)
        relative = metrics_path.relative_to(task_root)
        parts = relative.parts
        model = parts[0] if parts else metrics_path.parent.name
        run_name = "default"
        if len(parts) > 2:
            run_name = parts[1]
        row = {
            "task": task,
            "model": model,
            "run_name": run_name,
            "metrics_path": str(metrics_path),
        }
        for key in METRIC_COLUMNS:
            row[key] = metrics.get(key)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize classification baseline metrics into paper-ready tables.")
    parser.add_argument("--root", default="results/experiments/baselines")
    parser.add_argument("--task", default="binary", choices=["binary", "multiclass"])
    parser.add_argument("--out-dir", default="results/tables")
    args = parser.parse_args()

    rows = collect_metrics(Path(args.root), args.task)
    if not rows:
        raise RuntimeError(f"No test_metrics.json files found under {Path(args.root) / args.task}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    sort_cols = [col for col in ["f1", "balanced_accuracy", "auc"] if col in df.columns]
    df = df.sort_values(sort_cols, ascending=False, na_position="last")
    csv_path = out_dir / f"{args.task}_baseline_summary.csv"
    md_path = out_dir / f"{args.task}_baseline_summary.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    display_cols = ["model", "run_name"] + [col for col in METRIC_COLUMNS if col in df.columns]
    table = df[display_cols].copy()
    for col in table.columns:
        if col not in {"model", "run_name", "tp", "fp", "tn", "fn"}:
            table[col] = table[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    md_lines = [
        f"# {args.task.capitalize()} Baseline Summary",
        "",
        "```text",
        table.to_string(index=False),
        "```",
        "",
        f"Source root: `{Path(args.root) / args.task}`",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Saved CSV: {csv_path.resolve()}")
    print(f"Saved Markdown: {md_path.resolve()}")


if __name__ == "__main__":
    main()
