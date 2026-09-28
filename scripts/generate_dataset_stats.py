# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evimetaskin.datasets import DIAGNOSIS_TO_INDEX


def _read_split(root: Path, split: str) -> pd.DataFrame:
    path = root / f"{split}-metadata.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df["split"] = split
    df["image_file"] = df["isic_id"].map(lambda x: root / f"{split}-image" / "image" / f"{x}.jpg")
    df["image_exists"] = df["image_file"].map(lambda p: p.exists())
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate ISIC2018 Task 3 dataset statistics for paper/reporting.")
    parser.add_argument("--data-root", default="data/ISIC2018 Task3")
    parser.add_argument("--out-dir", default="results/dataset_stats")
    args = parser.parse_args()

    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = [_read_split(root, split) for split in ("train", "validation", "test")]
    df = pd.concat(frames, ignore_index=True)

    binary_map = pd.DataFrame(
        [
            {"target": 0, "binary_label": "benign"},
            {"target": 1, "binary_label": "malignant"},
        ]
    )
    class_map = pd.DataFrame(
        [{"class_index": idx, "diagnosis": label} for label, idx in sorted(DIAGNOSIS_TO_INDEX.items(), key=lambda x: x[1])]
    )

    binary_counts = (
        df.groupby(["split", "target"], dropna=False)
        .size()
        .reset_index(name="count")
        .merge(binary_map, on="target", how="left")
        [["split", "target", "binary_label", "count"]]
    )
    diagnosis_counts = (
        df.groupby(["split", "diagnosis"], dropna=False)
        .size()
        .reset_index(name="count")
        .merge(class_map, on="diagnosis", how="left")
        [["split", "class_index", "diagnosis", "count"]]
        .sort_values(["split", "class_index"])
    )
    missing_cols = ["age_approx", "sex", "anatom_site_general", "benign_malignant", "diagnosis", "target", "lesion_id"]
    missing_rows = []
    for split, split_df in df.groupby("split"):
        for col in missing_cols:
            if col in split_df.columns:
                missing_rows.append({"split": split, "field": col, "missing_count": int(split_df[col].isna().sum())})
        missing_rows.append({"split": split, "field": "image_file", "missing_count": int((~split_df["image_exists"]).sum())})
    missing = pd.DataFrame(missing_rows)
    summary = (
        df.groupby("split")
        .agg(total_samples=("isic_id", "count"), matched_images=("image_exists", "sum"), unique_lesions=("lesion_id", "nunique"))
        .reset_index()
    )

    binary_counts.to_csv(out_dir / "binary_counts_by_split.csv", index=False, encoding="utf-8-sig")
    diagnosis_counts.to_csv(out_dir / "diagnosis_counts_by_split.csv", index=False, encoding="utf-8-sig")
    missing.to_csv(out_dir / "missing_values_by_split.csv", index=False, encoding="utf-8-sig")
    binary_map.to_csv(out_dir / "binary_label_mapping.csv", index=False, encoding="utf-8-sig")
    class_map.to_csv(out_dir / "diagnosis_label_mapping.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out_dir / "dataset_summary.csv", index=False, encoding="utf-8-sig")

    report = [
        "# ISIC2018 Task 3 Dataset Statistics",
        "",
        "## Summary",
        "```text",
        summary.to_string(index=False),
        "```",
        "",
        "## Binary Counts",
        "```text",
        binary_counts.to_string(index=False),
        "```",
        "",
        "## Diagnosis Counts",
        "```text",
        diagnosis_counts.to_string(index=False),
        "```",
        "",
        "## Missing Values",
        "```text",
        missing.to_string(index=False),
        "```",
        "",
        "## Label Mapping",
        "```text",
        class_map.to_string(index=False),
        "```",
    ]
    (out_dir / "dataset_statistics.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved dataset statistics to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
