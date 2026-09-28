# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from evimetaskin.datasets import DIAGNOSIS_TO_INDEX, build_clinical_prompt


def prepare_split(data_root: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(data_root / f"{split}-metadata.csv")
    image_dir = data_root / f"{split}-image" / "image"
    df["split"] = split
    df["image_path"] = df["isic_id"].map(lambda x: str((image_dir / f"{x}.jpg").resolve()))
    df["image_exists"] = df["image_path"].map(lambda p: Path(p).exists())
    df["binary_label"] = df["target"].astype(int)
    df["binary_name"] = df["binary_label"].map({0: "benign", 1: "malignant"})
    df["multiclass_label"] = df["diagnosis"].map(DIAGNOSIS_TO_INDEX)
    if df["multiclass_label"].isna().any():
        unknown = sorted(df.loc[df["multiclass_label"].isna(), "diagnosis"].dropna().unique().tolist())
        raise ValueError(f"Unknown diagnosis labels in {split}: {unknown}")
    df["multiclass_label"] = df["multiclass_label"].astype(int)
    df["clinical_prompt_binary"] = df.apply(lambda row: build_clinical_prompt(row, "binary"), axis=1)
    df["clinical_prompt_multiclass"] = df.apply(lambda row: build_clinical_prompt(row, "multiclass"), axis=1)
    for col in ["age_approx", "sex", "anatom_site_general"]:
        df[f"{col}_missing"] = df[col].isna() | (df[col].astype(str).str.strip() == "")
    keep_cols = [
        "isic_id",
        "split",
        "image_path",
        "image_exists",
        "binary_label",
        "binary_name",
        "multiclass_label",
        "diagnosis",
        "age_approx",
        "sex",
        "anatom_site_general",
        "lesion_id",
        "diagnosis_confirm_type",
        "clinical_prompt_binary",
        "clinical_prompt_multiclass",
        "age_approx_missing",
        "sex_missing",
        "anatom_site_general_missing",
    ]
    return df[keep_cols]


def main():
    parser = argparse.ArgumentParser(description="Prepare unified CSV manifests for ISIC2018 Task 3 classification.")
    parser.add_argument("--data-root", default="data/ISIC2018 Task3")
    parser.add_argument("--out-dir", default="data/ISIC2018 Task3/manifests")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_splits = []
    for split in ("train", "validation", "test"):
        manifest = prepare_split(data_root, split)
        manifest.to_csv(out_dir / f"{split}_manifest.csv", index=False, encoding="utf-8-sig")
        all_splits.append(manifest)
        print(f"{split}: {len(manifest)} samples, matched images={int(manifest['image_exists'].sum())}")
    pd.concat(all_splits, ignore_index=True).to_csv(out_dir / "all_manifest.csv", index=False, encoding="utf-8-sig")
    print(f"Saved manifests to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
