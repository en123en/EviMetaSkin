# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import pandas as pd


PH2_DIAGNOSIS_MAP = {
    "common nevus": ("nevus", 0),
    "atypical nevus": ("nevus", 0),
    "melanoma": ("melanoma", 1),
}

PAD_DIAGNOSIS_MAP = {
    "NEV": ("nevus", 0),
    "SEK": ("pigmented benign keratosis", 0),
    "ACK": ("actinic keratosis", 0),
    "BCC": ("basal cell carcinoma", 1),
    "SCC": ("squamous cell carcinoma", 1),
    "MEL": ("melanoma", 1),
}

PAD_SEX_MAP = {
    "MALE": "male",
    "FEMALE": "female",
}

PAD_SITE_MAP = {
    "ARM": "upper extremity",
    "FOREARM": "upper extremity",
    "HAND": "upper extremity",
    "FACE": "head/neck",
    "NECK": "head/neck",
    "SCALP": "head/neck",
    "EAR": "head/neck",
    "NOSE": "head/neck",
    "LIP": "head/neck",
    "CHEST": "anterior torso",
    "ABDOMEN": "anterior torso",
    "BACK": "posterior torso",
    "FOOT": "lower extremity",
    "LEG": "lower extremity",
    "THIGH": "lower extremity",
}


def _safe_str(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def prepare_ph2(ph2_root: Path, output_dir: Path) -> dict:
    csv_path = ph2_root / "PH2_simple_dataset.csv"
    image_dir = ph2_root / "images"
    if not csv_path.exists():
        raise FileNotFoundError(f"PH2 label file not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"PH2 image directory not found: {image_dir}")

    df = pd.read_csv(csv_path)
    rows = []
    for _, row in df.iterrows():
        raw_diag = _safe_str(row.get("diagnosis")).lower()
        if raw_diag not in PH2_DIAGNOSIS_MAP:
            raise ValueError(f"Unknown PH2 diagnosis: {row.get('diagnosis')}")
        diagnosis, target = PH2_DIAGNOSIS_MAP[raw_diag]
        image_id = _safe_str(row.get("image_name"))
        image_path = image_dir / f"{image_id}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"PH2 image not found: {image_path}")
        rows.append(
            {
                "dataset": "PH2",
                "isic_id": image_id,
                "image_path": str(image_path),
                "target": target,
                "diagnosis": diagnosis,
                "original_diagnosis": row.get("diagnosis"),
                "age_approx": "",
                "sex": "",
                "anatom_site_general": "",
            }
        )

    out = pd.DataFrame(rows)
    out_path = output_dir / "ph2_external_manifest.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return {
        "dataset": "PH2",
        "manifest": str(out_path),
        "n": int(len(out)),
        "target_counts": out["target"].value_counts().sort_index().to_dict(),
        "diagnosis_counts": out["diagnosis"].value_counts().to_dict(),
    }


def maybe_extract_pad_images(pad_root: Path) -> None:
    image_dir = pad_root / "images"
    pngs = list(image_dir.rglob("*.png"))
    if pngs:
        return
    zip_paths = sorted(image_dir.glob("*.zip"))
    if not zip_paths:
        raise FileNotFoundError(f"No PAD-UFES-20 images or zip files found under: {image_dir}")
    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(image_dir)


def find_pad_image(pad_root: Path, img_id: str) -> Path:
    image_dir = pad_root / "images"
    direct = image_dir / img_id
    if direct.exists():
        return direct
    matches = list(image_dir.rglob(img_id))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"PAD-UFES-20 image not found: {img_id}")


def prepare_pad(pad_root: Path, output_dir: Path, extract_images: bool) -> dict:
    metadata_path = pad_root / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"PAD-UFES-20 metadata not found: {metadata_path}")
    if extract_images:
        maybe_extract_pad_images(pad_root)

    df = pd.read_csv(metadata_path)
    rows = []
    skipped = {}
    for _, row in df.iterrows():
        raw_diag = _safe_str(row.get("diagnostic")).upper()
        if raw_diag not in PAD_DIAGNOSIS_MAP:
            skipped[raw_diag] = skipped.get(raw_diag, 0) + 1
            continue
        img_id = _safe_str(row.get("img_id"))
        image_path = find_pad_image(pad_root, img_id)
        diagnosis, target = PAD_DIAGNOSIS_MAP[raw_diag]
        sex = PAD_SEX_MAP.get(_safe_str(row.get("gender")).upper(), "")
        site = PAD_SITE_MAP.get(_safe_str(row.get("region")).upper(), "")
        rows.append(
            {
                "dataset": "PAD-UFES-20",
                "isic_id": Path(img_id).stem,
                "image_path": str(image_path),
                "target": target,
                "diagnosis": diagnosis,
                "original_diagnosis": raw_diag,
                "age_approx": row.get("age", ""),
                "sex": sex,
                "anatom_site_general": site,
            }
        )

    out = pd.DataFrame(rows)
    out_path = output_dir / "pad_ufes20_external_manifest.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    return {
        "dataset": "PAD-UFES-20",
        "manifest": str(out_path),
        "n": int(len(out)),
        "target_counts": out["target"].value_counts().sort_index().to_dict(),
        "diagnosis_counts": out["diagnosis"].value_counts().to_dict(),
        "skipped_diagnosis_counts": skipped,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare PH2 and PAD-UFES-20 external validation manifests.")
    parser.add_argument("--ph2-root", default="data/PH2-dataset-master")
    parser.add_argument("--pad-root", default="data/zr7vgbcyr2-1")
    parser.add_argument("--output-dir", default="data/external_validation")
    parser.add_argument("--extract-pad", action="store_true", help="Extract PAD-UFES-20 image zip files before indexing.")
    parser.add_argument("--skip-ph2", action="store_true")
    parser.add_argument("--skip-pad", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    if not args.skip_ph2:
        summary.append(prepare_ph2(Path(args.ph2_root), output_dir))
    if not args.skip_pad:
        summary.append(prepare_pad(Path(args.pad_root), output_dir, args.extract_pad))

    summary_path = output_dir / "external_validation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
