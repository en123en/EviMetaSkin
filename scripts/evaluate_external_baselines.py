# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evimetaskin.baselines import build_model
from evimetaskin.datasets import default_image_transform
from evimetaskin.metrics import binary_preds_from_scores, compute_metrics


class ExternalImageDataset(Dataset):
    def __init__(self, manifest_path: str | Path, image_size: int = 224):
        self.manifest_path = Path(manifest_path)
        self.transform = default_image_transform(False, image_size)
        self.df = pd.read_csv(self.manifest_path)
        required = {"image_path", "target", "diagnosis", "isic_id"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        self.df = self.df[self.df["image_path"].map(lambda p: Path(str(p)).exists())].reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError(f"No matched images found in manifest: {self.manifest_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "true_label": int(row["target"]),
            "isic_id": str(row["isic_id"]),
            "diagnosis": str(row["diagnosis"]),
            "dataset": str(row.get("dataset", self.manifest_path.stem)),
            "original_diagnosis": str(row.get("original_diagnosis", "")),
            "image_path": str(row["image_path"]),
        }


def collate_batch(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "true_label": torch.tensor([item["true_label"] for item in batch], dtype=torch.long),
        "isic_id": [item["isic_id"] for item in batch],
        "diagnosis": [item["diagnosis"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "original_diagnosis": [item["original_diagnosis"] for item in batch],
        "image_path": [item["image_path"] for item in batch],
    }


def load_threshold(run_dir: Path, fallback: float = 0.5) -> float:
    path = run_dir / "threshold_selection.json"
    if not path.exists():
        return fallback
    data = json.loads(path.read_text(encoding="utf-8"))
    return float(data.get("threshold", fallback))


@torch.no_grad()
def evaluate(model, loader, device, threshold: float):
    model.eval()
    y_true, y_score, rows = [], [], []
    seen = 0
    start = time.perf_counter()
    for batch in tqdm(loader, desc="external-baseline", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).detach().float().cpu().numpy()
        labels = batch["true_label"].numpy().tolist()
        scores = probs[:, 1].tolist()
        y_true.extend(labels)
        y_score.extend(scores)
        seen += images.size(0)
        for i, isic_id in enumerate(batch["isic_id"]):
            rows.append(
                {
                    "dataset": batch["dataset"][i],
                    "isic_id": isic_id,
                    "image_path": batch["image_path"][i],
                    "diagnosis": batch["diagnosis"][i],
                    "original_diagnosis": batch["original_diagnosis"][i],
                    "true_label": int(labels[i]),
                    "score_malignant": float(scores[i]),
                }
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    y_pred = binary_preds_from_scores(y_score, threshold)
    metrics = compute_metrics(y_true, y_pred, "binary", y_score)
    metrics["threshold"] = float(threshold)
    metrics["fps"] = float(seen / elapsed) if elapsed > 0 else 0.0
    for row, pred, score in zip(rows, y_pred, y_score):
        row["pred_label"] = int(pred)
        row["confidence"] = float(score if pred == 1 else 1.0 - score)
    return metrics, rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate an ISIC-trained pure-vision baseline on an external manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    threshold = float(args.threshold) if args.threshold is not None else load_threshold(run_dir)
    manifest = Path(args.manifest)
    dataset_name = manifest.stem.replace("_external_manifest", "")
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "external_validation" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device)
    train_args = ckpt.get("args", {})
    model_name = train_args.get("model", run_dir.parent.name)
    task = train_args.get("task", "binary")
    if task != "binary":
        raise ValueError("External baseline evaluation currently supports binary checkpoints only.")
    model = build_model(model_name, 2, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    dataset = ExternalImageDataset(manifest, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch)
    metrics, rows = evaluate(model, loader, device, threshold)
    metrics["n"] = int(len(dataset))
    metrics["manifest"] = str(manifest)
    metrics["checkpoint"] = str(ckpt_path)
    metrics["trained_run_name"] = str(train_args.get("run_name", run_dir.name))
    metrics["model"] = str(model_name)

    pd.DataFrame(rows).to_csv(output_dir / "external_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "external_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
