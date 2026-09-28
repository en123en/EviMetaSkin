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

from evimetaskin.datasets import DIAGNOSIS_TO_INDEX, METADATA_DIM, build_clinical_prompt, build_structured_metadata, default_image_transform
from evimetaskin.metrics import binary_preds_from_scores, compute_metrics
from evimetaskin.model import EviMetaSkin


class ExternalManifestDataset(Dataset):
    def __init__(self, manifest_path: str | Path, task: str = "binary", image_size: int = 224):
        if task != "binary":
            raise ValueError("External validation currently supports binary classification only.")
        self.manifest_path = Path(manifest_path)
        self.task = task
        self.transform = default_image_transform(False, image_size)
        self.df = pd.read_csv(self.manifest_path)
        required = {"image_path", "target", "diagnosis", "isic_id"}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        self.df["image_path"] = self.df["image_path"].map(lambda p: str(Path(p)))
        self.df = self.df[self.df["image_path"].map(lambda p: Path(p).exists())].reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError(f"No matched external images found in manifest: {self.manifest_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        diagnosis = str(row["diagnosis"])
        multiclass_label = DIAGNOSIS_TO_INDEX.get(diagnosis, -1)
        return {
            "image": image,
            "binary_label": int(row["target"]),
            "multiclass_label": int(multiclass_label),
            "prompt": build_clinical_prompt(row, task=self.task),
            "metadata": build_structured_metadata(row),
            "isic_id": str(row["isic_id"]),
            "diagnosis": diagnosis,
            "dataset": str(row.get("dataset", self.manifest_path.stem)),
            "original_diagnosis": str(row.get("original_diagnosis", "")),
            "image_path": str(row["image_path"]),
        }


def collate_batch(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "binary_label": torch.tensor([item["binary_label"] for item in batch], dtype=torch.long),
        "multiclass_label": torch.tensor([item["multiclass_label"] for item in batch], dtype=torch.long),
        "prompt": [item["prompt"] for item in batch],
        "metadata": torch.tensor(np.stack([item["metadata"] for item in batch]), dtype=torch.float32),
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


def build_model_from_checkpoint(ckpt: dict, device: torch.device):
    args = ckpt.get("args", {})
    task = args.get("task", "binary")
    num_classes = 2 if task == "binary" else len(DIAGNOSIS_TO_INDEX)
    visual_backbone = args.get("visual_backbone", "efficientnet_b0")
    pretrained_visual = bool(args.get("pretrained_visual", False))
    model = EviMetaSkin(
        visual_backbone=visual_backbone,
        metadata_dim=METADATA_DIM,
        num_classes=num_classes,
        freeze_image_encoder=bool(args.get("freeze_image_encoder", False)),
        pretrained_visual=pretrained_visual,
        fusion_mode=args.get("fusion_mode", "gated"),
    )
    model.uses_structured_metadata = True
    model.disable_metadata = bool(args.get("disable_metadata", False))

    model.load_state_dict(ckpt["model"])
    return model.to(device), args


@torch.no_grad()
def evaluate(model, loader, device, threshold: float):
    model.eval()
    y_true, y_score, probs_all, uncertainty_all, rows = [], [], [], [], []
    start = time.perf_counter()
    seen = 0
    for batch in tqdm(loader, desc="external-eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        if getattr(model, "disable_metadata", False):
            metadata = torch.zeros_like(metadata)
        outputs = model(images, metadata)
        probs = outputs["evidential_probs"].detach().float().cpu().numpy()
        uncertainty = outputs["uncertainty"].detach().float().cpu().numpy()
        labels = batch["binary_label"].numpy().tolist()
        y_true.extend(labels)
        y_score.extend(probs[:, 1].tolist())
        probs_all.extend(probs.tolist())
        uncertainty_all.extend(uncertainty.tolist())
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
                    "score_malignant": float(probs[i, 1]),
                    "uncertainty": float(uncertainty[i]),
                }
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    y_pred = binary_preds_from_scores(y_score, threshold)
    metrics = compute_metrics(y_true, y_pred, "binary", y_score)
    correct_unc = []
    wrong_unc = []
    for row, pred, probs, uncertainty in zip(rows, y_pred, probs_all, uncertainty_all):
        row["pred_label"] = int(pred)
        row["confidence"] = float(probs[pred])
        if int(pred) == int(row["true_label"]):
            correct_unc.append(float(uncertainty))
        else:
            wrong_unc.append(float(uncertainty))
    metrics["threshold"] = float(threshold)
    metrics["fps"] = float(seen / elapsed) if elapsed > 0 else 0.0
    metrics["mean_uncertainty_correct"] = float(np.mean(correct_unc)) if correct_unc else 0.0
    metrics["mean_uncertainty_wrong"] = float(np.mean(wrong_unc)) if wrong_unc else 0.0
    metrics["uncertainty_gap_wrong_minus_correct"] = metrics["mean_uncertainty_wrong"] - metrics["mean_uncertainty_correct"]
    return metrics, rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained classifier on PH2 or PAD-UFES-20 external manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", default="results/experiments/evimetaskin/binary/effb0_gatedmeta_ev005")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--threshold", type=float, default=None, help="Override the ISIC validation-selected threshold.")
    parser.add_argument("--batch-size", type=int, default=16)
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
    model, train_args = build_model_from_checkpoint(ckpt, device)
    dataset = ExternalManifestDataset(manifest, task="binary", image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch)

    metrics, rows = evaluate(model, loader, device, threshold)
    metrics["n"] = int(len(dataset))
    metrics["manifest"] = str(manifest)
    metrics["checkpoint"] = str(ckpt_path)
    metrics["trained_run_name"] = str(train_args.get("run_name", run_dir.name))

    pd.DataFrame(rows).to_csv(output_dir / "external_predictions.csv", index=False, encoding="utf-8-sig")
    (output_dir / "external_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
