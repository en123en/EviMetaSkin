# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evimetaskin.datasets import DIAGNOSIS_TO_INDEX, METADATA_DIM, ISIC2018Task3Dataset, default_image_transform
from evimetaskin.metrics import (
    binary_preds_from_scores,
    compute_metrics,
    select_binary_threshold,
)
from evimetaskin.model import EviMetaSkin, evidential_classification_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_batch(batch):
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "binary_label": torch.tensor([item["binary_label"] for item in batch], dtype=torch.long),
        "multiclass_label": torch.tensor([item["multiclass_label"] for item in batch], dtype=torch.long),
        "prompt": [item["prompt"] for item in batch],
        "metadata": torch.tensor(np.stack([item["metadata"] for item in batch]), dtype=torch.float32),
        "isic_id": [item["isic_id"] for item in batch],
        "diagnosis": [item["diagnosis"] for item in batch],
    }


def maybe_disable_metadata(metadata, model):
    if getattr(model, "disable_metadata", False):
        return torch.zeros_like(metadata)
    return metadata


def get_labels(batch, task):
    return batch["binary_label"] if task == "binary" else batch["multiclass_label"]


def build_model(args, num_classes: int, device: torch.device):
    """Construct the paper model or one of its fusion ablations."""
    model = EviMetaSkin(
        visual_backbone=getattr(args, "visual_backbone", "efficientnet_b0"),
        metadata_dim=METADATA_DIM,
        num_classes=num_classes,
        freeze_image_encoder=getattr(args, "freeze_image_encoder", False),
        pretrained_visual=getattr(args, "pretrained_visual", False),
        fusion_mode=getattr(args, "fusion_mode", "gated"),
    ).to(device)
    model.uses_structured_metadata = True
    model.disable_metadata = getattr(args, "disable_metadata", False)
    return model


def resolve_prediction_mode(requested_mode: str, evidential_weight: float) -> str:
    """Choose the probability path used for validation, checkpointing, and testing.

    CE-only runs must be evaluated with softmax(logits). Evidential runs use the
    Dirichlet mean alpha / sum(alpha). Keeping this decision explicit prevents a
    CE-only checkpoint from being selected with an untrained evidence branch.
    """
    if requested_mode == "auto":
        return "evidential" if evidential_weight > 0.0 else "softmax"
    return requested_mode


def prediction_tensors(outputs, prediction_mode: str):
    """Return probabilities, uncertainty, and a human-readable uncertainty type."""
    if prediction_mode == "evidential":
        return outputs["evidential_probs"], outputs["uncertainty"], "evidential_k_over_s"
    if prediction_mode == "softmax":
        probs = F.softmax(outputs["logits"], dim=1)
        num_classes = probs.size(1)
        entropy = -(probs.clamp_min(1e-12) * probs.clamp_min(1e-12).log()).sum(dim=1)
        if num_classes > 1:
            entropy = entropy / float(np.log(num_classes))
        return probs, entropy, "normalized_predictive_entropy"
    raise ValueError(f"Unsupported prediction mode: {prediction_mode}")


def make_loader(dataset, batch_size, num_workers, train=False, balanced=False, task="binary", seed=42):
    sampler = None
    shuffle = train
    generator = torch.Generator()
    generator.manual_seed(seed)
    if train and balanced:
        label_col = "target" if task == "binary" else "diagnosis"
        labels = dataset.df[label_col].tolist()
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        weights = [1.0 / counts[label] for label in labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def run_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    task,
    epoch,
    train,
    ce_weight,
    evidential_weight,
    prediction_mode,
):
    model.train(train)
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    y_true, y_pred, y_score = [], [], []
    desc = "train" if train else "eval"
    for batch in tqdm(loader, desc=desc, leave=False):
        images = batch["image"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        metadata = maybe_disable_metadata(metadata, model)
        labels = get_labels(batch, task).to(device, non_blocking=True)
        prompts = batch["prompt"]
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images, metadata) if getattr(model, "uses_structured_metadata", False) else model(images, prompts)
                ce_loss = ce(outputs["logits"], labels)
                if evidential_weight > 0.0:
                    ev_loss = evidential_classification_loss(outputs["alpha"], labels, epoch=epoch)
                else:
                    ev_loss = ce_loss.new_zeros(())
                loss = ce_weight * ce_loss + evidential_weight * ev_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        probs, _, _ = prediction_tensors(outputs, prediction_mode)
        probs = probs.detach().float().cpu().numpy()
        preds = probs.argmax(axis=1).tolist()
        total_loss += float(loss.item()) * images.size(0)
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds)
        if task == "binary":
            y_score.extend(probs[:, 1].tolist())

    metrics = compute_metrics(y_true, y_pred, task, y_score if task == "binary" else None)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics


@torch.no_grad()
def collect_predictions(model, loader, device, task, prediction_mode):
    model.eval()
    y_true, y_score, probs_all, uncertainty_all, rows = [], [], [], [], []
    for batch in tqdm(loader, desc="predict", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        metadata = maybe_disable_metadata(metadata, model)
        labels = get_labels(batch, task)
        outputs = model(images, metadata) if getattr(model, "uses_structured_metadata", False) else model(images, batch["prompt"])
        probs, uncertainty, uncertainty_type = prediction_tensors(outputs, prediction_mode)
        probs = probs.detach().float().cpu().numpy()
        uncertainty = uncertainty.detach().float().cpu().numpy()
        y_true.extend(labels.numpy().tolist())
        probs_all.extend(probs.tolist())
        uncertainty_all.extend(uncertainty.tolist())
        if task == "binary":
            y_score.extend(probs[:, 1].tolist())
        for i, isic_id in enumerate(batch["isic_id"]):
            rows.append(
                {
                    "isic_id": isic_id,
                    "diagnosis": batch["diagnosis"][i],
                    "true_label": int(labels[i]),
                    "uncertainty": float(uncertainty[i]),
                    "uncertainty_type": uncertainty_type,
                }
            )
    return y_true, y_score, probs_all, uncertainty_all, rows


def evaluate_with_predictions(model, loader, device, task, prediction_mode, threshold=0.5):
    y_true, y_score, probs_all, uncertainty_all, rows = collect_predictions(
        model, loader, device, task, prediction_mode
    )
    if task == "binary":
        y_pred = binary_preds_from_scores(y_score, threshold)
        metrics = compute_metrics(y_true, y_pred, task, y_score)
    else:
        y_pred = np.asarray(probs_all).argmax(axis=1).tolist()
        metrics = compute_metrics(y_true, y_pred, task)

    correct_unc = []
    wrong_unc = []
    for row, pred, probs, uncertainty in zip(rows, y_pred, probs_all, uncertainty_all):
        row["pred_label"] = int(pred)
        row["confidence"] = float(probs[pred])
        if task == "binary":
            row["score_malignant"] = float(probs[1])
        is_correct = int(pred) == int(row["true_label"])
        (correct_unc if is_correct else wrong_unc).append(float(uncertainty))
    metrics["mean_uncertainty_correct"] = float(np.mean(correct_unc)) if correct_unc else 0.0
    metrics["mean_uncertainty_wrong"] = float(np.mean(wrong_unc)) if wrong_unc else 0.0
    metrics["uncertainty_gap_wrong_minus_correct"] = (
        metrics["mean_uncertainty_wrong"] - metrics["mean_uncertainty_correct"]
    )
    error_labels = (np.asarray(y_pred, dtype=int) != np.asarray(y_true, dtype=int)).astype(int)
    metrics["error_detection_auc"] = (
        float(roc_auc_score(error_labels, np.asarray(uncertainty_all, dtype=float)))
        if len(np.unique(error_labels)) > 1
        else None
    )
    metrics["prediction_mode"] = prediction_mode
    metrics["uncertainty_type"] = rows[0]["uncertainty_type"] if rows else ""
    return metrics, rows


@torch.no_grad()
def measure_fps(model, loader, device, warmup_batches=3, max_batches=30):
    model.eval()
    seen = 0
    elapsed = 0.0
    for idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        metadata = batch["metadata"].to(device, non_blocking=True)
        metadata = maybe_disable_metadata(metadata, model)
        prompts = batch["prompt"]
        if idx < warmup_batches:
            _ = model(images, metadata) if getattr(model, "uses_structured_metadata", False) else model(images, prompts)
            if device.type == "cuda":
                torch.cuda.synchronize()
            continue
        if idx >= warmup_batches + max_batches:
            break
        start = time.perf_counter()
        _ = model(images, metadata) if getattr(model, "uses_structured_metadata", False) else model(images, prompts)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        seen += images.size(0)
    return float(seen / elapsed) if elapsed > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train EviMetaSkin on ISIC2018 Task 3.")
    parser.add_argument("--data-root", default="data/ISIC2018 Task3")
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--output-dir", default="results/experiments/evimetaskin")
    parser.add_argument("--run-name", default="effb0_gatedmeta_ev005")
    parser.add_argument("--architecture", choices=["structured_metadata"], default="structured_metadata", help=argparse.SUPPRESS)
    parser.add_argument("--visual-backbone", choices=["efficientnet_b0", "efficientnet_b3", "resnet50"], default="efficientnet_b0")
    parser.add_argument("--fusion-mode", choices=["gated", "concat", "visual_only"], default="gated")
    parser.add_argument("--pretrained-visual", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--disable-metadata", action="store_true", help="Zero out structured metadata for ablation.")
    parser.add_argument("--freeze-image-encoder", action="store_true")
    parser.add_argument("--ce-weight", type=float, default=0.95)
    parser.add_argument("--evidential-weight", type=float, default=0.05)
    parser.add_argument(
        "--prediction-mode",
        choices=["auto", "evidential", "softmax"],
        default="auto",
        help="Probability path for validation/checkpointing/testing. Auto uses softmax for CE-only and evidential probabilities otherwise.",
    )
    parser.add_argument("--threshold-selection", choices=["fixed", "f1", "balanced_accuracy", "recall_constraint"], default="recall_constraint")
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--min-recall", type=float, default=0.85)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.ce_weight < 0.0 or args.evidential_weight < 0.0:
        parser.error("Loss weights must be non-negative.")
    if args.ce_weight == 0.0 and args.evidential_weight == 0.0:
        parser.error("At least one loss weight must be positive.")
    prediction_mode = resolve_prediction_mode(args.prediction_mode, args.evidential_weight)
    args.resolved_prediction_mode = prediction_mode
    if args.evidential_weight == 0.0 and prediction_mode == "evidential":
        parser.error("CE-only runs cannot use evidential predictions because the evidence branch is not supervised.")

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir) / args.task / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ISIC2018Task3Dataset(args.data_root, "train", args.task, default_image_transform(True, args.image_size))
    val_ds = ISIC2018Task3Dataset(args.data_root, "validation", args.task, default_image_transform(False, args.image_size))
    test_ds = ISIC2018Task3Dataset(args.data_root, "test", args.task, default_image_transform(False, args.image_size))
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, train=True, balanced=args.balanced_sampler, task=args.task, seed=args.seed)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, seed=args.seed)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers, seed=args.seed)

    num_classes = 2 if args.task == "binary" else len(DIAGNOSIS_TO_INDEX)
    model = build_model(args, num_classes, device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1 = -1.0
    best_path = out_dir / "best_model.pth"
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            args.task,
            epoch,
            True,
            args.ce_weight,
            args.evidential_weight,
            prediction_mode,
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            optimizer,
            scaler,
            device,
            args.task,
            epoch,
            False,
            args.ce_weight,
            args.evidential_weight,
            prediction_mode,
        )
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"model": model.state_dict(), "args": vars(args), "best_f1": best_f1}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    threshold = 0.5
    threshold_metrics = None
    if args.task == "binary":
        val_true, val_score, _, _, _ = collect_predictions(model, val_loader, device, args.task, prediction_mode)
        threshold, threshold_metrics = select_binary_threshold(
            val_true,
            val_score,
            args.threshold_selection,
            fixed_threshold=args.fixed_threshold,
            min_recall=args.min_recall,
        )
        (out_dir / "threshold_selection.json").write_text(
            json.dumps(
                {
                    "mode": args.threshold_selection,
                    "threshold": threshold,
                    "prediction_mode": prediction_mode,
                    "validation_metrics_at_threshold": threshold_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    test_metrics, pred_rows = evaluate_with_predictions(
        model, test_loader, device, args.task, prediction_mode, threshold
    )
    if args.task == "binary":
        test_metrics["threshold"] = float(threshold)
    test_metrics["fps"] = measure_fps(model, test_loader, device)
    test_metrics["best_val_f1"] = float(best_f1)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Final test metrics:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
