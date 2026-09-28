# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from evimetaskin.baselines import build_model
from evimetaskin.datasets import DIAGNOSIS_TO_INDEX, ISIC2018Task3Dataset, default_image_transform
from evimetaskin.metrics import binary_preds_from_scores, compute_metrics, select_binary_threshold


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_batch(batch):
    images = torch.stack([item["image"] for item in batch])
    return {
        "image": images,
        "binary_label": torch.tensor([item["binary_label"] for item in batch], dtype=torch.long),
        "multiclass_label": torch.tensor([item["multiclass_label"] for item in batch], dtype=torch.long),
        "isic_id": [item["isic_id"] for item in batch],
        "diagnosis": [item["diagnosis"] for item in batch],
        "prompt": [item["prompt"] for item in batch],
    }


def get_labels(batch, task: str):
    return batch["binary_label"] if task == "binary" else batch["multiclass_label"]


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


def run_epoch(model, loader, criterion, optimizer, scaler, device, task: str, train: bool):
    model.train(train)
    total_loss = 0.0
    y_true, y_pred, y_score = [], [], []
    iterator = tqdm(loader, desc="train" if train else "eval", leave=False)
    for batch in iterator:
        images = batch["image"].to(device, non_blocking=True)
        labels = get_labels(batch, task).to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        total_loss += float(loss.item()) * images.size(0)
        preds = logits.argmax(dim=1).detach().cpu().numpy().tolist()
        if task == "binary":
            scores = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()
            y_score.extend(scores)
        y_pred.extend(preds)
        y_true.extend(labels.detach().cpu().numpy().tolist())
    metrics = compute_metrics(y_true, y_pred, task, y_score if task == "binary" else None)
    metrics["loss"] = total_loss / max(1, len(loader.dataset))
    return metrics


@torch.no_grad()
def collect_predictions(model, loader, device, task: str):
    model.eval()
    y_true, y_score, probs_all, rows = [], [], [], []
    for batch in tqdm(loader, desc="predict", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        labels = get_labels(batch, task)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_true.extend(labels.numpy().tolist())
        probs_all.extend(probs.tolist())
        if task == "binary":
            y_score.extend(probs[:, 1].tolist())
        for i, isic_id in enumerate(batch["isic_id"]):
            rows.append(
                {
                    "isic_id": isic_id,
                    "diagnosis": batch["diagnosis"][i],
                    "true_label": int(labels[i]),
                    "probabilities": probs[i].tolist(),
                }
            )
    return y_true, y_score, probs_all, rows


def evaluate_with_predictions(model, loader, device, task: str, threshold: float = 0.5):
    y_true, y_score, probs_all, rows = collect_predictions(model, loader, device, task)
    if task == "binary":
        y_pred = binary_preds_from_scores(y_score, threshold)
        metrics = compute_metrics(y_true, y_pred, task, y_score)
    else:
        y_pred = np.asarray(probs_all).argmax(axis=1).tolist()
        metrics = compute_metrics(y_true, y_pred, task)
    for row, pred, probs in zip(rows, y_pred, probs_all):
        row["pred_label"] = int(pred)
        row["confidence"] = float(probs[pred])
        row.pop("probabilities", None)
    return metrics, rows


@torch.no_grad()
def measure_fps(model, loader, device, warmup_batches=5, max_batches=50):
    model.eval()
    seen = 0
    elapsed = 0.0
    for idx, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        if idx < warmup_batches:
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            continue
        if idx >= warmup_batches + max_batches:
            break
        start = time.perf_counter()
        _ = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        seen += images.size(0)
    return float(seen / elapsed) if elapsed > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train pure-vision baselines on ISIC2018 Task 3.")
    parser.add_argument("--data-root", default="data/ISIC2018 Task3")
    parser.add_argument("--model", choices=["resnet50", "efficientnet_b0", "efficientnet_b3", "vit_b_16"], required=True)
    parser.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--output-dir", default="results/experiments/baselines")
    parser.add_argument("--run-name", default="", help="Optional subfolder name to keep different training strategies separated.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--class-weights", action="store_true", help="Use inverse-frequency class weights in CrossEntropyLoss.")
    parser.add_argument("--threshold-selection", choices=["fixed", "f1", "balanced_accuracy", "recall_constraint"], default="f1")
    parser.add_argument("--fixed-threshold", type=float, default=0.5)
    parser.add_argument("--min-recall", type=float, default=0.85)
    parser.add_argument("--positive-class-weight", type=float, default=None, help="Binary-only manual malignant class weight.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir) / args.task / args.model
    if args.run_name:
        out_dir = out_dir / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ISIC2018Task3Dataset(args.data_root, "train", args.task, default_image_transform(True, args.image_size))
    val_ds = ISIC2018Task3Dataset(args.data_root, "validation", args.task, default_image_transform(False, args.image_size))
    test_ds = ISIC2018Task3Dataset(args.data_root, "test", args.task, default_image_transform(False, args.image_size))
    train_loader = make_loader(train_ds, args.batch_size, args.num_workers, train=True, balanced=args.balanced_sampler, task=args.task, seed=args.seed)
    val_loader = make_loader(val_ds, args.batch_size, args.num_workers, seed=args.seed)
    test_loader = make_loader(test_ds, args.batch_size, args.num_workers, seed=args.seed)

    model = build_model(args.model, train_ds.num_classes, args.pretrained).to(device)
    loss_weight = None
    if args.positive_class_weight is not None:
        if args.task != "binary":
            raise ValueError("--positive-class-weight is only supported for binary classification.")
        loss_weight = torch.tensor([1.0, float(args.positive_class_weight)], dtype=torch.float32, device=device)
    elif args.class_weights:
        if args.task == "binary":
            counts = train_ds.df["target"].value_counts().sort_index().to_numpy()
        else:
            value_counts = train_ds.df["diagnosis"].value_counts()
            counts = np.array(
                [int(value_counts.get(label, 0)) for label, _ in sorted(DIAGNOSIS_TO_INDEX.items(), key=lambda x: x[1])]
            )
        weights = 1.0 / np.maximum(counts, 1)
        weights = weights / weights.mean()
        loss_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_f1 = -1.0
    history = []
    best_path = out_dir / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, scaler, device, args.task, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, scaler, device, args.task, train=False)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save({"model": model.state_dict(), "args": vars(args), "best_f1": best_f1}, best_path)

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    decision_threshold = 0.5
    threshold_metrics = None
    if args.task == "binary":
        val_true, val_score, _, _ = collect_predictions(model, val_loader, device, args.task)
        decision_threshold, threshold_metrics = select_binary_threshold(
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
                    "threshold": decision_threshold,
                    "validation_metrics_at_threshold": threshold_metrics,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    test_metrics, pred_rows = evaluate_with_predictions(model, test_loader, device, args.task, decision_threshold)
    if args.task == "binary":
        test_metrics["threshold"] = float(decision_threshold)
    test_metrics["fps"] = measure_fps(model, test_loader, device)
    test_metrics["best_val_f1"] = float(best_f1)
    pd = __import__("pandas")
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Final test metrics:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
