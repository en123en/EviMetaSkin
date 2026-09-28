# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

CURRENT_DIR = Path(__file__).resolve().parent
CODE_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from evimetaskin.datasets import (
    DIAGNOSIS_TO_INDEX,
    METADATA_DIM,
    build_clinical_prompt,
    build_structured_metadata,
    default_image_transform,
)
from evimetaskin.model import EviMetaSkin


class GradCamDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path | None,
        split: str,
        manifest: str | Path | None,
        image_size: int = 224,
    ):
        self.transform = default_image_transform(False, image_size)
        if manifest:
            self.df = pd.read_csv(manifest)
            required = {"image_path", "target", "diagnosis", "isic_id"}
            missing = required.difference(self.df.columns)
            if missing:
                raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
            self.df["image_path"] = self.df["image_path"].map(lambda p: str(Path(str(p))))
        else:
            if not data_root:
                raise ValueError("Either --data-root or --manifest must be provided.")
            data_root = Path(data_root)
            metadata_path = data_root / f"{split}-metadata.csv"
            image_dir = data_root / f"{split}-image" / "image"
            self.df = pd.read_csv(metadata_path)
            self.df["image_path"] = self.df["isic_id"].map(lambda x: str(image_dir / f"{x}.jpg"))
        self.df = self.df[self.df["image_path"].map(lambda p: Path(str(p)).exists())].reset_index(drop=True)
        if self.df.empty:
            raise RuntimeError("No matched images found for Grad-CAM++ generation.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]
        image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image)
        diagnosis = str(row["diagnosis"])
        return {
            "image": tensor,
            "metadata": torch.tensor(build_structured_metadata(row), dtype=torch.float32),
            "prompt": build_clinical_prompt(row, task="binary"),
            "true_label": int(row["target"]),
            "multiclass_label": int(DIAGNOSIS_TO_INDEX.get(diagnosis, -1)),
            "isic_id": str(row["isic_id"]),
            "diagnosis": diagnosis,
            "image_path": str(image_path),
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


def get_target_layer(model):
    if not hasattr(model, "visual_encoder"):
        raise ValueError("Grad-CAM++ script currently expects a model with visual_encoder.")
    if hasattr(model.visual_encoder, "features"):
        return model.visual_encoder.features[-1]
    if hasattr(model.visual_encoder, "layer4"):
        return model.visual_encoder.layer4[-1]
    raise ValueError("Unsupported visual encoder for target layer selection.")


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def _forward_hook(self, _module, _inputs, output):
        self.activations = output

    def _backward_hook(self, _module, _grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, image, metadata, prompt, target_class=None):
        self.model.zero_grad(set_to_none=True)
        if getattr(self.model, "uses_structured_metadata", False):
            outputs = self.model(image, metadata)
        else:
            outputs = self.model(image, [prompt])
        probs = outputs["evidential_probs"]
        logits = outputs["logits"]
        if target_class is None:
            target_class = int(probs.argmax(dim=1).item())
        score = logits[:, target_class].sum()
        score.backward(retain_graph=False)
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM++ hooks did not capture activations/gradients.")

        activations = self.activations.detach()
        gradients = self.gradients.detach()
        grad_2 = gradients.pow(2)
        grad_3 = gradients.pow(3)
        spatial_sum = (activations * grad_3).sum(dim=(2, 3), keepdim=True)
        alpha = grad_2 / (2.0 * grad_2 + spatial_sum + 1e-8)
        weights = (alpha * F.relu(gradients)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), outputs


def jet_colormap(gray: np.ndarray) -> np.ndarray:
    gray = np.clip(gray, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * gray - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * gray - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * gray - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def make_overlay(image_path: str | Path, cam: np.ndarray, image_size: int, alpha: float = 0.45):
    original = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    original_np = np.asarray(original).astype(np.float32) / 255.0
    heat_np = jet_colormap(cam)
    overlay_np = (1.0 - alpha) * original_np + alpha * heat_np
    heat = Image.fromarray((heat_np * 255).astype(np.uint8))
    overlay = Image.fromarray((np.clip(overlay_np, 0, 1) * 255).astype(np.uint8))
    return original, heat, overlay


def add_label(image: Image.Image, label: str):
    canvas = Image.new("RGB", (image.width, image.height + 28), "white")
    canvas.paste(image, (0, 28))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except Exception:
        font = ImageFont.load_default()
    draw.text((4, 7), label, fill=(0, 0, 0), font=font)
    return canvas


def save_triplet(original, heat, overlay, out_path: Path, title: str):
    panels = [
        add_label(original, "Original"),
        add_label(heat, "Grad-CAM++"),
        add_label(overlay, "Overlay"),
    ]
    canvas = Image.new("RGB", (sum(p.width for p in panels), panels[0].height), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width
    draw = ImageDraw.Draw(canvas)
    draw.text((6, canvas.height - 18), title[:120], fill=(0, 0, 0))
    canvas.save(out_path)


def select_indices(dataset: GradCamDataset, predictions_path: str | Path | None, max_per_group: int):
    if predictions_path is None:
        return [(i, "sample") for i in range(min(max_per_group, len(dataset)))]

    pred = pd.read_csv(predictions_path)
    if "isic_id" not in pred.columns:
        raise ValueError("Prediction CSV must contain isic_id for sample selection.")
    pred = pred.copy()
    pred["is_correct"] = pred["true_label"].astype(int) == pred["pred_label"].astype(int)
    if "uncertainty" not in pred.columns:
        pred["uncertainty"] = 0.0
    id_to_index = {str(row["isic_id"]): i for i, row in dataset.df.iterrows()}

    groups = {
        "correct_benign_low_unc": pred[(pred["is_correct"]) & (pred["true_label"].astype(int) == 0)].sort_values("uncertainty", ascending=True),
        "correct_malignant_low_unc": pred[(pred["is_correct"]) & (pred["true_label"].astype(int) == 1)].sort_values("uncertainty", ascending=True),
        "wrong_high_unc": pred[~pred["is_correct"]].sort_values("uncertainty", ascending=False),
        "high_uncertainty": pred.sort_values("uncertainty", ascending=False),
    }
    selected = []
    seen = set()
    for group_name, frame in groups.items():
        count = 0
        for _, row in frame.iterrows():
            isic_id = str(row["isic_id"])
            if isic_id in seen or isic_id not in id_to_index:
                continue
            selected.append((id_to_index[isic_id], group_name))
            seen.add(isic_id)
            count += 1
            if count >= max_per_group:
                break
    return selected


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM++ visualizations for the trained multimodal classifier.")
    parser.add_argument("--run-dir", default="results/experiments/evimetaskin/binary/effb0_gatedmeta_ev005")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data-root", default="data/ISIC2018 Task3")
    parser.add_argument("--split", default="test")
    parser.add_argument("--manifest", default="", help="External validation manifest, e.g. PAD-UFES-20.")
    parser.add_argument("--predictions", default="", help="Prediction CSV used to select representative cases.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-per-group", type=int, default=3)
    parser.add_argument("--target-class", type=int, default=-1, help="Use -1 for predicted class, 1 for malignant class.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else run_dir / "best_model.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    dataset_name = "external" if args.manifest else args.split
    out_dir = Path(args.output_dir) if args.output_dir else run_dir / "gradcampp" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model, _train_args = build_model_from_checkpoint(ckpt, device)
    model.eval()
    target_layer = get_target_layer(model)
    cam_engine = GradCAMPlusPlus(model, target_layer)
    dataset = GradCamDataset(args.data_root, args.split, args.manifest or None, args.image_size)
    selected = select_indices(dataset, args.predictions or None, args.max_per_group)
    records = []
    try:
        for rank, (index, group_name) in enumerate(selected, start=1):
            item = dataset[index]
            image = item["image"].unsqueeze(0).to(device)
            metadata = item["metadata"].unsqueeze(0).to(device)
            if getattr(model, "disable_metadata", False):
                metadata = torch.zeros_like(metadata)
            target_class = None if args.target_class < 0 else int(args.target_class)
            cam, outputs = cam_engine(image, metadata, item["prompt"], target_class=target_class)
            probs = outputs["evidential_probs"].detach().float().cpu().numpy()[0]
            uncertainty = float(outputs["uncertainty"].detach().float().cpu().numpy()[0])
            pred_label = int(probs.argmax())
            true_label = int(item["true_label"])
            original, heat, overlay = make_overlay(item["image_path"], cam, args.image_size)
            stem = f"{rank:02d}_{group_name}_{item['isic_id']}"
            original.save(out_dir / f"{stem}_original.png")
            heat.save(out_dir / f"{stem}_heatmap.png")
            overlay.save(out_dir / f"{stem}_overlay.png")
            title = (
                f"{group_name} | id={item['isic_id']} | true={true_label} pred={pred_label} "
                f"p_mal={probs[1]:.3f} unc={uncertainty:.3f}"
            )
            save_triplet(original, heat, overlay, out_dir / f"{stem}_triplet.png", title)
            records.append(
                {
                    "rank": rank,
                    "group": group_name,
                    "isic_id": item["isic_id"],
                    "image_path": item["image_path"],
                    "diagnosis": item["diagnosis"],
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "prob_benign": float(probs[0]),
                    "prob_malignant": float(probs[1]),
                    "uncertainty": uncertainty,
                    "triplet_path": str(out_dir / f"{stem}_triplet.png"),
                }
            )
    finally:
        cam_engine.close()

    pd.DataFrame(records).to_csv(out_dir / "gradcampp_index.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(records)} Grad-CAM++ cases to: {out_dir}")


if __name__ == "__main__":
    main()
