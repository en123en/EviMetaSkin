# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import brier_score_loss, log_loss


def load_font(size: int, bold: bool = False):
    candidates = [
        "arialbd.ttf" if bold else "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except Exception:
            pass
    return ImageFont.load_default()


def get_scores(df: pd.DataFrame):
    if "score_malignant" in df.columns:
        scores = df["score_malignant"].astype(float).to_numpy()
    elif "prob_malignant" in df.columns:
        scores = df["prob_malignant"].astype(float).to_numpy()
    elif "confidence" in df.columns and "pred_label" in df.columns:
        conf = df["confidence"].astype(float).to_numpy()
        pred = df["pred_label"].astype(int).to_numpy()
        scores = np.where(pred == 1, conf, 1.0 - conf)
    else:
        raise ValueError("Prediction CSV must contain score_malignant, prob_malignant, or confidence+pred_label.")
    return np.clip(scores, 1e-7, 1.0 - 1e-7)


def compute_calibration(y_true, scores, n_bins: int):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0
    mce = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        count = int(mask.sum())
        if count == 0:
            avg_conf = 0.0
            frac_pos = 0.0
            gap = 0.0
        else:
            avg_conf = float(scores[mask].mean())
            frac_pos = float(y_true[mask].mean())
            gap = abs(avg_conf - frac_pos)
            ece += (count / n) * gap
            mce = max(mce, gap)
        rows.append(
            {
                "bin": i + 1,
                "lower": float(lo),
                "upper": float(hi),
                "n": count,
                "avg_confidence": avg_conf,
                "fraction_positive": frac_pos,
                "abs_gap": float(gap),
            }
        )
    return rows, float(ece), float(mce)


def draw_reliability_diagram(rows, metrics, output_path: Path, title: str):
    width, height = 760, 620
    margin_l, margin_r, margin_t, margin_b = 86, 36, 72, 88
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(15)
    small = load_font(12)
    bold = load_font(18, bold=True)

    draw.text((margin_l, 22), title, fill=(0, 0, 0), font=bold)
    subtitle = f"ECE={metrics['ece']:.4f}, MCE={metrics['mce']:.4f}, Brier={metrics['brier']:.4f}, NLL={metrics['nll']:.4f}"
    draw.text((margin_l, 48), subtitle, fill=(60, 60, 60), font=small)

    x0, y0 = margin_l, margin_t + plot_h
    x1, y1 = margin_l + plot_w, margin_t
    draw.rectangle([x0, y1, x1, y0], outline=(0, 0, 0), width=1)
    for t in np.linspace(0, 1, 6):
        x = x0 + int(t * plot_w)
        y = y0 - int(t * plot_h)
        draw.line([x, y0, x, y0 + 5], fill=(0, 0, 0))
        draw.line([x0 - 5, y, x0, y], fill=(0, 0, 0))
        draw.text((x - 10, y0 + 10), f"{t:.1f}", fill=(0, 0, 0), font=small)
        draw.text((x0 - 38, y - 7), f"{t:.1f}", fill=(0, 0, 0), font=small)
        if t not in (0, 1):
            draw.line([x0, y, x1, y], fill=(230, 230, 230))
            draw.line([x, y0, x, y1], fill=(240, 240, 240))

    draw.line([x0, y0, x1, y1], fill=(120, 120, 120), width=2)
    n_bins = len(rows)
    bar_w = plot_w / n_bins * 0.74
    for row in rows:
        center = x0 + ((row["lower"] + row["upper"]) / 2) * plot_w
        bar_x0 = int(center - bar_w / 2)
        bar_x1 = int(center + bar_w / 2)
        bar_y = y0 - int(row["fraction_positive"] * plot_h)
        draw.rectangle([bar_x0, bar_y, bar_x1, y0], fill=(74, 144, 226), outline=(48, 108, 176))
        conf_y = y0 - int(row["avg_confidence"] * plot_h)
        draw.line([bar_x0, conf_y, bar_x1, conf_y], fill=(220, 60, 60), width=2)

    draw.text((x0 + plot_w // 2 - 70, height - 42), "Mean predicted probability", fill=(0, 0, 0), font=font)
    draw.text((16, y1 + plot_h // 2 - 10), "Observed", fill=(0, 0, 0), font=font)
    draw.text((width - 255, height - 65), "Blue: observed fraction; red line: confidence", fill=(80, 80, 80), font=small)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def main():
    parser = argparse.ArgumentParser(description="Compute calibration metrics and draw reliability diagrams.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()

    pred_path = Path(args.predictions)
    output_dir = Path(args.output_dir) if args.output_dir else pred_path.parent / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(pred_path)
    if "true_label" not in df.columns:
        raise ValueError("Prediction CSV must contain true_label.")
    y_true = df["true_label"].astype(int).to_numpy()
    scores = get_scores(df)
    rows, ece, mce = compute_calibration(y_true, scores, args.n_bins)
    metrics = {
        "n": int(len(y_true)),
        "ece": ece,
        "mce": mce,
        "brier": float(brier_score_loss(y_true, scores)),
        "nll": float(log_loss(y_true, np.vstack([1.0 - scores, scores]).T, labels=[0, 1])),
        "mean_score": float(scores.mean()),
        "positive_rate": float(y_true.mean()),
        "predictions": str(pred_path),
    }
    pd.DataFrame(rows).to_csv(output_dir / "calibration_bins.csv", index=False, encoding="utf-8-sig")
    (output_dir / "calibration_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    title = "Reliability diagram"
    if args.model_name or args.dataset_name:
        title = f"{args.model_name} on {args.dataset_name}".strip()
    draw_reliability_diagram(rows, metrics, output_dir / "reliability_diagram.png", title)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
