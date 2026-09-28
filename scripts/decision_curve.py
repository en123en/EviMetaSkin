# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


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


def net_benefit(y_true, scores, threshold: float):
    pred = scores >= threshold
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    n = len(y_true)
    odds = threshold / (1.0 - threshold)
    return tp / n - fp / n * odds, tp, fp


def treat_all_net_benefit(y_true, threshold: float):
    prevalence = float(np.mean(y_true))
    odds = threshold / (1.0 - threshold)
    return prevalence - (1.0 - prevalence) * odds


def compute_dca(curves, thresholds):
    rows = []
    first_y = next(iter(curves.values()))["y_true"]
    for threshold in thresholds:
        row = {
            "threshold": float(threshold),
            "treat_none": 0.0,
            "treat_all": float(treat_all_net_benefit(first_y, threshold)),
        }
        for name, payload in curves.items():
            nb, tp, fp = net_benefit(payload["y_true"], payload["scores"], threshold)
            row[f"{name}_net_benefit"] = float(nb)
            row[f"{name}_tp"] = int(tp)
            row[f"{name}_fp"] = int(fp)
        rows.append(row)
    return pd.DataFrame(rows)


def draw_dca(df: pd.DataFrame, model_names: list[str], output_path: Path, title: str):
    width, height = 900, 640
    margin_l, margin_r, margin_t, margin_b = 86, 170, 72, 82
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = load_font(14)
    small = load_font(12)
    bold = load_font(18, bold=True)

    x_values = df["threshold"].to_numpy()
    y_cols = ["treat_none", "treat_all"] + [f"{name}_net_benefit" for name in model_names]
    y_values = np.concatenate([df[col].to_numpy() for col in y_cols])
    y_min = min(-0.05, float(np.nanmin(y_values)) - 0.02)
    y_max = max(0.25, float(np.nanmax(y_values)) + 0.02)

    x0, y0 = margin_l, margin_t + plot_h
    x1, y1 = margin_l + plot_w, margin_t
    draw.text((margin_l, 24), title, fill=(0, 0, 0), font=bold)
    draw.rectangle([x0, y1, x1, y0], outline=(0, 0, 0), width=1)

    def map_x(x):
        return x0 + int((x - x_values.min()) / (x_values.max() - x_values.min()) * plot_w)

    def map_y(y):
        return y0 - int((y - y_min) / (y_max - y_min) * plot_h)

    for t in np.linspace(x_values.min(), x_values.max(), 6):
        x = map_x(t)
        draw.line([x, y0, x, y0 + 5], fill=(0, 0, 0))
        draw.text((x - 10, y0 + 10), f"{t:.2f}", fill=(0, 0, 0), font=small)
        draw.line([x, y0, x, y1], fill=(240, 240, 240))
    for y in np.linspace(y_min, y_max, 6):
        yy = map_y(y)
        draw.line([x0 - 5, yy, x0, yy], fill=(0, 0, 0))
        draw.text((x0 - 72, yy - 7), f"{y:.2f}", fill=(0, 0, 0), font=small)
        draw.line([x0, yy, x1, yy], fill=(235, 235, 235))

    colors = {
        "treat_none": (80, 80, 80),
        "treat_all": (150, 150, 150),
    }
    palette = [(220, 50, 47), (38, 139, 210), (133, 153, 0), (108, 113, 196), (203, 75, 22)]
    series = [("Treat-none", "treat_none", colors["treat_none"]), ("Treat-all", "treat_all", colors["treat_all"])]
    for idx, name in enumerate(model_names):
        series.append((name, f"{name}_net_benefit", palette[idx % len(palette)]))

    for label, col, color in series:
        pts = [(map_x(float(x)), map_y(float(y))) for x, y in zip(df["threshold"], df[col])]
        width_line = 3 if label not in {"Treat-none", "Treat-all"} else 2
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=width_line)

    legend_x = x1 + 22
    legend_y = y1 + 14
    for label, _col, color in series:
        draw.line([legend_x, legend_y + 8, legend_x + 28, legend_y + 8], fill=color, width=3)
        draw.text((legend_x + 36, legend_y), label, fill=(0, 0, 0), font=font)
        legend_y += 28

    draw.text((x0 + plot_w // 2 - 65, height - 40), "Threshold probability", fill=(0, 0, 0), font=font)
    draw.text((14, y1 + plot_h // 2 - 8), "Net benefit", fill=(0, 0, 0), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)


def parse_model_arg(value: str):
    if "::" not in value:
        raise ValueError("--model entries must use Name::path/to/predictions.csv")
    name, path = value.split("::", 1)
    return name.strip(), Path(path)


def main():
    parser = argparse.ArgumentParser(description="Decision curve analysis for binary skin lesion classification.")
    parser.add_argument("--model", action="append", required=True, help="Use Name::predictions.csv. Can be repeated.")
    parser.add_argument("--output-dir", default="results/analysis/dca")
    parser.add_argument("--threshold-min", type=float, default=0.05)
    parser.add_argument("--threshold-max", type=float, default=0.80)
    parser.add_argument("--num-thresholds", type=int, default=76)
    parser.add_argument("--title", default="Decision curve analysis")
    args = parser.parse_args()

    curves = {}
    n_ref = None
    y_ref = None
    for value in args.model:
        name, path = parse_model_arg(value)
        df = pd.read_csv(path)
        if "true_label" not in df.columns:
            raise ValueError(f"{path} must contain true_label.")
        y_true = df["true_label"].astype(int).to_numpy()
        scores = get_scores(df)
        if n_ref is None:
            n_ref = len(y_true)
            y_ref = y_true
        elif len(y_true) != n_ref or not np.array_equal(y_true, y_ref):
            raise ValueError("All prediction files must use the same samples in the same order for fair DCA.")
        curves[name] = {"y_true": y_true, "scores": scores, "path": str(path)}

    thresholds = np.linspace(args.threshold_min, args.threshold_max, args.num_thresholds)
    dca_df = compute_dca(curves, thresholds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dca_df.to_csv(output_dir / "dca_curve.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n": int(n_ref),
        "prevalence": float(np.mean(y_ref)),
        "threshold_min": float(args.threshold_min),
        "threshold_max": float(args.threshold_max),
        "models": {name: payload["path"] for name, payload in curves.items()},
        "mean_net_benefit": {
            name: float(dca_df[f"{name}_net_benefit"].mean()) for name in curves
        },
    }
    (output_dir / "dca_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_dca(dca_df, list(curves.keys()), output_dir / "dca_curve.png", args.title)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
