from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


SERIES = [
    ("Treat-none", "treat_none", "#505050", 1.35),
    ("Treat-all", "treat_all", "#9A9A9A", 1.35),
    ("ResNet50", "ResNet50_net_benefit", "#D84A42", 1.85),
    ("EfficientNet-B0", "EfficientNet-B0_net_benefit", "#2D82BD", 1.85),
    ("EfficientNet-B3", "EfficientNet-B3_net_benefit", "#7E9D2B", 1.85),
    ("Ours", "Ours_net_benefit", "#665CC7", 2.25),
]

PANELS = [
    {
        "title": "(a) PAD-UFES-20 external validation",
        "csv": "results/analysis/dca/pad_ufes20/dca_curve.csv",
        "ylim": (-0.05, 0.47),
        "yticks": [-0.05, 0.00, 0.10, 0.20, 0.30, 0.40, 0.47],
    },
    {
        "title": "(b) ISIC2018 test",
        "csv": "results/analysis/dca/isic_test/dca_curve.csv",
        "ylim": (-0.05, 0.25),
        "yticks": [-0.05, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
    },
]

X_TICKS = [0.05, 0.20, 0.35, 0.50, 0.65, 0.80]


def read_curve(path: Path) -> dict[str, list[float]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return {
        key: [float(row[key]) for row in rows]
        for key in rows[0]
    }


def draw_figure(output_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.85), sharex=True)

    legend_handles = []
    legend_labels = []
    for panel_index, (ax, panel) in enumerate(zip(axes, PANELS)):
        data = read_curve(Path(panel["csv"]))
        thresholds = data["threshold"]

        for label, column, color, linewidth in SERIES:
            line, = ax.plot(
                thresholds,
                data[column],
                color=color,
                linewidth=linewidth,
                solid_capstyle="round",
                solid_joinstyle="round",
                label=label,
            )
            if not legend_handles:
                # Populated after the first series loop below.
                pass

        if ax is axes[0]:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

        ax.set_xlim(0.05, 0.80)
        ax.set_ylim(*panel["ylim"])
        ax.set_xticks(X_TICKS)
        ax.set_xticklabels([f"{value:.2f}" for value in X_TICKS])
        ax.set_yticks(panel["yticks"])
        ax.set_yticklabels([f"{value:.2f}" for value in panel["yticks"]])
        if panel_index == 0:
            ax.set_xlabel("Threshold probability", fontsize=15, labelpad=10)
            ax.set_ylabel("Net benefit", fontsize=15, labelpad=11)
        ax.set_title(panel["title"], fontsize=13, fontweight="bold", pad=11)
        ax.tick_params(axis="both", labelsize=10.5, width=0.8, length=4)
        ax.grid(True, color="#E7E7E7", linewidth=0.65, alpha=0.9)
        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#333333")
        ax.spines["bottom"].set_color("#333333")
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=6,
        frameon=False,
        fontsize=12,
        handlelength=2.7,
        columnspacing=1.8,
        handletextpad=0.65,
    )
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.82, wspace=0.23)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(output_base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine PAD-UFES-20 and ISIC2018 DCA curves into one figure.")
    parser.add_argument(
        "--output-base",
        default="results/figures/dca_combined_final",
    )
    args = parser.parse_args()
    output_base = Path(args.output_base)
    draw_figure(output_base)
    for suffix in (".svg", ".pdf", ".png"):
        print(output_base.with_suffix(suffix))


if __name__ == "__main__":
    main()
