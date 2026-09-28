from __future__ import annotations

import base64
import csv
import os
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(
    os.environ.get(
        "EVIMETASKIN_GRADCAM_ROOT",
        "results/experiments/evimetaskin/binary/"
        "effb0_structmeta_ev005_recall085/gradcampp",
    )
)
OUT_DIR = Path(os.environ.get("EVIMETASKIN_FIGURE_OUT", "results/figures"))
OUTPUT_STEM = "gradcampp_figure_8case_final"

DPI = 600
W, H = 3900, 2670
MARGIN_X = 240
RIGHT_RESERVE = 120
PANEL_GAP = 150
CELL_GAP = 36
CELL_SIZE = 520
GRID_W = 3 * CELL_SIZE + 2 * CELL_GAP
GRID_TOP = 118
ROW_LABEL_H = 72
ROW_GAP = 12

TEXT = "#181818"
MUTED = "#505050"
BORDER = "#D9D9D9"
IMAGE_COLUMNS = ("Original", "Grad-CAM++", "Overlay")

PANELS = (
    {
        "title": "ISIC2018",
        "cases": (
            {
                "letter": "a",
                "label": "Correct non-malignant",
                "uncertainty": 0.0682231188,
                "sample_id": "ISIC_0035628",
                "stem": "isic_test/02_correct_benign_low_unc_ISIC_0035628",
                "true_label": 0,
                "pred_label": 0,
            },
            {
                "letter": "b",
                "label": "Correct malignant",
                "uncertainty": 0.2198319137,
                "sample_id": "ISIC_0035772",
                "stem": "isic_test/04_correct_malignant_low_unc_ISIC_0035772",
                "true_label": 1,
                "pred_label": 1,
            },
            {
                "letter": "c",
                "label": "False positive",
                "uncertainty": 0.7504006624,
                "sample_id": "ISIC_0035641",
                "stem": "isic_test/10_high_uncertainty_ISIC_0035641",
                "true_label": 0,
                "pred_label": 1,
            },
            {
                "letter": "d",
                "label": "False negative",
                "uncertainty": 0.7423240542,
                "sample_id": "ISIC_0035048",
                "stem": "isic_test/08_wrong_high_unc_ISIC_0035048",
                "true_label": 1,
                "pred_label": 0,
            },
        ),
    },
    {
        "title": "PAD-UFES-20",
        "cases": (
            {
                "letter": "e",
                "label": "Correct non-malignant",
                "uncertainty": 0.0935169682,
                "sample_id": "PAT_1018_107_68",
                "stem": "pad_ufes20/01_correct_benign_low_unc_PAT_1018_107_68",
                "true_label": 0,
                "pred_label": 0,
            },
            {
                "letter": "f",
                "label": "Correct malignant",
                "uncertainty": 0.2203113139,
                "sample_id": "PAT_56_86_479",
                "stem": "pad_ufes20/04_correct_malignant_low_unc_PAT_56_86_479",
                "true_label": 1,
                "pred_label": 1,
            },
            {
                "letter": "g",
                "label": "False positive",
                "uncertainty": 0.7547227144,
                "sample_id": "PAT_1765_3337_365",
                "stem": "pad_ufes20/07_wrong_high_unc_PAT_1765_3337_365",
                "true_label": 0,
                "pred_label": 1,
            },
            {
                "letter": "h",
                "label": "False negative",
                "uncertainty": 0.7508774996,
                "sample_id": "PAT_513_965_496",
                "stem": "pad_ufes20/08_wrong_high_unc_PAT_513_965_496",
                "true_label": 1,
                "pred_label": 0,
            },
        ),
    },
)


def image_paths(case: dict) -> tuple[Path, Path, Path]:
    stem = ROOT / case["stem"]
    return tuple(
        Path(f"{stem}_{suffix}.png")
        for suffix in ("original", "heatmap", "overlay")
    )


def check_inputs() -> None:
    missing = [
        str(path)
        for panel in PANELS
        for case in panel["cases"]
        for path in image_paths(case)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing Grad-CAM++ source images:\n" + "\n".join(missing)
        )


def load_font(size: int, bold: bool = False):
    candidates = (
        Path("C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf"),
        Path(
            "/usr/share/fonts/truetype/msttcorefonts/"
            + ("Times_New_Roman_Bold.ttf" if bold else "Times_New_Roman.ttf")
        ),
        Path(
            "/usr/share/fonts/truetype/dejavu/"
            + ("DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf")
        ),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


GROUP_SIZE = 70
HEADER_SIZE = 56
LABEL_SIZE = 56
NOTE_SIZE = 50

GROUP_FONT = load_font(GROUP_SIZE, bold=True)
HEADER_FONT = load_font(HEADER_SIZE, bold=True)
LABEL_FONT = load_font(LABEL_SIZE, bold=True)
NOTE_FONT = load_font(NOTE_SIZE, bold=True)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def panel_left(panel_index: int) -> int:
    return MARGIN_X + panel_index * (GRID_W + PANEL_GAP)


def image_left(panel_index: int, column: int) -> int:
    return panel_left(panel_index) + column * (CELL_SIZE + CELL_GAP)


def row_block_top(row: int) -> int:
    return GRID_TOP + row * (ROW_LABEL_H + CELL_SIZE + ROW_GAP)


def row_top(row: int) -> int:
    return row_block_top(row) + ROW_LABEL_H


def centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: float,
    center_y: float,
    text: str,
    font,
    fill: str = TEXT,
) -> None:
    box = draw.textbbox((0, 0), text, font=font)
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.text(
        (center_x - width / 2, center_y - height / 2 - box[1]),
        text,
        font=font,
        fill=fill,
    )


def left_multiline_text(
    draw: ImageDraw.ImageDraw,
    x: int,
    center_y: float,
    lines: tuple[str, ...],
) -> None:
    line_gap = 12
    boxes = [draw.textbbox((0, 0), line, font=LABEL_FONT) for line in lines]
    heights = [box[3] - box[1] for box in boxes]
    total_height = sum(heights) + line_gap
    y = center_y - total_height / 2
    for line, box, height in zip(lines, boxes, heights):
        draw.text((x, y - box[1]), line, font=LABEL_FONT, fill=TEXT)
        y += height + line_gap


def right_multiline_text(
    draw: ImageDraw.ImageDraw,
    right_x: int,
    center_y: float,
    lines: tuple[str, ...],
) -> None:
    line_gap = 8
    boxes = [draw.textbbox((0, 0), line, font=LABEL_FONT) for line in lines]
    heights = [box[3] - box[1] for box in boxes]
    total_height = sum(heights) + line_gap * (len(lines) - 1)
    y = center_y - total_height / 2
    for line, box, height in zip(lines, boxes, heights):
        width = box[2] - box[0]
        draw.text(
            (right_x - width, y - box[1]),
            line,
            font=LABEL_FONT,
            fill=TEXT,
        )
        y += height + line_gap


def case_lines(case: dict) -> tuple[str, ...]:
    abbreviations = {
        "Correct non-malignant": "TN",
        "Correct malignant": "TP",
        "False positive": "FP",
        "False negative": "FN",
    }
    return (
        f"({case['letter']}) {abbreviations[case['label']]}, "
        f"u = {case['uncertainty']:.3f}",
    )


def prepare_image(path: Path, column: int) -> Image.Image:
    with Image.open(path) as source:
        image = ImageOps.fit(
            source.convert("RGB"),
            (CELL_SIZE, CELL_SIZE),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    if column == 1:
        image = ImageEnhance.Color(image).enhance(0.90)
    elif column == 2:
        image = ImageEnhance.Color(image).enhance(0.86)
    return image


ACTIVATION_STOPS = (
    (0.00, (35, 55, 145)),
    (0.25, (40, 145, 205)),
    (0.50, (80, 185, 135)),
    (0.73, (230, 195, 55)),
    (1.00, (195, 55, 45)),
)


def interpolate_activation(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    for index in range(len(ACTIVATION_STOPS) - 1):
        left_value, left_color = ACTIVATION_STOPS[index]
        right_value, right_color = ACTIVATION_STOPS[index + 1]
        if value <= right_value:
            ratio = (value - left_value) / (right_value - left_value)
            return tuple(
                round(left + ratio * (right - left))
                for left, right in zip(left_color, right_color)
            )
    return ACTIVATION_STOPS[-1][1]


def draw_png() -> Path:
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)

    for panel_index, panel in enumerate(PANELS):
        grid_left = image_left(panel_index, 0)
        grid_right = image_left(panel_index, 2) + CELL_SIZE
        centered_text(
            draw,
            (grid_left + grid_right) / 2,
            48,
            panel["title"],
            GROUP_FONT,
        )

        for row, case in enumerate(panel["cases"]):
            y0 = row_top(row)
            for column, path in enumerate(image_paths(case)):
                x0 = image_left(panel_index, column)
                cell = prepare_image(path, column)
                image.paste(cell, (x0, y0))
                draw.rectangle(
                    (x0, y0, x0 + CELL_SIZE - 1, y0 + CELL_SIZE - 1),
                    outline=BORDER,
                    width=2,
                )

            label = case_lines(case)[0]
            label_x = image_left(panel_index, 0) + 2
            label_y = row_block_top(row) + 5
            label_box = draw.textbbox((0, 0), label, font=LABEL_FONT)
            draw.text(
                (label_x, label_y - label_box[1]),
                label,
                font=LABEL_FONT,
                fill=TEXT,
            )

    grid_bottom = row_top(3) + CELL_SIZE
    for panel_index in range(len(PANELS)):
        for column, header in enumerate(IMAGE_COLUMNS):
            centered_text(
                draw,
                image_left(panel_index, column) + CELL_SIZE / 2,
                grid_bottom + 62,
                header,
                HEADER_FONT,
            )

    bar_width = 54
    bar_height = 430
    bar_left = W - 185
    bar_top = 1640
    for offset in range(bar_height):
        color = interpolate_activation(
            1.0 - offset / max(1, bar_height - 1)
        )
        draw.line(
            (bar_left, bar_top + offset, bar_left + bar_width, bar_top + offset),
            fill=color,
            width=1,
        )
    draw.rectangle(
        (bar_left, bar_top, bar_left + bar_width, bar_top + bar_height),
        outline=BORDER,
        width=2,
    )
    low_box = draw.textbbox((0, 0), "Low", font=NOTE_FONT)
    draw.text(
        (
            bar_left + bar_width / 2 - (low_box[2] - low_box[0]) / 2,
            bar_top + bar_height + 10,
        ),
        "Low",
        font=NOTE_FONT,
        fill=MUTED,
    )
    high_box = draw.textbbox((0, 0), "High", font=NOTE_FONT)
    draw.text(
        (bar_left + bar_width / 2 - (high_box[2] - high_box[0]) / 2, bar_top - 62),
        "High",
        font=NOTE_FONT,
        fill=MUTED,
    )

    output = OUT_DIR / f"{OUTPUT_STEM}.png"
    image.save(output, format="PNG", dpi=(DPI, DPI), compress_level=6)
    return output


def image_to_png_buffer(path: Path, column: int) -> BytesIO:
    buffer = BytesIO()
    prepare_image(path, column).save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def image_to_data_uri(path: Path, column: int) -> str:
    buffer = image_to_png_buffer(path, column)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/png;base64," + encoded


def svg_text(
    x: float,
    y: float,
    text: str,
    size: int,
    bold: bool = False,
    anchor: str = "start",
    color: str = TEXT,
) -> str:
    weight = "700" if bold else "400"
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="Times New Roman, Times, serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{escape(text)}</text>'
    )


def draw_svg() -> Path:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}">',
        "<defs>",
        '<linearGradient id="activation-gradient" x1="0%" y1="100%" x2="0%" y2="0%">',
        *[
            f'<stop offset="{position * 100:.0f}%" '
            f'stop-color="rgb({color[0]},{color[1]},{color[2]})"/>'
            for position, color in ACTIVATION_STOPS
        ],
        "</linearGradient>",
        "</defs>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    for panel_index, panel in enumerate(PANELS):
        grid_left = image_left(panel_index, 0)
        grid_right = image_left(panel_index, 2) + CELL_SIZE
        parts.append(
            svg_text(
                (grid_left + grid_right) / 2,
                82,
                panel["title"],
                GROUP_SIZE,
                bold=True,
                anchor="middle",
            )
        )

        for row, case in enumerate(panel["cases"]):
            y0 = row_top(row)
            for column, path in enumerate(image_paths(case)):
                x0 = image_left(panel_index, column)
                parts.append(
                    f'<image x="{x0}" y="{y0}" width="{CELL_SIZE}" '
                    f'height="{CELL_SIZE}" preserveAspectRatio="none" '
                    f'href="{image_to_data_uri(path, column)}"/>'
                )
                parts.append(
                    f'<rect x="{x0}" y="{y0}" width="{CELL_SIZE}" '
                    f'height="{CELL_SIZE}" fill="none" stroke="{BORDER}" '
                    'stroke-width="2"/>'
                )

            label_x = image_left(panel_index, 0) + 2
            label_y = row_block_top(row) + 55
            parts.append(
                svg_text(
                    label_x,
                    label_y,
                    case_lines(case)[0],
                    LABEL_SIZE,
                    bold=True,
                    color=TEXT,
                )
            )

    grid_bottom = row_top(3) + CELL_SIZE
    for panel_index in range(len(PANELS)):
        for column, header in enumerate(IMAGE_COLUMNS):
            parts.append(
                svg_text(
                    image_left(panel_index, column) + CELL_SIZE / 2,
                    grid_bottom + 76,
                    header,
                    HEADER_SIZE,
                    bold=True,
                    anchor="middle",
                )
            )

    bar_width = 54
    bar_height = 430
    bar_left = W - 185
    bar_top = 1640
    parts.extend(
        (
            f'<rect x="{bar_left:.2f}" y="{bar_top}" width="{bar_width}" '
            f'height="{bar_height}" fill="url(#activation-gradient)" '
            f'stroke="{BORDER}" stroke-width="2"/>',
            svg_text(
                bar_left + bar_width / 2,
                bar_top + bar_height + 60,
                "Low",
                NOTE_SIZE,
                anchor="middle",
                color=MUTED,
            ),
            svg_text(
                bar_left + bar_width / 2,
                bar_top - 22,
                "High",
                NOTE_SIZE,
                anchor="middle",
                color=MUTED,
            ),
        )
    )
    parts.append("</svg>")

    output = OUT_DIR / f"{OUTPUT_STEM}.svg"
    output.write_text("\n".join(parts), encoding="utf-8")
    return output


PDF_W = 14.2 * 72
PDF_H = PDF_W * H / W


def register_pdf_fonts() -> tuple[str, str]:
    regular = Path("C:/Windows/Fonts/times.ttf")
    bold = Path("C:/Windows/Fonts/timesbd.ttf")
    if regular.exists() and bold.exists():
        pdfmetrics.registerFont(TTFont("FigureTimes", str(regular)))
        pdfmetrics.registerFont(TTFont("FigureTimesBold", str(bold)))
        return "FigureTimes", "FigureTimesBold"
    return "Times-Roman", "Times-Bold"


def px_x(value: float) -> float:
    return value * PDF_W / W


def px_y(value: float) -> float:
    return PDF_H - value * PDF_H / H


def set_pdf_color(pdf: canvas.Canvas, color: str, stroke: bool = False) -> None:
    red, green, blue = hex_to_rgb(color)
    method = pdf.setStrokeColorRGB if stroke else pdf.setFillColorRGB
    method(red / 255, green / 255, blue / 255)


def draw_pdf() -> Path:
    regular_font, bold_font = register_pdf_fonts()
    output = OUT_DIR / f"{OUTPUT_STEM}.pdf"
    pdf = canvas.Canvas(str(output), pagesize=(PDF_W, PDF_H))

    for panel_index, panel in enumerate(PANELS):
        grid_left = image_left(panel_index, 0)
        grid_right = image_left(panel_index, 2) + CELL_SIZE
        set_pdf_color(pdf, TEXT)
        pdf.setFont(bold_font, px_x(GROUP_SIZE))
        pdf.drawCentredString(
            px_x((grid_left + grid_right) / 2),
            px_y(82),
            panel["title"],
        )

        for row, case in enumerate(panel["cases"]):
            y0 = row_top(row)
            for column, path in enumerate(image_paths(case)):
                x0 = image_left(panel_index, column)
                image_buffer = image_to_png_buffer(path, column)
                pdf.drawImage(
                    ImageReader(image_buffer),
                    px_x(x0),
                    px_y(y0 + CELL_SIZE),
                    width=px_x(CELL_SIZE),
                    height=px_x(CELL_SIZE),
                    preserveAspectRatio=False,
                    mask=None,
                )
                set_pdf_color(pdf, BORDER, stroke=True)
                pdf.setLineWidth(px_x(2))
                pdf.rect(
                    px_x(x0),
                    px_y(y0 + CELL_SIZE),
                    px_x(CELL_SIZE),
                    px_x(CELL_SIZE),
                    fill=0,
                    stroke=1,
                )

            label_x = image_left(panel_index, 0) + 2
            label_y = row_block_top(row) + 55
            set_pdf_color(pdf, TEXT)
            pdf.setFont(bold_font, px_x(LABEL_SIZE))
            pdf.drawString(
                px_x(label_x),
                px_y(label_y),
                case_lines(case)[0],
            )

    grid_bottom = row_top(3) + CELL_SIZE
    for panel_index in range(len(PANELS)):
        for column, header in enumerate(IMAGE_COLUMNS):
            set_pdf_color(pdf, TEXT)
            pdf.setFont(bold_font, px_x(HEADER_SIZE))
            pdf.drawCentredString(
                px_x(image_left(panel_index, column) + CELL_SIZE / 2),
                px_y(grid_bottom + 76),
                header,
            )

    bar_width = 54
    bar_height = 430
    bar_left = W - 185
    bar_top = 1640
    steps = 190
    segment_height = bar_height / steps
    for step in range(steps):
        red, green, blue = interpolate_activation(
            1.0 - step / max(1, steps - 1)
        )
        pdf.setFillColorRGB(red / 255, green / 255, blue / 255)
        pdf.rect(
            px_x(bar_left),
            px_y(bar_top + (step + 1) * segment_height),
            px_x(bar_width),
            px_x(segment_height + 0.6),
            fill=1,
            stroke=0,
        )
    set_pdf_color(pdf, BORDER, stroke=True)
    pdf.setLineWidth(px_x(2))
    pdf.rect(
        px_x(bar_left),
        px_y(bar_top + bar_height),
        px_x(bar_width),
        px_x(bar_height),
        fill=0,
        stroke=1,
    )
    set_pdf_color(pdf, MUTED)
    pdf.setFont(bold_font, px_x(NOTE_SIZE))
    pdf.drawCentredString(
        px_x(bar_left + bar_width / 2),
        px_y(bar_top + bar_height + 60),
        "Low",
    )
    pdf.drawCentredString(
        px_x(bar_left + bar_width / 2),
        px_y(bar_top - 22),
        "High",
    )
    pdf.showPage()
    pdf.save()
    return output


def save_selection_manifest(path: Path) -> None:
    fieldnames = (
        "panel",
        "case",
        "sample_id",
        "condition",
        "uncertainty",
        "true_label",
        "pred_label",
        "original",
        "heatmap",
        "overlay",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for panel in PANELS:
            for case in panel["cases"]:
                paths = image_paths(case)
                writer.writerow(
                    {
                        "panel": panel["title"],
                        "case": case["letter"],
                        "sample_id": case["sample_id"],
                        "condition": case["label"],
                        "uncertainty": f"{case['uncertainty']:.10f}",
                        "true_label": case["true_label"],
                        "pred_label": case["pred_label"],
                        "original": paths[0],
                        "heatmap": paths[1],
                        "overlay": paths[2],
                    }
                )


def main() -> None:
    check_inputs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = OUT_DIR / f"{OUTPUT_STEM}_samples.csv"
    save_selection_manifest(manifest)
    outputs = (draw_svg(), draw_pdf(), draw_png(), manifest)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
