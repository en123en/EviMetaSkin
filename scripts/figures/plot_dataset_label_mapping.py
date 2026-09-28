from pathlib import Path
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfgen import canvas


OUT_DIR = Path("results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 4000, 1480
PDF_W, PDF_H = 7.2 * 72, 2.664 * 72
BG = "white"
TEXT = "#222222"
GRID = "#E4E4E4"
AXIS = "#4A4A4A"
NON_MALIGNANT = "#4E79A7"
MALIGNANT = "#B85650"
NON_MALIGNANT_TEXT = "#244D76"
MALIGNANT_TEXT = "#8C302C"


def font(size, bold=False):
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/timesbd.ttf" if bold else "C:/Windows/Fonts/times.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


F_TITLE = font(72, True)
F_AXIS = font(62, False)
F_TICK = font(54, False)
F_LABEL = font(56, False)
F_COUNT = font(52, False)
F_BOX = font(48, True)
F_CAPTION = font(54, False)
F_LEGEND = font(68, True)

LEGEND_SWATCH_W = 90
LEGEND_SWATCH_H = 54
LEGEND_TEXT_GAP = 24
LEGEND_Y = 130
LEGEND_X_NON_MALIGNANT = 90
LEGEND_X_MALIGNANT = 1330


def text_size(draw, text, fnt):
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]


def draw_centered(draw, xy, text, fnt, fill=TEXT):
    x, y = xy
    tw, th = text_size(draw, text, fnt)
    draw.text((x - tw / 2, y - th / 2), text, font=fnt, fill=fill)


def nice_step(max_value):
    if max_value <= 1000:
        return 200
    if max_value <= 2000:
        return 500
    return 1000


def panel_geometry(box):
    left, top, right, bottom = box
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "plot_left": left + 220,
        "plot_right": right - 45,
        "plot_top": top + 170,
        "plot_bottom": bottom - 175,
    }


def px_to_pdf(x, y=None):
    sx = PDF_W / W
    sy = PDF_H / H
    if y is None:
        return x * sx
    return x * sx, PDF_H - y * sy


def draw_rotated_label(img, center, text):
    pad = 20
    tmp = Image.new("RGBA", (520, 90), (255, 255, 255, 0))
    d = ImageDraw.Draw(tmp)
    tw, th = text_size(d, text, F_AXIS)
    d.text(((520 - tw) / 2, (90 - th) / 2), text, font=F_AXIS, fill=TEXT)
    tmp = tmp.rotate(90, expand=True)
    img.alpha_composite(tmp, (int(center[0] - tmp.width / 2 - pad), int(center[1] - tmp.height / 2)))


def draw_panel(img, draw, box, title, items, totals, y_max):
    g = panel_geometry(box)
    left = g["left"]
    top = g["top"]
    right = g["right"]
    bottom = g["bottom"]
    plot_left = g["plot_left"]
    plot_right = g["plot_right"]
    plot_top = g["plot_top"]
    plot_bottom = g["plot_bottom"]
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    step = nice_step(y_max)
    ticks = list(range(0, y_max + 1, step))
    for tick in ticks:
        y = plot_bottom - tick / y_max * plot_h
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        draw.line((plot_left - 12, y, plot_left, y), fill=AXIS, width=2)
        label = str(tick)
        tw, th = text_size(draw, label, F_TICK)
        draw.text((plot_left - 25 - tw, y - th / 2), label, font=F_TICK, fill=TEXT)

    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=AXIS, width=2)
    draw_rotated_label(img, (left + 55, (plot_top + plot_bottom) / 2), "Number of images")
    draw_centered(draw, ((plot_left + plot_right) / 2, bottom - 34), title, F_TITLE)

    n = len(items)
    slot = plot_w / n
    bar_w = slot * 0.62
    for idx, (label, count, group) in enumerate(items):
        cx = plot_left + slot * (idx + 0.5)
        bar_h = count / y_max * plot_h
        x0 = cx - bar_w / 2
        x1 = cx + bar_w / 2
        y0 = plot_bottom - bar_h
        color = MALIGNANT if group == "Malignant" else NON_MALIGNANT
        draw.rectangle((x0, y0, x1, plot_bottom), fill=color, outline=AXIS, width=1)
        draw_centered(draw, (cx, plot_bottom + 52), label, F_LABEL)
        draw_centered(draw, (cx, y0 - 35), str(count), F_COUNT)

    total_n = sum(totals.values())
    box_w, box_h = 710, 175
    bx1 = plot_right - 35
    by0 = plot_top + 35
    bx0 = bx1 - box_w
    by1 = by0 + box_h
    draw.rounded_rectangle((bx0, by0, bx1, by1), radius=18, fill="white", outline="#C8C8C8", width=2)
    lines = [
        f"Non-malignant: {totals['Non-malignant']} ({totals['Non-malignant'] / total_n:.1%})",
        f"Malignant: {totals['Malignant']} ({totals['Malignant'] / total_n:.1%})",
    ]
    draw.text((bx0 + 28, by0 + 30), lines[0], font=F_BOX, fill=NON_MALIGNANT_TEXT)
    draw.text((bx0 + 28, by0 + 98), lines[1], font=F_BOX, fill=MALIGNANT_TEXT)


def svg_text(x, y, text, size, weight="normal", color=TEXT, anchor="middle", rotate=None):
    rotate_attr = f' transform="rotate({rotate} {x:.2f} {y:.2f})"' if rotate is not None else ""
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" '
        f'font-family="Times New Roman, Times, serif" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}"{rotate_attr}>{escape(text)}</text>'
    )


def svg_panel(parts, box, title, items, totals, y_max):
    g = panel_geometry(box)
    left = g["left"]
    top = g["top"]
    bottom = g["bottom"]
    plot_left = g["plot_left"]
    plot_right = g["plot_right"]
    plot_top = g["plot_top"]
    plot_bottom = g["plot_bottom"]
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    step = nice_step(y_max)
    for tick in range(0, y_max + 1, step):
        y = plot_bottom - tick / y_max * plot_h
        parts.append(f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="{GRID}" stroke-width="2"/>')
        parts.append(f'<line x1="{plot_left - 12}" y1="{y:.2f}" x2="{plot_left}" y2="{y:.2f}" stroke="{AXIS}" stroke-width="2"/>')
        parts.append(svg_text(plot_left - 22, y + 17, str(tick), 54, anchor="end"))
    parts.append(
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_right - plot_left}" height="{plot_bottom - plot_top}" '
        f'fill="none" stroke="{AXIS}" stroke-width="2"/>'
    )
    parts.append(svg_text(left + 55, (plot_top + plot_bottom) / 2, "Number of images", 62, anchor="middle", rotate="-90"))
    parts.append(svg_text((plot_left + plot_right) / 2, bottom - 10, title, 72, "700"))

    n = len(items)
    slot = plot_w / n
    bar_w = slot * 0.62
    for idx, (label, count, group) in enumerate(items):
        cx = plot_left + slot * (idx + 0.5)
        bar_h = count / y_max * plot_h
        x0 = cx - bar_w / 2
        y0 = plot_bottom - bar_h
        color = MALIGNANT if group == "Malignant" else NON_MALIGNANT
        parts.append(
            f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" '
            f'fill="{color}" stroke="{AXIS}" stroke-width="1"/>'
        )
        parts.append(svg_text(cx, plot_bottom + 70, label, 56))
        parts.append(svg_text(cx, y0 - 24, str(count), 52))

    total_n = sum(totals.values())
    box_w, box_h = 710, 175
    bx1 = plot_right - 35
    by0 = plot_top + 35
    bx0 = bx1 - box_w
    parts.append(
        f'<rect x="{bx0}" y="{by0}" width="{box_w}" height="{box_h}" rx="22" ry="22" '
        f'fill="white" stroke="#C8C8C8" stroke-width="2"/>'
    )
    parts.append(svg_text(bx0 + 28, by0 + 62, f"Non-malignant: {totals['Non-malignant']} ({totals['Non-malignant'] / total_n:.1%})", 48, weight="bold", color=NON_MALIGNANT_TEXT, anchor="start"))
    parts.append(svg_text(bx0 + 28, by0 + 130, f"Malignant: {totals['Malignant']} ({totals['Malignant'] / total_n:.1%})", 48, weight="bold", color=MALIGNANT_TEXT, anchor="start"))


def save_svg(path, isic_items, pad_items):
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    svg_panel(parts, (70, 120, 1950, 1380), "(a) ISIC2018 Task3", isic_items, {"Non-malignant": 8191, "Malignant": 1824}, 8000)
    svg_panel(parts, (2050, 120, 3930, 1380), "(b) PAD-UFES-20", pad_items, {"Non-malignant": 1209, "Malignant": 1089}, 1000)
    legend_y = LEGEND_Y
    legend_x = LEGEND_X_NON_MALIGNANT
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="{LEGEND_SWATCH_W}" height="{LEGEND_SWATCH_H}" fill="{NON_MALIGNANT}"/>')
    parts.append(svg_text(legend_x + LEGEND_SWATCH_W + LEGEND_TEXT_GAP, legend_y + 48, "Mapped to non-malignant", 68, weight="bold", anchor="start"))
    legend_x2 = LEGEND_X_MALIGNANT
    parts.append(f'<rect x="{legend_x2}" y="{legend_y}" width="{LEGEND_SWATCH_W}" height="{LEGEND_SWATCH_H}" fill="{MALIGNANT}"/>')
    parts.append(svg_text(legend_x2 + LEGEND_SWATCH_W + LEGEND_TEXT_GAP, legend_y + 48, "Mapped to malignant", 68, weight="bold", anchor="start"))
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def set_pdf_color(c, color):
    color = color.lstrip("#")
    c.setFillColorRGB(int(color[0:2], 16) / 255, int(color[2:4], 16) / 255, int(color[4:6], 16) / 255)


def set_pdf_stroke(c, color):
    color = color.lstrip("#")
    c.setStrokeColorRGB(int(color[0:2], 16) / 255, int(color[2:4], 16) / 255, int(color[4:6], 16) / 255)


def pdf_text(c, x, y, text, size, bold=False, color=TEXT, anchor="middle", rotate=None):
    px, py = px_to_pdf(x, y)
    set_pdf_color(c, color)
    c.setFont("Times-Bold" if bold else "Times-Roman", px_to_pdf(size))
    c.saveState()
    c.translate(px, py)
    if rotate:
        c.rotate(rotate)
    if anchor == "middle":
        c.drawCentredString(0, 0, text)
    elif anchor == "end":
        c.drawRightString(0, 0, text)
    else:
        c.drawString(0, 0, text)
    c.restoreState()


def pdf_panel(c, box, title, items, totals, y_max):
    g = panel_geometry(box)
    left = g["left"]
    top = g["top"]
    bottom = g["bottom"]
    plot_left = g["plot_left"]
    plot_right = g["plot_right"]
    plot_top = g["plot_top"]
    plot_bottom = g["plot_bottom"]
    plot_w = plot_right - plot_left
    plot_h = plot_bottom - plot_top

    step = nice_step(y_max)
    for tick in range(0, y_max + 1, step):
        y = plot_bottom - tick / y_max * plot_h
        set_pdf_stroke(c, GRID)
        c.setLineWidth(px_to_pdf(2))
        c.line(*px_to_pdf(plot_left, y), *px_to_pdf(plot_right, y))
        set_pdf_stroke(c, AXIS)
        c.line(*px_to_pdf(plot_left - 14, y), *px_to_pdf(plot_left, y))
        pdf_text(c, plot_left - 22, y + 17, str(tick), 54, anchor="end")

    set_pdf_stroke(c, AXIS)
    c.setLineWidth(px_to_pdf(2))
    x0, y0 = px_to_pdf(plot_left, plot_top)
    x1, y1 = px_to_pdf(plot_right, plot_bottom)
    c.rect(x0, y1, x1 - x0, y0 - y1, fill=0, stroke=1)
    pdf_text(c, left + 55, (plot_top + plot_bottom) / 2, "Number of images", 62, rotate=90)
    pdf_text(c, (plot_left + plot_right) / 2, bottom - 10, title, 72, True)

    n = len(items)
    slot = plot_w / n
    bar_w = slot * 0.62
    for idx, (label, count, group) in enumerate(items):
        cx = plot_left + slot * (idx + 0.5)
        bar_h = count / y_max * plot_h
        bx0 = cx - bar_w / 2
        by0 = plot_bottom - bar_h
        color = MALIGNANT if group == "Malignant" else NON_MALIGNANT
        set_pdf_color(c, color)
        set_pdf_stroke(c, AXIS)
        c.setLineWidth(px_to_pdf(1))
        x0, y_top = px_to_pdf(bx0, by0)
        x1, y_bottom = px_to_pdf(bx0 + bar_w, plot_bottom)
        c.rect(x0, y_bottom, x1 - x0, y_top - y_bottom, fill=1, stroke=1)
        pdf_text(c, cx, plot_bottom + 70, label, 56)
        pdf_text(c, cx, by0 - 24, str(count), 52)

    total_n = sum(totals.values())
    box_w, box_h = 710, 175
    bx1 = plot_right - 35
    by0 = plot_top + 35
    bx0 = bx1 - box_w
    x0, y0 = px_to_pdf(bx0, by0)
    set_pdf_color(c, "FFFFFF")
    set_pdf_stroke(c, "C8C8C8")
    c.setLineWidth(px_to_pdf(2))
    c.roundRect(x0, y0 - px_to_pdf(box_h), px_to_pdf(box_w), px_to_pdf(box_h), px_to_pdf(22), fill=1, stroke=1)
    pdf_text(c, bx0 + 28, by0 + 62, f"Non-malignant: {totals['Non-malignant']} ({totals['Non-malignant'] / total_n:.1%})", 48, bold=True, color=NON_MALIGNANT_TEXT, anchor="start")
    pdf_text(c, bx0 + 28, by0 + 130, f"Malignant: {totals['Malignant']} ({totals['Malignant'] / total_n:.1%})", 48, bold=True, color=MALIGNANT_TEXT, anchor="start")


def save_pdf(path, isic_items, pad_items):
    c = canvas.Canvas(str(path), pagesize=(PDF_W, PDF_H))
    pdf_panel(c, (70, 120, 1950, 1380), "(a) ISIC2018 Task3", isic_items, {"Non-malignant": 8191, "Malignant": 1824}, 8000)
    pdf_panel(c, (2050, 120, 3930, 1380), "(b) PAD-UFES-20", pad_items, {"Non-malignant": 1209, "Malignant": 1089}, 1000)

    legend_y = LEGEND_Y
    legend_x = LEGEND_X_NON_MALIGNANT
    set_pdf_color(c, NON_MALIGNANT)
    set_pdf_stroke(c, AXIS)
    c.rect(*px_to_pdf(legend_x, legend_y + LEGEND_SWATCH_H), px_to_pdf(LEGEND_SWATCH_W), px_to_pdf(LEGEND_SWATCH_H), fill=1, stroke=0)
    pdf_text(c, legend_x + LEGEND_SWATCH_W + LEGEND_TEXT_GAP, legend_y + 48, "Mapped to non-malignant", 68, bold=True, anchor="start")
    legend_x2 = LEGEND_X_MALIGNANT
    set_pdf_color(c, MALIGNANT)
    c.rect(*px_to_pdf(legend_x2, legend_y + LEGEND_SWATCH_H), px_to_pdf(LEGEND_SWATCH_W), px_to_pdf(LEGEND_SWATCH_H), fill=1, stroke=0)
    pdf_text(c, legend_x2 + LEGEND_SWATCH_W + LEGEND_TEXT_GAP, legend_y + 48, "Mapped to malignant", 68, bold=True, anchor="start")
    c.showPage()
    c.save()


def main():
    isic_items = [
        ("NV", 6705, "Non-malignant"),
        ("MEL", 1113, "Malignant"),
        ("BKL", 1099, "Non-malignant"),
        ("BCC", 514, "Malignant"),
        ("SCC", 197, "Malignant"),
        ("VASC", 142, "Non-malignant"),
        ("AK", 130, "Non-malignant"),
        ("DF", 115, "Non-malignant"),
    ]
    pad_items = [
        ("BCC", 845, "Malignant"),
        ("AK", 730, "Non-malignant"),
        ("NV", 244, "Non-malignant"),
        ("BKL", 235, "Non-malignant"),
        ("SCC", 192, "Malignant"),
        ("MEL", 52, "Malignant"),
    ]

    img = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(img)

    draw_panel(
        img,
        draw,
        (70, 120, 1950, 1380),
        "(a) ISIC2018 Task3",
        isic_items,
        {"Non-malignant": 8191, "Malignant": 1824},
        8000,
    )
    draw_panel(
        img,
        draw,
        (2050, 120, 3930, 1380),
        "(b) PAD-UFES-20",
        pad_items,
        {"Non-malignant": 1209, "Malignant": 1089},
        1000,
    )

    legend_y = LEGEND_Y
    legend_x = LEGEND_X_NON_MALIGNANT
    draw.rectangle((legend_x, legend_y, legend_x + LEGEND_SWATCH_W, legend_y + LEGEND_SWATCH_H), fill=NON_MALIGNANT)
    draw.text((legend_x + LEGEND_SWATCH_W + LEGEND_TEXT_GAP, legend_y - 7), "Mapped to non-malignant", font=F_LEGEND, fill=TEXT)
    legend_x2 = LEGEND_X_MALIGNANT
    draw.rectangle((legend_x2, legend_y, legend_x2 + LEGEND_SWATCH_W, legend_y + LEGEND_SWATCH_H), fill=MALIGNANT)
    draw.text((legend_x2 + LEGEND_SWATCH_W + LEGEND_TEXT_GAP, legend_y - 7), "Mapped to malignant", font=F_LEGEND, fill=TEXT)

    png_path = OUT_DIR / "dataset_label_mapping.png"
    svg_path = OUT_DIR / "dataset_label_mapping.svg"
    pdf_path = OUT_DIR / "dataset_label_mapping.pdf"
    img.convert("RGB").save(png_path, dpi=(600, 600), optimize=True)
    save_svg(svg_path, isic_items, pad_items)
    save_pdf(pdf_path, isic_items, pad_items)
    print(png_path)
    print(svg_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
