#!/usr/bin/env python3
"""Draw TurboDiffusion-style acceleration decomposition figures."""

from __future__ import annotations

from pathlib import Path
import math

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "assets"

FONT_CANDIDATES = [
    Path("/home/jovyan/.cache/rattler/cache/pkgs/matplotlib-base-3.10.9-py310hfde16b3_0/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"),
    Path("/home/jovyan/.cache/rattler/cache/pkgs/matplotlib-base-3.10.9-py310hfde16b3_0/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans-Bold.ttf"),
    Path("/home/jovyan/.cache/rattler/cache/pkgs/matplotlib-base-3.10.7-py310hfde16b3_0/lib/python3.10/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf"),
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf"] if bold else ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf"]
    for candidate in FONT_CANDIDATES:
        if candidate.name in names and candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    for candidate in FONT_CANDIDATES:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.multiline_textbbox((0, 0), text, font=fnt, spacing=4)
    return box[2] - box[0], box[3] - box[1]


def draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    anchor: str = "la",
) -> None:
    draw.multiline_text(xy, text, font=fnt, fill=fill, anchor=anchor, spacing=4)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 5,
    head: int = 14,
) -> None:
    x1, y1 = start
    x2, y2 = end
    draw.line((x1, y1, x2, y2), fill=color, width=width)
    if abs(x2 - x1) >= abs(y2 - y1):
        sign = 1 if x2 > x1 else -1
        pts = [(x2, y2), (x2 - sign * head, y2 - head * 0.55), (x2 - sign * head, y2 + head * 0.55)]
    else:
        sign = 1 if y2 > y1 else -1
        pts = [(x2, y2), (x2 - head * 0.55, y2 - sign * head), (x2 + head * 0.55, y2 - sign * head)]
    draw.polygon(pts, fill=color)


def bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    steps: int = 80,
) -> list[tuple[float, float]]:
    points = []
    for i in range(steps + 1):
        t = i / steps
        u = 1 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def draw_curve_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    width: int = 5,
    head: int = 18,
) -> None:
    draw.line(points, fill=color, width=width, joint="curve")
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    angle = math.atan2(y2 - y1, x2 - x1)
    left = angle + math.pi * 0.82
    right = angle - math.pi * 0.82
    pts = [
        (x2, y2),
        (x2 + math.cos(left) * head, y2 + math.sin(left) * head),
        (x2 + math.cos(right) * head, y2 + math.sin(right) * head),
    ]
    draw.polygon(pts, fill=color)


def draw_double_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color: tuple[int, int, int],
    width: int = 5,
    head: int = 14,
) -> None:
    draw_arrow(draw, start, end, color, width=width, head=head)
    draw_arrow(draw, end, start, color, width=width, head=head)


def draw_latency_figure(
    *,
    title_resolution: str,
    latencies: list[float],
    labels: list[str],
    speedups: list[str],
    total_speedup: str,
    output: Path,
) -> None:
    width, height = 2048, 768
    scale = 2
    img = Image.new("RGB", (width * scale, height * scale), "white")
    draw = ImageDraw.Draw(img)

    def s(value: float) -> int:
        return int(round(value * scale))

    def p(x: float, y: float) -> tuple[int, int]:
        return s(x), s(y)

    black = (18, 22, 26)
    gray = (198, 205, 211)
    red = (153, 9, 36)
    teal = (39, 132, 118)

    title_font = font(s(35), bold=True)
    label_font = font(s(35))
    label_bold_font = font(s(35), bold=True)
    value_font = font(s(40))
    ratio_font = font(s(31))
    axis_font = font(s(40), bold=True)
    small_font = font(s(24))

    x0 = 500
    max_width = 1410
    max_latency = max(latencies)
    y_centers = [105, 220, 335, 450]
    bar_h = 68

    def x_for(value: float) -> float:
        return x0 + value / max_latency * max_width

    # Axis guides.
    draw.line((s(x0), s(42), s(x0), s(610)), fill=black, width=s(4))
    # Dashed right guide.
    for y in range(44, 610, 24):
        draw.line((s(x0 + max_width), s(y), s(x0 + max_width), s(y + 12)), fill=black, width=s(3))

    final_x = x_for(latencies[-1]) + 10
    for y in range(505, 610, 22):
        draw.line((s(final_x), s(y), s(final_x), s(y + 10)), fill=black, width=s(3))

    for idx, (label, value, y) in enumerate(zip(labels, latencies, y_centers)):
        color = gray if idx == 0 else red
        label_font_used = title_font if idx == 0 else label_font
        if "(final version)" in label:
            lines = label.split("\n")
            draw_text(draw, p(72, y - 46), lines[0], label_font, black)
            draw_text(draw, p(192, y - 4), lines[1], label_bold_font, black)
        else:
            draw_text(draw, p(28 if idx == 0 else 74, y - 18 if "\n" not in label else y - 38), label, label_font_used, black)
        x_end = x_for(value)
        draw.rectangle((s(x0), s(y - bar_h / 2), s(x_end), s(y + bar_h / 2)), fill=color)

        value_text = f"{value:.0f}" if value >= 10 else f"{value:.2f}".rstrip("0").rstrip(".")
        if idx == len(latencies) - 1:
            tx = x_end + 22
        else:
            tx = min(x_end + 12, x0 + max_width + 10)
        draw_text(draw, p(tx, y - 22), value_text, value_font, black)

    # TD-style curved stage arrows, placed in whitespace so labels remain clear.
    top_start = (x_for(latencies[0]) - 420, y_centers[1] + 20)
    top_end = (x_for(latencies[1]) + 26, y_centers[1] + 32)
    top_curve = bezier(top_start, (top_start[0] + 90, top_start[1] + 84), (top_end[0] + 135, top_end[1] + 64), top_end)
    draw_curve_arrow(draw, [p(x, y) for x, y in top_curve], teal, width=s(5), head=s(18))
    draw_text(draw, p((top_start[0] + top_end[0]) / 2 + 105, y_centers[1] + 48), speedups[0], ratio_font, teal)

    middle_start = (x_for(latencies[1]) - 22, y_centers[1] + 58)
    middle_end = (x_for(latencies[2]) + 46, y_centers[2] + 2)
    middle_curve = bezier(
        middle_start,
        (middle_start[0] - 320, middle_start[1] + 100),
        (middle_end[0] + 500, middle_end[1] + 62),
        middle_end,
    )
    draw_curve_arrow(draw, [p(x, y) for x, y in middle_curve], teal, width=s(5), head=s(18))
    draw_text(draw, p((middle_start[0] + middle_end[0]) / 2 - 35, y_centers[2] - 44), speedups[1], ratio_font, teal)

    small_start = (x_for(latencies[2]) + 142, y_centers[2] + 30)
    small_end = (x_for(latencies[3]) + 150, y_centers[3] + 8)
    small_curve = bezier(small_start, (small_start[0] + 36, small_start[1] + 58), (small_end[0] + 54, small_end[1] - 34), small_end)
    draw_curve_arrow(draw, [p(x, y) for x, y in small_curve], teal, width=s(5), head=s(17))
    draw_text(draw, p(small_end[0] + 46, y_centers[3] - 26), speedups[2], ratio_font, teal)

    # Overall teacher-to-final arrow.
    bottom_y = 605
    draw_double_arrow(draw, p(x0 + max_width - 2, bottom_y), p(final_x + 12, bottom_y), teal, width=s(5), head=s(16))
    draw_text(draw, p((x0 + max_width + final_x) / 2 - 34, bottom_y - 48), total_speedup, ratio_font, teal)

    axis_label = "Generation latency (s) on a Single H20"
    tw, _ = text_size(draw, axis_label, axis_font)
    draw_text(draw, p((width - tw / scale) / 2 + 130, 675), axis_label, axis_font, black)

    img = img.resize((width, height), Image.Resampling.LANCZOS)
    output.parent.mkdir(parents=True, exist_ok=True)
    img.save(output)


def main() -> None:
    labels = [
        "LTX-2-teacher-512x768",
        "+ W8A8 & FusedNorm",
        "+ rCM\n(4-step student)",
        "+ SageSLA\n(final version)",
    ]
    draw_latency_figure(
        title_resolution="512x768",
        labels=labels,
        latencies=[46.4757, 27.3201, 2.4679, 1.1904],
        speedups=["1.70x", "11.07x", "2.07x"],
        total_speedup="39x",
        output=ASSET_DIR / "turbot2av_td_style_no_cpuoffload_512x768.png",
    )
    labels_high_res = [
        "LTX-2-teacher-1024x1792",
        "+ W8A8 & FusedNorm",
        "+ rCM\n(4-step student)",
        "+ SageSLA\n(final version)",
    ]
    draw_latency_figure(
        title_resolution="1024x1792",
        labels=labels_high_res,
        latencies=[318.7405, 233.3424, 16.6961, 5.8173],
        speedups=["1.37x", "13.98x", "2.87x"],
        total_speedup="55x",
        output=ASSET_DIR / "turbot2av_td_style_no_cpuoffload_1024x1792.png",
    )


if __name__ == "__main__":
    main()
