from __future__ import annotations

import math
from typing import Optional

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont

from .models import Annotation, Bbox

STROKE_ALPHA = 200
ARROW_ALPHA = 215
LABEL_BG_ALPHA = 215
LABEL_TEXT_ALPHA = 255
LABEL_TEXT_RGB = (255, 255, 255)

MIN_FONT_SIZE = 14


def _load_font(font_path: Optional[str], size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


_DEFAULT_FALLBACK_RGB = (220, 38, 38)


def _parse_color(spec: Optional[str], fallback: tuple[int, int, int] = _DEFAULT_FALLBACK_RGB) -> tuple[int, int, int]:
    """Parse hex strings, CSS color names, or rgb(...) notation. Falls back on garbage input."""
    if not spec:
        return fallback
    try:
        rgba = ImageColor.getrgb(spec.strip())
    except (ValueError, AttributeError):
        return fallback
    return rgba[:3]


def _line_height(font: ImageFont.ImageFont) -> int:
    ascent, descent = font.getmetrics()
    return ascent + descent


def _measure_line(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
) -> int:
    return int(draw.textlength(text, font=font))


def _wrap_text(
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
    max_lines: int,
    draw: ImageDraw.ImageDraw,
) -> tuple[list[str], int, int] | None:
    """Greedy word wrap. Returns (lines, total_w, total_h) or None if it doesn't fit."""
    words = text.split()
    if not words:
        return [], 0, 0

    lines: list[str] = []
    current = words[0]
    if _measure_line(draw, current, font) > max_width:
        return None

    for word in words[1:]:
        candidate = current + " " + word
        if _measure_line(draw, candidate, font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
            if _measure_line(draw, current, font) > max_width:
                return None
            if len(lines) >= max_lines:
                return None

    lines.append(current)
    if len(lines) > max_lines:
        return None

    line_h = _line_height(font)
    spacing = max(2, line_h // 5)
    total_h = line_h * len(lines) + spacing * (len(lines) - 1)
    total_w = max(_measure_line(draw, ln, font) for ln in lines)
    return lines, total_w, total_h


def _fit_label(
    bbox: Bbox,
    text: str,
    image_size: tuple[int, int],
    base_font_size: int,
    font_path: Optional[str],
    gap: int,
    margin: int,
    draw: ImageDraw.ImageDraw,
) -> dict:
    """Pick the best label layout: vertical side, horizontal alignment, font size, wrapping.

    Tries (in order of preference):
      vertical:   below if there's room, else above
      horizontal: left-aligned to bbox, else right-aligned
      font:       100% → 85% → 70% → 55% of base
      lines:      1 → 2 → 3
    Returns a dict with font, lines, text_w, text_h, lx, ly, vertical, horizontal.
    """
    img_w, img_h = image_size
    space_below = img_h - bbox.y - bbox.height - gap - margin
    space_above = bbox.y - gap - margin

    if space_below >= 30:
        vertical_options = ["below", "above"]
    else:
        vertical_options = ["above", "below"]

    horizontal_options = ["left", "right"]
    scales = (1.0, 0.85, 0.7, 0.55)
    line_options = (1, 2, 3)

    for vertical in vertical_options:
        max_h = space_below if vertical == "below" else space_above
        if max_h < 20:
            continue
        for horizontal in horizontal_options:
            if horizontal == "left":
                max_w = img_w - bbox.x - margin
            else:
                max_w = bbox.x + bbox.width - margin
            if max_w < 60:
                continue
            for scale in scales:
                font_size = max(MIN_FONT_SIZE, int(base_font_size * scale))
                font = _load_font(font_path, font_size)
                for max_lines in line_options:
                    fit = _wrap_text(text, font, max_w, max_lines, draw)
                    if fit is None:
                        continue
                    lines, w, h = fit
                    if not lines or h > max_h:
                        continue
                    if vertical == "below":
                        ly = bbox.y + bbox.height + gap
                    else:
                        ly = bbox.y - h - gap
                    if horizontal == "left":
                        lx = bbox.x
                    else:
                        lx = bbox.x + bbox.width - w
                    lx = max(margin, min(lx, img_w - w - margin))
                    ly = max(margin, min(ly, img_h - h - margin))
                    return {
                        "font": font,
                        "lines": lines,
                        "text_w": w,
                        "text_h": h,
                        "lx": lx,
                        "ly": ly,
                        "vertical": vertical,
                        "horizontal": horizontal,
                    }

    # Final fallback: smallest font, force-fit single block, BL preferred.
    font = _load_font(font_path, MIN_FONT_SIZE)
    fit = _wrap_text(text, font, img_w - 2 * margin, 4, draw)
    if fit is None or not fit[0]:
        # Truncate and retry on a single line
        ellipsized = text[: max(8, len(text) // 2)] + "…"
        fit = _wrap_text(ellipsized, font, img_w - 2 * margin, 1, draw) or ([ellipsized], img_w - 2 * margin, _line_height(font))
    lines, w, h = fit
    lx = max(margin, min(bbox.x, img_w - w - margin))
    if bbox.y + bbox.height + gap + h <= img_h - margin:
        ly = bbox.y + bbox.height + gap
    else:
        ly = max(margin, bbox.y - h - gap)
    return {
        "font": font,
        "lines": lines,
        "text_w": w,
        "text_h": h,
        "lx": lx,
        "ly": ly,
        "vertical": "below",
        "horizontal": "left",
    }


def _draw_translucent_rectangle(
    overlay: Image.Image,
    bbox: Bbox,
    color_rgb: tuple[int, int, int],
    stroke: int,
) -> None:
    draw = ImageDraw.Draw(overlay)
    x0, y0 = bbox.x, bbox.y
    x1, y1 = bbox.x + bbox.width, bbox.y + bbox.height
    radius = max(4, stroke)
    draw.rounded_rectangle(
        [(x0, y0), (x1, y1)],
        radius=radius,
        outline=color_rgb + (STROKE_ALPHA,),
        width=stroke,
    )


def _draw_arrow(
    overlay: Image.Image,
    start: tuple[float, float],
    end: tuple[float, float],
    color_rgb: tuple[int, int, int],
    stroke: int,
) -> None:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    if length < 1:
        return

    head_len = max(stroke * 4, 12)
    head_half_width = max(stroke * 2.4, 8)
    ux, uy = dx / length, dy / length
    perp_x, perp_y = -uy, ux
    base_x = ex - ux * head_len
    base_y = ey - uy * head_len
    line_end_x = ex - ux * (head_len * 0.7)
    line_end_y = ey - uy * (head_len * 0.7)

    color_rgba = color_rgb + (ARROW_ALPHA,)
    draw = ImageDraw.Draw(overlay)
    draw.line(
        [(sx, sy), (line_end_x, line_end_y)],
        fill=color_rgba,
        width=stroke,
    )
    tip = (ex, ey)
    left = (base_x + perp_x * head_half_width, base_y + perp_y * head_half_width)
    right = (base_x - perp_x * head_half_width, base_y - perp_y * head_half_width)
    draw.polygon([tip, left, right], fill=color_rgba)


def _arrow_endpoints(
    label_rect: tuple[int, int, int, int], bbox: Bbox
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Pick a clean start (on label edge) and end (on bbox edge) for the arrow."""
    lx, ly, lx2, ly2 = label_rect
    bx, by = bbox.x, bbox.y
    bx2, by2 = bbox.x + bbox.width, bbox.y + bbox.height
    label_cx = (lx + lx2) / 2

    if ly >= by2:  # label below bbox → arrow points up
        sx = label_cx
        sy = ly
        ex = max(bx + 4, min(label_cx, bx2 - 4))
        ey = by2
    elif ly2 <= by:  # label above bbox → arrow points down
        sx = label_cx
        sy = ly2
        ex = max(bx + 4, min(label_cx, bx2 - 4))
        ey = by
    elif lx >= bx2:  # label to the right → arrow points left
        sx = lx
        sy = (ly + ly2) / 2
        ex = bx2
        ey = max(by + 4, min((ly + ly2) / 2, by2 - 4))
    else:  # label to the left → arrow points right
        sx = lx2
        sy = (ly + ly2) / 2
        ex = bx
        ey = max(by + 4, min((ly + ly2) / 2, by2 - 4))
    return (sx, sy), (ex, ey)


def _render_label_with_blur_bg(
    canvas: Image.Image,
    layout: dict,
    color_rgb: tuple[int, int, int],
    pad_x: int,
    pad_y: int,
    blur_radius: int,
) -> None:
    """Blur the canvas region under the label, mask with rounded corners, draw text on top."""
    lines: list[str] = layout["lines"]
    font: ImageFont.ImageFont = layout["font"]
    lx, ly = layout["lx"], layout["ly"]
    text_w, text_h = layout["text_w"], layout["text_h"]

    bg_x0 = max(0, lx - pad_x)
    bg_y0 = max(0, ly - pad_y)
    bg_x1 = min(canvas.width, lx + text_w + pad_x)
    bg_y1 = min(canvas.height, ly + text_h + pad_y)
    if bg_x1 <= bg_x0 or bg_y1 <= bg_y0:
        return

    bg_w = bg_x1 - bg_x0
    bg_h = bg_y1 - bg_y0
    radius = max(4, min(pad_x, pad_y))

    crop = canvas.crop((bg_x0, bg_y0, bg_x1, bg_y1))
    blurred = crop.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    mask = Image.new("L", (bg_w, bg_h), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [(0, 0), (bg_w - 1, bg_h - 1)], radius=radius, fill=255
    )
    canvas.paste(blurred, (bg_x0, bg_y0), mask)

    tint = Image.new("RGBA", (bg_w, bg_h), (0, 0, 0, 0))
    ImageDraw.Draw(tint).rounded_rectangle(
        [(0, 0), (bg_w - 1, bg_h - 1)],
        radius=radius,
        fill=color_rgb + (LABEL_BG_ALPHA,),
    )
    canvas.paste(tint, (bg_x0, bg_y0), tint)

    draw = ImageDraw.Draw(canvas)
    line_h = _line_height(font)
    spacing = max(2, line_h // 5)
    cy = ly
    for line in lines:
        draw.text(
            (lx, cy),
            line,
            font=font,
            fill=LABEL_TEXT_RGB + (LABEL_TEXT_ALPHA,),
        )
        cy += line_h + spacing


def render(
    image: Image.Image,
    annotations: list[Annotation],
    *,
    default_color: str,
    stroke_width: int,
    font_path: Optional[str],
) -> Image.Image:
    canvas = image.convert("RGBA").copy()
    img_w, img_h = canvas.size

    base_font_size = max(20, stroke_width * 4)
    gap = max(stroke_width * 2, 12)
    margin = max(stroke_width, 8)
    label_pad_x = max(stroke_width + 2, 10)
    label_pad_y = max(stroke_width // 2 + 2, 6)
    blur_radius = max(6, stroke_width * 2)

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    measure_draw = ImageDraw.Draw(overlay)

    layouts: list[Optional[dict]] = []
    for ann in annotations:
        if ann.not_found or ann.bbox is None:
            layouts.append(None)
            continue

        default_rgb = _parse_color(default_color)
        color_rgb = _parse_color(ann.color, fallback=default_rgb)
        _draw_translucent_rectangle(overlay, ann.bbox, color_rgb, stroke_width)

        if ann.label_text and ann.label_text.strip():
            layout = _fit_label(
                ann.bbox,
                ann.label_text.strip(),
                (img_w, img_h),
                base_font_size,
                font_path,
                gap,
                margin,
                measure_draw,
            )
            layout["color_rgb"] = color_rgb
            layouts.append(layout)

            label_rect = (
                layout["lx"],
                layout["ly"],
                layout["lx"] + layout["text_w"],
                layout["ly"] + layout["text_h"],
            )
            arrow_start, arrow_end = _arrow_endpoints(label_rect, ann.bbox)
            _draw_arrow(overlay, arrow_start, arrow_end, color_rgb, stroke_width)
        else:
            layouts.append(None)

    canvas = Image.alpha_composite(canvas, overlay)

    for layout in layouts:
        if layout is None:
            continue
        _render_label_with_blur_bg(
            canvas,
            layout,
            layout["color_rgb"],
            label_pad_x,
            label_pad_y,
            blur_radius,
        )

    return canvas.convert("RGB")
