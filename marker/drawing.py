from __future__ import annotations

import math
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from .models import Annotation, Arrow, Bbox, Label, Point


def _load_font(font_path: Optional[str], size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def draw_rectangle(
    draw: ImageDraw.ImageDraw,
    bbox: Bbox,
    color: str,
    stroke: int,
) -> None:
    x0, y0 = bbox.x, bbox.y
    x1, y1 = bbox.x + bbox.width, bbox.y + bbox.height
    draw.rectangle([(x0, y0), (x1, y1)], outline=color, width=stroke)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: Point,
    end: Point,
    color: str,
    stroke: int,
) -> None:
    dx = end.x - start.x
    dy = end.y - start.y
    length = math.hypot(dx, dy)
    if length < 1:
        return

    head_len = max(stroke * 4, 12)
    head_half_width = max(stroke * 2.2, 7)

    ux, uy = dx / length, dy / length
    perp_x, perp_y = -uy, ux

    base_x = end.x - ux * head_len
    base_y = end.y - uy * head_len

    line_end_x = end.x - ux * (head_len * 0.6)
    line_end_y = end.y - uy * (head_len * 0.6)
    draw.line(
        [(start.x, start.y), (line_end_x, line_end_y)],
        fill=color,
        width=stroke,
    )

    tip = (end.x, end.y)
    left = (base_x + perp_x * head_half_width, base_y + perp_y * head_half_width)
    right = (base_x - perp_x * head_half_width, base_y - perp_y * head_half_width)
    draw.polygon([tip, left, right], fill=color)


_OPPOSITE = {"above": "below", "below": "above", "left": "right", "right": "left"}


def _position_for(
    placement: str, ax: int, ay: int, w: int, h: int, pad: int
) -> tuple[int, int]:
    if placement == "above":
        return (ax - w // 2, ay - h - pad)
    if placement == "below":
        return (ax - w // 2, ay + pad)
    if placement == "left":
        return (ax - w - pad, ay - h // 2)
    return (ax + pad, ay - h // 2)


def _fits(x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> bool:
    return 0 <= x and x + w <= img_w and 0 <= y and y + h <= img_h


def _label_position(
    text_size: tuple[int, int],
    label: Label,
    pad: int,
    image_size: tuple[int, int],
) -> tuple[int, int]:
    w, h = text_size
    img_w, img_h = image_size
    ax, ay = label.anchor.x, label.anchor.y
    for p in (label.placement, _OPPOSITE[label.placement]):
        x, y = _position_for(p, ax, ay, w, h, pad)
        if _fits(x, y, w, h, img_w, img_h):
            return (x, y)
    return _position_for(label.placement, ax, ay, w, h, pad)


def draw_label(
    draw: ImageDraw.ImageDraw,
    label: Label,
    color: str,
    font: ImageFont.ImageFont,
    pad: int,
    image_size: tuple[int, int],
) -> None:
    bbox = draw.textbbox((0, 0), label.text, font=font, stroke_width=2)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x, y = _label_position((text_w, text_h), label, pad, image_size)
    img_w, img_h = image_size
    margin = max(2, pad // 2)
    x = max(margin, min(x, img_w - text_w - margin))
    y = max(margin, min(y, img_h - text_h - margin))
    draw.text(
        (x, y),
        label.text,
        font=font,
        fill=color,
        stroke_width=2,
        stroke_fill="white",
    )


def render(
    image: Image.Image,
    annotations: list[Annotation],
    *,
    default_color: str,
    stroke_width: int,
    font_path: Optional[str],
) -> Image.Image:
    canvas = image.convert("RGBA").copy()
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(16, stroke_width * 4)
    font = _load_font(font_path, font_size)
    label_pad = max(6, stroke_width * 2)

    for ann in annotations:
        if ann.not_found:
            continue
        color = ann.color or default_color
        if ann.bbox is not None:
            draw_rectangle(draw, ann.bbox, color, stroke_width)
        if ann.arrow is not None:
            draw_arrow(draw, ann.arrow.start, ann.arrow.end, color, stroke_width)
        if ann.label is not None:
            draw_label(draw, ann.label, color, font, label_pad, canvas.size)

    return Image.alpha_composite(canvas, overlay).convert("RGB")
