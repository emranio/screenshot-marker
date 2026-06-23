from __future__ import annotations

import math
from typing import Optional

from PIL import Image, ImageColor, ImageDraw, ImageFont

from .models import Annotation, Bbox

STROKE_ALPHA = 145
ARROW_ALPHA = 255
LABEL_BG_ALPHA = 215
LABEL_TEXT_ALPHA = 255
LABEL_TEXT_RGB = (255, 255, 255)

MIN_FONT_SIZE = 14
ANTIALIAS_SCALE = 3
MAX_SUPERSAMPLED_PIXELS = 48_000_000


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


def _line_spacing(line_h: int) -> int:
    """Vertical gap between wrapped label lines (kept tight)."""
    return max(2, line_h // 8)


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
    spacing = _line_spacing(line_h)
    total_h = line_h * len(lines) + spacing * (len(lines) - 1)
    total_w = max(_measure_line(draw, ln, font) for ln in lines)
    return lines, total_w, total_h


def _supersample_scale(width: int, height: int, preferred: int = ANTIALIAS_SCALE) -> int:
    if width <= 0 or height <= 0:
        return 1
    for scale in range(preferred, 1, -1):
        if width * height * scale * scale <= MAX_SUPERSAMPLED_PIXELS:
            return scale
    return 1


def _paste_antialiased_shape(
    target: Image.Image,
    bounds: tuple[int, int, int, int],
    draw_shape,
    *,
    preferred_scale: int = ANTIALIAS_SCALE,
) -> None:
    left, top, right, bottom = bounds
    if right <= left or bottom <= top:
        return

    width = right - left
    height = bottom - top
    scale = _supersample_scale(width, height, preferred_scale)
    patch = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
    draw_shape(ImageDraw.Draw(patch), scale, left, top)

    if scale > 1:
        patch = patch.resize((width, height), Image.Resampling.LANCZOS)
    target.alpha_composite(patch, (left, top))


def _shape_bounds(
    target: Image.Image,
    points: list[tuple[float, float]],
    pad: float,
) -> tuple[int, int, int, int]:
    min_x = min(x for x, _ in points)
    min_y = min(y for _, y in points)
    max_x = max(x for x, _ in points)
    max_y = max(y for _, y in points)
    left = max(0, int(math.floor(min_x - pad)))
    top = max(0, int(math.floor(min_y - pad)))
    right = min(target.width, int(math.ceil(max_x + pad)) + 1)
    bottom = min(target.height, int(math.ceil(max_y + pad)) + 1)
    return left, top, right, bottom


def _rounded_mask(size: tuple[int, int], radius: int) -> Image.Image:
    width, height = size
    scale = _supersample_scale(width, height)
    mask = Image.new("L", (width * scale, height * scale), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [(0, 0), (width * scale - 1, height * scale - 1)],
        radius=radius * scale,
        fill=255,
    )
    if scale > 1:
        mask = mask.resize(size, Image.Resampling.LANCZOS)
    return mask


def _rect_overlap_area(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    pad: float = 0,
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    bx0 -= pad
    by0 -= pad
    bx1 += pad
    by1 += pad
    width = min(ax1, bx1) - max(ax0, bx0)
    height = min(ay1, by1) - max(ay0, by0)
    if width <= 0 or height <= 0:
        return 0
    return width * height


def _bbox_rect(bbox: Bbox) -> tuple[int, int, int, int]:
    return (bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height)


def _clamp_bbox(x0: int, y0: int, x1: int, y1: int, image_size: tuple[int, int]) -> Bbox:
    img_w, img_h = image_size
    x0 = max(0, min(x0, img_w - 1))
    y0 = max(0, min(y0, img_h - 1))
    x1 = max(x0 + 1, min(x1, img_w))
    y1 = max(y0 + 1, min(y1, img_h))
    return Bbox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def _looks_like_tight_text_target(annotation: Annotation, stroke: int) -> bool:
    bbox = annotation.bbox
    if bbox is None or bbox.height > max(44, stroke * 5):
        return False

    description = _target_text(annotation)
    control_words = (
        "input",
        "field",
        "card",
        "row",
        "panel",
        "popup",
        "banner",
        "bar",
    )
    if any(word in description for word in control_words):
        return False

    text_words = (
        "text",
        "link",
        "line",
        "heading",
        "subtitle",
        "placeholder",
        "title",
        "shortcut",
    )
    return any(word in description for word in text_words)


def _looks_like_tab_target(annotation: Annotation, stroke: int) -> bool:
    bbox = annotation.bbox
    if bbox is None or bbox.height > max(48, stroke * 5):
        return False

    words = _target_text(annotation).replace("-", " ").replace("_", " ").split()
    tab_words = {"nav", "navigation", "tab", "tabs"}
    return any(word in tab_words for word in words)


def _target_text(annotation: Annotation) -> str:
    return f"{annotation.target_description or ''} {annotation.request_text or ''}".lower()


def _render_bbox(
    annotation: Annotation,
    image_size: tuple[int, int],
    stroke: int,
) -> Bbox:
    bbox = annotation.bbox
    if bbox is None:
        raise ValueError("annotation must have a bbox before rendering")

    if _looks_like_tab_target(annotation, stroke):
        side_pad = max(stroke * 5, round(bbox.width * 0.55))
        top_pad = max(stroke // 2, round(bbox.height * 0.08))
        bottom_pad = max(stroke * 2, round(bbox.height * 0.55))
        return _clamp_bbox(
            bbox.x - side_pad,
            bbox.y - top_pad,
            bbox.x + bbox.width + side_pad,
            bbox.y + bbox.height + bottom_pad,
            image_size,
        )

    if not _looks_like_tight_text_target(annotation, stroke):
        return bbox

    side_pad = max(stroke // 2, round(bbox.height * 0.18))
    top_pad = max(stroke, round(bbox.height * 0.35))
    bottom_pad = max(stroke // 2, round(bbox.height * 0.2))
    return _clamp_bbox(
        bbox.x - side_pad,
        bbox.y - top_pad,
        bbox.x + bbox.width + side_pad,
        bbox.y + bbox.height + bottom_pad,
        image_size,
    )


def _fit_label(
    bbox: Bbox,
    text: str,
    image_size: tuple[int, int],
    base_font_size: int,
    font_path: Optional[str],
    gap: int,
    margin: int,
    pad_x: int,
    pad_y: int,
    draw: ImageDraw.ImageDraw,
    avoid_rects: list[tuple[float, float, float, float]] | None = None,
    avoid_label_rects: list[tuple[float, float, float, float]] | None = None,
    avoid_target_rects: list[tuple[float, float, float, float]] | None = None,
) -> dict:
    """Pick the best label layout: vertical side, horizontal alignment, font size, wrapping.

    Evaluates nearby label positions and scores them by collision, distance,
    wrapping, and font shrinkage. This keeps labels from covering each other
    when several targets sit in the same region.
    Returns a dict with font, lines, text_w, text_h, lx, ly, bg_rect, vertical, horizontal.
    """
    img_w, img_h = image_size
    space_below = img_h - bbox.y - bbox.height - gap - margin
    space_above = bbox.y - gap - margin

    min_bg_h = MIN_FONT_SIZE + pad_y * 2
    vertical_options: list[tuple[str, float]] = []
    if space_below >= min_bg_h:
        vertical_options.append(("below", 0))
    if space_above >= min_bg_h:
        vertical_options.append(("above", 8 if vertical_options else 0))
    if not vertical_options:
        vertical_options = [("below", 18), ("above", 24)]

    scales = (1.0, 0.85, 0.7, 0.55)
    line_options = (1, 2, 3)
    collision_pad = max(8, gap * 0.2)
    best_layout: dict | None = None
    best_score: float | None = None

    label_avoids = list(avoid_label_rects or [])
    target_avoids = list(avoid_target_rects or [])
    if avoid_rects:
        target_avoids.extend(avoid_rects)

    def collision_score(rect: tuple[float, float, float, float]) -> float:
        rect_area = max(1.0, (rect[2] - rect[0]) * (rect[3] - rect[1]))
        label_overlap = sum(
            _rect_overlap_area(rect, avoid, collision_pad)
            for avoid in label_avoids
        )
        target_overlap = sum(
            _rect_overlap_area(rect, avoid, collision_pad)
            for avoid in target_avoids
        )
        target_overlap_ratio = min(1.0, target_overlap / rect_area)
        return label_overlap * 1000 + target_overlap_ratio * 120

    def candidate_positions(
        vertical: str, bg_w: float, bg_h: float
    ) -> list[tuple[str, float, float, float]]:
        if vertical == "below":
            bg_y0 = bbox.y + bbox.height + gap
        else:
            bg_y0 = bbox.y - bg_h - gap
        bg_y0 = max(margin, min(bg_y0, img_h - bg_h - margin))

        anchors = (
            ("left", bbox.x, 0),
            ("right", bbox.x + bbox.width - bg_w, 4),
            ("center", bbox.x + (bbox.width - bg_w) / 2, 36),
            ("image-left", margin, 320),
            ("image-right", img_w - bg_w - margin, 320),
        )
        positions: list[tuple[str, float, float, float]] = []
        seen: set[tuple[int, int]] = set()
        for horizontal, raw_x, penalty in anchors:
            bg_x0 = max(margin, min(raw_x, img_w - bg_w - margin))
            key = (round(bg_x0), round(bg_y0))
            if key in seen:
                continue
            seen.add(key)
            positions.append((horizontal, bg_x0, bg_y0, penalty))
        return positions

    bbox_cx = bbox.x + bbox.width / 2
    bbox_cy = bbox.y + bbox.height / 2

    for vertical, vertical_penalty in vertical_options:
        max_bg_h = space_below if vertical == "below" else space_above
        if max_bg_h < min_bg_h:
            max_bg_h = img_h - 2 * margin
        max_text_h = max_bg_h - pad_y * 2
        if max_text_h < MIN_FONT_SIZE:
            continue
        max_bg_w = img_w - 2 * margin
        max_text_w = max_bg_w - pad_x * 2
        if max_text_w < 60:
            continue
        for scale_index, scale in enumerate(scales):
            font_size = max(MIN_FONT_SIZE, int(base_font_size * scale))
            font = _load_font(font_path, font_size)
            for line_index, max_lines in enumerate(line_options):
                fit = _wrap_text(text, font, max_text_w, max_lines, draw)
                if fit is None:
                    continue
                lines, w, h = fit
                if not lines or h > max_text_h:
                    continue
                bg_w = w + pad_x * 2
                bg_h = h + pad_y * 2
                for horizontal, bg_x0, bg_y0, horizontal_penalty in candidate_positions(vertical, bg_w, bg_h):
                    lx = bg_x0 + pad_x
                    ly = bg_y0 + pad_y
                    bg_rect = (bg_x0, bg_y0, bg_x0 + bg_w, bg_y0 + bg_h)
                    label_cx = bg_x0 + bg_w / 2
                    label_cy = bg_y0 + bg_h / 2
                    distance = math.hypot(label_cx - bbox_cx, label_cy - bbox_cy)
                    collisions = collision_score(bg_rect)
                    score = (
                        collisions
                        + vertical_penalty
                        + horizontal_penalty
                        + scale_index * 12
                        + line_index * 6
                        + distance * 0.25
                    )
                    layout = {
                        "font": font,
                        "lines": lines,
                        "text_w": w,
                        "text_h": h,
                        "lx": lx,
                        "ly": ly,
                        "bg_rect": bg_rect,
                        "vertical": vertical,
                        "horizontal": horizontal,
                    }
                    if best_score is None or score < best_score:
                        best_score = score
                        best_layout = layout

    if best_layout is not None:
        return best_layout

    # Final fallback: smallest font, force-fit single block, BL preferred.
    font = _load_font(font_path, MIN_FONT_SIZE)
    max_bg_w = max(1, img_w - 2 * margin)
    max_text_w = max(1, max_bg_w - 2 * pad_x)
    fit = _wrap_text(text, font, max_text_w, 4, draw)
    if fit is None or not fit[0]:
        # Truncate and retry on a single line
        ellipsized = text[: max(8, len(text) // 2)] + "…"
        fit = _wrap_text(ellipsized, font, max_text_w, 1, draw) or ([ellipsized], max_text_w, _line_height(font))
    lines, w, h = fit
    bg_w = min(max_bg_w, w + 2 * pad_x)
    bg_h = h + 2 * pad_y
    bg_x0 = max(margin, min(bbox.x, img_w - bg_w - margin))
    if bbox.y + bbox.height + gap + bg_h <= img_h - margin:
        bg_y0 = bbox.y + bbox.height + gap
    else:
        bg_y0 = max(margin, bbox.y - bg_h - gap)
    lx = bg_x0 + pad_x
    ly = bg_y0 + pad_y
    return {
        "font": font,
        "lines": lines,
        "text_w": w,
        "text_h": h,
        "lx": lx,
        "ly": ly,
        "bg_rect": (bg_x0, bg_y0, bg_x0 + bg_w, bg_y0 + bg_h),
        "vertical": "below",
        "horizontal": "left",
    }


def _fit_label_at_position(
    label_position: Bbox,
    text: str,
    image_size: tuple[int, int],
    base_font_size: int,
    font_path: Optional[str],
    margin: int,
    pad_x: int,
    pad_y: int,
    draw: ImageDraw.ImageDraw,
    avoid_label_rects: list[tuple[float, float, float, float]] | None = None,
    avoid_target_rects: list[tuple[float, float, float, float]] | None = None,
) -> dict | None:
    """Fit a label near a model-provided preferred rectangle.

    The model decides the preferred spot, but several preferred spots can land
    on top of each other when their targets are close. This fitter therefore
    treats the model rectangle as a starting point and nudges the capsule off it
    (up/down/left/right) to dodge already-placed labels and other targets, while
    penalizing drift so it stays as near the requested spot as possible. If even
    the best nudged spot still overlaps another label, it returns None so the
    caller falls back to the bbox-anchored collision-aware fitter.
    """
    img_w, img_h = image_size
    if label_position.width <= 0 or label_position.height <= 0:
        return None

    label_avoids = list(avoid_label_rects or [])
    target_avoids = list(avoid_target_rects or [])
    collision_pad = max(8, margin)

    def label_overlap(rect: tuple[float, float, float, float]) -> float:
        return sum(_rect_overlap_area(rect, a, collision_pad) for a in label_avoids)

    def collision_score(rect: tuple[float, float, float, float]) -> float:
        rect_area = max(1.0, (rect[2] - rect[0]) * (rect[3] - rect[1]))
        target_overlap = sum(
            _rect_overlap_area(rect, a, collision_pad) for a in target_avoids
        )
        target_ratio = min(1.0, target_overlap / rect_area)
        return label_overlap(rect) * 1000 + target_ratio * 120

    max_image_bg_w = max(1, img_w - 2 * margin)
    max_image_bg_h = max(1, img_h - 2 * margin)
    preferred_w = max(1, min(label_position.width, max_image_bg_w))
    preferred_h = max(1, min(label_position.height, max_image_bg_h))

    limit_options = (
        (preferred_w, preferred_h, 0),
        (
            max(preferred_w, min(max_image_bg_w, round(preferred_w * 1.5))),
            max(preferred_h, min(max_image_bg_h, round(preferred_h * 1.5))),
            45,
        ),
        (max_image_bg_w, max_image_bg_h, 120),
    )
    scales = (1.0, 0.85, 0.7, 0.55)
    line_options = (1, 2, 3, 4)
    best_layout: dict | None = None
    best_score: float | None = None
    best_label_overlap: float = 0.0

    for limit_w, limit_h, expansion_penalty in limit_options:
        max_text_w = limit_w - pad_x * 2
        max_text_h = limit_h - pad_y * 2
        if max_text_w < 60 or max_text_h < MIN_FONT_SIZE:
            continue
        for scale_index, scale in enumerate(scales):
            font_size = max(MIN_FONT_SIZE, int(base_font_size * scale))
            font = _load_font(font_path, font_size)
            for line_index, max_lines in enumerate(line_options):
                fit = _wrap_text(text, font, max_text_w, max_lines, draw)
                if fit is None:
                    continue
                lines, w, h = fit
                if not lines or h > max_text_h:
                    continue

                bg_w = min(limit_w, w + pad_x * 2)
                bg_h = min(limit_h, h + pad_y * 2)

                # The model spot is offset (0, 0); the rest are escape nudges
                # along the capsule's own size so it can clear a neighbour.
                dv = bg_h + max(pad_y * 2, 12)
                dh = bg_w * 0.6 + max(pad_x, 12)
                offsets = [
                    (0, 0),
                    (0, dv), (0, -dv), (dh, 0), (-dh, 0),
                    (dh, dv), (-dh, dv), (dh, -dv), (-dh, -dv),
                    (0, 2 * dv), (0, -2 * dv), (2 * dh, 0), (-2 * dh, 0),
                    (0, 3 * dv), (0, -3 * dv),
                ]
                for off_x, off_y in offsets:
                    bg_x0 = max(margin, min(label_position.x + off_x, img_w - bg_w - margin))
                    bg_y0 = max(margin, min(label_position.y + off_y, img_h - bg_h - margin))
                    lx = bg_x0 + pad_x
                    ly = bg_y0 + pad_y
                    bg_rect = (bg_x0, bg_y0, bg_x0 + bg_w, bg_y0 + bg_h)
                    position_drift = math.hypot(
                        bg_x0 - label_position.x,
                        bg_y0 - label_position.y,
                    )
                    score = (
                        collision_score(bg_rect)
                        + expansion_penalty
                        + scale_index * 12
                        + line_index * 6
                        + position_drift * 0.2
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_label_overlap = label_overlap(bg_rect)
                        best_layout = {
                            "font": font,
                            "lines": lines,
                            "text_w": w,
                            "text_h": h,
                            "lx": lx,
                            "ly": ly,
                            "bg_rect": bg_rect,
                            "vertical": "manual",
                            "horizontal": "manual",
                        }

    # Couldn't dodge a sibling label even after nudging — let the bbox-anchored
    # fitter (which searches more anchors around the target) try instead.
    if best_layout is not None and label_avoids and best_label_overlap > 1.0:
        return None

    return best_layout


def _draw_translucent_rectangle(
    overlay: Image.Image,
    bbox: Bbox,
    color_rgb: tuple[int, int, int],
    stroke: float,
) -> None:
    x0, y0 = bbox.x, bbox.y
    x1, y1 = bbox.x + bbox.width, bbox.y + bbox.height
    radius = int(min(max(stroke * 2.2, 10), max(4, min(bbox.width, bbox.height) / 3)))
    bounds = _shape_bounds(
        overlay,
        [(x0, y0), (x1, y1)],
        pad=max(stroke * 2, radius * 0.12),
    )

    def draw_shape(draw: ImageDraw.ImageDraw, scale: int, left: int, top: int) -> None:
        draw.rounded_rectangle(
            [
                ((x0 - left) * scale, (y0 - top) * scale),
                ((x1 - left) * scale, (y1 - top) * scale),
            ],
            radius=radius * scale,
            outline=color_rgb + (STROKE_ALPHA,),
            width=max(1, round(stroke * scale)),
        )

    _paste_antialiased_shape(overlay, bounds, draw_shape)


def _quadratic_point(
    start: tuple[float, float],
    control: tuple[float, float],
    end: tuple[float, float],
    t: float,
) -> tuple[float, float]:
    inv = 1.0 - t
    return (
        inv * inv * start[0] + 2 * inv * t * control[0] + t * t * end[0],
        inv * inv * start[1] + 2 * inv * t * control[1] + t * t * end[1],
    )


def _curved_arrow_points(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    bend: float,
) -> list[tuple[float, float]]:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    length = math.hypot(dx, dy)
    if length < 1:
        return [start, end]

    ux, uy = dx / length, dy / length
    perp_x, perp_y = -uy, ux
    control = (
        (sx + ex) / 2 + perp_x * bend,
        (sy + ey) / 2 + perp_y * bend,
    )
    steps = max(18, min(80, round(length / 5)))
    return [_quadratic_point(start, control, end, i / steps) for i in range(steps + 1)]


def _arrow_bend(start: tuple[float, float], end: tuple[float, float], style: str) -> float:
    # Arrows are always straight; curved arrows were removed. Kept as a hook so
    # the call site (and the style argument) stay intact.
    return 0.0


def _draw_arrow(
    overlay: Image.Image,
    start: tuple[float, float],
    end: tuple[float, float],
    color_rgb: tuple[int, int, int],
    stroke: int,
    style: str = "curved",
) -> None:
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    desired_len = math.hypot(dx, dy)
    if desired_len < 1:
        return

    arrow_stroke = max(3, round(stroke * 0.62))
    bend = _arrow_bend(start, end, style)
    points = _curved_arrow_points(start, end, bend=bend)
    tangent_start = points[-2]
    tx = end[0] - tangent_start[0]
    ty = end[1] - tangent_start[1]
    tangent_len = math.hypot(tx, ty)
    if tangent_len < 1:
        tx, ty = dx, dy
        tangent_len = desired_len
    ux, uy = tx / tangent_len, ty / tangent_len

    head_len = max(arrow_stroke * 1.75, 8.5)
    head_angle = math.radians(42)
    tangent_angle = math.atan2(uy, ux)
    wing_left = (
        end[0] + math.cos(tangent_angle + math.pi - head_angle) * head_len,
        end[1] + math.sin(tangent_angle + math.pi - head_angle) * head_len,
    )
    wing_right = (
        end[0] + math.cos(tangent_angle + math.pi + head_angle) * head_len,
        end[1] + math.sin(tangent_angle + math.pi + head_angle) * head_len,
    )
    color_rgba = color_rgb + (ARROW_ALPHA,)
    bounds = _shape_bounds(
        overlay,
        points + [end, wing_left, wing_right],
        pad=arrow_stroke * 3,
    )

    def draw_shape(draw: ImageDraw.ImageDraw, scale: int, left: int, top: int) -> None:
        def point(p: tuple[float, float]) -> tuple[float, float]:
            return ((p[0] - left) * scale, (p[1] - top) * scale)

        def round_line(
            line_points: list[tuple[float, float]],
            *,
            fill: tuple[int, int, int, int],
            width: int,
            joint: str | None = None,
        ) -> None:
            if len(line_points) < 2:
                return
            if joint is None:
                draw.line(line_points, fill=fill, width=width)
            else:
                draw.line(line_points, fill=fill, width=width, joint=joint)
            radius = width / 2
            for x, y in (line_points[0], line_points[-1]):
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=fill)

        scaled_points = [point(p) for p in points]
        line_width = max(1, round(arrow_stroke * scale))
        round_line(scaled_points, fill=color_rgba, width=line_width, joint="curve")
        head_width = max(1, round(arrow_stroke * 0.95 * scale))
        round_line([point(wing_left), point(end)], fill=color_rgba, width=head_width)
        round_line([point(wing_right), point(end)], fill=color_rgba, width=head_width)

    _paste_antialiased_shape(overlay, bounds, draw_shape)


def _rect_distance(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    dx = max(bx0 - ax1, ax0 - bx1, 0)
    dy = max(by0 - ay1, ay0 - by1, 0)
    return math.hypot(dx, dy)


def _label_touches_bbox(
    label_rect: tuple[float, float, float, float],
    bbox: Bbox,
    stroke: int,
) -> bool:
    """True only when the label capsule overlaps or sits flush against the box.

    An arrow is redundant when the label is physically attached to its target,
    but useful for any visible gap. ``_rect_distance`` returns 0 when the
    rectangles overlap and otherwise the size of the gap between their nearest
    edges; we treat anything within a couple of stroke widths as "touching".
    """
    target_rect = _bbox_rect(bbox)
    touch_gap = max(6, stroke * 2)
    return _rect_distance(label_rect, target_rect) <= touch_gap


def _should_draw_arrow(
    label_rect: tuple[float, float, float, float],
    bbox: Bbox,
    draw_arrows: bool,
    stroke: int,
) -> bool:
    # Arrows are on by default for labeled annotations. Disabled globally via
    # draw_arrows=False (the --no-arrow flag), or suppressed when the label is
    # glued to the box (a pointer would be redundant).
    if not draw_arrows:
        return False
    return not _label_touches_bbox(label_rect, bbox, stroke)


def _arrow_endpoints(
    label_rect: tuple[float, float, float, float], bbox: Bbox
) -> tuple[tuple[float, float], tuple[float, float], str]:
    """Pick a clean start (on label edge) and end (on bbox edge) for the arrow."""
    lx, ly, lx2, ly2 = label_rect
    bx, by = bbox.x, bbox.y
    bx2, by2 = bbox.x + bbox.width, bbox.y + bbox.height
    label_cx = (lx + lx2) / 2
    label_cy = (ly + ly2) / 2
    bbox_cx = (bx + bx2) / 2
    bbox_cy = (by + by2) / 2
    x_offset = min(max(36, (lx2 - lx) * 0.22), max(40, bbox.width * 0.35))
    y_offset = min(max(28, (ly2 - ly) * 0.55), max(32, bbox.height * 0.35))
    edge_gap = 6

    if ly >= by2:  # label below bbox → arrow points up
        sx = label_cx
        sy = ly - edge_gap
        centered = abs(label_cx - bbox_cx) <= max(24, min((lx2 - lx) * 0.18, bbox.width * 0.3))
        raw_ex = label_cx if centered else label_cx - x_offset if label_cx >= bbox_cx else label_cx + x_offset
        ex = max(bx + 4, min(raw_ex, bx2 - 4))
        ey = by2 + edge_gap
        style = "straight" if centered else "curved"
    elif ly2 <= by:  # label above bbox → arrow points down
        sx = label_cx
        sy = ly2 + edge_gap
        centered = abs(label_cx - bbox_cx) <= max(24, min((lx2 - lx) * 0.18, bbox.width * 0.3))
        raw_ex = label_cx if centered else label_cx + x_offset if label_cx >= bbox_cx else label_cx - x_offset
        ex = max(bx + 4, min(raw_ex, bx2 - 4))
        ey = by - edge_gap
        style = "straight" if centered else "curved"
    elif lx >= bx2:  # label to the right → arrow points left
        sx = lx - edge_gap
        sy = label_cy
        ex = bx2 + edge_gap
        centered = abs(label_cy - bbox_cy) <= max(20, min((ly2 - ly) * 0.45, bbox.height * 0.3))
        raw_ey = label_cy if centered else label_cy - y_offset if label_cy >= bbox_cy else label_cy + y_offset
        ey = max(by + 4, min(raw_ey, by2 - 4))
        style = "straight" if centered else "curved"
    else:  # label to the left → arrow points right
        sx = lx2 + edge_gap
        sy = label_cy
        ex = bx - edge_gap
        centered = abs(label_cy - bbox_cy) <= max(20, min((ly2 - ly) * 0.45, bbox.height * 0.3))
        raw_ey = label_cy if centered else label_cy + y_offset if label_cy >= bbox_cy else label_cy - y_offset
        ey = max(by + 4, min(raw_ey, by2 - 4))
        style = "straight" if centered else "curved"
    return (sx, sy), (ex, ey), style


def _render_label(
    canvas: Image.Image,
    layout: dict,
    color_rgb: tuple[int, int, int],
) -> None:
    """Draw a flat translucent label capsule and text."""
    lines: list[str] = layout["lines"]
    font: ImageFont.ImageFont = layout["font"]
    lx, ly = layout["lx"], layout["ly"]
    bg_x0, bg_y0, bg_x1, bg_y1 = layout["bg_rect"]

    bg_x0 = max(0, int(round(bg_x0)))
    bg_y0 = max(0, int(round(bg_y0)))
    bg_x1 = min(canvas.width, int(round(bg_x1)))
    bg_y1 = min(canvas.height, int(round(bg_y1)))
    if bg_x1 <= bg_x0 or bg_y1 <= bg_y0:
        return

    bg_w = bg_x1 - bg_x0
    bg_h = bg_y1 - bg_y0
    if len(lines) > 1:
        # Multi-line: a rounded rectangle (not a pill) — modest, bounded corners.
        radius = min(max(12, min(bg_w, bg_h) // 5), 34)
    else:
        # Single line: fully rounded pill ends.
        radius = max(8, min(bg_w, bg_h) // 2)

    mask = _rounded_mask((bg_w, bg_h), radius)
    tint = Image.new("RGBA", (bg_w, bg_h), color_rgb + (LABEL_BG_ALPHA,))
    canvas.paste(tint, (bg_x0, bg_y0), mask)

    draw = ImageDraw.Draw(canvas)
    line_h = _line_height(font)
    spacing = _line_spacing(line_h)
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
    draw_arrows: bool = True,
) -> Image.Image:
    canvas = image.convert("RGBA").copy()
    img_w, img_h = canvas.size

    base_font_size = max(20, stroke_width * 4)
    gap = max(stroke_width * 6, 48)
    margin = max(stroke_width, 8)
    label_pad_x = max(stroke_width * 2 + 8, 24)
    label_pad_y = max(stroke_width, 10)
    rectangle_stroke = max(2.5, round(stroke_width * 0.45))

    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    measure_draw = ImageDraw.Draw(overlay)
    default_rgb = _parse_color(default_color)

    render_items: list[Optional[dict]] = []
    for ann in annotations:
        if ann.not_found or ann.bbox is None:
            render_items.append(None)
            continue

        render_bbox = _render_bbox(ann, (img_w, img_h), stroke_width)
        color_rgb = _parse_color(ann.color, fallback=default_rgb)
        render_items.append({"annotation": ann, "bbox": render_bbox, "color_rgb": color_rgb})

    for item in render_items:
        if item is None:
            continue
        render_bbox: Bbox = item["bbox"]
        color_rgb: tuple[int, int, int] = item["color_rgb"]
        _draw_translucent_rectangle(overlay, render_bbox, color_rgb, rectangle_stroke)

    layouts: list[Optional[dict]] = [None] * len(render_items)
    label_indices = [
        index
        for index, item in enumerate(render_items)
        if item is not None and item["annotation"].label_text and item["annotation"].label_text.strip()
    ]
    label_indices.sort(
        key=lambda index: (
            render_items[index]["bbox"].y,
            render_items[index]["bbox"].x,
            render_items[index]["bbox"].width * render_items[index]["bbox"].height,
            index,
        )
    )

    for index in label_indices:
        item = render_items[index]
        if item is None:
            continue

        ann: Annotation = item["annotation"]
        render_bbox: Bbox = item["bbox"]
        color_rgb: tuple[int, int, int] = item["color_rgb"]
        avoid_target_rects = [
            _bbox_rect(other["bbox"])
            for other_index, other in enumerate(render_items)
            if other is not None and other_index != index
        ]
        avoid_label_rects = [
            layout["bg_rect"] for layout in layouts if layout is not None
        ]
        layout = None
        if ann.label_position is not None:
            layout = _fit_label_at_position(
                ann.label_position,
                ann.label_text.strip(),
                (img_w, img_h),
                base_font_size,
                font_path,
                margin,
                label_pad_x,
                label_pad_y,
                measure_draw,
                avoid_label_rects=avoid_label_rects,
                avoid_target_rects=avoid_target_rects,
            )
        if layout is None:
            layout = _fit_label(
                render_bbox,
                ann.label_text.strip(),
                (img_w, img_h),
                base_font_size,
                font_path,
                gap,
                margin,
                label_pad_x,
                label_pad_y,
                measure_draw,
                avoid_label_rects=avoid_label_rects,
                avoid_target_rects=avoid_target_rects,
            )
        layout["color_rgb"] = color_rgb
        layouts[index] = layout

        label_rect = (
            *layout["bg_rect"],
        )
        if _should_draw_arrow(label_rect, render_bbox, draw_arrows, stroke_width):
            arrow_start, arrow_end, arrow_style = _arrow_endpoints(label_rect, render_bbox)
            _draw_arrow(overlay, arrow_start, arrow_end, color_rgb, stroke_width, arrow_style)

    canvas = Image.alpha_composite(canvas, overlay)

    for layout in layouts:
        if layout is None:
            continue
        _render_label(
            canvas,
            layout,
            layout["color_rgb"],
        )

    return canvas.convert("RGB")
