from __future__ import annotations

import base64
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from PIL import Image

from .config import (
    DEFAULT_COLOR,
    DEFAULT_MODEL,
    auto_stroke_width,
    resolve_font_path,
)
from .drawing import render
from .models import (
    Annotation,
    AnnotationResult,
    Arrow,
    Bbox,
    Label,
    Point,
)
from .parser import parse_response
from .vision import call_vision, encode_image, refine_bbox_call

__all__ = [
    "annotate",
    "Annotation",
    "AnnotationResult",
    "Arrow",
    "Bbox",
    "Label",
    "Point",
]


def _default_output_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_annotated.png")


def annotate(
    image_path: str | Path,
    queries: list[str],
    output_path: Optional[str | Path] = None,
    *,
    model: str = DEFAULT_MODEL,
    color: str = DEFAULT_COLOR,
    stroke_width: Optional[int] = None,
    font_path: Optional[str] = None,
    refine: bool = True,
    refine_padding: float = 0.15,
) -> AnnotationResult:
    """Annotate ``image_path`` with one or more natural-language ``queries``.

    A single vision-model call resolves all queries; the resulting rectangles,
    arrows, and labels are drawn onto the image and saved as PNG.

    If ``output_path`` is not provided, the result is written next to the
    input as ``<image_stem>_annotated.png``.

    When ``refine`` is True (default), each located bbox is tightened by a
    second pass: the image is cropped around the rough bbox plus
    ``refine_padding`` and the model returns precise edges within the crop.
    This dramatically improves border alignment on tall screenshots at the
    cost of one extra LLM call per resolved query (run in parallel).

    Returns an :class:`AnnotationResult` with the resolved annotations, the
    output path, and the list of queries the model could not locate.
    """
    if not queries:
        raise ValueError("queries must contain at least one annotation request.")

    image_path = Path(image_path)
    output_path = Path(output_path) if output_path is not None else _default_output_path(image_path)

    image_b64, mime, width, height = encode_image(image_path)
    raw = call_vision(image_b64, mime, width, height, queries, model)
    annotations, unresolved = parse_response(raw, queries, width, height)

    if refine:
        with Image.open(image_path) as src:
            annotations = _refine_bboxes(
                src, annotations, model=model, padding=refine_padding
            )

    stroke = stroke_width if stroke_width is not None else auto_stroke_width(width, height)
    font = resolve_font_path(font_path)

    with Image.open(image_path) as src:
        annotated = render(
            src,
            annotations,
            default_color=color,
            stroke_width=stroke,
            font_path=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path, format="PNG")

    return AnnotationResult(
        output_path=output_path,
        annotations=annotations,
        unresolved=unresolved,
    )


def _crop_for_refine(
    image: Image.Image, bbox: Bbox, padding: float
) -> tuple[str, int, int, int, int]:
    pad_x = max(8, int(bbox.width * padding))
    pad_y = max(8, int(bbox.height * padding))
    left = max(0, bbox.x - pad_x)
    top = max(0, bbox.y - pad_y)
    right = min(image.width, bbox.x + bbox.width + pad_x)
    bottom = min(image.height, bbox.y + bbox.height + pad_y)
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii"), left, top, right, bottom


def _refine_one(
    image: Image.Image,
    annotation: Annotation,
    *,
    model: str,
    padding: float,
) -> Annotation:
    if annotation.not_found or annotation.bbox is None:
        return annotation

    crop_b64, left, top, right, bottom = _crop_for_refine(image, annotation.bbox, padding)
    target = annotation.target_description or annotation.request_text
    refined = refine_bbox_call(crop_b64, target, model)
    if refined is None:
        return annotation

    crop_w = right - left
    crop_h = bottom - top
    new_x = left + round(refined.x * crop_w)
    new_y = top + round(refined.y * crop_h)
    new_w = max(1, round(refined.width * crop_w))
    new_h = max(1, round(refined.height * crop_h))
    new_x = max(0, min(new_x, image.width - 1))
    new_y = max(0, min(new_y, image.height - 1))
    new_w = min(new_w, image.width - new_x)
    new_h = min(new_h, image.height - new_y)

    return annotation.model_copy(
        update={"bbox": Bbox(x=new_x, y=new_y, width=new_w, height=new_h)}
    )


def _refine_bboxes(
    image: Image.Image,
    annotations: list[Annotation],
    *,
    model: str,
    padding: float,
) -> list[Annotation]:
    if not annotations:
        return annotations
    rgb = image.convert("RGB")
    with ThreadPoolExecutor(max_workers=min(8, len(annotations))) as pool:
        return list(
            pool.map(
                lambda ann: _refine_one(rgb, ann, model=model, padding=padding),
                annotations,
            )
        )
