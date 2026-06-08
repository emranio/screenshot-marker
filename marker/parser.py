from __future__ import annotations

from typing import Any

from .models import (
    Annotation,
    Bbox,
    NormalizedBbox,
    RawAnnotation,
    RawAnnotationResponse,
)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


# Bboxes covering a larger share of the image than this are almost always
# "I have no idea, here's the whole thing" hallucinations. Treat them as
# not_found rather than drawing a giant rectangle over the screenshot.
MAX_BBOX_AREA_FRACTION = 0.90


def _to_pixels_bbox(b: NormalizedBbox, width: int, height: int) -> Bbox | None:
    x = _clamp(round(b.x * width), 0, width - 1)
    y = _clamp(round(b.y * height), 0, height - 1)
    w = _clamp(round(b.width * width), 1, width - x)
    h = _clamp(round(b.height * height), 1, height - y)
    if w <= 0 or h <= 0:
        return None
    return Bbox(x=x, y=y, width=w, height=h)


def _bbox_looks_hallucinated(bbox: Bbox, image_w: int, image_h: int) -> bool:
    image_area = image_w * image_h
    if image_area <= 0:
        return False
    return (bbox.width * bbox.height) / image_area >= MAX_BBOX_AREA_FRACTION


def _resolve_one(raw: RawAnnotation, width: int, height: int) -> Annotation:
    if raw.not_found:
        return Annotation(
            request_index=raw.request_index,
            request_text=raw.request_text,
            target_description=raw.target_description,
            label_text=raw.label_text,
            show_arrow=raw.show_arrow,
            color=raw.color,
            not_found=True,
            notes=raw.notes,
        )

    bbox = _to_pixels_bbox(raw.bbox, width, height) if raw.bbox else None
    label_position = (
        _to_pixels_bbox(raw.label_position, width, height)
        if raw.label_position and raw.label_text
        else None
    )
    not_found = bbox is None
    notes = raw.notes

    if bbox is not None and _bbox_looks_hallucinated(bbox, width, height):
        not_found = True
        bbox = None
        label_position = None
        pct = round(MAX_BBOX_AREA_FRACTION * 100)
        notes = (
            f"Model returned a bbox covering >={pct}% of the image — "
            "treating as not found to avoid drawing junk over the screenshot."
        )

    if not_found and not notes:
        notes = "Model returned no usable bbox."
    if not_found:
        label_position = None

    return Annotation(
        request_index=raw.request_index,
        request_text=raw.request_text,
        target_description=raw.target_description,
        label_text=raw.label_text,
        bbox=bbox,
        label_position=label_position,
        show_arrow=raw.show_arrow,
        color=raw.color,
        not_found=not_found,
        notes=notes,
    )


def parse_response(
    raw: dict[str, Any],
    queries: list[str],
    width: int,
    height: int,
) -> tuple[list[Annotation], list[str]]:
    """Validate, denormalize, and clamp model output. Returns (annotations, unresolved_query_texts)."""
    parsed = RawAnnotationResponse.model_validate(raw)

    by_index: dict[int, Annotation] = {}
    for raw_ann in parsed.annotations:
        ann = _resolve_one(raw_ann, width, height)
        by_index[ann.request_index] = ann

    annotations: list[Annotation] = []
    unresolved: list[str] = []
    for i, q in enumerate(queries):
        ann = by_index.get(i)
        if ann is None:
            ann = Annotation(
                request_index=i,
                request_text=q,
                target_description="",
                not_found=True,
                notes="Model omitted this request from its response.",
            )
        annotations.append(ann)
        if ann.not_found:
            unresolved.append(q)

    return annotations, unresolved
