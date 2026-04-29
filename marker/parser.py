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


def _to_pixels_bbox(b: NormalizedBbox, width: int, height: int) -> Bbox | None:
    x = _clamp(round(b.x * width), 0, width - 1)
    y = _clamp(round(b.y * height), 0, height - 1)
    w = _clamp(round(b.width * width), 1, width - x)
    h = _clamp(round(b.height * height), 1, height - y)
    if w <= 0 or h <= 0:
        return None
    return Bbox(x=x, y=y, width=w, height=h)


def _resolve_one(raw: RawAnnotation, width: int, height: int) -> Annotation:
    if raw.not_found:
        return Annotation(
            request_index=raw.request_index,
            request_text=raw.request_text,
            target_description=raw.target_description,
            label_text=raw.label_text,
            color=raw.color,
            not_found=True,
            notes=raw.notes,
        )

    bbox = _to_pixels_bbox(raw.bbox, width, height) if raw.bbox else None
    not_found = bbox is None
    notes = raw.notes
    if not_found and not notes:
        notes = "Model returned no usable bbox."

    return Annotation(
        request_index=raw.request_index,
        request_text=raw.request_text,
        target_description=raw.target_description,
        label_text=raw.label_text,
        bbox=bbox,
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
