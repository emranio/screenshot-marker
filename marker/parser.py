from __future__ import annotations

from typing import Any

from .models import (
    Annotation,
    Arrow,
    Bbox,
    Label,
    NormalizedArrow,
    NormalizedBbox,
    NormalizedLabel,
    Point,
    RawAnnotation,
    RawAnnotationResponse,
)


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _to_pixels_point(p, width: int, height: int) -> Point:
    return Point(
        x=_clamp(round(p.x * width), 0, width - 1),
        y=_clamp(round(p.y * height), 0, height - 1),
    )


def _to_pixels_bbox(b: NormalizedBbox, width: int, height: int) -> Bbox | None:
    x = _clamp(round(b.x * width), 0, width - 1)
    y = _clamp(round(b.y * height), 0, height - 1)
    w = _clamp(round(b.width * width), 1, width - x)
    h = _clamp(round(b.height * height), 1, height - y)
    if w <= 0 or h <= 0:
        return None
    return Bbox(x=x, y=y, width=w, height=h)


def _to_pixels_arrow(a: NormalizedArrow, width: int, height: int) -> Arrow:
    return Arrow(
        start=_to_pixels_point(a.start, width, height),
        end=_to_pixels_point(a.end, width, height),
    )


def _to_pixels_label(l: NormalizedLabel, width: int, height: int) -> Label:
    return Label(
        text=l.text,
        anchor=_to_pixels_point(l.anchor, width, height),
        placement=l.placement,
    )


def _resolve_one(raw: RawAnnotation, width: int, height: int) -> Annotation:
    if raw.not_found:
        return Annotation(
            request_index=raw.request_index,
            request_text=raw.request_text,
            target_description=raw.target_description,
            shape=raw.shape,
            color=raw.color,
            not_found=True,
            notes=raw.notes,
        )

    bbox = _to_pixels_bbox(raw.bbox, width, height) if raw.bbox else None
    arrow = _to_pixels_arrow(raw.arrow, width, height) if raw.arrow else None
    label = _to_pixels_label(raw.label, width, height) if raw.label else None

    not_found = bbox is None and arrow is None and label is None
    notes = raw.notes
    if not_found and not notes:
        notes = "Model returned no drawable geometry."

    return Annotation(
        request_index=raw.request_index,
        request_text=raw.request_text,
        target_description=raw.target_description,
        shape=raw.shape,
        bbox=bbox,
        arrow=arrow,
        label=label,
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
