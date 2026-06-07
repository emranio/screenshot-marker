from __future__ import annotations

import base64
import io
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from PIL import Image

from .config import (
    AuthMode,
    DEFAULT_COLOR,
    DEFAULT_MODEL,
    auto_stroke_width,
    resolve_auth_mode,
    resolve_reasoning_effort,
    resolve_font_path,
)
from .drawing import render
from .models import (
    Annotation,
    AnnotationResult,
    Bbox,
)
from .parser import parse_response
from .vision import (
    call_vision,
    image_size,
    refine_bbox_call,
    run_annotation_step,
)

__all__ = [
    "annotate",
    "Annotation",
    "AnnotationResult",
    "Bbox",
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
    auth: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    steps: bool = False,
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

    resolved_auth = resolve_auth_mode(auth)
    resolved_effort = resolve_reasoning_effort(reasoning_effort)
    image_path = Path(image_path)
    output_path = Path(output_path) if output_path is not None else _default_output_path(image_path)

    if steps:
        return _annotate_with_steps(
            image_path=image_path,
            queries=queries,
            output_path=output_path,
            model=model,
            color=color,
            stroke_width=stroke_width,
            font_path=font_path,
            refine=refine,
            refine_padding=refine_padding,
            auth=resolved_auth,
            api_key=api_key,
            reasoning_effort=resolved_effort,
        )

    return _annotate_once(
        image_path=image_path,
        queries=queries,
        output_path=output_path,
        model=model,
        color=color,
        stroke_width=stroke_width,
        font_path=font_path,
        refine=refine,
        refine_padding=refine_padding,
        auth=resolved_auth,
        api_key=api_key,
        reasoning_effort=resolved_effort,
    )


def _annotate_once(
    *,
    image_path: Path,
    queries: list[str],
    output_path: Path,
    model: str,
    color: str,
    stroke_width: Optional[int],
    font_path: Optional[str],
    refine: bool,
    refine_padding: float,
    auth: AuthMode,
    api_key: str | None,
    reasoning_effort: str,
) -> AnnotationResult:
    width, height = image_size(image_path)
    raw = call_vision(
        image_path,
        width,
        height,
        queries,
        model,
        auth=auth,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
    )
    annotations, unresolved = parse_response(raw, queries, width, height)

    if refine:
        with Image.open(image_path) as src:
            annotations = _refine_bboxes(
                src,
                annotations,
                model=model,
                padding=refine_padding,
                auth=auth,
                api_key=api_key,
                reasoning_effort=reasoning_effort,
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


def _annotate_with_steps(
    *,
    image_path: Path,
    queries: list[str],
    output_path: Path,
    model: str,
    color: str,
    stroke_width: Optional[int],
    font_path: Optional[str],
    refine: bool,
    refine_padding: float,
    auth: AuthMode,
    api_key: str | None,
    reasoning_effort: str,
) -> AnnotationResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width, height = image_size(image_path)

    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}-steps-",
        dir=output_path.parent,
    ) as tmpdir:
        tmp_root = Path(tmpdir)
        candidate_path = tmp_root / "candidate.png"
        candidate = _annotate_once(
            image_path=image_path,
            queries=queries,
            output_path=candidate_path,
            model=model,
            color=color,
            stroke_width=stroke_width,
            font_path=font_path,
            refine=refine,
            refine_padding=refine_padding,
            auth=auth,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
        step = run_annotation_step(
            rendered_image_path=candidate_path,
            queries=queries,
            annotations_json=_annotations_json(candidate),
            width=width,
            height=height,
            model=model,
            auth=auth,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
        if step.get("decision") == "accept":
            return _promote_candidate(candidate, candidate_path, output_path)

        corrected = _apply_step_annotations(candidate, step, width, height)
        _render_result(
            image_path=image_path,
            output_path=output_path,
            annotations=corrected.annotations,
            color=color,
            stroke_width=stroke_width,
            font_path=font_path,
        )
        return corrected.model_copy(update={"output_path": output_path})


def _annotations_json(result: AnnotationResult) -> str:
    return result.model_dump_json(indent=2)


def _promote_candidate(
    result: AnnotationResult,
    candidate_path: Path,
    output_path: Path,
) -> AnnotationResult:
    candidate_path.replace(output_path)
    return result.model_copy(update={"output_path": output_path})


def _render_result(
    *,
    image_path: Path,
    output_path: Path,
    annotations: list[Annotation],
    color: str,
    stroke_width: Optional[int],
    font_path: Optional[str],
) -> None:
    width, height = image_size(image_path)
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


def _apply_step_annotations(
    current: AnnotationResult,
    step: dict[str, object],
    width: int,
    height: int,
) -> AnnotationResult:
    raw_annotations = step.get("annotations")
    if not isinstance(raw_annotations, list):
        return current

    by_index: dict[int, Annotation] = {}
    for raw in raw_annotations:
        if not isinstance(raw, dict):
            continue
        try:
            ann = Annotation.model_validate(raw)
        except Exception:
            continue
        if ann.not_found or ann.bbox is None:
            by_index[ann.request_index] = ann.model_copy(update={"bbox": None, "not_found": True})
            continue
        by_index[ann.request_index] = ann.model_copy(
            update={"bbox": _clamp_pixel_bbox(ann.bbox, width, height), "not_found": False}
        )

    annotations: list[Annotation] = []
    unresolved: list[str] = []
    for current_ann in current.annotations:
        corrected = by_index.get(current_ann.request_index, current_ann)
        if corrected.request_text != current_ann.request_text:
            corrected = corrected.model_copy(update={"request_text": current_ann.request_text})
        annotations.append(corrected)
        if corrected.not_found:
            unresolved.append(corrected.request_text)

    return current.model_copy(update={"annotations": annotations, "unresolved": unresolved})


def _clamp_pixel_bbox(bbox: Bbox, width: int, height: int) -> Bbox:
    x = max(0, min(bbox.x, width - 1))
    y = max(0, min(bbox.y, height - 1))
    w = max(1, min(bbox.width, width - x))
    h = max(1, min(bbox.height, height - y))
    return Bbox(x=x, y=y, width=w, height=h)


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
    auth: AuthMode,
    api_key: str | None,
    reasoning_effort: str,
) -> Annotation:
    if annotation.not_found or annotation.bbox is None:
        return annotation

    crop_b64, left, top, right, bottom = _crop_for_refine(image, annotation.bbox, padding)
    target = annotation.target_description or annotation.request_text
    refined = refine_bbox_call(
        crop_b64,
        target,
        model,
        auth=auth,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
    )
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
    auth: AuthMode,
    api_key: str | None,
    reasoning_effort: str,
) -> list[Annotation]:
    if not annotations:
        return annotations
    rgb = image.convert("RGB")
    max_workers = 2
    with ThreadPoolExecutor(max_workers=min(max_workers, len(annotations))) as pool:
        return list(
            pool.map(
                lambda ann: _refine_one(
                    rgb,
                    ann,
                    model=model,
                    padding=padding,
                    auth=auth,
                    api_key=api_key,
                    reasoning_effort=reasoning_effort,
                ),
                annotations,
            )
        )
