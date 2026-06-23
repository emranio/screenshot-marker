from __future__ import annotations

import base64
import io
import math
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from .config import (
    AuthMode,
    DEFAULT_COLOR,
    Provider,
    auto_stroke_width,
    resolve_auth_mode,
    resolve_model,
    resolve_provider,
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
from .progress import Progress
from .usage import UsageMeter
from .vision import (
    call_vision,
    determine_crop_region_for_queries,
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
    provider: str | None = None,
    model: Optional[str] = None,
    color: str = DEFAULT_COLOR,
    stroke_width: Optional[int] = None,
    font_path: Optional[str] = None,
    refine: bool = True,
    refine_padding: float = 0.15,
    crop: bool = False,
    crop_padding: float = 0.12,
    draw_arrows: bool = True,
    auth: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    steps: int = 0,
    on_progress: Optional[Callable[[str], None]] = None,
) -> AnnotationResult:
    """Annotate ``image_path`` with one or more natural-language ``queries``.

    A single vision-model call resolves all queries; the resulting rectangles,
    optional arrows, and labels are drawn onto the image and saved as PNG.

    ``provider`` selects the vision backend ("codex" or "gemini"); when omitted
    it falls back to ``$MARKER_PROVIDER`` (default "codex"). ``model`` defaults
    to that provider's default model. ``auth`` is "auth" (the provider's native
    login/credentials) or "api" (an API key).

    If ``output_path`` is not provided, the result is written next to the
    input as ``<image_stem>_annotated.png``.

    When ``refine`` is True (default), each located bbox is tightened by a
    second pass: the image is cropped around the rough bbox plus
    ``refine_padding`` and the model returns precise edges within the crop.
    This dramatically improves border alignment on tall screenshots at the
    cost of one extra LLM call per resolved query (run in parallel).

    Returns an :class:`AnnotationResult` with the resolved annotations, the
    output path, and the list of queries the model could not locate.

    When ``crop`` is True, a vision call runs FIRST (before locating anything):
    it picks a focus rectangle holding all the query targets plus surrounding
    context, the source image is cropped to it (with ``crop_padding`` margin so
    it is never tight), and the whole locate/refine/render pipeline then runs on
    that smaller crop. This sharply improves bbox accuracy on large/tall
    screenshots. Returned annotation coordinates are in the cropped image's
    space (which is what the saved output shows). Falls back to the full image
    when no usable region is returned.

    ``steps`` is the number of review/correction passes (0 disables review).
    Each pass re-renders the corrected annotations and the next pass validates
    that render; the loop stops early as soon as a pass accepts the result.

    ``on_progress`` is an optional callable that receives one preformatted
    ``[i/total] ...`` line per pipeline stage (and indented sub-steps). The CLI
    wires this to stderr so a run streams what it is doing; pass ``None`` (the
    default) to stay silent.
    """
    if not queries:
        raise ValueError("queries must contain at least one annotation request.")

    resolved_provider = resolve_provider(provider)
    resolved_model = resolve_model(model, resolved_provider)
    resolved_auth = resolve_auth_mode(auth, resolved_provider)
    resolved_effort = resolve_reasoning_effort(reasoning_effort)
    image_path = Path(image_path)
    output_path = Path(output_path) if output_path is not None else _default_output_path(image_path)

    progress = Progress(
        _plan_total(refine=refine, steps=steps, crop=crop),
        sink=on_progress,
    )
    meter = UsageMeter()
    started = time.monotonic()

    # Crop FIRST (before locating): the model frames the region holding the
    # query targets on the original image, we crop the source to it, and the
    # whole locate/refine/render pipeline then runs on that smaller crop —
    # bbox grounding is far more accurate on a focused image than on a tall or
    # large full screenshot. All result coordinates are in the cropped image's
    # space, which is exactly what the saved (cropped) output shows.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_path = image_path
    crop_tmp: "tempfile.TemporaryDirectory | None" = None
    if crop:
        work_path, crop_tmp = _crop_source_first(
            image_path,
            queries,
            output_path.parent,
            crop_padding=crop_padding,
            model=resolved_model,
            provider=resolved_provider,
            auth=resolved_auth,
            api_key=api_key,
            reasoning_effort=resolved_effort,
            progress=progress,
            meter=meter,
        )

    try:
        if steps:
            result = _annotate_with_steps(
                image_path=work_path,
                queries=queries,
                output_path=output_path,
                model=resolved_model,
                color=color,
                stroke_width=stroke_width,
                font_path=font_path,
                refine=refine,
                refine_padding=refine_padding,
                draw_arrows=draw_arrows,
                provider=resolved_provider,
                auth=resolved_auth,
                api_key=api_key,
                reasoning_effort=resolved_effort,
                steps=steps,
                progress=progress,
                meter=meter,
            )
        else:
            result = _annotate_once(
                image_path=work_path,
                queries=queries,
                output_path=output_path,
                model=resolved_model,
                color=color,
                stroke_width=stroke_width,
                font_path=font_path,
                refine=refine,
                refine_padding=refine_padding,
                draw_arrows=draw_arrows,
                provider=resolved_provider,
                auth=resolved_auth,
                api_key=api_key,
                reasoning_effort=resolved_effort,
                progress=progress,
                meter=meter,
            )
    finally:
        if crop_tmp is not None:
            crop_tmp.cleanup()

    progress.summary(meter.summary_lines(time.monotonic() - started))

    return result


def _plan_total(*, refine: bool, steps: int, crop: bool) -> int:
    """Planned stage count for the progress counter.

    The initial render always reads the image, locates the queries, and renders
    (3); refinement, each review pass, and the crop add one apiece. Review
    passes are the maximum — the loop may stop early, so the counter can finish
    below this total, which is fine.
    """
    total = 2  # read image + locate/parse
    if refine:
        total += 1
    total += 1  # render & save
    total += steps  # up to `steps` review passes
    if crop:
        total += 1
    return total


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
    draw_arrows: bool,
    provider: Provider,
    auth: AuthMode,
    api_key: str | None,
    reasoning_effort: str,
    progress: Progress,
    meter: UsageMeter,
) -> AnnotationResult:
    with progress.stage("Reading image"):
        width, height = image_size(image_path)
        progress.note(f"{width}×{height}px")

    with progress.stage(f"Locating {len(queries)} annotation(s)"):
        raw = call_vision(
            image_path,
            width,
            height,
            queries,
            model,
            provider=provider,
            auth=auth,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            meter=meter,
        )
        annotations, unresolved = parse_response(raw, queries, width, height)
        progress.note(
            f"{len(annotations) - len(unresolved)} resolved, {len(unresolved)} unresolved"
        )

    if refine:
        refinable = sum(
            1 for a in annotations if not a.not_found and a.bbox is not None
        )
        with progress.stage(f"Refining {refinable} box(es)"):
            with Image.open(image_path) as src:
                annotations = _refine_bboxes(
                    src,
                    annotations,
                    model=model,
                    padding=refine_padding,
                    provider=provider,
                    auth=auth,
                    api_key=api_key,
                    reasoning_effort=reasoning_effort,
                    progress=progress,
                    meter=meter,
                )

    with progress.stage("Rendering & saving"):
        stroke = stroke_width if stroke_width is not None else auto_stroke_width(width, height)
        font = resolve_font_path(font_path)

        with Image.open(image_path) as src:
            annotated = render(
                src,
                annotations,
                default_color=color,
                stroke_width=stroke,
                font_path=font,
                draw_arrows=draw_arrows,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated.save(output_path, format="PNG")
        progress.note(str(output_path))

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
    draw_arrows: bool,
    provider: Provider,
    auth: AuthMode,
    api_key: str | None,
    reasoning_effort: str,
    steps: int,
    progress: Progress,
    meter: UsageMeter,
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
            draw_arrows=draw_arrows,
            provider=provider,
            auth=auth,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            progress=progress,
            meter=meter,
        )

        # Up to ``steps`` review passes. Each pass validates the current render;
        # on "accept" we stop early, otherwise we apply its corrections, re-render
        # the candidate, and the next pass reviews that corrected image.
        total_passes = max(1, steps)
        for i in range(total_passes):
            with progress.stage(f"Review pass {i + 1}/{total_passes}"):
                step = run_annotation_step(
                    rendered_image_path=candidate_path,
                    queries=queries,
                    annotations_json=_annotations_json(candidate),
                    width=width,
                    height=height,
                    model=model,
                    provider=provider,
                    auth=auth,
                    api_key=api_key,
                    reasoning_effort=reasoning_effort,
                    meter=meter,
                )
                decision = step.get("decision")
                if decision == "accept":
                    progress.note("accepted — stopping early")
                    break

                progress.note("redraw — applying corrections")
                candidate = _apply_step_annotations(candidate, step, width, height)
                _render_result(
                    image_path=image_path,
                    output_path=candidate_path,
                    annotations=candidate.annotations,
                    color=color,
                    stroke_width=stroke_width,
                    font_path=font_path,
                    draw_arrows=draw_arrows,
                )
                candidate = candidate.model_copy(update={"output_path": candidate_path})

        return _promote_candidate(candidate, candidate_path, output_path)


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
    draw_arrows: bool = True,
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
            draw_arrows=draw_arrows,
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
            by_index[ann.request_index] = ann.model_copy(
                update={"bbox": None, "label_position": None, "not_found": True}
            )
            continue
        update = {"bbox": _clamp_pixel_bbox(ann.bbox, width, height), "not_found": False}
        if ann.label_position is not None:
            update["label_position"] = _clamp_pixel_bbox(ann.label_position, width, height)
        by_index[ann.request_index] = ann.model_copy(update=update)

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
    provider: Provider,
    auth: AuthMode,
    api_key: str | None,
    reasoning_effort: str,
    meter: UsageMeter | None = None,
) -> Annotation:
    if annotation.not_found or annotation.bbox is None:
        return annotation

    crop_b64, left, top, right, bottom = _crop_for_refine(image, annotation.bbox, padding)
    target = annotation.target_description or annotation.request_text
    refined = refine_bbox_call(
        crop_b64,
        target,
        model,
        provider=provider,
        auth=auth,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
        meter=meter,
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
    provider: Provider,
    auth: AuthMode,
    api_key: str | None,
    reasoning_effort: str,
    progress: Progress,
    meter: UsageMeter | None = None,
) -> list[Annotation]:
    if not annotations:
        return annotations
    rgb = image.convert("RGB")
    # Only resolvable boxes hit the model; not_found items pass through unchanged.
    targets = [i for i, a in enumerate(annotations) if not a.not_found and a.bbox is not None]
    if not targets:
        return annotations

    results = list(annotations)
    max_workers = 2
    done = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, len(targets))) as pool:
        futures = {
            pool.submit(
                _refine_one,
                rgb,
                annotations[i],
                model=model,
                padding=padding,
                provider=provider,
                auth=auth,
                api_key=api_key,
                reasoning_effort=reasoning_effort,
                meter=meter,
            ): i
            for i in targets
        }
        for future in as_completed(futures):
            i = futures[future]
            results[i] = future.result()
            done += 1
            label = results[i].label_text or results[i].request_text
            progress.note(f"{done}/{len(targets)}: {label[:60]}")
    return results


def _pad_and_clamp(
    rect: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    padding: float,
) -> tuple[int, int, int, int]:
    """Grow ``rect`` by ``padding`` (and a minimum) on every side, clamped to image."""
    x0, y0, x1, y1 = rect
    pad_x = max(round((x1 - x0) * padding), round(img_w * 0.03), 24)
    pad_y = max(round((y1 - y0) * padding), round(img_h * 0.03), 24)
    nx0 = max(0, int(math.floor(x0 - pad_x)))
    ny0 = max(0, int(math.floor(y0 - pad_y)))
    nx1 = min(img_w, int(math.ceil(x1 + pad_x)))
    ny1 = min(img_h, int(math.ceil(y1 + pad_y)))
    if nx1 <= nx0:
        nx1 = min(img_w, nx0 + 1)
    if ny1 <= ny0:
        ny1 = min(img_h, ny0 + 1)
    return nx0, ny0, nx1, ny1


def _crop_source_first(
    image_path: Path,
    queries: list[str],
    output_dir: Path,
    *,
    crop_padding: float,
    model: str,
    provider: Provider,
    auth: AuthMode,
    api_key: str | None,
    reasoning_effort: str,
    progress: Progress,
    meter: UsageMeter,
) -> tuple[Path, "tempfile.TemporaryDirectory | None"]:
    """Crop the SOURCE image down to the region holding the query targets, before
    any annotation runs. The model picks a focus region from the request text on
    the original image; locating the boxes on the smaller crop is far more
    accurate than on a tall/large full screenshot.

    Returns ``(working_image_path, tmpdir)``. ``tmpdir`` is the temp directory
    holding the cropped image (the caller cleans it up once the run is done) or
    ``None`` when no crop was applied — in which case the original ``image_path``
    is returned and the pipeline runs on the full image as before.
    """
    with progress.stage("Choosing crop region"):
        width, height = image_size(image_path)
        region = determine_crop_region_for_queries(
            image_path,
            queries,
            width,
            height,
            model,
            provider=provider,
            auth=auth,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
            meter=meter,
        )
        if region is None:
            progress.note("no usable region — annotating full image")
            return image_path, None

        rect = (
            region.x * width,
            region.y * height,
            (region.x + region.width) * width,
            (region.y + region.height) * height,
        )
        box = _pad_and_clamp(rect, width, height, crop_padding)
        if box == (0, 0, width, height):
            progress.note("region spans full image — no crop")
            return image_path, None

        tmpdir = tempfile.TemporaryDirectory(
            prefix=f".{image_path.stem}-crop-",
            dir=output_dir,
        )
        cropped_path = Path(tmpdir.name) / "source.png"
        with Image.open(image_path) as src:
            src.crop(box).save(cropped_path, format="PNG")
        progress.note(f"cropped to {box[2] - box[0]}×{box[3] - box[1]}px")
        return cropped_path, tmpdir
