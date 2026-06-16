from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image

from .config import get_codex_api_key, resolve_auth_mode, resolve_reasoning_effort
from .models import NormalizedBbox

def image_size(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as img:
        return img.size


_NORMALIZED_BBOX_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "width": {"type": "number"},
        "height": {"type": "number"},
    },
    "required": ["x", "y", "width", "height"],
    "additionalProperties": False,
}

_ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "request_index": {"type": "integer"},
        "request_text": {"type": "string"},
        "target_description": {"type": "string"},
        "label_text": {"type": ["string", "null"]},
        "bbox": _NORMALIZED_BBOX_SCHEMA,
        "label_position": _NORMALIZED_BBOX_SCHEMA,
        "color": {"type": ["string", "null"]},
        "not_found": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": [
        "request_index",
        "request_text",
        "target_description",
        "label_text",
        "bbox",
        "label_position",
        "color",
        "not_found",
        "notes",
    ],
    "additionalProperties": False,
}

_PIXEL_BBOX_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "width": {"type": "integer"},
        "height": {"type": "integer"},
    },
    "required": ["x", "y", "width", "height"],
    "additionalProperties": False,
}

_PIXEL_ANNOTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "request_index": {"type": "integer"},
        "request_text": {"type": "string"},
        "target_description": {"type": "string"},
        "label_text": {"type": ["string", "null"]},
        "bbox": _PIXEL_BBOX_SCHEMA,
        "label_position": _PIXEL_BBOX_SCHEMA,
        "color": {"type": ["string", "null"]},
        "not_found": {"type": "boolean"},
        "notes": {"type": "string"},
    },
    "required": [
        "request_index",
        "request_text",
        "target_description",
        "label_text",
        "bbox",
        "label_position",
        "color",
        "not_found",
        "notes",
    ],
    "additionalProperties": False,
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "annotations": {
            "type": "array",
            "items": _ANNOTATION_SCHEMA,
        }
    },
    "required": ["annotations"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You are a UI screenshot annotation assistant.

For each annotation request you receive, locate the target UI element in the
screenshot and return its bounding box. When a label is requested, also choose
a safe label position that avoids important UI content. You do NOT decide
arrows — the renderer draws them automatically.

============================================================
OUTPUT FORMAT — FOLLOW EXACTLY. NO EXCEPTIONS.
============================================================

1. Return JSON ONLY. No prose, no markdown, no code fences, no commentary.
2. Match the schema exactly. EVERY required field must appear in EVERY
   annotation object, even when the value is null or an empty string.
3. Return EXACTLY ONE annotation per request, in the original order.
   request_index = the [N] number from the input. request_text = the request
   string copied verbatim (do not paraphrase).
4. Coordinates are NORMALIZED floats in the range [0.0, 1.0]. They are NOT
   pixels, NOT percentages, NOT 0–100. (0, 0) is the top-left of the image
   and (1, 1) is the bottom-right.
   Example: a target whose pixel rect is (100, 50, 400, 200) on a 1000×800
   image becomes {"x": 0.10, "y": 0.0625, "width": 0.40, "height": 0.25}.
5. color MUST be one of:
     (a) null  — preferred. Use this when the request did not request a
         specific color. The renderer applies the default red.
     (b) A 7-character hex string starting with "#", e.g. "#DC2626",
         "#1F8FFF", "#22C55E".
   NEVER a CSS color name ("red", "blue", "green"), NEVER an "rgb(...)"
   string, NEVER three-digit shorthand like "#f00". Hex or null only.
6. label_text MUST be one of:
     (a) null  — when the request did not specify a caption.
     (b) The exact short caption inside the user's quotes. Look for cues
         like   label 'X' / labeled "X" / with caption X / tagged X.
         Return ONLY the caption text — never the whole request sentence.
   Keep it under ~40 characters when possible.
7. not_found = true is the CORRECT answer whenever the target is missing,
   ambiguous, or you are not confident you can locate it precisely. In that
   case set bbox = null, label_position = null, and put a one-line reason in
   notes (e.g., "no element matching 'export button' is visible").
   DO NOT GUESS. DO NOT return a near-full-image bbox as a placeholder.
   DO NOT pick the closest-looking element. An empty annotation is far better
   than a wrong one. Renderer will skip not_found items, so they cost
   nothing.
   When not_found is false, bbox MUST be a populated object that genuinely
   contains the target.
8. label_position:
     (a) null when label_text is null.
     (b) A NORMALIZED rectangle for the desired label capsule when label_text
         is present. It is the label's own top-left x/y and approximate
         width/height, NOT the target bbox. Do not cover the requested target,
         primary buttons, form fields, menus, headings, readable body text,
         icons the user needs to inspect, or another annotation label.
   PLACE THE LABEL NEAR THE BBOX, AT A MEANINGFUL, READABLE DISTANCE — close
   enough that it clearly belongs to the target, but NOT glued to the border.
   The label capsule should sit just off the target's nearest edge — directly
   above, below, left, or right of the bbox — separated by a small, clear gap
   (aim for ~3-5% of the image dimension): enough open space that the capsule
   does not touch the bbox border and a short connector arrow between them reads
   cleanly. Do NOT cram the capsule against the box (no hairline or overlapping
   gap), and do NOT float it far away. A comfortable, deliberate gap is the
   goal, not the smallest possible one.
   Pick the edge that has free whitespace nearest the target: check
   above/below/left/right of the bbox in that priority order and use the FIRST
   side that has room for the capsule without covering other important content.
   Do NOT push the label into distant empty space, a far corner, the page
   margin, or across the screenshot just because more open room exists there.
   A label far from its target is WRONG even if the gap area looks emptier —
   nearness to the bbox beats roominess. Only move the label one capsule-width
   further out (still on the nearest viable side) if every immediately-adjacent
   side would overlap protected content.
   Arrows are added automatically by the renderer (on by default whenever a
   label sits apart from its target). Do NOT reason about arrows — there is no
   arrow field to return.
9. notes is always a string — use "" when not_found is false and you have
   nothing to add.

============================================================
LOCATING THE TARGET — PRECISION RULES
============================================================

- FIRST identify the exact requested target before thinking about coordinates.
  If the request says "row", "card", "button", "tab", "input", or a quoted
  label, find that actual UI object, not a nearby label, icon, column, group,
  container, or empty region. If multiple similar objects exist and the request
  does not disambiguate them, return not_found=true.
- Determine each of the four edges INDEPENDENTLY. Find the target's LEFT edge
  (smallest x), TOP edge (smallest y), RIGHT edge (largest x), BOTTOM edge
  (largest y). Then x = left, y = top, width = right - left,
  height = bottom - top. Do NOT estimate x and width together — that
  produces a systematic horizontal offset.
- The bbox MUST be anchored on the target's visual edges. Do not return a
  rectangle that is merely near the target, centered around the target with
  extra margin, offset toward the label/callout location, or shifted into blank
  whitespace. A box that floats beside the target is wrong even if it overlaps
  the correct row/card/button.
- For bordered elements (cards, panels, sections, table rows, modals,
  buttons): align bbox edges WITH the visible border stroke. The drawn
  rectangle should land on top of the existing border line — not inside the
  content area, not floating in the surrounding margin.
- For table/list rows: include the entire row height from row separator to row
  separator and the row's full horizontal extent within the table/list, unless
  the request asks for one cell or one piece of text.
- For tabs and buttons: include the clickable control bounds, including its
  background/pill/border if visible. Do not box only the text unless the
  request explicitly asks for the text.
- Include the element's full extent — header / title bar, internal padding,
  pinned footer all belong inside the bbox.
- For elements without a visible border (text labels, icons, plain regions):
  align tightly to the outer bounds of the visible content. Do not include
  surrounding empty space.
- For text, links, headings, and placeholders: return the full visual line
  box, not just the ink/glyph bounds. Include ascenders, descenders,
  underline, and a small amount of natural line-height breathing room. If
  unsure, bias the top edge slightly upward and the bottom edge slightly
  downward rather than returning a short box that cuts through the text.
- Sanity-check: the four corners of your bbox must land on the target's
  corners, not in a neighbour or in white space.
- Final overlay check before returning: imagine drawing the rectangle. If any
  side would visibly miss the target edge, cut through the wrong object, or
  sit in whitespace with the target outside/partly outside the rectangle,
  correct the bbox. If you cannot confidently correct it, return not_found=true.

============================================================
EXAMPLE — input/output shape (for ONE request)
============================================================

Input request:
  [0] red box on the customer info card, label 'Customer Details'

Correct response:
{
  "annotations": [
    {
      "request_index": 0,
      "request_text": "red box on the customer info card, label 'Customer Details'",
      "target_description": "Customer Information card on the right column of the page",
      "label_text": "Customer Details",
      "bbox": {"x": 0.52, "y": 0.18, "width": 0.42, "height": 0.12},
      "label_position": {"x": 0.52, "y": 0.34, "width": 0.16, "height": 0.04},
      "color": null,
      "not_found": false,
      "notes": ""
    }
  ]
}

Notice: the label sits below the card with a visible gap; the renderer adds the
arrow automatically — you return no arrow field. label_text is the quoted phrase
only, not the whole sentence. color
is null (request said "red" generically, which is the default — only return
a hex string when the user names a non-default color like "blue card" or
"green box"). All required fields are present. JSON only, no prose.

Return ONE annotation per request, in input order, no extras, no omissions."""


def build_user_prompt(
    queries: list[str],
    width: int,
    height: int,
) -> str:
    lines = [
        f"Image dimensions: {width}px wide x {height}px tall.",
        f"Number of annotation requests: {len(queries)}.",
        "",
        "Requests:",
    ]
    for i, q in enumerate(queries):
        lines.append(f"  [{i}] {q}")
    lines.append("")
    lines.append(
        "Return one annotation object per request, with request_index matching "
        "the bracketed number above and request_text echoing the request verbatim."
    )
    return "\n".join(lines)


def call_vision(
    image_path: str | Path,
    width: int,
    height: int,
    queries: list[str],
    model: str,
    *,
    auth: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    resolved_auth = resolve_auth_mode(auth)
    resolved_effort = resolve_reasoning_effort(reasoning_effort)
    user_text = build_user_prompt(
        queries,
        width,
        height,
    )
    return _call_codex_json(
        image_paths=[image_path],
        system_prompt=SYSTEM_PROMPT,
        user_text=user_text,
        model=model,
        output_schema=RESPONSE_SCHEMA,
        auth=resolved_auth,
        api_key=api_key,
        reasoning_effort=resolved_effort,
    )


_CODEX_STRIPPED_ENV_KEYS = {"OPENAI_API_KEY", "CODEX_API_KEY"}
_CODEX_IDLE_TIMEOUT_SECONDS = 300


def _codex_subprocess_env(
    source: dict[str, str] | None = None,
) -> dict[str, str]:
    env = dict(os.environ if source is None else source)
    for key in _CODEX_STRIPPED_ENV_KEYS:
        env.pop(key, None)
    return env


def _load_codex_sdk() -> tuple[Any, Any, Any]:
    try:
        from agents.extensions.experimental.codex import Codex, ThreadOptions, TurnOptions
    except ImportError as exc:
        raise RuntimeError(
            "The Agents SDK Codex path requires openai-agents>=0.17.4. "
            "Install requirements.txt, run `codex login`, and retry."
        ) from exc
    return Codex, ThreadOptions, TurnOptions


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def _build_codex_prompt(system_prompt: str, user_text: str) -> str:
    return "\n\n".join(
        [
            system_prompt,
            user_text,
            "Return JSON only. Do not modify files or run shell commands.",
        ]
    )


def _call_codex_json(
    *,
    image_paths: list[str | Path],
    system_prompt: str,
    user_text: str,
    model: str,
    output_schema: dict[str, Any],
    auth: str,
    api_key: str | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    return _run_async(
        _call_codex_json_async(
            image_paths=image_paths,
            system_prompt=system_prompt,
            user_text=user_text,
            model=model,
            output_schema=output_schema,
            auth=auth,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
    )


async def _call_codex_json_async(
    *,
    image_paths: list[str | Path],
    system_prompt: str,
    user_text: str,
    model: str,
    output_schema: dict[str, Any],
    auth: str,
    api_key: str | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    Codex, ThreadOptions, TurnOptions = _load_codex_sdk()
    resolved_image_paths = [Path(path).expanduser().resolve() for path in image_paths]
    resolved_api_key = get_codex_api_key(api_key) if auth == "api" else None
    codex = Codex(env=_codex_subprocess_env(), api_key=resolved_api_key)
    thread = codex.start_thread(
        ThreadOptions(
            model=model,
            sandbox_mode="read-only",
            approval_policy="never",
            web_search_mode="disabled",
            skip_git_repo_check=True,
            model_reasoning_effort=reasoning_effort,
        )
    )
    inputs: list[dict[str, str]] = [
        {"type": "text", "text": _build_codex_prompt(system_prompt, user_text)}
    ]
    inputs.extend(
        {"type": "local_image", "path": str(image_path)}
        for image_path in resolved_image_paths
    )
    turn = await thread.run(
        inputs,
        TurnOptions(
            output_schema=output_schema,
            idle_timeout_seconds=_CODEX_IDLE_TIMEOUT_SECONDS,
        ),
    )
    content = getattr(turn, "final_response", None)
    if not content:
        raise RuntimeError("Agents SDK Codex path returned an empty response.")
    return json.loads(content)


REFINE_SYSTEM_PROMPT = """You are refining a bounding box on a CROPPED region
of a UI screenshot. The target element named in the user message is fully
visible inside this crop, with a small margin around it.

OUTPUT FORMAT — FOLLOW EXACTLY:

1. Return JSON ONLY. No prose, no markdown, no code fences.
2. Schema:
     {"x": <number>, "y": <number>, "width": <number>, "height": <number>}
   All four fields required. No extra fields.
3. Coordinates are NORMALIZED floats in [0.0, 1.0] of the CROP (NOT pixels,
   NOT 0–100). (0, 0) is the top-left of the crop, (1, 1) is the
   bottom-right of the crop.
   Example: a target filling the middle 80% of the crop width and starting
   10% from the top would be
     {"x": 0.10, "y": 0.10, "width": 0.80, "height": 0.30}.
4. width and height must be > 0.

LOCATING RULES:

- The crop includes margin around the rough bbox. The crop itself is NOT the
  target. Return the target's true edges inside the crop and leave the visible
  margin outside the bbox. If there is 10% margin on the left, x should be near
  0.10, not 0.0.
- First identify the exact target element named by the user. Do not lock onto
  a nearby label, icon, neighboring row/card, or empty padding inside the crop.
- Determine the four edges INDEPENDENTLY: find the target's left edge, top
  edge, right edge, bottom edge. Then x = left, y = top,
  width = right - left, height = bottom - top.
- The bbox MUST be anchored on the target's visual edges. Do not return a
  rectangle that is merely near the target, centered around it with extra
  margin, shifted toward whitespace, or offset onto a neighbor.
- For bordered elements (cards, panels, sections, rows): align to the
  visible border stroke. The bbox should sit on the existing border line —
  not inside the content area, not in the surrounding margin.
- For table/list rows: include the entire row height from separator to
  separator and the row's full horizontal extent inside the table/list, unless
  the target is explicitly one cell or text span.
- For tabs and buttons: include the clickable control bounds, including its
  background/pill/border if visible. Do not box only the text unless the
  request explicitly asks for text.
- Include the full element — header / title bar, internal padding, footer.
- Do not include surrounding empty space outside the element.
- For text, links, headings, and placeholders: return the full visual line
  box, not only the glyph ink. Include underline/descenders and avoid a
  bbox that sits low or cuts through the text.
- Sanity-check: the four corners of the bbox must land on the target's
  corners, not on the crop boundary or in crop margin.
- Final overlay check before returning: imagine drawing the rectangle on the
  crop. If the target would appear shifted outside the rectangle, or the
  rectangle would float in whitespace beside the target, correct it. If you
  cannot confidently correct it, return the best precise target-edge bbox; do
  not return the whole crop."""


_REFINE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "width": {"type": "number"},
        "height": {"type": "number"},
    },
    "required": ["x", "y", "width", "height"],
    "additionalProperties": False,
}


def refine_bbox_call(
    crop_b64: str,
    target_description: str,
    model: str,
    *,
    auth: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> NormalizedBbox | None:
    """Ask the model for a tight bbox of ``target_description`` within a cropped image."""
    resolved_auth = resolve_auth_mode(auth)
    resolved_effort = resolve_reasoning_effort(reasoning_effort)
    try:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            tmp.write(base64.b64decode(crop_b64))
            tmp.flush()
            raw = _call_codex_json(
                image_paths=[tmp.name],
                system_prompt=REFINE_SYSTEM_PROMPT,
                user_text=f"Target element: {target_description}",
                model=model,
                output_schema=_REFINE_RESPONSE_SCHEMA,
                auth=resolved_auth,
                api_key=api_key,
                reasoning_effort=resolved_effort,
            )
        return NormalizedBbox.model_validate(raw)
    except Exception:
        return None


CROP_SYSTEM_PROMPT = """You are choosing a CROP RECTANGLE for an annotated UI
screenshot.

The image already has annotation boxes, labels, and optional arrows drawn on it.
Your job is to return ONE rectangle that frames all of the annotated content so
the final image can be cropped down to the region that matters — dropping the
irrelevant top/bottom/side margins of a tall or wide screenshot.

OUTPUT FORMAT — FOLLOW EXACTLY:

1. Return JSON ONLY. No prose, no markdown, no code fences.
2. Schema:
     {"x": <number>, "y": <number>, "width": <number>, "height": <number>}
   All four fields required. No extra fields.
3. Coordinates are NORMALIZED floats in [0.0, 1.0] of the FULL image. (0, 0) is
   the top-left, (1, 1) is the bottom-right. width and height must be > 0.

RULES:

- The rectangle MUST fully contain every drawn annotation box, its label
  capsule, and any arrow. NEVER clip a drawn element — when unsure, include
  more.
- The rectangle MUST also include all the RELATED and IMPORTANT visuals around
  the annotations — the surrounding UI that gives them meaning. Include the
  whole card / panel / section / table / form the annotation sits in, its
  header or title, the column or row headers needed to read it, and any nearby
  element the annotation refers to or depends on. The crop must still make sense
  on its own once the rest of the screenshot is gone.
- Do NOT crop tightly. Leave comfortable breathing room on EVERY side so the
  result does not look cramped: keep a band of surrounding UI / whitespace
  around the outermost annotation, not a rectangle hugging the boxes.
- When in doubt between a smaller and a larger rectangle, choose the LARGER one
  — losing context is worse than keeping a little extra.
- If the annotations (or the context they need) are spread across most of the
  image, just return the whole image: {"x": 0, "y": 0, "width": 1, "height": 1}."""


def _coerce_normalized_bbox(raw: dict[str, Any]) -> NormalizedBbox | None:
    try:
        x = float(raw["x"])
        y = float(raw["y"])
        w = float(raw["width"])
        h = float(raw["height"])
    except (KeyError, TypeError, ValueError):
        return None
    x = min(max(x, 0.0), 1.0)
    y = min(max(y, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0 - x)
    h = min(max(h, 0.0), 1.0 - y)
    if w <= 0 or h <= 0:
        return None
    return NormalizedBbox(x=x, y=y, width=w, height=h)


def determine_crop_region(
    image_path: str | Path,
    annotations_summary: str,
    width: int,
    height: int,
    model: str,
    *,
    auth: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> NormalizedBbox | None:
    """Ask the model for a focus rectangle framing all drawn annotations.

    Returns a normalized crop rectangle, or ``None`` if the model could not be
    reached or returned an unusable answer (the caller then falls back to the
    union of the drawn annotation rects).
    """
    resolved_auth = resolve_auth_mode(auth)
    resolved_effort = resolve_reasoning_effort(reasoning_effort)
    user_text = "\n".join(
        [
            f"Image dimensions: {width}px wide x {height}px tall.",
            "",
            "Annotated regions already drawn on the image (absolute pixels):",
            annotations_summary,
            "",
            "Return ONE normalized crop rectangle that frames all annotated "
            "content AND the related, important surrounding visuals (the card / "
            "panel / section / headers the annotations belong to), with "
            "comfortable margin on every side.",
        ]
    )
    try:
        raw = _call_codex_json(
            image_paths=[image_path],
            system_prompt=CROP_SYSTEM_PROMPT,
            user_text=user_text,
            model=model,
            output_schema=_REFINE_RESPONSE_SCHEMA,
            auth=resolved_auth,
            api_key=api_key,
            reasoning_effort=resolved_effort,
        )
    except Exception:
        return None
    return _coerce_normalized_bbox(raw)


_STEP_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["accept", "redraw"]},
        "notes": {"type": "string"},
        "annotations": {
            "type": "array",
            "items": _PIXEL_ANNOTATION_SCHEMA,
        },
    },
    "required": ["decision", "notes", "annotations"],
    "additionalProperties": False,
}

STEP_SYSTEM_PROMPT = """You are a strict QA/correction step for UI screenshot annotations.

You receive one rendered annotated screenshot and the full current annotation
JSON. The rendered image contains the original screenshot plus boxes, labels,
and optional arrows drawn from the JSON's absolute pixel coordinates.

Your job is not to write feedback for another marker pass. Your job is to
decide whether the current JSON can be accepted or to directly correct the
bbox and label_position values yourself so the renderer can redraw the image
immediately.

Return JSON only.

Arrows are NOT your concern — the renderer draws them automatically (on by
default whenever a label sits apart from its target). There is no arrow field
to return; do not reason about arrows.

Decision rules:

- Return decision="accept" only when every requested target is correctly
  identified, each bbox is anchored to the target's visual edges, and each label
  is in a clear position.
- Return decision="redraw" when any bbox is shifted, floating in whitespace,
  attached to a nearby label instead of the requested target, missing an edge,
  too tight, too loose, or covering a neighboring row/card/control.
- Return decision="redraw" when a label covers important UI content, covers its
  own target, or overlaps another label.
- Return decision="redraw" when a label sits farther from its target than
  necessary — in distant whitespace, a far corner, or the page margin — while a
  nearer edge of the bbox had room. Also redraw when a label is crammed against
  the box (touching it or with only a hairline gap); it should have a small,
  clear gap instead. When correcting, MOVE the label_position so the capsule
  sits just off the target's nearest free edge at a meaningful but modest
  distance — a small, clear gap (roughly 3-5% of the image dimension), not glued
  to the border and not parked in distant whitespace. Nearness to the bbox
  takes priority over picking the emptiest region.

Annotation correction rules:

- Always return a full annotations array, in the same order as the input JSON.
- Preserve request_index, request_text, label_text, color, and target_description
  unless the current value is plainly inconsistent with the visible target.
- Coordinates are ABSOLUTE PIXEL integers in the original screenshot coordinate
  system, not normalized floats.
- bbox is {x, y, width, height}. x/y are top-left. width/height are positive.
- label_position is null when label_text is null. Otherwise it is the desired
  label capsule rectangle in absolute pixels. Put it near the target — just off
  the bbox's nearest free edge with a small, clear gap (a meaningful but modest
  distance, not glued to the border) — not on important controls, readable text,
  icons, headings, the target itself, or another label. Do not park it in
  distant whitespace.
- Clamp all coordinates inside the image dimensions supplied in the prompt.
- For bordered elements, align to the visible border stroke.
- For rows/lists, include the whole row from separator to separator and the full
  row width inside the list/table unless the request asks for a cell or text.
- For buttons/tabs, include the clickable control bounds, not only text.
- For plain text targets, include the full visual text line box with natural
  line-height, not only glyph ink.
- If a requested target is missing or ambiguous, set not_found=true and bbox=null.

Final check: imagine redrawing from your returned JSON. If the rectangle would
still visibly miss the requested target edge, sit in whitespace, or if the label
would hide important UI, correct it before returning."""


def run_annotation_step(
    *,
    rendered_image_path: str | Path,
    queries: list[str],
    annotations_json: str,
    width: int,
    height: int,
    model: str,
    auth: str | None = None,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> dict[str, Any]:
    user_text = "\n".join(
        [
            "Images:",
            "1. Rendered annotated screenshot to validate/correct.",
            "",
            f"Image dimensions: {width}px wide x {height}px tall.",
            "",
            "Annotation requirements:",
            *_format_numbered_queries(queries),
            "",
            "Current annotation JSON:",
            annotations_json,
            "",
            "Return decision='accept' with the current annotations if the image is correct. "
            "Otherwise return decision='redraw' with corrected absolute-pixel annotations.",
        ]
    )
    return _call_json_with_images(
        image_paths=[rendered_image_path],
        system_prompt=STEP_SYSTEM_PROMPT,
        user_text=user_text,
        model=model,
        output_schema=_STEP_RESPONSE_SCHEMA,
        auth=auth,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
    )


def _format_numbered_queries(queries: list[str]) -> list[str]:
    return [f"[{i}] {query}" for i, query in enumerate(queries)]


def _call_json_with_images(
    *,
    image_paths: list[str | Path],
    system_prompt: str,
    user_text: str,
    model: str,
    output_schema: dict[str, Any],
    auth: str | None,
    api_key: str | None,
    reasoning_effort: str | None,
) -> dict[str, Any]:
    resolved_auth = resolve_auth_mode(auth)
    resolved_effort = resolve_reasoning_effort(reasoning_effort)
    return _call_codex_json(
        image_paths=image_paths,
        system_prompt=system_prompt,
        user_text=user_text,
        model=model,
        output_schema=output_schema,
        auth=resolved_auth,
        api_key=api_key,
        reasoning_effort=resolved_effort,
    )
