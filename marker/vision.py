from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

from PIL import Image

from .config import get_api_key
from .models import NormalizedBbox

_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
}


def encode_image(path: str | Path) -> tuple[str, str, int, int]:
    """Returns (base64_data, mime_type, width, height)."""
    path = Path(path)
    with Image.open(path) as img:
        width, height = img.size
        fmt = (img.format or "PNG").upper()
        mime = _FORMAT_TO_MIME.get(fmt)
        if mime is None:
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="PNG")
            data = buf.getvalue()
            mime = "image/png"
        else:
            data = path.read_bytes()
    return base64.b64encode(data).decode("ascii"), mime, width, height


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

You receive a screenshot and a numbered list of annotation requests written
in free-form natural language. For each request you must:

1. Locate the target UI element described in the request.
2. Extract the user's desired LABEL TEXT — the short caption to render next
   to the highlighted element. Look for explicit phrases like
   "label 'Customer Details'", "labeled 'Activity Log'", "with caption ...",
   or "tagged ...". If the user did not specify any label text, return null.
3. Return the bbox in NORMALIZED coordinates [0.0, 1.0], where (0, 0) is
   the top-left of the image and (1, 1) is the bottom-right. The renderer
   converts these to absolute pixels.

PRECISION RULES — these matter as much as locating the right element:

- Determine each of the four edges INDEPENDENTLY. Find the target's LEFT edge
  (smallest x), TOP edge (smallest y), RIGHT edge (largest x), BOTTOM edge
  (largest y), then compute x = left, y = top, width = right - left,
  height = bottom - top. Do not estimate x and width together — that produces
  systematic horizontal offset.
- For bordered elements (cards, panels, sections, table rows, modals): align
  bbox edges WITH the visible border stroke. The drawn rectangle should sit
  on top of the existing border line, not inside the content area and not
  floating in the surrounding margin.
- Include the element's full extent — header/title bar, padding inside the
  border, and any pinned footer all belong inside the bbox.
- For elements without a visible border (text labels, icons, plain regions):
  align tightly to the outer bounds of the visible content; do not include
  surrounding empty space.
- Sanity-check: the four corners of your bbox must land on the target's
  corners, not in a neighbor or in white space.

You DO NOT need to return arrow positions or label placement — the renderer
computes those automatically from the bbox and image dimensions.

If you cannot locate the target with reasonable confidence, set
"not_found": true, leave bbox as null, and explain in "notes". Always set
"color" to null unless the user explicitly requested a non-red color.

Return only valid JSON matching the provided schema. Always include all
requests in your response, in the original order, with the original
request_index and request_text echoed back."""


def build_user_prompt(queries: list[str], width: int, height: int) -> str:
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
    image_b64: str,
    mime: str,
    width: int,
    height: int,
    queries: list[str],
    model: str,
) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=get_api_key())
    user_text = build_user_prompt(queries, width, height)
    data_url = f"data:{mime};base64,{image_b64}"

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                ],
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "annotations",
                "strict": True,
                "schema": RESPONSE_SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Vision model returned an empty response.")
    return json.loads(content)


REFINE_SYSTEM_PROMPT = """You will be shown a CROPPED region of a UI
screenshot. The target element described in the user message is fully visible
in this crop, surrounded by a small margin.

Return a precise bbox for that target in NORMALIZED coordinates of the CROP
(0–1, where (0,0) is the top-left of the crop and (1,1) is the bottom-right).

Rules:
- Determine the four edges INDEPENDENTLY: find the target's left edge, top
  edge, right edge, bottom edge. Then x = left, y = top, width = right - left,
  height = bottom - top.
- For bordered elements (cards, panels, sections, rows): align to the visible
  border stroke. The bbox should sit on the existing border line, not inside
  the content area and not in the surrounding margin.
- Include the full element — header/title bar, internal padding, and any
  footer all belong inside the bbox.
- Do not include surrounding empty space outside the element.
- Sanity-check: the four corners of the bbox must land on the target's
  corners.

Return only JSON: {"x": ..., "y": ..., "width": ..., "height": ...}."""


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
) -> NormalizedBbox | None:
    """Ask the model for a tight bbox of ``target_description`` within a cropped image."""
    from openai import OpenAI

    client = OpenAI(api_key=get_api_key())
    data_url = f"data:image/png;base64,{crop_b64}"

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REFINE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Target element: {target_description}"},
                        {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
                    ],
                },
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "refined_bbox",
                    "strict": True,
                    "schema": _REFINE_RESPONSE_SCHEMA,
                },
            },
        )
        content = response.choices[0].message.content
        if not content:
            return None
        return NormalizedBbox.model_validate_json(content)
    except Exception:
        return None
