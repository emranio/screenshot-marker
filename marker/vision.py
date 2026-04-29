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

For each annotation request you receive, locate the target UI element in the
screenshot and return its bounding box. The renderer draws the rectangle,
arrow, and label automatically — your job is JUST to locate and tag.

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
   case set bbox = null and put a one-line reason in notes
   (e.g., "no element matching 'export button' is visible").
   DO NOT GUESS. DO NOT return a near-full-image bbox as a placeholder.
   DO NOT pick the closest-looking element. An empty annotation is far better
   than a wrong one. Renderer will skip not_found items, so they cost
   nothing.
   When not_found is false, bbox MUST be a populated object that genuinely
   contains the target.
8. notes is always a string — use "" when not_found is false and you have
   nothing to add.

============================================================
LOCATING THE TARGET — PRECISION RULES
============================================================

- Determine each of the four edges INDEPENDENTLY. Find the target's LEFT edge
  (smallest x), TOP edge (smallest y), RIGHT edge (largest x), BOTTOM edge
  (largest y). Then x = left, y = top, width = right - left,
  height = bottom - top. Do NOT estimate x and width together — that
  produces a systematic horizontal offset.
- For bordered elements (cards, panels, sections, table rows, modals,
  buttons): align bbox edges WITH the visible border stroke. The drawn
  rectangle should land on top of the existing border line — not inside the
  content area, not floating in the surrounding margin.
- Include the element's full extent — header / title bar, internal padding,
  pinned footer all belong inside the bbox.
- For elements without a visible border (text labels, icons, plain regions):
  align tightly to the outer bounds of the visible content. Do not include
  surrounding empty space.
- Sanity-check: the four corners of your bbox must land on the target's
  corners, not in a neighbour or in white space.

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
      "color": null,
      "not_found": false,
      "notes": ""
    }
  ]
}

Notice: label_text is the quoted phrase only, not the whole sentence. color
is null (request said "red" generically, which is the default — only return
a hex string when the user names a non-default color like "blue card" or
"green box"). All required fields are present. JSON only, no prose.

Return ONE annotation per request, in input order, no extras, no omissions."""


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

- Determine the four edges INDEPENDENTLY: find the target's left edge, top
  edge, right edge, bottom edge. Then x = left, y = top,
  width = right - left, height = bottom - top.
- For bordered elements (cards, panels, sections, rows): align to the
  visible border stroke. The bbox should sit on the existing border line —
  not inside the content area, not in the surrounding margin.
- Include the full element — header / title bar, internal padding, footer.
- Do not include surrounding empty space outside the element.
- Sanity-check: the four corners of the bbox must land on the target's
  corners."""


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
