# screenshot-marker

A small Python tool that takes a UI screenshot plus one or more **plain-English
annotation requests** and produces an annotated image: translucent red
outlines, opaque arrows, and labelled callouts. A vision LLM (default:
GPT‑5.5) handles spatial reasoning; Pillow handles the final rendering.

```text
tests/screens/test_2.webp + "rectangle around the 'Site restructure' row labeled 'Latest annotation'"
                         + "rectangle around the Annotations tab labeled 'Active tab'"
                                             ↓
                tests/rendered/test_2.png  (smooth translucent outlines,
                                           procedural bent arrows, capsule labels)
```

- One LLM round‑trip per call regardless of how many queries you pass.
- Free‑form natural‑language queries — the model both parses intent and
  locates the region.
- Modular: vision call, JSON parsing, drawing, fixture runner, and CLI are
  separate units.
- No HTTP server, no web UI — just a Python module and a thin CLI.
- Always returns structured JSON (Python API and CLI both).
- Safety: when the model can't confidently locate a target, the request is
  reported as unresolved and **nothing is drawn** — no junk annotations.

---

## Install

Requires Python 3.10+ (3.13 recommended). By default, the tool uses your
existing Codex subscription via local Codex auth, not an OpenAI API key.

```bash
git clone <this repo>
cd screenshot-marker
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Sign in to Codex once, then select subscription auth:

```bash
codex login
export MARKER_AUTH=codex
unset OPENAI_API_KEY CODEX_API_KEY
```

Use the ChatGPT/Codex subscription login path in `codex login`. If you
previously authenticated Codex with an API key, run `codex logout` and sign in
again with ChatGPT before using `MARKER_AUTH=codex`.

Optional `.env` values:

```env
# .env
MARKER_AUTH=codex
OPENAI_MODEL=gpt-5.5
OPENAI_REASONING_EFFORT=medium
```

You can change `OPENAI_MODEL` any time — both the CLI and the Python API
pick it up automatically. `OPENAI_REASONING_EFFORT` is passed to the Agents SDK
Codex extension and defaults to `medium`. Higher effort can improve difficult
spatial reasoning, but it will be slower.

If you prefer OpenAI Platform API-key billing, keep using the Agents SDK path
and switch auth mode:

```bash
export MARKER_AUTH=api
export CODEX_API_KEY=sk-...   # OPENAI_API_KEY also works
```

---

## CLI usage

```bash
.venv/bin/python annotate.py \
  --image tests/screens/test_1.jpeg \
  --output tests/rendered/test_1.png \
  --query "red box on the customer info section, label 'Customer Details'" \
  --query "rectangle around the payment timeline labeled 'Activity Log'"
```

Output: writes the annotated PNG to the requested path, prints
the full JSON result to **stdout**, and a one‑line human summary to
**stderr**.

```text
$ python annotate.py --image tests/screens/test_2.webp \
    --output tests/rendered/test_2.png \
    --query "rectangle around the Annotations tab labeled 'Active tab'" \
  > result.json
Wrote tests/rendered/test_2.png  (1/1 resolved, 0 unresolved)
$ jq '.annotations[0].bbox' result.json
{
  "x": 258,
  "y": 99,
  "width": 92,
  "height": 42
}
```

### CLI flags

| Flag | Default | Notes |
|---|---|---|
| `--image PATH` | (required) | Input image. JPEG, PNG, or WebP. |
| `--output PATH` | `<image_dir>/<stem>_annotated.png` | Where to write the annotated PNG. Optional. |
| `--query "..."` | — | A natural‑language annotation request. Repeatable. |
| `--queries-file PATH` | — | A JSON array of query strings, or a saved annotation result JSON from `tests/annotations`. Combine with `--query` if you want. |
| `--model NAME` | `$OPENAI_MODEL` or `gpt-5.5` | Override the vision model for this run. |
| `--reasoning-effort minimal\|low\|medium\|high\|xhigh` | `$OPENAI_REASONING_EFFORT` or `medium` | Codex model reasoning effort. |
| `--auth codex\|api` | `$MARKER_AUTH` or `codex` | `codex` uses local `codex login`; `api` passes `CODEX_API_KEY` / `OPENAI_API_KEY` through the Agents SDK. |
| `--color HEX` | `#DC2626` | Default annotation color. |
| `--stroke INT` | auto‑scaled | Stroke width in pixels. Scales with `sqrt(min(w,h)) × 0.27` if omitted. |
| `--font PATH` | system default | Path to a TrueType font file. |
| `--no-refine` | off | Skip the per‑bbox refinement pass (faster, less accurate). |
| `--validate` | off | Ask a validator model to inspect the rendered output and rerun once with validator feedback if needed. Slower. |
| `--validator-reruns N` | `1` | Maximum validator-driven marker reruns when `--validate` is enabled. |
| `--allow-unresolved` | off | Exit `0` even if the model couldn't resolve some queries. Default exit is `1`. |

---

## Python API

```python
from marker import annotate

result = annotate(
    image_path="tests/screens/test_1.jpeg",
    output_path="tests/rendered/test_1.png",
    queries=[
        "red box on the customer info section, label 'Customer Details'",
        "rectangle around the payment timeline labeled 'Activity Log'",
    ],
    # all kwargs below are optional:
    model="gpt-5.5",
    auth="codex",
    reasoning_effort="medium",
    color="#DC2626",
    stroke_width=None,
    font_path=None,
    refine=True,
    refine_padding=0.15,
    validate=False,
    validator_reruns=1,
)

# AnnotationResult is a Pydantic model — JSON-shaped:
print(result.model_dump_json(indent=2))
print(result.output_path)         # Path to the written PNG
print(len(result.unresolved))     # number of queries the model couldn't resolve
```

### Result schema

The Python API and the CLI both yield the same JSON structure:

```jsonc
{
  "output_path": "tests/rendered/test_1.png",
  "annotations": [
    {
      "request_index": 0,
      "request_text": "red box on the customer info section, label 'Customer Details'",
      "target_description": "Customer Information card on the right side",
      "label_text": "Customer Details",
      "bbox": { "x": 1247, "y": 412, "width": 1180, "height": 320 },
      "color": null,
      "not_found": false,
      "notes": ""
    },
    {
      "request_index": 1,
      "request_text": "annotate the export button",
      "target_description": "",
      "label_text": null,
      "bbox": null,
      "color": null,
      "not_found": true,                                 // ← safety
      "notes": "No 'export' button is visible in this screenshot."
    }
  ],
  "unresolved": ["annotate the export button"]
}
```

`bbox` is in absolute pixel coordinates of the input image. `null` means the
request was not resolved.

### Test fixtures

```text
tests/
├── screens/       # source screenshots
├── annotations/   # annotation JSON only
└── rendered/      # annotated output images
```

Run the local fixture set with:

```bash
./run_tests.sh --allow-unresolved
```

The runner loops over `tests/screens/*`, reads request text from the matching
`tests/annotations/<stem>.json`, writes updated images to `tests/rendered/`,
and replaces each annotation JSON with the fresh CLI result after a successful
run. Screens without a matching annotation JSON are skipped.

---

## Writing good queries

Queries are free‑form English. The vision model extracts three things from
each query: (a) the target element, (b) the optional caption text, and (c)
the optional color.

| You write | Model extracts |
|---|---|
| `red box on the customer info card, label 'Customer Details'` | target = customer info card · label_text = "Customer Details" · color = default red |
| `rectangle around the payment timeline labeled 'Activity Log'` | target = payment timeline · label_text = "Activity Log" |
| `green outline around the 'Save' button` | target = "Save" button · color = `#22C55E` |
| `highlight the search bar` | target = search bar · label_text = null (no caption) |
| `box on the avatar at top‑right` | target = top‑right avatar |

Tips:

- **Quote the caption** if you want a label drawn (`labeled 'X'`,
  `label "X"`, `with caption X`). Otherwise just the rectangle is drawn.
- **Use distinguishing words** — "the **payment timeline** card" beats
  "the timeline" when there are multiple timeline‑like elements on screen.
- Specifying `red` is redundant (it's the default). Mention a color only
  when you want a non‑default one (`green`, `blue`, `#1F8FFF`).

---

## How it works

```text
   image  ─┐
           ▼
   ┌──── image_size ──────┐
   │  dimensions only     │
   └──────────┬───────────┘
              ▼
   ┌──── call_vision (1 LLM call, all queries) ────┐
   │  → JSON: bbox, label_text, color per query     │
   └──────────┬─────────────────────────────────────┘
              ▼
   ┌──── parse_response ────┐
   │  validate, denormalise │
   │  clamp, sanity-check   │  ← drops giant hallucinated bboxes
   └──────────┬─────────────┘
              ▼
   ┌──── refine_bboxes (optional, parallel, 1 LLM call/bbox) ────┐
   │  crop around each rough bbox + 15% padding,                  │
   │  ask the model for a precise bbox in the crop,               │
   │  map back to image coords                                    │
   └──────────┬───────────────────────────────────────────────────┘
              ▼
   ┌──── render (Pillow) ─────────┐
   │  antialiased outline         │
   │  blurred capsule label       │
   │  procedural curved arrow     │
   └──────────┬───────────────────┘
              ▼
        annotated PNG
```

- **Two‑pass refinement** — each rough bbox is tightened by a second
  per‑target LLM call on a cropped view of the image. This is what makes
  edges line up with card borders on tall / dense screenshots. Disable
  with `--no-refine` (or `refine=False` in Python) to halve the LLM cost.
- **Sanity check** — if the model returns a bbox covering ≥90% of the
  image area, that's the typical "I have no idea, here's the whole
  thing" hallucination. The parser converts it to `not_found` so nothing
  gets drawn.
- **Auto‑arrow & label placement** — you don't tell the model where the
  caption should go. The renderer picks bottom‑left under the bbox by
  default and falls back to top‑left, bottom‑right, or top‑right when
  there isn't room. The arrow is drawn from the capsule edge to the nearest
  bbox edge, so the tail always starts at the annotation text and the head
  always points at the target.
- **Procedural arrows** — arrows are drawn on demand from start/end geometry
  using a quadratic Bezier curve and an open stroked arrowhead. The curve,
  tangent, and head angle are computed per annotation; no pre-rendered arrow
  bitmap is rotated or scaled during rendering.
- **Text bbox breathing room** — very tight text/link/heading boxes get a
  small render-time expansion so outlines don't cut through glyphs or sit too
  low on the line.

---

## Module layout

```text
screenshot-marker/
├── annotate.py             # CLI entry point
├── run_tests.sh            # Local fixture runner
├── marker/
│   ├── __init__.py         # Public API: annotate()
│   ├── vision.py           # OpenAI call, prompt, schema
│   ├── parser.py           # JSON → typed annotations + sanity check
│   ├── drawing.py          # Pillow rendering: outline, arrow, label, bg
│   ├── models.py           # Pydantic schemas
│   └── config.py           # Defaults + env loading
├── tests/
│   ├── screens/            # Source screenshots
│   ├── annotations/        # Annotation result JSON sidecars
│   └── rendered/           # Rendered annotated PNG outputs
├── requirements.txt
└── .env.example
```

---

## Tuning notes

- **Bigger / denser screenshots benefit from refinement.** Leave it on.
- **Low‑resolution screenshots** (under ~600px on the short edge) can lose
  precision because both passes process them at low detail. Upscale before
  feeding in if you need pixel‑perfect alignment.
- Lower-capability or lower-effort settings are faster but can be noticeably
  less precise. Use `gpt-5.5` with `medium` effort as the baseline for bbox
  quality.
- **Stroke width** is auto‑scaled with image size. Rectangle outlines render
  thinner than arrows/labels (`max(2.5, round(stroke × 0.45))`) so the box
  stays readable without overpowering the screenshot.
- **Rectangle outlines are intentionally translucent.** Arrows are opaque so
  the pointer remains crisp.
- **The label text uses white on a translucent colored capsule**. Change the
  capsule / arrow / outline color via `--color` (or `color=` in Python).

---

## Limitations

- One image per `annotate()` call.
- Vision models can still pick the wrong element on dense or visually
  similar UI. Verify the output before publishing.
- Very tall / wide screenshots (≫4000 px on either axis) cost more tokens
  and may need refinement disabled if you hit OpenAI rate limits.
