# screenshot-marker

A small Python tool that takes a UI screenshot plus one or more **plain-English
annotation requests** and produces an annotated image: translucent red
outlines, optional arrows, and labelled callouts. A vision LLM handles spatial
reasoning and label placement; Pillow handles the final rendering.

The vision backend is pluggable. Two **providers** ship today — `codex`
(OpenAI Agents SDK, default, `gpt-5.5`) and `gemini` (Google Gen AI SDK,
`gemini-2.5-pro`) — and each supports two auth modes: native credentials
(`auth`) or an API key (`api`). The prompt and flow are identical across
providers; only the underlying agent changes.

```text
tests/screens/test_2.webp + "rectangle around the 'Site restructure' row labeled 'Latest annotation'"
                         + "rectangle around the Annotations tab labeled 'Active tab'"
                                             ↓
                tests/rendered/test_2.png  (smooth translucent outlines,
                                           optional arrows, flat capsule labels)
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

Requires Python 3.10+ (3.13 recommended).

```bash
git clone <this repo>
cd screenshot-marker
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Pick a provider with `MARKER_PROVIDER` (default `codex`) and an auth mode with
`MARKER_AUTH`. Both `auth` (native credentials) and `api` (an API key) work for
every provider.

### Codex / OpenAI (default)

By default the tool uses your existing Codex subscription via local Codex auth,
not an OpenAI API key:

```bash
codex login                  # use the ChatGPT/Codex subscription login path
export MARKER_PROVIDER=codex
export MARKER_AUTH=auth       # native subscription auth
unset OPENAI_API_KEY CODEX_API_KEY
```

If you previously authenticated Codex with an API key, run `codex logout` and
sign in again with ChatGPT before using native auth. To use OpenAI Platform
API-key billing instead:

```bash
export MARKER_PROVIDER=codex
export MARKER_AUTH=api
export CODEX_API_KEY=sk-...   # OPENAI_API_KEY also works
```

`OPENAI_MODEL` (default `gpt-5.5`) and `OPENAI_REASONING_EFFORT` (default
`medium`) tune the Codex path. Higher effort can improve difficult spatial
reasoning, but it will be slower.

### Gemini (Google)

The Gemini provider uses the [Google Gen AI SDK](https://googleapis.github.io/python-genai/)
(`google-genai`), the same SDK the Google [Agent Development Kit](https://adk.dev/agents/models/google-gemini/)
uses for Gemini. Use a Google AI Studio API key:

```bash
export MARKER_PROVIDER=gemini
export MARKER_AUTH=api
export GEMINI_API_KEY=...     # GOOGLE_API_KEY also works
```

…or native auth via Vertex AI Application Default Credentials:

```bash
gcloud auth application-default login
export MARKER_PROVIDER=gemini
export MARKER_AUTH=auth
export GOOGLE_CLOUD_PROJECT=your-gcp-project
export GOOGLE_CLOUD_LOCATION=us-central1   # optional, defaults to us-central1
```

`GEMINI_MODEL` (default `gemini-2.5-pro`) selects the model. Newer models such
as `gemini-3-pro-preview` also work — set `GEMINI_MODEL` or pass `--model`.
`OPENAI_REASONING_EFFORT` is ignored by Gemini.

### Optional `.env`

See [`.env.example`](.env.example) for the full set. A minimal Gemini setup:

```env
MARKER_PROVIDER=gemini
MARKER_AUTH=api
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-pro
```

Both the CLI and the Python API pick these up automatically. The legacy value
`MARKER_AUTH=codex` still works and is treated as `auth`.

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
| `--provider codex\|gemini` | `$MARKER_PROVIDER` or `codex` | Vision backend. `codex` = OpenAI Agents SDK; `gemini` = Google Gen AI SDK. |
| `--model NAME` | provider default (`gpt-5.5` / `gemini-2.5-pro`) | Override the vision model for this run. Reads `$OPENAI_MODEL` / `$GEMINI_MODEL`. |
| `--reasoning-effort minimal\|low\|medium\|high\|xhigh` | `$OPENAI_REASONING_EFFORT` or `medium` | Codex/OpenAI reasoning effort. Ignored by Gemini. |
| `--auth auth\|api` | `$MARKER_AUTH` or provider default | `auth` uses native credentials (`codex login` / Vertex AI); `api` uses an API key. `codex` is accepted as a legacy alias for `auth`. |
| `--color HEX` | `#DC2626` | Default annotation color. |
| `--stroke INT` | auto‑scaled | Stroke width in pixels. Scales with `sqrt(min(w,h)) × 0.27` if omitted. |
| `--font PATH` | system default | Path to a TrueType font file. |
| `--no-refine` | off | Skip the per‑bbox refinement pass (faster, less accurate). |
| `--steps` | off | After the first render, ask a validator/corrector step to inspect the drawn image plus full JSON. If needed, it returns corrected pixel bboxes and the renderer redraws once. Slower. |
| `--crop` | off | After annotating, ask the model for a focus region around the drawn annotations **plus the related, important surrounding visuals** (the card/panel/section/headers they belong to) and crop the output PNG to it. The crop is unioned with every drawn box/label and padded generously, so it never clips an annotation and never looks tight. One extra LLM call. |
| `--no-arrow` | off | Never draw arrows. Arrows are on by default for any labeled annotation that sits apart from its target; this flag suppresses them all. The model does not decide arrows. |
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
    provider="codex",          # or "gemini"
    model=None,                 # defaults to the provider's model
    auth="auth",                # "auth" (native creds) or "api" (API key)
    reasoning_effort="medium",  # Codex/OpenAI only; ignored by Gemini
    color="#DC2626",
    stroke_width=None,
    font_path=None,
    refine=True,
    refine_padding=0.15,
    crop=False,
    crop_padding=0.12,
    draw_arrows=True,
    steps=False,
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
      "label_position": { "x": 1247, "y": 792, "width": 350, "height": 72 },
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
      "label_position": null,
      "color": null,
      "not_found": true,                                 // ← safety
      "notes": "No 'export' button is visible in this screenshot."
    }
  ],
  "unresolved": ["annotate the export button"]
}
```

`bbox` and `label_position` are in absolute pixel coordinates of the input
image. `null` means the request was not resolved or no label was requested.

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
   │  → JSON: bbox, label_position, arrow, color    │
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
   │  flat capsule label          │
   │  optional procedural arrow   │
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
- **Focus crop** (`--crop`) — after rendering, the model returns a focus
  rectangle framing the drawn annotations **and the related, important visuals
  around them** (the card/panel/section/headers they belong to), so the crop
  still makes sense on its own. Python unions that rectangle with every drawn
  box and label (so a stray model answer can never clip an annotation), pads it
  by `crop_padding` plus a pixel floor on every side, then crops the output PNG.
  The result trims dead margins off tall screenshots without ever hugging the
  boxes. If nothing resolved, the full image is kept.
- **Model-guided label placement** — when a caption is requested, the model
  returns a `label_position` rectangle near the target that avoids important UI
  content. Older JSON without this field still renders through the fallback
  auto-layout.
- **Arrows on by default** — a labeled box gets a procedural arrow from the
  capsule edge to the nearest bbox edge whenever the label sits apart from the
  target with any visible gap. The model never decides this; arrows are a pure
  render-time behavior. They are suppressed only when you pass `--no-arrow`
  (`draw_arrows=False`) or the capsule physically overlaps / sits flush against
  the box (where a pointer would be redundant).
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
│   ├── vision.py           # Prompts, schemas, provider dispatch, Codex backend
│   ├── providers/          # Vision backends (one per provider)
│   │   └── gemini.py       # Google Gen AI SDK backend
│   ├── parser.py           # JSON → typed annotations + sanity check
│   ├── drawing.py          # Pillow rendering: outline, arrow, label, bg
│   ├── models.py           # Pydantic schemas
│   └── config.py           # Provider/auth/model resolution + env loading
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
  less precise. Use a capable "pro" model as the baseline for bbox quality —
  `gpt-5.5` with `medium` effort for Codex, or `gemini-2.5-pro` for Gemini.
- **Stroke width** is auto‑scaled with image size. Rectangle outlines render
  thinner than arrows/labels (`max(2.5, round(stroke × 0.45))`) so the box
  stays readable without overpowering the screenshot.
- **Rectangle outlines are intentionally translucent.** Arrows are drawn only
  when needed.
- **The label text uses white on a flat translucent colored capsule**. Change
  the capsule / arrow / outline color via `--color` (or `color=` in Python).

---

## Limitations

- One image per `annotate()` call.
- Vision models can still pick the wrong element on dense or visually
  similar UI. Verify the output before publishing.
- Very tall / wide screenshots (≫4000 px on either axis) cost more tokens
  and may need refinement disabled if you hit OpenAI rate limits.
