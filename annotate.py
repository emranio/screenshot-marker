#!/usr/bin/env python3
"""CLI for the screenshot-marker tool.

Examples:

  python annotate.py --image tests/screens/test_1.jpeg --output tests/rendered/test_1.png --query "red box + arrow on the customer info card, label 'Customer Details'"   --query "rectangle around the payment timeline labeled 'Activity Log'"

  python annotate.py --image tests/screens/test_1.jpeg --output tests/rendered/test_1.png --queries-file tests/annotations/test_1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from marker import annotate
from marker.config import (
    AUTH_MODES,
    DEFAULT_COLOR,
    DEFAULT_PROVIDER,
    DEFAULT_REASONING_EFFORT,
    PROVIDERS,
    REASONING_EFFORTS,
    default_model_for,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotate a UI screenshot using a vision LLM."
    )
    p.add_argument("--image", required=True, type=Path, help="Path to the input image.")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write the annotated PNG. Defaults to <image_dir>/<stem>_annotated.png.",
    )
    p.add_argument(
        "--query",
        action="append",
        default=[],
        help="A natural-language annotation request. Repeatable.",
    )
    p.add_argument(
        "--queries-file",
        type=Path,
        help="Path to a JSON array of query strings, or a saved annotation result JSON.",
    )
    p.add_argument(
        "--provider",
        choices=PROVIDERS,
        default=None,
        help=(
            f"Vision backend. Defaults to $MARKER_PROVIDER or {DEFAULT_PROVIDER}. "
            f"Default models — codex: {default_model_for('codex')}; "
            f"gemini: {default_model_for('gemini')}; "
            f"claude: {default_model_for('claude')}."
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help="Vision model. Defaults to $MODEL or the selected provider's default model.",
    )
    p.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default=None,
        help=(
            "Reasoning effort, applied to every provider (Codex's "
            "model_reasoning_effort, Claude's effort, Gemini's thinking budget). "
            f"Defaults to $REASONING_EFFORT or {DEFAULT_REASONING_EFFORT}."
        ),
    )
    p.add_argument(
        "--auth",
        default=None,
        metavar="{%s}" % "|".join(AUTH_MODES),
        help=(
            "Auth mode for the selected provider: 'auth' uses native credentials "
            "(codex login / Vertex AI), 'api' uses an API key. Defaults to "
            "$MARKER_AUTH or the provider default."
        ),
    )
    p.add_argument("--color", default=DEFAULT_COLOR, help=f"Default annotation color (default: {DEFAULT_COLOR}).")
    p.add_argument("--stroke", type=int, default=None, help="Stroke width in pixels (auto-scaled if omitted).")
    p.add_argument("--font", default=None, help="Path to a TrueType font file.")
    p.add_argument(
        "--allow-unresolved",
        action="store_true",
        help="Exit 0 even if some queries could not be resolved.",
    )
    p.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip the per-bbox refinement pass (faster, less accurate).",
    )
    p.add_argument(
        "--crop",
        action="store_true",
        help=(
            "After annotating, ask the model for a focus region around the "
            "drawn annotations and crop the output PNG to it (with margin)."
        ),
    )
    p.add_argument(
        "--no-arrow",
        action="store_true",
        help="Never draw arrows. Arrows are drawn by default for labeled annotations.",
    )
    p.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stream per-stage progress (current/total step, status, timing) to "
            "stderr. On by default; use --no-progress to silence."
        ),
    )
    p.add_argument(
        "--steps",
        nargs="?",
        const=1,
        default=0,
        type=int,
        metavar="N",
        help=(
            "Run up to N review/correction passes after the first render "
            "(bare --steps = 1; omitted = 0; max 4). The loop stops early as soon "
            "as a pass accepts the result."
        ),
    )
    args = p.parse_args(argv)
    if not 0 <= args.steps <= 4:
        p.error("--steps must be between 0 and 4")
    return args


def _collect_queries(args: argparse.Namespace) -> list[str]:
    queries: list[str] = list(args.query)
    if args.queries_file:
        data = json.loads(args.queries_file.read_text())
        if isinstance(data, list) and all(isinstance(s, str) for s in data):
            queries.extend(data)
        elif isinstance(data, dict) and isinstance(data.get("annotations"), list):
            extracted = [
                ann.get("request_text")
                for ann in data["annotations"]
                if isinstance(ann, dict) and isinstance(ann.get("request_text"), str)
            ]
            if len(extracted) != len(data["annotations"]):
                raise SystemExit(
                    f"--queries-file annotations must all contain request_text: {args.queries_file}"
                )
            queries.extend(extracted)
        else:
            raise SystemExit(
                f"--queries-file must contain a JSON array of strings or an annotation result JSON: {args.queries_file}"
            )
    if not queries:
        raise SystemExit("Provide at least one --query or --queries-file.")
    return queries


def _stderr_progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    queries = _collect_queries(args)

    result = annotate(
        image_path=args.image,
        queries=queries,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        auth=args.auth,
        reasoning_effort=args.reasoning_effort,
        color=args.color,
        stroke_width=args.stroke,
        font_path=args.font,
        refine=not args.no_refine,
        crop=args.crop,
        draw_arrows=not args.no_arrow,
        steps=args.steps,
        on_progress=_stderr_progress if args.progress else None,
    )

    # Structured result on stdout (always JSON), human summary on stderr.
    print(result.model_dump_json(indent=2))

    resolved = len(result.annotations) - len(result.unresolved)
    total = len(result.annotations)
    print(
        f"Wrote {result.output_path}  ({resolved}/{total} resolved, "
        f"{len(result.unresolved)} unresolved)",
        file=sys.stderr,
    )
    if result.unresolved and not args.allow_unresolved:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
