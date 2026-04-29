#!/usr/bin/env python3
"""CLI for the screenshot-marker tool.

Examples:

  python annotate.py --image test_1.jpeg --output out.png --query "red box + arrow on the customer info card, label 'Customer Details'"   --query "rectangle around the payment timeline labeled 'Activity Log'"

  python annotate.py --image test.jpeg --output out.png --queries-file queries.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from marker import annotate
from marker.config import DEFAULT_COLOR, DEFAULT_MODEL


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
        help="Path to a JSON file containing a list of query strings.",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"Vision model (default: {DEFAULT_MODEL}).")
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
    return p.parse_args(argv)


def _collect_queries(args: argparse.Namespace) -> list[str]:
    queries: list[str] = list(args.query)
    if args.queries_file:
        data = json.loads(args.queries_file.read_text())
        if not isinstance(data, list) or not all(isinstance(s, str) for s in data):
            raise SystemExit(f"--queries-file must contain a JSON array of strings: {args.queries_file}")
        queries.extend(data)
    if not queries:
        raise SystemExit("Provide at least one --query or --queries-file.")
    return queries


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    queries = _collect_queries(args)

    result = annotate(
        image_path=args.image,
        queries=queries,
        output_path=args.output,
        model=args.model,
        color=args.color,
        stroke_width=args.stroke,
        font_path=args.font,
        refine=not args.no_refine,
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
