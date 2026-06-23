"""Token / request / cost accounting for an annotation run.

Every provider call records its token usage into a shared :class:`UsageMeter`
(thread-safe, because the refine pass runs in parallel). At the end of a run the
CLI prints a one-block summary: requests, tokens, estimated dollars, wall time.

Token counts and request counts come straight from each SDK's response, so they
are exact. Dollar cost is an ESTIMATE: it multiplies token counts by a built-in
price table (USD per 1M tokens) that you can edit below or override at runtime
with the ``MARKER_PRICES`` env var (JSON: ``{"model-prefix": [in, out]}``).
When no price matches the model, cost is reported as unavailable rather than 0.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Optional

__all__ = ["UsageMeter", "price_for"]

# USD per 1,000,000 tokens, as (input_rate, output_rate). Matched by longest
# model-name prefix (case-insensitive). These are public list-price ESTIMATES —
# correct them here or via MARKER_PRICES if your billing differs.
_PRICES: dict[str, tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.0),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-3-pro": (1.25, 10.0),
    "claude-opus-4": (5.0, 25.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-haiku-4": (1.0, 5.0),
    "gpt-5.5": (1.25, 10.0),
    "gpt-5": (1.25, 10.0),
}


def _env_prices() -> dict[str, tuple[float, float]]:
    raw = os.environ.get("MARKER_PRICES")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return {
            str(k): (float(v[0]), float(v[1]))
            for k, v in data.items()
            if isinstance(v, (list, tuple)) and len(v) == 2
        }
    except Exception:
        return {}


def price_for(model: str) -> Optional[tuple[float, float]]:
    """(input_rate, output_rate) per 1M tokens for ``model``, or None if unknown."""
    name = (model or "").strip().lower()
    table = {**_PRICES, **_env_prices()}
    best: Optional[str] = None
    for key in table:
        if name.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return table[best] if best is not None else None


class UsageMeter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.input_tokens = 0
        self.output_tokens = 0  # includes thinking/reasoning tokens (billed as output)
        self.cached_tokens = 0
        self.cost = 0.0
        self.cost_complete = True  # False once any request's model has no price
        self.models: set[str] = set()

    def record(
        self,
        *,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
    ) -> None:
        with self._lock:
            self.requests += 1
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.cached_tokens += cached_tokens
            if model:
                self.models.add(model)
            price = price_for(model)
            if price is None:
                self.cost_complete = False
            else:
                in_rate, out_rate = price
                self.cost += input_tokens / 1e6 * in_rate
                self.cost += output_tokens / 1e6 * out_rate

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def summary_lines(self, elapsed_seconds: float) -> list[str]:
        cached = f", {self.cached_tokens:,} cached" if self.cached_tokens else ""
        if self.requests and self.cost_complete:
            cost = f"≈ ${self.cost:.4f} (est.)"
        elif self.requests:
            cost = "n/a (no price for: " + ", ".join(sorted(self.models)) + ")"
        else:
            cost = "n/a"
        return [
            "Run summary:",
            f"  requests : {self.requests}",
            f"  tokens   : {self.total_tokens:,} "
            f"({self.input_tokens:,} in, {self.output_tokens:,} out{cached})",
            f"  cost     : {cost}",
            f"  time     : {elapsed_seconds:.1f}s",
        ]
