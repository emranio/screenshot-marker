"""Claude (Anthropic) vision backend, built on the official ``anthropic`` SDK.

Mirrors the Codex and Gemini backends: one stateless call that takes images plus
a system/user prompt and returns JSON matching the given schema. The prompts and
flow are identical to the other providers — only the underlying agent changes.

Auth maps onto the two shared modes:

* ``api``  -> an Anthropic API key (``ANTHROPIC_API_KEY`` / ``CLAUDE_API_KEY``).
* ``auth`` -> a Claude subscription bearer token (``ANTHROPIC_AUTH_TOKEN`` /
              ``CLAUDE_CODE_OAUTH_TOKEN``), sent with the ``oauth-2025-04-20``
              beta header. Generate one with ``claude setup-token``. The SDK does
              NOT read a local ``claude`` login from disk, so a token must be set;
              native auth raises a clear error when none is found.

Structured JSON uses the Messages API ``output_config.format`` (json_schema), so
the package's existing JSON schemas are reused verbatim. Adaptive thinking is on
for spatial-reasoning quality, and ``reasoning_effort`` maps to ``effort``.
"""
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any

from ..config import get_claude_api_key, get_claude_auth_token, is_native_auth
from ..usage import UsageMeter

# Bearer (OAuth/subscription) tokens require this beta header on /v1/messages.
_OAUTH_BETA = "oauth-2025-04-20"
_MAX_TOKENS = 8192
# Let the SDK absorb transient 429/5xx with exponential backoff (honors
# Retry-After). Higher than the SDK default of 2 because subscription bearer
# tokens hit tighter per-window rate limits.
_MAX_RETRIES = 5

# Map the package's reasoning-effort levels onto Claude's output_config.effort.
_EFFORT_MAP = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}


def _load_anthropic() -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The Claude provider requires the anthropic SDK. "
            "Install it with `pip install anthropic` (or `pip install -r requirements.txt`)."
        ) from exc
    return anthropic


def _build_client(anthropic: Any, auth: str, api_key: str | None) -> tuple[Any, bool]:
    """Return (client, is_bearer_auth)."""
    if is_native_auth(auth):
        token = get_claude_auth_token()
        if not token:
            raise RuntimeError(
                "Claude native auth (MARKER_AUTH=auth) needs a subscription bearer "
                "token, but none was visible to this process. Create one with "
                "`claude setup-token`, then put it where the run can see it — most "
                "reliably as a line in the project .env file:\n"
                "    CLAUDE_CODE_OAUTH_TOKEN=<token>\n"
                "A shell `export CLAUDE_CODE_OAUTH_TOKEN=...` also works, but only in "
                "that same shell (a bare `VAR=...` without `export` is NOT inherited "
                "by run_tests.sh). ANTHROPIC_AUTH_TOKEN is accepted too. To use an API "
                "key instead, set MARKER_AUTH=api with ANTHROPIC_API_KEY."
            )
        return anthropic.Anthropic(auth_token=token, max_retries=_MAX_RETRIES), True
    return (
        anthropic.Anthropic(api_key=get_claude_api_key(api_key), max_retries=_MAX_RETRIES),
        False,
    )


def _record_usage(meter: UsageMeter | None, response: Any, model: str) -> None:
    """Record token usage from ``response.usage``.

    Anthropic already counts thinking tokens inside ``output_tokens``; cache-read
    input tokens are reported separately (and still part of ``input_tokens``).
    """
    if meter is None:
        return
    usage = getattr(response, "usage", None)
    if usage is None:
        meter.record(model=model)
        return
    meter.record(
        model=model,
        input_tokens=int(getattr(usage, "input_tokens", None) or 0),
        output_tokens=int(getattr(usage, "output_tokens", None) or 0),
        cached_tokens=int(getattr(usage, "cache_read_input_tokens", None) or 0),
    )


def _image_block(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    mime = mimetypes.guess_type(resolved.name)[0] or "image/png"
    data = base64.standard_b64encode(resolved.read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}


def call_json(
    *,
    image_paths: list[str | Path],
    system_prompt: str,
    user_text: str,
    model: str,
    output_schema: dict[str, Any],
    auth: str,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
    meter: UsageMeter | None = None,
) -> dict[str, Any]:
    anthropic = _load_anthropic()
    client, is_bearer = _build_client(anthropic, auth, api_key)

    content: list[dict[str, Any]] = [_image_block(path) for path in image_paths]
    content.append({"type": "text", "text": f"{user_text}\n\nReturn JSON only."})

    output_config: dict[str, Any] = {
        "format": {"type": "json_schema", "schema": output_schema}
    }
    effort = _EFFORT_MAP.get((reasoning_effort or "").strip().lower())
    if effort:
        output_config["effort"] = effort

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": _MAX_TOKENS,
        "system": system_prompt,
        "messages": [{"role": "user", "content": content}],
        "output_config": output_config,
        "thinking": {"type": "adaptive"},
    }
    if is_bearer:
        kwargs["extra_headers"] = {"anthropic-beta": _OAUTH_BETA}

    try:
        response = client.messages.create(**kwargs)
    except anthropic.RateLimitError as exc:
        raise RuntimeError(
            "Claude rate limit hit (HTTP 429) after automatic retries. Subscription "
            "bearer tokens (`claude setup-token`) have tighter limits than API keys. "
            "Wait for the limit window to reset and retry, run a smaller job (one "
            "image, add --no-refine, drop --steps/--crop), or switch to MARKER_AUTH=api "
            "with ANTHROPIC_API_KEY."
        ) from exc

    _record_usage(meter, response, model)

    if getattr(response, "stop_reason", None) == "refusal":
        raise RuntimeError("Claude declined the request (stop_reason=refusal).")
    text = next(
        (block.text for block in response.content if getattr(block, "type", None) == "text"),
        None,
    )
    if not text:
        raise RuntimeError(
            f"Claude returned no text content (stop_reason={getattr(response, 'stop_reason', None)})."
        )
    return json.loads(text)
