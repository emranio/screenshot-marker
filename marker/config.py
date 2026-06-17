from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, cast

# ---------------------------------------------------------------------------
# Providers (vision backends)
# ---------------------------------------------------------------------------
# Each provider is a separate "agent SDK" backend that turns an image + prompt
# into structured JSON. Codex uses the OpenAI Agents SDK; Gemini uses the
# Google Gen AI SDK; Claude uses the Anthropic SDK. New providers plug in the
# same way.
Provider = Literal["codex", "gemini", "claude"]
PROVIDERS: tuple[Provider, ...] = ("codex", "gemini", "claude")
_FALLBACK_PROVIDER: Provider = "codex"

# ---------------------------------------------------------------------------
# Auth modes (shared across every provider)
# ---------------------------------------------------------------------------
#   "auth" -> the provider's native, non-API-key credentials
#               codex  : a local `codex login` (ChatGPT/Codex subscription)
#               gemini : Vertex AI via Google Cloud Application Default Creds
#               claude : a Claude subscription bearer token (ANTHROPIC_AUTH_TOKEN
#                        / CLAUDE_CODE_OAUTH_TOKEN) or a `claude`/`ant` login
#   "api"  -> an explicit API key
#               codex  : CODEX_API_KEY / OPENAI_API_KEY
#               gemini : GEMINI_API_KEY / GOOGLE_API_KEY
#               claude : ANTHROPIC_API_KEY / CLAUDE_API_KEY
AuthMode = Literal["auth", "api"]
AUTH_MODES: tuple[AuthMode, ...] = ("auth", "api")

# A provider's native (non-API-key) auth is selected by any of these values.
NATIVE_AUTH_MODES: tuple[str, ...] = ("auth",)

ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)

_FALLBACK_REASONING_EFFORT: ReasoningEffort = "medium"

# Model selection. A single shared MODEL env var overrides the active provider's
# built-in default below. (Each provider previously had its own var — OPENAI_MODEL
# / GEMINI_MODEL / CLAUDE_MODEL — but one MODEL is simpler and avoids the trap of
# setting the wrong provider's var; pass --model to override per run.)
_MODEL_ENV = "MODEL"
_FALLBACK_MODEL: dict[Provider, str] = {
    "codex": "gpt-5.5",
    "gemini": "gemini-2.5-pro",
    "claude": "claude-opus-4-8",
}

# Per-provider default auth mode when neither a flag nor MARKER_AUTH is set.
# Codex defaults to its subscription login; Gemini and Claude default to an API
# key (the common Google AI Studio / Anthropic API path).
_DEFAULT_AUTH: dict[Provider, AuthMode] = {
    "codex": "auth",
    "gemini": "api",
    "claude": "api",
}


def load_env() -> None:
    """Best-effort .env loading. Silent if python-dotenv isn't installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


load_env()

DEFAULT_PROVIDER: Provider = _FALLBACK_PROVIDER
# Default model for the default provider. Provider-specific resolution lives in
# resolve_model(); this is kept for callers/tests that want a plain default.
DEFAULT_MODEL = os.environ.get(_MODEL_ENV, _FALLBACK_MODEL[DEFAULT_PROVIDER])
DEFAULT_AUTH_MODE: AuthMode = _DEFAULT_AUTH[DEFAULT_PROVIDER]
DEFAULT_REASONING_EFFORT: ReasoningEffort = "medium"
DEFAULT_COLOR = "#DC2626"
DEFAULT_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "C:\\Windows\\Fonts\\arialbd.ttf",
]


@dataclass
class RenderDefaults:
    provider: Provider = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    auth: AuthMode = DEFAULT_AUTH_MODE
    reasoning_effort: ReasoningEffort = DEFAULT_REASONING_EFFORT
    color: str = DEFAULT_COLOR
    stroke_width: Optional[int] = None
    font_path: Optional[str] = None


def resolve_provider(provider: str | None = None) -> Provider:
    load_env()
    value = provider if provider is not None else os.environ.get("MARKER_PROVIDER", _FALLBACK_PROVIDER)
    normalized = value.strip().lower()
    if normalized not in PROVIDERS:
        choices = ", ".join(PROVIDERS)
        raise ValueError(f"Unsupported provider {value!r}. Expected one of: {choices}.")
    return cast(Provider, normalized)


def resolve_auth_mode(auth: str | None = None, provider: str | None = None) -> AuthMode:
    load_env()
    raw = auth if auth is not None else os.environ.get("MARKER_AUTH")
    if raw is None:
        # No explicit choice: fall back to the selected provider's default.
        return _DEFAULT_AUTH[resolve_provider(provider)]
    normalized = raw.strip().lower()
    if normalized not in AUTH_MODES:
        choices = ", ".join(AUTH_MODES)
        raise ValueError(f"Unsupported auth mode {raw!r}. Expected one of: {choices}.")
    return cast(AuthMode, normalized)


def is_native_auth(auth: str) -> bool:
    """True when ``auth`` selects a provider's native (non-API-key) credentials."""
    return auth in NATIVE_AUTH_MODES


def resolve_model(model: str | None = None, provider: str | None = None) -> str:
    load_env()
    if model:
        return model
    prov = resolve_provider(provider)
    return os.environ.get(_MODEL_ENV, _FALLBACK_MODEL[prov])


def default_model_for(provider: str | None = None) -> str:
    """The default model for ``provider`` (env override or built-in fallback)."""
    return resolve_model(None, provider)


def resolve_reasoning_effort(effort: str | None = None) -> ReasoningEffort:
    load_env()
    value = (
        effort
        if effort is not None
        else os.environ.get("REASONING_EFFORT", _FALLBACK_REASONING_EFFORT)
    )
    normalized = value.strip().lower()
    if normalized not in REASONING_EFFORTS:
        choices = ", ".join(REASONING_EFFORTS)
        raise ValueError(
            f"Unsupported reasoning effort {value!r}. Expected one of: {choices}."
        )
    return cast(ReasoningEffort, normalized)


def get_codex_api_key(api_key: str | None = None) -> str:
    load_env()
    key = api_key or os.environ.get("CODEX_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "API auth requires CODEX_API_KEY or OPENAI_API_KEY. "
            "For subscription auth, use MARKER_AUTH=auth and run `codex login`."
        )
    return key


def get_gemini_api_key(api_key: str | None = None) -> str:
    load_env()
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "API auth requires GEMINI_API_KEY or GOOGLE_API_KEY. "
            "For Vertex AI auth, use MARKER_AUTH=auth with GOOGLE_CLOUD_PROJECT set "
            "and `gcloud auth application-default login`."
        )
    return key


def get_claude_api_key(api_key: str | None = None) -> str:
    load_env()
    key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    if not key:
        raise RuntimeError(
            "API auth requires ANTHROPIC_API_KEY or CLAUDE_API_KEY. "
            "For subscription auth, use MARKER_AUTH=auth with ANTHROPIC_AUTH_TOKEN "
            "(or a `claude` / `ant` login)."
        )
    return key


def get_claude_auth_token() -> str | None:
    """Bearer token for Claude native auth, or None to use default credential resolution."""
    load_env()
    return os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")


def get_api_key(provider: str, api_key: str | None = None) -> str:
    """Resolve the API key for ``provider`` from the argument or environment."""
    prov = resolve_provider(provider)
    if prov == "gemini":
        return get_gemini_api_key(api_key)
    if prov == "claude":
        return get_claude_api_key(api_key)
    return get_codex_api_key(api_key)


def get_vertex_config() -> tuple[Optional[str], str]:
    """(project, location) for Gemini Vertex AI auth. Location defaults to us-central1."""
    load_env()
    project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT_ID")
    location = (
        os.environ.get("GOOGLE_CLOUD_LOCATION")
        or os.environ.get("GOOGLE_CLOUD_REGION")
        or "us-central1"
    )
    return project, location


def auto_stroke_width(image_width: int, image_height: int) -> int:
    base = min(image_width, image_height)
    return max(5, round(base**0.5 * 0.27))


def resolve_font_path(font_path: Optional[str]) -> Optional[str]:
    if font_path:
        if Path(font_path).is_file():
            return font_path
        raise FileNotFoundError(f"Font file not found: {font_path}")
    for candidate in DEFAULT_FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    return None
