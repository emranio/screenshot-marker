from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, cast

_FALLBACK_MODEL = "gpt-5.5"
_FALLBACK_AUTH_MODE = "codex"
_FALLBACK_REASONING_EFFORT = "medium"

AuthMode = Literal["codex", "api"]
AUTH_MODES: tuple[AuthMode, ...] = ("codex", "api")
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = (
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
)


def load_env() -> None:
    """Best-effort .env loading. Silent if python-dotenv isn't installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


load_env()

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", _FALLBACK_MODEL)
DEFAULT_AUTH_MODE: AuthMode = "codex"
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
    model: str = DEFAULT_MODEL
    auth: AuthMode = DEFAULT_AUTH_MODE
    reasoning_effort: ReasoningEffort = DEFAULT_REASONING_EFFORT
    color: str = DEFAULT_COLOR
    stroke_width: Optional[int] = None
    font_path: Optional[str] = None


def resolve_auth_mode(auth: str | None = None) -> AuthMode:
    load_env()
    value = auth if auth is not None else os.environ.get("MARKER_AUTH", _FALLBACK_AUTH_MODE)
    normalized = value.strip().lower()
    if normalized not in AUTH_MODES:
        choices = ", ".join(AUTH_MODES)
        raise ValueError(f"Unsupported auth mode {value!r}. Expected one of: {choices}.")
    return cast(AuthMode, normalized)


def resolve_reasoning_effort(effort: str | None = None) -> ReasoningEffort:
    load_env()
    value = (
        effort
        if effort is not None
        else os.environ.get("OPENAI_REASONING_EFFORT", _FALLBACK_REASONING_EFFORT)
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
            "For subscription auth, use MARKER_AUTH=codex and run `codex login`."
        )
    return key


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
