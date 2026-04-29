from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_FALLBACK_MODEL = "gpt-5.4"


def load_env() -> None:
    """Best-effort .env loading. Silent if python-dotenv isn't installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


load_env()

DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", _FALLBACK_MODEL)
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
    color: str = DEFAULT_COLOR
    stroke_width: Optional[int] = None
    font_path: Optional[str] = None


def get_api_key() -> str:
    load_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or a .env file."
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
