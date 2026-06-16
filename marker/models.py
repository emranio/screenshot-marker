from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


class NormalizedBbox(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., gt=0.0, le=1.0)
    height: float = Field(..., gt=0.0, le=1.0)


class RawAnnotation(BaseModel):
    """What we expect the LLM to return for one query (normalized coords)."""

    request_index: int
    request_text: str
    target_description: str
    label_text: Optional[str] = None
    bbox: Optional[NormalizedBbox] = None
    label_position: Optional[NormalizedBbox] = None
    color: Optional[str] = None
    not_found: bool = False
    notes: str = ""


class RawAnnotationResponse(BaseModel):
    annotations: list[RawAnnotation]


class Bbox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Annotation(BaseModel):
    """A resolved annotation in absolute pixel coordinates."""

    request_index: int
    request_text: str
    target_description: str
    label_text: Optional[str] = None
    bbox: Optional[Bbox] = None
    label_position: Optional[Bbox] = None
    color: Optional[str] = None
    not_found: bool = False
    notes: str = ""


class AnnotationResult(BaseModel):
    output_path: Path
    annotations: list[Annotation]
    unresolved: list[str]
