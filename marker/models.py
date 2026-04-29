from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field


Placement = Literal["above", "below", "left", "right"]


class NormalizedBbox(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)
    width: float = Field(..., gt=0.0, le=1.0)
    height: float = Field(..., gt=0.0, le=1.0)


class NormalizedPoint(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    y: float = Field(..., ge=0.0, le=1.0)


class NormalizedArrow(BaseModel):
    start: NormalizedPoint
    end: NormalizedPoint


class NormalizedLabel(BaseModel):
    text: str
    anchor: NormalizedPoint
    placement: Placement = "right"


class RawAnnotation(BaseModel):
    """What we expect the LLM to return for one query (normalized coords)."""

    request_index: int
    request_text: str
    target_description: str
    shape: Optional[Literal["rectangle"]] = "rectangle"
    bbox: Optional[NormalizedBbox] = None
    arrow: Optional[NormalizedArrow] = None
    label: Optional[NormalizedLabel] = None
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


class Point(BaseModel):
    x: int
    y: int


class Arrow(BaseModel):
    start: Point
    end: Point


class Label(BaseModel):
    text: str
    anchor: Point
    placement: Placement = "right"


class Annotation(BaseModel):
    """A resolved annotation in absolute pixel coordinates."""

    request_index: int
    request_text: str
    target_description: str
    shape: Optional[Literal["rectangle"]] = "rectangle"
    bbox: Optional[Bbox] = None
    arrow: Optional[Arrow] = None
    label: Optional[Label] = None
    color: Optional[str] = None
    not_found: bool = False
    notes: str = ""


class AnnotationResult(BaseModel):
    output_path: Path
    annotations: list[Annotation]
    unresolved: list[str]
