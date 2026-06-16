"""Gemini vision backend, built on the Google Gen AI SDK (``google-genai``).

This mirrors the Codex backend's contract: a single stateless call that takes
one or more images plus a system/user prompt and returns JSON matching the
given schema. The prompts and the overall flow are identical to Codex — only
the underlying agent changes.

Auth maps onto the two shared modes:

* ``api``  -> Google AI Studio API key (``genai.Client(api_key=...)``).
* ``auth`` -> Vertex AI via Application Default Credentials
              (``genai.Client(vertexai=True, project=..., location=...)``).

This is the same SDK the Google Agent Development Kit (ADK) uses under the hood
for every Gemini turn; here we call it directly because the marker flow is a
single structured vision request, not a multi-turn agent loop.
"""
from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Any

from ..config import get_gemini_api_key, get_vertex_config, is_native_auth


def _load_genai() -> tuple[Any, Any]:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "The Gemini provider requires the google-genai SDK. "
            "Install it with `pip install google-genai` (or `pip install -r requirements.txt`)."
        ) from exc
    return genai, types


_JSON_TYPE_TO_GENAI = {
    "object": "OBJECT",
    "array": "ARRAY",
    "string": "STRING",
    "integer": "INTEGER",
    "number": "NUMBER",
    "boolean": "BOOLEAN",
}


def _to_genai_schema(node: dict[str, Any], types: Any) -> Any:
    """Translate one of the package's JSON-schema dicts into a genai ``Schema``.

    The marker schemas express nullability as ``"type": ["object", "null"]`` and
    use ``additionalProperties``/``required`` — genai's Schema instead wants a
    single ``type`` plus a ``nullable`` flag, so we normalize here. The marker
    schemas carry no numeric constraints, so nothing is silently dropped.
    """
    raw_type = node.get("type")
    nullable = False
    if isinstance(raw_type, list):
        nullable = "null" in raw_type
        concrete = [t for t in raw_type if t != "null"]
        raw_type = concrete[0] if concrete else "string"

    kwargs: dict[str, Any] = {}
    if raw_type is not None:
        kwargs["type"] = getattr(types.Type, _JSON_TYPE_TO_GENAI[raw_type])
    if nullable:
        kwargs["nullable"] = True
    if "enum" in node:
        kwargs["enum"] = list(node["enum"])

    if raw_type == "object":
        properties = node.get("properties", {})
        kwargs["properties"] = {
            key: _to_genai_schema(value, types) for key, value in properties.items()
        }
        # Preserve declaration order so the model emits fields predictably.
        kwargs["property_ordering"] = list(properties.keys())
        if node.get("required"):
            kwargs["required"] = list(node["required"])
    elif raw_type == "array" and "items" in node:
        kwargs["items"] = _to_genai_schema(node["items"], types)

    return types.Schema(**kwargs)


def _build_client(genai: Any, auth: str, api_key: str | None) -> Any:
    if is_native_auth(auth):
        project, location = get_vertex_config()
        if not project:
            raise RuntimeError(
                "Gemini auth=auth (Vertex AI) requires GOOGLE_CLOUD_PROJECT to be set "
                "(and `gcloud auth application-default login`). "
                "For an API key instead, use MARKER_AUTH=api with GEMINI_API_KEY."
            )
        return genai.Client(vertexai=True, project=project, location=location)
    return genai.Client(api_key=get_gemini_api_key(api_key))


def _image_part(path: str | Path, types: Any) -> Any:
    resolved = Path(path).expanduser().resolve()
    mime = mimetypes.guess_type(resolved.name)[0] or "image/png"
    return types.Part.from_bytes(data=resolved.read_bytes(), mime_type=mime)


def call_json(
    *,
    image_paths: list[str | Path],
    system_prompt: str,
    user_text: str,
    model: str,
    output_schema: dict[str, Any],
    auth: str,
    api_key: str | None = None,
    reasoning_effort: str | None = None,  # noqa: ARG001 - Codex-only; accepted for parity
) -> dict[str, Any]:
    genai, types = _load_genai()
    client = _build_client(genai, auth, api_key)

    parts: list[Any] = [types.Part.from_text(text=f"{user_text}\n\nReturn JSON only.")]
    parts.extend(_image_part(path, types) for path in image_paths)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=_to_genai_schema(output_schema, types) if output_schema else None,
        temperature=0,
    )
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
        config=config,
    )

    text = getattr(response, "text", None)
    if not text:
        reason = ""
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            reason = f" (finish_reason={getattr(candidates[0], 'finish_reason', None)})"
        raise RuntimeError(f"Gemini returned an empty response{reason}.")
    return json.loads(text)
