"""Vision provider backends.

Each backend turns an image plus a system/user prompt into structured JSON that
matches a supplied JSON schema. They share the same call signature so the rest
of the package is provider-agnostic:

    call_json(
        image_paths=[...],
        system_prompt="...",
        user_text="...",
        model="...",
        output_schema={...},      # JSON-schema dict
        auth="auth" | "api",
        api_key=None,
        reasoning_effort=None,    # only meaningful for some providers
    ) -> dict

The Codex backend lives in ``marker.vision`` (kept there so its long-standing
test seams stay valid); Gemini lives in ``marker.providers.gemini``. New
providers (e.g. Claude) can be added as another module here.
"""
