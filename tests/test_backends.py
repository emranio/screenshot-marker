from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import annotate as cli
import marker
from marker import drawing
from marker import config, vision
from marker.models import Bbox
from marker.parser import parse_response


RAW_RESPONSE = {
    "annotations": [
        {
            "request_index": 0,
            "request_text": "box around main",
            "target_description": "main panel",
            "label_text": None,
            "bbox": {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.2},
            "color": None,
            "not_found": False,
            "notes": "",
        }
    ]
}


class AuthConfigTests(unittest.TestCase):
    def test_resolve_auth_defaults_to_codex(self) -> None:
        with patch("marker.config.load_env", lambda: None), patch.dict(
            os.environ, {}, clear=True
        ):
            self.assertEqual(config.resolve_auth_mode(), "codex")

    def test_resolve_auth_uses_env_and_allows_explicit_override(self) -> None:
        with patch("marker.config.load_env", lambda: None), patch.dict(
            os.environ, {"MARKER_AUTH": "api"}, clear=True
        ):
            self.assertEqual(config.resolve_auth_mode(), "api")
            self.assertEqual(config.resolve_auth_mode("codex"), "codex")

    def test_resolve_auth_rejects_unknown_values(self) -> None:
        with patch("marker.config.load_env", lambda: None):
            with self.assertRaisesRegex(ValueError, "Unsupported auth mode"):
                config.resolve_auth_mode("bogus")

    def test_resolve_reasoning_effort_defaults_to_medium(self) -> None:
        with patch("marker.config.load_env", lambda: None), patch.dict(
            os.environ, {}, clear=True
        ):
            self.assertEqual(config.resolve_reasoning_effort(), "medium")

    def test_get_codex_api_key_accepts_codex_or_openai_key_env(self) -> None:
        with patch("marker.config.load_env", lambda: None), patch.dict(
            os.environ, {"CODEX_API_KEY": "codex-key"}, clear=True
        ):
            self.assertEqual(config.get_codex_api_key(), "codex-key")

        with patch("marker.config.load_env", lambda: None), patch.dict(
            os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True
        ):
            self.assertEqual(config.get_codex_api_key(), "openai-key")

    def test_cli_accepts_auth_validation_and_effort_flags(self) -> None:
        args = cli._parse_args(
            [
                "--image",
                "tests/screens/test_1.jpeg",
                "--query",
                "box around main",
                "--auth",
                "api",
                "--reasoning-effort",
                "medium",
                "--steps",
            ]
        )
        self.assertEqual(args.auth, "api")
        self.assertEqual(args.reasoning_effort, "medium")
        self.assertTrue(args.steps)


class AnnotationContractTests(unittest.TestCase):
    def test_parse_response_converts_label_position_and_arrow_flag(self) -> None:
        raw = {
            "annotations": [
                {
                    "request_index": 0,
                    "request_text": "box around main, label 'Main'",
                    "target_description": "main panel",
                    "label_text": "Main",
                    "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.4},
                    "label_position": {
                        "x": 0.5,
                        "y": 0.7,
                        "width": 0.2,
                        "height": 0.1,
                    },
                    "show_arrow": False,
                    "color": None,
                    "not_found": False,
                    "notes": "",
                }
            ]
        }

        annotations, unresolved = parse_response(
            raw,
            ["box around main, label 'Main'"],
            200,
            100,
        )

        self.assertEqual(unresolved, [])
        self.assertEqual(annotations[0].bbox, Bbox(x=20, y=20, width=60, height=40))
        self.assertEqual(
            annotations[0].label_position,
            Bbox(x=100, y=70, width=40, height=10),
        )
        self.assertIs(annotations[0].show_arrow, False)

    def test_touching_labels_suppress_arrows(self) -> None:
        # A label flush against the box (no visible gap) needs no arrow, even
        # if the model asked for one.
        bbox = Bbox(x=50, y=50, width=80, height=30)
        touching_label = (52.0, 86.0, 150.0, 116.0)  # ~6px below the box edge

        self.assertFalse(drawing._should_draw_arrow(touching_label, bbox, None, 4))
        self.assertFalse(drawing._should_draw_arrow(touching_label, bbox, True, 4))

    def test_offset_labels_draw_arrows_by_default(self) -> None:
        # A label sitting apart from the box gets an arrow by default; only an
        # explicit show_arrow=False suppresses it.
        bbox = Bbox(x=50, y=50, width=80, height=30)
        offset_label = (10.0, 170.0, 108.0, 200.0)  # 90px below, clear gap

        self.assertTrue(drawing._should_draw_arrow(offset_label, bbox, None, 4))
        self.assertTrue(drawing._should_draw_arrow(offset_label, bbox, True, 4))
        self.assertFalse(drawing._should_draw_arrow(offset_label, bbox, False, 4))

    def test_render_accepts_model_label_position(self) -> None:
        image = Image.new("RGB", (240, 160), "white")
        annotation = marker.Annotation(
            request_index=0,
            request_text="box around main, label 'Main'",
            target_description="main panel",
            label_text="Main",
            bbox=Bbox(x=40, y=40, width=80, height=40),
            label_position=Bbox(x=42, y=92, width=96, height=36),
            show_arrow=False,
        )

        rendered = drawing.render(
            image,
            [annotation],
            default_color="#DC2626",
            stroke_width=4,
            font_path=None,
        )

        self.assertEqual(rendered.size, image.size)
        self.assertEqual(rendered.mode, "RGB")


class CodexSdkPathTests(unittest.TestCase):
    def test_codex_env_strips_api_keys(self) -> None:
        env = vision._codex_subprocess_env(
            {
                "OPENAI_API_KEY": "sk-test",
                "CODEX_API_KEY": "codex-api-test",
                "CODEX_ACCESS_TOKEN": "subscription-token",
                "PATH": "/bin",
            }
        )

        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("CODEX_API_KEY", env)
        self.assertEqual(env["CODEX_ACCESS_TOKEN"], "subscription-token")
        self.assertEqual(env["PATH"], "/bin")

    def test_subscription_auth_uses_codex_sdk_without_api_key(self) -> None:
        captured: dict[str, object] = {}

        fake_classes = _fake_codex_classes(captured, json.dumps(RAW_RESPONSE))

        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            Image.new("RGB", (2, 2), "white").save(image_file.name)

            with patch(
                "marker.vision._load_codex_sdk",
                return_value=fake_classes,
            ), patch.dict(
                os.environ,
                {
                    "OPENAI_API_KEY": "sk-test",
                    "CODEX_API_KEY": "codex-api-test",
                    "CODEX_ACCESS_TOKEN": "subscription-token",
                },
                clear=True,
            ):
                result = vision.call_vision(
                    image_file.name,
                    100,
                    50,
                    ["box around main"],
                    "gpt-5.5",
                    auth="codex",
                    reasoning_effort="medium",
                )

        self.assertEqual(result, RAW_RESPONSE)
        self.assertIsNone(captured["api_key"])
        env = captured["env"]
        self.assertIsInstance(env, dict)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("CODEX_API_KEY", env)
        self.assertEqual(env["CODEX_ACCESS_TOKEN"], "subscription-token")

        thread_options = captured["thread_options"]
        self.assertEqual(thread_options.kwargs["model"], "gpt-5.5")
        self.assertEqual(thread_options.kwargs["sandbox_mode"], "read-only")
        self.assertEqual(thread_options.kwargs["approval_policy"], "never")
        self.assertEqual(thread_options.kwargs["web_search_mode"], "disabled")
        self.assertEqual(thread_options.kwargs["model_reasoning_effort"], "medium")
        self.assertTrue(thread_options.kwargs["skip_git_repo_check"])

        inputs = captured["inputs"]
        self.assertIsInstance(inputs, list)
        self.assertEqual(inputs[1]["type"], "local_image")
        self.assertTrue(Path(inputs[1]["path"]).is_absolute())

        turn_options = captured["turn_options"]
        self.assertIs(turn_options.kwargs["output_schema"], vision.RESPONSE_SCHEMA)

    def test_api_auth_uses_codex_sdk_api_key_not_openai_client(self) -> None:
        captured: dict[str, object] = {}
        fake_classes = _fake_codex_classes(captured, json.dumps(RAW_RESPONSE))

        with tempfile.NamedTemporaryFile(suffix=".png") as image_file:
            Image.new("RGB", (2, 2), "white").save(image_file.name)

            with patch(
                "marker.vision._load_codex_sdk",
                return_value=fake_classes,
            ), patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "openai-api-key"},
                clear=True,
            ):
                result = vision.call_vision(
                    image_file.name,
                    100,
                    50,
                    ["box around main"],
                    "gpt-5.5",
                    auth="api",
                    reasoning_effort="medium",
                )

        self.assertEqual(result, RAW_RESPONSE)
        self.assertEqual(captured["api_key"], "openai-api-key")
        env = captured["env"]
        self.assertIsInstance(env, dict)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertNotIn("CODEX_API_KEY", env)

    def test_api_auth_requires_api_key(self) -> None:
        with patch("marker.config.load_env", lambda: None), patch.dict(
            os.environ, {}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "API auth requires"):
                vision.call_vision(
                    "tests/screens/test_1.jpeg",
                    100,
                    50,
                    ["box around main"],
                    "gpt-5.5",
                    auth="api",
                )


class AnnotateAuthTests(unittest.TestCase):
    def test_python_api_uses_env_auth_when_no_explicit_auth(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "screen.png"
            output_path = Path(tmpdir) / "annotated.png"
            Image.new("RGB", (20, 20), "white").save(image_path)

            with patch("marker.config.load_env", lambda: None), patch.dict(
                os.environ, {"MARKER_AUTH": "api", "CODEX_API_KEY": "key"}, clear=True
            ), patch("marker.call_vision", return_value=RAW_RESPONSE) as call:
                marker.annotate(
                    image_path=image_path,
                    output_path=output_path,
                    queries=["box around main"],
                    refine=False,
                )

        self.assertEqual(call.call_args.kwargs["auth"], "api")

    def test_python_api_explicit_auth_and_effort_override_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "screen.png"
            output_path = Path(tmpdir) / "annotated.png"
            Image.new("RGB", (20, 20), "white").save(image_path)

            with patch("marker.config.load_env", lambda: None), patch.dict(
                os.environ,
                {"MARKER_AUTH": "api", "OPENAI_REASONING_EFFORT": "high"},
                clear=True,
            ), patch("marker.call_vision", return_value=RAW_RESPONSE) as call:
                marker.annotate(
                    image_path=image_path,
                    output_path=output_path,
                    queries=["box around main"],
                    refine=False,
                    auth="codex",
                    reasoning_effort="medium",
                )

        self.assertEqual(call.call_args.kwargs["auth"], "codex")
        self.assertEqual(call.call_args.kwargs["reasoning_effort"], "medium")

    def test_steps_accepts_first_candidate_and_promotes_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "screen.png"
            output_path = Path(tmpdir) / "annotated.png"
            Image.new("RGB", (20, 20), "white").save(image_path)

            with patch("marker.call_vision", return_value=RAW_RESPONSE), patch(
                "marker.run_annotation_step",
                return_value={
                    "decision": "accept",
                    "notes": "",
                    "annotations": RAW_RESPONSE["annotations"],
                },
            ) as step:
                result = marker.annotate(
                    image_path=image_path,
                    output_path=output_path,
                    queries=["box around main"],
                    refine=False,
                    auth="codex",
                    steps=True,
                )

            self.assertEqual(result.output_path, output_path)
            self.assertTrue(output_path.is_file())
            self.assertEqual(step.call_count, 1)

    def test_steps_redraws_with_corrected_coordinates_without_marker_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "screen.png"
            output_path = Path(tmpdir) / "annotated.png"
            Image.new("RGB", (20, 20), "white").save(image_path)

            with patch("marker.call_vision", return_value=RAW_RESPONSE) as call, patch(
                "marker.run_annotation_step",
                return_value={
                    "decision": "redraw",
                    "notes": "box was too low",
                    "annotations": [
                        {
                            "request_index": 0,
                            "request_text": "box around main",
                            "target_description": "main panel",
                            "label_text": None,
                            "bbox": {"x": 2, "y": 3, "width": 4, "height": 5},
                            "color": None,
                            "not_found": False,
                            "notes": "corrected by step",
                        }
                    ],
                },
            ) as step:
                result = marker.annotate(
                    image_path=image_path,
                    output_path=output_path,
                    queries=["box around main"],
                    refine=False,
                    auth="codex",
                    steps=True,
                )

            self.assertEqual(result.output_path, output_path)
            self.assertTrue(output_path.is_file())
            self.assertEqual(call.call_count, 1)
            self.assertEqual(step.call_count, 1)
            self.assertEqual(result.annotations[0].bbox.x, 2)
            self.assertEqual(result.annotations[0].bbox.y, 3)


def _fake_codex_classes(captured: dict[str, object], final_response: str) -> tuple[type, type, type]:
    class FakeCodex:
        def __init__(self, *, env: dict[str, str], api_key: str | None = None) -> None:
            captured["env"] = env
            captured["api_key"] = api_key

        def start_thread(self, options: object) -> "FakeThread":
            captured["thread_options"] = options
            return FakeThread()

    class FakeThreadOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeTurnOptions:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class FakeTurn:
        def __init__(self, response: str) -> None:
            self.final_response = response

    class FakeThread:
        async def run(self, inputs: object, turn_options: object) -> FakeTurn:
            captured["inputs"] = inputs
            captured["turn_options"] = turn_options
            return FakeTurn(final_response)

    return FakeCodex, FakeThreadOptions, FakeTurnOptions


if __name__ == "__main__":
    unittest.main()
