from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from searchgen_agent.cli.reason import main as reason_main
from searchgen_agent.generator.api import mock_image_generator
from searchgen_agent.generator.runner import generate_row


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    request_count = 0

    def do_POST(self) -> None:  # noqa: N802
        type(self).request_count += 1
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages") or []
        system = str(messages[0].get("content") if messages else "")
        user = str(messages[-1].get("content") if messages else "")
        if "predicting where image generation models" in system:
            if "current Widget X" in user:
                analysis = {
                    "analysis_reasoning": "The current product appearance needs web grounding.",
                    "knowledge_gaps": [
                        {
                            "entity": "Widget X",
                            "category": "TK-C",
                            "severity": "critical",
                            "reasoning": "The design may have changed.",
                            "suggested_search": "current Widget X appearance",
                            "search_type": "web",
                        }
                    ],
                    "search_queries": [],
                    "search_justification": "Verify the current appearance.",
                    "needs_search": True,
                }
            else:
                analysis = {
                    "analysis_reasoning": "The request is generic and needs no external grounding.",
                    "knowledge_gaps": [],
                    "search_queries": [],
                    "search_justification": "",
                    "needs_search": False,
                }
            content = json.dumps(analysis)
        else:
            content = json.dumps(
                {
                    "visual_planning": "Center the subject with natural depth.",
                    "reasoning": "Clarify lighting and material detail.",
                    "borrow_from_references": [],
                    "combine_strategy": "No external references are needed.",
                    "create_new": ["soft window light"],
                    "refined_prompt": "A detailed red apple on a wooden table in soft window light.",
                }
            )
        body = json.dumps(
            {
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class OfflineEndToEndTest(unittest.TestCase):
    def test_agentic_reasoner_to_image_generator(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        old_base = os.environ.get("SEARCHGEN_CHAT_BASE_URL")
        old_format = os.environ.get("SEARCHGEN_CHAT_API_FORMAT")
        old_no_proxy = os.environ.get("NO_PROXY")
        old_no_proxy_lower = os.environ.get("no_proxy")
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset = root / "prompts.jsonl"
                dataset.write_text(json.dumps({"prompt": "A red apple on a wooden table"}) + "\n")
                code = reason_main(
                    [
                        "--input",
                        str(dataset),
                        "--output-dir",
                        str(root / "runs"),
                        "--lane-name",
                        "demo",
                        "--model",
                        "fake-model",
                        "--chat-base-url",
                        f"http://127.0.0.1:{server.server_port}/v1",
                        "--workers-s1",
                        "1",
                        "--workers-s1d1-search",
                        "1",
                        "--workers-s1d1-download",
                        "1",
                        "--workers-s2",
                        "1",
                        "--workers-s3",
                        "1",
                        "--yes",
                    ]
                )
                row = root / "runs" / "demo" / "eval_row_001"
                artifacts = row / "artifacts_files"
                if code != 0:
                    diagnostics = []
                    for path in sorted((root / "runs").rglob("*")) if (root / "runs").exists() else []:
                        if path.is_file() and path.suffix in {".json", ".jsonl", ".log"}:
                            diagnostics.append(f"--- {path.name} ---\n{path.read_text(errors='replace')}")
                    self.fail("reasoner failed:\n" + "\n".join(diagnostics))
                self.assertTrue((artifacts / "s1_analysis.json").is_file())
                self.assertFalse((artifacts / "s2_reference_selection.json").is_file())
                self.assertTrue((artifacts / "s3_refined_prompt.json").is_file())
                self.assertTrue((artifacts / "s4_generation_manifest.json").is_file())
                self.assertTrue(
                    generate_row(
                        row,
                        generator_name="mock",
                        generator=mock_image_generator,
                        model="offline",
                        width=384,
                        height=384,
                    )
                )
                self.assertGreater(
                    (row / "mock_generator" / "generated_image.png").stat().st_size,
                    256,
                )
                requests_before_resume = _FakeOpenAIHandler.request_count
                resume_code = reason_main(
                    [
                        "--input",
                        str(dataset),
                        "--output-dir",
                        str(root / "runs"),
                        "--lane-name",
                        "demo",
                        "--model",
                        "fake-model",
                        "--chat-base-url",
                        f"http://127.0.0.1:{server.server_port}/v1",
                        "--yes",
                    ]
                )
                self.assertEqual(resume_code, 0)
                self.assertEqual(_FakeOpenAIHandler.request_count, requests_before_resume)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
            if old_base is None:
                os.environ.pop("SEARCHGEN_CHAT_BASE_URL", None)
            else:
                os.environ["SEARCHGEN_CHAT_BASE_URL"] = old_base
            if old_format is None:
                os.environ.pop("SEARCHGEN_CHAT_API_FORMAT", None)
            else:
                os.environ["SEARCHGEN_CHAT_API_FORMAT"] = old_format
            if old_no_proxy is None:
                os.environ.pop("NO_PROXY", None)
            else:
                os.environ["NO_PROXY"] = old_no_proxy
            if old_no_proxy_lower is None:
                os.environ.pop("no_proxy", None)
            else:
                os.environ["no_proxy"] = old_no_proxy_lower

    def test_reasoner_web_search_branch(self) -> None:
        chat_server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)

        class SearchHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                size = int(self.headers.get("content-length", "0"))
                request = json.loads(self.rfile.read(size) or b"{}")
                self.assert_request(request)
                response = json.dumps(
                    {
                        "success": True,
                        "results": [
                            {
                                "title": "Widget X product page",
                                "link": "https://example.com/widget-x",
                                "snippet": "Widget X currently has a blue enclosure.",
                            }
                        ],
                    }
                ).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            @staticmethod
            def assert_request(request: dict[str, object]) -> None:
                if request.get("search_type") != "web":
                    raise AssertionError(f"unexpected request: {request}")

            def log_message(self, format: str, *args: object) -> None:
                return

        search_server = ThreadingHTTPServer(("127.0.0.1", 0), SearchHandler)
        threads = [
            threading.Thread(target=chat_server.serve_forever, daemon=True),
            threading.Thread(target=search_server.serve_forever, daemon=True),
        ]
        for thread in threads:
            thread.start()
        old_upper = os.environ.get("NO_PROXY")
        old_lower = os.environ.get("no_proxy")
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                dataset = root / "prompts.jsonl"
                dataset.write_text(json.dumps({"prompt": "Show the current Widget X product"}) + "\n")
                code = reason_main(
                    [
                        "--input",
                        str(dataset),
                        "--output-dir",
                        str(root / "runs"),
                        "--lane-name",
                        "search_demo",
                        "--model",
                        "fake-model",
                        "--chat-base-url",
                        f"http://127.0.0.1:{chat_server.server_port}/v1",
                        "--search-api-url",
                        f"http://127.0.0.1:{search_server.server_port}/search",
                        "--workers-s1",
                        "1",
                        "--workers-s1d1-search",
                        "1",
                        "--workers-s1d1-download",
                        "1",
                        "--workers-s2",
                        "1",
                        "--workers-s3",
                        "1",
                        "--yes",
                    ]
                )
                self.assertEqual(code, 0)
                artifacts = root / "runs" / "search_demo" / "eval_row_001" / "artifacts_files"
                self.assertTrue((artifacts / "s1d1_search_results.json").is_file())
                s1d1 = json.loads((artifacts / "s1d1_search_results.json").read_text())
                self.assertTrue((s1d1.get("search_results") or {}).get("web-0"))
                self.assertTrue((artifacts / "s4_generation_manifest.json").is_file())
        finally:
            chat_server.shutdown()
            search_server.shutdown()
            chat_server.server_close()
            search_server.server_close()
            for thread in threads:
                thread.join(timeout=2)
            if old_upper is None:
                os.environ.pop("NO_PROXY", None)
            else:
                os.environ["NO_PROXY"] = old_upper
            if old_lower is None:
                os.environ.pop("no_proxy", None)
            else:
                os.environ["no_proxy"] = old_lower


if __name__ == "__main__":
    unittest.main()
