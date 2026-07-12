#!/usr/bin/env python3
"""Tiny offline OpenAI-compatible server for the Quick Start."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers.get("content-length", "0"))
        request = json.loads(self.rfile.read(size) or b"{}")
        messages = request.get("messages") or []
        system = str(messages[0].get("content") if messages else "")
        if "predicting where image generation models" in system:
            content = {
                "analysis_reasoning": "This generic prompt needs no external grounding.",
                "knowledge_gaps": [],
                "search_queries": [],
                "search_justification": "",
                "needs_search": False,
            }
        else:
            content = {
                "visual_planning": "Use a centered subject and natural depth.",
                "reasoning": "Add concrete lighting and material details.",
                "borrow_from_references": [],
                "combine_strategy": "No external references are needed.",
                "create_new": ["soft window light"],
                "refined_prompt": "A detailed red apple on a wooden table in soft morning window light.",
            }
        response = json.dumps(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": json.dumps(content)}}
                ]
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"offline chat server: http://{args.host}:{server.server_port}/v1")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
