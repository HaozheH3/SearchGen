from __future__ import annotations

import json
import os
import threading
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from searchgen_agent.reasoner.search_client import SearchClient
from searchgen_agent.signing import SearchRequest


class _SearchHandler(BaseHTTPRequestHandler):
    received_signature = ""
    received_body: dict[str, object] = {}

    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers.get("content-length", "0"))
        type(self).received_signature = self.headers.get("X-Test-Signature", "")
        type(self).received_body = json.loads(self.rfile.read(size) or b"{}")
        response = json.dumps(
            {
                "success": True,
                "results": [
                    {"title": "Apple", "link": "https://example.com/apple", "snippet": "A fruit"}
                ],
            }
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format: str, *args: object) -> None:
        return


class SearchClientTest(unittest.TestCase):
    def test_user_signer_is_applied_to_real_transport_call(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _SearchHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        old_upper = os.environ.get("NO_PROXY")
        old_lower = os.environ.get("no_proxy")
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"
        os.environ["no_proxy"] = "127.0.0.1,localhost"
        try:
            def signer(request: SearchRequest) -> SearchRequest:
                headers = dict(request.headers)
                headers["X-Test-Signature"] = "signed-by-user-hook"
                return replace(request, headers=headers)

            client = SearchClient(
                api_url=f"http://127.0.0.1:{server.server_port}/search",
                signer=signer,
            )
            result = client.text_search("apple", num=3)
            self.assertTrue(result["success"])
            self.assertEqual(result["results"][0]["title"], "Apple")
            self.assertEqual(_SearchHandler.received_signature, "signed-by-user-hook")
            self.assertEqual(_SearchHandler.received_body["search_type"], "web")
        finally:
            server.shutdown()
            server.server_close()
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
