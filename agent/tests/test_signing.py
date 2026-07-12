from __future__ import annotations

import unittest
from dataclasses import replace

from searchgen_agent.signing import (
    ObjectSignRequest,
    SearchRequest,
    SignedObjectRequest,
    apply_search_signature,
    sign_object_request,
    unsigned_search_request,
)


class SigningHooksTest(unittest.TestCase):
    def test_unsigned_search_request_is_unchanged(self) -> None:
        request = SearchRequest("POST", "https://search.example.com/v1", json_body={"q": "apple"})
        self.assertIs(apply_search_signature(request, unsigned_search_request), request)

    def test_search_signer_can_add_header_without_mutating_input(self) -> None:
        request = SearchRequest("POST", "https://search.example.com/v1", headers={"Accept": "application/json"})

        def signer(value: SearchRequest) -> SearchRequest:
            headers = dict(value.headers)
            headers["X-Example-Signature"] = "test-only"
            return replace(value, headers=headers)

        signed = apply_search_signature(request, signer)
        self.assertNotIn("X-Example-Signature", request.headers)
        self.assertEqual(signed.headers["X-Example-Signature"], "test-only")

    def test_object_signer_returns_http_request(self) -> None:
        request = ObjectSignRequest("GET", "references/example.png", expires_seconds=60)

        def signer(value: ObjectSignRequest) -> SignedObjectRequest:
            self.assertEqual(value.method, "GET")
            return SignedObjectRequest("https://storage.example.com/signed/example.png")

        signed = sign_object_request(request, signer)
        self.assertTrue(signed.url.startswith("https://"))

    def test_object_signer_is_required(self) -> None:
        request = ObjectSignRequest("PUT", "references/example.png")
        with self.assertRaises(TypeError):
            sign_object_request(request, None)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
