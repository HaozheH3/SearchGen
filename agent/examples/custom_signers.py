"""Templates for credentials owned and configured by the release user.

Copy these functions into your own package and adapt them to your providers.
Do not put real credentials in this file or any committed configuration.
"""

from __future__ import annotations

import os
from dataclasses import replace

from searchgen_agent.signing import (
    ObjectSignRequest,
    SearchRequest,
    SignedObjectRequest,
)


def sign_search_request(request: SearchRequest) -> SearchRequest:
    """Example header-token signer using a user-owned environment variable."""

    token = os.environ.get("MY_SEARCH_API_KEY", "").strip()
    if not token:
        raise RuntimeError("MY_SEARCH_API_KEY is required by this signer")
    headers = dict(request.headers)
    headers["Authorization"] = f"Bearer {token}"
    return replace(request, headers=headers)


def sign_object_request(request: ObjectSignRequest) -> SignedObjectRequest:
    """Template for an S3-, OSS-, GCS-, or custom-storage SDK integration.

    Replace the body with your provider SDK's presign operation. The SDK should
    obtain credentials from an environment variable, workload identity, instance
    role, or secret manager. Never return or log the underlying credential.
    """

    raise NotImplementedError(
        "Create a presigned URL with your storage SDK, then return "
        "SignedObjectRequest(url=presigned_url, headers=required_headers)"
    )
