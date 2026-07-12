"""Credential-free extension points for provider-specific request signing.

The release pipeline passes request metadata to user-supplied callables. Those
callables may read credentials from environment variables, a secret manager, or
a provider SDK and return a signed request. This module deliberately contains no
provider endpoint, account identifier, or credential fallback.

Avoid logging these objects after signing: headers, request bodies, and signed
URLs can contain secrets and are intentionally omitted from their representations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Callable, Mapping, Protocol, runtime_checkable
from urllib.parse import urlsplit


def _validate_http_url(url: str, *, field_name: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an absolute HTTP(S) URL")


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """Search request before or after user-provided authentication is applied."""

    method: str
    url: str
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    query_params: Mapping[str, str] = field(default_factory=dict, repr=False)
    json_body: Mapping[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        method = self.method.strip().upper()
        if not method:
            raise ValueError("method must not be empty")
        _validate_http_url(self.url, field_name="url")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "headers", dict(self.headers))
        object.__setattr__(self, "query_params", dict(self.query_params))
        if self.json_body is not None:
            object.__setattr__(self, "json_body", dict(self.json_body))


@runtime_checkable
class SearchRequestSigner(Protocol):
    """Callable implemented by a user or provider adapter.

    A signer may return a copy of the request with authentication in headers,
    query parameters, or the JSON body. It must not persist the credential.
    """

    def __call__(self, request: SearchRequest, /) -> SearchRequest:
        """Return an authenticated request without mutating ``request``."""


def unsigned_search_request(request: SearchRequest, /) -> SearchRequest:
    """No-op signer for public search endpoints that require no authentication."""

    return request


def apply_search_signature(
    request: SearchRequest,
    signer: SearchRequestSigner | None,
) -> SearchRequest:
    """Apply ``signer`` immediately before transport sends a search request."""

    if signer is None:
        return request
    signed = signer(request)
    if not isinstance(signed, SearchRequest):
        raise TypeError("search signer must return SearchRequest")
    return signed


@dataclass(frozen=True, slots=True)
class ObjectSignRequest:
    """Description of an object-storage operation that needs a signed URL."""

    method: str
    object_key: str
    expires_seconds: int = 900
    content_type: str | None = None

    def __post_init__(self) -> None:
        method = self.method.strip().upper()
        if method not in {"GET", "PUT", "HEAD", "DELETE"}:
            raise ValueError("object method must be GET, PUT, HEAD, or DELETE")
        if not self.object_key.strip() or "\x00" in self.object_key:
            raise ValueError("object_key must not be empty or contain NUL")
        if self.expires_seconds <= 0:
            raise ValueError("expires_seconds must be positive")
        object.__setattr__(self, "method", method)


@dataclass(frozen=True, slots=True)
class SignedObjectRequest:
    """Provider result used by an HTTP transport for an object operation."""

    url: str = field(repr=False)
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        _validate_http_url(self.url, field_name="signed object URL")
        object.__setattr__(self, "headers", dict(self.headers))


@runtime_checkable
class ObjectStorageSigner(Protocol):
    """Callable that creates a provider-specific signed object request."""

    def __call__(self, request: ObjectSignRequest, /) -> SignedObjectRequest:
        """Sign ``request`` using credentials held outside the release package."""


def sign_object_request(
    request: ObjectSignRequest,
    signer: ObjectStorageSigner,
) -> SignedObjectRequest:
    """Ask a required user/provider signer to authorize an object operation."""

    if signer is None:
        raise TypeError("an object-storage signer is required for remote storage")
    signed = signer(request)
    if not isinstance(signed, SignedObjectRequest):
        raise TypeError("object-storage signer must return SignedObjectRequest")
    return signed


def load_signer(spec: str) -> Callable[..., Any]:
    """Load a trusted signer from ``"package.module:function_name"``.

    This is intended for explicit CLI/configuration wiring. Importing arbitrary
    modules executes their code, so callers must accept only user-trusted specs.
    """

    module_name, separator, attribute_name = spec.strip().partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("signer must use 'package.module:function_name' syntax")
    module = import_module(module_name)
    signer = getattr(module, attribute_name)
    if not callable(signer):
        raise TypeError(f"configured signer is not callable: {spec}")
    return signer
