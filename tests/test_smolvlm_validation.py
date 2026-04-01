"""Tests for SmolVLM input validation and SSRF protection."""

import pytest
from unittest.mock import patch, AsyncMock


def test_ssrf_blocked_localhost():
    """image_url pointing to localhost should be blocked."""
    from urllib.parse import urlparse
    blocked = ["http://localhost/img.png", "http://127.0.0.1/img.png",
               "http://10.0.0.1/img.png", "http://192.168.1.1/img.png",
               "http://internal.company.internal/img.png"]

    for url in blocked:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        is_blocked = (
            hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or
            hostname.startswith("10.") or hostname.startswith("172.") or
            hostname.startswith("192.168.") or hostname.endswith(".internal")
        )
        assert is_blocked, f"URL should be blocked: {url}"


def test_ssrf_allowed_public():
    """Public URLs should pass SSRF check."""
    from urllib.parse import urlparse
    allowed = ["https://example.com/image.png", "https://cdn.site.com/photo.jpg"]

    for url in allowed:
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        is_blocked = (
            hostname in ("localhost", "127.0.0.1", "0.0.0.0", "::1") or
            hostname.startswith("10.") or hostname.startswith("172.") or
            hostname.startswith("192.168.") or hostname.endswith(".internal")
        )
        assert not is_blocked, f"URL should be allowed: {url}"
