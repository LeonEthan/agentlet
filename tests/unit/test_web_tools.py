from __future__ import annotations

import agentlet.tools.web.search as search_module
from agentlet.tools.web.fetch import WebFetchTool, _FetchedResponse
from agentlet.tools.web.search import WebSearchTool, _fetch_search_html


def test_web_search_returns_ranked_results_from_html_fixture() -> None:
    html = """
    <html>
      <body>
        <a class="result__a" href="/l/?uddg=https%3A%2F%2Fexample.com%2Falpha">Alpha Result</a>
        <div class="result__snippet">First snippet with useful context.</div>
        <a class="result__a" href="https://example.com/beta">Beta Result</a>
        <div class="result__snippet">Second snippet.</div>
      </body>
    </html>
    """

    tool = WebSearchTool(fetch_html=lambda url, timeout: html)

    result = tool.execute({"query": "agentlet", "max_results": 2})

    assert result.is_error is False
    assert result.output == (
        "1. Alpha Result\n"
        "URL: https://example.com/alpha\n"
        "Snippet: First snippet with useful context.\n\n"
        "2. Beta Result\n"
        "URL: https://example.com/beta\n"
        "Snippet: Second snippet."
    )
    assert result.metadata == {
        "query": "agentlet",
        "search_url": "https://duckduckgo.com/html/?q=agentlet",
        "count": 2,
        "results": [
            {
                "rank": 1,
                "title": "Alpha Result",
                "url": "https://example.com/alpha",
                "snippet": "First snippet with useful context.",
            },
            {
                "rank": 2,
                "title": "Beta Result",
                "url": "https://example.com/beta",
                "snippet": "Second snippet.",
            },
        ],
        "max_results": 2,
    }


def test_web_search_returns_normalized_error_for_invalid_query() -> None:
    tool = WebSearchTool(fetch_html=lambda url, timeout: "")

    result = tool.execute({"query": "   "})

    assert result.is_error is True
    assert result.output == "WebSearch requires a non-empty string 'query'."


def test_web_search_normalizes_fetch_failures() -> None:
    def fail(url: str, timeout: float | None) -> str:
        del timeout
        raise OSError(f"cannot reach {url}")

    tool = WebSearchTool(fetch_html=fail)

    result = tool.execute({"query": "agentlet"})

    assert result.is_error is True
    assert result.output == (
        "WebSearch failed: cannot reach https://duckduckgo.com/html/?q=agentlet"
    )
    assert result.metadata == {
        "query": "agentlet",
        "search_url": "https://duckduckgo.com/html/?q=agentlet",
        "error_type": "OSError",
    }


def test_web_search_falls_back_when_response_charset_is_unknown() -> None:
    html = "<a class='result__a' href='https://example.com'>café</a>".encode("utf-8")

    class _Headers:
        @staticmethod
        def get_content_charset() -> str:
            return "unknown-charset"

    class _Response:
        headers = _Headers()

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        @staticmethod
        def read(limit: int) -> bytes:
            del limit
            return html

    original_urlopen = search_module.urlopen
    search_module.urlopen = lambda request, timeout: _Response()  # type: ignore[assignment]
    try:
        assert _fetch_search_html(
            "https://duckduckgo.com/html/?q=agentlet",
            1.0,
        ) == "<a class='result__a' href='https://example.com'>café</a>"
    finally:
        search_module.urlopen = original_urlopen


def test_web_search_preserves_encoded_reserved_characters_in_redirect_targets() -> None:
    normalized = search_module._normalize_result_url(
        "https://duckduckgo.com/l/?uddg="
        "https%3A%2F%2Fexample.com%2Fa%253Fb%253D1%2526c%253D2"
    )

    assert normalized == "https://example.com/a%3Fb%3D1%26c%3D2"


def test_web_fetch_extracts_html_title_and_text() -> None:
    response = _FetchedResponse(
        requested_url="https://example.com/post",
        final_url="https://example.com/post",
        status_code=200,
        content_type="text/html; charset=utf-8",
        body=(
            b"<html><head><title>Example Post</title></head>"
            b"<body><main><h1>Headline</h1><p>First paragraph.</p>"
            b"<script>ignored()</script><p>Second paragraph.</p></main></body></html>"
        ),
    )
    tool = WebFetchTool(fetch_url=lambda url, timeout: response)

    result = tool.execute({"url": "https://example.com/post"})

    assert result.is_error is False
    assert result.output == "# Example Post\n\nHeadline\n\nFirst paragraph.\n\nSecond paragraph."
    assert result.metadata == {
        "url": "https://example.com/post",
        "final_url": "https://example.com/post",
        "status_code": 200,
        "content_type": "text/html; charset=utf-8",
        "response_truncated": False,
        "source_format": "html",
        "title": "Example Post",
        "extracted_chars": len(result.output),
        "truncated": False,
    }


def test_web_fetch_truncates_plain_text_output() -> None:
    response = _FetchedResponse(
        requested_url="https://example.com/data.txt",
        final_url="https://example.com/data.txt",
        status_code=200,
        content_type="text/plain; charset=utf-8",
        body=b"alpha\n\nbeta\n\ngamma\n",
    )
    tool = WebFetchTool(fetch_url=lambda url, timeout: response)

    result = tool.execute({"url": "https://example.com/data.txt", "max_chars": 7})

    assert result.is_error is False
    assert result.output == "alpha\n\n"
    assert result.metadata == {
        "url": "https://example.com/data.txt",
        "final_url": "https://example.com/data.txt",
        "status_code": 200,
        "content_type": "text/plain; charset=utf-8",
        "response_truncated": False,
        "max_chars": 7,
        "source_format": "text",
        "extracted_chars": 7,
        "truncated": True,
    }


def test_web_fetch_rejects_non_http_urls() -> None:
    tool = WebFetchTool(fetch_url=lambda url, timeout: None)  # type: ignore[arg-type]

    result = tool.execute({"url": "file:///tmp/example.txt"})

    assert result.is_error is True
    assert result.output == "WebFetch url must be an absolute http or https URL."


def test_web_fetch_rejects_urls_with_control_characters() -> None:
    tool = WebFetchTool(fetch_url=lambda url, timeout: None)  # type: ignore[arg-type]

    result = tool.execute({"url": "https://example.com/\nfoo"})

    assert result.is_error is True
    assert (
        result.output
        == "WebFetch url must not contain whitespace or control characters."
    )


def test_web_fetch_normalizes_transport_failures() -> None:
    def fail(url: str, timeout: float | None) -> _FetchedResponse:
        del timeout
        raise OSError(f"cannot reach {url}")

    tool = WebFetchTool(fetch_url=fail)

    result = tool.execute({"url": "https://example.com/post"})

    assert result.is_error is True
    assert result.output == "WebFetch failed: cannot reach https://example.com/post"
    assert result.metadata == {
        "url": "https://example.com/post",
        "error_type": "OSError",
    }


def test_web_fetch_rejects_unsupported_content_type() -> None:
    response = _FetchedResponse(
        requested_url="https://example.com/image.png",
        final_url="https://example.com/image.png",
        status_code=200,
        content_type="image/png",
        body=b"\x89PNG\r\n",
    )
    tool = WebFetchTool(fetch_url=lambda url, timeout: response)

    result = tool.execute({"url": "https://example.com/image.png"})

    assert result.is_error is True
    assert result.output == "WebFetch does not support content type: image/png"
    assert result.metadata == {
        "url": "https://example.com/image.png",
        "final_url": "https://example.com/image.png",
        "status_code": 200,
        "content_type": "image/png",
        "response_truncated": False,
    }


def test_web_fetch_falls_back_when_response_charset_is_unknown() -> None:
    response = _FetchedResponse(
        requested_url="https://example.com/post",
        final_url="https://example.com/post",
        status_code=200,
        content_type="text/plain; charset=unknown-charset",
        body="café".encode("utf-8"),
    )
    tool = WebFetchTool(fetch_url=lambda url, timeout: response)

    result = tool.execute({"url": "https://example.com/post"})

    assert result.is_error is False
    assert result.output == "café"
    assert result.metadata == {
        "url": "https://example.com/post",
        "final_url": "https://example.com/post",
        "status_code": 200,
        "content_type": "text/plain; charset=unknown-charset",
        "response_truncated": False,
        "source_format": "text",
        "extracted_chars": len("café"),
        "truncated": False,
    }
