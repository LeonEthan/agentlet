from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from agentlet.agent.tools.policy import ToolRuntimeConfig
from agentlet.agent.tools.registry import ToolExecutionError
from agentlet.agent.tools.web import WebFetchTool, WebSearchTool


def test_web_search_uses_installed_ddgs_package(monkeypatch, tmp_path) -> None:
    calls: list[dict[str, object]] = []

    class FakeDDGS:
        def __enter__(self) -> "FakeDDGS":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def text(self, query: str, **kwargs):
            calls.append({"query": query, **kwargs})
            return [
                {
                    "title": "Example",
                    "href": "https://example.com",
                    "body": "Snippet",
                }
            ]

    fake_module = ModuleType("ddgs")
    fake_module.DDGS = FakeDDGS
    monkeypatch.setitem(sys.modules, "ddgs", fake_module)

    tool = WebSearchTool(ToolRuntimeConfig(cwd=tmp_path, max_search_results=3))

    result = json.loads(
        asyncio.run(
            tool.execute(
                {
                    "query": "agentlet",
                    "max_results": 5,
                    "region": "us-en",
                    "safesearch": "off",
                }
            )
        )
    )

    assert result["ok"] is True
    assert result["results"][0]["url"] == "https://example.com"
    assert calls == [
        {
            "query": "agentlet",
            "max_results": 3,
            "region": "us-en",
            "safesearch": "off",
        }
    ]
    assert result["results"][0]["source"] == "ddgs"


def test_web_fetch_extracts_html_title_without_shadowing_html_module(
    monkeypatch, tmp_path
) -> None:
    class FakeStreamResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        url = httpx.URL("https://example.com/page")
        encoding = "utf-8"

        def raise_for_status(self) -> None:
            return None

        async def __aenter__(self) -> "FakeStreamResponse":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def aiter_bytes(self):
            yield b"<html><title>Hello &amp; goodbye</title><body>Body</body></html>"

    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def stream(self, method: str, url: str) -> FakeStreamResponse:
            assert method == "GET"
            assert url == "https://example.com/page"
            return FakeStreamResponse()

    fake_trafilatura = ModuleType("trafilatura")
    fake_trafilatura.extract = lambda *args, **kwargs: "Extracted body"
    monkeypatch.setitem(sys.modules, "trafilatura", fake_trafilatura)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    tool = WebFetchTool(ToolRuntimeConfig(cwd=tmp_path))

    result = json.loads(
        asyncio.run(tool.execute({"url": "https://example.com/page"}))
    )

    assert result["ok"] is True
    assert result["title"] == "Hello & goodbye"
    assert result["content"] == "Extracted body"


def test_web_fetch_keeps_small_max_chars_independent_from_fetch_byte_cap(
    monkeypatch, tmp_path
) -> None:
    long_head = "x" * 600
    html = (
        f"<html><head><title>Example title</title><meta name='x' content='{long_head}'></head>"
        "<body><main>Readable body text that should survive extraction.</main></body></html>"
    )

    class FakeStreamResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        url = httpx.URL("https://example.com/article")
        encoding = "utf-8"

        def raise_for_status(self) -> None:
            return None

        async def __aenter__(self) -> "FakeStreamResponse":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def aiter_bytes(self):
            yield html.encode("utf-8")

    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def stream(self, method: str, url: str) -> FakeStreamResponse:
            assert method == "GET"
            assert url == "https://example.com/article"
            return FakeStreamResponse()

    fake_trafilatura = ModuleType("trafilatura")
    fake_trafilatura.extract = lambda *args, **kwargs: "Readable body text that should survive extraction."
    monkeypatch.setitem(sys.modules, "trafilatura", fake_trafilatura)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    tool = WebFetchTool(
        ToolRuntimeConfig(cwd=tmp_path, max_fetch_chars=20_000, max_fetch_bytes=4096)
    )

    result = json.loads(
        asyncio.run(tool.execute({"url": "https://example.com/article", "max_chars": 100}))
    )

    assert result["title"] == "Example title"
    assert result["content"].startswith("Readable body text")
    assert result["truncated"] is False


def test_web_fetch_applies_configured_fetch_byte_limit(monkeypatch, tmp_path) -> None:
    observed: dict[str, int] = {}

    class FakeStreamResponse:
        status_code = 200
        headers = {"content-type": "text/plain; charset=utf-8"}
        url = httpx.URL("https://example.com/large")
        encoding = "utf-8"

        def raise_for_status(self) -> None:
            return None

        async def __aenter__(self) -> "FakeStreamResponse":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def aiter_bytes(self):
            yield b"a" * 250
            yield b"b" * 250

    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def stream(self, method: str, url: str) -> FakeStreamResponse:
            assert method == "GET"
            assert url == "https://example.com/large"
            return FakeStreamResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    tool = WebFetchTool(
        ToolRuntimeConfig(cwd=tmp_path, max_fetch_chars=20_000, max_fetch_bytes=400)
    )
    original_decode = tool._decode_response_text

    def capture_decode(response, raw_bytes: bytes, *, truncated: bool) -> str:
        observed["raw_bytes"] = len(raw_bytes)
        observed["truncated"] = int(truncated)
        return original_decode(response, raw_bytes, truncated=truncated)

    monkeypatch.setattr(tool, "_decode_response_text", capture_decode)

    result = json.loads(
        asyncio.run(tool.execute({"url": "https://example.com/large", "max_chars": 100}))
    )

    assert observed["raw_bytes"] == 400
    assert observed["truncated"] == 1
    assert result["truncated"] is True
    assert result["content"] == "a" * 100


def test_web_fetch_uses_separate_html_extract_byte_limit(monkeypatch, tmp_path) -> None:
    observed: dict[str, int] = {}
    html = (
        "<html><head><title>Example</title></head><body>"
        + ("<div>content</div>" * 30)
        + "</body></html>"
    )

    class FakeStreamResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        url = httpx.URL("https://example.com/article")
        encoding = "utf-8"

        def raise_for_status(self) -> None:
            return None

        async def __aenter__(self) -> "FakeStreamResponse":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def aiter_bytes(self):
            yield html.encode("utf-8")

    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def stream(self, method: str, url: str) -> FakeStreamResponse:
            assert method == "GET"
            assert url == "https://example.com/article"
            return FakeStreamResponse()

    fake_trafilatura = ModuleType("trafilatura")
    fake_trafilatura.extract = lambda *args, **kwargs: "Extracted body"
    monkeypatch.setitem(sys.modules, "trafilatura", fake_trafilatura)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    tool = WebFetchTool(
        ToolRuntimeConfig(
            cwd=tmp_path,
            max_fetch_bytes=128,
            max_html_extract_bytes=1024,
        )
    )
    original_decode = tool._decode_response_text

    def capture_decode(response, raw_bytes: bytes, *, truncated: bool) -> str:
        observed["raw_bytes"] = len(raw_bytes)
        observed["truncated"] = int(truncated)
        return original_decode(response, raw_bytes, truncated=truncated)

    monkeypatch.setattr(tool, "_decode_response_text", capture_decode)

    result = json.loads(asyncio.run(tool.execute({"url": "https://example.com/article"})))

    assert observed["raw_bytes"] == len(html.encode("utf-8"))
    assert observed["truncated"] == 0
    assert result["content"] == "Extracted body"
    assert result["truncated"] is False


def test_web_fetch_persists_full_content_when_result_is_truncated(
    monkeypatch, tmp_path
) -> None:
    text_body = " ".join(f"word{i}" for i in range(120))

    class FakeStreamResponse:
        status_code = 200
        headers = {"content-type": "text/plain; charset=utf-8"}
        url = httpx.URL("https://example.com/long.txt")
        encoding = "utf-8"

        def raise_for_status(self) -> None:
            return None

        async def __aenter__(self) -> "FakeStreamResponse":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def aiter_bytes(self):
            yield text_body.encode("utf-8")

    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def stream(self, method: str, url: str) -> FakeStreamResponse:
            assert method == "GET"
            assert url == "https://example.com/long.txt"
            return FakeStreamResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    tool = WebFetchTool(ToolRuntimeConfig(cwd=tmp_path, max_fetch_chars=20_000))

    result = json.loads(
        asyncio.run(tool.execute({"url": "https://example.com/long.txt", "max_chars": 120}))
    )

    artifact_path = result.get("artifact_path")
    assert result["truncated"] is True
    assert isinstance(artifact_path, str)
    artifact_text = Path(artifact_path).read_text(encoding="utf-8")
    assert "URL: https://example.com/long.txt" in artifact_text
    assert text_body in artifact_text
    assert len(result["content"]) <= 120


def test_web_fetch_truncated_html_without_extraction_fails_cleanly(
    monkeypatch, tmp_path
) -> None:
    html = (
        "<html><head><title>Example</title></head><body>"
        "<script>" + ("x" * 600) + "</script><main>Readable body</main></body></html>"
    )

    class FakeStreamResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        url = httpx.URL("https://example.com/large-page")
        encoding = "utf-8"

        def raise_for_status(self) -> None:
            return None

        async def __aenter__(self) -> "FakeStreamResponse":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def aiter_bytes(self):
            yield html.encode("utf-8")

    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def stream(self, method: str, url: str) -> FakeStreamResponse:
            assert method == "GET"
            assert url == "https://example.com/large-page"
            return FakeStreamResponse()

    fake_trafilatura = ModuleType("trafilatura")
    fake_trafilatura.extract = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "trafilatura", fake_trafilatura)
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    tool = WebFetchTool(
        ToolRuntimeConfig(
            cwd=tmp_path,
            max_fetch_chars=20_000,
            max_fetch_bytes=512_000,
            max_html_extract_bytes=128,
        )
    )

    with pytest.raises(
        ToolExecutionError,
        match="HTML exceeded fetch byte limit before readable text could be extracted",
    ):
        asyncio.run(tool.execute({"url": "https://example.com/large-page"}))


def test_web_fetch_decode_preserves_complete_utf8_when_not_truncated(tmp_path) -> None:
    class FakeResponse:
        encoding = "utf-8"

    tool = WebFetchTool(ToolRuntimeConfig(cwd=tmp_path))

    assert (
        tool._decode_response_text(
            FakeResponse(),
            "你好".encode("utf-8"),
            truncated=False,
        )
        == "你好"
    )


def test_web_fetch_decode_preserves_complete_utf8_prefix_on_exact_byte_boundary(
    tmp_path,
) -> None:
    class FakeResponse:
        encoding = "utf-8"

    tool = WebFetchTool(ToolRuntimeConfig(cwd=tmp_path))

    assert (
        tool._decode_response_text(
            FakeResponse(),
            "你好".encode("utf-8"),
            truncated=True,
        )
        == "你好"
    )


def test_web_fetch_decode_drops_only_incomplete_utf8_suffix_when_truncated(tmp_path) -> None:
    class FakeResponse:
        encoding = "utf-8"

    tool = WebFetchTool(ToolRuntimeConfig(cwd=tmp_path))

    assert (
        tool._decode_response_text(
            FakeResponse(),
            "你好".encode("utf-8")[:-1],
            truncated=True,
        )
        == "你"
    )
