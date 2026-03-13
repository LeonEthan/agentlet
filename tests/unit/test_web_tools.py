from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType

import httpx

from agentlet.agent.tools.policy import ToolRuntimeConfig
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
            "backend": "api",
        }
    ]


def test_web_fetch_extracts_html_title_without_shadowing_html_module(
    monkeypatch, tmp_path
) -> None:
    class FakeResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        text = "<html><title>Hello &amp; goodbye</title><body>Body</body></html>"
        url = httpx.URL("https://example.com/page")

        def raise_for_status(self) -> None:
            return None

    class FakeAsyncClient:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def __aenter__(self) -> "FakeAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str) -> FakeResponse:
            assert url == "https://example.com/page"
            return FakeResponse()

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
