from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType

from agentlet.agent.tools.policy import ToolRuntimeConfig
from agentlet.agent.tools.web import WebSearchTool


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
