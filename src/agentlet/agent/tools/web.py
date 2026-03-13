from __future__ import annotations

"""Web tools: WebSearch and WebFetch."""

import html
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from agentlet.agent.tools.policy import ToolRuntimeConfig
from agentlet.agent.tools.registry import Tool, ToolExecutionError, ToolSpec, build_tool_result_content


class WebSearchTool(Tool):
    """Search the web using DuckDuckGo."""

    def __init__(self, runtime: ToolRuntimeConfig) -> None:
        self.runtime = runtime

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_search",
            description="Search the web using DuckDuckGo. Returns ranked results with title, URL, and snippet.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 5)",
                        "minimum": 1,
                        "maximum": 20,
                    },
                    "region": {
                        "type": "string",
                        "description": "Region code (e.g., 'us-en', 'uk-en')",
                    },
                    "safesearch": {
                        "type": "string",
                        "description": "SafeSearch setting: 'on', 'moderate', or 'off'",
                        "enum": ["on", "moderate", "off"],
                    },
                },
                "required": ["query"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        query = arguments.get("query", "").strip()
        max_results = arguments.get("max_results") or 5
        region = arguments.get("region")
        safesearch = arguments.get("safesearch", "moderate")

        if not query:
            raise ToolExecutionError("Search query cannot be empty.")

        # Clamp max_results
        max_results = max(1, min(max_results, self.runtime.max_search_results))

        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise ToolExecutionError(
                "Web search requires the 'ddgs' package. Install with: pip install ddgs"
            )

        try:
            with DDGS() as ddgs:
                results = []
                ddgs_results = ddgs.text(
                    query,
                    max_results=max_results,
                    region=region,
                    safesearch=safesearch,
                    backend="api",  # Use the default working backend
                )
                for rank, result in enumerate(ddgs_results, start=1):
                    results.append({
                        "rank": rank,
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                        "source": "duckduckgo",
                    })

            return build_tool_result_content({
                "ok": True,
                "tool": "web_search",
                "query": query,
                "results": results,
            })
        except Exception as exc:
            raise ToolExecutionError(f"Web search failed: {exc}") from exc


class WebFetchTool(Tool):
    """Fetch and extract readable content from a web page."""

    def __init__(self, runtime: ToolRuntimeConfig) -> None:
        self.runtime = runtime

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_fetch",
            description="Fetch a web page and extract readable content. Returns title, content, and metadata.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Maximum characters to return",
                        "minimum": 100,
                        "maximum": 100_000,
                    },
                },
                "required": ["url"],
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        url = arguments.get("url", "").strip()
        max_chars = arguments.get("max_chars") or self.runtime.max_fetch_chars

        if not url:
            raise ToolExecutionError("URL cannot be empty.")

        # Validate URL scheme
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ToolExecutionError(
                f"Unsupported URL scheme: {parsed.scheme or 'none'}. "
                "Only http and https are supported."
            )

        # Clamp max_chars
        max_chars = max(100, min(max_chars, 100_000))

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.runtime.web_timeout_seconds,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.0"
                    ),
                },
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

                final_url = str(response.url)
                content_type = response.headers.get("content-type", "").lower()

                # Cache response text to avoid re-decoding
                response_text = response.text

                # Try to extract readable content
                title = ""
                content = ""

                if "text/html" in content_type:
                    try:
                        from trafilatura import extract

                        extracted = extract(
                            response_text,
                            include_comments=False,
                            include_tables=True,
                            deduplicate=True,
                            url=final_url,
                        )
                        if extracted:
                            content = extracted
                        else:
                            # Fallback: use the raw text with basic HTML stripping
                            content = self._extract_text_fallback(response_text)
                    except ImportError:
                        # trafilatura not installed, use fallback
                        content = self._extract_text_fallback(response_text)
                elif "text/" in content_type:
                    content = response_text
                else:
                    raise ToolExecutionError(
                        f"Unsupported content type: {content_type}. "
                        "Only text content can be fetched."
                    )

                # Extract title from HTML if we can
                if "text/html" in content_type:
                    title = self._extract_title(response_text)

                # Apply character limit
                truncated = len(content) > max_chars
                if truncated:
                    # Try to break at a word boundary
                    truncated_content = content[:max_chars]
                    last_space = truncated_content.rfind(" ")
                    if last_space > max_chars * 0.8:
                        truncated_content = truncated_content[:last_space]
                    content = truncated_content

                return build_tool_result_content({
                    "ok": True,
                    "tool": "web_fetch",
                    "url": url,
                    "final_url": final_url,
                    "status_code": response.status_code,
                    "title": title,
                    "content": content,
                    "truncated": truncated,
                })

        except httpx.HTTPStatusError as exc:
            raise ToolExecutionError(
                f"HTTP error {exc.response.status_code} fetching {url}"
            ) from exc
        except httpx.RequestError as exc:
            raise ToolExecutionError(f"Failed to fetch {url}: {exc}") from exc
        except Exception as exc:
            raise ToolExecutionError(f"Failed to fetch {url}: {exc}") from exc

    def _extract_text_fallback(self, html: str) -> str:
        """Basic HTML to text fallback when trafilatura is not available."""
        # Remove script and style tags and their content
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)

        # Replace common block elements with newlines
        text = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<(br|hr)[^>]*>", "\n", text, flags=re.IGNORECASE)

        # Remove remaining HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Decode common HTML entities
        text = html.unescape(text)

        # Normalize whitespace
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)

        return text

    def _extract_title(self, html: str) -> str:
        """Extract title from HTML."""
        match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if match:
            return html.unescape(match.group(1).strip())
        return ""
