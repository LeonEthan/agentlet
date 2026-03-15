from __future__ import annotations

"""Web tools: WebSearch and WebFetch."""

import codecs
import html as html_lib
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx

# Pre-compiled regex patterns for HTML text extraction
_SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.DOTALL | re.IGNORECASE)
_STYLE_RE = re.compile(r"<style[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE)
_BLOCK_ELEM_RE = re.compile(r"</(p|div|h[1-6]|li|tr)>", re.IGNORECASE)
_BR_RE = re.compile(r"<(br|hr)[^>]*>", re.IGNORECASE)
_ALL_TAGS_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
from agentlet.agent.tools.policy import ToolRuntimeConfig
from agentlet.agent.tools.registry import Tool, ToolExecutionError, ToolSpec, build_tool_result_content


class WebSearchTool(Tool):
    """Search the web using ddgs."""

    def __init__(self, runtime: ToolRuntimeConfig) -> None:
        self.runtime = runtime

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="web_search",
            description=(
                "Search the web using ddgs and return ranked results with title, URL, "
                "snippet, and normalized source metadata."
            ),
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
            from ddgs import DDGS
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
                )
                for rank, result in enumerate(ddgs_results, start=1):
                    results.append({
                        "rank": rank,
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                        "source": result.get("source") or result.get("provider") or "ddgs",
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
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                },
            ) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()

                    final_url = str(response.url)
                    content_type = response.headers.get("content-type", "").lower()
                    is_html = "text/html" in content_type
                    max_bytes = (
                        self.runtime.max_html_extract_bytes
                        if is_html
                        else self.runtime.max_fetch_bytes
                    )

                    if not is_html and "text/" not in content_type:
                        raise ToolExecutionError(
                            f"Unsupported content type: {content_type}. "
                            "Only text content can be fetched."
                        )

                    raw_bytes, response_byte_truncated = await self._read_limited_bytes(
                        response,
                        max_bytes=max_bytes,
                    )

                response_text = self._decode_response_text(
                    response,
                    raw_bytes,
                    truncated=response_byte_truncated,
                )

                # Try to extract readable content
                title = ""
                content = ""

                if is_html:
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
                        elif response_byte_truncated:
                            raise ToolExecutionError(
                                "HTML exceeded fetch byte limit before readable text "
                                "could be extracted; retry with a larger fetch budget."
                            )
                        else:
                            # Fallback: use the raw text with basic HTML stripping
                            content = self._extract_text_fallback(response_text)
                    except ImportError:
                        if response_byte_truncated:
                            raise ToolExecutionError(
                                "HTML exceeded fetch byte limit before readable text "
                                "could be extracted; install trafilatura or retry with "
                                "a larger fetch budget."
                            )
                        # trafilatura not installed, use fallback for complete documents
                        content = self._extract_text_fallback(response_text)
                else:
                    content = response_text

                # Extract title from HTML if we can
                if is_html:
                    title = self._extract_title(response_text)

                # Apply character limit
                full_content = content
                content_truncated = len(full_content) > max_chars
                artifact_path: str | None = None
                if content_truncated:
                    artifact_path = self._persist_fetch_artifact(
                        final_url=final_url,
                        title=title,
                        content=full_content,
                    )
                    # Try to break at a word boundary
                    truncated_content = full_content[:max_chars]
                    last_space = truncated_content.rfind(" ")
                    if last_space > max_chars * 0.8:
                        truncated_content = truncated_content[:last_space]
                    content = truncated_content
                else:
                    content = full_content

                truncated = response_byte_truncated or content_truncated

                payload = {
                    "ok": True,
                    "tool": "web_fetch",
                    "url": url,
                    "final_url": final_url,
                    "status_code": response.status_code,
                    "title": title,
                    "content": content,
                    "truncated": truncated,
                }
                if artifact_path is not None:
                    payload["artifact_path"] = artifact_path

                return build_tool_result_content(payload)

        except httpx.HTTPStatusError as exc:
            raise ToolExecutionError(
                f"HTTP error {exc.response.status_code} fetching {url}"
            ) from exc
        except httpx.RequestError as exc:
            raise ToolExecutionError(f"Failed to fetch {url}: {exc}") from exc
        except ToolExecutionError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"Failed to fetch {url}: {exc}") from exc

    async def _read_limited_bytes(
        self,
        response: httpx.Response,
        *,
        max_bytes: int,
    ) -> tuple[bytes, bool]:
        """Read at most max_bytes from a streamed response."""
        buffer = bytearray()
        truncated = False

        async for chunk in response.aiter_bytes():
            remaining = max_bytes - len(buffer)
            if remaining <= 0:
                truncated = True
                break
            if len(chunk) > remaining:
                buffer.extend(chunk[:remaining])
                truncated = True
                break
            buffer.extend(chunk)

        return bytes(buffer), truncated

    def _decode_response_text(
        self,
        response: httpx.Response,
        raw_bytes: bytes,
        *,
        truncated: bool,
    ) -> str:
        """Decode streamed bytes while preserving complete characters at byte limits."""
        encoding = response.encoding or "utf-8"

        try:
            decoder_cls = codecs.getincrementaldecoder(encoding)
        except LookupError:
            decoder_cls = codecs.getincrementaldecoder("utf-8")

        decoder = decoder_cls(errors="replace")
        return decoder.decode(raw_bytes, final=not truncated)

    def _persist_fetch_artifact(
        self,
        *,
        final_url: str,
        title: str,
        content: str,
    ) -> str:
        """Persist large fetch content so the full text remains available."""
        artifact_dir = Path(tempfile.gettempdir()) / "agentlet-fetch"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = artifact_dir / f"{uuid4().hex}.txt"

        artifact_body = self._build_artifact_body(
            final_url=final_url,
            title=title,
            content=content,
        )
        try:
            artifact_path.write_text(artifact_body, encoding="utf-8")
            import os

            if os.name != "nt":
                artifact_path.chmod(0o600)
        except OSError as exc:
            raise ToolExecutionError(
                f"Failed to persist fetched content artifact: {exc.strerror or exc}"
            ) from exc
        return str(artifact_path)

    def _build_artifact_body(
        self,
        *,
        final_url: str,
        title: str,
        content: str,
    ) -> str:
        lines = [f"URL: {final_url}"]
        if title:
            lines.append(f"Title: {title}")
        lines.append("")
        lines.append(content)
        return "\n".join(lines)

    def _extract_text_fallback(self, html_text: str) -> str:
        """Basic HTML to text fallback when trafilatura is not available."""
        # Remove script and style tags and their content
        text = _SCRIPT_RE.sub("", html_text)
        text = _STYLE_RE.sub("", text)

        # Replace common block elements with newlines
        text = _BLOCK_ELEM_RE.sub("\n", text)
        text = _BR_RE.sub("\n", text)

        # Remove remaining HTML tags
        text = _ALL_TAGS_RE.sub("", text)

        # Decode common HTML entities
        text = html_lib.unescape(text)

        # Normalize whitespace
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)

        return text

    def _extract_title(self, html_text: str) -> str:
        """Extract title from HTML."""
        match = _TITLE_RE.search(html_text)
        if match:
            return html_lib.unescape(match.group(1).strip())
        return ""
