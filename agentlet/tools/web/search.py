"""Built-in web search tool."""

from __future__ import annotations

from html.parser import HTMLParser
import re
from typing import Callable
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse
from urllib.request import Request, urlopen

from agentlet.core.types import JSONObject
from agentlet.tools.base import ToolDefinition, ToolResult

_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_MAX_RESULTS = 5
_MAX_RESPONSE_BYTES = 500_000
_RESULT_TITLE_CLASSES = {"result__a", "result-link"}
_RESULT_SNIPPET_CLASSES = {"result__snippet", "result-snippet"}
_SEARCH_ENDPOINT = "https://duckduckgo.com/html/?q={query}"
_USER_AGENT = (
    "Mozilla/5.0 (compatible; agentlet/0.1; +https://example.invalid/agentlet)"
)
_WHITESPACE_RE = re.compile(r"\s+")


class WebSearchTool:
    """Search the web and return ranked result summaries."""

    def __init__(
        self,
        *,
        fetch_html: Callable[[str, float | None], str] | None = None,
    ) -> None:
        self._fetch_html = fetch_html or _fetch_search_html
        self._definition = ToolDefinition(
            name="WebSearch",
            description=(
                "Search the web for current information and return ranked result "
                "titles, URLs, and snippets."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "minimum": 1},
                    "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            approval_category="external_or_interrupt",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, arguments: JSONObject) -> ToolResult:
        try:
            query = _require_non_empty_string(arguments, "query", tool_name="WebSearch")
            max_results = _optional_positive_int(
                arguments,
                "max_results",
                tool_name="WebSearch",
            )
            timeout_seconds = _optional_positive_number(
                arguments,
                "timeout_seconds",
                tool_name="WebSearch",
            )
        except ValueError as exc:
            return ToolResult.error(str(exc))

        normalized_max_results = max_results or _DEFAULT_MAX_RESULTS
        search_url = _SEARCH_ENDPOINT.format(query=quote_plus(query))

        try:
            html = self._fetch_html(search_url, timeout_seconds)
        except OSError as exc:
            return ToolResult.error(
                f"WebSearch failed: {exc}",
                metadata={
                    "query": query,
                    "search_url": search_url,
                    "error_type": type(exc).__name__,
                },
            )

        raw_results = _SearchHTMLParser.parse(html)
        ranked_results = []
        for index, item in enumerate(raw_results[:normalized_max_results], start=1):
            ranked_item: JSONObject = {
                "rank": index,
                "title": item["title"],
                "url": item["url"],
            }
            if item["snippet"]:
                ranked_item["snippet"] = item["snippet"]
            ranked_results.append(ranked_item)

        metadata: JSONObject = {
            "query": query,
            "search_url": search_url,
            "count": len(ranked_results),
            "results": ranked_results,
        }
        if max_results is not None:
            metadata["max_results"] = max_results
        if timeout_seconds is not None:
            metadata["timeout_seconds"] = timeout_seconds

        if not ranked_results:
            return ToolResult(output="No search results found.", metadata=metadata)

        return ToolResult(
            output=_format_search_results(ranked_results),
            metadata=metadata,
        )


def _fetch_search_html(url: str, timeout_seconds: float | None) -> str:
    request = Request(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": _USER_AGENT,
        },
    )
    timeout = timeout_seconds or _DEFAULT_TIMEOUT_SECONDS
    with urlopen(request, timeout=timeout) as response:
        body = response.read(_MAX_RESPONSE_BYTES + 1)
        charset = response.headers.get_content_charset() or "utf-8"
    if len(body) > _MAX_RESPONSE_BYTES:
        body = body[:_MAX_RESPONSE_BYTES]
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


class _SearchHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._capture_title = False
        self._capture_snippet = False
        self._current_href = ""
        self._current_tag = ""
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []

    @classmethod
    def parse(cls, html: str) -> list[dict[str, str]]:
        parser = cls()
        parser.feed(html)
        parser.close()
        return parser.results

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        classes = set(attr_map.get("class", "").split())

        if tag == "a" and classes.intersection(_RESULT_TITLE_CLASSES):
            href = attr_map.get("href", "").strip()
            if href:
                self._capture_title = True
                self._current_href = href
                self._current_tag = tag
                self._title_parts = []
            return

        if classes.intersection(_RESULT_SNIPPET_CLASSES) and self.results:
            if not self.results[-1].get("snippet"):
                self._capture_snippet = True
                self._current_tag = tag
                self._snippet_parts = []
            return

    def handle_endtag(self, tag: str) -> None:
        if self._capture_title and tag == self._current_tag:
            title = _normalize_inline_text("".join(self._title_parts))
            if title:
                self.results.append(
                    {
                        "title": title,
                        "url": _normalize_result_url(self._current_href),
                        "snippet": "",
                    }
                )
            self._capture_title = False
            self._current_href = ""
            self._title_parts = []
            self._current_tag = ""
            return

        if self._capture_snippet and tag == self._current_tag:
            snippet = _normalize_inline_text("".join(self._snippet_parts))
            if snippet and self.results:
                self.results[-1]["snippet"] = snippet
            self._capture_snippet = False
            self._snippet_parts = []
            self._current_tag = ""

    def handle_data(self, data: str) -> None:
        if self._capture_title:
            self._title_parts.append(data)
        elif self._capture_snippet:
            self._snippet_parts.append(data)


def _format_search_results(results: list[JSONObject]) -> str:
    blocks: list[str] = []
    for item in results:
        block_lines = [
            f"{item['rank']}. {item['title']}",
            f"URL: {item['url']}",
        ]
        snippet = item.get("snippet")
        if isinstance(snippet, str) and snippet:
            block_lines.append(f"Snippet: {snippet}")
        blocks.append("\n".join(block_lines))
    return "\n\n".join(blocks)


def _normalize_result_url(raw_url: str) -> str:
    absolute_url = urljoin("https://duckduckgo.com", raw_url.strip())
    parsed = urlparse(absolute_url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        redirect_target = parse_qs(parsed.query).get("uddg")
        if redirect_target:
            return unquote(redirect_target[0])
    return absolute_url


def _normalize_inline_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _require_non_empty_string(
    arguments: JSONObject,
    key: str,
    *,
    tool_name: str,
) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{tool_name} requires a non-empty string '{key}'.")
    return value.strip()


def _optional_positive_int(
    arguments: JSONObject,
    key: str,
    *,
    tool_name: str,
) -> int | None:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{tool_name} requires integer '{key}' > 0 when provided.")
    return value


def _optional_positive_number(
    arguments: JSONObject,
    key: str,
    *,
    tool_name: str,
) -> float | None:
    value = arguments.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(
            f"{tool_name} requires number '{key}' > 0 when provided."
        )
    return float(value)


__all__ = ["WebSearchTool"]
