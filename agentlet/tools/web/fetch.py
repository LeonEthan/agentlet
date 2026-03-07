"""Built-in web fetch tool."""

from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import re
from typing import Callable
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from agentlet.core.types import JSONObject
from agentlet.tools.base import ToolDefinition, ToolResult

_DEFAULT_TIMEOUT_SECONDS = 10.0
_MAX_RESPONSE_BYTES = 1_000_000
_TEXTUAL_CONTENT_TYPES = {
    "application/atom+xml",
    "application/javascript",
    "application/json",
    "application/ld+json",
    "application/rss+xml",
    "application/xhtml+xml",
    "application/xml",
}
_BLOCK_TAGS = {
    "article",
    "aside",
    "blockquote",
    "br",
    "div",
    "figcaption",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "td",
    "th",
    "tr",
    "ul",
}
_IGNORED_TAGS = {"noscript", "script", "style", "svg"}
_USER_AGENT = (
    "Mozilla/5.0 (compatible; agentlet/0.1; +https://example.invalid/agentlet)"
)
_WHITESPACE_RE = re.compile(r"[ \t\f\v]+")


@dataclass(frozen=True, slots=True)
class _FetchedResponse:
    requested_url: str
    final_url: str
    status_code: int
    content_type: str | None
    body: bytes
    response_truncated: bool = False


class WebFetchTool:
    """Fetch one URL and normalize the response into usable text."""

    def __init__(
        self,
        *,
        fetch_url: Callable[[str, float | None], _FetchedResponse] | None = None,
    ) -> None:
        self._fetch_url = fetch_url or _fetch_url
        self._definition = ToolDefinition(
            name="WebFetch",
            description=(
                "Fetch a specific URL and return normalized text extracted from "
                "the response."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "format": "uri"},
                    "max_chars": {"type": "integer", "minimum": 1},
                    "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            approval_category="external_or_interrupt",
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def execute(self, arguments: JSONObject) -> ToolResult:
        try:
            url = _require_http_url(arguments)
            max_chars = _optional_positive_int(
                arguments,
                "max_chars",
                tool_name="WebFetch",
            )
            timeout_seconds = _optional_positive_number(
                arguments,
                "timeout_seconds",
                tool_name="WebFetch",
            )
        except ValueError as exc:
            return ToolResult.error(str(exc))

        try:
            response = self._fetch_url(url, timeout_seconds)
        except OSError as exc:
            return ToolResult.error(
                f"WebFetch failed: {exc}",
                metadata={
                    "url": url,
                    "error_type": type(exc).__name__,
                },
            )

        try:
            output, metadata = _normalize_response(response, max_chars=max_chars)
        except ValueError as exc:
            return ToolResult.error(
                str(exc),
                metadata=_base_metadata(
                    response,
                    max_chars=max_chars,
                ),
            )

        if timeout_seconds is not None:
            metadata["timeout_seconds"] = timeout_seconds
        return ToolResult(output=output, metadata=metadata)


def _fetch_url(url: str, timeout_seconds: float | None) -> _FetchedResponse:
    request = Request(
        url,
        headers={
            "Accept": (
                "text/html,application/xhtml+xml,text/plain,application/json;q=0.9,*/*;q=0.1"
            ),
            "User-Agent": _USER_AGENT,
        },
    )
    timeout = timeout_seconds or _DEFAULT_TIMEOUT_SECONDS
    with urlopen(request, timeout=timeout) as response:
        body = response.read(_MAX_RESPONSE_BYTES + 1)
        response_truncated = len(body) > _MAX_RESPONSE_BYTES
        if response_truncated:
            body = body[:_MAX_RESPONSE_BYTES]
        return _FetchedResponse(
            requested_url=url,
            final_url=response.geturl(),
            status_code=getattr(response, "status", 200),
            content_type=response.headers.get("Content-Type"),
            body=body,
            response_truncated=response_truncated,
        )


def _normalize_response(
    response: _FetchedResponse,
    *,
    max_chars: int | None,
) -> tuple[str, JSONObject]:
    content_type = _normalized_content_type(response.content_type)
    decoded_body = _decode_body(response.body, response.content_type)
    title = ""

    if _is_html_content_type(content_type, decoded_body):
        parser = _HTMLTextExtractor()
        parser.feed(decoded_body)
        parser.close()
        title = parser.title
        body_text = parser.text
        output = body_text
        if title and not body_text.startswith(title):
            output = f"# {title}\n\n{body_text}" if body_text else f"# {title}"
        source_format = "html"
    elif _is_textual_content_type(content_type):
        output = _normalize_text(decoded_body)
        source_format = "text"
    else:
        readable_type = content_type or "unknown"
        raise ValueError(f"WebFetch does not support content type: {readable_type}")

    if not output:
        raise ValueError("WebFetch found no usable text content.")

    truncated = False
    if max_chars is not None and len(output) > max_chars:
        output = output[:max_chars]
        truncated = True

    metadata = _base_metadata(
        response,
        max_chars=max_chars,
        source_format=source_format,
        title=title or None,
        extracted_chars=len(output),
        truncated=truncated,
    )
    return output, metadata


def _base_metadata(
    response: _FetchedResponse,
    *,
    max_chars: int | None,
    source_format: str | None = None,
    title: str | None = None,
    extracted_chars: int | None = None,
    truncated: bool | None = None,
) -> JSONObject:
    metadata: JSONObject = {
        "url": response.requested_url,
        "final_url": response.final_url,
        "status_code": response.status_code,
        "content_type": response.content_type or "",
        "response_truncated": response.response_truncated,
    }
    if max_chars is not None:
        metadata["max_chars"] = max_chars
    if source_format is not None:
        metadata["source_format"] = source_format
    if title is not None:
        metadata["title"] = title
    if extracted_chars is not None:
        metadata["extracted_chars"] = extracted_chars
    if truncated is not None:
        metadata["truncated"] = truncated
    return metadata


def _require_http_url(arguments: JSONObject) -> str:
    value = arguments.get("url")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("WebFetch requires a non-empty string 'url'.")
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("WebFetch url must be an absolute http or https URL.")
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


def _decode_body(body: bytes, content_type: str | None) -> str:
    charset = "utf-8"
    if content_type:
        for part in content_type.split(";")[1:]:
            name, _, raw_value = part.partition("=")
            if name.strip().lower() == "charset" and raw_value.strip():
                charset = raw_value.strip().strip('"')
                break
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _normalized_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""
    return content_type.split(";", maxsplit=1)[0].strip().lower()


def _is_html_content_type(content_type: str, decoded_body: str) -> bool:
    if content_type in {"application/xhtml+xml", "text/html"}:
        return True
    sample = decoded_body.lstrip()[:128].lower()
    return sample.startswith("<!doctype html") or sample.startswith("<html")


def _is_textual_content_type(content_type: str) -> bool:
    return content_type.startswith("text/") or content_type in _TEXTUAL_CONTENT_TYPES


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._ignore_depth = 0
        self._inside_title = False
        self._title_parts: list[str] = []
        self._text_parts: list[str] = []

    @property
    def title(self) -> str:
        return _normalize_text("".join(self._title_parts))

    @property
    def text(self) -> str:
        return _normalize_text("".join(self._text_parts))

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if tag in _IGNORED_TAGS:
            self._ignore_depth += 1
            return
        if self._ignore_depth:
            return
        if tag == "title":
            self._inside_title = True
            return
        if tag in _BLOCK_TAGS:
            self._append_break()
            if tag == "li":
                self._text_parts.append("- ")

    def handle_endtag(self, tag: str) -> None:
        if tag in _IGNORED_TAGS:
            if self._ignore_depth:
                self._ignore_depth -= 1
            return
        if self._ignore_depth:
            return
        if tag == "title":
            self._inside_title = False
            return
        if tag in _BLOCK_TAGS:
            self._append_break()

    def handle_data(self, data: str) -> None:
        if self._ignore_depth:
            return
        if self._inside_title:
            self._title_parts.append(data)
            return
        self._text_parts.append(data)

    def _append_break(self) -> None:
        if self._text_parts and self._text_parts[-1].endswith("\n\n"):
            return
        self._text_parts.append("\n\n")


def _normalize_text(value: str) -> str:
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    blank_pending = False

    for raw_line in normalized.split("\n"):
        line = _WHITESPACE_RE.sub(" ", raw_line).strip()
        if line:
            if blank_pending and lines:
                lines.append("")
            lines.append(line)
            blank_pending = False
        else:
            blank_pending = True

    return "\n".join(lines).strip()


__all__ = ["WebFetchTool"]
