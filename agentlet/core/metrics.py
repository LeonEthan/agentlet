"""Metrics collection for observability and cost tracking."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from time import time

from agentlet.core.types import get_logger

logger = get_logger("agentlet.metrics")


@dataclass
class Counter:
    """Monotonically increasing counter metric."""
    name: str
    description: str
    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def get(self) -> float:
        with self._lock:
            return self._value

    def reset(self) -> float:
        with self._lock:
            prev = self._value
            self._value = 0.0
            return prev


@dataclass
class Gauge:
    """Point-in-time gauge metric."""
    name: str
    description: str
    _value: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set(self, value: float) -> None:
        with self._lock:
            self._value = value

    def get(self) -> float:
        with self._lock:
            return self._value

    def inc(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value += amount

    def dec(self, amount: float = 1.0) -> None:
        with self._lock:
            self._value -= amount


@dataclass
class Histogram:
    """Histogram for tracking value distributions."""
    name: str
    description: str
    buckets: list[float] = field(default_factory=lambda: [.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10])
    _values: deque[float] = field(default_factory=lambda: deque(maxlen=10000))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float) -> None:
        with self._lock:
            self._values.append(value)

    def get_stats(self) -> dict[str, float]:
        with self._lock:
            if not self._values:
                return {"count": 0, "sum": 0.0, "avg": 0.0, "min": 0.0, "max": 0.0}
            sorted_vals = sorted(self._values)
            n = len(sorted_vals)
            return {
                "count": n, "sum": sum(sorted_vals), "avg": sum(sorted_vals) / n,
                "min": sorted_vals[0], "max": sorted_vals[-1],
                "p50": sorted_vals[n // 2],
                "p95": sorted_vals[int(n * 0.95)] if n >= 20 else sorted_vals[-1],
                "p99": sorted_vals[int(n * 0.99)] if n >= 100 else sorted_vals[-1],
            }

    def get_bucket_counts(self) -> dict[str, int]:
        with self._lock:
            counts = {f"le_{b}": 0 for b in self.buckets}
            counts["le_inf"] = len(self._values)
            for v in self._values:
                for b in self.buckets:
                    if v <= b:
                        counts[f"le_{b}"] += 1
            return counts


# Cost per 1M tokens (as of 2024)
_DEFAULT_PRICING = {
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-3-opus": {"input": 15.00, "output": 75.00},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
}


@dataclass
class LLMUsageMetrics:
    """Track LLM usage with automatic cost calculation."""

    pricing: dict[str, dict[str, float]] = field(default_factory=lambda: _DEFAULT_PRICING.copy())
    requests_total: Counter = field(init=False)
    tokens_input_total: Counter = field(init=False)
    tokens_output_total: Counter = field(init=False)
    cost_usd_total: Counter = field(init=False)
    latency_seconds: Histogram = field(init=False)
    tokens_per_request: Histogram = field(init=False)
    active_requests: Gauge = field(init=False)

    def __post_init__(self):
        self.requests_total = Counter("llm_requests_total", "Total LLM requests")
        self.tokens_input_total = Counter("llm_tokens_input_total", "Total input tokens")
        self.tokens_output_total = Counter("llm_tokens_output_total", "Total output tokens")
        self.cost_usd_total = Counter("llm_cost_usd_total", "Total estimated cost in USD")
        self.latency_seconds = Histogram("llm_latency_seconds", "Request latency in seconds")
        self.tokens_per_request = Histogram("llm_tokens_per_request", "Total tokens per request",
                                            buckets=[100, 500, 1000, 2000, 4000, 8000, 16000, 32000])
        self.active_requests = Gauge("llm_active_requests", "Currently active requests")

    def record_request(self, model: str, input_tokens: int, output_tokens: int, latency_seconds: float) -> dict[str, float]:
        total_tokens = input_tokens + output_tokens
        self.requests_total.inc()
        self.tokens_input_total.inc(input_tokens)
        self.tokens_output_total.inc(output_tokens)
        self.latency_seconds.observe(latency_seconds)
        self.tokens_per_request.observe(total_tokens)

        cost = self._calculate_cost(model, input_tokens, output_tokens)
        if cost > 0:
            self.cost_usd_total.inc(cost)

        return {"input_tokens": input_tokens, "output_tokens": output_tokens,
                "total_tokens": total_tokens, "latency_seconds": latency_seconds, "cost_usd": cost}

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        sorted_pricing = sorted(self.pricing.items(), key=lambda x: len(x[0]), reverse=True)
        for model_prefix, prices in sorted_pricing:
            if model.startswith(model_prefix) or model_prefix in model:
                return (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices["output"]
        return 0.0

    def get_summary(self) -> dict[str, object]:
        return {
            "requests": {"total": self.requests_total.get(), "active": self.active_requests.get()},
            "tokens": {"input": self.tokens_input_total.get(), "output": self.tokens_output_total.get(),
                       "total": self.tokens_input_total.get() + self.tokens_output_total.get()},
            "cost_usd": round(self.cost_usd_total.get(), 6),
            "latency": self.latency_seconds.get_stats(),
        }


@dataclass
class ToolMetrics:
    """Track tool execution metrics."""
    executions_total: Counter = field(init=False)
    execution_errors_total: Counter = field(init=False)
    execution_duration_seconds: Histogram = field(init=False)

    def __post_init__(self):
        self.executions_total = Counter("tool_executions_total", "Total tool executions")
        self.execution_errors_total = Counter("tool_execution_errors_total", "Total tool execution errors")
        self.execution_duration_seconds = Histogram("tool_execution_duration_seconds", "Tool execution duration")

    def record_execution(self, tool_name: str, duration_seconds: float, error: bool = False) -> None:
        self.executions_total.inc()
        self.execution_duration_seconds.observe(duration_seconds)
        if error:
            self.execution_errors_total.inc()

    def get_summary(self) -> dict[str, object]:
        total = self.executions_total.get()
        errors = self.execution_errors_total.get()
        return {"executions_total": total, "errors_total": errors,
                "error_rate": errors / max(total, 1), "duration": self.execution_duration_seconds.get_stats()}


@dataclass
class MetricsRegistry:
    """Central registry for all application metrics."""
    llm: LLMUsageMetrics = field(default_factory=LLMUsageMetrics)
    tools: ToolMetrics = field(default_factory=ToolMetrics)
    sessions_started: Counter = field(init=False)
    sessions_completed: Counter = field(init=False)
    interrupts_total: Counter = field(init=False)
    approvals_requested: Counter = field(init=False)
    approvals_granted: Counter = field(init=False)

    def __post_init__(self):
        self.sessions_started = Counter("sessions_started_total", "Total sessions started")
        self.sessions_completed = Counter("sessions_completed_total", "Total sessions completed")
        self.interrupts_total = Counter("interrupts_total", "Total interrupts")
        self.approvals_requested = Counter("approvals_requested_total", "Total approval requests")
        self.approvals_granted = Counter("approvals_granted_total", "Total approvals granted")

    def get_all_metrics(self) -> dict[str, object]:
        return {
            "llm": self.llm.get_summary(),
            "tools": self.tools.get_summary(),
            "sessions": {"started": self.sessions_started.get(), "completed": self.sessions_completed.get()},
            "interrupts": self.interrupts_total.get(),
            "approvals": {"requested": self.approvals_requested.get(), "granted": self.approvals_granted.get()},
            "timestamp": time(),
        }

    def export_prometheus(self) -> str:
        lines = []

        def fmt_counter(c: Counter):
            return f"{c.name} {c.get()}"

        def fmt_histogram(h: Histogram):
            stats = h.get_stats()
            buckets = h.get_bucket_counts()
            result = [f'{h.name}_bucket{{le="{b}"}} {buckets[f"le_{b}"]}' for b in h.buckets]
            result.append(f'{h.name}_bucket{{le="+Inf"}} {buckets["le_inf"]}')
            result.append(f"{h.name}_sum {stats['sum']}")
            result.append(f"{h.name}_count {stats['count']}")
            return "\n".join(result)

        metrics = [
            ("# HELP", self.llm.requests_total), ("# TYPE", self.llm.requests_total, "counter"),
            ("", self.llm.requests_total), ("# HELP", self.llm.tokens_input_total),
            ("# TYPE", self.llm.tokens_input_total, "counter"), ("", self.llm.tokens_input_total),
            ("# HELP", self.llm.tokens_output_total), ("# TYPE", self.llm.tokens_output_total, "counter"),
            ("", self.llm.tokens_output_total), ("# HELP", self.llm.cost_usd_total),
            ("# TYPE", self.llm.cost_usd_total, "counter"), ("", self.llm.cost_usd_total),
            ("# HELP", self.llm.latency_seconds), ("# TYPE", self.llm.latency_seconds, "histogram"),
            ("hist", self.llm.latency_seconds), ("# HELP", self.tools.executions_total),
            ("# TYPE", self.tools.executions_total, "counter"), ("", self.tools.executions_total),
            ("# HELP", self.tools.execution_errors_total), ("# TYPE", self.tools.execution_errors_total, "counter"),
            ("", self.tools.execution_errors_total),
        ]

        for item in metrics:
            if item[0] == "# HELP":
                lines.append(f"# HELP {item[1].name} {item[1].description}")
            elif item[0] == "# TYPE":
                lines.append(f"# TYPE {item[1].name} {item[2]}")
            elif item[0] == "hist":
                lines.append(fmt_histogram(item[1]))
            elif item[0] == "":
                lines.append(fmt_counter(item[1]))

        return "\n".join(lines)


_global_registry: MetricsRegistry | None = None
_registry_lock = threading.Lock()


def get_metrics_registry() -> MetricsRegistry:
    global _global_registry
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = MetricsRegistry()
    return _global_registry


def set_metrics_registry(registry: MetricsRegistry) -> None:
    global _global_registry
    with _registry_lock:
        _global_registry = registry


__all__ = [
    "Counter", "Gauge", "Histogram", "LLMUsageMetrics", "MetricsExporter",
    "MetricsRegistry", "ToolMetrics", "get_metrics_registry", "set_metrics_registry",
]
