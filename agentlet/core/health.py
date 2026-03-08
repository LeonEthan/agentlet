"""Health check system for monitoring system status."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Callable, Protocol

from agentlet.core.types import get_logger

logger = get_logger("agentlet.health")


class HealthStatus:
    """Health status constants."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass(frozen=True, slots=True)
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    status: str
    message: str
    details: dict[str, object] = field(default_factory=dict)
    response_time_ms: float = 0.0


class HealthCheck(Protocol):
    """Protocol for health check implementations."""
    def check(self) -> HealthCheckResult:
        """Execute health check and return result."""


@dataclass
class DiskSpaceCheck:
    """Check available disk space."""
    path: Path = field(default_factory=Path.cwd)
    min_free_gb: float = 1.0
    name: str = "disk_space"

    def check(self) -> HealthCheckResult:
        start = monotonic()
        try:
            usage = shutil.disk_usage(self.path)
            free_gb = usage.free / (1024**3)
            used_percent = (usage.used / usage.total) * 100

            if free_gb < self.min_free_gb:
                status = HealthStatus.UNHEALTHY
                message = f"Low disk space: {free_gb:.1f}GB free"
            elif free_gb < self.min_free_gb * 2:
                status = HealthStatus.DEGRADED
                message = f"Disk space low: {free_gb:.1f}GB free"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space OK: {free_gb:.1f}GB free"

            return HealthCheckResult(
                name=self.name, status=status, message=message,
                details={"free_bytes": usage.free, "total_bytes": usage.total,
                         "used_percent": round(used_percent, 2), "free_gb": round(free_gb, 2)},
                response_time_ms=(monotonic() - start) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name, status=HealthStatus.UNHEALTHY,
                message=f"Disk check failed: {e}", response_time_ms=(monotonic() - start) * 1000,
            )


@dataclass
class MemoryUsageCheck:
    """Check system memory usage."""
    name: str = "memory"
    max_usage_percent: float = 90.0

    def check(self) -> HealthCheckResult:
        start = monotonic()
        try:
            # Try to get actual memory info on Linux
            try:
                with open("/proc/meminfo") as f:
                    meminfo = f.read()
                values = {}
                for line in meminfo.split("\n")[:3]:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        values[key.strip()] = int(value.strip().split()[0]) * 1024

                total = values.get("MemTotal", 0)
                available = values.get("MemAvailable", values.get("MemFree", 0))
                used = total - available
                used_percent = (used / total * 100) if total > 0 else 0

                if used_percent > self.max_usage_percent:
                    status = HealthStatus.UNHEALTHY
                    message = f"High memory usage: {used_percent:.1f}%"
                elif used_percent > self.max_usage_percent * 0.8:
                    status = HealthStatus.DEGRADED
                    message = f"Memory usage elevated: {used_percent:.1f}%"
                else:
                    status = HealthStatus.HEALTHY
                    message = f"Memory usage OK: {used_percent:.1f}%"

                return HealthCheckResult(
                    name=self.name, status=status, message=message,
                    details={"total_bytes": total, "available_bytes": available, "used_percent": round(used_percent, 2)},
                    response_time_ms=(monotonic() - start) * 1000,
                )
            except FileNotFoundError:
                return HealthCheckResult(
                    name=self.name, status=HealthStatus.HEALTHY,
                    message="Memory check not available on this platform",
                    details={}, response_time_ms=(monotonic() - start) * 1000,
                )
        except Exception as e:
            return HealthCheckResult(
                name=self.name, status=HealthStatus.UNHEALTHY,
                message=f"Memory check failed: {e}", response_time_ms=(monotonic() - start) * 1000,
            )


@dataclass
class HealthChecker:
    """Aggregates multiple health checks."""
    checks: list[HealthCheck] = field(default_factory=list)

    def add_check(self, check: HealthCheck) -> "HealthChecker":
        """Add a health check."""
        self.checks.append(check)
        return self

    def check_all(self) -> dict[str, object]:
        """Run all health checks and return aggregated result."""
        results = [check.check() for check in self.checks]
        statuses = [r.status for r in results]

        if HealthStatus.UNHEALTHY in statuses:
            overall_status = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        return {
            "status": overall_status,
            "checks": [{"name": r.name, "status": r.status, "message": r.message,
                        "details": r.details, "response_time_ms": round(r.response_time_ms, 2)}
                       for r in results],
            "timestamp": monotonic(),
        }

    def is_healthy(self) -> bool:
        """Quick check if all systems are healthy."""
        return self.check_all()["status"] == HealthStatus.HEALTHY


def create_default_health_checker(workspace_root: Path | None = None) -> HealthChecker:
    """Create a health checker with default checks."""
    checker = HealthChecker()
    checker.add_check(DiskSpaceCheck(path=workspace_root or Path.cwd()))
    checker.add_check(MemoryUsageCheck())
    return checker


__all__ = [
    "DiskSpaceCheck", "HealthCheck", "HealthCheckResult", "HealthChecker",
    "HealthStatus", "MemoryUsageCheck", "create_default_health_checker",
]
