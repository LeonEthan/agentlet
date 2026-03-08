"""Tests for health check system."""

from __future__ import annotations

from pathlib import Path

import pytest

from agentlet.core.health import (
    DiskSpaceCheck,
    HealthCheckResult,
    HealthChecker,
    HealthStatus,
    MemoryUsageCheck,
    create_default_health_checker,
)


def test_disk_space_check_returns_result():
    """Test disk space check returns valid result."""
    check = DiskSpaceCheck(path=Path.cwd())

    result = check.check()

    assert isinstance(result, HealthCheckResult)
    assert result.name == "disk_space"
    assert result.status in {HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY}
    assert "free" in result.message.lower()
    assert "free_bytes" in result.details


def test_disk_space_check_low_space():
    """Test disk space check with very low threshold."""
    check = DiskSpaceCheck(path=Path.cwd(), min_free_gb=10000.0)  # Impossibly high

    result = check.check()

    # Should be unhealthy or degraded
    assert result.status in {HealthStatus.UNHEALTHY, HealthStatus.DEGRADED}


def test_memory_check_returns_result():
    """Test memory check returns valid result."""
    check = MemoryUsageCheck()

    result = check.check()

    assert isinstance(result, HealthCheckResult)
    assert result.name == "memory"
    assert result.status == HealthStatus.HEALTHY


def test_health_checker_empty():
    """Test health checker with no checks."""
    checker = HealthChecker()

    result = checker.check_all()

    assert result["status"] == HealthStatus.HEALTHY
    assert result["checks"] == []


def test_health_checker_add_check():
    """Test adding checks to health checker."""
    checker = HealthChecker()
    check = DiskSpaceCheck()

    checker.add_check(check)

    assert len(checker.checks) == 1


def test_health_checker_multiple_checks():
    """Test health checker with multiple checks."""
    checker = HealthChecker()
    checker.add_check(DiskSpaceCheck())
    checker.add_check(MemoryUsageCheck())

    result = checker.check_all()

    assert result["status"] in {HealthStatus.HEALTHY, HealthStatus.DEGRADED}
    assert len(result["checks"]) == 2
    assert "timestamp" in result


def test_health_checker_is_healthy():
    """Test is_healthy convenience method."""
    checker = HealthChecker()
    checker.add_check(DiskSpaceCheck())

    is_healthy = checker.is_healthy()

    assert isinstance(is_healthy, bool)


def test_create_default_health_checker():
    """Test factory function for default health checker."""
    checker = create_default_health_checker()

    assert len(checker.checks) >= 2
    assert checker.is_healthy()  # Should be healthy in test environment


def test_health_check_result_details():
    """Test health check result includes details."""
    check = DiskSpaceCheck()

    result = check.check()

    assert "free_gb" in result.details
    assert "used_percent" in result.details
    assert result.response_time_ms >= 0


def test_health_status_constants():
    """Test health status constants."""
    assert HealthStatus.HEALTHY == "healthy"
    assert HealthStatus.DEGRADED == "degraded"
    assert HealthStatus.UNHEALTHY == "unhealthy"
