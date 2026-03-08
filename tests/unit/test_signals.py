"""Tests for signal handling and graceful shutdown."""

from __future__ import annotations

import signal
import threading
import time

import pytest

from agentlet.runtime.signals import ShutdownManager, graceful_shutdown_context


def test_shutdown_manager_initial_state():
    """Test initial state of shutdown manager."""
    manager = ShutdownManager()

    assert manager.should_exit is False
    assert manager.exit_code == 0


def test_shutdown_manager_request_shutdown():
    """Test requesting shutdown."""
    manager = ShutdownManager()

    manager.request_shutdown(exit_code=1)

    assert manager.should_exit is True
    assert manager.exit_code == 1


def test_shutdown_manager_runs_cleanup_handlers():
    """Test that cleanup handlers are run on shutdown."""
    manager = ShutdownManager()
    cleanup_called = []

    @manager.on_shutdown
    def cleanup():
        cleanup_called.append(True)

    manager.request_shutdown()

    assert len(cleanup_called) == 1


def test_shutdown_manager_multiple_handlers():
    """Test multiple cleanup handlers."""
    manager = ShutdownManager()
    calls = []

    @manager.on_shutdown
    def cleanup1():
        calls.append(1)

    @manager.on_shutdown
    def cleanup2():
        calls.append(2)

    manager.request_shutdown()

    assert calls == [1, 2]


def test_shutdown_manager_handler_exception_handling():
    """Test that handler exceptions don't stop other handlers."""
    manager = ShutdownManager()
    calls = []

    @manager.on_shutdown
    def failing_cleanup():
        raise ValueError("cleanup failed")

    @manager.on_shutdown
    def good_cleanup():
        calls.append("good")

    # Should not raise
    manager.request_shutdown()

    assert calls == ["good"]


def test_graceful_shutdown_context():
    """Test graceful shutdown context manager."""
    with graceful_shutdown_context() as manager:
        assert isinstance(manager, ShutdownManager)
        assert manager.should_exit is False


def test_shutdown_manager_is_thread_safe():
    """Test that shutdown manager is thread-safe."""
    manager = ShutdownManager()
    results = []

    def check_shutdown():
        results.append(manager.should_exit)

    threads = [threading.Thread(target=check_shutdown) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r is False for r in results)


def test_shutdown_manager_signal_simulation():
    """Test signal handler behavior."""
    manager = ShutdownManager()

    # Simulate signal handler being called
    manager._signal_handler(signal.SIGINT, None)

    assert manager.should_exit is True
    assert manager.exit_code == 0


def test_shutdown_manager_sigterm_exit_code():
    """Test that SIGTERM uses non-zero exit code."""
    manager = ShutdownManager()

    manager._signal_handler(signal.SIGTERM, None)

    assert manager.should_exit is True
    assert manager.exit_code == 1


def test_shutdown_manager_decorator_returns_handler():
    """Test that on_shutdown decorator returns the handler."""
    manager = ShutdownManager()

    @manager.on_shutdown
    def my_handler():
        pass

    assert my_handler is not None
    # Should be able to call the handler directly
    my_handler()
