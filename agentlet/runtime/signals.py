"""Signal handling for graceful shutdown.

Ensures in-flight operations complete and state is persisted on SIGINT/SIGTERM.
"""

from __future__ import annotations

import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

from agentlet.core.types import get_logger

logger = get_logger("agentlet.signals")


@dataclass
class ShutdownManager:
    """Manages graceful shutdown coordination.

    Usage:
        manager = ShutdownManager()
        with manager.install_handlers():
            # Main application logic
            while not manager.should_exit:
                do_work()
    """

    _should_exit: threading.Event = field(default_factory=threading.Event)
    _exit_code: int = 0
    _handlers: list[Callable[[], None]] = field(default_factory=list)

    @property
    def should_exit(self) -> bool:
        """Check if shutdown has been requested."""
        return self._should_exit.is_set()

    @property
    def exit_code(self) -> int:
        """Get the exit code for the shutdown."""
        return self._exit_code

    def request_shutdown(self, exit_code: int = 0) -> None:
        """Request graceful shutdown.

        Args:
            exit_code: Exit code to return (0 for clean, non-zero for error).
        """
        self._exit_code = exit_code
        self._should_exit.set()
        logger.info("Shutdown requested", exit_code=exit_code)

        # Run registered cleanup handlers
        for handler in self._handlers:
            try:
                handler()
            except Exception as e:
                logger.exception("Cleanup handler failed", exc=e)

    def on_shutdown(self, handler: Callable[[], None]) -> Callable[[], None]:
        """Register a handler to run on shutdown.

        Args:
            handler: Callable to run during shutdown.

        Returns:
            The handler (for use as decorator).
        """
        self._handlers.append(handler)
        return handler

    def _signal_handler(self, signum: int, frame) -> None:
        """Internal signal handler."""
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, initiating graceful shutdown")
        self.request_shutdown(exit_code=0 if signum == signal.SIGINT else 1)

    @contextmanager
    def install_handlers(self):
        """Context manager that installs signal handlers.

        Yields:
            self for use in the context.
        """
        # Save original handlers
        original_int = signal.signal(signal.SIGINT, self._signal_handler)
        original_term = signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            logger.info("Signal handlers installed")
            yield self
        finally:
            # Restore original handlers
            signal.signal(signal.SIGINT, original_int)
            signal.signal(signal.SIGTERM, original_term)
            logger.info("Signal handlers restored")


@contextmanager
def graceful_shutdown_context(exit_timeout: float = 30.0):
    """Context manager for graceful shutdown handling.

    Automatically handles SIGINT/SIGTERM and ensures cleanup.

    Args:
        exit_timeout: Maximum time to wait for cleanup (not yet enforced).

    Yields:
        ShutdownManager instance.
    """
    manager = ShutdownManager()
    with manager.install_handlers():
        yield manager


__all__ = [
    "ShutdownManager",
    "graceful_shutdown_context",
]
