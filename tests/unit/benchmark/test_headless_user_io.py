"""Unit tests for HeadlessUserIO."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentlet.benchmark.headless_user_io import HeadlessUserIO
from agentlet.core.interrupts import (
    ApprovalRequest,
    UserQuestionRequest,
    UserQuestionResponse,
)
from agentlet.core.types import InterruptOption
from agentlet.runtime.events import RuntimeEvent


class TestHeadlessUserIO:
    """Tests for HeadlessUserIO benchmark adapter."""

    def test_init_default_values(self):
        """Test default initialization."""
        user_io = HeadlessUserIO()

        assert user_io.auto_approve_all is True
        assert user_io.default_question_response is None
        assert user_io.events == []
        assert user_io.event_log_path is None

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        log_path = Path("/tmp/test_trajectory.jsonl")
        user_io = HeadlessUserIO(
            auto_approve_all=False,
            default_question_response="custom_response",
            event_log_path=log_path,
        )

        assert user_io.auto_approve_all is False
        assert user_io.default_question_response == "custom_response"
        assert user_io.event_log_path == log_path

    def test_request_approval_auto_approve_all(self):
        """Test that all requests are approved when auto_approve_all is True."""
        user_io = HeadlessUserIO(auto_approve_all=True)

        request = ApprovalRequest(
            request_id="req_1",
            tool_name="Write",
            approval_category="mutating",
            prompt="Write to file?",
        )

        response = user_io.request_approval(request)

        assert response.request_id == "req_1"
        assert response.decision == "approved"

    def test_request_approval_read_only_allowed(self):
        """Test that read_only requests are approved even when auto_approve_all is False."""
        user_io = HeadlessUserIO(auto_approve_all=False)

        request = ApprovalRequest(
            request_id="req_1",
            tool_name="Read",
            approval_category="read_only",
            prompt="Read file?",
        )

        response = user_io.request_approval(request)

        assert response.decision == "approved"

    def test_request_approval_mutating_rejected(self):
        """Test that non-read_only requests are rejected when auto_approve_all is False."""
        user_io = HeadlessUserIO(auto_approve_all=False)

        request = ApprovalRequest(
            request_id="req_1",
            tool_name="Write",
            approval_category="mutating",
            prompt="Write to file?",
        )

        response = user_io.request_approval(request)

        assert response.decision == "rejected"

    def test_resolve_question_interrupt_with_default(self):
        """Test auto-selecting first option when no default configured."""
        user_io = HeadlessUserIO()

        request = UserQuestionRequest(
            request_id="q_1",
            prompt="Which file?",
            options=(
                InterruptOption(value="file1", label="File 1"),
                InterruptOption(value="file2", label="File 2"),
            ),
        )

        response = user_io.resolve_question_interrupt(request)

        assert isinstance(response, UserQuestionResponse)
        assert response.request_id == "q_1"
        assert response.selected_option == "file1"

    def test_resolve_question_interrupt_with_override(self):
        """Test using configured default response."""
        user_io = HeadlessUserIO(default_question_response="file2")

        request = UserQuestionRequest(
            request_id="q_1",
            prompt="Which file?",
            options=(
                InterruptOption(value="file1", label="File 1"),
                InterruptOption(value="file2", label="File 2"),
            ),
        )

        response = user_io.resolve_question_interrupt(request)

        assert response.selected_option == "file2"

    def test_resolve_question_interrupt_no_options(self):
        """Test handling when no options are provided."""
        user_io = HeadlessUserIO(default_question_response="free_text_answer")

        request = UserQuestionRequest(
            request_id="q_1",
            prompt="Enter value:",
            options=(),
            allow_free_text=True,
        )

        response = user_io.resolve_question_interrupt(request)

        # When no options, selected_option is used with default_question_response
        assert response.selected_option == "free_text_answer"
        assert response.free_text is None

    def test_emit_event_without_logging(self):
        """Test that events are stored in memory."""
        user_io = HeadlessUserIO()

        request = ApprovalRequest(
            request_id="req_1",
            tool_name="Read",
            approval_category="read_only",
            prompt="Read file?",
        )
        event = RuntimeEvent.approval_requested(request)

        user_io.emit_event(event)

        assert len(user_io.events) == 1
        assert user_io.events[0].kind == "approval_requested"

    def test_emit_event_with_logging(self, tmp_path):
        """Test that events are written to file when path configured."""
        log_path = tmp_path / "trajectory.jsonl"
        user_io = HeadlessUserIO(event_log_path=log_path)

        request = ApprovalRequest(
            request_id="req_1",
            tool_name="Read",
            approval_category="read_only",
            prompt="Read file?",
        )
        event = RuntimeEvent.approval_requested(request)

        user_io.emit_event(event)
        user_io.close()  # Ensure file is flushed

        assert log_path.exists()
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["kind"] == "approval_requested"
        assert data["payload"]["request_id"] == "req_1"

    def test_begin_question_interrupt(self):
        """Test that begin_question_interrupt is a no-op (as designed)."""
        user_io = HeadlessUserIO()

        request = UserQuestionRequest(
            request_id="q_1",
            prompt="Which file?",
            options=(InterruptOption(value="file1", label="File 1"),),
        )

        # Should not raise or modify state
        user_io.begin_question_interrupt(request)

        assert user_io.events == []

    def test_get_trajectory_summary(self):
        """Test trajectory summary generation."""
        user_io = HeadlessUserIO()

        # Add approval event
        approval_req = ApprovalRequest(
            request_id="req_1",
            tool_name="Read",
            approval_category="read_only",
            prompt="Read?",
        )
        user_io.emit_event(RuntimeEvent.approval_requested(approval_req))

        # Add question event
        question_req = UserQuestionRequest(
            request_id="q_1",
            prompt="Which?",
            options=(InterruptOption(value="a", label="A"),),
        )
        user_io.emit_event(RuntimeEvent.question_interrupted(question_req))

        summary = user_io.get_trajectory_summary()

        assert summary["total_events"] == 2
        assert summary["approval_requests"] == 1
        assert summary["question_requests"] == 1
        assert summary["event_kinds"] == ["approval_requested", "question_interrupted"]

    def test_protocol_compliance(self):
        """Test that HeadlessUserIO conforms to UserIO protocol."""
        from agentlet.runtime.user_io import UserIO

        user_io = HeadlessUserIO()

        # This will raise if not compliant
        assert isinstance(user_io, UserIO)

    def test_close_method(self, tmp_path):
        """Test that close() properly closes the log file."""
        log_path = tmp_path / "trajectory.jsonl"
        user_io = HeadlessUserIO(event_log_path=log_path)

        # Write something
        request = ApprovalRequest(
            request_id="req_1",
            tool_name="Read",
            approval_category="read_only",
            prompt="Read?",
        )
        user_io.emit_event(RuntimeEvent.approval_requested(request))

        # Close should not raise
        user_io.close()
        user_io.close()  # Double close should be safe
