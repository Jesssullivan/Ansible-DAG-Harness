"""Tests for observability integration."""

import os
from unittest.mock import MagicMock, patch

import pytest

from harness.config import ObservabilityConfig
from harness.observability import (
    AnonymizingCallback,
    StreamingEventHandler,
    create_tracing_config,
    get_callbacks,
)

# ============================================================================
# OBSERVABILITY CONFIG TESTS
# ============================================================================


class TestObservabilityConfig:
    """Tests for ObservabilityConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ObservabilityConfig()

        assert config.langsmith_enabled is False
        assert config.langsmith_project == "dag-harness"
        assert config.anonymize_sensitive is True

    def test_from_env_disabled(self):
        """Test loading config when tracing is disabled."""
        with patch.dict(os.environ, {}, clear=True):
            config = ObservabilityConfig.from_env()

            assert config.langsmith_enabled is False
            assert config.langsmith_project == "dag-harness"
            assert config.anonymize_sensitive is True

    def test_from_env_enabled(self):
        """Test loading config when tracing is enabled."""
        env = {
            "LANGCHAIN_TRACING_V2": "true",
            "LANGCHAIN_PROJECT": "my-custom-project",
            "HARNESS_ANONYMIZE_SENSITIVE": "true",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ObservabilityConfig.from_env()

            assert config.langsmith_enabled is True
            assert config.langsmith_project == "my-custom-project"
            assert config.anonymize_sensitive is True

    def test_from_env_anonymization_disabled(self):
        """Test disabling anonymization via environment."""
        env = {
            "LANGCHAIN_TRACING_V2": "true",
            "HARNESS_ANONYMIZE_SENSITIVE": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ObservabilityConfig.from_env()

            assert config.langsmith_enabled is True
            assert config.anonymize_sensitive is False

    def test_from_env_case_insensitive(self):
        """Test that boolean parsing is case insensitive."""
        env = {
            "LANGCHAIN_TRACING_V2": "TRUE",
            "HARNESS_ANONYMIZE_SENSITIVE": "FALSE",
        }
        with patch.dict(os.environ, env, clear=True):
            config = ObservabilityConfig.from_env()

            assert config.langsmith_enabled is True
            assert config.anonymize_sensitive is False


# ============================================================================
# ANONYMIZING CALLBACK TESTS
# ============================================================================


class TestAnonymizingCallback:
    """Tests for AnonymizingCallback."""

    def test_anonymize_gitlab_token_env_var(self):
        """Test anonymization of GITLAB_TOKEN environment variable format."""
        callback = AnonymizingCallback()
        text = "GITLAB_TOKEN=glpat-abc123xyz789"
        result = callback._anonymize(text)
        assert "glpat-abc123xyz789" not in result
        assert "[REDACTED]" in result or "REDACTED" in result

    def test_anonymize_gitlab_pat(self):
        """Test anonymization of GitLab personal access tokens."""
        callback = AnonymizingCallback()
        text = "Using token glpat-abcdefghijk123456"
        result = callback._anonymize(text)
        assert "glpat-abcdefghijk123456" not in result
        assert "REDACTED_GITLAB_TOKEN" in result

    def test_anonymize_gitlab_project_token(self):
        """Test anonymization of GitLab project access tokens."""
        callback = AnonymizingCallback()
        text = "Project access glptt-xyz987654321 used"
        result = callback._anonymize(text)
        assert "glptt-xyz987654321" not in result
        assert "REDACTED_GITLAB_TOKEN" in result

    def test_anonymize_password(self):
        """Test anonymization of password fields."""
        callback = AnonymizingCallback()

        # Various password formats
        test_cases = [
            "password=secret123",
            "password: supersecret",
            "passwd=mypass",
            "PASSWORD=ALLCAPS",
        ]

        for text in test_cases:
            result = callback._anonymize(text)
            assert "secret" not in result.lower()
            assert "[REDACTED]" in result or "REDACTED" in result

    def test_anonymize_api_key(self):
        """Test anonymization of API keys."""
        callback = AnonymizingCallback()

        test_cases = [
            "api_key=sk_live_123456789",
            "API_KEY=pk_test_abcdefg",
            "api-key: myapikey123",
        ]

        for text in test_cases:
            result = callback._anonymize(text)
            # The actual key values should be gone
            assert "sk_live" not in result
            assert "pk_test" not in result
            assert "myapikey123" not in result

    def test_anonymize_bearer_token(self):
        """Test anonymization of Bearer tokens."""
        callback = AnonymizingCallback()

        test_cases = [
            "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
            "bearer abc123token",
        ]

        for text in test_cases:
            result = callback._anonymize(text)
            assert "eyJ" not in result
            assert "abc123token" not in result

    def test_anonymize_op_read(self):
        """Test anonymization of 1Password op_read references."""
        callback = AnonymizingCallback()
        text = "{{ op_read('my-secret-entry', 'password') }}"
        result = callback._anonymize(text)
        assert "my-secret-entry" not in result
        assert "REDACTED" in result

    def test_anonymize_ssh_private_key(self):
        """Test anonymization of SSH private keys."""
        callback = AnonymizingCallback()
        text = """
-----BEGIN RSA PRIVATE KEY-----
MIIEowIBAAKCAQEAx/rN...
...actual key content...
-----END RSA PRIVATE KEY-----
        """
        result = callback._anonymize(text)
        assert "MIIEowIBAAKCAQEAx/rN" not in result
        assert "REDACTED_PRIVATE_KEY" in result

    def test_anonymize_aws_access_key(self):
        """Test anonymization of AWS access keys."""
        callback = AnonymizingCallback()
        text = "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE"
        result = callback._anonymize(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_anonymize_langsmith_key(self):
        """Test anonymization of LangSmith API keys."""
        callback = AnonymizingCallback()
        text = "LANGCHAIN_API_KEY=ls__abcdef123456"
        result = callback._anonymize(text)
        assert "ls__abcdef123456" not in result

    def test_anonymize_preserves_non_sensitive(self):
        """Test that non-sensitive text is preserved."""
        callback = AnonymizingCallback()
        text = "This is a normal log message about role deployment"
        result = callback._anonymize(text)
        assert result == text

    def test_anonymize_multiple_patterns(self):
        """Test anonymization with multiple sensitive values."""
        callback = AnonymizingCallback()
        text = "Connecting with GITLAB_TOKEN=glpat-secret123 and password=mysecret to API"
        result = callback._anonymize(text)

        assert "glpat-secret123" not in result
        assert "mysecret" not in result
        assert "Connecting" in result
        assert "API" in result

    def test_anonymize_with_emails_disabled(self):
        """Test that emails are not anonymized by default."""
        callback = AnonymizingCallback(anonymize_emails=False)
        text = "Contact: user@example.com"
        result = callback._anonymize(text)
        assert "user@example.com" in result

    def test_anonymize_with_emails_enabled(self):
        """Test email anonymization when enabled."""
        callback = AnonymizingCallback(anonymize_emails=True)
        text = "Contact: user@example.com"
        result = callback._anonymize(text)
        assert "user@example.com" not in result
        assert "REDACTED_EMAIL" in result

    def test_anonymize_with_ips_disabled(self):
        """Test that IPs are not anonymized by default."""
        callback = AnonymizingCallback(anonymize_ips=False)
        text = "Server at 192.168.1.100"
        result = callback._anonymize(text)
        assert "192.168.1.100" in result

    def test_anonymize_with_ips_enabled(self):
        """Test IP anonymization when enabled."""
        callback = AnonymizingCallback(anonymize_ips=True)
        text = "Server at 192.168.1.100"
        result = callback._anonymize(text)
        assert "192.168.1.100" not in result
        assert "REDACTED_IP" in result

    def test_anonymize_with_extra_patterns(self):
        """Test custom extra patterns."""
        extra = [
            (r"CUSTOM_SECRET_[A-Z0-9]+", "[REDACTED_CUSTOM]"),
        ]
        callback = AnonymizingCallback(extra_patterns=extra)
        text = "Using CUSTOM_SECRET_ABC123"
        result = callback._anonymize(text)
        assert "CUSTOM_SECRET_ABC123" not in result
        assert "REDACTED_CUSTOM" in result

    def test_anonymize_dict(self):
        """Test recursive dictionary anonymization."""
        callback = AnonymizingCallback()
        # Use patterns that match (value=secret format or token prefixes)
        data = {
            "user": "admin",
            "config": "password=secret123",  # Uses password= format
            "nested": {
                "auth": "glpat-abc123",  # GitLab token pattern
            },
            "list_field": ["item1", "token=hidden"],
        }
        result = callback._anonymize_dict(data)

        assert result["user"] == "admin"
        assert "secret123" not in result["config"]
        assert "glpat-abc123" not in result["nested"]["auth"]
        assert "hidden" not in result["list_field"][1]

    def test_anonymize_list(self):
        """Test recursive list anonymization."""
        callback = AnonymizingCallback()
        data = [
            "normal",
            "GITLAB_TOKEN=secret",
            {"key": "password=value"},
        ]
        result = callback._anonymize_list(data)

        assert result[0] == "normal"
        assert "secret" not in result[1]
        assert "value" not in result[2]["key"]

    def test_on_llm_start_modifies_prompts(self):
        """Test that on_llm_start anonymizes prompts."""
        callback = AnonymizingCallback()
        prompts = [
            "Use GITLAB_TOKEN=glpat-secret123 to connect",
            "Normal prompt without secrets",
        ]

        callback.on_llm_start({}, prompts)

        assert "glpat-secret123" not in prompts[0]
        assert "Normal prompt without secrets" == prompts[1]


# ============================================================================
# GET CALLBACKS TESTS
# ============================================================================


class TestGetCallbacks:
    """Tests for get_callbacks function."""

    def test_returns_empty_when_disabled(self):
        """Test no callbacks when LangSmith is disabled."""
        config = ObservabilityConfig(langsmith_enabled=False)
        callbacks = get_callbacks(config)
        assert callbacks == []

    def test_returns_empty_when_anonymization_disabled(self):
        """Test no anonymizing callback when anonymization is disabled."""
        config = ObservabilityConfig(
            langsmith_enabled=True,
            anonymize_sensitive=False,
        )
        callbacks = get_callbacks(config)
        assert callbacks == []

    def test_returns_anonymizing_callback_when_enabled(self):
        """Test returns AnonymizingCallback when both are enabled."""
        config = ObservabilityConfig(
            langsmith_enabled=True,
            anonymize_sensitive=True,
        )
        callbacks = get_callbacks(config)

        assert len(callbacks) == 1
        assert isinstance(callbacks[0], AnonymizingCallback)

    def test_accepts_extra_patterns(self):
        """Test that extra patterns are passed to callback."""
        config = ObservabilityConfig(
            langsmith_enabled=True,
            anonymize_sensitive=True,
        )
        extra = [(r"CUSTOM_[A-Z]+", "[CUSTOM_REDACTED]")]
        callbacks = get_callbacks(config, extra_patterns=extra)

        assert len(callbacks) == 1
        callback = callbacks[0]
        result = callback._anonymize("CUSTOM_SECRET")
        assert "CUSTOM_SECRET" not in result


# ============================================================================
# STREAMING EVENT HANDLER TESTS
# ============================================================================


class TestStreamingEventHandler:
    """Tests for StreamingEventHandler."""

    def test_initial_state(self):
        """Test handler initial state."""
        handler = StreamingEventHandler()

        assert handler.completed_nodes == []
        assert handler.current_node is None
        assert handler.errors == []
        assert handler.events == []

    def test_process_chain_start(self):
        """Test processing on_chain_start event."""
        handler = StreamingEventHandler()
        event = {
            "event": "on_chain_start",
            "name": "validate_role",
            "data": {},
        }

        handler.process(event)

        assert handler.current_node == "validate_role"
        assert len(handler.events) == 1

    def test_process_chain_end(self):
        """Test processing on_chain_end event."""
        handler = StreamingEventHandler()
        start_event = {
            "event": "on_chain_start",
            "name": "validate_role",
            "data": {},
        }
        end_event = {
            "event": "on_chain_end",
            "name": "validate_role",
            "data": {},
        }

        handler.process(start_event)
        handler.process(end_event)

        assert handler.current_node is None
        assert "validate_role" in handler.completed_nodes

    def test_process_chain_error(self):
        """Test processing on_chain_error event."""
        handler = StreamingEventHandler()
        event = {
            "event": "on_chain_error",
            "name": "run_molecule",
            "data": {
                "error": "Test failed",
                "error_type": "RuntimeError",
            },
        }

        handler.process(event)

        assert len(handler.errors) == 1
        assert handler.errors[0]["node"] == "run_molecule"
        assert handler.errors[0]["error"] == "Test failed"

    def test_process_chain_stream(self):
        """Test processing on_chain_stream event."""
        handler = StreamingEventHandler()
        event = {
            "event": "on_chain_stream",
            "name": "workflow",
            "data": {"progress": 50},
        }

        handler.process(event)

        assert len(handler.state_updates) == 1
        assert handler.state_updates[0]["progress"] == 50

    def test_get_summary(self):
        """Test get_summary output."""
        handler = StreamingEventHandler()
        handler.completed_nodes = ["node1", "node2"]
        handler.current_node = "node3"
        handler.errors = [{"node": "node1", "error": "test"}]
        handler.events = [{"event": "test"}] * 5

        summary = handler.get_summary()

        assert summary["completed_nodes"] == ["node1", "node2"]
        assert summary["current_node"] == "node3"
        assert summary["error_count"] == 1
        assert summary["event_count"] == 5

    def test_has_errors(self):
        """Test has_errors method."""
        handler = StreamingEventHandler()
        assert handler.has_errors() is False

        handler.errors.append({"error": "test"})
        assert handler.has_errors() is True

    def test_no_duplicate_completed_nodes(self):
        """Test that completed nodes are not duplicated."""
        handler = StreamingEventHandler()
        event = {
            "event": "on_chain_end",
            "name": "validate_role",
            "data": {},
        }

        handler.process(event)
        handler.process(event)

        assert handler.completed_nodes.count("validate_role") == 1


# ============================================================================
# CREATE TRACING CONFIG TESTS
# ============================================================================


class TestCreateTracingConfig:
    """Tests for create_tracing_config function."""

    def test_basic_config(self):
        """Test basic config creation."""
        obs_config = ObservabilityConfig(
            langsmith_enabled=True,
            anonymize_sensitive=True,
        )
        config = create_tracing_config(obs_config)

        assert "callbacks" in config
        assert len(config["callbacks"]) == 1

    def test_with_run_name(self):
        """Test config with run name."""
        obs_config = ObservabilityConfig(langsmith_enabled=False)
        config = create_tracing_config(obs_config, run_name="test-run")

        assert config["run_name"] == "test-run"

    def test_with_tags(self):
        """Test config with tags."""
        obs_config = ObservabilityConfig(langsmith_enabled=False)
        config = create_tracing_config(obs_config, tags=["tag1", "tag2"])

        assert config["tags"] == ["tag1", "tag2"]

    def test_with_metadata(self):
        """Test config with metadata."""
        obs_config = ObservabilityConfig(langsmith_enabled=False)
        config = create_tracing_config(obs_config, metadata={"role": "common", "wave": 0})

        assert config["metadata"]["role"] == "common"
        assert config["metadata"]["wave"] == 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


@pytest.mark.asyncio
class TestStreamingIntegration:
    """Integration tests for streaming functionality."""

    async def test_execute_with_streaming_basic(self):
        """Test basic streaming execution flow."""
        from harness.observability import execute_with_streaming

        # Create a mock graph that yields events
        mock_graph = MagicMock()
        events = [
            {"event": "on_chain_start", "name": "validate_role", "data": {}},
            {"event": "on_chain_end", "name": "validate_role", "data": {}},
        ]

        async def mock_astream_events(*args, **kwargs):
            for event in events:
                yield event

        mock_graph.astream_events = mock_astream_events

        collected_events = []
        async for event in execute_with_streaming("test_role", mock_graph):
            collected_events.append(event)

        assert len(collected_events) == 2
        assert collected_events[0]["event"] == "on_chain_start"
        assert collected_events[1]["event"] == "on_chain_end"

    async def test_execute_with_streaming_callback(self):
        """Test that callback is invoked for each event."""
        from harness.observability import execute_with_streaming

        mock_graph = MagicMock()
        events = [
            {"event": "on_chain_start", "name": "node1", "data": {}},
        ]

        async def mock_astream_events(*args, **kwargs):
            for event in events:
                yield event

        mock_graph.astream_events = mock_astream_events

        callback_calls = []

        async def test_callback(event):
            callback_calls.append(event)

        async for _ in execute_with_streaming("test_role", mock_graph, callback=test_callback):
            pass

        assert len(callback_calls) == 1

    async def test_execute_with_streaming_error_handling(self):
        """Test error handling in streaming execution."""
        from harness.observability import execute_with_streaming

        mock_graph = MagicMock()

        async def mock_astream_events(*args, **kwargs):
            yield {"event": "on_chain_start", "name": "node1", "data": {}}
            raise RuntimeError("Test error")

        mock_graph.astream_events = mock_astream_events

        events = []
        with pytest.raises(RuntimeError):
            async for event in execute_with_streaming("test_role", mock_graph):
                events.append(event)

        # Should have received at least one event before error
        assert len(events) >= 1
        # Last event should be error event
        assert events[-1]["event"] == "on_chain_error"
        assert "Test error" in events[-1]["data"]["error"]
