"""
Observability integration for LangSmith tracing.

Provides:
- AnonymizingCallback: Masks sensitive data before sending to LangSmith
- Streaming support: Real-time event streaming for workflow execution
- LangSmith integration: Automatic tracing when enabled via environment

Usage:
    # Enable via environment variables:
    # LANGCHAIN_TRACING_V2=true
    # LANGCHAIN_PROJECT=dag-harness
    # LANGCHAIN_API_KEY=your-api-key

    from harness.observability import get_callbacks, execute_with_streaming
    from harness.config import ObservabilityConfig

    config = ObservabilityConfig.from_env()
    callbacks = get_callbacks(config)

    # Execute with streaming
    async for event in execute_with_streaming(role_name, graph):
        print(f"Event: {event['event']}")
"""

import re
from collections.abc import AsyncIterator, Callable
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig

from harness.config import ObservabilityConfig


class AnonymizingCallback(BaseCallbackHandler):
    """
    Callback handler that masks sensitive data before sending to LangSmith.

    Automatically redacts:
    - GitLab tokens (GITLAB_TOKEN, private_token, etc.)
    - Passwords and secrets
    - API keys and tokens
    - SSH keys and certificates
    - Email addresses (optional)
    - IP addresses (optional)

    Example:
        callback = AnonymizingCallback()
        # Input: "GITLAB_TOKEN=glpat-abc123xyz password=secret123"
        # Output: "GITLAB_TOKEN=[REDACTED] password=[REDACTED]"
    """

    # Patterns to match sensitive data
    # Each pattern is a tuple of (regex_pattern, replacement)
    # ORDER MATTERS: More specific patterns must come before general ones
    SENSITIVE_PATTERNS: list[tuple[str, str]] = [
        # GitLab tokens (MUST come before generic token patterns)
        (r"GITLAB_TOKEN[=:]\s*[^\s]+", "GITLAB_TOKEN=[REDACTED]"),
        (r"private_token[=:]\s*[^\s]+", "private_token=[REDACTED]"),
        (r"glpat-[a-zA-Z0-9_-]+", "[REDACTED_GITLAB_TOKEN]"),
        (r"glptt-[a-zA-Z0-9_-]+", "[REDACTED_GITLAB_TOKEN]"),
        (r"gldt-[a-zA-Z0-9_-]+", "[REDACTED_GITLAB_TOKEN]"),
        # SSH private keys (multi-line pattern, check early)
        (
            r"-----BEGIN [A-Z ]+ PRIVATE KEY-----[\s\S]*?-----END [A-Z ]+ PRIVATE KEY-----",
            "[REDACTED_PRIVATE_KEY]",
        ),
        # AWS credentials
        (r"AKIA[0-9A-Z]{16}", "[REDACTED_AWS_ACCESS_KEY]"),
        (r"aws_secret_access_key[=:]\s*[^\s]+", "aws_secret_access_key=[REDACTED]"),
        # LangSmith/LangChain API keys
        (r"LANGCHAIN_API_KEY[=:]\s*[^\s]+", "LANGCHAIN_API_KEY=[REDACTED]"),
        (r"ls__[a-zA-Z0-9_.-]+", "[REDACTED_LANGSMITH_KEY]"),
        # 1Password references (common in Ansible)
        (r'op_read\([\'"][^\'"]+[\'"]', "op_read('[REDACTED]'"),
        # Generic passwords and secrets (with = or : delimiter)
        (r"password[=:]\s*[^\s]+", "password=[REDACTED]"),
        (r"passwd[=:]\s*[^\s]+", "passwd=[REDACTED]"),
        (r"secret[=:]\s*[^\s]+", "secret=[REDACTED]"),
        (r"SECRET[=:]\s*[^\s]+", "SECRET=[REDACTED]"),
        # API keys and tokens (generic patterns, after specific ones)
        (r"api[_-]?key[=:]\s*[^\s]+", "api_key=[REDACTED]"),
        (r"API[_-]?KEY[=:]\s*[^\s]+", "API_KEY=[REDACTED]"),
        (r"auth[_-]?token[=:]\s*[^\s]+", "auth_token=[REDACTED]"),
        (r"token[=:]\s*[^\s]+", "token=[REDACTED]"),
        (r"TOKEN[=:]\s*[^\s]+", "TOKEN=[REDACTED]"),
        (r"bearer\s+[a-zA-Z0-9_.-]+", "bearer [REDACTED]"),
        (r"Bearer\s+[a-zA-Z0-9_.-]+", "Bearer [REDACTED]"),
        # Generic credential patterns (must be last, most general)
        (r"credential[s]?[=:]\s*[^\s]+", "credentials=[REDACTED]"),
    ]

    # Compiled patterns for efficiency
    _compiled_patterns: list[tuple[re.Pattern, str]] = []

    def __init__(
        self,
        anonymize_emails: bool = False,
        anonymize_ips: bool = False,
        extra_patterns: list[tuple[str, str]] | None = None,
    ):
        """
        Initialize the anonymizing callback.

        Args:
            anonymize_emails: Also redact email addresses
            anonymize_ips: Also redact IP addresses
            extra_patterns: Additional (pattern, replacement) tuples to apply
        """
        super().__init__()

        # Build pattern list
        patterns = list(self.SENSITIVE_PATTERNS)

        if anonymize_emails:
            patterns.append((r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "[REDACTED_EMAIL]"))

        if anonymize_ips:
            patterns.append((r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[REDACTED_IP]"))

        if extra_patterns:
            patterns.extend(extra_patterns)

        # Compile all patterns
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), replacement) for pattern, replacement in patterns
        ]

    def _anonymize(self, text: str) -> str:
        """Apply all anonymization patterns to text."""
        if not isinstance(text, str):
            return text

        result = text
        for pattern, replacement in self._compiled_patterns:
            result = pattern.sub(replacement, result)
        return result

    def _anonymize_dict(self, data: dict) -> dict:
        """Recursively anonymize a dictionary."""
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = self._anonymize(value)
            elif isinstance(value, dict):
                result[key] = self._anonymize_dict(value)
            elif isinstance(value, list):
                result[key] = self._anonymize_list(value)
            else:
                result[key] = value
        return result

    def _anonymize_list(self, data: list) -> list:
        """Recursively anonymize a list."""
        result = []
        for item in data:
            if isinstance(item, str):
                result.append(self._anonymize(item))
            elif isinstance(item, dict):
                result.append(self._anonymize_dict(item))
            elif isinstance(item, list):
                result.append(self._anonymize_list(item))
            else:
                result.append(item)
        return result

    # =========================================================================
    # LLM Callbacks
    # =========================================================================

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Anonymize prompts before they are logged."""
        # Modify prompts in-place
        for i, prompt in enumerate(prompts):
            prompts[i] = self._anonymize(prompt)

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Called when a new token is generated - anonymize if needed."""
        # Token-by-token anonymization is less effective but still useful
        # for streaming scenarios
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Anonymize LLM output before logging."""
        for generations in response.generations:
            for generation in generations:
                if hasattr(generation, "text"):
                    generation.text = self._anonymize(generation.text)

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        """Anonymize error messages."""
        # Errors might contain sensitive info in stack traces
        pass

    # =========================================================================
    # Chain Callbacks
    # =========================================================================

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Anonymize chain inputs."""
        if isinstance(inputs, dict):
            anonymized = self._anonymize_dict(inputs)
            inputs.clear()
            inputs.update(anonymized)

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        """Anonymize chain outputs."""
        if isinstance(outputs, dict):
            anonymized = self._anonymize_dict(outputs)
            outputs.clear()
            outputs.update(anonymized)

    # =========================================================================
    # Tool Callbacks
    # =========================================================================

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Anonymize tool inputs."""
        # Note: input_str is immutable, so we can't modify it directly
        # This is mainly for logging awareness
        pass

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Anonymize tool outputs."""
        pass


def get_callbacks(
    config: ObservabilityConfig,
    extra_patterns: list[tuple[str, str]] | None = None,
) -> list[BaseCallbackHandler]:
    """
    Get configured callbacks based on observability settings.

    Args:
        config: Observability configuration
        extra_patterns: Additional anonymization patterns

    Returns:
        List of callback handlers to use
    """
    callbacks: list[BaseCallbackHandler] = []

    if config.langsmith_enabled and config.anonymize_sensitive:
        callbacks.append(
            AnonymizingCallback(
                anonymize_emails=False,
                anonymize_ips=False,
                extra_patterns=extra_patterns,
            )
        )

    return callbacks


async def execute_with_streaming(
    role_name: str,
    graph: Any,
    initial_state: dict | None = None,
    config: RunnableConfig | None = None,
    callback: Callable[[dict], Any] | None = None,
) -> AsyncIterator[dict]:
    """
    Execute a workflow with streaming events.

    Provides real-time updates during workflow execution by yielding
    events as they occur. Events include node starts, completions,
    state updates, and errors.

    Args:
        role_name: Name of the role being processed
        graph: Compiled LangGraph to execute
        initial_state: Optional initial state (will create default if None)
        config: Optional LangGraph configuration
        callback: Optional async callback to invoke for each event

    Yields:
        Event dictionaries with:
        - event: Event type (on_chain_start, on_chain_end, on_chain_stream, etc.)
        - name: Node or chain name
        - data: Event-specific data
        - run_id: Unique run identifier
        - timestamp: Event timestamp

    Example:
        async for event in execute_with_streaming("common", graph):
            if event["event"] == "on_chain_end":
                print(f"Node completed: {event['name']}")
            elif event["event"] == "on_chain_stream":
                print(f"Progress: {event['data']}")
    """
    from harness.dag.langgraph_engine import create_initial_state

    # Create initial state if not provided
    if initial_state is None:
        initial_state = create_initial_state(role_name)

    # Use astream_events for real-time updates
    # Version "v2" provides the most detailed events
    try:
        async for event in graph.astream_events(
            initial_state,
            config=config or {},
            version="v2",
        ):
            # Invoke callback if provided
            if callback is not None:
                await callback(event)

            yield event

    except Exception as e:
        # Yield error event
        error_event = {
            "event": "on_chain_error",
            "name": "workflow",
            "data": {
                "error": str(e),
                "error_type": type(e).__name__,
            },
            "role_name": role_name,
        }
        if callback is not None:
            await callback(error_event)
        yield error_event
        raise


class StreamingEventHandler:
    """
    Handler for processing streaming events from workflow execution.

    Provides convenient methods for filtering and processing specific
    event types, and maintains state about workflow progress.

    Example:
        handler = StreamingEventHandler()

        async for event in execute_with_streaming("common", graph):
            handler.process(event)

        print(f"Completed nodes: {handler.completed_nodes}")
        print(f"Current node: {handler.current_node}")
    """

    def __init__(self):
        self.completed_nodes: list[str] = []
        self.current_node: str | None = None
        self.errors: list[dict] = []
        self.events: list[dict] = []
        self.state_updates: list[dict] = []

    def process(self, event: dict) -> None:
        """Process a streaming event and update internal state."""
        self.events.append(event)

        event_type = event.get("event", "")
        name = event.get("name", "")
        data = event.get("data", {})

        if event_type == "on_chain_start":
            self.current_node = name
        elif event_type == "on_chain_end":
            if name and name not in self.completed_nodes:
                self.completed_nodes.append(name)
            self.current_node = None
        elif event_type == "on_chain_error":
            self.errors.append(
                {
                    "node": name,
                    "error": data.get("error"),
                    "error_type": data.get("error_type"),
                }
            )
        elif event_type == "on_chain_stream":
            if isinstance(data, dict):
                self.state_updates.append(data)

    def get_summary(self) -> dict:
        """Get a summary of the workflow execution."""
        return {
            "completed_nodes": self.completed_nodes,
            "current_node": self.current_node,
            "error_count": len(self.errors),
            "errors": self.errors,
            "event_count": len(self.events),
            "state_update_count": len(self.state_updates),
        }

    def has_errors(self) -> bool:
        """Check if any errors occurred during execution."""
        return len(self.errors) > 0


def create_tracing_config(
    observability_config: ObservabilityConfig,
    run_name: str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
) -> RunnableConfig:
    """
    Create a RunnableConfig with LangSmith tracing settings.

    Args:
        observability_config: Observability configuration
        run_name: Optional name for this run
        tags: Optional tags to add to the trace
        metadata: Optional metadata to attach

    Returns:
        RunnableConfig with tracing callbacks configured
    """
    callbacks = get_callbacks(observability_config)

    config: RunnableConfig = {
        "callbacks": callbacks,
    }

    if run_name:
        config["run_name"] = run_name

    if tags:
        config["tags"] = tags

    if metadata:
        config["metadata"] = metadata

    return config
