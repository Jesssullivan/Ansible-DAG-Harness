"""Context engineering for HOTL mode - Manus-inspired patterns.

This module implements context management to prevent context bloat in long-running
HOTL sessions. Key patterns from Manus:
- Proactive context compaction at a threshold (e.g., 25% of limit)
- Always preserve error stack traces
- Keep recent changes in full detail, summarize older ones
- Inject diversity to prevent pattern lock-in
- File-based memory for large observations
"""

import hashlib
import logging
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from harness.hotl.state import HOTLState

logger = logging.getLogger(__name__)


@dataclass
class ContextConfig:
    """Configuration for context management.

    Attributes:
        compaction_threshold: Fraction of max_context_tokens at which to compact (0.0-1.0)
        max_context_tokens: Maximum tokens Claude can handle
        keep_recent_changes: Number of recent file changes to keep in full detail
        keep_recent_progress: Number of recent progress updates to keep
        preserve_errors: Whether to always preserve error stack traces
        diversity_enabled: Whether to apply prompt diversity injection
        memory_dir: Directory for file-based memory storage
    """

    compaction_threshold: float = 0.25
    max_context_tokens: int = 200000
    keep_recent_changes: int = 5
    keep_recent_progress: int = 10
    preserve_errors: bool = True
    diversity_enabled: bool = True
    memory_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if not 0.0 < self.compaction_threshold < 1.0:
            raise ValueError("compaction_threshold must be between 0.0 and 1.0")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if self.keep_recent_changes < 0:
            raise ValueError("keep_recent_changes must be non-negative")
        if self.keep_recent_progress < 0:
            raise ValueError("keep_recent_progress must be non-negative")


@dataclass
class ContextStats:
    """Statistics about context usage and compaction.

    Attributes:
        estimated_tokens: Current estimated token count
        compaction_count: Number of times context was compacted
        last_compaction_time: Timestamp of last compaction
        items_summarized: Number of items summarized in last compaction
        memory_files_count: Number of files stored in file memory
    """

    estimated_tokens: int = 0
    compaction_count: int = 0
    last_compaction_time: float | None = None
    items_summarized: int = 0
    memory_files_count: int = 0


class ContextManager:
    """Manus-inspired context engineering for HOTL.

    Manages context to prevent bloat during long-running HOTL sessions.
    Key strategies:
    - Estimates token usage and compacts proactively
    - Summarizes old observations while keeping recent ones detailed
    - Always preserves error stack traces (Manus pattern)
    - Injects diversity to prevent pattern lock-in
    - Stores large content in files, keeps references in context
    """

    # Prompt diversity templates for task injection
    TASK_TEMPLATES = [
        "Consider the following task:",
        "Your objective is:",
        "Please work on:",
        "Task for completion:",
        "Assignment:",
        "Focus on this goal:",
        "Your next task:",
        "Work on the following:",
        "Address this requirement:",
        "Complete this task:",
    ]

    # Common error patterns to always preserve
    ERROR_PATTERNS = [
        r"(?:Error|ERROR|Exception|EXCEPTION)",
        r"Traceback \(most recent call last\)",
        r"(?:Failed|FAILED|Failure|FAILURE)",
        r"(?:raise|Raise)\s+\w+Error",
        r"^\s*File \".*\", line \d+",
        r"AssertionError",
        r"AttributeError",
        r"ImportError",
        r"KeyError",
        r"TypeError",
        r"ValueError",
        r"RuntimeError",
    ]

    def __init__(self, config: ContextConfig | None = None):
        """Initialize the context manager.

        Args:
            config: Context configuration. Uses defaults if not provided.
        """
        self.config = config or ContextConfig()
        self._stats = ContextStats()
        self._used_templates: set[str] = set()
        self._error_regex = re.compile("|".join(self.ERROR_PATTERNS), re.MULTILINE | re.IGNORECASE)

    @property
    def stats(self) -> ContextStats:
        """Get current context statistics."""
        return self._stats

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a given text.

        Uses a simple heuristic of ~4 characters per token, which is
        a reasonable approximation for English text with code.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        # ~4 chars per token is a reasonable estimate for mixed text/code
        return len(text) // 4

    def estimate_state_tokens(self, state: HOTLState) -> int:
        """Estimate total token count for a HOTL state.

        Args:
            state: HOTL state to estimate.

        Returns:
            Estimated token count for the entire state.
        """
        total = 0

        # Estimate each field
        for key, value in state.items():
            if value is None:
                continue
            if isinstance(value, str):
                total += self.estimate_tokens(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        total += self.estimate_tokens(item)
                    elif isinstance(item, dict):
                        total += self.estimate_tokens(str(item))
            elif isinstance(value, dict):
                total += self.estimate_tokens(str(value))

        self._stats.estimated_tokens = total
        return total

    def should_compact(self, token_count: int | None = None) -> bool:
        """Check if context should be compacted.

        Args:
            token_count: Token count to check. Uses stats if not provided.

        Returns:
            True if context exceeds compaction threshold.
        """
        if token_count is None:
            token_count = self._stats.estimated_tokens

        threshold = int(self.config.max_context_tokens * self.config.compaction_threshold)
        return token_count > threshold

    def _contains_error(self, text: str) -> bool:
        """Check if text contains error patterns that should be preserved.

        Args:
            text: Text to check for error patterns.

        Returns:
            True if text contains error patterns.
        """
        if not text:
            return False
        return bool(self._error_regex.search(text))

    def _summarize_text(self, text: str, max_chars: int = 200) -> str:
        """Summarize a text to a shorter version.

        Args:
            text: Text to summarize.
            max_chars: Maximum characters for the summary.

        Returns:
            Summarized text.
        """
        if not text or len(text) <= max_chars:
            return text

        # Keep the beginning (usually most important) and add indicator
        return text[:max_chars] + "... [truncated]"

    def _summarize_dict(self, data: dict[str, Any], max_chars: int = 300) -> dict[str, Any]:
        """Summarize a dictionary by truncating large values.

        Args:
            data: Dictionary to summarize.
            max_chars: Maximum characters for string values.

        Returns:
            Summarized dictionary.
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str) and len(value) > max_chars:
                result[key] = self._summarize_text(value, max_chars)
            elif isinstance(value, list) and len(value) > 5:
                result[key] = value[:3] + [f"... and {len(value) - 3} more items"]
            elif isinstance(value, dict):
                result[key] = self._summarize_dict(value, max_chars)
            else:
                result[key] = value
        return result

    def compact_context(self, state: HOTLState) -> HOTLState:
        """Compact context by summarizing old observations.

        Strategy:
        - Keep recent file changes in full detail
        - Summarize older progress updates
        - ALWAYS preserve error stack traces (Manus pattern)
        - Compress research findings and insights

        Args:
            state: Current HOTL state to compact.

        Returns:
            Compacted HOTL state.
        """
        logger.info("Compacting HOTL context...")
        items_summarized = 0

        # Create a copy of the state to modify
        compacted: dict[str, Any] = dict(state)

        # Compact research_findings - keep structure but summarize content
        if "research_findings" in compacted and compacted["research_findings"]:
            findings = compacted["research_findings"]
            if len(findings) > 5:
                # Keep last 5 in detail, summarize earlier ones
                summarized_findings = []
                for i, finding in enumerate(findings):
                    if i < len(findings) - 5:
                        # Older findings - summarize
                        if isinstance(finding, dict):
                            summarized = self._summarize_dict(finding, 150)
                            summarized["_compacted"] = True
                            summarized_findings.append(summarized)
                        else:
                            summarized_findings.append(self._summarize_text(str(finding), 100))
                        items_summarized += 1
                    else:
                        # Recent findings - keep full
                        summarized_findings.append(finding)
                compacted["research_findings"] = summarized_findings

        # Compact codebase_insights
        if "codebase_insights" in compacted and compacted["codebase_insights"]:
            insights = compacted["codebase_insights"]
            if len(insights) > self.config.keep_recent_progress:
                # Merge older insights into summary
                older_insights = insights[: -self.config.keep_recent_progress]
                recent_insights = insights[-self.config.keep_recent_progress :]

                # Check for errors in older insights - preserve them
                preserved = []
                summarized = []
                for insight in older_insights:
                    if self.config.preserve_errors and self._contains_error(insight):
                        preserved.append(insight)
                    else:
                        summarized.append(insight)
                        items_summarized += 1

                # Create summary of non-error insights
                summary_items = []
                if summarized:
                    summary_items.append(f"[{len(summarized)} earlier insights summarized]")

                compacted["codebase_insights"] = summary_items + preserved + list(recent_insights)

        # Compact plan_gaps - keep recent, summarize old
        if "plan_gaps" in compacted and compacted["plan_gaps"]:
            gaps = compacted["plan_gaps"]
            if len(gaps) > 10:
                older = gaps[:-10]
                recent = gaps[-10:]
                compacted["plan_gaps"] = [f"[{len(older)} earlier gaps addressed]"] + list(recent)
                items_summarized += len(older)

        # Compact completed_tasks and failed_tasks - just keep counts for old ones
        for task_key in ["completed_tasks", "failed_tasks"]:
            if task_key in compacted and len(compacted.get(task_key, [])) > 20:
                tasks = compacted[task_key]
                older = tasks[:-10]
                recent = tasks[-10:]
                compacted[task_key] = list(recent)  # Keep only recent
                # Add a count note to errors if failed
                if task_key == "failed_tasks" and older:
                    if "errors" not in compacted:
                        compacted["errors"] = []
                    compacted["errors"] = [f"[{len(older)} earlier failed tasks]"] + list(
                        compacted.get("errors", [])
                    )
                items_summarized += len(older)

        # NEVER compact errors if preserve_errors is True
        if self.config.preserve_errors and "errors" in compacted:
            # Errors are preserved as-is
            pass

        # Compact warnings - less critical, can be more aggressive
        if "warnings" in compacted and compacted["warnings"]:
            warnings = compacted["warnings"]
            if len(warnings) > 5:
                older = warnings[:-5]
                recent = warnings[-5:]
                compacted["warnings"] = [f"[{len(older)} earlier warnings]"] + list(recent)
                items_summarized += len(older)

        # Compact completed_agent_sessions - keep only IDs, not full data
        if (
            "completed_agent_sessions" in compacted
            and len(compacted.get("completed_agent_sessions", [])) > 10
        ):
            sessions = compacted["completed_agent_sessions"]
            compacted["completed_agent_sessions"] = list(sessions[-10:])
            items_summarized += len(sessions) - 10

        # Compact web_search_results
        if "web_search_results" in compacted and compacted["web_search_results"]:
            results = compacted["web_search_results"]
            if len(results) > 5:
                compacted["web_search_results"] = [
                    self._summarize_dict(r, 100)
                    if isinstance(r, dict)
                    else self._summarize_text(str(r), 100)
                    for r in results[-5:]
                ]
                items_summarized += len(results) - 5

        # Update stats
        self._stats.compaction_count += 1
        self._stats.last_compaction_time = time.time()
        self._stats.items_summarized = items_summarized

        # Re-estimate tokens after compaction
        new_tokens = self.estimate_state_tokens(HOTLState(**compacted))
        logger.info(
            f"Context compacted: {items_summarized} items summarized, "
            f"estimated tokens: {new_tokens}"
        )

        return HOTLState(**compacted)

    def inject_diversity(self, prompt: str) -> str:
        """Add structured noise to prevent pattern lock-in.

        Uses rotating templates to add variety to prompts, which helps
        prevent the model from falling into repetitive patterns.

        Args:
            prompt: Original prompt text.

        Returns:
            Prompt with diversity prefix added.
        """
        if not self.config.diversity_enabled:
            return prompt

        # Try to use an unused template first
        available = set(self.TASK_TEMPLATES) - self._used_templates
        if not available:
            # All templates used, reset and pick randomly
            self._used_templates.clear()
            available = set(self.TASK_TEMPLATES)

        template = random.choice(list(available))
        self._used_templates.add(template)

        return f"{template} {prompt}"

    def reset_diversity(self) -> None:
        """Reset diversity tracking to allow template reuse."""
        self._used_templates.clear()


class FileMemory:
    """Store large observations as files, keep references in context.

    This helps manage context by offloading large content to disk while
    keeping compact references in the active context.
    """

    def __init__(self, memory_dir: Path):
        """Initialize file-based memory.

        Args:
            memory_dir: Directory to store memory files.
        """
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, Path] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load existing memory files into index."""
        for path in self.memory_dir.glob("*.txt"):
            key = path.stem
            self._index[key] = path

    def _generate_key(self, content: str, prefix: str = "mem") -> str:
        """Generate a unique key for content.

        Args:
            content: Content to generate key for.
            prefix: Prefix for the key.

        Returns:
            Unique key string.
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}_{content_hash}"

    def store(self, content: str, key: str | None = None, prefix: str = "mem") -> str:
        """Store content to file and return reference.

        Args:
            content: Content to store.
            key: Optional specific key. Auto-generated if not provided.
            prefix: Prefix for auto-generated keys.

        Returns:
            Reference string for retrieval.
        """
        if key is None:
            key = self._generate_key(content, prefix)

        path = self.memory_dir / f"{key}.txt"
        path.write_text(content, encoding="utf-8")
        self._index[key] = path

        logger.debug(f"Stored memory file: {key} ({len(content)} chars)")
        return f"[See file: {key}]"

    def retrieve(self, key: str) -> str | None:
        """Retrieve content from file.

        Args:
            key: Key for the stored content.

        Returns:
            Content string if found, None otherwise.
        """
        # Handle reference format "[See file: key]"
        if key.startswith("[See file:"):
            key = key.replace("[See file:", "").replace("]", "").strip()

        path = self._index.get(key)
        if path and path.exists():
            return path.read_text(encoding="utf-8")

        # Try direct path lookup
        path = self.memory_dir / f"{key}.txt"
        if path.exists():
            content = path.read_text(encoding="utf-8")
            self._index[key] = path
            return content

        return None

    def summarize_for_context(self, key: str, max_chars: int = 500) -> str:
        """Get summarized version for context.

        Args:
            key: Key for the stored content.
            max_chars: Maximum characters for the summary.

        Returns:
            Summarized content or empty string if not found.
        """
        content = self.retrieve(key)
        if not content:
            return ""

        if len(content) <= max_chars:
            return content

        return content[:max_chars] + f"... [truncated, full content in {key}.txt]"

    def delete(self, key: str) -> bool:
        """Delete a memory file.

        Args:
            key: Key for the content to delete.

        Returns:
            True if deleted, False if not found.
        """
        path = self._index.get(key)
        if path and path.exists():
            path.unlink()
            del self._index[key]
            return True
        return False

    def list_keys(self) -> list[str]:
        """List all stored memory keys.

        Returns:
            List of memory keys.
        """
        return list(self._index.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get memory storage statistics.

        Returns:
            Dictionary with storage statistics.
        """
        total_size = 0
        for path in self._index.values():
            if path.exists():
                total_size += path.stat().st_size

        return {
            "file_count": len(self._index),
            "total_size_bytes": total_size,
            "memory_dir": str(self.memory_dir),
        }

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """Clean up memory files older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours.

        Returns:
            Number of files cleaned up.
        """
        cutoff = time.time() - (max_age_hours * 3600)
        cleaned = 0

        to_delete = []
        for key, path in self._index.items():
            if path.exists() and path.stat().st_mtime < cutoff:
                to_delete.append(key)

        for key in to_delete:
            if self.delete(key):
                cleaned += 1

        if cleaned:
            logger.info(f"Cleaned up {cleaned} old memory files")

        return cleaned


class ContextAwareSession:
    """Wrapper for agent sessions with context tracking.

    Adds token counting and context-aware features to agent sessions.
    """

    def __init__(
        self,
        context_manager: ContextManager,
        file_memory: FileMemory | None = None,
    ):
        """Initialize context-aware session.

        Args:
            context_manager: Context manager instance.
            file_memory: Optional file memory for large content.
        """
        self.context_manager = context_manager
        self.file_memory = file_memory
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.context_tokens: int = 0
        self._large_outputs: list[str] = []

    def update_token_count(self, input_t: int, output_t: int) -> None:
        """Update token counts after an API call.

        Args:
            input_t: Input tokens used.
            output_t: Output tokens generated.
        """
        self.input_tokens += input_t
        self.output_tokens += output_t
        self.context_tokens = self.input_tokens  # Approximation

    def should_compact_context(self) -> bool:
        """Check if context should be compacted based on token usage."""
        return self.context_manager.should_compact(self.context_tokens)

    def store_large_output(self, content: str, prefix: str = "output") -> str:
        """Store large output in file memory if available.

        Args:
            content: Content to potentially store.
            prefix: Prefix for storage key.

        Returns:
            Reference string if stored, original content otherwise.
        """
        # Threshold for "large" content (roughly 1000 tokens)
        large_threshold = 4000

        if len(content) > large_threshold and self.file_memory:
            ref = self.file_memory.store(content, prefix=prefix)
            self._large_outputs.append(ref)
            return ref

        return content

    def get_token_stats(self) -> dict[str, int]:
        """Get token usage statistics.

        Returns:
            Dictionary with token counts.
        """
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "context_tokens": self.context_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "large_outputs_stored": len(self._large_outputs),
        }
