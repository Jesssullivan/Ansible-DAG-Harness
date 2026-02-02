"""
Tests for HOTL context management (Manus patterns).

Tests cover:
- ContextConfig validation
- Token estimation
- Context compaction logic
- Diversity injection
- FileMemory storage/retrieval
- ContextAwareSession token tracking
- Error preservation during compaction
"""

import time

import pytest

from harness.hotl.context import (
    ContextAwareSession,
    ContextConfig,
    ContextManager,
    FileMemory,
)
from harness.hotl.state import create_initial_state

# ============================================================================
# CONTEXT CONFIG TESTS
# ============================================================================


class TestContextConfig:
    """Tests for ContextConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ContextConfig()

        assert config.compaction_threshold == 0.25
        assert config.max_context_tokens == 200000
        assert config.keep_recent_changes == 5
        assert config.keep_recent_progress == 10
        assert config.preserve_errors is True
        assert config.diversity_enabled is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = ContextConfig(
            compaction_threshold=0.5,
            max_context_tokens=100000,
            keep_recent_changes=10,
            keep_recent_progress=20,
            preserve_errors=False,
            diversity_enabled=False,
        )

        assert config.compaction_threshold == 0.5
        assert config.max_context_tokens == 100000
        assert config.keep_recent_changes == 10
        assert config.keep_recent_progress == 20
        assert config.preserve_errors is False
        assert config.diversity_enabled is False

    def test_invalid_compaction_threshold_low(self):
        """Test that compaction_threshold must be > 0."""
        with pytest.raises(ValueError, match="compaction_threshold"):
            ContextConfig(compaction_threshold=0.0)

    def test_invalid_compaction_threshold_high(self):
        """Test that compaction_threshold must be < 1."""
        with pytest.raises(ValueError, match="compaction_threshold"):
            ContextConfig(compaction_threshold=1.0)

    def test_invalid_max_context_tokens(self):
        """Test that max_context_tokens must be positive."""
        with pytest.raises(ValueError, match="max_context_tokens"):
            ContextConfig(max_context_tokens=0)

    def test_invalid_keep_recent_changes(self):
        """Test that keep_recent_changes must be non-negative."""
        with pytest.raises(ValueError, match="keep_recent_changes"):
            ContextConfig(keep_recent_changes=-1)

    def test_invalid_keep_recent_progress(self):
        """Test that keep_recent_progress must be non-negative."""
        with pytest.raises(ValueError, match="keep_recent_progress"):
            ContextConfig(keep_recent_progress=-1)


# ============================================================================
# CONTEXT MANAGER TESTS
# ============================================================================


class TestContextManager:
    """Tests for ContextManager class."""

    @pytest.fixture
    def manager(self):
        """Create a default context manager."""
        return ContextManager()

    @pytest.fixture
    def custom_manager(self):
        """Create a context manager with custom config."""
        config = ContextConfig(
            compaction_threshold=0.1,
            max_context_tokens=10000,
            keep_recent_progress=3,
        )
        return ContextManager(config)

    def test_estimate_tokens_empty(self, manager):
        """Test token estimation for empty text."""
        assert manager.estimate_tokens("") == 0
        assert manager.estimate_tokens(None) == 0

    def test_estimate_tokens_simple(self, manager):
        """Test token estimation for simple text."""
        text = "Hello, world!"  # 13 chars
        # Roughly 3-4 tokens
        tokens = manager.estimate_tokens(text)
        assert 2 <= tokens <= 5

    def test_estimate_tokens_long_text(self, manager):
        """Test token estimation for longer text."""
        # 1000 characters should be ~250 tokens
        text = "a" * 1000
        tokens = manager.estimate_tokens(text)
        assert 200 <= tokens <= 300

    def test_estimate_state_tokens(self, manager):
        """Test token estimation for HOTL state."""
        state = create_initial_state()
        tokens = manager.estimate_state_tokens(state)
        assert tokens > 0
        assert manager.stats.estimated_tokens == tokens

    def test_should_compact_below_threshold(self, manager):
        """Test that compaction is not triggered below threshold."""
        # Default threshold is 25% of 200000 = 50000
        assert not manager.should_compact(10000)
        assert not manager.should_compact(50000)

    def test_should_compact_above_threshold(self, manager):
        """Test that compaction is triggered above threshold."""
        # Above 25% of 200000 = 50000
        assert manager.should_compact(50001)
        assert manager.should_compact(100000)

    def test_should_compact_uses_stats(self, manager):
        """Test that should_compact uses stats when no count provided."""
        manager._stats.estimated_tokens = 60000
        assert manager.should_compact()

        manager._stats.estimated_tokens = 10000
        assert not manager.should_compact()

    def test_compact_context_basic(self, manager):
        """Test basic context compaction."""
        state = create_initial_state()
        # Add some data to compact
        state["codebase_insights"] = [f"Insight {i}" for i in range(20)]
        state["research_findings"] = [{"type": f"finding_{i}"} for i in range(10)]

        compacted = manager.compact_context(state)

        # Should have fewer detailed insights
        assert len(compacted.get("codebase_insights", [])) < 20
        assert manager.stats.compaction_count == 1
        assert manager.stats.items_summarized > 0

    def test_compact_context_preserves_errors(self, manager):
        """Test that compaction preserves error stack traces."""
        state = create_initial_state()
        error_insight = (
            "Traceback (most recent call last):\n  File test.py, line 1\nValueError: test error"
        )
        state["codebase_insights"] = [f"Normal insight {i}" for i in range(15)] + [error_insight]

        compacted = manager.compact_context(state)

        # Error should still be present
        insights = compacted.get("codebase_insights", [])
        assert any("Traceback" in str(i) for i in insights)

    def test_compact_context_preserves_recent_findings(self, manager):
        """Test that recent findings are kept in detail."""
        state = create_initial_state()
        state["research_findings"] = [
            {"type": f"finding_{i}", "data": "x" * 500} for i in range(10)
        ]

        compacted = manager.compact_context(state)

        # Last 5 should be preserved in full
        findings = compacted.get("research_findings", [])
        assert len(findings) == 10  # All kept, but older ones compacted
        # Recent ones should have full data
        assert len(str(findings[-1].get("data", ""))) == 500

    def test_contains_error_patterns(self, manager):
        """Test error pattern detection."""
        assert manager._contains_error("Error: something went wrong")
        assert manager._contains_error("EXCEPTION occurred")
        assert manager._contains_error("Traceback (most recent call last):")
        assert manager._contains_error("raise ValueError('test')")
        assert manager._contains_error('  File "test.py", line 42')

        assert not manager._contains_error("Everything is fine")
        assert not manager._contains_error("No problems here")

    def test_inject_diversity(self, manager):
        """Test diversity injection."""
        prompt = "Do the thing."
        diversified = manager.inject_diversity(prompt)

        # Should have a prefix added
        assert diversified.endswith(prompt)
        assert len(diversified) > len(prompt)

    def test_inject_diversity_rotates_templates(self, manager):
        """Test that diversity injection rotates through templates."""
        prompts_used = set()

        for _ in range(len(manager.TASK_TEMPLATES)):
            result = manager.inject_diversity("test")
            # Extract the prefix
            prefix = result.replace(" test", "")
            prompts_used.add(prefix)

        # Should have used multiple templates
        assert len(prompts_used) > 1

    def test_inject_diversity_disabled(self):
        """Test that diversity can be disabled."""
        config = ContextConfig(diversity_enabled=False)
        manager = ContextManager(config)

        prompt = "Do the thing."
        result = manager.inject_diversity(prompt)

        assert result == prompt

    def test_reset_diversity(self, manager):
        """Test diversity template reset."""
        # Use some templates
        for _ in range(5):
            manager.inject_diversity("test")

        assert len(manager._used_templates) > 0

        manager.reset_diversity()
        assert len(manager._used_templates) == 0


# ============================================================================
# FILE MEMORY TESTS
# ============================================================================


class TestFileMemory:
    """Tests for FileMemory class."""

    @pytest.fixture
    def memory(self, tmp_path):
        """Create a file memory instance."""
        return FileMemory(tmp_path / "memory")

    def test_store_and_retrieve(self, memory):
        """Test basic store and retrieve."""
        content = "This is test content."
        ref = memory.store(content)

        retrieved = memory.retrieve(ref.replace("[See file:", "").replace("]", "").strip())
        assert retrieved == content

    def test_store_with_custom_key(self, memory):
        """Test storing with a custom key."""
        content = "Custom key content."
        ref = memory.store(content, key="my_custom_key")

        assert "my_custom_key" in ref
        retrieved = memory.retrieve("my_custom_key")
        assert retrieved == content

    def test_store_with_prefix(self, memory):
        """Test storing with a custom prefix."""
        content = "Prefixed content."
        ref = memory.store(content, prefix="custom")

        assert "custom_" in ref

    def test_retrieve_with_reference_format(self, memory):
        """Test retrieving using the reference format."""
        content = "Reference format content."
        memory.store(content, key="ref_test")

        retrieved = memory.retrieve("[See file: ref_test]")
        assert retrieved == content

    def test_retrieve_nonexistent(self, memory):
        """Test retrieving non-existent key."""
        result = memory.retrieve("nonexistent_key")
        assert result is None

    def test_summarize_for_context_short(self, memory):
        """Test summarization of short content."""
        short_content = "Short content."
        memory.store(short_content, key="short")

        summary = memory.summarize_for_context("short", max_chars=500)
        assert summary == short_content

    def test_summarize_for_context_long(self, memory):
        """Test summarization of long content."""
        long_content = "x" * 1000
        memory.store(long_content, key="long")

        summary = memory.summarize_for_context("long", max_chars=100)
        assert len(summary) < 200  # Should be truncated
        assert "truncated" in summary
        assert "long.txt" in summary

    def test_delete(self, memory):
        """Test deleting memory files."""
        memory.store("To delete", key="delete_me")

        assert memory.retrieve("delete_me") is not None
        assert memory.delete("delete_me") is True
        assert memory.retrieve("delete_me") is None
        assert memory.delete("delete_me") is False

    def test_list_keys(self, memory):
        """Test listing stored keys."""
        memory.store("Content 1", key="key1")
        memory.store("Content 2", key="key2")
        memory.store("Content 3", key="key3")

        keys = memory.list_keys()
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys

    def test_get_stats(self, memory):
        """Test getting memory statistics."""
        memory.store("Content 1", key="stat1")
        memory.store("Content 2" * 100, key="stat2")

        stats = memory.get_stats()
        assert stats["file_count"] == 2
        assert stats["total_size_bytes"] > 0
        assert "memory_dir" in stats

    def test_cleanup_old(self, memory, tmp_path):
        """Test cleaning up old memory files."""
        # Create files manually with old timestamps
        memory.store("Recent", key="recent")

        old_path = memory.memory_dir / "old.txt"
        old_path.write_text("Old content")
        # Set old modification time (>24 hours ago)
        old_time = time.time() - (25 * 3600)
        import os

        os.utime(old_path, (old_time, old_time))

        # Reload index
        memory._load_index()

        cleaned = memory.cleanup_old(max_age_hours=24)
        assert cleaned == 1
        assert memory.retrieve("recent") is not None
        assert memory.retrieve("old") is None


# ============================================================================
# CONTEXT AWARE SESSION TESTS
# ============================================================================


class TestContextAwareSession:
    """Tests for ContextAwareSession class."""

    @pytest.fixture
    def manager(self):
        """Create a context manager."""
        return ContextManager()

    @pytest.fixture
    def memory(self, tmp_path):
        """Create a file memory."""
        return FileMemory(tmp_path / "session_memory")

    @pytest.fixture
    def session(self, manager, memory):
        """Create a context-aware session."""
        return ContextAwareSession(manager, memory)

    def test_initial_token_counts(self, session):
        """Test initial token counts are zero."""
        assert session.input_tokens == 0
        assert session.output_tokens == 0
        assert session.context_tokens == 0

    def test_update_token_count(self, session):
        """Test updating token counts."""
        session.update_token_count(100, 50)
        assert session.input_tokens == 100
        assert session.output_tokens == 50
        assert session.context_tokens == 100

        session.update_token_count(200, 100)
        assert session.input_tokens == 300
        assert session.output_tokens == 150
        assert session.context_tokens == 300

    def test_should_compact_context(self, session):
        """Test context compaction check."""
        # Initially false
        assert not session.should_compact_context()

        # After many tokens
        session.context_tokens = 60000  # Above 25% of 200000
        session.context_manager._stats.estimated_tokens = 60000
        # Note: should_compact uses stats, so we need to update it

    def test_store_large_output_small(self, session):
        """Test that small outputs are not stored."""
        small_output = "Small output"
        result = session.store_large_output(small_output)
        assert result == small_output
        assert len(session._large_outputs) == 0

    def test_store_large_output_large(self, session):
        """Test that large outputs are stored."""
        large_output = "x" * 5000  # Above threshold
        result = session.store_large_output(large_output)

        assert result != large_output
        assert "[See file:" in result
        assert len(session._large_outputs) == 1

    def test_store_large_output_no_memory(self, manager):
        """Test large output handling without file memory."""
        session = ContextAwareSession(manager, file_memory=None)
        large_output = "x" * 5000

        result = session.store_large_output(large_output)
        assert result == large_output  # Not stored, returned as-is

    def test_get_token_stats(self, session):
        """Test getting token statistics."""
        session.update_token_count(500, 200)
        session.store_large_output("x" * 5000)

        stats = session.get_token_stats()

        assert stats["input_tokens"] == 500
        assert stats["output_tokens"] == 200
        assert stats["context_tokens"] == 500
        assert stats["total_tokens"] == 700
        assert stats["large_outputs_stored"] == 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestContextIntegration:
    """Integration tests for context management components."""

    @pytest.fixture
    def full_setup(self, tmp_path):
        """Create a full context management setup."""
        config = ContextConfig(
            compaction_threshold=0.2,
            max_context_tokens=10000,
        )
        manager = ContextManager(config)
        memory = FileMemory(tmp_path / "integration_memory")
        session = ContextAwareSession(manager, memory)
        return manager, memory, session

    def test_full_workflow(self, full_setup):
        """Test a complete context management workflow."""
        manager, memory, session = full_setup

        # Create a state with lots of data
        state = create_initial_state()
        state["codebase_insights"] = [f"Insight {i}: " + "x" * 100 for i in range(30)]
        state["research_findings"] = [{"id": i, "data": "y" * 200} for i in range(20)]
        state["errors"] = ["Error: Test error with Traceback (most recent call last):"]

        # Estimate tokens
        tokens = manager.estimate_state_tokens(state)
        assert tokens > 0

        # Check if compaction needed
        if manager.should_compact(tokens):
            compacted = manager.compact_context(state)

            # Verify compaction happened
            assert manager.stats.compaction_count > 0

            # Verify errors preserved
            assert any("Traceback" in str(e) for e in compacted.get("errors", []))

        # Store large output
        large_result = "z" * 5000
        ref = session.store_large_output(large_result, prefix="result")
        assert "[See file:" in ref or ref == large_result

        # Update token counts
        session.update_token_count(1000, 500)
        stats = session.get_token_stats()
        assert stats["total_tokens"] == 1500

    def test_diversity_across_multiple_tasks(self, full_setup):
        """Test diversity injection across multiple task creations."""
        manager, _, _ = full_setup

        tasks = []
        for i in range(15):
            task = manager.inject_diversity(f"Task {i}")
            tasks.append(task)

        # Should have variety in prefixes
        prefixes = set()
        for task in tasks:
            # Extract prefix (everything before "Task")
            prefix = task.split("Task")[0].strip()
            prefixes.add(prefix)

        # Should have multiple different prefixes
        assert len(prefixes) > 3
