"""Skills framework for agent capabilities.

This module provides a skills system that extends agent capabilities
with predefined, composable actions. Skills encapsulate common tasks
that agents can perform autonomously.

Skills Types:
    - TestingSkill: Run tests, analyze failures
    - DependencySkill: Analyze dependencies, check updates
    - Custom skills via base class

Usage:
    from harness.skills import TestingSkill, SkillRegistry

    registry = SkillRegistry()
    registry.register(TestingSkill())

    result = await registry.execute("testing", "run_tests", context)
"""

from harness.skills.base import (
    Skill,
    SkillAction,
    SkillContext,
    SkillError,
    SkillRegistry,
    SkillResult,
    SkillStatus,
)
from harness.skills.dependencies import (
    DependencyInfo,
    DependencySkill,
    DependencyUpdate,
)
from harness.skills.testing import (
    TestFailure,
    TestingSkill,
    TestResult,
)

__all__ = [
    # Base classes
    "Skill",
    "SkillAction",
    "SkillContext",
    "SkillError",
    "SkillRegistry",
    "SkillResult",
    "SkillStatus",
    # Testing skill
    "TestingSkill",
    "TestResult",
    "TestFailure",
    # Dependency skill
    "DependencySkill",
    "DependencyInfo",
    "DependencyUpdate",
]
