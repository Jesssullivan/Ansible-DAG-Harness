"""AgentCard dataclass and utilities for A2A protocol.

The AgentCard is the primary mechanism for agent discovery and capability
advertisement in the A2A protocol. It is analogous to an API schema but
focused on agent-to-agent communication capabilities.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.a2a.protocol import A2ACapability


@dataclass
class AgentEndpoint:
    """Describes an endpoint exposed by an agent."""

    name: str
    path: str
    method: str = "POST"
    description: str = ""
    accepts: list[str] = field(default_factory=lambda: ["application/json"])
    returns: list[str] = field(default_factory=lambda: ["application/json"])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "method": self.method,
            "description": self.description,
            "accepts": self.accepts,
            "returns": self.returns,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentEndpoint":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            path=data["path"],
            method=data.get("method", "POST"),
            description=data.get("description", ""),
            accepts=data.get("accepts", ["application/json"]),
            returns=data.get("returns", ["application/json"]),
        )


@dataclass
class AgentCard:
    """
    Agent Card describing an agent's identity and capabilities.

    This is the primary discovery mechanism for A2A communication.
    Agents expose their card at a well-known endpoint (e.g., /a2a/discover
    or /.well-known/agent-card.json) to allow other agents to discover
    and understand their capabilities.

    Attributes:
        name: Unique agent identifier
        description: Human-readable description of the agent
        version: Agent version string
        capabilities: List of capabilities this agent supports
        endpoints: List of endpoints exposed by this agent
        metadata: Additional metadata about the agent
        base_url: Base URL for agent endpoints (if remote)
        mcp_server: MCP server info if agent has MCP integration
        owner: Agent owner/maintainer info
        tags: Tags for categorization
    """

    name: str
    description: str
    version: str
    capabilities: list[str] = field(default_factory=list)
    endpoints: list[AgentEndpoint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    base_url: str | None = None
    mcp_server: dict[str, Any] | None = None
    owner: str | None = None
    tags: list[str] = field(default_factory=list)

    def has_capability(self, capability: str | A2ACapability) -> bool:
        """Check if agent has a specific capability.

        Args:
            capability: Capability to check (string or enum)

        Returns:
            True if agent has the capability
        """
        cap_str = capability.value if isinstance(capability, A2ACapability) else capability
        return cap_str in self.capabilities

    def get_endpoint(self, name: str) -> AgentEndpoint | None:
        """Get an endpoint by name.

        Args:
            name: Endpoint name

        Returns:
            AgentEndpoint if found, None otherwise
        """
        for endpoint in self.endpoints:
            if endpoint.name == name:
                return endpoint
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "capabilities": self.capabilities,
            "endpoints": [ep.to_dict() for ep in self.endpoints],
            "metadata": self.metadata,
            "base_url": self.base_url,
            "mcp_server": self.mcp_server,
            "owner": self.owner,
            "tags": self.tags,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentCard":
        """Create AgentCard from dictionary.

        Args:
            data: Dictionary with agent card data

        Returns:
            AgentCard instance
        """
        endpoints = [
            AgentEndpoint.from_dict(ep) if isinstance(ep, dict) else ep
            for ep in data.get("endpoints", [])
        ]
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "0.0.0"),
            capabilities=data.get("capabilities", []),
            endpoints=endpoints,
            metadata=data.get("metadata", {}),
            base_url=data.get("base_url"),
            mcp_server=data.get("mcp_server"),
            owner=data.get("owner"),
            tags=data.get("tags", []),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "AgentCard":
        """Create AgentCard from JSON string.

        Args:
            json_str: JSON string

        Returns:
            AgentCard instance
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> "AgentCard":
        """Load AgentCard from JSON file.

        Args:
            path: Path to JSON file

        Returns:
            AgentCard instance
        """
        with open(path) as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save_to_file(self, path: str | Path) -> None:
        """Save AgentCard to JSON file.

        Args:
            path: Path to save JSON file
        """
        with open(path, "w") as f:
            f.write(self.to_json())


def create_harness_agent_card(
    base_url: str | None = None,
    version: str = "0.2.0",
) -> AgentCard:
    """Create the default agent card for the DAG harness.

    Args:
        base_url: Base URL for the harness server
        version: Harness version string

    Returns:
        AgentCard for the DAG harness
    """
    return AgentCard(
        name="dag-harness",
        description="DAG orchestration harness for Ansible role management with LangGraph "
        "workflow execution and MCP integration",
        version=version,
        capabilities=[
            A2ACapability.INVOKE.value,
            A2ACapability.HANDOFF.value,
            A2ACapability.DISCOVERY.value,
            A2ACapability.ANSIBLE_ROLES.value,
            A2ACapability.DEPENDENCY_ANALYSIS.value,
            A2ACapability.WORKFLOW_EXECUTION.value,
            A2ACapability.TESTING.value,
            A2ACapability.MCP_CLIENT.value,
            A2ACapability.GITLAB_INTEGRATION.value,
        ],
        endpoints=[
            AgentEndpoint(
                name="discover",
                path="/a2a/discover",
                method="GET",
                description="Discover agent capabilities and endpoints",
            ),
            AgentEndpoint(
                name="invoke",
                path="/a2a/invoke",
                method="POST",
                description="Invoke an agent capability",
            ),
            AgentEndpoint(
                name="health",
                path="/a2a/health",
                method="GET",
                description="Health check endpoint",
            ),
            AgentEndpoint(
                name="handoff",
                path="/a2a/handoff",
                method="POST",
                description="Request handoff from another agent",
            ),
        ],
        metadata={
            "protocol_version": "1.0",
            "supported_message_types": [
                "discover_request",
                "invoke_request",
                "handoff_request",
                "health_check",
            ],
        },
        base_url=base_url,
        mcp_server={
            "name": "dag-harness",
            "transport": "stdio",
            "command": ["harness", "mcp"],
        },
        owner="DAG Harness Team",
        tags=["ansible", "dag", "langgraph", "orchestration", "mcp"],
    )
