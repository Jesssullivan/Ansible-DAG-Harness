#!/usr/bin/env python3
"""Generate MCP tool documentation from server definitions.

This script extracts tool definitions from the MCP server and generates
markdown documentation.

Usage:
    python scripts/generate-mcp-docs.py > docs/api/mcp-tools-generated.md
"""

import ast
import sys
from pathlib import Path
from typing import Optional


def extract_tools_from_source(source_path: Path) -> list[dict]:
    """Extract tool definitions from MCP server source code.

    Args:
        source_path: Path to the MCP server source file

    Returns:
        List of tool definitions with name, docstring, and parameters
    """
    with open(source_path) as f:
        source = f.read()

    tree = ast.parse(source)
    tools = []

    for node in ast.walk(tree):
        # Look for functions decorated with @mcp.tool()
        if isinstance(node, ast.FunctionDef):
            is_tool = False
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    if isinstance(decorator.func, ast.Attribute):
                        if decorator.func.attr == "tool":
                            is_tool = True
                    elif isinstance(decorator.func, ast.Name):
                        if decorator.func.id == "tool":
                            is_tool = True

            if is_tool:
                tool = extract_tool_info(node)
                if tool:
                    tools.append(tool)

    return tools


def extract_tool_info(node: ast.FunctionDef) -> Optional[dict]:
    """Extract tool information from a function definition.

    Args:
        node: AST function definition node

    Returns:
        Dict with tool name, docstring, and parameters
    """
    name = node.name
    docstring = ast.get_docstring(node) or ""

    # Extract parameters
    params = []
    for arg in node.args.args:
        if arg.arg == "self":
            continue

        param = {
            "name": arg.arg,
            "type": "Any",
            "description": "",
        }

        # Try to get type annotation
        if arg.annotation:
            param["type"] = ast.unparse(arg.annotation)

        params.append(param)

    # Parse parameter descriptions from docstring
    if docstring:
        lines = docstring.split("\n")
        in_params = False
        current_param = None

        for line in lines:
            stripped = line.strip()

            if stripped.lower().startswith("args:") or stripped.lower().startswith("parameters:"):
                in_params = True
                continue

            if stripped.lower().startswith("returns:"):
                in_params = False
                continue

            if in_params and stripped:
                # Check for parameter definition
                if ":" in stripped and not stripped.startswith(" "):
                    parts = stripped.split(":", 1)
                    param_name = parts[0].strip()
                    param_desc = parts[1].strip() if len(parts) > 1 else ""

                    # Find matching parameter
                    for p in params:
                        if p["name"] == param_name:
                            p["description"] = param_desc
                            current_param = p
                            break
                elif current_param and stripped:
                    # Continuation of previous parameter description
                    current_param["description"] += " " + stripped

    # Extract return description
    returns = ""
    if "Returns:" in docstring:
        parts = docstring.split("Returns:", 1)
        if len(parts) > 1:
            returns_section = parts[1].strip()
            # Take first paragraph
            returns = returns_section.split("\n\n")[0].strip()

    return {
        "name": name,
        "docstring": docstring.split("\n")[0] if docstring else "",
        "full_docstring": docstring,
        "params": params,
        "returns": returns,
    }


def generate_markdown(tools: list[dict]) -> str:
    """Generate markdown documentation from tool definitions.

    Args:
        tools: List of tool definitions

    Returns:
        Markdown documentation string
    """
    lines = [
        "# MCP Tools Reference (Auto-Generated)",
        "",
        "> This documentation is auto-generated from the MCP server source code.",
        "",
        "## Available Tools",
        "",
    ]

    # Table of contents
    lines.append("| Tool | Description |")
    lines.append("|------|-------------|")
    for tool in sorted(tools, key=lambda t: t["name"]):
        lines.append(f"| [`{tool['name']}`](#{tool['name']}) | {tool['docstring']} |")
    lines.append("")

    # Tool details
    lines.append("## Tool Details")
    lines.append("")

    for tool in sorted(tools, key=lambda t: t["name"]):
        lines.append(f"### {tool['name']}")
        lines.append("")

        if tool["docstring"]:
            lines.append(tool["docstring"])
            lines.append("")

        # Signature
        param_strs = [f"{p['name']}: {p['type']}" for p in tool["params"]]
        signature = f"{tool['name']}({', '.join(param_strs)})"
        lines.append("```python")
        lines.append(signature)
        lines.append("```")
        lines.append("")

        # Parameters
        if tool["params"]:
            lines.append("**Parameters:**")
            lines.append("")
            lines.append("| Name | Type | Description |")
            lines.append("|------|------|-------------|")
            for p in tool["params"]:
                desc = p["description"] or "-"
                lines.append(f"| `{p['name']}` | `{p['type']}` | {desc} |")
            lines.append("")

        # Returns
        if tool["returns"]:
            lines.append("**Returns:**")
            lines.append("")
            lines.append(tool["returns"])
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main():
    """Main entry point."""
    # Find MCP server source
    project_root = Path(__file__).parent.parent
    server_path = project_root / "harness" / "harness" / "mcp" / "server.py"

    if not server_path.exists():
        print(f"Error: MCP server not found at {server_path}", file=sys.stderr)
        sys.exit(1)

    # Extract tools
    tools = extract_tools_from_source(server_path)

    if not tools:
        print("Warning: No tools found in MCP server", file=sys.stderr)

    # Generate markdown
    markdown = generate_markdown(tools)
    print(markdown)


if __name__ == "__main__":
    main()
