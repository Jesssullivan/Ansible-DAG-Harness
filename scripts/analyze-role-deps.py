#!/usr/bin/env python3
"""
Standalone CLI for role dependency analysis.

This script provides a command-line interface for analyzing Ansible role
dependencies. It delegates to the harness StateDB for actual data retrieval.

Usage:
    python scripts/analyze-role-deps.py <role_name> [--json]
    python scripts/analyze-role-deps.py <role_name> --transitive
    python scripts/analyze-role-deps.py --list-all
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add harness to path if needed
harness_path = Path(__file__).parent.parent / "harness"
if harness_path.exists():
    sys.path.insert(0, str(harness_path))


def get_db():
    """Get database connection."""
    from harness.db.state import StateDB

    db_path = os.environ.get("HARNESS_DB_PATH", "./harness/harness.db")
    return StateDB(db_path)


def analyze_role(role_name: str, transitive: bool = False) -> dict:
    """Analyze dependencies for a single role."""
    db = get_db()

    role = db.get_role(role_name)
    if not role:
        return {"error": f"Role '{role_name}' not found in database"}

    # Get dependencies
    explicit_deps = db.get_dependencies(role_name, transitive=False)
    all_deps = db.get_dependencies(role_name, transitive=transitive)
    reverse_deps = db.get_reverse_dependencies(role_name, transitive=False)

    # Get credentials
    credentials = db.get_credentials(role_name)

    # Build result
    result = {
        "role": role_name,
        "wave": role.wave,
        "wave_name": role.wave_name,
        "has_molecule_tests": role.has_molecule_tests,
        "explicit_deps": [dep[0] for dep in explicit_deps],
        "reverse_deps": [dep[0] for dep in reverse_deps],
        "credentials": [
            {
                "entry_name": c.entry_name,
                "purpose": c.purpose,
                "attribute": c.attribute,
                "is_base58": c.is_base58,
            }
            for c in credentials
        ],
        "tags": [],  # Not yet implemented
    }

    if transitive:
        # Add transitive deps (those with depth > 1)
        transitive_deps = [dep[0] for dep in all_deps if dep[1] > 1]
        result["transitive_deps"] = transitive_deps
        result["implicit_deps"] = transitive_deps  # Alias for compatibility

    return result


def list_all_roles() -> list[dict]:
    """List all roles with basic info."""
    db = get_db()
    roles = db.list_roles()

    return [
        {
            "name": r.name,
            "wave": r.wave,
            "wave_name": r.wave_name,
            "has_molecule_tests": r.has_molecule_tests,
        }
        for r in roles
    ]


def get_deployment_order() -> list[str]:
    """Get topologically sorted deployment order."""
    db = get_db()
    return db.get_deployment_order(raise_on_cycle=False)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze Ansible role dependencies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single role
    %(prog)s postgresql

    # Get JSON output
    %(prog)s postgresql --json

    # Include transitive dependencies
    %(prog)s postgresql --transitive

    # List all roles
    %(prog)s --list-all

    # Get deployment order
    %(prog)s --deployment-order
        """,
    )

    parser.add_argument("role_name", nargs="?", help="Name of the role to analyze")
    parser.add_argument(
        "--json", "-j", action="store_true", help="Output as JSON"
    )
    parser.add_argument(
        "--transitive", "-t", action="store_true", help="Include transitive dependencies"
    )
    parser.add_argument(
        "--list-all", "-l", action="store_true", help="List all roles"
    )
    parser.add_argument(
        "--deployment-order", "-o", action="store_true", help="Show deployment order"
    )

    args = parser.parse_args()

    try:
        if args.list_all:
            result = list_all_roles()
        elif args.deployment_order:
            result = get_deployment_order()
        elif args.role_name:
            result = analyze_role(args.role_name, transitive=args.transitive)
        else:
            parser.print_help()
            sys.exit(1)

        if args.json or args.list_all or args.deployment_order:
            print(json.dumps(result, indent=2))
        else:
            # Pretty print for single role
            if "error" in result:
                print(f"Error: {result['error']}", file=sys.stderr)
                sys.exit(1)

            print(f"Role: {result['role']}")
            print(f"Wave: {result['wave']} ({result['wave_name']})")
            print(f"Has molecule tests: {result['has_molecule_tests']}")
            print()

            print("Direct dependencies:")
            for dep in result.get("explicit_deps", []):
                print(f"  - {dep}")
            if not result.get("explicit_deps"):
                print("  (none)")
            print()

            if args.transitive and result.get("transitive_deps"):
                print("Transitive dependencies:")
                for dep in result.get("transitive_deps", []):
                    print(f"  - {dep}")
                print()

            print("Reverse dependencies (roles that depend on this):")
            for dep in result.get("reverse_deps", []):
                print(f"  - {dep}")
            if not result.get("reverse_deps"):
                print("  (none)")
            print()

            print("Credentials required:")
            for cred in result.get("credentials", []):
                attr = f"[{cred['attribute']}]" if cred.get("attribute") else ""
                print(f"  - {cred['entry_name']}{attr}: {cred.get('purpose', 'unknown')}")
            if not result.get("credentials"):
                print("  (none)")

    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
