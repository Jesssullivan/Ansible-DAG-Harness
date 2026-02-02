"""Token usage tracking and cost aggregation.

Provides per-session token accounting and cost summaries.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

from harness.costs.pricing import CLAUDE_PRICING, get_model_pricing
from harness.db.state import StateDB


@dataclass
class TokenUsageRecord:
    """A single token usage record.

    Attributes:
        id: Database record ID
        session_id: Session identifier
        model: Model used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        cost: Calculated cost in USD
        timestamp: When the usage occurred
        context: Optional context metadata (JSON)
    """

    id: int | None
    session_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: Decimal
    timestamp: datetime | None
    context: str | None = None


@dataclass
class CostSummary:
    """Summary of costs over a period.

    Attributes:
        total_cost: Total cost in USD
        total_input_tokens: Total input tokens
        total_output_tokens: Total output tokens
        record_count: Number of usage records
        by_model: Cost breakdown by model
        by_session: Cost breakdown by session (if available)
    """

    total_cost: Decimal
    total_input_tokens: int
    total_output_tokens: int
    record_count: int
    by_model: dict[str, Decimal]
    by_session: dict[str, Decimal] | None = None


class TokenUsageTracker:
    """Tracks token usage and calculates costs.

    Uses the database to persist usage records and provides
    aggregation methods for reporting.
    """

    def __init__(self, db: StateDB):
        """Initialize tracker with database connection.

        Args:
            db: StateDB instance for persistence
        """
        self.db = db
        self._ensure_table_exists()

    def _ensure_table_exists(self) -> None:
        """Ensure the token_usage table exists."""
        with self.db.connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS token_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    model TEXT NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost REAL NOT NULL,
                    context TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Create indexes for common queries
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_token_usage_session
                ON token_usage(session_id)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_token_usage_model
                ON token_usage(model)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_token_usage_timestamp
                ON token_usage(timestamp)
                """
            )

    def record_usage(
        self,
        session_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        context: dict[str, Any] | None = None,
    ) -> TokenUsageRecord:
        """Record token usage for a session.

        Args:
            session_id: Session identifier
            model: Model used (e.g., "claude-opus-4-5")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            context: Optional context metadata

        Returns:
            TokenUsageRecord with calculated cost

        Raises:
            ValueError: If model pricing is not found
        """
        pricing = get_model_pricing(model)
        if pricing is None:
            # Use a fallback for unknown models (assume Sonnet pricing)
            from harness.costs.pricing import ModelPricing

            pricing = CLAUDE_PRICING.get(
                "claude-sonnet-4",
                ModelPricing(
                    model_id=model,
                    input_cost_per_mtok=Decimal("3.00"),
                    output_cost_per_mtok=Decimal("15.00"),
                    display_name=model,
                ),
            )

        cost = pricing.calculate_cost(input_tokens, output_tokens)

        import json

        context_json = json.dumps(context) if context else None

        with self.db.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO token_usage
                (session_id, model, input_tokens, output_tokens, cost, context)
                VALUES (?, ?, ?, ?, ?, ?)
                RETURNING id, timestamp
                """,
                (session_id, model, input_tokens, output_tokens, float(cost), context_json),
            )
            row = cursor.fetchone()

        return TokenUsageRecord(
            id=row[0],
            session_id=session_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            timestamp=row[1],
            context=context_json,
        )

    def get_session_cost(self, session_id: str) -> CostSummary:
        """Get total cost for a specific session.

        Args:
            session_id: Session identifier

        Returns:
            CostSummary for the session
        """
        with self.db.connection() as conn:
            # Get totals
            row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(cost), 0) as total_cost,
                    COALESCE(SUM(input_tokens), 0) as total_input,
                    COALESCE(SUM(output_tokens), 0) as total_output,
                    COUNT(*) as record_count
                FROM token_usage
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()

            # Get breakdown by model
            model_rows = conn.execute(
                """
                SELECT model, SUM(cost) as model_cost
                FROM token_usage
                WHERE session_id = ?
                GROUP BY model
                """,
                (session_id,),
            ).fetchall()

        by_model = {r["model"]: Decimal(str(r["model_cost"])) for r in model_rows}

        return CostSummary(
            total_cost=Decimal(str(row["total_cost"])),
            total_input_tokens=row["total_input"],
            total_output_tokens=row["total_output"],
            record_count=row["record_count"],
            by_model=by_model,
            by_session={session_id: Decimal(str(row["total_cost"]))},
        )

    def get_total_cost(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> CostSummary:
        """Get total cost over a date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            CostSummary for the period
        """
        with self.db.connection() as conn:
            # Build query with optional date filters
            query = """
                SELECT
                    COALESCE(SUM(cost), 0) as total_cost,
                    COALESCE(SUM(input_tokens), 0) as total_input,
                    COALESCE(SUM(output_tokens), 0) as total_output,
                    COUNT(*) as record_count
                FROM token_usage
                WHERE 1=1
            """
            params: list[Any] = []

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())

            row = conn.execute(query, params).fetchone()

            # Get breakdown by model
            model_query = """
                SELECT model, SUM(cost) as model_cost
                FROM token_usage
                WHERE 1=1
            """
            if start_date:
                model_query += " AND timestamp >= ?"
            if end_date:
                model_query += " AND timestamp <= ?"
            model_query += " GROUP BY model"

            model_rows = conn.execute(model_query, params).fetchall()

            # Get breakdown by session
            session_query = """
                SELECT session_id, SUM(cost) as session_cost
                FROM token_usage
                WHERE 1=1
            """
            if start_date:
                session_query += " AND timestamp >= ?"
            if end_date:
                session_query += " AND timestamp <= ?"
            session_query += " GROUP BY session_id"

            session_rows = conn.execute(session_query, params).fetchall()

        by_model = {r["model"]: Decimal(str(r["model_cost"])) for r in model_rows}
        by_session = {r["session_id"]: Decimal(str(r["session_cost"])) for r in session_rows}

        return CostSummary(
            total_cost=Decimal(str(row["total_cost"])),
            total_input_tokens=row["total_input"],
            total_output_tokens=row["total_output"],
            record_count=row["record_count"],
            by_model=by_model,
            by_session=by_session,
        )

    def get_costs_by_model(
        self,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Get detailed cost breakdown by model.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)

        Returns:
            Dict mapping model to cost details
        """
        with self.db.connection() as conn:
            query = """
                SELECT
                    model,
                    SUM(cost) as total_cost,
                    SUM(input_tokens) as total_input,
                    SUM(output_tokens) as total_output,
                    COUNT(*) as record_count,
                    MIN(timestamp) as first_use,
                    MAX(timestamp) as last_use
                FROM token_usage
                WHERE 1=1
            """
            params: list[Any] = []

            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date.isoformat())
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date.isoformat())

            query += " GROUP BY model ORDER BY total_cost DESC"

            rows = conn.execute(query, params).fetchall()

        result = {}
        for row in rows:
            result[row["model"]] = {
                "total_cost": Decimal(str(row["total_cost"])),
                "total_input_tokens": row["total_input"],
                "total_output_tokens": row["total_output"],
                "record_count": row["record_count"],
                "first_use": row["first_use"],
                "last_use": row["last_use"],
            }

        return result

    def get_daily_costs(
        self,
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Get daily cost totals.

        Args:
            days: Number of days to look back

        Returns:
            List of daily cost records
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    date(timestamp) as day,
                    SUM(cost) as total_cost,
                    SUM(input_tokens) as total_input,
                    SUM(output_tokens) as total_output,
                    COUNT(*) as record_count
                FROM token_usage
                WHERE timestamp >= date('now', ?)
                GROUP BY date(timestamp)
                ORDER BY day DESC
                """,
                (f"-{days} days",),
            ).fetchall()

        return [
            {
                "date": row["day"],
                "total_cost": Decimal(str(row["total_cost"])),
                "total_input_tokens": row["total_input"],
                "total_output_tokens": row["total_output"],
                "record_count": row["record_count"],
            }
            for row in rows
        ]

    def get_session_records(self, session_id: str, limit: int = 100) -> list[TokenUsageRecord]:
        """Get individual usage records for a session.

        Args:
            session_id: Session identifier
            limit: Maximum records to return

        Returns:
            List of TokenUsageRecord
        """
        with self.db.connection() as conn:
            rows = conn.execute(
                """
                SELECT id, session_id, model, input_tokens, output_tokens,
                       cost, context, timestamp
                FROM token_usage
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()

        return [
            TokenUsageRecord(
                id=row["id"],
                session_id=row["session_id"],
                model=row["model"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                cost=Decimal(str(row["cost"])),
                timestamp=row["timestamp"],
                context=row["context"],
            )
            for row in rows
        ]

    def purge_old_records(self, days: int = 90) -> int:
        """Delete records older than specified days.

        Args:
            days: Records older than this many days will be deleted

        Returns:
            Number of records deleted
        """
        with self.db.connection() as conn:
            cursor = conn.execute(
                """
                DELETE FROM token_usage
                WHERE timestamp < date('now', ?)
                """,
                (f"-{days} days",),
            )
            return cursor.rowcount
