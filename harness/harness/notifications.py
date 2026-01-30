"""
Notification service for workflow events.

Provides:
- Discord webhook notifications
- Email notifications via SMTP
- Retry logic for delivery
- Integration with DAG event system
"""

import asyncio
import json
import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from typing import Any, Optional

import httpx

from harness.config import NotificationConfig

logger = logging.getLogger(__name__)


class NotificationType(str, Enum):
    """Types of notifications."""
    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    NODE_FAILED = "node_failed"
    TEST_REGRESSION = "test_regression"
    WAVE_COMPLETED = "wave_completed"
    MERGE_TRAIN_UPDATED = "merge_train_updated"
    HUMAN_NEEDED = "human_needed"


class NotificationPriority(str, Enum):
    """Priority levels for notifications."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """A notification to be sent."""
    type: NotificationType
    title: str
    message: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    data: Optional[dict] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()


class NotificationService:
    """
    Service for sending notifications via multiple channels.

    Supports:
    - Discord webhooks
    - Email via SMTP
    - Retry logic with exponential backoff
    - Rate limiting to prevent spam
    """

    def __init__(self, config: NotificationConfig):
        self.config = config
        self._last_notification: Optional[datetime] = None
        self._rate_limit_seconds = 5  # Minimum seconds between notifications

    async def send(self, notification: Notification) -> bool:
        """
        Send a notification via all configured channels.

        Args:
            notification: The notification to send

        Returns:
            True if sent successfully via at least one channel
        """
        if not self.config.enabled:
            logger.debug("Notifications disabled")
            return False

        # Rate limiting
        if self._last_notification:
            elapsed = (datetime.utcnow() - self._last_notification).total_seconds()
            if elapsed < self._rate_limit_seconds:
                await asyncio.sleep(self._rate_limit_seconds - elapsed)

        self._last_notification = datetime.utcnow()

        success = False

        # Try Discord
        if self.config.discord_webhook_url:
            try:
                await self._send_discord(notification)
                success = True
            except Exception as e:
                logger.warning(f"Discord notification failed: {e}")

        # Try Email
        if self.config.email_recipient:
            try:
                await self._send_email(notification)
                success = True
            except Exception as e:
                logger.warning(f"Email notification failed: {e}")

        return success

    async def _send_discord(self, notification: Notification, retries: int = 3) -> None:
        """Send notification to Discord webhook with retry logic."""
        # Map priority to Discord colors
        color_map = {
            NotificationPriority.LOW: 0x808080,     # Gray
            NotificationPriority.NORMAL: 0x3498db,  # Blue
            NotificationPriority.HIGH: 0xf39c12,    # Orange
            NotificationPriority.URGENT: 0xe74c3c,  # Red
        }

        # Map notification types to emojis
        emoji_map = {
            NotificationType.WORKFLOW_STARTED: "🚀",
            NotificationType.WORKFLOW_COMPLETED: "✅",
            NotificationType.WORKFLOW_FAILED: "❌",
            NotificationType.NODE_FAILED: "⚠️",
            NotificationType.TEST_REGRESSION: "🔴",
            NotificationType.WAVE_COMPLETED: "🌊",
            NotificationType.MERGE_TRAIN_UPDATED: "🚂",
            NotificationType.HUMAN_NEEDED: "👤",
        }

        emoji = emoji_map.get(notification.type, "📢")

        # Build Discord embed
        embed = {
            "title": f"{emoji} {notification.title}",
            "description": notification.message,
            "color": color_map.get(notification.priority, 0x3498db),
            "timestamp": notification.timestamp.isoformat(),
            "footer": {"text": "EMS Harness"}
        }

        # Add fields from data
        if notification.data:
            fields = []
            for key, value in notification.data.items():
                if isinstance(value, (list, dict)):
                    value = json.dumps(value, indent=2)[:1024]
                fields.append({
                    "name": key.replace("_", " ").title(),
                    "value": str(value)[:1024],
                    "inline": len(str(value)) < 50
                })
            embed["fields"] = fields[:25]  # Discord limit

        payload = {"embeds": [embed]}

        # Send with retry
        for attempt in range(retries):
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        self.config.discord_webhook_url,
                        json=payload,
                        timeout=10.0
                    )
                    response.raise_for_status()
                    return
            except Exception as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    async def _send_email(self, notification: Notification, retries: int = 3) -> None:
        """Send notification via email with retry logic."""
        # Priority to subject prefix mapping
        priority_prefix = {
            NotificationPriority.LOW: "[LOW]",
            NotificationPriority.NORMAL: "",
            NotificationPriority.HIGH: "[HIGH]",
            NotificationPriority.URGENT: "[URGENT]",
        }

        prefix = priority_prefix.get(notification.priority, "")
        subject = f"{prefix} EMS Harness: {notification.title}".strip()

        # Build email body
        body_parts = [
            f"# {notification.title}",
            "",
            notification.message,
            "",
            f"Type: {notification.type.value}",
            f"Priority: {notification.priority.value}",
            f"Timestamp: {notification.timestamp.isoformat()}",
        ]

        if notification.data:
            body_parts.extend(["", "## Details", ""])
            for key, value in notification.data.items():
                body_parts.append(f"**{key.replace('_', ' ').title()}**: {value}")

        body = "\n".join(body_parts)

        # Create email message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.config.email_from or "harness@localhost"
        msg["To"] = self.config.email_recipient

        # Add plain text part
        msg.attach(MIMEText(body, "plain"))

        # Send with retry (using localhost SMTP by default)
        for attempt in range(retries):
            try:
                # Run SMTP in thread pool to not block
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._send_smtp,
                    msg
                )
                return
            except Exception as e:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)

    def _send_smtp(self, msg: MIMEMultipart) -> None:
        """Send email via SMTP (blocking, run in thread pool)."""
        smtp_host = self.config.smtp_host or "localhost"
        smtp_port = self.config.smtp_port or 25

        with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
            if self.config.smtp_username and self.config.smtp_password:
                server.starttls()
                server.login(self.config.smtp_username, self.config.smtp_password)
            server.send_message(msg)


# ============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON NOTIFICATIONS
# ============================================================================

async def notify_workflow_started(
    service: NotificationService,
    role_name: str,
    execution_id: int
) -> bool:
    """Send workflow started notification."""
    return await service.send(Notification(
        type=NotificationType.WORKFLOW_STARTED,
        title=f"Workflow Started: {role_name}",
        message=f"Box-up-role workflow started for `{role_name}`",
        priority=NotificationPriority.LOW,
        data={
            "role_name": role_name,
            "execution_id": execution_id
        }
    ))


async def notify_workflow_completed(
    service: NotificationService,
    role_name: str,
    execution_id: int,
    summary: dict
) -> bool:
    """Send workflow completed notification."""
    return await service.send(Notification(
        type=NotificationType.WORKFLOW_COMPLETED,
        title=f"Workflow Completed: {role_name}",
        message=f"Box-up-role workflow completed successfully for `{role_name}`",
        priority=NotificationPriority.NORMAL,
        data={
            "role_name": role_name,
            "execution_id": execution_id,
            "issue_url": summary.get("issue_url"),
            "mr_url": summary.get("mr_url"),
            "wave": summary.get("wave"),
        }
    ))


async def notify_workflow_failed(
    service: NotificationService,
    role_name: str,
    execution_id: int,
    error: str,
    failed_node: Optional[str] = None
) -> bool:
    """Send workflow failed notification."""
    return await service.send(Notification(
        type=NotificationType.WORKFLOW_FAILED,
        title=f"Workflow Failed: {role_name}",
        message=f"Box-up-role workflow failed for `{role_name}`:\n{error}",
        priority=NotificationPriority.HIGH,
        data={
            "role_name": role_name,
            "execution_id": execution_id,
            "failed_node": failed_node,
            "error": error
        }
    ))


async def notify_test_regression(
    service: NotificationService,
    role_name: str,
    test_name: str,
    consecutive_failures: int,
    error_message: Optional[str] = None
) -> bool:
    """Send test regression notification."""
    return await service.send(Notification(
        type=NotificationType.TEST_REGRESSION,
        title=f"Test Regression: {role_name}",
        message=f"Test `{test_name}` has failed {consecutive_failures} times consecutively",
        priority=NotificationPriority.HIGH if consecutive_failures >= 3 else NotificationPriority.NORMAL,
        data={
            "role_name": role_name,
            "test_name": test_name,
            "consecutive_failures": consecutive_failures,
            "error_message": error_message
        }
    ))


async def notify_wave_completed(
    service: NotificationService,
    wave: int,
    wave_name: str,
    success_count: int,
    failure_count: int
) -> bool:
    """Send wave completion notification."""
    total = success_count + failure_count
    priority = NotificationPriority.NORMAL if failure_count == 0 else NotificationPriority.HIGH

    return await service.send(Notification(
        type=NotificationType.WAVE_COMPLETED,
        title=f"Wave {wave} Completed: {wave_name}",
        message=f"Wave completed: {success_count}/{total} roles succeeded",
        priority=priority,
        data={
            "wave": wave,
            "wave_name": wave_name,
            "success_count": success_count,
            "failure_count": failure_count,
            "total_roles": total
        }
    ))


async def notify_human_needed(
    service: NotificationService,
    role_name: str,
    execution_id: int,
    reason: str,
    node_name: str
) -> bool:
    """Send human-in-the-loop notification."""
    return await service.send(Notification(
        type=NotificationType.HUMAN_NEEDED,
        title=f"Human Input Required: {role_name}",
        message=f"Workflow paused at `{node_name}` - human input needed:\n{reason}",
        priority=NotificationPriority.URGENT,
        data={
            "role_name": role_name,
            "execution_id": execution_id,
            "node_name": node_name,
            "reason": reason
        }
    ))
