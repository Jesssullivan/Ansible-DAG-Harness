"""Notification service for HOTL mode alerts."""

import logging
from dataclasses import dataclass
from datetime import datetime

import httpx

logger = logging.getLogger(__name__)


@dataclass
class NotificationConfig:
    """Configuration for notification channels."""

    discord_webhook_url: str | None = None
    email_smtp_host: str | None = None
    email_smtp_port: int = 587
    email_from: str | None = None
    email_to: str | None = None
    email_username: str | None = None
    email_password: str | None = None


class NotificationService:
    """
    Service for sending notifications via Discord and email.

    Used by HOTL mode to send status updates and alerts.
    Supports both async and sync operation.
    """

    def __init__(self, config: NotificationConfig):
        """
        Initialize the notification service.

        Args:
            config: Notification configuration
        """
        self.config = config
        self._async_client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=30.0)
        return self._async_client

    def _get_sync_client(self) -> httpx.Client:
        """Get or create sync HTTP client."""
        if self._sync_client is None:
            self._sync_client = httpx.Client(timeout=30.0)
        return self._sync_client

    async def close(self) -> None:
        """Close the async HTTP client."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None

    def close_sync(self) -> None:
        """Close the sync HTTP client."""
        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

    async def send_discord(
        self,
        title: str,
        description: str,
        color: int = 0x00FF00,  # Green by default
        fields: list[dict] | None = None,
        footer: str | None = None,
    ) -> bool:
        """
        Send a Discord webhook notification.

        Args:
            title: Embed title
            description: Embed description
            color: Embed color (hex)
            fields: Optional list of field dicts with 'name' and 'value'
            footer: Optional footer text

        Returns:
            True if sent successfully
        """
        if not self.config.discord_webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False

        embed = {
            "title": title,
            "description": description[:4096],  # Discord limit
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if fields:
            embed["fields"] = [
                {
                    "name": f["name"][:256],
                    "value": f["value"][:1024],
                    "inline": f.get("inline", False),
                }
                for f in fields[:25]  # Discord limit
            ]

        if footer:
            embed["footer"] = {"text": footer[:2048]}

        payload = {"embeds": [embed]}

        try:
            client = await self._get_client()
            response = await client.post(self.config.discord_webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"Discord notification sent: {title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    async def send_email(
        self, subject: str, body: str, to: str | None = None, html: bool = False
    ) -> bool:
        """
        Send an email notification.

        Requires aiosmtplib to be installed (optional dependency).

        Args:
            subject: Email subject
            body: Email body
            to: Override recipient (uses config default if not provided)
            html: If True, send as HTML email

        Returns:
            True if sent successfully
        """
        try:
            from email.message import EmailMessage

            import aiosmtplib
        except ImportError:
            logger.warning("aiosmtplib not installed. Install with: pip install aiosmtplib")
            return False

        if not self.config.email_smtp_host:
            logger.warning("Email SMTP host not configured")
            return False

        recipient = to or self.config.email_to
        if not recipient:
            logger.warning("Email recipient not configured")
            return False

        message = EmailMessage()
        message["Subject"] = subject
        message["From"] = self.config.email_from or "harness@localhost"
        message["To"] = recipient

        if html:
            message.set_content(body, subtype="html")
        else:
            message.set_content(body)

        try:
            await aiosmtplib.send(
                message,
                hostname=self.config.email_smtp_host,
                port=self.config.email_smtp_port,
                username=self.config.email_username,
                password=self.config.email_password,
                start_tls=True,
            )
            logger.info(f"Email notification sent: {subject}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False

    async def send_status_update(self, state: dict, summary: str) -> dict[str, bool]:
        """
        Send status update to all configured channels.

        Args:
            state: Current HOTL state
            summary: Status summary text

        Returns:
            Dict mapping channel to success status
        """
        results = {}

        # Determine color based on state
        errors = state.get("errors", [])
        warnings = state.get("warnings", [])

        if errors:
            color = 0xFF0000  # Red
        elif warnings:
            color = 0xFFFF00  # Yellow
        else:
            color = 0x00FF00  # Green

        # Prepare fields
        fields = [
            {"name": "Phase", "value": state.get("phase", "unknown"), "inline": True},
            {
                "name": "Iteration",
                "value": f"{state.get('iteration_count', 0)}/{state.get('max_iterations', 0)}",
                "inline": True,
            },
            {
                "name": "Tasks",
                "value": f"Done: {len(state.get('completed_tasks', []))}, Failed: {len(state.get('failed_tasks', []))}",
                "inline": True,
            },
        ]

        if errors:
            fields.append({"name": "Recent Errors", "value": "\n".join(errors[-3:])[:1024]})

        # Send Discord
        if self.config.discord_webhook_url:
            results["discord"] = await self.send_discord(
                title=f"HOTL Status - Iteration {state.get('iteration_count', 0)}",
                description=summary[:4000],
                color=color,
                fields=fields,
                footer="EMS Harness HOTL Mode",
            )

        # Send Email
        if self.config.email_to:
            email_body = f"""
HOTL Status Update
==================

{summary}

Phase: {state.get("phase", "unknown")}
Iteration: {state.get("iteration_count", 0)}/{state.get("max_iterations", 0)}
Completed Tasks: {len(state.get("completed_tasks", []))}
Failed Tasks: {len(state.get("failed_tasks", []))}

Errors:
{chr(10).join(errors[-5:]) if errors else "None"}

Warnings:
{chr(10).join(warnings[-5:]) if warnings else "None"}

---
EMS Harness HOTL Mode
            """.strip()

            results["email"] = await self.send_email(
                subject=f"[HOTL] Status Update - Iteration {state.get('iteration_count', 0)}",
                body=email_body,
            )

        return results

    def send_discord_sync(
        self,
        title: str,
        description: str,
        color: int = 0x00FF00,
        fields: list[dict] | None = None,
        footer: str | None = None,
    ) -> bool:
        """
        Send a Discord webhook notification synchronously.

        Args:
            title: Embed title
            description: Embed description
            color: Embed color (hex)
            fields: Optional list of field dicts with 'name' and 'value'
            footer: Optional footer text

        Returns:
            True if sent successfully
        """
        if not self.config.discord_webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False

        embed = {
            "title": title,
            "description": description[:4096],
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if fields:
            embed["fields"] = [
                {
                    "name": f["name"][:256],
                    "value": f["value"][:1024],
                    "inline": f.get("inline", False),
                }
                for f in fields[:25]
            ]

        if footer:
            embed["footer"] = {"text": footer[:2048]}

        payload = {"embeds": [embed]}

        try:
            client = self._get_sync_client()
            response = client.post(self.config.discord_webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"Discord notification sent: {title}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Discord notification: {e}")
            return False

    def send_status_update_sync(self, state: dict, summary: str) -> dict[str, bool]:
        """
        Send status update to all configured channels synchronously.

        Args:
            state: Current HOTL state
            summary: Status summary text

        Returns:
            Dict mapping channel to success status
        """
        results = {}

        # Determine color based on state
        errors = state.get("errors", [])
        warnings = state.get("warnings", [])

        if errors:
            color = 0xFF0000  # Red
        elif warnings:
            color = 0xFFFF00  # Yellow
        else:
            color = 0x00FF00  # Green

        # Prepare fields
        fields = [
            {"name": "Phase", "value": str(state.get("phase", "unknown")), "inline": True},
            {
                "name": "Iteration",
                "value": f"{state.get('iteration_count', 0)}/{state.get('max_iterations', 0)}",
                "inline": True,
            },
            {
                "name": "Tasks",
                "value": f"Done: {len(state.get('completed_tasks', []))}, Failed: {len(state.get('failed_tasks', []))}",
                "inline": True,
            },
        ]

        if errors:
            fields.append(
                {"name": "Recent Errors", "value": "\n".join(str(e) for e in errors[-3:])[:1024]}
            )

        # Send Discord
        if self.config.discord_webhook_url:
            results["discord"] = self.send_discord_sync(
                title=f"HOTL Status - Iteration {state.get('iteration_count', 0)}",
                description=summary[:4000],
                color=color,
                fields=fields,
                footer="EMS Harness HOTL Mode",
            )

        # Email sending would need smtplib for sync - omit for now
        # as Discord is the primary notification channel

        return results

    async def send_alert(
        self,
        title: str,
        message: str,
        severity: str = "warning",  # "info", "warning", "error", "critical"
    ) -> dict[str, bool]:
        """
        Send an alert to all configured channels.

        Args:
            title: Alert title
            message: Alert message
            severity: Alert severity level

        Returns:
            Dict mapping channel to success status
        """
        results = {}

        color_map = {"info": 0x0099FF, "warning": 0xFFFF00, "error": 0xFF0000, "critical": 0x990000}
        color = color_map.get(severity, 0xFFFFFF)

        # Send Discord
        if self.config.discord_webhook_url:
            results["discord"] = await self.send_discord(
                title=f"[{severity.upper()}] {title}", description=message, color=color
            )

        # Send Email
        if self.config.email_to:
            results["email"] = await self.send_email(
                subject=f"[HOTL {severity.upper()}] {title}", body=message
            )

        return results
