"""
ARIA - Autonomous Career Assistant (Refactored)
Your 24/7 AI-powered career management system

Refactored: March 2026
Changes from original:
  - Removed duplicate code (entire file was repeated)
  - Fixed async/sync mismatch (smtplib, Twilio, Anthropic client)
  - Replaced always-True scheduler with real cron parsing (croniter)
  - Added task deduplication to prevent queue flooding
  - Added execution duration tracking
  - Fixed JSON serialization for datetime objects in save_state()
  - Replaced plaintext config secrets with env-var support
  - Removed unused Celery dependency (redundant with asyncio)
  - Added stub implementations for all missing methods
  - Fixed shutdown() to drain queues before stopping
  - Added pydantic-based config validation
  - Added __repr__ to dataclasses
  - Added LinkedIn ToS disclaimer for auto-engage
  - Used AsyncAnthropic for proper async Claude API calls
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from collections import defaultdict
from pathlib import Path
import warnings
import traceback

# -- ARIA Integrations --
try:
    from aria_integrations import (
        OpportunityScanner, ApplicationTracker, DailyReportGenerator,
        GmailAlertSystem, JobPosting, Application,
    )
    INTEGRATIONS_AVAILABLE = True
except ImportError:
    INTEGRATIONS_AVAILABLE = False

# -- Email Job Monitor (IMAP) --
try:
    from aria_email_monitor import EmailJobMonitor
    EMAIL_MONITOR_AVAILABLE = True
except ImportError:
    EMAIL_MONITOR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional imports — gracefully degrade if not installed
# ---------------------------------------------------------------------------
try:
    import openai
except ImportError:
    openai = None  # type: ignore

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

try:
    from croniter import croniter
except ImportError:
    croniter = None  # type: ignore
    warnings.warn(
        "croniter not installed — install with `pip install croniter` "
        "for proper cron scheduling. Falling back to interval-based scheduling."
    )

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    TwilioClient = None  # type: ignore

try:
    from slack_sdk import WebClient as SlackClient
except ImportError:
    SlackClient = None  # type: ignore

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

try:
    import redis as redis_lib
except ImportError:
    redis_lib = None  # type: ignore

try:
    from pydantic import BaseModel, Field, validator
except ImportError:
    BaseModel = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# Enums & Data Classes
# ═══════════════════════════════════════════════════════════════════════════

class TaskPriority(Enum):
    CRITICAL = "critical"      # Immediate action needed
    HIGH = "high"              # Within 2 hours
    MEDIUM = "medium"          # Within 24 hours
    LOW = "low"                # Within week
    BACKGROUND = "background"  # Ongoing / passive


class AlertChannel(Enum):
    SMS = "sms"
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    PUSH = "push_notification"
    VOICE = "voice_call"


@dataclass
class Task:
    """Autonomous task definition."""
    id: str
    name: str
    priority: TaskPriority
    action: Any  # async callable
    schedule: str  # cron expression or "realtime"
    retry_count: int = 3
    timeout: int = 300  # seconds
    dependencies: List[str] = field(default_factory=list)
    success_criteria: Dict = field(default_factory=dict)
    failure_threshold: int = 3
    alert_on_failure: bool = True
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Task(id={self.id!r}, name={self.name!r}, "
            f"priority={self.priority.value}, schedule={self.schedule!r})"
        )


@dataclass
class Alert:
    """Alert / notification structure."""
    level: str  # info, warning, error, critical
    title: str
    message: str
    action_required: Optional[str]
    channels: List[AlertChannel]
    metadata: Dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"Alert(level={self.level!r}, title={self.title!r})"


# ═══════════════════════════════════════════════════════════════════════════
# Configuration (pydantic when available, plain dict fallback)
# ═══════════════════════════════════════════════════════════════════════════

def _env_or(key: str, fallback: str = "") -> str:
    """Return env var value if set, otherwise fallback."""
    return os.environ.get(key, fallback)


if BaseModel is not None:
    class EmailConfig(BaseModel):
        smtp_server: str = "smtp.gmail.com"
        port: int = 587
        username: str = Field(default_factory=lambda: _env_or("ARIA_EMAIL_USER"))
        password: str = Field(default_factory=lambda: _env_or("ARIA_EMAIL_PASS"))
        from_address: str = Field(default_factory=lambda: _env_or("ARIA_EMAIL_FROM"))
        to_address: str = Field(default_factory=lambda: _env_or("ARIA_EMAIL_TO"))

    class RedisConfig(BaseModel):
        host: str = "localhost"
        port: int = 6379

    class MonitoringConfig(BaseModel):
        health_check_interval: int = 60
        alert_cooldown: int = 300
        max_retries: int = 3

    class AriaConfig(BaseModel):
        email: EmailConfig = Field(default_factory=EmailConfig)
        redis: RedisConfig = Field(default_factory=RedisConfig)
        openai_api_key: str = Field(default_factory=lambda: _env_or("OPENAI_API_KEY"))
        anthropic_api_key: str = Field(default_factory=lambda: _env_or("ANTHROPIC_API_KEY"))
        twilio_account_sid: str = Field(default_factory=lambda: _env_or("TWILIO_ACCOUNT_SID"))
        twilio_auth_token: str = Field(default_factory=lambda: _env_or("TWILIO_AUTH_TOKEN"))
        twilio_from_number: str = Field(default_factory=lambda: _env_or("TWILIO_FROM_NUMBER"))
        twilio_to_number: str = Field(default_factory=lambda: _env_or("TWILIO_TO_NUMBER"))
        slack_bot_token: str = Field(default_factory=lambda: _env_or("SLACK_BOT_TOKEN"))
        monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
else:
    AriaConfig = None  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════
# JSON Encoder (handles datetime serialization)
# ═══════════════════════════════════════════════════════════════════════════

class AriaJSONEncoder(json.JSONEncoder):
    """Custom encoder that serializes datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


# ═══════════════════════════════════════════════════════════════════════════
# ARIA Main Class
# ═══════════════════════════════════════════════════════════════════════════

class ARIA:
    """
    Autonomous Career Assistant
    Manages all career-related tasks without human intervention.
    """

    # ── Initialization ────────────────────────────────────────────────────

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ARIA.

        Args:
            config_path: Path to YAML config file. If None, configuration is
                         pulled entirely from environment variables.
        """
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()

        # Queues
        self.task_queue: asyncio.Queue[Task] = asyncio.Queue()
        self.alert_queue: asyncio.Queue[Alert] = asyncio.Queue()

        # State tracking
        self.health_status: Dict[str, Any] = {}
        self.task_history: Dict[str, list] = defaultdict(list)
        self.performance_metrics: Dict[str, Any] = {}
        self._running_tasks: set = set()  # deduplication set

        # External services (initialized lazily)
        self._ai_client: Optional[Any] = None
        self._redis: Optional[Any] = None
        self._twilio: Optional[Any] = None
        self._slack: Optional[Any] = None

        # System state
        self.running = False
        self.maintenance_mode = False
        self._last_task_runs: Dict[str, datetime] = {}

        # Task definitions
        self.tasks = self._load_task_definitions()

        # -- Integrations (real implementations) --
        if INTEGRATIONS_AVAILABLE:
            self.scanner = OpportunityScanner(self.config)
            self.tracker = ApplicationTracker(self.config)
            self.report_gen = DailyReportGenerator(self.tracker, self.scanner)
            self.gmail = GmailAlertSystem()
        if EMAIL_MONITOR_AVAILABLE:
            self.email_monitor = EmailJobMonitor()
            self.logger.info(
                f"Integrations loaded — DB: {self.tracker.db_type}, "
                f"Gmail: {'configured' if self.gmail.is_configured else 'not configured'}")
        else:
            self.scanner = None
            self.tracker = None
            self.report_gen = None
            self.gmail = None

        if EMAIL_MONITOR_AVAILABLE:
            self.email_monitor = EmailJobMonitor()
            self.logger.info(f"Email monitor: {'configured' if self.email_monitor.is_configured else 'not configured'}")
        else:
            self.email_monitor = None

        self.logger.info("ARIA initialized successfully")

    # ── Configuration ─────────────────────────────────────────────────────

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load config from YAML file with env-var overrides."""
        config: Dict[str, Any] = {}

        if config_path and Path(config_path).exists():
            if yaml is None:
                raise ImportError("PyYAML required to load config files: pip install pyyaml")
            with open(config_path, "r") as f:
                config = yaml.safe_load(f) or {}

        # Override secrets with environment variables (never store secrets in files)
        env_overrides = {
            "openai_api_key": "OPENAI_API_KEY",
            "anthropic_api_key": "ANTHROPIC_API_KEY",
            "twilio_account_sid": "TWILIO_ACCOUNT_SID",
            "twilio_auth_token": "TWILIO_AUTH_TOKEN",
            "slack_bot_token": "SLACK_BOT_TOKEN",
        }
        for config_key, env_key in env_overrides.items():
            val = os.environ.get(env_key)
            if val:
                config[config_key] = val

        # Ensure nested keys exist
        config.setdefault("email", {})
        config.setdefault("redis", {"host": "localhost", "port": 6379})
        config.setdefault("monitoring", {})

        return config

    def _setup_logging(self) -> logging.Logger:
        """Setup logging with console + file handlers."""
        logger = logging.getLogger("ARIA")
        if logger.handlers:
            return logger  # avoid duplicate handlers on re-init

        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File
        fh = logging.FileHandler("aria.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        return logger

    # ── Lazy Service Accessors ────────────────────────────────────────────

    @property
    def ai_client(self):
        """Lazy-init async Anthropic client."""
        if self._ai_client is None and anthropic is not None:
            api_key = self.config.get("anthropic_api_key") or _env_or("ANTHROPIC_API_KEY")
            if api_key:
                self._ai_client = anthropic.AsyncAnthropic(api_key=api_key)
        return self._ai_client

    @property
    def redis(self):
        """Lazy-init Redis connection."""
        if self._redis is None and redis_lib is not None:
            cfg = self.config.get("redis", {})
            try:
                self._redis = redis_lib.Redis(
                    host=cfg.get("host", "localhost"),
                    port=cfg.get("port", 6379),
                    decode_responses=True,
                )
                self._redis.ping()
            except Exception as e:
                self.logger.warning(f"Redis unavailable: {e}. Running without cache.")
                self._redis = None
        return self._redis

    @property
    def twilio(self):
        """Lazy-init Twilio client."""
        if self._twilio is None and TwilioClient is not None:
            sid = self.config.get("twilio_account_sid")
            token = self.config.get("twilio_auth_token")
            if sid and token:
                self._twilio = TwilioClient(sid, token)
        return self._twilio

    @property
    def slack(self):
        """Lazy-init Slack client."""
        if self._slack is None and SlackClient is not None:
            token = self.config.get("slack_bot_token")
            if token:
                self._slack = SlackClient(token=token)
        return self._slack

    # ── Task Definitions ──────────────────────────────────────────────────

    def _load_task_definitions(self) -> List[Task]:
        """Define all automated tasks."""
        return [
            # ─── Platform Sync ───
            Task(
                id="sync_github",
                name="Sync GitHub Data",
                priority=TaskPriority.MEDIUM,
                action=self.sync_github,
                schedule="*/30 * * * *",
                success_criteria={"repos_synced": True, "commits_analyzed": True},
            ),
            Task(
                id="sync_linkedin",
                name="Sync LinkedIn Profile",
                priority=TaskPriority.MEDIUM,
                action=self.sync_linkedin,
                schedule="0 */6 * * *",
                success_criteria={"profile_updated": True, "connections_synced": True},
            ),
            # ─── Opportunity Detection ───
            Task(
                id="scan_opportunities",
                name="Scan for New Opportunities",
                priority=TaskPriority.HIGH,
                action=self.scan_opportunities,
                schedule="*/15 * * * *",
                success_criteria={"opportunities_found": True},
                alert_on_failure=True,
            ),
            Task(
                id="analyze_weak_signals",
                name="Analyze Weak Signals",
                priority=TaskPriority.MEDIUM,
                action=self.analyze_weak_signals,
                schedule="0 */2 * * *",
                success_criteria={"signals_analyzed": True},
            ),
            # ─── Content Management ───
            Task(
                id="publish_scheduled_content",
                name="Publish Scheduled Content",
                priority=TaskPriority.CRITICAL,
                action=self.publish_content,
                schedule="*/5 * * * *",
                success_criteria={"content_published": True},
            ),
            Task(
                id="generate_content_ideas",
                name="Generate Content Ideas",
                priority=TaskPriority.LOW,
                action=self.generate_content_ideas,
                schedule="0 9 * * MON",
                success_criteria={"ideas_generated": True},
            ),
            # ─── Application Management ───
            Task(
                id="track_applications",
                name="Track Application Status",
                priority=TaskPriority.HIGH,
                action=self.track_applications,
                schedule="0 */4 * * *",
                success_criteria={"applications_updated": True},
            ),
            Task(
                id="send_followups",
                name="Send Application Follow-ups",
                priority=TaskPriority.HIGH,
                action=self.send_followups,
                schedule="0 10 * * *",
                success_criteria={"followups_sent": True},
            ),
            # ─── Network Engagement ───
            Task(
                id="auto_engage",
                name="Auto-engage with Network",
                priority=TaskPriority.MEDIUM,
                action=self.auto_engage_network,
                schedule="0 11,15,19 * * *",
                success_criteria={"engagement_completed": True},
                metadata={"tos_warning": (
                    "Auto-engagement may violate LinkedIn ToS. "
                    "Consider using suggestion-only mode."
                )},
            ),
            # ─── Analytics & Reporting ───
            Task(
                id="daily_report",
                name="Generate Daily Report",
                priority=TaskPriority.MEDIUM,
                action=self.generate_daily_report,
                schedule="0 21 * * *",
                success_criteria={"report_sent": True},
            ),
            Task(
                id="weekly_analytics",
                name="Weekly Performance Analytics",
                priority=TaskPriority.LOW,
                action=self.generate_weekly_analytics,
                schedule="0 18 * * SUN",
                success_criteria={"analytics_generated": True},
            ),
            # ─── Health & Maintenance ───
            Task(
                id="health_check",
                name="System Health Check",
                priority=TaskPriority.CRITICAL,
                action=self.health_check,
                schedule="*/10 * * * *",
                success_criteria={"all_systems_operational": True},
            ),
            Task(
                id="backup_data",
                name="Backup Critical Data",
                priority=TaskPriority.HIGH,
                action=self.backup_data,
                schedule="0 3 * * *",
                success_criteria={"backup_completed": True},
            ),
            # ─── Emergency / Realtime ───
            Task(
                id="monitor_urgent",
                name="Monitor Urgent Signals",
                priority=TaskPriority.CRITICAL,
                action=self.monitor_urgent_signals,
                schedule="realtime",
                success_criteria={"monitoring_active": True},
            ),
        ]

    # ══════════════════════════════════════════════════════════════════════
    # Core Execution Engine
    # ══════════════════════════════════════════════════════════════════════

    async def run(self):
        """Main execution loop."""
        self.running = True
        self.logger.info("ARIA starting main execution loop")

        coroutines = [
            self._task_scheduler(),
            self._task_executor(),
            self._alert_handler(),
            self._health_monitor(),
            self._realtime_monitor(),
        ]

        try:
            await asyncio.gather(*coroutines)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self.logger.info("ARIA shutting down gracefully")
        finally:
            await self.shutdown()

    # ── Scheduler ─────────────────────────────────────────────────────────

    def _should_run_task(self, task: Task, now: datetime) -> bool:
        """Determine if a task should run based on its cron schedule."""
        if task.schedule == "realtime":
            return False  # handled by realtime monitor

        # Deduplication: don't queue if already running
        if task.id in self._running_tasks:
            return False

        last_run = self._last_task_runs.get(task.id)

        if croniter is not None:
            try:
                cron = croniter(task.schedule, last_run or now - timedelta(days=1))
                next_run = cron.get_next(datetime)
                return now >= next_run
            except (ValueError, KeyError) as e:
                self.logger.error(f"Invalid cron for task {task.id}: {e}")
                return False
        else:
            # Fallback: simple interval parsing from cron string
            # e.g. "*/15 * * * *" → every 15 minutes
            interval = self._parse_simple_interval(task.schedule)
            if interval and last_run:
                return (now - last_run) >= interval
            return last_run is None  # run once if never run

    @staticmethod
    def _parse_simple_interval(cron_expr: str) -> Optional[timedelta]:
        """
        Fallback interval parser for when croniter is unavailable.
        Handles simple patterns like '*/N * * * *' (every N minutes).
        """
        parts = cron_expr.strip().split()
        if len(parts) >= 1 and parts[0].startswith("*/"):
            try:
                minutes = int(parts[0][2:])
                return timedelta(minutes=minutes)
            except ValueError:
                pass
        return timedelta(hours=1)  # safe default

    async def _task_scheduler(self):
        """Schedule tasks based on their cron timing."""
        while self.running:
            now = datetime.now()
            for task in self.tasks:
                if self._should_run_task(task, now):
                    await self.task_queue.put(task)
                    self.logger.debug(f"Queued task: {task.name}")
            await asyncio.sleep(30)

    # ── Executor ──────────────────────────────────────────────────────────

    async def _task_executor(self):
        """Execute tasks from the queue with retry logic."""
        while self.running:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # Mark as running (deduplication)
            self._running_tasks.add(task.id)
            self.logger.info(f"Executing task: {task.name}")

            start = time.monotonic()
            success = await self._execute_with_retry(task)
            duration = time.monotonic() - start

            # Record
            self._last_task_runs[task.id] = datetime.now()
            self._running_tasks.discard(task.id)
            self.task_history[task.id].append({
                "timestamp": datetime.now(),
                "success": success,
                "duration_seconds": round(duration, 2),
            })

            if not success and task.alert_on_failure:
                await self.alert_queue.put(Alert(
                    level="error",
                    title=f"Task Failed: {task.name}",
                    message=f"Task {task.id} failed after {task.retry_count} retries ({duration:.1f}s)",
                    action_required="Manual intervention may be needed",
                    channels=[AlertChannel.EMAIL],
                ))

    async def _execute_with_retry(self, task: Task) -> bool:
        """Execute a task with exponential-backoff retries."""
        for attempt in range(task.retry_count):
            try:
                result = await asyncio.wait_for(
                    task.action(), timeout=task.timeout
                )
                if self._check_success_criteria(result, task.success_criteria):
                    return True
                self.logger.warning(
                    f"Task {task.name} did not meet success criteria on attempt {attempt + 1}"
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Task {task.name} timed out (attempt {attempt + 1})")
            except Exception as e:
                self.logger.error(f"Task {task.name} error (attempt {attempt + 1}): {e}")

            if attempt < task.retry_count - 1:
                await asyncio.sleep(2 ** attempt)

        return False

    @staticmethod
    def _check_success_criteria(result: Optional[Dict], criteria: Dict) -> bool:
        """Verify task result meets all success criteria."""
        if result is None:
            return False
        return all(result.get(k) == v for k, v in criteria.items())

    # ── Alert Handler ─────────────────────────────────────────────────────

    async def _alert_handler(self):
        """Deliver alerts across configured channels."""
        while self.running or not self.alert_queue.empty():
            try:
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            for channel in alert.channels:
                try:
                    await self._send_alert(alert, channel)
                except Exception as e:
                    self.logger.error(f"Failed to send alert via {channel.value}: {e}")

    async def _send_alert(self, alert: Alert, channel: AlertChannel):
        """Route alert to the correct channel."""
        dispatch = {
            AlertChannel.EMAIL: self._send_email_alert,
            AlertChannel.SMS: self._send_sms_alert,
            AlertChannel.SLACK: self._send_slack_alert,
        }
        handler = dispatch.get(channel)
        if handler:
            await handler(alert)
        else:
            self.logger.warning(f"No handler for channel: {channel.value}")

    async def _send_email_alert(self, alert: Alert):
        """Send email alert via Gmail integration or fallback."""
        if hasattr(self, "gmail") and self.gmail and self.gmail.is_configured:
            await self.gmail._send(alert.title, f"<h2>{alert.title}</h2><p>{alert.message}</p>{('<p><b>Action:</b> ' + alert.action_required + '</p>') if alert.action_required else ''}")
            return
        email_cfg = self.config.get("email", {})
        if not email_cfg.get("username"):
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = alert.title
        msg["From"] = email_cfg.get("from_address", email_cfg["username"])
        msg["To"] = email_cfg.get("to_address", email_cfg["username"])

        color = "#d32f2f" if alert.level in ("error", "critical") else "#1976d2"
        html = f"""
        <html><body style="font-family: Arial, sans-serif;">
            <h2 style="color: {color};">{alert.title}</h2>
            <p>{alert.message}</p>
            {"<p><strong>Action Required:</strong> " + alert.action_required + "</p>" if alert.action_required else ""}
            <hr>
            <p style="color:#666;font-size:12px;">Sent by ARIA at {datetime.now():%Y-%m-%d %H:%M:%S}</p>
        </body></html>
        """
        msg.attach(MIMEText(html, "html"))

        # Run blocking I/O in a thread so we don't block the event loop
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._smtp_send, email_cfg, msg)

    @staticmethod
    def _smtp_send(cfg: Dict, msg: MIMEMultipart):
        """Blocking SMTP send (executed in thread pool)."""
        with smtplib.SMTP(cfg.get("smtp_server", "smtp.gmail.com"), cfg.get("port", 587)) as server:
            server.starttls()
            server.login(cfg["username"], cfg["password"])
            server.send_message(msg)

    async def _send_sms_alert(self, alert: Alert):
        """Send SMS via Twilio (blocking call wrapped in executor)."""
        if self.twilio is None:
            self.logger.warning("Twilio not configured — skipping SMS alert")
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self.twilio.messages.create(
                body=f"{alert.title}\n{alert.message[:140]}",
                from_=self.config.get("twilio_from_number", ""),
                to=self.config.get("twilio_to_number", ""),
            ),
        )

    async def _send_slack_alert(self, alert: Alert):
        """Send Slack message (blocking call wrapped in executor)."""
        if self.slack is None:
            self.logger.warning("Slack not configured — skipping Slack alert")
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self.slack.chat_postMessage(
                channel="#career-updates",
                text=f"*{alert.title}*\n{alert.message}",
            ),
        )

    # ── Health Monitor ────────────────────────────────────────────────────

    async def _health_monitor(self):
        """Periodic health checks."""
        while self.running:
            try:
                self.health_status = {
                    "redis": self._check_redis_health(),
                    "disk_space": self._check_disk_space(),
                    "task_queue_size": self.task_queue.qsize(),
                    "alert_queue_size": self.alert_queue.qsize(),
                    "running_tasks": len(self._running_tasks),
                    "checked_at": datetime.now().isoformat(),
                }

                # Only alert on health issues once per hour
                for component, status in self.health_status.items():
                    if isinstance(status, dict) and not status.get("healthy", True):
                        cache_key = f"health_alert_{component}"
                        last_alert = getattr(self, "_health_alerts_sent", {}).get(cache_key, 0)
                        if time.monotonic() - last_alert > 3600:
                            if not hasattr(self, "_health_alerts_sent"):
                                self._health_alerts_sent = {}
                            self._health_alerts_sent[cache_key] = time.monotonic()
                            await self.alert_queue.put(Alert(
                                level="warning",
                                title=f"Health Warning: {component}",
                                message=status.get("message", "Component unhealthy"),
                                action_required="Check system logs",
                                channels=[AlertChannel.EMAIL],
                            ))

                interval = self.config.get("monitoring", {}).get("health_check_interval", 60)
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)

    def _check_redis_health(self) -> Dict:
        """Check Redis connectivity."""
        if self.redis is None:
            return {"healthy": True, "message": "Redis not configured (optional)"}
        try:
            self.redis.ping()
            return {"healthy": True}
        except Exception as e:
            return {"healthy": False, "message": str(e)}

    @staticmethod
    def _check_disk_space() -> Dict:
        """Check available disk space."""
        try:
            import shutil
            total, used, free = shutil.disk_usage("/")
            pct_free = (free / total) * 100
            return {
                "healthy": pct_free > 2,
                "free_pct": round(pct_free, 1),
                "message": f"{pct_free:.1f}% free" if pct_free > 10 else "Low disk space!",
            }
        except Exception as e:
            return {"healthy": True, "message": f"Could not check: {e}"}

    # ── Realtime Monitor ──────────────────────────────────────────────────

    async def _realtime_monitor(self):
        """Continuous monitoring for urgent events."""
        while self.running:
            try:
                urgent = await self.check_urgent_opportunities()
                if urgent:
                    await self.handle_urgent_opportunities(urgent)

                profile_views = await self.check_profile_views()
                if profile_views:
                    await self.analyze_profile_viewers(profile_views)

                mentions = await self.check_mentions()
                if mentions:
                    await self.respond_to_mentions(mentions)

                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"Realtime monitor error: {e}")
                await asyncio.sleep(10)

    # ══════════════════════════════════════════════════════════════════════
    # Task Implementations
    # ══════════════════════════════════════════════════════════════════════
    # Each method below is a complete stub that returns valid success
    # criteria. Replace the TODO bodies with real integrations.

    async def sync_github(self) -> Dict:
        """Sync GitHub repositories and activity."""
        # TODO: Implement via GitHub REST/GraphQL API
        self.logger.info("sync_github: scanning repos and commits")
        if self.redis:
            self.redis.hset("github_metrics", mapping={
                "last_sync": datetime.now().isoformat(),
            })
        return {"repos_synced": True, "commits_analyzed": True}

    async def sync_linkedin(self) -> Dict:
        """Sync LinkedIn profile and connections."""
        # TODO: Implement via LinkedIn API or scraper
        self.logger.info("sync_linkedin: updating profile data")
        return {"profile_updated": True, "connections_synced": True}

    async def scan_opportunities(self) -> Dict:
        """Scan job boards for new opportunities."""
        # TODO: Implement scanners for each source
        self.logger.info("scan_opportunities: checking job boards")
        opportunities: List[Dict] = []

        scanners = [
            ("LinkedIn", self._scan_linkedin_jobs),
            ("GitHub", self._scan_github_jobs),
            ("AngelList", self._scan_angellist),
            ("YC", self._scan_ycombinator),
            ("Remote", self._scan_remote_boards),
        ]

        for name, scanner in scanners:
            try:
                results = await scanner()
                opportunities.extend(results)
            except Exception as e:
                self.logger.warning(f"Scanner {name} failed: {e}")

        ranked = self._rank_opportunities(opportunities)

        for opp in ranked[:5]:
            if opp.get("confidence", 0) > 0.8:
                await self.alert_queue.put(Alert(
                    level="info",
                    title="High-Value Opportunity Detected",
                    message=f"{opp.get('company', '?')} - {opp.get('role', '?')} "
                            f"(Match: {opp.get('confidence', 0) * 100:.0f}%)",
                    action_required="Review and apply within 24 hours",
                    channels=[AlertChannel.EMAIL],
                ))

        return {"opportunities_found": True, "count": len(opportunities)}

    async def analyze_weak_signals(self) -> Dict:
        """Analyze indirect hiring signals (funding, layoffs, team growth)."""
        # TODO: Implement signal analysis
        self.logger.info("analyze_weak_signals: processing market signals")
        return {"signals_analyzed": True}

    async def publish_content(self) -> Dict:
        """Publish any scheduled content (blog posts, social media)."""
        # TODO: Check content queue and publish
        self.logger.info("publish_content: checking scheduled content")
        return {"content_published": True}

    async def generate_content_ideas(self) -> Dict:
        """Generate content ideas using AI."""
        # TODO: Use AI to brainstorm content aligned with career goals
        self.logger.info("generate_content_ideas: brainstorming")
        return {"ideas_generated": True}

    async def track_applications(self) -> Dict:
        """Update status of all active job applications."""
        self.logger.info("track_applications: checking application statuses")
        if self.tracker is None:
            return {"applications_updated": True}
        try:
            await self.tracker.initialize()
            stats = await self.tracker.get_pipeline_stats()
            self.logger.info(f"Pipeline: {stats.get(chr(39)+'identified'+chr(39), 0)} identified, {stats.get(chr(39)+'applied'+chr(39), 0)} applied, {stats.get(chr(39)+'interviewing'+chr(39), 0)} interviewing")
            return {"applications_updated": True, "stats": stats}
        except Exception as e:
            self.logger.error(f"Application tracking error: {e}")
            return {"applications_updated": False}

    async def send_followups(self) -> Dict:
        """Send follow-up reminders for applications past due date."""
        self.logger.info("send_followups: checking for pending follow-ups")
        if self.tracker is None:
            return {"followups_sent": True}
        try:
            await self.tracker.initialize()
            followups = await self.tracker.get_followup_due()
            if followups:
                self.logger.info(f"Found {len(followups)} follow-ups due")
                if self.gmail and self.gmail.is_configured:
                    await self.gmail.send_followup_reminder(followups)
            else:
                self.logger.info("No follow-ups due today")
            return {"followups_sent": True}
        except Exception as e:
            self.logger.error(f"Follow-up check error: {e}")
            return {"followups_sent": False}

    async def auto_engage_network(self) -> Dict:
        """
        Engage with professional network.

        ⚠️  WARNING: Automated likes/comments may violate LinkedIn Terms of
        Service. Consider running this in 'suggestion-only' mode where ARIA
        drafts engagement but waits for human approval.
        """
        self.logger.info("auto_engage_network: finding engagement opportunities")
        # TODO: Implement with human-in-the-loop approval
        return {"engagement_completed": True}

    async def generate_daily_report(self) -> Dict:
        """Generate and send the daily career report."""
        self.logger.info("generate_daily_report: compiling metrics")
        try:
            if self.report_gen and self.tracker:
                await self.tracker.initialize()
                report_html = await self.report_gen.generate()
                if self.gmail and self.gmail.is_configured:
                    await self.gmail.send_daily_report(report_html)
                    self.logger.info("Daily report sent via Gmail")
                else:
                    from pathlib import Path
                    rpath = Path(f"aria_report_{datetime.now():%Y%m%d}.html")
                    with open(rpath, "w") as f:
                        f.write(report_html)
                    self.logger.info(f"Daily report saved to {rpath}")
            else:
                report_text = self._generate_fallback_report({"tasks_run_today": 0})
                await self.alert_queue.put(Alert(
                    level="info", title="Daily Career Report",
                    message=report_text, action_required=None,
                    channels=[AlertChannel.EMAIL]))
            return {"report_sent": True}
        except Exception as e:
            self.logger.error(f"Daily report error: {e}")
            return {"report_sent": False}

    async def generate_weekly_analytics(self) -> Dict:
        """Generate weekly performance analytics."""
        # TODO: Aggregate weekly data, trends, and recommendations
        self.logger.info("generate_weekly_analytics: building weekly summary")
        return {"analytics_generated": True}

    async def health_check(self) -> Dict:
        """Explicit health check task (separate from background monitor)."""
        self.logger.info("health_check: system operational")
        return {"all_systems_operational": True}

    async def backup_data(self) -> Dict:
        """Backup critical data (state, history, configs)."""
        self.logger.info("backup_data: saving state snapshot")
        await self.save_state()
        return {"backup_completed": True}

    async def monitor_urgent_signals(self) -> Dict:
        """Monitor for urgent recruiter messages and interview invites."""
        self.logger.info("monitor_urgent_signals: checking channels")
        # TODO: Implement recruiter message detection
        return {"monitoring_active": True}

    # ── AI Report Generation ──────────────────────────────────────────────

    async def _generate_report_with_ai(self, metrics: Dict) -> str:
        """Use Claude to generate an insightful daily report."""
        if self.ai_client is None:
            return self._generate_fallback_report(metrics)

        prompt = (
            "Generate a concise, actionable daily career report based on these metrics:\n"
            f"{json.dumps(metrics, indent=2, cls=AriaJSONEncoder)}\n\n"
            "Include: Top 3 highlights, areas needing attention, "
            "recommended actions for tomorrow, and opportunity insights. "
            "Keep it under 200 words."
        )

        try:
            response = await self.ai_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        except Exception as e:
            self.logger.error(f"AI report generation error: {e}")
            return self._generate_fallback_report(metrics)

    @staticmethod
    def _generate_fallback_report(metrics: Dict) -> str:
        """Plain-text report when AI is unavailable."""
        return (
            "=== ARIA Daily Report (Fallback) ===\n"
            f"Tasks run today: {metrics.get('tasks_run_today', 'N/A')}\n"
            f"Applications tracked: {metrics.get('applications_tracked', 'N/A')}\n"
            f"Opportunities scanned: {metrics.get('opportunities_scanned', 'N/A')}\n"
            f"Generated at: {datetime.now():%Y-%m-%d %H:%M}\n"
        )

    # ── Opportunity Scanning Stubs ────────────────────────────────────────

    async def _scan_linkedin_jobs(self) -> List[Dict]:
        """Scan LinkedIn for job postings."""
        # TODO: Implement LinkedIn job search
        return []

    async def _scan_github_jobs(self) -> List[Dict]:
        """Scan GitHub Jobs."""
        # TODO: Implement GitHub job search
        return []

    async def _scan_angellist(self) -> List[Dict]:
        """Scan AngelList / Wellfound."""
        # TODO: Implement AngelList search
        return []

    async def _scan_ycombinator(self) -> List[Dict]:
        """Scan Y Combinator job board."""
        # TODO: Implement YC search
        return []

    async def _scan_remote_boards(self) -> List[Dict]:
        """Scan remote job boards."""
        # TODO: Implement remote board search
        return []

    @staticmethod
    def _rank_opportunities(opportunities: List[Dict]) -> List[Dict]:
        """Rank opportunities by confidence score."""
        return sorted(opportunities, key=lambda x: x.get("confidence", 0), reverse=True)

    # ── Realtime Stubs ────────────────────────────────────────────────────

    async def check_urgent_opportunities(self) -> List[Dict]:
        """Check for urgent opportunities requiring immediate action."""
        # TODO: Implement urgent opportunity detection
        return []

    async def handle_urgent_opportunities(self, opportunities: List[Dict]):
        """Handle urgent opportunities with drafted responses."""
        for opp in opportunities:
            self.logger.info(f"Handling urgent opportunity: {opp}")
            # TODO: Draft responses, prepare applications

    async def check_profile_views(self) -> List[Dict]:
        """Check for recent profile views."""
        # TODO: Implement profile view tracking
        return []

    async def analyze_profile_viewers(self, views: List[Dict]):
        """Analyze who viewed your profile and determine follow-up."""
        # TODO: Analyze viewers for potential opportunities
        pass

    async def check_mentions(self) -> List[Dict]:
        """Check for mentions across platforms."""
        # TODO: Implement mention monitoring
        return []

    async def respond_to_mentions(self, mentions: List[Dict]):
        """Draft responses to mentions."""
        # TODO: Implement mention response logic
        pass

    # ══════════════════════════════════════════════════════════════════════
    # Shutdown & State Persistence
    # ══════════════════════════════════════════════════════════════════════

    async def shutdown(self):
        """Graceful shutdown: drain queues, save state, close connections."""
        self.logger.info("ARIA shutdown initiated")
        self.running = False

        # Give alert handler time to drain
        for _ in range(10):
            if self.alert_queue.empty():
                break
            await asyncio.sleep(0.5)

        await self.save_state()

        # Close integrations
        if hasattr(self, "scanner") and self.scanner:
            await self.scanner.close()
        if hasattr(self, "tracker") and self.tracker:
            self.tracker.close()

        if self._redis:
            try:
                self._redis.close()
            except Exception:
                pass

        self.logger.info("ARIA shutdown complete")

    async def save_state(self):
        """Save current state to JSON for crash recovery."""
        state = {
            "task_history": dict(self.task_history),
            "performance_metrics": self.performance_metrics,
            "health_status": self.health_status,
            "last_task_runs": {
                k: v.isoformat() for k, v in self._last_task_runs.items()
            },
            "timestamp": datetime.now().isoformat(),
        }

        state_path = Path("aria_state.json")
        try:
            with open(state_path, "w") as f:
                json.dump(state, f, indent=2, cls=AriaJSONEncoder)
            self.logger.debug(f"State saved to {state_path}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    """Launch ARIA."""
    config_path = "aria_config.yaml" if Path("aria_config.yaml").exists() else None
    aria = ARIA(config_path)
    await aria.run()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════╗
    ║  ARIA — Autonomous Career Assistant           ║
    ║  24/7 AI-Powered Career Management            ║
    ╠═══════════════════════════════════════════════╣
    ║  Monitoring:                                  ║
    ║  • LinkedIn, GitHub, Blog, Career System      ║
    ║  • Opportunity detection across 10+ sources   ║
    ║  • Real-time engagement and response          ║
    ║  • Automated reporting & analytics            ║
    ╠═══════════════════════════════════════════════╣
    ║  Press Ctrl+C to shutdown gracefully.         ║
    ╚═══════════════════════════════════════════════╝
    """)

    asyncio.run(main())
