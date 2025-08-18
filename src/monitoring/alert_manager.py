#!/usr/bin/env python3
"""
Alert Manager for DipMaster Trading System
告警管理器 - 智能告警系统和通知管理

Features:
- Multi-level alert system (info, warning, critical)
- Multiple notification channels (email, Slack, webhook)
- Alert correlation and deduplication
- Escalation policies and acknowledgment
- Alert history and analytics
- Intelligent alert suppression
"""

import time
import json
import threading
import requests
import smtplib
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import defaultdict, deque
import logging

from .metrics_collector import MetricsCollector, get_metrics_collector

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Individual alert instance."""
    id: str
    name: str
    level: AlertLevel
    message: str
    source: str
    metric_name: Optional[str] = None
    current_value: Optional[float] = None
    threshold_value: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    escalation_count: int = 0
    suppression_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format."""
        return {
            'id': self.id,
            'name': self.name,
            'level': self.level.value,
            'message': self.message,
            'source': self.source,
            'metric_name': self.metric_name,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'timestamp': self.timestamp,
            'status': self.status.value,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at,
            'resolved_at': self.resolved_at,
            'tags': self.tags,
            'escalation_count': self.escalation_count,
            'suppression_reason': self.suppression_reason
        }


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    metric_name: str
    condition: str  # 'greater_than', 'less_than', 'equals', 'not_equals'
    threshold: float
    level: AlertLevel
    message_template: str
    duration_seconds: int = 60  # How long condition must persist
    cooldown_seconds: int = 300  # Minimum time between alerts
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class NotificationChannel:
    """Notification channel configuration."""
    name: str
    type: str  # 'email', 'slack', 'webhook', 'sms'
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    min_level: AlertLevel = AlertLevel.INFO
    rate_limit: int = 10  # Max notifications per hour


class AlertManager:
    """
    Comprehensive alert management system.
    
    Manages alert rules, correlation, escalation, and multiple
    notification channels with intelligent suppression.
    """
    
    def __init__(self,
                 metrics_collector: Optional[MetricsCollector] = None,
                 config_file: Optional[str] = None):
        """
        Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector instance
            config_file: Path to alert configuration file
        """
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.config_file = config_file
        
        # Thread-safe storage
        self._lock = threading.RLock()
        
        # Alert management
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=10000)
        self._alert_rules: Dict[str, AlertRule] = {}
        self._notification_channels: Dict[str, NotificationChannel] = {}
        
        # Alert correlation and suppression
        self._correlation_groups: Dict[str, Set[str]] = {}
        self._suppression_rules: Dict[str, Dict[str, Any]] = {}
        self._alert_frequencies: Dict[str, List[float]] = defaultdict(list)
        
        # Notification tracking
        self._notification_history: Dict[str, List[float]] = defaultdict(list)
        self._last_notification_time: Dict[str, float] = {}
        
        # Background processing
        self._processing_active = False
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Load configuration
        self._load_configuration()
        
        # Register alert metrics
        self._register_alert_metrics()
        
        logger.info("🚨 AlertManager initialized")
    
    def _load_configuration(self):
        """Load alert configuration from file."""
        if self.config_file:
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                
                # Load alert rules
                for rule_config in config.get('alert_rules', []):
                    rule = AlertRule(**rule_config)
                    self._alert_rules[rule.name] = rule
                
                # Load notification channels
                for channel_config in config.get('notification_channels', []):
                    channel = NotificationChannel(**channel_config)
                    self._notification_channels[channel.name] = channel
                
                # Load suppression rules
                self._suppression_rules = config.get('suppression_rules', {})
                
                logger.info(f"✅ Loaded alert configuration: {len(self._alert_rules)} rules, {len(self._notification_channels)} channels")
                
            except Exception as e:
                logger.error(f"❌ Failed to load alert configuration: {e}")
                self._create_default_configuration()
        else:
            self._create_default_configuration()
    
    def _create_default_configuration(self):
        """Create default alert configuration."""
        # Default alert rules
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric_name="system.cpu.usage_percent",
                condition="greater_than",
                threshold=80.0,
                level=AlertLevel.WARNING,
                message_template="High CPU usage: {current_value}% (threshold: {threshold}%)",
                duration_seconds=120,
                tags={"category": "system", "resource": "cpu"}
            ),
            AlertRule(
                name="critical_cpu_usage",
                metric_name="system.cpu.usage_percent",
                condition="greater_than",
                threshold=95.0,
                level=AlertLevel.CRITICAL,
                message_template="Critical CPU usage: {current_value}% (threshold: {threshold}%)",
                duration_seconds=60,
                tags={"category": "system", "resource": "cpu"}
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system.memory.usage_percent",
                condition="greater_than",
                threshold=85.0,
                level=AlertLevel.WARNING,
                message_template="High memory usage: {current_value}% (threshold: {threshold}%)",
                duration_seconds=180,
                tags={"category": "system", "resource": "memory"}
            ),
            AlertRule(
                name="low_win_rate",
                metric_name="trading.win_rate",
                condition="less_than",
                threshold=70.0,
                level=AlertLevel.WARNING,
                message_template="Low win rate: {current_value}% (threshold: {threshold}%)",
                duration_seconds=600,  # 10 minutes
                tags={"category": "trading", "metric": "performance"}
            ),
            AlertRule(
                name="high_drawdown",
                metric_name="risk.current_drawdown",
                condition="greater_than",
                threshold=10.0,
                level=AlertLevel.CRITICAL,
                message_template="High drawdown: {current_value}% (threshold: {threshold}%)",
                duration_seconds=300,
                tags={"category": "risk", "metric": "drawdown"}
            )
        ]
        
        for rule in default_rules:
            self._alert_rules[rule.name] = rule
        
        # Default notification channels would be configured via environment variables
        # or external configuration in production
        
        logger.info("📋 Created default alert configuration")
    
    def _register_alert_metrics(self):
        """Register alert-related metrics."""
        from .metrics_collector import MetricType
        
        alert_metrics = [
            ('alerts.total_active', MetricType.GAUGE, 'Number of active alerts', 'count'),
            ('alerts.total_critical', MetricType.GAUGE, 'Number of critical alerts', 'count'),
            ('alerts.total_warning', MetricType.GAUGE, 'Number of warning alerts', 'count'),
            ('alerts.notifications_sent', MetricType.COUNTER, 'Total notifications sent', 'count'),
            ('alerts.notifications_failed', MetricType.COUNTER, 'Failed notification attempts', 'count'),
            ('alerts.escalations', MetricType.COUNTER, 'Alert escalations', 'count'),
            ('alerts.avg_resolution_time', MetricType.GAUGE, 'Average alert resolution time', 'seconds')
        ]
        
        for name, metric_type, description, unit in alert_metrics:
            self.metrics_collector.register_metric(name, metric_type, description, unit)
    
    def start_processing(self):
        """Start background alert processing."""
        if self._processing_active:
            logger.warning("⚠️  Alert processing already active")
            return
        
        self._processing_active = True
        self._stop_event.clear()
        
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()
        
        logger.info("🚀 Alert processing started")
    
    def stop_processing(self):
        """Stop background alert processing."""
        if not self._processing_active:
            return
        
        logger.info("🛑 Stopping alert processing...")
        
        self._processing_active = False
        self._stop_event.set()
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=10)
        
        logger.info("✅ Alert processing stopped")
    
    def _processing_loop(self):
        """Main alert processing loop."""
        while self._processing_active and not self._stop_event.is_set():
            try:
                start_time = time.time()
                
                # Evaluate alert rules
                self._evaluate_alert_rules()
                
                # Process escalations
                self._process_escalations()
                
                # Update metrics
                self._update_alert_metrics()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Sleep for remaining time in cycle (30 seconds)
                elapsed = time.time() - start_time
                sleep_time = max(0, 30 - elapsed)
                
                if self._stop_event.wait(sleep_time):
                    break
                
            except Exception as e:
                logger.error(f"❌ Alert processing error: {e}")
                if self._stop_event.wait(10):
                    break
    
    def _evaluate_alert_rules(self):
        """Evaluate all alert rules against current metrics."""
        current_time = time.time()
        
        for rule_name, rule in self._alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Get current metric value
                current_value = self.metrics_collector.get_latest_value(rule.metric_name)
                if current_value is None:
                    continue
                
                # Check condition
                condition_met = self._check_condition(current_value, rule.condition, rule.threshold)
                
                alert_id = f"{rule_name}_{rule.metric_name}"
                
                if condition_met:
                    # Check if this is a new alert or existing one
                    if alert_id not in self._active_alerts:
                        # Check duration requirement
                        if self._check_duration_requirement(rule, current_value):
                            self._create_alert(rule, current_value, alert_id)
                    else:
                        # Update existing alert
                        self._update_alert(alert_id, current_value)
                else:
                    # Condition not met, resolve if alert exists
                    if alert_id in self._active_alerts:
                        self._resolve_alert(alert_id, "Condition no longer met")
                        
            except Exception as e:
                logger.error(f"❌ Error evaluating alert rule {rule_name}: {e}")
    
    def _check_condition(self, current_value: float, condition: str, threshold: float) -> bool:
        """Check if alert condition is met."""
        if condition == "greater_than":
            return current_value > threshold
        elif condition == "less_than":
            return current_value < threshold
        elif condition == "equals":
            return abs(current_value - threshold) < 0.001  # Float comparison
        elif condition == "not_equals":
            return abs(current_value - threshold) >= 0.001
        else:
            logger.warning(f"⚠️  Unknown condition: {condition}")
            return False
    
    def _check_duration_requirement(self, rule: AlertRule, current_value: float) -> bool:
        """Check if condition has been met for required duration."""
        # This is a simplified implementation
        # In production, you'd track condition states over time
        return True  # For now, always return True
    
    def _create_alert(self, rule: AlertRule, current_value: float, alert_id: str):
        """Create a new alert."""
        with self._lock:
            # Check cooldown period
            if self._is_in_cooldown(rule.name):
                return
            
            # Check suppression rules
            if self._is_suppressed(rule.name, current_value):
                logger.debug(f"🔇 Alert suppressed: {rule.name}")
                return
            
            # Create alert
            message = rule.message_template.format(
                current_value=current_value,
                threshold=rule.threshold,
                metric=rule.metric_name
            )
            
            alert = Alert(
                id=alert_id,
                name=rule.name,
                level=rule.level,
                message=message,
                source="alert_manager",
                metric_name=rule.metric_name,
                current_value=current_value,
                threshold_value=rule.threshold,
                tags=rule.tags.copy()
            )
            
            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)
            
            # Send notifications
            self._send_notifications(alert)
            
            # Update frequency tracking
            self._alert_frequencies[rule.name].append(time.time())
            
            logger.warning(f"🚨 Alert created: {rule.name} - {message}")
    
    def _update_alert(self, alert_id: str, current_value: float):
        """Update existing alert with new value."""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.current_value = current_value
                
                # Update message
                if alert.metric_name:
                    rule = self._alert_rules.get(alert.name)
                    if rule:
                        alert.message = rule.message_template.format(
                            current_value=current_value,
                            threshold=alert.threshold_value,
                            metric=alert.metric_name
                        )
    
    def _resolve_alert(self, alert_id: str, reason: str = ""):
        """Resolve an active alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts.pop(alert_id)
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = time.time()
                
                # Calculate resolution time
                resolution_time = alert.resolved_at - alert.timestamp
                self.metrics_collector.record_value('alerts.avg_resolution_time', resolution_time)
                
                logger.info(f"✅ Alert resolved: {alert.name} - {reason}")
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if rule is in cooldown period."""
        if rule_name not in self._alert_rules:
            return False
        
        rule = self._alert_rules[rule_name]
        last_time = self._last_notification_time.get(rule_name, 0)
        
        return (time.time() - last_time) < rule.cooldown_seconds
    
    def _is_suppressed(self, rule_name: str, current_value: float) -> bool:
        """Check if alert should be suppressed."""
        # Frequency-based suppression
        if rule_name in self._alert_frequencies:
            recent_alerts = [t for t in self._alert_frequencies[rule_name] 
                           if time.time() - t < 3600]  # Last hour
            
            if len(recent_alerts) > 10:  # Too many alerts in last hour
                return True
        
        # Custom suppression rules
        suppression_rule = self._suppression_rules.get(rule_name)
        if suppression_rule:
            # Implement custom suppression logic here
            pass
        
        return False
    
    def _send_notifications(self, alert: Alert):
        """Send notifications for alert to all configured channels."""
        notifications_sent = 0
        notifications_failed = 0
        
        for channel_name, channel in self._notification_channels.items():
            if not channel.enabled:
                continue
            
            # Check minimum level
            if alert.level.value < channel.min_level.value:
                continue
            
            # Check rate limiting
            if self._is_rate_limited(channel_name):
                logger.warning(f"⚠️  Rate limiting notification channel: {channel_name}")
                continue
            
            try:
                success = self._send_notification(channel, alert)
                if success:
                    notifications_sent += 1
                    self._notification_history[channel_name].append(time.time())
                    self._last_notification_time[alert.name] = time.time()
                else:
                    notifications_failed += 1
                    
            except Exception as e:
                logger.error(f"❌ Notification failed for channel {channel_name}: {e}")
                notifications_failed += 1
        
        # Update metrics
        self.metrics_collector.increment_counter('alerts.notifications_sent', notifications_sent)
        self.metrics_collector.increment_counter('alerts.notifications_failed', notifications_failed)
    
    def _is_rate_limited(self, channel_name: str) -> bool:
        """Check if notification channel is rate limited."""
        channel = self._notification_channels.get(channel_name)
        if not channel:
            return True
        
        recent_notifications = [t for t in self._notification_history[channel_name]
                               if time.time() - t < 3600]  # Last hour
        
        return len(recent_notifications) >= channel.rate_limit
    
    def _send_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send notification via specific channel."""
        try:
            if channel.type == 'slack':
                return self._send_slack_notification(channel, alert)
            elif channel.type == 'email':
                return self._send_email_notification(channel, alert)
            elif channel.type == 'webhook':
                return self._send_webhook_notification(channel, alert)
            else:
                logger.warning(f"⚠️  Unknown notification channel type: {channel.type}")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to send {channel.type} notification: {e}")
            return False
    
    def _send_slack_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send Slack notification."""
        webhook_url = channel.config.get('webhook_url')
        if not webhook_url:
            logger.error("❌ Slack webhook URL not configured")
            return False
        
        # Color coding for different alert levels
        color_map = {
            AlertLevel.INFO: "#36a64f",      # Green
            AlertLevel.WARNING: "#ff9500",   # Orange
            AlertLevel.CRITICAL: "#ff0000",  # Red
            AlertLevel.EMERGENCY: "#8B0000"  # Dark Red
        }
        
        payload = {
            "attachments": [{
                "color": color_map.get(alert.level, "#36a64f"),
                "title": f"🚨 {alert.level.value.upper()}: {alert.name}",
                "text": alert.message,
                "fields": [
                    {
                        "title": "Source",
                        "value": alert.source,
                        "short": True
                    },
                    {
                        "title": "Metric",
                        "value": alert.metric_name or "N/A",
                        "short": True
                    },
                    {
                        "title": "Current Value",
                        "value": f"{alert.current_value:.2f}" if alert.current_value else "N/A",
                        "short": True
                    },
                    {
                        "title": "Threshold",
                        "value": f"{alert.threshold_value:.2f}" if alert.threshold_value else "N/A",
                        "short": True
                    }
                ],
                "timestamp": int(alert.timestamp)
            }]
        }
        
        response = requests.post(webhook_url, json=payload, timeout=10)
        return response.status_code == 200
    
    def _send_email_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send email notification."""
        smtp_config = channel.config
        
        try:
            msg = MIMEMultipart()
            msg['From'] = smtp_config.get('from_email')
            msg['To'] = smtp_config.get('to_email')
            msg['Subject'] = f"DipMaster Alert: {alert.level.value.upper()} - {alert.name}"
            
            body = f"""
            Alert: {alert.name}
            Level: {alert.level.value.upper()}
            Message: {alert.message}
            
            Details:
            - Source: {alert.source}
            - Metric: {alert.metric_name or 'N/A'}
            - Current Value: {alert.current_value or 'N/A'}
            - Threshold: {alert.threshold_value or 'N/A'}
            - Time: {datetime.fromtimestamp(alert.timestamp)}
            
            Alert ID: {alert.id}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_config.get('smtp_server'), smtp_config.get('smtp_port', 587))
            server.starttls()
            server.login(smtp_config.get('username'), smtp_config.get('password'))
            
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Email notification failed: {e}")
            return False
    
    def _send_webhook_notification(self, channel: NotificationChannel, alert: Alert) -> bool:
        """Send webhook notification."""
        webhook_url = channel.config.get('url')
        if not webhook_url:
            return False
        
        payload = alert.to_dict()
        headers = channel.config.get('headers', {'Content-Type': 'application/json'})
        
        response = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
        return response.status_code < 400
    
    def _process_escalations(self):
        """Process alert escalations."""
        current_time = time.time()
        
        with self._lock:
            for alert_id, alert in self._active_alerts.items():
                # Check if alert needs escalation
                if (alert.status == AlertStatus.ACTIVE and 
                    alert.escalation_count == 0 and
                    current_time - alert.timestamp > 1800):  # 30 minutes
                    
                    # Escalate alert
                    alert.escalation_count += 1
                    alert.level = AlertLevel.CRITICAL  # Escalate to critical
                    
                    self._send_notifications(alert)
                    self.metrics_collector.increment_counter('alerts.escalations')
                    
                    logger.warning(f"📈 Alert escalated: {alert.name}")
    
    def _update_alert_metrics(self):
        """Update alert-related metrics."""
        with self._lock:
            active_count = len(self._active_alerts)
            critical_count = sum(1 for alert in self._active_alerts.values() 
                               if alert.level == AlertLevel.CRITICAL)
            warning_count = sum(1 for alert in self._active_alerts.values() 
                              if alert.level == AlertLevel.WARNING)
            
            self.metrics_collector.set_gauge('alerts.total_active', active_count)
            self.metrics_collector.set_gauge('alerts.total_critical', critical_count)
            self.metrics_collector.set_gauge('alerts.total_warning', warning_count)
    
    def _cleanup_old_data(self):
        """Clean up old notification and frequency data."""
        current_time = time.time()
        
        # Clean notification history (keep last 24 hours)
        for channel_name in self._notification_history:
            self._notification_history[channel_name] = [
                t for t in self._notification_history[channel_name]
                if current_time - t < 86400
            ]
        
        # Clean alert frequency data (keep last 24 hours)
        for rule_name in self._alert_frequencies:
            self._alert_frequencies[rule_name] = [
                t for t in self._alert_frequencies[rule_name]
                if current_time - t < 86400
            ]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an active alert."""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = acknowledged_by
                alert.acknowledged_at = time.time()
                
                logger.info(f"👍 Alert acknowledged: {alert.name} by {acknowledged_by}")
                return True
            return False
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of all active alerts."""
        with self._lock:
            return [alert.to_dict() for alert in self._active_alerts.values()]
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified time period."""
        cutoff_time = time.time() - (hours * 3600)
        
        with self._lock:
            recent_alerts = [alert for alert in self._alert_history 
                           if alert.timestamp >= cutoff_time]
            
            return [alert.to_dict() for alert in recent_alerts]
    
    def add_notification_channel(self, channel: NotificationChannel):
        """Add a new notification channel."""
        self._notification_channels[channel.name] = channel
        logger.info(f"📢 Added notification channel: {channel.name} ({channel.type})")
    
    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self._alert_rules[rule.name] = rule
        logger.info(f"📏 Added alert rule: {rule.name}")


def create_alert_manager(metrics_collector: Optional[MetricsCollector] = None,
                        config_file: Optional[str] = None) -> AlertManager:
    """Factory function to create alert manager."""
    return AlertManager(metrics_collector, config_file)