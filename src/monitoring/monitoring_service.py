#!/usr/bin/env python3
"""
Monitoring Service for DipMaster Trading System
监控服务 - 集成所有监控组件的统一服务

Features:
- Unified monitoring service orchestration
- Integration with all monitoring components
- Real-time dashboard data provision
- Health check endpoints
- Monitoring service lifecycle management
"""

import asyncio
import threading
import time
from typing import Dict, Any, Optional, List
import logging
from pathlib import Path

from .metrics_collector import MetricsCollector
from .business_kpi import BusinessKPITracker  
from .system_monitor import SystemMonitor
from .alert_manager import AlertManager, AlertRule, NotificationChannel, AlertLevel

logger = logging.getLogger(__name__)


class MonitoringService:
    """
    Unified monitoring service that orchestrates all monitoring components.
    
    Provides a single interface for managing metrics collection, business KPI tracking,
    system monitoring, and alerting across the entire trading system.
    """
    
    def __init__(self,
                 config: Optional[Dict[str, Any]] = None,
                 alert_config_file: Optional[str] = None):
        """
        Initialize monitoring service.
        
        Args:
            config: Monitoring configuration dictionary
            alert_config_file: Path to alert configuration file
        """
        self.config = config or {}
        self.alert_config_file = alert_config_file
        
        # Initialize components
        self.metrics_collector = MetricsCollector(
            retention_hours=self.config.get('metrics_retention_hours', 24),
            cleanup_interval=self.config.get('metrics_cleanup_interval', 300)
        )
        
        self.business_kpi = BusinessKPITracker(
            metrics_collector=self.metrics_collector,
            history_retention_hours=self.config.get('kpi_retention_hours', 720)
        )
        
        self.system_monitor = SystemMonitor(
            metrics_collector=self.metrics_collector,
            monitoring_interval=self.config.get('system_monitoring_interval', 30)
        )
        
        self.alert_manager = AlertManager(
            metrics_collector=self.metrics_collector,
            config_file=alert_config_file
        )
        
        # Service state
        self._running = False
        self._start_time = None
        
        # Setup default alert rules and notification channels
        self._setup_default_alerts()
        
        logger.info("📊 MonitoringService initialized")
    
    def _setup_default_alerts(self):
        """Setup default alert rules and notification channels."""
        try:
            # Setup Slack notification if configured
            slack_webhook = self.config.get('slack_webhook_url')
            if slack_webhook:
                slack_channel = NotificationChannel(
                    name="slack_alerts",
                    type="slack",
                    config={"webhook_url": slack_webhook},
                    enabled=True,
                    min_level=AlertLevel.WARNING
                )
                self.alert_manager.add_notification_channel(slack_channel)
                logger.info("📢 Slack notifications configured")
            
            # Setup email notifications if configured
            email_config = self.config.get('email_config')
            if email_config:
                email_channel = NotificationChannel(
                    name="email_alerts",
                    type="email",
                    config=email_config,
                    enabled=True,
                    min_level=AlertLevel.CRITICAL
                )
                self.alert_manager.add_notification_channel(email_channel)
                logger.info("📧 Email notifications configured")
            
            # Add trading-specific alert rules
            trading_rules = [
                AlertRule(
                    name="trading_system_down",
                    metric_name="health.overall_score",
                    condition="less_than",
                    threshold=50.0,
                    level=AlertLevel.CRITICAL,
                    message_template="Trading system health critical: {current_value} (threshold: {threshold})",
                    duration_seconds=60,
                    cooldown_seconds=300,
                    tags={"category": "system", "criticality": "high"}
                ),
                AlertRule(
                    name="api_error_rate_high",
                    metric_name="app.api_error_rate",
                    condition="greater_than",
                    threshold=15.0,
                    level=AlertLevel.WARNING,
                    message_template="High API error rate: {current_value}% (threshold: {threshold}%)",
                    duration_seconds=180,
                    cooldown_seconds=600,
                    tags={"category": "api", "metric": "errors"}
                ),
                AlertRule(
                    name="websocket_disconnected",
                    metric_name="app.websocket_connections",
                    condition="equals",
                    threshold=0.0,
                    level=AlertLevel.CRITICAL,
                    message_template="WebSocket connections lost: {current_value} active connections",
                    duration_seconds=30,
                    cooldown_seconds=180,
                    tags={"category": "connectivity", "criticality": "high"}
                ),
                AlertRule(
                    name="excessive_drawdown",
                    metric_name="risk.current_drawdown",
                    condition="greater_than",
                    threshold=15.0,
                    level=AlertLevel.EMERGENCY,
                    message_template="Emergency: Excessive drawdown {current_value}% (threshold: {threshold}%)",
                    duration_seconds=60,
                    cooldown_seconds=0,  # No cooldown for emergency alerts
                    tags={"category": "risk", "criticality": "emergency"}
                )
            ]
            
            for rule in trading_rules:
                self.alert_manager.add_alert_rule(rule)
            
            logger.info(f"📏 Added {len(trading_rules)} trading-specific alert rules")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup default alerts: {e}")
    
    async def start(self):
        """Start all monitoring components."""
        try:
            if self._running:
                logger.warning("⚠️  Monitoring service already running")
                return
            
            logger.info("🚀 Starting monitoring service...")
            self._start_time = time.time()
            
            # Start system monitor
            self.system_monitor.start_monitoring()
            
            # Start alert processing
            self.alert_manager.start_processing()
            
            # Record startup
            self.metrics_collector.record_value('monitoring.service_starts', 1)
            self.metrics_collector.set_gauge('monitoring.service_running', 1)
            
            self._running = True
            
            logger.info("✅ Monitoring service started successfully")
            
            # Record system startup event
            await self._record_system_event('MONITORING_STARTED', {
                'components': ['metrics_collector', 'business_kpi', 'system_monitor', 'alert_manager'],
                'config_keys': list(self.config.keys())
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring service: {e}")
            raise
    
    async def stop(self):
        """Stop all monitoring components."""
        try:
            if not self._running:
                logger.warning("⚠️  Monitoring service not running")
                return
            
            logger.info("🛑 Stopping monitoring service...")
            
            # Calculate uptime
            uptime = time.time() - self._start_time if self._start_time else 0
            
            # Stop components
            self.system_monitor.stop_monitoring()
            self.alert_manager.stop_processing()
            self.metrics_collector.shutdown()
            
            # Record shutdown
            self.metrics_collector.set_gauge('monitoring.service_running', 0)
            self.metrics_collector.record_value('monitoring.service_uptime_seconds', uptime)
            
            # Record system shutdown event
            await self._record_system_event('MONITORING_STOPPED', {
                'uptime_seconds': uptime,
                'graceful_shutdown': True
            })
            
            self._running = False
            
            logger.info("✅ Monitoring service stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping monitoring service: {e}")
    
    async def _record_system_event(self, event_type: str, details: Dict[str, Any]):
        """Record system event (placeholder for audit logging integration)."""
        # This would integrate with the security audit logger in production
        logger.info(f"📋 System event: {event_type} - {details}")
    
    def record_trade_entry(self, **kwargs):
        """Record trade entry in business KPI tracker."""
        return self.business_kpi.record_trade_entry(**kwargs)
    
    def record_trade_exit(self, **kwargs):
        """Record trade exit in business KPI tracker."""
        return self.business_kpi.record_trade_exit(**kwargs)
    
    def update_position_price(self, trade_id: str, current_price: float):
        """Update position price in business KPI tracker."""
        return self.business_kpi.update_position_price(trade_id, current_price)
    
    def record_websocket_connection(self, connection_id: str, info: Dict[str, Any]):
        """Record WebSocket connection in system monitor."""
        return self.system_monitor.record_websocket_connection(connection_id, info)
    
    def remove_websocket_connection(self, connection_id: str):
        """Remove WebSocket connection from system monitor."""
        return self.system_monitor.remove_websocket_connection(connection_id)
    
    def record_api_call(self, endpoint: str, response_time: float, error: bool = False):
        """Record API call statistics in system monitor."""
        return self.system_monitor.record_api_call(endpoint, response_time, error)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        try:
            current_time = time.time()
            
            # Get metrics summary
            metrics_summary = self.metrics_collector.get_metrics_summary(300)  # 5 minutes
            
            # Get trading performance
            trading_summary = self.business_kpi.get_trading_summary(24)  # 24 hours
            
            # Get system status
            system_status = self.system_monitor.get_system_status()
            
            # Get active alerts
            active_alerts = self.alert_manager.get_active_alerts()
            
            # Get service status
            uptime = current_time - self._start_time if self._start_time else 0
            
            dashboard_data = {
                'timestamp': current_time,
                'service_status': {
                    'running': self._running,
                    'uptime_seconds': uptime,
                    'uptime_human': self._format_uptime(uptime),
                    'start_time': self._start_time
                },
                'metrics_summary': metrics_summary,
                'trading_performance': trading_summary,
                'system_status': system_status,
                'alerts': {
                    'active_count': len(active_alerts),
                    'active_alerts': active_alerts[:10],  # Limit to 10 most recent
                    'critical_count': len([a for a in active_alerts if a['level'] == 'critical']),
                    'warning_count': len([a for a in active_alerts if a['level'] == 'warning'])
                },
                'key_metrics': {
                    'win_rate': self.metrics_collector.get_latest_value('trading.win_rate'),
                    'daily_pnl': self.metrics_collector.get_latest_value('trading.daily_pnl'),
                    'active_positions': self.metrics_collector.get_latest_value('risk.active_positions'),
                    'cpu_usage': self.metrics_collector.get_latest_value('system.cpu.usage_percent'),
                    'memory_usage': self.metrics_collector.get_latest_value('system.memory.usage_percent'),
                    'health_score': self.metrics_collector.get_latest_value('health.overall_score')
                }
            }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"❌ Failed to get dashboard data: {e}")
            return {
                'error': str(e),
                'timestamp': time.time(),
                'service_status': {'running': self._running}
            }
    
    def _format_uptime(self, uptime_seconds: float) -> str:
        """Format uptime in human-readable format."""
        if uptime_seconds < 60:
            return f"{int(uptime_seconds)}s"
        elif uptime_seconds < 3600:
            return f"{int(uptime_seconds / 60)}m"
        elif uptime_seconds < 86400:
            hours = int(uptime_seconds / 3600)
            minutes = int((uptime_seconds % 3600) / 60)
            return f"{hours}h {minutes}m"
        else:
            days = int(uptime_seconds / 86400)
            hours = int((uptime_seconds % 86400) / 3600)
            return f"{days}d {hours}h"
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get service health status for health checks."""
        try:
            health_score = self.metrics_collector.get_latest_value('health.overall_score') or 0
            active_alerts = len(self.alert_manager.get_active_alerts())
            
            # Determine overall health
            if health_score >= 90:
                status = "healthy"
            elif health_score >= 70:
                status = "degraded" 
            else:
                status = "unhealthy"
            
            return {
                'status': status,
                'health_score': health_score,
                'active_alerts': active_alerts,
                'uptime_seconds': time.time() - self._start_time if self._start_time else 0,
                'components': {
                    'metrics_collector': 'running' if self.metrics_collector else 'stopped',
                    'business_kpi': 'running' if self.business_kpi else 'stopped',
                    'system_monitor': 'running' if self.system_monitor._monitoring_active else 'stopped',
                    'alert_manager': 'running' if self.alert_manager._processing_active else 'stopped'
                },
                'timestamp': time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }
    
    def export_metrics(self, format_type: str = "json") -> str:
        """Export all metrics in specified format."""
        return self.metrics_collector.export_metrics(format_type)
    
    def get_trading_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Get detailed trading performance report."""
        try:
            trading_summary = self.business_kpi.get_trading_summary(hours)
            strategy_performance = self.business_kpi.get_strategy_performance()
            
            # Get relevant metrics
            current_time = time.time()
            start_time = current_time - (hours * 3600)
            
            report = {
                'report_period': {
                    'hours': hours,
                    'start_time': start_time,
                    'end_time': current_time
                },
                'trading_summary': trading_summary,
                'strategy_performance': strategy_performance,
                'risk_metrics': {
                    'current_drawdown': self.metrics_collector.get_latest_value('risk.current_drawdown'),
                    'max_drawdown': self.metrics_collector.get_latest_value('risk.max_drawdown'),
                    'active_positions': self.metrics_collector.get_latest_value('risk.active_positions'),
                    'total_exposure': self.metrics_collector.get_latest_value('risk.total_exposure'),
                    'unrealized_pnl': self.metrics_collector.get_latest_value('risk.unrealized_pnl')
                },
                'system_health': {
                    'overall_score': self.metrics_collector.get_latest_value('health.overall_score'),
                    'active_alerts': len(self.alert_manager.get_active_alerts()),
                    'api_error_rate': self.metrics_collector.get_latest_value('app.api_error_rate'),
                    'websocket_connections': self.metrics_collector.get_latest_value('app.websocket_connections')
                },
                'generated_at': current_time
            }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Failed to generate performance report: {e}")
            return {'error': str(e)}
    
    def is_running(self) -> bool:
        """Check if monitoring service is running."""
        return self._running


def create_monitoring_service(config: Optional[Dict[str, Any]] = None,
                            alert_config_file: Optional[str] = None) -> MonitoringService:
    """Factory function to create monitoring service."""
    return MonitoringService(config, alert_config_file)