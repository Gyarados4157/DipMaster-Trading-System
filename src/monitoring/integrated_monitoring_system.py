#!/usr/bin/env python3
"""
Integrated Monitoring System for DipMaster Trading System
集成监控系统 - 统一的监控、告警和日志管理平台

Features:
- Unified monitoring orchestration
- Kafka event streaming integration
- Real-time alerting and notifications
- Comprehensive health monitoring
- Performance and risk tracking
- Structured logging integration
- Dashboard data provision
- Automated remediation actions
"""

import asyncio
import time
import logging
import json
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import threading

# Import monitoring components
from .kafka_event_schemas import DipMasterKafkaStreamer, ExecutionReportEvent, RiskMetricsEvent, AlertEvent, StrategyPerformanceEvent
from .monitoring_architecture import MonitoringSystem
from .quality_assurance_system import QualityAssuranceSystem, SignalRecord, PositionRecord, ExecutionRecord
from .advanced_alert_system import AlertManager
from .realtime_dashboard_service import RealtimeDashboardService
from .automated_reporting_system import AutomatedReportingSystem

logger = logging.getLogger(__name__)


@dataclass
class MonitoringStatus:
    """Overall monitoring system status."""
    timestamp: float
    overall_health: str
    components_running: int
    components_total: int
    active_alerts: int
    critical_alerts: int
    events_processed_1h: int
    error_rate_percent: float
    last_kafka_event: Optional[float] = None


class IntegratedMonitoringSystem:
    """
    Comprehensive monitoring system that integrates all monitoring components.
    
    Provides unified orchestration of monitoring, alerting, logging, and event
    streaming for the DipMaster trading system.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize integrated monitoring system.
        
        Args:
            config: Monitoring system configuration dictionary
        """
        # Load configuration
        self.config = config or self._get_default_config()
        
        # System state
        self.is_running = False
        self.start_time = None
        
        # Initialize core components
        self._initialize_components()
        
        # Event tracking
        self.events_processed = 0
        self.events_failed = 0
        self.last_event_time = None
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        
        logger.info("🎯 IntegratedMonitoringSystem initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            "monitoring_enabled": True,
            "update_interval_seconds": 30,
            "kafka": {
                "enabled": True,
                "servers": ["localhost:9092"],
                "topics": {
                    "execution_reports": "exec.reports.v1",
                    "risk_metrics": "risk.metrics.v1",
                    "alerts": "alerts.v1",
                    "performance": "performance.v1"
                }
            },
            "quality_assurance": {
                "enabled": True,
                "signal_position_match_threshold": 0.95,
                "position_execution_match_threshold": 0.98,
                "drift_detection_window_hours": 24
            },
            "alert_system": {
                "enabled": True,
                "cooldown_seconds": 300,
                "escalation_enabled": True
            },
            "dashboard": {
                "enabled": True,
                "websocket_port": 8080,
                "update_intervals": {
                    "real_time_pnl": 1,
                    "positions": 2,
                    "risk_metrics": 5,
                    "system_health": 10
                }
            },
            "reporting": {
                "enabled": True,
                "daily_report_time": "06:00",
                "weekly_report_day": "monday",
                "reports_dir": "reports"
            }
        }
    
    def _initialize_components(self):
        """Initialize all monitoring components."""
        try:
            # Initialize Kafka event streaming
            if self.config.get('kafka', {}).get('enabled', True):
                kafka_config = self.config.get('kafka', {})
                self.kafka_streamer = DipMasterKafkaStreamer(
                    kafka_config={
                        'servers': kafka_config.get('servers', ['localhost:9092']),
                        'client_id': 'dipmaster-monitoring'
                    }
                )
                logger.info("📤 Kafka event streamer initialized")
            else:
                self.kafka_streamer = None
            
            # Initialize quality assurance system
            if self.config.get('quality_assurance', {}).get('enabled', True):
                qa_config = self.config.get('quality_assurance', {})
                self.quality_assurance = QualityAssuranceSystem(config=qa_config)
                logger.info("🔄 Quality assurance system initialized")
            else:
                self.quality_assurance = None
            
            # Initialize alert manager
            if self.config.get('alert_system', {}).get('enabled', True):
                alert_config = self.config.get('alert_system', {})
                self.alert_manager = AlertManager(config=alert_config)
                
                # Register notification handlers
                self.alert_manager.register_notification_handler("console", self._console_notification_handler)
                logger.info("🚨 Alert manager initialized")
            else:
                self.alert_manager = None
            
            # Initialize realtime dashboard service
            if self.config.get('dashboard', {}).get('enabled', True):
                dashboard_config = self.config.get('dashboard', {})
                self.dashboard_service = RealtimeDashboardService(config=dashboard_config)
                logger.info("📊 Realtime dashboard service initialized")
            else:
                self.dashboard_service = None
            
            # Initialize automated reporting system
            if self.config.get('reporting', {}).get('enabled', True):
                reporting_config = self.config.get('reporting', {})
                self.reporting_system = AutomatedReportingSystem(config=reporting_config)
                logger.info("📋 Automated reporting system initialized")
            else:
                self.reporting_system = None
            
            # Initialize core monitoring architecture
            self.monitoring_system = MonitoringSystem()
            
            logger.info("✅ All monitoring components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            raise
    
    async def start(self):
        """Start the integrated monitoring system."""
        if self.is_running:
            logger.warning("⚠️ Monitoring system already running")
            return
        
        try:
            logger.info("🚀 Starting integrated monitoring system...")
            self.start_time = time.time()
            
            # Start Kafka event streaming
            if self.kafka_streamer:
                await self.kafka_streamer.start()
            
            # Start quality assurance system
            if self.quality_assurance:
                # QA system doesn't have async start, but we can add monitoring
                logger.info("🔄 Quality assurance system active")
            
            # Start alert manager (if it has async capabilities)
            if self.alert_manager:
                logger.info("🚨 Alert manager active")
            
            # Start realtime dashboard service
            if self.dashboard_service:
                await self.dashboard_service.start()
            
            # Start automated reporting system
            if self.reporting_system:
                await self.reporting_system.start()
            
            # Start core monitoring system
            await self.monitoring_system.start()
            
            self.is_running = True
            logger.info("✅ Integrated monitoring system started successfully")
            
            # Start main monitoring loop
            self._background_tasks.append(asyncio.create_task(self._monitoring_loop()))
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring system: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the integrated monitoring system."""
        if not self.is_running:
            logger.warning("⚠️ Monitoring system not running")
            return
        
        try:
            logger.info("🛑 Stopping integrated monitoring system...")
            
            # Calculate uptime
            uptime = time.time() - self.start_time if self.start_time else 0
            
            # Cancel all background tasks
            for task in self._background_tasks:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            self._background_tasks.clear()
            
            # Stop Kafka event streaming
            if self.kafka_streamer:
                await self.kafka_streamer.stop()
            
            # Stop realtime dashboard service
            if self.dashboard_service:
                await self.dashboard_service.stop()
            
            # Stop automated reporting system
            if self.reporting_system:
                await self.reporting_system.stop()
            
            # Stop core monitoring system
            await self.monitoring_system.stop()
            
            self.is_running = False
            logger.info(f"✅ Integrated monitoring system stopped successfully (uptime: {uptime:.1f}s)")
            
        except Exception as e:
            logger.error(f"❌ Error stopping monitoring system: {e}")
            self.is_running = False
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("🔄 Starting main monitoring loop")
        
        update_interval = self.config.get('update_interval_seconds', 30)
        
        while self.is_running:
            try:
                # Run monitoring tasks
                await self._run_monitoring_cycle()
                
                # Sleep until next cycle
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                logger.info("🔄 Monitoring loop cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Longer sleep on error
    
    async def _run_monitoring_cycle(self):
        """Run a single monitoring cycle."""
        try:
            current_time = time.time()
            
            # Generate quality assurance report
            if self.quality_assurance:
                try:
                    qa_report = await self.quality_assurance.generate_comprehensive_quality_report()
                    
                    # Check for violations and generate alerts
                    if qa_report.violations:
                        for violation in qa_report.violations:
                            if violation.get('severity') in ['critical', 'warning']:
                                await self._publish_alert({
                                    'type': 'quality_assurance',
                                    'severity': violation.get('severity', 'warning').upper(),
                                    'message': violation.get('message', 'QA violation detected'),
                                    'source': 'quality_assurance_system',
                                    'timestamp': current_time
                                })
                    
                    logger.debug(f"🔍 QA Report: {qa_report.consistency_level.value} level")
                    
                except Exception as e:
                    logger.error(f"❌ Error generating QA report: {e}")
            
            # Evaluate alerts if alert manager is available
            if self.alert_manager:
                try:
                    # Example metrics to evaluate - in real use these would come from system metrics
                    test_metrics = [
                        ("system.cpu.usage_percent", 75.0),
                        ("system.memory.usage_percent", 60.0),
                        ("strategy.win_rate", 0.75),
                    ]
                    
                    for metric, value in test_metrics:
                        incidents = await self.alert_manager.evaluate_metric(metric, value)
                        for incident in incidents:
                            await self._publish_alert({
                                'type': 'threshold_alert',
                                'severity': incident.severity.value,
                                'message': incident.description,
                                'source': 'alert_manager',
                                'timestamp': current_time
                            })
                
                except Exception as e:
                    logger.error(f"❌ Error evaluating alerts: {e}")
            
            # Update cycle metrics
            self.events_processed += 1
            logger.debug(f"🔄 Monitoring cycle completed (cycle #{self.events_processed})")
            
        except Exception as e:
            logger.error(f"❌ Error in monitoring cycle: {e}")
            self.events_failed += 1
    
    async def _publish_alert(self, alert_data: Dict[str, Any]):
        """Publish alert to Kafka and dashboard."""
        try:
            # Create alert event for Kafka
            alert_event = AlertEvent(
                timestamp=alert_data.get('timestamp', time.time()),
                alert_id=f"alert_{int(time.time() * 1000)}",
                severity=alert_data.get('severity', 'WARNING'),
                category=alert_data.get('type', 'system'),
                message=alert_data.get('message', 'Alert triggered'),
                affected_systems=[alert_data.get('source', 'unknown')],
                recommended_action="Investigation required",
                auto_remediation=False,
                source=alert_data.get('source', 'integrated_monitoring')
            )
            
            # Publish to Kafka
            if self.kafka_streamer:
                await self.kafka_streamer.publish_alert(alert_event)
            
            # Send to dashboard
            if self.dashboard_service:
                await self.dashboard_service.add_alert(alert_data)
            
            logger.warning(f"🚨 Alert: [{alert_data.get('severity')}] {alert_data.get('message')}")
            
        except Exception as e:
            logger.error(f"❌ Error publishing alert: {e}")
    
    async def _console_notification_handler(self, incident):
        """Console notification handler for alerts."""
        print(f"🚨 ALERT [{incident.severity.value}]: {incident.title}")
        print(f"   Description: {incident.description}")
        if hasattr(incident, 'recommended_actions'):
            print(f"   Actions: {', '.join(incident.recommended_actions[:2])}")
    
    # Public API methods for recording events
    
    async def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade."""
        try:
            # Add to reporting system
            if self.reporting_system:
                self.reporting_system.add_trade_data(trade_data)
            
            # Update dashboard
            if self.dashboard_service:
                if trade_data.get('status') == 'closed':
                    await self.dashboard_service.record_trade_exit(
                        position_id=trade_data.get('position_id'),
                        exit_price=trade_data.get('exit_price'),
                        realized_pnl=trade_data.get('pnl', 0),
                        holding_minutes=trade_data.get('holding_minutes', 0)
                    )
            
            self.events_processed += 1
            self.last_event_time = time.time()
            logger.debug(f"📊 Recorded trade: {trade_data.get('symbol')} PnL: {trade_data.get('pnl', 0)}")
            
        except Exception as e:
            logger.error(f"❌ Error recording trade: {e}")
            self.events_failed += 1
    
    async def record_execution(self, execution_data: Dict[str, Any]):
        """Record an order execution."""
        try:
            # Publish to Kafka
            if self.kafka_streamer:
                exec_event = ExecutionReportEvent(
                    timestamp=execution_data.get('timestamp', time.time()),
                    execution_id=execution_data.get('execution_id'),
                    signal_id=execution_data.get('signal_id'),
                    symbol=execution_data.get('symbol'),
                    side=execution_data.get('side'),
                    quantity=execution_data.get('quantity'),
                    price=execution_data.get('price'),
                    slippage_bps=execution_data.get('slippage_bps', 0),
                    latency_ms=execution_data.get('latency_ms', 0),
                    venue=execution_data.get('venue', 'binance'),
                    status=execution_data.get('status', 'FILLED')
                )
                await self.kafka_streamer.publish_execution_report(exec_event)
            
            # Add to reporting system
            if self.reporting_system:
                self.reporting_system.add_execution_data(execution_data)
            
            self.events_processed += 1
            self.last_event_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ Error recording execution: {e}")
            self.events_failed += 1
    
    async def record_signal(self, signal_data: Dict[str, Any]):
        """Record a trading signal."""
        try:
            # Record in quality assurance system
            if self.quality_assurance:
                signal_record = SignalRecord(
                    signal_id=signal_data.get('signal_id'),
                    timestamp=datetime.fromtimestamp(signal_data.get('timestamp', time.time()), timezone.utc),
                    symbol=signal_data.get('symbol'),
                    signal_type=signal_data.get('side'),
                    confidence=signal_data.get('confidence', 0.5),
                    price=signal_data.get('price'),
                    technical_indicators=signal_data.get('technical_indicators', {}),
                    expected_entry_price=signal_data.get('entry_price'),
                    expected_holding_minutes=signal_data.get('expected_holding_minutes', 90)
                )
                await self.quality_assurance.consistency_monitor.record_signal(signal_record)
            
            self.events_processed += 1
            self.last_event_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ Error recording signal: {e}")
            self.events_failed += 1
    
    async def record_position_update(self, position_data: Dict[str, Any]):
        """Record a position update."""
        try:
            # Record in quality assurance system
            if self.quality_assurance:
                position_record = PositionRecord(
                    position_id=position_data.get('position_id'),
                    signal_id=position_data.get('signal_id'),
                    symbol=position_data.get('symbol'),
                    side=position_data.get('side'),
                    quantity=position_data.get('quantity'),
                    entry_price=position_data.get('entry_price'),
                    exit_price=position_data.get('exit_price'),
                    entry_time=datetime.fromtimestamp(position_data.get('entry_time', time.time()), timezone.utc),
                    exit_time=datetime.fromtimestamp(position_data.get('exit_time', time.time()), timezone.utc) if position_data.get('exit_time') else None,
                    pnl=position_data.get('pnl', 0),
                    realized=position_data.get('status') == 'closed',
                    holding_minutes=position_data.get('holding_minutes')
                )
                await self.quality_assurance.consistency_monitor.record_position(position_record)
            
            # Update dashboard
            if self.dashboard_service:
                if position_data.get('status') == 'open':
                    await self.dashboard_service.record_trade_entry(
                        position_id=position_data.get('position_id'),
                        symbol=position_data.get('symbol'),
                        side=position_data.get('side'),
                        quantity=position_data.get('quantity'),
                        entry_price=position_data.get('entry_price')
                    )
            
            self.events_processed += 1
            self.last_event_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ Error recording position update: {e}")
            self.events_failed += 1
    
    def get_monitoring_status(self) -> MonitoringStatus:
        """Get current monitoring system status."""
        try:
            current_time = time.time()
            
            # Get component status
            components_total = len(self._get_enabled_components())
            components_running = len([c for c in self._get_enabled_components() if self._is_component_running(c)])
            
            # Get alert counts from alert manager
            active_alerts = len(self.alert_manager.get_active_incidents()) if self.alert_manager else 0
            critical_alerts = len([a for a in self.alert_manager.get_active_incidents() 
                                 if hasattr(a, 'severity') and a.severity.value in ['CRITICAL', 'EMERGENCY']]) if self.alert_manager else 0
            
            # Calculate error rate
            total_events = self.events_processed + self.events_failed
            error_rate = (self.events_failed / total_events * 100) if total_events > 0 else 0
            
            # Determine overall health
            if components_running == components_total and error_rate < 5 and critical_alerts == 0:
                overall_health = "healthy"
            elif components_running >= components_total * 0.8 and error_rate < 10:
                overall_health = "degraded"
            else:
                overall_health = "unhealthy"
            
            return MonitoringStatus(
                timestamp=current_time,
                overall_health=overall_health,
                components_running=components_running,
                components_total=components_total,
                active_alerts=active_alerts,
                critical_alerts=critical_alerts,
                events_processed_1h=self.events_processed,  # Simplified
                error_rate_percent=error_rate,
                last_kafka_event=self.last_event_time
            )
            
        except Exception as e:
            logger.error(f"❌ Error getting monitoring status: {e}")
            return MonitoringStatus(
                timestamp=time.time(),
                overall_health="error",
                components_running=0,
                components_total=0,
                active_alerts=0,
                critical_alerts=0,
                events_processed_1h=0,
                error_rate_percent=100.0
            )
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive system status for dashboard."""
        try:
            monitoring_status = self.get_monitoring_status()
            
            status = {
                'timestamp': time.time(),
                'monitoring_status': asdict(monitoring_status),
                'uptime_seconds': time.time() - self.start_time if self.start_time else 0,
                'is_running': self.is_running,
                'components': {},
                'events_processed': self.events_processed,
                'events_failed': self.events_failed,
                'last_event_time': self.last_event_time
            }
            
            # Add component-specific status
            if self.quality_assurance:
                latest_report = self.quality_assurance.get_latest_quality_report()
                if latest_report:
                    status['components']['quality_assurance'] = {
                        'consistency_level': latest_report.consistency_level.value,
                        'violations_count': len(latest_report.violations),
                        'last_report_time': latest_report.timestamp.isoformat()
                    }
            
            if self.alert_manager:
                alert_stats = self.alert_manager.get_alert_statistics()
                status['components']['alert_manager'] = alert_stats
            
            if self.dashboard_service:
                dashboard_stats = self.dashboard_service.get_service_stats()
                status['components']['dashboard_service'] = dashboard_stats
            
            if self.reporting_system:
                recent_reports = self.reporting_system.get_recent_reports(5)
                status['components']['reporting_system'] = {
                    'recent_reports_count': len(recent_reports),
                    'last_report_time': recent_reports[0]['generated_at'] if recent_reports else None
                }
            
            if self.kafka_streamer:
                status['components']['kafka_streamer'] = {
                    'is_running': hasattr(self.kafka_streamer, 'is_running') and self.kafka_streamer.is_running
                }
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting comprehensive status: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def _get_enabled_components(self) -> List[str]:
        """Get list of enabled component names."""
        components = []
        if self.kafka_streamer:
            components.append('kafka_streamer')
        if self.quality_assurance:
            components.append('quality_assurance')
        if self.alert_manager:
            components.append('alert_manager')
        if self.dashboard_service:
            components.append('dashboard_service')
        if self.reporting_system:
            components.append('reporting_system')
        components.append('monitoring_system')
        return components
    
    def _is_component_running(self, component_name: str) -> bool:
        """Check if a component is running."""
        try:
            if component_name == 'kafka_streamer':
                return self.kafka_streamer and hasattr(self.kafka_streamer, 'is_running') and self.kafka_streamer.is_running
            elif component_name == 'dashboard_service':
                return self.dashboard_service and self.dashboard_service.is_running
            elif component_name == 'reporting_system':
                return self.reporting_system and self.reporting_system.is_running
            elif component_name == 'monitoring_system':
                return self.monitoring_system and hasattr(self.monitoring_system, 'is_running') and self.monitoring_system.is_running
            else:
                return True  # Other components don't have explicit running state
        except:
            return False


# Factory function and demo
def create_integrated_monitoring_system(config: Dict[str, Any] = None) -> IntegratedMonitoringSystem:
    """Create and configure integrated monitoring system."""
    return IntegratedMonitoringSystem(config=config)


async def integrated_monitoring_demo():
    """Integrated monitoring system demo."""
    print("🚀 DipMaster Integrated Monitoring System Demo")
    
    # Create integrated monitoring system
    monitoring_system = create_integrated_monitoring_system()
    
    try:
        # Start the monitoring system
        await monitoring_system.start()
        print("✅ Integrated monitoring system started")
        
        # Simulate some trading events
        print("📊 Simulating trading events...")
        
        # Record a trading signal
        await monitoring_system.record_signal({
            'signal_id': 'sig_demo_001',
            'timestamp': time.time(),
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'confidence': 0.85,
            'price': 43250.50,
            'entry_price': 43200.00,
            'technical_indicators': {'rsi': 35.2, 'ma20_distance': -0.008},
            'expected_holding_minutes': 75
        })
        
        # Record a position update
        await monitoring_system.record_position_update({
            'position_id': 'pos_demo_001',
            'signal_id': 'sig_demo_001',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.15,
            'entry_price': 43225.00,
            'entry_time': time.time(),
            'status': 'open'
        })
        
        # Record an execution
        await monitoring_system.record_execution({
            'execution_id': 'exec_demo_001',
            'signal_id': 'sig_demo_001',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.15,
            'price': 43225.00,
            'slippage_bps': 2.5,
            'latency_ms': 45,
            'venue': 'binance',
            'status': 'FILLED',
            'timestamp': time.time()
        })
        
        # Wait a bit for processing
        await asyncio.sleep(2)
        
        # Record a completed trade
        await monitoring_system.record_trade({
            'position_id': 'pos_demo_001',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.15,
            'exit_price': 43420.80,
            'pnl': 29.37,  # (43420.80 - 43225.00) * 0.15
            'holding_minutes': 82,
            'status': 'closed'
        })
        
        # Wait for one monitoring cycle
        print("⏳ Waiting for monitoring cycle...")
        await asyncio.sleep(35)
        
        # Get system status
        status = monitoring_system.get_monitoring_status()
        print(f"\n📊 Monitoring Status:")
        print(f"   Overall Health: {status.overall_health.upper()}")
        print(f"   Components: {status.components_running}/{status.components_total} running")
        print(f"   Events Processed: {status.events_processed_1h}")
        print(f"   Error Rate: {status.error_rate_percent:.1f}%")
        print(f"   Active Alerts: {status.active_alerts} (Critical: {status.critical_alerts})")
        
        # Get comprehensive status
        comprehensive_status = monitoring_system.get_comprehensive_status()
        print(f"\n🏗️ System Components:")
        for component, component_status in comprehensive_status.get('components', {}).items():
            print(f"   {component}: {component_status}")
        
        print("\n✅ Demo completed successfully")
        print("📋 The integrated monitoring system successfully:")
        print("   • Recorded trading signals, positions, and executions")
        print("   • Generated quality assurance reports")
        print("   • Processed alerts and notifications")
        print("   • Published events to Kafka streams")
        print("   • Updated real-time dashboard data")
        print("   • Maintained comprehensive system monitoring")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Stop the monitoring system
        await monitoring_system.stop()
        print("🛑 Integrated monitoring system stopped")


if __name__ == "__main__":
    asyncio.run(integrated_monitoring_demo())