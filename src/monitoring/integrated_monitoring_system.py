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
from pathlib import Path
import threading

# Import monitoring components
from .kafka_event_producer import KafkaEventProducer, ExecutionReportEvent, RiskMetricsEvent, AlertEvent, StrategyPerformanceEvent
from .kafka_event_consumer import KafkaEventConsumer
from .consistency_monitor import ConsistencyMonitor, SignalRecord, PositionRecord, ExecutionRecord
from .risk_monitor import RiskMonitor, RiskLimits
from .strategy_drift_detector import StrategyDriftDetector
from .performance_monitor import PerformanceMonitor, TradeRecord
from .system_health_monitor import SystemHealthMonitor
from .structured_logger import StructuredLogger, LogContext, LogCategory
from .alert_manager import AlertManager
from .metrics_collector import MetricsCollector

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
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize integrated monitoring system.
        
        Args:
            config_file: Path to monitoring configuration file
        """
        # Load configuration
        self.config = self._load_config(config_file)
        
        # System state
        self.is_running = False
        self.start_time = None
        
        # Initialize core components
        self._initialize_components()
        
        # Event tracking
        self.events_processed = 0
        self.events_failed = 0
        self.last_event_time = None
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Health status cache
        self._health_status_cache = None
        self._last_health_check = 0
        
        logger.info("🎯 IntegratedMonitoringSystem initialized")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load monitoring configuration."""
        try:
            if config_file and Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                logger.info(f"📋 Loaded config from {config_file}")
                return config
            else:
                # Default configuration
                logger.warning("⚠️ Using default monitoring configuration")
                return {
                    "monitoring": {"enabled": True, "update_interval_seconds": 30},
                    "kafka": {"enabled": True, "servers": ["localhost:9092"]},
                    "logging": {"enabled": True, "log_directory": "logs"}
                }
        except Exception as e:
            logger.error(f"❌ Failed to load config: {e}")
            return {"monitoring": {"enabled": True}}
    
    def _initialize_components(self):
        """Initialize all monitoring components."""
        try:
            # Structured logging
            if self.config.get('logging', {}).get('enabled', True):
                self.logger = StructuredLogger(
                    config=self.config.get('logging', {}),
                    log_directory=self.config.get('logging', {}).get('log_directory', 'logs'),
                    component_name="integrated_monitoring"
                )
                logger.info("📝 Structured logger initialized")
            else:
                self.logger = None
            
            # Metrics collector
            self.metrics_collector = MetricsCollector(
                retention_hours=self.config.get('monitoring', {}).get('retention_hours', 24)
            )
            
            # Alert manager
            self.alert_manager = AlertManager(
                metrics_collector=self.metrics_collector,
                config_file=None  # Will use embedded config
            )
            
            # Kafka event producer
            if self.config.get('kafka', {}).get('enabled', True):
                self.kafka_producer = KafkaEventProducer(
                    bootstrap_servers=self.config.get('kafka', {}).get('servers', ['localhost:9092']),
                    client_id=self.config.get('kafka', {}).get('client_id', 'dipmaster-producer'),
                    config=self.config.get('kafka', {})
                )
                
                # Kafka event consumer
                self.kafka_consumer = KafkaEventConsumer(
                    bootstrap_servers=self.config.get('kafka', {}).get('servers', ['localhost:9092']),
                    group_id=self.config.get('kafka', {}).get('consumer_config', {}).get('group_id', 'dipmaster-consumer'),
                    topics=list(self.config.get('kafka', {}).get('topics', {}).values()),
                    config=self.config.get('kafka', {})
                )
                logger.info("📤📥 Kafka producer and consumer initialized")
            else:
                self.kafka_producer = None
                self.kafka_consumer = None
            
            # Consistency monitor
            if self.config.get('consistency_monitoring', {}).get('enabled', True):
                self.consistency_monitor = ConsistencyMonitor(
                    config=self.config.get('consistency_monitoring', {}),
                    retention_hours=self.config.get('monitoring', {}).get('retention_hours', 168)
                )
                logger.info("🔄 Consistency monitor initialized")
            else:
                self.consistency_monitor = None
            
            # Risk monitor
            if self.config.get('risk_monitoring', {}).get('enabled', True):
                risk_limits_config = self.config.get('risk_monitoring', {}).get('risk_limits', {})
                risk_limits = RiskLimits(**risk_limits_config) if risk_limits_config else RiskLimits()
                
                self.risk_monitor = RiskMonitor(
                    limits=risk_limits,
                    config=self.config.get('risk_monitoring', {})
                )
                
                # Add risk alert callback
                self.risk_monitor.add_alert_callback(self._handle_risk_alert)
                logger.info("🛡️ Risk monitor initialized")
            else:
                self.risk_monitor = None
            
            # Strategy drift detector
            if self.config.get('drift_detection', {}).get('enabled', True):
                self.drift_detector = StrategyDriftDetector(
                    config=self.config.get('drift_detection', {}),
                    detection_window_hours=self.config.get('drift_detection', {}).get('detection_window_hours', 24),
                    baseline_window_hours=self.config.get('drift_detection', {}).get('baseline_window_hours', 168)
                )
                logger.info("📈 Strategy drift detector initialized")
            else:
                self.drift_detector = None
            
            # Performance monitor
            if self.config.get('performance_monitoring', {}).get('enabled', True):
                self.performance_monitor = PerformanceMonitor(
                    config=self.config.get('performance_monitoring', {}),
                    benchmark_targets=self.config.get('performance_monitoring', {}).get('benchmark_targets', {})
                )
                logger.info("📊 Performance monitor initialized")
            else:
                self.performance_monitor = None
            
            # System health monitor
            if self.config.get('system_health', {}).get('enabled', True):
                self.system_health_monitor = SystemHealthMonitor(
                    config=self.config.get('system_health', {})
                )
                
                # Add health alert callback
                self.system_health_monitor.add_health_alert_callback(self._handle_health_alert)
                logger.info("🏥 System health monitor initialized")
            else:
                self.system_health_monitor = None
            
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
            
            # Start Kafka components
            if self.kafka_producer:
                await self.kafka_producer.connect()
            
            if self.kafka_consumer:
                await self.kafka_consumer.start()
            
            # Start system health monitoring
            if self.system_health_monitor:
                await self.system_health_monitor.start_monitoring()
            
            # Start alert processing
            self.alert_manager.start_processing()
            
            # Record startup metrics
            self.metrics_collector.record_value('monitoring.system_starts', 1)
            self.metrics_collector.set_gauge('monitoring.system_running', 1)
            
            # Log startup event
            if self.logger:
                context = self.logger.create_context(component="integrated_monitoring")
                self.logger.log_audit(
                    "Integrated monitoring system started",
                    context,
                    data={
                        'components_enabled': self._get_enabled_components(),
                        'config_loaded': bool(self.config)
                    }
                )
            
            self.is_running = True
            logger.info("✅ Integrated monitoring system started successfully")
            
            # Start main monitoring loop
            asyncio.create_task(self._monitoring_loop())
            
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
            
            # Stop Kafka components
            if self.kafka_producer:
                await self.kafka_producer.disconnect()
            
            if self.kafka_consumer:
                await self.kafka_consumer.stop()
            
            # Stop system health monitoring
            if self.system_health_monitor:
                await self.system_health_monitor.stop_monitoring()
            
            # Stop alert processing
            self.alert_manager.stop_processing()
            
            # Record shutdown metrics
            self.metrics_collector.set_gauge('monitoring.system_running', 0)
            self.metrics_collector.record_value('monitoring.system_uptime_seconds', uptime)
            
            # Log shutdown event
            if self.logger:
                context = self.logger.create_context(component="integrated_monitoring")
                self.logger.log_audit(
                    "Integrated monitoring system stopped",
                    context,
                    data={
                        'uptime_seconds': uptime,
                        'events_processed': self.events_processed,
                        'events_failed': self.events_failed,
                        'graceful_shutdown': True
                    }
                )
            
            self.is_running = False
            logger.info("✅ Integrated monitoring system stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping monitoring system: {e}")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("🔄 Starting main monitoring loop")
        
        update_interval = self.config.get('monitoring', {}).get('update_interval_seconds', 30)
        
        while self.is_running:
            try:
                # Run monitoring tasks
                await self._run_monitoring_cycle()
                
                # Sleep until next cycle
                await asyncio.sleep(update_interval)
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Longer sleep on error
    
    async def _run_monitoring_cycle(self):
        """Run a single monitoring cycle."""
        try:
            current_time = time.time()
            
            # Calculate risk metrics
            if self.risk_monitor:
                risk_metrics = self.risk_monitor.calculate_risk_metrics()
                risk_alerts = self.risk_monitor.check_risk_limits()
                
                # Publish risk metrics to Kafka
                if self.kafka_producer and risk_metrics:
                    risk_event = RiskMetricsEvent(
                        timestamp=risk_metrics.timestamp,
                        portfolio_id="main_portfolio",
                        var_95=risk_metrics.var_95,
                        var_99=risk_metrics.var_99,
                        expected_shortfall=risk_metrics.expected_shortfall,
                        sharpe_ratio=0.0,  # Would be calculated from performance
                        max_drawdown=risk_metrics.max_drawdown,
                        leverage=risk_metrics.leverage,
                        correlation_stability=0.0,  # Would be calculated
                        active_positions=risk_metrics.active_positions,
                        total_exposure=risk_metrics.total_exposure,
                        unrealized_pnl=risk_metrics.unrealized_pnl
                    )
                    await self.kafka_producer.publish_risk_metrics(risk_event)
            
            # Run drift detection
            if self.drift_detector:
                drift_alerts = self.drift_detector.run_drift_detection()
                
                # Process drift alerts
                for alert in drift_alerts:
                    await self._process_drift_alert(alert)
            
            # Calculate performance metrics
            if self.performance_monitor:
                performance_metrics = self.performance_monitor.calculate_performance_metrics()
                
                # Publish performance metrics
                if self.kafka_producer and performance_metrics:
                    perf_event = StrategyPerformanceEvent(
                        timestamp=current_time,
                        strategy_name="dipmaster",
                        timeframe="live",
                        win_rate=performance_metrics.win_rate,
                        total_trades=performance_metrics.total_trades,
                        winning_trades=performance_metrics.winning_trades,
                        losing_trades=performance_metrics.losing_trades,
                        avg_win=performance_metrics.avg_win,
                        avg_loss=performance_metrics.avg_loss,
                        profit_factor=performance_metrics.profit_factor,
                        sharpe_ratio=performance_metrics.sharpe_ratio,
                        max_drawdown=performance_metrics.max_drawdown,
                        total_pnl=performance_metrics.total_return_usd,
                        active_positions=0,  # Would be from position tracker
                        signals_generated=0,  # Would be tracked
                        signals_executed=0,  # Would be tracked
                        execution_rate=0.0  # Would be calculated
                    )
                    await self.kafka_producer.publish_strategy_performance(perf_event)
            
            # Update system metrics
            self.metrics_collector.record_value('monitoring.cycle_count', 1)
            self.metrics_collector.set_gauge('monitoring.last_cycle_time', current_time)
            
            # Log monitoring cycle
            if self.logger:
                context = self.logger.create_context(component="monitoring_cycle")
                self.logger.log_info(
                    "Monitoring cycle completed",
                    context,
                    category=LogCategory.MONITORING,
                    data={
                        'cycle_time': current_time,
                        'components_active': len(self._get_enabled_components())
                    }
                )
            
        except Exception as e:
            logger.error(f"❌ Error in monitoring cycle: {e}")
            if self.logger:
                context = self.logger.create_context(component="monitoring_cycle")
                self.logger.log_error(
                    "Monitoring cycle failed",
                    context,
                    category=LogCategory.MONITORING,
                    exception=e
                )
    
    async def _process_drift_alert(self, drift_alert):
        """Process strategy drift alert."""
        try:
            # Create alert event
            alert_event = AlertEvent(
                timestamp=drift_alert.timestamp,
                alert_id=drift_alert.alert_id,
                severity=drift_alert.severity.value,
                category="strategy_drift",
                message=drift_alert.description,
                affected_systems=["strategy_engine"],
                recommended_action=drift_alert.recommendation,
                auto_remediation=drift_alert.auto_action,
                source="drift_detector"
            )
            
            # Publish to Kafka
            if self.kafka_producer:
                await self.kafka_producer.publish_alert(alert_event)
            
            # Log alert
            if self.logger:
                context = self.logger.create_context(component="drift_detector")
                self.logger.log_warning(
                    f"Strategy drift detected: {drift_alert.description}",
                    context,
                    category=LogCategory.MONITORING,
                    data={
                        'drift_type': drift_alert.drift_type.value,
                        'severity': drift_alert.severity.value,
                        'metric_name': drift_alert.metric_name,
                        'drift_percentage': drift_alert.drift_percentage
                    }
                )
            
            self.events_processed += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing drift alert: {e}")
            self.events_failed += 1
    
    def _handle_risk_alert(self, risk_alert):
        """Handle risk alert from risk monitor."""
        try:
            # Create alert event
            alert_event = AlertEvent(
                timestamp=risk_alert.timestamp,
                alert_id=risk_alert.alert_id,
                severity=risk_alert.risk_level.value,
                category="risk_management",
                message=risk_alert.message,
                affected_systems=["risk_engine", "trading_engine"],
                recommended_action=risk_alert.recommended_action,
                auto_remediation=risk_alert.auto_remediation,
                source="risk_monitor"
            )
            
            # Publish to Kafka (in background)
            if self.kafka_producer:
                asyncio.create_task(self.kafka_producer.publish_alert(alert_event))
            
            # Log critical risk alerts
            if risk_alert.risk_level.value in ['critical', 'emergency']:
                if self.logger:
                    context = self.logger.create_context(component="risk_monitor")
                    self.logger.log_critical(
                        f"Critical risk alert: {risk_alert.message}",
                        context,
                        category=LogCategory.RISK,
                        data={
                            'alert_type': risk_alert.alert_type.value,
                            'current_value': risk_alert.current_value,
                            'limit_value': risk_alert.limit_value,
                            'breach_percentage': risk_alert.breach_percentage
                        }
                    )
            
            self.events_processed += 1
            
        except Exception as e:
            logger.error(f"❌ Error handling risk alert: {e}")
            self.events_failed += 1
    
    def _handle_health_alert(self, component_name: str, component_health):
        """Handle health alert from system health monitor."""
        try:
            # Create alert event
            alert_event = AlertEvent(
                timestamp=component_health.last_check_time,
                alert_id=f"health_{component_name}_{int(time.time())}",
                severity=component_health.status.value,
                category="system_health",
                message=f"Component {component_name} is {component_health.status.value}: {component_health.error_message or 'No details'}",
                affected_systems=[component_name],
                recommended_action=f"Investigate {component_name} component health",
                auto_remediation=False,
                source="health_monitor"
            )
            
            # Publish to Kafka (in background)
            if self.kafka_producer:
                asyncio.create_task(self.kafka_producer.publish_alert(alert_event))
            
            # Log health alerts
            if self.logger:
                context = self.logger.create_context(component="health_monitor")
                self.logger.log_warning(
                    f"Health alert for {component_name}: {component_health.status.value}",
                    context,
                    category=LogCategory.SYSTEM,
                    data={
                        'component_type': component_health.component_type.value,
                        'response_time_ms': component_health.response_time_ms,
                        'consecutive_failures': component_health.consecutive_failures,
                        'uptime_percentage': component_health.uptime_percentage
                    }
                )
            
            self.events_processed += 1
            
        except Exception as e:
            logger.error(f"❌ Error handling health alert: {e}")
            self.events_failed += 1
    
    # Public API methods for recording events
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record a completed trade."""
        try:
            # Record in performance monitor
            if self.performance_monitor:
                self.performance_monitor.record_trade(trade_data)
            
            # Log trade
            if self.logger:
                context = self.logger.create_context(
                    component="trading_engine",
                    request_id=trade_data.get('trade_id')
                )
                self.logger.log_trading(
                    f"Trade completed: {trade_data.get('symbol')} {trade_data.get('side')}",
                    context,
                    data=trade_data,
                    symbol=trade_data.get('symbol'),
                    trade_id=trade_data.get('trade_id'),
                    strategy=trade_data.get('strategy', 'dipmaster')
                )
            
            self.events_processed += 1
            self.last_event_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ Error recording trade: {e}")
            self.events_failed += 1
    
    def record_execution(self, execution_data: Dict[str, Any]):
        """Record an order execution."""
        try:
            # Record in consistency monitor
            if self.consistency_monitor:
                execution_record = ExecutionRecord(
                    execution_id=execution_data.get('execution_id'),
                    signal_id=execution_data.get('signal_id'),
                    position_id=execution_data.get('position_id'),
                    timestamp=execution_data.get('timestamp', time.time()),
                    symbol=execution_data.get('symbol'),
                    side=execution_data.get('side'),
                    quantity=execution_data.get('quantity'),
                    price=execution_data.get('price'),
                    slippage_bps=execution_data.get('slippage_bps', 0),
                    latency_ms=execution_data.get('latency_ms', 0),
                    status=execution_data.get('status'),
                    venue=execution_data.get('venue'),
                    fees=execution_data.get('fees')
                )
                self.consistency_monitor.record_execution(execution_record)
            
            # Publish to Kafka
            if self.kafka_producer:
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
                    venue=execution_data.get('venue'),
                    status=execution_data.get('status'),
                    fill_qty=execution_data.get('fill_qty'),
                    fill_price=execution_data.get('fill_price'),
                    fees=execution_data.get('fees'),
                    strategy=execution_data.get('strategy'),
                    session_id=execution_data.get('session_id')
                )
                asyncio.create_task(self.kafka_producer.publish_execution_report(exec_event))
            
            self.events_processed += 1
            self.last_event_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ Error recording execution: {e}")
            self.events_failed += 1
    
    def record_signal(self, signal_data: Dict[str, Any]):
        """Record a trading signal."""
        try:
            # Record in consistency monitor
            if self.consistency_monitor:
                signal_record = SignalRecord(
                    signal_id=signal_data.get('signal_id'),
                    timestamp=signal_data.get('timestamp', time.time()),
                    symbol=signal_data.get('symbol'),
                    side=signal_data.get('side'),
                    confidence=signal_data.get('confidence', 0),
                    parameters=signal_data.get('parameters', {}),
                    strategy_version=signal_data.get('strategy_version', '1.0'),
                    entry_price=signal_data.get('entry_price'),
                    stop_loss=signal_data.get('stop_loss'),
                    take_profit=signal_data.get('take_profit'),
                    position_size=signal_data.get('position_size')
                )
                self.consistency_monitor.record_signal(signal_record)
            
            # Record in drift detector
            if self.drift_detector:
                self.drift_detector.record_feature_values(
                    signal_data.get('features', {})
                )
            
            self.events_processed += 1
            self.last_event_time = time.time()
            
        except Exception as e:
            logger.error(f"❌ Error recording signal: {e}")
            self.events_failed += 1
    
    def record_position_update(self, position_data: Dict[str, Any]):
        """Record a position update."""
        try:
            # Record in consistency monitor
            if self.consistency_monitor:
                position_record = PositionRecord(
                    position_id=position_data.get('position_id'),
                    signal_id=position_data.get('signal_id'),
                    timestamp=position_data.get('timestamp', time.time()),
                    symbol=position_data.get('symbol'),
                    side=position_data.get('side'),
                    entry_price=position_data.get('entry_price'),
                    quantity=position_data.get('quantity'),
                    status=position_data.get('status'),
                    current_price=position_data.get('current_price'),
                    unrealized_pnl=position_data.get('unrealized_pnl'),
                    realized_pnl=position_data.get('realized_pnl')
                )
                self.consistency_monitor.record_position(position_record)
            
            # Update risk monitor
            if self.risk_monitor:
                self.risk_monitor.update_position(
                    position_data.get('symbol'),
                    position_data
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
            
            # Get alert counts
            active_alerts = len(self.alert_manager.get_active_alerts()) if self.alert_manager else 0
            critical_alerts = len([a for a in self.alert_manager.get_active_alerts() 
                                 if a.get('level') == 'critical']) if self.alert_manager else 0
            
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
                'components': {}
            }
            
            # Add component-specific status
            if self.consistency_monitor:
                status['components']['consistency'] = self.consistency_monitor.get_monitoring_summary()
            
            if self.risk_monitor:
                status['components']['risk'] = self.risk_monitor.get_risk_summary()
            
            if self.drift_detector:
                status['components']['drift'] = self.drift_detector.get_drift_summary()
            
            if self.performance_monitor:
                status['components']['performance'] = self.performance_monitor.get_performance_summary()
            
            if self.system_health_monitor:
                status['components']['health'] = self.system_health_monitor.get_health_summary()
            
            if self.kafka_producer:
                status['components']['kafka_producer'] = self.kafka_producer.get_producer_stats()
            
            if self.kafka_consumer:
                status['components']['kafka_consumer'] = self.kafka_consumer.get_consumer_stats()
            
            if self.logger:
                status['components']['logging'] = self.logger.get_log_statistics()
            
            return status
            
        except Exception as e:
            logger.error(f"❌ Error getting comprehensive status: {e}")
            return {'error': str(e), 'timestamp': time.time()}
    
    def _get_enabled_components(self) -> List[str]:
        """Get list of enabled component names."""
        components = []
        if self.kafka_producer:
            components.append('kafka_producer')
        if self.kafka_consumer:
            components.append('kafka_consumer')
        if self.consistency_monitor:
            components.append('consistency_monitor')
        if self.risk_monitor:
            components.append('risk_monitor')
        if self.drift_detector:
            components.append('drift_detector')
        if self.performance_monitor:
            components.append('performance_monitor')
        if self.system_health_monitor:
            components.append('system_health_monitor')
        if self.logger:
            components.append('structured_logger')
        components.append('alert_manager')
        components.append('metrics_collector')
        return components
    
    def _is_component_running(self, component_name: str) -> bool:
        """Check if a component is running."""
        try:
            if component_name == 'kafka_producer':
                return self.kafka_producer and self.kafka_producer.is_connected
            elif component_name == 'kafka_consumer':
                return self.kafka_consumer and self.kafka_consumer.state.value == 'running'
            elif component_name == 'system_health_monitor':
                return self.system_health_monitor and self.system_health_monitor._monitoring_active
            elif component_name == 'alert_manager':
                return self.alert_manager and self.alert_manager._processing_active
            else:
                return True  # Other components don't have explicit running state
        except:
            return False


# Factory function
def create_integrated_monitoring_system(config_file: Optional[str] = None) -> IntegratedMonitoringSystem:
    """Create and configure integrated monitoring system."""
    return IntegratedMonitoringSystem(config_file=config_file)