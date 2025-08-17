#!/usr/bin/env python3
"""
Complete Integrated Monitoring System for DipMaster Trading System
完整集成监控系统 - 统一的交易系统监控解决方案

Features:
- Unified monitoring orchestration
- Real-time event streaming with Kafka
- Structured logging with metrics extraction
- Trading consistency and strategy compliance monitoring
- Dashboard data generation and WebSocket streaming
- Comprehensive health assessment and alerting
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import threading

# Import all monitoring components
from .kafka_event_producer import create_kafka_producer
from .enhanced_event_producer import create_enhanced_event_producer
from .structured_logging_system import create_structured_logger
from .trading_consistency_monitor import create_consistency_monitor, SignalData, PositionData
from .dipmaster_strategy_monitor import create_dipmaster_monitor
from .dashboard_data_generator import create_dashboard_generator
from .metrics_collector import MetricsCollector
from .business_kpi import BusinessKPITracker
from .monitoring_service import MonitoringService

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Comprehensive monitoring configuration."""
    # Kafka configuration
    kafka_servers: List[str]
    kafka_client_id: str = "dipmaster-monitoring"
    
    # Logging configuration
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_rotation_size: int = 100 * 1024 * 1024  # 100MB
    log_retention_days: int = 30
    
    # Monitoring intervals
    metrics_collection_interval: int = 10  # seconds
    dashboard_update_interval: int = 1     # seconds
    health_check_interval: int = 60        # seconds
    
    # Alert thresholds
    critical_drawdown_threshold: float = 15.0
    error_rate_threshold: float = 10.0
    latency_threshold_ms: float = 1000.0
    
    # DipMaster specific
    dipmaster_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dipmaster_params is None:
            self.dipmaster_params = {
                'rsi_min': 30.0,
                'rsi_max': 50.0,
                'max_holding_minutes': 180,
                'boundary_minutes': [15, 30, 45, 60],
                'target_profit_pct': 0.8
            }


class CompleteMonitoringSystem:
    """
    Complete integrated monitoring system for DipMaster Trading System.
    
    Orchestrates all monitoring components to provide complete visibility
    into trading operations, risk metrics, and system health.
    """
    
    def __init__(self, config: MonitoringConfig):
        """
        Initialize complete monitoring system.
        
        Args:
            config: Monitoring configuration
        """
        self.config = config
        self.is_running = False
        self.start_time = None
        
        # Core components
        self.kafka_producer = None
        self.event_producer = None
        self.structured_logger = None
        self.metrics_collector = None
        self.business_kpi = None
        self.monitoring_service = None
        
        # Specialized monitors
        self.consistency_monitor = None
        self.strategy_monitor = None
        self.dashboard_generator = None
        
        # Background tasks
        self.background_tasks = []
        
        # Statistics
        self.stats = {
            'events_published': 0,
            'logs_processed': 0,
            'alerts_generated': 0,
            'health_checks': 0,
            'start_time': None,
            'uptime_seconds': 0
        }
        
        logger.info("🚀 CompleteMonitoringSystem initializing...")
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all monitoring components."""
        try:
            # 1. Initialize Kafka and event production
            self._init_event_system()
            
            # 2. Initialize structured logging
            self._init_logging_system()
            
            # 3. Initialize metrics and KPI tracking
            self._init_metrics_system()
            
            # 4. Initialize specialized monitors
            self._init_specialized_monitors()
            
            # 5. Initialize dashboard system
            self._init_dashboard_system()
            
            # 6. Initialize monitoring service
            self._init_monitoring_service()
            
            logger.info("✅ All monitoring components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize monitoring components: {e}")
            raise
    
    def _init_event_system(self):
        """Initialize Kafka and event production system."""
        try:
            # Create Kafka producer
            kafka_config = {
                'kafka_servers': self.config.kafka_servers,
                'client_id': self.config.kafka_client_id,
                'producer_config': {
                    'retries': 3,
                    'batch_size': 16384,
                    'linger_ms': 10,
                    'compression_type': 'gzip'
                }
            }
            
            self.kafka_producer = create_kafka_producer(kafka_config)
            
            # Create enhanced event producer
            event_config = {
                'buffer_max_size': 1000,
                'buffer_flush_interval': 1.0
            }
            
            self.event_producer = create_enhanced_event_producer(
                self.kafka_producer, event_config
            )
            
            logger.info("📤 Event production system initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize event system: {e}")
            # Continue without Kafka (mock mode)
            self.kafka_producer = None
            self.event_producer = None
    
    def _init_logging_system(self):
        """Initialize structured logging system."""
        try:
            log_config = {
                'max_file_size': self.config.log_rotation_size,
                'max_files': self.config.log_retention_days,
                'compression': True
            }
            
            self.structured_logger = create_structured_logger(
                name="dipmaster_monitoring",
                log_dir=self.config.log_dir,
                config=log_config
            )
            
            logger.info("📝 Structured logging system initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize logging system: {e}")
            self.structured_logger = None
    
    def _init_metrics_system(self):
        """Initialize metrics collection and KPI tracking."""
        try:
            # Create metrics collector
            self.metrics_collector = MetricsCollector(
                retention_hours=24,
                cleanup_interval=300
            )
            
            # Create business KPI tracker
            self.business_kpi = BusinessKPITracker(
                metrics_collector=self.metrics_collector,
                history_retention_hours=720  # 30 days
            )
            
            logger.info("📊 Metrics and KPI systems initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize metrics system: {e}")
            raise
    
    def _init_specialized_monitors(self):
        """Initialize specialized monitoring components."""
        try:
            # Trading consistency monitor
            self.consistency_monitor = create_consistency_monitor(
                kafka_producer=self.event_producer,
                config={
                    'thresholds': {
                        'signal_position_match': 95.0,
                        'position_execution_match': 98.0,
                        'price_deviation_bps': 20.0,
                        'timing_deviation_minutes': 2.0
                    }
                }
            )
            
            # DipMaster strategy monitor
            self.strategy_monitor = create_dipmaster_monitor(
                event_producer=self.event_producer,
                structured_logger=self.structured_logger,
                config={'dipmaster_params': self.config.dipmaster_params}
            )
            
            logger.info("🔍 Specialized monitors initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize specialized monitors: {e}")
            raise
    
    def _init_dashboard_system(self):
        """Initialize dashboard data generation system."""
        try:
            dashboard_config = {
                'max_chart_points': 1000,
                'update_interval': self.config.dashboard_update_interval,
                'cache_ttl': 5
            }
            
            self.dashboard_generator = create_dashboard_generator(
                metrics_collector=self.metrics_collector,
                business_kpi=self.business_kpi,
                event_producer=self.event_producer,
                config=dashboard_config
            )
            
            logger.info("📈 Dashboard system initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize dashboard system: {e}")
            self.dashboard_generator = None
    
    def _init_monitoring_service(self):
        """Initialize unified monitoring service."""
        try:
            monitoring_config = {
                'metrics_retention_hours': 24,
                'kpi_retention_hours': 720,
                'system_monitoring_interval': 30
            }
            
            self.monitoring_service = MonitoringService(
                config=monitoring_config
            )
            
            logger.info("🎯 Monitoring service initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize monitoring service: {e}")
            self.monitoring_service = None
    
    async def start(self):
        """Start the complete monitoring system."""
        if self.is_running:
            logger.warning("⚠️ Monitoring system already running")
            return
        
        try:
            self.is_running = True
            self.start_time = time.time()
            self.stats['start_time'] = self.start_time
            
            logger.info("🚀 Starting complete monitoring system...")
            
            # Start Kafka connection
            if self.kafka_producer:
                await self.kafka_producer.connect()
            
            # Start event producer
            if self.event_producer:
                await self.event_producer.start_processing()
            
            # Start structured logging
            if self.structured_logger:
                await self.structured_logger.start()
            
            # Start dashboard updates
            if self.dashboard_generator:
                await self.dashboard_generator.start_updates()
            
            # Start monitoring service
            if self.monitoring_service:
                await self.monitoring_service.start()
            
            # Start background tasks
            self.background_tasks = [
                asyncio.create_task(self._metrics_collection_loop()),
                asyncio.create_task(self._health_check_loop()),
                asyncio.create_task(self._statistics_update_loop())
            ]
            
            # Record startup event
            await self._record_system_event("MONITORING_SYSTEM_STARTED", {
                'components': self._get_active_components(),
                'config': {
                    'kafka_enabled': self.kafka_producer is not None,
                    'logging_enabled': self.structured_logger is not None,
                    'dashboard_enabled': self.dashboard_generator is not None
                }
            })
            
            logger.info("✅ Complete monitoring system started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring system: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """Stop the complete monitoring system."""
        if not self.is_running:
            logger.warning("⚠️ Monitoring system not running")
            return
        
        try:
            logger.info("🛑 Stopping complete monitoring system...")
            
            self.is_running = False
            uptime = time.time() - self.start_time if self.start_time else 0
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Stop components
            if self.monitoring_service:
                await self.monitoring_service.stop()
            
            if self.dashboard_generator:
                await self.dashboard_generator.stop_updates()
            
            if self.structured_logger:
                await self.structured_logger.stop()
            
            if self.event_producer:
                await self.event_producer.stop_processing()
            
            if self.kafka_producer:
                await self.kafka_producer.disconnect()
            
            # Record shutdown event
            await self._record_system_event("MONITORING_SYSTEM_STOPPED", {
                'uptime_seconds': uptime,
                'stats': self.stats
            })
            
            logger.info("✅ Complete monitoring system stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping monitoring system: {e}")
    
    async def _metrics_collection_loop(self):
        """Background loop for metrics collection."""
        while self.is_running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Update business KPIs
                await self._update_business_kpis()
                
                await asyncio.sleep(self.config.metrics_collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in metrics collection loop: {e}")
                await asyncio.sleep(1)
    
    async def _health_check_loop(self):
        """Background loop for health checks."""
        while self.is_running:
            try:
                await self._perform_health_check()
                self.stats['health_checks'] += 1
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in health check loop: {e}")
                await asyncio.sleep(1)
    
    async def _statistics_update_loop(self):
        """Background loop for statistics updates."""
        while self.is_running:
            try:
                # Update uptime
                if self.start_time:
                    self.stats['uptime_seconds'] = time.time() - self.start_time
                
                # Collect component statistics
                if self.event_producer:
                    event_stats = self.event_producer.get_performance_stats()
                    self.stats['events_published'] = sum(
                        count for key, count in event_stats.get('event_stats', {}).items()
                        if 'success' in key
                    )
                
                if self.structured_logger:
                    log_stats = self.structured_logger.get_stats()
                    self.stats['logs_processed'] = log_stats.get('logs_processed', 0)
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in statistics update loop: {e}")
                await asyncio.sleep(1)
    
    async def _collect_system_metrics(self):
        """Collect system performance metrics."""
        try:
            import psutil
            
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Record metrics
            current_time = time.time()
            self.metrics_collector.record_value('system.cpu.usage_percent', cpu_percent, current_time)
            self.metrics_collector.record_value('system.memory.usage_percent', memory.percent, current_time)
            self.metrics_collector.record_value('system.disk.usage_percent', disk.percent, current_time)
            
            # Network metrics (if available)
            try:
                net_io = psutil.net_io_counters()
                self.metrics_collector.record_value('system.network.bytes_sent', net_io.bytes_sent, current_time)
                self.metrics_collector.record_value('system.network.bytes_recv', net_io.bytes_recv, current_time)
            except:
                pass
            
        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            logger.error(f"❌ Failed to collect system metrics: {e}")
    
    async def _update_business_kpis(self):
        """Update business KPIs."""
        try:
            # Get trading summary
            trading_summary = self.business_kpi.get_trading_summary(1)  # Last hour
            
            if trading_summary:
                current_time = time.time()
                
                # Record KPI metrics
                self.metrics_collector.record_value('trading.win_rate', 
                                                  trading_summary.get('win_rate', 0), current_time)
                self.metrics_collector.record_value('trading.total_pnl', 
                                                  trading_summary.get('total_pnl', 0), current_time)
                self.metrics_collector.record_value('trading.profit_factor', 
                                                  trading_summary.get('profit_factor', 0), current_time)
                self.metrics_collector.record_value('trading.sharpe_ratio', 
                                                  trading_summary.get('sharpe_ratio', 0), current_time)
            
        except Exception as e:
            logger.error(f"❌ Failed to update business KPIs: {e}")
    
    async def _perform_health_check(self):
        """Perform comprehensive health check."""
        try:
            health_components = {}
            overall_score = 100.0
            
            # Check Kafka health
            if self.kafka_producer:
                kafka_health = await self.kafka_producer.health_check()
                health_components['kafka'] = kafka_health
                if kafka_health['status'] != 'healthy':
                    overall_score -= 20
            
            # Check event producer health
            if self.event_producer:
                event_health = await self.event_producer.health_check()
                health_components['event_producer'] = event_health
                if event_health['status'] != 'healthy':
                    overall_score -= 15
            
            # Check dashboard health
            if self.dashboard_generator:
                dashboard_stats = self.dashboard_generator.get_stats()
                health_components['dashboard'] = {
                    'status': 'healthy' if dashboard_stats['is_running'] else 'unhealthy',
                    'subscribers': dashboard_stats['subscribers_count']
                }
                if not dashboard_stats['is_running']:
                    overall_score -= 10
            
            # Record overall health score
            self.metrics_collector.record_value('health.overall_score', overall_score, time.time())
            
            # Generate health alert if needed
            if overall_score < 70:
                await self._generate_health_alert(overall_score, health_components)
            
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            self.metrics_collector.record_value('health.overall_score', 0, time.time())
    
    async def _generate_health_alert(self, score: float, components: Dict[str, Any]):
        """Generate health alert for poor system health."""
        if self.event_producer:
            await self.event_producer.publish_alert(
                alert_id=f"system_health_{int(time.time())}",
                severity="CRITICAL" if score < 50 else "WARNING",
                category="SYSTEM_HEALTH",
                message=f"System health degraded: score {score:.1f}/100",
                affected_systems=list(components.keys()),
                recommended_action="Investigate component health and address issues",
                source="complete_monitoring_system",
                tags={'health_score': str(score)}
            )
    
    async def _record_system_event(self, event_type: str, details: Dict[str, Any]):
        """Record system event."""
        if self.structured_logger:
            self.structured_logger.info(
                f"System event: {event_type}",
                data=details,
                tags={'event_type': event_type}
            )
    
    def _get_active_components(self) -> List[str]:
        """Get list of active components."""
        components = []
        if self.kafka_producer:
            components.append('kafka_producer')
        if self.event_producer:
            components.append('event_producer')
        if self.structured_logger:
            components.append('structured_logger')
        if self.metrics_collector:
            components.append('metrics_collector')
        if self.business_kpi:
            components.append('business_kpi')
        if self.consistency_monitor:
            components.append('consistency_monitor')
        if self.strategy_monitor:
            components.append('strategy_monitor')
        if self.dashboard_generator:
            components.append('dashboard_generator')
        if self.monitoring_service:
            components.append('monitoring_service')
        return components
    
    # Public API methods for integration with trading system
    
    async def record_signal(self, signal_data: Dict[str, Any]):
        """Record trading signal for monitoring."""
        try:
            # Convert to SignalData
            signal = SignalData(
                signal_id=signal_data['signal_id'],
                timestamp=datetime.now(timezone.utc),
                symbol=signal_data['symbol'],
                signal_type=signal_data['signal_type'],
                confidence=signal_data['confidence'],
                price=signal_data['price'],
                rsi=signal_data['technical_indicators']['rsi'],
                ma20_distance=signal_data['technical_indicators']['ma20_distance'],
                volume_ratio=signal_data['technical_indicators']['volume_ratio'],
                expected_entry_price=signal_data.get('expected_entry_price', signal_data['price']),
                expected_holding_minutes=signal_data.get('expected_holding_minutes', 60),
                strategy_params=signal_data.get('strategy_params', {})
            )
            
            # Record in consistency monitor
            if self.consistency_monitor:
                await self.consistency_monitor.record_signal(signal)
            
            # Validate with strategy monitor
            if self.strategy_monitor:
                violations = await self.strategy_monitor.validate_signal(signal)
                if violations:
                    logger.warning(f"Signal {signal.signal_id} has {len(violations)} violations")
            
            # Publish signal event
            if self.event_producer:
                await self.event_producer.publish_trade_signal(
                    signal_id=signal.signal_id,
                    strategy="dipmaster",
                    symbol=signal.symbol,
                    signal_type=signal.signal_type,
                    confidence=signal.confidence,
                    price=signal.price,
                    technical_indicators=signal_data['technical_indicators'],
                    market_conditions=signal_data.get('market_conditions', {})
                )
            
        except Exception as e:
            logger.error(f"❌ Failed to record signal: {e}")
    
    async def record_trade_entry(self, trade_data: Dict[str, Any]):
        """Record trade entry for monitoring."""
        try:
            # Record in business KPI
            if self.business_kpi:
                self.business_kpi.record_trade_entry(
                    trade_id=trade_data['trade_id'],
                    symbol=trade_data['symbol'],
                    side=trade_data['side'],
                    quantity=trade_data['quantity'],
                    entry_price=trade_data['entry_price'],
                    strategy=trade_data.get('strategy', 'dipmaster'),
                    signal_id=trade_data.get('signal_id')
                )
            
            # Record in dashboard
            if self.dashboard_generator:
                await self.dashboard_generator.record_trade_entry(
                    position_id=trade_data['trade_id'],
                    symbol=trade_data['symbol'],
                    side=trade_data['side'],
                    quantity=trade_data['quantity'],
                    entry_price=trade_data['entry_price'],
                    strategy=trade_data.get('strategy', 'dipmaster')
                )
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade entry: {e}")
    
    async def record_trade_exit(self, trade_data: Dict[str, Any]):
        """Record trade exit for monitoring."""
        try:
            # Create PositionData for monitoring
            position = PositionData(
                position_id=trade_data['trade_id'],
                signal_id=trade_data.get('signal_id', ''),
                symbol=trade_data['symbol'],
                side=trade_data['side'],
                quantity=trade_data['quantity'],
                entry_price=trade_data['entry_price'],
                exit_price=trade_data['exit_price'],
                entry_time=trade_data.get('entry_time', datetime.now(timezone.utc)),
                exit_time=trade_data.get('exit_time', datetime.now(timezone.utc)),
                pnl=trade_data['pnl'],
                realized=True
            )
            
            # Calculate holding time
            if position.entry_time and position.exit_time:
                holding_minutes = (position.exit_time - position.entry_time).total_seconds() / 60
                position.holding_minutes = int(holding_minutes)
            
            # Record in consistency monitor
            if self.consistency_monitor:
                await self.consistency_monitor.record_position(position)
            
            # Validate with strategy monitor
            if self.strategy_monitor:
                violations = await self.strategy_monitor.validate_position_exit(position)
                if violations:
                    logger.warning(f"Position {position.position_id} has {len(violations)} exit violations")
            
            # Record in business KPI
            if self.business_kpi:
                self.business_kpi.record_trade_exit(
                    trade_id=trade_data['trade_id'],
                    exit_price=trade_data['exit_price'],
                    exit_time=trade_data.get('exit_time', datetime.now(timezone.utc)),
                    pnl=trade_data['pnl']
                )
            
            # Record in dashboard
            if self.dashboard_generator:
                await self.dashboard_generator.record_trade_exit(
                    position_id=trade_data['trade_id'],
                    exit_price=trade_data['exit_price'],
                    realized_pnl=trade_data['pnl'],
                    holding_minutes=position.holding_minutes or 0
                )
            
        except Exception as e:
            logger.error(f"❌ Failed to record trade exit: {e}")
    
    async def record_execution(self, execution_data: Dict[str, Any]):
        """Record order execution for monitoring."""
        try:
            # Publish execution event
            if self.event_producer:
                await self.event_producer.publish_execution_report(
                    symbol=execution_data['symbol'],
                    execution_id=execution_data['execution_id'],
                    signal_id=execution_data.get('signal_id', ''),
                    side=execution_data['side'],
                    quantity=execution_data['quantity'],
                    price=execution_data['price'],
                    slippage_bps=execution_data.get('slippage_bps', 0),
                    latency_ms=execution_data.get('latency_ms', 0),
                    venue=execution_data.get('venue', 'binance'),
                    status=execution_data.get('status', 'FILLED')
                )
            
        except Exception as e:
            logger.error(f"❌ Failed to record execution: {e}")
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for frontend."""
        if self.dashboard_generator:
            return await self.dashboard_generator.get_dashboard_data()
        return {'error': 'Dashboard not available'}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get system health status."""
        if self.monitoring_service:
            return self.monitoring_service.get_health_status()
        
        return {
            'status': 'degraded' if self.is_running else 'unhealthy',
            'uptime_seconds': self.stats['uptime_seconds'],
            'components': self._get_active_components()
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = dict(self.stats)
        
        # Add component stats
        if self.consistency_monitor:
            stats['consistency_monitor'] = self.consistency_monitor.get_consistency_report()
        
        if self.strategy_monitor:
            stats['strategy_monitor'] = self.strategy_monitor.get_monitoring_stats()
        
        if self.dashboard_generator:
            stats['dashboard_generator'] = self.dashboard_generator.get_stats()
        
        if self.event_producer:
            stats['event_producer'] = self.event_producer.get_performance_stats()
        
        return stats


# Factory function
def create_complete_monitoring_system(config: MonitoringConfig) -> CompleteMonitoringSystem:
    """Create and configure complete monitoring system."""
    return CompleteMonitoringSystem(config)


# Example usage demonstration
async def monitoring_system_demo():
    """Demonstrate the complete monitoring system."""
    print("🚀 DipMaster Complete Monitoring System Demo")
    
    # Configuration
    config = MonitoringConfig(
        kafka_servers=['localhost:9092'],
        log_dir='logs/monitoring',
        dipmaster_params={
            'rsi_min': 30.0,
            'rsi_max': 50.0,
            'max_holding_minutes': 180,
            'boundary_minutes': [15, 30, 45, 60]
        }
    )
    
    # Create and start monitoring system
    monitoring = create_complete_monitoring_system(config)
    
    try:
        await monitoring.start()
        print("✅ Monitoring system started")
        
        # Simulate trading events
        print("📊 Simulating trading events...")
        
        # Record a signal
        signal_data = {
            'signal_id': 'sig_123',
            'symbol': 'BTCUSDT',
            'signal_type': 'BUY',
            'confidence': 0.85,
            'price': 42150.50,
            'technical_indicators': {
                'rsi': 35.5,
                'ma20_distance': -0.005,
                'volume_ratio': 1.8
            },
            'expected_entry_price': 42100.00,
            'expected_holding_minutes': 75
        }
        await monitoring.record_signal(signal_data)
        
        # Record trade entry
        trade_entry_data = {
            'trade_id': 'trade_456',
            'signal_id': 'sig_123',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.1,
            'entry_price': 42120.00,
            'strategy': 'dipmaster'
        }
        await monitoring.record_trade_entry(trade_entry_data)
        
        # Record execution
        execution_data = {
            'execution_id': 'exec_789',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.1,
            'price': 42120.00,
            'slippage_bps': 2.5,
            'latency_ms': 45,
            'venue': 'binance',
            'status': 'FILLED'
        }
        await monitoring.record_execution(execution_data)
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Record trade exit
        trade_exit_data = {
            'trade_id': 'trade_456',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.1,
            'entry_price': 42120.00,
            'exit_price': 42450.00,
            'pnl': 33.00,  # (42450 - 42120) * 0.1
            'entry_time': datetime.now(timezone.utc) - timedelta(minutes=75),
            'exit_time': datetime.now(timezone.utc)
        }
        await monitoring.record_trade_exit(trade_exit_data)
        
        # Get dashboard data
        print("📈 Getting dashboard data...")
        dashboard_data = await monitoring.get_dashboard_data()
        print(f"Dashboard data keys: {list(dashboard_data.keys())}")
        
        # Get health status
        health_status = await monitoring.get_health_status()
        print(f"System health: {health_status['status']}")
        
        # Get system stats
        stats = monitoring.get_system_stats()
        print(f"System uptime: {stats['uptime_seconds']:.1f} seconds")
        
        print("✅ Demo completed successfully")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        
    finally:
        await monitoring.stop()
        print("🛑 Monitoring system stopped")


if __name__ == "__main__":
    asyncio.run(monitoring_system_demo())