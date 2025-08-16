#!/usr/bin/env python3
"""
Kafka Event Consumer for DipMaster Trading System
事件流消费者 - 处理和分析交易系统事件

Features:
- High-throughput event consumption from Kafka
- Event processing and analysis
- Real-time alerting and notifications
- Event storage and archival
- Consumer group management
- Metrics and monitoring
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Set
from dataclasses import dataclass
from enum import Enum
import threading
from datetime import datetime, timezone
from collections import defaultdict, deque

# Kafka imports (will gracefully degrade if not available)
try:
    from kafka import KafkaConsumer
    from kafka.errors import KafkaError, CommitFailedError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaConsumer = None
    KafkaError = Exception
    CommitFailedError = Exception

logger = logging.getLogger(__name__)


class ConsumerState(Enum):
    """Consumer state enumeration."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


@dataclass
class EventStats:
    """Event processing statistics."""
    total_processed: int = 0
    total_failed: int = 0
    bytes_processed: int = 0
    last_processed_time: Optional[float] = None
    processing_rate: float = 0.0
    error_rate: float = 0.0


class EventProcessor:
    """Base class for event processors."""
    
    def __init__(self, name: str):
        self.name = name
        self.processed_count = 0
        self.error_count = 0
    
    async def process(self, event: Dict[str, Any], headers: Dict[str, str]) -> bool:
        """Process an event. Override in subclasses."""
        raise NotImplementedError
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processor statistics."""
        return {
            'name': self.name,
            'processed': self.processed_count,
            'errors': self.error_count,
            'success_rate': self.processed_count / max(self.processed_count + self.error_count, 1)
        }


class ExecutionReportProcessor(EventProcessor):
    """Processor for execution report events."""
    
    def __init__(self):
        super().__init__("execution_report_processor")
        self.execution_metrics = defaultdict(list)
        self.slippage_history = deque(maxlen=1000)
        self.latency_history = deque(maxlen=1000)
    
    async def process(self, event: Dict[str, Any], headers: Dict[str, str]) -> bool:
        """Process execution report event."""
        try:
            # Extract key metrics
            symbol = event.get('symbol')
            slippage_bps = event.get('slippage_bps', 0)
            latency_ms = event.get('latency_ms', 0)
            fill_rate = event.get('fill_qty', 0) / event.get('quantity', 1)
            
            # Update metrics
            self.execution_metrics[symbol].append({
                'timestamp': event.get('timestamp'),
                'slippage_bps': slippage_bps,
                'latency_ms': latency_ms,
                'fill_rate': fill_rate,
                'venue': event.get('venue'),
                'status': event.get('status')
            })
            
            # Track global metrics
            self.slippage_history.append(slippage_bps)
            self.latency_history.append(latency_ms)
            
            # Check for anomalies
            await self._check_execution_anomalies(event)
            
            self.processed_count += 1
            logger.debug(f"📊 Processed execution report: {symbol} (slippage: {slippage_bps:.2f}bps)")
            
            return True
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Failed to process execution report: {e}")
            return False
    
    async def _check_execution_anomalies(self, event: Dict[str, Any]):
        """Check for execution anomalies and generate alerts."""
        slippage_bps = event.get('slippage_bps', 0)
        latency_ms = event.get('latency_ms', 0)
        
        # High slippage alert
        if slippage_bps > 50:  # 5 basis points
            logger.warning(f"⚠️ High slippage detected: {slippage_bps:.2f}bps for {event.get('symbol')}")
        
        # High latency alert
        if latency_ms > 1000:  # 1 second
            logger.warning(f"⚠️ High execution latency: {latency_ms:.0f}ms for {event.get('symbol')}")
        
        # Failed execution alert
        if event.get('status') == 'failed':
            logger.error(f"❌ Execution failed for {event.get('symbol')}: {event.get('error', 'Unknown error')}")
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get execution summary statistics."""
        if not self.slippage_history or not self.latency_history:
            return {}
        
        return {
            'avg_slippage_bps': sum(self.slippage_history) / len(self.slippage_history),
            'p95_slippage_bps': sorted(self.slippage_history)[int(len(self.slippage_history) * 0.95)],
            'avg_latency_ms': sum(self.latency_history) / len(self.latency_history),
            'p95_latency_ms': sorted(self.latency_history)[int(len(self.latency_history) * 0.95)],
            'total_executions': len(self.slippage_history),
            'symbols_traded': len(self.execution_metrics)
        }


class RiskMetricsProcessor(EventProcessor):
    """Processor for risk metrics events."""
    
    def __init__(self):
        super().__init__("risk_metrics_processor")
        self.risk_history = deque(maxlen=1000)
        self.var_breaches = []
        self.drawdown_alerts = []
    
    async def process(self, event: Dict[str, Any], headers: Dict[str, str]) -> bool:
        """Process risk metrics event."""
        try:
            # Store risk metrics
            risk_data = {
                'timestamp': event.get('timestamp'),
                'var_95': event.get('var_95'),
                'var_99': event.get('var_99'),
                'expected_shortfall': event.get('expected_shortfall'),
                'max_drawdown': event.get('max_drawdown'),
                'sharpe_ratio': event.get('sharpe_ratio'),
                'leverage': event.get('leverage'),
                'active_positions': event.get('active_positions'),
                'total_exposure': event.get('total_exposure')
            }
            
            self.risk_history.append(risk_data)
            
            # Check risk limits
            await self._check_risk_limits(event)
            
            self.processed_count += 1
            logger.debug(f"📊 Processed risk metrics: VaR95={event.get('var_95'):.2f}")
            
            return True
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Failed to process risk metrics: {e}")
            return False
    
    async def _check_risk_limits(self, event: Dict[str, Any]):
        """Check risk limits and generate alerts."""
        var_95 = event.get('var_95', 0)
        max_drawdown = event.get('max_drawdown', 0)
        leverage = event.get('leverage', 0)
        
        # VaR limit check (example: 200k USD)
        if var_95 > 200000:
            alert = {
                'timestamp': time.time(),
                'type': 'var_breach',
                'severity': 'critical',
                'message': f"VaR 95% exceeded limit: {var_95:.2f} > 200000",
                'current_value': var_95,
                'limit': 200000
            }
            self.var_breaches.append(alert)
            logger.critical(f"🚨 VaR BREACH: {alert['message']}")
        
        # Drawdown limit check (example: 20%)
        if max_drawdown > 0.20:
            alert = {
                'timestamp': time.time(),
                'type': 'drawdown_limit',
                'severity': 'critical',
                'message': f"Maximum drawdown exceeded: {max_drawdown:.1%} > 20%",
                'current_value': max_drawdown,
                'limit': 0.20
            }
            self.drawdown_alerts.append(alert)
            logger.critical(f"🚨 DRAWDOWN ALERT: {alert['message']}")
        
        # Leverage limit check (example: 3x)
        if leverage > 3.0:
            logger.warning(f"⚠️ High leverage detected: {leverage:.2f}x")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get risk summary statistics."""
        if not self.risk_history:
            return {}
        
        latest = self.risk_history[-1]
        return {
            'current_var_95': latest['var_95'],
            'current_var_99': latest['var_99'],
            'current_drawdown': latest['max_drawdown'],
            'current_sharpe': latest['sharpe_ratio'],
            'current_leverage': latest['leverage'],
            'var_breaches_count': len(self.var_breaches),
            'drawdown_alerts_count': len(self.drawdown_alerts),
            'risk_score': self._calculate_risk_score(latest)
        }
    
    def _calculate_risk_score(self, risk_data: Dict[str, Any]) -> float:
        """Calculate overall risk score (0-100)."""
        score = 100.0
        
        # Reduce score based on various factors
        if risk_data['max_drawdown'] > 0.10:
            score -= 20
        if risk_data['leverage'] > 2.0:
            score -= 15
        if risk_data['var_95'] > 150000:
            score -= 25
        if risk_data['sharpe_ratio'] < 1.0:
            score -= 10
        
        return max(score, 0.0)


class AlertProcessor(EventProcessor):
    """Processor for alert events."""
    
    def __init__(self):
        super().__init__("alert_processor")
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.alert_counts = defaultdict(int)
    
    async def process(self, event: Dict[str, Any], headers: Dict[str, str]) -> bool:
        """Process alert event."""
        try:
            alert_id = event.get('alert_id')
            severity = event.get('severity', 'info')
            category = event.get('category', 'unknown')
            
            # Store alert
            self.active_alerts[alert_id] = event
            self.alert_history.append(event)
            self.alert_counts[severity] += 1
            
            # Log based on severity
            message = event.get('message', 'No message')
            if severity.lower() == 'critical':
                logger.critical(f"🚨 CRITICAL ALERT: {message}")
            elif severity.lower() == 'warning':
                logger.warning(f"⚠️ WARNING: {message}")
            else:
                logger.info(f"ℹ️ INFO: {message}")
            
            # Handle auto-remediation
            if event.get('auto_remediation', False):
                await self._handle_auto_remediation(event)
            
            self.processed_count += 1
            return True
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"❌ Failed to process alert: {e}")
            return False
    
    async def _handle_auto_remediation(self, alert: Dict[str, Any]):
        """Handle automatic remediation actions."""
        action = alert.get('recommended_action', '')
        
        if 'reduce position' in action.lower():
            logger.info(f"🔧 Auto-remediation: Position reduction requested")
            # In production, this would trigger position reduction
        elif 'halt trading' in action.lower():
            logger.warning(f"🛑 Auto-remediation: Trading halt requested")
            # In production, this would halt trading systems
        
        logger.info(f"🔧 Auto-remediation triggered for alert: {alert.get('alert_id')}")


class KafkaEventConsumer:
    """
    High-performance Kafka event consumer for trading system events.
    
    Provides reliable event consumption with processing pipelines,
    error handling, and comprehensive monitoring.
    """
    
    def __init__(self,
                 bootstrap_servers: List[str],
                 group_id: str = "dipmaster-consumer",
                 topics: List[str] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kafka event consumer.
        
        Args:
            bootstrap_servers: List of Kafka broker addresses
            group_id: Consumer group ID
            topics: List of topics to subscribe to
            config: Additional Kafka configuration
        """
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics or []
        self.config = config or {}
        
        # Consumer instance
        self.consumer = None
        self.state = ConsumerState.STOPPED
        
        # Processing components
        self.processors = {}
        self._setup_processors()
        
        # Statistics
        self.stats = EventStats()
        self.topic_stats = defaultdict(lambda: EventStats())
        
        # Control
        self._stop_event = threading.Event()
        self._consumer_thread = None
        
        logger.info(f"📥 KafkaEventConsumer initialized for {bootstrap_servers}")
    
    def _setup_processors(self):
        """Setup event processors for different event types."""
        self.processors = {
            'exec.reports.v1': ExecutionReportProcessor(),
            'risk.metrics.v1': RiskMetricsProcessor(),
            'alerts.v1': AlertProcessor(),
            'strategy.performance.v1': EventProcessor('strategy_performance')
        }
        
        logger.info(f"📋 Setup {len(self.processors)} event processors")
    
    async def start(self):
        """Start consuming events."""
        if self.state != ConsumerState.STOPPED:
            logger.warning("⚠️ Consumer already running or starting")
            return
        
        try:
            self.state = ConsumerState.STARTING
            logger.info("🚀 Starting Kafka event consumer...")
            
            # Start consumer thread
            self._consumer_thread = threading.Thread(target=self._consume_events, daemon=True)
            self._consumer_thread.start()
            
            # Wait a bit for thread to start
            await asyncio.sleep(1)
            
            if self.state == ConsumerState.RUNNING:
                logger.info("✅ Kafka event consumer started successfully")
            else:
                raise Exception("Failed to start consumer thread")
                
        except Exception as e:
            self.state = ConsumerState.ERROR
            logger.error(f"❌ Failed to start consumer: {e}")
            raise
    
    async def stop(self):
        """Stop consuming events."""
        if self.state == ConsumerState.STOPPED:
            logger.warning("⚠️ Consumer already stopped")
            return
        
        try:
            self.state = ConsumerState.STOPPING
            logger.info("🛑 Stopping Kafka event consumer...")
            
            # Signal stop
            self._stop_event.set()
            
            # Wait for consumer thread
            if self._consumer_thread and self._consumer_thread.is_alive():
                self._consumer_thread.join(timeout=10)
            
            # Close consumer
            if self.consumer:
                self.consumer.close()
                self.consumer = None
            
            self.state = ConsumerState.STOPPED
            logger.info("✅ Kafka event consumer stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping consumer: {e}")
            self.state = ConsumerState.ERROR
    
    def _consume_events(self):
        """Main event consumption loop (runs in separate thread)."""
        try:
            if not KAFKA_AVAILABLE:
                logger.warning("⚠️ Kafka not available - using mock consumer")
                self.state = ConsumerState.RUNNING
                # Mock consumption loop
                while not self._stop_event.is_set():
                    time.sleep(1)
                return
            
            # Create consumer
            consumer_config = {
                'bootstrap_servers': self.bootstrap_servers,
                'group_id': self.group_id,
                'auto_offset_reset': 'latest',
                'enable_auto_commit': True,
                'auto_commit_interval_ms': 1000,
                'value_deserializer': lambda x: json.loads(x.decode('utf-8')),
                'key_deserializer': lambda x: x.decode('utf-8') if x else None,
                **self.config.get('consumer_config', {})
            }
            
            self.consumer = KafkaConsumer(**consumer_config)
            self.consumer.subscribe(self.topics)
            
            self.state = ConsumerState.RUNNING
            logger.info(f"📥 Subscribed to topics: {self.topics}")
            
            # Main consumption loop
            for message in self.consumer:
                if self._stop_event.is_set():
                    break
                
                try:
                    # Process message
                    asyncio.run(self._process_message(message))
                    
                except Exception as e:
                    logger.error(f"❌ Error processing message: {e}")
                    self.stats.total_failed += 1
            
        except Exception as e:
            logger.error(f"❌ Consumer thread error: {e}")
            self.state = ConsumerState.ERROR
        finally:
            if self.consumer:
                self.consumer.close()
    
    async def _process_message(self, message):
        """Process a Kafka message."""
        try:
            topic = message.topic
            event_data = message.value
            headers = {k: v.decode('utf-8') for k, v in message.headers or []}
            
            # Update statistics
            self.stats.total_processed += 1
            self.stats.bytes_processed += len(message.value) if hasattr(message, 'value') else 0
            self.stats.last_processed_time = time.time()
            
            self.topic_stats[topic].total_processed += 1
            self.topic_stats[topic].last_processed_time = time.time()
            
            # Process with appropriate processor
            processor = self.processors.get(topic)
            if processor:
                success = await processor.process(event_data, headers)
                if not success:
                    self.stats.total_failed += 1
                    self.topic_stats[topic].total_failed += 1
            else:
                logger.warning(f"⚠️ No processor for topic: {topic}")
            
            logger.debug(f"📥 Processed message from {topic}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process message: {e}")
            self.stats.total_failed += 1
    
    def get_consumer_stats(self) -> Dict[str, Any]:
        """Get consumer statistics."""
        current_time = time.time()
        
        # Calculate processing rate
        if self.stats.last_processed_time:
            time_diff = current_time - self.stats.last_processed_time
            if time_diff > 0:
                self.stats.processing_rate = self.stats.total_processed / time_diff
        
        # Calculate error rate
        total_events = self.stats.total_processed + self.stats.total_failed
        if total_events > 0:
            self.stats.error_rate = self.stats.total_failed / total_events
        
        return {
            'state': self.state.value,
            'total_processed': self.stats.total_processed,
            'total_failed': self.stats.total_failed,
            'bytes_processed': self.stats.bytes_processed,
            'processing_rate': self.stats.processing_rate,
            'error_rate': self.stats.error_rate,
            'last_processed': self.stats.last_processed_time,
            'topic_stats': {topic: {
                'processed': stats.total_processed,
                'failed': stats.total_failed,
                'last_processed': stats.last_processed_time
            } for topic, stats in self.topic_stats.items()},
            'processor_stats': {name: proc.get_stats() for name, proc in self.processors.items()}
        }
    
    def get_processor(self, processor_type: str) -> Optional[EventProcessor]:
        """Get specific processor instance."""
        return self.processors.get(processor_type)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if self.state != ConsumerState.RUNNING:
                return {'status': 'unhealthy', 'reason': f'state_{self.state.value}'}
            
            # Check error rate
            if self.stats.error_rate > 0.05:  # 5% error rate
                return {'status': 'degraded', 'reason': 'high_error_rate', 'error_rate': self.stats.error_rate}
            
            # Check if we're receiving messages
            if self.stats.last_processed_time:
                time_since_last = time.time() - self.stats.last_processed_time
                if time_since_last > 300:  # 5 minutes
                    return {'status': 'degraded', 'reason': 'no_recent_messages', 'seconds_since_last': time_since_last}
            
            return {
                'status': 'healthy',
                'processing_rate': self.stats.processing_rate,
                'error_rate': self.stats.error_rate
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


# Factory function
def create_kafka_consumer(config: Dict[str, Any]) -> KafkaEventConsumer:
    """Create and configure Kafka event consumer."""
    return KafkaEventConsumer(
        bootstrap_servers=config.get('kafka_servers', ['localhost:9092']),
        group_id=config.get('group_id', 'dipmaster-consumer'),
        topics=config.get('topics', ['exec.reports.v1', 'risk.metrics.v1', 'alerts.v1']),
        config=config.get('consumer_config', {})
    )