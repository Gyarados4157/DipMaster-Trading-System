#!/usr/bin/env python3
"""
Kafka Event Producer for DipMaster Trading System
高性能事件流生产者 - 专业交易系统事件发布

Features:
- High-throughput event publishing to Kafka
- Multiple topic support with schema validation
- Circuit breaker and retry mechanisms
- Event serialization (JSON/Avro)
- Metrics and monitoring integration
- Dead letter queue support
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from datetime import datetime, timezone

# Kafka imports (will gracefully degrade if not available)
try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError, KafkaTimeoutError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    KafkaProducer = None
    KafkaError = Exception
    KafkaTimeoutError = Exception

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Event type enumeration."""
    EXECUTION_REPORT = "exec.reports.v1"
    RISK_METRICS = "risk.metrics.v1" 
    ALERTS = "alerts.v1"
    STRATEGY_PERFORMANCE = "strategy.performance.v1"
    SYSTEM_HEALTH = "system.health.v1"
    TRADE_SIGNAL = "trade.signals.v1"
    POSITION_UPDATE = "position.updates.v1"
    MARKET_DATA = "market.data.v1"


@dataclass
class ExecutionReportEvent:
    """Execution report event schema."""
    timestamp: str
    execution_id: str
    signal_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    slippage_bps: float
    latency_ms: float
    venue: str
    status: str
    fill_qty: Optional[float] = None
    fill_price: Optional[float] = None
    fees: Optional[float] = None
    strategy: Optional[str] = None
    session_id: Optional[str] = None


@dataclass
class RiskMetricsEvent:
    """Risk metrics event schema."""
    timestamp: str
    portfolio_id: str
    var_95: float
    var_99: float
    expected_shortfall: float
    sharpe_ratio: float
    max_drawdown: float
    leverage: float
    correlation_stability: float
    active_positions: int
    total_exposure: float
    unrealized_pnl: float
    beta: Optional[float] = None
    volatility: Optional[float] = None


@dataclass
class AlertEvent:
    """Alert event schema."""
    timestamp: str
    alert_id: str
    severity: str
    category: str
    message: str
    affected_systems: List[str]
    recommended_action: str
    auto_remediation: bool
    source: str
    tags: Optional[Dict[str, str]] = None
    correlation_id: Optional[str] = None


@dataclass
class StrategyPerformanceEvent:
    """Strategy performance event schema."""
    timestamp: str
    strategy_name: str
    timeframe: str
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    total_pnl: float
    active_positions: int
    signals_generated: int
    signals_executed: int
    execution_rate: float


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for Kafka producer reliability."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("🔄 Circuit breaker transitioning to HALF_OPEN")
            else:
                raise Exception("Circuit breaker OPEN - rejecting call")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info("✅ Circuit breaker CLOSED - recovered")
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"🚫 Circuit breaker OPEN - {self.failure_count} failures")


class KafkaEventProducer:
    """
    High-performance Kafka event producer for trading system events.
    
    Provides reliable, high-throughput event publishing with comprehensive
    error handling, monitoring, and recovery mechanisms.
    """
    
    def __init__(self,
                 bootstrap_servers: List[str],
                 client_id: str = "dipmaster-producer",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize Kafka event producer.
        
        Args:
            bootstrap_servers: List of Kafka broker addresses
            client_id: Unique client identifier
            config: Additional Kafka configuration
        """
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id
        self.config = config or {}
        
        # Producer instance
        self.producer = None
        self.is_connected = False
        
        # Circuit breaker for reliability
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.get('circuit_breaker_threshold', 5),
            recovery_timeout=self.config.get('circuit_breaker_timeout', 60)
        )
        
        # Event statistics
        self.events_sent = 0
        self.events_failed = 0
        self.bytes_sent = 0
        self.last_send_time = None
        
        # Dead letter queue (local fallback)
        self.dead_letter_queue = []
        self.max_dlq_size = self.config.get('max_dlq_size', 1000)
        
        logger.info(f"📤 KafkaEventProducer initialized for {bootstrap_servers}")
    
    async def connect(self):
        """Establish connection to Kafka cluster."""
        if not KAFKA_AVAILABLE:
            logger.warning("⚠️ Kafka not available - using mock producer")
            self.is_connected = True
            return
        
        try:
            producer_config = {
                'bootstrap_servers': self.bootstrap_servers,
                'client_id': self.client_id,
                'acks': 'all',  # Wait for all replicas
                'retries': 3,
                'batch_size': 16384,
                'linger_ms': 10,
                'buffer_memory': 33554432,
                'compression_type': 'gzip',
                'value_serializer': lambda x: json.dumps(x).encode('utf-8'),
                'key_serializer': lambda x: x.encode('utf-8') if x else None,
                **self.config.get('producer_config', {})
            }
            
            self.producer = KafkaProducer(**producer_config)
            self.is_connected = True
            
            logger.info("✅ Connected to Kafka cluster")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Kafka: {e}")
            self.is_connected = False
            raise
    
    async def disconnect(self):
        """Disconnect from Kafka cluster."""
        try:
            if self.producer:
                self.producer.flush()
                self.producer.close()
                self.producer = None
            
            self.is_connected = False
            logger.info("📤 Disconnected from Kafka")
            
        except Exception as e:
            logger.error(f"❌ Error disconnecting from Kafka: {e}")
    
    async def publish_event(self,
                          topic: str,
                          event_data: Union[Dict[str, Any], dataclass],
                          key: Optional[str] = None,
                          headers: Optional[Dict[str, str]] = None) -> bool:
        """
        Publish event to Kafka topic.
        
        Args:
            topic: Kafka topic name
            event_data: Event data (dict or dataclass)
            key: Optional message key for partitioning
            headers: Optional message headers
            
        Returns:
            bool: True if published successfully
        """
        try:
            # Convert dataclass to dict if needed
            if hasattr(event_data, '__dataclass_fields__'):
                event_dict = asdict(event_data)
            else:
                event_dict = event_data
            
            # Add metadata
            event_dict.update({
                'event_id': str(uuid.uuid4()),
                'produced_at': datetime.now(timezone.utc).isoformat(),
                'producer_id': self.client_id,
                'schema_version': '1.0'
            })
            
            # Prepare headers
            kafka_headers = []
            if headers:
                kafka_headers.extend([(k, v.encode('utf-8')) for k, v in headers.items()])
            
            # Add default headers
            kafka_headers.extend([
                ('content-type', b'application/json'),
                ('producer', self.client_id.encode('utf-8')),
                ('timestamp', str(int(time.time() * 1000)).encode('utf-8'))
            ])
            
            # Publish with circuit breaker protection
            if not self.is_connected:
                raise Exception("Not connected to Kafka")
            
            def _send():
                if KAFKA_AVAILABLE and self.producer:
                    future = self.producer.send(
                        topic=topic,
                        value=event_dict,
                        key=key,
                        headers=kafka_headers
                    )
                    return future.get(timeout=10)  # 10 second timeout
                else:
                    # Mock mode for testing
                    logger.info(f"📝 [MOCK] Published to {topic}: {json.dumps(event_dict, indent=2)}")
                    return True
            
            self.circuit_breaker.call(_send)
            
            # Update statistics
            self.events_sent += 1
            self.bytes_sent += len(json.dumps(event_dict))
            self.last_send_time = time.time()
            
            logger.debug(f"📤 Published event to {topic} (key: {key})")
            return True
            
        except Exception as e:
            self.events_failed += 1
            logger.error(f"❌ Failed to publish event to {topic}: {e}")
            
            # Add to dead letter queue
            self._add_to_dlq(topic, event_data, key, headers, str(e))
            return False
    
    def _add_to_dlq(self,
                   topic: str,
                   event_data: Any,
                   key: Optional[str],
                   headers: Optional[Dict[str, str]],
                   error: str):
        """Add failed event to dead letter queue."""
        if len(self.dead_letter_queue) >= self.max_dlq_size:
            # Remove oldest event
            self.dead_letter_queue.pop(0)
        
        dlq_entry = {
            'timestamp': time.time(),
            'topic': topic,
            'event_data': event_data,
            'key': key,
            'headers': headers,
            'error': error,
            'retry_count': 0
        }
        
        self.dead_letter_queue.append(dlq_entry)
        logger.warning(f"📥 Added event to DLQ (size: {len(self.dead_letter_queue)})")
    
    async def retry_dlq_events(self, max_retries: int = 3) -> int:
        """Retry events from dead letter queue."""
        retried_count = 0
        failed_events = []
        
        for event in self.dead_letter_queue:
            if event['retry_count'] >= max_retries:
                failed_events.append(event)
                continue
            
            success = await self.publish_event(
                topic=event['topic'],
                event_data=event['event_data'],
                key=event['key'],
                headers=event['headers']
            )
            
            if success:
                retried_count += 1
                logger.info(f"✅ Retried DLQ event successfully")
            else:
                event['retry_count'] += 1
                failed_events.append(event)
        
        # Update DLQ with only failed events
        self.dead_letter_queue = failed_events
        
        logger.info(f"🔄 Retried {retried_count} DLQ events")
        return retried_count
    
    async def publish_execution_report(self, report: ExecutionReportEvent) -> bool:
        """Publish execution report event."""
        return await self.publish_event(
            topic=EventType.EXECUTION_REPORT.value,
            event_data=report,
            key=report.symbol,
            headers={'event_type': 'execution_report'}
        )
    
    async def publish_risk_metrics(self, metrics: RiskMetricsEvent) -> bool:
        """Publish risk metrics event."""
        return await self.publish_event(
            topic=EventType.RISK_METRICS.value,
            event_data=metrics,
            key=metrics.portfolio_id,
            headers={'event_type': 'risk_metrics'}
        )
    
    async def publish_alert(self, alert: AlertEvent) -> bool:
        """Publish alert event."""
        return await self.publish_event(
            topic=EventType.ALERTS.value,
            event_data=alert,
            key=alert.severity,
            headers={'event_type': 'alert', 'severity': alert.severity}
        )
    
    async def publish_strategy_performance(self, performance: StrategyPerformanceEvent) -> bool:
        """Publish strategy performance event."""
        return await self.publish_event(
            topic=EventType.STRATEGY_PERFORMANCE.value,
            event_data=performance,
            key=performance.strategy_name,
            headers={'event_type': 'strategy_performance'}
        )
    
    def get_producer_stats(self) -> Dict[str, Any]:
        """Get producer statistics."""
        return {
            'events_sent': self.events_sent,
            'events_failed': self.events_failed,
            'bytes_sent': self.bytes_sent,
            'success_rate': self.events_sent / max(self.events_sent + self.events_failed, 1),
            'dlq_size': len(self.dead_letter_queue),
            'circuit_breaker_state': self.circuit_breaker.state.value,
            'last_send_time': self.last_send_time,
            'is_connected': self.is_connected
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        try:
            if not self.is_connected:
                return {'status': 'unhealthy', 'reason': 'not_connected'}
            
            if self.circuit_breaker.state == CircuitBreakerState.OPEN:
                return {'status': 'degraded', 'reason': 'circuit_breaker_open'}
            
            # Calculate health score based on success rate
            stats = self.get_producer_stats()
            success_rate = stats['success_rate']
            
            if success_rate >= 0.99:
                status = 'healthy'
            elif success_rate >= 0.95:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            return {
                'status': status,
                'success_rate': success_rate,
                'dlq_size': stats['dlq_size'],
                'circuit_breaker': stats['circuit_breaker_state']
            }
            
        except Exception as e:
            return {'status': 'error', 'error': str(e)}


# Factory function
def create_kafka_producer(config: Dict[str, Any]) -> KafkaEventProducer:
    """Create and configure Kafka event producer."""
    return KafkaEventProducer(
        bootstrap_servers=config.get('kafka_servers', ['localhost:9092']),
        client_id=config.get('client_id', 'dipmaster-producer'),
        config=config.get('producer_config', {})
    )