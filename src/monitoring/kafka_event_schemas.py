#!/usr/bin/env python3
"""
DipMaster Trading System - Kafka Event Schemas and Event Stream Producer
Kafka事件流 - 事件Schema定义和实时事件流生产系统

Author: DipMaster Trading System Monitoring Agent
Date: 2025-08-18
Version: 1.0.0
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from abc import ABC, abstractmethod
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EventVersion(Enum):
    """事件版本枚举"""
    V1 = "v1"
    V2 = "v2"


class KafkaTopics(Enum):
    """Kafka主题枚举"""
    EXECUTION_REPORTS = "exec.reports.v1"
    RISK_METRICS = "risk.metrics.v1"
    ALERTS = "alerts.v1"
    STRATEGY_PERFORMANCE = "strategy.performance.v1"
    SYSTEM_HEALTH = "system.health.v1"
    TRADE_SIGNALS = "trade.signals.v1"
    POSITION_UPDATES = "position.updates.v1"
    MARKET_DATA_QUALITY = "market.data.quality.v1"


@dataclass
class BaseKafkaEvent:
    """Kafka事件基类"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = EventVersion.V1.value
    source: str = "dipmaster-monitoring"
    
    def to_kafka_message(self) -> Dict[str, Any]:
        """转换为Kafka消息格式"""
        data = asdict(self)
        # 转换datetime为ISO字符串
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
        return data


@dataclass
class ExecutionReportEvent(BaseKafkaEvent):
    """交易执行报告事件 - Topic: exec.reports.v1"""
    execution_id: str = ""
    signal_id: str = ""
    symbol: str = ""
    side: str = ""  # BUY, SELL
    quantity: float = 0.0
    price: float = 0.0
    slippage_bps: float = 0.0
    latency_ms: float = 0.0
    venue: str = "binance"
    status: str = "FILLED"  # FILLED, PARTIAL, REJECTED, CANCELLED
    strategy: str = "dipmaster"
    order_type: str = "MARKET"
    commission: float = 0.0
    commission_asset: str = "USDT"
    
    def to_kafka_message(self) -> Dict[str, Any]:
        """转换为Kafka消息格式"""
        message = super().to_kafka_message()
        message.update({
            'topic': KafkaTopics.EXECUTION_REPORTS.value,
            'key': f"{self.symbol}_{self.execution_id}",
            'execution_quality': {
                'slippage_bps': self.slippage_bps,
                'latency_ms': self.latency_ms,
                'venue_quality_score': self._calculate_venue_quality_score()
            }
        })
        return message
    
    def _calculate_venue_quality_score(self) -> float:
        """计算交易所质量评分"""
        # 基于滑点和延迟计算质量评分
        slippage_penalty = min(self.slippage_bps / 10.0, 50.0)  # 最大扣50分
        latency_penalty = min(self.latency_ms / 100.0, 30.0)    # 最大扣30分
        return max(0.0, 100.0 - slippage_penalty - latency_penalty)


@dataclass
class RiskMetricsEvent(BaseKafkaEvent):
    """风险指标事件 - Topic: risk.metrics.v1"""
    portfolio_id: str = "main_portfolio"
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    leverage: float = 0.0
    correlation_stability: float = 0.0
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    position_count: int = 0
    risk_utilization: float = 0.0
    
    def to_kafka_message(self) -> Dict[str, Any]:
        """转换为Kafka消息格式"""
        message = super().to_kafka_message()
        message.update({
            'topic': KafkaTopics.RISK_METRICS.value,
            'key': f"{self.portfolio_id}_{int(self.timestamp.timestamp())}",
            'risk_assessment': {
                'risk_score': self._calculate_risk_score(),
                'risk_level': self._get_risk_level(),
                'breach_alerts': self._check_risk_breaches()
            }
        })
        return message
    
    def _calculate_risk_score(self) -> float:
        """计算整体风险评分 (0-100)"""
        # 基于多个风险指标的综合评分
        var_score = min(self.var_95 / 100000.0 * 50, 50)  # VaR评分，最大50分
        drawdown_score = min(self.current_drawdown * 100, 30)  # 回撤评分，最大30分
        leverage_score = min(self.leverage / 3.0 * 20, 20)     # 杠杆评分，最大20分
        
        return min(100.0, var_score + drawdown_score + leverage_score)
    
    def _get_risk_level(self) -> str:
        """获取风险等级"""
        risk_score = self._calculate_risk_score()
        if risk_score < 30:
            return "LOW"
        elif risk_score < 60:
            return "MEDIUM"
        elif risk_score < 80:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def _check_risk_breaches(self) -> List[str]:
        """检查风险限制违规"""
        breaches = []
        
        # 预设的风险限制
        if self.var_95 > 200000:
            breaches.append("VAR_95_EXCEEDED")
        if self.var_99 > 300000:
            breaches.append("VAR_99_EXCEEDED")
        if self.current_drawdown > 0.20:
            breaches.append("MAX_DRAWDOWN_EXCEEDED")
        if self.leverage > 3.0:
            breaches.append("MAX_LEVERAGE_EXCEEDED")
        
        return breaches


@dataclass
class AlertEvent(BaseKafkaEvent):
    """告警事件 - Topic: alerts.v1"""
    alert_id: str = ""
    severity: str = "INFO"  # INFO, WARNING, CRITICAL, EMERGENCY
    category: str = ""
    message: str = ""
    affected_systems: List[str] = field(default_factory=list)
    recommended_action: str = ""
    auto_remediation: bool = False
    tags: Dict[str, str] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_kafka_message(self) -> Dict[str, Any]:
        """转换为Kafka消息格式"""
        message = super().to_kafka_message()
        message.update({
            'topic': KafkaTopics.ALERTS.value,
            'key': self.alert_id,
            'alert_metadata': {
                'priority_score': self._calculate_priority_score(),
                'escalation_required': self._should_escalate(),
                'similar_alerts_count': 0  # 需要从历史数据计算
            }
        })
        return message
    
    def _calculate_priority_score(self) -> int:
        """计算告警优先级评分 (1-10)"""
        severity_scores = {
            "INFO": 1,
            "WARNING": 4,
            "CRITICAL": 8,
            "EMERGENCY": 10
        }
        return severity_scores.get(self.severity, 1)
    
    def _should_escalate(self) -> bool:
        """判断是否需要升级"""
        return self.severity in ["CRITICAL", "EMERGENCY"] and not self.resolved


@dataclass
class StrategyPerformanceEvent(BaseKafkaEvent):
    """策略性能事件 - Topic: strategy.performance.v1"""
    strategy_name: str = "dipmaster"
    time_window: str = "1h"  # 1h, 4h, 1d, 1w
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_time_minutes: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    
    def to_kafka_message(self) -> Dict[str, Any]:
        """转换为Kafka消息格式"""
        message = super().to_kafka_message()
        message.update({
            'topic': KafkaTopics.STRATEGY_PERFORMANCE.value,
            'key': f"{self.strategy_name}_{self.time_window}_{int(self.timestamp.timestamp())}",
            'performance_assessment': {
                'performance_score': self._calculate_performance_score(),
                'performance_grade': self._get_performance_grade(),
                'benchmark_comparison': self._compare_to_benchmark()
            }
        })
        return message
    
    def _calculate_performance_score(self) -> float:
        """计算性能评分 (0-100)"""
        # 基于多个性能指标的综合评分
        win_rate_score = min(self.win_rate * 100, 40)  # 胜率评分，最大40分
        sharpe_score = min(max(self.sharpe_ratio, 0) * 20, 30)  # 夏普比率评分，最大30分
        profit_factor_score = min(max(self.profit_factor - 1, 0) * 30, 30)  # 盈亏比评分，最大30分
        
        return win_rate_score + sharpe_score + profit_factor_score
    
    def _get_performance_grade(self) -> str:
        """获取性能等级"""
        score = self._calculate_performance_score()
        if score >= 80:
            return "EXCELLENT"
        elif score >= 60:
            return "GOOD"
        elif score >= 40:
            return "AVERAGE"
        else:
            return "POOR"
    
    def _compare_to_benchmark(self) -> Dict[str, float]:
        """与基准对比"""
        # DipMaster策略基准指标
        benchmark = {
            'win_rate': 0.55,
            'sharpe_ratio': 1.5,
            'profit_factor': 1.5,
            'max_drawdown': 0.15
        }
        
        return {
            'win_rate_vs_benchmark': self.win_rate - benchmark['win_rate'],
            'sharpe_vs_benchmark': self.sharpe_ratio - benchmark['sharpe_ratio'],
            'profit_factor_vs_benchmark': self.profit_factor - benchmark['profit_factor'],
            'drawdown_vs_benchmark': benchmark['max_drawdown'] - self.max_drawdown  # 越小越好
        }


@dataclass
class SystemHealthEvent(BaseKafkaEvent):
    """系统健康事件 - Topic: system.health.v1"""
    component_name: str = ""
    health_score: float = 100.0
    status: str = "healthy"  # healthy, degraded, unhealthy
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0
    network_latency_ms: float = 0.0
    api_response_time_ms: float = 0.0
    error_rate_percent: float = 0.0
    uptime_seconds: float = 0.0
    active_connections: int = 0
    
    def to_kafka_message(self) -> Dict[str, Any]:
        """转换为Kafka消息格式"""
        message = super().to_kafka_message()
        message.update({
            'topic': KafkaTopics.SYSTEM_HEALTH.value,
            'key': f"{self.component_name}_{int(self.timestamp.timestamp())}",
            'health_assessment': {
                'resource_stress': self._calculate_resource_stress(),
                'performance_impact': self._assess_performance_impact(),
                'recommendations': self._generate_recommendations()
            }
        })
        return message
    
    def _calculate_resource_stress(self) -> float:
        """计算资源压力值 (0-100)"""
        stress_factors = [
            self.cpu_usage_percent,
            self.memory_usage_percent,
            self.disk_usage_percent,
            min(self.error_rate_percent * 10, 100)  # 错误率权重更高
        ]
        return sum(stress_factors) / len(stress_factors)
    
    def _assess_performance_impact(self) -> str:
        """评估性能影响"""
        stress = self._calculate_resource_stress()
        if stress < 50:
            return "MINIMAL"
        elif stress < 75:
            return "MODERATE"
        else:
            return "SEVERE"
    
    def _generate_recommendations(self) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if self.cpu_usage_percent > 80:
            recommendations.append("Optimize CPU-intensive operations")
        if self.memory_usage_percent > 80:
            recommendations.append("Investigate memory leaks or increase memory allocation")
        if self.disk_usage_percent > 85:
            recommendations.append("Clean up old log files or increase disk space")
        if self.api_response_time_ms > 1000:
            recommendations.append("Optimize API response time or check network connectivity")
        if self.error_rate_percent > 5:
            recommendations.append("Investigate and fix recurring errors")
        
        return recommendations


@dataclass
class TradeSignalEvent(BaseKafkaEvent):
    """交易信号事件 - Topic: trade.signals.v1"""
    signal_id: str = ""
    strategy: str = "dipmaster"
    symbol: str = ""
    signal_type: str = ""  # BUY, SELL
    confidence: float = 0.0
    price: float = 0.0
    expected_entry_price: float = 0.0
    expected_holding_minutes: int = 60
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_kafka_message(self) -> Dict[str, Any]:
        """转换为Kafka消息格式"""
        message = super().to_kafka_message()
        message.update({
            'topic': KafkaTopics.TRADE_SIGNALS.value,
            'key': f"{self.symbol}_{self.signal_id}",
            'signal_quality': {
                'strength_score': self._calculate_signal_strength(),
                'quality_grade': self._get_quality_grade(),
                'dipmaster_compliance': self._check_dipmaster_compliance()
            }
        })
        return message
    
    def _calculate_signal_strength(self) -> float:
        """计算信号强度 (0-100)"""
        # 基于置信度和技术指标
        confidence_score = self.confidence * 60  # 置信度权重60%
        
        # 技术指标评分
        rsi = self.technical_indicators.get('rsi', 50)
        ma20_distance = self.technical_indicators.get('ma20_distance', 0)
        volume_ratio = self.technical_indicators.get('volume_ratio', 1)
        
        # DipMaster特定的技术指标评分
        tech_score = 0
        if 30 <= rsi <= 50:  # RSI在目标区间
            tech_score += 20
        if ma20_distance < 0:  # 价格低于MA20
            tech_score += 10
        if volume_ratio > 1.5:  # 成交量放大
            tech_score += 10
        
        return min(100, confidence_score + tech_score)
    
    def _get_quality_grade(self) -> str:
        """获取信号质量等级"""
        strength = self._calculate_signal_strength()
        if strength >= 80:
            return "EXCELLENT"
        elif strength >= 60:
            return "GOOD"
        elif strength >= 40:
            return "AVERAGE"
        else:
            return "POOR"
    
    def _check_dipmaster_compliance(self) -> Dict[str, bool]:
        """检查DipMaster策略合规性"""
        rsi = self.technical_indicators.get('rsi', 50)
        ma20_distance = self.technical_indicators.get('ma20_distance', 0)
        volume_ratio = self.technical_indicators.get('volume_ratio', 1)
        
        return {
            'rsi_in_range': 30 <= rsi <= 50,
            'price_below_ma20': ma20_distance < 0,
            'volume_confirmation': volume_ratio > 1.5,
            'dip_buying': self.signal_type == 'BUY' and self.price < self.expected_entry_price
        }


@dataclass
class PositionUpdateEvent(BaseKafkaEvent):
    """持仓更新事件 - Topic: position.updates.v1"""
    position_id: str = ""
    signal_id: str = ""
    symbol: str = ""
    side: str = ""  # BUY, SELL
    quantity: float = 0.0
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    holding_time_minutes: int = 0
    status: str = "OPEN"  # OPEN, CLOSED, PARTIAL
    strategy: str = "dipmaster"
    
    def to_kafka_message(self) -> Dict[str, Any]:
        """转换为Kafka消息格式"""
        message = super().to_kafka_message()
        message.update({
            'topic': KafkaTopics.POSITION_UPDATES.value,
            'key': f"{self.symbol}_{self.position_id}",
            'position_analysis': {
                'pnl_percentage': self._calculate_pnl_percentage(),
                'holding_efficiency': self._calculate_holding_efficiency(),
                'exit_timing_score': self._assess_exit_timing()
            }
        })
        return message
    
    def _calculate_pnl_percentage(self) -> float:
        """计算P&L百分比"""
        if self.entry_price == 0:
            return 0.0
        
        if self.status == "CLOSED" and self.realized_pnl != 0:
            return (self.realized_pnl / (self.entry_price * self.quantity)) * 100
        else:
            return ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    def _calculate_holding_efficiency(self) -> float:
        """计算持仓效率评分 (0-100)"""
        # DipMaster目标持仓时间为60-120分钟
        target_min, target_max = 60, 120
        
        if target_min <= self.holding_time_minutes <= target_max:
            return 100.0
        elif self.holding_time_minutes < target_min:
            # 持仓时间过短
            return max(0, 100 - (target_min - self.holding_time_minutes) * 2)
        else:
            # 持仓时间过长
            return max(0, 100 - (self.holding_time_minutes - target_max) * 0.5)
    
    def _assess_exit_timing(self) -> float:
        """评估出场时机评分 (0-100)"""
        pnl_pct = self._calculate_pnl_percentage()
        
        # DipMaster目标利润为0.8%
        target_profit = 0.8
        
        if pnl_pct >= target_profit:
            return 100.0
        elif pnl_pct > 0:
            return (pnl_pct / target_profit) * 100
        else:
            # 亏损情况下的时间止损评分
            if self.holding_time_minutes >= 180:  # 最大持仓时间
                return 60.0  # 及时止损
            else:
                return max(0, 60 - abs(pnl_pct) * 10)


class KafkaEventProducer:
    """Kafka事件生产者"""
    
    def __init__(self, kafka_config: Dict[str, Any] = None):
        self.kafka_config = kafka_config or {}
        self.producer = None
        self.is_connected = False
        self.event_buffer: List[Dict[str, Any]] = []
        self.buffer_max_size = self.kafka_config.get('buffer_max_size', 1000)
        self.stats = {
            'events_published': 0,
            'events_failed': 0,
            'events_buffered': 0
        }
    
    async def connect(self) -> bool:
        """连接到Kafka"""
        try:
            # 这里应该初始化真实的Kafka producer
            # 为了演示，我们使用模拟连接
            self.is_connected = True
            logger.info("📤 Kafka producer connected")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to Kafka: {e}")
            return False
    
    async def disconnect(self) -> None:
        """断开Kafka连接"""
        try:
            if self.producer:
                # 这里应该关闭真实的Kafka producer
                pass
            self.is_connected = False
            logger.info("📤 Kafka producer disconnected")
        except Exception as e:
            logger.error(f"❌ Failed to disconnect from Kafka: {e}")
    
    async def publish_event(self, event: BaseKafkaEvent) -> bool:
        """发布事件到Kafka"""
        try:
            message = event.to_kafka_message()
            
            if self.is_connected:
                # 这里应该发送到真实的Kafka
                # 为了演示，我们只是记录日志
                logger.debug(f"📤 Published event to {message.get('topic')}: {message.get('key')}")
                self.stats['events_published'] += 1
                return True
            else:
                # 连接断开时缓存事件
                await self._buffer_event(message)
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to publish event: {e}")
            self.stats['events_failed'] += 1
            return False
    
    async def _buffer_event(self, message: Dict[str, Any]) -> None:
        """缓存事件到本地缓冲区"""
        if len(self.event_buffer) >= self.buffer_max_size:
            # 移除最老的事件
            self.event_buffer.pop(0)
        
        self.event_buffer.append(message)
        self.stats['events_buffered'] += 1
        logger.debug(f"📦 Buffered event, buffer size: {len(self.event_buffer)}")
    
    async def flush_buffer(self) -> int:
        """刷新缓冲区中的事件"""
        if not self.is_connected or not self.event_buffer:
            return 0
        
        flushed_count = 0
        buffer_copy = self.event_buffer.copy()
        self.event_buffer.clear()
        
        for message in buffer_copy:
            try:
                # 这里应该发送到真实的Kafka
                logger.debug(f"📤 Flushed buffered event to {message.get('topic')}")
                flushed_count += 1
                self.stats['events_published'] += 1
            except Exception as e:
                logger.error(f"❌ Failed to flush buffered event: {e}")
                # 重新加入缓冲区
                await self._buffer_event(message)
        
        logger.info(f"📤 Flushed {flushed_count} buffered events")
        return flushed_count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取生产者统计信息"""
        return {
            'is_connected': self.is_connected,
            'buffer_size': len(self.event_buffer),
            'buffer_max_size': self.buffer_max_size,
            **self.stats
        }


class DipMasterKafkaStreamer:
    """DipMaster Kafka事件流生产器"""
    
    def __init__(self, kafka_config: Dict[str, Any] = None):
        self.kafka_config = kafka_config or {
            'servers': ['localhost:9092'],
            'client_id': 'dipmaster-events'
        }
        self.producer = KafkaEventProducer(kafka_config)
        self.is_running = False
        self.event_stats = {topic.value: 0 for topic in KafkaTopics}
    
    async def start(self) -> None:
        """启动事件流生产器"""
        if self.is_running:
            return
        
        self.is_running = True
        await self.producer.connect()
        logger.info("🚀 DipMaster Kafka Streamer started")
    
    async def stop(self) -> None:
        """停止事件流生产器"""
        self.is_running = False
        await self.producer.disconnect()
        logger.info("🛑 DipMaster Kafka Streamer stopped")
    
    # 便捷方法用于发布特定类型的事件
    
    async def publish_execution_report(
        self,
        execution_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        **kwargs
    ) -> bool:
        """发布交易执行报告"""
        event = ExecutionReportEvent(
            execution_id=execution_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            **kwargs
        )
        
        success = await self.producer.publish_event(event)
        if success:
            self.event_stats[KafkaTopics.EXECUTION_REPORTS.value] += 1
        return success
    
    async def publish_risk_metrics(
        self,
        portfolio_id: str = "main_portfolio",
        var_95: float = 0.0,
        var_99: float = 0.0,
        **kwargs
    ) -> bool:
        """发布风险指标"""
        event = RiskMetricsEvent(
            portfolio_id=portfolio_id,
            var_95=var_95,
            var_99=var_99,
            **kwargs
        )
        
        success = await self.producer.publish_event(event)
        if success:
            self.event_stats[KafkaTopics.RISK_METRICS.value] += 1
        return success
    
    async def publish_alert(
        self,
        alert_id: str,
        severity: str,
        category: str,
        message: str,
        **kwargs
    ) -> bool:
        """发布告警事件"""
        event = AlertEvent(
            alert_id=alert_id,
            severity=severity,
            category=category,
            message=message,
            **kwargs
        )
        
        success = await self.producer.publish_event(event)
        if success:
            self.event_stats[KafkaTopics.ALERTS.value] += 1
        return success
    
    async def publish_strategy_performance(
        self,
        strategy_name: str = "dipmaster",
        time_window: str = "1h",
        win_rate: float = 0.0,
        **kwargs
    ) -> bool:
        """发布策略性能指标"""
        event = StrategyPerformanceEvent(
            strategy_name=strategy_name,
            time_window=time_window,
            win_rate=win_rate,
            **kwargs
        )
        
        success = await self.producer.publish_event(event)
        if success:
            self.event_stats[KafkaTopics.STRATEGY_PERFORMANCE.value] += 1
        return success
    
    async def publish_system_health(
        self,
        component_name: str,
        health_score: float,
        status: str = "healthy",
        **kwargs
    ) -> bool:
        """发布系统健康指标"""
        event = SystemHealthEvent(
            component_name=component_name,
            health_score=health_score,
            status=status,
            **kwargs
        )
        
        success = await self.producer.publish_event(event)
        if success:
            self.event_stats[KafkaTopics.SYSTEM_HEALTH.value] += 1
        return success
    
    async def publish_trade_signal(
        self,
        signal_id: str,
        symbol: str,
        signal_type: str,
        confidence: float,
        price: float,
        **kwargs
    ) -> bool:
        """发布交易信号"""
        event = TradeSignalEvent(
            signal_id=signal_id,
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            **kwargs
        )
        
        success = await self.producer.publish_event(event)
        if success:
            self.event_stats[KafkaTopics.TRADE_SIGNALS.value] += 1
        return success
    
    async def publish_position_update(
        self,
        position_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        current_price: float,
        **kwargs
    ) -> bool:
        """发布持仓更新"""
        event = PositionUpdateEvent(
            position_id=position_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            **kwargs
        )
        
        success = await self.producer.publish_event(event)
        if success:
            self.event_stats[KafkaTopics.POSITION_UPDATES.value] += 1
        return success
    
    def get_event_stats(self) -> Dict[str, Any]:
        """获取事件统计信息"""
        return {
            'event_counts_by_topic': self.event_stats,
            'total_events': sum(self.event_stats.values()),
            'producer_stats': self.producer.get_stats(),
            'is_running': self.is_running
        }


# 工厂函数
def create_kafka_streamer(config: Dict[str, Any] = None) -> DipMasterKafkaStreamer:
    """创建Kafka事件流生产器"""
    return DipMasterKafkaStreamer(config)


# 演示函数
async def kafka_events_demo():
    """Kafka事件流演示"""
    print("🚀 DipMaster Kafka Event Streams Demo")
    
    # 创建Kafka流生产器
    streamer = create_kafka_streamer({
        'servers': ['localhost:9092'],
        'client_id': 'dipmaster-demo'
    })
    
    try:
        # 启动流生产器
        await streamer.start()
        print("✅ Kafka streamer started")
        
        # 发布交易信号事件
        await streamer.publish_trade_signal(
            signal_id="sig_demo_001",
            symbol="BTCUSDT",
            signal_type="BUY",
            confidence=0.87,
            price=43250.50,
            technical_indicators={
                'rsi': 34.2,
                'ma20_distance': -0.008,
                'volume_ratio': 1.6
            }
        )
        print("📊 Published trade signal event")
        
        # 发布执行报告事件
        await streamer.publish_execution_report(
            execution_id="exec_demo_001",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.15,
            price=43245.00,
            slippage_bps=1.3,
            latency_ms=42,
            signal_id="sig_demo_001"
        )
        print("⚡ Published execution report event")
        
        # 发布风险指标事件
        await streamer.publish_risk_metrics(
            var_95=125000.50,
            var_99=187500.75,
            expected_shortfall=210000.00,
            sharpe_ratio=1.85,
            max_drawdown=0.082,
            leverage=1.5
        )
        print("📈 Published risk metrics event")
        
        # 发布告警事件
        await streamer.publish_alert(
            alert_id="alert_demo_001",
            severity="WARNING",
            category="RISK_LIMIT",
            message="Portfolio exposure approaching 80% of limit",
            affected_systems=["portfolio_manager", "risk_engine"],
            recommended_action="Consider reducing position sizes"
        )
        print("🚨 Published alert event")
        
        # 发布系统健康事件
        await streamer.publish_system_health(
            component_name="trading_engine",
            health_score=95.5,
            status="healthy",
            cpu_usage_percent=45.2,
            memory_usage_percent=62.8,
            api_response_time_ms=125
        )
        print("💊 Published system health event")
        
        # 发布策略性能事件
        await streamer.publish_strategy_performance(
            strategy_name="dipmaster",
            time_window="1h",
            win_rate=0.823,
            profit_factor=2.1,
            sharpe_ratio=1.85,
            total_trades=15,
            winning_trades=12,
            total_pnl=287.50
        )
        print("🎯 Published strategy performance event")
        
        # 获取统计信息
        stats = streamer.get_event_stats()
        print(f"📊 Event statistics:")
        for topic, count in stats['event_counts_by_topic'].items():
            if count > 0:
                print(f"  {topic}: {count} events")
        print(f"📊 Total events published: {stats['total_events']}")
        
        print("✅ Demo completed successfully")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        await streamer.stop()
        print("🛑 Kafka streamer stopped")


if __name__ == "__main__":
    asyncio.run(kafka_events_demo())