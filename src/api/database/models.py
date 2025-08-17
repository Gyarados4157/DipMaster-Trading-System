"""
数据库模型定义
============

定义数据库记录的Python模型类。
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import json

from ..schemas.kafka_events import (
    ExecutionReportV1, RiskMetricsV1, AlertV1, SystemHealthV1,
    Fill, Order, Position
)


class ExecReportModel(BaseModel):
    """执行报告数据库模型"""
    event_id: str
    timestamp: datetime
    strategy_id: str = "dipmaster"
    account_id: str
    total_realized_pnl: Decimal
    total_unrealized_pnl: Decimal
    total_commission: Decimal
    total_cost: Decimal
    slippage: Decimal
    market_impact: Decimal
    execution_time_ms: int
    venue: str = "binance"
    order_count: int
    fill_count: int
    metadata: str = "{}"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_kafka_event(cls, event: ExecutionReportV1) -> "ExecReportModel":
        """从Kafka事件创建模型"""
        return cls(
            event_id=event.event_id,
            timestamp=event.timestamp,
            strategy_id=event.strategy_id,
            account_id=event.account_id,
            total_realized_pnl=event.total_realized_pnl,
            total_unrealized_pnl=event.total_unrealized_pnl,
            total_commission=event.total_commission,
            total_cost=event.total_cost,
            slippage=event.slippage,
            market_impact=event.market_impact,
            execution_time_ms=event.execution_time_ms,
            venue=event.venue,
            order_count=len(event.orders),
            fill_count=len(event.fills),
            metadata=json.dumps(event.metadata)
        )
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class FillModel(BaseModel):
    """成交记录数据库模型"""
    fill_id: str
    trade_id: str
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: Decimal
    price: Decimal
    commission: Decimal
    commission_asset: str
    timestamp: datetime
    strategy_id: str = "dipmaster"
    account_id: str = ""
    venue: str = "binance"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_fill(cls, fill: Fill, account_id: str = "") -> "FillModel":
        """从Fill对象创建模型"""
        return cls(
            fill_id=fill.fill_id,
            trade_id=fill.trade_id,
            order_id=fill.order_id,
            symbol=fill.symbol,
            side=fill.side.value,
            quantity=fill.quantity,
            price=fill.price,
            commission=fill.commission,
            commission_asset=fill.commission_asset,
            timestamp=fill.timestamp,
            account_id=account_id
        )
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class OrderModel(BaseModel):
    """订单记录数据库模型"""
    order_id: str
    client_order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    type: str
    quantity: Decimal
    price: Optional[Decimal]
    stop_price: Optional[Decimal]
    status: str
    filled_quantity: Decimal
    remaining_quantity: Decimal
    create_time: datetime
    update_time: datetime
    strategy_id: str = "dipmaster"
    account_id: str = ""
    venue: str = "binance"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_order(cls, order: Order, account_id: str = "") -> "OrderModel":
        """从Order对象创建模型"""
        return cls(
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side.value,
            type=order.type.value,
            quantity=order.quantity,
            price=order.price,
            stop_price=order.stop_price,
            status=order.status.value,
            filled_quantity=order.filled_quantity,
            remaining_quantity=order.remaining_quantity,
            create_time=order.create_time,
            update_time=order.update_time,
            account_id=account_id
        )
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class PositionModel(BaseModel):
    """持仓记录数据库模型"""
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    side: str  # 'LONG', 'SHORT', 'FLAT'
    timestamp: datetime
    strategy_id: str = "dipmaster"
    account_id: str = ""
    venue: str = "binance"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_position(cls, position: Position, timestamp: datetime, account_id: str = "") -> "PositionModel":
        """从Position对象创建模型"""
        return cls(
            symbol=position.symbol,
            quantity=position.quantity,
            avg_price=position.avg_price,
            market_value=position.market_value,
            unrealized_pnl=position.unrealized_pnl,
            realized_pnl=position.realized_pnl,
            side=position.side,
            timestamp=timestamp,
            account_id=account_id
        )
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class RiskMetricModel(BaseModel):
    """风险指标数据库模型"""
    event_id: str
    timestamp: datetime
    strategy_id: str = "dipmaster"
    account_id: str
    total_exposure: Decimal
    max_position_size: Decimal
    var_1d: Decimal
    var_5d: Decimal
    max_drawdown: Decimal
    sharpe_ratio: Decimal
    volatility: Decimal
    daily_pnl: Decimal
    daily_loss_limit: Decimal
    max_positions: int
    current_positions: int
    risk_score: Decimal
    position_concentration: str = "{}"
    risk_metrics_detail: str = "{}"
    correlation_matrix: str = "{}"
    metadata: str = "{}"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_kafka_event(cls, event: RiskMetricsV1) -> "RiskMetricModel":
        """从Kafka事件创建模型"""
        return cls(
            event_id=event.event_id,
            timestamp=event.timestamp,
            strategy_id=event.strategy_id,
            account_id=event.account_id,
            total_exposure=event.total_exposure,
            max_position_size=event.max_position_size,
            var_1d=event.var_1d,
            var_5d=event.var_5d,
            max_drawdown=event.max_drawdown,
            sharpe_ratio=event.sharpe_ratio,
            volatility=event.volatility,
            daily_pnl=event.daily_pnl,
            daily_loss_limit=event.daily_loss_limit,
            max_positions=event.max_positions,
            current_positions=event.current_positions,
            risk_score=event.risk_score,
            position_concentration=json.dumps({k: str(v) for k, v in event.position_concentration.items()}),
            risk_metrics_detail=json.dumps([metric.dict() for metric in event.risk_metrics]),
            correlation_matrix=json.dumps({
                k: {k2: str(v2) for k2, v2 in v.items()} 
                for k, v in event.correlation_matrix.items()
            }),
            metadata=json.dumps(event.metadata)
        )
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class AlertModel(BaseModel):
    """告警记录数据库模型"""
    event_id: str
    alert_id: str
    timestamp: datetime
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    code: str
    title: str
    message: str
    source: str
    strategy_id: str = "dipmaster"
    account_id: str = ""
    symbol: str = ""
    auto_resolved: bool = False
    resolution_action: str = ""
    context: str = "{}"
    metrics: str = "{}"
    metadata: str = "{}"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_kafka_event(cls, event: AlertV1) -> "AlertModel":
        """从Kafka事件创建模型"""
        return cls(
            event_id=event.event_id,
            alert_id=event.alert_id,
            timestamp=event.timestamp,
            severity=event.severity.value,
            code=event.code,
            title=event.title,
            message=event.message,
            source=event.source.value,
            strategy_id=event.strategy_id,
            account_id=event.account_id or "",
            symbol=event.symbol or "",
            auto_resolved=event.auto_resolved,
            resolution_action=event.resolution_action or "",
            context=json.dumps(event.context),
            metrics=json.dumps({k: str(v) for k, v in event.metrics.items()}),
            metadata=json.dumps(event.metadata)
        )
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthModel(BaseModel):
    """系统健康状态数据库模型"""
    event_id: str
    timestamp: datetime
    overall_status: str  # 'HEALTHY', 'DEGRADED', 'UNHEALTHY'
    strategy_id: str = "dipmaster"
    total_cpu_usage: Decimal
    total_memory_usage: Decimal
    active_connections: int
    total_positions: int
    daily_trades: int
    daily_pnl: Decimal
    market_data_delay_ms: int
    execution_delay_ms: int
    components: str = "{}"
    websocket_connections: str = "{}"
    metadata: str = "{}"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    @classmethod
    def from_kafka_event(cls, event: SystemHealthV1) -> "HealthModel":
        """从Kafka事件创建模型"""
        return cls(
            event_id=event.event_id,
            timestamp=event.timestamp,
            overall_status=event.overall_status,
            strategy_id=event.strategy_id,
            total_cpu_usage=event.total_cpu_usage,
            total_memory_usage=event.total_memory_usage,
            active_connections=event.active_connections,
            total_positions=event.total_positions,
            daily_trades=event.daily_trades,
            daily_pnl=event.daily_pnl,
            market_data_delay_ms=event.market_data_delay_ms,
            execution_delay_ms=event.execution_delay_ms,
            components=json.dumps([comp.dict() for comp in event.components]),
            websocket_connections=json.dumps(event.websocket_connections),
            metadata=json.dumps(event.metadata)
        )
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class PnLCurveModel(BaseModel):
    """PnL曲线数据库模型"""
    timestamp: datetime
    time_bucket: datetime
    interval_type: str  # '1m', '5m', '15m', '1h', '1d'
    strategy_id: str = "dipmaster"
    account_id: str = ""
    symbol: str = "ALL"
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    cumulative_pnl: Decimal
    trade_count: int
    commission: Decimal
    volume: Decimal
    drawdown: Decimal
    var_1d: Decimal
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }