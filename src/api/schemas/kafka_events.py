"""
DipMaster Kafka事件模式定义
==========================

定义所有Kafka主题的事件模式，确保数据一致性和类型安全。
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class OrderStatus(str, Enum):
    """订单状态枚举"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class OrderSide(str, Enum):
    """订单方向枚举"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """订单类型枚举"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"


class AlertSeverity(str, Enum):
    """告警严重级别"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SystemComponent(str, Enum):
    """系统组件枚举"""
    TRADING_ENGINE = "TRADING_ENGINE"
    SIGNAL_DETECTOR = "SIGNAL_DETECTOR"
    POSITION_MANAGER = "POSITION_MANAGER"
    ORDER_EXECUTOR = "ORDER_EXECUTOR"
    RISK_MANAGER = "RISK_MANAGER"
    WEBSOCKET_CLIENT = "WEBSOCKET_CLIENT"
    DATA_PIPELINE = "DATA_PIPELINE"


class Fill(BaseModel):
    """成交信息"""
    fill_id: str = Field(..., description="成交ID")
    trade_id: str = Field(..., description="交易ID")
    order_id: str = Field(..., description="订单ID")
    symbol: str = Field(..., description="交易对")
    side: OrderSide = Field(..., description="买卖方向")
    quantity: Decimal = Field(..., description="成交数量", gt=0)
    price: Decimal = Field(..., description="成交价格", gt=0)
    commission: Decimal = Field(..., description="手续费", ge=0)
    commission_asset: str = Field(..., description="手续费币种")
    timestamp: datetime = Field(..., description="成交时间")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class Order(BaseModel):
    """订单信息"""
    order_id: str = Field(..., description="订单ID")
    client_order_id: str = Field(..., description="客户端订单ID")
    symbol: str = Field(..., description="交易对")
    side: OrderSide = Field(..., description="买卖方向")
    type: OrderType = Field(..., description="订单类型")
    quantity: Decimal = Field(..., description="订单数量", gt=0)
    price: Optional[Decimal] = Field(None, description="订单价格")
    stop_price: Optional[Decimal] = Field(None, description="止损价格")
    status: OrderStatus = Field(..., description="订单状态")
    filled_quantity: Decimal = Field(default=Decimal("0"), description="已成交数量", ge=0)
    remaining_quantity: Decimal = Field(..., description="剩余数量", ge=0)
    create_time: datetime = Field(..., description="创建时间")
    update_time: datetime = Field(..., description="更新时间")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class Position(BaseModel):
    """持仓信息"""
    symbol: str = Field(..., description="交易对")
    quantity: Decimal = Field(..., description="持仓数量")
    avg_price: Decimal = Field(..., description="平均成本价", gt=0)
    market_value: Decimal = Field(..., description="市值")
    unrealized_pnl: Decimal = Field(..., description="未实现盈亏")
    realized_pnl: Decimal = Field(..., description="已实现盈亏")
    side: str = Field(..., description="持仓方向", regex="^(LONG|SHORT|FLAT)$")
    
    class Config:
        json_encoders = {
            Decimal: str
        }


class ExecutionReportV1(BaseModel):
    """
    exec.reports.v1 - 执行报告事件
    包含订单执行、成交和PnL信息
    """
    event_id: str = Field(..., description="事件ID")
    timestamp: datetime = Field(..., description="事件时间戳")
    strategy_id: str = Field(default="dipmaster", description="策略ID")
    account_id: str = Field(..., description="账户ID")
    
    # 订单信息
    orders: List[Order] = Field(default_factory=list, description="相关订单列表")
    
    # 成交信息
    fills: List[Fill] = Field(default_factory=list, description="成交列表")
    
    # PnL信息
    total_realized_pnl: Decimal = Field(default=Decimal("0"), description="总已实现盈亏")
    total_unrealized_pnl: Decimal = Field(default=Decimal("0"), description="总未实现盈亏")
    total_commission: Decimal = Field(default=Decimal("0"), description="总手续费")
    
    # 成本分析
    total_cost: Decimal = Field(default=Decimal("0"), description="总成本")
    slippage: Decimal = Field(default=Decimal("0"), description="滑点成本")
    market_impact: Decimal = Field(default=Decimal("0"), description="市场冲击成本")
    
    # 执行统计
    execution_time_ms: int = Field(default=0, description="执行时间(毫秒)")
    venue: str = Field(default="binance", description="交易所")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class RiskMetric(BaseModel):
    """风险指标"""
    name: str = Field(..., description="指标名称")
    value: Decimal = Field(..., description="指标值")
    threshold: Optional[Decimal] = Field(None, description="阈值")
    status: str = Field(..., description="状态", regex="^(NORMAL|WARNING|BREACH)$")
    
    class Config:
        json_encoders = {Decimal: str}


class RiskMetricsV1(BaseModel):
    """
    risk.metrics.v1 - 风险指标事件
    包含实时风险监控数据
    """
    event_id: str = Field(..., description="事件ID")
    timestamp: datetime = Field(..., description="事件时间戳")
    strategy_id: str = Field(default="dipmaster", description="策略ID")
    account_id: str = Field(..., description="账户ID")
    
    # 持仓风险
    positions: List[Position] = Field(default_factory=list, description="当前持仓")
    total_exposure: Decimal = Field(default=Decimal("0"), description="总敞口")
    max_position_size: Decimal = Field(default=Decimal("1000"), description="最大单仓规模")
    position_concentration: Dict[str, Decimal] = Field(default_factory=dict, description="持仓集中度")
    
    # 风险指标
    var_1d: Decimal = Field(default=Decimal("0"), description="1日VaR")
    var_5d: Decimal = Field(default=Decimal("0"), description="5日VaR")
    max_drawdown: Decimal = Field(default=Decimal("0"), description="最大回撤")
    sharpe_ratio: Decimal = Field(default=Decimal("0"), description="夏普比率")
    volatility: Decimal = Field(default=Decimal("0"), description="波动率")
    
    # 限制检查
    daily_pnl: Decimal = Field(default=Decimal("0"), description="当日盈亏")
    daily_loss_limit: Decimal = Field(default=Decimal("-500"), description="日损失限制")
    max_positions: int = Field(default=3, description="最大持仓数")
    current_positions: int = Field(default=0, description="当前持仓数")
    
    # 风险评估
    risk_score: Decimal = Field(default=Decimal("0"), description="风险评分(0-100)")
    risk_metrics: List[RiskMetric] = Field(default_factory=list, description="详细风险指标")
    
    # 相关性风险
    correlation_matrix: Dict[str, Dict[str, Decimal]] = Field(default_factory=dict, description="相关性矩阵")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class AlertV1(BaseModel):
    """
    alerts.v1 - 告警事件
    包含各种系统和交易告警
    """
    event_id: str = Field(..., description="事件ID")
    timestamp: datetime = Field(..., description="事件时间戳")
    alert_id: str = Field(..., description="告警ID")
    
    # 告警基本信息
    severity: AlertSeverity = Field(..., description="严重级别")
    code: str = Field(..., description="告警代码")
    title: str = Field(..., description="告警标题")
    message: str = Field(..., description="告警描述")
    
    # 来源信息
    source: SystemComponent = Field(..., description="告警来源组件")
    strategy_id: str = Field(default="dipmaster", description="策略ID")
    account_id: Optional[str] = Field(None, description="账户ID")
    symbol: Optional[str] = Field(None, description="相关交易对")
    
    # 上下文信息
    context: Dict[str, Any] = Field(default_factory=dict, description="告警上下文")
    metrics: Dict[str, Decimal] = Field(default_factory=dict, description="相关指标")
    
    # 处理信息
    auto_resolved: bool = Field(default=False, description="是否自动解决")
    resolution_action: Optional[str] = Field(None, description="解决措施")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class ComponentHealth(BaseModel):
    """组件健康状态"""
    component: SystemComponent = Field(..., description="组件名称")
    status: str = Field(..., description="状态", regex="^(HEALTHY|DEGRADED|UNHEALTHY|UNKNOWN)$")
    uptime_seconds: int = Field(..., description="运行时间(秒)", ge=0)
    last_heartbeat: datetime = Field(..., description="最后心跳时间")
    cpu_usage: Decimal = Field(..., description="CPU使用率(%)", ge=0, le=100)
    memory_usage: Decimal = Field(..., description="内存使用率(%)", ge=0, le=100)
    error_rate: Decimal = Field(default=Decimal("0"), description="错误率(%)", ge=0, le=100)
    response_time_ms: Decimal = Field(default=Decimal("0"), description="响应时间(毫秒)", ge=0)
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class SystemHealthV1(BaseModel):
    """
    system.health.v1 - 系统健康事件
    包含系统各组件的健康状态
    """
    event_id: str = Field(..., description="事件ID")
    timestamp: datetime = Field(..., description="事件时间戳")
    
    # 整体系统状态
    overall_status: str = Field(..., description="整体状态", regex="^(HEALTHY|DEGRADED|UNHEALTHY)$")
    strategy_id: str = Field(default="dipmaster", description="策略ID")
    
    # 组件健康状态
    components: List[ComponentHealth] = Field(default_factory=list, description="组件健康状态")
    
    # 系统指标
    total_cpu_usage: Decimal = Field(default=Decimal("0"), description="总CPU使用率(%)")
    total_memory_usage: Decimal = Field(default=Decimal("0"), description="总内存使用率(%)")
    active_connections: int = Field(default=0, description="活跃连接数")
    
    # 交易相关指标
    total_positions: int = Field(default=0, description="总持仓数")
    daily_trades: int = Field(default=0, description="当日交易数")
    daily_pnl: Decimal = Field(default=Decimal("0"), description="当日盈亏")
    
    # 数据延迟指标
    market_data_delay_ms: int = Field(default=0, description="市场数据延迟(毫秒)")
    execution_delay_ms: int = Field(default=0, description="执行延迟(毫秒)")
    
    # WebSocket连接状态
    websocket_connections: Dict[str, bool] = Field(default_factory=dict, description="WebSocket连接状态")
    
    # 元数据
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


# 验证器
@validator("symbol", pre=True, always=True)
def validate_symbol(cls, v):
    """验证交易对格式"""
    if v and isinstance(v, str):
        return v.upper()
    return v