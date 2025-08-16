"""
Dashboard API 事件模式定义
支持exec.reports.v1, risk.metrics.v1, alerts.v1, strategy.performance.v1事件流
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, validator
from enum import Enum

class EventBase(BaseModel):
    """事件基类"""
    event_id: str = Field(..., description="事件唯一ID")
    timestamp: datetime = Field(..., description="事件时间戳")
    version: str = Field(default="v1", description="事件版本")
    source: str = Field(..., description="事件源")

class ExecType(str, Enum):
    """执行类型"""
    NEW = "NEW"
    PARTIAL_FILL = "PARTIAL_FILL"
    FILL = "FILL"
    TRADE = "TRADE"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class OrderSide(str, Enum):
    """订单方向"""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(str, Enum):
    """订单类型"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(str, Enum):
    """订单状态"""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

class ExecReportEvent(EventBase):
    """执行报告事件 - exec.reports.v1"""
    
    # 订单信息
    order_id: str = Field(..., description="订单ID")
    client_order_id: Optional[str] = Field(None, description="客户端订单ID")
    symbol: str = Field(..., description="交易对")
    side: OrderSide = Field(..., description="买卖方向")
    order_type: OrderType = Field(..., description="订单类型")
    order_status: OrderStatus = Field(..., description="订单状态")
    
    # 执行信息
    exec_type: ExecType = Field(..., description="执行类型")
    exec_id: Optional[str] = Field(None, description="执行ID")
    
    # 数量和价格
    quantity: Decimal = Field(..., description="订单数量")
    price: Decimal = Field(..., description="订单价格")
    last_qty: Optional[Decimal] = Field(None, description="最后成交数量")
    last_price: Optional[Decimal] = Field(None, description="最后成交价格")
    cum_qty: Optional[Decimal] = Field(None, description="累计成交数量")
    avg_price: Optional[Decimal] = Field(None, description="平均成交价格")
    
    # 费用和PnL
    commission: Optional[Decimal] = Field(None, description="手续费")
    commission_asset: Optional[str] = Field(None, description="手续费资产")
    realized_pnl: Optional[Decimal] = Field(None, description="已实现盈亏")
    unrealized_pnl: Optional[Decimal] = Field(None, description="未实现盈亏")
    
    # 仓位信息
    position_qty: Optional[Decimal] = Field(None, description="仓位数量")
    position_side: Optional[str] = Field(None, description="仓位方向")
    
    # 时间信息
    order_time: Optional[datetime] = Field(None, description="订单时间")
    trade_time: Optional[datetime] = Field(None, description="成交时间")
    
    # 策略信息
    strategy_id: Optional[str] = Field(None, description="策略ID")
    account_id: str = Field(..., description="账户ID")
    
    # 市场信息
    venue: Optional[str] = Field(None, description="交易所")
    
    @validator('quantity', 'price', 'last_qty', 'last_price', 'cum_qty', 'avg_price', 
              'commission', 'realized_pnl', 'unrealized_pnl', 'position_qty', pre=True)
    def parse_decimal(cls, v):
        if v is None:
            return v
        return Decimal(str(v))

class PositionInfo(BaseModel):
    """仓位信息"""
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    position_side: str = "LONG"

class RiskMetricsEvent(EventBase):
    """风险指标事件 - risk.metrics.v1"""
    
    account_id: str = Field(..., description="账户ID")
    
    # 组合价值
    portfolio_value: Decimal = Field(..., description="组合总价值")
    cash_balance: Decimal = Field(..., description="现金余额")
    margin_balance: Optional[Decimal] = Field(None, description="保证金余额")
    
    # 损益指标
    total_pnl: Decimal = Field(..., description="总损益")
    daily_pnl: Decimal = Field(..., description="日损益")
    realized_pnl: Decimal = Field(..., description="已实现损益")
    unrealized_pnl: Decimal = Field(..., description="未实现损益")
    
    # 风险指标
    var_1d: Optional[Decimal] = Field(None, description="1日VaR")
    var_5d: Optional[Decimal] = Field(None, description="5日VaR")
    max_drawdown: Optional[Decimal] = Field(None, description="最大回撤")
    leverage: Optional[Decimal] = Field(None, description="杠杆率")
    risk_score: Optional[Decimal] = Field(None, description="风险评分")
    
    # 仓位分布
    positions: List[PositionInfo] = Field(default=[], description="持仓列表")
    num_positions: int = Field(default=0, description="持仓数量")
    
    # 风险限额
    position_limit: Optional[Decimal] = Field(None, description="仓位限额")
    loss_limit: Optional[Decimal] = Field(None, description="损失限额")
    
    # 市场风险
    market_exposure: Optional[Decimal] = Field(None, description="市场敞口")
    currency_exposure: Optional[Dict[str, Decimal]] = Field(None, description="货币敞口")
    
    @validator('portfolio_value', 'cash_balance', 'margin_balance', 'total_pnl', 
              'daily_pnl', 'realized_pnl', 'unrealized_pnl', 'var_1d', 'var_5d',
              'max_drawdown', 'leverage', 'risk_score', 'position_limit', 
              'loss_limit', 'market_exposure', pre=True)
    def parse_decimal(cls, v):
        if v is None:
            return v
        return Decimal(str(v))

class AlertSeverity(str, Enum):
    """告警严重性"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertType(str, Enum):
    """告警类型"""
    RISK_LIMIT = "RISK_LIMIT"
    POSITION_LIMIT = "POSITION_LIMIT"
    LOSS_LIMIT = "LOSS_LIMIT"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    CONNECTIVITY = "CONNECTIVITY"
    MARKET_DATA = "MARKET_DATA"
    EXECUTION = "EXECUTION"
    STRATEGY = "STRATEGY"

class AlertEvent(EventBase):
    """告警事件 - alerts.v1"""
    
    alert_id: str = Field(..., description="告警ID")
    alert_type: AlertType = Field(..., description="告警类型")
    severity: AlertSeverity = Field(..., description="严重性")
    
    # 告警内容
    title: str = Field(..., description="告警标题")
    message: str = Field(..., description="告警消息")
    description: Optional[str] = Field(None, description="详细描述")
    
    # 关联信息
    account_id: Optional[str] = Field(None, description="关联账户")
    strategy_id: Optional[str] = Field(None, description="关联策略")
    symbol: Optional[str] = Field(None, description="关联交易对")
    order_id: Optional[str] = Field(None, description="关联订单")
    
    # 上下文数据
    context: Optional[Dict[str, Any]] = Field(None, description="上下文数据")
    
    # 状态信息
    status: str = Field(default="ACTIVE", description="告警状态")
    acknowledged: bool = Field(default=False, description="是否已确认")
    resolved: bool = Field(default=False, description="是否已解决")
    
    # 阈值信息
    threshold_value: Optional[Decimal] = Field(None, description="阈值")
    current_value: Optional[Decimal] = Field(None, description="当前值")
    
    @validator('threshold_value', 'current_value', pre=True)
    def parse_decimal(cls, v):
        if v is None:
            return v
        return Decimal(str(v))

class StrategyPerformanceEvent(EventBase):
    """策略性能事件 - strategy.performance.v1"""
    
    strategy_id: str = Field(..., description="策略ID")
    strategy_name: str = Field(..., description="策略名称")
    account_id: str = Field(..., description="账户ID")
    
    # 基础统计
    total_trades: int = Field(..., description="总交易次数")
    winning_trades: int = Field(..., description="盈利交易次数")
    losing_trades: int = Field(..., description="亏损交易次数")
    win_rate: Decimal = Field(..., description="胜率")
    
    # 损益指标
    total_pnl: Decimal = Field(..., description="总损益")
    gross_profit: Decimal = Field(..., description="总盈利")
    gross_loss: Decimal = Field(..., description="总亏损")
    profit_factor: Decimal = Field(..., description="盈利因子")
    
    # 风险调整指标
    sharpe_ratio: Optional[Decimal] = Field(None, description="夏普比率")
    sortino_ratio: Optional[Decimal] = Field(None, description="索提诺比率")
    calmar_ratio: Optional[Decimal] = Field(None, description="卡玛比率")
    
    # 回撤指标
    max_drawdown: Decimal = Field(..., description="最大回撤")
    max_drawdown_duration: Optional[int] = Field(None, description="最大回撤持续时间(天)")
    
    # 收益分布
    avg_win: Decimal = Field(..., description="平均盈利")
    avg_loss: Decimal = Field(..., description="平均亏损")
    largest_win: Decimal = Field(..., description="最大盈利")
    largest_loss: Decimal = Field(..., description="最大亏损")
    
    # 时间统计
    avg_hold_time: Optional[Decimal] = Field(None, description="平均持仓时间(分钟)")
    avg_win_time: Optional[Decimal] = Field(None, description="平均盈利时间(分钟)")
    avg_loss_time: Optional[Decimal] = Field(None, description="平均亏损时间(分钟)")
    
    # 收益序列
    daily_returns: Optional[List[Decimal]] = Field(None, description="日收益序列")
    monthly_returns: Optional[List[Decimal]] = Field(None, description="月收益序列")
    
    # 交易分布
    symbol_distribution: Optional[Dict[str, int]] = Field(None, description="交易对分布")
    time_distribution: Optional[Dict[str, int]] = Field(None, description="时间分布")
    
    # 资金使用
    max_capital_used: Decimal = Field(..., description="最大资金使用")
    avg_capital_used: Decimal = Field(..., description="平均资金使用")
    capital_efficiency: Decimal = Field(..., description="资金效率")
    
    @validator('win_rate', 'total_pnl', 'gross_profit', 'gross_loss', 'profit_factor',
              'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'max_drawdown',
              'avg_win', 'avg_loss', 'largest_win', 'largest_loss', 'avg_hold_time',
              'avg_win_time', 'avg_loss_time', 'max_capital_used', 'avg_capital_used',
              'capital_efficiency', pre=True)
    def parse_decimal(cls, v):
        if v is None:
            return v
        return Decimal(str(v))

# API响应模型

class PaginationInfo(BaseModel):
    """分页信息"""
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=50, ge=1, le=1000)
    total: int = 0
    total_pages: int = 0

class APIResponse(BaseModel):
    """API标准响应"""
    success: bool = True
    message: str = "OK"
    data: Optional[Any] = None
    pagination: Optional[PaginationInfo] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class PnLDataPoint(BaseModel):
    """PnL数据点"""
    timestamp: datetime
    symbol: str
    realized_pnl: Decimal
    unrealized_pnl: Decimal
    total_pnl: Decimal
    cumulative_pnl: Decimal

class PositionSnapshot(BaseModel):
    """仓位快照"""
    timestamp: datetime
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    market_price: Decimal
    market_value: Decimal
    unrealized_pnl: Decimal
    position_side: str

class FillRecord(BaseModel):
    """成交记录"""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    commission: Decimal
    order_id: str
    trade_id: Optional[str] = None

class RiskSnapshot(BaseModel):
    """风险快照"""
    timestamp: datetime
    account_id: str
    portfolio_value: Decimal
    total_pnl: Decimal
    var_1d: Optional[Decimal]
    max_drawdown: Decimal
    leverage: Optional[Decimal]
    risk_score: Optional[Decimal]
    num_positions: int

class PerformanceSnapshot(BaseModel):
    """性能快照"""
    timestamp: datetime
    strategy_id: str
    total_trades: int
    win_rate: Decimal
    total_pnl: Decimal
    sharpe_ratio: Optional[Decimal]
    max_drawdown: Decimal
    profit_factor: Decimal