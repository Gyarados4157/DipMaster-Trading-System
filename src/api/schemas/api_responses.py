"""
API响应模式定义
==============

定义所有REST API端点的响应格式，确保API接口的一致性。
"""

from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

from .kafka_events import Position, Fill, Order, AlertSeverity, RiskMetric


class PaginationInfo(BaseModel):
    """分页信息"""
    page: int = Field(..., description="当前页码", ge=1)
    page_size: int = Field(..., description="每页大小", ge=1, le=1000)
    total_count: int = Field(..., description="总数量", ge=0)
    total_pages: int = Field(..., description="总页数", ge=0)
    has_next: bool = Field(..., description="是否有下一页")
    has_prev: bool = Field(..., description="是否有上一页")


class TimeSeriesPoint(BaseModel):
    """时间序列数据点"""
    timestamp: datetime = Field(..., description="时间戳")
    value: Decimal = Field(..., description="数值")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class PnLSummary(BaseModel):
    """PnL汇总信息"""
    total_realized_pnl: Decimal = Field(..., description="总已实现盈亏")
    total_unrealized_pnl: Decimal = Field(..., description="总未实现盈亏")
    total_pnl: Decimal = Field(..., description="总盈亏")
    daily_pnl: Decimal = Field(..., description="当日盈亏")
    weekly_pnl: Decimal = Field(..., description="本周盈亏")
    monthly_pnl: Decimal = Field(..., description="本月盈亏")
    win_rate: Decimal = Field(..., description="胜率(%)")
    profit_factor: Decimal = Field(..., description="盈利因子")
    max_drawdown: Decimal = Field(..., description="最大回撤")
    sharpe_ratio: Decimal = Field(..., description="夏普比率")
    
    class Config:
        json_encoders = {Decimal: str}


class PnLResponse(BaseModel):
    """
    GET /api/pnl 响应
    历史和实时PnL数据
    """
    success: bool = Field(True, description="请求是否成功")
    timestamp: datetime = Field(..., description="响应时间戳")
    
    # PnL汇总
    summary: PnLSummary = Field(..., description="PnL汇总")
    
    # 时间序列数据
    timeseries: List[TimeSeriesPoint] = Field(default_factory=list, description="PnL时间序列")
    
    # 按交易对分组
    by_symbol: Dict[str, PnLSummary] = Field(default_factory=dict, description="按交易对分组的PnL")
    
    # 查询信息
    query_params: Dict[str, Any] = Field(default_factory=dict, description="查询参数")
    execution_time_ms: int = Field(..., description="查询执行时间(毫秒)")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class PositionSummary(BaseModel):
    """持仓汇总信息"""
    total_positions: int = Field(..., description="总持仓数")
    total_exposure: Decimal = Field(..., description="总敞口")
    total_market_value: Decimal = Field(..., description="总市值")
    largest_position_pct: Decimal = Field(..., description="最大持仓占比(%)")
    concentration_risk: str = Field(..., description="集中度风险评级")
    
    class Config:
        json_encoders = {Decimal: str}


class PositionResponse(BaseModel):
    """
    GET /api/positions 响应
    当前持仓和历史持仓
    """
    success: bool = Field(True, description="请求是否成功")
    timestamp: datetime = Field(..., description="响应时间戳")
    
    # 当前持仓
    current_positions: List[Position] = Field(default_factory=list, description="当前持仓列表")
    
    # 持仓汇总
    summary: PositionSummary = Field(..., description="持仓汇总")
    
    # 历史持仓快照
    historical_snapshots: List[Dict[str, Any]] = Field(default_factory=list, description="历史持仓快照")
    
    # 查询信息
    query_params: Dict[str, Any] = Field(default_factory=dict, description="查询参数")
    execution_time_ms: int = Field(..., description="查询执行时间(毫秒)")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class FillSummary(BaseModel):
    """成交汇总信息"""
    total_fills: int = Field(..., description="总成交数")
    total_volume: Decimal = Field(..., description="总成交量")
    total_commission: Decimal = Field(..., description="总手续费")
    avg_fill_size: Decimal = Field(..., description="平均成交规模")
    avg_slippage: Decimal = Field(..., description="平均滑点")
    
    class Config:
        json_encoders = {Decimal: str}


class FillResponse(BaseModel):
    """
    GET /api/fills 响应
    成交记录查询
    """
    success: bool = Field(True, description="请求是否成功")
    timestamp: datetime = Field(..., description="响应时间戳")
    
    # 成交列表
    fills: List[Fill] = Field(default_factory=list, description="成交列表")
    
    # 成交汇总
    summary: FillSummary = Field(..., description="成交汇总")
    
    # 分页信息
    pagination: Optional[PaginationInfo] = Field(None, description="分页信息")
    
    # 查询信息
    query_params: Dict[str, Any] = Field(default_factory=dict, description="查询参数")
    execution_time_ms: int = Field(..., description="查询执行时间(毫秒)")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class RiskSummary(BaseModel):
    """风险汇总信息"""
    overall_risk_score: Decimal = Field(..., description="整体风险评分(0-100)")
    risk_level: str = Field(..., description="风险等级", regex="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    active_alerts: int = Field(..., description="活跃告警数")
    breached_limits: int = Field(..., description="违规限制数")
    var_utilization: Decimal = Field(..., description="VaR利用率(%)")
    
    class Config:
        json_encoders = {Decimal: str}


class RiskResponse(BaseModel):
    """
    GET /api/risk 响应
    风险指标和限制状态
    """
    success: bool = Field(True, description="请求是否成功")
    timestamp: datetime = Field(..., description="响应时间戳")
    
    # 风险汇总
    summary: RiskSummary = Field(..., description="风险汇总")
    
    # 详细风险指标
    metrics: List[RiskMetric] = Field(default_factory=list, description="风险指标列表")
    
    # 限制检查
    limit_checks: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="限制检查结果")
    
    # 历史风险指标
    historical_metrics: List[TimeSeriesPoint] = Field(default_factory=list, description="历史风险指标")
    
    # 查询信息
    query_params: Dict[str, Any] = Field(default_factory=dict, description="查询参数")
    execution_time_ms: int = Field(..., description="查询执行时间(毫秒)")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class AlertInfo(BaseModel):
    """告警信息"""
    alert_id: str = Field(..., description="告警ID")
    timestamp: datetime = Field(..., description="告警时间")
    severity: AlertSeverity = Field(..., description="严重级别")
    code: str = Field(..., description="告警代码")
    title: str = Field(..., description="告警标题")
    message: str = Field(..., description="告警描述")
    source: str = Field(..., description="告警来源")
    symbol: Optional[str] = Field(None, description="相关交易对")
    resolved: bool = Field(default=False, description="是否已解决")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AlertSummary(BaseModel):
    """告警汇总信息"""
    total_alerts: int = Field(..., description="总告警数")
    active_alerts: int = Field(..., description="活跃告警数")
    critical_alerts: int = Field(..., description="严重告警数")
    resolved_alerts: int = Field(..., description="已解决告警数")
    alert_rate: Decimal = Field(..., description="告警率(每小时)")
    
    class Config:
        json_encoders = {Decimal: str}


class AlertResponse(BaseModel):
    """
    GET /api/alerts 响应
    告警历史和状态
    """
    success: bool = Field(True, description="请求是否成功")
    timestamp: datetime = Field(..., description="响应时间戳")
    
    # 告警列表
    alerts: List[AlertInfo] = Field(default_factory=list, description="告警列表")
    
    # 告警汇总
    summary: AlertSummary = Field(..., description="告警汇总")
    
    # 分页信息
    pagination: Optional[PaginationInfo] = Field(None, description="分页信息")
    
    # 查询信息
    query_params: Dict[str, Any] = Field(default_factory=dict, description="查询参数")
    execution_time_ms: int = Field(..., description="查询执行时间(毫秒)")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


class PerformanceMetrics(BaseModel):
    """性能指标"""
    win_rate: Decimal = Field(..., description="胜率(%)")
    profit_factor: Decimal = Field(..., description="盈利因子")
    sharpe_ratio: Decimal = Field(..., description="夏普比率")
    max_drawdown: Decimal = Field(..., description="最大回撤(%)")
    avg_holding_time: Decimal = Field(..., description="平均持仓时间(分钟)")
    total_trades: int = Field(..., description="总交易数")
    profitable_trades: int = Field(..., description="盈利交易数")
    avg_profit_per_trade: Decimal = Field(..., description="平均每笔盈利")
    
    class Config:
        json_encoders = {Decimal: str}


class PerformanceResponse(BaseModel):
    """
    GET /api/performance 响应
    策略性能分析
    """
    success: bool = Field(True, description="请求是否成功")
    timestamp: datetime = Field(..., description="响应时间戳")
    
    # 核心性能指标
    metrics: PerformanceMetrics = Field(..., description="性能指标")
    
    # 时间序列性能
    daily_returns: List[TimeSeriesPoint] = Field(default_factory=list, description="日收益率序列")
    cumulative_pnl: List[TimeSeriesPoint] = Field(default_factory=list, description="累计PnL序列")
    drawdown_series: List[TimeSeriesPoint] = Field(default_factory=list, description="回撤序列")
    
    # 按交易对分析
    by_symbol_performance: Dict[str, PerformanceMetrics] = Field(default_factory=dict, description="按交易对的性能")
    
    # 查询信息
    query_params: Dict[str, Any] = Field(default_factory=dict, description="查询参数")
    execution_time_ms: int = Field(..., description="查询执行时间(毫秒)")
    
    class Config:
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }


# WebSocket消息格式
class WebSocketMessage(BaseModel):
    """WebSocket消息基类"""
    type: str = Field(..., description="消息类型")
    timestamp: datetime = Field(..., description="消息时间戳")
    data: Dict[str, Any] = Field(..., description="消息数据")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """错误响应格式"""
    success: bool = Field(False, description="请求是否成功")
    error_code: str = Field(..., description="错误代码")
    error_message: str = Field(..., description="错误描述")
    timestamp: datetime = Field(..., description="错误时间戳")
    request_id: Optional[str] = Field(None, description="请求ID")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }