"""
性能分析API端点
==============

提供策略性能分析和回测数据。
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ...schemas.api_responses import PerformanceResponse, PerformanceMetrics, TimeSeriesPoint
from ...database import ClickHouseClient
from ..dependencies import get_db_client, get_timing_context

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/performance",
    response_model=PerformanceResponse,
    summary="获取策略性能分析",
    description="查询策略性能指标和分析数据"
)
async def get_performance_analysis(
    request: Request,
    start_time: Optional[datetime] = Query(
        None,
        description="开始时间 (默认为30天前)"
    ),
    end_time: Optional[datetime] = Query(
        None,
        description="结束时间 (默认为当前时间)"
    ),
    symbol: Optional[str] = Query(None, description="交易对过滤"),
    include_daily_returns: bool = Query(
        True,
        description="是否包含日收益率序列"
    ),
    include_drawdown: bool = Query(
        True,
        description="是否包含回撤序列"
    ),
    db: ClickHouseClient = Depends(get_db_client),
    timing: dict = Depends(get_timing_context)
):
    """获取策略性能分析"""
    
    try:
        # 设置默认时间范围
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=30)
        
        # 验证时间范围
        if start_time >= end_time:
            raise HTTPException(
                status_code=400,
                detail="开始时间必须早于结束时间"
            )
        
        # 获取核心性能指标
        metrics = await _calculate_performance_metrics(db, start_time, end_time, symbol)
        
        # 获取日收益率序列
        daily_returns = []
        if include_daily_returns:
            daily_returns = await _get_daily_returns(db, start_time, end_time, symbol)
        
        # 获取累计PnL序列
        cumulative_pnl = await _get_cumulative_pnl(db, start_time, end_time, symbol)
        
        # 获取回撤序列
        drawdown_series = []
        if include_drawdown:
            drawdown_series = await _get_drawdown_series(db, start_time, end_time, symbol)
        
        # 按交易对分析
        by_symbol_performance = {}
        if symbol is None:  # 只有查询所有交易对时才返回分组数据
            by_symbol_performance = await _get_performance_by_symbol(
                db, start_time, end_time
            )
        
        # 构建响应
        response = PerformanceResponse(
            timestamp=datetime.utcnow(),
            metrics=metrics,
            daily_returns=daily_returns,
            cumulative_pnl=cumulative_pnl,
            drawdown_series=drawdown_series,
            by_symbol_performance=by_symbol_performance,
            query_params={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "symbol": symbol,
                "include_daily_returns": include_daily_returns,
                "include_drawdown": include_drawdown
            },
            execution_time_ms=timing["duration_ms"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"性能分析失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="性能分析失败"
        )


@router.get(
    "/performance/summary",
    response_model=PerformanceMetrics,
    summary="获取性能指标汇总",
    description="获取核心性能指标汇总"
)
async def get_performance_summary(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    symbol: Optional[str] = Query(None),
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取性能指标汇总"""
    
    try:
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=30)
        
        metrics = await _calculate_performance_metrics(db, start_time, end_time, symbol)
        return metrics
        
    except Exception as e:
        logger.error(f"获取性能汇总失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="获取性能汇总失败"
        )


@router.get(
    "/performance/trades",
    summary="获取交易分析",
    description="获取交易明细和统计分析"
)
async def get_trade_analysis(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    symbol: Optional[str] = Query(None),
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取交易分析"""
    
    try:
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=30)
        
        # 获取交易统计
        symbol_filter = f"AND symbol = '{symbol}'" if symbol else ""
        
        query = f"""
        SELECT 
            count(*) as total_trades,
            sum(CASE WHEN total_realized_pnl > 0 THEN 1 ELSE 0 END) as profitable_trades,
            sum(CASE WHEN total_realized_pnl < 0 THEN 1 ELSE 0 END) as losing_trades,
            sum(total_realized_pnl) as total_pnl,
            avg(total_realized_pnl) as avg_pnl_per_trade,
            max(total_realized_pnl) as best_trade,
            min(total_realized_pnl) as worst_trade,
            stddev(total_realized_pnl) as pnl_std,
            sum(total_commission) as total_commission,
            avg(execution_time_ms) as avg_execution_time_ms
        FROM exec_reports
        WHERE timestamp >= %(start_time)s 
          AND timestamp <= %(end_time)s
          {symbol_filter}
        """
        
        result_df = await db.query_to_dataframe(query, {
            'start_time': start_time,
            'end_time': end_time
        })
        
        if result_df.empty:
            return {
                "trade_stats": {},
                "win_loss_analysis": {},
                "timestamp": datetime.utcnow().isoformat()
            }
        
        row = result_df.iloc[0]
        
        # 交易统计
        total_trades = int(row['total_trades'] or 0)
        profitable_trades = int(row['profitable_trades'] or 0)
        losing_trades = int(row['losing_trades'] or 0)
        
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        # 盈亏分析
        total_profit = max(0, float(row['total_pnl'] or 0))
        total_loss = abs(min(0, float(row['total_pnl'] or 0)))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        trade_stats = {
            "total_trades": total_trades,
            "profitable_trades": profitable_trades,
            "losing_trades": losing_trades,
            "win_rate_percent": win_rate,
            "total_pnl": float(row['total_pnl'] or 0),
            "avg_pnl_per_trade": float(row['avg_pnl_per_trade'] or 0),
            "best_trade": float(row['best_trade'] or 0),
            "worst_trade": float(row['worst_trade'] or 0),
            "pnl_standard_deviation": float(row['pnl_std'] or 0),
            "total_commission": float(row['total_commission'] or 0),
            "avg_execution_time_ms": float(row['avg_execution_time_ms'] or 0)
        }
        
        win_loss_analysis = {
            "profit_factor": profit_factor,
            "total_profit": total_profit,
            "total_loss": total_loss,
            "avg_win": total_profit / profitable_trades if profitable_trades > 0 else 0,
            "avg_loss": total_loss / losing_trades if losing_trades > 0 else 0,
            "largest_win": float(row['best_trade'] or 0),
            "largest_loss": abs(float(row['worst_trade'] or 0))
        }
        
        return {
            "trade_stats": trade_stats,
            "win_loss_analysis": win_loss_analysis,
            "query_params": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "symbol": symbol
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"交易分析失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="交易分析失败"
        )


@router.get(
    "/performance/returns",
    summary="获取收益率分析",
    description="获取日/周/月收益率分析"
)
async def get_returns_analysis(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    period: str = Query("daily", regex="^(daily|weekly|monthly)$"),
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取收益率分析"""
    
    try:
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            days = 30 if period == "daily" else 90 if period == "weekly" else 365
            start_time = end_time - timedelta(days=days)
        
        # 根据周期设置间隔
        interval_map = {
            "daily": "1 DAY",
            "weekly": "1 WEEK", 
            "monthly": "1 MONTH"
        }
        interval = interval_map[period]
        
        query = f"""
        SELECT 
            toStartOfInterval(timestamp, INTERVAL {interval}) as period_start,
            sum(total_realized_pnl + total_unrealized_pnl) as period_pnl,
            count(*) as trade_count
        FROM exec_reports
        WHERE timestamp >= %(start_time)s 
          AND timestamp <= %(end_time)s
        GROUP BY period_start
        ORDER BY period_start
        """
        
        result_df = await db.query_to_dataframe(query, {
            'start_time': start_time,
            'end_time': end_time
        })
        
        if result_df.empty:
            return {
                "returns": [],
                "statistics": {},
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # 计算收益率
        returns_data = []
        period_returns = []
        
        for _, row in result_df.iterrows():
            period_pnl = float(row['period_pnl'])
            returns_data.append({
                "period": row['period_start'],
                "pnl": period_pnl,
                "trade_count": int(row['trade_count'])
            })
            period_returns.append(period_pnl)
        
        # 计算统计指标
        import numpy as np
        
        if period_returns:
            returns_array = np.array(period_returns)
            
            statistics = {
                "total_periods": len(returns_array),
                "positive_periods": int(np.sum(returns_array > 0)),
                "negative_periods": int(np.sum(returns_array < 0)),
                "avg_return": float(np.mean(returns_array)),
                "return_std": float(np.std(returns_array)),
                "best_period": float(np.max(returns_array)),
                "worst_period": float(np.min(returns_array)),
                "sharpe_ratio": float(np.mean(returns_array) / np.std(returns_array)) if np.std(returns_array) > 0 else 0,
                "win_rate_percent": float(np.sum(returns_array > 0) / len(returns_array) * 100)
            }
        else:
            statistics = {}
        
        return {
            "returns": returns_data,
            "statistics": statistics,
            "period": period,
            "query_params": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "period": period
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"收益率分析失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="收益率分析失败"
        )


async def _calculate_performance_metrics(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime,
    symbol: Optional[str] = None
) -> PerformanceMetrics:
    """计算性能指标"""
    
    # 获取基础数据
    perf_data = await db.get_performance_metrics(start_time, end_time)
    
    if not perf_data:
        return PerformanceMetrics(
            win_rate=Decimal('0'),
            profit_factor=Decimal('0'),
            sharpe_ratio=Decimal('0'),
            max_drawdown=Decimal('0'),
            avg_holding_time=Decimal('0'),
            total_trades=0,
            profitable_trades=0,
            avg_profit_per_trade=Decimal('0')
        )
    
    # 计算盈利因子
    total_profit = max(0, perf_data.get('max_profit', 0))
    total_loss = abs(min(0, perf_data.get('max_loss', 0)))
    profit_factor = Decimal(str(total_profit / total_loss)) if total_loss > 0 else Decimal('0')
    
    return PerformanceMetrics(
        win_rate=Decimal(str(perf_data.get('win_rate', 0))),
        profit_factor=profit_factor,
        sharpe_ratio=Decimal(str(perf_data.get('avg_sharpe_ratio', 0))),
        max_drawdown=Decimal(str(abs(perf_data.get('max_drawdown', 0)))),
        avg_holding_time=Decimal('96'),  # DipMaster平均96分钟
        total_trades=int(perf_data.get('total_trades', 0)),
        profitable_trades=int(perf_data.get('profitable_trades', 0)),
        avg_profit_per_trade=Decimal(str(perf_data.get('avg_pnl_per_trade', 0)))
    )


async def _get_daily_returns(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime,
    symbol: Optional[str] = None
) -> list:
    """获取日收益率序列"""
    
    symbol_filter = f"AND symbol = '{symbol}'" if symbol else ""
    
    query = f"""
    SELECT 
        toStartOfDay(timestamp) as date,
        sum(total_realized_pnl + total_unrealized_pnl) as daily_pnl
    FROM exec_reports
    WHERE timestamp >= %(start_time)s 
      AND timestamp <= %(end_time)s
      {symbol_filter}
    GROUP BY date
    ORDER BY date
    """
    
    result_df = await db.query_to_dataframe(query, {
        'start_time': start_time,
        'end_time': end_time
    })
    
    if result_df.empty:
        return []
    
    return [
        TimeSeriesPoint(
            timestamp=row['date'],
            value=Decimal(str(row['daily_pnl']))
        )
        for _, row in result_df.iterrows()
    ]


async def _get_cumulative_pnl(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime,
    symbol: Optional[str] = None
) -> list:
    """获取累计PnL序列"""
    
    symbol_filter = f"AND symbol = '{symbol}'" if symbol else ""
    
    query = f"""
    SELECT 
        toStartOfDay(timestamp) as date,
        sum(total_realized_pnl + total_unrealized_pnl) as daily_pnl
    FROM exec_reports
    WHERE timestamp >= %(start_time)s 
      AND timestamp <= %(end_time)s
      {symbol_filter}
    GROUP BY date
    ORDER BY date
    """
    
    result_df = await db.query_to_dataframe(query, {
        'start_time': start_time,
        'end_time': end_time
    })
    
    if result_df.empty:
        return []
    
    # 计算累计值
    cumulative_pnl = 0
    result = []
    
    for _, row in result_df.iterrows():
        cumulative_pnl += float(row['daily_pnl'])
        result.append(TimeSeriesPoint(
            timestamp=row['date'],
            value=Decimal(str(cumulative_pnl))
        ))
    
    return result


async def _get_drawdown_series(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime,
    symbol: Optional[str] = None
) -> list:
    """获取回撤序列"""
    
    # 获取累计PnL
    cumulative_pnl = await _get_cumulative_pnl(db, start_time, end_time, symbol)
    
    if not cumulative_pnl:
        return []
    
    # 计算回撤
    drawdown_series = []
    peak = 0
    
    for point in cumulative_pnl:
        current_value = float(point.value)
        peak = max(peak, current_value)
        drawdown = (current_value - peak) / peak * 100 if peak > 0 else 0
        
        drawdown_series.append(TimeSeriesPoint(
            timestamp=point.timestamp,
            value=Decimal(str(drawdown))
        ))
    
    return drawdown_series


async def _get_performance_by_symbol(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime
) -> Dict[str, PerformanceMetrics]:
    """按交易对获取性能指标"""
    
    query = """
    SELECT 
        symbol,
        count(*) as total_trades,
        sum(CASE WHEN total_realized_pnl > 0 THEN 1 ELSE 0 END) as profitable_trades,
        sum(total_realized_pnl) as total_pnl,
        avg(total_realized_pnl) as avg_pnl_per_trade,
        max(total_realized_pnl) as max_profit,
        min(total_realized_pnl) as max_loss
    FROM exec_reports
    WHERE timestamp >= %(start_time)s 
      AND timestamp <= %(end_time)s
    GROUP BY symbol
    ORDER BY total_pnl DESC
    """
    
    result_df = await db.query_to_dataframe(query, {
        'start_time': start_time,
        'end_time': end_time
    })
    
    by_symbol = {}
    
    for _, row in result_df.iterrows():
        symbol = row['symbol']
        total_trades = int(row['total_trades'])
        profitable_trades = int(row['profitable_trades'])
        
        win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = max(0, float(row['max_profit'] or 0))
        total_loss = abs(min(0, float(row['max_loss'] or 0)))
        profit_factor = Decimal(str(total_profit / total_loss)) if total_loss > 0 else Decimal('0')
        
        by_symbol[symbol] = PerformanceMetrics(
            win_rate=Decimal(str(win_rate)),
            profit_factor=profit_factor,
            sharpe_ratio=Decimal('0'),  # 简化处理
            max_drawdown=Decimal('0'),
            avg_holding_time=Decimal('96'),
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            avg_profit_per_trade=Decimal(str(row['avg_pnl_per_trade'] or 0))
        )
    
    return by_symbol