"""
PnL API端点
==========

提供PnL查询和时间序列数据。
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ...schemas.api_responses import PnLResponse, PnLSummary, TimeSeriesPoint
from ...database import ClickHouseClient
from ..dependencies import get_db_client, get_timing_context

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/pnl",
    response_model=PnLResponse,
    summary="获取PnL数据",
    description="查询历史和实时PnL数据，支持时间范围和交易对过滤"
)
async def get_pnl(
    request: Request,
    start_time: Optional[datetime] = Query(
        None, 
        description="开始时间 (ISO格式，默认为24小时前)"
    ),
    end_time: Optional[datetime] = Query(
        None,
        description="结束时间 (ISO格式，默认为当前时间)"
    ),
    symbol: Optional[str] = Query(
        None,
        description="交易对过滤 (如 BTCUSDT)"
    ),
    interval: str = Query(
        "1h",
        description="时间间隔 (1m, 5m, 15m, 1h, 1d)",
        regex="^(1m|5m|15m|1h|1d)$"
    ),
    include_timeseries: bool = Query(
        True,
        description="是否包含时间序列数据"
    ),
    db: ClickHouseClient = Depends(get_db_client),
    timing: dict = Depends(get_timing_context)
):
    """获取PnL数据"""
    
    try:
        # 设置默认时间范围
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        # 验证时间范围
        if start_time >= end_time:
            raise HTTPException(
                status_code=400,
                detail="开始时间必须早于结束时间"
            )
        
        # 检查时间范围限制
        time_diff = end_time - start_time
        if time_diff.days > 90:
            raise HTTPException(
                status_code=400,
                detail="时间范围不能超过90天"
            )
        
        # 查询PnL汇总
        summary = await _get_pnl_summary(db, start_time, end_time, symbol)
        
        # 查询时间序列数据
        timeseries = []
        if include_timeseries:
            timeseries_df = await db.get_pnl_timeseries(
                start_time, end_time, symbol, interval
            )
            
            if not timeseries_df.empty:
                timeseries = [
                    TimeSeriesPoint(
                        timestamp=row['time_bucket'],
                        value=row['total_pnl']
                    )
                    for _, row in timeseries_df.iterrows()
                ]
        
        # 按交易对分组数据
        by_symbol = {}
        if symbol is None:  # 只有查询所有交易对时才返回分组数据
            by_symbol = await _get_pnl_by_symbol(db, start_time, end_time)
        
        # 构建响应
        response = PnLResponse(
            timestamp=datetime.utcnow(),
            summary=summary,
            timeseries=timeseries,
            by_symbol=by_symbol,
            query_params={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "symbol": symbol,
                "interval": interval,
                "include_timeseries": include_timeseries
            },
            execution_time_ms=timing["duration_ms"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询PnL数据失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="查询PnL数据失败"
        )


@router.get(
    "/pnl/summary",
    response_model=PnLSummary,
    summary="获取PnL汇总",
    description="获取指定时间范围的PnL汇总统计"
)
async def get_pnl_summary(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    symbol: Optional[str] = Query(None),
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取PnL汇总"""
    
    try:
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        summary = await _get_pnl_summary(db, start_time, end_time, symbol)
        return summary
        
    except Exception as e:
        logger.error(f"查询PnL汇总失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="查询PnL汇总失败"
        )


@router.get(
    "/pnl/realtime",
    summary="获取实时PnL",
    description="获取当前实时PnL状态"
)
async def get_realtime_pnl(
    symbol: Optional[str] = Query(None),
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取实时PnL"""
    
    try:
        # 查询最近5分钟的数据
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=5)
        
        # 获取最新的执行报告
        query = """
        SELECT 
            sum(total_realized_pnl) as total_realized,
            sum(total_unrealized_pnl) as total_unrealized,
            sum(total_commission) as total_commission,
            max(timestamp) as last_update
        FROM exec_reports
        WHERE timestamp >= %(start_time)s
        """
        
        params = {'start_time': start_time}
        if symbol:
            query += " AND symbol = %(symbol)s"
            params['symbol'] = symbol
        
        result_df = await db.query_to_dataframe(query, params)
        
        if result_df.empty:
            return {
                "total_realized_pnl": 0,
                "total_unrealized_pnl": 0,
                "total_pnl": 0,
                "total_commission": 0,
                "last_update": None,
                "symbol": symbol
            }
        
        row = result_df.iloc[0]
        
        return {
            "total_realized_pnl": float(row['total_realized'] or 0),
            "total_unrealized_pnl": float(row['total_unrealized'] or 0),
            "total_pnl": float((row['total_realized'] or 0) + (row['total_unrealized'] or 0)),
            "total_commission": float(row['total_commission'] or 0),
            "last_update": row['last_update'],
            "symbol": symbol
        }
        
    except Exception as e:
        logger.error(f"查询实时PnL失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="查询实时PnL失败"
        )


async def _get_pnl_summary(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime,
    symbol: Optional[str] = None
) -> PnLSummary:
    """获取PnL汇总统计"""
    
    # 构建查询
    symbol_filter = f"AND symbol = '{symbol}'" if symbol else ""
    
    query = f"""
    SELECT 
        sum(total_realized_pnl) as total_realized_pnl,
        sum(total_unrealized_pnl) as total_unrealized_pnl,
        sum(total_realized_pnl + total_unrealized_pnl) as total_pnl,
        count(*) as total_trades,
        sum(CASE WHEN total_realized_pnl > 0 THEN 1 ELSE 0 END) as profitable_trades,
        avg(total_realized_pnl) as avg_pnl,
        max(total_realized_pnl) as max_profit,
        min(total_realized_pnl) as max_loss
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
        return PnLSummary(
            total_realized_pnl=Decimal('0'),
            total_unrealized_pnl=Decimal('0'),
            total_pnl=Decimal('0'),
            daily_pnl=Decimal('0'),
            weekly_pnl=Decimal('0'),
            monthly_pnl=Decimal('0'),
            win_rate=Decimal('0'),
            profit_factor=Decimal('0'),
            max_drawdown=Decimal('0'),
            sharpe_ratio=Decimal('0')
        )
    
    row = result_df.iloc[0]
    
    # 计算胜率
    win_rate = Decimal('0')
    if row['total_trades'] and row['total_trades'] > 0:
        win_rate = Decimal(str(row['profitable_trades'])) / Decimal(str(row['total_trades'])) * 100
    
    # 计算盈利因子
    profit_factor = Decimal('0')
    if row['max_loss'] and row['max_loss'] < 0:
        total_profit = max(0, row['max_profit'] or 0)
        total_loss = abs(min(0, row['max_loss'] or 0))
        if total_loss > 0:
            profit_factor = Decimal(str(total_profit)) / Decimal(str(total_loss))
    
    # 获取时间段PnL
    daily_pnl = await _get_period_pnl(db, 1, symbol)
    weekly_pnl = await _get_period_pnl(db, 7, symbol)
    monthly_pnl = await _get_period_pnl(db, 30, symbol)
    
    # 获取风险指标
    risk_metrics = await _get_risk_metrics(db, start_time, end_time)
    
    return PnLSummary(
        total_realized_pnl=Decimal(str(row['total_realized_pnl'] or 0)),
        total_unrealized_pnl=Decimal(str(row['total_unrealized_pnl'] or 0)),
        total_pnl=Decimal(str(row['total_pnl'] or 0)),
        daily_pnl=daily_pnl,
        weekly_pnl=weekly_pnl,
        monthly_pnl=monthly_pnl,
        win_rate=win_rate,
        profit_factor=profit_factor,
        max_drawdown=risk_metrics.get('max_drawdown', Decimal('0')),
        sharpe_ratio=risk_metrics.get('sharpe_ratio', Decimal('0'))
    )


async def _get_period_pnl(
    db: ClickHouseClient,
    days: int,
    symbol: Optional[str] = None
) -> Decimal:
    """获取指定天数的PnL"""
    
    symbol_filter = f"AND symbol = '{symbol}'" if symbol else ""
    
    query = f"""
    SELECT sum(total_realized_pnl + total_unrealized_pnl) as period_pnl
    FROM exec_reports
    WHERE timestamp >= now() - INTERVAL {days} DAY
      {symbol_filter}
    """
    
    result = await db.query_scalar(query)
    return Decimal(str(result or 0))


async def _get_risk_metrics(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime
) -> dict:
    """获取风险指标"""
    
    query = """
    SELECT 
        max(max_drawdown) as max_drawdown,
        avg(sharpe_ratio) as sharpe_ratio
    FROM risk_metrics
    WHERE timestamp >= %(start_time)s 
      AND timestamp <= %(end_time)s
    """
    
    result_df = await db.query_to_dataframe(query, {
        'start_time': start_time,
        'end_time': end_time
    })
    
    if result_df.empty:
        return {'max_drawdown': Decimal('0'), 'sharpe_ratio': Decimal('0')}
    
    row = result_df.iloc[0]
    return {
        'max_drawdown': Decimal(str(row['max_drawdown'] or 0)),
        'sharpe_ratio': Decimal(str(row['sharpe_ratio'] or 0))
    }


async def _get_pnl_by_symbol(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime
) -> dict:
    """按交易对获取PnL"""
    
    query = """
    SELECT 
        symbol,
        sum(total_realized_pnl) as total_realized_pnl,
        sum(total_unrealized_pnl) as total_unrealized_pnl,
        sum(total_realized_pnl + total_unrealized_pnl) as total_pnl,
        count(*) as trade_count
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
        by_symbol[row['symbol']] = PnLSummary(
            total_realized_pnl=Decimal(str(row['total_realized_pnl'] or 0)),
            total_unrealized_pnl=Decimal(str(row['total_unrealized_pnl'] or 0)),
            total_pnl=Decimal(str(row['total_pnl'] or 0)),
            daily_pnl=Decimal('0'),  # 简化处理
            weekly_pnl=Decimal('0'),
            monthly_pnl=Decimal('0'),
            win_rate=Decimal('0'),
            profit_factor=Decimal('0'),
            max_drawdown=Decimal('0'),
            sharpe_ratio=Decimal('0')
        )
    
    return by_symbol