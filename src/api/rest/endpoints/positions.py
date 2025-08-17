"""
持仓API端点
==========

提供持仓查询和历史持仓数据。
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ...schemas.api_responses import PositionResponse, PositionSummary
from ...database import ClickHouseClient
from ..dependencies import get_db_client, get_timing_context

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/positions",
    response_model=PositionResponse,
    summary="获取持仓数据",
    description="查询当前持仓和历史持仓快照"
)
async def get_positions(
    request: Request,
    symbol: Optional[str] = Query(None, description="交易对过滤"),
    include_history: bool = Query(False, description="是否包含历史快照"),
    history_hours: int = Query(24, description="历史快照时间范围(小时)", ge=1, le=168),
    db: ClickHouseClient = Depends(get_db_client),
    timing: dict = Depends(get_timing_context)
):
    """获取持仓数据"""
    
    try:
        # 获取当前持仓
        current_positions_df = await db.get_current_positions()
        
        # 过滤交易对
        if symbol and not current_positions_df.empty:
            current_positions_df = current_positions_df[
                current_positions_df['symbol'] == symbol.upper()
            ]
        
        # 转换为持仓列表
        current_positions = []
        if not current_positions_df.empty:
            current_positions = [
                {
                    "symbol": row['symbol'],
                    "quantity": float(row['quantity']),
                    "avg_price": float(row['avg_price']),
                    "market_value": float(row['market_value']),
                    "unrealized_pnl": float(row['unrealized_pnl']),
                    "realized_pnl": float(row['realized_pnl']),
                    "side": row['side']
                }
                for _, row in current_positions_df.iterrows()
            ]
        
        # 计算持仓汇总
        summary = await _calculate_position_summary(current_positions_df)
        
        # 获取历史快照
        historical_snapshots = []
        if include_history:
            historical_snapshots = await _get_historical_snapshots(
                db, symbol, history_hours
            )
        
        # 构建响应
        response = PositionResponse(
            timestamp=datetime.utcnow(),
            current_positions=current_positions,
            summary=summary,
            historical_snapshots=historical_snapshots,
            query_params={
                "symbol": symbol,
                "include_history": include_history,
                "history_hours": history_hours
            },
            execution_time_ms=timing["duration_ms"]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"查询持仓数据失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="查询持仓数据失败"
        )


@router.get(
    "/positions/current",
    summary="获取当前持仓",
    description="获取当前所有非零持仓"
)
async def get_current_positions(
    symbol: Optional[str] = Query(None),
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取当前持仓"""
    
    try:
        positions_df = await db.get_current_positions()
        
        if symbol and not positions_df.empty:
            positions_df = positions_df[
                positions_df['symbol'] == symbol.upper()
            ]
        
        if positions_df.empty:
            return {"positions": [], "count": 0}
        
        positions = [
            {
                "symbol": row['symbol'],
                "quantity": float(row['quantity']),
                "avg_price": float(row['avg_price']),
                "market_value": float(row['market_value']),
                "unrealized_pnl": float(row['unrealized_pnl']),
                "realized_pnl": float(row['realized_pnl']),
                "side": row['side'],
                "last_update": row['timestamp']
            }
            for _, row in positions_df.iterrows()
        ]
        
        return {
            "positions": positions,
            "count": len(positions),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"查询当前持仓失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="查询当前持仓失败"
        )


@router.get(
    "/positions/{symbol}",
    summary="获取特定交易对持仓",
    description="获取指定交易对的详细持仓信息"
)
async def get_position_by_symbol(
    symbol: str,
    include_history: bool = Query(False),
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取特定交易对持仓"""
    
    try:
        symbol = symbol.upper()
        
        # 获取当前持仓
        query = """
        SELECT 
            symbol,
            quantity,
            avg_price,
            market_value,
            unrealized_pnl,
            realized_pnl,
            side,
            timestamp
        FROM positions
        WHERE symbol = %(symbol)s
          AND timestamp = (
              SELECT max(timestamp) FROM positions WHERE symbol = %(symbol)s
          )
        """
        
        result_df = await db.query_to_dataframe(query, {'symbol': symbol})
        
        if result_df.empty:
            return {
                "symbol": symbol,
                "current_position": None,
                "history": [] if include_history else None,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        row = result_df.iloc[0]
        current_position = {
            "symbol": row['symbol'],
            "quantity": float(row['quantity']),
            "avg_price": float(row['avg_price']),
            "market_value": float(row['market_value']),
            "unrealized_pnl": float(row['unrealized_pnl']),
            "realized_pnl": float(row['realized_pnl']),
            "side": row['side'],
            "last_update": row['timestamp']
        }
        
        # 获取历史数据
        history = None
        if include_history:
            history = await _get_position_history(db, symbol, 24)
        
        return {
            "symbol": symbol,
            "current_position": current_position,
            "history": history,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"查询交易对持仓失败 [{symbol}]: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"查询交易对持仓失败"
        )


@router.get(
    "/positions/exposure",
    summary="获取敞口分析",
    description="获取持仓敞口和风险分析"
)
async def get_exposure_analysis(
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取敞口分析"""
    
    try:
        # 获取当前持仓
        positions_df = await db.get_current_positions()
        
        if positions_df.empty:
            return {
                "total_exposure": 0,
                "long_exposure": 0,
                "short_exposure": 0,
                "net_exposure": 0,
                "positions_count": 0,
                "concentration": {},
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # 计算敞口
        total_exposure = positions_df['market_value'].abs().sum()
        long_exposure = positions_df[
            positions_df['side'] == 'LONG'
        ]['market_value'].sum()
        short_exposure = positions_df[
            positions_df['side'] == 'SHORT'
        ]['market_value'].abs().sum()
        net_exposure = long_exposure - short_exposure
        
        # 计算集中度
        concentration = {}
        for _, row in positions_df.iterrows():
            symbol = row['symbol']
            exposure_pct = abs(row['market_value']) / total_exposure * 100
            concentration[symbol] = {
                "exposure_amount": float(row['market_value']),
                "exposure_percentage": float(exposure_pct),
                "side": row['side']
            }
        
        # 按敞口排序
        concentration = dict(
            sorted(
                concentration.items(),
                key=lambda x: x[1]['exposure_percentage'],
                reverse=True
            )
        )
        
        return {
            "total_exposure": float(total_exposure),
            "long_exposure": float(long_exposure),
            "short_exposure": float(short_exposure),
            "net_exposure": float(net_exposure),
            "positions_count": len(positions_df),
            "concentration": concentration,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取敞口分析失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="获取敞口分析失败"
        )


async def _calculate_position_summary(positions_df) -> PositionSummary:
    """计算持仓汇总"""
    
    if positions_df.empty:
        return PositionSummary(
            total_positions=0,
            total_exposure=Decimal('0'),
            total_market_value=Decimal('0'),
            largest_position_pct=Decimal('0'),
            concentration_risk="LOW"
        )
    
    total_positions = len(positions_df)
    total_exposure = positions_df['market_value'].abs().sum()
    total_market_value = positions_df['market_value'].sum()
    
    # 计算最大持仓占比
    largest_position = positions_df['market_value'].abs().max()
    largest_position_pct = Decimal('0')
    if total_exposure > 0:
        largest_position_pct = Decimal(str(largest_position / total_exposure * 100))
    
    # 评估集中度风险
    concentration_risk = "LOW"
    if largest_position_pct > 50:
        concentration_risk = "HIGH"
    elif largest_position_pct > 30:
        concentration_risk = "MEDIUM"
    
    return PositionSummary(
        total_positions=total_positions,
        total_exposure=Decimal(str(total_exposure)),
        total_market_value=Decimal(str(total_market_value)),
        largest_position_pct=largest_position_pct,
        concentration_risk=concentration_risk
    )


async def _get_historical_snapshots(
    db: ClickHouseClient,
    symbol: Optional[str],
    hours: int
) -> List[dict]:
    """获取历史持仓快照"""
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    
    symbol_filter = f"AND symbol = '{symbol}'" if symbol else ""
    
    query = f"""
    SELECT 
        timestamp,
        symbol,
        sum(market_value) as total_market_value,
        sum(unrealized_pnl) as total_unrealized_pnl,
        count(*) as positions_count
    FROM positions
    WHERE timestamp >= %(start_time)s
      AND timestamp <= %(end_time)s
      {symbol_filter}
    GROUP BY timestamp, symbol
    ORDER BY timestamp DESC
    LIMIT 100
    """
    
    result_df = await db.query_to_dataframe(query, {
        'start_time': start_time,
        'end_time': end_time
    })
    
    if result_df.empty:
        return []
    
    return [
        {
            "timestamp": row['timestamp'],
            "symbol": row.get('symbol'),
            "total_market_value": float(row['total_market_value']),
            "total_unrealized_pnl": float(row['total_unrealized_pnl']),
            "positions_count": int(row['positions_count'])
        }
        for _, row in result_df.iterrows()
    ]


async def _get_position_history(
    db: ClickHouseClient,
    symbol: str,
    hours: int
) -> List[dict]:
    """获取特定交易对的持仓历史"""
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=hours)
    
    query = """
    SELECT 
        timestamp,
        quantity,
        avg_price,
        market_value,
        unrealized_pnl,
        side
    FROM positions
    WHERE symbol = %(symbol)s
      AND timestamp >= %(start_time)s
      AND timestamp <= %(end_time)s
    ORDER BY timestamp DESC
    LIMIT 100
    """
    
    result_df = await db.query_to_dataframe(query, {
        'symbol': symbol,
        'start_time': start_time,
        'end_time': end_time
    })
    
    if result_df.empty:
        return []
    
    return [
        {
            "timestamp": row['timestamp'],
            "quantity": float(row['quantity']),
            "avg_price": float(row['avg_price']),
            "market_value": float(row['market_value']),
            "unrealized_pnl": float(row['unrealized_pnl']),
            "side": row['side']
        }
        for _, row in result_df.iterrows()
    ]