"""
成交记录API端点
==============

提供成交记录查询和分析。
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ...schemas.api_responses import FillResponse, FillSummary, PaginationInfo
from ...database import ClickHouseClient
from ..dependencies import get_db_client, get_timing_context, create_pagination_params

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/fills",
    response_model=FillResponse,
    summary="获取成交记录",
    description="查询成交记录，支持分页和过滤"
)
async def get_fills(
    request: Request,
    start_time: Optional[datetime] = Query(
        None,
        description="开始时间 (默认为24小时前)"
    ),
    end_time: Optional[datetime] = Query(
        None,
        description="结束时间 (默认为当前时间)"
    ),
    symbol: Optional[str] = Query(None, description="交易对过滤"),
    side: Optional[str] = Query(None, description="买卖方向 (BUY/SELL)"),
    order_id: Optional[str] = Query(None, description="订单ID过滤"),
    page: int = Query(1, description="页码", ge=1),
    page_size: int = Query(100, description="每页大小", ge=1, le=1000),
    db: ClickHouseClient = Depends(get_db_client),
    timing: dict = Depends(get_timing_context)
):
    """获取成交记录"""
    
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
        if time_diff.days > 30:
            raise HTTPException(
                status_code=400,
                detail="时间范围不能超过30天"
            )
        
        # 验证买卖方向
        if side and side.upper() not in ["BUY", "SELL"]:
            raise HTTPException(
                status_code=400,
                detail="买卖方向必须是BUY或SELL"
            )
        
        # 获取成交数据
        fills_df = await _get_fills_data(
            db, start_time, end_time, symbol, side, order_id, page, page_size
        )
        
        # 转换为填充列表
        fills = []
        if not fills_df.empty:
            fills = [
                {
                    "fill_id": row['fill_id'],
                    "trade_id": row['trade_id'],
                    "order_id": row['order_id'],
                    "symbol": row['symbol'],
                    "side": row['side'],
                    "quantity": float(row['quantity']),
                    "price": float(row['price']),
                    "commission": float(row['commission']),
                    "commission_asset": row['commission_asset'],
                    "timestamp": row['timestamp']
                }
                for _, row in fills_df.iterrows()
            ]
        
        # 计算汇总信息
        summary = await _calculate_fill_summary(
            db, start_time, end_time, symbol, side, order_id
        )
        
        # 获取总数用于分页
        total_count = await _get_fills_total_count(
            db, start_time, end_time, symbol, side, order_id
        )
        
        # 构建分页信息
        pagination = PaginationInfo(
            page=page,
            page_size=page_size,
            total_count=total_count,
            total_pages=(total_count + page_size - 1) // page_size,
            has_next=page * page_size < total_count,
            has_prev=page > 1
        )
        
        # 构建响应
        response = FillResponse(
            timestamp=datetime.utcnow(),
            fills=fills,
            summary=summary,
            pagination=pagination,
            query_params={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "symbol": symbol,
                "side": side,
                "order_id": order_id,
                "page": page,
                "page_size": page_size
            },
            execution_time_ms=timing["duration_ms"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询成交记录失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="查询成交记录失败"
        )


@router.get(
    "/fills/summary",
    response_model=FillSummary,
    summary="获取成交汇总",
    description="获取指定时间范围的成交汇总统计"
)
async def get_fills_summary(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    symbol: Optional[str] = Query(None),
    side: Optional[str] = Query(None),
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取成交汇总"""
    
    try:
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        summary = await _calculate_fill_summary(
            db, start_time, end_time, symbol, side
        )
        
        return summary
        
    except Exception as e:
        logger.error(f"查询成交汇总失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="查询成交汇总失败"
        )


@router.get(
    "/fills/{fill_id}",
    summary="获取特定成交记录",
    description="获取指定ID的成交记录详情"
)
async def get_fill_by_id(
    fill_id: str,
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取特定成交记录"""
    
    try:
        query = """
        SELECT 
            fill_id,
            trade_id,
            order_id,
            symbol,
            side,
            quantity,
            price,
            commission,
            commission_asset,
            timestamp,
            strategy_id,
            account_id,
            venue
        FROM fills
        WHERE fill_id = %(fill_id)s
        """
        
        result_df = await db.query_to_dataframe(query, {'fill_id': fill_id})
        
        if result_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"成交记录未找到: {fill_id}"
            )
        
        row = result_df.iloc[0]
        
        return {
            "fill_id": row['fill_id'],
            "trade_id": row['trade_id'],
            "order_id": row['order_id'],
            "symbol": row['symbol'],
            "side": row['side'],
            "quantity": float(row['quantity']),
            "price": float(row['price']),
            "commission": float(row['commission']),
            "commission_asset": row['commission_asset'],
            "timestamp": row['timestamp'],
            "strategy_id": row['strategy_id'],
            "account_id": row['account_id'],
            "venue": row['venue']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询成交记录失败 [{fill_id}]: {e}")
        raise HTTPException(
            status_code=500,
            detail="查询成交记录失败"
        )


@router.get(
    "/fills/analysis/volume",
    summary="成交量分析",
    description="获取成交量分析数据"
)
async def get_volume_analysis(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    symbol: Optional[str] = Query(None),
    interval: str = Query("1h", regex="^(5m|15m|1h|1d)$"),
    db: ClickHouseClient = Depends(get_db_client)
):
    """成交量分析"""
    
    try:
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        # 构建查询
        symbol_filter = f"AND symbol = '{symbol}'" if symbol else ""
        
        query = f"""
        SELECT 
            toStartOfInterval(timestamp, INTERVAL 1 {interval.replace('m', ' MINUTE').replace('h', ' HOUR').replace('d', ' DAY')}) as time_bucket,
            {f"symbol," if symbol else ""}
            sum(quantity) as total_volume,
            sum(quantity * price) as total_value,
            count(*) as fill_count,
            avg(price) as avg_price,
            min(price) as min_price,
            max(price) as max_price,
            sum(commission) as total_commission
        FROM fills
        WHERE timestamp >= %(start_time)s 
          AND timestamp <= %(end_time)s
          {symbol_filter}
        GROUP BY time_bucket{', symbol' if symbol else ''}
        ORDER BY time_bucket
        """
        
        result_df = await db.query_to_dataframe(query, {
            'start_time': start_time,
            'end_time': end_time
        })
        
        if result_df.empty:
            return {
                "volume_data": [],
                "total_volume": 0,
                "total_value": 0,
                "total_fills": 0,
                "avg_fill_size": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # 转换数据
        volume_data = [
            {
                "timestamp": row['time_bucket'],
                "volume": float(row['total_volume']),
                "value": float(row['total_value']),
                "fill_count": int(row['fill_count']),
                "avg_price": float(row['avg_price']),
                "min_price": float(row['min_price']),
                "max_price": float(row['max_price']),
                "commission": float(row['total_commission'])
            }
            for _, row in result_df.iterrows()
        ]
        
        # 计算汇总
        total_volume = result_df['total_volume'].sum()
        total_value = result_df['total_value'].sum()
        total_fills = result_df['fill_count'].sum()
        avg_fill_size = total_volume / total_fills if total_fills > 0 else 0
        
        return {
            "volume_data": volume_data,
            "total_volume": float(total_volume),
            "total_value": float(total_value),
            "total_fills": int(total_fills),
            "avg_fill_size": float(avg_fill_size),
            "query_params": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "symbol": symbol,
                "interval": interval
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"成交量分析失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="成交量分析失败"
        )


async def _get_fills_data(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    order_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 100
):
    """获取成交数据"""
    
    # 构建过滤条件
    filters = []
    params = {
        'start_time': start_time,
        'end_time': end_time,
        'page_size': page_size,
        'offset': (page - 1) * page_size
    }
    
    if symbol:
        filters.append("AND symbol = %(symbol)s")
        params['symbol'] = symbol.upper()
    
    if side:
        filters.append("AND side = %(side)s")
        params['side'] = side.upper()
    
    if order_id:
        filters.append("AND order_id = %(order_id)s")
        params['order_id'] = order_id
    
    filter_clause = " ".join(filters)
    
    query = f"""
    SELECT 
        fill_id,
        trade_id,
        order_id,
        symbol,
        side,
        quantity,
        price,
        commission,
        commission_asset,
        timestamp
    FROM fills
    WHERE timestamp >= %(start_time)s 
      AND timestamp <= %(end_time)s
      {filter_clause}
    ORDER BY timestamp DESC
    LIMIT %(page_size)s OFFSET %(offset)s
    """
    
    return await db.query_to_dataframe(query, params)


async def _get_fills_total_count(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    order_id: Optional[str] = None
) -> int:
    """获取成交记录总数"""
    
    # 构建过滤条件
    filters = []
    params = {
        'start_time': start_time,
        'end_time': end_time
    }
    
    if symbol:
        filters.append("AND symbol = %(symbol)s")
        params['symbol'] = symbol.upper()
    
    if side:
        filters.append("AND side = %(side)s")
        params['side'] = side.upper()
    
    if order_id:
        filters.append("AND order_id = %(order_id)s")
        params['order_id'] = order_id
    
    filter_clause = " ".join(filters)
    
    query = f"""
    SELECT count(*) as total_count
    FROM fills
    WHERE timestamp >= %(start_time)s 
      AND timestamp <= %(end_time)s
      {filter_clause}
    """
    
    result = await db.query_scalar(query, params)
    return int(result or 0)


async def _calculate_fill_summary(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime,
    symbol: Optional[str] = None,
    side: Optional[str] = None,
    order_id: Optional[str] = None
) -> FillSummary:
    """计算成交汇总"""
    
    # 构建过滤条件
    filters = []
    params = {
        'start_time': start_time,
        'end_time': end_time
    }
    
    if symbol:
        filters.append("AND symbol = %(symbol)s")
        params['symbol'] = symbol.upper()
    
    if side:
        filters.append("AND side = %(side)s")
        params['side'] = side.upper()
    
    if order_id:
        filters.append("AND order_id = %(order_id)s")
        params['order_id'] = order_id
    
    filter_clause = " ".join(filters)
    
    query = f"""
    SELECT 
        count(*) as total_fills,
        sum(quantity) as total_volume,
        sum(commission) as total_commission,
        avg(quantity) as avg_fill_size,
        avg(abs(price - lag(price) OVER (ORDER BY timestamp))) as avg_slippage
    FROM fills
    WHERE timestamp >= %(start_time)s 
      AND timestamp <= %(end_time)s
      {filter_clause}
    """
    
    result_df = await db.query_to_dataframe(query, params)
    
    if result_df.empty:
        return FillSummary(
            total_fills=0,
            total_volume=Decimal('0'),
            total_commission=Decimal('0'),
            avg_fill_size=Decimal('0'),
            avg_slippage=Decimal('0')
        )
    
    row = result_df.iloc[0]
    
    return FillSummary(
        total_fills=int(row['total_fills'] or 0),
        total_volume=Decimal(str(row['total_volume'] or 0)),
        total_commission=Decimal(str(row['total_commission'] or 0)),
        avg_fill_size=Decimal(str(row['avg_fill_size'] or 0)),
        avg_slippage=Decimal(str(row['avg_slippage'] or 0))
    )