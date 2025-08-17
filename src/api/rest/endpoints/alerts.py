"""
告警API端点
==========

提供告警查询和管理功能。
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ...schemas.api_responses import AlertResponse, AlertSummary, AlertInfo, PaginationInfo
from ...database import ClickHouseClient
from ..dependencies import get_db_client, get_timing_context

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/alerts",
    response_model=AlertResponse,
    summary="获取告警记录",
    description="查询告警记录，支持分页和过滤"
)
async def get_alerts(
    request: Request,
    start_time: Optional[datetime] = Query(
        None,
        description="开始时间 (默认为24小时前)"
    ),
    end_time: Optional[datetime] = Query(
        None,
        description="结束时间 (默认为当前时间)"
    ),
    severity: Optional[str] = Query(
        None,
        description="严重级别过滤 (LOW/MEDIUM/HIGH/CRITICAL)"
    ),
    source: Optional[str] = Query(None, description="告警来源过滤"),
    symbol: Optional[str] = Query(None, description="交易对过滤"),
    resolved: Optional[bool] = Query(None, description="是否已解决"),
    page: int = Query(1, description="页码", ge=1),
    page_size: int = Query(100, description="每页大小", ge=1, le=1000),
    db: ClickHouseClient = Depends(get_db_client),
    timing: dict = Depends(get_timing_context)
):
    """获取告警记录"""
    
    try:
        # 设置默认时间范围
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        # 验证严重级别
        if severity and severity.upper() not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            raise HTTPException(
                status_code=400,
                detail="严重级别必须是LOW、MEDIUM、HIGH或CRITICAL"
            )
        
        # 获取告警数据
        alerts_df = await db.get_alerts(
            start_time, end_time, severity, resolved, page, page_size
        )
        
        # 转换为告警列表
        alerts = []
        if not alerts_df.empty:
            alerts = [
                AlertInfo(
                    alert_id=row['alert_id'],
                    timestamp=row['timestamp'],
                    severity=row['severity'],
                    code=row['code'],
                    title=row['title'],
                    message=row['message'],
                    source=row['source'],
                    symbol=row['symbol'] if row['symbol'] else None,
                    resolved=bool(row['auto_resolved'])
                )
                for _, row in alerts_df.iterrows()
            ]
        
        # 计算汇总信息
        summary = await _calculate_alert_summary(
            db, start_time, end_time, severity, source, symbol, resolved
        )
        
        # 获取总数用于分页
        total_count = await _get_alerts_total_count(
            db, start_time, end_time, severity, source, symbol, resolved
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
        response = AlertResponse(
            timestamp=datetime.utcnow(),
            alerts=alerts,
            summary=summary,
            pagination=pagination,
            query_params={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "severity": severity,
                "source": source,
                "symbol": symbol,
                "resolved": resolved,
                "page": page,
                "page_size": page_size
            },
            execution_time_ms=timing["duration_ms"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询告警记录失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="查询告警记录失败"
        )


@router.get(
    "/alerts/active",
    summary="获取活跃告警",
    description="获取当前所有未解决的告警"
)
async def get_active_alerts(
    severity: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取活跃告警"""
    
    try:
        # 查询未解决的告警
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=7)  # 查询7天内的未解决告警
        
        alerts_df = await db.get_alerts(
            start_time, end_time, severity, resolved=False, page=1, page_size=1000
        )
        
        # 过滤数据源
        if source and not alerts_df.empty:
            alerts_df = alerts_df[alerts_df['source'] == source]
        
        # 转换为告警列表
        alerts = []
        if not alerts_df.empty:
            alerts = [
                {
                    "alert_id": row['alert_id'],
                    "timestamp": row['timestamp'],
                    "severity": row['severity'],
                    "code": row['code'],
                    "title": row['title'],
                    "message": row['message'],
                    "source": row['source'],
                    "symbol": row['symbol'] if row['symbol'] else None,
                    "age_hours": (datetime.utcnow() - row['timestamp']).total_seconds() / 3600
                }
                for _, row in alerts_df.iterrows()
            ]
        
        # 按严重级别分组
        by_severity = {}
        for alert in alerts:
            severity_level = alert['severity']
            if severity_level not in by_severity:
                by_severity[severity_level] = []
            by_severity[severity_level].append(alert)
        
        return {
            "active_alerts": alerts,
            "total_count": len(alerts),
            "by_severity": by_severity,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取活跃告警失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="获取活跃告警失败"
        )


@router.get(
    "/alerts/{alert_id}",
    summary="获取特定告警",
    description="获取指定ID的告警详情"
)
async def get_alert_by_id(
    alert_id: str,
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取特定告警"""
    
    try:
        query = """
        SELECT 
            alert_id,
            timestamp,
            severity,
            code,
            title,
            message,
            source,
            strategy_id,
            account_id,
            symbol,
            auto_resolved,
            resolution_action,
            context,
            metrics,
            metadata
        FROM alerts
        WHERE alert_id = %(alert_id)s
        """
        
        result_df = await db.query_to_dataframe(query, {'alert_id': alert_id})
        
        if result_df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"告警记录未找到: {alert_id}"
            )
        
        row = result_df.iloc[0]
        
        # 解析JSON字段
        import json
        context = {}
        metrics = {}
        metadata = {}
        
        try:
            context = json.loads(row['context']) if row['context'] else {}
            metrics = json.loads(row['metrics']) if row['metrics'] else {}
            metadata = json.loads(row['metadata']) if row['metadata'] else {}
        except json.JSONDecodeError:
            pass
        
        return {
            "alert_id": row['alert_id'],
            "timestamp": row['timestamp'],
            "severity": row['severity'],
            "code": row['code'],
            "title": row['title'],
            "message": row['message'],
            "source": row['source'],
            "strategy_id": row['strategy_id'],
            "account_id": row['account_id'],
            "symbol": row['symbol'] if row['symbol'] else None,
            "resolved": bool(row['auto_resolved']),
            "resolution_action": row['resolution_action'],
            "context": context,
            "metrics": metrics,
            "metadata": metadata
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"查询告警详情失败 [{alert_id}]: {e}")
        raise HTTPException(
            status_code=500,
            detail="查询告警详情失败"
        )


@router.get(
    "/alerts/summary",
    response_model=AlertSummary,
    summary="获取告警汇总",
    description="获取指定时间范围的告警汇总统计"
)
async def get_alerts_summary(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取告警汇总"""
    
    try:
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        summary = await _calculate_alert_summary(db, start_time, end_time)
        return summary
        
    except Exception as e:
        logger.error(f"查询告警汇总失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="查询告警汇总失败"
        )


@router.get(
    "/alerts/stats",
    summary="获取告警统计",
    description="获取告警统计信息和趋势"
)
async def get_alert_stats(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    interval: str = Query("1h", regex="^(1h|1d)$"),
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取告警统计"""
    
    try:
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=7)
        
        # 按时间间隔统计
        interval_clause = "1 HOUR" if interval == "1h" else "1 DAY"
        
        query = f"""
        SELECT 
            toStartOfInterval(timestamp, INTERVAL {interval_clause}) as time_bucket,
            severity,
            count(*) as alert_count
        FROM alerts
        WHERE timestamp >= %(start_time)s 
          AND timestamp <= %(end_time)s
        GROUP BY time_bucket, severity
        ORDER BY time_bucket, severity
        """
        
        result_df = await db.query_to_dataframe(query, {
            'start_time': start_time,
            'end_time': end_time
        })
        
        # 按来源统计
        source_query = """
        SELECT 
            source,
            severity,
            count(*) as alert_count
        FROM alerts
        WHERE timestamp >= %(start_time)s 
          AND timestamp <= %(end_time)s
        GROUP BY source, severity
        ORDER BY alert_count DESC
        """
        
        source_df = await db.query_to_dataframe(source_query, {
            'start_time': start_time,
            'end_time': end_time
        })
        
        # 处理时间序列数据
        time_series = []
        if not result_df.empty:
            # 重组数据
            grouped = result_df.groupby('time_bucket')
            for time_bucket, group in grouped:
                entry = {"timestamp": time_bucket}
                for _, row in group.iterrows():
                    entry[row['severity'].lower()] = int(row['alert_count'])
                time_series.append(entry)
        
        # 处理来源统计
        by_source = []
        if not source_df.empty:
            grouped = source_df.groupby('source')
            for source, group in grouped:
                entry = {"source": source}
                total_count = 0
                for _, row in group.iterrows():
                    entry[row['severity'].lower()] = int(row['alert_count'])
                    total_count += int(row['alert_count'])
                entry['total'] = total_count
                by_source.append(entry)
            
            # 按总数排序
            by_source.sort(key=lambda x: x['total'], reverse=True)
        
        return {
            "time_series": time_series,
            "by_source": by_source,
            "query_params": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "interval": interval
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取告警统计失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="获取告警统计失败"
        )


async def _get_alerts_total_count(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime,
    severity: Optional[str] = None,
    source: Optional[str] = None,
    symbol: Optional[str] = None,
    resolved: Optional[bool] = None
) -> int:
    """获取告警总数"""
    
    # 构建过滤条件
    filters = []
    params = {
        'start_time': start_time,
        'end_time': end_time
    }
    
    if severity:
        filters.append("AND severity = %(severity)s")
        params['severity'] = severity.upper()
    
    if source:
        filters.append("AND source = %(source)s")
        params['source'] = source
    
    if symbol:
        filters.append("AND symbol = %(symbol)s")
        params['symbol'] = symbol.upper()
    
    if resolved is not None:
        filters.append("AND auto_resolved = %(resolved)s")
        params['resolved'] = resolved
    
    filter_clause = " ".join(filters)
    
    query = f"""
    SELECT count(*) as total_count
    FROM alerts
    WHERE timestamp >= %(start_time)s 
      AND timestamp <= %(end_time)s
      {filter_clause}
    """
    
    result = await db.query_scalar(query, params)
    return int(result or 0)


async def _calculate_alert_summary(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime,
    severity: Optional[str] = None,
    source: Optional[str] = None,
    symbol: Optional[str] = None,
    resolved: Optional[bool] = None
) -> AlertSummary:
    """计算告警汇总"""
    
    # 构建过滤条件
    filters = []
    params = {
        'start_time': start_time,
        'end_time': end_time
    }
    
    if severity:
        filters.append("AND severity = %(severity)s")
        params['severity'] = severity.upper()
    
    if source:
        filters.append("AND source = %(source)s")
        params['source'] = source
    
    if symbol:
        filters.append("AND symbol = %(symbol)s")
        params['symbol'] = symbol.upper()
    
    if resolved is not None:
        filters.append("AND auto_resolved = %(resolved)s")
        params['resolved'] = resolved
    
    filter_clause = " ".join(filters)
    
    query = f"""
    SELECT 
        count(*) as total_alerts,
        sum(CASE WHEN auto_resolved = false THEN 1 ELSE 0 END) as active_alerts,
        sum(CASE WHEN severity = 'CRITICAL' THEN 1 ELSE 0 END) as critical_alerts,
        sum(CASE WHEN auto_resolved = true THEN 1 ELSE 0 END) as resolved_alerts
    FROM alerts
    WHERE timestamp >= %(start_time)s 
      AND timestamp <= %(end_time)s
      {filter_clause}
    """
    
    result_df = await db.query_to_dataframe(query, params)
    
    if result_df.empty:
        return AlertSummary(
            total_alerts=0,
            active_alerts=0,
            critical_alerts=0,
            resolved_alerts=0,
            alert_rate=0
        )
    
    row = result_df.iloc[0]
    
    # 计算告警率 (每小时)
    time_diff_hours = (end_time - start_time).total_seconds() / 3600
    alert_rate = (row['total_alerts'] / time_diff_hours) if time_diff_hours > 0 else 0
    
    return AlertSummary(
        total_alerts=int(row['total_alerts'] or 0),
        active_alerts=int(row['active_alerts'] or 0),
        critical_alerts=int(row['critical_alerts'] or 0),
        resolved_alerts=int(row['resolved_alerts'] or 0),
        alert_rate=alert_rate
    )