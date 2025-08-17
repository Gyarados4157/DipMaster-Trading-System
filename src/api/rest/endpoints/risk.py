"""
风险管理API端点
==============

提供风险指标查询和监控。
"""

import logging
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from ...schemas.api_responses import RiskResponse, RiskSummary, TimeSeriesPoint
from ...database import ClickHouseClient
from ..dependencies import get_db_client, get_timing_context

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/risk",
    response_model=RiskResponse,
    summary="获取风险指标",
    description="查询风险指标和限制状态"
)
async def get_risk_metrics(
    request: Request,
    start_time: Optional[datetime] = Query(
        None,
        description="开始时间 (默认为24小时前)"
    ),
    end_time: Optional[datetime] = Query(
        None,
        description="结束时间 (默认为当前时间)"
    ),
    include_historical: bool = Query(
        True,
        description="是否包含历史数据"
    ),
    db: ClickHouseClient = Depends(get_db_client),
    timing: dict = Depends(get_timing_context)
):
    """获取风险指标"""
    
    try:
        # 设置默认时间范围
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(hours=24)
        
        # 获取最新风险指标
        latest_metrics = await _get_latest_risk_metrics(db)
        
        # 计算风险汇总
        summary = await _calculate_risk_summary(latest_metrics)
        
        # 获取详细风险指标
        metrics = await _get_detailed_risk_metrics(latest_metrics)
        
        # 获取限制检查
        limit_checks = await _get_limit_checks(latest_metrics)
        
        # 获取历史数据
        historical_metrics = []
        if include_historical:
            historical_metrics = await _get_historical_risk_metrics(
                db, start_time, end_time
            )
        
        # 构建响应
        response = RiskResponse(
            timestamp=datetime.utcnow(),
            summary=summary,
            metrics=metrics,
            limit_checks=limit_checks,
            historical_metrics=historical_metrics,
            query_params={
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "include_historical": include_historical
            },
            execution_time_ms=timing["duration_ms"]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"查询风险指标失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="查询风险指标失败"
        )


@router.get(
    "/risk/current",
    summary="获取当前风险状态",
    description="获取最新的风险指标和状态"
)
async def get_current_risk_status(
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取当前风险状态"""
    
    try:
        # 获取最新风险指标
        latest_metrics = await _get_latest_risk_metrics(db)
        
        if not latest_metrics:
            return {
                "status": "unknown",
                "message": "暂无风险数据",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # 评估整体风险状态
        risk_score = latest_metrics.get('risk_score', 0)
        daily_pnl = latest_metrics.get('daily_pnl', 0)
        current_positions = latest_metrics.get('current_positions', 0)
        max_positions = latest_metrics.get('max_positions', 3)
        
        # 风险等级判断
        if risk_score >= 80 or daily_pnl < -500 or current_positions >= max_positions:
            status = "high"
            message = "高风险状态"
        elif risk_score >= 60 or daily_pnl < -200:
            status = "medium"  
            message = "中等风险状态"
        elif risk_score >= 40:
            status = "low"
            message = "低风险状态"
        else:
            status = "normal"
            message = "正常风险状态"
        
        return {
            "status": status,
            "risk_score": float(risk_score),
            "message": message,
            "metrics": {
                "daily_pnl": float(daily_pnl),
                "current_positions": int(current_positions),
                "max_positions": int(max_positions),
                "total_exposure": float(latest_metrics.get('total_exposure', 0)),
                "var_1d": float(latest_metrics.get('var_1d', 0)),
                "max_drawdown": float(latest_metrics.get('max_drawdown', 0))
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取当前风险状态失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="获取当前风险状态失败"
        )


@router.get(
    "/risk/limits",
    summary="获取风险限制",
    description="获取风险限制设置和当前使用情况"
)
async def get_risk_limits(
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取风险限制"""
    
    try:
        # 获取最新风险指标
        latest_metrics = await _get_latest_risk_metrics(db)
        
        if not latest_metrics:
            return {
                "limits": {},
                "utilization": {},
                "breaches": [],
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # 定义限制
        limits = {
            "max_positions": latest_metrics.get('max_positions', 3),
            "max_position_size": latest_metrics.get('max_position_size', 1000),
            "daily_loss_limit": abs(latest_metrics.get('daily_loss_limit', -500)),
            "max_drawdown_limit": 5.0,  # 5%
            "var_1d_limit": 100.0,  # $100
            "total_exposure_limit": 5000.0  # $5000
        }
        
        # 当前使用情况
        current_positions = latest_metrics.get('current_positions', 0)
        daily_pnl = latest_metrics.get('daily_pnl', 0)
        max_drawdown = latest_metrics.get('max_drawdown', 0)
        var_1d = latest_metrics.get('var_1d', 0)
        total_exposure = latest_metrics.get('total_exposure', 0)
        
        # 计算利用率
        utilization = {
            "positions": {
                "current": int(current_positions),
                "limit": int(limits["max_positions"]),
                "percentage": (current_positions / limits["max_positions"] * 100) if limits["max_positions"] > 0 else 0
            },
            "daily_loss": {
                "current": float(abs(min(0, daily_pnl))),
                "limit": float(limits["daily_loss_limit"]),
                "percentage": (abs(min(0, daily_pnl)) / limits["daily_loss_limit"] * 100) if limits["daily_loss_limit"] > 0 else 0
            },
            "drawdown": {
                "current": float(abs(max_drawdown)),
                "limit": float(limits["max_drawdown_limit"]),
                "percentage": (abs(max_drawdown) / limits["max_drawdown_limit"] * 100) if limits["max_drawdown_limit"] > 0 else 0
            },
            "var_1d": {
                "current": float(var_1d),
                "limit": float(limits["var_1d_limit"]),
                "percentage": (var_1d / limits["var_1d_limit"] * 100) if limits["var_1d_limit"] > 0 else 0
            },
            "exposure": {
                "current": float(total_exposure),
                "limit": float(limits["total_exposure_limit"]),
                "percentage": (total_exposure / limits["total_exposure_limit"] * 100) if limits["total_exposure_limit"] > 0 else 0
            }
        }
        
        # 检查违规
        breaches = []
        for limit_name, util in utilization.items():
            if util["percentage"] >= 100:
                breaches.append({
                    "limit_type": limit_name,
                    "current_value": util["current"],
                    "limit_value": util["limit"],
                    "breach_percentage": util["percentage"]
                })
        
        return {
            "limits": limits,
            "utilization": utilization,
            "breaches": breaches,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"获取风险限制失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="获取风险限制失败"
        )


@router.get(
    "/risk/var",
    summary="获取VaR分析",
    description="获取风险价值(VaR)分析数据"
)
async def get_var_analysis(
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    confidence_level: float = Query(0.95, ge=0.9, le=0.99),
    db: ClickHouseClient = Depends(get_db_client)
):
    """获取VaR分析"""
    
    try:
        if end_time is None:
            end_time = datetime.utcnow()
        if start_time is None:
            start_time = end_time - timedelta(days=30)
        
        # 获取历史VaR数据
        query = """
        SELECT 
            timestamp,
            var_1d,
            var_5d,
            daily_pnl,
            total_exposure
        FROM risk_metrics
        WHERE timestamp >= %(start_time)s 
          AND timestamp <= %(end_time)s
        ORDER BY timestamp
        """
        
        result_df = await db.query_to_dataframe(query, {
            'start_time': start_time,
            'end_time': end_time
        })
        
        if result_df.empty:
            return {
                "var_series": [],
                "breach_analysis": {},
                "current_var": {},
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # VaR时间序列
        var_series = [
            {
                "timestamp": row['timestamp'],
                "var_1d": float(row['var_1d']),
                "var_5d": float(row['var_5d']),
                "daily_pnl": float(row['daily_pnl']),
                "exposure": float(row['total_exposure'])
            }
            for _, row in result_df.iterrows()
        ]
        
        # VaR违规分析
        var_1d_breaches = result_df[result_df['daily_pnl'] < -result_df['var_1d']]
        breach_rate = len(var_1d_breaches) / len(result_df) * 100 if len(result_df) > 0 else 0
        
        breach_analysis = {
            "total_observations": len(result_df),
            "var_breaches": len(var_1d_breaches),
            "breach_rate_percent": float(breach_rate),
            "expected_breach_rate_percent": (1 - confidence_level) * 100,
            "model_accuracy": "good" if abs(breach_rate - (1 - confidence_level) * 100) < 2 else "needs_calibration"
        }
        
        # 当前VaR
        latest_row = result_df.iloc[-1]
        current_var = {
            "var_1d": float(latest_row['var_1d']),
            "var_5d": float(latest_row['var_5d']),
            "confidence_level": confidence_level,
            "last_update": latest_row['timestamp']
        }
        
        return {
            "var_series": var_series,
            "breach_analysis": breach_analysis,
            "current_var": current_var,
            "query_params": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "confidence_level": confidence_level
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"VaR分析失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="VaR分析失败"
        )


async def _get_latest_risk_metrics(db: ClickHouseClient) -> Dict[str, Any]:
    """获取最新风险指标"""
    
    query = """
    SELECT 
        timestamp,
        total_exposure,
        max_position_size,
        var_1d,
        var_5d,
        max_drawdown,
        sharpe_ratio,
        volatility,
        daily_pnl,
        daily_loss_limit,
        max_positions,
        current_positions,
        risk_score
    FROM risk_metrics
    WHERE timestamp = (SELECT max(timestamp) FROM risk_metrics)
    """
    
    result_df = await db.query_to_dataframe(query)
    
    if result_df.empty:
        return {}
    
    return result_df.iloc[0].to_dict()


async def _calculate_risk_summary(latest_metrics: Dict[str, Any]) -> RiskSummary:
    """计算风险汇总"""
    
    if not latest_metrics:
        return RiskSummary(
            overall_risk_score=Decimal('0'),
            risk_level="UNKNOWN",
            active_alerts=0,
            breached_limits=0,
            var_utilization=Decimal('0')
        )
    
    risk_score = latest_metrics.get('risk_score', 0)
    
    # 风险等级
    if risk_score >= 80:
        risk_level = "CRITICAL"
    elif risk_score >= 60:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    # VaR利用率 (简化计算)
    var_1d = latest_metrics.get('var_1d', 0)
    var_limit = 100.0  # $100 限制
    var_utilization = min(100, (var_1d / var_limit * 100)) if var_limit > 0 else 0
    
    return RiskSummary(
        overall_risk_score=Decimal(str(risk_score)),
        risk_level=risk_level,
        active_alerts=0,  # 需要从告警系统获取
        breached_limits=0,  # 需要计算
        var_utilization=Decimal(str(var_utilization))
    )


async def _get_detailed_risk_metrics(latest_metrics: Dict[str, Any]) -> list:
    """获取详细风险指标"""
    
    if not latest_metrics:
        return []
    
    from ...schemas.kafka_events import RiskMetric
    
    metrics = []
    
    # VaR指标
    var_1d = latest_metrics.get('var_1d', 0)
    metrics.append(RiskMetric(
        name="VaR_1D",
        value=Decimal(str(var_1d)),
        threshold=Decimal('100'),
        status="NORMAL" if var_1d < 100 else "BREACH"
    ))
    
    # 最大回撤
    max_drawdown = latest_metrics.get('max_drawdown', 0)
    metrics.append(RiskMetric(
        name="Max_Drawdown",
        value=Decimal(str(abs(max_drawdown))),
        threshold=Decimal('5'),
        status="NORMAL" if abs(max_drawdown) < 5 else "BREACH"
    ))
    
    # 夏普比率
    sharpe_ratio = latest_metrics.get('sharpe_ratio', 0)
    metrics.append(RiskMetric(
        name="Sharpe_Ratio",
        value=Decimal(str(sharpe_ratio)),
        threshold=Decimal('1'),
        status="NORMAL" if sharpe_ratio > 1 else "WARNING"
    ))
    
    return metrics


async def _get_limit_checks(latest_metrics: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """获取限制检查"""
    
    if not latest_metrics:
        return {}
    
    limit_checks = {}
    
    # 持仓限制
    current_positions = latest_metrics.get('current_positions', 0)
    max_positions = latest_metrics.get('max_positions', 3)
    limit_checks["position_limit"] = {
        "current": current_positions,
        "limit": max_positions,
        "utilization_percent": (current_positions / max_positions * 100) if max_positions > 0 else 0,
        "status": "OK" if current_positions < max_positions else "BREACH"
    }
    
    # 日损失限制
    daily_pnl = latest_metrics.get('daily_pnl', 0)
    daily_loss_limit = latest_metrics.get('daily_loss_limit', -500)
    limit_checks["daily_loss_limit"] = {
        "current": daily_pnl,
        "limit": daily_loss_limit,
        "utilization_percent": (abs(min(0, daily_pnl)) / abs(daily_loss_limit) * 100) if daily_loss_limit < 0 else 0,
        "status": "OK" if daily_pnl > daily_loss_limit else "BREACH"
    }
    
    return limit_checks


async def _get_historical_risk_metrics(
    db: ClickHouseClient,
    start_time: datetime,
    end_time: datetime
) -> list:
    """获取历史风险指标"""
    
    query = """
    SELECT 
        timestamp,
        risk_score
    FROM risk_metrics
    WHERE timestamp >= %(start_time)s 
      AND timestamp <= %(end_time)s
    ORDER BY timestamp
    """
    
    result_df = await db.query_to_dataframe(query, {
        'start_time': start_time,
        'end_time': end_time
    })
    
    if result_df.empty:
        return []
    
    return [
        TimeSeriesPoint(
            timestamp=row['timestamp'],
            value=Decimal(str(row['risk_score']))
        )
        for _, row in result_df.iterrows()
    ]