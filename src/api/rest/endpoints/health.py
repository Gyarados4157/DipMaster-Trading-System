"""
健康检查API端点
==============

提供系统健康状态监控。
"""

import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, Request

from ...database import ClickHouseClient
from ...kafka import KafkaConsumerManager
from ..dependencies import get_db_client, get_kafka_consumer, get_config

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/health",
    summary="完整健康检查",
    description="检查所有组件的健康状态"
)
async def health_check(
    request: Request,
    db: ClickHouseClient = Depends(get_db_client),
    kafka: KafkaConsumerManager = Depends(get_kafka_consumer),
    config = Depends(get_config)
):
    """完整健康检查"""
    
    try:
        checks = {}
        overall_status = "healthy"
        issues = []
        
        # 检查应用状态
        app_health = await _check_app_health(request)
        checks["application"] = app_health
        
        if app_health["status"] != "healthy":
            overall_status = "degraded"
            issues.extend(app_health.get("issues", []))
        
        # 检查数据库
        db_health = await db.health_check()
        checks["database"] = db_health
        
        if db_health["status"] != "healthy":
            if db_health["status"] == "unhealthy":
                overall_status = "unhealthy"
            elif overall_status != "unhealthy":
                overall_status = "degraded"
            issues.extend(db_health.get("issues", []))
        
        # 检查Kafka消费者
        kafka_health = await kafka.health_check()
        checks["kafka_consumer"] = kafka_health
        
        if kafka_health["status"] != "healthy":
            if kafka_health["status"] == "unhealthy":
                overall_status = "unhealthy"
            elif overall_status != "unhealthy":
                overall_status = "degraded"
            issues.extend(kafka_health.get("issues", []))
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "issues": issues,
            "checks": checks,
            "version": "1.0.0",
            "uptime_seconds": (datetime.utcnow() - request.app.state.start_time).total_seconds()
        }
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "issues": [f"健康检查异常: {str(e)}"],
            "checks": {},
            "version": "1.0.0"
        }


@router.get(
    "/health/live",
    summary="存活检查",
    description="简单的存活性检查"
)
async def liveness_check(request: Request):
    """存活检查"""
    
    return {
        "status": "alive",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "DipMaster Data API"
    }


@router.get(
    "/health/ready",
    summary="就绪检查",
    description="检查服务是否就绪处理请求"
)
async def readiness_check(
    request: Request,
    db: ClickHouseClient = Depends(get_db_client),
    kafka: KafkaConsumerManager = Depends(get_kafka_consumer)
):
    """就绪检查"""
    
    try:
        ready = True
        issues = []
        
        # 检查数据库连接
        if not db.is_connected:
            ready = False
            issues.append("数据库未连接")
        
        # 检查Kafka消费者
        if not kafka.running:
            ready = False
            issues.append("Kafka消费者未运行")
        
        status = "ready" if ready else "not_ready"
        
        return {
            "status": status,
            "ready": ready,
            "timestamp": datetime.utcnow().isoformat(),
            "issues": issues
        }
        
    except Exception as e:
        logger.error(f"就绪检查失败: {e}")
        return {
            "status": "not_ready",
            "ready": False,
            "timestamp": datetime.utcnow().isoformat(),
            "issues": [f"就绪检查异常: {str(e)}"]
        }


@router.get(
    "/health/stats",
    summary="系统统计",
    description="获取系统运行统计信息"
)
async def system_stats(
    request: Request,
    kafka: KafkaConsumerManager = Depends(get_kafka_consumer)
):
    """系统统计"""
    
    try:
        # 获取Kafka统计
        kafka_stats = await kafka.get_stats()
        
        # 获取应用统计
        app_stats = {
            "start_time": request.app.state.start_time.isoformat(),
            "uptime_seconds": (datetime.utcnow() - request.app.state.start_time).total_seconds(),
            "version": "1.0.0"
        }
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "application": app_stats,
            "kafka_consumer": kafka_stats
        }
        
    except Exception as e:
        logger.error(f"获取系统统计失败: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


@router.get(
    "/health/metrics",
    summary="性能指标",
    description="获取系统性能指标"
)
async def performance_metrics(
    request: Request,
    db: ClickHouseClient = Depends(get_db_client)
):
    """性能指标"""
    
    try:
        # 获取数据库性能指标
        db_metrics = await _get_database_metrics(db)
        
        # 获取应用性能指标
        app_metrics = await _get_application_metrics(request)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "database": db_metrics,
            "application": app_metrics
        }
        
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}")
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }


async def _check_app_health(request: Request) -> Dict[str, Any]:
    """检查应用健康状态"""
    
    try:
        uptime = (datetime.utcnow() - request.app.state.start_time).total_seconds()
        
        # 基本检查
        issues = []
        if uptime < 30:  # 启动时间不足30秒
            issues.append("应用启动时间不足")
        
        status = "healthy" if not issues else "degraded"
        
        return {
            "status": status,
            "uptime_seconds": uptime,
            "start_time": request.app.state.start_time.isoformat(),
            "issues": issues
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "issues": [f"应用健康检查失败: {str(e)}"]
        }


async def _get_database_metrics(db: ClickHouseClient) -> Dict[str, Any]:
    """获取数据库性能指标"""
    
    try:
        # 查询数据库指标
        query = """
        SELECT 
            count(*) as total_queries,
            avg(query_duration_ms) as avg_query_time,
            max(query_duration_ms) as max_query_time
        FROM system.query_log
        WHERE event_time >= now() - INTERVAL 1 HOUR
          AND type = 'QueryFinish'
        """
        
        result_df = await db.query_to_dataframe(query)
        
        if result_df.empty:
            return {
                "total_queries": 0,
                "avg_query_time_ms": 0,
                "max_query_time_ms": 0,
                "connection_status": "connected" if db.is_connected else "disconnected"
            }
        
        row = result_df.iloc[0]
        
        return {
            "total_queries": int(row['total_queries'] or 0),
            "avg_query_time_ms": float(row['avg_query_time'] or 0),
            "max_query_time_ms": float(row['max_query_time'] or 0),
            "connection_status": "connected" if db.is_connected else "disconnected"
        }
        
    except Exception as e:
        logger.warning(f"获取数据库指标失败: {e}")
        return {
            "error": str(e),
            "connection_status": "connected" if db.is_connected else "disconnected"
        }


async def _get_application_metrics(request: Request) -> Dict[str, Any]:
    """获取应用性能指标"""
    
    try:
        import psutil
        import os
        
        # 获取进程信息
        process = psutil.Process(os.getpid())
        
        # CPU和内存使用
        cpu_percent = process.cpu_percent()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # 文件描述符
        num_fds = process.num_fds() if hasattr(process, 'num_fds') else None
        
        # 线程数
        num_threads = process.num_threads()
        
        return {
            "cpu_percent": cpu_percent,
            "memory_mb": memory_mb,
            "num_threads": num_threads,
            "num_fds": num_fds,
            "uptime_seconds": (datetime.utcnow() - request.app.state.start_time).total_seconds()
        }
        
    except ImportError:
        return {
            "error": "psutil未安装，无法获取系统指标",
            "uptime_seconds": (datetime.utcnow() - request.app.state.start_time).total_seconds()
        }
    except Exception as e:
        return {
            "error": str(e),
            "uptime_seconds": (datetime.utcnow() - request.app.state.start_time).total_seconds()
        }