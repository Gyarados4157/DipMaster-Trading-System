"""
Dashboard API REST端点实现
/api/pnl, /api/positions, /api/fills, /api/risk, /api/performance, /api/health
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.security import HTTPBearer
from slowapi import Limiter
from slowapi.util import get_remote_address
import structlog

from .schemas import (
    APIResponse, PaginationInfo, PnLDataPoint, PositionSnapshot, 
    FillRecord, RiskSnapshot, PerformanceSnapshot
)
from .auth import get_current_user, require_permission
from .database import DatabaseManager
from .cache import CacheManager

logger = structlog.get_logger(__name__)
security = HTTPBearer()
limiter = Limiter(key_func=get_remote_address)

def create_api_routes() -> APIRouter:
    """创建API路由"""
    router = APIRouter()
    
    # PnL端点
    @router.get("/pnl", response_model=APIResponse)
    @limiter.limit("30/minute")
    async def get_pnl_data(
        request: Request,
        account_id: str = Query(..., description="账户ID"),
        start_time: datetime = Query(..., description="开始时间"),
        end_time: datetime = Query(..., description="结束时间"),
        symbols: Optional[str] = Query(None, description="交易对列表，逗号分隔"),
        page: int = Query(1, ge=1, description="页码"),
        page_size: int = Query(50, ge=1, le=1000, description="每页大小"),
        current_user: dict = Depends(get_current_user),
        db: DatabaseManager = Depends(get_database),
        cache: CacheManager = Depends(get_cache)
    ):
        """获取PnL曲线数据"""
        try:
            # 权限检查
            await require_permission(current_user, "read")
            
            # 参数验证
            if (end_time - start_time).days > 365:
                raise HTTPException(status_code=400, detail="查询时间范围不能超过1年")
            
            # 解析交易对列表
            symbol_list = None
            if symbols:
                symbol_list = [s.strip().upper() for s in symbols.split(",")]
            
            # 尝试从缓存获取
            cache_key = f"pnl:{account_id}:{start_time.isoformat()}:{end_time.isoformat()}:{symbols or 'all'}"
            cached_data = await cache.get_cached_data(cache_key)
            
            if cached_data:
                logger.info(f"PnL数据缓存命中: {cache_key}")
                pnl_data = cached_data
            else:
                # 从数据库查询
                pnl_data = await db.get_pnl_data(
                    account_id=account_id,
                    start_time=start_time,
                    end_time=end_time,
                    symbols=symbol_list,
                    limit=page_size * 10  # 获取更多数据用于分页
                )
                
                # 缓存结果
                await cache.cache_data(cache_key, pnl_data, ttl=60)
            
            # 分页处理
            total = len(pnl_data)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            page_data = pnl_data[start_idx:end_idx]
            
            pagination = PaginationInfo(
                page=page,
                page_size=page_size,
                total=total,
                total_pages=(total + page_size - 1) // page_size
            )
            
            return APIResponse(
                success=True,
                data=page_data,
                pagination=pagination,
                message=f"获取PnL数据成功，共{total}条记录"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取PnL数据失败: {e}")
            raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")
    
    # 仓位端点
    @router.get("/positions", response_model=APIResponse)
    @limiter.limit("60/minute")
    async def get_positions(
        request: Request,
        account_id: str = Query(..., description="账户ID"),
        latest: bool = Query(True, description="是否获取最新仓位"),
        start_time: Optional[datetime] = Query(None, description="开始时间"),
        end_time: Optional[datetime] = Query(None, description="结束时间"),
        current_user: dict = Depends(get_current_user),
        db: DatabaseManager = Depends(get_database),
        cache: CacheManager = Depends(get_cache)
    ):
        """获取仓位数据"""
        try:
            await require_permission(current_user, "read")
            
            if latest:
                # 获取最新仓位
                cache_key = f"positions:latest:{account_id}"
                cached_data = await cache.get_cached_data(cache_key)
                
                if cached_data:
                    positions = cached_data
                else:
                    positions = await db.get_latest_positions(account_id)
                    await cache.cache_data(cache_key, positions, ttl=30)
                    
                return APIResponse(
                    success=True,
                    data=positions,
                    message=f"获取最新仓位成功，共{len(positions)}个仓位"
                )
            else:
                # 获取历史仓位（需要实现历史仓位查询）
                if not start_time or not end_time:
                    raise HTTPException(status_code=400, detail="查询历史仓位需要提供时间范围")
                
                # 这里可以扩展历史仓位查询逻辑
                return APIResponse(
                    success=True,
                    data=[],
                    message="历史仓位查询功能待实现"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取仓位数据失败: {e}")
            raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")
    
    # 成交记录端点
    @router.get("/fills", response_model=APIResponse)
    @limiter.limit("30/minute")
    async def get_fills(
        request: Request,
        account_id: str = Query(..., description="账户ID"),
        start_time: datetime = Query(..., description="开始时间"),
        end_time: datetime = Query(..., description="结束时间"),
        symbols: Optional[str] = Query(None, description="交易对列表，逗号分隔"),
        page: int = Query(1, ge=1, description="页码"),
        page_size: int = Query(50, ge=1, le=500, description="每页大小"),
        current_user: dict = Depends(get_current_user),
        db: DatabaseManager = Depends(get_database),
        cache: CacheManager = Depends(get_cache)
    ):
        """获取成交记录"""
        try:
            await require_permission(current_user, "read")
            
            # 参数验证
            if (end_time - start_time).days > 30:
                raise HTTPException(status_code=400, detail="成交记录查询时间范围不能超过30天")
            
            # 解析交易对列表
            symbol_list = None
            if symbols:
                symbol_list = [s.strip().upper() for s in symbols.split(",")]
            
            # 查询成交记录
            fills = await db.get_fills(
                account_id=account_id,
                start_time=start_time,
                end_time=end_time,
                symbols=symbol_list,
                limit=page_size
            )
            
            # 简单分页（实际应该在数据库层实现）
            total = len(fills)
            pagination = PaginationInfo(
                page=page,
                page_size=page_size,
                total=total,
                total_pages=(total + page_size - 1) // page_size
            )
            
            return APIResponse(
                success=True,
                data=fills,
                pagination=pagination,
                message=f"获取成交记录成功，共{total}条记录"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取成交记录失败: {e}")
            raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")
    
    # 风险指标端点
    @router.get("/risk", response_model=APIResponse)
    @limiter.limit("60/minute")
    async def get_risk_metrics(
        request: Request,
        account_id: str = Query(..., description="账户ID"),
        start_time: datetime = Query(..., description="开始时间"),
        end_time: datetime = Query(..., description="结束时间"),
        latest: bool = Query(False, description="是否只获取最新数据"),
        current_user: dict = Depends(get_current_user),
        db: DatabaseManager = Depends(get_database),
        cache: CacheManager = Depends(get_cache)
    ):
        """获取风险指标"""
        try:
            await require_permission(current_user, "read")
            
            if latest:
                # 获取最新风险指标
                cache_key = f"risk:latest:{account_id}"
                cached_data = await cache.get_cached_data(cache_key)
                
                if cached_data:
                    risk_data = cached_data
                else:
                    # 获取最近的风险数据
                    recent_time = datetime.utcnow() - timedelta(hours=1)
                    risk_metrics = await db.get_risk_metrics(
                        account_id=account_id,
                        start_time=recent_time,
                        end_time=datetime.utcnow()
                    )
                    risk_data = risk_metrics[0] if risk_metrics else None
                    await cache.cache_data(cache_key, risk_data, ttl=60)
                
                return APIResponse(
                    success=True,
                    data=risk_data,
                    message="获取最新风险指标成功"
                )
            else:
                # 获取历史风险指标
                if (end_time - start_time).days > 90:
                    raise HTTPException(status_code=400, detail="风险指标查询时间范围不能超过90天")
                
                risk_metrics = await db.get_risk_metrics(
                    account_id=account_id,
                    start_time=start_time,
                    end_time=end_time
                )
                
                return APIResponse(
                    success=True,
                    data=risk_metrics,
                    message=f"获取风险指标成功，共{len(risk_metrics)}条记录"
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取风险指标失败: {e}")
            raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")
    
    # 策略性能端点
    @router.get("/performance", response_model=APIResponse)
    @limiter.limit("30/minute")
    async def get_strategy_performance(
        request: Request,
        strategy_id: str = Query(..., description="策略ID"),
        start_time: datetime = Query(..., description="开始时间"),
        end_time: datetime = Query(..., description="结束时间"),
        current_user: dict = Depends(get_current_user),
        db: DatabaseManager = Depends(get_database),
        cache: CacheManager = Depends(get_cache)
    ):
        """获取策略性能数据"""
        try:
            await require_permission(current_user, "read")
            
            # 参数验证
            if (end_time - start_time).days > 365:
                raise HTTPException(status_code=400, detail="策略性能查询时间范围不能超过1年")
            
            # 尝试从缓存获取
            cache_key = f"performance:{strategy_id}:{start_time.isoformat()}:{end_time.isoformat()}"
            cached_data = await cache.get_cached_data(cache_key)
            
            if cached_data:
                performance_data = cached_data
            else:
                performance_data = await db.get_strategy_performance(
                    strategy_id=strategy_id,
                    start_time=start_time,
                    end_time=end_time
                )
                await cache.cache_data(cache_key, performance_data, ttl=300)  # 缓存5分钟
            
            # 计算聚合指标
            if performance_data:
                latest_performance = performance_data[0]
                summary = {
                    "latest_performance": latest_performance,
                    "data_points": len(performance_data),
                    "time_range": {
                        "start": start_time,
                        "end": end_time
                    }
                }
            else:
                summary = {
                    "latest_performance": None,
                    "data_points": 0,
                    "time_range": {
                        "start": start_time,
                        "end": end_time
                    }
                }
            
            return APIResponse(
                success=True,
                data={
                    "summary": summary,
                    "history": performance_data
                },
                message=f"获取策略性能成功，共{len(performance_data)}条记录"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取策略性能失败: {e}")
            raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")
    
    # 健康检查端点
    @router.get("/health", response_model=APIResponse)
    @limiter.limit("10/minute")
    async def health_check(
        request: Request,
        db: DatabaseManager = Depends(get_database),
        cache: CacheManager = Depends(get_cache)
    ):
        """系统健康检查"""
        try:
            health_status = {
                "service": "Dashboard API",
                "timestamp": datetime.utcnow(),
                "status": "healthy",
                "components": {}
            }
            
            # 检查数据库
            try:
                db_health = await db.get_health_status()
                health_status["components"]["database"] = db_health
            except Exception as e:
                health_status["components"]["database"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # 检查缓存
            try:
                cache_health = await cache.get_health_status()
                health_status["components"]["cache"] = cache_health
            except Exception as e:
                health_status["components"]["cache"] = {
                    "status": "unhealthy", 
                    "error": str(e)
                }
                health_status["status"] = "degraded"
            
            # 总体状态判断
            if any(comp.get("status") == "unhealthy" for comp in health_status["components"].values()):
                health_status["status"] = "unhealthy"
            
            return APIResponse(
                success=True,
                data=health_status,
                message="健康检查完成"
            )
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return APIResponse(
                success=False,
                data={
                    "service": "Dashboard API",
                    "timestamp": datetime.utcnow(),
                    "status": "unhealthy",
                    "error": str(e)
                },
                message="健康检查失败"
            )
    
    # 聚合数据端点
    @router.get("/summary", response_model=APIResponse)
    @limiter.limit("20/minute")
    async def get_summary(
        request: Request,
        account_id: str = Query(..., description="账户ID"),
        current_user: dict = Depends(get_current_user),
        db: DatabaseManager = Depends(get_database),
        cache: CacheManager = Depends(get_cache)
    ):
        """获取账户概览数据"""
        try:
            await require_permission(current_user, "read")
            
            # 尝试从缓存获取概览数据
            cache_key = f"summary:{account_id}"
            cached_data = await cache.get_cached_data(cache_key)
            
            if cached_data:
                return APIResponse(
                    success=True,
                    data=cached_data,
                    message="获取账户概览成功（缓存）"
                )
            
            # 并行获取各种数据
            now = datetime.utcnow()
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # 获取最新仓位
            positions = await db.get_latest_positions(account_id)
            
            # 获取今日PnL
            today_pnl = await db.get_pnl_data(
                account_id=account_id,
                start_time=start_of_day,
                end_time=now,
                limit=100
            )
            
            # 获取最新风险指标
            recent_time = now - timedelta(hours=1)
            risk_metrics = await db.get_risk_metrics(
                account_id=account_id,
                start_time=recent_time,
                end_time=now
            )
            
            # 构建概览数据
            summary_data = {
                "account_id": account_id,
                "timestamp": now,
                "positions": {
                    "count": len(positions),
                    "total_value": sum(pos.market_value for pos in positions),
                    "total_pnl": sum(pos.unrealized_pnl for pos in positions)
                },
                "daily_pnl": {
                    "realized": sum(pnl.realized_pnl for pnl in today_pnl),
                    "unrealized": sum(pnl.unrealized_pnl for pnl in today_pnl),
                    "total": sum(pnl.total_pnl for pnl in today_pnl)
                },
                "risk": risk_metrics[0] if risk_metrics else None
            }
            
            # 缓存概览数据
            await cache.cache_data(cache_key, summary_data, ttl=30)
            
            return APIResponse(
                success=True,
                data=summary_data,
                message="获取账户概览成功"
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取账户概览失败: {e}")
            raise HTTPException(status_code=500, detail=f"内部服务器错误: {str(e)}")
    
    return router

# 依赖注入函数
async def get_database() -> DatabaseManager:
    """获取数据库管理器实例"""
    from .main import dashboard_service
    if dashboard_service and dashboard_service.db_manager:
        return dashboard_service.db_manager
    raise HTTPException(status_code=503, detail="数据库服务不可用")

async def get_cache() -> CacheManager:
    """获取缓存管理器实例"""
    from .main import dashboard_service
    if dashboard_service and dashboard_service.cache_manager:
        return dashboard_service.cache_manager
    raise HTTPException(status_code=503, detail="缓存服务不可用")