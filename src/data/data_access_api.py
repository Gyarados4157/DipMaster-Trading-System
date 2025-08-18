"""
数据访问API - Data Access API
为DipMaster Trading System提供统一的数据访问接口

Features:
- RESTful API接口
- WebSocket实时数据推送
- 数据查询和聚合
- 缓存和性能优化
- 访问控制和限流
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import json
import logging
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from pathlib import Path
import sqlite3
import aioredis
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
import msgpack
import gzip as gzip_lib
import lz4.frame
from contextlib import asynccontextmanager
import uvicorn
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import hashlib
import traceback

# 导入自定义模块
from .professional_data_infrastructure import ProfessionalDataInfrastructure
from .data_quality_monitor import DataQualityMonitor
from .realtime_data_stream import DataStreamManager

# Pydantic模型
class MarketDataRequest(BaseModel):
    symbol: str = Field(..., description="交易对符号，如BTCUSDT")
    timeframe: str = Field(..., description="时间框架，如1m, 5m, 15m, 1h, 4h, 1d")
    start_date: Optional[str] = Field(None, description="开始日期 YYYY-MM-DD")
    end_date: Optional[str] = Field(None, description="结束日期 YYYY-MM-DD")
    limit: Optional[int] = Field(1000, description="返回记录数限制")
    
class MultiSymbolRequest(BaseModel):
    symbols: List[str] = Field(..., description="交易对列表")
    timeframe: str = Field(..., description="时间框架")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: Optional[int] = 1000
    
class QualityReportRequest(BaseModel):
    symbol: Optional[str] = None
    days: int = Field(7, description="报告天数")
    
class MarketDataResponse(BaseModel):
    symbol: str
    timeframe: str
    data: List[Dict[str, Any]]
    count: int
    query_time_ms: float
    cache_hit: bool = False
    
class QualityReportResponse(BaseModel):
    report_period: Dict[str, str]
    summary: Dict[str, Any]
    metrics_by_symbol: Dict[str, Any]
    issues_summary: Dict[str, Any]
    recommendations: List[str]

# 限流器
limiter = Limiter(key_func=get_remote_address)

class DataAccessAPI:
    """数据访问API"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 核心组件
        self.data_infrastructure = None
        self.quality_monitor = None
        self.stream_manager = None
        
        # 缓存
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.cache_ttl = 300  # 5分钟TTL
        self.cache_stats = {'hits': 0, 'misses': 0}
        
        # Redis连接
        self.redis_client = None
        
        # WebSocket连接管理
        self.websocket_connections = set()
        
        # 性能统计
        self.request_stats = defaultdict(int)
        self.response_times = deque(maxlen=1000)
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def initialize(self, config: Dict[str, Any] = None):
        """初始化API组件"""
        config = config or {}
        
        self.logger.info("初始化数据访问API...")
        
        # 初始化数据基础设施
        self.data_infrastructure = ProfessionalDataInfrastructure(
            config_path=config.get('infrastructure_config')
        )
        
        # 初始化质量监控
        self.quality_monitor = DataQualityMonitor(
            config=config.get('quality_config', {})
        )
        
        # 初始化实时数据流
        self.stream_manager = DataStreamManager(
            config=config.get('stream_config', {})
        )
        
        # 初始化Redis
        if config.get('redis_enabled', True):
            try:
                redis_config = config.get('redis', {})
                self.redis_client = await aioredis.from_url(
                    redis_config.get('url', 'redis://localhost:6379'),
                    password=redis_config.get('password'),
                    db=redis_config.get('db', 1)  # 使用db=1避免与其他服务冲突
                )
                await self.redis_client.ping()
                self.logger.info("Redis连接成功")
            except Exception as e:
                self.logger.warning(f"Redis连接失败: {e}")
                
        self.logger.info("数据访问API初始化完成")
        
    def _generate_cache_key(self, **kwargs) -> str:
        """生成缓存键"""
        key_data = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
        
    async def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """获取缓存数据"""
        # 首先检查内存缓存
        with self.cache_lock:
            if cache_key in self.cache:
                data, timestamp = self.cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    self.cache_stats['hits'] += 1
                    return data
                else:
                    del self.cache[cache_key]
                    
        # 检查Redis缓存
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"api_cache:{cache_key}")
                if cached_data:
                    self.cache_stats['hits'] += 1
                    return json.loads(cached_data)
            except Exception as e:
                self.logger.error(f"Redis缓存读取失败: {e}")
                
        self.cache_stats['misses'] += 1
        return None
        
    async def _set_cached_data(self, cache_key: str, data: Any):
        """设置缓存数据"""
        # 设置内存缓存
        with self.cache_lock:
            self.cache[cache_key] = (data, time.time())
            
            # 限制内存缓存大小
            if len(self.cache) > 1000:
                # 删除最旧的缓存
                oldest_key = min(self.cache.keys(), 
                               key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]
                
        # 设置Redis缓存
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"api_cache:{cache_key}",
                    self.cache_ttl,
                    json.dumps(data, default=str)
                )
            except Exception as e:
                self.logger.error(f"Redis缓存写入失败: {e}")
                
    async def get_market_data(self, request: MarketDataRequest) -> MarketDataResponse:
        """获取市场数据"""
        start_time = time.time()
        
        try:
            # 生成缓存键
            cache_key = self._generate_cache_key(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date,
                limit=request.limit
            )
            
            # 检查缓存
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                cached_data['cache_hit'] = True
                cached_data['query_time_ms'] = (time.time() - start_time) * 1000
                return MarketDataResponse(**cached_data)
            
            # 从数据基础设施获取数据
            df = self.data_infrastructure.get_data(
                symbol=request.symbol,
                timeframe=request.timeframe,
                start_date=request.start_date,
                end_date=request.end_date
            )
            
            if df.empty:
                raise HTTPException(
                    status_code=404, 
                    detail=f"未找到数据: {request.symbol} {request.timeframe}"
                )
            
            # 应用限制
            if request.limit and len(df) > request.limit:
                df = df.tail(request.limit)
                
            # 转换为API响应格式
            data_records = []
            for idx, row in df.iterrows():
                record = {
                    'timestamp': idx.isoformat(),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume'])
                }
                data_records.append(record)
                
            response_data = {
                'symbol': request.symbol,
                'timeframe': request.timeframe,
                'data': data_records,
                'count': len(data_records),
                'query_time_ms': (time.time() - start_time) * 1000,
                'cache_hit': False
            }
            
            # 缓存结果
            await self._set_cached_data(cache_key, response_data)
            
            # 更新统计
            self.request_stats['market_data'] += 1
            self.response_times.append((time.time() - start_time) * 1000)
            
            return MarketDataResponse(**response_data)
            
        except Exception as e:
            self.logger.error(f"获取市场数据失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def get_multi_symbol_data(self, request: MultiSymbolRequest) -> Dict[str, MarketDataResponse]:
        """获取多币种数据"""
        start_time = time.time()
        
        try:
            results = {}
            
            # 并发获取多个币种数据
            tasks = []
            for symbol in request.symbols:
                market_request = MarketDataRequest(
                    symbol=symbol,
                    timeframe=request.timeframe,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    limit=request.limit
                )
                task = self.get_market_data(market_request)
                tasks.append((symbol, task))
                
            # 等待所有任务完成
            for symbol, task in tasks:
                try:
                    result = await task
                    results[symbol] = result
                except Exception as e:
                    self.logger.error(f"获取 {symbol} 数据失败: {e}")
                    results[symbol] = None
                    
            self.request_stats['multi_symbol'] += 1
            return results
            
        except Exception as e:
            self.logger.error(f"获取多币种数据失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def get_quality_report(self, request: QualityReportRequest) -> QualityReportResponse:
        """获取质量报告"""
        try:
            report = self.quality_monitor.get_quality_report(
                symbol=request.symbol,
                days=request.days
            )
            
            self.request_stats['quality_report'] += 1
            return QualityReportResponse(**report)
            
        except Exception as e:
            self.logger.error(f"获取质量报告失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def get_health_status(self) -> Dict[str, Any]:
        """获取系统健康状态"""
        try:
            # 数据基础设施状态
            infra_status = self.data_infrastructure.health_check()
            
            # 质量监控状态
            quality_status = self.quality_monitor.get_health_status()
            
            # 实时流状态
            stream_status = self.stream_manager.get_stream_status() if self.stream_manager else {}
            
            # API统计
            cache_hit_rate = (self.cache_stats['hits'] / 
                            (self.cache_stats['hits'] + self.cache_stats['misses'])) \
                            if (self.cache_stats['hits'] + self.cache_stats['misses']) > 0 else 0
                            
            avg_response_time = np.mean(list(self.response_times)) if self.response_times else 0
            
            status = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'overall_status': 'healthy',
                'infrastructure': infra_status,
                'data_quality': {
                    'overall_health': quality_status.overall_health,
                    'quality_score': quality_status.quality_score,
                    'active_symbols': quality_status.active_symbols,
                    'issues_count': quality_status.issues_count
                },
                'realtime_stream': stream_status,
                'api_performance': {
                    'cache_hit_rate': cache_hit_rate,
                    'avg_response_time_ms': avg_response_time,
                    'request_counts': dict(self.request_stats),
                    'active_websockets': len(self.websocket_connections)
                }
            }
            
            # 确定整体状态
            if (quality_status.overall_health == 'critical' or 
                infra_status.get('infrastructure_status') == 'error'):
                status['overall_status'] = 'critical'
            elif (quality_status.overall_health == 'warning' or 
                  infra_status.get('infrastructure_status') == 'degraded'):
                status['overall_status'] = 'warning'
                
            return status
            
        except Exception as e:
            self.logger.error(f"获取健康状态失败: {e}")
            raise HTTPException(status_code=500, detail=str(e))
            
    async def handle_websocket(self, websocket: WebSocket, symbol: str):
        """处理WebSocket连接"""
        await websocket.accept()
        self.websocket_connections.add(websocket)
        
        try:
            self.logger.info(f"WebSocket连接建立: {symbol}")
            
            # 发送初始数据
            try:
                initial_data = await self.get_market_data(
                    MarketDataRequest(symbol=symbol, timeframe='1m', limit=100)
                )
                await websocket.send_json({
                    'type': 'initial_data',
                    'data': initial_data.dict()
                })
            except Exception as e:
                self.logger.error(f"发送初始数据失败: {e}")
                
            # 保持连接并发送实时更新
            while True:
                try:
                    # 等待客户端消息或发送心跳
                    await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                except asyncio.TimeoutError:
                    # 发送心跳
                    await websocket.send_json({
                        'type': 'heartbeat',
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    })
                except WebSocketDisconnect:
                    break
                    
        except WebSocketDisconnect:
            self.logger.info(f"WebSocket连接断开: {symbol}")
        except Exception as e:
            self.logger.error(f"WebSocket错误: {e}")
        finally:
            self.websocket_connections.discard(websocket)
            
    async def broadcast_to_websockets(self, message: Dict[str, Any]):
        """向所有WebSocket连接广播消息"""
        if not self.websocket_connections:
            return
            
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)
                
        # 清理断开的连接
        self.websocket_connections -= disconnected
        
    def get_api_stats(self) -> Dict[str, Any]:
        """获取API统计信息"""
        return {
            'request_stats': dict(self.request_stats),
            'cache_stats': dict(self.cache_stats),
            'response_times': {
                'avg': np.mean(list(self.response_times)) if self.response_times else 0,
                'p95': np.percentile(list(self.response_times), 95) if self.response_times else 0,
                'p99': np.percentile(list(self.response_times), 99) if self.response_times else 0
            },
            'active_connections': len(self.websocket_connections),
            'cache_size': len(self.cache)
        }

# 创建FastAPI应用
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动
    config = {
        'redis_enabled': True,
        'redis': {
            'url': 'redis://localhost:6379',
            'db': 1
        }
    }
    
    app.state.api = DataAccessAPI()
    await app.state.api.initialize(config)
    
    yield
    
    # 关闭
    if hasattr(app.state, 'api') and app.state.api.redis_client:
        await app.state.api.redis_client.close()

app = FastAPI(
    title="DipMaster Data Access API",
    description="专业级量化交易数据访问接口",
    version="1.0.0",
    lifespan=lifespan
)

# 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SlowAPIMiddleware)

# 限流异常处理
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API依赖
def get_api() -> DataAccessAPI:
    return app.state.api

# API路由
@app.get("/", summary="API信息")
async def root():
    return {
        "name": "DipMaster Data Access API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/v1/market-data", response_model=MarketDataResponse, summary="获取市场数据")
@limiter.limit("100/minute")
async def get_market_data_endpoint(
    request: Request,
    data_request: MarketDataRequest, 
    api: DataAccessAPI = Depends(get_api)
):
    """获取单个币种的市场数据"""
    return await api.get_market_data(data_request)

@app.post("/api/v1/multi-symbol-data", summary="获取多币种数据")
@limiter.limit("50/minute")
async def get_multi_symbol_data_endpoint(
    request: Request,
    data_request: MultiSymbolRequest,
    api: DataAccessAPI = Depends(get_api)
):
    """获取多个币种的市场数据"""
    return await api.get_multi_symbol_data(data_request)

@app.post("/api/v1/quality-report", response_model=QualityReportResponse, summary="获取质量报告")
@limiter.limit("20/minute")
async def get_quality_report_endpoint(
    request: Request,
    report_request: QualityReportRequest,
    api: DataAccessAPI = Depends(get_api)
):
    """获取数据质量报告"""
    return await api.get_quality_report(report_request)

@app.get("/api/v1/health", summary="系统健康检查")
@limiter.limit("60/minute")
async def health_check_endpoint(
    request: Request,
    api: DataAccessAPI = Depends(get_api)
):
    """获取系统健康状态"""
    return await api.get_health_status()

@app.get("/api/v1/stats", summary="API统计信息")
@limiter.limit("30/minute")
async def get_stats_endpoint(
    request: Request,
    api: DataAccessAPI = Depends(get_api)
):
    """获取API统计信息"""
    return api.get_api_stats()

@app.websocket("/ws/market-data/{symbol}")
async def websocket_market_data(
    websocket: WebSocket,
    symbol: str,
    api: DataAccessAPI = Depends(get_api)
):
    """WebSocket市场数据推送"""
    await api.handle_websocket(websocket, symbol)

# 启动函数
def start_api(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """启动API服务"""
    uvicorn.run(
        "src.data.data_access_api:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )

if __name__ == "__main__":
    start_api(debug=True)