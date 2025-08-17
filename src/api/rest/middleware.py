"""
FastAPI中间件
============

提供请求处理、认证、限流等中间件。
"""

import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from collections import defaultdict, deque

from fastapi import Request, Response, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    """请求计时中间件"""
    
    async def dispatch(self, request: Request, call_next):
        # 记录开始时间和请求ID
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        request.state.start_time = start_time
        request.state.request_id = request_id
        
        # 处理请求
        response = await call_next(request)
        
        # 计算处理时间
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request_id
        
        # 记录日志
        logger.info(
            f"{request.method} {request.url.path} - "
            f"{response.status_code} - {process_time*1000:.2f}ms - {request_id}"
        )
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """错误处理中间件"""
    
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"未处理的异常: {e}", exc_info=True)
            
            return JSONResponse(
                status_code=500,
                content={
                    "success": False,
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "error_message": "服务器内部错误",
                    "timestamp": datetime.utcnow().isoformat(),
                    "request_id": getattr(request.state, "request_id", None)
                }
            )


class RateLimitMiddleware(BaseHTTPMiddleware):
    """限流中间件"""
    
    def __init__(self, app, calls: int = 100, period: int = 60):
        super().__init__(app)
        self.calls = calls
        self.period = period
        self.clients: Dict[str, deque] = defaultdict(deque)
    
    def _get_client_id(self, request: Request) -> str:
        """获取客户端标识"""
        # 优先使用API密钥
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return f"api_key:{auth_header[7:]}"
        
        # 使用IP地址
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return f"ip:{forwarded_for.split(',')[0].strip()}"
        
        client_ip = getattr(request.client, "host", "unknown")
        return f"ip:{client_ip}"
    
    def _is_rate_limited(self, client_id: str) -> bool:
        """检查是否超过限制"""
        now = time.time()
        cutoff = now - self.period
        
        # 获取客户端记录
        requests = self.clients[client_id]
        
        # 清除过期记录
        while requests and requests[0] < cutoff:
            requests.popleft()
        
        # 检查是否超限
        if len(requests) >= self.calls:
            return True
        
        # 记录当前请求
        requests.append(now)
        return False
    
    async def dispatch(self, request: Request, call_next):
        client_id = self._get_client_id(request)
        
        if self._is_rate_limited(client_id):
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "error_message": f"请求频率超限，最多{self.calls}次/{self.period}秒",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        response = await call_next(request)
        
        # 添加限流头
        remaining = max(0, self.calls - len(self.clients[client_id]))
        response.headers["X-RateLimit-Limit"] = str(self.calls)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.period)
        
        return response


class AuthMiddleware(BaseHTTPMiddleware):
    """认证中间件"""
    
    def __init__(self, app, api_key: str):
        super().__init__(app)
        self.api_key = api_key
        self.public_paths = {
            "/", 
            "/docs", 
            "/redoc", 
            "/openapi.json",
            "/health",
            "/health/live",
            "/health/ready"
        }
    
    async def dispatch(self, request: Request, call_next):
        # 检查是否为公共路径
        if request.url.path in self.public_paths:
            return await call_next(request)
        
        # 检查认证头
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error_code": "MISSING_AUTH_TOKEN",
                    "error_message": "缺少认证令牌",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        # 验证API密钥
        token = auth_header[7:]  # 移除 "Bearer " 前缀
        if token != self.api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "success": False,
                    "error_code": "INVALID_AUTH_TOKEN",
                    "error_message": "无效的认证令牌",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        # 设置用户信息
        request.state.authenticated = True
        request.state.api_key = token
        
        return await call_next(request)


class CacheMiddleware(BaseHTTPMiddleware):
    """缓存中间件"""
    
    def __init__(self, app, cache_ttl: int = 60):
        super().__init__(app)
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, tuple] = {}  # key -> (data, expires_at)
        self.cacheable_paths = {
            "/api/performance",
            "/api/risk",
            "/health"
        }
    
    def _get_cache_key(self, request: Request) -> str:
        """生成缓存键"""
        return f"{request.method}:{request.url.path}:{str(request.query_params)}"
    
    def _is_cacheable(self, request: Request) -> bool:
        """检查是否可缓存"""
        return (
            request.method == "GET" and
            request.url.path in self.cacheable_paths
        )
    
    def _get_cached_response(self, cache_key: str) -> Optional[Response]:
        """获取缓存响应"""
        if cache_key not in self.cache:
            return None
        
        data, expires_at = self.cache[cache_key]
        if datetime.utcnow() > expires_at:
            del self.cache[cache_key]
            return None
        
        response = JSONResponse(content=data)
        response.headers["X-Cache"] = "HIT"
        return response
    
    def _cache_response(self, cache_key: str, response_data: dict):
        """缓存响应"""
        expires_at = datetime.utcnow() + timedelta(seconds=self.cache_ttl)
        self.cache[cache_key] = (response_data, expires_at)
    
    async def dispatch(self, request: Request, call_next):
        if not self._is_cacheable(request):
            return await call_next(request)
        
        cache_key = self._get_cache_key(request)
        
        # 尝试从缓存获取
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            return cached_response
        
        # 处理请求
        response = await call_next(request)
        
        # 缓存成功响应
        if response.status_code == 200:
            try:
                # 注意：这里简化处理，实际实现可能需要更复杂的逻辑
                response.headers["X-Cache"] = "MISS"
            except Exception as e:
                logger.warning(f"缓存响应失败: {e}")
        
        return response


class CompressionMiddleware(BaseHTTPMiddleware):
    """压缩中间件"""
    
    def __init__(self, app, minimum_size: int = 1000):
        super().__init__(app)
        self.minimum_size = minimum_size
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # 检查是否支持gzip
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding.lower():
            return response
        
        # 添加压缩提示头
        if (
            hasattr(response, "body") and 
            len(getattr(response, "body", b"")) >= self.minimum_size
        ):
            response.headers["Content-Encoding"] = "gzip"
        
        return response