"""
FastAPI依赖项
============

提供公共依赖项注入。
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..database import ClickHouseClient
from ..kafka import KafkaConsumerManager

security = HTTPBearer()


async def get_db_client(request: Request) -> ClickHouseClient:
    """获取数据库客户端"""
    return request.app.state.db_client


async def get_kafka_consumer(request: Request) -> KafkaConsumerManager:
    """获取Kafka消费者"""
    return request.app.state.kafka_consumer


async def get_config(request: Request):
    """获取配置"""
    return request.app.state.config


async def get_timing_context(request: Request) -> Dict[str, Any]:
    """获取请求计时上下文"""
    start_time = getattr(request.state, "start_time", time.time())
    duration_ms = int((time.time() - start_time) * 1000)
    
    return {
        "start_time": start_time,
        "duration_ms": duration_ms,
        "request_id": getattr(request.state, "request_id", str(uuid.uuid4()))
    }


async def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """验证API密钥"""
    config = await get_config(request)
    
    if not config.enable_auth:
        return "anonymous"
    
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="需要API密钥认证"
        )
    
    if credentials.credentials != config.api_key:
        raise HTTPException(
            status_code=401,
            detail="无效的API密钥"
        )
    
    return credentials.credentials


def create_pagination_params(
    page: int = 1,
    page_size: int = 100,
    max_page_size: int = 1000
):
    """创建分页参数"""
    if page < 1:
        raise HTTPException(
            status_code=400,
            detail="页码必须大于0"
        )
    
    if page_size < 1 or page_size > max_page_size:
        raise HTTPException(
            status_code=400,
            detail=f"每页大小必须在1-{max_page_size}之间"
        )
    
    return {
        "page": page,
        "page_size": page_size,
        "offset": (page - 1) * page_size
    }