"""
WebSocket端点
===========

提供WebSocket连接和实时数据推送。
"""

import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, HTTPException
from starlette.websockets import WebSocketState

from ...websocket import WebSocketManager
from ..dependencies import get_config

logger = logging.getLogger(__name__)

router = APIRouter()


async def get_websocket_manager() -> WebSocketManager:
    """获取WebSocket管理器 - 这将在主应用中设置"""
    # 这个依赖项将在主应用启动时配置
    pass


@router.websocket("/ws/alerts")
async def websocket_alerts(
    websocket: WebSocket,
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """告警WebSocket端点"""
    connection_id = None
    
    try:
        # 获取客户端IP
        client_ip = websocket.client.host if websocket.client else "unknown"
        
        # 建立连接
        connection_id = await ws_manager.connect(websocket, client_ip)
        
        # 自动订阅告警
        from ...websocket.manager import SubscriptionType
        await ws_manager.subscribe(connection_id, SubscriptionType.ALERTS)
        
        # 保持连接并处理消息
        while True:
            try:
                message = await websocket.receive_text()
                await ws_manager.handle_message(connection_id, message)
            except WebSocketDisconnect:
                break
            
    except Exception as e:
        logger.error(f"WebSocket告警端点异常: {e}")
    finally:
        if connection_id:
            await ws_manager.disconnect(connection_id)


@router.websocket("/ws/pnl")
async def websocket_pnl(
    websocket: WebSocket,
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """PnL WebSocket端点"""
    connection_id = None
    
    try:
        client_ip = websocket.client.host if websocket.client else "unknown"
        connection_id = await ws_manager.connect(websocket, client_ip)
        
        # 自动订阅PnL更新
        from ...websocket.manager import SubscriptionType
        await ws_manager.subscribe(connection_id, SubscriptionType.PNL)
        
        while True:
            try:
                message = await websocket.receive_text()
                await ws_manager.handle_message(connection_id, message)
            except WebSocketDisconnect:
                break
            
    except Exception as e:
        logger.error(f"WebSocket PnL端点异常: {e}")
    finally:
        if connection_id:
            await ws_manager.disconnect(connection_id)


@router.websocket("/ws/positions")
async def websocket_positions(
    websocket: WebSocket,
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """持仓WebSocket端点"""
    connection_id = None
    
    try:
        client_ip = websocket.client.host if websocket.client else "unknown"
        connection_id = await ws_manager.connect(websocket, client_ip)
        
        # 自动订阅持仓更新
        from ...websocket.manager import SubscriptionType
        await ws_manager.subscribe(connection_id, SubscriptionType.POSITIONS)
        
        while True:
            try:
                message = await websocket.receive_text()
                await ws_manager.handle_message(connection_id, message)
            except WebSocketDisconnect:
                break
            
    except Exception as e:
        logger.error(f"WebSocket持仓端点异常: {e}")
    finally:
        if connection_id:
            await ws_manager.disconnect(connection_id)


@router.websocket("/ws/health")
async def websocket_health(
    websocket: WebSocket,
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """系统健康WebSocket端点"""
    connection_id = None
    
    try:
        client_ip = websocket.client.host if websocket.client else "unknown"
        connection_id = await ws_manager.connect(websocket, client_ip)
        
        # 自动订阅健康状态更新
        from ...websocket.manager import SubscriptionType
        await ws_manager.subscribe(connection_id, SubscriptionType.HEALTH)
        
        while True:
            try:
                message = await websocket.receive_text()
                await ws_manager.handle_message(connection_id, message)
            except WebSocketDisconnect:
                break
            
    except Exception as e:
        logger.error(f"WebSocket健康端点异常: {e}")
    finally:
        if connection_id:
            await ws_manager.disconnect(connection_id)


@router.websocket("/ws")
async def websocket_general(
    websocket: WebSocket,
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """通用WebSocket端点 - 支持多种订阅"""
    connection_id = None
    
    try:
        client_ip = websocket.client.host if websocket.client else "unknown"
        connection_id = await ws_manager.connect(websocket, client_ip)
        
        # 不自动订阅，等待客户端发送订阅请求
        
        while True:
            try:
                message = await websocket.receive_text()
                await ws_manager.handle_message(connection_id, message)
            except WebSocketDisconnect:
                break
            
    except Exception as e:
        logger.error(f"WebSocket通用端点异常: {e}")
    finally:
        if connection_id:
            await ws_manager.disconnect(connection_id)


# WebSocket管理API端点
@router.get(
    "/ws/stats",
    summary="获取WebSocket统计",
    description="获取WebSocket连接和消息统计信息"
)
async def get_websocket_stats(
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """获取WebSocket统计"""
    try:
        return ws_manager.get_stats()
    except Exception as e:
        logger.error(f"获取WebSocket统计失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="获取WebSocket统计失败"
        )


@router.get(
    "/ws/connections",
    summary="获取WebSocket连接列表",
    description="获取当前活跃的WebSocket连接信息"
)
async def get_websocket_connections(
    ws_manager: WebSocketManager = Depends(get_websocket_manager)
):
    """获取WebSocket连接列表"""
    try:
        return {
            "connections": ws_manager.get_connections(),
            "total_count": len(ws_manager.connections),
            "timestamp": "datetime.utcnow().isoformat()"
        }
    except Exception as e:
        logger.error(f"获取WebSocket连接失败: {e}")
        raise HTTPException(
            status_code=500,
            detail="获取WebSocket连接失败"
        )