"""
WebSocket连接管理器
==================

管理WebSocket连接和消息广播。
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import uuid
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ConnectionState(str, Enum):
    """连接状态"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"


class SubscriptionType(str, Enum):
    """订阅类型"""
    ALERTS = "alerts"
    PNL = "pnl"
    POSITIONS = "positions"
    HEALTH = "health"


class WebSocketMessage(BaseModel):
    """WebSocket消息格式"""
    type: str
    timestamp: datetime
    data: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ConnectionInfo(BaseModel):
    """连接信息"""
    connection_id: str
    client_ip: str
    connect_time: datetime
    last_ping: datetime
    subscriptions: Set[SubscriptionType]
    state: ConnectionState
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self, max_connections: int = 1000, heartbeat_interval: int = 30):
        self.max_connections = max_connections
        self.heartbeat_interval = heartbeat_interval
        
        # 连接管理
        self.connections: Dict[str, WebSocket] = {}
        self.connection_info: Dict[str, ConnectionInfo] = {}
        
        # 订阅管理 - 每种类型的订阅者列表
        self.subscriptions: Dict[SubscriptionType, Set[str]] = defaultdict(set)
        
        # 消息队列 - 每个连接的待发送消息
        self.message_queues: Dict[str, asyncio.Queue] = {}
        
        # 统计信息
        self.stats = {
            'total_connections': 0,
            'active_connections': 0,
            'messages_sent': 0,
            'messages_received': 0,
            'connection_errors': 0,
            'start_time': datetime.utcnow()
        }
        
        # 心跳任务
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.running = False
    
    async def start(self):
        """启动WebSocket管理器"""
        if self.running:
            return
        
        self.running = True
        
        # 启动心跳任务
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        
        logger.info("WebSocket管理器已启动")
    
    async def stop(self):
        """停止WebSocket管理器"""
        if not self.running:
            return
        
        self.running = False
        
        # 停止心跳任务
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # 关闭所有连接
        await self._close_all_connections()
        
        logger.info("WebSocket管理器已停止")
    
    async def connect(self, websocket: WebSocket, client_ip: str = "unknown") -> str:
        """建立WebSocket连接"""
        # 检查连接数限制
        if len(self.connections) >= self.max_connections:
            await websocket.close(code=1008, reason="Too many connections")
            raise RuntimeError("达到最大连接数限制")
        
        # 接受连接
        await websocket.accept()
        
        # 生成连接ID
        connection_id = str(uuid.uuid4())
        
        # 存储连接信息
        self.connections[connection_id] = websocket
        self.connection_info[connection_id] = ConnectionInfo(
            connection_id=connection_id,
            client_ip=client_ip,
            connect_time=datetime.utcnow(),
            last_ping=datetime.utcnow(),
            subscriptions=set(),
            state=ConnectionState.CONNECTED
        )
        
        # 创建消息队列
        self.message_queues[connection_id] = asyncio.Queue(maxsize=1000)
        
        # 更新统计
        self.stats['total_connections'] += 1
        self.stats['active_connections'] = len(self.connections)
        
        # 启动消息发送任务
        asyncio.create_task(self._message_sender(connection_id))
        
        # 发送欢迎消息
        await self.send_message(connection_id, {
            "type": "welcome",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "connection_id": connection_id,
                "server_time": datetime.utcnow().isoformat(),
                "heartbeat_interval": self.heartbeat_interval
            }
        })
        
        logger.info(f"WebSocket连接已建立: {connection_id} ({client_ip})")
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """断开WebSocket连接"""
        if connection_id not in self.connections:
            return
        
        # 移除所有订阅
        if connection_id in self.connection_info:
            for sub_type in self.connection_info[connection_id].subscriptions.copy():
                await self.unsubscribe(connection_id, sub_type)
        
        # 关闭连接
        websocket = self.connections.get(connection_id)
        if websocket:
            try:
                await websocket.close()
            except Exception as e:
                logger.warning(f"关闭WebSocket连接失败 [{connection_id}]: {e}")
        
        # 清理资源
        self.connections.pop(connection_id, None)
        self.connection_info.pop(connection_id, None)
        self.message_queues.pop(connection_id, None)
        
        # 更新统计
        self.stats['active_connections'] = len(self.connections)
        
        logger.info(f"WebSocket连接已断开: {connection_id}")
    
    async def subscribe(self, connection_id: str, subscription_type: SubscriptionType):
        """订阅特定类型的消息"""
        if connection_id not in self.connections:
            return False
        
        # 添加订阅
        self.subscriptions[subscription_type].add(connection_id)
        
        # 更新连接信息
        if connection_id in self.connection_info:
            self.connection_info[connection_id].subscriptions.add(subscription_type)
        
        # 发送订阅确认
        await self.send_message(connection_id, {
            "type": "subscription_confirmed",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "subscription_type": subscription_type.value,
                "status": "subscribed"
            }
        })
        
        logger.debug(f"连接 {connection_id} 订阅了 {subscription_type.value}")
        return True
    
    async def unsubscribe(self, connection_id: str, subscription_type: SubscriptionType):
        """取消订阅"""
        # 移除订阅
        self.subscriptions[subscription_type].discard(connection_id)
        
        # 更新连接信息
        if connection_id in self.connection_info:
            self.connection_info[connection_id].subscriptions.discard(subscription_type)
        
        # 发送取消订阅确认
        await self.send_message(connection_id, {
            "type": "subscription_cancelled",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "subscription_type": subscription_type.value,
                "status": "unsubscribed"
            }
        })
        
        logger.debug(f"连接 {connection_id} 取消订阅了 {subscription_type.value}")
    
    async def send_message(self, connection_id: str, message: Dict[str, Any]):
        """发送消息给特定连接"""
        if connection_id not in self.connections:
            return False
        
        try:
            # 添加到消息队列
            queue = self.message_queues.get(connection_id)
            if queue:
                await queue.put(message)
                return True
        except Exception as e:
            logger.error(f"发送消息失败 [{connection_id}]: {e}")
            await self.disconnect(connection_id)
        
        return False
    
    async def broadcast(self, subscription_type: SubscriptionType, message: Dict[str, Any]):
        """广播消息给订阅者"""
        subscribers = self.subscriptions[subscription_type].copy()
        
        if not subscribers:
            return 0
        
        success_count = 0
        failed_connections = []
        
        # 并发发送消息
        send_tasks = []
        for connection_id in subscribers:
            if connection_id in self.connections:
                task = asyncio.create_task(self.send_message(connection_id, message))
                send_tasks.append((connection_id, task))
        
        # 等待所有发送任务完成
        for connection_id, task in send_tasks:
            try:
                success = await task
                if success:
                    success_count += 1
                else:
                    failed_connections.append(connection_id)
            except Exception as e:
                logger.error(f"广播消息失败 [{connection_id}]: {e}")
                failed_connections.append(connection_id)
        
        # 清理失败的连接
        for connection_id in failed_connections:
            await self.disconnect(connection_id)
        
        self.stats['messages_sent'] += success_count
        
        logger.debug(
            f"广播消息 [{subscription_type.value}]: "
            f"成功={success_count}, 失败={len(failed_connections)}"
        )
        
        return success_count
    
    async def handle_message(self, connection_id: str, message: str):
        """处理接收到的消息"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "ping":
                await self._handle_ping(connection_id)
            elif msg_type == "subscribe":
                await self._handle_subscribe(connection_id, data)
            elif msg_type == "unsubscribe":
                await self._handle_unsubscribe(connection_id, data)
            else:
                logger.warning(f"未知消息类型 [{connection_id}]: {msg_type}")
            
            self.stats['messages_received'] += 1
            
        except json.JSONDecodeError:
            logger.error(f"无效的JSON消息 [{connection_id}]: {message}")
        except Exception as e:
            logger.error(f"处理消息失败 [{connection_id}]: {e}")
    
    async def _handle_ping(self, connection_id: str):
        """处理ping消息"""
        # 更新最后ping时间
        if connection_id in self.connection_info:
            self.connection_info[connection_id].last_ping = datetime.utcnow()
        
        # 发送pong响应
        await self.send_message(connection_id, {
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {}
        })
    
    async def _handle_subscribe(self, connection_id: str, data: Dict[str, Any]):
        """处理订阅请求"""
        subscription_type = data.get("subscription_type")
        
        try:
            sub_type = SubscriptionType(subscription_type)
            await self.subscribe(connection_id, sub_type)
        except ValueError:
            await self.send_message(connection_id, {
                "type": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "error": "invalid_subscription_type",
                    "message": f"无效的订阅类型: {subscription_type}"
                }
            })
    
    async def _handle_unsubscribe(self, connection_id: str, data: Dict[str, Any]):
        """处理取消订阅请求"""
        subscription_type = data.get("subscription_type")
        
        try:
            sub_type = SubscriptionType(subscription_type)
            await self.unsubscribe(connection_id, sub_type)
        except ValueError:
            await self.send_message(connection_id, {
                "type": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "error": "invalid_subscription_type",
                    "message": f"无效的订阅类型: {subscription_type}"
                }
            })
    
    async def _message_sender(self, connection_id: str):
        """消息发送任务"""
        websocket = self.connections.get(connection_id)
        queue = self.message_queues.get(connection_id)
        
        if not websocket or not queue:
            return
        
        try:
            while connection_id in self.connections:
                try:
                    # 等待消息，最多等待1秒
                    message = await asyncio.wait_for(queue.get(), timeout=1.0)
                    
                    # 发送消息
                    message_json = json.dumps(message, default=str)
                    await websocket.send_text(message_json)
                    
                except asyncio.TimeoutError:
                    # 超时是正常的，继续循环
                    continue
                except Exception as e:
                    logger.error(f"发送消息失败 [{connection_id}]: {e}")
                    break
        
        except Exception as e:
            logger.error(f"消息发送任务异常 [{connection_id}]: {e}")
        finally:
            await self.disconnect(connection_id)
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                
                if not self.running:
                    break
                
                # 检查超时连接
                now = datetime.utcnow()
                timeout_connections = []
                
                for connection_id, info in self.connection_info.items():
                    time_diff = (now - info.last_ping).total_seconds()
                    if time_diff > self.heartbeat_interval * 3:  # 3倍心跳间隔超时
                        timeout_connections.append(connection_id)
                
                # 断开超时连接
                for connection_id in timeout_connections:
                    logger.warning(f"连接超时，断开连接: {connection_id}")
                    await self.disconnect(connection_id)
                
                # 发送心跳消息
                await self.broadcast(SubscriptionType.HEALTH, {
                    "type": "heartbeat",
                    "timestamp": now.isoformat(),
                    "data": {
                        "active_connections": len(self.connections),
                        "server_time": now.isoformat()
                    }
                })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"心跳循环异常: {e}")
    
    async def _close_all_connections(self):
        """关闭所有连接"""
        connection_ids = list(self.connections.keys())
        
        for connection_id in connection_ids:
            await self.disconnect(connection_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = (datetime.utcnow() - self.stats['start_time']).total_seconds()
        
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'subscription_stats': {
                sub_type.value: len(subscribers)
                for sub_type, subscribers in self.subscriptions.items()
            }
        }
    
    def get_connections(self) -> List[Dict[str, Any]]:
        """获取连接列表"""
        return [
            {
                "connection_id": info.connection_id,
                "client_ip": info.client_ip,
                "connect_time": info.connect_time.isoformat(),
                "last_ping": info.last_ping.isoformat(),
                "subscriptions": [sub.value for sub in info.subscriptions],
                "state": info.state.value
            }
            for info in self.connection_info.values()
        ]