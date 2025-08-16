"""
WebSocket实时流管理器
支持 /ws/alerts, /ws/positions, /ws/pnl, /ws/risk 实时推送
"""

import asyncio
import json
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.websockets import WebSocketState
import structlog
from collections import defaultdict
import weakref

from .config import WebSocketConfig
from .schemas import EventBase
from .auth import AuthManager, verify_websocket_token

logger = structlog.get_logger(__name__)

@dataclass
class WebSocketConnection:
    """WebSocket连接信息"""
    websocket: WebSocket
    user_id: str
    channels: Set[str]
    filters: Dict[str, Any]
    connected_at: datetime
    last_heartbeat: datetime
    
    def __post_init__(self):
        # 使用弱引用避免循环引用
        self._websocket_ref = weakref.ref(self.websocket)

class MessageQueue:
    """消息队列管理器"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.queues: Dict[str, asyncio.Queue] = {}
        self.locks: Dict[str, asyncio.Lock] = {}
    
    async def get_queue(self, connection_id: str) -> asyncio.Queue:
        """获取连接的消息队列"""
        if connection_id not in self.queues:
            self.queues[connection_id] = asyncio.Queue(maxsize=self.max_size)
            self.locks[connection_id] = asyncio.Lock()
        return self.queues[connection_id]
    
    async def put_message(self, connection_id: str, message: Dict[str, Any]):
        """添加消息到队列"""
        queue = await self.get_queue(connection_id)
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            # 队列满时，移除最旧的消息
            try:
                queue.get_nowait()
                queue.put_nowait(message)
            except asyncio.QueueEmpty:
                pass
    
    async def get_message(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """从队列获取消息"""
        queue = await self.get_queue(connection_id)
        try:
            return queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
    
    def cleanup_connection(self, connection_id: str):
        """清理连接的队列"""
        if connection_id in self.queues:
            del self.queues[connection_id]
        if connection_id in self.locks:
            del self.locks[connection_id]

class ChannelManager:
    """频道管理器"""
    
    def __init__(self):
        self.channels: Dict[str, Set[str]] = defaultdict(set)  # channel -> connection_ids
        self.connection_channels: Dict[str, Set[str]] = defaultdict(set)  # connection_id -> channels
    
    def subscribe(self, connection_id: str, channel: str):
        """订阅频道"""
        self.channels[channel].add(connection_id)
        self.connection_channels[connection_id].add(channel)
        logger.debug(f"连接 {connection_id} 订阅频道 {channel}")
    
    def unsubscribe(self, connection_id: str, channel: str):
        """取消订阅频道"""
        self.channels[channel].discard(connection_id)
        self.connection_channels[connection_id].discard(channel)
        logger.debug(f"连接 {connection_id} 取消订阅频道 {channel}")
    
    def get_subscribers(self, channel: str) -> Set[str]:
        """获取频道订阅者"""
        return self.channels.get(channel, set()).copy()
    
    def get_channels(self, connection_id: str) -> Set[str]:
        """获取连接订阅的频道"""
        return self.connection_channels.get(connection_id, set()).copy()
    
    def remove_connection(self, connection_id: str):
        """移除连接的所有订阅"""
        channels = self.get_channels(connection_id)
        for channel in channels:
            self.unsubscribe(connection_id, channel)
        
        if connection_id in self.connection_channels:
            del self.connection_channels[connection_id]

class WebSocketManager:
    """WebSocket管理器主类"""
    
    def __init__(self, cache_manager, auth_manager: AuthManager):
        self.cache_manager = cache_manager
        self.auth_manager = auth_manager
        self.connections: Dict[str, WebSocketConnection] = {}
        self.message_queue = MessageQueue()
        self.channel_manager = ChannelManager()
        self.running = False
        self.heartbeat_task = None
        
        # 支持的频道
        self.supported_channels = {
            "alerts", "positions", "pnl", "risk", "fills", "system"
        }
    
    async def handle_connection(self, websocket: WebSocket, path: str):
        """处理WebSocket连接"""
        try:
            # 解析路径获取频道
            if not path.startswith("ws/"):
                path = path.lstrip("/")
            
            if not path.startswith("ws/"):
                await websocket.close(code=1002, reason="Invalid path")
                return
            
            channel = path.replace("ws/", "")
            if channel not in self.supported_channels:
                await websocket.close(code=1002, reason=f"Unsupported channel: {channel}")
                return
            
            await websocket.accept()
            logger.info(f"WebSocket连接已建立: {channel}")
            
            # 认证检查
            connection_id = await self._authenticate_connection(websocket)
            if not connection_id:
                await websocket.close(code=1008, reason="Authentication failed")
                return
            
            # 创建连接对象
            connection = WebSocketConnection(
                websocket=websocket,
                user_id=connection_id,
                channels=set(),
                filters={},
                connected_at=datetime.utcnow(),
                last_heartbeat=datetime.utcnow()
            )
            
            self.connections[connection_id] = connection
            
            # 订阅初始频道
            self.channel_manager.subscribe(connection_id, channel)
            connection.channels.add(channel)
            
            # 启动消息处理任务
            await asyncio.gather(
                self._handle_incoming_messages(connection_id, websocket),
                self._handle_outgoing_messages(connection_id, websocket),
                return_exceptions=True
            )
            
        except WebSocketDisconnect:
            logger.info(f"WebSocket连接断开: {path}")
        except Exception as e:
            logger.error(f"WebSocket处理异常: {e}")
        finally:
            await self._cleanup_connection(connection_id if 'connection_id' in locals() else None)
    
    async def _authenticate_connection(self, websocket: WebSocket) -> Optional[str]:
        """认证WebSocket连接"""
        try:
            # 等待认证消息
            auth_message = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            
            if auth_message.get("type") != "auth":
                return None
            
            token = auth_message.get("token")
            if not token:
                return None
            
            # 验证JWT令牌
            user_info = await verify_websocket_token(token, self.auth_manager)
            if not user_info:
                return None
            
            # 发送认证成功消息
            await websocket.send_json({
                "type": "auth_success",
                "user_id": user_info["user_id"],
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return user_info["user_id"]
            
        except Exception as e:
            logger.error(f"WebSocket认证失败: {e}")
            return None
    
    async def _handle_incoming_messages(self, connection_id: str, websocket: WebSocket):
        """处理客户端发送的消息"""
        try:
            async for message in websocket.iter_json():
                await self._process_client_message(connection_id, message)
                
                # 更新心跳时间
                if connection_id in self.connections:
                    self.connections[connection_id].last_heartbeat = datetime.utcnow()
                    
        except WebSocketDisconnect:
            logger.info(f"客户端断开连接: {connection_id}")
        except Exception as e:
            logger.error(f"处理客户端消息失败: {e}")
    
    async def _handle_outgoing_messages(self, connection_id: str, websocket: WebSocket):
        """处理发送给客户端的消息"""
        try:
            while websocket.client_state == WebSocketState.CONNECTED:
                message = await self.message_queue.get_message(connection_id)
                if message:
                    await websocket.send_json(message)
                else:
                    # 没有消息时短暂等待
                    await asyncio.sleep(0.1)
                    
        except WebSocketDisconnect:
            logger.info(f"发送消息中断，客户端断开: {connection_id}")
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
    
    async def _process_client_message(self, connection_id: str, message: Dict[str, Any]):
        """处理客户端消息"""
        try:
            msg_type = message.get("type")
            
            if msg_type == "ping":
                # 心跳响应
                await self.message_queue.put_message(connection_id, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            elif msg_type == "subscribe":
                # 订阅频道
                channel = message.get("channel")
                if channel in self.supported_channels:
                    self.channel_manager.subscribe(connection_id, channel)
                    if connection_id in self.connections:
                        self.connections[connection_id].channels.add(channel)
                    
                    await self.message_queue.put_message(connection_id, {
                        "type": "subscribed",
                        "channel": channel,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
            elif msg_type == "unsubscribe":
                # 取消订阅频道
                channel = message.get("channel")
                self.channel_manager.unsubscribe(connection_id, channel)
                if connection_id in self.connections:
                    self.connections[connection_id].channels.discard(channel)
                
                await self.message_queue.put_message(connection_id, {
                    "type": "unsubscribed",
                    "channel": channel,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            elif msg_type == "filter":
                # 设置过滤器
                if connection_id in self.connections:
                    filters = message.get("filters", {})
                    self.connections[connection_id].filters.update(filters)
                    
                    await self.message_queue.put_message(connection_id, {
                        "type": "filter_set",
                        "filters": filters,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
        except Exception as e:
            logger.error(f"处理客户端消息失败: {e}")
    
    async def broadcast_to_channel(self, channel: str, data: Any, 
                                 priority: str = "normal", filters: Dict[str, Any] = None):
        """向频道广播消息"""
        try:
            if channel not in self.supported_channels:
                logger.warning(f"不支持的频道: {channel}")
                return
            
            subscribers = self.channel_manager.get_subscribers(channel)
            if not subscribers:
                return
            
            # 构建消息
            message = {
                "type": "data",
                "channel": channel,
                "data": data if isinstance(data, (dict, list)) else [asdict(d) if hasattr(d, '__dict__') else d for d in data],
                "timestamp": datetime.utcnow().isoformat(),
                "priority": priority
            }
            
            # 并行发送给所有订阅者
            send_tasks = []
            for connection_id in subscribers:
                if self._should_send_to_connection(connection_id, message, filters):
                    task = self.message_queue.put_message(connection_id, message)
                    send_tasks.append(task)
            
            if send_tasks:
                await asyncio.gather(*send_tasks, return_exceptions=True)
                logger.debug(f"向 {len(send_tasks)} 个连接广播频道 {channel} 消息")
            
        except Exception as e:
            logger.error(f"频道广播失败: {e}")
    
    def _should_send_to_connection(self, connection_id: str, message: Dict[str, Any], 
                                 filters: Dict[str, Any] = None) -> bool:
        """检查是否应该向连接发送消息"""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        # 检查连接状态
        if connection.websocket.client_state != WebSocketState.CONNECTED:
            return False
        
        # 应用过滤器
        if filters:
            # 这里可以实现复杂的过滤逻辑
            for filter_key, filter_value in filters.items():
                if filter_key in connection.filters:
                    if connection.filters[filter_key] != filter_value:
                        return False
        
        return True
    
    async def _cleanup_connection(self, connection_id: str):
        """清理连接资源"""
        if not connection_id:
            return
        
        try:
            # 移除频道订阅
            self.channel_manager.remove_connection(connection_id)
            
            # 清理消息队列
            self.message_queue.cleanup_connection(connection_id)
            
            # 移除连接记录
            if connection_id in self.connections:
                del self.connections[connection_id]
            
            logger.info(f"连接 {connection_id} 资源清理完成")
            
        except Exception as e:
            logger.error(f"连接清理失败: {e}")
    
    async def start_heartbeat_monitor(self):
        """启动心跳监控"""
        self.running = True
        
        async def heartbeat_monitor():
            while self.running:
                try:
                    current_time = datetime.utcnow()
                    timeout_connections = []
                    
                    for connection_id, connection in self.connections.items():
                        # 检查超时连接
                        if (current_time - connection.last_heartbeat).seconds > 60:  # 60秒超时
                            timeout_connections.append(connection_id)
                    
                    # 清理超时连接
                    for connection_id in timeout_connections:
                        await self._cleanup_connection(connection_id)
                        logger.info(f"清理超时连接: {connection_id}")
                    
                    await asyncio.sleep(30)  # 每30秒检查一次
                    
                except Exception as e:
                    logger.error(f"心跳监控异常: {e}")
                    await asyncio.sleep(30)
        
        self.heartbeat_task = asyncio.create_task(heartbeat_monitor())
    
    async def send_system_message(self, message: str, level: str = "info"):
        """发送系统消息到所有连接"""
        system_message = {
            "type": "system",
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.broadcast_to_channel("system", system_message, priority="high")
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        stats = {
            "total_connections": len(self.connections),
            "channels": {},
            "connection_details": []
        }
        
        # 频道统计
        for channel in self.supported_channels:
            subscribers = self.channel_manager.get_subscribers(channel)
            stats["channels"][channel] = len(subscribers)
        
        # 连接详情
        for connection_id, connection in self.connections.items():
            stats["connection_details"].append({
                "connection_id": connection_id,
                "user_id": connection.user_id,
                "channels": list(connection.channels),
                "connected_at": connection.connected_at.isoformat(),
                "last_heartbeat": connection.last_heartbeat.isoformat()
            })
        
        return stats
    
    async def shutdown(self):
        """关闭WebSocket管理器"""
        self.running = False
        
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
        
        # 关闭所有连接
        close_tasks = []
        for connection_id, connection in self.connections.items():
            if connection.websocket.client_state == WebSocketState.CONNECTED:
                task = connection.websocket.close(code=1001, reason="Server shutdown")
                close_tasks.append(task)
        
        if close_tasks:
            await asyncio.gather(*close_tasks, return_exceptions=True)
        
        # 清理资源
        self.connections.clear()
        self.channel_manager = ChannelManager()
        self.message_queue = MessageQueue()
        
        logger.info("WebSocket管理器关闭完成")