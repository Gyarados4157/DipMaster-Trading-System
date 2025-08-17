"""
WebSocket事件处理器
==================

处理不同类型的实时数据推送。
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from decimal import Decimal

from ..schemas.kafka_events import AlertV1, SystemHealthV1
from ..schemas.api_responses import WebSocketMessage
from .manager import WebSocketManager, SubscriptionType

logger = logging.getLogger(__name__)


class BaseWebSocketHandler:
    """WebSocket处理器基类"""
    
    def __init__(self, ws_manager: WebSocketManager):
        self.ws_manager = ws_manager
        self.running = False
        self.task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动处理器"""
        if self.running:
            return
        
        self.running = True
        self.task = asyncio.create_task(self._run())
        logger.info(f"{self.__class__.__name__} 已启动")
    
    async def stop(self):
        """停止处理器"""
        if not self.running:
            return
        
        self.running = False
        
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"{self.__class__.__name__} 已停止")
    
    async def _run(self):
        """运行循环 - 子类需要实现"""
        pass


class AlertsHandler(BaseWebSocketHandler):
    """告警WebSocket处理器"""
    
    def __init__(self, ws_manager: WebSocketManager):
        super().__init__(ws_manager)
        self.alert_queue = asyncio.Queue(maxsize=1000)
    
    async def handle_alert(self, alert: AlertV1):
        """处理告警事件"""
        try:
            await self.alert_queue.put(alert)
        except asyncio.QueueFull:
            logger.warning("告警队列已满，丢弃告警")
    
    async def _run(self):
        """告警处理循环"""
        logger.info("启动告警WebSocket处理器")
        
        while self.running:
            try:
                # 等待告警事件
                alert = await asyncio.wait_for(
                    self.alert_queue.get(), 
                    timeout=1.0
                )
                
                # 构建WebSocket消息
                message = {
                    "type": "alert",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "alert_id": alert.alert_id,
                        "severity": alert.severity.value,
                        "code": alert.code,
                        "title": alert.title,
                        "message": alert.message,
                        "source": alert.source.value,
                        "symbol": alert.symbol,
                        "timestamp": alert.timestamp.isoformat(),
                        "context": alert.context,
                        "auto_resolved": alert.auto_resolved
                    }
                }
                
                # 广播给所有订阅者
                await self.ws_manager.broadcast(SubscriptionType.ALERTS, message)
                
                # 对于关键告警，额外处理
                if alert.severity.value == "CRITICAL":
                    await self._handle_critical_alert(alert, message)
                
            except asyncio.TimeoutError:
                # 超时是正常的，继续循环
                continue
            except Exception as e:
                logger.error(f"处理告警失败: {e}")
    
    async def _handle_critical_alert(self, alert: AlertV1, message: Dict[str, Any]):
        """处理关键告警"""
        # 为关键告警添加特殊标记
        critical_message = {
            **message,
            "type": "critical_alert",
            "data": {
                **message["data"],
                "urgent": True,
                "requires_action": True
            }
        }
        
        # 再次广播关键告警
        await self.ws_manager.broadcast(SubscriptionType.ALERTS, critical_message)
        
        logger.critical(f"关键告警已推送: {alert.title}")


class PnLHandler(BaseWebSocketHandler):
    """PnL WebSocket处理器"""
    
    def __init__(self, ws_manager: WebSocketManager, db_client):
        super().__init__(ws_manager)
        self.db_client = db_client
        self.update_interval = 5  # 5秒更新一次
        self.last_pnl_data = {}
    
    async def _run(self):
        """PnL更新循环"""
        logger.info("启动PnL WebSocket处理器")
        
        while self.running:
            try:
                # 获取最新PnL数据
                pnl_data = await self._get_current_pnl()
                
                # 检查是否有变化
                if self._has_pnl_changed(pnl_data):
                    message = {
                        "type": "pnl_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": pnl_data
                    }
                    
                    # 广播PnL更新
                    await self.ws_manager.broadcast(SubscriptionType.PNL, message)
                    
                    # 更新缓存
                    self.last_pnl_data = pnl_data
                
                # 等待下次更新
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"PnL更新失败: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _get_current_pnl(self) -> Dict[str, Any]:
        """获取当前PnL数据"""
        try:
            # 查询最近的执行报告
            query = """
            SELECT 
                sum(total_realized_pnl) as total_realized,
                sum(total_unrealized_pnl) as total_unrealized,
                sum(total_commission) as total_commission,
                count(*) as trade_count,
                max(timestamp) as last_update
            FROM exec_reports
            WHERE timestamp >= now() - INTERVAL 1 HOUR
            """
            
            result_df = await self.db_client.query_to_dataframe(query)
            
            if result_df.empty:
                return {
                    "total_realized_pnl": 0,
                    "total_unrealized_pnl": 0,
                    "total_pnl": 0,
                    "total_commission": 0,
                    "trade_count": 0,
                    "last_update": None
                }
            
            row = result_df.iloc[0]
            realized = float(row['total_realized'] or 0)
            unrealized = float(row['total_unrealized'] or 0)
            
            return {
                "total_realized_pnl": realized,
                "total_unrealized_pnl": unrealized,
                "total_pnl": realized + unrealized,
                "total_commission": float(row['total_commission'] or 0),
                "trade_count": int(row['trade_count'] or 0),
                "last_update": row['last_update'].isoformat() if row['last_update'] else None,
                "daily_change": await self._get_daily_change()
            }
            
        except Exception as e:
            logger.error(f"获取PnL数据失败: {e}")
            return self.last_pnl_data
    
    async def _get_daily_change(self) -> float:
        """获取日内变化"""
        try:
            query = """
            SELECT sum(total_realized_pnl + total_unrealized_pnl) as daily_pnl
            FROM exec_reports
            WHERE timestamp >= toStartOfDay(now())
            """
            
            result = await self.db_client.query_scalar(query)
            return float(result or 0)
            
        except Exception:
            return 0
    
    def _has_pnl_changed(self, new_data: Dict[str, Any]) -> bool:
        """检查PnL是否有变化"""
        if not self.last_pnl_data:
            return True
        
        # 检查关键字段是否变化
        key_fields = ["total_pnl", "total_realized_pnl", "total_unrealized_pnl", "trade_count"]
        
        for field in key_fields:
            old_value = self.last_pnl_data.get(field, 0)
            new_value = new_data.get(field, 0)
            
            # 对于金额，检查是否有显著变化（超过0.01）
            if field.endswith("_pnl"):
                if abs(new_value - old_value) > 0.01:
                    return True
            else:
                if new_value != old_value:
                    return True
        
        return False


class PositionsHandler(BaseWebSocketHandler):
    """持仓WebSocket处理器"""
    
    def __init__(self, ws_manager: WebSocketManager, db_client):
        super().__init__(ws_manager)
        self.db_client = db_client
        self.update_interval = 10  # 10秒更新一次
        self.last_positions = {}
    
    async def _run(self):
        """持仓更新循环"""
        logger.info("启动持仓WebSocket处理器")
        
        while self.running:
            try:
                # 获取当前持仓
                positions_data = await self._get_current_positions()
                
                # 检查是否有变化
                if self._has_positions_changed(positions_data):
                    message = {
                        "type": "positions_update",
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": positions_data
                    }
                    
                    # 广播持仓更新
                    await self.ws_manager.broadcast(SubscriptionType.POSITIONS, message)
                    
                    # 更新缓存
                    self.last_positions = positions_data
                
                # 等待下次更新
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"持仓更新失败: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _get_current_positions(self) -> Dict[str, Any]:
        """获取当前持仓"""
        try:
            positions_df = await self.db_client.get_current_positions()
            
            if positions_df.empty:
                return {
                    "positions": [],
                    "total_count": 0,
                    "total_exposure": 0,
                    "total_unrealized_pnl": 0
                }
            
            # 转换持仓数据
            positions = []
            total_exposure = 0
            total_unrealized_pnl = 0
            
            for _, row in positions_df.iterrows():
                position = {
                    "symbol": row['symbol'],
                    "quantity": float(row['quantity']),
                    "avg_price": float(row['avg_price']),
                    "market_value": float(row['market_value']),
                    "unrealized_pnl": float(row['unrealized_pnl']),
                    "side": row['side'],
                    "last_update": row['timestamp'].isoformat()
                }
                
                positions.append(position)
                total_exposure += abs(position["market_value"])
                total_unrealized_pnl += position["unrealized_pnl"]
            
            return {
                "positions": positions,
                "total_count": len(positions),
                "total_exposure": total_exposure,
                "total_unrealized_pnl": total_unrealized_pnl,
                "update_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"获取持仓数据失败: {e}")
            return self.last_positions
    
    def _has_positions_changed(self, new_data: Dict[str, Any]) -> bool:
        """检查持仓是否有变化"""
        if not self.last_positions:
            return True
        
        # 检查持仓数量变化
        if new_data.get("total_count", 0) != self.last_positions.get("total_count", 0):
            return True
        
        # 检查总敞口变化
        old_exposure = self.last_positions.get("total_exposure", 0)
        new_exposure = new_data.get("total_exposure", 0)
        
        if abs(new_exposure - old_exposure) > 1.0:  # 敞口变化超过$1
            return True
        
        # 检查未实现盈亏变化
        old_pnl = self.last_positions.get("total_unrealized_pnl", 0)
        new_pnl = new_data.get("total_unrealized_pnl", 0)
        
        if abs(new_pnl - old_pnl) > 0.1:  # 盈亏变化超过$0.1
            return True
        
        return False


class HealthHandler(BaseWebSocketHandler):
    """系统健康WebSocket处理器"""
    
    def __init__(self, ws_manager: WebSocketManager):
        super().__init__(ws_manager)
        self.health_queue = asyncio.Queue(maxsize=100)
        self.update_interval = 30  # 30秒更新一次
    
    async def handle_health_update(self, health: SystemHealthV1):
        """处理健康状态更新"""
        try:
            await self.health_queue.put(health)
        except asyncio.QueueFull:
            logger.warning("健康状态队列已满，丢弃更新")
    
    async def _run(self):
        """健康状态处理循环"""
        logger.info("启动健康WebSocket处理器")
        
        while self.running:
            try:
                # 等待健康状态更新或定时发送心跳
                try:
                    health = await asyncio.wait_for(
                        self.health_queue.get(),
                        timeout=self.update_interval
                    )
                    
                    # 处理健康状态更新
                    await self._handle_health_event(health)
                    
                except asyncio.TimeoutError:
                    # 超时时发送心跳
                    await self._send_heartbeat()
                
            except Exception as e:
                logger.error(f"处理健康状态失败: {e}")
    
    async def _handle_health_event(self, health: SystemHealthV1):
        """处理健康状态事件"""
        message = {
            "type": "health_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "overall_status": health.overall_status,
                "strategy_id": health.strategy_id,
                "cpu_usage": float(health.total_cpu_usage),
                "memory_usage": float(health.total_memory_usage),
                "active_connections": health.active_connections,
                "total_positions": health.total_positions,
                "daily_trades": health.daily_trades,
                "daily_pnl": float(health.daily_pnl),
                "market_data_delay_ms": health.market_data_delay_ms,
                "execution_delay_ms": health.execution_delay_ms,
                "components": [
                    {
                        "name": comp.component.value,
                        "status": comp.status,
                        "cpu_usage": float(comp.cpu_usage),
                        "memory_usage": float(comp.memory_usage),
                        "error_rate": float(comp.error_rate),
                        "response_time_ms": float(comp.response_time_ms)
                    }
                    for comp in health.components
                ],
                "timestamp": health.timestamp.isoformat()
            }
        }
        
        # 广播健康状态更新
        await self.ws_manager.broadcast(SubscriptionType.HEALTH, message)
    
    async def _send_heartbeat(self):
        """发送心跳消息"""
        message = {
            "type": "heartbeat",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {
                "server_time": datetime.utcnow().isoformat(),
                "active_connections": len(self.ws_manager.connections),
                "uptime_seconds": (
                    datetime.utcnow() - self.ws_manager.stats['start_time']
                ).total_seconds()
            }
        }
        
        # 广播心跳消息
        await self.ws_manager.broadcast(SubscriptionType.HEALTH, message)


class WebSocketHandlerManager:
    """WebSocket处理器管理器"""
    
    def __init__(self, ws_manager: WebSocketManager, db_client=None):
        self.ws_manager = ws_manager
        self.db_client = db_client
        
        # 创建处理器
        self.alerts_handler = AlertsHandler(ws_manager)
        self.health_handler = HealthHandler(ws_manager)
        
        # 需要数据库的处理器
        if db_client:
            self.pnl_handler = PnLHandler(ws_manager, db_client)
            self.positions_handler = PositionsHandler(ws_manager, db_client)
        else:
            self.pnl_handler = None
            self.positions_handler = None
        
        self.handlers = [
            self.alerts_handler,
            self.health_handler,
            self.pnl_handler,
            self.positions_handler
        ]
        
        # 过滤None值
        self.handlers = [h for h in self.handlers if h is not None]
    
    async def start(self):
        """启动所有处理器"""
        for handler in self.handlers:
            await handler.start()
        
        logger.info("所有WebSocket处理器已启动")
    
    async def stop(self):
        """停止所有处理器"""
        for handler in self.handlers:
            await handler.stop()
        
        logger.info("所有WebSocket处理器已停止")
    
    async def handle_alert(self, alert: AlertV1):
        """处理告警事件"""
        await self.alerts_handler.handle_alert(alert)
    
    async def handle_health_update(self, health: SystemHealthV1):
        """处理健康状态更新"""
        await self.health_handler.handle_health_update(health)