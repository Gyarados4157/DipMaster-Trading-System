"""
Kafka事件处理器
==============

处理不同类型的Kafka事件并存储到ClickHouse。
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional
import json
from decimal import Decimal

from ..schemas.kafka_events import (
    ExecutionReportV1, RiskMetricsV1, AlertV1, SystemHealthV1
)
from ..database import ClickHouseClient
from ..database.models import (
    ExecReportModel, FillModel, OrderModel, PositionModel,
    RiskMetricModel, AlertModel, HealthModel
)

logger = logging.getLogger(__name__)


class BaseEventHandler(ABC):
    """事件处理器基类"""
    
    def __init__(self, db_client: ClickHouseClient):
        self.db_client = db_client
        self.batch_buffer: List[Any] = []
        self.batch_size = 100
        self.batch_timeout = 5  # 秒
        self.last_flush = datetime.utcnow()
        
    @abstractmethod
    async def handle(self, event_data: Dict[str, Any]):
        """处理事件"""
        pass
    
    @abstractmethod
    async def flush_batch(self):
        """批量写入数据"""
        pass
    
    async def add_to_batch(self, item: Any):
        """添加到批处理缓冲区"""
        self.batch_buffer.append(item)
        
        # 检查是否需要刷新
        should_flush = (
            len(self.batch_buffer) >= self.batch_size or
            (datetime.utcnow() - self.last_flush).total_seconds() >= self.batch_timeout
        )
        
        if should_flush:
            await self.flush_batch()
    
    async def force_flush(self):
        """强制刷新缓冲区"""
        if self.batch_buffer:
            await self.flush_batch()


class ExecReportHandler(BaseEventHandler):
    """执行报告事件处理器"""
    
    def __init__(self, db_client: ClickHouseClient):
        super().__init__(db_client)
        self.batch_size = 50
        
    async def handle(self, event_data: Dict[str, Any]):
        """处理执行报告事件"""
        try:
            # 解析事件
            event = ExecutionReportV1(**event_data)
            
            # 转换为数据库模型
            exec_report = ExecReportModel.from_kafka_event(event)
            
            # 添加到批处理
            await self.add_to_batch(exec_report)
            
            # 处理关联的订单和成交数据
            await self._process_related_data(event)
            
            logger.debug(f"处理执行报告: {event.event_id}")
            
        except Exception as e:
            logger.error(f"处理执行报告失败: {e}")
            raise
    
    async def _process_related_data(self, event: ExecutionReportV1):
        """处理关联的订单和成交数据"""
        # 处理成交数据
        if event.fills:
            fill_models = [
                FillModel.from_fill(fill, event.account_id) 
                for fill in event.fills
            ]
            
            # 批量插入成交数据
            if fill_models:
                fill_data = [model.dict() for model in fill_models]
                await self.db_client.insert_batch('fills', fill_data, batch_size=100)
        
        # 处理订单数据
        if event.orders:
            order_models = [
                OrderModel.from_order(order, event.account_id)
                for order in event.orders
            ]
            
            # 批量插入订单数据
            if order_models:
                order_data = [model.dict() for model in order_models]
                await self.db_client.insert_batch('orders', order_data, batch_size=100)
    
    async def flush_batch(self):
        """批量写入执行报告"""
        if not self.batch_buffer:
            return
        
        try:
            # 转换为字典列表
            data = [item.dict() for item in self.batch_buffer]
            
            # 批量插入
            await self.db_client.insert_batch('exec_reports', data)
            
            logger.info(f"批量写入执行报告: {len(data)} 条")
            
            # 清空缓冲区
            self.batch_buffer.clear()
            self.last_flush = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"批量写入执行报告失败: {e}")
            raise


class RiskMetricsHandler(BaseEventHandler):
    """风险指标事件处理器"""
    
    def __init__(self, db_client: ClickHouseClient):
        super().__init__(db_client)
        self.batch_size = 20
        
    async def handle(self, event_data: Dict[str, Any]):
        """处理风险指标事件"""
        try:
            # 解析事件
            event = RiskMetricsV1(**event_data)
            
            # 转换为数据库模型
            risk_metrics = RiskMetricModel.from_kafka_event(event)
            
            # 添加到批处理
            await self.add_to_batch(risk_metrics)
            
            # 处理持仓快照
            await self._process_positions(event)
            
            logger.debug(f"处理风险指标: {event.event_id}")
            
        except Exception as e:
            logger.error(f"处理风险指标失败: {e}")
            raise
    
    async def _process_positions(self, event: RiskMetricsV1):
        """处理持仓快照"""
        if not event.positions:
            return
        
        try:
            # 转换持仓数据
            position_models = [
                PositionModel.from_position(pos, event.timestamp, event.account_id)
                for pos in event.positions
                if pos.quantity != Decimal('0')  # 只保存非零持仓
            ]
            
            if position_models:
                position_data = [model.dict() for model in position_models]
                await self.db_client.insert_batch('positions', position_data)
                
                logger.debug(f"更新持仓快照: {len(position_data)} 个持仓")
        
        except Exception as e:
            logger.error(f"处理持仓快照失败: {e}")
    
    async def flush_batch(self):
        """批量写入风险指标"""
        if not self.batch_buffer:
            return
        
        try:
            # 转换为字典列表
            data = [item.dict() for item in self.batch_buffer]
            
            # 批量插入
            await self.db_client.insert_batch('risk_metrics', data)
            
            logger.info(f"批量写入风险指标: {len(data)} 条")
            
            # 清空缓冲区
            self.batch_buffer.clear()
            self.last_flush = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"批量写入风险指标失败: {e}")
            raise


class AlertHandler(BaseEventHandler):
    """告警事件处理器"""
    
    def __init__(self, db_client: ClickHouseClient):
        super().__init__(db_client)
        self.batch_size = 100
        self.batch_timeout = 2  # 告警需要更快的响应
        
        # 告警处理统计
        self.alert_stats = {
            'total': 0,
            'by_severity': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0},
            'by_source': {}
        }
    
    async def handle(self, event_data: Dict[str, Any]):
        """处理告警事件"""
        try:
            # 解析事件
            event = AlertV1(**event_data)
            
            # 转换为数据库模型
            alert = AlertModel.from_kafka_event(event)
            
            # 添加到批处理
            await self.add_to_batch(alert)
            
            # 更新统计
            self._update_stats(event)
            
            # 处理紧急告警
            if event.severity.value == 'CRITICAL':
                await self._handle_critical_alert(event)
            
            logger.debug(f"处理告警: {event.alert_id} [{event.severity.value}]")
            
        except Exception as e:
            logger.error(f"处理告警失败: {e}")
            raise
    
    def _update_stats(self, event: AlertV1):
        """更新告警统计"""
        self.alert_stats['total'] += 1
        self.alert_stats['by_severity'][event.severity.value] += 1
        
        source = event.source.value
        if source not in self.alert_stats['by_source']:
            self.alert_stats['by_source'][source] = 0
        self.alert_stats['by_source'][source] += 1
    
    async def _handle_critical_alert(self, event: AlertV1):
        """处理关键告警"""
        try:
            # 立即写入数据库(不等待批处理)
            alert = AlertModel.from_kafka_event(event)
            await self.db_client.insert_batch('alerts', [alert.dict()])
            
            # 记录关键告警
            logger.critical(
                f"关键告警: {event.title} - {event.message} "
                f"[{event.source.value}] [{event.symbol or 'N/A'}]"
            )
            
            # 这里可以添加额外的紧急处理逻辑
            # 如发送通知、触发紧急停止等
            
        except Exception as e:
            logger.error(f"处理关键告警失败: {e}")
    
    async def flush_batch(self):
        """批量写入告警"""
        if not self.batch_buffer:
            return
        
        try:
            # 转换为字典列表
            data = [item.dict() for item in self.batch_buffer]
            
            # 批量插入
            await self.db_client.insert_batch('alerts', data)
            
            logger.info(f"批量写入告警: {len(data)} 条")
            
            # 清空缓冲区
            self.batch_buffer.clear()
            self.last_flush = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"批量写入告警失败: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取告警统计"""
        return {
            **self.alert_stats,
            'buffer_size': len(self.batch_buffer),
            'last_flush': self.last_flush.isoformat()
        }


class HealthHandler(BaseEventHandler):
    """系统健康事件处理器"""
    
    def __init__(self, db_client: ClickHouseClient):
        super().__init__(db_client)
        self.batch_size = 10  # 健康数据量较小
        self.batch_timeout = 10
        
        # 保存最新状态
        self.latest_health = None
    
    async def handle(self, event_data: Dict[str, Any]):
        """处理系统健康事件"""
        try:
            # 解析事件
            event = SystemHealthV1(**event_data)
            
            # 转换为数据库模型
            health = HealthModel.from_kafka_event(event)
            
            # 添加到批处理
            await self.add_to_batch(health)
            
            # 保存最新状态
            self.latest_health = event
            
            # 检查系统状态
            await self._check_system_status(event)
            
            logger.debug(f"处理系统健康: {event.event_id} [{event.overall_status}]")
            
        except Exception as e:
            logger.error(f"处理系统健康失败: {e}")
            raise
    
    async def _check_system_status(self, event: SystemHealthV1):
        """检查系统状态"""
        if event.overall_status == 'UNHEALTHY':
            logger.warning(
                f"系统状态不健康: CPU={event.total_cpu_usage}%, "
                f"内存={event.total_memory_usage}%, "
                f"活跃连接={event.active_connections}"
            )
        
        # 检查组件状态
        unhealthy_components = [
            comp for comp in event.components 
            if comp.status == 'UNHEALTHY'
        ]
        
        if unhealthy_components:
            comp_names = [comp.component.value for comp in unhealthy_components]
            logger.warning(f"不健康组件: {', '.join(comp_names)}")
    
    async def flush_batch(self):
        """批量写入系统健康数据"""
        if not self.batch_buffer:
            return
        
        try:
            # 转换为字典列表
            data = [item.dict() for item in self.batch_buffer]
            
            # 批量插入
            await self.db_client.insert_batch('system_health', data)
            
            logger.info(f"批量写入系统健康: {len(data)} 条")
            
            # 清空缓冲区
            self.batch_buffer.clear()
            self.last_flush = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"批量写入系统健康失败: {e}")
            raise
    
    def get_latest_health(self) -> Optional[SystemHealthV1]:
        """获取最新健康状态"""
        return self.latest_health


class EventHandlerFactory:
    """事件处理器工厂"""
    
    def __init__(self, db_client: ClickHouseClient):
        self.db_client = db_client
        self.handlers = {}
    
    def create_handler(self, handler_class: str) -> BaseEventHandler:
        """创建事件处理器"""
        if handler_class in self.handlers:
            return self.handlers[handler_class]
        
        handler_map = {
            'ExecReportHandler': ExecReportHandler,
            'RiskMetricsHandler': RiskMetricsHandler,
            'AlertHandler': AlertHandler,
            'HealthHandler': HealthHandler
        }
        
        if handler_class not in handler_map:
            raise ValueError(f"未知的处理器类型: {handler_class}")
        
        handler = handler_map[handler_class](self.db_client)
        self.handlers[handler_class] = handler
        
        return handler
    
    async def cleanup(self):
        """清理所有处理器的缓冲区"""
        for handler in self.handlers.values():
            await handler.force_flush()
        
        logger.info("所有事件处理器缓冲区已清理")