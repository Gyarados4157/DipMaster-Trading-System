"""
Kafka消费者服务 - 高可用并行消费事件流
支持exec.reports.v1, risk.metrics.v1, alerts.v1, strategy.performance.v1
"""

import asyncio
import json
from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from aiokafka import AIOKafkaConsumer, ConsumerRecord
from aiokafka.errors import KafkaError
import structlog
from pydantic import BaseModel, ValidationError

from .config import KafkaConfig
from .schemas import (
    ExecReportEvent, RiskMetricsEvent, AlertEvent, 
    StrategyPerformanceEvent, EventBase
)

logger = structlog.get_logger(__name__)

class EventProcessor:
    """事件处理器基类"""
    
    def __init__(self, db_manager, cache_manager, websocket_manager, monitoring):
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.websocket_manager = websocket_manager
        self.monitoring = monitoring
    
    async def process_batch(self, events: List[EventBase]) -> Dict[str, Any]:
        """批量处理事件"""
        raise NotImplementedError
    
    async def process_single(self, event: EventBase) -> Dict[str, Any]:
        """处理单个事件"""
        raise NotImplementedError

class ExecReportProcessor(EventProcessor):
    """执行报告事件处理器"""
    
    async def process_batch(self, events: List[ExecReportEvent]) -> Dict[str, Any]:
        """批量处理执行报告"""
        try:
            # 批量插入到ClickHouse
            await self.db_manager.insert_exec_reports(events)
            
            # 更新缓存
            await self._update_cache(events)
            
            # 实时推送
            await self._broadcast_updates(events)
            
            # 更新监控指标
            await self.monitoring.update_exec_metrics(len(events))
            
            return {"processed": len(events), "status": "success"}
            
        except Exception as e:
            logger.error(f"执行报告批量处理失败: {e}")
            await self.monitoring.record_error("exec_report_batch_processing", str(e))
            raise
    
    async def _update_cache(self, events: List[ExecReportEvent]):
        """更新缓存数据"""
        # 更新PnL缓存
        pnl_updates = {}
        fills_updates = []
        
        for event in events:
            symbol = event.symbol
            
            # 累积PnL数据
            if symbol not in pnl_updates:
                pnl_updates[symbol] = {
                    "realized_pnl": 0,
                    "unrealized_pnl": 0,
                    "total_volume": 0,
                    "trade_count": 0,
                    "last_update": event.timestamp
                }
            
            pnl_updates[symbol]["realized_pnl"] += event.realized_pnl or 0
            pnl_updates[symbol]["unrealized_pnl"] += event.unrealized_pnl or 0
            pnl_updates[symbol]["total_volume"] += abs(event.quantity * event.price)
            pnl_updates[symbol]["trade_count"] += 1
            
            # 更新成交记录
            if event.exec_type == "TRADE":
                fills_updates.append({
                    "timestamp": event.timestamp,
                    "symbol": event.symbol,
                    "side": event.side,
                    "quantity": event.quantity,
                    "price": event.price,
                    "commission": event.commission,
                    "order_id": event.order_id
                })
        
        # 批量更新缓存
        await asyncio.gather(
            self.cache_manager.update_pnl_data(pnl_updates),
            self.cache_manager.update_fills_data(fills_updates),
            return_exceptions=True
        )
    
    async def _broadcast_updates(self, events: List[ExecReportEvent]):
        """广播实时更新"""
        # 按类型分组广播
        fills = [e for e in events if e.exec_type == "TRADE"]
        position_updates = [e for e in events if e.position_qty is not None]
        
        await asyncio.gather(
            self.websocket_manager.broadcast_to_channel("fills", fills),
            self.websocket_manager.broadcast_to_channel("positions", position_updates),
            return_exceptions=True
        )

class RiskMetricsProcessor(EventProcessor):
    """风险指标事件处理器"""
    
    async def process_batch(self, events: List[RiskMetricsEvent]) -> Dict[str, Any]:
        """批量处理风险指标"""
        try:
            # 批量插入到ClickHouse
            await self.db_manager.insert_risk_metrics(events)
            
            # 更新风险缓存
            await self._update_risk_cache(events)
            
            # 实时推送风险更新
            await self.websocket_manager.broadcast_to_channel("risk", events)
            
            # 更新监控指标
            await self.monitoring.update_risk_metrics(len(events))
            
            return {"processed": len(events), "status": "success"}
            
        except Exception as e:
            logger.error(f"风险指标批量处理失败: {e}")
            await self.monitoring.record_error("risk_metrics_batch_processing", str(e))
            raise
    
    async def _update_risk_cache(self, events: List[RiskMetricsEvent]):
        """更新风险指标缓存"""
        risk_data = {}
        
        for event in events:
            account_id = event.account_id
            risk_data[account_id] = {
                "portfolio_value": event.portfolio_value,
                "total_pnl": event.total_pnl,
                "var_1d": event.var_1d,
                "max_drawdown": event.max_drawdown,
                "leverage": event.leverage,
                "risk_score": event.risk_score,
                "positions": event.positions,
                "last_update": event.timestamp
            }
        
        await self.cache_manager.update_risk_data(risk_data)

class AlertProcessor(EventProcessor):
    """告警事件处理器"""
    
    async def process_batch(self, events: List[AlertEvent]) -> Dict[str, Any]:
        """批量处理告警事件"""
        try:
            # 批量插入到ClickHouse
            await self.db_manager.insert_alerts(events)
            
            # 实时推送告警（高优先级）
            await self._process_alerts(events)
            
            # 更新监控指标
            await self.monitoring.update_alert_metrics(events)
            
            return {"processed": len(events), "status": "success"}
            
        except Exception as e:
            logger.error(f"告警事件批量处理失败: {e}")
            await self.monitoring.record_error("alert_batch_processing", str(e))
            raise
    
    async def _process_alerts(self, events: List[AlertEvent]):
        """处理告警事件"""
        # 按严重性分类
        critical_alerts = [e for e in events if e.severity == "CRITICAL"]
        warning_alerts = [e for e in events if e.severity == "WARNING"]
        info_alerts = [e for e in events if e.severity == "INFO"]
        
        # 立即推送关键告警
        if critical_alerts:
            await self.websocket_manager.broadcast_to_channel(
                "alerts", critical_alerts, priority="high"
            )
        
        # 推送其他告警
        if warning_alerts or info_alerts:
            await self.websocket_manager.broadcast_to_channel(
                "alerts", warning_alerts + info_alerts
            )
        
        # 更新告警缓存
        await self.cache_manager.update_alert_data(events)

class StrategyPerformanceProcessor(EventProcessor):
    """策略性能事件处理器"""
    
    async def process_batch(self, events: List[StrategyPerformanceEvent]) -> Dict[str, Any]:
        """批量处理策略性能事件"""
        try:
            # 批量插入到ClickHouse
            await self.db_manager.insert_strategy_performance(events)
            
            # 更新性能缓存
            await self._update_performance_cache(events)
            
            # 更新监控指标
            await self.monitoring.update_strategy_metrics(events)
            
            return {"processed": len(events), "status": "success"}
            
        except Exception as e:
            logger.error(f"策略性能批量处理失败: {e}")
            await self.monitoring.record_error("strategy_performance_batch_processing", str(e))
            raise
    
    async def _update_performance_cache(self, events: List[StrategyPerformanceEvent]):
        """更新策略性能缓存"""
        performance_data = {}
        
        for event in events:
            strategy_id = event.strategy_id
            performance_data[strategy_id] = {
                "total_trades": event.total_trades,
                "win_rate": event.win_rate,
                "profit_factor": event.profit_factor,
                "sharpe_ratio": event.sharpe_ratio,
                "max_drawdown": event.max_drawdown,
                "total_pnl": event.total_pnl,
                "daily_returns": event.daily_returns,
                "last_update": event.timestamp
            }
        
        await self.cache_manager.update_strategy_performance(performance_data)

class KafkaConsumerService:
    """Kafka消费者服务主类"""
    
    def __init__(self, config: KafkaConfig, db_manager, cache_manager, 
                 websocket_manager, monitoring):
        self.config = config
        self.db_manager = db_manager
        self.cache_manager = cache_manager
        self.websocket_manager = websocket_manager
        self.monitoring = monitoring
        
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.processors: Dict[str, EventProcessor] = {}
        self.running = False
        
        # 初始化事件处理器
        self._init_processors()
    
    def _init_processors(self):
        """初始化事件处理器"""
        base_args = (self.db_manager, self.cache_manager, 
                    self.websocket_manager, self.monitoring)
        
        self.processors = {
            "exec.reports.v1": ExecReportProcessor(*base_args),
            "risk.metrics.v1": RiskMetricsProcessor(*base_args),
            "alerts.v1": AlertProcessor(*base_args),
            "strategy.performance.v1": StrategyPerformanceProcessor(*base_args)
        }
    
    async def initialize(self):
        """初始化消费者"""
        try:
            for topic in self.config.topics.keys():
                consumer = AIOKafkaConsumer(
                    topic,
                    bootstrap_servers=self.config.bootstrap_servers,
                    group_id=self.config.consumer_group,
                    auto_offset_reset=self.config.auto_offset_reset,
                    enable_auto_commit=self.config.enable_auto_commit,
                    max_poll_interval_ms=self.config.max_poll_interval_ms,
                    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                    security_protocol=self.config.security_protocol
                )
                
                self.consumers[topic] = consumer
                logger.info(f"初始化Kafka消费者: {topic}")
            
            logger.info("Kafka消费者服务初始化完成")
            
        except Exception as e:
            logger.error(f"Kafka消费者初始化失败: {e}")
            raise
    
    async def start_consuming(self):
        """开始消费事件"""
        self.running = True
        
        # 启动所有消费者
        tasks = []
        for topic, consumer in self.consumers.items():
            task = asyncio.create_task(self._consume_topic(topic, consumer))
            tasks.append(task)
        
        logger.info("开始Kafka事件消费")
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Kafka消费任务失败: {e}")
            await self.monitoring.record_error("kafka_consumption", str(e))
            raise
    
    async def _consume_topic(self, topic: str, consumer: AIOKafkaConsumer):
        """消费单个主题"""
        try:
            await consumer.start()
            logger.info(f"开始消费主题: {topic}")
            
            batch_buffer = []
            last_commit_time = datetime.now(timezone.utc)
            
            async for message in consumer:
                if not self.running:
                    break
                
                try:
                    # 解析事件
                    event = self._parse_event(topic, message)
                    if event:
                        batch_buffer.append(event)
                    
                    # 批量处理检查
                    if (len(batch_buffer) >= self.config.batch_size or 
                        (datetime.now(timezone.utc) - last_commit_time).seconds >= 5):
                        
                        if batch_buffer:
                            await self._process_batch(topic, batch_buffer)
                            batch_buffer.clear()
                        
                        # 手动提交偏移量
                        await consumer.commit()
                        last_commit_time = datetime.now(timezone.utc)
                        
                        # 更新消费指标
                        await self.monitoring.update_kafka_metrics(topic, len(batch_buffer))
                
                except Exception as e:
                    logger.error(f"处理消息失败 {topic}: {e}")
                    await self.monitoring.record_error(f"kafka_message_processing_{topic}", str(e))
                    continue
        
        except Exception as e:
            logger.error(f"消费主题失败 {topic}: {e}")
            raise
        finally:
            await consumer.stop()
    
    def _parse_event(self, topic: str, message: ConsumerRecord) -> Optional[EventBase]:
        """解析事件消息"""
        try:
            data = message.value
            
            # 根据主题类型解析
            if topic == "exec.reports.v1":
                return ExecReportEvent(**data)
            elif topic == "risk.metrics.v1":
                return RiskMetricsEvent(**data)
            elif topic == "alerts.v1":
                return AlertEvent(**data)
            elif topic == "strategy.performance.v1":
                return StrategyPerformanceEvent(**data)
            else:
                logger.warning(f"未知主题类型: {topic}")
                return None
                
        except ValidationError as e:
            logger.error(f"事件解析失败 {topic}: {e}")
            return None
        except Exception as e:
            logger.error(f"事件处理异常 {topic}: {e}")
            return None
    
    async def _process_batch(self, topic: str, events: List[EventBase]):
        """批量处理事件"""
        processor = self.processors.get(topic)
        if processor:
            try:
                result = await processor.process_batch(events)
                logger.debug(f"批量处理完成 {topic}: {result}")
            except Exception as e:
                logger.error(f"批量处理失败 {topic}: {e}")
                # 尝试逐个处理
                await self._fallback_process(processor, events)
        else:
            logger.warning(f"未找到处理器: {topic}")
    
    async def _fallback_process(self, processor: EventProcessor, events: List[EventBase]):
        """降级处理 - 逐个处理事件"""
        for event in events:
            try:
                await processor.process_single(event)
            except Exception as e:
                logger.error(f"单个事件处理失败: {e}")
                continue
    
    async def shutdown(self):
        """关闭消费者服务"""
        self.running = False
        
        for topic, consumer in self.consumers.items():
            try:
                await consumer.stop()
                logger.info(f"关闭消费者: {topic}")
            except Exception as e:
                logger.error(f"关闭消费者失败 {topic}: {e}")
        
        logger.info("Kafka消费者服务关闭完成")