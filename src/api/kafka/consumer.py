"""
Kafka消费者管理器
================

管理多个主题的异步消费，确保高性能和可靠性。
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import json
from dataclasses import dataclass

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError, ConsumerStoppedError
import backoff

from ..config import KafkaConfig
from ..database import ClickHouseClient
from .handlers import EventHandlerFactory

logger = logging.getLogger(__name__)


@dataclass
class TopicConfig:
    """主题配置"""
    name: str
    handler_class: str
    batch_size: int = 100
    max_poll_interval_ms: int = 300000  # 5分钟
    auto_commit_interval_ms: int = 5000  # 5秒


class KafkaConsumerManager:
    """Kafka消费者管理器"""
    
    def __init__(self, kafka_config: KafkaConfig, db_client: ClickHouseClient):
        self.kafka_config = kafka_config
        self.db_client = db_client
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        self.handlers: Dict[str, Any] = {}
        self.running = False
        self.tasks: List[asyncio.Task] = []
        
        # 主题配置
        self.topic_configs = {
            'exec.reports.v1': TopicConfig(
                name='exec.reports.v1',
                handler_class='ExecReportHandler',
                batch_size=50,
                max_poll_interval_ms=180000
            ),
            'risk.metrics.v1': TopicConfig(
                name='risk.metrics.v1', 
                handler_class='RiskMetricsHandler',
                batch_size=20,
                max_poll_interval_ms=300000
            ),
            'alerts.v1': TopicConfig(
                name='alerts.v1',
                handler_class='AlertHandler',
                batch_size=100,
                max_poll_interval_ms=60000
            ),
            'system.health.v1': TopicConfig(
                name='system.health.v1',
                handler_class='HealthHandler',
                batch_size=10,
                max_poll_interval_ms=300000
            )
        }
        
        # 统计信息
        self.stats = {
            topic: {
                'messages_consumed': 0,
                'messages_processed': 0,
                'errors': 0,
                'last_message_time': None,
                'lag': 0
            }
            for topic in self.topic_configs.keys()
        }
    
    async def start(self):
        """启动所有消费者"""
        if self.running:
            logger.warning("Kafka消费者已在运行")
            return
        
        logger.info("启动Kafka消费者管理器")
        
        try:
            # 初始化事件处理器
            await self._initialize_handlers()
            
            # 创建消费者
            await self._create_consumers()
            
            # 启动消费任务
            await self._start_consumer_tasks()
            
            self.running = True
            logger.info("所有Kafka消费者启动成功")
            
        except Exception as e:
            logger.error(f"启动Kafka消费者失败: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """停止所有消费者"""
        if not self.running:
            return
        
        logger.info("停止Kafka消费者管理器")
        self.running = False
        
        # 取消所有任务
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        # 等待任务完成
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # 关闭消费者
        for consumer in self.consumers.values():
            await consumer.stop()
        
        self.consumers.clear()
        self.tasks.clear()
        
        logger.info("Kafka消费者管理器已停止")
    
    async def _initialize_handlers(self):
        """初始化事件处理器"""
        factory = EventHandlerFactory(self.db_client)
        
        for topic_config in self.topic_configs.values():
            handler = factory.create_handler(topic_config.handler_class)
            self.handlers[topic_config.name] = handler
            logger.info(f"初始化处理器: {topic_config.name} -> {topic_config.handler_class}")
    
    async def _create_consumers(self):
        """创建Kafka消费者"""
        for topic_config in self.topic_configs.values():
            consumer = AIOKafkaConsumer(
                topic_config.name,
                bootstrap_servers=self.kafka_config.bootstrap_servers,
                group_id=self.kafka_config.group_id,
                auto_offset_reset=self.kafka_config.auto_offset_reset,
                enable_auto_commit=True,
                auto_commit_interval_ms=topic_config.auto_commit_interval_ms,
                max_poll_interval_ms=topic_config.max_poll_interval_ms,
                max_poll_records=topic_config.batch_size,
                fetch_max_wait_ms=1000,
                fetch_min_bytes=1024,
                fetch_max_bytes=52428800,  # 50MB
                security_protocol=self.kafka_config.security_protocol,
                sasl_mechanism=self.kafka_config.sasl_mechanism,
                sasl_plain_username=self.kafka_config.sasl_username,
                sasl_plain_password=self.kafka_config.sasl_password,
                ssl_context=self.kafka_config.ssl_context,
                value_deserializer=self._deserialize_message,
                key_deserializer=lambda x: x.decode('utf-8') if x else None
            )
            
            self.consumers[topic_config.name] = consumer
            logger.info(f"创建消费者: {topic_config.name}")
    
    async def _start_consumer_tasks(self):
        """启动消费任务"""
        for topic_config in self.topic_configs.values():
            consumer = self.consumers[topic_config.name]
            handler = self.handlers[topic_config.name]
            
            # 启动消费者
            await consumer.start()
            
            # 创建消费任务
            task = asyncio.create_task(
                self._consume_topic(topic_config, consumer, handler)
            )
            self.tasks.append(task)
            
            logger.info(f"启动消费任务: {topic_config.name}")
    
    @backoff.on_exception(
        backoff.expo,
        (KafkaError, ConnectionError),
        max_tries=5,
        max_time=300
    )
    async def _consume_topic(self, topic_config: TopicConfig, consumer: AIOKafkaConsumer, handler):
        """消费单个主题"""
        logger.info(f"开始消费主题: {topic_config.name}")
        
        try:
            async for msg in consumer:
                if not self.running:
                    break
                
                try:
                    # 更新统计
                    self.stats[topic_config.name]['messages_consumed'] += 1
                    self.stats[topic_config.name]['last_message_time'] = datetime.utcnow()
                    
                    # 处理消息
                    await self._process_message(topic_config.name, msg, handler)
                    
                    # 更新处理统计
                    self.stats[topic_config.name]['messages_processed'] += 1
                    
                except Exception as e:
                    logger.error(f"处理消息失败 [{topic_config.name}]: {e}")
                    self.stats[topic_config.name]['errors'] += 1
                    
                    # 记录错误消息供调试
                    await self._log_error_message(topic_config.name, msg, e)
        
        except ConsumerStoppedError:
            logger.info(f"消费者已停止: {topic_config.name}")
        except Exception as e:
            logger.error(f"消费主题失败 [{topic_config.name}]: {e}")
            raise
    
    async def _process_message(self, topic: str, msg, handler):
        """处理单个消息"""
        try:
            # 解析消息
            if msg.value is None:
                logger.warning(f"收到空消息 [{topic}]: {msg.key}")
                return
            
            # 调用处理器
            await handler.handle(msg.value)
            
            logger.debug(f"消息处理成功 [{topic}]: {msg.key}")
            
        except Exception as e:
            logger.error(f"消息处理失败 [{topic}]: {e}")
            raise
    
    async def _log_error_message(self, topic: str, msg, error: Exception):
        """记录错误消息"""
        error_info = {
            'topic': topic,
            'partition': msg.partition,
            'offset': msg.offset,
            'key': msg.key,
            'timestamp': msg.timestamp,
            'error': str(error),
            'value_preview': str(msg.value)[:500] if msg.value else None
        }
        
        logger.error(f"错误消息详情: {json.dumps(error_info, indent=2)}")
    
    def _deserialize_message(self, value: bytes) -> Dict[str, Any]:
        """反序列化消息"""
        if value is None:
            return None
        
        try:
            return json.loads(value.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(f"消息反序列化失败: {e}")
            raise
    
    async def get_consumer_lag(self) -> Dict[str, int]:
        """获取消费者滞后情况"""
        lag_info = {}
        
        for topic, consumer in self.consumers.items():
            try:
                # 获取分区信息
                partitions = consumer.assignment()
                if not partitions:
                    lag_info[topic] = 0
                    continue
                
                total_lag = 0
                for partition in partitions:
                    # 获取高水位
                    high_water_mark = await consumer.highwater(partition)
                    
                    # 获取当前位置
                    current_position = await consumer.position(partition)
                    
                    # 计算滞后
                    lag = high_water_mark - current_position
                    total_lag += max(0, lag)
                
                lag_info[topic] = total_lag
                self.stats[topic]['lag'] = total_lag
                
            except Exception as e:
                logger.error(f"获取消费者滞后失败 [{topic}]: {e}")
                lag_info[topic] = -1
        
        return lag_info
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取消费者统计信息"""
        # 更新滞后信息
        lag_info = await self.get_consumer_lag()
        
        # 添加整体统计
        total_stats = {
            'total_messages_consumed': sum(s['messages_consumed'] for s in self.stats.values()),
            'total_messages_processed': sum(s['messages_processed'] for s in self.stats.values()),
            'total_errors': sum(s['errors'] for s in self.stats.values()),
            'total_lag': sum(lag_info.values()),
            'running': self.running,
            'active_consumers': len(self.consumers),
            'active_tasks': len([t for t in self.tasks if not t.done()])
        }
        
        return {
            'total': total_stats,
            'by_topic': self.stats,
            'lag': lag_info
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            stats = await self.get_stats()
            
            # 检查错误率
            error_rate = 0
            if stats['total']['total_messages_consumed'] > 0:
                error_rate = stats['total']['total_errors'] / stats['total']['total_messages_consumed'] * 100
            
            # 检查滞后
            high_lag = stats['total']['total_lag'] > 1000
            
            # 检查最近活动
            now = datetime.utcnow()
            inactive_topics = []
            
            for topic, topic_stats in stats['by_topic'].items():
                if topic_stats['last_message_time']:
                    time_diff = (now - topic_stats['last_message_time']).total_seconds()
                    if time_diff > 300:  # 5分钟无消息
                        inactive_topics.append(topic)
            
            # 判断健康状态
            if not self.running:
                status = 'unhealthy'
                issues = ['消费者未运行']
            elif error_rate > 10:
                status = 'degraded' 
                issues = [f'错误率过高: {error_rate:.2f}%']
            elif high_lag:
                status = 'degraded'
                issues = [f'消费滞后过高: {stats["total"]["total_lag"]}']
            elif inactive_topics:
                status = 'degraded'
                issues = [f'主题无活动: {", ".join(inactive_topics)}']
            else:
                status = 'healthy'
                issues = []
            
            return {
                'status': status,
                'issues': issues,
                'stats': stats,
                'timestamp': now.isoformat()
            }
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                'status': 'unhealthy',
                'issues': [f'健康检查异常: {str(e)}'],
                'timestamp': datetime.utcnow().isoformat()
            }