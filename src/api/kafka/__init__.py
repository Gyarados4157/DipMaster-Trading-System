"""
Kafka消费者服务
==============

提供异步Kafka消费和事件处理能力。
"""

from .consumer import KafkaConsumerManager
from .handlers import *

__all__ = [
    "KafkaConsumerManager",
    "ExecReportHandler",
    "RiskMetricsHandler", 
    "AlertHandler",
    "HealthHandler"
]