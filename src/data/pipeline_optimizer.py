#!/usr/bin/env python3
"""
DipMaster Trading System - Data Pipeline Optimizer
数据管道性能优化器，自动调整管道参数以提高效率

Author: DipMaster Development Team
Date: 2025-08-16
Version: 4.0.0
"""

import asyncio
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import sqlite3
import threading

logger = logging.getLogger(__name__)

class OptimizationStrategy(str, Enum):
    """优化策略"""
    THROUGHPUT = "THROUGHPUT"       # 最大化吞吐量
    LATENCY = "LATENCY"            # 最小化延迟
    BALANCED = "BALANCED"          # 平衡模式
    RESOURCE_EFFICIENT = "RESOURCE_EFFICIENT"  # 资源效率

class OptimizationAction(str, Enum):
    """优化动作"""
    INCREASE_WORKERS = "INCREASE_WORKERS"
    DECREASE_WORKERS = "DECREASE_WORKERS"
    INCREASE_BATCH_SIZE = "INCREASE_BATCH_SIZE"
    DECREASE_BATCH_SIZE = "DECREASE_BATCH_SIZE"
    ENABLE_CACHING = "ENABLE_CACHING"
    DISABLE_CACHING = "DISABLE_CACHING"
    ADJUST_BUFFER_SIZE = "ADJUST_BUFFER_SIZE"
    OPTIMIZE_MEMORY = "OPTIMIZE_MEMORY"

@dataclass
class OptimizationConfig:
    """优化配置"""
    strategy: OptimizationStrategy
    target_throughput_rps: float
    target_latency_ms: float
    max_memory_mb: float
    max_cpu_percent: float
    optimization_interval_minutes: int = 15
    auto_apply: bool = False  # 是否自动应用优化建议

@dataclass
class PipelineParameters:
    """管道参数"""
    worker_count: int = 4
    batch_size: int = 1000
    buffer_size: int = 10000
    cache_enabled: bool = True
    memory_limit_mb: float = 2048
    connection_pool_size: int = 20
    async_processing: bool = True
    compression_enabled: bool = True

@dataclass 
class OptimizationResult:
    """优化结果"""
    timestamp: datetime
    strategy: OptimizationStrategy
    current_params: PipelineParameters
    recommended_params: PipelineParameters
    expected_improvement: Dict[str, float]
    confidence_score: float
    actions: List[OptimizationAction]
    reason: str

class PipelineOptimizer:
    """数据管道优化器"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.current_params = PipelineParameters()
        self.optimization_history = []
        
        # 性能基线
        self.baseline_metrics = {
            'throughput_rps': 0.0,
            'latency_ms': 0.0,
            'cpu_usage': 0.0,
            'memory_usage': 0.0,
            'error_rate': 0.0
        }
        
        # 优化锁，避免并发优化
        self.optimization_lock = threading.Lock()
        
        # 启动自动优化
        if config.auto_apply:
            self._start_auto_optimization()
    
    def _start_auto_optimization(self):
        """启动自动优化"""
        def optimization_loop():
            while True:
                try:
                    time.sleep(self.config.optimization_interval_minutes * 60)
                    self.optimize_pipeline()
                except Exception as e:
                    logger.error(f"Auto optimization error: {e}")
        
        optimization_thread = threading.Thread(target=optimization_loop, daemon=True)
        optimization_thread.start()
        logger.info("Auto optimization started")
    
    def analyze_current_performance(self, 
                                  metrics_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """分析当前性能"""
        if not metrics_data:
            return self.baseline_metrics.copy()
        
        # 计算平均性能指标
        performance = {
            'throughput_rps': np.mean([m.get('throughput_rps', 0) for m in metrics_data]),
            'latency_ms': np.mean([m.get('latency_ms', 0) for m in metrics_data]),
            'cpu_usage': np.mean([m.get('cpu_usage_percent', 0) for m in metrics_data]),
            'memory_usage': np.mean([m.get('memory_usage_mb', 0) for m in metrics_data]),
            'error_rate': np.mean([m.get('error_count', 0) / max(1, m.get('total_processed', 1)) for m in metrics_data])
        }
        
        return performance
    
    def identify_bottlenecks(self, performance: Dict[str, float]) -> List[str]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        # 吞吐量瓶颈
        if performance['throughput_rps'] < self.config.target_throughput_rps * 0.8:
            bottlenecks.append("LOW_THROUGHPUT")
        
        # 延迟瓶颈
        if performance['latency_ms'] > self.config.target_latency_ms * 1.2:
            bottlenecks.append("HIGH_LATENCY")
        
        # 资源瓶颈
        if performance['cpu_usage'] > self.config.max_cpu_percent * 0.9:
            bottlenecks.append("HIGH_CPU_USAGE")
        
        if performance['memory_usage'] > self.config.max_memory_mb * 0.9:
            bottlenecks.append("HIGH_MEMORY_USAGE")
        
        # 错误率瓶颈
        if performance['error_rate'] > 0.05:  # 5% error rate
            bottlenecks.append("HIGH_ERROR_RATE")
        
        return bottlenecks
    
    def generate_optimization_recommendations(self, 
                                            performance: Dict[str, float],
                                            bottlenecks: List[str]) -> OptimizationResult:
        """生成优化建议"""
        recommended_params = PipelineParameters(
            worker_count=self.current_params.worker_count,
            batch_size=self.current_params.batch_size,
            buffer_size=self.current_params.buffer_size,
            cache_enabled=self.current_params.cache_enabled,
            memory_limit_mb=self.current_params.memory_limit_mb,
            connection_pool_size=self.current_params.connection_pool_size,
            async_processing=self.current_params.async_processing,
            compression_enabled=self.current_params.compression_enabled
        )
        
        actions = []
        expected_improvement = {
            'throughput_improvement': 0.0,
            'latency_improvement': 0.0,
            'memory_reduction': 0.0,
            'cpu_reduction': 0.0
        }
        
        confidence_score = 0.8  # 基础置信度
        reason_parts = []
        
        # 根据策略和瓶颈生成建议
        if self.config.strategy == OptimizationStrategy.THROUGHPUT:
            actions, expected_improvement, reason_parts = self._optimize_for_throughput(
                performance, bottlenecks, recommended_params
            )
        elif self.config.strategy == OptimizationStrategy.LATENCY:
            actions, expected_improvement, reason_parts = self._optimize_for_latency(
                performance, bottlenecks, recommended_params
            )
        elif self.config.strategy == OptimizationStrategy.BALANCED:
            actions, expected_improvement, reason_parts = self._optimize_balanced(
                performance, bottlenecks, recommended_params
            )
        elif self.config.strategy == OptimizationStrategy.RESOURCE_EFFICIENT:
            actions, expected_improvement, reason_parts = self._optimize_for_efficiency(
                performance, bottlenecks, recommended_params
            )
        
        # 调整置信度
        if len(bottlenecks) > 3:
            confidence_score *= 0.8  # 瓶颈太多，置信度降低
        if not actions:
            confidence_score = 0.0  # 没有建议，置信度为0
        
        reason = "; ".join(reason_parts) if reason_parts else "No optimization needed"
        
        return OptimizationResult(
            timestamp=datetime.now(),
            strategy=self.config.strategy,
            current_params=self.current_params,
            recommended_params=recommended_params,
            expected_improvement=expected_improvement,
            confidence_score=confidence_score,
            actions=actions,
            reason=reason
        )
    
    def _optimize_for_throughput(self, 
                               performance: Dict[str, float],
                               bottlenecks: List[str],
                               params: PipelineParameters) -> Tuple[List[OptimizationAction], Dict[str, float], List[str]]:
        """针对吞吐量优化"""
        actions = []
        expected_improvement = {
            'throughput_improvement': 0.0,
            'latency_improvement': 0.0,
            'memory_reduction': 0.0,
            'cpu_reduction': 0.0
        }
        reasons = []
        
        if "LOW_THROUGHPUT" in bottlenecks:
            # 增加工作线程
            if params.worker_count < 16 and performance['cpu_usage'] < self.config.max_cpu_percent * 0.7:
                new_workers = min(16, params.worker_count * 2)
                params.worker_count = new_workers
                actions.append(OptimizationAction.INCREASE_WORKERS)
                expected_improvement['throughput_improvement'] += 0.3  # 预期30%提升
                reasons.append(f"Increased workers to {new_workers} for higher throughput")
            
            # 增加批处理大小
            if params.batch_size < 5000:
                new_batch_size = min(5000, params.batch_size * 2)
                params.batch_size = new_batch_size
                actions.append(OptimizationAction.INCREASE_BATCH_SIZE)
                expected_improvement['throughput_improvement'] += 0.2  # 预期20%提升
                reasons.append(f"Increased batch size to {new_batch_size}")
            
            # 启用缓存
            if not params.cache_enabled:
                params.cache_enabled = True
                actions.append(OptimizationAction.ENABLE_CACHING)
                expected_improvement['throughput_improvement'] += 0.15  # 预期15%提升
                reasons.append("Enabled caching for better throughput")
        
        if "HIGH_LATENCY" in bottlenecks and performance['throughput_rps'] > self.config.target_throughput_rps * 0.9:
            # 在保持吞吐量的情况下优化延迟
            if params.buffer_size > 1000:
                params.buffer_size = max(1000, params.buffer_size // 2)
                actions.append(OptimizationAction.ADJUST_BUFFER_SIZE)
                expected_improvement['latency_improvement'] = 0.1
                reasons.append("Reduced buffer size to improve latency")
        
        return actions, expected_improvement, reasons
    
    def _optimize_for_latency(self, 
                            performance: Dict[str, float],
                            bottlenecks: List[str],
                            params: PipelineParameters) -> Tuple[List[OptimizationAction], Dict[str, float], List[str]]:
        """针对延迟优化"""
        actions = []
        expected_improvement = {
            'throughput_improvement': 0.0,
            'latency_improvement': 0.0,
            'memory_reduction': 0.0,
            'cpu_reduction': 0.0
        }
        reasons = []
        
        if "HIGH_LATENCY" in bottlenecks:
            # 减少批处理大小
            if params.batch_size > 100:
                new_batch_size = max(100, params.batch_size // 2)
                params.batch_size = new_batch_size
                actions.append(OptimizationAction.DECREASE_BATCH_SIZE)
                expected_improvement['latency_improvement'] += 0.25  # 预期25%改善
                reasons.append(f"Reduced batch size to {new_batch_size} for lower latency")
            
            # 减少缓冲区大小
            if params.buffer_size > 1000:
                new_buffer_size = max(1000, params.buffer_size // 2)
                params.buffer_size = new_buffer_size
                actions.append(OptimizationAction.ADJUST_BUFFER_SIZE)
                expected_improvement['latency_improvement'] += 0.15
                reasons.append(f"Reduced buffer size to {new_buffer_size}")
            
            # 启用异步处理
            if not params.async_processing:
                params.async_processing = True
                expected_improvement['latency_improvement'] += 0.2
                reasons.append("Enabled async processing for better latency")
        
        return actions, expected_improvement, reasons
    
    def _optimize_balanced(self, 
                         performance: Dict[str, float],
                         bottlenecks: List[str],
                         params: PipelineParameters) -> Tuple[List[OptimizationAction], Dict[str, float], List[str]]:
        """平衡优化"""
        actions = []
        expected_improvement = {
            'throughput_improvement': 0.0,
            'latency_improvement': 0.0,
            'memory_reduction': 0.0,
            'cpu_reduction': 0.0
        }
        reasons = []
        
        # 平衡吞吐量和延迟
        throughput_ratio = performance['throughput_rps'] / max(1, self.config.target_throughput_rps)
        latency_ratio = performance['latency_ms'] / max(1, self.config.target_latency_ms)
        
        if throughput_ratio < 0.8 and latency_ratio < 1.2:
            # 吞吐量不足但延迟可接受，适度增加并行度
            if params.worker_count < 8:
                params.worker_count = min(8, params.worker_count + 2)
                actions.append(OptimizationAction.INCREASE_WORKERS)
                expected_improvement['throughput_improvement'] += 0.15
                reasons.append("Moderately increased workers for balanced performance")
        
        elif latency_ratio > 1.5 and throughput_ratio > 0.9:
            # 延迟过高但吞吐量充足，减少批大小
            if params.batch_size > 500:
                params.batch_size = max(500, params.batch_size - 200)
                actions.append(OptimizationAction.DECREASE_BATCH_SIZE)
                expected_improvement['latency_improvement'] += 0.15
                reasons.append("Reduced batch size for balanced latency")
        
        # 资源优化
        if performance['memory_usage'] > self.config.max_memory_mb * 0.8:
            if params.compression_enabled == False:
                params.compression_enabled = True
                expected_improvement['memory_reduction'] += 0.1
                reasons.append("Enabled compression to reduce memory usage")
        
        return actions, expected_improvement, reasons
    
    def _optimize_for_efficiency(self, 
                               performance: Dict[str, float],
                               bottlenecks: List[str],
                               params: PipelineParameters) -> Tuple[List[OptimizationAction], Dict[str, float], List[str]]:
        """资源效率优化"""
        actions = []
        expected_improvement = {
            'throughput_improvement': 0.0,
            'latency_improvement': 0.0,
            'memory_reduction': 0.0,
            'cpu_reduction': 0.0
        }
        reasons = []
        
        # 减少资源使用
        if "HIGH_CPU_USAGE" in bottlenecks:
            if params.worker_count > 2:
                params.worker_count = max(2, params.worker_count - 1)
                actions.append(OptimizationAction.DECREASE_WORKERS)
                expected_improvement['cpu_reduction'] += 0.15
                reasons.append("Reduced workers to lower CPU usage")
        
        if "HIGH_MEMORY_USAGE" in bottlenecks:
            # 启用压缩
            if not params.compression_enabled:
                params.compression_enabled = True
                expected_improvement['memory_reduction'] += 0.2
                reasons.append("Enabled compression for memory efficiency")
            
            # 减少缓冲区
            if params.buffer_size > 2000:
                params.buffer_size = max(2000, params.buffer_size // 2)
                actions.append(OptimizationAction.ADJUST_BUFFER_SIZE)
                expected_improvement['memory_reduction'] += 0.1
                reasons.append("Reduced buffer size for memory efficiency")
        
        # 优化缓存策略
        if performance['throughput_rps'] > self.config.target_throughput_rps * 1.2:
            # 吞吐量过剩，可以关闭一些缓存以节省内存
            if params.cache_enabled and performance['memory_usage'] > self.config.max_memory_mb * 0.7:
                params.cache_enabled = False
                actions.append(OptimizationAction.DISABLE_CACHING)
                expected_improvement['memory_reduction'] += 0.15
                reasons.append("Disabled caching to save memory")
        
        return actions, expected_improvement, reasons
    
    def apply_optimization(self, optimization_result: OptimizationResult) -> bool:
        """应用优化建议"""
        with self.optimization_lock:
            try:
                logger.info(f"Applying optimization: {optimization_result.reason}")
                
                # 备份当前参数
                previous_params = PipelineParameters(
                    worker_count=self.current_params.worker_count,
                    batch_size=self.current_params.batch_size,
                    buffer_size=self.current_params.buffer_size,
                    cache_enabled=self.current_params.cache_enabled,
                    memory_limit_mb=self.current_params.memory_limit_mb,
                    connection_pool_size=self.current_params.connection_pool_size,
                    async_processing=self.current_params.async_processing,
                    compression_enabled=self.current_params.compression_enabled
                )
                
                # 应用新参数
                self.current_params = optimization_result.recommended_params
                
                # 记录优化历史
                self.optimization_history.append({
                    'timestamp': optimization_result.timestamp,
                    'strategy': optimization_result.strategy,
                    'previous_params': previous_params,
                    'new_params': self.current_params,
                    'expected_improvement': optimization_result.expected_improvement,
                    'actions': optimization_result.actions,
                    'confidence_score': optimization_result.confidence_score
                })
                
                # 限制历史记录长度
                if len(self.optimization_history) > 100:
                    self.optimization_history = self.optimization_history[-50:]
                
                logger.info(f"Optimization applied successfully. New params: "
                          f"workers={self.current_params.worker_count}, "
                          f"batch_size={self.current_params.batch_size}, "
                          f"buffer_size={self.current_params.buffer_size}")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to apply optimization: {e}")
                return False
    
    def optimize_pipeline(self, recent_metrics: Optional[List[Dict[str, Any]]] = None) -> OptimizationResult:
        """执行管道优化"""
        with self.optimization_lock:
            # 如果没有提供指标，使用默认性能数据
            if recent_metrics is None:
                recent_metrics = []
            
            # 分析当前性能
            current_performance = self.analyze_current_performance(recent_metrics)
            
            # 识别瓶颈
            bottlenecks = self.identify_bottlenecks(current_performance)
            
            # 生成优化建议
            optimization_result = self.generate_optimization_recommendations(
                current_performance, bottlenecks
            )
            
            logger.info(f"Pipeline optimization analysis completed. "
                      f"Bottlenecks: {bottlenecks}, "
                      f"Confidence: {optimization_result.confidence_score:.2f}, "
                      f"Actions: {len(optimization_result.actions)}")
            
            # 如果启用自动应用且置信度足够高
            if self.config.auto_apply and optimization_result.confidence_score > 0.7:
                self.apply_optimization(optimization_result)
            
            return optimization_result
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        return {
            'current_parameters': {
                'worker_count': self.current_params.worker_count,
                'batch_size': self.current_params.batch_size,
                'buffer_size': self.current_params.buffer_size,
                'cache_enabled': self.current_params.cache_enabled,
                'memory_limit_mb': self.current_params.memory_limit_mb,
                'async_processing': self.current_params.async_processing,
                'compression_enabled': self.current_params.compression_enabled
            },
            'optimization_config': {
                'strategy': self.config.strategy,
                'target_throughput_rps': self.config.target_throughput_rps,
                'target_latency_ms': self.config.target_latency_ms,
                'max_memory_mb': self.config.max_memory_mb,
                'auto_apply': self.config.auto_apply
            },
            'optimization_history_count': len(self.optimization_history),
            'last_optimization': (
                self.optimization_history[-1]['timestamp'].isoformat() 
                if self.optimization_history else None
            ),
            'baseline_metrics': self.baseline_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def rollback_last_optimization(self) -> bool:
        """回滚最后一次优化"""
        with self.optimization_lock:
            if not self.optimization_history:
                logger.warning("No optimization history to rollback")
                return False
            
            try:
                last_optimization = self.optimization_history[-1]
                self.current_params = last_optimization['previous_params']
                self.optimization_history.pop()
                
                logger.info("Last optimization rolled back successfully")
                return True
                
            except Exception as e:
                logger.error(f"Failed to rollback optimization: {e}")
                return False


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 创建优化配置
    config = OptimizationConfig(
        strategy=OptimizationStrategy.BALANCED,
        target_throughput_rps=200.0,
        target_latency_ms=500.0,
        max_memory_mb=4096.0,
        max_cpu_percent=80.0,
        optimization_interval_minutes=15,
        auto_apply=False
    )
    
    # 创建优化器
    optimizer = PipelineOptimizer(config)
    
    print("Testing pipeline optimizer...")
    
    # 模拟性能数据
    test_metrics = [
        {
            'throughput_rps': 150.0,  # 低于目标
            'latency_ms': 800.0,      # 高于目标
            'cpu_usage_percent': 90.0, # 高CPU使用
            'memory_usage_mb': 3000.0,
            'error_count': 5,
            'total_processed': 1000
        },
        {
            'throughput_rps': 160.0,
            'latency_ms': 750.0,
            'cpu_usage_percent': 85.0,
            'memory_usage_mb': 3200.0,
            'error_count': 3,
            'total_processed': 1200
        }
    ]
    
    # 执行优化
    result = optimizer.optimize_pipeline(test_metrics)
    
    print(f"Optimization Strategy: {result.strategy}")
    print(f"Confidence Score: {result.confidence_score:.2f}")
    print(f"Actions: {result.actions}")
    print(f"Reason: {result.reason}")
    print(f"Expected Improvements: {result.expected_improvement}")
    
    # 应用优化
    if result.confidence_score > 0.5:
        success = optimizer.apply_optimization(result)
        print(f"Optimization applied: {success}")
    
    # 获取报告
    report = optimizer.get_optimization_report()
    print(f"Optimization Report: {json.dumps(report, indent=2)}")
    
    print("Pipeline optimizer test completed!")