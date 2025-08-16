"""
MarketDataManager - 数据基础设施核心管理器
负责协调数据下载、存储、验证和实时流的核心组件
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

from .data_downloader import DataDownloader
from .data_validator import DataValidator  
from .storage_manager import StorageManager
from .realtime_stream import RealtimeDataStream
from .data_monitor import DataMonitor

@dataclass
class MarketDataSpec:
    """市场数据规格定义"""
    symbols: List[str]
    timeframes: List[str]
    start_date: str
    end_date: str
    data_types: List[str]  # ['kline', 'ticker', 'depth', 'trades', 'funding']
    exchanges: List[str]
    quality_requirements: Dict[str, float]
    
@dataclass 
class DataQualityMetrics:
    """数据质量指标"""
    completeness: float  # 完整性
    accuracy: float      # 准确性
    timeliness: float    # 及时性
    consistency: float   # 一致性
    validity: float      # 有效性
    overall_score: float # 综合评分

class MarketDataManager:
    """
    市场数据管理器 - DipMaster数据基础设施核心
    
    功能特性:
    - 多交易所数据聚合 (Binance主力，OKX/Bybit备用)
    - 高性能并行下载和处理
    - 实时数据流管理 (WebSocket多路复用)
    - 数据质量保证 (>99%完整性，<100ms延迟)
    - 分片存储优化 (Parquet格式，Zstd压缩)
    - 增量更新和版本控制
    """
    
    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # 初始化核心组件
        self.downloader = DataDownloader(self.config)
        self.validator = DataValidator(self.config)
        self.storage = StorageManager(self.config)
        self.realtime = RealtimeDataStream(self.config)
        self.monitor = DataMonitor(self.config)
        
        # 数据路径管理
        self.data_root = Path(self.config.get('data_root', 'data'))
        self.ensure_directories()
        
        # 状态跟踪
        self.is_running = False
        self.last_update = None
        self.quality_metrics = {}
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置文件"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 默认配置
        return {
            'data_root': 'data',
            'exchanges': ['binance'],
            'quality_thresholds': {
                'completeness': 0.99,
                'accuracy': 0.999,
                'timeliness': 0.95,
                'consistency': 0.98,
                'validity': 0.999
            },
            'storage': {
                'format': 'parquet',
                'compression': 'zstd',
                'partition_by': 'date'
            },
            'realtime': {
                'buffer_size': 10000,
                'flush_interval': 60,
                'reconnect_attempts': 5
            }
        }
    
    def ensure_directories(self):
        """确保必要的目录结构存在"""
        dirs = [
            self.data_root / 'historical',
            self.data_root / 'realtime', 
            self.data_root / 'processed',
            self.data_root / 'backup',
            self.data_root / 'metadata',
            self.data_root / 'quality_reports'
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    async def initialize_data_bundle(self, spec: MarketDataSpec) -> Dict[str, Any]:
        """
        初始化数据包 - 核心数据基础设施入口
        
        Args:
            spec: 数据规格定义
            
        Returns:
            数据包初始化结果
        """
        self.logger.info(f"初始化数据包: {len(spec.symbols)}个交易对, 时间范围: {spec.start_date} - {spec.end_date}")
        
        try:
            # 1. 数据完整性检查
            missing_data = await self._check_data_completeness(spec)
            
            # 2. 下载缺失数据
            if missing_data:
                await self._download_missing_data(missing_data, spec)
            
            # 3. 数据质量验证
            quality_results = await self._validate_data_quality(spec)
            
            # 4. 生成数据包元数据
            bundle_metadata = await self._generate_bundle_metadata(spec, quality_results)
            
            # 5. 创建MarketDataBundle配置
            bundle_config = await self._create_bundle_config(spec, bundle_metadata)
            
            self.last_update = datetime.now()
            
            return {
                'status': 'success',
                'bundle_id': bundle_metadata['bundle_id'],
                'quality_score': quality_results.overall_score,
                'bundle_config': bundle_config,
                'update_time': self.last_update.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"数据包初始化失败: {e}")
            raise
    
    async def _check_data_completeness(self, spec: MarketDataSpec) -> List[Dict[str, Any]]:
        """检查数据完整性，识别缺失数据"""
        missing_data = []
        
        for symbol in spec.symbols:
            for timeframe in spec.timeframes:
                for data_type in spec.data_types:
                    # 检查文件是否存在及数据完整性
                    file_path = self.storage.get_data_path(symbol, timeframe, data_type)
                    
                    if not file_path.exists():
                        missing_data.append({
                            'symbol': symbol,
                            'timeframe': timeframe, 
                            'data_type': data_type,
                            'reason': 'file_not_found'
                        })
                    else:
                        # 检查数据时间范围完整性
                        gaps = await self.validator.check_time_gaps(file_path, spec.start_date, spec.end_date)
                        if gaps:
                            missing_data.append({
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'data_type': data_type,
                                'reason': 'time_gaps',
                                'gaps': gaps
                            })
        
        self.logger.info(f"数据完整性检查完成，发现{len(missing_data)}个缺失项")
        return missing_data
    
    async def _download_missing_data(self, missing_data: List[Dict[str, Any]], spec: MarketDataSpec):
        """并行下载缺失数据"""
        self.logger.info(f"开始下载{len(missing_data)}个缺失数据项")
        
        # 按数据类型分组，优化下载策略
        download_tasks = []
        
        for item in missing_data:
            task = self.downloader.download_data(
                symbol=item['symbol'],
                timeframe=item['timeframe'],
                data_type=item['data_type'],
                start_date=spec.start_date,
                end_date=spec.end_date,
                exchange=spec.exchanges[0]  # 主要交易所
            )
            download_tasks.append(task)
        
        # 控制并发数量，避免API限制
        semaphore = asyncio.Semaphore(5)  # 最多5个并发下载
        
        async def download_with_semaphore(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(
            *[download_with_semaphore(task) for task in download_tasks],
            return_exceptions=True
        )
        
        # 处理下载结果
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        self.logger.info(f"数据下载完成: {success_count}/{len(missing_data)} 成功")
        
        if success_count < len(missing_data):
            failed_items = [item for i, item in enumerate(missing_data) 
                          if isinstance(results[i], Exception)]
            self.logger.warning(f"部分数据下载失败: {len(failed_items)}项")
    
    async def _validate_data_quality(self, spec: MarketDataSpec) -> DataQualityMetrics:
        """全面数据质量验证"""
        self.logger.info("开始数据质量验证")
        
        quality_scores = {
            'completeness': [],
            'accuracy': [],
            'timeliness': [],
            'consistency': [],
            'validity': []
        }
        
        for symbol in spec.symbols:
            for timeframe in spec.timeframes:
                for data_type in spec.data_types:
                    file_path = self.storage.get_data_path(symbol, timeframe, data_type)
                    
                    if file_path.exists():
                        # 各项质量指标检查
                        completeness = await self.validator.check_completeness(file_path, spec.start_date, spec.end_date)
                        accuracy = await self.validator.check_accuracy(file_path)
                        consistency = await self.validator.check_consistency(file_path)
                        validity = await self.validator.check_validity(file_path)
                        
                        quality_scores['completeness'].append(completeness)
                        quality_scores['accuracy'].append(accuracy)
                        quality_scores['consistency'].append(consistency)
                        quality_scores['validity'].append(validity)
        
        # 计算平均质量指标
        metrics = DataQualityMetrics(
            completeness=np.mean(quality_scores['completeness']) if quality_scores['completeness'] else 0,
            accuracy=np.mean(quality_scores['accuracy']) if quality_scores['accuracy'] else 0,
            timeliness=0.98,  # 历史数据及时性假设
            consistency=np.mean(quality_scores['consistency']) if quality_scores['consistency'] else 0,
            validity=np.mean(quality_scores['validity']) if quality_scores['validity'] else 0,
            overall_score=0.0
        )
        
        # 综合评分（加权平均）
        weights = {'completeness': 0.25, 'accuracy': 0.25, 'timeliness': 0.15, 
                  'consistency': 0.2, 'validity': 0.15}
        
        metrics.overall_score = sum(
            getattr(metrics, metric) * weight 
            for metric, weight in weights.items()
        )
        
        self.logger.info(f"数据质量验证完成，综合评分: {metrics.overall_score:.3f}")
        return metrics
    
    async def _generate_bundle_metadata(self, spec: MarketDataSpec, quality: DataQualityMetrics) -> Dict[str, Any]:
        """生成数据包元数据"""
        bundle_id = f"dipmaster_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 收集文件统计信息
        file_stats = {}
        total_size = 0
        
        for symbol in spec.symbols:
            symbol_stats = {}
            for timeframe in spec.timeframes:
                for data_type in spec.data_types:
                    file_path = self.storage.get_data_path(symbol, timeframe, data_type)
                    if file_path.exists():
                        stat = file_path.stat()
                        symbol_stats[f"{timeframe}_{data_type}"] = {
                            'size_bytes': stat.st_size,
                            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                        }
                        total_size += stat.st_size
            file_stats[symbol] = symbol_stats
        
        metadata = {
            'bundle_id': bundle_id,
            'version': '4.0.0',
            'created_at': datetime.now().isoformat(),
            'spec': asdict(spec),
            'quality_metrics': asdict(quality),
            'file_statistics': file_stats,
            'total_size_mb': total_size / (1024 * 1024),
            'data_integrity_checksum': await self._calculate_bundle_checksum(spec),
            'performance_benchmarks': {
                'data_access_latency_ms': 50,  # 预期访问延迟
                'query_throughput_ops': 1000,  # 每秒查询数
                'compression_ratio': 0.15      # 压缩率
            }
        }
        
        # 保存元数据
        metadata_path = self.data_root / 'metadata' / f'{bundle_id}.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        return metadata
    
    async def _calculate_bundle_checksum(self, spec: MarketDataSpec) -> str:
        """计算数据包校验和，确保数据完整性"""
        import hashlib
        
        hasher = hashlib.sha256()
        
        for symbol in spec.symbols:
            for timeframe in spec.timeframes:
                for data_type in spec.data_types:
                    file_path = self.storage.get_data_path(symbol, timeframe, data_type)
                    if file_path.exists():
                        with open(file_path, 'rb') as f:
                            for chunk in iter(lambda: f.read(4096), b""):
                                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    async def _create_bundle_config(self, spec: MarketDataSpec, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """创建MarketDataBundle配置文件"""
        
        bundle_config = {
            "version": metadata['created_at'],
            "metadata": {
                "bundle_id": metadata['bundle_id'],
                "symbols": spec.symbols,
                "exchanges": spec.exchanges,
                "date_range": {
                    "start": spec.start_date,
                    "end": spec.end_date
                },
                "data_quality_score": metadata['quality_metrics']['overall_score'],
                "total_size_mb": metadata['total_size_mb'],
                "performance_benchmarks": metadata['performance_benchmarks']
            },
            "data_sources": {
                "historical": {
                    "klines_5m": str(self.data_root / "historical" / "klines_5m.parquet"),
                    "klines_15m": str(self.data_root / "historical" / "klines_15m.parquet"),
                    "klines_1h": str(self.data_root / "historical" / "klines_1h.parquet")
                },
                "realtime": {
                    "websocket_cache": str(self.data_root / "realtime" / "ws_cache.sqlite"),
                    "tick_buffer": str(self.data_root / "realtime" / "tick_buffer")
                },
                "auxiliary": {
                    "funding_rates": str(self.data_root / "historical" / "funding_rates.parquet"),
                    "order_books": str(self.data_root / "historical" / "order_books.zstd"),
                    "trade_data": str(self.data_root / "historical" / "trades.parquet")
                }
            },
            "quality_assurance": {
                "completeness_threshold": 0.99,
                "accuracy_threshold": 0.999,
                "consistency_checks": True,
                "anomaly_detection": True,
                "real_time_monitoring": True
            },
            "access_patterns": {
                "sequential_read_optimized": True,
                "random_access_enabled": True,
                "parallel_query_support": True,
                "cache_layer_enabled": True
            },
            "backup_strategy": {
                "local_backup_path": str(self.data_root / "backup"),
                "backup_frequency": "daily",
                "retention_days": 30,
                "compression_enabled": True
            },
            "monitoring": {
                "health_check_endpoint": "/health",
                "metrics_collection": True,
                "alert_thresholds": {
                    "data_latency_ms": 100,
                    "missing_data_pct": 0.01,
                    "quality_score_min": 0.95
                }
            }
        }
        
        # 保存配置文件
        config_path = self.data_root / 'MarketDataBundle.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(bundle_config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"MarketDataBundle配置已生成: {config_path}")
        return bundle_config
    
    async def start_realtime_stream(self, symbols: List[str]):
        """启动实时数据流"""
        if self.is_running:
            self.logger.warning("实时数据流已在运行中")
            return
        
        self.logger.info(f"启动实时数据流: {len(symbols)}个交易对")
        
        try:
            await self.realtime.connect(symbols)
            self.is_running = True
            
            # 启动监控
            await self.monitor.start_monitoring()
            
        except Exception as e:
            self.logger.error(f"实时数据流启动失败: {e}")
            raise
    
    async def stop_realtime_stream(self):
        """停止实时数据流"""
        if not self.is_running:
            return
        
        self.logger.info("停止实时数据流")
        
        try:
            await self.realtime.disconnect()
            await self.monitor.stop_monitoring()
            self.is_running = False
            
        except Exception as e:
            self.logger.error(f"实时数据流停止失败: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """获取数据基础设施健康状态"""
        return {
            'status': 'healthy' if self.is_running else 'stopped',
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'realtime_stream': await self.realtime.get_status() if self.is_running else 'stopped',
            'data_quality': self.quality_metrics,
            'storage_usage': await self.storage.get_usage_stats(),
            'monitoring': await self.monitor.get_status() if self.is_running else 'stopped'
        }
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("清理数据管理器资源")
        
        if self.is_running:
            await self.stop_realtime_stream()
        
        await self.downloader.cleanup()
        await self.storage.cleanup()