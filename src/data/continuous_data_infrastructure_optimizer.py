"""
Continuous Data Infrastructure Optimizer for DipMaster Trading System
持续数据基础设施优化器 - 自动化数据质量提升和扩展

Features:
- 持续数据质量监控与自动修复
- TOP30币种数据扩展与管理
- 增量数据更新机制
- 高频数据支持优化
- 多时间框架数据准备
- 自动数据版本管理
- 实时数据gap检测与填补
- 数据标签系统管理
"""

import asyncio
import aiohttp
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta, timezone
import json
import sqlite3
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from pathlib import Path
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dataclasses import dataclass, asdict
import warnings
import schedule
from threading import Lock, Event
import subprocess
import shutil
import gzip
from collections import defaultdict, deque
import redis
from sqlalchemy import create_engine, text, MetaData, Table, Column, String, Float, DateTime, Integer, Boolean
from sqlalchemy.orm import sessionmaker
import duckdb
import polars as pl
from enum import Enum
import yaml
import msgpack

warnings.filterwarnings('ignore')

class DataTier(Enum):
    """数据质量等级"""
    TIER_S = "tier_s"    # 顶级质量：99.9%+ 完整性
    TIER_A = "tier_a"    # 优秀质量：99.5%+ 完整性
    TIER_B = "tier_b"    # 良好质量：99.0%+ 完整性
    TIER_C = "tier_c"    # 可用质量：95.0%+ 完整性
    TIER_D = "tier_d"    # 需要修复：<95% 完整性

@dataclass
class DataUpdateTask:
    """数据更新任务"""
    symbol: str
    timeframe: str
    priority: int
    last_update: datetime
    next_update: datetime
    status: str  # 'pending', 'running', 'completed', 'failed'
    retry_count: int = 0
    error_message: str = ""

@dataclass
class QualityIssue:
    """数据质量问题"""
    symbol: str
    timeframe: str
    issue_type: str
    description: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    detected_at: datetime
    resolved: bool = False
    resolution_method: str = ""

class ContinuousDataInfrastructureOptimizer:
    """持续数据基础设施优化器"""
    
    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger(__name__)
        self._setup_logging()
        
        # 配置加载
        self.config = self._load_config(config_path)
        
        # 核心组件
        self.base_path = Path(self.config.get('base_path', 'data/enhanced_market_data'))
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化交易所
        self.exchanges = {}
        self._init_exchanges()
        
        # 任务队列和状态管理
        self.update_queue = deque()
        self.quality_issues = []
        self.task_lock = Lock()
        self.stop_event = Event()
        
        # 性能监控
        self.performance_metrics = {
            'total_symbols': 0,
            'total_timeframes': 0,
            'data_quality_score': 0.0,
            'last_update_time': None,
            'gaps_detected': 0,
            'gaps_fixed': 0,
            'failed_updates': 0
        }
        
        # TOP30交易对配置
        self.top30_symbols = self._get_top30_symbols()
        
        # 时间框架配置
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        
        # Redis连接 (可选)
        try:
            self.redis_client = redis.Redis(
                host=self.config.get('redis_host', 'localhost'),
                port=self.config.get('redis_port', 6379),
                db=self.config.get('redis_db', 0),
                decode_responses=True
            )
            self.redis_available = True
        except:
            self.redis_client = None
            self.redis_available = False
            self.logger.warning("Redis不可用，将使用内存缓存")
    
    def _setup_logging(self):
        """设置日志"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler('logs/continuous_data_optimizer.log'),
                logging.StreamHandler()
            ]
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            'base_path': 'data/enhanced_market_data',
            'update_interval_minutes': 30,
            'quality_check_interval_minutes': 60,
            'max_concurrent_downloads': 5,
            'data_retention_days': 1095,  # 3年
            'quality_threshold': 0.995,
            'auto_repair': True,
            'enable_notifications': False,
            'exchanges': {
                'binance': {
                    'rateLimit': 1200,
                    'enableRateLimit': True
                }
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) if config_path.endswith('.yaml') else json.load(f)
                default_config.update(user_config)
            except Exception as e:
                self.logger.warning(f"配置文件加载失败: {e}")
        
        return default_config
    
    def _init_exchanges(self):
        """初始化交易所"""
        for exchange_name, config in self.config.get('exchanges', {}).items():
            try:
                exchange_class = getattr(ccxt, exchange_name)
                self.exchanges[exchange_name] = exchange_class(config)
                self.logger.info(f"初始化交易所: {exchange_name}")
            except Exception as e:
                self.logger.error(f"初始化 {exchange_name} 失败: {e}")
    
    def _get_top30_symbols(self) -> List[str]:
        """获取TOP30交易对"""
        return [
            # TOP 10 - Tier S
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT",
            
            # TOP 11-20 - Tier A
            "LTCUSDT", "DOTUSDT", "MATICUSDT", "UNIUSDT", "ICPUSDT",
            "NEARUSDT", "XLMUSDT", "ATOMUSDT", "VETUSDT", "FILUSDT",
            
            # TOP 21-30 - Tier B
            "APTUSDT", "ARBUSDT", "OPUSDT", "GRTUSDT", "MKRUSDT",
            "AAVEUSDT", "COMPUSDT", "ALGOUSDT", "TONUSDT", "INJUSDT"
        ]
    
    async def start_continuous_optimization(self):
        """启动持续优化服务"""
        self.logger.info("启动持续数据基础设施优化服务...")
        
        # 初始数据收集
        await self.initial_data_collection()
        
        # 启动定时任务
        self._schedule_tasks()
        
        # 启动主循环
        await self._main_optimization_loop()
    
    async def initial_data_collection(self):
        """初始数据收集"""
        self.logger.info("执行初始数据收集...")
        
        # 批量下载3年历史数据
        await self.expand_data_collection(days=1095)
        
        # 执行初始质量评估
        await self.comprehensive_quality_assessment()
        
        # 生成初始报告
        self.generate_infrastructure_report()
    
    async def expand_data_collection(self, days: int = 1095):
        """扩展数据收集"""
        self.logger.info(f"扩展数据收集: {len(self.top30_symbols)} 币种, {len(self.timeframes)} 时间框架, {days} 天")
        
        semaphore = asyncio.Semaphore(self.config.get('max_concurrent_downloads', 5))
        tasks = []
        
        for symbol in self.top30_symbols:
            for timeframe in self.timeframes:
                task = self._download_with_semaphore(semaphore, symbol, timeframe, days)
                tasks.append(task)
        
        # 并发执行下载任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        successful_downloads = sum(1 for r in results if not isinstance(r, Exception))
        failed_downloads = len(results) - successful_downloads
        
        self.logger.info(f"数据收集完成: 成功 {successful_downloads}, 失败 {failed_downloads}")
        self.performance_metrics['total_symbols'] = len(self.top30_symbols)
        self.performance_metrics['total_timeframes'] = len(self.timeframes)
    
    async def _download_with_semaphore(self, semaphore: asyncio.Semaphore, 
                                     symbol: str, timeframe: str, days: int):
        """使用信号量控制的下载"""
        async with semaphore:
            try:
                return await self._download_historical_data(symbol, timeframe, days)
            except Exception as e:
                self.logger.error(f"下载失败 {symbol} {timeframe}: {e}")
                self.performance_metrics['failed_updates'] += 1
                return None
    
    async def _download_historical_data(self, symbol: str, timeframe: str, days: int):
        """下载历史数据"""
        if 'binance' not in self.exchanges:
            raise ValueError("Binance交易所未初始化")
        
        exchange = self.exchanges['binance']
        
        # 计算时间范围
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)
        
        # 检查是否已存在数据
        existing_data = self._load_existing_data(symbol, timeframe)
        if not existing_data.empty:
            # 增量更新
            start_time = max(start_time, existing_data.index.max() + pd.Timedelta(minutes=1))
        
        if start_time >= end_time:
            self.logger.info(f"{symbol} {timeframe} 数据已是最新")
            return existing_data
        
        # 分批下载
        all_data = []
        current_since = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        limit = 1000
        
        self.logger.info(f"下载 {symbol} {timeframe}: {start_time.strftime('%Y-%m-%d')} - {end_time.strftime('%Y-%m-%d')}")
        
        while current_since < end_timestamp:
            try:
                ohlcv = await exchange.fetch_ohlcv(
                    symbol, timeframe, current_since, limit
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                current_since = ohlcv[-1][0] + 1
                
                # 避免频率限制
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.logger.warning(f"下载批次失败 {symbol} {timeframe}: {e}")
                await asyncio.sleep(1)
                continue
        
        if not all_data:
            return existing_data
        
        # 转换为DataFrame
        new_df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms', utc=True)
        new_df.set_index('timestamp', inplace=True)
        new_df = new_df.drop_duplicates().sort_index()
        
        # 合并新旧数据
        if not existing_data.empty:
            combined_df = pd.concat([existing_data, new_df]).drop_duplicates().sort_index()
        else:
            combined_df = new_df
        
        # 数据质量检查
        quality_metrics = self._assess_data_quality(combined_df, symbol, timeframe)
        
        # 自动修复
        if quality_metrics['overall_score'] < self.config.get('quality_threshold', 0.995):
            self.logger.warning(f"{symbol} {timeframe} 质量较低 ({quality_metrics['overall_score']:.3f}), 执行修复")
            combined_df = self._repair_data(combined_df)
        
        # 保存数据
        self._save_data(combined_df, symbol, timeframe)
        
        self.logger.info(f"{symbol} {timeframe} 完成: {len(combined_df)} 条记录")
        return combined_df
    
    def _load_existing_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """加载现有数据"""
        file_path = self.base_path / f"{symbol}_{timeframe}_2years.parquet"
        
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
                    df.set_index('timestamp', inplace=True)
                return df
            except Exception as e:
                self.logger.error(f"加载数据失败 {file_path}: {e}")
        
        return pd.DataFrame()
    
    def _save_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """保存数据"""
        if df.empty:
            return
        
        file_path = self.base_path / f"{symbol}_{timeframe}_2years.parquet"
        metadata_path = self.base_path / f"{symbol}_{timeframe}_2years_metadata.json"
        
        try:
            # 保存数据
            df.to_parquet(
                file_path,
                compression='zstd',
                index=True
            )
            
            # 保存元数据
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'records': len(df),
                'date_range': {
                    'start': df.index.min().isoformat() if not df.empty else None,
                    'end': df.index.max().isoformat() if not df.empty else None
                },
                'file_size_mb': file_path.stat().st_size / 1024 / 1024,
                'last_updated': datetime.now(timezone.utc).isoformat(),
                'quality_score': self._assess_data_quality(df, symbol, timeframe)['overall_score']
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"保存数据失败 {symbol} {timeframe}: {e}")
    
    def _assess_data_quality(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, float]:
        """评估数据质量"""
        if df.empty:
            return {'completeness': 0.0, 'consistency': 0.0, 'accuracy': 0.0, 'overall_score': 0.0}
        
        metrics = {}
        
        # 1. 完整性 - 检查缺失值和时间gaps
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        # 时间序列连续性
        if len(df) > 1:
            time_diff = df.index.to_series().diff()
            expected_interval = self._get_expected_interval(timeframe)
            large_gaps = (time_diff > expected_interval * 2).sum()
            gap_ratio = large_gaps / len(df)
        else:
            gap_ratio = 0
        
        metrics['completeness'] = max(0, 1 - missing_ratio - gap_ratio * 0.1)
        
        # 2. 一致性 - OHLC关系检查
        ohlc_violations = 0
        if len(df) > 0:
            ohlc_violations += (df['high'] < df[['open', 'close']].max(axis=1)).sum()
            ohlc_violations += (df['low'] > df[['open', 'close']].min(axis=1)).sum()
            ohlc_violations += (df['high'] < df['low']).sum()
        
        metrics['consistency'] = max(0, 1 - ohlc_violations / (len(df) * 3))
        
        # 3. 准确性 - 异常值检测
        if len(df) > 10:
            returns = df['close'].pct_change().abs()
            extreme_moves = (returns > 0.5).sum()  # 超过50%变动
            metrics['accuracy'] = max(0, 1 - extreme_moves / len(df))
        else:
            metrics['accuracy'] = 1.0
        
        # 4. 综合评分
        metrics['overall_score'] = np.mean([metrics['completeness'], metrics['consistency'], metrics['accuracy']])
        
        return metrics
    
    def _get_expected_interval(self, timeframe: str) -> pd.Timedelta:
        """获取预期时间间隔"""
        intervals = {
            '1m': pd.Timedelta(minutes=1),
            '5m': pd.Timedelta(minutes=5),
            '15m': pd.Timedelta(minutes=15),
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            '1d': pd.Timedelta(days=1)
        }
        return intervals.get(timeframe, pd.Timedelta(minutes=5))
    
    def _repair_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据修复"""
        if df.empty:
            return df
        
        df_repaired = df.copy()
        
        # 1. 填充缺失值
        df_repaired = df_repaired.ffill().bfill()
        
        # 2. 修复OHLC关系
        df_repaired['high'] = df_repaired[['high', 'open', 'close']].max(axis=1)
        df_repaired['low'] = df_repaired[['low', 'open', 'close']].min(axis=1)
        
        # 3. 处理异常值 - 使用rolling median
        for col in ['open', 'high', 'low', 'close']:
            if col in df_repaired.columns:
                rolling_median = df_repaired[col].rolling(window=20, center=True).median()
                rolling_std = df_repaired[col].rolling(window=20, center=True).std()
                
                # 识别异常值 (超过3倍标准差)
                outliers = np.abs(df_repaired[col] - rolling_median) > 3 * rolling_std
                df_repaired.loc[outliers, col] = rolling_median.loc[outliers]
        
        # 4. 确保正值
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df_repaired.columns:
                df_repaired[col] = df_repaired[col].abs()
        
        return df_repaired
    
    def _schedule_tasks(self):
        """安排定时任务"""
        # 数据更新任务
        schedule.every(self.config.get('update_interval_minutes', 30)).minutes.do(
            lambda: asyncio.create_task(self._scheduled_data_update())
        )
        
        # 质量检查任务
        schedule.every(self.config.get('quality_check_interval_minutes', 60)).minutes.do(
            lambda: asyncio.create_task(self._scheduled_quality_check())
        )
        
        # 每日完整性检查
        schedule.every().day.at("02:00").do(
            lambda: asyncio.create_task(self.comprehensive_quality_assessment())
        )
    
    async def _scheduled_data_update(self):
        """定时数据更新"""
        self.logger.info("执行定时数据更新...")
        
        # 增量更新最近24小时数据
        await self.expand_data_collection(days=1)
        
        self.performance_metrics['last_update_time'] = datetime.now(timezone.utc).isoformat()
    
    async def _scheduled_quality_check(self):
        """定时质量检查"""
        self.logger.info("执行定时质量检查...")
        
        issues_detected = 0
        issues_fixed = 0
        
        for symbol in self.top30_symbols:
            for timeframe in self.timeframes:
                df = self._load_existing_data(symbol, timeframe)
                if df.empty:
                    continue
                
                # 检查数据gaps
                gaps = self._detect_data_gaps(df, timeframe)
                if gaps:
                    self.logger.warning(f"{symbol} {timeframe} 发现 {len(gaps)} 个数据缺口")
                    issues_detected += len(gaps)
                    
                    # 尝试修复gaps
                    if self.config.get('auto_repair', True):
                        fixed_gaps = await self._fix_data_gaps(symbol, timeframe, gaps)
                        issues_fixed += fixed_gaps
        
        self.performance_metrics['gaps_detected'] = issues_detected
        self.performance_metrics['gaps_fixed'] = issues_fixed
        
        self.logger.info(f"质量检查完成: 发现 {issues_detected} 个问题, 修复 {issues_fixed} 个")
    
    def _detect_data_gaps(self, df: pd.DataFrame, timeframe: str) -> List[Tuple[datetime, datetime]]:
        """检测数据缺口"""
        if len(df) < 2:
            return []
        
        expected_interval = self._get_expected_interval(timeframe)
        time_diffs = df.index.to_series().diff()
        
        # 查找大于预期间隔2倍的gaps
        large_gaps = time_diffs > expected_interval * 2
        
        gaps = []
        for i, is_gap in enumerate(large_gaps):
            if is_gap:
                gap_start = df.index[i-1]
                gap_end = df.index[i]
                gaps.append((gap_start, gap_end))
        
        return gaps
    
    async def _fix_data_gaps(self, symbol: str, timeframe: str, gaps: List[Tuple[datetime, datetime]]) -> int:
        """修复数据缺口"""
        fixed_count = 0
        
        for gap_start, gap_end in gaps:
            try:
                # 下载缺失时间段的数据
                exchange = self.exchanges.get('binance')
                if not exchange:
                    continue
                
                since = int(gap_start.timestamp() * 1000)
                until = int(gap_end.timestamp() * 1000)
                
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                
                if ohlcv:
                    # 创建DataFrame
                    gap_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    gap_df['timestamp'] = pd.to_datetime(gap_df['timestamp'], unit='ms', utc=True)
                    gap_df.set_index('timestamp', inplace=True)
                    
                    # 过滤到gap时间范围
                    gap_df = gap_df[(gap_df.index > gap_start) & (gap_df.index < gap_end)]
                    
                    if not gap_df.empty:
                        # 加载现有数据
                        existing_df = self._load_existing_data(symbol, timeframe)
                        
                        # 合并数据
                        combined_df = pd.concat([existing_df, gap_df]).drop_duplicates().sort_index()
                        
                        # 保存修复后的数据
                        self._save_data(combined_df, symbol, timeframe)
                        
                        fixed_count += 1
                        self.logger.info(f"修复 {symbol} {timeframe} gap: {gap_start} - {gap_end}")
                
                await asyncio.sleep(0.1)  # 避免频率限制
                
            except Exception as e:
                self.logger.error(f"修复gap失败 {symbol} {timeframe}: {e}")
        
        return fixed_count
    
    async def comprehensive_quality_assessment(self):
        """全面质量评估"""
        self.logger.info("执行全面数据质量评估...")
        
        quality_report = {
            'assessment_time': datetime.now(timezone.utc).isoformat(),
            'total_symbols': len(self.top30_symbols),
            'total_timeframes': len(self.timeframes),
            'quality_by_symbol': {},
            'quality_by_timeframe': {},
            'overall_metrics': {
                'avg_completeness': 0.0,
                'avg_consistency': 0.0,
                'avg_accuracy': 0.0,
                'overall_score': 0.0
            },
            'data_tiers': defaultdict(list),
            'issues': []
        }
        
        all_quality_scores = []
        
        for symbol in self.top30_symbols:
            symbol_qualities = []
            
            for timeframe in self.timeframes:
                df = self._load_existing_data(symbol, timeframe)
                
                if df.empty:
                    quality = {'completeness': 0.0, 'consistency': 0.0, 'accuracy': 0.0, 'overall_score': 0.0}
                    quality_report['issues'].append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'issue': 'No data available',
                        'severity': 'high'
                    })
                else:
                    quality = self._assess_data_quality(df, symbol, timeframe)
                
                symbol_qualities.append(quality['overall_score'])
                all_quality_scores.append(quality['overall_score'])
                
                # 分配数据等级
                score = quality['overall_score']
                if score >= 0.999:
                    tier = DataTier.TIER_S
                elif score >= 0.995:
                    tier = DataTier.TIER_A
                elif score >= 0.990:
                    tier = DataTier.TIER_B
                elif score >= 0.950:
                    tier = DataTier.TIER_C
                else:
                    tier = DataTier.TIER_D
                
                quality_report['data_tiers'][tier.value].append(f"{symbol}_{timeframe}")
            
            # 计算币种平均质量
            if symbol_qualities:
                quality_report['quality_by_symbol'][symbol] = {
                    'avg_score': np.mean(symbol_qualities),
                    'min_score': np.min(symbol_qualities),
                    'max_score': np.max(symbol_qualities)
                }
        
        # 计算时间框架平均质量
        for timeframe in self.timeframes:
            timeframe_qualities = []
            for symbol in self.top30_symbols:
                df = self._load_existing_data(symbol, timeframe)
                if not df.empty:
                    quality = self._assess_data_quality(df, symbol, timeframe)
                    timeframe_qualities.append(quality['overall_score'])
            
            if timeframe_qualities:
                quality_report['quality_by_timeframe'][timeframe] = {
                    'avg_score': np.mean(timeframe_qualities),
                    'symbol_count': len(timeframe_qualities)
                }
        
        # 计算总体指标
        if all_quality_scores:
            quality_report['overall_metrics']['overall_score'] = np.mean(all_quality_scores)
        
        # 更新性能指标
        self.performance_metrics['data_quality_score'] = quality_report['overall_metrics']['overall_score']
        
        # 保存质量报告
        report_path = Path("data/quality_assessment_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, default=str)
        
        self.logger.info(f"质量评估完成: 总体评分 {quality_report['overall_metrics']['overall_score']:.3f}")
        
        return quality_report
    
    async def _main_optimization_loop(self):
        """主优化循环"""
        self.logger.info("启动主优化循环...")
        
        while not self.stop_event.is_set():
            try:
                # 执行定时任务
                schedule.run_pending()
                
                # 处理更新队列
                await self._process_update_queue()
                
                # 监控性能
                self._monitor_performance()
                
                # 清理缓存
                self._cleanup_cache()
                
                # 等待下一次循环
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except Exception as e:
                self.logger.error(f"主循环错误: {e}")
                await asyncio.sleep(60)
    
    async def _process_update_queue(self):
        """处理更新队列"""
        with self.task_lock:
            if not self.update_queue:
                return
            
            # 获取下一个任务
            task = self.update_queue.popleft()
        
        if task.status != 'pending':
            return
        
        try:
            task.status = 'running'
            
            # 执行更新
            await self._download_historical_data(task.symbol, task.timeframe, days=7)
            
            task.status = 'completed'
            task.next_update = datetime.now(timezone.utc) + timedelta(hours=1)
            
        except Exception as e:
            task.status = 'failed'
            task.retry_count += 1
            task.error_message = str(e)
            
            if task.retry_count < 3:
                task.status = 'pending'
                task.next_update = datetime.now(timezone.utc) + timedelta(minutes=30)
                
                with self.task_lock:
                    self.update_queue.append(task)
    
    def _monitor_performance(self):
        """监控性能"""
        # 检查磁盘空间
        disk_usage = shutil.disk_usage(self.base_path)
        free_gb = disk_usage.free / (1024**3)
        
        if free_gb < 10:  # 少于10GB
            self.logger.warning(f"磁盘空间不足: {free_gb:.1f}GB")
        
        # 检查内存使用
        if self.redis_available:
            try:
                memory_info = self.redis_client.info('memory')
                used_memory_mb = memory_info.get('used_memory', 0) / 1024 / 1024
                self.logger.debug(f"Redis内存使用: {used_memory_mb:.1f}MB")
            except:
                pass
    
    def _cleanup_cache(self):
        """清理缓存"""
        # 清理过期的临时文件
        temp_pattern = self.base_path / "*.tmp"
        for temp_file in self.base_path.glob("*.tmp"):
            if temp_file.stat().st_mtime < time.time() - 3600:  # 1小时前的临时文件
                try:
                    temp_file.unlink()
                except:
                    pass
    
    def generate_infrastructure_report(self) -> Dict[str, Any]:
        """生成基础设施报告"""
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'infrastructure_status': 'operational',
            'symbols': {
                'configured': len(self.top30_symbols),
                'active': len([s for s in self.top30_symbols if (self.base_path / f"{s}_5m_2years.parquet").exists()])
            },
            'timeframes': {
                'configured': len(self.timeframes),
                'coverage': self.timeframes
            },
            'data_coverage': {
                'total_files': len(list(self.base_path.glob("*.parquet"))),
                'total_size_gb': sum(f.stat().st_size for f in self.base_path.glob("*.parquet")) / (1024**3)
            },
            'quality_metrics': self.performance_metrics,
            'optimization_config': {
                'update_interval_minutes': self.config.get('update_interval_minutes', 30),
                'quality_threshold': self.config.get('quality_threshold', 0.995),
                'auto_repair': self.config.get('auto_repair', True)
            }
        }
        
        # 保存报告
        report_path = Path("data/infrastructure_optimization_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def stop_optimization(self):
        """停止优化服务"""
        self.logger.info("停止持续优化服务...")
        self.stop_event.set()
        
        # 关闭交易所连接
        for exchange in self.exchanges.values():
            try:
                if hasattr(exchange, 'close'):
                    asyncio.create_task(exchange.close())
            except:
                pass
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """获取优化状态"""
        return {
            'status': 'running' if not self.stop_event.is_set() else 'stopped',
            'performance_metrics': self.performance_metrics,
            'queue_size': len(self.update_queue),
            'quality_issues': len(self.quality_issues),
            'last_check': datetime.now(timezone.utc).isoformat()
        }

# 主函数
async def main():
    """主函数"""
    optimizer = ContinuousDataInfrastructureOptimizer()
    
    try:
        await optimizer.start_continuous_optimization()
    except KeyboardInterrupt:
        optimizer.logger.info("收到中断信号，正在停止...")
        optimizer.stop_optimization()
    except Exception as e:
        optimizer.logger.error(f"服务异常: {e}")
    finally:
        optimizer.logger.info("持续优化服务已停止")

if __name__ == "__main__":
    asyncio.run(main())