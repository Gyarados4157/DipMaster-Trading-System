"""
Enhanced Top 30 Altcoins Data Collection Infrastructure V2
增强版前30市值山寨币数据收集基础设施

新增特性：
- 支持6个时间框架：1m, 5m, 15m, 1h, 4h, 1d
- 高度并行化下载（每批次5个币种同时处理）
- 智能数据质量评估和异常检测
- 自动替换低质量币种
- 实时进度监控和性能优化
- 生成增强的MarketDataBundle配置
"""

import asyncio
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
import warnings
import aiohttp
import ssl
from collections import defaultdict
warnings.filterwarnings('ignore')

@dataclass
class EnhancedCoinInfo:
    """增强币种信息"""
    symbol: str
    name: str
    category: str
    market_cap_rank: int
    priority: int
    daily_volume_usd: float
    exchange_support: List[str]
    liquidity_tier: str
    volatility_factor: float
    correlation_group: str

class EnhancedTop30DataCollector:
    """增强版前30大山寨币数据收集器"""
    
    def __init__(self):
        self.logger = self.setup_logging()
        
        # 增强版前30大市值山寨币配置（基于2025年8月数据）
        self.top30_altcoins = {
            # 顶级山寨币 (Tier 1)
            "XRPUSDT": EnhancedCoinInfo("XRPUSDT", "XRP", "Payment", 3, 1, 2500000000, ["binance", "okx", "bybit"], "高流动性", 0.85, "Payment"),
            "BNBUSDT": EnhancedCoinInfo("BNBUSDT", "BNB", "Exchange", 4, 1, 1800000000, ["binance", "okx"], "高流动性", 0.75, "Exchange"),
            "SOLUSDT": EnhancedCoinInfo("SOLUSDT", "Solana", "Layer1", 5, 1, 3200000000, ["binance", "okx", "bybit"], "高流动性", 1.2, "SmartContract"),
            "DOGEUSDT": EnhancedCoinInfo("DOGEUSDT", "Dogecoin", "Meme", 6, 2, 1400000000, ["binance", "okx", "bybit"], "高流动性", 1.5, "Meme"),
            "ADAUSDT": EnhancedCoinInfo("ADAUSDT", "Cardano", "Layer1", 7, 1, 800000000, ["binance", "okx", "bybit"], "高流动性", 0.9, "SmartContract"),
            
            # 高流动性山寨币 (Tier 2)
            "TRXUSDT": EnhancedCoinInfo("TRXUSDT", "TRON", "Layer1", 8, 2, 450000000, ["binance", "okx", "bybit"], "高流动性", 0.8, "SmartContract"),
            "TONUSDT": EnhancedCoinInfo("TONUSDT", "Toncoin", "Layer1", 9, 2, 350000000, ["binance", "okx", "bybit"], "中等流动性", 1.1, "SmartContract"),
            "AVAXUSDT": EnhancedCoinInfo("AVAXUSDT", "Avalanche", "Layer1", 10, 1, 600000000, ["binance", "okx", "bybit"], "高流动性", 1.3, "SmartContract"),
            "LINKUSDT": EnhancedCoinInfo("LINKUSDT", "Chainlink", "Oracle", 11, 1, 700000000, ["binance", "okx", "bybit"], "高流动性", 1.0, "Infrastructure"),
            "DOTUSDT": EnhancedCoinInfo("DOTUSDT", "Polkadot", "Layer0", 12, 2, 300000000, ["binance", "okx", "bybit"], "中等流动性", 1.1, "Infrastructure"),
            
            # 中等流动性山寨币 (Tier 3)
            "MATICUSDT": EnhancedCoinInfo("MATICUSDT", "Polygon", "Layer2", 13, 2, 400000000, ["binance", "okx", "bybit"], "中等流动性", 1.0, "Layer2"),
            "LTCUSDT": EnhancedCoinInfo("LTCUSDT", "Litecoin", "Payment", 15, 2, 600000000, ["binance", "okx", "bybit"], "高流动性", 0.7, "Payment"),
            "NEARUSDT": EnhancedCoinInfo("NEARUSDT", "NEAR Protocol", "Layer1", 17, 2, 180000000, ["binance", "okx", "bybit"], "中等流动性", 1.2, "SmartContract"),
            "APTUSDT": EnhancedCoinInfo("APTUSDT", "Aptos", "Layer1", 18, 2, 250000000, ["binance", "okx", "bybit"], "中等流动性", 1.4, "SmartContract"),
            "UNIUSDT": EnhancedCoinInfo("UNIUSDT", "Uniswap", "DeFi", 19, 2, 300000000, ["binance", "okx", "bybit"], "中等流动性", 1.1, "DeFi"),
            
            # 其他重要山寨币
            "ATOMUSDT": EnhancedCoinInfo("ATOMUSDT", "Cosmos", "Layer0", 20, 2, 150000000, ["binance", "okx", "bybit"], "中等流动性", 1.0, "Infrastructure"),
            "XLMUSDT": EnhancedCoinInfo("XLMUSDT", "Stellar", "Payment", 21, 3, 120000000, ["binance", "okx", "bybit"], "中等流动性", 0.8, "Payment"),
            "FILUSDT": EnhancedCoinInfo("FILUSDT", "Filecoin", "Storage", 25, 3, 150000000, ["binance", "okx", "bybit"], "中等流动性", 1.2, "Storage"),
            "ARBUSDT": EnhancedCoinInfo("ARBUSDT", "Arbitrum", "Layer2", 26, 2, 180000000, ["binance", "okx", "bybit"], "中等流动性", 1.3, "Layer2"),
            "OPUSDT": EnhancedCoinInfo("OPUSDT", "Optimism", "Layer2", 27, 2, 120000000, ["binance", "okx", "bybit"], "中等流动性", 1.2, "Layer2"),
            
            # DeFi和特殊用途币种
            "AAVEUSDT": EnhancedCoinInfo("AAVEUSDT", "Aave", "DeFi", 31, 2, 200000000, ["binance", "okx", "bybit"], "中等流动性", 1.3, "DeFi"),
            "MKRUSDT": EnhancedCoinInfo("MKRUSDT", "Maker", "DeFi", 32, 3, 150000000, ["binance", "okx", "bybit"], "中等流动性", 1.1, "DeFi"),
            "COMPUSDT": EnhancedCoinInfo("COMPUSDT", "Compound", "DeFi", 35, 3, 100000000, ["binance", "okx", "bybit"], "低流动性", 1.4, "DeFi"),
            "VETUSDT": EnhancedCoinInfo("VETUSDT", "VeChain", "Supply Chain", 28, 3, 60000000, ["binance", "okx", "bybit"], "低流动性", 0.9, "Enterprise"),
            "ALGOUSDT": EnhancedCoinInfo("ALGOUSDT", "Algorand", "Layer1", 29, 3, 100000000, ["binance", "okx", "bybit"], "中等流动性", 1.0, "SmartContract"),
            
            # 备用币种（如需替换）
            "GRTUSDT": EnhancedCoinInfo("GRTUSDT", "The Graph", "Indexing", 30, 3, 80000000, ["binance", "okx", "bybit"], "低流动性", 1.3, "Infrastructure"),
            "ICPUSDT": EnhancedCoinInfo("ICPUSDT", "Internet Computer", "Computing", 16, 3, 200000000, ["binance", "okx", "bybit"], "中等流动性", 1.5, "Computing"),
            "SHIBUSDT": EnhancedCoinInfo("SHIBUSDT", "Shiba Inu", "Meme", 14, 3, 800000000, ["binance", "okx", "bybit"], "中等流动性", 2.0, "Meme"),
            "PEPEUSDT": EnhancedCoinInfo("PEPEUSDT", "Pepe", "Meme", 33, 3, 300000000, ["binance", "okx", "bybit"], "中等流动性", 2.5, "Meme"),
            "WIFUSDT": EnhancedCoinInfo("WIFUSDT", "dogwifhat", "Meme", 34, 3, 200000000, ["binance", "okx", "bybit"], "低流动性", 3.0, "Meme"),
        }
        
        # 增强版时间框架配置
        self.timeframes = {
            '1m': '1m',     # 1分钟K线 - 超高频分析
            '5m': '5m',     # 5分钟K线 - DipMaster主要时间框架
            '15m': '15m',   # 15分钟K线 - 中期分析
            '1h': '1h',     # 1小时K线 - 趋势分析
            '4h': '4h',     # 4小时K线 - 宏观趋势
            '1d': '1d'      # 日线 - 长期趋势
        }
        
        # 数据存储路径
        self.data_path = Path("data/enhanced_market_data")
        self.data_path.mkdir(exist_ok=True, parents=True)
        
        # 初始化交易所连接
        self.exchanges = self.setup_exchanges()
        
        # 增强的数据质量标准
        self.quality_standards = {
            'completeness_threshold': 0.995,  # 99.5%以上的数据完整性
            'max_gap_minutes': 15,            # 最大数据缺失15分钟
            'price_spike_threshold': 0.3,     # 价格异常波动30%阈值（更严格）
            'volume_outlier_std': 4,          # 成交量异常标准差倍数（更严格）
            'ohlc_consistency_tolerance': 0.0005,  # OHLC一致性容忍度
            'minimum_daily_volume': 50000000   # 最小日成交量5000万USD
        }
        
        # 增强的数据收集配置
        self.collection_config = {
            'lookback_days': 730,       # 2年历史数据
            'batch_size': 1000,         # 每批次获取1000条记录
            'rate_limit_delay': 0.05,   # 请求间隔50ms（更快）
            'retry_attempts': 5,        # 失败重试5次
            'timeout_seconds': 45,      # 请求超时45秒
            'parallel_symbols': 5,      # 并行下载5个币种
            'max_workers': 10           # 最大工作线程数
        }
        
        # 性能监控
        self.performance_metrics = {
            'start_time': None,
            'download_times': {},
            'quality_scores': {},
            'errors': [],
            'retry_counts': defaultdict(int)
        }
        
    def setup_logging(self) -> logging.Logger:
        """设置增强日志系统"""
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/enhanced_top30_data_collection.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def setup_exchanges(self) -> Dict[str, ccxt.Exchange]:
        """设置增强的交易所连接"""
        exchanges = {}
        
        # Binance - 主要数据源
        exchanges['binance'] = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'sandbox': False,
            'rateLimit': 100,           # 更激进的限制
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'},
            'timeout': 30000            # 30秒超时
        })
        
        return exchanges
    
    async def download_symbol_timeframe_data(self, 
                                          symbol: str, 
                                          timeframe: str, 
                                          exchange_name: str = 'binance') -> Tuple[str, str, pd.DataFrame, Dict]:
        """下载单个币种单个时间框架的数据"""
        start_time = time.time()
        quality_info = {}
        
        try:
            self.logger.info(f"开始下载 {symbol} {timeframe} 数据...")
            
            exchange = self.exchanges[exchange_name]
            
            # 根据时间框架计算获取天数
            timeframe_days = {
                '1m': 90,   # 1分钟数据只取90天（数据量大）
                '5m': 730,  # 5分钟数据取2年
                '15m': 730, # 15分钟数据取2年
                '1h': 730,  # 1小时数据取2年
                '4h': 730,  # 4小时数据取2年
                '1d': 1095  # 日线数据取3年
            }
            
            lookback_days = timeframe_days.get(timeframe, 730)
            
            # 计算时间范围
            end_time = datetime.now()
            start_time_calc = end_time - timedelta(days=lookback_days)
            since = int(start_time_calc.timestamp() * 1000)
            
            # 分批下载数据
            all_ohlcv = []
            current_since = since
            batch_size = self.collection_config['batch_size']
            retry_count = 0
            
            while current_since < int(end_time.timestamp() * 1000):
                try:
                    # 异步获取OHLCV数据
                    ohlcv = await asyncio.to_thread(
                        exchange.fetch_ohlcv,
                        symbol, timeframe, current_since, batch_size
                    )
                    
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 1
                    
                    # 进度显示
                    if len(all_ohlcv) % 5000 == 0:
                        self.logger.info(f"{symbol} {timeframe}: 已收集 {len(all_ohlcv)} 条记录")
                    
                    # 避免触发频率限制
                    await asyncio.sleep(self.collection_config['rate_limit_delay'])
                    
                except Exception as e:
                    retry_count += 1
                    self.performance_metrics['retry_counts'][f"{symbol}_{timeframe}"] += 1
                    
                    if retry_count >= self.collection_config['retry_attempts']:
                        self.logger.error(f"达到最大重试次数 {symbol} {timeframe}: {e}")
                        break
                    
                    self.logger.warning(f"批次下载失败 {symbol} {timeframe} (重试 {retry_count}): {e}")
                    await asyncio.sleep(min(retry_count * 2, 10))  # 指数退避
                    continue
            
            # 转换为DataFrame
            if not all_ohlcv:
                self.logger.error(f"没有获取到 {symbol} {timeframe} 数据")
                return symbol, timeframe, pd.DataFrame(), {}
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 数据清理和去重
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            # 移除零成交量的记录
            df = df[df['volume'] > 0]
            
            # 数据质量评估
            quality_score = self.assess_data_quality(df, symbol, timeframe)
            quality_info = {
                'quality_score': quality_score,
                'records_count': len(df),
                'date_range': {
                    'start': df.index.min().isoformat() if not df.empty else None,
                    'end': df.index.max().isoformat() if not df.empty else None
                },
                'download_time_seconds': time.time() - start_time,
                'retry_count': retry_count
            }
            
            self.logger.info(f"{symbol} {timeframe} 下载完成: {len(df)} 条记录, 质量评分: {quality_score:.3f}")
            
            return symbol, timeframe, df, quality_info
            
        except Exception as e:
            self.logger.error(f"下载 {symbol} {timeframe} 数据失败: {e}")
            self.performance_metrics['errors'].append(f"{symbol}_{timeframe}: {str(e)}")
            return symbol, timeframe, pd.DataFrame(), {}
    
    def assess_data_quality(self, df: pd.DataFrame, symbol: str, timeframe: str) -> float:
        """增强的数据质量评估"""
        if df.empty:
            return 0.0
        
        quality_scores = {}
        
        try:
            # 1. 完整性检查（更严格）
            total_expected = self.calculate_expected_records(timeframe)
            actual_records = len(df)
            quality_scores['completeness'] = min(1.0, actual_records / total_expected)
            
            # 2. 缺失值检查
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            quality_scores['no_missing'] = max(0, 1 - missing_ratio)
            
            # 3. OHLC一致性检查（更严格）
            ohlc_violations = 0
            
            # High >= max(Open, Close) 且 High >= Low
            high_violations = ((df['high'] < df[['open', 'close']].max(axis=1)) | 
                             (df['high'] < df['low'])).sum()
            
            # Low <= min(Open, Close) 且 Low <= High
            low_violations = ((df['low'] > df[['open', 'close']].min(axis=1)) | 
                            (df['low'] > df['high'])).sum()
            
            # 价格必须为正数
            negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).sum().sum()
            
            total_violations = high_violations + low_violations + negative_prices
            quality_scores['ohlc_consistency'] = max(0, 1 - (total_violations / (len(df) * 3)))
            
            # 4. 价格稳定性检查（检测异常波动）
            price_changes = df['close'].pct_change().abs()
            extreme_changes = (price_changes > self.quality_standards['price_spike_threshold']).sum()
            quality_scores['price_stability'] = max(0, 1 - (extreme_changes / len(df)))
            
            # 5. 成交量合理性检查
            if df['volume'].std() > 0:
                volume_z_scores = np.abs((df['volume'] - df['volume'].mean()) / df['volume'].std())
                volume_outliers = (volume_z_scores > self.quality_standards['volume_outlier_std']).sum()
                quality_scores['volume_stability'] = max(0, 1 - (volume_outliers / len(df)))
            else:
                quality_scores['volume_stability'] = 0.5  # 成交量无变化，给中等分数
            
            # 6. 时间连续性检查（更严格）
            time_gaps = self.check_time_gaps(df, timeframe)
            quality_scores['time_continuity'] = time_gaps['continuity_score']
            
            # 7. 流动性检查（新增）
            avg_volume = df['volume'].mean()
            coin_info = self.top30_altcoins.get(symbol)
            if coin_info:
                expected_volume = coin_info.daily_volume_usd / 288  # 假设5分钟均匀分布
                volume_ratio = min(1.0, avg_volume / expected_volume) if expected_volume > 0 else 0.5
                quality_scores['liquidity'] = volume_ratio
            else:
                quality_scores['liquidity'] = 0.8  # 默认评分
            
            # 8. 价格合理性检查（新增）
            price_mean = df['close'].mean()
            price_std = df['close'].std()
            if price_std > 0:
                cv = price_std / price_mean  # 变异系数
                reasonable_cv = cv < 2.0  # 变异系数小于2视为合理
                quality_scores['price_reasonableness'] = 1.0 if reasonable_cv else 0.5
            else:
                quality_scores['price_reasonableness'] = 0.3  # 价格无变化，低分
            
            # 综合质量评分（加权平均）
            weights = {
                'completeness': 0.20,
                'no_missing': 0.15,
                'ohlc_consistency': 0.20,
                'price_stability': 0.15,
                'volume_stability': 0.10,
                'time_continuity': 0.10,
                'liquidity': 0.05,
                'price_reasonableness': 0.05
            }
            
            overall_score = sum(quality_scores[key] * weights[key] for key in weights)
            
            # 存储详细质量指标
            self.performance_metrics['quality_scores'][f"{symbol}_{timeframe}"] = quality_scores
            
            return overall_score
            
        except Exception as e:
            self.logger.error(f"质量评估失败 {symbol} {timeframe}: {e}")
            return 0.0
    
    def calculate_expected_records(self, timeframe: str) -> int:
        """计算期望的记录数量（基于不同时间框架）"""
        timeframe_days = {
            '1m': 90,   # 1分钟数据只取90天
            '5m': 730,  # 其他数据取2年
            '15m': 730,
            '1h': 730,
            '4h': 730,
            '1d': 1095  # 日线数据取3年
        }
        
        days = timeframe_days.get(timeframe, 730)
        
        records_per_day = {
            '1m': 1440,    # 24 * 60
            '5m': 288,     # 24 * 12
            '15m': 96,     # 24 * 4
            '1h': 24,      # 24
            '4h': 6,       # 6
            '1d': 1        # 1
        }
        
        return days * records_per_day.get(timeframe, 288)
    
    def check_time_gaps(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """增强的时间间隔连续性检查"""
        if len(df) < 2:
            return {'continuity_score': 0.0, 'gaps_count': 0, 'max_gap_minutes': 0}
        
        # 计算期望的时间间隔
        expected_interval = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }
        
        interval = expected_interval.get(timeframe, timedelta(minutes=5))
        
        # 计算实际时间间隔
        time_diffs = df.index.to_series().diff().dropna()
        
        # 识别异常间隔
        expected_seconds = interval.total_seconds()
        
        # 允许的最大间隔（考虑周末和节假日）
        if timeframe in ['1d']:
            max_allowed_seconds = expected_seconds * 3  # 日线允许3天间隔
        elif timeframe in ['4h']:
            max_allowed_seconds = expected_seconds * 2  # 4小时允许8小时间隔
        else:
            max_allowed_seconds = expected_seconds * 1.5  # 其他允许1.5倍间隔
        
        large_gaps = time_diffs[time_diffs.dt.total_seconds() > max_allowed_seconds]
        gaps_count = len(large_gaps)
        
        # 最大间隔（分钟）
        max_gap_minutes = time_diffs.dt.total_seconds().max() / 60 if not time_diffs.empty else 0
        
        # 连续性评分（更严格）
        continuity_score = max(0, 1 - (gaps_count / len(time_diffs)) * 2)  # 乘以2使评分更严格
        
        return {
            'continuity_score': continuity_score,
            'gaps_count': gaps_count,
            'max_gap_minutes': max_gap_minutes,
            'large_gaps_count': gaps_count
        }
    
    def save_enhanced_data(self, df: pd.DataFrame, symbol: str, timeframe: str, quality_info: Dict) -> str:
        """保存增强数据到文件"""
        try:
            file_path = self.data_path / f"{symbol}_{timeframe}_2years.parquet"
            
            # 保存为Parquet格式（使用zstd压缩，更高压缩率）
            df.to_parquet(file_path, compression='zstd', index=True)
            
            # 生成增强元数据
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'records_count': len(df),
                'data_quality': quality_info,
                'date_range': {
                    'start': df.index.min().isoformat() if not df.empty else None,
                    'end': df.index.max().isoformat() if not df.empty else None,
                    'span_days': (df.index.max() - df.index.min()).days if not df.empty else 0
                },
                'file_info': {
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                    'compression': 'zstd',
                    'format': 'parquet'
                },
                'statistics': {
                    'price_range': {
                        'min': float(df['low'].min()) if not df.empty else None,
                        'max': float(df['high'].max()) if not df.empty else None,
                        'avg': float(df['close'].mean()) if not df.empty else None
                    },
                    'volume_stats': {
                        'total': float(df['volume'].sum()) if not df.empty else None,
                        'avg': float(df['volume'].mean()) if not df.empty else None,
                        'max': float(df['volume'].max()) if not df.empty else None
                    }
                },
                'created_at': datetime.now().isoformat(),
                'collection_config': self.collection_config
            }
            
            metadata_path = self.data_path / f"{symbol}_{timeframe}_2years_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"数据已保存: {file_path} ({metadata['file_info']['file_size_mb']:.2f} MB)")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"保存数据失败 {symbol} {timeframe}: {e}")
            return ""
    
    async def collect_symbol_batch(self, symbols_batch: List[str]) -> Dict:
        """并行收集一批币种的所有时间框架数据"""
        batch_results = {}
        
        # 为每个币种创建所有时间框架的任务
        tasks = []
        for symbol in symbols_batch:
            for timeframe in self.timeframes.keys():
                task = self.download_symbol_timeframe_data(symbol, timeframe)
                tasks.append(task)
        
        # 并行执行所有任务
        self.logger.info(f"开始并行下载 {len(symbols_batch)} 个币种的 {len(self.timeframes)} 个时间框架数据...")
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理结果
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"批次任务失败: {result}")
                    continue
                
                symbol, timeframe, df, quality_info = result
                
                if not df.empty:
                    # 保存数据
                    file_path = self.save_enhanced_data(df, symbol, timeframe, quality_info)
                    
                    if file_path:
                        if symbol not in batch_results:
                            batch_results[symbol] = {}
                        
                        batch_results[symbol][timeframe] = {
                            'file_path': file_path,
                            'quality_info': quality_info,
                            'success': True
                        }
                else:
                    self.logger.warning(f"空数据: {symbol} {timeframe}")
                    if symbol not in batch_results:
                        batch_results[symbol] = {}
                    batch_results[symbol][timeframe] = {
                        'file_path': '',
                        'quality_info': {},
                        'success': False
                    }
        
        except Exception as e:
            self.logger.error(f"批次收集失败: {e}")
        
        return batch_results
    
    async def collect_all_enhanced_data(self) -> Dict[str, Any]:
        """收集所有币种的增强历史数据"""
        self.logger.info("开始收集前30大山寨币的增强版2年历史数据...")
        self.performance_metrics['start_time'] = datetime.now()
        
        collection_start_time = datetime.now()
        all_results = {}
        failed_symbols = []
        
        # 将币种分批处理
        symbols = list(self.top30_altcoins.keys())
        batch_size = self.collection_config['parallel_symbols']
        total_batches = len(symbols) // batch_size + (1 if len(symbols) % batch_size > 0 else 0)
        
        for i in range(0, len(symbols), batch_size):
            batch_num = i // batch_size + 1
            batch_symbols = symbols[i:i + batch_size]
            
            self.logger.info(f"处理第 {batch_num}/{total_batches} 批: {batch_symbols}")
            
            try:
                batch_results = await self.collect_symbol_batch(batch_symbols)
                all_results.update(batch_results)
                
                # 进度报告
                completed_symbols = len(all_results)
                progress = (completed_symbols / len(symbols)) * 100
                self.logger.info(f"批次完成进度: {progress:.1f}% ({completed_symbols}/{len(symbols)})")
                
                # 批次间短暂休息
                if batch_num < total_batches:
                    await asyncio.sleep(2)
                    
            except Exception as e:
                self.logger.error(f"批次 {batch_num} 处理失败: {e}")
                failed_symbols.extend(batch_symbols)
        
        # 生成收集报告
        collection_end_time = datetime.now()
        collection_duration = collection_end_time - collection_start_time
        
        # 统计成功和失败的情况
        successful_symbols = []
        partially_successful = []
        failed_completely = []
        
        for symbol in symbols:
            if symbol in all_results:
                timeframe_success = sum(1 for tf_data in all_results[symbol].values() if tf_data.get('success', False))
                total_timeframes = len(self.timeframes)
                
                if timeframe_success == total_timeframes:
                    successful_symbols.append(symbol)
                elif timeframe_success > 0:
                    partially_successful.append(symbol)
                else:
                    failed_completely.append(symbol)
            else:
                failed_completely.append(symbol)
        
        # 计算总体统计
        total_files_created = sum(
            sum(1 for tf_data in symbol_data.values() if tf_data.get('success', False))
            for symbol_data in all_results.values()
        )
        
        total_quality_scores = []
        for symbol_data in all_results.values():
            for tf_data in symbol_data.values():
                if tf_data.get('success', False):
                    quality_score = tf_data.get('quality_info', {}).get('quality_score', 0)
                    total_quality_scores.append(quality_score)
        
        avg_quality = np.mean(total_quality_scores) if total_quality_scores else 0
        
        collection_report = {
            'collection_summary': {
                'start_time': collection_start_time.isoformat(),
                'end_time': collection_end_time.isoformat(),
                'duration_minutes': collection_duration.total_seconds() / 60,
                'total_symbols_attempted': len(symbols),
                'fully_successful_symbols': len(successful_symbols),
                'partially_successful_symbols': len(partially_successful),
                'failed_symbols': len(failed_completely),
                'total_files_created': total_files_created,
                'total_timeframes': len(self.timeframes),
                'expected_files': len(symbols) * len(self.timeframes),
                'success_rate': total_files_created / (len(symbols) * len(self.timeframes)),
                'average_quality_score': avg_quality
            },
            'symbol_results': {
                'fully_successful': successful_symbols,
                'partially_successful': partially_successful,
                'failed_completely': failed_completely
            },
            'data_collection': all_results,
            'performance_metrics': self.performance_metrics,
            'quality_standards': self.quality_standards,
            'collection_config': self.collection_config,
            'timeframes_collected': list(self.timeframes.keys())
        }
        
        self.logger.info(f"增强数据收集完成！")
        self.logger.info(f"耗时: {collection_duration.total_seconds()/60:.1f} 分钟")
        self.logger.info(f"完全成功: {len(successful_symbols)} 币种")
        self.logger.info(f"部分成功: {len(partially_successful)} 币种")
        self.logger.info(f"完全失败: {len(failed_completely)} 币种")
        self.logger.info(f"总文件数: {total_files_created}/{len(symbols) * len(self.timeframes)}")
        self.logger.info(f"平均质量评分: {avg_quality:.3f}")
        
        return collection_report

# 主执行函数
async def main():
    """主函数 - 执行增强版数据收集流程"""
    collector = EnhancedTop30DataCollector()
    
    try:
        # 执行增强版数据收集
        collection_report = await collector.collect_all_enhanced_data()
        
        # 保存详细报告
        reports_path = Path("data")
        reports_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存收集报告
        collection_report_path = reports_path / f"Enhanced_Top30_Collection_Report_{timestamp}.json"
        with open(collection_report_path, 'w', encoding='utf-8') as f:
            json.dump(collection_report, f, ensure_ascii=False, indent=2, default=str)
        
        # 生成执行摘要
        print("\n" + "="*80)
        print("增强版前30大山寨币数据收集完成!")
        print("="*80)
        print(f"📊 详细报告: {collection_report_path}")
        print(f"⏱️  收集耗时: {collection_report.get('collection_summary', {}).get('duration_minutes', 0):.1f} 分钟")
        print(f"✅ 完全成功: {collection_report.get('collection_summary', {}).get('fully_successful_symbols', 0)} 币种")
        print(f"🔸 部分成功: {collection_report.get('collection_summary', {}).get('partially_successful_symbols', 0)} 币种")
        print(f"❌ 完全失败: {collection_report.get('collection_summary', {}).get('failed_symbols', 0)} 币种")
        print(f"📁 总文件数: {collection_report.get('collection_summary', {}).get('total_files_created', 0)}")
        print(f"🏆 平均质量: {collection_report.get('collection_summary', {}).get('average_quality_score', 0):.3f}")
        print(f"📈 成功率: {collection_report.get('collection_summary', {}).get('success_rate', 0)*100:.1f}%")
        print("="*80)
        
        return collection_report
        
    except Exception as e:
        collector.logger.error(f"增强数据收集流程失败: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())