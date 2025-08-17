"""
Top 30 Altcoins Data Collection Infrastructure
为DipMaster策略优化收集前30市值山寨币的完整历史数据

支持：
- 前30大市值山寨币（排除BTC、ETH、稳定币）
- 多时间框架：1分钟、5分钟、15分钟、1小时K线数据
- 2年历史数据
- 高质量数据验证和清理
- 自动生成数据质量报告
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
warnings.filterwarnings('ignore')

@dataclass
class CoinInfo:
    """币种信息"""
    symbol: str
    name: str
    category: str
    market_cap_rank: int
    priority: int
    daily_volume_usd: float
    exchange_support: List[str]
    liquidity_tier: str

class Top30AltcoinsDataCollector:
    """前30大山寨币数据收集器"""
    
    def __init__(self):
        self.logger = self.setup_logging()
        
        # 前30大市值山寨币配置（基于2025年8月数据）
        self.top30_altcoins = {
            # Top 10 山寨币
            "XRPUSDT": CoinInfo("XRPUSDT", "XRP", "Payment", 3, 1, 2500000000, ["binance", "okx", "bybit"], "高流动性"),
            "BNBUSDT": CoinInfo("BNBUSDT", "BNB", "Exchange", 4, 1, 1800000000, ["binance", "okx"], "高流动性"),
            "SOLUSDT": CoinInfo("SOLUSDT", "Solana", "Layer1", 5, 1, 3200000000, ["binance", "okx", "bybit"], "高流动性"),
            "DOGEUSDT": CoinInfo("DOGEUSDT", "Dogecoin", "Meme", 6, 2, 1400000000, ["binance", "okx", "bybit"], "高流动性"),
            "ADAUSDT": CoinInfo("ADAUSDT", "Cardano", "Layer1", 7, 1, 800000000, ["binance", "okx", "bybit"], "高流动性"),
            "TRXUSDT": CoinInfo("TRXUSDT", "TRON", "Layer1", 8, 2, 450000000, ["binance", "okx", "bybit"], "中等流动性"),
            "TONUSDT": CoinInfo("TONUSDT", "Toncoin", "Layer1", 9, 2, 350000000, ["binance", "okx", "bybit"], "中等流动性"),
            "AVAXUSDT": CoinInfo("AVAXUSDT", "Avalanche", "Layer1", 10, 1, 600000000, ["binance", "okx", "bybit"], "高流动性"),
            "LINKUSDT": CoinInfo("LINKUSDT", "Chainlink", "Oracle", 11, 1, 700000000, ["binance", "okx", "bybit"], "高流动性"),
            "DOTUSDT": CoinInfo("DOTUSDT", "Polkadot", "Layer0", 12, 2, 300000000, ["binance", "okx", "bybit"], "中等流动性"),
            
            # 11-20名山寨币
            "MATICUSDT": CoinInfo("MATICUSDT", "Polygon", "Layer2", 13, 2, 400000000, ["binance", "okx", "bybit"], "中等流动性"),
            "SHIBUSDT": CoinInfo("SHIBUSDT", "Shiba Inu", "Meme", 14, 3, 800000000, ["binance", "okx", "bybit"], "中等流动性"),
            "LTCUSDT": CoinInfo("LTCUSDT", "Litecoin", "Payment", 15, 2, 600000000, ["binance", "okx", "bybit"], "高流动性"),
            "ICPUSDT": CoinInfo("ICPUSDT", "Internet Computer", "Computing", 16, 3, 200000000, ["binance", "okx", "bybit"], "中等流动性"),
            "NEARUSDT": CoinInfo("NEARUSDT", "NEAR Protocol", "Layer1", 17, 2, 180000000, ["binance", "okx", "bybit"], "中等流动性"),
            "APTUSDT": CoinInfo("APTUSDT", "Aptos", "Layer1", 18, 2, 250000000, ["binance", "okx", "bybit"], "中等流动性"),
            "UNIUSDT": CoinInfo("UNIUSDT", "Uniswap", "DeFi", 19, 2, 300000000, ["binance", "okx", "bybit"], "中等流动性"),
            "ATOMUSDT": CoinInfo("ATOMUSDT", "Cosmos", "Layer0", 20, 2, 150000000, ["binance", "okx", "bybit"], "中等流动性"),
            "XLMUSDT": CoinInfo("XLMUSDT", "Stellar", "Payment", 21, 3, 120000000, ["binance", "okx", "bybit"], "中等流动性"),
            "BCHUSDT": CoinInfo("BCHUSDT", "Bitcoin Cash", "Payment", 22, 3, 400000000, ["binance", "okx", "bybit"], "中等流动性"),
            
            # 21-30名山寨币
            "HBARUSDT": CoinInfo("HBARUSDT", "Hedera", "Enterprise", 23, 3, 80000000, ["binance", "okx", "bybit"], "低流动性"),
            "ETCUSDT": CoinInfo("ETCUSDT", "Ethereum Classic", "Layer1", 24, 3, 200000000, ["binance", "okx", "bybit"], "中等流动性"),
            "FILUSDT": CoinInfo("FILUSDT", "Filecoin", "Storage", 25, 3, 150000000, ["binance", "okx", "bybit"], "中等流动性"),
            "ARBUSDT": CoinInfo("ARBUSDT", "Arbitrum", "Layer2", 26, 2, 180000000, ["binance", "okx", "bybit"], "中等流动性"),
            "OPUSDT": CoinInfo("OPUSDT", "Optimism", "Layer2", 27, 2, 120000000, ["binance", "okx", "bybit"], "中等流动性"),
            "VETUSDT": CoinInfo("VETUSDT", "VeChain", "Supply Chain", 28, 3, 60000000, ["binance", "okx", "bybit"], "低流动性"),
            "ALGOUSDT": CoinInfo("ALGOUSDT", "Algorand", "Layer1", 29, 3, 100000000, ["binance", "okx", "bybit"], "中等流动性"),
            "GRTUSDT": CoinInfo("GRTUSDT", "The Graph", "Indexing", 30, 3, 80000000, ["binance", "okx", "bybit"], "低流动性"),
            "AAVEUSDT": CoinInfo("AAVEUSDT", "Aave", "DeFi", 31, 2, 200000000, ["binance", "okx", "bybit"], "中等流动性"),
            "MKRUSDT": CoinInfo("MKRUSDT", "Maker", "DeFi", 32, 3, 150000000, ["binance", "okx", "bybit"], "中等流动性"),
        }
        
        # 时间框架配置
        self.timeframes = {
            '1m': '1m',     # 1分钟K线 - 高频策略分析
            '5m': '5m',     # 5分钟K线 - DipMaster主要时间框架
            '15m': '15m',   # 15分钟K线 - 中期趋势分析
            '1h': '1h',     # 1小时K线 - 长期趋势分析
        }
        
        # 数据存储路径
        self.data_path = Path("data/enhanced_market_data")
        self.data_path.mkdir(exist_ok=True, parents=True)
        
        # 初始化交易所连接
        self.exchanges = self.setup_exchanges()
        
        # 数据质量标准
        self.quality_standards = {
            'completeness_threshold': 0.99,  # 99%以上的数据完整性
            'max_gap_minutes': 30,           # 最大数据缺失30分钟
            'price_spike_threshold': 0.5,    # 价格异常波动50%阈值
            'volume_outlier_std': 5,         # 成交量异常标准差倍数
            'ohlc_consistency_tolerance': 0.001  # OHLC一致性容忍度
        }
        
        # 数据收集配置
        self.collection_config = {
            'lookback_days': 730,    # 2年历史数据
            'batch_size': 1000,      # 每批次获取1000条记录
            'rate_limit_delay': 0.1, # 请求间隔100ms
            'retry_attempts': 3,     # 失败重试3次
            'timeout_seconds': 30    # 请求超时30秒
        }
        
    def setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/top30_data_collection.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def setup_exchanges(self) -> Dict[str, ccxt.Exchange]:
        """设置交易所连接"""
        exchanges = {}
        
        # Binance - 主要数据源
        exchanges['binance'] = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'sandbox': False,
            'rateLimit': 1200,
            'enableRateLimit': True,
            'options': {'defaultType': 'spot'}
        })
        
        # 备用交易所（如果需要）
        exchanges['okx'] = ccxt.okx({
            'apiKey': '',
            'secret': '',
            'passphrase': '',
            'sandbox': False,
            'rateLimit': 1000,
            'enableRateLimit': True,
        })
        
        return exchanges
    
    async def download_symbol_data(self, 
                                 symbol: str, 
                                 timeframe: str, 
                                 exchange_name: str = 'binance') -> pd.DataFrame:
        """下载单个币种的历史数据"""
        try:
            self.logger.info(f"开始下载 {symbol} {timeframe} 数据...")
            
            exchange = self.exchanges[exchange_name]
            
            # 计算时间范围
            end_time = datetime.now()
            start_time = end_time - timedelta(days=self.collection_config['lookback_days'])
            since = int(start_time.timestamp() * 1000)
            
            # 分批下载数据
            all_ohlcv = []
            current_since = since
            batch_size = self.collection_config['batch_size']
            
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
                    progress = len(all_ohlcv)
                    if progress % 10000 == 0:
                        self.logger.info(f"{symbol} {timeframe}: 已收集 {progress} 条记录")
                    
                    # 避免触发频率限制
                    await asyncio.sleep(self.collection_config['rate_limit_delay'])
                    
                except Exception as e:
                    self.logger.warning(f"批次下载失败 {symbol} {timeframe}: {e}")
                    await asyncio.sleep(1)
                    continue
            
            # 转换为DataFrame
            if not all_ohlcv:
                self.logger.error(f"没有获取到 {symbol} {timeframe} 数据")
                return pd.DataFrame()
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # 数据去重和排序
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            # 数据质量评估
            quality_score = self.assess_data_quality(df, symbol, timeframe)
            
            self.logger.info(f"{symbol} {timeframe} 下载完成: {len(df)} 条记录, 质量评分: {quality_score:.3f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"下载 {symbol} {timeframe} 数据失败: {e}")
            return pd.DataFrame()
    
    def assess_data_quality(self, df: pd.DataFrame, symbol: str, timeframe: str) -> float:
        """评估数据质量"""
        if df.empty:
            return 0.0
        
        quality_scores = {}
        
        try:
            # 1. 完整性检查
            total_expected = self.calculate_expected_records(timeframe)
            actual_records = len(df)
            quality_scores['completeness'] = min(1.0, actual_records / total_expected)
            
            # 2. 缺失值检查
            missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
            quality_scores['no_missing'] = max(0, 1 - missing_ratio)
            
            # 3. OHLC一致性检查
            ohlc_violations = 0
            total_checks = len(df)
            
            # High >= max(Open, Close)
            high_violations = (df['high'] < df[['open', 'close']].max(axis=1)).sum()
            # Low <= min(Open, Close)
            low_violations = (df['low'] > df[['open', 'close']].min(axis=1)).sum()
            
            ohlc_violations = high_violations + low_violations
            quality_scores['ohlc_consistency'] = max(0, 1 - (ohlc_violations / (total_checks * 2)))
            
            # 4. 价格异常检查（异常波动）
            price_changes = df['close'].pct_change().abs()
            extreme_changes = (price_changes > self.quality_standards['price_spike_threshold']).sum()
            quality_scores['price_stability'] = max(0, 1 - (extreme_changes / len(df)))
            
            # 5. 成交量异常检查
            volume_z_scores = np.abs((df['volume'] - df['volume'].mean()) / df['volume'].std())
            volume_outliers = (volume_z_scores > self.quality_standards['volume_outlier_std']).sum()
            quality_scores['volume_stability'] = max(0, 1 - (volume_outliers / len(df)))
            
            # 6. 时间连续性检查
            time_gaps = self.check_time_gaps(df, timeframe)
            quality_scores['time_continuity'] = time_gaps['continuity_score']
            
            # 综合质量评分
            overall_score = np.mean(list(quality_scores.values()))
            
            # 记录详细质量指标
            self.log_quality_metrics(symbol, timeframe, quality_scores, time_gaps)
            
            return overall_score
            
        except Exception as e:
            self.logger.error(f"质量评估失败 {symbol} {timeframe}: {e}")
            return 0.0
    
    def calculate_expected_records(self, timeframe: str) -> int:
        """计算期望的记录数量"""
        days = self.collection_config['lookback_days']
        
        records_per_day = {
            '1m': 1440,    # 24 * 60
            '5m': 288,     # 24 * 12
            '15m': 96,     # 24 * 4
            '1h': 24       # 24
        }
        
        return days * records_per_day.get(timeframe, 288)
    
    def check_time_gaps(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """检查时间间隔的连续性"""
        if len(df) < 2:
            return {'continuity_score': 0.0, 'gaps_count': 0, 'max_gap_minutes': 0}
        
        # 计算期望的时间间隔
        expected_interval = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1)
        }
        
        interval = expected_interval.get(timeframe, timedelta(minutes=5))
        
        # 计算实际时间间隔
        time_diffs = df.index.to_series().diff().dropna()
        
        # 识别异常间隔（超过期望间隔的2倍）
        expected_seconds = interval.total_seconds()
        max_allowed_seconds = expected_seconds * 2
        
        large_gaps = time_diffs[time_diffs.dt.total_seconds() > max_allowed_seconds]
        gaps_count = len(large_gaps)
        
        # 最大间隔（分钟）
        max_gap_minutes = time_diffs.dt.total_seconds().max() / 60 if not time_diffs.empty else 0
        
        # 连续性评分
        continuity_score = max(0, 1 - (gaps_count / len(time_diffs)))
        
        return {
            'continuity_score': continuity_score,
            'gaps_count': gaps_count,
            'max_gap_minutes': max_gap_minutes,
            'large_gaps': large_gaps.tolist()
        }
    
    def log_quality_metrics(self, symbol: str, timeframe: str, scores: Dict, gaps: Dict):
        """记录质量指标"""
        self.logger.info(f"{symbol} {timeframe} 质量指标:")
        self.logger.info(f"  完整性: {scores.get('completeness', 0):.3f}")
        self.logger.info(f"  无缺失: {scores.get('no_missing', 0):.3f}")
        self.logger.info(f"  OHLC一致性: {scores.get('ohlc_consistency', 0):.3f}")
        self.logger.info(f"  价格稳定性: {scores.get('price_stability', 0):.3f}")
        self.logger.info(f"  成交量稳定性: {scores.get('volume_stability', 0):.3f}")
        self.logger.info(f"  时间连续性: {scores.get('time_continuity', 0):.3f}")
        self.logger.info(f"  时间间隔数: {gaps.get('gaps_count', 0)}")
        self.logger.info(f"  最大间隔: {gaps.get('max_gap_minutes', 0):.1f} 分钟")
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> str:
        """保存数据到文件"""
        try:
            file_path = self.data_path / f"{symbol}_{timeframe}_2years.parquet"
            
            # 保存为Parquet格式（高效压缩）
            df.to_parquet(file_path, compression='snappy', index=True)
            
            # 生成元数据
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'records_count': len(df),
                'date_range': {
                    'start': df.index.min().isoformat() if not df.empty else None,
                    'end': df.index.max().isoformat() if not df.empty else None
                },
                'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                'created_at': datetime.now().isoformat()
            }
            
            metadata_path = self.data_path / f"{symbol}_{timeframe}_2years_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"数据已保存: {file_path}")
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"保存数据失败 {symbol} {timeframe}: {e}")
            return ""
    
    async def collect_all_data(self) -> Dict[str, Any]:
        """收集所有币种的历史数据"""
        self.logger.info("开始收集前30大山寨币的2年历史数据...")
        
        collection_start_time = datetime.now()
        successful_downloads = {}
        failed_downloads = []
        quality_reports = {}
        
        # 创建下载任务
        total_tasks = len(self.top30_altcoins) * len(self.timeframes)
        completed_tasks = 0
        
        for symbol, coin_info in self.top30_altcoins.items():
            self.logger.info(f"处理币种: {coin_info.name} ({symbol})")
            
            symbol_data = {}
            
            for timeframe in self.timeframes.keys():
                try:
                    # 下载数据
                    df = await self.download_symbol_data(symbol, timeframe)
                    
                    if not df.empty:
                        # 保存数据
                        file_path = self.save_data(df, symbol, timeframe)
                        
                        if file_path:
                            symbol_data[timeframe] = {
                                'file_path': file_path,
                                'records_count': len(df),
                                'quality_score': self.assess_data_quality(df, symbol, timeframe),
                                'date_range': {
                                    'start': df.index.min().isoformat(),
                                    'end': df.index.max().isoformat()
                                }
                            }
                    else:
                        failed_downloads.append(f"{symbol}_{timeframe}")
                        
                except Exception as e:
                    self.logger.error(f"处理失败 {symbol} {timeframe}: {e}")
                    failed_downloads.append(f"{symbol}_{timeframe}")
                
                completed_tasks += 1
                progress = (completed_tasks / total_tasks) * 100
                self.logger.info(f"总体进度: {progress:.1f}% ({completed_tasks}/{total_tasks})")
            
            if symbol_data:
                successful_downloads[symbol] = {
                    'coin_info': asdict(coin_info),
                    'timeframes': symbol_data
                }
        
        collection_end_time = datetime.now()
        collection_duration = collection_end_time - collection_start_time
        
        # 生成收集报告
        collection_report = {
            'collection_summary': {
                'start_time': collection_start_time.isoformat(),
                'end_time': collection_end_time.isoformat(),
                'duration_minutes': collection_duration.total_seconds() / 60,
                'total_symbols': len(self.top30_altcoins),
                'successful_symbols': len(successful_downloads),
                'failed_downloads': failed_downloads,
                'total_files_created': sum(
                    len(data['timeframes']) for data in successful_downloads.values()
                )
            },
            'data_collection': successful_downloads,
            'quality_standards': self.quality_standards,
            'collection_config': self.collection_config
        }
        
        self.logger.info(f"数据收集完成！耗时: {collection_duration.total_seconds()/60:.1f} 分钟")
        self.logger.info(f"成功: {len(successful_downloads)} 币种, 失败: {len(failed_downloads)} 任务")
        
        return collection_report
    
    def analyze_market_characteristics(self, collection_report: Dict) -> Dict:
        """分析市场特征"""
        self.logger.info("分析市场特征...")
        
        analysis_results = {
            'liquidity_distribution': self.analyze_liquidity_tiers(),
            'category_distribution': self.analyze_category_distribution(),
            'volume_analysis': self.analyze_volume_patterns(),
            'quality_summary': self.analyze_quality_distribution(collection_report),
            'correlation_potential': self.estimate_correlation_patterns(),
            'trading_suitability': self.assess_trading_suitability()
        }
        
        return analysis_results
    
    def analyze_liquidity_tiers(self) -> Dict:
        """流动性分层分析"""
        tiers = {}
        for symbol, coin_info in self.top30_altcoins.items():
            tier = coin_info.liquidity_tier
            if tier not in tiers:
                tiers[tier] = []
            tiers[tier].append({
                'symbol': symbol,
                'name': coin_info.name,
                'volume_usd': coin_info.daily_volume_usd
            })
        
        return tiers
    
    def analyze_category_distribution(self) -> Dict:
        """类别分布分析"""
        categories = {}
        for symbol, coin_info in self.top30_altcoins.items():
            category = coin_info.category
            if category not in categories:
                categories[category] = []
            categories[category].append({
                'symbol': symbol,
                'name': coin_info.name,
                'rank': coin_info.market_cap_rank
            })
        
        return categories
    
    def analyze_volume_patterns(self) -> Dict:
        """成交量模式分析"""
        volume_stats = {}
        total_volume = sum(coin.daily_volume_usd for coin in self.top30_altcoins.values())
        
        for symbol, coin_info in self.top30_altcoins.items():
            volume_share = coin_info.daily_volume_usd / total_volume
            volume_stats[symbol] = {
                'daily_volume_usd': coin_info.daily_volume_usd,
                'volume_share': volume_share,
                'volume_tier': 'High' if volume_share > 0.05 else 'Medium' if volume_share > 0.02 else 'Low'
            }
        
        return volume_stats
    
    def analyze_quality_distribution(self, collection_report: Dict) -> Dict:
        """质量分布分析"""
        quality_scores = []
        quality_by_symbol = {}
        
        for symbol, data in collection_report.get('data_collection', {}).items():
            symbol_scores = []
            for timeframe, tf_data in data.get('timeframes', {}).items():
                score = tf_data.get('quality_score', 0)
                symbol_scores.append(score)
                quality_scores.append(score)
            
            if symbol_scores:
                quality_by_symbol[symbol] = {
                    'avg_quality': np.mean(symbol_scores),
                    'min_quality': min(symbol_scores),
                    'max_quality': max(symbol_scores)
                }
        
        return {
            'overall_avg_quality': np.mean(quality_scores) if quality_scores else 0,
            'quality_distribution': {
                'high_quality': len([s for s in quality_scores if s > 0.95]),
                'medium_quality': len([s for s in quality_scores if 0.9 <= s <= 0.95]),
                'low_quality': len([s for s in quality_scores if s < 0.9])
            },
            'quality_by_symbol': quality_by_symbol
        }
    
    def estimate_correlation_patterns(self) -> Dict:
        """估算相关性模式"""
        correlation_estimates = {
            'high_correlation_groups': [
                ['BTCUSDT', 'LTCUSDT', 'BCHUSDT'],  # Bitcoin forks
                ['ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'AVAXUSDT'],  # Smart contract platforms
                ['UNIUSDT', 'AAVEUSDT', 'MKRUSDT'],  # DeFi tokens
                ['ARBUSDT', 'OPUSDT', 'MATICUSDT']   # Layer 2 solutions
            ],
            'low_correlation_candidates': [
                ['XRPUSDT', 'DOGEUSDT'],  # Different use cases
                ['FILUSDT', 'HBARUSDT'],  # Enterprise/storage
                ['LINKUSDT', 'GRTUSDT'],  # Infrastructure
                ['TONUSDT', 'NEARUSDT']   # Alternative Layer 1s
            ]
        }
        
        return correlation_estimates
    
    def assess_trading_suitability(self) -> Dict:
        """评估交易适用性"""
        suitability_scores = {}
        
        for symbol, coin_info in self.top30_altcoins.items():
            # 评估标准
            liquidity_score = 1.0 if coin_info.liquidity_tier == "高流动性" else 0.7 if coin_info.liquidity_tier == "中等流动性" else 0.4
            volume_score = min(1.0, coin_info.daily_volume_usd / 1000000000)  # 基于10亿USD标准化
            rank_score = max(0.3, 1 - (coin_info.market_cap_rank - 3) / 30)  # 排名越靠前越好
            priority_score = (4 - coin_info.priority) / 3  # 优先级转换为评分
            
            overall_score = (liquidity_score * 0.3 + volume_score * 0.3 + 
                           rank_score * 0.2 + priority_score * 0.2)
            
            suitability_scores[symbol] = {
                'overall_score': overall_score,
                'liquidity_score': liquidity_score,
                'volume_score': volume_score,
                'rank_score': rank_score,
                'priority_score': priority_score,
                'recommendation': 'Excellent' if overall_score > 0.8 else 
                               'Good' if overall_score > 0.6 else 
                               'Fair' if overall_score > 0.4 else 'Poor'
            }
        
        return suitability_scores
    
    def create_market_data_bundle(self, collection_report: Dict, market_analysis: Dict) -> Dict:
        """创建MarketDataBundle_Top30.json"""
        
        timestamp = datetime.now().isoformat()
        
        # 筛选高质量数据
        high_quality_symbols = []
        for symbol, data in collection_report.get('data_collection', {}).items():
            avg_quality = np.mean([
                tf_data.get('quality_score', 0) 
                for tf_data in data.get('timeframes', {}).values()
            ])
            if avg_quality > self.quality_standards['completeness_threshold']:
                high_quality_symbols.append(symbol)
        
        bundle = {
            "version": timestamp,
            "metadata": {
                "bundle_id": f"top30_altcoins_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "strategy_name": "DipMaster_Top30_Altcoins_V1",
                "description": "前30大市值山寨币完整历史数据集 - 为DipMaster策略优化准备",
                "total_symbols": len(self.top30_altcoins),
                "high_quality_symbols": len(high_quality_symbols),
                "data_coverage": "2年历史数据 (2023-2025)",
                "timeframes": list(self.timeframes.keys()),
                "exchanges": ["binance"],
                "collection_date": timestamp,
                "data_quality_standard": self.quality_standards
            },
            
            "symbol_specifications": {
                symbol: {
                    "coin_info": asdict(coin_info),
                    "trading_suitability": market_analysis['trading_suitability'].get(symbol, {}),
                    "data_quality": collection_report.get('data_collection', {}).get(symbol, {})
                }
                for symbol, coin_info in self.top30_altcoins.items()
            },
            
            "market_analysis": market_analysis,
            
            "data_files": {
                symbol: {
                    timeframe: {
                        "file_path": f"data/enhanced_market_data/{symbol}_{timeframe}_2years.parquet",
                        "metadata_path": f"data/enhanced_market_data/{symbol}_{timeframe}_2years_metadata.json",
                        "format": "parquet",
                        "compression": "snappy"
                    }
                    for timeframe in self.timeframes.keys()
                }
                for symbol in self.top30_altcoins.keys()
            },
            
            "quality_assurance": {
                "standards": self.quality_standards,
                "validation_rules": {
                    "minimum_completeness": "99%",
                    "maximum_gap_minutes": 30,
                    "price_spike_detection": "50% threshold",
                    "volume_outlier_detection": "5 sigma",
                    "ohlc_consistency_check": "enabled"
                },
                "quality_distribution": market_analysis.get('quality_summary', {}),
                "recommended_symbols": high_quality_symbols
            },
            
            "usage_recommendations": {
                "excellent_quality": [
                    symbol for symbol, data in market_analysis.get('trading_suitability', {}).items()
                    if data.get('recommendation') == 'Excellent'
                ],
                "portfolio_size": min(15, len(high_quality_symbols)),
                "correlation_groups": market_analysis.get('correlation_potential', {}).get('high_correlation_groups', []),
                "low_correlation_pairs": market_analysis.get('correlation_potential', {}).get('low_correlation_candidates', []),
                "risk_considerations": [
                    "监控数据质量变化",
                    "注意流动性分层差异", 
                    "考虑类别相关性",
                    "定期验证成交量",
                    "关注市场制度变化"
                ]
            },
            
            "performance_metrics": {
                "collection_time_minutes": collection_report.get('collection_summary', {}).get('duration_minutes', 0),
                "success_rate": len(collection_report.get('data_collection', {})) / len(self.top30_altcoins),
                "average_quality_score": market_analysis.get('quality_summary', {}).get('overall_avg_quality', 0),
                "total_data_points": sum(
                    sum(tf_data.get('records_count', 0) for tf_data in data.get('timeframes', {}).values())
                    for data in collection_report.get('data_collection', {}).values()
                ),
                "estimated_storage_mb": len(self.top30_altcoins) * len(self.timeframes) * 50  # 估算
            },
            
            "timestamp": timestamp
        }
        
        return bundle

# 主执行函数
async def main():
    """主函数 - 执行完整的数据收集流程"""
    collector = Top30AltcoinsDataCollector()
    
    try:
        # 第一步：收集所有数据
        collection_report = await collector.collect_all_data()
        
        # 第二步：市场特征分析
        market_analysis = collector.analyze_market_characteristics(collection_report)
        
        # 第三步：创建数据束配置
        market_data_bundle = collector.create_market_data_bundle(collection_report, market_analysis)
        
        # 保存报告
        reports_path = Path("data")
        reports_path.mkdir(exist_ok=True)
        
        # 保存收集报告
        collection_report_path = reports_path / f"Top30_Collection_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(collection_report_path, 'w', encoding='utf-8') as f:
            json.dump(collection_report, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存市场分析
        market_analysis_path = reports_path / f"Top30_Market_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(market_analysis_path, 'w', encoding='utf-8') as f:
            json.dump(market_analysis, f, ensure_ascii=False, indent=2, default=str)
        
        # 保存数据束配置
        bundle_path = reports_path / "MarketDataBundle_Top30.json"
        with open(bundle_path, 'w', encoding='utf-8') as f:
            json.dump(market_data_bundle, f, ensure_ascii=False, indent=2, default=str)
        
        # 生成执行摘要
        print("\n" + "="*80)
        print("前30大山寨币数据收集完成!")
        print("="*80)
        print(f"📊 数据收集报告: {collection_report_path}")
        print(f"📈 市场分析报告: {market_analysis_path}")
        print(f"🎯 数据束配置: {bundle_path}")
        print(f"⏱️  收集耗时: {collection_report.get('collection_summary', {}).get('duration_minutes', 0):.1f} 分钟")
        print(f"✅ 成功币种: {collection_report.get('collection_summary', {}).get('successful_symbols', 0)}/{len(collector.top30_altcoins)}")
        print(f"🏆 平均质量: {market_analysis.get('quality_summary', {}).get('overall_avg_quality', 0):.3f}")
        print("="*80)
        
        return market_data_bundle
        
    except Exception as e:
        collector.logger.error(f"数据收集流程失败: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())