#!/usr/bin/env python3
"""
Ultra Symbol Data Manager - 扩展币种数据管理器
==============================================

功能：
1. 下载扩展的优质币种数据（避开BTC/ETH）
2. 数据质量检查和清洗
3. 实时数据更新和维护
4. 币种评级和筛选

目标币种池：
- Tier 1: MATIC, DOT, AVAX, LINK, NEAR, ATOM (高流动性Layer1/DeFi)
- Tier 2: UNI, VET, XLM, FTM (主流币种)
- Tier 3: SAND, MANA, CHZ, ENJ, GALA (游戏/元宇宙)

Author: DipMaster Ultra Team
Date: 2025-08-15
Version: 1.0.0
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
import asyncio
import logging
import time
from binance.client import Client
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """币种信息"""
    symbol: str
    tier: int                    # 1-3级分级
    market_cap_rank: int = 0     # 市值排名
    daily_volume_24h: float = 0  # 24小时成交量
    price_precision: int = 4     # 价格精度
    quantity_precision: int = 6  # 数量精度
    min_notional: float = 10     # 最小订单价值
    is_active: bool = True       # 是否活跃交易
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass  
class DataQuality:
    """数据质量评估"""
    symbol: str
    completeness: float = 0.0    # 数据完整性 0-1
    gap_count: int = 0          # 数据缺口数量
    price_anomaly_count: int = 0 # 价格异常数量
    volume_anomaly_count: int = 0# 成交量异常数量
    quality_score: float = 0.0   # 综合质量评分 0-100
    issues: List[str] = field(default_factory=list)  # 问题列表


class UltraSymbolDataManager:
    """超级币种数据管理器"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path("data/market_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Binance客户端（使用免费API）
        self.client = Client()  # 不需要API密钥获取市场数据
        
        # 扩展的币种池定义
        self.tier_1_symbols = [
            "MATICUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT", 
            "NEARUSDT", "ATOMUSDT", "FTMUSDT"
        ]
        
        self.tier_2_symbols = [
            "UNIUSDT", "VETUSDT", "XLMUSDT", "HBARUSDT",
            "ARUSDT", "IMXUSDT", "FLOWUSDT"
        ]
        
        self.tier_3_symbols = [
            "SANDUSDT", "MANAUSDT", "CHZUSDT", "ENJUSDT",
            "GALAUSDT", "AXSUSDT"
        ]
        
        # 已有的币种（不重复下载）
        self.existing_symbols = [
            "DOGEUSDT", "IOTAUSDT", "SOLUSDT", "SUIUSDT", 
            "ALGOUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "ICPUSDT"
        ]
        
        # 币种信息存储
        self.symbol_info: Dict[str, SymbolInfo] = {}
        self.data_quality: Dict[str, DataQuality] = {}
        
        # 数据下载配置
        self.timeframe = "5m"
        self.lookback_days = 730  # 2年数据
        self.batch_size = 1000    # 每批次下载数量
        self.request_delay = 0.1  # API请求延迟
        
    def initialize_symbol_pool(self):
        """初始化币种池"""
        logger.info("🔄 Initializing expanded symbol pool...")
        
        # 获取交易所交易信息
        exchange_info = self.client.get_exchange_info()
        symbols_info = {s['symbol']: s for s in exchange_info['symbols']}
        
        # 初始化各级别币种
        for tier, symbols in enumerate([
            self.tier_1_symbols, 
            self.tier_2_symbols, 
            self.tier_3_symbols
        ], 1):
            for symbol in symbols:
                if symbol in symbols_info:
                    info = symbols_info[symbol]
                    
                    # 提取精度和最小订单信息
                    price_precision = len(str(info['filters'][0]['tickSize']).split('.')[-1].rstrip('0'))
                    quantity_precision = len(str(info['filters'][2]['stepSize']).split('.')[-1].rstrip('0'))
                    min_notional = float(info['filters'][3]['minNotional'])
                    
                    self.symbol_info[symbol] = SymbolInfo(
                        symbol=symbol,
                        tier=tier,
                        price_precision=price_precision,
                        quantity_precision=quantity_precision,
                        min_notional=min_notional,
                        is_active=info['status'] == 'TRADING'
                    )
                else:
                    logger.warning(f"❌ Symbol {symbol} not found on Binance")
                    
        # 获取24小时统计
        self._update_market_stats()
        
        logger.info(f"✅ Symbol pool initialized: {len(self.symbol_info)} symbols")
        logger.info(f"  • Tier 1: {len(self.tier_1_symbols)} symbols")
        logger.info(f"  • Tier 2: {len(self.tier_2_symbols)} symbols") 
        logger.info(f"  • Tier 3: {len(self.tier_3_symbols)} symbols")
        
    def _update_market_stats(self):
        """更新市场统计信息"""
        try:
            # 获取24小时统计
            stats = self.client.get_ticker()
            stats_dict = {s['symbol']: s for s in stats}
            
            for symbol, info in self.symbol_info.items():
                if symbol in stats_dict:
                    stat = stats_dict[symbol]
                    info.daily_volume_24h = float(stat['quoteVolume'])
                    
        except Exception as e:
            logger.error(f"Error updating market stats: {e}")
            
    def download_symbol_data(self, symbol: str, start_date: str = None, 
                           end_date: str = None) -> bool:
        """下载单个币种数据"""
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
            
        logger.info(f"📥 Downloading {symbol} data from {start_date} to {end_date}")
        
        try:
            # 获取K线数据
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=self.timeframe,
                start_str=start_date,
                end_str=end_date
            )
            
            if not klines:
                logger.error(f"❌ No data received for {symbol}")
                return False
                
            # 转换为DataFrame
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # 数据类型转换
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # 数据质量检查
            quality = self._assess_data_quality(symbol, df)
            self.data_quality[symbol] = quality
            
            # 保存数据
            filename = self.data_dir / f"{symbol}_{self.timeframe}_2years.csv"
            df.to_csv(filename, index=False)
            
            # 保存元数据
            metadata = {
                "symbol": symbol,
                "interval": self.timeframe,
                "total_records": len(df),
                "start_time": df['timestamp'].min().strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": df['timestamp'].max().strftime("%Y-%m-%d %H:%M:%S"),
                "download_time": datetime.now().isoformat(),
                "file_size_mb": filename.stat().st_size / 1024 / 1024,
                "data_quality": quality.__dict__
            }
            
            metadata_file = self.data_dir / f"{symbol}_{self.timeframe}_2years_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
                
            logger.info(f"✅ {symbol}: {len(df)} records, Quality: {quality.quality_score:.1f}/100")
            
            # API限制延迟
            time.sleep(self.request_delay)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading {symbol}: {e}")
            return False
            
    def _assess_data_quality(self, symbol: str, df: pd.DataFrame) -> DataQuality:
        """评估数据质量"""
        quality = DataQuality(symbol=symbol)
        
        if len(df) == 0:
            return quality
            
        # 1. 完整性检查
        expected_records = self.lookback_days * 24 * 60 / 5  # 5分钟数据
        quality.completeness = len(df) / expected_records
        
        # 2. 数据缺口检查
        df_sorted = df.sort_values('timestamp')
        time_diffs = df_sorted['timestamp'].diff()
        expected_diff = pd.Timedelta(minutes=5)
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        quality.gap_count = len(gaps)
        
        # 3. 价格异常检查
        price_changes = df['close'].pct_change()
        price_anomalies = abs(price_changes) > 0.2  # 20%以上价格变化视为异常
        quality.price_anomaly_count = price_anomalies.sum()
        
        # 4. 成交量异常检查
        volume_median = df['volume'].median()
        volume_anomalies = df['volume'] > volume_median * 50  # 50倍中位数成交量
        quality.volume_anomaly_count = volume_anomalies.sum()
        
        # 5. 零值检查
        zero_prices = (df[['open', 'high', 'low', 'close']] == 0).any(axis=1).sum()
        zero_volumes = (df['volume'] == 0).sum()
        
        # 综合评分
        completeness_score = quality.completeness * 40
        gap_penalty = min(quality.gap_count / 100, 0.2) * 20
        anomaly_penalty = min((quality.price_anomaly_count + quality.volume_anomaly_count) / 1000, 0.2) * 20
        zero_penalty = min((zero_prices + zero_volumes) / len(df), 0.2) * 20
        
        quality.quality_score = max(0, completeness_score - gap_penalty - anomaly_penalty - zero_penalty)
        
        # 问题记录
        if quality.completeness < 0.95:
            quality.issues.append(f"数据完整性不足: {quality.completeness:.1%}")
        if quality.gap_count > 100:
            quality.issues.append(f"数据缺口过多: {quality.gap_count}个")
        if quality.price_anomaly_count > 10:
            quality.issues.append(f"价格异常: {quality.price_anomaly_count}次")
        if quality.volume_anomaly_count > 10:
            quality.issues.append(f"成交量异常: {quality.volume_anomaly_count}次")
            
        return quality
        
    def download_all_symbols(self, max_concurrent: int = 3):
        """批量下载所有币种数据"""
        all_new_symbols = self.tier_1_symbols + self.tier_2_symbols + self.tier_3_symbols
        
        # 过滤已存在的币种
        symbols_to_download = []
        for symbol in all_new_symbols:
            data_file = self.data_dir / f"{symbol}_{self.timeframe}_2years.csv"
            if not data_file.exists():
                symbols_to_download.append(symbol)
            else:
                logger.info(f"⏭️  Skipping {symbol} (already exists)")
                
        logger.info(f"📦 Starting batch download: {len(symbols_to_download)} symbols")
        
        successful_downloads = 0
        failed_downloads = 0
        
        # 串行下载（避免API限制）
        for i, symbol in enumerate(symbols_to_download, 1):
            logger.info(f"📥 [{i}/{len(symbols_to_download)}] Downloading {symbol}...")
            
            if self.download_symbol_data(symbol):
                successful_downloads += 1
            else:
                failed_downloads += 1
                
            # 进度报告
            if i % 5 == 0 or i == len(symbols_to_download):
                logger.info(f"📊 Progress: {i}/{len(symbols_to_download)} "
                           f"(✅{successful_downloads} ❌{failed_downloads})")
                           
        # 生成下载摘要
        summary = {
            "download_summary": {
                "total_symbols": len(symbols_to_download),
                "successful_downloads": successful_downloads,
                "failed_downloads": failed_downloads,
                "download_completion_time": datetime.now().isoformat()
            },
            "quality_summary": {
                symbol: quality.__dict__ 
                for symbol, quality in self.data_quality.items()
            }
        }
        
        summary_file = self.data_dir / f"ultra_download_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        logger.info(f"🎉 Batch download completed!")
        logger.info(f"  • Successful: {successful_downloads}")
        logger.info(f"  • Failed: {failed_downloads}")
        logger.info(f"  • Summary saved: {summary_file}")
        
        return successful_downloads, failed_downloads
        
    def get_quality_report(self) -> Dict:
        """获取数据质量报告"""
        if not self.data_quality:
            return {"message": "No data quality information available"}
            
        report = {
            "总币种数": len(self.data_quality),
            "高质量币种": len([q for q in self.data_quality.values() if q.quality_score >= 85]),
            "中等质量币种": len([q for q in self.data_quality.values() if 70 <= q.quality_score < 85]),
            "低质量币种": len([q for q in self.data_quality.values() if q.quality_score < 70]),
            "平均质量评分": np.mean([q.quality_score for q in self.data_quality.values()]),
            "详细信息": {}
        }
        
        for symbol, quality in self.data_quality.items():
            report["详细信息"][symbol] = {
                "质量评分": f"{quality.quality_score:.1f}/100",
                "数据完整性": f"{quality.completeness:.1%}",
                "数据缺口": quality.gap_count,
                "价格异常": quality.price_anomaly_count,
                "成交量异常": quality.volume_anomaly_count,
                "问题列表": quality.issues
            }
            
        return report
        
    def get_recommended_symbols(self, min_quality: float = 80, 
                              min_volume_24h: float = 10_000_000) -> List[str]:
        """获取推荐的优质币种"""
        recommended = []
        
        for symbol, info in self.symbol_info.items():
            # 检查质量要求
            quality = self.data_quality.get(symbol)
            if not quality or quality.quality_score < min_quality:
                continue
                
            # 检查成交量要求
            if info.daily_volume_24h < min_volume_24h:
                continue
                
            # 检查是否活跃交易
            if not info.is_active:
                continue
                
            recommended.append(symbol)
            
        # 按Tier排序
        def sort_key(symbol):
            tier = self.symbol_info[symbol].tier
            quality_score = self.data_quality[symbol].quality_score
            volume = self.symbol_info[symbol].daily_volume_24h
            return (tier, -quality_score, -volume)
            
        recommended.sort(key=sort_key)
        
        return recommended
        
    def create_combined_dataset(self, symbols: List[str]) -> pd.DataFrame:
        """创建合并的数据集供回测使用"""
        combined_data = []
        
        for symbol in symbols:
            data_file = self.data_dir / f"{symbol}_{self.timeframe}_2years.csv"
            if data_file.exists():
                df = pd.read_csv(data_file)
                df['symbol'] = symbol
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                combined_data.append(df)
            else:
                logger.warning(f"Data file not found for {symbol}")
                
        if combined_data:
            result = pd.concat(combined_data, ignore_index=True)
            result = result.sort_values(['timestamp', 'symbol'])
            logger.info(f"✅ Combined dataset created: {len(result)} records, {len(symbols)} symbols")
            return result
        else:
            logger.error("❌ No data available for combination")
            return pd.DataFrame()


async def main():
    """主函数 - 下载扩展币种数据"""
    logger.info("🚀 Starting Ultra Symbol Data Manager")
    
    # 创建数据管理器
    manager = UltraSymbolDataManager()
    
    # 初始化币种池
    manager.initialize_symbol_pool()
    
    # 下载所有新币种数据
    success_count, fail_count = manager.download_all_symbols()
    
    # 生成质量报告
    quality_report = manager.get_quality_report()
    logger.info("📊 Data Quality Report:")
    for key, value in quality_report.items():
        if key != "详细信息":
            logger.info(f"  • {key}: {value}")
            
    # 获取推荐币种
    recommended = manager.get_recommended_symbols()
    logger.info(f"🎯 Recommended high-quality symbols ({len(recommended)}):")
    for symbol in recommended[:10]:  # 显示前10个
        info = manager.symbol_info[symbol]
        quality = manager.data_quality[symbol]
        logger.info(f"  • {symbol} [Tier {info.tier}] - Quality: {quality.quality_score:.1f}, "
                   f"Volume: ${info.daily_volume_24h/1e6:.1f}M")
        
    logger.info("🎉 Ultra Symbol Data Manager completed successfully!")
    return manager


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行主函数
    asyncio.run(main())