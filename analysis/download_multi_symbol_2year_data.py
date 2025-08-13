#!/usr/bin/env python3
"""
多币种2年历史数据下载器
下载指定9个币种的2年5分钟K线数据，用于DipMaster V3深度回测

目标币种:
- XRPUSDT, DOGEUSDT, ICPUSDT, IOTAUSDT
- SOLUSDT, SUIUSDT, ALGOUSDT, BNBUSDT, ADAUSDT
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import List, Dict, Optional
import os

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_download_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class MultiSymbolDataDownloader:
    """多币种历史数据下载器"""
    
    def __init__(self):
        # 目标币种列表
        self.symbols = [
            'XRPUSDT', 'DOGEUSDT', 'ICPUSDT', 'IOTAUSDT',
            'SOLUSDT', 'SUIUSDT', 'ALGOUSDT', 'BNBUSDT', 'ADAUSDT'
        ]
        
        # Binance API配置
        self.base_url = "https://api.binance.com"
        self.klines_endpoint = "/api/v3/klines"
        
        # 数据参数
        self.interval = "5m"  # 5分钟K线
        self.limit = 1000     # 每次请求最大数量
        
        # 时间范围：2年数据
        self.end_time = datetime.now()
        self.start_time = self.end_time - timedelta(days=730)  # 2年
        
        # 输出目录
        self.output_dir = Path("data/market_data")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.download_stats = {}
        
    def get_klines(self, symbol: str, start_time: int, end_time: int) -> List[List]:
        """获取K线数据"""
        url = f"{self.base_url}{self.klines_endpoint}"
        params = {
            'symbol': symbol,
            'interval': self.interval,
            'startTime': start_time,
            'endTime': end_time,
            'limit': self.limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"获取{symbol}数据失败: {e}")
            return []
    
    def download_symbol_data(self, symbol: str) -> bool:
        """下载单个币种的完整数据"""
        logger.info(f"📊 开始下载 {symbol} 数据...")
        
        all_klines = []
        current_start = int(self.start_time.timestamp() * 1000)
        end_timestamp = int(self.end_time.timestamp() * 1000)
        
        request_count = 0
        
        while current_start < end_timestamp:
            # 计算本次请求的结束时间
            current_end = min(
                current_start + (self.limit * 5 * 60 * 1000),  # 5分钟 * limit条 * 毫秒
                end_timestamp
            )
            
            # 获取数据
            klines = self.get_klines(symbol, current_start, current_end)
            
            if not klines:
                logger.warning(f"⚠️ {symbol}: 未获取到数据，时间范围: {current_start} - {current_end}")
                break
            
            all_klines.extend(klines)
            request_count += 1
            
            # 更新起始时间为最后一条数据的时间+1分钟
            if klines:
                last_time = klines[-1][0]  # 开盘时间
                current_start = last_time + (5 * 60 * 1000)  # +5分钟
                
                # 进度显示
                progress = (current_start - int(self.start_time.timestamp() * 1000)) / (end_timestamp - int(self.start_time.timestamp() * 1000)) * 100
                if request_count % 10 == 0:
                    logger.info(f"   {symbol} 下载进度: {progress:.1f}% ({len(all_klines)}条数据)")
            
            # API限制：避免请求过频
            time.sleep(0.1)
            
            # 安全退出
            if request_count > 1000:  # 防止无限循环
                logger.warning(f"⚠️ {symbol}: 请求次数过多，停止下载")
                break
        
        if not all_klines:
            logger.error(f"❌ {symbol}: 未获取到任何数据")
            return False
        
        # 转换为DataFrame
        df = self.klines_to_dataframe(all_klines)
        
        if df.empty:
            logger.error(f"❌ {symbol}: 数据转换失败")
            return False
        
        # 数据清理和验证
        df = self.clean_data(df, symbol)
        
        if len(df) < 1000:
            logger.warning(f"⚠️ {symbol}: 数据量不足({len(df)}条)，建议重新下载")
        
        # 保存数据
        output_file = self.output_dir / f"{symbol}_5m_2years.csv"
        df.to_csv(output_file, index=False)
        
        # 保存元数据
        metadata = {
            'symbol': symbol,
            'interval': self.interval,
            'total_records': len(df),
            'start_time': str(df['timestamp'].min()),
            'end_time': str(df['timestamp'].max()),
            'download_time': datetime.now().isoformat(),
            'file_size_mb': output_file.stat().st_size / 1024 / 1024,
            'api_requests': request_count
        }
        
        metadata_file = self.output_dir / f"{symbol}_5m_2years_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 更新统计
        self.download_stats[symbol] = metadata
        
        logger.info(f"✅ {symbol}: 成功下载{len(df)}条数据，时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
        logger.info(f"   文件: {output_file}")
        
        return True
    
    def klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """将K线数据转换为DataFrame"""
        if not klines:
            return pd.DataFrame()
        
        columns = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        
        df = pd.DataFrame(klines, columns=columns)
        
        # 数据类型转换
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'taker_buy_base_asset_volume', 
                          'taker_buy_quote_asset_volume']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['number_of_trades'] = pd.to_numeric(df['number_of_trades'], errors='coerce')
        
        return df
    
    def clean_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """数据清理"""
        initial_count = len(df)
        
        # 移除重复数据
        df = df.drop_duplicates(subset=['timestamp'])
        
        # 移除空值
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # 移除价格为0或负数的数据
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df = df[df[col] > 0]
        
        # 移除异常价格变动（超过50%的跳变）
        for col in price_columns:
            price_change = df[col].pct_change().abs()
            df = df[price_change < 0.5]
        
        # 确保OHLC逻辑正确
        df = df[
            (df['high'] >= df['low']) & 
            (df['high'] >= df['open']) & 
            (df['high'] >= df['close']) &
            (df['low'] <= df['open']) & 
            (df['low'] <= df['close'])
        ]
        
        # 按时间排序
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        cleaned_count = len(df)
        removed_count = initial_count - cleaned_count
        
        if removed_count > 0:
            logger.info(f"   {symbol}: 清理了{removed_count}条异常数据 ({removed_count/initial_count*100:.1f}%)")
        
        return df
    
    def download_all_symbols(self) -> Dict[str, bool]:
        """下载所有币种数据"""
        logger.info("🚀 开始下载所有币种的2年历史数据...")
        logger.info(f"📊 目标币种: {self.symbols}")
        logger.info(f"📅 时间范围: {self.start_time.strftime('%Y-%m-%d')} ~ {self.end_time.strftime('%Y-%m-%d')}")
        logger.info(f"🔢 数据周期: {self.interval}")
        
        results = {}
        success_count = 0
        
        for i, symbol in enumerate(self.symbols, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📈 [{i}/{len(self.symbols)}] 处理 {symbol}")
            logger.info(f"{'='*60}")
            
            # 检查是否已存在数据
            existing_file = self.output_dir / f"{symbol}_5m_2years.csv"
            if existing_file.exists():
                logger.info(f"⚠️ {symbol}: 数据文件已存在，跳过下载")
                logger.info(f"   如需重新下载，请删除: {existing_file}")
                results[symbol] = True
                success_count += 1
                continue
            
            try:
                success = self.download_symbol_data(symbol)
                results[symbol] = success
                
                if success:
                    success_count += 1
                else:
                    logger.error(f"❌ {symbol}: 下载失败")
                
            except Exception as e:
                logger.error(f"💥 {symbol}: 下载异常: {e}")
                results[symbol] = False
            
            # 下载间隔，避免API限制
            if i < len(self.symbols):
                logger.info("⏳ 等待3秒后继续下载下一个币种...")
                time.sleep(3)
        
        # 下载总结
        logger.info(f"\n{'='*60}")
        logger.info("📊 下载总结")
        logger.info(f"{'='*60}")
        logger.info(f"✅ 成功: {success_count}/{len(self.symbols)} 个币种")
        logger.info(f"❌ 失败: {len(self.symbols) - success_count} 个币种")
        
        if success_count > 0:
            logger.info(f"\n📁 数据文件位置: {self.output_dir}")
            
            # 显示详细统计
            total_records = sum(stats['total_records'] for stats in self.download_stats.values())
            total_size = sum(stats['file_size_mb'] for stats in self.download_stats.values())
            
            logger.info(f"📈 总数据量: {total_records:,} 条K线")
            logger.info(f"💾 总文件大小: {total_size:.2f} MB")
        
        # 失败币种详情
        failed_symbols = [symbol for symbol, success in results.items() if not success]
        if failed_symbols:
            logger.warning(f"⚠️ 以下币种下载失败: {failed_symbols}")
            logger.warning("   建议检查网络连接或币种名称是否正确")
        
        return results
    
    def generate_summary_report(self) -> None:
        """生成下载摘要报告"""
        if not self.download_stats:
            logger.warning("⚠️ 没有下载统计数据可生成报告")
            return
        
        summary = {
            'download_summary': {
                'total_symbols': len(self.symbols),
                'successful_downloads': len(self.download_stats),
                'failed_downloads': len(self.symbols) - len(self.download_stats),
                'download_completion_time': datetime.now().isoformat()
            },
            'symbol_details': self.download_stats,
            'aggregate_statistics': {
                'total_records': sum(stats['total_records'] for stats in self.download_stats.values()),
                'total_size_mb': sum(stats['file_size_mb'] for stats in self.download_stats.values()),
                'earliest_data': min(stats['start_time'] for stats in self.download_stats.values()),
                'latest_data': max(stats['end_time'] for stats in self.download_stats.values()),
                'total_api_requests': sum(stats['api_requests'] for stats in self.download_stats.values())
            }
        }
        
        # 保存摘要报告
        summary_file = self.output_dir / f"download_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 摘要报告已保存: {summary_file}")

def main():
    """主函数"""
    print("📊 多币种2年历史数据下载器")
    print("=" * 60)
    print("目标币种: XRPUSDT, DOGEUSDT, ICPUSDT, IOTAUSDT")
    print("          SOLUSDT, SUIUSDT, ALGOUSDT, BNBUSDT, ADAUSDT")
    print("数据周期: 5分钟K线")
    print("时间范围: 2年历史数据")
    print("=" * 60)
    
    # 创建下载器
    downloader = MultiSymbolDataDownloader()
    
    # 开始下载
    results = downloader.download_all_symbols()
    
    # 生成摘要报告
    downloader.generate_summary_report()
    
    # 根据结果返回退出码
    success_count = sum(1 for success in results.values() if success)
    total_count = len(results)
    
    if success_count == total_count:
        print(f"\n🎉 所有{total_count}个币种数据下载完成！")
        return 0
    elif success_count > 0:
        print(f"\n⚠️ 部分成功：{success_count}/{total_count}个币种下载完成")
        return 1
    else:
        print(f"\n❌ 下载失败：没有成功下载任何币种数据")
        return 2

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)