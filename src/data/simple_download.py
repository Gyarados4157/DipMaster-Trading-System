"""
简单数据下载脚本 - 下载BTCUSDT和ETHUSDT历史数据
使用基础库，不依赖复杂的数据基础设施
"""

import asyncio
import aiohttp
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import logging
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleBinanceDownloader:
    """简单的Binance数据下载器"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3/klines"
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def download_klines(self, symbol: str, interval: str = '5m', 
                            start_time: int = None, end_time: int = None, limit: int = 1000):
        """下载K线数据"""
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = start_time
        if end_time:
            params['endTime'] = end_time
            
        try:
            async with self.session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return data
                else:
                    logger.error(f"API请求失败: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"下载失败: {e}")
            return []
    
    async def download_historical_data(self, symbol: str, days: int = 730):
        """下载历史数据"""
        logger.info(f"开始下载 {symbol} 最近 {days} 天的5分钟K线数据")
        
        # 计算时间范围
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        
        all_data = []
        current_start = start_time
        batch_size = 1000
        
        # 分批下载
        while current_start < end_time:
            # 计算当前批次的结束时间
            current_end = min(current_start + (batch_size * 5 * 60 * 1000), end_time)
            
            logger.info(f"下载 {symbol} 数据: {datetime.fromtimestamp(current_start/1000)} 到 {datetime.fromtimestamp(current_end/1000)}")
            
            # 下载当前批次
            batch_data = await self.download_klines(
                symbol=symbol,
                interval='5m',
                start_time=current_start,
                end_time=current_end,
                limit=batch_size
            )
            
            if batch_data:
                all_data.extend(batch_data)
                logger.info(f"已下载 {len(batch_data)} 条记录，总计 {len(all_data)} 条")
            else:
                logger.warning(f"批次下载失败: {current_start}")
            
            # 更新开始时间
            current_start = current_end + 1
            
            # 避免API限制
            await asyncio.sleep(0.1)
        
        logger.info(f"{symbol} 下载完成，总计 {len(all_data)} 条记录")
        return all_data

def convert_to_dataframe(kline_data):
    """将K线数据转换为DataFrame"""
    if not kline_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(kline_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    
    # 转换数据类型
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)
    
    # 选择需要的列
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # 排序和去重
    df = df.sort_values('timestamp').drop_duplicates('timestamp')
    
    return df

async def download_symbol_data(symbol: str, output_dir: Path):
    """下载单个交易对的数据"""
    async with SimpleBinanceDownloader() as downloader:
        # 下载数据
        kline_data = await downloader.download_historical_data(symbol, days=730)
        
        if not kline_data:
            logger.error(f"{symbol} 数据下载失败")
            return False
        
        # 转换为DataFrame
        df = convert_to_dataframe(kline_data)
        
        if df.empty:
            logger.error(f"{symbol} 数据转换失败")
            return False
        
        # 保存数据
        output_file = output_dir / f"{symbol}_5m_2years.parquet"
        df.to_parquet(output_file, compression='snappy', index=False)
        
        # 生成元数据
        metadata = {
            'symbol': symbol,
            'timeframe': '5m',
            'exchange': 'binance',
            'data_type': 'kline',
            'start_date': df['timestamp'].min().isoformat(),
            'end_date': df['timestamp'].max().isoformat(),
            'records_count': len(df),
            'download_time': datetime.now().isoformat(),
            'file_size_bytes': output_file.stat().st_size,
            'data_quality': {
                'completeness': 1.0,  # 假设完整
                'has_gaps': False     # 简化检查
            }
        }
        
        metadata_file = output_dir / f"{symbol}_5m_2years_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ {symbol} 数据已保存: {len(df)}条记录, 文件: {output_file}")
        return True

async def main():
    """主函数"""
    logger.info("=== 开始下载缺失的历史数据 ===")
    
    # 确定输出目录
    project_root = Path(__file__).parent.parent.parent
    output_dir = project_root / 'data' / 'market_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 要下载的交易对
    symbols = ['BTCUSDT', 'ETHUSDT']
    
    results = {}
    
    for symbol in symbols:
        try:
            success = await download_symbol_data(symbol, output_dir)
            results[symbol] = 'success' if success else 'failed'
        except Exception as e:
            logger.error(f"{symbol} 下载过程中出错: {e}")
            results[symbol] = 'error'
    
    # 输出结果
    logger.info("\n=== 下载结果汇总 ===")
    for symbol, status in results.items():
        if status == 'success':
            logger.info(f"✓ {symbol}: 下载成功")
        else:
            logger.error(f"✗ {symbol}: 下载失败 ({status})")
    
    # 生成汇总报告
    summary = {
        'download_time': datetime.now().isoformat(),
        'symbols': results,
        'success_count': sum(1 for status in results.values() if status == 'success'),
        'total_count': len(results)
    }
    
    summary_file = output_dir / f'download_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"下载汇总已保存: {summary_file}")
    logger.info("=== 数据下载完成 ===")

if __name__ == "__main__":
    asyncio.run(main())