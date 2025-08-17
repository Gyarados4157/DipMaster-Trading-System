#!/usr/bin/env python3
"""
DipMaster Top30 Expansion Data Downloader
为30币种策略扩展下载额外5个币种的数据

新增币种: SHIBUSDT, DOGEUSDT, TONUSDT, PEPEUSDT, INJUSDT
"""

import pandas as pd
import numpy as np
import ccxt
from datetime import datetime, timedelta
import time
import logging
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class Top30ExpansionDataDownloader:
    def __init__(self):
        self.exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # 新增币种列表
        self.new_symbols = ['SHIBUSDT', 'DOGEUSDT', 'TONUSDT', 'PEPEUSDT', 'INJUSDT']
        self.timeframes = ['5m', '15m', '1h']
        self.days_back = 90
        
        # 数据存储路径
        self.data_dir = Path('data/enhanced_market_data')
        self.data_dir.mkdir(exist_ok=True)
        
    def download_symbol_data(self, symbol: str, timeframe: str, days: int = 90) -> pd.DataFrame:
        """下载单个币种的数据"""
        try:
            self.logger.info(f"Downloading {symbol} {timeframe} data for {days} days...")
            
            # 计算开始时间
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            # 转换为毫秒时间戳
            since = int(start_time.timestamp() * 1000)
            
            # 下载数据
            all_data = []
            while since < int(end_time.timestamp() * 1000):
                try:
                    ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
                    if not ohlcv:
                        break
                    
                    all_data.extend(ohlcv)
                    since = ohlcv[-1][0] + 1
                    time.sleep(0.1)  # 限速
                    
                except Exception as e:
                    self.logger.error(f"Error downloading {symbol} {timeframe}: {e}")
                    time.sleep(2)
                    continue
            
            if not all_data:
                self.logger.error(f"No data downloaded for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # 转换为DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 数据清洗
            df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
            
            # 基本验证
            df = df.dropna()
            if len(df) < 100:
                self.logger.warning(f"Insufficient data for {symbol} {timeframe}: {len(df)} rows")
                return pd.DataFrame()
            
            self.logger.info(f"Downloaded {len(df)} rows for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to download {symbol} {timeframe}: {e}")
            return pd.DataFrame()
    
    def validate_data_quality(self, df: pd.DataFrame, symbol: str, timeframe: str) -> dict:
        """验证数据质量"""
        if df.empty:
            return {'quality_score': 0, 'issues': ['No data']}
        
        issues = []
        quality_metrics = {}
        
        # 检查数据完整性
        total_rows = len(df)
        null_count = df.isnull().sum().sum()
        quality_metrics['completeness'] = 1 - (null_count / (total_rows * len(df.columns)))
        
        if quality_metrics['completeness'] < 0.98:
            issues.append(f"Data completeness: {quality_metrics['completeness']:.2%}")
        
        # 检查时间序列连续性
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_diff = df['timestamp'].diff().dt.total_seconds()
        expected_interval = {'5m': 300, '15m': 900, '1h': 3600}[timeframe]
        
        irregular_intervals = (time_diff != expected_interval).sum()
        quality_metrics['time_regularity'] = 1 - (irregular_intervals / total_rows)
        
        if quality_metrics['time_regularity'] < 0.95:
            issues.append(f"Time irregularity: {irregular_intervals} gaps")
        
        # 检查价格异常
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                # 检查零值或负值
                invalid_prices = (df[col] <= 0).sum()
                if invalid_prices > 0:
                    issues.append(f"Invalid {col} prices: {invalid_prices}")
                
                # 检查极端变化
                pct_changes = df[col].pct_change().abs()
                extreme_changes = (pct_changes > 0.5).sum()
                if extreme_changes > total_rows * 0.001:  # 超过0.1%
                    issues.append(f"Extreme {col} changes: {extreme_changes}")
        
        # 检查OHLC逻辑
        if all(col in df.columns for col in price_cols):
            ohlc_violations = (
                (df['high'] < df['open']) | 
                (df['high'] < df['close']) | 
                (df['low'] > df['open']) | 
                (df['low'] > df['close'])
            ).sum()
            
            if ohlc_violations > 0:
                issues.append(f"OHLC logic violations: {ohlc_violations}")
        
        # 检查成交量
        if 'volume' in df.columns:
            zero_volume = (df['volume'] <= 0).sum()
            if zero_volume > total_rows * 0.05:  # 超过5%
                issues.append(f"Zero volume periods: {zero_volume}")
        
        # 计算综合质量分数
        quality_score = np.mean([
            quality_metrics.get('completeness', 0),
            quality_metrics.get('time_regularity', 0),
            1 - (len(issues) * 0.1)  # 每个问题扣10%
        ])
        
        return {
            'quality_score': max(0, quality_score),
            'issues': issues,
            'metrics': quality_metrics,
            'row_count': total_rows
        }
    
    def save_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> bool:
        """保存数据到文件"""
        try:
            if df.empty:
                return False
            
            # 构建文件名
            filename = f"{symbol}_{timeframe}_{self.days_back}days.parquet"
            filepath = self.data_dir / filename
            
            # 保存数据
            df.to_parquet(filepath, index=False)
            
            # 创建元数据
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'days_back': self.days_back,
                'total_rows': len(df),
                'start_date': df['timestamp'].min().isoformat(),
                'end_date': df['timestamp'].max().isoformat(),
                'download_date': datetime.now().isoformat(),
                'file_size_mb': filepath.stat().st_size / (1024 * 1024)
            }
            
            metadata_file = filepath.with_suffix('.parquet_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved {filename} ({len(df)} rows)")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save {symbol} {timeframe}: {e}")
            return False
    
    def download_all_new_symbols(self) -> dict:
        """下载所有新币种数据"""
        results = {
            'downloaded': [],
            'failed': [],
            'quality_scores': {},
            'total_files': 0,
            'summary': {}
        }
        
        for symbol in self.new_symbols:
            symbol_results = {'symbol': symbol, 'timeframes': {}}
            
            for timeframe in self.timeframes:
                try:
                    # 下载数据
                    df = self.download_symbol_data(symbol, timeframe, self.days_back)
                    
                    if not df.empty:
                        # 验证质量
                        quality_result = self.validate_data_quality(df, symbol, timeframe)
                        
                        # 保存数据
                        if quality_result['quality_score'] > 0.8:  # 质量阈值
                            if self.save_data(df, symbol, timeframe):
                                symbol_results['timeframes'][timeframe] = {
                                    'status': 'success',
                                    'rows': len(df),
                                    'quality_score': quality_result['quality_score']
                                }
                                results['total_files'] += 1
                            else:
                                symbol_results['timeframes'][timeframe] = {
                                    'status': 'save_failed',
                                    'error': 'Failed to save file'
                                }
                        else:
                            symbol_results['timeframes'][timeframe] = {
                                'status': 'quality_failed',
                                'quality_score': quality_result['quality_score'],
                                'issues': quality_result['issues']
                            }
                    else:
                        symbol_results['timeframes'][timeframe] = {
                            'status': 'download_failed',
                            'error': 'No data returned'
                        }
                        
                except Exception as e:
                    symbol_results['timeframes'][timeframe] = {
                        'status': 'error',
                        'error': str(e)
                    }
                
                # 休息一下避免限速
                time.sleep(1)
            
            # 判断整体成功或失败
            success_count = sum(1 for tf_result in symbol_results['timeframes'].values() 
                              if tf_result.get('status') == 'success')
            
            if success_count >= 2:  # 至少2个时间框架成功
                results['downloaded'].append(symbol)
            else:
                results['failed'].append(symbol)
            
            results['quality_scores'][symbol] = symbol_results
        
        # 生成摘要
        results['summary'] = {
            'total_symbols': len(self.new_symbols),
            'successful_symbols': len(results['downloaded']),
            'failed_symbols': len(results['failed']),
            'total_files_created': results['total_files'],
            'success_rate': len(results['downloaded']) / len(self.new_symbols)
        }
        
        return results

def main():
    """主函数"""
    print("DipMaster Top30 Expansion Data Downloader")
    print("=" * 50)
    
    downloader = Top30ExpansionDataDownloader()
    
    print(f"新增币种: {', '.join(downloader.new_symbols)}")
    print(f"时间框架: {', '.join(downloader.timeframes)}")
    print(f"历史数据: {downloader.days_back} 天")
    print()
    
    # 开始下载
    print("开始下载数据...")
    start_time = time.time()
    
    results = downloader.download_all_new_symbols()
    
    end_time = time.time()
    
    # 打印结果
    print(f"\n下载完成! 耗时: {end_time - start_time:.1f} 秒")
    print(f"成功下载: {results['summary']['successful_symbols']}/{results['summary']['total_symbols']} 币种")
    print(f"创建文件: {results['summary']['total_files']} 个")
    print(f"成功率: {results['summary']['success_rate']:.1%}")
    
    if results['downloaded']:
        print(f"\n✅ 成功币种: {', '.join(results['downloaded'])}")
    
    if results['failed']:
        print(f"\n❌ 失败币种: {', '.join(results['failed'])}")
    
    # 保存结果报告
    report_file = Path('data') / f'top30_expansion_download_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 详细报告已保存: {report_file}")
    
    return results

if __name__ == "__main__":
    main()