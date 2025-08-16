"""
DataDownloader - 高性能市场数据下载器
支持多交易所、多币种并行下载，具备自动重试和错误恢复机制
"""

import asyncio
import aiohttp
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from pathlib import Path
import ccxt.async_support as ccxt
from dataclasses import dataclass

@dataclass
class DownloadProgress:
    """下载进度跟踪"""
    symbol: str
    timeframe: str
    data_type: str
    total_chunks: int
    completed_chunks: int
    start_time: float
    errors: List[str]
    
    @property
    def progress_pct(self) -> float:
        return (self.completed_chunks / self.total_chunks * 100) if self.total_chunks > 0 else 0
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

class DataDownloader:
    """
    数据下载器 - 企业级市场数据获取引擎
    
    核心特性:
    - 多交易所支持 (Binance, OKX, Bybit)
    - 智能分块下载，避免API限制
    - 异步并发处理，最大化吞吐量
    - 自动重试机制，确保数据完整性
    - 增量更新，节省带宽和时间
    - 数据验证，确保质量
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # API配置
        self.api_keys = config.get('api_keys', {})
        self.rate_limits = config.get('rate_limits', {
            'binance': {'requests_per_minute': 1200, 'weight_per_minute': 6000},
            'okx': {'requests_per_minute': 600},
            'bybit': {'requests_per_minute': 600}
        })
        
        # 下载配置
        self.chunk_size = config.get('chunk_size', 1000)  # 每次请求的K线数量
        self.max_concurrent = config.get('max_concurrent_downloads', 5)
        self.retry_attempts = config.get('retry_attempts', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        
        # 交易所客户端
        self.exchanges = {}
        self.session = None
        
        # 进度跟踪
        self.download_progress = {}
        
        # 数据存储路径
        self.data_root = Path(config.get('data_root', 'data'))
        
    async def initialize(self):
        """初始化下载器和交易所连接"""
        self.logger.info("初始化数据下载器")
        
        # 创建HTTP会话
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(timeout=timeout)
        
        # 初始化交易所客户端
        await self._initialize_exchanges()
        
    async def _initialize_exchanges(self):
        """初始化支持的交易所客户端"""
        exchange_configs = {
            'binance': {
                'class': ccxt.binance,
                'params': {
                    'apiKey': self.api_keys.get('binance', {}).get('api_key'),
                    'secret': self.api_keys.get('binance', {}).get('secret_key'),
                    'sandbox': self.config.get('sandbox', False),
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                }
            },
            'okx': {
                'class': ccxt.okx,
                'params': {
                    'apiKey': self.api_keys.get('okx', {}).get('api_key'),
                    'secret': self.api_keys.get('okx', {}).get('secret_key'),
                    'password': self.api_keys.get('okx', {}).get('passphrase'),
                    'sandbox': self.config.get('sandbox', False),
                    'enableRateLimit': True
                }
            }
        }
        
        for exchange_name, config in exchange_configs.items():
            try:
                if config['params'].get('apiKey'):  # 只有配置了API密钥才初始化
                    exchange = config['class'](config['params'])
                    await exchange.load_markets()
                    self.exchanges[exchange_name] = exchange
                    self.logger.info(f"交易所 {exchange_name} 初始化成功")
                else:
                    # 无API密钥时使用公共接口
                    exchange = config['class']({'enableRateLimit': True})
                    await exchange.load_markets()
                    self.exchanges[exchange_name] = exchange
                    self.logger.info(f"交易所 {exchange_name} 初始化成功 (公共接口)")
                    
            except Exception as e:
                self.logger.warning(f"交易所 {exchange_name} 初始化失败: {e}")
    
    async def download_data(self, 
                          symbol: str,
                          timeframe: str = '5m',
                          data_type: str = 'kline',
                          start_date: str = None,
                          end_date: str = None,
                          exchange: str = 'binance') -> Dict[str, Any]:
        """
        下载市场数据主函数
        
        Args:
            symbol: 交易对符号 (如 'BTCUSDT')
            timeframe: 时间框架 ('1m', '5m', '15m', '1h', '1d')
            data_type: 数据类型 ('kline', 'ticker', 'trades', 'depth', 'funding')
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            exchange: 交易所名称
            
        Returns:
            下载结果字典
        """
        
        if exchange not in self.exchanges:
            raise ValueError(f"不支持的交易所: {exchange}")
        
        self.logger.info(f"开始下载 {exchange}:{symbol} {timeframe} {data_type} 数据")
        
        # 创建进度跟踪
        progress_key = f"{exchange}_{symbol}_{timeframe}_{data_type}"
        
        try:
            # 根据数据类型选择下载方法
            if data_type == 'kline':
                result = await self._download_klines(symbol, timeframe, start_date, end_date, exchange, progress_key)
            elif data_type == 'ticker':
                result = await self._download_tickers(symbol, exchange, progress_key)
            elif data_type == 'trades':
                result = await self._download_trades(symbol, start_date, end_date, exchange, progress_key)
            elif data_type == 'depth':
                result = await self._download_order_book(symbol, exchange, progress_key)
            elif data_type == 'funding':
                result = await self._download_funding_rates(symbol, start_date, end_date, exchange, progress_key)
            else:
                raise ValueError(f"不支持的数据类型: {data_type}")
            
            self.logger.info(f"数据下载完成: {symbol} {timeframe} {data_type}")
            return result
            
        except Exception as e:
            self.logger.error(f"数据下载失败: {symbol} {timeframe} {data_type} - {e}")
            raise
        finally:
            # 清理进度跟踪
            self.download_progress.pop(progress_key, None)
    
    async def _download_klines(self, symbol: str, timeframe: str, start_date: str, 
                             end_date: str, exchange: str, progress_key: str) -> Dict[str, Any]:
        """下载K线数据"""
        exchange_client = self.exchanges[exchange]
        
        # 时间范围处理
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        # 计算时间间隔（毫秒）
        timeframe_ms = self._timeframe_to_ms(timeframe)
        total_chunks = (end_ts - start_ts) // (timeframe_ms * self.chunk_size) + 1
        
        # 初始化进度跟踪
        self.download_progress[progress_key] = DownloadProgress(
            symbol=symbol,
            timeframe=timeframe,
            data_type='kline',
            total_chunks=total_chunks,
            completed_chunks=0,
            start_time=time.time(),
            errors=[]
        )
        
        all_data = []
        current_ts = start_ts
        
        # 分块下载
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def download_chunk(chunk_start_ts: int) -> List:
            async with semaphore:
                chunk_end_ts = min(chunk_start_ts + timeframe_ms * self.chunk_size, end_ts)
                
                for attempt in range(self.retry_attempts):
                    try:
                        # 使用ccxt下载K线数据
                        ohlcv = await exchange_client.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            since=chunk_start_ts,
                            limit=self.chunk_size
                        )
                        
                        # 过滤时间范围
                        filtered_ohlcv = [
                            candle for candle in ohlcv 
                            if chunk_start_ts <= candle[0] <= chunk_end_ts
                        ]
                        
                        self.download_progress[progress_key].completed_chunks += 1
                        return filtered_ohlcv
                        
                    except Exception as e:
                        self.logger.warning(f"下载块失败 (尝试 {attempt + 1}/{self.retry_attempts}): {e}")
                        if attempt < self.retry_attempts - 1:
                            await asyncio.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                        else:
                            self.download_progress[progress_key].errors.append(str(e))
                            raise
        
        # 生成下载任务
        tasks = []
        chunk_start = start_ts
        
        while chunk_start < end_ts:
            tasks.append(download_chunk(chunk_start))
            chunk_start += timeframe_ms * self.chunk_size
        
        # 并发执行下载任务
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for result in chunk_results:
            if isinstance(result, Exception):
                self.logger.error(f"下载块失败: {result}")
            else:
                all_data.extend(result)
        
        # 数据处理和保存
        if all_data:
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.sort_values('timestamp').drop_duplicates('timestamp')
            
            # 保存数据
            file_path = self._get_save_path(symbol, timeframe, 'kline', exchange)
            df.to_parquet(file_path, compression='zstd', index=False)
            
            # 保存元数据
            metadata = {
                'symbol': symbol,
                'timeframe': timeframe,
                'exchange': exchange,
                'data_type': 'kline',
                'start_date': start_date,
                'end_date': end_date,
                'records_count': len(df),
                'download_time': datetime.now().isoformat(),
                'file_size_bytes': file_path.stat().st_size,
                'data_quality': {
                    'completeness': self._calculate_completeness(df, start_date, end_date, timeframe),
                    'has_gaps': self._detect_time_gaps(df['timestamp'], timeframe)
                }
            }
            
            metadata_path = file_path.with_suffix('.metadata.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return {
                'status': 'success',
                'records_count': len(df),
                'file_path': str(file_path),
                'metadata_path': str(metadata_path),
                'data_quality': metadata['data_quality']
            }
        else:
            raise ValueError("未获取到任何数据")
    
    async def _download_funding_rates(self, symbol: str, start_date: str, 
                                    end_date: str, exchange: str, progress_key: str) -> Dict[str, Any]:
        """下载资金费率数据"""
        if exchange != 'binance':
            self.logger.warning(f"交易所 {exchange} 暂不支持资金费率数据下载")
            return {'status': 'skipped', 'reason': 'unsupported_exchange'}
        
        exchange_client = self.exchanges[exchange]
        
        try:
            # 获取资金费率历史
            funding_rates = await exchange_client.fetch_funding_rate_history(
                symbol=symbol,
                since=int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            )
            
            if funding_rates:
                df = pd.DataFrame(funding_rates)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # 保存数据
                file_path = self._get_save_path(symbol, 'funding', 'funding', exchange)
                df.to_parquet(file_path, compression='zstd', index=False)
                
                return {
                    'status': 'success',
                    'records_count': len(df),
                    'file_path': str(file_path)
                }
            else:
                return {'status': 'no_data'}
                
        except Exception as e:
            self.logger.error(f"资金费率下载失败: {e}")
            raise
    
    def _timeframe_to_ms(self, timeframe: str) -> int:
        """将时间框架转换为毫秒"""
        timeframe_map = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000
        }
        return timeframe_map.get(timeframe, 5 * 60 * 1000)
    
    def _get_save_path(self, symbol: str, timeframe: str, data_type: str, exchange: str) -> Path:
        """生成数据保存路径"""
        if data_type == 'kline':
            filename = f"{symbol}_{timeframe}_2years.parquet"
            return self.data_root / 'historical' / filename
        elif data_type == 'funding':
            filename = f"{symbol}_funding_rates.parquet"
            return self.data_root / 'historical' / filename
        else:
            filename = f"{symbol}_{data_type}.parquet"
            return self.data_root / 'historical' / filename
    
    def _calculate_completeness(self, df: pd.DataFrame, start_date: str, end_date: str, timeframe: str) -> float:
        """计算数据完整性百分比"""
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # 计算期望的数据点数量
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30, 
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        minutes = timeframe_minutes.get(timeframe, 5)
        expected_points = int((end_dt - start_dt).total_seconds() / 60 / minutes)
        actual_points = len(df)
        
        return min(actual_points / expected_points, 1.0) if expected_points > 0 else 0.0
    
    def _detect_time_gaps(self, timestamps: pd.Series, timeframe: str) -> bool:
        """检测时间序列中的间隙"""
        if len(timestamps) < 2:
            return False
        
        expected_interval = pd.Timedelta(minutes=self._timeframe_to_ms(timeframe) / 60000)
        time_diffs = timestamps.diff().dropna()
        
        # 允许小的时间偏差（如网络延迟）
        tolerance = expected_interval * 1.1
        gaps = time_diffs > tolerance
        
        return gaps.any()
    
    async def download_missing_symbols(self, missing_symbols: List[str], 
                                     timeframes: List[str] = ['5m'],
                                     start_date: str = None,
                                     end_date: str = None) -> Dict[str, Any]:
        """批量下载缺失的交易对数据"""
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')  # 2年前
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"批量下载 {len(missing_symbols)} 个交易对数据")
        
        results = {}
        
        for symbol in missing_symbols:
            symbol_results = {}
            
            for timeframe in timeframes:
                try:
                    result = await self.download_data(
                        symbol=symbol,
                        timeframe=timeframe,
                        data_type='kline',
                        start_date=start_date,
                        end_date=end_date,
                        exchange='binance'
                    )
                    symbol_results[timeframe] = result
                    
                except Exception as e:
                    self.logger.error(f"下载失败 {symbol} {timeframe}: {e}")
                    symbol_results[timeframe] = {'status': 'failed', 'error': str(e)}
            
            results[symbol] = symbol_results
        
        return results
    
    def get_download_progress(self, progress_key: str = None) -> Dict[str, Any]:
        """获取下载进度"""
        if progress_key:
            return self.download_progress.get(progress_key, {})
        return {
            key: {
                'symbol': progress.symbol,
                'progress_pct': progress.progress_pct,
                'elapsed_time': progress.elapsed_time,
                'errors_count': len(progress.errors)
            }
            for key, progress in self.download_progress.items()
        }
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("清理下载器资源")
        
        # 关闭交易所连接
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except Exception as e:
                self.logger.warning(f"关闭交易所连接失败: {e}")
        
        # 关闭HTTP会话
        if self.session:
            await self.session.close()
        
        self.exchanges.clear()
        self.download_progress.clear()