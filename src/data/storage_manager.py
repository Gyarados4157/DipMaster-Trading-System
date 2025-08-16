"""
StorageManager - 高性能数据存储引擎
针对量化交易优化的存储系统，支持高频读写、压缩存储和快速查询
"""

import asyncio
import logging
import shutil
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
import json
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
import zstandard as zstd
from dataclasses import dataclass

@dataclass
class StorageConfig:
    """存储配置"""
    format: str = 'parquet'
    compression: str = 'zstd'
    partition_by: str = 'date'
    max_file_size_mb: int = 100
    backup_enabled: bool = True
    cache_size_mb: int = 512

@dataclass
class StorageStats:
    """存储统计信息"""
    total_size_mb: float
    file_count: int
    compression_ratio: float
    avg_read_latency_ms: float
    avg_write_latency_ms: float
    cache_hit_ratio: float

class StorageManager:
    """
    存储管理器 - DipMaster数据存储核心
    
    核心特性:
    - 列式存储优化 (Parquet + Arrow)
    - 智能分片策略 (按日期/符号分区)
    - 高压缩比存储 (Zstandard压缩)
    - 内存映射访问 (零拷贝读取)
    - 并行I/O操作 (异步读写)
    - 自动备份机制 (增量备份)
    - 查询性能优化 (索引和缓存)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = StorageConfig(**config.get('storage', {}))
        
        # 存储路径配置
        self.data_root = Path(config.get('data_root', 'data'))
        self.historical_path = self.data_root / 'historical'
        self.realtime_path = self.data_root / 'realtime'
        self.backup_path = self.data_root / 'backup'
        self.metadata_path = self.data_root / 'metadata'
        
        # 确保目录存在
        self._ensure_directories()
        
        # 线程池用于I/O操作
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # 内存缓存
        self.cache = {}
        self.cache_size_limit = self.config.cache_size_mb * 1024 * 1024
        self.current_cache_size = 0
        
        # 性能监控
        self.read_times = []
        self.write_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # SQLite连接（用于元数据和索引）
        self.metadata_db = None
        self._init_metadata_db()
    
    def _ensure_directories(self):
        """确保必要的目录结构存在"""
        directories = [
            self.historical_path,
            self.realtime_path,
            self.backup_path,
            self.metadata_path,
            self.historical_path / 'klines',
            self.historical_path / 'trades',
            self.historical_path / 'funding',
            self.historical_path / 'orderbooks',
            self.realtime_path / 'cache',
            self.realtime_path / 'buffers'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _init_metadata_db(self):
        """初始化元数据数据库"""
        db_path = self.metadata_path / 'storage_metadata.db'
        
        try:
            self.metadata_db = sqlite3.connect(str(db_path), check_same_thread=False)
            self.metadata_db.execute('''
                CREATE TABLE IF NOT EXISTS file_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    data_type TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    record_count INTEGER,
                    file_size INTEGER,
                    compression_ratio REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(symbol, timeframe, data_type, start_date)
                )
            ''')
            
            self.metadata_db.execute('''
                CREATE INDEX IF NOT EXISTS idx_symbol_timeframe 
                ON file_metadata(symbol, timeframe)
            ''')
            
            self.metadata_db.execute('''
                CREATE INDEX IF NOT EXISTS idx_date_range 
                ON file_metadata(start_date, end_date)
            ''')
            
            self.metadata_db.commit()
            
        except Exception as e:
            self.logger.error(f"元数据数据库初始化失败: {e}")
    
    async def save_kline_data(self, df: pd.DataFrame, symbol: str, timeframe: str,
                            start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        保存K线数据
        
        Args:
            df: K线数据DataFrame
            symbol: 交易对符号
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            保存结果
        """
        start_time = datetime.now()
        
        try:
            # 数据预处理
            df_processed = self._preprocess_kline_data(df)
            
            # 确定保存路径
            file_path = self._get_kline_file_path(symbol, timeframe)
            
            # 分片保存策略
            if len(df_processed) > 100000:  # 大数据集分片保存
                result = await self._save_partitioned_data(df_processed, file_path, symbol, timeframe)
            else:
                result = await self._save_single_file(df_processed, file_path, symbol, timeframe)
            
            # 更新元数据
            await self._update_metadata(symbol, timeframe, 'kline', file_path, 
                                      start_date, end_date, len(df_processed), result['file_size'])
            
            # 记录性能指标
            write_time = (datetime.now() - start_time).total_seconds() * 1000
            self.write_times.append(write_time)
            
            self.logger.info(f"K线数据保存完成: {symbol} {timeframe}, "
                           f"{len(df_processed)}条记录, 耗时{write_time:.1f}ms")
            
            return {
                'status': 'success',
                'file_path': str(file_path),
                'record_count': len(df_processed),
                'file_size': result['file_size'],
                'compression_ratio': result.get('compression_ratio', 0),
                'write_time_ms': write_time
            }
            
        except Exception as e:
            self.logger.error(f"K线数据保存失败: {symbol} {timeframe} - {e}")
            raise
    
    def _preprocess_kline_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理K线数据"""
        df_copy = df.copy()
        
        # 确保列名标准化
        column_mapping = {
            'open_time': 'timestamp',
            'kline_open_time': 'timestamp'
        }
        df_copy = df_copy.rename(columns=column_mapping)
        
        # 确保时间列为datetime类型
        if 'timestamp' in df_copy.columns:
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        
        # 排序和去重
        if 'timestamp' in df_copy.columns:
            df_copy = df_copy.sort_values('timestamp').drop_duplicates('timestamp')
        
        # 数据类型优化
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df_copy.columns:
                # 使用适当的数据类型减少内存使用
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                
                # 价格列使用float32（足够精度且节省空间）
                if col in ['open', 'high', 'low', 'close']:
                    df_copy[col] = df_copy[col].astype('float32')
                # 成交量使用float64（可能很大）
                elif col == 'volume':
                    df_copy[col] = df_copy[col].astype('float64')
        
        return df_copy
    
    def _get_kline_file_path(self, symbol: str, timeframe: str) -> Path:
        """生成K线数据文件路径"""
        filename = f"{symbol}_{timeframe}_klines.parquet"
        return self.historical_path / 'klines' / filename
    
    async def _save_single_file(self, df: pd.DataFrame, file_path: Path, 
                              symbol: str, timeframe: str) -> Dict[str, Any]:
        """保存单个文件"""
        
        def _write_parquet():
            # 转换为Arrow表以提高性能
            table = pa.Table.from_pandas(df)
            
            # 写入Parquet文件，使用高压缩比
            pq.write_table(
                table,
                file_path,
                compression=self.config.compression,
                use_dictionary=True,  # 使用字典编码
                write_statistics=True,  # 写入统计信息用于查询优化
                row_group_size=50000   # 优化行组大小
            )
            
            return file_path.stat().st_size
        
        # 在线程池中执行I/O操作
        loop = asyncio.get_event_loop()
        file_size = await loop.run_in_executor(self.thread_pool, _write_parquet)
        
        # 计算压缩比
        uncompressed_size = df.memory_usage(deep=True).sum()
        compression_ratio = file_size / uncompressed_size if uncompressed_size > 0 else 0
        
        return {
            'file_size': file_size,
            'compression_ratio': compression_ratio
        }
    
    async def _save_partitioned_data(self, df: pd.DataFrame, base_path: Path,
                                   symbol: str, timeframe: str) -> Dict[str, Any]:
        """分片保存大数据集"""
        
        if 'timestamp' not in df.columns:
            # 如果没有时间列，按行数分片
            return await self._save_by_chunks(df, base_path)
        
        # 按日期分片
        df['date'] = df['timestamp'].dt.date
        total_size = 0
        file_count = 0
        
        for date, group in df.groupby('date'):
            if len(group) == 0:
                continue
            
            # 生成分片文件路径
            date_str = date.strftime('%Y%m%d')
            partition_path = base_path.parent / f"{symbol}_{timeframe}_{date_str}.parquet"
            
            # 删除临时date列
            group_clean = group.drop('date', axis=1)
            
            result = await self._save_single_file(group_clean, partition_path, symbol, timeframe)
            total_size += result['file_size']
            file_count += 1
        
        uncompressed_size = df.memory_usage(deep=True).sum()
        compression_ratio = total_size / uncompressed_size if uncompressed_size > 0 else 0
        
        return {
            'file_size': total_size,
            'file_count': file_count,
            'compression_ratio': compression_ratio
        }
    
    async def load_kline_data(self, symbol: str, timeframe: str,
                            start_date: str = None, end_date: str = None,
                            columns: List[str] = None) -> pd.DataFrame:
        """
        加载K线数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间框架
            start_date: 开始日期
            end_date: 结束日期
            columns: 需要的列名列表
            
        Returns:
            K线数据DataFrame
        """
        start_time = datetime.now()
        
        try:
            # 检查缓存
            cache_key = f"{symbol}_{timeframe}_{start_date}_{end_date}"
            if cache_key in self.cache:
                self.cache_hits += 1
                df = self.cache[cache_key].copy()
                
                if columns:
                    available_columns = [col for col in columns if col in df.columns]
                    df = df[available_columns]
                
                read_time = (datetime.now() - start_time).total_seconds() * 1000
                self.read_times.append(read_time)
                
                return df
            
            self.cache_misses += 1
            
            # 查找相关文件
            file_paths = await self._find_data_files(symbol, timeframe, start_date, end_date)
            
            if not file_paths:
                self.logger.warning(f"未找到数据文件: {symbol} {timeframe}")
                return pd.DataFrame()
            
            # 并行加载多个文件
            dfs = await self._load_multiple_files(file_paths, columns)
            
            if not dfs:
                return pd.DataFrame()
            
            # 合并数据
            df = pd.concat(dfs, ignore_index=True)
            
            # 时间范围过滤
            if start_date and end_date and 'timestamp' in df.columns:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                df = df[(df['timestamp'] >= start_dt) & (df['timestamp'] <= end_dt)]
            
            # 排序
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            # 缓存管理
            await self._manage_cache(cache_key, df)
            
            # 记录性能指标
            read_time = (datetime.now() - start_time).total_seconds() * 1000
            self.read_times.append(read_time)
            
            self.logger.info(f"K线数据加载完成: {symbol} {timeframe}, "
                           f"{len(df)}条记录, 耗时{read_time:.1f}ms")
            
            return df
            
        except Exception as e:
            self.logger.error(f"K线数据加载失败: {symbol} {timeframe} - {e}")
            return pd.DataFrame()
    
    async def _find_data_files(self, symbol: str, timeframe: str,
                             start_date: str = None, end_date: str = None) -> List[Path]:
        """查找匹配的数据文件"""
        
        def _query_metadata():
            cursor = self.metadata_db.cursor()
            
            query = '''
                SELECT file_path FROM file_metadata 
                WHERE symbol = ? AND timeframe = ? AND data_type = 'kline'
            '''
            params = [symbol, timeframe]
            
            if start_date and end_date:
                query += ' AND end_date >= ? AND start_date <= ?'
                params.extend([start_date, end_date])
            
            cursor.execute(query, params)
            results = cursor.fetchall()
            
            return [Path(row[0]) for row in results if Path(row[0]).exists()]
        
        loop = asyncio.get_event_loop()
        file_paths = await loop.run_in_executor(self.thread_pool, _query_metadata)
        
        # 如果元数据查询为空，回退到文件系统扫描
        if not file_paths:
            file_paths = await self._scan_filesystem(symbol, timeframe)
        
        return file_paths
    
    async def _scan_filesystem(self, symbol: str, timeframe: str) -> List[Path]:
        """扫描文件系统查找数据文件"""
        klines_dir = self.historical_path / 'klines'
        
        def _scan():
            patterns = [
                f"{symbol}_{timeframe}_*.parquet",
                f"{symbol}_{timeframe}_klines.parquet"
            ]
            
            files = []
            for pattern in patterns:
                files.extend(klines_dir.glob(pattern))
            
            return files
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, _scan)
    
    async def _load_multiple_files(self, file_paths: List[Path], 
                                 columns: List[str] = None) -> List[pd.DataFrame]:
        """并行加载多个文件"""
        
        def _load_file(file_path: Path) -> pd.DataFrame:
            try:
                if columns:
                    # 只读取需要的列
                    df = pd.read_parquet(file_path, columns=columns)
                else:
                    df = pd.read_parquet(file_path)
                
                return df
                
            except Exception as e:
                self.logger.error(f"文件加载失败: {file_path} - {e}")
                return pd.DataFrame()
        
        # 并行加载
        loop = asyncio.get_event_loop()
        tasks = [
            loop.run_in_executor(self.thread_pool, _load_file, file_path)
            for file_path in file_paths
        ]
        
        dfs = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤掉异常和空DataFrame
        valid_dfs = [
            df for df in dfs 
            if isinstance(df, pd.DataFrame) and not df.empty
        ]
        
        return valid_dfs
    
    async def _manage_cache(self, cache_key: str, df: pd.DataFrame):
        """管理内存缓存"""
        data_size = df.memory_usage(deep=True).sum()
        
        # 如果数据太大，不缓存
        if data_size > self.cache_size_limit * 0.5:
            return
        
        # 缓存空间不足时清理旧数据
        while self.current_cache_size + data_size > self.cache_size_limit and self.cache:
            # LRU策略：删除最少使用的缓存项
            oldest_key = next(iter(self.cache))
            old_df = self.cache.pop(oldest_key)
            self.current_cache_size -= old_df.memory_usage(deep=True).sum()
        
        # 添加到缓存
        self.cache[cache_key] = df.copy()
        self.current_cache_size += data_size
    
    async def _update_metadata(self, symbol: str, timeframe: str, data_type: str,
                             file_path: Path, start_date: str, end_date: str,
                             record_count: int, file_size: int):
        """更新元数据"""
        
        def _update():
            cursor = self.metadata_db.cursor()
            
            now = datetime.now().isoformat()
            
            cursor.execute('''
                INSERT OR REPLACE INTO file_metadata 
                (symbol, timeframe, data_type, file_path, start_date, end_date, 
                 record_count, file_size, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, timeframe, data_type, str(file_path), start_date, end_date,
                  record_count, file_size, now, now))
            
            self.metadata_db.commit()
        
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self.thread_pool, _update)
    
    def get_data_path(self, symbol: str, timeframe: str, data_type: str) -> Path:
        """获取数据文件路径"""
        if data_type == 'kline':
            return self._get_kline_file_path(symbol, timeframe)
        elif data_type == 'funding':
            return self.historical_path / 'funding' / f"{symbol}_funding.parquet"
        elif data_type == 'trades':
            return self.historical_path / 'trades' / f"{symbol}_trades.parquet"
        elif data_type == 'depth':
            return self.historical_path / 'orderbooks' / f"{symbol}_depth.zstd"
        else:
            return self.historical_path / f"{symbol}_{data_type}.parquet"
    
    async def backup_data(self, incremental: bool = True) -> Dict[str, Any]:
        """数据备份"""
        if not self.config.backup_enabled:
            return {'status': 'disabled'}
        
        backup_start = datetime.now()
        
        try:
            backup_dir = self.backup_path / backup_start.strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(exist_ok=True)
            
            if incremental:
                # 增量备份：只备份最近修改的文件
                cutoff_time = backup_start - timedelta(days=1)
                result = await self._incremental_backup(backup_dir, cutoff_time)
            else:
                # 全量备份
                result = await self._full_backup(backup_dir)
            
            backup_time = (datetime.now() - backup_start).total_seconds()
            
            self.logger.info(f"数据备份完成: {result['file_count']}个文件, "
                           f"耗时{backup_time:.1f}秒")
            
            return {
                'status': 'success',
                'backup_path': str(backup_dir),
                'file_count': result['file_count'],
                'total_size_mb': result['total_size'] / (1024 * 1024),
                'backup_time_seconds': backup_time,
                'backup_type': 'incremental' if incremental else 'full'
            }
            
        except Exception as e:
            self.logger.error(f"数据备份失败: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    async def _incremental_backup(self, backup_dir: Path, cutoff_time: datetime) -> Dict[str, Any]:
        """增量备份"""
        file_count = 0
        total_size = 0
        
        def _copy_if_newer(source_path: Path, target_path: Path):
            nonlocal file_count, total_size
            
            if (source_path.exists() and 
                datetime.fromtimestamp(source_path.stat().st_mtime) > cutoff_time):
                
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, target_path)
                
                file_count += 1
                total_size += source_path.stat().st_size
        
        # 备份历史数据
        for source_file in self.historical_path.rglob('*.parquet'):
            relative_path = source_file.relative_to(self.data_root)
            target_file = backup_dir / relative_path
            _copy_if_newer(source_file, target_file)
        
        # 备份元数据
        metadata_files = self.metadata_path.glob('*')
        for source_file in metadata_files:
            relative_path = source_file.relative_to(self.data_root)
            target_file = backup_dir / relative_path
            _copy_if_newer(source_file, target_file)
        
        return {'file_count': file_count, 'total_size': total_size}
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """获取存储使用统计"""
        
        def _calculate_stats():
            stats = {
                'total_size_mb': 0,
                'file_count': 0,
                'directories': {},
                'compression_ratio': 0,
                'cache_stats': {
                    'cache_size_mb': self.current_cache_size / (1024 * 1024),
                    'cache_items': len(self.cache),
                    'hit_ratio': self.cache_hits / (self.cache_hits + self.cache_misses) 
                               if (self.cache_hits + self.cache_misses) > 0 else 0
                }
            }
            
            # 计算各目录的使用量
            for directory in [self.historical_path, self.realtime_path, self.backup_path]:
                if directory.exists():
                    dir_size = 0
                    dir_files = 0
                    
                    for file_path in directory.rglob('*'):
                        if file_path.is_file():
                            size = file_path.stat().st_size
                            dir_size += size
                            dir_files += 1
                    
                    stats['directories'][directory.name] = {
                        'size_mb': dir_size / (1024 * 1024),
                        'file_count': dir_files
                    }
                    
                    stats['total_size_mb'] += dir_size / (1024 * 1024)
                    stats['file_count'] += dir_files
            
            return stats
        
        loop = asyncio.get_event_loop()
        stats = await loop.run_in_executor(self.thread_pool, _calculate_stats)
        
        # 添加性能指标
        if self.read_times:
            stats['performance'] = {
                'avg_read_latency_ms': np.mean(self.read_times),
                'avg_write_latency_ms': np.mean(self.write_times) if self.write_times else 0,
                'total_reads': len(self.read_times),
                'total_writes': len(self.write_times)
            }
        
        return stats
    
    async def optimize_storage(self) -> Dict[str, Any]:
        """优化存储性能"""
        self.logger.info("开始存储优化")
        
        optimization_results = {
            'actions_taken': [],
            'space_saved_mb': 0,
            'performance_improvement': {}
        }
        
        try:
            # 1. 清理旧的临时文件
            temp_cleaned = await self._clean_temp_files()
            if temp_cleaned['files_removed'] > 0:
                optimization_results['actions_taken'].append('清理临时文件')
                optimization_results['space_saved_mb'] += temp_cleaned['space_saved_mb']
            
            # 2. 重新压缩低效文件
            recompressed = await self._recompress_files()
            if recompressed['files_processed'] > 0:
                optimization_results['actions_taken'].append('重新压缩文件')
                optimization_results['space_saved_mb'] += recompressed['space_saved_mb']
            
            # 3. 合并小文件
            merged = await self._merge_small_files()
            if merged['files_merged'] > 0:
                optimization_results['actions_taken'].append('合并小文件')
                optimization_results['performance_improvement']['read_latency'] = 'improved'
            
            # 4. 清理缓存
            self._clear_old_cache()
            optimization_results['actions_taken'].append('清理内存缓存')
            
            self.logger.info(f"存储优化完成，节省空间{optimization_results['space_saved_mb']:.1f}MB")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"存储优化失败: {e}")
            return {'error': str(e)}
    
    async def _clean_temp_files(self) -> Dict[str, Any]:
        """清理临时文件"""
        files_removed = 0
        space_saved = 0
        
        # 清理.tmp文件
        for temp_file in self.data_root.rglob('*.tmp'):
            size = temp_file.stat().st_size
            temp_file.unlink()
            files_removed += 1
            space_saved += size
        
        return {
            'files_removed': files_removed,
            'space_saved_mb': space_saved / (1024 * 1024)
        }
    
    def _clear_old_cache(self):
        """清理内存缓存"""
        self.cache.clear()
        self.current_cache_size = 0
        
        # 重置性能计数器
        if len(self.read_times) > 1000:
            self.read_times = self.read_times[-100:]  # 保留最近100次
        if len(self.write_times) > 1000:
            self.write_times = self.write_times[-100:]
    
    async def cleanup(self):
        """清理资源"""
        self.logger.info("清理存储管理器资源")
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        
        # 关闭数据库连接
        if self.metadata_db:
            self.metadata_db.close()
        
        # 清理缓存
        self.cache.clear()
        self.current_cache_size = 0