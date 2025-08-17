"""
High Performance Storage and Access Optimization System
高性能存储和访问优化系统 - 为DipMaster Trading System优化
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import duckdb
import polars as pl
import zstandard as zstd
import lz4.frame
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import queue
import time
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import mmap
import pickle
import redis
from collections import defaultdict, OrderedDict
import sqlite3
import h5py
import zarr
import psutil
import gc
import warnings

warnings.filterwarnings('ignore')

class StorageFormat(Enum):
    """存储格式"""
    PARQUET = "parquet"
    ARROW = "arrow"
    FEATHER = "feather"
    HDF5 = "hdf5"
    ZARR = "zarr"
    CSV = "csv"
    PICKLE = "pickle"

class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"
    SNAPPY = "snappy"
    GZIP = "gzip"
    ZSTD = "zstd"
    LZ4 = "lz4"
    BROTLI = "brotli"

class IndexStrategy(Enum):
    """索引策略"""
    TIMESTAMP = "timestamp"
    SYMBOL = "symbol"
    COMPOSITE = "composite"
    HASH = "hash"
    BTREE = "btree"

@dataclass
class StorageConfig:
    """存储配置"""
    format: StorageFormat = StorageFormat.PARQUET
    compression: CompressionType = CompressionType.ZSTD
    chunk_size: int = 100000
    index_strategy: IndexStrategy = IndexStrategy.TIMESTAMP
    enable_memory_mapping: bool = True
    enable_caching: bool = True
    cache_size_mb: int = 1024
    enable_async_writes: bool = True
    enable_parallel_processing: bool = True
    max_workers: int = 4

@dataclass
class QueryPerformanceMetrics:
    """查询性能指标"""
    query_time_ms: float
    data_size_mb: float
    rows_processed: int
    cache_hit_rate: float
    compression_ratio: float
    memory_usage_mb: float

class HighPerformanceStorage:
    """高性能存储系统"""
    
    def __init__(self, config: StorageConfig = None):
        self.config = config or StorageConfig()
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 存储路径
        self.base_path = Path("data/high_performance_storage")
        self.base_path.mkdir(exist_ok=True, parents=True)
        
        # 索引路径
        self.index_path = self.base_path / "indexes"
        self.index_path.mkdir(exist_ok=True, parents=True)
        
        # 缓存路径
        self.cache_path = self.base_path / "cache"
        self.cache_path.mkdir(exist_ok=True, parents=True)
        
        # 内存缓存
        self.memory_cache = OrderedDict()
        self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0}
        self.cache_lock = threading.RLock()
        
        # DuckDB连接 - 用于高性能分析
        self.duckdb_conn = duckdb.connect(':memory:')
        self.setup_duckdb()
        
        # Redis连接 - 用于分布式缓存
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=2, decode_responses=False)
            self.redis_client.ping()
            self.redis_enabled = True
        except:
            self.logger.warning("Redis未连接，禁用分布式缓存")
            self.redis_enabled = False
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers // 2)
        
        # 压缩器
        self.compressors = self.setup_compressors()
        
        # 性能统计
        self.performance_stats = defaultdict(list)
        
        # 数据分区管理
        self.partition_manager = PartitionManager(self.base_path)
        
        # 索引管理器
        self.index_manager = IndexManager(self.index_path)
        
        # 查询优化器
        self.query_optimizer = QueryOptimizer()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler('logs/high_performance_storage.log'),
                logging.StreamHandler()
            ]
        )
    
    def setup_duckdb(self):
        """设置DuckDB"""
        # 配置DuckDB性能参数
        self.duckdb_conn.execute("SET memory_limit='8GB'")
        self.duckdb_conn.execute("SET threads=4")
        self.duckdb_conn.execute("SET enable_progress_bar=false")
        
    def setup_compressors(self) -> Dict:
        """设置压缩器"""
        return {
            CompressionType.ZSTD: zstd.ZstdCompressor(level=3),
            CompressionType.LZ4: None,  # 使用lz4.frame模块
        }
    
    async def store_dataframe(self, 
                            df: pd.DataFrame, 
                            symbol: str, 
                            timeframe: str = '5m',
                            partition_by: str = 'date') -> str:
        """高性能存储DataFrame"""
        start_time = time.time()
        
        try:
            # 数据预处理
            df_processed = self.preprocess_dataframe(df)
            
            # 生成存储键
            storage_key = f"{symbol}_{timeframe}_{partition_by}"
            
            # 选择存储策略
            if len(df_processed) > 1000000:  # 大数据集使用分区存储
                file_path = await self.store_large_dataset(df_processed, storage_key, partition_by)
            else:
                file_path = await self.store_regular_dataset(df_processed, storage_key)
            
            # 创建索引
            await self.index_manager.create_index(file_path, df_processed, self.config.index_strategy)
            
            # 更新缓存
            if self.config.enable_caching:
                await self.update_cache(storage_key, df_processed)
            
            # 记录性能指标
            elapsed_time = (time.time() - start_time) * 1000
            self.record_performance('store', {
                'time_ms': elapsed_time,
                'rows': len(df_processed),
                'size_mb': df_processed.memory_usage(deep=True).sum() / 1024 / 1024,
                'symbol': symbol,
                'timeframe': timeframe
            })
            
            self.logger.info(f"存储完成 {symbol}_{timeframe}: {len(df_processed)} 行, {elapsed_time:.2f}ms")
            
            return str(file_path)
            
        except Exception as e:
            self.logger.error(f"存储 {symbol}_{timeframe} 失败: {e}")
            raise
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理DataFrame以优化存储"""
        df_processed = df.copy()
        
        # 优化数据类型
        for col in df_processed.columns:
            if col in ['open', 'high', 'low', 'close']:
                df_processed[col] = pd.to_numeric(df_processed[col], downcast='float')
            elif col == 'volume':
                df_processed[col] = pd.to_numeric(df_processed[col], downcast='integer')
        
        # 确保时间索引
        if not isinstance(df_processed.index, pd.DatetimeIndex):
            if 'timestamp' in df_processed.columns:
                df_processed.set_index('timestamp', inplace=True)
            df_processed.index = pd.to_datetime(df_processed.index)
        
        # 排序以优化查询
        df_processed = df_processed.sort_index()
        
        # 去重
        df_processed = df_processed[~df_processed.index.duplicated(keep='first')]
        
        return df_processed
    
    async def store_large_dataset(self, 
                                df: pd.DataFrame, 
                                storage_key: str, 
                                partition_by: str) -> Path:
        """存储大数据集（分区）"""
        
        # 创建分区目录
        partition_dir = self.base_path / "partitioned" / storage_key
        partition_dir.mkdir(exist_ok=True, parents=True)
        
        # 按时间分区
        if partition_by == 'date':
            partitions = df.groupby(df.index.date)
        elif partition_by == 'month':
            partitions = df.groupby(df.index.to_period('M'))
        elif partition_by == 'week':
            partitions = df.groupby(df.index.to_period('W'))
        else:
            partitions = [('all', df)]
        
        partition_files = []
        
        # 并行写入分区
        tasks = []
        for partition_key, partition_df in partitions:
            task = self.write_partition_async(partition_dir, partition_key, partition_df)
            tasks.append(task)
        
        partition_files = await asyncio.gather(*tasks)
        
        # 创建分区元数据
        metadata = {
            'storage_key': storage_key,
            'partition_by': partition_by,
            'partitions': len(partition_files),
            'total_rows': len(df),
            'files': [str(f) for f in partition_files if f],
            'created_at': datetime.now().isoformat(),
            'format': self.config.format.value,
            'compression': self.config.compression.value
        }
        
        metadata_file = partition_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return partition_dir
    
    async def write_partition_async(self, 
                                  partition_dir: Path, 
                                  partition_key: Any, 
                                  partition_df: pd.DataFrame) -> Optional[Path]:
        """异步写入分区"""
        try:
            filename = f"partition_{partition_key}.{self.config.format.value}"
            file_path = partition_dir / filename
            
            # 在线程池中执行写入
            await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                self.write_dataframe_to_file,
                partition_df,
                file_path
            )
            
            return file_path
            
        except Exception as e:
            self.logger.error(f"写入分区 {partition_key} 失败: {e}")
            return None
    
    async def store_regular_dataset(self, df: pd.DataFrame, storage_key: str) -> Path:
        """存储常规数据集"""
        filename = f"{storage_key}.{self.config.format.value}"
        file_path = self.base_path / filename
        
        # 异步写入
        await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self.write_dataframe_to_file,
            df,
            file_path
        )
        
        return file_path
    
    def write_dataframe_to_file(self, df: pd.DataFrame, file_path: Path):
        """写入DataFrame到文件"""
        if self.config.format == StorageFormat.PARQUET:
            table = pa.Table.from_pandas(df)
            
            # 配置写入选项
            write_options = {
                'compression': self.config.compression.value,
                'use_dictionary': True,
                'write_statistics': True,
                'data_page_size': 64 * 1024,  # 64KB
                'flavor': 'spark'  # 兼容性
            }
            
            pq.write_table(table, file_path, **write_options)
            
        elif self.config.format == StorageFormat.FEATHER:
            df.to_feather(file_path, compression=self.config.compression.value)
            
        elif self.config.format == StorageFormat.HDF5:
            with h5py.File(file_path, 'w') as f:
                # 存储数据
                for col in df.columns:
                    if df[col].dtype == 'object':
                        f.create_dataset(col, data=df[col].astype(str).values, compression='gzip')
                    else:
                        f.create_dataset(col, data=df[col].values, compression='gzip')
                
                # 存储索引
                f.create_dataset('index', data=df.index.values, compression='gzip')
                
                # 存储元数据
                f.attrs['columns'] = list(df.columns)
                f.attrs['shape'] = df.shape
                f.attrs['created_at'] = datetime.now().isoformat()
        
        elif self.config.format == StorageFormat.ZARR:
            store = zarr.DirectoryStore(str(file_path))
            root = zarr.group(store=store, overwrite=True)
            
            # 存储每列数据
            for col in df.columns:
                if df[col].dtype == 'object':
                    root.create_dataset(col, data=df[col].astype(str).values, compressor=zarr.Blosc(cname='zstd'))
                else:
                    root.create_dataset(col, data=df[col].values, compressor=zarr.Blosc(cname='zstd'))
            
            # 存储索引
            root.create_dataset('index', data=df.index.values, compressor=zarr.Blosc(cname='zstd'))
            
            # 元数据
            root.attrs['columns'] = list(df.columns)
            root.attrs['shape'] = df.shape
            
        else:
            # 默认使用pickle
            with open(file_path, 'wb') as f:
                if self.config.compression == CompressionType.ZSTD:
                    compressed_data = self.compressors[CompressionType.ZSTD].compress(pickle.dumps(df))
                    f.write(compressed_data)
                elif self.config.compression == CompressionType.LZ4:
                    compressed_data = lz4.frame.compress(pickle.dumps(df))
                    f.write(compressed_data)
                else:
                    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    async def load_dataframe(self, 
                           symbol: str, 
                           timeframe: str = '5m',
                           start_time: datetime = None,
                           end_time: datetime = None,
                           columns: List[str] = None) -> pd.DataFrame:
        """高性能加载DataFrame"""
        start_load_time = time.time()
        
        try:
            # 生成存储键
            storage_key = f"{symbol}_{timeframe}"
            
            # 检查缓存
            if self.config.enable_caching:
                cached_df = await self.get_from_cache(storage_key, start_time, end_time)
                if cached_df is not None:
                    return cached_df
            
            # 查找数据文件
            data_files = await self.find_data_files(storage_key)
            
            if not data_files:
                self.logger.warning(f"未找到数据文件: {storage_key}")
                return pd.DataFrame()
            
            # 加载数据
            if len(data_files) == 1 and data_files[0].is_file():
                # 单文件加载
                df = await self.load_single_file(data_files[0], start_time, end_time, columns)
            else:
                # 多文件/分区加载
                df = await self.load_partitioned_data(data_files, start_time, end_time, columns)
            
            # 时间范围过滤
            if start_time or end_time:
                df = self.filter_by_time_range(df, start_time, end_time)
            
            # 列过滤
            if columns:
                available_columns = [col for col in columns if col in df.columns]
                df = df[available_columns]
            
            # 更新缓存
            if self.config.enable_caching and len(df) < 100000:  # 只缓存小数据集
                await self.update_cache(storage_key, df)
            
            # 记录性能指标
            elapsed_time = (time.time() - start_load_time) * 1000
            self.record_performance('load', {
                'time_ms': elapsed_time,
                'rows': len(df),
                'size_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                'symbol': symbol,
                'timeframe': timeframe
            })
            
            self.logger.info(f"加载完成 {symbol}_{timeframe}: {len(df)} 行, {elapsed_time:.2f}ms")
            
            return df
            
        except Exception as e:
            self.logger.error(f"加载 {symbol}_{timeframe} 失败: {e}")
            return pd.DataFrame()
    
    async def find_data_files(self, storage_key: str) -> List[Path]:
        """查找数据文件"""
        files = []
        
        # 查找单文件
        for format_type in StorageFormat:
            file_path = self.base_path / f"{storage_key}.{format_type.value}"
            if file_path.exists():
                files.append(file_path)
                break
        
        # 查找分区文件
        partition_dir = self.base_path / "partitioned" / storage_key
        if partition_dir.exists():
            metadata_file = partition_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                for file_path_str in metadata.get('files', []):
                    file_path = Path(file_path_str)
                    if file_path.exists():
                        files.append(file_path)
        
        return files
    
    async def load_single_file(self, 
                             file_path: Path,
                             start_time: datetime = None,
                             end_time: datetime = None,
                             columns: List[str] = None) -> pd.DataFrame:
        """加载单个文件"""
        
        # 在线程池中执行加载
        df = await asyncio.get_event_loop().run_in_executor(
            self.thread_pool,
            self.read_dataframe_from_file,
            file_path,
            start_time,
            end_time,
            columns
        )
        
        return df
    
    def read_dataframe_from_file(self,
                               file_path: Path,
                               start_time: datetime = None,
                               end_time: datetime = None,
                               columns: List[str] = None) -> pd.DataFrame:
        """从文件读取DataFrame"""
        
        file_format = StorageFormat(file_path.suffix[1:]) if file_path.suffix[1:] in [f.value for f in StorageFormat] else StorageFormat.PARQUET
        
        if file_format == StorageFormat.PARQUET:
            # 使用PyArrow读取，支持列式过滤
            table = pq.read_table(file_path, columns=columns)
            
            # 时间过滤
            if start_time or end_time:
                filters = []
                if start_time:
                    filters.append(('timestamp', '>=', start_time))
                if end_time:
                    filters.append(('timestamp', '<=', end_time))
                
                if filters:
                    try:
                        table = table.filter(pc.and_(*[pc.greater_equal(pc.field('timestamp'), start_time) if start_time else pc.literal(True),
                                                      pc.less_equal(pc.field('timestamp'), end_time) if end_time else pc.literal(True)]))
                    except:
                        pass  # 如果过滤失败，返回全部数据
            
            df = table.to_pandas()
            
        elif file_format == StorageFormat.FEATHER:
            df = pd.read_feather(file_path, columns=columns)
            
        elif file_format == StorageFormat.HDF5:
            df = self.read_hdf5_file(file_path, columns)
            
        elif file_format == StorageFormat.ZARR:
            df = self.read_zarr_file(file_path, columns)
            
        else:
            # Pickle格式
            with open(file_path, 'rb') as f:
                if self.config.compression == CompressionType.ZSTD:
                    compressed_data = f.read()
                    decompressed_data = zstd.decompress(compressed_data)
                    df = pickle.loads(decompressed_data)
                elif self.config.compression == CompressionType.LZ4:
                    compressed_data = f.read()
                    decompressed_data = lz4.frame.decompress(compressed_data)
                    df = pickle.loads(decompressed_data)
                else:
                    df = pickle.load(f)
        
        return df
    
    def read_hdf5_file(self, file_path: Path, columns: List[str] = None) -> pd.DataFrame:
        """读取HDF5文件"""
        with h5py.File(file_path, 'r') as f:
            # 读取列名
            available_columns = f.attrs['columns'] if 'columns' in f.attrs else list(f.keys())
            
            # 确定要读取的列
            read_columns = columns if columns else [col for col in available_columns if col != 'index']
            
            # 读取数据
            data = {}
            for col in read_columns:
                if col in f:
                    data[col] = f[col][:]
            
            # 读取索引
            if 'index' in f:
                index = pd.to_datetime(f['index'][:])
            else:
                index = range(len(data[read_columns[0]]))
            
            df = pd.DataFrame(data, index=index)
        
        return df
    
    def read_zarr_file(self, file_path: Path, columns: List[str] = None) -> pd.DataFrame:
        """读取Zarr文件"""
        root = zarr.open(str(file_path), mode='r')
        
        # 获取可用列
        available_columns = root.attrs.get('columns', list(root.keys()))
        
        # 确定要读取的列
        read_columns = columns if columns else [col for col in available_columns if col != 'index']
        
        # 读取数据
        data = {}
        for col in read_columns:
            if col in root:
                data[col] = root[col][:]
        
        # 读取索引
        if 'index' in root:
            index = pd.to_datetime(root['index'][:])
        else:
            index = range(len(data[read_columns[0]]))
        
        df = pd.DataFrame(data, index=index)
        
        return df
    
    async def load_partitioned_data(self,
                                  partition_files: List[Path],
                                  start_time: datetime = None,
                                  end_time: datetime = None,
                                  columns: List[str] = None) -> pd.DataFrame:
        """加载分区数据"""
        
        # 并行加载所有分区
        tasks = []
        for file_path in partition_files:
            task = self.load_single_file(file_path, start_time, end_time, columns)
            tasks.append(task)
        
        partition_dfs = await asyncio.gather(*tasks)
        
        # 合并分区数据
        valid_dfs = [df for df in partition_dfs if not df.empty]
        
        if not valid_dfs:
            return pd.DataFrame()
        
        # 使用Polars进行高效合并（如果数据量大）
        if sum(len(df) for df in valid_dfs) > 1000000:
            return self.merge_large_dataframes(valid_dfs)
        else:
            return pd.concat(valid_dfs, ignore_index=False).sort_index()
    
    def merge_large_dataframes(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """合并大型DataFrame"""
        try:
            # 转换为Polars进行高效合并
            pl_dfs = []
            for df in dfs:
                pl_df = pl.from_pandas(df.reset_index())
                pl_dfs.append(pl_df)
            
            # 合并
            merged_pl = pl.concat(pl_dfs)
            
            # 排序
            if 'timestamp' in merged_pl.columns:
                merged_pl = merged_pl.sort('timestamp')
            
            # 转回Pandas
            merged_df = merged_pl.to_pandas()
            
            if 'timestamp' in merged_df.columns:
                merged_df.set_index('timestamp', inplace=True)
            
            return merged_df
            
        except Exception as e:
            self.logger.warning(f"Polars合并失败，使用Pandas: {e}")
            return pd.concat(dfs, ignore_index=False).sort_index()
    
    def filter_by_time_range(self, 
                           df: pd.DataFrame,
                           start_time: datetime = None,
                           end_time: datetime = None) -> pd.DataFrame:
        """按时间范围过滤"""
        if df.empty:
            return df
        
        mask = pd.Series(True, index=df.index)
        
        if start_time:
            mask &= (df.index >= start_time)
        
        if end_time:
            mask &= (df.index <= end_time)
        
        return df[mask]
    
    async def get_from_cache(self, 
                           cache_key: str,
                           start_time: datetime = None,
                           end_time: datetime = None) -> Optional[pd.DataFrame]:
        """从缓存获取数据"""
        
        # 内存缓存
        with self.cache_lock:
            if cache_key in self.memory_cache:
                self.cache_stats['hits'] += 1
                df = self.memory_cache[cache_key]
                
                # 移到最前面（LRU）
                self.memory_cache.move_to_end(cache_key)
                
                # 时间过滤
                if start_time or end_time:
                    df = self.filter_by_time_range(df, start_time, end_time)
                
                return df.copy()
        
        # Redis缓存
        if self.redis_enabled:
            try:
                cached_data = self.redis_client.get(f"df:{cache_key}")
                if cached_data:
                    df = pickle.loads(cached_data)
                    self.cache_stats['hits'] += 1
                    
                    # 添加到内存缓存
                    await self.update_memory_cache(cache_key, df)
                    
                    # 时间过滤
                    if start_time or end_time:
                        df = self.filter_by_time_range(df, start_time, end_time)
                    
                    return df
            except Exception as e:
                self.logger.warning(f"Redis缓存读取失败: {e}")
        
        self.cache_stats['misses'] += 1
        return None
    
    async def update_cache(self, cache_key: str, df: pd.DataFrame):
        """更新缓存"""
        # 更新内存缓存
        await self.update_memory_cache(cache_key, df)
        
        # 更新Redis缓存
        if self.redis_enabled and len(df) < 50000:  # 只缓存小数据集到Redis
            try:
                cached_data = pickle.dumps(df)
                self.redis_client.setex(f"df:{cache_key}", 3600, cached_data)  # 1小时TTL
            except Exception as e:
                self.logger.warning(f"Redis缓存写入失败: {e}")
    
    async def update_memory_cache(self, cache_key: str, df: pd.DataFrame):
        """更新内存缓存"""
        with self.cache_lock:
            # 检查缓存大小限制
            df_size_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            max_size_mb = self.config.cache_size_mb
            
            # 清理缓存以腾出空间
            current_size_mb = self.get_cache_size_mb()
            
            while current_size_mb + df_size_mb > max_size_mb and self.memory_cache:
                # 移除最旧的项目
                oldest_key, oldest_df = self.memory_cache.popitem(last=False)
                self.cache_stats['evictions'] += 1
                current_size_mb -= oldest_df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # 添加新项目
            self.memory_cache[cache_key] = df.copy()
    
    def get_cache_size_mb(self) -> float:
        """获取缓存大小（MB）"""
        total_size = 0
        for df in self.memory_cache.values():
            total_size += df.memory_usage(deep=True).sum()
        return total_size / 1024 / 1024
    
    def record_performance(self, operation: str, metrics: Dict):
        """记录性能指标"""
        self.performance_stats[operation].append({
            'timestamp': datetime.now(),
            **metrics
        })
        
        # 保持最近1000条记录
        if len(self.performance_stats[operation]) > 1000:
            self.performance_stats[operation] = self.performance_stats[operation][-1000:]
    
    async def optimize_storage(self, symbol: str, timeframe: str) -> Dict:
        """存储优化"""
        try:
            storage_key = f"{symbol}_{timeframe}"
            
            # 查找数据文件
            data_files = await self.find_data_files(storage_key)
            
            if not data_files:
                return {"error": "未找到数据文件"}
            
            optimization_results = {}
            
            for file_path in data_files:
                if file_path.is_file():
                    # 分析文件
                    file_stats = self.analyze_file(file_path)
                    
                    # 优化建议
                    recommendations = self.generate_optimization_recommendations(file_stats)
                    
                    optimization_results[str(file_path)] = {
                        'current_stats': file_stats,
                        'recommendations': recommendations
                    }
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"存储优化失败: {e}")
            return {"error": str(e)}
    
    def analyze_file(self, file_path: Path) -> Dict:
        """分析文件统计信息"""
        stats = {
            'file_size_mb': file_path.stat().st_size / 1024 / 1024,
            'format': file_path.suffix[1:],
            'created_time': datetime.fromtimestamp(file_path.stat().st_ctime),
            'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime)
        }
        
        try:
            if file_path.suffix == '.parquet':
                # Parquet文件分析
                parquet_file = pq.ParquetFile(file_path)
                stats.update({
                    'num_rows': parquet_file.metadata.num_rows,
                    'num_columns': parquet_file.metadata.num_columns,
                    'num_row_groups': parquet_file.metadata.num_row_groups,
                    'compression': parquet_file.metadata.row_group(0).column(0).compression,
                    'uncompressed_size_mb': parquet_file.metadata.serialized_size / 1024 / 1024
                })
                
                # 计算压缩比
                if stats['uncompressed_size_mb'] > 0:
                    stats['compression_ratio'] = stats['file_size_mb'] / stats['uncompressed_size_mb']
                
        except Exception as e:
            self.logger.warning(f"文件分析失败 {file_path}: {e}")
        
        return stats
    
    def generate_optimization_recommendations(self, file_stats: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 压缩比建议
        if file_stats.get('compression_ratio', 1.0) > 0.5:
            recommendations.append("建议使用更高效的压缩算法（如ZSTD）")
        
        # 文件大小建议
        if file_stats.get('file_size_mb', 0) > 500:
            recommendations.append("建议将大文件分割为多个较小的分区")
        
        # 行组大小建议
        if file_stats.get('num_row_groups', 0) > 100:
            recommendations.append("行组过多，建议增加行组大小")
        
        # 格式建议
        if file_stats.get('format') == 'csv':
            recommendations.append("建议转换为Parquet格式以获得更好的性能")
        
        return recommendations
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        summary = {
            'cache_stats': self.cache_stats.copy(),
            'cache_size_mb': self.get_cache_size_mb(),
            'operations': {}
        }
        
        for operation, metrics_list in self.performance_stats.items():
            if metrics_list:
                recent_metrics = metrics_list[-100:]  # 最近100次操作
                
                times = [m['time_ms'] for m in recent_metrics]
                sizes = [m['size_mb'] for m in recent_metrics]
                rows = [m['rows'] for m in recent_metrics]
                
                summary['operations'][operation] = {
                    'count': len(recent_metrics),
                    'avg_time_ms': np.mean(times),
                    'min_time_ms': np.min(times),
                    'max_time_ms': np.max(times),
                    'avg_size_mb': np.mean(sizes),
                    'avg_rows': np.mean(rows),
                    'throughput_rows_per_second': np.mean(rows) / (np.mean(times) / 1000) if np.mean(times) > 0 else 0
                }
        
        # 缓存命中率
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        if total_requests > 0:
            summary['cache_hit_rate'] = self.cache_stats['hits'] / total_requests
        else:
            summary['cache_hit_rate'] = 0.0
        
        return summary
    
    async def cleanup(self):
        """清理资源"""
        try:
            # 关闭线程池
            self.thread_pool.shutdown(wait=True)
            self.process_pool.shutdown(wait=True)
            
            # 关闭数据库连接
            self.duckdb_conn.close()
            
            # 清理缓存
            with self.cache_lock:
                self.memory_cache.clear()
            
            self.logger.info("高性能存储系统资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")

# 辅助类
class PartitionManager:
    """分区管理器"""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.partitions_info = {}
    
    def get_partition_strategy(self, df: pd.DataFrame) -> str:
        """获取分区策略"""
        time_span = df.index.max() - df.index.min()
        
        if time_span > timedelta(days=365):
            return 'month'
        elif time_span > timedelta(days=30):
            return 'week'
        else:
            return 'date'

class IndexManager:
    """索引管理器"""
    
    def __init__(self, index_path: Path):
        self.index_path = index_path
    
    async def create_index(self, file_path: Path, df: pd.DataFrame, strategy: IndexStrategy):
        """创建索引"""
        try:
            index_name = f"{file_path.stem}_index"
            index_file = self.index_path / f"{index_name}.json"
            
            if strategy == IndexStrategy.TIMESTAMP:
                # 时间戳索引
                time_index = {
                    'min_timestamp': df.index.min().isoformat(),
                    'max_timestamp': df.index.max().isoformat(),
                    'count': len(df),
                    'file_path': str(file_path)
                }
                
                with open(index_file, 'w', encoding='utf-8') as f:
                    json.dump(time_index, f, indent=2)
                    
        except Exception as e:
            logging.getLogger(__name__).error(f"创建索引失败: {e}")

class QueryOptimizer:
    """查询优化器"""
    
    def __init__(self):
        self.query_cache = {}
    
    def optimize_query(self, query_params: Dict) -> Dict:
        """优化查询参数"""
        optimized = query_params.copy()
        
        # 如果查询范围很小，建议使用内存缓存
        if query_params.get('time_range_hours', 0) < 24:
            optimized['use_memory_cache'] = True
        
        # 如果只需要特定列，启用列式读取
        if query_params.get('columns'):
            optimized['columnar_read'] = True
        
        return optimized

# 使用示例
async def main():
    """高性能存储系统演示"""
    
    # 创建配置
    config = StorageConfig(
        format=StorageFormat.PARQUET,
        compression=CompressionType.ZSTD,
        chunk_size=100000,
        cache_size_mb=512,
        enable_async_writes=True,
        max_workers=4
    )
    
    # 初始化存储系统
    storage = HighPerformanceStorage(config)
    
    # 生成示例数据
    dates = pd.date_range('2023-01-01', '2025-08-17', freq='5min')
    df = pd.DataFrame({
        'open': np.random.uniform(50000, 60000, len(dates)),
        'high': np.random.uniform(50000, 60000, len(dates)),
        'low': np.random.uniform(45000, 55000, len(dates)),
        'close': np.random.uniform(45000, 60000, len(dates)),
        'volume': np.random.uniform(1000, 100000, len(dates))
    }, index=dates)
    
    print(f"生成示例数据: {len(df)} 行")
    
    # 存储数据
    start_time = time.time()
    file_path = await storage.store_dataframe(df, 'BTCUSDT', '5m')
    store_time = time.time() - start_time
    print(f"存储完成: {store_time:.2f}秒, 文件: {file_path}")
    
    # 加载数据
    start_time = time.time()
    loaded_df = await storage.load_dataframe(
        'BTCUSDT', '5m',
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 12, 31),
        columns=['close', 'volume']
    )
    load_time = time.time() - start_time
    print(f"加载完成: {load_time:.2f}秒, 数据: {len(loaded_df)} 行")
    
    # 性能摘要
    summary = storage.get_performance_summary()
    print("\n=== 性能摘要 ===")
    print(json.dumps(summary, indent=2, default=str))
    
    # 存储优化分析
    optimization = await storage.optimize_storage('BTCUSDT', '5m')
    print("\n=== 存储优化建议 ===")
    print(json.dumps(optimization, indent=2, default=str))
    
    # 清理资源
    await storage.cleanup()

if __name__ == "__main__":
    asyncio.run(main())