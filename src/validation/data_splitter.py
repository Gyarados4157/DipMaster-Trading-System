#!/usr/bin/env python3
"""
严格数据分割器 - 解决过拟合的核心组件
Strict Data Splitter - Core Component for Overfitting Prevention

核心原则:
1. 时间顺序分割 (60% 训练 / 20% 验证 / 20% 测试)
2. 测试集绝对不可触碰原则
3. 消除选择偏差，所有币种一致验证
4. 数据泄漏防护

Author: DipMaster Trading Team
Date: 2025-08-15
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class DataSplitConfig:
    """数据分割配置"""
    train_ratio: float = 0.60
    val_ratio: float = 0.20
    test_ratio: float = 0.20
    min_train_samples: int = 50000  # 最少训练样本
    min_val_samples: int = 10000    # 最少验证样本
    min_test_samples: int = 10000   # 最少测试样本
    symbols: List[str] = None       # 币种列表

@dataclass
class DataSplit:
    """数据分割结果"""
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    test_start: datetime
    test_end: datetime
    train_samples: int
    val_samples: int
    test_samples: int
    split_timestamp: datetime
    integrity_hash: str

class DataSplitter:
    """
    严格数据分割器
    
    核心功能:
    1. 时间顺序严格分割
    2. 防止数据泄漏
    3. 消除选择偏差
    4. 分割完整性验证
    """
    
    def __init__(self, config: DataSplitConfig = None):
        self.config = config or DataSplitConfig()
        self.splits: Dict[str, DataSplit] = {}
        self.lock_file_path = Path("data/validation/SPLIT_LOCK.json")
        self.split_metadata_path = Path("data/validation/split_metadata.json")
        
        # 确保验证目录存在
        self.validation_dir = Path("data/validation")
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果没有指定币种，使用标准9币种
        if self.config.symbols is None:
            self.config.symbols = [
                'BTCUSDT', 'ADAUSDT', 'ALGOUSDT', 'BNBUSDT', 'DOGEUSDT',
                'ICPUSDT', 'IOTAUSDT', 'SOLUSDT', 'SUIUSDT', 'XRPUSDT'
            ]
    
    def create_strict_split(self, symbol: str, data_path: str) -> DataSplit:
        """
        创建严格的时间顺序分割
        
        Args:
            symbol: 交易对符号
            data_path: 数据文件路径
            
        Returns:
            DataSplit: 分割结果
        """
        logger.info(f"为 {symbol} 创建严格数据分割...")
        
        # 检查是否已存在锁定的分割
        if self._is_split_locked():
            logger.warning("发现已锁定的数据分割，加载现有分割...")
            return self._load_locked_split(symbol)
        
        # 加载数据
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        total_samples = len(df)
        logger.info(f"{symbol} 总样本数: {total_samples}")
        
        # 验证最小样本要求
        if not self._validate_minimum_samples(total_samples):
            raise ValueError(f"数据量不足: {total_samples} < 最小要求")
        
        # 计算分割点
        train_end_idx = int(total_samples * self.config.train_ratio)
        val_end_idx = int(total_samples * (self.config.train_ratio + self.config.val_ratio))
        
        # 获取时间戳
        train_start = df.iloc[0]['timestamp']
        train_end = df.iloc[train_end_idx - 1]['timestamp']
        val_start = df.iloc[train_end_idx]['timestamp']
        val_end = df.iloc[val_end_idx - 1]['timestamp']
        test_start = df.iloc[val_end_idx]['timestamp']
        test_end = df.iloc[-1]['timestamp']
        
        # 创建分割对象
        split = DataSplit(
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            test_end=test_end,
            train_samples=train_end_idx,
            val_samples=val_end_idx - train_end_idx,
            test_samples=total_samples - val_end_idx,
            split_timestamp=datetime.now(),
            integrity_hash=self._calculate_integrity_hash(df)
        )
        
        logger.info(f"数据分割完成:")
        logger.info(f"  训练集: {train_start} -> {train_end} ({split.train_samples} 样本)")
        logger.info(f"  验证集: {val_start} -> {val_end} ({split.val_samples} 样本)")
        logger.info(f"  测试集: {test_start} -> {test_end} ({split.test_samples} 样本)")
        
        return split
    
    def split_all_symbols(self, data_dir: str = "data/market_data") -> Dict[str, DataSplit]:
        """
        为所有币种创建一致的数据分割
        
        Args:
            data_dir: 数据目录路径
            
        Returns:
            Dict[str, DataSplit]: 所有币种的分割结果
        """
        logger.info("开始为所有币种创建一致的数据分割...")
        
        data_path = Path(data_dir)
        splits = {}
        
        for symbol in self.config.symbols:
            # 查找5分钟数据文件
            symbol_file = f"{symbol}_5m_2years.csv"
            file_path = data_path / symbol_file
            
            if not file_path.exists():
                logger.warning(f"未找到 {symbol} 的数据文件: {file_path}")
                continue
            
            try:
                split = self.create_strict_split(symbol, str(file_path))
                splits[symbol] = split
                
                # 保存单个币种分割
                self._save_symbol_split(symbol, split)
                
            except Exception as e:
                logger.error(f"为 {symbol} 创建分割失败: {e}")
                continue
        
        # 验证所有分割的一致性
        self._validate_split_consistency(splits)
        
        # 锁定分割（防止后续修改）
        self._lock_splits(splits)
        
        logger.info(f"成功为 {len(splits)} 个币种创建数据分割")
        return splits
    
    def get_split_data(self, symbol: str, split_type: str, data_path: str) -> pd.DataFrame:
        """
        获取指定分割的数据
        
        Args:
            symbol: 交易对符号
            split_type: 分割类型 ('train', 'val', 'test')
            data_path: 数据文件路径
            
        Returns:
            pd.DataFrame: 分割后的数据
        """
        if symbol not in self.splits:
            raise ValueError(f"未找到 {symbol} 的分割信息")
        
        split = self.splits[symbol]
        
        # 加载完整数据
        df = pd.read_csv(data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # 根据分割类型返回数据
        if split_type == 'train':
            return df[(df['timestamp'] >= split.train_start) & 
                     (df['timestamp'] <= split.train_end)]
        elif split_type == 'val':
            return df[(df['timestamp'] >= split.val_start) & 
                     (df['timestamp'] <= split.val_end)]
        elif split_type == 'test':
            # 🚨 重要警告：测试集访问记录
            self._log_test_access(symbol)
            return df[(df['timestamp'] >= split.test_start) & 
                     (df['timestamp'] <= split.test_end)]
        else:
            raise ValueError(f"无效的分割类型: {split_type}")
    
    def _validate_minimum_samples(self, total_samples: int) -> bool:
        """验证最小样本要求"""
        required_samples = (self.config.min_train_samples + 
                          self.config.min_val_samples + 
                          self.config.min_test_samples)
        return total_samples >= required_samples
    
    def _calculate_integrity_hash(self, df: pd.DataFrame) -> str:
        """计算数据完整性哈希"""
        import hashlib
        data_str = f"{len(df)}_{df.iloc[0]['timestamp']}_{df.iloc[-1]['timestamp']}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _validate_split_consistency(self, splits: Dict[str, DataSplit]) -> None:
        """验证所有分割的时间一致性"""
        logger.info("验证分割一致性...")
        
        if not splits:
            raise ValueError("没有有效的分割数据")
        
        # 检查时间范围一致性
        first_split = list(splits.values())[0]
        base_train_period = (first_split.train_end - first_split.train_start).days
        base_val_period = (first_split.val_end - first_split.val_start).days
        base_test_period = (first_split.test_end - first_split.test_start).days
        
        for symbol, split in splits.items():
            train_period = (split.train_end - split.train_start).days
            val_period = (split.val_end - split.val_start).days
            test_period = (split.test_end - split.test_start).days
            
            if abs(train_period - base_train_period) > 7:  # 允许7天误差
                logger.warning(f"{symbol} 训练期不一致: {train_period} vs {base_train_period}")
            
            if abs(val_period - base_val_period) > 7:
                logger.warning(f"{symbol} 验证期不一致: {val_period} vs {base_val_period}")
            
            if abs(test_period - base_test_period) > 7:
                logger.warning(f"{symbol} 测试期不一致: {test_period} vs {base_test_period}")
        
        logger.info("分割一致性验证完成")
    
    def _is_split_locked(self) -> bool:
        """检查分割是否已锁定"""
        return self.lock_file_path.exists()
    
    def _lock_splits(self, splits: Dict[str, DataSplit]) -> None:
        """锁定分割，防止后续修改"""
        lock_data = {
            'locked_at': datetime.now().isoformat(),
            'symbols': list(splits.keys()),
            'warning': '🚨 测试集已锁定！禁止任何修改！',
            'test_access_log': []
        }
        
        with open(self.lock_file_path, 'w') as f:
            json.dump(lock_data, f, indent=2)
        
        logger.warning("🔒 数据分割已锁定！测试集不可再次访问用于优化！")
    
    def _load_locked_split(self, symbol: str) -> DataSplit:
        """加载已锁定的分割"""
        split_file = self.validation_dir / f"{symbol}_split.json"
        if not split_file.exists():
            raise ValueError(f"未找到 {symbol} 的锁定分割")
        
        with open(split_file) as f:
            data = json.load(f)
        
        return DataSplit(**data)
    
    def _save_symbol_split(self, symbol: str, split: DataSplit) -> None:
        """保存单个币种的分割信息"""
        split_file = self.validation_dir / f"{symbol}_split.json"
        
        split_data = {
            'train_start': split.train_start.isoformat(),
            'train_end': split.train_end.isoformat(),
            'val_start': split.val_start.isoformat(),
            'val_end': split.val_end.isoformat(),
            'test_start': split.test_start.isoformat(),
            'test_end': split.test_end.isoformat(),
            'train_samples': split.train_samples,
            'val_samples': split.val_samples,
            'test_samples': split.test_samples,
            'split_timestamp': split.split_timestamp.isoformat(),
            'integrity_hash': split.integrity_hash
        }
        
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
    
    def _log_test_access(self, symbol: str) -> None:
        """记录测试集访问（重要的审计功能）"""
        access_log = {
            'symbol': symbol,
            'accessed_at': datetime.now().isoformat(),
            'warning': '⚠️  测试集被访问！确保这是最终验证！'
        }
        
        log_file = self.validation_dir / "test_access_log.json"
        
        if log_file.exists():
            with open(log_file) as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(access_log)
        
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        logger.critical(f"🚨 测试集访问记录: {symbol} - 确保这是最终验证！")

    def get_split_summary(self) -> Dict:
        """获取分割摘要"""
        if not self.splits:
            return {"error": "没有可用的分割数据"}
        
        summary = {
            'total_symbols': len(self.splits),
            'split_ratios': {
                'train': self.config.train_ratio,
                'val': self.config.val_ratio,
                'test': self.config.test_ratio
            },
            'symbols_detail': {}
        }
        
        for symbol, split in self.splits.items():
            summary['symbols_detail'][symbol] = {
                'train_period': f"{split.train_start.date()} to {split.train_end.date()}",
                'val_period': f"{split.val_start.date()} to {split.val_end.date()}",
                'test_period': f"{split.test_start.date()} to {split.test_end.date()}",
                'sample_counts': {
                    'train': split.train_samples,
                    'val': split.val_samples,
                    'test': split.test_samples
                }
            }
        
        return summary