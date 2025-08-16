#!/usr/bin/env python3
"""
DipMaster V4 Feature Engineering Pipeline
特征工程管道 - 为DipMaster策略生成高质量ML特征

Author: DipMaster Development Team
Date: 2025-08-16
Version: 4.0.0
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging

# 抑制警告
warnings.filterwarnings('ignore')

class DipMasterFeatureEngineer:
    """DipMaster特征工程管道"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_names = []
        self.feature_config = {}
        
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标特征"""
        try:
            # RSI指标
            df['rsi'] = self._calculate_rsi(df['close'], period=14)
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_dip_zone'] = ((df['rsi'] >= 25) & (df['rsi'] <= 45)).astype(int)
            
            # 移动平均线
            df['ma_5'] = df['close'].rolling(window=5).mean()
            df['ma_10'] = df['close'].rolling(window=10).mean()
            df['ma_15'] = df['close'].rolling(window=15).mean()
            df['ma_20'] = df['close'].rolling(window=20).mean()
            
            # 价格相对位置
            df['price_ma_ratio'] = df['close'] / df['ma_20']
            df['below_ma20'] = (df['close'] < df['ma_20']).astype(int)
            
            # 布林带
            bb_data = self._calculate_bollinger_bands(df['close'], period=20, std_dev=2.0)
            df['bb_upper'] = bb_data['upper']
            df['bb_middle'] = bb_data['middle']
            df['bb_lower'] = bb_data['lower']
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            df['bb_squeeze'] = ((df['bb_upper'] - df['bb_lower']) / df['bb_middle']).rolling(5).mean()
            
            # 价格动量
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = abs(df['price_change'])
            df['momentum_5'] = df['close'].pct_change(periods=5)
            df['momentum_10'] = df['close'].pct_change(periods=10)
            df['momentum_20'] = df['close'].pct_change(periods=20)
            
            # 逢跌检测
            df['dip_signal'] = (df['close'] < df['open']).astype(int)
            df['red_candle'] = (df['close'] < df['open']).astype(int)
            df['consecutive_red'] = self._count_consecutive(df['red_candle'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"技术指标计算失败: {e}")
            return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加成交量特征"""
        try:
            # 成交量移动平均
            df['volume_ma_5'] = df['volume'].rolling(window=5).mean()
            df['volume_ma_10'] = df['volume'].rolling(window=10).mean()
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            
            # 成交量比率
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
            df['volume_surge'] = (df['volume_ratio'] > 1.5).astype(int)
            df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
            
            # 成交量确认
            df['volume_confirm_dip'] = (df['dip_signal'] & (df['volume_ratio'] > 1.2)).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"成交量特征计算失败: {e}")
            return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加波动率特征"""
        try:
            # ATR (Average True Range)
            df['tr'] = self._calculate_true_range(df)
            df['atr'] = df['tr'].rolling(window=14).mean()
            df['atr_pct'] = df['atr'] / df['close']
            
            # 滚动波动率
            df['volatility_5'] = df['price_change'].rolling(window=5).std()
            df['volatility_10'] = df['price_change'].rolling(window=10).std()
            df['volatility_20'] = df['price_change'].rolling(window=20).std()
            
            # 波动率状态
            df['high_volatility'] = (df['volatility_20'] > df['volatility_20'].rolling(60).quantile(0.8)).astype(int)
            df['low_volatility'] = (df['volatility_20'] < df['volatility_20'].rolling(60).quantile(0.2)).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"波动率特征计算失败: {e}")
            return df
    
    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        try:
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['month'] = df['timestamp'].dt.month
                
                # 交易时段
                df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
                df['european_session'] = ((df['hour'] >= 7) & (df['hour'] < 15)).astype(int)
                df['american_session'] = ((df['hour'] >= 14) & (df['hour'] < 22)).astype(int)
                
                # 周末效应
                df['weekend'] = (df['day_of_week'] >= 5).astype(int)
                df['monday'] = (df['day_of_week'] == 0).astype(int)
                df['friday'] = (df['day_of_week'] == 4).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"时间特征计算失败: {e}")
            return df
    
    def add_dipmaster_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加DipMaster专用信号特征"""
        try:
            # DipMaster核心信号
            dipmaster_conditions = [
                df['rsi_dip_zone'] == 1,  # RSI在逢跌区间
                df['below_ma20'] == 1,     # 价格低于MA20
                df['dip_signal'] == 1,     # 逢跌信号
                df['volume_surge'] == 1    # 成交量放大
            ]
            
            df['dipmaster_signal_count'] = sum(dipmaster_conditions)
            df['dipmaster_strong_signal'] = (df['dipmaster_signal_count'] >= 3).astype(int)
            df['dipmaster_perfect_signal'] = (df['dipmaster_signal_count'] == 4).astype(int)
            
            # 信号强度评分
            signal_weights = [0.3, 0.25, 0.25, 0.2]  # RSI, MA, Dip, Volume权重
            df['dipmaster_signal_strength'] = sum(
                cond.astype(float) * weight 
                for cond, weight in zip(dipmaster_conditions, signal_weights)
            )
            
            # 15分钟边界特征
            df['minutes_in_hour'] = df.index % 4  # 假设5分钟K线
            df['boundary_15min'] = (df['minutes_in_hour'] == 2).astype(int)  # 15分钟边界
            df['boundary_30min'] = (df['minutes_in_hour'] == 1).astype(int)  # 30分钟边界
            df['boundary_45min'] = (df['minutes_in_hour'] == 0).astype(int)  # 45分钟边界
            
            return df
            
        except Exception as e:
            self.logger.error(f"DipMaster信号计算失败: {e}")
            return df
    
    def generate_labels(self, df: pd.DataFrame, 
                       forward_periods: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
        """生成预测标签"""
        try:
            for period in forward_periods:
                # 未来收益率
                df[f'return_{period}p'] = df['close'].pct_change(periods=period).shift(-period)
                
                # 二分类标签（是否盈利）
                df[f'profitable_{period}p'] = (df[f'return_{period}p'] > 0).astype(int)
                
                # 达到目标收益标签
                df[f'target_return_{period}p'] = (df[f'return_{period}p'] > 0.008).astype(int)  # 0.8%目标
                
                # 未来最大收益和最大亏损
                future_highs = df['high'].rolling(window=period, min_periods=1).max().shift(-period)
                future_lows = df['low'].rolling(window=period, min_periods=1).min().shift(-period)
                
                df[f'max_return_{period}p'] = (future_highs - df['close']) / df['close']
                df[f'max_drawdown_{period}p'] = (future_lows - df['close']) / df['close']
            
            # 主要标签 - 15分钟未来收益（3个period）
            df['target'] = df['return_3p']
            df['target_binary'] = df['profitable_3p']
            
            return df
            
        except Exception as e:
            self.logger.error(f"标签生成失败: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, 
                                 std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """计算布林带"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """计算真实波动幅度"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range
    
    def _count_consecutive(self, series: pd.Series) -> pd.Series:
        """计算连续值的数量"""
        # 创建分组标识
        groups = (series != series.shift()).cumsum()
        
        # 只对True值计算连续数量
        consecutive = series.groupby(groups).cumsum()
        consecutive = consecutive.where(series == 1, 0)
        
        return consecutive
    
    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理特征数据"""
        try:
            # 处理无穷大值
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # 填充NaN值
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(0)
            
            # 移除极端异常值（超过5个标准差）
            for col in numeric_columns:
                if col not in ['timestamp', 'hour', 'day_of_week', 'month']:
                    mean = df[col].mean()
                    std = df[col].std()
                    df[col] = df[col].clip(lower=mean - 5*std, upper=mean + 5*std)
            
            return df
            
        except Exception as e:
            self.logger.error(f"特征清理失败: {e}")
            return df
    
    def process_symbol_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """处理单个交易对的数据"""
        try:
            self.logger.info(f"处理 {symbol} 的特征工程...")
            
            # 确保数据按时间排序
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            # 添加各类特征
            df = self.add_technical_indicators(df)
            df = self.add_volume_features(df)
            df = self.add_volatility_features(df)
            df = self.add_time_features(df)
            df = self.add_dipmaster_signals(df)
            
            # 生成标签
            df = self.generate_labels(df)
            
            # 清理数据
            df = self.clean_features(df)
            
            # 添加symbol标识
            df['symbol'] = symbol
            
            # 移除前面的NaN行（由于滚动计算产生）
            df = df.dropna(subset=['target']).reset_index(drop=True)
            
            self.logger.info(f"{symbol} 特征工程完成: {len(df)} 样本, {len([c for c in df.columns if c not in ['symbol', 'timestamp']])} 特征")
            
            return df
            
        except Exception as e:
            self.logger.error(f"{symbol} 特征工程失败: {e}")
            return pd.DataFrame()
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """获取特征名称列表"""
        # 排除非特征列
        exclude_columns = [
            'timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume',
            'target', 'target_binary'
        ]
        
        # 排除标签列
        label_columns = [col for col in df.columns if 'return_' in col or 'profitable_' in col 
                        or 'target_return_' in col or 'max_return_' in col or 'max_drawdown_' in col]
        
        exclude_columns.extend(label_columns)
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        return feature_columns
    
    def save_feature_config(self, features_df: pd.DataFrame, save_path: str) -> Dict:
        """保存特征配置信息"""
        try:
            feature_names = self.get_feature_names(features_df)
            
            config = {
                'feature_engineering_config': {
                    'pipeline_version': '4.0.0',
                    'generation_time': datetime.now().isoformat(),
                    'feature_count': len(feature_names),
                    'sample_count': len(features_df),
                    'symbols': features_df['symbol'].unique().tolist() if 'symbol' in features_df.columns else [],
                    'feature_names': feature_names,
                    'feature_categories': {
                        'technical_indicators': [f for f in feature_names if any(x in f for x in ['rsi', 'ma_', 'bb_', 'momentum_'])],
                        'volume_features': [f for f in feature_names if 'volume' in f],
                        'volatility_features': [f for f in feature_names if any(x in f for x in ['volatility', 'atr'])],
                        'time_features': [f for f in feature_names if any(x in f for x in ['hour', 'day_', 'session', 'weekend'])],
                        'dipmaster_signals': [f for f in feature_names if 'dipmaster' in f or 'signal' in f],
                        'price_features': [f for f in feature_names if any(x in f for x in ['price_', 'dip_', 'red_', 'consecutive'])]
                    },
                    'data_quality': {
                        'missing_values': features_df.isnull().sum().sum(),
                        'infinite_values': np.isinf(features_df.select_dtypes(include=[np.number])).sum().sum(),
                        'duplicate_rows': features_df.duplicated().sum(),
                        'memory_usage_mb': features_df.memory_usage(deep=True).sum() / 1024 / 1024
                    }
                },
                'dipmaster_strategy_features': {
                    'core_signals': ['rsi_dip_zone', 'below_ma20', 'dip_signal', 'volume_surge'],
                    'signal_strength': 'dipmaster_signal_strength',
                    'boundary_features': ['boundary_15min', 'boundary_30min', 'boundary_45min'],
                    'target_label': 'target',
                    'binary_label': 'target_binary',
                    'strategy_description': 'DipMaster AI逢跌买入策略特征工程，目标胜率85%+'
                }
            }
            
            # 保存配置
            config_path = save_path.replace('.parquet', '_config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"特征配置已保存: {config_path}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"保存特征配置失败: {e}")
            return {}

def main():
    """主函数 - 演示特征工程管道"""
    logging.basicConfig(level=logging.INFO)
    
    # 创建特征工程器
    engineer = DipMasterFeatureEngineer()
    
    # 模拟数据处理
    print("DipMaster V4 特征工程管道已就绪")
    print("主要功能:")
    print("1. 技术指标特征 (RSI, MA, 布林带, 动量)")
    print("2. 成交量特征 (量价确认)")
    print("3. 波动率特征 (ATR, 滚动波动率)")
    print("4. 时间特征 (交易时段, 周末效应)")
    print("5. DipMaster专用信号 (逢跌检测, 信号强度)")
    print("6. 预测标签生成 (多时间窗口收益)")
    print("7. 数据清理和质量控制")

if __name__ == "__main__":
    main()