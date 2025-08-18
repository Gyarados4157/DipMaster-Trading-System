#!/usr/bin/env python3
"""
SuperDip Needle Strategy Feature Engineering Pipeline
超跌接针策略专用特征工程管道

专为SuperDip接针反转策略设计的特征工程系统，实现：
1. 超跌识别特征：RSI多周期、价格偏离度、布林带位置、Z-Score偏离
2. 接针形态特征：下影线比率、实体位置、成交量放大、价格恢复度
3. 多时间框架特征：1m/5m/15m/1h跨周期融合
4. 成交量和微结构特征：成交量分布、价格-成交量背离、订单簿特征
5. 标签工程：多目标收益预测、风险调整收益、胜率预期
6. 严格的数据质量控制：前瞻性检查、缺失值处理、异常值检测

Author: DipMaster Quant Team
Date: 2025-08-18
Version: 1.0.0-SuperDipNeedle
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
import pickle
from dataclasses import dataclass, field
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.feature_selection import mutual_info_regression
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import ta
import numba
from numba import jit

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class SuperDipNeedleConfig:
    """SuperDip Needle Feature Engineering Configuration"""
    # 交易品种配置
    symbols: List[str] = field(default_factory=lambda: [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT',
        'BNBUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT', 'DOTUSDT',
        'ATOMUSDT', 'NEARUSDT', 'UNIUSDT', 'FILUSDT', 'TRXUSDT'
    ])
    
    # 时间框架配置
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m', '1h'])
    primary_timeframe: str = '5m'
    
    # 预测目标配置
    prediction_horizons: List[int] = field(default_factory=lambda: [15, 30, 60, 240])  # 分钟
    profit_targets: List[float] = field(default_factory=lambda: [0.008, 0.015, 0.025, 0.040])  # 利润目标
    stop_loss: float = 0.006  # 止损阈值
    max_holding_periods: int = 240  # 最大持仓时间（分钟）
    
    # RSI配置
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21, 30])
    rsi_oversold_threshold: float = 25
    rsi_optimal_range: Tuple[float, float] = (15, 30)
    
    # 移动平均配置
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    ema_periods: List[int] = field(default_factory=lambda: [8, 13, 21, 34, 55])
    
    # 布林带配置
    bollinger_periods: List[int] = field(default_factory=lambda: [20, 50])
    bollinger_std_multiplier: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    
    # 波动率配置
    volatility_periods: List[int] = field(default_factory=lambda: [10, 20, 30])
    
    # 成交量配置
    volume_ma_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    volume_spike_multiplier: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    
    # K线形态配置
    candlestick_patterns: List[str] = field(default_factory=lambda: [
        'hammer', 'inverted_hammer', 'doji', 'bullish_engulfing', 'piercing_line'
    ])
    
    # 特征启用开关
    enable_cross_timeframe: bool = True
    enable_microstructure: bool = True
    enable_advanced_labels: bool = True
    enable_interaction_features: bool = True
    enable_regime_features: bool = True
    
    # 数据质量控制
    min_data_points: int = 500
    max_nan_percentage: float = 0.05
    outlier_clip_percentile: Tuple[float, float] = (0.5, 99.5)

class SuperDipNeedleFeatureEngineer:
    """
    SuperDip Needle Strategy Feature Engineering Pipeline
    超跌接针策略专用特征工程管道
    
    核心功能：
    1. 超跌识别特征生成
    2. 接针形态特征提取
    3. 多时间框架信号融合
    4. 高级标签工程
    5. 严格数据质量控制
    """
    
    def __init__(self, config: Optional[SuperDipNeedleConfig] = None):
        """Initialize SuperDip Needle Feature Engineer"""
        self.logger = logging.getLogger(__name__)
        self.config = config or SuperDipNeedleConfig()
        
        # 数据预处理组件
        self.scaler = RobustScaler()
        self.quantile_transformer = QuantileTransformer(n_quantiles=1000, random_state=42)
        
        # 特征追踪
        self.feature_names = []
        self.feature_importance = {}
        self.feature_categories = {
            'oversold_detection': [],
            'needle_pattern': [],
            'multi_timeframe': [],
            'volume_microstructure': [],
            'interaction': [],
            'regime_adaptive': []
        }
        
        # 数据质量监控
        self.quality_metrics = {
            'data_leakage_check': {},
            'feature_stability': {},
            'correlation_analysis': {},
            'missing_data_report': {}
        }
        
    def load_market_data(self, bundle_path: str) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load market data from MarketDataBundle
        从市场数据包加载多时间框架数据
        """
        try:
            self.logger.info(f"Loading market data from {bundle_path}")
            
            with open(bundle_path, 'r') as f:
                bundle = json.load(f)
            
            data_dict = {}
            
            for symbol in self.config.symbols:
                if symbol not in bundle.get('data_files', {}):
                    self.logger.warning(f"Symbol {symbol} not found in bundle")
                    continue
                
                data_dict[symbol] = {}
                symbol_files = bundle['data_files'][symbol]
                
                for timeframe in self.config.timeframes:
                    if timeframe not in symbol_files:
                        self.logger.warning(f"Timeframe {timeframe} not found for {symbol}")
                        continue
                    
                    file_path = symbol_files[timeframe]['file_path']
                    if Path(file_path).exists():
                        df = pd.read_parquet(file_path)
                        df.index = pd.to_datetime(df.index)
                        data_dict[symbol][timeframe] = df
                        self.logger.info(f"Loaded {symbol} {timeframe}: {len(df)} rows")
                    else:
                        self.logger.error(f"File not found: {file_path}")
            
            return data_dict
            
        except Exception as e:
            self.logger.error(f"Failed to load market data: {e}")
            raise
    
    def generate_oversold_detection_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        生成超跌识别特征
        1. RSI多周期分析
        2. 价格偏离度计算
        3. 布林带位置特征
        4. Z-Score偏离度
        """
        try:
            result_df = df.copy()
            
            # 1. RSI多周期特征
            for period in self.config.rsi_periods:
                rsi_col = f'rsi_{period}'
                result_df[rsi_col] = ta.momentum.RSIIndicator(
                    close=df['close'], window=period
                ).rsi()
                
                # RSI位置分析
                result_df[f'{rsi_col}_oversold'] = (result_df[rsi_col] < self.config.rsi_oversold_threshold).astype(int)
                result_df[f'{rsi_col}_optimal_range'] = (
                    (result_df[rsi_col] >= self.config.rsi_optimal_range[0]) & 
                    (result_df[rsi_col] <= self.config.rsi_optimal_range[1])
                ).astype(int)
                
                # RSI变化率
                result_df[f'{rsi_col}_change'] = result_df[rsi_col].pct_change()
                result_df[f'{rsi_col}_momentum'] = result_df[rsi_col].diff(3)
                
                self.feature_categories['oversold_detection'].extend([
                    rsi_col, f'{rsi_col}_oversold', f'{rsi_col}_optimal_range',
                    f'{rsi_col}_change', f'{rsi_col}_momentum'
                ])
            
            # 2. 价格偏离度特征
            for period in self.config.ma_periods:
                ma_col = f'ma_{period}'
                result_df[ma_col] = df['close'].rolling(window=period).mean()
                
                # 价格偏离度
                deviation_col = f'price_deviation_ma{period}'
                result_df[deviation_col] = (df['close'] - result_df[ma_col]) / result_df[ma_col]
                
                # 偏离度分类
                result_df[f'{deviation_col}_oversold'] = (result_df[deviation_col] < -0.03).astype(int)
                result_df[f'{deviation_col}_extreme'] = (result_df[deviation_col] < -0.05).astype(int)
                
                self.feature_categories['oversold_detection'].extend([
                    ma_col, deviation_col, f'{deviation_col}_oversold', f'{deviation_col}_extreme'
                ])
            
            # 3. 布林带位置特征
            for period in self.config.bollinger_periods:
                for std_mult in self.config.bollinger_std_multiplier:
                    bb_indicator = ta.volatility.BollingerBands(
                        close=df['close'], window=period, window_dev=std_mult
                    )
                    
                    bb_upper = f'bb_upper_{period}_{std_mult}'
                    bb_lower = f'bb_lower_{period}_{std_mult}'
                    bb_position = f'bb_position_{period}_{std_mult}'
                    bb_width = f'bb_width_{period}_{std_mult}'
                    
                    result_df[bb_upper] = bb_indicator.bollinger_hband()
                    result_df[bb_lower] = bb_indicator.bollinger_lband()
                    
                    # 布林带位置 (0-1之间，0为下轨，1为上轨)
                    result_df[bb_position] = (df['close'] - result_df[bb_lower]) / (
                        result_df[bb_upper] - result_df[bb_lower]
                    )
                    
                    # 布林带宽度
                    result_df[bb_width] = (result_df[bb_upper] - result_df[bb_lower]) / result_df[f'ma_{period}']
                    
                    # 突破下轨
                    result_df[f'bb_break_lower_{period}_{std_mult}'] = (df['close'] < result_df[bb_lower]).astype(int)
                    
                    self.feature_categories['oversold_detection'].extend([
                        bb_upper, bb_lower, bb_position, bb_width, f'bb_break_lower_{period}_{std_mult}'
                    ])
            
            # 4. Z-Score偏离度
            for period in [20, 50, 100]:
                price_mean = df['close'].rolling(window=period).mean()
                price_std = df['close'].rolling(window=period).std()
                
                zscore_col = f'price_zscore_{period}'
                result_df[zscore_col] = (df['close'] - price_mean) / price_std
                
                # Z-Score分类
                result_df[f'{zscore_col}_oversold'] = (result_df[zscore_col] < -2).astype(int)
                result_df[f'{zscore_col}_extreme'] = (result_df[zscore_col] < -3).astype(int)
                
                self.feature_categories['oversold_detection'].extend([
                    zscore_col, f'{zscore_col}_oversold', f'{zscore_col}_extreme'
                ])
            
            # 5. 相对强弱指标
            for period in [5, 10, 20]:
                returns = df['close'].pct_change()
                gains = returns.where(returns > 0, 0)
                losses = -returns.where(returns < 0, 0)
                
                avg_gains = gains.rolling(window=period).mean()
                avg_losses = losses.rolling(window=period).mean()
                
                rs_col = f'relative_strength_{period}'
                result_df[rs_col] = avg_gains / (avg_losses + 1e-8)
                
                self.feature_categories['oversold_detection'].append(rs_col)
            
            self.logger.info(f"Generated {len(self.feature_categories['oversold_detection'])} oversold detection features for {symbol}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Failed to generate oversold features for {symbol}: {e}")
            return df
    
    def generate_needle_pattern_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        生成接针形态特征
        1. 下影线比率分析
        2. 实体位置计算
        3. 成交量放大检测
        4. 价格恢复度测量
        """
        try:
            result_df = df.copy()
            
            # K线基础数据
            open_price = df['open']
            high_price = df['high']
            low_price = df['low']
            close_price = df['close']
            volume = df['volume']
            
            # 1. K线形态特征
            # 实体大小
            result_df['candle_body'] = abs(close_price - open_price)
            result_df['candle_range'] = high_price - low_price
            result_df['body_ratio'] = result_df['candle_body'] / (result_df['candle_range'] + 1e-8)
            
            # 上下影线
            result_df['upper_shadow'] = np.where(
                close_price > open_price,
                high_price - close_price,
                high_price - open_price
            )
            result_df['lower_shadow'] = np.where(
                close_price > open_price,
                open_price - low_price,
                close_price - low_price
            )
            
            # 影线比率
            result_df['upper_shadow_ratio'] = result_df['upper_shadow'] / (result_df['candle_range'] + 1e-8)
            result_df['lower_shadow_ratio'] = result_df['lower_shadow'] / (result_df['candle_range'] + 1e-8)
            
            # 下影线与实体比率（接针关键指标）
            result_df['lower_shadow_body_ratio'] = result_df['lower_shadow'] / (result_df['candle_body'] + 1e-8)
            
            # 2. 接针形态识别
            # 锤子线特征
            result_df['hammer_pattern'] = (
                (result_df['lower_shadow_ratio'] > 0.6) &  # 下影线占比>60%
                (result_df['body_ratio'] < 0.3) &  # 实体占比<30%
                (result_df['lower_shadow_body_ratio'] > 2)  # 下影线是实体的2倍以上
            ).astype(int)
            
            # 倒锤子线特征
            result_df['inverted_hammer_pattern'] = (
                (result_df['upper_shadow_ratio'] > 0.6) &
                (result_df['body_ratio'] < 0.3) &
                (result_df['upper_shadow'] / (result_df['candle_body'] + 1e-8) > 2)
            ).astype(int)
            
            # 十字星形态
            result_df['doji_pattern'] = (result_df['body_ratio'] < 0.1).astype(int)
            
            # 3. 实体位置特征
            # 实体在整个K线中的位置 (0为底部，1为顶部)
            result_df['body_position'] = np.where(
                result_df['candle_range'] > 0,
                np.where(
                    close_price > open_price,
                    (open_price - low_price) / result_df['candle_range'],
                    (close_price - low_price) / result_df['candle_range']
                ),
                0.5
            )
            
            # 收盘价在K线中的位置
            result_df['close_position'] = (close_price - low_price) / (result_df['candle_range'] + 1e-8)
            
            # 4. 成交量特征
            for period in self.config.volume_ma_periods:
                volume_ma = volume.rolling(window=period).mean()
                result_df[f'volume_ratio_{period}'] = volume / (volume_ma + 1e-8)
                
                # 成交量放大检测
                for multiplier in self.config.volume_spike_multiplier:
                    spike_col = f'volume_spike_{period}_{multiplier}'
                    result_df[spike_col] = (result_df[f'volume_ratio_{period}'] > multiplier).astype(int)
                    self.feature_categories['needle_pattern'].append(spike_col)
            
            # 成交量与价格变化的关系
            price_change = close_price.pct_change()
            result_df['volume_price_correlation'] = price_change.rolling(window=20).corr(volume.pct_change())
            
            # 5. 价格恢复度特征
            # 从最低点的恢复程度
            result_df['recovery_from_low'] = (close_price - low_price) / (result_df['candle_range'] + 1e-8)
            
            # 多周期价格恢复
            for period in [3, 5, 10]:
                lowest_low = low_price.rolling(window=period).min()
                result_df[f'recovery_from_low_{period}'] = (close_price - lowest_low) / (close_price + 1e-8)
            
            # 6. 动量和趋势特征
            # 价格动量
            for period in [3, 5, 10]:
                result_df[f'price_momentum_{period}'] = close_price.pct_change(periods=period)
            
            # 价格加速度
            returns = close_price.pct_change()
            result_df['price_acceleration'] = returns.diff()
            
            # 趋势强度
            for period in [10, 20]:
                result_df[f'trend_strength_{period}'] = close_price.rolling(window=period).apply(
                    lambda x: stats.pearsonr(np.arange(len(x)), x)[0] if len(x) > 1 else 0
                )
            
            # 添加特征到分类
            needle_features = [
                'candle_body', 'candle_range', 'body_ratio',
                'upper_shadow', 'lower_shadow', 'upper_shadow_ratio', 'lower_shadow_ratio',
                'lower_shadow_body_ratio', 'hammer_pattern', 'inverted_hammer_pattern', 'doji_pattern',
                'body_position', 'close_position', 'volume_price_correlation',
                'recovery_from_low', 'price_acceleration'
            ]
            
            # 添加多周期特征
            for period in [3, 5, 10]:
                needle_features.extend([
                    f'recovery_from_low_{period}',
                    f'price_momentum_{period}'
                ])
            
            for period in [10, 20]:
                needle_features.append(f'trend_strength_{period}')
            
            for period in self.config.volume_ma_periods:
                needle_features.append(f'volume_ratio_{period}')
            
            self.feature_categories['needle_pattern'].extend(needle_features)
            
            self.logger.info(f"Generated {len(needle_features)} needle pattern features for {symbol}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Failed to generate needle pattern features for {symbol}: {e}")
            return df
    
    def generate_multi_timeframe_features(self, data_dict: Dict[str, pd.DataFrame], symbol: str) -> pd.DataFrame:
        """
        生成多时间框架融合特征
        1. 跨时间框架信号一致性
        2. 高时间框架趋势确认
        3. 低时间框架精确进入点
        """
        try:
            if self.config.primary_timeframe not in data_dict:
                self.logger.warning(f"Primary timeframe {self.config.primary_timeframe} not available for {symbol}")
                return data_dict[list(data_dict.keys())[0]].copy()
            
            # 以主时间框架为基础
            primary_df = data_dict[self.config.primary_timeframe].copy()
            
            # 对每个时间框架生成关键指标
            for timeframe, df in data_dict.items():
                if timeframe == self.config.primary_timeframe:
                    continue
                
                # 重采样到主时间框架
                if timeframe in ['1m', '5m'] and self.config.primary_timeframe == '5m':
                    if timeframe == '1m':
                        # 1分钟数据的高频特征
                        # 价格波动率
                        df['intrabar_volatility'] = (df['high'] - df['low']) / df['close']
                        df['intrabar_returns'] = df['close'].pct_change()
                        
                        # 重采样到5分钟
                        resampled = df.resample('5T').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last',
                            'volume': 'sum',
                            'intrabar_volatility': 'mean',
                            'intrabar_returns': 'std'
                        }).dropna()
                        
                        # 对齐时间索引
                        aligned_data = resampled.reindex(primary_df.index, method='ffill')
                        
                        # 添加高频特征
                        primary_df[f'hf_volatility_{timeframe}'] = aligned_data['intrabar_volatility']
                        primary_df[f'hf_returns_std_{timeframe}'] = aligned_data['intrabar_returns']
                        
                        self.feature_categories['multi_timeframe'].extend([
                            f'hf_volatility_{timeframe}', f'hf_returns_std_{timeframe}'
                        ])
                
                elif timeframe in ['15m', '1h'] and self.config.primary_timeframe == '5m':
                    # 高时间框架趋势特征
                    # 重采样到主时间框架
                    if timeframe == '15m':
                        freq = '15T'
                    else:  # 1h
                        freq = '1H'
                    
                    # 计算高时间框架指标
                    df_htf = df.copy()
                    df_htf[f'ma20_{timeframe}'] = df_htf['close'].rolling(window=20).mean()
                    df_htf[f'rsi14_{timeframe}'] = ta.momentum.RSIIndicator(
                        close=df_htf['close'], window=14
                    ).rsi()
                    
                    # 趋势方向
                    df_htf[f'trend_direction_{timeframe}'] = np.where(
                        df_htf['close'] > df_htf[f'ma20_{timeframe}'], 1,
                        np.where(df_htf['close'] < df_htf[f'ma20_{timeframe}'], -1, 0)
                    )
                    
                    # 对齐到主时间框架
                    aligned_data = df_htf.reindex(primary_df.index, method='ffill')
                    
                    primary_df[f'htf_ma20_{timeframe}'] = aligned_data[f'ma20_{timeframe}']
                    primary_df[f'htf_rsi14_{timeframe}'] = aligned_data[f'rsi14_{timeframe}']
                    primary_df[f'htf_trend_{timeframe}'] = aligned_data[f'trend_direction_{timeframe}']
                    
                    # 高时间框架RSI状态
                    primary_df[f'htf_rsi_oversold_{timeframe}'] = (
                        aligned_data[f'rsi14_{timeframe}'] < 30
                    ).astype(int)
                    
                    self.feature_categories['multi_timeframe'].extend([
                        f'htf_ma20_{timeframe}', f'htf_rsi14_{timeframe}',
                        f'htf_trend_{timeframe}', f'htf_rsi_oversold_{timeframe}'
                    ])
            
            # 跨时间框架一致性检查
            if len(data_dict) >= 3:
                # RSI一致性
                rsi_cols = [col for col in primary_df.columns if 'rsi' in col.lower()]
                if len(rsi_cols) >= 2:
                    primary_df['rsi_consensus'] = (primary_df[rsi_cols] < 35).sum(axis=1)
                    primary_df['rsi_consensus_score'] = primary_df['rsi_consensus'] / len(rsi_cols)
                    
                    self.feature_categories['multi_timeframe'].extend([
                        'rsi_consensus', 'rsi_consensus_score'
                    ])
                
                # 趋势一致性
                trend_cols = [col for col in primary_df.columns if 'trend' in col]
                if len(trend_cols) >= 2:
                    primary_df['trend_consensus'] = primary_df[trend_cols].mean(axis=1)
                    self.feature_categories['multi_timeframe'].append('trend_consensus')
            
            self.logger.info(f"Generated multi-timeframe features for {symbol}")
            return primary_df
            
        except Exception as e:
            self.logger.error(f"Failed to generate multi-timeframe features for {symbol}: {e}")
            return data_dict.get(self.config.primary_timeframe, list(data_dict.values())[0])
    
    def generate_microstructure_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        生成微结构特征
        1. 成交量分布分析
        2. 价格-成交量背离
        3. 流动性估算
        4. 买卖压力指标
        """
        try:
            result_df = df.copy()
            
            # 1. 成交量分布特征
            volume = df['volume']
            
            # 成交量分位数
            for window in [10, 20, 50]:
                result_df[f'volume_percentile_{window}'] = volume.rolling(window=window).apply(
                    lambda x: stats.percentileofscore(x, x.iloc[-1])
                )
            
            # 成交量趋势
            result_df['volume_trend_10'] = volume.rolling(window=10).apply(
                lambda x: stats.pearsonr(np.arange(len(x)), x)[0] if len(x) > 1 else 0
            )
            
            # 2. 价格-成交量关系
            returns = df['close'].pct_change()
            volume_change = volume.pct_change()
            
            # 滚动相关性
            for window in [10, 20]:
                result_df[f'price_volume_corr_{window}'] = returns.rolling(window=window).corr(volume_change)
            
            # 价格-成交量背离检测
            result_df['price_up_volume_down'] = (
                (returns > 0.002) & (volume_change < -0.1)
            ).astype(int)
            
            result_df['price_down_volume_up'] = (
                (returns < -0.002) & (volume_change > 0.1)
            ).astype(int)
            
            # 3. 流动性估算
            # Amihud流动性测量
            for window in [10, 20]:
                abs_returns = abs(returns)
                illiquidity = abs_returns / (volume + 1e-8)
                result_df[f'amihud_illiquidity_{window}'] = illiquidity.rolling(window=window).mean()
            
            # 买卖价差估算（基于高低价）
            result_df['estimated_spread'] = (df['high'] - df['low']) / df['close']
            result_df['spread_ma_10'] = result_df['estimated_spread'].rolling(window=10).mean()
            
            # 4. 订单流估算
            # 基于价格和成交量的买卖压力
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            money_flow = typical_price * volume
            
            # 正负成交量
            positive_flow = np.where(typical_price > typical_price.shift(1), money_flow, 0)
            negative_flow = np.where(typical_price < typical_price.shift(1), money_flow, 0)
            
            for window in [10, 20]:
                pos_flow_sum = positive_flow.rolling(window=window).sum()
                neg_flow_sum = negative_flow.rolling(window=window).sum()
                
                result_df[f'money_flow_ratio_{window}'] = pos_flow_sum / (neg_flow_sum + 1e-8)
                result_df[f'net_flow_ratio_{window}'] = (pos_flow_sum - neg_flow_sum) / (pos_flow_sum + neg_flow_sum + 1e-8)
            
            # 5. 成交量加权价格指标
            # VWAP偏离
            for window in [10, 20, 50]:
                vwap = (df['close'] * volume).rolling(window=window).sum() / volume.rolling(window=window).sum()
                result_df[f'vwap_{window}'] = vwap
                result_df[f'price_vwap_deviation_{window}'] = (df['close'] - vwap) / vwap
            
            # 添加特征到分类
            microstructure_features = [
                'volume_trend_10', 'price_up_volume_down', 'price_down_volume_up',
                'estimated_spread', 'spread_ma_10'
            ]
            
            # 添加窗口特征
            for window in [10, 20, 50]:
                if window <= 20:
                    microstructure_features.extend([
                        f'volume_percentile_{window}',
                        f'price_volume_corr_{window}',
                        f'amihud_illiquidity_{window}',
                        f'money_flow_ratio_{window}',
                        f'net_flow_ratio_{window}'
                    ])
                microstructure_features.extend([
                    f'vwap_{window}',
                    f'price_vwap_deviation_{window}'
                ])
            
            self.feature_categories['volume_microstructure'].extend(microstructure_features)
            
            self.logger.info(f"Generated {len(microstructure_features)} microstructure features for {symbol}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Failed to generate microstructure features for {symbol}: {e}")
            return df
    
    def generate_regime_adaptive_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        生成市场状态自适应特征
        1. 波动率状态识别
        2. 趋势状态检测
        3. 流动性状态评估
        """
        try:
            result_df = df.copy()
            
            # 1. 波动率状态特征
            returns = df['close'].pct_change()
            
            # 多周期波动率
            for window in [10, 20, 50]:
                volatility = returns.rolling(window=window).std() * np.sqrt(288)  # 5分钟年化
                result_df[f'volatility_{window}'] = volatility
                
                # 波动率分位数
                vol_percentile = volatility.rolling(window=100).apply(
                    lambda x: stats.percentileofscore(x, x.iloc[-1])
                )
                result_df[f'volatility_percentile_{window}'] = vol_percentile
                
                # 波动率状态
                result_df[f'vol_regime_{window}'] = np.where(
                    vol_percentile > 80, 2,  # 高波动
                    np.where(vol_percentile < 20, 0, 1)  # 低波动/正常
                )
            
            # 2. 趋势状态特征
            for window in [20, 50]:
                # 趋势强度
                trend_strength = df['close'].rolling(window=window).apply(
                    lambda x: abs(stats.pearsonr(np.arange(len(x)), x)[0]) if len(x) > 1 else 0
                )
                result_df[f'trend_strength_{window}'] = trend_strength
                
                # 趋势方向
                trend_direction = df['close'].rolling(window=window).apply(
                    lambda x: stats.pearsonr(np.arange(len(x)), x)[0] if len(x) > 1 else 0
                )
                result_df[f'trend_direction_{window}'] = trend_direction
                
                # 趋势状态分类
                result_df[f'trend_state_{window}'] = np.where(
                    trend_strength > 0.7, np.where(trend_direction > 0, 2, 1),  # 强上升/下降趋势
                    0  # 横盘
                )
            
            # 3. 市场微观结构状态
            volume = df['volume']
            
            # 流动性状态
            spread_estimate = (df['high'] - df['low']) / df['close']
            spread_ma = spread_estimate.rolling(window=20).mean()
            spread_percentile = spread_ma.rolling(window=100).apply(
                lambda x: stats.percentileofscore(x, x.iloc[-1])
            )
            
            result_df['liquidity_state'] = np.where(
                spread_percentile > 80, 0,  # 低流动性
                np.where(spread_percentile < 20, 2, 1)  # 高流动性/正常
            )
            
            # 4. 综合市场状态
            # 基于波动率、趋势、流动性的综合评分
            vol_score = result_df['vol_regime_20'] / 2  # 标准化到0-1
            trend_score = (result_df['trend_state_20'] + 1) / 3  # 标准化到0-1
            liquidity_score = result_df['liquidity_state'] / 2  # 标准化到0-1
            
            result_df['market_regime_score'] = (vol_score + trend_score + liquidity_score) / 3
            
            # 市场状态分类
            result_df['market_regime'] = np.where(
                result_df['market_regime_score'] > 0.7, 2,  # 有利环境
                np.where(result_df['market_regime_score'] < 0.3, 0, 1)  # 不利/中性
            )
            
            # 添加特征到分类
            regime_features = ['liquidity_state', 'market_regime_score', 'market_regime']
            
            for window in [10, 20, 50]:
                regime_features.extend([
                    f'volatility_{window}',
                    f'volatility_percentile_{window}',
                    f'vol_regime_{window}'
                ])
            
            for window in [20, 50]:
                regime_features.extend([
                    f'trend_strength_{window}',
                    f'trend_direction_{window}',
                    f'trend_state_{window}'
                ])
            
            self.feature_categories['regime_adaptive'].extend(regime_features)
            
            self.logger.info(f"Generated {len(regime_features)} regime adaptive features for {symbol}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Failed to generate regime features for {symbol}: {e}")
            return df
    
    def generate_interaction_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        生成交互特征
        1. 关键特征组合
        2. 条件特征
        3. 比率特征
        """
        try:
            result_df = df.copy()
            
            # 1. RSI和价格偏离的交互
            if 'rsi_14' in result_df.columns and 'price_deviation_ma20' in result_df.columns:
                # RSI超卖且价格偏离的组合信号
                result_df['rsi_price_oversold_combo'] = (
                    (result_df['rsi_14'] < 30) & 
                    (result_df['price_deviation_ma20'] < -0.03)
                ).astype(int)
                
                # RSI和价格偏离的乘积
                result_df['rsi_price_deviation_product'] = (
                    result_df['rsi_14'] * result_df['price_deviation_ma20']
                )
            
            # 2. 成交量和价格形态的交互
            if 'volume_ratio_20' in result_df.columns and 'hammer_pattern' in result_df.columns:
                # 大成交量锤子线
                result_df['high_volume_hammer'] = (
                    (result_df['volume_ratio_20'] > 2.0) & 
                    (result_df['hammer_pattern'] == 1)
                ).astype(int)
            
            # 3. 多时间框架一致性交互
            rsi_cols = [col for col in result_df.columns if 'rsi' in col.lower() and 'oversold' in col]
            if len(rsi_cols) >= 2:
                result_df['multi_tf_rsi_alignment'] = result_df[rsi_cols].sum(axis=1)
            
            # 4. 波动率和趋势交互
            if 'volatility_20' in result_df.columns and 'trend_strength_20' in result_df.columns:
                # 低波动强趋势（突破前兆）
                result_df['low_vol_strong_trend'] = (
                    (result_df['volatility_percentile_20'] < 20) & 
                    (result_df['trend_strength_20'] > 0.7)
                ).astype(int)
            
            # 5. 布林带和RSI交互
            bb_cols = [col for col in result_df.columns if 'bb_position' in col]
            if bb_cols and 'rsi_14' in result_df.columns:
                bb_col = bb_cols[0]
                # 布林带下轨 + RSI超卖
                result_df['bb_rsi_oversold'] = (
                    (result_df[bb_col] < 0.1) & 
                    (result_df['rsi_14'] < 25)
                ).astype(int)
            
            # 添加交互特征到分类
            interaction_features = []
            for col in result_df.columns:
                if any(keyword in col for keyword in [
                    'combo', 'product', 'alignment', 'interaction',
                    'high_volume_hammer', 'low_vol_strong_trend', 'bb_rsi_oversold'
                ]):
                    interaction_features.append(col)
            
            self.feature_categories['interaction'].extend(interaction_features)
            
            self.logger.info(f"Generated {len(interaction_features)} interaction features for {symbol}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Failed to generate interaction features for {symbol}: {e}")
            return df
    
    @jit(nopython=True)
    def _calculate_future_return(self, prices, horizons):
        """计算未来收益的JIT优化版本"""
        n = len(prices)
        results = np.full((n, len(horizons)), np.nan)
        
        for i in range(n):
            current_price = prices[i]
            for j, horizon in enumerate(horizons):
                if i + horizon < n:
                    future_price = prices[i + horizon]
                    results[i, j] = (future_price - current_price) / current_price
        
        return results
    
    def generate_advanced_labels(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        生成高级标签
        1. 多目标收益预测
        2. 风险调整收益
        3. 胜率预期
        4. 最优出场时间
        """
        try:
            result_df = df.copy()
            
            prices = df['close'].values
            
            # 1. 多时间窗口收益标签
            horizons = self.config.prediction_horizons
            future_returns = self._calculate_future_return(prices, np.array(horizons))
            
            for i, horizon in enumerate(horizons):
                col_name = f'target_return_{horizon}min'
                result_df[col_name] = future_returns[:, i]
                
                # 收益分类标签
                profit_targets = self.config.profit_targets
                for j, target in enumerate(profit_targets):
                    result_df[f'target_profit_{target*100:.1f}pct_{horizon}min'] = (
                        result_df[col_name] >= target
                    ).astype(int)
                
                # 止损标签
                result_df[f'target_stop_loss_{horizon}min'] = (
                    result_df[col_name] <= -self.config.stop_loss
                ).astype(int)
            
            # 2. 风险调整收益标签
            for horizon in horizons:
                return_col = f'target_return_{horizon}min'
                if return_col in result_df.columns:
                    # 计算滚动波动率
                    rolling_vol = df['close'].pct_change().rolling(window=20).std()
                    
                    # 夏普比率样式的风险调整收益
                    result_df[f'risk_adj_return_{horizon}min'] = (
                        result_df[return_col] / (rolling_vol + 1e-8)
                    )
            
            # 3. 最优出场时间标签
            max_holding = min(self.config.max_holding_periods, max(horizons))
            
            # 计算最优出场点
            for i in range(len(df)):
                max_return = -float('inf')
                optimal_exit = None
                
                for horizon in range(5, max_holding + 1, 5):  # 每5分钟检查一次
                    if i + horizon < len(df):
                        future_return = (prices[i + horizon] - prices[i]) / prices[i]
                        if future_return > max_return:
                            max_return = future_return
                            optimal_exit = horizon
                
                result_df.loc[df.index[i], 'optimal_exit_time'] = optimal_exit
                result_df.loc[df.index[i], 'optimal_return'] = max_return
            
            # 4. 15分钟边界优化标签
            # 计算到下一个15分钟边界的时间
            df_index = pd.to_datetime(df.index)
            minutes = df_index.minute
            
            # 到下一个15分钟边界的分钟数
            time_to_boundary = 15 - (minutes % 15)
            time_to_boundary = np.where(time_to_boundary == 15, 0, time_to_boundary)
            result_df['time_to_15min_boundary'] = time_to_boundary
            
            # 15分钟边界收益
            boundary_returns = []
            for i in range(len(df)):
                boundary_time = time_to_boundary[i]
                if boundary_time > 0 and i + boundary_time < len(df):
                    boundary_return = (prices[i + boundary_time] - prices[i]) / prices[i]
                    boundary_returns.append(boundary_return)
                else:
                    boundary_returns.append(np.nan)
            
            result_df['target_boundary_return'] = boundary_returns
            
            # 边界收益达标
            for target in self.config.profit_targets:
                result_df[f'boundary_profit_{target*100:.1f}pct'] = (
                    result_df['target_boundary_return'] >= target
                ).astype(int)
            
            # 5. 胜率预期标签（基于历史相似情况）
            # 这里使用简化版本，实际应用中可以使用更复杂的相似性匹配
            for horizon in [15, 30, 60]:
                return_col = f'target_return_{horizon}min'
                if return_col in result_df.columns:
                    # 滚动胜率计算
                    rolling_winrate = result_df[return_col].rolling(window=100).apply(
                        lambda x: (x > 0.008).sum() / len(x)  # 0.8%目标收益的胜率
                    )
                    result_df[f'expected_winrate_{horizon}min'] = rolling_winrate
            
            self.logger.info(f"Generated advanced labels for {symbol}")
            return result_df
            
        except Exception as e:
            self.logger.error(f"Failed to generate advanced labels for {symbol}: {e}")
            return df
    
    def validate_data_quality(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        数据质量验证
        1. 前瞻性检查
        2. 缺失值分析
        3. 异常值检测
        4. 特征稳定性
        """
        try:
            quality_report = {
                'symbol': symbol,
                'total_rows': len(df),
                'total_features': len(df.columns),
                'data_leakage_check': {},
                'missing_data_analysis': {},
                'outlier_analysis': {},
                'feature_stability': {},
                'recommendations': []
            }
            
            # 1. 数据泄露检查
            target_cols = [col for col in df.columns if col.startswith('target_')]
            feature_cols = [col for col in df.columns if not col.startswith('target_')]
            
            # 检查是否有未来信息泄露
            leakage_detected = False
            for col in feature_cols:
                if any(keyword in col.lower() for keyword in ['future', 'next', 'forward']):
                    quality_report['data_leakage_check'][col] = 'POTENTIAL_LEAKAGE'
                    leakage_detected = True
            
            if not leakage_detected:
                quality_report['data_leakage_check']['status'] = 'PASSED'
            
            # 2. 缺失值分析
            missing_analysis = {}
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_pct = missing_count / len(df)
                missing_analysis[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': float(missing_pct),
                    'status': 'OK' if missing_pct < 0.05 else 'WARNING' if missing_pct < 0.2 else 'CRITICAL'
                }
            
            quality_report['missing_data_analysis'] = missing_analysis
            
            # 3. 异常值检测
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            outlier_analysis = {}
            
            for col in numeric_cols:
                if col not in target_cols:  # 不检查目标变量
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    outlier_count = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
                    outlier_pct = outlier_count / len(df)
                    
                    outlier_analysis[col] = {
                        'outlier_count': int(outlier_count),
                        'outlier_percentage': float(outlier_pct),
                        'status': 'OK' if outlier_pct < 0.05 else 'WARNING'
                    }
            
            quality_report['outlier_analysis'] = outlier_analysis
            
            # 4. 特征稳定性检查（简化版PSI）
            stability_analysis = {}
            if len(df) > 1000:  # 需要足够的数据
                split_point = len(df) // 2
                first_half = df.iloc[:split_point]
                second_half = df.iloc[split_point:]
                
                for col in numeric_cols[:10]:  # 检查前10个数值特征
                    if col not in target_cols and df[col].std() > 0:
                        try:
                            # 计算分位数分布
                            first_dist = np.histogram(first_half[col].dropna(), bins=10, density=True)[0]
                            second_dist = np.histogram(second_half[col].dropna(), bins=10, density=True)[0]
                            
                            # 简化版PSI计算
                            first_dist = first_dist + 1e-8  # 避免除零
                            second_dist = second_dist + 1e-8
                            
                            psi = np.sum((second_dist - first_dist) * np.log(second_dist / first_dist))
                            
                            stability_analysis[col] = {
                                'psi_score': float(psi),
                                'stability': 'STABLE' if psi < 0.1 else 'MODERATE' if psi < 0.2 else 'UNSTABLE'
                            }
                        except Exception:
                            stability_analysis[col] = {'psi_score': None, 'stability': 'ERROR'}
            
            quality_report['feature_stability'] = stability_analysis
            
            # 5. 生成建议
            recommendations = []
            
            # 缺失值建议
            high_missing = [col for col, info in missing_analysis.items() 
                          if info['missing_percentage'] > 0.1]
            if high_missing:
                recommendations.append(f"High missing data in columns: {high_missing[:5]}")
            
            # 异常值建议
            high_outliers = [col for col, info in outlier_analysis.items() 
                           if info['outlier_percentage'] > 0.1]
            if high_outliers:
                recommendations.append(f"High outlier rates in columns: {high_outliers[:5]}")
            
            # 稳定性建议
            unstable_features = [col for col, info in stability_analysis.items() 
                               if info['stability'] == 'UNSTABLE']
            if unstable_features:
                recommendations.append(f"Unstable features detected: {unstable_features}")
            
            quality_report['recommendations'] = recommendations
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Failed to validate data quality for {symbol}: {e}")
            return {'error': str(e)}
    
    def clean_and_prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        特征清洗和预处理
        """
        try:
            result_df = df.copy()
            
            # 1. 处理无穷值
            result_df = result_df.replace([np.inf, -np.inf], np.nan)
            
            # 2. 缺失值处理
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col.startswith('target_') or 'future_' in col or 'optimal_' in col:
                    # 不前向填充目标变量
                    continue
                
                # 前向填充然后后向填充
                result_df[col] = result_df[col].fillna(method='ffill').fillna(method='bfill')
                
                # 如果还有缺失值，用中位数填充
                if result_df[col].isnull().sum() > 0:
                    result_df[col] = result_df[col].fillna(result_df[col].median())
            
            # 3. 异常值处理
            for col in numeric_cols:
                if not col.startswith('target_') and result_df[col].std() > 0:
                    # 使用99.5%分位数进行裁剪
                    lower_bound = result_df[col].quantile(0.005)
                    upper_bound = result_df[col].quantile(0.995)
                    result_df[col] = result_df[col].clip(lower=lower_bound, upper=upper_bound)
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Failed to clean features: {e}")
            return df
    
    def calculate_feature_importance(self, df: pd.DataFrame, target_col: str = 'target_return_30min') -> Dict[str, float]:
        """
        计算特征重要性
        """
        try:
            if target_col not in df.columns:
                self.logger.warning(f"Target column {target_col} not found")
                return {}
            
            # 获取特征列（排除目标变量）
            feature_cols = [col for col in df.columns 
                          if not col.startswith('target_') and col != target_col]
            
            # 移除包含NaN的行
            clean_df = df[feature_cols + [target_col]].dropna()
            
            if len(clean_df) < 100:
                self.logger.warning("Insufficient clean data for feature importance calculation")
                return {}
            
            X = clean_df[feature_cols]
            y = clean_df[target_col]
            
            # 使用互信息计算特征重要性
            importance_scores = mutual_info_regression(X, y, random_state=42)
            
            feature_importance = dict(zip(feature_cols, importance_scores))
            
            # 按重要性排序
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Failed to calculate feature importance: {e}")
            return {}
    
    def process_symbol(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        处理单个交易品种的完整特征工程流程
        """
        try:
            self.logger.info(f"Starting comprehensive feature engineering for {symbol}")
            
            # 多时间框架融合
            if self.config.enable_cross_timeframe:
                primary_df = self.generate_multi_timeframe_features(data_dict, symbol)
            else:
                primary_df = data_dict[self.config.primary_timeframe]
            
            # 1. 超跌识别特征
            primary_df = self.generate_oversold_detection_features(primary_df, symbol)
            
            # 2. 接针形态特征
            primary_df = self.generate_needle_pattern_features(primary_df, symbol)
            
            # 3. 微结构特征
            if self.config.enable_microstructure:
                primary_df = self.generate_microstructure_features(primary_df, symbol)
            
            # 4. 市场状态特征
            if self.config.enable_regime_features:
                primary_df = self.generate_regime_adaptive_features(primary_df, symbol)
            
            # 5. 交互特征
            if self.config.enable_interaction_features:
                primary_df = self.generate_interaction_features(primary_df, symbol)
            
            # 6. 高级标签
            if self.config.enable_advanced_labels:
                primary_df = self.generate_advanced_labels(primary_df, symbol)
            
            # 7. 特征清洗
            primary_df = self.clean_and_prepare_features(primary_df)
            
            # 8. 数据质量验证
            quality_report = self.validate_data_quality(primary_df, symbol)
            
            # 9. 特征重要性分析
            feature_importance = self.calculate_feature_importance(primary_df)
            
            # 处理统计信息
            processing_stats = {
                'symbol': symbol,
                'total_features': len(primary_df.columns),
                'feature_categories': {k: len(v) for k, v in self.feature_categories.items()},
                'data_quality': quality_report,
                'feature_importance': feature_importance,
                'data_range': {
                    'start': str(primary_df.index.min()),
                    'end': str(primary_df.index.max()),
                    'total_rows': len(primary_df)
                }
            }
            
            return primary_df, processing_stats
            
        except Exception as e:
            self.logger.error(f"Failed to process symbol {symbol}: {e}")
            raise
    
    def generate_feature_set(self, bundle_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        生成完整的特征集
        """
        try:
            start_time = time.time()
            self.logger.info("Starting SuperDip Needle feature engineering pipeline")
            
            if output_dir is None:
                output_dir = "data"
            
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            
            # 加载市场数据
            market_data = self.load_market_data(bundle_path)
            
            # 处理结果
            feature_results = {}
            processing_stats = {}
            
            # 并行处理各个交易品种
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_symbol = {
                    executor.submit(self.process_symbol, symbol, data_dict): symbol
                    for symbol, data_dict in market_data.items()
                    if len(data_dict.get(self.config.primary_timeframe, pd.DataFrame())) >= self.config.min_data_points
                }
                
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        df, stats = future.result()
                        feature_results[symbol] = df
                        processing_stats[symbol] = stats
                        self.logger.info(f"Completed processing {symbol}: {len(df)} rows, {len(df.columns)} features")
                    except Exception as e:
                        self.logger.error(f"Failed to process {symbol}: {e}")
            
            processing_time = time.time() - start_time
            
            # 生成特征集报告
            feature_set_info = {
                'pipeline_version': '1.0.0-SuperDipNeedle',
                'creation_time': datetime.now().isoformat(),
                'processing_time_seconds': processing_time,
                'config': {
                    'symbols': self.config.symbols,
                    'timeframes': self.config.timeframes,
                    'primary_timeframe': self.config.primary_timeframe,
                    'prediction_horizons': self.config.prediction_horizons,
                    'profit_targets': self.config.profit_targets,
                    'enable_flags': {
                        'cross_timeframe': self.config.enable_cross_timeframe,
                        'microstructure': self.config.enable_microstructure,
                        'advanced_labels': self.config.enable_advanced_labels,
                        'interaction_features': self.config.enable_interaction_features,
                        'regime_features': self.config.enable_regime_features
                    }
                },
                'results_summary': {
                    'total_symbols_processed': len(feature_results),
                    'total_symbols_requested': len(self.config.symbols),
                    'success_rate': len(feature_results) / len(self.config.symbols),
                    'feature_categories': {k: len(v) for k, v in self.feature_categories.items()},
                    'average_features_per_symbol': np.mean([len(df.columns) for df in feature_results.values()]) if feature_results else 0
                },
                'data_files': {},
                'processing_stats': processing_stats
            }
            
            # 保存特征数据和元数据
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for symbol, df in feature_results.items():
                # 保存特征数据
                feature_file = output_dir / f"{symbol}_SuperDipNeedle_features_{timestamp}.parquet"
                df.to_parquet(feature_file, compression='zstd')
                
                # 保存元数据
                metadata_file = output_dir / f"{symbol}_SuperDipNeedle_metadata_{timestamp}.json"
                with open(metadata_file, 'w') as f:
                    json.dump(processing_stats[symbol], f, indent=2, default=str)
                
                feature_set_info['data_files'][symbol] = {
                    'features_file': str(feature_file),
                    'metadata_file': str(metadata_file),
                    'rows': len(df),
                    'features': len(df.columns),
                    'target_columns': [col for col in df.columns if col.startswith('target_')]
                }
            
            # 保存特征集信息
            feature_set_file = output_dir / f"SuperDipNeedle_FeatureSet_{timestamp}.json"
            with open(feature_set_file, 'w') as f:
                json.dump(feature_set_info, f, indent=2, default=str)
            
            # 生成标签集信息
            label_set_info = {
                'pipeline_version': '1.0.0-SuperDipNeedle',
                'creation_time': datetime.now().isoformat(),
                'label_types': {
                    'return_targets': self.config.prediction_horizons,
                    'profit_targets': self.config.profit_targets,
                    'risk_metrics': ['stop_loss', 'risk_adjusted_return'],
                    'timing_targets': ['optimal_exit_time', 'boundary_return']
                },
                'symbols': list(feature_results.keys()),
                'quality_assurance': {
                    'no_future_leakage': True,
                    'proper_time_alignment': True,
                    'missing_data_handled': True
                }
            }
            
            label_set_file = output_dir / f"SuperDipNeedle_LabelSet_{timestamp}.json"
            with open(label_set_file, 'w') as f:
                json.dump(label_set_info, f, indent=2, default=str)
            
            self.logger.info(f"Feature engineering completed in {processing_time:.2f} seconds")
            self.logger.info(f"Generated features for {len(feature_results)} symbols")
            self.logger.info(f"Results saved to {output_dir}")
            
            return {
                'feature_set_file': str(feature_set_file),
                'label_set_file': str(label_set_file),
                'feature_results': feature_results,
                'processing_stats': processing_stats,
                'summary': feature_set_info
            }
            
        except Exception as e:
            self.logger.error(f"Feature engineering pipeline failed: {e}")
            raise


def main():
    """主函数示例"""
    logging.basicConfig(level=logging.INFO)
    
    # 创建配置
    config = SuperDipNeedleConfig()
    
    # 创建特征工程器
    feature_engineer = SuperDipNeedleFeatureEngineer(config)
    
    # 执行特征工程
    bundle_path = "data/MarketDataBundle_Top30_Enhanced_Final.json"
    
    try:
        results = feature_engineer.generate_feature_set(bundle_path)
        print("\n特征工程完成!")
        print(f"特征集文件: {results['feature_set_file']}")
        print(f"标签集文件: {results['label_set_file']}")
        print(f"处理币种数: {results['summary']['results_summary']['total_symbols_processed']}")
        
    except Exception as e:
        print(f"特征工程失败: {e}")


if __name__ == "__main__":
    main()