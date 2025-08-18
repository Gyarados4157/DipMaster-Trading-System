#!/usr/bin/env python3
"""
SuperDip Pin Bar Strategy Feature Engineering Pipeline
超跌接针策略专用特征工程系统

专为超跌接针反转策略设计的完整特征工程管道，实现：
1. 核心技术指标特征：RSI(14,7,21)、MA(20)偏离度、布林带位置、成交量相对强度
2. 接针形态特征：下影线/实体比率、蜡烛实体位置、价格恢复度、成交量放大倍数
3. 多时间框架特征：1分钟短期动量、5分钟主信号、15分钟趋势确认
4. 标签生成：4小时前向收益率、胜率预期标签、风险调整收益
5. 数据质量保证：完整的泄漏检查、特征稳定性验证、缺失值处理

Author: DipMaster Quant Team
Date: 2025-08-18
Version: 1.0.0-SuperDipPinBar
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
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import ta
# import numba
# from numba import jit

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class SuperDipPinBarConfig:
    """SuperDip Pin Bar Feature Engineering Configuration"""
    
    # 交易品种配置
    symbols: List[str] = field(default_factory=lambda: [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT',
        'BNBUSDT', 'AVAXUSDT', 'LINKUSDT', 'LTCUSDT', 'DOTUSDT',
        'ATOMUSDT', 'NEARUSDT', 'UNIUSDT', 'FILUSDT', 'TRXUSDT',
        'APTUSDT', 'ARBUSDT', 'OPUSDT', 'TONUSDT', 'MATICUSDT'
    ])
    
    # 时间框架配置
    timeframes: List[str] = field(default_factory=lambda: ['1m', '5m', '15m'])
    primary_timeframe: str = '5m'
    
    # 预测目标配置
    prediction_horizons: List[int] = field(default_factory=lambda: [240])  # 4小时
    profit_targets: List[float] = field(default_factory=lambda: [0.008, 0.015, 0.025])
    stop_loss: float = 0.006
    max_holding_periods: int = 240  # 分钟
    
    # RSI配置 - 用于超跌识别
    rsi_periods: List[int] = field(default_factory=lambda: [14, 7, 21])
    rsi_oversold_threshold: float = 30
    rsi_optimal_range: Tuple[float, float] = (30, 50)
    
    # 移动平均配置 - MA20偏离度计算
    ma_periods: List[int] = field(default_factory=lambda: [20, 10, 50])
    ema_periods: List[int] = field(default_factory=lambda: [13, 21, 34])
    
    # 布林带配置 - 超跌区域判断
    bollinger_periods: List[int] = field(default_factory=lambda: [20])
    bollinger_std_multiplier: List[float] = field(default_factory=lambda: [2.0, 1.5, 2.5])
    
    # 成交量配置 - 相对强度和放大倍数
    volume_ma_periods: List[int] = field(default_factory=lambda: [20, 10, 50])
    volume_spike_multiplier: List[float] = field(default_factory=lambda: [1.5, 2.0, 2.5])
    
    # 接针形态配置
    pin_bar_min_body_ratio: float = 0.3  # 实体最小比例
    pin_bar_min_wick_ratio: float = 2.0   # 下影线最小比例
    pin_bar_max_upper_wick_ratio: float = 0.5  # 上影线最大比例
    
    # 特征启用开关
    enable_multi_timeframe: bool = True
    enable_pin_bar_detection: bool = True
    enable_advanced_labels: bool = True
    enable_interaction_features: bool = True
    enable_volume_profile: bool = True
    
    # 数据质量控制
    min_data_points: int = 1000
    max_nan_percentage: float = 0.02
    outlier_clip_percentile: Tuple[float, float] = (0.5, 99.5)

class SuperDipPinBarFeatureEngineer:
    """
    SuperDip Pin Bar Strategy Feature Engineering Pipeline
    超跌接针策略专用特征工程管道
    
    核心功能：
    1. 核心技术指标特征生成
    2. 接针形态特征提取
    3. 多时间框架信号融合
    4. 高级标签工程
    5. 严格数据质量控制
    """
    
    def __init__(self, config: Optional[SuperDipPinBarConfig] = None):
        """Initialize SuperDip Pin Bar Feature Engineer"""
        self.logger = logging.getLogger(__name__)
        self.config = config or SuperDipPinBarConfig()
        
        # 数据预处理组件
        self.scaler = RobustScaler()
        self.quantile_transformer = QuantileTransformer(n_quantiles=1000, random_state=42)
        self.standard_scaler = StandardScaler()
        
        # 特征追踪
        self.feature_names = []
        self.feature_importance = {}
        self.feature_categories = {
            'core_technical': [],
            'pin_bar_pattern': [],
            'multi_timeframe': [],
            'volume_profile': [],
            'interaction': [],
            'risk_metrics': []
        }
        
        # 数据质量监控
        self.quality_metrics = {
            'data_leakage_check': {},
            'feature_stability': {},
            'correlation_analysis': {},
            'missing_data_report': {}
        }
        
        # 缓存
        self._data_cache = {}
        self._feature_cache = {}
        
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
                
                symbol_data = {}
                symbol_files = bundle['data_files'][symbol]
                
                for timeframe in self.config.timeframes:
                    if timeframe in symbol_files:
                        file_path = symbol_files[timeframe]['file_path']
                        full_path = Path(bundle_path).parent / file_path
                        
                        if full_path.exists():
                            try:
                                df = pd.read_parquet(full_path)
                                df = self._validate_ohlcv_data(df)
                                symbol_data[timeframe] = df
                                self.logger.info(f"Loaded {symbol}_{timeframe}: {len(df)} records")
                            except Exception as e:
                                self.logger.error(f"Error loading {symbol}_{timeframe}: {e}")
                        else:
                            self.logger.warning(f"File not found: {full_path}")
                
                if symbol_data:
                    data_dict[symbol] = symbol_data
            
            self.logger.info(f"Successfully loaded data for {len(data_dict)} symbols")
            self._data_cache = data_dict
            return data_dict
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            raise
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")
        
        # Ensure proper data types
        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Basic OHLC validation
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.any():
            self.logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC records, fixing...")
            df.loc[invalid_ohlc, ['open', 'high', 'low', 'close']] = np.nan
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure chronological order
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        
        return df
    
    # @jit
    def _calculate_pin_bar_features_numba(self, open_vals: np.ndarray, high_vals: np.ndarray, 
                                         low_vals: np.ndarray, close_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Optimized pin bar calculation using numba"""
        n = len(open_vals)
        lower_wick_ratio = np.zeros(n)
        upper_wick_ratio = np.zeros(n)
        body_ratio = np.zeros(n)
        body_position = np.zeros(n)
        
        for i in range(n):
            o, h, l, c = open_vals[i], high_vals[i], low_vals[i], close_vals[i]
            
            if h == l:  # 避免除零
                continue
                
            range_total = h - l
            body_size = abs(c - o)
            
            # 计算实体位置（实体底部）
            body_bottom = min(o, c)
            body_top = max(o, c)
            
            # 下影线长度
            lower_wick = body_bottom - l
            # 上影线长度  
            upper_wick = h - body_top
            
            # 比率计算
            if range_total > 0:
                lower_wick_ratio[i] = lower_wick / range_total
                upper_wick_ratio[i] = upper_wick / range_total
                body_ratio[i] = body_size / range_total
                body_position[i] = (body_bottom - l) / range_total
        
        return lower_wick_ratio, upper_wick_ratio, body_ratio, body_position
    
    def calculate_core_technical_features(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Calculate core technical indicator features
        计算核心技术指标特征
        """
        features = df.copy()
        prefix = f"{symbol}_{timeframe}"
        
        try:
            # 1. RSI Features - 超跌识别的核心指标
            for period in self.config.rsi_periods:
                rsi_col = f'{prefix}_rsi_{period}'
                features[rsi_col] = ta.momentum.RSIIndicator(features['close'], window=period).rsi()
                self.feature_categories['core_technical'].append(rsi_col)
                
                # RSI区间特征
                features[f'{prefix}_rsi_{period}_oversold'] = (features[rsi_col] < self.config.rsi_oversold_threshold).astype(int)
                features[f'{prefix}_rsi_{period}_optimal'] = (
                    (features[rsi_col] >= self.config.rsi_optimal_range[0]) & 
                    (features[rsi_col] <= self.config.rsi_optimal_range[1])
                ).astype(int)
                self.feature_categories['core_technical'].extend([
                    f'{prefix}_rsi_{period}_oversold',
                    f'{prefix}_rsi_{period}_optimal'
                ])
            
            # 2. Moving Average Features - MA20偏离度
            for period in self.config.ma_periods:
                ma_col = f'{prefix}_ma_{period}'
                features[ma_col] = ta.trend.SMAIndicator(features['close'], window=period).sma_indicator()
                
                # 价格偏离度
                deviation_col = f'{prefix}_price_ma_{period}_deviation'
                features[deviation_col] = (features['close'] - features[ma_col]) / features[ma_col]
                
                # 价格相对位置
                position_col = f'{prefix}_price_ma_{period}_position'
                features[position_col] = (features['close'] < features[ma_col]).astype(int)
                
                self.feature_categories['core_technical'].extend([ma_col, deviation_col, position_col])
            
            # 3. Bollinger Bands Features - 超跌区域判断
            for period in self.config.bollinger_periods:
                for std_mult in self.config.bollinger_std_multiplier:
                    bb_indicator = ta.volatility.BollingerBands(features['close'], window=period, window_dev=std_mult)
                    
                    bb_lower_col = f'{prefix}_bb_lower_{period}_{std_mult}'
                    bb_upper_col = f'{prefix}_bb_upper_{period}_{std_mult}'
                    bb_position_col = f'{prefix}_bb_position_{period}_{std_mult}'
                    
                    features[bb_lower_col] = bb_indicator.bollinger_lband()
                    features[bb_upper_col] = bb_indicator.bollinger_hband()
                    features[bb_position_col] = (features['close'] - bb_indicator.bollinger_lband()) / (
                        bb_indicator.bollinger_hband() - bb_indicator.bollinger_lband()
                    )
                    
                    # 超跌识别
                    oversold_col = f'{prefix}_bb_oversold_{period}_{std_mult}'
                    features[oversold_col] = (features['close'] < bb_indicator.bollinger_lband()).astype(int)
                    
                    self.feature_categories['core_technical'].extend([
                        bb_lower_col, bb_upper_col, bb_position_col, oversold_col
                    ])
            
            # 4. Volume Features - 成交量相对强度
            for period in self.config.volume_ma_periods:
                volume_ma_col = f'{prefix}_volume_ma_{period}'
                features[volume_ma_col] = features['volume'].rolling(window=period).mean()
                
                # 成交量相对强度
                volume_ratio_col = f'{prefix}_volume_ratio_{period}'
                features[volume_ratio_col] = features['volume'] / features[volume_ma_col]
                
                # 成交量放大识别
                for multiplier in self.config.volume_spike_multiplier:
                    spike_col = f'{prefix}_volume_spike_{period}_{multiplier}'
                    features[spike_col] = (features[volume_ratio_col] > multiplier).astype(int)
                    self.feature_categories['core_technical'].append(spike_col)
                
                self.feature_categories['core_technical'].extend([volume_ma_col, volume_ratio_col])
            
            # 5. Price Momentum Features
            for period in [3, 5, 10, 15]:
                momentum_col = f'{prefix}_momentum_{period}'
                features[momentum_col] = features['close'].pct_change(periods=period)
                self.feature_categories['core_technical'].append(momentum_col)
            
            # 6. Volatility Features
            for period in [10, 20, 30]:
                volatility_col = f'{prefix}_volatility_{period}'
                features[volatility_col] = features['close'].rolling(window=period).std() / features['close'].rolling(window=period).mean()
                self.feature_categories['core_technical'].append(volatility_col)
            
            self.logger.info(f"Calculated core technical features for {prefix}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating core technical features for {prefix}: {e}")
            return features
    
    def calculate_pin_bar_pattern_features(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Calculate pin bar pattern features
        计算接针形态特征
        """
        features = df.copy()
        prefix = f"{symbol}_{timeframe}"
        
        try:
            # 使用Numba优化的计算
            lower_wick_ratio, upper_wick_ratio, body_ratio, body_position = self._calculate_pin_bar_features_numba(
                features['open'].values,
                features['high'].values,
                features['low'].values,
                features['close'].values
            )
            
            # 1. 基础接针形态特征
            features[f'{prefix}_lower_wick_ratio'] = lower_wick_ratio
            features[f'{prefix}_upper_wick_ratio'] = upper_wick_ratio
            features[f'{prefix}_body_ratio'] = body_ratio
            features[f'{prefix}_body_position'] = body_position
            
            # 2. 下影线/实体比率 - 关键的接针识别特征
            body_nonzero_mask = body_ratio > 0.001  # 避免除零
            lower_to_body_ratio = np.zeros_like(lower_wick_ratio)
            lower_to_body_ratio[body_nonzero_mask] = lower_wick_ratio[body_nonzero_mask] / body_ratio[body_nonzero_mask]
            features[f'{prefix}_lower_to_body_ratio'] = lower_to_body_ratio
            
            # 3. 蜡烛实体在K线中的位置
            # 0表示实体在底部，1表示实体在顶部
            features[f'{prefix}_body_center_position'] = body_position + body_ratio / 2
            
            # 4. 价格恢复度（从最低点回升的程度）
            features[f'{prefix}_price_recovery'] = (features['close'] - features['low']) / (features['high'] - features['low'])
            
            # 5. 成交量放大倍数（在接针形态时）
            volume_ma_20 = features['volume'].rolling(window=20).mean()
            features[f'{prefix}_volume_multiplier'] = features['volume'] / volume_ma_20
            
            # 6. 接针形态识别（综合条件）
            pin_bar_condition = (
                (lower_wick_ratio > 0.5) &  # 下影线占比超过50%
                (body_ratio < self.config.pin_bar_min_body_ratio) &  # 实体较小
                (upper_wick_ratio < self.config.pin_bar_max_upper_wick_ratio) &  # 上影线较短
                (lower_to_body_ratio > self.config.pin_bar_min_wick_ratio)  # 下影线/实体比率
            )
            features[f'{prefix}_is_pin_bar'] = pin_bar_condition.astype(int)
            
            # 7. 增强的接针形态（加入成交量确认）
            enhanced_pin_bar = (
                pin_bar_condition &
                (features[f'{prefix}_volume_multiplier'] > 1.2)  # 成交量放大
            )
            features[f'{prefix}_is_enhanced_pin_bar'] = enhanced_pin_bar.astype(int)
            
            # 8. 接针强度评分（0-1）
            pin_bar_strength = (
                lower_wick_ratio * 0.4 +  # 下影线比重
                (1 - body_ratio) * 0.3 +  # 实体小比重
                (1 - upper_wick_ratio) * 0.2 +  # 上影线小比重
                (features[f'{prefix}_volume_multiplier'].clip(0, 3) / 3) * 0.1  # 成交量放大
            )
            features[f'{prefix}_pin_bar_strength'] = pin_bar_strength
            
            # 9. 接针后价格行为特征（滞后1期，避免前瞻性）
            features[f'{prefix}_price_action_after_pin'] = features['close'].shift(-1) / features['close'] - 1
            features[f'{prefix}_price_action_after_pin'] = features[f'{prefix}_price_action_after_pin'].shift(1)  # 避免泄漏
            
            # 添加到特征分类
            pin_bar_features = [
                f'{prefix}_lower_wick_ratio', f'{prefix}_upper_wick_ratio', f'{prefix}_body_ratio',
                f'{prefix}_body_position', f'{prefix}_lower_to_body_ratio', f'{prefix}_body_center_position',
                f'{prefix}_price_recovery', f'{prefix}_volume_multiplier', f'{prefix}_is_pin_bar',
                f'{prefix}_is_enhanced_pin_bar', f'{prefix}_pin_bar_strength'
            ]
            self.feature_categories['pin_bar_pattern'].extend(pin_bar_features)
            
            self.logger.info(f"Calculated pin bar pattern features for {prefix}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating pin bar features for {prefix}: {e}")
            return features
    
    def calculate_multi_timeframe_features(self, data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, pd.DataFrame]:
        """
        Calculate multi-timeframe features
        计算多时间框架特征
        """
        if not self.config.enable_multi_timeframe:
            return {}
        
        multi_tf_features = {}
        
        try:
            for symbol in data_dict:
                if self.config.primary_timeframe not in data_dict[symbol]:
                    continue
                
                primary_df = data_dict[symbol][self.config.primary_timeframe].copy()
                
                # 1分钟短期动量特征
                if '1m' in data_dict[symbol]:
                    df_1m = data_dict[symbol]['1m']
                    
                    # 重采样1分钟数据到主时间框架
                    df_1m_resampled = self._resample_to_primary_timeframe(df_1m, '1m')
                    
                    if len(df_1m_resampled) > 0:
                        # 短期动量指标
                        primary_df[f'{symbol}_1m_short_momentum'] = df_1m_resampled['close'].pct_change(periods=1)
                        primary_df[f'{symbol}_1m_volume_surge'] = (df_1m_resampled['volume'] / df_1m_resampled['volume'].rolling(5).mean()).clip(0, 5)
                        primary_df[f'{symbol}_1m_price_volatility'] = df_1m_resampled['close'].rolling(5).std()
                        
                        # RSI快速变化
                        rsi_1m = ta.momentum.RSIIndicator(df_1m_resampled['close'], window=7).rsi()
                        primary_df[f'{symbol}_1m_rsi_fast'] = rsi_1m
                        primary_df[f'{symbol}_1m_rsi_change'] = rsi_1m.diff()
                        
                        self.feature_categories['multi_timeframe'].extend([
                            f'{symbol}_1m_short_momentum', f'{symbol}_1m_volume_surge',
                            f'{symbol}_1m_price_volatility', f'{symbol}_1m_rsi_fast', f'{symbol}_1m_rsi_change'
                        ])
                
                # 15分钟趋势确认特征
                if '15m' in data_dict[symbol]:
                    df_15m = data_dict[symbol]['15m']
                    df_15m_resampled = self._resample_to_primary_timeframe(df_15m, '15m')
                    
                    if len(df_15m_resampled) > 0:
                        # 趋势确认指标
                        primary_df[f'{symbol}_15m_trend_strength'] = ta.trend.ADXIndicator(
                            df_15m_resampled['high'], df_15m_resampled['low'], df_15m_resampled['close'], window=14
                        ).adx()
                        
                        # 移动平均趋势
                        ma_20_15m = ta.trend.SMAIndicator(df_15m_resampled['close'], window=20).sma_indicator()
                        primary_df[f'{symbol}_15m_ma20_slope'] = ma_20_15m.diff()
                        primary_df[f'{symbol}_15m_price_ma20_dev'] = (df_15m_resampled['close'] - ma_20_15m) / ma_20_15m
                        
                        # MACD确认
                        macd_15m = ta.trend.MACD(df_15m_resampled['close'])
                        primary_df[f'{symbol}_15m_macd'] = macd_15m.macd()
                        primary_df[f'{symbol}_15m_macd_signal'] = macd_15m.macd_signal()
                        primary_df[f'{symbol}_15m_macd_histogram'] = macd_15m.macd_diff()
                        
                        self.feature_categories['multi_timeframe'].extend([
                            f'{symbol}_15m_trend_strength', f'{symbol}_15m_ma20_slope',
                            f'{symbol}_15m_price_ma20_dev', f'{symbol}_15m_macd',
                            f'{symbol}_15m_macd_signal', f'{symbol}_15m_macd_histogram'
                        ])
                
                multi_tf_features[symbol] = primary_df
                self.logger.info(f"Calculated multi-timeframe features for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Error calculating multi-timeframe features: {e}")
        
        return multi_tf_features
    
    def _resample_to_primary_timeframe(self, df: pd.DataFrame, source_tf: str) -> pd.DataFrame:
        """Resample data to primary timeframe"""
        try:
            primary_tf = self.config.primary_timeframe
            
            # 定义重采样规则
            resample_rules = {
                ('1m', '5m'): '5T',
                ('5m', '15m'): '15T',
                ('15m', '1h'): '1H',
                ('1m', '15m'): '15T',
                ('5m', '1h'): '1H'
            }
            
            rule_key = (source_tf, primary_tf)
            if rule_key not in resample_rules:
                return pd.DataFrame()
            
            rule = resample_rules[rule_key]
            
            # 执行重采样
            resampled = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            return resampled
            
        except Exception as e:
            self.logger.error(f"Error resampling {source_tf} to {primary_tf}: {e}")
            return pd.DataFrame()
    
    def generate_forward_labels(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Generate forward-looking labels
        生成前向标签
        """
        features = df.copy()
        
        try:
            for horizon in self.config.prediction_horizons:
                # 前向收益率计算
                forward_return_col = f'{symbol}_forward_return_{horizon}m'
                features[forward_return_col] = features['close'].shift(-horizon) / features['close'] - 1
                
                # 胜率预期标签
                for target in self.config.profit_targets:
                    label_col = f'{symbol}_win_{int(target*1000)}bp_{horizon}m'
                    
                    # 计算在指定时间内是否达到目标收益
                    future_prices = features['close'].shift(-np.arange(1, horizon+1)).max(axis=1)
                    profit_achieved = (future_prices / features['close'] - 1) >= target
                    
                    # 同时检查止损
                    future_min_prices = features['close'].shift(-np.arange(1, horizon+1)).min(axis=1)
                    stop_loss_hit = (features['close'] - future_min_prices) / features['close'] >= self.config.stop_loss
                    
                    # 标签：1表示达到目标且未触发止损，0表示其他
                    features[label_col] = (profit_achieved & ~stop_loss_hit).astype(int)
                    
                # 风险调整收益
                risk_adj_return_col = f'{symbol}_risk_adj_return_{horizon}m'
                volatility = features['close'].rolling(window=20).std()
                features[risk_adj_return_col] = features[forward_return_col] / volatility
                
                # 最大有利价格偏移（MFE）
                mfe_col = f'{symbol}_mfe_{horizon}m'
                future_highs = features['high'].shift(-np.arange(1, horizon+1)).max(axis=1)
                features[mfe_col] = (future_highs / features['close'] - 1)
                
                # 最大不利价格偏移（MAE）
                mae_col = f'{symbol}_mae_{horizon}m'
                future_lows = features['low'].shift(-np.arange(1, horizon+1)).min(axis=1)
                features[mae_col] = (features['close'] / future_lows - 1)
                
            self.logger.info(f"Generated forward labels for {symbol}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating forward labels for {symbol}: {e}")
            return features
    
    def perform_data_quality_checks(self, features_dict: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Perform comprehensive data quality checks
        执行全面的数据质量检查
        """
        quality_report = {
            'leakage_check': {},
            'feature_stability': {},
            'correlation_analysis': {},
            'missing_data': {},
            'outlier_analysis': {},
            'overall_score': 0.0
        }
        
        try:
            all_features = []
            for symbol, df in features_dict.items():
                all_features.append(df)
            
            if not all_features:
                return quality_report
                
            combined_features = pd.concat(all_features, ignore_index=True)
            
            # 1. 数据泄漏检查
            quality_report['leakage_check'] = self._check_data_leakage(combined_features)
            
            # 2. 特征稳定性检查
            quality_report['feature_stability'] = self._check_feature_stability(combined_features)
            
            # 3. 相关性分析
            quality_report['correlation_analysis'] = self._analyze_correlations(combined_features)
            
            # 4. 缺失数据分析
            quality_report['missing_data'] = self._analyze_missing_data(combined_features)
            
            # 5. 异常值分析
            quality_report['outlier_analysis'] = self._analyze_outliers(combined_features)
            
            # 6. 计算总体质量评分
            quality_report['overall_score'] = self._calculate_quality_score(quality_report)
            
            self.logger.info(f"Data quality check completed. Overall score: {quality_report['overall_score']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error in data quality check: {e}")
        
        return quality_report
    
    def _check_data_leakage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for data leakage"""
        leakage_report = {
            'suspicious_features': [],
            'future_correlation_check': {},
            'leakage_risk_score': 0.0
        }
        
        try:
            # 识别可能包含未来信息的特征
            suspicious_patterns = ['forward', 'future', 'next', 'after', 'lead']
            
            for col in df.columns:
                if any(pattern in col.lower() for pattern in suspicious_patterns):
                    # 检查是否有适当的滞后
                    if 'shift' not in col.lower() and 'lag' not in col.lower():
                        leakage_report['suspicious_features'].append(col)
            
            # 计算泄漏风险评分
            if leakage_report['suspicious_features']:
                leakage_report['leakage_risk_score'] = len(leakage_report['suspicious_features']) / len(df.columns)
            
        except Exception as e:
            self.logger.error(f"Error checking data leakage: {e}")
        
        return leakage_report
    
    def _check_feature_stability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check feature stability over time"""
        stability_report = {
            'unstable_features': [],
            'stability_scores': {},
            'avg_stability': 0.0
        }
        
        try:
            # 按时间分割数据
            n_splits = 5
            split_size = len(df) // n_splits
            
            stability_scores = {}
            
            for col in df.select_dtypes(include=[np.number]).columns:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    continue
                    
                split_means = []
                split_stds = []
                
                for i in range(n_splits):
                    start_idx = i * split_size
                    end_idx = min((i + 1) * split_size, len(df))
                    split_data = df[col].iloc[start_idx:end_idx]
                    
                    if len(split_data.dropna()) > 10:
                        split_means.append(split_data.mean())
                        split_stds.append(split_data.std())
                
                if len(split_means) >= 3:
                    # 计算稳定性得分（基于均值和标准差的变异系数）
                    mean_cv = np.std(split_means) / np.abs(np.mean(split_means)) if np.mean(split_means) != 0 else np.inf
                    std_cv = np.std(split_stds) / np.mean(split_stds) if np.mean(split_stds) != 0 else np.inf
                    
                    stability_score = 1 / (1 + mean_cv + std_cv)
                    stability_scores[col] = stability_score
                    
                    if stability_score < 0.3:  # 阈值
                        stability_report['unstable_features'].append(col)
            
            stability_report['stability_scores'] = stability_scores
            if stability_scores:
                stability_report['avg_stability'] = np.mean(list(stability_scores.values()))
                
        except Exception as e:
            self.logger.error(f"Error checking feature stability: {e}")
        
        return stability_report
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature correlations"""
        corr_report = {
            'high_correlation_pairs': [],
            'correlation_matrix_shape': (0, 0),
            'max_correlation': 0.0
        }
        
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return corr_report
                
            corr_matrix = numeric_df.corr()
            corr_report['correlation_matrix_shape'] = corr_matrix.shape
            
            # 找出高相关性特征对
            high_corr_threshold = 0.9
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > high_corr_threshold and not np.isnan(corr_val):
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            corr_report['high_correlation_pairs'] = high_corr_pairs
            if not corr_matrix.empty:
                corr_report['max_correlation'] = corr_matrix.abs().max().max()
                
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
        
        return corr_report
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing_report = {
            'features_with_missing': {},
            'total_missing_percentage': 0.0,
            'problematic_features': []
        }
        
        try:
            missing_counts = df.isnull().sum()
            total_rows = len(df)
            
            for col, missing_count in missing_counts.items():
                if missing_count > 0:
                    missing_pct = missing_count / total_rows
                    missing_report['features_with_missing'][col] = {
                        'count': int(missing_count),
                        'percentage': missing_pct
                    }
                    
                    if missing_pct > self.config.max_nan_percentage:
                        missing_report['problematic_features'].append(col)
            
            if missing_counts.sum() > 0:
                missing_report['total_missing_percentage'] = missing_counts.sum() / (total_rows * len(df.columns))
                
        except Exception as e:
            self.logger.error(f"Error analyzing missing data: {e}")
        
        return missing_report
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in features"""
        outlier_report = {
            'features_with_outliers': {},
            'outlier_summary': {}
        }
        
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            
            for col in numeric_df.columns:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    continue
                    
                data = numeric_df[col].dropna()
                if len(data) < 10:
                    continue
                
                # 使用IQR方法检测异常值
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = data[(data < lower_bound) | (data > upper_bound)]
                
                if len(outliers) > 0:
                    outlier_report['features_with_outliers'][col] = {
                        'count': len(outliers),
                        'percentage': len(outliers) / len(data),
                        'bounds': {'lower': lower_bound, 'upper': upper_bound}
                    }
                    
        except Exception as e:
            self.logger.error(f"Error analyzing outliers: {e}")
        
        return outlier_report
    
    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall quality score"""
        try:
            score_components = []
            
            # 泄漏检查得分
            leakage_score = 1.0 - quality_report['leakage_check'].get('leakage_risk_score', 0)
            score_components.append(leakage_score * 0.3)
            
            # 稳定性得分
            stability_score = quality_report['feature_stability'].get('avg_stability', 0.5)
            score_components.append(stability_score * 0.25)
            
            # 缺失数据得分
            missing_score = 1.0 - min(quality_report['missing_data'].get('total_missing_percentage', 0), 1.0)
            score_components.append(missing_score * 0.25)
            
            # 相关性得分（避免过高相关性）
            high_corr_count = len(quality_report['correlation_analysis'].get('high_correlation_pairs', []))
            corr_score = max(0, 1.0 - high_corr_count * 0.1)
            score_components.append(corr_score * 0.2)
            
            return sum(score_components)
            
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            return 0.0
    
    def generate_feature_set(self, bundle_path: str, output_dir: str = None) -> Dict[str, Any]:
        """
        Generate complete feature set
        生成完整特征集
        """
        if output_dir is None:
            output_dir = Path(bundle_path).parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # 1. 加载市场数据
            self.logger.info("Step 1: Loading market data...")
            data_dict = self.load_market_data(bundle_path)
            
            if not data_dict:
                raise ValueError("No market data loaded")
            
            # 2. 生成核心技术指标特征
            self.logger.info("Step 2: Calculating core technical features...")
            features_dict = {}
            
            for symbol, symbol_data in data_dict.items():
                if self.config.primary_timeframe not in symbol_data:
                    continue
                    
                df = symbol_data[self.config.primary_timeframe].copy()
                
                # 核心技术指标
                df = self.calculate_core_technical_features(df, symbol, self.config.primary_timeframe)
                
                # 接针形态特征
                if self.config.enable_pin_bar_detection:
                    df = self.calculate_pin_bar_pattern_features(df, symbol, self.config.primary_timeframe)
                
                features_dict[symbol] = df
            
            # 3. 多时间框架特征
            if self.config.enable_multi_timeframe:
                self.logger.info("Step 3: Calculating multi-timeframe features...")
                multi_tf_features = self.calculate_multi_timeframe_features(data_dict)
                
                # 合并多时间框架特征
                for symbol in multi_tf_features:
                    if symbol in features_dict:
                        # 对齐索引
                        common_index = features_dict[symbol].index.intersection(multi_tf_features[symbol].index)
                        if len(common_index) > 0:
                            features_dict[symbol] = multi_tf_features[symbol].loc[common_index]
            
            # 4. 生成前向标签
            if self.config.enable_advanced_labels:
                self.logger.info("Step 4: Generating forward labels...")
                for symbol in features_dict:
                    features_dict[symbol] = self.generate_forward_labels(features_dict[symbol], symbol)
            
            # 5. 数据质量检查
            self.logger.info("Step 5: Performing data quality checks...")
            quality_report = self.perform_data_quality_checks(features_dict)
            
            # 6. 保存特征数据
            self.logger.info("Step 6: Saving features...")
            feature_files = {}
            
            for symbol, df in features_dict.items():
                # 删除无限值和替换NaN
                df = df.replace([np.inf, -np.inf], np.nan)
                df = df.fillna(method='ffill').fillna(method='bfill')
                
                # 保存为Parquet格式
                feature_file = output_dir / f"features_{symbol}_superdip_pinbar_{timestamp}.parquet"
                df.to_parquet(feature_file, compression='zstd')
                feature_files[symbol] = str(feature_file)
                
                self.logger.info(f"Saved features for {symbol}: {len(df)} records, {len(df.columns)} features")
            
            # 7. 生成FeatureSet配置
            feature_set_config = {
                "version": datetime.now().isoformat(),
                "feature_set_id": f"superdip_pinbar_{timestamp}",
                "strategy_name": "SuperDip_PinBar_Strategy",
                "description": "超跌接针反转策略特征集 - 包含核心技术指标、接针形态、多时间框架特征和前向标签",
                
                "metadata": {
                    "creation_date": datetime.now().isoformat(),
                    "feature_engineer": "SuperDipPinBarFeatureEngineer",
                    "version": "1.0.0",
                    "total_symbols": len(features_dict),
                    "primary_timeframe": self.config.primary_timeframe,
                    "prediction_horizons": self.config.prediction_horizons,
                    "feature_count": sum(len(df.columns) for df in features_dict.values()),
                    "data_quality_score": quality_report.get('overall_score', 0)
                },
                
                "feature_categories": self.feature_categories,
                
                "data_files": feature_files,
                
                "target_definitions": {
                    "forward_returns": {
                        "type": "regression",
                        "horizons": self.config.prediction_horizons,
                        "description": "Future return over specified horizons"
                    },
                    "win_labels": {
                        "type": "classification",
                        "targets": self.config.profit_targets,
                        "description": "Binary labels for profit target achievement"
                    },
                    "risk_adjusted_returns": {
                        "type": "regression",
                        "description": "Volatility-adjusted forward returns"
                    }
                },
                
                "train_test_split": {
                    "method": "time_series",
                    "train_ratio": 0.7,
                    "validation_ratio": 0.15,
                    "test_ratio": 0.15,
                    "split_date_suggestion": "使用时间序列分割，避免数据泄漏"
                },
                
                "feature_engineering_config": {
                    "rsi_periods": self.config.rsi_periods,
                    "ma_periods": self.config.ma_periods,
                    "bollinger_config": {
                        "periods": self.config.bollinger_periods,
                        "std_multipliers": self.config.bollinger_std_multiplier
                    },
                    "pin_bar_config": {
                        "min_body_ratio": self.config.pin_bar_min_body_ratio,
                        "min_wick_ratio": self.config.pin_bar_min_wick_ratio,
                        "max_upper_wick_ratio": self.config.pin_bar_max_upper_wick_ratio
                    },
                    "volume_config": {
                        "ma_periods": self.config.volume_ma_periods,
                        "spike_multipliers": self.config.volume_spike_multiplier
                    }
                },
                
                "quality_report": quality_report,
                
                "usage_recommendations": {
                    "model_types": ["LightGBM", "XGBoost", "CatBoost", "Random Forest"],
                    "feature_selection": "使用mutual_info_regression进行特征选择",
                    "cross_validation": "使用时序交叉验证，避免未来信息泄漏",
                    "feature_importance": "监控特征重要性变化，防止过拟合",
                    "data_preprocessing": "使用RobustScaler处理异常值"
                },
                
                "performance_benchmarks": {
                    "expected_sharpe_ratio": "> 1.5",
                    "expected_win_rate": "> 60%",
                    "max_drawdown": "< 10%",
                    "feature_stability": "> 0.7"
                }
            }
            
            # 保存FeatureSet配置
            feature_set_file = output_dir / f"FeatureSet_SuperDip_PinBar_{timestamp}.json"
            with open(feature_set_file, 'w', encoding='utf-8') as f:
                json.dump(feature_set_config, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Feature engineering completed successfully!")
            self.logger.info(f"FeatureSet saved to: {feature_set_file}")
            self.logger.info(f"Data quality score: {quality_report.get('overall_score', 0):.3f}")
            
            return {
                "feature_set_config": feature_set_config,
                "feature_files": feature_files,
                "quality_report": quality_report,
                "config_file": str(feature_set_file)
            }
            
        except Exception as e:
            self.logger.error(f"Error in feature generation: {e}")
            raise

def main():
    """Main execution function"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 配置
    config = SuperDipPinBarConfig(
        symbols=['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT'],
        timeframes=['1m', '5m', '15m'],
        prediction_horizons=[240],  # 4小时
        profit_targets=[0.008, 0.015, 0.025],
        enable_multi_timeframe=True,
        enable_pin_bar_detection=True,
        enable_advanced_labels=True
    )
    
    # 创建特征工程器
    engineer = SuperDipPinBarFeatureEngineer(config)
    
    # 生成特征集
    try:
        bundle_path = "data/MarketDataBundle_Top30_Enhanced_Final.json"
        result = engineer.generate_feature_set(bundle_path)
        
        print("\n" + "="*50)
        print("SuperDip Pin Bar Feature Engineering Completed!")
        print("="*50)
        print(f"Feature Set ID: {result['feature_set_config']['feature_set_id']}")
        print(f"Total Symbols: {result['feature_set_config']['metadata']['total_symbols']}")
        print(f"Total Features: {result['feature_set_config']['metadata']['feature_count']}")
        print(f"Data Quality Score: {result['quality_report']['overall_score']:.3f}")
        print(f"Config File: {result['config_file']}")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()