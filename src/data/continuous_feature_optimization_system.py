#!/usr/bin/env python3
"""
DipMaster持续特征工程优化系统
Continuous Feature Engineering Optimization System

这是一个自适应的特征工程系统，专门为DipMaster策略设计，能够：
1. 持续挖掘新的有效特征
2. 自动评估和优化现有特征
3. 检测特征退化并动态调整
4. 确保严格的数据泄漏检测
5. 生成特征质量监控报告

Author: DipMaster Quant Team
Date: 2025-08-18
Version: 1.0.0-ContinuousOptimization
"""

import pandas as pd
import numpy as np
import warnings
import logging
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from sklearn.feature_selection import (
    mutual_info_regression, mutual_info_classif,
    SelectKBest, RFE, RFECV, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
import lightgbm as lgb
import scipy.stats as stats
import ta
import numba
from numba import jit
import pickle
import warnings

warnings.filterwarnings('ignore')

@dataclass
class FeatureOptimizationConfig:
    """特征优化配置"""
    symbols: List[str]
    feature_update_interval_hours: int = 1
    max_features_per_category: int = 50
    min_feature_importance: float = 0.001
    max_correlation_threshold: float = 0.95
    stability_threshold: float = 0.8  # PSI threshold
    validation_window_days: int = 30
    innovation_rate: float = 0.1  # 10% new features each cycle
    enable_advanced_patterns: bool = True
    enable_microstructure_innovation: bool = True
    enable_cross_timeframe_features: bool = True

@dataclass 
class FeatureQualityReport:
    """特征质量报告"""
    timestamp: str
    total_features: int
    active_features: int
    new_features: int
    deprecated_features: int
    feature_stability_scores: Dict[str, float]
    feature_importance_scores: Dict[str, float]
    leakage_detected_features: List[str]
    performance_metrics: Dict[str, float]

class ContinuousFeatureOptimizer:
    """
    持续特征工程优化器
    """
    
    def __init__(self, config: FeatureOptimizationConfig):
        self.config = config
        self.logger = self._setup_logger()
        self.feature_registry = {}  # 特征注册表
        self.feature_performance = {}  # 特征性能历史
        self.feature_stability = {}  # 特征稳定性追踪
        self.innovation_cache = {}  # 创新特征缓存
        self.model_cache = {}  # 模型缓存
        self.scaler = StandardScaler()
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(f"{__name__}.ContinuousOptimizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def generate_advanced_momentum_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        生成高级动量特征
        """
        try:
            self.logger.info(f"Generating advanced momentum features for {symbol}")
            
            # 1. 多时间框架动量特征
            momentum_periods = [3, 5, 8, 13, 21, 34]
            for period in momentum_periods:
                # 价格动量
                df[f'{symbol}_momentum_{period}m'] = df['close'].pct_change(period)
                
                # 加速度 (二阶导数)
                df[f'{symbol}_acceleration_{period}m'] = df[f'{symbol}_momentum_{period}m'].diff()
                
                # 动量强度
                momentum_strength = abs(df[f'{symbol}_momentum_{period}m'])
                df[f'{symbol}_momentum_strength_{period}m'] = momentum_strength
                
                # 动量一致性 (方向稳定性)
                momentum_direction = np.sign(df[f'{symbol}_momentum_{period}m'])
                df[f'{symbol}_momentum_consistency_{period}m'] = (
                    momentum_direction.rolling(5).apply(lambda x: (x == x.iloc[-1]).mean())
                )
            
            # 2. 量价动量背离
            price_momentum = df['close'].pct_change(10)
            volume_momentum = df['volume'].pct_change(10)
            df[f'{symbol}_volume_price_divergence'] = price_momentum - volume_momentum
            
            # 3. 相对强度指数变体
            for period in [7, 14, 21]:
                rsi = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
                df[f'{symbol}_rsi_{period}_slope'] = rsi.diff()
                df[f'{symbol}_rsi_{period}_acceleration'] = rsi.diff().diff()
                
                # RSI背离检测
                price_highs = df['close'].rolling(period).max()
                price_lows = df['close'].rolling(period).min()
                rsi_highs = rsi.rolling(period).max()
                rsi_lows = rsi.rolling(period).min()
                
                # 牛背离：价格创新低，RSI不创新低
                df[f'{symbol}_rsi_bull_divergence_{period}'] = (
                    (df['close'] == price_lows) & (rsi > rsi_lows)
                ).astype(int)
                
                # 熊背离：价格创新高，RSI不创新高
                df[f'{symbol}_rsi_bear_divergence_{period}'] = (
                    (df['close'] == price_highs) & (rsi < rsi_highs)
                ).astype(int)
            
            # 4. 波动率调整动量
            returns = df['close'].pct_change()
            for window in [10, 20, 50]:
                vol = returns.rolling(window).std()
                df[f'{symbol}_vol_adj_momentum_{window}'] = returns / (vol + 1e-8)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Advanced momentum features failed for {symbol}: {e}")
            return df
    
    def generate_microstructure_innovation_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        生成创新的微观结构特征
        """
        try:
            self.logger.info(f"Generating microstructure innovation features for {symbol}")
            
            # 1. 增强的接针形态检测
            high_low_range = df['high'] - df['low']
            body_size = abs(df['close'] - df['open'])
            upper_shadow = df['high'] - np.maximum(df['open'], df['close'])
            lower_shadow = np.minimum(df['open'], df['close']) - df['low']
            
            # 超级接针检测 (更严格的条件)
            super_pin_conditions = (
                (lower_shadow / (high_low_range + 1e-8) > 0.6) &  # 下影线占比>60%
                (body_size / (high_low_range + 1e-8) < 0.2) &     # 实体占比<20%
                (upper_shadow / (high_low_range + 1e-8) < 0.2) &  # 上影线占比<20%
                (df['volume'] > df['volume'].rolling(20).mean() * 1.5)  # 成交量放大
            )
            df[f'{symbol}_super_pin_bar'] = super_pin_conditions.astype(int)
            
            # 接针强度评分 (0-1)
            pin_strength = (
                (lower_shadow / (high_low_range + 1e-8)) * 0.4 +
                (1 - body_size / (high_low_range + 1e-8)) * 0.3 +
                (1 - upper_shadow / (high_low_range + 1e-8)) * 0.3
            )
            df[f'{symbol}_pin_strength_score'] = np.clip(pin_strength, 0, 1)
            
            # 2. 订单流不平衡指标
            # 模拟买卖压力
            buy_pressure = np.where(df['close'] > df['open'], df['volume'], 0)
            sell_pressure = np.where(df['close'] < df['open'], df['volume'], 0)
            
            for window in [5, 10, 20]:
                buy_vol = pd.Series(buy_pressure).rolling(window).sum()
                sell_vol = pd.Series(sell_pressure).rolling(window).sum()
                total_vol = buy_vol + sell_vol
                
                df[f'{symbol}_order_flow_imbalance_{window}'] = (
                    (buy_vol - sell_vol) / (total_vol + 1e-8)
                )
                
                # 净买入强度
                df[f'{symbol}_net_buying_intensity_{window}'] = buy_vol / (total_vol + 1e-8)
            
            # 3. 流动性枯竭指标
            price_impact = abs(df['close'].pct_change()) / (
                df['volume'] / df['volume'].rolling(50).mean() + 1e-8
            )
            df[f'{symbol}_price_impact'] = price_impact
            df[f'{symbol}_liquidity_shortage'] = (
                price_impact > price_impact.rolling(100).quantile(0.9)
            ).astype(int)
            
            # 4. 支撑阻力强度
            for lookback in [20, 50]:
                # 支撑位
                support_level = df['low'].rolling(lookback).min()
                support_distance = (df['close'] - support_level) / support_level
                
                # 计算支撑强度 (该价位被测试的次数)
                support_tests = pd.Series(index=df.index, dtype=float)
                for i in range(lookback, len(df)):
                    recent_lows = df['low'].iloc[i-lookback:i]
                    current_support = support_level.iloc[i]
                    # 计算接近支撑位的次数
                    near_support = abs(recent_lows - current_support) / current_support < 0.01
                    support_tests.iloc[i] = near_support.sum()
                
                df[f'{symbol}_support_strength_{lookback}'] = support_tests
                df[f'{symbol}_near_support_{lookback}'] = (support_distance < 0.02).astype(int)
                
                # 阻力位
                resistance_level = df['high'].rolling(lookback).max()
                resistance_distance = (resistance_level - df['close']) / df['close']
                df[f'{symbol}_near_resistance_{lookback}'] = (resistance_distance < 0.02).astype(int)
            
            # 5. 成交量剖面特征
            # 模拟成交量加权平均价格 (VWAP) 偏离
            for period in [20, 50]:
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                vwap = (typical_price * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
                
                vwap_deviation = (df['close'] - vwap) / vwap
                df[f'{symbol}_vwap_deviation_{period}'] = vwap_deviation
                
                # VWAP偏离极值
                df[f'{symbol}_vwap_extreme_deviation_{period}'] = (
                    abs(vwap_deviation) > abs(vwap_deviation).rolling(100).quantile(0.9)
                ).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Microstructure innovation features failed for {symbol}: {e}")
            return df
    
    def generate_market_regime_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        生成市场制度识别特征
        """
        try:
            self.logger.info(f"Generating market regime features for {symbol}")
            
            returns = df['close'].pct_change()
            
            # 1. 波动率制度
            volatility = returns.rolling(20).std()
            vol_percentiles = volatility.rolling(200).quantile([0.25, 0.75])
            
            df[f'{symbol}_low_vol_regime'] = (volatility <= vol_percentiles[0.25]).astype(int)
            df[f'{symbol}_high_vol_regime'] = (volatility >= vol_percentiles[0.75]).astype(int)
            
            # 波动率持续性
            df[f'{symbol}_vol_persistence'] = volatility.rolling(10).std() / volatility
            
            # 2. 趋势制度
            # 多重移动平均线趋势
            ma_periods = [10, 20, 50]
            trend_signals = []
            
            for period in ma_periods:
                ma = df['close'].rolling(period).mean()
                ma_slope = ma.pct_change()
                trend_signals.append(ma_slope > 0)
                df[f'{symbol}_ma_{period}_slope'] = ma_slope
            
            # 趋势一致性 (所有MA同方向的比例)
            df[f'{symbol}_trend_consistency'] = np.mean(trend_signals, axis=0)
            
            # 强趋势识别
            df[f'{symbol}_strong_uptrend'] = (df[f'{symbol}_trend_consistency'] > 0.8).astype(int)
            df[f'{symbol}_strong_downtrend'] = (df[f'{symbol}_trend_consistency'] < 0.2).astype(int)
            df[f'{symbol}_sideways_market'] = (
                (df[f'{symbol}_trend_consistency'] >= 0.4) & 
                (df[f'{symbol}_trend_consistency'] <= 0.6)
            ).astype(int)
            
            # 3. 均值回归 vs 动量制度
            # 半衰期估计
            for window in [50, 100]:
                rolling_returns = returns.rolling(window)
                # 简化的半衰期计算
                autocorr = rolling_returns.apply(lambda x: x.autocorr(lag=1) if len(x) > 10 else 0)
                half_life = -np.log(2) / np.log(abs(autocorr) + 1e-8)
                df[f'{symbol}_half_life_{window}'] = half_life
                
                # 制度分类
                df[f'{symbol}_mean_reversion_regime_{window}'] = (half_life < 10).astype(int)
                df[f'{symbol}_momentum_regime_{window}'] = (half_life > 30).astype(int)
            
            # 4. 流动性制度
            volume_ma = df['volume'].rolling(50).mean()
            volume_std = df['volume'].rolling(50).std()
            
            df[f'{symbol}_high_liquidity_regime'] = (
                df['volume'] > volume_ma + volume_std
            ).astype(int)
            df[f'{symbol}_low_liquidity_regime'] = (
                df['volume'] < volume_ma - volume_std
            ).astype(int)
            
            # 5. 市场压力制度
            # 回撤幅度
            rolling_max = df['close'].rolling(100).max()
            drawdown = (df['close'] - rolling_max) / rolling_max
            df[f'{symbol}_current_drawdown'] = drawdown
            
            # 压力等级
            df[f'{symbol}_market_stress'] = pd.cut(
                -drawdown, 
                bins=[0, 0.05, 0.10, 0.20, 1.0], 
                labels=[0, 1, 2, 3]
            ).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Market regime features failed for {symbol}: {e}")
            return df
    
    def generate_cross_timeframe_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        生成跨时间框架特征
        """
        try:
            self.logger.info(f"Generating cross-timeframe features for {symbol}")
            
            # 模拟不同时间框架的数据
            # 5分钟 -> 15分钟 -> 1小时
            
            # 1. 15分钟级别特征 (3个5分钟K线合并)
            for agg_period in [3, 12]:  # 15分钟和1小时
                # 价格聚合
                high_agg = df['high'].rolling(agg_period).max()
                low_agg = df['low'].rolling(agg_period).min()
                open_agg = df['open'].rolling(agg_period).first()
                close_agg = df['close']
                volume_agg = df['volume'].rolling(agg_period).sum()
                
                # 高时间框架RSI
                rsi_agg = ta.momentum.RSIIndicator(close_agg, window=14).rsi()
                df[f'{symbol}_rsi_htf_{agg_period*5}m'] = rsi_agg
                
                # 高时间框架MACD
                macd_agg = ta.trend.MACD(close_agg)
                df[f'{symbol}_macd_htf_{agg_period*5}m'] = macd_agg.macd()
                df[f'{symbol}_macd_signal_htf_{agg_period*5}m'] = macd_agg.macd_signal()
                
                # 高时间框架布林带
                bb_agg = ta.volatility.BollingerBands(close_agg, window=20)
                bb_pos = bb_agg.bollinger_pband()
                df[f'{symbol}_bb_position_htf_{agg_period*5}m'] = bb_pos
                
                # 时间框架一致性检查
                rsi_5m = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
                df[f'{symbol}_rsi_consistency_{agg_period*5}m'] = (
                    np.sign(rsi_5m - 50) == np.sign(rsi_agg - 50)
                ).astype(int)
            
            # 2. 多时间框架趋势对齐
            # 5分钟趋势
            ma_5m_10 = df['close'].rolling(10).mean()
            ma_5m_20 = df['close'].rolling(20).mean()
            trend_5m = (ma_5m_10 > ma_5m_20).astype(int)
            
            # 15分钟趋势 (基于3周期聚合)
            ma_15m_10 = df['close'].rolling(30).mean()  # 10个15分钟周期
            ma_15m_20 = df['close'].rolling(60).mean()  # 20个15分钟周期
            trend_15m = (ma_15m_10 > ma_15m_20).astype(int)
            
            # 趋势对齐度
            df[f'{symbol}_trend_alignment'] = (trend_5m == trend_15m).astype(int)
            
            # 3. 多时间框架动量分歧
            momentum_5m = df['close'].pct_change(10)  # 50分钟动量
            momentum_15m = df['close'].pct_change(36)  # 3小时动量
            
            df[f'{symbol}_momentum_divergence'] = abs(momentum_5m - momentum_15m)
            df[f'{symbol}_momentum_convergence'] = (
                np.sign(momentum_5m) == np.sign(momentum_15m)
            ).astype(int)
            
            # 4. 跨周期支撑阻力
            # 15分钟支撑阻力对5分钟价格的影响
            support_15m = df['low'].rolling(60).min()  # 15分钟支撑 (5小时)
            resistance_15m = df['high'].rolling(60).max()  # 15分钟阻力
            
            # 价格相对于高级别支撑阻力的位置
            price_position = (df['close'] - support_15m) / (resistance_15m - support_15m + 1e-8)
            df[f'{symbol}_htf_price_position'] = price_position
            
            # 接近高级别关键位
            df[f'{symbol}_near_htf_support'] = (price_position < 0.1).astype(int)
            df[f'{symbol}_near_htf_resistance'] = (price_position > 0.9).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Cross-timeframe features failed for {symbol}: {e}")
            return df
    
    def generate_interaction_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        生成特征交互项
        """
        try:
            self.logger.info(f"Generating interaction features for {symbol}")
            
            # 1. 经典技术指标交互
            if f'{symbol}_rsi_htf_15m' in df.columns and 'volume' in df.columns:
                # RSI与成交量交互
                volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
                df[f'{symbol}_rsi_volume_interaction'] = df[f'{symbol}_rsi_htf_15m'] * volume_ratio
            
            # 2. 波动率与动量交互
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std()
            momentum = returns.rolling(10).sum()
            df[f'{symbol}_vol_momentum_interaction'] = volatility * momentum
            
            # 3. 趋势与均值回归特征交互
            if f'{symbol}_trend_consistency' in df.columns and f'{symbol}_bb_position_htf_15m' in df.columns:
                df[f'{symbol}_trend_bb_interaction'] = (
                    df[f'{symbol}_trend_consistency'] * df[f'{symbol}_bb_position_htf_15m']
                )
            
            # 4. 制度条件特征
            if f'{symbol}_high_vol_regime' in df.columns:
                # 在高波动制度下的RSI行为
                base_rsi = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
                df[f'{symbol}_rsi_high_vol_regime'] = (
                    base_rsi * df[f'{symbol}_high_vol_regime']
                )
                df[f'{symbol}_rsi_normal_vol_regime'] = (
                    base_rsi * (1 - df[f'{symbol}_high_vol_regime'])
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Interaction features failed for {symbol}: {e}")
            return df
    
    def calculate_feature_importance(self, df: pd.DataFrame, target_cols: List[str]) -> Dict[str, float]:
        """
        计算特征重要性
        """
        try:
            feature_importance = {}
            
            # 排除目标变量和非特征列
            feature_cols = [col for col in df.columns 
                          if col not in target_cols 
                          and col not in ['timestamp', 'symbol']
                          and not col.startswith('future_')
                          and not col.startswith('target_')]
            
            if not feature_cols or not target_cols:
                return feature_importance
            
            X = df[feature_cols].fillna(0)
            
            for target_col in target_cols:
                if target_col not in df.columns:
                    continue
                    
                y = df[target_col].fillna(0)
                
                # 过滤有效数据
                valid_mask = ~(X.isnull().all(axis=1) | y.isnull())
                X_valid = X[valid_mask]
                y_valid = y[valid_mask]
                
                if len(X_valid) < 100:
                    continue
                
                try:
                    # 使用互信息计算特征重要性
                    if y_valid.dtype in ['int64', 'bool'] and len(y_valid.unique()) <= 10:
                        # 分类问题
                        mi_scores = mutual_info_classif(X_valid, y_valid, random_state=42)
                    else:
                        # 回归问题
                        mi_scores = mutual_info_regression(X_valid, y_valid, random_state=42)
                    
                    # 存储结果
                    for i, feature in enumerate(feature_cols):
                        if feature not in feature_importance:
                            feature_importance[feature] = 0
                        feature_importance[feature] += mi_scores[i]
                        
                except Exception as e:
                    self.logger.warning(f"Feature importance calculation failed for {target_col}: {e}")
                    continue
            
            # 标准化重要性分数
            if feature_importance:
                max_importance = max(feature_importance.values())
                if max_importance > 0:
                    feature_importance = {
                        k: v / max_importance for k, v in feature_importance.items()
                    }
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"Feature importance calculation failed: {e}")
            return {}
    
    def detect_feature_stability(self, df: pd.DataFrame, feature_name: str, window_days: int = 30) -> float:
        """
        检测特征稳定性 (PSI - Population Stability Index)
        """
        try:
            if feature_name not in df.columns or 'timestamp' not in df.columns:
                return 0.0
            
            df_copy = df.copy()
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
            df_copy = df_copy.sort_values('timestamp')
            
            feature_data = df_copy[feature_name].dropna()
            if len(feature_data) < 200:
                return 0.0
            
            # 分割数据为基准期和测试期
            split_point = len(feature_data) // 2
            baseline = feature_data.iloc[:split_point]
            current = feature_data.iloc[split_point:]
            
            # 创建分位数区间
            bins = np.percentile(baseline, [0, 10, 25, 50, 75, 90, 100])
            bins = np.unique(bins)  # 去除重复值
            
            if len(bins) < 3:
                return 0.0
            
            # 计算各区间的分布
            baseline_dist, _ = np.histogram(baseline, bins=bins)
            current_dist, _ = np.histogram(current, bins=bins)
            
            # 转换为概率分布
            baseline_dist = baseline_dist / baseline_dist.sum()
            current_dist = current_dist / current_dist.sum()
            
            # 计算PSI
            psi = 0
            for i in range(len(baseline_dist)):
                if baseline_dist[i] > 0 and current_dist[i] > 0:
                    psi += (current_dist[i] - baseline_dist[i]) * np.log(
                        current_dist[i] / baseline_dist[i]
                    )
            
            # 返回稳定性分数 (PSI越小越稳定)
            stability_score = max(0, 1 - psi / 0.25)  # 0.25是常用的PSI阈值
            return stability_score
            
        except Exception as e:
            self.logger.error(f"Feature stability detection failed for {feature_name}: {e}")
            return 0.0
    
    def detect_data_leakage(self, df: pd.DataFrame, target_cols: List[str]) -> List[str]:
        """
        检测数据泄漏
        """
        leakage_features = []
        
        try:
            feature_cols = [col for col in df.columns 
                          if col not in target_cols 
                          and col not in ['timestamp', 'symbol']
                          and not col.startswith('future_')
                          and not col.startswith('target_')]
            
            for target_col in target_cols:
                if target_col not in df.columns:
                    continue
                    
                target_data = df[target_col].dropna()
                if len(target_data) < 100:
                    continue
                
                for feature_col in feature_cols:
                    if feature_col not in df.columns:
                        continue
                    
                    feature_data = df[feature_col].dropna()
                    if len(feature_data) < 100:
                        continue
                    
                    # 对齐数据
                    aligned_data = pd.concat([feature_data, target_data], axis=1, join='inner')
                    if len(aligned_data) < 50:
                        continue
                    
                    # 计算相关性
                    correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                    
                    # 检测异常高的相关性 (可能的泄漏)
                    if abs(correlation) > 0.9:
                        leakage_features.append(feature_col)
                        self.logger.warning(
                            f"Potential data leakage detected: {feature_col} -> {target_col} "
                            f"(correlation: {correlation:.3f})"
                        )
            
        except Exception as e:
            self.logger.error(f"Data leakage detection failed: {e}")
        
        return list(set(leakage_features))
    
    def optimize_features_for_symbol(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict]:
        """
        为单个币种优化特征
        """
        try:
            self.logger.info(f"Optimizing features for {symbol}...")
            
            # 复制数据
            optimized_df = df.copy()
            
            # 1. 生成高级动量特征
            optimized_df = self.generate_advanced_momentum_features(optimized_df, symbol)
            
            # 2. 生成微观结构创新特征
            if self.config.enable_microstructure_innovation:
                optimized_df = self.generate_microstructure_innovation_features(optimized_df, symbol)
            
            # 3. 生成市场制度特征
            optimized_df = self.generate_market_regime_features(optimized_df, symbol)
            
            # 4. 生成跨时间框架特征
            if self.config.enable_cross_timeframe_features:
                optimized_df = self.generate_cross_timeframe_features(optimized_df, symbol)
            
            # 5. 生成交互特征
            optimized_df = self.generate_interaction_features(optimized_df, symbol)
            
            # 6. 特征质量评估
            target_cols = [col for col in optimized_df.columns if col.startswith('target_') or 'future_return' in col]
            
            # 计算特征重要性
            feature_importance = self.calculate_feature_importance(optimized_df, target_cols)
            
            # 检测数据泄漏
            leakage_features = self.detect_data_leakage(optimized_df, target_cols)
            
            # 移除泄漏特征
            if leakage_features:
                optimized_df = optimized_df.drop(columns=leakage_features)
                self.logger.info(f"Removed {len(leakage_features)} features with data leakage")
            
            # 7. 特征选择 (保留重要特征)
            if feature_importance:
                important_features = [
                    feature for feature, importance in feature_importance.items() 
                    if importance >= self.config.min_feature_importance
                ]
                
                # 保留基础列和重要特征
                base_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                keep_cols = base_cols + important_features + target_cols
                keep_cols = [col for col in keep_cols if col in optimized_df.columns]
                
                optimized_df = optimized_df[keep_cols]
                self.logger.info(f"Selected {len(important_features)} important features for {symbol}")
            
            # 8. 生成优化报告
            optimization_report = {
                'symbol': symbol,
                'original_features': len(df.columns),
                'optimized_features': len(optimized_df.columns),
                'feature_importance': feature_importance,
                'leakage_features_removed': leakage_features,
                'optimization_timestamp': datetime.now().isoformat()
            }
            
            return optimized_df, optimization_report
            
        except Exception as e:
            self.logger.error(f"Feature optimization failed for {symbol}: {e}")
            return df, {'symbol': symbol, 'error': str(e)}
    
    def run_continuous_optimization(self, data_dict: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], FeatureQualityReport]:
        """
        运行持续特征优化
        """
        try:
            self.logger.info(f"Starting continuous feature optimization for {len(data_dict)} symbols...")
            start_time = time.time()
            
            optimized_data = {}
            optimization_reports = {}
            
            # 并行处理各币种
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self.optimize_features_for_symbol, df, symbol): symbol
                    for symbol, df in data_dict.items()
                }
                
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        optimized_df, report = future.result()
                        optimized_data[symbol] = optimized_df
                        optimization_reports[symbol] = report
                        self.logger.info(f"Optimization completed for {symbol}")
                    except Exception as e:
                        self.logger.error(f"Optimization failed for {symbol}: {e}")
                        optimized_data[symbol] = data_dict[symbol]  # 使用原始数据
                        optimization_reports[symbol] = {'symbol': symbol, 'error': str(e)}
            
            # 生成质量报告
            total_features = sum(len(df.columns) for df in optimized_data.values())
            avg_features = total_features / len(optimized_data) if optimized_data else 0
            
            # 统计新特征
            new_features = []
            for symbol, report in optimization_reports.items():
                if 'feature_importance' in report:
                    new_features.extend(list(report['feature_importance'].keys()))
            new_features = list(set(new_features))
            
            # 统计泄漏特征
            leakage_features = []
            for symbol, report in optimization_reports.items():
                if 'leakage_features_removed' in report:
                    leakage_features.extend(report['leakage_features_removed'])
            leakage_features = list(set(leakage_features))
            
            quality_report = FeatureQualityReport(
                timestamp=datetime.now().isoformat(),
                total_features=int(total_features),
                active_features=int(avg_features),
                new_features=len(new_features),
                deprecated_features=len(leakage_features),
                feature_stability_scores={},
                feature_importance_scores={},
                leakage_detected_features=leakage_features,
                performance_metrics={
                    'optimization_time_seconds': time.time() - start_time,
                    'symbols_processed': len(optimized_data),
                    'avg_features_per_symbol': avg_features
                }
            )
            
            self.logger.info(f"Continuous optimization completed in {quality_report.performance_metrics['optimization_time_seconds']:.1f}s")
            self.logger.info(f"Total features: {quality_report.total_features}, New features: {quality_report.new_features}")
            
            return optimized_data, quality_report
            
        except Exception as e:
            self.logger.error(f"Continuous optimization failed: {e}")
            return data_dict, FeatureQualityReport(
                timestamp=datetime.now().isoformat(),
                total_features=0,
                active_features=0,
                new_features=0,
                deprecated_features=0,
                feature_stability_scores={},
                feature_importance_scores={},
                leakage_detected_features=[],
                performance_metrics={'error': str(e)}
            )

def main():
    """演示持续特征优化系统"""
    
    print("DipMaster持续特征工程优化系统")
    print("=" * 60)
    print("🎯 目标: 持续发现和优化有效特征组合")
    print("\n🔧 核心功能:")
    print("1. 高级动量特征挖掘")
    print("   - 多时间框架动量分析")
    print("   - 动量背离检测") 
    print("   - 波动率调整动量")
    print("\n2. 微观结构创新特征")
    print("   - 超级接针形态检测")
    print("   - 订单流不平衡分析")
    print("   - 流动性枯竭指标")
    print("   - 支撑阻力强度量化")
    print("\n3. 市场制度识别")
    print("   - 波动率制度分类")
    print("   - 趋势vs均值回归检测")
    print("   - 流动性制度监控")
    print("   - 市场压力等级评估")
    print("\n4. 跨时间框架特征")
    print("   - 多周期信号一致性")
    print("   - 趋势对齐分析")
    print("   - 动量分歧检测")
    print("\n5. 持续质量监控")
    print("   - 特征重要性追踪")
    print("   - 数据泄漏检测")
    print("   - 特征稳定性监控")
    print("   - 自动特征选择")
    print("\n✅ 预期效果:")
    print("- 发现更多预测性特征")
    print("- 提升信号质量和稳定性") 
    print("- 减少过拟合风险")
    print("- 增强策略适应性")
    print("=" * 60)

if __name__ == "__main__":
    main()