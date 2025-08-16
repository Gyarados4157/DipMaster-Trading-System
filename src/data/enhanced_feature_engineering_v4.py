#!/usr/bin/env python3
"""
Enhanced DipMaster V4 Feature Engineering Pipeline
增强版特征工程管道 - 专门为DipMaster策略优化，目标85%+胜率

This module provides advanced feature engineering for the DipMaster trading strategy,
incorporating multi-timeframe analysis, cross-symbol signals, market microstructure,
and machine learning features to achieve 85%+ win rate.

Author: DipMaster Quant Team
Date: 2025-08-16
Version: 4.0.0-Enhanced
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Suppress warnings
warnings.filterwarnings('ignore')

class EnhancedDipMasterFeatureEngineer:
    """
    Enhanced DipMaster Feature Engineering Pipeline
    增强版DipMaster特征工程管道
    
    Key Features:
    1. Multi-timeframe RSI convergence analysis
    2. Dynamic RSI threshold optimization
    3. Cross-symbol relative strength and rotation signals
    4. Market microstructure features
    5. Advanced ML features (PCA, factor decomposition)
    6. Multi-layer signal scoring system
    7. Enhanced label engineering with risk adjustment
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the enhanced feature engineer"""
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.symbols = self.config.get('symbols', [])
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.feature_names = []
        self.cross_symbol_features = {}
        
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            'symbols': [
                'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT',
                'BNBUSDT', 'DOGEUSDT', 'SUIUSDT', 'ICPUSDT', 'ALGOUSDT', 'IOTAUSDT',
                'TRXUSDT', 'LINKUSDT', 'AVAXUSDT', 'MATICUSDT', 'UNIUSDT',
                'LTCUSDT', 'DOTUSDT', 'ATOMUSDT', 'FILUSDT', 'NEARUSDT',
                'ARBUSDT', 'OPUSDT', 'APTUSDT', 'QNTUSDT'
            ],
            'timeframes': ['5m', '15m', '1h'],
            'rsi_periods': [14, 21, 50],
            'volatility_periods': [10, 20, 50],
            'ma_periods': [5, 10, 20, 50, 100],
            'prediction_horizons': [3, 6, 12, 24, 48],  # In 5-minute intervals
            'profit_targets': [0.006, 0.012, 0.020],
            'stop_loss': 0.004,
            'max_holding_periods': 36,  # 180 minutes = 36 * 5min
        }
    
    def add_enhanced_dipmaster_core(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add enhanced DipMaster core signals with multi-timeframe analysis
        添加增强版DipMaster核心信号，包含多时间框架分析
        """
        try:
            # Multi-timeframe RSI convergence
            for period in self.config['rsi_periods']:
                rsi_col = f'rsi_{period}'
                df[rsi_col] = self._calculate_rsi(df['close'], period)
                
                # Dynamic RSI thresholds based on volatility
                vol_20 = df['close'].pct_change().rolling(20).std()
                vol_percentile = vol_20.rolling(100).rank(pct=True)
                
                # Adjust RSI thresholds: higher volatility = wider thresholds
                rsi_lower = 25 + (vol_percentile * 10)  # 25-35 range
                rsi_upper = 40 + (vol_percentile * 15)  # 40-55 range
                
                df[f'rsi_{period}_dip_zone'] = (
                    (df[rsi_col] >= rsi_lower) & (df[rsi_col] <= rsi_upper)
                ).astype(int)
                
                # RSI gradient analysis
                df[f'rsi_{period}_gradient'] = df[rsi_col].diff()
                df[f'rsi_{period}_acceleration'] = df[f'rsi_{period}_gradient'].diff()
                
            # RSI convergence signal - when multiple timeframes align
            rsi_signals = [f'rsi_{p}_dip_zone' for p in self.config['rsi_periods']]
            df['rsi_convergence_score'] = df[rsi_signals].sum(axis=1) / len(rsi_signals)
            df['rsi_convergence_strong'] = (df['rsi_convergence_score'] >= 0.67).astype(int)
            
            # Enhanced dip detection with consecutive analysis
            df['price_dip_1'] = (df['close'] < df['open']).astype(int)
            df['price_dip_2'] = (df['close'].shift(1) < df['open'].shift(1)).astype(int)
            df['price_dip_3'] = (df['close'].shift(2) < df['open'].shift(2)).astype(int)
            
            # Consecutive red candles with diminishing impact
            df['consecutive_dips'] = df['price_dip_1'] + (df['price_dip_2'] * 0.5) + (df['price_dip_3'] * 0.25)
            
            # Price drop magnitude analysis
            df['drop_magnitude_1p'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
            df['drop_magnitude_3p'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
            df['drop_magnitude_5p'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
            
            # Enhanced volume confirmation
            for ma_period in [10, 20, 50]:
                vol_ma = df['volume'].rolling(ma_period).mean()
                df[f'volume_ratio_{ma_period}'] = df['volume'] / vol_ma
                df[f'volume_spike_{ma_period}'] = (df[f'volume_ratio_{ma_period}'] > 1.5).astype(int)
            
            # Volume-weighted price impact
            df['vwap_5'] = (df['close'] * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
            df['vwap_deviation'] = (df['close'] - df['vwap_5']) / df['vwap_5']
            
            # Bollinger Bands squeeze analysis
            bb_data = self._calculate_bollinger_bands(df['close'], period=20, std_dev=2.0)
            df['bb_upper'] = bb_data['upper']
            df['bb_middle'] = bb_data['middle']
            df['bb_lower'] = bb_data['lower']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(int)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # MA cloud analysis and support/resistance
            for ma_period in self.config['ma_periods']:
                df[f'ma_{ma_period}'] = df['close'].rolling(ma_period).mean()
                df[f'ma_{ma_period}_distance'] = (df['close'] - df[f'ma_{ma_period}']) / df[f'ma_{ma_period}']
                df[f'below_ma_{ma_period}'] = (df['close'] < df[f'ma_{ma_period}']).astype(int)
            
            # MA trend strength
            ma_short = df['ma_10']
            ma_long = df['ma_50']
            df['ma_trend_strength'] = (ma_short - ma_long) / ma_long
            df['ma_alignment'] = (ma_short > ma_long).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Enhanced DipMaster core features failed for {symbol}: {e}")
            return df
    
    def add_cross_symbol_features(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Add cross-symbol relative strength and rotation features
        添加跨币种相对强度和轮动特征
        """
        try:
            # Calculate market-wide metrics
            market_prices = {}
            market_volumes = {}
            market_volatilities = {}
            
            # Collect price and volume data for all symbols
            for symbol, df in all_data.items():
                if len(df) > 0:
                    market_prices[symbol] = df.set_index('timestamp')['close']
                    market_volumes[symbol] = df.set_index('timestamp')['volume']
                    returns = df['close'].pct_change()
                    market_volatilities[symbol] = returns.rolling(20).std()
            
            # Create market dataframes
            price_matrix = pd.DataFrame(market_prices).ffill()
            volume_matrix = pd.DataFrame(market_volumes).ffill()
            
            # Calculate relative strength features
            for symbol in self.symbols:
                if symbol in all_data and len(all_data[symbol]) > 0:
                    df = all_data[symbol].copy()
                    
                    # Market relative performance
                    market_returns = price_matrix.pct_change().mean(axis=1)
                    symbol_returns = price_matrix[symbol].pct_change() if symbol in price_matrix.columns else pd.Series()
                    
                    if len(symbol_returns) > 0:
                        relative_performance = symbol_returns - market_returns
                        
                        # Align with original dataframe
                        df = df.set_index('timestamp')
                        df['relative_strength_1d'] = relative_performance.rolling(24*12).mean()  # 1 day
                        df['relative_strength_3d'] = relative_performance.rolling(24*12*3).mean()  # 3 days
                        df['relative_strength_7d'] = relative_performance.rolling(24*12*7).mean()  # 7 days
                        
                        # Relative strength rank
                        daily_returns = price_matrix.pct_change(24*12)  # Daily returns
                        df['rs_rank'] = daily_returns.rank(axis=1, pct=True)[symbol] if symbol in daily_returns.columns else 0.5
                        
                        # Volume relative to market
                        if symbol in volume_matrix.columns:
                            market_volume = volume_matrix.mean(axis=1)
                            symbol_volume = volume_matrix[symbol]
                            df['volume_vs_market'] = symbol_volume / market_volume
                        
                        # Correlation with market leaders (BTC, ETH)
                        if 'BTCUSDT' in price_matrix.columns:
                            btc_returns = price_matrix['BTCUSDT'].pct_change()
                            symbol_rets = price_matrix[symbol].pct_change() if symbol in price_matrix.columns else pd.Series()
                            if len(symbol_rets) > 0:
                                correlation_window = 24*12*7  # 7 days
                                df['btc_correlation'] = symbol_rets.rolling(correlation_window).corr(btc_returns)
                        
                        # Momentum divergence
                        market_momentum = market_returns.rolling(12).sum()  # 1 hour momentum
                        symbol_momentum = symbol_returns.rolling(12).sum() if len(symbol_returns) > 0 else pd.Series()
                        if len(symbol_momentum) > 0:
                            df['momentum_divergence'] = symbol_momentum - market_momentum
                        
                        # Reset index
                        df = df.reset_index()
                        all_data[symbol] = df
            
            return all_data
            
        except Exception as e:
            self.logger.error(f"Cross-symbol features calculation failed: {e}")
            return all_data
    
    def add_microstructure_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add market microstructure features
        添加市场微观结构特征
        """
        try:
            # Price impact estimation
            df['price_impact'] = abs(df['close'].pct_change()) / (df['volume'] / df['volume'].rolling(20).mean())
            df['price_impact_smoothed'] = df['price_impact'].rolling(5).mean()
            
            # Intraday volatility patterns
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['timestamp'].dt.hour
                df['minute'] = df['timestamp'].dt.minute
                
                # Calculate hourly volatility patterns
                hourly_vol = df.groupby('hour')['close'].pct_change().std()
                df['hourly_vol_pattern'] = df['hour'].map(hourly_vol)
                
                # Time-based volume patterns
                hourly_volume = df.groupby('hour')['volume'].mean()
                df['hourly_volume_pattern'] = df['hour'].map(hourly_volume)
            
            # Liquidity proxies
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            df['effective_spread'] = df['high_low_ratio'] * 2  # Simplified spread proxy
            
            # Order flow imbalance (simplified)
            df['order_flow_imbalance'] = np.where(
                df['close'] > (df['high'] + df['low']) / 2,
                df['volume'],  # Buying pressure
                -df['volume']  # Selling pressure
            )
            df['ofi_cumulative'] = df['order_flow_imbalance'].rolling(20).sum()
            
            # VWAP deviations at different intervals
            for period in [5, 10, 20]:
                vwap = (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
                df[f'vwap_dev_{period}'] = (df['close'] - vwap) / vwap
            
            # Tick-by-tick momentum (using OHLC)
            df['intra_candle_momentum'] = (df['close'] - df['open']) / (df['high'] - df['low'])
            df['candle_body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
            
            # Volatility clustering
            returns = df['close'].pct_change()
            df['volatility_5'] = returns.rolling(5).std()
            df['volatility_20'] = returns.rolling(20).std()
            df['vol_clustering'] = df['volatility_5'] / df['volatility_20']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Microstructure features failed for {symbol}: {e}")
            return df
    
    def add_machine_learning_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add advanced machine learning features
        添加高级机器学习特征
        """
        try:
            # Feature interaction terms
            if 'rsi_14' in df.columns and 'volume_ratio_20' in df.columns:
                df['rsi_volume_interaction'] = df['rsi_14'] * df['volume_ratio_20']
            
            if 'bb_position' in df.columns and 'ma_trend_strength' in df.columns:
                df['bb_trend_interaction'] = df['bb_position'] * df['ma_trend_strength']
            
            # Rolling statistical features
            price_returns = df['close'].pct_change()
            
            # Rolling statistics
            for window in [10, 20, 50]:
                df[f'return_skew_{window}'] = price_returns.rolling(window).skew()
                df[f'return_kurtosis_{window}'] = price_returns.rolling(window).apply(lambda x: stats.kurtosis(x))
                df[f'return_std_{window}'] = price_returns.rolling(window).std()
            
            # Regime detection using volatility clustering
            vol_regime = df['volatility_20'].rolling(50).quantile(0.8)
            df['high_vol_regime'] = (df['volatility_20'] > vol_regime).astype(int)
            
            # Trend strength using multiple MAs
            ma_cols = [col for col in df.columns if col.startswith('ma_') and col.split('_')[1].isdigit()]
            if len(ma_cols) >= 3:
                # Calculate trend consistency
                trend_signals = []
                for i in range(len(ma_cols)-1):
                    for j in range(i+1, len(ma_cols)):
                        ma1, ma2 = ma_cols[i], ma_cols[j]
                        trend_signals.append((df[ma1] > df[ma2]).astype(int))
                
                if trend_signals:
                    df['trend_consistency'] = np.mean(trend_signals, axis=0)
            
            # Fibonacci retracement levels (simplified)
            rolling_high = df['high'].rolling(50).max()
            rolling_low = df['low'].rolling(50).min()
            fib_range = rolling_high - rolling_low
            
            df['fib_382'] = rolling_low + (fib_range * 0.382)
            df['fib_618'] = rolling_low + (fib_range * 0.618)
            df['near_fib_382'] = (abs(df['close'] - df['fib_382']) / df['close'] < 0.01).astype(int)
            df['near_fib_618'] = (abs(df['close'] - df['fib_618']) / df['close'] < 0.01).astype(int)
            
            # Support and resistance levels
            df['resistance_20'] = df['high'].rolling(20).max()
            df['support_20'] = df['low'].rolling(20).min()
            df['near_resistance'] = (df['close'] / df['resistance_20'] > 0.98).astype(int)
            df['near_support'] = (df['close'] / df['support_20'] < 1.02).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"ML features failed for {symbol}: {e}")
            return df
    
    def add_advanced_strategy_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add advanced multi-layer strategy signals
        添加高级多层级策略信号
        """
        try:
            # Layer 1: Primary DipMaster signals
            primary_signals = []
            weights = []
            
            if 'rsi_convergence_strong' in df.columns:
                primary_signals.append(df['rsi_convergence_strong'])
                weights.append(0.25)
            
            if 'consecutive_dips' in df.columns:
                # Normalize consecutive dips to 0-1 scale
                normalized_dips = np.clip(df['consecutive_dips'] / 2.0, 0, 1)
                primary_signals.append(normalized_dips)
                weights.append(0.20)
            
            if 'volume_spike_20' in df.columns:
                primary_signals.append(df['volume_spike_20'])
                weights.append(0.15)
            
            if 'bb_squeeze' in df.columns:
                primary_signals.append(df['bb_squeeze'])
                weights.append(0.10)
            
            if 'below_ma_20' in df.columns:
                primary_signals.append(df['below_ma_20'])
                weights.append(0.15)
            
            if 'vwap_deviation' in df.columns:
                # Convert negative VWAP deviation (below VWAP) to positive signal
                vwap_signal = np.clip(-df['vwap_deviation'] * 10, 0, 1)
                primary_signals.append(vwap_signal)
                weights.append(0.15)
            
            # Calculate primary signal strength
            if primary_signals and weights:
                # Normalize weights
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                df['primary_signal_strength'] = sum(
                    signal * weight for signal, weight in zip(primary_signals, weights)
                )
            else:
                df['primary_signal_strength'] = 0
            
            # Layer 2: Market regime filter
            regime_signals = []
            regime_weights = []
            
            if 'high_vol_regime' in df.columns:
                # Prefer normal volatility regime for DipMaster
                normal_vol_signal = 1 - df['high_vol_regime']
                regime_signals.append(normal_vol_signal)
                regime_weights.append(0.4)
            
            if 'trend_consistency' in df.columns:
                # Prefer weak trends for mean reversion
                weak_trend_signal = 1 - df['trend_consistency']
                regime_signals.append(weak_trend_signal)
                regime_weights.append(0.3)
            
            if 'relative_strength_1d' in df.columns:
                # Prefer relatively weak performance (for mean reversion)
                rel_weak_signal = np.clip(-df['relative_strength_1d'] * 50, 0, 1)
                regime_signals.append(rel_weak_signal)
                regime_weights.append(0.3)
            
            if regime_signals and regime_weights:
                regime_weights = np.array(regime_weights)
                regime_weights = regime_weights / regime_weights.sum()
                
                df['regime_filter_strength'] = sum(
                    signal * weight for signal, weight in zip(regime_signals, regime_weights)
                )
            else:
                df['regime_filter_strength'] = 0.5  # Neutral
            
            # Layer 3: Timing and execution signals
            timing_signals = []
            
            if 'near_support' in df.columns:
                timing_signals.append(df['near_support'] * 0.3)
            
            if 'near_fib_618' in df.columns:
                timing_signals.append(df['near_fib_618'] * 0.2)
            
            if 'ofi_cumulative' in df.columns:
                # Normalize order flow imbalance
                ofi_normalized = np.clip(-df['ofi_cumulative'] / df['ofi_cumulative'].rolling(100).std(), 0, 1)
                timing_signals.append(ofi_normalized * 0.3)
            
            if 'hourly_vol_pattern' in df.columns:
                # Prefer higher volatility hours for better opportunities
                vol_signal = np.clip(df['hourly_vol_pattern'] / df['hourly_vol_pattern'].rolling(100).mean(), 0, 2) / 2
                timing_signals.append(vol_signal * 0.2)
            
            df['timing_signal_strength'] = sum(timing_signals) if timing_signals else 0
            
            # Combined signal score
            df['dipmaster_v4_signal_score'] = (
                df['primary_signal_strength'] * 0.6 +
                df['regime_filter_strength'] * 0.25 +
                df['timing_signal_strength'] * 0.15
            )
            
            # Signal confidence based on multiple confirmations
            confirmations = 0
            if 'rsi_convergence_strong' in df.columns:
                confirmations += df['rsi_convergence_strong']
            if 'volume_spike_20' in df.columns:
                confirmations += df['volume_spike_20']
            if 'bb_squeeze' in df.columns:
                confirmations += df['bb_squeeze']
            
            df['signal_confidence'] = np.clip(confirmations / 3.0, 0, 1)
            
            # Final signal with confidence adjustment
            df['dipmaster_v4_final_signal'] = df['dipmaster_v4_signal_score'] * df['signal_confidence']
            
            # Signal quality flags
            df['high_quality_signal'] = (df['dipmaster_v4_final_signal'] > 0.7).astype(int)
            df['medium_quality_signal'] = (
                (df['dipmaster_v4_final_signal'] > 0.5) & (df['dipmaster_v4_final_signal'] <= 0.7)
            ).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Advanced strategy signals failed for {symbol}: {e}")
            return df
    
    def add_enhanced_labels(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add enhanced multi-horizon and risk-adjusted labels
        添加增强版多时间窗口和风险调整标签
        """
        try:
            # Multi-horizon future returns
            for horizon in self.config['prediction_horizons']:
                # Future return
                future_return = df['close'].pct_change(periods=horizon).shift(-horizon)
                df[f'future_return_{horizon}p'] = future_return
                
                # Binary profitable label
                df[f'is_profitable_{horizon}p'] = (future_return > 0).astype(int)
                
                # Risk-adjusted return (Sharpe-like)
                return_std = df['volatility_20'].shift(-horizon)
                risk_adj_return = future_return / (return_std + 1e-8)
                df[f'risk_adj_return_{horizon}p'] = risk_adj_return
                
                # Hit rate for different profit targets
                for target in self.config['profit_targets']:
                    df[f'hits_target_{target:.1%}_{horizon}p'] = (future_return >= target).astype(int)
                
                # Stop loss hit
                df[f'hits_stop_loss_{horizon}p'] = (future_return <= -self.config['stop_loss']).astype(int)
                
                # Maximum favorable excursion (MFE) and maximum adverse excursion (MAE)
                future_highs = df['high'].rolling(window=horizon, min_periods=1).max().shift(-horizon)
                future_lows = df['low'].rolling(window=horizon, min_periods=1).min().shift(-horizon)
                
                mfe = (future_highs - df['close']) / df['close']
                mae = (future_lows - df['close']) / df['close']
                
                df[f'mfe_{horizon}p'] = mfe
                df[f'mae_{horizon}p'] = mae
                
                # Optimal holding time (when to exit for best return)
                exit_returns = []
                for exit_period in range(1, min(horizon + 1, 37)):  # Up to 36 periods (3 hours)
                    exit_price = df['close'].shift(-exit_period)
                    exit_return = (exit_price - df['close']) / df['close']
                    exit_returns.append(exit_return)
                
                if exit_returns:
                    exit_returns_df = pd.DataFrame(exit_returns).T
                    optimal_exit_period = exit_returns_df.idxmax(axis=1) + 1
                    optimal_return = exit_returns_df.max(axis=1)
                    
                    df[f'optimal_exit_period_{horizon}p'] = optimal_exit_period
                    df[f'optimal_return_{horizon}p'] = optimal_return
            
            # Survival analysis labels (time to profit/loss)
            current_price = df['close']
            
            # Time to first profit (target achievement)
            time_to_profit = self._calculate_time_to_event(
                df, 'close', current_price, self.config['profit_targets'][0], direction='up'
            )
            df['time_to_profit'] = time_to_profit
            
            # Time to stop loss
            time_to_stop = self._calculate_time_to_event(
                df, 'close', current_price, -self.config['stop_loss'], direction='down'
            )
            df['time_to_stop_loss'] = time_to_stop
            
            # Probability of profit within different time windows
            for window in [6, 12, 24, 36]:  # 30min, 1h, 2h, 3h
                profit_within_window = (
                    (df[f'future_return_{window}p'] > self.config['profit_targets'][0]) & 
                    (df[f'mae_{window}p'] > -self.config['stop_loss'])
                ).astype(int)
                df[f'profit_within_{window}p'] = profit_within_window
            
            # Multi-target classification
            # 0: Loss, 1: Small profit, 2: Medium profit, 3: Large profit
            main_return = df['future_return_12p']  # 1-hour return
            df['return_class'] = pd.cut(
                main_return,
                bins=[-np.inf, -0.002, 0.006, 0.015, np.inf],
                labels=[0, 1, 2, 3]
            ).astype(float)
            
            # Primary labels for DipMaster strategy
            df['target_return'] = df['future_return_12p']  # 1-hour return
            df['target_binary'] = df['is_profitable_12p']   # 1-hour profitability
            df['target_risk_adjusted'] = df['risk_adj_return_12p']  # Risk-adjusted
            
            return df
            
        except Exception as e:
            self.logger.error(f"Enhanced labels failed for {symbol}: {e}")
            return df
    
    def _calculate_time_to_event(self, df: pd.DataFrame, price_col: str, 
                                current_price: pd.Series, threshold: float, 
                                direction: str = 'up') -> pd.Series:
        """Calculate time to reach a price threshold"""
        try:
            time_to_event = pd.Series(index=df.index, dtype=float)
            
            for i in range(len(df)):
                if pd.isna(current_price.iloc[i]):
                    continue
                    
                target_price = current_price.iloc[i] * (1 + threshold)
                
                # Look forward to find when target is hit
                for j in range(i + 1, min(i + 37, len(df))):  # Max 36 periods forward
                    if direction == 'up' and df[price_col].iloc[j] >= target_price:
                        time_to_event.iloc[i] = j - i
                        break
                    elif direction == 'down' and df[price_col].iloc[j] <= target_price:
                        time_to_event.iloc[i] = j - i
                        break
                
                # If target not reached within window, set to max window
                if pd.isna(time_to_event.iloc[i]):
                    time_to_event.iloc[i] = 36
            
            return time_to_event
            
        except Exception as e:
            self.logger.error(f"Time to event calculation failed: {e}")
            return pd.Series(index=df.index, dtype=float)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
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
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }
    
    def apply_feature_selection_and_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature selection and advanced engineering"""
        try:
            # Get numeric features for PCA
            numeric_features = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_features if not any(
                x in col for x in ['future_return', 'target', 'is_profitable', 'hits_', 'mfe_', 'mae_', 'optimal_', 'time_to_', 'return_class']
            )]
            
            if len(feature_cols) > 10:
                # Apply PCA for dimensionality reduction
                feature_data = df[feature_cols].fillna(0)
                
                if len(feature_data) > 100:  # Ensure enough samples
                    try:
                        # Standardize features
                        scaled_features = self.scaler.fit_transform(feature_data)
                        
                        # Apply PCA
                        pca_features = self.pca.fit_transform(scaled_features)
                        
                        # Add PCA components as features
                        n_components = pca_features.shape[1]
                        for i in range(min(n_components, 20)):  # Max 20 PCA components
                            df[f'pca_component_{i+1}'] = pca_features[:, i]
                        
                        # Feature importance from PCA
                        feature_importance = pd.DataFrame({
                            'feature': feature_cols,
                            'importance': np.sum(np.abs(self.pca.components_[:5]), axis=0)  # Top 5 components
                        })
                        feature_importance = feature_importance.sort_values('importance', ascending=False)
                        
                        # Add top features interaction
                        top_features = feature_importance.head(5)['feature'].tolist()
                        if len(top_features) >= 2:
                            df['top_feature_interaction'] = 1
                            for i in range(min(3, len(top_features))):
                                for j in range(i+1, min(3, len(top_features))):
                                    feat1, feat2 = top_features[i], top_features[j]
                                    if feat1 in df.columns and feat2 in df.columns:
                                        df[f'interact_{i}_{j}'] = df[feat1] * df[feat2]
                        
                    except Exception as e:
                        self.logger.warning(f"PCA feature engineering failed: {e}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Feature selection and engineering failed: {e}")
            return df
    
    def clean_and_validate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate feature data"""
        try:
            # Handle infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values intelligently
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col.startswith('target') or 'future_return' in col:
                    # Don't forward fill target variables
                    continue
                
                # Forward fill then backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Remove extreme outliers (beyond 5 standard deviations)
            for col in numeric_cols:
                if col not in ['timestamp', 'hour', 'minute', 'day_of_week'] and not col.startswith('target'):
                    Q1 = df[col].quantile(0.01)
                    Q3 = df[col].quantile(0.99)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Feature cleaning failed: {e}")
            return df
    
    def process_enhanced_features(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Process enhanced features for all symbols
        处理所有币种的增强特征
        """
        try:
            self.logger.info(f"Starting enhanced feature engineering for {len(data_dict)} symbols...")
            
            # Step 1: Add individual symbol features
            processed_data = {}
            for symbol, df in data_dict.items():
                if len(df) > 100:  # Ensure minimum data quality
                    self.logger.info(f"Processing enhanced features for {symbol}...")
                    
                    # Make a copy to avoid modifying original
                    enhanced_df = df.copy()
                    
                    # Add enhanced DipMaster core features
                    enhanced_df = self.add_enhanced_dipmaster_core(enhanced_df, symbol)
                    
                    # Add microstructure features
                    enhanced_df = self.add_microstructure_features(enhanced_df, symbol)
                    
                    # Add ML features
                    enhanced_df = self.add_machine_learning_features(enhanced_df, symbol)
                    
                    # Add enhanced labels
                    enhanced_df = self.add_enhanced_labels(enhanced_df, symbol)
                    
                    # Clean and validate
                    enhanced_df = self.clean_and_validate_features(enhanced_df)
                    
                    processed_data[symbol] = enhanced_df
                    
            # Step 2: Add cross-symbol features
            if len(processed_data) > 1:
                self.logger.info("Adding cross-symbol features...")
                processed_data = self.add_cross_symbol_features(processed_data)
            
            # Step 3: Add advanced strategy signals (after cross-symbol features)
            for symbol in processed_data:
                self.logger.info(f"Adding advanced strategy signals for {symbol}...")
                processed_data[symbol] = self.add_advanced_strategy_signals(processed_data[symbol], symbol)
                processed_data[symbol] = self.apply_feature_selection_and_engineering(processed_data[symbol])
                
                # Final cleaning
                processed_data[symbol] = self.clean_and_validate_features(processed_data[symbol])
            
            self.logger.info("Enhanced feature engineering completed successfully!")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Enhanced feature processing failed: {e}")
            return data_dict

def main():
    """Demonstration of enhanced feature engineering"""
    logging.basicConfig(level=logging.INFO)
    
    engineer = EnhancedDipMasterFeatureEngineer()
    
    print("Enhanced DipMaster V4 Feature Engineering Pipeline")
    print("=" * 60)
    print("Core Features:")
    print("1. Multi-timeframe RSI convergence analysis")
    print("2. Dynamic RSI thresholds based on volatility")
    print("3. Enhanced dip detection with consecutive analysis")
    print("4. Cross-symbol relative strength and rotation")
    print("5. Market microstructure features")
    print("6. Advanced ML features (PCA, interactions)")
    print("7. Multi-layer strategy signal scoring")
    print("8. Enhanced labels with risk adjustment")
    print("9. Survival analysis for optimal timing")
    print("10. Feature selection and dimensionality reduction")
    print("=" * 60)
    print("Target: 85%+ win rate with comprehensive feature engineering")

if __name__ == "__main__":
    main()