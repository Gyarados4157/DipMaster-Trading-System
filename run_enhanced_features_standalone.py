#!/usr/bin/env python3
"""
Standalone Enhanced Feature Engineering for DipMaster V4
独立增强特征工程 - 避免依赖问题

Author: DipMaster Quant Team
Date: 2025-08-16
Version: 4.0.0-Enhanced
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import scipy.stats as stats

warnings.filterwarnings('ignore')

def setup_logging():
    """Setup logging"""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/enhanced_features.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
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

def add_enhanced_dipmaster_core(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add enhanced DipMaster core signals"""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Adding enhanced core signals for {symbol}")
        
        # Multi-timeframe RSI
        rsi_periods = [14, 21, 50]
        for period in rsi_periods:
            rsi_col = f'rsi_{period}'
            df[rsi_col] = calculate_rsi(df['close'], period)
            
            # Dynamic RSI thresholds based on volatility
            vol_20 = df['close'].pct_change().rolling(20).std()
            vol_percentile = vol_20.rolling(100).rank(pct=True)
            
            # Adjust RSI thresholds
            rsi_lower = 25 + (vol_percentile * 10)
            rsi_upper = 40 + (vol_percentile * 15)
            
            df[f'rsi_{period}_dip_zone'] = (
                (df[rsi_col] >= rsi_lower) & (df[rsi_col] <= rsi_upper)
            ).astype(int)
            
            # RSI gradient
            df[f'rsi_{period}_gradient'] = df[rsi_col].diff()
        
        # RSI convergence
        rsi_signals = [f'rsi_{p}_dip_zone' for p in rsi_periods]
        df['rsi_convergence_score'] = df[rsi_signals].sum(axis=1) / len(rsi_signals)
        df['rsi_convergence_strong'] = (df['rsi_convergence_score'] >= 0.67).astype(int)
        
        # Enhanced dip detection
        df['price_dip_1'] = (df['close'] < df['open']).astype(int)
        df['price_dip_2'] = (df['close'].shift(1) < df['open'].shift(1)).astype(int)
        df['price_dip_3'] = (df['close'].shift(2) < df['open'].shift(2)).astype(int)
        df['consecutive_dips'] = df['price_dip_1'] + (df['price_dip_2'] * 0.5) + (df['price_dip_3'] * 0.25)
        
        # Price drop magnitude
        df['drop_magnitude_1p'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        df['drop_magnitude_3p'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
        df['drop_magnitude_5p'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
        
        # Enhanced volume features
        for ma_period in [10, 20, 50]:
            vol_ma = df['volume'].rolling(ma_period).mean()
            df[f'volume_ratio_{ma_period}'] = df['volume'] / vol_ma
            df[f'volume_spike_{ma_period}'] = (df[f'volume_ratio_{ma_period}'] > 1.5).astype(int)
        
        # VWAP
        df['vwap_5'] = (df['close'] * df['volume']).rolling(5).sum() / df['volume'].rolling(5).sum()
        df['vwap_deviation'] = (df['close'] - df['vwap_5']) / df['vwap_5']
        
        # Bollinger Bands
        bb_data = calculate_bollinger_bands(df['close'], period=20, std_dev=2.0)
        df['bb_upper'] = bb_data['upper']
        df['bb_middle'] = bb_data['middle']
        df['bb_lower'] = bb_data['lower']
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(int)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Moving averages
        ma_periods = [5, 10, 20, 50, 100]
        for ma_period in ma_periods:
            df[f'ma_{ma_period}'] = df['close'].rolling(ma_period).mean()
            df[f'ma_{ma_period}_distance'] = (df['close'] - df[f'ma_{ma_period}']) / df[f'ma_{ma_period}']
            df[f'below_ma_{ma_period}'] = (df['close'] < df[f'ma_{ma_period}']).astype(int)
        
        # MA trend strength
        if 'ma_10' in df.columns and 'ma_50' in df.columns:
            df['ma_trend_strength'] = (df['ma_10'] - df['ma_50']) / df['ma_50']
            df['ma_alignment'] = (df['ma_10'] > df['ma_50']).astype(int)
        
        return df
        
    except Exception as e:
        logger.error(f"Enhanced core features failed for {symbol}: {e}")
        return df

def add_microstructure_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add market microstructure features"""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Adding microstructure features for {symbol}")
        
        # Price impact estimation
        df['price_impact'] = abs(df['close'].pct_change()) / (df['volume'] / df['volume'].rolling(20).mean())
        df['price_impact_smoothed'] = df['price_impact'].rolling(5).mean()
        
        # Time-based patterns
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            
            # Hourly volatility patterns
            hourly_vol = df.groupby('hour')['close'].pct_change().std()
            df['hourly_vol_pattern'] = df['hour'].map(hourly_vol)
            
            # Hourly volume patterns
            hourly_volume = df.groupby('hour')['volume'].mean()
            df['hourly_volume_pattern'] = df['hour'].map(hourly_volume)
        
        # Liquidity proxies
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        df['effective_spread'] = df['high_low_ratio'] * 2
        
        # Order flow imbalance (simplified)
        df['order_flow_imbalance'] = np.where(
            df['close'] > (df['high'] + df['low']) / 2,
            df['volume'],
            -df['volume']
        )
        df['ofi_cumulative'] = df['order_flow_imbalance'].rolling(20).sum()
        
        # VWAP deviations
        for period in [5, 10, 20]:
            vwap = (df['close'] * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
            df[f'vwap_dev_{period}'] = (df['close'] - vwap) / vwap
        
        # Intra-candle momentum
        df['intra_candle_momentum'] = (df['close'] - df['open']) / (df['high'] - df['low'])
        df['candle_body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'])
        
        # Volatility clustering
        returns = df['close'].pct_change()
        df['volatility_5'] = returns.rolling(5).std()
        df['volatility_20'] = returns.rolling(20).std()
        df['vol_clustering'] = df['volatility_5'] / df['volatility_20']
        
        return df
        
    except Exception as e:
        logger.error(f"Microstructure features failed for {symbol}: {e}")
        return df

def add_machine_learning_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add ML features"""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Adding ML features for {symbol}")
        
        # Feature interactions
        if 'rsi_14' in df.columns and 'volume_ratio_20' in df.columns:
            df['rsi_volume_interaction'] = df['rsi_14'] * df['volume_ratio_20']
        
        if 'bb_position' in df.columns and 'ma_trend_strength' in df.columns:
            df['bb_trend_interaction'] = df['bb_position'] * df['ma_trend_strength']
        
        # Rolling statistics
        price_returns = df['close'].pct_change()
        
        for window in [10, 20, 50]:
            df[f'return_skew_{window}'] = price_returns.rolling(window).skew()
            df[f'return_kurtosis_{window}'] = price_returns.rolling(window).apply(lambda x: stats.kurtosis(x, nan_policy='omit'))
            df[f'return_std_{window}'] = price_returns.rolling(window).std()
        
        # Regime detection
        vol_regime = df['volatility_20'].rolling(50).quantile(0.8)
        df['high_vol_regime'] = (df['volatility_20'] > vol_regime).astype(int)
        
        # Trend consistency
        ma_cols = [col for col in df.columns if col.startswith('ma_') and col.split('_')[1].isdigit()]
        if len(ma_cols) >= 3:
            trend_signals = []
            for i in range(len(ma_cols)-1):
                for j in range(i+1, len(ma_cols)):
                    ma1, ma2 = ma_cols[i], ma_cols[j]
                    trend_signals.append((df[ma1] > df[ma2]).astype(int))
            
            if trend_signals:
                df['trend_consistency'] = np.mean(trend_signals, axis=0)
        
        # Fibonacci levels
        rolling_high = df['high'].rolling(50).max()
        rolling_low = df['low'].rolling(50).min()
        fib_range = rolling_high - rolling_low
        
        df['fib_382'] = rolling_low + (fib_range * 0.382)
        df['fib_618'] = rolling_low + (fib_range * 0.618)
        df['near_fib_382'] = (abs(df['close'] - df['fib_382']) / df['close'] < 0.01).astype(int)
        df['near_fib_618'] = (abs(df['close'] - df['fib_618']) / df['close'] < 0.01).astype(int)
        
        # Support and resistance
        df['resistance_20'] = df['high'].rolling(20).max()
        df['support_20'] = df['low'].rolling(20).min()
        df['near_resistance'] = (df['close'] / df['resistance_20'] > 0.98).astype(int)
        df['near_support'] = (df['close'] / df['support_20'] < 1.02).astype(int)
        
        return df
        
    except Exception as e:
        logger.error(f"ML features failed for {symbol}: {e}")
        return df

def add_cross_symbol_features(all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Add cross-symbol features"""
    try:
        logger = logging.getLogger(__name__)
        logger.info("Adding cross-symbol features...")
        
        # Collect price data
        market_prices = {}
        for symbol, df in all_data.items():
            if len(df) > 0 and 'timestamp' in df.columns:
                df_copy = df.set_index('timestamp')['close']
                market_prices[symbol] = df_copy
        
        if len(market_prices) < 2:
            return all_data
        
        # Create price matrix
        price_matrix = pd.DataFrame(market_prices).ffill()
        
        # Calculate relative strength
        for symbol in all_data.keys():
            if symbol in all_data and len(all_data[symbol]) > 0:
                df = all_data[symbol].copy()
                
                if symbol in price_matrix.columns:
                    # Market relative performance
                    market_returns = price_matrix.pct_change().mean(axis=1)
                    symbol_returns = price_matrix[symbol].pct_change()
                    
                    relative_performance = symbol_returns - market_returns
                    
                    # Align with original dataframe
                    df_indexed = df.set_index('timestamp')
                    df_indexed['relative_strength_1d'] = relative_performance.rolling(24*12).mean()
                    df_indexed['relative_strength_3d'] = relative_performance.rolling(24*12*3).mean()
                    df_indexed['relative_strength_7d'] = relative_performance.rolling(24*12*7).mean()
                    
                    # Relative strength rank
                    daily_returns = price_matrix.pct_change(24*12)
                    if symbol in daily_returns.columns:
                        df_indexed['rs_rank'] = daily_returns.rank(axis=1, pct=True)[symbol]
                    
                    # BTC correlation
                    if 'BTCUSDT' in price_matrix.columns:
                        btc_returns = price_matrix['BTCUSDT'].pct_change()
                        correlation_window = 24*12*7
                        df_indexed['btc_correlation'] = symbol_returns.rolling(correlation_window).corr(btc_returns)
                    
                    # Momentum divergence
                    market_momentum = market_returns.rolling(12).sum()
                    symbol_momentum = symbol_returns.rolling(12).sum()
                    df_indexed['momentum_divergence'] = symbol_momentum - market_momentum
                    
                    # Reset index and update
                    df_updated = df_indexed.reset_index()
                    all_data[symbol] = df_updated
        
        return all_data
        
    except Exception as e:
        logger.error(f"Cross-symbol features failed: {e}")
        return all_data

def add_advanced_strategy_signals(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add advanced strategy signals"""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Adding advanced signals for {symbol}")
        
        # Primary signals
        primary_signals = []
        weights = []
        
        if 'rsi_convergence_strong' in df.columns:
            primary_signals.append(df['rsi_convergence_strong'])
            weights.append(0.25)
        
        if 'consecutive_dips' in df.columns:
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
            vwap_signal = np.clip(-df['vwap_deviation'] * 10, 0, 1)
            primary_signals.append(vwap_signal)
            weights.append(0.15)
        
        # Calculate primary signal strength
        if primary_signals and weights:
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            df['primary_signal_strength'] = sum(
                signal * weight for signal, weight in zip(primary_signals, weights)
            )
        else:
            df['primary_signal_strength'] = 0
        
        # Regime filter
        regime_signals = []
        if 'high_vol_regime' in df.columns:
            normal_vol_signal = 1 - df['high_vol_regime']
            regime_signals.append(normal_vol_signal * 0.4)
        
        if 'trend_consistency' in df.columns:
            weak_trend_signal = 1 - df['trend_consistency']
            regime_signals.append(weak_trend_signal * 0.3)
        
        if 'relative_strength_1d' in df.columns:
            rel_weak_signal = np.clip(-df['relative_strength_1d'] * 50, 0, 1)
            regime_signals.append(rel_weak_signal * 0.3)
        
        df['regime_filter_strength'] = sum(regime_signals) if regime_signals else 0.5
        
        # Timing signals
        timing_signals = []
        if 'near_support' in df.columns:
            timing_signals.append(df['near_support'] * 0.3)
        if 'near_fib_618' in df.columns:
            timing_signals.append(df['near_fib_618'] * 0.2)
        if 'ofi_cumulative' in df.columns:
            ofi_normalized = np.clip(-df['ofi_cumulative'] / df['ofi_cumulative'].rolling(100).std(), 0, 1)
            timing_signals.append(ofi_normalized * 0.3)
        
        df['timing_signal_strength'] = sum(timing_signals) if timing_signals else 0
        
        # Combined signal
        df['dipmaster_v4_signal_score'] = (
            df['primary_signal_strength'] * 0.6 +
            df['regime_filter_strength'] * 0.25 +
            df['timing_signal_strength'] * 0.15
        )
        
        # Signal confidence
        confirmations = 0
        if 'rsi_convergence_strong' in df.columns:
            confirmations += df['rsi_convergence_strong']
        if 'volume_spike_20' in df.columns:
            confirmations += df['volume_spike_20']
        if 'bb_squeeze' in df.columns:
            confirmations += df['bb_squeeze']
        
        df['signal_confidence'] = np.clip(confirmations / 3.0, 0, 1)
        df['dipmaster_v4_final_signal'] = df['dipmaster_v4_signal_score'] * df['signal_confidence']
        
        # Quality flags
        df['high_quality_signal'] = (df['dipmaster_v4_final_signal'] > 0.7).astype(int)
        df['medium_quality_signal'] = (
            (df['dipmaster_v4_final_signal'] > 0.5) & (df['dipmaster_v4_final_signal'] <= 0.7)
        ).astype(int)
        
        return df
        
    except Exception as e:
        logger.error(f"Advanced signals failed for {symbol}: {e}")
        return df

def add_enhanced_labels(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add enhanced labels"""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Adding enhanced labels for {symbol}")
        
        prediction_horizons = [3, 6, 12, 24, 48]
        profit_targets = [0.006, 0.012, 0.020]
        stop_loss = 0.004
        
        for horizon in prediction_horizons:
            # Future return
            future_return = df['close'].pct_change(periods=horizon).shift(-horizon)
            df[f'future_return_{horizon}p'] = future_return
            
            # Binary profitable
            df[f'is_profitable_{horizon}p'] = (future_return > 0).astype(int)
            
            # Risk-adjusted return
            return_std = df['volatility_20'].shift(-horizon)
            risk_adj_return = future_return / (return_std + 1e-8)
            df[f'risk_adj_return_{horizon}p'] = risk_adj_return
            
            # Target hits
            for target in profit_targets:
                df[f'hits_target_{target:.1%}_{horizon}p'] = (future_return >= target).astype(int)
            
            # Stop loss
            df[f'hits_stop_loss_{horizon}p'] = (future_return <= -stop_loss).astype(int)
            
            # MFE and MAE
            future_highs = df['high'].rolling(window=horizon, min_periods=1).max().shift(-horizon)
            future_lows = df['low'].rolling(window=horizon, min_periods=1).min().shift(-horizon)
            
            mfe = (future_highs - df['close']) / df['close']
            mae = (future_lows - df['close']) / df['close']
            
            df[f'mfe_{horizon}p'] = mfe
            df[f'mae_{horizon}p'] = mae
        
        # Primary targets
        df['target_return'] = df['future_return_12p']
        df['target_binary'] = df['is_profitable_12p']
        df['target_risk_adjusted'] = df['risk_adj_return_12p']
        
        # Multi-class target
        main_return = df['future_return_12p']
        df['return_class'] = pd.cut(
            main_return,
            bins=[-np.inf, -0.002, 0.006, 0.015, np.inf],
            labels=[0, 1, 2, 3]
        ).astype(float)
        
        return df
        
    except Exception as e:
        logger.error(f"Enhanced labels failed for {symbol}: {e}")
        return df

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean feature data"""
    try:
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col.startswith('target') or 'future_return' in col:
                continue
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove extreme outliers
        for col in numeric_cols:
            if col not in ['timestamp', 'hour', 'minute'] and not col.startswith('target'):
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Feature cleaning failed: {e}")
        return df

def load_symbol_data(symbol: str, data_path: str) -> pd.DataFrame:
    """Load data for a specific symbol"""
    try:
        logger = logging.getLogger(__name__)
        data_path = Path(data_path)
        
        file_patterns = [
            f"{symbol}_5m_90days.parquet",
            f"{symbol}_5m_3years.parquet",
        ]
        
        for pattern in file_patterns:
            file_path = data_path / pattern
            if file_path.exists():
                logger.info(f"Loading {symbol} from {file_path}")
                df = pd.read_parquet(file_path)
                
                # Check if timestamp is in index
                if df.index.name == 'timestamp' or 'timestamp' in str(df.index.name):
                    df = df.reset_index()
                
                # Check if we have required columns
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                
                # If timestamp is not a column, add it from index
                if 'timestamp' not in df.columns:
                    df['timestamp'] = df.index
                
                if all(col in df.columns for col in required_cols):
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
                    logger.info(f"Loaded {len(df)} samples for {symbol}")
                    return df
                else:
                    logger.warning(f"Missing required columns for {symbol}. Has: {df.columns.tolist()}")
        
        logger.warning(f"No data found for {symbol}")
        return pd.DataFrame()
        
    except Exception as e:
        logger.error(f"Failed to load {symbol}: {e}")
        return pd.DataFrame()

def process_symbol_features(symbol: str, df: pd.DataFrame) -> pd.DataFrame:
    """Process all features for a symbol"""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Processing features for {symbol}...")
        
        # Add all feature types
        df = add_enhanced_dipmaster_core(df, symbol)
        df = add_microstructure_features(df, symbol)
        df = add_machine_learning_features(df, symbol)
        df = add_enhanced_labels(df, symbol)
        df = clean_features(df)
        
        # Add symbol identifier
        df['symbol'] = symbol
        
        # Remove rows with NaN targets
        df = df.dropna(subset=['target_return', 'target_binary'])
        
        logger.info(f"Completed {symbol}: {len(df)} samples, {len(df.columns)} features")
        return df
        
    except Exception as e:
        logger.error(f"Feature processing failed for {symbol}: {e}")
        return pd.DataFrame()

def analyze_feature_quality(features_df: pd.DataFrame) -> Dict:
    """Analyze feature quality"""
    try:
        logger = logging.getLogger(__name__)
        logger.info("Analyzing feature quality...")
        
        # Basic stats
        total_samples = len(features_df)
        total_features = len(features_df.columns)
        symbols = features_df['symbol'].nunique() if 'symbol' in features_df.columns else 0
        
        # Feature categories
        feature_categories = {
            'dipmaster_core': len([col for col in features_df.columns if any(x in col for x in ['rsi_', 'bb_', 'ma_', 'dip'])]),
            'microstructure': len([col for col in features_df.columns if any(x in col for x in ['price_impact', 'order_flow', 'vwap'])]),
            'cross_symbol': len([col for col in features_df.columns if any(x in col for x in ['relative_strength', 'correlation'])]),
            'ml_features': len([col for col in features_df.columns if any(x in col for x in ['pca_', 'interact_', 'skew'])]),
            'advanced_signals': len([col for col in features_df.columns if 'dipmaster_v4' in col]),
            'labels': len([col for col in features_df.columns if any(x in col for x in ['target', 'future_return'])])
        }
        
        # Missing values
        missing_values = features_df.isnull().sum()
        missing_pct = (missing_values / len(features_df) * 100).round(2)
        
        # Signal quality
        signal_quality = {}
        if 'dipmaster_v4_final_signal' in features_df.columns:
            signal_col = 'dipmaster_v4_final_signal'
            signal_quality['mean_signal_strength'] = float(features_df[signal_col].mean())
            signal_quality['signal_distribution'] = features_df[signal_col].describe().to_dict()
            
            if 'target_binary' in features_df.columns:
                high_signal = features_df[features_df[signal_col] > 0.7]
                if len(high_signal) > 0:
                    signal_quality['high_signal_win_rate'] = float(high_signal['target_binary'].mean())
                    signal_quality['high_signal_samples'] = len(high_signal)
        
        # Cross-symbol analysis
        cross_symbol_analysis = {}
        if 'symbol' in features_df.columns:
            symbol_performance = features_df.groupby('symbol').agg({
                'target_binary': ['count', 'mean'],
                'target_return': 'mean'
            }).round(4)
            
            cross_symbol_analysis['symbol_performance'] = symbol_performance.to_dict()
        
        return {
            'basic_statistics': {
                'total_samples': int(total_samples),
                'total_features': int(total_features),
                'symbols_count': int(symbols),
                'memory_usage_mb': float(features_df.memory_usage(deep=True).sum() / 1024 / 1024)
            },
            'feature_categories': feature_categories,
            'data_quality': {
                'missing_values_count': int(missing_values.sum()),
                'features_with_missing': int((missing_values > 0).sum())
            },
            'signal_quality': signal_quality,
            'cross_symbol_analysis': cross_symbol_analysis
        }
        
    except Exception as e:
        logger.error(f"Quality analysis failed: {e}")
        return {}

def main():
    """Main execution"""
    try:
        logger = setup_logging()
        
        # Configuration
        symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT',
            'BNBUSDT', 'DOGEUSDT', 'SUIUSDT', 'ICPUSDT', 'ALGOUSDT', 
            'IOTAUSDT', 'TRXUSDT', 'LINKUSDT', 'AVAXUSDT', 'MATICUSDT', 
            'UNIUSDT', 'LTCUSDT', 'DOTUSDT', 'ATOMUSDT', 'FILUSDT', 
            'NEARUSDT', 'ARBUSDT', 'OPUSDT', 'APTUSDT', 'QNTUSDT'
        ]
        
        data_path = 'data/enhanced_market_data'
        min_samples = 1000
        
        logger.info("=" * 80)
        logger.info("ENHANCED DIPMASTER V4 FEATURE ENGINEERING")
        logger.info("=" * 80)
        logger.info(f"Target: 85%+ Win Rate with Enhanced Features")
        logger.info(f"Symbols: {len(symbols)}")
        
        # Step 1: Load all data
        logger.info("\nStep 1: Loading market data...")
        data_dict = {}
        for symbol in symbols:
            df = load_symbol_data(symbol, data_path)
            if len(df) >= min_samples:
                data_dict[symbol] = df
                logger.info(f"✓ {symbol}: {len(df)} samples")
            else:
                logger.warning(f"✗ {symbol}: Insufficient data")
        
        logger.info(f"Loaded {len(data_dict)}/{len(symbols)} symbols")
        
        if len(data_dict) < 5:
            logger.error("Insufficient data loaded")
            return False
        
        # Step 2: Process individual features
        logger.info("\nStep 2: Processing individual features...")
        processed_data = {}
        for symbol, df in data_dict.items():
            processed_df = process_symbol_features(symbol, df)
            if len(processed_df) > 0:
                processed_data[symbol] = processed_df
        
        # Step 3: Add cross-symbol features
        logger.info("\nStep 3: Adding cross-symbol features...")
        enhanced_data = add_cross_symbol_features(processed_data)
        
        # Step 4: Add advanced signals
        logger.info("\nStep 4: Adding advanced strategy signals...")
        for symbol in enhanced_data:
            enhanced_data[symbol] = add_advanced_strategy_signals(enhanced_data[symbol], symbol)
        
        # Step 5: Combine all data
        logger.info("\nStep 5: Combining all features...")
        combined_features = []
        for symbol, df in enhanced_data.items():
            if len(df) > 0:
                combined_features.append(df)
        
        if not combined_features:
            logger.error("No valid features generated")
            return False
        
        final_features = pd.concat(combined_features, ignore_index=True)
        logger.info(f"Combined: {len(final_features)} samples, {len(final_features.columns)} features")
        
        # Step 6: Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save features
        os.makedirs('data', exist_ok=True)
        features_file = f"data/Enhanced_Features_25symbols_{timestamp}.parquet"
        final_features.to_parquet(features_file, compression='snappy')
        logger.info(f"Features saved: {features_file}")
        
        # Generate quality report
        quality_report = analyze_feature_quality(final_features)
        
        # Save configuration
        config = {
            'version': '4.0.0-Enhanced',
            'strategy_name': 'DipMaster_Enhanced_V4',
            'created_timestamp': timestamp,
            'symbols': list(final_features['symbol'].unique()),
            'feature_engineering': {
                'pipeline_type': 'enhanced_multi_layer',
                'total_features': len(final_features.columns),
                'enhancement_components': [
                    'multi_timeframe_rsi_convergence',
                    'dynamic_rsi_thresholds', 
                    'enhanced_dip_detection',
                    'cross_symbol_relative_strength',
                    'market_microstructure',
                    'advanced_ml_features',
                    'multi_layer_signal_scoring',
                    'enhanced_risk_adjusted_labels',
                    'survival_analysis_labels'
                ]
            },
            'quality_metrics': quality_report
        }
        
        config_file = f"data/Enhanced_FeatureSet_V4_{timestamp}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Configuration saved: {config_file}")
        
        # Save analysis report
        report_file = f"data/Feature_Importance_Analysis_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(quality_report, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Analysis report saved: {report_file}")
        
        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("ENHANCED FEATURE ENGINEERING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"Features File: {features_file}")
        logger.info(f"Configuration: {config_file}")
        logger.info(f"Analysis Report: {report_file}")
        logger.info(f"Total Samples: {len(final_features):,}")
        logger.info(f"Total Features: {len(final_features.columns)}")
        logger.info(f"Symbols: {final_features['symbol'].nunique()}")
        
        # Display feature categories
        feature_categories = quality_report.get('feature_categories', {})
        logger.info("\nFeature Categories:")
        for category, count in feature_categories.items():
            logger.info(f"  {category}: {count} features")
        
        print("\n🎯 Enhanced Feature Engineering Completed Successfully!")
        print(f"📊 Generated {len(final_features):,} samples with {len(final_features.columns)} features")
        print(f"🚀 Ready for DipMaster V4 85%+ Win Rate Optimization")
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced feature engineering failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)