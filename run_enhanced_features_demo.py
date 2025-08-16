#!/usr/bin/env python3
"""
Enhanced Feature Engineering Demo for DipMaster V4
增强特征工程演示版 - 快速生成主要币种的增强特征

Author: DipMaster Quant Team
Date: 2025-08-16
Version: 4.0.0-Enhanced-Demo
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

warnings.filterwarnings('ignore')

def setup_logging():
    """Setup logging"""
    os.makedirs('logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/enhanced_features_demo.log'),
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
        rsi_periods = [14, 21]  # Simplified
        for period in rsi_periods:
            rsi_col = f'rsi_{period}'
            df[rsi_col] = calculate_rsi(df['close'], period)
            
            # Dynamic RSI thresholds
            vol_20 = df['close'].pct_change().rolling(20).std()
            vol_percentile = vol_20.rolling(100).rank(pct=True)
            
            rsi_lower = 25 + (vol_percentile * 10)
            rsi_upper = 40 + (vol_percentile * 15)
            
            df[f'rsi_{period}_dip_zone'] = (
                (df[rsi_col] >= rsi_lower) & (df[rsi_col] <= rsi_upper)
            ).astype(int)
            
            df[f'rsi_{period}_gradient'] = df[rsi_col].diff()
        
        # RSI convergence
        rsi_signals = [f'rsi_{p}_dip_zone' for p in rsi_periods]
        df['rsi_convergence_score'] = df[rsi_signals].sum(axis=1) / len(rsi_signals)
        df['rsi_convergence_strong'] = (df['rsi_convergence_score'] >= 0.5).astype(int)
        
        # Enhanced dip detection
        df['price_dip_1'] = (df['close'] < df['open']).astype(int)
        df['price_dip_2'] = (df['close'].shift(1) < df['open'].shift(1)).astype(int)
        df['consecutive_dips'] = df['price_dip_1'] + (df['price_dip_2'] * 0.5)
        
        # Price drop magnitude
        df['drop_magnitude_1p'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
        df['drop_magnitude_3p'] = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
        
        # Enhanced volume features
        for ma_period in [20]:  # Simplified
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
        ma_periods = [10, 20, 50]
        for ma_period in ma_periods:
            df[f'ma_{ma_period}'] = df['close'].rolling(ma_period).mean()
            df[f'ma_{ma_period}_distance'] = (df['close'] - df[f'ma_{ma_period}']) / df[f'ma_{ma_period}']
            df[f'below_ma_{ma_period}'] = (df['close'] < df[f'ma_{ma_period}']).astype(int)
        
        # MA trend strength
        df['ma_trend_strength'] = (df['ma_10'] - df['ma_50']) / df['ma_50']
        df['ma_alignment'] = (df['ma_10'] > df['ma_50']).astype(int)
        
        return df
        
    except Exception as e:
        logger.error(f"Enhanced core features failed for {symbol}: {e}")
        return df

def add_microstructure_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add simplified microstructure features"""
    try:
        logger = logging.getLogger(__name__)
        logger.info(f"Adding microstructure features for {symbol}")
        
        # Price impact estimation
        volume_mean = df['volume'].rolling(20).mean()
        df['price_impact'] = abs(df['close'].pct_change()) / (df['volume'] / volume_mean)
        df['price_impact_smoothed'] = df['price_impact'].rolling(5).mean()
        
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
        
        # Intra-candle momentum
        high_low_diff = df['high'] - df['low']
        high_low_diff = high_low_diff.replace(0, np.nan)  # Avoid division by zero
        df['intra_candle_momentum'] = (df['close'] - df['open']) / high_low_diff
        df['candle_body_ratio'] = abs(df['close'] - df['open']) / high_low_diff
        
        # Volatility clustering
        returns = df['close'].pct_change()
        df['volatility_5'] = returns.rolling(5).std()
        df['volatility_20'] = returns.rolling(20).std()
        vol_20_safe = df['volatility_20'].replace(0, np.nan)
        df['vol_clustering'] = df['volatility_5'] / vol_20_safe
        
        return df
        
    except Exception as e:
        logger.error(f"Microstructure features failed for {symbol}: {e}")
        return df

def add_machine_learning_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Add simplified ML features"""
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
        
        for window in [20]:  # Simplified
            df[f'return_skew_{window}'] = price_returns.rolling(window).skew()
            df[f'return_std_{window}'] = price_returns.rolling(window).std()
        
        # Regime detection
        if 'volatility_20' in df.columns:
            vol_regime = df['volatility_20'].rolling(50).quantile(0.8)
            df['high_vol_regime'] = (df['volatility_20'] > vol_regime).astype(int)
        
        # Support and resistance
        df['resistance_20'] = df['high'].rolling(20).max()
        df['support_20'] = df['low'].rolling(20).min()
        df['near_resistance'] = (df['close'] / df['resistance_20'] > 0.98).astype(int)
        df['near_support'] = (df['close'] / df['support_20'] < 1.02).astype(int)
        
        return df
        
    except Exception as e:
        logger.error(f"ML features failed for {symbol}: {e}")
        return df

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
            regime_signals.append(normal_vol_signal * 0.5)
        
        df['regime_filter_strength'] = sum(regime_signals) if regime_signals else 0.5
        
        # Timing signals
        timing_signals = []
        if 'near_support' in df.columns:
            timing_signals.append(df['near_support'] * 0.5)
        if 'ofi_cumulative' in df.columns:
            ofi_std = df['ofi_cumulative'].rolling(100).std()
            ofi_std = ofi_std.replace(0, np.nan)
            ofi_normalized = np.clip(-df['ofi_cumulative'] / ofi_std, 0, 1)
            timing_signals.append(ofi_normalized * 0.5)
        
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
        
        df['signal_confidence'] = np.clip(confirmations / 2.0, 0, 1)
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
        
        prediction_horizons = [3, 6, 12, 24]  # Simplified
        profit_targets = [0.006, 0.012]  # Simplified
        stop_loss = 0.004
        
        for horizon in prediction_horizons:
            # Future return
            future_return = df['close'].pct_change(periods=horizon).shift(-horizon)
            df[f'future_return_{horizon}p'] = future_return
            
            # Binary profitable
            df[f'is_profitable_{horizon}p'] = (future_return > 0).astype(int)
            
            # Risk-adjusted return
            if 'volatility_20' in df.columns:
                return_std = df['volatility_20'].shift(-horizon)
                return_std = return_std.replace(0, np.nan)
                risk_adj_return = future_return / (return_std + 1e-8)
                df[f'risk_adj_return_{horizon}p'] = risk_adj_return
            
            # Target hits
            for target in profit_targets:
                df[f'hits_target_{target:.1%}_{horizon}p'] = (future_return >= target).astype(int)
            
            # Stop loss
            df[f'hits_stop_loss_{horizon}p'] = (future_return <= -stop_loss).astype(int)
        
        # Primary targets
        df['target_return'] = df['future_return_12p']
        df['target_binary'] = df['is_profitable_12p']
        if 'risk_adj_return_12p' in df.columns:
            df['target_risk_adjusted'] = df['risk_adj_return_12p']
        
        return df
        
    except Exception as e:
        logger.error(f"Enhanced labels failed for {symbol}: {e}")
        return df

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean feature data"""
    try:
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values intelligently
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col.startswith('target') or 'future_return' in col:
                continue
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
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
        df = add_advanced_strategy_signals(df, symbol)
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
            'ml_features': len([col for col in features_df.columns if any(x in col for x in ['interact_', 'skew'])]),
            'advanced_signals': len([col for col in features_df.columns if 'dipmaster_v4' in col]),
            'labels': len([col for col in features_df.columns if any(x in col for x in ['target', 'future_return'])])
        }
        
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
        
        # Demo configuration - top performing symbols
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT']
        data_path = 'data/enhanced_market_data'
        min_samples = 1000
        
        logger.info("=" * 80)
        logger.info("ENHANCED DIPMASTER V4 FEATURE ENGINEERING - DEMO")
        logger.info("=" * 80)
        logger.info(f"Target: 85%+ Win Rate with Enhanced Features")
        logger.info(f"Demo Symbols: {symbols}")
        
        # Step 1: Load all data
        logger.info("\nStep 1: Loading market data...")
        data_dict = {}
        for symbol in symbols:
            df = load_symbol_data(symbol, data_path)
            if len(df) >= min_samples:
                data_dict[symbol] = df
                logger.info(f"OK {symbol}: {len(df)} samples")
            else:
                logger.warning(f"X {symbol}: Insufficient data")
        
        logger.info(f"Loaded {len(data_dict)}/{len(symbols)} symbols")
        
        if len(data_dict) == 0:
            logger.error("No data loaded")
            return False
        
        # Step 2: Process features
        logger.info(f"\nStep 2: Processing enhanced features...")
        processed_data = {}
        for symbol, df in data_dict.items():
            processed_df = process_symbol_features(symbol, df)
            if len(processed_df) > 0:
                processed_data[symbol] = processed_df
        
        # Step 3: Combine all data
        logger.info("\nStep 3: Combining all features...")
        combined_features = []
        for symbol, df in processed_data.items():
            if len(df) > 0:
                combined_features.append(df)
        
        if not combined_features:
            logger.error("No valid features generated")
            return False
        
        final_features = pd.concat(combined_features, ignore_index=True)
        logger.info(f"Combined: {len(final_features)} samples, {len(final_features.columns)} features")
        
        # Step 4: Save results
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
                    'market_microstructure',
                    'advanced_ml_features',
                    'multi_layer_signal_scoring',
                    'enhanced_risk_adjusted_labels'
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
        logger.info("ENHANCED FEATURE ENGINEERING DEMO COMPLETED!")
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
        
        # Signal quality summary
        signal_quality = quality_report.get('signal_quality', {})
        if signal_quality:
            logger.info("\nSignal Quality:")
            logger.info(f"  Mean Signal Strength: {signal_quality.get('mean_signal_strength', 0):.4f}")
            if 'high_signal_win_rate' in signal_quality:
                logger.info(f"  High Signal Win Rate: {signal_quality['high_signal_win_rate']:.2%}")
                logger.info(f"  High Signal Samples: {signal_quality['high_signal_samples']}")
        
        print("\n🎯 Enhanced Feature Engineering Demo Completed Successfully!")
        print(f"📊 Generated {len(final_features):,} samples with {len(final_features.columns)} features")
        print(f"🚀 Ready for DipMaster V4 85%+ Win Rate Optimization")
        
        return True
        
    except Exception as e:
        logger.error(f"Enhanced feature engineering demo failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)