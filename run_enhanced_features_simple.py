#!/usr/bin/env python3
"""
DipMaster Enhanced Feature Generation - Simple Version
增强版特征生成 - 简化版

Generate comprehensive features for DipMaster strategy with validation.
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
import time
import warnings
import ta

warnings.filterwarnings('ignore')

# Set up logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger(__name__)

def load_sample_data():
    """Load sample data from available symbols"""
    data_dir = Path("data/enhanced_market_data")
    
    # Try a few key symbols first
    priority_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT']
    market_data = {}
    
    for symbol in priority_symbols:
        try:
            file_path = data_dir / f"{symbol}_5m_90days.parquet"
            if file_path.exists():
                logger.info(f"Loading {symbol}...")
                df = pd.read_parquet(file_path)
                
                # Check if timestamp is index
                if 'timestamp' in df.index.names or isinstance(df.index, pd.DatetimeIndex):
                    df = df.reset_index()
                
                # Validate columns
                required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                if all(col in df.columns for col in required_cols):
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df[df['close'] > 0].copy()
                    
                    if len(df) >= 1000:
                        market_data[symbol] = df
                        logger.info(f"Successfully loaded {symbol}: {len(df)} rows")
                    else:
                        logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                else:
                    logger.warning(f"Missing required columns for {symbol}")
            else:
                logger.warning(f"File not found: {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to load {symbol}: {e}")
    
    logger.info(f"Loaded {len(market_data)} symbols")
    return market_data

def add_technical_indicators(df):
    """Add comprehensive technical indicators"""
    try:
        # RSI variants
        for period in [7, 14, 21, 30]:
            df[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'], window_fast=12, window_slow=26, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14)
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        df['williams_r'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=14).williams_r()
        
        # CCI
        df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=14).cci()
        
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = ta.trend.SMAIndicator(df['close'], window=period).sma_indicator()
            df[f'ema_{period}'] = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_position'] = bb.bollinger_pband()
        
        # Volume indicators
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
        df['volume_sma'] = ta.volume.VolumeSMAIndicator(df['close'], df['volume'], window=20).volume_sma()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
        
        # Momentum
        df['momentum'] = df['close'] / df['close'].shift(10) - 1
        df['roc'] = ta.momentum.ROCIndicator(df['close'], window=10).roc()
        
        return df
        
    except Exception as e:
        logger.error(f"Technical indicators failed: {e}")
        return df

def add_microstructure_features(df):
    """Add market microstructure features"""
    try:
        # Basic candle analysis
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        df['total_range'] = df['high'] - df['low']
        
        # Ratios
        df['body_ratio'] = df['body_size'] / (df['total_range'] + 1e-8)
        df['upper_shadow_ratio'] = df['upper_shadow'] / (df['total_range'] + 1e-8)
        df['lower_shadow_ratio'] = df['lower_shadow'] / (df['total_range'] + 1e-8)
        
        # Patterns
        df['hammer'] = ((df['lower_shadow_ratio'] > 0.5) & (df['body_ratio'] < 0.3)).astype(int)
        df['doji'] = (df['body_ratio'] < 0.1).astype(int)
        
        # VWAP
        for period in [5, 20]:
            vwap_num = (df['close'] * df['volume']).rolling(period).sum()
            vwap_den = df['volume'].rolling(period).sum()
            df[f'vwap_{period}'] = vwap_num / vwap_den
            df[f'vwap_dev_{period}'] = (df['close'] - df[f'vwap_{period}']) / df[f'vwap_{period}']
        
        # Volume analysis
        for period in [10, 20]:
            df[f'volume_ratio_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
        
        # Order flow proxy
        df['buy_volume'] = np.where(df['close'] > df['open'], df['volume'], 0)
        df['sell_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)
        
        for period in [5, 10]:
            buy_vol = df['buy_volume'].rolling(period).sum()
            sell_vol = df['sell_volume'].rolling(period).sum()
            total_vol = df['volume'].rolling(period).sum()
            df[f'buy_ratio_{period}'] = buy_vol / total_vol
            df[f'order_flow_imbalance_{period}'] = (buy_vol - sell_vol) / total_vol
        
        # Time features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        return df
        
    except Exception as e:
        logger.error(f"Microstructure features failed: {e}")
        return df

def add_dipmaster_signals(df):
    """Add DipMaster specific signals"""
    try:
        # Core DipMaster conditions
        df['rsi_dip_zone'] = ((df['rsi_14'] >= 30) & (df['rsi_14'] <= 50)).astype(int)
        df['price_dip'] = (df['close'] < df['open']).astype(int)
        df['below_ma20'] = (df['close'] < df['sma_20']).astype(int)
        df['volume_confirmation'] = (df['volume_ratio_20'] > 1.2).astype(int)
        df['bb_lower_zone'] = (df['bb_position'] < 0.2).astype(int)
        
        # Combined signal
        components = [
            df['rsi_dip_zone'] * 0.3,
            df['price_dip'] * 0.25,
            df['below_ma20'] * 0.2,
            df['volume_confirmation'] * 0.15,
            df['bb_lower_zone'] * 0.1
        ]
        
        df['dipmaster_signal'] = sum(components)
        df['high_signal'] = (df['dipmaster_signal'] > 0.6).astype(int)
        df['medium_signal'] = ((df['dipmaster_signal'] > 0.4) & (df['dipmaster_signal'] <= 0.6)).astype(int)
        
        return df
        
    except Exception as e:
        logger.error(f"DipMaster signals failed: {e}")
        return df

def add_labels(df):
    """Add DipMaster optimized labels"""
    try:
        # Future returns for different horizons
        horizons = [3, 6, 12, 24, 36]  # 15min, 30min, 1h, 2h, 3h
        
        for horizon in horizons:
            future_return = df['close'].pct_change(periods=horizon).shift(-horizon)
            df[f'future_return_{horizon}p'] = future_return
            df[f'is_profitable_{horizon}p'] = (future_return > 0).astype(int)
            
            # Profit targets
            targets = [0.003, 0.006, 0.012]  # 0.3%, 0.6%, 1.2%
            for target in targets:
                df[f'hits_{target:.1%}_{horizon}p'] = (future_return >= target).astype(int)
            
            # Stop loss
            df[f'hits_stop_{horizon}p'] = (future_return <= -0.004).astype(int)
        
        # 15-minute boundary analysis
        df['minute_boundary'] = df['minute'].isin([15, 30, 45, 0]).astype(int)
        
        # Primary targets (1-hour horizon)
        main_return = df['future_return_12p']
        df['target_return'] = main_return
        df['target_binary'] = (main_return > 0).astype(int)
        df['target_0.6%'] = (main_return >= 0.006).astype(int)
        
        # DipMaster win condition
        df['dipmaster_win'] = (
            (main_return >= 0.006) |  # Direct profit target
            ((main_return >= 0.003) & (df['minute_boundary'].shift(-12) == 1))  # Boundary exit
        ).astype(int)
        
        # Multi-class targets
        conditions = [
            (main_return <= -0.004),  # Loss
            ((main_return > -0.004) & (main_return <= 0)),  # Small loss
            ((main_return > 0) & (main_return < 0.006)),  # Small profit
            ((main_return >= 0.006) & (main_return < 0.012)),  # Good profit
            (main_return >= 0.012)  # Excellent profit
        ]
        df['return_class'] = np.select(conditions, [0, 1, 2, 3, 4], default=1)
        
        return df
        
    except Exception as e:
        logger.error(f"Labels failed: {e}")
        return df

def add_cross_asset_features(all_data):
    """Add cross-asset features"""
    try:
        if len(all_data) < 2:
            return all_data
        
        # Create market matrix
        price_data = {}
        for symbol, df in all_data.items():
            df_indexed = df.set_index('timestamp')
            price_data[symbol] = df_indexed['close']
        
        price_df = pd.DataFrame(price_data).fillna(method='ffill').fillna(method='bfill')
        returns_df = price_df.pct_change()
        market_return = returns_df.mean(axis=1)
        
        # Add relative features
        for symbol in all_data:
            if symbol in returns_df.columns:
                df = all_data[symbol].copy()
                df_indexed = df.set_index('timestamp')
                
                symbol_returns = returns_df[symbol]
                relative_returns = symbol_returns - market_return
                
                # Relative strength
                df_indexed['relative_strength_1h'] = relative_returns.rolling(12).mean()
                df_indexed['relative_strength_4h'] = relative_returns.rolling(48).mean()
                
                # Market correlation
                df_indexed['market_correlation'] = symbol_returns.rolling(144).corr(market_return)
                
                # BTC correlation
                if 'BTCUSDT' in returns_df.columns and symbol != 'BTCUSDT':
                    btc_returns = returns_df['BTCUSDT']
                    df_indexed['btc_correlation'] = symbol_returns.rolling(144).corr(btc_returns)
                
                # Update data
                all_data[symbol] = df_indexed.reset_index()
        
        return all_data
        
    except Exception as e:
        logger.error(f"Cross-asset features failed: {e}")
        return all_data

def clean_data(df):
    """Clean and validate data"""
    try:
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col.startswith('target') or 'future_' in col:
                continue
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove extreme outliers
        for col in numeric_cols:
            if col not in ['timestamp', 'hour', 'minute', 'day_of_week'] and not col.startswith('target'):
                Q1 = df[col].quantile(0.01)
                Q3 = df[col].quantile(0.99)
                IQR = Q3 - Q1
                lower = Q1 - 2 * IQR
                upper = Q3 + 2 * IQR
                df[col] = df[col].clip(lower=lower, upper=upper)
        
        return df
        
    except Exception as e:
        logger.error(f"Data cleaning failed: {e}")
        return df

def analyze_features(combined_df):
    """Analyze feature quality"""
    feature_cols = [col for col in combined_df.columns 
                   if not any(x in col for x in ['target', 'future_', 'hits_'])
                   and col not in ['timestamp', 'symbol']]
    
    label_cols = [col for col in combined_df.columns 
                 if any(x in col for x in ['target', 'future_', 'hits_'])]
    
    # Feature categories
    categories = {
        'technical': [col for col in feature_cols if any(
            x in col for x in ['rsi', 'macd', 'stoch', 'williams', 'cci', 'adx', 'sma', 'ema', 'bb_', 'atr', 'momentum', 'roc', 'obv']
        )],
        'microstructure': [col for col in feature_cols if any(
            x in col for x in ['body', 'shadow', 'hammer', 'doji', 'vwap', 'volume_ratio', 'buy_ratio', 'order_flow']
        )],
        'cross_asset': [col for col in feature_cols if any(
            x in col for x in ['relative_strength', 'correlation']
        )],
        'dipmaster': [col for col in feature_cols if any(
            x in col for x in ['dipmaster', 'dip_zone', 'signal']
        )],
        'time': [col for col in feature_cols if any(
            x in col for x in ['hour', 'minute', 'day_of_week', 'boundary']
        )]
    }
    
    # Calculate category counts
    categorized = set()
    for cat_features in categories.values():
        categorized.update(cat_features)
    categories['other'] = [col for col in feature_cols if col not in categorized]
    
    # Label analysis
    label_analysis = {}
    if 'target_binary' in combined_df.columns:
        target_binary = combined_df['target_binary'].dropna()
        label_analysis['target_binary'] = {
            'positive_rate': float(target_binary.mean()),
            'samples': int(len(target_binary))
        }
    
    if 'dipmaster_win' in combined_df.columns:
        dipmaster_win = combined_df['dipmaster_win'].dropna()
        label_analysis['dipmaster_win'] = {
            'win_rate': float(dipmaster_win.mean()),
            'samples': int(len(dipmaster_win))
        }
    
    analysis = {
        'total_features': len(feature_cols),
        'total_labels': len(label_cols),
        'total_samples': len(combined_df),
        'symbols': combined_df['symbol'].nunique(),
        'feature_categories': {cat: len(features) for cat, features in categories.items()},
        'label_analysis': label_analysis,
        'time_range': {
            'start': str(combined_df['timestamp'].min()),
            'end': str(combined_df['timestamp'].max())
        },
        'memory_mb': float(combined_df.memory_usage(deep=True).sum() / 1024 / 1024)
    }
    
    return analysis

def main():
    """Main execution"""
    start_time = time.time()
    
    print("DipMaster Enhanced Feature Generation")
    print("=" * 50)
    
    try:
        # Load data
        print("Loading market data...")
        market_data = load_sample_data()
        if not market_data:
            print("ERROR: No market data loaded")
            return
        
        # Process each symbol
        print("Processing features...")
        processed_data = {}
        
        for symbol, df in market_data.items():
            print(f"Processing {symbol}...")
            
            # Add all features
            enhanced_df = df.copy()
            enhanced_df = add_technical_indicators(enhanced_df)
            enhanced_df = add_microstructure_features(enhanced_df)
            enhanced_df = add_dipmaster_signals(enhanced_df)
            enhanced_df = add_labels(enhanced_df)
            enhanced_df = clean_data(enhanced_df)
            
            processed_data[symbol] = enhanced_df
        
        # Add cross-asset features
        print("Adding cross-asset features...")
        processed_data = add_cross_asset_features(processed_data)
        
        # Combine all data
        print("Combining data...")
        combined_dfs = []
        for symbol, df in processed_data.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            combined_dfs.append(df_copy)
        
        combined_df = pd.concat(combined_dfs, ignore_index=True, sort=False)
        combined_df = combined_df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        
        # Create splits
        print("Creating data splits...")
        total_samples = len(combined_df)
        train_end = int(total_samples * 0.70)
        val_end = int(total_samples * 0.85)
        
        train_df = combined_df.iloc[:train_end].copy()
        val_df = combined_df.iloc[train_end:val_end].copy()
        test_df = combined_df.iloc[val_end:].copy()
        
        # Analyze features
        print("Analyzing features...")
        analysis = analyze_features(combined_df)
        
        # Create timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create FeatureSet config
        feature_set_config = {
            "metadata": {
                "version": "5.0.0-Enhanced",
                "strategy_name": "DipMaster_Enhanced_V5",
                "created_timestamp": timestamp,
                "description": "Comprehensive enhanced feature set for DipMaster strategy"
            },
            "symbols": list(combined_df['symbol'].unique()),
            "data_splits": {
                "train": {
                    "start_date": str(train_df['timestamp'].min()),
                    "end_date": str(train_df['timestamp'].max()),
                    "samples": len(train_df)
                },
                "validation": {
                    "start_date": str(val_df['timestamp'].min()),
                    "end_date": str(val_df['timestamp'].max()),
                    "samples": len(val_df)
                },
                "test": {
                    "start_date": str(test_df['timestamp'].min()),
                    "end_date": str(test_df['timestamp'].max()),
                    "samples": len(test_df)
                }
            },
            "feature_engineering": {
                "total_features": analysis['total_features'],
                "total_labels": analysis['total_labels'],
                "feature_categories": analysis['feature_categories'],
                "components": [
                    "comprehensive_technical_indicators",
                    "market_microstructure_features", 
                    "cross_asset_relative_strength",
                    "dipmaster_signal_engineering",
                    "optimized_15min_boundary_labels"
                ]
            },
            "quality_metrics": analysis,
            "target_specifications": {
                "primary_target": "target_return",
                "binary_target": "target_binary",
                "strategy_target": "dipmaster_win",
                "profit_targets": [0.003, 0.006, 0.012],
                "time_horizons": [3, 6, 12, 24, 36]
            },
            "files": {
                "feature_data": f"Enhanced_Features_V5_{timestamp}.parquet",
                "train_data": f"train_features_V5_{timestamp}.parquet",
                "validation_data": f"validation_features_V5_{timestamp}.parquet",
                "test_data": f"test_features_V5_{timestamp}.parquet"
            }
        }
        
        # Save files
        print("Saving results...")
        output_dir = Path("data")
        
        # Save config
        config_path = output_dir / f"Enhanced_FeatureSet_V5_{timestamp}.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(feature_set_config, f, indent=2, default=str, ensure_ascii=False)
        
        # Save data
        features_path = output_dir / f"Enhanced_Features_V5_{timestamp}.parquet"
        combined_df.to_parquet(features_path, compression='snappy', index=False)
        
        train_path = output_dir / f"train_features_V5_{timestamp}.parquet"
        train_df.to_parquet(train_path, compression='snappy', index=False)
        
        val_path = output_dir / f"validation_features_V5_{timestamp}.parquet"
        val_df.to_parquet(val_path, compression='snappy', index=False)
        
        test_path = output_dir / f"test_features_V5_{timestamp}.parquet"
        test_df.to_parquet(test_path, compression='snappy', index=False)
        
        # Results
        runtime = time.time() - start_time
        
        print("\n" + "=" * 50)
        print("RESULTS")
        print("=" * 50)
        print(f"Status: SUCCESS")
        print(f"Runtime: {runtime:.1f} seconds")
        print(f"Symbols: {analysis['symbols']}")
        print(f"Samples: {analysis['total_samples']:,}")
        print(f"Features: {analysis['total_features']}")
        print(f"Labels: {analysis['total_labels']}")
        
        print(f"\nFiles Created:")
        print(f"  Config: {config_path}")
        print(f"  Features: {features_path}")
        print(f"  Train: {train_path}")
        print(f"  Validation: {val_path}")
        print(f"  Test: {test_path}")
        
        print(f"\nFeature Categories:")
        for category, count in analysis['feature_categories'].items():
            print(f"  {category}: {count}")
        
        if 'label_analysis' in analysis:
            print(f"\nLabel Quality:")
            for label, metrics in analysis['label_analysis'].items():
                if 'win_rate' in metrics:
                    print(f"  {label}: {metrics['win_rate']:.1%} win rate")
                elif 'positive_rate' in metrics:
                    print(f"  {label}: {metrics['positive_rate']:.1%} positive rate")
        
        print(f"\nMemory Usage: {analysis['memory_mb']:.1f} MB")
        print("=" * 50)
        
    except Exception as e:
        print(f"ERROR: Pipeline failed - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()