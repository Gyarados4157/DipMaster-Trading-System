#!/usr/bin/env python3
"""
Final Top30 Enhanced Feature Engineering
最终30币种增强特征工程系统
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import warnings
import time
import glob
import ta

warnings.filterwarnings('ignore')

def main():
    print('Final Top30 Enhanced Feature Engineering System')
    print('=' * 60)
    
    start_time = time.time()
    
    # Load all data
    print('Loading 30 symbol data...')
    data_files = glob.glob('data/enhanced_market_data/*_5m_90days.parquet')
    symbols = sorted(list(set([Path(f).name.split('_')[0] for f in data_files])))
    
    all_data = {}
    for symbol in symbols:
        file_path = f'data/enhanced_market_data/{symbol}_5m_90days.parquet'
        if Path(file_path).exists():
            df = pd.read_parquet(file_path)
            df = df.reset_index()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            if len(df) > 1000:
                all_data[symbol] = df
                print(f'  Loaded {symbol}: {len(df)} rows')
    
    print(f'Successfully loaded {len(all_data)} symbols')
    
    # Symbol categorization
    volatility_groups = {
        'low_vol': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT'],
        'medium_vol': ['SOLUSDT', 'ADAUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT'],
        'high_vol': ['SHIBUSDT', 'DOGEUSDT', 'TONUSDT', 'PEPEUSDT', 'INJUSDT']
    }
    
    symbol_categories = {
        'layer1': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'AVAXUSDT'],
        'defi': ['UNIUSDT', 'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'LINKUSDT'],
        'meme': ['SHIBUSDT', 'DOGEUSDT', 'PEPEUSDT'],
        'exchange': ['BNBUSDT'],
        'payments': ['XRPUSDT', 'LTCUSDT', 'TRXUSDT']
    }
    
    def get_volatility_group(symbol):
        for group, symbols_list in volatility_groups.items():
            if symbol in symbols_list:
                return group
        return 'medium_vol'
    
    def get_category(symbol):
        for category, symbols_list in symbol_categories.items():
            if symbol in symbols_list:
                return category
        return 'other'
    
    # Step 1: Add comprehensive features for each symbol
    print('Step 1: Adding comprehensive features...')
    for symbol in all_data:
        try:
            df = all_data[symbol].copy()
            
            # Get symbol characteristics
            vol_group = get_volatility_group(symbol)
            category = get_category(symbol)
            
            # Adaptive parameters
            if vol_group == 'low_vol':
                rsi_period, ma_period, bb_period = 21, 20, 20
                vol_mult = 0.8
            elif vol_group == 'medium_vol':
                rsi_period, ma_period, bb_period = 14, 16, 16
                vol_mult = 1.0
            else:
                rsi_period, ma_period, bb_period = 10, 12, 12
                vol_mult = 1.3
            
            # Technical indicators
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=rsi_period).rsi()
            df['rsi_oversold'] = (df['rsi'] <= 30 / vol_mult).astype(int)
            df['rsi_overbought'] = (df['rsi'] >= 70 * vol_mult).astype(int)
            
            # Moving averages
            df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=ma_period).sma_indicator()
            df['ema_20'] = ta.trend.EMAIndicator(df['close'], window=ma_period).ema_indicator()
            df['price_vs_sma'] = (df['close'] - df['sma_20']) / df['sma_20']
            df['ma_trend'] = (df['sma_20'] > df['sma_20'].shift(1)).astype(int)
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(df['close'], window=bb_period, window_dev=2)
            df['bb_position'] = bb.bollinger_pband()
            df['bb_width'] = bb.bollinger_wband()
            df['bb_squeeze'] = (df['bb_width'] < df['bb_width'].rolling(50).quantile(0.2)).astype(int)
            
            # MACD
            macd = ta.trend.MACD(df['close'])
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_cross'] = (df['macd'] > df['macd_signal']).astype(int)
            
            # Volume analysis
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
            
            # OBV
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['obv_trend'] = (df['obv'] > df['obv'].shift(1)).astype(int)
            
            # Volatility
            returns = df['close'].pct_change()
            df['volatility_20'] = returns.rolling(20).std() * np.sqrt(20)
            df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
            df['atr_ratio'] = df['atr'] / df['close']
            
            # Support/Resistance
            df['resistance_20'] = df['high'].rolling(20).max()
            df['support_20'] = df['low'].rolling(20).min()
            df['resistance_distance'] = (df['resistance_20'] - df['close']) / df['close']
            df['support_distance'] = (df['close'] - df['support_20']) / df['close']
            
            # Microstructure features
            df['price_range'] = df['high'] - df['low']
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            
            df['body_ratio'] = df['body_size'] / (df['price_range'] + 1e-8)
            df['upper_shadow_ratio'] = df['upper_shadow'] / (df['price_range'] + 1e-8)
            df['lower_shadow_ratio'] = df['lower_shadow'] / (df['price_range'] + 1e-8)
            
            # Patterns
            df['hammer'] = ((df['lower_shadow_ratio'] > 0.6) & (df['body_ratio'] < 0.3)).astype(int)
            df['doji'] = (df['body_ratio'] < 0.1).astype(int)
            df['shooting_star'] = ((df['upper_shadow_ratio'] > 0.6) & (df['body_ratio'] < 0.3)).astype(int)
            
            # VWAP
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap_20 = (typical_price * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
            df['vwap_20'] = vwap_20
            df['vwap_deviation'] = (df['close'] - vwap_20) / vwap_20
            
            # Category features
            for cat, symbols_list in symbol_categories.items():
                df[f'is_{cat}'] = 1 if symbol in symbols_list else 0
            
            # Time features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            all_data[symbol] = df
            print(f'  Enhanced {symbol}: {len(df.columns)} features')
            
        except Exception as e:
            print(f'  Error {symbol}: {e}')
    
    # Step 2: Cross-symbol features
    print('Step 2: Adding cross-symbol features...')
    try:
        # Prepare aligned data
        price_data = {}
        for symbol, df in all_data.items():
            price_data[symbol] = df.set_index('timestamp')['close']
        
        price_df = pd.DataFrame(price_data).ffill().bfill()
        returns_df = price_df.pct_change()
        
        # Market metrics
        market_return = returns_df.mean(axis=1)
        
        # Add cross-symbol features
        for symbol in all_data:
            if symbol not in returns_df.columns:
                continue
                
            df = all_data[symbol].copy()
            df_indexed = df.set_index('timestamp')
            symbol_returns = returns_df[symbol]
            
            # Market relative performance
            relative_perf = symbol_returns - market_return
            df_indexed['market_relative_1h'] = relative_perf.rolling(12).mean()
            df_indexed['market_relative_4h'] = relative_perf.rolling(48).mean()
            
            # Market ranking
            returns_1h = returns_df.rolling(12).sum()
            df_indexed['market_rank_1h'] = returns_1h.rank(axis=1, pct=True)[symbol]
            
            # BTC correlation
            if 'BTCUSDT' in returns_df.columns and symbol != 'BTCUSDT':
                df_indexed['btc_correlation'] = symbol_returns.rolling(144).corr(returns_df['BTCUSDT'])
            else:
                df_indexed['btc_correlation'] = 0.5
            
            # Reset and merge
            df_indexed = df_indexed.reset_index()
            merge_cols = ['timestamp'] + [col for col in df_indexed.columns if col not in df.columns]
            all_data[symbol] = df.merge(df_indexed[merge_cols], on='timestamp', how='left')
        
        print('  Cross-symbol features completed')
        
    except Exception as e:
        print(f'  Cross-symbol features failed: {e}')
    
    # Step 3: Dynamic labels
    print('Step 3: Adding dynamic labels...')
    for symbol in all_data:
        try:
            df = all_data[symbol].copy()
            vol_group = get_volatility_group(symbol)
            vol_mult = {'low_vol': 0.8, 'medium_vol': 1.0, 'high_vol': 1.5}[vol_group]
            
            # Multi-horizon returns
            for horizon in [3, 6, 12, 24]:
                future_return = df['close'].pct_change(periods=horizon).shift(-horizon)
                df[f'future_return_{horizon}p'] = future_return
                
                # Targets
                targets = [0.003, 0.006, 0.008, 0.012]
                for i, target in enumerate(targets):
                    adjusted_target = target * vol_mult
                    df[f'hits_target_{i}_{horizon}p'] = (future_return >= adjusted_target).astype(int)
            
            # 15-minute boundary
            df['minute'] = df['timestamp'].dt.minute
            boundary_minutes = [15, 30, 45, 0]
            
            for horizon in [3, 6, 12]:
                future_minute = df['minute'].shift(-horizon)
                is_boundary = future_minute.isin(boundary_minutes)
                df[f'boundary_exit_{horizon}p'] = is_boundary.astype(int)
            
            # Main targets
            main_return = df['future_return_12p']
            df['target_return'] = main_return
            df['target_binary'] = (main_return > 0).astype(int)
            df['target_profitable'] = (main_return >= 0.006 * vol_mult).astype(int)
            
            # Multi-class
            conditions = [
                (main_return <= -0.004 * vol_mult),
                ((main_return > -0.004 * vol_mult) & (main_return <= 0)),
                ((main_return > 0) & (main_return < 0.003 * vol_mult)),
                ((main_return >= 0.003 * vol_mult) & (main_return < 0.008 * vol_mult)),
                ((main_return >= 0.008 * vol_mult) & (main_return < 0.015 * vol_mult)),
                (main_return >= 0.015 * vol_mult)
            ]
            labels = [0, 1, 2, 3, 4, 5]
            df['return_class'] = np.select(conditions, labels, default=1)
            
            all_data[symbol] = df
            print(f'  Labels added for {symbol}')
            
        except Exception as e:
            print(f'  Label error {symbol}: {e}')
    
    # Step 4: Clean data
    print('Step 4: Cleaning data...')
    for symbol in all_data:
        df = all_data[symbol]
        
        # Replace inf
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Handle NaN
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'symbol'] 
                       and not col.startswith('target')
                       and not col.startswith('future_')]
        
        for col in feature_cols:
            if col in df.columns:
                null_pct = df[col].isnull().sum() / len(df)
                if null_pct > 0.5:
                    df[col] = df[col].fillna(0)
                elif null_pct > 0.1:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(method='ffill').fillna(0)
        
        # Outlier clipping
        for col in feature_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                Q1, Q3 = df[col].quantile([0.01, 0.99])
                df[col] = df[col].clip(Q1, Q3)
        
        all_data[symbol] = df
        print(f'  Cleaned {symbol}')
    
    # Step 5: Combine and save
    print('Step 5: Combining and saving...')
    combined_data = []
    for symbol, df in all_data.items():
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        combined_data.append(df_copy)
    
    if combined_data:
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'data/Enhanced_Features_Top30_V6_{timestamp}.parquet'
        combined_df.to_parquet(output_file, index=False)
        
        # Metadata
        feature_cols = [col for col in combined_df.columns if col not in ['timestamp', 'symbol']]
        
        metadata = {
            'file_info': {
                'filename': Path(output_file).name,
                'creation_date': datetime.now().isoformat(),
                'total_rows': len(combined_df),
                'total_symbols': len(all_data),
                'total_features': len(feature_cols),
                'processing_time_seconds': time.time() - start_time
            },
            'symbols': list(all_data.keys()),
            'feature_categories': {
                'technical': len([col for col in feature_cols if any(x in col for x in ['rsi', 'sma', 'ema', 'bb_', 'macd', 'atr'])]),
                'volume': len([col for col in feature_cols if 'volume' in col or 'obv' in col]),
                'cross_symbol': len([col for col in feature_cols if any(x in col for x in ['market_', 'correlation', 'rank'])]),
                'microstructure': len([col for col in feature_cols if any(x in col for x in ['vwap', 'shadow', 'body', 'pattern', 'hammer', 'doji'])]),
                'labels': len([col for col in feature_cols if any(x in col for x in ['target', 'future_', 'hits_', 'return_class'])]),
                'category': len([col for col in feature_cols if col.startswith('is_')]),
                'time': len([col for col in feature_cols if any(x in col for x in ['hour', 'weekend', 'boundary'])])
            }
        }
        
        metadata_file = f'data/FeatureSet_Top30_V6_{timestamp}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Feature importance analysis
        feature_analysis = analyze_features(combined_df, feature_cols)
        analysis_file = f'data/Feature_Importance_Analysis_Top30_{timestamp}.json'
        with open(analysis_file, 'w') as f:
            json.dump(feature_analysis, f, indent=2, default=str)
        
        total_time = time.time() - start_time
        
        print(f'\\nFinal Top30 Enhanced Feature Engineering Completed!')
        print(f'Output file: {output_file}')
        print(f'Metadata: {metadata_file}')
        print(f'Analysis: {analysis_file}')
        print(f'Total rows: {len(combined_df):,}')
        print(f'Total features: {len(feature_cols)}')
        print(f'Symbols: {len(all_data)}')
        print(f'Processing time: {total_time:.1f}s')
        
        print(f'\\nFeature Categories:')
        for category, count in metadata['feature_categories'].items():
            if count > 0:
                print(f'  - {category}: {count}')
        
        print(f'\\nTop30 Enhanced Feature Engineering completed successfully!')
        
        return {
            'output_file': output_file,
            'metadata_file': metadata_file,
            'analysis_file': analysis_file,
            'total_rows': len(combined_df),
            'total_features': len(feature_cols),
            'symbols_count': len(all_data)
        }
        
    else:
        print('No data to save')
        return None

def analyze_features(df, feature_cols):
    """Analyze feature importance and quality"""
    analysis = {
        'total_features': len(feature_cols),
        'feature_statistics': {},
        'high_correlation_pairs': [],
        'feature_stability': {},
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    try:
        # Basic statistics for each feature
        for col in feature_cols[:50]:  # Limit to first 50 for performance
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                stats_dict = {
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else 0,
                    'std': float(df[col].std()) if not pd.isna(df[col].std()) else 0,
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else 0,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else 0,
                    'null_pct': float(df[col].isnull().sum() / len(df)),
                    'unique_values': int(df[col].nunique())
                }
                analysis['feature_statistics'][col] = stats_dict
        
        # Sample correlation analysis
        sample_size = min(10000, len(df))
        sample_df = df[feature_cols[:30]].sample(n=sample_size, random_state=42) if len(df) > sample_size else df[feature_cols[:30]]
        
        if len(sample_df.columns) > 1:
            corr_matrix = sample_df.corr()
            
            # Find high correlation pairs
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8 and not pd.isna(corr_val):
                        analysis['high_correlation_pairs'].append({
                            'feature_1': corr_matrix.columns[i],
                            'feature_2': corr_matrix.columns[j],
                            'correlation': float(corr_val)
                        })
            
            # Sort by correlation strength
            analysis['high_correlation_pairs'].sort(key=lambda x: abs(x['correlation']), reverse=True)
            analysis['high_correlation_pairs'] = analysis['high_correlation_pairs'][:10]  # Top 10
        
    except Exception as e:
        analysis['error'] = str(e)
    
    return analysis

if __name__ == "__main__":
    result = main()
    if result:
        print(f"\\nFinal Results:")
        print(f"  File: {result['output_file']}")
        print(f"  Features: {result['total_features']}")
        print(f"  Symbols: {result['symbols_count']}")
        print(f"  Rows: {result['total_rows']:,}")