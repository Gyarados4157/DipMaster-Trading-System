#!/usr/bin/env python3
"""
SuperDip Pin Bar Feature Engineering Demo
超跌接针策略特征工程演示

使用现有CSV数据演示完整的特征工程管道
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
import json
from datetime import datetime
import logging
import ta
from sklearn.preprocessing import RobustScaler
from typing import Dict, List, Tuple

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SuperDipPinBarDemo:
    """SuperDip Pin Bar Feature Engineering Demo"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_categories = {
            'core_technical': [],
            'pin_bar_pattern': [],
            'volume_profile': [],
            'momentum': [],
            'volatility': []
        }
        
    def load_csv_data(self, symbol: str, timeframe: str = '5m') -> pd.DataFrame:
        """Load CSV data"""
        try:
            data_file = f"data/market_data/{symbol}_{timeframe}_2years.csv"
            if not Path(data_file).exists():
                self.logger.warning(f"File not found: {data_file}")
                return pd.DataFrame()
            
            df = pd.read_csv(data_file)
            
            # Standardize column names
            column_mapping = {
                'timestamp': 'timestamp',
                'open_time': 'timestamp',
                'Open': 'open', 'open_price': 'open',
                'High': 'high', 'high_price': 'high',
                'Low': 'low', 'low_price': 'low',
                'Close': 'close', 'close_price': 'close',
                'Volume': 'volume', 'vol': 'volume'
            }
            
            df = df.rename(columns=column_mapping)
            
            # Convert timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"Required column '{col}' not found")
                    return pd.DataFrame()
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Basic validation
            df = df.dropna()
            invalid_ohlc = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close'])
            )
            
            if invalid_ohlc.any():
                self.logger.warning(f"Fixed {invalid_ohlc.sum()} invalid OHLC records")
                df = df[~invalid_ohlc]
            
            self.logger.info(f"Loaded {symbol} data: {len(df)} records")
            return df.sort_index()
            
        except Exception as e:
            self.logger.error(f"Error loading {symbol} data: {e}")
            return pd.DataFrame()
    
    def calculate_rsi_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate RSI features for oversold detection"""
        features = df.copy()
        
        try:
            # RSI indicators for different periods
            for period in [7, 14, 21]:
                rsi_col = f'{symbol}_rsi_{period}'
                features[rsi_col] = ta.momentum.RSIIndicator(features['close'], window=period).rsi()
                self.feature_categories['core_technical'].append(rsi_col)
                
                # RSI oversold conditions
                features[f'{symbol}_rsi_{period}_oversold'] = (features[rsi_col] < 30).astype(int)
                features[f'{symbol}_rsi_{period}_optimal'] = ((features[rsi_col] >= 30) & (features[rsi_col] <= 50)).astype(int)
                
                self.feature_categories['core_technical'].extend([
                    f'{symbol}_rsi_{period}_oversold',
                    f'{symbol}_rsi_{period}_optimal'
                ])
            
            self.logger.info(f"Calculated RSI features for {symbol}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI features: {e}")
            return features
    
    def calculate_moving_average_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate MA deviation features"""
        features = df.copy()
        
        try:
            # Moving averages
            for period in [10, 20, 50]:
                ma_col = f'{symbol}_ma_{period}'
                features[ma_col] = ta.trend.SMAIndicator(features['close'], window=period).sma_indicator()
                
                # Price deviation from MA
                deviation_col = f'{symbol}_price_ma_{period}_deviation'
                features[deviation_col] = (features['close'] - features[ma_col]) / features[ma_col]
                
                # Price position relative to MA
                position_col = f'{symbol}_price_below_ma_{period}'
                features[position_col] = (features['close'] < features[ma_col]).astype(int)
                
                self.feature_categories['core_technical'].extend([ma_col, deviation_col, position_col])
            
            self.logger.info(f"Calculated MA features for {symbol}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating MA features: {e}")
            return features
    
    def calculate_bollinger_bands_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate Bollinger Bands features for oversold detection"""
        features = df.copy()
        
        try:
            # Bollinger Bands
            bb_indicator = ta.volatility.BollingerBands(features['close'], window=20, window_dev=2)
            
            features[f'{symbol}_bb_lower'] = bb_indicator.bollinger_lband()
            features[f'{symbol}_bb_upper'] = bb_indicator.bollinger_hband()
            features[f'{symbol}_bb_middle'] = bb_indicator.bollinger_mavg()
            
            # Bollinger Bands position (0 = lower band, 1 = upper band)
            bb_width = features[f'{symbol}_bb_upper'] - features[f'{symbol}_bb_lower']
            features[f'{symbol}_bb_position'] = (features['close'] - features[f'{symbol}_bb_lower']) / bb_width
            
            # Oversold condition (below lower band)
            features[f'{symbol}_bb_oversold'] = (features['close'] < features[f'{symbol}_bb_lower']).astype(int)
            
            # BB width (volatility measure)
            features[f'{symbol}_bb_width'] = bb_width / features[f'{symbol}_bb_middle']
            
            self.feature_categories['core_technical'].extend([
                f'{symbol}_bb_lower', f'{symbol}_bb_upper', f'{symbol}_bb_middle',
                f'{symbol}_bb_position', f'{symbol}_bb_oversold', f'{symbol}_bb_width'
            ])
            
            self.logger.info(f"Calculated Bollinger Bands features for {symbol}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands features: {e}")
            return features
    
    def calculate_volume_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate volume-based features"""
        features = df.copy()
        
        try:
            # Volume moving averages
            for period in [10, 20, 50]:
                vol_ma_col = f'{symbol}_volume_ma_{period}'
                features[vol_ma_col] = features['volume'].rolling(window=period).mean()
                
                # Volume ratio (relative strength)
                vol_ratio_col = f'{symbol}_volume_ratio_{period}'
                features[vol_ratio_col] = features['volume'] / features[vol_ma_col]
                
                # Volume spike detection
                features[f'{symbol}_volume_spike_{period}'] = (features[vol_ratio_col] > 2.0).astype(int)
                
                self.feature_categories['volume_profile'].extend([
                    vol_ma_col, vol_ratio_col, f'{symbol}_volume_spike_{period}'
                ])
            
            # Volume-Price Trend (VPT)
            features[f'{symbol}_vpt'] = ta.volume.VolumePriceTrendIndicator(features['close'], features['volume']).volume_price_trend()
            self.feature_categories['volume_profile'].append(f'{symbol}_vpt')
            
            self.logger.info(f"Calculated volume features for {symbol}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating volume features: {e}")
            return features
    
    def calculate_pin_bar_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate pin bar pattern features"""
        features = df.copy()
        
        try:
            # Basic candlestick calculations
            candle_range = features['high'] - features['low']
            body_size = abs(features['close'] - features['open'])
            
            # Avoid division by zero
            valid_range = candle_range > 0
            
            # Lower wick (shadow) calculations
            lower_wick = np.minimum(features['open'], features['close']) - features['low']
            upper_wick = features['high'] - np.maximum(features['open'], features['close'])
            
            # Pin bar ratios
            features[f'{symbol}_lower_wick_ratio'] = np.where(valid_range, lower_wick / candle_range, 0)
            features[f'{symbol}_upper_wick_ratio'] = np.where(valid_range, upper_wick / candle_range, 0)
            features[f'{symbol}_body_ratio'] = np.where(valid_range, body_size / candle_range, 0)
            
            # Body position in the candle (0 = bottom, 1 = top)
            features[f'{symbol}_body_position'] = np.where(
                valid_range, 
                (np.minimum(features['open'], features['close']) - features['low']) / candle_range,
                0
            )
            
            # Lower wick to body ratio (key pin bar indicator)
            features[f'{symbol}_lower_to_body_ratio'] = np.where(
                (valid_range) & (body_size > 0),
                lower_wick / body_size,
                0
            )
            
            # Price recovery (how much price recovered from low)
            features[f'{symbol}_price_recovery'] = np.where(
                valid_range,
                (features['close'] - features['low']) / candle_range,
                0
            )
            
            # Pin bar identification
            pin_bar_condition = (
                (features[f'{symbol}_lower_wick_ratio'] > 0.5) &  # Long lower wick
                (features[f'{symbol}_body_ratio'] < 0.3) &  # Small body
                (features[f'{symbol}_upper_wick_ratio'] < 0.2) &  # Short upper wick
                (features[f'{symbol}_lower_to_body_ratio'] > 2.0)  # Lower wick >> body
            )
            
            features[f'{symbol}_is_pin_bar'] = pin_bar_condition.astype(int)
            
            # Enhanced pin bar (with volume confirmation)
            volume_ma_20 = features['volume'].rolling(window=20).mean()
            volume_multiplier = features['volume'] / volume_ma_20
            
            enhanced_pin_bar = (
                pin_bar_condition &
                (volume_multiplier > 1.2)  # Volume confirmation
            )
            features[f'{symbol}_is_enhanced_pin_bar'] = enhanced_pin_bar.astype(int)
            
            # Pin bar strength score (0-1)
            pin_bar_strength = (
                features[f'{symbol}_lower_wick_ratio'] * 0.4 +
                (1 - features[f'{symbol}_body_ratio']) * 0.3 +
                (1 - features[f'{symbol}_upper_wick_ratio']) * 0.2 +
                np.clip(volume_multiplier / 3, 0, 1) * 0.1
            )
            features[f'{symbol}_pin_bar_strength'] = pin_bar_strength
            
            pin_bar_features = [
                f'{symbol}_lower_wick_ratio', f'{symbol}_upper_wick_ratio', f'{symbol}_body_ratio',
                f'{symbol}_body_position', f'{symbol}_lower_to_body_ratio', f'{symbol}_price_recovery',
                f'{symbol}_is_pin_bar', f'{symbol}_is_enhanced_pin_bar', f'{symbol}_pin_bar_strength'
            ]
            self.feature_categories['pin_bar_pattern'].extend(pin_bar_features)
            
            self.logger.info(f"Calculated pin bar features for {symbol}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating pin bar features: {e}")
            return features
    
    def calculate_momentum_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate momentum features"""
        features = df.copy()
        
        try:
            # Price momentum over different periods
            for period in [3, 5, 10, 15]:
                momentum_col = f'{symbol}_momentum_{period}'
                features[momentum_col] = features['close'].pct_change(periods=period)
                self.feature_categories['momentum'].append(momentum_col)
            
            # Rate of change
            features[f'{symbol}_roc_10'] = ta.momentum.ROCIndicator(features['close'], window=10).roc()
            self.feature_categories['momentum'].append(f'{symbol}_roc_10')
            
            # MACD
            macd = ta.trend.MACD(features['close'])
            features[f'{symbol}_macd'] = macd.macd()
            features[f'{symbol}_macd_signal'] = macd.macd_signal()
            features[f'{symbol}_macd_histogram'] = macd.macd_diff()
            
            self.feature_categories['momentum'].extend([
                f'{symbol}_macd', f'{symbol}_macd_signal', f'{symbol}_macd_histogram'
            ])
            
            self.logger.info(f"Calculated momentum features for {symbol}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum features: {e}")
            return features
    
    def calculate_volatility_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Calculate volatility features"""
        features = df.copy()
        
        try:
            # Rolling volatility (coefficient of variation)
            for period in [10, 20, 30]:
                vol_col = f'{symbol}_volatility_{period}'
                rolling_std = features['close'].rolling(window=period).std()
                rolling_mean = features['close'].rolling(window=period).mean()
                features[vol_col] = rolling_std / rolling_mean
                self.feature_categories['volatility'].append(vol_col)
            
            # Average True Range (ATR)
            features[f'{symbol}_atr_14'] = ta.volatility.AverageTrueRange(
                features['high'], features['low'], features['close'], window=14
            ).average_true_range()
            self.feature_categories['volatility'].append(f'{symbol}_atr_14')
            
            self.logger.info(f"Calculated volatility features for {symbol}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility features: {e}")
            return features
    
    def generate_labels(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate forward-looking labels"""
        features = df.copy()
        
        try:
            # 4-hour forward return
            horizon = 48  # 48 * 5m = 4 hours
            forward_return_col = f'{symbol}_forward_return_4h'
            features[forward_return_col] = features['close'].shift(-horizon) / features['close'] - 1
            
            # Win/loss labels for different profit targets
            profit_targets = [0.008, 0.015, 0.025]  # 0.8%, 1.5%, 2.5%
            stop_loss = 0.006  # 0.6%
            
            for target in profit_targets:
                label_col = f'{symbol}_win_{int(target*1000)}bp_4h'
                
                # Calculate maximum favorable excursion (MFE) over horizon
                future_highs = []
                future_lows = []
                
                for i in range(len(features)):
                    end_idx = min(i + horizon, len(features))
                    if end_idx > i:
                        future_high = features['high'].iloc[i+1:end_idx+1].max()
                        future_low = features['low'].iloc[i+1:end_idx+1].min()
                        future_highs.append(future_high)
                        future_lows.append(future_low)
                    else:
                        future_highs.append(np.nan)
                        future_lows.append(np.nan)
                
                future_highs = pd.Series(future_highs, index=features.index)
                future_lows = pd.Series(future_lows, index=features.index)
                
                # Profit achieved condition
                profit_achieved = (future_highs / features['close'] - 1) >= target
                
                # Stop loss hit condition
                stop_loss_hit = (features['close'] - future_lows) / features['close'] >= stop_loss
                
                # Label: 1 if profit target reached without hitting stop loss
                features[label_col] = (profit_achieved & ~stop_loss_hit).astype(int)
            
            # Risk-adjusted return
            volatility_20 = features['close'].rolling(window=20).std()
            features[f'{symbol}_risk_adj_return_4h'] = features[forward_return_col] / volatility_20
            
            # Maximum Favorable/Adverse Excursion
            features[f'{symbol}_mfe_4h'] = (future_highs / features['close'] - 1)
            features[f'{symbol}_mae_4h'] = (features['close'] / future_lows - 1)
            
            self.logger.info(f"Generated labels for {symbol}")
            return features
            
        except Exception as e:
            self.logger.error(f"Error generating labels: {e}")
            return features
    
    def perform_quality_checks(self, df: pd.DataFrame) -> Dict:
        """Perform basic quality checks"""
        quality_report = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'infinite_values': (np.isinf(df.select_dtypes(include=[np.number]).values).sum()),
            'data_range': {
                'start_date': str(df.index.min()) if not df.empty else None,
                'end_date': str(df.index.max()) if not df.empty else None
            }
        }
        
        return quality_report
    
    def run_demo(self, symbols: List[str] = None) -> Dict:
        """Run the complete demo"""
        if symbols is None:
            symbols = ['ADAUSDT', 'SOLUSDT', 'BNBUSDT', 'DOGEUSDT', 'ICPUSDT']
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {}
        
        for symbol in symbols:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Processing {symbol}")
            self.logger.info(f"{'='*50}")
            
            # Load data
            df = self.load_csv_data(symbol)
            if df.empty:
                self.logger.warning(f"No data loaded for {symbol}, skipping...")
                continue
            
            # Calculate all features
            df = self.calculate_rsi_features(df, symbol)
            df = self.calculate_moving_average_features(df, symbol)
            df = self.calculate_bollinger_bands_features(df, symbol)
            df = self.calculate_volume_features(df, symbol)
            df = self.calculate_pin_bar_features(df, symbol)
            df = self.calculate_momentum_features(df, symbol)
            df = self.calculate_volatility_features(df, symbol)
            df = self.generate_labels(df, symbol)
            
            # Clean data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Quality check
            quality_report = self.perform_quality_checks(df)
            
            # Save features
            output_file = f"data/features_{symbol}_superdip_pinbar_{timestamp}.parquet"
            df.to_parquet(output_file, compression='zstd')
            
            results[symbol] = {
                'feature_file': output_file,
                'feature_count': len(df.columns),
                'record_count': len(df),
                'quality_report': quality_report
            }
            
            self.logger.info(f"Saved {len(df.columns)} features for {symbol} ({len(df)} records)")
        
        # Generate summary report
        feature_set_config = {
            "version": datetime.now().isoformat(),
            "feature_set_id": f"superdip_pinbar_demo_{timestamp}",
            "strategy_name": "SuperDip_PinBar_Strategy_Demo",
            "description": "超跌接针反转策略特征集演示版 - 包含核心技术指标、接针形态识别、成交量分析和前向标签",
            
            "metadata": {
                "creation_date": datetime.now().isoformat(),
                "feature_engineer": "SuperDipPinBarDemo",
                "version": "1.0.0-demo",
                "total_symbols": len(results),
                "symbols_processed": list(results.keys())
            },
            
            "feature_categories": self.feature_categories,
            
            "core_features": {
                "rsi_features": "RSI(7,14,21) 用于超跌识别",
                "ma_deviation": "MA(10,20,50) 价格偏离度计算",
                "bollinger_bands": "布林带位置和超跌区域判断",
                "volume_analysis": "成交量相对强度和放大倍数",
                "pin_bar_detection": "接针形态识别和强度评分",
                "momentum_indicators": "价格动量和MACD确认",
                "volatility_measures": "波动率和ATR计算"
            },
            
            "target_definitions": {
                "forward_returns": "4小时前向收益率",
                "profit_targets": "0.8%, 1.5%, 2.5% 利润目标达成标签",
                "risk_metrics": "风险调整收益和最大不利偏移"
            },
            
            "data_files": results,
            
            "usage_recommendations": {
                "primary_use": "超跌接针反转策略信号生成",
                "model_types": ["LightGBM", "XGBoost", "Random Forest"],
                "validation_method": "时序交叉验证",
                "feature_selection": "基于特征重要性筛选",
                "risk_management": "结合止损和仓位管理"
            }
        }
        
        # Save configuration
        config_file = f"data/FeatureSet_SuperDip_PinBar_Demo_{timestamp}.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(feature_set_config, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"\n{'='*60}")
        self.logger.info("SuperDip Pin Bar Feature Engineering Demo Completed!")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Processed symbols: {list(results.keys())}")
        self.logger.info(f"Total feature files created: {len(results)}")
        self.logger.info(f"Feature set configuration: {config_file}")
        self.logger.info(f"{'='*60}")
        
        return {
            "feature_set_config": feature_set_config,
            "config_file": config_file,
            "results": results
        }

def main():
    """Main execution"""
    demo = SuperDipPinBarDemo()
    result = demo.run_demo()
    
    # Print summary
    print("\n" + "="*60)
    print("SUPERDIP PIN BAR FEATURE ENGINEERING SUMMARY")
    print("="*60)
    
    config = result['feature_set_config']
    print(f"Strategy: {config['strategy_name']}")
    print(f"Feature Set ID: {config['feature_set_id']}")
    print(f"Symbols Processed: {', '.join(config['metadata']['symbols_processed'])}")
    print(f"Configuration File: {result['config_file']}")
    
    print("\nFeature Categories:")
    for category, features in config['feature_categories'].items():
        if features:  # Only show categories with features
            print(f"  {category}: {len(features)} features")
    
    print("\nData Files Created:")
    for symbol, data in result['results'].items():
        print(f"  {symbol}: {data['feature_count']} features, {data['record_count']} records")
        print(f"    File: {data['feature_file']}")
    
    print("="*60)

if __name__ == "__main__":
    main()