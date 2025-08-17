#!/usr/bin/env python3
"""
Regime-Aware Feature Engineering Module for DipMaster Strategy
市场体制感知特征工程模块

This module extends the existing feature engineering pipeline with market regime
detection capabilities. It addresses the core issue of DipMaster's poor performance
in trending markets by generating regime-specific features and adaptive labels.

Key Features:
1. Market regime detection integration
2. Regime-specific feature generation
3. Adaptive labeling based on market conditions
4. Cross-regime feature importance analysis
5. Enhanced technical indicators for regime classification

Author: Strategy Orchestrator
Date: 2025-08-17
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import ta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, SelectKBest
import scipy.stats as stats
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Import our regime detector
from ..core.market_regime_detector import MarketRegimeDetector, MarketRegime, RegimeSignal

warnings.filterwarnings('ignore')

@dataclass
class RegimeFeatureConfig:
    """Configuration for regime-aware feature engineering"""
    symbols: List[str]
    timeframes: List[str] = None
    lookback_periods: Dict[str, int] = None
    regime_windows: Dict[str, int] = None
    feature_selection_k: int = 50
    enable_regime_interactions: bool = True
    enable_adaptive_labels: bool = True
    enable_cross_regime_features: bool = True

class RegimeAwareFeatureEngineer:
    """
    Regime-Aware Feature Engineering Pipeline
    市场体制感知特征工程管道
    
    Generates features that adapt to different market regimes to improve
    DipMaster strategy performance from 47.7% to 65%+ win rate.
    """
    
    def __init__(self, config: Optional[RegimeFeatureConfig] = None):
        """Initialize the regime-aware feature engineer"""
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.regime_detector = MarketRegimeDetector()
        self.scaler = RobustScaler()
        self.feature_selector = SelectKBest(mutual_info_regression, k=self.config.feature_selection_k)
        
        # Feature tracking
        self.feature_names = []
        self.regime_features = {}
        self.feature_importance = {}
        self.regime_transitions = {}
        
    def _get_default_config(self) -> RegimeFeatureConfig:
        """Get default configuration"""
        return RegimeFeatureConfig(
            symbols=[
                'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT',
                'BNBUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT'
            ],
            timeframes=['5m', '15m', '1h'],
            lookback_periods={
                'short': 24,    # 2 hours (5min bars)
                'medium': 96,   # 8 hours  
                'long': 288     # 24 hours
            },
            regime_windows={
                'trend_analysis': 48,
                'volatility_analysis': 96,
                'momentum_analysis': 24
            }
        )
    
    def generate_base_technical_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate comprehensive base technical indicators"""
        features_df = df.copy()
        
        # Price-based features
        features_df['returns'] = features_df['close'].pct_change()
        features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
        features_df['price_momentum_5'] = features_df['close'] / features_df['close'].shift(5) - 1
        features_df['price_momentum_20'] = features_df['close'] / features_df['close'].shift(20) - 1
        
        # Volatility features
        features_df['volatility_5'] = features_df['returns'].rolling(5).std()
        features_df['volatility_20'] = features_df['returns'].rolling(20).std()
        features_df['volatility_ratio'] = features_df['volatility_5'] / features_df['volatility_20']
        
        # RSI multiple timeframes
        for period in [7, 14, 21, 30]:
            rsi = ta.momentum.RSIIndicator(features_df['close'], window=period)
            features_df[f'rsi_{period}'] = rsi.rsi()
        
        # MACD
        macd = ta.trend.MACD(features_df['close'])
        features_df['macd'] = macd.macd()
        features_df['macd_signal'] = macd.macd_signal()
        features_df['macd_histogram'] = macd.macd_diff()
        
        # Bollinger Bands
        for window in [10, 20, 30]:
            bb = ta.volatility.BollingerBands(features_df['close'], window=window)
            features_df[f'bb_upper_{window}'] = bb.bollinger_hband()
            features_df[f'bb_lower_{window}'] = bb.bollinger_lband()
            features_df[f'bb_middle_{window}'] = bb.bollinger_mavg()
            features_df[f'bb_width_{window}'] = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
            features_df[f'bb_position_{window}'] = (features_df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100]:
            if len(features_df) >= period:
                features_df[f'ma_{period}'] = features_df['close'].rolling(period).mean()
                features_df[f'ma_ratio_{period}'] = features_df['close'] / features_df[f'ma_{period}']
        
        # EMA
        for period in [9, 12, 21, 26]:
            features_df[f'ema_{period}'] = features_df['close'].ewm(span=period).mean()
            features_df[f'ema_ratio_{period}'] = features_df['close'] / features_df[f'ema_{period}']
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(features_df['high'], features_df['low'], features_df['close'])
        features_df['stoch_k'] = stoch.stoch()
        features_df['stoch_d'] = stoch.stoch_signal()
        
        # Williams %R
        williams = ta.momentum.WilliamsRIndicator(features_df['high'], features_df['low'], features_df['close'])
        features_df['williams_r'] = williams.williams_r()
        
        # ADX
        adx = ta.trend.ADXIndicator(features_df['high'], features_df['low'], features_df['close'])
        features_df['adx'] = adx.adx()
        features_df['adx_di_plus'] = adx.adx_pos()
        features_df['adx_di_minus'] = adx.adx_neg()
        
        # Volume indicators
        features_df['volume_sma_20'] = features_df['volume'].rolling(20).mean()
        features_df['volume_ratio'] = features_df['volume'] / features_df['volume_sma_20']
        
        # On-Balance Volume
        obv = ta.volume.OnBalanceVolumeIndicator(features_df['close'], features_df['volume'])
        features_df['obv'] = obv.on_balance_volume()
        features_df['obv_sma'] = features_df['obv'].rolling(20).mean()
        features_df['obv_ratio'] = features_df['obv'] / features_df['obv_sma']
        
        # Accumulation/Distribution
        ad = ta.volume.AccDistIndexIndicator(features_df['high'], features_df['low'], features_df['close'], features_df['volume'])
        features_df['ad_line'] = ad.acc_dist_index()
        
        # Money Flow Index
        mfi = ta.volume.MFIIndicator(features_df['high'], features_df['low'], features_df['close'], features_df['volume'])
        features_df['mfi'] = mfi.money_flow_index()
        
        # Average True Range
        atr = ta.volatility.AverageTrueRange(features_df['high'], features_df['low'], features_df['close'])
        features_df['atr'] = atr.average_true_range()
        features_df['atr_ratio'] = features_df['atr'] / features_df['close']
        
        return features_df
    
    def generate_regime_specific_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate features specific to market regime analysis"""
        features_df = df.copy()
        
        # Detect regime for each row (using expanding window)
        regime_signals = []
        regimes = []
        regime_confidences = []
        
        for i in range(50, len(df)):  # Start after minimum required data
            window_df = df.iloc[:i+1]
            regime_signal = self.regime_detector.identify_regime(window_df, symbol)
            regime_signals.append(regime_signal)
            regimes.append(regime_signal.regime.value)
            regime_confidences.append(regime_signal.confidence)
        
        # Pad with initial values
        regimes = ['TRANSITION'] * 50 + regimes
        regime_confidences = [0.5] * 50 + regime_confidences
        
        features_df['regime'] = regimes
        features_df['regime_confidence'] = regime_confidences
        
        # Regime stability features
        features_df['regime_stability'] = features_df['regime'].rolling(10).apply(
            lambda x: (x == x.iloc[-1]).sum() / len(x), raw=False
        )
        
        # Regime transition features
        features_df['regime_changed'] = (features_df['regime'] != features_df['regime'].shift(1)).astype(int)
        features_df['bars_in_regime'] = features_df.groupby(
            (features_df['regime'] != features_df['regime'].shift()).cumsum()
        ).cumcount() + 1
        
        # Trend strength in different regimes
        for window in [12, 24, 48]:
            price_change = features_df['close'] / features_df['close'].shift(window) - 1
            volatility = features_df['returns'].rolling(window).std()
            features_df[f'trend_strength_{window}'] = price_change / (volatility + 1e-6)
        
        # Regime-conditional features
        for regime in MarketRegime:
            regime_mask = features_df['regime'] == regime.value
            if regime_mask.sum() > 10:  # Sufficient data points
                # RSI behavior in this regime
                features_df[f'rsi_14_in_{regime.value.lower()}'] = np.where(
                    regime_mask, features_df['rsi_14'], np.nan
                )
                
                # Volume behavior in this regime
                features_df[f'volume_ratio_in_{regime.value.lower()}'] = np.where(
                    regime_mask, features_df['volume_ratio'], np.nan
                )
        
        # Regime transition probability
        features_df['regime_transition_prob'] = 0.5  # Default
        for i, regime_signal in enumerate(regime_signals):
            if i < len(features_df) - 50:
                features_df.iloc[i + 50, features_df.columns.get_loc('regime_transition_prob')] = regime_signal.transition_probability
        
        return features_df
    
    def generate_adaptive_dip_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate adaptive features for dip detection across regimes"""
        features_df = df.copy()
        
        # Basic dip detection (original DipMaster logic)
        features_df['price_vs_open'] = (features_df['close'] - features_df['open']) / features_df['open']
        features_df['is_dip'] = (features_df['price_vs_open'] < -0.002).astype(int)
        
        # Regime-adaptive dip features
        regime_thresholds = {
            'RANGE_BOUND': -0.002,
            'STRONG_UPTREND': -0.003,
            'STRONG_DOWNTREND': -0.005,
            'HIGH_VOLATILITY': -0.004,
            'LOW_VOLATILITY': -0.001
        }
        
        features_df['adaptive_dip_signal'] = 0
        for regime, threshold in regime_thresholds.items():
            regime_mask = features_df['regime'] == regime
            dip_mask = features_df['price_vs_open'] < threshold
            features_df.loc[regime_mask & dip_mask, 'adaptive_dip_signal'] = 1
        
        # Dip magnitude relative to recent volatility
        features_df['dip_magnitude'] = abs(features_df['price_vs_open']) / (features_df['volatility_20'] + 1e-6)
        
        # Dip confirmation features
        features_df['volume_on_dip'] = features_df['volume_ratio'] * features_df['is_dip']
        features_df['rsi_on_dip'] = features_df['rsi_14'] * features_df['is_dip']
        
        # Multi-timeframe dip confirmation
        for window in [3, 5, 10]:
            features_df[f'dip_persistence_{window}'] = features_df['is_dip'].rolling(window).sum()
        
        # Dip recovery features
        for forward_window in [3, 5, 10, 15]:
            if len(features_df) > forward_window:
                features_df[f'price_recovery_{forward_window}'] = (
                    features_df['close'].shift(-forward_window) / features_df['close'] - 1
                )
        
        return features_df
    
    def generate_regime_interaction_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate interaction features between regime and technical indicators"""
        features_df = df.copy()
        
        if not self.config.enable_regime_interactions:
            return features_df
        
        # RSI x Regime interactions
        for regime in ['RANGE_BOUND', 'STRONG_UPTREND', 'STRONG_DOWNTREND']:
            regime_mask = features_df['regime'] == regime
            features_df[f'rsi_14_x_{regime.lower()}'] = features_df['rsi_14'] * regime_mask.astype(int)
        
        # Volume x Regime interactions
        for regime in ['HIGH_VOLATILITY', 'LOW_VOLATILITY']:
            regime_mask = features_df['regime'] == regime
            features_df[f'volume_ratio_x_{regime.lower()}'] = features_df['volume_ratio'] * regime_mask.astype(int)
        
        # MACD x Regime interactions
        for regime in ['STRONG_UPTREND', 'STRONG_DOWNTREND']:
            regime_mask = features_df['regime'] == regime
            features_df[f'macd_x_{regime.lower()}'] = features_df['macd'] * regime_mask.astype(int)
        
        # Volatility x Regime interactions
        features_df['volatility_regime_adjustment'] = features_df['volatility_20']
        high_vol_mask = features_df['regime'] == 'HIGH_VOLATILITY'
        low_vol_mask = features_df['regime'] == 'LOW_VOLATILITY'
        features_df.loc[high_vol_mask, 'volatility_regime_adjustment'] *= 2.0
        features_df.loc[low_vol_mask, 'volatility_regime_adjustment'] *= 0.5
        
        return features_df
    
    def generate_adaptive_labels(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate adaptive labels based on market regime"""
        features_df = df.copy()
        
        if not self.config.enable_adaptive_labels:
            # Default DipMaster labeling
            return self._generate_default_labels(features_df)
        
        # Regime-specific profit targets and holding periods
        regime_targets = {
            'RANGE_BOUND': {'profit': 0.008, 'holding': 180},
            'STRONG_UPTREND': {'profit': 0.012, 'holding': 90},
            'STRONG_DOWNTREND': {'profit': 0.006, 'holding': 60},
            'HIGH_VOLATILITY': {'profit': 0.015, 'holding': 45},
            'LOW_VOLATILITY': {'profit': 0.005, 'holding': 240}
        }
        
        # Initialize label columns
        features_df['adaptive_target'] = 0
        features_df['adaptive_target_15min'] = 0
        features_df['adaptive_holding_return'] = 0
        
        for i in range(len(features_df) - 300):  # Ensure enough forward data
            current_regime = features_df.iloc[i]['regime']
            if current_regime not in regime_targets:
                current_regime = 'RANGE_BOUND'  # Default
            
            target_params = regime_targets[current_regime]
            profit_target = target_params['profit']
            max_holding = min(target_params['holding'], 300)  # Cap at available data
            
            entry_price = features_df.iloc[i]['close']
            
            # Check for profit target achievement within regime-specific timeframe
            for j in range(1, max_holding + 1):
                if i + j >= len(features_df):
                    break
                
                future_price = features_df.iloc[i + j]['close']
                future_return = (future_price - entry_price) / entry_price
                
                # Check 15-minute boundary alignment (every 3 bars in 5-min data)
                is_15min_boundary = (j % 3 == 0)
                
                if future_return >= profit_target:
                    features_df.iloc[i, features_df.columns.get_loc('adaptive_target')] = 1
                    if is_15min_boundary:
                        features_df.iloc[i, features_df.columns.get_loc('adaptive_target_15min')] = 1
                    break
                elif future_return <= -0.015:  # Stop loss
                    break
            
            # Record holding period return
            if i + max_holding < len(features_df):
                exit_price = features_df.iloc[i + max_holding]['close']
                holding_return = (exit_price - entry_price) / entry_price
                features_df.iloc[i, features_df.columns.get_loc('adaptive_holding_return')] = holding_return
        
        return features_df
    
    def _generate_default_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate default DipMaster labels"""
        features_df = df.copy()
        
        # Original DipMaster logic
        features_df['target'] = 0
        features_df['target_15min'] = 0
        features_df['holding_return'] = 0
        
        for i in range(len(features_df) - 180):
            entry_price = features_df.iloc[i]['close']
            
            for j in range(1, 181):
                if i + j >= len(features_df):
                    break
                
                future_price = features_df.iloc[i + j]['close']
                future_return = (future_price - entry_price) / entry_price
                
                is_15min_boundary = (j % 3 == 0)
                
                if future_return >= 0.008:  # 0.8% profit target
                    features_df.iloc[i, features_df.columns.get_loc('target')] = 1
                    if is_15min_boundary:
                        features_df.iloc[i, features_df.columns.get_loc('target_15min')] = 1
                    break
                elif future_return <= -0.015:  # 1.5% stop loss
                    break
            
            # 180-minute holding return
            if i + 180 < len(features_df):
                exit_price = features_df.iloc[i + 180]['close']
                holding_return = (exit_price - entry_price) / entry_price
                features_df.iloc[i, features_df.columns.get_loc('holding_return')] = holding_return
        
        return features_df
    
    def generate_cross_regime_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Generate features based on regime transitions and cross-regime analysis"""
        features_df = df.copy()
        
        if not self.config.enable_cross_regime_features:
            return features_df
        
        # Regime transition momentum
        features_df['regime_momentum'] = 0
        for i in range(10, len(features_df)):
            recent_regimes = features_df['regime'].iloc[i-10:i].tolist()
            if len(set(recent_regimes)) == 1:  # Stable regime
                features_df.iloc[i, features_df.columns.get_loc('regime_momentum')] = 1
            elif len(set(recent_regimes)) > 3:  # High transition
                features_df.iloc[i, features_df.columns.get_loc('regime_momentum')] = -1
        
        # Time since last regime change
        regime_changes = features_df['regime_changed'].cumsum()
        features_df['bars_since_regime_change'] = features_df.groupby(regime_changes).cumcount()
        
        # Regime cycle features
        regime_sequence = features_df['regime'].tolist()
        features_df['in_regime_cycle'] = 0
        
        # Look for common regime patterns
        common_patterns = [
            ['HIGH_VOLATILITY', 'STRONG_DOWNTREND', 'RANGE_BOUND'],
            ['LOW_VOLATILITY', 'STRONG_UPTREND', 'HIGH_VOLATILITY'],
            ['RANGE_BOUND', 'STRONG_UPTREND', 'RANGE_BOUND']
        ]
        
        for i in range(len(features_df) - 2):
            current_sequence = regime_sequence[i:i+3]
            if current_sequence in common_patterns:
                features_df.iloc[i:i+3, features_df.columns.get_loc('in_regime_cycle')] = 1
        
        return features_df
    
    def process_symbol_data(self, symbol: str, data_path: str) -> Dict:
        """Process complete feature engineering for a single symbol"""
        start_time = time.time()
        self.logger.info(f"Starting regime-aware feature engineering for {symbol}")
        
        try:
            # Load data
            if data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                df = pd.read_csv(data_path)
            
            # Ensure required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Missing required columns in {symbol} data")
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Convert timestamp if needed
            if df['timestamp'].dtype == 'object':
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            self.logger.info(f"Loaded {len(df)} rows for {symbol}")
            
            # Feature engineering pipeline
            features_df = self.generate_base_technical_features(df, symbol)
            features_df = self.generate_regime_specific_features(features_df, symbol)
            features_df = self.generate_adaptive_dip_features(features_df, symbol)
            features_df = self.generate_regime_interaction_features(features_df, symbol)
            features_df = self.generate_cross_regime_features(features_df, symbol)
            features_df = self.generate_adaptive_labels(features_df, symbol)
            
            # Remove rows with insufficient data
            features_df = features_df.dropna(subset=['regime', 'regime_confidence'])
            
            # Feature selection and cleaning
            feature_cols = [col for col in features_df.columns 
                          if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Remove infinite and extreme values
            for col in feature_cols:
                if features_df[col].dtype in ['float64', 'int64']:
                    features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
                    features_df[col] = features_df[col].fillna(features_df[col].median())
            
            processing_time = time.time() - start_time
            
            result = {
                'symbol': symbol,
                'features_generated': len(feature_cols),
                'total_rows': len(features_df),
                'valid_rows': len(features_df.dropna()),
                'processing_time': processing_time,
                'features_df': features_df,
                'regime_distribution': features_df['regime'].value_counts().to_dict(),
                'adaptive_target_rate': features_df['adaptive_target'].mean() if 'adaptive_target' in features_df.columns else 0,
                'default_target_rate': features_df['target'].mean() if 'target' in features_df.columns else 0
            }
            
            self.logger.info(f"Completed {symbol} in {processing_time:.2f}s - {len(feature_cols)} features generated")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'error': str(e),
                'features_generated': 0,
                'processing_time': time.time() - start_time
            }
    
    def process_multiple_symbols(self, data_mapping: Dict[str, str], 
                                max_workers: int = 4) -> Dict:
        """Process feature engineering for multiple symbols in parallel"""
        start_time = time.time()
        results = {}
        
        self.logger.info(f"Starting parallel processing for {len(data_mapping)} symbols")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_symbol = {
                executor.submit(self.process_symbol_data, symbol, data_path): symbol
                for symbol, data_path in data_mapping.items()
            }
            
            # Collect results
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results[symbol] = result
                except Exception as e:
                    self.logger.error(f"Failed to process {symbol}: {str(e)}")
                    results[symbol] = {'symbol': symbol, 'error': str(e)}
        
        total_time = time.time() - start_time
        
        # Summary statistics
        successful_symbols = [s for s, r in results.items() if 'error' not in r]
        total_features = sum(r.get('features_generated', 0) for r in results.values())
        total_rows = sum(r.get('total_rows', 0) for r in results.values())
        
        summary = {
            'processing_summary': {
                'total_symbols': len(data_mapping),
                'successful_symbols': len(successful_symbols),
                'failed_symbols': len(data_mapping) - len(successful_symbols),
                'total_processing_time': total_time,
                'total_features_generated': total_features,
                'total_rows_processed': total_rows
            },
            'symbol_results': results
        }
        
        self.logger.info(f"Completed processing {len(successful_symbols)}/{len(data_mapping)} symbols in {total_time:.2f}s")
        return summary
    
    def export_features(self, results: Dict, output_dir: str) -> Dict:
        """Export processed features to files"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exports = {}
        
        # Export individual symbol features
        for symbol, result in results['symbol_results'].items():
            if 'features_df' in result:
                # Export features
                feature_file = output_path / f"{symbol}_regime_features_{timestamp}.parquet"
                result['features_df'].to_parquet(feature_file, index=False)
                
                # Export regime analysis
                regime_analysis = self.regime_detector.export_regime_analysis(
                    symbol, result['features_df'], output_path
                )
                
                exports[symbol] = {
                    'features_file': str(feature_file),
                    'regime_analysis': regime_analysis
                }
        
        # Export combined dataset
        if exports:
            combined_features = []
            for symbol, result in results['symbol_results'].items():
                if 'features_df' in result:
                    symbol_df = result['features_df'].copy()
                    symbol_df['symbol'] = symbol
                    combined_features.append(symbol_df)
            
            if combined_features:
                combined_df = pd.concat(combined_features, ignore_index=True)
                combined_file = output_path / f"regime_aware_features_combined_{timestamp}.parquet"
                combined_df.to_parquet(combined_file, index=False)
                exports['combined'] = str(combined_file)
        
        # Export summary report
        summary_file = output_path / f"regime_feature_engineering_report_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        exports['summary'] = str(summary_file)
        
        self.logger.info(f"Features exported to {output_path}")
        return exports

# Utility functions for easy integration
def create_regime_feature_engineer(symbols: List[str] = None) -> RegimeAwareFeatureEngineer:
    """Factory function to create regime-aware feature engineer"""
    config = RegimeFeatureConfig(symbols=symbols) if symbols else None
    return RegimeAwareFeatureEngineer(config)

def process_symbol_with_regime_features(symbol: str, data_path: str, 
                                      output_dir: str = None) -> Dict:
    """Process a single symbol with regime-aware features"""
    engineer = create_regime_feature_engineer([symbol])
    result = engineer.process_symbol_data(symbol, data_path)
    
    if output_dir and 'features_df' in result:
        exports = engineer.export_features({'symbol_results': {symbol: result}}, output_dir)
        result['exports'] = exports
    
    return result