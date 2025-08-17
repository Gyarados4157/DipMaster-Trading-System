#!/usr/bin/env python3
"""
DipMaster Standalone Enhanced Feature Generation V5
独立增强版特征生成V5

Complete standalone pipeline for generating enhanced features without complex dependencies.
Generates FeatureSet.json and features.parquet with comprehensive validation.

Author: DipMaster Quant Team
Date: 2025-08-17
Version: 5.0.0-Standalone
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import time
import warnings
import ta

warnings.filterwarnings('ignore')

class StandaloneEnhancedFeatureGenerator:
    """Standalone enhanced feature generator with comprehensive indicators"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        
        # Configuration
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT',
            'BNBUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT',
            'LTCUSDT', 'DOTUSDT', 'ATOMUSDT', 'FILUSDT', 'NEARUSDT',
            'ARBUSDT', 'OPUSDT', 'APTUSDT', 'AAVEUSDT', 'COMPUSDT',
            'MKRUSDT', 'QNTUSDT', 'TRXUSDT', 'VETUSDT', 'XLMUSDT'
        ]
        
        self.config = {
            'rsi_periods': [7, 14, 21, 30, 50],
            'ma_periods': [5, 10, 20, 50, 100, 200],
            'prediction_horizons': [1, 3, 6, 12, 24, 36],
            'profit_targets': [0.003, 0.006, 0.008, 0.012, 0.015, 0.020],
            'stop_loss': 0.004,
            'max_holding_periods': 36
        }
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data from parquet files"""
        self.logger.info("Loading market data...")
        
        data_dir = Path("data/enhanced_market_data")
        market_data = {}
        
        for symbol in self.symbols:
            try:
                file_path = data_dir / f"{symbol}_5m_90days.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    
                    # Validate data
                    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_cols):
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df[df['close'] > 0].copy()
                        
                        if len(df) >= 1000:
                            market_data[symbol] = df
                            self.logger.info(f"Loaded {symbol}: {len(df)} rows")
            except Exception as e:
                self.logger.warning(f"Failed to load {symbol}: {e}")
        
        self.logger.info(f"Loaded {len(market_data)} symbols")
        return market_data
    
    def add_technical_indicators(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        try:
            # 1. RSI indicators
            for period in self.config['rsi_periods']:
                df[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['close'], window=period).rsi()
                df[f'rsi_{period}_ma5'] = df[f'rsi_{period}'].rolling(5).mean()
                df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
                df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
                df[f'rsi_{period}_dip_zone'] = ((df[f'rsi_{period}'] >= 25) & (df[f'rsi_{period}'] <= 45)).astype(int)
            
            # 2. MACD indicators  
            for fast, slow, signal in [(12, 26, 9), (5, 35, 5), (8, 21, 5)]:
                macd = ta.trend.MACD(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
                df[f'macd_{fast}_{slow}'] = macd.macd()
                df[f'macd_signal_{fast}_{slow}'] = macd.macd_signal()
                df[f'macd_histogram_{fast}_{slow}'] = macd.macd_diff()
                df[f'macd_bullish_{fast}_{slow}'] = (df[f'macd_{fast}_{slow}'] > df[f'macd_signal_{fast}_{slow}']).astype(int)
            
            # 3. Stochastic indicators (KDJ)
            for k_period, d_period in [(14, 3), (9, 3), (21, 5)]:
                stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=k_period, smooth_window=d_period)
                df[f'stoch_k_{k_period}'] = stoch.stoch()
                df[f'stoch_d_{k_period}'] = stoch.stoch_signal()
                df[f'stoch_j_{k_period}'] = 3 * df[f'stoch_k_{k_period}'] - 2 * df[f'stoch_d_{k_period}']
            
            # 4. Williams %R
            for period in [14, 21]:
                df[f'williams_r_{period}'] = ta.momentum.WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=period).williams_r()
            
            # 5. Moving Averages
            for period in self.config['ma_periods']:
                df[f'sma_{period}'] = ta.trend.SMAIndicator(df['close'], window=period).sma_indicator()
                df[f'ema_{period}'] = ta.trend.EMAIndicator(df['close'], window=period).ema_indicator()
                df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
                df[f'above_sma_{period}'] = (df['close'] > df[f'sma_{period}']).astype(int)
            
            # 6. Bollinger Bands
            for period, std_mult in [(20, 2.0), (20, 1.5), (50, 2.0)]:
                bb = ta.volatility.BollingerBands(df['close'], window=period, window_dev=std_mult)
                df[f'bb_upper_{period}_{std_mult}'] = bb.bollinger_hband()
                df[f'bb_middle_{period}_{std_mult}'] = bb.bollinger_mavg()
                df[f'bb_lower_{period}_{std_mult}'] = bb.bollinger_lband()
                df[f'bb_width_{period}_{std_mult}'] = bb.bollinger_wband()
                df[f'bb_position_{period}_{std_mult}'] = bb.bollinger_pband()
                df[f'bb_squeeze_{period}_{std_mult}'] = (df[f'bb_width_{period}_{std_mult}'] < 
                                                        df[f'bb_width_{period}_{std_mult}'].rolling(50).quantile(0.2)).astype(int)
            
            # 7. Volume indicators
            df['volume_sma_20'] = ta.volume.VolumeSMAIndicator(df['close'], df['volume'], window=20).volume_sma()
            df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
            df['obv_slope'] = df['obv'].pct_change()
            
            for period in [10, 20]:
                df[f'volume_ratio_{period}'] = df['volume'] / df['volume'].rolling(period).mean()
                df[f'volume_spike_{period}'] = (df[f'volume_ratio_{period}'] > 1.5).astype(int)
            
            # 8. Volatility indicators
            for period in [5, 10, 20, 50]:
                df[f'atr_{period}'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close'], window=period).average_true_range()
                returns = df['close'].pct_change()
                df[f'volatility_{period}'] = returns.rolling(period).std()
            
            # 9. Momentum indicators
            for period in [10, 14, 20]:
                df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
                df[f'roc_{period}'] = ta.momentum.ROCIndicator(df['close'], window=period).roc()
            
            # 10. CCI and ADX
            for period in [14, 20]:
                df[f'cci_{period}'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=period).cci()
            
            for period in [14, 21]:
                adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=period)
                df[f'adx_{period}'] = adx.adx()
                df[f'adx_trend_strength_{period}'] = df[f'adx_{period}'] / 100.0
            
            return df
            
        except Exception as e:
            self.logger.error(f"Technical indicators failed for {symbol}: {e}")
            return df
    
    def add_microstructure_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add market microstructure features"""
        try:
            # Price and volume patterns
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            df['body_size'] = abs(df['close'] - df['open'])
            df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
            df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
            df['total_range'] = df['high'] - df['low']
            
            # Shadow ratios
            df['upper_shadow_ratio'] = df['upper_shadow'] / (df['total_range'] + 1e-8)
            df['lower_shadow_ratio'] = df['lower_shadow'] / (df['total_range'] + 1e-8)
            df['body_ratio'] = df['body_size'] / (df['total_range'] + 1e-8)
            
            # Candlestick patterns
            df['hammer'] = ((df['lower_shadow_ratio'] > 0.5) & (df['body_ratio'] < 0.3)).astype(int)
            df['doji'] = (df['body_ratio'] < 0.1).astype(int)
            df['shooting_star'] = ((df['upper_shadow_ratio'] > 0.5) & (df['body_ratio'] < 0.3)).astype(int)
            
            # VWAP features
            for period in [5, 10, 20]:
                vwap_num = (df['close'] * df['volume']).rolling(period).sum()
                vwap_den = df['volume'].rolling(period).sum()
                df[f'vwap_{period}'] = vwap_num / vwap_den
                df[f'vwap_deviation_{period}'] = (df['close'] - df[f'vwap_{period}']) / df[f'vwap_{period}']
                df[f'above_vwap_{period}'] = (df['close'] > df[f'vwap_{period}']).astype(int)
            
            # Order flow approximation
            df['buy_volume_proxy'] = np.where(df['close'] > df['open'], df['volume'], 0)
            df['sell_volume_proxy'] = np.where(df['close'] < df['open'], df['volume'], 0)
            
            for period in [5, 10]:
                buy_vol = df['buy_volume_proxy'].rolling(period).sum()
                sell_vol = df['sell_volume_proxy'].rolling(period).sum()
                total_vol = df['volume'].rolling(period).sum()
                df[f'buy_ratio_{period}'] = buy_vol / total_vol
                df[f'order_flow_imbalance_{period}'] = (buy_vol - sell_vol) / total_vol
            
            # Price impact proxy
            returns = df['close'].pct_change().abs()
            volume_norm = df['volume'] / df['volume'].rolling(20).mean()
            df['price_impact'] = returns / (volume_norm + 1e-8)
            
            # Time-based features
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                df['minute'] = df['timestamp'].dt.minute
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                
                # Session indicators
                df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
                df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
                df['ny_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Microstructure features failed for {symbol}: {e}")
            return df
    
    def add_cross_asset_features(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Add cross-asset relative strength features"""
        try:
            if len(all_data) < 2:
                return all_data
            
            # Create price and volume matrices
            price_data = {}
            volume_data = {}
            
            for symbol, df in all_data.items():
                if len(df) > 100:
                    df_indexed = df.set_index('timestamp')
                    price_data[symbol] = df_indexed['close']
                    volume_data[symbol] = df_indexed['volume']
            
            price_df = pd.DataFrame(price_data).ffill().bfill()
            volume_df = pd.DataFrame(volume_data).ffill().bfill()
            
            if price_df.empty:
                return all_data
            
            # Calculate market metrics
            returns_df = price_df.pct_change()
            market_return = returns_df.mean(axis=1)
            
            # Add relative features to each symbol
            for symbol in all_data:
                if symbol not in price_df.columns:
                    continue
                
                df = all_data[symbol].copy()
                df_indexed = df.set_index('timestamp')
                
                symbol_returns = returns_df[symbol]
                
                # Relative strength
                relative_returns = symbol_returns - market_return
                df_indexed['relative_strength_1h'] = relative_returns.rolling(12).mean()
                df_indexed['relative_strength_4h'] = relative_returns.rolling(48).mean()
                df_indexed['relative_strength_1d'] = relative_returns.rolling(288).mean()
                
                # Relative rankings
                daily_returns = returns_df.rolling(288).sum()
                df_indexed['daily_rank'] = daily_returns.rank(axis=1, pct=True)[symbol]
                
                # Correlation with BTC (if available)
                if 'BTCUSDT' in returns_df.columns and symbol != 'BTCUSDT':
                    btc_returns = returns_df['BTCUSDT']
                    df_indexed['btc_correlation'] = symbol_returns.rolling(144).corr(btc_returns)
                
                # Volume relative to market
                if symbol in volume_df.columns:
                    market_volume = volume_df.mean(axis=1)
                    symbol_volume = volume_df[symbol]
                    df_indexed['volume_vs_market'] = symbol_volume / market_volume
                
                # Reset index and update
                df_updated = df_indexed.reset_index()
                all_data[symbol] = df_updated
            
            return all_data
            
        except Exception as e:
            self.logger.error(f"Cross-asset features failed: {e}")
            return all_data
    
    def add_dipmaster_signals(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add DipMaster-specific signal features"""
        try:
            # Core DipMaster signals
            if 'rsi_14' in df.columns:
                df['rsi_dip_signal'] = ((df['rsi_14'] >= 30) & (df['rsi_14'] <= 50)).astype(int)
            
            # Price dip detection
            df['price_dip_current'] = (df['close'] < df['open']).astype(int)
            df['price_dip_prev'] = (df['close'].shift(1) < df['open'].shift(1)).astype(int)
            df['consecutive_dips'] = df['price_dip_current'] + df['price_dip_prev'] * 0.5
            
            # Volume confirmation
            if 'volume_ratio_20' in df.columns:
                df['volume_confirmation'] = (df['volume_ratio_20'] > 1.2).astype(int)
            
            # MA position
            if 'sma_20' in df.columns:
                df['below_ma20'] = (df['close'] < df['sma_20']).astype(int)
            
            # Combined DipMaster signal
            signal_components = []
            weights = []
            
            if 'rsi_dip_signal' in df.columns:
                signal_components.append(df['rsi_dip_signal'])
                weights.append(0.3)
            
            if 'consecutive_dips' in df.columns:
                signal_components.append(np.clip(df['consecutive_dips'] / 1.5, 0, 1))
                weights.append(0.25)
            
            if 'volume_confirmation' in df.columns:
                signal_components.append(df['volume_confirmation'])
                weights.append(0.2)
            
            if 'below_ma20' in df.columns:
                signal_components.append(df['below_ma20'])
                weights.append(0.15)
            
            if 'bb_position_20_2.0' in df.columns:
                bb_signal = np.clip((0.2 - df['bb_position_20_2.0']) * 5, 0, 1)
                signal_components.append(bb_signal)
                weights.append(0.1)
            
            # Calculate weighted signal
            if signal_components and weights:
                weights = np.array(weights) / sum(weights)
                df['dipmaster_signal_strength'] = sum(
                    comp * weight for comp, weight in zip(signal_components, weights)
                )
            else:
                df['dipmaster_signal_strength'] = 0
            
            # Signal quality flags
            df['high_quality_signal'] = (df['dipmaster_signal_strength'] > 0.7).astype(int)
            df['medium_quality_signal'] = ((df['dipmaster_signal_strength'] > 0.5) & 
                                          (df['dipmaster_signal_strength'] <= 0.7)).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"DipMaster signals failed for {symbol}: {e}")
            return df
    
    def add_optimized_labels(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add optimized labels for DipMaster strategy"""
        try:
            # Ensure timestamp and minute extraction
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['minute'] = df['timestamp'].dt.minute
            else:
                df['minute'] = 0
            
            # 15-minute boundary minutes
            boundary_minutes = [15, 30, 45, 0]
            
            # Generate labels for different horizons
            for horizon in self.config['prediction_horizons']:
                # Future return
                future_return = df['close'].pct_change(periods=horizon).shift(-horizon)
                df[f'future_return_{horizon}p'] = future_return
                
                # Binary profitability
                df[f'is_profitable_{horizon}p'] = (future_return > 0).astype(int)
                
                # Check if exit is at 15-minute boundary
                future_minute = df['minute'].shift(-horizon)
                is_boundary_exit = future_minute.isin(boundary_minutes)
                df[f'boundary_exit_{horizon}p'] = is_boundary_exit.astype(int)
                
                # Profit targets
                for target in self.config['profit_targets']:
                    df[f'hits_target_{target:.1%}_{horizon}p'] = (future_return >= target).astype(int)
                    df[f'hits_target_boundary_{target:.1%}_{horizon}p'] = (
                        (future_return >= target) & is_boundary_exit
                    ).astype(int)
                
                # Stop loss
                df[f'hits_stop_loss_{horizon}p'] = (future_return <= -self.config['stop_loss']).astype(int)
                
                # Maximum favorable/adverse excursion
                if horizon <= 36:  # Only for reasonable horizons
                    future_highs = df['high'].rolling(window=horizon, min_periods=1).max().shift(-horizon)
                    future_lows = df['low'].rolling(window=horizon, min_periods=1).min().shift(-horizon)
                    
                    mfe = (future_highs - df['close']) / df['close']
                    mae = (future_lows - df['close']) / df['close']
                    df[f'mfe_{horizon}p'] = mfe
                    df[f'mae_{horizon}p'] = mae
            
            # Primary DipMaster labels (12-period = 1 hour)
            main_return = df['future_return_12p'] if 'future_return_12p' in df.columns else pd.Series(0, index=df.index)
            df['target_return'] = main_return
            df['target_binary'] = (main_return > 0).astype(int)
            df['target_profitable_0.6%'] = (main_return >= 0.006).astype(int)
            
            # DipMaster win condition
            if 'boundary_exit_12p' in df.columns:
                df['dipmaster_win'] = (
                    (main_return >= 0.006) |  # 0.6% profit OR
                    ((main_return >= 0.003) & (df['boundary_exit_12p'] == 1))  # 0.3% at boundary
                ).astype(int)
            else:
                df['dipmaster_win'] = df['target_profitable_0.6%']
            
            # Multi-class labels
            conditions = [
                (main_return <= -0.004),  # Stop loss
                ((main_return > -0.004) & (main_return <= 0)),  # Small loss
                ((main_return > 0) & (main_return < 0.006)),  # Small profit
                ((main_return >= 0.006) & (main_return < 0.012)),  # Good profit
                (main_return >= 0.012)  # Excellent profit
            ]
            labels = [0, 1, 2, 3, 4]
            df['return_class'] = np.select(conditions, labels, default=1)
            
            # Risk-adjusted return
            if 'volatility_20' in df.columns:
                vol_adj_return = main_return / (df['volatility_20'] + 1e-8)
                df['target_risk_adjusted'] = vol_adj_return
            
            return df
            
        except Exception as e:
            self.logger.error(f"Optimized labels failed for {symbol}: {e}")
            return df
    
    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate features"""
        try:
            # Handle infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values intelligently
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col.startswith('target') or 'future_' in col:
                    continue  # Don't fill target variables
                
                # Forward fill then backward fill, then zero
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Remove extreme outliers
            for col in numeric_cols:
                if col not in ['timestamp', 'hour', 'minute', 'day_of_week'] and not col.startswith('target'):
                    Q1 = df[col].quantile(0.01)
                    Q3 = df[col].quantile(0.99)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 2 * IQR
                    upper_bound = Q3 + 2 * IQR
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Feature cleaning failed: {e}")
            return df
    
    def process_all_symbols(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Process features for all symbols"""
        self.logger.info("Processing features for all symbols...")
        
        processed_data = {}
        
        # Step 1: Individual symbol processing
        for symbol, df in market_data.items():
            try:
                self.logger.info(f"Processing {symbol}...")
                enhanced_df = df.copy()
                
                # Add technical indicators
                enhanced_df = self.add_technical_indicators(enhanced_df, symbol)
                
                # Add microstructure features
                enhanced_df = self.add_microstructure_features(enhanced_df, symbol)
                
                # Add DipMaster signals
                enhanced_df = self.add_dipmaster_signals(enhanced_df, symbol)
                
                # Add labels
                enhanced_df = self.add_optimized_labels(enhanced_df, symbol)
                
                # Clean features
                enhanced_df = self.clean_features(enhanced_df)
                
                processed_data[symbol] = enhanced_df
                
            except Exception as e:
                self.logger.error(f"Processing failed for {symbol}: {e}")
        
        # Step 2: Cross-asset features
        if len(processed_data) > 1:
            self.logger.info("Adding cross-asset features...")
            processed_data = self.add_cross_asset_features(processed_data)
        
        return processed_data
    
    def analyze_features(self, combined_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature quality and generate metrics"""
        # Feature categorization
        feature_cols = [col for col in combined_df.columns 
                       if not any(x in col for x in ['target', 'future_', 'hits_', 'mfe_', 'mae_'])
                       and col not in ['timestamp', 'symbol']]
        
        label_cols = [col for col in combined_df.columns 
                     if any(x in col for x in ['target', 'future_', 'hits_', 'mfe_', 'mae_'])]
        
        # Categorize features
        categories = {
            'technical_indicators': [col for col in feature_cols if any(
                x in col for x in ['rsi', 'macd', 'stoch', 'williams', 'sma', 'ema', 'bb_', 'atr', 'momentum', 'roc', 'cci', 'adx']
            )],
            'microstructure': [col for col in feature_cols if any(
                x in col for x in ['vwap', 'shadow', 'body', 'hammer', 'doji', 'order_flow', 'buy_ratio', 'price_impact']
            )],
            'cross_asset': [col for col in feature_cols if any(
                x in col for x in ['relative_strength', 'correlation', 'rank', 'volume_vs_market']
            )],
            'dipmaster_signals': [col for col in feature_cols if any(
                x in col for x in ['dipmaster', 'dip_signal', 'quality_signal']
            )],
            'time_features': [col for col in feature_cols if any(
                x in col for x in ['hour', 'minute', 'day_of_week', 'session']
            )],
            'other': []
        }
        
        # Classify remaining features
        categorized = set()
        for cat_features in categories.values():
            categorized.update(cat_features)
        
        categories['other'] = [col for col in feature_cols if col not in categorized]
        
        # Feature statistics
        feature_stats = {}
        for col in feature_cols:
            if combined_df[col].dtype in ['int64', 'float64']:
                feature_stats[col] = {
                    'nan_pct': float(combined_df[col].isnull().sum() / len(combined_df)),
                    'unique_values': int(combined_df[col].nunique()),
                    'mean': float(combined_df[col].mean()),
                    'std': float(combined_df[col].std()),
                    'min': float(combined_df[col].min()),
                    'max': float(combined_df[col].max())
                }
        
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
            'symbols_count': combined_df['symbol'].nunique(),
            'feature_categories': {cat: len(features) for cat, features in categories.items()},
            'feature_statistics': feature_stats,
            'label_analysis': label_analysis,
            'time_range': {
                'start': str(combined_df['timestamp'].min()),
                'end': str(combined_df['timestamp'].max())
            },
            'memory_usage_mb': float(combined_df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        return analysis
    
    def create_data_splits(self, combined_df: pd.DataFrame) -> Dict[str, Any]:
        """Create train/validation/test splits"""
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        total_samples = len(combined_df)
        train_end = int(total_samples * 0.70)
        val_end = int(total_samples * 0.85)
        
        train_df = combined_df.iloc[:train_end].copy()
        val_df = combined_df.iloc[train_end:val_end].copy()
        test_df = combined_df.iloc[val_end:].copy()
        
        split_info = {
            'train': {
                'start_date': str(train_df['timestamp'].min()),
                'end_date': str(train_df['timestamp'].max()),
                'samples': len(train_df)
            },
            'validation': {
                'start_date': str(val_df['timestamp'].min()),
                'end_date': str(val_df['timestamp'].max()),
                'samples': len(val_df)
            },
            'test': {
                'start_date': str(test_df['timestamp'].min()),
                'end_date': str(test_df['timestamp'].max()),
                'samples': len(test_df)
            }
        }
        
        return {
            'combined_data': combined_df,
            'train_data': train_df,
            'validation_data': val_df,
            'test_data': test_df,
            'split_info': split_info
        }
    
    def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete feature generation pipeline"""
        start_time = time.time()
        
        try:
            self.logger.info("🚀 Starting DipMaster Enhanced Feature Generation V5")
            
            # Load data
            market_data = self.load_market_data()
            if not market_data:
                raise ValueError("No market data loaded")
            
            # Process features
            processed_data = self.process_all_symbols(market_data)
            
            # Combine data
            combined_dfs = []
            for symbol, df in processed_data.items():
                df_copy = df.copy()
                df_copy['symbol'] = symbol
                combined_dfs.append(df_copy)
            
            combined_df = pd.concat(combined_dfs, ignore_index=True, sort=False)
            combined_df = combined_df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
            
            # Create splits
            split_data = self.create_data_splits(combined_df)
            
            # Analyze features
            analysis = self.analyze_features(combined_df)
            
            # Create feature set config
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            feature_set_config = {
                "metadata": {
                    "version": "5.0.0-Standalone",
                    "strategy_name": "DipMaster_Enhanced_V5",
                    "created_timestamp": timestamp,
                    "description": "Comprehensive enhanced feature set with 100+ indicators",
                    "pipeline_type": "standalone_enhanced"
                },
                "symbols": list(combined_df['symbol'].unique()),
                "data_splits": split_data['split_info'],
                "feature_engineering": {
                    "total_features": analysis['total_features'],
                    "total_labels": analysis['total_labels'],
                    "feature_categories": analysis['feature_categories'],
                    "enhancement_components": [
                        "comprehensive_technical_indicators",
                        "market_microstructure_features",
                        "cross_asset_relative_strength",
                        "dipmaster_signal_engineering",
                        "optimized_15min_boundary_labels",
                        "intelligent_data_cleaning"
                    ]
                },
                "quality_metrics": analysis,
                "target_specifications": {
                    "primary_target": "target_return",
                    "binary_target": "target_binary", 
                    "strategy_target": "dipmaster_win",
                    "profit_targets": self.config['profit_targets'],
                    "time_horizons": self.config['prediction_horizons']
                },
                "files": {
                    "feature_data": f"Enhanced_Features_V5_Standalone_{timestamp}.parquet",
                    "train_data": f"train_features_V5_{timestamp}.parquet",
                    "validation_data": f"validation_features_V5_{timestamp}.parquet",
                    "test_data": f"test_features_V5_{timestamp}.parquet"
                }
            }
            
            # Save files
            output_dir = Path("data")
            
            # Save config
            config_path = output_dir / f"Enhanced_FeatureSet_V5_Standalone_{timestamp}.json"
            with open(config_path, 'w') as f:
                json.dump(feature_set_config, f, indent=2, default=str)
            
            # Save data files
            features_path = output_dir / f"Enhanced_Features_V5_Standalone_{timestamp}.parquet"
            combined_df.to_parquet(features_path, compression='snappy', index=False)
            
            train_path = output_dir / f"train_features_V5_{timestamp}.parquet"
            split_data['train_data'].to_parquet(train_path, compression='snappy', index=False)
            
            val_path = output_dir / f"validation_features_V5_{timestamp}.parquet"
            split_data['validation_data'].to_parquet(val_path, compression='snappy', index=False)
            
            test_path = output_dir / f"test_features_V5_{timestamp}.parquet"
            split_data['test_data'].to_parquet(test_path, compression='snappy', index=False)
            
            total_time = time.time() - start_time
            
            result = {
                'status': 'SUCCESS',
                'runtime_seconds': total_time,
                'symbols_processed': len(processed_data),
                'total_samples': len(combined_df),
                'total_features': analysis['total_features'],
                'total_labels': analysis['total_labels'],
                'files_created': {
                    'config': str(config_path),
                    'features': str(features_path),
                    'train': str(train_path),
                    'validation': str(val_path),
                    'test': str(test_path)
                },
                'analysis': analysis
            }
            
            self.logger.info("✅ Pipeline completed successfully!")
            self.logger.info(f"⏱️  Runtime: {total_time:.1f} seconds")
            self.logger.info(f"📊 Features: {analysis['total_features']}")
            self.logger.info(f"🏷️  Labels: {analysis['total_labels']}")
            self.logger.info(f"💾 Samples: {len(combined_df):,}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            return {
                'status': 'FAILED',
                'error': str(e),
                'runtime_seconds': time.time() - start_time
            }

def main():
    """Main execution"""
    print("DipMaster Enhanced Feature Generation V5 - Standalone")
    print("=" * 60)
    
    generator = StandaloneEnhancedFeatureGenerator()
    result = generator.run_pipeline()
    
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    
    if result['status'] == 'SUCCESS':
        print(f"✅ Status: {result['status']}")
        print(f"⏱️  Runtime: {result['runtime_seconds']:.1f} seconds")
        print(f"🪙 Symbols: {result['symbols_processed']}")
        print(f"💾 Samples: {result['total_samples']:,}")
        print(f"📊 Features: {result['total_features']}")
        print(f"🏷️  Labels: {result['total_labels']}")
        
        print("\n📁 Files Created:")
        for file_type, path in result['files_created'].items():
            print(f"   {file_type}: {path}")
        
        print("\n📊 Feature Categories:")
        for category, count in result['analysis']['feature_categories'].items():
            print(f"   {category}: {count}")
        
        if 'label_analysis' in result['analysis']:
            print("\n📈 Label Quality:")
            for label, metrics in result['analysis']['label_analysis'].items():
                if 'win_rate' in metrics:
                    print(f"   {label}: {metrics['win_rate']:.1%} win rate")
                elif 'positive_rate' in metrics:
                    print(f"   {label}: {metrics['positive_rate']:.1%} positive rate")
        
        print(f"\n💾 Memory Usage: {result['analysis']['memory_usage_mb']:.1f} MB")
        
    else:
        print(f"❌ Status: {result['status']}")
        print(f"💥 Error: {result.get('error', 'Unknown error')}")
    
    print("=" * 60)

if __name__ == "__main__":
    main()