#!/usr/bin/env python3
"""
Enhanced Feature Engineering for Top30 DipMaster Strategy V6
为30币种DipMaster策略构建币种特异性和跨资产增强特征工程系统

Key Features:
1. 币种特异性特征 (Symbol-Specific Features)
2. 跨币种相关性特征 (Cross-Symbol Features) 
3. 市场微观结构特征 (Microstructure Features)
4. 时间特征优化 (Time-Based Features)
5. 动态特征生成 (Dynamic Feature Generation)
6. 智能标签工程 (Smart Label Engineering)

Author: DipMaster Quant Team
Date: 2025-08-17
Version: 6.0.0-Top30Enhanced
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import multiprocessing as mp
from dataclasses import dataclass, field
import ta
import scipy.stats as stats
from sklearn.preprocessing import RobustScaler, StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_regression
import numba
from numba import jit

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class Top30FeatureConfig:
    """Enhanced configuration for Top30 strategy feature engineering"""
    
    # All 30 symbols
    symbols: List[str] = field(default_factory=lambda: [
        # Core coins (Tier S)
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT',
        # Major altcoins (Tier A)
        'ADAUSDT', 'XRPUSDT', 'BNBUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT',
        # Mid-tier altcoins (Tier B)
        'LTCUSDT', 'DOTUSDT', 'ATOMUSDT', 'FILUSDT', 'NEARUSDT', 'ARBUSDT', 'OPUSDT',
        'APTUSDT', 'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'QNTUSDT', 'TRXUSDT', 'VETUSDT', 'XLMUSDT',
        # New additions (Tier B/C)
        'SHIBUSDT', 'DOGEUSDT', 'TONUSDT', 'PEPEUSDT', 'INJUSDT'
    ])
    
    # Symbol categorization for tailored features
    symbol_categories: Dict[str, List[str]] = field(default_factory=lambda: {
        'tier_s': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
        'tier_a': ['ADAUSDT', 'XRPUSDT', 'BNBUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT'],
        'tier_b': ['LTCUSDT', 'DOTUSDT', 'ATOMUSDT', 'FILUSDT', 'NEARUSDT', 'ARBUSDT', 'OPUSDT',
                   'APTUSDT', 'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'QNTUSDT', 'TRXUSDT', 'VETUSDT', 'XLMUSDT'],
        'tier_c': ['SHIBUSDT', 'DOGEUSDT', 'TONUSDT', 'PEPEUSDT', 'INJUSDT'],
        'layer1': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'ATOMUSDT', 'NEARUSDT', 'TONUSDT'],
        'defi': ['UNIUSDT', 'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'LINKUSDT'],
        'exchange': ['BNBUSDT'],
        'meme': ['SHIBUSDT', 'DOGEUSDT', 'PEPEUSDT'],
        'infrastructure': ['LINKUSDT', 'FILUSDT', 'QNTUSDT', 'VETUSDT', 'XLMUSDT'],
        'payments': ['XRPUSDT', 'LTCUSDT', 'TRXUSDT']
    })
    
    # Symbol-specific parameters
    symbol_params: Dict[str, Dict] = field(default_factory=lambda: {
        'volatility_groups': {
            'low_vol': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT'],
            'medium_vol': ['SOLUSDT', 'ADAUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT', 
                          'DOTUSDT', 'ATOMUSDT', 'FILUSDT', 'NEARUSDT', 'ARBUSDT', 'OPUSDT'],
            'high_vol': ['APTUSDT', 'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'QNTUSDT', 'TRXUSDT', 
                        'VETUSDT', 'XLMUSDT', 'SHIBUSDT', 'DOGEUSDT', 'TONUSDT', 'PEPEUSDT', 'INJUSDT']
        },
        'rsi_params': {
            'conservative': {'periods': [14, 21, 50], 'thresholds': [30, 70]},
            'aggressive': {'periods': [7, 14, 30], 'thresholds': [25, 75]},
            'balanced': {'periods': [10, 20, 40], 'thresholds': [28, 72]}
        },
        'ma_params': {
            'short_term': [5, 10, 20],
            'medium_term': [20, 50, 100],
            'long_term': [100, 200, 400]
        }
    })
    
    # Feature generation parameters
    timeframes: List[str] = field(default_factory=lambda: ['5m', '15m', '1h'])
    prediction_horizons: List[int] = field(default_factory=lambda: [1, 3, 6, 12, 24, 36])
    profit_targets: List[float] = field(default_factory=lambda: [0.003, 0.006, 0.008, 0.012, 0.015, 0.020])
    stop_loss: float = 0.004
    max_holding_periods: int = 36
    
    # Advanced feature flags
    enable_symbol_specific: bool = True
    enable_cross_symbol: bool = True
    enable_microstructure: bool = True
    enable_regime_detection: bool = True
    enable_sector_rotation: bool = True
    enable_dynamic_labels: bool = True

class Top30EnhancedFeatureEngineer:
    """
    Top30 Enhanced Feature Engineering Pipeline V6
    为30币种DipMaster策略构建全面的特征工程系统
    """
    
    def __init__(self, config: Optional[Top30FeatureConfig] = None):
        """Initialize the enhanced feature engineer"""
        self.logger = logging.getLogger(__name__)
        self.config = config or Top30FeatureConfig()
        
        # Initialize scalers and transformers
        self.scalers = {
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'quantile': QuantileTransformer(n_quantiles=1000, random_state=42)
        }
        
        # Feature tracking
        self.feature_registry = {}
        self.symbol_feature_stats = {}
        self.cross_symbol_features = {}
        
        # Market state tracking
        self.market_regimes = {}
        self.sector_states = {}
        
    def load_all_symbol_data(self, data_dir: str = 'data/enhanced_market_data') -> Dict[str, pd.DataFrame]:
        """Load data for all 30 symbols"""
        data_dict = {}
        data_path = Path(data_dir)
        
        self.logger.info(f"Loading data for {len(self.config.symbols)} symbols...")
        
        for symbol in self.config.symbols:
            try:
                # Load 5m data as primary timeframe
                file_path = data_path / f"{symbol}_5m_90days.parquet"
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').reset_index(drop=True)
                    
                    if len(df) > 100:
                        data_dict[symbol] = df
                        self.logger.info(f"Loaded {symbol}: {len(df)} rows")
                    else:
                        self.logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                else:
                    self.logger.error(f"Data file not found for {symbol}: {file_path}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load {symbol}: {e}")
        
        self.logger.info(f"Successfully loaded {len(data_dict)} symbols")
        return data_dict
    
    def add_symbol_specific_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add symbol-specific features based on coin category and characteristics
        添加基于币种类别和特征的特异性特征
        """
        try:
            enhanced_df = df.copy()
            
            # Determine symbol characteristics
            volatility_group = self._get_volatility_group(symbol)
            category = self._get_symbol_category(symbol)
            
            # 1. Adaptive RSI based on symbol volatility
            if volatility_group == 'low_vol':
                rsi_periods = [14, 21, 50]
                rsi_oversold, rsi_overbought = 30, 70
            elif volatility_group == 'medium_vol':
                rsi_periods = [10, 20, 40]
                rsi_oversold, rsi_overbought = 28, 72
            else:  # high_vol
                rsi_periods = [7, 14, 30]
                rsi_oversold, rsi_overbought = 25, 75
            
            # Generate adaptive RSI features
            for period in rsi_periods:
                rsi = ta.momentum.RSIIndicator(enhanced_df['close'], window=period).rsi()
                enhanced_df[f'adaptive_rsi_{period}'] = rsi
                enhanced_df[f'rsi_oversold_{period}'] = (rsi <= rsi_oversold).astype(int)
                enhanced_df[f'rsi_overbought_{period}'] = (rsi >= rsi_overbought).astype(int)
                enhanced_df[f'rsi_neutral_{period}'] = ((rsi > rsi_oversold) & (rsi < rsi_overbought)).astype(int)
            
            # 2. Symbol-specific volatility features
            returns = enhanced_df['close'].pct_change()
            
            # Volatility adjusted for symbol type
            vol_multiplier = {'low_vol': 0.8, 'medium_vol': 1.0, 'high_vol': 1.3}[volatility_group]
            
            for window in [10, 20, 50]:
                vol = returns.rolling(window).std() * np.sqrt(window)
                enhanced_df[f'{symbol.lower()}_vol_{window}'] = vol * vol_multiplier
                
                # Volatility percentile within symbol's own history
                enhanced_df[f'{symbol.lower()}_vol_pct_{window}'] = vol.rolling(200).rank(pct=True)
            
            # 3. Category-specific features
            if category in ['defi', 'layer1']:
                # DeFi/Layer1 specific: network activity proxies
                enhanced_df[f'{category}_momentum'] = returns.rolling(20).sum()
                enhanced_df[f'{category}_adoption_proxy'] = enhanced_df['volume'].rolling(50).mean() / enhanced_df['volume'].rolling(200).mean()
                
            elif category == 'meme':
                # Meme coin specific: social sentiment proxies
                enhanced_df['meme_volatility_spike'] = (returns.abs() > returns.rolling(100).quantile(0.95)).astype(int)
                enhanced_df['meme_volume_spike'] = (enhanced_df['volume'] > enhanced_df['volume'].rolling(50).quantile(0.9)).astype(int)
                enhanced_df['meme_social_proxy'] = enhanced_df['meme_volatility_spike'] * enhanced_df['meme_volume_spike']
                
            elif category == 'exchange':
                # Exchange token specific: trading activity
                enhanced_df['exchange_utility'] = enhanced_df['volume'].rolling(20).mean()
                enhanced_df['exchange_dominance'] = enhanced_df['close'] / enhanced_df['close'].rolling(100).max()
                
            elif category == 'payments':
                # Payment token specific: adoption metrics
                enhanced_df['payment_efficiency'] = 1 / (enhanced_df['close'].pct_change().abs().rolling(20).mean() + 1e-8)
                enhanced_df['payment_stability'] = (enhanced_df['close'].rolling(50).std() / enhanced_df['close'].rolling(50).mean()).rolling(20).mean()
            
            # 4. Symbol-specific support/resistance levels
            # Calculate symbol-specific percentile levels
            for pct in [0.1, 0.2, 0.8, 0.9]:
                for window in [50, 100, 200]:
                    level = enhanced_df['close'].rolling(window).quantile(pct)
                    enhanced_df[f'{symbol.lower()}_level_{int(pct*100)}_{window}'] = level
                    enhanced_df[f'{symbol.lower()}_near_level_{int(pct*100)}_{window}'] = (
                        abs(enhanced_df['close'] - level) / enhanced_df['close'] < 0.02
                    ).astype(int)
            
            # 5. Symbol-specific momentum
            # Adaptive momentum based on symbol characteristics
            momentum_windows = {
                'low_vol': [10, 20, 50],
                'medium_vol': [5, 15, 30],
                'high_vol': [3, 10, 20]
            }[volatility_group]
            
            for window in momentum_windows:
                momentum = enhanced_df['close'].pct_change(window)
                enhanced_df[f'{symbol.lower()}_momentum_{window}'] = momentum
                enhanced_df[f'{symbol.lower()}_momentum_rank_{window}'] = momentum.rolling(100).rank(pct=True)
            
            # 6. Symbol-specific market position
            # Position relative to recent range
            for window in [20, 50, 100]:
                high_range = enhanced_df['high'].rolling(window).max()
                low_range = enhanced_df['low'].rolling(window).min()
                range_position = (enhanced_df['close'] - low_range) / (high_range - low_range + 1e-8)
                enhanced_df[f'{symbol.lower()}_range_position_{window}'] = range_position
            
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"Symbol-specific features failed for {symbol}: {e}")
            return df
    
    def add_cross_symbol_intelligence(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Add sophisticated cross-symbol features and sector rotation signals
        添加跨币种智能特征和板块轮动信号
        """
        try:
            if len(all_data) < 10:
                self.logger.warning("Insufficient symbols for cross-symbol features")
                return all_data
            
            self.logger.info("Generating cross-symbol intelligence features...")
            
            # Prepare data matrices
            price_matrices = {}
            volume_matrices = {}
            returns_matrices = {}
            
            # Align timestamps and create matrices
            common_timestamps = None
            for symbol, df in all_data.items():
                if len(df) > 100:
                    df_indexed = df.set_index('timestamp')
                    
                    if common_timestamps is None:
                        common_timestamps = df_indexed.index
                    else:
                        common_timestamps = common_timestamps.intersection(df_indexed.index)
            
            # Build aligned data matrices
            for symbol, df in all_data.items():
                if len(df) > 100:
                    df_indexed = df.set_index('timestamp')
                    df_aligned = df_indexed.reindex(common_timestamps).ffill().bfill()
                    
                    price_matrices[symbol] = df_aligned['close']
                    volume_matrices[symbol] = df_aligned['volume']
                    returns_matrices[symbol] = df_aligned['close'].pct_change()
            
            if not price_matrices:
                return all_data
            
            # Create DataFrames
            prices_df = pd.DataFrame(price_matrices)
            volumes_df = pd.DataFrame(volume_matrices)
            returns_df = pd.DataFrame(returns_matrices)
            
            # Calculate market-wide metrics
            market_returns = returns_df.mean(axis=1)
            market_volatility = returns_df.std(axis=1)
            
            # Sector-specific metrics
            sector_returns = {}
            for sector, symbols in self.config.symbol_categories.items():
                sector_symbols = [s for s in symbols if s in returns_df.columns]
                if len(sector_symbols) > 0:
                    sector_returns[sector] = returns_df[sector_symbols].mean(axis=1)
            
            # Process each symbol with cross-symbol features
            for symbol in all_data:
                if symbol not in returns_df.columns:
                    continue
                
                df = all_data[symbol].copy()
                
                # Align with common timestamps for feature calculation
                df_indexed = df.set_index('timestamp').reindex(common_timestamps)
                symbol_returns = returns_df[symbol]
                symbol_prices = prices_df[symbol]
                
                # 1. Market relative performance
                relative_performance = symbol_returns - market_returns
                df_indexed['market_relative_1h'] = relative_performance.rolling(12).mean()
                df_indexed['market_relative_4h'] = relative_performance.rolling(48).mean()
                df_indexed['market_relative_1d'] = relative_performance.rolling(288).mean()
                
                # Performance percentile rank
                df_indexed['market_rank_1h'] = returns_df.rolling(12).sum().rank(axis=1, pct=True)[symbol]
                df_indexed['market_rank_1d'] = returns_df.rolling(288).sum().rank(axis=1, pct=True)[symbol]
                
                # 2. Sector rotation features
                symbol_sector = self._get_symbol_primary_sector(symbol)
                if symbol_sector and symbol_sector in sector_returns:
                    sector_relative = symbol_returns - sector_returns[symbol_sector]
                    df_indexed['sector_relative_performance'] = sector_relative.rolling(48).mean()
                    df_indexed['sector_leadership'] = (sector_relative.rolling(12).mean() > 0).astype(int)
                    
                    # Cross-sector strength
                    for other_sector, other_returns in sector_returns.items():
                        if other_sector != symbol_sector:
                            cross_sector_strength = sector_returns[symbol_sector] - other_returns
                            df_indexed[f'{symbol_sector}_vs_{other_sector}_strength'] = cross_sector_strength.rolling(48).mean()
                
                # 3. Correlation-based features
                # Rolling correlation with BTC and ETH
                if 'BTCUSDT' in returns_df.columns and symbol != 'BTCUSDT':
                    btc_corr = symbol_returns.rolling(144).corr(returns_df['BTCUSDT'])
                    df_indexed['btc_correlation_12h'] = btc_corr
                    df_indexed['btc_decoupling'] = (btc_corr < 0.3).astype(int)
                
                if 'ETHUSDT' in returns_df.columns and symbol != 'ETHUSDT':
                    eth_corr = symbol_returns.rolling(144).corr(returns_df['ETHUSDT'])
                    df_indexed['eth_correlation_12h'] = eth_corr
                    df_indexed['eth_decoupling'] = (eth_corr < 0.3).astype(int)
                
                # 4. Cross-symbol momentum divergence
                # Find momentum leaders and laggards
                momentum_1h = returns_df.rolling(12).sum()
                momentum_rank = momentum_1h.rank(axis=1, pct=True)
                
                df_indexed['momentum_leadership'] = momentum_rank[symbol]
                df_indexed['is_momentum_leader'] = (momentum_rank[symbol] > 0.8).astype(int)
                df_indexed['is_momentum_laggard'] = (momentum_rank[symbol] < 0.2).astype(int)
                
                # 5. Liquidity and volume cross-analysis
                if symbol in volumes_df.columns:
                    volume_rank = volumes_df.rank(axis=1, pct=True)
                    df_indexed['volume_rank'] = volume_rank[symbol]
                    df_indexed['high_volume_day'] = (volume_rank[symbol] > 0.9).astype(int)
                    
                    # Volume vs price momentum divergence
                    price_momentum = symbol_returns.rolling(12).sum()
                    volume_momentum = volumes_df[symbol].pct_change().rolling(12).sum()
                    df_indexed['volume_price_divergence'] = (
                        (price_momentum > 0) & (volume_momentum < 0)
                    ).astype(int)
                
                # 6. Risk-on/Risk-off regime features
                # Use correlation with risk assets vs safe havens
                risk_assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
                risk_on_proxy = 0
                
                for risk_asset in risk_assets:
                    if risk_asset in returns_df.columns:
                        risk_corr = symbol_returns.rolling(48).corr(returns_df[risk_asset])
                        risk_on_proxy += risk_corr.fillna(0)
                
                if len(risk_assets) > 0:
                    risk_on_proxy /= len(risk_assets)
                    df_indexed['risk_on_correlation'] = risk_on_proxy
                    df_indexed['risk_on_regime'] = (risk_on_proxy > 0.6).astype(int)
                    df_indexed['risk_off_regime'] = (risk_on_proxy < 0.3).astype(int)
                
                # 7. Pair trading opportunities
                # Find highly correlated pairs for mean reversion
                correlations = returns_df.rolling(144).corr()[symbol].dropna()
                high_corr_symbols = correlations[correlations > 0.7].index.tolist()
                
                for corr_symbol in high_corr_symbols[:3]:  # Top 3 correlated
                    if corr_symbol != symbol and corr_symbol in returns_df.columns:
                        price_ratio = symbol_prices / prices_df[corr_symbol]
                        ratio_zscore = (price_ratio - price_ratio.rolling(100).mean()) / price_ratio.rolling(100).std()
                        df_indexed[f'pair_zscore_{corr_symbol[:3]}'] = ratio_zscore
                        df_indexed[f'pair_mean_reversion_{corr_symbol[:3]}'] = (abs(ratio_zscore) > 2).astype(int)
                
                # Reset index and merge back
                df_indexed = df_indexed.reset_index()
                
                # Merge new features back to original dataframe
                merge_cols = ['timestamp'] + [col for col in df_indexed.columns if col not in df.columns]
                all_data[symbol] = df.merge(df_indexed[merge_cols], on='timestamp', how='left')
            
            return all_data
            
        except Exception as e:
            self.logger.error(f"Cross-symbol intelligence failed: {e}")
            return all_data
    
    def add_enhanced_microstructure_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add enhanced market microstructure features
        添加增强的市场微观结构特征
        """
        try:
            enhanced_df = df.copy()
            
            # 1. Order book depth simulation
            # Simulate bid-ask spread using OHLC data
            typical_price = (enhanced_df['high'] + enhanced_df['low'] + enhanced_df['close']) / 3
            spread_proxy = (enhanced_df['high'] - enhanced_df['low']) / typical_price
            enhanced_df['bid_ask_spread_proxy'] = spread_proxy
            
            # 2. Market impact and liquidity
            price_change = enhanced_df['close'].pct_change().abs()
            volume_normalized = enhanced_df['volume'] / enhanced_df['volume'].rolling(20).mean()
            enhanced_df['market_impact'] = price_change / (volume_normalized + 1e-8)
            enhanced_df['liquidity_proxy'] = 1 / (enhanced_df['market_impact'] + 1e-8)
            
            # 3. Order flow imbalance
            # Estimate buy/sell pressure from intrabar data
            enhanced_df['buy_pressure'] = np.where(
                enhanced_df['close'] > enhanced_df['open'],
                enhanced_df['volume'] * (enhanced_df['close'] - enhanced_df['open']) / (enhanced_df['high'] - enhanced_df['low'] + 1e-8),
                0
            )
            
            enhanced_df['sell_pressure'] = np.where(
                enhanced_df['close'] < enhanced_df['open'],
                enhanced_df['volume'] * (enhanced_df['open'] - enhanced_df['close']) / (enhanced_df['high'] - enhanced_df['low'] + 1e-8),
                0
            )
            
            # Order flow imbalance
            for window in [5, 10, 20]:
                buy_flow = enhanced_df['buy_pressure'].rolling(window).sum()
                sell_flow = enhanced_df['sell_pressure'].rolling(window).sum()
                total_flow = buy_flow + sell_flow
                enhanced_df[f'order_flow_imbalance_{window}'] = (buy_flow - sell_flow) / (total_flow + 1e-8)
            
            # 4. Volume-weighted metrics
            for window in [5, 10, 20]:
                # Volume Weighted Average Price
                vwap_num = (typical_price * enhanced_df['volume']).rolling(window).sum()
                vwap_den = enhanced_df['volume'].rolling(window).sum()
                vwap = vwap_num / vwap_den
                enhanced_df[f'vwap_{window}'] = vwap
                enhanced_df[f'vwap_deviation_{window}'] = (enhanced_df['close'] - vwap) / vwap
                
                # Volume profile approximation
                price_levels = pd.qcut(enhanced_df['close'].rolling(window*5), q=10, duplicates='drop')
                volume_profile = enhanced_df.groupby(price_levels)['volume'].sum()
                enhanced_df[f'volume_at_price_{window}'] = enhanced_df['close'].map(
                    lambda x: volume_profile.get(x, volume_profile.mean()) if pd.notnull(x) else volume_profile.mean()
                )
            
            # 5. Transaction cost estimation
            # Estimate slippage based on market conditions
            volatility = enhanced_df['close'].pct_change().rolling(20).std()
            enhanced_df['estimated_slippage'] = spread_proxy + volatility * 0.5
            
            # 6. Microstructure patterns
            # Detect reversal patterns from intrabar data
            enhanced_df['upper_wick'] = enhanced_df['high'] - np.maximum(enhanced_df['open'], enhanced_df['close'])
            enhanced_df['lower_wick'] = np.minimum(enhanced_df['open'], enhanced_df['close']) - enhanced_df['low']
            enhanced_df['body_size'] = abs(enhanced_df['close'] - enhanced_df['open'])
            enhanced_df['candle_range'] = enhanced_df['high'] - enhanced_df['low']
            
            # Wick ratios
            enhanced_df['upper_wick_ratio'] = enhanced_df['upper_wick'] / (enhanced_df['candle_range'] + 1e-8)
            enhanced_df['lower_wick_ratio'] = enhanced_df['lower_wick'] / (enhanced_df['candle_range'] + 1e-8)
            enhanced_df['body_ratio'] = enhanced_df['body_size'] / (enhanced_df['candle_range'] + 1e-8)
            
            # Pattern detection
            enhanced_df['hammer_pattern'] = (
                (enhanced_df['lower_wick_ratio'] > 0.6) & 
                (enhanced_df['body_ratio'] < 0.3)
            ).astype(int)
            
            enhanced_df['shooting_star_pattern'] = (
                (enhanced_df['upper_wick_ratio'] > 0.6) & 
                (enhanced_df['body_ratio'] < 0.3)
            ).astype(int)
            
            enhanced_df['doji_pattern'] = (enhanced_df['body_ratio'] < 0.1).astype(int)
            
            # 7. Time-based microstructure effects
            if 'timestamp' in enhanced_df.columns:
                enhanced_df['timestamp'] = pd.to_datetime(enhanced_df['timestamp'])
                enhanced_df['hour'] = enhanced_df['timestamp'].dt.hour
                enhanced_df['day_of_week'] = enhanced_df['timestamp'].dt.dayofweek
                
                # Trading session effects
                enhanced_df['asian_session'] = ((enhanced_df['hour'] >= 0) & (enhanced_df['hour'] < 8)).astype(int)
                enhanced_df['european_session'] = ((enhanced_df['hour'] >= 8) & (enhanced_df['hour'] < 16)).astype(int)
                enhanced_df['us_session'] = ((enhanced_df['hour'] >= 16) & (enhanced_df['hour'] < 24)).astype(int)
                
                # Weekend effect
                enhanced_df['weekend_effect'] = (enhanced_df['day_of_week'] >= 5).astype(int)
            
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"Enhanced microstructure features failed for {symbol}: {e}")
            return df
    
    def add_dynamic_label_engineering(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Add dynamic label engineering with symbol-specific optimizations
        添加带有币种特异性优化的动态标签工程
        """
        try:
            enhanced_df = df.copy()
            
            # Get symbol characteristics for label customization
            volatility_group = self._get_volatility_group(symbol)
            category = self._get_symbol_category(symbol)
            
            # Adjust profit targets based on symbol volatility
            vol_multipliers = {'low_vol': 0.8, 'medium_vol': 1.0, 'high_vol': 1.5}
            vol_mult = vol_multipliers[volatility_group]
            
            adjusted_targets = [target * vol_mult for target in self.config.profit_targets]
            adjusted_stop_loss = self.config.stop_loss * vol_mult
            
            # 1. Multi-horizon return labels
            for horizon in self.config.prediction_horizons:
                future_return = enhanced_df['close'].pct_change(periods=horizon).shift(-horizon)
                enhanced_df[f'future_return_{horizon}p'] = future_return
                
                # Symbol-specific profitability labels
                for i, target in enumerate(adjusted_targets):
                    enhanced_df[f'hits_target_{i}_{horizon}p'] = (future_return >= target).astype(int)
                
                # Risk labels
                enhanced_df[f'hits_stop_loss_{horizon}p'] = (future_return <= -adjusted_stop_loss).astype(int)
                
                # Risk-adjusted returns
                if 'close' in enhanced_df.columns:
                    vol_20 = enhanced_df['close'].pct_change().rolling(20).std()
                    risk_adj_return = future_return / (vol_20.shift(-horizon) + 1e-8)
                    enhanced_df[f'risk_adj_return_{horizon}p'] = risk_adj_return
            
            # 2. 15-minute boundary optimization for DipMaster
            if 'timestamp' in enhanced_df.columns:
                enhanced_df['timestamp'] = pd.to_datetime(enhanced_df['timestamp'])
                enhanced_df['minute'] = enhanced_df['timestamp'].dt.minute
                
                boundary_minutes = [15, 30, 45, 0]
                
                for horizon in [6, 12, 18, 24]:  # 30min, 1h, 1.5h, 2h
                    future_minute = enhanced_df['minute'].shift(-horizon)
                    is_boundary_exit = future_minute.isin(boundary_minutes)
                    enhanced_df[f'boundary_exit_{horizon}p'] = is_boundary_exit.astype(int)
                    
                    # Boundary-adjusted returns (bonus for boundary timing)
                    future_return = enhanced_df[f'future_return_{horizon}p']
                    boundary_bonus = 0.001 * vol_mult  # Volatility-adjusted bonus
                    enhanced_df[f'boundary_adj_return_{horizon}p'] = future_return + (is_boundary_exit * boundary_bonus)
            
            # 3. Category-specific labels
            if category == 'meme':
                # Meme coins: focus on extreme movements
                for horizon in [1, 3, 6]:
                    future_return = enhanced_df[f'future_return_{horizon}p']
                    enhanced_df[f'meme_explosion_{horizon}p'] = (future_return > 0.05).astype(int)  # 5% spike
                    enhanced_df[f'meme_crash_{horizon}p'] = (future_return < -0.05).astype(int)    # 5% crash
                    
            elif category in ['defi', 'layer1']:
                # DeFi/Layer1: focus on sustainable growth
                for horizon in [12, 24, 36]:
                    future_return = enhanced_df[f'future_return_{horizon}p']
                    enhanced_df[f'sustainable_growth_{horizon}p'] = (
                        (future_return > 0.01) & (future_return < 0.08)
                    ).astype(int)
                    
            elif category == 'exchange':
                # Exchange tokens: focus on utility correlation
                for horizon in [6, 12, 24]:
                    future_return = enhanced_df[f'future_return_{horizon}p']
                    volume_growth = enhanced_df['volume'].pct_change(horizon).shift(-horizon)
                    enhanced_df[f'utility_correlated_growth_{horizon}p'] = (
                        (future_return > 0.005) & (volume_growth > 0.1)
                    ).astype(int)
            
            # 4. Optimal exit timing analysis
            max_periods = min(36, len(enhanced_df) // 10)
            
            # Calculate best exit within maximum holding period
            exit_returns = pd.DataFrame(index=enhanced_df.index)
            for exit_period in range(1, max_periods + 1):
                exit_price = enhanced_df['close'].shift(-exit_period)
                exit_return = (exit_price - enhanced_df['close']) / enhanced_df['close']
                exit_returns[f'exit_{exit_period}'] = exit_return
            
            # Find optimal exit
            enhanced_df['optimal_exit_period'] = exit_returns.idxmax(axis=1).str.extract(r'(\d+)').astype(float)
            enhanced_df['optimal_return'] = exit_returns.max(axis=1)
            
            # Find optimal boundary exit
            if 'minute' in enhanced_df.columns:
                boundary_returns = exit_returns.copy()
                for col in boundary_returns.columns:
                    period = int(col.split('_')[1])
                    future_minute = enhanced_df['minute'].shift(-period)
                    is_boundary = future_minute.isin(boundary_minutes)
                    boundary_returns[col] = boundary_returns[col].where(is_boundary, -np.inf)
                
                enhanced_df['optimal_boundary_exit_period'] = boundary_returns.idxmax(axis=1).str.extract('(\d+)').astype(float)
                enhanced_df['optimal_boundary_return'] = boundary_returns.max(axis=1).replace(-np.inf, np.nan)
            
            # 5. Time-to-target labels
            for target in adjusted_targets[:3]:  # First 3 targets
                time_to_target = self._calculate_time_to_target(enhanced_df, target, max_periods)
                enhanced_df[f'time_to_target_{target:.1%}'] = time_to_target
            
            # 6. Multi-class return classification
            main_return = enhanced_df.get('future_return_12p', enhanced_df.get('future_return_6p'))
            if main_return is not None:
                # Volatility-adjusted classification thresholds
                loss_threshold = -adjusted_stop_loss
                small_profit = adjusted_targets[0]
                good_profit = adjusted_targets[2]
                excellent_profit = adjusted_targets[4]
                
                conditions = [
                    (main_return <= loss_threshold),
                    ((main_return > loss_threshold) & (main_return <= 0)),
                    ((main_return > 0) & (main_return < small_profit)),
                    ((main_return >= small_profit) & (main_return < good_profit)),
                    ((main_return >= good_profit) & (main_return < excellent_profit)),
                    (main_return >= excellent_profit)
                ]
                labels = [0, 1, 2, 3, 4, 5]
                enhanced_df['return_class'] = np.select(conditions, labels, default=1)
                
                # Primary target labels
                enhanced_df['target_return'] = main_return
                enhanced_df['target_binary'] = (main_return > 0).astype(int)
                enhanced_df['target_profitable'] = (main_return >= small_profit).astype(int)
            
            return enhanced_df
            
        except Exception as e:
            self.logger.error(f"Dynamic label engineering failed for {symbol}: {e}")
            return df
    
    def _get_volatility_group(self, symbol: str) -> str:
        """Get volatility group for symbol"""
        for group, symbols in self.config.symbol_params['volatility_groups'].items():
            if symbol in symbols:
                return group
        return 'medium_vol'  # default
    
    def _get_symbol_category(self, symbol: str) -> str:
        """Get primary category for symbol"""
        for category, symbols in self.config.symbol_categories.items():
            if symbol in symbols and category not in ['tier_s', 'tier_a', 'tier_b', 'tier_c']:
                return category
        return 'other'
    
    def _get_symbol_primary_sector(self, symbol: str) -> str:
        """Get primary sector for symbol"""
        priority_sectors = ['layer1', 'defi', 'meme', 'exchange', 'infrastructure', 'payments']
        for sector in priority_sectors:
            if symbol in self.config.symbol_categories.get(sector, []):
                return sector
        return None
    
    def _calculate_time_to_target(self, df: pd.DataFrame, target_return: float, max_periods: int) -> pd.Series:
        """Calculate time to reach target return"""
        time_to_target = pd.Series(index=df.index, dtype=float)
        
        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            if pd.isna(current_price):
                continue
                
            target_price = current_price * (1 + target_return)
            
            # Look forward to find when target is reached
            for j in range(i + 1, min(i + max_periods + 1, len(df))):
                if df['close'].iloc[j] >= target_price:
                    time_to_target.iloc[i] = j - i
                    break
            
            # If target not reached, set to max periods
            if pd.isna(time_to_target.iloc[i]):
                time_to_target.iloc[i] = max_periods
        
        return time_to_target
    
    def clean_and_validate_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Clean and validate features with enhanced quality control"""
        try:
            # Handle infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Get numeric columns (exclude timestamp and target columns)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols 
                          if not col.startswith('target') 
                          and not col.startswith('future_')
                          and not col.startswith('optimal_')
                          and 'timestamp' not in col.lower()]
            
            # Intelligent NaN handling
            for col in feature_cols:
                nan_pct = df[col].isnull().sum() / len(df)
                
                if nan_pct > 0.5:  # High NaN percentage
                    df[col] = df[col].fillna(0)
                elif nan_pct > 0.1:  # Medium NaN percentage
                    df[col] = df[col].fillna(df[col].median())
                else:  # Low NaN percentage
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Robust outlier treatment
            for col in feature_cols:
                if col in df.columns:
                    # Use robust statistics for outlier detection
                    Q1 = df[col].quantile(0.01)
                    Q3 = df[col].quantile(0.99)
                    IQR = Q3 - Q1
                    
                    # Conservative outlier bounds
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Clip outliers
                    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
            
            # Validate feature stability
            feature_stats = {}
            for col in feature_cols:
                if col in df.columns and len(df[col].dropna()) > 10:
                    stats_dict = {
                        'mean': df[col].mean(),
                        'std': df[col].std(),
                        'min': df[col].min(),
                        'max': df[col].max(),
                        'null_pct': df[col].isnull().sum() / len(df)
                    }
                    feature_stats[col] = stats_dict
            
            self.symbol_feature_stats[symbol] = feature_stats
            
            return df
            
        except Exception as e:
            self.logger.error(f"Feature cleaning failed for {symbol}: {e}")
            return df
    
    def generate_top30_features(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
        """
        Generate comprehensive feature set for all 30 symbols
        为所有30币种生成全面的特征集
        """
        try:
            self.logger.info("Starting Top30 enhanced feature engineering...")
            start_time = time.time()
            
            # Load all symbol data
            all_data = self.load_all_symbol_data()
            
            if len(all_data) < 10:
                raise ValueError(f"Insufficient symbols loaded: {len(all_data)}")
            
            # Initialize results
            results = {
                'symbols_processed': list(all_data.keys()),
                'total_symbols': len(all_data),
                'feature_categories': {},
                'processing_times': {},
                'validation_results': {},
                'errors': []
            }
            
            # Step 1: Add symbol-specific features
            self.logger.info("Adding symbol-specific features...")
            step_start = time.time()
            
            for symbol in all_data:
                try:
                    all_data[symbol] = self.add_symbol_specific_features(all_data[symbol], symbol)
                except Exception as e:
                    self.logger.error(f"Symbol-specific features failed for {symbol}: {e}")
                    results['errors'].append(f"{symbol} symbol-specific: {str(e)}")
            
            results['processing_times']['symbol_specific'] = time.time() - step_start
            
            # Step 2: Add enhanced microstructure features
            if self.config.enable_microstructure:
                self.logger.info("Adding enhanced microstructure features...")
                step_start = time.time()
                
                for symbol in all_data:
                    try:
                        all_data[symbol] = self.add_enhanced_microstructure_features(all_data[symbol], symbol)
                    except Exception as e:
                        self.logger.error(f"Microstructure features failed for {symbol}: {e}")
                        results['errors'].append(f"{symbol} microstructure: {str(e)}")
                
                results['processing_times']['microstructure'] = time.time() - step_start
            
            # Step 3: Add cross-symbol intelligence
            if self.config.enable_cross_symbol:
                self.logger.info("Adding cross-symbol intelligence features...")
                step_start = time.time()
                
                try:
                    all_data = self.add_cross_symbol_intelligence(all_data)
                except Exception as e:
                    self.logger.error(f"Cross-symbol features failed: {e}")
                    results['errors'].append(f"Cross-symbol: {str(e)}")
                
                results['processing_times']['cross_symbol'] = time.time() - step_start
            
            # Step 4: Add dynamic label engineering
            if self.config.enable_dynamic_labels:
                self.logger.info("Adding dynamic label engineering...")
                step_start = time.time()
                
                for symbol in all_data:
                    try:
                        all_data[symbol] = self.add_dynamic_label_engineering(all_data[symbol], symbol)
                    except Exception as e:
                        self.logger.error(f"Label engineering failed for {symbol}: {e}")
                        results['errors'].append(f"{symbol} labels: {str(e)}")
                
                results['processing_times']['label_engineering'] = time.time() - step_start
            
            # Step 5: Clean and validate features
            self.logger.info("Cleaning and validating features...")
            step_start = time.time()
            
            for symbol in all_data:
                try:
                    all_data[symbol] = self.clean_and_validate_features(all_data[symbol], symbol)
                except Exception as e:
                    self.logger.error(f"Feature cleaning failed for {symbol}: {e}")
                    results['errors'].append(f"{symbol} cleaning: {str(e)}")
            
            results['processing_times']['cleaning'] = time.time() - step_start
            
            # Step 6: Calculate feature statistics
            if all_data:
                sample_df = next(iter(all_data.values()))
                results['total_features'] = len(sample_df.columns)
                
                # Categorize features
                feature_categories = {
                    'symbol_specific': 0,
                    'cross_symbol': 0,
                    'microstructure': 0,
                    'technical': 0,
                    'labels': 0,
                    'other': 0
                }
                
                for col in sample_df.columns:
                    if any(x in col.lower() for x in ['adaptive', 'symbol', 'tier', 'category']):
                        feature_categories['symbol_specific'] += 1
                    elif any(x in col.lower() for x in ['market_', 'sector_', 'correlation', 'rank', 'relative']):
                        feature_categories['cross_symbol'] += 1
                    elif any(x in col.lower() for x in ['vwap', 'flow', 'pressure', 'spread', 'liquidity', 'impact']):
                        feature_categories['microstructure'] += 1
                    elif any(x in col.lower() for x in ['rsi', 'macd', 'momentum', 'sma', 'ema', 'bb_', 'atr']):
                        feature_categories['technical'] += 1
                    elif any(x in col.lower() for x in ['target', 'future_', 'hits_', 'time_to_', 'optimal_', 'return_class']):
                        feature_categories['labels'] += 1
                    else:
                        feature_categories['other'] += 1
                
                results['feature_categories'] = feature_categories
            
            # Total processing time
            results['total_processing_time'] = time.time() - start_time
            
            self.logger.info(f"Top30 feature engineering completed in {results['total_processing_time']:.1f}s")
            self.logger.info(f"Generated {results.get('total_features', 0)} features for {results['total_symbols']} symbols")
            
            return all_data, results
            
        except Exception as e:
            self.logger.error(f"Top30 feature generation failed: {e}")
            return {}, {'error': str(e), 'symbols_processed': []}

def main():
    """Main demonstration function"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Enhanced Feature Engineering for Top30 DipMaster Strategy V6")
    print("=" * 80)
    print("🎯 Target: 30币种特异性和跨资产增强特征工程")
    print("\n📊 Feature Categories:")
    print("1. 币种特异性特征 (Symbol-Specific Features)")
    print("   - 自适应RSI和移动平均参数")
    print("   - 币种类别特化指标 (DeFi, Layer1, Meme等)")
    print("   - 波动率调整技术指标")
    print("   - 币种特有支撑阻力位")
    print("\n2. 跨币种智能特征 (Cross-Symbol Intelligence)")
    print("   - 市场相对表现和排名")
    print("   - 板块轮动信号")
    print("   - 相关性分析和脱钩检测")
    print("   - 风险偏好制度识别")
    print("\n3. 增强微观结构 (Enhanced Microstructure)")
    print("   - 订单簿深度模拟")
    print("   - 订单流失衡指标")
    print("   - 成交量加权价格偏离")
    print("   - 交易成本估算")
    print("\n4. 动态标签工程 (Dynamic Label Engineering)")
    print("   - 币种特异性盈利目标")
    print("   - 15分钟边界优化")
    print("   - 类别特化标签 (Meme爆发, DeFi增长等)")
    print("   - 最优退出时机分析")
    print("\n🛡️ Quality Assurance:")
    print("- 30币种数据一致性验证")
    print("- 智能缺失值处理")
    print("- 稳健异常值检测")
    print("- 特征稳定性监控")
    print("=" * 80)

if __name__ == "__main__":
    main()