#!/usr/bin/env python3
"""
Ultra Enhanced Top30 Feature Engineering for DipMaster Strategy
超级增强版30币种特征工程系统 - 包含跨币种相关性和市场微观结构特征

Features:
1. 币种特异性技术指标
2. 跨币种相关性和排名特征
3. 市场微观结构特征
4. 板块轮动信号
5. 动态标签工程
6. 质量控制验证
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
from scipy import stats
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings('ignore')

class UltraEnhancedTop30FeatureEngineer:
    """Ultra Enhanced Top30 Feature Engineering System"""
    
    def __init__(self):
        self.symbols = []
        self.symbol_categories = {
            'tier_s': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
            'layer1': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'AVAXUSDT', 'DOTUSDT', 'ATOMUSDT', 'NEARUSDT', 'TONUSDT'],
            'defi': ['UNIUSDT', 'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'LINKUSDT'],
            'meme': ['SHIBUSDT', 'DOGEUSDT', 'PEPEUSDT'],
            'exchange': ['BNBUSDT'],
            'payments': ['XRPUSDT', 'LTCUSDT', 'TRXUSDT'],
            'infrastructure': ['LINKUSDT', 'FILUSDT', 'QNTUSDT', 'VETUSDT', 'XLMUSDT']
        }
        self.volatility_groups = {
            'low_vol': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'LTCUSDT'],
            'medium_vol': ['SOLUSDT', 'ADAUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT', 
                          'DOTUSDT', 'ATOMUSDT', 'FILUSDT', 'NEARUSDT', 'ARBUSDT', 'OPUSDT'],
            'high_vol': ['APTUSDT', 'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'QNTUSDT', 'TRXUSDT', 
                        'VETUSDT', 'XLMUSDT', 'SHIBUSDT', 'DOGEUSDT', 'TONUSDT', 'PEPEUSDT', 'INJUSDT']
        }
        
    def load_all_data(self):
        """Load all 30 symbol data"""
        print("Loading 30 symbol data...")
        data_files = glob.glob('data/enhanced_market_data/*_5m_90days.parquet')
        symbols = [Path(f).name.split('_')[0] for f in data_files]
        symbols = sorted(list(set(symbols)))
        
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
                    print(f"  Loaded {symbol}: {len(df)} rows")
        
        self.symbols = list(all_data.keys())
        print(f"Successfully loaded {len(all_data)} symbols")
        return all_data
    
    def get_symbol_category(self, symbol):
        """Get symbol primary category"""
        for category, symbols in self.symbol_categories.items():
            if symbol in symbols:
                return category
        return 'other'
    
    def get_volatility_group(self, symbol):
        """Get symbol volatility group"""
        for group, symbols in self.volatility_groups.items():
            if symbol in symbols:
                return group
        return 'medium_vol'
    
    def add_symbol_specific_features(self, df, symbol):
        """Add comprehensive symbol-specific features"""
        enhanced_df = df.copy()
        
        # Get symbol characteristics
        vol_group = self.get_volatility_group(symbol)
        category = self.get_symbol_category(symbol)
        
        # Adaptive parameters based on volatility
        if vol_group == 'low_vol':
            rsi_period, ma_fast, ma_slow = 21, 10, 30
            bb_period, bb_std = 20, 2.0
            vol_mult = 0.8
        elif vol_group == 'medium_vol':
            rsi_period, ma_fast, ma_slow = 14, 8, 20
            bb_period, bb_std = 16, 1.8
            vol_mult = 1.0
        else:  # high_vol
            rsi_period, ma_fast, ma_slow = 10, 5, 15
            bb_period, bb_std = 12, 1.5
            vol_mult = 1.3
        
        # 1. Adaptive Technical Indicators
        enhanced_df['rsi'] = ta.momentum.RSIIndicator(enhanced_df['close'], window=rsi_period).rsi()
        enhanced_df['rsi_oversold'] = (enhanced_df['rsi'] <= 30 / vol_mult).astype(int)
        enhanced_df['rsi_overbought'] = (enhanced_df['rsi'] >= 70 * vol_mult).astype(int)
        enhanced_df['rsi_ma'] = enhanced_df['rsi'].rolling(5).mean()
        enhanced_df['rsi_divergence'] = enhanced_df['rsi'] - enhanced_df['rsi_ma']
        
        # Moving averages
        enhanced_df['sma_fast'] = ta.trend.SMAIndicator(enhanced_df['close'], window=ma_fast).sma_indicator()
        enhanced_df['sma_slow'] = ta.trend.SMAIndicator(enhanced_df['close'], window=ma_slow).sma_indicator()
        enhanced_df['ema_fast'] = ta.trend.EMAIndicator(enhanced_df['close'], window=ma_fast).ema_indicator()
        enhanced_df['ema_slow'] = ta.trend.EMAIndicator(enhanced_df['close'], window=ma_slow).ema_indicator()
        
        # MA signals
        enhanced_df['ma_golden_cross'] = (enhanced_df['sma_fast'] > enhanced_df['sma_slow']).astype(int)
        enhanced_df['price_vs_sma_fast'] = (enhanced_df['close'] - enhanced_df['sma_fast']) / enhanced_df['sma_fast']
        enhanced_df['price_vs_sma_slow'] = (enhanced_df['close'] - enhanced_df['sma_slow']) / enhanced_df['sma_slow']
        
        # 2. Bollinger Bands
        bb = ta.volatility.BollingerBands(enhanced_df['close'], window=bb_period, window_dev=bb_std)
        enhanced_df['bb_upper'] = bb.bollinger_hband()
        enhanced_df['bb_middle'] = bb.bollinger_mavg()
        enhanced_df['bb_lower'] = bb.bollinger_lband()
        enhanced_df['bb_position'] = bb.bollinger_pband()
        enhanced_df['bb_width'] = bb.bollinger_wband()
        enhanced_df['bb_squeeze'] = (enhanced_df['bb_width'] < enhanced_df['bb_width'].rolling(50).quantile(0.2)).astype(int)
        
        # 3. MACD with different parameters
        macd = ta.trend.MACD(enhanced_df['close'], window_fast=ma_fast, window_slow=ma_slow, window_sign=9)
        enhanced_df['macd'] = macd.macd()
        enhanced_df['macd_signal'] = macd.macd_signal()
        enhanced_df['macd_histogram'] = macd.macd_diff()
        enhanced_df['macd_cross'] = (enhanced_df['macd'] > enhanced_df['macd_signal']).astype(int)
        
        # 4. Volume Analysis
        enhanced_df['volume_sma'] = enhanced_df['volume'].rolling(20).mean()
        enhanced_df['volume_ratio'] = enhanced_df['volume'] / enhanced_df['volume_sma']
        enhanced_df['volume_spike'] = (enhanced_df['volume_ratio'] > 2.0).astype(int)
        enhanced_df['volume_dry_up'] = (enhanced_df['volume_ratio'] < 0.5).astype(int)
        
        # On-Balance Volume
        enhanced_df['obv'] = ta.volume.OnBalanceVolumeIndicator(enhanced_df['close'], enhanced_df['volume']).on_balance_volume()
        enhanced_df['obv_ma'] = enhanced_df['obv'].rolling(20).mean()
        enhanced_df['obv_divergence'] = (enhanced_df['obv'] - enhanced_df['obv_ma']) / (enhanced_df['obv_ma'] + 1e-8)
        
        # 5. Volatility and ATR
        enhanced_df['atr'] = ta.volatility.AverageTrueRange(enhanced_df['high'], enhanced_df['low'], enhanced_df['close'], window=14).average_true_range()
        enhanced_df['atr_ratio'] = enhanced_df['atr'] / enhanced_df['close']
        
        returns = enhanced_df['close'].pct_change()
        enhanced_df['volatility_20'] = returns.rolling(20).std() * np.sqrt(20)
        enhanced_df['volatility_percentile'] = enhanced_df['volatility_20'].rolling(100).rank(pct=True)
        
        # 6. Support and Resistance
        for window in [20, 50]:
            enhanced_df[f'resistance_{window}'] = enhanced_df['high'].rolling(window).max()
            enhanced_df[f'support_{window}'] = enhanced_df['low'].rolling(window).min()
            enhanced_df[f'resistance_distance_{window}'] = (enhanced_df[f'resistance_{window}'] - enhanced_df['close']) / enhanced_df['close']
            enhanced_df[f'support_distance_{window}'] = (enhanced_df['close'] - enhanced_df[f'support_{window}']) / enhanced_df['close']
        
        # 7. Category-specific features
        if category == 'meme':
            # Meme coin specific features
            enhanced_df['meme_volatility_spike'] = (enhanced_df['volatility_20'] > enhanced_df['volatility_20'].rolling(50).quantile(0.9)).astype(int)
            enhanced_df['meme_volume_spike'] = (enhanced_df['volume_ratio'] > 3.0).astype(int)
            enhanced_df['meme_momentum'] = returns.rolling(5).sum()
            
        elif category in ['layer1', 'defi']:
            # Layer1/DeFi specific features
            enhanced_df['network_activity_proxy'] = enhanced_df['volume_ratio'] * enhanced_df['volatility_percentile']
            enhanced_df['adoption_trend'] = enhanced_df['volume_sma'].pct_change(20)
            
        elif category == 'exchange':
            # Exchange token features
            enhanced_df['utility_score'] = enhanced_df['volume_sma'] / enhanced_df['volume_sma'].rolling(100).mean()
            
        # 8. Symbol identifier features
        for cat, symbols in self.symbol_categories.items():
            enhanced_df[f'is_{cat}'] = 1 if symbol in symbols else 0
        
        return enhanced_df
    
    def add_cross_symbol_features(self, all_data):
        """Add comprehensive cross-symbol features"""
        print("Adding cross-symbol features...")
        
        if len(all_data) < 10:
            return all_data
        
        # Prepare aligned data matrices
        common_timestamps = None
        for symbol, df in all_data.items():
            if common_timestamps is None:
                common_timestamps = set(df['timestamp'])
            else:
                common_timestamps = common_timestamps.intersection(set(df['timestamp']))
        
        common_timestamps = sorted(list(common_timestamps))
        
        # Create aligned matrices
        price_matrix = {}
        volume_matrix = {}
        returns_matrix = {}
        
        for symbol, df in all_data.items():
            df_aligned = df.set_index('timestamp').reindex(common_timestamps).ffill().bfill()
            price_matrix[symbol] = df_aligned['close']
            volume_matrix[symbol] = df_aligned['volume']
            returns_matrix[symbol] = df_aligned['close'].pct_change()
        
        price_df = pd.DataFrame(price_matrix)
        volume_df = pd.DataFrame(volume_matrix)
        returns_df = pd.DataFrame(returns_matrix)
        
        # Market-wide metrics
        market_return = returns_df.mean(axis=1)
        market_volatility = returns_df.std(axis=1)
        
        # Sector metrics
        sector_returns = {}
        for sector, symbols in self.symbol_categories.items():
            if sector not in ['other']:
                sector_symbols = [s for s in symbols if s in returns_df.columns]
                if len(sector_symbols) > 0:
                    sector_returns[sector] = returns_df[sector_symbols].mean(axis=1)
        
        # Add cross-symbol features to each symbol
        for symbol in all_data:
            if symbol not in returns_df.columns:
                continue
                
            df = all_data[symbol].copy()
            df_indexed = df.set_index('timestamp').reindex(common_timestamps)
            
            symbol_returns = returns_df[symbol]
            symbol_prices = price_df[symbol]
            
            # 1. Market relative performance
            relative_performance = symbol_returns - market_return
            df_indexed['market_relative_1h'] = relative_performance.rolling(12).mean()
            df_indexed['market_relative_4h'] = relative_performance.rolling(48).mean()
            df_indexed['market_relative_1d'] = relative_performance.rolling(288).mean()
            
            # Performance momentum
            df_indexed['relative_momentum'] = relative_performance.rolling(12).sum()
            df_indexed['relative_acceleration'] = df_indexed['relative_momentum'].diff()
            
            # 2. Market ranking
            returns_1h = returns_df.rolling(12).sum()
            returns_4h = returns_df.rolling(48).sum()
            returns_1d = returns_df.rolling(288).sum()
            
            df_indexed['market_rank_1h'] = returns_1h.rank(axis=1, pct=True)[symbol]
            df_indexed['market_rank_4h'] = returns_4h.rank(axis=1, pct=True)[symbol]
            df_indexed['market_rank_1d'] = returns_1d.rank(axis=1, pct=True)[symbol]
            
            # Ranking momentum
            df_indexed['rank_momentum'] = df_indexed['market_rank_1h'] - df_indexed['market_rank_4h']
            df_indexed['is_top_performer'] = (df_indexed['market_rank_1h'] > 0.8).astype(int)
            df_indexed['is_laggard'] = (df_indexed['market_rank_1h'] < 0.2).astype(int)
            
            # 3. Correlation analysis
            corr_window = 144  # 12 hours
            
            # Market correlation
            df_indexed['market_correlation'] = symbol_returns.rolling(corr_window).corr(market_return)
            df_indexed['correlation_stability'] = df_indexed['market_correlation'].rolling(48).std()
            
            # Beta calculation
            market_var = market_return.rolling(corr_window).var()
            covariance = symbol_returns.rolling(corr_window).cov(market_return)
            df_indexed['beta'] = covariance / (market_var + 1e-8)
            df_indexed['beta_stability'] = df_indexed['beta'].rolling(48).std()
            
            # Major coin correlations
            if 'BTCUSDT' in returns_df.columns and symbol != 'BTCUSDT':
                df_indexed['btc_correlation'] = symbol_returns.rolling(corr_window).corr(returns_df['BTCUSDT'])
                df_indexed['btc_decoupling'] = (abs(df_indexed['btc_correlation']) < 0.3).astype(int)
            else:
                df_indexed['btc_correlation'] = 0.5
                df_indexed['btc_decoupling'] = 0
                
            if 'ETHUSDT' in returns_df.columns and symbol != 'ETHUSDT':
                df_indexed['eth_correlation'] = symbol_returns.rolling(corr_window).corr(returns_df['ETHUSDT'])
                df_indexed['eth_decoupling'] = (abs(df_indexed['eth_correlation']) < 0.3).astype(int)
            else:
                df_indexed['eth_correlation'] = 0.5
                df_indexed['eth_decoupling'] = 0
            
            # 4. Sector rotation features
            symbol_sector = self.get_symbol_category(symbol)
            if symbol_sector in sector_returns:
                sector_relative = symbol_returns - sector_returns[symbol_sector]
                df_indexed['sector_relative_performance'] = sector_relative.rolling(48).mean()
                df_indexed['sector_leadership'] = (sector_relative.rolling(12).mean() > 0).astype(int)
                
                # Cross-sector strength
                for other_sector, other_returns in sector_returns.items():
                    if other_sector != symbol_sector:
                        strength = sector_returns[symbol_sector] - other_returns
                        df_indexed[f'vs_{other_sector}_strength'] = strength.rolling(48).mean()
            
            # 5. Volume analysis
            if symbol in volume_df.columns:
                volume_rank = volume_df.rank(axis=1, pct=True)
                df_indexed['volume_rank'] = volume_rank[symbol]
                df_indexed['high_volume_event'] = (volume_rank[symbol] > 0.95).astype(int)
                
                # Volume vs price divergence
                volume_momentum = volume_df[symbol].pct_change().rolling(12).sum()
                price_momentum = symbol_returns.rolling(12).sum()
                
                divergence_signal = ((price_momentum > 0) & (volume_momentum < 0)) | ((price_momentum < 0) & (volume_momentum > 0))
                df_indexed['volume_price_divergence'] = divergence_signal.astype(int)
            
            # 6. Risk regime features
            # Calculate market stress
            market_stress = (-returns_df.rolling(48).sum().min(axis=1)).rolling(12).mean()
            df_indexed['market_stress_level'] = market_stress / market_stress.rolling(100).mean()
            df_indexed['high_stress_regime'] = (df_indexed['market_stress_level'] > 1.5).astype(int)
            
            # Risk-on/Risk-off
            risk_assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
            risk_correlations = []
            for risk_asset in risk_assets:
                if risk_asset in returns_df.columns:
                    corr = symbol_returns.rolling(48).corr(returns_df[risk_asset])
                    risk_correlations.append(corr.fillna(0))
            
            if risk_correlations:
                avg_risk_corr = np.mean(risk_correlations, axis=0)
                df_indexed['risk_asset_correlation'] = avg_risk_corr
                df_indexed['risk_on_regime'] = (avg_risk_corr > 0.6).astype(int)
                df_indexed['risk_off_regime'] = (avg_risk_corr < 0.3).astype(int)
            
            # Reset index and merge back
            df_indexed = df_indexed.reset_index()
            merge_cols = ['timestamp'] + [col for col in df_indexed.columns if col not in df.columns]
            all_data[symbol] = df.merge(df_indexed[merge_cols], on='timestamp', how='left')
        
        return all_data
    
    def add_microstructure_features(self, df, symbol):
        """Add market microstructure features"""
        enhanced_df = df.copy()
        
        # 1. Intrabar analysis
        enhanced_df['typical_price'] = (enhanced_df['high'] + enhanced_df['low'] + enhanced_df['close']) / 3
        enhanced_df['price_range'] = enhanced_df['high'] - enhanced_df['low']
        enhanced_df['price_range_pct'] = enhanced_df['price_range'] / enhanced_df['close']
        
        # Candle patterns
        enhanced_df['upper_shadow'] = enhanced_df['high'] - np.maximum(enhanced_df['open'], enhanced_df['close'])
        enhanced_df['lower_shadow'] = np.minimum(enhanced_df['open'], enhanced_df['close']) - enhanced_df['low']
        enhanced_df['body_size'] = abs(enhanced_df['close'] - enhanced_df['open'])
        
        enhanced_df['upper_shadow_ratio'] = enhanced_df['upper_shadow'] / (enhanced_df['price_range'] + 1e-8)
        enhanced_df['lower_shadow_ratio'] = enhanced_df['lower_shadow'] / (enhanced_df['price_range'] + 1e-8)
        enhanced_df['body_ratio'] = enhanced_df['body_size'] / (enhanced_df['price_range'] + 1e-8)
        
        # Pattern detection
        enhanced_df['hammer_pattern'] = ((enhanced_df['lower_shadow_ratio'] > 0.6) & (enhanced_df['body_ratio'] < 0.3)).astype(int)
        enhanced_df['doji_pattern'] = (enhanced_df['body_ratio'] < 0.1).astype(int)
        enhanced_df['shooting_star'] = ((enhanced_df['upper_shadow_ratio'] > 0.6) & (enhanced_df['body_ratio'] < 0.3)).astype(int)
        
        # 2. Order flow estimation
        # Estimate buy/sell pressure
        enhanced_df['buy_pressure'] = np.where(
            enhanced_df['close'] > enhanced_df['open'],
            enhanced_df['volume'] * (enhanced_df['close'] - enhanced_df['low']) / (enhanced_df['price_range'] + 1e-8),
            0
        )
        
        enhanced_df['sell_pressure'] = np.where(
            enhanced_df['close'] < enhanced_df['open'],
            enhanced_df['volume'] * (enhanced_df['high'] - enhanced_df['close']) / (enhanced_df['price_range'] + 1e-8),
            0
        )
        
        # Order flow imbalance
        for window in [5, 10, 20]:
            buy_flow = enhanced_df['buy_pressure'].rolling(window).sum()
            sell_flow = enhanced_df['sell_pressure'].rolling(window).sum()
            total_flow = buy_flow + sell_flow
            enhanced_df[f'order_flow_imbalance_{window}'] = (buy_flow - sell_flow) / (total_flow + 1e-8)
        
        # 3. VWAP analysis
        for window in [20, 50]:
            # Volume Weighted Average Price
            vwap_num = (enhanced_df['typical_price'] * enhanced_df['volume']).rolling(window).sum()
            vwap_den = enhanced_df['volume'].rolling(window).sum()
            vwap = vwap_num / vwap_den
            
            enhanced_df[f'vwap_{window}'] = vwap
            enhanced_df[f'vwap_deviation_{window}'] = (enhanced_df['close'] - vwap) / vwap
            enhanced_df[f'above_vwap_{window}'] = (enhanced_df['close'] > vwap).astype(int)
            
            # VWAP bands
            price_var = ((enhanced_df['typical_price'] - vwap) ** 2 * enhanced_df['volume']).rolling(window).sum() / vwap_den
            vwap_std = np.sqrt(price_var)
            enhanced_df[f'vwap_zscore_{window}'] = enhanced_df[f'vwap_deviation_{window}'] / (vwap_std + 1e-8)
        
        # 4. Liquidity proxies
        # Estimate bid-ask spread
        enhanced_df['spread_proxy'] = enhanced_df['price_range_pct'] * 0.5
        
        # Market impact estimation
        price_change = enhanced_df['close'].pct_change().abs()
        volume_norm = enhanced_df['volume'] / enhanced_df['volume'].rolling(20).mean()
        enhanced_df['market_impact'] = price_change / (volume_norm + 1e-8)
        enhanced_df['liquidity_score'] = 1 / (enhanced_df['market_impact'] + 1e-8)
        
        # 5. Time-based effects
        enhanced_df['hour'] = enhanced_df['timestamp'].dt.hour
        enhanced_df['day_of_week'] = enhanced_df['timestamp'].dt.dayofweek
        
        # Session effects
        enhanced_df['asian_session'] = ((enhanced_df['hour'] >= 0) & (enhanced_df['hour'] < 8)).astype(int)
        enhanced_df['european_session'] = ((enhanced_df['hour'] >= 8) & (enhanced_df['hour'] < 16)).astype(int)
        enhanced_df['us_session'] = ((enhanced_df['hour'] >= 16) & (enhanced_df['hour'] < 24)).astype(int)
        enhanced_df['weekend_effect'] = (enhanced_df['day_of_week'] >= 5).astype(int)
        
        return enhanced_df
    
    def add_dynamic_labels(self, df, symbol):
        """Add dynamic DipMaster-optimized labels"""
        enhanced_df = df.copy()
        
        # Get symbol characteristics for label adjustment
        vol_group = self.get_volatility_group(symbol)
        category = self.get_symbol_category(symbol)
        
        # Volatility-adjusted targets
        vol_multipliers = {'low_vol': 0.8, 'medium_vol': 1.0, 'high_vol': 1.5}
        vol_mult = vol_multipliers[vol_group]
        
        base_targets = [0.003, 0.006, 0.008, 0.012, 0.015, 0.020]
        adjusted_targets = [t * vol_mult for t in base_targets]
        
        # 1. Multi-horizon returns
        horizons = [1, 3, 6, 12, 18, 24, 36]  # 5min to 3h
        
        for horizon in horizons:
            future_return = enhanced_df['close'].pct_change(periods=horizon).shift(-horizon)
            enhanced_df[f'future_return_{horizon}p'] = future_return
            
            # Profitability targets
            for i, target in enumerate(adjusted_targets[:4]):
                enhanced_df[f'hits_target_{i}_{horizon}p'] = (future_return >= target).astype(int)
            
            # Risk metrics
            enhanced_df[f'hits_stop_loss_{horizon}p'] = (future_return <= -0.004 * vol_mult).astype(int)
        
        # 2. 15-minute boundary optimization
        enhanced_df['minute'] = enhanced_df['timestamp'].dt.minute
        boundary_minutes = [15, 30, 45, 0]
        
        for horizon in [3, 6, 9, 12, 15, 18]:  # Up to 1.5 hours
            future_minute = enhanced_df['minute'].shift(-horizon)
            is_boundary = future_minute.isin(boundary_minutes)
            enhanced_df[f'boundary_exit_{horizon}p'] = is_boundary.astype(int)
            
            # Boundary-adjusted return (bonus for good timing)
            if f'future_return_{horizon}p' in enhanced_df.columns:
                future_return = enhanced_df[f'future_return_{horizon}p']
                boundary_bonus = 0.001 * vol_mult
                enhanced_df[f'boundary_adj_return_{horizon}p'] = future_return + (is_boundary * boundary_bonus)
        
        # 3. Category-specific labels
        if category == 'meme':
            # Meme coin explosive moves
            for horizon in [1, 3, 6]:
                future_return = enhanced_df[f'future_return_{horizon}p']
                enhanced_df[f'meme_explosion_{horizon}p'] = (future_return > 0.05).astype(int)
                enhanced_df[f'meme_crash_{horizon}p'] = (future_return < -0.05).astype(int)
                
        elif category in ['layer1', 'defi']:
            # Sustainable growth patterns
            for horizon in [12, 24, 36]:
                future_return = enhanced_df[f'future_return_{horizon}p']
                enhanced_df[f'sustainable_growth_{horizon}p'] = ((future_return > 0.01) & (future_return < 0.1)).astype(int)
        
        # 4. Main DipMaster targets
        main_horizon = 12  # 1 hour
        main_return = enhanced_df[f'future_return_{main_horizon}p']
        
        enhanced_df['target_return'] = main_return
        enhanced_df['target_binary'] = (main_return > 0).astype(int)
        enhanced_df['target_profitable'] = (main_return >= adjusted_targets[1]).astype(int)  # 0.6% adjusted
        enhanced_df['target_excellent'] = (main_return >= adjusted_targets[3]).astype(int)   # 1.2% adjusted
        
        # 5. Risk-adjusted labels
        if 'volatility_20' in enhanced_df.columns:
            future_vol = enhanced_df['volatility_20'].shift(-main_horizon)
            enhanced_df['risk_adj_return'] = main_return / (future_vol + 1e-8)
            enhanced_df['high_sharpe_trade'] = (enhanced_df['risk_adj_return'] > 2.0).astype(int)
        
        # 6. Multi-class return classification
        conditions = [
            (main_return <= -0.004 * vol_mult),  # Loss
            ((main_return > -0.004 * vol_mult) & (main_return <= 0)),  # Small loss
            ((main_return > 0) & (main_return < adjusted_targets[0])),  # Small profit
            ((main_return >= adjusted_targets[0]) & (main_return < adjusted_targets[2])),  # Good profit
            ((main_return >= adjusted_targets[2]) & (main_return < adjusted_targets[4])),  # Great profit
            (main_return >= adjusted_targets[4])  # Excellent profit
        ]
        labels = [0, 1, 2, 3, 4, 5]
        enhanced_df['return_class'] = np.select(conditions, labels, default=1)
        
        return enhanced_df
    
    def clean_and_validate(self, df, symbol):
        """Clean data and validate quality"""
        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Get feature columns
        feature_cols = [col for col in df.columns 
                       if col not in ['timestamp', 'symbol'] 
                       and not col.startswith('target')
                       and not col.startswith('future_')
                       and not col.startswith('hits_')
                       and not col.startswith('return_class')]
        
        # Handle missing values intelligently
        for col in feature_cols:
            if col in df.columns:
                null_pct = df[col].isnull().sum() / len(df)
                
                if null_pct > 0.5:
                    df[col] = df[col].fillna(0)
                elif null_pct > 0.1:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Robust outlier handling
        for col in feature_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                Q1 = df[col].quantile(0.005)
                Q3 = df[col].quantile(0.995)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        print(f"  Cleaned {symbol}: {len(feature_cols)} features")
        return df
    
    def generate_ultra_features(self):
        """Generate ultra enhanced features for all symbols"""
        print("Starting Ultra Enhanced Top30 Feature Engineering...")
        start_time = time.time()
        
        # Load all data
        all_data = self.load_all_data()
        
        if len(all_data) < 10:
            raise ValueError(f"Insufficient symbols: {len(all_data)}")
        
        # Step 1: Symbol-specific features
        print("\\nStep 1: Adding symbol-specific features...")
        for symbol in all_data:
            try:
                all_data[symbol] = self.add_symbol_specific_features(all_data[symbol], symbol)
                print(f"  ✓ {symbol}: symbol-specific features added")
            except Exception as e:
                print(f"  ✗ {symbol}: {e}")
        
        # Step 2: Cross-symbol features
        print("\\nStep 2: Adding cross-symbol features...")
        try:
            all_data = self.add_cross_symbol_features(all_data)
            print("  ✓ Cross-symbol features completed")
        except Exception as e:
            print(f"  ✗ Cross-symbol features failed: {e}")
        
        # Step 3: Microstructure features
        print("\\nStep 3: Adding microstructure features...")
        for symbol in all_data:
            try:
                all_data[symbol] = self.add_microstructure_features(all_data[symbol], symbol)
                print(f"  ✓ {symbol}: microstructure features added")
            except Exception as e:
                print(f"  ✗ {symbol}: {e}")
        
        # Step 4: Dynamic labels
        print("\\nStep 4: Adding dynamic labels...")
        for symbol in all_data:
            try:
                all_data[symbol] = self.add_dynamic_labels(all_data[symbol], symbol)
                print(f"  ✓ {symbol}: labels added")
            except Exception as e:
                print(f"  ✗ {symbol}: {e}")
        
        # Step 5: Clean and validate
        print("\\nStep 5: Cleaning and validating...")
        for symbol in all_data:
            try:
                all_data[symbol] = self.clean_and_validate(all_data[symbol], symbol)
            except Exception as e:
                print(f"  ✗ {symbol} cleaning failed: {e}")
        
        # Combine and save
        print("\\nStep 6: Combining and saving results...")
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
            
            # Generate metadata
            feature_cols = [col for col in combined_df.columns 
                           if col not in ['timestamp', 'symbol']]
            
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
                    'cross_symbol': len([col for col in feature_cols if any(x in col for x in ['market_', 'correlation', 'rank', 'relative', 'beta'])]),
                    'microstructure': len([col for col in feature_cols if any(x in col for x in ['vwap', 'flow', 'pressure', 'spread', 'liquidity', 'pattern'])]),
                    'labels': len([col for col in feature_cols if any(x in col for x in ['target', 'future_', 'hits_', 'return_class'])]),
                    'category': len([col for col in feature_cols if col.startswith('is_')]),
                    'time': len([col for col in feature_cols if any(x in col for x in ['hour', 'session', 'weekend', 'boundary'])])
                }
            }
            
            metadata_file = f'data/FeatureSet_Top30_V6_{timestamp}.json'
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Cross-symbol analysis
            cross_analysis = {
                'correlation_matrix': {},
                'sector_performance': {},
                'feature_stability': {}
            }
            
            analysis_file = f'data/Cross_Symbol_Analysis_{timestamp}.json'
            with open(analysis_file, 'w') as f:
                json.dump(cross_analysis, f, indent=2, default=str)
            
            total_time = time.time() - start_time
            
            print(f"\\nUltra Enhanced Feature Engineering Completed!")
            print(f"📁 Output file: {output_file}")
            print(f"📋 Metadata: {metadata_file}")
            print(f"📊 Analysis: {analysis_file}")
            print(f"📈 Total rows: {len(combined_df):,}")
            print(f"🎯 Total features: {len(feature_cols)}")
            print(f"💰 Symbols: {len(all_data)}")
            print(f"⏱️ Processing time: {total_time:.1f}s")
            
            # Feature breakdown
            print(f"\\n📊 Feature Categories:")
            for category, count in metadata['feature_categories'].items():
                if count > 0:
                    print(f"  - {category}: {count}")
            
            return {
                'output_file': output_file,
                'metadata_file': metadata_file,
                'analysis_file': analysis_file,
                'total_rows': len(combined_df),
                'total_features': len(feature_cols),
                'symbols_count': len(all_data),
                'processing_time': total_time
            }
        else:
            print("No data to save")
            return None

def main():
    """Main execution function"""
    engineer = UltraEnhancedTop30FeatureEngineer()
    result = engineer.generate_ultra_features()
    
    if result:
        print(f"\\n🎉 Success! Generated ultra enhanced features:")
        print(f"  📁 File: {result['output_file']}")
        print(f"  📊 Features: {result['total_features']}")
        print(f"  💰 Symbols: {result['symbols_count']}")
        print(f"  📈 Rows: {result['total_rows']:,}")
        print(f"  ⏱️ Time: {result['processing_time']:.1f}s")

if __name__ == "__main__":
    main()