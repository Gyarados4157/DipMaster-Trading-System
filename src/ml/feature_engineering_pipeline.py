"""
Advanced Feature Engineering Pipeline for DipMaster Strategy
Implements rigorous feature engineering with leakage prevention.
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

# Technical analysis
import talib as ta

warnings.filterwarnings('ignore')

class DipMasterFeatureEngineer:
    """
    Advanced feature engineering pipeline specifically for DipMaster strategy
    """
    
    def __init__(self, 
                 lookback_windows: List[int] = [5, 10, 20, 50],
                 future_horizons: List[int] = [15, 30, 60, 240],  # minutes
                 target_returns: List[float] = [0.006, 0.008, 0.012],  # 0.6%, 0.8%, 1.2%
                 max_holding_minutes: int = 180):
        """
        Initialize feature engineer
        
        Args:
            lookback_windows: Historical windows for features
            future_horizons: Forward-looking horizons for labels
            target_returns: Target return thresholds
            max_holding_minutes: Maximum position holding time
        """
        self.lookback_windows = lookback_windows
        self.future_horizons = future_horizons
        self.target_returns = target_returns
        self.max_holding_minutes = max_holding_minutes
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('DipMasterFeatureEngineer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline
        
        Args:
            df: OHLCV dataframe with datetime index
            
        Returns:
            Enhanced dataframe with engineered features
        """
        
        self.logger.info("Starting feature engineering")
        self.logger.info(f"Input data shape: {df.shape}")
        
        # Create copy to avoid modifying original
        data = df.copy()
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 1. Basic price features
        self.logger.info("Engineering basic price features")
        data = self._add_basic_price_features(data)
        
        # 2. Technical indicators
        self.logger.info("Engineering technical indicators")
        data = self._add_technical_indicators(data)
        
        # 3. Volume analysis
        self.logger.info("Engineering volume features")
        data = self._add_volume_features(data)
        
        # 4. Volatility measures
        self.logger.info("Engineering volatility features")
        data = self._add_volatility_features(data)
        
        # 5. Momentum features
        self.logger.info("Engineering momentum features")
        data = self._add_momentum_features(data)
        
        # 6. Market microstructure
        self.logger.info("Engineering microstructure features")
        data = self._add_microstructure_features(data)
        
        # 7. DipMaster specific features
        self.logger.info("Engineering DipMaster-specific features")
        data = self._add_dipmaster_features(data)
        
        # 8. Regime detection features
        self.logger.info("Engineering market regime features")
        data = self._add_regime_features(data)
        
        # 9. Time-based features
        self.logger.info("Engineering time-based features")
        data = self._add_time_features(data)
        
        # 10. Interaction features
        self.logger.info("Engineering interaction features")
        data = self._add_interaction_features(data)
        
        # 11. Labels (must be last to prevent leakage)
        self.logger.info("Engineering labels")
        data = self._add_labels(data)
        
        # Clean up - remove NaN values
        initial_len = len(data)
        data = data.dropna()
        final_len = len(data)
        
        self.logger.info(f"Feature engineering complete")
        self.logger.info(f"Final data shape: {data.shape}")
        self.logger.info(f"Samples removed due to NaN: {initial_len - final_len}")
        
        return data
    
    def _add_basic_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic price-derived features"""
        
        # Returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Price spreads
        data['hl_spread'] = (data['high'] - data['low']) / data['close']
        data['oc_spread'] = (data['close'] - data['open']) / data['open']
        data['ho_spread'] = (data['high'] - data['open']) / data['open']
        data['lo_spread'] = (data['low'] - data['open']) / data['open']
        
        # Range indicators
        data['true_range'] = ta.TRANGE(data['high'], data['low'], data['close'])\n        data['price_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])\n        \n        # Gap analysis\n        data['gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)\n        data['gap_filled'] = np.where(\n            (data['gap'] > 0) & (data['low'] <= data['close'].shift(1)), 1,\n            np.where((data['gap'] < 0) & (data['high'] >= data['close'].shift(1)), 1, 0)\n        )\n        \n        return data\n    \n    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Add comprehensive technical indicators\"\"\"\n        \n        # RSI with different periods\n        for period in [14, 21, 30]:\n            data[f'rsi_{period}'] = ta.RSI(data['close'], timeperiod=period)\n            \n        # MACD\n        macd, macd_signal, macd_hist = ta.MACD(data['close'])\n        data['macd'] = macd\n        data['macd_signal'] = macd_signal\n        data['macd_histogram'] = macd_hist\n        data['macd_convergence'] = (macd - macd_signal) / abs(macd_signal + 1e-10)\n        \n        # Bollinger Bands with different periods\n        for period in [20, 50]:\n            bb_upper, bb_middle, bb_lower = ta.BBANDS(\n                data['close'], timeperiod=period, nbdevup=2, nbdevdn=2\n            )\n            data[f'bb_upper_{period}'] = bb_upper\n            data[f'bb_middle_{period}'] = bb_middle\n            data[f'bb_lower_{period}'] = bb_lower\n            data[f'bb_position_{period}'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)\n            data[f'bb_width_{period}'] = (bb_upper - bb_lower) / bb_middle\n            data[f'bb_squeeze_{period}'] = (data[f'bb_width_{period}'].rolling(20).rank() / 20 < 0.2).astype(int)\n        \n        # Stochastic Oscillator\n        stoch_k, stoch_d = ta.STOCH(data['high'], data['low'], data['close'])\n        data['stoch_k'] = stoch_k\n        data['stoch_d'] = stoch_d\n        data['stoch_divergence'] = stoch_k - stoch_d\n        \n        # Williams %R\n        data['williams_r'] = ta.WILLR(data['high'], data['low'], data['close'])\n        \n        # Commodity Channel Index\n        data['cci'] = ta.CCI(data['high'], data['low'], data['close'])\n        \n        # Average Directional Index\n        data['adx'] = ta.ADX(data['high'], data['low'], data['close'])\n        data['plus_di'] = ta.PLUS_DI(data['high'], data['low'], data['close'])\n        data['minus_di'] = ta.MINUS_DI(data['high'], data['low'], data['close'])\n        \n        # Moving averages\n        for period in [5, 10, 15, 20, 50, 100]:\n            data[f'sma_{period}'] = ta.SMA(data['close'], timeperiod=period)\n            data[f'ema_{period}'] = ta.EMA(data['close'], timeperiod=period)\n            \n            # Price relative to MA\n            data[f'price_vs_sma_{period}'] = (data['close'] - data[f'sma_{period}']) / data[f'sma_{period}']\n            data[f'price_vs_ema_{period}'] = (data['close'] - data[f'ema_{period}']) / data[f'ema_{period}']\n            \n        # MA crossovers\n        data['sma_5_20_cross'] = np.where(data['sma_5'] > data['sma_20'], 1, 0)\n        data['ema_10_20_cross'] = np.where(data['ema_10'] > data['ema_20'], 1, 0)\n        \n        return data\n    \n    def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Add volume-based features\"\"\"\n        \n        # Volume moving averages\n        for period in [5, 10, 20, 50]:\n            data[f'volume_ma_{period}'] = data['volume'].rolling(period).mean()\n            data[f'volume_ratio_{period}'] = data['volume'] / data[f'volume_ma_{period}']\n        \n        # Volume momentum\n        data['volume_change'] = data['volume'].pct_change()\n        data['volume_momentum'] = data['volume'].rolling(5).mean() / data['volume'].rolling(20).mean()\n        \n        # Price-Volume relationship\n        data['pv_trend'] = data['returns'] * np.log(data['volume'] + 1)\n        data['volume_price_trend'] = ta.OBV(data['close'], data['volume'])\n        \n        # Volume spikes\n        volume_threshold = data['volume'].rolling(50).quantile(0.8)\n        data['volume_spike'] = (data['volume'] > volume_threshold).astype(int)\n        \n        # Accumulation/Distribution Line\n        data['ad_line'] = ta.AD(data['high'], data['low'], data['close'], data['volume'])\n        \n        # Chaikin Money Flow\n        data['cmf'] = ta.ADOSC(data['high'], data['low'], data['close'], data['volume'])\n        \n        # Volume-Weighted Average Price (approximation)\n        data['vwap'] = (data['volume'] * data['close']).rolling(20).sum() / data['volume'].rolling(20).sum()\n        data['price_vs_vwap'] = (data['close'] - data['vwap']) / data['vwap']\n        \n        return data\n    \n    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Add volatility measures\"\"\"\n        \n        # Historical volatility (different windows)\n        for window in self.lookback_windows:\n            data[f'volatility_{window}'] = data['returns'].rolling(window).std() * np.sqrt(288)  # 5-min to daily\n            data[f'volatility_rank_{window}'] = data[f'volatility_{window}'].rolling(100).rank() / 100\n        \n        # GARCH-like volatility\n        data['garch_vol'] = data['returns'].ewm(span=20).std() * np.sqrt(288)\n        \n        # Volatility of volatility\n        data['vol_of_vol'] = data['volatility_20'].rolling(10).std()\n        \n        # Average True Range\n        for period in [14, 20]:\n            data[f'atr_{period}'] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=period)\n            data[f'atr_ratio_{period}'] = data[f'atr_{period}'] / data['close']\n        \n        # Volatility regime indicators\n        vol_median = data['volatility_20'].rolling(100).median()\n        data['high_vol_regime'] = (data['volatility_20'] > vol_median * 1.5).astype(int)\n        data['low_vol_regime'] = (data['volatility_20'] < vol_median * 0.7).astype(int)\n        \n        return data\n    \n    def _add_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Add momentum indicators\"\"\"\n        \n        # Price momentum (different horizons)\n        for window in self.lookback_windows:\n            data[f'momentum_{window}'] = data['close'] / data['close'].shift(window) - 1\n            data[f'momentum_rank_{window}'] = data[f'momentum_{window}'].rolling(100).rank() / 100\n        \n        # Rate of Change\n        for period in [5, 10, 20]:\n            data[f'roc_{period}'] = ta.ROC(data['close'], timeperiod=period)\n        \n        # Momentum oscillators\n        data['momentum_oscillator'] = ta.MOM(data['close'])\n        \n        # Acceleration (second derivative of price)\n        data['price_acceleration'] = data['returns'].diff()\n        \n        # Momentum divergence (price vs RSI)\n        data['momentum_divergence'] = (\n            (data['momentum_5'] > 0) & (data['rsi_14'] < 50)\n        ).astype(int) - (\n            (data['momentum_5'] < 0) & (data['rsi_14'] > 50)\n        ).astype(int)\n        \n        return data\n    \n    def _add_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Add market microstructure features\"\"\"\n        \n        # Bid-ask spread proxy (using high-low)\n        data['effective_spread'] = data['hl_spread']\n        \n        # Price impact measures\n        data['price_impact'] = abs(data['returns']) / np.log(data['volume'] + 1)\n        \n        # Order flow imbalance (proxy)\n        data['buy_pressure'] = np.where(\n            data['close'] > data['open'],\n            data['volume'] * (data['close'] - data['open']) / (data['high'] - data['low']),\n            0\n        )\n        data['sell_pressure'] = np.where(\n            data['close'] < data['open'],\n            data['volume'] * (data['open'] - data['close']) / (data['high'] - data['low']),\n            0\n        )\n        data['order_flow_imbalance'] = (\n            (data['buy_pressure'] - data['sell_pressure']) / \n            (data['buy_pressure'] + data['sell_pressure'] + 1e-10)\n        )\n        \n        # Tick direction (approximation)\n        data['tick_direction'] = np.sign(data['close'].diff())\n        data['tick_runs'] = data.groupby((data['tick_direction'] != data['tick_direction'].shift()).cumsum())['tick_direction'].cumsum()\n        \n        # Arrival rate (inverse of time between significant moves)\n        significant_move = abs(data['returns']) > data['returns'].rolling(50).quantile(0.8)\n        data['arrival_rate'] = significant_move.rolling(20).sum()\n        \n        return data\n    \n    def _add_dipmaster_features(self, data: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Add DipMaster strategy specific features\"\"\"\n        \n        # Dip identification\n        data['is_dip'] = (data['returns'] < -0.002).astype(int)  # 0.2% dip\n        data['dip_magnitude'] = np.where(data['is_dip'] == 1, abs(data['returns']), 0)\n        \n        # RSI dip zone (core DipMaster signal)\n        data['rsi_dip_zone'] = ((data['rsi_14'] >= 25) & (data['rsi_14'] <= 50)).astype(int)\n        \n        # Combined dip signal\n        data['dipmaster_base_signal'] = (\n            (data['is_dip'] == 1) & \n            (data['rsi_dip_zone'] == 1)\n        ).astype(int)\n        \n        # Enhanced dip conditions\n        data['volume_confirmed_dip'] = (\n            (data['dipmaster_base_signal'] == 1) & \n            (data['volume_ratio_5'] > 1.2)\n        ).astype(int)\n        \n        # Mean reversion strength\n        data['mean_reversion_strength'] = (\n            abs(data['price_vs_sma_20']) * data['bb_position_20']\n        )\n        \n        # DipMaster signal strength composite\n        signal_components = [\n            data['rsi_dip_zone'],\n            data['is_dip'],\n            (data['volume_ratio_5'] > 1.1).astype(int),\n            (data['bb_position_20'] < 0.3).astype(int),\n            (data['williams_r'] < -70).astype(int)\n        ]\n        data['dipmaster_signal_strength'] = sum(signal_components) / len(signal_components)\n        \n        # Recovery potential (for exit timing)\n        data['recovery_potential'] = (\n            data['rsi_14'].rolling(5).mean() + \n            (1 - data['bb_position_20']) * 50 +\n            data['stoch_k']\n        ) / 3\n        \n        return data\n    \n    def _add_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Add market regime detection features\"\"\"\n        \n        # Trend regime\n        trend_window = 50\n        data['trend_slope'] = data['close'].rolling(trend_window).apply(\n            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == trend_window else np.nan\n        )\n        \n        # Regime classification\n        data['bull_regime'] = (data['trend_slope'] > 0.001).astype(int)\n        data['bear_regime'] = (data['trend_slope'] < -0.001).astype(int)\n        data['sideways_regime'] = (\n            (data['trend_slope'] >= -0.001) & (data['trend_slope'] <= 0.001)\n        ).astype(int)\n        \n        # Volatility regime transition\n        data['vol_regime_change'] = (\n            data['high_vol_regime'] != data['high_vol_regime'].shift(1)\n        ).astype(int)\n        \n        # Market stress indicator\n        data['market_stress'] = (\n            data['volatility_20'] * 0.4 +\n            (1 - data['bb_position_20']) * 0.3 +\n            (abs(data['rsi_14'] - 50) / 50) * 0.3\n        )\n        \n        return data\n    \n    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Add time-based features\"\"\"\n        \n        if isinstance(data.index, pd.DatetimeIndex):\n            # Hour of day (0-23)\n            data['hour'] = data.index.hour\n            data['minute'] = data.index.minute\n            \n            # Day of week (0=Monday, 6=Sunday)\n            data['day_of_week'] = data.index.dayofweek\n            data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)\n            \n            # Trading session indicators (UTC)\n            # Asian session: 22:00-06:00\n            data['asian_session'] = (\n                (data.index.hour >= 22) | (data.index.hour <= 6)\n            ).astype(int)\n            \n            # European session: 06:00-14:00  \n            data['european_session'] = (\n                (data.index.hour >= 6) & (data.index.hour <= 14)\n            ).astype(int)\n            \n            # US session: 14:00-22:00\n            data['us_session'] = (\n                (data.index.hour >= 14) & (data.index.hour <= 22)\n            ).astype(int)\n            \n            # Session transitions (higher volatility)\n            data['session_transition'] = (\n                (data.index.hour == 6) | (data.index.hour == 14) | (data.index.hour == 22)\n            ).astype(int)\n        \n        return data\n    \n    def _add_interaction_features(self, data: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Add feature interactions\"\"\"\n        \n        # RSI-Volume interaction\n        data['rsi_volume_interaction'] = data['rsi_14'] * data['volume_ratio_5']\n        \n        # Volatility-Momentum interaction\n        data['vol_momentum_interaction'] = data['volatility_20'] * abs(data['momentum_10'])\n        \n        # Price position in BB vs RSI\n        data['bb_rsi_interaction'] = data['bb_position_20'] * (data['rsi_14'] / 100)\n        \n        # Trend-Volume confirmation\n        data['trend_volume_confirmation'] = (\n            np.sign(data['trend_slope']) * data['volume_ratio_10']\n        )\n        \n        return data\n    \n    def _add_labels(self, data: pd.DataFrame) -> pd.DataFrame:\n        \"\"\"Add forward-looking labels (MUST be last to prevent leakage)\"\"\"\n        \n        # Calculate future returns for different horizons\n        for horizon_minutes in self.future_horizons:\n            horizon_periods = horizon_minutes // 5  # Convert to 5-minute periods\n            \n            # Future returns\n            future_closes = data['close'].shift(-horizon_periods)\n            data[f'future_return_{horizon_minutes}m'] = (\n                (future_closes - data['close']) / data['close']\n            )\n            \n            # Binary profitable labels for different thresholds\n            for threshold in self.target_returns:\n                data[f'profitable_{threshold:.1%}_{horizon_minutes}m'] = (\n                    data[f'future_return_{horizon_minutes}m'] > threshold\n                ).astype(int)\n        \n        # DipMaster specific labels (15-minute boundary exit)\n        # Maximum return achieved within max holding period\n        max_holding_periods = self.max_holding_minutes // 5\n        \n        future_highs = data['high'].rolling(\n            window=max_holding_periods\n        ).max().shift(-max_holding_periods)\n        \n        data['max_return_within_holding'] = (\n            (future_highs - data['close']) / data['close']\n        )\n        \n        # Hit target within holding period\n        for threshold in self.target_returns:\n            data[f'hits_target_{threshold:.1%}'] = (\n                data['max_return_within_holding'] > threshold\n            ).astype(int)\n        \n        # Stop loss hit (2% loss)\n        future_lows = data['low'].rolling(\n            window=max_holding_periods\n        ).min().shift(-max_holding_periods)\n        \n        data['min_return_within_holding'] = (\n            (future_lows - data['close']) / data['close']\n        )\n        \n        data['hits_stop_loss'] = (\n            data['min_return_within_holding'] < -0.02\n        ).astype(int)\n        \n        # Primary label for DipMaster (15-minute exit with 0.8% target)\n        data['dipmaster_primary_label'] = (\n            data['profitable_0.8%_15m'] & ~data['hits_stop_loss']\n        ).astype(int)\n        \n        return data\n    \n    def validate_features(self, data: pd.DataFrame) -> Dict[str, Any]:\n        \"\"\"Validate feature engineering results\"\"\"\n        \n        validation_results = {\n            'total_features': data.shape[1],\n            'total_samples': data.shape[0],\n            'feature_categories': {},\n            'missing_values': {},\n            'label_distribution': {},\n            'potential_issues': []\n        }\n        \n        # Categorize features\n        feature_categories = {\n            'price': [col for col in data.columns if any(term in col.lower() for term in ['price', 'open', 'high', 'low', 'close'])],\n            'volume': [col for col in data.columns if 'volume' in col.lower()],\n            'technical': [col for col in data.columns if any(term in col.lower() for term in ['rsi', 'macd', 'bb_', 'ema', 'sma'])],\n            'volatility': [col for col in data.columns if any(term in col.lower() for term in ['vol', 'atr'])],\n            'momentum': [col for col in data.columns if 'momentum' in col.lower()],\n            'microstructure': [col for col in data.columns if any(term in col.lower() for term in ['spread', 'flow', 'impact'])],\n            'regime': [col for col in data.columns if 'regime' in col.lower()],\n            'time': [col for col in data.columns if any(term in col.lower() for term in ['hour', 'day', 'session'])],\n            'labels': [col for col in data.columns if any(term in col.lower() for term in ['future', 'profitable', 'hits', 'label'])],\n            'dipmaster': [col for col in data.columns if 'dipmaster' in col.lower()]\n        }\n        \n        validation_results['feature_categories'] = {k: len(v) for k, v in feature_categories.items()}\n        \n        # Check for missing values\n        missing_pct = (data.isnull().sum() / len(data)) * 100\n        validation_results['missing_values'] = {\n            col: pct for col, pct in missing_pct.items() if pct > 0\n        }\n        \n        # Label distribution\n        label_cols = feature_categories['labels']\n        for col in label_cols:\n            if col in data.columns and data[col].dtype in ['int64', 'float64']:\n                if data[col].nunique() <= 10:  # Categorical-like\n                    validation_results['label_distribution'][col] = data[col].value_counts().to_dict()\n        \n        # Potential issues\n        if len(validation_results['missing_values']) > 0:\n            validation_results['potential_issues'].append(\"Missing values detected\")\n        \n        # Check for constant features\n        constant_features = [col for col in data.columns if data[col].nunique() <= 1]\n        if constant_features:\n            validation_results['potential_issues'].append(f\"Constant features: {constant_features}\")\n        \n        # Check for features with extreme skewness\n        numeric_cols = data.select_dtypes(include=[np.number]).columns\n        highly_skewed = []\n        for col in numeric_cols:\n            if not col.startswith('future_') and not col.endswith('_label'):\n                skewness = abs(data[col].skew())\n                if skewness > 5:\n                    highly_skewed.append(col)\n        \n        if highly_skewed:\n            validation_results['potential_issues'].append(f\"Highly skewed features (|skew| > 5): {highly_skewed[:5]}\")\n        \n        self.logger.info(f\"Feature validation complete: {validation_results['total_features']} features, {validation_results['total_samples']} samples\")\n        \n        return validation_results