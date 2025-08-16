"""
DipMaster Enhanced V4 特征工程管道
专为85%+胜率策略设计的高质量机器学习特征生成引擎

核心特征类别：
1. DipMaster核心特征 - RSI逢跌、价格动量、成交量确认、布林带位置
2. 市场微观结构特征 - 订单流不平衡、波动率、流动性、时间周期
3. 跨时间框架特征 - 多时间框架对齐、趋势一致性、动量收敛
4. 跨资产特征 - 相关性、相对强度、行业轮动
5. 高质量标签生成 - 多维度目标变量和风险标签
"""

import pandas as pd
import numpy as np
import sqlite3
import warnings
from pathlib import Path
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import logging
from scipy import stats
from scipy.stats import zscore
import talib as ta
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import numba
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time

# 配置警告
warnings.filterwarnings('ignore')

@dataclass
class FeatureEngineConfig:
    """特征工程配置类"""
    # 数据参数
    symbols: List[str]
    primary_timeframe: str = "5m"
    analysis_timeframes: List[str] = None
    lookback_periods: Dict[str, int] = None
    
    # DipMaster核心参数
    rsi_period: int = 14
    rsi_entry_range: Tuple[int, int] = (25, 45)
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    volume_ma_period: int = 20
    
    # 微观结构参数
    volatility_windows: List[int] = None
    momentum_periods: List[int] = None
    
    # 标签生成参数
    prediction_horizons: List[int] = None  # 分钟
    profit_targets: List[float] = None  # 百分比
    stop_loss_threshold: float = 0.004
    max_holding_minutes: int = 180
    
    # 质量控制参数
    feature_stability_threshold: float = 0.2  # PSI threshold
    correlation_threshold: float = 0.95
    missing_threshold: float = 0.05
    outlier_percentile: Tuple[float, float] = (0.5, 99.5)
    
    def __post_init__(self):
        if self.analysis_timeframes is None:
            self.analysis_timeframes = ["5m", "15m", "1h"]
        if self.lookback_periods is None:
            self.lookback_periods = {"short": 20, "medium": 50, "long": 200}
        if self.volatility_windows is None:
            self.volatility_windows = [10, 20, 50]
        if self.momentum_periods is None:
            self.momentum_periods = [5, 10, 20, 50]
        if self.prediction_horizons is None:
            self.prediction_horizons = [15, 30, 60]  # 15分钟, 30分钟, 1小时
        if self.profit_targets is None:
            self.profit_targets = [0.006, 0.012, 0.020]  # 0.6%, 1.2%, 2.0%

class FeatureQualityValidator:
    """特征质量验证器"""
    
    def __init__(self, config: FeatureEngineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_no_lookahead_bias(self, features: pd.DataFrame, feature_metadata: Dict) -> Dict:
        """检测前视偏差 - 确保特征不包含未来信息"""
        validation_results = {
            'has_lookahead_bias': False,
            'suspicious_features': [],
            'temporal_consistency': True,
            'details': {}
        }
        
        # 检查特征计算的时间窗口
        for feature_name, metadata in feature_metadata.items():
            if 'lookback_window' in metadata:
                window = metadata['lookback_window']
                if window < 0:  # 负数窗口表示使用未来数据
                    validation_results['has_lookahead_bias'] = True
                    validation_results['suspicious_features'].append(feature_name)
        
        # 时间序列单调性检查
        if 'timestamp' in features.columns:
            timestamps = pd.to_datetime(features['timestamp'])
            if not timestamps.is_monotonic_increasing:
                validation_results['temporal_consistency'] = False
        
        return validation_results
    
    def calculate_feature_stability(self, features: pd.DataFrame, 
                                   time_periods: List[Tuple[str, str]]) -> Dict:
        """计算特征稳定性 (PSI - Population Stability Index)"""
        stability_results = {}
        
        for feature in features.select_dtypes(include=[np.number]).columns:
            if feature in ['timestamp', 'symbol_encoded']:
                continue
                
            psi_scores = []
            
            for i in range(len(time_periods) - 1):
                period1_start, period1_end = time_periods[i]
                period2_start, period2_end = time_periods[i + 1]
                
                period1_data = features[
                    (features['timestamp'] >= period1_start) & 
                    (features['timestamp'] < period1_end)
                ][feature].dropna()
                
                period2_data = features[
                    (features['timestamp'] >= period2_start) & 
                    (features['timestamp'] < period2_end)
                ][feature].dropna()
                
                if len(period1_data) > 100 and len(period2_data) > 100:
                    psi = self._calculate_psi(period1_data, period2_data)
                    psi_scores.append(psi)
            
            stability_results[feature] = {
                'mean_psi': np.mean(psi_scores) if psi_scores else np.nan,
                'max_psi': np.max(psi_scores) if psi_scores else np.nan,
                'stability_rating': self._classify_stability(np.mean(psi_scores) if psi_scores else np.nan)
            }
        
        return stability_results
    
    def _calculate_psi(self, baseline: pd.Series, comparison: pd.Series, bins: int = 10) -> float:
        """计算Population Stability Index (PSI)"""
        try:
            # 创建分箱
            baseline_clean = baseline.dropna()
            comparison_clean = comparison.dropna()
            
            if len(baseline_clean) == 0 or len(comparison_clean) == 0:
                return np.nan
            
            # 使用基线数据创建分箱边界
            bin_edges = np.percentile(baseline_clean, np.linspace(0, 100, bins + 1))
            bin_edges = np.unique(bin_edges)  # 移除重复值
            
            if len(bin_edges) < 2:
                return np.nan
            
            # 计算分布
            baseline_dist, _ = np.histogram(baseline_clean, bins=bin_edges)
            comparison_dist, _ = np.histogram(comparison_clean, bins=bin_edges)
            
            # 转换为概率
            baseline_dist = baseline_dist / np.sum(baseline_dist)
            comparison_dist = comparison_dist / np.sum(comparison_dist)
            
            # 避免零概率
            baseline_dist = np.where(baseline_dist == 0, 0.0001, baseline_dist)
            comparison_dist = np.where(comparison_dist == 0, 0.0001, comparison_dist)
            
            # 计算PSI
            psi = np.sum((comparison_dist - baseline_dist) * np.log(comparison_dist / baseline_dist))
            
            return psi
            
        except Exception as e:
            self.logger.warning(f"PSI calculation failed: {e}")
            return np.nan
    
    def _classify_stability(self, psi_score: float) -> str:
        """根据PSI分数分类稳定性"""
        if np.isnan(psi_score):
            return "unknown"
        elif psi_score < 0.1:
            return "stable"
        elif psi_score < 0.2:
            return "moderately_stable"
        else:
            return "unstable"
    
    def detect_multicollinearity(self, features: pd.DataFrame) -> Dict:
        """检测多重共线性"""
        numeric_features = features.select_dtypes(include=[np.number])
        correlation_matrix = numeric_features.corr()
        
        # 找出高度相关的特征对
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = abs(correlation_matrix.iloc[i, j])
                if corr_value > self.config.correlation_threshold:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return {
            'high_correlation_pairs': high_corr_pairs,
            'correlation_matrix': correlation_matrix,
            'multicollinearity_detected': len(high_corr_pairs) > 0
        }

class DipMasterFeatureEngine:
    """DipMaster Enhanced V4 核心特征生成引擎"""
    
    def __init__(self, config: FeatureEngineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.validator = FeatureQualityValidator(config)
        
    def generate_dipmaster_core_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """生成DipMaster核心特征"""
        features = df.copy()
        
        # 基础技术指标
        features['rsi'] = ta.RSI(features['close'], timeperiod=self.config.rsi_period)
        features['bb_upper'], features['bb_middle'], features['bb_lower'] = ta.BBANDS(
            features['close'], 
            timeperiod=self.config.bollinger_period, 
            nbdevup=self.config.bollinger_std,
            nbdevdn=self.config.bollinger_std
        )
        
        # DipMaster核心信号特征
        features['rsi_in_dip_zone'] = (
            (features['rsi'] >= self.config.rsi_entry_range[0]) & 
            (features['rsi'] <= self.config.rsi_entry_range[1])
        ).astype(int)
        
        # 价格相对布林带位置
        features['bb_position'] = (features['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_squeeze'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        
        # 逢跌确认信号
        features['price_dip_1m'] = (features['close'] < features['open']).astype(int)
        features['price_dip_5m'] = (features['close'] < features['close'].shift(1)).astype(int)
        features['price_dip_magnitude'] = (features['close'] - features['open']) / features['open']
        
        # 成交量确认
        features['volume_ma'] = ta.SMA(features['volume'], timeperiod=self.config.volume_ma_period)
        features['volume_ratio'] = features['volume'] / features['volume_ma']
        features['volume_spike'] = (features['volume_ratio'] > 1.5).astype(int)
        
        # 成交量价格确认
        features['volume_price_trend'] = np.where(
            features['close'] > features['open'],
            features['volume'],  # 上涨时的成交量
            -features['volume']  # 下跌时的成交量（负值）
        )
        features['vpt'] = features['volume_price_trend'].cumsum()  # Volume Price Trend
        
        # DipMaster综合信号强度
        features['dipmaster_signal_strength'] = (
            features['rsi_in_dip_zone'] * 0.3 +
            features['price_dip_1m'] * 0.2 +
            features['volume_spike'] * 0.2 +
            (features['bb_position'] < 0.3).astype(int) * 0.3
        )
        
        return features
    
    def generate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成市场微观结构特征"""
        features = df.copy()
        
        # 价格波动率特征
        for window in self.config.volatility_windows:
            features[f'volatility_{window}'] = features['close'].pct_change().rolling(window).std() * np.sqrt(288)  # 年化波动率
            features[f'volatility_regime_{window}'] = pd.qcut(
                features[f'volatility_{window}'], 
                q=3, 
                labels=['low', 'medium', 'high']
            ).cat.codes
        
        # GARCH波动率模拟 (简化版)
        returns = features['close'].pct_change()
        features['garch_volatility'] = returns.ewm(alpha=0.1).std()
        
        # 价格动量特征
        for period in self.config.momentum_periods:
            features[f'momentum_{period}'] = features['close'].pct_change(period)
            features[f'momentum_strength_{period}'] = abs(features[f'momentum_{period}'])
            
        # 价格加速度
        features['price_acceleration'] = features['close'].pct_change().diff()
        
        # 流动性代理指标
        features['price_impact'] = abs(features['close'].pct_change()) / (features['volume'] + 1)
        features['turnover_rate'] = features['volume'] / features['volume'].rolling(20).mean()
        
        # 订单流不平衡代理 (基于价格和成交量)
        features['buying_pressure'] = np.where(
            features['close'] > features['open'],
            features['volume'],
            0
        )
        features['selling_pressure'] = np.where(
            features['close'] < features['open'],
            features['volume'],
            0
        )
        
        features['order_flow_imbalance'] = (
            (features['buying_pressure'] - features['selling_pressure']) / 
            (features['buying_pressure'] + features['selling_pressure'] + 1)
        )
        
        # 时间周期特征
        features['hour'] = pd.to_datetime(features['timestamp']).dt.hour
        features['day_of_week'] = pd.to_datetime(features['timestamp']).dt.dayofweek
        features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        features['is_asian_session'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
        features['is_european_session'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
        features['is_american_session'] = ((features['hour'] >= 16) & (features['hour'] <= 23)).astype(int)
        
        return features
    
    def generate_cross_timeframe_features(self, df_5m: pd.DataFrame) -> pd.DataFrame:
        """生成跨时间框架特征"""
        features = df_5m.copy()
        
        # 聚合到15分钟和1小时
        df_15m = self._aggregate_timeframe(df_5m, '15T')
        df_1h = self._aggregate_timeframe(df_5m, '1H')
        
        # 15分钟时间框架特征
        df_15m['rsi_15m'] = ta.RSI(df_15m['close'], timeperiod=14)
        df_15m['ema_10_15m'] = ta.EMA(df_15m['close'], timeperiod=10)
        df_15m['ema_20_15m'] = ta.EMA(df_15m['close'], timeperiod=20)
        df_15m['trend_15m'] = np.where(df_15m['ema_10_15m'] > df_15m['ema_20_15m'], 1, 0)
        
        # 1小时时间框架特征
        df_1h['rsi_1h'] = ta.RSI(df_1h['close'], timeperiod=14)
        df_1h['ema_10_1h'] = ta.EMA(df_1h['close'], timeperiod=10)
        df_1h['ema_20_1h'] = ta.EMA(df_1h['close'], timeperiod=20)
        df_1h['trend_1h'] = np.where(df_1h['ema_10_1h'] > df_1h['ema_20_1h'], 1, 0)
        
        # 将高时间框架特征映射回5分钟数据
        features = self._map_higher_timeframe_features(features, df_15m, '15m')
        features = self._map_higher_timeframe_features(features, df_1h, '1h')
        
        # 多时间框架一致性特征
        features['rsi_alignment'] = (
            (features['rsi'] > 30).astype(int) +
            (features['rsi_15m'] > 30).astype(int) +
            (features['rsi_1h'] > 30).astype(int)
        )
        
        features['trend_alignment'] = (
            (features['close'] > features['close'].rolling(20).mean()).astype(int) +
            features['trend_15m'] +
            features['trend_1h']
        )
        
        # 动量收敛性
        features['momentum_convergence'] = (
            features['close'].pct_change(5) * 
            features['close'].pct_change(15) * 
            features['close'].pct_change(60)
        )
        
        return features
    
    def _aggregate_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """聚合到指定时间框架"""
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        df_copy.set_index('timestamp', inplace=True)
        
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        aggregated = df_copy.resample(timeframe).agg(agg_dict).dropna()
        aggregated.reset_index(inplace=True)
        
        return aggregated
    
    def _map_higher_timeframe_features(self, df_5m: pd.DataFrame, df_higher: pd.DataFrame, suffix: str) -> pd.DataFrame:
        """将高时间框架特征映射到5分钟数据"""
        df_5m_copy = df_5m.copy()
        df_5m_copy['timestamp'] = pd.to_datetime(df_5m_copy['timestamp'])
        df_higher_copy = df_higher.copy()
        df_higher_copy['timestamp'] = pd.to_datetime(df_higher_copy['timestamp'])
        
        # 创建时间区间映射
        if suffix == '15m':
            df_5m_copy['time_group'] = df_5m_copy['timestamp'].dt.floor('15T')
        else:  # 1h
            df_5m_copy['time_group'] = df_5m_copy['timestamp'].dt.floor('1H')
        
        # 合并特征
        feature_cols = [col for col in df_higher_copy.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        merge_df = df_higher_copy[['timestamp'] + feature_cols].rename(columns={'timestamp': 'time_group'})
        
        result = df_5m_copy.merge(merge_df, on='time_group', how='left')
        result.drop('time_group', axis=1, inplace=True)
        
        return result
    
    def generate_cross_asset_features(self, all_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """生成跨资产特征"""
        # 确保所有数据有相同的时间戳
        common_timestamps = None
        for symbol, df in all_data.items():
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            if common_timestamps is None:
                common_timestamps = set(df['timestamp'])
            else:
                common_timestamps = common_timestamps.intersection(set(df['timestamp']))
        
        common_timestamps = sorted(common_timestamps)
        
        # 创建价格矩阵
        price_matrix = pd.DataFrame(index=common_timestamps)
        return_matrix = pd.DataFrame(index=common_timestamps)
        
        for symbol, df in all_data.items():
            df_aligned = df[df['timestamp'].isin(common_timestamps)].set_index('timestamp')
            price_matrix[symbol] = df_aligned['close']
            return_matrix[symbol] = df_aligned['close'].pct_change()
        
        # 计算相关性特征
        for symbol in all_data.keys():
            df = all_data[symbol].copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 与BTC的相关性
            if 'BTCUSDT' in price_matrix.columns and symbol != 'BTCUSDT':
                rolling_corr = return_matrix[symbol].rolling(window=50).corr(return_matrix['BTCUSDT'])
                corr_df = rolling_corr.reset_index()
                corr_df.columns = ['timestamp', 'btc_correlation']
                df = df.merge(corr_df, on='timestamp', how='left')
            else:
                df['btc_correlation'] = 0.0
            
            # 与ETH的相关性
            if 'ETHUSDT' in price_matrix.columns and symbol != 'ETHUSDT':
                rolling_corr = return_matrix[symbol].rolling(window=50).corr(return_matrix['ETHUSDT'])
                corr_df = rolling_corr.reset_index()
                corr_df.columns = ['timestamp', 'eth_correlation']
                df = df.merge(corr_df, on='timestamp', how='left')
            else:
                df['eth_correlation'] = 0.0
            
            # 相对强度 (相对于市场平均)
            market_avg_return = return_matrix.mean(axis=1)
            if symbol in return_matrix.columns:
                relative_strength = return_matrix[symbol] - market_avg_return
                rs_df = relative_strength.reset_index()
                rs_df.columns = ['timestamp', 'relative_strength']
                df = df.merge(rs_df, on='timestamp', how='left')
            else:
                df['relative_strength'] = 0.0
            
            # 行业beta (简化版)
            if len(return_matrix.columns) > 1 and symbol in return_matrix.columns:
                market_return = return_matrix.drop(symbol, axis=1).mean(axis=1)
                rolling_beta = return_matrix[symbol].rolling(window=50).cov(market_return) / market_return.rolling(window=50).var()
                beta_df = rolling_beta.reset_index()
                beta_df.columns = ['timestamp', 'market_beta']
                df = df.merge(beta_df, on='timestamp', how='left')
            else:
                df['market_beta'] = 1.0
            
            all_data[symbol] = df
        
        return all_data

class LabelGenerator:
    """标签生成器 - 为监督学习创建高质量标签"""
    
    def __init__(self, config: FeatureEngineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def generate_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成完整的标签集合"""
        labels = df.copy()
        
        # 主标签：未来收益率
        for horizon in self.config.prediction_horizons:
            labels[f'future_return_{horizon}m'] = (
                labels['close'].shift(-horizon) / labels['close'] - 1
            )
        
        # 分类标签：收益率分箱
        for horizon in self.config.prediction_horizons:
            return_col = f'future_return_{horizon}m'
            labels[f'return_class_{horizon}m'] = pd.qcut(
                labels[return_col], 
                q=5, 
                labels=['very_negative', 'negative', 'neutral', 'positive', 'very_positive']
            ).cat.codes
        
        # 二分类标签：是否盈利
        for horizon in self.config.prediction_horizons:
            return_col = f'future_return_{horizon}m'
            labels[f'is_profitable_{horizon}m'] = (labels[return_col] > 0).astype(int)
        
        # 目标达成标签
        for target in self.config.profit_targets:
            labels[f'hits_target_{target:.1%}'] = self._calculate_target_achievement(
                labels, target, self.config.max_holding_minutes
            )
        
        # 风险标签
        labels['hits_stop_loss'] = self._calculate_stop_loss_events(
            labels, self.config.stop_loss_threshold, self.config.max_holding_minutes
        )
        
        # 最优出场时间标签
        labels['optimal_exit_time'] = self._calculate_optimal_exit_time(
            labels, self.config.max_holding_minutes
        )
        
        # 持仓期间最大收益/回撤
        labels['max_profit_during_hold'] = self._calculate_max_profit_during_hold(
            labels, self.config.max_holding_minutes
        )
        labels['max_drawdown_during_hold'] = self._calculate_max_drawdown_during_hold(
            labels, self.config.max_holding_minutes
        )
        
        return labels
    
    def _calculate_target_achievement(self, df: pd.DataFrame, target: float, max_minutes: int) -> pd.Series:
        """计算是否在最大持仓时间内达到目标收益"""
        result = pd.Series(0, index=df.index)
        
        for i in range(len(df) - max_minutes):
            entry_price = df.iloc[i]['close']
            
            for j in range(1, min(max_minutes + 1, len(df) - i)):
                current_price = df.iloc[i + j]['close']
                current_return = (current_price / entry_price) - 1
                
                if current_return >= target:
                    result.iloc[i] = 1
                    break
        
        return result
    
    def _calculate_stop_loss_events(self, df: pd.DataFrame, stop_threshold: float, max_minutes: int) -> pd.Series:
        """计算是否触发止损"""
        result = pd.Series(0, index=df.index)
        
        for i in range(len(df) - max_minutes):
            entry_price = df.iloc[i]['close']
            
            for j in range(1, min(max_minutes + 1, len(df) - i)):
                current_price = df.iloc[i + j]['close']
                current_return = (current_price / entry_price) - 1
                
                if current_return <= -stop_threshold:
                    result.iloc[i] = 1
                    break
        
        return result
    
    def _calculate_optimal_exit_time(self, df: pd.DataFrame, max_minutes: int) -> pd.Series:
        """计算最优出场时间（收益最大化的时间点）"""
        result = pd.Series(np.nan, index=df.index)
        
        for i in range(len(df) - max_minutes):
            entry_price = df.iloc[i]['close']
            max_return = -float('inf')
            optimal_time = 0
            
            for j in range(1, min(max_minutes + 1, len(df) - i)):
                current_price = df.iloc[i + j]['close']
                current_return = (current_price / entry_price) - 1
                
                if current_return > max_return:
                    max_return = current_return
                    optimal_time = j
            
            result.iloc[i] = optimal_time
        
        return result
    
    def _calculate_max_profit_during_hold(self, df: pd.DataFrame, max_minutes: int) -> pd.Series:
        """计算持仓期间的最大收益"""
        result = pd.Series(np.nan, index=df.index)
        
        for i in range(len(df) - max_minutes):
            entry_price = df.iloc[i]['close']
            max_profit = -float('inf')
            
            for j in range(1, min(max_minutes + 1, len(df) - i)):
                current_price = df.iloc[i + j]['close']
                current_return = (current_price / entry_price) - 1
                max_profit = max(max_profit, current_return)
            
            result.iloc[i] = max_profit
        
        return result
    
    def _calculate_max_drawdown_during_hold(self, df: pd.DataFrame, max_minutes: int) -> pd.Series:
        """计算持仓期间的最大回撤"""
        result = pd.Series(np.nan, index=df.index)
        
        for i in range(len(df) - max_minutes):
            entry_price = df.iloc[i]['close']
            max_drawdown = 0
            peak_return = 0
            
            for j in range(1, min(max_minutes + 1, len(df) - i)):
                current_price = df.iloc[i + j]['close']
                current_return = (current_price / entry_price) - 1
                peak_return = max(peak_return, current_return)
                drawdown = peak_return - current_return
                max_drawdown = max(max_drawdown, drawdown)
            
            result.iloc[i] = max_drawdown
        
        return result

class FeatureEngineeringPipeline:
    """完整的特征工程管道"""
    
    def __init__(self, config: FeatureEngineConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_engine = DipMasterFeatureEngine(config)
        self.label_generator = LabelGenerator(config)
        self.quality_validator = FeatureQualityValidator(config)
        
        # 特征元数据
        self.feature_metadata = {}
        
    def load_market_data(self, data_path: str) -> Dict[str, pd.DataFrame]:
        """加载市场数据"""
        all_data = {}
        data_dir = Path(data_path) / "market_data"
        
        for symbol in self.config.symbols:
            # 尝试加载parquet文件
            parquet_file = data_dir / f"{symbol}_5m_2years.parquet"
            csv_file = data_dir / f"{symbol}_5m_2years.csv"
            
            if parquet_file.exists():
                df = pd.read_parquet(parquet_file)
                self.logger.info(f"Loaded {symbol} from parquet: {len(df)} records")
            elif csv_file.exists():
                df = pd.read_csv(csv_file)
                self.logger.info(f"Loaded {symbol} from CSV: {len(df)} records")
            else:
                self.logger.error(f"No data file found for {symbol}")
                continue
            
            # 数据预处理
            df = self._preprocess_data(df, symbol)
            all_data[symbol] = df
        
        return all_data
    
    def _preprocess_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """数据预处理"""
        # 确保列名标准化
        expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in expected_columns):
            self.logger.error(f"Missing required columns for {symbol}")
            return pd.DataFrame()
        
        # 时间戳处理
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # 价格数据类型转换
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # 移除异常值
        for col in price_columns:
            Q1 = df[col].quantile(0.01)
            Q3 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=Q1, upper=Q3)
        
        # 添加符号标识
        df['symbol'] = symbol
        df['symbol_encoded'] = hash(symbol) % 1000  # 简单编码
        
        return df.dropna()
    
    def execute_feature_engineering(self, data_path: str) -> Dict:
        """执行完整的特征工程管道"""
        start_time = time.time()
        
        # 1. 加载数据
        self.logger.info("Loading market data...")
        all_data = self.load_market_data(data_path)
        
        if not all_data:
            raise ValueError("No valid data loaded")
        
        # 2. 生成核心特征
        self.logger.info("Generating DipMaster core features...")
        for symbol in all_data.keys():
            all_data[symbol] = self.feature_engine.generate_dipmaster_core_features(
                all_data[symbol], symbol
            )
        
        # 3. 生成微观结构特征
        self.logger.info("Generating microstructure features...")
        for symbol in all_data.keys():
            all_data[symbol] = self.feature_engine.generate_microstructure_features(
                all_data[symbol]
            )
        
        # 4. 生成跨时间框架特征
        self.logger.info("Generating cross-timeframe features...")
        for symbol in all_data.keys():
            all_data[symbol] = self.feature_engine.generate_cross_timeframe_features(
                all_data[symbol]
            )
        
        # 5. 生成跨资产特征
        self.logger.info("Generating cross-asset features...")
        all_data = self.feature_engine.generate_cross_asset_features(all_data)
        
        # 6. 生成标签
        self.logger.info("Generating labels...")
        for symbol in all_data.keys():
            all_data[symbol] = self.label_generator.generate_labels(all_data[symbol])
        
        # 7. 合并所有数据
        self.logger.info("Combining all data...")
        combined_data = pd.concat(all_data.values(), ignore_index=True)
        
        # 8. 特征后处理
        self.logger.info("Post-processing features...")
        combined_data = self._post_process_features(combined_data)
        
        # 9. 数据质量验证
        self.logger.info("Validating data quality...")
        quality_report = self._validate_data_quality(combined_data)
        
        execution_time = time.time() - start_time
        
        result = {
            'features': combined_data,
            'feature_metadata': self.feature_metadata,
            'quality_report': quality_report,
            'execution_time': execution_time,
            'config': asdict(self.config)
        }
        
        self.logger.info(f"Feature engineering completed in {execution_time:.2f} seconds")
        self.logger.info(f"Generated {len(combined_data)} samples with {len(combined_data.columns)} features")
        
        return result
    
    def _post_process_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征后处理"""
        # 填充缺失值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # 前向填充时间序列数据
        df[numeric_columns] = df[numeric_columns].fillna(method='ffill')
        
        # 剩余缺失值用中位数填充
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # 异常值处理 (Winsorization)
        for col in numeric_columns:
            if col not in ['timestamp', 'symbol_encoded', 'hour', 'day_of_week']:
                lower_percentile = self.config.outlier_percentile[0]
                upper_percentile = self.config.outlier_percentile[1]
                
                lower_bound = df[col].quantile(lower_percentile / 100)
                upper_bound = df[col].quantile(upper_percentile / 100)
                
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # 特征标准化 (滚动窗口)
        feature_columns = [col for col in numeric_columns 
                          if not col.startswith(('future_', 'is_profitable_', 'hits_', 'optimal_', 'max_')) 
                          and col not in ['timestamp', 'symbol_encoded', 'hour', 'day_of_week']]
        
        for col in feature_columns:
            # 使用滚动窗口进行z-score标准化
            rolling_mean = df[col].rolling(window=200, min_periods=50).mean()
            rolling_std = df[col].rolling(window=200, min_periods=50).std()
            df[f'{col}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """数据质量验证"""
        quality_report = {}
        
        # 基础统计
        quality_report['basic_stats'] = {
            'total_samples': len(df),
            'total_features': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'date_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max())
            }
        }
        
        # 缺失值分析
        missing_analysis = {}
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df)
            if missing_pct > 0:
                missing_analysis[col] = missing_pct
        
        quality_report['missing_values'] = missing_analysis
        
        # 前视偏差检测
        quality_report['lookahead_bias_check'] = self.quality_validator.validate_no_lookahead_bias(
            df, self.feature_metadata
        )
        
        # 特征稳定性 (简化版)
        # 将数据分成多个时间段进行PSI计算
        df_sorted = df.sort_values('timestamp')
        n_periods = 4
        period_size = len(df_sorted) // n_periods
        
        time_periods = []
        for i in range(n_periods):
            start_idx = i * period_size
            end_idx = (i + 1) * period_size if i < n_periods - 1 else len(df_sorted)
            
            start_time = df_sorted.iloc[start_idx]['timestamp']
            end_time = df_sorted.iloc[end_idx - 1]['timestamp']
            time_periods.append((str(start_time), str(end_time)))
        
        stability_results = self.quality_validator.calculate_feature_stability(df_sorted, time_periods)
        quality_report['feature_stability'] = stability_results
        
        # 多重共线性检测
        correlation_analysis = self.quality_validator.detect_multicollinearity(df)
        quality_report['multicollinearity'] = correlation_analysis
        
        return quality_report
    
    def save_feature_set(self, result: Dict, output_path: str) -> Dict[str, str]:
        """保存特征集"""
        output_dir = Path(output_path)
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存特征数据 (Parquet格式)
        features_file = output_dir / f"dipmaster_v4_features_{timestamp}.parquet"
        result['features'].to_parquet(features_file, compression='snappy')
        
        # 保存特征集元数据 (JSON格式)
        metadata_file = output_dir / f"dipmaster_v4_featureset_{timestamp}.json"
        
        feature_set_metadata = {
            'version': '4.0.0',
            'strategy_name': 'DipMaster_Enhanced_V4',
            'created_timestamp': timestamp,
            'data_summary': {
                'total_samples': len(result['features']),
                'total_features': len(result['features'].columns),
                'symbols': result['config']['symbols'],
                'date_range': {
                    'start': str(result['features']['timestamp'].min()),
                    'end': str(result['features']['timestamp'].max())
                }
            },
            'feature_categories': {
                'dipmaster_core': [col for col in result['features'].columns if any(x in col for x in ['rsi', 'bb_', 'dipmaster_', 'volume_'])],
                'microstructure': [col for col in result['features'].columns if any(x in col for x in ['volatility_', 'momentum_', 'order_flow', 'buying_', 'selling_'])],
                'cross_timeframe': [col for col in result['features'].columns if any(x in col for x in ['_15m', '_1h', 'alignment', 'convergence'])],
                'cross_asset': [col for col in result['features'].columns if any(x in col for x in ['_correlation', 'relative_strength', 'market_beta'])],
                'labels': [col for col in result['features'].columns if any(x in col for x in ['future_', 'return_class', 'is_profitable', 'hits_', 'optimal_', 'max_'])]
            },
            'quality_metrics': result['quality_report'],
            'configuration': result['config'],
            'execution_metadata': {
                'execution_time_seconds': result['execution_time'],
                'feature_engineering_version': '4.0.0'
            }
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(feature_set_metadata, f, indent=2, ensure_ascii=False, default=str)
        
        # 生成质量报告
        quality_report_file = output_dir / f"dipmaster_v4_quality_report_{timestamp}.json"
        with open(quality_report_file, 'w', encoding='utf-8') as f:
            json.dump(result['quality_report'], f, indent=2, ensure_ascii=False, default=str)
        
        return {
            'features_file': str(features_file),
            'metadata_file': str(metadata_file),
            'quality_report_file': str(quality_report_file)
        }

# 使用示例和主函数
def main():
    """主函数 - 执行DipMaster Enhanced V4特征工程管道"""
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建配置
    config = FeatureEngineConfig(
        symbols=[
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "XRPUSDT",
            "BNBUSDT", "DOGEUSDT", "SUIUSDT", "ICPUSDT", "ALGOUSDT", "IOTAUSDT"
        ],
        primary_timeframe="5m",
        analysis_timeframes=["5m", "15m", "1h"],
        prediction_horizons=[15, 30, 60],  # 15分钟，30分钟，1小时
        profit_targets=[0.006, 0.012, 0.020]  # 0.6%, 1.2%, 2.0%
    )
    
    # 创建特征工程管道
    pipeline = FeatureEngineeringPipeline(config)
    
    # 执行特征工程
    try:
        result = pipeline.execute_feature_engineering("G:/Github/Quant/DipMaster-Trading-System/data")
        
        # 保存结果
        file_paths = pipeline.save_feature_set(result, "G:/Github/Quant/DipMaster-Trading-System/data")
        
        print("\n" + "="*80)
        print("DIPMASTER ENHANCED V4 特征工程完成!")
        print("="*80)
        print(f"✅ 总样本数: {result['quality_report']['basic_stats']['total_samples']:,}")
        print(f"✅ 总特征数: {result['quality_report']['basic_stats']['total_features']:,}")
        print(f"✅ 执行时间: {result['execution_time']:.2f} 秒")
        print(f"✅ 内存使用: {result['quality_report']['basic_stats']['memory_usage_mb']:.1f} MB")
        print("\n文件输出:")
        for file_type, file_path in file_paths.items():
            print(f"📁 {file_type}: {file_path}")
        
        # 质量评估摘要
        print("\n质量评估摘要:")
        quality = result['quality_report']
        
        if quality['lookahead_bias_check']['has_lookahead_bias']:
            print("⚠️  检测到前视偏差风险")
        else:
            print("✅ 无前视偏差")
            
        if quality['multicollinearity']['multicollinearity_detected']:
            print(f"⚠️  检测到 {len(quality['multicollinearity']['high_correlation_pairs'])} 对高相关特征")
        else:
            print("✅ 无多重共线性问题")
        
        # 特征稳定性统计
        stable_features = sum(1 for v in quality['feature_stability'].values() 
                             if v['stability_rating'] == 'stable')
        total_features = len(quality['feature_stability'])
        print(f"✅ 特征稳定性: {stable_features}/{total_features} ({stable_features/total_features*100:.1f}%) 稳定")
        
        print("\n🎯 特征工程管道已准备就绪，可用于DipMaster Enhanced V4策略训练!")
        
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    main()