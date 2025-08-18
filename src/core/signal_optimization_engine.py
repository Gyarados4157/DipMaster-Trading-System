#!/usr/bin/env python3
"""
策略信号优化引擎
基于模型概率生成信号强度，实施动态阈值调整和智能过滤器
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path

# 机器学习和统计
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

# 信号处理
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SignalConfig:
    """信号配置参数"""
    # 基础阈值参数
    base_threshold: float = 0.5
    min_threshold: float = 0.3
    max_threshold: float = 0.8
    
    # 动态调整参数
    volatility_adjustment: bool = True
    volume_adjustment: bool = True
    trend_adjustment: bool = True
    time_adjustment: bool = True
    
    # 信号过滤器参数
    min_signal_strength: float = 0.6
    signal_decay_hours: int = 4
    duplicate_signal_window: int = 3  # 小时
    
    # 市场制度过滤
    high_vol_threshold_multiplier: float = 1.2
    low_vol_threshold_multiplier: float = 0.8
    trending_threshold_multiplier: float = 1.1
    ranging_threshold_multiplier: float = 0.9
    
    # 时间过滤器
    preferred_hours: List[int] = None
    avoid_hours: List[int] = None
    
    # 质量控制
    max_daily_signals: int = 10
    min_time_between_signals: int = 1  # 小时
    signal_confidence_weight: float = 0.3

    def __post_init__(self):
        if self.preferred_hours is None:
            # DipMaster策略偏好时间（UTC）
            self.preferred_hours = [2, 3, 6, 7, 10, 11, 14, 15, 18, 19, 22, 23]
        if self.avoid_hours is None:
            # 避免低流动性时间
            self.avoid_hours = [0, 1, 4, 5]

class MarketRegimeDetector:
    """市场制度检测器"""
    
    def __init__(self):
        self.volatility_lookback = 48  # 4小时回看（5分钟数据）
        self.trend_lookback = 96      # 8小时回看
        
    def detect_market_regime(self, prices: pd.Series, volumes: pd.Series = None) -> Dict:
        """检测市场制度"""
        if len(prices) < self.volatility_lookback:
            return self._default_regime()
            
        try:
            # 计算收益率
            returns = prices.pct_change().dropna()
            
            # 波动率制度
            rolling_vol = returns.rolling(self.volatility_lookback).std()
            current_vol = rolling_vol.iloc[-1] if len(rolling_vol) > 0 else 0.02
            historical_vol = rolling_vol.quantile(0.5) if len(rolling_vol) > 0 else 0.02
            
            vol_regime = 'high' if current_vol > historical_vol * 1.5 else 'low'
            
            # 趋势制度
            trend_lookback_prices = prices.tail(self.trend_lookback)
            if len(trend_lookback_prices) >= 10:
                slope, _, r_value, _, _ = stats.linregress(
                    range(len(trend_lookback_prices)), 
                    trend_lookback_prices.values
                )
                
                # 趋势强度
                trend_strength = abs(r_value)
                trend_direction = 'up' if slope > 0 else 'down'
                
                if trend_strength > 0.7:
                    trend_regime = f'trending_{trend_direction}'
                else:
                    trend_regime = 'ranging'
            else:
                trend_regime = 'ranging'
                trend_strength = 0
            
            # 流动性制度（如果有成交量数据）
            liquidity_regime = 'normal'
            if volumes is not None and len(volumes) > 0:
                recent_volume = volumes.tail(24).mean()  # 2小时平均
                historical_volume = volumes.quantile(0.5)
                
                if recent_volume > historical_volume * 1.5:
                    liquidity_regime = 'high'
                elif recent_volume < historical_volume * 0.5:
                    liquidity_regime = 'low'
            
            return {
                'volatility_regime': vol_regime,
                'current_volatility': current_vol,
                'historical_volatility': historical_vol,
                'trend_regime': trend_regime,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction if 'trending' in trend_regime else 'neutral',
                'liquidity_regime': liquidity_regime
            }
            
        except Exception as e:
            logger.warning(f"市场制度检测失败: {e}")
            return self._default_regime()
    
    def _default_regime(self) -> Dict:
        """默认市场制度"""
        return {
            'volatility_regime': 'normal',
            'current_volatility': 0.02,
            'historical_volatility': 0.02,
            'trend_regime': 'ranging',
            'trend_strength': 0.5,
            'trend_direction': 'neutral',
            'liquidity_regime': 'normal'
        }

class DynamicThresholdOptimizer:
    """动态阈值优化器"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        
    def optimize_threshold(self, predictions: np.ndarray, actual_returns: np.ndarray, 
                          market_data: pd.DataFrame = None) -> Dict:
        """优化阈值"""
        logger.info("优化动态阈值...")
        
        # 基础阈值优化
        base_results = self._optimize_base_threshold(predictions, actual_returns)
        
        # 如果有市场数据，进行条件阈值优化
        if market_data is not None:
            conditional_results = self._optimize_conditional_thresholds(
                predictions, actual_returns, market_data
            )
        else:
            conditional_results = {}
        
        return {
            'base_threshold': base_results,
            'conditional_thresholds': conditional_results,
            'optimization_timestamp': datetime.now().isoformat()
        }
    
    def _optimize_base_threshold(self, predictions: np.ndarray, actual_returns: np.ndarray) -> Dict:
        """优化基础阈值"""
        thresholds = np.linspace(0.1, 0.9, 50)
        results = []
        
        for threshold in thresholds:
            signals = (predictions >= threshold).astype(int)
            
            if signals.sum() == 0:
                continue
            
            # 计算性能指标
            signal_returns = actual_returns[signals == 1]
            
            if len(signal_returns) < 10:
                continue
            
            win_rate = (signal_returns > 0).mean()
            avg_return = signal_returns.mean()
            std_return = signal_returns.std()
            
            # 风险调整收益
            sharpe = avg_return / std_return if std_return > 0 else 0
            
            # 综合得分
            score = win_rate * 0.4 + sharpe * 0.4 + (avg_return > 0) * 0.2
            
            results.append({
                'threshold': threshold,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'sharpe': sharpe,
                'num_signals': len(signal_returns),
                'score': score
            })
        
        if not results:
            return {'optimal_threshold': self.config.base_threshold, 'score': 0}
        
        # 选择最佳阈值
        best_result = max(results, key=lambda x: x['score'])
        
        return {
            'optimal_threshold': best_result['threshold'],
            'performance': best_result,
            'all_results': results
        }
    
    def _optimize_conditional_thresholds(self, predictions: np.ndarray, actual_returns: np.ndarray,
                                       market_data: pd.DataFrame) -> Dict:
        """优化条件阈值"""
        conditional_thresholds = {}
        
        # 检测器
        regime_detector = MarketRegimeDetector()
        
        # 按时间分段优化
        time_groups = self._create_time_groups(market_data)
        
        for group_name, group_indices in time_groups.items():
            if len(group_indices) < 50:  # 样本太少
                continue
                
            group_predictions = predictions[group_indices]
            group_returns = actual_returns[group_indices]
            
            # 基础优化
            group_result = self._optimize_base_threshold(group_predictions, group_returns)
            conditional_thresholds[group_name] = group_result['optimal_threshold']
        
        # 按市场制度优化
        if 'close' in market_data.columns:
            prices = market_data['close']
            volumes = market_data.get('volume', None)
            
            # 分段检测市场制度
            window_size = 100  # 约8小时的数据
            regime_thresholds = {}
            
            for i in range(0, len(market_data) - window_size, window_size // 2):
                end_idx = min(i + window_size, len(market_data))
                window_indices = range(i, end_idx)
                
                # 检测制度
                window_prices = prices.iloc[window_indices]
                window_volumes = volumes.iloc[window_indices] if volumes is not None else None
                regime = regime_detector.detect_market_regime(window_prices, window_volumes)
                
                # 优化阈值
                window_predictions = predictions[window_indices]
                window_returns = actual_returns[window_indices]
                
                regime_key = f"{regime['volatility_regime']}_{regime['trend_regime']}"
                
                if regime_key not in regime_thresholds:
                    regime_thresholds[regime_key] = []
                
                if len(window_predictions) >= 20:
                    result = self._optimize_base_threshold(window_predictions, window_returns)
                    if result['optimal_threshold'] > 0:
                        regime_thresholds[regime_key].append(result['optimal_threshold'])
            
            # 计算制度平均阈值
            for regime_key, thresholds in regime_thresholds.items():
                if thresholds:
                    conditional_thresholds[f'regime_{regime_key}'] = np.median(thresholds)
        
        return conditional_thresholds
    
    def _create_time_groups(self, market_data: pd.DataFrame) -> Dict[str, List[int]]:
        """创建时间分组"""
        if 'timestamp' not in market_data.columns:
            return {}
        
        timestamps = pd.to_datetime(market_data['timestamp'])
        hours = timestamps.dt.hour
        days_of_week = timestamps.dt.dayofweek
        
        groups = {
            'asian_session': [],      # 0-8 UTC
            'european_session': [],   # 8-16 UTC
            'american_session': [],   # 16-24 UTC
            'weekday': [],           # Mon-Fri
            'weekend': []            # Sat-Sun
        }
        
        for i, (hour, dow) in enumerate(zip(hours, days_of_week)):
            # 时区分组
            if 0 <= hour < 8:
                groups['asian_session'].append(i)
            elif 8 <= hour < 16:
                groups['european_session'].append(i)
            else:
                groups['american_session'].append(i)
            
            # 工作日/周末分组
            if dow < 5:
                groups['weekday'].append(i)
            else:
                groups['weekend'].append(i)
        
        return groups

class SignalStrengthCalculator:
    """信号强度计算器"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        
    def calculate_signal_strength(self, prediction_proba: float, 
                                features: Dict, market_regime: Dict) -> float:
        """计算信号强度"""
        base_strength = prediction_proba
        
        # 1. 概率置信度调整
        confidence_adjustment = self._calculate_confidence_adjustment(prediction_proba)
        
        # 2. 市场制度调整
        regime_adjustment = self._calculate_regime_adjustment(market_regime)
        
        # 3. 特征支持度调整
        feature_adjustment = self._calculate_feature_adjustment(features)
        
        # 4. 时间因子调整
        time_adjustment = self._calculate_time_adjustment(features.get('timestamp'))
        
        # 综合信号强度
        total_strength = (
            base_strength * 0.4 +
            confidence_adjustment * 0.2 +
            regime_adjustment * 0.2 +
            feature_adjustment * 0.1 +
            time_adjustment * 0.1
        )
        
        # 确保在合理范围内
        total_strength = np.clip(total_strength, 0.0, 1.0)
        
        return total_strength
    
    def _calculate_confidence_adjustment(self, prediction_proba: float) -> float:
        """计算概率置信度调整"""
        # 距离0.5越远，置信度越高
        distance_from_neutral = abs(prediction_proba - 0.5)
        confidence = distance_from_neutral * 2  # 归一化到[0,1]
        
        # 非线性调整，增强极端值
        confidence_adjusted = confidence ** 0.8
        
        return confidence_adjusted
    
    def _calculate_regime_adjustment(self, market_regime: Dict) -> float:
        """计算市场制度调整"""
        adjustment = 0.5  # 基础值
        
        # 波动率制度调整
        vol_regime = market_regime.get('volatility_regime', 'normal')
        if vol_regime == 'high':
            adjustment += 0.1  # 高波动率有利于DipMaster
        elif vol_regime == 'low':
            adjustment -= 0.1
        
        # 趋势制度调整
        trend_regime = market_regime.get('trend_regime', 'ranging')
        trend_strength = market_regime.get('trend_strength', 0.5)
        
        if 'trending_down' in trend_regime:
            adjustment += 0.15 * trend_strength  # 下跌趋势有利
        elif 'trending_up' in trend_regime:
            adjustment -= 0.1 * trend_strength   # 上涨趋势不利
        
        # 流动性制度调整
        liquidity_regime = market_regime.get('liquidity_regime', 'normal')
        if liquidity_regime == 'low':
            adjustment -= 0.05  # 低流动性风险
        
        return np.clip(adjustment, 0.0, 1.0)
    
    def _calculate_feature_adjustment(self, features: Dict) -> float:
        """计算特征支持度调整"""
        adjustment = 0.5
        
        # RSI特征支持
        rsi_features = [k for k in features.keys() if 'rsi' in k.lower()]
        for rsi_key in rsi_features:
            rsi_value = features.get(rsi_key, 50)
            if isinstance(rsi_value, (int, float)):
                # DipMaster偏好30-50的RSI
                if 30 <= rsi_value <= 50:
                    adjustment += 0.1
                elif rsi_value < 30:
                    adjustment += 0.05  # 超卖有一定支持
        
        # 动量特征支持
        momentum_features = [k for k in features.keys() if 'momentum' in k.lower()]
        negative_momentum_count = 0
        for mom_key in momentum_features:
            mom_value = features.get(mom_key, 0)
            if isinstance(mom_value, (int, float)) and mom_value < -0.001:
                negative_momentum_count += 1
        
        if negative_momentum_count >= 2:
            adjustment += 0.1  # 多个负动量支持
        
        # Pin Bar特征支持
        pin_bar_score = features.get('pin_strength_score', 0)
        if pin_bar_score > 0.5:
            adjustment += 0.05
        
        return np.clip(adjustment, 0.0, 1.0)
    
    def _calculate_time_adjustment(self, timestamp) -> float:
        """计算时间因子调整"""
        if timestamp is None:
            return 0.5
            
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
        
        hour = timestamp.hour
        
        # DipMaster偏好时间
        if hour in self.config.preferred_hours:
            return 0.7
        elif hour in self.config.avoid_hours:
            return 0.3
        else:
            return 0.5

class SignalFilterEngine:
    """信号过滤引擎"""
    
    def __init__(self, config: SignalConfig):
        self.config = config
        self.signal_history = []
        
    def filter_signals(self, signals: List[Dict]) -> List[Dict]:
        """过滤信号"""
        if not signals:
            return []
        
        filtered_signals = []
        
        for signal in signals:
            # 1. 基础质量过滤
            if not self._passes_basic_quality_filter(signal):
                continue
            
            # 2. 重复信号过滤
            if self._is_duplicate_signal(signal):
                continue
            
            # 3. 时间间隔过滤
            if not self._passes_time_interval_filter(signal):
                continue
            
            # 4. 日限额过滤
            if not self._passes_daily_limit_filter(signal):
                continue
            
            # 5. 衰减调整
            signal = self._apply_signal_decay(signal)
            
            filtered_signals.append(signal)
            
        # 更新历史记录
        self.signal_history.extend(filtered_signals)
        self._cleanup_signal_history()
        
        logger.info(f"信号过滤: {len(signals)} → {len(filtered_signals)}")
        
        return filtered_signals
    
    def _passes_basic_quality_filter(self, signal: Dict) -> bool:
        """基础质量过滤"""
        signal_strength = signal.get('strength', 0)
        confidence = signal.get('confidence', 0)
        
        if signal_strength < self.config.min_signal_strength:
            return False
        
        if confidence < 0.5:
            return False
        
        return True
    
    def _is_duplicate_signal(self, signal: Dict) -> bool:
        """检查重复信号"""
        symbol = signal.get('symbol', '')
        timestamp = signal.get('timestamp')
        
        if not timestamp or not symbol:
            return False
        
        current_time = pd.to_datetime(timestamp)
        window_start = current_time - timedelta(hours=self.config.duplicate_signal_window)
        
        for hist_signal in self.signal_history:
            hist_symbol = hist_signal.get('symbol', '')
            hist_timestamp = pd.to_datetime(hist_signal.get('timestamp'))
            
            if (hist_symbol == symbol and 
                hist_timestamp >= window_start and 
                hist_timestamp <= current_time):
                return True
        
        return False
    
    def _passes_time_interval_filter(self, signal: Dict) -> bool:
        """时间间隔过滤"""
        if not self.signal_history:
            return True
        
        current_time = pd.to_datetime(signal.get('timestamp'))
        min_interval = timedelta(hours=self.config.min_time_between_signals)
        
        for hist_signal in reversed(self.signal_history):
            hist_time = pd.to_datetime(hist_signal.get('timestamp'))
            
            if current_time - hist_time < min_interval:
                return False
            
            # 只检查最近的信号即可
            break
        
        return True
    
    def _passes_daily_limit_filter(self, signal: Dict) -> bool:
        """日限额过滤"""
        current_time = pd.to_datetime(signal.get('timestamp'))
        today_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        
        today_signals = [
            s for s in self.signal_history
            if today_start <= pd.to_datetime(s.get('timestamp')) < today_end
        ]
        
        return len(today_signals) < self.config.max_daily_signals
    
    def _apply_signal_decay(self, signal: Dict) -> Dict:
        """应用信号衰减"""
        # 简单的时间衰减模型
        # 在实际应用中，这里可以根据信号的年龄进行强度调整
        signal_copy = signal.copy()
        
        # 这里可以添加更复杂的衰减逻辑
        # 例如根据信号产生到现在的时间来调整强度
        
        return signal_copy
    
    def _cleanup_signal_history(self):
        """清理信号历史"""
        if not self.signal_history:
            return
        
        # 保留最近24小时的信号历史
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        self.signal_history = [
            signal for signal in self.signal_history
            if pd.to_datetime(signal.get('timestamp', datetime.now())) >= cutoff_time
        ]

class SignalOptimizationEngine:
    """信号优化引擎主类"""
    
    def __init__(self, config: SignalConfig = None):
        self.config = config or SignalConfig()
        self.threshold_optimizer = DynamicThresholdOptimizer(self.config)
        self.strength_calculator = SignalStrengthCalculator(self.config)
        self.filter_engine = SignalFilterEngine(self.config)
        self.regime_detector = MarketRegimeDetector()
        
        # 状态存储
        self.current_thresholds = {}
        self.performance_history = []
        
    def generate_optimized_signals(self, model_predictions: np.ndarray, 
                                 features_df: pd.DataFrame,
                                 market_data: pd.DataFrame = None) -> List[Dict]:
        """生成优化信号"""
        logger.info("生成优化交易信号...")
        
        if len(model_predictions) != len(features_df):
            raise ValueError("预测数组与特征DataFrame长度不匹配")
        
        signals = []
        
        for i, prediction_proba in enumerate(model_predictions):
            try:
                # 获取当前行特征
                current_features = features_df.iloc[i].to_dict()
                
                # 检测市场制度
                if market_data is not None and i >= 48:  # 需要足够历史数据
                    window_data = market_data.iloc[max(0, i-48):i+1]
                    market_regime = self.regime_detector.detect_market_regime(
                        window_data['close'], 
                        window_data.get('volume')
                    )
                else:
                    market_regime = self.regime_detector._default_regime()
                
                # 计算动态阈值
                dynamic_threshold = self._calculate_dynamic_threshold(
                    current_features, market_regime
                )
                
                # 检查是否超过阈值
                if prediction_proba >= dynamic_threshold:
                    # 计算信号强度
                    signal_strength = self.strength_calculator.calculate_signal_strength(
                        prediction_proba, current_features, market_regime
                    )
                    
                    # 创建信号
                    signal = {
                        'timestamp': current_features.get('timestamp', datetime.now()),
                        'symbol': self._extract_symbol(current_features),
                        'prediction_proba': prediction_proba,
                        'threshold': dynamic_threshold,
                        'strength': signal_strength,
                        'confidence': prediction_proba,
                        'market_regime': market_regime,
                        'features': current_features
                    }
                    
                    signals.append(signal)
                    
            except Exception as e:
                logger.warning(f"生成信号时出错 (index {i}): {e}")
                continue
        
        # 应用过滤器
        filtered_signals = self.filter_engine.filter_signals(signals)
        
        logger.info(f"生成信号: {len(signals)} 原始 → {len(filtered_signals)} 过滤后")
        
        return filtered_signals
    
    def _calculate_dynamic_threshold(self, features: Dict, market_regime: Dict) -> float:
        """计算动态阈值"""
        base_threshold = self.config.base_threshold
        
        # 市场制度调整
        regime_multiplier = 1.0
        
        vol_regime = market_regime.get('volatility_regime', 'normal')
        if vol_regime == 'high':
            regime_multiplier *= self.config.high_vol_threshold_multiplier
        elif vol_regime == 'low':
            regime_multiplier *= self.config.low_vol_threshold_multiplier
        
        trend_regime = market_regime.get('trend_regime', 'ranging')
        if 'trending' in trend_regime:
            regime_multiplier *= self.config.trending_threshold_multiplier
        else:
            regime_multiplier *= self.config.ranging_threshold_multiplier
        
        # 时间调整
        timestamp = features.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                timestamp = pd.to_datetime(timestamp)
            
            hour = timestamp.hour
            if hour in self.config.preferred_hours:
                regime_multiplier *= 0.9  # 降低阈值，更容易触发
            elif hour in self.config.avoid_hours:
                regime_multiplier *= 1.2  # 提高阈值，更难触发
        
        # 计算最终阈值
        dynamic_threshold = base_threshold * regime_multiplier
        
        # 限制在配置范围内
        dynamic_threshold = np.clip(
            dynamic_threshold, 
            self.config.min_threshold, 
            self.config.max_threshold
        )
        
        return dynamic_threshold
    
    def _extract_symbol(self, features: Dict) -> str:
        """从特征中提取交易对名称"""
        # 查找包含交易对名称的特征
        for key in features.keys():
            if 'USDT' in key:
                parts = key.split('_')
                for part in parts:
                    if 'USDT' in part:
                        return part
        
        # 默认返回
        return 'UNKNOWN'
    
    def optimize_thresholds_from_backtest(self, predictions: np.ndarray, 
                                        actual_returns: np.ndarray,
                                        market_data: pd.DataFrame = None) -> Dict:
        """从回测结果优化阈值"""
        logger.info("基于回测结果优化阈值...")
        
        optimization_results = self.threshold_optimizer.optimize_threshold(
            predictions, actual_returns, market_data
        )
        
        # 更新当前阈值
        if 'base_threshold' in optimization_results:
            optimal_base = optimization_results['base_threshold'].get('optimal_threshold')
            if optimal_base:
                self.config.base_threshold = optimal_base
                logger.info(f"更新基础阈值: {optimal_base:.3f}")
        
        return optimization_results
    
    def get_signal_quality_metrics(self, signals: List[Dict], 
                                 actual_outcomes: List[float] = None) -> Dict:
        """获取信号质量指标"""
        if not signals:
            return {'error': 'No signals to analyze'}
        
        metrics = {
            'total_signals': len(signals),
            'avg_strength': np.mean([s['strength'] for s in signals]),
            'avg_confidence': np.mean([s['confidence'] for s in signals]),
            'strength_distribution': self._calculate_distribution([s['strength'] for s in signals]),
            'time_distribution': self._analyze_time_distribution(signals),
            'symbol_distribution': self._analyze_symbol_distribution(signals)
        }
        
        # 如果有实际结果，计算表现指标
        if actual_outcomes and len(actual_outcomes) == len(signals):
            metrics['performance'] = {
                'win_rate': (np.array(actual_outcomes) > 0).mean(),
                'avg_return': np.mean(actual_outcomes),
                'total_return': np.sum(actual_outcomes),
                'best_signal_return': np.max(actual_outcomes),
                'worst_signal_return': np.min(actual_outcomes)
            }
        
        return metrics
    
    def _calculate_distribution(self, values: List[float]) -> Dict:
        """计算数值分布"""
        if not values:
            return {}
            
        return {
            'min': np.min(values),
            'max': np.max(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'quartiles': {
                'q25': np.percentile(values, 25),
                'q50': np.percentile(values, 50),
                'q75': np.percentile(values, 75)
            }
        }
    
    def _analyze_time_distribution(self, signals: List[Dict]) -> Dict:
        """分析时间分布"""
        timestamps = [pd.to_datetime(s['timestamp']) for s in signals if s.get('timestamp')]
        
        if not timestamps:
            return {}
        
        hours = [ts.hour for ts in timestamps]
        days_of_week = [ts.dayofweek for ts in timestamps]
        
        return {
            'hour_distribution': {f'hour_{h}': hours.count(h) for h in range(24)},
            'day_of_week_distribution': {f'day_{d}': days_of_week.count(d) for d in range(7)},
            'peak_hours': [h for h in range(24) if hours.count(h) == max([hours.count(i) for i in range(24)])]
        }
    
    def _analyze_symbol_distribution(self, signals: List[Dict]) -> Dict:
        """分析交易对分布"""
        symbols = [s.get('symbol', 'UNKNOWN') for s in signals]
        symbol_counts = {}
        
        for symbol in symbols:
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        return symbol_counts
    
    def save_optimization_results(self, results: Dict, output_path: str = None):
        """保存优化结果"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"signal_optimization_results_{timestamp}.json"
        
        # 转换为JSON可序列化格式
        import json
        
        def make_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.float64)):
                return obj.item()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
        
        serializable_results = {}
        for key, value in results.items():
            try:
                serializable_results[key] = make_serializable(value)
            except Exception as e:
                logger.warning(f"无法序列化 {key}: {e}")
                serializable_results[key] = str(value)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 优化结果已保存: {output_path}")
        
        return output_path

# 示例使用
def main():
    """主函数示例"""
    # 创建配置
    config = SignalConfig()
    
    # 创建优化引擎
    optimizer = SignalOptimizationEngine(config)
    
    # 模拟数据
    np.random.seed(42)
    n_samples = 1000
    
    # 模拟模型预测
    predictions = np.random.beta(2, 2, n_samples)  # 更真实的概率分布
    
    # 模拟特征数据
    features_df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
        'BTCUSDT_rsi_14': np.random.uniform(20, 80, n_samples),
        'BTCUSDT_momentum_5m': np.random.normal(0, 0.01, n_samples),
        'BTCUSDT_pin_strength_score': np.random.uniform(0, 1, n_samples)
    })
    
    # 模拟市场数据
    market_data = pd.DataFrame({
        'timestamp': features_df['timestamp'],
        'close': 50000 + np.cumsum(np.random.normal(0, 100, n_samples)),
        'volume': np.random.exponential(1000, n_samples)
    })
    
    # 生成优化信号
    signals = optimizer.generate_optimized_signals(
        predictions, features_df, market_data
    )
    
    print(f"生成了 {len(signals)} 个优化信号")
    
    if signals:
        # 分析信号质量
        quality_metrics = optimizer.get_signal_quality_metrics(signals)
        print("信号质量指标:")
        print(f"  平均强度: {quality_metrics['avg_strength']:.3f}")
        print(f"  平均置信度: {quality_metrics['avg_confidence']:.3f}")
        print(f"  交易对分布: {quality_metrics['symbol_distribution']}")
    
    return signals

if __name__ == "__main__":
    main()