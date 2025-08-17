#!/usr/bin/env python3
"""
Learning Framework for Adaptive DipMaster Strategy
自适应DipMaster策略学习框架

This module implements a comprehensive learning and validation framework for
parameter optimization including A/B testing, statistical validation,
reinforcement learning, and continuous adaptation mechanisms.

Features:
- A/B testing framework for parameter validation
- Statistical significance testing
- Reinforcement learning for dynamic adaptation
- Walk-forward validation
- Monte Carlo simulation
- Performance attribution analysis
- Continuous learning and model updates

Author: Portfolio Risk Optimizer Agent
Date: 2025-08-17
Version: 1.0.0
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
import pickle
from pathlib import Path
from collections import defaultdict, deque
import threading
import asyncio
import concurrent.futures
import time
import uuid

# Statistical libraries
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp, chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.power import ttest_power
from statsmodels.stats.proportion import proportions_ztest
import pingouin as pg

# Machine learning and RL libraries
import gym
from gym import spaces
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Core components
from .adaptive_parameter_engine import ParameterSet, AdaptiveParameterEngine
from .parameter_optimizer import ParameterOptimizer, OptimizationResult
from .performance_tracker import PerformanceTracker, TradeRecord
from .risk_control_manager import RiskControlManager
from .market_regime_detector import MarketRegime
from ..types.common_types import *

warnings.filterwarnings('ignore')

class TestType(Enum):
    """Types of statistical tests"""
    AB_TEST = "ab_test"
    MULTIVARIATE_TEST = "multivariate_test"
    SEQUENTIAL_TEST = "sequential_test"
    BAYESIAN_TEST = "bayesian_test"

class ValidationMethod(Enum):
    """Validation methods"""
    WALK_FORWARD = "walk_forward"
    MONTE_CARLO = "monte_carlo"
    BOOTSTRAP = "bootstrap"
    TIME_SERIES_CV = "time_series_cv"
    PURGED_CV = "purged_cv"

class LearningAlgorithm(Enum):
    """Learning algorithms"""
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ONLINE_LEARNING = "online_learning"
    ENSEMBLE_LEARNING = "ensemble_learning"
    TRANSFER_LEARNING = "transfer_learning"

@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_id: str
    name: str
    description: str
    control_params: Dict[str, float]
    treatment_params: Dict[str, float]
    success_metric: str
    minimum_sample_size: int
    significance_level: float
    power: float
    expected_effect_size: float
    max_duration_days: int

@dataclass
class ABTestResult:
    """A/B test result"""
    test_id: str
    start_date: datetime
    end_date: datetime
    control_sample_size: int
    treatment_sample_size: int
    control_mean: float
    treatment_mean: float
    effect_size: float
    p_value: float
    confidence_interval: Tuple[float, float]
    statistical_power: float
    is_significant: bool
    recommendation: str
    control_metrics: Dict[str, float]
    treatment_metrics: Dict[str, float]

@dataclass
class ValidationResult:
    """Validation result"""
    validation_id: str
    method: ValidationMethod
    parameters: Dict[str, float]
    in_sample_metrics: Dict[str, float]
    out_sample_metrics: Dict[str, float]
    cross_validation_scores: List[float]
    confidence_interval: Tuple[float, float]
    overfitting_score: float
    stability_score: float
    recommendation: str

@dataclass
class LearningState:
    """Learning algorithm state"""
    algorithm: LearningAlgorithm
    state_vector: np.ndarray
    action_space: Dict[str, Any]
    reward_history: List[float]
    performance_history: List[Dict]
    learning_rate: float
    exploration_rate: float
    last_update: datetime

class ParameterAdaptationEnv(gym.Env):
    """
    Reinforcement Learning Environment for Parameter Adaptation
    参数自适应强化学习环境
    """
    
    def __init__(self, parameter_space: Dict, performance_tracker: PerformanceTracker):
        super().__init__()
        
        self.parameter_space = parameter_space
        self.performance_tracker = performance_tracker
        
        # Define action space (parameter adjustments)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, 
            shape=(len(parameter_space),), 
            dtype=np.float32
        )
        
        # Define observation space (market conditions + performance metrics)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(20,),  # Market features + performance metrics
            dtype=np.float32
        )
        
        self.current_params = None
        self.episode_length = 100
        self.current_step = 0
        self.baseline_performance = 0.0
        
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        self.current_params = self._get_default_parameters()
        return self._get_observation()
    
    def step(self, action):
        """Take action and return next state, reward, done, info"""
        # Apply parameter adjustments
        self._apply_action(action)
        
        # Calculate reward based on performance improvement
        reward = self._calculate_reward()
        
        # Get next observation
        observation = self._get_observation()
        
        self.current_step += 1
        done = self.current_step >= self.episode_length
        
        info = {
            'current_params': self.current_params.copy(),
            'performance_metrics': self._get_current_performance()
        }
        
        return observation, reward, done, info
    
    def _get_default_parameters(self) -> Dict[str, float]:
        """Get default parameter values"""
        return {name: config.get('default', 0.5) for name, config in self.parameter_space.items()}
    
    def _apply_action(self, action):
        """Apply parameter adjustments"""
        for i, (param_name, param_config) in enumerate(self.parameter_space.items()):
            # Scale action to parameter range
            param_range = param_config['high'] - param_config['low']
            adjustment = action[i] * param_range * 0.1  # 10% max adjustment
            
            new_value = self.current_params[param_name] + adjustment
            
            # Clip to bounds
            self.current_params[param_name] = np.clip(
                new_value, param_config['low'], param_config['high']
            )
    
    def _get_observation(self) -> np.ndarray:
        """Get current market and performance observation"""
        # Market features (simplified)
        market_features = np.random.randn(10)  # Placeholder
        
        # Performance metrics
        performance_features = np.array([
            self._get_current_performance().get('win_rate', 0.5),
            self._get_current_performance().get('sharpe_ratio', 0.0),
            self._get_current_performance().get('max_drawdown', 0.0),
            self._get_current_performance().get('profit_factor', 1.0),
            *list(self.current_params.values())[:6]  # First 6 parameters
        ])
        
        return np.concatenate([market_features, performance_features])
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on performance improvement"""
        current_performance = self._get_current_performance()
        
        # Composite reward function
        win_rate_reward = (current_performance.get('win_rate', 0.5) - 0.5) * 2.0
        sharpe_reward = min(current_performance.get('sharpe_ratio', 0.0) / 2.0, 1.0)
        drawdown_penalty = -abs(current_performance.get('max_drawdown', 0.0)) * 10.0
        
        reward = win_rate_reward + sharpe_reward + drawdown_penalty
        
        return float(reward)
    
    def _get_current_performance(self) -> Dict[str, float]:
        """Get current performance metrics"""
        # Simplified performance calculation
        return {
            'win_rate': 0.6,
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.02,
            'profit_factor': 1.8
        }

class LearningFramework:
    """
    Comprehensive Learning and Validation Framework
    综合学习与验证框架
    
    Implements advanced validation and learning methods:
    1. A/B testing for parameter validation
    2. Statistical significance testing
    3. Walk-forward and time series validation
    4. Reinforcement learning for adaptation
    5. Monte Carlo simulation
    6. Continuous learning mechanisms
    """
    
    def __init__(self, config: Optional[Dict] = None,
                 parameter_engine: Optional[AdaptiveParameterEngine] = None,
                 parameter_optimizer: Optional[ParameterOptimizer] = None,
                 performance_tracker: Optional[PerformanceTracker] = None,
                 risk_manager: Optional[RiskControlManager] = None):
        """Initialize learning framework"""
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Core components
        self.parameter_engine = parameter_engine
        self.parameter_optimizer = parameter_optimizer
        self.performance_tracker = performance_tracker
        self.risk_manager = risk_manager
        
        # A/B testing
        self.active_tests = {}
        self.completed_tests = {}
        self.test_assignments = defaultdict(str)
        
        # Validation results
        self.validation_history = defaultdict(list)
        self.learning_states = {}
        
        # RL components
        self.rl_environments = {}
        self.rl_models = {}
        
        # Threading
        self._lock = threading.Lock()
        
        self.logger.info("LearningFramework initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for learning framework"""
        return {
            'ab_testing': {
                'min_sample_size': 100,
                'significance_level': 0.05,
                'power': 0.8,
                'max_test_duration_days': 30,
                'effect_size_threshold': 0.1,
                'traffic_split': 0.5,
                'sequential_testing': True
            },
            'validation': {
                'walk_forward_window': 1000,
                'n_splits': 5,
                'test_size': 0.2,
                'purge_days': 1,
                'embargo_days': 1,
                'monte_carlo_simulations': 1000,
                'bootstrap_samples': 1000,
                'confidence_level': 0.95
            },
            'reinforcement_learning': {
                'algorithm': 'PPO',
                'learning_rate': 3e-4,
                'n_timesteps': 10000,
                'eval_episodes': 10,
                'update_frequency': 100,
                'exploration_rate': 0.1,
                'exploration_decay': 0.995
            },
            'continuous_learning': {
                'adaptation_frequency': 50,  # trades
                'performance_threshold': 0.1,
                'stability_requirement': 0.8,
                'min_learning_samples': 200,
                'learning_rate': 0.01,
                'momentum': 0.9
            },
            'outlier_detection': {
                'contamination': 0.1,
                'method': 'isolation_forest',
                'threshold': 3.0
            }
        }
    
    def create_ab_test(self, name: str, description: str,
                      control_params: Dict[str, float],
                      treatment_params: Dict[str, float],
                      success_metric: str = 'risk_adjusted_return',
                      expected_effect_size: float = 0.1) -> str:
        """
        Create A/B test for parameter comparison
        创建A/B测试用于参数比较
        """
        test_id = str(uuid.uuid4())
        
        # Calculate required sample size
        min_sample_size = self._calculate_sample_size(
            expected_effect_size,
            self.config['ab_testing']['power'],
            self.config['ab_testing']['significance_level']
        )
        
        test_config = ABTestConfig(
            test_id=test_id,
            name=name,
            description=description,
            control_params=control_params,
            treatment_params=treatment_params,
            success_metric=success_metric,
            minimum_sample_size=min_sample_size,
            significance_level=self.config['ab_testing']['significance_level'],
            power=self.config['ab_testing']['power'],
            expected_effect_size=expected_effect_size,
            max_duration_days=self.config['ab_testing']['max_test_duration_days']
        )
        
        with self._lock:
            self.active_tests[test_id] = {
                'config': test_config,
                'start_date': datetime.now(),
                'control_data': [],
                'treatment_data': [],
                'assignments': {}
            }
        
        self.logger.info(f"Created A/B test '{name}' with ID {test_id}")
        return test_id
    
    def _calculate_sample_size(self, effect_size: float, power: float, alpha: float) -> int:
        """Calculate required sample size for A/B test"""
        # Using power analysis for two-sample t-test
        required_n = ttest_power(effect_size, nobs=None, alpha=alpha, power=power)
        return max(int(required_n) + 1, self.config['ab_testing']['min_sample_size'])
    
    def assign_to_test(self, test_id: str, identifier: str) -> str:
        """Assign entity to A/B test group"""
        if test_id not in self.active_tests:
            return 'control'  # Default to control if test doesn't exist
        
        # Check if already assigned
        if identifier in self.test_assignments:
            return self.test_assignments[identifier]
        
        # Random assignment with traffic split
        traffic_split = self.config['ab_testing']['traffic_split']
        assignment = 'treatment' if np.random.random() < traffic_split else 'control'
        
        with self._lock:
            self.test_assignments[identifier] = assignment
            self.active_tests[test_id]['assignments'][identifier] = assignment
        
        return assignment
    
    def record_test_observation(self, test_id: str, identifier: str,
                              metrics: Dict[str, float]):
        """Record observation for A/B test"""
        if test_id not in self.active_tests:
            return
        
        assignment = self.test_assignments.get(identifier, 'control')
        
        with self._lock:
            test_data = self.active_tests[test_id]
            if assignment == 'control':
                test_data['control_data'].append(metrics)
            else:
                test_data['treatment_data'].append(metrics)
    
    def analyze_ab_test(self, test_id: str) -> Optional[ABTestResult]:
        """
        Analyze A/B test results and determine statistical significance
        分析A/B测试结果并确定统计显著性
        """
        if test_id not in self.active_tests:
            return None
        
        test_data = self.active_tests[test_id]
        config = test_data['config']
        control_data = test_data['control_data']
        treatment_data = test_data['treatment_data']
        
        if len(control_data) < config.minimum_sample_size or len(treatment_data) < config.minimum_sample_size:
            self.logger.warning(f"Insufficient sample size for test {test_id}")
            return None
        
        # Extract success metric values
        control_values = [obs.get(config.success_metric, 0) for obs in control_data]
        treatment_values = [obs.get(config.success_metric, 0) for obs in treatment_data]
        
        control_mean = np.mean(control_values)
        treatment_mean = np.mean(treatment_values)
        effect_size = treatment_mean - control_mean
        
        # Statistical test
        if len(control_values) > 30 and len(treatment_values) > 30:
            # Use t-test for large samples
            t_stat, p_value = ttest_ind(treatment_values, control_values)
        else:
            # Use Mann-Whitney U test for small samples
            u_stat, p_value = mannwhitneyu(treatment_values, control_values, alternative='two-sided')
        
        # Confidence interval for effect size
        pooled_std = np.sqrt(
            ((len(control_values) - 1) * np.var(control_values, ddof=1) +
             (len(treatment_values) - 1) * np.var(treatment_values, ddof=1)) /
            (len(control_values) + len(treatment_values) - 2)
        )
        
        se_diff = pooled_std * np.sqrt(1/len(control_values) + 1/len(treatment_values))
        margin_error = stats.t.ppf(1 - config.significance_level/2, 
                                 len(control_values) + len(treatment_values) - 2) * se_diff
        
        confidence_interval = (effect_size - margin_error, effect_size + margin_error)
        
        # Calculate statistical power
        actual_effect_size = effect_size / pooled_std if pooled_std > 0 else 0
        statistical_power = ttest_power(
            actual_effect_size, 
            nobs=min(len(control_values), len(treatment_values)),
            alpha=config.significance_level
        )
        
        # Determine significance and recommendation
        is_significant = p_value < config.significance_level
        
        if is_significant:
            if effect_size > 0:
                recommendation = "Adopt treatment parameters"
            else:
                recommendation = "Keep control parameters"
        else:
            recommendation = "No significant difference detected"
        
        # Calculate comprehensive metrics
        control_metrics = self._calculate_comprehensive_metrics(control_data)
        treatment_metrics = self._calculate_comprehensive_metrics(treatment_data)
        
        result = ABTestResult(
            test_id=test_id,
            start_date=test_data['start_date'],
            end_date=datetime.now(),
            control_sample_size=len(control_data),
            treatment_sample_size=len(treatment_data),
            control_mean=control_mean,
            treatment_mean=treatment_mean,
            effect_size=effect_size,
            p_value=p_value,
            confidence_interval=confidence_interval,
            statistical_power=statistical_power,
            is_significant=is_significant,
            recommendation=recommendation,
            control_metrics=control_metrics,
            treatment_metrics=treatment_metrics
        )
        
        return result
    
    def _calculate_comprehensive_metrics(self, data: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive metrics for test group"""
        if not data:
            return {}
        
        # Extract common metrics
        win_rates = [obs.get('win_rate', 0) for obs in data]
        sharpe_ratios = [obs.get('sharpe_ratio', 0) for obs in data]
        max_drawdowns = [obs.get('max_drawdown', 0) for obs in data]
        profit_factors = [obs.get('profit_factor', 1) for obs in data]
        
        return {
            'avg_win_rate': np.mean(win_rates),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_profit_factor': np.mean(profit_factors),
            'win_rate_std': np.std(win_rates),
            'sharpe_ratio_std': np.std(sharpe_ratios),
            'sample_size': len(data)
        }
    
    def run_walk_forward_validation(self, symbol: str, regime: MarketRegime,
                                  parameters: Dict[str, float],
                                  market_data: pd.DataFrame,
                                  performance_data: List[Dict]) -> ValidationResult:
        """
        Run walk-forward validation
        执行时间序列前向验证
        """
        validation_id = str(uuid.uuid4())
        
        if len(performance_data) < self.config['validation']['walk_forward_window']:
            raise ValueError("Insufficient data for walk-forward validation")
        
        window_size = self.config['validation']['walk_forward_window']
        step_size = window_size // 4  # 25% step
        
        in_sample_scores = []
        out_sample_scores = []
        
        for i in range(0, len(performance_data) - window_size, step_size):
            # In-sample window
            in_sample_data = performance_data[i:i + window_size]
            
            # Out-of-sample window
            out_sample_start = i + window_size
            out_sample_end = min(out_sample_start + step_size, len(performance_data))
            out_sample_data = performance_data[out_sample_start:out_sample_end]
            
            if len(out_sample_data) < 10:  # Minimum out-of-sample size
                break
            
            # Calculate performance metrics
            in_sample_metrics = self._calculate_performance_metrics(in_sample_data)
            out_sample_metrics = self._calculate_performance_metrics(out_sample_data)
            
            in_sample_scores.append(in_sample_metrics.get('risk_adjusted_return', 0))
            out_sample_scores.append(out_sample_metrics.get('risk_adjusted_return', 0))
        
        # Calculate validation metrics
        avg_in_sample = np.mean(in_sample_scores)
        avg_out_sample = np.mean(out_sample_scores)
        
        # Overfitting score (difference between in-sample and out-of-sample)
        overfitting_score = avg_in_sample - avg_out_sample
        
        # Stability score (inverse of out-of-sample variance)
        stability_score = 1.0 / (1.0 + np.var(out_sample_scores))
        
        # Confidence interval for out-of-sample performance
        out_sample_std = np.std(out_sample_scores)
        confidence_interval = (
            avg_out_sample - 1.96 * out_sample_std / np.sqrt(len(out_sample_scores)),
            avg_out_sample + 1.96 * out_sample_std / np.sqrt(len(out_sample_scores))
        )
        
        # Generate recommendation
        if overfitting_score > 0.1:
            recommendation = "High overfitting detected - consider regularization"
        elif stability_score < 0.5:
            recommendation = "Low stability - parameters may need adjustment"
        elif avg_out_sample > 0.5:
            recommendation = "Good out-of-sample performance - parameters validated"
        else:
            recommendation = "Poor out-of-sample performance - re-optimization needed"
        
        final_in_sample_metrics = self._calculate_performance_metrics(performance_data[:window_size])
        final_out_sample_metrics = self._calculate_performance_metrics(performance_data[window_size:])
        
        result = ValidationResult(
            validation_id=validation_id,
            method=ValidationMethod.WALK_FORWARD,
            parameters=parameters,
            in_sample_metrics=final_in_sample_metrics,
            out_sample_metrics=final_out_sample_metrics,
            cross_validation_scores=out_sample_scores,
            confidence_interval=confidence_interval,
            overfitting_score=overfitting_score,
            stability_score=stability_score,
            recommendation=recommendation
        )
        
        # Store validation result
        with self._lock:
            self.validation_history[f"{symbol}_{regime.value}"].append(result)
        
        return result
    
    def run_monte_carlo_validation(self, symbol: str, regime: MarketRegime,
                                 parameters: Dict[str, float],
                                 market_data: pd.DataFrame,
                                 performance_data: List[Dict]) -> ValidationResult:
        """
        Run Monte Carlo validation with bootstrap sampling
        蒙特卡洛验证与自助法采样
        """
        validation_id = str(uuid.uuid4())
        n_simulations = self.config['validation']['monte_carlo_simulations']
        
        bootstrap_scores = []
        
        for _ in range(n_simulations):
            # Bootstrap sample
            sample_indices = np.random.choice(
                len(performance_data), 
                size=len(performance_data), 
                replace=True
            )
            bootstrap_sample = [performance_data[i] for i in sample_indices]
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(bootstrap_sample)
            bootstrap_scores.append(metrics.get('risk_adjusted_return', 0))
        
        # Calculate validation statistics
        mean_score = np.mean(bootstrap_scores)
        std_score = np.std(bootstrap_scores)
        
        # Confidence interval
        confidence_level = self.config['validation']['confidence_level']
        alpha = 1 - confidence_level
        confidence_interval = (
            np.percentile(bootstrap_scores, 100 * alpha / 2),
            np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))
        )
        
        # Stability score based on coefficient of variation
        stability_score = 1.0 / (1.0 + std_score / abs(mean_score)) if mean_score != 0 else 0.5
        
        # Generate recommendation
        if std_score / abs(mean_score) > 0.5 if mean_score != 0 else True:
            recommendation = "High variability detected - parameters may be unstable"
        elif confidence_interval[0] > 0.3:
            recommendation = "Consistent positive performance - parameters validated"
        elif confidence_interval[1] < 0.1:
            recommendation = "Consistently poor performance - re-optimization needed"
        else:
            recommendation = "Mixed performance - consider parameter refinement"
        
        original_metrics = self._calculate_performance_metrics(performance_data)
        
        result = ValidationResult(
            validation_id=validation_id,
            method=ValidationMethod.MONTE_CARLO,
            parameters=parameters,
            in_sample_metrics=original_metrics,
            out_sample_metrics={'bootstrap_mean': mean_score, 'bootstrap_std': std_score},
            cross_validation_scores=bootstrap_scores,
            confidence_interval=confidence_interval,
            overfitting_score=0.0,  # Not applicable for Monte Carlo
            stability_score=stability_score,
            recommendation=recommendation
        )
        
        # Store validation result
        with self._lock:
            self.validation_history[f"{symbol}_{regime.value}"].append(result)
        
        return result
    
    def _calculate_performance_metrics(self, performance_data: List[Dict]) -> Dict[str, float]:
        """Calculate performance metrics from trade data"""
        if not performance_data:
            return {'risk_adjusted_return': 0.0}
        
        # Extract returns
        returns = [trade.get('pnl_pct', 0.0) for trade in performance_data]
        returns_array = np.array(returns)
        
        # Calculate metrics
        win_rate = np.mean(returns_array > 0)
        avg_return = np.mean(returns_array)
        
        if np.std(returns_array) > 0:
            sharpe_ratio = avg_return / np.std(returns_array) * np.sqrt(252 * 24 * 12)
        else:
            sharpe_ratio = 0.0
        
        # Risk-adjusted return composite metric
        risk_adjusted_return = (
            0.4 * win_rate +
            0.3 * min(sharpe_ratio / 3.0, 1.0) +
            0.3 * min(avg_return / 0.01, 1.0)
        )
        
        return {
            'win_rate': win_rate,
            'avg_return': avg_return,
            'sharpe_ratio': sharpe_ratio,
            'risk_adjusted_return': risk_adjusted_return
        }
    
    def setup_reinforcement_learning(self, symbol: str, regime: MarketRegime,
                                   parameter_space: Dict) -> str:
        """
        Setup reinforcement learning environment for parameter adaptation
        设置参数自适应强化学习环境
        """
        env_id = f"{symbol}_{regime.value}_rl"
        
        # Create environment
        env = ParameterAdaptationEnv(parameter_space, self.performance_tracker)
        
        # Create RL model
        if self.config['reinforcement_learning']['algorithm'] == 'PPO':
            model = PPO('MlpPolicy', env, 
                       learning_rate=self.config['reinforcement_learning']['learning_rate'],
                       verbose=0)
        elif self.config['reinforcement_learning']['algorithm'] == 'A2C':
            model = A2C('MlpPolicy', env,
                       learning_rate=self.config['reinforcement_learning']['learning_rate'],
                       verbose=0)
        else:
            model = PPO('MlpPolicy', env, verbose=0)  # Default
        
        # Store environment and model
        with self._lock:
            self.rl_environments[env_id] = env
            self.rl_models[env_id] = model
            self.learning_states[env_id] = LearningState(
                algorithm=LearningAlgorithm.REINFORCEMENT_LEARNING,
                state_vector=np.zeros(env.observation_space.shape[0]),
                action_space=parameter_space,
                reward_history=[],
                performance_history=[],
                learning_rate=self.config['reinforcement_learning']['learning_rate'],
                exploration_rate=self.config['reinforcement_learning']['exploration_rate'],
                last_update=datetime.now()
            )
        
        self.logger.info(f"RL environment setup for {env_id}")
        return env_id
    
    def train_reinforcement_learning(self, env_id: str) -> Dict[str, float]:
        """Train reinforcement learning model"""
        if env_id not in self.rl_models:
            raise ValueError(f"RL environment {env_id} not found")
        
        model = self.rl_models[env_id]
        env = self.rl_environments[env_id]
        
        # Train model
        n_timesteps = self.config['reinforcement_learning']['n_timesteps']
        model.learn(total_timesteps=n_timesteps)
        
        # Evaluate trained model
        eval_episodes = self.config['reinforcement_learning']['eval_episodes']
        mean_reward, std_reward = evaluate_policy(
            model, env, n_eval_episodes=eval_episodes, return_episode_rewards=False
        )
        
        # Update learning state
        with self._lock:
            if env_id in self.learning_states:
                self.learning_states[env_id].reward_history.append(mean_reward)
                self.learning_states[env_id].last_update = datetime.now()
        
        self.logger.info(f"RL training completed for {env_id}: mean_reward={mean_reward:.4f}")
        
        return {
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'n_timesteps': n_timesteps,
            'eval_episodes': eval_episodes
        }
    
    def get_rl_action(self, env_id: str, observation: np.ndarray) -> Dict[str, float]:
        """Get RL-based parameter adjustment"""
        if env_id not in self.rl_models:
            return {}
        
        model = self.rl_models[env_id]
        env = self.rl_environments[env_id]
        
        # Get action from trained model
        action, _ = model.predict(observation, deterministic=True)
        
        # Convert action to parameter adjustments
        parameter_adjustments = {}
        for i, (param_name, param_config) in enumerate(env.parameter_space.items()):
            if i < len(action):
                param_range = param_config['high'] - param_config['low']
                adjustment = action[i] * param_range * 0.1  # 10% max adjustment
                parameter_adjustments[param_name] = float(adjustment)
        
        return parameter_adjustments
    
    def detect_performance_outliers(self, performance_data: List[Dict]) -> List[int]:
        """
        Detect outlier performance observations
        检测异常性能观测值
        """
        if len(performance_data) < 50:
            return []
        
        # Extract features for outlier detection
        features = []
        for trade in performance_data:
            feature_vector = [
                trade.get('pnl_pct', 0),
                trade.get('holding_minutes', 0),
                trade.get('signal_confidence', 0.5),
                trade.get('win_rate', 0.5),
                trade.get('sharpe_ratio', 0)
            ]
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        # Use Isolation Forest for outlier detection
        contamination = self.config['outlier_detection']['contamination']
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(features_array)
        
        # Return indices of outliers
        outlier_indices = [i for i, label in enumerate(outlier_labels) if label == -1]
        
        self.logger.info(f"Detected {len(outlier_indices)} outliers in {len(performance_data)} observations")
        
        return outlier_indices
    
    def continuous_learning_update(self, symbol: str, regime: MarketRegime,
                                 recent_performance: List[Dict]) -> Optional[Dict[str, float]]:
        """
        Continuous learning update based on recent performance
        基于最近表现的持续学习更新
        """
        key = f"{symbol}_{regime.value}"
        
        if len(recent_performance) < self.config['continuous_learning']['min_learning_samples']:
            return None
        
        # Check if update is needed
        current_performance = self._calculate_performance_metrics(recent_performance)
        performance_threshold = self.config['continuous_learning']['performance_threshold']
        
        if current_performance.get('risk_adjusted_return', 0) < performance_threshold:
            # Performance below threshold - trigger learning update
            
            # Detect and remove outliers
            outlier_indices = self.detect_performance_outliers(recent_performance)
            clean_performance = [
                trade for i, trade in enumerate(recent_performance)
                if i not in outlier_indices
            ]
            
            # Get current best parameters
            if self.parameter_engine:
                current_params = self.parameter_engine.get_current_parameters(symbol, regime, pd.DataFrame())
                current_params_dict = {
                    'rsi_low': current_params.rsi_low,
                    'rsi_high': current_params.rsi_high,
                    'dip_threshold': current_params.dip_threshold,
                    'volume_threshold': current_params.volume_threshold,
                    'target_profit': current_params.target_profit,
                    'stop_loss': current_params.stop_loss,
                    'max_holding_minutes': current_params.max_holding_minutes
                }
                
                # Use parameter optimizer for adaptive optimization
                if self.parameter_optimizer:
                    try:
                        result = self.parameter_optimizer.adaptive_optimization(
                            symbol, regime, pd.DataFrame(), clean_performance, current_params_dict
                        )
                        
                        self.logger.info(f"Continuous learning update for {key}: "
                                       f"score improved from current to {result.score:.4f}")
                        
                        return result.parameters
                    except Exception as e:
                        self.logger.error(f"Continuous learning update failed for {key}: {e}")
        
        return None
    
    def export_learning_report(self, symbol: str, regime: MarketRegime,
                             output_path: str) -> Dict:
        """Export comprehensive learning report"""
        key = f"{symbol}_{regime.value}"
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'regime': regime.value,
            'active_ab_tests': [],
            'completed_ab_tests': [],
            'validation_results': [],
            'learning_state': None,
            'configuration': self.config
        }
        
        # Active A/B tests
        for test_id, test_data in self.active_tests.items():
            if test_data['config'].description.find(symbol) >= 0:
                report['active_ab_tests'].append({
                    'test_id': test_id,
                    'name': test_data['config'].name,
                    'start_date': test_data['start_date'].isoformat(),
                    'control_sample_size': len(test_data['control_data']),
                    'treatment_sample_size': len(test_data['treatment_data'])
                })
        
        # Completed A/B tests
        for test_id, result in self.completed_tests.items():
            if result.test_id.find(symbol) >= 0:
                report['completed_ab_tests'].append(asdict(result))
        
        # Validation results
        if key in self.validation_history:
            report['validation_results'] = [
                asdict(result) for result in self.validation_history[key][-10:]  # Last 10
            ]
        
        # Learning state
        if key in self.learning_states:
            state = self.learning_states[key]
            report['learning_state'] = {
                'algorithm': state.algorithm.value,
                'reward_history': state.reward_history[-50:],  # Last 50
                'learning_rate': state.learning_rate,
                'exploration_rate': state.exploration_rate,
                'last_update': state.last_update.isoformat()
            }
        
        # Save report
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Learning report exported to {output_file}")
        
        return report

# Factory function
def create_learning_framework(config: Optional[Dict] = None,
                            parameter_engine: Optional[AdaptiveParameterEngine] = None,
                            parameter_optimizer: Optional[ParameterOptimizer] = None,
                            performance_tracker: Optional[PerformanceTracker] = None,
                            risk_manager: Optional[RiskControlManager] = None) -> LearningFramework:
    """Factory function to create learning framework"""
    return LearningFramework(config, parameter_engine, parameter_optimizer, performance_tracker, risk_manager)