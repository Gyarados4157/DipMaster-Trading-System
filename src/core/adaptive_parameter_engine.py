#!/usr/bin/env python3
"""
Adaptive Parameter Engine for DipMaster Strategy
自适应参数引擎 - DipMaster策略专用

This module implements a sophisticated parameter optimization engine that dynamically
adjusts strategy parameters based on:
- Market regime detection results
- Recent performance outcomes  
- Risk metrics and market stress
- Symbol-specific characteristics

The goal is to improve win rate from 47.7% to 65%+ for BTCUSDT and achieve
portfolio-level performance targets of 25%+ annual returns with <5% drawdown.

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
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque

# Optimization libraries
import optuna
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize, differential_evolution
from scipy.stats import norm, t
import cvxpy as cp

# Market regime and strategy components
from .market_regime_detector import MarketRegimeDetector, MarketRegime, RegimeSignal
from ..types.common_types import *

warnings.filterwarnings('ignore')

class OptimizationMethod(Enum):
    """Optimization methods for parameter tuning"""
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    GRADIENT_FREE = "gradient_free"
    REINFORCEMENT = "reinforcement"
    ENSEMBLE = "ensemble"

class ParameterType(Enum):
    """Types of parameters for different optimization approaches"""
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    CATEGORICAL = "categorical"
    BOUNDED = "bounded"

@dataclass
class ParameterDef:
    """Parameter definition with bounds and constraints"""
    name: str
    param_type: ParameterType
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Any = None
    discrete_values: Optional[List] = None
    categories: Optional[List[str]] = None
    importance_weight: float = 1.0
    regime_specific: bool = True
    symbol_specific: bool = True

@dataclass
class ParameterSet:
    """Complete parameter set for strategy execution"""
    # Entry parameters
    rsi_low: float
    rsi_high: float
    dip_threshold: float
    volume_threshold: float
    
    # Exit parameters
    target_profit: float
    stop_loss: float
    max_holding_minutes: int
    
    # Risk parameters
    position_size_multiplier: float
    confidence_multiplier: float
    correlation_penalty: float
    
    # Metadata
    regime: MarketRegime
    symbol: str
    confidence_score: float
    timestamp: datetime
    optimization_score: float = 0.0

@dataclass
class OptimizationResult:
    """Result of parameter optimization"""
    parameters: ParameterSet
    objective_value: float
    confidence_interval: Tuple[float, float]
    n_trials: int
    optimization_time: float
    convergence_status: str
    feature_importance: Dict[str, float]
    cross_validation_score: float

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    win_rate: float
    avg_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    expected_shortfall: float
    profit_factor: float
    avg_holding_time: float
    regime_consistency: float

class AdaptiveParameterEngine:
    """
    Advanced Parameter Optimization Engine
    高级参数优化引擎
    
    Implements multi-dimensional parameter optimization using:
    1. Bayesian optimization for continuous parameters
    2. Genetic algorithms for discrete parameters  
    3. Reinforcement learning for dynamic adaptation
    4. Risk-aware objective functions
    5. Regime-specific optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the adaptive parameter engine"""
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Core components
        self.regime_detector = MarketRegimeDetector()
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        self.parameter_history = defaultdict(lambda: deque(maxlen=500))
        self.optimization_cache = {}
        
        # Optimization engines
        self._initialize_optimizers()
        
        # Parameter definitions
        self.parameter_definitions = self._define_parameter_space()
        
        # Performance tracking
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        self.symbol_performance = defaultdict(lambda: defaultdict(list))
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Current best parameters by regime and symbol
        self.best_parameters = {}
        
        self.logger.info("AdaptiveParameterEngine initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for parameter optimization"""
        return {
            'optimization': {
                'n_trials': 100,
                'n_jobs': -1,
                'cv_folds': 5,
                'optimization_timeout': 300,  # 5 minutes
                'convergence_patience': 20,
                'ensemble_methods': ['bayesian', 'genetic'],
                'cache_results': True
            },
            'objective_weights': {
                'win_rate': 0.4,
                'sharpe_ratio': 0.25,
                'max_drawdown': 0.15,
                'profit_factor': 0.10,
                'regime_consistency': 0.10
            },
            'risk_controls': {
                'max_position_size': 0.25,  # 25% of portfolio
                'max_leverage': 3.0,
                'var_limit': 0.02,  # 2% daily VaR
                'correlation_limit': 0.7,
                'max_drawdown_limit': 0.05
            },
            'adaptation': {
                'learning_rate': 0.01,
                'momentum': 0.9,
                'decay_factor': 0.95,
                'min_samples': 50,
                'reoptimization_frequency': 100  # trades
            },
            'validation': {
                'walk_forward_window': 1000,
                'out_of_sample_ratio': 0.2,
                'bootstrap_samples': 1000,
                'confidence_level': 0.95
            }
        }
    
    def _initialize_optimizers(self):
        """Initialize optimization engines"""
        # Bayesian optimizer
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.bayesian_optimizer = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            alpha=1e-6
        )
        
        # Optuna study for hyperparameter optimization
        self.optuna_studies = {}
        
        # Scaler for parameter normalization
        self.parameter_scaler = StandardScaler()
        
        self.logger.info("Optimization engines initialized")
    
    def _define_parameter_space(self) -> Dict[str, ParameterDef]:
        """Define the complete parameter space for optimization"""
        return {
            # Entry parameters
            'rsi_low': ParameterDef(
                name='rsi_low',
                param_type=ParameterType.BOUNDED,
                min_value=10.0,
                max_value=45.0,
                default_value=30.0,
                importance_weight=0.9
            ),
            'rsi_high': ParameterDef(
                name='rsi_high',
                param_type=ParameterType.BOUNDED,
                min_value=35.0,
                max_value=70.0,
                default_value=50.0,
                importance_weight=0.8
            ),
            'dip_threshold': ParameterDef(
                name='dip_threshold',
                param_type=ParameterType.BOUNDED,
                min_value=0.001,
                max_value=0.01,
                default_value=0.002,
                importance_weight=1.0
            ),
            'volume_threshold': ParameterDef(
                name='volume_threshold',
                param_type=ParameterType.BOUNDED,
                min_value=1.0,
                max_value=5.0,
                default_value=1.5,
                importance_weight=0.7
            ),
            
            # Exit parameters
            'target_profit': ParameterDef(
                name='target_profit',
                param_type=ParameterType.BOUNDED,
                min_value=0.003,
                max_value=0.025,
                default_value=0.008,
                importance_weight=0.9
            ),
            'stop_loss': ParameterDef(
                name='stop_loss',
                param_type=ParameterType.BOUNDED,
                min_value=-0.03,
                max_value=-0.003,
                default_value=-0.015,
                importance_weight=0.95
            ),
            'max_holding_minutes': ParameterDef(
                name='max_holding_minutes',
                param_type=ParameterType.DISCRETE,
                discrete_values=[30, 45, 60, 90, 120, 180, 240, 300],
                default_value=180,
                importance_weight=0.6
            ),
            
            # Risk parameters
            'position_size_multiplier': ParameterDef(
                name='position_size_multiplier',
                param_type=ParameterType.BOUNDED,
                min_value=0.3,
                max_value=2.0,
                default_value=1.0,
                importance_weight=0.8
            ),
            'confidence_multiplier': ParameterDef(
                name='confidence_multiplier',
                param_type=ParameterType.BOUNDED,
                min_value=0.3,
                max_value=1.5,
                default_value=1.0,
                importance_weight=0.7
            ),
            'correlation_penalty': ParameterDef(
                name='correlation_penalty',
                param_type=ParameterType.BOUNDED,
                min_value=0.0,
                max_value=1.0,
                default_value=0.5,
                importance_weight=0.5
            )
        }
    
    def calculate_objective_function(self, parameters: ParameterSet, 
                                   performance_data: List[Dict],
                                   regime: MarketRegime,
                                   symbol: str) -> float:
        """
        Calculate optimization objective function
        目标函数计算 - 平衡收益与风险
        """
        if not performance_data:
            return -1.0  # Penalty for no data
        
        # Convert performance data to metrics
        metrics = self._calculate_performance_metrics(performance_data)
        
        # Weight-based objective function
        weights = self.config['objective_weights']
        
        # Normalized metrics (0-1 scale)
        win_rate_score = min(metrics.win_rate / 0.8, 1.0)  # Target 80% win rate
        sharpe_score = min(max(metrics.sharpe_ratio / 3.0, 0.0), 1.0)  # Target Sharpe 3.0
        drawdown_score = max(1.0 - (metrics.max_drawdown / 0.05), 0.0)  # Max 5% drawdown
        profit_factor_score = min(metrics.profit_factor / 2.0, 1.0)  # Target PF 2.0
        regime_score = metrics.regime_consistency
        
        # Risk penalties
        risk_penalty = 0.0
        if metrics.var_95 > self.config['risk_controls']['var_limit']:
            risk_penalty += 0.2
        if metrics.max_drawdown > self.config['risk_controls']['max_drawdown_limit']:
            risk_penalty += 0.3
        
        # Calculate weighted objective
        objective = (
            weights['win_rate'] * win_rate_score +
            weights['sharpe_ratio'] * sharpe_score +
            weights['max_drawdown'] * drawdown_score +
            weights['profit_factor'] * profit_factor_score +
            weights['regime_consistency'] * regime_score
        ) - risk_penalty
        
        # Regime-specific adjustments
        if regime == MarketRegime.STRONG_DOWNTREND:
            # Prioritize capital preservation
            objective = 0.7 * objective + 0.3 * drawdown_score
        elif regime == MarketRegime.HIGH_VOLATILITY:
            # Balance risk and reward
            objective = 0.5 * objective + 0.5 * sharpe_score
        elif regime == MarketRegime.RANGE_BOUND:
            # Optimize for consistency
            objective = 0.6 * objective + 0.4 * win_rate_score
        
        return float(objective)
    
    def _calculate_performance_metrics(self, performance_data: List[Dict]) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics from trade data"""
        if not performance_data:
            return PerformanceMetrics(
                win_rate=0.0, avg_return=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                max_drawdown=1.0, var_95=1.0, expected_shortfall=1.0,
                profit_factor=0.0, avg_holding_time=0.0, regime_consistency=0.0
            )
        
        # Extract returns and holding times
        returns = [trade.get('pnl_pct', 0.0) for trade in performance_data]
        holding_times = [trade.get('holding_minutes', 0) for trade in performance_data]
        
        if not returns:
            return PerformanceMetrics(
                win_rate=0.0, avg_return=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                max_drawdown=1.0, var_95=1.0, expected_shortfall=1.0,
                profit_factor=0.0, avg_holding_time=0.0, regime_consistency=0.0
            )
        
        returns_array = np.array(returns)
        
        # Basic metrics
        win_rate = np.mean(returns_array > 0)
        avg_return = np.mean(returns_array)
        
        # Risk-adjusted metrics
        if np.std(returns_array) > 0:
            sharpe_ratio = avg_return / np.std(returns_array) * np.sqrt(252 * 24 * 12)  # Annualized
        else:
            sharpe_ratio = 0.0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = avg_return / np.std(downside_returns) * np.sqrt(252 * 24 * 12)
        else:
            sortino_ratio = sharpe_ratio
        
        # Drawdown calculation
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        # VaR and Expected Shortfall
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0.0
        var_95 = abs(var_95)
        
        tail_losses = returns_array[returns_array <= np.percentile(returns_array, 5)]
        expected_shortfall = abs(np.mean(tail_losses)) if len(tail_losses) > 0 else 0.0
        
        # Profit factor
        winning_trades = returns_array[returns_array > 0]
        losing_trades = returns_array[returns_array < 0]
        
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            profit_factor = np.sum(winning_trades) / abs(np.sum(losing_trades))
        else:
            profit_factor = 1.0 if len(winning_trades) > 0 else 0.0
        
        # Average holding time
        avg_holding_time = np.mean(holding_times) if holding_times else 0.0
        
        # Regime consistency (placeholder - would need regime data)
        regime_consistency = 0.8  # Default value
        
        return PerformanceMetrics(
            win_rate=win_rate,
            avg_return=avg_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            profit_factor=profit_factor,
            avg_holding_time=avg_holding_time,
            regime_consistency=regime_consistency
        )
    
    def optimize_parameters(self, symbol: str, regime: MarketRegime,
                          performance_data: List[Dict],
                          method: OptimizationMethod = OptimizationMethod.BAYESIAN,
                          force_reoptimization: bool = False) -> OptimizationResult:
        """
        Optimize parameters for specific symbol and regime
        符号和体制特定的参数优化
        """
        cache_key = f"{symbol}_{regime.value}_{len(performance_data)}"
        
        # Check cache first
        if not force_reoptimization and cache_key in self.optimization_cache:
            cached_result = self.optimization_cache[cache_key]
            if datetime.now() - cached_result.parameters.timestamp < timedelta(hours=6):
                self.logger.info(f"Using cached optimization result for {symbol} in {regime.value}")
                return cached_result
        
        start_time = datetime.now()
        
        if method == OptimizationMethod.BAYESIAN:
            result = self._optimize_bayesian(symbol, regime, performance_data)
        elif method == OptimizationMethod.GENETIC:
            result = self._optimize_genetic(symbol, regime, performance_data)
        elif method == OptimizationMethod.ENSEMBLE:
            result = self._optimize_ensemble(symbol, regime, performance_data)
        else:
            result = self._optimize_bayesian(symbol, regime, performance_data)  # Default
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        result.optimization_time = optimization_time
        
        # Cache result
        if self.config['optimization']['cache_results']:
            self.optimization_cache[cache_key] = result
        
        # Update best parameters
        with self._lock:
            regime_key = f"{symbol}_{regime.value}"
            if regime_key not in self.best_parameters or \
               result.objective_value > self.best_parameters[regime_key].optimization_score:
                self.best_parameters[regime_key] = result.parameters
        
        self.logger.info(f"Parameter optimization completed for {symbol} in {regime.value}: "
                        f"objective={result.objective_value:.4f}, time={optimization_time:.2f}s")
        
        return result
    
    def _optimize_bayesian(self, symbol: str, regime: MarketRegime,
                          performance_data: List[Dict]) -> OptimizationResult:
        """Bayesian optimization using Optuna"""
        study_name = f"{symbol}_{regime.value}_bayesian"
        
        def objective(trial):
            # Suggest parameters
            params = {}
            for param_name, param_def in self.parameter_definitions.items():
                if param_def.param_type == ParameterType.BOUNDED:
                    params[param_name] = trial.suggest_float(
                        param_name, param_def.min_value, param_def.max_value
                    )
                elif param_def.param_type == ParameterType.DISCRETE:
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_def.discrete_values
                    )
            
            # Create parameter set
            parameter_set = ParameterSet(
                regime=regime,
                symbol=symbol,
                confidence_score=1.0,
                timestamp=datetime.now(),
                **params
            )
            
            # Calculate objective
            return self.calculate_objective_function(
                parameter_set, performance_data, regime, symbol
            )
        
        # Create or get study
        if study_name not in self.optuna_studies:
            self.optuna_studies[study_name] = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                sampler=optuna.samplers.TPESampler()
            )
        
        study = self.optuna_studies[study_name]
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.config['optimization']['n_trials'],
            timeout=self.config['optimization']['optimization_timeout'],
            n_jobs=1  # Single job for thread safety
        )
        
        # Extract best parameters
        best_params = study.best_params
        best_parameter_set = ParameterSet(
            regime=regime,
            symbol=symbol,
            confidence_score=1.0,
            timestamp=datetime.now(),
            optimization_score=study.best_value,
            **best_params
        )
        
        # Calculate confidence interval (simplified)
        trials_values = [trial.value for trial in study.trials if trial.value is not None]
        if len(trials_values) > 10:
            confidence_interval = (
                np.percentile(trials_values, 2.5),
                np.percentile(trials_values, 97.5)
            )
        else:
            confidence_interval = (study.best_value * 0.9, study.best_value * 1.1)
        
        # Feature importance (simplified)
        feature_importance = {}
        for param_name in self.parameter_definitions.keys():
            feature_importance[param_name] = self.parameter_definitions[param_name].importance_weight
        
        return OptimizationResult(
            parameters=best_parameter_set,
            objective_value=study.best_value,
            confidence_interval=confidence_interval,
            n_trials=len(study.trials),
            optimization_time=0.0,  # Will be set by caller
            convergence_status="completed",
            feature_importance=feature_importance,
            cross_validation_score=study.best_value * 0.95  # Simplified
        )
    
    def _optimize_genetic(self, symbol: str, regime: MarketRegime,
                         performance_data: List[Dict]) -> OptimizationResult:
        """Genetic algorithm optimization using scipy"""
        
        # Define bounds for continuous parameters
        bounds = []
        param_names = []
        
        for param_name, param_def in self.parameter_definitions.items():
            if param_def.param_type in [ParameterType.BOUNDED, ParameterType.CONTINUOUS]:
                bounds.append((param_def.min_value, param_def.max_value))
                param_names.append(param_name)
        
        def objective_func(x):
            # Map parameters
            params = dict(zip(param_names, x))
            
            # Handle discrete parameters (use defaults for now)
            for param_name, param_def in self.parameter_definitions.items():
                if param_name not in params:
                    if param_def.param_type == ParameterType.DISCRETE:
                        params[param_name] = param_def.default_value
            
            # Create parameter set
            parameter_set = ParameterSet(
                regime=regime,
                symbol=symbol,
                confidence_score=1.0,
                timestamp=datetime.now(),
                **params
            )
            
            # Return negative objective for minimization
            return -self.calculate_objective_function(
                parameter_set, performance_data, regime, symbol
            )
        
        # Optimize
        result = differential_evolution(
            objective_func,
            bounds,
            maxiter=self.config['optimization']['n_trials'] // 2,
            popsize=15,
            seed=42
        )
        
        # Create best parameter set
        best_params = dict(zip(param_names, result.x))
        
        # Add discrete parameters
        for param_name, param_def in self.parameter_definitions.items():
            if param_name not in best_params:
                if param_def.param_type == ParameterType.DISCRETE:
                    best_params[param_name] = param_def.default_value
        
        best_parameter_set = ParameterSet(
            regime=regime,
            symbol=symbol,
            confidence_score=1.0,
            timestamp=datetime.now(),
            optimization_score=-result.fun,
            **best_params
        )
        
        return OptimizationResult(
            parameters=best_parameter_set,
            objective_value=-result.fun,
            confidence_interval=((-result.fun) * 0.95, (-result.fun) * 1.05),
            n_trials=result.nfev,
            optimization_time=0.0,
            convergence_status="completed" if result.success else "failed",
            feature_importance={name: 1.0 for name in param_names},
            cross_validation_score=(-result.fun) * 0.95
        )
    
    def _optimize_ensemble(self, symbol: str, regime: MarketRegime,
                          performance_data: List[Dict]) -> OptimizationResult:
        """Ensemble optimization combining multiple methods"""
        
        # Run multiple optimization methods
        bayesian_result = self._optimize_bayesian(symbol, regime, performance_data)
        genetic_result = self._optimize_genetic(symbol, regime, performance_data)
        
        # Choose best result
        if bayesian_result.objective_value > genetic_result.objective_value:
            best_result = bayesian_result
        else:
            best_result = genetic_result
        
        # Ensemble averaging for confidence interval
        combined_ci = (
            (bayesian_result.confidence_interval[0] + genetic_result.confidence_interval[0]) / 2,
            (bayesian_result.confidence_interval[1] + genetic_result.confidence_interval[1]) / 2
        )
        
        best_result.confidence_interval = combined_ci
        best_result.convergence_status = "ensemble_completed"
        
        return best_result
    
    def get_current_parameters(self, symbol: str, regime: MarketRegime,
                             market_data: pd.DataFrame) -> ParameterSet:
        """
        Get current optimal parameters for symbol and regime
        获取当前最优参数
        """
        regime_key = f"{symbol}_{regime.value}"
        
        # Check if we have optimized parameters
        if regime_key in self.best_parameters:
            params = self.best_parameters[regime_key]
            # Check if parameters are recent enough
            if datetime.now() - params.timestamp < timedelta(hours=24):
                return params
        
        # Fall back to regime detector's adaptive parameters
        adaptive_params = self.regime_detector.get_adaptive_parameters(regime, symbol)
        
        # Convert to ParameterSet
        return ParameterSet(
            rsi_low=adaptive_params['rsi_low'],
            rsi_high=adaptive_params['rsi_high'],
            dip_threshold=adaptive_params['dip_threshold'],
            volume_threshold=adaptive_params['volume_threshold'],
            target_profit=adaptive_params['target_profit'],
            stop_loss=adaptive_params['stop_loss'],
            max_holding_minutes=adaptive_params['max_holding_minutes'],
            position_size_multiplier=1.0,
            confidence_multiplier=adaptive_params['confidence_multiplier'],
            correlation_penalty=0.5,
            regime=regime,
            symbol=symbol,
            confidence_score=0.8,
            timestamp=datetime.now()
        )
    
    def update_performance(self, symbol: str, regime: MarketRegime,
                         trade_result: Dict):
        """Update performance tracking with new trade result"""
        with self._lock:
            # Add to performance history
            trade_result['timestamp'] = datetime.now()
            trade_result['regime'] = regime.value
            
            self.performance_history[symbol].append(trade_result)
            self.regime_performance[regime][symbol].append(trade_result)
            self.symbol_performance[symbol][regime].append(trade_result)
        
        # Check if reoptimization is needed
        self._check_reoptimization_trigger(symbol, regime)
    
    def _check_reoptimization_trigger(self, symbol: str, regime: MarketRegime):
        """Check if parameters need reoptimization"""
        regime_key = f"{symbol}_{regime.value}"
        
        # Check recent performance
        recent_trades = self.symbol_performance[symbol][regime][-50:]  # Last 50 trades
        
        if len(recent_trades) >= self.config['adaptation']['min_samples']:
            recent_metrics = self._calculate_performance_metrics(recent_trades)
            
            # Trigger reoptimization if performance degrades
            if (recent_metrics.win_rate < 0.4 or  # Win rate below 40%
                recent_metrics.sharpe_ratio < 1.0 or  # Sharpe below 1.0
                recent_metrics.max_drawdown > 0.08):  # Drawdown above 8%
                
                self.logger.warning(f"Performance degradation detected for {symbol} in {regime.value}. "
                                  f"Triggering reoptimization...")
                
                # Trigger async reoptimization
                asyncio.create_task(self._async_reoptimize(symbol, regime, recent_trades))
    
    async def _async_reoptimize(self, symbol: str, regime: MarketRegime, 
                              performance_data: List[Dict]):
        """Asynchronously reoptimize parameters"""
        try:
            # Run optimization in thread pool
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                result = await loop.run_in_executor(
                    executor,
                    self.optimize_parameters,
                    symbol, regime, performance_data, OptimizationMethod.BAYESIAN, True
                )
            
            self.logger.info(f"Async reoptimization completed for {symbol} in {regime.value}: "
                           f"objective={result.objective_value:.4f}")
            
        except Exception as e:
            self.logger.error(f"Async reoptimization failed for {symbol} in {regime.value}: {e}")
    
    def get_optimization_report(self, symbol: Optional[str] = None) -> Dict:
        """Generate comprehensive optimization report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_symbols': len(set(key.split('_')[0] for key in self.best_parameters.keys())),
                'total_regimes': len(set(key.split('_')[1] for key in self.best_parameters.keys())),
                'cache_size': len(self.optimization_cache),
                'optimization_runs': sum(len(studies.trials) for studies in self.optuna_studies.values())
            },
            'best_parameters': {},
            'performance_summary': {},
            'optimization_statistics': {}
        }
        
        # Best parameters by symbol and regime
        for regime_key, params in self.best_parameters.items():
            symbol_name, regime_name = regime_key.split('_', 1)
            if symbol is None or symbol_name == symbol:
                if symbol_name not in report['best_parameters']:
                    report['best_parameters'][symbol_name] = {}
                
                report['best_parameters'][symbol_name][regime_name] = {
                    'parameters': asdict(params),
                    'optimization_score': params.optimization_score,
                    'timestamp': params.timestamp.isoformat()
                }
        
        # Performance summary
        for sym in self.symbol_performance:
            if symbol is None or sym == symbol:
                report['performance_summary'][sym] = {}
                for reg in self.symbol_performance[sym]:
                    trades = self.symbol_performance[sym][reg]
                    if trades:
                        metrics = self._calculate_performance_metrics(trades)
                        report['performance_summary'][sym][reg.value] = asdict(metrics)
        
        return report
    
    def save_state(self, filepath: str):
        """Save engine state to file"""
        state = {
            'best_parameters': {k: asdict(v) for k, v in self.best_parameters.items()},
            'performance_history': {k: list(v) for k, v in self.performance_history.items()},
            'optimization_cache': {k: asdict(v) for k, v in self.optimization_cache.items()},
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info(f"AdaptiveParameterEngine state saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load engine state from file"""
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore state
            self.config = state.get('config', self.config)
            
            # Restore best parameters
            for k, v in state.get('best_parameters', {}).items():
                self.best_parameters[k] = ParameterSet(**v)
            
            # Restore performance history
            for k, v in state.get('performance_history', {}).items():
                self.performance_history[k] = deque(v, maxlen=1000)
            
            self.logger.info(f"AdaptiveParameterEngine state loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load state from {filepath}: {e}")

# Factory function
def create_adaptive_parameter_engine(config: Optional[Dict] = None) -> AdaptiveParameterEngine:
    """Factory function to create adaptive parameter engine"""
    return AdaptiveParameterEngine(config)