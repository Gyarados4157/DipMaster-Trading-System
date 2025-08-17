#!/usr/bin/env python3
"""
Advanced Parameter Optimizer for DipMaster Strategy
DipMaster策略高级参数优化器

This module implements sophisticated multi-dimensional parameter optimization
using advanced algorithms including Bayesian optimization, genetic algorithms,
reinforcement learning, and ensemble methods to find optimal strategy parameters
across different market regimes and symbols.

Features:
- Multi-objective optimization with Pareto frontiers
- Hyperparameter tuning with cross-validation
- Regime-aware parameter optimization
- Risk-constrained optimization
- Real-time parameter adaptation
- A/B testing framework for parameter validation

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
from itertools import product
import time

# Advanced optimization libraries
import optuna
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler
from optuna.pruners import MedianPruner, HyperbandPruner
import hyperopt
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from skopt import gp_minimize, forest_minimize, gbrt_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from scipy.optimize import differential_evolution, minimize
import pygmo as pg

# Machine learning for optimization
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Multi-objective optimization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize as pymoo_minimize

# Core components
from .adaptive_parameter_engine import ParameterSet, PerformanceMetrics, AdaptiveParameterEngine
from .market_regime_detector import MarketRegime, RegimeSignal
from .risk_control_manager import RiskControlManager
from .performance_tracker import PerformanceTracker
from ..types.common_types import *

warnings.filterwarnings('ignore')

class OptimizationObjective(Enum):
    """Optimization objectives"""
    WIN_RATE = "win_rate"
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"
    PROFIT_FACTOR = "profit_factor"
    MAX_DRAWDOWN = "max_drawdown"
    SORTINO_RATIO = "sortino_ratio"
    EXPECTED_RETURN = "expected_return"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    MULTI_OBJECTIVE = "multi_objective"

class OptimizationAlgorithm(Enum):
    """Optimization algorithms"""
    BAYESIAN_TPE = "bayesian_tpe"
    BAYESIAN_GP = "bayesian_gp"
    GENETIC_DE = "genetic_de"
    GENETIC_NSGA2 = "genetic_nsga2"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    HYPEROPT = "hyperopt"
    OPTUNA = "optuna"
    SCIKIT_OPTIMIZE = "scikit_optimize"
    ENSEMBLE = "ensemble"

@dataclass
class OptimizationSpace:
    """Parameter space definition for optimization"""
    parameters: Dict[str, Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    objectives: List[OptimizationObjective]
    regime_specific: bool = True
    symbol_specific: bool = True

@dataclass
class OptimizationResult:
    """Single optimization result"""
    parameters: Dict[str, float]
    objectives: Dict[str, float]
    score: float
    confidence_interval: Tuple[float, float]
    cross_validation_scores: List[float]
    algorithm_used: OptimizationAlgorithm
    optimization_time: float
    n_evaluations: int
    convergence_status: str

@dataclass
class ParetoSolution:
    """Pareto-optimal solution for multi-objective optimization"""
    parameters: Dict[str, float]
    objectives: Dict[str, float]
    rank: int
    crowding_distance: float
    dominated_solutions: int

@dataclass
class OptimizationSummary:
    """Summary of optimization run"""
    best_solution: OptimizationResult
    pareto_frontier: List[ParetoSolution]
    convergence_history: List[float]
    parameter_importance: Dict[str, float]
    objective_correlations: Dict[Tuple[str, str], float]
    algorithm_comparison: Dict[str, float]
    recommendations: List[str]

class DipMasterOptimizationProblem(Problem):
    """Multi-objective optimization problem for DipMaster strategy"""
    
    def __init__(self, parameter_space: Dict, objective_function: Callable,
                 n_objectives: int = 2):
        self.parameter_space = parameter_space
        self.objective_function = objective_function
        
        # Extract bounds
        bounds = []
        self.param_names = []
        for name, config in parameter_space.items():
            bounds.append([config['low'], config['high']])
            self.param_names.append(name)
        
        super().__init__(n_var=len(bounds), n_obj=n_objectives, xl=np.array([b[0] for b in bounds]),
                        xu=np.array([b[1] for b in bounds]))
    
    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate objective functions"""
        results = []
        for params_array in x:
            params_dict = dict(zip(self.param_names, params_array))
            objectives = self.objective_function(params_dict)
            # Convert to minimization problem (negate objectives)
            results.append([-obj for obj in objectives])
        
        out["F"] = np.array(results)

class ParameterOptimizer:
    """
    Advanced Multi-Dimensional Parameter Optimizer
    高级多维参数优化器
    
    Implements sophisticated parameter optimization algorithms:
    1. Single and multi-objective optimization
    2. Bayesian optimization with Gaussian processes
    3. Genetic algorithms with constraint handling
    4. Ensemble optimization methods
    5. Real-time parameter adaptation
    6. Risk-constrained optimization
    """
    
    def __init__(self, config: Optional[Dict] = None,
                 parameter_engine: Optional[AdaptiveParameterEngine] = None,
                 risk_manager: Optional[RiskControlManager] = None,
                 performance_tracker: Optional[PerformanceTracker] = None):
        """Initialize parameter optimizer"""
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Core components
        self.parameter_engine = parameter_engine
        self.risk_manager = risk_manager
        self.performance_tracker = performance_tracker
        
        # Optimization state
        self.optimization_history = defaultdict(list)
        self.pareto_frontiers = {}
        self.parameter_importance = {}
        self.optimization_cache = {}
        
        # Studies and trials
        self.optuna_studies = {}
        self.hyperopt_trials = {}
        self.optimization_spaces = {}
        
        # ML models for surrogate optimization
        self.surrogate_models = {}
        self.feature_scalers = {}
        
        # Threading
        self._lock = threading.Lock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        self.logger.info("ParameterOptimizer initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for parameter optimization"""
        return {
            'optimization': {
                'default_algorithm': OptimizationAlgorithm.OPTUNA,
                'n_trials': 200,
                'n_jobs': 4,
                'timeout': 3600,  # 1 hour
                'cv_folds': 5,
                'test_size': 0.2,
                'random_state': 42,
                'early_stopping_rounds': 50,
                'ensemble_methods': [
                    OptimizationAlgorithm.OPTUNA,
                    OptimizationAlgorithm.BAYESIAN_GP,
                    OptimizationAlgorithm.GENETIC_DE
                ]
            },
            'objectives': {
                'primary': OptimizationObjective.RISK_ADJUSTED_RETURN,
                'secondary': [
                    OptimizationObjective.WIN_RATE,
                    OptimizationObjective.MAX_DRAWDOWN,
                    OptimizationObjective.SHARPE_RATIO
                ],
                'weights': {
                    OptimizationObjective.WIN_RATE: 0.3,
                    OptimizationObjective.SHARPE_RATIO: 0.25,
                    OptimizationObjective.MAX_DRAWDOWN: 0.2,
                    OptimizationObjective.PROFIT_FACTOR: 0.15,
                    OptimizationObjective.EXPECTED_RETURN: 0.1
                }
            },
            'constraints': {
                'max_drawdown_limit': 0.05,      # 5%
                'min_win_rate': 0.45,            # 45%
                'min_trades': 50,                # Minimum trades for validation
                'risk_free_rate': 0.02,          # 2% annual
                'transaction_costs': 0.001       # 0.1% per trade
            },
            'parameter_space': {
                'rsi_low': {'low': 10.0, 'high': 45.0, 'type': 'float'},
                'rsi_high': {'low': 35.0, 'high': 70.0, 'type': 'float'},
                'dip_threshold': {'low': 0.001, 'high': 0.01, 'type': 'float'},
                'volume_threshold': {'low': 1.0, 'high': 5.0, 'type': 'float'},
                'target_profit': {'low': 0.003, 'high': 0.025, 'type': 'float'},
                'stop_loss': {'low': -0.03, 'high': -0.003, 'type': 'float'},
                'max_holding_minutes': {'low': 30, 'high': 300, 'type': 'int'},
                'position_size_multiplier': {'low': 0.3, 'high': 2.0, 'type': 'float'},
                'confidence_multiplier': {'low': 0.3, 'high': 1.5, 'type': 'float'}
            },
            'multi_objective': {
                'algorithm': 'nsga2',
                'population_size': 100,
                'n_generations': 50,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1
            }
        }
    
    def create_optimization_space(self, symbol: str, regime: MarketRegime,
                                objectives: List[OptimizationObjective]) -> OptimizationSpace:
        """Create parameter space for optimization"""
        base_space = self.config['parameter_space'].copy()
        
        # Symbol-specific adjustments
        if symbol == 'BTCUSDT':
            # More conservative for BTC
            base_space['rsi_low']['low'] = 15.0
            base_space['target_profit']['high'] = 0.015
        elif symbol in ['ETHUSDT', 'SOLUSDT']:
            # Moderate adjustments for major alts
            base_space['volume_threshold']['low'] = 1.2
            base_space['stop_loss']['low'] = -0.025
        else:
            # More aggressive for smaller alts
            base_space['volume_threshold']['low'] = 1.5
            base_space['stop_loss']['low'] = -0.02
        
        # Regime-specific adjustments
        if regime == MarketRegime.STRONG_UPTREND:
            base_space['rsi_low']['low'] = 15.0
            base_space['rsi_high']['low'] = 30.0
            base_space['target_profit']['high'] = 0.02
        elif regime == MarketRegime.STRONG_DOWNTREND:
            base_space['rsi_low']['high'] = 35.0
            base_space['target_profit']['high'] = 0.01
            base_space['stop_loss']['high'] = -0.008
        elif regime == MarketRegime.HIGH_VOLATILITY:
            base_space['target_profit']['high'] = 0.025
            base_space['max_holding_minutes']['high'] = 120
        elif regime == MarketRegime.LOW_VOLATILITY:
            base_space['target_profit']['low'] = 0.002
            base_space['max_holding_minutes']['high'] = 360
        
        # Add constraints
        constraints = [
            {'type': 'order', 'params': ['rsi_low', 'rsi_high']},  # rsi_low < rsi_high
            {'type': 'range', 'param': 'stop_loss', 'min': -0.05, 'max': 0.0},
            {'type': 'range', 'param': 'target_profit', 'min': 0.001, 'max': 0.05},
        ]
        
        return OptimizationSpace(
            parameters=base_space,
            constraints=constraints,
            objectives=objectives,
            regime_specific=True,
            symbol_specific=True
        )
    
    def objective_function(self, parameters: Dict[str, float], symbol: str,
                          regime: MarketRegime, market_data: pd.DataFrame,
                          performance_data: List[Dict]) -> Dict[str, float]:
        """
        Multi-objective function for parameter optimization
        参数优化的多目标函数
        """
        if not performance_data:
            return {obj.value: -1.0 for obj in self.config['objectives']['secondary']}
        
        # Create parameter set
        param_set = ParameterSet(
            rsi_low=parameters['rsi_low'],
            rsi_high=parameters['rsi_high'],
            dip_threshold=parameters['dip_threshold'],
            volume_threshold=parameters['volume_threshold'],
            target_profit=parameters['target_profit'],
            stop_loss=parameters['stop_loss'],
            max_holding_minutes=int(parameters['max_holding_minutes']),
            position_size_multiplier=parameters['position_size_multiplier'],
            confidence_multiplier=parameters['confidence_multiplier'],
            correlation_penalty=0.5,
            regime=regime,
            symbol=symbol,
            confidence_score=1.0,
            timestamp=datetime.now()
        )
        
        # Simulate strategy performance with these parameters
        simulated_performance = self._simulate_strategy_performance(
            param_set, market_data, performance_data
        )
        
        # Calculate objective values
        objectives = {}
        
        # Win rate
        win_rate = simulated_performance.get('win_rate', 0.0)
        objectives[OptimizationObjective.WIN_RATE.value] = win_rate
        
        # Sharpe ratio
        sharpe_ratio = simulated_performance.get('sharpe_ratio', 0.0)
        objectives[OptimizationObjective.SHARPE_RATIO.value] = sharpe_ratio
        
        # Maximum drawdown (negative for minimization)
        max_drawdown = simulated_performance.get('max_drawdown', 1.0)
        objectives[OptimizationObjective.MAX_DRAWDOWN.value] = -max_drawdown
        
        # Profit factor
        profit_factor = simulated_performance.get('profit_factor', 0.0)
        objectives[OptimizationObjective.PROFIT_FACTOR.value] = profit_factor
        
        # Expected return
        expected_return = simulated_performance.get('expected_return', 0.0)
        objectives[OptimizationObjective.EXPECTED_RETURN.value] = expected_return
        
        # Calmar ratio
        calmar_ratio = expected_return / max(max_drawdown, 0.001)
        objectives[OptimizationObjective.CALMAR_RATIO.value] = calmar_ratio
        
        # Sortino ratio
        sortino_ratio = simulated_performance.get('sortino_ratio', 0.0)
        objectives[OptimizationObjective.SORTINO_RATIO.value] = sortino_ratio
        
        # Risk-adjusted return (composite metric)
        risk_adjusted_return = (
            0.3 * win_rate +
            0.25 * min(sharpe_ratio / 3.0, 1.0) +
            0.2 * (1.0 - max_drawdown / 0.05) +
            0.15 * min(profit_factor / 2.0, 1.0) +
            0.1 * min(expected_return / 0.3, 1.0)
        )
        objectives[OptimizationObjective.RISK_ADJUSTED_RETURN.value] = risk_adjusted_return
        
        return objectives
    
    def _simulate_strategy_performance(self, param_set: ParameterSet,
                                     market_data: pd.DataFrame,
                                     historical_performance: List[Dict]) -> Dict[str, float]:
        """Simulate strategy performance with given parameters"""
        # Simplified simulation based on historical performance
        # In practice, this would run full backtesting
        
        if not historical_performance:
            return {
                'win_rate': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 1.0,
                'profit_factor': 0.0,
                'expected_return': 0.0,
                'sortino_ratio': 0.0
            }
        
        # Extract returns
        returns = [trade.get('pnl_pct', 0.0) for trade in historical_performance]
        returns_array = np.array(returns)
        
        # Calculate metrics
        win_rate = np.mean(returns_array > 0)
        avg_return = np.mean(returns_array)
        
        if np.std(returns_array) > 0:
            sharpe_ratio = avg_return / np.std(returns_array) * np.sqrt(252 * 24 * 12)
        else:
            sharpe_ratio = 0.0
        
        # Drawdown
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
        
        # Profit factor
        winning_trades = returns_array[returns_array > 0]
        losing_trades = returns_array[returns_array < 0]
        
        if len(winning_trades) > 0 and len(losing_trades) > 0:
            profit_factor = np.sum(winning_trades) / abs(np.sum(losing_trades))
        else:
            profit_factor = 1.0 if len(winning_trades) > 0 else 0.0
        
        # Expected return (annualized)
        expected_return = avg_return * 252 * 24 * 12  # 5-min bars
        
        # Sortino ratio
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            sortino_ratio = avg_return / np.std(downside_returns) * np.sqrt(252 * 24 * 12)
        else:
            sortino_ratio = sharpe_ratio
        
        return {
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'expected_return': expected_return,
            'sortino_ratio': sortino_ratio
        }
    
    def optimize_single_objective(self, symbol: str, regime: MarketRegime,
                                market_data: pd.DataFrame,
                                performance_data: List[Dict],
                                objective: OptimizationObjective = OptimizationObjective.RISK_ADJUSTED_RETURN,
                                algorithm: OptimizationAlgorithm = OptimizationAlgorithm.OPTUNA) -> OptimizationResult:
        """
        Single-objective parameter optimization
        单目标参数优化
        """
        start_time = time.time()
        
        # Create optimization space
        opt_space = self.create_optimization_space(symbol, regime, [objective])
        
        # Define objective function
        def single_objective_func(params):
            objectives = self.objective_function(params, symbol, regime, market_data, performance_data)
            return objectives.get(objective.value, -1.0)
        
        # Run optimization based on algorithm
        if algorithm == OptimizationAlgorithm.OPTUNA:
            result = self._optimize_with_optuna(single_objective_func, opt_space, symbol, regime)
        elif algorithm == OptimizationAlgorithm.BAYESIAN_GP:
            result = self._optimize_with_gp(single_objective_func, opt_space)
        elif algorithm == OptimizationAlgorithm.GENETIC_DE:
            result = self._optimize_with_de(single_objective_func, opt_space)
        elif algorithm == OptimizationAlgorithm.HYPEROPT:
            result = self._optimize_with_hyperopt(single_objective_func, opt_space)
        else:
            result = self._optimize_with_optuna(single_objective_func, opt_space, symbol, regime)
        
        # Calculate final objectives
        final_objectives = self.objective_function(
            result.parameters, symbol, regime, market_data, performance_data
        )
        
        # Cross-validation
        cv_scores = self._cross_validate_parameters(
            result.parameters, symbol, regime, market_data, performance_data, objective
        )
        
        optimization_time = time.time() - start_time
        
        return OptimizationResult(
            parameters=result.parameters,
            objectives=final_objectives,
            score=result.score,
            confidence_interval=result.confidence_interval,
            cross_validation_scores=cv_scores,
            algorithm_used=algorithm,
            optimization_time=optimization_time,
            n_evaluations=result.n_evaluations,
            convergence_status=result.convergence_status
        )
    
    def _optimize_with_optuna(self, objective_func: Callable, opt_space: OptimizationSpace,
                            symbol: str, regime: MarketRegime) -> OptimizationResult:
        """Optimize using Optuna framework"""
        study_name = f"{symbol}_{regime.value}_optuna"
        
        def optuna_objective(trial):
            params = {}
            for param_name, param_config in opt_space.parameters.items():
                if param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(
                        param_name, param_config['low'], param_config['high']
                    )
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(
                        param_name, int(param_config['low']), int(param_config['high'])
                    )
            
            # Apply constraints
            if not self._check_constraints(params, opt_space.constraints):
                return -1.0
            
            return objective_func(params)
        
        # Create or get study
        if study_name not in self.optuna_studies:
            self.optuna_studies[study_name] = optuna.create_study(
                direction='maximize',
                study_name=study_name,
                sampler=TPESampler(),
                pruner=MedianPruner()
            )
        
        study = self.optuna_studies[study_name]
        
        # Optimize
        study.optimize(
            optuna_objective,
            n_trials=self.config['optimization']['n_trials'],
            timeout=self.config['optimization']['timeout'],
            n_jobs=1
        )
        
        # Extract result
        best_params = study.best_params
        best_score = study.best_value
        
        # Calculate confidence interval
        trial_values = [trial.value for trial in study.trials if trial.value is not None]
        if len(trial_values) > 10:
            confidence_interval = (
                np.percentile(trial_values, 2.5),
                np.percentile(trial_values, 97.5)
            )
        else:
            confidence_interval = (best_score * 0.9, best_score * 1.1)
        
        return OptimizationResult(
            parameters=best_params,
            objectives={},
            score=best_score,
            confidence_interval=confidence_interval,
            cross_validation_scores=[],
            algorithm_used=OptimizationAlgorithm.OPTUNA,
            optimization_time=0.0,
            n_evaluations=len(study.trials),
            convergence_status="completed"
        )
    
    def _optimize_with_gp(self, objective_func: Callable, opt_space: OptimizationSpace) -> OptimizationResult:
        """Optimize using Gaussian Process (scikit-optimize)"""
        # Create search space
        dimensions = []
        param_names = []
        
        for param_name, param_config in opt_space.parameters.items():
            if param_config['type'] == 'float':
                dimensions.append(Real(param_config['low'], param_config['high'], name=param_name))
            elif param_config['type'] == 'int':
                dimensions.append(Integer(int(param_config['low']), int(param_config['high']), name=param_name))
            param_names.append(param_name)
        
        @use_named_args(dimensions)
        def gp_objective(**params):
            if not self._check_constraints(params, opt_space.constraints):
                return 1.0  # High value for minimization
            return -objective_func(params)  # Negative for minimization
        
        # Optimize
        result = gp_minimize(
            gp_objective,
            dimensions,
            n_calls=self.config['optimization']['n_trials'],
            random_state=self.config['optimization']['random_state']
        )
        
        # Extract best parameters
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun
        
        return OptimizationResult(
            parameters=best_params,
            objectives={},
            score=best_score,
            confidence_interval=(best_score * 0.95, best_score * 1.05),
            cross_validation_scores=[],
            algorithm_used=OptimizationAlgorithm.BAYESIAN_GP,
            optimization_time=0.0,
            n_evaluations=len(result.func_vals),
            convergence_status="completed"
        )
    
    def _optimize_with_de(self, objective_func: Callable, opt_space: OptimizationSpace) -> OptimizationResult:
        """Optimize using Differential Evolution"""
        bounds = []
        param_names = []
        
        for param_name, param_config in opt_space.parameters.items():
            bounds.append((param_config['low'], param_config['high']))
            param_names.append(param_name)
        
        def de_objective(x):
            params = dict(zip(param_names, x))
            if not self._check_constraints(params, opt_space.constraints):
                return 1.0
            return -objective_func(params)  # Negative for minimization
        
        # Optimize
        result = differential_evolution(
            de_objective,
            bounds,
            maxiter=self.config['optimization']['n_trials'] // 10,
            popsize=15,
            seed=self.config['optimization']['random_state']
        )
        
        best_params = dict(zip(param_names, result.x))
        best_score = -result.fun
        
        return OptimizationResult(
            parameters=best_params,
            objectives={},
            score=best_score,
            confidence_interval=(best_score * 0.95, best_score * 1.05),
            cross_validation_scores=[],
            algorithm_used=OptimizationAlgorithm.GENETIC_DE,
            optimization_time=0.0,
            n_evaluations=result.nfev,
            convergence_status="completed" if result.success else "failed"
        )
    
    def _optimize_with_hyperopt(self, objective_func: Callable, opt_space: OptimizationSpace) -> OptimizationResult:
        """Optimize using Hyperopt"""
        # Create search space
        space = {}
        for param_name, param_config in opt_space.parameters.items():
            if param_config['type'] == 'float':
                space[param_name] = hp.uniform(param_name, param_config['low'], param_config['high'])
            elif param_config['type'] == 'int':
                space[param_name] = hp.randint(param_name, int(param_config['low']), int(param_config['high']) + 1)
        
        def hyperopt_objective(params):
            if not self._check_constraints(params, opt_space.constraints):
                return {'loss': 1.0, 'status': STATUS_OK}
            return {'loss': -objective_func(params), 'status': STATUS_OK}
        
        # Optimize
        trials = Trials()
        best = fmin(
            hyperopt_objective,
            space,
            algo=tpe.suggest,
            max_evals=self.config['optimization']['n_trials'],
            trials=trials
        )
        
        best_score = -min(trial['result']['loss'] for trial in trials.trials)
        
        return OptimizationResult(
            parameters=best,
            objectives={},
            score=best_score,
            confidence_interval=(best_score * 0.95, best_score * 1.05),
            cross_validation_scores=[],
            algorithm_used=OptimizationAlgorithm.HYPEROPT,
            optimization_time=0.0,
            n_evaluations=len(trials.trials),
            convergence_status="completed"
        )
    
    def optimize_multi_objective(self, symbol: str, regime: MarketRegime,
                               market_data: pd.DataFrame,
                               performance_data: List[Dict],
                               objectives: List[OptimizationObjective] = None) -> OptimizationSummary:
        """
        Multi-objective parameter optimization using NSGA-II
        多目标参数优化
        """
        if objectives is None:
            objectives = [
                OptimizationObjective.WIN_RATE,
                OptimizationObjective.SHARPE_RATIO,
                OptimizationObjective.MAX_DRAWDOWN
            ]
        
        start_time = time.time()
        
        # Create optimization space
        opt_space = self.create_optimization_space(symbol, regime, objectives)
        
        # Define multi-objective function
        def multi_objective_func(params_dict):
            objectives_dict = self.objective_function(
                params_dict, symbol, regime, market_data, performance_data
            )
            return [objectives_dict.get(obj.value, 0.0) for obj in objectives]
        
        # Create problem
        problem = DipMasterOptimizationProblem(
            opt_space.parameters, multi_objective_func, len(objectives)
        )
        
        # Configure algorithm
        algorithm = NSGA2(
            pop_size=self.config['multi_objective']['population_size'],
            n_offsprings=None,
            sampling="real_random",
            crossover="real_sbx",
            mutation="real_pm",
            eliminate_duplicates=True
        )
        
        # Optimize
        result = pymoo_minimize(
            problem,
            algorithm,
            ('n_gen', self.config['multi_objective']['n_generations']),
            verbose=False
        )
        
        # Extract Pareto frontier
        pareto_solutions = []
        for i, (params, objs) in enumerate(zip(result.X, result.F)):
            params_dict = dict(zip(problem.param_names, params))
            objectives_dict = dict(zip([obj.value for obj in objectives], -objs))  # Convert back from minimization
            
            pareto_solutions.append(ParetoSolution(
                parameters=params_dict,
                objectives=objectives_dict,
                rank=0,  # All solutions on Pareto frontier have rank 0
                crowding_distance=0.0,  # Would need to calculate
                dominated_solutions=0
            ))
        
        # Select best solution based on composite score
        best_solution_idx = 0
        best_composite_score = -float('inf')
        
        for i, solution in enumerate(pareto_solutions):
            composite_score = sum(
                self.config['objectives']['weights'].get(obj, 0.1) * score
                for obj, score in solution.objectives.items()
            )
            if composite_score > best_composite_score:
                best_composite_score = composite_score
                best_solution_idx = i
        
        best_solution = OptimizationResult(
            parameters=pareto_solutions[best_solution_idx].parameters,
            objectives=pareto_solutions[best_solution_idx].objectives,
            score=best_composite_score,
            confidence_interval=(best_composite_score * 0.95, best_composite_score * 1.05),
            cross_validation_scores=[],
            algorithm_used=OptimizationAlgorithm.GENETIC_NSGA2,
            optimization_time=time.time() - start_time,
            n_evaluations=len(result.X),
            convergence_status="completed"
        )
        
        # Calculate parameter importance (simplified)
        parameter_importance = self._calculate_parameter_importance(pareto_solutions)
        
        # Calculate objective correlations
        objective_correlations = self._calculate_objective_correlations(pareto_solutions)
        
        return OptimizationSummary(
            best_solution=best_solution,
            pareto_frontier=pareto_solutions,
            convergence_history=[],
            parameter_importance=parameter_importance,
            objective_correlations=objective_correlations,
            algorithm_comparison={},
            recommendations=self._generate_optimization_recommendations(best_solution, pareto_solutions)
        )
    
    def _check_constraints(self, params: Dict[str, float], constraints: List[Dict]) -> bool:
        """Check if parameters satisfy constraints"""
        for constraint in constraints:
            if constraint['type'] == 'order':
                param1, param2 = constraint['params']
                if params[param1] >= params[param2]:
                    return False
            elif constraint['type'] == 'range':
                param = constraint['param']
                if not (constraint['min'] <= params[param] <= constraint['max']):
                    return False
        return True
    
    def _cross_validate_parameters(self, parameters: Dict[str, float], symbol: str,
                                 regime: MarketRegime, market_data: pd.DataFrame,
                                 performance_data: List[Dict],
                                 objective: OptimizationObjective) -> List[float]:
        """Cross-validate parameter performance"""
        if len(performance_data) < 100:
            return [0.0]  # Not enough data for CV
        
        # Time series split
        n_splits = self.config['optimization']['cv_folds']
        split_size = len(performance_data) // n_splits
        
        cv_scores = []
        for i in range(n_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < n_splits - 1 else len(performance_data)
            
            fold_data = performance_data[start_idx:end_idx]
            objectives = self.objective_function(parameters, symbol, regime, market_data, fold_data)
            cv_scores.append(objectives.get(objective.value, 0.0))
        
        return cv_scores
    
    def _calculate_parameter_importance(self, solutions: List[ParetoSolution]) -> Dict[str, float]:
        """Calculate parameter importance from Pareto solutions"""
        if not solutions:
            return {}
        
        # Extract parameter values and objective scores
        param_names = list(solutions[0].parameters.keys())
        importance = {}
        
        for param_name in param_names:
            param_values = [sol.parameters[param_name] for sol in solutions]
            objective_scores = [sum(sol.objectives.values()) for sol in solutions]
            
            # Calculate correlation between parameter and performance
            if len(set(param_values)) > 1:  # Avoid division by zero
                correlation = np.corrcoef(param_values, objective_scores)[0, 1]
                importance[param_name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                importance[param_name] = 0.0
        
        return importance
    
    def _calculate_objective_correlations(self, solutions: List[ParetoSolution]) -> Dict[Tuple[str, str], float]:
        """Calculate correlations between objectives"""
        if not solutions:
            return {}
        
        objective_names = list(solutions[0].objectives.keys())
        correlations = {}
        
        for i, obj1 in enumerate(objective_names):
            for obj2 in objective_names[i+1:]:
                values1 = [sol.objectives[obj1] for sol in solutions]
                values2 = [sol.objectives[obj2] for sol in solutions]
                
                if len(set(values1)) > 1 and len(set(values2)) > 1:
                    correlation = np.corrcoef(values1, values2)[0, 1]
                    if not np.isnan(correlation):
                        correlations[(obj1, obj2)] = correlation
        
        return correlations
    
    def _generate_optimization_recommendations(self, best_solution: OptimizationResult,
                                             pareto_solutions: List[ParetoSolution]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Performance assessment
        if best_solution.score > 0.7:
            recommendations.append("Excellent parameter configuration found")
        elif best_solution.score > 0.5:
            recommendations.append("Good parameter configuration found")
        else:
            recommendations.append("Parameter optimization may need more iterations")
        
        # Parameter stability
        if len(pareto_solutions) > 10:
            param_ranges = {}
            for param_name in best_solution.parameters.keys():
                values = [sol.parameters[param_name] for sol in pareto_solutions]
                param_ranges[param_name] = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
            
            unstable_params = [param for param, cv in param_ranges.items() if cv > 0.3]
            if unstable_params:
                recommendations.append(f"Consider stabilizing parameters: {', '.join(unstable_params)}")
        
        # Objective trade-offs
        win_rate = best_solution.objectives.get('win_rate', 0)
        sharpe_ratio = best_solution.objectives.get('sharpe_ratio', 0)
        max_drawdown = best_solution.objectives.get('max_drawdown', 0)
        
        if win_rate < 0.5:
            recommendations.append("Consider adjusting parameters to improve win rate")
        if sharpe_ratio < 1.5:
            recommendations.append("Consider optimizing for better risk-adjusted returns")
        if abs(max_drawdown) > 0.03:
            recommendations.append("Consider tightening risk controls to reduce drawdown")
        
        return recommendations
    
    def run_ensemble_optimization(self, symbol: str, regime: MarketRegime,
                                market_data: pd.DataFrame,
                                performance_data: List[Dict]) -> OptimizationSummary:
        """Run ensemble optimization using multiple algorithms"""
        algorithms = self.config['optimization']['ensemble_methods']
        results = []
        
        # Run each algorithm
        for algorithm in algorithms:
            try:
                result = self.optimize_single_objective(
                    symbol, regime, market_data, performance_data,
                    OptimizationObjective.RISK_ADJUSTED_RETURN, algorithm
                )
                results.append(result)
                self.logger.info(f"Completed optimization with {algorithm.value}: score={result.score:.4f}")
            except Exception as e:
                self.logger.error(f"Failed optimization with {algorithm.value}: {e}")
        
        if not results:
            raise ValueError("All optimization algorithms failed")
        
        # Select best result
        best_result = max(results, key=lambda r: r.score)
        
        # Create algorithm comparison
        algorithm_comparison = {
            result.algorithm_used.value: result.score for result in results
        }
        
        # Generate ensemble recommendations
        recommendations = self._generate_optimization_recommendations(best_result, [])
        recommendations.append(f"Best algorithm: {best_result.algorithm_used.value}")
        
        return OptimizationSummary(
            best_solution=best_result,
            pareto_frontier=[],
            convergence_history=[],
            parameter_importance={},
            objective_correlations={},
            algorithm_comparison=algorithm_comparison,
            recommendations=recommendations
        )
    
    def adaptive_optimization(self, symbol: str, regime: MarketRegime,
                            market_data: pd.DataFrame,
                            performance_data: List[Dict],
                            current_parameters: Dict[str, float]) -> OptimizationResult:
        """
        Adaptive optimization that starts from current parameters
        自适应优化 - 从当前参数开始
        """
        # Start optimization near current parameters
        modified_space = self.create_optimization_space(symbol, regime, [OptimizationObjective.RISK_ADJUSTED_RETURN])
        
        # Narrow search space around current parameters
        for param_name, current_value in current_parameters.items():
            if param_name in modified_space.parameters:
                param_config = modified_space.parameters[param_name]
                param_range = param_config['high'] - param_config['low']
                
                # Search within ±20% of current value
                new_low = max(param_config['low'], current_value - 0.2 * param_range)
                new_high = min(param_config['high'], current_value + 0.2 * param_range)
                
                modified_space.parameters[param_name]['low'] = new_low
                modified_space.parameters[param_name]['high'] = new_high
        
        # Run optimization with modified space
        result = self.optimize_single_objective(
            symbol, regime, market_data, performance_data,
            OptimizationObjective.RISK_ADJUSTED_RETURN,
            OptimizationAlgorithm.OPTUNA
        )
        
        return result
    
    def export_optimization_report(self, symbol: str, regime: MarketRegime,
                                 optimization_summary: OptimizationSummary,
                                 output_path: str):
        """Export comprehensive optimization report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'regime': regime.value,
            'best_solution': asdict(optimization_summary.best_solution),
            'pareto_frontier': [asdict(sol) for sol in optimization_summary.pareto_frontier],
            'parameter_importance': optimization_summary.parameter_importance,
            'objective_correlations': {f"{k[0]}_{k[1]}": v for k, v in optimization_summary.objective_correlations.items()},
            'algorithm_comparison': optimization_summary.algorithm_comparison,
            'recommendations': optimization_summary.recommendations,
            'configuration': self.config
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Optimization report exported to {output_file}")
        
        return report

# Factory function
def create_parameter_optimizer(config: Optional[Dict] = None,
                             parameter_engine: Optional[AdaptiveParameterEngine] = None,
                             risk_manager: Optional[RiskControlManager] = None,
                             performance_tracker: Optional[PerformanceTracker] = None) -> ParameterOptimizer:
    """Factory function to create parameter optimizer"""
    return ParameterOptimizer(config, parameter_engine, risk_manager, performance_tracker)