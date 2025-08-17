#!/usr/bin/env python3
"""
Risk Control Manager for Adaptive DipMaster Strategy
自适应DipMaster策略风险控制管理器

This module implements comprehensive multi-layered risk management for the
adaptive parameter system. It provides real-time risk monitoring, position
sizing, portfolio-level risk controls, and emergency stop mechanisms.

Key Features:
- Real-time VaR and Expected Shortfall monitoring
- Dynamic position sizing based on volatility and correlation
- Portfolio-level risk aggregation and limits
- Regime-aware risk adjustments
- Emergency stop and liquidation controls
- Stress testing and scenario analysis

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
from pathlib import Path
from collections import defaultdict, deque
import threading
import asyncio
import time

# Risk and optimization libraries
import cvxpy as cp
from scipy import stats
from scipy.stats import norm, t, chi2
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, EmpiricalCovariance
from sklearn.preprocessing import StandardScaler
import arch  # GARCH models

# Market components
from .market_regime_detector import MarketRegime, RegimeSignal
from .adaptive_parameter_engine import ParameterSet
from .performance_tracker import PerformanceTracker, TradeRecord, PortfolioSnapshot
from ..types.common_types import *

warnings.filterwarnings('ignore')

class RiskLevel(Enum):
    """Risk severity levels"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class RiskMetric(Enum):
    """Types of risk metrics"""
    VAR = "var"
    EXPECTED_SHORTFALL = "expected_shortfall"
    BETA = "beta"
    CORRELATION = "correlation"
    VOLATILITY = "volatility"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    LIQUIDITY = "liquidity"

class ActionType(Enum):
    """Risk control actions"""
    MONITOR = "monitor"
    REDUCE_POSITION = "reduce_position"
    CLOSE_POSITION = "close_position"
    STOP_NEW_TRADES = "stop_new_trades"
    EMERGENCY_LIQUIDATE = "emergency_liquidate"

@dataclass
class RiskLimit:
    """Risk limit definition"""
    metric: RiskMetric
    warning_threshold: float
    critical_threshold: float
    emergency_threshold: float
    scope: str  # 'position', 'symbol', 'regime', 'portfolio'
    action_warning: ActionType
    action_critical: ActionType
    action_emergency: ActionType

@dataclass
class RiskMeasure:
    """Individual risk measurement"""
    metric: RiskMetric
    value: float
    threshold: float
    level: RiskLevel
    scope: str
    entity: str  # symbol, regime, or 'portfolio'
    timestamp: datetime
    confidence_interval: Optional[Tuple[float, float]] = None

@dataclass
class PositionRisk:
    """Risk assessment for individual position"""
    symbol: str
    regime: MarketRegime
    position_size: float
    market_value: float
    var_1d: float
    var_5d: float
    expected_shortfall: float
    volatility: float
    beta: float
    correlation_penalty: float
    liquidity_score: float
    concentration_risk: float
    recommended_size: float
    risk_level: RiskLevel

@dataclass
class PortfolioRisk:
    """Portfolio-level risk assessment"""
    total_value: float
    var_1d: float
    var_5d: float
    expected_shortfall: float
    beta: float
    tracking_error: float
    correlation_risk: float
    concentration_risk: float
    leverage: float
    liquidity_risk: float
    tail_risk: float
    stress_test_results: Dict[str, float]
    risk_level: RiskLevel
    recommended_actions: List[str]

@dataclass
class RiskAlert:
    """Risk management alert"""
    alert_id: str
    timestamp: datetime
    level: RiskLevel
    metric: RiskMetric
    title: str
    message: str
    entity: str
    current_value: float
    threshold: float
    recommended_action: ActionType
    priority: int
    auto_execute: bool

class RiskControlManager:
    """
    Comprehensive Risk Control Management System
    综合风险控制管理系统
    
    Implements multi-layered risk management:
    1. Position-level risk assessment and sizing
    2. Portfolio-level risk aggregation and monitoring
    3. Real-time VaR and Expected Shortfall calculation
    4. Regime-aware risk adjustments
    5. Emergency stop and liquidation controls
    6. Stress testing and scenario analysis
    """
    
    def __init__(self, config: Optional[Dict] = None, 
                 performance_tracker: Optional[PerformanceTracker] = None):
        """Initialize risk control manager"""
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        self.performance_tracker = performance_tracker
        
        # Risk limits and thresholds
        self.risk_limits = self._initialize_risk_limits()
        
        # Market data and correlations
        self.correlation_matrix = {}
        self.volatility_estimates = {}
        self.beta_estimates = {}
        self.liquidity_scores = {}
        
        # Risk tracking
        self.position_risks = {}
        self.portfolio_risk = None
        self.risk_alerts = deque(maxlen=1000)
        self.risk_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Emergency controls
        self.emergency_stop = False
        self.risk_override = False
        self.last_stress_test = None
        
        # Threading
        self._lock = threading.Lock()
        
        # Covariance estimators
        self.covariance_estimator = LedoitWolf()
        self.scaler = StandardScaler()
        
        self.logger.info("RiskControlManager initialized successfully")
    
    def _get_default_config(self) -> Dict:
        """Get default risk control configuration"""
        return {
            'portfolio_limits': {
                'max_total_exposure': 10000,     # USD
                'max_single_position': 2000,     # USD
                'max_leverage': 3.0,
                'daily_var_limit': 0.02,         # 2%
                'weekly_var_limit': 0.05,        # 5%
                'max_drawdown_cutoff': 0.05,     # 5%
                'max_correlation': 0.7,
                'min_liquidity_score': 0.3
            },
            'position_limits': {
                'max_position_pct': 0.25,        # 25% of portfolio
                'min_position_size': 10,         # USD
                'var_multiplier': 2.5,           # Position sizing based on VaR
                'concentration_limit': 0.3,      # 30% in single asset
                'beta_limit': 2.0,               # Maximum beta exposure
                'volatility_limit': 0.15         # 15% daily volatility
            },
            'regime_adjustments': {
                'RANGE_BOUND': {
                    'risk_multiplier': 1.0,
                    'var_limit_adjustment': 1.0,
                    'max_positions': 3
                },
                'STRONG_UPTREND': {
                    'risk_multiplier': 1.2,
                    'var_limit_adjustment': 1.1,
                    'max_positions': 2
                },
                'STRONG_DOWNTREND': {
                    'risk_multiplier': 0.5,
                    'var_limit_adjustment': 0.7,
                    'max_positions': 1
                },
                'HIGH_VOLATILITY': {
                    'risk_multiplier': 0.6,
                    'var_limit_adjustment': 0.8,
                    'max_positions': 1
                },
                'LOW_VOLATILITY': {
                    'risk_multiplier': 1.3,
                    'var_limit_adjustment': 1.2,
                    'max_positions': 4
                }
            },
            'risk_models': {
                'var_confidence': 0.95,
                'es_confidence': 0.95,
                'correlation_window': 100,
                'volatility_window': 50,
                'beta_window': 100,
                'stress_scenarios': ['covid', 'crypto_winter', 'flash_crash'],
                'monte_carlo_simulations': 10000
            },
            'alerts': {
                'alert_cooldown': 300,            # 5 minutes
                'auto_execute_threshold': 'high', # Auto-execute at high risk
                'notification_channels': ['log', 'email', 'webhook']
            }
        }
    
    def _initialize_risk_limits(self) -> Dict[str, RiskLimit]:
        """Initialize risk limits and thresholds"""
        return {
            'portfolio_var': RiskLimit(
                metric=RiskMetric.VAR,
                warning_threshold=0.015,      # 1.5%
                critical_threshold=0.02,      # 2%
                emergency_threshold=0.025,    # 2.5%
                scope='portfolio',
                action_warning=ActionType.MONITOR,
                action_critical=ActionType.REDUCE_POSITION,
                action_emergency=ActionType.EMERGENCY_LIQUIDATE
            ),
            'portfolio_es': RiskLimit(
                metric=RiskMetric.EXPECTED_SHORTFALL,
                warning_threshold=0.025,      # 2.5%
                critical_threshold=0.035,     # 3.5%
                emergency_threshold=0.05,     # 5%
                scope='portfolio',
                action_warning=ActionType.MONITOR,
                action_critical=ActionType.REDUCE_POSITION,
                action_emergency=ActionType.EMERGENCY_LIQUIDATE
            ),
            'portfolio_correlation': RiskLimit(
                metric=RiskMetric.CORRELATION,
                warning_threshold=0.6,        # 60%
                critical_threshold=0.7,       # 70%
                emergency_threshold=0.8,      # 80%
                scope='portfolio',
                action_warning=ActionType.MONITOR,
                action_critical=ActionType.REDUCE_POSITION,
                action_emergency=ActionType.STOP_NEW_TRADES
            ),
            'portfolio_concentration': RiskLimit(
                metric=RiskMetric.CONCENTRATION,
                warning_threshold=0.4,        # 40%
                critical_threshold=0.5,       # 50%
                emergency_threshold=0.6,      # 60%
                scope='portfolio',
                action_warning=ActionType.MONITOR,
                action_critical=ActionType.REDUCE_POSITION,
                action_emergency=ActionType.CLOSE_POSITION
            ),
            'position_volatility': RiskLimit(
                metric=RiskMetric.VOLATILITY,
                warning_threshold=0.1,        # 10%
                critical_threshold=0.15,      # 15%
                emergency_threshold=0.2,      # 20%
                scope='position',
                action_warning=ActionType.MONITOR,
                action_critical=ActionType.REDUCE_POSITION,
                action_emergency=ActionType.CLOSE_POSITION
            ),
            'position_beta': RiskLimit(
                metric=RiskMetric.BETA,
                warning_threshold=1.5,
                critical_threshold=2.0,
                emergency_threshold=2.5,
                scope='position',
                action_warning=ActionType.MONITOR,
                action_critical=ActionType.REDUCE_POSITION,
                action_emergency=ActionType.CLOSE_POSITION
            )
        }
    
    def calculate_position_risk(self, symbol: str, regime: MarketRegime,
                              position_size: float, market_price: float,
                              market_data: pd.DataFrame) -> PositionRisk:
        """
        Calculate comprehensive risk assessment for a position
        单个仓位的综合风险评估
        """
        market_value = position_size * market_price
        
        # Calculate volatility
        if len(market_data) >= self.config['risk_models']['volatility_window']:
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.rolling(
                window=self.config['risk_models']['volatility_window']
            ).std().iloc[-1] * np.sqrt(365 * 24 * 12)  # Annualized
        else:
            volatility = 0.02  # Default 2% daily volatility
        
        # Calculate VaR
        var_confidence = self.config['risk_models']['var_confidence']
        z_score = norm.ppf(1 - var_confidence)
        var_1d = abs(market_value * volatility / np.sqrt(365) * z_score)
        var_5d = var_1d * np.sqrt(5)
        
        # Expected Shortfall (simplified)
        es_multiplier = norm.pdf(z_score) / (1 - var_confidence)
        expected_shortfall = var_1d * es_multiplier
        
        # Beta calculation (simplified)
        beta = self._calculate_beta(symbol, market_data)
        
        # Correlation penalty
        correlation_penalty = self._calculate_correlation_penalty(symbol)
        
        # Liquidity score
        liquidity_score = self._calculate_liquidity_score(symbol, market_data)
        
        # Concentration risk
        concentration_risk = self._calculate_concentration_risk(symbol, market_value)
        
        # Apply regime adjustments
        regime_config = self.config['regime_adjustments'].get(regime.value, {})
        risk_multiplier = regime_config.get('risk_multiplier', 1.0)
        
        var_1d *= risk_multiplier
        var_5d *= risk_multiplier
        expected_shortfall *= risk_multiplier
        
        # Recommended position size based on risk
        recommended_size = self._calculate_optimal_position_size(
            symbol, market_price, volatility, beta, correlation_penalty
        )
        
        # Overall risk level
        risk_level = self._assess_position_risk_level(
            var_1d, volatility, beta, correlation_penalty, concentration_risk
        )
        
        return PositionRisk(
            symbol=symbol,
            regime=regime,
            position_size=position_size,
            market_value=market_value,
            var_1d=var_1d,
            var_5d=var_5d,
            expected_shortfall=expected_shortfall,
            volatility=volatility,
            beta=beta,
            correlation_penalty=correlation_penalty,
            liquidity_score=liquidity_score,
            concentration_risk=concentration_risk,
            recommended_size=recommended_size,
            risk_level=risk_level
        )
    
    def _calculate_beta(self, symbol: str, market_data: pd.DataFrame) -> float:
        """Calculate beta relative to market (simplified)"""
        if symbol in self.beta_estimates:
            return self.beta_estimates[symbol]
        
        # Simplified beta calculation
        if 'BTC' in symbol:
            beta = 1.0
        elif 'ETH' in symbol:
            beta = 1.2
        elif symbol in ['SOLUSDT', 'BNBUSDT', 'ADAUSDT']:
            beta = 1.5
        else:
            beta = 2.0  # Higher beta for altcoins
        
        self.beta_estimates[symbol] = beta
        return beta
    
    def _calculate_correlation_penalty(self, symbol: str) -> float:
        """Calculate correlation penalty for position sizing"""
        if not self.correlation_matrix:
            return 0.0
        
        # Find average correlation with existing positions
        existing_symbols = list(self.position_risks.keys())
        if not existing_symbols:
            return 0.0
        
        correlations = []
        for existing_symbol in existing_symbols:
            # Simplified correlation (would use actual market data)
            if symbol == existing_symbol:
                continue
            
            # Assume high correlation for crypto
            if 'USD' in symbol and 'USD' in existing_symbol:
                correlations.append(0.7)
        
        if correlations:
            avg_correlation = np.mean(correlations)
            return max(0.0, avg_correlation - 0.5)  # Penalty for correlation > 50%
        
        return 0.0
    
    def _calculate_liquidity_score(self, symbol: str, market_data: pd.DataFrame) -> float:
        """Calculate liquidity score for position"""
        if symbol in self.liquidity_scores:
            return self.liquidity_scores[symbol]
        
        # Simplified liquidity score based on volume
        if len(market_data) >= 20:
            avg_volume = market_data['volume'].rolling(window=20).mean().iloc[-1]
            volume_score = min(avg_volume / 1000000, 1.0)  # Normalize to [0,1]
        else:
            volume_score = 0.5
        
        # Tier-based adjustments
        if symbol in ['BTCUSDT', 'ETHUSDT']:
            liquidity_score = max(0.9, volume_score)
        elif symbol in ['SOLUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT']:
            liquidity_score = max(0.7, volume_score)
        else:
            liquidity_score = max(0.3, volume_score)
        
        self.liquidity_scores[symbol] = liquidity_score
        return liquidity_score
    
    def _calculate_concentration_risk(self, symbol: str, market_value: float) -> float:
        """Calculate concentration risk for symbol"""
        total_portfolio_value = sum(
            risk.market_value for risk in self.position_risks.values()
        ) + market_value
        
        if total_portfolio_value == 0:
            return 0.0
        
        concentration = market_value / total_portfolio_value
        return concentration
    
    def _calculate_optimal_position_size(self, symbol: str, market_price: float,
                                       volatility: float, beta: float,
                                       correlation_penalty: float) -> float:
        """Calculate optimal position size based on risk metrics"""
        # Target portfolio volatility
        target_portfolio_vol = 0.02  # 2% daily
        
        # Base position size using volatility targeting
        base_size = (target_portfolio_vol / volatility) * self.config['portfolio_limits']['max_total_exposure']
        
        # Adjust for beta
        beta_adjusted_size = base_size / max(beta, 0.5)
        
        # Adjust for correlation
        correlation_adjusted_size = beta_adjusted_size * (1 - correlation_penalty)
        
        # Apply position limits
        max_position = self.config['portfolio_limits']['max_single_position']
        min_position = self.config['position_limits']['min_position_size']
        
        optimal_size = max(min_position, min(max_position, correlation_adjusted_size))
        
        return optimal_size / market_price  # Return in units
    
    def _assess_position_risk_level(self, var_1d: float, volatility: float,
                                  beta: float, correlation_penalty: float,
                                  concentration_risk: float) -> RiskLevel:
        """Assess overall risk level for position"""
        risk_factors = [
            var_1d / self.config['portfolio_limits']['daily_var_limit'],
            volatility / 0.1,  # 10% volatility threshold
            beta / 2.0,        # Beta 2.0 threshold
            correlation_penalty / 0.5,  # 50% correlation threshold
            concentration_risk / 0.3     # 30% concentration threshold
        ]
        
        max_risk_factor = max(risk_factors)
        
        if max_risk_factor < 0.5:
            return RiskLevel.LOW
        elif max_risk_factor < 0.8:
            return RiskLevel.MODERATE
        elif max_risk_factor < 1.2:
            return RiskLevel.HIGH
        elif max_risk_factor < 1.5:
            return RiskLevel.CRITICAL
        else:
            return RiskLevel.EMERGENCY
    
    def calculate_portfolio_risk(self, positions: Dict, market_data: Dict[str, pd.DataFrame]) -> PortfolioRisk:
        """
        Calculate portfolio-level risk metrics
        投资组合层面风险指标计算
        """
        if not positions:
            return PortfolioRisk(
                total_value=0.0, var_1d=0.0, var_5d=0.0, expected_shortfall=0.0,
                beta=0.0, tracking_error=0.0, correlation_risk=0.0,
                concentration_risk=0.0, leverage=0.0, liquidity_risk=0.0,
                tail_risk=0.0, stress_test_results={}, risk_level=RiskLevel.LOW,
                recommended_actions=[]
            )
        
        # Calculate individual position risks
        position_risks = {}
        total_value = 0.0
        
        for symbol, position in positions.items():
            if symbol in market_data and len(market_data[symbol]) > 0:
                current_price = market_data[symbol]['close'].iloc[-1]
                market_value = position.get('quantity', 0) * current_price
                
                if market_value > 0:
                    regime = position.get('regime', MarketRegime.RANGE_BOUND)
                    position_risk = self.calculate_position_risk(
                        symbol, regime, position.get('quantity', 0),
                        current_price, market_data[symbol]
                    )
                    position_risks[symbol] = position_risk
                    total_value += market_value
        
        # Update position risks
        with self._lock:
            self.position_risks = position_risks
        
        if not position_risks:
            return PortfolioRisk(
                total_value=0.0, var_1d=0.0, var_5d=0.0, expected_shortfall=0.0,
                beta=0.0, tracking_error=0.0, correlation_risk=0.0,
                concentration_risk=0.0, leverage=0.0, liquidity_risk=0.0,
                tail_risk=0.0, stress_test_results={}, risk_level=RiskLevel.LOW,
                recommended_actions=[]
            )
        
        # Portfolio VaR calculation
        portfolio_var_1d = self._calculate_portfolio_var(position_risks)
        portfolio_var_5d = portfolio_var_1d * np.sqrt(5)
        
        # Expected Shortfall
        expected_shortfall = portfolio_var_1d * 1.3  # Simplified multiplier
        
        # Portfolio beta
        portfolio_beta = self._calculate_portfolio_beta(position_risks, total_value)
        
        # Correlation risk
        correlation_risk = self._calculate_portfolio_correlation_risk(position_risks)
        
        # Concentration risk
        concentration_risk = self._calculate_portfolio_concentration_risk(position_risks, total_value)
        
        # Leverage
        leverage = total_value / self.config['portfolio_limits']['max_total_exposure']
        
        # Liquidity risk
        liquidity_risk = self._calculate_portfolio_liquidity_risk(position_risks, total_value)
        
        # Tail risk
        tail_risk = max(portfolio_var_1d, expected_shortfall) / total_value if total_value > 0 else 0
        
        # Stress testing
        stress_test_results = self._run_stress_tests(position_risks, total_value)
        
        # Overall risk level
        risk_level = self._assess_portfolio_risk_level(
            portfolio_var_1d / total_value if total_value > 0 else 0,
            correlation_risk, concentration_risk, leverage
        )
        
        # Recommended actions
        recommended_actions = self._generate_risk_recommendations(
            portfolio_var_1d / total_value if total_value > 0 else 0,
            correlation_risk, concentration_risk, leverage, risk_level
        )
        
        portfolio_risk = PortfolioRisk(
            total_value=total_value,
            var_1d=portfolio_var_1d,
            var_5d=portfolio_var_5d,
            expected_shortfall=expected_shortfall,
            beta=portfolio_beta,
            tracking_error=0.0,  # Placeholder
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk,
            leverage=leverage,
            liquidity_risk=liquidity_risk,
            tail_risk=tail_risk,
            stress_test_results=stress_test_results,
            risk_level=risk_level,
            recommended_actions=recommended_actions
        )
        
        # Update portfolio risk
        with self._lock:
            self.portfolio_risk = portfolio_risk
        
        # Check risk limits and generate alerts
        self._check_risk_limits(portfolio_risk)
        
        return portfolio_risk
    
    def _calculate_portfolio_var(self, position_risks: Dict[str, PositionRisk]) -> float:
        """Calculate portfolio Value at Risk"""
        if not position_risks:
            return 0.0
        
        # Simple summation approach (ignoring correlations for now)
        individual_vars = [risk.var_1d for risk in position_risks.values()]
        
        # Portfolio VaR with correlation (simplified)
        # Assume average correlation of 0.7 for crypto
        avg_correlation = 0.7
        portfolio_var = np.sqrt(
            sum(var**2 for var in individual_vars) +
            2 * avg_correlation * sum(
                individual_vars[i] * individual_vars[j]
                for i in range(len(individual_vars))
                for j in range(i+1, len(individual_vars))
            )
        )
        
        return portfolio_var
    
    def _calculate_portfolio_beta(self, position_risks: Dict[str, PositionRisk], 
                                total_value: float) -> float:
        """Calculate portfolio beta"""
        if total_value == 0:
            return 0.0
        
        weighted_beta = sum(
            (risk.market_value / total_value) * risk.beta
            for risk in position_risks.values()
        )
        
        return weighted_beta
    
    def _calculate_portfolio_correlation_risk(self, position_risks: Dict[str, PositionRisk]) -> float:
        """Calculate portfolio correlation risk"""
        if len(position_risks) <= 1:
            return 0.0
        
        # Simplified correlation risk calculation
        # High correlation in crypto markets
        return 0.7  # Placeholder
    
    def _calculate_portfolio_concentration_risk(self, position_risks: Dict[str, PositionRisk],
                                              total_value: float) -> float:
        """Calculate portfolio concentration risk"""
        if total_value == 0:
            return 0.0
        
        concentrations = [
            risk.market_value / total_value
            for risk in position_risks.values()
        ]
        
        # Herfindahl-Hirschman Index for concentration
        hhi = sum(c**2 for c in concentrations)
        
        # Convert to risk score (0-1)
        return min(hhi * 2, 1.0)
    
    def _calculate_portfolio_liquidity_risk(self, position_risks: Dict[str, PositionRisk],
                                          total_value: float) -> float:
        """Calculate portfolio liquidity risk"""
        if total_value == 0:
            return 0.0
        
        weighted_liquidity = sum(
            (risk.market_value / total_value) * (1 - risk.liquidity_score)
            for risk in position_risks.values()
        )
        
        return weighted_liquidity
    
    def _run_stress_tests(self, position_risks: Dict[str, PositionRisk],
                         total_value: float) -> Dict[str, float]:
        """Run portfolio stress tests"""
        if total_value == 0:
            return {}
        
        stress_scenarios = {
            'crypto_winter_50': -0.5,     # 50% market decline
            'flash_crash_20': -0.2,       # 20% flash crash
            'volatility_spike_2x': 2.0,   # 2x volatility increase
            'correlation_100': 1.0        # Perfect correlation
        }
        
        results = {}
        
        for scenario, shock in stress_scenarios.items():
            if 'decline' in scenario or 'crash' in scenario:
                # Price shock scenario
                stressed_value = total_value * (1 + shock)
                results[scenario] = (stressed_value - total_value) / total_value
            elif 'volatility' in scenario:
                # Volatility shock - increase VaR
                stressed_var = sum(risk.var_1d * shock for risk in position_risks.values())
                results[scenario] = stressed_var / total_value
            elif 'correlation' in scenario:
                # Correlation shock - perfect correlation
                individual_vars = [risk.var_1d for risk in position_risks.values()]
                stressed_var = sum(individual_vars)  # Perfect correlation
                results[scenario] = stressed_var / total_value
        
        return results
    
    def _assess_portfolio_risk_level(self, var_pct: float, correlation_risk: float,
                                   concentration_risk: float, leverage: float) -> RiskLevel:
        """Assess overall portfolio risk level"""
        risk_factors = [
            var_pct / 0.02,              # 2% VaR threshold
            correlation_risk / 0.7,       # 70% correlation threshold
            concentration_risk / 0.5,     # 50% concentration threshold
            leverage / 3.0                # 3x leverage threshold
        ]
        
        max_risk_factor = max(risk_factors)
        
        if max_risk_factor < 0.6:
            return RiskLevel.LOW
        elif max_risk_factor < 0.8:
            return RiskLevel.MODERATE
        elif max_risk_factor < 1.0:
            return RiskLevel.HIGH
        elif max_risk_factor < 1.3:
            return RiskLevel.CRITICAL
        else:
            return RiskLevel.EMERGENCY
    
    def _generate_risk_recommendations(self, var_pct: float, correlation_risk: float,
                                     concentration_risk: float, leverage: float,
                                     risk_level: RiskLevel) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        if var_pct > 0.02:
            recommendations.append("Reduce portfolio VaR by scaling down positions")
        
        if correlation_risk > 0.7:
            recommendations.append("Diversify holdings to reduce correlation risk")
        
        if concentration_risk > 0.5:
            recommendations.append("Reduce concentration in largest positions")
        
        if leverage > 2.5:
            recommendations.append("Reduce leverage to below 2.5x")
        
        if risk_level in [RiskLevel.CRITICAL, RiskLevel.EMERGENCY]:
            recommendations.append("Consider emergency position reduction")
        
        return recommendations
    
    def _check_risk_limits(self, portfolio_risk: PortfolioRisk):
        """Check risk limits and generate alerts"""
        current_time = datetime.now()
        
        # Check portfolio VaR
        var_pct = portfolio_risk.var_1d / portfolio_risk.total_value if portfolio_risk.total_value > 0 else 0
        self._check_limit('portfolio_var', var_pct, 'portfolio', current_time)
        
        # Check Expected Shortfall
        es_pct = portfolio_risk.expected_shortfall / portfolio_risk.total_value if portfolio_risk.total_value > 0 else 0
        self._check_limit('portfolio_es', es_pct, 'portfolio', current_time)
        
        # Check correlation
        self._check_limit('portfolio_correlation', portfolio_risk.correlation_risk, 'portfolio', current_time)
        
        # Check concentration
        self._check_limit('portfolio_concentration', portfolio_risk.concentration_risk, 'portfolio', current_time)
        
        # Check individual positions
        for symbol, position_risk in self.position_risks.items():
            self._check_limit('position_volatility', position_risk.volatility, symbol, current_time)
            self._check_limit('position_beta', position_risk.beta, symbol, current_time)
    
    def _check_limit(self, limit_name: str, current_value: float, entity: str, timestamp: datetime):
        """Check individual risk limit"""
        if limit_name not in self.risk_limits:
            return
        
        risk_limit = self.risk_limits[limit_name]
        
        # Determine risk level
        if current_value >= risk_limit.emergency_threshold:
            level = RiskLevel.EMERGENCY
            action = risk_limit.action_emergency
        elif current_value >= risk_limit.critical_threshold:
            level = RiskLevel.CRITICAL
            action = risk_limit.action_critical
        elif current_value >= risk_limit.warning_threshold:
            level = RiskLevel.HIGH
            action = risk_limit.action_warning
        else:
            return  # No alert needed
        
        # Create alert
        alert = RiskAlert(
            alert_id=f"risk_{int(time.time() * 1000)}",
            timestamp=timestamp,
            level=level,
            metric=risk_limit.metric,
            title=f"{risk_limit.metric.value.title()} Limit Breach - {entity}",
            message=f"{risk_limit.metric.value} for {entity}: {current_value:.4f} > {risk_limit.warning_threshold:.4f}",
            entity=entity,
            current_value=current_value,
            threshold=risk_limit.warning_threshold,
            recommended_action=action,
            priority=self._get_alert_priority(level),
            auto_execute=level in [RiskLevel.EMERGENCY] and not self.risk_override
        )
        
        with self._lock:
            self.risk_alerts.append(alert)
        
        self.logger.warning(f"Risk Alert: {alert.title} - {alert.message}")
        
        # Auto-execute critical actions if enabled
        if alert.auto_execute and not self.emergency_stop:
            self._execute_risk_action(alert)
    
    def _get_alert_priority(self, level: RiskLevel) -> int:
        """Get numeric priority for alert level"""
        priority_map = {
            RiskLevel.LOW: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
            RiskLevel.EMERGENCY: 5
        }
        return priority_map.get(level, 1)
    
    def _execute_risk_action(self, alert: RiskAlert):
        """Execute automatic risk management action"""
        try:
            if alert.recommended_action == ActionType.EMERGENCY_LIQUIDATE:
                self.emergency_stop = True
                self.logger.critical(f"EMERGENCY STOP ACTIVATED: {alert.message}")
                
            elif alert.recommended_action == ActionType.STOP_NEW_TRADES:
                # Signal to stop new trades (would need integration with trading engine)
                self.logger.warning(f"NEW TRADES STOPPED: {alert.message}")
                
            elif alert.recommended_action == ActionType.REDUCE_POSITION:
                self.logger.warning(f"POSITION REDUCTION RECOMMENDED: {alert.message}")
                
            elif alert.recommended_action == ActionType.CLOSE_POSITION:
                self.logger.warning(f"POSITION CLOSURE RECOMMENDED: {alert.message}")
            
            # Log action
            self.logger.info(f"Risk action executed: {alert.recommended_action.value} for {alert.entity}")
            
        except Exception as e:
            self.logger.error(f"Failed to execute risk action: {e}")
    
    def validate_new_position(self, symbol: str, regime: MarketRegime,
                            proposed_size: float, market_price: float,
                            market_data: pd.DataFrame) -> Tuple[bool, float, str]:
        """
        Validate if new position meets risk criteria
        新仓位风险验证
        """
        if self.emergency_stop:
            return False, 0.0, "Emergency stop is active"
        
        # Calculate proposed position risk
        proposed_risk = self.calculate_position_risk(
            symbol, regime, proposed_size, market_price, market_data
        )
        
        # Check position-level limits
        if proposed_risk.volatility > self.config['position_limits']['volatility_limit']:
            return False, 0.0, f"Volatility too high: {proposed_risk.volatility:.3f}"
        
        if proposed_risk.beta > self.config['position_limits']['beta_limit']:
            return False, 0.0, f"Beta too high: {proposed_risk.beta:.2f}"
        
        # Check portfolio impact
        current_portfolio_value = sum(risk.market_value for risk in self.position_risks.values())
        new_portfolio_value = current_portfolio_value + proposed_risk.market_value
        
        # Check concentration
        new_concentration = proposed_risk.market_value / new_portfolio_value if new_portfolio_value > 0 else 0
        if new_concentration > self.config['position_limits']['max_position_pct']:
            suggested_size = (new_portfolio_value * self.config['position_limits']['max_position_pct']) / market_price
            return False, suggested_size, f"Concentration too high: {new_concentration:.2%}"
        
        # Check total exposure
        if new_portfolio_value > self.config['portfolio_limits']['max_total_exposure']:
            return False, 0.0, "Portfolio exposure limit exceeded"
        
        # Regime-specific checks
        regime_config = self.config['regime_adjustments'].get(regime.value, {})
        max_positions = regime_config.get('max_positions', 3)
        
        if len(self.position_risks) >= max_positions:
            return False, 0.0, f"Maximum positions for {regime.value} regime: {max_positions}"
        
        # All checks passed
        return True, proposed_size, "Position approved"
    
    def get_risk_dashboard_data(self) -> Dict:
        """Get real-time risk data for dashboard"""
        current_time = datetime.now()
        
        # Portfolio risk summary
        portfolio_summary = {}
        if self.portfolio_risk:
            portfolio_summary = {
                'total_value': self.portfolio_risk.total_value,
                'var_1d': self.portfolio_risk.var_1d,
                'var_1d_pct': self.portfolio_risk.var_1d / self.portfolio_risk.total_value if self.portfolio_risk.total_value > 0 else 0,
                'expected_shortfall': self.portfolio_risk.expected_shortfall,
                'beta': self.portfolio_risk.beta,
                'correlation_risk': self.portfolio_risk.correlation_risk,
                'concentration_risk': self.portfolio_risk.concentration_risk,
                'leverage': self.portfolio_risk.leverage,
                'risk_level': self.portfolio_risk.risk_level.value,
                'recommended_actions': self.portfolio_risk.recommended_actions
            }
        
        # Position risks
        position_summary = {}
        for symbol, risk in self.position_risks.items():
            position_summary[symbol] = {
                'market_value': risk.market_value,
                'var_1d': risk.var_1d,
                'volatility': risk.volatility,
                'beta': risk.beta,
                'risk_level': risk.risk_level.value,
                'recommended_size': risk.recommended_size
            }
        
        # Recent alerts
        recent_alerts = [
            {
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.value,
                'title': alert.title,
                'message': alert.message,
                'entity': alert.entity,
                'current_value': alert.current_value,
                'threshold': alert.threshold
            }
            for alert in list(self.risk_alerts)[-10:]
        ]
        
        # Risk limits status
        limits_status = {}
        for limit_name, limit_def in self.risk_limits.items():
            limits_status[limit_name] = {
                'metric': limit_def.metric.value,
                'warning_threshold': limit_def.warning_threshold,
                'critical_threshold': limit_def.critical_threshold,
                'emergency_threshold': limit_def.emergency_threshold,
                'scope': limit_def.scope
            }
        
        return {
            'timestamp': current_time.isoformat(),
            'emergency_stop': self.emergency_stop,
            'risk_override': self.risk_override,
            'portfolio_risk': portfolio_summary,
            'position_risks': position_summary,
            'recent_alerts': recent_alerts,
            'risk_limits': limits_status,
            'system_status': {
                'total_positions': len(self.position_risks),
                'active_alerts': len([a for a in self.risk_alerts if a.level in [RiskLevel.HIGH, RiskLevel.CRITICAL, RiskLevel.EMERGENCY]]),
                'last_update': current_time.isoformat()
            }
        }
    
    def set_emergency_stop(self, enabled: bool, reason: str = ""):
        """Set emergency stop state"""
        with self._lock:
            self.emergency_stop = enabled
        
        if enabled:
            self.logger.critical(f"EMERGENCY STOP ENABLED: {reason}")
        else:
            self.logger.info("Emergency stop disabled")
    
    def set_risk_override(self, enabled: bool, reason: str = ""):
        """Set risk override state (disable automatic risk actions)"""
        with self._lock:
            self.risk_override = enabled
        
        if enabled:
            self.logger.warning(f"RISK OVERRIDE ENABLED: {reason}")
        else:
            self.logger.info("Risk override disabled")
    
    def export_risk_report(self, output_path: str):
        """Export comprehensive risk report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'emergency_stop': self.emergency_stop,
            'risk_override': self.risk_override,
            'portfolio_risk': asdict(self.portfolio_risk) if self.portfolio_risk else None,
            'position_risks': {symbol: asdict(risk) for symbol, risk in self.position_risks.items()},
            'risk_limits': {name: asdict(limit) for name, limit in self.risk_limits.items()},
            'recent_alerts': [asdict(alert) for alert in list(self.risk_alerts)[-100:]],
            'configuration': self.config
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Risk report exported to {output_file}")
        
        return report

# Factory function
def create_risk_control_manager(config: Optional[Dict] = None,
                              performance_tracker: Optional[PerformanceTracker] = None) -> RiskControlManager:
    """Factory function to create risk control manager"""
    return RiskControlManager(config, performance_tracker)