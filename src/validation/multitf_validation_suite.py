#!/usr/bin/env python3
"""
Comprehensive Multi-Timeframe Validation Suite for DipMaster Strategy
多时间框架综合验证套件 - DipMaster策略专用

This module implements a comprehensive validation and testing framework for the
multi-timeframe signal integration system. It validates all components from
signal generation to execution optimization and performance tracking.

Validation Components:
- Signal quality and confluence validation
- Execution optimization testing
- Risk management verification
- Performance tracking accuracy
- Integration layer testing
- End-to-end system validation

Test Categories:
- Unit tests for individual components
- Integration tests for component interaction
- Performance benchmarking
- Stress testing under various market conditions
- Historical backtesting validation
- Real-time simulation testing

Author: Model Backtest Validator Agent
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
import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import defaultdict, deque
from pathlib import Path
import sqlite3
import tempfile
import shutil

# Statistical testing
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Core components to test
from ..core.multi_timeframe_signal_engine import (
    MultiTimeframeSignalEngine, MultitimeframeSignal, TimeFrame, ExecutionRecommendation,
    TrendAlignmentAnalyzer, ConfluenceCalculator, ExecutionOptimizer,
    TrendDirection, SignalStrength
)
from ..core.market_regime_detector import MarketRegimeDetector, MarketRegime
from ..core.adaptive_parameter_engine import AdaptiveParameterEngine, ParameterSet
from ..core.multitf_performance_tracker import (
    MultitimeframePerformanceTracker, TradeResult, SignalOutcome
)
from ..core.multitf_strategy_orchestrator import (
    MultitimeframeStrategyOrchestrator, StrategyDecision, PortfolioContext
)

warnings.filterwarnings('ignore')

class ValidationLevel(Enum):
    """Validation testing levels"""
    UNIT = "unit"
    INTEGRATION = "integration"
    SYSTEM = "system"
    PERFORMANCE = "performance"
    STRESS = "stress"

class TestResult(Enum):
    """Test result status"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"

@dataclass
class ValidationResult:
    """Validation test result"""
    test_name: str
    test_level: ValidationLevel
    status: TestResult
    score: Optional[float]  # 0-1 performance score
    message: str
    duration_seconds: float
    timestamp: datetime
    details: Dict[str, Any]

class MarketDataGenerator:
    """Generate synthetic market data for testing"""
    
    @staticmethod
    def generate_trending_market(periods: int = 1000, trend_strength: float = 0.6,
                               volatility: float = 0.02, initial_price: float = 50000.0) -> pd.DataFrame:
        """Generate trending market data"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='5T')
        
        # Generate price with trend
        returns = np.random.normal(trend_strength/1000, volatility, periods)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = data['close'].shift(1).fillna(initial_price)
        
        # Generate high/low with some spread
        spread = volatility * prices * np.random.uniform(0.5, 1.5, periods)
        data['high'] = np.maximum(data['open'], data['close']) + spread/2
        data['low'] = np.minimum(data['open'], data['close']) - spread/2
        
        # Generate volume
        base_volume = 1000000
        volume_variation = np.random.uniform(0.5, 2.0, periods)
        data['volume'] = base_volume * volume_variation
        
        return data
    
    @staticmethod
    def generate_ranging_market(periods: int = 1000, range_size: float = 0.1,
                              volatility: float = 0.02, initial_price: float = 50000.0) -> pd.DataFrame:
        """Generate ranging/sideways market data"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='5T')
        
        # Generate mean-reverting price
        mean_reversion_speed = 0.1
        noise = np.random.normal(0, volatility, periods)
        
        prices = [initial_price]
        for i in range(1, periods):
            deviation = (prices[-1] - initial_price) / initial_price
            mean_reversion = -mean_reversion_speed * deviation
            new_price = prices[-1] * (1 + mean_reversion + noise[i])
            prices.append(new_price)
        
        # Generate OHLCV data
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = data['close'].shift(1).fillna(initial_price)
        
        # Generate high/low
        spread = volatility * np.array(prices) * np.random.uniform(0.5, 1.5, periods)
        data['high'] = np.maximum(data['open'], data['close']) + spread/2
        data['low'] = np.minimum(data['open'], data['close']) - spread/2
        
        # Generate volume
        base_volume = 1000000
        volume_variation = np.random.uniform(0.5, 2.0, periods)
        data['volume'] = base_volume * volume_variation
        
        return data
    
    @staticmethod
    def generate_volatile_market(periods: int = 1000, volatility: float = 0.05,
                               initial_price: float = 50000.0) -> pd.DataFrame:
        """Generate high volatility market data"""
        dates = pd.date_range(start='2023-01-01', periods=periods, freq='5T')
        
        # Generate volatile returns with clustering
        returns = []
        current_vol = volatility
        for i in range(periods):
            # Volatility clustering
            current_vol = 0.95 * current_vol + 0.05 * volatility + 0.1 * abs(np.random.normal(0, volatility/2))
            returns.append(np.random.normal(0, current_vol))
        
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame(index=dates)
        data['close'] = prices
        data['open'] = data['close'].shift(1).fillna(initial_price)
        
        # Generate high/low with larger spreads
        spread = current_vol * prices * np.random.uniform(1.0, 3.0, periods)
        data['high'] = np.maximum(data['open'], data['close']) + spread/2
        data['low'] = np.minimum(data['open'], data['close']) - spread/2
        
        # Generate volume (higher during volatile periods)
        base_volume = 1000000
        volume_multiplier = 1 + 2 * abs(np.array(returns) / volatility)
        data['volume'] = base_volume * volume_multiplier
        
        return data

class SignalEngineValidator:
    """Validator for multi-timeframe signal engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_results = []
    
    def validate_trend_alignment_analyzer(self) -> List[ValidationResult]:
        """Validate trend alignment analyzer"""
        results = []
        
        # Test 1: Trend direction detection accuracy
        start_time = datetime.now()
        try:
            analyzer = TrendAlignmentAnalyzer()
            
            # Test with known trending data
            trending_data = MarketDataGenerator.generate_trending_market(500, trend_strength=1.0)
            trend_direction = analyzer.analyze_trend_direction(trending_data, TimeFrame.M15)
            
            # Should detect bullish trend
            is_bullish = trend_direction in [TrendDirection.BULLISH, TrendDirection.STRONG_BULLISH]
            
            duration = (datetime.now() - start_time).total_seconds()
            
            results.append(ValidationResult(
                test_name="trend_direction_detection",
                test_level=ValidationLevel.UNIT,
                status=TestResult.PASS if is_bullish else TestResult.FAIL,
                score=1.0 if is_bullish else 0.0,
                message=f"Detected trend: {trend_direction.value}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details={'detected_trend': trend_direction.value, 'expected': 'bullish'}
            ))
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            results.append(ValidationResult(
                test_name="trend_direction_detection",
                test_level=ValidationLevel.UNIT,
                status=TestResult.FAIL,
                score=0.0,
                message=f"Error: {str(e)}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details={'error': str(e)}
            ))
        
        # Test 2: Trend strength calculation
        start_time = datetime.now()
        try:
            analyzer = TrendAlignmentAnalyzer()
            
            # Test with high trend strength data
            strong_trend_data = MarketDataGenerator.generate_trending_market(500, trend_strength=2.0)
            weak_trend_data = MarketDataGenerator.generate_ranging_market(500)
            
            strong_strength = analyzer.calculate_trend_strength(strong_trend_data, TimeFrame.M15)
            weak_strength = analyzer.calculate_trend_strength(weak_trend_data, TimeFrame.M15)
            
            # Strong trend should have higher strength
            strength_test_passed = strong_strength > weak_strength
            
            duration = (datetime.now() - start_time).total_seconds()
            
            results.append(ValidationResult(
                test_name="trend_strength_calculation",
                test_level=ValidationLevel.UNIT,
                status=TestResult.PASS if strength_test_passed else TestResult.FAIL,
                score=1.0 if strength_test_passed else 0.0,
                message=f"Strong: {strong_strength:.3f}, Weak: {weak_strength:.3f}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details={'strong_trend_strength': strong_strength, 'weak_trend_strength': weak_strength}
            ))
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            results.append(ValidationResult(
                test_name="trend_strength_calculation",
                test_level=ValidationLevel.UNIT,
                status=TestResult.FAIL,
                score=0.0,
                message=f"Error: {str(e)}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details={'error': str(e)}
            ))
        
        return results
    
    def validate_confluence_calculator(self) -> List[ValidationResult]:
        """Validate confluence calculator"""
        results = []
        
        start_time = datetime.now()
        try:
            calculator = ConfluenceCalculator()
            
            # Create mock timeframe signals with varying strengths
            from ..core.multi_timeframe_signal_engine import TimeFrameSignal
            
            # Strong bullish signal across timeframes
            strong_signals = {}
            for tf in [TimeFrame.H1, TimeFrame.M15, TimeFrame.M5]:
                strong_signals[tf] = TimeFrameSignal(
                    timeframe=tf,
                    symbol='TESTUSDT',
                    timestamp=datetime.now(),
                    trend_direction=TrendDirection.STRONG_BULLISH,
                    trend_strength=0.9,
                    momentum=0.8,
                    volatility=0.3,
                    volume_profile=0.8,
                    support_resistance={},
                    technical_indicators={},
                    signal_confidence=0.9,
                    dip_quality=0.8,
                    exit_timing_score=0.7
                )
            
            strong_confluence = calculator.calculate_confluence_score(strong_signals)
            
            # Weak mixed signals
            weak_signals = {}
            weak_signals[TimeFrame.H1] = TimeFrameSignal(
                timeframe=TimeFrame.H1,
                symbol='TESTUSDT',
                timestamp=datetime.now(),
                trend_direction=TrendDirection.BULLISH,
                trend_strength=0.3,
                momentum=0.2,
                volatility=0.7,
                volume_profile=0.4,
                support_resistance={},
                technical_indicators={},
                signal_confidence=0.4,
                dip_quality=0.3,
                exit_timing_score=0.3
            )
            weak_signals[TimeFrame.M15] = TimeFrameSignal(
                timeframe=TimeFrame.M15,
                symbol='TESTUSDT',
                timestamp=datetime.now(),
                trend_direction=TrendDirection.BEARISH,
                trend_strength=0.2,
                momentum=-0.3,
                volatility=0.8,
                volume_profile=0.3,
                support_resistance={},
                technical_indicators={},
                signal_confidence=0.3,
                dip_quality=0.2,
                exit_timing_score=0.4
            )
            
            weak_confluence = calculator.calculate_confluence_score(weak_signals)
            
            # Strong signals should have higher confluence
            confluence_test_passed = strong_confluence > weak_confluence
            
            duration = (datetime.now() - start_time).total_seconds()
            
            results.append(ValidationResult(
                test_name="confluence_calculation",
                test_level=ValidationLevel.UNIT,
                status=TestResult.PASS if confluence_test_passed else TestResult.FAIL,
                score=1.0 if confluence_test_passed else 0.0,
                message=f"Strong: {strong_confluence:.3f}, Weak: {weak_confluence:.3f}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details={
                    'strong_confluence': strong_confluence,
                    'weak_confluence': weak_confluence,
                    'difference': strong_confluence - weak_confluence
                }
            ))
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            results.append(ValidationResult(
                test_name="confluence_calculation",
                test_level=ValidationLevel.UNIT,
                status=TestResult.FAIL,
                score=0.0,
                message=f"Error: {str(e)}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details={'error': str(e)}
            ))
        
        return results
    
    def validate_signal_engine_integration(self) -> List[ValidationResult]:
        """Validate signal engine integration"""
        results = []
        
        start_time = datetime.now()
        try:
            engine = MultiTimeframeSignalEngine()
            
            # Generate test data for multiple timeframes
            test_data = {
                TimeFrame.H1: MarketDataGenerator.generate_trending_market(100, trend_strength=1.0),
                TimeFrame.M15: MarketDataGenerator.generate_trending_market(400, trend_strength=1.0),
                TimeFrame.M5: MarketDataGenerator.generate_trending_market(1200, trend_strength=1.0),
                TimeFrame.M1: MarketDataGenerator.generate_trending_market(6000, trend_strength=1.0)
            }
            
            # Update engine with data
            symbol = 'TESTUSDT'
            for timeframe, data in test_data.items():
                engine.update_market_data(symbol, timeframe, data)
            
            # Generate multi-timeframe signal
            mtf_signal = engine.generate_multitimeframe_signal(symbol)
            
            # Validate signal generation
            signal_generated = mtf_signal is not None
            
            if signal_generated:
                # Check signal properties
                has_timeframes = len(mtf_signal.timeframe_signals) >= 2
                has_confluence = mtf_signal.confluence_score is not None
                has_execution_rec = mtf_signal.recommended_action is not None
                
                overall_test_passed = has_timeframes and has_confluence and has_execution_rec
                score = sum([has_timeframes, has_confluence, has_execution_rec]) / 3
            else:
                overall_test_passed = False
                score = 0.0
            
            duration = (datetime.now() - start_time).total_seconds()
            
            details = {
                'signal_generated': signal_generated,
                'timeframes_count': len(mtf_signal.timeframe_signals) if mtf_signal else 0,
                'confluence_score': mtf_signal.confluence_score if mtf_signal else None,
                'execution_signal': mtf_signal.execution_signal if mtf_signal else None
            }
            
            results.append(ValidationResult(
                test_name="signal_engine_integration",
                test_level=ValidationLevel.INTEGRATION,
                status=TestResult.PASS if overall_test_passed else TestResult.FAIL,
                score=score,
                message=f"Signal generation: {'Success' if signal_generated else 'Failed'}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details=details
            ))
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            results.append(ValidationResult(
                test_name="signal_engine_integration",
                test_level=ValidationLevel.INTEGRATION,
                status=TestResult.FAIL,
                score=0.0,
                message=f"Error: {str(e)}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details={'error': str(e)}
            ))
        
        return results

class PerformanceTrackerValidator:
    """Validator for performance tracking system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.temp_db = None
    
    def setUp(self):
        """Set up temporary database for testing"""
        self.temp_db = tempfile.mktemp(suffix='.db')
    
    def tearDown(self):
        """Clean up temporary database"""
        if self.temp_db and Path(self.temp_db).exists():
            Path(self.temp_db).unlink()
    
    def validate_trade_recording(self) -> List[ValidationResult]:
        """Validate trade recording functionality"""
        results = []
        
        self.setUp()
        start_time = datetime.now()
        
        try:
            tracker = MultitimeframePerformanceTracker(db_path=self.temp_db)
            
            # Create test trade result
            test_trade = TradeResult(
                trade_id='TEST_001',
                symbol='TESTUSDT',
                entry_time=datetime.now() - timedelta(minutes=30),
                exit_time=datetime.now(),
                entry_price=50000.0,
                exit_price=50400.0,
                size=0.1,
                side='buy',
                pnl=40.0,
                pnl_pct=0.008,
                holding_period_minutes=30,
                execution_method='limit',
                slippage_bps=0.5,
                entry_signal=None,  # Mock signal
                exit_signal=None,
                confluence_score_entry=0.8,
                confluence_score_exit=0.6,
                signal_accuracy=0.9,
                execution_quality=0.95,
                risk_adjusted_return=0.007,
                market_regime=MarketRegime.RANGE_BOUND,
                volatility_regime='normal',
                timeframe_consistency=0.8
            )
            
            # Record trade
            tracker.record_trade_result(test_trade)
            
            # Verify recording
            recorded_trades = list(tracker.trade_results)
            recording_success = len(recorded_trades) == 1 and recorded_trades[0].trade_id == 'TEST_001'
            
            duration = (datetime.now() - start_time).total_seconds()
            
            results.append(ValidationResult(
                test_name="trade_recording",
                test_level=ValidationLevel.UNIT,
                status=TestResult.PASS if recording_success else TestResult.FAIL,
                score=1.0 if recording_success else 0.0,
                message=f"Trade recording: {'Success' if recording_success else 'Failed'}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details={'trades_recorded': len(recorded_trades)}
            ))
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            results.append(ValidationResult(
                test_name="trade_recording",
                test_level=ValidationLevel.UNIT,
                status=TestResult.FAIL,
                score=0.0,
                message=f"Error: {str(e)}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details={'error': str(e)}
            ))
        finally:
            self.tearDown()
        
        return results
    
    def validate_performance_metrics(self) -> List[ValidationResult]:
        """Validate performance metrics calculation"""
        results = []
        
        self.setUp()
        start_time = datetime.now()
        
        try:
            tracker = MultitimeframePerformanceTracker(db_path=self.temp_db)
            
            # Create multiple test trades with known outcomes
            test_trades = []
            for i in range(10):
                pnl_pct = 0.01 if i < 7 else -0.005  # 70% win rate
                test_trades.append(TradeResult(
                    trade_id=f'TEST_{i:03d}',
                    symbol='TESTUSDT',
                    entry_time=datetime.now() - timedelta(minutes=60-i*5),
                    exit_time=datetime.now() - timedelta(minutes=30-i*2),
                    entry_price=50000.0,
                    exit_price=50000.0 * (1 + pnl_pct),
                    size=0.1,
                    side='buy',
                    pnl=5000.0 * pnl_pct,
                    pnl_pct=pnl_pct,
                    holding_period_minutes=30,
                    execution_method='limit',
                    slippage_bps=0.5,
                    entry_signal=None,
                    exit_signal=None,
                    confluence_score_entry=0.8,
                    confluence_score_exit=0.6,
                    signal_accuracy=0.9,
                    execution_quality=0.95,
                    risk_adjusted_return=pnl_pct * 0.9,
                    market_regime=MarketRegime.RANGE_BOUND,
                    volatility_regime='normal',
                    timeframe_consistency=0.8
                ))
            
            # Record all trades
            for trade in test_trades:
                tracker.record_trade_result(trade)
            
            # Calculate metrics
            risk_metrics = tracker.calculate_risk_metrics('TESTUSDT')
            
            # Validate metrics
            expected_win_rate = 0.7
            calculated_win_rate = tracker.live_metrics['TESTUSDT']['win_rate']
            win_rate_accuracy = abs(calculated_win_rate - expected_win_rate) < 0.05
            
            metrics_valid = (
                risk_metrics.volatility > 0 and
                risk_metrics.max_drawdown >= 0 and
                win_rate_accuracy
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            results.append(ValidationResult(
                test_name="performance_metrics_calculation",
                test_level=ValidationLevel.UNIT,
                status=TestResult.PASS if metrics_valid else TestResult.FAIL,
                score=1.0 if metrics_valid else 0.0,
                message=f"Metrics calculation: {'Success' if metrics_valid else 'Failed'}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details={
                    'calculated_win_rate': calculated_win_rate,
                    'expected_win_rate': expected_win_rate,
                    'volatility': risk_metrics.volatility,
                    'max_drawdown': risk_metrics.max_drawdown
                }
            ))
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            results.append(ValidationResult(
                test_name="performance_metrics_calculation",
                test_level=ValidationLevel.UNIT,
                status=TestResult.FAIL,
                score=0.0,
                message=f"Error: {str(e)}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details={'error': str(e)}
            ))
        finally:
            self.tearDown()
        
        return results

class StrategyOrchestratorValidator:
    """Validator for strategy orchestrator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_decision_generation(self) -> List[ValidationResult]:
        """Validate strategy decision generation"""
        results = []
        
        start_time = datetime.now()
        try:
            orchestrator = MultitimeframeStrategyOrchestrator()
            
            # Add test symbol
            orchestrator.add_symbol('TESTUSDT')
            
            # Generate test data
            test_data = {
                TimeFrame.H1: MarketDataGenerator.generate_trending_market(100),
                TimeFrame.M15: MarketDataGenerator.generate_trending_market(400),
                TimeFrame.M5: MarketDataGenerator.generate_trending_market(1200)
            }
            
            # Update orchestrator with data
            for timeframe, data in test_data.items():
                orchestrator.update_market_data('TESTUSDT', timeframe, data)
            
            # Generate strategy decision
            decision = orchestrator.generate_strategy_decision('TESTUSDT')
            
            # Validate decision
            decision_generated = decision is not None
            
            if decision_generated:
                has_action = decision.action in ['buy', 'sell', 'hold']
                has_confidence = 0 <= decision.confidence <= 1
                has_size = decision.size >= 0
                has_reasoning = len(decision.reasoning) > 0
                
                decision_valid = has_action and has_confidence and has_size and has_reasoning
                score = sum([has_action, has_confidence, has_size, has_reasoning]) / 4
            else:
                decision_valid = False
                score = 0.0
            
            duration = (datetime.now() - start_time).total_seconds()
            
            details = {
                'decision_generated': decision_generated,
                'action': decision.action if decision else None,
                'confidence': decision.confidence if decision else None,
                'size': decision.size if decision else None,
                'has_reasoning': len(decision.reasoning) > 0 if decision else False
            }
            
            results.append(ValidationResult(
                test_name="strategy_decision_generation",
                test_level=ValidationLevel.INTEGRATION,
                status=TestResult.PASS if decision_valid else TestResult.FAIL,
                score=score,
                message=f"Decision generation: {'Success' if decision_generated else 'Failed'}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details=details
            ))
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            results.append(ValidationResult(
                test_name="strategy_decision_generation",
                test_level=ValidationLevel.INTEGRATION,
                status=TestResult.FAIL,
                score=0.0,
                message=f"Error: {str(e)}",
                duration_seconds=duration,
                timestamp=datetime.now(),
                details={'error': str(e)}
            ))
        
        return results

class SystemStressTester:
    """Stress testing for the entire multi-timeframe system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def run_market_stress_test(self) -> List[ValidationResult]:
        """Test system under various market conditions"""
        results = []
        
        market_conditions = [
            ('trending_bull', lambda: MarketDataGenerator.generate_trending_market(1000, trend_strength=2.0)),
            ('trending_bear', lambda: MarketDataGenerator.generate_trending_market(1000, trend_strength=-2.0)),
            ('high_volatility', lambda: MarketDataGenerator.generate_volatile_market(1000, volatility=0.08)),
            ('ranging_market', lambda: MarketDataGenerator.generate_ranging_market(1000)),
            ('low_volatility', lambda: MarketDataGenerator.generate_ranging_market(1000, volatility=0.005))
        ]
        
        for condition_name, data_generator in market_conditions:
            start_time = datetime.now()
            try:
                # Initialize system
                orchestrator = MultitimeframeStrategyOrchestrator()
                orchestrator.add_symbol('TESTUSDT')
                
                # Generate test data
                test_data = data_generator()
                
                # Split data into timeframes
                timeframe_data = {
                    TimeFrame.H1: test_data[::12],  # Every 12th point (1H from 5M)
                    TimeFrame.M15: test_data[::3],  # Every 3rd point (15M from 5M)
                    TimeFrame.M5: test_data        # Full data
                }
                
                # Update system with data
                for timeframe, data in timeframe_data.items():
                    orchestrator.update_market_data('TESTUSDT', timeframe, data)
                
                # Generate multiple decisions
                decisions = []
                for _ in range(10):
                    decision = orchestrator.generate_strategy_decision('TESTUSDT')
                    if decision:
                        decisions.append(decision)
                
                # Evaluate system performance under stress
                decisions_generated = len(decisions)
                avg_confidence = np.mean([d.confidence for d in decisions]) if decisions else 0.0
                
                # System should generate decisions and maintain reasonable confidence
                stress_test_passed = decisions_generated >= 5 and avg_confidence > 0.3
                
                duration = (datetime.now() - start_time).total_seconds()
                
                results.append(ValidationResult(
                    test_name=f"market_stress_test_{condition_name}",
                    test_level=ValidationLevel.STRESS,
                    status=TestResult.PASS if stress_test_passed else TestResult.FAIL,
                    score=min(decisions_generated / 10, 1.0),
                    message=f"Stress test {condition_name}: {decisions_generated} decisions, avg confidence {avg_confidence:.2f}",
                    duration_seconds=duration,
                    timestamp=datetime.now(),
                    details={
                        'market_condition': condition_name,
                        'decisions_generated': decisions_generated,
                        'avg_confidence': avg_confidence,
                        'data_points': len(test_data)
                    }
                ))
                
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                results.append(ValidationResult(
                    test_name=f"market_stress_test_{condition_name}",
                    test_level=ValidationLevel.STRESS,
                    status=TestResult.FAIL,
                    score=0.0,
                    message=f"Error in {condition_name}: {str(e)}",
                    duration_seconds=duration,
                    timestamp=datetime.now(),
                    details={'error': str(e), 'market_condition': condition_name}
                ))
        
        return results

class MultitimeframeValidationSuite:
    """
    Master validation suite for multi-timeframe system
    多时间框架系统主验证套件
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or self._get_default_config()
        
        # Initialize validators
        self.signal_validator = SignalEngineValidator()
        self.performance_validator = PerformanceTrackerValidator()
        self.orchestrator_validator = StrategyOrchestratorValidator()
        self.stress_tester = SystemStressTester()
        
        # Results storage
        self.validation_results = []
        
    def _get_default_config(self) -> Dict:
        """Default validation configuration"""
        return {
            'test_levels': [
                ValidationLevel.UNIT,
                ValidationLevel.INTEGRATION,
                ValidationLevel.SYSTEM,
                ValidationLevel.STRESS
            ],
            'performance_benchmarks': {
                'signal_accuracy_min': 0.75,
                'confluence_reliability_min': 0.80,
                'execution_quality_min': 0.90,
                'decision_generation_rate_min': 0.85
            },
            'stress_test_thresholds': {
                'min_decisions_per_condition': 5,
                'min_avg_confidence': 0.3,
                'max_error_rate': 0.1
            },
            'output_path': 'results/validation'
        }
    
    def run_all_validations(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        self.logger.info("Starting comprehensive multi-timeframe validation suite")
        
        start_time = datetime.now()
        all_results = []
        
        # Unit tests
        self.logger.info("Running unit tests...")
        all_results.extend(self.signal_validator.validate_trend_alignment_analyzer())
        all_results.extend(self.signal_validator.validate_confluence_calculator())
        all_results.extend(self.performance_validator.validate_trade_recording())
        all_results.extend(self.performance_validator.validate_performance_metrics())
        
        # Integration tests
        self.logger.info("Running integration tests...")
        all_results.extend(self.signal_validator.validate_signal_engine_integration())
        all_results.extend(self.orchestrator_validator.validate_decision_generation())
        
        # Stress tests
        self.logger.info("Running stress tests...")
        all_results.extend(self.stress_tester.run_market_stress_test())
        
        total_duration = (datetime.now() - start_time).total_seconds()
        
        # Analyze results
        summary = self._analyze_results(all_results, total_duration)
        
        # Generate report
        report = self._generate_validation_report(all_results, summary)
        
        self.logger.info(f"Validation suite completed in {total_duration:.2f} seconds")
        self.logger.info(f"Overall score: {summary['overall_score']:.2f}")
        
        return report
    
    def _analyze_results(self, results: List[ValidationResult], total_duration: float) -> Dict:
        """Analyze validation results"""
        
        # Group by test level
        results_by_level = defaultdict(list)
        for result in results:
            results_by_level[result.test_level].append(result)
        
        # Calculate summary statistics
        total_tests = len(results)
        passed_tests = len([r for r in results if r.status == TestResult.PASS])
        failed_tests = len([r for r in results if r.status == TestResult.FAIL])
        
        # Calculate scores
        scored_results = [r for r in results if r.score is not None]
        overall_score = np.mean([r.score for r in scored_results]) if scored_results else 0.0
        
        # Performance by level
        level_performance = {}
        for level, level_results in results_by_level.items():
            level_scored = [r for r in level_results if r.score is not None]
            level_performance[level.value] = {
                'total_tests': len(level_results),
                'passed': len([r for r in level_results if r.status == TestResult.PASS]),
                'failed': len([r for r in level_results if r.status == TestResult.FAIL]),
                'avg_score': np.mean([r.score for r in level_scored]) if level_scored else 0.0,
                'avg_duration': np.mean([r.duration_seconds for r in level_results])
            }
        
        # Identify critical failures
        critical_failures = [
            r for r in results 
            if r.status == TestResult.FAIL and r.test_level in [ValidationLevel.SYSTEM, ValidationLevel.INTEGRATION]
        ]
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0.0,
            'overall_score': overall_score,
            'total_duration': total_duration,
            'level_performance': level_performance,
            'critical_failures': len(critical_failures),
            'benchmark_compliance': self._check_benchmark_compliance(results)
        }
    
    def _check_benchmark_compliance(self, results: List[ValidationResult]) -> Dict:
        """Check compliance with performance benchmarks"""
        benchmarks = self.config['performance_benchmarks']
        compliance = {}
        
        # Signal accuracy compliance
        signal_results = [r for r in results if 'signal' in r.test_name.lower() and r.score is not None]
        if signal_results:
            avg_signal_score = np.mean([r.score for r in signal_results])
            compliance['signal_accuracy'] = avg_signal_score >= benchmarks['signal_accuracy_min']
        
        # Overall system compliance
        system_results = [r for r in results if r.test_level == ValidationLevel.SYSTEM and r.score is not None]
        if system_results:
            avg_system_score = np.mean([r.score for r in system_results])
            compliance['system_performance'] = avg_system_score >= benchmarks['execution_quality_min']
        
        return compliance
    
    def _generate_validation_report(self, results: List[ValidationResult], summary: Dict) -> Dict:
        """Generate comprehensive validation report"""
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'validation_suite_version': '1.0.0',
                'total_duration_seconds': summary['total_duration'],
                'config': self.config
            },
            'summary': summary,
            'detailed_results': [asdict(r) for r in results],
            'recommendations': self._generate_recommendations(results, summary),
            'benchmark_analysis': self._analyze_benchmarks(results),
            'risk_assessment': self._assess_validation_risks(results)
        }
        
        # Save report
        self._save_validation_report(report)
        
        return report
    
    def _generate_recommendations(self, results: List[ValidationResult], summary: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Overall performance recommendations
        if summary['overall_score'] < 0.8:
            recommendations.append(
                f"Overall validation score ({summary['overall_score']:.2f}) is below optimal threshold (0.8). "
                "Consider reviewing failed tests and optimizing system components."
            )
        
        # Critical failure recommendations
        if summary['critical_failures'] > 0:
            recommendations.append(
                f"Found {summary['critical_failures']} critical failures in system/integration tests. "
                "These should be addressed before deployment."
            )
        
        # Performance by level recommendations
        for level, perf in summary['level_performance'].items():
            if perf['avg_score'] < 0.7:
                recommendations.append(
                    f"{level} tests show suboptimal performance (score: {perf['avg_score']:.2f}). "
                    f"Review {perf['failed']} failed tests in this category."
                )
        
        # Specific component recommendations
        signal_tests = [r for r in results if 'signal' in r.test_name.lower()]
        signal_failures = [r for r in signal_tests if r.status == TestResult.FAIL]
        if signal_failures:
            recommendations.append(
                f"Signal generation component has {len(signal_failures)} failures. "
                "Review trend analysis and confluence calculation algorithms."
            )
        
        performance_tests = [r for r in results if 'performance' in r.test_name.lower()]
        performance_failures = [r for r in performance_tests if r.status == TestResult.FAIL]
        if performance_failures:
            recommendations.append(
                f"Performance tracking component has {len(performance_failures)} failures. "
                "Review metrics calculation and data recording functionality."
            )
        
        return recommendations
    
    def _analyze_benchmarks(self, results: List[ValidationResult]) -> Dict:
        """Analyze performance against benchmarks"""
        benchmarks = self.config['performance_benchmarks']
        analysis = {}
        
        # Group results by component
        signal_results = [r for r in results if 'signal' in r.test_name.lower() or 'confluence' in r.test_name.lower()]
        performance_results = [r for r in results if 'performance' in r.test_name.lower() or 'metric' in r.test_name.lower()]
        integration_results = [r for r in results if r.test_level == ValidationLevel.INTEGRATION]
        
        # Calculate component scores
        if signal_results:
            signal_scores = [r.score for r in signal_results if r.score is not None]
            analysis['signal_component'] = {
                'avg_score': np.mean(signal_scores) if signal_scores else 0.0,
                'benchmark': benchmarks['signal_accuracy_min'],
                'meets_benchmark': (np.mean(signal_scores) if signal_scores else 0.0) >= benchmarks['signal_accuracy_min']
            }
        
        if performance_results:
            perf_scores = [r.score for r in performance_results if r.score is not None]
            analysis['performance_component'] = {
                'avg_score': np.mean(perf_scores) if perf_scores else 0.0,
                'benchmark': benchmarks['execution_quality_min'],
                'meets_benchmark': (np.mean(perf_scores) if perf_scores else 0.0) >= benchmarks['execution_quality_min']
            }
        
        if integration_results:
            integration_scores = [r.score for r in integration_results if r.score is not None]
            analysis['integration_component'] = {
                'avg_score': np.mean(integration_scores) if integration_scores else 0.0,
                'benchmark': benchmarks['decision_generation_rate_min'],
                'meets_benchmark': (np.mean(integration_scores) if integration_scores else 0.0) >= benchmarks['decision_generation_rate_min']
            }
        
        return analysis
    
    def _assess_validation_risks(self, results: List[ValidationResult]) -> Dict:
        """Assess risks based on validation results"""
        
        risks = {
            'high': [],
            'medium': [],
            'low': []
        }
        
        # High risk: System or integration test failures
        system_failures = [r for r in results if r.status == TestResult.FAIL and r.test_level in [ValidationLevel.SYSTEM, ValidationLevel.INTEGRATION]]
        if system_failures:
            risks['high'].append(f"System integration failures: {len(system_failures)} critical tests failed")
        
        # Medium risk: Multiple unit test failures
        unit_failures = [r for r in results if r.status == TestResult.FAIL and r.test_level == ValidationLevel.UNIT]
        if len(unit_failures) > 2:
            risks['medium'].append(f"Multiple unit test failures: {len(unit_failures)} component tests failed")
        
        # Medium risk: Poor stress test performance
        stress_results = [r for r in results if r.test_level == ValidationLevel.STRESS]
        stress_failures = [r for r in stress_results if r.status == TestResult.FAIL]
        if len(stress_failures) > len(stress_results) * 0.3:  # More than 30% stress test failures
            risks['medium'].append(f"Poor stress test performance: {len(stress_failures)}/{len(stress_results)} tests failed")
        
        # Low risk: Individual component issues
        if len(unit_failures) <= 2 and not system_failures:
            risks['low'].append("Minor component issues that don't affect overall system integrity")
        
        return risks
    
    def _save_validation_report(self, report: Dict):
        """Save validation report to file"""
        output_path = Path(self.config['output_path'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_path / f"multitf_validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Validation report saved to {report_file}")

# Factory function
def run_comprehensive_validation(config: Optional[Dict] = None) -> Dict[str, Any]:
    """Run comprehensive validation suite"""
    validator = MultitimeframeValidationSuite(config)
    return validator.run_all_validations()

# CLI interface for running validations
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Timeframe Validation Suite")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--output", type=str, default="results/validation", help="Output directory")
    parser.add_argument("--level", choices=['unit', 'integration', 'system', 'stress', 'all'], 
                       default='all', help="Validation level to run")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load config if provided
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    if not config:
        config = {'output_path': args.output}
    
    # Run validation
    report = run_comprehensive_validation(config)
    
    print(f"\n{'='*50}")
    print("MULTI-TIMEFRAME VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['passed_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Pass Rate: {report['summary']['pass_rate']:.1%}")
    print(f"Overall Score: {report['summary']['overall_score']:.2f}")
    print(f"Duration: {report['summary']['total_duration']:.2f} seconds")
    
    if report['summary']['critical_failures'] > 0:
        print(f"\n⚠️  CRITICAL FAILURES: {report['summary']['critical_failures']}")
    
    if report['recommendations']:
        print(f"\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'][:3], 1):
            print(f"{i}. {rec}")
    
    print(f"\n📊 Full report saved to: {config['output_path']}")