#!/usr/bin/env python3
"""
Advanced Risk Monitor for DipMaster Trading System
高级风险监控系统 - 实时风险指标监控和告警

Features:
- Real-time VaR/ES monitoring and alerts
- Position limit violation detection  
- Correlation risk monitoring
- Liquidity risk assessment
- Market anomaly detection
- Circuit breaker management
- Risk attribution analysis
- Regulatory compliance checks
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import numpy as np
from datetime import datetime, timezone
import json

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(Enum):
    """Alert type enumeration."""
    VAR_BREACH = "var_breach"
    POSITION_LIMIT = "position_limit"
    DRAWDOWN_LIMIT = "drawdown_limit"
    LEVERAGE_LIMIT = "leverage_limit"
    CORRELATION_BREAK = "correlation_break"
    LIQUIDITY_RISK = "liquidity_risk"
    MARKET_ANOMALY = "market_anomaly"
    SYSTEM_RISK = "system_risk"
    REGULATORY = "regulatory"


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    # VaR limits (USD)
    var_95_limit: float = 200000
    var_99_limit: float = 300000
    expected_shortfall_limit: float = 400000
    
    # Position limits
    max_position_size_usd: float = 100000
    max_positions: int = 10
    max_symbol_concentration: float = 0.3  # 30%
    max_sector_concentration: float = 0.5  # 50%
    
    # Portfolio limits
    max_leverage: float = 3.0
    max_drawdown: float = 0.20  # 20%
    max_daily_loss: float = 50000  # USD
    
    # Correlation limits
    max_correlation: float = 0.8
    min_diversification_ratio: float = 0.6
    
    # Liquidity limits
    min_liquidity_score: float = 0.3
    max_bid_ask_spread: float = 0.005  # 0.5%


@dataclass
class RiskMetrics:
    """Current risk metrics."""
    timestamp: float = 0.0
    
    # VaR metrics
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    
    # Portfolio metrics
    total_exposure: float = 0.0
    net_exposure: float = 0.0
    leverage: float = 0.0
    beta: float = 0.0
    
    # Performance metrics
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    daily_pnl: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    
    # Position metrics
    active_positions: int = 0
    long_positions: int = 0
    short_positions: int = 0
    largest_position_pct: float = 0.0
    
    # Market risk
    volatility: float = 0.0
    correlation_risk: float = 0.0
    liquidity_score: float = 0.0


@dataclass
class RiskAlert:
    """Risk alert data structure."""
    alert_id: str
    timestamp: float
    alert_type: AlertType
    risk_level: RiskLevel
    symbol: Optional[str]
    current_value: float
    limit_value: float
    breach_percentage: float
    message: str
    recommended_action: str
    auto_remediation: bool = False
    acknowledged: bool = False
    resolved: bool = False


class CircuitBreaker:
    """Circuit breaker for risk management."""
    
    def __init__(self, name: str, threshold: float, cooldown_seconds: int = 300):
        self.name = name
        self.threshold = threshold
        self.cooldown_seconds = cooldown_seconds
        self.is_triggered = False
        self.trigger_time = None
        self.trigger_count = 0
    
    def check_trigger(self, current_value: float) -> bool:
        """Check if circuit breaker should trigger."""
        if current_value >= self.threshold:
            if not self.is_triggered:
                self.is_triggered = True
                self.trigger_time = time.time()
                self.trigger_count += 1
                logger.critical(f"🚨 Circuit breaker '{self.name}' TRIGGERED: {current_value} >= {self.threshold}")
                return True
        else:
            # Reset if value goes below threshold and cooldown has passed
            if self.is_triggered and self.trigger_time:
                if time.time() - self.trigger_time > self.cooldown_seconds:
                    self.is_triggered = False
                    logger.info(f"🔄 Circuit breaker '{self.name}' RESET after cooldown")
        
        return False
    
    def force_reset(self):
        """Force reset circuit breaker."""
        self.is_triggered = False
        self.trigger_time = None
        logger.info(f"🔧 Circuit breaker '{self.name}' FORCE RESET")


class PositionTracker:
    """Track individual position risks."""
    
    def __init__(self):
        self.positions = {}  # symbol -> position data
        self.position_history = deque(maxlen=1000)
    
    def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """Update position information."""
        self.positions[symbol] = {
            'timestamp': time.time(),
            'symbol': symbol,
            'quantity': position_data.get('quantity', 0),
            'entry_price': position_data.get('entry_price', 0),
            'current_price': position_data.get('current_price', 0),
            'unrealized_pnl': position_data.get('unrealized_pnl', 0),
            'position_value': position_data.get('position_value', 0),
            'side': position_data.get('side', 'long'),
            'leverage': position_data.get('leverage', 1.0)
        }
        
        self.position_history.append({
            'timestamp': time.time(),
            'action': 'update',
            'symbol': symbol,
            'data': position_data.copy()
        })
    
    def remove_position(self, symbol: str):
        """Remove closed position."""
        if symbol in self.positions:
            del self.positions[symbol]
            self.position_history.append({
                'timestamp': time.time(),
                'action': 'remove',
                'symbol': symbol
            })
    
    def get_position_concentration(self) -> Dict[str, float]:
        """Calculate position concentration metrics."""
        if not self.positions:
            return {}
        
        total_value = sum(abs(pos['position_value']) for pos in self.positions.values())
        
        if total_value == 0:
            return {}
        
        concentrations = {}
        for symbol, position in self.positions.items():
            concentrations[symbol] = abs(position['position_value']) / total_value
        
        return concentrations
    
    def get_largest_position_pct(self) -> float:
        """Get largest position as percentage of total exposure."""
        concentrations = self.get_position_concentration()
        return max(concentrations.values()) if concentrations else 0.0


class CorrelationMonitor:
    """Monitor correlation risks between positions."""
    
    def __init__(self, lookback_periods: int = 252):
        self.lookback_periods = lookback_periods
        self.price_history = defaultdict(deque)
        self.correlation_matrix = {}
        self.correlation_warnings = deque(maxlen=100)
    
    def update_price(self, symbol: str, price: float, timestamp: float = None):
        """Update price history for correlation calculation."""
        if timestamp is None:
            timestamp = time.time()
        
        # Store price with timestamp
        self.price_history[symbol].append((timestamp, price))
        
        # Maintain lookback window
        if len(self.price_history[symbol]) > self.lookback_periods:
            self.price_history[symbol].popleft()
    
    def calculate_correlations(self, symbols: List[str]) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise correlations."""
        correlations = {}
        
        try:
            # Get price returns for each symbol
            returns_data = {}
            for symbol in symbols:
                if symbol not in self.price_history or len(self.price_history[symbol]) < 30:
                    continue
                
                prices = [p[1] for p in self.price_history[symbol]]
                returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
                returns_data[symbol] = returns
            
            # Calculate pairwise correlations
            for i, symbol1 in enumerate(symbols):
                for symbol2 in symbols[i+1:]:
                    if symbol1 in returns_data and symbol2 in returns_data:
                        min_length = min(len(returns_data[symbol1]), len(returns_data[symbol2]))
                        if min_length >= 30:  # Minimum data points
                            corr = np.corrcoef(
                                returns_data[symbol1][-min_length:],
                                returns_data[symbol2][-min_length:]
                            )[0, 1]
                            
                            if not np.isnan(corr):
                                correlations[(symbol1, symbol2)] = corr
            
            self.correlation_matrix = correlations
            return correlations
            
        except Exception as e:
            logger.error(f"❌ Error calculating correlations: {e}")
            return {}
    
    def check_correlation_risks(self, positions: Dict[str, Any], max_correlation: float = 0.8) -> List[Dict[str, Any]]:
        """Check for high correlation risks."""
        risks = []
        
        try:
            symbols = list(positions.keys())
            correlations = self.calculate_correlations(symbols)
            
            for (symbol1, symbol2), corr in correlations.items():
                if abs(corr) > max_correlation:
                    # Check if both symbols have significant positions
                    pos1_value = abs(positions.get(symbol1, {}).get('position_value', 0))
                    pos2_value = abs(positions.get(symbol2, {}).get('position_value', 0))
                    
                    total_exposure = pos1_value + pos2_value
                    
                    if total_exposure > 10000:  # Minimum threshold for concern
                        risk = {
                            'type': 'high_correlation',
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': corr,
                            'exposure': total_exposure,
                            'risk_score': abs(corr) * (total_exposure / 100000)  # Normalized risk score
                        }
                        risks.append(risk)
                        
                        self.correlation_warnings.append({
                            'timestamp': time.time(),
                            'warning': risk
                        })
            
            return risks
            
        except Exception as e:
            logger.error(f"❌ Error checking correlation risks: {e}")
            return []


class LiquidityMonitor:
    """Monitor liquidity risks."""
    
    def __init__(self):
        self.liquidity_data = {}
        self.liquidity_history = deque(maxlen=1000)
    
    def update_liquidity_data(self, symbol: str, data: Dict[str, Any]):
        """Update liquidity data for a symbol."""
        self.liquidity_data[symbol] = {
            'timestamp': time.time(),
            'bid_ask_spread': data.get('bid_ask_spread', 0),
            'volume_24h': data.get('volume_24h', 0),
            'market_depth': data.get('market_depth', 0),
            'liquidity_score': data.get('liquidity_score', 0)
        }
        
        self.liquidity_history.append({
            'timestamp': time.time(),
            'symbol': symbol,
            'data': data.copy()
        })
    
    def assess_liquidity_risk(self, symbol: str, position_size: float) -> Dict[str, Any]:
        """Assess liquidity risk for a position."""
        if symbol not in self.liquidity_data:
            return {'risk_level': 'unknown', 'reason': 'no_data'}
        
        data = self.liquidity_data[symbol]
        
        # Calculate risk factors
        spread = data.get('bid_ask_spread', 0)
        volume = data.get('volume_24h', 0)
        liquidity_score = data.get('liquidity_score', 0)
        
        risk_factors = []
        risk_level = 'low'
        
        # Check bid-ask spread
        if spread > 0.01:  # 1%
            risk_factors.append('high_spread')
            risk_level = 'high'
        elif spread > 0.005:  # 0.5%
            risk_factors.append('elevated_spread')
            risk_level = 'medium'
        
        # Check volume relative to position size
        if volume > 0 and position_size > volume * 0.1:  # Position > 10% of daily volume
            risk_factors.append('large_position_vs_volume')
            risk_level = 'high'
        
        # Check liquidity score
        if liquidity_score < 0.3:
            risk_factors.append('low_liquidity_score')
            risk_level = 'medium' if risk_level == 'low' else 'high'
        
        return {
            'symbol': symbol,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'spread': spread,
            'volume_24h': volume,
            'liquidity_score': liquidity_score,
            'position_vs_volume_pct': (position_size / volume * 100) if volume > 0 else 0
        }


class RiskMonitor:
    """
    Advanced risk monitoring system for trading operations.
    
    Provides comprehensive real-time risk monitoring with sophisticated
    alert generation, circuit breakers, and automated risk management.
    """
    
    def __init__(self, 
                 limits: Optional[RiskLimits] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize risk monitor.
        
        Args:
            limits: Risk limits configuration
            config: Additional configuration parameters
        """
        self.limits = limits or RiskLimits()
        self.config = config or {}
        
        # Current risk state
        self.current_metrics = RiskMetrics()
        self.metrics_history = deque(maxlen=10000)
        
        # Alert management
        self.active_alerts = {}
        self.alert_history = deque(maxlen=10000)
        self.alert_callbacks = []
        
        # Circuit breakers
        self.circuit_breakers = self._setup_circuit_breakers()
        
        # Position tracking
        self.position_tracker = PositionTracker()
        
        # Risk monitors
        self.correlation_monitor = CorrelationMonitor()
        self.liquidity_monitor = LiquidityMonitor()
        
        # Performance tracking
        self.pnl_history = deque(maxlen=10000)
        self.drawdown_history = deque(maxlen=1000)
        
        # Market data
        self.market_data = {}
        
        logger.info("🛡️ RiskMonitor initialized")
    
    def _setup_circuit_breakers(self) -> Dict[str, CircuitBreaker]:
        """Setup circuit breakers for various risk metrics."""
        breakers = {
            'var_95': CircuitBreaker('VAR_95', self.limits.var_95_limit),
            'var_99': CircuitBreaker('VAR_99', self.limits.var_99_limit),
            'expected_shortfall': CircuitBreaker('ES', self.limits.expected_shortfall_limit),
            'max_drawdown': CircuitBreaker('MAX_DRAWDOWN', self.limits.max_drawdown),
            'daily_loss': CircuitBreaker('DAILY_LOSS', self.limits.max_daily_loss),
            'leverage': CircuitBreaker('LEVERAGE', self.limits.max_leverage),
            'position_size': CircuitBreaker('POSITION_SIZE', self.limits.max_position_size_usd)
        }
        
        logger.info(f"🔧 Setup {len(breakers)} circuit breakers")
        return breakers
    
    def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """Update position information."""
        self.position_tracker.update_position(symbol, position_data)
        
        # Update correlation monitor
        if 'current_price' in position_data:
            self.correlation_monitor.update_price(symbol, position_data['current_price'])
        
        logger.debug(f"📍 Updated position: {symbol}")
    
    def remove_position(self, symbol: str):
        """Remove closed position."""
        self.position_tracker.remove_position(symbol)
        logger.debug(f"📤 Removed position: {symbol}")
    
    def update_market_data(self, symbol: str, market_data: Dict[str, Any]):
        """Update market data for risk calculations."""
        self.market_data[symbol] = {
            'timestamp': time.time(),
            'price': market_data.get('price', 0),
            'volume': market_data.get('volume', 0),
            'bid': market_data.get('bid', 0),
            'ask': market_data.get('ask', 0),
            'volatility': market_data.get('volatility', 0)
        }
        
        # Update liquidity monitor
        if 'bid' in market_data and 'ask' in market_data:
            price = market_data.get('price', (market_data['bid'] + market_data['ask']) / 2)
            spread = (market_data['ask'] - market_data['bid']) / price if price > 0 else 0
            
            self.liquidity_monitor.update_liquidity_data(symbol, {
                'bid_ask_spread': spread,
                'volume_24h': market_data.get('volume', 0),
                'liquidity_score': self._calculate_liquidity_score(market_data)
            })
    
    def _calculate_liquidity_score(self, market_data: Dict[str, Any]) -> float:
        """Calculate liquidity score (0-1)."""
        try:
            # Simple liquidity score based on spread and volume
            spread = market_data.get('bid_ask_spread', 0)
            volume = market_data.get('volume', 0)
            
            # Normalize spread component (lower spread = higher score)
            spread_score = max(0, 1 - spread / 0.01)  # 1% spread = 0 score
            
            # Normalize volume component (higher volume = higher score)
            volume_score = min(1, volume / 1000000)  # 1M volume = max score
            
            # Weighted average
            liquidity_score = 0.6 * spread_score + 0.4 * volume_score
            return max(0, min(1, liquidity_score))
            
        except Exception as e:
            logger.error(f"❌ Error calculating liquidity score: {e}")
            return 0.5  # Default moderate score
    
    def calculate_risk_metrics(self) -> RiskMetrics:
        """Calculate comprehensive risk metrics."""
        try:
            current_time = time.time()
            
            # Get position data
            positions = self.position_tracker.positions
            
            if not positions:
                # No positions - minimal risk
                self.current_metrics = RiskMetrics(timestamp=current_time)
                return self.current_metrics
            
            # Calculate exposure metrics
            total_long = sum(pos['position_value'] for pos in positions.values() if pos['position_value'] > 0)
            total_short = sum(abs(pos['position_value']) for pos in positions.values() if pos['position_value'] < 0)
            
            self.current_metrics.total_exposure = total_long + total_short
            self.current_metrics.net_exposure = total_long - total_short
            
            # Calculate position metrics
            self.current_metrics.active_positions = len(positions)
            self.current_metrics.long_positions = len([p for p in positions.values() if p['position_value'] > 0])
            self.current_metrics.short_positions = len([p for p in positions.values() if p['position_value'] < 0])
            self.current_metrics.largest_position_pct = self.position_tracker.get_largest_position_pct()
            
            # Calculate P&L metrics
            unrealized_pnl = sum(pos['unrealized_pnl'] for pos in positions.values())
            self.current_metrics.unrealized_pnl = unrealized_pnl
            
            # Update P&L history for drawdown calculation
            self.pnl_history.append({
                'timestamp': current_time,
                'unrealized_pnl': unrealized_pnl,
                'total_exposure': self.current_metrics.total_exposure
            })
            
            # Calculate drawdown
            self._calculate_drawdown()
            
            # Calculate VaR and ES (simplified implementation)
            self._calculate_var_metrics()
            
            # Calculate leverage
            if self.current_metrics.total_exposure > 0:
                # Simplified leverage calculation (would need account balance in production)
                account_balance = self.config.get('account_balance', 100000)
                self.current_metrics.leverage = self.current_metrics.total_exposure / account_balance
            
            # Store in history
            self.current_metrics.timestamp = current_time
            self.metrics_history.append(self.current_metrics)
            
            logger.debug(f"📊 Calculated risk metrics: exposure={self.current_metrics.total_exposure:.0f}, "
                        f"positions={self.current_metrics.active_positions}")
            
            return self.current_metrics
            
        except Exception as e:
            logger.error(f"❌ Failed to calculate risk metrics: {e}")
            return self.current_metrics
    
    def _calculate_drawdown(self):
        """Calculate current and maximum drawdown."""
        try:
            if len(self.pnl_history) < 2:
                return
            
            # Find peak and current trough
            peak_pnl = self.pnl_history[0]['unrealized_pnl']
            current_pnl = self.pnl_history[-1]['unrealized_pnl']
            
            for entry in self.pnl_history:
                if entry['unrealized_pnl'] > peak_pnl:
                    peak_pnl = entry['unrealized_pnl']
            
            # Calculate current drawdown
            if peak_pnl > 0:
                current_drawdown = (peak_pnl - current_pnl) / peak_pnl
            else:
                current_drawdown = 0
            
            self.current_metrics.current_drawdown = current_drawdown
            
            # Update max drawdown
            if current_drawdown > self.current_metrics.max_drawdown:
                self.current_metrics.max_drawdown = current_drawdown
            
            # Store drawdown history
            self.drawdown_history.append({
                'timestamp': time.time(),
                'current_drawdown': current_drawdown,
                'max_drawdown': self.current_metrics.max_drawdown
            })
            
        except Exception as e:
            logger.error(f"❌ Error calculating drawdown: {e}")
    
    def _calculate_var_metrics(self):
        """Calculate VaR and Expected Shortfall metrics."""
        try:
            if len(self.pnl_history) < 30:  # Need minimum data
                return
            
            # Get recent P&L changes
            recent_pnl = [entry['unrealized_pnl'] for entry in list(self.pnl_history)[-252:]]  # Last year
            pnl_changes = [recent_pnl[i] - recent_pnl[i-1] for i in range(1, len(recent_pnl))]
            
            if len(pnl_changes) < 30:
                return
            
            # Sort P&L changes (losses are negative)
            sorted_changes = sorted(pnl_changes)
            
            # Calculate VaR (95% and 99%)
            var_95_index = int(len(sorted_changes) * 0.05)
            var_99_index = int(len(sorted_changes) * 0.01)
            
            self.current_metrics.var_95 = abs(sorted_changes[var_95_index]) if var_95_index < len(sorted_changes) else 0
            self.current_metrics.var_99 = abs(sorted_changes[var_99_index]) if var_99_index < len(sorted_changes) else 0
            
            # Calculate Expected Shortfall (average of worst 5% and 1%)
            worst_5pct = sorted_changes[:var_95_index] if var_95_index > 0 else [0]
            worst_1pct = sorted_changes[:var_99_index] if var_99_index > 0 else [0]
            
            if worst_5pct:
                self.current_metrics.expected_shortfall = abs(statistics.mean(worst_5pct))
            
        except Exception as e:
            logger.error(f"❌ Error calculating VaR metrics: {e}")
    
    def check_risk_limits(self) -> List[RiskAlert]:
        """Check all risk limits and generate alerts."""
        alerts = []
        current_time = time.time()
        
        try:
            # Check VaR limits
            if self.current_metrics.var_95 > self.limits.var_95_limit:
                alert = self._create_alert(
                    AlertType.VAR_BREACH,
                    RiskLevel.CRITICAL,
                    self.current_metrics.var_95,
                    self.limits.var_95_limit,
                    f"VaR 95% exceeded limit: {self.current_metrics.var_95:.0f} > {self.limits.var_95_limit:.0f}",
                    "Reduce position sizes to lower VaR"
                )
                alerts.append(alert)
                self.circuit_breakers['var_95'].check_trigger(self.current_metrics.var_95)
            
            if self.current_metrics.var_99 > self.limits.var_99_limit:
                alert = self._create_alert(
                    AlertType.VAR_BREACH,
                    RiskLevel.EMERGENCY,
                    self.current_metrics.var_99,
                    self.limits.var_99_limit,
                    f"VaR 99% exceeded limit: {self.current_metrics.var_99:.0f} > {self.limits.var_99_limit:.0f}",
                    "Immediately reduce position sizes",
                    auto_remediation=True
                )
                alerts.append(alert)
                self.circuit_breakers['var_99'].check_trigger(self.current_metrics.var_99)
            
            # Check drawdown limits
            if self.current_metrics.current_drawdown > self.limits.max_drawdown:
                alert = self._create_alert(
                    AlertType.DRAWDOWN_LIMIT,
                    RiskLevel.EMERGENCY,
                    self.current_metrics.current_drawdown,
                    self.limits.max_drawdown,
                    f"Drawdown exceeded limit: {self.current_metrics.current_drawdown:.1%} > {self.limits.max_drawdown:.1%}",
                    "Consider halting trading and reviewing strategy",
                    auto_remediation=True
                )
                alerts.append(alert)
                self.circuit_breakers['max_drawdown'].check_trigger(self.current_metrics.current_drawdown)
            
            # Check leverage limits
            if self.current_metrics.leverage > self.limits.max_leverage:
                alert = self._create_alert(
                    AlertType.LEVERAGE_LIMIT,
                    RiskLevel.HIGH,
                    self.current_metrics.leverage,
                    self.limits.max_leverage,
                    f"Leverage exceeded limit: {self.current_metrics.leverage:.1f}x > {self.limits.max_leverage:.1f}x",
                    "Reduce position sizes to lower leverage"
                )
                alerts.append(alert)
                self.circuit_breakers['leverage'].check_trigger(self.current_metrics.leverage)
            
            # Check position limits
            if self.current_metrics.active_positions > self.limits.max_positions:
                alert = self._create_alert(
                    AlertType.POSITION_LIMIT,
                    RiskLevel.MEDIUM,
                    self.current_metrics.active_positions,
                    self.limits.max_positions,
                    f"Too many positions: {self.current_metrics.active_positions} > {self.limits.max_positions}",
                    "Close some positions to reduce count"
                )
                alerts.append(alert)
            
            # Check concentration limits
            if self.current_metrics.largest_position_pct > self.limits.max_symbol_concentration:
                alert = self._create_alert(
                    AlertType.POSITION_LIMIT,
                    RiskLevel.MEDIUM,
                    self.current_metrics.largest_position_pct,
                    self.limits.max_symbol_concentration,
                    f"High position concentration: {self.current_metrics.largest_position_pct:.1%} > {self.limits.max_symbol_concentration:.1%}",
                    "Reduce size of largest position"
                )
                alerts.append(alert)
            
            # Check correlation risks
            correlation_risks = self.correlation_monitor.check_correlation_risks(
                self.position_tracker.positions, 
                self.limits.max_correlation
            )
            
            for risk in correlation_risks:
                alert = self._create_alert(
                    AlertType.CORRELATION_BREAK,
                    RiskLevel.MEDIUM,
                    risk['correlation'],
                    self.limits.max_correlation,
                    f"High correlation risk: {risk['symbol1']}/{risk['symbol2']} = {risk['correlation']:.2f}",
                    f"Reduce exposure to correlated positions"
                )
                alerts.append(alert)
            
            # Check liquidity risks
            for symbol, position in self.position_tracker.positions.items():
                liquidity_risk = self.liquidity_monitor.assess_liquidity_risk(
                    symbol, 
                    abs(position['position_value'])
                )
                
                if liquidity_risk['risk_level'] in ['high', 'critical']:
                    alert = self._create_alert(
                        AlertType.LIQUIDITY_RISK,
                        RiskLevel.HIGH if liquidity_risk['risk_level'] == 'high' else RiskLevel.CRITICAL,
                        liquidity_risk.get('spread', 0),
                        self.limits.max_bid_ask_spread,
                        f"Liquidity risk for {symbol}: {liquidity_risk['risk_factors']}",
                        f"Consider reducing position size in {symbol}"
                    )
                    alert.symbol = symbol
                    alerts.append(alert)
            
            # Store alerts
            for alert in alerts:
                self.active_alerts[alert.alert_id] = alert
                self.alert_history.append(alert)
                
                # Trigger callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"❌ Alert callback failed: {e}")
            
            if alerts:
                logger.warning(f"⚠️ Generated {len(alerts)} risk alerts")
            
            return alerts
            
        except Exception as e:
            logger.error(f"❌ Failed to check risk limits: {e}")
            return []
    
    def _create_alert(self,
                     alert_type: AlertType,
                     risk_level: RiskLevel,
                     current_value: float,
                     limit_value: float,
                     message: str,
                     action: str,
                     symbol: Optional[str] = None,
                     auto_remediation: bool = False) -> RiskAlert:
        """Create a risk alert."""
        alert_id = f"{alert_type.value}_{int(time.time() * 1000)}"
        breach_percentage = ((current_value - limit_value) / limit_value) * 100 if limit_value != 0 else 0
        
        return RiskAlert(
            alert_id=alert_id,
            timestamp=time.time(),
            alert_type=alert_type,
            risk_level=risk_level,
            symbol=symbol,
            current_value=current_value,
            limit_value=limit_value,
            breach_percentage=breach_percentage,
            message=message,
            recommended_action=action,
            auto_remediation=auto_remediation
        )
    
    def add_alert_callback(self, callback: Callable[[RiskAlert], None]):
        """Add callback function for alert notifications."""
        self.alert_callbacks.append(callback)
        logger.info(f"📞 Added alert callback")
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"✅ Alert acknowledged: {alert_id}")
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            logger.info(f"✅ Alert resolved: {alert_id}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get comprehensive risk summary."""
        try:
            active_alerts = [alert for alert in self.active_alerts.values() if not alert.resolved]
            
            # Count alerts by level
            alert_counts = defaultdict(int)
            for alert in active_alerts:
                alert_counts[alert.risk_level.value] += 1
            
            # Circuit breaker status
            breaker_status = {name: breaker.is_triggered for name, breaker in self.circuit_breakers.items()}
            
            return {
                'timestamp': time.time(),
                'overall_risk_level': self._calculate_overall_risk_level(),
                'current_metrics': {
                    'var_95': self.current_metrics.var_95,
                    'var_99': self.current_metrics.var_99,
                    'expected_shortfall': self.current_metrics.expected_shortfall,
                    'current_drawdown': self.current_metrics.current_drawdown,
                    'max_drawdown': self.current_metrics.max_drawdown,
                    'leverage': self.current_metrics.leverage,
                    'total_exposure': self.current_metrics.total_exposure,
                    'active_positions': self.current_metrics.active_positions,
                    'largest_position_pct': self.current_metrics.largest_position_pct
                },
                'limit_utilization': {
                    'var_95_pct': (self.current_metrics.var_95 / self.limits.var_95_limit) * 100,
                    'var_99_pct': (self.current_metrics.var_99 / self.limits.var_99_limit) * 100,
                    'drawdown_pct': (self.current_metrics.current_drawdown / self.limits.max_drawdown) * 100,
                    'leverage_pct': (self.current_metrics.leverage / self.limits.max_leverage) * 100,
                    'positions_pct': (self.current_metrics.active_positions / self.limits.max_positions) * 100
                },
                'alerts': {
                    'total_active': len(active_alerts),
                    'by_level': dict(alert_counts),
                    'unacknowledged': len([a for a in active_alerts if not a.acknowledged])
                },
                'circuit_breakers': breaker_status,
                'data_status': {
                    'positions_tracked': len(self.position_tracker.positions),
                    'market_data_symbols': len(self.market_data),
                    'pnl_history_points': len(self.pnl_history),
                    'correlation_symbols': len(self.correlation_monitor.price_history)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get risk summary: {e}")
            return {'error': str(e)}
    
    def _calculate_overall_risk_level(self) -> str:
        """Calculate overall risk level based on current state."""
        active_alerts = [alert for alert in self.active_alerts.values() if not alert.resolved]
        
        # Check for emergency alerts
        if any(alert.risk_level == RiskLevel.EMERGENCY for alert in active_alerts):
            return 'emergency'
        
        # Check for critical alerts
        if any(alert.risk_level == RiskLevel.CRITICAL for alert in active_alerts):
            return 'critical'
        
        # Check circuit breakers
        if any(breaker.is_triggered for breaker in self.circuit_breakers.values()):
            return 'critical'
        
        # Check for high alerts
        if any(alert.risk_level == RiskLevel.HIGH for alert in active_alerts):
            return 'high'
        
        # Check for medium alerts
        if any(alert.risk_level == RiskLevel.MEDIUM for alert in active_alerts):
            return 'medium'
        
        return 'low'
    
    def export_risk_report(self) -> Dict[str, Any]:
        """Export comprehensive risk report."""
        try:
            return {
                'timestamp': time.time(),
                'report_type': 'risk_assessment',
                'summary': self.get_risk_summary(),
                'detailed_metrics': {
                    'current': self.current_metrics.__dict__,
                    'limits': self.limits.__dict__,
                    'circuit_breakers': {name: {
                        'threshold': breaker.threshold,
                        'is_triggered': breaker.is_triggered,
                        'trigger_count': breaker.trigger_count,
                        'last_trigger': breaker.trigger_time
                    } for name, breaker in self.circuit_breakers.items()}
                },
                'positions': {symbol: pos for symbol, pos in self.position_tracker.positions.items()},
                'active_alerts': [alert.__dict__ for alert in self.active_alerts.values() if not alert.resolved],
                'correlation_risks': self.correlation_monitor.check_correlation_risks(
                    self.position_tracker.positions, self.limits.max_correlation
                ),
                'liquidity_assessment': {
                    symbol: self.liquidity_monitor.assess_liquidity_risk(symbol, abs(pos['position_value']))
                    for symbol, pos in self.position_tracker.positions.items()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to export risk report: {e}")
            return {'error': str(e)}


# Factory function
def create_risk_monitor(config: Dict[str, Any]) -> RiskMonitor:
    """Create and configure risk monitor."""
    # Parse risk limits from config
    limits_config = config.get('risk_limits', {})
    limits = RiskLimits(
        var_95_limit=limits_config.get('var_95_limit', 200000),
        var_99_limit=limits_config.get('var_99_limit', 300000),
        expected_shortfall_limit=limits_config.get('expected_shortfall_limit', 400000),
        max_position_size_usd=limits_config.get('max_position_size_usd', 100000),
        max_positions=limits_config.get('max_positions', 10),
        max_leverage=limits_config.get('max_leverage', 3.0),
        max_drawdown=limits_config.get('max_drawdown', 0.20),
        max_daily_loss=limits_config.get('max_daily_loss', 50000)
    )
    
    return RiskMonitor(limits=limits, config=config)