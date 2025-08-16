"""
DipMaster Enhanced V4 - Execution Risk Manager
Real-time risk management during order execution with circuit breakers and anomaly detection
"""

import asyncio
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)

@dataclass
class RiskMetrics:
    """Real-time risk metrics during execution"""
    current_exposure_usd: float
    max_exposure_usd: float
    cumulative_slippage_bps: float
    max_slippage_bps: float
    rejection_rate: float
    timeout_rate: float
    latency_p95_ms: float
    market_impact_bps: float
    
@dataclass
class RiskViolation:
    """Risk limit violation record"""
    timestamp: datetime
    violation_type: str
    symbol: str
    current_value: float
    limit_value: float
    severity: str  # 'warning', 'critical', 'emergency'
    action_taken: str
    
@dataclass
class CircuitBreaker:
    """Circuit breaker configuration"""
    metric_name: str
    threshold: float
    window_minutes: int
    min_occurrences: int
    cooldown_minutes: int
    action: str  # 'pause', 'reduce_size', 'cancel_all'
    is_triggered: bool = False
    triggered_at: Optional[datetime] = None

class ExecutionRiskManager:
    """Real-time execution risk management with circuit breakers"""
    
    def __init__(self):
        # Risk limits
        self.max_position_usd = 10000
        self.max_daily_loss_usd = 500
        self.max_cumulative_slippage_bps = 50
        self.max_rejection_rate = 0.05  # 5%
        self.max_latency_ms = 500
        self.max_order_rate_per_minute = 20
        
        # Circuit breakers
        self.circuit_breakers = {
            'high_slippage': CircuitBreaker(
                metric_name='slippage_bps',
                threshold=25.0,
                window_minutes=5,
                min_occurrences=3,
                cooldown_minutes=10,
                action='pause'
            ),
            'high_rejection_rate': CircuitBreaker(
                metric_name='rejection_rate',
                threshold=0.1,
                window_minutes=2,
                min_occurrences=5,
                cooldown_minutes=5,
                action='reduce_size'
            ),
            'high_latency': CircuitBreaker(
                metric_name='latency_ms',
                threshold=1000,
                window_minutes=3,
                min_occurrences=5,
                cooldown_minutes=5,
                action='pause'
            ),
            'market_stress': CircuitBreaker(
                metric_name='market_impact_bps',
                threshold=100.0,
                window_minutes=1,
                min_occurrences=2,
                cooldown_minutes=15,
                action='cancel_all'
            )
        }
        
        # State tracking
        self.current_positions = defaultdict(float)
        self.daily_pnl = 0.0
        self.execution_metrics = defaultdict(deque)
        self.violations_log: List[RiskViolation] = []
        self.order_times = deque()
        self.emergency_stop = False
        
        # Performance tracking
        self.slippage_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        self.rejection_history = deque(maxlen=100)
        
    async def pre_execution_check(self, symbol: str, side: str, qty: float, price: float) -> Tuple[bool, List[str]]:
        """Pre-execution risk checks before placing order"""
        
        violations = []
        
        # Position size check
        position_value = qty * price
        if position_value > self.max_position_usd:
            violations.append(f"Position size exceeds limit: ${position_value:.0f} > ${self.max_position_usd}")
        
        # Total exposure check
        current_exposure = sum(abs(pos) for pos in self.current_positions.values())
        if current_exposure + position_value > self.max_position_usd * 3:
            violations.append(f"Total exposure limit exceeded: ${current_exposure + position_value:.0f}")
        
        # Daily loss check
        if self.daily_pnl < -self.max_daily_loss_usd:
            violations.append(f"Daily loss limit exceeded: ${self.daily_pnl:.2f}")
        
        # Order rate limiting
        recent_orders = len([t for t in self.order_times if datetime.now() - t < timedelta(minutes=1)])
        if recent_orders >= self.max_order_rate_per_minute:
            violations.append(f"Order rate limit exceeded: {recent_orders} orders/minute")
        
        # Circuit breaker check
        for name, breaker in self.circuit_breakers.items():
            if breaker.is_triggered:
                cooldown_remaining = breaker.cooldown_minutes - (datetime.now() - breaker.triggered_at).total_seconds() / 60
                if cooldown_remaining > 0:
                    violations.append(f"Circuit breaker '{name}' active: {cooldown_remaining:.1f} min remaining")
        
        # Emergency stop check
        if self.emergency_stop:
            violations.append("Emergency stop activated - all trading halted")
        
        # Market hours check (simplified)
        current_hour = datetime.now().hour
        if current_hour < 6 or current_hour > 22:  # Outside active trading hours
            violations.append("Outside recommended trading hours")
        
        return len(violations) == 0, violations
    
    async def post_execution_update(self, 
                                  symbol: str, 
                                  side: str, 
                                  executed_qty: float, 
                                  executed_price: float,
                                  slippage_bps: float,
                                  latency_ms: float,
                                  status: str) -> List[RiskViolation]:
        """Update risk metrics after order execution"""
        
        violations = []
        
        # Update position
        position_change = executed_qty if side == 'BUY' else -executed_qty
        self.current_positions[symbol] += position_change
        
        # Track order timing
        self.order_times.append(datetime.now())
        
        # Update metrics
        self.slippage_history.append(slippage_bps)
        self.latency_history.append(latency_ms)
        self.rejection_history.append(1 if status == 'REJECTED' else 0)
        
        # Add to time-windowed metrics
        current_time = datetime.now()
        self.execution_metrics['slippage'].append((current_time, slippage_bps))
        self.execution_metrics['latency'].append((current_time, latency_ms))
        self.execution_metrics['rejection'].append((current_time, 1 if status == 'REJECTED' else 0))
        
        # Clean old metrics (keep only last hour)
        cutoff_time = current_time - timedelta(hours=1)
        for metric_name in self.execution_metrics:
            while (self.execution_metrics[metric_name] and 
                   self.execution_metrics[metric_name][0][0] < cutoff_time):
                self.execution_metrics[metric_name].popleft()
        
        # Check individual order violations
        if abs(slippage_bps) > self.max_cumulative_slippage_bps:
            violation = RiskViolation(
                timestamp=current_time,
                violation_type='EXCESSIVE_SLIPPAGE',
                symbol=symbol,
                current_value=slippage_bps,
                limit_value=self.max_cumulative_slippage_bps,
                severity='warning',
                action_taken='logged'
            )
            violations.append(violation)
            self.violations_log.append(violation)
        
        if latency_ms > self.max_latency_ms:
            violation = RiskViolation(
                timestamp=current_time,
                violation_type='HIGH_LATENCY',
                symbol=symbol,
                current_value=latency_ms,
                limit_value=self.max_latency_ms,
                severity='warning',
                action_taken='logged'
            )
            violations.append(violation)
            self.violations_log.append(violation)
        
        # Check circuit breakers
        await self._check_circuit_breakers()
        
        return violations
    
    async def _check_circuit_breakers(self):
        """Check and trigger circuit breakers based on current metrics"""
        
        current_time = datetime.now()
        
        for name, breaker in self.circuit_breakers.items():
            if breaker.is_triggered:
                # Check if cooldown period has passed
                if (current_time - breaker.triggered_at).total_seconds() / 60 >= breaker.cooldown_minutes:
                    breaker.is_triggered = False
                    breaker.triggered_at = None
                    logger.info(f"Circuit breaker '{name}' reset after cooldown")
                continue
            
            # Check if breaker should be triggered
            window_start = current_time - timedelta(minutes=breaker.window_minutes)
            
            if breaker.metric_name == 'slippage_bps':
                recent_values = [v for t, v in self.execution_metrics['slippage'] if t >= window_start]
                violations = len([v for v in recent_values if abs(v) > breaker.threshold])
                
            elif breaker.metric_name == 'rejection_rate':
                recent_rejections = [v for t, v in self.execution_metrics['rejection'] if t >= window_start]
                rejection_rate = np.mean(recent_rejections) if recent_rejections else 0
                violations = 1 if rejection_rate > breaker.threshold else 0
                
            elif breaker.metric_name == 'latency_ms':
                recent_values = [v for t, v in self.execution_metrics['latency'] if t >= window_start]
                violations = len([v for v in recent_values if v > breaker.threshold])
                
            elif breaker.metric_name == 'market_impact_bps':
                # Calculate market impact from recent slippage
                recent_slippage = [abs(v) for t, v in self.execution_metrics['slippage'] if t >= window_start]
                avg_impact = np.mean(recent_slippage) if recent_slippage else 0
                violations = 1 if avg_impact > breaker.threshold else 0
            
            else:
                violations = 0
            
            # Trigger breaker if threshold exceeded
            if violations >= breaker.min_occurrences:
                await self._trigger_circuit_breaker(name, breaker)
    
    async def _trigger_circuit_breaker(self, name: str, breaker: CircuitBreaker):
        """Trigger a circuit breaker and take appropriate action"""
        
        breaker.is_triggered = True
        breaker.triggered_at = datetime.now()
        
        violation = RiskViolation(
            timestamp=datetime.now(),
            violation_type='CIRCUIT_BREAKER',
            symbol='ALL',
            current_value=0,
            limit_value=breaker.threshold,
            severity='critical',
            action_taken=breaker.action
        )
        self.violations_log.append(violation)
        
        logger.critical(f"Circuit breaker '{name}' TRIGGERED - Action: {breaker.action}")
        
        # Take action based on breaker configuration
        if breaker.action == 'pause':
            await self._pause_trading(breaker.cooldown_minutes)
        elif breaker.action == 'reduce_size':
            await self._reduce_order_sizes()
        elif breaker.action == 'cancel_all':
            await self._emergency_stop_all()
    
    async def _pause_trading(self, duration_minutes: int):
        """Pause all trading for specified duration"""
        logger.warning(f"Trading paused for {duration_minutes} minutes")
        # Implementation would set flags to prevent new orders
    
    async def _reduce_order_sizes(self):
        """Reduce all future order sizes by 50%"""
        logger.warning("Reducing order sizes by 50% due to risk concerns")
        # Implementation would adjust order sizing parameters
    
    async def _emergency_stop_all(self):
        """Emergency stop all trading activities"""
        self.emergency_stop = True
        logger.critical("EMERGENCY STOP ACTIVATED - All trading halted")
        # Implementation would cancel all pending orders
    
    async def get_current_risk_metrics(self) -> RiskMetrics:
        """Get current real-time risk metrics"""
        
        current_exposure = sum(abs(pos) for pos in self.current_positions.values())
        
        # Calculate metrics from recent history
        recent_slippage = list(self.slippage_history)[-20:] if self.slippage_history else [0]
        recent_latency = list(self.latency_history)[-20:] if self.latency_history else [0]
        recent_rejections = list(self.rejection_history)[-20:] if self.rejection_history else [0]
        
        cumulative_slippage = np.sum(np.abs(recent_slippage))
        rejection_rate = np.mean(recent_rejections)
        latency_p95 = np.percentile(recent_latency, 95) if recent_latency else 0
        
        return RiskMetrics(
            current_exposure_usd=current_exposure,
            max_exposure_usd=self.max_position_usd * 3,
            cumulative_slippage_bps=cumulative_slippage,
            max_slippage_bps=self.max_cumulative_slippage_bps,
            rejection_rate=rejection_rate,
            timeout_rate=0.0,  # Not implemented
            latency_p95_ms=latency_p95,
            market_impact_bps=np.mean(np.abs(recent_slippage)) if recent_slippage else 0
        )
    
    async def generate_risk_report(self) -> Dict:
        """Generate comprehensive risk report"""
        
        current_metrics = await self.get_current_risk_metrics()
        
        # Active violations
        active_violations = [v for v in self.violations_log 
                           if (datetime.now() - v.timestamp).total_seconds() < 3600]  # Last hour
        
        # Circuit breaker status
        breaker_status = {}
        for name, breaker in self.circuit_breakers.items():
            breaker_status[name] = {
                'is_triggered': breaker.is_triggered,
                'triggered_at': breaker.triggered_at.isoformat() if breaker.triggered_at else None,
                'threshold': breaker.threshold,
                'action': breaker.action
            }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': {
                'exposure_usd': current_metrics.current_exposure_usd,
                'exposure_utilization': current_metrics.current_exposure_usd / current_metrics.max_exposure_usd,
                'slippage_bps': current_metrics.cumulative_slippage_bps,
                'rejection_rate': current_metrics.rejection_rate,
                'latency_p95_ms': current_metrics.latency_p95_ms,
                'market_impact_bps': current_metrics.market_impact_bps
            },
            'positions': dict(self.current_positions),
            'daily_pnl': self.daily_pnl,
            'emergency_stop': self.emergency_stop,
            'active_violations': len(active_violations),
            'circuit_breakers': breaker_status,
            'recent_violations': [
                {
                    'timestamp': v.timestamp.isoformat(),
                    'type': v.violation_type,
                    'symbol': v.symbol,
                    'severity': v.severity,
                    'current_value': v.current_value,
                    'limit_value': v.limit_value
                }
                for v in self.violations_log[-10:]  # Last 10 violations
            ]
        }
        
        return report
    
    async def reset_daily_metrics(self):
        """Reset daily metrics at start of new trading day"""
        self.daily_pnl = 0.0
        self.violations_log.clear()
        logger.info("Daily risk metrics reset")
    
    async def force_reset_emergency_stop(self, reason: str):
        """Force reset emergency stop (admin function)"""
        self.emergency_stop = False
        
        violation = RiskViolation(
            timestamp=datetime.now(),
            violation_type='EMERGENCY_RESET',
            symbol='ALL',
            current_value=0,
            limit_value=0,
            severity='warning',
            action_taken=f'Manual reset: {reason}'
        )
        self.violations_log.append(violation)
        
        logger.warning(f"Emergency stop manually reset: {reason}")

class PositionRiskMonitor:
    """Monitor position-level risks in real-time"""
    
    def __init__(self, risk_manager: ExecutionRiskManager):
        self.risk_manager = risk_manager
        self.position_limits = {
            'BTCUSDT': 3000,
            'ETHUSDT': 2500,
            'SOLUSDT': 2000
        }
        self.correlation_matrix = {
            ('BTCUSDT', 'ETHUSDT'): 0.8,
            ('BTCUSDT', 'SOLUSDT'): 0.7,
            ('ETHUSDT', 'SOLUSDT'): 0.75
        }
    
    async def check_position_risk(self, symbol: str, new_position_usd: float) -> Tuple[bool, List[str]]:
        """Check position-specific risk before execution"""
        
        violations = []
        
        # Single position limit
        if symbol in self.position_limits:
            if abs(new_position_usd) > self.position_limits[symbol]:
                violations.append(f"{symbol} position limit exceeded: ${abs(new_position_usd):.0f} > ${self.position_limits[symbol]}")
        
        # Correlation risk
        current_positions = self.risk_manager.current_positions
        for other_symbol, other_size in current_positions.items():
            if other_symbol != symbol and other_size != 0:
                correlation_key = tuple(sorted([symbol, other_symbol]))
                correlation = self.correlation_matrix.get(correlation_key, 0.5)
                
                if correlation > 0.7:  # High correlation
                    combined_risk = abs(new_position_usd) + abs(other_size)
                    if combined_risk > 4000:  # Combined limit for correlated positions
                        violations.append(f"High correlation risk: {symbol} + {other_symbol} = ${combined_risk:.0f}")
        
        return len(violations) == 0, violations

async def main():
    """Test execution risk manager"""
    
    risk_manager = ExecutionRiskManager()
    
    # Simulate some executions
    print("Testing execution risk management...")
    
    # Test 1: Normal execution
    can_execute, violations = await risk_manager.pre_execution_check('BTCUSDT', 'BUY', 0.1, 50000)
    print(f"Pre-execution check: {can_execute}, violations: {violations}")
    
    if can_execute:
        violations = await risk_manager.post_execution_update(
            'BTCUSDT', 'BUY', 0.1, 50100, 2.0, 100, 'FILLED'
        )
        print(f"Post-execution violations: {len(violations)}")
    
    # Test 2: High slippage execution
    for i in range(5):
        await risk_manager.post_execution_update(
            'ETHUSDT', 'BUY', 0.5, 3000, 30.0, 200, 'FILLED'
        )
    
    # Generate risk report
    report = await risk_manager.generate_risk_report()
    print(f"\nRisk Report:")
    print(f"Exposure: ${report['current_metrics']['exposure_usd']:.0f}")
    print(f"Active violations: {report['active_violations']}")
    print(f"Circuit breakers triggered: {sum(1 for cb in report['circuit_breakers'].values() if cb['is_triggered'])}")

if __name__ == "__main__":
    asyncio.run(main())