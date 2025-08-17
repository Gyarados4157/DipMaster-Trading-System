#!/usr/bin/env python3
"""
Dashboard Data Generator for DipMaster Trading System
实时监控仪表板数据生成器 - 专业级交易系统可视化数据

Features:
- Real-time PnL curve generation
- Position and portfolio monitoring
- Risk metrics calculation and trending
- Trading performance analytics
- System health and status monitoring
- WebSocket streaming for live updates
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import threading

from .metrics_collector import MetricsCollector
from .business_kpi import BusinessKPITracker
from .enhanced_event_producer import EnhancedEventProducer

logger = logging.getLogger(__name__)


class ChartTimeframe(Enum):
    """Chart timeframe options."""
    REALTIME = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"


@dataclass
class PnLDataPoint:
    """PnL data point for charting."""
    timestamp: float
    cumulative_pnl: float
    period_pnl: float
    unrealized_pnl: float
    realized_pnl: float
    trade_count: int
    win_count: int
    loss_count: int


@dataclass
class PositionSummary:
    """Position summary for dashboard."""
    symbol: str
    side: str
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    holding_time_minutes: int
    risk_score: float
    strategy: str
    status: str


@dataclass
class RiskMetrics:
    """Risk metrics for dashboard."""
    portfolio_value: float
    total_exposure: float
    leverage: float
    var_95: float
    var_99: float
    expected_shortfall: float
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float


@dataclass
class TradingPerformance:
    """Trading performance metrics."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_holding_time: float
    best_trade: float
    worst_trade: float
    avg_daily_pnl: float
    volatility: float
    calmar_ratio: float


@dataclass
class SystemHealth:
    """System health metrics."""
    overall_score: float
    components: Dict[str, Dict[str, Any]]
    uptime_seconds: float
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    active_connections: int
    error_rate: float
    last_update: float


class PnLCurveGenerator:
    """Generates real-time PnL curves for dashboard."""
    
    def __init__(self, max_points: int = 1000):
        self.max_points = max_points
        self.data_points = deque(maxlen=max_points)
        self.timeframe_caches = {tf: deque(maxlen=max_points) for tf in ChartTimeframe}
        self.last_aggregation = {tf: 0 for tf in ChartTimeframe}
        self.cumulative_pnl = 0.0
        self.lock = threading.Lock()
    
    def add_trade_result(self,
                        timestamp: float,
                        realized_pnl: float,
                        unrealized_pnl: float = 0.0):
        """Add trade result to PnL curve."""
        with self.lock:
            self.cumulative_pnl += realized_pnl
            
            data_point = PnLDataPoint(
                timestamp=timestamp,
                cumulative_pnl=self.cumulative_pnl,
                period_pnl=realized_pnl,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                trade_count=1 if realized_pnl != 0 else 0,
                win_count=1 if realized_pnl > 0 else 0,
                loss_count=1 if realized_pnl < 0 else 0
            )
            
            self.data_points.append(data_point)
            self._update_timeframe_caches(data_point)
    
    def update_unrealized_pnl(self, timestamp: float, unrealized_pnl: float):
        """Update current unrealized PnL."""
        with self.lock:
            if self.data_points:
                # Update the most recent point
                last_point = self.data_points[-1]
                updated_point = PnLDataPoint(
                    timestamp=timestamp,
                    cumulative_pnl=last_point.cumulative_pnl + unrealized_pnl,
                    period_pnl=last_point.period_pnl,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=last_point.realized_pnl,
                    trade_count=last_point.trade_count,
                    win_count=last_point.win_count,
                    loss_count=last_point.loss_count
                )
                self.data_points[-1] = updated_point
            else:
                # First data point
                data_point = PnLDataPoint(
                    timestamp=timestamp,
                    cumulative_pnl=unrealized_pnl,
                    period_pnl=0.0,
                    unrealized_pnl=unrealized_pnl,
                    realized_pnl=0.0,
                    trade_count=0,
                    win_count=0,
                    loss_count=0
                )
                self.data_points.append(data_point)
    
    def _update_timeframe_caches(self, data_point: PnLDataPoint):
        """Update timeframe-specific caches."""
        current_time = data_point.timestamp
        
        for timeframe in ChartTimeframe:
            interval_seconds = self._get_interval_seconds(timeframe)
            last_agg = self.last_aggregation[timeframe]
            
            if current_time - last_agg >= interval_seconds:
                # Aggregate data for this timeframe
                aggregated = self._aggregate_timeframe_data(timeframe, current_time)
                if aggregated:
                    self.timeframe_caches[timeframe].append(aggregated)
                    self.last_aggregation[timeframe] = current_time
    
    def _get_interval_seconds(self, timeframe: ChartTimeframe) -> int:
        """Get interval in seconds for timeframe."""
        intervals = {
            ChartTimeframe.REALTIME: 60,      # 1 minute
            ChartTimeframe.MINUTE_5: 300,     # 5 minutes
            ChartTimeframe.MINUTE_15: 900,    # 15 minutes
            ChartTimeframe.HOUR_1: 3600,      # 1 hour
            ChartTimeframe.HOUR_4: 14400,     # 4 hours
            ChartTimeframe.DAY_1: 86400       # 1 day
        }
        return intervals[timeframe]
    
    def _aggregate_timeframe_data(self,
                                timeframe: ChartTimeframe,
                                end_time: float) -> Optional[PnLDataPoint]:
        """Aggregate data points for timeframe."""
        interval = self._get_interval_seconds(timeframe)
        start_time = end_time - interval
        
        # Find data points in time window
        window_points = [
            dp for dp in self.data_points
            if start_time <= dp.timestamp <= end_time
        ]
        
        if not window_points:
            return None
        
        # Aggregate metrics
        total_pnl = sum(dp.period_pnl for dp in window_points)
        total_trades = sum(dp.trade_count for dp in window_points)
        total_wins = sum(dp.win_count for dp in window_points)
        total_losses = sum(dp.loss_count for dp in window_points)
        
        # Use final values for cumulative and unrealized
        last_point = window_points[-1]
        
        return PnLDataPoint(
            timestamp=end_time,
            cumulative_pnl=last_point.cumulative_pnl,
            period_pnl=total_pnl,
            unrealized_pnl=last_point.unrealized_pnl,
            realized_pnl=sum(dp.realized_pnl for dp in window_points),
            trade_count=total_trades,
            win_count=total_wins,
            loss_count=total_losses
        )
    
    def get_chart_data(self,
                      timeframe: ChartTimeframe = ChartTimeframe.REALTIME,
                      points: int = 100) -> List[Dict[str, Any]]:
        """Get chart data for specific timeframe."""
        with self.lock:
            if timeframe == ChartTimeframe.REALTIME:
                data_source = list(self.data_points)
            else:
                data_source = list(self.timeframe_caches[timeframe])
            
            # Limit number of points
            if len(data_source) > points:
                step = len(data_source) // points
                data_source = data_source[::step]
            
            return [
                {
                    'timestamp': int(dp.timestamp * 1000),  # JavaScript timestamp
                    'cumulative_pnl': round(dp.cumulative_pnl, 2),
                    'period_pnl': round(dp.period_pnl, 2),
                    'unrealized_pnl': round(dp.unrealized_pnl, 2),
                    'trade_count': dp.trade_count,
                    'win_count': dp.win_count,
                    'loss_count': dp.loss_count
                }
                for dp in data_source
            ]


class DashboardDataGenerator:
    """
    Comprehensive dashboard data generator for trading system monitoring.
    
    Provides real-time data for trading performance visualization,
    risk monitoring, and system health tracking.
    """
    
    def __init__(self,
                 metrics_collector: MetricsCollector,
                 business_kpi: BusinessKPITracker,
                 event_producer: Optional[EnhancedEventProducer] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize dashboard data generator.
        
        Args:
            metrics_collector: Metrics collection system
            business_kpi: Business KPI tracker
            event_producer: Event producer for updates
            config: Configuration parameters
        """
        self.metrics_collector = metrics_collector
        self.business_kpi = business_kpi
        self.event_producer = event_producer
        self.config = config or {}
        
        # PnL curve generator
        self.pnl_generator = PnLCurveGenerator(
            max_points=self.config.get('max_chart_points', 1000)
        )
        
        # Cache for dashboard data
        self.dashboard_cache = {}
        self.cache_ttl = self.config.get('cache_ttl', 5)  # 5 seconds
        self.last_cache_update = 0
        
        # Active positions tracking
        self.active_positions = {}
        self.position_lock = threading.Lock()
        
        # Performance tracking
        self.performance_window = deque(maxlen=1000)
        self.daily_stats = defaultdict(list)
        
        # WebSocket subscribers
        self.subscribers = set()
        self.update_interval = self.config.get('update_interval', 1)  # 1 second
        self.update_task = None
        self.is_running = False
        
        logger.info("📊 DashboardDataGenerator initialized")
    
    async def start_updates(self):
        """Start real-time dashboard updates."""
        if self.is_running:
            logger.warning("⚠️ Dashboard updates already running")
            return
        
        self.is_running = True
        self.update_task = asyncio.create_task(self._update_loop())
        logger.info("✅ Started dashboard updates")
    
    async def stop_updates(self):
        """Stop real-time dashboard updates."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("🛑 Stopped dashboard updates")
    
    async def _update_loop(self):
        """Main update loop for real-time data."""
        while self.is_running:
            try:
                # Update dashboard data
                await self._update_dashboard_data()
                
                # Notify subscribers
                if self.subscribers:
                    dashboard_data = await self.get_dashboard_data()
                    await self._notify_subscribers(dashboard_data)
                
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in dashboard update loop: {e}")
                await asyncio.sleep(1)  # Error backoff
    
    async def _update_dashboard_data(self):
        """Update internal dashboard data."""
        current_time = time.time()
        
        # Update PnL with current positions
        total_unrealized = 0.0
        with self.position_lock:
            for position in self.active_positions.values():
                total_unrealized += position.unrealized_pnl
        
        self.pnl_generator.update_unrealized_pnl(current_time, total_unrealized)
        
        # Update performance metrics
        trading_summary = self.business_kpi.get_trading_summary(24)  # Last 24 hours
        if trading_summary:
            self.performance_window.append({
                'timestamp': current_time,
                'win_rate': trading_summary.get('win_rate', 0),
                'total_pnl': trading_summary.get('total_pnl', 0),
                'sharpe_ratio': trading_summary.get('sharpe_ratio', 0)
            })
    
    async def _notify_subscribers(self, data: Dict[str, Any]):
        """Notify WebSocket subscribers of updates."""
        if not self.subscribers:
            return
        
        update_message = {
            'type': 'dashboard_update',
            'timestamp': time.time(),
            'data': data
        }
        
        # Send to all subscribers (would be WebSocket connections in practice)
        disconnected = set()
        for subscriber in self.subscribers:
            try:
                # In real implementation, this would be WebSocket send
                await subscriber.send(json.dumps(update_message))
            except Exception as e:
                logger.warning(f"Failed to notify subscriber: {e}")
                disconnected.add(subscriber)
        
        # Remove disconnected subscribers
        self.subscribers -= disconnected
    
    def add_subscriber(self, subscriber):
        """Add WebSocket subscriber."""
        self.subscribers.add(subscriber)
        logger.debug(f"Added dashboard subscriber (total: {len(self.subscribers)})")
    
    def remove_subscriber(self, subscriber):
        """Remove WebSocket subscriber."""
        self.subscribers.discard(subscriber)
        logger.debug(f"Removed dashboard subscriber (total: {len(self.subscribers)})")
    
    async def record_trade_entry(self,
                               position_id: str,
                               symbol: str,
                               side: str,
                               quantity: float,
                               entry_price: float,
                               strategy: str,
                               **kwargs):
        """Record trade entry for dashboard tracking."""
        position = PositionSummary(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            current_price=entry_price,  # Initial
            unrealized_pnl=0.0,
            unrealized_pnl_pct=0.0,
            holding_time_minutes=0,
            risk_score=kwargs.get('risk_score', 0.0),
            strategy=strategy,
            status="open"
        )
        
        with self.position_lock:
            self.active_positions[position_id] = position
        
        logger.debug(f"📊 Recorded trade entry: {position_id}")
    
    async def record_trade_exit(self,
                              position_id: str,
                              exit_price: float,
                              realized_pnl: float,
                              holding_minutes: int,
                              **kwargs):
        """Record trade exit for dashboard tracking."""
        with self.position_lock:
            if position_id in self.active_positions:
                # Update position status
                position = self.active_positions[position_id]
                position.status = "closed"
                position.current_price = exit_price
                position.holding_time_minutes = holding_minutes
                
                # Remove from active positions
                del self.active_positions[position_id]
        
        # Update PnL curve
        self.pnl_generator.add_trade_result(
            timestamp=time.time(),
            realized_pnl=realized_pnl
        )
        
        logger.debug(f"📊 Recorded trade exit: {position_id}, PnL: {realized_pnl:.2f}")
    
    async def update_position_price(self,
                                  position_id: str,
                                  current_price: float):
        """Update position with current market price."""
        with self.position_lock:
            if position_id in self.active_positions:
                position = self.active_positions[position_id]
                position.current_price = current_price
                
                # Calculate unrealized PnL
                price_diff = current_price - position.entry_price
                if position.side == "SELL":
                    price_diff = -price_diff
                
                position.unrealized_pnl = price_diff * position.quantity
                position.unrealized_pnl_pct = (price_diff / position.entry_price) * 100
                
                # Update holding time
                position.holding_time_minutes = int((time.time() - time.time()) / 60)  # Simplified
    
    async def get_dashboard_data(self,
                               use_cache: bool = True) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        current_time = time.time()
        
        # Check cache
        if (use_cache and 
            self.dashboard_cache and 
            current_time - self.last_cache_update < self.cache_ttl):
            return self.dashboard_cache
        
        try:
            # Generate fresh dashboard data
            dashboard_data = {
                'timestamp': current_time,
                'summary': await self._get_summary_data(),
                'pnl_chart': await self._get_pnl_chart_data(),
                'positions': await self._get_positions_data(),
                'performance': await self._get_performance_data(),
                'risk_metrics': await self._get_risk_metrics_data(),
                'system_health': await self._get_system_health_data(),
                'trading_activity': await self._get_trading_activity_data()
            }
            
            # Update cache
            self.dashboard_cache = dashboard_data
            self.last_cache_update = current_time
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"❌ Failed to generate dashboard data: {e}")
            return {
                'timestamp': current_time,
                'error': str(e),
                'status': 'error'
            }
    
    async def _get_summary_data(self) -> Dict[str, Any]:
        """Get summary metrics for dashboard."""
        trading_summary = self.business_kpi.get_trading_summary(24)
        
        with self.position_lock:
            active_count = len(self.active_positions)
            total_unrealized = sum(pos.unrealized_pnl for pos in self.active_positions.values())
        
        return {
            'total_pnl': trading_summary.get('total_pnl', 0.0),
            'unrealized_pnl': total_unrealized,
            'active_positions': active_count,
            'trades_today': trading_summary.get('total_trades', 0),
            'win_rate': trading_summary.get('win_rate', 0.0),
            'profit_factor': trading_summary.get('profit_factor', 0.0),
            'sharpe_ratio': trading_summary.get('sharpe_ratio', 0.0),
            'max_drawdown': trading_summary.get('max_drawdown', 0.0)
        }
    
    async def _get_pnl_chart_data(self) -> Dict[str, Any]:
        """Get PnL chart data for different timeframes."""
        return {
            'realtime': self.pnl_generator.get_chart_data(ChartTimeframe.REALTIME, 100),
            '5m': self.pnl_generator.get_chart_data(ChartTimeframe.MINUTE_5, 100),
            '15m': self.pnl_generator.get_chart_data(ChartTimeframe.MINUTE_15, 100),
            '1h': self.pnl_generator.get_chart_data(ChartTimeframe.HOUR_1, 100),
            '4h': self.pnl_generator.get_chart_data(ChartTimeframe.HOUR_4, 100),
            '1d': self.pnl_generator.get_chart_data(ChartTimeframe.DAY_1, 100)
        }
    
    async def _get_positions_data(self) -> List[Dict[str, Any]]:
        """Get current positions data."""
        with self.position_lock:
            return [asdict(position) for position in self.active_positions.values()]
    
    async def _get_performance_data(self) -> Dict[str, Any]:
        """Get trading performance data."""
        if not self.performance_window:
            return {}
        
        recent_performance = list(self.performance_window)[-100:]  # Last 100 data points
        
        win_rates = [p['win_rate'] for p in recent_performance]
        pnls = [p['total_pnl'] for p in recent_performance]
        sharpe_ratios = [p['sharpe_ratio'] for p in recent_performance]
        
        return {
            'win_rate_trend': win_rates,
            'pnl_trend': pnls,
            'sharpe_trend': sharpe_ratios,
            'avg_win_rate': np.mean(win_rates) if win_rates else 0,
            'avg_pnl': np.mean(pnls) if pnls else 0,
            'avg_sharpe': np.mean(sharpe_ratios) if sharpe_ratios else 0,
            'volatility': np.std(pnls) if len(pnls) > 1 else 0
        }
    
    async def _get_risk_metrics_data(self) -> Dict[str, Any]:
        """Get risk metrics data."""
        # Get latest risk metrics from metrics collector
        var_95 = self.metrics_collector.get_latest_value('risk.var_95') or 0
        var_99 = self.metrics_collector.get_latest_value('risk.var_99') or 0
        expected_shortfall = self.metrics_collector.get_latest_value('risk.expected_shortfall') or 0
        max_drawdown = self.metrics_collector.get_latest_value('risk.max_drawdown') or 0
        current_drawdown = self.metrics_collector.get_latest_value('risk.current_drawdown') or 0
        leverage = self.metrics_collector.get_latest_value('risk.leverage') or 1.0
        
        with self.position_lock:
            total_exposure = sum(
                abs(pos.quantity * pos.current_price) 
                for pos in self.active_positions.values()
            )
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall': expected_shortfall,
            'max_drawdown': max_drawdown,
            'current_drawdown': current_drawdown,
            'leverage': leverage,
            'total_exposure': total_exposure,
            'risk_score': min(100, max(0, 100 - (current_drawdown * 10)))  # Simple risk score
        }
    
    async def _get_system_health_data(self) -> Dict[str, Any]:
        """Get system health data."""
        health_score = self.metrics_collector.get_latest_value('health.overall_score') or 100
        cpu_usage = self.metrics_collector.get_latest_value('system.cpu.usage_percent') or 0
        memory_usage = self.metrics_collector.get_latest_value('system.memory.usage_percent') or 0
        error_rate = self.metrics_collector.get_latest_value('app.api_error_rate') or 0
        
        return {
            'overall_score': health_score,
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'error_rate': error_rate,
            'status': 'healthy' if health_score > 80 else 'degraded' if health_score > 50 else 'unhealthy'
        }
    
    async def _get_trading_activity_data(self) -> Dict[str, Any]:
        """Get recent trading activity data."""
        # Get recent trades from business KPI tracker
        recent_trades = self.business_kpi.get_recent_trades(50)  # Last 50 trades
        
        if not recent_trades:
            return {'recent_trades': [], 'activity_summary': {}}
        
        # Calculate activity metrics
        total_volume = sum(trade.get('quantity', 0) * trade.get('price', 0) for trade in recent_trades)
        avg_trade_size = total_volume / len(recent_trades) if recent_trades else 0
        
        return {
            'recent_trades': recent_trades[-10:],  # Last 10 for display
            'activity_summary': {
                'total_trades': len(recent_trades),
                'total_volume': total_volume,
                'avg_trade_size': avg_trade_size,
                'most_traded_symbol': self._get_most_traded_symbol(recent_trades)
            }
        }
    
    def _get_most_traded_symbol(self, trades: List[Dict[str, Any]]) -> str:
        """Get most traded symbol from recent trades."""
        if not trades:
            return ""
        
        symbol_counts = defaultdict(int)
        for trade in trades:
            symbol = trade.get('symbol', '')
            if symbol:
                symbol_counts[symbol] += 1
        
        return max(symbol_counts.items(), key=lambda x: x[1])[0] if symbol_counts else ""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dashboard generator statistics."""
        with self.position_lock:
            active_positions_count = len(self.active_positions)
        
        return {
            'is_running': self.is_running,
            'subscribers_count': len(self.subscribers),
            'active_positions': active_positions_count,
            'pnl_data_points': len(self.pnl_generator.data_points),
            'performance_data_points': len(self.performance_window),
            'cache_age': time.time() - self.last_cache_update if self.dashboard_cache else 0
        }


# Factory function
def create_dashboard_generator(metrics_collector: MetricsCollector,
                             business_kpi: BusinessKPITracker,
                             event_producer: Optional[EnhancedEventProducer] = None,
                             config: Optional[Dict[str, Any]] = None) -> DashboardDataGenerator:
    """Create and configure dashboard data generator."""
    return DashboardDataGenerator(metrics_collector, business_kpi, event_producer, config)