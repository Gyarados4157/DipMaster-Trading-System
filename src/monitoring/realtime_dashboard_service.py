#!/usr/bin/env python3
"""
DipMaster Trading System - Real-time Dashboard Data Service
实时仪表板数据服务 - WebSocket实时数据推送和REST API服务

Author: DipMaster Trading System Monitoring Agent
Date: 2025-08-18
Version: 1.0.0
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
import weakref
import uuid
from collections import deque, defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SubscriptionType(Enum):
    """订阅类型"""
    REAL_TIME_PNL = "real_time_pnl"
    POSITION_UPDATES = "position_updates"
    RISK_METRICS = "risk_metrics"
    ALERT_STREAM = "alert_stream"
    SYSTEM_HEALTH = "system_health"
    TRADE_EXECUTION = "trade_execution"
    STRATEGY_PERFORMANCE = "strategy_performance"
    MARKET_DATA = "market_data"


class DashboardDataType(Enum):
    """仪表板数据类型"""
    OVERVIEW_STATS = "overview_stats"
    PNL_CHART = "pnl_chart"
    POSITIONS_TABLE = "positions_table"
    RISK_DASHBOARD = "risk_dashboard"
    PERFORMANCE_METRICS = "performance_metrics"
    RECENT_TRADES = "recent_trades"
    ALERTS_PANEL = "alerts_panel"
    SYSTEM_STATUS = "system_status"


@dataclass
class DashboardClient:
    """仪表板客户端"""
    client_id: str
    websocket: Any = None  # WebSocket连接对象
    subscriptions: Set[SubscriptionType] = field(default_factory=set)
    connected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_agent: str = ""
    ip_address: str = ""
    
    def is_active(self, timeout_minutes: int = 30) -> bool:
        """检查客户端是否活跃"""
        return (datetime.now(timezone.utc) - self.last_activity).total_seconds() < timeout_minutes * 60


@dataclass
class RealtimeMetric:
    """实时指标数据"""
    name: str
    value: float
    unit: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'tags': self.tags
        }


@dataclass
class ChartDataPoint:
    """图表数据点"""
    timestamp: datetime
    value: float
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'timestamp': int(self.timestamp.timestamp() * 1000),  # JavaScript时间戳
            'value': self.value,
            'label': self.label,
            'metadata': self.metadata
        }


class TimeSeriesBuffer:
    """时间序列数据缓冲区"""
    
    def __init__(self, max_points: int = 1000, retention_hours: int = 24):
        self.max_points = max_points
        self.retention_hours = retention_hours
        self.data: deque = deque(maxlen=max_points)
        self.last_cleanup = datetime.now(timezone.utc)
    
    def add_point(self, timestamp: datetime, value: float, metadata: Dict[str, Any] = None) -> None:
        """添加数据点"""
        point = ChartDataPoint(
            timestamp=timestamp,
            value=value,
            metadata=metadata or {}
        )
        self.data.append(point)
        
        # 定期清理过期数据
        if (datetime.now(timezone.utc) - self.last_cleanup).total_seconds() > 300:  # 5分钟清理一次
            self._cleanup_expired_data()
    
    def _cleanup_expired_data(self) -> None:
        """清理过期数据"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.retention_hours)
        while self.data and self.data[0].timestamp < cutoff_time:
            self.data.popleft()
        self.last_cleanup = datetime.now(timezone.utc)
    
    def get_recent_points(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """获取最近N分钟的数据点"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [
            point.to_dict() for point in self.data
            if point.timestamp >= cutoff_time
        ]
    
    def get_all_points(self) -> List[Dict[str, Any]]:
        """获取所有数据点"""
        return [point.to_dict() for point in self.data]
    
    def get_latest_value(self) -> Optional[float]:
        """获取最新值"""
        return self.data[-1].value if self.data else None


class MetricsAggregator:
    """指标聚合器"""
    
    def __init__(self):
        self.time_series: Dict[str, TimeSeriesBuffer] = {}
        self.latest_metrics: Dict[str, RealtimeMetric] = {}
        self.aggregated_stats: Dict[str, Dict[str, float]] = {}
    
    def record_metric(self, metric: RealtimeMetric) -> None:
        """记录指标"""
        # 更新最新指标
        self.latest_metrics[metric.name] = metric
        
        # 添加到时间序列
        if metric.name not in self.time_series:
            self.time_series[metric.name] = TimeSeriesBuffer()
        
        self.time_series[metric.name].add_point(
            metric.timestamp,
            metric.value,
            {'tags': metric.tags, 'unit': metric.unit}
        )
        
        # 更新聚合统计
        self._update_aggregated_stats(metric)
    
    def _update_aggregated_stats(self, metric: RealtimeMetric) -> None:
        """更新聚合统计"""
        if metric.name not in self.aggregated_stats:
            self.aggregated_stats[metric.name] = {
                'count': 0,
                'sum': 0.0,
                'min': float('inf'),
                'max': float('-inf'),
                'avg': 0.0
            }
        
        stats = self.aggregated_stats[metric.name]
        stats['count'] += 1
        stats['sum'] += metric.value
        stats['min'] = min(stats['min'], metric.value)
        stats['max'] = max(stats['max'], metric.value)
        stats['avg'] = stats['sum'] / stats['count']
    
    def get_chart_data(self, metric_name: str, minutes: int = 60) -> List[Dict[str, Any]]:
        """获取图表数据"""
        if metric_name not in self.time_series:
            return []
        return self.time_series[metric_name].get_recent_points(minutes)
    
    def get_latest_value(self, metric_name: str) -> Optional[float]:
        """获取最新值"""
        if metric_name in self.latest_metrics:
            return self.latest_metrics[metric_name].value
        return None
    
    def get_stats_summary(self, metric_name: str) -> Dict[str, float]:
        """获取统计摘要"""
        return self.aggregated_stats.get(metric_name, {})


class DashboardDataGenerator:
    """仪表板数据生成器"""
    
    def __init__(self):
        self.metrics_aggregator = MetricsAggregator()
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.recent_trades: deque = deque(maxlen=100)
        self.active_alerts: List[Dict[str, Any]] = []
        self.system_status: Dict[str, Any] = {
            'overall_health': 100.0,
            'components': {},
            'uptime_seconds': 0,
            'last_update': datetime.now(timezone.utc)
        }
        
        # 缓存仪表板数据
        self._dashboard_cache: Dict[DashboardDataType, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[DashboardDataType, datetime] = {}
        self._cache_ttl_seconds = 5  # 缓存TTL
    
    async def record_trade_entry(
        self,
        position_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        strategy: str = "dipmaster"
    ) -> None:
        """记录交易入场"""
        position_data = {
            'position_id': position_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'current_price': entry_price,
            'unrealized_pnl': 0.0,
            'entry_time': datetime.now(timezone.utc),
            'strategy': strategy,
            'status': 'OPEN'
        }
        
        self.active_positions[position_id] = position_data
        
        # 记录到最近交易
        trade_record = {
            'timestamp': datetime.now(timezone.utc),
            'type': 'ENTRY',
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': entry_price,
            'strategy': strategy
        }
        self.recent_trades.append(trade_record)
        
        # 记录指标
        entry_metric = RealtimeMetric(
            name="trading.entries_count",
            value=1.0,
            tags={'symbol': symbol, 'strategy': strategy}
        )
        self.metrics_aggregator.record_metric(entry_metric)
        
        logger.debug(f"📊 Recorded trade entry: {position_id}")
    
    async def record_trade_exit(
        self,
        position_id: str,
        exit_price: float,
        realized_pnl: float,
        holding_minutes: int
    ) -> None:
        """记录交易出场"""
        if position_id not in self.active_positions:
            logger.warning(f"⚠️ Position {position_id} not found for exit")
            return
        
        position = self.active_positions[position_id]
        position['exit_price'] = exit_price
        position['realized_pnl'] = realized_pnl
        position['exit_time'] = datetime.now(timezone.utc)
        position['holding_minutes'] = holding_minutes
        position['status'] = 'CLOSED'
        
        # 移除活跃持仓
        del self.active_positions[position_id]
        
        # 记录到最近交易
        trade_record = {
            'timestamp': datetime.now(timezone.utc),
            'type': 'EXIT',
            'symbol': position['symbol'],
            'side': position['side'],
            'quantity': position['quantity'],
            'price': exit_price,
            'pnl': realized_pnl,
            'holding_minutes': holding_minutes,
            'strategy': position['strategy']
        }
        self.recent_trades.append(trade_record)
        
        # 记录PnL指标
        pnl_metric = RealtimeMetric(
            name="trading.realized_pnl",
            value=realized_pnl,
            unit="USDT",
            tags={'symbol': position['symbol'], 'strategy': position['strategy']}
        )
        self.metrics_aggregator.record_metric(pnl_metric)
        
        # 记录持仓时间指标
        holding_metric = RealtimeMetric(
            name="trading.holding_time",
            value=holding_minutes,
            unit="minutes",
            tags={'symbol': position['symbol']}
        )
        self.metrics_aggregator.record_metric(holding_metric)
        
        logger.debug(f"📊 Recorded trade exit: {position_id}, PnL: {realized_pnl}")
    
    async def update_position_prices(self, price_updates: Dict[str, float]) -> None:
        """更新持仓价格和未实现PnL"""
        for position_id, position in self.active_positions.items():
            symbol = position['symbol']
            if symbol in price_updates:
                current_price = price_updates[symbol]
                position['current_price'] = current_price
                
                # 计算未实现PnL
                entry_price = position['entry_price']
                quantity = position['quantity']
                side = position['side']
                
                if side == 'BUY':
                    unrealized_pnl = (current_price - entry_price) * quantity
                else:  # SELL
                    unrealized_pnl = (entry_price - current_price) * quantity
                
                position['unrealized_pnl'] = unrealized_pnl
    
    async def record_risk_metrics(self, risk_data: Dict[str, float]) -> None:
        """记录风险指标"""
        for metric_name, value in risk_data.items():
            risk_metric = RealtimeMetric(
                name=f"risk.{metric_name}",
                value=value,
                tags={'category': 'risk'}
            )
            self.metrics_aggregator.record_metric(risk_metric)
    
    async def add_alert(self, alert_data: Dict[str, Any]) -> None:
        """添加告警"""
        alert_with_timestamp = {
            **alert_data,
            'timestamp': datetime.now(timezone.utc),
            'id': str(uuid.uuid4())
        }
        self.active_alerts.append(alert_with_timestamp)
        
        # 限制告警数量
        if len(self.active_alerts) > 50:
            self.active_alerts = self.active_alerts[-50:]
    
    async def update_system_status(self, status_data: Dict[str, Any]) -> None:
        """更新系统状态"""
        self.system_status.update(status_data)
        self.system_status['last_update'] = datetime.now(timezone.utc)
    
    async def get_dashboard_data(self, data_type: DashboardDataType, force_refresh: bool = False) -> Dict[str, Any]:
        """获取仪表板数据"""
        # 检查缓存
        if not force_refresh and self._is_cache_valid(data_type):
            return self._dashboard_cache[data_type]
        
        # 生成新数据
        if data_type == DashboardDataType.OVERVIEW_STATS:
            data = await self._generate_overview_stats()
        elif data_type == DashboardDataType.PNL_CHART:
            data = await self._generate_pnl_chart()
        elif data_type == DashboardDataType.POSITIONS_TABLE:
            data = await self._generate_positions_table()
        elif data_type == DashboardDataType.RISK_DASHBOARD:
            data = await self._generate_risk_dashboard()
        elif data_type == DashboardDataType.PERFORMANCE_METRICS:
            data = await self._generate_performance_metrics()
        elif data_type == DashboardDataType.RECENT_TRADES:
            data = await self._generate_recent_trades()
        elif data_type == DashboardDataType.ALERTS_PANEL:
            data = await self._generate_alerts_panel()
        elif data_type == DashboardDataType.SYSTEM_STATUS:
            data = await self._generate_system_status()
        else:
            data = {'error': f'Unknown data type: {data_type}'}
        
        # 更新缓存
        self._dashboard_cache[data_type] = data
        self._cache_timestamps[data_type] = datetime.now(timezone.utc)
        
        return data
    
    def _is_cache_valid(self, data_type: DashboardDataType) -> bool:
        """检查缓存是否有效"""
        if data_type not in self._cache_timestamps:
            return False
        
        age = (datetime.now(timezone.utc) - self._cache_timestamps[data_type]).total_seconds()
        return age < self._cache_ttl_seconds
    
    async def _generate_overview_stats(self) -> Dict[str, Any]:
        """生成概览统计"""
        # 计算总PnL
        total_realized_pnl = sum(
            trade['pnl'] for trade in self.recent_trades
            if trade.get('pnl') is not None
        )
        
        total_unrealized_pnl = sum(
            pos['unrealized_pnl'] for pos in self.active_positions.values()
        )
        
        # 计算胜率
        completed_trades = [trade for trade in self.recent_trades if trade.get('pnl') is not None]
        winning_trades = sum(1 for trade in completed_trades if trade['pnl'] > 0)
        win_rate = (winning_trades / len(completed_trades)) * 100 if completed_trades else 0
        
        # 计算平均持仓时间
        avg_holding_time = sum(
            trade.get('holding_minutes', 0) for trade in completed_trades
        ) / len(completed_trades) if completed_trades else 0
        
        return {
            'total_pnl': total_realized_pnl + total_unrealized_pnl,
            'realized_pnl': total_realized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'active_positions': len(self.active_positions),
            'total_trades': len(completed_trades),
            'win_rate': win_rate,
            'avg_holding_time': avg_holding_time,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def _generate_pnl_chart(self) -> Dict[str, Any]:
        """生成PnL图表数据"""
        pnl_chart_data = self.metrics_aggregator.get_chart_data('trading.realized_pnl', minutes=240)  # 4小时
        
        # 计算累积PnL
        cumulative_pnl = 0
        cumulative_data = []
        
        for point in pnl_chart_data:
            cumulative_pnl += point['value']
            cumulative_data.append({
                'timestamp': point['timestamp'],
                'value': cumulative_pnl
            })
        
        return {
            'realized_pnl': pnl_chart_data,
            'cumulative_pnl': cumulative_data,
            'chart_config': {
                'type': 'line',
                'time_range': '4h',
                'update_interval': 60
            }
        }
    
    async def _generate_positions_table(self) -> Dict[str, Any]:
        """生成持仓表格数据"""
        positions_list = []
        
        for position_id, position in self.active_positions.items():
            # 计算持仓时间
            holding_minutes = (datetime.now(timezone.utc) - position['entry_time']).total_seconds() / 60
            
            # 计算收益率
            pnl_percentage = (position['unrealized_pnl'] / (position['entry_price'] * position['quantity'])) * 100
            
            positions_list.append({
                'position_id': position_id,
                'symbol': position['symbol'],
                'side': position['side'],
                'quantity': position['quantity'],
                'entry_price': position['entry_price'],
                'current_price': position['current_price'],
                'unrealized_pnl': position['unrealized_pnl'],
                'pnl_percentage': pnl_percentage,
                'holding_minutes': int(holding_minutes),
                'strategy': position['strategy'],
                'entry_time': position['entry_time'].isoformat()
            })
        
        return {
            'positions': positions_list,
            'total_positions': len(positions_list),
            'total_unrealized_pnl': sum(pos['unrealized_pnl'] for pos in positions_list)
        }
    
    async def _generate_risk_dashboard(self) -> Dict[str, Any]:
        """生成风险仪表板数据"""
        risk_metrics = {}
        
        # 获取最新风险指标
        for metric_name in ['var_95', 'var_99', 'expected_shortfall', 'max_drawdown', 'leverage']:
            full_name = f'risk.{metric_name}'
            value = self.metrics_aggregator.get_latest_value(full_name)
            if value is not None:
                risk_metrics[metric_name] = value
        
        # 计算风险利用率
        total_exposure = sum(
            abs(pos['unrealized_pnl']) + (pos['entry_price'] * pos['quantity'])
            for pos in self.active_positions.values()
        )
        
        # 风险限制（示例值）
        risk_limits = {
            'var_95': 200000,
            'var_99': 300000,
            'max_drawdown': 0.20,
            'max_leverage': 3.0,
            'max_exposure': 1000000
        }
        
        # 计算风险利用率
        risk_utilization = {}
        for metric, limit in risk_limits.items():
            current_value = risk_metrics.get(metric, 0)
            if limit > 0:
                utilization = (current_value / limit) * 100
                risk_utilization[metric] = min(100, max(0, utilization))
        
        return {
            'risk_metrics': risk_metrics,
            'risk_limits': risk_limits,
            'risk_utilization': risk_utilization,
            'total_exposure': total_exposure,
            'risk_score': sum(risk_utilization.values()) / len(risk_utilization) if risk_utilization else 0
        }
    
    async def _generate_performance_metrics(self) -> Dict[str, Any]:
        """生成性能指标数据"""
        # 计算策略性能指标
        completed_trades = [trade for trade in self.recent_trades if trade.get('pnl') is not None]
        
        if not completed_trades:
            return {
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'total_trades': 0,
                'sharpe_ratio': 0
            }
        
        # 分离盈利和亏损交易
        winning_trades = [trade for trade in completed_trades if trade['pnl'] > 0]
        losing_trades = [trade for trade in completed_trades if trade['pnl'] <= 0]
        
        # 计算基本指标
        win_rate = (len(winning_trades) / len(completed_trades)) * 100
        avg_win = sum(trade['pnl'] for trade in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(abs(trade['pnl']) for trade in losing_trades) / len(losing_trades) if losing_trades else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # 计算夏普比率（简化版）
        pnl_series = [trade['pnl'] for trade in completed_trades]
        if len(pnl_series) > 1:
            import statistics
            avg_return = statistics.mean(pnl_series)
            return_std = statistics.stdev(pnl_series)
            sharpe_ratio = avg_return / return_std if return_std != 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(completed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'sharpe_ratio': sharpe_ratio,
            'total_pnl': sum(pnl_series)
        }
    
    async def _generate_recent_trades(self) -> Dict[str, Any]:
        """生成最近交易数据"""
        trades_list = []
        
        for trade in list(self.recent_trades)[-20:]:  # 最近20笔交易
            trade_data = {
                'timestamp': trade['timestamp'].isoformat(),
                'type': trade['type'],
                'symbol': trade['symbol'],
                'side': trade['side'],
                'quantity': trade['quantity'],
                'price': trade['price'],
                'strategy': trade['strategy']
            }
            
            # 添加出场特有字段
            if trade['type'] == 'EXIT':
                trade_data.update({
                    'pnl': trade.get('pnl', 0),
                    'holding_minutes': trade.get('holding_minutes', 0)
                })
            
            trades_list.append(trade_data)
        
        return {
            'trades': trades_list,
            'total_count': len(self.recent_trades)
        }
    
    async def _generate_alerts_panel(self) -> Dict[str, Any]:
        """生成告警面板数据"""
        # 按严重性分类告警
        alerts_by_severity = {
            'CRITICAL': [],
            'WARNING': [],
            'INFO': []
        }
        
        for alert in self.active_alerts[-20:]:  # 最近20个告警
            severity = alert.get('severity', 'INFO').upper()
            if severity in alerts_by_severity:
                alerts_by_severity[severity].append({
                    'id': alert['id'],
                    'timestamp': alert['timestamp'].isoformat(),
                    'message': alert.get('message', ''),
                    'category': alert.get('category', ''),
                    'severity': severity
                })
        
        # 统计告警数量
        alert_counts = {
            severity: len(alerts)
            for severity, alerts in alerts_by_severity.items()
        }
        
        return {
            'alerts_by_severity': alerts_by_severity,
            'alert_counts': alert_counts,
            'total_alerts': len(self.active_alerts),
            'latest_alert': self.active_alerts[-1] if self.active_alerts else None
        }
    
    async def _generate_system_status(self) -> Dict[str, Any]:
        """生成系统状态数据"""
        # 计算组件健康度
        component_health = {}
        for component, status in self.system_status.get('components', {}).items():
            if isinstance(status, dict):
                health_score = status.get('health_score', 100)
                component_health[component] = {
                    'score': health_score,
                    'status': 'healthy' if health_score > 80 else 'degraded' if health_score > 50 else 'unhealthy'
                }
        
        return {
            'overall_health': self.system_status.get('overall_health', 100),
            'component_health': component_health,
            'uptime_seconds': self.system_status.get('uptime_seconds', 0),
            'last_update': self.system_status['last_update'].isoformat(),
            'system_info': {
                'active_connections': len(component_health),
                'memory_usage': self.metrics_aggregator.get_latest_value('system.memory.usage_percent') or 0,
                'cpu_usage': self.metrics_aggregator.get_latest_value('system.cpu.usage_percent') or 0
            }
        }


class WebSocketManager:
    """WebSocket连接管理器"""
    
    def __init__(self):
        self.clients: Dict[str, DashboardClient] = {}
        self.subscription_handlers: Dict[SubscriptionType, List[Callable]] = defaultdict(list)
        self.broadcast_queue: asyncio.Queue = asyncio.Queue()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._broadcast_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """启动WebSocket管理器"""
        self._cleanup_task = asyncio.create_task(self._cleanup_inactive_clients())
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        logger.info("🔌 WebSocket manager started")
    
    async def stop(self) -> None:
        """停止WebSocket管理器"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._broadcast_task:
            self._broadcast_task.cancel()
        
        # 关闭所有连接
        for client in self.clients.values():
            if client.websocket:
                try:
                    await client.websocket.close()
                except:
                    pass
        
        self.clients.clear()
        logger.info("🔌 WebSocket manager stopped")
    
    async def register_client(self, client_id: str, websocket: Any, user_agent: str = "", ip_address: str = "") -> DashboardClient:
        """注册新客户端"""
        client = DashboardClient(
            client_id=client_id,
            websocket=websocket,
            user_agent=user_agent,
            ip_address=ip_address
        )
        
        self.clients[client_id] = client
        logger.info(f"🔌 Registered client: {client_id}")
        return client
    
    async def unregister_client(self, client_id: str) -> None:
        """注销客户端"""
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"🔌 Unregistered client: {client_id}")
    
    async def subscribe(self, client_id: str, subscription_type: SubscriptionType) -> bool:
        """客户端订阅"""
        if client_id in self.clients:
            self.clients[client_id].subscriptions.add(subscription_type)
            self.clients[client_id].last_activity = datetime.now(timezone.utc)
            logger.debug(f"📺 Client {client_id} subscribed to {subscription_type.value}")
            return True
        return False
    
    async def unsubscribe(self, client_id: str, subscription_type: SubscriptionType) -> bool:
        """客户端取消订阅"""
        if client_id in self.clients:
            self.clients[client_id].subscriptions.discard(subscription_type)
            self.clients[client_id].last_activity = datetime.now(timezone.utc)
            logger.debug(f"📺 Client {client_id} unsubscribed from {subscription_type.value}")
            return True
        return False
    
    async def broadcast_to_subscribers(self, subscription_type: SubscriptionType, data: Dict[str, Any]) -> int:
        """向订阅者广播数据"""
        message = {
            'type': subscription_type.value,
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        await self.broadcast_queue.put((subscription_type, message))
        return len([c for c in self.clients.values() if subscription_type in c.subscriptions])
    
    async def send_to_client(self, client_id: str, data: Dict[str, Any]) -> bool:
        """发送数据给特定客户端"""
        if client_id in self.clients:
            client = self.clients[client_id]
            if client.websocket:
                try:
                    await client.websocket.send(json.dumps(data))
                    client.last_activity = datetime.now(timezone.utc)
                    return True
                except Exception as e:
                    logger.error(f"❌ Failed to send data to client {client_id}: {e}")
                    await self.unregister_client(client_id)
        return False
    
    async def _broadcast_loop(self) -> None:
        """广播循环"""
        while True:
            try:
                subscription_type, message = await self.broadcast_queue.get()
                
                # 发送给所有订阅者
                subscribers = [
                    client for client in self.clients.values()
                    if subscription_type in client.subscriptions and client.websocket
                ]
                
                for client in subscribers:
                    try:
                        await client.websocket.send(json.dumps(message))
                        client.last_activity = datetime.now(timezone.utc)
                    except Exception as e:
                        logger.error(f"❌ Failed to broadcast to client {client.client_id}: {e}")
                        await self.unregister_client(client.client_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in broadcast loop: {e}")
                await asyncio.sleep(1)
    
    async def _cleanup_inactive_clients(self) -> None:
        """清理不活跃的客户端"""
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                inactive_clients = []
                
                for client_id, client in self.clients.items():
                    if not client.is_active(timeout_minutes=30):
                        inactive_clients.append(client_id)
                
                for client_id in inactive_clients:
                    await self.unregister_client(client_id)
                
                await asyncio.sleep(300)  # 5分钟清理一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取WebSocket统计信息"""
        subscription_counts = defaultdict(int)
        for client in self.clients.values():
            for sub_type in client.subscriptions:
                subscription_counts[sub_type.value] += 1
        
        return {
            'total_clients': len(self.clients),
            'active_clients': len([c for c in self.clients.values() if c.is_active()]),
            'subscription_counts': dict(subscription_counts),
            'queue_size': self.broadcast_queue.qsize()
        }


class RealtimeDashboardService:
    """实时仪表板服务"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.data_generator = DashboardDataGenerator()
        self.websocket_manager = WebSocketManager()
        self.is_running = False
        self._update_task: Optional[asyncio.Task] = None
        
        # 更新间隔配置
        self.update_intervals = {
            SubscriptionType.REAL_TIME_PNL: 1,      # 1秒
            SubscriptionType.POSITION_UPDATES: 2,   # 2秒
            SubscriptionType.RISK_METRICS: 5,       # 5秒
            SubscriptionType.SYSTEM_HEALTH: 10,     # 10秒
            SubscriptionType.STRATEGY_PERFORMANCE: 30, # 30秒
        }
    
    async def start(self) -> None:
        """启动仪表板服务"""
        if self.is_running:
            return
        
        self.is_running = True
        await self.websocket_manager.start()
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("🚀 Realtime Dashboard Service started")
    
    async def stop(self) -> None:
        """停止仪表板服务"""
        self.is_running = False
        
        if self._update_task:
            self._update_task.cancel()
        
        await self.websocket_manager.stop()
        logger.info("🛑 Realtime Dashboard Service stopped")
    
    async def _update_loop(self) -> None:
        """更新循环"""
        last_updates = {sub_type: 0 for sub_type in SubscriptionType}
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # 检查每个订阅类型是否需要更新
                for sub_type, interval in self.update_intervals.items():
                    if current_time - last_updates[sub_type] >= interval:
                        await self._update_subscription_data(sub_type)
                        last_updates[sub_type] = current_time
                
                await asyncio.sleep(1)  # 1秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in update loop: {e}")
                await asyncio.sleep(1)
    
    async def _update_subscription_data(self, subscription_type: SubscriptionType) -> None:
        """更新订阅数据"""
        try:
            if subscription_type == SubscriptionType.REAL_TIME_PNL:
                data = await self.data_generator.get_dashboard_data(DashboardDataType.PNL_CHART)
                await self.websocket_manager.broadcast_to_subscribers(subscription_type, data)
            
            elif subscription_type == SubscriptionType.POSITION_UPDATES:
                data = await self.data_generator.get_dashboard_data(DashboardDataType.POSITIONS_TABLE)
                await self.websocket_manager.broadcast_to_subscribers(subscription_type, data)
            
            elif subscription_type == SubscriptionType.RISK_METRICS:
                data = await self.data_generator.get_dashboard_data(DashboardDataType.RISK_DASHBOARD)
                await self.websocket_manager.broadcast_to_subscribers(subscription_type, data)
            
            elif subscription_type == SubscriptionType.SYSTEM_HEALTH:
                data = await self.data_generator.get_dashboard_data(DashboardDataType.SYSTEM_STATUS)
                await self.websocket_manager.broadcast_to_subscribers(subscription_type, data)
            
            elif subscription_type == SubscriptionType.STRATEGY_PERFORMANCE:
                data = await self.data_generator.get_dashboard_data(DashboardDataType.PERFORMANCE_METRICS)
                await self.websocket_manager.broadcast_to_subscribers(subscription_type, data)
        
        except Exception as e:
            logger.error(f"❌ Failed to update {subscription_type.value}: {e}")
    
    # 公共API方法
    
    async def record_trade_entry(self, position_id: str, symbol: str, side: str, quantity: float, entry_price: float, strategy: str = "dipmaster") -> None:
        """记录交易入场"""
        await self.data_generator.record_trade_entry(position_id, symbol, side, quantity, entry_price, strategy)
        
        # 立即推送位置更新
        data = await self.data_generator.get_dashboard_data(DashboardDataType.POSITIONS_TABLE, force_refresh=True)
        await self.websocket_manager.broadcast_to_subscribers(SubscriptionType.POSITION_UPDATES, data)
    
    async def record_trade_exit(self, position_id: str, exit_price: float, realized_pnl: float, holding_minutes: int) -> None:
        """记录交易出场"""
        await self.data_generator.record_trade_exit(position_id, exit_price, realized_pnl, holding_minutes)
        
        # 立即推送更新
        positions_data = await self.data_generator.get_dashboard_data(DashboardDataType.POSITIONS_TABLE, force_refresh=True)
        await self.websocket_manager.broadcast_to_subscribers(SubscriptionType.POSITION_UPDATES, positions_data)
        
        pnl_data = await self.data_generator.get_dashboard_data(DashboardDataType.PNL_CHART, force_refresh=True)
        await self.websocket_manager.broadcast_to_subscribers(SubscriptionType.REAL_TIME_PNL, pnl_data)
    
    async def update_market_prices(self, price_updates: Dict[str, float]) -> None:
        """更新市场价格"""
        await self.data_generator.update_position_prices(price_updates)
    
    async def record_risk_metrics(self, risk_data: Dict[str, float]) -> None:
        """记录风险指标"""
        await self.data_generator.record_risk_metrics(risk_data)
    
    async def add_alert(self, alert_data: Dict[str, Any]) -> None:
        """添加告警"""
        await self.data_generator.add_alert(alert_data)
        
        # 立即推送告警
        data = await self.data_generator.get_dashboard_data(DashboardDataType.ALERTS_PANEL, force_refresh=True)
        await self.websocket_manager.broadcast_to_subscribers(SubscriptionType.ALERT_STREAM, data)
    
    async def get_dashboard_data(self, data_type: DashboardDataType) -> Dict[str, Any]:
        """获取仪表板数据"""
        return await self.data_generator.get_dashboard_data(data_type)
    
    async def register_websocket_client(self, client_id: str, websocket: Any) -> DashboardClient:
        """注册WebSocket客户端"""
        return await self.websocket_manager.register_client(client_id, websocket)
    
    async def subscribe_client(self, client_id: str, subscription_type: SubscriptionType) -> bool:
        """客户端订阅"""
        return await self.websocket_manager.subscribe(client_id, subscription_type)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        return {
            'is_running': self.is_running,
            'websocket_stats': self.websocket_manager.get_stats(),
            'active_positions': len(self.data_generator.active_positions),
            'recent_trades_count': len(self.data_generator.recent_trades),
            'active_alerts_count': len(self.data_generator.active_alerts)
        }


# 工厂函数
def create_dashboard_service(config: Dict[str, Any] = None) -> RealtimeDashboardService:
    """创建实时仪表板服务"""
    return RealtimeDashboardService(config)


# 演示函数
async def dashboard_service_demo():
    """仪表板服务演示"""
    print("🚀 DipMaster Realtime Dashboard Service Demo")
    
    # 创建仪表板服务
    dashboard_service = create_dashboard_service()
    
    try:
        # 启动服务
        await dashboard_service.start()
        print("✅ Dashboard service started")
        
        # 模拟交易数据
        print("📊 Simulating trading data...")
        
        # 记录交易入场
        await dashboard_service.record_trade_entry(
            position_id="pos_demo_001",
            symbol="BTCUSDT",
            side="BUY",
            quantity=0.15,
            entry_price=43250.50,
            strategy="dipmaster"
        )
        
        await dashboard_service.record_trade_entry(
            position_id="pos_demo_002",
            symbol="ETHUSDT",
            side="BUY",
            quantity=2.5,
            entry_price=2645.30,
            strategy="dipmaster"
        )
        
        # 更新市场价格
        await dashboard_service.update_market_prices({
            "BTCUSDT": 43350.75,
            "ETHUSDT": 2675.20
        })
        
        # 记录风险指标
        await dashboard_service.record_risk_metrics({
            'var_95': 125000.50,
            'var_99': 187500.75,
            'expected_shortfall': 210000.00,
            'max_drawdown': 0.082,
            'leverage': 1.5
        })
        
        # 添加告警
        await dashboard_service.add_alert({
            'severity': 'WARNING',
            'category': 'RISK_LIMIT',
            'message': 'Portfolio exposure approaching 80% of limit',
            'source': 'risk_monitor'
        })
        
        # 等待数据处理
        await asyncio.sleep(2)
        
        # 模拟交易出场
        await dashboard_service.record_trade_exit(
            position_id="pos_demo_001",
            exit_price=43420.80,
            realized_pnl=25.545,  # (43420.80 - 43250.50) * 0.15
            holding_minutes=75
        )
        
        # 获取各种仪表板数据
        print("\n📊 Dashboard Data:")
        
        overview = await dashboard_service.get_dashboard_data(DashboardDataType.OVERVIEW_STATS)
        print(f"   Total PnL: {overview['total_pnl']:.2f}")
        print(f"   Active Positions: {overview['active_positions']}")
        print(f"   Win Rate: {overview['win_rate']:.1f}%")
        
        positions = await dashboard_service.get_dashboard_data(DashboardDataType.POSITIONS_TABLE)
        print(f"   Open Positions: {len(positions['positions'])}")
        
        performance = await dashboard_service.get_dashboard_data(DashboardDataType.PERFORMANCE_METRICS)
        print(f"   Total Trades: {performance['total_trades']}")
        print(f"   Profit Factor: {performance['profit_factor']:.2f}")
        
        # 获取服务统计
        stats = dashboard_service.get_service_stats()
        print(f"\n📈 Service Stats:")
        print(f"   Active Positions: {stats['active_positions']}")
        print(f"   Recent Trades: {stats['recent_trades_count']}")
        print(f"   Active Alerts: {stats['active_alerts_count']}")
        
        print("✅ Demo completed successfully")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
    
    finally:
        await dashboard_service.stop()
        print("🛑 Dashboard service stopped")


if __name__ == "__main__":
    asyncio.run(dashboard_service_demo())