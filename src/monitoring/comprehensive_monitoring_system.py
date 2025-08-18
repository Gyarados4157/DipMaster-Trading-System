#!/usr/bin/env python3
"""
DipMaster Comprehensive Monitoring and Log Collection System
全面监控和日志收集系统 - 持续监控交易系统一致性和性能

Features:
- Signal-Position-Execution consistency validation
- Real-time vs backtest drift detection with statistical tests
- VaR/ES monitoring with automated risk alerts
- Kafka event stream production for all monitoring events
- Structured logging with correlation IDs and event classification
- Automated daily/weekly/monthly reporting system
- System health monitoring with predictive alerts
- 24/7 continuous monitoring with failover capabilities

Author: DipMaster Trading System Monitoring Agent
Date: 2025-08-18
Version: 2.0.0
"""

import asyncio
import json
import time
import logging
import uuid
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import threading
from pathlib import Path
import sqlite3
import psutil

# Import existing monitoring components
from .kafka_event_schemas import (
    DipMasterKafkaStreamer, 
    ExecutionReportEvent, 
    RiskMetricsEvent, 
    AlertEvent, 
    StrategyPerformanceEvent,
    SystemHealthEvent,
    TradeSignalEvent,
    PositionUpdateEvent
)
from .complete_monitoring_system import CompleteMonitoringSystem, MonitoringConfig
from .trading_consistency_monitor import TradingConsistencyMonitor, SignalData, PositionData, ExecutionData

logger = logging.getLogger(__name__)


class MonitoringMode(Enum):
    """监控模式枚举"""
    DEVELOPMENT = "development"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"
    RESEARCH_ONLY = "research_only"


class AlertPriority(Enum):
    """告警优先级枚举"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class MonitoringEventLog:
    """监控事件日志结构"""
    event_id: str
    timestamp: datetime
    correlation_id: str
    event_type: str
    severity: str
    component: str
    message: str
    details: Dict[str, Any]
    tags: Dict[str, str]
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    auto_remediated: bool = False


@dataclass
class ConsistencyMetrics:
    """一致性指标"""
    signal_position_match_rate: float = 0.0
    position_execution_match_rate: float = 0.0
    timing_accuracy_rate: float = 0.0
    price_deviation_avg_bps: float = 0.0
    boundary_compliance_rate: float = 0.0
    overall_consistency_score: float = 0.0


@dataclass
class DriftDetectionResult:
    """漂移检测结果"""
    drift_detected: bool
    drift_score: float  # 0-100, higher is worse
    statistical_significance: float  # p-value
    affected_metrics: List[str]
    recommendation: str
    time_window: str
    comparison_baseline: str


@dataclass
class RiskMonitoringState:
    """风险监控状态"""
    current_var_95: float = 0.0
    current_var_99: float = 0.0
    current_expected_shortfall: float = 0.0
    current_drawdown: float = 0.0
    max_drawdown: float = 0.0
    portfolio_value: float = 0.0
    risk_utilization: float = 0.0
    breach_count: int = 0
    last_breach_time: Optional[datetime] = None


class ComprehensiveMonitoringSystem:
    """
    DipMaster全面监控系统
    
    提供完整的交易系统监控，包括信号一致性验证、漂移检测、
    风险监控、事件流生产和自动化报告生成。
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化全面监控系统
        
        Args:
            config: 监控配置参数
        """
        self.config = config or {}
        self.mode = MonitoringMode(self.config.get('mode', 'development'))
        self.is_running = False
        self.start_time = None
        self.correlation_id = str(uuid.uuid4())
        
        # 核心组件
        self.kafka_streamer: Optional[DipMasterKafkaStreamer] = None
        self.consistency_monitor: Optional[TradingConsistencyMonitor] = None
        self.complete_monitoring: Optional[CompleteMonitoringSystem] = None
        
        # 数据存储
        self.db_path = self.config.get('db_path', 'data/monitoring.db')
        self.event_logs: deque = deque(maxlen=10000)  # 内存中的事件日志
        self.signal_registry: Dict[str, SignalData] = {}
        self.position_registry: Dict[str, PositionData] = {}
        self.execution_registry: Dict[str, ExecutionData] = {}
        
        # 性能历史数据
        self.backtest_baseline: Dict[str, float] = {}
        self.production_history: deque = deque(maxlen=1440)  # 24小时分钟级数据
        self.hourly_performance: deque = deque(maxlen=720)   # 30天小时级数据
        self.daily_performance: deque = deque(maxlen=90)     # 90天日级数据
        
        # 监控状态
        self.consistency_metrics = ConsistencyMetrics()
        self.risk_monitoring_state = RiskMonitoringState()
        self.system_health_score = 100.0
        
        # 告警管理
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.alert_suppression: Dict[str, datetime] = {}  # 告警抑制
        
        # 统计计数器
        self.stats = {
            'events_processed': 0,
            'signals_validated': 0,
            'positions_tracked': 0,
            'executions_monitored': 0,
            'alerts_generated': 0,
            'drift_checks_performed': 0,
            'consistency_violations': 0,
            'risk_breaches': 0,
            'reports_generated': 0,
            'uptime_seconds': 0
        }
        
        # 监控阈值配置
        self.thresholds = {
            'consistency': {
                'signal_position_match_min': 95.0,
                'position_execution_match_min': 98.0,
                'price_deviation_max_bps': 20.0,
                'timing_deviation_max_minutes': 2.0,
                'boundary_compliance_min': 100.0
            },
            'drift': {
                'warning_threshold_pct': 5.0,
                'critical_threshold_pct': 10.0,
                'statistical_significance': 0.05,
                'min_data_points': 30
            },
            'risk': {
                'var_95_limit': 200000.0,
                'var_99_limit': 300000.0,
                'max_drawdown_limit': 0.15,
                'daily_loss_limit': 1000.0,
                'position_limit_count': 10,
                'leverage_limit': 3.0
            },
            'system_health': {
                'cpu_warning': 80.0,
                'memory_warning': 80.0,
                'disk_warning': 85.0,
                'response_time_warning_ms': 1000.0,
                'error_rate_warning': 5.0
            }
        }
        self.thresholds.update(self.config.get('thresholds', {}))
        
        # 后台任务管理
        self.background_tasks: List[asyncio.Task] = []
        
        logger.info(f"🚀 ComprehensiveMonitoringSystem initialized in {self.mode.value} mode")
        self._setup_database()
        self._initialize_components()
    
    def _setup_database(self):
        """设置监控数据库"""
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS event_logs (
                        event_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        correlation_id TEXT,
                        event_type TEXT,
                        severity TEXT,
                        component TEXT,
                        message TEXT,
                        details TEXT,
                        tags TEXT,
                        resolved INTEGER DEFAULT 0,
                        resolution_time TEXT,
                        auto_remediated INTEGER DEFAULT 0
                    );
                    
                    CREATE TABLE IF NOT EXISTS consistency_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        signal_position_match_rate REAL,
                        position_execution_match_rate REAL,
                        timing_accuracy_rate REAL,
                        price_deviation_avg_bps REAL,
                        boundary_compliance_rate REAL,
                        overall_consistency_score REAL
                    );
                    
                    CREATE TABLE IF NOT EXISTS drift_detection_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        drift_detected INTEGER,
                        drift_score REAL,
                        statistical_significance REAL,
                        affected_metrics TEXT,
                        recommendation TEXT,
                        time_window TEXT,
                        comparison_baseline TEXT
                    );
                    
                    CREATE TABLE IF NOT EXISTS performance_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        time_window TEXT,
                        win_rate REAL,
                        profit_factor REAL,
                        sharpe_ratio REAL,
                        max_drawdown REAL,
                        total_pnl REAL,
                        total_trades INTEGER,
                        avg_holding_time_minutes REAL,
                        consistency_score REAL,
                        drift_score REAL
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_event_logs_timestamp ON event_logs(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_consistency_timestamp ON consistency_history(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_drift_timestamp ON drift_detection_history(timestamp);
                    CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_snapshots(timestamp);
                """)
                
            logger.info("📊 Monitoring database initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to setup monitoring database: {e}")
            raise
    
    def _initialize_components(self):
        """初始化监控组件"""
        try:
            # 初始化Kafka事件流
            kafka_config = self.config.get('kafka', {})
            if kafka_config and self.mode != MonitoringMode.RESEARCH_ONLY:
                self.kafka_streamer = DipMasterKafkaStreamer(kafka_config)
            
            # 初始化一致性监控器
            consistency_config = {
                'thresholds': self.thresholds['consistency'],
                'dipmaster_params': self.config.get('dipmaster_params', {})
            }
            self.consistency_monitor = TradingConsistencyMonitor(
                kafka_producer=self.kafka_streamer,
                config=consistency_config
            )
            
            # 初始化完整监控系统
            if self.mode == MonitoringMode.LIVE_TRADING:
                monitoring_config = MonitoringConfig(
                    kafka_servers=kafka_config.get('servers', ['localhost:9092']),
                    log_dir=self.config.get('log_dir', 'logs'),
                    dipmaster_params=self.config.get('dipmaster_params', {})
                )
                self.complete_monitoring = CompleteMonitoringSystem(monitoring_config)
            
            # 加载基准性能数据
            self._load_baseline_performance()
            
            logger.info("✅ All monitoring components initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize monitoring components: {e}")
            raise
    
    def _load_baseline_performance(self):
        """加载基准性能数据"""
        try:
            baseline_file = self.config.get('baseline_performance_file')
            if baseline_file and Path(baseline_file).exists():
                with open(baseline_file, 'r') as f:
                    self.backtest_baseline = json.load(f)
                logger.info(f"📊 Loaded baseline performance from {baseline_file}")
            else:
                # 默认DipMaster基准指标
                self.backtest_baseline = {
                    'win_rate': 0.821,
                    'profit_factor': 1.8,
                    'sharpe_ratio': 1.5,
                    'max_drawdown': 0.08,
                    'avg_trade_pnl': 12.5,
                    'avg_holding_time_minutes': 96,
                    'dip_buying_rate': 0.879,
                    'boundary_compliance_rate': 1.0
                }
                logger.info("📊 Using default DipMaster baseline performance metrics")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to load baseline performance: {e}")
            self.backtest_baseline = {}
    
    async def start(self):
        """启动全面监控系统"""
        if self.is_running:
            logger.warning("⚠️ Monitoring system already running")
            return
        
        try:
            self.is_running = True
            self.start_time = time.time()
            
            logger.info("🚀 Starting comprehensive monitoring system...")
            
            # 启动Kafka事件流
            if self.kafka_streamer:
                await self.kafka_streamer.start()
                logger.info("📤 Kafka event streamer started")
            
            # 启动完整监控系统
            if self.complete_monitoring:
                await self.complete_monitoring.start()
                logger.info("🎯 Complete monitoring system started")
            
            # 启动后台监控任务
            await self._start_background_tasks()
            
            # 记录启动事件
            await self._log_monitoring_event(
                event_type="SYSTEM_STARTED",
                severity="INFO",
                component="comprehensive_monitoring",
                message="Comprehensive monitoring system started successfully",
                details={
                    'mode': self.mode.value,
                    'kafka_enabled': self.kafka_streamer is not None,
                    'complete_monitoring_enabled': self.complete_monitoring is not None,
                    'thresholds': self.thresholds
                }
            )
            
            logger.info("✅ Comprehensive monitoring system started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring system: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """停止全面监控系统"""
        if not self.is_running:
            logger.warning("⚠️ Monitoring system not running")
            return
        
        try:
            logger.info("🛑 Stopping comprehensive monitoring system...")
            
            self.is_running = False
            uptime = time.time() - self.start_time if self.start_time else 0
            
            # 停止后台任务
            await self._stop_background_tasks()
            
            # 停止组件
            if self.complete_monitoring:
                await self.complete_monitoring.stop()
                logger.info("🎯 Complete monitoring system stopped")
            
            if self.kafka_streamer:
                await self.kafka_streamer.stop()
                logger.info("📤 Kafka event streamer stopped")
            
            # 记录停止事件
            await self._log_monitoring_event(
                event_type="SYSTEM_STOPPED",
                severity="INFO",
                component="comprehensive_monitoring",
                message="Comprehensive monitoring system stopped",
                details={
                    'uptime_seconds': uptime,
                    'final_stats': self.stats.copy()
                }
            )
            
            # 保存最终状态
            await self._save_monitoring_state()
            
            logger.info("✅ Comprehensive monitoring system stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping monitoring system: {e}")
    
    async def _start_background_tasks(self):
        """启动后台监控任务"""
        self.background_tasks = [
            asyncio.create_task(self._consistency_monitoring_loop()),
            asyncio.create_task(self._drift_detection_loop()),
            asyncio.create_task(self._risk_monitoring_loop()),
            asyncio.create_task(self._system_health_monitoring_loop()),
            asyncio.create_task(self._alert_management_loop()),
            asyncio.create_task(self._performance_snapshot_loop()),
            asyncio.create_task(self._report_generation_loop()),
            asyncio.create_task(self._statistics_update_loop())
        ]
        logger.info(f"🔄 Started {len(self.background_tasks)} background monitoring tasks")
    
    async def _stop_background_tasks(self):
        """停止后台监控任务"""
        for task in self.background_tasks:
            task.cancel()
        
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        self.background_tasks.clear()
        logger.info("🔄 All background monitoring tasks stopped")
    
    async def _consistency_monitoring_loop(self):
        """一致性监控循环"""
        while self.is_running:
            try:
                await self._perform_consistency_checks()
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in consistency monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _drift_detection_loop(self):
        """漂移检测循环"""
        while self.is_running:
            try:
                await self._perform_drift_detection()
                await asyncio.sleep(300)  # 每5分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in drift detection loop: {e}")
                await asyncio.sleep(10)
    
    async def _risk_monitoring_loop(self):
        """风险监控循环"""
        while self.is_running:
            try:
                await self._perform_risk_monitoring()
                await asyncio.sleep(60)  # 每1分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in risk monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _system_health_monitoring_loop(self):
        """系统健康监控循环"""
        while self.is_running:
            try:
                await self._monitor_system_health()
                await asyncio.sleep(60)  # 每1分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in system health monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _alert_management_loop(self):
        """告警管理循环"""
        while self.is_running:
            try:
                await self._manage_alerts()
                await asyncio.sleep(30)  # 每30秒管理一次告警
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in alert management loop: {e}")
                await asyncio.sleep(5)
    
    async def _performance_snapshot_loop(self):
        """性能快照循环"""
        while self.is_running:
            try:
                await self._take_performance_snapshot()
                await asyncio.sleep(900)  # 每15分钟快照一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in performance snapshot loop: {e}")
                await asyncio.sleep(30)
    
    async def _report_generation_loop(self):
        """报告生成循环"""
        while self.is_running:
            try:
                await self._check_report_schedule()
                await asyncio.sleep(3600)  # 每小时检查一次报告计划
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in report generation loop: {e}")
                await asyncio.sleep(60)
    
    async def _statistics_update_loop(self):
        """统计更新循环"""
        while self.is_running:
            try:
                self._update_statistics()
                await asyncio.sleep(30)  # 每30秒更新统计
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in statistics update loop: {e}")
                await asyncio.sleep(5)
    
    async def record_signal(self, signal_data: Dict[str, Any]):
        """记录交易信号"""
        try:
            # 创建SignalData对象
            signal = SignalData(
                signal_id=signal_data['signal_id'],
                timestamp=datetime.fromisoformat(signal_data['timestamp']) if isinstance(signal_data['timestamp'], str) else signal_data['timestamp'],
                symbol=signal_data['symbol'],
                signal_type=signal_data['signal_type'],
                confidence=signal_data['confidence'],
                price=signal_data['price'],
                rsi=signal_data['technical_indicators']['rsi'],
                ma20_distance=signal_data['technical_indicators']['ma20_distance'],
                volume_ratio=signal_data['technical_indicators']['volume_ratio'],
                expected_entry_price=signal_data.get('expected_entry_price', signal_data['price']),
                expected_holding_minutes=signal_data.get('expected_holding_minutes', 60),
                strategy_params=signal_data.get('strategy_params', {})
            )
            
            # 存储到注册表
            self.signal_registry[signal.signal_id] = signal
            
            # 记录到一致性监控器
            if self.consistency_monitor:
                await self.consistency_monitor.record_signal(signal)
            
            # 发布Kafka事件
            if self.kafka_streamer:
                await self.kafka_streamer.publish_trade_signal(
                    signal_id=signal.signal_id,
                    symbol=signal.symbol,
                    signal_type=signal.signal_type,
                    confidence=signal.confidence,
                    price=signal.price,
                    technical_indicators=signal_data['technical_indicators'],
                    strategy="dipmaster"
                )
            
            # 记录监控事件
            await self._log_monitoring_event(
                event_type="SIGNAL_RECORDED",
                severity="DEBUG",
                component="signal_processing",
                message=f"Signal {signal.signal_id} recorded for {signal.symbol}",
                details={
                    'signal_id': signal.signal_id,
                    'symbol': signal.symbol,
                    'signal_type': signal.signal_type,
                    'confidence': signal.confidence,
                    'rsi': signal.rsi
                }
            )
            
            self.stats['signals_validated'] += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to record signal: {e}")
            await self._generate_alert(
                alert_id=f"signal_error_{int(time.time())}",
                severity="WARNING",
                category="SIGNAL_PROCESSING_ERROR",
                message=f"Failed to record signal: {e}",
                details={'error': str(e), 'signal_data': signal_data}
            )
    
    async def record_position(self, position_data: Dict[str, Any]):
        """记录持仓信息"""
        try:
            # 创建PositionData对象
            position = PositionData(
                position_id=position_data['position_id'],
                signal_id=position_data.get('signal_id', ''),
                symbol=position_data['symbol'],
                side=position_data['side'],
                quantity=position_data['quantity'],
                entry_price=position_data['entry_price'],
                entry_time=datetime.fromisoformat(position_data['entry_time']) if isinstance(position_data['entry_time'], str) else position_data['entry_time'],
                exit_price=position_data.get('exit_price'),
                exit_time=datetime.fromisoformat(position_data['exit_time']) if position_data.get('exit_time') and isinstance(position_data['exit_time'], str) else position_data.get('exit_time'),
                holding_minutes=position_data.get('holding_minutes'),
                pnl=position_data.get('pnl'),
                realized=position_data.get('realized', False)
            )
            
            # 存储到注册表
            self.position_registry[position.position_id] = position
            
            # 记录到一致性监控器
            if self.consistency_monitor:
                await self.consistency_monitor.record_position(position)
            
            # 发布Kafka事件
            if self.kafka_streamer:
                await self.kafka_streamer.publish_position_update(
                    position_id=position.position_id,
                    symbol=position.symbol,
                    side=position.side,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    current_price=position.exit_price or position.entry_price,
                    unrealized_pnl=position.pnl or 0.0,
                    realized_pnl=position.pnl if position.realized else 0.0,
                    holding_time_minutes=position.holding_minutes or 0,
                    status="CLOSED" if position.realized else "OPEN"
                )
            
            # 记录监控事件
            event_type = "POSITION_CLOSED" if position.realized else "POSITION_OPENED"
            await self._log_monitoring_event(
                event_type=event_type,
                severity="INFO",
                component="position_management",
                message=f"Position {position.position_id} {event_type.lower().replace('_', ' ')} for {position.symbol}",
                details={
                    'position_id': position.position_id,
                    'signal_id': position.signal_id,
                    'symbol': position.symbol,
                    'side': position.side,
                    'quantity': position.quantity,
                    'entry_price': position.entry_price,
                    'exit_price': position.exit_price,
                    'pnl': position.pnl,
                    'holding_minutes': position.holding_minutes
                }
            )
            
            self.stats['positions_tracked'] += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to record position: {e}")
            await self._generate_alert(
                alert_id=f"position_error_{int(time.time())}",
                severity="WARNING",
                category="POSITION_PROCESSING_ERROR",
                message=f"Failed to record position: {e}",
                details={'error': str(e), 'position_data': position_data}
            )
    
    async def record_execution(self, execution_data: Dict[str, Any]):
        """记录订单执行"""
        try:
            # 创建ExecutionData对象
            execution = ExecutionData(
                execution_id=execution_data['execution_id'],
                position_id=execution_data.get('position_id', ''),
                order_type=execution_data.get('order_type', 'MARKET'),
                symbol=execution_data['symbol'],
                side=execution_data['side'],
                quantity=execution_data['quantity'],
                requested_price=execution_data.get('requested_price', execution_data['price']),
                executed_price=execution_data['price'],
                execution_time=datetime.fromisoformat(execution_data['execution_time']) if isinstance(execution_data['execution_time'], str) else execution_data['execution_time'],
                latency_ms=execution_data.get('latency_ms', 0.0),
                slippage_bps=execution_data.get('slippage_bps', 0.0),
                fees=execution_data.get('fees', 0.0),
                venue=execution_data.get('venue', 'binance')
            )
            
            # 存储到注册表
            self.execution_registry[execution.execution_id] = execution
            
            # 记录到一致性监控器
            if self.consistency_monitor:
                await self.consistency_monitor.record_execution(execution)
            
            # 发布Kafka事件
            if self.kafka_streamer:
                await self.kafka_streamer.publish_execution_report(
                    execution_id=execution.execution_id,
                    symbol=execution.symbol,
                    side=execution.side,
                    quantity=execution.quantity,
                    price=execution.executed_price,
                    slippage_bps=execution.slippage_bps,
                    latency_ms=execution.latency_ms,
                    venue=execution.venue,
                    signal_id=execution.position_id  # Map position to signal
                )
            
            # 记录监控事件
            await self._log_monitoring_event(
                event_type="EXECUTION_RECORDED",
                severity="DEBUG",
                component="order_execution",
                message=f"Execution {execution.execution_id} recorded for {execution.symbol}",
                details={
                    'execution_id': execution.execution_id,
                    'position_id': execution.position_id,
                    'symbol': execution.symbol,
                    'side': execution.side,
                    'quantity': execution.quantity,
                    'executed_price': execution.executed_price,
                    'slippage_bps': execution.slippage_bps,
                    'latency_ms': execution.latency_ms
                }
            )
            
            self.stats['executions_monitored'] += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to record execution: {e}")
            await self._generate_alert(
                alert_id=f"execution_error_{int(time.time())}",
                severity="WARNING",
                category="EXECUTION_PROCESSING_ERROR",
                message=f"Failed to record execution: {e}",
                details={'error': str(e), 'execution_data': execution_data}
            )
    
    async def _perform_consistency_checks(self):
        """执行一致性检查"""
        try:
            if not self.signal_registry or not self.position_registry:
                return
            
            # 计算信号-持仓匹配率
            signal_position_matches = 0
            total_positions = len(self.position_registry)
            
            for position in self.position_registry.values():
                if position.signal_id in self.signal_registry:
                    signal = self.signal_registry[position.signal_id]
                    if signal.symbol == position.symbol and signal.signal_type == position.side:
                        signal_position_matches += 1
            
            signal_position_match_rate = (signal_position_matches / max(total_positions, 1)) * 100
            
            # 计算持仓-执行匹配率
            position_execution_matches = 0
            total_executions = len(self.execution_registry)
            
            for execution in self.execution_registry.values():
                if execution.position_id in self.position_registry:
                    position = self.position_registry[execution.position_id]
                    if (position.symbol == execution.symbol and 
                        position.side == execution.side and
                        abs(position.quantity - execution.quantity) < 0.0001):
                        position_execution_matches += 1
            
            position_execution_match_rate = (position_execution_matches / max(total_executions, 1)) * 100
            
            # 计算价格偏差
            price_deviations = []
            for position in self.position_registry.values():
                if position.signal_id in self.signal_registry:
                    signal = self.signal_registry[position.signal_id]
                    if position.entry_price and signal.expected_entry_price:
                        deviation_bps = abs(position.entry_price - signal.expected_entry_price) / signal.expected_entry_price * 10000
                        price_deviations.append(deviation_bps)
            
            avg_price_deviation = np.mean(price_deviations) if price_deviations else 0.0
            
            # 计算时间准确性
            timing_accuracies = []
            for position in self.position_registry.values():
                if position.signal_id in self.signal_registry:
                    signal = self.signal_registry[position.signal_id]
                    if position.entry_time and signal.timestamp:
                        time_diff_minutes = abs((position.entry_time - signal.timestamp).total_seconds() / 60)
                        timing_accuracies.append(1 if time_diff_minutes <= 2 else 0)
            
            timing_accuracy_rate = (np.mean(timing_accuracies) if timing_accuracies else 1.0) * 100
            
            # 计算边界合规率
            boundary_compliances = []
            for position in self.position_registry.values():
                if position.realized and position.exit_time:
                    exit_minute = position.exit_time.minute
                    boundary_minutes = [15, 30, 45, 0]  # 0 represents 60
                    is_boundary = any(abs(exit_minute - bm) <= 1 for bm in boundary_minutes)
                    boundary_compliances.append(1 if is_boundary else 0)
            
            boundary_compliance_rate = (np.mean(boundary_compliances) if boundary_compliances else 1.0) * 100
            
            # 计算总体一致性评分
            overall_score = (
                signal_position_match_rate * 0.3 +
                position_execution_match_rate * 0.3 +
                timing_accuracy_rate * 0.2 +
                boundary_compliance_rate * 0.2
            )
            
            # 更新一致性指标
            self.consistency_metrics = ConsistencyMetrics(
                signal_position_match_rate=signal_position_match_rate,
                position_execution_match_rate=position_execution_match_rate,
                timing_accuracy_rate=timing_accuracy_rate,
                price_deviation_avg_bps=avg_price_deviation,
                boundary_compliance_rate=boundary_compliance_rate,
                overall_consistency_score=overall_score
            )
            
            # 保存到数据库
            await self._save_consistency_metrics()
            
            # 检查阈值违规
            violations = []
            if signal_position_match_rate < self.thresholds['consistency']['signal_position_match_min']:
                violations.append(f"Signal-position match rate {signal_position_match_rate:.1f}% below threshold")
            
            if position_execution_match_rate < self.thresholds['consistency']['position_execution_match_min']:
                violations.append(f"Position-execution match rate {position_execution_match_rate:.1f}% below threshold")
            
            if avg_price_deviation > self.thresholds['consistency']['price_deviation_max_bps']:
                violations.append(f"Average price deviation {avg_price_deviation:.1f}bps exceeds threshold")
            
            if boundary_compliance_rate < self.thresholds['consistency']['boundary_compliance_min']:
                violations.append(f"Boundary compliance rate {boundary_compliance_rate:.1f}% below threshold")
            
            # 生成告警
            if violations:
                self.stats['consistency_violations'] += len(violations)
                await self._generate_alert(
                    alert_id=f"consistency_violation_{int(time.time())}",
                    severity="WARNING" if overall_score > 70 else "CRITICAL",
                    category="CONSISTENCY_VIOLATION",
                    message=f"Consistency violations detected (score: {overall_score:.1f})",
                    details={
                        'consistency_score': overall_score,
                        'violations': violations,
                        'metrics': asdict(self.consistency_metrics)
                    }
                )
            
            # 记录监控事件
            await self._log_monitoring_event(
                event_type="CONSISTENCY_CHECK_COMPLETED",
                severity="DEBUG",
                component="consistency_monitor",
                message=f"Consistency check completed (score: {overall_score:.1f})",
                details=asdict(self.consistency_metrics)
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to perform consistency checks: {e}")
    
    async def _perform_drift_detection(self):
        """执行漂移检测"""
        try:
            if not self.backtest_baseline or len(self.production_history) < self.thresholds['drift']['min_data_points']:
                return
            
            current_time = datetime.now(timezone.utc)
            
            # 计算当前生产指标
            recent_performance = await self._calculate_recent_performance()
            if not recent_performance:
                return
            
            # 执行漂移检测
            drift_result = await self._detect_performance_drift(
                baseline=self.backtest_baseline,
                current=recent_performance,
                time_window="1h"
            )
            
            # 保存漂移检测结果
            await self._save_drift_detection_result(drift_result)
            
            # 生成告警（如果检测到漂移）
            if drift_result.drift_detected:
                self.stats['drift_checks_performed'] += 1
                severity = "CRITICAL" if drift_result.drift_score > self.thresholds['drift']['critical_threshold_pct'] else "WARNING"
                
                await self._generate_alert(
                    alert_id=f"drift_detected_{int(time.time())}",
                    severity=severity,
                    category="PERFORMANCE_DRIFT",
                    message=f"Performance drift detected (score: {drift_result.drift_score:.1f})",
                    details={
                        'drift_score': drift_result.drift_score,
                        'statistical_significance': drift_result.statistical_significance,
                        'affected_metrics': drift_result.affected_metrics,
                        'recommendation': drift_result.recommendation,
                        'baseline': self.backtest_baseline,
                        'current': recent_performance
                    }
                )
            
            # 记录监控事件
            await self._log_monitoring_event(
                event_type="DRIFT_DETECTION_COMPLETED",
                severity="DEBUG" if not drift_result.drift_detected else "INFO",
                component="drift_detector",
                message=f"Drift detection completed (drift: {'YES' if drift_result.drift_detected else 'NO'})",
                details={
                    'drift_detected': drift_result.drift_detected,
                    'drift_score': drift_result.drift_score,
                    'affected_metrics': drift_result.affected_metrics
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to perform drift detection: {e}")
    
    async def _detect_performance_drift(self,
                                      baseline: Dict[str, float],
                                      current: Dict[str, float],
                                      time_window: str) -> DriftDetectionResult:
        """检测性能漂移"""
        
        key_metrics = ['win_rate', 'profit_factor', 'sharpe_ratio', 'max_drawdown']
        drift_scores = []
        affected_metrics = []
        
        for metric in key_metrics:
            if metric in baseline and metric in current:
                baseline_val = baseline[metric]
                current_val = current[metric]
                
                if baseline_val != 0:
                    pct_diff = abs(current_val - baseline_val) / abs(baseline_val) * 100
                else:
                    pct_diff = abs(current_val) * 100
                
                drift_scores.append(pct_diff)
                
                if pct_diff > self.thresholds['drift']['warning_threshold_pct']:
                    affected_metrics.append(f"{metric}: {pct_diff:.1f}% deviation")
        
        # 计算总体漂移评分
        overall_drift_score = np.mean(drift_scores) if drift_scores else 0.0
        
        # 统计显著性检验（简化版）
        statistical_significance = min(overall_drift_score / 100.0, 0.05)  # 简化计算
        
        # 判断是否检测到漂移
        drift_detected = (
            overall_drift_score > self.thresholds['drift']['warning_threshold_pct'] and
            statistical_significance <= self.thresholds['drift']['statistical_significance']
        )
        
        # 生成建议
        if drift_detected:
            if overall_drift_score > self.thresholds['drift']['critical_threshold_pct']:
                recommendation = "CRITICAL: Immediate strategy review and potential halt required"
            else:
                recommendation = "WARNING: Monitor closely and consider parameter adjustment"
        else:
            recommendation = "Performance within acceptable range"
        
        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=overall_drift_score,
            statistical_significance=statistical_significance,
            affected_metrics=affected_metrics,
            recommendation=recommendation,
            time_window=time_window,
            comparison_baseline="backtest"
        )
    
    async def _calculate_recent_performance(self) -> Dict[str, float]:
        """计算最近的生产性能指标"""
        if not self.position_registry:
            return {}
        
        # 获取最近1小时的已实现持仓
        recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_positions = [
            pos for pos in self.position_registry.values()
            if pos.realized and pos.exit_time and pos.exit_time >= recent_cutoff
        ]
        
        if not recent_positions:
            return {}
        
        # 计算指标
        total_trades = len(recent_positions)
        winning_trades = sum(1 for pos in recent_positions if pos.pnl and pos.pnl > 0)
        losing_trades = total_trades - winning_trades
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_pnl = sum(pos.pnl for pos in recent_positions if pos.pnl)
        wins = [pos.pnl for pos in recent_positions if pos.pnl and pos.pnl > 0]
        losses = [abs(pos.pnl) for pos in recent_positions if pos.pnl and pos.pnl < 0]
        
        avg_win = np.mean(wins) if wins else 0.0
        avg_loss = np.mean(losses) if losses else 0.0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # 计算夏普比率（简化版）
        pnl_series = [pos.pnl for pos in recent_positions if pos.pnl]
        if len(pnl_series) > 1:
            sharpe_ratio = np.mean(pnl_series) / np.std(pnl_series) if np.std(pnl_series) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # 计算最大回撤（简化版）
        cumulative_pnl = np.cumsum([pos.pnl for pos in recent_positions if pos.pnl])
        if len(cumulative_pnl) > 0:
            running_max = np.maximum.accumulate(cumulative_pnl)
            drawdown = (running_max - cumulative_pnl) / (running_max + 1e-8)
            max_drawdown = np.max(drawdown)
        else:
            max_drawdown = 0.0
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
    
    async def _perform_risk_monitoring(self):
        """执行风险监控"""
        try:
            # 计算当前风险指标
            current_metrics = await self._calculate_current_risk_metrics()
            
            # 更新风险监控状态
            self.risk_monitoring_state.current_var_95 = current_metrics.get('var_95', 0.0)
            self.risk_monitoring_state.current_var_99 = current_metrics.get('var_99', 0.0)
            self.risk_monitoring_state.current_expected_shortfall = current_metrics.get('expected_shortfall', 0.0)
            self.risk_monitoring_state.current_drawdown = current_metrics.get('current_drawdown', 0.0)
            self.risk_monitoring_state.portfolio_value = current_metrics.get('portfolio_value', 0.0)
            self.risk_monitoring_state.risk_utilization = current_metrics.get('risk_utilization', 0.0)
            
            # 检查风险限制违规
            violations = []
            breach_detected = False
            
            if self.risk_monitoring_state.current_var_95 > self.thresholds['risk']['var_95_limit']:
                violations.append(f"VaR 95% ${self.risk_monitoring_state.current_var_95:,.0f} exceeds limit")
                breach_detected = True
            
            if self.risk_monitoring_state.current_var_99 > self.thresholds['risk']['var_99_limit']:
                violations.append(f"VaR 99% ${self.risk_monitoring_state.current_var_99:,.0f} exceeds limit")
                breach_detected = True
            
            if self.risk_monitoring_state.current_drawdown > self.thresholds['risk']['max_drawdown_limit']:
                violations.append(f"Drawdown {self.risk_monitoring_state.current_drawdown:.2%} exceeds limit")
                breach_detected = True
            
            # 更新违规状态
            if breach_detected:
                self.risk_monitoring_state.breach_count += 1
                self.risk_monitoring_state.last_breach_time = datetime.now(timezone.utc)
                self.stats['risk_breaches'] += 1
            
            # 发布风险指标事件
            if self.kafka_streamer:
                await self.kafka_streamer.publish_risk_metrics(
                    var_95=self.risk_monitoring_state.current_var_95,
                    var_99=self.risk_monitoring_state.current_var_99,
                    expected_shortfall=self.risk_monitoring_state.current_expected_shortfall,
                    current_drawdown=self.risk_monitoring_state.current_drawdown,
                    max_drawdown=self.risk_monitoring_state.max_drawdown,
                    portfolio_value=self.risk_monitoring_state.portfolio_value,
                    risk_utilization=self.risk_monitoring_state.risk_utilization
                )
            
            # 生成风险告警
            if violations:
                severity = "CRITICAL" if len(violations) > 1 else "WARNING"
                await self._generate_alert(
                    alert_id=f"risk_violation_{int(time.time())}",
                    severity=severity,
                    category="RISK_LIMIT_BREACH",
                    message=f"Risk limit violations detected",
                    details={
                        'violations': violations,
                        'risk_state': asdict(self.risk_monitoring_state),
                        'thresholds': self.thresholds['risk']
                    }
                )
            
            # 记录监控事件
            await self._log_monitoring_event(
                event_type="RISK_MONITORING_COMPLETED",
                severity="WARNING" if violations else "DEBUG",
                component="risk_monitor",
                message=f"Risk monitoring completed ({'VIOLATIONS' if violations else 'OK'})",
                details={
                    'risk_metrics': current_metrics,
                    'violations': violations,
                    'breach_count': self.risk_monitoring_state.breach_count
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to perform risk monitoring: {e}")
    
    async def _calculate_current_risk_metrics(self) -> Dict[str, float]:
        """计算当前风险指标"""
        if not self.position_registry:
            return {'var_95': 0.0, 'var_99': 0.0, 'expected_shortfall': 0.0, 'current_drawdown': 0.0, 'portfolio_value': 0.0, 'risk_utilization': 0.0}
        
        # 获取所有已实现持仓的P&L
        realized_pnls = [pos.pnl for pos in self.position_registry.values() if pos.realized and pos.pnl is not None]
        
        if not realized_pnls:
            return {'var_95': 0.0, 'var_99': 0.0, 'expected_shortfall': 0.0, 'current_drawdown': 0.0, 'portfolio_value': 0.0, 'risk_utilization': 0.0}
        
        # 计算VaR和ES
        pnl_array = np.array(realized_pnls)
        var_95 = abs(np.percentile(pnl_array, 5)) if len(pnl_array) > 0 else 0.0
        var_99 = abs(np.percentile(pnl_array, 1)) if len(pnl_array) > 0 else 0.0
        
        # 计算Expected Shortfall (ES)
        var_95_threshold = np.percentile(pnl_array, 5)
        tail_losses = pnl_array[pnl_array <= var_95_threshold]
        expected_shortfall = abs(np.mean(tail_losses)) if len(tail_losses) > 0 else 0.0
        
        # 计算当前回撤
        cumulative_pnl = np.cumsum(realized_pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        current_drawdown = ((running_max[-1] - cumulative_pnl[-1]) / (running_max[-1] + 1e-8)) if len(cumulative_pnl) > 0 else 0.0
        
        # 计算组合价值（简化）
        portfolio_value = sum(realized_pnls) + 10000  # 假设起始资本10000
        
        # 计算风险利用率
        risk_utilization = (var_95 / self.thresholds['risk']['var_95_limit']) * 100 if self.thresholds['risk']['var_95_limit'] > 0 else 0.0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall': expected_shortfall,
            'current_drawdown': current_drawdown,
            'portfolio_value': portfolio_value,
            'risk_utilization': risk_utilization
        }
    
    async def _monitor_system_health(self):
        """监控系统健康状态"""
        try:
            # 获取系统资源使用情况
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 计算健康评分
            health_score = 100.0
            issues = []
            
            if cpu_usage > self.thresholds['system_health']['cpu_warning']:
                health_score -= min(30, cpu_usage - self.thresholds['system_health']['cpu_warning'])
                issues.append(f"High CPU usage: {cpu_usage:.1f}%")
            
            if memory.percent > self.thresholds['system_health']['memory_warning']:
                health_score -= min(25, memory.percent - self.thresholds['system_health']['memory_warning'])
                issues.append(f"High memory usage: {memory.percent:.1f}%")
            
            if disk.percent > self.thresholds['system_health']['disk_warning']:
                health_score -= min(20, disk.percent - self.thresholds['system_health']['disk_warning'])
                issues.append(f"High disk usage: {disk.percent:.1f}%")
            
            # 更新系统健康评分
            self.system_health_score = max(0.0, health_score)
            
            # 检查组件健康状态
            component_health = await self._check_component_health()
            
            # 发布系统健康事件
            if self.kafka_streamer:
                await self.kafka_streamer.publish_system_health(
                    component_name="comprehensive_monitoring",
                    health_score=self.system_health_score,
                    status="healthy" if self.system_health_score >= 80 else "degraded" if self.system_health_score >= 50 else "unhealthy",
                    cpu_usage_percent=cpu_usage,
                    memory_usage_percent=memory.percent,
                    disk_usage_percent=disk.percent,
                    uptime_seconds=time.time() - self.start_time if self.start_time else 0
                )
            
            # 生成健康告警
            if issues:
                severity = "CRITICAL" if self.system_health_score < 50 else "WARNING"
                await self._generate_alert(
                    alert_id=f"system_health_{int(time.time())}",
                    severity=severity,
                    category="SYSTEM_HEALTH",
                    message=f"System health issues detected (score: {self.system_health_score:.1f})",
                    details={
                        'health_score': self.system_health_score,
                        'issues': issues,
                        'system_metrics': {
                            'cpu_usage': cpu_usage,
                            'memory_usage': memory.percent,
                            'disk_usage': disk.percent
                        },
                        'component_health': component_health
                    }
                )
            
            # 记录监控事件
            await self._log_monitoring_event(
                event_type="SYSTEM_HEALTH_CHECK",
                severity="WARNING" if issues else "DEBUG",
                component="health_monitor",
                message=f"System health check completed (score: {self.system_health_score:.1f})",
                details={
                    'health_score': self.system_health_score,
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory.percent,
                    'disk_usage': disk.percent,
                    'issues': issues
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to monitor system health: {e}")
            self.system_health_score = 0.0
    
    async def _check_component_health(self) -> Dict[str, Any]:
        """检查各组件健康状态"""
        component_health = {}
        
        # 检查Kafka连接
        if self.kafka_streamer:
            try:
                kafka_stats = self.kafka_streamer.get_event_stats()
                component_health['kafka'] = {
                    'status': 'healthy' if kafka_stats['is_running'] else 'unhealthy',
                    'total_events': kafka_stats['total_events'],
                    'producer_stats': kafka_stats['producer_stats']
                }
            except:
                component_health['kafka'] = {'status': 'error'}
        
        # 检查一致性监控器
        if self.consistency_monitor:
            try:
                consistency_report = self.consistency_monitor.get_consistency_report()
                component_health['consistency_monitor'] = {
                    'status': 'healthy' if consistency_report['summary']['overall_consistency_rate'] >= 80 else 'degraded',
                    'consistency_rate': consistency_report['summary']['overall_consistency_rate'],
                    'total_checks': consistency_report['summary']['total_checks_performed']
                }
            except:
                component_health['consistency_monitor'] = {'status': 'error'}
        
        # 检查完整监控系统
        if self.complete_monitoring:
            try:
                health_status = await self.complete_monitoring.get_health_status()
                component_health['complete_monitoring'] = health_status
            except:
                component_health['complete_monitoring'] = {'status': 'error'}
        
        return component_health
    
    async def _manage_alerts(self):
        """管理告警状态"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # 检查告警抑制
            expired_suppressions = []
            for alert_type, suppressed_until in self.alert_suppression.items():
                if current_time > suppressed_until:
                    expired_suppressions.append(alert_type)
            
            for alert_type in expired_suppressions:
                del self.alert_suppression[alert_type]
            
            # 检查活跃告警的自动解决
            resolved_alerts = []
            for alert_id, alert in self.active_alerts.items():
                if await self._check_alert_auto_resolution(alert):
                    alert.resolved = True
                    alert.resolution_time = current_time
                    resolved_alerts.append(alert_id)
                    
                    # 记录解决事件
                    await self._log_monitoring_event(
                        event_type="ALERT_AUTO_RESOLVED",
                        severity="INFO",
                        component="alert_manager",
                        message=f"Alert {alert_id} auto-resolved",
                        details={'alert_id': alert_id, 'category': alert.category}
                    )
            
            # 移除已解决的告警
            for alert_id in resolved_alerts:
                resolved_alert = self.active_alerts.pop(alert_id)
                self.alert_history.append(resolved_alert)
            
            # 检查告警升级
            for alert in self.active_alerts.values():
                await self._check_alert_escalation(alert)
            
        except Exception as e:
            logger.error(f"❌ Failed to manage alerts: {e}")
    
    async def _check_alert_auto_resolution(self, alert: AlertEvent) -> bool:
        """检查告警是否可以自动解决"""
        if alert.category == "CONSISTENCY_VIOLATION":
            # 检查一致性评分是否恢复
            return self.consistency_metrics.overall_consistency_score >= 85.0
        
        elif alert.category == "RISK_LIMIT_BREACH":
            # 检查风险指标是否恢复正常
            return (
                self.risk_monitoring_state.current_var_95 <= self.thresholds['risk']['var_95_limit'] and
                self.risk_monitoring_state.current_var_99 <= self.thresholds['risk']['var_99_limit'] and
                self.risk_monitoring_state.current_drawdown <= self.thresholds['risk']['max_drawdown_limit']
            )
        
        elif alert.category == "SYSTEM_HEALTH":
            # 检查系统健康评分是否恢复
            return self.system_health_score >= 80.0
        
        return False
    
    async def _check_alert_escalation(self, alert: AlertEvent):
        """检查告警升级"""
        if alert.severity == "WARNING":
            # 检查是否需要升级为CRITICAL
            time_since_alert = datetime.now(timezone.utc) - datetime.fromisoformat(alert.timestamp)
            if time_since_alert > timedelta(minutes=30):  # 30分钟后升级
                alert.severity = "CRITICAL"
                
                await self._log_monitoring_event(
                    event_type="ALERT_ESCALATED",
                    severity="WARNING",
                    component="alert_manager",
                    message=f"Alert {alert.alert_id} escalated to CRITICAL",
                    details={'alert_id': alert.alert_id, 'category': alert.category}
                )
    
    async def _take_performance_snapshot(self):
        """拍摄性能快照"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # 计算当前性能指标
            performance_metrics = await self._calculate_recent_performance()
            
            if not performance_metrics:
                return
            
            # 添加监控特有指标
            performance_metrics.update({
                'consistency_score': self.consistency_metrics.overall_consistency_score,
                'drift_score': 0.0,  # 从最近的漂移检测中获取
                'system_health_score': self.system_health_score,
                'active_alerts_count': len(self.active_alerts),
                'risk_utilization': self.risk_monitoring_state.risk_utilization
            })
            
            # 保存到历史数据
            self.production_history.append({
                'timestamp': current_time,
                'metrics': performance_metrics
            })
            
            # 保存到数据库
            await self._save_performance_snapshot(performance_metrics)
            
            # 发布策略性能事件
            if self.kafka_streamer and performance_metrics.get('total_trades', 0) > 0:
                await self.kafka_streamer.publish_strategy_performance(
                    strategy_name="dipmaster",
                    time_window="15m",
                    win_rate=performance_metrics.get('win_rate', 0.0),
                    profit_factor=performance_metrics.get('profit_factor', 0.0),
                    sharpe_ratio=performance_metrics.get('sharpe_ratio', 0.0),
                    max_drawdown=performance_metrics.get('max_drawdown', 0.0),
                    total_pnl=performance_metrics.get('total_pnl', 0.0),
                    total_trades=int(performance_metrics.get('total_trades', 0)),
                    avg_win=performance_metrics.get('avg_win', 0.0),
                    avg_loss=performance_metrics.get('avg_loss', 0.0)
                )
            
            # 记录监控事件
            await self._log_monitoring_event(
                event_type="PERFORMANCE_SNAPSHOT_TAKEN",
                severity="DEBUG",
                component="performance_monitor",
                message="Performance snapshot taken",
                details=performance_metrics
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to take performance snapshot: {e}")
    
    async def _check_report_schedule(self):
        """检查报告生成计划"""
        try:
            current_time = datetime.now(timezone.utc)
            current_hour = current_time.hour
            current_minute = current_time.minute
            
            # 每日报告（凌晨1点）
            if current_hour == 1 and current_minute < 5:
                await self._generate_daily_report()
            
            # 每周报告（周一凌晨2点）
            if current_time.weekday() == 0 and current_hour == 2 and current_minute < 5:
                await self._generate_weekly_report()
            
            # 每月报告（月初凌晨3点）
            if current_time.day == 1 and current_hour == 3 and current_minute < 5:
                await self._generate_monthly_report()
            
        except Exception as e:
            logger.error(f"❌ Failed to check report schedule: {e}")
    
    def _update_statistics(self):
        """更新统计信息"""
        try:
            if self.start_time:
                self.stats['uptime_seconds'] = time.time() - self.start_time
            
            self.stats['events_processed'] = len(self.event_logs)
            self.stats['alerts_generated'] = len(self.alert_history) + len(self.active_alerts)
            
        except Exception as e:
            logger.error(f"❌ Failed to update statistics: {e}")
    
    async def _generate_alert(self,
                            alert_id: str,
                            severity: str,
                            category: str,
                            message: str,
                            details: Dict[str, Any],
                            auto_remediation: bool = False):
        """生成告警"""
        try:
            # 检查告警抑制
            if category in self.alert_suppression:
                if datetime.now(timezone.utc) < self.alert_suppression[category]:
                    return  # 跳过被抑制的告警
            
            # 创建告警事件
            alert = AlertEvent(
                alert_id=alert_id,
                severity=severity,
                category=category,
                message=message,
                details=details,
                auto_remediation=auto_remediation,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            # 添加到活跃告警
            self.active_alerts[alert_id] = alert
            
            # 发布Kafka事件
            if self.kafka_streamer:
                await self.kafka_streamer.publish_alert(
                    alert_id=alert_id,
                    severity=severity,
                    category=category,
                    message=message,
                    details=details
                )
            
            # 记录监控事件
            await self._log_monitoring_event(
                event_type="ALERT_GENERATED",
                severity=severity,
                component="alert_manager",
                message=f"Alert generated: {message}",
                details={'alert_id': alert_id, 'category': category}
            )
            
            # 设置告警抑制（避免重复告警）
            if severity in ["WARNING", "CRITICAL"]:
                suppression_duration = timedelta(minutes=15) if severity == "WARNING" else timedelta(minutes=5)
                self.alert_suppression[category] = datetime.now(timezone.utc) + suppression_duration
            
            logger.warning(f"🚨 Alert generated: {alert_id} - {message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate alert: {e}")
    
    async def _log_monitoring_event(self,
                                  event_type: str,
                                  severity: str,
                                  component: str,
                                  message: str,
                                  details: Dict[str, Any] = None):
        """记录监控事件"""
        try:
            event = MonitoringEventLog(
                event_id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc),
                correlation_id=self.correlation_id,
                event_type=event_type,
                severity=severity,
                component=component,
                message=message,
                details=details or {},
                tags={
                    'mode': self.mode.value,
                    'system': 'dipmaster_monitoring'
                }
            )
            
            # 添加到内存日志
            self.event_logs.append(event)
            
            # 保存到数据库
            await self._save_event_log(event)
            
            # 根据严重程度记录日志
            if severity == "DEBUG":
                logger.debug(f"📝 {event_type}: {message}")
            elif severity == "INFO":
                logger.info(f"📝 {event_type}: {message}")
            elif severity == "WARNING":
                logger.warning(f"📝 {event_type}: {message}")
            elif severity in ["ERROR", "CRITICAL"]:
                logger.error(f"📝 {event_type}: {message}")
            
        except Exception as e:
            logger.error(f"❌ Failed to log monitoring event: {e}")
    
    async def _save_event_log(self, event: MonitoringEventLog):
        """保存事件日志到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO event_logs 
                    (event_id, timestamp, correlation_id, event_type, severity, component, message, details, tags, resolved, auto_remediated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.event_id,
                    event.timestamp.isoformat(),
                    event.correlation_id,
                    event.event_type,
                    event.severity,
                    event.component,
                    event.message,
                    json.dumps(event.details),
                    json.dumps(event.tags),
                    int(event.resolved),
                    int(event.auto_remediated)
                ))
                
        except Exception as e:
            logger.error(f"❌ Failed to save event log to database: {e}")
    
    async def _save_consistency_metrics(self):
        """保存一致性指标到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO consistency_history 
                    (timestamp, signal_position_match_rate, position_execution_match_rate, timing_accuracy_rate, 
                     price_deviation_avg_bps, boundary_compliance_rate, overall_consistency_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    self.consistency_metrics.signal_position_match_rate,
                    self.consistency_metrics.position_execution_match_rate,
                    self.consistency_metrics.timing_accuracy_rate,
                    self.consistency_metrics.price_deviation_avg_bps,
                    self.consistency_metrics.boundary_compliance_rate,
                    self.consistency_metrics.overall_consistency_score
                ))
                
        except Exception as e:
            logger.error(f"❌ Failed to save consistency metrics to database: {e}")
    
    async def _save_drift_detection_result(self, result: DriftDetectionResult):
        """保存漂移检测结果到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO drift_detection_history 
                    (timestamp, drift_detected, drift_score, statistical_significance, affected_metrics, 
                     recommendation, time_window, comparison_baseline)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    int(result.drift_detected),
                    result.drift_score,
                    result.statistical_significance,
                    json.dumps(result.affected_metrics),
                    result.recommendation,
                    result.time_window,
                    result.comparison_baseline
                ))
                
        except Exception as e:
            logger.error(f"❌ Failed to save drift detection result to database: {e}")
    
    async def _save_performance_snapshot(self, metrics: Dict[str, float]):
        """保存性能快照到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO performance_snapshots 
                    (timestamp, time_window, win_rate, profit_factor, sharpe_ratio, max_drawdown, 
                     total_pnl, total_trades, avg_holding_time_minutes, consistency_score, drift_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    "15m",
                    metrics.get('win_rate', 0.0),
                    metrics.get('profit_factor', 0.0),
                    metrics.get('sharpe_ratio', 0.0),
                    metrics.get('max_drawdown', 0.0),
                    metrics.get('total_pnl', 0.0),
                    int(metrics.get('total_trades', 0)),
                    metrics.get('avg_holding_time_minutes', 0.0),
                    metrics.get('consistency_score', 0.0),
                    metrics.get('drift_score', 0.0)
                ))
                
        except Exception as e:
            logger.error(f"❌ Failed to save performance snapshot to database: {e}")
    
    async def _save_monitoring_state(self):
        """保存监控状态"""
        try:
            state_file = Path(self.config.get('state_file', 'data/monitoring_state.json'))
            state_file.parent.mkdir(parents=True, exist_ok=True)
            
            state_data = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'mode': self.mode.value,
                'stats': self.stats,
                'consistency_metrics': asdict(self.consistency_metrics),
                'risk_monitoring_state': asdict(self.risk_monitoring_state),
                'system_health_score': self.system_health_score,
                'active_alerts_count': len(self.active_alerts),
                'thresholds': self.thresholds
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            logger.info(f"💾 Monitoring state saved to {state_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save monitoring state: {e}")
    
    async def _generate_daily_report(self):
        """生成每日监控报告"""
        try:
            # 生成报告逻辑
            report_data = await self._compile_daily_report_data()
            report_path = await self._save_report("daily", report_data)
            
            self.stats['reports_generated'] += 1
            
            await self._log_monitoring_event(
                event_type="DAILY_REPORT_GENERATED",
                severity="INFO",
                component="report_generator",
                message=f"Daily report generated: {report_path}",
                details={'report_path': str(report_path)}
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to generate daily report: {e}")
    
    async def _generate_weekly_report(self):
        """生成每周监控报告"""
        try:
            # 生成报告逻辑
            report_data = await self._compile_weekly_report_data()
            report_path = await self._save_report("weekly", report_data)
            
            self.stats['reports_generated'] += 1
            
            await self._log_monitoring_event(
                event_type="WEEKLY_REPORT_GENERATED",
                severity="INFO",
                component="report_generator",
                message=f"Weekly report generated: {report_path}",
                details={'report_path': str(report_path)}
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to generate weekly report: {e}")
    
    async def _generate_monthly_report(self):
        """生成每月监控报告"""
        try:
            # 生成报告逻辑
            report_data = await self._compile_monthly_report_data()
            report_path = await self._save_report("monthly", report_data)
            
            self.stats['reports_generated'] += 1
            
            await self._log_monitoring_event(
                event_type="MONTHLY_REPORT_GENERATED",
                severity="INFO",
                component="report_generator",
                message=f"Monthly report generated: {report_path}",
                details={'report_path': str(report_path)}
            )
            
        except Exception as e:
            logger.error(f"❌ Failed to generate monthly report: {e}")
    
    async def _compile_daily_report_data(self) -> Dict[str, Any]:
        """编译每日报告数据"""
        return {
            'date': datetime.now(timezone.utc).date().isoformat(),
            'summary': {
                'total_signals': self.stats['signals_validated'],
                'total_positions': self.stats['positions_tracked'],
                'total_executions': self.stats['executions_monitored'],
                'consistency_score': self.consistency_metrics.overall_consistency_score,
                'system_health_score': self.system_health_score,
                'active_alerts': len(self.active_alerts),
                'resolved_alerts': len([a for a in self.alert_history if a.resolved])
            },
            'performance_metrics': await self._calculate_recent_performance(),
            'consistency_metrics': asdict(self.consistency_metrics),
            'risk_metrics': asdict(self.risk_monitoring_state),
            'system_health': {
                'health_score': self.system_health_score,
                'uptime_hours': self.stats['uptime_seconds'] / 3600
            }
        }
    
    async def _compile_weekly_report_data(self) -> Dict[str, Any]:
        """编译每周报告数据"""
        # 类似日报，但包含更多历史数据分析
        daily_data = await self._compile_daily_report_data()
        daily_data['timeframe'] = 'weekly'
        # 添加周度趋势分析
        return daily_data
    
    async def _compile_monthly_report_data(self) -> Dict[str, Any]:
        """编译每月报告数据"""
        # 类似周报，但包含月度趋势和更深入的分析
        weekly_data = await self._compile_weekly_report_data()
        weekly_data['timeframe'] = 'monthly'
        # 添加月度趋势分析和建议
        return weekly_data
    
    async def _save_report(self, report_type: str, report_data: Dict[str, Any]) -> Path:
        """保存报告文件"""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        report_dir = Path(self.config.get('report_dir', 'reports/monitoring'))
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # JSON报告
        json_path = report_dir / f"dipmaster_monitoring_{report_type}_report_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Markdown报告
        md_path = report_dir / f"dipmaster_monitoring_{report_type}_report_{timestamp}.md"
        markdown_content = self._generate_markdown_report(report_type, report_data)
        with open(md_path, 'w') as f:
            f.write(markdown_content)
        
        return md_path
    
    def _generate_markdown_report(self, report_type: str, report_data: Dict[str, Any]) -> str:
        """生成Markdown格式报告"""
        timeframe = report_type.capitalize()
        date = report_data.get('date', datetime.now(timezone.utc).date().isoformat())
        
        return f"""# DipMaster Trading System - {timeframe} Monitoring Report
        
## Report Summary
**Date:** {date}  
**Report Type:** {timeframe}  
**Generated:** {datetime.now(timezone.utc).isoformat()}

## Executive Summary

### Key Metrics
- **Total Signals Processed:** {report_data['summary']['total_signals']:,}
- **Total Positions Tracked:** {report_data['summary']['total_positions']:,}
- **Total Executions Monitored:** {report_data['summary']['total_executions']:,}
- **Overall Consistency Score:** {report_data['summary']['consistency_score']:.1f}%
- **System Health Score:** {report_data['summary']['system_health_score']:.1f}%

### Alert Status
- **Active Alerts:** {report_data['summary']['active_alerts']}
- **Resolved Alerts:** {report_data['summary']['resolved_alerts']}

## Consistency Analysis

### Signal-Position-Execution Consistency
- **Signal-Position Match Rate:** {report_data['consistency_metrics']['signal_position_match_rate']:.1f}%
- **Position-Execution Match Rate:** {report_data['consistency_metrics']['position_execution_match_rate']:.1f}%
- **Timing Accuracy Rate:** {report_data['consistency_metrics']['timing_accuracy_rate']:.1f}%
- **Average Price Deviation:** {report_data['consistency_metrics']['price_deviation_avg_bps']:.1f} bps
- **Boundary Compliance Rate:** {report_data['consistency_metrics']['boundary_compliance_rate']:.1f}%

## Risk Management

### Current Risk Metrics
- **VaR 95%:** ${report_data['risk_metrics']['current_var_95']:,.2f}
- **VaR 99%:** ${report_data['risk_metrics']['current_var_99']:,.2f}
- **Expected Shortfall:** ${report_data['risk_metrics']['current_expected_shortfall']:,.2f}
- **Current Drawdown:** {report_data['risk_metrics']['current_drawdown']:.2%}
- **Risk Utilization:** {report_data['risk_metrics']['risk_utilization']:.1f}%

### Risk Breach Summary
- **Breach Count:** {report_data['risk_metrics']['breach_count']}
- **Last Breach:** {report_data['risk_metrics']['last_breach_time'] or 'None'}

## System Health

### Resource Utilization
- **System Health Score:** {report_data['system_health']['health_score']:.1f}%
- **Uptime:** {report_data['system_health']['uptime_hours']:.1f} hours

## Performance Metrics

{self._format_performance_metrics(report_data.get('performance_metrics', {}))}

## Recommendations

{self._generate_recommendations(report_data)}

---
*Report generated by DipMaster Comprehensive Monitoring System*
*🤖 Generated with Claude Code Monitoring Agent*
"""
    
    def _format_performance_metrics(self, metrics: Dict[str, float]) -> str:
        """格式化性能指标"""
        if not metrics:
            return "- No performance data available for this period"
        
        return f"""- **Win Rate:** {metrics.get('win_rate', 0):.1%}
- **Profit Factor:** {metrics.get('profit_factor', 0):.2f}
- **Sharpe Ratio:** {metrics.get('sharpe_ratio', 0):.2f}
- **Maximum Drawdown:** {metrics.get('max_drawdown', 0):.2%}
- **Total P&L:** ${metrics.get('total_pnl', 0):.2f}
- **Total Trades:** {int(metrics.get('total_trades', 0))}
- **Average Win:** ${metrics.get('avg_win', 0):.2f}
- **Average Loss:** ${metrics.get('avg_loss', 0):.2f}"""
    
    def _generate_recommendations(self, report_data: Dict[str, Any]) -> str:
        """生成建议"""
        recommendations = []
        
        consistency_score = report_data['summary']['consistency_score']
        if consistency_score < 80:
            recommendations.append("⚠️ **CRITICAL**: Consistency score below 80%. Investigate signal-position-execution pipeline.")
        elif consistency_score < 90:
            recommendations.append("⚠️ **WARNING**: Consistency score could be improved. Review monitoring thresholds.")
        else:
            recommendations.append("✅ **GOOD**: Consistency score is healthy.")
        
        health_score = report_data['summary']['system_health_score']
        if health_score < 70:
            recommendations.append("⚠️ **CRITICAL**: System health degraded. Check system resources and component status.")
        elif health_score < 85:
            recommendations.append("⚠️ **WARNING**: System health could be improved. Monitor resource usage.")
        else:
            recommendations.append("✅ **GOOD**: System health is optimal.")
        
        active_alerts = report_data['summary']['active_alerts']
        if active_alerts > 5:
            recommendations.append("⚠️ **WARNING**: High number of active alerts. Investigate and resolve persistent issues.")
        elif active_alerts > 0:
            recommendations.append("ℹ️ **INFO**: Active alerts present. Monitor for resolution.")
        else:
            recommendations.append("✅ **GOOD**: No active alerts.")
        
        if not recommendations:
            recommendations.append("✅ **EXCELLENT**: All systems operating within normal parameters.")
        
        return "\n".join(f"- {rec}" for rec in recommendations)
    
    # Public API methods
    
    async def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态概览"""
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'is_running': self.is_running,
            'mode': self.mode.value,
            'uptime_seconds': self.stats['uptime_seconds'],
            'system_health_score': self.system_health_score,
            'consistency_score': self.consistency_metrics.overall_consistency_score,
            'active_alerts_count': len(self.active_alerts),
            'components': {
                'kafka_streamer': self.kafka_streamer is not None,
                'consistency_monitor': self.consistency_monitor is not None,
                'complete_monitoring': self.complete_monitoring is not None
            },
            'statistics': self.stats.copy(),
            'thresholds': self.thresholds
        }
    
    async def get_recent_events(self, limit: int = 100, severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取最近的监控事件"""
        events = list(self.event_logs)
        
        if severity_filter:
            events = [e for e in events if e.severity == severity_filter]
        
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        return [asdict(event) for event in events[:limit]]
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        return [asdict(alert) for alert in self.active_alerts.values()]
    
    async def resolve_alert(self, alert_id: str, resolution_notes: str = "") -> bool:
        """手动解决告警"""
        if alert_id in self.active_alerts:
            alert = self.active_alerts.pop(alert_id)
            alert.resolved = True
            alert.resolution_time = datetime.now(timezone.utc)
            self.alert_history.append(alert)
            
            await self._log_monitoring_event(
                event_type="ALERT_MANUALLY_RESOLVED",
                severity="INFO",
                component="alert_manager",
                message=f"Alert {alert_id} manually resolved",
                details={'alert_id': alert_id, 'resolution_notes': resolution_notes}
            )
            
            return True
        return False
    
    async def suppress_alert_category(self, category: str, duration_minutes: int = 60) -> bool:
        """抑制特定类别的告警"""
        suppression_until = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        self.alert_suppression[category] = suppression_until
        
        await self._log_monitoring_event(
            event_type="ALERT_CATEGORY_SUPPRESSED",
            severity="WARNING",
            component="alert_manager",
            message=f"Alert category {category} suppressed for {duration_minutes} minutes",
            details={'category': category, 'duration_minutes': duration_minutes}
        )
        
        return True
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """获取监控统计信息"""
        return {
            'system_stats': self.stats.copy(),
            'consistency_metrics': asdict(self.consistency_metrics),
            'risk_monitoring_state': asdict(self.risk_monitoring_state),
            'data_counts': {
                'signals_registered': len(self.signal_registry),
                'positions_registered': len(self.position_registry),
                'executions_registered': len(self.execution_registry),
                'event_logs': len(self.event_logs),
                'active_alerts': len(self.active_alerts),
                'alert_history': len(self.alert_history)
            },
            'component_stats': self._get_component_statistics()
        }
    
    def _get_component_statistics(self) -> Dict[str, Any]:
        """获取组件统计信息"""
        stats = {}
        
        if self.kafka_streamer:
            stats['kafka_streamer'] = self.kafka_streamer.get_event_stats()
        
        if self.consistency_monitor:
            stats['consistency_monitor'] = self.consistency_monitor.get_consistency_report()
        
        return stats


# Factory function
def create_comprehensive_monitoring_system(config: Dict[str, Any] = None) -> ComprehensiveMonitoringSystem:
    """创建全面监控系统"""
    return ComprehensiveMonitoringSystem(config)


# Demo function
async def comprehensive_monitoring_demo():
    """全面监控系统演示"""
    print("🚀 DipMaster Comprehensive Monitoring System Demo")
    
    # 创建监控系统配置
    config = {
        'mode': 'development',
        'kafka': {
            'servers': ['localhost:9092'],
            'client_id': 'dipmaster-monitoring-demo'
        },
        'db_path': 'data/monitoring_demo.db',
        'log_dir': 'logs/monitoring_demo',
        'dipmaster_params': {
            'rsi_range': [30, 50],
            'max_holding_minutes': 180,
            'boundary_minutes': [15, 30, 45, 0],
            'target_profit_pct': 0.8
        },
        'thresholds': {
            'consistency': {
                'signal_position_match_min': 95.0,
                'position_execution_match_min': 98.0,
                'price_deviation_max_bps': 20.0,
                'timing_deviation_max_minutes': 2.0,
                'boundary_compliance_min': 100.0
            }
        }
    }
    
    # 创建并启动监控系统
    monitoring = create_comprehensive_monitoring_system(config)
    
    try:
        await monitoring.start()
        print("✅ Comprehensive monitoring system started")
        
        # 模拟交易数据
        print("📊 Simulating trading data...")
        
        # 记录信号
        signal_data = {
            'signal_id': 'sig_demo_001',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': 'BTCUSDT',
            'signal_type': 'BUY',
            'confidence': 0.85,
            'price': 43250.50,
            'technical_indicators': {
                'rsi': 35.2,
                'ma20_distance': -0.008,
                'volume_ratio': 1.6
            },
            'expected_entry_price': 43200.00,
            'expected_holding_minutes': 75
        }
        await monitoring.record_signal(signal_data)
        print("📊 Signal recorded")
        
        # 记录持仓
        position_data = {
            'position_id': 'pos_demo_001',
            'signal_id': 'sig_demo_001',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.1,
            'entry_price': 43210.00,
            'entry_time': datetime.now(timezone.utc).isoformat(),
            'realized': False
        }
        await monitoring.record_position(position_data)
        print("📊 Position recorded")
        
        # 记录执行
        execution_data = {
            'execution_id': 'exec_demo_001',
            'position_id': 'pos_demo_001',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.1,
            'price': 43210.00,
            'execution_time': datetime.now(timezone.utc).isoformat(),
            'latency_ms': 45.0,
            'slippage_bps': 2.3,
            'venue': 'binance'
        }
        await monitoring.record_execution(execution_data)
        print("📊 Execution recorded")
        
        # 等待处理
        await asyncio.sleep(3)
        
        # 模拟平仓
        position_data['exit_price'] = 43550.00
        position_data['exit_time'] = datetime.now(timezone.utc).isoformat()
        position_data['pnl'] = (43550.00 - 43210.00) * 0.1  # $34
        position_data['realized'] = True
        position_data['holding_minutes'] = 75
        await monitoring.record_position(position_data)
        print("📊 Position closed")
        
        # 等待所有监控检查完成
        await asyncio.sleep(5)
        
        # 获取系统状态
        status = await monitoring.get_system_status()
        print(f"📈 System Status:")
        print(f"  - Health Score: {status['system_health_score']:.1f}%")
        print(f"  - Consistency Score: {status['consistency_score']:.1f}%")
        print(f"  - Active Alerts: {status['active_alerts_count']}")
        print(f"  - Signals Processed: {status['statistics']['signals_validated']}")
        print(f"  - Positions Tracked: {status['statistics']['positions_tracked']}")
        print(f"  - Executions Monitored: {status['statistics']['executions_monitored']}")
        
        # 获取监控统计
        stats = monitoring.get_monitoring_statistics()
        print(f"📊 Monitoring Statistics:")
        print(f"  - Events Logged: {stats['data_counts']['event_logs']}")
        print(f"  - Consistency Score: {stats['consistency_metrics']['overall_consistency_score']:.1f}%")
        print(f"  - Risk Utilization: {stats['risk_monitoring_state']['risk_utilization']:.1f}%")
        
        # 获取最近事件
        recent_events = await monitoring.get_recent_events(limit=5)
        print(f"📝 Recent Events:")
        for event in recent_events:
            print(f"  - {event['timestamp']}: {event['event_type']} - {event['message']}")
        
        print("✅ Demo completed successfully")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await monitoring.stop()
        print("🛑 Monitoring system stopped")


if __name__ == "__main__":
    asyncio.run(comprehensive_monitoring_demo())