#!/usr/bin/env python3
"""
Simple Monitor - Phase 3 of Overfitting Optimization
简化监控系统：只关注3个核心指标，避免过度复杂的监控

核心原则:
- 监控3个关键指标: 胜率、最大回撤、参数稳定性
- 自动化决策: 停止交易/保守模式/正常运行
- 避免过度分析: 简单、快速、可操作的监控

Author: DipMaster Optimization Team
Date: 2025-08-15
Version: 1.0.0
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class OperatingMode(Enum):
    """运行模式"""
    NORMAL_OPERATION = "normal"
    CONSERVATIVE_MODE = "conservative"
    STOP_TRADING = "stop"
    HALT_INVESTIGATE = "halt"


@dataclass
class MonitoringAlert:
    """监控警报"""
    timestamp: datetime
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    message: str
    current_value: float
    threshold_value: float
    recommended_action: str


@dataclass
class PerformanceSnapshot:
    """性能快照"""
    timestamp: datetime
    period_days: int
    total_trades: int
    win_rate: float
    total_return_pct: float
    max_drawdown_pct: float
    avg_holding_minutes: float
    recent_trades_win_rate: float  # 最近10笔交易胜率
    parameter_drift_score: float   # 参数漂移评分


@dataclass
class HealthStatus:
    """系统健康状态"""
    operating_mode: OperatingMode
    health_score: float  # 0-1
    days_since_last_review: int
    critical_alerts: List[MonitoringAlert]
    performance_trend: str  # IMPROVING, STABLE, DECLINING
    recommendation: str


class SimpleMonitor:
    """简化监控系统"""
    
    def __init__(self, initial_parameters: Dict[str, float]):
        
        # 存储初始参数用于漂移检测
        self.initial_parameters = initial_parameters.copy()
        self.last_review_date = datetime.now()
        
        # 核心监控阈值 - 只有3个关键指标
        self.thresholds = {
            # 1. 胜率监控
            'win_rate_critical': 0.45,      # 低于45%立即停止
            'win_rate_warning': 0.50,       # 低于50%进入保守模式
            'win_rate_target': 0.55,        # 目标胜率55%
            
            # 2. 最大回撤监控
            'max_drawdown_critical': 0.20,  # 20%立即停止
            'max_drawdown_warning': 0.15,   # 15%进入保守模式
            'max_drawdown_target': 0.10,    # 目标回撤10%
            
            # 3. 参数稳定性监控
            'param_drift_critical': 0.30,   # 30%参数变化需要调查
            'param_drift_warning': 0.20,    # 20%参数变化需要警告
            'param_drift_acceptable': 0.10, # 10%参数变化可接受
        }
        
        # 性能历史记录
        self.performance_history: List[PerformanceSnapshot] = []
        self.active_alerts: List[MonitoringAlert] = []
        
        # 最后一次参数检查
        self.last_parameter_check = initial_parameters.copy()
        
        logger.info("✅ SimpleMonitor initialized")
        logger.info(f"   Initial parameters: {initial_parameters}")
        logger.info(f"   Monitoring 3 core metrics: win_rate, max_drawdown, param_stability")
    
    def update_performance_snapshot(self, trades_data: List[Dict]) -> PerformanceSnapshot:
        """更新性能快照"""
        
        if not trades_data:
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                period_days=0,
                total_trades=0,
                win_rate=0.0,
                total_return_pct=0.0,
                max_drawdown_pct=0.0,
                avg_holding_minutes=0.0,
                recent_trades_win_rate=0.0,
                parameter_drift_score=0.0
            )
        
        # 基础统计
        winning_trades = [t for t in trades_data if t.get('win', False)]
        win_rate = len(winning_trades) / len(trades_data) if trades_data else 0
        
        # 收益统计
        total_pnl = sum(t.get('pnl_usd', 0) for t in trades_data)
        initial_capital = 10000  # 假设初始资金
        total_return_pct = (total_pnl / initial_capital) * 100
        
        # 回撤计算 - 简化版本
        cumulative_pnl = np.cumsum([t.get('pnl_usd', 0) for t in trades_data])
        running_max = np.maximum.accumulate(np.concatenate([[0], cumulative_pnl]))
        drawdowns = running_max - np.concatenate([[0], cumulative_pnl])
        max_drawdown_pct = (max(drawdowns) / initial_capital * 100) if len(drawdowns) > 0 else 0
        
        # 平均持仓时间
        avg_holding = np.mean([t.get('holding_minutes', 60) for t in trades_data])
        
        # 最近交易表现 (最近10笔)
        recent_trades = trades_data[-10:] if len(trades_data) >= 10 else trades_data
        recent_wins = [t for t in recent_trades if t.get('win', False)]
        recent_win_rate = len(recent_wins) / len(recent_trades) if recent_trades else 0
        
        # 时间跨度
        if trades_data:
            first_trade_time = min(t.get('entry_time', datetime.now()) for t in trades_data)
            last_trade_time = max(t.get('entry_time', datetime.now()) for t in trades_data)
            period_days = (last_trade_time - first_trade_time).days or 1
        else:
            period_days = 1
        
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            period_days=period_days,
            total_trades=len(trades_data),
            win_rate=win_rate,
            total_return_pct=total_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            avg_holding_minutes=avg_holding,
            recent_trades_win_rate=recent_win_rate,
            parameter_drift_score=0.0  # 稍后更新
        )
        
        self.performance_history.append(snapshot)
        
        # 只保留最近30天的历史
        if len(self.performance_history) > 30:
            self.performance_history = self.performance_history[-30:]
        
        logger.info(f"📊 Performance updated: WR={win_rate:.1%}, "
                   f"Return={total_return_pct:+.1f}%, "
                   f"Drawdown={max_drawdown_pct:.1f}%")
        
        return snapshot
    
    def check_parameter_drift(self, current_parameters: Dict[str, float]) -> float:
        """检查参数漂移"""
        
        if not self.initial_parameters:
            return 0.0
        
        drift_scores = []
        
        for param_name, initial_value in self.initial_parameters.items():
            current_value = current_parameters.get(param_name, initial_value)
            
            if initial_value != 0:
                drift_pct = abs(current_value - initial_value) / abs(initial_value)
                drift_scores.append(drift_pct)
            else:
                # 处理初始值为0的情况
                drift_scores.append(abs(current_value))
        
        max_drift_score = max(drift_scores) if drift_scores else 0.0
        
        # 更新最后检查的参数
        self.last_parameter_check = current_parameters.copy()
        
        # 更新最新快照的漂移评分
        if self.performance_history:
            self.performance_history[-1].parameter_drift_score = max_drift_score
        
        return max_drift_score
    
    def generate_alerts(self, current_snapshot: PerformanceSnapshot, 
                       parameter_drift: float) -> List[MonitoringAlert]:
        """生成监控警报"""
        
        alerts = []
        current_time = datetime.now()
        
        # 1. 胜率警报
        if current_snapshot.win_rate <= self.thresholds['win_rate_critical']:
            alerts.append(MonitoringAlert(
                timestamp=current_time,
                alert_type="win_rate",
                severity="CRITICAL",
                message=f"Win rate {current_snapshot.win_rate:.1%} below critical threshold",
                current_value=current_snapshot.win_rate,
                threshold_value=self.thresholds['win_rate_critical'],
                recommended_action="STOP_TRADING"
            ))
        elif current_snapshot.win_rate <= self.thresholds['win_rate_warning']:
            alerts.append(MonitoringAlert(
                timestamp=current_time,
                alert_type="win_rate",
                severity="HIGH",
                message=f"Win rate {current_snapshot.win_rate:.1%} below warning threshold",
                current_value=current_snapshot.win_rate,
                threshold_value=self.thresholds['win_rate_warning'],
                recommended_action="CONSERVATIVE_MODE"
            ))
        
        # 2. 最大回撤警报
        if current_snapshot.max_drawdown_pct >= self.thresholds['max_drawdown_critical']:
            alerts.append(MonitoringAlert(
                timestamp=current_time,
                alert_type="max_drawdown",
                severity="CRITICAL",
                message=f"Max drawdown {current_snapshot.max_drawdown_pct:.1f}% exceeds critical threshold",
                current_value=current_snapshot.max_drawdown_pct,
                threshold_value=self.thresholds['max_drawdown_critical'],
                recommended_action="STOP_TRADING"
            ))
        elif current_snapshot.max_drawdown_pct >= self.thresholds['max_drawdown_warning']:
            alerts.append(MonitoringAlert(
                timestamp=current_time,
                alert_type="max_drawdown",
                severity="HIGH",
                message=f"Max drawdown {current_snapshot.max_drawdown_pct:.1f}% above warning threshold",
                current_value=current_snapshot.max_drawdown_pct,
                threshold_value=self.thresholds['max_drawdown_warning'],
                recommended_action="CONSERVATIVE_MODE"
            ))
        
        # 3. 参数漂移警报
        if parameter_drift >= self.thresholds['param_drift_critical']:
            alerts.append(MonitoringAlert(
                timestamp=current_time,
                alert_type="parameter_drift",
                severity="CRITICAL",
                message=f"Parameter drift {parameter_drift:.1%} requires investigation",
                current_value=parameter_drift,
                threshold_value=self.thresholds['param_drift_critical'],
                recommended_action="HALT_INVESTIGATE"
            ))
        elif parameter_drift >= self.thresholds['param_drift_warning']:
            alerts.append(MonitoringAlert(
                timestamp=current_time,
                alert_type="parameter_drift",
                severity="MEDIUM",
                message=f"Parameter drift {parameter_drift:.1%} above warning threshold",
                current_value=parameter_drift,
                threshold_value=self.thresholds['param_drift_warning'],
                recommended_action="REVIEW_PARAMETERS"
            ))
        
        return alerts
    
    def determine_operating_mode(self, alerts: List[MonitoringAlert]) -> OperatingMode:
        """确定运行模式"""
        
        # 检查是否有关键警报
        critical_alerts = [a for a in alerts if a.severity == "CRITICAL"]
        high_alerts = [a for a in alerts if a.severity == "HIGH"]
        
        if any("HALT_INVESTIGATE" in a.recommended_action for a in critical_alerts):
            return OperatingMode.HALT_INVESTIGATE
        
        if any("STOP_TRADING" in a.recommended_action for a in critical_alerts):
            return OperatingMode.STOP_TRADING
        
        if critical_alerts or any("CONSERVATIVE_MODE" in a.recommended_action for a in high_alerts):
            return OperatingMode.CONSERVATIVE_MODE
        
        return OperatingMode.NORMAL_OPERATION
    
    def assess_performance_trend(self) -> str:
        """评估性能趋势"""
        
        if len(self.performance_history) < 3:
            return "INSUFFICIENT_DATA"
        
        # 使用最近7天的数据评估趋势
        recent_snapshots = self.performance_history[-7:]
        
        # 胜率趋势
        win_rates = [s.win_rate for s in recent_snapshots]
        
        # 简单线性趋势
        if len(win_rates) >= 3:
            recent_trend = win_rates[-1] - win_rates[0]
            
            if recent_trend > 0.05:  # 5%改善
                return "IMPROVING"
            elif recent_trend < -0.05:  # 5%恶化
                return "DECLINING"
            else:
                return "STABLE"
        
        return "STABLE"
    
    def generate_health_report(self, trades_data: List[Dict], 
                              current_parameters: Dict[str, float]) -> HealthStatus:
        """生成健康报告"""
        
        logger.info("📋 Generating health report...")
        
        # 更新性能快照
        current_snapshot = self.update_performance_snapshot(trades_data)
        
        # 检查参数漂移
        parameter_drift = self.check_parameter_drift(current_parameters)
        
        # 生成警报
        new_alerts = self.generate_alerts(current_snapshot, parameter_drift)
        
        # 更新活跃警报 (保留最近24小时的警报)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.active_alerts = [a for a in self.active_alerts if a.timestamp > cutoff_time] + new_alerts
        
        # 确定运行模式
        operating_mode = self.determine_operating_mode(self.active_alerts)
        
        # 评估性能趋势
        performance_trend = self.assess_performance_trend()
        
        # 计算健康评分 (0-1)
        health_score = self._calculate_health_score(current_snapshot, parameter_drift)
        
        # 生成建议
        recommendation = self._generate_recommendation(operating_mode, health_score, performance_trend)
        
        # 计算天数
        days_since_review = (datetime.now() - self.last_review_date).days
        
        health_status = HealthStatus(
            operating_mode=operating_mode,
            health_score=health_score,
            days_since_last_review=days_since_review,
            critical_alerts=[a for a in self.active_alerts if a.severity == "CRITICAL"],
            performance_trend=performance_trend,
            recommendation=recommendation
        )
        
        logger.info(f"🏥 Health assessment: {operating_mode.value.upper()}, "
                   f"Score: {health_score:.1%}, Trend: {performance_trend}")
        
        return health_status
    
    def _calculate_health_score(self, snapshot: PerformanceSnapshot, parameter_drift: float) -> float:
        """计算健康评分"""
        
        score = 1.0
        
        # 胜率评分 (40%权重)
        if snapshot.win_rate >= self.thresholds['win_rate_target']:
            win_rate_score = 1.0
        elif snapshot.win_rate >= self.thresholds['win_rate_warning']:
            win_rate_score = 0.7
        elif snapshot.win_rate >= self.thresholds['win_rate_critical']:
            win_rate_score = 0.3
        else:
            win_rate_score = 0.0
        
        # 回撤评分 (40%权重)
        if snapshot.max_drawdown_pct <= self.thresholds['max_drawdown_target']:
            drawdown_score = 1.0
        elif snapshot.max_drawdown_pct <= self.thresholds['max_drawdown_warning']:
            drawdown_score = 0.7
        elif snapshot.max_drawdown_pct <= self.thresholds['max_drawdown_critical']:
            drawdown_score = 0.3
        else:
            drawdown_score = 0.0
        
        # 参数稳定性评分 (20%权重)
        if parameter_drift <= self.thresholds['param_drift_acceptable']:
            param_score = 1.0
        elif parameter_drift <= self.thresholds['param_drift_warning']:
            param_score = 0.7
        elif parameter_drift <= self.thresholds['param_drift_critical']:
            param_score = 0.3
        else:
            param_score = 0.0
        
        # 加权平均
        health_score = (win_rate_score * 0.4 + drawdown_score * 0.4 + param_score * 0.2)
        
        return max(0.0, min(1.0, health_score))
    
    def _generate_recommendation(self, mode: OperatingMode, health_score: float, trend: str) -> str:
        """生成建议"""
        
        if mode == OperatingMode.STOP_TRADING:
            return "Immediately stop all trading. Investigate critical performance issues."
        
        elif mode == OperatingMode.HALT_INVESTIGATE:
            return "Halt trading and investigate parameter drift or system changes."
        
        elif mode == OperatingMode.CONSERVATIVE_MODE:
            if trend == "DECLINING":
                return "Reduce position sizes and increase monitoring frequency. Performance is declining."
            else:
                return "Operate in conservative mode with reduced risk until performance improves."
        
        else:  # NORMAL_OPERATION
            if health_score >= 0.8:
                return "System operating normally. Continue current approach."
            elif health_score >= 0.6:
                return "Performance is acceptable but monitor closely for any degradation."
            elif trend == "DECLINING":
                return "Performance declining despite normal thresholds. Consider review."
            else:
                return "System stable. Regular monitoring sufficient."
    
    def save_monitoring_report(self, health_status: HealthStatus, 
                              current_snapshot: PerformanceSnapshot) -> str:
        """保存监控报告"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"monitoring_report_{timestamp}.json"
        
        report_data = {
            'health_status': asdict(health_status),
            'current_performance': asdict(current_snapshot),
            'active_alerts': [asdict(alert) for alert in self.active_alerts],
            'monitoring_thresholds': self.thresholds,
            'initial_parameters': self.initial_parameters,
            'current_parameters': self.last_parameter_check,
            'report_timestamp': datetime.now().isoformat()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📁 Monitoring report saved to: {filename}")
        
        return filename


def main():
    """测试简化监控系统"""
    
    print("📊 Simple Monitor - 3-Metric Monitoring System")
    print("="*80)
    
    # 模拟初始参数
    initial_params = {
        'rsi_threshold': 40.0,
        'take_profit_pct': 0.015,
        'stop_loss_pct': 0.008
    }
    
    # 创建监控器
    monitor = SimpleMonitor(initial_params)
    
    print("🎯 MONITORING CONFIGURATION:")
    print("Core Metrics: Win Rate, Max Drawdown, Parameter Stability")
    print(f"Win Rate Thresholds: Critical<{monitor.thresholds['win_rate_critical']:.0%}, "
          f"Warning<{monitor.thresholds['win_rate_warning']:.0%}, "
          f"Target>{monitor.thresholds['win_rate_target']:.0%}")
    print(f"Drawdown Thresholds: Critical>{monitor.thresholds['max_drawdown_critical']:.0%}, "
          f"Warning>{monitor.thresholds['max_drawdown_warning']:.0%}, "
          f"Target<{monitor.thresholds['max_drawdown_target']:.0%}")
    print(f"Parameter Drift Thresholds: Critical>{monitor.thresholds['param_drift_critical']:.0%}, "
          f"Warning>{monitor.thresholds['param_drift_warning']:.0%}")
    
    # 模拟一些交易数据进行测试
    sample_trades = [
        {'entry_time': datetime.now() - timedelta(days=i), 'pnl_usd': 50 if i % 2 == 0 else -30, 
         'win': i % 2 == 0, 'holding_minutes': 45} 
        for i in range(20)
    ]
    
    # 模拟当前参数（有一些漂移）
    current_params = {
        'rsi_threshold': 42.0,     # 5%漂移
        'take_profit_pct': 0.016,  # 6.7%漂移  
        'stop_loss_pct': 0.008     # 无漂移
    }
    
    # 生成健康报告
    health_status = monitor.generate_health_report(sample_trades, current_params)
    
    # 显示结果
    print(f"\n🏥 HEALTH STATUS:")
    print(f"Operating Mode: {health_status.operating_mode.value.upper()}")
    print(f"Health Score: {health_status.health_score:.1%}")
    print(f"Performance Trend: {health_status.performance_trend}")
    print(f"Days Since Review: {health_status.days_since_last_review}")
    
    if health_status.critical_alerts:
        print(f"\n🚨 CRITICAL ALERTS:")
        for alert in health_status.critical_alerts:
            print(f"   • {alert.alert_type}: {alert.message}")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"{health_status.recommendation}")
    
    # 保存报告
    current_snapshot = monitor.performance_history[-1] if monitor.performance_history else None
    if current_snapshot:
        report_file = monitor.save_monitoring_report(health_status, current_snapshot)
        print(f"\n📁 Report saved to: {report_file}")
    
    print("\n✅ Simple Monitor test completed")


if __name__ == "__main__":
    main()