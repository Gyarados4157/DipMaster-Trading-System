#!/usr/bin/env python3
"""
DipMaster Trading System - Advanced Alert System
高级告警系统 - 智能阈值管理和多级告警处理

Author: DipMaster Trading System Monitoring Agent
Date: 2025-08-18
Version: 1.0.0
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import statistics
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """告警严重性级别"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "active"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ESCALATED = "escalated"


class AlertCategory(Enum):
    """告警分类"""
    RISK_MANAGEMENT = "risk_management"
    STRATEGY_PERFORMANCE = "strategy_performance"
    SYSTEM_HEALTH = "system_health"
    DATA_QUALITY = "data_quality"
    EXECUTION_QUALITY = "execution_quality"
    COMPLIANCE = "compliance"
    SECURITY = "security"


class ThresholdType(Enum):
    """阈值类型"""
    STATIC = "static"
    DYNAMIC = "dynamic"
    PERCENTILE = "percentile"
    STATISTICAL = "statistical"


@dataclass
class AlertThreshold:
    """告警阈值配置"""
    name: str
    metric: str
    threshold_type: ThresholdType
    warning_value: float
    critical_value: float
    emergency_value: Optional[float] = None
    comparison: str = "greater_than"  # greater_than, less_than, equals, not_equals
    duration_seconds: int = 60  # 持续时间
    cooldown_seconds: int = 300  # 冷却时间
    enabled: bool = True
    tags: Dict[str, str] = field(default_factory=dict)
    
    # 动态阈值参数
    lookback_window_hours: int = 24
    percentile: float = 95.0
    standard_deviations: float = 2.0
    adaptive_factor: float = 1.2


@dataclass
class AlertRule:
    """告警规则"""
    rule_id: str
    name: str
    description: str
    category: AlertCategory
    thresholds: List[AlertThreshold]
    enabled: bool = True
    auto_resolve: bool = True
    auto_escalate: bool = False
    escalation_delay_minutes: int = 30
    suppression_rules: List[str] = field(default_factory=list)
    notification_channels: List[str] = field(default_factory=list)
    custom_actions: List[str] = field(default_factory=list)


@dataclass
class AlertIncident:
    """告警事件"""
    incident_id: str
    rule_id: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    description: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: Optional[datetime] = None
    metric_value: float = 0.0
    threshold_value: float = 0.0
    source: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    affected_components: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    escalation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        # 转换datetime为ISO字符串
        for key in ['created_at', 'updated_at', 'resolved_at']:
            if data[key]:
                data[key] = data[key].isoformat()
        return data


class ThresholdCalculator:
    """动态阈值计算器"""
    
    def __init__(self):
        self.metric_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.max_history_points = 10000
    
    def record_metric(self, metric: str, value: float, timestamp: datetime = None) -> None:
        """记录指标值"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        if metric not in self.metric_history:
            self.metric_history[metric] = []
        
        self.metric_history[metric].append((timestamp, value))
        
        # 限制历史数据量
        if len(self.metric_history[metric]) > self.max_history_points:
            self.metric_history[metric] = self.metric_history[metric][-self.max_history_points:]
    
    def calculate_dynamic_threshold(
        self,
        metric: str,
        threshold_config: AlertThreshold
    ) -> Tuple[float, float, Optional[float]]:
        """计算动态阈值"""
        if metric not in self.metric_history:
            # 没有历史数据，返回静态阈值
            return (
                threshold_config.warning_value,
                threshold_config.critical_value,
                threshold_config.emergency_value
            )
        
        # 获取指定时间窗口内的数据
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=threshold_config.lookback_window_hours)
        recent_values = [
            value for timestamp, value in self.metric_history[metric]
            if timestamp >= cutoff_time
        ]
        
        if not recent_values:
            return (
                threshold_config.warning_value,
                threshold_config.critical_value,
                threshold_config.emergency_value
            )
        
        if threshold_config.threshold_type == ThresholdType.PERCENTILE:
            return self._calculate_percentile_thresholds(recent_values, threshold_config)
        elif threshold_config.threshold_type == ThresholdType.STATISTICAL:
            return self._calculate_statistical_thresholds(recent_values, threshold_config)
        else:
            # 动态调整静态阈值
            return self._calculate_adaptive_thresholds(recent_values, threshold_config)
    
    def _calculate_percentile_thresholds(
        self,
        values: List[float],
        config: AlertThreshold
    ) -> Tuple[float, float, Optional[float]]:
        """基于百分位数计算阈值"""
        try:
            warning_threshold = statistics.quantile(values, config.percentile / 100.0)
            critical_threshold = statistics.quantile(values, min(99.0, config.percentile + 4) / 100.0)
            emergency_threshold = statistics.quantile(values, 99.9 / 100.0) if config.emergency_value else None
            
            return warning_threshold, critical_threshold, emergency_threshold
        except statistics.StatisticsError:
            return config.warning_value, config.critical_value, config.emergency_value
    
    def _calculate_statistical_thresholds(
        self,
        values: List[float],
        config: AlertThreshold
    ) -> Tuple[float, float, Optional[float]]:
        """基于统计分布计算阈值"""
        try:
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values) if len(values) > 1 else 0
            
            warning_threshold = mean_val + (config.standard_deviations * std_val)
            critical_threshold = mean_val + ((config.standard_deviations + 1) * std_val)
            emergency_threshold = mean_val + ((config.standard_deviations + 2) * std_val) if config.emergency_value else None
            
            return warning_threshold, critical_threshold, emergency_threshold
        except statistics.StatisticsError:
            return config.warning_value, config.critical_value, config.emergency_value
    
    def _calculate_adaptive_thresholds(
        self,
        values: List[float],
        config: AlertThreshold
    ) -> Tuple[float, float, Optional[float]]:
        """自适应阈值计算"""
        try:
            recent_avg = statistics.mean(values)
            baseline_avg = statistics.mean(values[-int(len(values) * 0.3):]) if len(values) > 10 else recent_avg
            
            # 根据最近趋势调整阈值
            adjustment_factor = config.adaptive_factor if recent_avg > baseline_avg else 1.0
            
            warning_threshold = config.warning_value * adjustment_factor
            critical_threshold = config.critical_value * adjustment_factor
            emergency_threshold = config.emergency_value * adjustment_factor if config.emergency_value else None
            
            return warning_threshold, critical_threshold, emergency_threshold
        except Exception:
            return config.warning_value, config.critical_value, config.emergency_value


class AlertEvaluator:
    """告警评估器"""
    
    def __init__(self, threshold_calculator: ThresholdCalculator):
        self.threshold_calculator = threshold_calculator
        self.active_violations: Dict[str, Dict[str, Any]] = {}
    
    async def evaluate_metric(
        self,
        metric: str,
        value: float,
        rule: AlertRule,
        timestamp: datetime = None
    ) -> Optional[AlertIncident]:
        """评估指标是否触发告警"""
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # 记录指标值
        self.threshold_calculator.record_metric(metric, value, timestamp)
        
        for threshold in rule.thresholds:
            if threshold.metric != metric or not threshold.enabled:
                continue
            
            violation_key = f"{rule.rule_id}_{threshold.name}"
            
            # 计算当前阈值
            warning_val, critical_val, emergency_val = self.threshold_calculator.calculate_dynamic_threshold(
                metric, threshold
            )
            
            # 评估是否违规
            severity = self._evaluate_threshold_violation(value, threshold, warning_val, critical_val, emergency_val)
            
            if severity:
                # 检查持续时间
                if await self._check_violation_duration(violation_key, severity, threshold, timestamp):
                    incident = await self._create_incident(
                        rule, threshold, severity, metric, value,
                        critical_val if severity == AlertSeverity.CRITICAL else 
                        emergency_val if severity == AlertSeverity.EMERGENCY else warning_val,
                        timestamp
                    )
                    return incident
            else:
                # 清除违规记录
                if violation_key in self.active_violations:
                    del self.active_violations[violation_key]
        
        return None
    
    def _evaluate_threshold_violation(
        self,
        value: float,
        threshold: AlertThreshold,
        warning_val: float,
        critical_val: float,
        emergency_val: Optional[float]
    ) -> Optional[AlertSeverity]:
        """评估阈值违规严重性"""
        
        def compare_values(val1: float, val2: float, comparison: str) -> bool:
            if comparison == "greater_than":
                return val1 > val2
            elif comparison == "less_than":
                return val1 < val2
            elif comparison == "equals":
                return abs(val1 - val2) < 1e-6
            elif comparison == "not_equals":
                return abs(val1 - val2) >= 1e-6
            return False
        
        # 检查紧急级别
        if emergency_val is not None and compare_values(value, emergency_val, threshold.comparison):
            return AlertSeverity.EMERGENCY
        
        # 检查严重级别
        if compare_values(value, critical_val, threshold.comparison):
            return AlertSeverity.CRITICAL
        
        # 检查警告级别
        if compare_values(value, warning_val, threshold.comparison):
            return AlertSeverity.WARNING
        
        return None
    
    async def _check_violation_duration(
        self,
        violation_key: str,
        severity: AlertSeverity,
        threshold: AlertThreshold,
        timestamp: datetime
    ) -> bool:
        """检查违规持续时间"""
        if violation_key not in self.active_violations:
            self.active_violations[violation_key] = {
                'first_violation': timestamp,
                'last_violation': timestamp,
                'severity': severity,
                'count': 1
            }
            return False
        
        violation_data = self.active_violations[violation_key]
        violation_data['last_violation'] = timestamp
        violation_data['severity'] = severity
        violation_data['count'] += 1
        
        # 检查是否满足持续时间要求
        duration = (timestamp - violation_data['first_violation']).total_seconds()
        return duration >= threshold.duration_seconds
    
    async def _create_incident(
        self,
        rule: AlertRule,
        threshold: AlertThreshold,
        severity: AlertSeverity,
        metric: str,
        value: float,
        threshold_value: float,
        timestamp: datetime
    ) -> AlertIncident:
        """创建告警事件"""
        incident_id = f"{rule.rule_id}_{threshold.name}_{int(timestamp.timestamp())}"
        
        # 生成推荐操作
        recommended_actions = self._generate_recommended_actions(rule, threshold, severity, metric, value)
        
        incident = AlertIncident(
            incident_id=incident_id,
            rule_id=rule.rule_id,
            severity=severity,
            status=AlertStatus.ACTIVE,
            title=f"[{severity.value}] {rule.name}",
            description=f"Metric '{metric}' value {value:.4f} exceeded {severity.value.lower()} threshold {threshold_value:.4f}",
            metric_value=value,
            threshold_value=threshold_value,
            source="alert_evaluator",
            tags=threshold.tags,
            recommended_actions=recommended_actions,
            created_at=timestamp,
            updated_at=timestamp
        )
        
        return incident
    
    def _generate_recommended_actions(
        self,
        rule: AlertRule,
        threshold: AlertThreshold,
        severity: AlertSeverity,
        metric: str,
        value: float
    ) -> List[str]:
        """生成推荐操作"""
        actions = []
        
        # 基于告警类别的通用建议
        if rule.category == AlertCategory.RISK_MANAGEMENT:
            if "var" in metric.lower():
                actions.append("Review and reduce portfolio exposure")
                actions.append("Implement additional hedging strategies")
            elif "drawdown" in metric.lower():
                actions.append("Consider stopping trading until market stabilizes")
                actions.append("Review position sizing and risk limits")
            elif "leverage" in metric.lower():
                actions.append("Reduce leverage immediately")
                actions.append("Close non-essential positions")
        
        elif rule.category == AlertCategory.STRATEGY_PERFORMANCE:
            if "win_rate" in metric.lower():
                actions.append("Analyze recent losing trades for patterns")
                actions.append("Consider adjusting strategy parameters")
            elif "sharpe" in metric.lower():
                actions.append("Review risk-adjusted returns")
                actions.append("Evaluate strategy allocation")
        
        elif rule.category == AlertCategory.SYSTEM_HEALTH:
            if "cpu" in metric.lower():
                actions.append("Investigate CPU-intensive processes")
                actions.append("Consider scaling resources")
            elif "memory" in metric.lower():
                actions.append("Check for memory leaks")
                actions.append("Restart services if necessary")
            elif "latency" in metric.lower():
                actions.append("Check network connectivity")
                actions.append("Optimize API calls and data processing")
        
        # 基于严重性的紧急操作
        if severity == AlertSeverity.EMERGENCY:
            actions.insert(0, "IMMEDIATE ACTION REQUIRED")
            actions.append("Contact system administrator immediately")
        elif severity == AlertSeverity.CRITICAL:
            actions.append("Escalate to operations team")
        
        return actions[:5]  # 限制建议数量


class AlertManager:
    """告警管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.threshold_calculator = ThresholdCalculator()
        self.alert_evaluator = AlertEvaluator(self.threshold_calculator)
        self.rules: Dict[str, AlertRule] = {}
        self.active_incidents: Dict[str, AlertIncident] = {}
        self.incident_history: List[AlertIncident] = []
        self.notification_handlers: Dict[str, Callable] = {}
        self.suppression_rules: Dict[str, Dict[str, Any]] = {}
        
        # 统计信息
        self.stats = {
            'total_incidents': 0,
            'active_incidents': 0,
            'resolved_incidents': 0,
            'suppressed_incidents': 0,
            'notifications_sent': 0
        }
        
        # 初始化默认规则
        self._initialize_default_rules()
    
    def _initialize_default_rules(self) -> None:
        """初始化默认告警规则"""
        
        # 风险管理告警规则
        risk_rules = [
            AlertRule(
                rule_id="risk_var_95",
                name="VaR 95% Limit Exceeded",
                description="Portfolio Value at Risk (95% confidence) exceeded limit",
                category=AlertCategory.RISK_MANAGEMENT,
                thresholds=[
                    AlertThreshold(
                        name="var_95_threshold",
                        metric="risk.var_95",
                        threshold_type=ThresholdType.STATIC,
                        warning_value=160000.0,   # 80% of limit
                        critical_value=190000.0,  # 95% of limit
                        emergency_value=200000.0, # 100% of limit
                        comparison="greater_than",
                        duration_seconds=60,
                        cooldown_seconds=300
                    )
                ]
            ),
            AlertRule(
                rule_id="risk_drawdown",
                name="Maximum Drawdown Alert",
                description="Portfolio drawdown exceeded acceptable levels",
                category=AlertCategory.RISK_MANAGEMENT,
                thresholds=[
                    AlertThreshold(
                        name="drawdown_threshold",
                        metric="risk.current_drawdown",
                        threshold_type=ThresholdType.STATIC,
                        warning_value=0.10,   # 10% drawdown
                        critical_value=0.15,  # 15% drawdown
                        emergency_value=0.20, # 20% drawdown
                        comparison="greater_than",
                        duration_seconds=30,
                        cooldown_seconds=600
                    )
                ]
            )
        ]
        
        # 策略性能告警规则
        strategy_rules = [
            AlertRule(
                rule_id="strategy_win_rate",
                name="Strategy Win Rate Degradation",
                description="Trading strategy win rate below target",
                category=AlertCategory.STRATEGY_PERFORMANCE,
                thresholds=[
                    AlertThreshold(
                        name="win_rate_threshold",
                        metric="strategy.win_rate",
                        threshold_type=ThresholdType.DYNAMIC,
                        warning_value=0.45,   # 45% win rate
                        critical_value=0.30,  # 30% win rate
                        comparison="less_than",
                        duration_seconds=300,
                        cooldown_seconds=900,
                        lookback_window_hours=24,
                        adaptive_factor=0.9
                    )
                ]
            ),
            AlertRule(
                rule_id="strategy_sharpe_ratio",
                name="Strategy Sharpe Ratio Alert",
                description="Strategy Sharpe ratio below acceptable level",
                category=AlertCategory.STRATEGY_PERFORMANCE,
                thresholds=[
                    AlertThreshold(
                        name="sharpe_threshold",
                        metric="strategy.sharpe_ratio",
                        threshold_type=ThresholdType.STATISTICAL,
                        warning_value=1.0,
                        critical_value=0.5,
                        comparison="less_than",
                        duration_seconds=600,
                        cooldown_seconds=1800,
                        standard_deviations=1.5
                    )
                ]
            )
        ]
        
        # 系统健康告警规则
        system_rules = [
            AlertRule(
                rule_id="system_cpu_usage",
                name="High CPU Usage Alert",
                description="System CPU usage exceeded threshold",
                category=AlertCategory.SYSTEM_HEALTH,
                thresholds=[
                    AlertThreshold(
                        name="cpu_threshold",
                        metric="system.cpu.usage_percent",
                        threshold_type=ThresholdType.STATIC,
                        warning_value=80.0,
                        critical_value=95.0,
                        comparison="greater_than",
                        duration_seconds=180,
                        cooldown_seconds=300
                    )
                ]
            ),
            AlertRule(
                rule_id="system_memory_usage",
                name="High Memory Usage Alert",
                description="System memory usage exceeded threshold",
                category=AlertCategory.SYSTEM_HEALTH,
                thresholds=[
                    AlertThreshold(
                        name="memory_threshold",
                        metric="system.memory.usage_percent",
                        threshold_type=ThresholdType.STATIC,
                        warning_value=80.0,
                        critical_value=95.0,
                        comparison="greater_than",
                        duration_seconds=180,
                        cooldown_seconds=300
                    )
                ]
            ),
            AlertRule(
                rule_id="api_latency",
                name="API Response Latency Alert",
                description="API response latency exceeded acceptable levels",
                category=AlertCategory.EXECUTION_QUALITY,
                thresholds=[
                    AlertThreshold(
                        name="latency_threshold",
                        metric="execution.api_latency_ms",
                        threshold_type=ThresholdType.PERCENTILE,
                        warning_value=500.0,
                        critical_value=1000.0,
                        emergency_value=2000.0,
                        comparison="greater_than",
                        duration_seconds=120,
                        cooldown_seconds=300,
                        percentile=95.0
                    )
                ]
            )
        ]
        
        # 注册所有规则
        for rule in risk_rules + strategy_rules + system_rules:
            self.rules[rule.rule_id] = rule
    
    async def add_rule(self, rule: AlertRule) -> None:
        """添加告警规则"""
        self.rules[rule.rule_id] = rule
        logger.info(f"📋 Added alert rule: {rule.rule_id}")
    
    async def remove_rule(self, rule_id: str) -> bool:
        """移除告警规则"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"🗑️ Removed alert rule: {rule_id}")
            return True
        return False
    
    async def evaluate_metric(self, metric: str, value: float, timestamp: datetime = None) -> List[AlertIncident]:
        """评估指标并生成告警"""
        incidents = []
        
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            # 检查是否有匹配的阈值
            if any(threshold.metric == metric for threshold in rule.thresholds):
                incident = await self.alert_evaluator.evaluate_metric(metric, value, rule, timestamp)
                if incident:
                    incidents.append(incident)
                    await self._process_incident(incident)
        
        return incidents
    
    async def _process_incident(self, incident: AlertIncident) -> None:
        """处理告警事件"""
        # 检查抑制规则
        if await self._should_suppress_incident(incident):
            incident.status = AlertStatus.SUPPRESSED
            self.stats['suppressed_incidents'] += 1
            logger.info(f"🔇 Suppressed incident: {incident.incident_id}")
            return
        
        # 添加到活跃事件
        self.active_incidents[incident.incident_id] = incident
        self.incident_history.append(incident)
        
        # 更新统计
        self.stats['total_incidents'] += 1
        self.stats['active_incidents'] += 1
        
        # 发送通知
        await self._send_notifications(incident)
        
        # 执行自定义操作
        await self._execute_custom_actions(incident)
        
        logger.warning(f"🚨 Created incident: {incident.incident_id} - {incident.title}")
    
    async def _should_suppress_incident(self, incident: AlertIncident) -> bool:
        """检查是否应该抑制告警"""
        rule = self.rules.get(incident.rule_id)
        if not rule or not rule.suppression_rules:
            return False
        
        # 检查抑制规则
        for suppression_rule_id in rule.suppression_rules:
            if suppression_rule_id in self.suppression_rules:
                suppression_rule = self.suppression_rules[suppression_rule_id]
                if await self._evaluate_suppression_rule(incident, suppression_rule):
                    return True
        
        return False
    
    async def _evaluate_suppression_rule(self, incident: AlertIncident, suppression_rule: Dict[str, Any]) -> bool:
        """评估抑制规则"""
        # 示例抑制规则：如果有更高级别的告警活跃，则抑制较低级别的告警
        if suppression_rule.get('type') == 'severity_based':
            higher_severities = {
                AlertSeverity.WARNING: [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY],
                AlertSeverity.CRITICAL: [AlertSeverity.EMERGENCY]
            }.get(incident.severity, [])
            
            for active_incident in self.active_incidents.values():
                if (active_incident.severity in higher_severities and 
                    active_incident.status == AlertStatus.ACTIVE):
                    return True
        
        return False
    
    async def _send_notifications(self, incident: AlertIncident) -> None:
        """发送告警通知"""
        rule = self.rules.get(incident.rule_id)
        if not rule:
            return
        
        for channel in rule.notification_channels:
            if channel in self.notification_handlers:
                try:
                    await self.notification_handlers[channel](incident)
                    self.stats['notifications_sent'] += 1
                except Exception as e:
                    logger.error(f"❌ Failed to send notification via {channel}: {e}")
    
    async def _execute_custom_actions(self, incident: AlertIncident) -> None:
        """执行自定义操作"""
        rule = self.rules.get(incident.rule_id)
        if not rule or not rule.custom_actions:
            return
        
        for action in rule.custom_actions:
            try:
                # 这里可以实现具体的自定义操作
                logger.info(f"⚡ Executing custom action: {action} for incident {incident.incident_id}")
            except Exception as e:
                logger.error(f"❌ Failed to execute custom action {action}: {e}")
    
    async def resolve_incident(self, incident_id: str, resolution_message: str = "") -> bool:
        """解决告警事件"""
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            incident.status = AlertStatus.RESOLVED
            incident.resolved_at = datetime.now(timezone.utc)
            incident.updated_at = incident.resolved_at
            
            # 从活跃事件中移除
            del self.active_incidents[incident_id]
            
            # 更新统计
            self.stats['active_incidents'] -= 1
            self.stats['resolved_incidents'] += 1
            
            logger.info(f"✅ Resolved incident: {incident_id} - {resolution_message}")
            return True
        
        return False
    
    def register_notification_handler(self, channel: str, handler: Callable) -> None:
        """注册通知处理器"""
        self.notification_handlers[channel] = handler
        logger.info(f"📢 Registered notification handler: {channel}")
    
    def get_active_incidents(self, severity: AlertSeverity = None) -> List[AlertIncident]:
        """获取活跃告警"""
        incidents = list(self.active_incidents.values())
        if severity:
            incidents = [inc for inc in incidents if inc.severity == severity]
        return incidents
    
    def get_incident_history(self, hours: int = 24) -> List[AlertIncident]:
        """获取告警历史"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        return [inc for inc in self.incident_history if inc.created_at >= cutoff_time]
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计信息"""
        return {
            **self.stats,
            'rules_count': len(self.rules),
            'active_rules_count': sum(1 for rule in self.rules.values() if rule.enabled),
            'notification_handlers_count': len(self.notification_handlers)
        }


# 通知处理器示例
async def console_notification_handler(incident: AlertIncident) -> None:
    """控制台通知处理器"""
    print(f"🚨 ALERT [{incident.severity.value}]: {incident.title}")
    print(f"   Description: {incident.description}")
    print(f"   Metric: {incident.metric_value:.4f} (threshold: {incident.threshold_value:.4f})")
    print(f"   Recommended Actions: {', '.join(incident.recommended_actions)}")


async def slack_notification_handler(incident: AlertIncident) -> None:
    """Slack通知处理器（模拟）"""
    # 这里应该实现真实的Slack通知
    logger.info(f"📱 Would send Slack notification for incident: {incident.incident_id}")


# 工厂函数
def create_alert_manager(config: Dict[str, Any] = None) -> AlertManager:
    """创建告警管理器"""
    manager = AlertManager(config)
    
    # 注册默认通知处理器
    manager.register_notification_handler("console", console_notification_handler)
    manager.register_notification_handler("slack", slack_notification_handler)
    
    return manager


# 演示函数
async def alert_system_demo():
    """告警系统演示"""
    print("🚀 DipMaster Advanced Alert System Demo")
    
    # 创建告警管理器
    alert_manager = create_alert_manager()
    
    # 模拟指标数据
    test_metrics = [
        ("risk.var_95", 150000.0),  # 正常
        ("risk.var_95", 170000.0),  # 接近警告
        ("risk.var_95", 185000.0),  # 警告级别
        ("risk.var_95", 195000.0),  # 严重级别
        ("strategy.win_rate", 0.6),  # 正常
        ("strategy.win_rate", 0.4),  # 警告
        ("strategy.win_rate", 0.25), # 严重
        ("system.cpu.usage_percent", 75.0),  # 正常
        ("system.cpu.usage_percent", 85.0),  # 警告
        ("system.cpu.usage_percent", 97.0),  # 严重
    ]
    
    print("📊 Evaluating test metrics...")
    
    # 评估指标
    all_incidents = []
    for metric, value in test_metrics:
        print(f"   Testing {metric}: {value}")
        incidents = await alert_manager.evaluate_metric(metric, value)
        all_incidents.extend(incidents)
        
        # 模拟时间推移
        await asyncio.sleep(0.1)
    
    # 显示生成的告警
    print(f"\n🚨 Generated {len(all_incidents)} incidents:")
    for incident in all_incidents:
        print(f"   [{incident.severity.value}] {incident.title}")
        print(f"     Value: {incident.metric_value}, Threshold: {incident.threshold_value}")
        print(f"     Actions: {', '.join(incident.recommended_actions[:2])}")
    
    # 显示活跃告警
    active_incidents = alert_manager.get_active_incidents()
    print(f"\n📋 Active incidents: {len(active_incidents)}")
    
    # 解决一些告警
    if active_incidents:
        await alert_manager.resolve_incident(
            active_incidents[0].incident_id,
            "Issue resolved manually"
        )
        print(f"✅ Resolved incident: {active_incidents[0].incident_id}")
    
    # 显示统计信息
    stats = alert_manager.get_alert_statistics()
    print(f"\n📊 Alert Statistics:")
    print(f"   Total incidents: {stats['total_incidents']}")
    print(f"   Active incidents: {stats['active_incidents']}")
    print(f"   Resolved incidents: {stats['resolved_incidents']}")
    print(f"   Active rules: {stats['active_rules_count']}/{stats['rules_count']}")
    
    print("✅ Demo completed successfully")


if __name__ == "__main__":
    asyncio.run(alert_system_demo())