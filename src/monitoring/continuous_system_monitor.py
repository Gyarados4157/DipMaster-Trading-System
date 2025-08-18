#!/usr/bin/env python3
"""
DipMaster Trading System - 持续系统监控和一致性验证
创建时间: 2025-08-18
版本: 1.0.0

功能: 
- 实时监控所有系统组件
- 验证信号-执行一致性
- 检测系统健康状态
- 生成告警和报告
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import psutil
import aiofiles
import hashlib

@dataclass
class SystemHealthMetrics:
    """系统健康指标"""
    timestamp: str
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    active_connections: int
    error_count: int
    warning_count: int
    uptime: float

@dataclass
class SignalExecutionConsistency:
    """信号执行一致性指标"""
    signal_id: str
    signal_timestamp: str
    signal_strength: float
    execution_timestamp: Optional[str]
    execution_delay: Optional[float]
    target_position: float
    actual_position: float
    consistency_score: float
    deviation_reason: Optional[str]

@dataclass
class RiskAlert:
    """风险告警"""
    alert_id: str
    timestamp: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    category: str  # "RISK", "PERFORMANCE", "SYSTEM", "CONSISTENCY"
    message: str
    affected_component: str
    suggested_action: str
    auto_resolved: bool = False

class ContinuousSystemMonitor:
    """持续系统监控器"""
    
    def __init__(self, config_path: str = "config/monitoring_config.json"):
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.monitoring_active = False
        
        # 监控状态
        self.system_metrics_history = []
        self.signal_consistency_history = []
        self.alerts_history = []
        self.component_status = {}
        
        # 阈值配置
        self.thresholds = {
            "cpu_usage_critical": 90.0,
            "memory_usage_critical": 85.0,
            "disk_usage_critical": 90.0,
            "signal_delay_warning": 5.0,  # 秒
            "consistency_score_warning": 0.8,
            "error_rate_critical": 0.05
        }
        
        # 组件路径
        self.monitored_paths = {
            "data_infrastructure": "data/continuous_optimization/",
            "feature_engineering": "data/continuous_optimization/",
            "model_training": "results/model_training/",
            "portfolio_optimization": "results/portfolio_optimization/",
            "execution_reports": "results/execution_reports/",
            "logs": "logs/"
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载监控配置"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            # 默认配置
            default_config = {
                "monitoring_interval": 60,  # 秒
                "health_check_interval": 300,  # 秒
                "alert_cooldown": 1800,  # 秒
                "max_history_size": 1000,
                "kafka_enabled": False,
                "report_generation_hours": [8, 16, 0]
            }
            return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger("SystemMonitor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 创建文件处理器
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            file_handler = logging.FileHandler(
                log_dir / f"system_monitor_{datetime.now().strftime('%Y%m%d')}.log"
            )
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    async def start_monitoring(self):
        """启动持续监控"""
        self.monitoring_active = True
        self.logger.info("🚀 启动DipMaster持续系统监控")
        
        # 启动并发监控任务
        tasks = [
            self._system_health_monitor(),
            self._signal_consistency_monitor(),
            self._component_status_monitor(),
            self._alert_manager(),
            self._report_generator()
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"监控系统异常: {e}")
            self.monitoring_active = False
    
    async def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        self.logger.info("🛑 停止系统监控")
    
    async def _system_health_monitor(self):
        """系统健康监控"""
        while self.monitoring_active:
            try:
                # 收集系统指标
                metrics = await self._collect_system_metrics()
                self.system_metrics_history.append(metrics)
                
                # 限制历史记录大小
                if len(self.system_metrics_history) > self.config["max_history_size"]:
                    self.system_metrics_history.pop(0)
                
                # 检查告警条件
                await self._check_system_alerts(metrics)
                
                await asyncio.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                self.logger.error(f"系统健康监控错误: {e}")
                await asyncio.sleep(30)
    
    async def _collect_system_metrics(self) -> SystemHealthMetrics:
        """收集系统指标"""
        # CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_usage = (disk.used / disk.total) * 100
        
        # 网络延迟（简化实现）
        network_latency = await self._measure_network_latency()
        
        # 活跃连接数
        connections = len(psutil.net_connections())
        
        # 错误和警告计数
        error_count, warning_count = await self._count_recent_errors()
        
        # 系统运行时间
        uptime = time.time() - psutil.boot_time()
        
        return SystemHealthMetrics(
            timestamp=datetime.now().isoformat(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_latency=network_latency,
            active_connections=connections,
            error_count=error_count,
            warning_count=warning_count,
            uptime=uptime
        )
    
    async def _measure_network_latency(self) -> float:
        """测量网络延迟"""
        try:
            import subprocess
            result = subprocess.run(
                ['ping', '-c', '1', '8.8.8.8'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                # 解析ping结果
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'time=' in line:
                        time_str = line.split('time=')[1].split(' ')[0]
                        return float(time_str)
            return 0.0
        except:
            return 0.0
    
    async def _count_recent_errors(self) -> tuple[int, int]:
        """统计最近的错误和警告数量"""
        error_count = 0
        warning_count = 0
        
        try:
            log_files = list(Path("logs").glob("*.log"))
            cutoff_time = datetime.now() - timedelta(hours=1)
            
            for log_file in log_files:
                async with aiofiles.open(log_file, 'r') as f:
                    content = await f.read()
                    lines = content.split('\n')
                    
                    for line in lines[-1000:]:  # 检查最后1000行
                        if 'ERROR' in line:
                            error_count += 1
                        elif 'WARNING' in line:
                            warning_count += 1
        except:
            pass
        
        return error_count, warning_count
    
    async def _signal_consistency_monitor(self):
        """信号执行一致性监控"""
        while self.monitoring_active:
            try:
                consistency_metrics = await self._analyze_signal_consistency()
                self.signal_consistency_history.extend(consistency_metrics)
                
                # 检查一致性告警
                for metric in consistency_metrics:
                    if metric.consistency_score < self.thresholds["consistency_score_warning"]:
                        await self._generate_alert(
                            severity="MEDIUM",
                            category="CONSISTENCY",
                            message=f"信号执行一致性低: {metric.consistency_score:.2f}",
                            affected_component="signal_execution",
                            suggested_action="检查信号延迟和执行路径"
                        )
                
                await asyncio.sleep(self.config["monitoring_interval"])
                
            except Exception as e:
                self.logger.error(f"信号一致性监控错误: {e}")
                await asyncio.sleep(30)
    
    async def _analyze_signal_consistency(self) -> List[SignalExecutionConsistency]:
        """分析信号执行一致性"""
        consistency_metrics = []
        
        try:
            # 读取最近的信号和执行数据
            signal_files = list(Path("results/model_training").glob("*signals*.json"))
            execution_files = list(Path("results/execution_reports").glob("*.json"))
            
            if not signal_files or not execution_files:
                return consistency_metrics
            
            # 读取最新的信号文件
            latest_signal_file = max(signal_files, key=lambda x: x.stat().st_mtime)
            with open(latest_signal_file, 'r') as f:
                signals_data = json.load(f)
            
            # 读取最新的执行文件
            latest_execution_file = max(execution_files, key=lambda x: x.stat().st_mtime)
            with open(latest_execution_file, 'r') as f:
                execution_data = json.load(f)
            
            # 比较信号和执行的一致性
            for signal in signals_data.get("signals", [])[-10:]:  # 检查最近10个信号
                signal_id = signal.get("signal_id", str(hash(str(signal))))
                signal_timestamp = signal.get("timestamp", "")
                signal_strength = signal.get("strength", 0.0)
                
                # 查找对应的执行记录
                execution_record = None
                for exec_rec in execution_data.get("executions", []):
                    if abs(float(exec_rec.get("timestamp", 0)) - float(signal.get("timestamp", 0))) < 300:  # 5分钟内
                        execution_record = exec_rec
                        break
                
                if execution_record:
                    execution_timestamp = execution_record.get("timestamp", "")
                    execution_delay = abs(float(execution_timestamp) - float(signal_timestamp))
                    target_position = signal.get("target_position", 0.0)
                    actual_position = execution_record.get("actual_position", 0.0)
                    
                    # 计算一致性分数
                    position_diff = abs(target_position - actual_position)
                    consistency_score = max(0, 1 - position_diff / max(abs(target_position), 1))
                    
                    consistency_metric = SignalExecutionConsistency(
                        signal_id=signal_id,
                        signal_timestamp=signal_timestamp,
                        signal_strength=signal_strength,
                        execution_timestamp=execution_timestamp,
                        execution_delay=execution_delay,
                        target_position=target_position,
                        actual_position=actual_position,
                        consistency_score=consistency_score,
                        deviation_reason=None if consistency_score > 0.9 else "仓位偏差过大"
                    )
                    
                    consistency_metrics.append(consistency_metric)
        
        except Exception as e:
            self.logger.error(f"信号一致性分析错误: {e}")
        
        return consistency_metrics
    
    async def _component_status_monitor(self):
        """组件状态监控"""
        while self.monitoring_active:
            try:
                for component, path in self.monitored_paths.items():
                    status = await self._check_component_status(component, path)
                    self.component_status[component] = status
                
                await asyncio.sleep(self.config["health_check_interval"])
                
            except Exception as e:
                self.logger.error(f"组件状态监控错误: {e}")
                await asyncio.sleep(60)
    
    async def _check_component_status(self, component: str, path: str) -> Dict[str, Any]:
        """检查组件状态"""
        status = {
            "component": component,
            "status": "UNKNOWN",
            "last_update": None,
            "file_count": 0,
            "total_size": 0,
            "health_score": 0.0
        }
        
        try:
            path_obj = Path(path)
            if path_obj.exists():
                files = list(path_obj.glob("**/*"))
                recent_files = [f for f in files if f.is_file() and 
                              (datetime.now() - datetime.fromtimestamp(f.stat().st_mtime)).days < 1]
                
                status.update({
                    "status": "ACTIVE" if recent_files else "STALE",
                    "last_update": max([f.stat().st_mtime for f in files if f.is_file()]) if files else None,
                    "file_count": len([f for f in files if f.is_file()]),
                    "total_size": sum([f.stat().st_size for f in files if f.is_file()]),
                    "health_score": min(1.0, len(recent_files) / max(1, len(files) * 0.1))
                })
            else:
                status["status"] = "MISSING"
        
        except Exception as e:
            status["status"] = "ERROR"
            status["error"] = str(e)
        
        return status
    
    async def _check_system_alerts(self, metrics: SystemHealthMetrics):
        """检查系统告警条件"""
        # CPU使用率告警
        if metrics.cpu_usage > self.thresholds["cpu_usage_critical"]:
            await self._generate_alert(
                severity="HIGH",
                category="SYSTEM",
                message=f"CPU使用率过高: {metrics.cpu_usage:.1f}%",
                affected_component="system",
                suggested_action="检查CPU密集型进程，考虑扩容"
            )
        
        # 内存使用率告警
        if metrics.memory_usage > self.thresholds["memory_usage_critical"]:
            await self._generate_alert(
                severity="HIGH",
                category="SYSTEM",
                message=f"内存使用率过高: {metrics.memory_usage:.1f}%",
                affected_component="system",
                suggested_action="检查内存泄漏，清理缓存"
            )
        
        # 磁盘使用率告警
        if metrics.disk_usage > self.thresholds["disk_usage_critical"]:
            await self._generate_alert(
                severity="CRITICAL",
                category="SYSTEM",
                message=f"磁盘使用率过高: {metrics.disk_usage:.1f}%",
                affected_component="system",
                suggested_action="清理日志文件，扩展存储空间"
            )
        
        # 错误率告警
        total_operations = max(1, metrics.error_count + metrics.warning_count + 100)
        error_rate = metrics.error_count / total_operations
        if error_rate > self.thresholds["error_rate_critical"]:
            await self._generate_alert(
                severity="HIGH",
                category="SYSTEM",
                message=f"错误率过高: {error_rate:.1%}",
                affected_component="system",
                suggested_action="检查系统日志，修复错误源"
            )
    
    async def _generate_alert(self, severity: str, category: str, message: str, 
                            affected_component: str, suggested_action: str):
        """生成告警"""
        alert_id = hashlib.md5(f"{category}_{message}_{affected_component}".encode()).hexdigest()[:8]
        
        # 检查重复告警（冷却期）
        recent_alerts = [a for a in self.alerts_history 
                        if (datetime.now() - datetime.fromisoformat(a.timestamp)).seconds < self.config["alert_cooldown"]]
        
        if any(a.message == message for a in recent_alerts):
            return  # 跳过重复告警
        
        alert = RiskAlert(
            alert_id=alert_id,
            timestamp=datetime.now().isoformat(),
            severity=severity,
            category=category,
            message=message,
            affected_component=affected_component,
            suggested_action=suggested_action
        )
        
        self.alerts_history.append(alert)
        self.logger.warning(f"告警生成 [{severity}] {category}: {message}")
        
        # 保存告警到文件
        await self._save_alert(alert)
    
    async def _save_alert(self, alert: RiskAlert):
        """保存告警到文件"""
        try:
            alerts_dir = Path("logs/alerts")
            alerts_dir.mkdir(exist_ok=True, parents=True)
            
            alert_file = alerts_dir / f"alerts_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            async with aiofiles.open(alert_file, 'a') as f:
                await f.write(json.dumps(asdict(alert), ensure_ascii=False) + '\n')
        
        except Exception as e:
            self.logger.error(f"保存告警失败: {e}")
    
    async def _alert_manager(self):
        """告警管理器"""
        while self.monitoring_active:
            try:
                # 清理过期告警
                cutoff_time = datetime.now() - timedelta(days=7)
                self.alerts_history = [
                    a for a in self.alerts_history 
                    if datetime.fromisoformat(a.timestamp) > cutoff_time
                ]
                
                await asyncio.sleep(3600)  # 每小时清理一次
                
            except Exception as e:
                self.logger.error(f"告警管理器错误: {e}")
                await asyncio.sleep(300)
    
    async def _report_generator(self):
        """报告生成器"""
        while self.monitoring_active:
            try:
                current_hour = datetime.now().hour
                if current_hour in self.config["report_generation_hours"]:
                    await self._generate_comprehensive_report()
                    await asyncio.sleep(3600)  # 等待1小时避免重复生成
                
                await asyncio.sleep(300)  # 每5分钟检查一次
                
            except Exception as e:
                self.logger.error(f"报告生成器错误: {e}")
                await asyncio.sleep(600)
    
    async def _generate_comprehensive_report(self):
        """生成综合监控报告"""
        try:
            report = {
                "report_id": f"monitor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "timestamp": datetime.now().isoformat(),
                "reporting_period": "24h",
                
                # 系统健康概述
                "system_health_summary": self._summarize_system_health(),
                
                # 信号一致性概述
                "signal_consistency_summary": self._summarize_signal_consistency(),
                
                # 组件状态概述
                "component_status_summary": self.component_status,
                
                # 告警概述
                "alerts_summary": self._summarize_alerts(),
                
                # 建议和行动项
                "recommendations": self._generate_recommendations()
            }
            
            # 保存报告
            reports_dir = Path("results/monitoring_reports")
            reports_dir.mkdir(exist_ok=True, parents=True)
            
            report_file = reports_dir / f"{report['report_id']}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"生成监控报告: {report_file}")
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
    
    def _summarize_system_health(self) -> Dict[str, Any]:
        """总结系统健康状况"""
        if not self.system_metrics_history:
            return {"status": "no_data"}
        
        recent_metrics = self.system_metrics_history[-24:]  # 最近24个数据点
        
        return {
            "avg_cpu_usage": np.mean([m.cpu_usage for m in recent_metrics]),
            "max_cpu_usage": np.max([m.cpu_usage for m in recent_metrics]),
            "avg_memory_usage": np.mean([m.memory_usage for m in recent_metrics]),
            "max_memory_usage": np.max([m.memory_usage for m in recent_metrics]),
            "avg_disk_usage": np.mean([m.disk_usage for m in recent_metrics]),
            "total_errors": sum([m.error_count for m in recent_metrics]),
            "total_warnings": sum([m.warning_count for m in recent_metrics]),
            "uptime_hours": recent_metrics[-1].uptime / 3600 if recent_metrics else 0
        }
    
    def _summarize_signal_consistency(self) -> Dict[str, Any]:
        """总结信号一致性"""
        if not self.signal_consistency_history:
            return {"status": "no_data"}
        
        recent_consistency = self.signal_consistency_history[-50:]  # 最近50个信号
        
        return {
            "avg_consistency_score": np.mean([c.consistency_score for c in recent_consistency]),
            "min_consistency_score": np.min([c.consistency_score for c in recent_consistency]),
            "avg_execution_delay": np.mean([c.execution_delay for c in recent_consistency if c.execution_delay]),
            "signals_with_issues": len([c for c in recent_consistency if c.consistency_score < 0.8]),
            "total_signals_analyzed": len(recent_consistency)
        }
    
    def _summarize_alerts(self) -> Dict[str, Any]:
        """总结告警情况"""
        recent_alerts = [a for a in self.alerts_history 
                        if (datetime.now() - datetime.fromisoformat(a.timestamp)).days < 1]
        
        return {
            "total_alerts_24h": len(recent_alerts),
            "critical_alerts": len([a for a in recent_alerts if a.severity == "CRITICAL"]),
            "high_alerts": len([a for a in recent_alerts if a.severity == "HIGH"]),
            "medium_alerts": len([a for a in recent_alerts if a.severity == "MEDIUM"]),
            "alerts_by_category": {
                category: len([a for a in recent_alerts if a.category == category])
                for category in set([a.category for a in recent_alerts])
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议和行动项"""
        recommendations = []
        
        # 基于系统健康生成建议
        if self.system_metrics_history:
            latest_metrics = self.system_metrics_history[-1]
            
            if latest_metrics.cpu_usage > 80:
                recommendations.append("CPU使用率较高，建议优化算法或扩容")
            
            if latest_metrics.memory_usage > 80:
                recommendations.append("内存使用率较高，建议检查内存泄漏")
            
            if latest_metrics.error_count > 10:
                recommendations.append("错误数量较多，建议检查系统日志")
        
        # 基于信号一致性生成建议
        if self.signal_consistency_history:
            recent_consistency = self.signal_consistency_history[-10:]
            avg_score = np.mean([c.consistency_score for c in recent_consistency])
            
            if avg_score < 0.8:
                recommendations.append("信号执行一致性较低，建议检查执行路径")
        
        # 基于组件状态生成建议
        for component, status in self.component_status.items():
            if status.get("status") == "STALE":
                recommendations.append(f"{component}组件数据陈旧，建议检查更新机制")
            elif status.get("status") == "ERROR":
                recommendations.append(f"{component}组件异常，建议立即检查")
        
        return recommendations if recommendations else ["系统运行正常，无特殊建议"]

# 运行接口
async def run_continuous_monitoring():
    """运行持续监控"""
    monitor = ContinuousSystemMonitor()
    
    try:
        await monitor.start_monitoring()
    except KeyboardInterrupt:
        await monitor.stop_monitoring()
        print("监控已停止")

if __name__ == "__main__":
    print("🚀 启动DipMaster持续系统监控...")
    asyncio.run(run_continuous_monitoring())