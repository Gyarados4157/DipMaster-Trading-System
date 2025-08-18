#!/usr/bin/env python3
"""
DipMaster Comprehensive Monitoring System Integration Runner
全面监控系统集成运行器 - 完整的监控生态系统启动和集成脚本

Features:
- Integrated startup of all monitoring components
- Configuration management and validation
- Health checks and failover capabilities
- Seamless integration with existing trading system
- Auto-recovery and monitoring system orchestration
- Production-ready deployment configuration

Author: DipMaster Trading System Monitoring Agent
Date: 2025-08-18
Version: 2.0.0
"""

import asyncio
import sys
import os
import json
import logging
import argparse
import signal
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.monitoring.comprehensive_monitoring_system import (
    ComprehensiveMonitoringSystem,
    create_comprehensive_monitoring_system,
    MonitoringMode
)
from src.monitoring.monitoring_dashboard_service import (
    MonitoringDashboardService,
    create_monitoring_dashboard_service
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/comprehensive_monitoring.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)


class MonitoringSystemOrchestrator:
    """
    监控系统编排器
    
    负责协调启动、管理和监控所有监控系统组件，
    提供完整的监控生态系统管理。
    """
    
    def __init__(self, config_path: Optional[str] = None, mode: str = "development"):
        """
        初始化监控系统编排器
        
        Args:
            config_path: 配置文件路径
            mode: 运行模式 (development, paper_trading, live_trading, research_only)
        """
        self.config_path = config_path
        self.mode = MonitoringMode(mode)
        self.config = {}
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # 核心组件
        self.monitoring_system: Optional[ComprehensiveMonitoringSystem] = None
        self.dashboard_service: Optional[MonitoringDashboardService] = None
        
        # 统计信息
        self.stats = {
            'start_time': None,
            'signals_processed': 0,
            'positions_tracked': 0,
            'executions_monitored': 0,
            'reports_generated': 0,
            'alerts_generated': 0,
            'uptime_seconds': 0
        }
        
        # 加载配置
        self._load_configuration()
        
        # 设置信号处理
        self._setup_signal_handlers()
        
        logger.info(f"🚀 MonitoringSystemOrchestrator initialized in {self.mode.value} mode")
    
    def _load_configuration(self):
        """加载配置文件"""
        try:
            # 默认配置
            default_config = {
                'mode': self.mode.value,
                'kafka': {
                    'enabled': True,
                    'servers': ['localhost:9092'],
                    'client_id': 'dipmaster-monitoring',
                    'buffer_max_size': 1000
                },
                'database': {
                    'path': 'data/monitoring_production.db',
                    'backup_enabled': True,
                    'retention_days': 90
                },
                'logging': {
                    'level': 'INFO',
                    'directory': 'logs',
                    'max_file_size_mb': 100,
                    'retention_days': 30,
                    'structured_logging': True
                },
                'monitoring': {
                    'consistency_check_interval_seconds': 30,
                    'drift_detection_interval_seconds': 300,
                    'risk_monitoring_interval_seconds': 60,
                    'health_check_interval_seconds': 60,
                    'performance_snapshot_interval_seconds': 900,
                    'report_generation_enabled': True
                },
                'dashboard': {
                    'enabled': True,
                    'host': '0.0.0.0',
                    'port': 8080,
                    'websocket_port': 8081,
                    'cache_duration_seconds': 30,
                    'cors_enabled': True,
                    'start_server': True
                },
                'thresholds': {
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
                },
                'dipmaster_params': {
                    'rsi_range': [30, 50],
                    'max_holding_minutes': 180,
                    'boundary_minutes': [15, 30, 45, 0],
                    'target_profit_pct': 0.8,
                    'dip_threshold_pct': 0.2,
                    'volume_multiplier': 1.5
                },
                'alerts': {
                    'email_notifications': False,
                    'webhook_url': None,
                    'suppression_enabled': True,
                    'escalation_enabled': True,
                    'auto_resolution_enabled': True
                },
                'backup': {
                    'enabled': True,
                    'interval_hours': 6,
                    'retention_days': 30,
                    'compress': True
                }
            }
            
            # 从配置文件加载
            if self.config_path and Path(self.config_path).exists():
                if self.config_path.endswith('.json'):
                    with open(self.config_path, 'r') as f:
                        file_config = json.load(f)
                elif self.config_path.endswith(('.yml', '.yaml')):
                    with open(self.config_path, 'r') as f:
                        file_config = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path}")
                
                # 合并配置
                self.config = self._deep_merge_dicts(default_config, file_config)
                logger.info(f"✅ Configuration loaded from {self.config_path}")
                
            else:
                self.config = default_config
                logger.info("✅ Using default configuration")
            
            # 验证配置
            self._validate_configuration()
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            self.config = {}
            raise
    
    def _deep_merge_dicts(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dicts(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_configuration(self):
        """验证配置有效性"""
        required_keys = ['mode', 'kafka', 'database', 'monitoring', 'thresholds']
        
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required configuration key: {key}")
        
        # 创建必需的目录
        log_dir = Path(self.config['logging']['directory'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        db_path = Path(self.config['database']['path'])
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Configuration validation completed")
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self._shutdown_gracefully())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def start(self):
        """启动完整的监控系统"""
        if self.is_running:
            logger.warning("⚠️ Monitoring system already running")
            return
        
        try:
            self.is_running = True
            self.stats['start_time'] = datetime.now(timezone.utc).timestamp()
            
            logger.info("🚀 Starting DipMaster Comprehensive Monitoring System...")
            
            # 1. 初始化和启动核心监控系统
            await self._start_monitoring_system()
            
            # 2. 初始化和启动仪表板服务
            if self.config['dashboard']['enabled']:
                await self._start_dashboard_service()
            
            # 3. 运行健康检查
            await self._perform_startup_health_check()
            
            # 4. 启动集成监控循环
            await self._start_integration_monitoring()
            
            logger.info("✅ DipMaster Comprehensive Monitoring System started successfully")
            
            # 等待关闭信号
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring system: {e}")
            self.is_running = False
            raise
    
    async def _start_monitoring_system(self):
        """启动核心监控系统"""
        try:
            logger.info("🔍 Initializing comprehensive monitoring system...")
            
            monitoring_config = {
                'mode': self.config['mode'],
                'kafka': self.config['kafka'],
                'db_path': self.config['database']['path'],
                'log_dir': self.config['logging']['directory'],
                'thresholds': self.config['thresholds'],
                'dipmaster_params': self.config['dipmaster_params'],
                'baseline_performance_file': self.config.get('baseline_performance_file')
            }
            
            self.monitoring_system = create_comprehensive_monitoring_system(monitoring_config)
            await self.monitoring_system.start()
            
            logger.info("✅ Comprehensive monitoring system started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start monitoring system: {e}")
            raise
    
    async def _start_dashboard_service(self):
        """启动仪表板服务"""
        try:
            logger.info("🎯 Initializing monitoring dashboard service...")
            
            if not self.monitoring_system:
                raise RuntimeError("Monitoring system must be started before dashboard service")
            
            dashboard_config = self.config['dashboard'].copy()
            
            self.dashboard_service = create_monitoring_dashboard_service(
                self.monitoring_system,
                dashboard_config
            )
            
            await self.dashboard_service.start()
            
            logger.info(f"✅ Monitoring dashboard service started on {dashboard_config['host']}:{dashboard_config['port']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to start dashboard service: {e}")
            raise
    
    async def _perform_startup_health_check(self):
        """执行启动健康检查"""
        try:
            logger.info("🏥 Performing startup health check...")
            
            # 检查监控系统状态
            if self.monitoring_system:
                system_status = await self.monitoring_system.get_system_status()
                if not system_status['is_running']:
                    raise RuntimeError("Monitoring system not running after startup")
                
                logger.info(f"✅ Monitoring system health: {system_status['system_health_score']:.1f}%")
            
            # 检查仪表板服务状态
            if self.dashboard_service:
                dashboard_stats = self.dashboard_service.get_monitoring_statistics()
                logger.info(f"✅ Dashboard service health: Active")
            
            # 检查Kafka连接
            if self.config['kafka']['enabled'] and self.monitoring_system:
                if hasattr(self.monitoring_system, 'kafka_streamer') and self.monitoring_system.kafka_streamer:
                    kafka_stats = self.monitoring_system.kafka_streamer.get_event_stats()
                    if kafka_stats['is_running']:
                        logger.info("✅ Kafka connection: Active")
                    else:
                        logger.warning("⚠️ Kafka connection: Inactive")
            
            logger.info("✅ Startup health check completed")
            
        except Exception as e:
            logger.error(f"❌ Startup health check failed: {e}")
            raise
    
    async def _start_integration_monitoring(self):
        """启动集成监控循环"""
        try:
            logger.info("🔄 Starting integration monitoring loop...")
            
            # 启动后台任务
            background_tasks = [
                asyncio.create_task(self._integration_health_monitor()),
                asyncio.create_task(self._statistics_updater()),
                asyncio.create_task(self._backup_manager()),
                asyncio.create_task(self._log_rotator())
            ]
            
            # 等待任务完成或关闭信号
            done, pending = await asyncio.wait(
                background_tasks + [asyncio.create_task(self.shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # 取消待处理的任务
            for task in pending:
                task.cancel()
            
            logger.info("✅ Integration monitoring loop completed")
            
        except Exception as e:
            logger.error(f"❌ Integration monitoring loop failed: {e}")
    
    async def _integration_health_monitor(self):
        """集成健康监控器"""
        while self.is_running:
            try:
                # 检查组件健康状态
                if self.monitoring_system:
                    health_status = await self.monitoring_system.get_health_status()
                    if health_status['status'] != 'healthy':
                        logger.warning(f"⚠️ Monitoring system health degraded: {health_status}")
                
                # 检查仪表板服务健康状态
                if self.dashboard_service:
                    # 简单的心跳检查
                    try:
                        stats = self.dashboard_service.get_monitoring_statistics()
                        if not stats:
                            logger.warning("⚠️ Dashboard service may be unresponsive")
                    except Exception as e:
                        logger.warning(f"⚠️ Dashboard service health check failed: {e}")
                
                await asyncio.sleep(60)  # 每分钟检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in integration health monitor: {e}")
                await asyncio.sleep(30)
    
    async def _statistics_updater(self):
        """统计信息更新器"""
        while self.is_running:
            try:
                if self.monitoring_system:
                    system_stats = self.monitoring_system.get_monitoring_statistics()
                    
                    self.stats['signals_processed'] = system_stats['system_stats']['signals_validated']
                    self.stats['positions_tracked'] = system_stats['system_stats']['positions_tracked']
                    self.stats['executions_monitored'] = system_stats['system_stats']['executions_monitored']
                    self.stats['alerts_generated'] = system_stats['system_stats']['alerts_generated']
                    self.stats['reports_generated'] = system_stats['system_stats']['reports_generated']
                
                if self.stats['start_time']:
                    self.stats['uptime_seconds'] = datetime.now(timezone.utc).timestamp() - self.stats['start_time']
                
                await asyncio.sleep(30)  # 每30秒更新统计
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error updating statistics: {e}")
                await asyncio.sleep(10)
    
    async def _backup_manager(self):
        """备份管理器"""
        if not self.config['backup']['enabled']:
            return
        
        backup_interval = self.config['backup']['interval_hours'] * 3600
        
        while self.is_running:
            try:
                await self._perform_backup()
                await asyncio.sleep(backup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in backup manager: {e}")
                await asyncio.sleep(3600)  # 错误后1小时重试
    
    async def _perform_backup(self):
        """执行备份"""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_dir = Path(f"backups/monitoring_{timestamp}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 备份数据库
            db_path = Path(self.config['database']['path'])
            if db_path.exists():
                backup_db_path = backup_dir / f"monitoring_{timestamp}.db"
                import shutil
                shutil.copy2(db_path, backup_db_path)
                
                # 压缩备份
                if self.config['backup']['compress']:
                    import gzip
                    with open(backup_db_path, 'rb') as f_in:
                        with gzip.open(f"{backup_db_path}.gz", 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    backup_db_path.unlink()  # 删除未压缩文件
                    
                logger.info(f"✅ Database backup completed: {backup_dir}")
            
            # 清理旧备份
            await self._cleanup_old_backups()
            
        except Exception as e:
            logger.error(f"❌ Failed to perform backup: {e}")
    
    async def _cleanup_old_backups(self):
        """清理旧备份"""
        try:
            backup_root = Path("backups")
            if not backup_root.exists():
                return
            
            retention_days = self.config['backup']['retention_days']
            cutoff_time = datetime.now().timestamp() - (retention_days * 24 * 3600)
            
            for backup_dir in backup_root.iterdir():
                if backup_dir.is_dir() and backup_dir.stat().st_mtime < cutoff_time:
                    import shutil
                    shutil.rmtree(backup_dir)
                    logger.info(f"🗑️ Removed old backup: {backup_dir}")
            
        except Exception as e:
            logger.error(f"❌ Failed to cleanup old backups: {e}")
    
    async def _log_rotator(self):
        """日志轮转器"""
        while self.is_running:
            try:
                log_dir = Path(self.config['logging']['directory'])
                max_size = self.config['logging']['max_file_size_mb'] * 1024 * 1024
                retention_days = self.config['logging']['retention_days']
                
                # 检查日志文件大小
                for log_file in log_dir.glob("*.log"):
                    if log_file.stat().st_size > max_size:
                        # 轮转日志文件
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        rotated_name = log_file.with_suffix(f".{timestamp}.log")
                        log_file.rename(rotated_name)
                        logger.info(f"📝 Rotated log file: {log_file} -> {rotated_name}")
                
                # 清理旧日志
                cutoff_time = datetime.now().timestamp() - (retention_days * 24 * 3600)
                for log_file in log_dir.glob("*.log*"):
                    if log_file.stat().st_mtime < cutoff_time:
                        log_file.unlink()
                        logger.info(f"🗑️ Removed old log file: {log_file}")
                
                await asyncio.sleep(3600)  # 每小时检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in log rotator: {e}")
                await asyncio.sleep(1800)  # 错误后30分钟重试
    
    async def stop(self):
        """停止监控系统"""
        if not self.is_running:
            logger.warning("⚠️ Monitoring system not running")
            return
        
        try:
            logger.info("🛑 Stopping DipMaster Comprehensive Monitoring System...")
            
            self.is_running = False
            
            # 停止仪表板服务
            if self.dashboard_service:
                await self.dashboard_service.stop()
                logger.info("✅ Dashboard service stopped")
            
            # 停止监控系统
            if self.monitoring_system:
                await self.monitoring_system.stop()
                logger.info("✅ Monitoring system stopped")
            
            # 最终备份
            if self.config['backup']['enabled']:
                logger.info("📦 Performing final backup...")
                await self._perform_backup()
            
            # 生成最终报告
            await self._generate_shutdown_report()
            
            logger.info("✅ DipMaster Comprehensive Monitoring System stopped successfully")
            
        except Exception as e:
            logger.error(f"❌ Error stopping monitoring system: {e}")
    
    async def _shutdown_gracefully(self):
        """优雅关闭"""
        try:
            await self.stop()
            self.shutdown_event.set()
        except Exception as e:
            logger.error(f"❌ Error during graceful shutdown: {e}")
            self.shutdown_event.set()
    
    async def _generate_shutdown_report(self):
        """生成关闭报告"""
        try:
            uptime_hours = self.stats['uptime_seconds'] / 3600
            
            report = {
                'shutdown_time': datetime.now(timezone.utc).isoformat(),
                'total_uptime_hours': uptime_hours,
                'statistics': self.stats.copy(),
                'mode': self.mode.value,
                'final_status': 'clean_shutdown'
            }
            
            # 保存报告
            report_path = Path('reports/monitoring/shutdown_reports')
            report_path.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_file = report_path / f"shutdown_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"📊 Shutdown report generated: {report_file}")
            logger.info(f"📊 Final statistics: {self.stats['signals_processed']} signals, "
                       f"{self.stats['positions_tracked']} positions, "
                       f"{self.stats['alerts_generated']} alerts, "
                       f"{uptime_hours:.1f}h uptime")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate shutdown report: {e}")
    
    # Integration methods for existing trading system
    
    async def record_trading_signal(self, signal_data: Dict[str, Any]):
        """记录交易信号（集成接口）"""
        if self.monitoring_system:
            await self.monitoring_system.record_signal(signal_data)
    
    async def record_trading_position(self, position_data: Dict[str, Any]):
        """记录交易持仓（集成接口）"""
        if self.monitoring_system:
            await self.monitoring_system.record_position(position_data)
    
    async def record_order_execution(self, execution_data: Dict[str, Any]):
        """记录订单执行（集成接口）"""
        if self.monitoring_system:
            await self.monitoring_system.record_execution(execution_data)
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态（集成接口）"""
        return {
            'is_running': self.is_running,
            'mode': self.mode.value,
            'uptime_seconds': self.stats['uptime_seconds'],
            'statistics': self.stats.copy(),
            'components': {
                'monitoring_system': self.monitoring_system is not None,
                'dashboard_service': self.dashboard_service is not None
            }
        }


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DipMaster Comprehensive Monitoring System")
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file path (JSON or YAML)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='development',
        choices=['development', 'paper_trading', 'live_trading', 'research_only'],
        help='Running mode'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # 设置日志级别
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # 创建并启动监控系统编排器
    orchestrator = MonitoringSystemOrchestrator(args.config, args.mode)
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("🛑 Received keyboard interrupt, shutting down...")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1
    finally:
        await orchestrator.stop()
    
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))