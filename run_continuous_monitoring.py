#!/usr/bin/env python3
"""
DipMaster Trading System - 持续监控运行器
创建时间: 2025-08-18
版本: 1.0.0

功能: 启动和管理DipMaster系统的持续监控
"""

import asyncio
import argparse
import json
import logging
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.monitoring.continuous_system_monitor import ContinuousSystemMonitor

class MonitoringOrchestrator:
    """监控编排器"""
    
    def __init__(self):
        self.monitor = None
        self.running = False
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/monitoring_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger("MonitoringOrchestrator")
    
    async def start_monitoring(self, config_path: str = None):
        """启动监控"""
        try:
            self.logger.info("🚀 启动DipMaster持续监控系统...")
            
            # 创建监控器
            self.monitor = ContinuousSystemMonitor(config_path) if config_path else ContinuousSystemMonitor()
            
            # 设置信号处理
            self._setup_signal_handlers()
            
            # 启动监控
            self.running = True
            await self.monitor.start_monitoring()
            
        except Exception as e:
            self.logger.error(f"监控启动失败: {e}")
            raise
    
    async def stop_monitoring(self):
        """停止监控"""
        if self.monitor and self.running:
            self.logger.info("🛑 停止监控系统...")
            self.running = False
            await self.monitor.stop_monitoring()
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(signum, frame):
            self.logger.info(f"接收到信号 {signum}，开始优雅关闭...")
            asyncio.create_task(self.stop_monitoring())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def status_check(self):
        """检查监控状态"""
        if not self.monitor:
            return {"status": "not_running", "message": "监控未启动"}
        
        # 获取最新状态
        status = {
            "status": "running" if self.running else "stopped",
            "timestamp": datetime.now().isoformat(),
            "system_metrics_count": len(self.monitor.system_metrics_history),
            "signal_consistency_count": len(self.monitor.signal_consistency_history),
            "alerts_count": len(self.monitor.alerts_history),
            "component_status": self.monitor.component_status
        }
        
        return status

def create_monitoring_config():
    """创建默认监控配置"""
    config = {
        "monitoring_interval": 60,
        "health_check_interval": 300,
        "alert_cooldown": 1800,
        "max_history_size": 1000,
        "kafka_enabled": False,
        "report_generation_hours": [8, 16, 0],
        "thresholds": {
            "cpu_usage_critical": 90.0,
            "memory_usage_critical": 85.0,
            "disk_usage_critical": 90.0,
            "signal_delay_warning": 5.0,
            "consistency_score_warning": 0.8,
            "error_rate_critical": 0.05
        },
        "monitored_components": [
            "data_infrastructure",
            "feature_engineering", 
            "model_training",
            "portfolio_optimization",
            "execution_reports"
        ]
    }
    
    config_path = Path("config/monitoring_config.json")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 创建监控配置文件: {config_path}")
    return config_path

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DipMaster持续监控系统")
    parser.add_argument('--config', '-c', help='监控配置文件路径')
    parser.add_argument('--create-config', action='store_true', help='创建默认配置文件')
    parser.add_argument('--status', action='store_true', help='检查监控状态')
    parser.add_argument('--daemon', action='store_true', help='后台运行模式')
    
    args = parser.parse_args()
    
    # 创建配置文件
    if args.create_config:
        create_monitoring_config()
        return
    
    # 创建编排器
    orchestrator = MonitoringOrchestrator()
    
    # 检查状态
    if args.status:
        status = await orchestrator.status_check()
        print(json.dumps(status, indent=2, ensure_ascii=False))
        return
    
    # 启动监控
    try:
        if args.daemon:
            print("🔄 后台模式启动监控...")
        else:
            print("🔄 前台模式启动监控（Ctrl+C停止）...")
        
        await orchestrator.start_monitoring(args.config)
        
    except KeyboardInterrupt:
        print("\n📋 接收到中断信号...")
    except Exception as e:
        print(f"❌ 监控异常: {e}")
    finally:
        await orchestrator.stop_monitoring()
        print("✅ 监控已停止")

if __name__ == "__main__":
    # 创建必要目录
    for directory in ["logs", "config", "results/monitoring_reports"]:
        Path(directory).mkdir(exist_ok=True, parents=True)
    
    # 运行主程序
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 再见！")
    except Exception as e:
        print(f"💥 系统错误: {e}")
        sys.exit(1)