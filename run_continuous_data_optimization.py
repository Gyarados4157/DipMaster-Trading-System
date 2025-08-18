#!/usr/bin/env python3
"""
Continuous Data Infrastructure Optimization Runner
DipMaster Trading System - 持续数据基础设施优化运行器

使用方法:
    python run_continuous_data_optimization.py --start
    python run_continuous_data_optimization.py --status
    python run_continuous_data_optimization.py --stop
    python run_continuous_data_optimization.py --report
"""

import asyncio
import argparse
import sys
import signal
import json
from pathlib import Path
from datetime import datetime, timezone
import logging

# 添加src路径到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.continuous_data_infrastructure_optimizer import ContinuousDataInfrastructureOptimizer

class OptimizationManager:
    """优化管理器"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or "config/continuous_data_optimization_config.yaml"
        self.optimizer = None
        self.running = False
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 设置日志
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self):
        """设置日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "optimization_manager.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        self.logger.info(f"收到信号 {signum}，正在停止优化服务...")
        self.stop_optimization()
    
    async def start_optimization(self):
        """启动优化服务"""
        if self.running:
            self.logger.warning("优化服务已经在运行中")
            return
        
        try:
            self.logger.info("启动持续数据基础设施优化服务...")
            
            # 创建优化器实例
            self.optimizer = ContinuousDataInfrastructureOptimizer(self.config_path)
            
            # 标记为运行状态
            self.running = True
            
            # 启动优化循环
            await self.optimizer.start_continuous_optimization()
            
        except Exception as e:
            self.logger.error(f"启动优化服务失败: {e}")
            self.running = False
            raise
    
    def stop_optimization(self):
        """停止优化服务"""
        if not self.running:
            self.logger.info("优化服务未运行")
            return
        
        self.logger.info("停止优化服务...")
        
        if self.optimizer:
            self.optimizer.stop_optimization()
        
        self.running = False
        self.logger.info("优化服务已停止")
    
    def get_status(self) -> dict:
        """获取状态"""
        if not self.optimizer:
            return {
                'status': 'not_initialized',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        if not self.running:
            return {
                'status': 'stopped',
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        return self.optimizer.get_optimization_status()
    
    async def generate_report(self) -> dict:
        """生成报告"""
        if not self.optimizer:
            self.logger.warning("优化器未初始化，创建临时实例生成报告")
            temp_optimizer = ContinuousDataInfrastructureOptimizer(self.config_path)
            report = temp_optimizer.generate_infrastructure_report()
            
            # 执行快速质量评估
            quality_report = await temp_optimizer.comprehensive_quality_assessment()
            report['quality_assessment'] = quality_report
            
            return report
        
        return self.optimizer.generate_infrastructure_report()
    
    async def run_initial_collection(self):
        """运行初始数据收集"""
        self.logger.info("执行初始数据收集...")
        
        if not self.optimizer:
            self.optimizer = ContinuousDataInfrastructureOptimizer(self.config_path)
        
        await self.optimizer.initial_data_collection()
        self.logger.info("初始数据收集完成")

def create_systemd_service():
    """创建systemd服务文件"""
    service_content = f"""[Unit]
Description=DipMaster Continuous Data Infrastructure Optimizer
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=ubuntu
Group=ubuntu
WorkingDirectory={Path.cwd()}
Environment=PYTHONPATH={Path.cwd()}
ExecStart=/usr/bin/python3 {Path.cwd() / 'run_continuous_data_optimization.py'} --start
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
    
    service_path = Path("/etc/systemd/system/dipmaster-data-optimizer.service")
    
    try:
        with open(service_path, 'w') as f:
            f.write(service_content)
        
        print(f"Systemd服务文件已创建: {service_path}")
        print("启用和启动服务:")
        print("  sudo systemctl daemon-reload")
        print("  sudo systemctl enable dipmaster-data-optimizer")
        print("  sudo systemctl start dipmaster-data-optimizer")
        print("  sudo systemctl status dipmaster-data-optimizer")
        
    except PermissionError:
        print("创建systemd服务文件需要sudo权限")
        print(f"请手动创建 {service_path} 文件，内容如下:")
        print(service_content)

def print_status(status: dict):
    """打印状态信息"""
    print("\n=== DipMaster 数据优化器状态 ===")
    print(f"状态: {status.get('status', 'unknown')}")
    print(f"检查时间: {status.get('last_check', 'N/A')}")
    
    if 'performance_metrics' in status:
        metrics = status['performance_metrics']
        print(f"\n性能指标:")
        print(f"  总币种数: {metrics.get('total_symbols', 0)}")
        print(f"  总时间框架: {metrics.get('total_timeframes', 0)}")
        print(f"  数据质量评分: {metrics.get('data_quality_score', 0):.3f}")
        print(f"  最后更新: {metrics.get('last_update_time', 'N/A')}")
        print(f"  发现gaps: {metrics.get('gaps_detected', 0)}")
        print(f"  修复gaps: {metrics.get('gaps_fixed', 0)}")
        print(f"  失败更新: {metrics.get('failed_updates', 0)}")
    
    if 'queue_size' in status:
        print(f"队列大小: {status['queue_size']}")
    
    if 'quality_issues' in status:
        print(f"质量问题: {status['quality_issues']}")

async def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="DipMaster Continuous Data Infrastructure Optimizer"
    )
    parser.add_argument(
        '--config', 
        default='config/continuous_data_optimization_config.yaml',
        help='配置文件路径'
    )
    
    # 互斥的操作选项
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--start', action='store_true', help='启动优化服务')
    action.add_argument('--status', action='store_true', help='查看服务状态')
    action.add_argument('--stop', action='store_true', help='停止优化服务')
    action.add_argument('--report', action='store_true', help='生成基础设施报告')
    action.add_argument('--initial-collection', action='store_true', help='执行初始数据收集')
    action.add_argument('--create-service', action='store_true', help='创建systemd服务')
    
    args = parser.parse_args()
    
    # 创建管理器
    manager = OptimizationManager(args.config)
    
    try:
        if args.start:
            print("启动持续数据基础设施优化服务...")
            await manager.start_optimization()
        
        elif args.status:
            status = manager.get_status()
            print_status(status)
        
        elif args.stop:
            manager.stop_optimization()
        
        elif args.report:
            print("生成基础设施报告...")
            report = await manager.generate_report()
            
            # 保存报告
            report_file = f"data/optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)
            
            print(f"报告已保存到: {report_file}")
            
            # 打印关键信息
            print(f"\n=== 基础设施报告摘要 ===")
            print(f"状态: {report.get('infrastructure_status', 'unknown')}")
            print(f"配置币种: {report.get('symbols', {}).get('configured', 0)}")
            print(f"活跃币种: {report.get('symbols', {}).get('active', 0)}")
            print(f"数据文件: {report.get('data_coverage', {}).get('total_files', 0)}")
            print(f"总大小: {report.get('data_coverage', {}).get('total_size_gb', 0):.2f} GB")
            
            if 'quality_assessment' in report:
                qa = report['quality_assessment']
                print(f"整体质量评分: {qa.get('overall_metrics', {}).get('overall_score', 0):.3f}")
        
        elif args.initial_collection:
            print("执行初始数据收集...")
            await manager.run_initial_collection()
        
        elif args.create_service:
            create_systemd_service()
        
    except KeyboardInterrupt:
        print("\n收到中断信号...")
        manager.stop_optimization()
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())