#!/usr/bin/env python3
"""
Setup Script for Continuous Data Infrastructure Optimization
DipMaster Trading System - 持续数据基础设施优化设置脚本

This script sets up and launches the continuous data infrastructure optimization system
with monitoring, alerting, and automated quality management.
"""

import asyncio
import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.continuous_data_infrastructure_optimizer import ContinuousDataInfrastructureOptimizer
from src.data.data_infrastructure_monitoring import DataInfrastructureMonitor
import threading

class ContinuousOptimizationSetup:
    """持续优化设置管理器"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # 核心组件
        self.optimizer = None
        self.monitor = None
        
        # 运行状态
        self.running = False
        self.setup_complete = False
        
        # 配置路径
        self.config_path = "config/continuous_data_optimization_config.yaml"
        
    def setup_logging(self):
        """设置日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "continuous_optimization_setup.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def check_dependencies(self) -> bool:
        """检查依赖项"""
        self.logger.info("检查系统依赖项...")
        
        required_dirs = [
            "data",
            "data/enhanced_market_data", 
            "logs",
            "config"
        ]
        
        missing_dirs = []
        for dir_path in required_dirs:
            path = Path(dir_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"创建目录: {path}")
        
        # 检查配置文件
        if not Path(self.config_path).exists():
            self.logger.error(f"配置文件不存在: {self.config_path}")
            return False
        
        # 检查Python依赖
        required_packages = [
            'pandas', 'numpy', 'ccxt', 'pyarrow', 
            'aiohttp', 'asyncio', 'schedule', 'yaml'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            self.logger.error(f"缺少Python包: {', '.join(missing_packages)}")
            self.logger.info("请运行: pip install -r requirements.txt")
            return False
        
        self.logger.info("依赖项检查完成")
        return True
    
    def initialize_components(self):
        """初始化组件"""
        self.logger.info("初始化系统组件...")
        
        try:
            # 初始化优化器
            self.logger.info("初始化数据基础设施优化器...")
            self.optimizer = ContinuousDataInfrastructureOptimizer(self.config_path)
            
            # 初始化监控器
            self.logger.info("初始化数据基础设施监控器...")
            monitor_config = {
                'base_path': 'data/enhanced_market_data',
                'monitoring_interval_seconds': 300
            }
            self.monitor = DataInfrastructureMonitor(monitor_config)
            
            self.logger.info("组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"组件初始化失败: {e}")
            raise
    
    def validate_configuration(self) -> bool:
        """验证配置"""
        self.logger.info("验证系统配置...")
        
        try:
            # 验证优化器配置
            if not self.optimizer:
                return False
            
            # 检查交易所连接
            if not self.optimizer.exchanges:
                self.logger.error("未配置交易所连接")
                return False
            
            # 验证symbol配置
            if not self.optimizer.top30_symbols:
                self.logger.error("未配置交易对列表")
                return False
            
            # 验证时间框架配置
            if not self.optimizer.timeframes:
                self.logger.error("未配置时间框架")
                return False
            
            self.logger.info("配置验证完成")
            return True
            
        except Exception as e:
            self.logger.error(f"配置验证失败: {e}")
            return False
    
    async def perform_initial_setup(self):
        """执行初始设置"""
        self.logger.info("执行初始系统设置...")
        
        try:
            # 1. 执行初始数据收集
            self.logger.info("步骤 1/3: 初始数据收集...")
            await self.optimizer.initial_data_collection()
            
            # 2. 启动监控系统
            self.logger.info("步骤 2/3: 启动监控系统...")
            self.monitor.start_monitoring()
            
            # 等待监控系统启动
            await asyncio.sleep(5)
            
            # 3. 生成初始报告
            self.logger.info("步骤 3/3: 生成初始报告...")
            self.generate_setup_report()
            
            self.setup_complete = True
            self.logger.info("初始设置完成")
            
        except Exception as e:
            self.logger.error(f"初始设置失败: {e}")
            raise
    
    def generate_setup_report(self):
        """生成设置报告"""
        report = {
            'setup_timestamp': datetime.now(timezone.utc).isoformat(),
            'system_status': 'initialized',
            'optimizer_status': 'ready',
            'monitor_status': 'active' if self.monitor.monitoring_active else 'inactive',
            'configuration': {
                'symbols_configured': len(self.optimizer.top30_symbols),
                'timeframes_configured': len(self.optimizer.timeframes),
                'exchanges_configured': list(self.optimizer.exchanges.keys())
            },
            'data_status': self._get_initial_data_status(),
            'next_steps': [
                "系统已准备就绪，开始持续优化",
                "监控系统已启动，将自动检测数据质量问题",
                "可使用 run_continuous_data_optimization.py --status 查看状态"
            ]
        }
        
        # 保存报告
        report_path = Path("data/continuous_optimization_setup_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"设置报告已保存: {report_path}")
        return report
    
    def _get_initial_data_status(self) -> Dict[str, Any]:
        """获取初始数据状态"""
        data_dir = Path("data/enhanced_market_data")
        
        if not data_dir.exists():
            return {'status': 'no_data', 'files': 0}
        
        parquet_files = list(data_dir.glob("*.parquet"))
        
        # 按交易对统计
        symbol_files = {}
        for symbol in self.optimizer.top30_symbols:
            symbol_files[symbol] = len([f for f in parquet_files if f.name.startswith(symbol)])
        
        # 按时间框架统计
        timeframe_files = {}
        for tf in self.optimizer.timeframes:
            timeframe_files[tf] = len([f for f in parquet_files if f"_{tf}_" in f.name])
        
        total_size_mb = sum(f.stat().st_size for f in parquet_files) / 1024 / 1024
        
        return {
            'status': 'data_available',
            'total_files': len(parquet_files),
            'total_size_mb': round(total_size_mb, 2),
            'by_symbol': symbol_files,
            'by_timeframe': timeframe_files,
            'coverage_percentage': (len(parquet_files) / 
                                   (len(self.optimizer.top30_symbols) * len(self.optimizer.timeframes)) * 100)
        }
    
    async def start_continuous_operations(self):
        """启动持续运营"""
        if not self.setup_complete:
            self.logger.error("系统未完成初始设置")
            return
        
        self.logger.info("启动持续优化运营...")
        self.running = True
        
        try:
            # 启动优化器
            await self.optimizer.start_continuous_optimization()
            
        except Exception as e:
            self.logger.error(f"持续运营启动失败: {e}")
            self.running = False
            raise
    
    def stop_operations(self):
        """停止运营"""
        self.logger.info("停止持续优化运营...")
        
        if self.optimizer:
            self.optimizer.stop_optimization()
        
        if self.monitor:
            self.monitor.stop_monitoring()
        
        self.running = False
        self.logger.info("运营已停止")
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        status = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'setup_complete': self.setup_complete,
            'running': self.running,
            'components': {
                'optimizer': 'initialized' if self.optimizer else 'not_initialized',
                'monitor': 'active' if (self.monitor and self.monitor.monitoring_active) else 'inactive'
            }
        }
        
        if self.optimizer:
            optimizer_status = self.optimizer.get_optimization_status()
            status['optimizer_details'] = optimizer_status
        
        if self.monitor:
            monitor_status = self.monitor.get_monitoring_summary()
            status['monitor_details'] = monitor_status
        
        return status
    
    def print_welcome_message(self):
        """打印欢迎信息"""
        print("\n" + "="*80)
        print("🚀 DipMaster Trading System")
        print("📊 Continuous Data Infrastructure Optimization")
        print("="*80)
        print("\n📋 系统功能:")
        print("  • TOP30币种数据自动收集和管理")
        print("  • 6个时间框架数据支持 (1m, 5m, 15m, 1h, 4h, 1d)")
        print("  • 实时数据质量监控和自动修复")
        print("  • 数据缺口检测和填补")
        print("  • 增量数据更新机制")
        print("  • 高性能Parquet存储格式")
        print("  • 自动化监控报告和告警")
        
        print("\n⚙️  配置信息:")
        if self.optimizer:
            print(f"  • 监控币种: {len(self.optimizer.top30_symbols)}")
            print(f"  • 时间框架: {len(self.optimizer.timeframes)}")
            print(f"  • 配置交易所: {list(self.optimizer.exchanges.keys())}")
        
        print("\n🔧 使用方法:")
        print("  python run_continuous_data_optimization.py --start    # 启动服务")
        print("  python run_continuous_data_optimization.py --status   # 查看状态") 
        print("  python run_continuous_data_optimization.py --report   # 生成报告")
        print("  python run_continuous_data_optimization.py --stop     # 停止服务")
        
        print("\n📁 重要文件:")
        print("  • logs/continuous_data_optimizer.log     # 优化器日志")
        print("  • data/enhanced_market_data/             # 数据存储目录")
        print("  • data/monitoring.db                     # 监控数据库")
        print("  • data/*_report.json                     # 系统报告")
        
        print("\n" + "="*80 + "\n")

async def main():
    """主函数"""
    setup_manager = ContinuousOptimizationSetup()
    
    try:
        # 打印欢迎信息
        setup_manager.print_welcome_message()
        
        # 1. 检查依赖项
        if not setup_manager.check_dependencies():
            print("❌ 依赖项检查失败")
            return 1
        print("✅ 依赖项检查通过")
        
        # 2. 初始化组件
        setup_manager.initialize_components()
        print("✅ 组件初始化完成")
        
        # 3. 验证配置
        if not setup_manager.validate_configuration():
            print("❌ 配置验证失败")
            return 1
        print("✅ 配置验证通过")
        
        # 4. 执行初始设置
        print("\n🔄 执行初始设置 (这可能需要几分钟时间)...")
        await setup_manager.perform_initial_setup()
        print("✅ 初始设置完成")
        
        # 5. 生成状态报告
        status = setup_manager.get_system_status()
        print(f"\n📊 系统状态: {status['components']}")
        
        # 6. 提示用户下一步操作
        print("\n🎉 系统设置成功!")
        print("\n🚀 现在可以启动持续优化服务:")
        print("   python run_continuous_data_optimization.py --start")
        
        print("\n📈 或查看当前状态:")
        print("   python run_continuous_data_optimization.py --status")
        
        print("\n📊 生成详细报告:")
        print("   python run_continuous_data_optimization.py --report")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  设置被用户中断")
        setup_manager.stop_operations()
        return 1
        
    except Exception as e:
        print(f"\n❌ 设置过程中发生错误: {e}")
        setup_manager.logger.error(f"Setup failed: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)