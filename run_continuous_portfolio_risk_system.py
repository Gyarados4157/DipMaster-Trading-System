#!/usr/bin/env python3
"""
DipMaster持续组合优化和风险控制系统执行器
Continuous Portfolio Optimization and Risk Control System Executor

整合执行：
1. 持续信号处理和组合优化
2. 实时风险监控和控制
3. Kelly优化的动态权重调整
4. Beta中性和波动率控制
5. 综合风险报告和可视化
6. 自动化告警和风险缓解

作者: DipMaster Trading System  
版本: V1.0.0 - Integrated Portfolio Risk Management
"""

import asyncio
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import signal
import sys

# 导入自定义组件
from src.core.continuous_portfolio_risk_manager import (
    ContinuousPortfolioRiskManager, 
    ContinuousRiskConfig
)
from src.monitoring.real_time_risk_monitor import (
    RealTimeRiskMonitor, 
    RiskThresholds
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_risk_management.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ContinuousPortfolioRiskSystem')

class IntegratedRiskManagementSystem:
    """集成风险管理系统"""
    
    def __init__(self):
        self.is_running = False
        self.shutdown_requested = False
        
        # 配置持续风险管理
        self.continuous_config = ContinuousRiskConfig(
            base_capital=100000,
            rebalance_frequency="hourly",
            min_signal_confidence=0.60,
            min_expected_return=0.005,
            max_portfolio_beta=0.10,
            max_portfolio_volatility=0.18,
            max_single_position=0.20,
            max_total_leverage=3.0,
            max_var_95=0.03,
            max_es_95=0.04,
            max_drawdown=0.03,
            kelly_fraction=0.25,
            max_correlation_threshold=0.70,
            min_diversification_ratio=1.20
        )
        
        # 配置实时监控阈值
        self.monitoring_thresholds = RiskThresholds(
            var_95_daily=0.03,
            var_99_daily=0.04,
            es_95_daily=0.04,
            portfolio_vol_annual=0.18,
            position_vol_annual=0.25,
            portfolio_beta=0.10,
            max_correlation=0.70,
            min_diversification=1.20,
            max_position_size=0.20,
            liquidity_score_min=0.60
        )
        
        # 初始化组件
        self.continuous_manager = ContinuousPortfolioRiskManager(self.continuous_config)
        self.risk_monitor = RealTimeRiskMonitor(self.monitoring_thresholds)
        
        # 系统状态
        self.system_stats = {
            'start_time': None,
            'total_optimizations': 0,
            'total_alerts': 0,
            'last_optimization_time': None,
            'last_risk_check_time': None,
            'system_health': 'UNKNOWN'
        }
        
        # 创建输出目录
        self.output_dir = Path("results/continuous_risk_management")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Integrated Risk Management System initialized")

    def setup_signal_handlers(self):
        """设置信号处理器以优雅关闭"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def perform_integrated_risk_assessment(self) -> Dict:
        """执行集成风险评估"""
        try:
            # 获取当前组合
            portfolio_summary = self.continuous_manager.get_current_portfolio_summary()
            
            if portfolio_summary.get('status') == 'NO_POSITIONS':
                logger.info("No positions to assess")
                return {'status': 'NO_POSITIONS'}
            
            # 构建仓位字典用于风险监控
            positions = {}
            for pos_detail in portfolio_summary.get('positions_details', []):
                positions[pos_detail['symbol']] = pos_detail['weight']
            
            if not positions:
                return {'status': 'NO_POSITIONS'}
            
            # 生成综合风险报告
            risk_report = self.risk_monitor.generate_comprehensive_risk_report(positions)
            
            # 检查风险违规
            violations = risk_report.get('limit_violations', [])
            high_priority_violations = [v for v in violations if v.get('severity') == 'HIGH']
            
            # 更新系统统计
            self.system_stats['last_risk_check_time'] = datetime.now()
            self.system_stats['total_alerts'] += len(violations)
            
            # 评估系统健康状态
            if high_priority_violations:
                self.system_stats['system_health'] = 'CRITICAL'
            elif violations:
                self.system_stats['system_health'] = 'WARNING'
            else:
                self.system_stats['system_health'] = 'HEALTHY'
            
            # 保存风险评估结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            risk_file = self.output_dir / f"integrated_risk_assessment_{timestamp}.json"
            
            with open(risk_file, 'w') as f:
                json.dump(risk_report, f, indent=2, default=str)
            
            # 创建风险可视化
            viz_file = self.output_dir / f"risk_dashboard_{timestamp}.html"
            self.risk_monitor.create_risk_visualization(risk_report, str(viz_file))
            
            logger.info(f"Risk assessment completed. Violations: {len(violations)}, Health: {self.system_stats['system_health']}")
            
            return {
                'status': 'COMPLETED',
                'risk_report': risk_report,
                'violations_count': len(violations),
                'high_priority_violations': len(high_priority_violations),
                'system_health': self.system_stats['system_health'],
                'report_file': str(risk_file),
                'dashboard_file': str(viz_file)
            }
            
        except Exception as e:
            logger.error(f"Error in integrated risk assessment: {e}")
            import traceback
            traceback.print_exc()
            return {'status': 'ERROR', 'error': str(e)}

    async def handle_risk_violations(self, assessment_result: Dict):
        """处理风险违规"""
        if assessment_result.get('status') != 'COMPLETED':
            return
            
        high_priority_violations = assessment_result.get('high_priority_violations', 0)
        
        if high_priority_violations > 0:
            logger.warning(f"Detected {high_priority_violations} high priority risk violations")
            
            # 风险缓解措施
            risk_report = assessment_result.get('risk_report', {})
            recommendations = risk_report.get('recommendations', [])
            
            for rec in recommendations:
                if rec.get('priority') == 'HIGH':
                    logger.warning(f"HIGH PRIORITY: {rec.get('type')} - {rec.get('description')}")
                    
                    # 在这里可以实现自动风险缓解措施
                    # 例如：减少仓位、调整权重、发送告警等
                    
            # 发送风险告警（模拟）
            await self.send_risk_alert(assessment_result)

    async def send_risk_alert(self, assessment_result: Dict):
        """发送风险告警"""
        try:
            alert_data = {
                'timestamp': datetime.now().isoformat(),
                'alert_type': 'RISK_VIOLATION',
                'severity': 'HIGH' if assessment_result.get('high_priority_violations', 0) > 0 else 'MEDIUM',
                'system_health': assessment_result.get('system_health'),
                'violations_count': assessment_result.get('violations_count', 0),
                'dashboard_url': assessment_result.get('dashboard_file'),
                'message': f"Portfolio risk assessment detected {assessment_result.get('violations_count', 0)} violations"
            }
            
            # 保存告警记录
            alert_file = self.output_dir / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(alert_file, 'w') as f:
                json.dump(alert_data, f, indent=2)
            
            logger.warning(f"Risk alert sent: {alert_data['message']}")
            
            # 在这里可以集成实际的告警系统：
            # - 发送邮件
            # - 发送短信
            # - 推送到Slack/Discord
            # - 写入Kafka消息队列
            
        except Exception as e:
            logger.error(f"Error sending risk alert: {e}")

    async def optimize_and_assess_cycle(self):
        """优化和评估周期"""
        try:
            logger.info("Starting optimization and assessment cycle...")
            
            # 1. 执行组合优化
            await self.continuous_manager.continuous_optimization_cycle()
            self.system_stats['total_optimizations'] += 1
            self.system_stats['last_optimization_time'] = datetime.now()
            
            # 2. 执行风险评估
            assessment_result = await self.perform_integrated_risk_assessment()
            
            # 3. 处理风险违规
            await self.handle_risk_violations(assessment_result)
            
            # 4. 生成系统状态报告
            await self.generate_system_status_report()
            
            logger.info("Optimization and assessment cycle completed successfully")
            
        except Exception as e:
            logger.error(f"Error in optimization and assessment cycle: {e}")
            import traceback
            traceback.print_exc()

    async def generate_system_status_report(self):
        """生成系统状态报告"""
        try:
            # 获取当前组合摘要
            portfolio_summary = self.continuous_manager.get_current_portfolio_summary()
            
            # 系统运行时间
            uptime = None
            if self.system_stats['start_time']:
                uptime = (datetime.now() - self.system_stats['start_time']).total_seconds()
            
            status_report = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'report_type': 'SYSTEM_STATUS',
                    'system_version': 'IntegratedRiskManagement_V1.0.0'
                },
                'system_statistics': {
                    'start_time': self.system_stats['start_time'].isoformat() if self.system_stats['start_time'] else None,
                    'uptime_seconds': uptime,
                    'total_optimizations': self.system_stats['total_optimizations'],
                    'total_alerts': self.system_stats['total_alerts'],
                    'last_optimization': self.system_stats['last_optimization_time'].isoformat() if self.system_stats['last_optimization_time'] else None,
                    'last_risk_check': self.system_stats['last_risk_check_time'].isoformat() if self.system_stats['last_risk_check_time'] else None,
                    'system_health': self.system_stats['system_health']
                },
                'current_portfolio': portfolio_summary,
                'configuration': {
                    'continuous_config': {
                        'base_capital': self.continuous_config.base_capital,
                        'rebalance_frequency': self.continuous_config.rebalance_frequency,
                        'max_portfolio_beta': self.continuous_config.max_portfolio_beta,
                        'max_portfolio_volatility': self.continuous_config.max_portfolio_volatility,
                        'max_total_leverage': self.continuous_config.max_total_leverage,
                        'kelly_fraction': self.continuous_config.kelly_fraction
                    },
                    'risk_thresholds': {
                        'var_95_daily': self.monitoring_thresholds.var_95_daily,
                        'portfolio_vol_annual': self.monitoring_thresholds.portfolio_vol_annual,
                        'max_correlation': self.monitoring_thresholds.max_correlation,
                        'max_position_size': self.monitoring_thresholds.max_position_size
                    }
                }
            }
            
            # 保存状态报告
            status_file = self.output_dir / "system_status_latest.json"
            with open(status_file, 'w') as f:
                json.dump(status_report, f, indent=2, default=str)
            
            # 也保存一个带时间戳的副本
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status_history_file = self.output_dir / f"system_status_{timestamp}.json"
            with open(status_history_file, 'w') as f:
                json.dump(status_report, f, indent=2, default=str)
            
            logger.info(f"System status report generated: {status_file}")
            
        except Exception as e:
            logger.error(f"Error generating system status report: {e}")

    async def run_continuous_system(self, max_cycles: Optional[int] = None):
        """运行持续系统"""
        logger.info("Starting Continuous Portfolio Risk Management System...")
        
        self.is_running = True
        self.system_stats['start_time'] = datetime.now()
        
        try:
            cycle_count = 0
            
            # 初始化运行
            await self.optimize_and_assess_cycle()
            cycle_count += 1
            
            while self.is_running and not self.shutdown_requested:
                if max_cycles and cycle_count >= max_cycles:
                    logger.info(f"Reached maximum cycles ({max_cycles}), stopping...")
                    break
                
                # 根据配置的频率等待
                if self.continuous_config.rebalance_frequency == "hourly":
                    wait_time = 3600  # 1小时
                elif self.continuous_config.rebalance_frequency == "daily":
                    wait_time = 86400  # 1天
                else:
                    wait_time = 1800  # 默认30分钟
                
                # 可中断的等待
                for _ in range(wait_time):
                    if self.shutdown_requested:
                        break
                    await asyncio.sleep(1)
                
                if not self.shutdown_requested:
                    await self.optimize_and_assess_cycle()
                    cycle_count += 1
                    
        except Exception as e:
            logger.error(f"Error in continuous system execution: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            logger.info("Continuous Portfolio Risk Management System stopped")

    def display_system_summary(self):
        """显示系统摘要"""
        print("\n" + "="*70)
        print("🚀 DipMaster持续组合优化和风险控制系统")
        print("="*70)
        
        print(f"\n📊 系统配置:")
        print(f"   基础资金: ${self.continuous_config.base_capital:,.2f}")
        print(f"   再平衡频率: {self.continuous_config.rebalance_frequency}")
        print(f"   最大组合Beta: {self.continuous_config.max_portfolio_beta}")
        print(f"   最大组合波动率: {self.continuous_config.max_portfolio_volatility:.1%}")
        print(f"   最大单仓位: {self.continuous_config.max_single_position:.1%}")
        print(f"   最大总杠杆: {self.continuous_config.max_total_leverage}x")
        print(f"   Kelly比例: {self.continuous_config.kelly_fraction}")
        
        print(f"\n🎯 风险控制目标:")
        print(f"   日度VaR(95%): {self.monitoring_thresholds.var_95_daily:.1%}")
        print(f"   年化波动率: {self.monitoring_thresholds.portfolio_vol_annual:.1%}")
        print(f"   最大回撤: {self.continuous_config.max_drawdown:.1%}")
        print(f"   最大相关性: {self.monitoring_thresholds.max_correlation:.2f}")
        print(f"   最小分散化比率: {self.continuous_config.min_diversification_ratio:.2f}")
        
        print(f"\n💾 输出目录: {self.output_dir}")
        print(f"\n✅ 系统已就绪，准备启动持续优化...")

async def main():
    """主执行函数"""
    # 创建集成系统
    system = IntegratedRiskManagementSystem()
    
    # 设置信号处理
    system.setup_signal_handlers()
    
    # 显示系统摘要
    system.display_system_summary()
    
    # 自动执行单次优化和风险评估演示
    print(f"\n🎯 执行单次优化和风险评估演示...")
    
    try:
        await system.optimize_and_assess_cycle()
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 用户中断，正在停止系统...")
        system.shutdown_requested = True
    except Exception as e:
        logger.error(f"执行错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 显示最终统计
    print(f"\n📈 系统执行统计:")
    print(f"   总优化次数: {system.system_stats['total_optimizations']}")
    print(f"   总告警数量: {system.system_stats['total_alerts']}")
    print(f"   系统健康状态: {system.system_stats['system_health']}")
    print(f"   输出目录: {system.output_dir}")
    
    if system.system_stats['start_time']:
        uptime = datetime.now() - system.system_stats['start_time']
        print(f"   运行时间: {uptime}")
    
    print(f"\n✅ DipMaster持续组合风险管理系统执行完成！")

if __name__ == "__main__":
    asyncio.run(main())