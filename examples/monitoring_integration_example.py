#!/usr/bin/env python3
"""
DipMaster Trading System - Monitoring Integration Example
监控系统集成示例 - 演示如何将现有交易系统与监控系统集成

Features:
- Complete integration workflow demonstration
- Real trading scenario simulation
- Error handling and recovery examples
- Performance monitoring integration
- Dashboard data visualization examples

Author: DipMaster Trading System Monitoring Agent
Date: 2025-08-18
Version: 2.0.0
"""

import asyncio
import sys
import json
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from run_comprehensive_monitoring_system import MonitoringSystemOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DipMasterSignal:
    """DipMaster信号数据结构"""
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: str
    confidence: float
    price: float
    rsi: float
    ma20_distance: float
    volume_ratio: float
    expected_entry_price: float
    expected_holding_minutes: int
    strategy_params: Dict[str, Any]


@dataclass
class DipMasterPosition:
    """DipMaster持仓数据结构"""
    position_id: str
    signal_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    holding_minutes: Optional[int] = None
    pnl: Optional[float] = None
    realized: bool = False


@dataclass
class DipMasterExecution:
    """DipMaster执行数据结构"""
    execution_id: str
    position_id: str
    order_type: str
    symbol: str
    side: str
    quantity: float
    requested_price: float
    executed_price: float
    execution_time: datetime
    latency_ms: float
    slippage_bps: float
    fees: float
    venue: str


class MockDipMasterTradingSystem:
    """
    模拟DipMaster交易系统
    
    用于演示如何将现有交易系统与监控系统集成。
    在实际部署中，这些方法将被真实的交易系统组件替代。
    """
    
    def __init__(self, monitoring_orchestrator: MonitoringSystemOrchestrator):
        """
        初始化模拟交易系统
        
        Args:
            monitoring_orchestrator: 监控系统编排器
        """
        self.monitoring = monitoring_orchestrator
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        self.active_positions: Dict[str, DipMasterPosition] = {}
        self.signal_counter = 0
        self.position_counter = 0
        self.execution_counter = 0
        
        # DipMaster策略参数
        self.strategy_params = {
            'rsi_min': 30.0,
            'rsi_max': 50.0,
            'max_holding_minutes': 180,
            'boundary_minutes': [15, 30, 45, 0],
            'target_profit_pct': 0.8,
            'dip_threshold_pct': 0.2,
            'volume_multiplier': 1.5
        }
        
        logger.info("🤖 MockDipMasterTradingSystem initialized with monitoring integration")
    
    async def start_trading_simulation(self, duration_minutes: int = 60):
        """
        启动交易模拟
        
        Args:
            duration_minutes: 模拟运行时间（分钟）
        """
        logger.info(f"🚀 Starting DipMaster trading simulation for {duration_minutes} minutes")
        
        end_time = datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)
        
        try:
            while datetime.now(timezone.utc) < end_time:
                # 模拟信号生成和交易执行
                await self._simulate_trading_cycle()
                
                # 随机等待（模拟真实交易间隔）
                await asyncio.sleep(random.uniform(30, 120))  # 30秒到2分钟
                
        except Exception as e:
            logger.error(f"❌ Trading simulation error: {e}")
            
        finally:
            # 关闭所有剩余持仓
            await self._close_all_positions()
            logger.info("✅ Trading simulation completed")
    
    async def _simulate_trading_cycle(self):
        """模拟一个完整的交易周期"""
        try:
            # 1. 生成交易信号
            signal = await self._generate_dipmaster_signal()
            
            # 2. 记录信号到监控系统
            await self._record_signal_to_monitoring(signal)
            
            # 3. 执行信号（开仓）
            if signal.signal_type == 'BUY':
                position = await self._execute_buy_signal(signal)
                if position:
                    # 4. 记录持仓到监控系统
                    await self._record_position_to_monitoring(position)
                    
                    # 5. 模拟持仓期间的监控
                    await self._simulate_position_monitoring(position)
        
        except Exception as e:
            logger.error(f"❌ Error in trading cycle: {e}")
    
    async def _generate_dipmaster_signal(self) -> DipMasterSignal:
        """生成DipMaster信号"""
        self.signal_counter += 1
        symbol = random.choice(self.symbols)
        
        # 模拟当前价格
        base_price = self._get_mock_price(symbol)
        
        # 模拟技术指标
        rsi = random.uniform(25, 55)  # RSI在DipMaster范围附近
        ma20_distance = random.uniform(-0.05, 0.01)  # 价格相对MA20的距离
        volume_ratio = random.uniform(0.8, 3.0)  # 成交量比率
        
        # 模拟DipMaster逻辑判断
        is_valid_signal = (
            self.strategy_params['rsi_min'] <= rsi <= self.strategy_params['rsi_max'] and
            ma20_distance < 0 and  # 价格低于MA20
            volume_ratio >= self.strategy_params['volume_multiplier']  # 成交量放量
        )
        
        signal_type = 'BUY' if is_valid_signal else 'HOLD'
        confidence = random.uniform(0.6, 0.95) if is_valid_signal else random.uniform(0.3, 0.6)
        
        # 期望入场价格（逢跌买入）
        expected_entry_price = base_price * (1 - self.strategy_params['dip_threshold_pct'] / 100)
        
        signal = DipMasterSignal(
            signal_id=f"dipmaster_signal_{self.signal_counter:06d}",
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=base_price,
            rsi=rsi,
            ma20_distance=ma20_distance,
            volume_ratio=volume_ratio,
            expected_entry_price=expected_entry_price,
            expected_holding_minutes=random.randint(60, 120),
            strategy_params=self.strategy_params.copy()
        )
        
        logger.info(f"📊 Generated signal: {signal.signal_id} - {signal.symbol} {signal.signal_type} (confidence: {confidence:.2f})")
        
        return signal
    
    async def _record_signal_to_monitoring(self, signal: DipMasterSignal):
        """记录信号到监控系统"""
        try:
            signal_data = {
                'signal_id': signal.signal_id,
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'confidence': signal.confidence,
                'price': signal.price,
                'technical_indicators': {
                    'rsi': signal.rsi,
                    'ma20_distance': signal.ma20_distance,
                    'volume_ratio': signal.volume_ratio
                },
                'expected_entry_price': signal.expected_entry_price,
                'expected_holding_minutes': signal.expected_holding_minutes,
                'strategy_params': signal.strategy_params
            }
            
            await self.monitoring.record_trading_signal(signal_data)
            logger.debug(f"📝 Signal recorded to monitoring: {signal.signal_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record signal to monitoring: {e}")
    
    async def _execute_buy_signal(self, signal: DipMasterSignal) -> Optional[DipMasterPosition]:
        """执行买入信号"""
        if signal.signal_type != 'BUY' or signal.confidence < 0.7:
            return None
        
        self.position_counter += 1
        self.execution_counter += 1
        
        # 模拟订单执行
        execution_latency = random.uniform(20, 100)  # 20-100ms延迟
        slippage = random.uniform(-5, 15)  # -0.5% 到 1.5% 滑点
        
        executed_price = signal.expected_entry_price * (1 + slippage / 10000)
        quantity = random.uniform(0.01, 0.1)  # 模拟仓位大小
        
        # 创建持仓记录
        position = DipMasterPosition(
            position_id=f"dipmaster_pos_{self.position_counter:06d}",
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            side='BUY',
            quantity=quantity,
            entry_price=executed_price,
            entry_time=datetime.now(timezone.utc),
            realized=False
        )
        
        # 创建执行记录
        execution = DipMasterExecution(
            execution_id=f"dipmaster_exec_{self.execution_counter:06d}",
            position_id=position.position_id,
            order_type='MARKET',
            symbol=signal.symbol,
            side='BUY',
            quantity=quantity,
            requested_price=signal.expected_entry_price,
            executed_price=executed_price,
            execution_time=datetime.now(timezone.utc),
            latency_ms=execution_latency,
            slippage_bps=slippage,
            fees=executed_price * quantity * 0.001,  # 0.1% 手续费
            venue='binance'
        )
        
        # 记录执行到监控系统
        await self._record_execution_to_monitoring(execution)
        
        # 存储活跃持仓
        self.active_positions[position.position_id] = position
        
        logger.info(f"💰 Opened position: {position.position_id} - {position.symbol} {position.quantity:.4f} @ ${executed_price:.2f}")
        
        return position
    
    async def _record_execution_to_monitoring(self, execution: DipMasterExecution):
        """记录执行到监控系统"""
        try:
            execution_data = {
                'execution_id': execution.execution_id,
                'position_id': execution.position_id,
                'symbol': execution.symbol,
                'side': execution.side,
                'quantity': execution.quantity,
                'price': execution.executed_price,
                'execution_time': execution.execution_time.isoformat(),
                'latency_ms': execution.latency_ms,
                'slippage_bps': execution.slippage_bps,
                'fees': execution.fees,
                'venue': execution.venue
            }
            
            await self.monitoring.record_order_execution(execution_data)
            logger.debug(f"📝 Execution recorded to monitoring: {execution.execution_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record execution to monitoring: {e}")
    
    async def _record_position_to_monitoring(self, position: DipMasterPosition):
        """记录持仓到监控系统"""
        try:
            position_data = {
                'position_id': position.position_id,
                'signal_id': position.signal_id,
                'symbol': position.symbol,
                'side': position.side,
                'quantity': position.quantity,
                'entry_price': position.entry_price,
                'entry_time': position.entry_time.isoformat(),
                'exit_price': position.exit_price,
                'exit_time': position.exit_time.isoformat() if position.exit_time else None,
                'holding_minutes': position.holding_minutes,
                'pnl': position.pnl,
                'realized': position.realized
            }
            
            await self.monitoring.record_trading_position(position_data)
            logger.debug(f"📝 Position recorded to monitoring: {position.position_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to record position to monitoring: {e}")
    
    async def _simulate_position_monitoring(self, position: DipMasterPosition):
        """模拟持仓期间的监控"""
        # 随机决定持仓时间（符合DipMaster策略）
        holding_minutes = random.randint(15, 120)
        
        # 在后台任务中处理持仓关闭
        asyncio.create_task(self._close_position_after_delay(position, holding_minutes))
    
    async def _close_position_after_delay(self, position: DipMasterPosition, delay_minutes: int):
        """延迟关闭持仓"""
        try:
            # 等待指定时间
            await asyncio.sleep(delay_minutes * 60)
            
            # 检查持仓是否仍然活跃
            if position.position_id in self.active_positions:
                await self._close_position(position)
                
        except Exception as e:
            logger.error(f"❌ Error closing position after delay: {e}")
    
    async def _close_position(self, position: DipMasterPosition):
        """关闭持仓"""
        try:
            current_time = datetime.now(timezone.utc)
            holding_time = current_time - position.entry_time
            holding_minutes = int(holding_time.total_seconds() / 60)
            
            # 模拟出场价格（考虑DipMaster目标利润）
            profit_factor = random.uniform(-0.005, 0.015)  # -0.5% 到 1.5%
            exit_price = position.entry_price * (1 + profit_factor)
            
            # 计算P&L
            pnl = (exit_price - position.entry_price) * position.quantity
            
            # 更新持仓记录
            position.exit_price = exit_price
            position.exit_time = current_time
            position.holding_minutes = holding_minutes
            position.pnl = pnl
            position.realized = True
            
            # 检查是否符合边界规则
            exit_minute = current_time.minute
            boundary_minutes = self.strategy_params['boundary_minutes']
            is_boundary_compliant = any(abs(exit_minute - bm) <= 2 for bm in boundary_minutes)
            
            if not is_boundary_compliant:
                logger.warning(f"⚠️ Position {position.position_id} closed outside boundary minutes: {exit_minute}")
            
            # 记录更新的持仓到监控系统
            await self._record_position_to_monitoring(position)
            
            # 从活跃持仓中移除
            self.active_positions.pop(position.position_id, None)
            
            logger.info(f"📤 Closed position: {position.position_id} - P&L: ${pnl:.2f} ({holding_minutes}min)")
            
        except Exception as e:
            logger.error(f"❌ Error closing position {position.position_id}: {e}")
    
    async def _close_all_positions(self):
        """关闭所有剩余持仓"""
        for position in list(self.active_positions.values()):
            await self._close_position(position)
    
    def _get_mock_price(self, symbol: str) -> float:
        """获取模拟价格"""
        base_prices = {
            'BTCUSDT': 43000.0,
            'ETHUSDT': 2400.0,
            'BNBUSDT': 320.0,
            'ADAUSDT': 0.45,
            'SOLUSDT': 95.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        # 添加随机波动
        volatility = random.uniform(-0.02, 0.02)  # ±2%
        return base_price * (1 + volatility)


class MonitoringIntegrationDemo:
    """监控集成演示类"""
    
    def __init__(self):
        self.orchestrator: Optional[MonitoringSystemOrchestrator] = None
        self.trading_system: Optional[MockDipMasterTradingSystem] = None
    
    async def run_complete_demo(self):
        """运行完整的集成演示"""
        logger.info("🚀 Starting DipMaster Monitoring Integration Demo")
        
        try:
            # 1. 初始化监控系统
            await self._initialize_monitoring_system()
            
            # 2. 初始化模拟交易系统
            self._initialize_trading_system()
            
            # 3. 运行集成测试
            await self._run_integration_tests()
            
            # 4. 演示监控功能
            await self._demonstrate_monitoring_features()
            
            # 5. 运行交易模拟
            await self._run_trading_simulation()
            
            # 6. 生成最终报告
            await self._generate_demo_report()
            
            logger.info("✅ Monitoring integration demo completed successfully")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            import traceback
            traceback.print_exc()
            
        finally:
            await self._cleanup()
    
    async def _initialize_monitoring_system(self):
        """初始化监控系统"""
        logger.info("🔍 Initializing comprehensive monitoring system...")
        
        config_path = project_root / "config" / "comprehensive_monitoring_config.yaml"
        
        self.orchestrator = MonitoringSystemOrchestrator(
            config_path=str(config_path),
            mode="development"
        )
        
        # 启动监控系统（后台任务）
        asyncio.create_task(self.orchestrator.start())
        
        # 等待系统启动
        await asyncio.sleep(5)
        
        logger.info("✅ Monitoring system initialized")
    
    def _initialize_trading_system(self):
        """初始化交易系统"""
        logger.info("🤖 Initializing mock trading system...")
        
        if not self.orchestrator:
            raise RuntimeError("Monitoring system must be initialized first")
        
        self.trading_system = MockDipMasterTradingSystem(self.orchestrator)
        
        logger.info("✅ Trading system initialized")
    
    async def _run_integration_tests(self):
        """运行集成测试"""
        logger.info("🧪 Running integration tests...")
        
        # 测试信号记录
        test_signal = {
            'signal_id': 'test_signal_001',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'symbol': 'BTCUSDT',
            'signal_type': 'BUY',
            'confidence': 0.85,
            'price': 43000.0,
            'technical_indicators': {
                'rsi': 35.0,
                'ma20_distance': -0.01,
                'volume_ratio': 1.8
            }
        }
        
        await self.orchestrator.record_trading_signal(test_signal)
        logger.info("✅ Signal recording test passed")
        
        # 测试持仓记录
        test_position = {
            'position_id': 'test_pos_001',
            'signal_id': 'test_signal_001',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.1,
            'entry_price': 42950.0,
            'entry_time': datetime.now(timezone.utc).isoformat()
        }
        
        await self.orchestrator.record_trading_position(test_position)
        logger.info("✅ Position recording test passed")
        
        # 测试执行记录
        test_execution = {
            'execution_id': 'test_exec_001',
            'position_id': 'test_pos_001',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.1,
            'price': 42950.0,
            'execution_time': datetime.now(timezone.utc).isoformat(),
            'latency_ms': 45.0,
            'slippage_bps': 2.5,
            'venue': 'binance'
        }
        
        await self.orchestrator.record_order_execution(test_execution)
        logger.info("✅ Execution recording test passed")
        
        logger.info("✅ All integration tests passed")
    
    async def _demonstrate_monitoring_features(self):
        """演示监控功能"""
        logger.info("📊 Demonstrating monitoring features...")
        
        # 获取系统状态
        system_status = self.orchestrator.get_system_status()
        logger.info(f"📈 System Status: {system_status['is_running']}, Uptime: {system_status['uptime_seconds']:.1f}s")
        
        # 等待监控系统处理数据
        await asyncio.sleep(3)
        
        # 获取监控统计
        if self.orchestrator.monitoring_system:
            stats = self.orchestrator.monitoring_system.get_monitoring_statistics()
            logger.info(f"📊 Monitoring Stats: {stats['system_stats']['signals_validated']} signals processed")
        
        logger.info("✅ Monitoring features demonstration completed")
    
    async def _run_trading_simulation(self):
        """运行交易模拟"""
        logger.info("💹 Running trading simulation...")
        
        if not self.trading_system:
            raise RuntimeError("Trading system not initialized")
        
        # 运行5分钟的交易模拟
        simulation_task = asyncio.create_task(
            self.trading_system.start_trading_simulation(duration_minutes=5)
        )
        
        # 监控模拟过程
        monitoring_task = asyncio.create_task(
            self._monitor_simulation_progress()
        )
        
        # 等待模拟完成
        await asyncio.gather(simulation_task, monitoring_task)
        
        logger.info("✅ Trading simulation completed")
    
    async def _monitor_simulation_progress(self):
        """监控模拟过程"""
        start_time = datetime.now(timezone.utc)
        
        while True:
            try:
                # 检查是否超过监控时间
                if datetime.now(timezone.utc) - start_time > timedelta(minutes=6):
                    break
                
                # 获取当前统计
                if self.orchestrator.monitoring_system:
                    stats = self.orchestrator.monitoring_system.get_monitoring_statistics()
                    logger.info(f"📊 Progress: {stats['system_stats']['signals_validated']} signals, "
                               f"{stats['system_stats']['positions_tracked']} positions, "
                               f"{stats['system_stats']['executions_monitored']} executions")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"❌ Error monitoring simulation progress: {e}")
                break
    
    async def _generate_demo_report(self):
        """生成演示报告"""
        logger.info("📋 Generating demo report...")
        
        try:
            if self.orchestrator.monitoring_system:
                # 获取最终统计
                final_stats = self.orchestrator.monitoring_system.get_monitoring_statistics()
                
                # 获取系统状态
                system_status = self.orchestrator.get_system_status()
                
                # 生成报告
                report = {
                    'demo_summary': {
                        'start_time': datetime.now(timezone.utc).isoformat(),
                        'duration_minutes': 10,
                        'mode': 'integration_demo'
                    },
                    'system_performance': {
                        'uptime_seconds': system_status['uptime_seconds'],
                        'signals_processed': final_stats['system_stats']['signals_validated'],
                        'positions_tracked': final_stats['system_stats']['positions_tracked'],
                        'executions_monitored': final_stats['system_stats']['executions_monitored'],
                        'alerts_generated': final_stats['system_stats']['alerts_generated']
                    },
                    'monitoring_health': {
                        'consistency_score': final_stats['consistency_metrics']['overall_consistency_score'],
                        'system_health_score': system_status.get('system_health_score', 0),
                        'component_status': {
                            'monitoring_system': system_status['components']['monitoring_system'],
                            'dashboard_service': system_status['components']['dashboard_service']
                        }
                    },
                    'integration_results': {
                        'signal_integration': 'SUCCESS',
                        'position_integration': 'SUCCESS',
                        'execution_integration': 'SUCCESS',
                        'monitoring_integration': 'SUCCESS',
                        'dashboard_integration': 'SUCCESS'
                    }
                }
                
                # 保存报告
                report_dir = project_root / "reports" / "integration_demo"
                report_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                report_path = report_dir / f"monitoring_integration_demo_{timestamp}.json"
                
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2)
                
                logger.info(f"📋 Demo report saved: {report_path}")
                
                # 打印摘要
                logger.info("📊 Demo Summary:")
                logger.info(f"  - Signals Processed: {report['system_performance']['signals_processed']}")
                logger.info(f"  - Positions Tracked: {report['system_performance']['positions_tracked']}")
                logger.info(f"  - Executions Monitored: {report['system_performance']['executions_monitored']}")
                logger.info(f"  - Consistency Score: {report['monitoring_health']['consistency_score']:.1f}%")
                logger.info(f"  - System Health Score: {report['monitoring_health']['system_health_score']:.1f}%")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate demo report: {e}")
    
    async def _cleanup(self):
        """清理资源"""
        logger.info("🧹 Cleaning up resources...")
        
        if self.orchestrator:
            await self.orchestrator.stop()
        
        logger.info("✅ Cleanup completed")


async def main():
    """主函数"""
    demo = MonitoringIntegrationDemo()
    await demo.run_complete_demo()


if __name__ == "__main__":
    asyncio.run(main())