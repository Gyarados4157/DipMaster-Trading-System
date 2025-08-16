#!/usr/bin/env python3
"""
DipMaster Enhanced V4 Monitoring System Demo
演示DipMaster增强版V4的完整监控和日志收集系统

本演示展示：
1. Kafka事件流生产和消费
2. 一致性监控（信号-持仓-执行）
3. 风险监控和告警
4. 策略漂移检测
5. 性能监控和分析
6. 系统健康监控
7. 结构化日志管理
8. 集成监控系统

运行方式：
python run_monitoring_demo.py
"""

import asyncio
import time
import json
import random
from datetime import datetime, timezone
from pathlib import Path
import uuid

# 导入监控系统组件
from src.monitoring.integrated_monitoring_system import create_integrated_monitoring_system
from src.monitoring.structured_logger import LogContext, LogCategory


async def main():
    """主演示函数"""
    print("🎯 DipMaster Enhanced V4 监控系统演示")
    print("=" * 60)
    
    # 创建监控系统
    config_file = "config/monitoring_config.json"
    monitoring_system = create_integrated_monitoring_system(config_file)
    
    try:
        # 启动监控系统
        print("🚀 启动集成监控系统...")
        await monitoring_system.start()
        
        # 等待系统完全启动
        await asyncio.sleep(3)
        
        # 演示各种监控功能
        await demo_trading_events(monitoring_system)
        await demo_risk_monitoring(monitoring_system)
        await demo_performance_tracking(monitoring_system)
        await demo_consistency_monitoring(monitoring_system)
        await demo_system_health(monitoring_system)
        await demo_structured_logging(monitoring_system)
        
        # 显示综合状态
        await show_comprehensive_status(monitoring_system)
        
        # 生成监控报告
        await generate_monitoring_reports(monitoring_system)
        
        print("\n✅ 监控系统演示完成！")
        print("📊 查看生成的日志和报告文件了解详细信息")
        
        # 保持系统运行一段时间进行观察
        print("\n🔄 系统将继续运行60秒进行实时监控...")
        await asyncio.sleep(60)
        
    except KeyboardInterrupt:
        print("\n⏹️  用户中断演示")
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
    finally:
        # 停止监控系统
        print("\n🛑 停止监控系统...")
        await monitoring_system.stop()


async def demo_trading_events(monitoring_system):
    """演示交易事件监控"""
    print("\n📈 演示交易事件监控")
    print("-" * 40)
    
    # 模拟交易信号
    signals = [
        {
            'signal_id': f'SIG_{uuid.uuid4().hex[:8]}',
            'timestamp': time.time(),
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'confidence': 0.85,
            'entry_price': 62500.0,
            'strategy_version': '4.0',
            'parameters': {'rsi': 35, 'volume_ratio': 1.8},
            'features': {'rsi': 35, 'volume_ratio': 1.8, 'price_momentum': -0.02}
        },
        {
            'signal_id': f'SIG_{uuid.uuid4().hex[:8]}',
            'timestamp': time.time() + 1,
            'symbol': 'ETHUSDT',
            'side': 'BUY',
            'confidence': 0.78,
            'entry_price': 3150.0,
            'strategy_version': '4.0',
            'parameters': {'rsi': 42, 'volume_ratio': 1.5},
            'features': {'rsi': 42, 'volume_ratio': 1.5, 'price_momentum': -0.015}
        }
    ]
    
    for signal in signals:
        monitoring_system.record_signal(signal)
        print(f"  📡 记录信号: {signal['symbol']} {signal['side']} (置信度: {signal['confidence']:.1%})")
        await asyncio.sleep(0.5)
    
    # 模拟持仓更新
    positions = [
        {
            'position_id': f'POS_{uuid.uuid4().hex[:8]}',
            'signal_id': signals[0]['signal_id'],
            'timestamp': time.time() + 2,
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'entry_price': 62500.0,
            'quantity': 0.015,
            'status': 'opened',
            'current_price': 62520.0,
            'unrealized_pnl': 30.0
        },
        {
            'position_id': f'POS_{uuid.uuid4().hex[:8]}',
            'signal_id': signals[1]['signal_id'],
            'timestamp': time.time() + 3,
            'symbol': 'ETHUSDT',
            'side': 'BUY',
            'entry_price': 3150.0,
            'quantity': 0.5,
            'status': 'opened',
            'current_price': 3155.0,
            'unrealized_pnl': 25.0
        }
    ]
    
    for position in positions:
        monitoring_system.record_position_update(position)
        print(f"  📍 记录持仓: {position['symbol']} {position['quantity']} (未实现盈亏: ${position['unrealized_pnl']:.2f})")
        await asyncio.sleep(0.5)
    
    # 模拟订单执行
    executions = [
        {
            'execution_id': f'EXEC_{uuid.uuid4().hex[:8]}',
            'signal_id': signals[0]['signal_id'],
            'position_id': positions[0]['position_id'],
            'timestamp': time.time() + 4,
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.015,
            'price': 62510.0,
            'slippage_bps': 1.6,
            'latency_ms': 45,
            'status': 'filled',
            'venue': 'binance',
            'fees': 0.935,
            'session_id': f'SESSION_{int(time.time())}'
        },
        {
            'execution_id': f'EXEC_{uuid.uuid4().hex[:8]}',
            'signal_id': signals[1]['signal_id'],
            'position_id': positions[1]['position_id'],
            'timestamp': time.time() + 5,
            'symbol': 'ETHUSDT',
            'side': 'BUY',
            'quantity': 0.5,
            'price': 3152.0,
            'slippage_bps': 0.63,
            'latency_ms': 38,
            'status': 'filled',
            'venue': 'binance',
            'fees': 1.576,
            'session_id': f'SESSION_{int(time.time())}'
        }
    ]
    
    for execution in executions:
        monitoring_system.record_execution(execution)
        print(f"  ⚡ 记录执行: {execution['symbol']} (滑点: {execution['slippage_bps']:.1f}bps, 延迟: {execution['latency_ms']}ms)")
        await asyncio.sleep(0.5)
    
    print("  ✅ 交易事件演示完成")


async def demo_risk_monitoring(monitoring_system):
    """演示风险监控"""
    print("\n🛡️ 演示风险监控")
    print("-" * 40)
    
    if monitoring_system.risk_monitor:
        # 模拟市场数据更新
        market_data = [
            {'symbol': 'BTCUSDT', 'price': 62520.0, 'volume': 1000000, 'bid': 62515.0, 'ask': 62525.0, 'volatility': 0.025},
            {'symbol': 'ETHUSDT', 'price': 3155.0, 'volume': 800000, 'bid': 3154.0, 'ask': 3156.0, 'volatility': 0.030},
            {'symbol': 'SOLUSDT', 'price': 145.0, 'volume': 500000, 'bid': 144.8, 'ask': 145.2, 'volatility': 0.035}
        ]
        
        for data in market_data:
            monitoring_system.risk_monitor.update_market_data(data['symbol'], data)
            print(f"  📊 更新市场数据: {data['symbol']} ${data['price']:.2f}")
        
        # 计算风险指标
        risk_metrics = monitoring_system.risk_monitor.calculate_risk_metrics()
        print(f"  📈 总敞口: ${risk_metrics.total_exposure:.2f}")
        print(f"  📉 当前回撤: {risk_metrics.current_drawdown:.1%}")
        print(f"  ⚖️ 杠杆率: {risk_metrics.leverage:.2f}x")
        
        # 检查风险限制
        risk_alerts = monitoring_system.risk_monitor.check_risk_limits()
        if risk_alerts:
            print(f"  ⚠️ 生成了 {len(risk_alerts)} 个风险告警")
        else:
            print("  ✅ 所有风险指标正常")
    
    print("  ✅ 风险监控演示完成")


async def demo_performance_tracking(monitoring_system):
    """演示性能监控"""
    print("\n📊 演示性能监控")
    print("-" * 40)
    
    if monitoring_system.performance_monitor:
        # 模拟已完成的交易
        completed_trades = [
            {
                'trade_id': f'TRADE_{uuid.uuid4().hex[:8]}',
                'timestamp': time.time() - 3600,
                'symbol': 'BTCUSDT',
                'side': 'buy',
                'entry_price': 62000.0,
                'exit_price': 62500.0,
                'quantity': 0.01,
                'entry_time': time.time() - 3600,
                'exit_time': time.time() - 3300,
                'duration_seconds': 300,
                'fees': 0.62,
                'slippage_bps': 1.2,
                'strategy': 'dipmaster_v4',
                'confidence': 0.82
            },
            {
                'trade_id': f'TRADE_{uuid.uuid4().hex[:8]}',
                'timestamp': time.time() - 2400,
                'symbol': 'ETHUSDT',
                'side': 'buy',
                'entry_price': 3100.0,
                'exit_price': 3130.0,
                'quantity': 0.3,
                'entry_time': time.time() - 2400,
                'exit_time': time.time() - 2100,
                'duration_seconds': 300,
                'fees': 0.93,
                'slippage_bps': 0.8,
                'strategy': 'dipmaster_v4',
                'confidence': 0.76
            },
            {
                'trade_id': f'TRADE_{uuid.uuid4().hex[:8]}',
                'timestamp': time.time() - 1800,
                'symbol': 'SOLUSDT',
                'side': 'buy',
                'entry_price': 144.0,
                'exit_price': 143.5,
                'quantity': 5.0,
                'entry_time': time.time() - 1800,
                'exit_time': time.time() - 1500,
                'duration_seconds': 300,
                'fees': 0.72,
                'slippage_bps': 2.1,
                'strategy': 'dipmaster_v4',
                'confidence': 0.68
            }
        ]
        
        for trade in completed_trades:
            monitoring_system.record_trade(trade)
            return_pct = ((trade['exit_price'] - trade['entry_price']) / trade['entry_price']) * 100
            print(f"  💹 记录交易: {trade['symbol']} 收益率: {return_pct:.2f}%")
        
        # 获取性能摘要
        performance = monitoring_system.performance_monitor.get_performance_summary()
        current_metrics = performance.get('current_metrics', {})
        
        print(f"  🎯 胜率: {current_metrics.get('win_rate', 0):.1%}")
        print(f"  📈 夏普比率: {current_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  💰 总收益: ${current_metrics.get('total_return_usd', 0):.2f}")
        print(f"  📉 最大回撤: {current_metrics.get('max_drawdown', 0):.1%}")
    
    print("  ✅ 性能监控演示完成")


async def demo_consistency_monitoring(monitoring_system):
    """演示一致性监控"""
    print("\n🔄 演示一致性监控")
    print("-" * 40)
    
    if monitoring_system.consistency_monitor:
        # 计算一致性指标
        consistency_metrics = monitoring_system.consistency_monitor.calculate_consistency_metrics()
        
        print(f"  🔗 信号-持仓一致性: {consistency_metrics.signal_position_consistency:.1%}")
        print(f"  ⚡ 持仓-执行一致性: {consistency_metrics.position_execution_consistency:.1%}")
        print(f"  ⏱️ 时间一致性: {consistency_metrics.timing_consistency:.1%}")
        print(f"  💵 价格一致性: {consistency_metrics.price_consistency:.1%}")
        print(f"  📊 整体一致性: {consistency_metrics.overall_consistency:.1%}")
        
        # 执行对账
        reconciliation = monitoring_system.consistency_monitor.perform_reconciliation()
        print(f"  🔍 对账状态: {reconciliation.get('status', 'unknown')}")
        if reconciliation.get('issues'):
            print(f"  ⚠️ 发现 {len(reconciliation['issues'])} 个问题")
        else:
            print("  ✅ 对账无问题")
    
    print("  ✅ 一致性监控演示完成")


async def demo_system_health(monitoring_system):
    """演示系统健康监控"""
    print("\n🏥 演示系统健康监控")
    print("-" * 40)
    
    if monitoring_system.system_health_monitor:
        # 获取健康摘要
        health_summary = monitoring_system.system_health_monitor.get_health_summary()
        
        print(f"  🎯 整体状态: {health_summary.get('overall_status', 'unknown')}")
        print(f"  📊 健康评分: {health_summary.get('overall_score', 0):.1f}/100")
        
        component_summary = health_summary.get('component_summary', {})
        print(f"  🔧 组件总数: {component_summary.get('total_components', 0)}")
        
        status_breakdown = component_summary.get('status_breakdown', {})
        for status, count in status_breakdown.items():
            print(f"    - {status}: {count}")
        
        # 系统资源状态
        resources = health_summary.get('system_resources', {})
        print(f"  💻 CPU使用率: {resources.get('cpu_usage_percent', 0):.1f}%")
        print(f"  🧠 内存使用率: {resources.get('memory_usage_percent', 0):.1f}%")
        print(f"  💾 磁盘使用率: {resources.get('disk_usage_percent', 0):.1f}%")
    
    print("  ✅ 系统健康监控演示完成")


async def demo_structured_logging(monitoring_system):
    """演示结构化日志"""
    print("\n📝 演示结构化日志管理")
    print("-" * 40)
    
    if monitoring_system.logger:
        # 创建日志上下文
        context = monitoring_system.logger.create_context(
            component="demo",
            session_id=f"DEMO_{int(time.time())}"
        )
        
        # 演示不同类型的日志
        monitoring_system.logger.log_info(
            "演示系统启动",
            context,
            category=LogCategory.SYSTEM,
            data={'demo_type': 'comprehensive', 'version': '4.0'}
        )
        
        monitoring_system.logger.log_trading(
            "模拟交易执行",
            context,
            symbol="BTCUSDT",
            trade_id="DEMO_TRADE_001",
            strategy="dipmaster_v4",
            data={'action': 'buy', 'price': 62500.0}
        )
        
        # 性能计时演示
        with monitoring_system.logger.timer("demo_operation", context):
            await asyncio.sleep(0.1)  # 模拟操作
        
        monitoring_system.logger.log_audit(
            "监控系统演示事件",
            context,
            data={
                'event_type': 'demo_completed',
                'components_tested': 8,
                'success': True
            }
        )
        
        # 获取日志统计
        log_stats = monitoring_system.logger.get_log_statistics()
        agg_stats = log_stats.get('aggregation', {})
        
        print(f"  📄 日志条目总数: {agg_stats.get('total_entries', 0)}")
        print(f"  📈 近1小时条目: {agg_stats.get('recent_entries_1h', 0)}")
        print(f"  ❌ 错误率: {agg_stats.get('error_rate_percent', 0):.1f}%")
        
        level_dist = agg_stats.get('level_distribution', {})
        for level, count in level_dist.items():
            print(f"    - {level}: {count}")
    
    print("  ✅ 结构化日志演示完成")


async def show_comprehensive_status(monitoring_system):
    """显示综合状态"""
    print("\n📋 综合监控状态")
    print("=" * 60)
    
    status = monitoring_system.get_comprehensive_status()
    
    # 监控系统状态
    monitoring_status = status.get('monitoring_status', {})
    print(f"整体健康状态: {monitoring_status.get('overall_health', 'unknown')}")
    print(f"运行时间: {status.get('uptime_seconds', 0):.0f} 秒")
    print(f"运行中的组件: {monitoring_status.get('components_running', 0)}/{monitoring_status.get('components_total', 0)}")
    print(f"活跃告警: {monitoring_status.get('active_alerts', 0)} (严重: {monitoring_status.get('critical_alerts', 0)})")
    print(f"事件处理成功率: {100 - monitoring_status.get('error_rate_percent', 0):.1f}%")
    
    # 组件状态
    components = status.get('components', {})
    print(f"\n📊 组件详细状态:")
    
    for component_name, component_status in components.items():
        if isinstance(component_status, dict):
            if 'error' in component_status:
                print(f"  ❌ {component_name}: 错误 - {component_status['error']}")
            elif 'overall_health' in component_status:
                print(f"  ✅ {component_name}: {component_status['overall_health']}")
            elif 'status' in component_status:
                print(f"  ℹ️  {component_name}: {component_status['status']}")
            else:
                print(f"  📊 {component_name}: 运行中")


async def generate_monitoring_reports(monitoring_system):
    """生成监控报告"""
    print("\n📄 生成监控报告")
    print("-" * 40)
    
    try:
        # 创建报告目录
        reports_dir = Path("reports/monitoring")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 综合状态报告
        status_report = monitoring_system.get_comprehensive_status()
        with open(reports_dir / f"comprehensive_status_{timestamp}.json", 'w', encoding='utf-8') as f:
            json.dump(status_report, f, indent=2, ensure_ascii=False, default=str)
        print(f"  📊 综合状态报告: comprehensive_status_{timestamp}.json")
        
        # 2. 风险报告
        if monitoring_system.risk_monitor:
            risk_report = monitoring_system.risk_monitor.export_risk_report()
            with open(reports_dir / f"risk_report_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(risk_report, f, indent=2, ensure_ascii=False, default=str)
            print(f"  🛡️ 风险报告: risk_report_{timestamp}.json")
        
        # 3. 性能报告
        if monitoring_system.performance_monitor:
            perf_report = monitoring_system.performance_monitor.export_performance_report()
            with open(reports_dir / f"performance_report_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(perf_report, f, indent=2, ensure_ascii=False, default=str)
            print(f"  📈 性能报告: performance_report_{timestamp}.json")
        
        # 4. 漂移检测报告
        if monitoring_system.drift_detector:
            drift_report = monitoring_system.drift_detector.export_drift_report()
            with open(reports_dir / f"drift_report_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(drift_report, f, indent=2, ensure_ascii=False, default=str)
            print(f"  📈 漂移检测报告: drift_report_{timestamp}.json")
        
        # 5. 系统健康报告
        if monitoring_system.system_health_monitor:
            health_report = monitoring_system.system_health_monitor.export_health_report()
            with open(reports_dir / f"health_report_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(health_report, f, indent=2, ensure_ascii=False, default=str)
            print(f"  🏥 健康报告: health_report_{timestamp}.json")
        
        # 6. Kafka统计报告
        kafka_stats = {}
        if monitoring_system.kafka_producer:
            kafka_stats['producer'] = monitoring_system.kafka_producer.get_producer_stats()
        if monitoring_system.kafka_consumer:
            kafka_stats['consumer'] = monitoring_system.kafka_consumer.get_consumer_stats()
        
        if kafka_stats:
            with open(reports_dir / f"kafka_stats_{timestamp}.json", 'w', encoding='utf-8') as f:
                json.dump(kafka_stats, f, indent=2, ensure_ascii=False, default=str)
            print(f"  📤 Kafka统计: kafka_stats_{timestamp}.json")
        
        print(f"  📁 所有报告已保存到: {reports_dir}")
        
    except Exception as e:
        print(f"  ❌ 生成报告时发生错误: {e}")


if __name__ == "__main__":
    """运行监控系统演示"""
    print("🎯 DipMaster Enhanced V4 - 监控系统演示启动")
    print("=" * 60)
    print("本演示将展示以下功能：")
    print("• Kafka事件流生产和消费")
    print("• 信号-持仓-执行一致性监控")
    print("• 实时风险监控和告警")
    print("• 策略漂移检测")
    print("• 性能监控和分析")
    print("• 系统健康监控")
    print("• 结构化日志管理")
    print("• 综合监控报告生成")
    print("=" * 60)
    
    try:
        # 运行异步主函数
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 DipMaster Enhanced V4 监控系统演示结束")
    print("📁 查看 logs/ 和 reports/ 目录了解详细信息")