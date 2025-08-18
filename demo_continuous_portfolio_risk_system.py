#!/usr/bin/env python3
"""
DipMaster持续组合优化和风险控制系统演示
Demonstration of Continuous Portfolio Risk Management System

快速演示核心功能：
1. 信号处理和Kelly优化
2. 组合构建和Beta中性
3. 实时风险计算
4. 压力测试分析
5. 风险监控报告

作者: DipMaster Trading System
版本: V1.0.0 - Demo Version
"""

import asyncio
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import time

# 导入系统组件
from src.core.continuous_portfolio_risk_manager import (
    ContinuousPortfolioRiskManager, 
    ContinuousRiskConfig,
    PortfolioPosition
)
from src.monitoring.real_time_risk_monitor import (
    RealTimeRiskMonitor, 
    RiskThresholds
)

def print_section_header(title):
    """打印节标题"""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print(f"{'='*60}")

def print_subsection(title):
    """打印子节标题"""  
    print(f"\n📊 {title}")
    print(f"{'-'*50}")

async def demo_signal_processing():
    """演示信号处理和Kelly优化"""
    print_section_header("信号处理和Kelly优化演示")
    
    # 创建演示配置
    config = ContinuousRiskConfig(
        base_capital=100000,
        kelly_fraction=0.25,
        min_signal_confidence=0.60,
        min_expected_return=0.005
    )
    
    manager = ContinuousPortfolioRiskManager(config)
    
    # 模拟信号数据
    demo_signals = pd.DataFrame({
        'timestamp': [datetime.now()] * 5,
        'symbol': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'],
        'signal': [0.75, 0.68, 0.72, 0.65, 0.80],
        'confidence': [0.75, 0.68, 0.72, 0.65, 0.80],
        'predicted_return': [0.008, 0.007, 0.009, 0.006, 0.010]
    })
    
    print_subsection("输入信号数据")
    print(demo_signals.to_string(index=False))
    
    # Kelly权重计算
    kelly_weights = manager.calculate_kelly_optimal_weights(demo_signals)
    
    print_subsection("Kelly优化权重")
    for symbol, weight in kelly_weights.items():
        print(f"   {symbol}: {weight:.4f} ({weight*100:.2f}%)")
    
    # 相关性调整
    correlation_adjustments = manager.check_correlation_constraints(demo_signals)
    
    print_subsection("相关性调整")
    if correlation_adjustments:
        for symbol, adjustment in correlation_adjustments.items():
            print(f"   {symbol}: 调整系数 {adjustment:.3f}")
    else:
        print("   无需相关性调整")
    
    return demo_signals, kelly_weights

async def demo_portfolio_optimization(signals_df):
    """演示组合优化"""
    print_section_header("组合优化和Beta中性演示")
    
    config = ContinuousRiskConfig(
        base_capital=100000,
        max_portfolio_beta=0.10,
        max_portfolio_volatility=0.18,
        max_single_position=0.20,
        max_total_leverage=3.0
    )
    
    manager = ContinuousPortfolioRiskManager(config)
    
    # 执行组合优化
    positions, optimization_info = await manager.optimize_portfolio(signals_df)
    
    print_subsection("优化后的组合仓位")
    for symbol, position in positions.items():
        print(f"   {symbol}:")
        print(f"     权重: {position.weight:.4f} ({position.weight*100:.2f}%)")
        print(f"     金额: ${position.dollar_amount:,.2f}")
        print(f"     置信度: {position.confidence:.3f}")
        print(f"     预期收益: {position.expected_return:.4f}")
    
    print_subsection("组合优化信息")
    print(f"   总仓位数: {optimization_info['total_positions']}")
    print(f"   总敞口: {optimization_info['gross_exposure']:.4f}")
    print(f"   净敞口: {optimization_info['net_exposure']:.4f}")
    print(f"   Kelly总权重: {optimization_info['kelly_total']:.4f}")
    
    return positions

async def demo_risk_monitoring(positions):
    """演示实时风险监控"""
    print_section_header("实时风险监控演示")
    
    thresholds = RiskThresholds(
        var_95_daily=0.03,
        portfolio_vol_annual=0.18,
        max_correlation=0.70
    )
    
    monitor = RealTimeRiskMonitor(thresholds)
    
    # 转换仓位格式
    position_weights = {symbol: pos.weight for symbol, pos in positions.items()}
    
    print_subsection("VaR和ES计算")
    var_95_param, es_95_param = monitor.calculate_portfolio_var_es(position_weights, 0.95, 'parametric')
    var_95_hist, es_95_hist = monitor.calculate_portfolio_var_es(position_weights, 0.95, 'historical')
    
    print(f"   参数法VaR (95%): {var_95_param:.4f} ({var_95_param*100:.2f}%)")
    print(f"   参数法ES (95%): {es_95_param:.4f} ({es_95_param*100:.2f}%)")
    print(f"   历史模拟VaR (95%): {var_95_hist:.4f} ({var_95_hist*100:.2f}%)")
    print(f"   历史模拟ES (95%): {es_95_hist:.4f} ({es_95_hist*100:.2f}%)")
    
    print_subsection("相关性分析")
    symbols = list(position_weights.keys())
    correlation_matrix = monitor.calculate_correlation_matrix(symbols)
    
    print(f"   资产数量: {len(symbols)}")
    if len(symbols) > 1:
        avg_correlation = np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        max_correlation = np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])
        print(f"   平均相关性: {avg_correlation:.3f}")
        print(f"   最大相关性: {max_correlation:.3f}")
    else:
        print(f"   单一资产，无相关性计算")
    
    return monitor

async def demo_stress_testing(positions, monitor):
    """演示压力测试"""
    print_section_header("压力测试演示")
    
    position_weights = {symbol: pos.weight for symbol, pos in positions.items()}
    stress_results = monitor.perform_stress_testing(position_weights)
    
    print_subsection("压力测试结果")
    for scenario_name, result in stress_results.items():
        print(f"   {result['description']}:")
        print(f"     预估损失: ${result['estimated_loss']*100000:,.2f} ({result['loss_percentage']*100:.2f}%)")
        print(f"     压力VaR: {result['stressed_var_95']:.4f}")
        print(f"     风险等级: {result['risk_level']}")
        print()

async def demo_risk_attribution(positions, monitor):
    """演示风险归因分析"""
    print_section_header("风险归因分析演示")
    
    position_weights = {symbol: pos.weight for symbol, pos in positions.items()}
    attribution = monitor.calculate_risk_attribution(position_weights)
    
    print_subsection("边际风险贡献 (MCR)")
    for symbol, mcr in attribution['marginal_contribution'].items():
        print(f"   {symbol}: {mcr:.6f}")
    
    print_subsection("成分风险贡献 (CCR)")  
    for symbol, ccr in attribution['component_contribution'].items():
        print(f"   {symbol}: {ccr:.6f}")
    
    print_subsection("风险贡献百分比")
    for symbol, risk_pct in attribution['risk_percentage'].items():
        print(f"   {symbol}: {risk_pct*100:.2f}%")
    
    print_subsection("组合风险指标")
    print(f"   组合波动率: {attribution['portfolio_volatility']:.6f}")
    print(f"   多样化比率: {attribution['diversification_ratio']:.3f}")

async def demo_liquidity_assessment(positions, monitor):
    """演示流动性风险评估"""
    print_section_header("流动性风险评估演示")
    
    position_weights = {symbol: pos.weight for symbol, pos in positions.items()}
    liquidity_assessment = monitor.assess_liquidity_risk(position_weights)
    
    print_subsection("个别资产流动性评估")
    for symbol, assessment in liquidity_assessment['positions_detail'].items():
        print(f"   {symbol}:")
        print(f"     流动性评分: {assessment['adjusted_liquidity_score']:.3f}")
        print(f"     清算天数: {assessment['days_to_liquidate']:.2f}")
        print(f"     流动性风险: {assessment['liquidity_risk_level']}")
    
    print_subsection("组合级流动性指标")
    print(f"   平均流动性评分: {liquidity_assessment['average_liquidity_score']:.3f}")
    print(f"   最大清算天数: {liquidity_assessment['max_liquidation_days']:.2f}")
    print(f"   组合流动性风险: {liquidity_assessment['portfolio_liquidity_risk']}")

async def demo_comprehensive_report(positions, monitor):
    """演示综合风险报告"""
    print_section_header("综合风险报告演示")
    
    position_weights = {symbol: pos.weight for symbol, pos in positions.items()}
    
    # 生成综合报告
    comprehensive_report = monitor.generate_comprehensive_risk_report(position_weights)
    
    print_subsection("执行摘要")
    executive_summary = comprehensive_report['risk_levels']
    for metric, level in executive_summary.items():
        status_emoji = "✅" if level == "LOW" else "⚠️" if level == "MEDIUM" else "🔴"
        print(f"   {metric}: {level} {status_emoji}")
    
    print_subsection("风险限制检查")
    violations = comprehensive_report.get('limit_violations', [])
    if violations:
        print(f"   发现 {len(violations)} 项违规:")
        for violation in violations:
            print(f"     {violation['type']}: {violation['current']:.4f} > {violation['threshold']:.4f}")
    else:
        print("   ✅ 所有风险限制均在正常范围内")
    
    print_subsection("风险管理建议")
    recommendations = comprehensive_report.get('recommendations', [])
    if recommendations:
        for i, rec in enumerate(recommendations[:3], 1):
            priority_emoji = "🔴" if rec['priority'] == 'HIGH' else "🟡" if rec['priority'] == 'MEDIUM' else "🟢"
            print(f"   {i}. [{rec['priority']}] {priority_emoji} {rec['description']}")
    else:
        print("   ✅ 当前无特殊风险管理建议")
    
    # 保存演示报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    demo_report_file = f"results/continuous_risk_management/demo_report_{timestamp}.json"
    
    with open(demo_report_file, 'w') as f:
        json.dump(comprehensive_report, f, indent=2, default=str)
    
    print_subsection("报告保存")
    print(f"   📄 详细报告已保存至: {demo_report_file}")
    
    # 创建可视化
    viz_file = f"results/continuous_risk_management/demo_dashboard_{timestamp}.html"
    monitor.create_risk_visualization(comprehensive_report, viz_file)
    print(f"   📊 风险仪表板: {viz_file}")

def display_demo_summary():
    """显示演示总结"""
    print_section_header("DipMaster持续组合优化和风险控制系统演示总结")
    
    print(f"\n🎯 演示完成的核心功能:")
    print(f"   ✅ 1. 信号处理和Kelly优化权重计算")
    print(f"   ✅ 2. 相关性约束和组合优化") 
    print(f"   ✅ 3. Beta中性组合构建")
    print(f"   ✅ 4. 多方法VaR和ES计算")
    print(f"   ✅ 5. 相关性矩阵分析")
    print(f"   ✅ 6. 5种情景压力测试")
    print(f"   ✅ 7. 风险归因分析 (MCR/CCR)")
    print(f"   ✅ 8. 流动性风险评估")
    print(f"   ✅ 9. 综合风险报告生成")
    print(f"   ✅ 10. 交互式风险仪表板")
    
    print(f"\n📊 系统特点:")
    print(f"   🚀 异步高性能处理")
    print(f"   🎯 专业级风险管理")
    print(f"   📈 实时监控和告警")
    print(f"   🔧 高度可配置化")
    print(f"   📋 完整审计追踪")
    print(f"   🖥️  交互式可视化")
    
    print(f"\n💾 输出文件:")
    output_dir = Path("results/continuous_risk_management")
    if output_dir.exists():
        files = list(output_dir.glob("demo_*"))
        for file in sorted(files)[-2:]:  # 显示最新的2个文件
            print(f"   📄 {file.name}")
    
    print(f"\n🎉 演示完成！系统已准备好投入生产使用。")

async def main():
    """主演示函数"""
    print("🚀 DipMaster持续组合优化和风险控制系统 - 功能演示")
    print("🕐 开始时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # 创建必要目录
    Path("results/continuous_risk_management").mkdir(parents=True, exist_ok=True)
    
    try:
        # 演示1: 信号处理和Kelly优化
        signals_df, kelly_weights = await demo_signal_processing()
        time.sleep(1)
        
        # 演示2: 组合优化
        positions = await demo_portfolio_optimization(signals_df)
        time.sleep(1)
        
        # 演示3: 实时风险监控  
        monitor = await demo_risk_monitoring(positions)
        time.sleep(1)
        
        # 演示4: 压力测试
        await demo_stress_testing(positions, monitor)
        time.sleep(1)
        
        # 演示5: 风险归因分析
        await demo_risk_attribution(positions, monitor)
        time.sleep(1)
        
        # 演示6: 流动性风险评估
        await demo_liquidity_assessment(positions, monitor)
        time.sleep(1)
        
        # 演示7: 综合风险报告
        await demo_comprehensive_report(positions, monitor)
        
        # 显示演示总结
        display_demo_summary()
        
    except Exception as e:
        print(f"\n❌ 演示过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n🕐 结束时间:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    asyncio.run(main())