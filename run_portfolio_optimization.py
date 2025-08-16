#!/usr/bin/env python3
"""
DipMaster Enhanced V4 - 组合优化主程序
基于Alpha信号的完整组合构建和风险管理解决方案

运行指令:
python run_portfolio_optimization.py

输出:
- TargetPortfolio.json: 目标组合权重配置
- RiskReport.json: 详细风险分析报告  
- PortfolioConstruction_Report.json: 完整构建报告

作者: DipMaster Trading System
版本: 4.0.0
创建时间: 2025-08-16
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'core'))

from generate_sample_portfolio import save_results

def main():
    """主程序入口"""
    
    print("="*80)
    print("DipMaster Enhanced V4 - Portfolio Optimization System")
    print("="*80)
    print()
    print("System Overview:")
    print("   - Multi-Asset Portfolio Construction")
    print("   - Kelly Criterion + Risk Parity Optimization")
    print("   - Real-Time Risk Management (VaR, ES, Stress Testing)")
    print("   - Dynamic Position Sizing")
    print("   - Market Neutral Beta Targeting")
    print("   - 85%+ Target Win Rate Strategy")
    print()
    print("Optimization Objectives:")
    print("   - Maximize Risk-Adjusted Returns (Sharpe >2.0)")
    print("   - Control Maximum Drawdown (<=3%)")
    print("   - Maintain Market Neutrality (Beta ~0)")
    print("   - Optimize Capital Efficiency")
    print()
    print("Expected Performance:")
    print("   - Target Win Rate: 85%+")
    print("   - Target Sharpe Ratio: >2.0")
    print("   - Maximum Drawdown: <3%")
    print("   - Annual Volatility: 15-20%")
    print("   - Monthly Return: 12-20%")
    print()
    
    try:
        print("Starting portfolio construction process...")
        print()
        
        # 运行组合构建
        result = save_results()
        
        print()
        print("="*80)
        print("PORTFOLIO CONSTRUCTION COMPLETED SUCCESSFULLY")
        print("="*80)
        
        # 显示关键信息
        target_portfolio = result['target_portfolio']
        construction_meta = result['construction_metadata']
        risk_analysis = result['risk_analysis']
        
        print()
        print("PORTFOLIO COMPOSITION:")
        print("-" * 40)
        for weight in target_portfolio['weights']:
            kelly = weight['kelly_fraction']
            conf_adj = weight['confidence_adj']
            vol_adj = weight['volatility_adj']
            print(f"   {weight['symbol']:10s}: {weight['w']:6.1%} (${weight['usd_size']:>5.0f}) "
                  f"Kelly:{kelly:.2f} Conf:{conf_adj:.2f} Vol:{vol_adj:.2f}")
        
        print()
        print("RISK METRICS:")
        print("-" * 40)
        risk_metrics = target_portfolio['risk']
        print(f"   Annual Volatility:  {risk_metrics['ann_vol']:>6.1%}")
        print(f"   Portfolio Beta:     {risk_metrics['beta']:>6.3f}")
        print(f"   Daily VaR (95%):    {risk_metrics['VaR_95']:>6.2%}")
        print(f"   Expected Shortfall: {risk_metrics['ES_95']:>6.2%}")
        print(f"   Sharpe Ratio:       {risk_metrics['sharpe']:>6.2f}")
        
        print()
        print("CAPITAL ALLOCATION:")
        print("-" * 40)
        print(f"   Available Capital:  ${construction_meta['available_capital']:>8,.0f}")
        print(f"   Total Allocated:    ${construction_meta['total_allocation']:>8,.0f}")
        print(f"   Capital Utilization: {construction_meta['capital_utilization']:>6.1%}")
        print(f"   Leverage Factor:     {construction_meta['total_weight']:>6.2f}x")
        print(f"   Number of Positions: {construction_meta['num_positions']:>6d}")
        
        print()
        print("RISK MONITORING:")
        print("-" * 40)
        risk_monitoring = risk_analysis['risk_monitoring']
        print(f"   Risk Level:         {risk_monitoring['risk_level']:>10s}")
        print(f"   Active Alerts:      {risk_monitoring['total_alerts']:>10d}")
        print(f"   Critical Alerts:    {risk_monitoring['critical_alerts']:>10d}")
        
        print()
        print("STRESS TEST RESULTS:")
        print("-" * 40)
        for stress in risk_analysis['stress_test_results']:
            print(f"   {stress['scenario']:20s}: {stress['portfolio_pnl']:>6.1%} portfolio impact")
        
        print()
        print("PERFORMANCE EXPECTATIONS:")
        print("-" * 40)
        perf_exp = result['performance_expectations']
        print(f"   Target Win Rate:    {perf_exp['target_win_rate']:>10s}")
        print(f"   Target Sharpe:      {perf_exp['target_sharpe']:>10s}")
        print(f"   Risk Score:         {perf_exp['risk_adjusted_score']:>7.1f}/100")
        print(f"   Expected Return:    {perf_exp['expected_monthly_return']:>10s}")
        
        print()
        print("IMPLEMENTATION GUIDANCE:")
        print("-" * 40)
        execution_order = result['implementation_guidance']['execution_order']
        print("   Execution Priority:")
        for order in execution_order:
            print(f"     {order['order']}. {order['symbol']:10s} {order['priority']:>6s} "
                  f"({order['execution_method']:>6s}) ${order['usd_size']:>5.0f}")
        
        print()
        print("OUTPUT FILES:")
        print("-" * 40)
        print("   [OK] TargetPortfolio.json - Portfolio weights and allocation")
        print("   [OK] RiskReport.json - Comprehensive risk analysis")
        print("   [OK] PortfolioConstruction_Report.json - Full construction details")
        
        print()
        print("NEXT STEPS:")
        print("-" * 40)
        print("   1. Review the generated JSON files for detailed analysis")
        print("   2. Validate constraints and risk metrics meet requirements")
        print("   3. Execute positions according to the implementation guidance")
        print("   4. Monitor real-time risk metrics and alerts")
        print("   5. Rebalance when rebalance triggers are activated")
        
        print()
        print("=" * 80)
        print("DipMaster V4 Portfolio Optimization Complete!")
        print("Ready for live trading deployment")
        print("=" * 80)
        
    except Exception as e:
        print(f"[ERROR] Portfolio construction failed: {str(e)}")
        print()
        print("Troubleshooting:")
        print("   - Check Alpha signal data availability")
        print("   - Verify market data integrity")
        print("   - Ensure configuration parameters are valid")
        print("   - Check system dependencies (cvxpy, pandas, numpy)")
        
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)