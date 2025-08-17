"""
DipMaster Enhanced Portfolio Optimization Demo
演示智能组合风险优化系统

运行此脚本生成:
1. 完整的TargetPortfolio.json配置
2. 详细的风险报告
3. 压力测试结果
4. 实时监控面板数据

作者: DipMaster Trading System
版本: 5.0.0
创建时间: 2025-08-17
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.enhanced_portfolio_optimizer import (
    create_enhanced_portfolio_optimizer, 
    MarketRegime,
    PositionMetrics
)

def generate_sample_data():
    """生成示例数据用于演示"""
    
    # 35个币种的样本数据
    symbols = [
        # Tier 1: 超大市值
        "BTCUSDT", "ETHUSDT",
        # Tier 2: 大市值
        "SOLUSDT", "ADAUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", 
        "LINKUSDT", "LTCUSDT", "MATICUSDT",
        # Tier 3: 中市值  
        "UNIUSDT", "AAVEUSDT", "DOTUSDT", "AVAXUSDT", "TRXUSDT",
        "ATOMUSDT", "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
        "FILUSDT", "VETUSDT", "ICPUSDT", "MKRUSDT", "COMPUSDT",
        "QNTUSDT", "XLMUSDT", "ALGOUSDT", "IOTAUSDT", "SUIUSDT"
    ]
    
    # 生成时间序列（过去90天，5分钟数据）
    end_time = datetime.now()
    start_time = end_time - timedelta(days=90)
    timestamps = pd.date_range(start_time, end_time, freq='5T')
    
    # 生成Alpha信号数据
    alpha_signals = []
    for symbol in symbols:
        # 模拟不同质量的信号
        if symbol in ["BTCUSDT", "ETHUSDT"]:
            base_score = 0.8
            base_confidence = 0.9
            base_return = 0.012
        elif symbol in ["SOLUSDT", "ADAUSDT", "XRPUSDT", "BNBUSDT"]:
            base_score = 0.7
            base_confidence = 0.85
            base_return = 0.015
        else:
            base_score = 0.6
            base_confidence = 0.75
            base_return = 0.018
        
        # 添加一些随机变化
        score = base_score + np.random.normal(0, 0.1)
        confidence = np.clip(base_confidence + np.random.normal(0, 0.05), 0.5, 1.0)
        predicted_return = base_return + np.random.normal(0, 0.005)
        
        alpha_signals.append({
            'timestamp': end_time,
            'symbol': symbol,
            'score': score,
            'confidence': confidence,
            'predicted_return': predicted_return
        })
    
    alpha_signals_df = pd.DataFrame(alpha_signals)
    
    # 生成市场数据
    market_data = []
    
    for symbol in symbols:
        # 基础价格和波动率设置
        if symbol == "BTCUSDT":
            base_price = 65000
            base_vol = 0.15
        elif symbol == "ETHUSDT":
            base_price = 3500
            base_vol = 0.18
        elif symbol in ["SOLUSDT", "ADAUSDT", "XRPUSDT"]:
            base_price = np.random.uniform(50, 500)
            base_vol = 0.20
        else:
            base_price = np.random.uniform(5, 100)
            base_vol = 0.25
        
        # 生成价格时间序列（简化的几何布朗运动）
        dt = 1/288  # 5分钟间隔，日化
        drift = 0.10 / 252  # 年化10%漂移
        
        prices = [base_price]
        volumes = []
        
        for i in range(1, len(timestamps)):
            # 价格演化
            dW = np.random.normal(0, np.sqrt(dt))
            price_change = drift * dt + base_vol * dW
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 0.01))  # 避免负价格
            
            # 成交量（与价格变化相关）
            volume_base = 1000000  # 基础成交量
            volume_multiplier = 1 + abs(price_change) * 5  # 波动越大成交量越大
            volume = volume_base * volume_multiplier * np.random.uniform(0.5, 2.0)
            volumes.append(volume)
        
        # 添加第一个成交量
        volumes.insert(0, 1000000)
        
        # 计算收益率
        price_series = pd.Series(prices)
        returns = price_series.pct_change().fillna(0)
        
        for i, timestamp in enumerate(timestamps):
            market_data.append({
                'timestamp': timestamp,
                'symbol': symbol,
                'close': prices[i],
                'volume': volumes[i],
                'returns': returns.iloc[i] if i > 0 else 0,
                'volatility': base_vol
            })
    
    market_data_df = pd.DataFrame(market_data)
    
    return alpha_signals_df, market_data_df

def generate_target_portfolio_demo():
    """生成完整的TargetPortfolio演示"""
    
    print("🚀 启动DipMaster Enhanced Portfolio Optimization V5...")
    print("=" * 80)
    
    # 1. 初始化优化器
    print("1️⃣ 初始化增强版组合优化器...")
    try:
        optimizer = create_enhanced_portfolio_optimizer()
        print(f"   ✅ 成功加载35币种配置")
        print(f"   ✅ 目标夏普比率: {optimizer.objectives['target_sharpe']}")
        print(f"   ✅ 目标波动率: {optimizer.objectives['target_volatility']:.1%}")
        print(f"   ✅ 最大回撤限制: {optimizer.objectives['max_drawdown']:.1%}")
    except Exception as e:
        print(f"   ❌ 初始化失败: {e}")
        return None
    
    # 2. 生成样本数据
    print("\\n2️⃣ 生成样本市场数据...")
    alpha_signals, market_data = generate_sample_data()
    print(f"   ✅ Alpha信号: {len(alpha_signals)} 个币种")
    print(f"   ✅ 市场数据: {len(market_data)} 条记录")
    print(f"   ✅ 时间范围: 过去90天，5分钟频率")
    
    # 3. 运行组合优化
    print("\\n3️⃣ 执行智能组合优化...")
    try:
        # 模拟当前持仓
        current_positions = {
            "BTCUSDT": 0.15,
            "ETHUSDT": 0.10,
            "SOLUSDT": 0.05
        }
        
        target_portfolio = optimizer.optimize_portfolio(
            alpha_signals=alpha_signals,
            market_data=market_data,
            current_positions=current_positions,
            market_regime=MarketRegime.NORMAL_VOLATILITY
        )
        
        print(f"   ✅ 优化完成")
        print(f"   ✅ 最终持仓数量: {target_portfolio['total_positions']}")
        print(f"   ✅ 总杠杆: {target_portfolio['leverage']:.2f}x")
        print(f"   ✅ 预期夏普比率: {target_portfolio['risk']['sharpe']:.2f}")
        print(f"   ✅ 年化波动率: {target_portfolio['risk']['ann_vol']:.1%}")
        
    except Exception as e:
        print(f"   ❌ 优化失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 4. 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results/enhanced_portfolio_optimization"
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存TargetPortfolio.json
    portfolio_file = f"{output_dir}/TargetPortfolio_Enhanced_V5_{timestamp}.json"
    with open(portfolio_file, 'w', encoding='utf-8') as f:
        json.dump(target_portfolio, f, indent=2, ensure_ascii=False)
    
    print(f"\\n4️⃣ 保存结果文件...")
    print(f"   ✅ 目标组合: {portfolio_file}")
    
    # 5. 生成风险报告
    risk_report = generate_comprehensive_risk_report(target_portfolio, optimizer)
    
    report_file = f"{output_dir}/RiskReport_Enhanced_V5_{timestamp}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(risk_report, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ 风险报告: {report_file}")
    
    # 6. 打印关键指标摘要
    print("\\n" + "=" * 80)
    print("📊 DipMaster Enhanced Portfolio V5 - 关键指标摘要")
    print("=" * 80)
    
    print("\\n🎯 投资目标达成情况:")
    constraints = target_portfolio['constraints_status']
    for constraint, status in constraints.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {constraint}: {status}")
    
    print("\\n💰 持仓配置:")
    for weight_info in target_portfolio['weights']:
        print(f"   • {weight_info['symbol']}: {weight_info['w']:.1%} "
              f"(${weight_info['usd_size']:,.0f}) - {weight_info['tier']}")
    
    print("\\n📈 风险指标:")
    risk = target_portfolio['risk']
    print(f"   • 年化波动率: {risk['ann_vol']:.1%}")
    print(f"   • Beta (vs BTC): {risk['beta']:.3f}")
    print(f"   • 夏普比率: {risk['sharpe']:.2f}")
    print(f"   • VaR (95%): {risk['VaR_95']:.1%}")
    print(f"   • 期望损失 (95%): {risk['ES_95']:.1%}")
    print(f"   • 最大回撤: {risk['maximum_drawdown']:.1%}")
    
    print("\\n🏢 交易所分配:")
    for venue, allocation in target_portfolio['venue_allocation'].items():
        print(f"   • {venue.upper()}: {allocation:.1%}")
    
    print("\\n🧮 分层配置:")
    for tier, info in target_portfolio['tier_allocation'].items():
        if info['total_weight'] > 0:
            print(f"   • {tier}: {info['total_weight']:.1%} "
                  f"({info['position_count']} 个币种) - {info['liquidity_tier']}")
    
    print("\\n⚠️  压力测试结果:")
    stress_results = target_portfolio['stress_test_results']
    for scenario, result in stress_results.items():
        var_breach = "⚠️ " if result.get('var_breach', False) else "✅"
        print(f"   {var_breach} {scenario}: "
              f"组合损失 {result['portfolio_loss']:.1%}, "
              f"VaR {result['var_95']:.1%}")
    
    print("\\n" + "=" * 80)
    print("🎉 DipMaster Enhanced Portfolio Optimization V5 完成!")
    print(f"📁 结果文件保存在: {output_dir}/")
    print("=" * 80)
    
    return target_portfolio, risk_report

def generate_comprehensive_risk_report(target_portfolio: dict, optimizer) -> dict:
    """生成详细风险报告"""
    
    risk_report = {
        "report_timestamp": datetime.now().isoformat(),
        "report_version": "DipMaster_Risk_Report_V5",
        "portfolio_summary": {
            "total_positions": target_portfolio['total_positions'],
            "gross_leverage": target_portfolio['leverage'],
            "net_leverage": sum([w['w'] for w in target_portfolio['weights']]),
            "largest_position": max([w['w'] for w in target_portfolio['weights']]) if target_portfolio['weights'] else 0,
            "market_regime": target_portfolio['market_regime']
        },
        
        "risk_metrics_detailed": {
            "volatility_analysis": {
                "annualized_volatility": target_portfolio['risk']['ann_vol'],
                "target_volatility": optimizer.objectives['target_volatility'],
                "volatility_deviation": abs(target_portfolio['risk']['ann_vol'] - optimizer.objectives['target_volatility']),
                "regime_adjusted_vol": target_portfolio['risk']['ann_vol'] * 1.2  # 制度调整
            },
            
            "beta_analysis": {
                "portfolio_beta": target_portfolio['risk']['beta'],
                "beta_target": 0.0,
                "market_neutrality": abs(target_portfolio['risk']['beta']) < 0.05,
                "systematic_risk_exposure": target_portfolio['risk']['beta'] * 0.15  # 假设市场波动15%
            },
            
            "downside_risk": {
                "value_at_risk_95": target_portfolio['risk']['VaR_95'],
                "value_at_risk_99": target_portfolio['risk']['VaR_99'],
                "expected_shortfall_95": target_portfolio['risk']['ES_95'],
                "maximum_drawdown": target_portfolio['risk']['maximum_drawdown'],
                "downside_deviation": target_portfolio['risk']['ann_vol'] * 0.7  # 假设70%下行偏差
            },
            
            "performance_ratios": {
                "sharpe_ratio": target_portfolio['risk']['sharpe'],
                "information_ratio": target_portfolio['risk'].get('information_ratio', 0),
                "calmar_ratio": target_portfolio['risk'].get('calmar_ratio', 0),
                "sortino_ratio": target_portfolio['risk']['sharpe'] * 1.4  # 估算Sortino
            }
        },
        
        "concentration_analysis": {
            "position_concentration": {
                pos['symbol']: pos['w'] for pos in target_portfolio['weights']
            },
            "tier_concentration": target_portfolio['tier_allocation'],
            "herfindahl_index": sum([w['w']**2 for w in target_portfolio['weights']]),
            "effective_number_positions": 1 / sum([w['w']**2 for w in target_portfolio['weights']]) if target_portfolio['weights'] else 0
        },
        
        "correlation_analysis": {
            "risk_attribution": target_portfolio['risk_attribution'],
            "diversification_benefit": target_portfolio['risk_attribution'].get('diversification_ratio', 1.0),
            "correlation_risk": "Medium",  # 简化评级
            "cluster_risk": "Low"
        },
        
        "stress_test_summary": {
            "scenarios_tested": len(target_portfolio['stress_test_results']),
            "worst_case_loss": min([r['portfolio_loss'] for r in target_portfolio['stress_test_results'].values()]),
            "var_breaches": sum([1 for r in target_portfolio['stress_test_results'].values() if r.get('var_breach', False)]),
            "stress_test_passed": all([not r.get('var_breach', False) for r in target_portfolio['stress_test_results'].values()])
        },
        
        "liquidity_analysis": {
            "tier_1_allocation": target_portfolio['tier_allocation'].get('tier_1_mega_cap', {}).get('total_weight', 0),
            "tier_2_allocation": target_portfolio['tier_allocation'].get('tier_2_large_cap', {}).get('total_weight', 0),
            "tier_3_allocation": target_portfolio['tier_allocation'].get('tier_3_mid_cap', {}).get('total_weight', 0),
            "liquidity_score": "High",  # 简化评分
            "estimated_liquidation_time": "< 30 minutes"
        },
        
        "compliance_status": target_portfolio['constraints_status'],
        
        "recommendations": [
            {
                "priority": "High",
                "category": "Risk Management",
                "recommendation": "监控Beta中性状态，确保市场风险敞口控制在±5%以内",
                "action_required": target_portfolio['constraints_status']['beta_neutral']
            },
            {
                "priority": "Medium", 
                "category": "Position Sizing",
                "recommendation": "定期评估Kelly分数有效性，基于最新绩效数据调整",
                "action_required": True
            },
            {
                "priority": "Low",
                "category": "Diversification",
                "recommendation": "考虑增加非相关资产以进一步降低组合风险",
                "action_required": False
            }
        ],
        
        "monitoring_alerts": {
            "real_time_alerts": [
                "Portfolio Beta exceeds ±0.05",
                "Individual position exceeds 25%",
                "Daily loss exceeds 2%",
                "VaR breach detected",
                "Correlation spike above 0.8"
            ],
            "alert_thresholds": optimizer.config['monitoring_and_alerts']['alert_thresholds']
        }
    }
    
    return risk_report

if __name__ == "__main__":
    # 运行完整演示
    try:
        target_portfolio, risk_report = generate_target_portfolio_demo()
        
        if target_portfolio:
            print("\\n🔗 快速访问链接:")
            print("   📊 查看结果: results/enhanced_portfolio_optimization/")
            print("   📈 启动监控: python src/dashboard/start_dashboard.py")
            print("   🔄 重新优化: python run_enhanced_portfolio_optimization.py")
            
    except KeyboardInterrupt:
        print("\\n\\n⏹️ 用户中断操作")
    except Exception as e:
        print(f"\\n\\n❌ 演示过程出错: {e}")
        import traceback
        traceback.print_exc()