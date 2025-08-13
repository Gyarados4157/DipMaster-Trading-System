#!/usr/bin/env python3
"""
Quick Overfitting Analysis
快速过拟合分析 - 基于检测结果的深入解读

Author: DipMaster Risk Analysis Team
Date: 2025-08-13
Version: 1.0.0
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

def analyze_overfitting_results():
    """分析过拟合检测结果"""
    
    # 加载检测结果
    result_file = "overfitting_analysis_20250813_201327.json"
    
    if not Path(result_file).exists():
        print("❌ 过拟合分析结果文件不存在")
        return
    
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print("🚨 DipMaster过拟合风险深入分析")
    print("=" * 80)
    
    # 1. 样本内外表现分析
    print("\n📊 1. 样本内外表现分析")
    print("-" * 50)
    
    sample_analysis = results['sample_analysis']
    original = sample_analysis['original_config']
    optimized = sample_analysis['optimized_config']
    
    print(f"原始参数 (RSI 30-50, MA20):")
    print(f"  样本内胜率: {original['train_win_rate']:.1f}%")
    print(f"  样本外胜率: {original['test_win_rate']:.1f}%")
    print(f"  性能衰减: {original['performance_diff']:+.1f}%")
    
    print(f"\n优化参数 (RSI 40-60, MA30):")
    print(f"  样本内胜率: {optimized['train_win_rate']:.1f}%")
    print(f"  样本外胜率: {optimized['test_win_rate']:.1f}%")
    print(f"  性能衰减: {optimized['performance_diff']:+.1f}%")
    
    # 关键发现
    performance_improvement = optimized['test_win_rate'] - original['test_win_rate']
    print(f"\n🎯 关键发现:")
    print(f"  样本外性能改进: {performance_improvement:+.1f}%")
    print(f"  优化后性能衰减更小: {optimized['performance_diff']:.1f}% vs {original['performance_diff']:.1f}%")
    
    # 2. 参数敏感性分析
    print("\n🔬 2. 参数敏感性分析")
    print("-" * 50)
    
    sensitivity = results['sensitivity_analysis']
    
    # RSI敏感性
    rsi_data = sensitivity['rsi_sensitivity']
    rsi_win_rates = [r['win_rate'] for r in rsi_data]
    rsi_range = max(rsi_win_rates) - min(rsi_win_rates)
    
    print(f"RSI参数敏感性:")
    print(f"  胜率范围: {min(rsi_win_rates):.1f}% - {max(rsi_win_rates):.1f}%")
    print(f"  差异幅度: {rsi_range:.1f}%")
    print(f"  敏感性评价: {'较高' if rsi_range > 10 else '中等' if rsi_range > 5 else '较低'}")
    
    # MA敏感性
    ma_data = sensitivity['ma_sensitivity']
    ma_win_rates = [r['win_rate'] for r in ma_data]
    ma_range = max(ma_win_rates) - min(ma_win_rates)
    
    print(f"\nMA周期敏感性:")
    print(f"  胜率范围: {min(ma_win_rates):.1f}% - {max(ma_win_rates):.1f}%")
    print(f"  差异幅度: {ma_range:.1f}%")
    print(f"  敏感性评价: {'较高' if ma_range > 10 else '中等' if ma_range > 5 else '较低'}")
    
    # 盈利目标敏感性
    profit_data = sensitivity['profit_sensitivity']
    profit_win_rates = [r['win_rate'] for r in profit_data]
    profit_range = max(profit_win_rates) - min(profit_win_rates)
    
    print(f"\n盈利目标敏感性:")
    print(f"  胜率范围: {min(profit_win_rates):.1f}% - {max(profit_win_rates):.1f}%")
    print(f"  差异幅度: {profit_range:.1f}%")
    print(f"  敏感性评价: {'较高' if profit_range > 10 else '中等' if profit_range > 5 else '较低'}")
    
    # 3. 前向验证分析
    print("\n🔄 3. 前向验证分析")
    print("-" * 50)
    
    forward_validation = results['forward_validation']
    validation_results = forward_validation['validation_results']
    
    if validation_results:
        degradations = [r['performance_degradation'] for r in validation_results]
        avg_degradation = np.mean(degradations)
        max_degradation = max(degradations)
        min_degradation = min(degradations)
        std_degradation = np.std(degradations)
        
        print(f"前向验证统计 (18个测试期间):")
        print(f"  平均性能衰减: {avg_degradation:+.1f}%")
        print(f"  最大性能衰减: {max_degradation:+.1f}%")
        print(f"  最小性能衰减: {min_degradation:+.1f}%")
        print(f"  衰减标准差: {std_degradation:.1f}%")
        
        # 稳定性评估
        stable_periods = len([d for d in degradations if abs(d) < 3])
        stability_ratio = stable_periods / len(degradations)
        
        print(f"\n稳定性评估:")
        print(f"  稳定期间数: {stable_periods}/{len(degradations)}")
        print(f"  稳定性比例: {stability_ratio:.1%}")
        print(f"  稳定性评价: {'高' if stability_ratio > 0.7 else '中等' if stability_ratio > 0.5 else '低'}")
    
    # 4. 统计显著性分析
    print("\n📊 4. 统计显著性分析")
    print("-" * 50)
    
    stats_test = results['statistical_test']
    
    print(f"t检验结果:")
    print(f"  t统计量: {stats_test['t_statistic']:.3f}")
    print(f"  p值: {stats_test['p_value']:.3f}")
    print(f"  统计显著: {'是' if stats_test['significant'] else '否'} (α=0.05)")
    print(f"  显著性解释: {'参数优化具有统计学意义' if stats_test['significant'] else '参数改进可能是随机波动'}")
    
    # 5. 综合风险评估
    print("\n🚨 5. 过拟合综合风险评估")
    print("-" * 50)
    
    assessment = results['overfitting_assessment']
    
    print(f"风险等级: {assessment['risk_level']}")
    print(f"风险评分: {assessment['overfitting_score']}/100")
    
    if assessment['risk_factors']:
        print(f"\n识别的风险因素:")
        for i, factor in enumerate(assessment['risk_factors'], 1):
            print(f"  {i}. {factor}")
    else:
        print(f"\n✅ 未发现明显的过拟合风险因素")
    
    # 6. 深度分析结论
    print("\n🎯 6. 深度分析结论")
    print("-" * 50)
    
    # 基于所有分析得出结论
    conclusions = []
    
    # 样本内外表现分析
    if optimized['performance_diff'] < 5:
        conclusions.append("✅ 样本内外表现差异较小，过拟合风险低")
    elif optimized['performance_diff'] < 10:
        conclusions.append("⚠️ 样本内外表现存在一定差异，需要关注")
    else:
        conclusions.append("🚨 样本内外表现差异较大，可能存在过拟合")
    
    # 参数敏感性分析
    if max(rsi_range, ma_range, profit_range) < 5:
        conclusions.append("✅ 参数敏感性较低，策略较为稳健")
    elif max(rsi_range, ma_range, profit_range) < 10:
        conclusions.append("⚠️ 参数具有中等敏感性，需要谨慎调整")
    else:
        conclusions.append("🚨 参数敏感性较高，策略可能不够稳健")
    
    # 前向验证分析
    if validation_results and abs(avg_degradation) < 3:
        conclusions.append("✅ 前向验证显示策略表现稳定")
    elif validation_results and abs(avg_degradation) < 8:
        conclusions.append("⚠️ 前向验证显示策略表现一般")
    else:
        conclusions.append("🚨 前向验证显示策略表现不稳定")
    
    # 统计显著性分析
    if stats_test['significant']:
        conclusions.append("✅ 参数改进具有统计学意义")
    else:
        conclusions.append("⚠️ 参数改进缺乏统计显著性")
    
    print("分析结论:")
    for conclusion in conclusions:
        print(f"  {conclusion}")
    
    # 7. 最终建议
    print("\n💡 7. 最终建议")
    print("-" * 50)
    
    risk_score = assessment['overfitting_score']
    
    if risk_score < 30:
        print("🟢 低风险策略 - 建议操作:")
        print("  • 可以谨慎使用优化后的参数")
        print("  • 建议小额资金先期测试")
        print("  • 持续监控实际表现")
        print("  • 每月评估策略效果")
    elif risk_score < 50:
        print("🟡 中等风险策略 - 建议操作:")
        print("  • 使用更保守的参数组合")
        print("  • 增加更多样本外验证")
        print("  • 考虑参数的鲁棒性")
        print("  • 实施严格的风险控制")
    elif risk_score < 70:
        print("🟠 高风险策略 - 建议操作:")
        print("  • 重新审视参数优化过程")
        print("  • 扩大验证数据集")
        print("  • 简化策略复杂度")
        print("  • 加强实时监控")
    else:
        print("🔴 严重过拟合风险 - 建议操作:")
        print("  • 立即停止使用优化参数")
        print("  • 回到更保守的设置")
        print("  • 重新设计优化流程")
        print("  • 增加正则化约束")
    
    # 8. 具体参数建议
    print("\n⚙️ 8. 具体参数建议")
    print("-" * 50)
    
    # 基于分析结果给出具体建议
    if risk_score < 30:
        print("推荐参数组合 (基于稳健性考虑):")
        print("  RSI范围: (38, 58) - 平衡敏感性和稳健性")
        print("  MA周期: 28-30 - 兼顾趋势识别和平滑性")
        print("  盈利目标: 1.0%-1.2% - 既不过于保守也不过于激进")
        print("  杠杆倍数: 5-8x - 降低风险")
    else:
        print("保守参数组合 (降低过拟合风险):")
        print("  RSI范围: (35, 55) - 更宽松的条件")
        print("  MA周期: 25 - 较短期趋势")
        print("  盈利目标: 0.8% - 更保守的目标")
        print("  杠杆倍数: 3-5x - 显著降低风险")

def main():
    """主函数"""
    analyze_overfitting_results()
    
    # 保存分析报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n💾 详细分析已完成，时间: {timestamp}")

if __name__ == "__main__":
    main()