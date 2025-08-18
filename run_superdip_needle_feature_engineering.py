#!/usr/bin/env python3
"""
SuperDip Needle Feature Engineering Execution Script
超跌接针策略特征工程执行脚本

执行SuperDip Needle策略的完整特征工程流程，包括：
1. 特征生成
2. 质量验证
3. 重要性分析
4. 报告生成

Author: DipMaster Quant Team
Date: 2025-08-18
Version: 1.0.0
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

from src.data.superdip_needle_feature_engineer import (
    SuperDipNeedleFeatureEngineer, 
    SuperDipNeedleConfig
)

# 配置
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/superdip_needle_feature_engineering.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def generate_feature_importance_report(results: Dict[str, Any], output_dir: Path) -> str:
    """生成特征重要性分析报告"""
    try:
        logger = logging.getLogger(__name__)
        logger.info("Generating feature importance analysis report...")
        
        # 收集所有特征重要性数据
        all_importance = {}
        symbol_count = {}
        
        for symbol, stats in results['processing_stats'].items():
            if 'feature_importance' in stats and stats['feature_importance']:
                for feature, importance in stats['feature_importance'].items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                        symbol_count[feature] = 0
                    all_importance[feature].append(importance)
                    symbol_count[feature] += 1
        
        # 计算平均重要性
        avg_importance = {}
        for feature, scores in all_importance.items():
            avg_importance[feature] = np.mean(scores)
        
        # 按重要性排序
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        # 生成报告
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f"SuperDipNeedle_FeatureImportance_Report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# SuperDip Needle Strategy - Feature Importance Analysis Report\n")
            f.write("# 超跌接针策略 - 特征重要性分析报告\n\n")
            
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**处理币种数**: {len(results['processing_stats'])}\n")
            f.write(f"**总特征数**: {len(sorted_features)}\n\n")
            
            # 执行摘要
            f.write("## 执行摘要\n\n")
            f.write("本报告分析了SuperDip Needle策略中各特征的预测重要性，")
            f.write("基于互信息方法计算特征与目标收益之间的关联度。\n\n")
            
            # Top 20 重要特征
            f.write("## 🏆 Top 20 重要特征\n\n")
            f.write("| 排名 | 特征名称 | 平均重要性 | 出现币种数 | 特征类别 |\n")
            f.write("|------|----------|------------|------------|----------|\n")
            
            for i, (feature, importance) in enumerate(sorted_features[:20]):
                rank = i + 1
                count = symbol_count[feature]
                category = classify_feature(feature)
                f.write(f"| {rank} | `{feature}` | {importance:.6f} | {count} | {category} |\n")
            
            # 特征类别分析
            f.write("\n## 📊 特征类别重要性分析\n\n")
            
            category_importance = {}
            for feature, importance in sorted_features:
                category = classify_feature(feature)
                if category not in category_importance:
                    category_importance[category] = []
                category_importance[category].append(importance)
            
            category_avg = {k: np.mean(v) for k, v in category_importance.items()}
            sorted_categories = sorted(category_avg.items(), key=lambda x: x[1], reverse=True)
            
            f.write("| 特征类别 | 平均重要性 | 特征数量 | 描述 |\n")
            f.write("|----------|------------|----------|---------|\n")
            
            category_descriptions = {
                'RSI超卖': 'RSI指标及其衍生特征，用于识别超卖状态',
                '布林带': '布林带位置和突破特征，衡量价格偏离程度',
                '成交量': '成交量相关特征，包括放大、比率和背离',
                '价格形态': 'K线形态特征，如锤子线、十字星等',
                '多时间框架': '跨时间框架的趋势和信号一致性',
                '微结构': '市场微结构特征，如流动性、买卖压力',
                '交互特征': '多个指标的组合和交互特征',
                '趋势动量': '价格趋势和动量相关特征',
                '波动率': '价格波动率和市场状态特征',
                '其他': '其他辅助特征'
            }
            
            for category, avg_imp in sorted_categories:
                count = len(category_importance[category])
                desc = category_descriptions.get(category, '未分类特征')
                f.write(f"| {category} | {avg_imp:.6f} | {count} | {desc} |\n")
            
            # 关键发现
            f.write("\n## 🔍 关键发现\n\n")
            
            # 分析最重要的特征类别
            top_category = sorted_categories[0][0]
            f.write(f"1. **最重要特征类别**: {top_category}，平均重要性为 {sorted_categories[0][1]:.6f}\n")
            
            # 分析最重要的单个特征
            top_feature = sorted_features[0][0]
            f.write(f"2. **最重要单个特征**: `{top_feature}`，重要性为 {sorted_features[0][1]:.6f}\n")
            
            # 特征一致性分析
            high_consistency = [f for f, c in symbol_count.items() if c >= len(results['processing_stats']) * 0.8]
            f.write(f"3. **高一致性特征**: {len(high_consistency)} 个特征在80%以上的币种中表现重要\n")
            
            # 推荐的特征选择策略
            f.write("\n## 💡 特征选择建议\n\n")
            f.write("基于重要性分析，推荐以下特征选择策略：\n\n")
            
            # 核心特征集（Top 30）
            f.write("### 核心特征集 (Top 30)\n\n")
            core_features = [f[0] for f in sorted_features[:30]]
            for i, feature in enumerate(core_features, 1):
                f.write(f"{i}. `{feature}`\n")
            
            # 扩展特征集（Top 50）
            f.write("\n### 扩展特征集 (Top 31-50)\n\n")
            extended_features = [f[0] for f in sorted_features[30:50]]
            for i, feature in enumerate(extended_features, 31):
                f.write(f"{i}. `{feature}`\n")
            
            # 特征工程建议
            f.write("\n## 🛠️ 特征工程优化建议\n\n")
            f.write("1. **重点关注RSI和布林带特征**: 这些传统技术指标在超跌识别中表现优异\n")
            f.write("2. **成交量确认很重要**: 成交量相关特征能有效确认价格信号\n")
            f.write("3. **多时间框架融合**: 跨时间框架特征提供重要的确认信息\n")
            f.write("4. **交互特征有价值**: 多指标组合特征提供额外的预测能力\n")
            f.write("5. **考虑特征稳定性**: 优先选择在多个币种中都重要的特征\n\n")
            
            # 风险提示
            f.write("## ⚠️ 风险提示\n\n")
            f.write("1. 特征重要性可能随市场环境变化，需要定期重新评估\n")
            f.write("2. 避免过度拟合，不建议使用过多相关性强的特征\n")
            f.write("3. 实盘应用时需要考虑特征计算的实时性和稳定性\n")
            f.write("4. 建议结合业务理解进行特征选择，不完全依赖统计指标\n\n")
            
            f.write("---\n")
            f.write("*报告由SuperDip Needle Feature Engineering Pipeline自动生成*\n")
        
        logger.info(f"Feature importance report saved to: {report_file}")
        return str(report_file)
        
    except Exception as e:
        logger.error(f"Failed to generate feature importance report: {e}")
        return ""

def classify_feature(feature_name: str) -> str:
    """根据特征名称分类"""
    feature_lower = feature_name.lower()
    
    if 'rsi' in feature_lower:
        return 'RSI超卖'
    elif 'bb_' in feature_lower or 'bollinger' in feature_lower:
        return '布林带'
    elif 'volume' in feature_lower or 'money_flow' in feature_lower or 'vwap' in feature_lower:
        return '成交量'
    elif any(pattern in feature_lower for pattern in ['hammer', 'doji', 'candle', 'shadow', 'body']):
        return '价格形态'
    elif 'htf_' in feature_lower or 'hf_' in feature_lower or 'consensus' in feature_lower:
        return '多时间框架'
    elif any(pattern in feature_lower for pattern in ['illiquidity', 'spread', 'flow']):
        return '微结构'
    elif any(pattern in feature_lower for pattern in ['combo', 'interaction', 'product']):
        return '交互特征'
    elif any(pattern in feature_lower for pattern in ['trend', 'momentum', 'acceleration']):
        return '趋势动量'
    elif any(pattern in feature_lower for pattern in ['volatility', 'regime', 'state']):
        return '波动率'
    else:
        return '其他'

def generate_data_quality_report(results: Dict[str, Any], output_dir: Path) -> str:
    """生成数据质量报告"""
    try:
        logger = logging.getLogger(__name__)
        logger.info("Generating data quality report...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f"SuperDipNeedle_DataQuality_Report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# SuperDip Needle Strategy - Data Quality Report\n")
            f.write("# 超跌接针策略 - 数据质量报告\n\n")
            
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**处理币种数**: {len(results['processing_stats'])}\n\n")
            
            # 数据质量概览
            f.write("## 📊 数据质量概览\n\n")
            
            total_symbols = len(results['processing_stats'])
            passed_symbols = 0
            warning_symbols = 0
            critical_symbols = 0
            
            for symbol, stats in results['processing_stats'].items():
                if 'data_quality' in stats:
                    quality = stats['data_quality']
                    missing_analysis = quality.get('missing_data_analysis', {})
                    
                    # 统计质量状态
                    has_critical = any(info.get('status') == 'CRITICAL' for info in missing_analysis.values())
                    has_warning = any(info.get('status') == 'WARNING' for info in missing_analysis.values())
                    
                    if has_critical:
                        critical_symbols += 1
                    elif has_warning:
                        warning_symbols += 1
                    else:
                        passed_symbols += 1
            
            f.write(f"- ✅ **通过检查**: {passed_symbols} 个币种 ({passed_symbols/total_symbols*100:.1f}%)\n")
            f.write(f"- ⚠️ **存在警告**: {warning_symbols} 个币种 ({warning_symbols/total_symbols*100:.1f}%)\n")
            f.write(f"- ❌ **严重问题**: {critical_symbols} 个币种 ({critical_symbols/total_symbols*100:.1f}%)\n\n")
            
            # 详细质量分析
            f.write("## 🔍 详细质量分析\n\n")
            
            for symbol, stats in results['processing_stats'].items():
                if 'data_quality' in stats:
                    quality = stats['data_quality']
                    f.write(f"### {symbol}\n\n")
                    
                    # 基本信息
                    f.write(f"- **数据行数**: {quality.get('total_rows', 'N/A')}\n")
                    f.write(f"- **特征列数**: {quality.get('total_features', 'N/A')}\n")
                    
                    # 数据泄露检查
                    leakage_check = quality.get('data_leakage_check', {})
                    if leakage_check.get('status') == 'PASSED':
                        f.write("- **数据泄露检查**: ✅ 通过\n")
                    else:
                        f.write("- **数据泄露检查**: ⚠️ 发现潜在问题\n")
                    
                    # 缺失值分析
                    missing_analysis = quality.get('missing_data_analysis', {})
                    if missing_analysis:
                        critical_missing = sum(1 for info in missing_analysis.values() if info.get('status') == 'CRITICAL')
                        warning_missing = sum(1 for info in missing_analysis.values() if info.get('status') == 'WARNING')
                        f.write(f"- **缺失值状态**: {critical_missing} 严重, {warning_missing} 警告\n")
                    
                    # 异常值分析
                    outlier_analysis = quality.get('outlier_analysis', {})
                    if outlier_analysis:
                        high_outliers = sum(1 for info in outlier_analysis.values() if info.get('status') == 'WARNING')
                        f.write(f"- **异常值状态**: {high_outliers} 个特征异常值较多\n")
                    
                    # 特征稳定性
                    stability = quality.get('feature_stability', {})
                    if stability:
                        unstable_features = sum(1 for info in stability.values() if info.get('stability') == 'UNSTABLE')
                        f.write(f"- **特征稳定性**: {unstable_features} 个特征不稳定\n")
                    
                    # 建议
                    recommendations = quality.get('recommendations', [])
                    if recommendations:
                        f.write("- **改进建议**:\n")
                        for rec in recommendations:
                            f.write(f"  - {rec}\n")
                    
                    f.write("\n")
            
            # 总结和建议
            f.write("## 💡 总结与建议\n\n")
            f.write("### 数据质量总结\n\n")
            if critical_symbols == 0:
                f.write("✅ **整体质量良好**: 所有币种数据都达到了基本质量要求\n\n")
            else:
                f.write(f"⚠️ **需要关注**: {critical_symbols} 个币种存在严重数据质量问题，建议重点检查\n\n")
            
            f.write("### 改进建议\n\n")
            f.write("1. **数据预处理**: 加强缺失值和异常值的预处理\n")
            f.write("2. **稳定性监控**: 定期检查特征稳定性，及时发现分布漂移\n")
            f.write("3. **质量流程**: 建立自动化的数据质量监控流程\n")
            f.write("4. **验证机制**: 增加更多的数据验证规则\n\n")
            
            f.write("---\n")
            f.write("*报告由SuperDip Needle Feature Engineering Pipeline自动生成*\n")
        
        logger.info(f"Data quality report saved to: {report_file}")
        return str(report_file)
        
    except Exception as e:
        logger.error(f"Failed to generate data quality report: {e}")
        return ""

def create_feature_visualization(results: Dict[str, Any], output_dir: Path) -> str:
    """创建特征重要性可视化"""
    try:
        logger = logging.getLogger(__name__)
        logger.info("Creating feature importance visualization...")
        
        # 收集特征重要性数据
        all_importance = {}
        for symbol, stats in results['processing_stats'].items():
            if 'feature_importance' in stats and stats['feature_importance']:
                for feature, importance in stats['feature_importance'].items():
                    if feature not in all_importance:
                        all_importance[feature] = []
                    all_importance[feature].append(importance)
        
        # 计算平均重要性
        avg_importance = {feature: np.mean(scores) for feature, scores in all_importance.items()}
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        # 创建可视化
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Top 20 特征重要性
        top_20 = sorted_features[:20]
        features = [f[0] for f in top_20]
        importances = [f[1] for f in top_20]
        
        bars = ax1.barh(range(len(features)), importances)
        ax1.set_yticks(range(len(features)))
        ax1.set_yticklabels(features, fontsize=10)
        ax1.set_xlabel('Feature Importance (Mutual Information)')
        ax1.set_title('Top 20 Most Important Features - SuperDip Needle Strategy')
        ax1.invert_yaxis()
        
        # 添加数值标签
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax1.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', ha='left', va='center', fontsize=8)
        
        # 特征类别重要性分布
        category_importance = {}
        for feature, importance in sorted_features:
            category = classify_feature(feature)
            if category not in category_importance:
                category_importance[category] = []
            category_importance[category].append(importance)
        
        categories = list(category_importance.keys())
        avg_by_category = [np.mean(category_importance[cat]) for cat in categories]
        
        bars2 = ax2.bar(categories, avg_by_category)
        ax2.set_xlabel('Feature Category')
        ax2.set_ylabel('Average Importance')
        ax2.set_title('Feature Importance by Category')
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = output_dir / f"SuperDipNeedle_FeatureImportance_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Feature importance visualization saved to: {plot_file}")
        return str(plot_file)
        
    except Exception as e:
        logger.error(f"Failed to create feature visualization: {e}")
        return ""

def main():
    """主执行函数"""
    # 设置日志
    os.makedirs('logs', exist_ok=True)
    logger = setup_logging()
    
    try:
        logger.info("=== SuperDip Needle Feature Engineering Started ===")
        
        # 检查数据文件
        bundle_path = "data/MarketDataBundle_Top30_Enhanced_Final.json"
        if not Path(bundle_path).exists():
            logger.error(f"Market data bundle not found: {bundle_path}")
            return
        
        # 创建输出目录
        output_dir = Path("data/superdip_needle_features")
        output_dir.mkdir(exist_ok=True)
        results_dir = Path("results/superdip_needle")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建配置
        config = SuperDipNeedleConfig(
            symbols=[
                'ADAUSDT', 'APTUSDT', 'ARBUSDT', 'ATOMUSDT', 'AVAXUSDT',
                'BNBUSDT', 'DOGEUSDT', 'DOTUSDT', 'FILUSDT', 'LINKUSDT',
                'LTCUSDT', 'NEARUSDT', 'OPUSDT', 'SOLUSDT', 'TONUSDT',
                'TRXUSDT', 'UNIUSDT', 'XLMUSDT', 'XRPUSDT', 'MATICUSDT'
            ],
            timeframes=['1m', '5m', '15m', '1h'],
            primary_timeframe='5m',
            prediction_horizons=[15, 30, 60, 240],
            profit_targets=[0.008, 0.015, 0.025, 0.040],
            enable_cross_timeframe=True,
            enable_microstructure=True,
            enable_advanced_labels=True,
            enable_interaction_features=True,
            enable_regime_features=True
        )
        
        # 创建特征工程器
        feature_engineer = SuperDipNeedleFeatureEngineer(config)
        
        # 执行特征工程
        logger.info("Starting feature engineering pipeline...")
        results = feature_engineer.generate_feature_set(bundle_path, str(output_dir))
        
        # 生成报告
        logger.info("Generating analysis reports...")
        
        # 特征重要性报告
        importance_report = generate_feature_importance_report(results, results_dir)
        
        # 数据质量报告
        quality_report = generate_data_quality_report(results, results_dir)
        
        # 特征可视化
        visualization_file = create_feature_visualization(results, results_dir)
        
        # 打印总结
        logger.info("=== Feature Engineering Completed Successfully ===")
        print("\n" + "="*80)
        print("🎉 SuperDip Needle Feature Engineering 完成!")
        print("="*80)
        print(f"✅ 特征集文件: {results['feature_set_file']}")
        print(f"✅ 标签集文件: {results['label_set_file']}")
        print(f"✅ 处理币种数: {results['summary']['results_summary']['total_symbols_processed']}")
        print(f"✅ 成功率: {results['summary']['results_summary']['success_rate']:.1%}")
        print(f"✅ 平均特征数: {results['summary']['results_summary']['average_features_per_symbol']:.0f}")
        
        if importance_report:
            print(f"📊 特征重要性报告: {importance_report}")
        if quality_report:
            print(f"📋 数据质量报告: {quality_report}")
        if visualization_file:
            print(f"📈 特征可视化: {visualization_file}")
        
        print("\n💡 下一步建议:")
        print("1. 查看生成的特征重要性报告，选择核心特征集")
        print("2. 检查数据质量报告，确保数据质量符合要求")
        print("3. 使用生成的特征数据进行模型训练和回测")
        print("4. 根据实盘表现调整特征工程策略")
        
        print("\n" + "="*80)
        
    except Exception as e:
        logger.error(f"Feature engineering pipeline failed: {e}")
        print(f"❌ 特征工程失败: {e}")
        raise

if __name__ == "__main__":
    main()