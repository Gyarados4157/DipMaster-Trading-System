#!/usr/bin/env python3
"""
多资产验证器 - 消除选择偏差
Multi-Asset Validator - Eliminate Selection Bias

核心功能:
1. 多币种一致性验证
2. 消除资产选择偏差
3. 跨资产稳定性测试
4. 全面性能评估

Author: DipMaster Trading Team
Date: 2025-08-15
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class AssetValidationResult:
    """单个资产验证结果"""
    symbol: str
    total_trades: int
    win_rate: float
    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_duration: float
    consistency_score: float
    performance_rank: int

@dataclass
class MultiAssetAnalysis:
    """多资产分析结果"""
    asset_results: List[AssetValidationResult]
    consistency_metrics: Dict
    selection_bias_score: float
    overall_stability: float
    recommended_assets: List[str]
    rejected_assets: List[str]
    warnings: List[str]

class MultiAssetValidator:
    """
    多资产验证器
    
    确保策略在所有资产上表现一致，消除选择偏差
    """
    
    def __init__(self, min_consistency_threshold: float = 0.6):
        self.min_consistency_threshold = min_consistency_threshold
        self.results_dir = Path("results/multi_asset_validation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 标准测试币种
        self.standard_symbols = [
            'BTCUSDT', 'ADAUSDT', 'ALGOUSDT', 'BNBUSDT', 'DOGEUSDT',
            'ICPUSDT', 'IOTAUSDT', 'SOLUSDT', 'SUIUSDT', 'XRPUSDT'
        ]
    
    def validate_multi_asset_strategy(self, 
                                    strategy_results: Dict[str, Dict],
                                    market_data: Dict[str, pd.DataFrame] = None) -> MultiAssetAnalysis:
        """
        验证多资产策略一致性
        
        Args:
            strategy_results: 各币种策略结果 {symbol: results}
            market_data: 各币种市场数据 {symbol: data}
            
        Returns:
            MultiAssetAnalysis: 多资产分析结果
        """
        logger.info("开始多资产策略验证...")
        
        # 验证数据完整性
        self._validate_data_completeness(strategy_results)
        
        # 计算各资产性能指标
        asset_results = self._calculate_asset_metrics(strategy_results)
        
        # 一致性分析
        consistency_metrics = self._analyze_cross_asset_consistency(asset_results)
        
        # 选择偏差检测
        selection_bias_score = self._detect_selection_bias(asset_results)
        
        # 稳定性评估
        overall_stability = self._assess_overall_stability(asset_results)
        
        # 资产筛选建议
        recommended_assets, rejected_assets = self._recommend_asset_selection(
            asset_results, consistency_metrics
        )
        
        # 生成警告
        warnings = self._generate_warnings(
            asset_results, consistency_metrics, selection_bias_score
        )
        
        # 创建分析结果
        analysis = MultiAssetAnalysis(
            asset_results=asset_results,
            consistency_metrics=consistency_metrics,
            selection_bias_score=selection_bias_score,
            overall_stability=overall_stability,
            recommended_assets=recommended_assets,
            rejected_assets=rejected_assets,
            warnings=warnings
        )
        
        # 保存结果
        self._save_validation_results(analysis)
        
        # 生成可视化报告
        self._generate_visualization_report(analysis)
        
        logger.info("多资产策略验证完成")
        return analysis
    
    def _validate_data_completeness(self, strategy_results: Dict[str, Dict]) -> None:
        """验证数据完整性"""
        missing_symbols = set(self.standard_symbols) - set(strategy_results.keys())
        
        if missing_symbols:
            logger.warning(f"缺少以下币种的数据: {missing_symbols}")
            
        available_symbols = set(strategy_results.keys()) & set(self.standard_symbols)
        if len(available_symbols) < 5:
            raise ValueError(f"可用币种数量不足: {len(available_symbols)} < 5")
        
        logger.info(f"验证 {len(available_symbols)} 个币种的数据")
    
    def _calculate_asset_metrics(self, strategy_results: Dict[str, Dict]) -> List[AssetValidationResult]:
        """计算各资产性能指标"""
        asset_results = []
        all_metrics = []
        
        for symbol, results in strategy_results.items():
            if symbol not in self.standard_symbols:
                continue
            
            # 提取基础指标
            trades_data = results.get('trades', [])
            if not trades_data:
                logger.warning(f"{symbol} 没有交易数据")
                continue
            
            # 转换为DataFrame
            if isinstance(trades_data, list):
                trades_df = pd.DataFrame(trades_data)
            else:
                trades_df = trades_data
            
            if len(trades_df) == 0:
                continue
            
            # 计算性能指标
            total_trades = len(trades_df)
            win_rate = (trades_df['pnl'] > 0).mean()
            total_pnl = trades_df['pnl'].sum()
            
            # 计算夏普比率
            if trades_df['pnl'].std() > 0:
                sharpe_ratio = trades_df['pnl'].mean() / trades_df['pnl'].std()
            else:
                sharpe_ratio = 0
            
            # 计算最大回撤
            cumulative_pnl = trades_df['pnl'].cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = cumulative_pnl - running_max
            max_drawdown = drawdown.min()
            
            # 计算平均交易持续时间
            if 'duration_minutes' in trades_df.columns:
                avg_trade_duration = trades_df['duration_minutes'].mean()
            else:
                avg_trade_duration = 0
            
            asset_result = AssetValidationResult(
                symbol=symbol,
                total_trades=total_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                avg_trade_duration=avg_trade_duration,
                consistency_score=0,  # 后续计算
                performance_rank=0   # 后续计算
            )
            
            asset_results.append(asset_result)
            all_metrics.append({
                'symbol': symbol,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'sharpe_ratio': sharpe_ratio
            })
        
        # 计算性能排名
        asset_results.sort(key=lambda x: x.total_pnl, reverse=True)
        for i, result in enumerate(asset_results):
            result.performance_rank = i + 1
        
        # 计算一致性得分
        self._calculate_consistency_scores(asset_results)
        
        logger.info(f"计算了 {len(asset_results)} 个资产的性能指标")
        return asset_results
    
    def _calculate_consistency_scores(self, asset_results: List[AssetValidationResult]) -> None:
        """计算一致性得分"""
        if len(asset_results) < 2:
            return
        
        # 提取指标
        win_rates = [r.win_rate for r in asset_results]
        sharpe_ratios = [r.sharpe_ratio for r in asset_results]
        
        # 计算变异系数
        win_rate_cv = np.std(win_rates) / np.mean(win_rates) if np.mean(win_rates) > 0 else float('inf')
        sharpe_cv = np.std(sharpe_ratios) / abs(np.mean(sharpe_ratios)) if np.mean(sharpe_ratios) != 0 else float('inf')
        
        # 为每个资产计算一致性得分
        for result in asset_results:
            # 基于偏离程度的一致性得分
            win_rate_deviation = abs(result.win_rate - np.mean(win_rates)) / np.std(win_rates) if np.std(win_rates) > 0 else 0
            sharpe_deviation = abs(result.sharpe_ratio - np.mean(sharpe_ratios)) / np.std(sharpe_ratios) if np.std(sharpe_ratios) > 0 else 0
            
            # 一致性得分 (0-1)
            consistency = 1 / (1 + win_rate_deviation + sharpe_deviation)
            result.consistency_score = consistency
    
    def _analyze_cross_asset_consistency(self, asset_results: List[AssetValidationResult]) -> Dict:
        """分析跨资产一致性"""
        if len(asset_results) < 2:
            return {"error": "资产数量不足进行一致性分析"}
        
        # 提取指标
        symbols = [r.symbol for r in asset_results]
        win_rates = [r.win_rate for r in asset_results]
        total_pnls = [r.total_pnl for r in asset_results]
        sharpe_ratios = [r.sharpe_ratio for r in asset_results]
        
        # 基础统计
        consistency_stats = {
            'win_rate': {
                'mean': np.mean(win_rates),
                'std': np.std(win_rates),
                'cv': np.std(win_rates) / np.mean(win_rates) if np.mean(win_rates) > 0 else float('inf'),
                'range': max(win_rates) - min(win_rates),
                'min': min(win_rates),
                'max': max(win_rates)
            },
            'total_pnl': {
                'mean': np.mean(total_pnls),
                'std': np.std(total_pnls),
                'cv': abs(np.std(total_pnls) / np.mean(total_pnls)) if np.mean(total_pnls) != 0 else float('inf'),
                'range': max(total_pnls) - min(total_pnls),
                'min': min(total_pnls),
                'max': max(total_pnls)
            },
            'sharpe_ratio': {
                'mean': np.mean(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'cv': abs(np.std(sharpe_ratios) / np.mean(sharpe_ratios)) if np.mean(sharpe_ratios) != 0 else float('inf'),
                'range': max(sharpe_ratios) - min(sharpe_ratios),
                'min': min(sharpe_ratios),
                'max': max(sharpe_ratios)
            }
        }
        
        # 一致性检验 (Friedman检验)
        if len(asset_results) >= 3:
            try:
                friedman_stat, friedman_p = stats.friedmanchisquare(
                    win_rates, total_pnls, sharpe_ratios
                )
                friedman_test = {
                    'statistic': friedman_stat,
                    'p_value': friedman_p,
                    'is_consistent': friedman_p > 0.05
                }
            except:
                friedman_test = {"error": "Friedman检验失败"}
        else:
            friedman_test = {"error": "样本不足"}
        
        # 相关性分析
        correlation_matrix = np.corrcoef([win_rates, total_pnls, sharpe_ratios])
        
        # 异常值检测
        outliers = self._detect_performance_outliers(asset_results)
        
        # 计算整体一致性得分
        overall_consistency = self._calculate_overall_consistency(consistency_stats)
        
        return {
            'consistency_stats': consistency_stats,
            'friedman_test': friedman_test,
            'correlation_matrix': correlation_matrix.tolist(),
            'outliers': outliers,
            'overall_consistency_score': overall_consistency,
            'asset_count': len(asset_results)
        }
    
    def _detect_selection_bias(self, asset_results: List[AssetValidationResult]) -> float:
        """检测选择偏差"""
        if len(asset_results) < 3:
            return 0
        
        # 计算性能分布偏斜度
        total_pnls = [r.total_pnl for r in asset_results]
        win_rates = [r.win_rate for r in asset_results]
        
        # 偏斜度检测
        pnl_skewness = stats.skew(total_pnls)
        winrate_skewness = stats.skew(win_rates)
        
        # 极值占比
        positive_count = sum(1 for pnl in total_pnls if pnl > 0)
        positive_ratio = positive_count / len(total_pnls)
        
        # 计算选择偏差得分 (0-100)
        bias_score = 0
        
        # 偏斜度惩罚
        bias_score += min(abs(pnl_skewness) * 20, 30)
        bias_score += min(abs(winrate_skewness) * 20, 30)
        
        # 极值比例惩罚
        if positive_ratio > 0.8 or positive_ratio < 0.2:
            bias_score += 25
        
        # 性能差异过大惩罚
        pnl_cv = np.std(total_pnls) / abs(np.mean(total_pnls)) if np.mean(total_pnls) != 0 else 0
        if pnl_cv > 1.0:
            bias_score += min(pnl_cv * 15, 25)
        
        return min(bias_score, 100)
    
    def _assess_overall_stability(self, asset_results: List[AssetValidationResult]) -> float:
        """评估整体稳定性"""
        if not asset_results:
            return 0
        
        # 一致性得分
        consistency_scores = [r.consistency_score for r in asset_results]
        avg_consistency = np.mean(consistency_scores)
        
        # 性能稳定性
        win_rates = [r.win_rate for r in asset_results]
        win_rate_stability = 1 / (1 + np.std(win_rates)) if len(win_rates) > 1 else 1
        
        # 盈利一致性
        positive_count = sum(1 for r in asset_results if r.total_pnl > 0)
        profit_consistency = positive_count / len(asset_results)
        
        # 综合稳定性得分
        overall_stability = (avg_consistency + win_rate_stability + profit_consistency) / 3
        return overall_stability
    
    def _recommend_asset_selection(self, 
                                 asset_results: List[AssetValidationResult],
                                 consistency_metrics: Dict) -> Tuple[List[str], List[str]]:
        """推荐资产选择"""
        if not asset_results:
            return [], []
        
        recommended = []
        rejected = []
        
        # 基于一致性得分和性能筛选
        overall_consistency = consistency_metrics.get('overall_consistency_score', 0)
        
        for result in asset_results:
            # 筛选条件
            conditions = [
                result.consistency_score >= 0.5,  # 一致性要求
                result.win_rate >= 0.45,         # 最低胜率
                result.total_trades >= 10,       # 最少交易数
                result.total_pnl > -1000         # 最大亏损限制
            ]
            
            # 如果整体一致性差，提高标准
            if overall_consistency < 0.6:
                conditions.extend([
                    result.win_rate >= 0.5,
                    result.consistency_score >= 0.6
                ])
            
            if all(conditions):
                recommended.append(result.symbol)
            else:
                rejected.append(result.symbol)
        
        return recommended, rejected
    
    def _generate_warnings(self, 
                         asset_results: List[AssetValidationResult],
                         consistency_metrics: Dict,
                         selection_bias_score: float) -> List[str]:
        """生成警告信息"""
        warnings = []
        
        # 资产数量警告
        if len(asset_results) < 5:
            warnings.append(f"测试资产数量不足: {len(asset_results)} < 5")
        
        # 一致性警告
        overall_consistency = consistency_metrics.get('overall_consistency_score', 0)
        if overall_consistency < 0.4:
            warnings.append(f"跨资产一致性差: {overall_consistency:.3f}")
        
        # 选择偏差警告
        if selection_bias_score > 50:
            warnings.append(f"存在严重选择偏差: {selection_bias_score:.1f}/100")
        
        # 性能分布警告
        win_rates = [r.win_rate for r in asset_results]
        if max(win_rates) - min(win_rates) > 0.3:
            warnings.append("资产间胜率差异过大")
        
        # 异常值警告
        outliers = consistency_metrics.get('outliers', [])
        if outliers:
            warnings.append(f"发现性能异常资产: {outliers}")
        
        return warnings
    
    def _detect_performance_outliers(self, asset_results: List[AssetValidationResult]) -> List[str]:
        """检测性能异常值"""
        if len(asset_results) < 3:
            return []
        
        outliers = []
        
        # 提取性能指标
        symbols = [r.symbol for r in asset_results]
        total_pnls = [r.total_pnl for r in asset_results]
        win_rates = [r.win_rate for r in asset_results]
        
        # 使用IQR方法检测异常值
        def detect_outliers_iqr(values, labels):
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            return [labels[i] for i, v in enumerate(values) 
                    if v < lower_bound or v > upper_bound]
        
        # 检测PnL异常值
        pnl_outliers = detect_outliers_iqr(total_pnls, symbols)
        outliers.extend(pnl_outliers)
        
        # 检测胜率异常值
        winrate_outliers = detect_outliers_iqr(win_rates, symbols)
        outliers.extend(winrate_outliers)
        
        return list(set(outliers))  # 去重
    
    def _calculate_overall_consistency(self, consistency_stats: Dict) -> float:
        """计算整体一致性得分"""
        # 基于变异系数计算一致性
        win_rate_cv = consistency_stats['win_rate']['cv']
        pnl_cv = consistency_stats['total_pnl']['cv']
        sharpe_cv = consistency_stats['sharpe_ratio']['cv']
        
        # 避免无穷大
        win_rate_cv = min(win_rate_cv, 10) if not np.isinf(win_rate_cv) else 10
        pnl_cv = min(pnl_cv, 10) if not np.isinf(pnl_cv) else 10
        sharpe_cv = min(sharpe_cv, 10) if not np.isinf(sharpe_cv) else 10
        
        # 一致性得分 (变异系数越小，一致性越高)
        consistency_score = 1 / (1 + (win_rate_cv + pnl_cv + sharpe_cv) / 3)
        return consistency_score
    
    def _generate_visualization_report(self, analysis: MultiAssetAnalysis) -> None:
        """生成可视化报告"""
        try:
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # 提取数据
            symbols = [r.symbol for r in analysis.asset_results]
            win_rates = [r.win_rate for r in analysis.asset_results]
            total_pnls = [r.total_pnl for r in analysis.asset_results]
            sharpe_ratios = [r.sharpe_ratio for r in analysis.asset_results]
            consistency_scores = [r.consistency_score for r in analysis.asset_results]
            
            # 1. 胜率对比
            axes[0, 0].bar(symbols, win_rates, color='skyblue')
            axes[0, 0].set_title('各资产胜率对比')
            axes[0, 0].set_ylabel('胜率')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            
            # 2. 总收益对比
            colors = ['green' if pnl > 0 else 'red' for pnl in total_pnls]
            axes[0, 1].bar(symbols, total_pnls, color=colors)
            axes[0, 1].set_title('各资产总收益对比')
            axes[0, 1].set_ylabel('总收益 (USD)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # 3. 夏普比率对比
            axes[1, 0].bar(symbols, sharpe_ratios, color='orange')
            axes[1, 0].set_title('各资产夏普比率对比')
            axes[1, 0].set_ylabel('夏普比率')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7)
            
            # 4. 一致性得分
            axes[1, 1].bar(symbols, consistency_scores, color='purple')
            axes[1, 1].set_title('各资产一致性得分')
            axes[1, 1].set_ylabel('一致性得分')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].axhline(y=0.6, color='red', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            
            # 保存图表
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            chart_path = self.results_dir / f"multi_asset_analysis_{timestamp}.png"
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"可视化报告已保存: {chart_path}")
            
        except Exception as e:
            logger.error(f"生成可视化报告失败: {e}")
    
    def _save_validation_results(self, analysis: MultiAssetAnalysis) -> None:
        """保存验证结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 转换为可序列化格式
        results_dict = {
            'timestamp': timestamp,
            'asset_results': [
                {
                    'symbol': r.symbol,
                    'total_trades': r.total_trades,
                    'win_rate': r.win_rate,
                    'total_pnl': r.total_pnl,
                    'sharpe_ratio': r.sharpe_ratio,
                    'max_drawdown': r.max_drawdown,
                    'avg_trade_duration': r.avg_trade_duration,
                    'consistency_score': r.consistency_score,
                    'performance_rank': r.performance_rank
                }
                for r in analysis.asset_results
            ],
            'consistency_metrics': analysis.consistency_metrics,
            'selection_bias_score': analysis.selection_bias_score,
            'overall_stability': analysis.overall_stability,
            'recommended_assets': analysis.recommended_assets,
            'rejected_assets': analysis.rejected_assets,
            'warnings': analysis.warnings
        }
        
        # 保存详细结果
        results_file = self.results_dir / f"multi_asset_validation_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # 保存摘要
        summary = {
            'timestamp': timestamp,
            'total_assets_tested': len(analysis.asset_results),
            'recommended_assets_count': len(analysis.recommended_assets),
            'rejected_assets_count': len(analysis.rejected_assets),
            'overall_stability': analysis.overall_stability,
            'selection_bias_score': analysis.selection_bias_score,
            'major_warnings': analysis.warnings[:3],  # 前3个警告
            'validation_passed': analysis.overall_stability > 0.6 and analysis.selection_bias_score < 50
        }
        
        summary_file = self.results_dir / f"multi_asset_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"多资产验证结果已保存: {results_file}")
        logger.info(f"验证摘要已保存: {summary_file}")