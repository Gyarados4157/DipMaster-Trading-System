#!/usr/bin/env python3
"""
统计验证器 - 严格的策略统计检验
Statistical Validator - Rigorous Strategy Statistical Testing

核心功能:
1. 蒙特卡洛随机化测试
2. 多重假设检验校正
3. 统计显著性验证
4. 时间稳定性分析

Author: DipMaster Trading Team
Date: 2025-08-15
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """验证结果"""
    test_name: str
    p_value: float
    is_significant: bool
    effect_size: float
    confidence_interval: Tuple[float, float]
    interpretation: str
    warning_level: str  # 'safe', 'caution', 'danger'

@dataclass
class MonteCarloResult:
    """蒙特卡洛测试结果"""
    original_metric: float
    random_mean: float
    random_std: float
    p_value: float
    percentile_rank: float
    is_significant: bool
    interpretation: str

class StatisticalValidator:
    """
    统计验证器
    
    提供严格的统计检验来验证策略的真实性能
    """
    
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        self.results_dir = Path("results/validation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def monte_carlo_randomization_test(self, 
                                     trades_df: pd.DataFrame,
                                     metric: str = 'pnl',
                                     n_simulations: int = 10000) -> MonteCarloResult:
        """
        蒙特卡洛随机化测试
        
        核心原理: 如果策略有真实预测能力，其表现应该显著优于随机交易
        
        Args:
            trades_df: 交易记录DataFrame
            metric: 测试指标 ('pnl', 'win_rate', 'sharpe')
            n_simulations: 模拟次数
            
        Returns:
            MonteCarloResult: 测试结果
        """
        logger.info(f"开始蒙特卡洛随机化测试 - 指标: {metric}")
        
        # 计算原始指标
        original_value = self._calculate_metric(trades_df, metric)
        
        # 随机化模拟
        random_values = []
        for i in range(n_simulations):
            if i % 1000 == 0:
                logger.info(f"蒙特卡洛进度: {i}/{n_simulations}")
            
            # 随机化交易结果
            randomized_df = self._randomize_trades(trades_df)
            random_value = self._calculate_metric(randomized_df, metric)
            random_values.append(random_value)
        
        random_values = np.array(random_values)
        
        # 计算统计量
        random_mean = np.mean(random_values)
        random_std = np.std(random_values)
        
        # 计算P值 (双尾检验)
        if original_value >= random_mean:
            p_value = np.mean(random_values >= original_value) * 2
        else:
            p_value = np.mean(random_values <= original_value) * 2
        
        p_value = min(p_value, 1.0)  # 确保P值不超过1
        
        # 计算百分位数
        percentile_rank = stats.percentileofscore(random_values, original_value)
        
        # 判断显著性
        is_significant = p_value < self.significance_level
        
        # 生成解释
        interpretation = self._interpret_monte_carlo(
            original_value, random_mean, p_value, is_significant, metric
        )
        
        result = MonteCarloResult(
            original_metric=original_value,
            random_mean=random_mean,
            random_std=random_std,
            p_value=p_value,
            percentile_rank=percentile_rank,
            is_significant=is_significant,
            interpretation=interpretation
        )
        
        logger.info(f"蒙特卡洛测试完成: P值={p_value:.4f}, 显著性={is_significant}")
        return result
    
    def multiple_testing_correction(self, p_values: List[float], 
                                  method: str = 'bonferroni') -> List[float]:
        """
        多重假设检验校正
        
        Args:
            p_values: P值列表
            method: 校正方法 ('bonferroni', 'holm', 'fdr_bh')
            
        Returns:
            List[float]: 校正后的P值
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)
        
        if method == 'bonferroni':
            # Bonferroni校正
            corrected_p = p_values * n_tests
            corrected_p = np.minimum(corrected_p, 1.0)
        
        elif method == 'holm':
            # Holm-Bonferroni校正
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            corrected_p = np.zeros_like(p_values)
            
            for i, idx in enumerate(sorted_indices):
                correction_factor = n_tests - i
                corrected_p[idx] = min(sorted_p[i] * correction_factor, 1.0)
        
        elif method == 'fdr_bh':
            # Benjamini-Hochberg FDR校正
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            corrected_p = np.zeros_like(p_values)
            
            for i, idx in enumerate(sorted_indices):
                correction_factor = n_tests / (i + 1)
                corrected_p[idx] = min(sorted_p[i] * correction_factor, 1.0)
        
        else:
            raise ValueError(f"不支持的校正方法: {method}")
        
        logger.info(f"多重检验校正完成: {method}, 原始P值数量: {n_tests}")
        return corrected_p.tolist()
    
    def time_stability_test(self, trades_df: pd.DataFrame, 
                          window_size: str = '30D') -> Dict:
        """
        时间稳定性测试
        
        Args:
            trades_df: 交易记录
            window_size: 滚动窗口大小
            
        Returns:
            Dict: 稳定性测试结果
        """
        logger.info(f"开始时间稳定性测试 - 窗口: {window_size}")
        
        # 确保时间戳列存在
        if 'timestamp' not in trades_df.columns:
            raise ValueError("交易数据缺少timestamp列")
        
        trades_df = trades_df.copy()
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.sort_values('timestamp')
        
        # 滚动窗口分析
        window_metrics = []
        
        # 设置时间范围
        start_date = trades_df['timestamp'].min()
        end_date = trades_df['timestamp'].max()
        
        # 滚动计算
        current_date = start_date
        while current_date <= end_date:
            window_end = current_date + pd.Timedelta(window_size)
            
            window_trades = trades_df[
                (trades_df['timestamp'] >= current_date) & 
                (trades_df['timestamp'] < window_end)
            ]
            
            if len(window_trades) >= 10:  # 最少10笔交易
                metrics = {
                    'period_start': current_date,
                    'period_end': window_end,
                    'trade_count': len(window_trades),
                    'win_rate': (window_trades['pnl'] > 0).mean(),
                    'total_pnl': window_trades['pnl'].sum(),
                    'avg_pnl': window_trades['pnl'].mean(),
                    'sharpe_ratio': self._calculate_sharpe(window_trades['pnl'])
                }
                window_metrics.append(metrics)
            
            current_date += pd.Timedelta(days=7)  # 每周滚动
        
        if not window_metrics:
            return {"error": "没有足够的数据进行时间稳定性分析"}
        
        # 转换为DataFrame
        metrics_df = pd.DataFrame(window_metrics)
        
        # 计算稳定性统计量
        stability_stats = {
            'win_rate': {
                'mean': metrics_df['win_rate'].mean(),
                'std': metrics_df['win_rate'].std(),
                'cv': metrics_df['win_rate'].std() / metrics_df['win_rate'].mean(),
                'min': metrics_df['win_rate'].min(),
                'max': metrics_df['win_rate'].max()
            },
            'pnl': {
                'mean': metrics_df['total_pnl'].mean(),
                'std': metrics_df['total_pnl'].std(),
                'cv': abs(metrics_df['total_pnl'].std() / metrics_df['total_pnl'].mean()),
                'min': metrics_df['total_pnl'].min(),
                'max': metrics_df['total_pnl'].max()
            },
            'sharpe': {
                'mean': metrics_df['sharpe_ratio'].mean(),
                'std': metrics_df['sharpe_ratio'].std(),
                'cv': abs(metrics_df['sharpe_ratio'].std() / metrics_df['sharpe_ratio'].mean()),
                'min': metrics_df['sharpe_ratio'].min(),
                'max': metrics_df['sharpe_ratio'].max()
            }
        }
        
        # 趋势测试 (Mann-Kendall)
        trend_test = self._mann_kendall_trend_test(metrics_df['win_rate'])
        
        # 评估稳定性
        stability_score = self._calculate_stability_score(stability_stats)
        
        result = {
            'window_count': len(window_metrics),
            'stability_stats': stability_stats,
            'trend_test': trend_test,
            'stability_score': stability_score,
            'interpretation': self._interpret_stability(stability_score),
            'raw_metrics': window_metrics
        }
        
        logger.info(f"时间稳定性测试完成: 稳定性得分={stability_score:.2f}")
        return result
    
    def cross_asset_consistency_test(self, 
                                   results_by_symbol: Dict[str, pd.DataFrame]) -> Dict:
        """
        跨资产一致性测试
        
        Args:
            results_by_symbol: 各币种的交易结果
            
        Returns:
            Dict: 一致性测试结果
        """
        logger.info("开始跨资产一致性测试...")
        
        symbol_metrics = {}
        
        # 计算各币种指标
        for symbol, trades_df in results_by_symbol.items():
            if len(trades_df) == 0:
                continue
                
            metrics = {
                'win_rate': (trades_df['pnl'] > 0).mean(),
                'total_pnl': trades_df['pnl'].sum(),
                'avg_pnl': trades_df['pnl'].mean(),
                'sharpe_ratio': self._calculate_sharpe(trades_df['pnl']),
                'max_drawdown': self._calculate_max_drawdown(trades_df['pnl']),
                'trade_count': len(trades_df)
            }
            symbol_metrics[symbol] = metrics
        
        if len(symbol_metrics) < 2:
            return {"error": "需要至少2个币种进行一致性测试"}
        
        # 转换为DataFrame
        metrics_df = pd.DataFrame(symbol_metrics).T
        
        # 计算一致性统计量
        consistency_stats = {}
        for metric in ['win_rate', 'sharpe_ratio', 'avg_pnl']:
            values = metrics_df[metric].dropna()
            consistency_stats[metric] = {
                'mean': values.mean(),
                'std': values.std(),
                'cv': abs(values.std() / values.mean()) if values.mean() != 0 else float('inf'),
                'range': values.max() - values.min(),
                'iqr': values.quantile(0.75) - values.quantile(0.25)
            }
        
        # 一致性检验 (Friedman检验)
        available_metrics = [col for col in ['win_rate', 'sharpe_ratio', 'avg_pnl'] if col in metrics_df.columns]
        
        if len(available_metrics) >= 3:
            friedman_stat, friedman_p = stats.friedmanchisquare(
                *[metrics_df[col].values for col in available_metrics]
            )
        else:
            # 如果指标不足3个，使用简化的方差分析
            friedman_stat = 0.0
            friedman_p = 1.0
        
        # 计算一致性得分
        consistency_score = self._calculate_consistency_score(consistency_stats)
        
        result = {
            'symbol_count': len(symbol_metrics),
            'symbol_metrics': symbol_metrics,
            'consistency_stats': consistency_stats,
            'friedman_test': {
                'statistic': friedman_stat,
                'p_value': friedman_p,
                'is_significant': friedman_p < self.significance_level
            },
            'consistency_score': consistency_score,
            'interpretation': self._interpret_consistency(consistency_score)
        }
        
        logger.info(f"跨资产一致性测试完成: 一致性得分={consistency_score:.2f}")
        return result
    
    def comprehensive_validation(self, 
                               trades_df: pd.DataFrame,
                               results_by_symbol: Dict[str, pd.DataFrame] = None) -> Dict:
        """
        综合验证测试
        
        Args:
            trades_df: 总体交易记录
            results_by_symbol: 各币种交易记录
            
        Returns:
            Dict: 综合验证结果
        """
        logger.info("开始综合策略验证...")
        
        validation_results = {}
        
        # 1. 蒙特卡洛测试
        mc_pnl = self.monte_carlo_randomization_test(trades_df, 'pnl')
        mc_winrate = self.monte_carlo_randomization_test(trades_df, 'win_rate')
        
        validation_results['monte_carlo'] = {
            'pnl_test': mc_pnl,
            'win_rate_test': mc_winrate
        }
        
        # 2. 时间稳定性测试
        stability_result = self.time_stability_test(trades_df)
        validation_results['time_stability'] = stability_result
        
        # 3. 跨资产一致性测试
        if results_by_symbol:
            consistency_result = self.cross_asset_consistency_test(results_by_symbol)
            validation_results['cross_asset_consistency'] = consistency_result
        
        # 4. 多重检验校正
        all_p_values = [
            mc_pnl.p_value,
            mc_winrate.p_value
        ]
        
        corrected_p_values = self.multiple_testing_correction(all_p_values)
        validation_results['corrected_p_values'] = {
            'original': all_p_values,
            'bonferroni_corrected': corrected_p_values
        }
        
        # 5. 综合评估
        overall_assessment = self._generate_overall_assessment(validation_results)
        validation_results['overall_assessment'] = overall_assessment
        
        # 保存结果
        self._save_validation_results(validation_results)
        
        logger.info("综合策略验证完成")
        return validation_results
    
    def _calculate_metric(self, trades_df: pd.DataFrame, metric: str) -> float:
        """计算指标值"""
        if metric == 'pnl':
            return trades_df['pnl'].sum()
        elif metric == 'win_rate':
            return (trades_df['pnl'] > 0).mean()
        elif metric == 'sharpe':
            return self._calculate_sharpe(trades_df['pnl'])
        else:
            raise ValueError(f"不支持的指标: {metric}")
    
    def _randomize_trades(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """
        随机化交易结果 - 生成真正的随机交易
        
        模拟随机交易的真实分布，而不是简单重排现有结果
        """
        randomized_df = trades_df.copy()
        
        # 基于历史收益率分布生成真正的随机交易
        n_trades = len(trades_df)
        
        # 估算合理的随机交易参数 (基于市场统计)
        # 加密货币5分钟K线收益率大约：均值=0, 标准差=0.01-0.02
        typical_return_std = 0.015  # 1.5%标准差
        
        # 生成符合市场特征的随机收益率
        random_returns = np.random.normal(0, typical_return_std, n_trades)
        
        # 转换为PnL (假设固定仓位大小1000)
        position_size = 1000
        randomized_df['pnl'] = random_returns * position_size
        
        return randomized_df
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """计算夏普比率"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252 * 24 * 12)  # 5分钟数据年化
    
    def _calculate_max_drawdown(self, pnl_series: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = pnl_series.cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        return drawdown.min()
    
    def _mann_kendall_trend_test(self, data: pd.Series) -> Dict:
        """Mann-Kendall趋势检验"""
        n = len(data)
        if n < 3:
            return {"error": "数据量不足"}
        
        # 计算S统计量
        s = 0
        for i in range(n-1):
            for j in range(i+1, n):
                s += np.sign(data.iloc[j] - data.iloc[i])
        
        # 计算方差
        var_s = n * (n-1) * (2*n+5) / 18
        
        # 计算Z统计量
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # 计算P值
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        return {
            'statistic': s,
            'z_score': z,
            'p_value': p_value,
            'has_trend': p_value < self.significance_level,
            'trend_direction': 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no_trend'
        }
    
    def _calculate_stability_score(self, stability_stats: Dict) -> float:
        """计算稳定性得分 (0-100)"""
        # 基于变异系数计算稳定性
        cv_winrate = stability_stats['win_rate']['cv']
        cv_pnl = stability_stats['pnl']['cv']
        cv_sharpe = stability_stats['sharpe']['cv']
        
        # 变异系数越小，稳定性越高
        stability_score = 100 / (1 + cv_winrate + cv_pnl + cv_sharpe)
        return min(max(stability_score, 0), 100)
    
    def _calculate_consistency_score(self, consistency_stats: Dict) -> float:
        """计算一致性得分 (0-100)"""
        avg_cv = np.mean([stats['cv'] for stats in consistency_stats.values() 
                         if not np.isinf(stats['cv'])])
        consistency_score = 100 / (1 + avg_cv)
        return min(max(consistency_score, 0), 100)
    
    def _interpret_monte_carlo(self, original: float, random_mean: float, 
                              p_value: float, is_significant: bool, metric: str) -> str:
        """解释蒙特卡洛结果"""
        if p_value < 0.001:
            significance_text = "极其显著"
        elif p_value < 0.01:
            significance_text = "高度显著"
        elif p_value < 0.05:
            significance_text = "显著"
        else:
            significance_text = "不显著"
        
        if is_significant:
            if original > random_mean:
                return f"策略在{metric}上表现{significance_text}优于随机交易 (P={p_value:.4f})"
            else:
                return f"策略在{metric}上表现{significance_text}劣于随机交易 (P={p_value:.4f})"
        else:
            return f"策略在{metric}上的表现与随机交易无显著差异 (P={p_value:.4f}) - 疑似过拟合"
    
    def _interpret_stability(self, score: float) -> str:
        """解释稳定性得分"""
        if score >= 80:
            return "策略表现极其稳定"
        elif score >= 60:
            return "策略表现较为稳定"
        elif score >= 40:
            return "策略表现中等稳定"
        elif score >= 20:
            return "策略表现不太稳定"
        else:
            return "策略表现极不稳定 - 高过拟合风险"
    
    def _interpret_consistency(self, score: float) -> str:
        """解释一致性得分"""
        if score >= 80:
            return "跨资产表现高度一致"
        elif score >= 60:
            return "跨资产表现较为一致"
        elif score >= 40:
            return "跨资产表现中等一致"
        elif score >= 20:
            return "跨资产表现不太一致"
        else:
            return "跨资产表现极不一致 - 选择偏差风险"
    
    def _generate_overall_assessment(self, results: Dict) -> Dict:
        """生成综合评估"""
        warnings = []
        risk_level = "LOW"
        
        # 检查蒙特卡洛结果
        mc_results = results['monte_carlo']
        if not mc_results['pnl_test'].is_significant:
            warnings.append("PnL表现与随机交易无显著差异")
            risk_level = "HIGH"
        
        if not mc_results['win_rate_test'].is_significant:
            warnings.append("胜率与随机交易无显著差异")
            risk_level = "HIGH"
        
        # 检查稳定性
        if 'time_stability' in results:
            stability_score = results['time_stability'].get('stability_score', 0)
            if stability_score < 40:
                warnings.append("时间稳定性差")
                risk_level = "MEDIUM" if risk_level == "LOW" else "HIGH"
        
        # 检查一致性
        if 'cross_asset_consistency' in results:
            consistency_score = results['cross_asset_consistency'].get('consistency_score', 0)
            if consistency_score < 40:
                warnings.append("跨资产一致性差")
                risk_level = "MEDIUM" if risk_level == "LOW" else "HIGH"
        
        # 生成建议
        if risk_level == "HIGH":
            recommendation = "🚨 严重过拟合风险 - 禁止实盘交易"
        elif risk_level == "MEDIUM":
            recommendation = "⚠️  中等过拟合风险 - 需要进一步验证"
        else:
            recommendation = "✅ 低过拟合风险 - 可考虑谨慎实盘"
        
        return {
            'risk_level': risk_level,
            'warnings': warnings,
            'recommendation': recommendation,
            'overall_pass': risk_level == "LOW"
        }
    
    def _save_validation_results(self, results: Dict) -> None:
        """保存验证结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"statistical_validation_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # 转换特殊对象为可序列化格式
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"验证结果已保存: {filepath}")
    
    def _make_serializable(self, obj):
        """转换对象为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (MonteCarloResult, ValidationResult)):
            return obj.__dict__
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj