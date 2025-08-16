#!/usr/bin/env python3
"""
Walk-Forward分析器 - 时间稳定性验证
Walk-Forward Analyzer - Time Stability Validation

核心功能:
1. 滚动窗口验证
2. 参数稳定性测试
3. 时间序列交叉验证
4. 前向性能衰减分析

Author: DipMaster Trading Team
Date: 2025-08-15
Version: 1.0.0
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable
import logging
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class WalkForwardConfig:
    """Walk-Forward配置"""
    train_window_months: int = 6      # 训练窗口长度
    test_window_months: int = 1       # 测试窗口长度
    step_size_months: int = 1         # 步进大小
    min_trades_per_window: int = 50   # 每窗口最少交易数
    refit_parameters: bool = True     # 是否重新拟合参数

@dataclass
class WindowResult:
    """单个窗口结果"""
    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_trades: int
    test_trades: int
    test_metrics: Dict
    optimal_parameters: Dict
    parameter_stability: float

class WalkForwardAnalyzer:
    """
    Walk-Forward分析器
    
    实现严格的时间序列交叉验证，验证策略的时间稳定性
    """
    
    def __init__(self, config: WalkForwardConfig = None):
        self.config = config or WalkForwardConfig()
        self.results_dir = Path("results/walk_forward")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def run_walk_forward_analysis(self, 
                                 data: pd.DataFrame,
                                 strategy_func: Callable,
                                 parameter_ranges: Dict,
                                 symbols: List[str] = None) -> Dict:
        """
        运行Walk-Forward分析
        
        Args:
            data: 市场数据
            strategy_func: 策略函数
            parameter_ranges: 参数搜索范围
            symbols: 币种列表
            
        Returns:
            Dict: 分析结果
        """
        logger.info("开始Walk-Forward分析...")
        
        # 准备数据
        data = data.copy()
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data = data.sort_values('timestamp')
        
        # 生成时间窗口
        windows = self._generate_time_windows(data)
        logger.info(f"生成了 {len(windows)} 个时间窗口")
        
        # 逐窗口分析
        window_results = []
        baseline_parameters = None
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"处理窗口 {i+1}/{len(windows)}: {test_start.date()} - {test_end.date()}")
            
            # 分割数据
            train_data = data[(data['timestamp'] >= train_start) & 
                            (data['timestamp'] <= train_end)]
            test_data = data[(data['timestamp'] >= test_start) & 
                           (data['timestamp'] <= test_end)]
            
            # 检查数据充足性
            if len(train_data) < 1000 or len(test_data) < 100:
                logger.warning(f"窗口 {i+1} 数据不足，跳过")
                continue
            
            try:
                # 参数优化
                if self.config.refit_parameters:
                    optimal_params = self._optimize_parameters(
                        train_data, strategy_func, parameter_ranges
                    )
                else:
                    optimal_params = baseline_parameters or self._get_default_parameters()
                
                if baseline_parameters is None:
                    baseline_parameters = optimal_params
                
                # 样本外测试
                test_results = self._run_strategy_test(test_data, strategy_func, optimal_params)
                
                # 计算参数稳定性
                param_stability = self._calculate_parameter_stability(
                    optimal_params, baseline_parameters
                )
                
                # 创建窗口结果
                window_result = WindowResult(
                    window_id=i+1,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_trades=len(train_data),
                    test_trades=len(test_data),
                    test_metrics=test_results,
                    optimal_parameters=optimal_params,
                    parameter_stability=param_stability
                )
                
                window_results.append(window_result)
                
            except Exception as e:
                logger.error(f"窗口 {i+1} 处理失败: {e}")
                continue
        
        if not window_results:
            raise ValueError("没有成功的窗口结果")
        
        # 分析结果
        analysis_results = self._analyze_walk_forward_results(window_results)
        
        # 保存结果
        self._save_walk_forward_results(window_results, analysis_results)
        
        logger.info("Walk-Forward分析完成")
        return analysis_results
    
    def _generate_time_windows(self, data: pd.DataFrame) -> List[Tuple]:
        """生成时间窗口"""
        windows = []
        
        start_date = data['timestamp'].min()
        end_date = data['timestamp'].max()
        
        current_date = start_date
        
        while current_date < end_date:
            # 训练窗口
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=self.config.train_window_months)
            
            # 测试窗口
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.DateOffset(months=self.config.test_window_months)
            
            # 检查是否超出数据范围
            if test_end > end_date:
                break
            
            windows.append((train_start, train_end, test_start, test_end))
            
            # 步进
            current_date += pd.DateOffset(months=self.config.step_size_months)
        
        return windows
    
    def _optimize_parameters(self, 
                           train_data: pd.DataFrame, 
                           strategy_func: Callable,
                           parameter_ranges: Dict) -> Dict:
        """
        参数优化
        
        使用网格搜索在训练数据上优化参数
        """
        best_params = {}
        best_score = -np.inf
        
        # 生成参数组合
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        logger.info(f"测试 {len(param_combinations)} 个参数组合")
        
        for i, params in enumerate(param_combinations):
            if i % 10 == 0:
                logger.info(f"参数优化进度: {i}/{len(param_combinations)}")
            
            try:
                # 运行策略
                results = self._run_strategy_test(train_data, strategy_func, params)
                
                # 计算优化目标 (这里使用风险调整收益)
                if results['trade_count'] < self.config.min_trades_per_window:
                    continue
                
                score = self._calculate_optimization_score(results)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    
            except Exception as e:
                logger.debug(f"参数组合失败: {params}, 错误: {e}")
                continue
        
        if not best_params:
            logger.warning("未找到有效参数，使用默认参数")
            return self._get_default_parameters()
        
        logger.info(f"最优参数: {best_params}, 得分: {best_score:.4f}")
        return best_params
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict) -> List[Dict]:
        """生成参数组合"""
        import itertools
        
        keys = list(parameter_ranges.keys())
        values = list(parameter_ranges.values())
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def _run_strategy_test(self, 
                          data: pd.DataFrame, 
                          strategy_func: Callable, 
                          parameters: Dict) -> Dict:
        """
        运行策略测试
        
        这里需要根据实际策略函数实现
        """
        # 这是一个模拟实现，实际需要调用真实的策略函数
        
        # 模拟交易结果
        np.random.seed(42)  # 为了可重复性
        
        trade_count = len(data) // 100  # 假设每100个数据点一笔交易
        trade_count = max(trade_count, 1)
        
        # 生成模拟PnL
        win_rate = parameters.get('expected_win_rate', 0.55)
        avg_win = parameters.get('avg_win', 10)
        avg_loss = parameters.get('avg_loss', -8)
        
        pnl_list = []
        for _ in range(trade_count):
            if np.random.random() < win_rate:
                pnl = np.random.normal(avg_win, avg_win * 0.3)
            else:
                pnl = np.random.normal(avg_loss, abs(avg_loss) * 0.3)
            pnl_list.append(pnl)
        
        pnl_series = pd.Series(pnl_list)
        
        # 计算指标
        total_pnl = pnl_series.sum()
        actual_win_rate = (pnl_series > 0).mean()
        sharpe_ratio = pnl_series.mean() / pnl_series.std() if pnl_series.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown(pnl_series)
        
        return {
            'total_pnl': total_pnl,
            'win_rate': actual_win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trade_count': trade_count,
            'avg_pnl_per_trade': total_pnl / trade_count if trade_count > 0 else 0
        }
    
    def _calculate_optimization_score(self, results: Dict) -> float:
        """计算优化目标得分"""
        # 风险调整收益
        pnl = results.get('total_pnl', 0)
        max_dd = abs(results.get('max_drawdown', 1))
        win_rate = results.get('win_rate', 0)
        trade_count = results.get('trade_count', 0)
        
        # 避免除零
        if max_dd == 0:
            max_dd = 0.01
        
        # 综合得分
        score = (pnl / max_dd) * win_rate * np.log(1 + trade_count)
        return score
    
    def _calculate_max_drawdown(self, pnl_series: pd.Series) -> float:
        """计算最大回撤"""
        cumulative = pnl_series.cumsum()
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        return drawdown.min()
    
    def _calculate_parameter_stability(self, 
                                     current_params: Dict, 
                                     baseline_params: Dict) -> float:
        """计算参数稳定性"""
        if not baseline_params:
            return 1.0
        
        stability_scores = []
        
        for key in current_params:
            if key in baseline_params:
                current_val = current_params[key]
                baseline_val = baseline_params[key]
                
                if isinstance(current_val, (int, float)) and isinstance(baseline_val, (int, float)):
                    if baseline_val != 0:
                        relative_change = abs(current_val - baseline_val) / abs(baseline_val)
                        stability = 1 / (1 + relative_change)
                        stability_scores.append(stability)
                    else:
                        stability_scores.append(1.0 if current_val == baseline_val else 0.0)
                else:
                    stability_scores.append(1.0 if current_val == baseline_val else 0.0)
        
        return np.mean(stability_scores) if stability_scores else 1.0
    
    def _get_default_parameters(self) -> Dict:
        """获取默认参数"""
        return {
            'rsi_lower': 30,
            'rsi_upper': 70,
            'ma_period': 20,
            'volume_threshold': 1.5,
            'expected_win_rate': 0.55,
            'avg_win': 10,
            'avg_loss': -8
        }
    
    def _analyze_walk_forward_results(self, window_results: List[WindowResult]) -> Dict:
        """分析Walk-Forward结果"""
        logger.info("分析Walk-Forward结果...")
        
        # 提取指标
        metrics_data = []
        for result in window_results:
            metrics = result.test_metrics.copy()
            metrics['window_id'] = result.window_id
            metrics['test_start'] = result.test_start
            metrics['parameter_stability'] = result.parameter_stability
            metrics_data.append(metrics)
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # 时间序列分析
        time_analysis = self._analyze_time_series_performance(metrics_df)
        
        # 参数稳定性分析
        parameter_analysis = self._analyze_parameter_stability(window_results)
        
        # 性能衰减分析
        performance_decay = self._analyze_performance_decay(metrics_df)
        
        # 稳定性评估
        stability_assessment = self._assess_overall_stability(metrics_df)
        
        # 综合评估
        overall_assessment = {
            'total_windows': len(window_results),
            'avg_win_rate': metrics_df['win_rate'].mean(),
            'win_rate_std': metrics_df['win_rate'].std(),
            'avg_sharpe': metrics_df['sharpe_ratio'].mean(),
            'sharpe_std': metrics_df['sharpe_ratio'].std(),
            'parameter_stability_avg': metrics_df['parameter_stability'].mean(),
            'performance_consistency': self._calculate_performance_consistency(metrics_df),
            'time_stability_score': time_analysis.get('stability_score', 0),
            'overall_pass': stability_assessment['overall_pass']
        }
        
        return {
            'window_results': [self._window_result_to_dict(wr) for wr in window_results],
            'time_analysis': time_analysis,
            'parameter_analysis': parameter_analysis,
            'performance_decay': performance_decay,
            'stability_assessment': stability_assessment,
            'overall_assessment': overall_assessment,
            'recommendations': self._generate_recommendations(overall_assessment)
        }
    
    def _analyze_time_series_performance(self, metrics_df: pd.DataFrame) -> Dict:
        """时间序列性能分析"""
        
        # 趋势分析
        from scipy import stats
        
        window_numbers = metrics_df['window_id'].values
        win_rates = metrics_df['win_rate'].values
        sharpe_ratios = metrics_df['sharpe_ratio'].values
        
        # 线性回归检测趋势
        win_rate_slope, win_rate_intercept, win_rate_r, win_rate_p, _ = stats.linregress(window_numbers, win_rates)
        sharpe_slope, sharpe_intercept, sharpe_r, sharpe_p, _ = stats.linregress(window_numbers, sharpe_ratios)
        
        # 稳定性得分
        win_rate_cv = metrics_df['win_rate'].std() / metrics_df['win_rate'].mean()
        sharpe_cv = abs(metrics_df['sharpe_ratio'].std() / metrics_df['sharpe_ratio'].mean())
        
        stability_score = 100 / (1 + win_rate_cv + sharpe_cv)
        
        return {
            'win_rate_trend': {
                'slope': win_rate_slope,
                'p_value': win_rate_p,
                'has_significant_trend': win_rate_p < 0.05,
                'direction': 'declining' if win_rate_slope < 0 else 'improving'
            },
            'sharpe_trend': {
                'slope': sharpe_slope,
                'p_value': sharpe_p,
                'has_significant_trend': sharpe_p < 0.05,
                'direction': 'declining' if sharpe_slope < 0 else 'improving'
            },
            'stability_score': stability_score,
            'win_rate_cv': win_rate_cv,
            'sharpe_cv': sharpe_cv
        }
    
    def _analyze_parameter_stability(self, window_results: List[WindowResult]) -> Dict:
        """参数稳定性分析"""
        
        if len(window_results) < 2:
            return {"error": "需要至少2个窗口进行参数稳定性分析"}
        
        # 收集所有参数
        all_parameters = {}
        for result in window_results:
            for param_name, param_value in result.optimal_parameters.items():
                if param_name not in all_parameters:
                    all_parameters[param_name] = []
                all_parameters[param_name].append(param_value)
        
        # 分析每个参数的稳定性
        parameter_stability = {}
        for param_name, values in all_parameters.items():
            if all(isinstance(v, (int, float)) for v in values):
                param_stability = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else float('inf'),
                    'range': max(values) - min(values),
                    'stability_score': 100 / (1 + np.std(values) / (abs(np.mean(values)) + 0.001))
                }
            else:
                # 分类参数
                unique_values = len(set(values))
                param_stability = {
                    'unique_count': unique_values,
                    'most_common': max(set(values), key=values.count),
                    'stability_score': 100 * (values.count(max(set(values), key=values.count)) / len(values))
                }
            
            parameter_stability[param_name] = param_stability
        
        # 整体参数稳定性
        overall_stability = np.mean([
            result.parameter_stability for result in window_results
        ])
        
        return {
            'parameter_details': parameter_stability,
            'overall_parameter_stability': overall_stability,
            'parameter_drift_detected': overall_stability < 0.8
        }
    
    def _analyze_performance_decay(self, metrics_df: pd.DataFrame) -> Dict:
        """性能衰减分析"""
        
        # 计算滚动性能
        window_size = min(5, len(metrics_df) // 2)
        if window_size < 2:
            return {"error": "数据不足进行性能衰减分析"}
        
        rolling_win_rate = metrics_df['win_rate'].rolling(window_size).mean()
        rolling_sharpe = metrics_df['sharpe_ratio'].rolling(window_size).mean()
        
        # 检测衰减
        recent_performance = rolling_win_rate.iloc[-3:].mean()
        early_performance = rolling_win_rate.iloc[:3].mean()
        
        win_rate_decay = (early_performance - recent_performance) / early_performance if early_performance != 0 else 0
        
        recent_sharpe = rolling_sharpe.iloc[-3:].mean()
        early_sharpe = rolling_sharpe.iloc[:3].mean()
        
        sharpe_decay = (early_sharpe - recent_sharpe) / abs(early_sharpe) if early_sharpe != 0 else 0
        
        return {
            'win_rate_decay_pct': win_rate_decay * 100,
            'sharpe_decay_pct': sharpe_decay * 100,
            'significant_decay_detected': win_rate_decay > 0.1 or sharpe_decay > 0.1,
            'early_vs_recent': {
                'early_win_rate': early_performance,
                'recent_win_rate': recent_performance,
                'early_sharpe': early_sharpe,
                'recent_sharpe': recent_sharpe
            }
        }
    
    def _assess_overall_stability(self, metrics_df: pd.DataFrame) -> Dict:
        """整体稳定性评估"""
        
        # 稳定性指标
        win_rate_cv = metrics_df['win_rate'].std() / metrics_df['win_rate'].mean()
        sharpe_cv = abs(metrics_df['sharpe_ratio'].std() / metrics_df['sharpe_ratio'].mean())
        
        # 一致性检查
        positive_windows = (metrics_df['total_pnl'] > 0).sum()
        consistency_ratio = positive_windows / len(metrics_df)
        
        # 综合评估
        stability_factors = {
            'win_rate_stability': 1 / (1 + win_rate_cv),
            'sharpe_stability': 1 / (1 + sharpe_cv),
            'consistency_ratio': consistency_ratio,
            'parameter_stability': metrics_df['parameter_stability'].mean()
        }
        
        overall_stability = np.mean(list(stability_factors.values()))
        
        # 判断是否通过
        overall_pass = (
            overall_stability > 0.7 and
            consistency_ratio > 0.6 and
            win_rate_cv < 0.3 and
            sharpe_cv < 0.5
        )
        
        return {
            'stability_factors': stability_factors,
            'overall_stability_score': overall_stability,
            'overall_pass': overall_pass,
            'warnings': self._generate_stability_warnings(stability_factors)
        }
    
    def _calculate_performance_consistency(self, metrics_df: pd.DataFrame) -> float:
        """计算性能一致性"""
        win_rate_consistency = 1 - (metrics_df['win_rate'].std() / metrics_df['win_rate'].mean())
        pnl_consistency = 1 - abs(metrics_df['total_pnl'].std() / metrics_df['total_pnl'].mean())
        
        return np.mean([win_rate_consistency, pnl_consistency])
    
    def _generate_stability_warnings(self, stability_factors: Dict) -> List[str]:
        """生成稳定性警告"""
        warnings = []
        
        if stability_factors['win_rate_stability'] < 0.7:
            warnings.append("胜率稳定性差")
        
        if stability_factors['sharpe_stability'] < 0.7:
            warnings.append("夏普比率稳定性差")
        
        if stability_factors['consistency_ratio'] < 0.6:
            warnings.append("盈利一致性差")
        
        if stability_factors['parameter_stability'] < 0.8:
            warnings.append("参数稳定性差，存在过拟合风险")
        
        return warnings
    
    def _generate_recommendations(self, assessment: Dict) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if not assessment['overall_pass']:
            recommendations.append("🚨 策略未通过Walk-Forward验证，不建议实盘交易")
        
        if assessment['parameter_stability_avg'] < 0.8:
            recommendations.append("⚠️ 参数不稳定，建议简化策略逻辑")
        
        if assessment['performance_consistency'] < 0.6:
            recommendations.append("⚠️ 性能一致性差，建议增加数据量或改进策略")
        
        if assessment['time_stability_score'] < 60:
            recommendations.append("⚠️ 时间稳定性差，策略可能过拟合")
        
        if assessment['overall_pass']:
            recommendations.append("✅ 策略通过Walk-Forward验证，表现稳定")
        
        return recommendations
    
    def _window_result_to_dict(self, window_result: WindowResult) -> Dict:
        """转换窗口结果为字典"""
        return {
            'window_id': window_result.window_id,
            'train_start': window_result.train_start.isoformat(),
            'train_end': window_result.train_end.isoformat(),
            'test_start': window_result.test_start.isoformat(),
            'test_end': window_result.test_end.isoformat(),
            'train_trades': window_result.train_trades,
            'test_trades': window_result.test_trades,
            'test_metrics': window_result.test_metrics,
            'optimal_parameters': window_result.optimal_parameters,
            'parameter_stability': window_result.parameter_stability
        }
    
    def _save_walk_forward_results(self, window_results: List[WindowResult], analysis: Dict) -> None:
        """保存Walk-Forward结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        results_file = self.results_dir / f"walk_forward_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # 保存摘要
        summary = {
            'timestamp': timestamp,
            'total_windows': len(window_results),
            'overall_pass': analysis['overall_assessment']['overall_pass'],
            'stability_score': analysis['overall_assessment']['time_stability_score'],
            'avg_win_rate': analysis['overall_assessment']['avg_win_rate'],
            'recommendations': analysis['recommendations']
        }
        
        summary_file = self.results_dir / f"walk_forward_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Walk-Forward结果已保存: {results_file}")
        logger.info(f"Walk-Forward摘要已保存: {summary_file}")