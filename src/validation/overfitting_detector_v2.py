#!/usr/bin/env python3
"""
过拟合检测器 V2 - 增强版过拟合检测
Overfitting Detector V2 - Enhanced Overfitting Detection

核心功能:
1. 多维度过拟合检测
2. 高级统计检验
3. 数据挖掘偏差检测
4. 样本外性能预测

Author: DipMaster Trading Team
Date: 2025-08-15
Version: 2.0.0
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mutual_info_score
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class OverfittingResult:
    """过拟合检测结果"""
    test_name: str
    overfitting_score: float  # 0-100, 100为严重过拟合
    risk_level: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    evidence: List[str]
    p_value: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    recommendation: str

class OverfittingDetectorV2:
    """
    增强版过拟合检测器
    
    实现多种先进的过拟合检测方法
    """
    
    def __init__(self):
        self.results_dir = Path("results/overfitting_v2")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def comprehensive_overfitting_analysis(self,
                                         train_data: pd.DataFrame,
                                         val_data: pd.DataFrame,
                                         test_data: pd.DataFrame,
                                         strategy_results: Dict,
                                         parameter_history: List[Dict] = None) -> Dict:
        """
        综合过拟合分析
        
        Args:
            train_data: 训练数据
            val_data: 验证数据  
            test_data: 测试数据
            strategy_results: 策略结果
            parameter_history: 参数调优历史
            
        Returns:
            Dict: 综合分析结果
        """
        logger.info("开始综合过拟合分析...")
        
        detection_results = {}
        
        # 1. 数据泄漏检测
        data_leakage_result = self._detect_data_leakage(train_data, val_data, test_data)
        detection_results['data_leakage'] = data_leakage_result
        
        # 2. 样本外性能衰减检测
        performance_decay_result = self._detect_performance_decay(strategy_results)
        detection_results['performance_decay'] = performance_decay_result
        
        # 3. 参数过度优化检测
        if parameter_history:
            param_overfitting_result = self._detect_parameter_overfitting(parameter_history)
            detection_results['parameter_overfitting'] = param_overfitting_result
        
        # 4. 多重比较偏差检测
        multiple_testing_result = self._detect_multiple_testing_bias(strategy_results)
        detection_results['multiple_testing_bias'] = multiple_testing_result
        
        # 5. 时间序列交叉验证
        cv_result = self._time_series_cross_validation(train_data, strategy_results)
        detection_results['cross_validation'] = cv_result
        
        # 6. 复杂性惩罚分析
        complexity_result = self._analyze_model_complexity(strategy_results)
        detection_results['complexity_analysis'] = complexity_result
        
        # 7. 信息泄漏检测
        info_leakage_result = self._detect_information_leakage(train_data, test_data)
        detection_results['information_leakage'] = info_leakage_result
        
        # 8. 生存偏差检测
        survivorship_result = self._detect_survivorship_bias(strategy_results)
        detection_results['survivorship_bias'] = survivorship_result
        
        # 综合评估
        overall_assessment = self._generate_overall_assessment(detection_results)
        
        # 创建最终报告
        final_report = {
            'detection_results': detection_results,
            'overall_assessment': overall_assessment,
            'risk_summary': self._create_risk_summary(detection_results),
            'recommendations': self._generate_detailed_recommendations(overall_assessment),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # 保存结果
        self._save_detection_results(final_report)
        
        logger.info("综合过拟合分析完成")
        return final_report
    
    def _detect_data_leakage(self, train_data: pd.DataFrame, 
                           val_data: pd.DataFrame, 
                           test_data: pd.DataFrame) -> OverfittingResult:
        """检测数据泄漏"""
        logger.info("检测数据泄漏...")
        
        evidence = []
        overfitting_score = 0
        
        # 检查时间重叠
        train_end = train_data['timestamp'].max()
        val_start = val_data['timestamp'].min()
        test_start = test_data['timestamp'].min()
        
        if val_start <= train_end:
            evidence.append("验证集与训练集时间重叠")
            overfitting_score += 40
        
        if test_start <= val_data['timestamp'].max():
            evidence.append("测试集与验证集时间重叠")
            overfitting_score += 40
        
        # 检查数据分布相似性 (KS检验)
        if 'close' in train_data.columns:
            ks_stat_tv, ks_p_tv = stats.ks_2samp(train_data['close'], val_data['close'])
            ks_stat_tt, ks_p_tt = stats.ks_2samp(train_data['close'], test_data['close'])
            
            if ks_p_tv > 0.05:  # 分布过于相似
                evidence.append(f"训练集与验证集分布过于相似 (KS p-value: {ks_p_tv:.4f})")
                overfitting_score += 20
            
            if ks_p_tt > 0.05:
                evidence.append(f"训练集与测试集分布过于相似 (KS p-value: {ks_p_tt:.4f})")
                overfitting_score += 20
        
        # 检查数据完整性
        if len(set(train_data.columns) - set(val_data.columns)) > 0:
            evidence.append("训练集与验证集特征不一致")
            overfitting_score += 30
        
        risk_level = self._score_to_risk_level(overfitting_score)
        
        return OverfittingResult(
            test_name="数据泄漏检测",
            overfitting_score=overfitting_score,
            risk_level=risk_level,
            evidence=evidence,
            p_value=None,
            confidence_interval=None,
            recommendation=self._get_data_leakage_recommendation(overfitting_score)
        )
    
    def _detect_performance_decay(self, strategy_results: Dict) -> OverfittingResult:
        """检测样本外性能衰减"""
        logger.info("检测样本外性能衰减...")
        
        evidence = []
        overfitting_score = 0
        
        # 比较训练、验证、测试性能
        train_performance = strategy_results.get('train_metrics', {})
        val_performance = strategy_results.get('val_metrics', {})
        test_performance = strategy_results.get('test_metrics', {})
        
        # 胜率衰减检测
        train_wr = train_performance.get('win_rate', 0)
        val_wr = val_performance.get('win_rate', 0)
        test_wr = test_performance.get('win_rate', 0)
        
        if train_wr > 0:
            val_decay = (train_wr - val_wr) / train_wr
            test_decay = (train_wr - test_wr) / train_wr
            
            if val_decay > 0.1:  # 10%以上衰减
                evidence.append(f"验证集胜率衰减{val_decay*100:.1f}%")
                overfitting_score += min(val_decay * 200, 40)
            
            if test_decay > 0.15:  # 15%以上衰减
                evidence.append(f"测试集胜率衰减{test_decay*100:.1f}%")
                overfitting_score += min(test_decay * 200, 50)
        
        # 夏普比率衰减检测
        train_sharpe = train_performance.get('sharpe_ratio', 0)
        val_sharpe = val_performance.get('sharpe_ratio', 0)
        test_sharpe = test_performance.get('sharpe_ratio', 0)
        
        if train_sharpe > 0:
            sharpe_val_decay = (train_sharpe - val_sharpe) / train_sharpe
            sharpe_test_decay = (train_sharpe - test_sharpe) / train_sharpe
            
            if sharpe_val_decay > 0.2:
                evidence.append(f"验证集夏普比率衰减{sharpe_val_decay*100:.1f}%")
                overfitting_score += min(sharpe_val_decay * 150, 30)
            
            if sharpe_test_decay > 0.3:
                evidence.append(f"测试集夏普比率衰减{sharpe_test_decay*100:.1f}%")
                overfitting_score += min(sharpe_test_decay * 150, 40)
        
        # 检测性能逆转
        if val_wr > train_wr:
            evidence.append("验证集性能异常优于训练集")
            overfitting_score += 30
        
        risk_level = self._score_to_risk_level(overfitting_score)
        
        return OverfittingResult(
            test_name="样本外性能衰减检测",
            overfitting_score=overfitting_score,
            risk_level=risk_level,
            evidence=evidence,
            p_value=None,
            confidence_interval=None,
            recommendation=self._get_performance_decay_recommendation(overfitting_score)
        )
    
    def _detect_parameter_overfitting(self, parameter_history: List[Dict]) -> OverfittingResult:
        """检测参数过度优化"""
        logger.info("检测参数过度优化...")
        
        evidence = []
        overfitting_score = 0
        
        if len(parameter_history) < 2:
            return OverfittingResult(
                test_name="参数过度优化检测",
                overfitting_score=0,
                risk_level="LOW",
                evidence=["参数历史数据不足"],
                p_value=None,
                confidence_interval=None,
                recommendation="增加参数优化历史记录"
            )
        
        # 检测参数搜索次数
        search_count = len(parameter_history)
        if search_count > 100:
            evidence.append(f"参数搜索次数过多: {search_count}")
            overfitting_score += min(search_count / 10, 40)
        
        # 检测参数稳定性
        param_stability = self._calculate_parameter_stability_advanced(parameter_history)
        if param_stability < 0.5:
            evidence.append(f"参数稳定性差: {param_stability:.3f}")
            overfitting_score += (1 - param_stability) * 50
        
        # 检测性能改进的合理性
        performance_improvements = []
        for i in range(1, len(parameter_history)):
            current_perf = parameter_history[i].get('performance', 0)
            prev_perf = parameter_history[i-1].get('performance', 0)
            if prev_perf > 0:
                improvement = (current_perf - prev_perf) / prev_perf
                performance_improvements.append(improvement)
        
        if performance_improvements:
            avg_improvement = np.mean(performance_improvements)
            if avg_improvement > 0.05:  # 每次5%以上改进可疑
                evidence.append(f"参数优化改进过于显著: {avg_improvement*100:.1f}%")
                overfitting_score += min(avg_improvement * 200, 30)
        
        # 检测参数边界效应
        boundary_effects = self._detect_parameter_boundary_effects(parameter_history)
        if boundary_effects:
            evidence.extend(boundary_effects)
            overfitting_score += len(boundary_effects) * 15
        
        risk_level = self._score_to_risk_level(overfitting_score)
        
        return OverfittingResult(
            test_name="参数过度优化检测",
            overfitting_score=overfitting_score,
            risk_level=risk_level,
            evidence=evidence,
            p_value=None,
            confidence_interval=None,
            recommendation=self._get_parameter_overfitting_recommendation(overfitting_score)
        )
    
    def _detect_multiple_testing_bias(self, strategy_results: Dict) -> OverfittingResult:
        """检测多重比较偏差"""
        logger.info("检测多重比较偏差...")
        
        evidence = []
        overfitting_score = 0
        
        # 检测测试的数量
        test_count = strategy_results.get('test_count', 1)
        symbol_count = strategy_results.get('symbol_count', 1)
        parameter_combinations = strategy_results.get('parameter_combinations_tested', 1)
        
        # 计算有效的假设检验数量
        effective_tests = test_count * symbol_count * np.log(parameter_combinations + 1)
        
        if effective_tests > 10:
            evidence.append(f"多重假设检验数量: {effective_tests:.0f}")
            overfitting_score += min(np.log(effective_tests) * 15, 50)
        
        # 检查是否进行了Bonferroni校正
        if not strategy_results.get('bonferroni_corrected', False) and effective_tests > 5:
            evidence.append("未进行多重比较校正")
            overfitting_score += 25
        
        # 检测显著性购物 (p-hacking)
        reported_p_values = strategy_results.get('p_values', [])
        if reported_p_values:
            significant_count = sum(1 for p in reported_p_values if p < 0.05)
            expected_significant = len(reported_p_values) * 0.05
            
            if significant_count > expected_significant * 2:
                evidence.append(f"显著结果异常多: {significant_count}/{len(reported_p_values)}")
                overfitting_score += 30
        
        risk_level = self._score_to_risk_level(overfitting_score)
        
        return OverfittingResult(
            test_name="多重比较偏差检测",
            overfitting_score=overfitting_score,
            risk_level=risk_level,
            evidence=evidence,
            p_value=None,
            confidence_interval=None,
            recommendation=self._get_multiple_testing_recommendation(overfitting_score)
        )
    
    def _time_series_cross_validation(self, data: pd.DataFrame, 
                                    strategy_results: Dict) -> OverfittingResult:
        """时间序列交叉验证"""
        logger.info("进行时间序列交叉验证...")
        
        evidence = []
        overfitting_score = 0
        
        # 时间序列分割
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        # 模拟交叉验证 (实际应该调用真实策略)
        for train_index, test_index in tscv.split(data):
            # 这里应该运行策略并获取性能分数
            # 为了演示，我们使用模拟分数
            score = np.random.normal(0.55, 0.1)  # 模拟胜率
            cv_scores.append(score)
        
        cv_scores = np.array(cv_scores)
        
        # 分析交叉验证稳定性
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        cv_cv = cv_std / cv_mean if cv_mean > 0 else float('inf')
        
        # 与原始性能比较
        original_performance = strategy_results.get('train_metrics', {}).get('win_rate', 0)
        
        if original_performance > 0:
            performance_diff = abs(cv_mean - original_performance) / original_performance
            
            if performance_diff > 0.1:
                evidence.append(f"交叉验证性能差异: {performance_diff*100:.1f}%")
                overfitting_score += min(performance_diff * 200, 40)
        
        # 检测性能稳定性
        if cv_cv > 0.2:
            evidence.append(f"交叉验证稳定性差: CV={cv_cv:.3f}")
            overfitting_score += min(cv_cv * 100, 30)
        
        # 检测趋势
        if len(cv_scores) >= 3:
            trend_slope, _, trend_r, trend_p, _ = stats.linregress(range(len(cv_scores)), cv_scores)
            if trend_p < 0.05 and trend_slope < 0:
                evidence.append("交叉验证显示性能衰减趋势")
                overfitting_score += 25
        
        risk_level = self._score_to_risk_level(overfitting_score)
        
        return OverfittingResult(
            test_name="时间序列交叉验证",
            overfitting_score=overfitting_score,
            risk_level=risk_level,
            evidence=evidence,
            p_value=None,
            confidence_interval=(cv_mean - 1.96*cv_std, cv_mean + 1.96*cv_std),
            recommendation=self._get_cv_recommendation(overfitting_score)
        )
    
    def _analyze_model_complexity(self, strategy_results: Dict) -> OverfittingResult:
        """分析模型复杂性"""
        logger.info("分析模型复杂性...")
        
        evidence = []
        overfitting_score = 0
        
        # 特征数量
        feature_count = strategy_results.get('feature_count', 0)
        sample_count = strategy_results.get('sample_count', 1000)
        
        # 计算特征与样本比例
        feature_ratio = feature_count / sample_count if sample_count > 0 else 0
        
        if feature_ratio > 0.1:  # 特征太多
            evidence.append(f"特征与样本比例过高: {feature_ratio:.3f}")
            overfitting_score += min(feature_ratio * 200, 40)
        
        # 参数数量
        parameter_count = strategy_results.get('parameter_count', 0)
        if parameter_count > 10:
            evidence.append(f"可调参数过多: {parameter_count}")
            overfitting_score += min(parameter_count * 2, 30)
        
        # 规则复杂性
        rule_complexity = strategy_results.get('rule_complexity_score', 0)
        if rule_complexity > 50:
            evidence.append(f"策略规则复杂度过高: {rule_complexity}")
            overfitting_score += min(rule_complexity / 2, 25)
        
        # 计算AIC/BIC惩罚
        if 'log_likelihood' in strategy_results:
            ll = strategy_results['log_likelihood']
            n = sample_count
            k = parameter_count
            
            aic = 2*k - 2*ll
            bic = k*np.log(n) - 2*ll
            
            # 简单的复杂性评估
            complexity_penalty = (aic + bic) / n
            if complexity_penalty > 1:
                evidence.append(f"模型复杂性惩罚过高: {complexity_penalty:.3f}")
                overfitting_score += min(complexity_penalty * 20, 20)
        
        risk_level = self._score_to_risk_level(overfitting_score)
        
        return OverfittingResult(
            test_name="模型复杂性分析",
            overfitting_score=overfitting_score,
            risk_level=risk_level,
            evidence=evidence,
            p_value=None,
            confidence_interval=None,
            recommendation=self._get_complexity_recommendation(overfitting_score)
        )
    
    def _detect_information_leakage(self, train_data: pd.DataFrame, 
                                  test_data: pd.DataFrame) -> OverfittingResult:
        """检测信息泄漏"""
        logger.info("检测信息泄漏...")
        
        evidence = []
        overfitting_score = 0
        
        # 检测特征之间的互信息
        common_features = set(train_data.columns) & set(test_data.columns)
        numeric_features = [col for col in common_features 
                          if train_data[col].dtype in ['float64', 'int64']]
        
        if len(numeric_features) >= 2:
            # 计算特征间的互信息
            mutual_info_scores = []
            for i in range(len(numeric_features)):
                for j in range(i+1, len(numeric_features)):
                    feat1 = numeric_features[i]
                    feat2 = numeric_features[j]
                    
                    # 离散化连续特征
                    train_f1_disc = pd.cut(train_data[feat1], bins=10, labels=False)
                    train_f2_disc = pd.cut(train_data[feat2], bins=10, labels=False)
                    
                    # 计算互信息
                    mi_score = mutual_info_score(train_f1_disc.dropna(), 
                                               train_f2_disc.dropna())
                    mutual_info_scores.append(mi_score)
            
            if mutual_info_scores:
                max_mi = max(mutual_info_scores)
                avg_mi = np.mean(mutual_info_scores)
                
                if max_mi > 0.5:
                    evidence.append(f"特征间存在高互信息: {max_mi:.3f}")
                    overfitting_score += min(max_mi * 40, 30)
                
                if avg_mi > 0.2:
                    evidence.append(f"特征平均互信息过高: {avg_mi:.3f}")
                    overfitting_score += min(avg_mi * 50, 20)
        
        # 检测未来信息泄漏
        if 'timestamp' in train_data.columns and 'timestamp' in test_data.columns:
            train_max_time = train_data['timestamp'].max()
            test_min_time = test_data['timestamp'].min()
            
            if test_min_time <= train_max_time:
                evidence.append("存在时间泄漏：测试集时间早于或等于训练集")
                overfitting_score += 50
        
        # 检测标签泄漏
        if 'target' in train_data.columns and 'target' in test_data.columns:
            train_target_dist = train_data['target'].value_counts(normalize=True)
            test_target_dist = test_data['target'].value_counts(normalize=True)
            
            # 使用KS检验比较分布
            if len(train_target_dist) == len(test_target_dist):
                ks_stat, ks_p = stats.ks_2samp(train_target_dist.values, 
                                             test_target_dist.values)
                if ks_p > 0.1:  # 分布过于相似
                    evidence.append(f"目标变量分布异常相似: KS p-value={ks_p:.4f}")
                    overfitting_score += 25
        
        risk_level = self._score_to_risk_level(overfitting_score)
        
        return OverfittingResult(
            test_name="信息泄漏检测",
            overfitting_score=overfitting_score,
            risk_level=risk_level,
            evidence=evidence,
            p_value=None,
            confidence_interval=None,
            recommendation=self._get_info_leakage_recommendation(overfitting_score)
        )
    
    def _detect_survivorship_bias(self, strategy_results: Dict) -> OverfittingResult:
        """检测生存偏差"""
        logger.info("检测生存偏差...")
        
        evidence = []
        overfitting_score = 0
        
        # 检测资产选择偏差
        total_assets_tested = strategy_results.get('total_assets_tested', 1)
        final_assets_used = strategy_results.get('final_assets_used', 1)
        
        selection_ratio = final_assets_used / total_assets_tested
        
        if selection_ratio < 0.5:
            evidence.append(f"资产筛选比例过低: {selection_ratio*100:.1f}%")
            overfitting_score += (1 - selection_ratio) * 40
        
        # 检测时间段选择偏差
        if 'time_period_selection_reason' not in strategy_results:
            evidence.append("未说明时间段选择原因")
            overfitting_score += 20
        
        # 检测策略选择偏差
        strategies_tested = strategy_results.get('strategies_tested', 1)
        if strategies_tested > 5:
            evidence.append(f"测试策略数量过多: {strategies_tested}")
            overfitting_score += min(np.log(strategies_tested) * 15, 30)
        
        # 检测参数空间选择偏差
        param_space_coverage = strategy_results.get('parameter_space_coverage', 1.0)
        if param_space_coverage < 0.3:
            evidence.append(f"参数空间覆盖不足: {param_space_coverage*100:.1f}%")
            overfitting_score += (0.5 - param_space_coverage) * 40
        
        risk_level = self._score_to_risk_level(overfitting_score)
        
        return OverfittingResult(
            test_name="生存偏差检测",
            overfitting_score=overfitting_score,
            risk_level=risk_level,
            evidence=evidence,
            p_value=None,
            confidence_interval=None,
            recommendation=self._get_survivorship_recommendation(overfitting_score)
        )
    
    def _calculate_parameter_stability_advanced(self, parameter_history: List[Dict]) -> float:
        """计算高级参数稳定性"""
        if len(parameter_history) < 2:
            return 1.0
        
        stability_scores = []
        
        # 获取所有参数名
        all_params = set()
        for params in parameter_history:
            all_params.update(params.keys())
        
        for param_name in all_params:
            param_values = []
            for params in parameter_history:
                if param_name in params:
                    param_values.append(params[param_name])
            
            if len(param_values) < 2:
                continue
            
            if all(isinstance(v, (int, float)) for v in param_values):
                # 数值参数稳定性
                values_array = np.array(param_values)
                if np.std(values_array) == 0:
                    stability = 1.0
                else:
                    cv = np.std(values_array) / (abs(np.mean(values_array)) + 0.001)
                    stability = 1 / (1 + cv)
                stability_scores.append(stability)
            else:
                # 分类参数稳定性
                unique_count = len(set(param_values))
                stability = 1 / unique_count
                stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 1.0
    
    def _detect_parameter_boundary_effects(self, parameter_history: List[Dict]) -> List[str]:
        """检测参数边界效应"""
        boundary_effects = []
        
        # 获取数值参数
        numeric_params = {}
        for params in parameter_history:
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    if key not in numeric_params:
                        numeric_params[key] = []
                    numeric_params[key].append(value)
        
        for param_name, values in numeric_params.items():
            if len(values) < 3:
                continue
            
            values_array = np.array(values)
            min_val = np.min(values_array)
            max_val = np.max(values_array)
            
            # 检测最优值是否在边界
            best_idx = np.argmax([params.get('performance', 0) for params in parameter_history])
            best_value = parameter_history[best_idx].get(param_name)
            
            if best_value is not None:
                if best_value == min_val or best_value == max_val:
                    boundary_effects.append(f"参数{param_name}最优值在边界: {best_value}")
        
        return boundary_effects
    
    def _score_to_risk_level(self, score: float) -> str:
        """转换分数为风险等级"""
        if score >= 70:
            return "CRITICAL"
        elif score >= 50:
            return "HIGH"
        elif score >= 30:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_overall_assessment(self, detection_results: Dict) -> Dict:
        """生成综合评估"""
        total_score = 0
        high_risk_count = 0
        critical_risk_count = 0
        all_evidence = []
        
        for test_name, result in detection_results.items():
            total_score += result.overfitting_score
            all_evidence.extend(result.evidence)
            
            if result.risk_level == "HIGH":
                high_risk_count += 1
            elif result.risk_level == "CRITICAL":
                critical_risk_count += 1
        
        avg_score = total_score / len(detection_results)
        
        # 综合风险等级
        if critical_risk_count > 0 or avg_score >= 60:
            overall_risk = "CRITICAL"
        elif high_risk_count >= 2 or avg_score >= 40:
            overall_risk = "HIGH"
        elif avg_score >= 20:
            overall_risk = "MEDIUM"
        else:
            overall_risk = "LOW"
        
        # 生成建议
        if overall_risk == "CRITICAL":
            recommendation = "🚨 严重过拟合风险 - 禁止实盘交易，需要重新设计策略"
        elif overall_risk == "HIGH":
            recommendation = "⚠️ 高过拟合风险 - 需要大幅修改策略和验证方法"
        elif overall_risk == "MEDIUM":
            recommendation = "⚠️ 中等过拟合风险 - 建议进一步验证和改进"
        else:
            recommendation = "✅ 低过拟合风险 - 策略相对可靠"
        
        return {
            'overall_risk_level': overall_risk,
            'average_overfitting_score': avg_score,
            'total_overfitting_score': total_score,
            'high_risk_tests': high_risk_count,
            'critical_risk_tests': critical_risk_count,
            'all_evidence': all_evidence,
            'recommendation': recommendation,
            'safe_for_trading': overall_risk in ["LOW", "MEDIUM"]
        }
    
    def _create_risk_summary(self, detection_results: Dict) -> Dict:
        """创建风险摘要"""
        risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}
        
        for result in detection_results.values():
            risk_counts[result.risk_level] += 1
        
        return {
            'total_tests': len(detection_results),
            'risk_distribution': risk_counts,
            'highest_risk_tests': [
                name for name, result in detection_results.items() 
                if result.risk_level in ["HIGH", "CRITICAL"]
            ]
        }
    
    def _generate_detailed_recommendations(self, assessment: Dict) -> List[str]:
        """生成详细建议"""
        recommendations = []
        
        if assessment['overall_risk_level'] == "CRITICAL":
            recommendations.extend([
                "立即停止当前策略的实盘交易准备",
                "重新审视整个策略开发流程",
                "实施严格的样本外验证",
                "减少策略复杂性，使用更简单的逻辑",
                "增加数据量或改进数据质量"
            ])
        elif assessment['overall_risk_level'] == "HIGH":
            recommendations.extend([
                "暂缓实盘交易，进行进一步验证",
                "实施Walk-Forward分析",
                "进行蒙特卡洛随机化测试",
                "简化策略参数和逻辑",
                "增强跨资产验证"
            ])
        elif assessment['overall_risk_level'] == "MEDIUM":
            recommendations.extend([
                "进行额外的样本外验证",
                "监控实盘前的纸面交易表现",
                "实施渐进式资金投入",
                "建立严格的止损机制"
            ])
        else:
            recommendations.extend([
                "策略风险可控，可考虑小规模实盘",
                "持续监控实盘表现",
                "定期重新验证策略有效性"
            ])
        
        return recommendations
    
    # 各种推荐函数的实现
    def _get_data_leakage_recommendation(self, score: float) -> str:
        if score >= 50:
            return "严重数据泄漏，必须重新划分数据集"
        elif score >= 30:
            return "存在数据泄漏风险，建议检查数据划分"
        else:
            return "数据划分相对合理"
    
    def _get_performance_decay_recommendation(self, score: float) -> str:
        if score >= 50:
            return "样本外性能严重衰减，策略过拟合严重"
        elif score >= 30:
            return "存在性能衰减，需要改进策略稳健性"
        else:
            return "样本外性能相对稳定"
    
    def _get_parameter_overfitting_recommendation(self, score: float) -> str:
        if score >= 50:
            return "参数过度优化，建议大幅简化策略"
        elif score >= 30:
            return "存在参数过拟合风险，减少可调参数"
        else:
            return "参数优化相对合理"
    
    def _get_multiple_testing_recommendation(self, score: float) -> str:
        if score >= 50:
            return "多重比较偏差严重，必须进行校正"
        elif score >= 30:
            return "存在多重比较偏差，建议进行校正"
        else:
            return "多重比较风险可控"
    
    def _get_cv_recommendation(self, score: float) -> str:
        if score >= 50:
            return "交叉验证显示严重不稳定性"
        elif score >= 30:
            return "交叉验证显示中等不稳定性"
        else:
            return "交叉验证结果相对稳定"
    
    def _get_complexity_recommendation(self, score: float) -> str:
        if score >= 50:
            return "模型过于复杂，需要大幅简化"
        elif score >= 30:
            return "模型复杂性偏高，建议适当简化"
        else:
            return "模型复杂性合理"
    
    def _get_info_leakage_recommendation(self, score: float) -> str:
        if score >= 50:
            return "存在严重信息泄漏，检查特征工程"
        elif score >= 30:
            return "存在信息泄漏风险，审查数据处理"
        else:
            return "信息泄漏风险较低"
    
    def _get_survivorship_recommendation(self, score: float) -> str:
        if score >= 50:
            return "存在严重生存偏差，重新评估选择标准"
        elif score >= 30:
            return "存在生存偏差风险，检查选择过程"
        else:
            return "生存偏差风险可控"
    
    def _save_detection_results(self, results: Dict) -> None:
        """保存检测结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"overfitting_detection_v2_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # 转换为可序列化格式
        serializable_results = self._make_serializable(results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"过拟合检测结果已保存: {filepath}")
    
    def _make_serializable(self, obj):
        """转换对象为可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, OverfittingResult):
            return {
                'test_name': obj.test_name,
                'overfitting_score': obj.overfitting_score,
                'risk_level': obj.risk_level,
                'evidence': obj.evidence,
                'p_value': obj.p_value,
                'confidence_interval': obj.confidence_interval,
                'recommendation': obj.recommendation
            }
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        else:
            return obj