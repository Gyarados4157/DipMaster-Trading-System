#!/usr/bin/env python3
"""
增强时序验证系统
实施严格的Purged K-Fold交叉验证和Walk-forward分析
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# 机器学习库
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_score, recall_score, f1_score, log_loss
)
from sklearn.model_selection import TimeSeriesSplit

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """验证配置"""
    # Purged K-Fold参数
    n_splits: int = 5
    embargo_hours: int = 2
    test_size_ratio: float = 0.2
    
    # Walk-forward参数
    min_train_size_months: int = 6  # 最小训练集大小（月）
    rebalance_frequency_days: int = 30  # 重新平衡频率（天）
    walk_forward_steps: int = 12  # Walk-forward步数
    
    # 数据泄漏检测
    max_feature_correlation_threshold: float = 0.95
    max_ic_threshold: float = 0.3  # 信息系数阈值
    
    # 统计显著性
    min_sample_size: int = 100
    significance_level: float = 0.05
    bootstrap_samples: int = 1000

class EnhancedPurgedKFold:
    """增强的Purged K-Fold交叉验证器"""
    
    def __init__(self, n_splits: int = 5, embargo_hours: int = 2, test_size: float = 0.2):
        self.n_splits = n_splits
        self.embargo_td = timedelta(hours=embargo_hours)
        self.test_size = test_size
        
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None) -> Generator:
        """生成时序分割索引"""
        if 'timestamp' not in X.columns:
            # 如果没有timestamp列，创建一个假的时间序列
            timestamps = pd.date_range(start='2022-01-01', periods=len(X), freq='5min')
            X = X.copy()
            X['timestamp'] = timestamps
        
        timestamps = pd.to_datetime(X['timestamp'])
        indices = np.arange(len(X))
        
        # 按时间排序
        sorted_idx = timestamps.argsort()
        timestamps_sorted = timestamps.iloc[sorted_idx]
        
        n_samples = len(X)
        
        # 为测试集预留数据
        train_end_idx = int(n_samples * (1 - self.test_size))
        
        # 计算每个fold的大小
        fold_size = train_end_idx // self.n_splits
        
        for i in range(self.n_splits):
            # 训练集索引范围
            train_start = i * (fold_size // 2) if i > 0 else 0  # 重叠训练以增加样本
            train_end = min((i + 1) * fold_size, train_end_idx)
            
            if train_end <= train_start:
                continue
                
            # 应用embargo期
            train_end_time = timestamps_sorted.iloc[train_end - 1]
            val_start_time = train_end_time + self.embargo_td
            
            # 找到验证集开始索引
            val_mask = timestamps_sorted >= val_start_time
            if not val_mask.any():
                continue
                
            val_start_idx = timestamps_sorted[val_mask].index[0]
            val_end_idx = min(train_end + fold_size, train_end_idx)
            
            # 获取原始索引
            train_indices = sorted_idx[train_start:train_end].values
            val_indices = sorted_idx[val_start_idx:val_end_idx].values
            
            # 确保有足够的样本
            if len(train_indices) >= 100 and len(val_indices) >= 20:
                yield train_indices, val_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

class WalkForwardAnalyzer:
    """Walk-forward分析器"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def create_walk_forward_splits(self, X: pd.DataFrame, y: pd.Series) -> List[Tuple]:
        """创建walk-forward分割"""
        if 'timestamp' not in X.columns:
            timestamps = pd.date_range(start='2022-01-01', periods=len(X), freq='5min')
            X = X.copy()
            X['timestamp'] = timestamps
            
        timestamps = pd.to_datetime(X['timestamp'])
        
        # 按时间排序
        sorted_indices = timestamps.sort_values().index
        
        # 计算最小训练集大小
        min_train_samples = int(len(X) * 0.3)  # 至少30%用于训练
        
        # 计算每个walk步骤的大小
        remaining_samples = len(X) - min_train_samples
        step_size = remaining_samples // self.config.walk_forward_steps
        
        splits = []
        
        for i in range(self.config.walk_forward_steps):
            # 训练集：从开始到当前步骤
            train_end = min_train_samples + i * step_size
            
            # 测试集：下一个步骤的数据
            test_start = train_end
            test_end = min(test_start + step_size, len(X))
            
            if test_end <= test_start:
                break
                
            train_indices = sorted_indices[:train_end].tolist()
            test_indices = sorted_indices[test_start:test_end].tolist()
            
            # 应用embargo
            if len(train_indices) > 0 and len(test_indices) > 0:
                train_end_time = timestamps.loc[train_indices[-1]]
                embargo_end_time = train_end_time + timedelta(hours=self.config.embargo_hours)
                
                # 过滤测试集中embargo期内的数据
                valid_test_indices = [
                    idx for idx in test_indices 
                    if timestamps.loc[idx] >= embargo_end_time
                ]
                
                if len(valid_test_indices) >= 20:  # 确保有足够的测试样本
                    splits.append((train_indices, valid_test_indices))
        
        logger.info(f"创建了 {len(splits)} 个walk-forward分割")
        return splits
    
    def walk_forward_validation(self, X: pd.DataFrame, y: pd.Series, 
                              model_factory, fit_params: dict = None) -> Dict:
        """执行walk-forward验证"""
        logger.info("开始walk-forward验证...")
        
        splits = self.create_walk_forward_splits(X, y)
        if not splits:
            return {'error': 'No valid splits created'}
        
        results = {
            'fold_results': [],
            'predictions': [],
            'actuals': [],
            'timestamps': [],
            'performance_over_time': []
        }
        
        for fold_idx, (train_indices, test_indices) in enumerate(splits):
            logger.info(f"Walk-forward fold {fold_idx + 1}/{len(splits)}")
            
            try:
                # 准备数据
                X_train = X.iloc[train_indices].drop(columns=['timestamp'], errors='ignore')
                X_test = X.iloc[test_indices].drop(columns=['timestamp'], errors='ignore')
                y_train = y.iloc[train_indices]
                y_test = y.iloc[test_indices]
                
                # 时间信息
                test_timestamps = X.iloc[test_indices]['timestamp'] if 'timestamp' in X.columns else None
                
                # 训练模型
                model = model_factory()
                if fit_params:
                    model.fit(X_train, y_train, **fit_params)
                else:
                    model.fit(X_train, y_train)
                
                # 预测
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                else:
                    y_pred = model.predict(X_test)
                    y_pred_proba = y_pred.astype(float)
                
                # 计算性能指标
                fold_metrics = self._calculate_fold_metrics(y_test, y_pred, y_pred_proba)
                fold_metrics['fold'] = fold_idx
                fold_metrics['train_size'] = len(train_indices)
                fold_metrics['test_size'] = len(test_indices)
                
                results['fold_results'].append(fold_metrics)
                results['predictions'].extend(y_pred_proba)
                results['actuals'].extend(y_test)
                if test_timestamps is not None:
                    results['timestamps'].extend(test_timestamps)
                
                # 记录时间序列表现
                if test_timestamps is not None:
                    time_performance = {
                        'fold': fold_idx,
                        'start_date': test_timestamps.iloc[0],
                        'end_date': test_timestamps.iloc[-1],
                        'accuracy': fold_metrics['accuracy'],
                        'auc': fold_metrics['auc'],
                        'precision': fold_metrics['precision'],
                        'recall': fold_metrics['recall']
                    }
                    results['performance_over_time'].append(time_performance)
                
            except Exception as e:
                logger.error(f"Fold {fold_idx} 失败: {e}")
                continue
        
        # 计算综合统计
        if results['fold_results']:
            results['summary_stats'] = self._calculate_summary_stats(results['fold_results'])
            results['stability_metrics'] = self._calculate_stability_metrics(results['fold_results'])
        
        return results
    
    def _calculate_fold_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                               y_pred_proba: np.ndarray) -> Dict:
        """计算单个fold的性能指标"""
        metrics = {}
        
        try:
            metrics['accuracy'] = (y_true == y_pred).mean()
            metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
            
            if len(np.unique(y_true)) > 1:
                metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            else:
                metrics['auc'] = 0.5
                metrics['log_loss'] = np.inf
                
        except Exception as e:
            logger.warning(f"计算指标时出错: {e}")
            metrics = {key: 0.0 for key in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'log_loss']}
            
        return metrics
    
    def _calculate_summary_stats(self, fold_results: List[Dict]) -> Dict:
        """计算汇总统计"""
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        summary = {}
        
        for metric in metrics:
            values = [result[metric] for result in fold_results if metric in result]
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_std'] = np.std(values)
                summary[f'{metric}_min'] = np.min(values)
                summary[f'{metric}_max'] = np.max(values)
                
                # 置信区间
                ci_lower, ci_upper = stats.t.interval(
                    0.95, len(values)-1, 
                    loc=np.mean(values), 
                    scale=stats.sem(values)
                )
                summary[f'{metric}_ci_lower'] = ci_lower
                summary[f'{metric}_ci_upper'] = ci_upper
        
        return summary
    
    def _calculate_stability_metrics(self, fold_results: List[Dict]) -> Dict:
        """计算模型稳定性指标"""
        metrics = ['accuracy', 'auc', 'precision', 'recall']
        stability = {}
        
        for metric in metrics:
            values = [result[metric] for result in fold_results if metric in result]
            if len(values) > 1:
                # 变异系数
                cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
                stability[f'{metric}_cv'] = cv
                
                # 趋势检测（Spearman相关性）
                fold_numbers = list(range(len(values)))
                correlation, p_value = stats.spearmanr(fold_numbers, values)
                stability[f'{metric}_trend_correlation'] = correlation
                stability[f'{metric}_trend_p_value'] = p_value
                
        return stability

class DataLeakageDetector:
    """数据泄漏检测器"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def detect_feature_leakage(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """检测特征泄漏"""
        logger.info("检测特征数据泄漏...")
        
        results = {
            'high_correlation_features': [],
            'future_information_features': [],
            'suspicious_ic_features': [],
            'leakage_score': 0.0
        }
        
        # 1. 检测高相关性特征（可能是标签泄漏）
        feature_cols = [col for col in X.columns if col not in ['timestamp']]
        
        for feature in feature_cols:
            if X[feature].dtype in ['int64', 'float64']:
                # 计算与目标的相关性
                correlation = X[feature].corr(y)
                
                if abs(correlation) > self.config.max_feature_correlation_threshold:
                    results['high_correlation_features'].append({
                        'feature': feature,
                        'correlation': correlation
                    })
        
        # 2. 检测未来信息泄漏
        if 'timestamp' in X.columns:
            results['future_information_features'] = self._detect_future_information(X, y)
        
        # 3. 检测可疑的信息系数
        results['suspicious_ic_features'] = self._calculate_information_coefficients(X, y)
        
        # 4. 计算综合泄漏分数
        leakage_score = (
            len(results['high_correlation_features']) * 0.4 +
            len(results['future_information_features']) * 0.4 +
            len(results['suspicious_ic_features']) * 0.2
        )
        results['leakage_score'] = leakage_score / len(feature_cols) if feature_cols else 0
        
        return results
    
    def _detect_future_information(self, X: pd.DataFrame, y: pd.Series) -> List[Dict]:
        """检测未来信息泄漏"""
        suspicious_features = []
        
        if 'timestamp' not in X.columns:
            return suspicious_features
            
        timestamps = pd.to_datetime(X['timestamp'])
        feature_cols = [col for col in X.columns if col not in ['timestamp']]
        
        # 检查特征是否包含未来信息
        for feature in feature_cols:
            if feature.lower() in ['target', 'return', 'profit', 'label', 'future']:
                suspicious_features.append({
                    'feature': feature,
                    'reason': 'Feature name suggests future information'
                })
                continue
                
            # 检查特征的时间一致性
            try:
                # 计算特征的滞后相关性
                feature_values = X[feature].values
                
                # 检查特征是否与未来目标值相关
                for lag in [1, 2, 3, 6, 12]:  # 不同的前瞻期
                    if len(feature_values) > lag:
                        future_target = y.shift(-lag).dropna()
                        aligned_feature = X[feature].iloc[:len(future_target)]
                        
                        if len(aligned_feature) > 0 and aligned_feature.std() > 0:
                            correlation = aligned_feature.corr(future_target)
                            
                            if abs(correlation) > 0.3:  # 可疑的高相关性
                                suspicious_features.append({
                                    'feature': feature,
                                    'reason': f'High correlation with future target (lag {lag})',
                                    'correlation': correlation,
                                    'lag': lag
                                })
                                break
                                
            except Exception as e:
                logger.warning(f"检查特征 {feature} 时出错: {e}")
                
        return suspicious_features
    
    def _calculate_information_coefficients(self, X: pd.DataFrame, y: pd.Series) -> List[Dict]:
        """计算信息系数"""
        suspicious_features = []
        feature_cols = [col for col in X.columns if col not in ['timestamp']]
        
        for feature in feature_cols:
            try:
                if X[feature].dtype in ['int64', 'float64'] and X[feature].std() > 0:
                    # 计算信息系数（IC）
                    ic = X[feature].corr(y)
                    
                    if abs(ic) > self.config.max_ic_threshold:
                        suspicious_features.append({
                            'feature': feature,
                            'ic': ic,
                            'reason': f'High information coefficient: {ic:.3f}'
                        })
                        
            except Exception as e:
                logger.warning(f"计算IC时出错 {feature}: {e}")
                
        return suspicious_features

class EnhancedTimeSeriesValidator:
    """增强时序验证系统主类"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.purged_kfold = EnhancedPurgedKFold(
            n_splits=self.config.n_splits,
            embargo_hours=self.config.embargo_hours,
            test_size=self.config.test_size_ratio
        )
        self.walk_forward_analyzer = WalkForwardAnalyzer(self.config)
        self.leakage_detector = DataLeakageDetector(self.config)
        
        # 结果存储
        self.validation_results = {}
        
    def comprehensive_validation(self, X: pd.DataFrame, y: pd.Series, 
                               model_factory, fit_params: dict = None) -> Dict:
        """执行综合验证"""
        logger.info("开始综合时序验证...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_summary': self._summarize_data(X, y),
            'leakage_detection': {},
            'purged_cv_results': {},
            'walk_forward_results': {},
            'stability_analysis': {},
            'recommendations': []
        }
        
        try:
            # 1. 数据泄漏检测
            logger.info("1. 执行数据泄漏检测...")
            results['leakage_detection'] = self.leakage_detector.detect_feature_leakage(X, y)
            
            # 2. Purged K-Fold交叉验证
            logger.info("2. 执行Purged K-Fold交叉验证...")
            results['purged_cv_results'] = self._purged_cv_validation(X, y, model_factory, fit_params)
            
            # 3. Walk-forward验证
            logger.info("3. 执行Walk-forward验证...")
            results['walk_forward_results'] = self.walk_forward_analyzer.walk_forward_validation(
                X, y, model_factory, fit_params
            )
            
            # 4. 稳定性分析
            logger.info("4. 执行稳定性分析...")
            results['stability_analysis'] = self._analyze_stability(results)
            
            # 5. 生成建议
            results['recommendations'] = self._generate_recommendations(results)
            
        except Exception as e:
            logger.error(f"验证过程中出错: {e}")
            results['error'] = str(e)
            
        # 保存结果
        self.validation_results = results
        
        return results
    
    def _summarize_data(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """数据摘要"""
        return {
            'n_samples': len(X),
            'n_features': len(X.columns),
            'target_distribution': {
                'positive_ratio': y.mean(),
                'negative_ratio': 1 - y.mean(),
                'total_positive': y.sum(),
                'total_negative': (1 - y).sum()
            },
            'time_span': {
                'start': X['timestamp'].min() if 'timestamp' in X.columns else None,
                'end': X['timestamp'].max() if 'timestamp' in X.columns else None,
                'duration_days': (X['timestamp'].max() - X['timestamp'].min()).days if 'timestamp' in X.columns else None
            },
            'data_quality': {
                'missing_values': X.isnull().sum().sum(),
                'duplicate_rows': X.duplicated().sum(),
                'infinite_values': np.isinf(X.select_dtypes(include=[np.number])).sum().sum()
            }
        }
    
    def _purged_cv_validation(self, X: pd.DataFrame, y: pd.Series, 
                             model_factory, fit_params: dict = None) -> Dict:
        """Purged交叉验证"""
        fold_results = []
        all_predictions = []
        all_actuals = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(self.purged_kfold.split(X, y)):
            try:
                # 准备数据
                X_train = X.iloc[train_indices].drop(columns=['timestamp'], errors='ignore')
                X_val = X.iloc[val_indices].drop(columns=['timestamp'], errors='ignore')
                y_train = y.iloc[train_indices]
                y_val = y.iloc[val_indices]
                
                # 训练模型
                model = model_factory()
                if fit_params:
                    model.fit(X_train, y_train, **fit_params)
                else:
                    model.fit(X_train, y_train)
                
                # 预测
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    y_pred = (y_pred_proba >= 0.5).astype(int)
                else:
                    y_pred = model.predict(X_val)
                    y_pred_proba = y_pred.astype(float)
                
                # 计算指标
                fold_metrics = self.walk_forward_analyzer._calculate_fold_metrics(y_val, y_pred, y_pred_proba)
                fold_metrics['fold'] = fold_idx
                fold_metrics['train_size'] = len(train_indices)
                fold_metrics['val_size'] = len(val_indices)
                
                fold_results.append(fold_metrics)
                all_predictions.extend(y_pred_proba)
                all_actuals.extend(y_val)
                
            except Exception as e:
                logger.error(f"Purged CV fold {fold_idx} 失败: {e}")
                
        # 汇总结果
        if fold_results:
            summary_stats = self.walk_forward_analyzer._calculate_summary_stats(fold_results)
            stability_metrics = self.walk_forward_analyzer._calculate_stability_metrics(fold_results)
            
            return {
                'fold_results': fold_results,
                'summary_stats': summary_stats,
                'stability_metrics': stability_metrics,
                'overall_predictions': all_predictions,
                'overall_actuals': all_actuals
            }
        else:
            return {'error': 'No successful folds in Purged CV'}
    
    def _analyze_stability(self, results: Dict) -> Dict:
        """分析模型稳定性"""
        stability = {
            'cv_stability': {},
            'walk_forward_stability': {},
            'temporal_consistency': {},
            'overall_stability_score': 0.0
        }
        
        # CV稳定性
        if 'purged_cv_results' in results and 'stability_metrics' in results['purged_cv_results']:
            stability['cv_stability'] = results['purged_cv_results']['stability_metrics']
        
        # Walk-forward稳定性
        if 'walk_forward_results' in results and 'stability_metrics' in results['walk_forward_results']:
            stability['walk_forward_stability'] = results['walk_forward_results']['stability_metrics']
        
        # 时间一致性分析
        if 'walk_forward_results' in results and 'performance_over_time' in results['walk_forward_results']:
            perf_over_time = results['walk_forward_results']['performance_over_time']
            if perf_over_time:
                accuracy_values = [p['accuracy'] for p in perf_over_time]
                auc_values = [p['auc'] for p in perf_over_time]
                
                stability['temporal_consistency'] = {
                    'accuracy_trend': np.corrcoef(range(len(accuracy_values)), accuracy_values)[0, 1] if len(accuracy_values) > 1 else 0,
                    'auc_trend': np.corrcoef(range(len(auc_values)), auc_values)[0, 1] if len(auc_values) > 1 else 0,
                    'accuracy_volatility': np.std(accuracy_values) if accuracy_values else 0,
                    'auc_volatility': np.std(auc_values) if auc_values else 0
                }
        
        # 综合稳定性分数
        cv_acc_cv = stability['cv_stability'].get('accuracy_cv', 1)
        wf_acc_cv = stability['walk_forward_stability'].get('accuracy_cv', 1)
        temporal_vol = stability['temporal_consistency'].get('accuracy_volatility', 1)
        
        stability['overall_stability_score'] = 1 / (1 + cv_acc_cv + wf_acc_cv + temporal_vol)
        
        return stability
    
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """生成验证建议"""
        recommendations = []
        
        # 数据泄漏建议
        leakage = results.get('leakage_detection', {})
        if leakage.get('leakage_score', 0) > 0.1:
            recommendations.append(
                f"⚠️ 检测到数据泄漏风险 (score: {leakage['leakage_score']:.3f})，"
                f"建议检查以下特征: {[f['feature'] for f in leakage.get('high_correlation_features', [])]}"
            )
        
        # CV表现建议
        cv_results = results.get('purged_cv_results', {})
        if 'summary_stats' in cv_results:
            cv_acc = cv_results['summary_stats'].get('accuracy_mean', 0)
            cv_std = cv_results['summary_stats'].get('accuracy_std', 1)
            
            if cv_acc < 0.6:
                recommendations.append("🔄 CV准确率较低，建议增加特征工程或调整模型参数")
            
            if cv_std > 0.1:
                recommendations.append("📈 CV结果方差较大，建议增加数据量或使用正则化")
        
        # Walk-forward建议
        wf_results = results.get('walk_forward_results', {})
        if 'summary_stats' in wf_results:
            wf_acc = wf_results['summary_stats'].get('accuracy_mean', 0)
            
            if abs(cv_results.get('summary_stats', {}).get('accuracy_mean', 0) - wf_acc) > 0.05:
                recommendations.append("⏰ CV和Walk-forward结果差异较大，可能存在时序偏差")
        
        # 稳定性建议
        stability = results.get('stability_analysis', {})
        if stability.get('overall_stability_score', 0) < 0.7:
            recommendations.append("🔧 模型稳定性较低，建议使用集成方法或增加正则化")
        
        # 时间趋势建议
        temporal = stability.get('temporal_consistency', {})
        if abs(temporal.get('accuracy_trend', 0)) > 0.5:
            recommendations.append("📊 检测到明显的时间趋势，建议考虑市场制度变化")
        
        return recommendations
    
    def save_validation_report(self, output_path: str = None):
        """保存验证报告"""
        if not self.validation_results:
            logger.warning("没有验证结果可保存")
            return
            
        if output_path is None:
            output_path = f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # 转换为JSON可序列化格式
        import json
        
        def json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.float64)):
                return obj.item()
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
        
        # 清理结果以便JSON序列化
        serializable_results = {}
        for key, value in self.validation_results.items():
            try:
                if key in ['overall_predictions', 'overall_actuals', 'predictions', 'actuals']:
                    # 对于大数组，只保存统计信息
                    serializable_results[key + '_stats'] = {
                        'count': len(value),
                        'mean': np.mean(value),
                        'std': np.std(value),
                        'min': np.min(value),
                        'max': np.max(value)
                    }
                else:
                    serializable_results[key] = json_serializable(value)
            except Exception as e:
                logger.warning(f"无法序列化 {key}: {e}")
                serializable_results[key] = str(value)
        
        # 保存JSON报告
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 验证报告已保存: {output_path}")
        
        return output_path

def create_model_factory(model_type: str = 'lgbm', **params):
    """创建模型工厂函数"""
    def model_factory():
        if model_type == 'lgbm':
            import lightgbm as lgb
            default_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            default_params.update(params)
            return lgb.LGBMClassifier(**default_params)
            
        elif model_type == 'xgb':
            import xgboost as xgb
            default_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
            default_params.update(params)
            return xgb.XGBClassifier(**default_params)
            
        else:  # Random Forest
            from sklearn.ensemble import RandomForestClassifier
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            default_params.update(params)
            return RandomForestClassifier(**default_params)
    
    return model_factory

# 主函数示例
def main():
    """主函数示例"""
    # 配置
    config = ValidationConfig(
        n_splits=5,
        embargo_hours=2,
        walk_forward_steps=6
    )
    
    # 创建验证器
    validator = EnhancedTimeSeriesValidator(config)
    
    # 模拟数据（实际使用时替换为真实数据）
    np.random.seed(42)
    n_samples = 5000
    
    X = pd.DataFrame({
        'timestamp': pd.date_range('2022-01-01', periods=n_samples, freq='5min'),
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples)
    })
    
    y = pd.Series((np.random.randn(n_samples) > 0).astype(int))
    
    # 创建模型工厂
    model_factory = create_model_factory('lgbm')
    
    # 执行验证
    results = validator.comprehensive_validation(X, y, model_factory)
    
    # 保存报告
    report_path = validator.save_validation_report()
    
    print(f"验证完成，报告保存至: {report_path}")
    print(f"综合建议数量: {len(results.get('recommendations', []))}")
    
    return results

if __name__ == "__main__":
    main()