"""
DipMaster Enhanced V4 特征验证和质量评估工具
专门验证机器学习特征的质量、稳定性和有效性
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class FeatureValidator:
    """特征验证器"""
    
    def __init__(self, features_file: str, metadata_file: str):
        self.features_df = pd.read_parquet(features_file)
        with open(metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded feature set: {len(self.features_df)} samples, {len(self.features_df.columns)} features")
    
    def validate_feature_quality(self) -> Dict:
        """全面验证特征质量"""
        validation_results = {}
        
        # 1. 基础数据质量检查
        validation_results['data_quality'] = self._check_data_quality()
        
        # 2. 特征分布检查
        validation_results['feature_distributions'] = self._check_feature_distributions()
        
        # 3. 标签分布检查
        validation_results['label_distributions'] = self._check_label_distributions()
        
        # 4. 前视偏差检查
        validation_results['lookahead_bias_check'] = self._check_lookahead_bias()
        
        # 5. 特征相关性分析
        validation_results['correlation_analysis'] = self._analyze_correlations()
        
        # 6. 时间稳定性检查
        validation_results['temporal_stability'] = self._check_temporal_stability()
        
        return validation_results
    
    def _check_data_quality(self) -> Dict:
        """检查基础数据质量"""
        quality_stats = {}
        
        # 缺失值统计
        missing_stats = {}
        for col in self.features_df.columns:
            missing_pct = self.features_df[col].isnull().sum() / len(self.features_df)
            if missing_pct > 0:
                missing_stats[col] = missing_pct
        
        quality_stats['missing_values'] = missing_stats
        
        # 无穷值检查
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        inf_stats = {}
        for col in numeric_cols:
            inf_count = np.isinf(self.features_df[col]).sum()
            if inf_count > 0:
                inf_stats[col] = inf_count
        
        quality_stats['infinite_values'] = inf_stats
        
        # 重复行检查
        duplicate_count = self.features_df.duplicated().sum()
        quality_stats['duplicate_rows'] = duplicate_count
        
        # 时间序列完整性
        timestamps = pd.to_datetime(self.features_df['timestamp'])
        expected_intervals = pd.date_range(
            start=timestamps.min(),
            end=timestamps.max(),
            freq='5T'
        )
        quality_stats['time_completeness'] = len(timestamps.unique()) / len(expected_intervals)
        
        return quality_stats
    
    def _check_feature_distributions(self) -> Dict:
        """检查特征分布"""
        feature_stats = {}
        
        # DipMaster核心特征分析
        dipmaster_features = self.metadata['feature_categories']['dipmaster_core']
        for feature in dipmaster_features:
            if feature in self.features_df.columns:
                series = self.features_df[feature]
                feature_stats[feature] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()),
                    'min': float(series.min()),
                    'max': float(series.max()),
                    'skewness': float(series.skew()),
                    'kurtosis': float(series.kurtosis()),
                    'zero_ratio': float((series == 0).sum() / len(series))
                }
        
        # 信号强度分析
        if 'dipmaster_signal_strength' in self.features_df.columns:
            signal_strength = self.features_df['dipmaster_signal_strength']
            feature_stats['signal_strength_analysis'] = {
                'mean_strength': float(signal_strength.mean()),
                'high_signal_ratio': float((signal_strength > 0.7).sum() / len(signal_strength)),
                'medium_signal_ratio': float(((signal_strength > 0.4) & (signal_strength <= 0.7)).sum() / len(signal_strength)),
                'low_signal_ratio': float((signal_strength <= 0.4).sum() / len(signal_strength))
            }
        
        return feature_stats
    
    def _check_label_distributions(self) -> Dict:
        """检查标签分布"""
        label_stats = {}
        
        # 未来收益率分布
        for horizon in [15, 30, 60]:
            return_col = f'future_return_{horizon}m'
            profit_col = f'is_profitable_{horizon}m'
            
            if return_col in self.features_df.columns:
                returns = self.features_df[return_col].dropna()
                profits = self.features_df[profit_col].dropna()
                
                label_stats[f'{horizon}m_returns'] = {
                    'mean_return': float(returns.mean()),
                    'std_return': float(returns.std()),
                    'positive_return_ratio': float((returns > 0).sum() / len(returns)),
                    'profit_probability': float(profits.mean()),
                    'extreme_gain_ratio': float((returns > 0.02).sum() / len(returns)),  # >2%收益率
                    'extreme_loss_ratio': float((returns < -0.02).sum() / len(returns))  # <-2%收益率
                }
        
        # 目标达成率分析
        for target in ['0.6%', '1.2%']:
            target_col = f'hits_target_{target}'
            if target_col in self.features_df.columns:
                target_rate = self.features_df[target_col].mean()
                label_stats[f'target_{target}_hit_rate'] = float(target_rate)
        
        # 止损触发率
        if 'hits_stop_loss' in self.features_df.columns:
            stop_loss_rate = self.features_df['hits_stop_loss'].mean()
            label_stats['stop_loss_hit_rate'] = float(stop_loss_rate)
        
        return label_stats
    
    def _check_lookahead_bias(self) -> Dict:
        """检查前视偏差"""
        bias_check = {}
        
        # 检查特征与未来标签的异常高相关性
        label_cols = [col for col in self.features_df.columns if 'future_' in col or 'hits_' in col]
        # 只选择数值型特征
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in label_cols and col not in ['timestamp', 'symbol']]
        
        suspicious_correlations = []
        
        for label_col in label_cols:
            if label_col in self.features_df.columns:
                for feature_col in feature_cols:
                    if feature_col in self.features_df.columns:
                        corr = self.features_df[feature_col].corr(self.features_df[label_col])
                        if abs(corr) > 0.8:  # 异常高相关性阈值
                            suspicious_correlations.append({
                                'feature': feature_col,
                                'label': label_col,
                                'correlation': float(corr)
                            })
        
        bias_check['suspicious_correlations'] = suspicious_correlations
        bias_check['bias_detected'] = len(suspicious_correlations) > 0
        
        return bias_check
    
    def _analyze_correlations(self) -> Dict:
        """分析特征相关性"""
        corr_analysis = {}
        
        # 计算数值特征相关性矩阵
        numeric_features = self.features_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_features if not any(x in col for x in ['future_', 'hits_', 'is_profitable'])]
        
        if len(feature_cols) > 1:
            corr_matrix = self.features_df[feature_cols].corr()
            
            # 找出高度相关的特征对
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': float(corr_value)
                        })
            
            corr_analysis['high_correlation_pairs'] = high_corr_pairs
            corr_analysis['multicollinearity_detected'] = len(high_corr_pairs) > 0
            
            # 平均相关性
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            corr_analysis['average_correlation'] = float(upper_triangle.stack().mean())
        
        return corr_analysis
    
    def _check_temporal_stability(self) -> Dict:
        """检查时间稳定性"""
        stability_analysis = {}
        
        # 按时间分段分析特征分布
        self.features_df['timestamp'] = pd.to_datetime(self.features_df['timestamp'])
        
        # 分为4个时间段
        periods = pd.qcut(self.features_df['timestamp'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # 检查核心特征的时间稳定性
        dipmaster_features = ['rsi', 'bb_position', 'volume_ratio', 'dipmaster_signal_strength']
        
        for feature in dipmaster_features:
            if feature in self.features_df.columns:
                period_stats = {}
                for period in ['Q1', 'Q2', 'Q3', 'Q4']:
                    period_data = self.features_df[periods == period][feature]
                    period_stats[period] = {
                        'mean': float(period_data.mean()),
                        'std': float(period_data.std())
                    }
                
                # 计算变异系数
                means = [period_stats[p]['mean'] for p in ['Q1', 'Q2', 'Q3', 'Q4']]
                cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
                
                stability_analysis[feature] = {
                    'period_stats': period_stats,
                    'coefficient_of_variation': float(cv),
                    'stability_rating': 'stable' if cv < 0.1 else 'moderate' if cv < 0.3 else 'unstable'
                }
        
        return stability_analysis
    
    def generate_feature_importance_proxy(self) -> Dict:
        """生成特征重要性代理指标"""
        importance_analysis = {}
        
        # 使用信息价值(IV)作为特征重要性的代理
        label_col = 'is_profitable_15m'
        if label_col in self.features_df.columns:
            feature_cols = [col for col in self.features_df.columns 
                          if col not in ['timestamp', 'symbol', label_col] 
                          and not any(x in col for x in ['future_', 'hits_', 'is_profitable'])]
            
            iv_scores = {}
            for feature in feature_cols:
                if feature in self.features_df.columns:
                    try:
                        iv_score = self._calculate_information_value(
                            self.features_df[feature], 
                            self.features_df[label_col]
                        )
                        iv_scores[feature] = iv_score
                    except:
                        iv_scores[feature] = 0.0
            
            # 排序特征重要性
            sorted_features = sorted(iv_scores.items(), key=lambda x: x[1], reverse=True)
            importance_analysis['feature_importance_ranking'] = sorted_features[:20]  # Top 20
            importance_analysis['top_features'] = [item[0] for item in sorted_features[:10]]
        
        return importance_analysis
    
    def _calculate_information_value(self, feature: pd.Series, target: pd.Series) -> float:
        """计算信息价值(Information Value)"""
        # 将特征分箱
        if feature.dtype in ['object', 'category']:
            bins = feature.value_counts().index[:10]  # 取前10个类别
        else:
            try:
                bins = pd.qcut(feature, q=10, duplicates='drop').cat.categories
            except:
                bins = pd.cut(feature, bins=10).cat.categories
        
        iv = 0
        total_good = target.sum()
        total_bad = len(target) - total_good
        
        if total_good == 0 or total_bad == 0:
            return 0
        
        for bin_range in bins:
            if feature.dtype in ['object', 'category']:
                bin_mask = feature == bin_range
            else:
                bin_mask = (feature >= bin_range.left) & (feature <= bin_range.right)
            
            good_count = target[bin_mask].sum()
            bad_count = bin_mask.sum() - good_count
            
            if good_count > 0 and bad_count > 0:
                good_rate = good_count / total_good
                bad_rate = bad_count / total_bad
                woe = np.log(good_rate / bad_rate)
                iv += (good_rate - bad_rate) * woe
        
        return iv
    
    def create_validation_report(self, output_path: str) -> str:
        """创建完整的验证报告"""
        validation_results = self.validate_feature_quality()
        importance_analysis = self.generate_feature_importance_proxy()
        
        # 合并所有结果
        full_report = {
            'validation_summary': {
                'feature_set_version': self.metadata['version'],
                'validation_timestamp': datetime.now().isoformat(),
                'total_samples': len(self.features_df),
                'total_features': len(self.features_df.columns),
                'data_date_range': {
                    'start': str(self.features_df['timestamp'].min()),
                    'end': str(self.features_df['timestamp'].max())
                }
            },
            'validation_results': validation_results,
            'feature_importance': importance_analysis,
            'overall_quality_score': self._calculate_overall_quality_score(validation_results)
        }
        
        # 保存报告
        report_file = Path(output_path) / f"feature_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)
        
        # 创建简化的文本摘要
        summary_file = Path(output_path) / f"feature_validation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self._create_text_summary(full_report, summary_file)
        
        return str(report_file)
    
    def _calculate_overall_quality_score(self, validation_results: Dict) -> float:
        """计算总体质量分数"""
        score = 100.0
        
        # 扣分项目
        # 缺失值扣分
        missing_ratio = len(validation_results['data_quality']['missing_values']) / len(self.features_df.columns)
        score -= missing_ratio * 20
        
        # 前视偏差扣分
        if validation_results['lookahead_bias_check']['bias_detected']:
            score -= 30
        
        # 多重共线性扣分
        if validation_results['correlation_analysis'].get('multicollinearity_detected', False):
            high_corr_count = len(validation_results['correlation_analysis']['high_correlation_pairs'])
            score -= min(high_corr_count * 2, 20)
        
        # 时间不稳定性扣分
        unstable_features = sum(1 for feature_stability in validation_results['temporal_stability'].values()
                               if feature_stability['stability_rating'] == 'unstable')
        score -= unstable_features * 5
        
        return max(score, 0)
    
    def _create_text_summary(self, report: Dict, output_file: str):
        """创建文本摘要"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("DipMaster Enhanced V4 Feature Validation Summary\n")
            f.write("="*60 + "\n\n")
            
            # 基本信息
            f.write(f"Validation Date: {report['validation_summary']['validation_timestamp']}\n")
            f.write(f"Total Samples: {report['validation_summary']['total_samples']:,}\n")
            f.write(f"Total Features: {report['validation_summary']['total_features']}\n")
            f.write(f"Overall Quality Score: {report['overall_quality_score']:.1f}/100\n\n")
            
            # 数据质量
            f.write("Data Quality Assessment:\n")
            f.write("-" * 30 + "\n")
            missing_values = report['validation_results']['data_quality']['missing_values']
            if missing_values:
                f.write(f"Missing Values Detected: {len(missing_values)} features\n")
                for col, ratio in missing_values.items():
                    f.write(f"  - {col}: {ratio:.2%}\n")
            else:
                f.write("No missing values detected\n")
            
            f.write(f"Time Completeness: {report['validation_results']['data_quality']['time_completeness']:.2%}\n\n")
            
            # 标签分布
            f.write("Label Distribution Analysis:\n")
            f.write("-" * 30 + "\n")
            label_dist = report['validation_results']['label_distributions']
            for key, value in label_dist.items():
                if 'hit_rate' in key:
                    f.write(f"{key}: {value:.2%}\n")
                elif isinstance(value, dict) and 'profit_probability' in value:
                    f.write(f"{key} profit probability: {value['profit_probability']:.2%}\n")
            
            f.write("\n")
            
            # 前视偏差检查
            f.write("Lookahead Bias Check:\n")
            f.write("-" * 30 + "\n")
            bias_check = report['validation_results']['lookahead_bias_check']
            if bias_check['bias_detected']:
                f.write("WARNING: Potential lookahead bias detected!\n")
                for corr in bias_check['suspicious_correlations'][:5]:
                    f.write(f"  - {corr['feature']} -> {corr['label']}: {corr['correlation']:.3f}\n")
            else:
                f.write("No lookahead bias detected\n")
            
            f.write("\n")
            
            # 特征重要性
            f.write("Top Features (by Information Value):\n")
            f.write("-" * 30 + "\n")
            if 'feature_importance_ranking' in report['feature_importance']:
                for i, (feature, score) in enumerate(report['feature_importance']['feature_importance_ranking'][:10], 1):
                    f.write(f"{i:2d}. {feature}: {score:.4f}\n")
            
            f.write("\n")
            
            # 建议
            f.write("Recommendations:\n")
            f.write("-" * 30 + "\n")
            
            if report['overall_quality_score'] >= 90:
                f.write("✓ Excellent feature quality. Ready for model training.\n")
            elif report['overall_quality_score'] >= 80:
                f.write("✓ Good feature quality with minor issues to address.\n")
            elif report['overall_quality_score'] >= 70:
                f.write("⚠ Moderate feature quality. Consider improvements before training.\n")
            else:
                f.write("✗ Poor feature quality. Significant improvements needed.\n")
            
            if bias_check['bias_detected']:
                f.write("- Address potential lookahead bias in suspicious features\n")
            
            if len(missing_values) > 0:
                f.write("- Handle missing values in affected features\n")
            
            if report['validation_results']['correlation_analysis'].get('multicollinearity_detected', False):
                f.write("- Consider removing highly correlated features\n")

def main():
    """主函数"""
    # 获取最新的特征文件
    data_dir = Path("G:/Github/Quant/DipMaster-Trading-System/data")
    
    # 查找最新的特征文件
    feature_files = list(data_dir.glob("dipmaster_v4_features_*.parquet"))
    metadata_files = list(data_dir.glob("FeatureSet_*.json"))
    
    if not feature_files or not metadata_files:
        print("No feature files found!")
        return
    
    # 使用最新的文件
    latest_feature_file = max(feature_files, key=lambda x: x.stat().st_mtime)
    latest_metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Validating feature set:")
    print(f"Features: {latest_feature_file}")
    print(f"Metadata: {latest_metadata_file}")
    
    # 创建验证器
    validator = FeatureValidator(str(latest_feature_file), str(latest_metadata_file))
    
    # 生成验证报告
    report_file = validator.create_validation_report(str(data_dir))
    
    print(f"\nValidation completed!")
    print(f"Report saved: {report_file}")

if __name__ == "__main__":
    main()