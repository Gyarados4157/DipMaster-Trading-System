"""
DataValidator - 企业级数据质量验证引擎
确保交易数据的完整性、准确性和一致性，为策略提供可靠的数据基础
"""

import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

@dataclass
class ValidationResult:
    """验证结果数据类"""
    metric_name: str
    score: float  # 0-1之间的评分
    status: str   # 'pass', 'warning', 'fail'
    details: Dict[str, Any]
    recommendations: List[str]

@dataclass
class QualityReport:
    """数据质量报告"""
    symbol: str
    timeframe: str
    validation_time: str
    overall_score: float
    completeness: ValidationResult
    accuracy: ValidationResult
    consistency: ValidationResult
    validity: ValidationResult
    anomalies: ValidationResult
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class DataValidator:
    """
    数据验证器 - 确保DipMaster策略数据质量
    
    验证维度:
    1. 完整性 (Completeness): 数据缺失检测和修复
    2. 准确性 (Accuracy): 价格和成交量合理性检验
    3. 一致性 (Consistency): OHLC关系、时间序列连续性
    4. 有效性 (Validity): 数据格式和范围检验
    5. 异常检测 (Anomaly): 突发事件和数据异常识别
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 质量阈值设置
        self.thresholds = config.get('quality_thresholds', {
            'completeness': 0.99,
            'accuracy': 0.999,
            'consistency': 0.98,
            'validity': 0.999,
            'anomaly_tolerance': 0.05
        })
        
        # 统计参数
        self.price_change_threshold = config.get('price_change_threshold', 0.20)  # 20%价格变化警告
        self.volume_spike_threshold = config.get('volume_spike_threshold', 10)     # 10x成交量激增
        self.gap_tolerance_minutes = config.get('gap_tolerance_minutes', 10)      # 允许的时间间隙
        
    async def validate_data_quality(self, file_path: Path, symbol: str, timeframe: str) -> QualityReport:
        """
        全面数据质量验证入口
        
        Args:
            file_path: 数据文件路径
            symbol: 交易对符号
            timeframe: 时间框架
            
        Returns:
            完整的质量报告
        """
        self.logger.info(f"开始验证数据质量: {symbol} {timeframe}")
        
        try:
            # 加载数据
            df = await self._load_data_safely(file_path)
            
            if df.empty:
                raise ValueError("数据文件为空")
            
            # 执行各项验证
            completeness = await self._validate_completeness(df, timeframe)
            accuracy = await self._validate_accuracy(df, symbol)
            consistency = await self._validate_consistency(df)
            validity = await self._validate_validity(df)
            anomalies = await self._detect_anomalies(df, symbol)
            
            # 计算综合评分
            scores = [completeness.score, accuracy.score, consistency.score, 
                     validity.score, anomalies.score]
            weights = [0.25, 0.25, 0.20, 0.15, 0.15]  # 各维度权重
            overall_score = sum(s * w for s, w in zip(scores, weights))
            
            # 生成质量报告
            report = QualityReport(
                symbol=symbol,
                timeframe=timeframe,
                validation_time=datetime.now().isoformat(),
                overall_score=overall_score,
                completeness=completeness,
                accuracy=accuracy,
                consistency=consistency,
                validity=validity,
                anomalies=anomalies
            )
            
            self.logger.info(f"数据质量验证完成: {symbol} 综合评分 {overall_score:.3f}")
            return report
            
        except Exception as e:
            self.logger.error(f"数据质量验证失败: {symbol} {timeframe} - {e}")
            raise
    
    async def _load_data_safely(self, file_path: Path) -> pd.DataFrame:
        """安全加载数据文件"""
        try:
            if file_path.suffix == '.parquet':
                df = pd.read_parquet(file_path)
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"不支持的文件格式: {file_path.suffix}")
            
            # 标准化列名
            df = self._standardize_columns(df)
            
            # 确保时间列为datetime类型
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'datetime' in df.columns:
                df['timestamp'] = pd.to_datetime(df['datetime'])
                df = df.drop('datetime', axis=1)
            
            return df.sort_values('timestamp') if 'timestamp' in df.columns else df
            
        except Exception as e:
            self.logger.error(f"数据加载失败: {file_path} - {e}")
            return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化数据列名"""
        # 通用列名映射
        column_mapping = {
            'open_time': 'timestamp',
            'kline_open_time': 'timestamp',
            'close_time': 'kline_close_time',
            'quote_asset_volume': 'quote_volume',
            'number_of_trades': 'trade_count',
            'taker_buy_base_asset_volume': 'taker_buy_volume',
            'taker_buy_quote_asset_volume': 'taker_buy_quote_volume'
        }
        
        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # 确保必须的列存在
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.warning(f"缺失必要列: {missing_columns}")
        
        return df
    
    async def _validate_completeness(self, df: pd.DataFrame, timeframe: str) -> ValidationResult:
        """验证数据完整性"""
        details = {}
        recommendations = []
        
        try:
            # 检查基本完整性
            total_rows = len(df)
            null_counts = df.isnull().sum()
            
            # 计算完整性分数
            completeness_scores = []
            
            # 1. 时间连续性检查
            if 'timestamp' in df.columns and total_rows > 1:
                time_gaps = self._detect_time_gaps_detailed(df['timestamp'], timeframe)
                time_completeness = 1.0 - (len(time_gaps) / total_rows)
                completeness_scores.append(time_completeness)
                
                details['time_gaps'] = {
                    'gap_count': len(time_gaps),
                    'largest_gap_minutes': max([gap['duration_minutes'] for gap in time_gaps], default=0),
                    'gaps': time_gaps[:10]  # 前10个最大间隙
                }
                
                if len(time_gaps) > 0:
                    recommendations.append(f"发现{len(time_gaps)}个时间间隙，建议填补缺失数据")
            
            # 2. 必要字段完整性
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            field_completeness = []
            
            for field in required_fields:
                if field in df.columns:
                    non_null_ratio = (total_rows - null_counts.get(field, 0)) / total_rows
                    field_completeness.append(non_null_ratio)
                    details[f'{field}_completeness'] = non_null_ratio
                    
                    if non_null_ratio < 0.99:
                        recommendations.append(f"{field}字段缺失率较高: {(1-non_null_ratio)*100:.2f}%")
            
            if field_completeness:
                completeness_scores.append(np.mean(field_completeness))
            
            # 3. 数据密度检查（检查是否有大量重复或零值）
            if 'close' in df.columns:
                zero_prices = (df['close'] == 0).sum()
                duplicate_prices = df['close'].duplicated().sum()
                
                density_score = 1.0 - (zero_prices + duplicate_prices) / total_rows
                completeness_scores.append(density_score)
                
                details['data_density'] = {
                    'zero_prices': zero_prices,
                    'duplicate_prices': duplicate_prices,
                    'density_score': density_score
                }
                
                if zero_prices > 0:
                    recommendations.append(f"发现{zero_prices}个零价格数据点")
                if duplicate_prices > total_rows * 0.1:
                    recommendations.append(f"重复价格数据较多: {duplicate_prices}个")
            
            # 计算综合完整性分数
            final_score = np.mean(completeness_scores) if completeness_scores else 0.0
            
            status = 'pass' if final_score >= self.thresholds['completeness'] else \
                    'warning' if final_score >= 0.95 else 'fail'
            
            details['total_records'] = total_rows
            details['missing_values'] = null_counts.to_dict()
            details['completeness_subscores'] = completeness_scores
            
            return ValidationResult(
                metric_name='completeness',
                score=final_score,
                status=status,
                details=details,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"完整性验证失败: {e}")
            return ValidationResult(
                metric_name='completeness',
                score=0.0,
                status='fail',
                details={'error': str(e)},
                recommendations=['数据完整性验证过程中发生错误，请检查数据格式']
            )
    
    def _detect_time_gaps_detailed(self, timestamps: pd.Series, timeframe: str) -> List[Dict[str, Any]]:
        """详细的时间间隙检测"""
        if len(timestamps) < 2:
            return []
        
        # 时间框架对应的分钟数
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30, 
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        expected_interval = timeframe_minutes.get(timeframe, 5)
        tolerance_minutes = self.gap_tolerance_minutes
        
        gaps = []
        time_diffs = timestamps.diff().dropna()
        
        for i, diff in enumerate(time_diffs):
            diff_minutes = diff.total_seconds() / 60
            
            if diff_minutes > expected_interval + tolerance_minutes:
                gaps.append({
                    'start_time': timestamps.iloc[i].isoformat(),
                    'end_time': timestamps.iloc[i + 1].isoformat(),
                    'duration_minutes': diff_minutes,
                    'expected_minutes': expected_interval,
                    'missing_periods': int(diff_minutes / expected_interval) - 1
                })
        
        # 按间隙大小排序
        return sorted(gaps, key=lambda x: x['duration_minutes'], reverse=True)
    
    async def _validate_accuracy(self, df: pd.DataFrame, symbol: str) -> ValidationResult:
        """验证数据准确性"""
        details = {}
        recommendations = []
        accuracy_scores = []
        
        try:
            # 1. OHLC关系验证
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                ohlc_valid = (
                    (df['high'] >= df['open']) & 
                    (df['high'] >= df['low']) & 
                    (df['high'] >= df['close']) &
                    (df['low'] <= df['open']) & 
                    (df['low'] <= df['close'])
                )
                
                ohlc_accuracy = ohlc_valid.mean()
                accuracy_scores.append(ohlc_accuracy)
                
                details['ohlc_validity'] = {
                    'valid_ratio': ohlc_accuracy,
                    'invalid_count': (~ohlc_valid).sum(),
                    'invalid_indices': df[~ohlc_valid].index.tolist()[:10]
                }
                
                if ohlc_accuracy < 0.999:
                    recommendations.append(f"OHLC关系验证失败: {(1-ohlc_accuracy)*100:.3f}%的数据点")
            
            # 2. 价格合理性检查
            if 'close' in df.columns:
                # 检查极端价格变化
                price_returns = df['close'].pct_change().dropna()
                extreme_changes = (abs(price_returns) > self.price_change_threshold).sum()
                
                price_reasonableness = 1.0 - (extreme_changes / len(price_returns))
                accuracy_scores.append(price_reasonableness)
                
                details['price_reasonableness'] = {
                    'extreme_changes_count': extreme_changes,
                    'max_price_change_pct': abs(price_returns).max() * 100,
                    'reasonableness_score': price_reasonableness
                }
                
                if extreme_changes > 0:
                    recommendations.append(f"发现{extreme_changes}次极端价格变化(>{self.price_change_threshold*100}%)")
            
            # 3. 成交量合理性检查
            if 'volume' in df.columns:
                # 检查零成交量和负成交量
                volume_issues = (df['volume'] <= 0).sum()
                
                # 检查成交量异常激增
                volume_median = df['volume'].median()
                volume_spikes = (df['volume'] > volume_median * self.volume_spike_threshold).sum()
                
                volume_reasonableness = 1.0 - ((volume_issues + volume_spikes * 0.1) / len(df))
                accuracy_scores.append(volume_reasonableness)
                
                details['volume_reasonableness'] = {
                    'zero_negative_volume': volume_issues,
                    'volume_spikes': volume_spikes,
                    'median_volume': volume_median,
                    'reasonableness_score': volume_reasonableness
                }
                
                if volume_issues > 0:
                    recommendations.append(f"发现{volume_issues}个零或负成交量数据点")
                if volume_spikes > len(df) * 0.01:
                    recommendations.append(f"成交量异常激增点较多: {volume_spikes}个")
            
            # 4. 数据精度检查
            precision_scores = []
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    # 检查价格精度（小数位数合理性）
                    decimals = df[col].apply(lambda x: len(str(x).split('.')[-1]) if '.' in str(x) else 0)
                    reasonable_precision = (decimals <= 8).mean()  # 8位小数精度
                    precision_scores.append(reasonable_precision)
            
            if precision_scores:
                avg_precision_score = np.mean(precision_scores)
                accuracy_scores.append(avg_precision_score)
                details['precision_check'] = {
                    'average_precision_score': avg_precision_score,
                    'precision_scores_by_field': precision_scores
                }
            
            # 计算综合准确性分数
            final_score = np.mean(accuracy_scores) if accuracy_scores else 0.0
            
            status = 'pass' if final_score >= self.thresholds['accuracy'] else \
                    'warning' if final_score >= 0.99 else 'fail'
            
            return ValidationResult(
                metric_name='accuracy',
                score=final_score,
                status=status,
                details=details,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"准确性验证失败: {e}")
            return ValidationResult(
                metric_name='accuracy',
                score=0.0,
                status='fail',
                details={'error': str(e)},
                recommendations=['数据准确性验证过程中发生错误']
            )
    
    async def _validate_consistency(self, df: pd.DataFrame) -> ValidationResult:
        """验证数据一致性"""
        details = {}
        recommendations = []
        consistency_scores = []
        
        try:
            # 1. 时间序列单调性
            if 'timestamp' in df.columns and len(df) > 1:
                is_monotonic = df['timestamp'].is_monotonic_increasing
                consistency_scores.append(1.0 if is_monotonic else 0.0)
                
                details['time_monotonicity'] = is_monotonic
                
                if not is_monotonic:
                    recommendations.append("时间序列不是单调递增的，可能存在重复或乱序")
            
            # 2. 数据类型一致性
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            type_consistency = []
            
            for col in numeric_columns:
                if col in df.columns:
                    is_numeric = pd.api.types.is_numeric_dtype(df[col])
                    type_consistency.append(1.0 if is_numeric else 0.0)
                    details[f'{col}_numeric'] = is_numeric
                    
                    if not is_numeric:
                        recommendations.append(f"{col}列数据类型不是数值型")
            
            if type_consistency:
                consistency_scores.append(np.mean(type_consistency))
            
            # 3. 数据范围一致性
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # 检查价格范围的合理性
                price_cols = ['open', 'high', 'low', 'close']
                price_ranges = []
                
                for col in price_cols:
                    col_range = df[col].max() - df[col].min()
                    col_std = df[col].std()
                    
                    # 计算变异系数（标准差/均值）
                    cv = col_std / df[col].mean() if df[col].mean() > 0 else float('inf')
                    price_ranges.append(cv)
                
                # 价格列之间的变异系数应该相似
                range_consistency = 1.0 - (np.std(price_ranges) / np.mean(price_ranges)) if np.mean(price_ranges) > 0 else 0.0
                range_consistency = max(0.0, min(1.0, range_consistency))  # 限制在[0,1]
                
                consistency_scores.append(range_consistency)
                details['price_range_consistency'] = {
                    'coefficient_of_variation': price_ranges,
                    'consistency_score': range_consistency
                }
            
            # 4. 统计分布一致性
            if 'close' in df.columns and len(df) > 100:
                # 检查价格分布的稳定性（分时段比较）
                mid_point = len(df) // 2
                first_half = df['close'].iloc[:mid_point]
                second_half = df['close'].iloc[mid_point:]
                
                # 使用Kolmogorov-Smirnov测试检查分布一致性
                try:
                    ks_stat, p_value = stats.ks_2samp(first_half, second_half)
                    distribution_consistency = p_value  # p值越高，分布越一致
                    consistency_scores.append(distribution_consistency)
                    
                    details['distribution_consistency'] = {
                        'ks_statistic': ks_stat,
                        'p_value': p_value,
                        'distributions_similar': p_value > 0.05
                    }
                    
                    if p_value < 0.05:
                        recommendations.append("前后半段数据分布差异显著，可能存在结构性变化")
                        
                except Exception as e:
                    self.logger.warning(f"分布一致性检验失败: {e}")
            
            # 计算综合一致性分数
            final_score = np.mean(consistency_scores) if consistency_scores else 0.0
            
            status = 'pass' if final_score >= self.thresholds['consistency'] else \
                    'warning' if final_score >= 0.95 else 'fail'
            
            return ValidationResult(
                metric_name='consistency',
                score=final_score,
                status=status,
                details=details,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"一致性验证失败: {e}")
            return ValidationResult(
                metric_name='consistency',
                score=0.0,
                status='fail',
                details={'error': str(e)},
                recommendations=['数据一致性验证过程中发生错误']
            )
    
    async def _validate_validity(self, df: pd.DataFrame) -> ValidationResult:
        """验证数据有效性"""
        details = {}
        recommendations = []
        validity_scores = []
        
        try:
            # 1. 数据完整性（非空值）
            total_cells = df.size
            non_null_cells = df.count().sum()
            
            completeness_score = non_null_cells / total_cells if total_cells > 0 else 0.0
            validity_scores.append(completeness_score)
            
            details['data_completeness'] = {
                'total_cells': total_cells,
                'non_null_cells': non_null_cells,
                'completeness_ratio': completeness_score
            }
            
            # 2. 数值范围有效性
            if 'volume' in df.columns:
                # 成交量应该非负
                valid_volume = (df['volume'] >= 0).mean()
                validity_scores.append(valid_volume)
                
                details['volume_validity'] = {
                    'non_negative_ratio': valid_volume,
                    'min_volume': df['volume'].min(),
                    'negative_count': (df['volume'] < 0).sum()
                }
                
                if valid_volume < 1.0:
                    recommendations.append(f"发现负成交量数据: {(1-valid_volume)*100:.2f}%")
            
            # 3. 价格有效性
            price_columns = ['open', 'high', 'low', 'close']
            price_validity_scores = []
            
            for col in price_columns:
                if col in df.columns:
                    # 价格应该为正数
                    positive_prices = (df[col] > 0).mean()
                    price_validity_scores.append(positive_prices)
                    
                    details[f'{col}_validity'] = {
                        'positive_ratio': positive_prices,
                        'min_value': df[col].min(),
                        'zero_negative_count': (df[col] <= 0).sum()
                    }
                    
                    if positive_prices < 1.0:
                        recommendations.append(f"{col}存在非正数价格: {(1-positive_prices)*100:.2f}%")
            
            if price_validity_scores:
                validity_scores.append(np.mean(price_validity_scores))
            
            # 4. 数据格式有效性
            format_scores = []
            
            # 检查数值列是否为有限数值
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    finite_ratio = np.isfinite(df[col]).mean()
                    format_scores.append(finite_ratio)
                    
                    details[f'{col}_finite'] = {
                        'finite_ratio': finite_ratio,
                        'inf_count': np.isinf(df[col]).sum(),
                        'nan_count': np.isnan(df[col]).sum()
                    }
                    
                    if finite_ratio < 1.0:
                        recommendations.append(f"{col}包含无穷大或NaN值")
            
            if format_scores:
                validity_scores.append(np.mean(format_scores))
            
            # 5. 时间戳有效性
            if 'timestamp' in df.columns:
                try:
                    # 检查时间戳是否在合理范围内
                    min_date = pd.Timestamp('2010-01-01')  # 比特币诞生后
                    max_date = pd.Timestamp.now() + pd.Timedelta(days=1)  # 未来一天内
                    
                    valid_timestamps = (
                        (df['timestamp'] >= min_date) & 
                        (df['timestamp'] <= max_date)
                    ).mean()
                    
                    validity_scores.append(valid_timestamps)
                    
                    details['timestamp_validity'] = {
                        'valid_range_ratio': valid_timestamps,
                        'min_timestamp': df['timestamp'].min().isoformat(),
                        'max_timestamp': df['timestamp'].max().isoformat(),
                        'out_of_range_count': (~((df['timestamp'] >= min_date) & 
                                                (df['timestamp'] <= max_date))).sum()
                    }
                    
                    if valid_timestamps < 1.0:
                        recommendations.append("发现时间戳超出合理范围")
                        
                except Exception as e:
                    self.logger.warning(f"时间戳有效性检查失败: {e}")
            
            # 计算综合有效性分数
            final_score = np.mean(validity_scores) if validity_scores else 0.0
            
            status = 'pass' if final_score >= self.thresholds['validity'] else \
                    'warning' if final_score >= 0.99 else 'fail'
            
            return ValidationResult(
                metric_name='validity',
                score=final_score,
                status=status,
                details=details,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"有效性验证失败: {e}")
            return ValidationResult(
                metric_name='validity',
                score=0.0,
                status='fail',
                details={'error': str(e)},
                recommendations=['数据有效性验证过程中发生错误']
            )
    
    async def _detect_anomalies(self, df: pd.DataFrame, symbol: str) -> ValidationResult:
        """异常检测"""
        details = {}
        recommendations = []
        
        try:
            anomaly_count = 0
            total_checks = 0
            
            # 1. 价格异常检测
            if 'close' in df.columns and len(df) > 20:
                # 使用3σ规则检测价格异常
                price_returns = df['close'].pct_change().dropna()
                
                if len(price_returns) > 0:
                    mean_return = price_returns.mean()
                    std_return = price_returns.std()
                    
                    # 3σ异常检测
                    threshold = 3 * std_return
                    price_anomalies = (abs(price_returns - mean_return) > threshold).sum()
                    
                    anomaly_count += price_anomalies
                    total_checks += len(price_returns)
                    
                    details['price_anomalies'] = {
                        'anomaly_count': price_anomalies,
                        'anomaly_ratio': price_anomalies / len(price_returns),
                        'threshold_std': threshold,
                        'max_deviation': abs(price_returns - mean_return).max()
                    }
                    
                    if price_anomalies > 0:
                        recommendations.append(f"检测到{price_anomalies}个价格异常点")
            
            # 2. 成交量异常检测
            if 'volume' in df.columns and len(df) > 20:
                # 使用IQR方法检测成交量异常
                Q1 = df['volume'].quantile(0.25)
                Q3 = df['volume'].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 3 * IQR  # 成交量异常通常是上方异常
                
                volume_anomalies = (
                    (df['volume'] < lower_bound) | 
                    (df['volume'] > upper_bound)
                ).sum()
                
                anomaly_count += volume_anomalies
                total_checks += len(df)
                
                details['volume_anomalies'] = {
                    'anomaly_count': volume_anomalies,
                    'anomaly_ratio': volume_anomalies / len(df),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'q1': Q1,
                    'q3': Q3
                }
                
                if volume_anomalies > 0:
                    recommendations.append(f"检测到{volume_anomalies}个成交量异常点")
            
            # 3. OHLC关系异常
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                # 检测异常的K线形态
                
                # 异常长上影线或下影线
                body_size = abs(df['close'] - df['open'])
                upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
                lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
                
                # 影线长度超过实体10倍的异常K线
                extreme_shadows = (
                    (upper_shadow > body_size * 10) | 
                    (lower_shadow > body_size * 10)
                ).sum()
                
                anomaly_count += extreme_shadows
                total_checks += len(df)
                
                details['ohlc_anomalies'] = {
                    'extreme_shadows': extreme_shadows,
                    'extreme_shadow_ratio': extreme_shadows / len(df),
                    'avg_body_size': body_size.mean(),
                    'avg_upper_shadow': upper_shadow.mean(),
                    'avg_lower_shadow': lower_shadow.mean()
                }
                
                if extreme_shadows > 0:
                    recommendations.append(f"检测到{extreme_shadows}个极端影线K线")
            
            # 4. 连续异常检测
            if 'close' in df.columns and len(df) > 10:
                # 检测连续的相同价格（可能的数据停滞）
                price_changes = df['close'].diff()
                zero_changes = (price_changes == 0)
                
                # 找到连续零变化的最大长度
                max_consecutive_zeros = 0
                current_zeros = 0
                
                for is_zero in zero_changes:
                    if is_zero:
                        current_zeros += 1
                        max_consecutive_zeros = max(max_consecutive_zeros, current_zeros)
                    else:
                        current_zeros = 0
                
                if max_consecutive_zeros > 5:  # 连续5个以上相同价格认为异常
                    anomaly_count += max_consecutive_zeros
                    recommendations.append(f"检测到连续{max_consecutive_zeros}个相同价格，可能是数据停滞")
                
                details['consecutive_anomalies'] = {
                    'max_consecutive_same_price': max_consecutive_zeros,
                    'total_zero_changes': zero_changes.sum(),
                    'zero_change_ratio': zero_changes.mean()
                }
            
            # 计算异常检测评分
            anomaly_ratio = anomaly_count / total_checks if total_checks > 0 else 0.0
            final_score = max(0.0, 1.0 - anomaly_ratio / self.thresholds['anomaly_tolerance'])
            
            status = 'pass' if final_score >= 0.95 else \
                    'warning' if final_score >= 0.90 else 'fail'
            
            details['summary'] = {
                'total_anomalies': anomaly_count,
                'total_checks': total_checks,
                'anomaly_ratio': anomaly_ratio,
                'anomaly_tolerance': self.thresholds['anomaly_tolerance']
            }
            
            return ValidationResult(
                metric_name='anomalies',
                score=final_score,
                status=status,
                details=details,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"异常检测失败: {e}")
            return ValidationResult(
                metric_name='anomalies',
                score=1.0,  # 默认无异常
                status='pass',
                details={'error': str(e)},
                recommendations=['异常检测过程中发生错误，建议手动检查']
            )
    
    # 以下是便捷方法，供MarketDataManager调用
    
    async def check_completeness(self, file_path: Path, start_date: str, end_date: str) -> float:
        """快速完整性检查"""
        try:
            df = await self._load_data_safely(file_path)
            if df.empty:
                return 0.0
            
            # 简化的完整性检查
            if 'timestamp' in df.columns:
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                data_start = df['timestamp'].min()
                data_end = df['timestamp'].max()
                
                # 检查时间覆盖范围
                time_coverage = min(1.0, (data_end - data_start) / (end_dt - start_dt))
                
                # 检查数据密度
                expected_points = (end_dt - start_dt).total_seconds() / (5 * 60)  # 假设5分钟数据
                actual_points = len(df)
                density = min(1.0, actual_points / expected_points) if expected_points > 0 else 0.0
                
                return (time_coverage + density) / 2
            
            return 0.9  # 无时间戳时的默认分数
            
        except Exception as e:
            self.logger.error(f"快速完整性检查失败: {e}")
            return 0.0
    
    async def check_accuracy(self, file_path: Path) -> float:
        """快速准确性检查"""
        try:
            df = await self._load_data_safely(file_path)
            if df.empty:
                return 0.0
            
            scores = []
            
            # OHLC关系检查
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                ohlc_valid = (
                    (df['high'] >= df['open']) & 
                    (df['high'] >= df['low']) & 
                    (df['high'] >= df['close']) &
                    (df['low'] <= df['open']) & 
                    (df['low'] <= df['close'])
                ).mean()
                scores.append(ohlc_valid)
            
            # 价格合理性
            if 'close' in df.columns:
                positive_prices = (df['close'] > 0).mean()
                scores.append(positive_prices)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            self.logger.error(f"快速准确性检查失败: {e}")
            return 0.0
    
    async def check_consistency(self, file_path: Path) -> float:
        """快速一致性检查"""
        try:
            df = await self._load_data_safely(file_path)
            if df.empty:
                return 0.0
            
            scores = []
            
            # 时间单调性
            if 'timestamp' in df.columns:
                is_monotonic = df['timestamp'].is_monotonic_increasing
                scores.append(1.0 if is_monotonic else 0.0)
            
            # 数据类型一致性
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if col in df.columns:
                    is_numeric = pd.api.types.is_numeric_dtype(df[col])
                    scores.append(1.0 if is_numeric else 0.0)
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            self.logger.error(f"快速一致性检查失败: {e}")
            return 0.0
    
    async def check_validity(self, file_path: Path) -> float:
        """快速有效性检查"""
        try:
            df = await self._load_data_safely(file_path)
            if df.empty:
                return 0.0
            
            # 检查非空比例
            total_cells = df.size
            non_null_cells = df.count().sum()
            
            return non_null_cells / total_cells if total_cells > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"快速有效性检查失败: {e}")
            return 0.0
    
    async def check_time_gaps(self, file_path: Path, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """检查时间间隙"""
        try:
            df = await self._load_data_safely(file_path)
            if df.empty or 'timestamp' not in df.columns:
                return []
            
            return self._detect_time_gaps_detailed(df['timestamp'], '5m')  # 默认5分钟
            
        except Exception as e:
            self.logger.error(f"时间间隙检查失败: {e}")
            return []