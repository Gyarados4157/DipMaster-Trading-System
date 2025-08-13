#!/usr/bin/env python3
"""
DipMaster Overfitting Detection Suite
过拟合检测套件 - 全面评估策略优化中的过拟合风险

专门用于检测DipMaster策略参数优化过程中可能存在的过拟合现象：
1. 样本内外表现差异分析
2. 时间序列前向验证
3. 参数敏感性分析
4. 交叉验证测试
5. 统计显著性检验

Author: DipMaster Risk Analysis Team
Date: 2025-08-13
Version: 1.0.0
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import itertools
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OverfittingMetrics:
    """过拟合检测指标"""
    parameter_set: str
    in_sample_win_rate: float
    out_sample_win_rate: float
    performance_degradation: float
    statistical_significance: float
    parameter_sensitivity: float
    overfitting_score: float  # 0-100, 100表示严重过拟合
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL

class OverfittingDetector:
    """过拟合检测器"""
    
    def __init__(self):
        self.original_config = {
            'rsi_range': (30, 50),
            'ma_period': 20,
            'profit_target': 0.008,
            'min_holding_minutes': 15,
            'max_holding_minutes': 180
        }
        
        # 优化后的参数（可能过拟合）
        self.optimized_config = {
            'rsi_range': (40, 60),
            'ma_period': 30,
            'profit_target': 0.012,
            'min_holding_minutes': 15,
            'max_holding_minutes': 180
        }
        
        self.results = {}
        
    def load_historical_data(self, symbol: str = "ICPUSDT") -> pd.DataFrame:
        """加载历史数据"""
        data_file = f"data/market_data/{symbol}_5m_2years.csv"
        
        logger.info(f"📊 加载数据用于过拟合检测: {symbol}")
        
        try:
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"✅ 数据加载完成: {len(df)}条记录")
            return df
            
        except Exception as e:
            logger.error(f"❌ 数据加载失败: {e}")
            return pd.DataFrame()
    
    def split_data_temporal(self, df: pd.DataFrame, 
                           train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """时间序列数据分割"""
        
        split_point = int(len(df) * train_ratio)
        
        train_data = df.iloc[:split_point].copy()
        test_data = df.iloc[split_point:].copy()
        
        logger.info(f"🔄 数据分割: 训练集{len(train_data)}条 ({train_ratio:.0%}), "
                   f"测试集{len(test_data)}条 ({1-train_ratio:.0%})")
        logger.info(f"📅 训练期间: {train_data.index[0].strftime('%Y-%m-%d')} 到 {train_data.index[-1].strftime('%Y-%m-%d')}")
        logger.info(f"📅 测试期间: {test_data.index[0].strftime('%Y-%m-%d')} 到 {test_data.index[-1].strftime('%Y-%m-%d')}")
        
        return train_data, test_data
    
    def calculate_indicators(self, df: pd.DataFrame, ma_period: int = 20) -> pd.DataFrame:
        """计算技术指标"""
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 移动平均线
        df[f'ma{ma_period}'] = df['close'].rolling(ma_period).mean()
        
        # 价格变化
        df['is_dip'] = df['close'] < df['open']
        
        return df
    
    def simulate_strategy(self, df: pd.DataFrame, config: Dict) -> Dict:
        """策略模拟"""
        
        # 计算指标
        df = self.calculate_indicators(df, config['ma_period'])
        
        # 初始化
        trades = []
        current_position = None
        capital = 10000
        
        rsi_low, rsi_high = config['rsi_range']
        ma_col = f"ma{config['ma_period']}"
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_time = df.index[i]
            
            # 跳过空值
            if pd.isna(row['rsi']) or pd.isna(row[ma_col]):
                continue
            
            # 出场检查
            if current_position:
                holding_minutes = (current_time - current_position['entry_time']).total_seconds() / 60
                pnl_pct = ((row['close'] - current_position['entry_price']) / current_position['entry_price']) * 100
                
                should_exit = False
                exit_reason = ""
                
                # 盈利目标
                if pnl_pct >= config['profit_target'] * 100:
                    should_exit = True
                    exit_reason = "profit_target"
                
                # 最大持仓时间
                elif holding_minutes >= config['max_holding_minutes']:
                    should_exit = True
                    exit_reason = "max_holding"
                
                # 边界出场（简化）
                elif holding_minutes >= config['min_holding_minutes'] and current_time.minute in [15, 30, 45, 0]:
                    if np.random.random() < 0.7:  # 70%概率边界出场
                        should_exit = True
                        exit_reason = "boundary"
                
                if should_exit:
                    # 记录交易
                    pnl_usd = (row['close'] - current_position['entry_price']) * current_position['quantity']
                    commission = abs(pnl_usd) * 0.0008  # 双边手续费
                    net_pnl = pnl_usd - commission
                    
                    trades.append({
                        'entry_time': current_position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': current_position['entry_price'],
                        'exit_price': row['close'],
                        'holding_minutes': holding_minutes,
                        'pnl_usd': net_pnl,
                        'pnl_percent': pnl_pct,
                        'exit_reason': exit_reason,
                        'win': net_pnl > 0
                    })
                    
                    capital += net_pnl
                    current_position = None
            
            # 入场检查
            if not current_position:
                # DipMaster入场条件
                if (rsi_low <= row['rsi'] <= rsi_high and  # RSI范围
                    row['is_dip'] and                      # 逢跌
                    row['close'] < row[ma_col]):           # 低于MA
                    
                    current_position = {
                        'entry_time': current_time,
                        'entry_price': row['close'],
                        'quantity': 1000 / row['close']  # 固定1000美元
                    }
        
        # 计算指标
        if trades:
            wins = [t for t in trades if t['win']]
            win_rate = len(wins) / len(trades) * 100
            total_return = (capital - 10000) / 10000 * 100
            avg_holding = np.mean([t['holding_minutes'] for t in trades])
        else:
            win_rate = 0
            total_return = 0
            avg_holding = 0
        
        return {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'total_return': total_return,
            'final_capital': capital,
            'avg_holding_minutes': avg_holding,
            'trades': trades
        }
    
    def test_parameter_sensitivity(self, df: pd.DataFrame) -> Dict:
        """参数敏感性测试"""
        
        logger.info("🔬 开始参数敏感性分析...")
        
        base_config = self.optimized_config.copy()
        sensitivity_results = {}
        
        # 测试RSI范围敏感性
        rsi_variations = [
            (35, 55), (38, 58), (40, 60), (42, 62), (45, 65)
        ]
        
        rsi_results = []
        for rsi_range in rsi_variations:
            config = base_config.copy()
            config['rsi_range'] = rsi_range
            
            result = self.simulate_strategy(df, config)
            rsi_results.append({
                'rsi_range': rsi_range,
                'win_rate': result['win_rate'],
                'total_return': result['total_return'],
                'total_trades': result['total_trades']
            })
        
        sensitivity_results['rsi_sensitivity'] = rsi_results
        
        # 测试MA周期敏感性
        ma_variations = [25, 28, 30, 32, 35]
        
        ma_results = []
        for ma_period in ma_variations:
            config = base_config.copy()
            config['ma_period'] = ma_period
            
            result = self.simulate_strategy(df, config)
            ma_results.append({
                'ma_period': ma_period,
                'win_rate': result['win_rate'],
                'total_return': result['total_return'],
                'total_trades': result['total_trades']
            })
        
        sensitivity_results['ma_sensitivity'] = ma_results
        
        # 测试盈利目标敏感性
        profit_variations = [0.008, 0.010, 0.012, 0.014, 0.016]
        
        profit_results = []
        for profit_target in profit_variations:
            config = base_config.copy()
            config['profit_target'] = profit_target
            
            result = self.simulate_strategy(df, config)
            profit_results.append({
                'profit_target': profit_target,
                'win_rate': result['win_rate'],
                'total_return': result['total_return'],
                'total_trades': result['total_trades']
            })
        
        sensitivity_results['profit_sensitivity'] = profit_results
        
        logger.info("✅ 参数敏感性分析完成")
        return sensitivity_results
    
    def forward_validation(self, df: pd.DataFrame, 
                          window_months: int = 6) -> Dict:
        """前向验证（Walk-Forward Analysis）"""
        
        logger.info(f"🔄 开始前向验证分析 (窗口: {window_months}个月)")
        
        # 按月份分割数据
        df['year_month'] = df.index.to_period('M')
        monthly_groups = df.groupby('year_month')
        
        # 获取所有月份
        all_periods = sorted(df['year_month'].unique())
        
        if len(all_periods) < window_months + 2:
            logger.warning("⚠️ 数据不足以进行前向验证")
            return {}
        
        validation_results = []
        
        # 滑动窗口验证
        for i in range(len(all_periods) - window_months - 1):
            # 训练期间
            train_periods = all_periods[i:i + window_months]
            train_data = pd.concat([monthly_groups.get_group(p) for p in train_periods])
            
            # 测试期间（下一个月）
            test_period = all_periods[i + window_months]
            test_data = monthly_groups.get_group(test_period)
            
            if len(test_data) < 100:  # 跳过数据不足的月份
                continue
            
            # 在训练集上"优化"（这里简化为使用固定的最优参数）
            train_result = self.simulate_strategy(train_data, self.optimized_config)
            
            # 在测试集上验证
            test_result = self.simulate_strategy(test_data, self.optimized_config)
            
            validation_results.append({
                'train_period': f"{train_periods[0]} to {train_periods[-1]}",
                'test_period': str(test_period),
                'train_win_rate': train_result['win_rate'],
                'test_win_rate': test_result['win_rate'],
                'train_return': train_result['total_return'],
                'test_return': test_result['total_return'],
                'performance_degradation': train_result['win_rate'] - test_result['win_rate'],
                'train_trades': train_result['total_trades'],
                'test_trades': test_result['total_trades']
            })
            
            logger.info(f"📊 验证 {test_period}: 训练胜率{train_result['win_rate']:.1f}%, "
                       f"测试胜率{test_result['win_rate']:.1f}%, "
                       f"差异{train_result['win_rate'] - test_result['win_rate']:+.1f}%")
        
        return {
            'validation_results': validation_results,
            'avg_performance_degradation': np.mean([r['performance_degradation'] for r in validation_results]) if validation_results else 0,
            'max_performance_degradation': max([r['performance_degradation'] for r in validation_results]) if validation_results else 0,
            'degradation_std': np.std([r['performance_degradation'] for r in validation_results]) if validation_results else 0
        }
    
    def detect_overfitting_comprehensive(self, symbol: str = "ICPUSDT") -> Dict:
        """综合过拟合检测"""
        
        logger.info("🚨 开始综合过拟合检测分析")
        
        # 加载数据
        df = self.load_historical_data(symbol)
        if df.empty:
            return {}
        
        # 1. 样本内外分析
        logger.info("📊 Phase 1: 样本内外表现分析")
        train_data, test_data = self.split_data_temporal(df, 0.7)
        
        # 原始参数表现
        original_train = self.simulate_strategy(train_data, self.original_config)
        original_test = self.simulate_strategy(test_data, self.original_config)
        
        # 优化参数表现
        optimized_train = self.simulate_strategy(train_data, self.optimized_config)
        optimized_test = self.simulate_strategy(test_data, self.optimized_config)
        
        # 2. 参数敏感性分析
        logger.info("📊 Phase 2: 参数敏感性分析")
        sensitivity_analysis = self.test_parameter_sensitivity(test_data)
        
        # 3. 前向验证
        logger.info("📊 Phase 3: 前向验证分析")
        forward_validation = self.forward_validation(df)
        
        # 4. 统计显著性检验
        logger.info("📊 Phase 4: 统计显著性检验")
        
        # 比较原始vs优化参数的显著性
        if original_test['total_trades'] > 30 and optimized_test['total_trades'] > 30:
            # 使用胜率差异进行t检验（简化）
            original_wins = [1 if t['win'] else 0 for t in original_test['trades']]
            optimized_wins = [1 if t['win'] else 0 for t in optimized_test['trades']]
            
            if len(original_wins) > 10 and len(optimized_wins) > 10:
                t_stat, p_value = stats.ttest_ind(original_wins, optimized_wins)
            else:
                t_stat, p_value = 0, 1.0
        else:
            t_stat, p_value = 0, 1.0
        
        # 5. 过拟合风险评估
        risk_factors = []
        overfitting_score = 0
        
        # 样本内外差异检查
        train_test_diff = optimized_train['win_rate'] - optimized_test['win_rate']
        if train_test_diff > 10:  # 差异超过10%
            risk_factors.append("样本内外表现差异过大")
            overfitting_score += 30
        elif train_test_diff > 5:
            risk_factors.append("样本内外表现存在差异")
            overfitting_score += 15
        
        # 参数敏感性检查
        rsi_win_rates = [r['win_rate'] for r in sensitivity_analysis.get('rsi_sensitivity', [])]
        if rsi_win_rates and max(rsi_win_rates) - min(rsi_win_rates) > 15:
            risk_factors.append("参数对RSI范围过度敏感")
            overfitting_score += 25
        
        # 前向验证检查
        if forward_validation and forward_validation.get('avg_performance_degradation', 0) > 8:
            risk_factors.append("前向验证显示性能显著衰减")
            overfitting_score += 35
        
        # 统计显著性检查
        if p_value > 0.05:
            risk_factors.append("参数改进缺乏统计显著性")
            overfitting_score += 20
        
        # 确定风险等级
        if overfitting_score >= 70:
            risk_level = "CRITICAL"
        elif overfitting_score >= 50:
            risk_level = "HIGH"  
        elif overfitting_score >= 30:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # 综合结果
        comprehensive_results = {
            'symbol': symbol,
            'analysis_date': datetime.now().isoformat(),
            
            'sample_analysis': {
                'original_config': {
                    'train_win_rate': original_train['win_rate'],
                    'test_win_rate': original_test['win_rate'],
                    'performance_diff': original_train['win_rate'] - original_test['win_rate']
                },
                'optimized_config': {
                    'train_win_rate': optimized_train['win_rate'],
                    'test_win_rate': optimized_test['win_rate'],
                    'performance_diff': optimized_train['win_rate'] - optimized_test['win_rate']
                }
            },
            
            'sensitivity_analysis': sensitivity_analysis,
            'forward_validation': forward_validation,
            
            'statistical_test': {
                't_statistic': float(t_stat) if not np.isnan(t_stat) else 0,
                'p_value': float(p_value) if not np.isnan(p_value) else 1.0,
                'significant': p_value < 0.05 if not np.isnan(p_value) else False
            },
            
            'overfitting_assessment': {
                'risk_factors': risk_factors,
                'overfitting_score': overfitting_score,
                'risk_level': risk_level,
                'recommendation': self.get_recommendation(overfitting_score, risk_level)
            }
        }
        
        logger.info(f"✅ 过拟合检测完成 - 风险等级: {risk_level} (评分: {overfitting_score}/100)")
        
        return comprehensive_results
    
    def get_recommendation(self, score: int, risk_level: str) -> str:
        """获取建议"""
        
        recommendations = {
            "LOW": "策略参数优化合理，过拟合风险较低。可以谨慎使用优化后的参数，但建议持续监控实际表现。",
            "MEDIUM": "存在中等程度的过拟合风险。建议使用更保守的参数，增加样本外验证，并考虑参数的鲁棒性。",
            "HIGH": "过拟合风险较高。建议重新审视参数优化过程，使用更大的验证集，并考虑简化模型复杂度。",
            "CRITICAL": "严重过拟合风险！强烈建议重新优化，使用交叉验证，增加正则化约束，或回到更保守的参数设置。"
        }
        
        return recommendations.get(risk_level, "未知风险等级")

def main():
    """主函数"""
    
    print("🚨 DipMaster Overfitting Detection Suite")
    print("=" * 80)
    
    detector = OverfittingDetector()
    
    # 执行综合过拟合检测
    results = detector.detect_overfitting_comprehensive("ICPUSDT")
    
    if not results:
        print("❌ 检测失败，请检查数据文件")
        return
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"overfitting_analysis_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # 显示关键结果
    print(f"\n✅ 过拟合检测完成，结果已保存: {filename}")
    print("\n🎯 关键发现:")
    
    sample_analysis = results['sample_analysis']
    optimized = sample_analysis['optimized_config']
    
    print(f"📊 优化参数样本内胜率: {optimized['train_win_rate']:.1f}%")
    print(f"📊 优化参数样本外胜率: {optimized['test_win_rate']:.1f}%")
    print(f"📉 性能衰减: {optimized['performance_diff']:+.1f}%")
    
    assessment = results['overfitting_assessment']
    print(f"\n🚨 过拟合风险评估:")
    print(f"风险等级: {assessment['risk_level']}")
    print(f"风险评分: {assessment['overfitting_score']}/100")
    
    if assessment['risk_factors']:
        print(f"\n⚠️ 风险因素:")
        for factor in assessment['risk_factors']:
            print(f"  • {factor}")
    
    print(f"\n💡 建议: {assessment['recommendation']}")

if __name__ == "__main__":
    main()