#!/usr/bin/env python3
"""
Fast Validator - Phase 2 of Overfitting Optimization
快速验证框架：3x3x3矩阵测试 (27种组合 vs 数百种参数组合)

测试维度:
1. 3个加密货币对: BTCUSDT, ETHUSDT, ADAUSDT
2. 3个时间段: 2020-2021, 2022-2023, 2023-2024  
3. 3个波动率状态: 低波动, 中波动, 高波动

目标：快速确定策略是否在不同条件下保持一致性能

Author: DipMaster Optimization Team
Date: 2025-08-15  
Version: 1.0.0
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json
import itertools

# 导入简化策略
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from simplified_dipmaster import SimplifiedDipMaster
from edge_analyzer import EdgeAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """验证结果"""
    symbol: str
    period: str
    volatility_regime: str
    win_rate: float
    total_return: float
    total_trades: int
    max_drawdown: float
    risk_reward_ratio: float
    passed: bool
    notes: str


@dataclass
class RobustnessReport:
    """鲁棒性报告"""
    strategy_robust: bool
    confidence_score: float
    consistency_score: float  # 不同条件下性能一致性
    worst_case_scenario: ValidationResult
    best_case_scenario: ValidationResult
    average_performance: Dict[str, float]
    recommendation: str
    validation_results: List[ValidationResult]


class FastValidator:
    """快速验证框架"""
    
    def __init__(self):
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']  # 3个币种
        
        # 时间段定义 - 代表不同市场阶段
        self.time_periods = {
            '2020-2021': ('2020-01-01', '2021-12-31'),  # 牛市
            '2022-2023': ('2022-01-01', '2023-12-31'),  # 熊市转换
            '2023-2024': ('2023-01-01', '2024-12-31')   # 恢复期
        }
        
        # 验证通过标准
        self.pass_criteria = {
            'min_win_rate': 0.50,        # 最低胜率50%
            'min_total_return': -5.0,    # 最低总回报-5%
            'min_trades': 20,            # 最少交易数20
            'max_drawdown': 25.0,        # 最大回撤25%
            'min_risk_reward': 0.8       # 最小风险回报比0.8
        }
        
        # 鲁棒性评估标准
        self.robustness_criteria = {
            'min_pass_rate': 0.70,       # 至少70%的测试通过
            'max_performance_std': 0.15, # 胜率标准差不超过15%
            'min_consistency': 0.75      # 最小一致性评分75%
        }
        
    def load_data_for_period(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """加载指定时期数据"""
        
        # 尝试加载数据文件
        data_file = f"data/market_data/{symbol}_5m_2years.csv"
        
        try:
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # 过滤时间段
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            
            period_data = df[(df.index >= start) & (df.index <= end)]
            
            logger.info(f"📊 Loaded {symbol} data for {start_date} to {end_date}: {len(period_data)} records")
            
            return period_data
            
        except Exception as e:
            logger.warning(f"⚠️ Could not load data for {symbol} {start_date}-{end_date}: {e}")
            return pd.DataFrame()
    
    def classify_volatility_regime(self, df: pd.DataFrame) -> str:
        """分类波动率状态"""
        
        if df.empty:
            return 'unknown'
            
        # 计算价格波动率（20日滚动标准差）
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=20*24).std().mean()  # 20天*24*12(5分钟)
        
        # 根据历史分位数分类（大致分类，可以根据实际数据调整）
        if volatility < 0.015:      # 1.5%
            return 'low_vol'
        elif volatility > 0.035:    # 3.5%
            return 'high_vol'
        else:
            return 'medium_vol'
    
    def test_single_combination(self, symbol: str, period: str, 
                              start_date: str, end_date: str,
                              strategy: SimplifiedDipMaster) -> ValidationResult:
        """测试单个组合"""
        
        logger.info(f"🧪 Testing {symbol} {period}...")
        
        # 加载数据
        df = self.load_data_for_period(symbol, start_date, end_date)
        
        if df.empty or len(df) < 100:
            return ValidationResult(
                symbol=symbol,
                period=period,
                volatility_regime='insufficient_data',
                win_rate=0.0,
                total_return=0.0,
                total_trades=0,
                max_drawdown=0.0,
                risk_reward_ratio=0.0,
                passed=False,
                notes="Insufficient data"
            )
        
        # 分类波动率
        volatility_regime = self.classify_volatility_regime(df)
        
        # 运行回测
        try:
            results = strategy.backtest_strategy(df, initial_capital=10000)
            
            # 提取关键指标
            win_rate = results['win_rate']
            total_return = results['total_return_pct']
            total_trades = results['total_trades']
            max_drawdown = results['max_drawdown_pct']
            risk_reward = results['risk_reward_ratio']
            
            # 检查是否通过标准
            passed = (
                win_rate >= self.pass_criteria['min_win_rate'] and
                total_return >= self.pass_criteria['min_total_return'] and
                total_trades >= self.pass_criteria['min_trades'] and
                max_drawdown <= self.pass_criteria['max_drawdown'] and
                risk_reward >= self.pass_criteria['min_risk_reward']
            )
            
            # 生成说明
            if not passed:
                failed_criteria = []
                if win_rate < self.pass_criteria['min_win_rate']:
                    failed_criteria.append(f"win_rate_{win_rate:.1%}")
                if total_return < self.pass_criteria['min_total_return']:
                    failed_criteria.append(f"return_{total_return:+.1f}%")
                if total_trades < self.pass_criteria['min_trades']:
                    failed_criteria.append(f"trades_{total_trades}")
                if max_drawdown > self.pass_criteria['max_drawdown']:
                    failed_criteria.append(f"drawdown_{max_drawdown:.1f}%")
                if risk_reward < self.pass_criteria['min_risk_reward']:
                    failed_criteria.append(f"risk_reward_{risk_reward:.2f}")
                notes = "Failed: " + ", ".join(failed_criteria)
            else:
                notes = "Passed all criteria"
            
            return ValidationResult(
                symbol=symbol,
                period=period,
                volatility_regime=volatility_regime,
                win_rate=win_rate,
                total_return=total_return,
                total_trades=total_trades,
                max_drawdown=max_drawdown,
                risk_reward_ratio=risk_reward,
                passed=passed,
                notes=notes
            )
            
        except Exception as e:
            logger.error(f"❌ Error testing {symbol} {period}: {e}")
            return ValidationResult(
                symbol=symbol,
                period=period,
                volatility_regime=volatility_regime,
                win_rate=0.0,
                total_return=0.0,
                total_trades=0,
                max_drawdown=100.0,
                risk_reward_ratio=0.0,
                passed=False,
                notes=f"Error: {str(e)}"
            )
    
    def run_3x3x3_validation(self, strategy: SimplifiedDipMaster) -> RobustnessReport:
        """运行3x3x3快速验证"""
        
        logger.info("🚀 Starting 3x3x3 Robustness Validation...")
        logger.info(f"Testing {len(self.symbols)} symbols × {len(self.time_periods)} periods = {len(self.symbols) * len(self.time_periods)} combinations")
        
        all_results: List[ValidationResult] = []
        
        # 遍历所有组合
        for symbol in self.symbols:
            for period, (start_date, end_date) in self.time_periods.items():
                result = self.test_single_combination(symbol, period, start_date, end_date, strategy)
                all_results.append(result)
                
                logger.info(f"📊 {symbol} {period}: "
                           f"WR={result.win_rate:.1%}, "
                           f"Return={result.total_return:+.1f}%, "
                           f"Trades={result.total_trades}, "
                           f"{'✅' if result.passed else '❌'}")
        
        # 分析结果
        return self._analyze_robustness(all_results, strategy)
    
    def _analyze_robustness(self, results: List[ValidationResult], 
                           strategy: SimplifiedDipMaster) -> RobustnessReport:
        """分析鲁棒性"""
        
        logger.info("📈 Analyzing robustness...")
        
        # 过滤掉数据不足的结果
        valid_results = [r for r in results if r.total_trades >= 10]
        
        if not valid_results:
            return RobustnessReport(
                strategy_robust=False,
                confidence_score=0.0,
                consistency_score=0.0,
                worst_case_scenario=results[0] if results else None,
                best_case_scenario=results[0] if results else None,
                average_performance={},
                recommendation="No valid test results. Strategy may be inactive.",
                validation_results=results
            )
        
        # 基础统计
        passed_tests = [r for r in valid_results if r.passed]
        pass_rate = len(passed_tests) / len(valid_results)
        
        # 性能一致性分析
        win_rates = [r.win_rate for r in valid_results]
        returns = [r.total_return for r in valid_results]
        
        win_rate_std = np.std(win_rates)
        return_std = np.std(returns)
        
        # 一致性评分 (0-1, 越高越一致)
        consistency_score = max(0, 1 - (win_rate_std / 0.3))  # 标准化到30%标准差
        
        # 最佳/最差情况
        best_case = max(valid_results, key=lambda x: x.win_rate * (1 + x.total_return/100))
        worst_case = min(valid_results, key=lambda x: x.win_rate * (1 + x.total_return/100))
        
        # 平均性能
        avg_performance = {
            'avg_win_rate': np.mean(win_rates),
            'avg_return': np.mean(returns),
            'avg_trades': np.mean([r.total_trades for r in valid_results]),
            'avg_max_drawdown': np.mean([r.max_drawdown for r in valid_results]),
            'win_rate_std': win_rate_std,
            'return_std': return_std
        }
        
        # 鲁棒性判断
        strategy_robust = (
            pass_rate >= self.robustness_criteria['min_pass_rate'] and
            win_rate_std <= self.robustness_criteria['max_performance_std'] and
            consistency_score >= self.robustness_criteria['min_consistency']
        )
        
        # 置信度评分
        if strategy_robust:
            confidence_score = min(pass_rate * consistency_score, 1.0)
        else:
            confidence_score = pass_rate * 0.5  # 惩罚不鲁棒的策略
        
        # 生成建议
        if strategy_robust:
            if confidence_score >= 0.80:
                recommendation = "Strategy is highly robust. Proceed to deployment."
            else:
                recommendation = "Strategy is moderately robust. Consider conservative position sizing."
        else:
            failed_criteria = []
            if pass_rate < self.robustness_criteria['min_pass_rate']:
                failed_criteria.append(f"low_pass_rate_{pass_rate:.1%}")
            if win_rate_std > self.robustness_criteria['max_performance_std']:
                failed_criteria.append(f"high_variance_{win_rate_std:.1%}")
            if consistency_score < self.robustness_criteria['min_consistency']:
                failed_criteria.append(f"low_consistency_{consistency_score:.1%}")
            
            recommendation = f"Strategy not robust. Issues: {', '.join(failed_criteria)}. Consider strategy pivot."
        
        report = RobustnessReport(
            strategy_robust=strategy_robust,
            confidence_score=confidence_score,
            consistency_score=consistency_score,
            worst_case_scenario=worst_case,
            best_case_scenario=best_case,
            average_performance=avg_performance,
            recommendation=recommendation,
            validation_results=results
        )
        
        # 日志摘要
        logger.info(f"✅ Robustness analysis complete:")
        logger.info(f"   Pass Rate: {pass_rate:.1%} ({len(passed_tests)}/{len(valid_results)})")
        logger.info(f"   Avg Win Rate: {avg_performance['avg_win_rate']:.1%} ± {win_rate_std:.1%}")
        logger.info(f"   Consistency: {consistency_score:.1%}")
        logger.info(f"   Robust: {'✅ YES' if strategy_robust else '❌ NO'}")
        
        return report
    
    def test_parameter_sensitivity(self, base_strategy: SimplifiedDipMaster) -> Dict[str, Any]:
        """参数敏感性测试 - 关键补充验证"""
        
        logger.info("🔬 Testing parameter sensitivity (±20% variation)...")
        
        base_params = {
            'rsi_threshold': base_strategy.rsi_threshold,
            'take_profit_pct': base_strategy.take_profit_pct,
            'stop_loss_pct': base_strategy.stop_loss_pct
        }
        
        # 生成参数变化范围
        sensitivity_tests = []
        
        for param_name, base_value in base_params.items():
            # ±20% 变化
            variations = [
                base_value * 0.8,   # -20%
                base_value * 0.9,   # -10%
                base_value,         # baseline
                base_value * 1.1,   # +10%
                base_value * 1.2    # +20%
            ]
            
            for variation in variations:
                test_params = base_params.copy()
                test_params[param_name] = variation
                
                test_strategy = SimplifiedDipMaster(**test_params)
                sensitivity_tests.append({
                    'param_name': param_name,
                    'param_value': variation,
                    'change_pct': (variation - base_value) / base_value * 100,
                    'strategy': test_strategy
                })
        
        # 在单个测试集上运行所有参数变化
        test_symbol = 'BTCUSDT'
        test_period = '2023-2024'
        start_date, end_date = self.time_periods[test_period]
        
        sensitivity_results = []
        
        for test_config in sensitivity_tests:
            try:
                df = self.load_data_for_period(test_symbol, start_date, end_date)
                if not df.empty:
                    results = test_config['strategy'].backtest_strategy(df)
                    
                    sensitivity_results.append({
                        'param_name': test_config['param_name'],
                        'param_value': test_config['param_value'],
                        'change_pct': test_config['change_pct'],
                        'win_rate': results['win_rate'],
                        'total_return': results['total_return_pct'],
                        'total_trades': results['total_trades']
                    })
                    
            except Exception as e:
                logger.warning(f"⚠️ Sensitivity test failed for {test_config['param_name']}: {e}")
        
        # 分析敏感性
        sensitivity_analysis = {}
        
        for param_name in base_params.keys():
            param_results = [r for r in sensitivity_results if r['param_name'] == param_name]
            
            if len(param_results) >= 3:  # 至少要有几个有效测试
                win_rates = [r['win_rate'] for r in param_results]
                returns = [r['total_return'] for r in param_results]
                
                win_rate_range = max(win_rates) - min(win_rates)
                return_range = max(returns) - min(returns)
                
                sensitivity_analysis[param_name] = {
                    'win_rate_sensitivity': win_rate_range,
                    'return_sensitivity': return_range,
                    'stable': win_rate_range < 0.25 and return_range < 30  # 25%胜率变化, 30%收益变化
                }
        
        overall_sensitivity = {
            'sensitive_parameters': [p for p, a in sensitivity_analysis.items() if not a['stable']],
            'stable_parameters': [p for p, a in sensitivity_analysis.items() if a['stable']],
            'max_win_rate_sensitivity': max([a['win_rate_sensitivity'] for a in sensitivity_analysis.values()]) if sensitivity_analysis else 0,
            'parameter_robust': all(a['stable'] for a in sensitivity_analysis.values()),
            'detailed_results': sensitivity_results
        }
        
        logger.info(f"🔬 Parameter sensitivity results:")
        for param, analysis in sensitivity_analysis.items():
            logger.info(f"   {param}: WR±{analysis['win_rate_sensitivity']:.1%}, "
                       f"Return±{analysis['return_sensitivity']:.1f}%, "
                       f"{'✅ Stable' if analysis['stable'] else '⚠️ Sensitive'}")
        
        return overall_sensitivity


def main():
    """主函数 - 执行快速验证"""
    
    print("⚡ Fast Validation Framework - 3x3x3 Robustness Test")
    print("="*80)
    
    # 创建验证器
    validator = FastValidator()
    
    # 创建简化策略
    strategy = SimplifiedDipMaster(
        rsi_threshold=40.0,
        take_profit_pct=0.015,
        stop_loss_pct=0.008
    )
    
    print(f"🎯 VALIDATION PLAN:")
    print(f"   Symbols: {', '.join(validator.symbols)}")
    print(f"   Periods: {', '.join(validator.time_periods.keys())}")
    print(f"   Total Tests: {len(validator.symbols) * len(validator.time_periods)}")
    print(f"   Pass Criteria: WR≥{validator.pass_criteria['min_win_rate']:.0%}, "
          f"Return≥{validator.pass_criteria['min_total_return']:+.0f}%, "
          f"Trades≥{validator.pass_criteria['min_trades']}")
    
    # 运行3x3x3验证
    report = validator.run_3x3x3_validation(strategy)
    
    # 运行参数敏感性测试
    sensitivity = validator.test_parameter_sensitivity(strategy)
    
    # 显示结果
    print(f"\n🎯 VALIDATION RESULTS:")
    print(f"Strategy Robust: {'✅ YES' if report.strategy_robust else '❌ NO'}")
    print(f"Confidence Score: {report.confidence_score:.1%}")
    print(f"Consistency Score: {report.consistency_score:.1%}")
    
    print(f"\n📊 AVERAGE PERFORMANCE:")
    avg = report.average_performance
    print(f"   Win Rate: {avg['avg_win_rate']:.1%} ± {avg['win_rate_std']:.1%}")
    print(f"   Total Return: {avg['avg_return']:+.1f}% ± {avg['return_std']:.1f}%")
    print(f"   Avg Trades: {avg['avg_trades']:.0f}")
    print(f"   Max Drawdown: {avg['avg_max_drawdown']:.1f}%")
    
    print(f"\n🔬 PARAMETER SENSITIVITY:")
    print(f"Parameters Robust: {'✅ YES' if sensitivity['parameter_robust'] else '❌ NO'}")
    if sensitivity['sensitive_parameters']:
        print(f"Sensitive Parameters: {', '.join(sensitivity['sensitive_parameters'])}")
    if sensitivity['stable_parameters']:
        print(f"Stable Parameters: {', '.join(sensitivity['stable_parameters'])}")
    
    print(f"\n💡 RECOMMENDATION:")
    print(f"{report.recommendation}")
    
    # 保存完整报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"validation_report_{timestamp}.json"
    
    report_data = {
        'validation_summary': {
            'strategy_robust': report.strategy_robust,
            'confidence_score': report.confidence_score,
            'consistency_score': report.consistency_score,
            'recommendation': report.recommendation
        },
        'average_performance': report.average_performance,
        'parameter_sensitivity': sensitivity,
        'detailed_results': [
            {
                'symbol': r.symbol,
                'period': r.period,
                'volatility_regime': r.volatility_regime,
                'win_rate': r.win_rate,
                'total_return': r.total_return,
                'total_trades': r.total_trades,
                'max_drawdown': r.max_drawdown,
                'passed': r.passed,
                'notes': r.notes
            } for r in report.validation_results
        ]
    }
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📁 Full report saved to: {report_filename}")
    
    # 决策建议
    if report.strategy_robust and sensitivity['parameter_robust']:
        print(f"\n🚀 DECISION: PROCEED TO DEPLOYMENT")
        print(f"Both robustness and parameter stability confirmed.")
    elif report.strategy_robust:
        print(f"\n⚠️ DECISION: PROCEED WITH CAUTION")
        print(f"Strategy is robust but parameters may be sensitive.")
    else:
        print(f"\n🛑 DECISION: DO NOT DEPLOY")
        print(f"Strategy lacks robustness across different market conditions.")


if __name__ == "__main__":
    main()