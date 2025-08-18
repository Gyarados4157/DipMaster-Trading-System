#!/usr/bin/env python3
"""
Signal-Based Portfolio Optimizer for DipMaster
专门处理AlphaSignal的组合优化器

版本: V1.0.0 - 专门用于从AlphaSignal文件构建投资组合
作者: DipMaster Trading System Portfolio Risk Optimizer Agent
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SignalPortfolioOptimizer:
    """基于信号的组合优化器"""
    
    def __init__(self, base_capital: float = 100000):
        self.base_capital = base_capital
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 优化参数
        self.max_position_pct = 0.08  # 单仓位最大8%
        self.max_leverage = 2.5  # 最大杠杆2.5倍
        self.beta_tolerance = 0.15  # Beta容忍度
        self.target_volatility = 0.15  # 目标波动率15%
        self.kelly_fraction = 0.25  # Kelly保守系数
        self.min_confidence = 0.6  # 最低置信度
        self.max_positions = 3  # DipMaster策略最多3个并发仓位
        
        print(f"🚀 Signal Portfolio Optimizer V1.0.0 Initialized")
        print(f"   Base Capital: ${base_capital:,.2f}")
        print(f"   Max Positions: {self.max_positions}")
        print(f"   Max Leverage: {self.max_leverage}x")
        print(f"   Beta Tolerance: ±{self.beta_tolerance}")
    
    def load_alpha_signals(self, signal_file: str) -> pd.DataFrame:
        """加载Alpha信号数据"""
        try:
            signals_df = pd.read_csv(signal_file)
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            
            # 过滤高置信度信号
            high_conf_signals = signals_df[
                signals_df['confidence'] >= self.min_confidence
            ].copy()
            
            # 按时间排序，取最新信号
            latest_signals = high_conf_signals.sort_values('timestamp').groupby('symbol').tail(1)
            
            print(f"📊 Signal Analysis:")
            print(f"   Total Raw Signals: {len(signals_df)}")
            print(f"   High Confidence Signals (≥{self.min_confidence}): {len(high_conf_signals)}")
            print(f"   Latest Signals per Symbol: {len(latest_signals)}")
            print(f"   Unique Symbols: {latest_signals['symbol'].nunique()}")
            print(f"   Avg Confidence: {latest_signals['confidence'].mean():.3f}")
            print(f"   Avg Expected Return: {latest_signals['predicted_return'].mean():.4f}")
            
            return latest_signals
            
        except Exception as e:
            print(f"❌ Error loading signals: {e}")
            return pd.DataFrame()
    
    def calculate_position_weights(self, signals_df: pd.DataFrame) -> Dict[str, float]:
        """计算仓位权重"""
        if signals_df.empty:
            return {}
        
        weights = {}
        
        print(f"\n📈 Position Weight Calculation:")
        
        for idx, signal in signals_df.iterrows():
            symbol = signal['symbol']
            confidence = signal['confidence']
            expected_return = signal['predicted_return']
            signal_strength = signal.get('signal', confidence)  # 使用信号强度或置信度
            
            # Kelly准则计算
            # 基于DipMaster策略参数：胜率78%，平均收益0.8%
            win_rate = 0.78  # 从策略表现获得
            avg_win = 0.008  # 0.8%
            avg_loss = 0.004  # 假设平均亏损0.4%
            
            if avg_loss > 0:
                # Kelly公式: f* = (p*b - q) / b
                # p=胜率, q=败率, b=赔率(平均盈利/平均亏损)
                odds_ratio = avg_win / avg_loss
                kelly_raw = (win_rate * odds_ratio - (1 - win_rate)) / odds_ratio
            else:
                kelly_raw = 0.1
            
            # 应用保守系数
            kelly_conservative = kelly_raw * self.kelly_fraction
            
            # 置信度调整
            confidence_adjusted = kelly_conservative * confidence
            
            # 预期收益调整
            return_multiplier = min(expected_return / 0.008, 2.0)  # 基于目标收益0.8%
            return_adjusted = confidence_adjusted * return_multiplier
            
            # 应用单仓位限制
            final_weight = min(return_adjusted, self.max_position_pct)
            
            # 确保非负
            final_weight = max(0, final_weight)
            
            weights[symbol] = final_weight
            
            print(f"   {symbol}: Weight={final_weight:.4f}, Kelly={kelly_raw:.4f}, "
                  f"Conf={confidence:.3f}, ExpRet={expected_return:.4f}")
        
        return weights
    
    def apply_position_limits(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用仓位数量限制"""
        if not weights or len(weights) <= self.max_positions:
            return weights
        
        # 按权重排序，选择最大的N个仓位
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        # 选择前max_positions个
        selected_positions = dict(sorted_weights[:self.max_positions])
        
        print(f"🎯 Position Limit Applied:")
        print(f"   Original Positions: {len(weights)}")
        print(f"   Selected Positions: {len(selected_positions)}")
        print(f"   Selected Symbols: {list(selected_positions.keys())}")
        
        return selected_positions
    
    def apply_market_neutrality(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用市场中性约束"""
        if not weights:
            return weights
        
        total_net_exposure = sum(weights.values())
        
        print(f"\n🎯 Market Neutrality Check:")
        print(f"   Total Net Exposure: {total_net_exposure:.4f}")
        print(f"   Beta Tolerance: ±{self.beta_tolerance}")
        
        # 如果净敞口超过容忍度，按比例缩减
        if abs(total_net_exposure) > self.beta_tolerance:
            adjustment_factor = self.beta_tolerance / abs(total_net_exposure)
            adjusted_weights = {
                symbol: weight * adjustment_factor 
                for symbol, weight in weights.items()
            }
            
            print(f"   🔧 Applied Adjustment Factor: {adjustment_factor:.3f}")
            print(f"   🔧 New Net Exposure: {sum(adjusted_weights.values()):.4f}")
            return adjusted_weights
        
        print(f"   ✅ Market Neutrality Satisfied")
        return weights
    
    def apply_leverage_constraints(self, weights: Dict[str, float]) -> Dict[str, float]:
        """应用杠杆约束"""
        if not weights:
            return weights
        
        total_leverage = sum(abs(w) for w in weights.values())
        
        print(f"\n🛡️ Leverage Constraint Check:")
        print(f"   Total Leverage: {total_leverage:.4f}")
        print(f"   Max Leverage: {self.max_leverage:.4f}")
        
        # 如果杠杆过高，按比例缩减
        if total_leverage > self.max_leverage:
            scale_factor = self.max_leverage / total_leverage
            scaled_weights = {
                symbol: weight * scale_factor 
                for symbol, weight in weights.items()
            }
            
            print(f"   🔧 Applied Scale Factor: {scale_factor:.3f}")
            print(f"   🔧 New Total Leverage: {sum(abs(w) for w in scaled_weights.values()):.4f}")
            return scaled_weights
        
        print(f"   ✅ Leverage Constraint Satisfied")
        return weights
    
    def calculate_portfolio_metrics(self, weights: Dict[str, float], 
                                  signals_df: pd.DataFrame) -> Dict:
        """计算组合指标"""
        if not weights:
            return self._empty_metrics()
        
        # 基础指标
        total_positions = len([w for w in weights.values() if abs(w) > 1e-6])
        gross_exposure = sum(abs(w) for w in weights.values())
        net_exposure = sum(weights.values())
        leverage = gross_exposure
        
        long_exposure = sum(w for w in weights.values() if w > 0)
        short_exposure = sum(w for w in weights.values() if w < 0)
        
        # 预期组合收益
        expected_portfolio_return = 0
        total_confidence_weighted = 0
        
        for symbol, weight in weights.items():
            symbol_signal = signals_df[signals_df['symbol'] == symbol]
            if not symbol_signal.empty:
                signal_data = symbol_signal.iloc[0]
                expected_portfolio_return += weight * signal_data['predicted_return']
                total_confidence_weighted += abs(weight) * signal_data['confidence']
        
        avg_confidence = total_confidence_weighted / gross_exposure if gross_exposure > 0 else 0
        
        # 风险指标估算
        # 基于单资产波动率3%，相关性0.3的简化模型
        individual_volatility = 0.03  # 3%日波动率
        avg_correlation = 0.3
        
        # 组合方差近似
        if total_positions > 1:
            portfolio_variance = (
                (gross_exposure ** 2) * (individual_volatility ** 2) *
                (1 / total_positions + avg_correlation * (total_positions - 1) / total_positions)
            )
        else:
            portfolio_variance = (gross_exposure * individual_volatility) ** 2
        
        portfolio_volatility = np.sqrt(portfolio_variance) * np.sqrt(252)  # 年化
        
        # VaR计算
        var_95 = 1.645 * np.sqrt(portfolio_variance)  # 日度VaR 95%
        var_99 = 2.33 * np.sqrt(portfolio_variance)   # 日度VaR 99%
        expected_shortfall_95 = var_95 * 1.28  # ES近似
        
        # 夏普比率
        risk_free_rate = 0.02
        expected_annual_return = expected_portfolio_return * 252  # 年化
        sharpe_ratio = (expected_annual_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
        
        return {
            'total_positions': total_positions,
            'gross_exposure': gross_exposure,
            'net_exposure': net_exposure,
            'leverage': leverage,
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'expected_annual_return': expected_annual_return,
            'portfolio_volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'portfolio_beta': net_exposure,  # 简化Beta = 净敞口
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': expected_shortfall_95,
            'avg_confidence': avg_confidence
        }
    
    def _empty_metrics(self) -> Dict:
        """空组合默认指标"""
        return {
            'total_positions': 0, 'gross_exposure': 0, 'net_exposure': 0,
            'leverage': 0, 'long_exposure': 0, 'short_exposure': 0,
            'expected_annual_return': 0, 'portfolio_volatility': 0,
            'sharpe_ratio': 0, 'portfolio_beta': 0, 'var_95': 0, 'var_99': 0,
            'expected_shortfall_95': 0, 'avg_confidence': 0
        }
    
    def generate_stress_tests(self, weights: Dict[str, float]) -> Dict:
        """生成压力测试结果"""
        scenarios = {
            'market_crash_20pct': {
                'description': 'Market crash -20%',
                'market_shock': -0.20
            },
            'market_rally_15pct': {
                'description': 'Market rally +15%',
                'market_shock': 0.15
            },
            'volatility_spike_2x': {
                'description': 'Volatility doubles',
                'vol_multiplier': 2.0
            },
            'correlation_shock_90pct': {
                'description': 'All correlations → 0.9',
                'correlation_shock': 0.9
            }
        }
        
        stress_results = {}
        
        for scenario_name, params in scenarios.items():
            if 'market_shock' in params:
                # 市场冲击
                portfolio_return = sum(weights.values()) * params['market_shock']
                portfolio_pnl_usd = portfolio_return * self.base_capital
                max_position_loss = max([abs(w) for w in weights.values()] + [0]) * abs(params['market_shock']) * self.base_capital
                
                stress_results[scenario_name] = {
                    'description': params['description'],
                    'portfolio_return_pct': portfolio_return,
                    'portfolio_pnl_usd': portfolio_pnl_usd,
                    'max_single_position_loss_usd': max_position_loss
                }
            
            elif 'vol_multiplier' in params:
                # 波动率冲击
                base_vol = 0.03
                stressed_vol = base_vol * params['vol_multiplier']
                portfolio_stressed_vol = stressed_vol * np.sqrt(sum(w**2 for w in weights.values()))
                
                stress_results[scenario_name] = {
                    'description': params['description'],
                    'stressed_portfolio_volatility': portfolio_stressed_vol,
                    'vol_increase_factor': params['vol_multiplier'],
                    'stressed_var_95': 1.645 * portfolio_stressed_vol
                }
            
            elif 'correlation_shock' in params:
                # 相关性冲击
                # 简化计算：假设所有相关性变为指定值
                n_positions = len(weights)
                total_weight_sq = sum(w**2 for w in weights.values())
                cross_product_sum = sum(weights.values())**2 - total_weight_sq
                
                new_correlation = params['correlation_shock']
                shocked_variance = total_weight_sq * (0.03**2) + cross_product_sum * new_correlation * (0.03**2)
                shocked_volatility = np.sqrt(shocked_variance)
                
                stress_results[scenario_name] = {
                    'description': params['description'],
                    'shocked_correlation': new_correlation,
                    'shocked_portfolio_volatility': shocked_volatility,
                    'correlation_impact': shocked_volatility / (0.03 * np.sqrt(sum(w**2 for w in weights.values()))) - 1
                }
        
        return stress_results
    
    def create_target_portfolio(self, weights: Dict[str, float], 
                              signals_df: pd.DataFrame,
                              metrics: Dict) -> Dict:
        """创建目标组合输出"""
        
        # 构建仓位列表
        positions = []
        for symbol, weight in weights.items():
            if abs(weight) > 1e-6:
                symbol_signal = signals_df[signals_df['symbol'] == symbol]
                signal_data = symbol_signal.iloc[0] if not symbol_signal.empty else None
                
                positions.append({
                    'symbol': symbol,
                    'weight': round(float(weight), 6),
                    'dollar_amount': round(float(weight * self.base_capital), 2),
                    'signal_strength': round(float(signal_data['signal']) if signal_data is not None else 0, 4),
                    'confidence': round(float(signal_data['confidence']) if signal_data is not None else 0, 4),
                    'predicted_return': round(float(signal_data['predicted_return']) if signal_data is not None else 0, 6),
                    'position_type': 'LONG' if weight > 0 else 'SHORT'
                })
        
        # 约束合规检查
        constraints_status = {
            'beta_neutral': bool(abs(metrics.get('portfolio_beta', 0)) <= self.beta_tolerance),
            'volatility_target': bool(metrics.get('portfolio_volatility', 0) <= self.target_volatility),
            'leverage_limit': bool(metrics.get('leverage', 0) <= self.max_leverage),
            'position_count_limit': bool(metrics.get('total_positions', 0) <= self.max_positions),
            'var_limit': bool(metrics.get('var_95', 0) <= 0.03),  # 3%日度VaR限制
            'position_size_limits': bool(all(abs(w) <= self.max_position_pct for w in weights.values()))
        }
        
        # 压力测试
        stress_tests = self.generate_stress_tests(weights)
        
        target_portfolio = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'optimizer_version': 'SignalPortfolioOptimizer_V1.0.0',
                'base_capital_usd': self.base_capital,
                'optimization_method': 'kelly_criterion_signal_weighting',
                'strategy': 'DipMaster_V4_Enhanced'
            },
            'portfolio_summary': {
                'total_positions': metrics.get('total_positions', 0),
                'gross_exposure': round(metrics.get('gross_exposure', 0), 4),
                'net_exposure': round(metrics.get('net_exposure', 0), 4),
                'leverage': round(metrics.get('leverage', 0), 4),
                'long_exposure': round(metrics.get('long_exposure', 0), 4),
                'short_exposure': round(metrics.get('short_exposure', 0), 4)
            },
            'positions': positions,
            'risk_metrics': {
                'expected_annual_return': round(metrics.get('expected_annual_return', 0), 4),
                'annualized_volatility': round(metrics.get('portfolio_volatility', 0), 4),
                'sharpe_ratio': round(metrics.get('sharpe_ratio', 0), 2),
                'portfolio_beta': round(metrics.get('portfolio_beta', 0), 4),
                'var_95_daily': round(metrics.get('var_95', 0), 4),
                'var_99_daily': round(metrics.get('var_99', 0), 4),
                'expected_shortfall_95': round(metrics.get('expected_shortfall_95', 0), 4),
                'avg_signal_confidence': round(metrics.get('avg_confidence', 0), 3)
            },
            'constraints_compliance': constraints_status,
            'stress_test_results': stress_tests,
            'optimization_details': {
                'kelly_fraction': self.kelly_fraction,
                'min_confidence_threshold': self.min_confidence,
                'max_positions_allowed': self.max_positions,
                'beta_tolerance': self.beta_tolerance,
                'max_leverage': self.max_leverage,
                'max_single_position': self.max_position_pct,
                'all_constraints_satisfied': all(constraints_status.values())
            }
        }
        
        return target_portfolio
    
    def create_risk_report(self, target_portfolio: Dict) -> Dict:
        """创建风险报告"""
        
        # 风险评估
        var_95 = target_portfolio['risk_metrics']['var_95_daily']
        if var_95 > 0.025:
            risk_level = 'HIGH'
        elif var_95 > 0.015:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'
        
        # 生成建议
        recommendations = []
        
        if not target_portfolio['constraints_compliance']['beta_neutral']:
            recommendations.append({
                'type': 'BETA_RISK',
                'priority': 'HIGH',
                'description': f"Portfolio beta ({target_portfolio['risk_metrics']['portfolio_beta']:.3f}) exceeds tolerance",
                'action': 'Consider adding hedge positions or reducing directional exposure'
            })
        
        if not target_portfolio['constraints_compliance']['volatility_target']:
            recommendations.append({
                'type': 'VOLATILITY_RISK',
                'priority': 'MEDIUM',
                'description': f"Portfolio volatility ({target_portfolio['risk_metrics']['annualized_volatility']:.3f}) exceeds target",
                'action': 'Reduce position sizes or increase diversification'
            })
        
        if target_portfolio['portfolio_summary']['leverage'] > self.max_leverage * 0.85:
            recommendations.append({
                'type': 'LEVERAGE_WARNING',
                'priority': 'MEDIUM',
                'description': f"Leverage usage ({target_portfolio['portfolio_summary']['leverage']:.2f}x) approaching limit",
                'action': 'Monitor closely and prepare position reduction if needed'
            })
        
        risk_report = {
            'report_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'report_type': 'Signal_Based_Portfolio_Risk_Assessment',
                'base_capital_usd': self.base_capital
            },
            'executive_summary': {
                'overall_risk_level': risk_level,
                'total_positions': target_portfolio['portfolio_summary']['total_positions'],
                'portfolio_leverage': target_portfolio['portfolio_summary']['leverage'],
                'expected_annual_return': target_portfolio['risk_metrics']['expected_annual_return'],
                'sharpe_ratio': target_portfolio['risk_metrics']['sharpe_ratio'],
                'var_95_daily': target_portfolio['risk_metrics']['var_95_daily'],
                'all_constraints_satisfied': target_portfolio['optimization_details']['all_constraints_satisfied']
            },
            'risk_breakdown': {
                'market_risk': {
                    'beta_exposure': abs(target_portfolio['risk_metrics']['portfolio_beta']),
                    'volatility_risk': target_portfolio['risk_metrics']['annualized_volatility'],
                    'concentration_risk': max([abs(pos['weight']) for pos in target_portfolio['positions']] + [0]),
                    'max_single_position_usd': max([abs(pos['dollar_amount']) for pos in target_portfolio['positions']] + [0])
                },
                'liquidity_risk': {
                    'assessment': 'LOW',  # DipMaster使用高流动性加密货币
                    'largest_position_usd': max([abs(pos['dollar_amount']) for pos in target_portfolio['positions']] + [0]),
                    'total_exposure_usd': sum([abs(pos['dollar_amount']) for pos in target_portfolio['positions']])
                }
            },
            'stress_test_summary': target_portfolio['stress_test_results'],
            'monitoring_recommendations': {
                'daily_var_monitoring': bool(target_portfolio['risk_metrics']['var_95_daily'] > 0.02),
                'position_rebalancing_needed': bool(not target_portfolio['optimization_details']['all_constraints_satisfied']),
                'leverage_monitoring': bool(target_portfolio['portfolio_summary']['leverage'] > 2.0)
            },
            'recommendations': recommendations
        }
        
        return risk_report
    
    def optimize_portfolio(self, signal_file: str) -> Tuple[Dict, Dict]:
        """执行完整的组合优化流程"""
        print(f"\n" + "="*60)
        print(f"🎯 DIPMASTER PORTFOLIO OPTIMIZATION")
        print(f"="*60)
        
        # Step 1: 加载信号
        signals_df = self.load_alpha_signals(signal_file)
        if signals_df.empty:
            print("❌ No valid signals found")
            return {}, {}
        
        # Step 2: 计算基础权重
        print(f"\n📊 Step 2: Calculate Position Weights")
        raw_weights = self.calculate_position_weights(signals_df)
        
        # Step 3: 应用仓位数量限制
        print(f"\n🎯 Step 3: Apply Position Limits")
        limited_weights = self.apply_position_limits(raw_weights)
        
        # Step 4: 应用市场中性
        print(f"\n⚖️ Step 4: Apply Market Neutrality")
        neutral_weights = self.apply_market_neutrality(limited_weights)
        
        # Step 5: 应用杠杆约束
        print(f"\n🛡️ Step 5: Apply Leverage Constraints")
        final_weights = self.apply_leverage_constraints(neutral_weights)
        
        # Step 6: 计算组合指标
        print(f"\n📈 Step 6: Calculate Portfolio Metrics")
        metrics = self.calculate_portfolio_metrics(final_weights, signals_df)
        
        # Step 7: 生成目标组合
        print(f"\n📋 Step 7: Generate Target Portfolio")
        target_portfolio = self.create_target_portfolio(final_weights, signals_df, metrics)
        
        # Step 8: 生成风险报告
        print(f"\n📊 Step 8: Generate Risk Report")
        risk_report = self.create_risk_report(target_portfolio)
        
        return target_portfolio, risk_report


def main():
    """主函数"""
    # 创建优化器
    optimizer = SignalPortfolioOptimizer(base_capital=100000)
    
    # 执行优化
    signal_file = "results/basic_ml_pipeline/signals_20250818_153608.csv"
    target_portfolio, risk_report = optimizer.optimize_portfolio(signal_file)
    
    if not target_portfolio:
        print("❌ Portfolio optimization failed")
        return
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存目标组合
    portfolio_file = f"results/portfolio_optimization/TargetPortfolio_Signal_V1_{timestamp}.json"
    with open(portfolio_file, 'w') as f:
        json.dump(target_portfolio, f, indent=2)
    
    # 保存风险报告
    risk_file = f"results/portfolio_optimization/RiskReport_Signal_V1_{timestamp}.json"
    with open(risk_file, 'w') as f:
        json.dump(risk_report, f, indent=2)
    
    # 输出摘要
    print(f"\n" + "="*60)
    print(f"✅ PORTFOLIO OPTIMIZATION COMPLETED")
    print(f"="*60)
    print(f"📊 Target Portfolio: {portfolio_file}")
    print(f"📋 Risk Report: {risk_file}")
    
    print(f"\n🎯 PORTFOLIO SUMMARY:")
    print(f"   Total Positions: {target_portfolio['portfolio_summary']['total_positions']}")
    print(f"   Gross Exposure: {target_portfolio['portfolio_summary']['gross_exposure']:.4f}")
    print(f"   Net Exposure: {target_portfolio['portfolio_summary']['net_exposure']:.4f}")
    print(f"   Leverage: {target_portfolio['portfolio_summary']['leverage']:.2f}x")
    print(f"   Expected Annual Return: {target_portfolio['risk_metrics']['expected_annual_return']:.2%}")
    print(f"   Sharpe Ratio: {target_portfolio['risk_metrics']['sharpe_ratio']:.2f}")
    print(f"   Portfolio Beta: {target_portfolio['risk_metrics']['portfolio_beta']:.4f}")
    print(f"   VaR (95% Daily): {target_portfolio['risk_metrics']['var_95_daily']:.4f}")
    
    print(f"\n🛡️ RISK ASSESSMENT:")
    print(f"   Overall Risk Level: {risk_report['executive_summary']['overall_risk_level']}")
    print(f"   All Constraints Satisfied: {risk_report['executive_summary']['all_constraints_satisfied']}")
    print(f"   Recommendations: {len(risk_report['recommendations'])} items")
    
    print(f"\n💼 POSITIONS:")
    for pos in target_portfolio['positions']:
        print(f"   {pos['symbol']}: ${pos['dollar_amount']:,.2f} "
              f"({pos['weight']:.2%}, Conf: {pos['confidence']:.3f})")
    
    print(f"\n🚀 READY FOR EXECUTION AGENT")
    return target_portfolio, risk_report


if __name__ == "__main__":
    main()