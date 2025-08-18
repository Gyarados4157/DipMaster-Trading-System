#!/usr/bin/env python3
"""
DipMaster Portfolio Risk Optimizer
组合风险优化Agent - 将AlphaSignal转化为市场中性的风险控制投资组合

核心功能：
1. 市场中性约束优化 (Beta ~ 0)  
2. 风险预算管理 (VaR, ES, 波动率控制)
3. 仓位构建与Kelly准则
4. 多资产协方差建模
5. 压力测试与风险归因

作者: DipMaster Trading System
版本: V4.0.0 - Portfolio Risk Optimization
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.optimize import minimize
import cvxpy as cp
from sklearn.covariance import LedoitWolf
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PortfolioConstraints:
    """组合约束配置"""
    target_beta: float = 0.0
    beta_tolerance: float = 0.15
    max_position: float = 0.08  # 单仓位上限8%
    max_leverage: float = 2.5
    min_cash_buffer: float = 0.10  # 10%现金缓冲
    target_volatility: float = 0.15  # 15%年化波动率
    max_var_95: float = 0.03  # 3%日度VaR
    max_es_95: float = 0.04  # 4% Expected Shortfall
    max_correlation: float = 0.7  # 最大相关性

@dataclass 
class RiskMetrics:
    """风险指标"""
    ann_volatility: float
    beta: float
    var_95: float
    var_99: float
    es_95: float
    sharpe_ratio: float
    max_drawdown: float
    tracking_error: float

@dataclass
class PositionWeight:
    """仓位权重"""
    symbol: str
    weight: float
    signal_strength: float
    confidence: float
    expected_return: float

class CovarianceEstimator:
    """协方差矩阵估计器"""
    
    def __init__(self, lookback_days: int = 252, decay_factor: float = 0.94):
        self.lookback_days = lookback_days
        self.decay_factor = decay_factor
        
    def estimate_covariance(self, returns: pd.DataFrame) -> np.ndarray:
        """估计指数加权协方差矩阵"""
        # 使用Ledoit-Wolf收缩估计器
        lw = LedoitWolf()
        cov_matrix = lw.fit(returns.fillna(0)).covariance_
        
        # 应用指数加权
        weights = np.array([self.decay_factor ** i for i in range(len(returns))][::-1])
        weights = weights / weights.sum()
        
        weighted_returns = returns.multiply(np.sqrt(weights), axis=0)
        exp_weighted_cov = np.cov(weighted_returns.T, ddof=1)
        
        # 年化
        return exp_weighted_cov * 252

class PortfolioRiskOptimizer:
    """组合风险优化引擎"""
    
    def __init__(self, 
                 base_capital: float = 100000,
                 constraints: Optional[PortfolioConstraints] = None):
        """
        初始化组合优化器
        
        Args:
            base_capital: 基础资金
            constraints: 约束条件配置
        """
        self.base_capital = base_capital
        self.constraints = constraints or PortfolioConstraints()
        self.cov_estimator = CovarianceEstimator()
        
        # 初始化状态
        self.current_positions = {}
        self.market_data = {}
        self.correlation_matrix = None
        self.covariance_matrix = None
        
        print(f"✅ Portfolio Risk Optimizer Initialized")
        print(f"   Base Capital: ${base_capital:,.2f}")
        print(f"   Target Beta: {self.constraints.target_beta} ± {self.constraints.beta_tolerance}")
        print(f"   Max Leverage: {self.constraints.max_leverage}x")
        print(f"   Target Volatility: {self.constraints.target_volatility:.1%}")
        
    def load_alpha_signals(self, signal_file: str) -> pd.DataFrame:
        """加载AlphaSignal数据"""
        try:
            signals_df = pd.read_csv(signal_file)
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            
            print(f"📊 Alpha Signals Loaded:")
            print(f"   Total Signals: {len(signals_df)}")
            print(f"   Symbols: {signals_df['symbol'].nunique()}")
            print(f"   Time Range: {signals_df['timestamp'].min()} to {signals_df['timestamp'].max()}")
            print(f"   Avg Confidence: {signals_df['confidence'].mean():.3f}")
            print(f"   Avg Predicted Return: {signals_df['predicted_return'].mean():.4f}")
            
            return signals_df
            
        except Exception as e:
            print(f"❌ Error loading alpha signals: {e}")
            return pd.DataFrame()
    
    def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """加载市场数据用于风险建模"""
        market_data = {}
        
        # 模拟市场数据（实际中应从数据源加载）
        symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT']
        
        print(f"📈 Generating Market Data for Risk Modeling...")
        
        for symbol in symbols:
            # 生成模拟价格数据
            dates = pd.date_range('2023-01-01', '2025-08-18', freq='D')
            np.random.seed(42)  # 可重复性
            
            # 模拟价格走势
            if symbol == 'BTCUSDT':
                base_return = 0.0003
                volatility = 0.04
            elif symbol == 'ETHUSDT': 
                base_return = 0.0002
                volatility = 0.045
            else:
                base_return = 0.0001
                volatility = 0.05
                
            returns = np.random.normal(base_return, volatility, len(dates))
            prices = 100 * np.exp(np.cumsum(returns))
            
            market_data[symbol] = pd.DataFrame({
                'timestamp': dates,
                'close': prices,
                'returns': returns
            })
            
        print(f"   Market Data Generated for {len(market_data)} symbols")
        return market_data
    
    def estimate_risk_model(self, signals_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """估计风险模型"""
        # 获取涉及的交易品种
        symbols = signals_df['symbol'].unique()
        
        # 加载市场数据
        self.market_data = self.load_market_data()
        
        # 构建收益率矩阵
        returns_data = []
        for symbol in symbols:
            if symbol in self.market_data:
                returns_data.append(self.market_data[symbol]['returns'])
            else:
                # 如果没有数据，使用默认风险参数
                returns_data.append(np.random.normal(0, 0.03, 500))
        
        # 确保我们有足够的数据点
        if len(returns_data) == 0:
            # 创建单资产情况的默认风险参数
            returns_data = [np.random.normal(0, 0.03, 500)]
            symbols = ['DEFAULT']
        
        returns_matrix = np.array(returns_data).T
        returns_df = pd.DataFrame(returns_matrix, columns=symbols)
        
        # 估计协方差矩阵
        self.covariance_matrix = self.cov_estimator.estimate_covariance(returns_df)
        
        # 确保协方差矩阵是2维的
        if self.covariance_matrix.ndim == 0:
            self.covariance_matrix = np.array([[self.covariance_matrix]])
        elif self.covariance_matrix.ndim == 1:
            self.covariance_matrix = np.array([self.covariance_matrix])
        
        # 计算相关性矩阵
        if len(symbols) > 1:
            self.correlation_matrix = np.corrcoef(returns_matrix.T)
            if self.correlation_matrix.ndim == 0:
                self.correlation_matrix = np.array([[1.0]])
        else:
            self.correlation_matrix = np.array([[1.0]])
        
        print(f"🔍 Risk Model Estimation Complete:")
        print(f"   Assets: {len(symbols)}")
        print(f"   Covariance Matrix: {self.covariance_matrix.shape}")
        if len(symbols) > 1:
            print(f"   Average Correlation: {np.mean(self.correlation_matrix[np.triu_indices_from(self.correlation_matrix, k=1)]):.3f}")
        else:
            print(f"   Single Asset Volatility: {np.sqrt(self.covariance_matrix[0,0]):.3f}")
        
        return self.covariance_matrix, self.correlation_matrix
    
    def calculate_kelly_weights(self, signals_df: pd.DataFrame) -> Dict[str, float]:
        """计算Kelly准则最优权重"""
        kelly_weights = {}
        
        for _, signal in signals_df.iterrows():
            symbol = signal['symbol']
            expected_return = signal['predicted_return']
            confidence = signal['confidence']
            
            # Kelly公式：f* = (bp - q) / b
            # b = 期望收益率, p = 胜率, q = 败率
            win_rate = confidence  # 置信度作为胜率
            loss_rate = 1 - win_rate
            
            if expected_return > 0 and win_rate > 0.5:
                # 保守的Kelly权重（使用25%的Kelly）
                kelly_fraction = 0.25 * (expected_return * win_rate - loss_rate) / expected_return
                kelly_weights[symbol] = max(0, min(kelly_fraction, self.constraints.max_position))
            else:
                kelly_weights[symbol] = 0
        
        return kelly_weights
    
    def optimize_portfolio(self, signals_df: pd.DataFrame) -> Tuple[Dict[str, float], Dict]:
        """执行组合优化"""
        if signals_df.empty:
            return {}, {}
        
        # 估计风险模型
        cov_matrix, corr_matrix = self.estimate_risk_model(signals_df)
        
        # 计算Kelly权重作为初始解
        kelly_weights = self.calculate_kelly_weights(signals_df)
        
        symbols = list(kelly_weights.keys())
        n_assets = len(symbols)
        
        if n_assets == 0:
            return {}, {}
        
        # 提取预期收益向量
        expected_returns = []
        for symbol in symbols:
            symbol_signals = signals_df[signals_df['symbol'] == symbol]
            if not symbol_signals.empty:
                expected_returns.append(symbol_signals['predicted_return'].mean())
            else:
                expected_returns.append(0)
        
        expected_returns = np.array(expected_returns)
        
        # 使用cvxpy进行凸优化
        print("🎯 Starting Portfolio Optimization...")
        
        try:
            # 定义优化变量
            w = cp.Variable(n_assets)
            
            # 目标函数：最大化风险调整收益
            portfolio_return = expected_returns.T @ w
            portfolio_risk = cp.quad_form(w, cov_matrix)
            
            # 目标：最大化夏普比率的近似
            risk_aversion = 1.0
            objective = cp.Maximize(portfolio_return - risk_aversion * portfolio_risk)
            
            # 约束条件
            constraints = []
            
            # 1. 预算约束（允许杠杆）
            constraints.append(cp.sum(cp.abs(w)) <= self.constraints.max_leverage)
            
            # 2. 市场中性约束（Beta中性）
            # 简化假设：所有资产Beta = 1
            market_betas = np.ones(n_assets)
            portfolio_beta = market_betas.T @ w
            constraints.append(cp.abs(portfolio_beta) <= self.constraints.beta_tolerance)
            
            # 3. 单仓位限制
            constraints.append(w >= -self.constraints.max_position)
            constraints.append(w <= self.constraints.max_position)
            
            # 4. 波动率约束 (使用二次形式避免DCP问题)
            constraints.append(cp.quad_form(w, cov_matrix) <= self.constraints.target_volatility**2)
            
            # 求解优化问题
            problem = cp.Problem(objective, constraints)
            problem.solve(solver=cp.OSQP, verbose=False, qcp=True)
            
            if problem.status == cp.OPTIMAL:
                optimal_weights = w.value
                optimized_weights = dict(zip(symbols, optimal_weights))
                
                # 计算优化后的风险指标
                portfolio_vol_final = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
                portfolio_return_final = expected_returns.T @ optimal_weights
                
                optimization_info = {
                    'status': 'OPTIMAL',
                    'objective_value': problem.value,
                    'portfolio_return': portfolio_return_final,
                    'portfolio_volatility': portfolio_vol_final,
                    'sharpe_ratio': portfolio_return_final / portfolio_vol_final if portfolio_vol_final > 0 else 0,
                    'leverage_used': np.sum(np.abs(optimal_weights)),
                    'portfolio_beta': np.sum(optimal_weights),  # 简化Beta计算
                    'total_positions': np.sum(optimal_weights != 0)
                }
                
                print(f"✅ Optimization Successful:")
                print(f"   Portfolio Return: {portfolio_return_final:.4f}")
                print(f"   Portfolio Volatility: {portfolio_vol_final:.4f}")
                print(f"   Sharpe Ratio: {optimization_info['sharpe_ratio']:.2f}")
                print(f"   Leverage Used: {optimization_info['leverage_used']:.2f}x")
                print(f"   Portfolio Beta: {optimization_info['portfolio_beta']:.4f}")
                
                return optimized_weights, optimization_info
            else:
                print(f"❌ Optimization Failed: {problem.status}")
                return kelly_weights, {'status': 'FAILED', 'fallback': 'KELLY_WEIGHTS'}
                
        except Exception as e:
            print(f"❌ Optimization Error: {e}")
            return kelly_weights, {'status': 'ERROR', 'fallback': 'KELLY_WEIGHTS'}
    
    def calculate_risk_metrics(self, weights: Dict[str, float]) -> RiskMetrics:
        """计算组合风险指标"""
        if not weights or self.covariance_matrix is None:
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        
        symbols = list(weights.keys())
        weight_vector = np.array([weights[s] for s in symbols])
        
        # 组合方差
        portfolio_variance = weight_vector.T @ self.covariance_matrix @ weight_vector
        ann_volatility = np.sqrt(portfolio_variance)
        
        # Beta（简化计算）
        portfolio_beta = np.sum(weight_vector)  # 假设所有资产Beta=1
        
        # VaR计算（正态分布假设）
        var_95 = 1.645 * ann_volatility / np.sqrt(252)  # 日度VaR
        var_99 = 2.33 * ann_volatility / np.sqrt(252)
        
        # Expected Shortfall (CVaR)
        es_95 = var_95 * 1.28  # 正态分布下的期望损失
        
        return RiskMetrics(
            ann_volatility=ann_volatility,
            beta=portfolio_beta,
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            sharpe_ratio=0.0,  # 需要收益率数据
            max_drawdown=0.0,  # 需要历史数据
            tracking_error=0.0  # 需要基准数据
        )
    
    def perform_stress_tests(self, weights: Dict[str, float]) -> Dict:
        """执行压力测试"""
        stress_scenarios = {
            'market_crash': {'factor': -0.20, 'description': '市场下跌20%'},
            'market_rally': {'factor': 0.15, 'description': '市场上涨15%'},
            'volatility_spike': {'factor': 2.0, 'description': '波动率翻倍'},
            'correlation_shock': {'factor': 0.9, 'description': '相关性升至0.9'}
        }
        
        stress_results = {}
        
        for scenario, params in stress_scenarios.items():
            if scenario == 'volatility_spike':
                # 波动率压力测试
                stressed_cov = self.covariance_matrix * params['factor']
                weight_vector = np.array([weights.get(s, 0) for s in weights.keys()])
                stressed_vol = np.sqrt(weight_vector.T @ stressed_cov @ weight_vector)
                
                stress_results[scenario] = {
                    'description': params['description'],
                    'stressed_volatility': stressed_vol,
                    'vol_change': stressed_vol / np.sqrt(weight_vector.T @ self.covariance_matrix @ weight_vector) - 1
                }
            
            elif scenario == 'correlation_shock':
                # 相关性冲击测试
                shocked_corr = np.full_like(self.correlation_matrix, params['factor'])
                np.fill_diagonal(shocked_corr, 1.0)
                
                # 重构协方差矩阵
                vol_diag = np.sqrt(np.diag(self.covariance_matrix))
                shocked_cov = np.outer(vol_diag, vol_diag) * shocked_corr
                
                weight_vector = np.array([weights.get(s, 0) for s in weights.keys()])
                stressed_vol = np.sqrt(weight_vector.T @ shocked_cov @ weight_vector)
                
                stress_results[scenario] = {
                    'description': params['description'],
                    'stressed_volatility': stressed_vol,
                    'vol_change': stressed_vol / np.sqrt(weight_vector.T @ self.covariance_matrix @ weight_vector) - 1
                }
            
            else:
                # 市场冲击测试
                portfolio_value_change = sum(weights.values()) * params['factor']
                stress_results[scenario] = {
                    'description': params['description'],
                    'portfolio_pnl': portfolio_value_change * self.base_capital,
                    'portfolio_return': portfolio_value_change
                }
        
        return stress_results
    
    def generate_risk_attribution(self, weights: Dict[str, float]) -> Dict:
        """生成风险归因分析"""
        if not weights or self.covariance_matrix is None:
            return {}
        
        symbols = list(weights.keys())
        weight_vector = np.array([weights[s] for s in symbols])
        
        # 边际风险贡献 (Marginal Contribution to Risk)
        mcr = self.covariance_matrix @ weight_vector
        portfolio_vol = np.sqrt(weight_vector.T @ self.covariance_matrix @ weight_vector)
        
        if portfolio_vol > 0:
            mcr = mcr / portfolio_vol
        
        # 成分风险贡献 (Component Contribution to Risk)
        ccr = weight_vector * mcr
        
        # 风险贡献百分比
        if np.sum(np.abs(ccr)) > 0:
            risk_contribution_pct = np.abs(ccr) / np.sum(np.abs(ccr))
        else:
            risk_contribution_pct = np.zeros(len(ccr))
        
        attribution = {
            'marginal_contribution': [
                {'symbol': symbol, 'mcr': float(mcr[i])}
                for i, symbol in enumerate(symbols)
            ],
            'component_contribution': [
                {'symbol': symbol, 'ccr': float(ccr[i]), 'risk_pct': float(risk_contribution_pct[i])}
                for i, symbol in enumerate(symbols)
            ],
            'diversification_ratio': portfolio_vol / np.sum(weight_vector * np.sqrt(np.diag(self.covariance_matrix))) if np.sum(weight_vector * np.sqrt(np.diag(self.covariance_matrix))) > 0 else 1.0
        }
        
        return attribution
    
    def generate_target_portfolio(self, weights: Dict[str, float], 
                                signals_df: pd.DataFrame,
                                optimization_info: Dict) -> Dict:
        """生成TargetPortfolio输出"""
        risk_metrics = self.calculate_risk_metrics(weights)
        stress_results = self.perform_stress_tests(weights)
        risk_attribution = self.generate_risk_attribution(weights)
        
        # 将权重转换为美元金额
        position_weights = []
        for symbol, weight in weights.items():
            if abs(weight) > 1e-6:  # 过滤极小权重
                signal_info = signals_df[signals_df['symbol'] == symbol].iloc[0] if not signals_df[signals_df['symbol'] == symbol].empty else None
                
                position_weights.append({
                    'symbol': symbol,
                    'weight': float(weight),
                    'dollar_amount': float(weight * self.base_capital),
                    'signal_strength': float(signal_info['signal']) if signal_info is not None else 0,
                    'confidence': float(signal_info['confidence']) if signal_info is not None else 0,
                    'expected_return': float(signal_info['predicted_return']) if signal_info is not None else 0
                })
        
        # 检查约束状态
        constraints_status = {
            'beta_neutral': bool(abs(risk_metrics.beta) <= self.constraints.beta_tolerance),
            'vol_target': bool(risk_metrics.ann_volatility <= self.constraints.target_volatility),
            'leverage_ok': bool(sum(abs(w) for w in weights.values()) <= self.constraints.max_leverage),
            'var_limit': bool(risk_metrics.var_95 <= self.constraints.max_var_95)
        }
        
        target_portfolio = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'optimizer_version': 'PortfolioRiskOptimizer_V4.0.0',
                'base_capital': self.base_capital,
                'optimization_status': optimization_info.get('status', 'UNKNOWN')
            },
            'positions': position_weights,
            'portfolio_metrics': {
                'total_positions': len(position_weights),
                'gross_exposure': sum(abs(pos['weight']) for pos in position_weights),
                'net_exposure': sum(pos['weight'] for pos in position_weights),
                'leverage': sum(abs(pos['weight']) for pos in position_weights),
                'long_exposure': sum(pos['weight'] for pos in position_weights if pos['weight'] > 0),
                'short_exposure': sum(pos['weight'] for pos in position_weights if pos['weight'] < 0)
            },
            'risk_metrics': {
                'annualized_volatility': float(risk_metrics.ann_volatility),
                'portfolio_beta': float(risk_metrics.beta),
                'var_95': float(risk_metrics.var_95),
                'var_99': float(risk_metrics.var_99),
                'expected_shortfall_95': float(risk_metrics.es_95),
                'sharpe_ratio': float(optimization_info.get('sharpe_ratio', 0)),
                'tracking_error': float(risk_metrics.tracking_error)
            },
            'stress_test_results': stress_results,
            'risk_attribution': risk_attribution,
            'constraints_compliance': constraints_status,
            'optimization_details': optimization_info
        }
        
        return target_portfolio
    
    def generate_risk_report(self, target_portfolio: Dict) -> Dict:
        """生成风险报告"""
        risk_report = {
            'report_metadata': {
                'generation_timestamp': datetime.now().isoformat(),
                'report_type': 'Portfolio_Risk_Assessment',
                'base_capital': self.base_capital
            },
            'executive_summary': {
                'total_positions': target_portfolio['portfolio_metrics']['total_positions'],
                'portfolio_leverage': target_portfolio['portfolio_metrics']['leverage'],
                'portfolio_beta': target_portfolio['risk_metrics']['portfolio_beta'],
                'annualized_volatility': target_portfolio['risk_metrics']['annualized_volatility'],
                'var_95_daily': target_portfolio['risk_metrics']['var_95'],
                'all_constraints_met': all(target_portfolio['constraints_compliance'].values()),
                'risk_assessment': 'LOW' if target_portfolio['risk_metrics']['var_95'] < 0.02 else 'MEDIUM' if target_portfolio['risk_metrics']['var_95'] < 0.04 else 'HIGH'
            },
            'detailed_risk_analysis': {
                'market_risk': {
                    'directional_risk': abs(target_portfolio['risk_metrics']['portfolio_beta']),
                    'volatility_risk': target_portfolio['risk_metrics']['annualized_volatility'],
                    'concentration_risk': max([abs(pos['weight']) for pos in target_portfolio['positions']] + [0])
                },
                'liquidity_risk': {
                    'assessment': 'LOW',  # 简化评估
                    'position_sizes': [pos['dollar_amount'] for pos in target_portfolio['positions']],
                    'max_position_size': max([abs(pos['dollar_amount']) for pos in target_portfolio['positions']] + [0])
                }
            },
            'risk_monitoring': {
                'daily_var_limit': self.constraints.max_var_95,
                'current_var_usage': target_portfolio['risk_metrics']['var_95'] / self.constraints.max_var_95,
                'leverage_limit': self.constraints.max_leverage,
                'current_leverage_usage': target_portfolio['portfolio_metrics']['leverage'] / self.constraints.max_leverage,
                'beta_limit': self.constraints.beta_tolerance,
                'current_beta': abs(target_portfolio['risk_metrics']['portfolio_beta'])
            },
            'recommendations': []
        }
        
        # 生成建议
        if not target_portfolio['constraints_compliance']['beta_neutral']:
            risk_report['recommendations'].append({
                'type': 'BETA_ADJUSTMENT',
                'priority': 'HIGH',
                'description': f"Portfolio beta ({target_portfolio['risk_metrics']['portfolio_beta']:.3f}) exceeds tolerance. Consider hedging."
            })
        
        if not target_portfolio['constraints_compliance']['vol_target']:
            risk_report['recommendations'].append({
                'type': 'VOLATILITY_REDUCTION',
                'priority': 'MEDIUM', 
                'description': f"Portfolio volatility ({target_portfolio['risk_metrics']['annualized_volatility']:.3f}) exceeds target. Consider position sizing reduction."
            })
        
        if target_portfolio['portfolio_metrics']['leverage'] > self.constraints.max_leverage * 0.9:
            risk_report['recommendations'].append({
                'type': 'LEVERAGE_WARNING',
                'priority': 'MEDIUM',
                'description': f"Leverage usage at {target_portfolio['portfolio_metrics']['leverage']:.2f}x is near limit."
            })
        
        return risk_report


def main():
    """主执行函数"""
    print("🚀 DipMaster Portfolio Risk Optimizer V4.0.0")
    print("=" * 60)
    
    # 初始化优化器
    constraints = PortfolioConstraints(
        target_beta=0.0,
        beta_tolerance=0.15,
        max_position=0.08,
        max_leverage=2.5,
        target_volatility=0.15,
        max_var_95=0.03
    )
    
    optimizer = PortfolioRiskOptimizer(
        base_capital=100000,
        constraints=constraints
    )
    
    # 加载AlphaSignal
    signal_file = "results/basic_ml_pipeline/signals_20250818_153608.csv"
    signals_df = optimizer.load_alpha_signals(signal_file)
    
    if signals_df.empty:
        print("❌ No signals loaded. Exiting.")
        return
    
    # 执行组合优化
    print("\n🎯 Starting Portfolio Optimization Process...")
    weights, optimization_info = optimizer.optimize_portfolio(signals_df)
    
    if not weights:
        print("❌ Portfolio optimization failed. Exiting.")
        return
    
    # 生成目标组合
    target_portfolio = optimizer.generate_target_portfolio(weights, signals_df, optimization_info)
    
    # 生成风险报告
    risk_report = optimizer.generate_risk_report(target_portfolio)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存TargetPortfolio
    portfolio_file = f"results/portfolio_optimization/TargetPortfolio_{timestamp}.json"
    with open(portfolio_file, 'w') as f:
        json.dump(target_portfolio, f, indent=2)
    
    # 保存RiskReport
    risk_file = f"results/portfolio_optimization/RiskReport_{timestamp}.json"
    with open(risk_file, 'w') as f:
        json.dump(risk_report, f, indent=2)
    
    print(f"\n✅ Portfolio Optimization Complete!")
    print(f"📊 Target Portfolio: {portfolio_file}")
    print(f"📋 Risk Report: {risk_file}")
    print(f"\n🎯 Portfolio Summary:")
    print(f"   Total Positions: {len(target_portfolio['positions'])}")
    print(f"   Net Exposure: {target_portfolio['portfolio_metrics']['net_exposure']:.4f}")
    print(f"   Gross Exposure: {target_portfolio['portfolio_metrics']['gross_exposure']:.4f}")
    print(f"   Portfolio Beta: {target_portfolio['risk_metrics']['portfolio_beta']:.4f}")
    print(f"   Annualized Vol: {target_portfolio['risk_metrics']['annualized_volatility']:.4f}")
    print(f"   VaR (95%): {target_portfolio['risk_metrics']['var_95']:.4f}")
    print(f"   All Constraints Met: {all(target_portfolio['constraints_compliance'].values())}")
    
    return target_portfolio, risk_report


if __name__ == "__main__":
    main()