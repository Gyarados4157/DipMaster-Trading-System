#!/usr/bin/env python3
"""
DipMaster实时风险监控系统
Real-time Risk Monitoring System

核心功能：
1. 实时VaR和ES计算
2. 压力测试和情景分析  
3. 相关性矩阵监控
4. 流动性风险评估
5. 风险归因分析
6. 自动化风险报告

作者: DipMaster Trading System
版本: V1.0.0 - Real-time Risk Monitoring
"""

import pandas as pd
import numpy as np
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# 数值计算和统计
from scipy import stats
from scipy.optimize import minimize
from sklearn.covariance import EmpiricalCovariance, LedoitWolf
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('RealTimeRiskMonitor')

@dataclass
class RiskThresholds:
    """风险阈值配置"""
    # VaR和ES阈值
    var_95_daily: float = 0.03
    var_99_daily: float = 0.04
    es_95_daily: float = 0.04
    
    # 波动率阈值
    portfolio_vol_annual: float = 0.18
    position_vol_annual: float = 0.25
    
    # Beta和相关性阈值
    portfolio_beta: float = 0.10
    max_correlation: float = 0.70
    min_diversification: float = 1.20
    
    # 流动性阈值
    max_position_size: float = 0.20
    liquidity_score_min: float = 0.60

@dataclass
class StressTestScenario:
    """压力测试情景"""
    name: str
    description: str
    market_shock: float  # 市场整体冲击
    volatility_multiplier: float  # 波动率倍数
    correlation_shock: Optional[float] = None  # 相关性冲击

@dataclass
class RiskMetric:
    """风险指标"""
    metric_name: str
    current_value: float
    threshold: float
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: datetime

class RealTimeRiskMonitor:
    """实时风险监控器"""
    
    def __init__(self, thresholds: RiskThresholds, lookback_days: int = 252):
        self.thresholds = thresholds
        self.lookback_days = lookback_days
        
        # 风险计算组件
        self.risk_models = {
            'empirical': EmpiricalCovariance(),
            'ledoit_wolf': LedoitWolf()
        }
        
        # 历史数据缓存
        self.price_history: Dict[str, pd.Series] = {}
        self.return_history: Dict[str, pd.Series] = {}
        self.correlation_history: List[np.ndarray] = []
        self.risk_metrics_history: List[Dict] = []
        
        # 压力测试情景
        self.stress_scenarios = [
            StressTestScenario("market_crash", "Market Crash (-20%)", -0.20, 2.0),
            StressTestScenario("flash_crash", "Flash Crash (-10%)", -0.10, 3.0),
            StressTestScenario("volatility_spike", "Volatility Spike", 0.0, 2.5),
            StressTestScenario("correlation_crisis", "Correlation Crisis", -0.05, 1.5, 0.9),
            StressTestScenario("liquidity_crunch", "Liquidity Crunch", -0.08, 2.0, 0.8),
        ]
        
        logger.info("Real-time Risk Monitor initialized")

    def load_historical_data(self, symbols: List[str], days: int = None) -> Dict[str, pd.DataFrame]:
        """加载历史市场数据"""
        if days is None:
            days = self.lookback_days
            
        # 模拟历史价格数据（实际中应从数据源加载）
        historical_data = {}
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start_date, end_date, freq='D')
        
        for symbol in symbols:
            np.random.seed(hash(symbol) % 2**32)  # 每个symbol使用不同的随机种子
            
            # 根据symbol类型设置不同的市场参数
            if 'BTC' in symbol:
                drift = 0.0005
                volatility = 0.04
                initial_price = 65000
            elif 'ETH' in symbol:
                drift = 0.0003
                volatility = 0.045
                initial_price = 3500
            elif 'BNB' in symbol:
                drift = 0.0002
                volatility = 0.05
                initial_price = 600
            else:
                drift = 0.0001
                volatility = 0.06
                initial_price = 100
            
            # 生成几何布朗运动
            returns = np.random.normal(drift, volatility, len(dates))
            prices = initial_price * np.exp(np.cumsum(returns))
            
            historical_data[symbol] = pd.DataFrame({
                'timestamp': dates,
                'close': prices,
                'volume': np.random.uniform(1e6, 1e8, len(dates)),
                'returns': returns
            })
            
            # 缓存到历史数据
            self.price_history[symbol] = pd.Series(prices, index=dates)
            self.return_history[symbol] = pd.Series(returns, index=dates)
        
        logger.info(f"Loaded historical data for {len(symbols)} symbols over {days} days")
        return historical_data

    def calculate_portfolio_var_es(self, positions: Dict[str, float], 
                                  confidence_level: float = 0.95,
                                  method: str = 'parametric') -> Tuple[float, float]:
        """计算组合VaR和Expected Shortfall"""
        if not positions:
            return 0.0, 0.0
            
        symbols = list(positions.keys())
        weights = np.array(list(positions.values()))
        
        # 获取收益率数据
        returns_matrix = []
        for symbol in symbols:
            if symbol in self.return_history and len(self.return_history[symbol]) > 0:
                returns_matrix.append(self.return_history[symbol].iloc[-self.lookback_days:])
            else:
                # 如果没有历史数据，使用模拟数据
                returns_matrix.append(pd.Series(np.random.normal(0, 0.03, self.lookback_days)))
        
        if not returns_matrix:
            return 0.0, 0.0
            
        returns_df = pd.DataFrame(returns_matrix).T
        returns_df.columns = symbols
        returns_df = returns_df.fillna(0)
        
        if method == 'parametric':
            # 参数法：假设正态分布
            portfolio_returns = returns_df @ weights
            portfolio_mean = portfolio_returns.mean()
            portfolio_std = portfolio_returns.std()
            
            # VaR计算
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(portfolio_mean + z_score * portfolio_std)
            
            # Expected Shortfall (条件期望损失)
            es = -(portfolio_mean - portfolio_std * stats.norm.pdf(z_score) / (1 - confidence_level))
            
        elif method == 'historical':
            # 历史模拟法
            portfolio_returns = returns_df @ weights
            sorted_returns = portfolio_returns.sort_values()
            
            var_index = int((1 - confidence_level) * len(sorted_returns))
            var = -sorted_returns.iloc[var_index]
            
            # ES是超过VaR的平均损失
            tail_returns = sorted_returns.iloc[:var_index]
            es = -tail_returns.mean() if len(tail_returns) > 0 else var
            
        else:  # monte_carlo
            # 蒙特卡洛模拟
            n_simulations = 10000
            
            # 估计协方差矩阵
            cov_matrix = returns_df.cov().values
            mean_returns = returns_df.mean().values
            
            # 生成随机收益率
            simulated_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, n_simulations
            )
            
            # 计算组合收益率
            portfolio_returns = simulated_returns @ weights
            sorted_returns = np.sort(portfolio_returns)
            
            var_index = int((1 - confidence_level) * n_simulations)
            var = -sorted_returns[var_index]
            
            tail_returns = sorted_returns[:var_index]
            es = -np.mean(tail_returns) if len(tail_returns) > 0 else var
        
        return var, es

    def calculate_correlation_matrix(self, symbols: List[str]) -> np.ndarray:
        """计算相关性矩阵"""
        if len(symbols) <= 1:
            return np.array([[1.0]])
            
        returns_data = []
        for symbol in symbols:
            if symbol in self.return_history:
                returns_data.append(self.return_history[symbol].iloc[-self.lookback_days:])
            else:
                # 生成模拟数据
                returns_data.append(pd.Series(np.random.normal(0, 0.03, self.lookback_days)))
        
        returns_df = pd.DataFrame(returns_data).T
        correlation_matrix = returns_df.corr().values
        
        # 缓存相关性历史
        self.correlation_history.append(correlation_matrix)
        
        # 只保留最近的100个相关性矩阵
        if len(self.correlation_history) > 100:
            self.correlation_history = self.correlation_history[-100:]
            
        return correlation_matrix

    def perform_stress_testing(self, positions: Dict[str, float]) -> Dict[str, Dict]:
        """执行压力测试"""
        stress_results = {}
        
        for scenario in self.stress_scenarios:
            symbols = list(positions.keys())
            weights = np.array(list(positions.values()))
            
            # 基础组合价值
            portfolio_value = np.sum(np.abs(weights))
            
            if scenario.name == "correlation_crisis" and scenario.correlation_shock:
                # 相关性冲击测试
                # 重新计算在高相关性下的风险
                shocked_corr = np.full((len(symbols), len(symbols)), scenario.correlation_shock)
                np.fill_diagonal(shocked_corr, 1.0)
                
                # 假设波动率
                vols = np.array([0.05] * len(symbols))  # 假设5%波动率
                shocked_cov = np.outer(vols, vols) * shocked_corr
                
                portfolio_var = weights.T @ shocked_cov @ weights
                portfolio_vol = np.sqrt(portfolio_var)
                
                shock_loss = portfolio_vol * scenario.volatility_multiplier
                
            else:
                # 市场冲击和波动率冲击
                market_loss = portfolio_value * abs(scenario.market_shock)
                volatility_loss = portfolio_value * 0.02 * scenario.volatility_multiplier  # 假设2%基础波动率
                shock_loss = market_loss + volatility_loss
            
            # 计算冲击后的VaR
            base_var, _ = self.calculate_portfolio_var_es(positions, 0.95)
            stressed_var = base_var * scenario.volatility_multiplier
            
            stress_results[scenario.name] = {
                'description': scenario.description,
                'market_shock': scenario.market_shock,
                'volatility_multiplier': scenario.volatility_multiplier,
                'estimated_loss': shock_loss,
                'loss_percentage': shock_loss / portfolio_value if portfolio_value > 0 else 0,
                'stressed_var_95': stressed_var,
                'risk_level': self._assess_risk_level(shock_loss / portfolio_value if portfolio_value > 0 else 0)
            }
        
        return stress_results

    def calculate_risk_attribution(self, positions: Dict[str, float]) -> Dict:
        """计算风险归因"""
        if not positions:
            return {}
            
        symbols = list(positions.keys())
        weights = np.array(list(positions.values()))
        
        # 构建收益率矩阵
        returns_matrix = []
        for symbol in symbols:
            if symbol in self.return_history:
                returns_matrix.append(self.return_history[symbol].iloc[-self.lookback_days:])
            else:
                returns_matrix.append(pd.Series(np.random.normal(0, 0.03, self.lookback_days)))
        
        returns_df = pd.DataFrame(returns_matrix).T
        returns_df.columns = symbols
        
        # 估计协方差矩阵
        cov_matrix = returns_df.cov().values
        
        # 组合方差
        portfolio_var = weights.T @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_var)
        
        # 边际风险贡献 (Marginal Contribution to Risk)
        if portfolio_vol > 0:
            mcr = (cov_matrix @ weights) / portfolio_vol
        else:
            mcr = np.zeros(len(weights))
        
        # 成分风险贡献 (Component Contribution to Risk)
        ccr = weights * mcr
        
        # 风险贡献百分比
        total_risk = np.sum(np.abs(ccr))
        if total_risk > 0:
            risk_contribution_pct = np.abs(ccr) / total_risk
        else:
            risk_contribution_pct = np.zeros(len(ccr))
        
        attribution = {
            'marginal_contribution': {
                symbols[i]: float(mcr[i]) for i in range(len(symbols))
            },
            'component_contribution': {
                symbols[i]: float(ccr[i]) for i in range(len(symbols))
            },
            'risk_percentage': {
                symbols[i]: float(risk_contribution_pct[i]) for i in range(len(symbols))
            },
            'portfolio_volatility': float(portfolio_vol),
            'diversification_ratio': float(portfolio_vol / np.sum(weights * np.sqrt(np.diag(cov_matrix)))) if np.sum(weights * np.sqrt(np.diag(cov_matrix))) > 0 else 1.0
        }
        
        return attribution

    def assess_liquidity_risk(self, positions: Dict[str, float]) -> Dict:
        """评估流动性风险"""
        liquidity_assessment = {}
        
        for symbol, weight in positions.items():
            # 简化的流动性评分（实际中应基于交易量、买卖价差等）
            if 'BTC' in symbol or 'ETH' in symbol:
                liquidity_score = 0.95  # 高流动性
                days_to_liquidate = 0.5
            elif 'BNB' in symbol or 'SOL' in symbol or 'ADA' in symbol:
                liquidity_score = 0.85  # 中高流动性
                days_to_liquidate = 1.0
            else:
                liquidity_score = 0.70  # 中等流动性
                days_to_liquidate = 2.0
            
            position_size_impact = min(abs(weight) / self.thresholds.max_position_size, 1.0)
            adjusted_liquidity_score = liquidity_score * (1 - position_size_impact * 0.2)
            
            liquidity_assessment[symbol] = {
                'base_liquidity_score': liquidity_score,
                'adjusted_liquidity_score': adjusted_liquidity_score,
                'days_to_liquidate': days_to_liquidate * (1 + position_size_impact),
                'position_size_impact': position_size_impact,
                'liquidity_risk_level': 'LOW' if adjusted_liquidity_score > 0.8 else 'MEDIUM' if adjusted_liquidity_score > 0.6 else 'HIGH'
            }
        
        # 组合级别流动性风险
        avg_liquidity_score = np.mean([assess['adjusted_liquidity_score'] for assess in liquidity_assessment.values()])
        max_liquidation_time = max([assess['days_to_liquidate'] for assess in liquidity_assessment.values()] + [0])
        
        portfolio_liquidity = {
            'average_liquidity_score': avg_liquidity_score,
            'max_liquidation_days': max_liquidation_time,
            'portfolio_liquidity_risk': 'LOW' if avg_liquidity_score > 0.8 else 'MEDIUM' if avg_liquidity_score > 0.6 else 'HIGH',
            'positions_detail': liquidity_assessment
        }
        
        return portfolio_liquidity

    def _assess_risk_level(self, value: float, thresholds: Tuple[float, float, float] = (0.02, 0.05, 0.10)) -> str:
        """评估风险等级"""
        low, medium, high = thresholds
        if value < low:
            return 'LOW'
        elif value < medium:
            return 'MEDIUM'  
        elif value < high:
            return 'HIGH'
        else:
            return 'CRITICAL'

    def generate_comprehensive_risk_report(self, positions: Dict[str, float]) -> Dict:
        """生成综合风险报告"""
        if not positions:
            return {'status': 'NO_POSITIONS', 'timestamp': datetime.now().isoformat()}
        
        symbols = list(positions.keys())
        
        # 加载历史数据
        self.load_historical_data(symbols)
        
        # 计算各种风险指标
        var_95, es_95 = self.calculate_portfolio_var_es(positions, 0.95, 'parametric')
        var_99, es_99 = self.calculate_portfolio_var_es(positions, 0.99, 'parametric')
        var_95_hist, es_95_hist = self.calculate_portfolio_var_es(positions, 0.95, 'historical')
        
        # 相关性分析
        correlation_matrix = self.calculate_correlation_matrix(symbols)
        
        # 压力测试
        stress_results = self.perform_stress_testing(positions)
        
        # 风险归因
        risk_attribution = self.calculate_risk_attribution(positions)
        
        # 流动性风险
        liquidity_risk = self.assess_liquidity_risk(positions)
        
        # 组合统计
        portfolio_stats = {
            'total_positions': len(positions),
            'gross_exposure': sum(abs(w) for w in positions.values()),
            'net_exposure': sum(positions.values()),
            'max_position_size': max(abs(w) for w in positions.values()),
            'leverage': sum(abs(w) for w in positions.values())
        }
        
        # 风险指标汇总
        risk_metrics = {
            'var_95_parametric': var_95,
            'var_99_parametric': var_99,
            'var_95_historical': var_95_hist,
            'es_95_parametric': es_95,
            'es_99_parametric': es_99,
            'es_95_historical': es_95_hist,
            'portfolio_volatility': risk_attribution.get('portfolio_volatility', 0),
            'diversification_ratio': risk_attribution.get('diversification_ratio', 1),
            'avg_correlation': float(np.mean(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])) if len(symbols) > 1 else 0,
            'max_correlation': float(np.max(correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)])) if len(symbols) > 1 else 0
        }
        
        # 风险等级评估
        risk_levels = {
            'var_95_level': self._assess_risk_level(var_95, (0.01, 0.02, 0.04)),
            'volatility_level': self._assess_risk_level(risk_metrics['portfolio_volatility'], (0.10, 0.15, 0.25)),
            'concentration_level': self._assess_risk_level(portfolio_stats['max_position_size'], (0.10, 0.20, 0.40)),
            'correlation_level': self._assess_risk_level(risk_metrics['max_correlation'], (0.50, 0.70, 0.85)),
            'liquidity_level': liquidity_risk['portfolio_liquidity_risk']
        }
        
        # 风险限制检查
        limit_violations = []
        if var_95 > self.thresholds.var_95_daily:
            limit_violations.append({
                'type': 'VAR_95_VIOLATION',
                'current': var_95,
                'threshold': self.thresholds.var_95_daily,
                'severity': 'HIGH'
            })
        
        if risk_metrics['portfolio_volatility'] > self.thresholds.portfolio_vol_annual:
            limit_violations.append({
                'type': 'VOLATILITY_VIOLATION', 
                'current': risk_metrics['portfolio_volatility'],
                'threshold': self.thresholds.portfolio_vol_annual,
                'severity': 'MEDIUM'
            })
        
        if risk_metrics['max_correlation'] > self.thresholds.max_correlation:
            limit_violations.append({
                'type': 'CORRELATION_VIOLATION',
                'current': risk_metrics['max_correlation'],
                'threshold': self.thresholds.max_correlation,
                'severity': 'MEDIUM'
            })
        
        # 编制最终报告
        comprehensive_report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'report_type': 'COMPREHENSIVE_RISK_ASSESSMENT',
                'positions_analyzed': len(positions),
                'risk_calculation_method': 'MULTI_METHOD_ENSEMBLE'
            },
            'portfolio_statistics': portfolio_stats,
            'risk_metrics': risk_metrics,
            'risk_levels': risk_levels,
            'limit_violations': limit_violations,
            'stress_test_results': stress_results,
            'risk_attribution': risk_attribution,
            'liquidity_assessment': liquidity_risk,
            'correlation_analysis': {
                'correlation_matrix': correlation_matrix.tolist(),
                'symbol_pairs': list(zip(symbols, symbols)) if len(symbols) > 1 else [],
                'high_correlation_pairs': [
                    (symbols[i], symbols[j], float(correlation_matrix[i, j]))
                    for i in range(len(symbols))
                    for j in range(i+1, len(symbols))
                    if abs(correlation_matrix[i, j]) > self.thresholds.max_correlation
                ] if len(symbols) > 1 else []
            },
            'recommendations': self._generate_risk_recommendations(risk_levels, limit_violations, stress_results)
        }
        
        return comprehensive_report

    def _generate_risk_recommendations(self, risk_levels: Dict, violations: List, stress_results: Dict) -> List[Dict]:
        """生成风险建议"""
        recommendations = []
        
        # 基于违规情况生成建议
        for violation in violations:
            if violation['type'] == 'VAR_95_VIOLATION':
                recommendations.append({
                    'type': 'REDUCE_POSITION_SIZES',
                    'priority': 'HIGH',
                    'description': f"VaR exceeds limit. Consider reducing position sizes by {((violation['current']/violation['threshold'] - 1) * 100):.1f}%"
                })
            elif violation['type'] == 'CORRELATION_VIOLATION':
                recommendations.append({
                    'type': 'DIVERSIFY_HOLDINGS',
                    'priority': 'MEDIUM',
                    'description': "High correlation detected. Consider diversifying into less correlated assets"
                })
        
        # 基于压力测试结果生成建议
        critical_scenarios = [name for name, result in stress_results.items() 
                            if result.get('risk_level') in ['HIGH', 'CRITICAL']]
        
        if critical_scenarios:
            recommendations.append({
                'type': 'STRESS_TEST_HEDGE',
                'priority': 'HIGH', 
                'description': f"Portfolio vulnerable to scenarios: {', '.join(critical_scenarios)}. Consider hedging strategies."
            })
        
        # 基于风险等级生成建议
        if risk_levels.get('concentration_level') in ['HIGH', 'CRITICAL']:
            recommendations.append({
                'type': 'REDUCE_CONCENTRATION',
                'priority': 'MEDIUM',
                'description': "High concentration risk detected. Spread investments across more positions."
            })
        
        if risk_levels.get('liquidity_level') in ['HIGH', 'CRITICAL']:
            recommendations.append({
                'type': 'IMPROVE_LIQUIDITY',
                'priority': 'MEDIUM',
                'description': "Liquidity risk elevated. Consider positions in more liquid assets."
            })
        
        return recommendations

    def create_risk_visualization(self, risk_report: Dict, output_path: str):
        """创建风险可视化图表"""
        try:
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=['风险指标概览', '相关性热力图', '压力测试结果', 
                               '风险归因', '流动性评估', '历史VaR趋势'],
                specs=[[{'type': 'bar'}, {'type': 'heatmap'}],
                       [{'type': 'bar'}, {'type': 'bar'}], 
                       [{'type': 'bar'}, {'type': 'scatter'}]]
            )
            
            # 1. 风险指标概览
            risk_metrics = risk_report['risk_metrics']
            metrics_names = ['VaR 95%', 'ES 95%', '组合波动率', '最大相关性']
            metrics_values = [
                risk_metrics['var_95_parametric'],
                risk_metrics['es_95_parametric'], 
                risk_metrics['portfolio_volatility'],
                risk_metrics['max_correlation']
            ]
            
            fig.add_trace(
                go.Bar(x=metrics_names, y=metrics_values, name='Risk Metrics'),
                row=1, col=1
            )
            
            # 2. 相关性热力图
            if 'correlation_matrix' in risk_report['correlation_analysis']:
                corr_matrix = np.array(risk_report['correlation_analysis']['correlation_matrix'])
                symbols = list(risk_report['risk_attribution']['marginal_contribution'].keys())
                
                fig.add_trace(
                    go.Heatmap(z=corr_matrix, x=symbols, y=symbols, name='Correlation'),
                    row=1, col=2
                )
            
            # 3. 压力测试结果
            stress_names = list(risk_report['stress_test_results'].keys())
            stress_losses = [result['loss_percentage'] for result in risk_report['stress_test_results'].values()]
            
            fig.add_trace(
                go.Bar(x=stress_names, y=stress_losses, name='Stress Loss %'),
                row=2, col=1
            )
            
            # 4. 风险归因
            if 'risk_percentage' in risk_report['risk_attribution']:
                symbols = list(risk_report['risk_attribution']['risk_percentage'].keys())
                risk_contribs = list(risk_report['risk_attribution']['risk_percentage'].values())
                
                fig.add_trace(
                    go.Bar(x=symbols, y=risk_contribs, name='Risk Contribution %'),
                    row=2, col=2
                )
            
            # 5. 流动性评估
            if 'positions_detail' in risk_report['liquidity_assessment']:
                liquidity_data = risk_report['liquidity_assessment']['positions_detail']
                symbols = list(liquidity_data.keys())
                liquidity_scores = [data['adjusted_liquidity_score'] for data in liquidity_data.values()]
                
                fig.add_trace(
                    go.Bar(x=symbols, y=liquidity_scores, name='Liquidity Score'),
                    row=3, col=1
                )
            
            # 6. 历史VaR趋势（模拟数据）
            dates = pd.date_range(datetime.now() - timedelta(days=30), datetime.now(), freq='D')
            var_trend = np.random.uniform(0.01, 0.04, len(dates))
            
            fig.add_trace(
                go.Scatter(x=dates, y=var_trend, mode='lines', name='VaR 95% Trend'),
                row=3, col=2
            )
            
            fig.update_layout(
                height=1200,
                title="DipMaster实时风险监控仪表板",
                showlegend=False
            )
            
            # 保存图表
            pio.write_html(fig, output_path)
            logger.info(f"Risk visualization saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating risk visualization: {e}")

def main():
    """主执行函数"""
    print("🚀 DipMaster Real-time Risk Monitoring System")
    print("=" * 60)
    
    # 配置风险阈值
    thresholds = RiskThresholds(
        var_95_daily=0.03,
        portfolio_vol_annual=0.18,
        portfolio_beta=0.10,
        max_correlation=0.70
    )
    
    # 创建监控器
    monitor = RealTimeRiskMonitor(thresholds)
    
    # 模拟组合仓位
    test_positions = {
        'BTCUSDT': 0.40,
        'ETHUSDT': 0.35,
        'BNBUSDT': 0.15,
        'SOLUSDT': 0.10
    }
    
    print(f"\n📊 Analyzing Test Portfolio:")
    for symbol, weight in test_positions.items():
        print(f"   {symbol}: {weight:.2f} ({weight*100:.1f}%)")
    
    # 生成综合风险报告
    print(f"\n🔍 Generating Comprehensive Risk Report...")
    risk_report = monitor.generate_comprehensive_risk_report(test_positions)
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results/risk_monitoring")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = output_dir / f"risk_report_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(risk_report, f, indent=2, default=str)
    
    # 创建可视化
    viz_file = output_dir / f"risk_dashboard_{timestamp}.html"
    monitor.create_risk_visualization(risk_report, str(viz_file))
    
    # 显示关键结果
    print(f"\n📋 Risk Assessment Summary:")
    print(f"   VaR (95%): {risk_report['risk_metrics']['var_95_parametric']:.4f}")
    print(f"   Expected Shortfall (95%): {risk_report['risk_metrics']['es_95_parametric']:.4f}")
    print(f"   Portfolio Volatility: {risk_report['risk_metrics']['portfolio_volatility']:.4f}")
    print(f"   Max Correlation: {risk_report['risk_metrics']['max_correlation']:.3f}")
    print(f"   Diversification Ratio: {risk_report['risk_metrics']['diversification_ratio']:.3f}")
    print(f"   Liquidity Risk: {risk_report['liquidity_assessment']['portfolio_liquidity_risk']}")
    
    if risk_report['limit_violations']:
        print(f"\n⚠️ Risk Limit Violations:")
        for violation in risk_report['limit_violations']:
            print(f"   {violation['type']}: {violation['current']:.4f} > {violation['threshold']:.4f}")
    else:
        print(f"\n✅ All Risk Limits Within Thresholds")
    
    if risk_report['recommendations']:
        print(f"\n💡 Risk Management Recommendations:")
        for rec in risk_report['recommendations'][:3]:  # 显示前3个建议
            print(f"   [{rec['priority']}] {rec['type']}: {rec['description']}")
    
    print(f"\n💾 Results saved:")
    print(f"   Risk Report: {report_file}")
    print(f"   Risk Dashboard: {viz_file}")
    
    return risk_report

if __name__ == "__main__":
    main()