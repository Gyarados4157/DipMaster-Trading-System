"""
DipMaster Enhanced V4 - 实时风险管理系统
多层级风险控制和监控框架

作者: DipMaster Trading System  
版本: 4.0.0
创建时间: 2025-08-16
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
warnings.filterwarnings('ignore')

@dataclass
class RiskMetrics:
    """风险指标数据类"""
    portfolio_var_95: float
    portfolio_var_99: float
    portfolio_es_95: float
    portfolio_volatility: float
    beta: float
    max_drawdown: float
    sharpe_ratio: float
    calmar_ratio: float
    correlation_risk: float
    concentration_risk: float

@dataclass
class StressTestResult:
    """压力测试结果"""
    scenario_name: str
    portfolio_change: float
    worst_position_change: float
    correlation_breakdown: bool
    liquidity_impact: float

class RealTimeRiskManager:
    """
    实时风险管理系统
    
    功能特性:
    1. 实时VaR和ES计算
    2. 动态相关性监控
    3. 压力测试和情景分析
    4. 流动性风险评估
    5. 集中度风险控制
    6. 多级预警系统
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 风险限制参数
        self.max_portfolio_var = config.get('max_portfolio_var', 0.02)  # 2%日VaR
        self.max_concentration = config.get('max_concentration', 0.30)  # 30%最大集中度
        self.max_correlation_exposure = config.get('max_correlation_exposure', 0.70)  # 70%相关性限制
        self.max_drawdown_threshold = config.get('max_drawdown_threshold', 0.03)  # 3%最大回撤
        
        # 预警阈值
        self.warning_var_threshold = self.max_portfolio_var * 0.8  # 80%警告线
        self.critical_var_threshold = self.max_portfolio_var * 0.95  # 95%临界线
        
        # 历史数据窗口
        self.var_window = config.get('var_window', 252)  # VaR计算窗口
        self.correlation_window = config.get('correlation_window', 30)  # 相关性窗口
        self.stress_scenarios = self._define_stress_scenarios()
        
        # 缓存
        self._risk_cache = {}
        self._last_update = None
        
    def calculate_real_time_risk(self, 
                                portfolio_weights: Dict[str, float],
                                market_data: pd.DataFrame,
                                returns_data: pd.DataFrame) -> RiskMetrics:
        """
        计算实时风险指标
        
        参数:
        - portfolio_weights: 组合权重 {symbol: weight}
        - market_data: 最新市场数据
        - returns_data: 历史收益率数据
        
        返回:
        - RiskMetrics对象
        """
        
        if not portfolio_weights:
            return self._empty_risk_metrics()
        
        # 数据准备
        symbols = list(portfolio_weights.keys())
        weights = np.array([portfolio_weights[symbol] for symbol in symbols])
        
        # 获取收益率子集
        returns_subset = returns_data[symbols].dropna()
        
        if len(returns_subset) < 30:
            raise ValueError("历史数据不足，无法计算风险指标")
        
        # 1. VaR和ES计算
        var_95, var_99, es_95 = self._calculate_var_es(weights, returns_subset)
        
        # 2. 组合波动率
        portfolio_vol = self._calculate_portfolio_volatility(weights, returns_subset)
        
        # 3. Beta计算
        portfolio_beta = self._calculate_portfolio_beta(weights, returns_subset, symbols)
        
        # 4. 最大回撤
        portfolio_returns = self._calculate_portfolio_returns(weights, returns_subset)
        max_dd = self._calculate_max_drawdown(portfolio_returns)
        
        # 5. 夏普比率和卡尔玛比率
        sharpe = self._calculate_sharpe_ratio(portfolio_returns)
        calmar = self._calculate_calmar_ratio(portfolio_returns, max_dd)
        
        # 6. 相关性风险
        correlation_risk = self._calculate_correlation_risk(weights, returns_subset)
        
        # 7. 集中度风险
        concentration_risk = self._calculate_concentration_risk(weights)
        
        return RiskMetrics(
            portfolio_var_95=var_95,
            portfolio_var_99=var_99,
            portfolio_es_95=es_95,
            portfolio_volatility=portfolio_vol,
            beta=portfolio_beta,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            correlation_risk=correlation_risk,
            concentration_risk=concentration_risk
        )
    
    def perform_stress_test(self, 
                           portfolio_weights: Dict[str, float],
                           returns_data: pd.DataFrame) -> List[StressTestResult]:
        """
        执行压力测试
        
        参数:
        - portfolio_weights: 当前组合权重
        - returns_data: 历史收益率数据
        
        返回:
        - 压力测试结果列表
        """
        
        results = []
        symbols = list(portfolio_weights.keys())
        weights = np.array([portfolio_weights[symbol] for symbol in symbols])
        returns_subset = returns_data[symbols].dropna()
        
        for scenario in self.stress_scenarios:
            result = self._run_stress_scenario(scenario, weights, returns_subset, symbols)
            results.append(result)
        
        return results
    
    def monitor_risk_limits(self, risk_metrics: RiskMetrics) -> Dict[str, Any]:
        """
        监控风险限制
        
        参数:
        - risk_metrics: 当前风险指标
        
        返回:
        - 风险监控结果
        """
        
        alerts = []
        risk_level = "LOW"
        
        # VaR检查
        if risk_metrics.portfolio_var_95 > self.critical_var_threshold:
            alerts.append({
                "type": "CRITICAL",
                "metric": "VaR_95",
                "value": risk_metrics.portfolio_var_95,
                "threshold": self.critical_var_threshold,
                "message": "组合VaR超过临界阈值，需要立即减仓"
            })
            risk_level = "CRITICAL"
        elif risk_metrics.portfolio_var_95 > self.warning_var_threshold:
            alerts.append({
                "type": "WARNING", 
                "metric": "VaR_95",
                "value": risk_metrics.portfolio_var_95,
                "threshold": self.warning_var_threshold,
                "message": "组合VaR接近警告线，建议调整仓位"
            })
            risk_level = max(risk_level, "MEDIUM")
        
        # 回撤检查
        if risk_metrics.max_drawdown > self.max_drawdown_threshold:
            alerts.append({
                "type": "CRITICAL",
                "metric": "max_drawdown", 
                "value": risk_metrics.max_drawdown,
                "threshold": self.max_drawdown_threshold,
                "message": "最大回撤超限，触发风险控制"
            })
            risk_level = "CRITICAL"
        
        # 集中度检查
        if risk_metrics.concentration_risk > self.max_concentration:
            alerts.append({
                "type": "WARNING",
                "metric": "concentration_risk",
                "value": risk_metrics.concentration_risk, 
                "threshold": self.max_concentration,
                "message": "仓位过度集中，建议分散化"
            })
            risk_level = max(risk_level, "MEDIUM")
        
        # 相关性检查
        if risk_metrics.correlation_risk > self.max_correlation_exposure:
            alerts.append({
                "type": "WARNING",
                "metric": "correlation_risk",
                "value": risk_metrics.correlation_risk,
                "threshold": self.max_correlation_exposure,
                "message": "相关性风险过高，建议调整组合"
            })
            risk_level = max(risk_level, "MEDIUM")
        
        return {
            "risk_level": risk_level,
            "alerts": alerts,
            "total_alerts": len(alerts),
            "critical_alerts": len([a for a in alerts if a["type"] == "CRITICAL"]),
            "warning_alerts": len([a for a in alerts if a["type"] == "WARNING"]),
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_risk_report(self, 
                           portfolio_weights: Dict[str, float],
                           market_data: pd.DataFrame,
                           returns_data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成完整风险报告
        
        参数:
        - portfolio_weights: 组合权重
        - market_data: 市场数据
        - returns_data: 收益率数据
        
        返回:
        - 完整风险报告
        """
        
        # 1. 基础风险指标
        risk_metrics = self.calculate_real_time_risk(portfolio_weights, market_data, returns_data)
        
        # 2. 风险限制监控
        risk_monitoring = self.monitor_risk_limits(risk_metrics)
        
        # 3. 压力测试
        stress_results = self.perform_stress_test(portfolio_weights, returns_data)
        
        # 4. 相关性矩阵
        symbols = list(portfolio_weights.keys())
        correlation_matrix = self._calculate_correlation_matrix(returns_data[symbols])
        
        # 5. VaR分解
        var_attribution = self._calculate_var_attribution(portfolio_weights, returns_data)
        
        # 6. 流动性分析
        liquidity_analysis = self._analyze_liquidity_risk(portfolio_weights, market_data)
        
        # 7. 历史VaR回测
        var_backtest = self._backtest_var_model(portfolio_weights, returns_data)
        
        risk_report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_summary": {
                "total_positions": len([w for w in portfolio_weights.values() if w > 1e-6]),
                "gross_exposure": sum(abs(w) for w in portfolio_weights.values()),
                "net_exposure": sum(portfolio_weights.values()),
                "largest_position": max(portfolio_weights.values()) if portfolio_weights else 0
            },
            "risk_metrics": {
                "VaR_95_daily": risk_metrics.portfolio_var_95,
                "VaR_99_daily": risk_metrics.portfolio_var_99,
                "ES_95_daily": risk_metrics.portfolio_es_95,
                "annualized_volatility": risk_metrics.portfolio_volatility,
                "portfolio_beta": risk_metrics.beta,
                "max_drawdown": risk_metrics.max_drawdown,
                "sharpe_ratio": risk_metrics.sharpe_ratio,
                "calmar_ratio": risk_metrics.calmar_ratio
            },
            "risk_monitoring": risk_monitoring,
            "stress_test_results": [
                {
                    "scenario": result.scenario_name,
                    "portfolio_pnl": result.portfolio_change,
                    "worst_position": result.worst_position_change,
                    "correlation_breakdown": result.correlation_breakdown,
                    "liquidity_impact": result.liquidity_impact
                }
                for result in stress_results
            ],
            "correlation_analysis": {
                "correlation_matrix": correlation_matrix.to_dict(),
                "average_correlation": correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
                "max_correlation": correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max(),
                "correlation_risk_score": risk_metrics.correlation_risk
            },
            "var_attribution": var_attribution,
            "liquidity_analysis": liquidity_analysis,
            "var_backtest": var_backtest,
            "recommendations": self._generate_risk_recommendations(risk_metrics, risk_monitoring)
        }
        
        return risk_report
    
    def _calculate_var_es(self, weights: np.ndarray, returns: pd.DataFrame) -> Tuple[float, float, float]:
        """计算VaR和期望损失(ES)"""
        
        # 组合收益率
        portfolio_returns = (returns.values @ weights).flatten()
        
        # VaR计算 (历史模拟法)
        var_95 = np.percentile(portfolio_returns, 5)  # 5%分位数
        var_99 = np.percentile(portfolio_returns, 1)  # 1%分位数
        
        # 期望损失 (ES) - 超过VaR的平均损失
        es_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        return abs(var_95), abs(var_99), abs(es_95)
    
    def _calculate_portfolio_volatility(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """计算组合波动率"""
        
        cov_matrix = returns.cov().values
        portfolio_variance = weights.T @ cov_matrix @ weights
        # 年化波动率
        return np.sqrt(portfolio_variance * 252)
    
    def _calculate_portfolio_beta(self, weights: np.ndarray, returns: pd.DataFrame, symbols: List[str]) -> float:
        """计算组合Beta"""
        
        # 假设BTC为市场基准
        if 'BTCUSDT' in symbols:
            market_returns = returns['BTCUSDT']
            portfolio_returns = (returns.values @ weights).flatten()
            
            # Beta = Cov(portfolio, market) / Var(market)
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            return covariance / market_variance if market_variance > 0 else 0.0
        else:
            # 使用等权重市场指数
            market_returns = returns.mean(axis=1)
            portfolio_returns = (returns.values @ weights).flatten()
            
            covariance = np.cov(portfolio_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            return covariance / market_variance if market_variance > 0 else 0.0
    
    def _calculate_portfolio_returns(self, weights: np.ndarray, returns: pd.DataFrame) -> pd.Series:
        """计算组合历史收益率"""
        
        portfolio_returns = (returns.values @ weights).flatten()
        return pd.Series(portfolio_returns, index=returns.index)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """计算最大回撤"""
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min())
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """计算夏普比率"""
        
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - 0.03/252  # 假设3%无风险利率
        
        if excess_returns.std() == 0:
            return 0.0
        
        return (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: pd.Series, max_drawdown: float) -> float:
        """计算卡尔玛比率"""
        
        if max_drawdown == 0:
            return 0.0
        
        annual_return = (1 + returns.mean()) ** 252 - 1
        return annual_return / max_drawdown
    
    def _calculate_correlation_risk(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """计算相关性风险分数"""
        
        correlation_matrix = returns.corr().values
        
        # 加权平均相关性
        weighted_correlations = []
        n = len(weights)
        
        for i in range(n):
            for j in range(i+1, n):
                corr = correlation_matrix[i, j]
                weight_product = weights[i] * weights[j]
                weighted_correlations.append(abs(corr) * weight_product)
        
        return sum(weighted_correlations) if weighted_correlations else 0.0
    
    def _calculate_concentration_risk(self, weights: np.ndarray) -> float:
        """计算集中度风险（HHI指数）"""
        
        # Herfindahl-Hirschman Index
        return sum(w**2 for w in weights)
    
    def _define_stress_scenarios(self) -> List[Dict[str, Any]]:
        """定义压力测试场景"""
        
        return [
            {
                "name": "市场暴跌5%",
                "type": "market_shock",
                "market_change": -0.05,
                "volatility_spike": 2.0
            },
            {
                "name": "市场暴跌10%", 
                "type": "market_shock",
                "market_change": -0.10,
                "volatility_spike": 3.0
            },
            {
                "name": "波动率飙升",
                "type": "volatility_shock",
                "market_change": 0.0,
                "volatility_spike": 5.0
            },
            {
                "name": "流动性枯竭",
                "type": "liquidity_shock",
                "liquidity_impact": 0.8,
                "spread_widening": 3.0
            },
            {
                "name": "相关性崩溃",
                "type": "correlation_shock",
                "correlation_increase": 0.9,
                "market_change": -0.03
            }
        ]
    
    def _run_stress_scenario(self, scenario: Dict[str, Any], 
                           weights: np.ndarray, 
                           returns: pd.DataFrame,
                           symbols: List[str]) -> StressTestResult:
        """运行单个压力测试场景"""
        
        scenario_name = scenario["name"]
        
        if scenario["type"] == "market_shock":
            # 市场冲击
            market_change = scenario["market_change"]
            portfolio_change = market_change * weights.sum()  # 简化假设
            worst_position_change = market_change * max(weights)
            correlation_breakdown = False
            liquidity_impact = 0.0
            
        elif scenario["type"] == "volatility_shock":
            # 波动率冲击
            vol_spike = scenario["volatility_spike"]
            current_vol = returns.std().mean()
            new_vol = current_vol * vol_spike
            # 估算组合影响
            portfolio_change = -new_vol * 2  # 简化估算
            worst_position_change = portfolio_change * max(weights) / weights.sum()
            correlation_breakdown = False
            liquidity_impact = 0.0
            
        elif scenario["type"] == "liquidity_shock":
            # 流动性冲击
            liquidity_impact = scenario["liquidity_impact"]
            spread_impact = scenario.get("spread_widening", 1.0)
            portfolio_change = -liquidity_impact * 0.02  # 2%流动性成本
            worst_position_change = portfolio_change * max(weights) / weights.sum()
            correlation_breakdown = False
            
        elif scenario["type"] == "correlation_shock":
            # 相关性冲击
            correlation_increase = scenario["correlation_increase"]
            market_change = scenario.get("market_change", -0.03)
            
            # 高相关性放大损失
            correlation_multiplier = 1 + correlation_increase
            portfolio_change = market_change * correlation_multiplier
            worst_position_change = portfolio_change * max(weights) / weights.sum()
            correlation_breakdown = True
            liquidity_impact = 0.0
            
        else:
            # 默认场景
            portfolio_change = 0.0
            worst_position_change = 0.0
            correlation_breakdown = False
            liquidity_impact = 0.0
        
        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_change=portfolio_change,
            worst_position_change=worst_position_change,
            correlation_breakdown=correlation_breakdown,
            liquidity_impact=liquidity_impact
        )
    
    def _calculate_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """计算相关性矩阵"""
        
        return returns.corr()
    
    def _calculate_var_attribution(self, portfolio_weights: Dict[str, float], 
                                  returns_data: pd.DataFrame) -> Dict[str, float]:
        """计算VaR归因"""
        
        symbols = list(portfolio_weights.keys())
        weights = np.array([portfolio_weights[symbol] for symbol in symbols])
        returns_subset = returns_data[symbols].dropna()
        
        # 组合VaR
        portfolio_returns = (returns_subset.values @ weights).flatten()
        portfolio_var = abs(np.percentile(portfolio_returns, 5))
        
        # 边际VaR计算
        marginal_vars = {}
        for i, symbol in enumerate(symbols):
            # 微小扰动
            perturbed_weights = weights.copy()
            perturbed_weights[i] += 0.001
            perturbed_weights = perturbed_weights / perturbed_weights.sum()  # 重新归一化
            
            perturbed_returns = (returns_subset.values @ perturbed_weights).flatten()
            perturbed_var = abs(np.percentile(perturbed_returns, 5))
            
            marginal_var = (perturbed_var - portfolio_var) / 0.001
            marginal_vars[symbol] = marginal_var
        
        return marginal_vars
    
    def _analyze_liquidity_risk(self, portfolio_weights: Dict[str, float], 
                               market_data: pd.DataFrame) -> Dict[str, Any]:
        """分析流动性风险"""
        
        # 简化的流动性分析
        liquidity_scores = {}
        
        for symbol in portfolio_weights.keys():
            # 基于交易量的流动性评分
            if symbol in market_data.columns:
                avg_volume = market_data[symbol].mean() if symbol in market_data.columns else 1000000
                liquidity_score = min(avg_volume / 10000000, 1.0)  # 标准化到0-1
            else:
                liquidity_score = 0.5  # 默认中等流动性
            
            liquidity_scores[symbol] = liquidity_score
        
        # 组合加权流动性分数
        weighted_liquidity = sum(
            portfolio_weights[symbol] * liquidity_scores[symbol] 
            for symbol in portfolio_weights.keys()
        )
        
        return {
            "individual_scores": liquidity_scores,
            "portfolio_liquidity_score": weighted_liquidity,
            "liquidity_risk_level": "LOW" if weighted_liquidity > 0.7 else "MEDIUM" if weighted_liquidity > 0.4 else "HIGH"
        }
    
    def _backtest_var_model(self, portfolio_weights: Dict[str, float],
                           returns_data: pd.DataFrame) -> Dict[str, Any]:
        """VaR模型回测"""
        
        symbols = list(portfolio_weights.keys())
        weights = np.array([portfolio_weights[symbol] for symbol in symbols])
        returns_subset = returns_data[symbols].dropna()
        
        # 滚动VaR预测
        window_size = 252  # 1年窗口
        var_forecasts = []
        actual_returns = []
        
        for i in range(window_size, len(returns_subset)):
            # 历史窗口
            hist_window = returns_subset.iloc[i-window_size:i]
            portfolio_hist_returns = (hist_window.values @ weights).flatten()
            
            # VaR预测
            var_forecast = abs(np.percentile(portfolio_hist_returns, 5))
            var_forecasts.append(var_forecast)
            
            # 实际收益
            actual_return = abs((returns_subset.iloc[i].values @ weights))
            actual_returns.append(actual_return)
        
        if len(var_forecasts) > 0:
            # VaR违约率
            violations = sum(1 for i in range(len(var_forecasts)) if actual_returns[i] > var_forecasts[i])
            violation_rate = violations / len(var_forecasts)
            
            # Kupiec检验 (预期违约率5%)
            expected_violations = len(var_forecasts) * 0.05
            kupiec_stat = 2 * np.log((violation_rate ** violations) * ((1-violation_rate) ** (len(var_forecasts)-violations)) / 
                                   (0.05 ** expected_violations) * (0.95 ** (len(var_forecasts)-expected_violations)))
            kupiec_pvalue = 1 - stats.chi2.cdf(kupiec_stat, 1)
            
            return {
                "backtest_period_days": len(var_forecasts),
                "total_violations": violations,
                "violation_rate": violation_rate,
                "expected_violation_rate": 0.05,
                "kupiec_test_statistic": kupiec_stat,
                "kupiec_p_value": kupiec_pvalue,
                "model_valid": kupiec_pvalue > 0.05
            }
        else:
            return {
                "backtest_period_days": 0,
                "total_violations": 0,
                "violation_rate": 0.0,
                "model_valid": False
            }
    
    def _generate_risk_recommendations(self, risk_metrics: RiskMetrics, 
                                     risk_monitoring: Dict[str, Any]) -> List[str]:
        """生成风险管理建议"""
        
        recommendations = []
        
        if risk_monitoring["risk_level"] == "CRITICAL":
            recommendations.append("🚨 立即减仓：风险指标超过临界阈值，建议减少50%仓位")
        
        if risk_metrics.portfolio_var_95 > self.warning_var_threshold:
            recommendations.append("⚠️ VaR预警：考虑降低杠杆或增加对冲仓位")
        
        if risk_metrics.concentration_risk > 0.25:
            recommendations.append("📊 分散化：仓位过度集中，建议增加币种多样化")
        
        if risk_metrics.correlation_risk > 0.6:
            recommendations.append("🔗 相关性风险：持仓币种相关性过高，考虑调整组合")
        
        if risk_metrics.max_drawdown > 0.02:
            recommendations.append("📉 回撤控制：当前回撤较大，建议加强止损纪律")
        
        if risk_metrics.sharpe_ratio < 1.0:
            recommendations.append("📈 收益优化：风险调整收益较低，建议优化交易策略")
        
        if not recommendations:
            recommendations.append("✅ 风险状况良好：当前风险水平在可接受范围内")
        
        return recommendations
    
    def _empty_risk_metrics(self) -> RiskMetrics:
        """返回空的风险指标"""
        
        return RiskMetrics(
            portfolio_var_95=0.0,
            portfolio_var_99=0.0,
            portfolio_es_95=0.0,
            portfolio_volatility=0.0,
            beta=0.0,
            max_drawdown=0.0,
            sharpe_ratio=0.0,
            calmar_ratio=0.0,
            correlation_risk=0.0,
            concentration_risk=0.0
        )


def create_risk_manager(config: Dict[str, Any]) -> RealTimeRiskManager:
    """工厂函数：创建风险管理器"""
    
    default_config = {
        'max_portfolio_var': 0.02,
        'max_concentration': 0.30,
        'max_correlation_exposure': 0.70,
        'max_drawdown_threshold': 0.03,
        'var_window': 252,
        'correlation_window': 30
    }
    
    # 合并配置
    merged_config = {**default_config, **config}
    
    return RealTimeRiskManager(merged_config)


if __name__ == "__main__":
    # 测试风险管理器
    test_config = {
        'max_portfolio_var': 0.02,
        'max_concentration': 0.30,
        'max_correlation_exposure': 0.70
    }
    
    risk_manager = create_risk_manager(test_config)
    print("DipMaster V4 风险管理器初始化成功")
    print(f"最大日VaR限制: {risk_manager.max_portfolio_var:.1%}")
    print(f"最大集中度: {risk_manager.max_concentration:.1%}")
    print(f"相关性阈值: {risk_manager.max_correlation_exposure:.1%}")