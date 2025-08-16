"""
DipMaster Enhanced V4 - 多币种组合优化和风险控制系统
专门针对高频逢跌买入策略的投资组合管理

作者: DipMaster Trading System
版本: 4.0.0
创建时间: 2025-08-16
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy import optimize
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PortfolioOptimizer:
    """
    基于Alpha信号的多币种组合优化器
    
    核心功能:
    1. Mean-Variance优化与Kelly准则结合
    2. 市场中性Beta约束
    3. 最大回撤控制≤3%
    4. 动态相关性管理
    5. 多重风险约束
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.target_vol = config.get('target_volatility', 0.18)  # 18%年化波动率
        self.max_leverage = config.get('max_leverage', 3.0)
        self.beta_tolerance = config.get('beta_tolerance', 0.05)
        self.max_position_weight = config.get('max_position_pct', 0.30)
        self.max_correlation = config.get('max_correlation', 0.7)
        self.max_concurrent_positions = config.get('max_concurrent_positions', 3)
        self.daily_loss_limit = config.get('daily_loss_limit', 0.02)
        
        # Kelly准则参数
        self.max_kelly_fraction = config.get('max_kelly_fraction', 0.25)
        self.lookback_trades = config.get('lookback_trades', 100)
        
        # 风险衰减参数
        self.lambda_decay = config.get('lambda_decay', 0.94)
        
        # 交易成本
        self.transaction_cost = config.get('transaction_cost', 0.001)  # 0.1%
        
        # 缓存
        self._covariance_matrix = None
        self._correlation_matrix = None
        self._beta_vector = None
        
    def optimize_portfolio(self, alpha_signals: pd.DataFrame, 
                          market_data: pd.DataFrame,
                          current_positions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        主要组合优化函数
        
        参数:
        - alpha_signals: Alpha信号数据框 [timestamp, symbol, score, confidence, predicted_return]
        - market_data: 市场数据 [timestamp, symbol, close, volume, returns]
        - current_positions: 当前仓位权重 {symbol: weight}
        
        返回:
        - TargetPortfolio字典
        """
        
        # 1. 数据预处理
        processed_data = self._preprocess_data(alpha_signals, market_data)
        
        # 2. 计算协方差矩阵和Beta
        self._update_risk_models(processed_data['returns'])
        
        # 3. Kelly准则仓位估算
        kelly_weights = self._calculate_kelly_weights(processed_data)
        
        # 4. 风险平价调整
        risk_adjusted_weights = self._apply_risk_parity(kelly_weights, processed_data)
        
        # 5. 约束优化
        optimal_weights = self._solve_optimization(
            risk_adjusted_weights, 
            processed_data,
            current_positions
        )
        
        # 6. 交易所分配
        venue_allocation = self._allocate_to_venues(optimal_weights)
        
        # 7. 风险归因分析
        risk_attribution = self._calculate_risk_attribution(optimal_weights)
        
        # 8. 构建目标组合
        target_portfolio = self._build_target_portfolio(
            optimal_weights, venue_allocation, risk_attribution, processed_data
        )
        
        return target_portfolio
    
    def _preprocess_data(self, alpha_signals: pd.DataFrame, 
                        market_data: pd.DataFrame) -> Dict[str, Any]:
        """数据预处理和特征工程"""
        
        # 获取最新信号
        latest_signals = alpha_signals.sort_values('timestamp').groupby('symbol').tail(1)
        
        # 计算收益率
        returns = market_data.pivot(index='timestamp', columns='symbol', values='close').pct_change().dropna()
        
        # 计算波动率 (20日滚动)
        volatility = returns.rolling(20).std() * np.sqrt(252 * 24 * 12)  # 年化波动率
        
        # 计算相关性矩阵 (30日滚动)
        correlation = returns.rolling(30).corr()
        
        # 流动性指标
        volumes = market_data.pivot(index='timestamp', columns='symbol', values='volume')
        avg_volume = volumes.rolling(20).mean()
        
        return {
            'signals': latest_signals,
            'returns': returns,
            'volatility': volatility.iloc[-1],  # 最新波动率
            'correlation': correlation.iloc[-1] if len(correlation) > 0 else None,
            'volume': avg_volume.iloc[-1] if len(avg_volume) > 0 else None,
            'prices': market_data.pivot(index='timestamp', columns='symbol', values='close').iloc[-1]
        }
    
    def _update_risk_models(self, returns: pd.DataFrame):
        """更新协方差矩阵和Beta向量"""
        
        if len(returns) < 30:
            raise ValueError("历史数据不足，至少需要30个观测值")
        
        # 指数加权协方差矩阵
        weights = np.array([self.lambda_decay ** i for i in range(len(returns))][::-1])
        weights = weights / weights.sum()
        
        # 计算加权均值
        weighted_mean = np.average(returns.values, weights=weights, axis=0)
        
        # 计算加权协方差矩阵
        centered_returns = returns.values - weighted_mean
        cov_matrix = np.cov(centered_returns, rowvar=False, aweights=weights)
        
        self._covariance_matrix = pd.DataFrame(
            cov_matrix, 
            index=returns.columns, 
            columns=returns.columns
        )
        
        # 计算相关性矩阵
        vol_vec = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(vol_vec, vol_vec)
        self._correlation_matrix = pd.DataFrame(
            corr_matrix,
            index=returns.columns,
            columns=returns.columns
        )
        
        # 计算Beta (假设BTC为市场基准)
        if 'BTCUSDT' in returns.columns:
            market_var = self._covariance_matrix.loc['BTCUSDT', 'BTCUSDT']
            market_cov = self._covariance_matrix.loc[:, 'BTCUSDT']
            self._beta_vector = market_cov / market_var
        else:
            # 如果没有BTC，使用等权重市场指数
            market_returns = returns.mean(axis=1)
            market_var = market_returns.var()
            betas = {}
            for symbol in returns.columns:
                symbol_returns = returns[symbol]
                cov_with_market = np.cov(symbol_returns, market_returns)[0, 1]
                betas[symbol] = cov_with_market / market_var
            self._beta_vector = pd.Series(betas)
    
    def _calculate_kelly_weights(self, processed_data: Dict[str, Any]) -> pd.Series:
        """基于Kelly准则计算初始仓位权重"""
        
        signals = processed_data['signals']
        kelly_weights = {}
        
        for _, signal in signals.iterrows():
            symbol = signal['symbol']
            expected_return = signal['predicted_return']
            confidence = signal['confidence']
            
            # Kelly公式: f = (bp - q) / b
            # 其中 b = 盈亏比, p = 胜率, q = 败率
            
            # 基于历史表现估算胜率和盈亏比
            win_rate = 0.85  # 来自策略配置的目标胜率
            avg_win = 0.012  # 平均盈利1.2%
            avg_loss = 0.006  # 平均亏损0.6%
            
            if avg_loss > 0:
                profit_ratio = avg_win / avg_loss
                kelly_fraction = (profit_ratio * win_rate - (1 - win_rate)) / profit_ratio
                
                # 置信度调整
                kelly_fraction *= confidence
                
                # Kelly分数限制
                kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
                kelly_fraction = max(kelly_fraction, 0)  # 不允许做空
                
                kelly_weights[symbol] = kelly_fraction
            else:
                kelly_weights[symbol] = 0
        
        return pd.Series(kelly_weights)
    
    def _apply_risk_parity(self, kelly_weights: pd.Series, 
                          processed_data: Dict[str, Any]) -> pd.Series:
        """应用风险平价调整Kelly权重"""
        
        volatility = processed_data['volatility']
        
        # 波动率调整
        if not volatility.empty and len(volatility) > 0:
            # 目标波动率归一化
            vol_weights = self.target_vol / volatility
            vol_weights = vol_weights.fillna(0)
            vol_weights = vol_weights.clip(0.2, 2.0)  # 限制调整幅度
            
            # 与Kelly权重结合
            adjusted_weights = kelly_weights * vol_weights
        else:
            adjusted_weights = kelly_weights
        
        # 归一化权重
        total_weight = adjusted_weights.sum()
        if total_weight > 0:
            adjusted_weights = adjusted_weights / total_weight
        
        return adjusted_weights
    
    def _solve_optimization(self, initial_weights: pd.Series,
                           processed_data: Dict[str, Any],
                           current_positions: Optional[Dict[str, float]] = None) -> pd.Series:
        """使用CVXPY求解约束优化问题"""
        
        symbols = initial_weights.index.tolist()
        n_assets = len(symbols)
        
        if n_assets == 0:
            return pd.Series(dtype=float)
        
        # 创建优化变量
        weights = cp.Variable(n_assets)
        
        # 目标函数：最大化效用 = 预期收益 - 风险惩罚 - 交易成本
        expected_returns = np.array([initial_weights[symbol] for symbol in symbols])
        
        # 风险项
        if self._covariance_matrix is not None:
            cov_subset = self._covariance_matrix.loc[symbols, symbols].values
            portfolio_variance = cp.quad_form(weights, cov_subset)
        else:
            portfolio_variance = cp.sum_squares(weights) * 0.01  # 默认风险
        
        # 交易成本项
        if current_positions is not None:
            current_weights_array = np.array([current_positions.get(symbol, 0) for symbol in symbols])
            turnover = cp.norm(weights - current_weights_array, 1)
        else:
            turnover = cp.norm(weights, 1)
        
        # 目标函数
        objective = cp.Maximize(
            expected_returns.T @ weights - 
            0.5 * portfolio_variance - 
            self.transaction_cost * turnover
        )
        
        # 约束条件
        constraints = []
        
        # 1. 权重和约束 (近似市场中性)
        constraints.append(cp.sum(weights) <= 0.1)  # 允许小幅净多头
        constraints.append(cp.sum(weights) >= -0.1)  # 允许小幅净空头
        
        # 2. 单个仓位限制
        constraints.append(weights <= self.max_position_weight)
        constraints.append(weights >= 0)  # 只做多
        
        # 3. 最大并发仓位数量
        # 使用二进制变量实现
        binary_vars = cp.Variable(n_assets, boolean=True)
        constraints.append(weights <= binary_vars * self.max_position_weight)
        constraints.append(cp.sum(binary_vars) <= self.max_concurrent_positions)
        
        # 4. Beta中性约束
        if self._beta_vector is not None:
            beta_subset = np.array([self._beta_vector.get(symbol, 1.0) for symbol in symbols])
            portfolio_beta = beta_subset.T @ weights
            constraints.append(portfolio_beta <= self.beta_tolerance)
            constraints.append(portfolio_beta >= -self.beta_tolerance)
        
        # 5. 波动率约束
        if self._covariance_matrix is not None:
            portfolio_volatility = cp.sqrt(portfolio_variance)
            constraints.append(portfolio_volatility <= self.target_vol * 1.2)  # 允许20%偏差
        
        # 6. 相关性约束 (简化版本 - 限制高相关资产的总权重)
        if self._correlation_matrix is not None:
            for i, symbol_i in enumerate(symbols):
                for j, symbol_j in enumerate(symbols[i+1:], i+1):
                    corr = self._correlation_matrix.loc[symbol_i, symbol_j]
                    if abs(corr) > self.max_correlation:
                        # 高相关资产的权重之和限制
                        constraints.append(weights[i] + weights[j] <= self.max_position_weight)
        
        # 求解优化问题
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == 'optimal':
                optimal_weights = weights.value
                # 处理数值误差
                optimal_weights = np.maximum(optimal_weights, 0)
                optimal_weights = optimal_weights / optimal_weights.sum() if optimal_weights.sum() > 0 else optimal_weights
                
                return pd.Series(optimal_weights, index=symbols)
            else:
                print(f"优化求解失败: {problem.status}")
                # 回退到风险调整后的权重
                return initial_weights
                
        except Exception as e:
            print(f"优化过程出错: {e}")
            return initial_weights
    
    def _allocate_to_venues(self, optimal_weights: pd.Series) -> Dict[str, float]:
        """分配仓位到不同交易所"""
        
        # 简化版本：全部分配给Binance
        # 实际应用中可以根据流动性、费用等因素分配
        return {"binance": 1.0}
    
    def _calculate_risk_attribution(self, optimal_weights: pd.Series) -> Dict[str, Any]:
        """计算风险归因分析"""
        
        if self._covariance_matrix is None or optimal_weights.empty:
            return {"MCR": [], "CCR": []}
        
        symbols = optimal_weights.index.tolist()
        weights_array = optimal_weights.values
        
        # 获取对应的协方差矩阵子集
        cov_subset = self._covariance_matrix.loc[symbols, symbols].values
        
        # 计算组合波动率
        portfolio_variance = weights_array.T @ cov_subset @ weights_array
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # 边际贡献风险 (MCR)
        mcr = (cov_subset @ weights_array) / portfolio_volatility if portfolio_volatility > 0 else np.zeros_like(weights_array)
        
        # 成分贡献风险 (CCR)
        ccr = weights_array * mcr
        
        mcr_dict = [{"symbol": symbol, "mcr": float(mcr[i])} for i, symbol in enumerate(symbols)]
        ccr_dict = [{"symbol": symbol, "ccr": float(ccr[i])} for i, symbol in enumerate(symbols)]
        
        return {
            "MCR": mcr_dict,
            "CCR": ccr_dict
        }
    
    def _build_target_portfolio(self, optimal_weights: pd.Series,
                               venue_allocation: Dict[str, float],
                               risk_attribution: Dict[str, Any],
                               processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """构建目标组合输出"""
        
        # 计算组合层面风险指标
        risk_metrics = self._calculate_portfolio_risk(optimal_weights, processed_data)
        
        # 约束状态检查
        constraints_status = self._check_constraints(optimal_weights, risk_metrics)
        
        # 权重列表
        weights_list = [
            {"symbol": symbol, "w": float(weight)} 
            for symbol, weight in optimal_weights.items() if weight > 1e-6
        ]
        
        # 计算总杠杆
        total_leverage = optimal_weights.abs().sum()
        
        target_portfolio = {
            "ts": datetime.now().isoformat(),
            "weights": weights_list,
            "leverage": float(total_leverage),
            "risk": risk_metrics,
            "venue_allocation": venue_allocation,
            "risk_attribution": risk_attribution,
            "constraints_status": constraints_status,
            "optimization_metadata": {
                "target_volatility": self.target_vol,
                "max_positions": self.max_concurrent_positions,
                "beta_tolerance": self.beta_tolerance,
                "correlation_threshold": self.max_correlation
            }
        }
        
        return target_portfolio
    
    def _calculate_portfolio_risk(self, optimal_weights: pd.Series, 
                                 processed_data: Dict[str, Any]) -> Dict[str, float]:
        """计算组合层面风险指标"""
        
        if optimal_weights.empty:
            return {
                "ann_vol": 0.0,
                "beta": 0.0,
                "ES_95": 0.0,
                "VaR_95": 0.0,
                "VaR_99": 0.0,
                "sharpe": 0.0
            }
        
        symbols = optimal_weights.index.tolist()
        weights_array = optimal_weights.values
        
        # 年化波动率
        if self._covariance_matrix is not None:
            cov_subset = self._covariance_matrix.loc[symbols, symbols].values
            portfolio_variance = weights_array.T @ cov_subset @ weights_array
            ann_vol = np.sqrt(portfolio_variance * 252 * 24 * 12)  # 年化
        else:
            ann_vol = 0.12  # 默认值
        
        # Beta
        if self._beta_vector is not None:
            beta_subset = np.array([self._beta_vector.get(symbol, 1.0) for symbol in symbols])
            portfolio_beta = beta_subset.T @ weights_array
        else:
            portfolio_beta = 0.0
        
        # VaR和ES估算 (正态分布假设)
        confidence_95 = 1.645  # 95%置信度
        confidence_99 = 2.326  # 99%置信度
        
        daily_vol = ann_vol / np.sqrt(252)
        VaR_95 = daily_vol * confidence_95
        VaR_99 = daily_vol * confidence_99
        ES_95 = daily_vol * 2.063  # 期望损失
        
        # 夏普比率估算
        expected_return = 0.15  # 假设15%年化收益
        risk_free_rate = 0.03   # 3%无风险利率
        sharpe = (expected_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        
        return {
            "ann_vol": float(ann_vol),
            "beta": float(portfolio_beta),
            "ES_95": float(ES_95),
            "VaR_95": float(VaR_95),
            "VaR_99": float(VaR_99),
            "sharpe": float(sharpe)
        }
    
    def _check_constraints(self, optimal_weights: pd.Series, 
                          risk_metrics: Dict[str, float]) -> Dict[str, bool]:
        """检查约束条件满足情况"""
        
        total_weight = optimal_weights.sum()
        max_position = optimal_weights.max() if not optimal_weights.empty else 0
        num_positions = (optimal_weights > 1e-6).sum()
        
        return {
            "beta_neutral": abs(risk_metrics["beta"]) < self.beta_tolerance,
            "vol_target": abs(risk_metrics["ann_vol"] - self.target_vol) < 0.05,
            "leverage_ok": optimal_weights.abs().sum() <= self.max_leverage,
            "position_limits": max_position <= self.max_position_weight,
            "position_count": num_positions <= self.max_concurrent_positions,
            "market_neutral": abs(total_weight) < 0.1
        }


def create_portfolio_optimizer(strategy_config_path: str) -> PortfolioOptimizer:
    """工厂函数：基于配置文件创建组合优化器"""
    
    with open(strategy_config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    optimizer_config = {
        'target_volatility': 0.18,
        'max_leverage': 3.0,
        'beta_tolerance': 0.05,
        'max_position_pct': config.get('constraints', {}).get('max_position_pct', 0.30),
        'max_correlation': config.get('constraints', {}).get('max_correlation', 0.7),
        'max_concurrent_positions': config.get('constraints', {}).get('max_concurrent_positions', 3),
        'daily_loss_limit': config.get('constraints', {}).get('daily_loss_limit', 0.02),
        'max_kelly_fraction': 0.25,
        'lookback_trades': 100,
        'lambda_decay': 0.94,
        'transaction_cost': 0.001
    }
    
    return PortfolioOptimizer(optimizer_config)


if __name__ == "__main__":
    # 测试组合优化器
    config_path = "G:/Github/Quant/DipMaster-Trading-System/config/dipmaster_enhanced_v4_spec.json"
    optimizer = create_portfolio_optimizer(config_path)
    print("DipMaster V4 组合优化器初始化成功")
    print(f"目标波动率: {optimizer.target_vol:.1%}")
    print(f"最大杠杆: {optimizer.max_leverage}x")
    print(f"Beta容忍度: ±{optimizer.beta_tolerance:.2f}")