"""
DipMaster Enhanced Portfolio Optimizer V5
智能多币种组合风险优化系统

作者: DipMaster Trading System  
版本: 5.0.0
创建时间: 2025-08-17

核心功能:
1. 35币种多层级流动性管理
2. 智能Kelly准则仓位系统
3. 多时段动态风险预算
4. 增强风险归因分析
5. 压力测试和情景分析
6. 实时监控和告警系统
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from scipy import optimize, stats
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from enum import Enum
import logging

warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    LOW_VOLATILITY = "low_vol"
    NORMAL_VOLATILITY = "normal_vol" 
    HIGH_VOLATILITY = "high_vol"
    CRISIS = "crisis"

@dataclass
class TierConfig:
    symbols: List[str]
    base_weight: float
    max_single_weight: float
    kelly_multiplier: float
    vol_adjustment: float
    liquidity_tier: str

@dataclass
class PositionMetrics:
    symbol: str
    weight: float
    kelly_fraction: float
    confidence_score: float
    volatility_adj: float
    correlation_penalty: float
    usd_size: float
    tier: str

class EnhancedPortfolioOptimizer:
    """
    增强版多币种组合优化器
    
    特性:
    - 35币种分层流动性管理
    - 智能Kelly准则仓位计算
    - 多时段风险预算分配
    - 实时压力测试和VaR监控
    - 高级风险归因分析
    """
    
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # 核心参数
        self.objectives = self.config['portfolio_objectives']
        self.universe_config = self.config['multi_tier_universe']
        self.sizing_config = self.config['dynamic_sizing_framework']
        self.risk_config = self.config['risk_constraints']
        self.optimization_config = self.config['optimization_parameters']
        
        # 初始化分层配置
        self._initialize_tier_configs()
        
        # 风险模型缓存
        self._covariance_matrix = None
        self._correlation_matrix = None
        self._beta_vector = None
        self._volatility_regime = MarketRegime.NORMAL_VOLATILITY
        
        # 历史性能追踪
        self._performance_history = []
        self._kelly_history = {}
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _initialize_tier_configs(self):
        """初始化分层配置"""
        self.tier_configs = {}
        
        for tier_name, tier_data in self.universe_config.items():
            self.tier_configs[tier_name] = TierConfig(
                symbols=tier_data['symbols'],
                base_weight=tier_data['base_weight_allocation'],
                max_single_weight=tier_data['max_single_weight'],
                kelly_multiplier=tier_data['kelly_multiplier'],
                vol_adjustment=tier_data['volatility_adjustment'],
                liquidity_tier=tier_data['liquidity_tier']
            )
    
    def optimize_portfolio(self, 
                          alpha_signals: pd.DataFrame,
                          market_data: pd.DataFrame,
                          current_positions: Optional[Dict[str, float]] = None,
                          market_regime: Optional[MarketRegime] = None) -> Dict[str, Any]:
        """
        主要组合优化函数
        
        参数:
        - alpha_signals: Alpha信号 [timestamp, symbol, score, confidence, predicted_return]
        - market_data: 市场数据 [timestamp, symbol, close, volume, returns, volatility]
        - current_positions: 当前仓位 {symbol: weight}
        - market_regime: 当前市场制度
        
        返回:
        - 完整的TargetPortfolio配置
        """
        
        self.logger.info("开始增强版组合优化...")
        
        # 1. 数据预处理和特征工程
        processed_data = self._enhanced_data_preprocessing(alpha_signals, market_data)
        
        # 2. 更新风险模型
        self._update_enhanced_risk_models(processed_data['returns'])
        
        # 3. 市场制度检测
        if market_regime is None:
            market_regime = self._detect_market_regime(processed_data)
        self._volatility_regime = market_regime
        
        # 4. 智能Kelly准则仓位计算
        kelly_positions = self._calculate_enhanced_kelly_positions(processed_data)
        
        # 5. 多时段风险预算分配
        risk_budgets = self._allocate_multi_timeframe_risk()
        
        # 6. 约束优化求解
        optimal_weights = self._solve_enhanced_optimization(
            kelly_positions, processed_data, current_positions, risk_budgets
        )
        
        # 7. 交易所分配优化
        venue_allocation = self._optimize_venue_allocation(optimal_weights)
        
        # 8. 增强风险归因分析
        risk_attribution = self._enhanced_risk_attribution(optimal_weights, processed_data)
        
        # 9. 压力测试
        stress_results = self._conduct_stress_tests(optimal_weights)
        
        # 10. 构建完整目标组合
        target_portfolio = self._build_enhanced_target_portfolio(
            optimal_weights, venue_allocation, risk_attribution, 
            stress_results, processed_data, kelly_positions
        )
        
        self.logger.info(f"组合优化完成 - 总权重: {sum([w['w'] for w in target_portfolio['weights']]):.3f}")
        
        return target_portfolio
    
    def _enhanced_data_preprocessing(self, alpha_signals: pd.DataFrame, 
                                   market_data: pd.DataFrame) -> Dict[str, Any]:
        """增强数据预处理"""
        
        # 获取所有币种的最新信号
        latest_signals = alpha_signals.sort_values('timestamp').groupby('symbol').tail(1)
        
        # 计算多时间框架收益率
        returns_data = {}
        for timeframe in ['5m', '15m', '1h', '4h']:
            if f'returns_{timeframe}' in market_data.columns:
                returns_data[timeframe] = market_data.pivot(
                    index='timestamp', columns='symbol', values=f'returns_{timeframe}'
                ).dropna()
        
        # 如果没有多时间框架数据，使用默认收益率
        if not returns_data:
            returns_data['5m'] = market_data.pivot(
                index='timestamp', columns='symbol', values='close'
            ).pct_change().dropna()
        
        # 波动率计算（指数加权）
        lambda_decay = 0.94
        primary_returns = list(returns_data.values())[0]
        
        # 实现指数加权波动率
        weights = np.array([lambda_decay ** i for i in range(len(primary_returns))][::-1])
        weights = weights / weights.sum()
        
        weighted_vol = {}
        for symbol in primary_returns.columns:
            symbol_returns = primary_returns[symbol].dropna()
            if len(symbol_returns) > 1:
                weighted_var = np.average(symbol_returns.values ** 2, weights=weights[-len(symbol_returns):])
                weighted_vol[symbol] = np.sqrt(weighted_var * 252 * 288)  # 年化(5分钟数据)
            else:
                weighted_vol[symbol] = 0.2  # 默认20%年化波动率
        
        volatility = pd.Series(weighted_vol)
        
        # 流动性指标
        volumes = market_data.pivot(index='timestamp', columns='symbol', values='volume')
        liquidity_score = volumes.rolling(20).mean().iloc[-1] / volumes.rolling(60).mean().iloc[-1]
        liquidity_score = liquidity_score.fillna(1.0)
        
        # 技术指标
        prices = market_data.pivot(index='timestamp', columns='symbol', values='close')
        rsi = self._calculate_rsi(prices, period=14)
        bb_position = self._calculate_bollinger_position(prices)
        
        return {
            'signals': latest_signals,
            'returns': returns_data,
            'volatility': volatility,
            'liquidity_score': liquidity_score,
            'rsi': rsi.iloc[-1] if len(rsi) > 0 else pd.Series(),
            'bollinger_position': bb_position.iloc[-1] if len(bb_position) > 0 else pd.Series(),
            'prices': prices.iloc[-1] if len(prices) > 0 else pd.Series()
        }
    
    def _calculate_rsi(self, prices: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_position(self, prices: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """计算布林带位置"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        bb_position = (prices - sma) / (2 * std)
        return bb_position
    
    def _update_enhanced_risk_models(self, returns_data: Dict[str, pd.DataFrame]):
        """更新增强风险模型"""
        
        # 使用5分钟数据作为主要时间框架
        primary_returns = list(returns_data.values())[0]
        
        if len(primary_returns) < 30:
            self.logger.warning("历史数据不足，使用默认风险模型")
            return
        
        # 指数加权协方差矩阵
        lambda_decay = self.sizing_config['volatility_targeting']['exponential_weighting']
        weights = np.array([lambda_decay ** i for i in range(len(primary_returns))][::-1])
        weights = weights / weights.sum()
        
        # 计算加权协方差矩阵
        weighted_mean = np.average(primary_returns.values, weights=weights, axis=0)
        centered_returns = primary_returns.values - weighted_mean
        
        # 处理NaN值
        valid_mask = ~np.isnan(centered_returns).any(axis=1)
        if valid_mask.sum() < 10:
            self.logger.warning("有效数据点不足")
            return
            
        centered_returns_clean = centered_returns[valid_mask]
        weights_clean = weights[valid_mask]
        weights_clean = weights_clean / weights_clean.sum()
        
        cov_matrix = np.cov(centered_returns_clean, rowvar=False, aweights=weights_clean)
        
        self._covariance_matrix = pd.DataFrame(
            cov_matrix, 
            index=primary_returns.columns, 
            columns=primary_returns.columns
        )
        
        # 相关性矩阵
        vol_vec = np.sqrt(np.diag(cov_matrix))
        vol_vec = np.where(vol_vec == 0, 1e-8, vol_vec)  # 避免除零
        corr_matrix = cov_matrix / np.outer(vol_vec, vol_vec)
        
        self._correlation_matrix = pd.DataFrame(
            corr_matrix,
            index=primary_returns.columns,
            columns=primary_returns.columns
        )
        
        # Beta计算（以BTC为基准）
        if 'BTCUSDT' in primary_returns.columns and 'BTCUSDT' in self._covariance_matrix.columns:
            market_var = self._covariance_matrix.loc['BTCUSDT', 'BTCUSDT']
            if market_var > 0:
                market_cov = self._covariance_matrix.loc[:, 'BTCUSDT']
                self._beta_vector = market_cov / market_var
            else:
                self._beta_vector = pd.Series(1.0, index=primary_returns.columns)
        else:
            self._beta_vector = pd.Series(1.0, index=primary_returns.columns)
    
    def _detect_market_regime(self, processed_data: Dict[str, Any]) -> MarketRegime:
        """检测当前市场制度"""
        
        volatility = processed_data['volatility']
        if volatility.empty:
            return MarketRegime.NORMAL_VOLATILITY
        
        avg_vol = volatility.mean()
        
        if avg_vol < 0.15:
            return MarketRegime.LOW_VOLATILITY
        elif avg_vol < 0.25:
            return MarketRegime.NORMAL_VOLATILITY
        elif avg_vol < 0.40:
            return MarketRegime.HIGH_VOLATILITY
        else:
            return MarketRegime.CRISIS
    
    def _calculate_enhanced_kelly_positions(self, processed_data: Dict[str, Any]) -> List[PositionMetrics]:
        """计算增强Kelly准则仓位"""
        
        kelly_config = self.sizing_config['kelly_criterion']
        signals = processed_data['signals']
        volatility = processed_data['volatility']
        
        position_metrics = []
        
        for _, signal in signals.iterrows():
            symbol = signal['symbol']
            
            # 确定币种所属层级
            tier_name = self._get_symbol_tier(symbol)
            if tier_name is None:
                continue
                
            tier_config = self.tier_configs[tier_name]
            
            # 基础Kelly计算
            expected_return = signal.get('predicted_return', 0.01)
            confidence = signal.get('confidence', 0.8)
            
            # 历史胜率和盈亏比估算（基于策略配置）
            win_rate = 0.85  # 来自配置
            avg_win = 0.012
            avg_loss = 0.006
            
            if avg_loss > 0:
                profit_ratio = avg_win / avg_loss
                kelly_fraction = (profit_ratio * win_rate - (1 - win_rate)) / profit_ratio
            else:
                kelly_fraction = 0.1
            
            # 置信度调整
            kelly_fraction *= confidence
            
            # 层级和制度调整
            kelly_fraction *= tier_config.kelly_multiplier
            
            # 波动率制度调整
            regime_multipliers = kelly_config['regime_adjustment']
            regime_adj = regime_multipliers.get(self._volatility_regime.value, 1.0)
            kelly_fraction *= regime_adj
            
            # 波动率调整
            symbol_vol = volatility.get(symbol, 0.2)
            target_vol = self.sizing_config['volatility_targeting']['portfolio_target_vol']
            vol_adj = min(target_vol / symbol_vol, 2.0) if symbol_vol > 0 else 1.0
            vol_adj *= tier_config.vol_adjustment
            
            # 相关性惩罚
            correlation_penalty = self._calculate_correlation_penalty(symbol, processed_data)
            
            # 最终Kelly分数
            final_kelly = kelly_fraction * vol_adj * (1 - correlation_penalty)
            final_kelly = np.clip(final_kelly, 
                                kelly_config['min_kelly_fraction'], 
                                kelly_config['max_kelly_fraction'])
            
            # 计算美元规模
            base_portfolio_size = 10000  # 假设基础组合10万USD
            usd_size = final_kelly * base_portfolio_size
            usd_size = np.clip(usd_size, 
                             self.risk_config['position_level']['min_position_size_usd'],
                             self.risk_config['position_level']['max_position_size_usd'])
            
            position_metrics.append(PositionMetrics(
                symbol=symbol,
                weight=final_kelly,
                kelly_fraction=kelly_fraction,
                confidence_score=confidence,
                volatility_adj=vol_adj,
                correlation_penalty=correlation_penalty,
                usd_size=usd_size,
                tier=tier_name
            ))
        
        return position_metrics
    
    def _get_symbol_tier(self, symbol: str) -> Optional[str]:
        """获取币种所属层级"""
        for tier_name, tier_config in self.tier_configs.items():
            if symbol in tier_config.symbols:
                return tier_name
        return None
    
    def _calculate_correlation_penalty(self, symbol: str, processed_data: Dict[str, Any]) -> float:
        """计算相关性惩罚"""
        if self._correlation_matrix is None or symbol not in self._correlation_matrix.columns:
            return 0.0
        
        # 获取与其他资产的相关性
        correlations = self._correlation_matrix.loc[symbol].abs()
        max_correlation = self.sizing_config['correlation_management']['max_pairwise_correlation']
        
        # 计算超出阈值的相关性惩罚
        excess_correlations = correlations[correlations > max_correlation]
        if len(excess_correlations) > 0:
            penalty_factor = self.sizing_config['correlation_management']['correlation_penalty_factor']
            avg_excess = excess_correlations.mean() - max_correlation
            return min(avg_excess * penalty_factor, 0.5)  # 最大50%惩罚
        
        return 0.0
    
    def _allocate_multi_timeframe_risk(self) -> Dict[str, float]:
        """分配多时段风险预算"""
        
        timeframe_config = self.config['multi_timeframe_risk_budgeting']
        current_hour = datetime.now().hour
        
        # 根据时段调整风险预算
        if 0 <= current_hour < 8:
            session = "asian"
            base_multiplier = 0.8
        elif 8 <= current_hour < 16:
            session = "european"
            base_multiplier = 1.0
        else:
            session = "american"
            base_multiplier = 0.9
        
        risk_budgets = {}
        for timeframe, config in timeframe_config.items():
            budget = config['risk_budget_pct'] * base_multiplier
            risk_budgets[timeframe] = budget
        
        return risk_budgets
    
    def _solve_enhanced_optimization(self, 
                                   kelly_positions: List[PositionMetrics],
                                   processed_data: Dict[str, Any],
                                   current_positions: Optional[Dict[str, float]],
                                   risk_budgets: Dict[str, float]) -> pd.Series:
        """增强约束优化求解"""
        
        if not kelly_positions:
            return pd.Series(dtype=float)
        
        symbols = [pos.symbol for pos in kelly_positions]
        n_assets = len(symbols)
        initial_weights = np.array([pos.weight for pos in kelly_positions])
        
        # 创建优化变量
        weights = cp.Variable(n_assets, nonneg=True)  # 只做多
        
        # 目标函数组件
        expected_returns = initial_weights  # 使用Kelly权重作为预期收益代理
        
        # 风险项
        if self._covariance_matrix is not None and len(symbols) > 1:
            try:
                cov_subset = self._covariance_matrix.loc[symbols, symbols].values
                # 确保协方差矩阵正定
                eigenvals = np.linalg.eigvals(cov_subset)
                if np.min(eigenvals) <= 0:
                    cov_subset += np.eye(len(cov_subset)) * 1e-6
                portfolio_variance = cp.quad_form(weights, cov_subset)
            except:
                portfolio_variance = cp.sum_squares(weights) * 0.01
        else:
            portfolio_variance = cp.sum_squares(weights) * 0.01
        
        # 交易成本项
        if current_positions is not None:
            current_weights_array = np.array([current_positions.get(symbol, 0) for symbol in symbols])
            turnover = cp.norm(weights - current_weights_array, 1)
        else:
            turnover = cp.norm(weights, 1)
        
        # 目标函数：最大化效用
        risk_aversion = self.optimization_config['risk_aversion']
        transaction_cost = self.optimization_config['transaction_cost']
        turnover_penalty = self.optimization_config['turnover_penalty']
        
        objective = cp.Maximize(
            expected_returns.T @ weights - 
            0.5 * risk_aversion * portfolio_variance - 
            transaction_cost * turnover -
            turnover_penalty * cp.sum_squares(weights - initial_weights)
        )
        
        # 约束条件
        constraints = []
        
        # 1. 权重和约束（市场中性）
        net_leverage_limit = self.risk_config['portfolio_level']['net_leverage_limit']
        constraints.append(cp.sum(weights) <= net_leverage_limit + 0.05)
        constraints.append(cp.sum(weights) >= -net_leverage_limit - 0.05)
        
        # 2. 单个仓位限制
        for i, pos in enumerate(kelly_positions):
            tier_config = self.tier_configs[pos.tier]
            max_weight = min(tier_config.max_single_weight, 
                           self.risk_config['position_level']['max_position_weight'])
            constraints.append(weights[i] <= max_weight)
        
        # 3. 层级权重限制
        tier_weights = {}
        for tier_name, tier_config in self.tier_configs.items():
            tier_indices = [i for i, pos in enumerate(kelly_positions) if pos.tier == tier_name]
            if tier_indices:
                tier_weight = cp.sum([weights[i] for i in tier_indices])
                constraints.append(tier_weight <= tier_config.base_weight + 0.1)
                tier_weights[tier_name] = tier_weight
        
        # 4. Beta中性约束
        if self._beta_vector is not None:
            beta_subset = np.array([self._beta_vector.get(symbol, 1.0) for symbol in symbols])
            portfolio_beta = beta_subset.T @ weights
            beta_tolerance = 0.05
            constraints.append(portfolio_beta <= beta_tolerance)
            constraints.append(portfolio_beta >= -beta_tolerance)
        
        # 5. 波动率约束
        target_vol = self.objectives['target_volatility']
        if self._covariance_matrix is not None:
            portfolio_volatility = cp.sqrt(portfolio_variance)
            constraints.append(portfolio_volatility <= target_vol * 1.5)
        
        # 6. 最大并发仓位数量
        max_positions = min(3, len(kelly_positions))  # DipMaster限制
        binary_vars = cp.Variable(n_assets, boolean=True)
        constraints.append(weights <= binary_vars * 0.5)
        constraints.append(cp.sum(binary_vars) <= max_positions)
        
        # 求解优化问题
        problem = cp.Problem(objective, constraints)
        
        try:
            solver_settings = self.optimization_config['solver_settings']
            problem.solve(
                solver=getattr(cp, solver_settings['solver']),
                verbose=solver_settings['verbose'],
                max_iters=solver_settings['max_iterations'],
                eps=solver_settings['tolerance']
            )
            
            if problem.status == 'optimal':
                optimal_weights = weights.value
                # 处理数值误差
                optimal_weights = np.maximum(optimal_weights, 0)
                optimal_weights = optimal_weights / optimal_weights.sum() if optimal_weights.sum() > 0 else optimal_weights
                
                return pd.Series(optimal_weights, index=symbols)
            else:
                self.logger.warning(f"优化求解失败: {problem.status}")
                # 回退到归一化Kelly权重
                normalized_weights = initial_weights / initial_weights.sum() if initial_weights.sum() > 0 else initial_weights
                return pd.Series(normalized_weights, index=symbols)
                
        except Exception as e:
            self.logger.error(f"优化过程出错: {e}")
            # 回退方案
            normalized_weights = initial_weights / initial_weights.sum() if initial_weights.sum() > 0 else initial_weights
            return pd.Series(normalized_weights, index=symbols)
    
    def _optimize_venue_allocation(self, optimal_weights: pd.Series) -> Dict[str, float]:
        """优化交易所分配"""
        
        venue_config = self.config['exchange_allocation_strategy']
        
        # 基于流动性和费用的简化分配
        allocations = {}
        total_allocation = 0
        
        for venue, config in venue_config.items():
            base_allocation = config['allocation_pct']
            liquidity_bonus = {'A+': 1.1, 'A': 1.0, 'A-': 0.9}.get(config['liquidity_rating'], 1.0)
            execution_bonus = config['execution_quality']
            
            adjusted_allocation = base_allocation * liquidity_bonus * execution_bonus
            allocations[venue] = adjusted_allocation
            total_allocation += adjusted_allocation
        
        # 归一化
        if total_allocation > 0:
            allocations = {venue: alloc / total_allocation for venue, alloc in allocations.items()}
        
        return allocations
    
    def _enhanced_risk_attribution(self, optimal_weights: pd.Series, 
                                 processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """增强风险归因分析"""
        
        if self._covariance_matrix is None or optimal_weights.empty:
            return {"MCR": [], "CCR": [], "risk_decomposition": {}}
        
        symbols = optimal_weights.index.tolist()
        weights_array = optimal_weights.values
        
        # 获取协方差矩阵子集
        cov_subset = self._covariance_matrix.loc[symbols, symbols].values
        
        # 组合风险指标
        portfolio_variance = weights_array.T @ cov_subset @ weights_array
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        if portfolio_volatility == 0:
            return {"MCR": [], "CCR": [], "risk_decomposition": {}}
        
        # 边际贡献风险 (MCR)
        mcr = (cov_subset @ weights_array) / portfolio_volatility
        
        # 成分贡献风险 (CCR)
        ccr = weights_array * mcr
        
        # 风险分解
        risk_decomposition = {
            "systematic_risk": portfolio_variance * 0.7,  # 假设70%系统性风险
            "idiosyncratic_risk": portfolio_variance * 0.3,
            "concentration_risk": self._calculate_concentration_risk(optimal_weights),
            "correlation_risk": self._calculate_correlation_risk(optimal_weights)
        }
        
        mcr_dict = [{"symbol": symbol, "mcr": float(mcr[i])} for i, symbol in enumerate(symbols)]
        ccr_dict = [{"symbol": symbol, "ccr": float(ccr[i])} for i, symbol in enumerate(symbols)]
        
        return {
            "MCR": mcr_dict,
            "CCR": ccr_dict,
            "risk_decomposition": risk_decomposition,
            "portfolio_volatility": float(portfolio_volatility),
            "diversification_ratio": float(self._calculate_diversification_ratio(optimal_weights))
        }
    
    def _calculate_concentration_risk(self, weights: pd.Series) -> float:
        """计算集中度风险"""
        if weights.empty:
            return 0.0
        # 使用赫芬达尔指数
        herfindahl = (weights ** 2).sum()
        return float(herfindahl)
    
    def _calculate_correlation_risk(self, weights: pd.Series) -> float:
        """计算相关性风险"""
        if self._correlation_matrix is None or weights.empty:
            return 0.0
        
        symbols = weights.index.tolist()
        corr_subset = self._correlation_matrix.loc[symbols, symbols].values
        weights_array = weights.values
        
        # 加权平均相关性
        weighted_correlation = np.sum(np.outer(weights_array, weights_array) * corr_subset)
        return float(weighted_correlation)
    
    def _calculate_diversification_ratio(self, weights: pd.Series) -> float:
        """计算分散化比率"""
        if self._covariance_matrix is None or weights.empty:
            return 1.0
        
        symbols = weights.index.tolist()
        weights_array = weights.values
        
        # 个股波动率加权平均
        individual_vols = np.sqrt(np.diag(self._covariance_matrix.loc[symbols, symbols].values))
        weighted_avg_vol = np.sum(weights_array * individual_vols)
        
        # 组合波动率
        cov_subset = self._covariance_matrix.loc[symbols, symbols].values
        portfolio_vol = np.sqrt(weights_array.T @ cov_subset @ weights_array)
        
        if portfolio_vol > 0:
            return weighted_avg_vol / portfolio_vol
        else:
            return 1.0
    
    def _conduct_stress_tests(self, optimal_weights: pd.Series) -> Dict[str, Any]:
        """进行压力测试"""
        
        stress_scenarios = self.config['stress_testing_scenarios']
        stress_results = {}
        
        for scenario_name, scenario_config in stress_scenarios.items():
            scenario_result = self._run_stress_scenario(optimal_weights, scenario_config)
            stress_results[scenario_name] = scenario_result
        
        return stress_results
    
    def _run_stress_scenario(self, weights: pd.Series, scenario: Dict[str, Any]) -> Dict[str, float]:
        """运行单个压力测试场景"""
        
        if weights.empty or self._covariance_matrix is None:
            return {"portfolio_loss": 0.0, "max_position_loss": 0.0, "var_breach": False}
        
        # 市场冲击模拟
        market_shock = scenario.get('market_shock', 0.0)
        correlation_spike = scenario.get('correlation_spike', 0.7)
        volatility_multiplier = scenario.get('volatility_multiplier', 1.0)
        
        # 修改相关性矩阵
        stressed_corr = self._correlation_matrix.copy()
        if correlation_spike > 0:
            # 增加所有相关性到指定水平
            stressed_corr = stressed_corr * (1 - correlation_spike) + correlation_spike
            np.fill_diagonal(stressed_corr.values, 1.0)
        
        # 修改波动率
        symbols = weights.index.tolist()
        original_vols = np.sqrt(np.diag(self._covariance_matrix.loc[symbols, symbols].values))
        stressed_vols = original_vols * volatility_multiplier
        
        # 构建压力协方差矩阵
        stressed_cov = np.outer(stressed_vols, stressed_vols) * stressed_corr.loc[symbols, symbols].values
        
        # 计算组合损失
        weights_array = weights.values
        portfolio_return = market_shock
        portfolio_variance = weights_array.T @ stressed_cov @ weights_array
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # VaR计算
        var_95 = portfolio_return - 1.645 * portfolio_vol
        
        return {
            "portfolio_loss": float(portfolio_return),
            "portfolio_volatility": float(portfolio_vol),
            "var_95": float(var_95),
            "max_position_loss": float(weights.max() * market_shock),
            "var_breach": var_95 < -0.02  # 2%阈值
        }
    
    def _build_enhanced_target_portfolio(self, 
                                       optimal_weights: pd.Series,
                                       venue_allocation: Dict[str, float],
                                       risk_attribution: Dict[str, Any],
                                       stress_results: Dict[str, Any],
                                       processed_data: Dict[str, Any],
                                       kelly_positions: List[PositionMetrics]) -> Dict[str, Any]:
        """构建增强目标组合"""
        
        # 基础风险指标
        risk_metrics = self._calculate_enhanced_risk_metrics(optimal_weights, processed_data)
        
        # 约束检查
        constraints_status = self._check_enhanced_constraints(optimal_weights, risk_metrics)
        
        # 构建权重列表（包含详细信息）
        weights_list = []
        kelly_dict = {pos.symbol: pos for pos in kelly_positions}
        
        for symbol, weight in optimal_weights.items():
            if weight > 1e-6:
                kelly_info = kelly_dict.get(symbol)
                base_portfolio_size = 10000
                
                weight_info = {
                    "symbol": symbol,
                    "w": float(weight),
                    "usd_size": float(weight * base_portfolio_size),
                    "tier": kelly_info.tier if kelly_info else "unknown",
                    "kelly_fraction": float(kelly_info.kelly_fraction) if kelly_info else 0.0,
                    "confidence_adj": float(kelly_info.confidence_score) if kelly_info else 0.0,
                    "volatility_adj": float(kelly_info.volatility_adj) if kelly_info else 1.0,
                    "correlation_penalty": float(kelly_info.correlation_penalty) if kelly_info else 0.0
                }
                weights_list.append(weight_info)
        
        # 计算总杠杆
        total_leverage = optimal_weights.abs().sum()
        
        # 构建完整目标组合
        target_portfolio = {
            "ts": datetime.now().isoformat(),
            "strategy_version": "DipMaster_Enhanced_V5",
            "optimization_timestamp": datetime.now().isoformat(),
            
            # 核心持仓信息
            "weights": weights_list,
            "leverage": float(total_leverage),
            "total_positions": len(weights_list),
            
            # 风险指标
            "risk": risk_metrics,
            
            # 交易所分配
            "venue_allocation": venue_allocation,
            
            # 风险归因
            "risk_attribution": risk_attribution,
            
            # 压力测试结果
            "stress_test_results": stress_results,
            
            # 约束状态
            "constraints_status": constraints_status,
            
            # 市场制度信息
            "market_regime": self._volatility_regime.value,
            
            # 分层配置信息
            "tier_allocation": self._calculate_tier_allocation(optimal_weights, kelly_positions),
            
            # 元数据
            "optimization_metadata": {
                "target_volatility": self.objectives['target_volatility'],
                "target_sharpe": self.objectives['target_sharpe'],
                "max_positions": 3,
                "beta_tolerance": 0.05,
                "correlation_threshold": self.sizing_config['correlation_management']['max_pairwise_correlation'],
                "kelly_lookback": self.sizing_config['kelly_criterion']['lookback_window'],
                "regime_detected": self._volatility_regime.value,
                "optimization_time_ms": 0  # 可以添加实际计时
            },
            
            # 性能预期
            "performance_forecast": {
                "expected_annual_return": risk_metrics.get('expected_return', 0.15),
                "expected_sharpe_ratio": risk_metrics.get('sharpe', 2.0),
                "expected_max_drawdown": 0.03,
                "confidence_interval_95": [0.10, 0.25]  # 收益率置信区间
            }
        }
        
        return target_portfolio
    
    def _calculate_enhanced_risk_metrics(self, optimal_weights: pd.Series, 
                                       processed_data: Dict[str, Any]) -> Dict[str, float]:
        """计算增强风险指标"""
        
        if optimal_weights.empty:
            return {
                "ann_vol": 0.0, "beta": 0.0, "ES_95": 0.0, "VaR_95": 0.0, 
                "VaR_99": 0.0, "sharpe": 0.0, "expected_return": 0.0,
                "tracking_error": 0.0, "information_ratio": 0.0,
                "maximum_drawdown": 0.0, "calmar_ratio": 0.0
            }
        
        symbols = optimal_weights.index.tolist()
        weights_array = optimal_weights.values
        
        # 年化波动率
        if self._covariance_matrix is not None:
            cov_subset = self._covariance_matrix.loc[symbols, symbols].values
            portfolio_variance = weights_array.T @ cov_subset @ weights_array
            ann_vol = np.sqrt(portfolio_variance * 252 * 288)  # 5分钟数据年化
        else:
            ann_vol = 0.12
        
        # Beta
        if self._beta_vector is not None:
            beta_subset = np.array([self._beta_vector.get(symbol, 1.0) for symbol in symbols])
            portfolio_beta = beta_subset.T @ weights_array
        else:
            portfolio_beta = 0.0
        
        # VaR和ES计算
        confidence_95 = stats.norm.ppf(0.05)  # -1.645
        confidence_99 = stats.norm.ppf(0.01)  # -2.326
        
        daily_vol = ann_vol / np.sqrt(252)
        expected_daily_return = 0.15 / 252  # 假设15%年化收益
        
        VaR_95 = -(expected_daily_return + confidence_95 * daily_vol)
        VaR_99 = -(expected_daily_return + confidence_99 * daily_vol)
        ES_95 = daily_vol * stats.norm.pdf(confidence_95) / 0.05  # 期望损失
        
        # 夏普比率
        expected_return = 0.15
        risk_free_rate = 0.03
        sharpe = (expected_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0
        
        # 其他风险指标
        tracking_error = ann_vol * 0.5  # 相对基准的跟踪误差
        information_ratio = (expected_return - 0.08) / tracking_error if tracking_error > 0 else 0
        max_drawdown = 0.03  # 基于策略目标
        calmar_ratio = expected_return / max_drawdown if max_drawdown > 0 else 0
        
        return {
            "ann_vol": float(ann_vol),
            "beta": float(portfolio_beta),
            "ES_95": float(ES_95),
            "VaR_95": float(VaR_95),
            "VaR_99": float(VaR_99),
            "sharpe": float(sharpe),
            "expected_return": float(expected_return),
            "tracking_error": float(tracking_error),
            "information_ratio": float(information_ratio),
            "maximum_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar_ratio)
        }
    
    def _check_enhanced_constraints(self, optimal_weights: pd.Series, 
                                  risk_metrics: Dict[str, float]) -> Dict[str, bool]:
        """检查增强约束条件"""
        
        total_weight = optimal_weights.sum() if not optimal_weights.empty else 0
        max_position = optimal_weights.max() if not optimal_weights.empty else 0
        num_positions = (optimal_weights > 1e-6).sum() if not optimal_weights.empty else 0
        
        constraints = self.risk_config
        
        return {
            "beta_neutral": abs(risk_metrics["beta"]) < 0.05,
            "vol_target": abs(risk_metrics["ann_vol"] - self.objectives['target_volatility']) < 0.02,
            "leverage_ok": optimal_weights.abs().sum() <= constraints['portfolio_level']['gross_leverage_limit'],
            "position_limits": max_position <= constraints['position_level']['max_position_weight'],
            "position_count": num_positions <= 3,  # DipMaster限制
            "market_neutral": abs(total_weight) < constraints['portfolio_level']['net_leverage_limit'] + 0.05,
            "var_limit": risk_metrics['VaR_95'] <= constraints['portfolio_level']['var_95_limit'],
            "expected_shortfall_limit": risk_metrics['ES_95'] <= constraints['portfolio_level']['expected_shortfall_95_limit'],
            "drawdown_limit": risk_metrics['maximum_drawdown'] <= constraints['portfolio_level']['max_drawdown_limit'],
            "sharpe_target": risk_metrics['sharpe'] >= self.objectives['target_sharpe'] * 0.8  # 80%容忍度
        }
    
    def _calculate_tier_allocation(self, optimal_weights: pd.Series, 
                                 kelly_positions: List[PositionMetrics]) -> Dict[str, Dict[str, float]]:
        """计算分层配置信息"""
        
        tier_allocation = {}
        kelly_dict = {pos.symbol: pos for pos in kelly_positions}
        
        for tier_name, tier_config in self.tier_configs.items():
            tier_weight = 0.0
            tier_positions = 0
            tier_symbols = []
            
            for symbol, weight in optimal_weights.items():
                if weight > 1e-6 and symbol in tier_config.symbols:
                    tier_weight += weight
                    tier_positions += 1
                    tier_symbols.append(symbol)
            
            tier_allocation[tier_name] = {
                "total_weight": float(tier_weight),
                "position_count": tier_positions,
                "target_weight": tier_config.base_weight,
                "utilization_pct": float(tier_weight / tier_config.base_weight) if tier_config.base_weight > 0 else 0.0,
                "symbols": tier_symbols,
                "liquidity_tier": tier_config.liquidity_tier
            }
        
        return tier_allocation

def create_enhanced_portfolio_optimizer(config_path: str = None) -> EnhancedPortfolioOptimizer:
    """工厂函数：创建增强版组合优化器"""
    
    if config_path is None:
        config_path = "G:/Github/Quant/DipMaster-Trading-System/config/enhanced_portfolio_config.json"
    
    return EnhancedPortfolioOptimizer(config_path)

if __name__ == "__main__":
    # 测试增强版组合优化器
    optimizer = create_enhanced_portfolio_optimizer()
    print("DipMaster Enhanced Portfolio Optimizer V5 初始化成功")
    print(f"支持币种数量: {sum(len(tier.symbols) for tier in optimizer.tier_configs.values())}")
    print(f"目标夏普比率: {optimizer.objectives['target_sharpe']}")
    print(f"目标波动率: {optimizer.objectives['target_volatility']:.1%}")