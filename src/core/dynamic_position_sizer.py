"""
DipMaster Enhanced V4 - 动态仓位大小优化器
基于信号置信度、市场波动率和风险预算的智能仓位管理

作者: DipMaster Trading System
版本: 4.0.0  
创建时间: 2025-08-16
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class PositionSizeResult:
    """仓位大小计算结果"""
    symbol: str
    base_size: float
    adjusted_size: float
    kelly_fraction: float
    volatility_adjustment: float
    confidence_adjustment: float
    risk_budget_used: float
    max_size_constraint: float
    final_size: float
    reasoning: str

class DynamicPositionSizer:
    """
    动态仓位大小优化器
    
    核心算法:
    1. Kelly准则基础仓位计算
    2. 波动率目标调整 (Volatility Targeting)
    3. 信号置信度权重调整
    4. 风险预算分配管理
    5. 市场制度识别调整
    6. 相关性惩罚调整
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 仓位大小限制
        self.min_position_usd = config.get('min_position_usd', 100)
        self.max_position_usd = config.get('max_position_usd', 3000)
        self.max_position_pct = config.get('max_position_pct', 0.30)
        self.max_portfolio_heat = config.get('max_portfolio_heat', 0.15)  # 最大组合风险
        
        # Kelly准则参数
        self.max_kelly_fraction = config.get('max_kelly_fraction', 0.25)
        self.kelly_confidence_threshold = config.get('kelly_confidence_threshold', 0.75)
        self.lookback_trades = config.get('lookback_trades', 100)
        
        # 波动率目标
        self.target_volatility = config.get('target_volatility', 0.08)  # 8%日标的波动率
        self.volatility_lookback = config.get('volatility_lookback', 20)  # 20日波动率窗口
        
        # 风险预算
        self.total_risk_budget = config.get('total_risk_budget', 1.0)
        self.risk_allocation_method = config.get('risk_allocation_method', 'equal_risk')  # equal_risk, signal_weighted
        
        # 市场制度调整
        self.regime_multipliers = config.get('regime_multipliers', {
            'low_vol': 1.3,
            'normal_vol': 1.0,
            'high_vol': 0.6,
            'crisis': 0.3
        })
        
        # 相关性惩罚
        self.correlation_penalty_threshold = config.get('correlation_penalty_threshold', 0.7)
        self.correlation_penalty_factor = config.get('correlation_penalty_factor', 0.5)
        
        # 历史性能数据 (用于Kelly计算)
        self.historical_performance = {
            'win_rate': config.get('historical_win_rate', 0.85),
            'avg_win': config.get('historical_avg_win', 0.012),
            'avg_loss': config.get('historical_avg_loss', 0.006),
            'profit_factor': config.get('historical_profit_factor', 1.8)
        }
        
    def calculate_optimal_sizes(self, 
                               alpha_signals: pd.DataFrame,
                               market_data: pd.DataFrame,
                               current_positions: Optional[Dict[str, float]] = None,
                               available_capital: float = 10000) -> List[PositionSizeResult]:
        """
        计算多币种的最优仓位大小
        
        参数:
        - alpha_signals: Alpha信号 [timestamp, symbol, score, confidence, predicted_return]
        - market_data: 市场数据 [timestamp, symbol, close, volume, returns]
        - current_positions: 当前仓位 {symbol: usd_value}
        - available_capital: 可用资金
        
        返回:
        - 仓位大小结果列表
        """
        
        results = []
        
        # 1. 数据预处理
        processed_data = self._preprocess_market_data(market_data)
        latest_signals = self._get_latest_signals(alpha_signals)
        
        # 2. 计算市场制度
        market_regime = self._identify_market_regime(processed_data)
        
        # 3. 风险预算分配
        risk_allocations = self._allocate_risk_budget(latest_signals)
        
        # 4. 计算相关性矩阵
        correlation_matrix = self._calculate_correlation_matrix(processed_data['returns'])
        
        # 5. 为每个信号计算仓位大小
        for _, signal in latest_signals.iterrows():
            symbol = signal['symbol']
            
            # 基础Kelly仓位
            kelly_size = self._calculate_kelly_position(signal, symbol)
            
            # 波动率调整
            vol_adjustment = self._calculate_volatility_adjustment(symbol, processed_data)
            
            # 置信度调整
            conf_adjustment = self._calculate_confidence_adjustment(signal['confidence'])
            
            # 市场制度调整
            regime_adjustment = self._get_regime_adjustment(market_regime)
            
            # 相关性惩罚
            corr_penalty = self._calculate_correlation_penalty(symbol, current_positions, correlation_matrix)
            
            # 风险预算约束
            risk_budget = risk_allocations.get(symbol, 0.1)
            
            # 综合计算
            result = self._compute_final_position_size(
                symbol=symbol,
                kelly_size=kelly_size,
                vol_adjustment=vol_adjustment,
                conf_adjustment=conf_adjustment,
                regime_adjustment=regime_adjustment,
                corr_penalty=corr_penalty,
                risk_budget=risk_budget,
                available_capital=available_capital,
                signal=signal
            )
            
            results.append(result)
        
        # 6. 组合层面约束检查和调整
        results = self._apply_portfolio_constraints(results, available_capital)
        
        return results
    
    def _preprocess_market_data(self, market_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """预处理市场数据"""
        
        # 确保timestamp为数值类型并去重
        if 'timestamp' in market_data.columns:
            market_data['timestamp'] = pd.to_numeric(market_data['timestamp'], errors='coerce')
        
        market_data = market_data.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        
        # 透视表转换
        try:
            prices = market_data.pivot(index='timestamp', columns='symbol', values='close')
            volumes = market_data.pivot(index='timestamp', columns='symbol', values='volume')
        except ValueError:
            prices = market_data.groupby(['timestamp', 'symbol'])['close'].last().unstack('symbol')
            volumes = market_data.groupby(['timestamp', 'symbol'])['volume'].last().unstack('symbol')
        
        # 计算收益率
        returns = prices.pct_change().dropna()
        
        # 计算波动率 (滚动窗口)
        volatility = returns.rolling(self.volatility_lookback).std()
        
        return {
            'prices': prices,
            'volumes': volumes, 
            'returns': returns,
            'volatility': volatility
        }
    
    def _get_latest_signals(self, alpha_signals: pd.DataFrame) -> pd.DataFrame:
        """获取最新的Alpha信号"""
        
        return alpha_signals.sort_values('timestamp').groupby('symbol').tail(1).reset_index(drop=True)
    
    def _identify_market_regime(self, processed_data: Dict[str, pd.DataFrame]) -> str:
        """识别当前市场制度"""
        
        returns = processed_data['returns']
        
        # 计算市场整体波动率 (等权重)
        market_returns = returns.mean(axis=1, skipna=True)
        current_vol = market_returns.rolling(self.volatility_lookback).std().iloc[-1]
        historical_vol = market_returns.rolling(252).std().mean()  # 历史平均波动率
        
        # 制度分类
        vol_ratio = current_vol / historical_vol if historical_vol > 0 else 1.0
        
        if vol_ratio < 0.7:
            return 'low_vol'
        elif vol_ratio > 2.0:
            return 'crisis'
        elif vol_ratio > 1.5:
            return 'high_vol'
        else:
            return 'normal_vol'
    
    def _allocate_risk_budget(self, signals: pd.DataFrame) -> Dict[str, float]:
        """分配风险预算"""
        
        if self.risk_allocation_method == 'equal_risk':
            # 等风险分配
            n_signals = len(signals)
            if n_signals > 0:
                equal_allocation = self.total_risk_budget / n_signals
                return {row['symbol']: equal_allocation for _, row in signals.iterrows()}
            else:
                return {}
        
        elif self.risk_allocation_method == 'signal_weighted':
            # 基于信号强度加权分配
            total_signal_strength = signals['score'].sum()
            if total_signal_strength > 0:
                return {
                    row['symbol']: (row['score'] / total_signal_strength) * self.total_risk_budget
                    for _, row in signals.iterrows()
                }
            else:
                return {}
        
        else:
            # 默认等权重
            n_signals = len(signals)
            if n_signals > 0:
                return {row['symbol']: 1.0 / n_signals for _, row in signals.iterrows()}
            else:
                return {}
    
    def _calculate_correlation_matrix(self, returns: pd.DataFrame) -> pd.DataFrame:
        """计算相关性矩阵"""
        
        return returns.corr()
    
    def _calculate_kelly_position(self, signal: pd.Series, symbol: str) -> float:
        """计算Kelly准则仓位分数"""
        
        # 从信号中获取预期收益
        expected_return = signal['predicted_return']
        confidence = signal['confidence']
        
        # 使用历史性能数据
        win_rate = self.historical_performance['win_rate']
        avg_win = self.historical_performance['avg_win']
        avg_loss = self.historical_performance['avg_loss']
        
        # Kelly公式: f = (bp - q) / b
        # b = 盈亏比, p = 胜率, q = 败率
        if avg_loss > 0:
            profit_ratio = avg_win / avg_loss  # b
            kelly_fraction = (profit_ratio * win_rate - (1 - win_rate)) / profit_ratio
            
            # 置信度调整 - 只有高置信度信号才使用完整Kelly
            if confidence >= self.kelly_confidence_threshold:
                kelly_fraction *= confidence
            else:
                kelly_fraction *= confidence * 0.5  # 低置信度减半
            
            # Kelly分数限制
            kelly_fraction = max(0, min(kelly_fraction, self.max_kelly_fraction))
            
            return kelly_fraction
        else:
            return 0.0
    
    def _calculate_volatility_adjustment(self, symbol: str, processed_data: Dict[str, pd.DataFrame]) -> float:
        """计算波动率调整因子"""
        
        volatility_data = processed_data['volatility']
        
        if symbol in volatility_data.columns:
            current_vol = volatility_data[symbol].iloc[-1]
            
            if pd.isna(current_vol) or current_vol <= 0:
                return 1.0
            
            # 波动率目标调整: target_vol / current_vol
            vol_adjustment = self.target_volatility / current_vol
            
            # 限制调整幅度 (0.3x - 3.0x)
            vol_adjustment = max(0.3, min(vol_adjustment, 3.0))
            
            return vol_adjustment
        else:
            return 1.0
    
    def _calculate_confidence_adjustment(self, confidence: float) -> float:
        """计算信号置信度调整因子"""
        
        # 非线性置信度调整
        if confidence >= 0.9:
            return 1.2  # 高置信度增强
        elif confidence >= 0.8:
            return 1.0  # 正常
        elif confidence >= 0.7:
            return 0.8  # 轻微减少
        elif confidence >= 0.6:
            return 0.6  # 显著减少
        else:
            return 0.3  # 低置信度大幅减少
    
    def _get_regime_adjustment(self, market_regime: str) -> float:
        """获取市场制度调整因子"""
        
        return self.regime_multipliers.get(market_regime, 1.0)
    
    def _calculate_correlation_penalty(self, symbol: str, 
                                     current_positions: Optional[Dict[str, float]],
                                     correlation_matrix: pd.DataFrame) -> float:
        """计算相关性惩罚因子"""
        
        if current_positions is None or len(current_positions) == 0:
            return 1.0
        
        if symbol not in correlation_matrix.index:
            return 1.0
        
        penalty = 1.0
        
        for pos_symbol, position_size in current_positions.items():
            if pos_symbol != symbol and pos_symbol in correlation_matrix.columns:
                correlation = abs(correlation_matrix.loc[symbol, pos_symbol])
                
                if correlation > self.correlation_penalty_threshold:
                    # 相关性惩罚与仓位大小和相关性强度成正比
                    position_weight = position_size / 10000  # 假设10k基准
                    penalty_strength = (correlation - self.correlation_penalty_threshold) / (1 - self.correlation_penalty_threshold)
                    penalty *= (1 - self.correlation_penalty_factor * penalty_strength * position_weight)
        
        return max(0.1, penalty)  # 最小保留10%
    
    def _compute_final_position_size(self, **kwargs) -> PositionSizeResult:
        """计算最终仓位大小"""
        
        symbol = kwargs['symbol']
        kelly_size = kwargs['kelly_size']
        vol_adjustment = kwargs['vol_adjustment']
        conf_adjustment = kwargs['conf_adjustment']
        regime_adjustment = kwargs['regime_adjustment']
        corr_penalty = kwargs['corr_penalty']
        risk_budget = kwargs['risk_budget']
        available_capital = kwargs['available_capital']
        signal = kwargs['signal']
        
        # 基础仓位大小 (Kelly)
        base_size = kelly_size
        
        # 综合调整
        adjusted_size = (base_size * 
                        vol_adjustment * 
                        conf_adjustment * 
                        regime_adjustment * 
                        corr_penalty)
        
        # 风险预算约束
        risk_budget_size = risk_budget * available_capital
        
        # 应用各种限制
        max_size_by_pct = self.max_position_pct * available_capital
        max_size_by_usd = self.max_position_usd
        max_size_by_risk = risk_budget_size
        
        max_size_constraint = min(max_size_by_pct, max_size_by_usd, max_size_by_risk)
        
        # 最终仓位大小
        final_size_usd = min(adjusted_size * available_capital, max_size_constraint)
        final_size_usd = max(final_size_usd, self.min_position_usd) if final_size_usd > 0 else 0
        
        # 计算最终仓位比例
        final_size_pct = final_size_usd / available_capital if available_capital > 0 else 0
        
        # 生成推理说明
        reasoning = self._generate_reasoning(
            kelly_size, vol_adjustment, conf_adjustment, 
            regime_adjustment, corr_penalty, signal
        )
        
        return PositionSizeResult(
            symbol=symbol,
            base_size=base_size,
            adjusted_size=adjusted_size,
            kelly_fraction=kelly_size,
            volatility_adjustment=vol_adjustment,
            confidence_adjustment=conf_adjustment,
            risk_budget_used=risk_budget,
            max_size_constraint=max_size_constraint,
            final_size=final_size_pct,
            reasoning=reasoning
        )
    
    def _apply_portfolio_constraints(self, results: List[PositionSizeResult], 
                                   available_capital: float) -> List[PositionSizeResult]:
        """应用组合层面约束"""
        
        # 计算总风险暴露
        total_risk_exposure = sum(result.final_size for result in results)
        
        # 如果总风险超过限制，按比例缩放
        if total_risk_exposure > self.max_portfolio_heat:
            scale_factor = self.max_portfolio_heat / total_risk_exposure
            
            for result in results:
                result.final_size *= scale_factor
                result.reasoning += f" | 组合风险约束缩放: {scale_factor:.2f}"
        
        return results
    
    def _generate_reasoning(self, kelly_size: float, vol_adjustment: float,
                           conf_adjustment: float, regime_adjustment: float,
                           corr_penalty: float, signal: pd.Series) -> str:
        """生成仓位大小推理说明"""
        
        reasons = []
        
        reasons.append(f"Kelly基础: {kelly_size:.3f}")
        
        if vol_adjustment != 1.0:
            reasons.append(f"波动率调整: {vol_adjustment:.2f}x")
        
        if conf_adjustment != 1.0:
            reasons.append(f"置信度调整: {conf_adjustment:.2f}x")
            
        if regime_adjustment != 1.0:
            reasons.append(f"市场制度: {regime_adjustment:.2f}x")
            
        if corr_penalty != 1.0:
            reasons.append(f"相关性惩罚: {corr_penalty:.2f}x")
        
        reasons.append(f"信号分数: {signal['score']:.3f}")
        reasons.append(f"置信度: {signal['confidence']:.3f}")
        
        return " | ".join(reasons)
    
    def generate_position_report(self, results: List[PositionSizeResult]) -> Dict[str, Any]:
        """生成仓位大小分析报告"""
        
        if not results:
            return {"error": "无仓位计算结果"}
        
        # 统计汇总
        total_capital_used = sum(result.final_size for result in results)
        avg_position_size = total_capital_used / len(results) if results else 0
        max_position = max(result.final_size for result in results)
        min_position = min(result.final_size for result in results)
        
        # 调整因子分析
        avg_vol_adj = np.mean([result.volatility_adjustment for result in results])
        avg_conf_adj = np.mean([result.confidence_adjustment for result in results])
        
        # 风险预算使用
        total_risk_budget_used = sum(result.risk_budget_used for result in results)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "position_summary": {
                "total_positions": len(results),
                "total_capital_allocated": total_capital_used,
                "avg_position_size": avg_position_size,
                "max_position_size": max_position,
                "min_position_size": min_position,
                "capital_utilization": total_capital_used
            },
            "adjustment_analysis": {
                "avg_volatility_adjustment": avg_vol_adj,
                "avg_confidence_adjustment": avg_conf_adj,
                "total_risk_budget_used": total_risk_budget_used
            },
            "individual_positions": [
                {
                    "symbol": result.symbol,
                    "final_size_pct": result.final_size,
                    "kelly_fraction": result.kelly_fraction,
                    "vol_adjustment": result.volatility_adjustment,
                    "conf_adjustment": result.confidence_adjustment,
                    "reasoning": result.reasoning
                }
                for result in results
            ],
            "constraints_check": {
                "max_portfolio_heat_used": total_capital_used,
                "max_portfolio_heat_limit": self.max_portfolio_heat,
                "largest_position_pct": max_position,
                "max_position_limit": self.max_position_pct,
                "within_limits": (total_capital_used <= self.max_portfolio_heat and 
                                max_position <= self.max_position_pct)
            }
        }


def create_position_sizer(config: Dict[str, Any]) -> DynamicPositionSizer:
    """工厂函数：创建动态仓位大小优化器"""
    
    default_config = {
        'min_position_usd': 100,
        'max_position_usd': 3000,
        'max_position_pct': 0.30,
        'max_portfolio_heat': 0.15,
        'max_kelly_fraction': 0.25,
        'target_volatility': 0.08,
        'volatility_lookback': 20,
        'total_risk_budget': 1.0,
        'risk_allocation_method': 'signal_weighted',
        'historical_win_rate': 0.85,
        'historical_avg_win': 0.012,
        'historical_avg_loss': 0.006
    }
    
    # 合并配置
    merged_config = {**default_config, **config}
    
    return DynamicPositionSizer(merged_config)


if __name__ == "__main__":
    # 测试动态仓位大小优化器
    test_config = {
        'max_position_pct': 0.25,
        'target_volatility': 0.08,
        'max_kelly_fraction': 0.20
    }
    
    sizer = create_position_sizer(test_config)
    print("DipMaster V4 动态仓位大小优化器初始化成功")
    print(f"目标波动率: {sizer.target_volatility:.1%}")
    print(f"最大Kelly分数: {sizer.max_kelly_fraction:.1%}")
    print(f"最大单仓位: {sizer.max_position_pct:.1%}")