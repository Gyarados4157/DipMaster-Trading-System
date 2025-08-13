"""
Dynamic Symbol Scoring System - Phase 4 Optimization
动态币种评分系统：实时评估和排序最优交易标的

核心功能：
1. 历史表现评分：胜率、盈亏比、夏普率
2. 流动性评分：点差、成交量、深度
3. 波动率适配：ATR分析、波动率稳定性
4. 相关性管理：避免过度集中风险
5. 市场结构：趋势强度、均值回归特征
6. 实时排名更新：动态权重调整

目标：选出最适合DipMaster策略的高质量标的
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict

logger = logging.getLogger(__name__)


class ScoreCategory(Enum):
    """评分类别"""
    PERFORMANCE = "performance"      # 历史表现
    LIQUIDITY = "liquidity"         # 流动性
    VOLATILITY = "volatility"       # 波动率适配
    CORRELATION = "correlation"     # 相关性
    STRUCTURE = "structure"         # 市场结构
    MOMENTUM = "momentum"           # 动量特征


@dataclass
class SymbolScore:
    """币种评分详情"""
    symbol: str
    total_score: float = 0.0
    
    # 各项评分
    performance_score: float = 0.0
    liquidity_score: float = 0.0
    volatility_score: float = 0.0
    correlation_score: float = 0.0
    structure_score: float = 0.0
    momentum_score: float = 0.0
    
    # 具体指标
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    avg_spread_bps: float = 0.0
    daily_volume_usd: float = 0.0
    atr_percent: float = 0.0
    correlation_penalty: float = 0.0
    
    # 元数据
    last_updated: datetime = field(default_factory=datetime.now)
    trade_count: int = 0
    data_quality: float = 1.0
    is_active: bool = True
    
    @property
    def grade(self) -> str:
        """评分等级"""
        if self.total_score >= 80:
            return "A+"
        elif self.total_score >= 70:
            return "A"
        elif self.total_score >= 60:
            return "B"
        elif self.total_score >= 50:
            return "C"
        else:
            return "D"
            
    @property
    def recommendation(self) -> str:
        """交易建议"""
        if self.total_score >= 75:
            return "强烈推荐"
        elif self.total_score >= 60:
            return "推荐"
        elif self.total_score >= 45:
            return "谨慎交易"
        else:
            return "避免交易"


class DynamicSymbolScorer:
    """动态币种评分器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # === 评分权重配置 ===
        self.score_weights = {
            ScoreCategory.PERFORMANCE: 0.35,    # 历史表现最重要
            ScoreCategory.LIQUIDITY: 0.20,      # 流动性重要
            ScoreCategory.VOLATILITY: 0.15,     # 波动率适配
            ScoreCategory.CORRELATION: 0.10,    # 相关性管理
            ScoreCategory.STRUCTURE: 0.10,      # 市场结构
            ScoreCategory.MOMENTUM: 0.10        # 动量特征
        }
        
        # === 历史数据要求 ===
        self.min_trade_count = 20              # 最少交易次数
        self.performance_lookback_days = 90    # 表现回看天数
        self.correlation_window = 30           # 相关性计算窗口
        
        # === 流动性阈值 ===
        self.min_daily_volume_usd = 10_000_000  # 最小日成交量1000万
        self.max_spread_bps = 20                # 最大点差20BP
        self.preferred_spread_bps = 5           # 理想点差5BP
        
        # === 波动率范围 ===
        self.optimal_atr_range = (0.03, 0.08)  # 3%-8%最优ATR范围
        self.max_acceptable_atr = 0.15          # 最大可接受ATR
        
        # === 相关性控制 ===
        self.max_correlation_threshold = 0.7   # 最大相关性阈值
        self.correlation_penalty_factor = 2.0  # 相关性惩罚因子
        
        # 数据存储
        self.symbol_scores: Dict[str, SymbolScore] = {}
        self.trade_history: Dict[str, List[Dict]] = defaultdict(list)
        self.price_data: Dict[str, pd.DataFrame] = {}
        self.correlation_matrix: pd.DataFrame = pd.DataFrame()
        self.last_update: datetime = datetime.now()
        
        # 实时权重调整
        self.adaptive_weights = self.score_weights.copy()
        
    def update_trade_history(self, symbol: str, trade_result: Dict):
        """更新交易历史"""
        trade_data = {
            'timestamp': trade_result.get('timestamp', datetime.now()),
            'entry_price': trade_result.get('entry_price', 0),
            'exit_price': trade_result.get('exit_price', 0),
            'pnl_percent': trade_result.get('pnl_percent', 0),
            'pnl_usd': trade_result.get('pnl_usd', 0),
            'holding_minutes': trade_result.get('holding_minutes', 0),
            'exit_reason': trade_result.get('exit_reason', 'unknown'),
            'is_winner': trade_result.get('pnl_percent', 0) > 0
        }
        
        self.trade_history[symbol].append(trade_data)
        
        # 限制历史长度
        if len(self.trade_history[symbol]) > 200:
            self.trade_history[symbol] = self.trade_history[symbol][-150:]
            
        # 触发重新计算该币种分数
        self._recalculate_symbol_score(symbol)
        
    def update_price_data(self, symbol: str, df: pd.DataFrame):
        """更新价格数据"""
        # 保留最近的数据
        if len(df) > 1000:
            df = df.tail(1000).copy()
            
        self.price_data[symbol] = df
        
        # 触发重新计算
        self._recalculate_symbol_score(symbol)
        
    def _calculate_performance_score(self, symbol: str) -> Tuple[float, Dict]:
        """计算历史表现评分"""
        if symbol not in self.trade_history:
            return 0.0, {}
            
        trades = self.trade_history[symbol]
        recent_trades = [t for t in trades 
                        if (datetime.now() - t['timestamp']).days <= self.performance_lookback_days]
        
        if len(recent_trades) < self.min_trade_count:
            return 0.0, {'insufficient_data': True}
            
        # === 胜率计算 ===
        winners = [t for t in recent_trades if t['is_winner']]
        win_rate = len(winners) / len(recent_trades) * 100
        
        # === 盈亏比计算 ===
        if winners:
            avg_win = np.mean([t['pnl_percent'] for t in winners])
            losers = [t for t in recent_trades if not t['is_winner']]
            avg_loss = abs(np.mean([t['pnl_percent'] for t in losers])) if losers else 1
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 0
        else:
            avg_win = 0
            profit_factor = 0
            
        # === 夏普率计算 ===
        returns = [t['pnl_percent'] for t in recent_trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns)
        else:
            sharpe_ratio = 0
            
        # === 综合评分 ===
        # 胜率评分 (40权重)
        win_rate_score = min(win_rate, 100) * 0.4  # 胜率越高越好
        
        # 盈亏比评分 (35权重)
        profit_factor_score = min(profit_factor * 20, 35)  # 盈亏比>1.75得满分
        
        # 夏普率评分 (25权重)
        sharpe_score = max(0, min(sharpe_ratio * 12.5, 25))  # 夏普率>2得满分
        
        performance_score = win_rate_score + profit_factor_score + sharpe_score
        
        metrics = {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'trade_count': len(recent_trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
        
        return performance_score, metrics
        
    def _calculate_liquidity_score(self, symbol: str) -> Tuple[float, Dict]:
        """计算流动性评分"""
        if symbol not in self.price_data:
            return 0.0, {}
            
        df = self.price_data[symbol]
        if len(df) < 20:
            return 0.0, {}
            
        # === 成交量分析 ===
        recent_volume = df['volume'].tail(20).mean()
        recent_price = df['close'].tail(20).mean()
        daily_volume_usd = recent_volume * recent_price
        
        # 成交量评分 (60%)
        if daily_volume_usd >= self.min_daily_volume_usd * 5:  # 5倍最小值
            volume_score = 60
        elif daily_volume_usd >= self.min_daily_volume_usd:
            ratio = daily_volume_usd / self.min_daily_volume_usd
            volume_score = 30 + (ratio - 1) / 4 * 30  # 线性插值
        else:
            volume_score = daily_volume_usd / self.min_daily_volume_usd * 30
            
        # === 点差估算 ===
        # 使用高低价差作为点差的代理指标
        spreads = (df['high'] - df['low']) / df['close'] * 10000  # 转换为BP
        avg_spread_bps = spreads.tail(20).mean()
        
        # 点差评分 (40%)
        if avg_spread_bps <= self.preferred_spread_bps:
            spread_score = 40
        elif avg_spread_bps <= self.max_spread_bps:
            spread_score = 40 - (avg_spread_bps - self.preferred_spread_bps) / (self.max_spread_bps - self.preferred_spread_bps) * 20
        else:
            spread_score = max(0, 20 - (avg_spread_bps - self.max_spread_bps))
            
        liquidity_score = volume_score + spread_score
        
        metrics = {
            'daily_volume_usd': daily_volume_usd,
            'avg_spread_bps': avg_spread_bps,
            'volume_score': volume_score,
            'spread_score': spread_score
        }
        
        return liquidity_score, metrics
        
    def _calculate_volatility_score(self, symbol: str) -> Tuple[float, Dict]:
        """计算波动率适配评分"""
        if symbol not in self.price_data:
            return 0.0, {}
            
        df = self.price_data[symbol]
        if len(df) < 30:
            return 0.0, {}
            
        # === ATR计算 ===
        atr = self._calculate_atr(df)
        atr_percent = atr / df['close'].iloc[-1]
        
        # === 波动率稳定性 ===
        rolling_atr = []
        for i in range(14, len(df)):
            chunk_atr = self._calculate_atr(df.iloc[i-14:i])
            rolling_atr.append(chunk_atr / df['close'].iloc[i])
            
        atr_stability = 1 / (1 + np.std(rolling_atr)) if rolling_atr else 0.5
        
        # === 波动率评分 ===
        optimal_min, optimal_max = self.optimal_atr_range
        
        if optimal_min <= atr_percent <= optimal_max:
            atr_score = 70  # 最优范围得高分
        elif atr_percent < optimal_min:
            # 波动率过低
            atr_score = 70 * (atr_percent / optimal_min)
        elif atr_percent <= self.max_acceptable_atr:
            # 波动率偏高但可接受
            excess = atr_percent - optimal_max
            max_excess = self.max_acceptable_atr - optimal_max
            atr_score = 70 - (excess / max_excess) * 40
        else:
            # 波动率过高
            atr_score = 10
            
        # 稳定性评分 (30%)
        stability_score = atr_stability * 30
        
        volatility_score = atr_score + stability_score
        
        metrics = {
            'atr_percent': atr_percent * 100,
            'atr_stability': atr_stability,
            'optimal_range': self.optimal_atr_range,
            'atr_score': atr_score,
            'stability_score': stability_score
        }
        
        return volatility_score, metrics
        
    def _calculate_correlation_score(self, symbol: str, active_symbols: Set[str]) -> Tuple[float, Dict]:
        """计算相关性评分"""
        if len(active_symbols) <= 1:
            return 100.0, {'no_correlation_risk': True}
            
        # 获取该symbol与其他活跃symbols的相关性
        correlations = []
        
        for other_symbol in active_symbols:
            if other_symbol != symbol and symbol in self.price_data and other_symbol in self.price_data:
                corr = self._calculate_price_correlation(symbol, other_symbol)
                if not np.isnan(corr):
                    correlations.append(abs(corr))
                    
        if not correlations:
            return 100.0, {'no_data': True}
            
        # 最高相关性
        max_correlation = max(correlations)
        avg_correlation = np.mean(correlations)
        
        # 相关性惩罚
        if max_correlation <= 0.3:
            correlation_score = 100
        elif max_correlation <= self.max_correlation_threshold:
            penalty = (max_correlation - 0.3) / (self.max_correlation_threshold - 0.3)
            correlation_score = 100 - penalty * 50
        else:
            # 严重超标，大幅惩罚
            correlation_score = 50 - (max_correlation - self.max_correlation_threshold) * self.correlation_penalty_factor * 100
            
        correlation_score = max(0, correlation_score)
        
        metrics = {
            'max_correlation': max_correlation,
            'avg_correlation': avg_correlation,
            'correlation_count': len(correlations),
            'penalty_applied': 100 - correlation_score
        }
        
        return correlation_score, metrics
        
    def _calculate_structure_score(self, symbol: str) -> Tuple[float, Dict]:
        """计算市场结构评分"""
        if symbol not in self.price_data:
            return 0.0, {}
            
        df = self.price_data[symbol]
        if len(df) < 50:
            return 0.0, {}
            
        # === 趋势强度分析 ===
        # 使用EMA斜率衡量趋势
        ema20 = df['close'].ewm(span=20).mean()
        ema_slope = (ema20.iloc[-1] - ema20.iloc[-10]) / ema20.iloc[-10]
        
        # DipMaster喜欢轻微下跌或横盘
        if -0.02 <= ema_slope <= 0.01:  # 轻微下跌到微涨
            trend_score = 40
        elif ema_slope > 0.01:  # 过于强势
            trend_score = 40 - min((ema_slope - 0.01) * 1000, 30)
        else:  # 过于弱势
            trend_score = 40 - min((abs(ema_slope) - 0.02) * 800, 35)
            
        # === 均值回归特征 ===
        # 计算价格偏离均线的分布
        price_deviation = (df['close'] - ema20) / ema20
        deviation_std = price_deviation.std()
        
        # 适中的偏离标准差最佳（0.02-0.05）
        if 0.02 <= deviation_std <= 0.05:
            mean_reversion_score = 35
        elif deviation_std < 0.02:
            mean_reversion_score = 25  # 过于稳定
        else:
            mean_reversion_score = max(10, 35 - (deviation_std - 0.05) * 400)
            
        # === 支撑阻力分析 ===
        # 使用布林带宽度评估
        bb_width = self._calculate_bollinger_width(df)
        if 0.04 <= bb_width <= 0.12:  # 4%-12%带宽最佳
            sr_score = 25
        else:
            sr_score = max(10, 25 - abs(bb_width - 0.08) * 200)
            
        structure_score = trend_score + mean_reversion_score + sr_score
        
        metrics = {
            'ema_slope': ema_slope,
            'deviation_std': deviation_std,
            'bollinger_width': bb_width,
            'trend_score': trend_score,
            'mean_reversion_score': mean_reversion_score,
            'support_resistance_score': sr_score
        }
        
        return structure_score, metrics
        
    def _calculate_momentum_score(self, symbol: str) -> Tuple[float, Dict]:
        """计算动量特征评分"""
        if symbol not in self.price_data:
            return 0.0, {}
            
        df = self.price_data[symbol]
        if len(df) < 30:
            return 0.0, {}
            
        # === RSI分布分析 ===
        rsi = self._calculate_rsi(df['close'])
        recent_rsi = rsi.tail(20)
        
        # DipMaster喜欢RSI在30-50区间
        rsi_in_range = ((recent_rsi >= 30) & (recent_rsi <= 50)).mean()
        rsi_score = rsi_in_range * 50  # 在目标区间的时间比例
        
        # === 动量衰减特征 ===
        # 计算价格动量的衰减模式
        momentum_3 = df['close'].pct_change(3)
        momentum_5 = df['close'].pct_change(5)
        momentum_10 = df['close'].pct_change(10)
        
        # 检查动量衰减（适合抄底）
        recent_mom3 = momentum_3.tail(10)
        recent_mom5 = momentum_5.tail(10)
        recent_mom10 = momentum_10.tail(10)
        
        # 动量从负向零回归的比例
        momentum_recovery = ((recent_mom3 > recent_mom5) & (recent_mom5 > recent_mom10) & (recent_mom10 < 0)).mean()
        momentum_score = momentum_recovery * 30
        
        # === 成交量-价格关系 ===
        # 下跌时成交量放大的程度
        price_change = df['close'].pct_change()
        volume_change = df['volume'].pct_change()
        
        # 价格下跌时的成交量表现
        down_days = price_change < -0.01
        if down_days.any():
            volume_on_down = volume_change[down_days].mean()
            volume_price_score = min(20, max(0, volume_on_down * 100 + 10))
        else:
            volume_price_score = 10
            
        momentum_total = rsi_score + momentum_score + volume_price_score
        
        metrics = {
            'rsi_in_target_range': rsi_in_range,
            'momentum_recovery_rate': momentum_recovery,
            'volume_on_decline': volume_on_down if down_days.any() else 0,
            'rsi_score': rsi_score,
            'momentum_score': momentum_score,
            'volume_price_score': volume_price_score
        }
        
        return momentum_total, metrics
        
    def _recalculate_symbol_score(self, symbol: str, active_symbols: Set[str] = None):
        """重新计算币种综合评分"""
        if active_symbols is None:
            active_symbols = set(self.symbol_scores.keys())
            
        # 各项评分计算
        perf_score, perf_metrics = self._calculate_performance_score(symbol)
        liq_score, liq_metrics = self._calculate_liquidity_score(symbol)
        vol_score, vol_metrics = self._calculate_volatility_score(symbol)
        corr_score, corr_metrics = self._calculate_correlation_score(symbol, active_symbols)
        struct_score, struct_metrics = self._calculate_structure_score(symbol)
        mom_score, mom_metrics = self._calculate_momentum_score(symbol)
        
        # 加权综合评分
        total_score = (
            perf_score * self.adaptive_weights[ScoreCategory.PERFORMANCE] +
            liq_score * self.adaptive_weights[ScoreCategory.LIQUIDITY] +
            vol_score * self.adaptive_weights[ScoreCategory.VOLATILITY] +
            corr_score * self.adaptive_weights[ScoreCategory.CORRELATION] +
            struct_score * self.adaptive_weights[ScoreCategory.STRUCTURE] +
            mom_score * self.adaptive_weights[ScoreCategory.MOMENTUM]
        )
        
        # 创建或更新SymbolScore
        if symbol not in self.symbol_scores:
            self.symbol_scores[symbol] = SymbolScore(symbol=symbol)
            
        score_obj = self.symbol_scores[symbol]
        score_obj.total_score = total_score
        score_obj.performance_score = perf_score
        score_obj.liquidity_score = liq_score
        score_obj.volatility_score = vol_score
        score_obj.correlation_score = corr_score
        score_obj.structure_score = struct_score
        score_obj.momentum_score = mom_score
        
        # 更新具体指标
        score_obj.win_rate = perf_metrics.get('win_rate', 0)
        score_obj.profit_factor = perf_metrics.get('profit_factor', 0)
        score_obj.sharpe_ratio = perf_metrics.get('sharpe_ratio', 0)
        score_obj.avg_spread_bps = liq_metrics.get('avg_spread_bps', 0)
        score_obj.daily_volume_usd = liq_metrics.get('daily_volume_usd', 0)
        score_obj.atr_percent = vol_metrics.get('atr_percent', 0)
        score_obj.correlation_penalty = 100 - corr_score
        score_obj.trade_count = perf_metrics.get('trade_count', 0)
        score_obj.last_updated = datetime.now()
        
        logger.info(f"{symbol}评分更新: {total_score:.1f} [P:{perf_score:.0f} L:{liq_score:.0f} V:{vol_score:.0f} C:{corr_score:.0f} S:{struct_score:.0f} M:{mom_score:.0f}]")
        
    def get_top_symbols(self, limit: int = 5, min_score: float = 50.0) -> List[SymbolScore]:
        """获取评分最高的币种"""
        valid_symbols = [
            score for score in self.symbol_scores.values()
            if score.total_score >= min_score and score.is_active
        ]
        
        # 按总分排序
        sorted_symbols = sorted(valid_symbols, key=lambda x: x.total_score, reverse=True)
        return sorted_symbols[:limit]
        
    def get_symbol_ranking(self) -> List[Dict]:
        """获取完整币种排名"""
        ranking = []
        
        for symbol, score in self.symbol_scores.items():
            ranking.append({
                'symbol': symbol,
                'total_score': score.total_score,
                'grade': score.grade,
                'recommendation': score.recommendation,
                'win_rate': score.win_rate,
                'profit_factor': score.profit_factor,
                'daily_volume_usd': score.daily_volume_usd,
                'atr_percent': score.atr_percent,
                'trade_count': score.trade_count,
                'last_updated': score.last_updated
            })
            
        return sorted(ranking, key=lambda x: x['total_score'], reverse=True)
        
    def update_all_scores(self):
        """更新所有币种评分"""
        active_symbols = set(self.symbol_scores.keys())
        
        for symbol in list(active_symbols):
            try:
                self._recalculate_symbol_score(symbol, active_symbols)
            except Exception as e:
                logger.error(f"Error updating score for {symbol}: {e}")
                
        self.last_update = datetime.now()
        
    def adjust_weights_based_on_performance(self):
        """根据历史表现调整评分权重"""
        # 统计各评分维度与实际盈利的相关性
        # 这里简化处理，实际可以更复杂
        
        # 如果最近胜率下降，增加历史表现权重
        recent_performance = []
        for trades in self.trade_history.values():
            if len(trades) >= 10:
                recent_trades = trades[-10:]
                win_rate = sum(1 for t in recent_trades if t['is_winner']) / len(recent_trades)
                recent_performance.append(win_rate)
                
        if recent_performance:
            avg_win_rate = np.mean(recent_performance)
            if avg_win_rate < 0.65:  # 胜率低于65%
                self.adaptive_weights[ScoreCategory.PERFORMANCE] *= 1.1
                self.adaptive_weights[ScoreCategory.LIQUIDITY] *= 0.95
                
        # 重新标准化权重
        total_weight = sum(self.adaptive_weights.values())
        for key in self.adaptive_weights:
            self.adaptive_weights[key] /= total_weight
            
    # === 辅助计算函数 ===
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR"""
        if len(df) < period + 1:
            return df['high'].iloc[-1] - df['low'].iloc[-1]
            
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        
        return true_range.rolling(period).mean().iloc[-1]
        
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_price_correlation(self, symbol1: str, symbol2: str, window: int = 30) -> float:
        """计算两个币种的价格相关性"""
        if symbol1 not in self.price_data or symbol2 not in self.price_data:
            return 0.0
            
        df1 = self.price_data[symbol1].tail(window)
        df2 = self.price_data[symbol2].tail(window)
        
        # 对齐时间
        min_len = min(len(df1), len(df2))
        if min_len < 10:
            return 0.0
            
        returns1 = df1['close'].tail(min_len).pct_change().dropna()
        returns2 = df2['close'].tail(min_len).pct_change().dropna()
        
        if len(returns1) < 5 or len(returns2) < 5:
            return 0.0
            
        return returns1.corr(returns2)
        
    def _calculate_bollinger_width(self, df: pd.DataFrame, period: int = 20, std: float = 2) -> float:
        """计算布林带宽度"""
        if len(df) < period:
            return 0.08
            
        ma = df['close'].rolling(period).mean()
        bb_std = df['close'].rolling(period).std()
        
        upper = ma + (bb_std * std)
        lower = ma - (bb_std * std)
        
        width = (upper - lower) / ma
        return width.iloc[-1] if not pd.isna(width.iloc[-1]) else 0.08