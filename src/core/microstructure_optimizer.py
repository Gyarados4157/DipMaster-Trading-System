"""
Microstructure Optimizer
微观结构优化 - 订单簿分析、主被动切换、时机选择
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque, defaultdict
import math

logger = logging.getLogger(__name__)


class OrderBookAnalysisType(Enum):
    """订单簿分析类型"""
    DEPTH_ANALYSIS = "depth"
    IMBALANCE_ANALYSIS = "imbalance"
    FLOW_ANALYSIS = "flow"
    SPREAD_ANALYSIS = "spread"


class ExecutionStrategy(Enum):
    """执行策略类型"""
    PASSIVE_MAKER = "passive"      # 被动挂单
    AGGRESSIVE_TAKER = "aggressive" # 主动吃单
    ADAPTIVE = "adaptive"          # 自适应
    STEALTH = "stealth"           # 隐蔽执行


@dataclass
class OrderBookLevel:
    """订单簿价格层级"""
    price: float
    quantity: float
    order_count: int = 1
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrderBookSnapshot:
    """订单簿快照"""
    symbol: str
    venue: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None
    
    @property
    def spread(self) -> float:
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return 0.0
    
    @property
    def mid_price(self) -> float:
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2
        return 0.0


@dataclass
class MicrostructureSignal:
    """微观结构信号"""
    signal_type: str
    strength: float  # 0-1
    confidence: float  # 0-1
    direction: str  # 'buy', 'sell', 'neutral'
    timeframe_seconds: int
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionTiming:
    """执行时机建议"""
    action: str  # 'execute_now', 'wait', 'cancel'
    optimal_strategy: ExecutionStrategy
    expected_slippage_bps: float
    confidence_score: float
    wait_time_seconds: Optional[int] = None
    price_improvement_bps: Optional[float] = None


class MicrostructureOptimizer:
    """微观结构优化器"""
    
    def __init__(self):
        # 订单簿数据存储
        self.order_book_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.trade_flow_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        
        # 微观结构指标
        self.imbalance_threshold = 0.3  # 30%失衡阈值
        self.spread_percentile_threshold = 0.8  # 价差80分位数阈值
        self.volume_surge_multiplier = 2.0  # 成交量激增倍数
        
        # 执行策略参数
        self.passive_timeout_seconds = 30  # 被动订单超时
        self.stealth_slice_ratio = 0.1  # 隐蔽执行每次分割比例
        self.liquidity_detection_window = 60  # 流动性检测窗口(秒)
        
        # 性能统计
        self.execution_performance: Dict[str, List[float]] = defaultdict(list)
        self.strategy_success_rate: Dict[ExecutionStrategy, float] = {}
        
    async def analyze_order_book(
        self, 
        order_book: OrderBookSnapshot, 
        analysis_type: OrderBookAnalysisType = OrderBookAnalysisType.DEPTH_ANALYSIS
    ) -> MicrostructureSignal:
        """
        分析订单簿微观结构
        """
        logger.debug(f"分析订单簿: {order_book.symbol} - {analysis_type.value}")
        
        # 存储历史数据
        self.order_book_history[order_book.symbol].append(order_book)
        
        if analysis_type == OrderBookAnalysisType.DEPTH_ANALYSIS:
            return await self._analyze_depth(order_book)
        elif analysis_type == OrderBookAnalysisType.IMBALANCE_ANALYSIS:
            return await self._analyze_imbalance(order_book)
        elif analysis_type == OrderBookAnalysisType.FLOW_ANALYSIS:
            return await self._analyze_flow(order_book)
        elif analysis_type == OrderBookAnalysisType.SPREAD_ANALYSIS:
            return await self._analyze_spread(order_book)
        else:
            raise ValueError(f"不支持的分析类型: {analysis_type}")
    
    async def _analyze_depth(self, order_book: OrderBookSnapshot) -> MicrostructureSignal:
        """深度分析 - 分析订单簿深度和流动性"""
        
        # 计算买卖盘深度
        bid_depth = sum(level.quantity for level in order_book.bids[:10])  # 前10档
        ask_depth = sum(level.quantity for level in order_book.asks[:10])
        
        total_depth = bid_depth + ask_depth
        if total_depth == 0:
            return MicrostructureSignal(
                signal_type="depth_analysis",
                strength=0.0,
                confidence=0.0,
                direction="neutral",
                timeframe_seconds=30,
                reasoning="无流动性数据"
            )
        
        # 计算深度失衡
        depth_imbalance = (bid_depth - ask_depth) / total_depth
        
        # 计算平均订单大小
        avg_bid_size = bid_depth / len(order_book.bids) if order_book.bids else 0
        avg_ask_size = ask_depth / len(order_book.asks) if order_book.asks else 0
        
        # 分析历史深度变化
        history = list(self.order_book_history[order_book.symbol])
        if len(history) > 5:
            recent_depths = [(sum(ob.bids[0].quantity for ob in history[-5:]), 
                            sum(ob.asks[0].quantity for ob in history[-5:])) for ob in history[-5:]]
            depth_trend = np.polyfit(range(len(recent_depths)), 
                                   [bd - ad for bd, ad in recent_depths], 1)[0]
        else:
            depth_trend = 0
        
        # 生成信号
        signal_strength = abs(depth_imbalance)
        confidence = min(1.0, total_depth / 100)  # 基于总深度的置信度
        
        if depth_imbalance > 0.2:  # 买盘深度占优
            direction = "buy"
            reasoning = f"买盘深度占优 {depth_imbalance:.2%}, 总深度: {total_depth:.2f}"
        elif depth_imbalance < -0.2:  # 卖盘深度占优
            direction = "sell"
            reasoning = f"卖盘深度占优 {abs(depth_imbalance):.2%}, 总深度: {total_depth:.2f}"
        else:
            direction = "neutral"
            reasoning = f"深度均衡, 失衡度: {depth_imbalance:.2%}"
        
        return MicrostructureSignal(
            signal_type="depth_analysis",
            strength=signal_strength,
            confidence=confidence,
            direction=direction,
            timeframe_seconds=30,
            reasoning=reasoning
        )
    
    async def _analyze_imbalance(self, order_book: OrderBookSnapshot) -> MicrostructureSignal:
        """失衡分析 - 分析买卖盘失衡情况"""
        
        # 计算价格加权失衡
        bid_volume_weighted = sum(level.price * level.quantity for level in order_book.bids[:5])
        ask_volume_weighted = sum(level.price * level.quantity for level in order_book.asks[:5])
        
        total_weighted = bid_volume_weighted + ask_volume_weighted
        if total_weighted == 0:
            return MicrostructureSignal(
                signal_type="imbalance_analysis",
                strength=0.0,
                confidence=0.0,
                direction="neutral",
                timeframe_seconds=15,
                reasoning="无有效价格数据"
            )
        
        weighted_imbalance = (bid_volume_weighted - ask_volume_weighted) / total_weighted
        
        # 计算订单数量失衡
        bid_orders = sum(level.order_count for level in order_book.bids[:5])
        ask_orders = sum(level.order_count for level in order_book.asks[:5])
        total_orders = bid_orders + ask_orders
        
        order_imbalance = (bid_orders - ask_orders) / total_orders if total_orders > 0 else 0
        
        # 综合失衡指标
        combined_imbalance = (weighted_imbalance + order_imbalance) / 2
        
        # 计算历史失衡变化率
        history = list(self.order_book_history[order_book.symbol])
        imbalance_momentum = 0
        if len(history) > 3:
            recent_imbalances = []
            for ob in history[-3:]:
                bid_wt = sum(l.price * l.quantity for l in ob.bids[:5])
                ask_wt = sum(l.price * l.quantity for l in ob.asks[:5])
                total_wt = bid_wt + ask_wt
                if total_wt > 0:
                    imb = (bid_wt - ask_wt) / total_wt
                    recent_imbalances.append(imb)
            
            if len(recent_imbalances) > 1:
                imbalance_momentum = recent_imbalances[-1] - recent_imbalances[0]
        
        signal_strength = abs(combined_imbalance)
        confidence = min(1.0, signal_strength * 2)  # 失衡越大置信度越高
        
        if combined_imbalance > self.imbalance_threshold:
            direction = "buy"
            reasoning = f"买盘失衡 {combined_imbalance:.2%}, 动量: {imbalance_momentum:.2%}"
        elif combined_imbalance < -self.imbalance_threshold:
            direction = "sell"
            reasoning = f"卖盘失衡 {abs(combined_imbalance):.2%}, 动量: {imbalance_momentum:.2%}"
        else:
            direction = "neutral"
            reasoning = f"失衡中性 {combined_imbalance:.2%}"
        
        return MicrostructureSignal(
            signal_type="imbalance_analysis",
            strength=signal_strength,
            confidence=confidence,
            direction=direction,
            timeframe_seconds=15,
            reasoning=reasoning
        )
    
    async def _analyze_flow(self, order_book: OrderBookSnapshot) -> MicrostructureSignal:
        """流量分析 - 分析订单流和价格变化"""
        
        history = list(self.order_book_history[order_book.symbol])
        if len(history) < 5:
            return MicrostructureSignal(
                signal_type="flow_analysis",
                strength=0.0,
                confidence=0.0,
                direction="neutral",
                timeframe_seconds=60,
                reasoning="历史数据不足"
            )
        
        # 计算价格动量
        prices = [ob.mid_price for ob in history[-10:]]
        price_momentum = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
        
        # 计算成交量动量
        volumes = []
        for ob in history[-5:]:
            vol = sum(level.quantity for level in ob.bids[:5] + ob.asks[:5])
            volumes.append(vol)
        
        volume_change = (volumes[-1] - volumes[0]) / volumes[0] if volumes[0] > 0 else 0
        
        # 计算价差变化
        spreads = [ob.spread for ob in history[-5:]]
        spread_trend = np.polyfit(range(len(spreads)), spreads, 1)[0] if len(spreads) > 2 else 0
        
        # 计算订单簿更新频率
        update_intervals = []
        for i in range(1, len(history)):
            interval = (history[i].timestamp - history[i-1].timestamp).total_seconds()
            update_intervals.append(interval)
        
        avg_update_interval = np.mean(update_intervals) if update_intervals else 10
        update_frequency = 1 / avg_update_interval if avg_update_interval > 0 else 0.1
        
        # 综合流量信号
        flow_intensity = abs(price_momentum) + abs(volume_change) * 0.5 + update_frequency * 0.1
        
        signal_strength = min(1.0, flow_intensity * 10)
        confidence = min(1.0, len(history) / 20)  # 基于历史数据长度
        
        if price_momentum > 0.001:  # 0.1%上涨
            direction = "buy"
            reasoning = f"价格上涨动量 {price_momentum:.3%}, 成交量变化 {volume_change:.2%}"
        elif price_momentum < -0.001:  # 0.1%下跌
            direction = "sell"
            reasoning = f"价格下跌动量 {abs(price_momentum):.3%}, 成交量变化 {volume_change:.2%}"
        else:
            direction = "neutral"
            reasoning = f"价格横盘, 动量 {price_momentum:.3%}"
        
        return MicrostructureSignal(
            signal_type="flow_analysis",
            strength=signal_strength,
            confidence=confidence,
            direction=direction,
            timeframe_seconds=60,
            reasoning=reasoning
        )
    
    async def _analyze_spread(self, order_book: OrderBookSnapshot) -> MicrostructureSignal:
        """价差分析 - 分析买卖价差的变化"""
        
        current_spread = order_book.spread
        mid_price = order_book.mid_price
        
        if mid_price == 0:
            return MicrostructureSignal(
                signal_type="spread_analysis",
                strength=0.0,
                confidence=0.0,
                direction="neutral",
                timeframe_seconds=20,
                reasoning="无有效价格"
            )
        
        spread_bps = (current_spread / mid_price) * 10000
        
        # 分析历史价差
        history = list(self.order_book_history[order_book.symbol])
        if len(history) > 10:
            historical_spreads = []
            for ob in history[-20:]:
                if ob.mid_price > 0:
                    hist_spread_bps = (ob.spread / ob.mid_price) * 10000
                    historical_spreads.append(hist_spread_bps)
            
            if historical_spreads:
                avg_spread = np.mean(historical_spreads)
                spread_percentile = np.percentile(historical_spreads, 80)
                spread_volatility = np.std(historical_spreads)
                
                # 价差相对位置
                spread_position = (spread_bps - avg_spread) / spread_volatility if spread_volatility > 0 else 0
            else:
                avg_spread = spread_bps
                spread_percentile = spread_bps
                spread_position = 0
        else:
            avg_spread = spread_bps
            spread_percentile = spread_bps
            spread_position = 0
        
        # 计算效率指标
        if spread_bps < avg_spread * 0.8:  # 价差收窄
            efficiency_signal = "tight"
            signal_strength = 0.7
            reasoning = f"价差收窄至 {spread_bps:.1f}bps (平均 {avg_spread:.1f}bps)"
        elif spread_bps > spread_percentile:  # 价差过宽
            efficiency_signal = "wide"
            signal_strength = 0.8
            reasoning = f"价差扩大至 {spread_bps:.1f}bps (80分位 {spread_percentile:.1f}bps)"
        else:
            efficiency_signal = "normal"
            signal_strength = 0.3
            reasoning = f"价差正常 {spread_bps:.1f}bps"
        
        # 基于价差状态给出执行建议
        if efficiency_signal == "tight":
            direction = "neutral"  # 价差收窄时市场流动性好，适合执行
        elif efficiency_signal == "wide":
            direction = "neutral"  # 价差过宽时等待更好时机
        else:
            direction = "neutral"
        
        confidence = min(1.0, len(history) / 15)  # 基于数据量的置信度
        
        return MicrostructureSignal(
            signal_type="spread_analysis",
            strength=signal_strength,
            confidence=confidence,
            direction=direction,
            timeframe_seconds=20,
            reasoning=reasoning
        )
    
    async def determine_execution_timing(
        self,
        symbol: str,
        side: str,
        quantity: float,
        signals: List[MicrostructureSignal],
        current_market_data: Optional[Dict] = None
    ) -> ExecutionTiming:
        """
        基于微观结构信号确定最优执行时机
        """
        logger.debug(f"确定执行时机: {symbol} {side} {quantity}")
        
        # 综合分析所有信号
        signal_scores = {}
        total_confidence = 0
        
        for signal in signals:
            weight = signal.confidence
            if signal.direction == side.lower():
                signal_scores[signal.signal_type] = signal.strength * weight
            elif signal.direction == "neutral":
                signal_scores[signal.signal_type] = 0.5 * weight
            else:
                signal_scores[signal.signal_type] = (1 - signal.strength) * weight
            
            total_confidence += weight
        
        # 计算综合得分
        if total_confidence > 0:
            overall_score = sum(signal_scores.values()) / total_confidence
        else:
            overall_score = 0.5
        
        # 获取当前市场状态
        order_book = self._get_latest_order_book(symbol)
        
        # 估算执行成本
        estimated_slippage = await self._estimate_execution_slippage(
            symbol, side, quantity, order_book
        )
        
        # 基于信号强度和市场状态决定策略
        if overall_score > 0.7:  # 强烈信号，立即执行
            optimal_strategy = ExecutionStrategy.AGGRESSIVE_TAKER
            action = "execute_now"
            confidence_score = overall_score
            
        elif overall_score > 0.5:  # 中等信号，尝试被动执行
            optimal_strategy = ExecutionStrategy.PASSIVE_MAKER
            action = "execute_now"
            confidence_score = overall_score
            
        elif overall_score > 0.3:  # 弱信号，等待更好时机
            optimal_strategy = ExecutionStrategy.ADAPTIVE
            action = "wait"
            confidence_score = overall_score
            wait_time = 30  # 等待30秒
            
        else:  # 非常弱的信号，考虑取消
            optimal_strategy = ExecutionStrategy.STEALTH
            action = "wait"
            confidence_score = overall_score
            wait_time = 60  # 等待更长时间
        
        # 计算潜在价格改善
        price_improvement = None
        if action == "wait":
            # 基于信号预测价格改善可能性
            improvement_signals = [s for s in signals if s.direction == side.lower()]
            if improvement_signals:
                avg_strength = np.mean([s.strength for s in improvement_signals])
                price_improvement = avg_strength * 5  # 预期改善基点
        
        return ExecutionTiming(
            action=action,
            optimal_strategy=optimal_strategy,
            expected_slippage_bps=estimated_slippage,
            confidence_score=confidence_score,
            wait_time_seconds=wait_time if action == "wait" else None,
            price_improvement_bps=price_improvement
        )
    
    async def _estimate_execution_slippage(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        order_book: Optional[OrderBookSnapshot]
    ) -> float:
        """估算执行滑点"""
        
        if not order_book:
            return 10.0  # 默认10个基点
        
        # 计算订单簿冲击
        if side.upper() == 'BUY':
            levels = order_book.asks
        else:
            levels = order_book.bids
        
        remaining_qty = quantity
        total_cost = 0
        total_qty = 0
        
        for level in levels:
            if remaining_qty <= 0:
                break
                
            level_qty = min(level.quantity, remaining_qty)
            total_cost += level_qty * level.price
            total_qty += level_qty
            remaining_qty -= level_qty
        
        if total_qty > 0:
            avg_price = total_cost / total_qty
            mid_price = order_book.mid_price
            if mid_price > 0:
                slippage_bps = abs(avg_price - mid_price) / mid_price * 10000
                return slippage_bps
        
        return 15.0  # 如果无法计算，返回保守估计
    
    def _get_latest_order_book(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """获取最新订单簿数据"""
        history = self.order_book_history.get(symbol, deque())
        return history[-1] if history else None
    
    async def optimize_passive_order_placement(
        self,
        symbol: str,
        side: str,
        quantity: float,
        signals: List[MicrostructureSignal]
    ) -> Dict:
        """
        优化被动订单放置位置
        """
        order_book = self._get_latest_order_book(symbol)
        if not order_book:
            return {"error": "无订单簿数据"}
        
        # 分析价格改善机会
        if side.upper() == 'BUY':
            best_price = order_book.best_bid.price if order_book.best_bid else 0
            competitor_levels = order_book.bids[:5]
        else:
            best_price = order_book.best_ask.price if order_book.best_ask else 0
            competitor_levels = order_book.asks[:5]
        
        # 基于信号调整报价策略
        depth_signals = [s for s in signals if s.signal_type == "depth_analysis"]
        flow_signals = [s for s in signals if s.signal_type == "flow_analysis"]
        
        price_adjustment = 0
        if depth_signals:
            # 如果深度有利，可以更激进地报价
            depth_signal = depth_signals[0]
            if depth_signal.direction == side.lower():
                price_adjustment = 1  # 向更好价格移动1个最小价位
        
        if flow_signals:
            # 如果有价格动量，调整报价策略
            flow_signal = flow_signals[0]
            if flow_signal.direction == side.lower():
                price_adjustment += 1
        
        # 计算建议价格
        tick_size = self._get_tick_size(symbol)
        if side.upper() == 'BUY':
            suggested_price = best_price + price_adjustment * tick_size
        else:
            suggested_price = best_price - price_adjustment * tick_size
        
        # 估算成交概率
        fill_probability = self._estimate_fill_probability(
            symbol, side, quantity, suggested_price, order_book
        )
        
        return {
            "suggested_price": suggested_price,
            "price_adjustment_ticks": price_adjustment,
            "fill_probability": fill_probability,
            "timeout_seconds": self.passive_timeout_seconds,
            "reasoning": f"基于{len(signals)}个微观结构信号的价格优化"
        }
    
    def _get_tick_size(self, symbol: str) -> float:
        """获取交易对的最小价位"""
        # 简化实现，实际应该从交易所API获取
        return 0.01
    
    def _estimate_fill_probability(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_book: OrderBookSnapshot
    ) -> float:
        """估算订单成交概率"""
        
        # 简化的成交概率模型
        if side.upper() == 'BUY':
            if order_book.best_ask and price >= order_book.best_ask.price:
                return 0.95  # 价格优于市场，高概率成交
            elif order_book.best_bid and price >= order_book.best_bid.price:
                return 0.7   # 价格在买一价或更好
            else:
                return 0.3   # 价格低于买一价
        else:
            if order_book.best_bid and price <= order_book.best_bid.price:
                return 0.95  # 价格优于市场
            elif order_book.best_ask and price <= order_book.best_ask.price:
                return 0.7   # 价格在卖一价或更好
            else:
                return 0.3   # 价格高于卖一价
    
    async def adapt_execution_strategy(
        self,
        current_strategy: ExecutionStrategy,
        execution_results: List[Dict],
        market_conditions: Dict
    ) -> ExecutionStrategy:
        """
        基于执行结果自适应调整策略
        """
        # 分析当前策略表现
        if len(execution_results) < 5:
            return current_strategy
        
        recent_results = execution_results[-10:]
        
        # 计算关键指标
        avg_slippage = np.mean([r.get('slippage_bps', 0) for r in recent_results])
        fill_rate = np.mean([r.get('fill_rate', 0) for r in recent_results])
        avg_execution_time = np.mean([r.get('execution_time_seconds', 0) for r in recent_results])
        
        # 更新策略成功率
        if current_strategy not in self.strategy_success_rate:
            self.strategy_success_rate[current_strategy] = 0.5
        
        # 基于表现调整成功率
        performance_score = 0
        if avg_slippage < 5:  # 低滑点
            performance_score += 0.3
        if fill_rate > 0.9:   # 高成交率
            performance_score += 0.3
        if avg_execution_time < 60:  # 快速执行
            performance_score += 0.2
        
        # 指数移动平均更新成功率
        alpha = 0.2
        self.strategy_success_rate[current_strategy] = (
            alpha * performance_score + 
            (1 - alpha) * self.strategy_success_rate[current_strategy]
        )
        
        # 基于市场条件选择最优策略
        volatility = market_conditions.get('volatility', 0.02)
        liquidity = market_conditions.get('liquidity', 1.0)
        
        if volatility > 0.05:  # 高波动
            return ExecutionStrategy.AGGRESSIVE_TAKER
        elif liquidity < 0.5:  # 低流动性
            return ExecutionStrategy.STEALTH
        elif self.strategy_success_rate.get(ExecutionStrategy.PASSIVE_MAKER, 0) > 0.7:
            return ExecutionStrategy.PASSIVE_MAKER
        else:
            return ExecutionStrategy.ADAPTIVE
    
    def get_microstructure_summary(self, symbol: str) -> Dict:
        """获取微观结构分析摘要"""
        
        order_book_hist = list(self.order_book_history.get(symbol, []))
        if not order_book_hist:
            return {"error": "无历史数据"}
        
        latest_ob = order_book_hist[-1]
        
        # 计算关键指标
        spreads = [ob.spread / ob.mid_price * 10000 for ob in order_book_hist[-20:] if ob.mid_price > 0]
        avg_spread_bps = np.mean(spreads) if spreads else 0
        
        depths = []
        for ob in order_book_hist[-10:]:
            bid_depth = sum(level.quantity for level in ob.bids[:5])
            ask_depth = sum(level.quantity for level in ob.asks[:5])
            depths.append(bid_depth + ask_depth)
        
        avg_depth = np.mean(depths) if depths else 0
        
        return {
            "symbol": symbol,
            "timestamp": latest_ob.timestamp.isoformat(),
            "current_spread_bps": latest_ob.spread / latest_ob.mid_price * 10000 if latest_ob.mid_price > 0 else 0,
            "avg_spread_bps": avg_spread_bps,
            "current_depth": sum(level.quantity for level in latest_ob.bids[:5] + latest_ob.asks[:5]),
            "avg_depth": avg_depth,
            "mid_price": latest_ob.mid_price,
            "data_points": len(order_book_hist),
            "strategy_performance": dict(self.strategy_success_rate)
        }