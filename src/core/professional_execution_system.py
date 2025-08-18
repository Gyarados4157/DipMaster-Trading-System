"""
Professional Execution Management System (EMS)
DipMaster Trading System - 专业级执行管理系统

核心功能:
1. 持续执行优化 - 智能订单分割和时间调度
2. 微观结构优化 - 实时流动性分析和市场影响最小化
3. 多交易所路由 - 最优价格发现和流动性聚合
4. 执行风险管理 - 实时监控和紧急平仓机制
5. DipMaster专用优化 - 15分钟边界和逢跌买入执行
6. 持续运行调度 - 自动化执行和质量监控

性能目标:
- 执行滑点: <5基点
- 订单完成率: >99%
- 执行延迟: <2秒
- 成本节约: >20基点
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
from collections import deque, defaultdict
import uuid

logger = logging.getLogger(__name__)


class ExecutionAlgorithm(Enum):
    """执行算法类型"""
    TWAP = "twap"
    VWAP = "vwap" 
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ICEBERG = "iceberg"
    MARKET_ON_CLOSE = "market_on_close"
    DIPMASTER_15MIN = "dipmaster_15min"  # DipMaster专用15分钟边界
    DIPMASTER_DIP_BUY = "dipmaster_dip_buy"  # 逢跌买入专用
    POV = "participation_of_volume"  # 成交量参与
    
    
class OrderUrgency(Enum):
    """订单紧急程度"""
    LOW = 0.2      # 低紧急度：成本优先
    MEDIUM = 0.5   # 中等紧急度：平衡成本和速度
    HIGH = 0.8     # 高紧急度：速度优先
    EMERGENCY = 0.95  # 紧急：立即执行


@dataclass
class TargetPosition:
    """目标仓位"""
    symbol: str
    target_size_usd: float
    current_size_usd: float = 0.0
    urgency: OrderUrgency = OrderUrgency.MEDIUM
    max_execution_time_minutes: int = 60
    side: str = "BUY"  # BUY or SELL
    algorithm_preference: Optional[ExecutionAlgorithm] = None
    constraints: Dict = field(default_factory=dict)
    dipmaster_context: Optional[Dict] = None  # DipMaster特定上下文


@dataclass  
class MarketMicrostructure:
    """市场微结构数据"""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    mid_price: float
    spread_bps: float
    bid_size: float
    ask_size: float
    depth_5_bid: float
    depth_5_ask: float
    volume_1min: float
    volume_5min: float
    volatility_1min_bps: float
    price_impact_coefficient: float
    liquidity_score: float
    order_flow_imbalance: float
    venue: str = "binance"
    

@dataclass
class ExecutionSlice:
    """执行切片"""
    slice_id: str
    parent_order_id: str
    symbol: str
    side: str
    quantity: float
    size_usd: float
    algorithm: ExecutionAlgorithm
    order_type: str = "LIMIT"
    limit_price: Optional[float] = None
    scheduled_time: datetime = field(default_factory=datetime.now)
    urgency_score: float = 0.5
    max_participation_rate: float = 0.2
    timeout_seconds: int = 300
    venue: str = "binance"
    
    # 执行状态
    status: str = "pending"  # pending, executing, filled, partial, failed, cancelled
    execution_start_time: Optional[datetime] = None
    execution_end_time: Optional[datetime] = None
    filled_quantity: float = 0.0
    filled_size_usd: float = 0.0
    avg_fill_price: float = 0.0
    total_fees: float = 0.0
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    
    # DipMaster专用字段
    dipmaster_timing_window: Optional[str] = None  # "15-30min", "45-60min"
    dipmaster_dip_detected: bool = False
    dipmaster_rsi_signal: Optional[float] = None


@dataclass
class ExecutionFill:
    """成交记录"""
    fill_id: str
    slice_id: str  
    parent_order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    venue: str
    fees_usd: float
    slippage_bps: float
    market_impact_bps: float
    latency_ms: float
    liquidity_type: str  # maker, taker
    arrival_price: float
    implementation_shortfall_bps: float = 0.0


@dataclass
class ExecutionReport:
    """完整执行报告"""
    session_id: str
    symbol: str
    target_position: TargetPosition
    execution_algorithm: str
    execution_start: datetime
    execution_end: datetime
    
    # 执行统计
    total_slices: int
    successful_slices: int
    failed_slices: int
    cancelled_slices: int
    fill_rate: float
    
    # 数量和金额
    target_quantity: float
    executed_quantity: float
    target_size_usd: float
    executed_size_usd: float
    completion_rate: float
    
    # 成本分析
    total_fees_usd: float
    total_market_impact_bps: float
    avg_slippage_bps: float
    implementation_shortfall_bps: float
    total_cost_usd: float
    cost_per_share_bps: float
    
    # 执行质量指标
    avg_latency_ms: float
    maker_ratio: float
    venue_distribution: Dict[str, float]
    time_distribution: Dict[str, int]
    
    # DipMaster专用指标
    dipmaster_metrics: Dict = field(default_factory=dict)
    
    # 风险和违规
    risk_violations: List[str] = field(default_factory=list)
    performance_alerts: List[str] = field(default_factory=list)
    
    # 所有成交记录
    fills: List[ExecutionFill] = field(default_factory=list)


class MarketDataProvider:
    """市场数据提供者"""
    
    def __init__(self):
        self.market_cache = {}
        self.last_update = {}
        self.base_prices = {
            'BTCUSDT': 65000.0,
            'ETHUSDT': 3200.0,
            'SOLUSDT': 140.0,
            'BNBUSDT': 580.0,
            'ADAUSDT': 0.45,
            'DOGEUSDT': 0.08,
            'XRPUSDT': 0.55
        }
        
    def get_microstructure(self, symbol: str, venue: str = "binance") -> MarketMicrostructure:
        """获取市场微结构数据"""
        # 模拟实时市场数据
        base_price = self.base_prices.get(symbol, 50000.0)
        
        # 添加价格波动
        price_change = np.random.normal(0, base_price * 0.001)  # 0.1%波动
        mid_price = base_price + price_change
        
        # 生成买卖价差
        spread_bps = np.random.uniform(5, 25)  # 5-25 bps
        spread = mid_price * spread_bps / 10000
        
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2
        
        # 订单簿深度
        bid_size = np.random.exponential(50) + 10
        ask_size = np.random.exponential(50) + 10
        depth_5_bid = bid_size * np.random.uniform(3, 8)
        depth_5_ask = ask_size * np.random.uniform(3, 8)
        
        # 成交量和流动性指标
        volume_1min = np.random.exponential(30) + 5
        volume_5min = volume_1min * 5 * np.random.uniform(0.8, 1.2)
        volatility_1min_bps = np.random.uniform(10, 200)  # 1分钟波动率
        
        # 价格影响系数和流动性评分
        price_impact_coeff = np.random.uniform(0.1, 0.8)
        liquidity_score = np.random.uniform(0.6, 0.98)
        order_flow_imbalance = np.random.uniform(-0.5, 0.5)
        
        return MarketMicrostructure(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=bid_price,
            ask_price=ask_price,
            mid_price=mid_price,
            spread_bps=spread_bps,
            bid_size=bid_size,
            ask_size=ask_size,
            depth_5_bid=depth_5_bid,
            depth_5_ask=depth_5_ask,
            volume_1min=volume_1min,
            volume_5min=volume_5min,
            volatility_1min_bps=volatility_1min_bps,
            price_impact_coefficient=price_impact_coeff,
            liquidity_score=liquidity_score,
            order_flow_imbalance=order_flow_imbalance,
            venue=venue
        )


class LiquidityAnalyzer:
    """流动性分析器"""
    
    def __init__(self, market_data: MarketDataProvider):
        self.market_data = market_data
        
    def analyze_execution_feasibility(self, target: TargetPosition) -> Dict[str, Any]:
        """分析执行可行性"""
        microstructure = self.market_data.get_microstructure(target.symbol)
        
        # 计算参与率
        daily_volume_estimate = microstructure.volume_5min * 12 * 24  # 5分钟->日成交量
        participation_rate = abs(target.target_size_usd - target.current_size_usd) / (daily_volume_estimate * microstructure.mid_price)
        
        # 估算市场影响
        market_impact_bps = self._estimate_market_impact(
            abs(target.target_size_usd - target.current_size_usd),
            microstructure
        )
        
        # 推荐执行算法
        recommended_algo = self._recommend_algorithm(target, microstructure, market_impact_bps)
        
        # 建议执行时间
        suggested_duration_min = self._suggest_duration(target, microstructure, market_impact_bps)
        
        return {
            'feasible': market_impact_bps < 50,  # 50bps以下认为可行
            'participation_rate': participation_rate,
            'estimated_impact_bps': market_impact_bps,
            'liquidity_score': microstructure.liquidity_score,
            'recommended_algorithm': recommended_algo,
            'suggested_duration_min': suggested_duration_min,
            'risk_level': self._assess_risk_level(market_impact_bps, microstructure),
            'microstructure': microstructure
        }
    
    def _estimate_market_impact(self, size_usd: float, micro: MarketMicrostructure) -> float:
        """估算市场影响（bps）"""
        # 基于Square-root law和流动性
        size_ratio = size_usd / (micro.volume_5min * micro.mid_price)
        base_impact = micro.price_impact_coefficient * np.sqrt(size_ratio) * 100
        
        # 流动性调整
        liquidity_penalty = (1 - micro.liquidity_score) * 20  # 流动性越差惩罚越大
        
        # 波动率调整
        volatility_penalty = micro.volatility_1min_bps / 100 * 5  # 波动率越高影响越大
        
        total_impact = base_impact + liquidity_penalty + volatility_penalty
        return min(total_impact, 100)  # 上限100bps
    
    def _recommend_algorithm(self, target: TargetPosition, micro: MarketMicrostructure, impact_bps: float) -> ExecutionAlgorithm:
        """推荐执行算法"""
        size_usd = abs(target.target_size_usd - target.current_size_usd)
        
        # DipMaster特定算法
        if target.dipmaster_context:
            if target.dipmaster_context.get('timing_critical'):
                return ExecutionAlgorithm.DIPMASTER_15MIN
            elif target.dipmaster_context.get('dip_signal'):
                return ExecutionAlgorithm.DIPMASTER_DIP_BUY
        
        # 基于紧急程度和市场影响选择
        if target.urgency.value >= 0.9:
            return ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL
        elif size_usd > 10000 and impact_bps > 15:
            return ExecutionAlgorithm.VWAP
        elif size_usd > 5000:
            return ExecutionAlgorithm.TWAP
        elif micro.liquidity_score > 0.85 and impact_bps < 10:
            return ExecutionAlgorithm.POV
        else:
            return ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL
    
    def _suggest_duration(self, target: TargetPosition, micro: MarketMicrostructure, impact_bps: float) -> int:
        """建议执行时间（分钟）"""
        size_usd = abs(target.target_size_usd - target.current_size_usd)
        
        # DipMaster时间约束
        if target.dipmaster_context and target.dipmaster_context.get('timing_critical'):
            return min(15, target.max_execution_time_minutes)  # DipMaster15分钟边界
        
        # 基于大小和影响的标准计算
        base_duration = size_usd / 1000 * 5  # 每1000USD需要5分钟
        
        # 市场影响调整
        if impact_bps > 20:
            base_duration *= 1.5
        elif impact_bps < 10:
            base_duration *= 0.7
            
        # 流动性调整
        base_duration *= (2 - micro.liquidity_score)  # 流动性越差时间越长
        
        # 紧急程度调整
        base_duration *= (2 - target.urgency.value)  # 越紧急时间越短
        
        return max(5, min(int(base_duration), target.max_execution_time_minutes))
    
    def _assess_risk_level(self, impact_bps: float, micro: MarketMicrostructure) -> str:
        """评估风险等级"""
        if impact_bps > 30 or micro.liquidity_score < 0.6:
            return "HIGH"
        elif impact_bps > 15 or micro.liquidity_score < 0.8:
            return "MEDIUM"  
        else:
            return "LOW"


class SmartOrderSlicer:
    """智能订单分割器"""
    
    def __init__(self, market_data: MarketDataProvider):
        self.market_data = market_data
        
    def slice_order(self, target: TargetPosition, algorithm: ExecutionAlgorithm, duration_min: int) -> List[ExecutionSlice]:
        """智能订单分割"""
        size_delta = target.target_size_usd - target.current_size_usd
        side = "BUY" if size_delta > 0 else "SELL"
        size_usd = abs(size_delta)
        
        logger.info(f"订单分割: {algorithm.value} {side} {size_usd:.0f}USD, {duration_min}分钟")
        
        if algorithm == ExecutionAlgorithm.TWAP:
            return self._slice_twap(target, size_usd, side, duration_min)
        elif algorithm == ExecutionAlgorithm.VWAP:
            return self._slice_vwap(target, size_usd, side, duration_min)
        elif algorithm == ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL:
            return self._slice_implementation_shortfall(target, size_usd, side, duration_min)
        elif algorithm == ExecutionAlgorithm.DIPMASTER_15MIN:
            return self._slice_dipmaster_15min(target, size_usd, side)
        elif algorithm == ExecutionAlgorithm.DIPMASTER_DIP_BUY:
            return self._slice_dipmaster_dip_buy(target, size_usd, side)
        elif algorithm == ExecutionAlgorithm.POV:
            return self._slice_pov(target, size_usd, side, duration_min)
        else:
            return self._slice_simple(target, size_usd, side, duration_min)
    
    def _slice_twap(self, target: TargetPosition, size_usd: float, side: str, duration_min: int) -> List[ExecutionSlice]:
        """TWAP分割 - 时间加权平均价格"""
        microstructure = self.market_data.get_microstructure(target.symbol)
        quantity = size_usd / microstructure.mid_price
        
        # 分割参数
        num_slices = max(3, min(20, duration_min // 3))  # 每3分钟一片
        slice_interval_sec = duration_min * 60 / num_slices
        base_quantity = quantity / num_slices
        
        slices = []
        parent_id = f"TWAP_{target.symbol}_{int(time.time())}"
        
        for i in range(num_slices):
            # 添加随机化避免被检测
            randomization = np.random.uniform(0.85, 1.15)
            slice_qty = base_quantity * randomization
            
            # 最后一片处理剩余
            if i == num_slices - 1:
                slice_qty = quantity - sum(s.quantity for s in slices)
            
            scheduled_time = datetime.now() + timedelta(seconds=i * slice_interval_sec)
            
            slice_obj = ExecutionSlice(
                slice_id=f"{parent_id}_T{i+1:02d}",
                parent_order_id=parent_id,
                symbol=target.symbol,
                side=side,
                quantity=slice_qty,
                size_usd=slice_qty * microstructure.mid_price,
                algorithm=ExecutionAlgorithm.TWAP,
                order_type="LIMIT",
                scheduled_time=scheduled_time,
                urgency_score=0.3,
                max_participation_rate=0.15
            )
            slices.append(slice_obj)
        
        return slices
    
    def _slice_vwap(self, target: TargetPosition, size_usd: float, side: str, duration_min: int) -> List[ExecutionSlice]:
        """VWAP分割 - 成交量加权平均价格"""
        microstructure = self.market_data.get_microstructure(target.symbol)
        quantity = size_usd / microstructure.mid_price
        
        # 获取历史成交量分布权重
        volume_weights = self._get_intraday_volume_pattern(duration_min)
        num_slices = len(volume_weights)
        slice_interval_sec = duration_min * 60 / num_slices
        
        slices = []
        parent_id = f"VWAP_{target.symbol}_{int(time.time())}"
        
        for i, weight in enumerate(volume_weights):
            slice_qty = quantity * weight
            scheduled_time = datetime.now() + timedelta(seconds=i * slice_interval_sec)
            
            slice_obj = ExecutionSlice(
                slice_id=f"{parent_id}_V{i+1:02d}",
                parent_order_id=parent_id,
                symbol=target.symbol,
                side=side,
                quantity=slice_qty,
                size_usd=slice_qty * microstructure.mid_price,
                algorithm=ExecutionAlgorithm.VWAP,
                order_type="LIMIT",
                scheduled_time=scheduled_time,
                urgency_score=0.4,
                max_participation_rate=0.25
            )
            slices.append(slice_obj)
            
        return slices
    
    def _slice_implementation_shortfall(self, target: TargetPosition, size_usd: float, side: str, duration_min: int) -> List[ExecutionSlice]:
        """Implementation Shortfall分割"""
        microstructure = self.market_data.get_microstructure(target.symbol)
        quantity = size_usd / microstructure.mid_price
        urgency = target.urgency.value
        
        # 基于紧急程度调整执行节奏
        if urgency > 0.8:
            # 高紧急：前重后轻
            num_slices = max(2, int(size_usd / 2000))
            weights = [np.exp(-i * 0.3) for i in range(num_slices)]
        elif urgency < 0.3:
            # 低紧急：均匀分布
            num_slices = max(5, int(size_usd / 1000))
            weights = [1.0] * num_slices
        else:
            # 中等紧急：轻微前倾
            num_slices = max(3, int(size_usd / 1500))
            weights = [1.2 - i * 0.1 for i in range(num_slices)]
        
        # 标准化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        slice_interval_sec = duration_min * 60 / num_slices
        slices = []
        parent_id = f"IS_{target.symbol}_{int(time.time())}"
        
        for i, weight in enumerate(weights):
            slice_qty = quantity * weight
            scheduled_time = datetime.now() + timedelta(seconds=i * slice_interval_sec)
            
            # 紧急订单用市价单
            order_type = "MARKET" if urgency > 0.8 else "LIMIT"
            
            slice_obj = ExecutionSlice(
                slice_id=f"{parent_id}_IS{i+1:02d}",
                parent_order_id=parent_id,
                symbol=target.symbol,
                side=side,
                quantity=slice_qty,
                size_usd=slice_qty * microstructure.mid_price,
                algorithm=ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL,
                order_type=order_type,
                scheduled_time=scheduled_time,
                urgency_score=urgency,
                max_participation_rate=0.3 if urgency > 0.7 else 0.2
            )
            slices.append(slice_obj)
            
        return slices
    
    def _slice_dipmaster_15min(self, target: TargetPosition, size_usd: float, side: str) -> List[ExecutionSlice]:
        """DipMaster 15分钟边界执行"""
        microstructure = self.market_data.get_microstructure(target.symbol)
        quantity = size_usd / microstructure.mid_price
        
        current_minute = datetime.now().minute
        
        # 计算到下个15分钟边界的时间
        if current_minute < 15:
            target_minute = 15
        elif current_minute < 30:
            target_minute = 30
        elif current_minute < 45:
            target_minute = 45
        else:
            target_minute = 60  # 下一小时的0分
            
        minutes_to_boundary = (target_minute - current_minute) % 60
        execution_time = datetime.now() + timedelta(minutes=minutes_to_boundary - 2)  # 提前2分钟开始
        
        # 分成2-3片在边界前执行完成
        num_slices = min(3, max(2, int(size_usd / 2000)))
        slice_interval_sec = 120 / num_slices  # 2分钟内完成
        
        slices = []
        parent_id = f"DM15_{target.symbol}_{int(time.time())}"
        
        for i in range(num_slices):
            slice_qty = quantity / num_slices
            scheduled_time = execution_time + timedelta(seconds=i * slice_interval_sec)
            
            slice_obj = ExecutionSlice(
                slice_id=f"{parent_id}_DM{i+1:02d}",
                parent_order_id=parent_id,
                symbol=target.symbol,
                side=side,
                quantity=slice_qty,
                size_usd=slice_qty * microstructure.mid_price,
                algorithm=ExecutionAlgorithm.DIPMASTER_15MIN,
                order_type="MARKET",  # 15分钟边界需要快速执行
                scheduled_time=scheduled_time,
                urgency_score=0.9,
                max_participation_rate=0.3,
                dipmaster_timing_window=f"{current_minute}-{target_minute}min"
            )
            slices.append(slice_obj)
            
        return slices
    
    def _slice_dipmaster_dip_buy(self, target: TargetPosition, size_usd: float, side: str) -> List[ExecutionSlice]:
        """DipMaster逢跌买入执行"""
        microstructure = self.market_data.get_microstructure(target.symbol)
        quantity = size_usd / microstructure.mid_price
        
        # 逢跌买入：快速执行捕捉下跌机会
        num_slices = max(2, min(4, int(size_usd / 1000)))
        slice_interval_sec = 30  # 30秒间隔快速执行
        
        slices = []
        parent_id = f"DIP_{target.symbol}_{int(time.time())}"
        
        for i in range(num_slices):
            slice_qty = quantity / num_slices
            scheduled_time = datetime.now() + timedelta(seconds=i * slice_interval_sec)
            
            # 设置略低于市价的限价以确保成交
            limit_price = microstructure.mid_price * 0.9995 if side == "BUY" else microstructure.mid_price * 1.0005
            
            slice_obj = ExecutionSlice(
                slice_id=f"{parent_id}_DIP{i+1:02d}",
                parent_order_id=parent_id,
                symbol=target.symbol,
                side=side,
                quantity=slice_qty,
                size_usd=slice_qty * microstructure.mid_price,
                algorithm=ExecutionAlgorithm.DIPMASTER_DIP_BUY,
                order_type="LIMIT",
                limit_price=limit_price,
                scheduled_time=scheduled_time,
                urgency_score=0.8,
                max_participation_rate=0.4,
                dipmaster_dip_detected=True,
                dipmaster_rsi_signal=target.dipmaster_context.get('rsi_value') if target.dipmaster_context else None
            )
            slices.append(slice_obj)
            
        return slices
    
    def _slice_pov(self, target: TargetPosition, size_usd: float, side: str, duration_min: int) -> List[ExecutionSlice]:
        """POV (Participation of Volume) 分割"""
        microstructure = self.market_data.get_microstructure(target.symbol)
        quantity = size_usd / microstructure.mid_price
        
        # 基于成交量参与率动态分割
        target_participation = min(0.2, target.urgency.value * 0.3)  # 最大20%参与率
        estimated_volume_per_slice = microstructure.volume_1min * target_participation
        
        num_slices = max(3, int(quantity / estimated_volume_per_slice))
        slice_interval_sec = duration_min * 60 / num_slices
        
        slices = []
        parent_id = f"POV_{target.symbol}_{int(time.time())}"
        
        for i in range(num_slices):
            slice_qty = quantity / num_slices
            scheduled_time = datetime.now() + timedelta(seconds=i * slice_interval_sec)
            
            slice_obj = ExecutionSlice(
                slice_id=f"{parent_id}_POV{i+1:02d}",
                parent_order_id=parent_id,
                symbol=target.symbol,
                side=side,
                quantity=slice_qty,
                size_usd=slice_qty * microstructure.mid_price,
                algorithm=ExecutionAlgorithm.POV,
                order_type="LIMIT",
                scheduled_time=scheduled_time,
                urgency_score=target.urgency.value,
                max_participation_rate=target_participation
            )
            slices.append(slice_obj)
            
        return slices
    
    def _slice_simple(self, target: TargetPosition, size_usd: float, side: str, duration_min: int) -> List[ExecutionSlice]:
        """简单分割策略"""
        microstructure = self.market_data.get_microstructure(target.symbol)
        quantity = size_usd / microstructure.mid_price
        
        num_slices = max(2, min(10, int(size_usd / 1000)))
        slice_interval_sec = duration_min * 60 / num_slices
        
        slices = []
        parent_id = f"SIMPLE_{target.symbol}_{int(time.time())}"
        
        for i in range(num_slices):
            slice_qty = quantity / num_slices
            scheduled_time = datetime.now() + timedelta(seconds=i * slice_interval_sec)
            
            slice_obj = ExecutionSlice(
                slice_id=f"{parent_id}_S{i+1:02d}",
                parent_order_id=parent_id,
                symbol=target.symbol,
                side=side,
                quantity=slice_qty,
                size_usd=slice_qty * microstructure.mid_price,
                algorithm=ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL,
                order_type="LIMIT",
                scheduled_time=scheduled_time,
                urgency_score=target.urgency.value,
                max_participation_rate=0.2
            )
            slices.append(slice_obj)
            
        return slices
    
    def _get_intraday_volume_pattern(self, duration_min: int) -> List[float]:
        """获取日内成交量分布模式"""
        current_hour = datetime.now().hour
        
        # 24小时成交量权重（加密货币市场模式）
        hourly_volume = [
            0.5, 0.4, 0.3, 0.3, 0.4, 0.5,   # 00-05: 低活跃
            0.7, 0.9, 1.2, 1.5, 1.7, 1.8,   # 06-11: 上升趋势
            2.0, 2.2, 2.0, 1.8, 1.5, 1.2,   # 12-17: 高峰时段
            1.0, 0.8, 0.6, 0.5, 0.4, 0.3    # 18-23: 下降趋势
        ]
        
        # 生成指定时长的权重分布
        num_intervals = max(3, duration_min // 15)  # 15分钟一个区间
        weights = []
        
        for i in range(num_intervals):
            hour_index = (current_hour + i // 4) % 24
            weight = hourly_volume[hour_index]
            # 添加随机扰动
            weight *= np.random.uniform(0.8, 1.2)
            weights.append(weight)
        
        # 归一化权重
        total_weight = sum(weights)
        return [w / total_weight for w in weights] if total_weight > 0 else [1.0 / len(weights)] * len(weights)


class ExecutionRiskManager:
    """执行风险管理器"""
    
    def __init__(self):
        # 风险限制参数
        self.max_slippage_bps = 50
        self.max_market_impact_bps = 30  
        self.max_single_order_usd = 20000
        self.max_participation_rate = 0.35
        self.max_execution_hours = 6
        
        # 风险监控
        self.violation_history = deque(maxlen=100)
        self.circuit_breaker_active = False
        self.risk_scores = defaultdict(float)
        
    def pre_execution_risk_check(self, slice_obj: ExecutionSlice, microstructure: MarketMicrostructure) -> Tuple[bool, List[str]]:
        """执行前风险检查"""
        violations = []
        
        # 检查订单大小
        if slice_obj.size_usd > self.max_single_order_usd:
            violations.append(f"单笔订单过大: ${slice_obj.size_usd:.0f} > ${self.max_single_order_usd}")
        
        # 检查参与率  
        estimated_volume_usd = microstructure.volume_5min * microstructure.mid_price
        if estimated_volume_usd > 0:
            participation_rate = slice_obj.size_usd / estimated_volume_usd
            if participation_rate > self.max_participation_rate:
                violations.append(f"参与率过高: {participation_rate:.2%} > {self.max_participation_rate:.2%}")
        
        # 检查流动性
        if microstructure.liquidity_score < 0.3:
            violations.append(f"流动性不足: {microstructure.liquidity_score:.2f} < 0.30")
            
        # 检查价差
        if microstructure.spread_bps > 100:
            violations.append(f"价差过大: {microstructure.spread_bps:.1f}bps > 100bps")
            
        # 检查熔断器状态
        if self.circuit_breaker_active:
            violations.append("风险熔断器已激活，暂停执行")
            
        can_execute = len(violations) == 0
        return can_execute, violations
    
    def post_execution_risk_check(self, fill: ExecutionFill) -> List[str]:
        """执行后风险检查"""
        violations = []
        
        # 检查滑点
        if abs(fill.slippage_bps) > self.max_slippage_bps:
            violations.append(f"滑点超限: {fill.slippage_bps:.1f}bps > {self.max_slippage_bps}bps")
            
        # 检查市场影响
        if fill.market_impact_bps > self.max_market_impact_bps:
            violations.append(f"市场冲击超限: {fill.market_impact_bps:.1f}bps > {self.max_market_impact_bps}bps")
            
        # 记录违规
        if violations:
            self.violation_history.extend(violations)
            
            # 检查是否需要触发熔断
            recent_violations = len([v for v in self.violation_history if time.time() - getattr(v, 'timestamp', 0) < 300])
            if recent_violations >= 5:  # 5分钟内5次违规
                self.circuit_breaker_active = True
                violations.append("触发风险熔断器：近期违规过多")
                
        return violations
    
    def update_risk_score(self, symbol: str, execution_quality: Dict[str, float]):
        """更新风险评分"""
        # 基于执行质量计算风险评分
        slippage_score = min(execution_quality.get('avg_slippage_bps', 0) / 10, 5)
        impact_score = min(execution_quality.get('market_impact_bps', 0) / 5, 5)  
        latency_score = min(execution_quality.get('avg_latency_ms', 0) / 100, 5)
        
        risk_score = (slippage_score + impact_score + latency_score) / 3
        self.risk_scores[symbol] = risk_score
        
    def reset_circuit_breaker(self):
        """重置熔断器"""
        self.circuit_breaker_active = False
        self.violation_history.clear()
        logger.info("执行风险熔断器已重置")
        
    def get_risk_status(self) -> Dict[str, Any]:
        """获取风险状态"""
        return {
            'circuit_breaker_active': self.circuit_breaker_active,
            'recent_violations': len(self.violation_history),
            'avg_risk_score': np.mean(list(self.risk_scores.values())) if self.risk_scores else 0,
            'high_risk_symbols': [k for k, v in self.risk_scores.items() if v > 3],
            'last_update': datetime.now()
        }


class ExecutionEngine:
    """核心执行引擎"""
    
    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        
        # 核心组件
        self.market_data = MarketDataProvider()
        self.liquidity_analyzer = LiquidityAnalyzer(self.market_data)
        self.order_slicer = SmartOrderSlicer(self.market_data)
        self.risk_manager = ExecutionRiskManager()
        
        # 执行状态
        self.active_sessions = {}
        self.execution_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(list)
        
        # 多线程执行器
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        
    async def execute_target_position(self, target: TargetPosition) -> ExecutionReport:
        """执行目标仓位"""
        session_id = f"EXEC_{target.symbol}_{uuid.uuid4().hex[:8]}"
        execution_start = datetime.now()
        
        logger.info(f"开始执行: {session_id} - {target.symbol} ${target.target_size_usd:.0f}")
        
        try:
            # 1. 可行性分析
            feasibility = self.liquidity_analyzer.analyze_execution_feasibility(target)
            
            if not feasibility['feasible']:
                raise ValueError(f"执行不可行: 预期市场冲击 {feasibility['estimated_impact_bps']:.1f}bps")
                
            logger.info(f"流动性分析完成: {feasibility['recommended_algorithm'].value}, {feasibility['suggested_duration_min']}分钟")
            
            # 2. 订单分割
            algorithm = feasibility['recommended_algorithm']
            duration_min = feasibility['suggested_duration_min']
            
            slices = self.order_slicer.slice_order(target, algorithm, duration_min)
            logger.info(f"订单分割完成: {len(slices)}个切片")
            
            # 3. 并发执行切片
            fills = await self._execute_slices_concurrent(slices)
            
            # 4. 生成执行报告
            report = await self._generate_comprehensive_report(
                session_id, target, algorithm, execution_start, slices, fills, feasibility
            )
            
            # 5. 更新性能指标
            self._update_performance_metrics(report)
            
            # 6. 存储执行历史
            self.execution_history.append(report)
            
            logger.info(f"执行完成: {session_id} - 成交率 {report.fill_rate:.1%}, 成本 {report.cost_per_share_bps:.2f}bps")
            return report
            
        except Exception as e:
            logger.error(f"执行失败: {session_id} - {e}")
            return self._create_error_report(session_id, target, execution_start, str(e))
    
    async def _execute_slices_concurrent(self, slices: List[ExecutionSlice]) -> List[ExecutionFill]:
        """并发执行订单切片"""
        fills = []
        semaphore = asyncio.Semaphore(3)  # 限制并发数
        
        async def execute_single_slice(slice_obj):
            async with semaphore:
                return await self._execute_single_slice(slice_obj)
        
        # 创建执行任务
        tasks = []
        for slice_obj in slices:
            # 等待调度时间
            if slice_obj.scheduled_time > datetime.now():
                wait_seconds = (slice_obj.scheduled_time - datetime.now()).total_seconds()
                if 0 < wait_seconds <= 300:  # 最多等待5分钟
                    await asyncio.sleep(wait_seconds)
            
            task = asyncio.create_task(execute_single_slice(slice_obj))
            tasks.append(task)
            
            # 添加小间隔避免同时执行
            await asyncio.sleep(0.1)
        
        # 等待所有执行完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, ExecutionFill):
                fills.append(result)
            elif isinstance(result, Exception):
                logger.error(f"切片执行异常: {result}")
                
        return fills
    
    async def _execute_single_slice(self, slice_obj: ExecutionSlice) -> Optional[ExecutionFill]:
        """执行单个订单切片"""
        execution_start_time = time.time()
        slice_obj.execution_start_time = datetime.now()
        slice_obj.status = "executing"
        
        try:
            # 获取最新市场数据
            microstructure = self.market_data.get_microstructure(slice_obj.symbol, slice_obj.venue)
            arrival_price = microstructure.mid_price
            
            # 执行前风险检查
            can_execute, violations = self.risk_manager.pre_execution_risk_check(slice_obj, microstructure)
            
            if not can_execute:
                logger.warning(f"切片 {slice_obj.slice_id} 被风控阻止: {violations}")
                slice_obj.status = "blocked"
                return None
            
            # 模拟执行（纸上交易）
            if self.paper_trading:
                fill_result = self._simulate_execution(slice_obj, microstructure)
            else:
                fill_result = await self._real_execution(slice_obj, microstructure)
            
            if fill_result:
                fill_price, fill_quantity, fees = fill_result
                
                # 计算指标
                slippage_bps = ((fill_price - arrival_price) / arrival_price) * 10000 if arrival_price > 0 else 0
                market_impact_bps = abs(slippage_bps) if abs(slippage_bps) > 5 else 0
                implementation_shortfall_bps = slippage_bps + (fees / (fill_quantity * fill_price) * 10000)
                latency_ms = (time.time() - execution_start_time) * 1000
                
                # 更新切片状态
                slice_obj.status = "filled"
                slice_obj.execution_end_time = datetime.now()
                slice_obj.filled_quantity = fill_quantity
                slice_obj.filled_size_usd = fill_quantity * fill_price
                slice_obj.avg_fill_price = fill_price
                slice_obj.total_fees = fees
                slice_obj.slippage_bps = slippage_bps
                slice_obj.market_impact_bps = market_impact_bps
                
                # 创建成交记录
                fill = ExecutionFill(
                    fill_id=f"{slice_obj.slice_id}_FILL_{uuid.uuid4().hex[:6]}",
                    slice_id=slice_obj.slice_id,
                    parent_order_id=slice_obj.parent_order_id,
                    symbol=slice_obj.symbol,
                    side=slice_obj.side,
                    quantity=fill_quantity,
                    price=fill_price,
                    timestamp=datetime.now(),
                    venue=slice_obj.venue,
                    fees_usd=fees,
                    slippage_bps=slippage_bps,
                    market_impact_bps=market_impact_bps,
                    latency_ms=latency_ms,
                    liquidity_type="taker" if slice_obj.order_type == "MARKET" else "maker",
                    arrival_price=arrival_price,
                    implementation_shortfall_bps=implementation_shortfall_bps
                )
                
                # 执行后风险检查
                post_violations = self.risk_manager.post_execution_risk_check(fill)
                if post_violations:
                    logger.warning(f"执行后风险警告: {post_violations}")
                
                logger.info(f"切片执行成功: {slice_obj.slice_id} - {fill_quantity:.6f} @ {fill_price:.4f}")
                return fill
            
            else:
                slice_obj.status = "no_fill"
                return None
                
        except Exception as e:
            slice_obj.status = "failed"
            slice_obj.execution_end_time = datetime.now()
            logger.error(f"切片执行失败: {slice_obj.slice_id} - {e}")
            return None
    
    def _simulate_execution(self, slice_obj: ExecutionSlice, microstructure: MarketMicrostructure) -> Optional[Tuple[float, float, float]]:
        """模拟订单执行"""
        try:
            if slice_obj.order_type == "MARKET":
                # 市价单执行
                if slice_obj.side == "BUY":
                    fill_price = microstructure.ask_price * (1 + np.random.uniform(0.0001, 0.002))
                else:
                    fill_price = microstructure.bid_price * (1 - np.random.uniform(0.0001, 0.002))
                
                fill_quantity = slice_obj.quantity  # 市价单完全成交
                
            else:
                # 限价单执行
                if slice_obj.limit_price:
                    fill_price = slice_obj.limit_price
                else:
                    fill_price = microstructure.mid_price * (1 + np.random.uniform(-0.0002, 0.0002))
                
                # 基于流动性和紧急程度模拟成交率
                base_fill_rate = microstructure.liquidity_score
                urgency_boost = slice_obj.urgency_score * 0.2
                fill_rate = min(1.0, base_fill_rate + urgency_boost + np.random.uniform(-0.1, 0.1))
                
                fill_quantity = slice_obj.quantity * fill_rate
            
            # 计算费用
            if slice_obj.order_type == "MARKET":
                fee_rate = 0.001  # 0.1% taker费用
            else:
                fee_rate = 0.0005  # 0.05% maker费用
                
            fees = fill_quantity * fill_price * fee_rate
            
            return fill_price, fill_quantity, fees
            
        except Exception as e:
            logger.error(f"模拟执行失败: {e}")
            return None
    
    async def _real_execution(self, slice_obj: ExecutionSlice, microstructure: MarketMicrostructure) -> Optional[Tuple[float, float, float]]:
        """真实订单执行（需要对接交易所API）"""
        # 这里需要实现真实的交易所API调用
        logger.warning("真实交易模式未实现，使用模拟执行")
        return self._simulate_execution(slice_obj, microstructure)
    
    async def _generate_comprehensive_report(
        self, 
        session_id: str, 
        target: TargetPosition,
        algorithm: ExecutionAlgorithm, 
        execution_start: datetime,
        slices: List[ExecutionSlice], 
        fills: List[ExecutionFill],
        feasibility: Dict[str, Any]
    ) -> ExecutionReport:
        """生成全面的执行报告"""
        
        execution_end = datetime.now()
        
        # 基本统计
        total_slices = len(slices)
        successful_slices = len([s for s in slices if s.status == "filled"])
        failed_slices = len([s for s in slices if s.status == "failed"])
        cancelled_slices = len([s for s in slices if s.status in ["blocked", "cancelled"]])
        fill_rate = successful_slices / total_slices if total_slices > 0 else 0
        
        # 数量和金额统计
        size_delta = target.target_size_usd - target.current_size_usd
        target_quantity = abs(size_delta) / feasibility['microstructure'].mid_price
        executed_quantity = sum(f.quantity for f in fills)
        executed_size_usd = sum(f.quantity * f.price for f in fills)
        completion_rate = executed_size_usd / abs(size_delta) if size_delta != 0 else 0
        
        # 成本分析
        total_fees = sum(f.fees_usd for f in fills)
        total_market_impact_bps = sum(f.market_impact_bps * f.quantity for f in fills) / executed_quantity if executed_quantity > 0 else 0
        avg_slippage_bps = sum(f.slippage_bps * f.quantity for f in fills) / executed_quantity if executed_quantity > 0 else 0
        implementation_shortfall_bps = sum(f.implementation_shortfall_bps * f.quantity for f in fills) / executed_quantity if executed_quantity > 0 else 0
        
        # 总成本
        total_cost_usd = total_fees + (total_market_impact_bps / 10000 * executed_size_usd)
        cost_per_share_bps = (total_cost_usd / executed_size_usd * 10000) if executed_size_usd > 0 else 0
        
        # 执行质量指标
        avg_latency_ms = sum(f.latency_ms for f in fills) / len(fills) if fills else 0
        maker_fills = [f for f in fills if f.liquidity_type == "maker"]
        maker_ratio = len(maker_fills) / len(fills) if fills else 0
        
        # 交易所分布
        venue_distribution = {}
        for fill in fills:
            venue_distribution[fill.venue] = venue_distribution.get(fill.venue, 0) + fill.quantity
        total_qty = sum(venue_distribution.values())
        venue_distribution = {k: v/total_qty for k, v in venue_distribution.items()} if total_qty > 0 else {}
        
        # 时间分布
        time_distribution = {}
        for fill in fills:
            hour = fill.timestamp.hour
            time_distribution[f"{hour:02d}:00"] = time_distribution.get(f"{hour:02d}:00", 0) + 1
        
        # DipMaster专用指标
        dipmaster_metrics = {}
        if target.dipmaster_context:
            dipmaster_slices = [s for s in slices if s.algorithm in [ExecutionAlgorithm.DIPMASTER_15MIN, ExecutionAlgorithm.DIPMASTER_DIP_BUY]]
            dipmaster_metrics = {
                'dipmaster_slices': len(dipmaster_slices),
                'timing_boundary_hits': len([s for s in dipmaster_slices if s.dipmaster_timing_window]),
                'dip_signals_executed': len([s for s in dipmaster_slices if s.dipmaster_dip_detected]),
                'avg_rsi_at_execution': np.mean([s.dipmaster_rsi_signal for s in dipmaster_slices if s.dipmaster_rsi_signal]) if any(s.dipmaster_rsi_signal for s in dipmaster_slices) else None
            }
        
        # 风险违规和性能警告
        risk_violations = []
        performance_alerts = []
        
        if avg_slippage_bps > 25:
            performance_alerts.append(f"平均滑点过高: {avg_slippage_bps:.2f}bps")
        if fill_rate < 0.9:
            performance_alerts.append(f"成交率偏低: {fill_rate:.1%}")
        if avg_latency_ms > 1000:
            performance_alerts.append(f"执行延迟过高: {avg_latency_ms:.0f}ms")
        if cost_per_share_bps > 20:
            performance_alerts.append(f"执行成本过高: {cost_per_share_bps:.2f}bps")
        
        return ExecutionReport(
            session_id=session_id,
            symbol=target.symbol,
            target_position=target,
            execution_algorithm=algorithm.value,
            execution_start=execution_start,
            execution_end=execution_end,
            
            total_slices=total_slices,
            successful_slices=successful_slices,
            failed_slices=failed_slices,
            cancelled_slices=cancelled_slices,
            fill_rate=fill_rate,
            
            target_quantity=target_quantity,
            executed_quantity=executed_quantity,
            target_size_usd=abs(size_delta),
            executed_size_usd=executed_size_usd,
            completion_rate=completion_rate,
            
            total_fees_usd=total_fees,
            total_market_impact_bps=total_market_impact_bps,
            avg_slippage_bps=avg_slippage_bps,
            implementation_shortfall_bps=implementation_shortfall_bps,
            total_cost_usd=total_cost_usd,
            cost_per_share_bps=cost_per_share_bps,
            
            avg_latency_ms=avg_latency_ms,
            maker_ratio=maker_ratio,
            venue_distribution=venue_distribution,
            time_distribution=time_distribution,
            
            dipmaster_metrics=dipmaster_metrics,
            risk_violations=risk_violations,
            performance_alerts=performance_alerts,
            fills=fills
        )
    
    def _create_error_report(self, session_id: str, target: TargetPosition, execution_start: datetime, error_msg: str) -> ExecutionReport:
        """创建错误报告"""
        return ExecutionReport(
            session_id=session_id,
            symbol=target.symbol,
            target_position=target,
            execution_algorithm="failed",
            execution_start=execution_start,
            execution_end=datetime.now(),
            
            total_slices=0,
            successful_slices=0,
            failed_slices=0,
            cancelled_slices=0,
            fill_rate=0.0,
            
            target_quantity=0.0,
            executed_quantity=0.0,
            target_size_usd=0.0,
            executed_size_usd=0.0,
            completion_rate=0.0,
            
            total_fees_usd=0.0,
            total_market_impact_bps=0.0,
            avg_slippage_bps=0.0,
            implementation_shortfall_bps=0.0,
            total_cost_usd=0.0,
            cost_per_share_bps=0.0,
            
            avg_latency_ms=0.0,
            maker_ratio=0.0,
            venue_distribution={},
            time_distribution={},
            
            risk_violations=[error_msg],
            performance_alerts=[f"执行失败: {error_msg}"]
        )
    
    def _update_performance_metrics(self, report: ExecutionReport):
        """更新性能指标"""
        symbol = report.symbol
        
        self.performance_metrics[f"{symbol}_fill_rate"].append(report.fill_rate)
        self.performance_metrics[f"{symbol}_slippage_bps"].append(report.avg_slippage_bps)
        self.performance_metrics[f"{symbol}_cost_bps"].append(report.cost_per_share_bps)
        self.performance_metrics[f"{symbol}_latency_ms"].append(report.avg_latency_ms)
        
        # 更新风险评分
        quality_metrics = {
            'avg_slippage_bps': report.avg_slippage_bps,
            'market_impact_bps': report.total_market_impact_bps,
            'avg_latency_ms': report.avg_latency_ms
        }
        self.risk_manager.update_risk_score(symbol, quality_metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        
        for key, values in self.performance_metrics.items():
            if values:
                summary[key] = {
                    'avg': np.mean(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'std': np.std(values),
                    'count': len(values)
                }
        
        summary['risk_status'] = self.risk_manager.get_risk_status()
        summary['last_update'] = datetime.now()
        
        return summary


class ContinuousExecutionScheduler:
    """持续执行调度器"""
    
    def __init__(self, execution_engine: ExecutionEngine):
        self.execution_engine = execution_engine
        self.pending_positions = {}
        self.scheduler_running = False
        self.scheduler_task = None
        
        # 调度参数
        self.check_interval_seconds = 60  # 每分钟检查一次
        self.max_concurrent_executions = 3
        
    async def start_continuous_execution(self):
        """启动持续执行调度"""
        if self.scheduler_running:
            logger.warning("调度器已经在运行")
            return
            
        self.scheduler_running = True
        self.scheduler_task = asyncio.create_task(self._execution_loop())
        logger.info("持续执行调度器已启动")
    
    async def stop_continuous_execution(self):
        """停止持续执行调度"""
        self.scheduler_running = False
        if self.scheduler_task:
            self.scheduler_task.cancel()
            try:
                await self.scheduler_task
            except asyncio.CancelledError:
                pass
        logger.info("持续执行调度器已停止")
    
    def add_target_position(self, position_id: str, target: TargetPosition):
        """添加目标仓位到执行队列"""
        self.pending_positions[position_id] = {
            'target': target,
            'added_time': datetime.now(),
            'retry_count': 0,
            'last_attempt': None,
            'status': 'pending'
        }
        logger.info(f"目标仓位已加入队列: {position_id} - {target.symbol} ${target.target_size_usd:.0f}")
    
    def remove_target_position(self, position_id: str):
        """从执行队列移除目标仓位"""
        if position_id in self.pending_positions:
            del self.pending_positions[position_id]
            logger.info(f"目标仓位已移除: {position_id}")
    
    async def _execution_loop(self):
        """主执行循环"""
        while self.scheduler_running:
            try:
                # 检查待执行的仓位
                ready_positions = self._get_ready_positions()
                
                if ready_positions:
                    # 并发执行（限制数量）
                    semaphore = asyncio.Semaphore(self.max_concurrent_executions)
                    tasks = []
                    
                    for position_id, position_data in ready_positions[:self.max_concurrent_executions]:
                        task = asyncio.create_task(
                            self._execute_position_with_semaphore(semaphore, position_id, position_data)
                        )
                        tasks.append(task)
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                
                # 清理已完成的仓位
                self._cleanup_completed_positions()
                
                # 等待下次检查
                await asyncio.sleep(self.check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"执行调度循环错误: {e}")
                await asyncio.sleep(10)
    
    def _get_ready_positions(self) -> List[Tuple[str, Dict]]:
        """获取准备执行的仓位"""
        ready = []
        current_time = datetime.now()
        
        for position_id, position_data in self.pending_positions.items():
            if position_data['status'] != 'pending':
                continue
                
            target = position_data['target']
            
            # 检查是否到了执行时间
            should_execute = False
            
            # DipMaster特殊时机检查
            if target.dipmaster_context:
                if target.dipmaster_context.get('timing_critical'):
                    # 15分钟边界检查
                    current_minute = current_time.minute
                    if current_minute in [13, 14, 28, 29, 43, 44, 58, 59]:  # 边界前1-2分钟
                        should_execute = True
                        
                elif target.dipmaster_context.get('dip_signal'):
                    # 逢跌信号立即执行
                    should_execute = True
            
            # 常规执行条件
            if not should_execute:
                # 检查是否超过最大等待时间
                wait_time = (current_time - position_data['added_time']).total_seconds() / 60
                if wait_time > 10:  # 超过10分钟自动执行
                    should_execute = True
                    
                # 检查重试间隔
                if position_data['last_attempt']:
                    retry_wait = (current_time - position_data['last_attempt']).total_seconds() / 60
                    if retry_wait > 5:  # 5分钟后重试
                        should_execute = True
            
            if should_execute:
                ready.append((position_id, position_data))
        
        return ready
    
    async def _execute_position_with_semaphore(self, semaphore: asyncio.Semaphore, position_id: str, position_data: Dict):
        """带信号量的仓位执行"""
        async with semaphore:
            try:
                position_data['status'] = 'executing'
                position_data['last_attempt'] = datetime.now()
                
                target = position_data['target']
                report = await self.execution_engine.execute_target_position(target)
                
                # 检查执行结果
                if report.completion_rate > 0.95:  # 95%以上完成率认为成功
                    position_data['status'] = 'completed'
                    position_data['report'] = report
                    logger.info(f"仓位执行完成: {position_id} - 完成率 {report.completion_rate:.1%}")
                else:
                    position_data['retry_count'] += 1
                    if position_data['retry_count'] >= 3:
                        position_data['status'] = 'failed'
                        logger.error(f"仓位执行失败（超过重试次数）: {position_id}")
                    else:
                        position_data['status'] = 'pending'
                        logger.warning(f"仓位执行未完成，将重试: {position_id} - 完成率 {report.completion_rate:.1%}")
                
            except Exception as e:
                position_data['status'] = 'error'
                logger.error(f"仓位执行异常: {position_id} - {e}")
    
    def _cleanup_completed_positions(self):
        """清理已完成的仓位"""
        completed_positions = [
            pid for pid, data in self.pending_positions.items() 
            if data['status'] in ['completed', 'failed', 'error']
        ]
        
        for position_id in completed_positions:
            # 保留一段时间后清理
            position_data = self.pending_positions[position_id]
            if position_data['last_attempt']:
                age_hours = (datetime.now() - position_data['last_attempt']).total_seconds() / 3600
                if age_hours > 1:  # 1小时后清理
                    del self.pending_positions[position_id]
                    logger.debug(f"已清理完成仓位: {position_id}")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """获取调度器状态"""
        status_counts = defaultdict(int)
        for data in self.pending_positions.values():
            status_counts[data['status']] += 1
        
        return {
            'running': self.scheduler_running,
            'pending_positions': len(self.pending_positions),
            'status_breakdown': dict(status_counts),
            'next_check_in_seconds': self.check_interval_seconds,
            'last_update': datetime.now()
        }


# 工厂函数和便利接口

def create_dipmaster_execution_system(paper_trading: bool = True) -> Tuple[ExecutionEngine, ContinuousExecutionScheduler]:
    """创建DipMaster专用执行系统"""
    engine = ExecutionEngine(paper_trading=paper_trading)
    scheduler = ContinuousExecutionScheduler(engine)
    return engine, scheduler


def create_dipmaster_target_position(
    symbol: str,
    target_size_usd: float,
    current_size_usd: float = 0.0,
    timing_critical: bool = False,
    dip_signal: bool = False,
    rsi_value: Optional[float] = None,
    urgency: OrderUrgency = OrderUrgency.MEDIUM
) -> TargetPosition:
    """创建DipMaster目标仓位"""
    dipmaster_context = {}
    
    if timing_critical:
        dipmaster_context['timing_critical'] = True
        urgency = OrderUrgency.HIGH
        
    if dip_signal:
        dipmaster_context['dip_signal'] = True
        urgency = OrderUrgency.HIGH
        
    if rsi_value:
        dipmaster_context['rsi_value'] = rsi_value
    
    return TargetPosition(
        symbol=symbol,
        target_size_usd=target_size_usd,
        current_size_usd=current_size_usd,
        urgency=urgency,
        max_execution_time_minutes=15 if timing_critical else 60,
        dipmaster_context=dipmaster_context if dipmaster_context else None
    )


# 演示函数
async def demo_professional_execution_system():
    """专业执行系统演示"""
    print("="*80)
    print("DipMaster Trading System - 专业级执行管理系统演示")
    print("="*80)
    
    # 创建执行系统
    engine, scheduler = create_dipmaster_execution_system(paper_trading=True)
    
    # 启动持续执行调度
    await scheduler.start_continuous_execution()
    
    try:
        # 场景1: DipMaster 15分钟边界执行
        print("\n场景1: DipMaster 15分钟边界执行")
        print("-" * 50)
        
        dipmaster_15min_position = create_dipmaster_target_position(
            symbol="BTCUSDT",
            target_size_usd=8000.0,
            current_size_usd=2000.0,
            timing_critical=True,
            urgency=OrderUrgency.HIGH
        )
        
        scheduler.add_target_position("dipmaster_15min", dipmaster_15min_position)
        
        # 场景2: DipMaster 逢跌买入
        print("\n场景2: DipMaster 逢跌买入执行")
        print("-" * 50)
        
        dipmaster_dip_position = create_dipmaster_target_position(
            symbol="ETHUSDT",
            target_size_usd=5000.0,
            current_size_usd=0.0,
            dip_signal=True,
            rsi_value=35.5,
            urgency=OrderUrgency.HIGH
        )
        
        scheduler.add_target_position("dipmaster_dip", dipmaster_dip_position)
        
        # 场景3: 常规大额订单VWAP执行
        print("\n场景3: 常规大额订单VWAP执行")
        print("-" * 50)
        
        large_position = TargetPosition(
            symbol="SOLUSDT",
            target_size_usd=15000.0,
            current_size_usd=3000.0,
            urgency=OrderUrgency.MEDIUM,
            max_execution_time_minutes=45
        )
        
        scheduler.add_target_position("large_vwap", large_position)
        
        # 等待执行完成
        print("\n等待执行完成...")
        await asyncio.sleep(30)  # 等待30秒观察执行
        
        # 检查调度器状态
        status = scheduler.get_scheduler_status()
        print(f"\n调度器状态: {json.dumps(status, indent=2, default=str)}")
        
        # 获取性能摘要
        performance = engine.get_performance_summary()
        print(f"\n性能摘要:")
        for key, metrics in performance.items():
            if isinstance(metrics, dict) and 'avg' in metrics:
                print(f"  {key}: 平均={metrics['avg']:.2f}, 标准差={metrics['std']:.2f}, 样本数={metrics['count']}")
        
    finally:
        await scheduler.stop_continuous_execution()
    
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(demo_professional_execution_system())