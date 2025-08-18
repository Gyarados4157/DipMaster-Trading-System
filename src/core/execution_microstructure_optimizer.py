"""
DipMaster Trading System - Execution Microstructure Optimizer
专业的执行微结构优化引擎 - 最小化交易成本和市场影响

核心功能:
1. 微结构分析和流动性评估
2. 智能订单分割和时间调度
3. 执行成本优化和滑点控制
4. 实时风险管理和监控
5. 执行质量分析和TCA
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import pandas as pd
from enum import Enum

logger = logging.getLogger(__name__)

class ExecutionAlgorithm(Enum):
    """执行算法类型"""
    TWAP = "twap"
    VWAP = "vwap"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    ICEBERG = "iceberg"
    MARKET = "market"
    LIMIT = "limit"

class OrderType(Enum):
    """订单类型"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LIMIT = "STOP_LIMIT"
    ICEBERG = "ICEBERG"
    OCO = "OCO"

@dataclass
class MarketMicrostructure:
    """市场微结构分析"""
    symbol: str
    timestamp: datetime
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    spread_bps: float
    mid_price: float
    depth_10_bid: float
    depth_10_ask: float
    volume_1min: float
    volume_5min: float
    price_impact_coefficient: float
    liquidity_score: float
    volatility_1min: float
    order_flow_imbalance: float

@dataclass
class ExecutionSlice:
    """执行切片"""
    slice_id: str
    parent_order_id: str
    symbol: str
    side: str
    size_usd: float
    quantity: float
    algorithm: ExecutionAlgorithm
    order_type: OrderType
    limit_price: Optional[float] = None
    scheduled_time: Optional[datetime] = None
    urgency_score: float = 0.5
    max_participation_rate: float = 0.2
    timeout_seconds: int = 300
    status: str = "pending"
    execution_time: Optional[datetime] = None
    fill_quantity: float = 0.0
    fill_price: float = 0.0
    slippage_bps: float = 0.0
    market_impact_bps: float = 0.0
    venue: str = "binance"

@dataclass
class ExecutionFill:
    """执行成交记录"""
    fill_id: str
    slice_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    timestamp: datetime
    fees_usd: float
    slippage_bps: float
    latency_ms: float
    venue: str
    liquidity_type: str  # maker, taker
    order_type: str
    arrival_price: float

@dataclass
class ExecutionCosts:
    """执行成本分析"""
    total_fees_usd: float
    market_impact_bps: float
    spread_cost_usd: float
    timing_cost_bps: float
    total_cost_usd: float
    cost_per_share_bps: float

@dataclass
class ExecutionReport:
    """执行报告"""
    session_id: str
    symbol: str
    target_size_usd: float
    executed_size_usd: float
    algorithm_used: str
    execution_start: datetime
    execution_end: datetime
    total_slices: int
    successful_fills: int
    fill_rate: float
    avg_slippage_bps: float
    total_costs: ExecutionCosts
    quality_metrics: Dict
    microstructure_analysis: Dict
    violation_alerts: List[str]

class MarketDataSimulator:
    """市场数据模拟器（用于演示）"""
    
    def __init__(self):
        self.base_prices = {
            'BTCUSDT': 65000.0,
            'ETHUSDT': 3200.0,
            'SOLUSDT': 140.0,
            'BNBUSDT': 580.0,
            'ADAUSDT': 0.45
        }
        self.volatilities = {
            'BTCUSDT': 0.02,
            'ETHUSDT': 0.025,
            'SOLUSDT': 0.035,
            'BNBUSDT': 0.02,
            'ADAUSDT': 0.03
        }
    
    def get_current_price(self, symbol: str) -> float:
        """获取当前价格"""
        base_price = self.base_prices.get(symbol, 50000.0)
        volatility = self.volatilities.get(symbol, 0.02)
        # 添加随机价格变动
        price_change = np.random.normal(0, volatility * base_price / np.sqrt(365 * 24 * 60))
        return base_price + price_change
    
    def get_market_microstructure(self, symbol: str) -> MarketMicrostructure:
        """获取市场微结构数据"""
        mid_price = self.get_current_price(symbol)
        spread_bps = np.random.uniform(5, 20)  # 5-20 bps spread
        spread = mid_price * spread_bps / 10000
        
        bid_price = mid_price - spread / 2
        ask_price = mid_price + spread / 2
        
        # 模拟订单簿深度
        bid_size = np.random.exponential(100)
        ask_size = np.random.exponential(100)
        
        return MarketMicrostructure(
            symbol=symbol,
            timestamp=datetime.now(),
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            spread_bps=spread_bps,
            mid_price=mid_price,
            depth_10_bid=bid_size * np.random.uniform(8, 12),
            depth_10_ask=ask_size * np.random.uniform(8, 12),
            volume_1min=np.random.exponential(50),
            volume_5min=np.random.exponential(250),
            price_impact_coefficient=np.random.uniform(0.1, 0.5),
            liquidity_score=np.random.uniform(0.6, 0.95),
            volatility_1min=self.volatilities.get(symbol, 0.02) / np.sqrt(365 * 24 * 60),
            order_flow_imbalance=np.random.uniform(-0.3, 0.3)
        )

class LiquidityAnalyzer:
    """流动性分析器"""
    
    def __init__(self, market_data: MarketDataSimulator):
        self.market_data = market_data
    
    def analyze_liquidity_profile(self, symbol: str, size_usd: float) -> Dict:
        """分析流动性概况"""
        microstructure = self.market_data.get_market_microstructure(symbol)
        
        # 计算市场影响
        participation_rate = min(0.2, size_usd / (microstructure.volume_5min * microstructure.mid_price))
        
        # 估算价格影响
        price_impact_bps = microstructure.price_impact_coefficient * np.sqrt(participation_rate) * 100
        
        # 流动性评分
        liquidity_score = min(1.0, microstructure.liquidity_score * (1 - participation_rate))
        
        # 建议执行时间
        if size_usd > 5000:
            suggested_duration_min = max(30, size_usd / 500)
        else:
            suggested_duration_min = max(15, size_usd / 300)
        
        return {
            'symbol': symbol,
            'size_usd': size_usd,
            'participation_rate': participation_rate,
            'estimated_impact_bps': price_impact_bps,
            'liquidity_score': liquidity_score,
            'suggested_duration_min': suggested_duration_min,
            'recommended_algorithm': self._recommend_algorithm(size_usd, price_impact_bps, liquidity_score),
            'microstructure': asdict(microstructure)
        }
    
    def _recommend_algorithm(self, size_usd: float, impact_bps: float, liquidity_score: float) -> str:
        """推荐执行算法"""
        if size_usd > 5000 and impact_bps > 10:
            return ExecutionAlgorithm.VWAP.value
        elif size_usd > 2000:
            return ExecutionAlgorithm.TWAP.value
        elif impact_bps < 5 and liquidity_score > 0.8:
            return ExecutionAlgorithm.MARKET.value
        else:
            return ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL.value

class SmartOrderSlicer:
    """智能订单分割器"""
    
    def __init__(self, market_data: MarketDataSimulator):
        self.market_data = market_data
    
    def slice_twap(self, symbol: str, size_usd: float, duration_minutes: int) -> List[ExecutionSlice]:
        """TWAP分割算法"""
        current_price = self.market_data.get_current_price(symbol)
        total_quantity = size_usd / current_price
        
        # 计算分割参数
        num_slices = max(3, min(20, duration_minutes // 3))
        slice_interval = duration_minutes * 60 / num_slices
        base_quantity = total_quantity / num_slices
        
        slices = []
        parent_id = f"TWAP_{symbol}_{int(time.time())}"
        
        for i in range(num_slices):
            # 添加随机化避免被识别
            quantity_factor = np.random.uniform(0.8, 1.2)
            slice_quantity = base_quantity * quantity_factor
            
            # 最后一片调整为剩余数量
            if i == num_slices - 1:
                slice_quantity = total_quantity - sum(s.quantity for s in slices)
            
            scheduled_time = datetime.now() + timedelta(seconds=i * slice_interval)
            
            slice_obj = ExecutionSlice(
                slice_id=f"{parent_id}_T{i+1:02d}",
                parent_order_id=parent_id,
                symbol=symbol,
                side="BUY",
                size_usd=slice_quantity * current_price,
                quantity=slice_quantity,
                algorithm=ExecutionAlgorithm.TWAP,
                order_type=OrderType.LIMIT,
                scheduled_time=scheduled_time,
                urgency_score=0.3,  # TWAP相对不急
                max_participation_rate=0.15
            )
            slices.append(slice_obj)
        
        logger.info(f"TWAP切片完成: {total_quantity:.6f} {symbol} -> {num_slices}片, 耗时{duration_minutes}分钟")
        return slices
    
    def slice_vwap(self, symbol: str, size_usd: float, duration_minutes: int) -> List[ExecutionSlice]:
        """VWAP分割算法"""
        current_price = self.market_data.get_current_price(symbol)
        total_quantity = size_usd / current_price
        
        # 获取历史成交量分布（简化模型）
        volume_weights = self._get_volume_profile(duration_minutes)
        num_slices = len(volume_weights)
        
        slices = []
        parent_id = f"VWAP_{symbol}_{int(time.time())}"
        slice_interval = duration_minutes * 60 / num_slices
        
        for i, weight in enumerate(volume_weights):
            slice_quantity = total_quantity * weight
            scheduled_time = datetime.now() + timedelta(seconds=i * slice_interval)
            
            slice_obj = ExecutionSlice(
                slice_id=f"{parent_id}_V{i+1:02d}",
                parent_order_id=parent_id,
                symbol=symbol,
                side="BUY",
                size_usd=slice_quantity * current_price,
                quantity=slice_quantity,
                algorithm=ExecutionAlgorithm.VWAP,
                order_type=OrderType.LIMIT,
                scheduled_time=scheduled_time,
                urgency_score=0.4,
                max_participation_rate=0.25
            )
            slices.append(slice_obj)
        
        logger.info(f"VWAP切片完成: {total_quantity:.6f} {symbol} -> {num_slices}片, 基于成交量分布")
        return slices
    
    def slice_implementation_shortfall(self, symbol: str, size_usd: float, urgency: float) -> List[ExecutionSlice]:
        """Implementation Shortfall算法"""
        current_price = self.market_data.get_current_price(symbol)
        total_quantity = size_usd / current_price
        
        # 基于紧急程度调整执行策略
        if urgency > 0.8:
            # 高紧急度：快速执行
            num_slices = max(2, int(size_usd / 2000))
            duration_minutes = max(5, 20 * (1 - urgency))
        else:
            # 低紧急度：成本优化
            num_slices = max(5, int(size_usd / 1000))
            duration_minutes = max(20, 60 * (1 - urgency))
        
        # 前置权重：紧急订单前重后轻
        weights = []
        for i in range(num_slices):
            if urgency > 0.7:
                weight = np.exp(-i * 0.2)  # 指数递减
            else:
                weight = 1.0  # 均匀分布
            weights.append(weight)
        
        # 标准化权重
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        slices = []
        parent_id = f"IS_{symbol}_{int(time.time())}"
        slice_interval = duration_minutes * 60 / num_slices
        
        for i, weight in enumerate(weights):
            slice_quantity = total_quantity * weight
            scheduled_time = datetime.now() + timedelta(seconds=i * slice_interval)
            
            # 根据紧急程度选择订单类型
            if urgency > 0.8:
                order_type = OrderType.MARKET
                urgency_score = urgency
            else:
                order_type = OrderType.LIMIT
                urgency_score = urgency * 0.7
            
            slice_obj = ExecutionSlice(
                slice_id=f"{parent_id}_IS{i+1:02d}",
                parent_order_id=parent_id,
                symbol=symbol,
                side="BUY",
                size_usd=slice_quantity * current_price,
                quantity=slice_quantity,
                algorithm=ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL,
                order_type=order_type,
                scheduled_time=scheduled_time,
                urgency_score=urgency_score,
                max_participation_rate=0.3 if urgency > 0.7 else 0.2
            )
            slices.append(slice_obj)
        
        logger.info(f"Implementation Shortfall切片完成: {total_quantity:.6f} {symbol} -> {num_slices}片, 紧急度{urgency:.2f}")
        return slices
    
    def _get_volume_profile(self, duration_minutes: int) -> List[float]:
        """获取成交量分布权重"""
        # 简化的成交量分布模型
        current_hour = datetime.now().hour
        
        # 24小时成交量权重（基于典型的加密货币交易模式）
        hourly_weights = [
            0.6, 0.5, 0.4, 0.3, 0.3, 0.4,  # 00-05: 低活跃
            0.6, 0.8, 1.0, 1.2, 1.4, 1.6,  # 06-11: 上升
            1.8, 2.0, 1.8, 1.6, 1.4, 1.2,  # 12-17: 高峰
            1.0, 0.8, 0.6, 0.5, 0.4, 0.3   # 18-23: 下降
        ]
        
        # 生成指定时长内的权重
        num_intervals = max(4, duration_minutes // 15)  # 每15分钟一个区间
        weights = []
        
        for i in range(num_intervals):
            hour_index = (current_hour + i // 4) % 24
            weight = hourly_weights[hour_index]
            weights.append(weight)
        
        # 标准化权重
        total_weight = sum(weights)
        return [w / total_weight for w in weights]

class ExecutionRiskManager:
    """执行风险管理器"""
    
    def __init__(self):
        self.max_slippage_bps = 50  # 最大滑点50bps
        self.max_market_impact_bps = 30  # 最大市场冲击30bps
        self.max_position_size_usd = 10000  # 最大单个头寸
        self.max_participation_rate = 0.3  # 最大参与率30%
        self.max_execution_time_hours = 4  # 最大执行时间
        
        self.violations = []
        self.circuit_breaker_triggered = False
    
    def pre_execution_check(self, slice_obj: ExecutionSlice, microstructure: MarketMicrostructure) -> Tuple[bool, List[str]]:
        """执行前风险检查"""
        violations = []
        
        # 检查仓位大小
        if slice_obj.size_usd > self.max_position_size_usd:
            violations.append(f"仓位大小超限: ${slice_obj.size_usd:.0f} > ${self.max_position_size_usd}")
        
        # 检查参与率
        estimated_volume = microstructure.volume_5min * microstructure.mid_price
        if estimated_volume > 0:
            participation_rate = slice_obj.size_usd / estimated_volume
            if participation_rate > self.max_participation_rate:
                violations.append(f"参与率过高: {participation_rate:.2%} > {self.max_participation_rate:.2%}")
        
        # 检查价格影响
        if microstructure.price_impact_coefficient > 0:
            estimated_impact = microstructure.price_impact_coefficient * np.sqrt(slice_obj.size_usd / 10000) * 100
            if estimated_impact > self.max_market_impact_bps:
                violations.append(f"预期市场冲击过大: {estimated_impact:.1f}bps > {self.max_market_impact_bps}bps")
        
        # 检查执行时间
        if slice_obj.timeout_seconds > self.max_execution_time_hours * 3600:
            violations.append(f"执行时间过长: {slice_obj.timeout_seconds//3600}h > {self.max_execution_time_hours}h")
        
        can_execute = len(violations) == 0 and not self.circuit_breaker_triggered
        return can_execute, violations
    
    def post_execution_check(self, fill: ExecutionFill) -> List[str]:
        """执行后风险检查"""
        violations = []
        
        # 检查滑点
        if abs(fill.slippage_bps) > self.max_slippage_bps:
            violations.append(f"滑点超限: {fill.slippage_bps:.1f}bps > {self.max_slippage_bps}bps")
        
        # 连续违规检查
        if violations:
            self.violations.extend(violations)
            
            # 如果连续3次违规，触发熔断
            if len(self.violations) >= 3:
                self.circuit_breaker_triggered = True
                violations.append("风险熔断器触发: 连续违规，暂停执行")
        
        return violations
    
    def reset_circuit_breaker(self):
        """重置熔断器"""
        self.circuit_breaker_triggered = False
        self.violations = []
        logger.info("风险熔断器已重置")

class ExecutionEngine:
    """主执行引擎"""
    
    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        self.market_data = MarketDataSimulator()
        self.liquidity_analyzer = LiquidityAnalyzer(self.market_data)
        self.order_slicer = SmartOrderSlicer(self.market_data)
        self.risk_manager = ExecutionRiskManager()
        
        self.execution_sessions = {}
        self.fills = []
    
    async def execute_target_position(self, symbol: str, target_size_usd: float, urgency: float = 0.5) -> ExecutionReport:
        """执行目标仓位"""
        session_id = f"EXEC_{symbol}_{int(time.time())}"
        execution_start = datetime.now()
        
        logger.info(f"开始执行: {session_id} - {symbol} ${target_size_usd:.0f} (紧急度: {urgency:.2f})")
        
        try:
            # 1. 流动性分析
            liquidity_profile = self.liquidity_analyzer.analyze_liquidity_profile(symbol, target_size_usd)
            
            # 2. 选择执行算法
            algorithm = liquidity_profile['recommended_algorithm']
            duration_min = int(liquidity_profile['suggested_duration_min'])
            
            logger.info(f"推荐算法: {algorithm}, 建议执行时间: {duration_min}分钟")
            
            # 3. 订单分割
            if algorithm == ExecutionAlgorithm.TWAP.value:
                slices = self.order_slicer.slice_twap(symbol, target_size_usd, duration_min)
            elif algorithm == ExecutionAlgorithm.VWAP.value:
                slices = self.order_slicer.slice_vwap(symbol, target_size_usd, duration_min)
            else:
                slices = self.order_slicer.slice_implementation_shortfall(symbol, target_size_usd, urgency)
            
            # 4. 执行订单切片
            executed_fills = await self._execute_slices(slices)
            
            # 5. 生成执行报告
            execution_report = await self._generate_execution_report(
                session_id, symbol, target_size_usd, algorithm, 
                execution_start, slices, executed_fills, liquidity_profile
            )
            
            # 6. 保存执行记录
            self.execution_sessions[session_id] = execution_report
            
            logger.info(f"执行完成: {session_id} - 成交{len(executed_fills)}笔")
            return execution_report
            
        except Exception as e:
            logger.error(f"执行失败: {session_id} - {e}")
            # 返回错误报告
            return ExecutionReport(
                session_id=session_id,
                symbol=symbol,
                target_size_usd=target_size_usd,
                executed_size_usd=0.0,
                algorithm_used="failed",
                execution_start=execution_start,
                execution_end=datetime.now(),
                total_slices=0,
                successful_fills=0,
                fill_rate=0.0,
                avg_slippage_bps=0.0,
                total_costs=ExecutionCosts(0, 0, 0, 0, 0, 0),
                quality_metrics={},
                microstructure_analysis={},
                violation_alerts=[str(e)]
            )
    
    async def _execute_slices(self, slices: List[ExecutionSlice]) -> List[ExecutionFill]:
        """执行订单切片"""
        fills = []
        
        for slice_obj in slices:
            try:
                # 获取最新市场微结构
                microstructure = self.market_data.get_market_microstructure(slice_obj.symbol)
                
                # 执行前风险检查
                can_execute, violations = self.risk_manager.pre_execution_check(slice_obj, microstructure)
                
                if not can_execute:
                    logger.warning(f"切片{slice_obj.slice_id}被风控阻止: {violations}")
                    slice_obj.status = "blocked"
                    continue
                
                # 等待调度时间
                if slice_obj.scheduled_time and slice_obj.scheduled_time > datetime.now():
                    wait_seconds = (slice_obj.scheduled_time - datetime.now()).total_seconds()
                    if wait_seconds > 0 and wait_seconds <= 300:  # 最多等待5分钟
                        await asyncio.sleep(wait_seconds)
                
                # 执行订单
                fill = await self._execute_single_slice(slice_obj, microstructure)
                
                if fill:
                    fills.append(fill)
                    self.fills.append(fill)
                    
                    # 执行后风险检查
                    post_violations = self.risk_manager.post_execution_check(fill)
                    if post_violations:
                        logger.warning(f"执行后发现风险: {post_violations}")
                        break  # 停止后续执行
                
                # 订单间隔
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"切片执行失败 {slice_obj.slice_id}: {e}")
                slice_obj.status = "failed"
        
        return fills
    
    async def _execute_single_slice(self, slice_obj: ExecutionSlice, microstructure: MarketMicrostructure) -> Optional[ExecutionFill]:
        """执行单个切片"""
        execution_start = time.time()
        arrival_price = microstructure.mid_price
        
        slice_obj.status = "executing"
        slice_obj.execution_time = datetime.now()
        
        try:
            if self.paper_trading:
                # 纸上交易模拟
                fill_price, fill_quantity = self._simulate_execution(slice_obj, microstructure)
            else:
                # 真实交易执行
                fill_price, fill_quantity = await self._real_execution(slice_obj, microstructure)
            
            if fill_quantity > 0:
                # 计算滑点和费用
                slippage_bps = ((fill_price - arrival_price) / arrival_price) * 10000 if arrival_price > 0 else 0
                fees_usd = fill_quantity * fill_price * 0.001  # 0.1%手续费
                latency_ms = (time.time() - execution_start) * 1000
                
                # 更新切片状态
                slice_obj.status = "filled"
                slice_obj.fill_quantity = fill_quantity
                slice_obj.fill_price = fill_price
                slice_obj.slippage_bps = slippage_bps
                
                # 创建成交记录
                fill = ExecutionFill(
                    fill_id=f"{slice_obj.slice_id}_FILL_{int(time.time())}",
                    slice_id=slice_obj.slice_id,
                    symbol=slice_obj.symbol,
                    side=slice_obj.side,
                    quantity=fill_quantity,
                    price=fill_price,
                    timestamp=datetime.now(),
                    fees_usd=fees_usd,
                    slippage_bps=slippage_bps,
                    latency_ms=latency_ms,
                    venue=slice_obj.venue,
                    liquidity_type="taker" if slice_obj.order_type == OrderType.MARKET else "maker",
                    order_type=slice_obj.order_type.value,
                    arrival_price=arrival_price
                )
                
                logger.info(f"切片执行成功: {slice_obj.slice_id} - {fill_quantity:.6f} @ {fill_price:.2f} (滑点: {slippage_bps:.2f}bps)")
                return fill
            
            else:
                slice_obj.status = "no_fill"
                return None
                
        except Exception as e:
            slice_obj.status = "failed"
            logger.error(f"切片执行异常 {slice_obj.slice_id}: {e}")
            return None
    
    def _simulate_execution(self, slice_obj: ExecutionSlice, microstructure: MarketMicrostructure) -> Tuple[float, float]:
        """模拟执行（纸上交易）"""
        if slice_obj.order_type == OrderType.MARKET:
            # 市价单：按ask价格成交，有滑点
            fill_price = microstructure.ask_price * (1 + np.random.uniform(0.0001, 0.001))  # 0.01%-0.1%滑点
            fill_quantity = slice_obj.quantity  # 市价单全部成交
            
        elif slice_obj.order_type == OrderType.LIMIT:
            # 限价单：设定限价，模拟部分成交
            if slice_obj.limit_price:
                fill_price = slice_obj.limit_price
            else:
                # 设置略优于mid价格的限价
                fill_price = microstructure.mid_price * (1 + np.random.uniform(-0.0005, 0.0005))
            
            # 根据市场流动性模拟成交率
            fill_rate = min(1.0, microstructure.liquidity_score * np.random.uniform(0.7, 1.0))
            fill_quantity = slice_obj.quantity * fill_rate
            
        else:
            # 其他订单类型默认处理
            fill_price = microstructure.mid_price
            fill_quantity = slice_obj.quantity * np.random.uniform(0.8, 1.0)
        
        return fill_price, fill_quantity
    
    async def _real_execution(self, slice_obj: ExecutionSlice, microstructure: MarketMicrostructure) -> Tuple[float, float]:
        """真实执行（连接交易所API）"""
        # 这里应该连接到真实的交易所API
        # 例如Binance API调用
        
        # 暂时返回模拟结果
        logger.warning("真实交易模式未实现，使用模拟执行")
        return self._simulate_execution(slice_obj, microstructure)
    
    async def _generate_execution_report(self, session_id: str, symbol: str, target_size_usd: float, 
                                       algorithm: str, execution_start: datetime, 
                                       slices: List[ExecutionSlice], fills: List[ExecutionFill],
                                       liquidity_profile: Dict) -> ExecutionReport:
        """生成执行报告"""
        execution_end = datetime.now()
        
        # 基本统计
        total_slices = len(slices)
        successful_fills = len(fills)
        fill_rate = successful_fills / total_slices if total_slices > 0 else 0
        
        # 执行量统计
        executed_quantity = sum(f.quantity for f in fills)
        executed_size_usd = sum(f.quantity * f.price for f in fills)
        
        # 成本分析
        total_fees = sum(f.fees_usd for f in fills)
        avg_slippage = sum(f.slippage_bps for f in fills) / len(fills) if fills else 0
        
        # 市场影响估算
        if fills and executed_quantity > 0:
            weighted_avg_price = executed_size_usd / executed_quantity
            arrival_price = fills[0].arrival_price if fills else 0
            market_impact_bps = abs((weighted_avg_price - arrival_price) / arrival_price) * 10000 if arrival_price > 0 else 0
        else:
            market_impact_bps = 0
        
        # 点差成本估算
        spread_cost_usd = executed_size_usd * 0.0005  # 估算0.05%点差成本
        
        total_costs = ExecutionCosts(
            total_fees_usd=total_fees,
            market_impact_bps=market_impact_bps,
            spread_cost_usd=spread_cost_usd,
            timing_cost_bps=0,  # 时机成本暂不计算
            total_cost_usd=total_fees + spread_cost_usd,
            cost_per_share_bps=(total_fees + spread_cost_usd) / executed_size_usd * 10000 if executed_size_usd > 0 else 0
        )
        
        # 质量指标
        quality_metrics = {
            "fill_rate": fill_rate,
            "avg_slippage_bps": avg_slippage,
            "market_impact_bps": market_impact_bps,
            "execution_efficiency": executed_size_usd / target_size_usd if target_size_usd > 0 else 0,
            "avg_latency_ms": sum(f.latency_ms for f in fills) / len(fills) if fills else 0,
            "cost_per_share_bps": total_costs.cost_per_share_bps,
            "maker_ratio": len([f for f in fills if f.liquidity_type == "maker"]) / len(fills) if fills else 0
        }
        
        # 风险违规
        violation_alerts = self.risk_manager.violations.copy()
        
        return ExecutionReport(
            session_id=session_id,
            symbol=symbol,
            target_size_usd=target_size_usd,
            executed_size_usd=executed_size_usd,
            algorithm_used=algorithm,
            execution_start=execution_start,
            execution_end=execution_end,
            total_slices=total_slices,
            successful_fills=successful_fills,
            fill_rate=fill_rate,
            avg_slippage_bps=avg_slippage,
            total_costs=total_costs,
            quality_metrics=quality_metrics,
            microstructure_analysis=liquidity_profile,
            violation_alerts=violation_alerts
        )

async def main():
    """执行微结构优化器演示"""
    
    # 初始化执行引擎
    engine = ExecutionEngine(paper_trading=True)
    
    # BTCUSDT $8,000多头仓位执行
    symbol = "BTCUSDT"
    target_size_usd = 8000.0
    urgency = 0.6  # 中等紧急程度
    
    print("="*80)
    print("DipMaster Trading System - 执行微结构优化器")
    print("="*80)
    print(f"目标仓位: {symbol} ${target_size_usd:,.0f} 多头")
    print(f"执行紧急度: {urgency:.1f}/1.0")
    print()
    
    # 执行订单
    execution_report = await engine.execute_target_position(symbol, target_size_usd, urgency)
    
    # 显示结果
    print("执行报告:")
    print("-" * 60)
    print(f"会话ID: {execution_report.session_id}")
    print(f"执行算法: {execution_report.algorithm_used}")
    print(f"执行时间: {(execution_report.execution_end - execution_report.execution_start).total_seconds():.1f}秒")
    print(f"目标金额: ${execution_report.target_size_usd:,.0f}")
    print(f"实际执行: ${execution_report.executed_size_usd:,.0f}")
    print(f"执行率: {execution_report.executed_size_usd/execution_report.target_size_usd:.1%}")
    print()
    
    print("成本分析:")
    print("-" * 60)
    print(f"总手续费: ${execution_report.total_costs.total_fees_usd:.2f}")
    print(f"市场冲击: {execution_report.total_costs.market_impact_bps:.2f} bps")
    print(f"点差成本: ${execution_report.total_costs.spread_cost_usd:.2f}")
    print(f"总成本: ${execution_report.total_costs.total_cost_usd:.2f}")
    print(f"每股成本: {execution_report.total_costs.cost_per_share_bps:.2f} bps")
    print()
    
    print("执行质量:")
    print("-" * 60)
    print(f"成交切片: {execution_report.successful_fills}/{execution_report.total_slices}")
    print(f"成交率: {execution_report.fill_rate:.1%}")
    print(f"平均滑点: {execution_report.avg_slippage_bps:.2f} bps")
    print(f"执行效率: {execution_report.quality_metrics['execution_efficiency']:.1%}")
    print(f"平均延迟: {execution_report.quality_metrics['avg_latency_ms']:.1f} ms")
    print(f"Maker比例: {execution_report.quality_metrics['maker_ratio']:.1%}")
    print()
    
    if execution_report.violation_alerts:
        print("风险警告:")
        print("-" * 60)
        for alert in execution_report.violation_alerts:
            print(f"⚠️  {alert}")
        print()
    
    # 保存执行报告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_filename = f"/Users/zhangxuanyang/Desktop/Quant/DipMaster-Trading-System/results/execution_reports/ExecutionReport_Microstructure_{timestamp}.json"
    
    import os
    os.makedirs(os.path.dirname(report_filename), exist_ok=True)
    
    # 转换为可JSON序列化的格式
    report_dict = asdict(execution_report)
    
    # 处理datetime对象
    def convert_datetime(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return obj
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False, default=convert_datetime)
    
    print(f"执行报告已保存: {report_filename}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())