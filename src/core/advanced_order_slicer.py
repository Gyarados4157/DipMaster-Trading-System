"""
Advanced Order Slicing Algorithms
实现智能订单分割算法 - TWAP/VWAP/Implementation Shortfall
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class SlicingAlgorithm(Enum):
    """订单分割算法类型"""
    TWAP = "twap"           # 时间加权平均价格
    VWAP = "vwap"           # 成交量加权平均价格
    IMPLEMENTATION_SHORTFALL = "is"  # 实施缺口
    PARTICIPATION_RATE = "por"        # 参与率算法
    ADAPTIVE = "adaptive"    # 自适应算法


@dataclass
class OrderSlice:
    """订单切片数据结构"""
    slice_id: str
    parent_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str  # 'limit', 'market', 'market_if_touched'
    limit_price: Optional[float] = None
    time_in_force: str = "IOC"  # IOC, FOK, GTC
    scheduled_time: Optional[datetime] = None
    venue: str = "binance"
    urgency_level: float = 0.5  # 0-1, 1为最急迫
    hidden: bool = False
    iceberg_qty: Optional[float] = None
    
    
@dataclass 
class SlicingParams:
    """分割参数配置"""
    total_quantity: float
    target_duration_minutes: int = 30
    participation_rate: float = 0.15  # 占当前成交量的比例
    risk_aversion: float = 0.5  # 0-1, 1为最保守
    price_volatility: float = 0.02  # 历史波动率
    market_impact_factor: float = 0.001  # 市场冲击系数
    min_slice_size: float = 10.0  # 最小切片大小(USD)
    max_slice_size: float = 5000.0  # 最大切片大小(USD)
    urgency_factor: float = 0.5  # 紧急程度因子


class AdvancedOrderSlicer:
    """高级订单分割引擎"""
    
    def __init__(self):
        self.active_slices: Dict[str, List[OrderSlice]] = {}
        self.execution_stats: Dict[str, Dict] = {}
        
    async def slice_order(
        self,
        parent_id: str,
        symbol: str,
        side: str,
        total_quantity: float,
        algorithm: SlicingAlgorithm,
        params: SlicingParams,
        market_data: Optional[Dict] = None
    ) -> List[OrderSlice]:
        """
        主要的订单分割函数
        """
        logger.info(f"开始分割订单 {parent_id}: {side} {total_quantity} {symbol} 使用 {algorithm.value}")
        
        # 根据算法类型选择分割策略
        if algorithm == SlicingAlgorithm.TWAP:
            slices = await self._twap_slice(parent_id, symbol, side, total_quantity, params)
        elif algorithm == SlicingAlgorithm.VWAP:
            slices = await self._vwap_slice(parent_id, symbol, side, total_quantity, params, market_data)
        elif algorithm == SlicingAlgorithm.IMPLEMENTATION_SHORTFALL:
            slices = await self._implementation_shortfall_slice(parent_id, symbol, side, total_quantity, params, market_data)
        elif algorithm == SlicingAlgorithm.PARTICIPATION_RATE:
            slices = await self._participation_rate_slice(parent_id, symbol, side, total_quantity, params, market_data)
        elif algorithm == SlicingAlgorithm.ADAPTIVE:
            slices = await self._adaptive_slice(parent_id, symbol, side, total_quantity, params, market_data)
        else:
            raise ValueError(f"不支持的分割算法: {algorithm}")
            
        # 存储切片信息
        self.active_slices[parent_id] = slices
        
        # 优化切片调度
        slices = await self._optimize_slice_scheduling(slices, params)
        
        logger.info(f"订单分割完成: {len(slices)} 个切片")
        return slices
    
    async def _twap_slice(
        self,
        parent_id: str,
        symbol: str,
        side: str,
        total_quantity: float,
        params: SlicingParams
    ) -> List[OrderSlice]:
        """
        时间加权平均价格(TWAP)分割算法
        在指定时间段内均匀分布订单
        """
        duration_minutes = params.target_duration_minutes
        
        # 计算分割间隔(秒)
        # 平衡执行时间和市场冲击
        if total_quantity * 50000 > 100000:  # 大订单，假设50k价格
            slice_interval_seconds = 30  # 30秒间隔
        else:
            slice_interval_seconds = 60  # 60秒间隔
            
        num_slices = max(1, duration_minutes * 60 // slice_interval_seconds)
        base_slice_size = total_quantity / num_slices
        
        slices = []
        current_time = datetime.now()
        
        for i in range(num_slices):
            # 添加随机化避免模式识别
            randomization_factor = np.random.uniform(0.8, 1.2)
            slice_quantity = base_slice_size * randomization_factor
            
            # 最后一个切片包含剩余数量
            if i == num_slices - 1:
                slice_quantity = total_quantity - sum(s.quantity for s in slices)
            
            # 确保切片大小在合理范围内
            slice_quantity = max(params.min_slice_size / 50000, slice_quantity)  # 假设价格50k
            slice_quantity = min(params.max_slice_size / 50000, slice_quantity)
            
            # 动态选择订单类型
            order_type = "limit" if params.risk_aversion > 0.6 else "market"
            
            slice = OrderSlice(
                slice_id=f"{parent_id}_TWAP_{i:03d}",
                parent_id=parent_id,
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                order_type=order_type,
                scheduled_time=current_time + timedelta(seconds=i * slice_interval_seconds),
                venue="binance",
                urgency_level=0.3,  # TWAP通常不急迫
                time_in_force="IOC" if order_type == "limit" else "FOK"
            )
            
            slices.append(slice)
            
        return slices
    
    async def _vwap_slice(
        self,
        parent_id: str,
        symbol: str,
        side: str,
        total_quantity: float,
        params: SlicingParams,
        market_data: Optional[Dict] = None
    ) -> List[OrderSlice]:
        """
        成交量加权平均价格(VWAP)分割算法
        根据历史成交量模式分配订单大小
        """
        # 获取历史成交量数据（如果没有提供，使用模拟数据）
        if market_data and 'volume_profile' in market_data:
            volume_profile = market_data['volume_profile']
        else:
            # 生成典型的成交量模式（模拟数据）
            volume_profile = self._generate_typical_volume_profile(params.target_duration_minutes)
        
        # 根据成交量分布计算切片大小
        total_expected_volume = sum(volume_profile)
        
        slices = []
        current_time = datetime.now()
        remaining_quantity = total_quantity
        
        for i, period_volume in enumerate(volume_profile):
            # 按成交量比例分配
            volume_weight = period_volume / total_expected_volume
            slice_quantity = total_quantity * volume_weight * params.participation_rate
            
            # 确保不超过剩余数量
            slice_quantity = min(slice_quantity, remaining_quantity)
            
            if slice_quantity < params.min_slice_size / 50000:  # 假设价格50k
                continue
                
            # 根据成交量活跃度选择策略
            if period_volume > np.mean(volume_profile):
                order_type = "limit"  # 高成交量时段使用limit
                urgency = 0.4
            else:
                order_type = "market"  # 低成交量时段使用market
                urgency = 0.6
            
            slice = OrderSlice(
                slice_id=f"{parent_id}_VWAP_{i:03d}",
                parent_id=parent_id,
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                order_type=order_type,
                scheduled_time=current_time + timedelta(minutes=i * 5),  # 每5分钟一次
                venue="binance",
                urgency_level=urgency,
                time_in_force="IOC"
            )
            
            slices.append(slice)
            remaining_quantity -= slice_quantity
            
            if remaining_quantity <= 0:
                break
        
        # 如果还有剩余数量，添加到最后一个切片
        if remaining_quantity > 0 and slices:
            slices[-1].quantity += remaining_quantity
            
        return slices
    
    async def _implementation_shortfall_slice(
        self,
        parent_id: str,
        symbol: str,
        side: str,
        total_quantity: float,
        params: SlicingParams,
        market_data: Optional[Dict] = None
    ) -> List[OrderSlice]:
        """
        实施缺口(Implementation Shortfall)算法
        平衡市场冲击成本和时间风险
        """
        # 估算市场冲击成本和时间风险
        volatility = params.price_volatility
        market_impact = params.market_impact_factor
        
        # Almgren-Chriss模型参数
        risk_aversion = params.risk_aversion
        total_time = params.target_duration_minutes * 60  # 转换为秒
        
        # 计算最优执行轨迹
        optimal_trajectory = self._calculate_optimal_trajectory(
            total_quantity, total_time, volatility, market_impact, risk_aversion
        )
        
        slices = []
        current_time = datetime.now()
        
        for i, (time_point, target_quantity) in enumerate(optimal_trajectory):
            if i == 0:
                slice_quantity = target_quantity
            else:
                slice_quantity = target_quantity - optimal_trajectory[i-1][1]
            
            if slice_quantity <= 0:
                continue
                
            # 根据执行紧迫程度选择订单类型
            urgency = min(1.0, (i + 1) / len(optimal_trajectory) + params.urgency_factor)
            
            if urgency > 0.7:
                order_type = "market"
                tif = "FOK"
            else:
                order_type = "limit"
                tif = "IOC"
            
            slice = OrderSlice(
                slice_id=f"{parent_id}_IS_{i:03d}",
                parent_id=parent_id,
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                order_type=order_type,
                scheduled_time=current_time + timedelta(seconds=time_point),
                venue="binance",
                urgency_level=urgency,
                time_in_force=tif
            )
            
            slices.append(slice)
            
        return slices
    
    async def _participation_rate_slice(
        self,
        parent_id: str,
        symbol: str,
        side: str,
        total_quantity: float,
        params: SlicingParams,
        market_data: Optional[Dict] = None
    ) -> List[OrderSlice]:
        """
        参与率算法
        根据实时市场成交量调整下单量
        """
        participation_rate = params.participation_rate
        
        # 预估每个时间段的市场成交量
        if market_data and 'realtime_volume' in market_data:
            expected_volumes = market_data['realtime_volume']
        else:
            # 使用历史平均成交量估算
            avg_volume_per_minute = 1000000  # 假设值
            expected_volumes = [avg_volume_per_minute] * params.target_duration_minutes
        
        slices = []
        current_time = datetime.now()
        remaining_quantity = total_quantity
        
        for i, expected_volume in enumerate(expected_volumes):
            # 计算允许的最大下单量
            max_slice_quantity = expected_volume * participation_rate / 50000  # 假设价格50k
            
            # 计算实际下单量
            slice_quantity = min(max_slice_quantity, remaining_quantity)
            slice_quantity = max(params.min_slice_size / 50000, slice_quantity)
            
            if slice_quantity <= 0:
                continue
            
            # 动态调整订单类型
            if expected_volume > np.mean(expected_volumes):
                order_type = "limit"
                urgency = 0.3
            else:
                order_type = "market"
                urgency = 0.6
            
            slice = OrderSlice(
                slice_id=f"{parent_id}_POR_{i:03d}",
                parent_id=parent_id,
                symbol=symbol,
                side=side,
                quantity=slice_quantity,
                order_type=order_type,
                scheduled_time=current_time + timedelta(minutes=i),
                venue="binance",
                urgency_level=urgency,
                time_in_force="IOC"
            )
            
            slices.append(slice)
            remaining_quantity -= slice_quantity
            
            if remaining_quantity <= 0:
                break
        
        return slices
    
    async def _adaptive_slice(
        self,
        parent_id: str,
        symbol: str,
        side: str,
        total_quantity: float,
        params: SlicingParams,
        market_data: Optional[Dict] = None
    ) -> List[OrderSlice]:
        """
        自适应分割算法
        根据市场条件动态选择最优策略
        """
        # 分析市场条件
        market_regime = self._analyze_market_regime(market_data)
        
        # 根据市场状态选择基础算法
        if market_regime == "high_volatility":
            # 高波动时期，使用VWAP减少市场冲击
            base_slices = await self._vwap_slice(parent_id, symbol, side, total_quantity, params, market_data)
        elif market_regime == "trending":
            # 趋势市场，使用Implementation Shortfall快速执行
            base_slices = await self._implementation_shortfall_slice(parent_id, symbol, side, total_quantity, params, market_data)
        else:
            # 正常市场，使用TWAP
            base_slices = await self._twap_slice(parent_id, symbol, side, total_quantity, params)
        
        # 动态调整切片参数
        for slice_obj in base_slices:
            slice_obj.slice_id = slice_obj.slice_id.replace("TWAP", "ADAPTIVE").replace("VWAP", "ADAPTIVE").replace("IS", "ADAPTIVE")
            
            # 根据市场条件调整紧急程度
            if market_regime == "high_volatility":
                slice_obj.urgency_level *= 1.5  # 提高紧急程度
            elif market_regime == "low_liquidity":
                slice_obj.urgency_level *= 0.7  # 降低紧急程度
        
        return base_slices
    
    def _generate_typical_volume_profile(self, duration_minutes: int) -> List[float]:
        """生成典型的成交量分布模式"""
        # 模拟U型成交量分布（开盘和收盘时段成交量较高）
        time_points = np.linspace(0, 1, duration_minutes // 5)  # 每5分钟一个点
        
        # U型曲线：两端高，中间低
        volume_profile = []
        for t in time_points:
            # 使用二次函数生成U型
            volume = 1000000 * (2 * t**2 - 2 * t + 1.5)  # 基础成交量100万
            volume_profile.append(max(500000, volume))  # 最低50万
            
        return volume_profile
    
    def _calculate_optimal_trajectory(
        self,
        total_quantity: float,
        total_time: float,
        volatility: float,
        market_impact: float,
        risk_aversion: float
    ) -> List[Tuple[float, float]]:
        """
        计算Almgren-Chriss模型的最优执行轨迹
        """
        # 简化的Almgren-Chriss模型实现
        num_steps = min(20, int(total_time / 60))  # 最多20步，每步至少1分钟
        dt = total_time / num_steps
        
        # 计算最优执行速度
        kappa = market_impact  # 市场冲击参数
        sigma = volatility  # 波动率
        gamma = risk_aversion  # 风险厌恶参数
        
        # 最优执行策略参数
        tau = total_time
        optimal_speed = []
        
        for i in range(num_steps):
            t = i * dt
            remaining_time = tau - t
            
            # Almgren-Chriss公式（简化版）
            if remaining_time > 0:
                speed = (total_quantity / tau) * np.exp(gamma * sigma**2 * remaining_time / (2 * kappa))
            else:
                speed = total_quantity  # 最后时刻执行所有剩余量
                
            optimal_speed.append(speed)
        
        # 转换为累积执行量
        trajectory = []
        cumulative_quantity = 0
        
        for i, speed in enumerate(optimal_speed):
            t = i * dt
            cumulative_quantity += speed * dt
            cumulative_quantity = min(cumulative_quantity, total_quantity)
            trajectory.append((t, cumulative_quantity))
            
        return trajectory
    
    def _analyze_market_regime(self, market_data: Optional[Dict]) -> str:
        """分析当前市场状态"""
        if not market_data:
            return "normal"
        
        # 分析波动率
        if market_data.get('volatility', 0.02) > 0.05:
            return "high_volatility"
        
        # 分析趋势
        if market_data.get('trend_strength', 0) > 0.7:
            return "trending"
        
        # 分析流动性
        if market_data.get('bid_ask_spread', 0.001) > 0.005:
            return "low_liquidity"
        
        return "normal"
    
    async def _optimize_slice_scheduling(
        self,
        slices: List[OrderSlice],
        params: SlicingParams
    ) -> List[OrderSlice]:
        """
        优化切片调度，避免市场冲击
        """
        # 添加随机化时间间隔，避免模式识别
        for i, slice_obj in enumerate(slices):
            if slice_obj.scheduled_time:
                # 添加±10秒的随机延迟
                random_delay = np.random.uniform(-10, 10)
                slice_obj.scheduled_time += timedelta(seconds=random_delay)
        
        # 根据市场流动性调整iceberg参数
        for slice_obj in slices:
            if slice_obj.quantity > params.max_slice_size / 50000:  # 大切片
                slice_obj.iceberg_qty = slice_obj.quantity * 0.3  # 显示30%
                slice_obj.hidden = True
        
        return slices
    
    def get_execution_stats(self, parent_id: str) -> Dict:
        """获取执行统计信息"""
        return self.execution_stats.get(parent_id, {})
    
    def update_execution_stats(self, parent_id: str, slice_id: str, fill_info: Dict):
        """更新执行统计"""
        if parent_id not in self.execution_stats:
            self.execution_stats[parent_id] = {
                'total_filled': 0,
                'total_quantity': 0,
                'avg_price': 0,
                'slippage': 0,
                'fills': []
            }
        
        stats = self.execution_stats[parent_id]
        stats['fills'].append({
            'slice_id': slice_id,
            'fill_info': fill_info,
            'timestamp': datetime.now()
        })
        
        # 更新汇总统计
        stats['total_filled'] += fill_info.get('quantity', 0)
        
        # 计算平均价格和滑点
        if stats['total_filled'] > 0:
            weighted_price = sum(f['fill_info'].get('price', 0) * f['fill_info'].get('quantity', 0) 
                               for f in stats['fills'])
            stats['avg_price'] = weighted_price / stats['total_filled']