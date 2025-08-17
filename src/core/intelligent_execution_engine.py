"""
Intelligent Execution Engine
智能执行引擎主控制器 - 整合所有执行优化组件
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid

# 导入所有执行组件
from .advanced_order_slicer import (
    AdvancedOrderSlicer, SlicingAlgorithm, SlicingParams, OrderSlice
)
from .smart_order_router import SmartOrderRouter, RouteResult
from .execution_risk_manager import ExecutionRiskManager, RiskLimits
from .microstructure_optimizer import (
    MicrostructureOptimizer, OrderBookSnapshot, MicrostructureSignal, 
    ExecutionStrategy, ExecutionTiming
)
from .execution_analytics import ExecutionAnalyzer

logger = logging.getLogger(__name__)


class ExecutionMode(Enum):
    """执行模式"""
    CONSERVATIVE = "conservative"   # 保守模式 - 优先控制风险
    BALANCED = "balanced"          # 平衡模式 - 平衡成本和风险
    AGGRESSIVE = "aggressive"      # 激进模式 - 优先执行速度
    STEALTH = "stealth"           # 隐蔽模式 - 最小化市场冲击


@dataclass
class ExecutionRequest:
    """执行请求"""
    request_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    target_quantity: float
    execution_mode: ExecutionMode
    max_execution_time_minutes: int = 60
    target_arrival_price: Optional[float] = None
    max_slippage_bps: Optional[float] = None
    allow_partial_fill: bool = True
    priority: int = 5  # 1-10, 10为最高优先级
    client_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionStatus:
    """执行状态"""
    request_id: str
    status: str  # 'pending', 'executing', 'completed', 'failed', 'cancelled'
    filled_quantity: float = 0.0
    average_price: float = 0.0
    total_fees: float = 0.0
    slippage_bps: float = 0.0
    execution_time_seconds: float = 0.0
    active_slices: int = 0
    completed_slices: int = 0
    error_message: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)


class IntelligentExecutionEngine:
    """智能执行引擎 - 统一执行管理系统"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 初始化核心组件
        self.order_slicer = AdvancedOrderSlicer()
        self.order_router = SmartOrderRouter()
        self.risk_manager = ExecutionRiskManager(RiskLimits())
        self.microstructure_optimizer = MicrostructureOptimizer()
        self.execution_analyzer = ExecutionAnalyzer()
        
        # 执行状态跟踪
        self.active_executions: Dict[str, ExecutionRequest] = {}
        self.execution_status: Dict[str, ExecutionStatus] = {}
        self.execution_history: List[Dict] = []
        
        # 性能统计
        self.execution_metrics = {
            'total_executions': 0,
            'successful_executions': 0,
            'avg_slippage_bps': 0.0,
            'avg_execution_time_seconds': 0.0,
            'total_volume_usd': 0.0
        }
        
        # 异步任务管理
        self.execution_tasks: Dict[str, asyncio.Task] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        
    async def initialize(self, symbols: List[str]):
        """初始化执行引擎"""
        logger.info("初始化智能执行引擎...")
        
        # 启动市场数据收集
        await self.order_router.start_market_data_collection(symbols)
        
        # 启动监控任务
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info(f"执行引擎初始化完成，支持 {len(symbols)} 个交易对")
    
    async def submit_execution_request(self, request: ExecutionRequest) -> str:
        """
        提交执行请求
        
        Returns:
            request_id: 执行请求ID
        """
        logger.info(f"收到执行请求: {request.symbol} {request.side} {request.target_quantity}")
        
        # 预执行风险检查
        is_valid, rejection_reason = await self.risk_manager.validate_order(
            request.request_id,
            request.symbol,
            request.side,
            request.target_quantity,
            "auto"  # 自动选择交易所
        )
        
        if not is_valid:
            logger.warning(f"执行请求被拒绝: {rejection_reason}")
            
            # 创建失败状态
            self.execution_status[request.request_id] = ExecutionStatus(
                request_id=request.request_id,
                status="failed",
                error_message=rejection_reason
            )
            return request.request_id
        
        # 存储执行请求
        self.active_executions[request.request_id] = request
        
        # 创建初始状态
        self.execution_status[request.request_id] = ExecutionStatus(
            request_id=request.request_id,
            status="pending"
        )
        
        # 启动执行任务
        execution_task = asyncio.create_task(
            self._execute_request(request)
        )
        self.execution_tasks[request.request_id] = execution_task
        
        return request.request_id
    
    async def _execute_request(self, request: ExecutionRequest):
        """执行单个请求的主流程"""
        request_id = request.request_id
        
        try:
            logger.info(f"开始执行请求 {request_id}")
            
            # 更新状态为执行中
            self.execution_status[request_id].status = "executing"
            self.execution_status[request_id].last_updated = datetime.now()
            
            # Step 1: 市场微观结构分析
            market_signals = await self._analyze_market_microstructure(request)
            
            # Step 2: 确定执行策略
            execution_strategy = await self._determine_execution_strategy(request, market_signals)
            
            # Step 3: 订单分割
            order_slices = await self._slice_order(request, execution_strategy)
            
            # Step 4: 智能路由
            routing_plan = await self._create_routing_plan(order_slices)
            
            # Step 5: 执行订单切片
            await self._execute_slices(request_id, routing_plan)
            
            # Step 6: 执行完成处理
            await self._complete_execution(request_id)
            
        except Exception as e:
            logger.error(f"执行请求 {request_id} 失败: {e}")
            await self._handle_execution_error(request_id, str(e))
    
    async def _analyze_market_microstructure(
        self, 
        request: ExecutionRequest
    ) -> List[MicrostructureSignal]:
        """分析市场微观结构"""
        
        logger.debug(f"分析 {request.symbol} 的市场微观结构")
        
        # 获取最新订单簿数据（这里应该从真实数据源获取）
        order_book = await self._get_order_book_snapshot(request.symbol)
        
        signals = []
        
        if order_book:
            # 深度分析
            depth_signal = await self.microstructure_optimizer.analyze_order_book(
                order_book, self.microstructure_optimizer.OrderBookAnalysisType.DEPTH_ANALYSIS
            )
            signals.append(depth_signal)
            
            # 失衡分析
            imbalance_signal = await self.microstructure_optimizer.analyze_order_book(
                order_book, self.microstructure_optimizer.OrderBookAnalysisType.IMBALANCE_ANALYSIS
            )
            signals.append(imbalance_signal)
            
            # 流量分析
            flow_signal = await self.microstructure_optimizer.analyze_order_book(
                order_book, self.microstructure_optimizer.OrderBookAnalysisType.FLOW_ANALYSIS
            )
            signals.append(flow_signal)
            
            # 价差分析
            spread_signal = await self.microstructure_optimizer.analyze_order_book(
                order_book, self.microstructure_optimizer.OrderBookAnalysisType.SPREAD_ANALYSIS
            )
            signals.append(spread_signal)
        
        logger.debug(f"生成 {len(signals)} 个微观结构信号")
        return signals
    
    async def _determine_execution_strategy(
        self, 
        request: ExecutionRequest,
        market_signals: List[MicrostructureSignal]
    ) -> Dict:
        """确定执行策略"""
        
        # 基于执行模式选择基础策略
        base_strategy = self._get_base_strategy(request.execution_mode)
        
        # 基于微观结构信号调整策略
        timing_recommendation = await self.microstructure_optimizer.determine_execution_timing(
            request.symbol,
            request.side,
            request.target_quantity,
            market_signals
        )
        
        # 综合决策
        strategy = {
            'slicing_algorithm': base_strategy['slicing_algorithm'],
            'execution_strategy': timing_recommendation.optimal_strategy,
            'urgency_level': self._calculate_urgency_level(request, timing_recommendation),
            'expected_slippage_bps': timing_recommendation.expected_slippage_bps,
            'confidence_score': timing_recommendation.confidence_score,
            'action': timing_recommendation.action
        }
        
        logger.info(f"执行策略: {strategy['slicing_algorithm'].value}, 紧急程度: {strategy['urgency_level']:.2f}")
        return strategy
    
    def _get_base_strategy(self, execution_mode: ExecutionMode) -> Dict:
        """根据执行模式获取基础策略"""
        
        strategies = {
            ExecutionMode.CONSERVATIVE: {
                'slicing_algorithm': SlicingAlgorithm.TWAP,
                'participation_rate': 0.1,
                'risk_aversion': 0.8
            },
            ExecutionMode.BALANCED: {
                'slicing_algorithm': SlicingAlgorithm.VWAP,
                'participation_rate': 0.15,
                'risk_aversion': 0.5
            },
            ExecutionMode.AGGRESSIVE: {
                'slicing_algorithm': SlicingAlgorithm.IMPLEMENTATION_SHORTFALL,
                'participation_rate': 0.25,
                'risk_aversion': 0.2
            },
            ExecutionMode.STEALTH: {
                'slicing_algorithm': SlicingAlgorithm.PARTICIPATION_RATE,
                'participation_rate': 0.05,
                'risk_aversion': 0.9
            }
        }
        
        return strategies.get(execution_mode, strategies[ExecutionMode.BALANCED])
    
    def _calculate_urgency_level(
        self, 
        request: ExecutionRequest, 
        timing: ExecutionTiming
    ) -> float:
        """计算执行紧急程度"""
        
        # 基础紧急程度
        base_urgency = request.priority / 10.0
        
        # 时间因素
        time_factor = 1.0
        if request.max_execution_time_minutes < 30:
            time_factor = 1.5
        elif request.max_execution_time_minutes < 15:
            time_factor = 2.0
        
        # 市场信号因素
        signal_factor = timing.confidence_score
        
        # 综合计算
        urgency = min(1.0, base_urgency * time_factor * signal_factor)
        return urgency
    
    async def _slice_order(
        self, 
        request: ExecutionRequest, 
        strategy: Dict
    ) -> List[OrderSlice]:
        """订单分割"""
        
        # 配置分割参数
        slicing_params = SlicingParams(
            total_quantity=request.target_quantity,
            target_duration_minutes=request.max_execution_time_minutes,
            participation_rate=strategy.get('participation_rate', 0.15),
            risk_aversion=strategy.get('risk_aversion', 0.5),
            urgency_factor=strategy['urgency_level']
        )
        
        # 执行分割
        slices = await self.order_slicer.slice_order(
            parent_id=request.request_id,
            symbol=request.symbol,
            side=request.side,
            total_quantity=request.target_quantity,
            algorithm=strategy['slicing_algorithm'],
            params=slicing_params
        )
        
        logger.info(f"订单分割完成: {len(slices)} 个切片")
        return slices
    
    async def _create_routing_plan(self, order_slices: List[OrderSlice]) -> Dict:
        """创建路由计划"""
        
        routing_plan = {
            'slices': [],
            'total_venues': 0,
            'expected_cost': 0.0
        }
        
        for slice_obj in order_slices:
            # 为每个切片寻找最优路由
            route = await self.order_router.find_best_route(
                symbol=slice_obj.symbol,
                side=slice_obj.side,
                quantity=slice_obj.quantity,
                max_venues=2,  # 最多使用2个交易所
                require_full_fill=False
            )
            
            routing_plan['slices'].append({
                'slice': slice_obj,
                'route': route
            })
            
            routing_plan['expected_cost'] += route.total_cost
        
        routing_plan['total_venues'] = len(set(
            venue for slice_route in routing_plan['slices'] 
            for venue in slice_route['route'].venues
        ))
        
        logger.info(f"路由计划: {len(routing_plan['slices'])} 个切片, {routing_plan['total_venues']} 个交易所")
        return routing_plan
    
    async def _execute_slices(self, request_id: str, routing_plan: Dict):
        """执行订单切片"""
        
        status = self.execution_status[request_id]
        status.active_slices = len(routing_plan['slices'])
        
        # 并发执行切片
        slice_tasks = []
        for slice_route in routing_plan['slices']:
            task = asyncio.create_task(
                self._execute_single_slice(request_id, slice_route)
            )
            slice_tasks.append(task)
        
        # 等待所有切片完成
        slice_results = await asyncio.gather(*slice_tasks, return_exceptions=True)
        
        # 处理执行结果
        successful_slices = 0
        total_filled = 0.0
        total_value = 0.0
        total_fees = 0.0
        
        for i, result in enumerate(slice_results):
            if isinstance(result, Exception):
                logger.error(f"切片执行失败: {result}")
                continue
            
            if result and result.get('filled_quantity', 0) > 0:
                successful_slices += 1
                filled_qty = result['filled_quantity']
                fill_price = result['average_price']
                
                total_filled += filled_qty
                total_value += filled_qty * fill_price
                total_fees += result.get('fees', 0)
        
        # 更新执行状态
        status.completed_slices = successful_slices
        status.filled_quantity = total_filled
        status.average_price = total_value / total_filled if total_filled > 0 else 0
        status.total_fees = total_fees
        
        # 计算滑点
        request = self.active_executions[request_id]
        if request.target_arrival_price and status.average_price > 0:
            if request.side == 'BUY':
                slippage = (status.average_price - request.target_arrival_price) / request.target_arrival_price * 10000
            else:
                slippage = (request.target_arrival_price - status.average_price) / request.target_arrival_price * 10000
            status.slippage_bps = slippage
        
        status.execution_time_seconds = (datetime.now() - status.last_updated).total_seconds()
        status.last_updated = datetime.now()
    
    async def _execute_single_slice(
        self, 
        request_id: str, 
        slice_route: Dict
    ) -> Optional[Dict]:
        """执行单个订单切片"""
        
        slice_obj = slice_route['slice']
        route = slice_route['route']
        
        try:
            # 模拟订单执行（实际实现需要调用真实交易API）
            await asyncio.sleep(0.1)  # 模拟执行延迟
            
            # 模拟执行结果
            execution_result = {
                'slice_id': slice_obj.slice_id,
                'filled_quantity': slice_obj.quantity,
                'average_price': route.expected_prices[0] if route.expected_prices else 50000,
                'fees': route.expected_fees[0] if route.expected_fees else 10.0,
                'venue': route.venues[0] if route.venues else 'binance',
                'execution_time_ms': 100,
                'status': 'FILLED'
            }
            
            # 更新风险管理器
            await self.risk_manager.update_fill(
                slice_obj.slice_id,
                execution_result['average_price'],
                execution_result['filled_quantity'],
                execution_result['fees'],
                execution_result['venue']
            )
            
            return execution_result
            
        except Exception as e:
            logger.error(f"切片执行失败 {slice_obj.slice_id}: {e}")
            return None
    
    async def _complete_execution(self, request_id: str):
        """完成执行处理"""
        
        status = self.execution_status[request_id]
        request = self.active_executions[request_id]
        
        # 判断执行是否成功
        fill_rate = status.filled_quantity / request.target_quantity
        
        if fill_rate >= 0.95:  # 95%以上视为成功
            status.status = "completed"
            self.execution_metrics['successful_executions'] += 1
        elif fill_rate >= 0.5 and request.allow_partial_fill:  # 部分成交
            status.status = "completed"
            self.execution_metrics['successful_executions'] += 1
        else:
            status.status = "failed"
            status.error_message = f"成交率过低: {fill_rate:.2%}"
        
        # 更新性能统计
        self.execution_metrics['total_executions'] += 1
        self.execution_metrics['avg_slippage_bps'] = (
            self.execution_metrics['avg_slippage_bps'] * (self.execution_metrics['total_executions'] - 1) +
            abs(status.slippage_bps)
        ) / self.execution_metrics['total_executions']
        
        self.execution_metrics['avg_execution_time_seconds'] = (
            self.execution_metrics['avg_execution_time_seconds'] * (self.execution_metrics['total_executions'] - 1) +
            status.execution_time_seconds
        ) / self.execution_metrics['total_executions']
        
        notional_value = status.filled_quantity * status.average_price
        self.execution_metrics['total_volume_usd'] += notional_value
        
        # 记录执行历史
        execution_record = {
            'request_id': request_id,
            'symbol': request.symbol,
            'side': request.side,
            'target_quantity': request.target_quantity,
            'filled_quantity': status.filled_quantity,
            'average_price': status.average_price,
            'slippage_bps': status.slippage_bps,
            'execution_time_seconds': status.execution_time_seconds,
            'total_fees': status.total_fees,
            'status': status.status,
            'completion_time': datetime.now().isoformat()
        }
        self.execution_history.append(execution_record)
        
        # 清理活跃执行
        if request_id in self.active_executions:
            del self.active_executions[request_id]
        
        if request_id in self.execution_tasks:
            self.execution_tasks[request_id].cancel()
            del self.execution_tasks[request_id]
        
        logger.info(f"执行完成 {request_id}: {status.status}, 成交率: {fill_rate:.2%}")
    
    async def _handle_execution_error(self, request_id: str, error_message: str):
        """处理执行错误"""
        
        status = self.execution_status[request_id]
        status.status = "failed"
        status.error_message = error_message
        status.last_updated = datetime.now()
        
        # 清理资源
        if request_id in self.active_executions:
            del self.active_executions[request_id]
        
        if request_id in self.execution_tasks:
            self.execution_tasks[request_id].cancel()
            del self.execution_tasks[request_id]
        
        logger.error(f"执行失败 {request_id}: {error_message}")
    
    async def _get_order_book_snapshot(self, symbol: str) -> Optional[OrderBookSnapshot]:
        """获取订单簿快照（模拟数据）"""
        
        # 实际实现中应该从真实数据源获取
        from .microstructure_optimizer import OrderBookLevel
        
        base_price = 50000
        spread = 2
        
        bids = [
            OrderBookLevel(base_price - spread/2, 5.0, 3),
            OrderBookLevel(base_price - spread/2 - 1, 3.0, 2),
            OrderBookLevel(base_price - spread/2 - 2, 4.0, 5),
        ]
        
        asks = [
            OrderBookLevel(base_price + spread/2, 4.0, 2),
            OrderBookLevel(base_price + spread/2 + 1, 6.0, 4),
            OrderBookLevel(base_price + spread/2 + 2, 3.0, 1),
        ]
        
        return OrderBookSnapshot(
            symbol=symbol,
            venue="binance",
            bids=bids,
            asks=asks
        )
    
    async def _monitoring_loop(self):
        """监控循环"""
        
        while True:
            try:
                # 检查超时的执行
                current_time = datetime.now()
                timeout_executions = []
                
                for request_id, request in self.active_executions.items():
                    execution_time = (current_time - request.created_at).total_seconds() / 60
                    if execution_time > request.max_execution_time_minutes:
                        timeout_executions.append(request_id)
                
                # 处理超时执行
                for request_id in timeout_executions:
                    await self._handle_execution_error(request_id, "执行超时")
                
                # 更新性能指标
                await self._update_performance_metrics()
                
                await asyncio.sleep(5)  # 每5秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                await asyncio.sleep(10)
    
    async def _update_performance_metrics(self):
        """更新性能指标"""
        
        # 更新路由器性能
        routing_stats = self.order_router.get_routing_stats()
        
        # 更新风险管理指标
        risk_summary = self.risk_manager.get_risk_summary()
        
        # 记录性能日志
        if len(self.execution_history) > 0 and len(self.execution_history) % 10 == 0:
            logger.info(f"执行性能: 总量 {self.execution_metrics['total_executions']}, "
                       f"成功率 {self.execution_metrics['successful_executions']/self.execution_metrics['total_executions']:.2%}, "
                       f"平均滑点 {self.execution_metrics['avg_slippage_bps']:.2f}bps")
    
    # 公共接口方法
    
    async def cancel_execution(self, request_id: str) -> bool:
        """取消执行请求"""
        
        if request_id not in self.execution_status:
            return False
        
        status = self.execution_status[request_id]
        if status.status in ['completed', 'failed', 'cancelled']:
            return False
        
        # 取消执行任务
        if request_id in self.execution_tasks:
            self.execution_tasks[request_id].cancel()
            del self.execution_tasks[request_id]
        
        # 更新状态
        status.status = "cancelled"
        status.last_updated = datetime.now()
        
        # 清理资源
        if request_id in self.active_executions:
            del self.active_executions[request_id]
        
        logger.info(f"执行请求已取消: {request_id}")
        return True
    
    def get_execution_status(self, request_id: str) -> Optional[ExecutionStatus]:
        """获取执行状态"""
        return self.execution_status.get(request_id)
    
    def get_active_executions(self) -> List[str]:
        """获取活跃执行列表"""
        return list(self.active_executions.keys())
    
    def get_performance_metrics(self) -> Dict:
        """获取性能指标"""
        return {
            'execution_metrics': self.execution_metrics.copy(),
            'routing_stats': self.order_router.get_routing_stats(),
            'risk_summary': self.risk_manager.get_risk_summary(),
            'microstructure_summary': {
                symbol: self.microstructure_optimizer.get_microstructure_summary(symbol)
                for symbol in ['BTCUSDT', 'ETHUSDT']  # 示例符号
            }
        }
    
    async def shutdown(self):
        """关闭执行引擎"""
        logger.info("关闭智能执行引擎...")
        
        # 取消所有活跃执行
        for request_id in list(self.active_executions.keys()):
            await self.cancel_execution(request_id)
        
        # 停止监控任务
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        # 停止市场数据收集
        await self.order_router.stop_market_data_collection()
        
        logger.info("执行引擎已关闭")


# 工厂函数
def create_execution_request(
    symbol: str,
    side: str,
    quantity: float,
    execution_mode: ExecutionMode = ExecutionMode.BALANCED,
    max_execution_time_minutes: int = 30,
    target_arrival_price: Optional[float] = None,
    max_slippage_bps: Optional[float] = None,
    priority: int = 5,
    client_id: Optional[str] = None
) -> ExecutionRequest:
    """创建执行请求的工厂函数"""
    
    request_id = f"EXE_{uuid.uuid4().hex[:8]}"
    
    return ExecutionRequest(
        request_id=request_id,
        symbol=symbol,
        side=side.upper(),
        target_quantity=quantity,
        execution_mode=execution_mode,
        max_execution_time_minutes=max_execution_time_minutes,
        target_arrival_price=target_arrival_price,
        max_slippage_bps=max_slippage_bps,
        priority=priority,
        client_id=client_id
    )