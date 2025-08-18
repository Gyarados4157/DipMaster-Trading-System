"""
DipMaster Execution Orchestrator
DipMaster执行编排器 - 整合所有执行管理组件

核心功能:
1. 整合专业执行系统、多交易所路由、质量监控
2. 基于组合目标自动生成和管理执行计划
3. 持续运行调度，自动处理目标仓位变化
4. 实时风险管理和紧急干预机制
5. 完整的执行报告和性能分析
6. DipMaster专用执行优化和监控
"""

import asyncio
import logging
import json
import time
import uuid
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import numpy as np

# 导入所有执行组件
from .professional_execution_system import (
    ExecutionEngine, ContinuousExecutionScheduler, TargetPosition,
    OrderUrgency, ExecutionReport, create_dipmaster_target_position
)
from .enhanced_multi_venue_router import (
    EnhancedMultiVenueRouter, OptimalRoute
)
from .execution_quality_analyzer import (
    ExecutionQualityMonitor, ExecutionMetrics, TCAReport
)

logger = logging.getLogger(__name__)


class OrchestrationMode(Enum):
    """编排模式"""
    RESEARCH = "research"           # 研究模式：仅分析不执行
    SIMULATION = "simulation"       # 模拟模式：纸上交易
    SEMI_LIVE = "semi_live"        # 半实盘：小量真实交易
    PRODUCTION = "production"       # 生产模式：完整实盘交易


class ExecutionPriority(Enum):
    """执行优先级"""
    LOW = 1
    NORMAL = 2  
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class PortfolioTarget:
    """组合目标"""
    target_id: str
    symbol: str
    current_position_usd: float
    target_position_usd: float
    
    # 执行参数
    urgency: OrderUrgency = OrderUrgency.MEDIUM
    max_execution_time_minutes: int = 60
    max_slippage_bps: float = 50
    preferred_venues: List[str] = field(default_factory=list)
    
    # DipMaster上下文
    dipmaster_context: Optional[Dict[str, Any]] = None
    
    # 优先级和约束
    priority: ExecutionPriority = ExecutionPriority.NORMAL
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    # 状态跟踪
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, executing, completed, failed, cancelled


@dataclass
class ExecutionSession:
    """执行会话"""
    session_id: str
    portfolio_target: PortfolioTarget
    
    # 执行计划
    execution_plan: Dict[str, Any]
    optimal_route: Optional[OptimalRoute] = None
    
    # 进度跟踪
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    current_status: str = "planning"  # planning, routing, executing, completed, failed
    
    # 执行结果
    execution_report: Optional[ExecutionReport] = None
    quality_metrics: Optional[ExecutionMetrics] = None
    route_execution_result: Optional[Dict[str, Any]] = None
    
    # 异常和重试
    error_messages: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3


class DipMasterExecutionOrchestrator:
    """DipMaster执行编排器"""
    
    def __init__(self, mode: OrchestrationMode = OrchestrationMode.SIMULATION):
        self.mode = mode
        self.paper_trading = mode in [OrchestrationMode.RESEARCH, OrchestrationMode.SIMULATION]
        
        # 核心组件
        self.execution_engine = ExecutionEngine(paper_trading=self.paper_trading)
        self.scheduler = ContinuousExecutionScheduler(self.execution_engine)
        self.router = EnhancedMultiVenueRouter(paper_trading=self.paper_trading)
        self.quality_monitor = ExecutionQualityMonitor()
        
        # 执行状态
        self.running = False
        self.portfolio_targets = {}  # target_id -> PortfolioTarget
        self.active_sessions = {}   # session_id -> ExecutionSession
        self.execution_history = deque(maxlen=1000)
        
        # 性能统计
        self.performance_stats = defaultdict(list)
        self.daily_stats = defaultdict(dict)
        
        # 风险管理
        self.risk_limits = {
            'max_concurrent_executions': 5,
            'max_daily_volume_usd': 100000,
            'max_single_execution_usd': 20000,
            'max_total_slippage_bps': 100
        }
        self.current_exposure = defaultdict(float)
        
        # 回调函数
        self.execution_callbacks = []
        
    async def start(self):
        """启动执行编排器"""
        if self.running:
            logger.warning("执行编排器已在运行")
            return
            
        logger.info(f"启动DipMaster执行编排器 (模式: {self.mode.value})")
        
        # 启动所有组件
        await self.scheduler.start_continuous_execution()
        await self.router.start()
        await self.quality_monitor.start_monitoring()
        
        # 启动主编排循环
        self.running = True
        asyncio.create_task(self._orchestration_loop())
        
        logger.info("DipMaster执行编排器启动完成")
    
    async def stop(self):
        """停止执行编排器"""
        logger.info("停止DipMaster执行编排器")
        
        self.running = False
        
        # 停止所有组件
        await self.scheduler.stop_continuous_execution()
        await self.router.stop()
        await self.quality_monitor.stop_monitoring()
        
        logger.info("DipMaster执行编排器已停止")
    
    def add_portfolio_targets(self, targets: List[PortfolioTarget]):
        """添加组合目标"""
        for target in targets:
            self.portfolio_targets[target.target_id] = target
            logger.info(f"添加组合目标: {target.target_id} - {target.symbol} ${target.target_position_usd:.0f}")
    
    def update_portfolio_target(self, target_id: str, updates: Dict[str, Any]):
        """更新组合目标"""
        if target_id in self.portfolio_targets:
            target = self.portfolio_targets[target_id]
            for key, value in updates.items():
                if hasattr(target, key):
                    setattr(target, key, value)
            target.last_updated = datetime.now()
            logger.info(f"更新组合目标: {target_id}")
    
    def remove_portfolio_target(self, target_id: str):
        """移除组合目标"""
        if target_id in self.portfolio_targets:
            del self.portfolio_targets[target_id]
            logger.info(f"移除组合目标: {target_id}")
    
    async def execute_portfolio_target(self, target_id: str) -> ExecutionSession:
        """执行组合目标"""
        if target_id not in self.portfolio_targets:
            raise ValueError(f"组合目标不存在: {target_id}")
        
        target = self.portfolio_targets[target_id]
        session_id = f"EXEC_{target_id}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"开始执行组合目标: {session_id}")
        
        # 创建执行会话
        session = ExecutionSession(
            session_id=session_id,
            portfolio_target=target
        )
        
        self.active_sessions[session_id] = session
        
        try:
            # 阶段1: 执行计划制定
            session.current_status = "planning"
            execution_plan = await self._create_execution_plan(target)
            session.execution_plan = execution_plan
            
            logger.info(f"执行计划完成: {session_id} - 策略: {execution_plan['strategy']}")
            
            # 阶段2: 路由优化
            session.current_status = "routing"
            if execution_plan['use_multi_venue_routing']:
                optimal_route = await self._optimize_execution_route(target, execution_plan)
                session.optimal_route = optimal_route
                
                logger.info(f"路由优化完成: {session_id} - {optimal_route.venue_count}个交易所")
            
            # 阶段3: 执行执行
            session.current_status = "executing"
            execution_result = await self._execute_with_monitoring(session)
            
            session.execution_report = execution_result.get('execution_report')
            session.route_execution_result = execution_result.get('route_result')
            
            # 阶段4: 质量分析
            if session.execution_report:
                quality_metrics = await self._analyze_execution_quality(session)
                session.quality_metrics = quality_metrics
            
            session.current_status = "completed"
            session.end_time = datetime.now()
            
            logger.info(f"执行完成: {session_id}")
            
            # 触发回调
            await self._trigger_execution_callbacks(session, "completed")
            
            return session
            
        except Exception as e:
            session.current_status = "failed"
            session.error_messages.append(str(e))
            session.end_time = datetime.now()
            
            logger.error(f"执行失败: {session_id} - {e}")
            
            # 触发错误回调
            await self._trigger_execution_callbacks(session, "failed")
            
            # 考虑重试
            if session.retry_count < session.max_retries:
                logger.info(f"准备重试执行: {session_id} (第{session.retry_count + 1}次)")
                session.retry_count += 1
                session.current_status = "pending"
                # 可以在这里添加延迟重试逻辑
            
            raise
        
        finally:
            # 清理和存储
            self.execution_history.append(session)
            self._update_performance_stats(session)
    
    async def _create_execution_plan(self, target: PortfolioTarget) -> Dict[str, Any]:
        """创建执行计划"""
        position_delta = target.target_position_usd - target.current_position_usd
        
        plan = {
            'target_id': target.target_id,
            'symbol': target.symbol,
            'side': 'BUY' if position_delta > 0 else 'SELL',
            'size_usd': abs(position_delta),
            'urgency': target.urgency,
            'max_execution_time_minutes': target.max_execution_time_minutes,
            'max_slippage_bps': target.max_slippage_bps,
            'created_at': datetime.now()
        }
        
        # 策略选择
        if target.dipmaster_context:
            if target.dipmaster_context.get('timing_critical'):
                plan['strategy'] = 'dipmaster_15min'
                plan['algorithm'] = 'DIPMASTER_15MIN'
            elif target.dipmaster_context.get('dip_signal'):
                plan['strategy'] = 'dipmaster_dip_buy'  
                plan['algorithm'] = 'DIPMASTER_DIP_BUY'
            else:
                plan['strategy'] = 'adaptive'
                plan['algorithm'] = 'implementation_shortfall'
        else:
            # 根据规模和紧急程度选择策略
            if abs(position_delta) > 10000:
                plan['strategy'] = 'vwap'
                plan['algorithm'] = 'VWAP'
            elif target.urgency.value >= 0.8:
                plan['strategy'] = 'implementation_shortfall'
                plan['algorithm'] = 'IMPLEMENTATION_SHORTFALL'
            else:
                plan['strategy'] = 'twap'
                plan['algorithm'] = 'TWAP'
        
        # 多交易所路由决策
        plan['use_multi_venue_routing'] = (
            abs(position_delta) > 5000 or  # 大额订单
            target.priority in [ExecutionPriority.HIGH, ExecutionPriority.CRITICAL] or  # 高优先级
            len(target.preferred_venues) > 1  # 指定多个交易所
        )
        
        return plan
    
    async def _optimize_execution_route(self, target: PortfolioTarget, plan: Dict[str, Any]) -> OptimalRoute:
        """优化执行路由"""
        routing_strategy = "balanced"
        
        # 根据目标选择路由策略
        if target.urgency.value >= 0.8:
            routing_strategy = "speed"
        elif plan['size_usd'] > 15000:
            routing_strategy = "cost"
        
        quantity_estimate = plan['size_usd'] / 50000  # 假设价格，实际应从市场数据获取
        
        optimal_route = await self.router.route_order(
            symbol=target.symbol,
            side=plan['side'],
            quantity=quantity_estimate,
            strategy=routing_strategy,
            max_venues=min(3, len(target.preferred_venues) if target.preferred_venues else 3),
            max_slippage_bps=target.max_slippage_bps
        )
        
        return optimal_route
    
    async def _execute_with_monitoring(self, session: ExecutionSession) -> Dict[str, Any]:
        """带监控的执行"""
        target = session.portfolio_target
        plan = session.execution_plan
        
        results = {}
        
        # 风险检查
        self._check_execution_risks(session)
        
        if session.optimal_route:
            # 多交易所执行
            logger.info(f"使用多交易所路由执行: {session.session_id}")
            
            route_result = await self.router.execute_route(session.optimal_route)
            results['route_result'] = route_result
            
            # 如果路由执行成功率不够，使用传统执行作为补充
            if route_result.get('fill_rate', 0) < 0.95:
                logger.warning(f"路由执行不完整，启用补充执行: {session.session_id}")
                remaining_target = self._create_remaining_target(target, route_result)
                if remaining_target:
                    execution_report = await self.execution_engine.execute_target_position(remaining_target)
                    results['execution_report'] = execution_report
            else:
                # 将路由结果转换为执行报告格式
                results['execution_report'] = self._convert_route_to_execution_report(session.optimal_route, route_result)
        else:
            # 传统单一执行
            logger.info(f"使用传统执行引擎: {session.session_id}")
            
            dipmaster_target = create_dipmaster_target_position(
                symbol=target.symbol,
                target_size_usd=target.target_position_usd,
                current_size_usd=target.current_position_usd,
                timing_critical=target.dipmaster_context.get('timing_critical', False) if target.dipmaster_context else False,
                dip_signal=target.dipmaster_context.get('dip_signal', False) if target.dipmaster_context else False,
                rsi_value=target.dipmaster_context.get('rsi_value') if target.dipmaster_context else None,
                urgency=target.urgency
            )
            
            execution_report = await self.execution_engine.execute_target_position(dipmaster_target)
            results['execution_report'] = execution_report
        
        return results
    
    async def _analyze_execution_quality(self, session: ExecutionSession) -> ExecutionMetrics:
        """分析执行质量"""
        execution_report = session.execution_report
        
        if not execution_report:
            return None
        
        # 构建执行数据
        execution_data = {
            'session_id': session.session_id,
            'symbol': execution_report.symbol,
            'side': 'BUY' if execution_report.executed_size_usd > 0 else 'SELL',
            'algorithm': execution_report.execution_algorithm,
            'target_quantity': execution_report.target_quantity,
            'executed_quantity': execution_report.executed_quantity,
            'execution_start': execution_report.execution_start,
            'execution_end': execution_report.execution_end,
            'fills': [asdict(fill) for fill in execution_report.fills],
            'arrival_price': execution_report.fills[0].arrival_price if execution_report.fills else 0,
            'total_fees_usd': execution_report.total_fees_usd,
            'avg_price': execution_report.executed_size_usd / execution_report.executed_quantity if execution_report.executed_quantity > 0 else 0
        }
        
        # 市场上下文
        market_context = {
            'volume_during_execution': 1000000,  # 简化处理
        }
        
        # DipMaster上下文
        if session.portfolio_target.dipmaster_context:
            market_context.update(session.portfolio_target.dipmaster_context)
        
        # 执行质量分析
        metrics, alerts = await self.quality_monitor.analyze_execution_realtime(
            execution_data=execution_data,
            market_context=market_context
        )
        
        if alerts:
            logger.warning(f"执行质量告警 {session.session_id}: {alerts}")
        
        return metrics
    
    def _check_execution_risks(self, session: ExecutionSession):
        """检查执行风险"""
        target = session.portfolio_target
        plan = session.execution_plan
        
        # 检查并发执行数量
        active_count = len([s for s in self.active_sessions.values() 
                           if s.current_status == "executing"])
        if active_count >= self.risk_limits['max_concurrent_executions']:
            raise RuntimeError(f"超过最大并发执行数量: {active_count}")
        
        # 检查单笔执行规模
        if plan['size_usd'] > self.risk_limits['max_single_execution_usd']:
            raise RuntimeError(f"单笔执行规模超限: ${plan['size_usd']:.0f}")
        
        # 检查日度交易量
        today = datetime.now().date()
        daily_volume = sum(
            abs(s.portfolio_target.target_position_usd - s.portfolio_target.current_position_usd)
            for s in self.execution_history
            if s.start_time.date() == today and s.current_status == "completed"
        )
        
        if daily_volume + plan['size_usd'] > self.risk_limits['max_daily_volume_usd']:
            raise RuntimeError(f"日度交易量超限: ${daily_volume + plan['size_usd']:.0f}")
    
    def _create_remaining_target(self, original_target: PortfolioTarget, route_result: Dict[str, Any]) -> Optional[TargetPosition]:
        """创建剩余目标（用于补充执行）"""
        filled_usd = route_result.get('filled_value', 0)
        target_usd = abs(original_target.target_position_usd - original_target.current_position_usd)
        
        remaining_usd = target_usd - filled_usd
        
        if remaining_usd < 100:  # 少于100美元不需要补充执行
            return None
        
        return create_dipmaster_target_position(
            symbol=original_target.symbol,
            target_size_usd=remaining_usd,
            current_size_usd=0,
            urgency=OrderUrgency.HIGH  # 补充执行使用高紧急度
        )
    
    def _convert_route_to_execution_report(self, route: OptimalRoute, route_result: Dict[str, Any]) -> ExecutionReport:
        """将路由结果转换为执行报告"""
        # 简化转换，实际实现需要更详细的映射
        from .professional_execution_system import ExecutionReport, ExecutionFill
        
        fills = []
        for result in route_result.get('execution_results', []):
            if result['status'] == 'success':
                fill = ExecutionFill(
                    fill_id=f"ROUTE_FILL_{uuid.uuid4().hex[:6]}",
                    slice_id="route_segment",
                    parent_order_id=route.route_id,
                    symbol=route.symbol,
                    side=route.side,
                    quantity=result['filled_qty'],
                    price=result['avg_price'],
                    timestamp=datetime.now(),
                    venue=result['venue'],
                    fees_usd=result['fees'],
                    slippage_bps=0,  # 简化
                    market_impact_bps=0,
                    latency_ms=100,
                    liquidity_type="mixed",
                    arrival_price=result['avg_price'],
                    implementation_shortfall_bps=0
                )
                fills.append(fill)
        
        return ExecutionReport(
            session_id=route.route_id,
            symbol=route.symbol,
            target_position=None,  # 简化
            execution_algorithm="multi_venue_routing",
            execution_start=datetime.now() - timedelta(minutes=5),
            execution_end=datetime.now(),
            
            total_slices=len(route.segments),
            successful_slices=len(fills),
            failed_slices=0,
            cancelled_slices=0,
            fill_rate=route_result.get('fill_rate', 0),
            
            target_quantity=route.total_quantity,
            executed_quantity=route_result.get('filled_quantity', 0),
            target_size_usd=route.total_size_usd,
            executed_size_usd=route_result.get('filled_value', 0),
            completion_rate=route_result.get('fill_rate', 0),
            
            total_fees_usd=route_result.get('total_fees', 0),
            total_market_impact_bps=route.estimated_slippage_bps,
            avg_slippage_bps=route.estimated_slippage_bps,
            implementation_shortfall_bps=0,
            total_cost_usd=route.total_fees_usd,
            cost_per_share_bps=route.total_cost_bps,
            
            avg_latency_ms=route.estimated_execution_time_ms,
            maker_ratio=0.5,  # 简化
            venue_distribution=dict(zip([s.venue for s in route.segments], [1.0/len(route.segments)]*len(route.segments))),
            time_distribution={},
            
            fills=fills
        )
    
    async def _orchestration_loop(self):
        """主编排循环"""
        while self.running:
            try:
                # 检查待执行的目标
                pending_targets = [
                    target for target in self.portfolio_targets.values()
                    if target.status == "pending" and self._should_execute_target(target)
                ]
                
                # 按优先级排序
                pending_targets.sort(key=lambda x: x.priority.value, reverse=True)
                
                # 并发执行（限制数量）
                max_concurrent = min(self.risk_limits['max_concurrent_executions'], 
                                   len(pending_targets))
                
                if max_concurrent > 0:
                    tasks = []
                    for target in pending_targets[:max_concurrent]:
                        target.status = "executing"
                        task = asyncio.create_task(
                            self.execute_portfolio_target(target.target_id)
                        )
                        tasks.append(task)
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                
                # 清理已完成的会话
                self._cleanup_completed_sessions()
                
                # 生成定期报告
                await self._generate_periodic_reports()
                
                # 等待下次循环
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"编排循环错误: {e}")
                await asyncio.sleep(60)  # 出错后等待1分钟
    
    def _should_execute_target(self, target: PortfolioTarget) -> bool:
        """判断是否应该执行目标"""
        # 基本条件检查
        if abs(target.target_position_usd - target.current_position_usd) < 10:  # 小于10美元不执行
            return False
        
        # DipMaster时机检查
        if target.dipmaster_context:
            if target.dipmaster_context.get('timing_critical'):
                # 检查是否接近15分钟边界
                current_minute = datetime.now().minute
                return current_minute in [13, 14, 28, 29, 43, 44, 58, 59]
            elif target.dipmaster_context.get('dip_signal'):
                # 逢跌信号立即执行
                return True
        
        # 优先级检查
        if target.priority in [ExecutionPriority.CRITICAL, ExecutionPriority.EMERGENCY]:
            return True
        
        # 时间检查
        age_minutes = (datetime.now() - target.created_at).total_seconds() / 60
        if age_minutes > 60:  # 超过1小时自动执行
            return True
        
        return False
    
    def _cleanup_completed_sessions(self):
        """清理已完成的会话"""
        completed_sessions = [
            sid for sid, session in self.active_sessions.items()
            if session.current_status in ["completed", "failed"] and
            session.end_time and (datetime.now() - session.end_time).total_seconds() > 3600  # 1小时后清理
        ]
        
        for session_id in completed_sessions:
            del self.active_sessions[session_id]
    
    async def _generate_periodic_reports(self):
        """生成定期报告"""
        now = datetime.now()
        
        # 每小时生成一次报告
        if now.minute == 0 and now.second < 30:
            try:
                report = self._generate_hourly_report()
                logger.info(f"小时报告: {json.dumps(report, indent=2, default=str)}")
            except Exception as e:
                logger.error(f"生成小时报告失败: {e}")
    
    def _generate_hourly_report(self) -> Dict[str, Any]:
        """生成小时报告"""
        now = datetime.now()
        hour_start = now.replace(minute=0, second=0, microsecond=0)
        
        # 统计过去一小时的执行
        recent_sessions = [
            session for session in self.execution_history
            if session.start_time >= hour_start
        ]
        
        if not recent_sessions:
            return {"message": "过去一小时无执行记录"}
        
        # 基本统计
        total_sessions = len(recent_sessions)
        completed_sessions = [s for s in recent_sessions if s.current_status == "completed"]
        failed_sessions = [s for s in recent_sessions if s.current_status == "failed"]
        
        success_rate = len(completed_sessions) / total_sessions if total_sessions > 0 else 0
        
        # 交易量和成本统计
        total_volume = sum(
            abs(s.portfolio_target.target_position_usd - s.portfolio_target.current_position_usd)
            for s in completed_sessions
        )
        
        avg_quality_score = np.mean([
            s.quality_metrics.quality_score for s in completed_sessions
            if s.quality_metrics
        ]) if any(s.quality_metrics for s in completed_sessions) else 0
        
        return {
            "period": f"{hour_start.strftime('%Y-%m-%d %H:00')} - {now.strftime('%H:%M')}",
            "total_executions": total_sessions,
            "completed_executions": len(completed_sessions), 
            "failed_executions": len(failed_sessions),
            "success_rate": success_rate,
            "total_volume_usd": total_volume,
            "avg_quality_score": avg_quality_score,
            "active_sessions": len(self.active_sessions),
            "pending_targets": len([t for t in self.portfolio_targets.values() if t.status == "pending"])
        }
    
    def _update_performance_stats(self, session: ExecutionSession):
        """更新性能统计"""
        if session.current_status == "completed" and session.quality_metrics:
            symbol = session.portfolio_target.symbol
            self.performance_stats[f"{symbol}_quality_score"].append(session.quality_metrics.quality_score)
            self.performance_stats[f"{symbol}_execution_time"].append(
                (session.end_time - session.start_time).total_seconds()
            )
    
    async def _trigger_execution_callbacks(self, session: ExecutionSession, event: str):
        """触发执行回调"""
        for callback in self.execution_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(session, event)
                else:
                    callback(session, event)
            except Exception as e:
                logger.error(f"执行回调失败: {e}")
    
    def add_execution_callback(self, callback: Callable):
        """添加执行回调"""
        self.execution_callbacks.append(callback)
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """获取编排器状态"""
        return {
            "mode": self.mode.value,
            "running": self.running,
            "paper_trading": self.paper_trading,
            
            "portfolio_targets": len(self.portfolio_targets),
            "pending_targets": len([t for t in self.portfolio_targets.values() if t.status == "pending"]),
            "active_sessions": len(self.active_sessions),
            "execution_history_count": len(self.execution_history),
            
            "current_exposure": dict(self.current_exposure),
            "risk_limits": self.risk_limits,
            
            "performance_summary": self._get_performance_summary(),
            "last_update": datetime.now()
        }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.execution_history:
            return {}
        
        recent_sessions = list(self.execution_history)[-20:]  # 最近20次
        
        completed_sessions = [s for s in recent_sessions if s.current_status == "completed"]
        if not completed_sessions:
            return {"message": "无已完成的执行"}
        
        avg_quality = np.mean([
            s.quality_metrics.quality_score for s in completed_sessions
            if s.quality_metrics
        ]) if any(s.quality_metrics for s in completed_sessions) else 0
        
        avg_duration = np.mean([
            (s.end_time - s.start_time).total_seconds() / 60
            for s in completed_sessions if s.end_time
        ])
        
        success_rate = len(completed_sessions) / len(recent_sessions)
        
        return {
            "recent_sessions": len(recent_sessions),
            "success_rate": success_rate,
            "avg_quality_score": avg_quality,
            "avg_duration_minutes": avg_duration
        }


# 演示函数
async def demo_dipmaster_execution_orchestrator():
    """DipMaster执行编排器演示"""
    
    print("="*80)
    print("DipMaster Trading System - 执行编排器综合演示")
    print("="*80)
    
    # 创建编排器
    orchestrator = DipMasterExecutionOrchestrator(mode=OrchestrationMode.SIMULATION)
    
    # 添加执行回调
    async def execution_callback(session: ExecutionSession, event: str):
        print(f"📢 回调事件: {session.session_id} - {event}")
        if event == "completed" and session.quality_metrics:
            print(f"   质量评分: {session.quality_metrics.quality_score:.1f}")
    
    orchestrator.add_execution_callback(execution_callback)
    
    try:
        # 启动编排器
        await orchestrator.start()
        
        # 创建多个组合目标
        targets = [
            # 目标1: DipMaster 15分钟边界执行
            PortfolioTarget(
                target_id="DM_15MIN_BTC",
                symbol="BTCUSDT", 
                current_position_usd=5000,
                target_position_usd=12000,
                urgency=OrderUrgency.HIGH,
                priority=ExecutionPriority.HIGH,
                dipmaster_context={
                    'timing_critical': True,
                    'target_boundary_minute': 15
                }
            ),
            
            # 目标2: DipMaster 逢跌买入
            PortfolioTarget(
                target_id="DM_DIP_ETH",
                symbol="ETHUSDT",
                current_position_usd=2000,
                target_position_usd=8000,
                urgency=OrderUrgency.HIGH,
                priority=ExecutionPriority.CRITICAL,
                dipmaster_context={
                    'dip_signal': True,
                    'rsi_value': 35.5,
                    'price_drop_confirmed': True
                }
            ),
            
            # 目标3: 大额常规执行
            PortfolioTarget(
                target_id="LARGE_SOL",
                symbol="SOLUSDT",
                current_position_usd=1000,
                target_position_usd=15000,
                urgency=OrderUrgency.MEDIUM,
                priority=ExecutionPriority.NORMAL,
                max_execution_time_minutes=45,
                preferred_venues=["binance", "okx", "bybit"]
            ),
            
            # 目标4: 紧急减仓
            PortfolioTarget(
                target_id="EMERGENCY_REDUCE",
                symbol="BNBUSDT",
                current_position_usd=10000,
                target_position_usd=3000,
                urgency=OrderUrgency.EMERGENCY,
                priority=ExecutionPriority.EMERGENCY,
                max_execution_time_minutes=10
            )
        ]
        
        # 添加组合目标
        orchestrator.add_portfolio_targets(targets)
        
        print(f"\n添加了 {len(targets)} 个组合目标")
        
        # 手动触发一些目标执行
        print("\n手动执行演示:")
        print("-" * 60)
        
        # 执行DipMaster逢跌买入（最高优先级）
        try:
            session1 = await orchestrator.execute_portfolio_target("DM_DIP_ETH")
            print(f"✅ DipMaster逢跌买入执行完成: {session1.session_id}")
            print(f"   状态: {session1.current_status}")
            if session1.execution_report:
                print(f"   成交率: {session1.execution_report.completion_rate:.1%}")
                print(f"   总成本: ${session1.execution_report.total_cost_usd:.2f}")
        except Exception as e:
            print(f"❌ DipMaster逢跌买入执行失败: {e}")
        
        # 执行大额常规订单
        try:
            session2 = await orchestrator.execute_portfolio_target("LARGE_SOL")
            print(f"✅ 大额常规执行完成: {session2.session_id}")
            if session2.optimal_route:
                print(f"   使用了{session2.optimal_route.venue_count}个交易所路由")
                print(f"   成本节约: {session2.optimal_route.cost_savings_bps:.2f}bps")
        except Exception as e:
            print(f"❌ 大额常规执行失败: {e}")
        
        # 让编排循环运行一段时间
        print("\n编排循环运行 (30秒)...")
        await asyncio.sleep(30)
        
        # 获取状态报告
        print("\n编排器状态报告:")
        print("-" * 60)
        
        status = orchestrator.get_orchestrator_status()
        
        print(f"运行模式: {status['mode']}")
        print(f"组合目标: {status['portfolio_targets']} (待处理: {status['pending_targets']})")
        print(f"活跃会话: {status['active_sessions']}")
        print(f"执行历史: {status['execution_history_count']}")
        
        if status['performance_summary']:
            perf = status['performance_summary']
            print(f"成功率: {perf.get('success_rate', 0):.1%}")
            print(f"平均质量: {perf.get('avg_quality_score', 0):.1f}")
            print(f"平均用时: {perf.get('avg_duration_minutes', 0):.1f}分钟")
        
        # 生成TCA报告
        print("\n生成TCA报告:")
        print("-" * 60)
        
        try:
            tca_report = orchestrator.quality_monitor.generate_daily_tca_report()
            print(f"TCA报告ID: {tca_report.report_id}")
            print(f"总执行数: {tca_report.total_executions}")
            print(f"总交易量: ${tca_report.total_volume_usd:,.0f}")
            print(f"平均成本: {tca_report.avg_cost_bps:.2f}bps")
            
            if tca_report.optimization_suggestions:
                print("优化建议:")
                for suggestion in tca_report.optimization_suggestions[:3]:
                    print(f"  • {suggestion}")
                    
        except Exception as e:
            print(f"TCA报告生成失败: {e}")
        
        print("\n组件状态:")
        print("-" * 60)
        
        # 检查各组件状态
        scheduler_status = orchestrator.scheduler.get_scheduler_status()
        print(f"调度器: 运行中={scheduler_status['running']}, 待处理={scheduler_status['pending_positions']}")
        
        router_status = orchestrator.router.get_venue_status()
        print(f"路由器: {router_status['active_venues']}/{router_status['total_venues']} 交易所在线")
        
        # 展示套利机会
        arbitrage_ops = orchestrator.router.get_arbitrage_opportunities()
        if arbitrage_ops:
            print(f"套利机会: {len(arbitrage_ops)}个，最优利润{arbitrage_ops[0]['profit_bps']:.2f}bps")
        
    finally:
        await orchestrator.stop()
    
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(demo_dipmaster_execution_orchestrator())