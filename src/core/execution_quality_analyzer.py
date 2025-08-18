"""
Execution Quality Analyzer & Transaction Cost Analysis (TCA)
执行质量分析器与交易成本分析系统

核心功能:
1. 实时执行质量监控和TCA分析
2. 多维度成本分解（手续费、滑点、市场冲击、时机成本）
3. 基准比较分析（TWAP、VWAP、Arrival Price）
4. 执行效率评估和算法性能比较
5. 异常检测和风险预警系统
6. DipMaster专用执行质量指标
7. 持续优化建议和反馈系统
"""

import asyncio
import logging
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import sqlite3
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class BenchmarkType(Enum):
    """基准类型"""
    ARRIVAL_PRICE = "arrival_price"
    TWAP = "twap"
    VWAP = "vwap"
    CLOSE_PRICE = "close_price"
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"


class CostComponent(Enum):
    """成本组成部分"""
    COMMISSION = "commission"          # 手续费
    SPREAD = "spread"                 # 价差成本
    MARKET_IMPACT = "market_impact"   # 市场冲击
    TIMING = "timing"                 # 时机成本
    OPPORTUNITY = "opportunity"       # 机会成本
    DELAY = "delay"                   # 延迟成本


@dataclass
class ExecutionBenchmark:
    """执行基准"""
    benchmark_type: BenchmarkType
    price: float
    timestamp: datetime
    confidence: float = 1.0
    source: str = "market_data"


@dataclass
class CostBreakdown:
    """成本分解"""
    symbol: str
    side: str
    total_quantity: float
    notional_usd: float
    
    # 基准价格
    arrival_price: float
    twap_benchmark: float
    vwap_benchmark: float
    
    # 成本组成（以bps计）
    commission_bps: float
    spread_cost_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    
    # 总成本
    total_cost_bps: float
    total_cost_usd: float
    
    # 相对基准的表现
    vs_arrival_bps: float
    vs_twap_bps: float
    vs_vwap_bps: float
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionMetrics:
    """执行指标"""
    session_id: str
    symbol: str
    side: str
    algorithm: str
    venue: str
    
    # 基本指标
    target_quantity: float
    executed_quantity: float
    fill_rate: float
    
    # 时间指标
    execution_start: datetime
    execution_end: datetime
    duration_seconds: float
    
    # 价格指标
    avg_fill_price: float
    arrival_price: float
    
    # 成本指标
    total_fees_usd: float
    slippage_bps: float
    market_impact_bps: float
    
    # 质量指标
    participation_rate: float
    maker_ratio: float
    venue_count: int
    
    # DipMaster专用指标
    dipmaster_timing_accuracy: Optional[float] = None  # 15分钟边界命中精度
    dipmaster_dip_capture_rate: Optional[float] = None  # 逢跌信号捕获率
    
    # 异常标记
    anomaly_flags: List[str] = field(default_factory=list)
    quality_score: float = 0.0


@dataclass
class TCAReport:
    """TCA报告"""
    report_id: str
    period_start: datetime
    period_end: datetime
    
    # 概览统计
    total_executions: int
    total_volume_usd: float
    total_cost_usd: float
    avg_cost_bps: float
    
    # 算法性能比较
    algorithm_performance: Dict[str, Dict[str, float]]
    
    # 交易所性能比较
    venue_performance: Dict[str, Dict[str, float]]
    
    # 成本分解
    cost_breakdown: CostBreakdown
    
    # 时间分析
    intraday_performance: Dict[str, float]
    
    # 异常分析
    anomaly_summary: Dict[str, int]
    
    # 改进建议
    optimization_suggestions: List[str]
    
    # DipMaster专用分析
    dipmaster_analysis: Dict[str, Any] = field(default_factory=dict)


class BenchmarkCalculator:
    """基准计算器"""
    
    def __init__(self):
        self.price_history = defaultdict(deque)  # symbol -> price history
        self.volume_history = defaultdict(deque)  # symbol -> volume history
        
    def update_market_data(self, symbol: str, price: float, volume: float, timestamp: datetime):
        """更新市场数据"""
        self.price_history[symbol].append((timestamp, price))
        self.volume_history[symbol].append((timestamp, volume))
        
        # 保留最近4小时的数据
        cutoff_time = timestamp - timedelta(hours=4)
        
        while (self.price_history[symbol] and 
               self.price_history[symbol][0][0] < cutoff_time):
            self.price_history[symbol].popleft()
            
        while (self.volume_history[symbol] and 
               self.volume_history[symbol][0][0] < cutoff_time):
            self.volume_history[symbol].popleft()
    
    def calculate_twap(self, symbol: str, start_time: datetime, end_time: datetime) -> Optional[float]:
        """计算TWAP基准"""
        prices = [(ts, price) for ts, price in self.price_history[symbol] 
                 if start_time <= ts <= end_time]
        
        if not prices:
            return None
            
        if len(prices) == 1:
            return prices[0][1]
        
        # 时间加权平均
        total_time = (end_time - start_time).total_seconds()
        if total_time <= 0:
            return prices[-1][1]
        
        weighted_sum = 0
        total_weight = 0
        
        for i in range(len(prices) - 1):
            current_time, current_price = prices[i]
            next_time, _ = prices[i + 1]
            
            weight = (next_time - current_time).total_seconds()
            weighted_sum += current_price * weight
            total_weight += weight
        
        # 处理最后一个点
        if total_weight < total_time:
            last_weight = total_time - total_weight
            weighted_sum += prices[-1][1] * last_weight
            total_weight += last_weight
        
        return weighted_sum / total_weight if total_weight > 0 else prices[-1][1]
    
    def calculate_vwap(self, symbol: str, start_time: datetime, end_time: datetime) -> Optional[float]:
        """计算VWAP基准"""
        prices = [(ts, price) for ts, price in self.price_history[symbol] 
                 if start_time <= ts <= end_time]
        volumes = [(ts, volume) for ts, volume in self.volume_history[symbol] 
                  if start_time <= ts <= end_time]
        
        if not prices or not volumes:
            return None
        
        # 对齐价格和成交量数据
        price_volume_pairs = []
        
        for price_ts, price in prices:
            # 找到最接近的成交量数据
            closest_volume = None
            min_time_diff = float('inf')
            
            for volume_ts, volume in volumes:
                time_diff = abs((price_ts - volume_ts).total_seconds())
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_volume = volume
            
            if closest_volume and min_time_diff < 300:  # 5分钟内的数据才使用
                price_volume_pairs.append((price, closest_volume))
        
        if not price_volume_pairs:
            return self.calculate_twap(symbol, start_time, end_time)
        
        # 成交量加权平均
        total_value = sum(price * volume for price, volume in price_volume_pairs)
        total_volume = sum(volume for _, volume in price_volume_pairs)
        
        return total_value / total_volume if total_volume > 0 else None
    
    def get_arrival_price(self, symbol: str, arrival_time: datetime, tolerance_seconds: int = 30) -> Optional[float]:
        """获取到达价格"""
        # 寻找最接近到达时间的价格
        closest_price = None
        min_time_diff = float('inf')
        
        for ts, price in self.price_history[symbol]:
            time_diff = abs((ts - arrival_time).total_seconds())
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_price = price
        
        return closest_price if min_time_diff <= tolerance_seconds else None


class ExecutionAnalyzer:
    """执行分析器"""
    
    def __init__(self):
        self.benchmark_calculator = BenchmarkCalculator()
        self.execution_database = ExecutionDatabase()
        
    def analyze_execution(
        self,
        execution_data: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> ExecutionMetrics:
        """分析单次执行"""
        
        session_id = execution_data.get('session_id', '')
        symbol = execution_data.get('symbol', '')
        side = execution_data.get('side', '')
        algorithm = execution_data.get('algorithm', '')
        
        logger.info(f"分析执行: {session_id} - {symbol} {side}")
        
        try:
            # 基本指标计算
            target_quantity = execution_data.get('target_quantity', 0)
            executed_quantity = execution_data.get('executed_quantity', 0)
            fill_rate = executed_quantity / target_quantity if target_quantity > 0 else 0
            
            # 时间指标
            execution_start = execution_data.get('execution_start')
            execution_end = execution_data.get('execution_end')
            duration_seconds = (execution_end - execution_start).total_seconds() if execution_start and execution_end else 0
            
            # 价格和成本指标
            fills = execution_data.get('fills', [])
            if fills:
                total_notional = sum(fill['quantity'] * fill['price'] for fill in fills)
                avg_fill_price = total_notional / executed_quantity if executed_quantity > 0 else 0
            else:
                avg_fill_price = execution_data.get('avg_price', 0)
            
            arrival_price = execution_data.get('arrival_price', avg_fill_price)
            
            # 成本计算
            total_fees = execution_data.get('total_fees_usd', 0)
            slippage_bps = ((avg_fill_price - arrival_price) / arrival_price * 10000) if arrival_price > 0 else 0
            if side == 'SELL':
                slippage_bps = -slippage_bps  # 卖出时滑点符号相反
            
            # 市场冲击估算
            market_impact_bps = max(0, abs(slippage_bps) - 5)  # 超过5bps的部分认为是市场冲击
            
            # 参与率计算
            market_volume = market_context.get('volume_during_execution', 1000)
            participation_rate = (executed_quantity * avg_fill_price) / (market_volume * avg_fill_price) if market_volume > 0 else 0
            
            # Maker比例
            maker_fills = [f for f in fills if f.get('liquidity_type') == 'maker']
            maker_ratio = len(maker_fills) / len(fills) if fills else 0
            
            # 交易所数量
            venues = set(f.get('venue', 'unknown') for f in fills)
            venue_count = len(venues)
            
            # DipMaster专用指标
            dipmaster_timing_accuracy = None
            dipmaster_dip_capture_rate = None
            
            if algorithm in ['dipmaster_15min', 'DIPMASTER_15MIN']:
                # 15分钟边界命中精度
                target_boundary_minute = market_context.get('target_boundary_minute')
                actual_completion_minute = execution_end.minute if execution_end else None
                
                if target_boundary_minute is not None and actual_completion_minute is not None:
                    timing_error = abs(actual_completion_minute - target_boundary_minute)
                    dipmaster_timing_accuracy = max(0, 1 - timing_error / 5)  # 5分钟内为满分
                    
            elif algorithm in ['dipmaster_dip_buy', 'DIPMASTER_DIP_BUY']:
                # 逢跌信号捕获率
                entry_rsi = market_context.get('entry_rsi')
                price_drop_confirmed = market_context.get('price_drop_confirmed', False)
                
                if entry_rsi is not None:
                    # RSI在30-50之间且价格确实下跌为最佳
                    rsi_score = 1.0 if 30 <= entry_rsi <= 50 else max(0, 1 - abs(entry_rsi - 40) / 20)
                    drop_score = 1.0 if price_drop_confirmed else 0.5
                    dipmaster_dip_capture_rate = (rsi_score + drop_score) / 2
            
            # 异常检测
            anomaly_flags = []
            
            if fill_rate < 0.8:
                anomaly_flags.append("LOW_FILL_RATE")
            if abs(slippage_bps) > 50:
                anomaly_flags.append("HIGH_SLIPPAGE")
            if market_impact_bps > 30:
                anomaly_flags.append("HIGH_MARKET_IMPACT")
            if duration_seconds > 3600:  # 超过1小时
                anomaly_flags.append("LONG_EXECUTION_TIME")
            if participation_rate > 0.5:
                anomaly_flags.append("HIGH_PARTICIPATION_RATE")
                
            # 质量评分计算
            quality_score = self._calculate_quality_score(
                fill_rate, slippage_bps, market_impact_bps, 
                participation_rate, maker_ratio, anomaly_flags
            )
            
            metrics = ExecutionMetrics(
                session_id=session_id,
                symbol=symbol,
                side=side,
                algorithm=algorithm,
                venue=list(venues)[0] if venues else 'unknown',
                
                target_quantity=target_quantity,
                executed_quantity=executed_quantity,
                fill_rate=fill_rate,
                
                execution_start=execution_start,
                execution_end=execution_end,
                duration_seconds=duration_seconds,
                
                avg_fill_price=avg_fill_price,
                arrival_price=arrival_price,
                
                total_fees_usd=total_fees,
                slippage_bps=slippage_bps,
                market_impact_bps=market_impact_bps,
                
                participation_rate=participation_rate,
                maker_ratio=maker_ratio,
                venue_count=venue_count,
                
                dipmaster_timing_accuracy=dipmaster_timing_accuracy,
                dipmaster_dip_capture_rate=dipmaster_dip_capture_rate,
                
                anomaly_flags=anomaly_flags,
                quality_score=quality_score
            )
            
            # 存储到数据库
            self.execution_database.store_execution_metrics(metrics)
            
            logger.info(f"执行分析完成: {session_id} - 质量评分 {quality_score:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"执行分析失败: {session_id} - {e}")
            raise
    
    def _calculate_quality_score(
        self, 
        fill_rate: float,
        slippage_bps: float,
        market_impact_bps: float,
        participation_rate: float,
        maker_ratio: float,
        anomaly_flags: List[str]
    ) -> float:
        """计算执行质量评分"""
        
        # 基础分数
        base_score = 100.0
        
        # 成交率评分 (权重: 30%)
        fill_rate_score = fill_rate * 30
        
        # 滑点评分 (权重: 25%)
        slippage_penalty = min(25, abs(slippage_bps) / 2)  # 每2bps扣1分
        slippage_score = max(0, 25 - slippage_penalty)
        
        # 市场冲击评分 (权重: 20%)
        impact_penalty = min(20, market_impact_bps)  # 每1bps扣1分
        impact_score = max(0, 20 - impact_penalty)
        
        # 参与率评分 (权重: 10%)
        # 适中的参与率最好
        if participation_rate < 0.05:
            participation_score = participation_rate * 200  # 低参与率线性增长
        elif participation_rate < 0.3:
            participation_score = 10  # 最优区间
        else:
            participation_score = max(0, 10 - (participation_rate - 0.3) * 25)  # 高参与率惩罚
        
        # Maker比例评分 (权重: 10%)
        maker_score = maker_ratio * 10
        
        # 异常惩罚 (权重: 5%)
        anomaly_penalty = len(anomaly_flags) * 1
        anomaly_score = max(0, 5 - anomaly_penalty)
        
        total_score = (fill_rate_score + slippage_score + impact_score + 
                      participation_score + maker_score + anomaly_score)
        
        return max(0, min(100, total_score))
    
    def calculate_cost_breakdown(
        self,
        execution_metrics: ExecutionMetrics,
        benchmarks: Dict[BenchmarkType, float]
    ) -> CostBreakdown:
        """计算成本分解"""
        
        notional_usd = execution_metrics.executed_quantity * execution_metrics.avg_fill_price
        
        # 手续费（bps）
        commission_bps = (execution_metrics.total_fees_usd / notional_usd) * 10000 if notional_usd > 0 else 0
        
        # 价差成本（估算）
        spread_cost_bps = max(5, abs(execution_metrics.slippage_bps) * 0.3)  # 简化估算
        
        # 市场冲击
        market_impact_bps = execution_metrics.market_impact_bps
        
        # 时机成本（基于执行时间）
        timing_cost_bps = min(10, execution_metrics.duration_seconds / 360)  # 每6分钟1bps
        
        # 总成本
        total_cost_bps = commission_bps + spread_cost_bps + market_impact_bps + timing_cost_bps
        total_cost_usd = (total_cost_bps / 10000) * notional_usd
        
        # 相对基准的表现
        arrival_price = benchmarks.get(BenchmarkType.ARRIVAL_PRICE, execution_metrics.arrival_price)
        twap_benchmark = benchmarks.get(BenchmarkType.TWAP, execution_metrics.avg_fill_price)
        vwap_benchmark = benchmarks.get(BenchmarkType.VWAP, execution_metrics.avg_fill_price)
        
        vs_arrival_bps = ((execution_metrics.avg_fill_price - arrival_price) / arrival_price) * 10000 if arrival_price > 0 else 0
        vs_twap_bps = ((execution_metrics.avg_fill_price - twap_benchmark) / twap_benchmark) * 10000 if twap_benchmark > 0 else 0
        vs_vwap_bps = ((execution_metrics.avg_fill_price - vwap_benchmark) / vwap_benchmark) * 10000 if vwap_benchmark > 0 else 0
        
        return CostBreakdown(
            symbol=execution_metrics.symbol,
            side=execution_metrics.side,
            total_quantity=execution_metrics.executed_quantity,
            notional_usd=notional_usd,
            
            arrival_price=arrival_price,
            twap_benchmark=twap_benchmark,
            vwap_benchmark=vwap_benchmark,
            
            commission_bps=commission_bps,
            spread_cost_bps=spread_cost_bps,
            market_impact_bps=market_impact_bps,
            timing_cost_bps=timing_cost_bps,
            
            total_cost_bps=total_cost_bps,
            total_cost_usd=total_cost_usd,
            
            vs_arrival_bps=vs_arrival_bps,
            vs_twap_bps=vs_twap_bps,
            vs_vwap_bps=vs_vwap_bps
        )


class ExecutionDatabase:
    """执行数据库"""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent / "data" / "execution_quality.db"
        
        self.db_path = db_path
        self._create_tables()
        
    def _create_tables(self):
        """创建数据库表"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    venue TEXT NOT NULL,
                    
                    target_quantity REAL NOT NULL,
                    executed_quantity REAL NOT NULL,
                    fill_rate REAL NOT NULL,
                    
                    execution_start TIMESTAMP NOT NULL,
                    execution_end TIMESTAMP NOT NULL,
                    duration_seconds REAL NOT NULL,
                    
                    avg_fill_price REAL NOT NULL,
                    arrival_price REAL NOT NULL,
                    
                    total_fees_usd REAL NOT NULL,
                    slippage_bps REAL NOT NULL,
                    market_impact_bps REAL NOT NULL,
                    
                    participation_rate REAL NOT NULL,
                    maker_ratio REAL NOT NULL,
                    venue_count INTEGER NOT NULL,
                    
                    dipmaster_timing_accuracy REAL,
                    dipmaster_dip_capture_rate REAL,
                    
                    anomaly_flags TEXT,
                    quality_score REAL NOT NULL,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_symbol_date 
                ON execution_metrics(symbol, execution_start)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_execution_algorithm 
                ON execution_metrics(algorithm, execution_start)
            """)
    
    def store_execution_metrics(self, metrics: ExecutionMetrics):
        """存储执行指标"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO execution_metrics (
                    session_id, symbol, side, algorithm, venue,
                    target_quantity, executed_quantity, fill_rate,
                    execution_start, execution_end, duration_seconds,
                    avg_fill_price, arrival_price,
                    total_fees_usd, slippage_bps, market_impact_bps,
                    participation_rate, maker_ratio, venue_count,
                    dipmaster_timing_accuracy, dipmaster_dip_capture_rate,
                    anomaly_flags, quality_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.session_id, metrics.symbol, metrics.side, 
                metrics.algorithm, metrics.venue,
                metrics.target_quantity, metrics.executed_quantity, metrics.fill_rate,
                metrics.execution_start, metrics.execution_end, metrics.duration_seconds,
                metrics.avg_fill_price, metrics.arrival_price,
                metrics.total_fees_usd, metrics.slippage_bps, metrics.market_impact_bps,
                metrics.participation_rate, metrics.maker_ratio, metrics.venue_count,
                metrics.dipmaster_timing_accuracy, metrics.dipmaster_dip_capture_rate,
                json.dumps(metrics.anomaly_flags), metrics.quality_score
            ))
    
    def query_executions(
        self, 
        start_time: datetime, 
        end_time: datetime,
        symbols: Optional[List[str]] = None,
        algorithms: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """查询执行记录"""
        
        query = """
            SELECT * FROM execution_metrics 
            WHERE execution_start >= ? AND execution_start <= ?
        """
        params = [start_time, end_time]
        
        if symbols:
            query += " AND symbol IN ({})".format(','.join('?' * len(symbols)))
            params.extend(symbols)
            
        if algorithms:
            query += " AND algorithm IN ({})".format(','.join('?' * len(algorithms)))
            params.extend(algorithms)
            
        query += " ORDER BY execution_start DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]


class TCAReportGenerator:
    """TCA报告生成器"""
    
    def __init__(self, execution_database: ExecutionDatabase):
        self.execution_database = execution_database
        
    def generate_tca_report(
        self,
        start_time: datetime,
        end_time: datetime,
        symbols: Optional[List[str]] = None
    ) -> TCAReport:
        """生成TCA报告"""
        
        report_id = f"TCA_{int(time.time())}"
        logger.info(f"生成TCA报告: {report_id} ({start_time} - {end_time})")
        
        # 查询执行数据
        executions = self.execution_database.query_executions(
            start_time=start_time,
            end_time=end_time,
            symbols=symbols
        )
        
        if not executions:
            raise ValueError("指定时间段内无执行数据")
        
        # 概览统计
        total_executions = len(executions)
        total_volume_usd = sum(ex['executed_quantity'] * ex['avg_fill_price'] for ex in executions)
        total_cost_usd = sum(ex['total_fees_usd'] for ex in executions)
        avg_cost_bps = np.mean([ex['slippage_bps'] + ex['market_impact_bps'] for ex in executions])
        
        # 算法性能分析
        algorithm_performance = self._analyze_algorithm_performance(executions)
        
        # 交易所性能分析
        venue_performance = self._analyze_venue_performance(executions)
        
        # 成本分解分析
        cost_breakdown = self._analyze_cost_breakdown(executions)
        
        # 日内表现分析
        intraday_performance = self._analyze_intraday_performance(executions)
        
        # 异常分析
        anomaly_summary = self._analyze_anomalies(executions)
        
        # 优化建议
        optimization_suggestions = self._generate_optimization_suggestions(executions)
        
        # DipMaster专用分析
        dipmaster_analysis = self._analyze_dipmaster_performance(executions)
        
        return TCAReport(
            report_id=report_id,
            period_start=start_time,
            period_end=end_time,
            
            total_executions=total_executions,
            total_volume_usd=total_volume_usd,
            total_cost_usd=total_cost_usd,
            avg_cost_bps=avg_cost_bps,
            
            algorithm_performance=algorithm_performance,
            venue_performance=venue_performance,
            cost_breakdown=cost_breakdown,
            intraday_performance=intraday_performance,
            anomaly_summary=anomaly_summary,
            optimization_suggestions=optimization_suggestions,
            dipmaster_analysis=dipmaster_analysis
        )
    
    def _analyze_algorithm_performance(self, executions: List[Dict]) -> Dict[str, Dict[str, float]]:
        """分析算法性能"""
        algo_stats = defaultdict(list)
        
        for ex in executions:
            algo = ex['algorithm']
            algo_stats[algo].append({
                'quality_score': ex['quality_score'],
                'slippage_bps': ex['slippage_bps'],
                'market_impact_bps': ex['market_impact_bps'],
                'fill_rate': ex['fill_rate'],
                'maker_ratio': ex['maker_ratio']
            })
        
        performance = {}
        for algo, stats in algo_stats.items():
            performance[algo] = {
                'avg_quality_score': np.mean([s['quality_score'] for s in stats]),
                'avg_slippage_bps': np.mean([s['slippage_bps'] for s in stats]),
                'avg_market_impact_bps': np.mean([s['market_impact_bps'] for s in stats]),
                'avg_fill_rate': np.mean([s['fill_rate'] for s in stats]),
                'avg_maker_ratio': np.mean([s['maker_ratio'] for s in stats]),
                'execution_count': len(stats)
            }
        
        return performance
    
    def _analyze_venue_performance(self, executions: List[Dict]) -> Dict[str, Dict[str, float]]:
        """分析交易所性能"""
        venue_stats = defaultdict(list)
        
        for ex in executions:
            venue = ex['venue']
            venue_stats[venue].append({
                'quality_score': ex['quality_score'],
                'slippage_bps': abs(ex['slippage_bps']),
                'fill_rate': ex['fill_rate'],
                'duration_seconds': ex['duration_seconds']
            })
        
        performance = {}
        for venue, stats in venue_stats.items():
            performance[venue] = {
                'avg_quality_score': np.mean([s['quality_score'] for s in stats]),
                'avg_slippage_bps': np.mean([s['slippage_bps'] for s in stats]),
                'avg_fill_rate': np.mean([s['fill_rate'] for s in stats]),
                'avg_duration_seconds': np.mean([s['duration_seconds'] for s in stats]),
                'execution_count': len(stats)
            }
        
        return performance
    
    def _analyze_cost_breakdown(self, executions: List[Dict]) -> CostBreakdown:
        """分析成本分解"""
        if not executions:
            return None
            
        # 聚合所有执行的成本
        total_quantity = sum(ex['executed_quantity'] for ex in executions)
        total_notional = sum(ex['executed_quantity'] * ex['avg_fill_price'] for ex in executions)
        
        # 加权平均价格
        weighted_avg_price = total_notional / total_quantity if total_quantity > 0 else 0
        
        # 成本组成（简化计算）
        total_fees = sum(ex['total_fees_usd'] for ex in executions)
        commission_bps = (total_fees / total_notional) * 10000 if total_notional > 0 else 0
        
        # 其他成本估算
        avg_slippage = np.mean([abs(ex['slippage_bps']) for ex in executions])
        avg_market_impact = np.mean([ex['market_impact_bps'] for ex in executions])
        
        spread_cost_bps = avg_slippage * 0.3  # 简化估算
        timing_cost_bps = np.mean([min(10, ex['duration_seconds'] / 360) for ex in executions])
        
        total_cost_bps = commission_bps + spread_cost_bps + avg_market_impact + timing_cost_bps
        total_cost_usd = (total_cost_bps / 10000) * total_notional
        
        return CostBreakdown(
            symbol="AGGREGATE",
            side="MIXED",
            total_quantity=total_quantity,
            notional_usd=total_notional,
            
            arrival_price=weighted_avg_price,  # 简化
            twap_benchmark=weighted_avg_price,
            vwap_benchmark=weighted_avg_price,
            
            commission_bps=commission_bps,
            spread_cost_bps=spread_cost_bps,
            market_impact_bps=avg_market_impact,
            timing_cost_bps=timing_cost_bps,
            
            total_cost_bps=total_cost_bps,
            total_cost_usd=total_cost_usd,
            
            vs_arrival_bps=0.0,  # 需要更详细的基准计算
            vs_twap_bps=0.0,
            vs_vwap_bps=0.0
        )
    
    def _analyze_intraday_performance(self, executions: List[Dict]) -> Dict[str, float]:
        """分析日内表现"""
        hourly_stats = defaultdict(list)
        
        for ex in executions:
            hour = datetime.fromisoformat(ex['execution_start']).hour if isinstance(ex['execution_start'], str) else ex['execution_start'].hour
            hourly_stats[f"{hour:02d}:00"].append(ex['quality_score'])
        
        return {hour: np.mean(scores) for hour, scores in hourly_stats.items()}
    
    def _analyze_anomalies(self, executions: List[Dict]) -> Dict[str, int]:
        """分析异常情况"""
        anomaly_counts = defaultdict(int)
        
        for ex in executions:
            if ex['anomaly_flags']:
                flags = json.loads(ex['anomaly_flags']) if isinstance(ex['anomaly_flags'], str) else ex['anomaly_flags']
                for flag in flags:
                    anomaly_counts[flag] += 1
        
        return dict(anomaly_counts)
    
    def _generate_optimization_suggestions(self, executions: List[Dict]) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        # 分析常见问题
        high_slippage_count = sum(1 for ex in executions if abs(ex['slippage_bps']) > 25)
        low_fill_rate_count = sum(1 for ex in executions if ex['fill_rate'] < 0.9)
        long_execution_count = sum(1 for ex in executions if ex['duration_seconds'] > 1800)
        
        total_executions = len(executions)
        
        if high_slippage_count / total_executions > 0.2:
            suggestions.append("高滑点执行较多，建议增加执行时间或使用更保守的算法")
            
        if low_fill_rate_count / total_executions > 0.1:
            suggestions.append("成交率偏低，建议检查订单价格设置和市场流动性")
            
        if long_execution_count / total_executions > 0.3:
            suggestions.append("执行时间过长，建议优化订单分割策略或增加紧急程度")
        
        # 算法建议
        algo_performance = self._analyze_algorithm_performance(executions)
        if algo_performance:
            best_algo = max(algo_performance.keys(), 
                          key=lambda x: algo_performance[x]['avg_quality_score'])
            suggestions.append(f"质量最优算法: {best_algo}, 建议在类似场景下优先使用")
        
        return suggestions
    
    def _analyze_dipmaster_performance(self, executions: List[Dict]) -> Dict[str, Any]:
        """分析DipMaster专用性能"""
        dipmaster_executions = [ex for ex in executions 
                              if 'dipmaster' in ex['algorithm'].lower()]
        
        if not dipmaster_executions:
            return {}
        
        analysis = {
            'total_dipmaster_executions': len(dipmaster_executions),
            'dipmaster_execution_ratio': len(dipmaster_executions) / len(executions)
        }
        
        # 15分钟边界执行分析
        timing_executions = [ex for ex in dipmaster_executions 
                           if ex['dipmaster_timing_accuracy'] is not None]
        if timing_executions:
            analysis['timing_accuracy_avg'] = np.mean([ex['dipmaster_timing_accuracy'] for ex in timing_executions])
            analysis['timing_executions_count'] = len(timing_executions)
        
        # 逢跌买入执行分析
        dip_executions = [ex for ex in dipmaster_executions 
                         if ex['dipmaster_dip_capture_rate'] is not None]
        if dip_executions:
            analysis['dip_capture_rate_avg'] = np.mean([ex['dipmaster_dip_capture_rate'] for ex in dip_executions])
            analysis['dip_executions_count'] = len(dip_executions)
        
        # DipMaster vs 其他算法比较
        other_executions = [ex for ex in executions 
                          if 'dipmaster' not in ex['algorithm'].lower()]
        
        if other_executions:
            dipmaster_quality = np.mean([ex['quality_score'] for ex in dipmaster_executions])
            other_quality = np.mean([ex['quality_score'] for ex in other_executions])
            analysis['quality_vs_others'] = dipmaster_quality - other_quality
        
        return analysis


class ExecutionQualityMonitor:
    """执行质量监控器"""
    
    def __init__(self):
        self.execution_analyzer = ExecutionAnalyzer()
        self.tca_generator = TCAReportGenerator(self.execution_analyzer.execution_database)
        
        # 实时监控状态
        self.monitoring_active = False
        self.alert_thresholds = {
            'min_quality_score': 60,
            'max_slippage_bps': 30,
            'max_market_impact_bps': 25,
            'min_fill_rate': 0.85
        }
        
        # 性能缓存
        self.recent_metrics = deque(maxlen=100)
        self.performance_cache = {}
        
    async def start_monitoring(self):
        """启动实时监控"""
        if self.monitoring_active:
            logger.warning("执行质量监控已在运行")
            return
            
        self.monitoring_active = True
        logger.info("执行质量监控已启动")
        
        # 可以添加定期清理和分析任务
        asyncio.create_task(self._periodic_analysis())
    
    async def stop_monitoring(self):
        """停止实时监控"""
        self.monitoring_active = False
        logger.info("执行质量监控已停止")
    
    async def analyze_execution_realtime(
        self,
        execution_data: Dict[str, Any],
        market_context: Dict[str, Any] = None
    ) -> Tuple[ExecutionMetrics, List[str]]:
        """实时分析执行质量"""
        
        if market_context is None:
            market_context = {}
            
        # 执行分析
        metrics = self.execution_analyzer.analyze_execution(
            execution_data=execution_data,
            market_context=market_context
        )
        
        # 缓存最近指标
        self.recent_metrics.append(metrics)
        
        # 检查告警条件
        alerts = self._check_alerts(metrics)
        
        if alerts:
            logger.warning(f"执行质量告警 {metrics.session_id}: {alerts}")
        
        return metrics, alerts
    
    def _check_alerts(self, metrics: ExecutionMetrics) -> List[str]:
        """检查告警条件"""
        alerts = []
        
        if metrics.quality_score < self.alert_thresholds['min_quality_score']:
            alerts.append(f"质量评分过低: {metrics.quality_score:.1f}")
            
        if abs(metrics.slippage_bps) > self.alert_thresholds['max_slippage_bps']:
            alerts.append(f"滑点过大: {metrics.slippage_bps:.1f}bps")
            
        if metrics.market_impact_bps > self.alert_thresholds['max_market_impact_bps']:
            alerts.append(f"市场冲击过大: {metrics.market_impact_bps:.1f}bps")
            
        if metrics.fill_rate < self.alert_thresholds['min_fill_rate']:
            alerts.append(f"成交率过低: {metrics.fill_rate:.1%}")
        
        return alerts
    
    async def _periodic_analysis(self):
        """定期分析任务"""
        while self.monitoring_active:
            try:
                # 每30分钟生成一次性能摘要
                await asyncio.sleep(1800)
                
                if len(self.recent_metrics) >= 10:  # 至少10个样本
                    summary = self._generate_performance_summary()
                    logger.info(f"执行性能摘要: {summary}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"定期分析任务错误: {e}")
                await asyncio.sleep(300)  # 出错后等待5分钟
    
    def _generate_performance_summary(self) -> Dict[str, float]:
        """生成性能摘要"""
        recent_metrics = list(self.recent_metrics)[-50:]  # 最近50个
        
        if not recent_metrics:
            return {}
        
        summary = {
            'avg_quality_score': np.mean([m.quality_score for m in recent_metrics]),
            'avg_slippage_bps': np.mean([abs(m.slippage_bps) for m in recent_metrics]),
            'avg_fill_rate': np.mean([m.fill_rate for m in recent_metrics]),
            'avg_market_impact_bps': np.mean([m.market_impact_bps for m in recent_metrics]),
            'alert_rate': len([m for m in recent_metrics if m.anomaly_flags]) / len(recent_metrics)
        }
        
        return summary
    
    def generate_daily_tca_report(self, target_date: datetime = None) -> TCAReport:
        """生成日度TCA报告"""
        if target_date is None:
            target_date = datetime.now().date()
        
        start_time = datetime.combine(target_date, datetime.min.time())
        end_time = datetime.combine(target_date, datetime.max.time())
        
        return self.tca_generator.generate_tca_report(start_time, end_time)
    
    def get_algorithm_rankings(self, days: int = 7) -> Dict[str, Dict[str, float]]:
        """获取算法排名"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        executions = self.execution_analyzer.execution_database.query_executions(
            start_time=start_time,
            end_time=end_time
        )
        
        return self.tca_generator._analyze_algorithm_performance(executions)
    
    def get_venue_rankings(self, days: int = 7) -> Dict[str, Dict[str, float]]:
        """获取交易所排名"""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        executions = self.execution_analyzer.execution_database.query_executions(
            start_time=start_time,
            end_time=end_time
        )
        
        return self.tca_generator._analyze_venue_performance(executions)


# 演示函数
async def demo_execution_quality_analyzer():
    """执行质量分析器演示"""
    
    print("="*80)
    print("DipMaster Trading System - 执行质量分析器 & TCA系统")
    print("="*80)
    
    # 创建质量监控器
    monitor = ExecutionQualityMonitor()
    
    # 启动监控
    await monitor.start_monitoring()
    
    try:
        # 模拟几个执行案例
        execution_cases = [
            # 案例1: 优秀的TWAP执行
            {
                'session_id': 'TWAP_BTC_001',
                'symbol': 'BTCUSDT',
                'side': 'BUY',
                'algorithm': 'twap',
                'target_quantity': 1.5,
                'executed_quantity': 1.48,
                'execution_start': datetime.now() - timedelta(minutes=30),
                'execution_end': datetime.now() - timedelta(minutes=5),
                'fills': [
                    {'quantity': 0.5, 'price': 65000, 'venue': 'binance', 'liquidity_type': 'maker'},
                    {'quantity': 0.48, 'price': 65010, 'venue': 'binance', 'liquidity_type': 'maker'},
                    {'quantity': 0.5, 'price': 65005, 'venue': 'okx', 'liquidity_type': 'taker'}
                ],
                'arrival_price': 65000,
                'total_fees_usd': 45.5
            },
            
            # 案例2: DipMaster 15分钟边界执行
            {
                'session_id': 'DM15_ETH_001', 
                'symbol': 'ETHUSDT',
                'side': 'BUY',
                'algorithm': 'DIPMASTER_15MIN',
                'target_quantity': 10.0,
                'executed_quantity': 10.0,
                'execution_start': datetime.now() - timedelta(minutes=17),
                'execution_end': datetime.now() - timedelta(minutes=15, seconds=30),
                'fills': [
                    {'quantity': 5.0, 'price': 3195, 'venue': 'binance', 'liquidity_type': 'taker'},
                    {'quantity': 5.0, 'price': 3198, 'venue': 'okx', 'liquidity_type': 'taker'}
                ],
                'arrival_price': 3200,
                'total_fees_usd': 32.0
            },
            
            # 案例3: 高滑点执行（问题案例）
            {
                'session_id': 'PROBLEM_SOL_001',
                'symbol': 'SOLUSDT', 
                'side': 'SELL',
                'algorithm': 'market',
                'target_quantity': 500,
                'executed_quantity': 480,
                'execution_start': datetime.now() - timedelta(minutes=5),
                'execution_end': datetime.now() - timedelta(minutes=3),
                'fills': [
                    {'quantity': 480, 'price': 138.5, 'venue': 'bybit', 'liquidity_type': 'taker'}
                ],
                'arrival_price': 140.2,
                'total_fees_usd': 66.5
            }
        ]
        
        print("\n执行质量实时分析:")
        print("-" * 60)
        
        analyzed_metrics = []
        
        for i, case in enumerate(execution_cases, 1):
            print(f"\n案例{i}: {case['session_id']}")
            
            # 构建市场上下文
            market_context = {
                'volume_during_execution': 1000000,  # 执行期间成交量
                'target_boundary_minute': 15 if 'DM15' in case['session_id'] else None,
                'entry_rsi': 35.5 if 'dip' in case['algorithm'].lower() else None,
                'price_drop_confirmed': True if 'dip' in case['algorithm'].lower() else False
            }
            
            # 执行实时分析
            metrics, alerts = await monitor.analyze_execution_realtime(
                execution_data=case,
                market_context=market_context
            )
            
            analyzed_metrics.append(metrics)
            
            # 显示结果
            print(f"  质量评分: {metrics.quality_score:.1f}/100")
            print(f"  滑点: {metrics.slippage_bps:.2f}bps")
            print(f"  市场冲击: {metrics.market_impact_bps:.2f}bps")
            print(f"  成交率: {metrics.fill_rate:.1%}")
            print(f"  Maker比例: {metrics.maker_ratio:.1%}")
            
            if metrics.dipmaster_timing_accuracy:
                print(f"  DipMaster时机精度: {metrics.dipmaster_timing_accuracy:.2f}")
            if metrics.dipmaster_dip_capture_rate:
                print(f"  DipMaster逢跌捕获率: {metrics.dipmaster_dip_capture_rate:.2f}")
                
            if alerts:
                print(f"  ⚠️ 告警: {', '.join(alerts)}")
            else:
                print("  ✅ 无异常")
        
        # 生成TCA报告
        print("\n\nTCA报告生成:")
        print("-" * 60)
        
        # 等待一下让数据库写入完成
        await asyncio.sleep(1)
        
        try:
            tca_report = monitor.generate_daily_tca_report()
            
            print(f"报告ID: {tca_report.report_id}")
            print(f"分析期间: {tca_report.period_start.strftime('%Y-%m-%d %H:%M')} - {tca_report.period_end.strftime('%Y-%m-%d %H:%M')}")
            print(f"总执行数: {tca_report.total_executions}")
            print(f"总交易量: ${tca_report.total_volume_usd:,.0f}")
            print(f"总成本: ${tca_report.total_cost_usd:.2f}")
            print(f"平均成本: {tca_report.avg_cost_bps:.2f}bps")
            
            print("\n算法性能排名:")
            for algo, perf in tca_report.algorithm_performance.items():
                print(f"  {algo}: 质量{perf['avg_quality_score']:.1f}, 滑点{perf['avg_slippage_bps']:.2f}bps")
            
            print("\n异常统计:")
            for anomaly, count in tca_report.anomaly_summary.items():
                print(f"  {anomaly}: {count}次")
            
            print("\n优化建议:")
            for suggestion in tca_report.optimization_suggestions:
                print(f"  • {suggestion}")
                
            if tca_report.dipmaster_analysis:
                print("\nDipMaster专用分析:")
                for key, value in tca_report.dipmaster_analysis.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")
        
        except Exception as e:
            print(f"TCA报告生成失败: {e}")
        
        # 算法和交易所排名
        print("\n\n性能排名 (最近7天):")
        print("-" * 60)
        
        try:
            algo_rankings = monitor.get_algorithm_rankings(days=1)  # 使用1天因为是演示
            
            print("算法质量排名:")
            sorted_algos = sorted(algo_rankings.items(), 
                               key=lambda x: x[1]['avg_quality_score'], reverse=True)
            for algo, metrics in sorted_algos:
                print(f"  {algo}: 质量{metrics['avg_quality_score']:.1f}, 执行{metrics['execution_count']}次")
            
            venue_rankings = monitor.get_venue_rankings(days=1)
            print("\n交易所质量排名:")
            sorted_venues = sorted(venue_rankings.items(),
                                key=lambda x: x[1]['avg_quality_score'], reverse=True)
            for venue, metrics in sorted_venues:
                print(f"  {venue}: 质量{metrics['avg_quality_score']:.1f}, 执行{metrics['execution_count']}次")
        
        except Exception as e:
            print(f"排名分析失败: {e}")
        
    finally:
        await monitor.stop_monitoring()
    
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(demo_execution_quality_analyzer())