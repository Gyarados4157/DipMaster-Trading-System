"""
Execution Reporter
执行报告生成和性能监控系统
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExecutionReport:
    """标准化执行报告格式"""
    orders: List[Dict]
    fills: List[Dict]
    costs: Dict[str, float]
    violations: List[Dict]
    pnl: Dict[str, float]
    latency_ms: float
    ts: str
    symbol: str
    venue: str
    execution_quality: Dict[str, float]


@dataclass
class PerformanceMetrics:
    """性能指标"""
    period_start: datetime
    period_end: datetime
    total_executions: int
    successful_executions: int
    success_rate: float
    avg_slippage_bps: float
    median_slippage_bps: float
    p95_slippage_bps: float
    avg_execution_time_seconds: float
    median_execution_time_seconds: float
    total_volume_usd: float
    total_fees_usd: float
    avg_fill_rate: float
    venue_breakdown: Dict[str, Dict]
    algorithm_performance: Dict[str, Dict]


class ExecutionReporter:
    """执行报告生成器"""
    
    def __init__(self, output_dir: str = "G:\\Github\\Quant\\DipMaster-Trading-System\\results\\execution_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 报告配置
        self.report_templates = {
            'daily_summary': self._generate_daily_summary,
            'execution_detail': self._generate_execution_detail,
            'performance_analysis': self._generate_performance_analysis,
            'risk_assessment': self._generate_risk_assessment,
            'cost_analysis': self._generate_cost_analysis
        }
        
        # 缓存执行数据
        self.execution_cache: List[Dict] = []
        self.performance_cache: Dict[str, Any] = {}
        
    async def generate_execution_report(
        self,
        target_portfolio: Dict,
        execution_results: List[Dict],
        report_type: str = 'execution_detail'
    ) -> ExecutionReport:
        """
        生成标准化执行报告
        
        Args:
            target_portfolio: 目标组合配置
            execution_results: 执行结果列表
            report_type: 报告类型
        
        Returns:
            ExecutionReport: 标准化执行报告
        """
        logger.info(f"生成执行报告: {report_type}")
        
        # 提取订单信息
        orders = self._extract_orders(execution_results)
        
        # 提取成交信息
        fills = self._extract_fills(execution_results)
        
        # 计算执行成本
        costs = await self._calculate_execution_costs(fills)
        
        # 检测风险违规
        violations = await self._detect_violations(execution_results)
        
        # 计算PnL
        pnl = await self._calculate_pnl(fills, target_portfolio)
        
        # 计算延迟指标
        latency_ms = self._calculate_avg_latency(execution_results)
        
        # 计算执行质量指标
        execution_quality = await self._calculate_execution_quality(execution_results)
        
        # 确定主要交易对和交易所
        primary_symbol = self._get_primary_symbol(orders)
        primary_venue = self._get_primary_venue(fills)
        
        report = ExecutionReport(
            orders=orders,
            fills=fills,
            costs=costs,
            violations=violations,
            pnl=pnl,
            latency_ms=latency_ms,
            ts=datetime.now().isoformat(),
            symbol=primary_symbol,
            venue=primary_venue,
            execution_quality=execution_quality
        )
        
        # 保存报告
        await self._save_report(report, report_type)
        
        return report
    
    def _extract_orders(self, execution_results: List[Dict]) -> List[Dict]:
        """提取订单信息"""
        orders = []
        
        for result in execution_results:
            if 'slices' in result:
                for slice_data in result['slices']:
                    order = {
                        "venue": slice_data.get('venue', 'binance'),
                        "symbol": result.get('symbol', 'BTCUSDT'),
                        "side": result.get('side', 'buy'),
                        "qty": slice_data.get('quantity', 0),
                        "tif": slice_data.get('time_in_force', 'IOC'),
                        "order_type": slice_data.get('order_type', 'limit'),
                        "limit_price": slice_data.get('limit_price'),
                        "slice_id": slice_data.get('slice_id', ''),
                        "parent_id": result.get('request_id', '')
                    }
                    orders.append(order)
        
        return orders
    
    def _extract_fills(self, execution_results: List[Dict]) -> List[Dict]:
        """提取成交信息"""
        fills = []
        
        for result in execution_results:
            if result.get('status') == 'completed' and result.get('filled_quantity', 0) > 0:
                # 计算滑点
                arrival_price = result.get('target_arrival_price', result.get('average_price', 0))
                fill_price = result.get('average_price', 0)
                
                if arrival_price > 0 and fill_price > 0:
                    slippage_bps = ((fill_price - arrival_price) / arrival_price) * 10000
                else:
                    slippage_bps = 0
                
                fill = {
                    "order_id": result.get('request_id', ''),
                    "price": fill_price,
                    "qty": result.get('filled_quantity', 0),
                    "slippage_bps": slippage_bps,
                    "venue": "binance",  # 默认值
                    "timestamp": result.get('completion_time', datetime.now().isoformat())
                }
                fills.append(fill)
        
        return fills
    
    async def _calculate_execution_costs(self, fills: List[Dict]) -> Dict[str, float]:
        """计算执行成本"""
        
        if not fills:
            return {
                "fees_usd": 0,
                "impact_bps": 0,
                "spread_cost_usd": 0,
                "total_cost_usd": 0
            }
        
        total_notional = sum(fill['qty'] * fill['price'] for fill in fills)
        
        # 交易费用（假设0.1%）
        fees_usd = total_notional * 0.001
        
        # 市场冲击（基于滑点）
        avg_slippage_bps = np.mean([abs(fill['slippage_bps']) for fill in fills])
        impact_bps = avg_slippage_bps * 0.6  # 假设60%的滑点来自市场冲击
        
        # 价差成本（假设2个基点）
        spread_cost_bps = 2.0
        spread_cost_usd = total_notional * spread_cost_bps / 10000
        
        # 总成本
        total_cost_usd = fees_usd + spread_cost_usd + (total_notional * impact_bps / 10000)
        
        return {
            "fees_usd": fees_usd,
            "impact_bps": impact_bps,
            "spread_cost_usd": spread_cost_usd,
            "total_cost_usd": total_cost_usd
        }
    
    async def _detect_violations(self, execution_results: List[Dict]) -> List[Dict]:
        """检测风险违规"""
        violations = []
        
        for result in execution_results:
            # 检查滑点违规
            if 'slippage_bps' in result:
                slippage = abs(result['slippage_bps'])
                if slippage > 50:  # 50个基点阈值
                    violations.append({
                        "type": "high_slippage",
                        "severity": "high" if slippage > 100 else "medium",
                        "value": slippage,
                        "limit": 50,
                        "order_id": result.get('request_id', ''),
                        "timestamp": result.get('completion_time', datetime.now().isoformat())
                    })
            
            # 检查执行时间违规
            if 'execution_time_seconds' in result:
                exec_time = result['execution_time_seconds']
                if exec_time > 300:  # 5分钟阈值
                    violations.append({
                        "type": "execution_timeout",
                        "severity": "medium",
                        "value": exec_time,
                        "limit": 300,
                        "order_id": result.get('request_id', ''),
                        "timestamp": result.get('completion_time', datetime.now().isoformat())
                    })
            
            # 检查成交率违规
            target_qty = result.get('target_quantity', 0)
            filled_qty = result.get('filled_quantity', 0)
            if target_qty > 0:
                fill_rate = filled_qty / target_qty
                if fill_rate < 0.9:  # 90%成交率阈值
                    violations.append({
                        "type": "low_fill_rate",
                        "severity": "medium" if fill_rate > 0.7 else "high",
                        "value": fill_rate,
                        "limit": 0.9,
                        "order_id": result.get('request_id', ''),
                        "timestamp": result.get('completion_time', datetime.now().isoformat())
                    })
        
        return violations
    
    async def _calculate_pnl(self, fills: List[Dict], target_portfolio: Dict) -> Dict[str, float]:
        """计算PnL"""
        
        if not fills:
            return {"realized": 0, "unrealized": 0}
        
        # 简化的PnL计算
        total_cost = sum(fill['qty'] * fill['price'] for fill in fills)
        
        # 假设市场价格变化1%
        market_move = 0.01
        unrealized_pnl = total_cost * market_move
        
        # 实现PnL（基于执行成本）
        realized_pnl = -sum(abs(fill['slippage_bps']) * fill['qty'] * fill['price'] / 10000 for fill in fills)
        
        return {
            "realized": realized_pnl,
            "unrealized": unrealized_pnl
        }
    
    def _calculate_avg_latency(self, execution_results: List[Dict]) -> float:
        """计算平均延迟"""
        latencies = []
        
        for result in execution_results:
            # 从执行时间估算延迟
            exec_time = result.get('execution_time_seconds', 0)
            if exec_time > 0:
                # 假设延迟是执行时间的一小部分
                estimated_latency = min(exec_time * 1000 / 10, 1000)  # 最多1秒
                latencies.append(estimated_latency)
        
        return np.mean(latencies) if latencies else 100.0  # 默认100ms
    
    async def _calculate_execution_quality(self, execution_results: List[Dict]) -> Dict[str, float]:
        """计算执行质量指标"""
        
        if not execution_results:
            return {
                "arrival_slippage_bps": 0,
                "vwap_slippage_bps": 0,
                "fill_rate": 0,
                "passive_ratio": 0
            }
        
        # 到达价格滑点
        slippages = [result.get('slippage_bps', 0) for result in execution_results]
        avg_slippage = np.mean([abs(s) for s in slippages])
        
        # VWAP滑点（简化计算）
        vwap_slippage = avg_slippage * 0.8  # 假设相对VWAP较好
        
        # 成交率
        fill_rates = []
        for result in execution_results:
            target_qty = result.get('target_quantity', 0)
            filled_qty = result.get('filled_quantity', 0)
            if target_qty > 0:
                fill_rates.append(filled_qty / target_qty)
        
        avg_fill_rate = np.mean(fill_rates) if fill_rates else 0
        
        # 被动比例（简化为50%）
        passive_ratio = 0.5
        
        return {
            "arrival_slippage_bps": avg_slippage,
            "vwap_slippage_bps": vwap_slippage,
            "fill_rate": avg_fill_rate,
            "passive_ratio": passive_ratio
        }
    
    def _get_primary_symbol(self, orders: List[Dict]) -> str:
        """获取主要交易对"""
        if not orders:
            return "BTCUSDT"
        
        # 统计交易对频率
        symbol_counts = {}
        for order in orders:
            symbol = order.get('symbol', 'BTCUSDT')
            symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        return max(symbol_counts.keys(), key=lambda k: symbol_counts[k])
    
    def _get_primary_venue(self, fills: List[Dict]) -> str:
        """获取主要交易所"""
        if not fills:
            return "binance"
        
        # 统计交易所频率
        venue_counts = {}
        for fill in fills:
            venue = fill.get('venue', 'binance')
            venue_counts[venue] = venue_counts.get(venue, 0) + 1
        
        return max(venue_counts.keys(), key=lambda k: venue_counts[k])
    
    async def _save_report(self, report: ExecutionReport, report_type: str):
        """保存报告到文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report_type}_{report.symbol}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(asdict(report), f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"执行报告已保存: {filepath}")
            
        except Exception as e:
            logger.error(f"保存报告失败: {e}")
    
    async def generate_performance_summary(
        self,
        execution_data: List[Dict],
        period_start: Optional[datetime] = None,
        period_end: Optional[datetime] = None
    ) -> PerformanceMetrics:
        """生成性能摘要"""
        
        if not execution_data:
            return PerformanceMetrics(
                period_start=datetime.now(),
                period_end=datetime.now(),
                total_executions=0,
                successful_executions=0,
                success_rate=0,
                avg_slippage_bps=0,
                median_slippage_bps=0,
                p95_slippage_bps=0,
                avg_execution_time_seconds=0,
                median_execution_time_seconds=0,
                total_volume_usd=0,
                total_fees_usd=0,
                avg_fill_rate=0,
                venue_breakdown={},
                algorithm_performance={}
            )
        
        # 时间范围
        if not period_start:
            period_start = datetime.now() - timedelta(days=1)
        if not period_end:
            period_end = datetime.now()
        
        # 过滤时间范围内的数据
        filtered_data = []
        for data in execution_data:
            if 'completion_time' in data:
                try:
                    completion_time = datetime.fromisoformat(data['completion_time'])
                    if period_start <= completion_time <= period_end:
                        filtered_data.append(data)
                except:
                    filtered_data.append(data)  # 如果解析失败，包含在内
            else:
                filtered_data.append(data)
        
        # 基本统计
        total_executions = len(filtered_data)
        successful_executions = len([d for d in filtered_data if d.get('status') == 'completed'])
        success_rate = successful_executions / total_executions if total_executions > 0 else 0
        
        # 滑点统计
        slippages = [abs(d.get('slippage_bps', 0)) for d in filtered_data if 'slippage_bps' in d]
        avg_slippage_bps = np.mean(slippages) if slippages else 0
        median_slippage_bps = np.median(slippages) if slippages else 0
        p95_slippage_bps = np.percentile(slippages, 95) if slippages else 0
        
        # 执行时间统计
        exec_times = [d.get('execution_time_seconds', 0) for d in filtered_data if 'execution_time_seconds' in d]
        avg_execution_time = np.mean(exec_times) if exec_times else 0
        median_execution_time = np.median(exec_times) if exec_times else 0
        
        # 成交量和费用
        total_volume_usd = 0
        total_fees_usd = 0
        fill_rates = []
        
        for data in filtered_data:
            filled_qty = data.get('filled_quantity', 0)
            avg_price = data.get('average_price', 0)
            if filled_qty > 0 and avg_price > 0:
                volume = filled_qty * avg_price
                total_volume_usd += volume
                total_fees_usd += data.get('total_fees', volume * 0.001)  # 默认0.1%费率
                
                target_qty = data.get('target_quantity', filled_qty)
                if target_qty > 0:
                    fill_rates.append(filled_qty / target_qty)
        
        avg_fill_rate = np.mean(fill_rates) if fill_rates else 0
        
        # 交易所分布分析
        venue_breakdown = self._analyze_venue_breakdown(filtered_data)
        
        # 算法性能分析
        algorithm_performance = self._analyze_algorithm_performance(filtered_data)
        
        return PerformanceMetrics(
            period_start=period_start,
            period_end=period_end,
            total_executions=total_executions,
            successful_executions=successful_executions,
            success_rate=success_rate,
            avg_slippage_bps=avg_slippage_bps,
            median_slippage_bps=median_slippage_bps,
            p95_slippage_bps=p95_slippage_bps,
            avg_execution_time_seconds=avg_execution_time,
            median_execution_time_seconds=median_execution_time,
            total_volume_usd=total_volume_usd,
            total_fees_usd=total_fees_usd,
            avg_fill_rate=avg_fill_rate,
            venue_breakdown=venue_breakdown,
            algorithm_performance=algorithm_performance
        )
    
    def _analyze_venue_breakdown(self, execution_data: List[Dict]) -> Dict[str, Dict]:
        """分析交易所分布"""
        venue_stats = {}
        
        for data in execution_data:
            venue = data.get('venue', 'unknown')
            if venue not in venue_stats:
                venue_stats[venue] = {
                    'executions': 0,
                    'volume_usd': 0,
                    'avg_slippage_bps': 0,
                    'success_rate': 0
                }
            
            venue_stats[venue]['executions'] += 1
            
            filled_qty = data.get('filled_quantity', 0)
            avg_price = data.get('average_price', 0)
            if filled_qty > 0 and avg_price > 0:
                venue_stats[venue]['volume_usd'] += filled_qty * avg_price
        
        # 计算平均值
        for venue, stats in venue_stats.items():
            venue_data = [d for d in execution_data if d.get('venue') == venue]
            
            slippages = [abs(d.get('slippage_bps', 0)) for d in venue_data if 'slippage_bps' in d]
            stats['avg_slippage_bps'] = np.mean(slippages) if slippages else 0
            
            successful = len([d for d in venue_data if d.get('status') == 'completed'])
            stats['success_rate'] = successful / len(venue_data) if venue_data else 0
        
        return venue_stats
    
    def _analyze_algorithm_performance(self, execution_data: List[Dict]) -> Dict[str, Dict]:
        """分析算法性能"""
        algo_stats = {}
        
        for data in execution_data:
            # 从执行模式推断算法
            algo = data.get('execution_mode', 'balanced')
            if algo not in algo_stats:
                algo_stats[algo] = {
                    'executions': 0,
                    'avg_slippage_bps': 0,
                    'avg_execution_time_seconds': 0,
                    'success_rate': 0,
                    'avg_fill_rate': 0
                }
            
            algo_stats[algo]['executions'] += 1
        
        # 计算平均值
        for algo, stats in algo_stats.items():
            algo_data = [d for d in execution_data if d.get('execution_mode') == algo]
            
            slippages = [abs(d.get('slippage_bps', 0)) for d in algo_data if 'slippage_bps' in d]
            stats['avg_slippage_bps'] = np.mean(slippages) if slippages else 0
            
            exec_times = [d.get('execution_time_seconds', 0) for d in algo_data if 'execution_time_seconds' in d]
            stats['avg_execution_time_seconds'] = np.mean(exec_times) if exec_times else 0
            
            successful = len([d for d in algo_data if d.get('status') == 'completed'])
            stats['success_rate'] = successful / len(algo_data) if algo_data else 0
            
            fill_rates = []
            for d in algo_data:
                target_qty = d.get('target_quantity', 0)
                filled_qty = d.get('filled_quantity', 0)
                if target_qty > 0:
                    fill_rates.append(filled_qty / target_qty)
            stats['avg_fill_rate'] = np.mean(fill_rates) if fill_rates else 0
        
        return algo_stats
    
    async def generate_daily_report(self, date: datetime) -> Dict:
        """生成日报"""
        # 从缓存或数据库获取当日执行数据
        daily_data = [d for d in self.execution_cache 
                     if datetime.fromisoformat(d.get('completion_time', '2023-01-01')).date() == date.date()]
        
        if not daily_data:
            return {"error": f"没有找到 {date.date()} 的执行数据"}
        
        # 生成性能摘要
        performance = await self.generate_performance_summary(
            daily_data,
            datetime.combine(date.date(), datetime.min.time()),
            datetime.combine(date.date(), datetime.max.time())
        )
        
        # 生成详细报告
        report = {
            "date": date.date().isoformat(),
            "summary": {
                "total_executions": performance.total_executions,
                "successful_executions": performance.successful_executions,
                "success_rate": performance.success_rate,
                "total_volume_usd": performance.total_volume_usd,
                "total_fees_usd": performance.total_fees_usd
            },
            "performance_metrics": asdict(performance),
            "top_performing_venues": self._get_top_venues(performance.venue_breakdown),
            "recommendations": self._generate_daily_recommendations(performance)
        }
        
        # 保存日报
        filename = f"daily_report_{date.strftime('%Y%m%d')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        return report
    
    def _get_top_venues(self, venue_breakdown: Dict[str, Dict]) -> List[Dict]:
        """获取表现最好的交易所"""
        venues = []
        
        for venue, stats in venue_breakdown.items():
            score = (stats['success_rate'] * 0.4 + 
                    (1 / (1 + stats['avg_slippage_bps'] / 10)) * 0.6)  # 综合评分
            
            venues.append({
                "venue": venue,
                "score": score,
                "executions": stats['executions'],
                "success_rate": stats['success_rate'],
                "avg_slippage_bps": stats['avg_slippage_bps']
            })
        
        return sorted(venues, key=lambda x: x['score'], reverse=True)[:3]
    
    def _generate_daily_recommendations(self, performance: PerformanceMetrics) -> List[str]:
        """生成日报建议"""
        recommendations = []
        
        if performance.avg_slippage_bps > 10:
            recommendations.append("平均滑点偏高，建议调整订单分割策略或使用更多被动订单")
        
        if performance.success_rate < 0.9:
            recommendations.append("成功率偏低，检查网络连接和API稳定性")
        
        if performance.avg_fill_rate < 0.95:
            recommendations.append("成交率偏低，考虑调整订单大小或执行时间")
        
        if performance.avg_execution_time_seconds > 120:
            recommendations.append("执行时间过长，考虑增加并发执行或优化路由策略")
        
        # 找出表现最好的算法
        if performance.algorithm_performance:
            best_algo = min(performance.algorithm_performance.keys(),
                          key=lambda a: performance.algorithm_performance[a]['avg_slippage_bps'])
            recommendations.append(f"推荐使用 {best_algo} 算法，表现最优")
        
        return recommendations
    
    def add_execution_data(self, execution_data: Dict):
        """添加执行数据到缓存"""
        self.execution_cache.append(execution_data)
        
        # 保持缓存大小
        if len(self.execution_cache) > 10000:
            self.execution_cache = self.execution_cache[-5000:]
    
    def get_cached_data(self, days: int = 7) -> List[Dict]:
        """获取缓存的执行数据"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        return [
            data for data in self.execution_cache
            if datetime.fromisoformat(data.get('completion_time', '2023-01-01')) >= cutoff_date
        ]


# 使用示例
async def demo_execution_reporter():
    """演示执行报告生成"""
    
    reporter = ExecutionReporter()
    
    # 模拟执行数据
    execution_results = [
        {
            'request_id': 'EXE_001',
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'target_quantity': 1.0,
            'filled_quantity': 0.98,
            'average_price': 50100,
            'target_arrival_price': 50000,
            'slippage_bps': 20,
            'execution_time_seconds': 45,
            'total_fees': 50.1,
            'status': 'completed',
            'completion_time': datetime.now().isoformat()
        }
    ]
    
    target_portfolio = {
        'weights': [{'symbol': 'BTCUSDT', 'usd_size': 50000}]
    }
    
    # 生成执行报告
    report = await reporter.generate_execution_report(
        target_portfolio, execution_results, 'execution_detail'
    )
    
    print("执行报告生成完成:")
    print(f"总成本: ${report.costs['total_cost_usd']:.2f}")
    print(f"平均滑点: {report.execution_quality['arrival_slippage_bps']:.2f}bps")
    print(f"成交率: {report.execution_quality['fill_rate']:.2%}")
    print(f"违规数量: {len(report.violations)}")


if __name__ == "__main__":
    asyncio.run(demo_execution_reporter())