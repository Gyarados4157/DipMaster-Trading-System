"""
DipMaster Enhanced V4 - Execution Analytics
Advanced analytics and performance tracking for order execution quality
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import sqlite3
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class ExecutionBenchmark:
    """Execution benchmark metrics"""
    arrival_price: float
    vwap_price: float
    twap_price: float
    market_close_price: float
    benchmark_period: str  # '1min', '5min', '15min', '30min'

@dataclass
class TransactionCostAnalysis:
    """Transaction Cost Analysis (TCA) metrics"""
    implementation_shortfall_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    fees_bps: float
    opportunity_cost_bps: float
    total_cost_bps: float
    
@dataclass
class ExecutionScore:
    """Overall execution quality score"""
    overall_score: float  # 0-100
    cost_score: float
    speed_score: float
    market_impact_score: float
    fill_rate_score: float
    consistency_score: float

class ExecutionDatabase:
    """Database for storing execution metrics"""
    
    def __init__(self, db_path: str = "G:\\Github\\Quant\\DipMaster-Trading-System\\data\\execution_analytics.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    fill_price REAL NOT NULL,
                    arrival_price REAL NOT NULL,
                    slippage_bps REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    venue TEXT NOT NULL,
                    algorithm TEXT NOT NULL,
                    parent_order_id TEXT NOT NULL,
                    slice_id TEXT NOT NULL,
                    fees REAL NOT NULL,
                    liquidity_type TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT PRIMARY KEY,
                    total_volume_usd REAL NOT NULL,
                    avg_slippage_bps REAL NOT NULL,
                    avg_latency_ms REAL NOT NULL,
                    fill_rate REAL NOT NULL,
                    total_fees_usd REAL NOT NULL,
                    implementation_shortfall_bps REAL NOT NULL,
                    execution_score REAL NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS algorithm_performance (
                    algorithm TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    avg_slippage_bps REAL NOT NULL,
                    avg_latency_ms REAL NOT NULL,
                    market_impact_bps REAL NOT NULL,
                    execution_score REAL NOT NULL,
                    PRIMARY KEY (algorithm, symbol, date)
                )
            """)
    
    def insert_execution(self, execution_data: Dict):
        """Insert execution record"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO executions (
                    timestamp, symbol, side, quantity, fill_price, arrival_price,
                    slippage_bps, latency_ms, venue, algorithm, parent_order_id,
                    slice_id, fees, liquidity_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                execution_data['timestamp'],
                execution_data['symbol'],
                execution_data['side'],
                execution_data['quantity'],
                execution_data['fill_price'],
                execution_data['arrival_price'],
                execution_data['slippage_bps'],
                execution_data['latency_ms'],
                execution_data['venue'],
                execution_data['algorithm'],
                execution_data['parent_order_id'],
                execution_data['slice_id'],
                execution_data['fees'],
                execution_data['liquidity_type']
            ))
    
    def get_daily_stats(self, date: str) -> Optional[Dict]:
        """Get daily execution statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM daily_performance WHERE date = ?
            """, (date,))
            row = cursor.fetchone()
            
            if row:
                columns = [desc[0] for desc in cursor.description]
                return dict(zip(columns, row))
            return None
    
    def get_symbol_performance(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get performance data for specific symbol"""
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM executions 
                WHERE symbol = ? AND date(timestamp) >= date('now', '-{} days')
                ORDER BY timestamp
            """.format(days)
            
            return pd.read_sql_query(query, conn, params=(symbol,))

class BenchmarkCalculator:
    """Calculate execution benchmarks (VWAP, TWAP, etc.)"""
    
    def __init__(self):
        self.price_cache = defaultdict(list)
        self.volume_cache = defaultdict(list)
    
    async def calculate_vwap_benchmark(self, 
                                     symbol: str, 
                                     start_time: datetime, 
                                     end_time: datetime) -> float:
        """Calculate VWAP benchmark for the execution period"""
        
        # In production, this would fetch real market data
        # For now, simulate VWAP calculation
        duration_minutes = (end_time - start_time).total_seconds() / 60
        
        # Simulate price movement and volume
        base_price = 50000  # Simplified
        price_variance = 0.001  # 0.1% variance
        
        total_value = 0
        total_volume = 0
        
        for i in range(int(duration_minutes)):
            # Simulate minute-by-minute data
            price = base_price * (1 + np.random.normal(0, price_variance))
            volume = np.random.exponential(100)  # Exponential volume distribution
            
            total_value += price * volume
            total_volume += volume
        
        vwap = total_value / total_volume if total_volume > 0 else base_price
        return vwap
    
    async def calculate_twap_benchmark(self, 
                                     symbol: str, 
                                     start_time: datetime, 
                                     end_time: datetime) -> float:
        """Calculate TWAP benchmark for the execution period"""
        
        duration_minutes = (end_time - start_time).total_seconds() / 60
        base_price = 50000
        price_variance = 0.001
        
        prices = []
        for i in range(int(duration_minutes)):
            price = base_price * (1 + np.random.normal(0, price_variance))
            prices.append(price)
        
        return np.mean(prices) if prices else base_price
    
    async def calculate_implementation_shortfall(self,
                                               arrival_price: float,
                                               executed_price: float,
                                               benchmark_price: float,
                                               side: str) -> float:
        """Calculate implementation shortfall vs benchmark"""
        
        if side == 'BUY':
            # For buy orders, positive shortfall is worse (paid more)
            shortfall = ((executed_price - arrival_price) / arrival_price) * 10000
        else:
            # For sell orders, negative shortfall is worse (received less)
            shortfall = ((arrival_price - executed_price) / arrival_price) * 10000
        
        return shortfall

class ExecutionAnalyzer:
    """Advanced execution analytics and performance tracking"""
    
    def __init__(self):
        self.database = ExecutionDatabase()
        self.benchmark_calc = BenchmarkCalculator()
        
        # Performance thresholds
        self.excellent_slippage_bps = 2.0
        self.good_slippage_bps = 5.0
        self.poor_slippage_bps = 15.0
        
        self.excellent_latency_ms = 50
        self.good_latency_ms = 150
        self.poor_latency_ms = 500
    
    async def analyze_execution_performance(self, 
                                          execution_data: List[Dict],
                                          benchmarks: Dict[str, ExecutionBenchmark]) -> Dict:
        """Comprehensive execution performance analysis"""
        
        if not execution_data:
            return {}
        
        # Group executions by parent order
        parent_orders = defaultdict(list)
        for exec_data in execution_data:
            parent_orders[exec_data.get('parent_order_id', 'unknown')].append(exec_data)
        
        analysis_results = {
            'summary': {},
            'parent_order_analysis': {},
            'algorithm_comparison': {},
            'time_analysis': {},
            'cost_breakdown': {},
            'quality_scores': {}
        }
        
        # Overall summary metrics
        total_volume = sum(e['quantity'] * e['fill_price'] for e in execution_data)
        avg_slippage = np.mean([e['slippage_bps'] for e in execution_data])
        avg_latency = np.mean([e['latency_ms'] for e in execution_data])
        fill_rate = len([e for e in execution_data if e['quantity'] > 0]) / len(execution_data)
        
        analysis_results['summary'] = {
            'total_volume_usd': total_volume,
            'total_orders': len(execution_data),
            'avg_slippage_bps': avg_slippage,
            'avg_latency_ms': avg_latency,
            'fill_rate': fill_rate,
            'execution_period_minutes': self._calculate_execution_duration(execution_data)
        }
        
        # Analyze each parent order
        for parent_id, executions in parent_orders.items():
            parent_analysis = await self._analyze_parent_order(executions, benchmarks)
            analysis_results['parent_order_analysis'][parent_id] = parent_analysis
        
        # Algorithm performance comparison
        algorithm_performance = await self._analyze_algorithm_performance(execution_data)
        analysis_results['algorithm_comparison'] = algorithm_performance
        
        # Time-based analysis
        time_analysis = await self._analyze_time_patterns(execution_data)
        analysis_results['time_analysis'] = time_analysis
        
        # Cost breakdown
        cost_breakdown = await self._analyze_execution_costs(execution_data, benchmarks)
        analysis_results['cost_breakdown'] = cost_breakdown
        
        # Quality scores
        quality_scores = await self._calculate_quality_scores(execution_data)
        analysis_results['quality_scores'] = quality_scores
        
        return analysis_results
    
    async def _analyze_parent_order(self, 
                                  executions: List[Dict], 
                                  benchmarks: Dict[str, ExecutionBenchmark]) -> Dict:
        """Analyze performance of a parent order (all its slices)"""
        
        if not executions:
            return {}
        
        symbol = executions[0]['symbol']
        total_qty = sum(e['quantity'] for e in executions)
        total_value = sum(e['quantity'] * e['fill_price'] for e in executions)
        avg_price = total_value / total_qty if total_qty > 0 else 0
        
        arrival_price = executions[0]['arrival_price']
        
        # Calculate vs benchmarks
        benchmark = benchmarks.get(symbol)
        if benchmark:
            vwap_slippage = ((avg_price - benchmark.vwap_price) / benchmark.vwap_price) * 10000
            arrival_slippage = ((avg_price - arrival_price) / arrival_price) * 10000
        else:
            vwap_slippage = 0
            arrival_slippage = 0
        
        # Timing analysis
        start_time = min(datetime.fromisoformat(e['timestamp']) for e in executions)
        end_time = max(datetime.fromisoformat(e['timestamp']) for e in executions)
        execution_duration = (end_time - start_time).total_seconds() / 60
        
        # Market impact estimation
        market_impact = await self._estimate_market_impact(executions)
        
        return {
            'symbol': symbol,
            'total_quantity': total_qty,
            'avg_fill_price': avg_price,
            'arrival_price': arrival_price,
            'slices_count': len(executions),
            'execution_duration_minutes': execution_duration,
            'arrival_slippage_bps': arrival_slippage,
            'vwap_slippage_bps': vwap_slippage,
            'market_impact_bps': market_impact,
            'total_fees': sum(e['fees'] for e in executions),
            'passive_ratio': len([e for e in executions if e['liquidity_type'] == 'maker']) / len(executions)
        }
    
    async def _analyze_algorithm_performance(self, execution_data: List[Dict]) -> Dict:
        """Compare performance across different execution algorithms"""
        
        algorithm_stats = defaultdict(lambda: {
            'executions': [],
            'avg_slippage': 0,
            'avg_latency': 0,
            'fill_rate': 0,
            'market_impact': 0
        })
        
        # Group by algorithm
        for exec_data in execution_data:
            algo = exec_data.get('algorithm', 'unknown')
            algorithm_stats[algo]['executions'].append(exec_data)
        
        # Calculate statistics for each algorithm
        for algo, stats in algorithm_stats.items():
            executions = stats['executions']
            if executions:
                stats['avg_slippage'] = np.mean([e['slippage_bps'] for e in executions])
                stats['avg_latency'] = np.mean([e['latency_ms'] for e in executions])
                stats['fill_rate'] = len([e for e in executions if e['quantity'] > 0]) / len(executions)
                stats['market_impact'] = await self._estimate_market_impact(executions)
                stats['execution_count'] = len(executions)
                
                # Remove executions list from output (too verbose)
                del stats['executions']
        
        return dict(algorithm_stats)
    
    async def _analyze_time_patterns(self, execution_data: List[Dict]) -> Dict:
        """Analyze execution performance patterns over time"""
        
        # Group by hour of day
        hourly_performance = defaultdict(list)
        
        for exec_data in execution_data:
            timestamp = datetime.fromisoformat(exec_data['timestamp'])
            hour = timestamp.hour
            hourly_performance[hour].append(exec_data['slippage_bps'])
        
        # Calculate hourly averages
        hourly_avg_slippage = {}
        for hour, slippages in hourly_performance.items():
            hourly_avg_slippage[hour] = np.mean(slippages)
        
        # Find best and worst hours
        if hourly_avg_slippage:
            best_hour = min(hourly_avg_slippage.keys(), key=lambda h: abs(hourly_avg_slippage[h]))
            worst_hour = max(hourly_avg_slippage.keys(), key=lambda h: abs(hourly_avg_slippage[h]))
        else:
            best_hour = worst_hour = None
        
        return {
            'hourly_avg_slippage': hourly_avg_slippage,
            'best_execution_hour': best_hour,
            'worst_execution_hour': worst_hour,
            'execution_time_spread_hours': len(hourly_avg_slippage)
        }
    
    async def _analyze_execution_costs(self, 
                                     execution_data: List[Dict], 
                                     benchmarks: Dict[str, ExecutionBenchmark]) -> Dict:
        """Detailed breakdown of execution costs"""
        
        total_fees = sum(e['fees'] for e in execution_data)
        total_volume = sum(e['quantity'] * e['fill_price'] for e in execution_data)
        
        # Market impact cost
        total_market_impact = 0
        for exec_data in execution_data:
            impact_cost = abs(exec_data['slippage_bps']) * exec_data['quantity'] * exec_data['fill_price'] / 10000
            total_market_impact += impact_cost
        
        # Spread cost (estimated)
        estimated_spread_cost = total_volume * 0.0005  # 5bps estimated spread
        
        # Timing cost (difference between execution time and arrival)
        timing_cost = 0  # Simplified for now
        
        total_cost = total_fees + total_market_impact + estimated_spread_cost + timing_cost
        
        return {
            'total_cost_usd': total_cost,
            'total_cost_bps': (total_cost / total_volume) * 10000 if total_volume > 0 else 0,
            'fees_usd': total_fees,
            'fees_bps': (total_fees / total_volume) * 10000 if total_volume > 0 else 0,
            'market_impact_usd': total_market_impact,
            'market_impact_bps': (total_market_impact / total_volume) * 10000 if total_volume > 0 else 0,
            'spread_cost_usd': estimated_spread_cost,
            'spread_cost_bps': (estimated_spread_cost / total_volume) * 10000 if total_volume > 0 else 0,
            'timing_cost_usd': timing_cost,
            'timing_cost_bps': (timing_cost / total_volume) * 10000 if total_volume > 0 else 0
        }
    
    async def _calculate_quality_scores(self, execution_data: List[Dict]) -> ExecutionScore:
        """Calculate execution quality scores (0-100)"""
        
        if not execution_data:
            return ExecutionScore(0, 0, 0, 0, 0, 0)
        
        # Cost score (based on slippage)
        avg_slippage = np.mean([abs(e['slippage_bps']) for e in execution_data])
        if avg_slippage <= self.excellent_slippage_bps:
            cost_score = 100
        elif avg_slippage <= self.good_slippage_bps:
            cost_score = 80
        elif avg_slippage <= self.poor_slippage_bps:
            cost_score = 60
        else:
            cost_score = max(20, 60 - (avg_slippage - self.poor_slippage_bps) * 2)
        
        # Speed score (based on latency)
        avg_latency = np.mean([e['latency_ms'] for e in execution_data])
        if avg_latency <= self.excellent_latency_ms:
            speed_score = 100
        elif avg_latency <= self.good_latency_ms:
            speed_score = 80
        elif avg_latency <= self.poor_latency_ms:
            speed_score = 60
        else:
            speed_score = max(20, 60 - (avg_latency - self.poor_latency_ms) / 10)
        
        # Fill rate score
        fill_rate = len([e for e in execution_data if e['quantity'] > 0]) / len(execution_data)
        fill_rate_score = fill_rate * 100
        
        # Market impact score (inverse of impact)
        market_impact = await self._estimate_market_impact(execution_data)
        if market_impact <= 5:
            market_impact_score = 100
        elif market_impact <= 15:
            market_impact_score = 80
        else:
            market_impact_score = max(20, 80 - (market_impact - 15) * 2)
        
        # Consistency score (low variance is better)
        slippage_std = np.std([e['slippage_bps'] for e in execution_data])
        if slippage_std <= 2:
            consistency_score = 100
        elif slippage_std <= 5:
            consistency_score = 80
        else:
            consistency_score = max(20, 80 - (slippage_std - 5) * 5)
        
        # Overall score (weighted average)
        overall_score = (
            cost_score * 0.3 +
            speed_score * 0.2 +
            fill_rate_score * 0.2 +
            market_impact_score * 0.2 +
            consistency_score * 0.1
        )
        
        return ExecutionScore(
            overall_score=overall_score,
            cost_score=cost_score,
            speed_score=speed_score,
            market_impact_score=market_impact_score,
            fill_rate_score=fill_rate_score,
            consistency_score=consistency_score
        )
    
    async def _estimate_market_impact(self, executions: List[Dict]) -> float:
        """Estimate market impact in basis points"""
        
        if not executions:
            return 0
        
        # Simplified market impact model
        total_volume = sum(e['quantity'] * e['fill_price'] for e in executions)
        avg_slippage = np.mean([abs(e['slippage_bps']) for e in executions])
        
        # Assume market impact is portion of slippage
        market_impact = avg_slippage * 0.6  # 60% of slippage attributed to market impact
        
        return market_impact
    
    def _calculate_execution_duration(self, execution_data: List[Dict]) -> float:
        """Calculate total execution duration in minutes"""
        
        if not execution_data:
            return 0
        
        timestamps = [datetime.fromisoformat(e['timestamp']) for e in execution_data]
        duration = (max(timestamps) - min(timestamps)).total_seconds() / 60
        
        return duration
    
    async def generate_execution_report(self, 
                                      execution_data: List[Dict],
                                      target_portfolio: Dict) -> Dict:
        """Generate comprehensive execution analytics report"""
        
        # Calculate benchmarks
        benchmarks = {}
        for position in target_portfolio.get('weights', []):
            symbol = position['symbol']
            # In production, calculate real benchmarks
            benchmarks[symbol] = ExecutionBenchmark(
                arrival_price=50000,  # Simplified
                vwap_price=50050,
                twap_price=50025,
                market_close_price=50100,
                benchmark_period='30min'
            )
        
        # Perform analysis
        analysis = await self.analyze_execution_performance(execution_data, benchmarks)
        
        # Calculate TCA
        tca_metrics = await self._calculate_tca_metrics(execution_data, benchmarks)
        
        # Store results in database
        for exec_data in execution_data:
            self.database.insert_execution(exec_data)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'execution_analysis': analysis,
            'transaction_cost_analysis': asdict(tca_metrics),
            'benchmarks': {k: asdict(v) for k, v in benchmarks.items()},
            'recommendations': await self._generate_recommendations(analysis),
            'risk_assessment': await self._assess_execution_risks(execution_data)
        }
        
        return report
    
    async def _calculate_tca_metrics(self, 
                                   execution_data: List[Dict], 
                                   benchmarks: Dict[str, ExecutionBenchmark]) -> TransactionCostAnalysis:
        """Calculate Transaction Cost Analysis metrics"""
        
        if not execution_data:
            return TransactionCostAnalysis(0, 0, 0, 0, 0, 0)
        
        total_volume = sum(e['quantity'] * e['fill_price'] for e in execution_data)
        
        # Implementation shortfall
        implementation_shortfall = np.mean([e['slippage_bps'] for e in execution_data])
        
        # Market impact (estimated)
        market_impact = await self._estimate_market_impact(execution_data)
        
        # Timing cost (simplified)
        timing_cost = 1.0  # 1bp assumed timing cost
        
        # Fees
        total_fees = sum(e['fees'] for e in execution_data)
        fees_bps = (total_fees / total_volume) * 10000 if total_volume > 0 else 0
        
        # Opportunity cost (simplified)
        opportunity_cost = 0.5  # 0.5bp assumed opportunity cost
        
        total_cost = abs(implementation_shortfall) + market_impact + timing_cost + fees_bps + opportunity_cost
        
        return TransactionCostAnalysis(
            implementation_shortfall_bps=implementation_shortfall,
            market_impact_bps=market_impact,
            timing_cost_bps=timing_cost,
            fees_bps=fees_bps,
            opportunity_cost_bps=opportunity_cost,
            total_cost_bps=total_cost
        )
    
    async def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on execution analysis"""
        
        recommendations = []
        
        summary = analysis.get('summary', {})
        avg_slippage = summary.get('avg_slippage_bps', 0)
        avg_latency = summary.get('avg_latency_ms', 0)
        fill_rate = summary.get('fill_rate', 1)
        
        # Slippage recommendations
        if avg_slippage > self.poor_slippage_bps:
            recommendations.append("High slippage detected. Consider using more passive order types or smaller slice sizes.")
        elif avg_slippage > self.good_slippage_bps:
            recommendations.append("Moderate slippage observed. Review execution timing and order placement strategy.")
        
        # Latency recommendations
        if avg_latency > self.poor_latency_ms:
            recommendations.append("High latency detected. Check network connectivity and consider co-location.")
        elif avg_latency > self.good_latency_ms:
            recommendations.append("Elevated latency observed. Monitor execution infrastructure performance.")
        
        # Fill rate recommendations
        if fill_rate < 0.9:
            recommendations.append("Low fill rate detected. Consider adjusting order aggressiveness or slice sizing.")
        
        # Algorithm recommendations
        algo_performance = analysis.get('algorithm_comparison', {})
        if len(algo_performance) > 1:
            best_algo = min(algo_performance.keys(), 
                          key=lambda a: abs(algo_performance[a].get('avg_slippage', float('inf'))))
            recommendations.append(f"Best performing algorithm: {best_algo}. Consider increasing allocation.")
        
        return recommendations
    
    async def _assess_execution_risks(self, execution_data: List[Dict]) -> Dict:
        """Assess execution-related risks"""
        
        risks = {
            'high_slippage_risk': False,
            'latency_risk': False,
            'concentration_risk': False,
            'timing_risk': False
        }
        
        if execution_data:
            # High slippage risk
            max_slippage = max(abs(e['slippage_bps']) for e in execution_data)
            risks['high_slippage_risk'] = max_slippage > 20
            
            # Latency risk
            max_latency = max(e['latency_ms'] for e in execution_data)
            risks['latency_risk'] = max_latency > 1000
            
            # Concentration risk (too many orders in short time)
            timestamps = [datetime.fromisoformat(e['timestamp']) for e in execution_data]
            duration = (max(timestamps) - min(timestamps)).total_seconds() / 60
            risks['concentration_risk'] = len(execution_data) / max(duration, 1) > 5  # >5 orders/minute
            
            # Timing risk (execution during volatile periods)
            slippage_variance = np.var([e['slippage_bps'] for e in execution_data])
            risks['timing_risk'] = slippage_variance > 25  # High variance indicates volatility
        
        return risks

async def main():
    """Test execution analytics"""
    
    analyzer = ExecutionAnalyzer()
    
    # Sample execution data
    execution_data = [
        {
            'timestamp': datetime.now().isoformat(),
            'symbol': 'BTCUSDT',
            'side': 'BUY',
            'quantity': 0.1,
            'fill_price': 50100,
            'arrival_price': 50000,
            'slippage_bps': 20,
            'latency_ms': 100,
            'venue': 'binance',
            'algorithm': 'TWAP',
            'parent_order_id': 'P001',
            'slice_id': 'S001',
            'fees': 50.1,
            'liquidity_type': 'taker'
        }
    ]
    
    target_portfolio = {
        'weights': [
            {'symbol': 'BTCUSDT', 'usd_size': 5000}
        ]
    }
    
    # Generate report
    report = await analyzer.generate_execution_report(execution_data, target_portfolio)
    
    print("Execution Analytics Report:")
    print(f"Total Cost: {report['transaction_cost_analysis']['total_cost_bps']:.1f} bps")
    print(f"Recommendations: {len(report['recommendations'])}")
    for rec in report['recommendations']:
        print(f"  - {rec}")

if __name__ == "__main__":
    asyncio.run(main())