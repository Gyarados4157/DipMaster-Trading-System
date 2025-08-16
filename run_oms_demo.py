"""
DipMaster Enhanced V4 - OMS Demo Script
Standalone execution of Order Management System with target portfolio
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import core modules without websocket dependency
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class OrderSlice:
    """Individual order slice"""
    slice_id: str
    parent_id: str
    symbol: str
    side: str
    qty: float
    order_type: str
    limit_price: Optional[float] = None
    tif: str = 'GTC'
    venue: str = 'binance'
    scheduled_time: Optional[datetime] = None
    executed_time: Optional[datetime] = None
    status: str = 'pending'
    fill_qty: float = 0.0
    fill_price: float = 0.0
    slippage_bps: float = 0.0

@dataclass
class Fill:
    """Order fill record"""
    order_id: str
    price: float
    qty: float
    slippage_bps: float
    venue: str
    timestamp: datetime
    fees: float = 0.0
    liquidity: str = 'unknown'

class SimplifiedOMS:
    """Simplified OMS for demonstration"""
    
    def __init__(self):
        self.fills = []
        self.orders = []
        
    async def execute_target_portfolio(self, target_portfolio: Dict) -> Dict:
        """Execute target portfolio with mock execution"""
        
        execution_start = datetime.now()
        session_id = f"SESSION_{int(execution_start.timestamp())}"
        
        logger.info(f"Starting portfolio execution: {session_id}")
        
        orders = []
        fills = []
        total_cost = 0.0
        
        for i, position in enumerate(target_portfolio.get('weights', [])):
            symbol = position['symbol']
            usd_size = position['usd_size']
            
            # Mock current price
            base_prices = {
                'BTCUSDT': 62430.0,
                'ETHUSDT': 3124.5,
                'SOLUSDT': 145.2
            }
            current_price = base_prices.get(symbol, 50000.0)
            quantity = usd_size / current_price
            
            # Create order slices (TWAP algorithm simulation)
            num_slices = min(5, max(1, int(usd_size / 1000)))  # 1 slice per $1000
            slice_qty = quantity / num_slices
            
            for j in range(num_slices):
                # Create order
                slice_id = f"{session_id}_S{i+1:02d}_{j+1:02d}"
                parent_id = f"{session_id}_P{i+1:02d}"
                
                # Add price randomness for realistic execution
                price_impact = np.random.uniform(0.0005, 0.0025)  # 0.5-2.5bps impact
                execution_price = current_price * (1 + price_impact)
                
                # Calculate slippage
                slippage_bps = (execution_price - current_price) / current_price * 10000
                
                # Create order slice
                order_slice = OrderSlice(
                    slice_id=slice_id,
                    parent_id=parent_id,
                    symbol=symbol,
                    side='BUY',
                    qty=slice_qty,
                    order_type='limit',
                    limit_price=execution_price,
                    tif='IOC',
                    venue='binance',
                    status='filled',
                    fill_qty=slice_qty,
                    fill_price=execution_price,
                    slippage_bps=slippage_bps
                )
                
                orders.append({
                    'venue': order_slice.venue,
                    'symbol': order_slice.symbol,
                    'side': order_slice.side,
                    'qty': order_slice.qty,
                    'tif': order_slice.tif,
                    'order_type': order_slice.order_type,
                    'limit_price': order_slice.limit_price,
                    'slice_id': order_slice.slice_id,
                    'parent_id': order_slice.parent_id,
                    'status': order_slice.status,
                    'fill_qty': order_slice.fill_qty,
                    'fill_price': order_slice.fill_price
                })
                
                # Create fill
                fees = slice_qty * execution_price * 0.001  # 0.1% fee
                total_cost += fees
                
                fill = Fill(
                    order_id=slice_id,
                    price=execution_price,
                    qty=slice_qty,
                    slippage_bps=slippage_bps,
                    venue='binance',
                    timestamp=datetime.now(),
                    fees=fees,
                    liquidity='taker'
                )
                
                fills.append({
                    'order_id': fill.order_id,
                    'price': fill.price,
                    'qty': fill.qty,
                    'slippage_bps': fill.slippage_bps,
                    'venue': fill.venue,
                    'timestamp': fill.timestamp.isoformat(),
                    'fees': fill.fees,
                    'liquidity': fill.liquidity
                })
        
        # Calculate execution metrics
        total_qty = sum(f['qty'] for f in fills)
        total_value = sum(f['qty'] * f['price'] for f in fills)
        avg_price = total_value / total_qty if total_qty > 0 else 0
        avg_slippage = sum(abs(f['slippage_bps']) for f in fills) / len(fills) if fills else 0
        
        # Calculate costs
        total_fees = sum(f['fees'] for f in fills)
        impact_bps = avg_slippage
        spread_cost = total_value * 0.0005  # 5bps spread cost
        total_execution_cost = total_fees + spread_cost
        
        # Execution quality metrics
        fill_rate = len([f for f in fills if f['qty'] > 0]) / len(orders) if orders else 0
        passive_ratio = len([f for f in fills if f['liquidity'] == 'maker']) / len(fills) if fills else 0
        execution_latency = (datetime.now() - execution_start).total_seconds() * 1000
        
        # Create comprehensive execution report
        execution_report = {
            'session_id': session_id,
            'timestamp': execution_start.isoformat(),
            'execution_start': execution_start.isoformat(),
            'execution_end': datetime.now().isoformat(),
            'total_execution_time_seconds': (datetime.now() - execution_start).total_seconds(),
            
            # Core execution data
            'orders': orders,
            'fills': fills,
            
            # Cost analysis
            'costs': {
                'fees_usd': total_fees,
                'impact_bps': impact_bps,
                'spread_cost_usd': spread_cost,
                'total_cost_usd': total_execution_cost,
                'timing_cost_bps': 0.5  # Mock timing cost
            },
            
            # Execution quality
            'execution_quality': {
                'arrival_slippage_bps': avg_slippage,
                'vwap_slippage_bps': avg_slippage * 0.8,  # Slightly better than arrival
                'fill_rate': fill_rate,
                'passive_ratio': passive_ratio,
                'latency_ms': execution_latency,
                'participation_rate': 0.05  # 5% estimated participation
            },
            
            # Risk and violations
            'violations': [],
            'pnl': {
                'realized': 0.0,
                'unrealized': 0.0
            },
            'latency_ms': execution_latency,
            
            # Portfolio and strategy info
            'target_portfolio': target_portfolio,
            'execution_strategy': 'auto',
            
            # Performance attribution
            'performance': {
                'execution_alpha_bps': -avg_slippage,  # Negative is cost
                'cost_attribution': {
                    'market_impact': impact_bps * 0.6,
                    'timing': impact_bps * 0.2,
                    'fees': (total_fees / total_value * 10000) if total_value > 0 else 0
                },
                'quality_score': fill_rate * 100,
                'vs_benchmark': {
                    'arrival_price': avg_slippage,
                    'vwap': avg_slippage * 0.8,
                    'twap': avg_slippage * 0.9
                }
            },
            
            # Analytics summary
            'analytics': {
                'summary': {
                    'total_volume_usd': total_value,
                    'total_orders': len(orders),
                    'avg_slippage_bps': avg_slippage,
                    'avg_latency_ms': execution_latency,
                    'fill_rate': fill_rate,
                    'execution_period_minutes': (datetime.now() - execution_start).total_seconds() / 60
                },
                'algorithm_comparison': {
                    'TWAP': {
                        'avg_slippage': avg_slippage,
                        'avg_latency': execution_latency,
                        'fill_rate': fill_rate,
                        'market_impact': impact_bps,
                        'execution_count': len(orders)
                    }
                },
                'recommendations': [
                    f"Execution completed with {avg_slippage:.1f}bps average slippage",
                    f"Fill rate of {fill_rate:.1%} achieved",
                    "Consider using more passive orders for large positions" if impact_bps > 10 else "Execution quality within acceptable parameters"
                ]
            },
            
            # Risk assessment
            'risk_assessment': {
                'current_metrics': {
                    'exposure_usd': total_value,
                    'exposure_utilization': total_value / 50000,  # Assume $50k limit
                    'slippage_bps': avg_slippage,
                    'rejection_rate': 0.0,
                    'latency_p95_ms': execution_latency,
                    'market_impact_bps': impact_bps
                },
                'emergency_stop': False,
                'active_violations': 0,
                'circuit_breakers': {
                    'high_slippage': {'is_triggered': False},
                    'high_rejection_rate': {'is_triggered': False},
                    'high_latency': {'is_triggered': False},
                    'market_stress': {'is_triggered': False}
                }
            }
        }
        
        logger.info(f"Portfolio execution completed: {len(orders)} orders, {len(fills)} fills")
        logger.info(f"Total volume: ${total_value:.0f}, Average slippage: {avg_slippage:.1f}bps")
        logger.info(f"Total cost: ${total_execution_cost:.2f}, Fill rate: {fill_rate:.1%}")
        
        return execution_report

async def main():
    """Main execution function"""
    
    # Load target portfolio
    portfolio_file = 'results/portfolio_construction/TargetPortfolio_20250816_183319.json'
    
    try:
        with open(portfolio_file, 'r') as f:
            target_portfolio = json.load(f)
        
        logger.info(f"Loaded target portfolio: {len(target_portfolio.get('weights', []))} positions")
        
    except FileNotFoundError:
        logger.error(f"Target portfolio file not found: {portfolio_file}")
        # Create sample portfolio
        target_portfolio = {
            'weights': [
                {'symbol': 'BTCUSDT', 'usd_size': 2500},
                {'symbol': 'ETHUSDT', 'usd_size': 2000},
                {'symbol': 'SOLUSDT', 'usd_size': 1500}
            ]
        }
        logger.info("Using sample target portfolio")
    
    # Initialize simplified OMS
    oms = SimplifiedOMS()
    
    # Execute target portfolio
    execution_report = await oms.execute_target_portfolio(target_portfolio)
    
    # Ensure output directory exists
    output_dir = 'results/execution_reports'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save execution report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/DipMaster_ExecutionReport_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(execution_report, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("DIPMASTER ENHANCED V4 - EXECUTION REPORT GENERATED")
    print("="*80)
    print(f"Report File: {filename}")
    print(f"Session ID: {execution_report['session_id']}")
    print(f"Execution Time: {execution_report['total_execution_time_seconds']:.2f} seconds")
    print("\nEXECUTION SUMMARY:")
    print(f"  Orders Placed: {len(execution_report['orders'])}")
    print(f"  Orders Filled: {len(execution_report['fills'])}")
    print(f"  Total Volume: ${execution_report['analytics']['summary']['total_volume_usd']:,.0f}")
    print(f"  Fill Rate: {execution_report['execution_quality']['fill_rate']:.1%}")
    print(f"  Average Slippage: {execution_report['execution_quality']['arrival_slippage_bps']:.1f} bps")
    print(f"  Execution Latency: {execution_report['latency_ms']:.0f} ms")
    
    print("\nCOST BREAKDOWN:")
    costs = execution_report['costs']
    print(f"  Total Cost: ${costs['total_cost_usd']:.2f}")
    print(f"  Fees: ${costs['fees_usd']:.2f}")
    print(f"  Market Impact: {costs['impact_bps']:.1f} bps")
    print(f"  Spread Cost: ${costs['spread_cost_usd']:.2f}")
    
    print("\nEXECUTION QUALITY SCORES:")
    quality = execution_report['execution_quality']
    print(f"  VWAP Slippage: {quality['vwap_slippage_bps']:.1f} bps")
    print(f"  Participation Rate: {quality['participation_rate']:.1%}")
    print(f"  Passive Ratio: {quality['passive_ratio']:.1%}")
    
    print("\nRISK ASSESSMENT:")
    risk = execution_report['risk_assessment']
    print(f"  Exposure Utilization: {risk['current_metrics']['exposure_utilization']:.1%}")
    print(f"  Active Violations: {risk['active_violations']}")
    print(f"  Emergency Stop: {risk['emergency_stop']}")
    
    print("\nRECOMMENDations:")
    for rec in execution_report['analytics']['recommendations']:
        print(f"  • {rec}")
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETED SUCCESSFULLY")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())