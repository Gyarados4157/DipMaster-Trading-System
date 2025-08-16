"""
DipMaster Enhanced V4 - Smart Execution Engine
Intelligent Order Management System with TWAP/VWAP algorithms, order slicing, and cost optimization
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException

logger = logging.getLogger(__name__)

@dataclass
class OrderSlice:
    """Individual order slice"""
    slice_id: str
    parent_id: str
    symbol: str
    side: str
    qty: float
    order_type: str  # 'limit', 'market', 'iceberg'
    limit_price: Optional[float] = None
    tif: str = 'GTC'  # Time In Force
    venue: str = 'binance'
    scheduled_time: Optional[datetime] = None
    executed_time: Optional[datetime] = None
    status: str = 'pending'  # pending, submitted, filled, cancelled
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
    liquidity: str = 'unknown'  # maker, taker

@dataclass
class ExecutionCosts:
    """Execution cost breakdown"""
    fees_usd: float
    impact_bps: float
    spread_cost_usd: float
    total_cost_usd: float
    timing_cost_bps: float = 0.0

@dataclass
class ExecutionQuality:
    """Execution quality metrics"""
    arrival_slippage_bps: float
    vwap_slippage_bps: float
    fill_rate: float
    passive_ratio: float
    latency_ms: float
    participation_rate: float = 0.0

class MarketDataProvider:
    """Real-time market data provider"""
    
    def __init__(self, client: Optional[Client] = None):
        self.client = client
        self.price_cache = {}
        self.volume_cache = {}
        self.orderbook_cache = {}
        
    async def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            if self.client:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                price = float(ticker['price'])
                self.price_cache[symbol] = price
                return price
            else:
                # Paper trading - use cached or default price
                return self.price_cache.get(symbol, 50000.0)
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return self.price_cache.get(symbol, 50000.0)
    
    async def get_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Get order book depth"""
        try:
            if self.client:
                depth = self.client.get_order_book(symbol=symbol, limit=limit)
                self.orderbook_cache[symbol] = depth
                return depth
            else:
                # Paper trading mock orderbook
                price = await self.get_current_price(symbol)
                return {
                    'bids': [[str(price * 0.999), '100.0']],
                    'asks': [[str(price * 1.001), '100.0']]
                }
        except Exception as e:
            logger.error(f"Error getting orderbook for {symbol}: {e}")
            return self.orderbook_cache.get(symbol, {})
    
    async def get_24h_volume(self, symbol: str) -> float:
        """Get 24h trading volume"""
        try:
            if self.client:
                stats = self.client.get_24hr_ticker(symbol=symbol)
                volume = float(stats['volume'])
                self.volume_cache[symbol] = volume
                return volume
            else:
                return self.volume_cache.get(symbol, 1000000.0)
        except Exception as e:
            logger.error(f"Error getting volume for {symbol}: {e}")
            return self.volume_cache.get(symbol, 1000000.0)

class OrderSlicingEngine:
    """Advanced order slicing algorithms"""
    
    def __init__(self, market_data: MarketDataProvider):
        self.market_data = market_data
        
    async def slice_twap(self, 
                        symbol: str, 
                        side: str, 
                        total_qty: float, 
                        duration_minutes: int = 30,
                        min_slice_size: float = 10.0) -> List[OrderSlice]:
        """Time-Weighted Average Price slicing"""
        
        # Calculate slice parameters
        num_slices = max(1, min(20, duration_minutes // 2))  # 2-minute intervals
        slice_qty = total_qty / num_slices
        slice_interval = duration_minutes * 60 / num_slices
        
        # Ensure minimum slice size
        if slice_qty < min_slice_size:
            num_slices = max(1, int(total_qty / min_slice_size))
            slice_qty = total_qty / num_slices
            slice_interval = duration_minutes * 60 / num_slices
        
        slices = []
        parent_id = f"TWAP_{symbol}_{int(time.time())}"
        
        for i in range(num_slices):
            # Add randomization to avoid predictable patterns
            qty_variation = np.random.uniform(0.8, 1.2)
            actual_qty = slice_qty * qty_variation
            
            # Last slice gets remaining quantity
            if i == num_slices - 1:
                actual_qty = total_qty - sum(s.qty for s in slices)
            
            scheduled_time = datetime.now() + timedelta(seconds=i * slice_interval)
            
            slice_obj = OrderSlice(
                slice_id=f"{parent_id}_S{i+1:02d}",
                parent_id=parent_id,
                symbol=symbol,
                side=side,
                qty=actual_qty,
                order_type='limit',
                tif='IOC',  # Immediate or Cancel for aggressive execution
                scheduled_time=scheduled_time
            )
            slices.append(slice_obj)
        
        logger.info(f"TWAP slicing: {total_qty} {symbol} -> {num_slices} slices over {duration_minutes}min")
        return slices
    
    async def slice_vwap(self, 
                        symbol: str, 
                        side: str, 
                        total_qty: float,
                        lookback_hours: int = 24) -> List[OrderSlice]:
        """Volume-Weighted Average Price slicing"""
        
        # Get historical volume profile (simplified)
        volume_24h = await self.market_data.get_24h_volume(symbol)
        
        # Create volume-weighted time buckets
        # Assume higher volume during certain hours (simplified model)
        hour_weights = [
            0.8, 0.6, 0.4, 0.3, 0.3, 0.4,  # 0-5 (low activity)
            0.6, 0.8, 1.0, 1.2, 1.5, 1.8,  # 6-11 (increasing)
            2.0, 2.2, 2.0, 1.8, 1.5, 1.2,  # 12-17 (peak)
            1.0, 0.8, 0.6, 0.5, 0.4, 0.3   # 18-23 (declining)
        ]
        
        current_hour = datetime.now().hour
        total_weight = sum(hour_weights)
        
        slices = []
        parent_id = f"VWAP_{symbol}_{int(time.time())}"
        
        # Create slices for next 4 hours with volume weighting
        for i in range(8):  # 30-minute intervals
            hour_idx = (current_hour + i // 2) % 24
            weight = hour_weights[hour_idx]
            
            slice_qty = total_qty * (weight / total_weight) * 8  # Scale for 8 slices
            scheduled_time = datetime.now() + timedelta(minutes=i * 30)
            
            slice_obj = OrderSlice(
                slice_id=f"{parent_id}_V{i+1:02d}",
                parent_id=parent_id,
                symbol=symbol,
                side=side,
                qty=slice_qty,
                order_type='limit',
                tif='GTC',
                scheduled_time=scheduled_time
            )
            slices.append(slice_obj)
        
        # Normalize quantities to match total
        actual_total = sum(s.qty for s in slices)
        for slice_obj in slices:
            slice_obj.qty = slice_obj.qty * total_qty / actual_total
        
        logger.info(f"VWAP slicing: {total_qty} {symbol} -> {len(slices)} volume-weighted slices")
        return slices
    
    async def slice_implementation_shortfall(self,
                                           symbol: str,
                                           side: str,
                                           total_qty: float,
                                           urgency: float = 0.5) -> List[OrderSlice]:
        """Implementation Shortfall optimization"""
        
        # Get market data
        current_price = await self.market_data.get_current_price(symbol)
        volume_24h = await self.market_data.get_24h_volume(symbol)
        
        # Estimate market impact parameters
        market_impact_rate = min(0.1, total_qty * current_price / (volume_24h * current_price * 0.1))
        volatility = 0.02  # Simplified 2% daily volatility
        
        # Calculate optimal participation rate
        participation_rate = min(0.2, urgency * 0.3)  # Max 20% of volume
        
        # Time horizon based on urgency
        total_minutes = max(15, int(120 * (1 - urgency)))
        num_slices = max(3, min(15, total_minutes // 8))
        
        slices = []
        parent_id = f"IS_{symbol}_{int(time.time())}"
        
        # Front-load execution based on urgency
        for i in range(num_slices):
            # Exponential decay for low urgency, linear for high urgency
            if urgency > 0.7:
                weight = 1.0  # Even distribution for urgent orders
            else:
                weight = np.exp(-i * 0.3)  # Front-loaded for patient orders
            
            slice_qty = total_qty * weight / sum(np.exp(-j * 0.3) for j in range(num_slices))
            scheduled_time = datetime.now() + timedelta(minutes=i * total_minutes / num_slices)
            
            # Use aggressive orders for urgent execution
            order_type = 'market' if urgency > 0.8 else 'limit'
            tif = 'IOC' if urgency > 0.6 else 'GTC'
            
            slice_obj = OrderSlice(
                slice_id=f"{parent_id}_IS{i+1:02d}",
                parent_id=parent_id,
                symbol=symbol,
                side=side,
                qty=slice_qty,
                order_type=order_type,
                tif=tif,
                scheduled_time=scheduled_time
            )
            slices.append(slice_obj)
        
        logger.info(f"Implementation Shortfall: {total_qty} {symbol} -> {num_slices} slices, urgency={urgency:.1f}")
        return slices

class SmartOrderRouter:
    """Intelligent order routing with best execution logic"""
    
    def __init__(self, market_data: MarketDataProvider):
        self.market_data = market_data
        self.venue_configs = {
            'binance': {
                'maker_fee': 0.001,
                'taker_fee': 0.001,
                'latency_ms': 50,
                'liquidity_score': 0.9
            }
        }
    
    async def route_order(self, slice_obj: OrderSlice) -> str:
        """Determine best venue for order execution"""
        
        # For now, single venue (Binance)
        # In production, would compare multiple venues
        best_venue = 'binance'
        
        # Adjust order type based on market conditions
        await self._optimize_order_parameters(slice_obj)
        
        return best_venue
    
    async def _optimize_order_parameters(self, slice_obj: OrderSlice):
        """Optimize order parameters based on market conditions"""
        
        # Get current market data
        orderbook = await self.market_data.get_orderbook(slice_obj.symbol)
        
        if not orderbook or 'bids' not in orderbook or 'asks' not in orderbook:
            return
        
        # Calculate spread
        best_bid = float(orderbook['bids'][0][0]) if orderbook['bids'] else 0
        best_ask = float(orderbook['asks'][0][0]) if orderbook['asks'] else 0
        
        if best_bid > 0 and best_ask > 0:
            spread_bps = ((best_ask - best_bid) / best_bid) * 10000
            
            # Adjust strategy based on spread width
            if spread_bps > 50:  # Wide spread - use patient approach
                slice_obj.order_type = 'limit'
                slice_obj.tif = 'GTC'
                # Price aggressively within spread
                if slice_obj.side == 'BUY':
                    slice_obj.limit_price = best_bid + (best_ask - best_bid) * 0.3
                else:
                    slice_obj.limit_price = best_ask - (best_ask - best_bid) * 0.3
            else:  # Tight spread - can be more aggressive
                slice_obj.order_type = 'limit'
                slice_obj.tif = 'IOC'
                if slice_obj.side == 'BUY':
                    slice_obj.limit_price = best_ask  # Cross spread for IOC
                else:
                    slice_obj.limit_price = best_bid

class ExecutionCostAnalyzer:
    """Analyze and minimize execution costs"""
    
    def __init__(self, market_data: MarketDataProvider):
        self.market_data = market_data
        
    async def calculate_slippage(self, 
                               symbol: str, 
                               fill_price: float, 
                               arrival_price: float) -> float:
        """Calculate slippage in basis points"""
        if arrival_price <= 0:
            return 0.0
        
        slippage_bps = ((fill_price - arrival_price) / arrival_price) * 10000
        return slippage_bps
    
    async def estimate_market_impact(self, 
                                   symbol: str, 
                                   qty: float, 
                                   side: str) -> float:
        """Estimate market impact in basis points"""
        
        # Get order book
        orderbook = await self.market_data.get_orderbook(symbol, limit=20)
        if not orderbook:
            return 5.0  # Default 5bps impact
        
        # Calculate impact based on order book depth
        levels = orderbook['asks'] if side == 'BUY' else orderbook['bids']
        
        cumulative_qty = 0.0
        total_cost = 0.0
        start_price = float(levels[0][0]) if levels else 0
        
        for price_str, qty_str in levels:
            price = float(price_str)
            available_qty = float(qty_str)
            
            take_qty = min(qty - cumulative_qty, available_qty)
            total_cost += take_qty * price
            cumulative_qty += take_qty
            
            if cumulative_qty >= qty:
                break
        
        if cumulative_qty > 0 and start_price > 0:
            avg_price = total_cost / cumulative_qty
            impact_bps = abs((avg_price - start_price) / start_price) * 10000
            return impact_bps
        
        return 5.0  # Default impact
    
    async def calculate_execution_costs(self, 
                                      fills: List[Fill], 
                                      arrival_price: float) -> ExecutionCosts:
        """Calculate comprehensive execution costs"""
        
        if not fills:
            return ExecutionCosts(0, 0, 0, 0)
        
        # Calculate weighted average fill price
        total_qty = sum(f.qty for f in fills)
        total_value = sum(f.qty * f.price for f in fills)
        avg_fill_price = total_value / total_qty if total_qty > 0 else 0
        
        # Fees
        total_fees = sum(f.fees for f in fills)
        
        # Market impact (slippage from arrival price)
        impact_bps = abs((avg_fill_price - arrival_price) / arrival_price) * 10000 if arrival_price > 0 else 0
        
        # Spread cost (estimated)
        spread_cost = total_qty * arrival_price * 0.0005  # 5bps spread cost
        
        # Total cost
        total_cost = total_fees + spread_cost
        
        return ExecutionCosts(
            fees_usd=total_fees,
            impact_bps=impact_bps,
            spread_cost_usd=spread_cost,
            total_cost_usd=total_cost
        )

class SmartExecutionEngine:
    """Main smart execution engine orchestrating all components"""
    
    def __init__(self, 
                 client: Optional[Client] = None,
                 paper_trading: bool = True):
        self.client = client
        self.paper_trading = paper_trading
        
        # Initialize components
        self.market_data = MarketDataProvider(client)
        self.slicer = OrderSlicingEngine(self.market_data)
        self.router = SmartOrderRouter(self.market_data)
        self.cost_analyzer = ExecutionCostAnalyzer(self.market_data)
        
        # Execution state
        self.active_orders: Dict[str, OrderSlice] = {}
        self.completed_fills: List[Fill] = []
        self.execution_start_time = None
        
        # Risk controls
        self.max_order_rate = 10  # orders per minute
        self.max_position_usd = 10000
        self.max_slippage_bps = 100  # 1%
        
    async def execute_portfolio_orders(self, target_portfolio: Dict) -> Dict:
        """Execute orders for target portfolio with smart algorithms"""
        
        self.execution_start_time = datetime.now()
        execution_report = {
            'orders': [],
            'fills': [],
            'costs': {},
            'violations': [],
            'pnl': {'realized': 0.0, 'unrealized': 0.0},
            'latency_ms': 0,
            'ts': self.execution_start_time.isoformat(),
            'execution_quality': {}
        }
        
        try:
            # Process each position in the target portfolio
            for position in target_portfolio.get('weights', []):
                symbol = position['symbol']
                target_usd = position['usd_size']
                
                logger.info(f"Processing {symbol}: ${target_usd}")
                
                # Calculate quantity needed
                current_price = await self.market_data.get_current_price(symbol)
                target_qty = target_usd / current_price
                
                # Choose execution algorithm based on order size
                if target_usd > 5000:
                    # Large order - use VWAP
                    slices = await self.slicer.slice_vwap(symbol, 'BUY', target_qty)
                elif target_usd > 2000:
                    # Medium order - use TWAP
                    slices = await self.slicer.slice_twap(symbol, 'BUY', target_qty, duration_minutes=15)
                else:
                    # Small order - use Implementation Shortfall
                    slices = await self.slicer.slice_implementation_shortfall(
                        symbol, 'BUY', target_qty, urgency=0.7)
                
                # Execute slices
                symbol_fills = await self._execute_slices(slices, current_price)
                
                # Add to execution report
                for slice_obj in slices:
                    execution_report['orders'].append({
                        'venue': slice_obj.venue,
                        'symbol': slice_obj.symbol,
                        'side': slice_obj.side,
                        'qty': slice_obj.qty,
                        'tif': slice_obj.tif,
                        'order_type': slice_obj.order_type,
                        'limit_price': slice_obj.limit_price,
                        'slice_id': slice_obj.slice_id,
                        'parent_id': slice_obj.parent_id,
                        'status': slice_obj.status,
                        'fill_qty': slice_obj.fill_qty,
                        'fill_price': slice_obj.fill_price
                    })
                
                execution_report['fills'].extend([asdict(f) for f in symbol_fills])
                self.completed_fills.extend(symbol_fills)
            
            # Calculate execution costs and quality metrics
            await self._calculate_execution_metrics(execution_report)
            
            # Save execution report
            await self._save_execution_report(execution_report)
            
            logger.info(f"Portfolio execution completed: {len(execution_report['orders'])} orders, "
                       f"{len(execution_report['fills'])} fills")
            
            return execution_report
            
        except Exception as e:
            logger.error(f"Portfolio execution failed: {e}")
            execution_report['violations'].append({
                'type': 'EXECUTION_ERROR',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return execution_report
    
    async def _execute_slices(self, slices: List[OrderSlice], arrival_price: float) -> List[Fill]:
        """Execute order slices with timing and risk controls"""
        
        fills = []
        
        for slice_obj in slices:
            try:
                # Route order to best venue
                best_venue = await self.router.route_order(slice_obj)
                slice_obj.venue = best_venue
                
                # Wait for scheduled time
                if slice_obj.scheduled_time:
                    wait_time = (slice_obj.scheduled_time - datetime.now()).total_seconds()
                    if wait_time > 0:
                        await asyncio.sleep(min(wait_time, 60))  # Max 1 minute wait
                
                # Execute order
                fill = await self._execute_single_order(slice_obj, arrival_price)
                if fill:
                    fills.append(fill)
                    
                # Check risk limits
                violations = await self._check_execution_risks(slice_obj, fill)
                if violations:
                    logger.warning(f"Risk violations detected: {violations}")
                    break  # Stop execution on risk violation
                    
                # Rate limiting
                await asyncio.sleep(0.1)  # Minimum delay between orders
                
            except Exception as e:
                logger.error(f"Failed to execute slice {slice_obj.slice_id}: {e}")
                slice_obj.status = 'failed'
        
        return fills
    
    async def _execute_single_order(self, slice_obj: OrderSlice, arrival_price: float) -> Optional[Fill]:
        """Execute a single order slice"""
        
        slice_obj.status = 'submitted'
        execution_start = time.time()
        
        try:
            if self.paper_trading or not self.client:
                # Paper trading simulation
                fill_price = slice_obj.limit_price if slice_obj.limit_price else arrival_price
                
                # Add realistic slippage
                if slice_obj.order_type == 'market':
                    slippage_factor = np.random.uniform(0.0005, 0.002)  # 0.5-2bps slippage
                    fill_price = arrival_price * (1 + slippage_factor if slice_obj.side == 'BUY' else 1 - slippage_factor)
                
                # Simulate partial fills for large orders
                fill_qty = slice_obj.qty
                if slice_obj.qty * fill_price > 2000:  # Large order
                    fill_qty = slice_obj.qty * np.random.uniform(0.8, 1.0)
                
                slice_obj.fill_qty = fill_qty
                slice_obj.fill_price = fill_price
                slice_obj.status = 'filled'
                
                # Calculate slippage
                slippage_bps = await self.cost_analyzer.calculate_slippage(
                    slice_obj.symbol, fill_price, arrival_price)
                slice_obj.slippage_bps = slippage_bps
                
            else:
                # Real trading execution
                if slice_obj.order_type == 'market':
                    order_result = await self._place_market_order(slice_obj)
                else:
                    order_result = await self._place_limit_order(slice_obj)
                
                if not order_result:
                    slice_obj.status = 'failed'
                    return None
                
                slice_obj.fill_qty = order_result.get('executedQty', 0)
                slice_obj.fill_price = order_result.get('price', arrival_price)
                slice_obj.status = order_result.get('status', 'failed')
            
            # Create fill record
            latency_ms = (time.time() - execution_start) * 1000
            
            fill = Fill(
                order_id=slice_obj.slice_id,
                price=slice_obj.fill_price,
                qty=slice_obj.fill_qty,
                slippage_bps=slice_obj.slippage_bps,
                venue=slice_obj.venue,
                timestamp=datetime.now(),
                fees=slice_obj.fill_qty * slice_obj.fill_price * 0.001,  # 0.1% fee
                liquidity='taker' if slice_obj.order_type == 'market' else 'maker'
            )
            
            slice_obj.executed_time = datetime.now()
            logger.info(f"Order executed: {slice_obj.slice_id} - {slice_obj.fill_qty:.6f} @ {slice_obj.fill_price:.2f}")
            
            return fill
            
        except Exception as e:
            logger.error(f"Order execution failed for {slice_obj.slice_id}: {e}")
            slice_obj.status = 'failed'
            return None
    
    async def _place_market_order(self, slice_obj: OrderSlice) -> Optional[Dict]:
        """Place market order via Binance API"""
        try:
            if slice_obj.side == 'BUY':
                result = self.client.order_market_buy(
                    symbol=slice_obj.symbol,
                    quantity=slice_obj.qty
                )
            else:
                result = self.client.order_market_sell(
                    symbol=slice_obj.symbol,
                    quantity=slice_obj.qty
                )
            return result
        except Exception as e:
            logger.error(f"Market order failed: {e}")
            return None
    
    async def _place_limit_order(self, slice_obj: OrderSlice) -> Optional[Dict]:
        """Place limit order via Binance API"""
        try:
            if slice_obj.side == 'BUY':
                result = self.client.order_limit_buy(
                    symbol=slice_obj.symbol,
                    quantity=slice_obj.qty,
                    price=slice_obj.limit_price,
                    timeInForce=slice_obj.tif
                )
            else:
                result = self.client.order_limit_sell(
                    symbol=slice_obj.symbol,
                    quantity=slice_obj.qty,
                    price=slice_obj.limit_price,
                    timeInForce=slice_obj.tif
                )
            return result
        except Exception as e:
            logger.error(f"Limit order failed: {e}")
            return None
    
    async def _check_execution_risks(self, slice_obj: OrderSlice, fill: Optional[Fill]) -> List[str]:
        """Check execution risks and limits"""
        violations = []
        
        # Slippage check
        if fill and abs(fill.slippage_bps) > self.max_slippage_bps:
            violations.append(f"Excessive slippage: {fill.slippage_bps:.1f}bps > {self.max_slippage_bps}bps")
        
        # Position size check
        if fill:
            position_value = fill.qty * fill.price
            if position_value > self.max_position_usd:
                violations.append(f"Position size exceeded: ${position_value:.0f} > ${self.max_position_usd}")
        
        return violations
    
    async def _calculate_execution_metrics(self, execution_report: Dict):
        """Calculate execution quality metrics"""
        
        if not self.completed_fills:
            return
        
        # Calculate aggregate metrics
        total_qty = sum(f.qty for f in self.completed_fills)
        total_value = sum(f.qty * f.price for f in self.completed_fills)
        avg_price = total_value / total_qty if total_qty > 0 else 0
        
        # Execution costs
        arrival_price = self.completed_fills[0].price if self.completed_fills else 0
        costs = await self.cost_analyzer.calculate_execution_costs(self.completed_fills, arrival_price)
        execution_report['costs'] = asdict(costs)
        
        # Execution quality
        fill_rate = len([f for f in self.completed_fills if f.qty > 0]) / len(execution_report['orders']) if execution_report['orders'] else 0
        passive_ratio = len([f for f in self.completed_fills if f.liquidity == 'maker']) / len(self.completed_fills) if self.completed_fills else 0
        
        execution_latency = (datetime.now() - self.execution_start_time).total_seconds() * 1000 if self.execution_start_time else 0
        
        execution_report['execution_quality'] = {
            'arrival_slippage_bps': costs.impact_bps,
            'vwap_slippage_bps': np.mean([f.slippage_bps for f in self.completed_fills]) if self.completed_fills else 0,
            'fill_rate': fill_rate,
            'passive_ratio': passive_ratio,
            'latency_ms': execution_latency,
            'participation_rate': 0.05  # Estimated 5% participation rate
        }
        
        execution_report['latency_ms'] = execution_latency
    
    async def _save_execution_report(self, execution_report: Dict):
        """Save execution report to file"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"G:\\Github\\Quant\\DipMaster-Trading-System\\results\\execution_reports\\ExecutionReport_{timestamp}.json"
        
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Convert datetime objects to strings for JSON serialization
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        with open(filename, 'w') as f:
            json.dump(execution_report, f, indent=2, default=convert_datetime)
        
        logger.info(f"Execution report saved: {filename}")

async def main():
    """Test the smart execution engine"""
    
    # Sample target portfolio
    target_portfolio = {
        'weights': [
            {'symbol': 'BTCUSDT', 'usd_size': 2500},
            {'symbol': 'ETHUSDT', 'usd_size': 2000},
            {'symbol': 'SOLUSDT', 'usd_size': 1500}
        ]
    }
    
    # Initialize engine in paper trading mode
    engine = SmartExecutionEngine(paper_trading=True)
    
    # Execute portfolio
    execution_report = await engine.execute_portfolio_orders(target_portfolio)
    
    print("Execution Report Summary:")
    print(f"Orders: {len(execution_report['orders'])}")
    print(f"Fills: {len(execution_report['fills'])}")
    print(f"Total Cost: ${execution_report['costs'].get('total_cost_usd', 0):.2f}")
    print(f"Fill Rate: {execution_report['execution_quality'].get('fill_rate', 0):.1%}")
    print(f"Execution Time: {execution_report['latency_ms']:.0f}ms")

if __name__ == "__main__":
    asyncio.run(main())