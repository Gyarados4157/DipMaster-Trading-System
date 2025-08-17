"""
Smart Order Router (SOR)
多交易所智能路由系统 - 实现最优价格发现和流动性聚合
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
import aiohttp
import json

logger = logging.getLogger(__name__)


class VenueType(Enum):
    """交易所类型"""
    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"
    HUOBI = "huobi"
    KRAKEN = "kraken"


@dataclass
class VenueConfig:
    """交易所配置"""
    name: str
    api_url: str
    ws_url: str
    maker_fee: float
    taker_fee: float
    min_order_size: float
    max_order_size: float
    latency_ms: float = 50.0
    reliability_score: float = 0.95
    enabled: bool = True


@dataclass
class OrderBookLevel:
    """订单簿层级"""
    price: float
    quantity: float
    venue: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketData:
    """市场数据"""
    symbol: str
    venue: str
    bid_price: float
    ask_price: float
    bid_quantity: float
    ask_quantity: float
    last_price: float
    volume_24h: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RouteResult:
    """路由结果"""
    venues: List[str]
    quantities: List[float]
    expected_prices: List[float]
    expected_fees: List[float]
    total_cost: float
    estimated_slippage: float
    confidence_score: float


class SmartOrderRouter:
    """智能订单路由器"""
    
    def __init__(self):
        self.venues: Dict[str, VenueConfig] = {}
        self.market_data: Dict[str, Dict[str, MarketData]] = {}  # {symbol: {venue: data}}
        self.order_books: Dict[str, Dict[str, List[OrderBookLevel]]] = {}  # {symbol: {venue: levels}}
        self.venue_status: Dict[str, bool] = {}
        self.latency_tracker: Dict[str, List[float]] = {}
        
        # 初始化交易所配置
        self._initialize_venues()
        
        # 启动市场数据收集
        self._market_data_task = None
        
    def _initialize_venues(self):
        """初始化交易所配置"""
        self.venues = {
            VenueType.BINANCE.value: VenueConfig(
                name="Binance",
                api_url="https://api.binance.com",
                ws_url="wss://stream.binance.com:9443",
                maker_fee=0.001,  # 0.1%
                taker_fee=0.001,  # 0.1%
                min_order_size=10.0,
                max_order_size=1000000.0,
                latency_ms=45.0,
                reliability_score=0.98
            ),
            VenueType.OKX.value: VenueConfig(
                name="OKX",
                api_url="https://www.okx.com",
                ws_url="wss://ws.okx.com:8443",
                maker_fee=0.0008,  # 0.08%
                taker_fee=0.001,   # 0.1%
                min_order_size=5.0,
                max_order_size=500000.0,
                latency_ms=55.0,
                reliability_score=0.95
            ),
            VenueType.BYBIT.value: VenueConfig(
                name="Bybit",
                api_url="https://api.bybit.com",
                ws_url="wss://stream.bybit.com",
                maker_fee=0.0001,  # 0.01%
                taker_fee=0.0006,  # 0.06%
                min_order_size=10.0,
                max_order_size=200000.0,
                latency_ms=60.0,
                reliability_score=0.93
            )
        }
        
        # 初始化状态跟踪
        for venue in self.venues:
            self.venue_status[venue] = True
            self.latency_tracker[venue] = []
    
    async def start_market_data_collection(self, symbols: List[str]):
        """启动市场数据收集"""
        logger.info(f"开始收集市场数据: {symbols}")
        
        if self._market_data_task:
            self._market_data_task.cancel()
        
        self._market_data_task = asyncio.create_task(
            self._collect_market_data(symbols)
        )
    
    async def stop_market_data_collection(self):
        """停止市场数据收集"""
        if self._market_data_task:
            self._market_data_task.cancel()
            self._market_data_task = None
    
    async def find_best_route(
        self,
        symbol: str,
        side: str,
        quantity: float,
        max_venues: int = 3,
        require_full_fill: bool = False
    ) -> RouteResult:
        """
        寻找最优执行路由
        
        Args:
            symbol: 交易对
            side: 买卖方向 ('BUY' or 'SELL')
            quantity: 数量
            max_venues: 最大使用交易所数量
            require_full_fill: 是否要求完全成交
        """
        logger.info(f"寻找最优路由: {side} {quantity} {symbol}")
        
        # 获取所有可用交易所的报价
        venue_quotes = await self._get_venue_quotes(symbol, side, quantity)
        
        if not venue_quotes:
            raise ValueError(f"无法获取 {symbol} 的市场报价")
        
        # 过滤可用交易所
        available_quotes = [
            quote for quote in venue_quotes
            if self.venue_status.get(quote['venue'], False) and
               quote['available_quantity'] >= self.venues[quote['venue']].min_order_size / quote['price']
        ]
        
        if not available_quotes:
            raise ValueError(f"没有可用的交易所执行 {symbol} 订单")
        
        # 计算最优路由
        if len(available_quotes) == 1:
            # 单交易所执行
            route = self._single_venue_route(available_quotes[0], quantity)
        else:
            # 多交易所执行
            route = await self._multi_venue_route(available_quotes, quantity, max_venues, require_full_fill)
        
        logger.info(f"最优路由: {len(route.venues)} 个交易所, 总成本: {route.total_cost:.4f}")
        return route
    
    async def _get_venue_quotes(self, symbol: str, side: str, quantity: float) -> List[Dict]:
        """获取各交易所报价"""
        quotes = []
        
        for venue_name, venue_config in self.venues.items():
            if not venue_config.enabled:
                continue
                
            try:
                # 获取市场数据
                market_data = self.market_data.get(symbol, {}).get(venue_name)
                if not market_data:
                    continue
                
                # 获取订单簿数据
                order_book = self.order_books.get(symbol, {}).get(venue_name, [])
                
                # 计算可用流动性和平均价格
                if side == 'BUY':
                    price = market_data.ask_price
                    available_quantity = market_data.ask_quantity
                    # 如果有订单簿数据，计算更精确的流动性
                    if order_book:
                        ask_levels = [level for level in order_book if level.price >= price]
                        if ask_levels:
                            available_quantity = sum(level.quantity for level in ask_levels[:10])  # 前10档
                            weighted_price = sum(level.price * level.quantity for level in ask_levels[:10])
                            price = weighted_price / available_quantity if available_quantity > 0 else price
                else:
                    price = market_data.bid_price
                    available_quantity = market_data.bid_quantity
                    if order_book:
                        bid_levels = [level for level in order_book if level.price <= price]
                        if bid_levels:
                            available_quantity = sum(level.quantity for level in bid_levels[:10])
                            weighted_price = sum(level.price * level.quantity for level in bid_levels[:10])
                            price = weighted_price / available_quantity if available_quantity > 0 else price
                
                # 计算费用
                fee_rate = venue_config.taker_fee  # 假设是taker订单
                
                # 计算市场冲击
                market_impact = self._estimate_market_impact(quantity, available_quantity, market_data.volume_24h)
                
                quotes.append({
                    'venue': venue_name,
                    'price': price,
                    'available_quantity': available_quantity,
                    'fee_rate': fee_rate,
                    'market_impact': market_impact,
                    'latency': venue_config.latency_ms,
                    'reliability': venue_config.reliability_score,
                    'spread': (market_data.ask_price - market_data.bid_price) / market_data.last_price
                })
                
            except Exception as e:
                logger.warning(f"获取 {venue_name} 报价失败: {e}")
                continue
        
        # 按价格排序
        if side == 'BUY':
            quotes.sort(key=lambda x: x['price'])  # 买入按价格从低到高
        else:
            quotes.sort(key=lambda x: x['price'], reverse=True)  # 卖出按价格从高到低
            
        return quotes
    
    def _single_venue_route(self, quote: Dict, quantity: float) -> RouteResult:
        """单交易所路由"""
        venue = quote['venue']
        price = quote['price']
        fee_rate = quote['fee_rate']
        
        # 计算执行成本
        notional = quantity * price
        fee = notional * fee_rate
        market_impact_cost = notional * quote['market_impact']
        total_cost = fee + market_impact_cost
        
        # 计算滑点
        slippage = quote['market_impact'] + quote['spread'] / 2
        
        return RouteResult(
            venues=[venue],
            quantities=[quantity],
            expected_prices=[price],
            expected_fees=[fee],
            total_cost=total_cost,
            estimated_slippage=slippage,
            confidence_score=quote['reliability']
        )
    
    async def _multi_venue_route(
        self, 
        quotes: List[Dict], 
        total_quantity: float, 
        max_venues: int,
        require_full_fill: bool
    ) -> RouteResult:
        """多交易所最优路由算法"""
        
        # 使用贪心算法分配订单
        venues = []
        quantities = []
        prices = []
        fees = []
        remaining_quantity = total_quantity
        
        # 按成本效率排序（价格 + 费用 + 市场冲击）
        quotes_with_cost = []
        for quote in quotes:
            total_cost_per_unit = quote['price'] * (1 + quote['fee_rate'] + quote['market_impact'])
            quotes_with_cost.append((quote, total_cost_per_unit))
        
        quotes_with_cost.sort(key=lambda x: x[1])  # 按单位成本排序
        
        for quote, _ in quotes_with_cost[:max_venues]:
            if remaining_quantity <= 0:
                break
                
            venue = quote['venue']
            available_qty = min(quote['available_quantity'], remaining_quantity)
            
            # 检查最小订单大小
            min_order_value = self.venues[venue].min_order_size
            min_qty = min_order_value / quote['price']
            
            if available_qty < min_qty:
                continue
                
            # 分配数量（考虑流动性限制）
            allocated_qty = min(available_qty, remaining_quantity * 0.7)  # 不超过70%避免过度冲击
            
            if allocated_qty >= min_qty:
                venues.append(venue)
                quantities.append(allocated_qty)
                prices.append(quote['price'])
                
                # 计算费用
                notional = allocated_qty * quote['price']
                fee = notional * quote['fee_rate']
                fees.append(fee)
                
                remaining_quantity -= allocated_qty
        
        # 检查是否满足完全成交要求
        if require_full_fill and remaining_quantity > 0.01:  # 允许1%的误差
            raise ValueError(f"无法完全成交，剩余数量: {remaining_quantity}")
        
        # 如果有剩余数量，分配给最后一个交易所
        if remaining_quantity > 0 and venues:
            quantities[-1] += remaining_quantity
            additional_notional = remaining_quantity * prices[-1]
            additional_fee = additional_notional * quotes_with_cost[len(venues)-1][0]['fee_rate']
            fees[-1] += additional_fee
        
        # 计算总成本和滑点
        total_cost = sum(fees)
        total_notional = sum(q * p for q, p in zip(quantities, prices))
        
        # 加权平均滑点
        weighted_slippage = 0
        total_weight = sum(quantities)
        for i, qty in enumerate(quantities):
            weight = qty / total_weight
            quote = next(q[0] for q in quotes_with_cost if q[0]['venue'] == venues[i])
            weighted_slippage += weight * (quote['market_impact'] + quote['spread'] / 2)
        
        # 计算置信度
        reliability_scores = []
        for venue in venues:
            quote = next(q[0] for q in quotes_with_cost if q[0]['venue'] == venue)
            reliability_scores.append(quote['reliability'])
        
        confidence_score = np.mean(reliability_scores) if reliability_scores else 0.5
        
        return RouteResult(
            venues=venues,
            quantities=quantities,
            expected_prices=prices,
            expected_fees=fees,
            total_cost=total_cost,
            estimated_slippage=weighted_slippage,
            confidence_score=confidence_score
        )
    
    def _estimate_market_impact(self, order_quantity: float, available_liquidity: float, daily_volume: float) -> float:
        """估算市场冲击成本"""
        if available_liquidity <= 0 or daily_volume <= 0:
            return 0.005  # 默认0.5%
        
        # 使用Square-root law估算市场冲击
        participation_rate = order_quantity / (daily_volume / 24)  # 假设平均分布
        
        # 市场冲击 = a * sqrt(participation_rate)
        # 其中a是市场冲击系数，通常在0.1-1.0之间
        impact_coefficient = 0.5
        market_impact = impact_coefficient * np.sqrt(participation_rate)
        
        # 流动性调整
        liquidity_ratio = order_quantity / available_liquidity
        if liquidity_ratio > 0.1:  # 如果订单超过可用流动性的10%
            market_impact *= (1 + liquidity_ratio)
        
        return min(market_impact, 0.02)  # 最大2%
    
    async def _collect_market_data(self, symbols: List[str]):
        """收集市场数据"""
        while True:
            try:
                tasks = []
                for symbol in symbols:
                    for venue in self.venues:
                        if self.venues[venue].enabled:
                            task = self._fetch_market_data(venue, symbol)
                            tasks.append(task)
                
                # 并发获取所有数据
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # 处理结果
                for result in results:
                    if isinstance(result, Exception):
                        logger.warning(f"获取市场数据失败: {result}")
                    elif result:
                        symbol, venue, data = result
                        if symbol not in self.market_data:
                            self.market_data[symbol] = {}
                        self.market_data[symbol][venue] = data
                
                # 更新交易所状态
                await self._update_venue_status()
                
                # 等待下次更新
                await asyncio.sleep(1)  # 每秒更新一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"市场数据收集错误: {e}")
                await asyncio.sleep(5)
    
    async def _fetch_market_data(self, venue: str, symbol: str) -> Optional[Tuple[str, str, MarketData]]:
        """从特定交易所获取市场数据"""
        try:
            start_time = datetime.now()
            
            # 模拟API调用（实际实现需要调用真实API）
            await asyncio.sleep(0.01)  # 模拟网络延迟
            
            # 计算延迟
            latency = (datetime.now() - start_time).total_seconds() * 1000
            self.latency_tracker[venue].append(latency)
            if len(self.latency_tracker[venue]) > 100:
                self.latency_tracker[venue] = self.latency_tracker[venue][-100:]
            
            # 模拟市场数据（实际实现需要解析真实数据）
            base_price = 50000  # BTC基础价格
            spread = base_price * 0.0002  # 0.02% spread
            
            market_data = MarketData(
                symbol=symbol,
                venue=venue,
                bid_price=base_price - spread/2,
                ask_price=base_price + spread/2,
                bid_quantity=10.0,
                ask_quantity=10.0,
                last_price=base_price,
                volume_24h=1000000.0,
                timestamp=datetime.now()
            )
            
            return symbol, venue, market_data
            
        except Exception as e:
            logger.warning(f"获取 {venue} {symbol} 市场数据失败: {e}")
            return None
    
    async def _update_venue_status(self):
        """更新交易所状态"""
        for venue in self.venues:
            # 基于延迟和错误率更新状态
            latencies = self.latency_tracker.get(venue, [])
            
            if not latencies:
                continue
                
            avg_latency = np.mean(latencies[-10:])  # 最近10次的平均延迟
            
            # 如果延迟过高，标记为不可用
            if avg_latency > 1000:  # 1秒
                self.venue_status[venue] = False
                logger.warning(f"{venue} 延迟过高: {avg_latency:.1f}ms")
            else:
                self.venue_status[venue] = True
    
    async def get_market_depth(self, symbol: str, venue: str, levels: int = 10) -> List[OrderBookLevel]:
        """获取市场深度"""
        order_book = self.order_books.get(symbol, {}).get(venue, [])
        return order_book[:levels]
    
    def get_venue_performance(self, venue: str) -> Dict:
        """获取交易所性能指标"""
        latencies = self.latency_tracker.get(venue, [])
        
        if not latencies:
            return {"status": "unknown"}
        
        return {
            "status": "online" if self.venue_status.get(venue, False) else "offline",
            "avg_latency_ms": np.mean(latencies),
            "max_latency_ms": np.max(latencies),
            "min_latency_ms": np.min(latencies),
            "reliability_score": self.venues[venue].reliability_score,
            "last_update": datetime.now()
        }
    
    def get_routing_stats(self) -> Dict:
        """获取路由统计信息"""
        total_venues = len(self.venues)
        online_venues = sum(1 for status in self.venue_status.values() if status)
        
        return {
            "total_venues": total_venues,
            "online_venues": online_venues,
            "availability_rate": online_venues / total_venues if total_venues > 0 else 0,
            "supported_symbols": list(self.market_data.keys()),
            "last_update": datetime.now()
        }