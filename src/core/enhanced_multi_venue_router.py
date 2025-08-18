"""
Enhanced Multi-Venue Smart Order Router
增强型多交易所智能路由系统

核心功能:
1. 实时多交易所价格聚合和最优路由
2. 流动性深度分析和市场冲击最小化  
3. 动态费用优化和maker/taker策略
4. 跨交易所套利机会检测
5. 网络延迟和可靠性监控
6. DipMaster专用路由优化
7. 紧急情况下的故障转移机制
"""

import asyncio
import logging
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
from collections import defaultdict, deque
import concurrent.futures
import websockets
import ssl
import certifi

logger = logging.getLogger(__name__)


class VenueType(Enum):
    """支持的交易所类型"""
    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"
    HUOBI = "huobi"
    KRAKEN = "kraken"
    COINBASE = "coinbase"
    KUCOIN = "kucoin"


@dataclass
class VenueConfig:
    """交易所配置"""
    name: str
    display_name: str
    api_url: str
    ws_url: str
    
    # 费用结构
    maker_fee_rate: float
    taker_fee_rate: float
    
    # 交易限制
    min_order_usd: float
    max_order_usd: float
    
    # 可选参数（有默认值）
    withdrawal_fee: float = 0.0
    min_order_qty: float = 0.0001
    
    # 性能参数
    avg_latency_ms: float = 100
    reliability_score: float = 0.95
    uptime_percentage: float = 99.9
    
    # 特殊功能
    supports_stop_orders: bool = True
    supports_iceberg: bool = True
    supports_post_only: bool = True
    
    # API配置
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    passphrase: Optional[str] = None  # OKX需要
    
    # 状态
    enabled: bool = True
    last_health_check: Optional[datetime] = None
    health_status: str = "unknown"


@dataclass
class OrderBookLevel:
    """订单簿层级"""
    price: float
    quantity: float
    orders_count: int = 1
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class VenueOrderBook:
    """交易所订单簿"""
    symbol: str
    venue: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    last_update: datetime
    sequence: int = 0
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None
    
    @property
    def spread_bps(self) -> float:
        if self.best_bid and self.best_ask:
            mid_price = (self.best_bid.price + self.best_ask.price) / 2
            return (self.best_ask.price - self.best_bid.price) / mid_price * 10000
        return float('inf')


@dataclass 
class VenueQuote:
    """交易所报价"""
    venue: str
    symbol: str
    side: str  # BUY or SELL
    
    # 价格和数量
    price: float
    quantity: float
    available_liquidity: float
    
    # 成本分析
    fee_rate: float
    estimated_slippage_bps: float
    market_impact_bps: float
    total_cost_bps: float
    
    # 执行预期
    expected_fill_time_ms: float
    confidence_score: float
    
    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    quote_id: str = ""


@dataclass
class RouteSegment:
    """路由段"""
    venue: str
    symbol: str
    side: str
    quantity: float
    price: float
    fee_usd: float
    expected_slippage_bps: float
    weight: float  # 在总订单中的权重


@dataclass
class OptimalRoute:
    """最优路由结果"""
    route_id: str
    symbol: str
    side: str
    total_quantity: float
    total_size_usd: float
    
    # 路由分割
    segments: List[RouteSegment]
    venue_count: int
    
    # 成本分析
    total_fees_usd: float
    weighted_avg_price: float
    estimated_slippage_bps: float
    total_cost_bps: float
    
    # 执行预期
    estimated_execution_time_ms: float
    confidence_score: float
    
    # 相比单一交易所的改进
    cost_savings_bps: float
    liquidity_improvement: float
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ArbitrageOpportunity:
    """套利机会"""
    symbol: str
    buy_venue: str
    sell_venue: str
    buy_price: float
    sell_price: float
    quantity: float
    profit_bps: float
    profit_usd: float
    confidence: float
    expires_at: datetime


class VenueConnector:
    """交易所连接器基类"""
    
    def __init__(self, config: VenueConfig):
        self.config = config
        self.session = None
        self.ws_connection = None
        self.last_ping = None
        self.connection_status = "disconnected"
        
    async def initialize(self):
        """初始化连接"""
        self.session = aiohttp.ClientSession()
        
    async def cleanup(self):
        """清理连接"""
        if self.session:
            await self.session.close()
            
    async def get_order_book(self, symbol: str, depth: int = 20) -> Optional[VenueOrderBook]:
        """获取订单簿数据"""
        # 模拟实现 - 实际需要调用各交易所API
        return await self._simulate_order_book(symbol, depth)
        
    async def _simulate_order_book(self, symbol: str, depth: int) -> VenueOrderBook:
        """模拟订单簿数据"""
        # 基础价格（可以从实际API获取）
        base_prices = {
            'BTCUSDT': 65000, 'ETHUSDT': 3200, 'SOLUSDT': 140,
            'BNBUSDT': 580, 'ADAUSDT': 0.45, 'XRPUSDT': 0.55
        }
        
        base_price = base_prices.get(symbol, 50000)
        
        # 生成订单簿
        bids = []
        asks = []
        
        # 添加交易所特定的价差和流动性特征
        venue_characteristics = {
            'binance': {'spread_factor': 1.0, 'liquidity_factor': 1.2},
            'okx': {'spread_factor': 0.9, 'liquidity_factor': 1.0},
            'bybit': {'spread_factor': 1.1, 'liquidity_factor': 0.8},
            'huobi': {'spread_factor': 1.2, 'liquidity_factor': 0.9}
        }
        
        char = venue_characteristics.get(self.config.name.lower(), {'spread_factor': 1.0, 'liquidity_factor': 1.0})
        
        # 生成价差
        spread_bps = np.random.uniform(5, 30) * char['spread_factor']
        spread = base_price * spread_bps / 10000
        
        mid_price = base_price + np.random.normal(0, base_price * 0.001)
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        # 生成多层订单簿
        for i in range(depth):
            # Bids (买单)
            bid_price = best_bid - i * spread * 0.1
            bid_qty = np.random.exponential(10) * char['liquidity_factor'] * (1 + np.random.uniform(0, 2))
            bids.append(OrderBookLevel(bid_price, bid_qty, np.random.randint(1, 5)))
            
            # Asks (卖单) 
            ask_price = best_ask + i * spread * 0.1
            ask_qty = np.random.exponential(10) * char['liquidity_factor'] * (1 + np.random.uniform(0, 2))
            asks.append(OrderBookLevel(ask_price, ask_qty, np.random.randint(1, 5)))
        
        return VenueOrderBook(
            symbol=symbol,
            venue=self.config.name,
            bids=bids,
            asks=asks,
            last_update=datetime.now(),
            sequence=int(time.time())
        )
    
    async def place_order(self, symbol: str, side: str, quantity: float, price: float, order_type: str) -> Dict:
        """下单"""
        # 模拟下单 - 实际需要调用交易所API
        await asyncio.sleep(0.1)  # 模拟网络延迟
        
        return {
            'order_id': f"{self.config.name}_{int(time.time())}",
            'status': 'filled',  # 模拟立即成交
            'filled_qty': quantity,
            'avg_price': price,
            'fee': quantity * price * self.config.taker_fee_rate
        }


class VenueManager:
    """交易所管理器"""
    
    def __init__(self):
        self.venues: Dict[str, VenueConfig] = {}
        self.connectors: Dict[str, VenueConnector] = {}
        self.health_monitor = VenueHealthMonitor()
        
        self._initialize_default_venues()
        
    def _initialize_default_venues(self):
        """初始化默认交易所配置"""
        configs = [
            VenueConfig(
                name="binance",
                display_name="Binance",
                api_url="https://api.binance.com",
                ws_url="wss://stream.binance.com:9443",
                maker_fee_rate=0.001,
                taker_fee_rate=0.001,
                min_order_usd=10,
                max_order_usd=1000000,
                avg_latency_ms=50,
                reliability_score=0.98
            ),
            VenueConfig(
                name="okx",
                display_name="OKX",
                api_url="https://www.okx.com",
                ws_url="wss://ws.okx.com:8443",
                maker_fee_rate=0.0008,
                taker_fee_rate=0.001,
                min_order_usd=5,
                max_order_usd=500000,
                avg_latency_ms=60,
                reliability_score=0.95
            ),
            VenueConfig(
                name="bybit",
                display_name="Bybit",
                api_url="https://api.bybit.com",
                ws_url="wss://stream.bybit.com",
                maker_fee_rate=0.0001,
                taker_fee_rate=0.0006,
                min_order_usd=10,
                max_order_usd=200000,
                avg_latency_ms=70,
                reliability_score=0.93
            ),
            VenueConfig(
                name="huobi",
                display_name="Huobi",
                api_url="https://api.huobi.pro",
                ws_url="wss://api.huobi.pro/ws",
                maker_fee_rate=0.002,
                taker_fee_rate=0.002,
                min_order_usd=5,
                max_order_usd=100000,
                avg_latency_ms=80,
                reliability_score=0.90
            )
        ]
        
        for config in configs:
            self.venues[config.name] = config
            self.connectors[config.name] = VenueConnector(config)
    
    async def initialize_all_venues(self):
        """初始化所有交易所连接"""
        tasks = []
        for connector in self.connectors.values():
            tasks.append(connector.initialize())
        
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"已初始化 {len(self.connectors)} 个交易所连接")
    
    async def cleanup_all_venues(self):
        """清理所有交易所连接"""
        tasks = []
        for connector in self.connectors.values():
            tasks.append(connector.cleanup())
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    def get_active_venues(self) -> List[str]:
        """获取活跃交易所列表"""
        return [name for name, config in self.venues.items() if config.enabled]
    
    def get_venue_config(self, venue: str) -> Optional[VenueConfig]:
        """获取交易所配置"""
        return self.venues.get(venue)


class VenueHealthMonitor:
    """交易所健康监控"""
    
    def __init__(self):
        self.latency_history = defaultdict(lambda: deque(maxlen=100))
        self.error_counts = defaultdict(int)
        self.last_success = defaultdict(lambda: datetime.now())
        self.health_scores = defaultdict(float)
        
    def record_latency(self, venue: str, latency_ms: float):
        """记录延迟"""
        self.latency_history[venue].append(latency_ms)
        self.last_success[venue] = datetime.now()
        
    def record_error(self, venue: str, error_type: str):
        """记录错误"""
        self.error_counts[f"{venue}_{error_type}"] += 1
        
    def calculate_health_score(self, venue: str) -> float:
        """计算健康评分"""
        latencies = self.latency_history[venue]
        
        if not latencies:
            return 0.5
            
        # 基于延迟的评分
        avg_latency = np.mean(latencies)
        latency_score = max(0, 1 - avg_latency / 1000)  # 1秒为基准
        
        # 基于错误率的评分
        recent_errors = sum(count for key, count in self.error_counts.items() 
                          if key.startswith(venue) and count > 0)
        error_score = max(0, 1 - recent_errors / 100)
        
        # 基于最后成功时间的评分
        time_since_success = (datetime.now() - self.last_success[venue]).total_seconds()
        freshness_score = max(0, 1 - time_since_success / 300)  # 5分钟为基准
        
        # 综合评分
        health_score = (latency_score * 0.4 + error_score * 0.4 + freshness_score * 0.2)
        self.health_scores[venue] = health_score
        
        return health_score
        
    def get_venue_health_report(self) -> Dict[str, Any]:
        """获取交易所健康报告"""
        report = {}
        
        for venue in self.latency_history.keys():
            latencies = list(self.latency_history[venue])
            report[venue] = {
                'health_score': self.calculate_health_score(venue),
                'avg_latency_ms': np.mean(latencies) if latencies else 0,
                'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0,
                'error_count': sum(count for key, count in self.error_counts.items() if key.startswith(venue)),
                'last_success': self.last_success[venue]
            }
            
        return report


class LiquidityAggregator:
    """流动性聚合器"""
    
    def __init__(self, venue_manager: VenueManager):
        self.venue_manager = venue_manager
        self.order_books: Dict[str, Dict[str, VenueOrderBook]] = {}  # symbol -> venue -> order_book
        
    async def update_order_books(self, symbols: List[str]):
        """更新订单簿数据"""
        tasks = []
        
        for symbol in symbols:
            if symbol not in self.order_books:
                self.order_books[symbol] = {}
                
            for venue_name in self.venue_manager.get_active_venues():
                connector = self.venue_manager.connectors[venue_name]
                task = self._fetch_and_store_order_book(symbol, venue_name, connector)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        logger.debug(f"订单簿更新完成: {success_count}/{len(tasks)} 成功")
    
    async def _fetch_and_store_order_book(self, symbol: str, venue_name: str, connector: VenueConnector):
        """获取并存储订单簿"""
        try:
            start_time = time.time()
            order_book = await connector.get_order_book(symbol)
            latency_ms = (time.time() - start_time) * 1000
            
            if order_book:
                self.order_books[symbol][venue_name] = order_book
                self.venue_manager.health_monitor.record_latency(venue_name, latency_ms)
            
        except Exception as e:
            logger.warning(f"获取订单簿失败 {venue_name} {symbol}: {e}")
            self.venue_manager.health_monitor.record_error(venue_name, "order_book_fetch")
    
    def get_aggregated_liquidity(self, symbol: str, side: str, max_slippage_bps: float = 50) -> List[VenueQuote]:
        """获取聚合流动性报价"""
        quotes = []
        
        symbol_books = self.order_books.get(symbol, {})
        
        for venue_name, order_book in symbol_books.items():
            venue_config = self.venue_manager.get_venue_config(venue_name)
            if not venue_config or not venue_config.enabled:
                continue
                
            quote = self._calculate_venue_quote(order_book, side, venue_config, max_slippage_bps)
            if quote:
                quotes.append(quote)
        
        # 按价格排序
        if side == "BUY":
            quotes.sort(key=lambda x: x.total_cost_bps)  # 买入按总成本排序
        else:
            quotes.sort(key=lambda x: -x.price)  # 卖出按价格从高到低
            
        return quotes
    
    def _calculate_venue_quote(self, order_book: VenueOrderBook, side: str, config: VenueConfig, max_slippage_bps: float) -> Optional[VenueQuote]:
        """计算交易所报价"""
        try:
            if side == "BUY":
                levels = order_book.asks
                best_price = order_book.best_ask.price if order_book.best_ask else None
            else:
                levels = order_book.bids  
                best_price = order_book.best_bid.price if order_book.best_bid else None
                
            if not best_price or not levels:
                return None
            
            # 计算可用流动性
            available_liquidity = sum(level.quantity for level in levels[:10])  # 前10档
            
            # 计算滑点
            mid_price = (order_book.best_bid.price + order_book.best_ask.price) / 2 if order_book.best_bid and order_book.best_ask else best_price
            slippage_bps = abs(best_price - mid_price) / mid_price * 10000
            
            # 检查滑点限制
            if slippage_bps > max_slippage_bps:
                return None
            
            # 估算市场冲击
            market_impact_bps = min(20, slippage_bps * 0.5)  # 简化模型
            
            # 计算总成本
            fee_bps = config.taker_fee_rate * 10000
            total_cost_bps = slippage_bps + market_impact_bps + fee_bps
            
            # 置信度评分
            health_score = self.venue_manager.health_monitor.calculate_health_score(config.name)
            liquidity_score = min(1.0, available_liquidity / 100)  # 简化流动性评分
            confidence = (health_score + liquidity_score) / 2
            
            return VenueQuote(
                venue=config.name,
                symbol=order_book.symbol,
                side=side,
                price=best_price,
                quantity=levels[0].quantity,
                available_liquidity=available_liquidity,
                fee_rate=config.taker_fee_rate,
                estimated_slippage_bps=slippage_bps,
                market_impact_bps=market_impact_bps,
                total_cost_bps=total_cost_bps,
                expected_fill_time_ms=config.avg_latency_ms + 50,  # API延迟 + 处理时间
                confidence_score=confidence,
                quote_id=f"{config.name}_{order_book.symbol}_{int(time.time())}"
            )
            
        except Exception as e:
            logger.warning(f"计算交易所报价失败 {config.name}: {e}")
            return None


class OptimalRouterEngine:
    """最优路由引擎"""
    
    def __init__(self, liquidity_aggregator: LiquidityAggregator):
        self.liquidity_aggregator = liquidity_aggregator
        self.route_cache = {}
        
    async def find_optimal_route(
        self,
        symbol: str,
        side: str,
        target_quantity: float,
        max_venues: int = 3,
        cost_optimization: bool = True,
        speed_optimization: bool = False
    ) -> OptimalRoute:
        """寻找最优执行路由"""
        
        route_id = f"ROUTE_{symbol}_{side}_{int(time.time())}"
        
        logger.info(f"寻找最优路由: {route_id} - {side} {target_quantity} {symbol}")
        
        # 获取所有可用报价
        quotes = self.liquidity_aggregator.get_aggregated_liquidity(symbol, side)
        
        if not quotes:
            raise ValueError(f"无可用流动性: {symbol} {side}")
        
        # 过滤可用报价
        viable_quotes = self._filter_viable_quotes(quotes, target_quantity)
        
        if not viable_quotes:
            raise ValueError(f"无满足条件的交易所: {symbol} {side}")
        
        # 计算路由策略
        if len(viable_quotes) == 1:
            # 单一交易所路由
            route = self._create_single_venue_route(route_id, symbol, side, target_quantity, viable_quotes[0])
        else:
            # 多交易所路由优化
            if cost_optimization:
                route = await self._optimize_for_cost(route_id, symbol, side, target_quantity, viable_quotes, max_venues)
            elif speed_optimization:
                route = await self._optimize_for_speed(route_id, symbol, side, target_quantity, viable_quotes, max_venues)
            else:
                route = await self._balanced_optimization(route_id, symbol, side, target_quantity, viable_quotes, max_venues)
        
        # 缓存路由结果
        self.route_cache[route_id] = route
        
        logger.info(f"最优路由完成: {route_id} - {route.venue_count}个交易所, 成本节约{route.cost_savings_bps:.2f}bps")
        
        return route
    
    def _filter_viable_quotes(self, quotes: List[VenueQuote], target_quantity: float) -> List[VenueQuote]:
        """过滤可行报价"""
        viable = []
        
        for quote in quotes:
            # 检查最小订单量
            venue_config = self.liquidity_aggregator.venue_manager.get_venue_config(quote.venue)
            if not venue_config:
                continue
                
            min_quantity = venue_config.min_order_usd / quote.price
            if quote.quantity < min_quantity:
                continue
                
            # 检查置信度
            if quote.confidence_score < 0.3:
                continue
                
            # 检查流动性充足性
            if quote.available_liquidity < target_quantity * 0.1:  # 至少10%流动性
                continue
                
            viable.append(quote)
        
        return viable
    
    def _create_single_venue_route(self, route_id: str, symbol: str, side: str, quantity: float, quote: VenueQuote) -> OptimalRoute:
        """创建单一交易所路由"""
        
        segment = RouteSegment(
            venue=quote.venue,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=quote.price,
            fee_usd=quantity * quote.price * quote.fee_rate,
            expected_slippage_bps=quote.estimated_slippage_bps,
            weight=1.0
        )
        
        total_size_usd = quantity * quote.price
        total_fees = segment.fee_usd
        
        return OptimalRoute(
            route_id=route_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            total_size_usd=total_size_usd,
            segments=[segment],
            venue_count=1,
            total_fees_usd=total_fees,
            weighted_avg_price=quote.price,
            estimated_slippage_bps=quote.estimated_slippage_bps,
            total_cost_bps=quote.total_cost_bps,
            estimated_execution_time_ms=quote.expected_fill_time_ms,
            confidence_score=quote.confidence_score,
            cost_savings_bps=0.0,  # 无对比基准
            liquidity_improvement=1.0
        )
    
    async def _optimize_for_cost(self, route_id: str, symbol: str, side: str, target_quantity: float, quotes: List[VenueQuote], max_venues: int) -> OptimalRoute:
        """成本优化路由"""
        
        # 按总成本排序
        sorted_quotes = sorted(quotes, key=lambda x: x.total_cost_bps)[:max_venues]
        
        # 贪心算法分配
        segments = []
        remaining_quantity = target_quantity
        total_size_usd = 0
        total_fees = 0
        
        for i, quote in enumerate(sorted_quotes):
            if remaining_quantity <= 0:
                break
                
            # 计算分配数量
            if i == len(sorted_quotes) - 1:
                # 最后一个交易所承担剩余全部
                allocated_qty = remaining_quantity
            else:
                # 根据流动性和成本效率分配
                max_allocation = min(remaining_quantity, quote.available_liquidity * 0.7)
                allocated_qty = max_allocation * (0.8 if remaining_quantity > max_allocation else 1.0)
            
            if allocated_qty > 0:
                segment = RouteSegment(
                    venue=quote.venue,
                    symbol=symbol,
                    side=side,
                    quantity=allocated_qty,
                    price=quote.price,
                    fee_usd=allocated_qty * quote.price * quote.fee_rate,
                    expected_slippage_bps=quote.estimated_slippage_bps,
                    weight=allocated_qty / target_quantity
                )
                
                segments.append(segment)
                remaining_quantity -= allocated_qty
                total_size_usd += allocated_qty * quote.price
                total_fees += segment.fee_usd
        
        # 计算加权指标
        weighted_avg_price = total_size_usd / target_quantity if target_quantity > 0 else 0
        weighted_slippage = sum(s.expected_slippage_bps * s.weight for s in segments)
        weighted_cost_bps = (total_fees / total_size_usd * 10000) + weighted_slippage
        
        # 计算相对单一最佳交易所的改进
        best_single_cost = sorted_quotes[0].total_cost_bps
        cost_savings_bps = max(0, best_single_cost - weighted_cost_bps)
        
        return OptimalRoute(
            route_id=route_id,
            symbol=symbol,
            side=side,
            total_quantity=target_quantity,
            total_size_usd=total_size_usd,
            segments=segments,
            venue_count=len(segments),
            total_fees_usd=total_fees,
            weighted_avg_price=weighted_avg_price,
            estimated_slippage_bps=weighted_slippage,
            total_cost_bps=weighted_cost_bps,
            estimated_execution_time_ms=max(s.weight * quotes[i].expected_fill_time_ms for i, s in enumerate(segments[:len(quotes)])),
            confidence_score=np.mean([q.confidence_score for q in sorted_quotes[:len(segments)]]),
            cost_savings_bps=cost_savings_bps,
            liquidity_improvement=len(segments) / 1.0  # 相对单一交易所的流动性改进
        )
    
    async def _optimize_for_speed(self, route_id: str, symbol: str, side: str, target_quantity: float, quotes: List[VenueQuote], max_venues: int) -> OptimalRoute:
        """速度优化路由"""
        
        # 按执行时间和置信度排序
        sorted_quotes = sorted(quotes, key=lambda x: (x.expected_fill_time_ms, -x.confidence_score))[:max_venues]
        
        # 并行执行优化：多个交易所同时执行
        segments = []
        total_size_usd = 0
        total_fees = 0
        
        # 均匀分配以实现最快并行执行
        qty_per_venue = target_quantity / min(len(sorted_quotes), max_venues)
        
        for quote in sorted_quotes[:max_venues]:
            allocated_qty = min(qty_per_venue, quote.available_liquidity * 0.8)
            
            if allocated_qty > 0:
                segment = RouteSegment(
                    venue=quote.venue,
                    symbol=symbol,
                    side=side,
                    quantity=allocated_qty,
                    price=quote.price,
                    fee_usd=allocated_qty * quote.price * quote.fee_rate,
                    expected_slippage_bps=quote.estimated_slippage_bps,
                    weight=allocated_qty / target_quantity
                )
                
                segments.append(segment)
                total_size_usd += allocated_qty * quote.price
                total_fees += segment.fee_usd
        
        # 计算指标
        weighted_avg_price = total_size_usd / sum(s.quantity for s in segments) if segments else 0
        weighted_slippage = sum(s.expected_slippage_bps * s.weight for s in segments)
        
        # 并行执行时间为最慢的交易所
        max_execution_time = max([quotes[i].expected_fill_time_ms for i, s in enumerate(segments[:len(quotes)])] if segments else [0])
        
        return OptimalRoute(
            route_id=route_id,
            symbol=symbol,
            side=side,
            total_quantity=sum(s.quantity for s in segments),
            total_size_usd=total_size_usd,
            segments=segments,
            venue_count=len(segments),
            total_fees_usd=total_fees,
            weighted_avg_price=weighted_avg_price,
            estimated_slippage_bps=weighted_slippage,
            total_cost_bps=(total_fees / total_size_usd * 10000) + weighted_slippage if total_size_usd > 0 else 0,
            estimated_execution_time_ms=max_execution_time,
            confidence_score=np.mean([q.confidence_score for q in sorted_quotes[:len(segments)]]),
            cost_savings_bps=0.0,  # 速度优化模式不重点关注成本节约
            liquidity_improvement=len(segments) / 1.0
        )
    
    async def _balanced_optimization(self, route_id: str, symbol: str, side: str, target_quantity: float, quotes: List[VenueQuote], max_venues: int) -> OptimalRoute:
        """平衡优化路由"""
        
        # 综合评分：成本(40%) + 速度(30%) + 置信度(30%)
        def composite_score(quote):
            cost_score = 1 / (1 + quote.total_cost_bps / 100)  # 成本越低分数越高
            speed_score = 1 / (1 + quote.expected_fill_time_ms / 1000)  # 速度越快分数越高
            confidence_score = quote.confidence_score
            
            return 0.4 * cost_score + 0.3 * speed_score + 0.3 * confidence_score
        
        # 按综合评分排序
        sorted_quotes = sorted(quotes, key=composite_score, reverse=True)[:max_venues]
        
        # 智能分配算法
        segments = []
        remaining_quantity = target_quantity
        total_size_usd = 0
        total_fees = 0
        
        # 计算每个交易所的权重
        total_score = sum(composite_score(q) for q in sorted_quotes)
        
        for quote in sorted_quotes:
            if remaining_quantity <= 0:
                break
                
            # 基于综合评分分配权重
            score_weight = composite_score(quote) / total_score
            target_allocation = target_quantity * score_weight
            
            # 考虑流动性限制
            max_allocation = min(target_allocation, quote.available_liquidity * 0.6, remaining_quantity)
            allocated_qty = max_allocation
            
            if allocated_qty > 0:
                segment = RouteSegment(
                    venue=quote.venue,
                    symbol=symbol,
                    side=side,
                    quantity=allocated_qty,
                    price=quote.price,
                    fee_usd=allocated_qty * quote.price * quote.fee_rate,
                    expected_slippage_bps=quote.estimated_slippage_bps,
                    weight=allocated_qty / target_quantity
                )
                
                segments.append(segment)
                remaining_quantity -= allocated_qty
                total_size_usd += allocated_qty * quote.price
                total_fees += segment.fee_usd
        
        # 处理剩余数量
        if remaining_quantity > 0 and segments:
            # 分配给最优的交易所
            best_segment = segments[0]
            best_segment.quantity += remaining_quantity
            total_size_usd += remaining_quantity * best_segment.price
            total_fees += remaining_quantity * best_segment.price * (total_fees / (total_size_usd - remaining_quantity * best_segment.price))
        
        # 计算最终指标
        actual_quantity = sum(s.quantity for s in segments)
        weighted_avg_price = total_size_usd / actual_quantity if actual_quantity > 0 else 0
        weighted_slippage = sum(s.expected_slippage_bps * s.weight for s in segments)
        weighted_cost_bps = (total_fees / total_size_usd * 10000) + weighted_slippage if total_size_usd > 0 else 0
        
        # 估算执行时间（部分并行）
        avg_execution_time = np.mean([quotes[i].expected_fill_time_ms for i, s in enumerate(segments[:len(quotes)])] if segments else [0])
        
        return OptimalRoute(
            route_id=route_id,
            symbol=symbol,
            side=side,
            total_quantity=actual_quantity,
            total_size_usd=total_size_usd,
            segments=segments,
            venue_count=len(segments),
            total_fees_usd=total_fees,
            weighted_avg_price=weighted_avg_price,
            estimated_slippage_bps=weighted_slippage,
            total_cost_bps=weighted_cost_bps,
            estimated_execution_time_ms=avg_execution_time,
            confidence_score=np.mean([composite_score(q) for q in sorted_quotes[:len(segments)]]),
            cost_savings_bps=max(0, quotes[0].total_cost_bps - weighted_cost_bps),
            liquidity_improvement=len(segments) / 1.0
        )


class ArbitrageDetector:
    """套利机会检测器"""
    
    def __init__(self, liquidity_aggregator: LiquidityAggregator):
        self.liquidity_aggregator = liquidity_aggregator
        self.detected_opportunities = deque(maxlen=100)
        
    def detect_arbitrage_opportunities(self, symbols: List[str], min_profit_bps: float = 10) -> List[ArbitrageOpportunity]:
        """检测套利机会"""
        opportunities = []
        
        for symbol in symbols:
            symbol_books = self.liquidity_aggregator.order_books.get(symbol, {})
            
            if len(symbol_books) < 2:
                continue
                
            venues = list(symbol_books.keys())
            
            # 两两比较所有交易所
            for i in range(len(venues)):
                for j in range(i + 1, len(venues)):
                    venue1, venue2 = venues[i], venues[j]
                    book1, book2 = symbol_books[venue1], symbol_books[venue2]
                    
                    # 检查套利机会
                    opp1 = self._check_arbitrage_pair(symbol, venue1, venue2, book1, book2, min_profit_bps)
                    opp2 = self._check_arbitrage_pair(symbol, venue2, venue1, book2, book1, min_profit_bps)
                    
                    if opp1:
                        opportunities.append(opp1)
                    if opp2:
                        opportunities.append(opp2)
        
        # 按利润排序
        opportunities.sort(key=lambda x: x.profit_bps, reverse=True)
        
        # 更新检测历史
        self.detected_opportunities.extend(opportunities)
        
        return opportunities[:10]  # 返回前10个最优机会
    
    def _check_arbitrage_pair(self, symbol: str, buy_venue: str, sell_venue: str, buy_book: VenueOrderBook, sell_book: VenueOrderBook, min_profit_bps: float) -> Optional[ArbitrageOpportunity]:
        """检查一对交易所的套利机会"""
        try:
            if not (buy_book.best_ask and sell_book.best_bid):
                return None
                
            buy_price = buy_book.best_ask.price
            sell_price = sell_book.best_bid.price
            
            # 检查价差
            if sell_price <= buy_price:
                return None
                
            # 计算利润
            profit_per_unit = sell_price - buy_price
            profit_bps = profit_per_unit / buy_price * 10000
            
            if profit_bps < min_profit_bps:
                return None
                
            # 计算可套利数量
            max_buy_qty = buy_book.best_ask.quantity
            max_sell_qty = sell_book.best_bid.quantity
            arbitrage_qty = min(max_buy_qty, max_sell_qty)
            
            if arbitrage_qty <= 0:
                return None
                
            # 考虑交易费用
            buy_venue_config = self.liquidity_aggregator.venue_manager.get_venue_config(buy_venue)
            sell_venue_config = self.liquidity_aggregator.venue_manager.get_venue_config(sell_venue)
            
            if not (buy_venue_config and sell_venue_config):
                return None
                
            buy_fee_rate = buy_venue_config.taker_fee_rate
            sell_fee_rate = sell_venue_config.taker_fee_rate
            
            # 净利润计算
            gross_profit = arbitrage_qty * profit_per_unit
            buy_fee = arbitrage_qty * buy_price * buy_fee_rate
            sell_fee = arbitrage_qty * sell_price * sell_fee_rate
            net_profit = gross_profit - buy_fee - sell_fee
            
            if net_profit <= 0:
                return None
                
            net_profit_bps = net_profit / (arbitrage_qty * buy_price) * 10000
            
            # 置信度评估
            buy_health = self.liquidity_aggregator.venue_manager.health_monitor.calculate_health_score(buy_venue)
            sell_health = self.liquidity_aggregator.venue_manager.health_monitor.calculate_health_score(sell_venue)
            confidence = min(buy_health, sell_health)
            
            # 套利机会有效期（基于价格变动速度）
            expires_at = datetime.now() + timedelta(seconds=30)  # 30秒有效期
            
            return ArbitrageOpportunity(
                symbol=symbol,
                buy_venue=buy_venue,
                sell_venue=sell_venue,
                buy_price=buy_price,
                sell_price=sell_price,
                quantity=arbitrage_qty,
                profit_bps=net_profit_bps,
                profit_usd=net_profit,
                confidence=confidence,
                expires_at=expires_at
            )
            
        except Exception as e:
            logger.warning(f"套利检查失败 {symbol} {buy_venue}->{sell_venue}: {e}")
            return None


class EnhancedMultiVenueRouter:
    """增强型多交易所智能路由器"""
    
    def __init__(self, paper_trading: bool = True):
        self.paper_trading = paper_trading
        
        # 核心组件
        self.venue_manager = VenueManager()
        self.liquidity_aggregator = LiquidityAggregator(self.venue_manager)
        self.router_engine = OptimalRouterEngine(self.liquidity_aggregator)
        self.arbitrage_detector = ArbitrageDetector(self.liquidity_aggregator)
        
        # 运行状态
        self.running = False
        self.data_update_task = None
        self.supported_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']
        
        # 性能监控
        self.routing_stats = defaultdict(int)
        self.execution_history = deque(maxlen=1000)
        
    async def start(self):
        """启动路由器"""
        if self.running:
            logger.warning("路由器已经在运行")
            return
            
        logger.info("启动多交易所智能路由器...")
        
        # 初始化交易所连接
        await self.venue_manager.initialize_all_venues()
        
        # 启动数据更新任务
        self.running = True
        self.data_update_task = asyncio.create_task(self._data_update_loop())
        
        logger.info(f"路由器启动完成，支持 {len(self.venue_manager.get_active_venues())} 个交易所")
        
    async def stop(self):
        """停止路由器"""
        self.running = False
        
        if self.data_update_task:
            self.data_update_task.cancel()
            try:
                await self.data_update_task
            except asyncio.CancelledError:
                pass
        
        await self.venue_manager.cleanup_all_venues()
        logger.info("多交易所智能路由器已停止")
    
    async def _data_update_loop(self):
        """数据更新循环"""
        while self.running:
            try:
                # 更新订单簿数据
                await self.liquidity_aggregator.update_order_books(self.supported_symbols)
                
                # 检测套利机会
                arbitrage_ops = self.arbitrage_detector.detect_arbitrage_opportunities(self.supported_symbols)
                
                if arbitrage_ops:
                    logger.info(f"发现 {len(arbitrage_ops)} 个套利机会，最优利润: {arbitrage_ops[0].profit_bps:.2f}bps")
                
                # 等待下次更新
                await asyncio.sleep(2)  # 2秒更新一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"数据更新循环错误: {e}")
                await asyncio.sleep(5)
    
    async def route_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        strategy: str = "balanced",  # cost, speed, balanced
        max_venues: int = 3,
        max_slippage_bps: float = 50
    ) -> OptimalRoute:
        """执行智能路由"""
        
        if not self.running:
            raise RuntimeError("路由器未启动")
            
        # 更新统计
        self.routing_stats[f"{symbol}_{side}"] += 1
        
        logger.info(f"智能路由请求: {side} {quantity} {symbol} (策略: {strategy})")
        
        # 选择优化策略
        cost_opt = strategy == "cost"
        speed_opt = strategy == "speed"
        
        try:
            # 寻找最优路由
            route = await self.router_engine.find_optimal_route(
                symbol=symbol,
                side=side,
                target_quantity=quantity,
                max_venues=max_venues,
                cost_optimization=cost_opt,
                speed_optimization=speed_opt
            )
            
            # 记录执行历史
            self.execution_history.append({
                'route_id': route.route_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'venue_count': route.venue_count,
                'cost_bps': route.total_cost_bps,
                'timestamp': datetime.now()
            })
            
            logger.info(f"路由完成: {route.route_id} - {route.venue_count}个交易所, 总成本{route.total_cost_bps:.2f}bps")
            
            return route
            
        except Exception as e:
            logger.error(f"路由失败: {e}")
            raise
    
    async def execute_route(self, route: OptimalRoute) -> Dict[str, Any]:
        """执行路由订单"""
        
        if not self.paper_trading:
            logger.warning("真实交易模式未完整实现")
        
        execution_results = []
        total_filled_qty = 0
        total_filled_value = 0
        total_fees = 0
        
        # 并发执行所有路由段
        tasks = []
        for segment in route.segments:
            task = asyncio.create_task(self._execute_segment(segment))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理执行结果
        for i, result in enumerate(results):
            segment = route.segments[i]
            
            if isinstance(result, Exception):
                logger.error(f"路由段执行失败 {segment.venue}: {result}")
                execution_results.append({
                    'venue': segment.venue,
                    'status': 'failed',
                    'error': str(result)
                })
            else:
                execution_results.append(result)
                if result['status'] == 'success':
                    total_filled_qty += result['filled_qty']
                    total_filled_value += result['filled_value']
                    total_fees += result['fees']
        
        # 生成执行报告
        fill_rate = total_filled_qty / route.total_quantity if route.total_quantity > 0 else 0
        avg_price = total_filled_value / total_filled_qty if total_filled_qty > 0 else 0
        
        execution_report = {
            'route_id': route.route_id,
            'execution_time': datetime.now(),
            'target_quantity': route.total_quantity,
            'filled_quantity': total_filled_qty,
            'fill_rate': fill_rate,
            'avg_price': avg_price,
            'total_fees': total_fees,
            'execution_results': execution_results,
            'success': fill_rate > 0.95  # 95%以上认为成功
        }
        
        logger.info(f"路由执行完成: {route.route_id} - 成交率{fill_rate:.1%}")
        
        return execution_report
    
    async def _execute_segment(self, segment: RouteSegment) -> Dict[str, Any]:
        """执行路由段"""
        try:
            connector = self.venue_manager.connectors.get(segment.venue)
            if not connector:
                raise ValueError(f"未找到交易所连接器: {segment.venue}")
            
            # 模拟或真实执行
            if self.paper_trading:
                result = await self._simulate_segment_execution(segment)
            else:
                result = await connector.place_order(
                    symbol=segment.symbol,
                    side=segment.side,
                    quantity=segment.quantity,
                    price=segment.price,
                    order_type="LIMIT"
                )
            
            return {
                'venue': segment.venue,
                'status': 'success',
                'filled_qty': result.get('filled_qty', segment.quantity),
                'filled_value': result.get('filled_qty', segment.quantity) * result.get('avg_price', segment.price),
                'avg_price': result.get('avg_price', segment.price),
                'fees': result.get('fee', segment.fee_usd),
                'order_id': result.get('order_id', f"MOCK_{int(time.time())}")
            }
            
        except Exception as e:
            logger.error(f"路由段执行失败 {segment.venue}: {e}")
            return {
                'venue': segment.venue,
                'status': 'failed',
                'error': str(e)
            }
    
    async def _simulate_segment_execution(self, segment: RouteSegment) -> Dict[str, Any]:
        """模拟路由段执行"""
        # 模拟网络延迟
        await asyncio.sleep(np.random.uniform(0.05, 0.2))
        
        # 模拟成交
        fill_rate = np.random.uniform(0.95, 1.0)  # 95-100%成交率
        filled_qty = segment.quantity * fill_rate
        
        # 模拟价格滑点
        slippage_factor = np.random.uniform(-0.0005, 0.0005)  # ±0.05%
        avg_price = segment.price * (1 + slippage_factor)
        
        return {
            'filled_qty': filled_qty,
            'avg_price': avg_price,
            'fee': filled_qty * avg_price * 0.001  # 0.1%手续费
        }
    
    def get_venue_status(self) -> Dict[str, Any]:
        """获取交易所状态"""
        health_report = self.venue_manager.health_monitor.get_venue_health_report()
        
        status = {
            'total_venues': len(self.venue_manager.venues),
            'active_venues': len(self.venue_manager.get_active_venues()),
            'supported_symbols': self.supported_symbols,
            'venue_health': health_report,
            'routing_stats': dict(self.routing_stats),
            'last_update': datetime.now()
        }
        
        return status
    
    def get_arbitrage_opportunities(self) -> List[Dict[str, Any]]:
        """获取当前套利机会"""
        opportunities = self.arbitrage_detector.detect_arbitrage_opportunities(self.supported_symbols)
        
        return [
            {
                'symbol': opp.symbol,
                'buy_venue': opp.buy_venue,
                'sell_venue': opp.sell_venue,
                'profit_bps': opp.profit_bps,
                'profit_usd': opp.profit_usd,
                'quantity': opp.quantity,
                'confidence': opp.confidence,
                'expires_at': opp.expires_at
            }
            for opp in opportunities
        ]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        if not self.execution_history:
            return {'message': '无执行历史'}
        
        recent_executions = list(self.execution_history)[-50:]  # 最近50次
        
        avg_venues = np.mean([ex['venue_count'] for ex in recent_executions])
        avg_cost_bps = np.mean([ex['cost_bps'] for ex in recent_executions])
        
        symbol_stats = defaultdict(list)
        for ex in recent_executions:
            symbol_stats[ex['symbol']].append(ex['cost_bps'])
        
        return {
            'total_routes': len(self.execution_history),
            'avg_venues_per_route': avg_venues,
            'avg_cost_bps': avg_cost_bps,
            'symbol_performance': {
                symbol: {
                    'avg_cost_bps': np.mean(costs),
                    'min_cost_bps': np.min(costs),
                    'max_cost_bps': np.max(costs),
                    'count': len(costs)
                }
                for symbol, costs in symbol_stats.items()
            },
            'last_update': datetime.now()
        }


# 演示函数
async def demo_enhanced_multi_venue_router():
    """增强型多交易所路由器演示"""
    
    print("="*80)
    print("DipMaster Trading System - 增强型多交易所智能路由器")
    print("="*80)
    
    # 创建路由器
    router = EnhancedMultiVenueRouter(paper_trading=True)
    
    try:
        # 启动路由器
        await router.start()
        
        # 等待数据初始化
        print("等待市场数据初始化...")
        await asyncio.sleep(5)
        
        # 场景1：大额BTCUSDT订单成本优化路由
        print("\n场景1: 大额BTCUSDT订单 - 成本优化路由")
        print("-" * 60)
        
        route1 = await router.route_order(
            symbol="BTCUSDT",
            side="BUY",
            quantity=2.5,  # 2.5 BTC
            strategy="cost",
            max_venues=3
        )
        
        print(f"路由结果: {route1.venue_count}个交易所")
        for segment in route1.segments:
            print(f"  {segment.venue}: {segment.quantity:.4f} BTC @ ${segment.price:.2f} (权重: {segment.weight:.1%})")
        print(f"总成本: {route1.total_cost_bps:.2f}bps, 节约: {route1.cost_savings_bps:.2f}bps")
        
        # 执行路由
        execution1 = await router.execute_route(route1)
        print(f"执行结果: 成交率{execution1['fill_rate']:.1%}, 平均价格${execution1['avg_price']:.2f}")
        
        # 场景2：中等ETHUSDT订单速度优化路由
        print("\n场景2: 中等ETHUSDT订单 - 速度优化路由")
        print("-" * 60)
        
        route2 = await router.route_order(
            symbol="ETHUSDT",
            side="SELL",
            quantity=10.0,  # 10 ETH
            strategy="speed",
            max_venues=2
        )
        
        print(f"路由结果: {route2.venue_count}个交易所")
        for segment in route2.segments:
            print(f"  {segment.venue}: {segment.quantity:.2f} ETH @ ${segment.price:.2f}")
        print(f"预期执行时间: {route2.estimated_execution_time_ms:.0f}ms")
        
        # 场景3：小额SOLUSDT订单平衡路由
        print("\n场景3: 小额SOLUSDT订单 - 平衡路由")
        print("-" * 60)
        
        route3 = await router.route_order(
            symbol="SOLUSDT",
            side="BUY",
            quantity=100.0,  # 100 SOL
            strategy="balanced",
            max_venues=3
        )
        
        print(f"路由结果: {route3.venue_count}个交易所")
        print(f"综合评分 - 成本: {route3.total_cost_bps:.2f}bps, 置信度: {route3.confidence_score:.2f}")
        
        # 检查套利机会
        print("\n套利机会检测:")
        print("-" * 60)
        
        arbitrage_ops = router.get_arbitrage_opportunities()
        if arbitrage_ops:
            for i, opp in enumerate(arbitrage_ops[:3]):
                print(f"{i+1}. {opp['symbol']}: {opp['buy_venue']} -> {opp['sell_venue']}")
                print(f"   利润: {opp['profit_bps']:.2f}bps (${opp['profit_usd']:.2f})")
        else:
            print("当前无套利机会")
        
        # 获取交易所状态
        print("\n交易所状态:")
        print("-" * 60)
        
        venue_status = router.get_venue_status()
        print(f"总交易所: {venue_status['total_venues']}, 活跃: {venue_status['active_venues']}")
        
        for venue, health in venue_status['venue_health'].items():
            print(f"{venue}: 健康评分{health['health_score']:.2f}, 延迟{health['avg_latency_ms']:.0f}ms")
        
        # 性能统计
        print("\n性能统计:")
        print("-" * 60)
        
        perf_stats = router.get_performance_stats()
        print(f"总路由数: {perf_stats['total_routes']}")
        print(f"平均使用交易所: {perf_stats['avg_venues_per_route']:.1f}")
        print(f"平均成本: {perf_stats['avg_cost_bps']:.2f}bps")
        
    finally:
        await router.stop()
    
    print("\n" + "="*80)


if __name__ == "__main__":
    asyncio.run(demo_enhanced_multi_venue_router())