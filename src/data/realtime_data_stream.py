"""
实时数据流处理器 - Real-time Data Stream Processor
为DipMaster Trading System提供毫秒级实时数据流处理

Features:
- WebSocket实时数据流 (Binance, OKX, Bybit)
- 内存缓存和Redis分布式缓存
- 数据预处理和质量监控
- 事件驱动架构
- 低延迟数据分发
"""

import asyncio
import websockets
import json
import time
from datetime import datetime, timezone
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional
import logging
from dataclasses import dataclass, asdict
import aioredis
import zmq
import zmq.asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import traceback
import gzip
import zlib
from collections import defaultdict, deque
import statistics
from urllib.parse import urljoin

@dataclass
class MarketTick:
    """市场tick数据"""
    symbol: str
    timestamp: float
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    exchange: str
    
@dataclass
class OrderBookSnapshot:
    """订单簿快照"""
    symbol: str
    timestamp: float
    bids: List[List[float]]  # [[price, size], ...]
    asks: List[List[float]]
    exchange: str
    
@dataclass
class KlineData:
    """K线数据"""
    symbol: str
    timestamp: float
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    exchange: str

class DataStreamManager:
    """数据流管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.StreamManager")
        
        # 连接状态
        self.connections = {}
        self.connection_status = {}
        
        # 数据处理器
        self.data_handlers = defaultdict(list)
        
        # 缓存
        self.memory_cache = defaultdict(deque)
        self.cache_max_size = self.config.get('cache_max_size', 10000)
        
        # Redis连接
        self.redis_client = None
        
        # ZMQ发布器
        self.zmq_context = None
        self.zmq_publisher = None
        
        # 性能监控
        self.performance_stats = {
            'messages_received': 0,
            'messages_processed': 0,
            'latency_samples': deque(maxlen=1000),
            'error_count': 0
        }
        
        # 运行状态
        self.running = False
        self.reconnect_attempts = {}
        
    async def initialize(self):
        """初始化连接"""
        self.logger.info("初始化实时数据流管理器...")
        
        # 初始化Redis
        if self.config.get('redis_enabled', True):
            try:
                redis_config = self.config.get('redis', {})
                self.redis_client = await aioredis.from_url(
                    redis_config.get('url', 'redis://localhost:6379'),
                    password=redis_config.get('password'),
                    db=redis_config.get('db', 0)
                )
                await self.redis_client.ping()
                self.logger.info("Redis连接成功")
            except Exception as e:
                self.logger.warning(f"Redis连接失败: {e}")
                
        # 初始化ZMQ
        if self.config.get('zmq_enabled', True):
            try:
                self.zmq_context = zmq.asyncio.Context()
                self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
                zmq_port = self.config.get('zmq_port', 5555)
                self.zmq_publisher.bind(f"tcp://*:{zmq_port}")
                self.logger.info(f"ZMQ发布器启动: 端口 {zmq_port}")
            except Exception as e:
                self.logger.warning(f"ZMQ初始化失败: {e}")
                
    async def start_streams(self, symbols: List[str], exchanges: List[str] = None):
        """启动数据流"""
        if exchanges is None:
            exchanges = ['binance']
            
        self.running = True
        self.logger.info(f"启动数据流: {symbols} on {exchanges}")
        
        # 启动各个交易所的数据流
        tasks = []
        for exchange in exchanges:
            if exchange == 'binance':
                task = asyncio.create_task(self._start_binance_streams(symbols))
                tasks.append(task)
            elif exchange == 'okx':
                task = asyncio.create_task(self._start_okx_streams(symbols))
                tasks.append(task)
                
        # 启动性能监控
        monitor_task = asyncio.create_task(self._monitor_performance())
        tasks.append(monitor_task)
        
        # 启动重连监控
        reconnect_task = asyncio.create_task(self._monitor_reconnections())
        tasks.append(reconnect_task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _start_binance_streams(self, symbols: List[str]):
        """启动Binance数据流"""
        streams = []
        
        # 构建流名称
        for symbol in symbols:
            symbol_lower = symbol.lower()
            streams.extend([
                f"{symbol_lower}@ticker",      # 24hr ticker
                f"{symbol_lower}@trade",       # 成交数据
                f"{symbol_lower}@depth20@100ms", # 订单簿
                f"{symbol_lower}@kline_1m",    # 1分钟K线
                f"{symbol_lower}@kline_5m",    # 5分钟K线
            ])
        
        stream_url = f"wss://stream.binance.com:9443/ws/{'/'.join(streams)}"
        
        while self.running:
            try:
                self.logger.info(f"连接Binance WebSocket: {len(streams)} 个流")
                
                async with websockets.connect(stream_url) as websocket:
                    self.connections['binance'] = websocket
                    self.connection_status['binance'] = 'connected'
                    self.reconnect_attempts['binance'] = 0
                    
                    async for message in websocket:
                        if not self.running:
                            break
                            
                        try:
                            data = json.loads(message)
                            await self._process_binance_message(data)
                            
                        except Exception as e:
                            self.logger.error(f"处理Binance消息失败: {e}")
                            self.performance_stats['error_count'] += 1
                            
            except Exception as e:
                self.connection_status['binance'] = 'disconnected'
                self.reconnect_attempts['binance'] = self.reconnect_attempts.get('binance', 0) + 1
                
                self.logger.error(f"Binance WebSocket连接失败: {e}")
                
                # 重连延迟
                delay = min(30, 2 ** self.reconnect_attempts['binance'])
                self.logger.info(f"Binance {delay}秒后重连...")
                await asyncio.sleep(delay)
                
    async def _process_binance_message(self, data: Dict[str, Any]):
        """处理Binance消息"""
        start_time = time.time()
        
        try:
            if 'stream' in data and 'data' in data:
                stream_name = data['stream']
                stream_data = data['data']
                
                # 解析流类型
                if '@ticker' in stream_name:
                    await self._handle_ticker_data(stream_data, 'binance')
                elif '@trade' in stream_name:
                    await self._handle_trade_data(stream_data, 'binance')
                elif '@depth' in stream_name:
                    await self._handle_depth_data(stream_data, 'binance')
                elif '@kline' in stream_name:
                    await self._handle_kline_data(stream_data, 'binance')
                    
                # 更新性能统计
                processing_time = (time.time() - start_time) * 1000
                self.performance_stats['latency_samples'].append(processing_time)
                self.performance_stats['messages_processed'] += 1
                
            self.performance_stats['messages_received'] += 1
            
        except Exception as e:
            self.logger.error(f"处理Binance消息错误: {e}")
            self.performance_stats['error_count'] += 1
            
    async def _handle_ticker_data(self, data: Dict[str, Any], exchange: str):
        """处理ticker数据"""
        symbol = data.get('s', '').upper()
        
        tick_data = {
            'symbol': symbol,
            'price': float(data.get('c', 0)),
            'volume': float(data.get('v', 0)),
            'price_change': float(data.get('P', 0)),
            'timestamp': float(data.get('E', 0)) / 1000,
            'exchange': exchange,
            'type': 'ticker'
        }
        
        await self._distribute_data('ticker', tick_data)
        
    async def _handle_trade_data(self, data: Dict[str, Any], exchange: str):
        """处理成交数据"""
        symbol = data.get('s', '').upper()
        
        trade_data = MarketTick(
            symbol=symbol,
            timestamp=float(data.get('T', 0)) / 1000,
            price=float(data.get('p', 0)),
            volume=float(data.get('q', 0)),
            side='buy' if data.get('m', False) else 'sell',
            exchange=exchange
        )
        
        await self._distribute_data('trade', asdict(trade_data))
        
    async def _handle_depth_data(self, data: Dict[str, Any], exchange: str):
        """处理订单簿数据"""
        symbol = data.get('s', '').upper()
        
        orderbook_data = OrderBookSnapshot(
            symbol=symbol,
            timestamp=float(data.get('E', 0)) / 1000,
            bids=[[float(bid[0]), float(bid[1])] for bid in data.get('bids', [])],
            asks=[[float(ask[0]), float(ask[1])] for ask in data.get('asks', [])],
            exchange=exchange
        )
        
        await self._distribute_data('orderbook', asdict(orderbook_data))
        
    async def _handle_kline_data(self, data: Dict[str, Any], exchange: str):
        """处理K线数据"""
        kline_info = data.get('k', {})
        symbol = kline_info.get('s', '').upper()
        
        kline_data = KlineData(
            symbol=symbol,
            timestamp=float(kline_info.get('t', 0)) / 1000,
            timeframe=kline_info.get('i', ''),
            open=float(kline_info.get('o', 0)),
            high=float(kline_info.get('h', 0)),
            low=float(kline_info.get('l', 0)),
            close=float(kline_info.get('c', 0)),
            volume=float(kline_info.get('v', 0)),
            exchange=exchange
        )
        
        await self._distribute_data('kline', asdict(kline_data))
        
    async def _distribute_data(self, data_type: str, data: Dict[str, Any]):
        """分发数据"""
        # 1. 存储到内存缓存
        cache_key = f"{data_type}_{data.get('symbol', 'unknown')}"
        self.memory_cache[cache_key].append(data)
        
        # 限制缓存大小
        if len(self.memory_cache[cache_key]) > self.cache_max_size:
            self.memory_cache[cache_key].popleft()
            
        # 2. 发布到Redis
        if self.redis_client:
            try:
                await self.redis_client.publish(f"market_data:{data_type}", json.dumps(data))
            except Exception as e:
                self.logger.error(f"Redis发布失败: {e}")
                
        # 3. 发布到ZMQ
        if self.zmq_publisher:
            try:
                topic = f"{data_type}.{data.get('symbol', 'unknown')}"
                message = f"{topic} {json.dumps(data)}"
                await self.zmq_publisher.send_string(message, zmq.NOBLOCK)
            except Exception as e:
                self.logger.error(f"ZMQ发布失败: {e}")
                
        # 4. 调用注册的处理器
        for handler in self.data_handlers[data_type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                self.logger.error(f"数据处理器错误: {e}")
                
    def register_handler(self, data_type: str, handler: Callable):
        """注册数据处理器"""
        self.data_handlers[data_type].append(handler)
        self.logger.info(f"注册处理器: {data_type}")
        
    def get_cached_data(self, data_type: str, symbol: str, limit: int = 100) -> List[Dict]:
        """获取缓存数据"""
        cache_key = f"{data_type}_{symbol}"
        cached = list(self.memory_cache.get(cache_key, []))
        return cached[-limit:] if cached else []
        
    async def _monitor_performance(self):
        """性能监控"""
        while self.running:
            try:
                await asyncio.sleep(30)  # 每30秒报告一次
                
                stats = self.performance_stats
                latency_samples = list(stats['latency_samples'])
                
                if latency_samples:
                    avg_latency = statistics.mean(latency_samples)
                    p95_latency = np.percentile(latency_samples, 95)
                    p99_latency = np.percentile(latency_samples, 99)
                else:
                    avg_latency = p95_latency = p99_latency = 0
                    
                self.logger.info(
                    f"性能统计 - "
                    f"收到: {stats['messages_received']}, "
                    f"处理: {stats['messages_processed']}, "
                    f"错误: {stats['error_count']}, "
                    f"延迟(ms) - 平均: {avg_latency:.2f}, "
                    f"P95: {p95_latency:.2f}, "
                    f"P99: {p99_latency:.2f}"
                )
                
                # 发布性能指标
                if self.redis_client:
                    performance_data = {
                        'timestamp': datetime.now(timezone.utc).isoformat(),
                        'messages_received': stats['messages_received'],
                        'messages_processed': stats['messages_processed'],
                        'error_count': stats['error_count'],
                        'avg_latency_ms': avg_latency,
                        'p95_latency_ms': p95_latency,
                        'p99_latency_ms': p99_latency,
                        'connections': {k: v for k, v in self.connection_status.items()}
                    }
                    
                    await self.redis_client.set(
                        'stream_performance',
                        json.dumps(performance_data),
                        ex=300  # 5分钟过期
                    )
                    
            except Exception as e:
                self.logger.error(f"性能监控错误: {e}")
                
    async def _monitor_reconnections(self):
        """重连监控"""
        while self.running:
            try:
                await asyncio.sleep(10)
                
                for exchange, status in self.connection_status.items():
                    if status == 'disconnected':
                        attempts = self.reconnect_attempts.get(exchange, 0)
                        self.logger.warning(f"{exchange} 连接断开，重连尝试: {attempts}")
                        
                        # 如果重连次数过多，发送警报
                        if attempts > 5:
                            await self._send_alert(f"{exchange} 重连失败次数过多: {attempts}")
                            
            except Exception as e:
                self.logger.error(f"重连监控错误: {e}")
                
    async def _send_alert(self, message: str):
        """发送警报"""
        alert_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': 'error',
            'message': message,
            'component': 'data_stream'
        }
        
        if self.redis_client:
            await self.redis_client.publish('alerts', json.dumps(alert_data))
            
    async def stop_streams(self):
        """停止数据流"""
        self.logger.info("停止实时数据流...")
        self.running = False
        
        # 关闭WebSocket连接
        for connection in self.connections.values():
            if connection:
                await connection.close()
                
        # 关闭ZMQ
        if self.zmq_publisher:
            self.zmq_publisher.close()
        if self.zmq_context:
            self.zmq_context.term()
            
        # 关闭Redis
        if self.redis_client:
            await self.redis_client.close()
            
    def get_stream_status(self) -> Dict[str, Any]:
        """获取流状态"""
        return {
            'running': self.running,
            'connections': dict(self.connection_status),
            'reconnect_attempts': dict(self.reconnect_attempts),
            'performance': dict(self.performance_stats),
            'cache_sizes': {k: len(v) for k, v in self.memory_cache.items()}
        }

class RealTimeDataProcessor:
    """实时数据处理器"""
    
    def __init__(self, stream_manager: DataStreamManager):
        self.stream_manager = stream_manager
        self.logger = logging.getLogger(f"{__name__}.DataProcessor")
        
        # 数据缓冲区
        self.buffers = defaultdict(list)
        self.buffer_lock = threading.Lock()
        
        # 注册处理器
        self._register_handlers()
        
    def _register_handlers(self):
        """注册数据处理器"""
        self.stream_manager.register_handler('ticker', self._process_ticker)
        self.stream_manager.register_handler('trade', self._process_trade)
        self.stream_manager.register_handler('orderbook', self._process_orderbook)
        self.stream_manager.register_handler('kline', self._process_kline)
        
    async def _process_ticker(self, data: Dict[str, Any]):
        """处理ticker数据"""
        symbol = data.get('symbol')
        if not symbol:
            return
            
        # 计算技术指标
        price = data.get('price', 0)
        volume = data.get('volume', 0)
        
        # 更新最新价格缓存
        with self.buffer_lock:
            self.buffers[f"{symbol}_price"].append({
                'timestamp': data.get('timestamp'),
                'price': price,
                'volume': volume
            })
            
            # 限制缓冲区大小
            if len(self.buffers[f"{symbol}_price"]) > 1000:
                self.buffers[f"{symbol}_price"] = self.buffers[f"{symbol}_price"][-1000:]
                
    async def _process_trade(self, data: Dict[str, Any]):
        """处理成交数据"""
        symbol = data.get('symbol')
        if not symbol:
            return
            
        # 聚合成交数据
        with self.buffer_lock:
            self.buffers[f"{symbol}_trades"].append(data)
            
            # 限制缓冲区大小
            if len(self.buffers[f"{symbol}_trades"]) > 5000:
                self.buffers[f"{symbol}_trades"] = self.buffers[f"{symbol}_trades"][-5000:]
                
    async def _process_orderbook(self, data: Dict[str, Any]):
        """处理订单簿数据"""
        symbol = data.get('symbol')
        if not symbol:
            return
            
        # 计算订单簿指标
        bids = data.get('bids', [])
        asks = data.get('asks', [])
        
        if bids and asks:
            best_bid = bids[0][0] if bids[0] else 0
            best_ask = asks[0][0] if asks[0] else 0
            spread = best_ask - best_bid if best_ask > best_bid else 0
            spread_pct = (spread / best_ask * 100) if best_ask > 0 else 0
            
            orderbook_metrics = {
                'symbol': symbol,
                'timestamp': data.get('timestamp'),
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_pct': spread_pct,
                'bid_depth': sum(bid[1] for bid in bids[:10]),
                'ask_depth': sum(ask[1] for ask in asks[:10])
            }
            
            # 更新订单簿指标缓存
            with self.buffer_lock:
                self.buffers[f"{symbol}_orderbook"].append(orderbook_metrics)
                
                if len(self.buffers[f"{symbol}_orderbook"]) > 1000:
                    self.buffers[f"{symbol}_orderbook"] = self.buffers[f"{symbol}_orderbook"][-1000:]
                    
    async def _process_kline(self, data: Dict[str, Any]):
        """处理K线数据"""
        symbol = data.get('symbol')
        timeframe = data.get('timeframe')
        
        if not symbol or not timeframe:
            return
            
        # 计算技术指标
        ohlc_data = {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': data.get('timestamp'),
            'open': data.get('open'),
            'high': data.get('high'),
            'low': data.get('low'),
            'close': data.get('close'),
            'volume': data.get('volume')
        }
        
        # 更新K线缓存
        key = f"{symbol}_{timeframe}_klines"
        with self.buffer_lock:
            self.buffers[key].append(ohlc_data)
            
            if len(self.buffers[key]) > 500:
                self.buffers[key] = self.buffers[key][-500:]
                
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """获取最新价格"""
        with self.buffer_lock:
            price_data = self.buffers.get(f"{symbol}_price", [])
            if price_data:
                return price_data[-1].get('price')
        return None
        
    def get_latest_orderbook_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取最新订单簿指标"""
        with self.buffer_lock:
            orderbook_data = self.buffers.get(f"{symbol}_orderbook", [])
            if orderbook_data:
                return orderbook_data[-1]
        return None
        
    def get_recent_klines(self, symbol: str, timeframe: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近K线数据"""
        key = f"{symbol}_{timeframe}_klines"
        with self.buffer_lock:
            klines = self.buffers.get(key, [])
            return klines[-limit:] if klines else []

# 使用示例
async def main():
    """主函数示例"""
    # 配置
    config = {
        'redis_enabled': True,
        'redis': {
            'url': 'redis://localhost:6379',
            'db': 0
        },
        'zmq_enabled': True,
        'zmq_port': 5555,
        'cache_max_size': 10000
    }
    
    # 创建流管理器
    stream_manager = DataStreamManager(config)
    
    # 创建数据处理器
    processor = RealTimeDataProcessor(stream_manager)
    
    # 初始化
    await stream_manager.initialize()
    
    # 启动数据流
    symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT']
    
    try:
        await stream_manager.start_streams(symbols, ['binance'])
    except KeyboardInterrupt:
        print("正在停止数据流...")
    finally:
        await stream_manager.stop_streams()

if __name__ == "__main__":
    asyncio.run(main())