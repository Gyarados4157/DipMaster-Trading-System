import asyncio
import json
import logging
from typing import Dict, Callable, Optional, List
from datetime import datetime
import websockets
from binance import AsyncClient

logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """Real-time WebSocket client for Binance data streaming"""
    
    def __init__(self, symbols: List[str], on_data_callback: Callable):
        self.symbols = [s.lower() for s in symbols]
        self.on_data_callback = on_data_callback
        self.ws_connections = {}
        self.running = False
        self.base_url = "wss://stream.binance.com:9443/ws"
        
    async def start(self):
        """Start WebSocket connections for all symbols"""
        self.running = True
        tasks = []
        
        for symbol in self.symbols:
            stream_name = f"{symbol}@kline_1m"
            url = f"{self.base_url}/{stream_name}"
            task = asyncio.create_task(self._connect_stream(symbol, url))
            tasks.append(task)
            
        await asyncio.gather(*tasks)
        
    async def _connect_stream(self, symbol: str, url: str):
        """Connect to individual WebSocket stream"""
        while self.running:
            try:
                async with websockets.connect(url) as websocket:
                    self.ws_connections[symbol] = websocket
                    logger.info(f"Connected to {symbol} stream")
                    
                    while self.running:
                        message = await websocket.recv()
                        data = json.loads(message)
                        await self._process_message(symbol, data)
                        
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                await asyncio.sleep(5)
                
    async def _process_message(self, symbol: str, data: Dict):
        """Process incoming WebSocket message"""
        try:
            if 'k' in data:
                kline = data['k']
                processed_data = {
                    'symbol': symbol.upper(),
                    'timestamp': datetime.fromtimestamp(kline['t'] / 1000),
                    'open': float(kline['o']),
                    'high': float(kline['h']),
                    'low': float(kline['l']),
                    'close': float(kline['c']),
                    'volume': float(kline['v']),
                    'is_closed': kline['x']
                }
                
                await self.on_data_callback(processed_data)
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            
    async def stop(self):
        """Stop all WebSocket connections"""
        self.running = False
        for ws in self.ws_connections.values():
            await ws.close()
        self.ws_connections.clear()
        

class MultiStreamManager:
    """Manage multiple WebSocket streams with aggregation"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.clients = []
        self.data_buffer = {}
        self.callbacks = []
        
    def add_callback(self, callback: Callable):
        """Add callback for data processing"""
        self.callbacks.append(callback)
        
    async def on_data(self, data: Dict):
        """Handle incoming data from streams"""
        symbol = data['symbol']
        self.data_buffer[symbol] = data
        
        for callback in self.callbacks:
            await callback(data)
            
    async def start(self):
        """Start all WebSocket streams"""
        client = BinanceWebSocketClient(self.symbols, self.on_data)
        self.clients.append(client)
        await client.start()
        
    async def stop(self):
        """Stop all streams"""
        for client in self.clients:
            await client.stop()