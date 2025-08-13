import logging
from typing import Dict, Optional
from binance.client import Client
from binance.exceptions import BinanceAPIException
import asyncio

logger = logging.getLogger(__name__)


class OrderExecutor:
    """Handles order execution with Binance API"""
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None, 
                 testnet: bool = False, strategy_mode: str = "optimized"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self.strategy_mode = strategy_mode
        
        # Original strategy parameters
        if strategy_mode == "original":
            self.use_market_orders_only = True  # Only use market orders
            self.leverage = 10  # 10x leverage for futures
            self.position_mode = "one-way"  # Simple one-way mode
        else:
            self.use_market_orders_only = False
            self.leverage = 1
            self.position_mode = "hedge"  # Hedge mode for complex strategies
        
        if api_key and api_secret:
            try:
                self.client = Client(api_key, api_secret, testnet=testnet)
                mode = "testnet" if testnet else "live trading"
                logger.info(f"Binance client initialized for {mode} [{strategy_mode} mode]")
                
                # Set leverage for futures if original strategy
                if strategy_mode == "original":
                    logger.info(f"Setting leverage to {self.leverage}x for original strategy")
            except Exception as e:
                logger.error(f"Failed to initialize Binance client: {e}")
                self.client = None
        else:
            logger.info("Running in paper trading mode (no API credentials)")
            
    async def place_market_order(self, symbol: str, side: str, quantity: float) -> Optional[Dict]:
        """Place a market order"""
        if not self.client:
            # Paper trading mode
            return {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': 0,  # Will be filled with current market price
                'status': 'FILLED',
                'orderId': f"PAPER_{symbol}_{side}_{quantity}"
            }
            
        try:
            # Format quantity based on symbol's lot size
            quantity = self._format_quantity(symbol, quantity)
            
            # Place order
            if side == 'BUY':
                order = self.client.order_market_buy(
                    symbol=symbol,
                    quantity=quantity
                )
            else:
                order = self.client.order_market_sell(
                    symbol=symbol,
                    quantity=quantity
                )
                
            logger.info(f"Order placed: {order}")
            
            # Get executed price
            executed_price = self._get_executed_price(order)
            
            return {
                'symbol': symbol,
                'side': side,
                'quantity': float(order['executedQty']),
                'price': executed_price,
                'status': order['status'],
                'orderId': order['orderId']
            }
            
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            return None
        except Exception as e:
            logger.error(f"Order execution error: {e}")
            return None
            
    async def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Optional[Dict]:
        """Place a limit order"""
        if not self.client:
            return {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'status': 'NEW',
                'orderId': f"PAPER_{symbol}_{side}_{quantity}_{price}"
            }
            
        try:
            quantity = self._format_quantity(symbol, quantity)
            price = self._format_price(symbol, price)
            
            if side == 'BUY':
                order = self.client.order_limit_buy(
                    symbol=symbol,
                    quantity=quantity,
                    price=price
                )
            else:
                order = self.client.order_limit_sell(
                    symbol=symbol,
                    quantity=quantity,
                    price=price
                )
                
            logger.info(f"Limit order placed: {order}")
            
            return {
                'symbol': symbol,
                'side': side,
                'quantity': float(order['origQty']),
                'price': float(order['price']),
                'status': order['status'],
                'orderId': order['orderId']
            }
            
        except Exception as e:
            logger.error(f"Limit order error: {e}")
            return None
            
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order"""
        if not self.client:
            return True  # Paper trading always succeeds
            
        try:
            result = self.client.cancel_order(
                symbol=symbol,
                orderId=order_id
            )
            logger.info(f"Order cancelled: {result}")
            return True
        except Exception as e:
            logger.error(f"Cancel order error: {e}")
            return False
            
    def get_order_status(self, symbol: str, order_id: str) -> Optional[str]:
        """Get order status"""
        if not self.client:
            return 'FILLED'  # Paper trading assumes instant fill
            
        try:
            order = self.client.get_order(
                symbol=symbol,
                orderId=order_id
            )
            return order['status']
        except Exception as e:
            logger.error(f"Get order status error: {e}")
            return None
            
    def _format_quantity(self, symbol: str, quantity: float) -> float:
        """Format quantity based on symbol's lot size"""
        # Get symbol info
        if not self.client:
            return round(quantity, 8)
            
        try:
            info = self.client.get_symbol_info(symbol)
            for filter in info['filters']:
                if filter['filterType'] == 'LOT_SIZE':
                    step_size = float(filter['stepSize'])
                    precision = len(str(step_size).split('.')[-1].rstrip('0'))
                    return round(quantity - (quantity % step_size), precision)
        except:
            pass
            
        return round(quantity, 8)
        
    def _format_price(self, symbol: str, price: float) -> float:
        """Format price based on symbol's tick size"""
        if not self.client:
            return round(price, 8)
            
        try:
            info = self.client.get_symbol_info(symbol)
            for filter in info['filters']:
                if filter['filterType'] == 'PRICE_FILTER':
                    tick_size = float(filter['tickSize'])
                    precision = len(str(tick_size).split('.')[-1].rstrip('0'))
                    return round(price - (price % tick_size), precision)
        except:
            pass
            
        return round(price, 8)
        
    def _get_executed_price(self, order: Dict) -> float:
        """Calculate average executed price from order fills"""
        try:
            if 'fills' in order:
                total_qty = 0
                total_value = 0
                for fill in order['fills']:
                    qty = float(fill['qty'])
                    price = float(fill['price'])
                    total_qty += qty
                    total_value += qty * price
                    
                if total_qty > 0:
                    return total_value / total_qty
                    
            # Fallback to order price if available
            if 'price' in order and float(order['price']) > 0:
                return float(order['price'])
                
            return 0
            
        except Exception as e:
            logger.error(f"Error calculating executed price: {e}")
            return 0
            
    async def execute_original_entry(self, symbol: str, size_usd: float = 1000) -> Optional[Dict]:
        """
        Execute entry for original strategy - simplified market buy with leverage
        Fixed position size, 10x leverage, no stop loss
        """
        if self.strategy_mode != "original":
            return await self.place_market_order(symbol, 'BUY', size_usd)
            
        # Get current price to calculate quantity
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol) if self.client else {'price': 0}
            current_price = float(ticker['price']) if ticker else 0
            
            if current_price <= 0:
                logger.error(f"Invalid price for {symbol}: {current_price}")
                return None
                
            # Calculate quantity based on USD size
            quantity = size_usd / current_price
            
            # Execute market buy
            order_result = await self.place_market_order(symbol, 'BUY', quantity)
            
            if order_result:
                order_result['leverage'] = self.leverage
                order_result['position_size_usd'] = size_usd
                order_result['effective_size_usd'] = size_usd * self.leverage
                logger.info(f"Original strategy entry: {symbol} ${size_usd} x{self.leverage}")
                
            return order_result
            
        except Exception as e:
            logger.error(f"Original strategy entry error: {e}")
            return None
            
    async def execute_original_exit(self, symbol: str, quantity: float) -> Optional[Dict]:
        """
        Execute exit for original strategy - simple market sell
        No stop loss, no take profit, just time-based exit
        """
        if self.strategy_mode != "original":
            return await self.place_market_order(symbol, 'SELL', quantity)
            
        # Execute market sell
        order_result = await self.place_market_order(symbol, 'SELL', quantity)
        
        if order_result:
            logger.info(f"Original strategy exit: {symbol} qty={quantity}")
            
        return order_result
        
    def set_symbol_leverage(self, symbol: str, leverage: int = 10):
        """Set leverage for a specific symbol (futures trading)"""
        if not self.client:
            logger.info(f"Paper trading: Would set {symbol} leverage to {leverage}x")
            return True
            
        try:
            # This is for Binance Futures API
            # Note: Actual implementation depends on whether using USDT-M or COIN-M futures
            response = self.client.futures_change_leverage(
                symbol=symbol,
                leverage=leverage
            )
            logger.info(f"Set {symbol} leverage to {leverage}x: {response}")
            return True
        except Exception as e:
            logger.error(f"Failed to set leverage for {symbol}: {e}")
            return False