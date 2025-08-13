import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import json

from .websocket_client import MultiStreamManager
from .timing_manager import BoundaryTimingManager, BoundaryEvent
from .signal_detector import RealTimeSignalDetector, TradingSignal, SignalType
from .position_manager import PositionManager
from .order_executor import OrderExecutor

logger = logging.getLogger(__name__)


class DipMasterTradingEngine:
    """Main trading engine coordinating all components"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Handle both flat and nested config structures
        trading_config = config.get('trading', {})
        api_config = config.get('api', {})
        
        self.symbols = trading_config.get('symbols', config.get('symbols', []))
        self.running = False
        
        # Initialize components
        self.stream_manager = MultiStreamManager(self.symbols)
        self.timing_manager = BoundaryTimingManager()
        self.signal_detector = RealTimeSignalDetector(config)
        self.position_manager = PositionManager()
        self.order_executor = OrderExecutor(
            api_config.get('api_key', config.get('api_key')), 
            api_config.get('api_secret', config.get('api_secret')),
            testnet=api_config.get('testnet', config.get('testnet', False))
        )
        
        # State tracking
        self.current_prices = {}
        self.pending_signals = []
        self.last_boundary_check = None
        
        # Performance tracking
        self.signal_history = []
        self.trade_history = []
        
        # Risk management
        self.max_positions = trading_config.get('max_positions', config.get('max_positions', 3))
        self.max_position_size = trading_config.get('max_position_size', config.get('max_position_size', 1000))  # USD
        self.daily_loss_limit = trading_config.get('daily_loss_limit', config.get('daily_loss_limit', -500))  # USD
        
    async def start(self):
        """Start the trading engine"""
        logger.info("🚀 Starting DipMaster Trading Engine")
        self.running = True
        
        # Load historical positions
        self.position_manager.load_positions()
        
        # Setup callbacks
        self.stream_manager.add_callback(self.on_price_update)
        self.timing_manager.add_boundary_callback(self.on_boundary_event)
        
        # Start all components
        tasks = [
            asyncio.create_task(self.stream_manager.start()),
            asyncio.create_task(self.timing_manager.start()),
            asyncio.create_task(self.main_loop())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            logger.info("Shutting down trading engine...")
            await self.stop()
            
    async def stop(self):
        """Stop the trading engine"""
        self.running = False
        
        # Close all open positions
        for position in self.position_manager.get_open_positions():
            current_price = self.current_prices.get(position.symbol, position.entry_price)
            await self.close_position(position.id, current_price, "Engine shutdown")
            
        # Stop components
        await self.stream_manager.stop()
        await self.timing_manager.stop()
        
        # Save final state
        self.save_state()
        
        logger.info("Trading engine stopped")
        
    async def main_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Check risk limits
                if not self.check_risk_limits():
                    logger.warning("Risk limits exceeded, pausing trading")
                    await asyncio.sleep(60)
                    continue
                    
                # Process pending signals
                await self.process_pending_signals()
                
                # Monitor existing positions
                await self.monitor_positions()
                
                # Log status
                if datetime.now().second == 0:
                    self.log_status()
                    
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)
                
    async def on_price_update(self, data: Dict):
        """Handle incoming price data"""
        symbol = data['symbol']
        self.current_prices[symbol] = data['close']
        
        # Update signal detector
        self.signal_detector.update_price_data(symbol, data)
        
        # Check for entry signals if no position
        if not self.position_manager.get_position_by_symbol(symbol):
            if len(self.position_manager.positions) < self.max_positions:
                signal = self.signal_detector.detect_entry_signal(symbol, data['close'])
                if signal:
                    await self.handle_entry_signal(signal)
                    
    async def on_boundary_event(self, event: BoundaryEvent):
        """Handle 15-minute boundary events"""
        logger.info(f"⏰ Boundary event: {event.slot.name} at {event.timestamp}")
        self.last_boundary_check = event.timestamp
        
        # Check all positions for boundary exits
        for position in list(self.position_manager.get_open_positions()):
            current_price = self.current_prices.get(position.symbol, position.entry_price)
            
            # Detect exit signal with boundary flag
            signal = self.signal_detector.detect_exit_signal(
                position.symbol, 
                {
                    'entry_time': position.entry_time,
                    'entry_price': position.entry_price
                },
                current_price,
                is_boundary=True
            )
            
            if signal:
                await self.handle_exit_signal(signal, position.id)
                
    async def handle_entry_signal(self, signal: TradingSignal):
        """Handle entry signal"""
        logger.info(f"📊 Entry signal: {signal.symbol} - {signal.reason} (Confidence: {signal.confidence:.2f})")
        
        # Calculate position size
        position_size = self.calculate_position_size(signal)
        
        if self.config.get('paper_trading', True):
            # Paper trading
            quantity = position_size / signal.price
            position = self.position_manager.create_position(
                signal.symbol,
                signal.price,
                quantity,
                signal.reason,
                signal.indicators
            )
            logger.info(f"📝 Paper trade opened: {position.id}")
        else:
            # Live trading
            order = await self.order_executor.place_market_order(
                signal.symbol,
                'BUY',
                position_size / signal.price
            )
            
            if order:
                position = self.position_manager.create_position(
                    signal.symbol,
                    order['price'],
                    order['quantity'],
                    signal.reason,
                    signal.indicators
                )
                logger.info(f"💰 Live trade opened: {position.id}")
                
        # Record signal
        self.signal_history.append({
            'timestamp': signal.timestamp,
            'signal': signal,
            'executed': True
        })
        
    async def handle_exit_signal(self, signal: TradingSignal, position_id: str):
        """Handle exit signal"""
        logger.info(f"📊 Exit signal: {signal.symbol} - {signal.reason}")
        
        position = self.position_manager.positions.get(position_id)
        if not position:
            return
            
        if self.config.get('paper_trading', True):
            # Paper trading
            closed_position = self.position_manager.close_position(
                position_id,
                signal.price,
                signal.reason,
                signal.indicators
            )
            
            if closed_position:
                logger.info(f"📝 Paper trade closed: {position_id} - PnL: ${closed_position.pnl:.2f}")
        else:
            # Live trading
            order = await self.order_executor.place_market_order(
                signal.symbol,
                'SELL',
                position.quantity
            )
            
            if order:
                closed_position = self.position_manager.close_position(
                    position_id,
                    order['price'],
                    signal.reason,
                    signal.indicators
                )
                
                if closed_position:
                    logger.info(f"💰 Live trade closed: {position_id} - PnL: ${closed_position.pnl:.2f}")
                    
    async def monitor_positions(self):
        """Monitor open positions for exit conditions"""
        for position in list(self.position_manager.get_open_positions()):
            current_price = self.current_prices.get(position.symbol, position.entry_price)
            
            # Skip if we just checked at boundary
            if self.last_boundary_check and (datetime.now() - self.last_boundary_check).seconds < 5:
                continue
                
            # Check for non-boundary exit signals
            signal = self.signal_detector.detect_exit_signal(
                position.symbol,
                {
                    'entry_time': position.entry_time,
                    'entry_price': position.entry_price
                },
                current_price,
                is_boundary=False
            )
            
            if signal and signal.signal_type in [SignalType.EXIT_TARGET, SignalType.EXIT_TIMEOUT]:
                await self.handle_exit_signal(signal, position.id)
                
    async def process_pending_signals(self):
        """Process any pending signals"""
        # Implement if needed for signal queueing
        pass
        
    def calculate_position_size(self, signal: TradingSignal) -> float:
        """Calculate position size based on signal and risk management"""
        base_size = self.max_position_size
        
        # Adjust based on confidence
        size = base_size * signal.confidence
        
        # Adjust based on daily performance
        stats = self.position_manager.get_performance_stats()
        if stats['total_pnl'] < self.daily_loss_limit / 2:
            size *= 0.5  # Reduce size if approaching daily loss limit
            
        return min(size, self.max_position_size)
        
    def check_risk_limits(self) -> bool:
        """Check if risk limits are breached"""
        stats = self.position_manager.get_performance_stats()
        
        # Check daily loss limit
        if stats['total_pnl'] <= self.daily_loss_limit:
            logger.error(f"Daily loss limit reached: ${stats['total_pnl']:.2f}")
            return False
            
        # Check max positions
        if len(self.position_manager.positions) >= self.max_positions:
            logger.debug("Max positions limit reached")
            return True  # Not an error, just at capacity
            
        return True
        
    def log_status(self):
        """Log current status"""
        stats = self.position_manager.get_performance_stats()
        exposure = self.position_manager.calculate_exposure()
        
        logger.info(f"""
        === Trading Status ===
        Open Positions: {exposure['total_positions']}
        Total Exposure: ${exposure['total_value']:.2f}
        Today's PnL: ${stats['total_pnl']:.2f}
        Win Rate: {stats['win_rate']:.1f}%
        Total Trades: {stats['total_trades']}
        ===================
        """)
        
    def save_state(self):
        """Save engine state to file"""
        state = {
            'timestamp': datetime.now().isoformat(),
            'performance': self.position_manager.get_performance_stats(),
            'signal_count': len(self.signal_history),
            'config': self.config
        }
        
        state_file = Path('data/engine_state.json')
        state_file.parent.mkdir(exist_ok=True)
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)