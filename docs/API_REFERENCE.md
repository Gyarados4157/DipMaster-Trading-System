# 🔧 API Reference - DipMaster Trading System

## Overview

The DipMaster Trading System provides a comprehensive API for programmatic access to all trading functionality, monitoring, and configuration management.

## Core Components API

### 1. Trading Engine API

#### `DipMasterTradingEngine`

Main orchestration class for the entire trading system.

```python
from src.core import DipMasterTradingEngine

# Initialize engine
config = {
    "symbols": ["BTCUSDT", "ETHUSDT"],
    "paper_trading": True,
    "api_key": "your_api_key",
    "api_secret": "your_api_secret"
}
engine = DipMasterTradingEngine(config)

# Start trading
await engine.start()

# Stop trading
await engine.stop()
```

**Methods:**

- `__init__(config: Dict)` - Initialize with configuration
- `async start()` - Start the trading engine
- `async stop()` - Stop the trading engine safely
- `get_status()` - Get current engine status
- `get_performance_stats()` - Get trading performance statistics

### 2. Signal Detection API

#### `RealTimeSignalDetector`

Handles entry and exit signal detection based on DipMaster strategy.

```python
from src.core import RealTimeSignalDetector

detector = RealTimeSignalDetector(config)

# Detect entry signal
signal = detector.detect_entry_signal(symbol="BTCUSDT", current_price=45000.0)
if signal:
    print(f"Entry signal: {signal.reason} (Confidence: {signal.confidence})")

# Detect exit signal
exit_signal = detector.detect_exit_signal(
    symbol="BTCUSDT",
    position_data={"entry_time": datetime.now(), "entry_price": 44500.0},
    current_price=45200.0,
    is_boundary=True
)
```

**Methods:**

- `detect_entry_signal(symbol: str, current_price: float) -> TradingSignal`
- `detect_exit_signal(symbol: str, position_data: Dict, current_price: float, is_boundary: bool) -> TradingSignal`
- `update_price_data(symbol: str, data: Dict)` - Update real-time price data
- `calculate_rsi(symbol: str) -> float` - Calculate RSI for symbol
- `calculate_volume_ratio(symbol: str) -> float` - Calculate volume ratio

#### `TradingSignal` Object

```python
class TradingSignal:
    symbol: str
    signal_type: SignalType  # ENTRY_DIP, EXIT_BOUNDARY, EXIT_TARGET, etc.
    price: float
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    reason: str
    indicators: Dict[str, float]
```

### 3. Position Management API

#### `PositionManager`

Manages all open positions and tracks P&L.

```python
from src.core import PositionManager

position_manager = PositionManager()

# Create position
position = position_manager.create_position(
    symbol="BTCUSDT",
    entry_price=45000.0,
    quantity=0.1,
    reason="DIP_BUY_RSI_42"
)

# Close position
closed_position = position_manager.close_position(
    position_id=position.id,
    exit_price=45800.0,
    reason="BOUNDARY_EXIT"
)

# Get performance stats
stats = position_manager.get_performance_stats()
print(f"Win Rate: {stats['win_rate']:.1f}%")
print(f"Total P&L: ${stats['total_pnl']:.2f}")
```

**Methods:**

- `create_position(symbol: str, entry_price: float, quantity: float, reason: str) -> Position`
- `close_position(position_id: str, exit_price: float, reason: str) -> Position`
- `get_open_positions() -> List[Position]`
- `get_position_by_symbol(symbol: str) -> Position`
- `get_performance_stats() -> Dict`
- `calculate_exposure() -> Dict`

### 4. Timing Management API

#### `BoundaryTimingManager`

Manages 15-minute boundary timing for exits.

```python
from src.core import BoundaryTimingManager, BoundarySlot

timing_manager = BoundaryTimingManager()

# Register callback for boundary events
async def on_boundary_event(event):
    print(f"Boundary event: {event.slot.name} at {event.timestamp}")

timing_manager.add_boundary_callback(on_boundary_event)

# Check if current time is near boundary
is_near = timing_manager.is_near_boundary(minutes_ahead=2)

# Get next boundary slot
next_slot = timing_manager.get_next_boundary_slot()
```

**Boundary Slots:**
- `SLOT_15`: 15-minute mark
- `SLOT_30`: 30-minute mark  
- `SLOT_45`: 45-minute mark
- `SLOT_60`: 60-minute mark (hour)

### 5. WebSocket Client API

#### `MultiStreamManager`

Manages real-time WebSocket connections to Binance.

```python
from src.core import MultiStreamManager

symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
stream_manager = MultiStreamManager(symbols)

# Add price update callback
async def on_price_update(data):
    print(f"{data['symbol']}: ${data['close']}")

stream_manager.add_callback(on_price_update)

# Start streaming
await stream_manager.start()

# Stop streaming
await stream_manager.stop()
```

### 6. Order Execution API

#### `OrderExecutor`

Handles order placement with Binance.

```python
from src.core import OrderExecutor

executor = OrderExecutor(api_key="your_key", api_secret="your_secret")

# Place market order
order = await executor.place_market_order(
    symbol="BTCUSDT",
    side="BUY",  # or "SELL"
    quantity=0.001
)

# Get account balance
balance = await executor.get_account_balance()
```

## Configuration API

### Configuration Schema

```python
CONFIG_SCHEMA = {
    # Trading pairs
    "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    
    # Trading mode
    "paper_trading": True,
    
    # API credentials
    "api_key": "your_binance_api_key",
    "api_secret": "your_binance_api_secret",
    
    # Risk management
    "max_positions": 3,
    "max_position_size": 1000,  # USD
    "daily_loss_limit": -500,   # USD
    
    # Strategy parameters
    "rsi_entry_range": [30, 50],
    "dip_threshold": 0.002,     # 0.2%
    "volume_multiplier": 1.5,
    "max_holding_minutes": 180,
    "target_profit": 0.008,     # 0.8%
    
    # Technical settings
    "min_confidence": 0.6,
    "enable_dashboard": True,
    "log_level": "INFO"
}
```

### Configuration Validation

```python
from config import validate_config

# Validate configuration
is_valid, errors = validate_config(config)
if not is_valid:
    print("Configuration errors:", errors)
```

## Dashboard & Monitoring API

### Dashboard Interface

```python
from src.dashboard import DashboardMonitor

# Initialize dashboard
dashboard = DashboardMonitor(engine)

# Start dashboard (runs in background)
await dashboard.start()

# Get dashboard data
data = dashboard.get_dashboard_data()
```

### Monitoring Metrics

```python
# Available metrics
metrics = {
    "positions": {
        "open_count": 2,
        "total_exposure": 1500.0,
        "unrealized_pnl": 23.45
    },
    "performance": {
        "total_trades": 156,
        "win_rate": 82.1,
        "total_pnl": 1367.35,
        "avg_holding_time": 96
    },
    "system": {
        "uptime": "2h 45m",
        "websocket_status": "connected",
        "last_signal": "5 minutes ago"
    }
}
```

## Event System API

### Event Types

```python
from src.core import EventType

# Available event types
EventType.SIGNAL_DETECTED     # New trading signal
EventType.POSITION_OPENED     # Position opened
EventType.POSITION_CLOSED     # Position closed  
EventType.BOUNDARY_REACHED    # 15-minute boundary
EventType.RISK_LIMIT_HIT      # Risk limit exceeded
EventType.WEBSOCKET_DISCONNECTED  # Connection lost
```

### Event Handlers

```python
from src.core import EventManager

event_manager = EventManager()

# Register event handler
@event_manager.on(EventType.POSITION_OPENED)
async def on_position_opened(event_data):
    position = event_data['position']
    print(f"Position opened: {position.symbol} at ${position.entry_price}")

# Emit event
await event_manager.emit(EventType.SIGNAL_DETECTED, {
    'symbol': 'BTCUSDT',
    'signal': signal_object
})
```

## Error Handling

### Exception Types

```python
from src.core import (
    TradingEngineError,
    SignalDetectionError,
    OrderExecutionError,
    WebSocketError,
    RiskManagementError
)

try:
    await engine.start()
except TradingEngineError as e:
    print(f"Engine error: {e}")
except WebSocketError as e:
    print(f"WebSocket error: {e}")
```

### Error Codes

| Code | Description | Recovery Action |
|------|-------------|----------------|
| `WS_001` | WebSocket connection failed | Automatic retry |
| `API_001` | API authentication failed | Check credentials |
| `RISK_001` | Daily loss limit exceeded | Trading paused |
| `SIG_001` | Invalid signal parameters | Signal ignored |
| `ORD_001` | Order execution failed | Retry with adjusted size |

## Utilities API

### Data Utilities

```python
from src.utils import DataProcessor

# Process market data
processor = DataProcessor()
indicators = processor.calculate_indicators(price_data)

# Format price
formatted_price = processor.format_price(45123.456789)  # "$45,123.46"

# Calculate percentage
pct_change = processor.calculate_percentage_change(45000, 45800)  # 1.78%
```

### Time Utilities

```python
from src.utils import TimeManager

time_manager = TimeManager()

# Check if market hours
is_active = time_manager.is_market_active()

# Get next boundary time
next_boundary = time_manager.get_next_boundary_time()

# Format duration
duration_str = time_manager.format_duration(timedelta(minutes=96))  # "1h 36m"
```

## REST API Endpoints

### Health Check
```
GET /health
Response: {"status": "healthy", "uptime": "2h 45m"}
```

### System Status
```
GET /status
Response: {
    "engine": "running",
    "positions": 2,
    "websocket": "connected",
    "last_update": "2025-08-12T10:30:00Z"
}
```

### Performance Metrics
```
GET /metrics
Response: {
    "total_trades": 156,
    "win_rate": 82.1,
    "total_pnl": 1367.35,
    "current_positions": 2
}
```

### Emergency Stop
```
POST /emergency-stop
Response: {"status": "stopped", "positions_closed": 2}
```

## WebSocket API

### Real-time Updates

```javascript
// Connect to WebSocket
ws = new WebSocket('ws://localhost:8080/ws');

// Subscribe to events
ws.send(JSON.stringify({
    "action": "subscribe",
    "events": ["position_updates", "signal_detected", "boundary_events"]
}));

// Handle messages
ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log(data);
};
```

### Event Messages

```json
{
    "event": "position_opened",
    "timestamp": "2025-08-12T10:30:00Z",
    "data": {
        "position_id": "pos_123456",
        "symbol": "BTCUSDT",
        "entry_price": 45000.0,
        "quantity": 0.001,
        "reason": "DIP_BUY_RSI_42"
    }
}
```

## Examples

### Complete Trading Bot

```python
import asyncio
from src.core import DipMasterTradingEngine

async def main():
    config = {
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "paper_trading": True,
        "max_positions": 2,
        "max_position_size": 500
    }
    
    engine = DipMasterTradingEngine(config)
    
    try:
        await engine.start()
        # Engine runs until manually stopped
        await asyncio.sleep(3600)  # Run for 1 hour
    except KeyboardInterrupt:
        print("Stopping engine...")
    finally:
        await engine.stop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Custom Signal Handler

```python
from src.core import RealTimeSignalDetector, SignalType

class CustomSignalHandler:
    def __init__(self, detector):
        self.detector = detector
        
    async def process_signal(self, symbol, price):
        signal = self.detector.detect_entry_signal(symbol, price)
        
        if signal and signal.signal_type == SignalType.ENTRY_DIP:
            # Custom logic for dip signals
            if signal.confidence > 0.8:
                print(f"High confidence dip signal: {symbol}")
                # Execute trade logic here
```

## Best Practices

1. **Always use paper trading first** before live trading
2. **Handle exceptions gracefully** - market conditions change rapidly
3. **Monitor resource usage** - WebSocket connections can accumulate
4. **Validate configuration** before starting the engine
5. **Use proper logging** for debugging and auditing
6. **Implement proper shutdown** handling for graceful exits
7. **Test boundary conditions** - especially around 15-minute marks

## Support

For API support and questions:
- Check the [CLAUDE.md](../CLAUDE.md) maintenance guide
- Review the [USAGE_GUIDE.md](USAGE_GUIDE.md) for practical examples
- Consult the source code in `src/core/` for implementation details

---

**Last Updated**: 2025-08-12  
**API Version**: 1.0.0  
**Compatible with**: DipMaster Trading System v1.0.0