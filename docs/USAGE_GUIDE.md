# 📖 DipMaster Trading System - Usage Guide

Complete usage guide for the DipMaster Trading System, covering everything from basic setup to advanced trading operations.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Gyarados4157/DipMaster-Trading-System.git
cd DipMaster-Trading-System

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import src.core; print('✅ Installation successful')"
```

### 2. Initial Configuration

```bash
# Copy example configuration
cp config/config.json.example config/config.json

# Edit configuration (important!)
nano config/config.json
```

**Essential Configuration:**
```json
{
  "symbols": ["BTCUSDT", "ETHUSDT"],
  "paper_trading": true,
  "max_positions": 2,
  "max_position_size": 500,
  "daily_loss_limit": -100
}
```

### 3. First Run (Paper Trading)

```bash
# Start in safe paper trading mode
python main.py --paper --config config/config.json

# Watch the logs
tail -f logs/dipmaster_$(date +%Y%m%d).log
```

## 📁 Project Structure

```
DipMaster-Trading-System/
├── 🏗️ src/                     # Core source code
│   ├── core/                  # Trading engine components
│   │   ├── trading_engine.py  # Main orchestration
│   │   ├── signal_detector.py # Entry/exit signals
│   │   ├── timing_manager.py  # 15-minute boundaries
│   │   ├── position_manager.py # Position tracking
│   │   ├── order_executor.py  # Binance integration
│   │   └── websocket_client.py # Real-time data
│   ├── dashboard/             # Monitoring interface
│   ├── scripts/               # Analysis tools
│   └── tools/                 # Strategy utilities
├── ⚙️ config/                  # Configuration files
├── 📊 data/                    # Market data storage
├── 📈 results/                 # Analysis results
├── 📚 docs/                    # Documentation
├── 🧪 tests/                   # Test files
├── 📝 logs/                    # System logs
├── 🐳 Dockerfile              # Container definition
├── 📋 CLAUDE.md                # Maintenance guide
└── 📖 README.md                # Project overview
```

## ⚙️ Configuration Guide

### Basic Configuration Options

**Trading Settings:**
```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "paper_trading": true,              // Start with true!
  "max_positions": 3,
  "max_position_size": 1000,          // USD per position
  "daily_loss_limit": -500            // Stop trading if hit
}
```

**Strategy Parameters:**
```json
{
  "rsi_entry_range": [30, 50],        // RSI range for entries
  "dip_threshold": 0.002,             // 0.2% dip confirmation
  "volume_multiplier": 1.5,           // Volume spike threshold
  "max_holding_minutes": 180,         // Max hold time
  "target_profit": 0.008,             // 0.8% profit target
  "min_confidence": 0.6               // Signal confidence threshold
}
```

**API Configuration (Live Trading Only):**
```json
{
  "api_key": "your_binance_api_key",
  "api_secret": "your_binance_api_secret"
}
```

### Environment Variables (Alternative)

```bash
# Create .env file for sensitive data
cat > .env << EOF
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
PAPER_TRADING=true
MAX_POSITIONS=3
EOF
```

## 🚀 Operating Modes

### 1. Paper Trading Mode (Recommended Start)

```bash
# Safe mode - no real money at risk
python main.py --paper --config config/config.json

# With specific symbols
python main.py --paper --symbols BTCUSDT,ETHUSDT

# With debug logging
python main.py --paper --log-level DEBUG
```

### 2. Live Trading Mode (After Testing)

```bash
# ⚠️ WARNING: Uses real money!
python main.py --config config/config.json

# With specific risk limits
python main.py --config config/config.json --max-positions 2 --daily-limit -200
```

### 3. Analysis Mode

```bash
# Run strategy validation
python src/scripts/core/strategy_validation.py

# Analyze historical performance
python src/scripts/core/analyze_dipmaster_trades.py

# Generate comprehensive report
python src/scripts/core/run_comprehensive_analysis.py
```

## 🤖 DipMaster AI Strategy Rules

### Entry Conditions (5-minute chart)
- ✅ **RSI Range**: 30-50 (catching dip, not extreme oversold)
- ✅ **Price Position**: Below 20-period MA (87% probability)
- ✅ **Dip Confirmation**: Buy price below open price
- ✅ **Volume Confirmation**: 1.5x average volume spike
- ✅ **Signal Confidence**: Minimum 60% confidence score

### Exit Conditions (15-minute chart)
- ✅ **Primary**: 15-minute boundary exits (100% compliance)
- ✅ **Preferred Windows**: 15-29 minutes (33.5%), 45-59 minutes (28.6%)
- ✅ **Target Profit**: Quick 0.8% gains
- ✅ **Maximum Hold**: 180 minutes hard limit
- ✅ **Time Discipline**: Strict boundary adherence

## 🛠️ Programming Guide

### 1. Using the API Programmatically

```python
import asyncio
from src.core import DipMasterTradingEngine

async def main():
    config = {
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "paper_trading": True,
        "max_positions": 2
    }
    
    engine = DipMasterTradingEngine(config)
    await engine.start()
    
    # Monitor for 1 hour
    await asyncio.sleep(3600)
    await engine.stop()

asyncio.run(main())
```

### 2. Custom Signal Detection

```python
from src.core import RealTimeSignalDetector

# Initialize detector
detector = RealTimeSignalDetector(config)

# Check for entry signals
signal = detector.detect_entry_signal("BTCUSDT", 45000.0)
if signal:
    print(f"Signal: {signal.reason} (Confidence: {signal.confidence:.2f})")

# Check for exit signals
exit_signal = detector.detect_exit_signal(
    symbol="BTCUSDT",
    position_data={"entry_time": datetime.now(), "entry_price": 44500},
    current_price=45200.0,
    is_boundary=True
)
```

### 3. Monitoring Positions

```python
from src.core import PositionManager

position_manager = PositionManager()

# Get all open positions
positions = position_manager.get_open_positions()
for pos in positions:
    print(f"{pos.symbol}: Entry ${pos.entry_price}, Current P&L: ${pos.pnl}")

# Get performance statistics
stats = position_manager.get_performance_stats()
print(f"Win Rate: {stats['win_rate']:.1f}%")
print(f"Total P&L: ${stats['total_pnl']:.2f}")
```

## ⚠️ Risk Management

### High-Risk Factors
- 🔴 **Contrarian Trading Risk**: Dip buying can fail in strong downtrends
- 🔴 **Time Sensitivity**: Must precisely monitor 15-minute boundaries
- 🔴 **Technical Dependence**: Strategy relies on indicator accuracy
- 🔴 **Market Conditions**: Works best in sideways/declining markets
- 🔴 **Capital Risk**: Real money at stake in live trading

### Risk Controls Built-in
- ✅ **Strict 15-minute exit discipline**
- ✅ **Fixed position sizing**
- ✅ **Daily loss limits**
- ✅ **Maximum concurrent positions**
- ✅ **Automatic position timeout**
- ✅ **Real-time monitoring**

## 📊 Monitoring & Dashboard

### Real-time Dashboard
```bash
# Start with dashboard (default)
python main.py --config config/config.json

# Dashboard will be available at http://localhost:8080
```

### Health Checks
```bash
# Check system health
curl http://localhost:8080/health

# Get current status
curl http://localhost:8080/status

# View performance metrics
curl http://localhost:8080/metrics
```

### Manual Analysis
```bash
# Generate comprehensive analysis
python src/scripts/core/run_comprehensive_analysis.py

# Show current results
python src/scripts/core/show_analysis_results.py

# Validate strategy performance
python src/scripts/core/strategy_validation.py
```

## 🔧 Customization

### Modifying Strategy Parameters
```json
// In config/config.json
{
  "rsi_entry_range": [25, 55],        // Adjust RSI range
  "max_holding_minutes": 120,         // Shorter max hold time
  "target_profit": 0.012,             // Higher profit target
  "min_confidence": 0.7               // Higher confidence threshold
}
```

### Adding New Trading Pairs
```json
{
  "symbols": [
    "BTCUSDT", "ETHUSDT", "BNBUSDT",
    "ADAUSDT", "SOLUSDT", "DOGEUSDT",
    "MATICUSDT", "LINKUSDT"            // Add new pairs
  ]
}
```

### Custom Signal Filters
```python
from src.core import RealTimeSignalDetector

class CustomSignalDetector(RealTimeSignalDetector):
    def detect_entry_signal(self, symbol, current_price):
        signal = super().detect_entry_signal(symbol, current_price)
        
        # Add custom filter
        if signal and self.is_market_hours_preferred():
            signal.confidence *= 1.1  # Boost confidence in preferred hours
            
        return signal
        
    def is_market_hours_preferred(self):
        # Custom logic for preferred trading hours
        from datetime import datetime
        hour = datetime.now().hour
        return 8 <= hour <= 16  # UTC hours
```

## 📚 Advanced Usage

### Docker Operations

```bash
# Build custom image with modifications
docker build -t dipmaster-custom .

# Run with custom parameters
docker run -d \
  --name dipmaster-custom \
  -e MAX_POSITIONS=5 \
  -e DAILY_LOSS_LIMIT=-1000 \
  -v $(pwd)/config:/app/config \
  dipmaster-custom

# Scale with docker-compose
docker-compose up --scale dipmaster=2
```

### Integration with External Systems

```python
# WebSocket client for real-time updates
import websocket
import json

def on_message(ws, message):
    data = json.loads(message)
    if data['event'] == 'position_opened':
        # Send alert to external system
        send_slack_notification(f"Position opened: {data['symbol']}")

ws = websocket.create_connection("ws://localhost:8080/ws")
ws.send(json.dumps({"action": "subscribe", "events": ["position_opened"]}))
```

### Performance Optimization

```python
# Custom configuration for high-frequency operation
config = {
    "symbols": ["BTCUSDT"],  # Focus on single high-volume pair
    "max_positions": 1,      # Single position for faster execution
    "min_confidence": 0.8,   # Higher confidence threshold
    "websocket_timeout": 5,  # Faster timeout
    "order_timeout": 10      # Quick order execution
}
```

## 🆘 Troubleshooting

### Common Issues

#### 1. WebSocket Connection Problems
```bash
# Check network connectivity
ping stream.binance.com

# Test WebSocket manually
curl --include \
     --no-buffer \
     --header "Connection: Upgrade" \
     --header "Upgrade: websocket" \
     --header "Sec-WebSocket-Key: SGVsbG8sIHdvcmxkIQ==" \
     --header "Sec-WebSocket-Version: 13" \
     wss://stream.binance.com:9443/ws

# Check firewall settings
sudo netstat -tlnp | grep :443
```

#### 2. API Authentication Failures
```bash
# Verify API key permissions
python -c "
from binance.client import Client
import os
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
print(client.get_account_status())
"

# Check API restrictions on Binance
# Go to: Binance -> API Management -> View restrictions
```

#### 3. Memory/Performance Issues
```bash
# Monitor resource usage
ps aux | grep python
top -p $(pgrep -f "main.py")

# Check for memory leaks in logs
grep -i "memory" logs/dipmaster_*.log

# Reduce memory usage
export PYTHONOPTIMIZE=1
python -O main.py --config config/config.json
```

#### 4. Signal Detection Problems
```bash
# Test signal detection manually
python -c "
from src.core import RealTimeSignalDetector
detector = RealTimeSignalDetector({'min_confidence': 0.5})
signal = detector.detect_entry_signal('BTCUSDT', 45000)
print(f'Signal: {signal}')
"

# Check technical indicators
python src/scripts/core/technical_analysis.py --symbol BTCUSDT --debug
```

### Debug Mode

```bash
# Enable comprehensive debugging
export LOG_LEVEL=DEBUG
export PYTHONPATH=/app/src
python -u main.py --config config/config.json --no-dashboard 2>&1 | tee debug.log

# Profile performance
python -m cProfile -o profile.stats main.py --config config/config.json
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats()"
```

### Log Analysis

```bash
# Find errors in logs
grep -E "(ERROR|CRITICAL)" logs/dipmaster_*.log

# Monitor real-time logs
tail -f logs/dipmaster_$(date +%Y%m%d).log | grep -E "(Signal|Position|Error)"

# Analyze trading performance
grep "Position closed" logs/dipmaster_*.log | awk '{print $NF}' | sort -n
```

## 📞 Support & Community

### Getting Help

1. **Check Documentation First**:
   - [README.md](../README.md) - Project overview
   - [CLAUDE.md](../CLAUDE.md) - Maintenance guide
   - [API_REFERENCE.md](API_REFERENCE.md) - API documentation
   - [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) - Deployment options

2. **Debug Steps**:
   - Enable DEBUG logging
   - Check system health endpoints
   - Review recent log entries
   - Validate configuration

3. **Issue Reporting**:
   - Include configuration (sanitized)
   - Provide log excerpts
   - Describe expected vs actual behavior
   - List system specifications

### Best Practices Summary

✅ **DO**:
- Always start with paper trading
- Monitor system resources regularly
- Keep API keys secure
- Use proper risk management
- Test thoroughly before live trading
- Monitor 15-minute boundaries closely

❌ **DON'T**:
- Trade with money you can't afford to lose
- Ignore risk limits
- Modify core strategy without testing
- Run without monitoring
- Use in strongly trending markets
- Skip the paper trading phase

---

**⚡ Remember: This strategy requires strict time discipline and comprehensive risk management!**

---

**Last Updated**: 2025-08-12  
**Usage Guide Version**: 2.0.0  
**Compatible with**: DipMaster Trading System v1.0.0