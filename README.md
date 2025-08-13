# 🚀 DipMaster Trading System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Binance](https://img.shields.io/badge/Exchange-Binance-yellow.svg)](https://binance.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

**Professional DipMaster AI Trading System with 82.1% Win Rate**

A sophisticated cryptocurrency automatic trading system implementing the reverse-engineered DipMaster AI strategy. Features real-time WebSocket trading engine, 15-minute boundary management, and comprehensive risk controls for optimal trading performance.

## 🎯 Core Features

- **🤖 82.1% Win Rate Strategy** - Complete DipMaster AI reverse engineering
- **⚡ Real-time WebSocket Engine** - Millisecond-precision market data
- **⏰ 15-Minute Boundary Management** - 100% strict time discipline
- **🛡️ Advanced Risk Management** - Multi-layer safety controls
- **📊 Rich Monitoring Dashboard** - Real-time position tracking
- **🐳 Docker Containerized** - Easy deployment and scaling
- **📝 Paper & Live Trading** - Safe testing before live execution

## 📊 Strategy Performance

Based on comprehensive analysis of 206 historical trades:

| Metric | Value | Status |
|--------|-------|--------|
| **Win Rate** | 82.1% | 🟢 Excellent |
| **Total Profit** | $1,367.35 | 🟢 Profitable |
| **Average Hold Time** | 96 minutes | 🟢 Optimal |
| **Dip Buying Rate** | 87.9% | 🟢 Superior |
| **15-min Boundary Exits** | 100% | 🟢 Perfect |
| **Max Holding Period** | 180 minutes | 🟡 Controlled |

### 🎯 DipMaster AI Strategy

**Entry Conditions (5-minute chart):**
- RSI between 30-50 (catching the dip, not extreme oversold)
- Price below 20-period MA (87% probability)
- Buy price below open price (dip buying confirmation)
- Volume spike confirmation (1.5x average)

**Exit Conditions (15-minute chart):**
- **Primary**: 15-minute boundary exits (100% compliance)
- **Secondary**: Target profit at 0.8%
- **Fallback**: Maximum 180-minute timeout

**Preferred Exit Windows:**
- 15-29 minutes: 33.5% of trades
- 45-59 minutes: 28.6% of trades

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- 8GB+ RAM recommended
- Stable internet connection
- Binance account (for live trading)

### 1. Installation

```bash
# Clone repository
git clone https://github.com/Gyarados4157/DipMaster-Trading-System.git
cd DipMaster-Trading-System

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy example config
cp config/config.json.example config/config.json

# Edit configuration (add your API keys)
nano config/config.json
```

### 3. Paper Trading (Recommended First)

```bash
# Run in safe paper trading mode
python main.py --paper --config config/config.json
```

### 4. Live Trading (After Testing)

```bash
# Run with real money (be careful!)
python main.py --config config/config.json
```

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Build image
docker build -t dipmaster-trading .

# Run container
docker run -d \
  --name dipmaster \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  dipmaster-trading
```

### Docker Compose

```bash
# Start with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f dipmaster
```

## ⚙️ Configuration

### Basic Configuration

Edit `config/config.json`:

```json
{
  "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
  "paper_trading": true,
  "api_key": "your_binance_api_key",
  "api_secret": "your_binance_api_secret",
  "max_positions": 3,
  "max_position_size": 1000,
  "daily_loss_limit": -500
}
```

### Strategy Parameters

```json
{
  "rsi_entry_range": [30, 50],
  "dip_threshold": 0.002,
  "max_holding_minutes": 180,
  "target_profit": 0.008,
  "volume_multiplier": 1.5
}
```

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  WebSocket      │    │  Signal         │    │  Timing         │
│  Client         │────│  Detector       │────│  Manager        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Position       │    │  Trading        │    │  Order          │
│  Manager        │────│  Engine         │────│  Executor       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Dashboard      │
                    │  Monitor        │
                    └─────────────────┘
```

## 🛡️ Risk Management

### Financial Safety
- **API Permissions**: Trading only, withdrawals disabled
- **Position Limits**: Maximum $1000 per position
- **Daily Loss Limits**: Maximum $500 daily loss
- **Concurrent Positions**: Maximum 3 simultaneous trades

### Technical Safety
- **Network Resilience**: Automatic reconnection
- **Data Validation**: Real-time data integrity checks
- **Graceful Shutdown**: Safe position closing on exit
- **Comprehensive Logging**: Full audit trail

## 📚 Documentation

- **[CLAUDE.md](CLAUDE.md)** - Maintenance and development guide
- **[docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - Detailed usage instructions
- **[Strategy Guides](docs/strategy_guides/)** - Strategy implementation details
- **[Analysis Reports](docs/analysis_reports/)** - Performance analysis

## 🔍 Monitoring

### Dashboard Features
- Real-time position monitoring
- P&L tracking and analytics
- Signal detection logs
- 15-minute boundary countdown
- Risk status indicators

### Logging
- **Location**: `logs/dipmaster_YYYYMMDD.log`
- **Levels**: INFO, WARNING, ERROR
- **Rotation**: Daily with automatic cleanup

## ⚠️ Risk Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. 

- Cryptocurrency trading involves substantial risk of loss
- Past performance does not guarantee future results
- Always test thoroughly in paper trading mode first
- Never invest more than you can afford to lose
- The authors assume no responsibility for trading losses

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DipMaster AI strategy reverse engineering
- Binance API and WebSocket support
- Python asyncio ecosystem
- Technical analysis libraries

---

**Built with precision timing and real-time analysis for optimal cryptocurrency trading performance.**

