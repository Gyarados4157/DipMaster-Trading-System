# ⚙️ Configuration Guide - DipMaster Trading System

Complete guide to configuring the DipMaster Trading System for different environments and use cases.

## 📋 Table of Contents

- [Quick Setup](#quick-setup)
- [Configuration Files](#configuration-files)
- [Environment Variables](#environment-variables)
- [Configuration Sections](#configuration-sections)
- [Environment-Specific Settings](#environment-specific-settings)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Setup

### 1. Basic Setup (Paper Trading)

```bash
# Copy example configuration
cp config/config.json.example config/config.json

# Create environment file
cp .env.example .env

# Edit .env with your settings
nano .env
```

**Essential .env settings:**
```bash
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
PAPER_TRADING=true
TESTNET=true
```

### 2. Development Setup

```bash
# Use development configuration
cp config/config.development.json config/config.json

# Set development environment
echo "ENVIRONMENT=development" >> .env
```

### 3. Production Setup

```bash
# Use production configuration  
cp config/config.production.json config/config.json

# Set production environment
echo "ENVIRONMENT=production" >> .env
echo "PAPER_TRADING=false" >> .env
echo "TESTNET=false" >> .env
```

## 📄 Configuration Files

### Available Configuration Files

| File | Purpose | When to Use |
|------|---------|-------------|
| `config.json.example` | Complete template | Reference and customization |
| `config.development.json` | Development settings | Local development |
| `config.production.json` | Production settings | Live trading |
| `config.docker.json` | Container settings | Docker deployment |

### File Priority

1. Command line arguments
2. Environment variables
3. `config.json` (if exists)
4. Environment-specific config files
5. Default values

## 🔧 Configuration Sections

### Trading Configuration

Controls core trading behavior:

```json
{
  "trading": {
    "symbols": ["BTCUSDT", "ETHUSDT"],      // Trading pairs
    "paper_trading": true,                   // Safe mode
    "max_positions": 3,                      // Concurrent trades
    "max_position_size": 1000,               // USD per position
    "daily_loss_limit": -500,                // Stop loss limit
    "trading_enabled": true,                 // Enable/disable trading
    "auto_start": false                      // Auto-start on boot
  }
}
```

**Key Parameters:**
- **symbols**: Array of Binance trading pairs (USDT pairs only)
- **paper_trading**: `true` for simulation, `false` for real money
- **max_positions**: Maximum concurrent positions (1-10 recommended)
- **max_position_size**: Position size in USD (50-5000 recommended)
- **daily_loss_limit**: Stop trading if daily loss exceeds this (negative value)

### API Configuration

Binance API settings:

```json
{
  "api": {
    "api_key": "${BINANCE_API_KEY}",         // API key from env
    "api_secret": "${BINANCE_API_SECRET}",   // API secret from env
    "testnet": false,                        // Use Binance testnet
    "timeout": 30,                           // Request timeout (seconds)
    "rate_limit_buffer": 0.1                 // Rate limit safety buffer
  }
}
```

**Important Notes:**
- Always use environment variables for credentials
- Set `testnet: true` for safe testing
- API keys need trading permissions only (no withdrawals)
- Rate limit buffer prevents API violations

### Strategy Configuration

DipMaster AI strategy parameters:

```json
{
  "strategy": {
    "rsi_entry_range": [30, 50],             // RSI entry range
    "rsi_period": 14,                        // RSI calculation period
    "dip_threshold": 0.002,                  // 0.2% dip confirmation
    "volume_multiplier": 1.5,                // Volume spike threshold
    "max_holding_minutes": 180,              // Maximum hold time
    "target_profit": 0.008,                  // 0.8% profit target
    "min_confidence": 0.6,                   // Signal confidence threshold
    "boundary_slots": [15, 30, 45, 60],      // 15-min boundary slots
    "preferred_exit_slots": [1, 3],          // Preferred exit slots
    "enable_15min_boundary_exits": true,     // Enable boundary exits
    "enable_profit_targets": true,           // Enable profit targets
    "enable_time_stops": true                // Enable time stops
  }
}
```

**Strategy Tuning:**
- **rsi_entry_range**: Lower values = earlier entries, higher risk
- **dip_threshold**: Higher values = fewer but stronger signals
- **max_holding_minutes**: Shorter = faster turnover, longer = more patience
- **min_confidence**: Higher = fewer but more reliable signals

### Risk Management

Safety controls and limits:

```json
{
  "risk_management": {
    "max_daily_trades": 50,                  // Daily trade limit
    "max_consecutive_losses": 5,             // Stop after consecutive losses
    "position_size_type": "fixed",           // Position sizing method
    "emergency_stop_loss": -0.05,            // Emergency stop (-5%)
    "max_drawdown": -0.20,                   // Maximum drawdown (-20%)
    "cool_down_minutes": 15,                 // Cool-down after loss
    "enable_risk_limits": true               // Enable risk controls
  }
}
```

### WebSocket Configuration

Real-time data settings:

```json
{
  "websocket": {
    "enabled": true,                         // Enable WebSocket
    "reconnect_attempts": 5,                 // Reconnection attempts
    "reconnect_delay": 5,                    // Delay between attempts
    "ping_interval": 30,                     // Ping interval (seconds)
    "timeout": 10,                           // Connection timeout
    "buffer_size": 1000                      // Message buffer size
  }
}
```

### Dashboard Configuration

Web interface settings:

```json
{
  "dashboard": {
    "enabled": true,                         // Enable web dashboard
    "port": 8080,                            // Dashboard port
    "host": "0.0.0.0",                       // Bind address
    "refresh_rate": 1,                       // Update frequency (seconds)
    "show_charts": true,                     // Show charts
    "show_positions": true,                  // Show positions
    "show_logs": true,                       // Show logs
    "theme": "dark"                          // UI theme
  }
}
```

### Logging Configuration

Log output settings:

```json
{
  "logging": {
    "level": "INFO",                         // Log level
    "file_enabled": true,                    // Enable file logging
    "console_enabled": true,                 // Enable console logging
    "rotation": "daily",                     // Log rotation
    "retention_days": 30,                    // Log retention
    "max_file_size": "50MB",                 // Max file size
    "format": "detailed"                     // Log format
  }
}
```

## 🌍 Environment Variables

### Loading Priority

Environment variables override configuration file values using this format:
- `${VARIABLE_NAME}` - Required variable
- `${VARIABLE_NAME:default}` - Variable with default value

### Essential Variables

```bash
# Required for live trading
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret

# Trading mode
PAPER_TRADING=true
TESTNET=true

# Basic limits
MAX_POSITIONS=3
MAX_POSITION_SIZE=1000
DAILY_LOSS_LIMIT=-500
```

### Optional Variables

```bash
# System settings
LOG_LEVEL=INFO
DASHBOARD_ENABLED=true
NOTIFICATIONS_ENABLED=false

# Performance
MAX_WORKERS=4
MEMORY_LIMIT=2GB

# Strategy parameters
MIN_CONFIDENCE=0.6
TARGET_PROFIT=0.008
```

## 🏗️ Environment-Specific Settings

### Development Environment

**Purpose**: Safe local development and testing

**Key Settings:**
- Paper trading enabled
- Testnet enabled
- Debug logging
- Single position limit
- Lower risk parameters

**Usage:**
```bash
export ENVIRONMENT=development
python main.py --config config/config.development.json
```

### Production Environment  

**Purpose**: Live trading with real money

**Key Settings:**
- Paper trading disabled
- Production API
- INFO logging
- Full risk management
- Performance optimizations

**Usage:**
```bash
export ENVIRONMENT=production
python main.py --config config/config.production.json
```

### Docker Environment

**Purpose**: Container deployment

**Key Settings:**
- All settings from environment variables
- Container-optimized paths
- No file logging (stdout only)
- Health check endpoints

**Usage:**
```bash
docker run -d --env-file .env dipmaster-trading
```

## 🔒 Security Best Practices

### API Key Security

1. **Never commit credentials to Git:**
   ```bash
   # Add to .gitignore
   echo ".env" >> .gitignore
   echo "config/config.json" >> .gitignore
   ```

2. **Use environment variables:**
   ```json
   {
     "api_key": "${BINANCE_API_KEY}",
     "api_secret": "${BINANCE_API_SECRET}"
   }
   ```

3. **Restrict API permissions:**
   - Enable: Spot & Margin Trading
   - Disable: Withdrawals, Futures, Internal Transfer
   - Set IP restrictions

### Configuration Security

1. **File permissions:**
   ```bash
   chmod 600 .env
   chmod 600 config/config.json
   ```

2. **Production hardening:**
   ```json
   {
     "dashboard": {"host": "127.0.0.1"},
     "logging": {"console_enabled": false},
     "development": {"debug_mode": false}
   }
   ```

### Docker Security

```bash
# Run as non-root user
docker run --user 1001:1001 dipmaster-trading

# Read-only config
docker run -v $(pwd)/config:/app/config:ro dipmaster-trading

# Secrets management
docker secret create binance_api_key api_key.txt
```

## 🛠️ Advanced Configuration

### Dynamic Configuration Updates

Some settings can be updated without restart:

```bash
# Update via API
curl -X POST http://localhost:8080/config \
  -H "Content-Type: application/json" \
  -d '{"max_positions": 2}'

# Update via environment variables
export MAX_POSITIONS=2
kill -USR1 $(pgrep -f "python main.py")
```

### Custom Strategy Parameters

Override strategy parameters for fine-tuning:

```json
{
  "strategy": {
    "custom_parameters": {
      "ma_period": 20,
      "bollinger_std": 2.0,
      "volume_sma_period": 10
    }
  }
}
```

### External Configuration Sources

Load configuration from external sources:

```bash
# From URL
python main.py --config-url https://config.example.com/dipmaster.json

# From database
python main.py --config-db postgresql://user:pass@host/db --config-table config

# From AWS Parameter Store
python main.py --config-aws-param /dipmaster/config
```

## 🔧 Troubleshooting

### Common Configuration Issues

#### 1. API Authentication Failures

```bash
# Check API key format
echo $BINANCE_API_KEY | wc -c  # Should be 64 characters

# Test API connectivity
python -c "
from binance.client import Client
import os
client = Client(os.getenv('BINANCE_API_KEY'), os.getenv('BINANCE_API_SECRET'))
print(client.get_account_status())
"
```

#### 2. Configuration Loading Issues

```bash
# Validate JSON syntax
python -m json.tool config/config.json

# Check environment variable substitution
python -c "
import json, os
from string import Template
with open('config/config.json') as f:
    config = Template(f.read()).safe_substitute(os.environ)
    print(json.loads(config))
"
```

#### 3. Permission Issues

```bash
# Fix file permissions
chmod 644 config/*.json
chmod 600 .env

# Check directory permissions
ls -la config/
ls -la logs/
```

#### 4. Port Binding Issues

```bash
# Check port availability
netstat -tlnp | grep 8080

# Use alternative port
export DASHBOARD_PORT=8081
```

### Configuration Validation

Validate configuration before starting:

```bash
# Built-in validation
python main.py --validate-config --config config/config.json

# JSON schema validation
pip install jsonschema
python -c "
import json, jsonschema
with open('config/schema.json') as f:
    schema = json.load(f)
with open('config/config.json') as f:
    config = json.load(f)
jsonschema.validate(config, schema)
print('Configuration is valid')
"
```

### Debug Configuration Loading

Enable debug output to see configuration loading:

```bash
export DEBUG_CONFIG=true
python main.py --config config/config.json
```

## 📞 Support

For configuration help:

1. **Check logs**: Look for configuration errors in startup logs
2. **Validate JSON**: Ensure configuration files are valid JSON
3. **Test API**: Verify API credentials work with Binance
4. **Check permissions**: Ensure file permissions are correct
5. **Review documentation**: Check this guide and API references

---

**Last Updated**: 2025-08-12  
**Configuration Guide Version**: 1.0.0  
**Compatible with**: DipMaster Trading System v1.0.0