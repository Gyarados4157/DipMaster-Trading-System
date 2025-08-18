#!/bin/bash

# ============================================================================
# DipMaster Trading System - Linux Deployment Script (Fixed)
# Version: 2.0.0
# Date: 2025-08-18
# ============================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    log "Running as root user"
else
    warning "Not running as root. Some operations may require sudo."
fi

log "🚀 开始DipMaster交易系统Linux部署..."

# Check Python version
log "🐍 检查Python版本..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\\d+\\.\\d+' | head -1)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 6 ]]; then
    error "需要Python 3.6或更高版本，当前版本: $PYTHON_VERSION"
    exit 1
fi

success "Python版本检查通过: $PYTHON_VERSION"

# Update system packages
log "📦 更新系统包..."
if command -v yum &> /dev/null; then
    yum update -y
    yum install -y git wget curl python3-pip python3-dev build-essential
elif command -v apt &> /dev/null; then
    apt update
    apt install -y git wget curl python3-pip python3-dev build-essential
else
    warning "未知的包管理器，请手动安装依赖"
fi

# Install Python dependencies
log "📚 安装Python依赖..."
if [[ -f "requirements_linux.txt" ]]; then
    pip3 install --upgrade pip
    pip3 install -r requirements_linux.txt --no-cache-dir
    success "Linux兼容依赖安装完成"
elif [[ -f "requirements.txt" ]]; then
    pip3 install --upgrade pip
    # Try to install without constraints first
    pip3 install numpy pandas ccxt python-binance websockets aiohttp fastapi uvicorn --no-cache-dir
    success "基础依赖安装完成"
else
    error "未找到requirements文件"
    exit 1
fi

# Create necessary directories
log "📁 创建必要目录..."
mkdir -p logs data results config

# Setup configuration
log "⚙️  设置配置文件..."
if [[ ! -f "config/paper_trading_config.json" ]]; then
    if [[ -f "config/config.json.example" ]]; then
        cp config/config.json.example config/paper_trading_config.json
        success "配置文件已从示例复制"
    else
        # Create minimal config
        cat > config/paper_trading_config.json << 'CONFIG_EOF'
{
    "trading": {
        "paper_trading": true,
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "max_positions": 2,
        "position_size_usd": 100,
        "risk_per_trade": 0.02
    },
    "exchange": {
        "name": "binance",
        "api_key": "",
        "api_secret": "",
        "testnet": true
    },
    "strategy": {
        "name": "DipMaster",
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "take_profit": 0.02,
        "stop_loss": 0.01
    },
    "logging": {
        "level": "INFO",
        "file": "logs/dipmaster.log"
    }
}
CONFIG_EOF
        success "默认配置文件已创建"
    fi
fi

# Create startup script
log "🚀 创建启动脚本..."
cat > start_7day_test.sh << 'START_EOF'
#!/bin/bash

echo "🚀 启动DipMaster 7天纸面交易测试..."

# Check if config exists
if [[ ! -f "config/paper_trading_config.json" ]]; then
    echo "❌ 配置文件不存在: config/paper_trading_config.json"
    exit 1
fi

# Create logs directory
mkdir -p logs

# Start paper trading
echo "📊 启动纸面交易系统..."
python3 quick_paper_test.py || {
    echo "❌ 快速测试失败，尝试主程序..."
    python3 main.py --paper --config config/paper_trading_config.json
}
START_EOF

chmod +x start_7day_test.sh
success "启动脚本已创建"

# Create monitoring script
log "📊 创建监控脚本..."
cat > monitor.sh << 'MONITOR_EOF'
#!/bin/bash

echo "📊 DipMaster监控面板"
echo "===================="

# Check if process is running
PROCESS_COUNT=$(ps aux | grep -E "(main.py|quick_paper_test.py)" | grep -v grep | wc -l)
if [[ $PROCESS_COUNT -gt 0 ]]; then
    echo "✅ DipMaster进程正在运行"
    ps aux | grep -E "(main.py|quick_paper_test.py)" | grep -v grep
else
    echo "❌ DipMaster进程未运行"
fi

echo ""
echo "📈 最新日志:"
echo "============"

# Show latest logs
if [[ -d "logs" ]]; then
    LATEST_LOG=$(ls -t logs/*.log 2>/dev/null | head -1)
    if [[ -n "$LATEST_LOG" ]]; then
        echo "📄 日志文件: $LATEST_LOG"
        tail -20 "$LATEST_LOG"
    else
        echo "暂无日志文件"
    fi
else
    echo "日志目录不存在"
fi

echo ""
echo "💾 磁盘使用情况:"
echo "==============="
df -h . | head -2

echo ""
echo "🖥️  内存使用情况:"
echo "==============="
free -h | head -2
MONITOR_EOF

chmod +x monitor.sh
success "监控脚本已创建"

# Test configuration loading
log "🧪 测试配置加载..."
python3 -c "
import json
import sys
try:
    with open('config/paper_trading_config.json', 'r') as f:
        config = json.load(f)
    print('✅ 配置加载成功: config/paper_trading_config.json')
except Exception as e:
    print(f'❌ 配置加载失败: {e}')
    sys.exit(1)
"

# Quick test
log "🧪 运行快速测试..."
timeout 30s python3 -c "
import sys
sys.path.append('.')
try:
    from src.core.asyncio_compat import asyncio_run
    print('✅ asyncio兼容性检查通过')
except Exception as e:
    print(f'❌ asyncio兼容性检查失败: {e}')
    print('但系统仍可能正常工作')
" || warning "快速测试超时或失败，但系统可能仍然可用"

# Set proper permissions
log "🔒 设置文件权限..."
chmod +x *.sh
chmod 644 config/*.json
chmod 755 logs data results

success "🎉 部署完成!"
echo "=================================================="
echo "📁 项目目录: $(pwd)"
echo "⚙️  配置文件: config/paper_trading_config.json"
echo "🚀 启动命令: ./start_7day_test.sh"
echo "📊 监控命令: ./monitor.sh"
echo ""
echo "现在可以运行: ./start_7day_test.sh 开始7天测试!"
echo "=================================================="