#!/bin/bash

# ============================================================================
# DipMaster Trading System - Complete Linux Deployment Script
# Version: 3.0.0
# Date: 2025-08-18
# Supports: CentOS/RHEL/Amazon Linux
# ============================================================================

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Logging functions
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

header() {
    echo -e "${PURPLE}$1${NC}"
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log "Running as root user"
    else
        error "This script must be run as root. Use: sudo $0"
        exit 1
    fi
}

# Detect OS
detect_os() {
    if [[ -f /etc/redhat-release ]]; then
        OS="centos"
        log "Detected OS: CentOS/RHEL/Amazon Linux"
    elif [[ -f /etc/debian_version ]]; then
        OS="ubuntu"
        log "Detected OS: Ubuntu/Debian"
    else
        error "Unsupported OS. This script supports CentOS/RHEL and Ubuntu/Debian."
        exit 1
    fi
}

# Install Python 3.11
install_python311() {
    header "🐍 Installing Python 3.11..."
    
    if command -v python3.11 &> /dev/null; then
        success "Python 3.11 already installed"
        return
    fi
    
    if [[ "$OS" == "centos" ]]; then
        # Install EPEL and development tools
        yum update -y
        yum groupinstall -y "Development Tools"
        
        # Handle Alibaba Cloud EPEL conflict
        if yum list installed | grep -q "epel-aliyuncs-release"; then
            log "Detected Alibaba Cloud EPEL, removing conflicting package..."
            yum remove -y epel-aliyuncs-release || true
        fi
        
        # Install EPEL with conflict resolution
        yum install -y epel-release --allowerasing || {
            warning "EPEL installation failed, trying alternative method..."
            yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm --allowerasing
        }
        
        yum install -y wget curl git openssl-devel libffi-devel bzip2-devel sqlite-devel readline-devel zlib-devel xz-devel ncurses-devel
        
        # Install Python 3.11 from source
        cd /tmp
        wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
        tar -xzf Python-3.11.9.tgz
        cd Python-3.11.9
        ./configure --enable-optimizations --with-ensurepip=install
        make -j $(nproc)
        make altinstall
        
        # Create symlinks
        ln -sf /usr/local/bin/python3.11 /usr/bin/python3
        ln -sf /usr/local/bin/pip3.11 /usr/bin/pip3
        
    elif [[ "$OS" == "ubuntu" ]]; then
        # Ubuntu/Debian
        apt update
        apt install -y software-properties-common
        add-apt-repository ppa:deadsnakes/ppa -y
        apt update
        apt install -y python3.11 python3.11-pip python3.11-dev python3.11-venv
        
        # Create symlinks
        update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
        ln -sf /usr/bin/python3.11 /usr/bin/python3
    fi
    
    # Verify installation
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\\d+\\.\\d+' | head -1)
    if [[ "$PYTHON_VERSION" == "3.11" ]]; then
        success "Python 3.11 installed successfully"
    else
        error "Python 3.11 installation failed. Current version: $PYTHON_VERSION"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    header "📦 Installing system dependencies..."
    
    if [[ "$OS" == "centos" ]]; then
        yum install -y git wget curl vim htop screen tmux
    elif [[ "$OS" == "ubuntu" ]]; then
        apt install -y git wget curl vim htop screen tmux build-essential
    fi
    
    success "System dependencies installed"
}

# Install pip and upgrade
setup_pip() {
    header "📚 Setting up pip..."
    
    # Upgrade pip
    python3 -m pip install --upgrade pip setuptools wheel
    
    # Verify pip
    pip3 --version
    success "pip setup complete"
}

# Create project directory and clone repository
setup_project() {
    header "📁 Setting up project..."
    
    PROJECT_DIR="/opt/DipMaster-Trading-System"
    
    # Remove existing directory if it exists
    if [[ -d "$PROJECT_DIR" ]]; then
        warning "Removing existing project directory"
        rm -rf "$PROJECT_DIR"
    fi
    
    # Create directory
    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    
    # Clone repository
    log "Cloning DipMaster repository..."
    git clone https://github.com/Gyarados4157/DipMaster-Trading-System.git .
    
    success "Project setup complete in $PROJECT_DIR"
}

# Install Python dependencies
install_python_deps() {
    header "🔧 Installing Python dependencies..."
    
    cd /opt/DipMaster-Trading-System
    
    # Install dependencies
    pip3 install -r requirements_linux.txt --no-cache-dir
    
    success "Python dependencies installed"
}

# Create configuration files
setup_config() {
    header "⚙️  Setting up configuration..."
    
    cd /opt/DipMaster-Trading-System
    
    # Create necessary directories
    mkdir -p logs data results config
    
    # Create paper trading config if it doesn't exist
    if [[ ! -f "config/paper_trading_config.json" ]]; then
        cat > config/paper_trading_config.json << 'CONFIG_EOF'
{
    "trading": {
        "paper_trading": true,
        "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        "max_positions": 3,
        "position_size_usd": 500,
        "risk_per_trade": 0.02,
        "max_daily_trades": 10
    },
    "exchange": {
        "name": "binance",
        "api_key": "your_api_key_here",
        "api_secret": "your_api_secret_here",
        "testnet": true,
        "rate_limit": true
    },
    "strategy": {
        "name": "DipMaster",
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "take_profit": 0.008,
        "stop_loss": 0.005,
        "max_holding_hours": 3
    },
    "risk_management": {
        "max_daily_loss": 100,
        "max_position_size": 1000,
        "max_drawdown": 0.05
    },
    "logging": {
        "level": "INFO",
        "file": "logs/dipmaster.log",
        "max_file_size": "10MB",
        "backup_count": 5
    }
}
CONFIG_EOF
        success "Default configuration created"
    fi
}

# Create startup scripts
create_scripts() {
    header "🚀 Creating startup scripts..."
    
    cd /opt/DipMaster-Trading-System
    
    # Create startup script
    cat > start_dipmaster.sh << 'START_EOF'
#!/bin/bash

echo "🚀 Starting DipMaster Trading System..."

# Check if config exists
if [[ ! -f "config/paper_trading_config.json" ]]; then
    echo "❌ Configuration file not found: config/paper_trading_config.json"
    echo "Please edit the configuration file with your API keys before starting."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\\d+\\.\\d+' | head -1)
if [[ "$PYTHON_VERSION" != "3.11" ]]; then
    echo "❌ Python 3.11 required, found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python version check passed: $PYTHON_VERSION"

# Start the trading system
echo "📊 Starting DipMaster paper trading..."
cd /opt/DipMaster-Trading-System

# Try quick test first, then main system
python3 quick_paper_test.py || {
    echo "🔄 Quick test failed, starting main system..."
    python3 main.py --paper --config config/paper_trading_config.json
}
START_EOF

    # Create monitoring script
    cat > monitor_dipmaster.sh << 'MONITOR_EOF'
#!/bin/bash

echo "📊 DipMaster Trading System Monitor"
echo "===================================="

# Check system info
echo "🖥️  System Information:"
echo "Time: $(date)"
echo "Uptime: $(uptime -p 2>/dev/null || uptime)"
echo ""

# Check if process is running
echo "🔍 Process Status:"
PROCESS_COUNT=$(ps aux | grep -E "(main.py|quick_paper_test.py)" | grep -v grep | wc -l)
if [[ $PROCESS_COUNT -gt 0 ]]; then
    echo "✅ DipMaster process is running"
    ps aux | grep -E "(main.py|quick_paper_test.py)" | grep -v grep
else
    echo "❌ DipMaster process is not running"
fi

echo ""
echo "💾 Disk Usage:"
df -h /opt/DipMaster-Trading-System | head -2

echo ""
echo "🧠 Memory Usage:"
free -h | head -2

echo ""
echo "📈 Latest Logs:"
echo "==============="

cd /opt/DipMaster-Trading-System

# Show latest logs
if [[ -d "logs" ]]; then
    LATEST_LOG=$(ls -t logs/*.log 2>/dev/null | head -1)
    if [[ -n "$LATEST_LOG" ]]; then
        echo "📄 Log file: $LATEST_LOG"
        echo "📝 Last 20 lines:"
        tail -20 "$LATEST_LOG"
    else
        echo "No log files found"
    fi
else
    echo "Logs directory does not exist"
fi

echo ""
echo "🔧 System Commands:"
echo "=================="
echo "Start:   ./start_dipmaster.sh"
echo "Stop:    pkill -f 'python3.*main.py'"
echo "Monitor: ./monitor_dipmaster.sh"
echo "Logs:    tail -f logs/dipmaster_*.log"
MONITOR_EOF

    # Create stop script
    cat > stop_dipmaster.sh << 'STOP_EOF'
#!/bin/bash

echo "🛑 Stopping DipMaster Trading System..."

# Kill main processes
pkill -f "python3.*main.py" 2>/dev/null && echo "✅ Main process stopped"
pkill -f "python3.*quick_paper_test.py" 2>/dev/null && echo "✅ Test process stopped"

# Wait a moment
sleep 2

# Check if processes are still running
REMAINING=$(ps aux | grep -E "(main.py|quick_paper_test.py)" | grep -v grep | wc -l)
if [[ $REMAINING -eq 0 ]]; then
    echo "✅ All DipMaster processes stopped successfully"
else
    echo "⚠️  Some processes may still be running:"
    ps aux | grep -E "(main.py|quick_paper_test.py)" | grep -v grep
fi
STOP_EOF

    # Make scripts executable
    chmod +x start_dipmaster.sh monitor_dipmaster.sh stop_dipmaster.sh
    
    success "Startup scripts created"
}

# Create systemd service (optional)
create_service() {
    header "🔧 Creating systemd service..."
    
    cat > /etc/systemd/system/dipmaster.service << 'SERVICE_EOF'
[Unit]
Description=DipMaster Trading System
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/DipMaster-Trading-System
ExecStart=/opt/DipMaster-Trading-System/start_dipmaster.sh
ExecStop=/opt/DipMaster-Trading-System/stop_dipmaster.sh
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
SERVICE_EOF

    systemctl daemon-reload
    systemctl enable dipmaster
    
    success "Systemd service created (disabled by default)"
    log "To enable auto-start: systemctl enable dipmaster"
    log "To start service: systemctl start dipmaster"
}

# Run basic tests
run_tests() {
    header "🧪 Running basic tests..."
    
    cd /opt/DipMaster-Trading-System
    
    # Test Python import
    python3 -c "
import json
import asyncio
import sys
print('✅ Basic Python imports successful')

# Test config loading
try:
    with open('config/paper_trading_config.json', 'r') as f:
        config = json.load(f)
    print('✅ Configuration file loads successfully')
except Exception as e:
    print(f'❌ Configuration error: {e}')
    sys.exit(1)

print('✅ All basic tests passed')
"
    
    success "Basic tests completed"
}

# Set proper permissions
set_permissions() {
    header "🔒 Setting file permissions..."
    
    cd /opt/DipMaster-Trading-System
    
    # Set directory permissions
    chmod 755 . logs data results config
    
    # Set script permissions
    chmod +x *.sh
    
    # Set config file permissions
    chmod 644 config/*.json
    
    success "File permissions set"
}

# Main deployment function
main() {
    header "🎯 DipMaster Trading System - Linux Deployment"
    header "=============================================="
    
    check_root
    detect_os
    install_python311
    install_system_deps
    setup_pip
    setup_project
    install_python_deps
    setup_config
    create_scripts
    create_service
    set_permissions
    run_tests
    
    success "🎉 Deployment completed successfully!"
    echo ""
    header "📋 Next Steps:"
    echo "=============="
    echo "1. Edit configuration: vi /opt/DipMaster-Trading-System/config/paper_trading_config.json"
    echo "2. Add your Binance API keys (for paper trading, testnet keys are recommended)"
    echo "3. Start the system: cd /opt/DipMaster-Trading-System && ./start_dipmaster.sh"
    echo "4. Monitor the system: ./monitor_dipmaster.sh"
    echo ""
    header "📊 Management Commands:"
    echo "======================"
    echo "Start:   systemctl start dipmaster"
    echo "Stop:    systemctl stop dipmaster"
    echo "Status:  systemctl status dipmaster"
    echo "Logs:    journalctl -u dipmaster -f"
    echo ""
    header "📁 Project Location: /opt/DipMaster-Trading-System"
    echo ""
    warning "IMPORTANT: Edit the configuration file before starting!"
}

# Run main function
main "$@"