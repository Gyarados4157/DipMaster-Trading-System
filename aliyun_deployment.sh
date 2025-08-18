#!/bin/bash

# ============================================================================
# DipMaster Trading System - 阿里云服务器专用部署脚本
# Version: 1.0.0
# Date: 2025-08-18
# 专门处理阿里云ECS的EPEL冲突问题
# ============================================================================

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }

log "🎯 DipMaster阿里云ECS专用部署脚本"

# 检查root权限
if [[ $EUID -ne 0 ]]; then
    error "需要root权限运行此脚本"
    exit 1
fi

# 处理阿里云EPEL冲突
log "🔧 处理阿里云EPEL冲突..."
if rpm -qa | grep -q "epel-aliyuncs-release"; then
    log "检测到阿里云EPEL，正在移除冲突包..."
    yum remove -y epel-aliyuncs-release --nodeps || true
fi

# 清理yum缓存
yum clean all
yum makecache

# 安装基础开发工具
log "📦 安装开发工具..."
yum groupinstall -y "Development Tools" --skip-broken || {
    # 如果群组安装失败，逐个安装核心包
    yum install -y gcc gcc-c++ make automake autoconf libtool
}

# 安装系统依赖
log "📚 安装系统依赖..."
yum install -y wget curl git vim htop \
    openssl-devel libffi-devel bzip2-devel \
    sqlite-devel readline-devel zlib-devel \
    xz-devel ncurses-devel expat-devel \
    gdbm-devel tk-devel uuid-devel

# 安装Python 3.11
log "🐍 安装Python 3.11..."
if ! command -v python3.11 &> /dev/null; then
    cd /tmp
    
    # 下载Python 3.11源码
    if [[ ! -f "Python-3.11.9.tgz" ]]; then
        wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
    fi
    
    tar -xzf Python-3.11.9.tgz
    cd Python-3.11.9
    
    # 配置编译选项
    ./configure --enable-optimizations \
                --with-ensurepip=install \
                --enable-shared \
                --prefix=/usr/local
    
    # 编译安装
    make -j$(nproc)
    make altinstall
    
    # 创建软链接
    ln -sf /usr/local/bin/python3.11 /usr/bin/python3
    ln -sf /usr/local/bin/pip3.11 /usr/bin/pip3
    
    # 更新动态链接库
    echo "/usr/local/lib" > /etc/ld.so.conf.d/python3.11.conf
    ldconfig
else
    success "Python 3.11已安装"
fi

# 验证Python安装
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
if [[ "$PYTHON_VERSION" == "3.11" ]]; then
    success "Python 3.11安装成功"
else
    error "Python 3.11安装失败，当前版本: $PYTHON_VERSION"
    exit 1
fi

# 升级pip
log "⬆️  升级pip..."
python3 -m pip install --upgrade pip setuptools wheel

# 创建项目目录
log "📁 设置项目目录..."
PROJECT_DIR="/opt/DipMaster-Trading-System"

if [[ -d "$PROJECT_DIR" ]]; then
    warning "项目目录已存在，正在备份..."
    mv "$PROJECT_DIR" "${PROJECT_DIR}_backup_$(date +%s)"
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 克隆项目
log "📥 克隆项目代码..."
git clone https://github.com/Gyarados4157/DipMaster-Trading-System.git .

# 安装Python依赖
log "📦 安装Python依赖..."
pip3 install -r requirements_linux.txt --no-cache-dir

# 创建必要目录
mkdir -p logs data results config

# 创建配置文件
log "⚙️  创建配置文件..."
if [[ ! -f "config/paper_trading_config.json" ]]; then
    cat > config/paper_trading_config.json << 'EOF'
{
    "trading": {
        "paper_trading": true,
        "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
        "max_positions": 3,
        "position_size_usd": 500,
        "risk_per_trade": 0.02
    },
    "exchange": {
        "name": "binance",
        "api_key": "your_api_key_here",
        "api_secret": "your_api_secret_here",
        "testnet": true
    },
    "strategy": {
        "name": "DipMaster",
        "rsi_period": 14,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "take_profit": 0.008,
        "stop_loss": 0.005
    },
    "logging": {
        "level": "INFO",
        "file": "logs/dipmaster.log"
    }
}
EOF
    success "配置文件已创建"
fi

# 创建启动脚本
log "🚀 创建管理脚本..."
cat > start_dipmaster.sh << 'EOF'
#!/bin/bash
echo "🚀 启动DipMaster交易系统..."
cd /opt/DipMaster-Trading-System

if [[ ! -f "config/paper_trading_config.json" ]]; then
    echo "❌ 配置文件不存在"
    exit 1
fi

mkdir -p logs
echo "📊 启动纸面交易..."
python3 quick_paper_test.py || python3 main.py --paper --config config/paper_trading_config.json
EOF

cat > stop_dipmaster.sh << 'EOF'
#!/bin/bash
echo "🛑 停止DipMaster..."
pkill -f "python3.*main.py" 2>/dev/null
pkill -f "python3.*quick_paper_test.py" 2>/dev/null
echo "✅ 已停止"
EOF

cat > monitor_dipmaster.sh << 'EOF'
#!/bin/bash
echo "📊 DipMaster监控"
echo "================"
ps aux | grep -E "(main.py|quick_paper_test.py)" | grep -v grep || echo "进程未运行"
echo ""
echo "📈 最新日志:"
cd /opt/DipMaster-Trading-System
if [[ -d logs ]]; then
    LATEST_LOG=$(ls -t logs/*.log 2>/dev/null | head -1)
    if [[ -n "$LATEST_LOG" ]]; then
        tail -20 "$LATEST_LOG"
    else
        echo "暂无日志"
    fi
fi
EOF

chmod +x *.sh

# 设置权限
chmod 755 logs data results config
chmod 644 config/*.json

# 测试安装
log "🧪 测试安装..."
python3 -c "
import json, asyncio, sys
print('✅ Python导入成功')
with open('config/paper_trading_config.json') as f:
    json.load(f)
print('✅ 配置文件加载成功')
"

success "🎉 阿里云ECS部署完成!"
echo ""
echo "📋 下一步操作:"
echo "1. 编辑配置: vi config/paper_trading_config.json"
echo "2. 添加API密钥"
echo "3. 启动系统: ./start_dipmaster.sh"
echo "4. 监控系统: ./monitor_dipmaster.sh"
echo ""
echo "📁 项目位置: $PROJECT_DIR"