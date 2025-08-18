#!/bin/bash

# ============================================================================
# DipMaster Trading System - Docker一键部署脚本
# Version: 1.0.0
# Date: 2025-08-18
# 支持: CentOS/Ubuntu/Debian + Docker
# ============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# 日志函数
log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
success() { echo -e "${GREEN}✅ $1${NC}"; }
warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
error() { echo -e "${RED}❌ $1${NC}"; }
header() { echo -e "${PURPLE}$1${NC}"; }

# 检查是否为root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        error "此脚本需要root权限运行"
        echo "请使用: sudo $0"
        exit 1
    fi
}

# 检测操作系统
detect_os() {
    if [[ -f /etc/redhat-release ]]; then
        OS="centos"
        PACKAGE_MANAGER="yum"
    elif [[ -f /etc/debian_version ]]; then
        OS="ubuntu"
        PACKAGE_MANAGER="apt"
    else
        error "不支持的操作系统"
        exit 1
    fi
    log "检测到系统: $OS"
}

# 安装Docker
install_docker() {
    header "🐳 安装Docker..."
    
    if command -v docker &> /dev/null; then
        success "Docker已安装: $(docker --version)"
        return
    fi
    
    if [[ "$OS" == "centos" ]]; then
        # CentOS/RHEL安装Docker
        yum update -y
        yum install -y yum-utils device-mapper-persistent-data lvm2
        yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
        yum install -y docker-ce docker-ce-cli containerd.io
        
    elif [[ "$OS" == "ubuntu" ]]; then
        # Ubuntu/Debian安装Docker
        apt update
        apt install -y apt-transport-https ca-certificates curl gnupg lsb-release
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
        apt update
        apt install -y docker-ce docker-ce-cli containerd.io
    fi
    
    # 启动Docker服务
    systemctl start docker
    systemctl enable docker
    
    # 添加当前用户到docker组
    usermod -aG docker $USER || true
    
    success "Docker安装完成"
}

# 安装Docker Compose
install_docker_compose() {
    header "🔧 安装Docker Compose..."
    
    if command -v docker-compose &> /dev/null; then
        success "Docker Compose已安装: $(docker-compose --version)"
        return
    fi
    
    # 下载Docker Compose
    curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    
    # 创建软链接
    ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
    
    success "Docker Compose安装完成"
}

# 准备项目目录
setup_project() {
    header "📁 准备项目目录..."
    
    PROJECT_DIR="/opt/DipMaster-Trading-System"
    
    # 备份现有目录
    if [[ -d "$PROJECT_DIR" ]]; then
        warning "备份现有项目目录"
        mv "$PROJECT_DIR" "${PROJECT_DIR}_backup_$(date +%s)"
    fi
    
    # 创建新目录
    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    
    # 克隆项目
    log "克隆DipMaster项目..."
    git clone https://github.com/Gyarados4157/DipMaster-Trading-System.git .
    
    # 创建必要目录
    mkdir -p logs data results config backup
    
    success "项目目录准备完成: $PROJECT_DIR"
}

# 创建配置文件
create_config() {
    header "⚙️  创建配置文件..."
    
    cd /opt/DipMaster-Trading-System
    
    # 创建Docker环境配置
    if [[ ! -f "config/paper_trading_config.json" ]]; then
        cat > config/paper_trading_config.json << 'EOF'
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
    
    # 创建环境变量文件
    cat > .env << 'EOF'
# DipMaster Docker环境变量
COMPOSE_PROJECT_NAME=dipmaster
TZ=Asia/Hong_Kong
LOG_LEVEL=INFO
PAPER_TRADING=true

# 资源限制
MEMORY_LIMIT=1g
CPU_LIMIT=0.8

# 网络端口
WEB_PORT=8080
MONITORING_PORT=9090
EOF
    
    success "环境配置完成"
}

# 构建Docker镜像
build_image() {
    header "🏗️  构建Docker镜像..."
    
    cd /opt/DipMaster-Trading-System
    
    # 构建镜像
    docker build -t dipmaster-trading:latest .
    
    success "Docker镜像构建完成"
}

# 创建管理脚本
create_scripts() {
    header "📝 创建管理脚本..."
    
    cd /opt/DipMaster-Trading-System
    
    # 启动脚本
    cat > start_docker.sh << 'EOF'
#!/bin/bash
echo "🚀 启动DipMaster Docker容器..."
cd /opt/DipMaster-Trading-System

# 检查配置文件
if [[ ! -f "config/paper_trading_config.json" ]]; then
    echo "❌ 配置文件不存在，请先编辑配置文件"
    echo "配置路径: /opt/DipMaster-Trading-System/config/paper_trading_config.json"
    exit 1
fi

# 启动服务
docker-compose -f docker-compose.simple.yml up -d

echo "✅ 启动完成"
echo "🌐 监控面板: http://localhost:8080"
echo "📊 查看日志: docker-compose -f docker-compose.simple.yml logs -f"
EOF

    # 停止脚本
    cat > stop_docker.sh << 'EOF'
#!/bin/bash
echo "🛑 停止DipMaster Docker容器..."
cd /opt/DipMaster-Trading-System
docker-compose -f docker-compose.simple.yml down
echo "✅ 已停止"
EOF

    # 监控脚本
    cat > monitor_docker.sh << 'EOF'
#!/bin/bash
echo "📊 DipMaster Docker监控"
echo "======================"
cd /opt/DipMaster-Trading-System

echo "📦 容器状态:"
docker-compose -f docker-compose.simple.yml ps

echo ""
echo "💾 系统资源:"
docker stats --no-stream dipmaster-trading

echo ""
echo "📈 最新日志:"
docker-compose -f docker-compose.simple.yml logs --tail=20 dipmaster

echo ""
echo "🔧 管理命令:"
echo "启动: ./start_docker.sh"
echo "停止: ./stop_docker.sh"
echo "重启: ./restart_docker.sh"
echo "日志: docker-compose -f docker-compose.simple.yml logs -f"
EOF

    # 重启脚本
    cat > restart_docker.sh << 'EOF'
#!/bin/bash
echo "🔄 重启DipMaster Docker容器..."
cd /opt/DipMaster-Trading-System
docker-compose -f docker-compose.simple.yml restart
echo "✅ 重启完成"
EOF

    # 设置执行权限
    chmod +x *.sh
    
    success "管理脚本创建完成"
}

# 启动服务
start_services() {
    header "🚀 启动DipMaster服务..."
    
    cd /opt/DipMaster-Trading-System
    
    # 使用简化版docker-compose启动
    docker-compose -f docker-compose.simple.yml up -d
    
    # 等待服务启动
    sleep 10
    
    # 检查服务状态
    if docker-compose -f docker-compose.simple.yml ps | grep -q "Up"; then
        success "服务启动成功"
    else
        warning "服务可能启动失败，请检查日志"
    fi
}

# 显示部署结果
show_result() {
    header "🎉 Docker部署完成!"
    echo ""
    echo "📋 部署信息:"
    echo "============"
    echo "📁 项目目录: /opt/DipMaster-Trading-System"
    echo "🐳 Docker镜像: dipmaster-trading:latest"
    echo "🌐 监控面板: http://$(hostname -I | awk '{print $1}'):8080"
    echo "📊 容器状态: docker-compose -f docker-compose.simple.yml ps"
    echo ""
    echo "🔧 管理命令:"
    echo "============"
    echo "启动服务: ./start_docker.sh"
    echo "停止服务: ./stop_docker.sh"
    echo "重启服务: ./restart_docker.sh"
    echo "查看监控: ./monitor_docker.sh"
    echo "查看日志: docker-compose -f docker-compose.simple.yml logs -f"
    echo ""
    echo "⚙️  配置文件:"
    echo "============"
    echo "主配置: /opt/DipMaster-Trading-System/config/paper_trading_config.json"
    echo "环境变量: /opt/DipMaster-Trading-System/.env"
    echo ""
    warning "重要: 请编辑配置文件添加您的API密钥后重启服务!"
    echo "编辑配置: vi /opt/DipMaster-Trading-System/config/paper_trading_config.json"
    echo "重启服务: cd /opt/DipMaster-Trading-System && ./restart_docker.sh"
}

# 主函数
main() {
    header "🎯 DipMaster Trading System - Docker一键部署"
    header "============================================="
    
    check_root
    detect_os
    install_docker
    install_docker_compose
    setup_project
    create_config
    build_image
    create_scripts
    start_services
    show_result
}

# 运行主函数
main "$@"