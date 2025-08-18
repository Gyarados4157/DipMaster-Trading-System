#!/bin/bash

#############################################################################
# DipMaster 7天纸面交易部署脚本
# 用途: 快速部署到远程服务器并启动长期测试
# 作者: DipMaster Trading System
# 日期: 2025-08-18
#############################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置变量 (使用SSH配置文件中的别名)
REMOTE_HOST="dipmaster-aliyun"
REMOTE_USER=""  # 使用SSH配置文件中的用户
PROJECT_NAME="DipMaster-Trading-System"
REMOTE_DIR="/opt/${PROJECT_NAME}"
LOCAL_PROJECT_DIR="."

echo -e "${BLUE}🚀 DipMaster 7天纸面交易部署开始${NC}"
echo "=========================================="

# 函数：打印带颜色的信息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查本地环境
check_local_environment() {
    print_info "检查本地环境..."
    
    if [ ! -f "config/paper_trading_config.json" ]; then
        print_error "配置文件不存在: config/paper_trading_config.json"
        exit 1
    fi
    
    if [ ! -f "run_paper_trading.py" ]; then
        print_error "主程序文件不存在: run_paper_trading.py"
        exit 1
    fi
    
    print_info "✅ 本地环境检查通过"
}

# 检查SSH连接
check_ssh_connection() {
    print_info "检查SSH连接到 ${REMOTE_HOST}..."
    
    if ! ssh -o ConnectTimeout=10 ${REMOTE_HOST} "echo 'SSH连接成功'" > /dev/null 2>&1; then
        print_error "无法连接到远程服务器"
        print_error "请确保SSH配置正确: ssh ${REMOTE_HOST}"
        exit 1
    fi
    
    print_info "✅ SSH连接正常"
}

# 准备远程服务器环境
prepare_remote_environment() {
    print_info "准备远程服务器环境..."
    
    ssh ${REMOTE_HOST} << 'EOF'
        # 更新系统包
        if command -v yum >/dev/null 2>&1; then
            yum update -y
            yum install -y python3 python3-pip git screen htop
        elif command -v apt >/dev/null 2>&1; then
            apt update
            apt install -y python3 python3-pip git screen htop
        else
            echo "不支持的包管理器"
            exit 1
        fi
        
        # 检查Python版本
        python3 --version
        pip3 --version
        
        # 创建项目目录
        mkdir -p /opt/DipMaster-Trading-System
        mkdir -p /opt/DipMaster-Trading-System/{logs,results,data}
        
        echo "✅ 远程环境准备完成"
EOF
    
    print_info "✅ 远程环境准备完成"
}

# 同步项目文件
sync_project_files() {
    print_info "同步项目文件到远程服务器..."
    
    # 创建临时排除文件
    cat > /tmp/rsync_exclude << 'EOF'
.git/
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.vscode/
.idea/
*.log
logs/*.log
results/*
data/*.parquet
data/*.csv
*.tmp
.DS_Store
.env
venv/
env/
EOF

    # 同步文件
    rsync -avz --delete \
        --exclude-from=/tmp/rsync_exclude \
        ${LOCAL_PROJECT_DIR}/ ${REMOTE_HOST}:${REMOTE_DIR}/
    
    # 清理临时文件
    rm -f /tmp/rsync_exclude
    
    print_info "✅ 文件同步完成"
}

# 安装Python依赖
install_dependencies() {
    print_info "安装Python依赖..."
    
    ssh ${REMOTE_HOST} << EOF
        cd ${REMOTE_DIR}
        
        # 升级pip
        python3 -m pip install --upgrade pip
        
        # 安装依赖
        if [ -f requirements.txt ]; then
            pip3 install -r requirements.txt
        else
            # 安装基本依赖
            pip3 install numpy pandas asyncio aiohttp websockets ccxt ta-lib python-binance
        fi
        
        echo "✅ 依赖安装完成"
EOF
    
    print_info "✅ Python依赖安装完成"
}

# 配置纸面交易参数
configure_paper_trading() {
    print_info "配置纸面交易参数..."
    
    ssh ${REMOTE_HOST} << EOF
        cd ${REMOTE_DIR}
        
        # 确保配置文件存在
        if [ ! -f config/paper_trading_config.json ]; then
            echo "❌ 配置文件不存在"
            exit 1
        fi
        
        # 创建服务器专用配置
        cp config/paper_trading_config.json config/server_paper_config.json
        
        # 验证配置
        python3 -c "
import json
with open('config/server_paper_config.json', 'r') as f:
    config = json.load(f)
    
print('✅ 配置验证成功')
print(f'初始资金: \${{config[\"trading\"][\"initial_capital\"]}}')
print(f'交易币种: {{config[\"trading\"][\"symbols\"]}}')
print(f'最大持仓: {{config[\"trading\"][\"max_concurrent_positions\"]}}')
print(f'纸面交易: {{config[\"trading\"][\"paper_trading\"]}}')
"
        
        echo "✅ 配置验证完成"
EOF
    
    print_info "✅ 纸面交易配置完成"
}

# 创建启动脚本
create_startup_script() {
    print_info "创建服务器启动脚本..."
    
    ssh ${REMOTE_HOST} << EOF
        cd ${REMOTE_DIR}
        
        # 创建启动脚本
        cat > start_7day_paper_test.sh << 'SCRIPT_EOF'
#!/bin/bash

# DipMaster 7天纸面交易启动脚本

PROJECT_DIR="/opt/DipMaster-Trading-System"
cd \${PROJECT_DIR}

echo "🚀 启动DipMaster 7天纸面交易测试"
echo "时间: \$(date)"
echo "位置: \$(pwd)"

# 创建新的screen会话
screen -dmS dipmaster-paper bash -c "
    cd \${PROJECT_DIR}
    python3 run_paper_trading.py \\
        --config config/server_paper_config.json \\
        --hours 168 \\
        --log-level INFO
"

echo "✅ 纸面交易已在screen会话'dipmaster-paper'中启动"
echo "查看状态: screen -r dipmaster-paper"
echo "停止交易: screen -S dipmaster-paper -X quit"

# 显示启动状态
sleep 3
screen -list | grep dipmaster-paper
SCRIPT_EOF

        # 给脚本执行权限
        chmod +x start_7day_paper_test.sh
        
        echo "✅ 启动脚本创建完成"
EOF
    
    print_info "✅ 启动脚本创建完成"
}

# 创建监控脚本
create_monitoring_script() {
    print_info "创建监控脚本..."
    
    ssh ${REMOTE_HOST} << EOF
        cd ${REMOTE_DIR}
        
        # 创建监控脚本
        cat > monitor_paper_trading.sh << 'SCRIPT_EOF'
#!/bin/bash

# DipMaster 纸面交易监控脚本

PROJECT_DIR="/opt/DipMaster-Trading-System"
cd \${PROJECT_DIR}

echo "📊 DipMaster 纸面交易监控面板"
echo "=================================="
echo "时间: \$(date)"
echo ""

# 检查进程状态
if screen -list | grep -q dipmaster-paper; then
    echo "✅ 交易系统正在运行"
else
    echo "❌ 交易系统未运行"
fi

echo ""

# 显示最新日志
if [ -d logs ] && [ "\$(ls -A logs)" ]; then
    latest_log=\$(ls -t logs/paper_trading_*.log 2>/dev/null | head -1)
    if [ -n "\$latest_log" ]; then
        echo "📝 最新日志: \$latest_log"
        echo "最后10行:"
        echo "----------"
        tail -10 "\$latest_log"
    fi
fi

echo ""

# 系统资源
echo "💻 系统资源:"
echo "CPU: \$(top -bn1 | grep "Cpu(s)" | awk '{print \$2}' | cut -d'%' -f1)%"
echo "内存: \$(free -m | awk 'NR==2{printf "%.1f%%", \$3*100/\$2}')"
echo "磁盘: \$(df -h . | awk 'NR==2{print \$5}')"

echo ""
echo "命令快捷方式:"
echo "查看实时日志: tail -f logs/paper_trading_*.log"
echo "进入交易会话: screen -r dipmaster-paper"
echo "停止交易: screen -S dipmaster-paper -X quit"
SCRIPT_EOF

        # 给脚本执行权限
        chmod +x monitor_paper_trading.sh
        
        echo "✅ 监控脚本创建完成"
EOF
    
    print_info "✅ 监控脚本创建完成"
}

# 启动测试
start_paper_trading() {
    print_info "启动7天纸面交易测试..."
    
    ssh ${REMOTE_HOST} << EOF
        cd ${REMOTE_DIR}
        
        # 确保没有旧的会话
        screen -S dipmaster-paper -X quit 2>/dev/null || true
        
        # 等待一秒
        sleep 1
        
        # 启动测试
        ./start_7day_paper_test.sh
        
        echo ""
        echo "🎉 7天纸面交易测试已启动！"
        echo ""
EOF
    
    print_info "✅ 7天纸面交易测试已启动"
}

# 显示部署完成信息
show_deployment_info() {
    print_info "部署完成！"
    echo ""
    echo "🎯 7天纸面交易部署信息"
    echo "========================"
    echo "服务器: ${REMOTE_HOST}"
    echo "目录: ${REMOTE_DIR}"
    echo "配置: config/server_paper_config.json"
    echo ""
    echo "📊 监控命令:"
    echo "ssh ${REMOTE_USER}@${REMOTE_HOST} 'cd ${REMOTE_DIR} && ./monitor_paper_trading.sh'"
    echo ""
    echo "🔍 查看实时日志:"
    echo "ssh ${REMOTE_USER}@${REMOTE_HOST} 'cd ${REMOTE_DIR} && tail -f logs/paper_trading_*.log'"
    echo ""
    echo "📺 进入交易会话:"
    echo "ssh ${REMOTE_USER}@${REMOTE_HOST}"
    echo "cd ${REMOTE_DIR}"
    echo "screen -r dipmaster-paper"
    echo ""
    echo "⏹️  停止交易:"
    echo "ssh ${REMOTE_USER}@${REMOTE_HOST} 'screen -S dipmaster-paper -X quit'"
    echo ""
    echo "🎉 预计运行时间: 7天 (168小时)"
    echo "💰 初始资金: $10,000 (虚拟)"
    echo "📈 目标胜率: 75%+"
}

# 主执行流程
main() {
    echo -e "${BLUE}开始执行部署流程...${NC}"
    
    check_local_environment
    check_ssh_connection
    prepare_remote_environment
    sync_project_files
    install_dependencies
    configure_paper_trading
    create_startup_script
    create_monitoring_script
    start_paper_trading
    show_deployment_info
    
    echo -e "${GREEN}🎉 部署完成！DipMaster 7天纸面交易测试已启动${NC}"
}

# 错误处理
trap 'print_error "部署过程中发生错误，请检查上面的输出"; exit 1' ERR

# 执行主函数
main "$@"