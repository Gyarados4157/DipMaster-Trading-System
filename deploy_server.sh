#!/bin/bash
# ============================================================================
# DipMaster Trading System - 阿里云服务器自动部署脚本
# Version: 1.0.0
# Date: 2025-08-18
# 
# 适用于: Alibaba Cloud Linux 3.2104 LTS 64位
# 配置: 2vCPU 4GB RAM
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${BLUE}[STEP]${NC} $1"; }

# 基础配置
PROJECT_NAME="dipmaster-trading"
WORK_DIR="/opt/${PROJECT_NAME}"
USER_NAME="dipmaster"
PYTHON_VERSION="3.11"
SERVICE_PORT="8080"

log_info "🚀 开始部署DipMaster交易系统到阿里云服务器"
log_info "服务器配置: Alibaba Cloud Linux 3.2104 LTS | 2vCPU | 4GB RAM"

# ============================================================================
# 第1步: 系统环境准备
# ============================================================================
log_step "第1步: 更新系统软件包"
sudo yum update -y
sudo yum install -y epel-release
sudo yum groupinstall -y "Development Tools"
sudo yum install -y git wget curl vim htop screen supervisor

# ============================================================================
# 第2步: 安装Python 3.11
# ============================================================================
log_step "第2步: 安装Python ${PYTHON_VERSION}"

# 检查Python版本
if command -v python3.11 &> /dev/null; then
    log_info "Python 3.11 已安装"
else
    log_info "正在安装Python 3.11..."
    sudo yum install -y python3.11 python3.11-pip python3.11-venv python3.11-devel
fi

# 设置Python别名
if ! grep -q "alias python3=python3.11" ~/.bashrc; then
    echo "alias python3=python3.11" >> ~/.bashrc
    echo "alias pip3=pip3.11" >> ~/.bashrc
    source ~/.bashrc
fi

# ============================================================================
# 第3步: 创建系统用户和目录
# ============================================================================
log_step "第3步: 创建系统用户和工作目录"

# 创建专用用户
if ! id "$USER_NAME" &>/dev/null; then
    sudo useradd -m -s /bin/bash $USER_NAME
    log_info "创建用户: $USER_NAME"
else
    log_info "用户已存在: $USER_NAME"
fi

# 创建工作目录
sudo mkdir -p $WORK_DIR
sudo chown $USER_NAME:$USER_NAME $WORK_DIR

# 创建必要的子目录
sudo -u $USER_NAME mkdir -p $WORK_DIR/{logs,data,config,tmp}

# ============================================================================
# 第4步: 安装Docker (可选)
# ============================================================================
log_step "第4步: 安装Docker"

if command -v docker &> /dev/null; then
    log_info "Docker 已安装"
else
    log_info "正在安装Docker..."
    sudo yum install -y yum-utils
    sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    sudo yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    sudo systemctl enable docker
    sudo systemctl start docker
    
    # 添加用户到docker组
    sudo usermod -aG docker $USER_NAME
    sudo usermod -aG docker $USER
fi

# ============================================================================
# 第5步: 克隆或上传项目代码
# ============================================================================
log_step "第5步: 部署项目代码"

# 如果工作目录为空，则需要从GitHub克隆或本地上传
if [ ! -f "$WORK_DIR/main.py" ]; then
    log_warn "项目代码不存在，请选择部署方式:"
    echo "1. 从本地上传代码"
    echo "2. 从Git仓库克隆"
    echo "3. 跳过（手动上传）"
    
    read -p "请选择 [1-3]: " deploy_choice
    
    case $deploy_choice in
        1)
            log_info "请使用 scp 或 rsync 将代码上传到: $WORK_DIR"
            log_info "示例: scp -r /path/to/DipMaster-Trading-System/* user@$SERVER_IP:$WORK_DIR/"
            ;;
        2)
            read -p "请输入Git仓库URL: " git_url
            sudo -u $USER_NAME git clone $git_url $WORK_DIR
            ;;
        3)
            log_warn "跳过代码部署，请手动上传代码"
            ;;
    esac
else
    log_info "项目代码已存在"
fi

# ============================================================================
# 第6步: 创建Python虚拟环境
# ============================================================================
log_step "第6步: 创建Python虚拟环境"

VENV_DIR="$WORK_DIR/venv"
if [ ! -d "$VENV_DIR" ]; then
    sudo -u $USER_NAME python3.11 -m venv $VENV_DIR
    log_info "创建虚拟环境: $VENV_DIR"
else
    log_info "虚拟环境已存在: $VENV_DIR"
fi

# 激活虚拟环境并安装依赖
if [ -f "$WORK_DIR/requirements.txt" ]; then
    log_info "安装Python依赖包..."
    sudo -u $USER_NAME bash -c "
        source $VENV_DIR/bin/activate
        pip install --upgrade pip setuptools wheel
        pip install -r $WORK_DIR/requirements.txt
    "
else
    log_warn "requirements.txt 不存在，跳过依赖安装"
fi

# ============================================================================
# 第7步: 配置防火墙
# ============================================================================
log_step "第7步: 配置防火墙规则"

# 检查防火墙服务
if systemctl is-active --quiet firewalld; then
    log_info "配置防火墙规则..."
    sudo firewall-cmd --permanent --add-port=$SERVICE_PORT/tcp
    sudo firewall-cmd --permanent --add-port=22/tcp  # SSH
    sudo firewall-cmd --reload
    log_info "防火墙规则已更新"
else
    log_info "防火墙未启用或不存在"
fi

# ============================================================================
# 第8步: 创建启动脚本
# ============================================================================
log_step "第8步: 创建系统启动脚本"

# 创建启动脚本
cat > /tmp/dipmaster_start.sh << 'EOF'
#!/bin/bash
# DipMaster Trading System 启动脚本

WORK_DIR="/opt/dipmaster-trading"
VENV_DIR="$WORK_DIR/venv"
PYTHON_SCRIPT="$WORK_DIR/main.py"
CONFIG_FILE="$WORK_DIR/config/paper_trading_config.json"
PID_FILE="/var/run/dipmaster.pid"
LOG_FILE="$WORK_DIR/logs/system.log"

# 确保日志目录存在
mkdir -p "$WORK_DIR/logs"

# 检查是否已运行
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "DipMaster已在运行 (PID: $PID)"
        exit 1
    else
        rm -f "$PID_FILE"
    fi
fi

echo "启动DipMaster Trading System..."
echo "工作目录: $WORK_DIR"
echo "配置文件: $CONFIG_FILE"
echo "日志文件: $LOG_FILE"

# 激活虚拟环境并启动
cd "$WORK_DIR"
source "$VENV_DIR/bin/activate"

# 纸面交易模式启动
nohup python3 "$PYTHON_SCRIPT" \
    --config "$CONFIG_FILE" \
    --paper \
    --log-level INFO \
    > "$LOG_FILE" 2>&1 &

# 保存PID
echo $! > "$PID_FILE"
echo "DipMaster已启动 (PID: $!)"
echo "查看日志: tail -f $LOG_FILE"
EOF

# 安装启动脚本
sudo mv /tmp/dipmaster_start.sh /usr/local/bin/dipmaster_start.sh
sudo chmod +x /usr/local/bin/dipmaster_start.sh

# 创建停止脚本
cat > /tmp/dipmaster_stop.sh << 'EOF'
#!/bin/bash
# DipMaster Trading System 停止脚本

PID_FILE="/var/run/dipmaster.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "停止DipMaster Trading System (PID: $PID)..."
        kill -TERM $PID
        
        # 等待优雅关闭
        for i in {1..10}; do
            if ! ps -p $PID > /dev/null 2>&1; then
                echo "DipMaster已成功停止"
                rm -f "$PID_FILE"
                exit 0
            fi
            sleep 1
        done
        
        # 强制关闭
        echo "强制停止DipMaster..."
        kill -KILL $PID
        rm -f "$PID_FILE"
        echo "DipMaster已强制停止"
    else
        echo "DipMaster未运行"
        rm -f "$PID_FILE"
    fi
else
    echo "PID文件不存在，DipMaster可能未运行"
fi
EOF

# 安装停止脚本
sudo mv /tmp/dipmaster_stop.sh /usr/local/bin/dipmaster_stop.sh
sudo chmod +x /usr/local/bin/dipmaster_stop.sh

# ============================================================================
# 第9步: 创建systemd服务
# ============================================================================
log_step "第9步: 创建systemd服务"

cat > /tmp/dipmaster.service << EOF
[Unit]
Description=DipMaster Trading System
After=network.target
Wants=network.target

[Service]
Type=forking
User=$USER_NAME
Group=$USER_NAME
WorkingDirectory=$WORK_DIR
Environment=PATH=$VENV_DIR/bin:/usr/bin:/bin
ExecStart=/usr/local/bin/dipmaster_start.sh
ExecStop=/usr/local/bin/dipmaster_stop.sh
PIDFile=/var/run/dipmaster.pid
Restart=always
RestartSec=10
StandardOutput=append:$WORK_DIR/logs/system.log
StandardError=append:$WORK_DIR/logs/system.log

# 资源限制
LimitNOFILE=65536
MemoryMax=3G
CPUQuota=180%

[Install]
WantedBy=multi-user.target
EOF

# 安装systemd服务
sudo mv /tmp/dipmaster.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable dipmaster.service

# ============================================================================
# 第10步: 创建监控脚本
# ============================================================================
log_step "第10步: 创建系统监控脚本"

cat > /tmp/monitor_dipmaster.sh << 'EOF'
#!/bin/bash
# DipMaster 系统监控脚本

WORK_DIR="/opt/dipmaster-trading"
LOG_FILE="$WORK_DIR/logs/system.log"
PID_FILE="/var/run/dipmaster.pid"

echo "=== DipMaster Trading System 监控面板 ==="
echo "时间: $(date)"
echo

# 检查服务状态
echo "🔍 服务状态:"
if systemctl is-active --quiet dipmaster; then
    echo "  ✅ systemd服务: 运行中"
else
    echo "  ❌ systemd服务: 未运行"
fi

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        echo "  ✅ 进程状态: 运行中 (PID: $PID)"
        
        # 显示进程信息
        echo "  📊 进程信息:"
        ps -p $PID -o pid,ppid,cmd,vsz,rss,pcpu,pmem,etime --no-headers | \
        awk '{printf "     PID: %s | 内存: %sMB | CPU: %s%% | 运行时间: %s\n", $1, int($5/1024), $6, $8}'
    else
        echo "  ❌ 进程状态: 未运行"
    fi
else
    echo "  ❌ PID文件不存在"
fi

# 检查端口监听
echo
echo "🌐 网络状态:"
if netstat -tuln | grep -q ":8080"; then
    echo "  ✅ 端口8080: 监听中"
else
    echo "  ❌ 端口8080: 未监听"
fi

# 系统资源使用情况
echo
echo "💻 系统资源:"
echo "  CPU使用率: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "  内存使用: $(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}')"
echo "  磁盘使用: $(df -h $WORK_DIR | awk 'NR==2{print $5}')"

# 显示最新日志
echo
echo "📋 最新日志 (最后10行):"
echo "----------------------------------------"
if [ -f "$LOG_FILE" ]; then
    tail -10 "$LOG_FILE"
else
    echo "日志文件不存在: $LOG_FILE"
fi

echo
echo "=== 监控完成 ==="
echo "💡 提示:"
echo "  - 实时日志: tail -f $LOG_FILE"
echo "  - 启动服务: sudo systemctl start dipmaster"
echo "  - 停止服务: sudo systemctl stop dipmaster"
echo "  - 查看状态: sudo systemctl status dipmaster"
EOF

# 安装监控脚本
sudo mv /tmp/monitor_dipmaster.sh /usr/local/bin/monitor_dipmaster.sh
sudo chmod +x /usr/local/bin/monitor_dipmaster.sh

# ============================================================================
# 第11步: 创建日志轮转配置
# ============================================================================
log_step "第11步: 配置日志轮转"

cat > /tmp/dipmaster-logs << EOF
$WORK_DIR/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $USER_NAME $USER_NAME
    postrotate
        systemctl reload dipmaster > /dev/null 2>&1 || true
    endscript
}
EOF

sudo mv /tmp/dipmaster-logs /etc/logrotate.d/
sudo chmod 644 /etc/logrotate.d/dipmaster-logs

# ============================================================================
# 第12步: 创建自动备份脚本
# ============================================================================
log_step "第12步: 创建自动备份脚本"

cat > /tmp/backup_dipmaster.sh << EOF
#!/bin/bash
# DipMaster 自动备份脚本

WORK_DIR="$WORK_DIR"
BACKUP_DIR="/backup/dipmaster"
DATE=\$(date +%Y%m%d_%H%M%S)

# 创建备份目录
mkdir -p \$BACKUP_DIR

# 备份配置文件
tar -czf "\$BACKUP_DIR/config_\$DATE.tar.gz" -C "\$WORK_DIR" config/

# 备份日志文件（最近7天）
find "\$WORK_DIR/logs" -name "*.log" -mtime -7 | \
tar -czf "\$BACKUP_DIR/logs_\$DATE.tar.gz" -T -

# 备份交易数据
if [ -d "\$WORK_DIR/data" ]; then
    tar -czf "\$BACKUP_DIR/data_\$DATE.tar.gz" -C "\$WORK_DIR" data/
fi

# 清理30天前的备份
find \$BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "备份完成: \$BACKUP_DIR/"
ls -lh "\$BACKUP_DIR/" | tail -5
EOF

sudo mv /tmp/backup_dipmaster.sh /usr/local/bin/backup_dipmaster.sh
sudo chmod +x /usr/local/bin/backup_dipmaster.sh

# 创建备份目录
sudo mkdir -p /backup/dipmaster
sudo chown $USER_NAME:$USER_NAME /backup/dipmaster

# 添加到crontab（每日备份）
if ! sudo crontab -u $USER_NAME -l 2>/dev/null | grep -q "backup_dipmaster"; then
    (sudo crontab -u $USER_NAME -l 2>/dev/null; echo "0 2 * * * /usr/local/bin/backup_dipmaster.sh") | sudo crontab -u $USER_NAME -
    log_info "已添加自动备份任务（每日2:00）"
fi

# ============================================================================
# 第13步: 创建配置模板
# ============================================================================
log_step "第13步: 创建配置文件模板"

# 如果配置目录不存在，创建模板配置
if [ ! -f "$WORK_DIR/config/paper_trading_config.json" ]; then
    log_info "创建纸面交易配置模板..."
    
    sudo -u $USER_NAME mkdir -p "$WORK_DIR/config"
    cat > /tmp/paper_trading_config.json << 'EOF'
{
  "strategy_name": "DipMaster_Paper_Trading_Server",
  "version": "1.0.0",
  "description": "DipMaster服务器纸面交易配置",
  "created_date": "2025-08-18",
  
  "trading": {
    "paper_trading": true,
    "initial_capital": 10000,
    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
    "max_concurrent_positions": 3,
    "position_size_usd": 800
  },
  
  "api": {
    "exchange": "binance",
    "api_key": "YOUR_API_KEY_HERE",
    "api_secret": "YOUR_API_SECRET_HERE",
    "testnet": true,
    "paper_mode": true
  },
  
  "risk_management": {
    "global_settings": {
      "max_daily_loss_usd": 300,
      "max_drawdown_percent": 8.0
    }
  },
  
  "logging_and_monitoring": {
    "log_level": "INFO",
    "dashboard_enabled": true
  },
  
  "deployment_settings": {
    "server_mode": true,
    "auto_restart": true,
    "health_check_interval": 300
  }
}
EOF
    
    sudo mv /tmp/paper_trading_config.json "$WORK_DIR/config/"
    sudo chown $USER_NAME:$USER_NAME "$WORK_DIR/config/paper_trading_config.json"
    
    log_warn "⚠️  请编辑配置文件并填入您的API密钥:"
    log_warn "   vim $WORK_DIR/config/paper_trading_config.json"
fi

# ============================================================================
# 部署完成
# ============================================================================
echo
log_info "🎉 DipMaster Trading System 部署完成！"
echo
echo "📋 部署信息:"
echo "  • 工作目录: $WORK_DIR"
echo "  • 系统用户: $USER_NAME"
echo "  • Python版本: $(python3.11 --version)"
echo "  • 服务端口: $SERVICE_PORT"
echo
echo "🚀 启动命令:"
echo "  • 启动服务: sudo systemctl start dipmaster"
echo "  • 停止服务: sudo systemctl stop dipmaster"
echo "  • 查看状态: sudo systemctl status dipmaster"
echo "  • 系统监控: monitor_dipmaster.sh"
echo
echo "📁 重要文件位置:"
echo "  • 配置文件: $WORK_DIR/config/paper_trading_config.json"
echo "  • 日志文件: $WORK_DIR/logs/"
echo "  • 启动脚本: /usr/local/bin/dipmaster_start.sh"
echo "  • 监控脚本: /usr/local/bin/monitor_dipmaster.sh"
echo
echo "⚠️  下一步操作:"
echo "  1. 上传项目代码到: $WORK_DIR"
echo "  2. 编辑配置文件并填入API密钥"
echo "  3. 启动服务: sudo systemctl start dipmaster"
echo "  4. 检查运行状态: monitor_dipmaster.sh"
echo
echo "🔧 故障排除:"
echo "  • 查看实时日志: tail -f $WORK_DIR/logs/system.log"
echo "  • 检查服务状态: journalctl -u dipmaster -f"
echo "  • 手动启动测试: cd $WORK_DIR && ./venv/bin/python main.py --paper"
echo

log_info "部署脚本执行完成 ✅"