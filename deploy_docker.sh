#!/bin/bash
# ============================================================================
# DipMaster Trading System - Docker 一键部署脚本
# Version: 1.0.0
# Date: 2025-08-18
# 
# 适用于阿里云服务器的Docker快速部署方案
# 服务器配置: Alibaba Cloud Linux 3.2104 LTS | 2vCPU | 4GB RAM
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

# 配置变量
PROJECT_NAME="dipmaster-trading"
PROJECT_DIR="/opt/${PROJECT_NAME}"
COMPOSE_FILE="docker-compose.yml"
CONFIG_FILE="config/paper_trading_config.json"

log_info "🐳 DipMaster Trading System - Docker一键部署"
log_info "服务器: 阿里云 | 配置: 2vCPU 4GB RAM | 系统: Alibaba Cloud Linux 3.2104"

# ============================================================================
# 第1步: 检查系统环境
# ============================================================================
log_step "第1步: 检查系统环境和依赖"

# 检查操作系统
if [ -f /etc/os-release ]; then
    . /etc/os-release
    log_info "操作系统: $PRETTY_NAME"
else
    log_warn "无法检测操作系统版本"
fi

# 检查是否为root用户
if [ "$EUID" -ne 0 ]; then
    log_error "请使用root用户运行此脚本"
    exit 1
fi

# 检查内存大小
TOTAL_MEM=$(free -m | awk 'NR==2{printf "%.0f", $2}')
if [ "$TOTAL_MEM" -lt 3500 ]; then
    log_warn "系统内存较小 (${TOTAL_MEM}MB)，建议至少4GB RAM"
fi

log_info "系统内存: ${TOTAL_MEM}MB"
log_info "系统架构: $(uname -m)"

# ============================================================================
# 第2步: 安装Docker和Docker Compose
# ============================================================================
log_step "第2步: 安装Docker和Docker Compose"

# 检查Docker是否已安装
if ! command -v docker &> /dev/null; then
    log_info "正在安装Docker..."
    
    # 安装依赖
    yum install -y yum-utils device-mapper-persistent-data lvm2
    
    # 添加Docker官方仓库
    yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    
    # 安装Docker
    yum install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    
    # 启动并启用Docker服务
    systemctl enable docker
    systemctl start docker
    
    log_info "Docker安装完成"
else
    log_info "Docker已安装: $(docker --version)"
fi

# 检查Docker服务状态
if ! systemctl is-active --quiet docker; then
    log_warn "Docker服务未运行，正在启动..."
    systemctl start docker
fi

# 检查Docker Compose
if ! command -v docker-compose &> /dev/null; then
    log_info "正在安装Docker Compose..."
    
    # 下载并安装Docker Compose
    curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    
    log_info "Docker Compose安装完成"
else
    log_info "Docker Compose已安装: $(docker-compose --version)"
fi

# ============================================================================
# 第3步: 创建项目目录和必要文件
# ============================================================================
log_step "第3步: 准备项目环境"

# 创建项目目录
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# 创建必要的子目录
mkdir -p logs data config backup monitoring

# 创建监控配置目录
log_info "创建Prometheus配置..."
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  - job_name: 'dipmaster-trading'
    static_configs:
      - targets: ['dipmaster-trading:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s
  
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-cache:6379']
EOF

log_info "创建Fluentd配置..."
cat > monitoring/fluentd.conf << 'EOF'
<source>
  @type tail
  path /app/logs/*.log
  pos_file /var/log/fluentd/dipmaster.log.pos
  tag dipmaster.logs
  <parse>
    @type none
  </parse>
</source>

<match dipmaster.logs>
  @type stdout
</match>
EOF

# ============================================================================
# 第4步: 处理项目代码
# ============================================================================
log_step "第4步: 准备项目代码"

# 检查是否有项目代码
if [ ! -f "main.py" ]; then
    log_warn "项目代码不存在，请选择部署方式:"
    echo "1. 从本地上传代码"
    echo "2. 从Git仓库克隆" 
    echo "3. 跳过（手动上传后重新运行）"
    
    read -p "请选择 [1-3]: " code_choice
    
    case $code_choice in
        1)
            log_info "请使用以下命令从本地上传代码:"
            log_info "scp -r /path/to/DipMaster-Trading-System/* root@$(curl -s ifconfig.me):$PROJECT_DIR/"
            log_warn "上传完成后请重新运行此脚本"
            exit 0
            ;;
        2)
            read -p "请输入Git仓库URL: " git_url
            if [ ! -z "$git_url" ]; then
                git clone $git_url temp_repo
                mv temp_repo/* .
                rm -rf temp_repo
                log_info "代码克隆完成"
            else
                log_error "Git仓库URL不能为空"
                exit 1
            fi
            ;;
        3)
            log_warn "跳过代码部署，请手动上传后重新运行"
            exit 0
            ;;
    esac
else
    log_info "项目代码已存在"
fi

# ============================================================================
# 第5步: 创建配置文件
# ============================================================================
log_step "第5步: 创建交易配置文件"

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    log_info "创建默认纸面交易配置..."
    
    cat > $CONFIG_FILE << 'EOF'
{
  "strategy_name": "DipMaster_Docker_Server",
  "version": "1.0.0",
  "description": "DipMaster Docker服务器部署配置",
  "created_date": "2025-08-18",
  
  "trading": {
    "paper_trading": true,
    "initial_capital": 10000,
    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"],
    "max_concurrent_positions": 3,
    "position_size_usd": 800,
    "min_position_size_usd": 300,
    "max_position_size_usd": 1200
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
      "max_drawdown_percent": 8.0,
      "position_size_limit_percent": 25
    },
    "circuit_breakers": {
      "daily_loss_limit": true,
      "drawdown_limit": true,
      "consecutive_loss_limit": 4
    }
  },
  
  "logging_and_monitoring": {
    "log_level": "INFO",
    "detailed_trade_logging": true,
    "dashboard_enabled": true,
    "save_results": true
  },
  
  "deployment_settings": {
    "server_mode": true,
    "auto_restart": true,
    "health_check_interval": 300,
    "max_memory_usage_mb": 2048
  },
  
  "docker_settings": {
    "container_mode": true,
    "redis_host": "redis-cache",
    "redis_port": 6379,
    "monitoring_enabled": true
  }
}
EOF

    log_warn "⚠️  配置文件已创建，请编辑API密钥:"
    log_warn "   vim $CONFIG_FILE"
    log_warn "   将 YOUR_API_KEY_HERE 和 YOUR_API_SECRET_HERE 替换为实际值"
else
    log_info "配置文件已存在: $CONFIG_FILE"
fi

# ============================================================================
# 第6步: 构建和启动Docker容器
# ============================================================================
log_step "第6步: 构建和启动Docker服务"

# 检查docker-compose.yml是否存在
if [ ! -f "$COMPOSE_FILE" ]; then
    log_error "docker-compose.yml 文件不存在"
    log_error "请确保项目代码已完整上传"
    exit 1
fi

# 停止现有服务（如果有）
log_info "停止现有Docker服务（如果有）..."
docker-compose down --remove-orphans 2>/dev/null || true

# 清理旧的镜像（可选）
read -p "是否清理旧的Docker镜像? [y/N]: " cleanup_choice
if [[ $cleanup_choice =~ ^[Yy]$ ]]; then
    log_info "清理Docker镜像..."
    docker system prune -f
    docker image prune -f
fi

# 构建镜像
log_info "构建DipMaster Docker镜像..."
docker-compose build --no-cache

# 启动服务
log_info "启动Docker Compose服务..."
docker-compose up -d

# 等待服务启动
log_info "等待服务启动..."
sleep 30

# ============================================================================
# 第7步: 验证部署
# ============================================================================
log_step "第7步: 验证部署状态"

# 检查容器状态
log_info "检查容器运行状态..."
docker-compose ps

# 检查服务健康状态
log_info "等待健康检查..."
sleep 60

# 显示服务状态
echo
echo "=== 🔍 服务状态检查 ==="

# 检查DipMaster主服务
if curl -f -s http://localhost:8080/health >/dev/null 2>&1; then
    log_info "✅ DipMaster交易系统: 运行正常"
else
    log_warn "❌ DipMaster交易系统: 可能未就绪"
fi

# 检查Prometheus监控
if curl -f -s http://localhost:9090 >/dev/null 2>&1; then
    log_info "✅ Prometheus监控: 运行正常"
else
    log_warn "❌ Prometheus监控: 可能未就绪"
fi

# 检查cAdvisor
if curl -f -s http://localhost:8081 >/dev/null 2>&1; then
    log_info "✅ cAdvisor监控: 运行正常"
else
    log_warn "❌ cAdvisor监控: 可能未就绪"
fi

# ============================================================================
# 第8步: 配置防火墙
# ============================================================================
log_step "第8步: 配置防火墙规则"

# 检查防火墙服务
if systemctl is-active --quiet firewalld; then
    log_info "配置防火墙规则..."
    
    # 开放必要端口
    firewall-cmd --permanent --add-port=8080/tcp    # DipMaster Web UI
    firewall-cmd --permanent --add-port=9090/tcp    # Prometheus
    firewall-cmd --permanent --add-port=8081/tcp    # cAdvisor
    firewall-cmd --permanent --add-port=22/tcp      # SSH
    
    firewall-cmd --reload
    log_info "防火墙配置完成"
    
    # 显示开放的端口
    log_info "已开放端口: $(firewall-cmd --list-ports)"
else
    log_info "防火墙未启用，跳过配置"
fi

# ============================================================================
# 第9步: 创建管理脚本
# ============================================================================
log_step "第9步: 创建系统管理脚本"

# 创建服务管理脚本
cat > /usr/local/bin/dipmaster-docker.sh << 'EOF'
#!/bin/bash
# DipMaster Docker 服务管理脚本

PROJECT_DIR="/opt/dipmaster-trading"
COMPOSE_FILE="docker-compose.yml"

cd $PROJECT_DIR

case "$1" in
    start)
        echo "启动DipMaster Docker服务..."
        docker-compose up -d
        ;;
    stop)
        echo "停止DipMaster Docker服务..."
        docker-compose down
        ;;
    restart)
        echo "重启DipMaster Docker服务..."
        docker-compose down
        docker-compose up -d
        ;;
    status)
        echo "=== DipMaster Docker服务状态 ==="
        docker-compose ps
        echo
        echo "=== 容器资源使用 ==="
        docker stats --no-stream
        ;;
    logs)
        echo "查看DipMaster服务日志..."
        docker-compose logs -f dipmaster-trading
        ;;
    update)
        echo "更新DipMaster Docker镜像..."
        docker-compose down
        docker-compose build --no-cache
        docker-compose up -d
        ;;
    cleanup)
        echo "清理Docker资源..."
        docker-compose down --rmi all --volumes --remove-orphans
        docker system prune -f
        ;;
    backup)
        echo "备份DipMaster数据..."
        BACKUP_FILE="/backup/dipmaster-$(date +%Y%m%d_%H%M%S).tar.gz"
        mkdir -p /backup
        tar -czf $BACKUP_FILE -C $PROJECT_DIR logs/ data/ config/
        echo "备份完成: $BACKUP_FILE"
        ;;
    *)
        echo "用法: $0 {start|stop|restart|status|logs|update|cleanup|backup}"
        echo "  start    - 启动服务"
        echo "  stop     - 停止服务"  
        echo "  restart  - 重启服务"
        echo "  status   - 查看状态"
        echo "  logs     - 查看日志"
        echo "  update   - 更新镜像"
        echo "  cleanup  - 清理资源"
        echo "  backup   - 备份数据"
        exit 1
        ;;
esac
EOF

chmod +x /usr/local/bin/dipmaster-docker.sh

# 创建监控脚本
cat > /usr/local/bin/dipmaster-monitor.sh << 'EOF'
#!/bin/bash
# DipMaster Docker 监控脚本

echo "=== DipMaster Trading System Docker 监控 ==="
echo "时间: $(date)"
echo

# 服务状态
echo "🐳 Docker服务状态:"
systemctl is-active docker && echo "  ✅ Docker: 运行中" || echo "  ❌ Docker: 未运行"

# 容器状态
echo
echo "📦 容器状态:"
cd /opt/dipmaster-trading
docker-compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

# 资源使用
echo
echo "💻 资源使用:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# 网络连接
echo
echo "🌐 网络状态:"
netstat -tuln | grep -E ":(8080|9090|8081|6379)" | while read line; do
    port=$(echo $line | awk '{print $4}' | cut -d: -f2)
    case $port in
        8080) echo "  ✅ DipMaster Web UI: 监听中" ;;
        9090) echo "  ✅ Prometheus: 监听中" ;;
        8081) echo "  ✅ cAdvisor: 监听中" ;;
        6379) echo "  ✅ Redis: 监听中" ;;
    esac
done

# 磁盘使用
echo
echo "💾 磁盘使用:"
df -h /opt/dipmaster-trading | awk 'NR==2{printf "  项目目录: %s 已用 %s (剩余 %s)\n", $5, $3, $4}'

# 最新日志
echo
echo "📋 最新日志 (最后5条):"
echo "----------------------------------------"
docker-compose logs --tail=5 dipmaster-trading 2>/dev/null | head -10
EOF

chmod +x /usr/local/bin/dipmaster-monitor.sh

# 创建自动备份脚本
cat > /usr/local/bin/dipmaster-backup.sh << 'EOF'
#!/bin/bash
# DipMaster Docker 自动备份脚本

BACKUP_DIR="/backup/dipmaster"
DATE=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR="/opt/dipmaster-trading"

mkdir -p $BACKUP_DIR

# 备份数据
echo "开始备份DipMaster数据..."
cd $PROJECT_DIR

# 备份配置和数据
tar -czf "$BACKUP_DIR/dipmaster_data_$DATE.tar.gz" config/ data/ logs/

# 备份Docker镜像
docker save dipmaster-trading:latest | gzip > "$BACKUP_DIR/dipmaster_image_$DATE.tar.gz"

# 清理30天前的备份
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete

echo "备份完成:"
ls -lh "$BACKUP_DIR/" | tail -5
EOF

chmod +x /usr/local/bin/dipmaster-backup.sh

# 添加定时备份任务
if ! crontab -l 2>/dev/null | grep -q "dipmaster-backup"; then
    (crontab -l 2>/dev/null; echo "0 3 * * * /usr/local/bin/dipmaster-backup.sh") | crontab -
    log_info "已添加自动备份任务（每日3:00）"
fi

# ============================================================================
# 部署完成总结
# ============================================================================
echo
log_info "🎉 DipMaster Trading System Docker部署完成！"
echo
echo "📋 部署信息总结:"
echo "  • 项目目录: $PROJECT_DIR"
echo "  • Docker镜像: dipmaster-trading:latest" 
echo "  • 配置文件: $CONFIG_FILE"
echo "  • 运行模式: 纸面交易 (Docker容器)"
echo
echo "🌐 访问地址:"
echo "  • DipMaster Web UI:   http://$(curl -s ifconfig.me):8080"
echo "  • Prometheus监控:     http://$(curl -s ifconfig.me):9090"
echo "  • cAdvisor监控:       http://$(curl -s ifconfig.me):8081"
echo
echo "🔧 管理命令:"
echo "  • 服务管理: dipmaster-docker.sh {start|stop|restart|status|logs}"
echo "  • 系统监控: dipmaster-monitor.sh"
echo "  • 数据备份: dipmaster-backup.sh"
echo
echo "📁 重要文件位置:"
echo "  • 日志目录: $PROJECT_DIR/logs/"
echo "  • 数据目录: $PROJECT_DIR/data/"
echo "  • 备份目录: /backup/dipmaster/"
echo
echo "⚠️  下一步操作:"
echo "  1. 编辑配置文件: vim $CONFIG_FILE"
echo "  2. 填入您的Binance API密钥"
echo "  3. 重启服务: dipmaster-docker.sh restart"
echo "  4. 监控运行: dipmaster-monitor.sh"
echo "  5. 查看日志: dipmaster-docker.sh logs"
echo
echo "🚨 重要提醒:"
echo "  • 首次运行建议使用纸面交易模式"
echo "  • 定期检查容器资源使用情况"
echo "  • 备份重要配置和交易数据"
echo "  • 确保API密钥安全，仅给予必要权限"
echo

log_info "Docker部署脚本执行完成 ✅"
log_info "系统将在后台持续运行，祝您交易顺利！ 🚀"