#!/bin/bash

# DipMaster数据API服务启动脚本
# ================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# 检查Python版本
check_python() {
    log_info "检查Python版本..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3未安装"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    log_info "Python版本: $PYTHON_VERSION"
    
    if [[ $(echo "$PYTHON_VERSION < 3.11" | bc -l) -eq 1 ]]; then
        log_warn "推荐使用Python 3.11+，当前版本可能存在兼容性问题"
    fi
}

# 检查依赖包
check_dependencies() {
    log_info "检查依赖包..."
    
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt文件不存在"
        exit 1
    fi
    
    # 检查虚拟环境
    if [ -d "venv" ]; then
        log_info "使用现有虚拟环境"
        source venv/bin/activate
    else
        log_info "创建虚拟环境..."
        python3 -m venv venv
        source venv/bin/activate
    fi
    
    # 安装依赖
    log_info "安装依赖包..."
    pip install --upgrade pip
    pip install -r requirements.txt
}

# 检查配置
check_config() {
    log_info "检查配置..."
    
    # 检查环境变量
    required_vars=(
        "CLICKHOUSE_HOST"
        "KAFKA_BOOTSTRAP_SERVERS"
    )
    
    missing_vars=()
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            missing_vars+=("$var")
        fi
    done
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        log_warn "以下环境变量未设置，将使用默认值:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
    fi
    
    # 设置默认环境变量
    export CLICKHOUSE_HOST=${CLICKHOUSE_HOST:-"localhost"}
    export CLICKHOUSE_PORT=${CLICKHOUSE_PORT:-"9000"}
    export CLICKHOUSE_DATABASE=${CLICKHOUSE_DATABASE:-"dipmaster"}
    export CLICKHOUSE_USERNAME=${CLICKHOUSE_USERNAME:-"default"}
    export CLICKHOUSE_PASSWORD=${CLICKHOUSE_PASSWORD:-""}
    
    export KAFKA_BOOTSTRAP_SERVERS=${KAFKA_BOOTSTRAP_SERVERS:-"localhost:9092"}
    export KAFKA_GROUP_ID=${KAFKA_GROUP_ID:-"dipmaster-api"}
    
    export API_HOST=${API_HOST:-"0.0.0.0"}
    export API_PORT=${API_PORT:-"8000"}
    export API_DEBUG=${API_DEBUG:-"false"}
    export API_LOG_LEVEL=${API_LOG_LEVEL:-"INFO"}
    
    log_info "配置检查完成"
}

# 检查外部依赖
check_external_dependencies() {
    log_info "检查外部依赖..."
    
    # 检查ClickHouse连接
    log_debug "检查ClickHouse连接..."
    if ! timeout 5 bash -c "</dev/tcp/$CLICKHOUSE_HOST/$CLICKHOUSE_PORT" 2>/dev/null; then
        log_warn "无法连接到ClickHouse ($CLICKHOUSE_HOST:$CLICKHOUSE_PORT)"
        log_warn "请确保ClickHouse服务正在运行"
    else
        log_info "ClickHouse连接正常"
    fi
    
    # 检查Kafka连接
    log_debug "检查Kafka连接..."
    IFS=',' read -ra KAFKA_HOSTS <<< "$KAFKA_BOOTSTRAP_SERVERS"
    kafka_ok=false
    
    for host_port in "${KAFKA_HOSTS[@]}"; do
        IFS=':' read -ra HOST_PORT <<< "$host_port"
        host=${HOST_PORT[0]}
        port=${HOST_PORT[1]:-9092}
        
        if timeout 5 bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
            kafka_ok=true
            log_info "Kafka连接正常 ($host:$port)"
            break
        fi
    done
    
    if [ "$kafka_ok" = false ]; then
        log_warn "无法连接到Kafka ($KAFKA_BOOTSTRAP_SERVERS)"
        log_warn "请确保Kafka服务正在运行"
    fi
}

# 启动服务
start_service() {
    log_info "启动DipMaster数据API服务..."
    
    # 设置Python路径
    export PYTHONPATH="$(pwd):$PYTHONPATH"
    
    # 创建日志目录
    mkdir -p logs
    
    # 启动服务
    python3 main.py
}

# 显示帮助信息
show_help() {
    echo "DipMaster数据API服务启动脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  -c, --check-only    仅检查环境，不启动服务"
    echo "  -d, --development   开发模式启动"
    echo "  -p, --production    生产模式启动"
    echo "  --skip-deps         跳过依赖检查"
    echo "  --skip-external     跳过外部依赖检查"
    echo ""
    echo "环境变量:"
    echo "  CLICKHOUSE_HOST     ClickHouse主机地址 (默认: localhost)"
    echo "  CLICKHOUSE_PORT     ClickHouse端口 (默认: 9000)"
    echo "  KAFKA_BOOTSTRAP_SERVERS  Kafka服务器地址 (默认: localhost:9092)"
    echo "  API_HOST            API监听地址 (默认: 0.0.0.0)"
    echo "  API_PORT            API监听端口 (默认: 8000)"
    echo "  API_LOG_LEVEL       日志级别 (默认: INFO)"
    echo ""
    echo "示例:"
    echo "  $0                  使用默认配置启动"
    echo "  $0 -d               开发模式启动"
    echo "  $0 -c               仅检查环境"
    echo ""
}

# 解析命令行参数
CHECK_ONLY=false
DEVELOPMENT=false
PRODUCTION=false
SKIP_DEPS=false
SKIP_EXTERNAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--check-only)
            CHECK_ONLY=true
            shift
            ;;
        -d|--development)
            DEVELOPMENT=true
            export API_DEBUG=true
            export API_LOG_LEVEL=DEBUG
            shift
            ;;
        -p|--production)
            PRODUCTION=true
            export API_DEBUG=false
            export API_LOG_LEVEL=INFO
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-external)
            SKIP_EXTERNAL=true
            shift
            ;;
        *)
            log_error "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 主程序
main() {
    echo "=================================="
    echo "DipMaster数据API服务启动脚本"
    echo "=================================="
    echo ""
    
    # 检查Python
    check_python
    
    # 检查依赖
    if [ "$SKIP_DEPS" = false ]; then
        check_dependencies
    else
        log_warn "跳过依赖检查"
    fi
    
    # 检查配置
    check_config
    
    # 检查外部依赖
    if [ "$SKIP_EXTERNAL" = false ]; then
        check_external_dependencies
    else
        log_warn "跳过外部依赖检查"
    fi
    
    if [ "$CHECK_ONLY" = true ]; then
        log_info "环境检查完成，退出"
        exit 0
    fi
    
    # 启动服务
    start_service
}

# 信号处理
trap 'log_info "收到中断信号，停止服务..."; exit 0' INT TERM

# 运行主程序
main "$@"