# ============================================================================
# DipMaster Trading System - Docker Image
# Version: 1.0.0
# Date: 2025-08-18
# 
# 多阶段构建优化的生产环境Docker镜像
# 基于Debian slim，兼容性好
# ============================================================================

# 阶段1: 构建阶段
FROM python:3.11-slim-bookworm AS builder

# 元数据标签
LABEL maintainer="DipMaster Trading Team"
LABEL version="1.0.0"
LABEL description="DipMaster Trading System - Crypto Trading Bot"

# 设置环境变量
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONFAULTHANDLER=1

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libssl-dev \
    libffi-dev \
    pkg-config \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 升级pip并安装Python依赖
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# 阶段2: 生产运行阶段
FROM python:3.11-slim-bookworm AS production

# 设置环境变量
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TZ=Asia/Hong_Kong

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    tzdata \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 设置时区
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 创建非root用户
RUN useradd -u 1001 -m -s /bin/bash dipmaster

# 设置工作目录
WORKDIR /app

# 从构建阶段复制Python环境
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 复制应用代码
COPY --chown=dipmaster:dipmaster . .

# 创建必要的目录
RUN mkdir -p /app/logs /app/data /app/tmp /app/backup && \
    chown -R dipmaster:dipmaster /app

# 创建启动脚本
RUN cat > /app/docker-entrypoint.sh << 'EOF' && \
cat > /app/docker-entrypoint.sh << 'EOF'
#!/bin/bash
set -e

# 初始化目录权限
mkdir -p /app/logs /app/data /app/tmp
chown -R dipmaster:dipmaster /app/logs /app/data /app/tmp

# 检查配置文件
if [ ! -f "/app/config/paper_trading_config.json" ]; then
    echo "⚠️  配置文件不存在，创建默认配置..."
    mkdir -p /app/config
    cat > /app/config/paper_trading_config.json << 'CONFIGEOF'
{
  "strategy_name": "DipMaster_Docker_Paper",
  "trading": {
    "paper_trading": true,
    "initial_capital": 10000,
    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
  },
  "api": {
    "exchange": "binance",
    "api_key": "YOUR_API_KEY_HERE", 
    "api_secret": "YOUR_API_SECRET_HERE",
    "testnet": true,
    "paper_mode": true
  },
  "logging_and_monitoring": {
    "log_level": "INFO",
    "dashboard_enabled": true
  }
}
CONFIGEOF
    chown dipmaster:dipmaster /app/config/paper_trading_config.json
fi

# 打印启动信息
echo "🚀 DipMaster Trading System Docker Container"
echo "时间: $(date)"
echo "用户: $(whoami)"
echo "工作目录: $(pwd)"
echo "Python版本: $(python --version)"
echo "配置文件: $1"
echo "运行模式: 纸面交易"

# 执行命令
exec "$@"
EOF

RUN chmod +x /app/docker-entrypoint.sh

# 切换到应用用户
USER dipmaster

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || python -c "import sys; sys.exit(0)"

# 暴露端口
EXPOSE 8080

# 设置启动点
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# 默认启动命令 (纸面交易模式)
CMD ["python", "main.py", "--config", "config/paper_trading_config.json", "--paper", "--log-level", "INFO"]
