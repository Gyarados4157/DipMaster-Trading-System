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
COPY requirements_linux.txt .

# 升级pip并安装Python依赖
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements_linux.txt

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
RUN echo '#!/bin/bash' > /app/docker-entrypoint.sh && \
    echo 'set -e' >> /app/docker-entrypoint.sh && \
    echo '' >> /app/docker-entrypoint.sh && \
    echo '# 初始化目录权限' >> /app/docker-entrypoint.sh && \
    echo 'mkdir -p /app/logs /app/data /app/tmp' >> /app/docker-entrypoint.sh && \
    echo '' >> /app/docker-entrypoint.sh && \
    echo '# 检查配置文件' >> /app/docker-entrypoint.sh && \
    echo 'if [ ! -f "/app/config/paper_trading_config.json" ]; then' >> /app/docker-entrypoint.sh && \
    echo '    echo "⚠️  配置文件不存在，创建默认配置..."' >> /app/docker-entrypoint.sh && \
    echo '    mkdir -p /app/config' >> /app/docker-entrypoint.sh && \
    echo '    cat > /app/config/paper_trading_config.json << '\''CONFIGEOF'\''' >> /app/docker-entrypoint.sh && \
    echo '{' >> /app/docker-entrypoint.sh && \
    echo '  "trading": {' >> /app/docker-entrypoint.sh && \
    echo '    "paper_trading": true,' >> /app/docker-entrypoint.sh && \
    echo '    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],' >> /app/docker-entrypoint.sh && \
    echo '    "max_positions": 3,' >> /app/docker-entrypoint.sh && \
    echo '    "position_size_usd": 500' >> /app/docker-entrypoint.sh && \
    echo '  },' >> /app/docker-entrypoint.sh && \
    echo '  "exchange": {' >> /app/docker-entrypoint.sh && \
    echo '    "name": "binance",' >> /app/docker-entrypoint.sh && \
    echo '    "api_key": "YOUR_API_KEY_HERE",' >> /app/docker-entrypoint.sh && \
    echo '    "api_secret": "YOUR_API_SECRET_HERE",' >> /app/docker-entrypoint.sh && \
    echo '    "testnet": true' >> /app/docker-entrypoint.sh && \
    echo '  },' >> /app/docker-entrypoint.sh && \
    echo '  "strategy": {' >> /app/docker-entrypoint.sh && \
    echo '    "name": "DipMaster"' >> /app/docker-entrypoint.sh && \
    echo '  },' >> /app/docker-entrypoint.sh && \
    echo '  "logging": {' >> /app/docker-entrypoint.sh && \
    echo '    "level": "INFO"' >> /app/docker-entrypoint.sh && \
    echo '  }' >> /app/docker-entrypoint.sh && \
    echo '}' >> /app/docker-entrypoint.sh && \
    echo 'CONFIGEOF' >> /app/docker-entrypoint.sh && \
    echo 'fi' >> /app/docker-entrypoint.sh && \
    echo '' >> /app/docker-entrypoint.sh && \
    echo '# 打印启动信息' >> /app/docker-entrypoint.sh && \
    echo 'echo "🚀 DipMaster Trading System Docker Container"' >> /app/docker-entrypoint.sh && \
    echo 'echo "时间: $(date)"' >> /app/docker-entrypoint.sh && \
    echo 'echo "Python版本: $(python --version)"' >> /app/docker-entrypoint.sh && \
    echo 'echo "运行模式: 纸面交易"' >> /app/docker-entrypoint.sh && \
    echo '' >> /app/docker-entrypoint.sh && \
    echo '# 执行命令' >> /app/docker-entrypoint.sh && \
    echo 'exec "$@"' >> /app/docker-entrypoint.sh && \
    chmod +x /app/docker-entrypoint.sh

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
