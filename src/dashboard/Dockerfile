# DipMaster Dashboard API Dockerfile

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建日志目录
RUN mkdir -p /app/logs

# 设置环境变量
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/config/dashboard_config.json

# 暴露端口
EXPOSE 8080 9090

# 健康检查
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# 创建非root用户
RUN useradd --create-home --shell /bin/bash dipmaster
RUN chown -R dipmaster:dipmaster /app
USER dipmaster

# 启动命令
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8080"]