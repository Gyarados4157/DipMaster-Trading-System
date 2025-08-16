#!/usr/bin/env python3
"""
DipMaster Dashboard API 启动脚本
支持开发模式、生产模式、Docker模式
"""

import sys
import os
import argparse
import subprocess
import signal
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('dashboard_startup.log')
        ]
    )
    return logging.getLogger(__name__)

def check_dependencies():
    """检查依赖服务"""
    logger = logging.getLogger(__name__)
    
    services = {
        'ClickHouse': ('localhost', 9000),
        'Redis': ('localhost', 6379),
        'Kafka': ('localhost', 9092)
    }
    
    import socket
    
    for service, (host, port) in services.items():
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                logger.info(f"✓ {service} 连接成功 ({host}:{port})")
            else:
                logger.warning(f"✗ {service} 连接失败 ({host}:{port})")
                
        except Exception as e:
            logger.error(f"✗ {service} 检查失败: {e}")

def start_development_mode(args):
    """启动开发模式"""
    logger = logging.getLogger(__name__)
    logger.info("启动开发模式...")
    
    # 检查依赖
    check_dependencies()
    
    # 启动命令
    cmd = [
        sys.executable, "main.py",
        "--config", args.config,
        "--host", args.host,
        "--port", str(args.port),
        "--reload"
    ]
    
    if args.debug:
        cmd.extend(["--log-level", "DEBUG"])
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        process = subprocess.Popen(cmd)
        
        def signal_handler(sig, frame):
            logger.info("收到停止信号，正在关闭...")
            process.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        process.wait()
        
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭...")
        process.terminate()
    except Exception as e:
        logger.error(f"启动失败: {e}")
        sys.exit(1)

def start_production_mode(args):
    """启动生产模式"""
    logger = logging.getLogger(__name__)
    logger.info("启动生产模式...")
    
    # 检查依赖
    check_dependencies()
    
    # 生产模式使用gunicorn
    cmd = [
        "gunicorn",
        "main:app",
        "--bind", f"{args.host}:{args.port}",
        "--workers", str(args.workers),
        "--worker-class", "uvicorn.workers.UvicornWorker",
        "--access-logfile", "logs/access.log",
        "--error-logfile", "logs/error.log",
        "--log-level", "info",
        "--preload"
    ]
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"生产模式启动失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭...")

def start_docker_mode(args):
    """启动Docker模式"""
    logger = logging.getLogger(__name__)
    logger.info("启动Docker容器...")
    
    # 构建Docker镜像
    if args.build:
        logger.info("构建Docker镜像...")
        subprocess.run([
            "docker", "build", "-t", "dipmaster-dashboard:latest", "."
        ], check=True)
    
    # 启动Docker Compose
    cmd = ["docker-compose", "up"]
    
    if args.detach:
        cmd.append("-d")
    
    if args.build:
        cmd.append("--build")
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Docker启动失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("用户中断，正在关闭...")

def stop_docker_mode():
    """停止Docker模式"""
    logger = logging.getLogger(__name__)
    logger.info("停止Docker容器...")
    
    try:
        subprocess.run(["docker-compose", "down"], check=True)
        logger.info("Docker容器已停止")
    except subprocess.CalledProcessError as e:
        logger.error(f"停止Docker失败: {e}")

def create_directories():
    """创建必要的目录"""
    directories = ["logs", "data", "config"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DipMaster Dashboard API 启动脚本")
    
    # 通用参数
    parser.add_argument("--config", default="config/dashboard_config.json", help="配置文件路径")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--debug", action="store_true", help="开启调试模式")
    
    # 子命令
    subparsers = parser.add_subparsers(dest="mode", help="启动模式")
    
    # 开发模式
    dev_parser = subparsers.add_parser("dev", help="开发模式")
    
    # 生产模式
    prod_parser = subparsers.add_parser("prod", help="生产模式")
    prod_parser.add_argument("--workers", type=int, default=4, help="工作进程数")
    
    # Docker模式
    docker_parser = subparsers.add_parser("docker", help="Docker模式")
    docker_parser.add_argument("--build", action="store_true", help="重新构建镜像")
    docker_parser.add_argument("--detach", action="store_true", help="后台运行")
    
    # 停止Docker
    stop_parser = subparsers.add_parser("stop", help="停止Docker容器")
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging()
    
    # 创建目录
    create_directories()
    
    # 根据模式启动
    if args.mode == "dev" or args.mode is None:
        start_development_mode(args)
    elif args.mode == "prod":
        start_production_mode(args)
    elif args.mode == "docker":
        start_docker_mode(args)
    elif args.mode == "stop":
        stop_docker_mode()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()