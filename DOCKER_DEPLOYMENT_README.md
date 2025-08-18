# DipMaster Trading System - Docker部署指南

## 🚀 一键Docker部署（推荐）

### 方法1：直接下载部署脚本
```bash
curl -o docker_deploy.sh https://raw.githubusercontent.com/Gyarados4157/DipMaster-Trading-System/main/docker_deploy.sh && chmod +x docker_deploy.sh && sudo ./docker_deploy.sh
```

### 方法2：克隆仓库后部署
```bash
git clone https://github.com/Gyarados4157/DipMaster-Trading-System.git
cd DipMaster-Trading-System
sudo ./docker_deploy.sh
```

## 🎯 Docker部署优势

✅ **零依赖问题**: 不需要手动安装Python、pip或处理版本冲突  
✅ **一键启动**: 自动安装Docker、构建镜像、启动服务  
✅ **环境隔离**: 完全独立的运行环境，不影响系统  
✅ **简单管理**: 提供start/stop/monitor脚本  
✅ **资源控制**: 自动限制内存和CPU使用  
✅ **日志管理**: 自动日志轮转和持久化  

## 📋 系统要求

- **操作系统**: CentOS 7+, Ubuntu 18.04+, Debian 9+
- **内存**: 最低2GB，推荐4GB+
- **存储**: 最低5GB可用空间
- **权限**: root权限（安装Docker）
- **网络**: 能访问GitHub和Docker Hub

## 🔧 部署脚本功能

`docker_deploy.sh` 自动完成：

1. **系统检测**: 自动识别CentOS/Ubuntu系统
2. **Docker安装**: 安装Docker CE和Docker Compose
3. **项目下载**: 克隆最新代码到/opt目录
4. **镜像构建**: 构建DipMaster Docker镜像
5. **配置创建**: 生成默认配置文件
6. **服务启动**: 启动Docker容器
7. **脚本生成**: 创建管理脚本

## 📂 部署后目录结构

```
/opt/DipMaster-Trading-System/
├── config/
│   ├── paper_trading_config.json    # 主配置文件
│   └── .env                         # 环境变量
├── logs/                            # 日志目录（持久化）
├── data/                           # 数据目录（持久化）
├── results/                        # 结果目录（持久化）
├── docker-compose.simple.yml      # 简化版Docker配置
├── start_docker.sh                # 启动脚本
├── stop_docker.sh                 # 停止脚本
├── restart_docker.sh              # 重启脚本
└── monitor_docker.sh              # 监控脚本
```

## ⚙️ 配置设置

### 1. 编辑主配置文件
```bash
vi /opt/DipMaster-Trading-System/config/paper_trading_config.json
```

### 2. 关键配置项
```json
{
    "trading": {
        "paper_trading": true,           # 纸面交易模式
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "position_size_usd": 500
    },
    "exchange": {
        "api_key": "your_api_key_here",      # 必须填入真实API密钥
        "api_secret": "your_api_secret_here", # 必须填入真实API密码
        "testnet": true                       # 使用测试网
    }
}
```

### 3. 重启应用配置
```bash
cd /opt/DipMaster-Trading-System
./restart_docker.sh
```

## 🎮 Docker管理命令

### 基础管理
```bash
cd /opt/DipMaster-Trading-System

# 启动服务
./start_docker.sh

# 停止服务
./stop_docker.sh

# 重启服务
./restart_docker.sh

# 查看状态
./monitor_docker.sh
```

### Docker原生命令
```bash
# 查看容器状态
docker-compose -f docker-compose.simple.yml ps

# 查看实时日志
docker-compose -f docker-compose.simple.yml logs -f

# 进入容器
docker exec -it dipmaster-trading bash

# 重新构建镜像
docker-compose -f docker-compose.simple.yml build

# 查看资源使用
docker stats dipmaster-trading
```

## 📊 监控和访问

### Web界面
- **交易监控面板**: http://你的服务器IP:8080
- **容器状态**: `docker ps`
- **系统监控**: `./monitor_docker.sh`

### 日志查看
```bash
# 实时日志
docker-compose -f docker-compose.simple.yml logs -f dipmaster

# 最近100行日志
docker-compose -f docker-compose.simple.yml logs --tail=100 dipmaster

# 容器内日志文件
docker exec -it dipmaster-trading tail -f /app/logs/dipmaster.log
```

## 🔍 故障排除

### 常见问题

**1. Docker服务未启动**
```bash
sudo systemctl start docker
sudo systemctl enable docker
```

**2. 镜像构建失败**
```bash
# 清理Docker缓存
docker system prune -f

# 重新构建
cd /opt/DipMaster-Trading-System
docker-compose -f docker-compose.simple.yml build --no-cache
```

**3. 容器启动失败**
```bash
# 查看详细错误
docker-compose -f docker-compose.simple.yml logs dipmaster

# 检查配置文件
cat config/paper_trading_config.json

# 重启容器
./restart_docker.sh
```

**4. 端口被占用**
```bash
# 检查端口占用
netstat -tlnp | grep 8080

# 修改端口（编辑docker-compose.simple.yml）
vi docker-compose.simple.yml
# 将 "8080:8080" 改为 "8081:8080"
```

**5. API连接失败**
- 确认API密钥配置正确
- 检查网络连接
- 验证Binance API权限

### 完全重置
```bash
# 停止并删除所有容器
docker-compose -f docker-compose.simple.yml down

# 删除镜像
docker rmi dipmaster-trading:latest

# 重新部署
sudo ./docker_deploy.sh
```

## 🔄 更新和升级

### 更新代码
```bash
cd /opt/DipMaster-Trading-System
git pull origin main
docker-compose -f docker-compose.simple.yml build
./restart_docker.sh
```

### 备份数据
```bash
# 备份配置和数据
tar -czf dipmaster_backup_$(date +%Y%m%d).tar.gz config/ logs/ data/ results/
```

## 🔒 安全建议

1. **防火墙设置**
   ```bash
   # 只允许必要端口
   ufw allow 22    # SSH
   ufw allow 8080  # 监控面板
   ufw enable
   ```

2. **API安全**
   - 使用测试网API密钥
   - 限制API权限为仅交易
   - 定期轮换密钥

3. **容器安全**
   - 定期更新基础镜像
   - 限制容器资源使用
   - 监控容器行为

## 📈 性能优化

### 资源调整
编辑 `docker-compose.simple.yml`：
```yaml
deploy:
  resources:
    limits:
      memory: 2G      # 增加内存限制
      cpus: '1.0'     # 增加CPU限制
```

### 日志管理
```bash
# 清理老日志
docker system prune -f

# 配置日志轮转（在docker-compose.simple.yml中）
logging:
  options:
    max-size: "10m"
    max-file: "3"
```

## 🆚 对比其他部署方式

| 部署方式 | 优点 | 缺点 | 适用场景 |
|---------|------|------|----------|
| **Docker** | 环境隔离、一键部署、无依赖冲突 | 需要学习Docker | **推荐所有场景** |
| 源码部署 | 直接控制、资源占用少 | 依赖复杂、环境问题多 | 开发环境 |
| 虚拟环境 | 环境隔离 | 仍有系统依赖问题 | 小规模测试 |

## 📞 技术支持

遇到问题请提供：
1. 操作系统信息：`cat /etc/os-release`
2. Docker版本：`docker --version`
3. 容器状态：`docker ps -a`
4. 错误日志：`docker-compose logs dipmaster`

---

**推荐：Docker部署是最简单可靠的方式，避免了所有环境配置问题！**