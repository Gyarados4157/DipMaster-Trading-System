# 🚀 DipMaster Trading System - 阿里云服务器部署指南

## 服务器信息
- **服务器ID**: iZj6c0m028i65r1jyf6xwjZ
- **配置**: Alibaba Cloud Linux 3.2104 LTS 64位
- **规格**: 2核(vCPU) 4GB RAM
- **公网IP**: 47.239.1.232
- **地域**: 中国香港

## 🎯 一键部署命令

### 方法1: 直接在服务器上部署

1. **SSH登录服务器**:
```bash
ssh root@47.239.1.232
```

2. **下载并执行部署脚本**:
```bash
# 下载部署脚本
wget https://raw.githubusercontent.com/your-repo/DipMaster-Trading-System/main/deploy_server.sh
# 或者如果没有GitHub，手动创建脚本文件后执行：
chmod +x deploy_server.sh
./deploy_server.sh
```

3. **上传项目代码**:
```bash
# 从本地上传到服务器
scp -r /path/to/DipMaster-Trading-System/* root@47.239.1.232:/opt/dipmaster-trading/

# 或者使用rsync (推荐)
rsync -avz --progress /path/to/DipMaster-Trading-System/ root@47.239.1.232:/opt/dipmaster-trading/
```

### 方法2: 完整手动部署流程

SSH连接到服务器后，按以下步骤执行：

```bash
# 1. 更新系统
sudo yum update -y && sudo yum install -y epel-release git python3.11 python3.11-pip

# 2. 创建工作目录
sudo mkdir -p /opt/dipmaster-trading
cd /opt/dipmaster-trading

# 3. 上传项目代码（选择以下任一方式）
# 方式A: 从GitHub克隆 (如果有公开仓库)
git clone https://github.com/your-username/DipMaster-Trading-System.git .

# 方式B: 从本地上传 (在本地机器执行)
# scp -r /Users/zhangxuanyang/Desktop/Quant/DipMaster-Trading-System/* root@47.239.1.232:/opt/dipmaster-trading/

# 4. 创建Python虚拟环境
python3.11 -m venv venv
source venv/bin/activate

# 5. 安装依赖 (如果requirements.txt存在)
pip install --upgrade pip
pip install -r requirements.txt

# 6. 配置API密钥
cp config/paper_trading_config.json config/my_trading_config.json
vim config/my_trading_config.json  # 编辑API密钥

# 7. 测试运行
python main.py --config config/my_trading_config.json --paper --log-level INFO

# 8. 设置开机自启 (可选)
# 参考下面的systemd服务配置
```

## ⚙️ 配置API密钥

1. **编辑配置文件**:
```bash
vim /opt/dipmaster-trading/config/paper_trading_config.json
```

2. **修改API配置**:
```json
{
  "api": {
    "exchange": "binance",
    "api_key": "您的Binance API Key",
    "api_secret": "您的Binance API Secret", 
    "testnet": true,
    "paper_mode": true
  }
}
```

3. **重要安全提示**:
- 确保API密钥仅有交易权限，禁用提现权限
- 建议使用Binance测试网进行纸面交易
- 配置文件权限: `chmod 600 config/paper_trading_config.json`

## 🔧 系统服务配置 (可选但推荐)

### 创建systemd服务文件

```bash
sudo tee /etc/systemd/system/dipmaster.service << EOF
[Unit]
Description=DipMaster Trading System
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/dipmaster-trading
Environment=PATH=/opt/dipmaster-trading/venv/bin:/usr/bin:/bin
ExecStart=/opt/dipmaster-trading/venv/bin/python main.py --config config/paper_trading_config.json --paper --log-level INFO
Restart=always
RestartSec=10
StandardOutput=append:/opt/dipmaster-trading/logs/system.log
StandardError=append:/opt/dipmaster-trading/logs/error.log

[Install]
WantedBy=multi-user.target
EOF
```

### 启用并启动服务

```bash
# 重载systemd配置
sudo systemctl daemon-reload

# 启用开机自启
sudo systemctl enable dipmaster

# 启动服务
sudo systemctl start dipmaster

# 查看状态
sudo systemctl status dipmaster

# 查看实时日志
journalctl -u dipmaster -f
```

## 📊 监控和日志

### 日志文件位置
```bash
# 主日志
tail -f /opt/dipmaster-trading/logs/system.log

# 错误日志
tail -f /opt/dipmaster-trading/logs/error.log

# 交易日志
tail -f /opt/dipmaster-trading/logs/dipmaster_$(date +%Y%m%d).log
```

### 监控脚本
```bash
# 创建监控脚本
cat > /usr/local/bin/dipmaster_status.sh << 'EOF'
#!/bin/bash
echo "=== DipMaster Status ==="
echo "Service: $(systemctl is-active dipmaster)"
echo "Process: $(pgrep -f 'python.*main.py' | wc -l) processes"
echo "Memory: $(ps aux | grep 'python.*main.py' | grep -v grep | awk '{sum+=$6} END {printf "%.1fMB\n", sum/1024}')"
echo "Port 8080: $(netstat -tuln | grep :8080 | wc -l) listeners"
echo "Last 3 log entries:"
tail -3 /opt/dipmaster-trading/logs/system.log
EOF

chmod +x /usr/local/bin/dipmaster_status.sh
```

### 使用监控脚本
```bash
# 检查系统状态
dipmaster_status.sh

# 定期监控 (每30秒)
watch -n 30 dipmaster_status.sh
```

## 🛠️ 常用管理命令

### 服务管理
```bash
# 启动
sudo systemctl start dipmaster

# 停止
sudo systemctl stop dipmaster

# 重启
sudo systemctl restart dipmaster

# 查看状态
sudo systemctl status dipmaster

# 查看日志
journalctl -u dipmaster -n 50
```

### 手动运行 (调试用)
```bash
cd /opt/dipmaster-trading
source venv/bin/activate

# 纸面交易模式
python main.py --config config/paper_trading_config.json --paper --log-level DEBUG

# 禁用仪表板
python main.py --config config/paper_trading_config.json --paper --no-dashboard

# 指定单个交易对测试
python main.py --config config/paper_trading_config.json --paper --symbols BTCUSDT
```

## 🔥 防火墙配置

```bash
# 开放必要端口
sudo firewall-cmd --permanent --add-port=8080/tcp  # Web仪表板
sudo firewall-cmd --permanent --add-port=22/tcp    # SSH
sudo firewall-cmd --reload

# 检查端口开放状态
sudo firewall-cmd --list-ports
```

## 📈 性能优化 (针对4GB RAM)

### 系统优化
```bash
# 调整swap使用
echo 'vm.swappiness=10' >> /etc/sysctl.conf

# 优化网络参数
echo 'net.core.rmem_max=134217728' >> /etc/sysctl.conf
echo 'net.core.wmem_max=134217728' >> /etc/sysctl.conf

# 应用配置
sysctl -p
```

### Python进程限制
在systemd服务文件中添加资源限制：
```ini
[Service]
MemoryMax=3G
CPUQuota=150%
LimitNOFILE=65536
```

## 🚨 故障排除

### 常见问题及解决方案

1. **服务启动失败**:
```bash
# 查看详细错误信息
journalctl -u dipmaster -n 20

# 检查配置文件语法
python -m json.tool config/paper_trading_config.json

# 手动启动测试
cd /opt/dipmaster-trading && source venv/bin/activate && python main.py --config config/paper_trading_config.json --paper
```

2. **API连接错误**:
```bash
# 测试网络连接
curl -I https://api.binance.com/api/v3/ping

# 验证API密钥权限
python -c "
from binance.client import Client
client = Client('YOUR_API_KEY', 'YOUR_SECRET')
print(client.get_account_status())
"
```

3. **内存不足**:
```bash
# 监控内存使用
free -h

# 清理日志
find /opt/dipmaster-trading/logs -name "*.log" -mtime +7 -delete

# 重启服务释放内存
sudo systemctl restart dipmaster
```

4. **端口被占用**:
```bash
# 检查端口使用
netstat -tuln | grep 8080

# 杀死占用进程
sudo kill -9 $(lsof -ti:8080)
```

## 🔄 更新和维护

### 代码更新
```bash
cd /opt/dipmaster-trading

# 备份当前版本
cp -r . ../dipmaster-trading-backup-$(date +%Y%m%d)

# 停止服务
sudo systemctl stop dipmaster

# 更新代码
git pull origin main
# 或重新上传文件

# 更新依赖
source venv/bin/activate
pip install -r requirements.txt --upgrade

# 重启服务
sudo systemctl start dipmaster
```

### 定期维护脚本
```bash
# 创建维护脚本
cat > /usr/local/bin/dipmaster_maintenance.sh << 'EOF'
#!/bin/bash
echo "开始DipMaster系统维护..."

# 备份日志
tar -czf /backup/logs_$(date +%Y%m%d).tar.gz /opt/dipmaster-trading/logs/

# 清理旧日志
find /opt/dipmaster-trading/logs -name "*.log" -mtime +30 -delete

# 检查磁盘空间
df -h /opt/dipmaster-trading

echo "维护完成"
EOF

chmod +x /usr/local/bin/dipmaster_maintenance.sh

# 添加到crontab (每周执行)
echo "0 2 * * 0 /usr/local/bin/dipmaster_maintenance.sh" | crontab -
```

## ✅ 部署完成检查清单

- [ ] 服务器环境准备完成 (Python 3.11, 依赖包)
- [ ] 项目代码上传到 `/opt/dipmaster-trading/`
- [ ] 虚拟环境创建并安装依赖
- [ ] 配置文件编辑完成 (API密钥)
- [ ] 防火墙配置 (8080端口开放)
- [ ] systemd服务配置并启用
- [ ] 手动测试运行正常
- [ ] 服务自动启动测试
- [ ] 监控脚本部署
- [ ] 日志轮转配置
- [ ] 备份策略实施

## 📞 紧急操作

### 立即停止交易
```bash
# 方法1: 停止系统服务
sudo systemctl stop dipmaster

# 方法2: 强制结束进程
pkill -f "python.*main.py"

# 方法3: 紧急停止 (如果有的话)
curl -X POST http://localhost:8080/emergency-stop
```

### 数据备份与恢复
```bash
# 完整备份
tar -czf dipmaster_backup_$(date +%Y%m%d_%H%M%S).tar.gz /opt/dipmaster-trading/

# 恢复备份
tar -xzf dipmaster_backup_YYYYMMDD_HHMMSS.tar.gz -C /
```

---

**🎉 部署完成后，你的DipMaster交易系统将在香港服务器上24/7运行纸面交易，通过 http://47.239.1.232:8080 访问监控面板！**

**⚠️ 重要提醒:**
1. 首次运行建议使用纸面交易模式充分测试
2. 定期检查日志确保系统正常运行
3. 监控服务器资源使用情况
4. 备份重要配置和交易数据
5. 遵循风险管理原则，控制仓位大小