# 🚀 DipMaster Trading System - 阿里云服务器一键部署

## 服务器信息
- **服务器ID**: iZj6c0m028i65r1jyf6xwjZ  
- **配置**: 2核(vCPU) 4GB RAM
- **系统**: Alibaba Cloud Linux 3.2104 LTS 64位
- **公网IP**: 47.239.1.232
- **地域**: 中国香港

---

## 🎯 三种部署方式选择

### 方式一：Docker部署 (推荐⭐)
**优势**: 环境隔离、易于管理、自动监控
```bash
# SSH登录服务器
ssh root@47.239.1.232

# 一键Docker部署
wget -O deploy_docker.sh https://raw.githubusercontent.com/your-repo/DipMaster-Trading-System/main/deploy_docker.sh
chmod +x deploy_docker.sh
./deploy_docker.sh
```

### 方式二：直接系统部署
**优势**: 资源消耗小、性能好
```bash
# SSH登录服务器  
ssh root@47.239.1.232

# 一键系统部署
wget -O deploy_server.sh https://raw.githubusercontent.com/your-repo/DipMaster-Trading-System/main/deploy_server.sh
chmod +x deploy_server.sh
./deploy_server.sh
```

### 方式三：手动部署 (高级用户)
参考详细部署指南: `deployment_guide.md`

---

## 📋 快速部署检查清单

### 部署前准备
- [ ] 确保有Binance API密钥 (仅交易权限)
- [ ] 服务器SSH访问正常
- [ ] 网络连接稳定
- [ ] 至少2GB可用磁盘空间

### 部署步骤
1. **SSH连接服务器**:
   ```bash
   ssh root@47.239.1.232
   ```

2. **选择部署方式并执行**:
   ```bash
   # Docker部署 (推荐)
   curl -fsSL https://raw.githubusercontent.com/your-repo/DipMaster-Trading-System/main/deploy_docker.sh | bash
   
   # 或直接系统部署
   curl -fsSL https://raw.githubusercontent.com/your-repo/DipMaster-Trading-System/main/deploy_server.sh | bash
   ```

3. **上传项目代码** (如果脚本没有自动克隆):
   ```bash
   # 从本地上传
   scp -r /Users/zhangxuanyang/Desktop/Quant/DipMaster-Trading-System/* root@47.239.1.232:/opt/dipmaster-trading/
   
   # 或使用rsync (推荐)
   rsync -avz --progress /Users/zhangxuanyang/Desktop/Quant/DipMaster-Trading-System/ root@47.239.1.232:/opt/dipmaster-trading/
   ```

4. **配置API密钥**:
   ```bash
   # Docker部署
   vim /opt/dipmaster-trading/config/paper_trading_config.json
   
   # 直接部署
   vim /opt/dipmaster-trading/config/paper_trading_config.json
   ```
   
   替换以下内容:
   ```json
   {
     "api": {
       "api_key": "YOUR_BINANCE_API_KEY",
       "api_secret": "YOUR_BINANCE_API_SECRET"
     }
   }
   ```

5. **启动服务**:
   ```bash
   # Docker方式
   dipmaster-docker.sh start
   
   # 直接部署方式  
   sudo systemctl start dipmaster
   ```

---

## 🔍 部署后验证

### 访问监控面板
- **DipMaster交易系统**: http://47.239.1.232:8080
- **Prometheus监控**: http://47.239.1.232:9090 (仅Docker部署)
- **cAdvisor监控**: http://47.239.1.232:8081 (仅Docker部署)

### 检查服务状态
```bash
# Docker部署
dipmaster-monitor.sh

# 直接部署
monitor_dipmaster.sh
```

### 查看实时日志
```bash
# Docker部署
dipmaster-docker.sh logs

# 直接部署  
tail -f /opt/dipmaster-trading/logs/dipmaster_$(date +%Y%m%d).log
```

---

## 🛠️ 常用管理命令

### Docker部署管理
```bash
dipmaster-docker.sh start      # 启动服务
dipmaster-docker.sh stop       # 停止服务
dipmaster-docker.sh restart    # 重启服务
dipmaster-docker.sh status     # 查看状态
dipmaster-docker.sh logs       # 查看日志
dipmaster-docker.sh backup     # 备份数据
```

### 直接部署管理
```bash
sudo systemctl start dipmaster     # 启动服务
sudo systemctl stop dipmaster      # 停止服务
sudo systemctl restart dipmaster   # 重启服务  
sudo systemctl status dipmaster    # 查看状态
monitor_dipmaster.sh               # 系统监控
```

---

## 🚨 故障排除

### 常见问题

**1. 服务启动失败**
```bash
# 检查日志
docker-compose logs dipmaster-trading  # Docker方式
sudo journalctl -u dipmaster -n 20     # 直接部署

# 检查配置文件
python -m json.tool /opt/dipmaster-trading/config/paper_trading_config.json
```

**2. 无法访问Web界面**
```bash
# 检查端口占用
netstat -tuln | grep 8080

# 检查防火墙
sudo firewall-cmd --list-ports
sudo firewall-cmd --permanent --add-port=8080/tcp
sudo firewall-cmd --reload
```

**3. API连接错误** 
```bash
# 测试网络连接
curl -I https://api.binance.com/api/v3/ping

# 验证API密钥格式
grep -E "api_key|api_secret" /opt/dipmaster-trading/config/paper_trading_config.json
```

**4. 内存不足**
```bash
# 检查内存使用
free -h
docker stats --no-stream  # Docker方式

# 重启释放内存
dipmaster-docker.sh restart        # Docker方式
sudo systemctl restart dipmaster   # 直接部署
```

### 紧急停止交易
```bash
# Docker方式
dipmaster-docker.sh stop

# 直接部署方式
sudo systemctl stop dipmaster

# 强制停止
pkill -f "python.*main.py"
```

---

## 📊 性能监控

### 系统资源监控
```bash
# CPU和内存使用
top -p $(pgrep -f "python.*main.py")

# 磁盘使用  
df -h /opt/dipmaster-trading

# 网络连接
ss -tuln | grep -E ":(8080|9090|8081)"
```

### 交易性能指标
```bash
# 查看交易日志
tail -100 /opt/dipmaster-trading/logs/dipmaster_$(date +%Y%m%d).log | grep -E "(买入|卖出|盈利)"

# 监控API调用频率
grep "binance" /opt/dipmaster-trading/logs/*.log | wc -l
```

---

## 🔄 系统维护

### 定期维护任务
```bash
# 每日备份 (已自动配置)
/usr/local/bin/dipmaster-backup.sh  # 直接部署
/usr/local/bin/dipmaster-backup.sh  # Docker部署

# 清理日志 (每周执行)
find /opt/dipmaster-trading/logs -name "*.log" -mtime +7 -delete

# 更新系统 (每月执行)
yum update -y
```

### 代码更新
```bash
# 停止服务
dipmaster-docker.sh stop           # Docker方式
sudo systemctl stop dipmaster      # 直接部署

# 备份当前版本
cp -r /opt/dipmaster-trading /opt/dipmaster-trading-backup-$(date +%Y%m%d)

# 上传新代码
rsync -avz --progress /path/to/new/code/ root@47.239.1.232:/opt/dipmaster-trading/

# 重启服务
dipmaster-docker.sh start          # Docker方式  
sudo systemctl start dipmaster     # 直接部署
```

---

## 📈 性能优化建议

### 针对4GB RAM服务器优化
```bash
# 调整swap使用
echo 'vm.swappiness=10' >> /etc/sysctl.conf

# 优化TCP参数
echo 'net.core.rmem_max=134217728' >> /etc/sysctl.conf  
echo 'net.core.wmem_max=134217728' >> /etc/sysctl.conf

# 应用配置
sysctl -p
```

### Python进程优化
```python
# 在配置文件中设置
{
  "deployment_settings": {
    "max_memory_usage_mb": 2048,
    "gc_threshold": [700, 10, 10],
    "max_workers": 2
  }
}
```

---

## 🔐 安全建议

### API安全
- 使用测试网络进行纸面交易
- API权限仅开启交易，禁用提现
- 定期更换API密钥
- 监控异常交易行为

### 服务器安全  
```bash
# 修改SSH端口
sed -i 's/#Port 22/Port 2022/' /etc/ssh/sshd_config
systemctl restart sshd

# 配置fail2ban
yum install -y fail2ban
systemctl enable fail2ban
systemctl start fail2ban

# 定期安全更新
yum update -y --security
```

---

## 📞 技术支持

### 获取帮助
- **查看日志**: 首要故障排除方法
- **检查配置**: 验证JSON格式和API密钥
- **网络测试**: 确保与Binance连接正常
- **资源监控**: 确保服务器资源充足

### 联系信息
- **项目文档**: `deployment_guide.md`
- **配置说明**: `config/paper_trading_config.json`
- **日志位置**: `/opt/dipmaster-trading/logs/`

---

**🎉 部署完成后，你的DipMaster交易系统将在阿里云香港服务器上24/7运行纸面交易！**

**访问地址**: http://47.239.1.232:8080

**⚠️ 重要提醒**: 
1. 首次运行建议使用纸面交易模式充分测试
2. 定期检查系统运行状态和日志
3. 监控服务器资源使用情况  
4. 遵循风险管理原则，控制仓位大小
5. 备份重要配置和交易数据

**祝您交易顺利！** 🚀📈