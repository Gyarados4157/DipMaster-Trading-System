# DipMaster Trading System - Linux服务器部署指南

## 🎯 快速部署（推荐）

### 一键部署脚本
```bash
# 方法1：直接下载并运行（推荐）
curl -o linux_deployment.sh https://raw.githubusercontent.com/Gyarados4157/DipMaster-Trading-System/main/linux_deployment.sh && chmod +x linux_deployment.sh && sudo ./linux_deployment.sh

# 方法2：克隆仓库后部署
mkdir dipmaster && cd dipmaster
git clone https://github.com/Gyarados4157/DipMaster-Trading-System.git .
sudo ./linux_deployment.sh
```

## 📋 系统要求

- **操作系统**: CentOS 7+, RHEL 7+, Amazon Linux 2, Ubuntu 18.04+
- **Python版本**: 3.11+ (脚本会自动安装)
- **权限**: root权限（用于安装系统依赖）
- **内存**: 最低1GB，推荐2GB+
- **存储**: 最低2GB可用空间

## 🔧 部署脚本功能

`linux_deployment.sh` 会自动完成以下操作：

1. **系统检测**: 自动识别CentOS/Ubuntu系统
2. **Python 3.11安装**: 从源码编译安装最新Python 3.11
3. **依赖安装**: 安装所有必需的系统和Python包
4. **项目配置**: 克隆代码库并设置配置文件
5. **脚本创建**: 生成启动、监控、停止脚本
6. **服务配置**: 创建systemd服务（可选）
7. **权限设置**: 设置正确的文件权限
8. **测试验证**: 运行基础功能测试

## 📂 安装后的目录结构

```
/opt/DipMaster-Trading-System/
├── config/
│   └── paper_trading_config.json  # 主配置文件
├── logs/                           # 日志目录
├── data/                          # 数据目录
├── results/                       # 结果目录
├── start_dipmaster.sh             # 启动脚本
├── monitor_dipmaster.sh           # 监控脚本
├── stop_dipmaster.sh              # 停止脚本
└── [源代码文件...]
```

## ⚙️ 配置设置

### 1. 编辑配置文件
```bash
cd /opt/DipMaster-Trading-System
vi config/paper_trading_config.json
```

### 2. 主要配置项
```json
{
    "trading": {
        "paper_trading": true,        # 纸面交易模式
        "symbols": ["BTCUSDT", "ETHUSDT"],
        "max_positions": 3,
        "position_size_usd": 500
    },
    "exchange": {
        "name": "binance",
        "api_key": "your_api_key_here",      # 填入你的API密钥
        "api_secret": "your_api_secret_here", # 填入你的API密码
        "testnet": true                       # 使用测试网
    }
}
```

### 3. 获取Binance API密钥
1. 访问 [Binance API管理](https://testnet.binance.vision/) (测试网)
2. 创建新的API密钥
3. 设置权限：仅需要"现货和杠杆交易"权限
4. 将密钥填入配置文件

## 🚀 启动和管理

### 直接启动
```bash
cd /opt/DipMaster-Trading-System
./start_dipmaster.sh
```

### 使用systemd服务
```bash
# 启动服务
systemctl start dipmaster

# 停止服务
systemctl stop dipmaster

# 查看状态
systemctl status dipmaster

# 开机自启
systemctl enable dipmaster

# 查看日志
journalctl -u dipmaster -f
```

### 监控系统
```bash
# 查看系统状态
./monitor_dipmaster.sh

# 实时查看日志
tail -f logs/dipmaster_*.log

# 停止系统
./stop_dipmaster.sh
```

## 📊 验证部署

### 1. 检查Python版本
```bash
python3 --version  # 应该显示 Python 3.11.x
```

### 2. 检查依赖安装
```bash
python3 -c "import ccxt, pandas, numpy; print('Dependencies OK')"
```

### 3. 检查配置文件
```bash
python3 -c "
import json
with open('config/paper_trading_config.json') as f:
    config = json.load(f)
print('Config loaded successfully')
"
```

### 4. 运行快速测试
```bash
python3 quick_paper_test.py
```

## 🔍 故障排除

### 常见问题

**1. Python版本错误**
```bash
# 检查Python版本
python3 --version
which python3

# 如果版本不对，重新运行部署脚本
sudo ./linux_deployment.sh
```

**2. 依赖安装失败**
```bash
# 手动安装核心依赖
pip3 install numpy pandas ccxt python-binance websockets aiohttp fastapi

# 检查pip版本
pip3 --version
```

**3. 权限问题**
```bash
# 确保脚本有执行权限
chmod +x *.sh

# 确保目录权限正确
sudo chown -R $(whoami):$(whoami) /opt/DipMaster-Trading-System
```

**4. API连接失败**
- 检查API密钥是否正确
- 确认网络连接正常
- 验证Binance API权限设置

### 日志查看
```bash
# 查看系统日志
tail -f logs/dipmaster_*.log

# 查看服务日志
journalctl -u dipmaster -f

# 查看错误日志
grep ERROR logs/dipmaster_*.log
```

## 🔒 安全建议

1. **API密钥安全**
   - 使用测试网API密钥进行初始测试
   - 限制API权限为仅交易，禁用提现
   - 定期轮换API密钥

2. **系统安全**
   - 使用防火墙限制访问端口
   - 定期更新系统包
   - 监控系统资源使用

3. **交易安全**
   - 从纸面交易开始测试
   - 设置合理的仓位大小
   - 设置日损失限制

## 🔄 更新和维护

### 更新代码
```bash
cd /opt/DipMaster-Trading-System
git pull origin main
pip3 install -r requirements_linux.txt
```

### 定期维护
- 每周检查日志文件大小
- 每月更新系统包
- 定期备份配置文件

## 📞 支持

如遇问题，请提供以下信息：
1. 操作系统版本：`cat /etc/os-release`
2. Python版本：`python3 --version`
3. 错误日志：`tail -50 logs/dipmaster_*.log`
4. 系统状态：`./monitor_dipmaster.sh`

---

**重要提醒**: 始终在纸面交易模式下充分测试系统，确认策略表现后再考虑实盘交易。