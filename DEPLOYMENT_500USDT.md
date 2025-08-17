# DipMaster Trading System v1.0.0 - 500 USDT部署指南

## 🎯 部署概览

本指南将帮助您在本地环境部署DipMaster交易系统，使用500 USDT进行实盘交易。

### 安全配置
- **初始资金**: 500 USDT
- **最大单仓**: 100 USDT (20%)
- **最大同时持仓**: 3个
- **日损失限制**: 25 USDT (5%)
- **紧急止损**: 15% (75 USDT)

## 🛠️ 系统要求

### 硬件要求
- **CPU**: 4核心+
- **内存**: 8GB RAM+
- **存储**: 10GB 可用空间
- **网络**: 稳定宽带连接

### 软件要求
- **操作系统**: Windows 10+, macOS 10.15+, Linux Ubuntu 20+
- **Python**: 3.11+
- **Node.js**: 18+ (仪表板)
- **Git**: 最新版本

## 📥 第一步：下载和安装

### 1. 克隆项目
```bash
git clone https://github.com/your-username/DipMaster-Trading-System.git
cd DipMaster-Trading-System
```

### 2. 安装Python依赖
```bash
# 创建虚拟环境
python -m venv dipmaster_env

# 激活虚拟环境
# Windows:
dipmaster_env\Scripts\activate
# macOS/Linux:
source dipmaster_env/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 安装前端依赖
```bash
cd frontend
npm install
cd ..
```

## 🔑 第二步：API配置

### 1. 获取Binance API密钥
1. 登录 [Binance](https://www.binance.com)
2. 前往 **API管理** → **创建API**
3. 设置权限：
   - ✅ **现货交易**
   - ✅ **合约交易** (如需要)
   - ❌ **提现** (安全起见)
   - ❌ **子账户** (不需要)

### 2. 配置环境变量
```bash
# Windows (创建 .env 文件)
echo BINANCE_API_KEY=your_api_key_here > .env
echo BINANCE_API_SECRET=your_api_secret_here >> .env

# macOS/Linux
export BINANCE_API_KEY="your_api_key_here"
export BINANCE_API_SECRET="your_api_secret_here"
```

### 3. 配置交易参数
编辑 `config/production_500usdt.json`:
```json
{
  "trading": {
    "initial_capital": 500,
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "max_position_size_usd": 100
  }
}
```

## 🧪 第三步：纸面交易测试

### 1. 启动纸面交易模式
```bash
python main.py --paper --config config/production_500usdt.json
```

### 2. 启动仪表板
```bash
# 新终端窗口
cd frontend
npm run dev
```

### 3. 访问仪表板
打开浏览器访问: http://localhost:3000

### 4. 测试时间建议
- **最少测试**: 24小时
- **建议测试**: 7天
- **充分测试**: 30天

## 🚀 第四步：实盘部署

### 1. 验证配置
```bash
python -c "
import json
with open('config/production_500usdt.json') as f:
    config = json.load(f)
print('Configuration loaded successfully')
print(f'Initial Capital: {config[\"trading\"][\"initial_capital\"]} USDT')
print(f'Max Position: {config[\"trading\"][\"max_position_size_usd\"]} USDT')
"
```

### 2. 启动实盘交易
```bash
python main.py --config config/production_500usdt.json
```

### 3. 启动监控仪表板
```bash
cd frontend
npm run dev
```

## 📊 第五步：监控和管理

### 1. 关键监控指标
- **胜率**: 目标 65%+
- **日盈亏**: 限制 ±25 USDT
- **持仓数量**: 最多3个
- **资金使用率**: 最多60% (300 USDT)

### 2. 每日检查清单
- [ ] 检查昨日交易汇总
- [ ] 确认API连接正常
- [ ] 验证余额和持仓
- [ ] 查看风险指标
- [ ] 检查系统日志

### 3. 紧急停止
如需紧急停止交易：
```bash
# 方法1: Ctrl+C 停止程序
# 方法2: 紧急停止脚本
python scripts/emergency_stop.py
```

## ⚠️ 安全检查清单

### API安全
- [ ] API密钥已设置正确权限
- [ ] 提现权限已禁用
- [ ] IP白名单已设置 (推荐)
- [ ] API密钥定期更换

### 资金安全
- [ ] 初始资金不超过可承受损失
- [ ] 单笔交易不超过100 USDT
- [ ] 日损失限制已设置 (25 USDT)
- [ ] 紧急止损已配置 (15%)

### 系统安全
- [ ] 交易系统运行在安全环境
- [ ] 防火墙已配置
- [ ] 系统保持最新更新
- [ ] 定期备份数据

## 🔧 故障排除

### 常见问题

**1. API连接失败**
```bash
# 检查API密钥
python -c "
import os
print('API Key:', os.getenv('BINANCE_API_KEY', 'NOT SET'))
print('API Secret:', 'SET' if os.getenv('BINANCE_API_SECRET') else 'NOT SET')
"
```

**2. 余额不足**
- 确保账户有足够USDT
- 检查最小交易金额设置
- 验证交易对可用性

**3. 交易失败**
- 检查网络连接
- 验证API权限
- 查看错误日志

**4. 仪表板无法访问**
```bash
# 检查端口占用
netstat -an | grep 3000
# 重启前端服务
cd frontend && npm run dev
```

### 日志位置
- **交易日志**: `logs/dipmaster_YYYYMMDD.log`
- **错误日志**: `logs/errors.log`
- **API日志**: `logs/api.log`

## 📞 支持和维护

### 性能监控
- 每日查看交易汇总
- 每周评估策略表现
- 每月调整参数优化

### 版本更新
```bash
# 检查更新
git pull origin main

# 重新安装依赖
pip install -r requirements.txt
cd frontend && npm install
```

### 联系支持
- **文档**: 查看 `docs/` 目录
- **问题报告**: 创建GitHub Issue
- **紧急情况**: 立即停止交易

## 📈 预期表现

### 保守预期 (前3个月)
- **胜率**: 55-65%
- **月收益**: 3-8%
- **最大回撤**: <5%
- **夏普比率**: 1.5-2.0

### 稳定期表现 (3-12个月)
- **胜率**: 65-75%
- **年化收益**: 25-40%
- **最大回撤**: <3%
- **夏普比率**: 2.0-3.0

## 📝 免责声明

**⚠️ 重要提醒**:
- 加密货币交易存在高风险
- 历史表现不保证未来收益
- 仅投资可承受损失的资金
- 建议从小额资金开始测试
- 持续监控和风险管理至关重要

**🚨 风险警告**:
- 市场波动可能导致快速损失
- 技术故障可能影响交易
- 监管变化可能影响可用性
- 始终保持理性和谨慎

---

**版本**: v1.0.0  
**更新日期**: 2025-08-17  
**支持**: DipMaster开发团队