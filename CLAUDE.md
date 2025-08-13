# CLAUDE.md - DipMaster Trading System 维护文档

## 📋 项目概览

**DipMaster Trading System** 是一个基于逆向工程的加密货币自动交易系统，专门实现DipMaster AI策略，具备82.1%胜率和完整的实时交易能力。

### 🎯 核心特性
- **DipMaster AI策略**: 完整逆向工程，87.9%逢跌买入率
- **实时交易引擎**: WebSocket数据流，毫秒级响应
- **15分钟边界管理**: 100%严格的时间纪律
- **风险管理系统**: 多重安全控制和监控
- **高胜率交易**: 历史验证82.1%胜率

## 🏗️ 系统架构

### 目录结构
```
DipMaster-Trading-System/
├── src/                          # 核心源代码
│   ├── core/                     # 核心交易模块
│   │   ├── trading_engine.py     # 主交易引擎
│   │   ├── signal_detector.py    # 信号检测器
│   │   ├── timing_manager.py     # 15分钟边界管理
│   │   ├── position_manager.py   # 仓位管理
│   │   ├── order_executor.py     # 订单执行
│   │   └── websocket_client.py   # WebSocket客户端
│   ├── dashboard/                # 监控仪表板
│   ├── scripts/                  # 分析和工具脚本
│   └── tools/                    # 策略工具
├── config/                       # 配置文件
├── data/                         # 市场数据
├── results/                      # 分析结果
├── docs/                         # 文档
├── tests/                        # 测试文件
└── logs/                         # 日志文件
```

### 核心组件架构

#### 1. 交易引擎 (trading_engine.py)
- **主要职责**: 协调所有组件，管理交易生命周期
- **关键功能**: 
  - 异步事件处理
  - 风险管理控制
  - 多仓位并发管理
  - 实时状态监控

#### 2. 信号检测器 (signal_detector.py)
- **入场信号**: RSI(30-50) + 价格下跌 + 成交量确认
- **出场信号**: 15分钟边界 + 盈利目标 + 时间止损
- **技术指标**: RSI, MA20, 布林带, 成交量

#### 3. 时间管理器 (timing_manager.py)
- **15分钟边界**: 100%严格执行时间纪律
- **出场时机**: 15-29分钟, 45-59分钟优选
- **最大持仓**: 180分钟强制平仓

#### 4. WebSocket客户端 (websocket_client.py)
- **实时数据**: Binance WebSocket多币种流
- **延迟优化**: 毫秒级价格更新
- **断线重连**: 自动故障恢复

## 🔧 DipMaster策略详解

### 策略原理
DipMaster AI是一个逢跌买入的短线策略，专门捕捉市场短期回调机会：

#### 入场条件 (5分钟图)
1. **RSI区间**: 30-50 (不等极端超卖)
2. **价格位置**: 低于MA20 (87%概率)
3. **逢跌确认**: 买入价低于开盘价
4. **成交量**: 放量确认信号有效性

#### 出场条件 (15分钟图)
1. **时间纪律**: 严格15分钟边界出场 (100%)
2. **盈利目标**: 快速获利0.8%
3. **时间止损**: 最大持仓180分钟
4. **优选时段**: 15-29分钟(33.5%), 45-59分钟(28.6%)

### 关键参数
```python
# 入场参数
RSI_ENTRY_RANGE = [30, 50]
DIP_THRESHOLD = 0.002  # 0.2%下跌确认
VOLUME_MULTIPLIER = 1.5  # 成交量放大倍数

# 出场参数
MAX_HOLDING_MINUTES = 180
TARGET_PROFIT = 0.008  # 0.8%目标利润
BOUNDARY_SLOTS = [15, 30, 45, 60]  # 15分钟边界

# 风险管理
MAX_POSITIONS = 3
MAX_POSITION_SIZE = 1000  # USD
DAILY_LOSS_LIMIT = -500  # USD
```

## 🚀 部署指南

### 环境要求
- Python 3.11+
- 8GB+ RAM
- 稳定网络连接
- Binance API权限

### 快速部署

#### 1. 安装依赖
```bash
pip install -r requirements.txt
```

#### 2. 配置API
```bash
cp config/config.json.example config/config.json
# 编辑config.json，填入API密钥
```

#### 3. 运行模式

**纸面交易** (推荐开始):
```bash
python main.py --paper --config config/config.json
```

**实盘交易** (充分测试后):
```bash
python main.py --config config/config.json
```

### Docker部署
```bash
# 构建镜像
docker build -t dipmaster-trading .

# 运行容器
docker run -d \
  --name dipmaster \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/logs:/app/logs \
  dipmaster-trading
```

## 🔍 监控和维护

### 日志监控
- **位置**: `logs/dipmaster_YYYYMMDD.log`
- **级别**: INFO, WARNING, ERROR
- **关键指标**: 交易信号, 执行状态, 风险警告

### 性能指标
```python
# 核心KPI
win_rate = 82.1%  # 目标胜率
avg_holding_time = 96  # 分钟
dip_buying_rate = 87.9%  # 逢跌买入率
boundary_compliance = 100%  # 边界执行率
```

### 健康检查
```bash
# 检查WebSocket连接
curl -f http://localhost:8080/health

# 监控内存使用
ps aux | grep python

# 检查日志错误
tail -f logs/dipmaster_$(date +%Y%m%d).log | grep ERROR
```

## 🛠️ 开发和调试

### 开发环境设置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装开发依赖
pip install -r requirements.txt
pip install pytest black isort

# 运行测试
pytest tests/
```

### 调试模式
```bash
# 详细日志输出
python main.py --log-level DEBUG --no-dashboard

# 单币种测试
python main.py --symbols BTCUSDT --paper

# 信号测试
python src/scripts/core/strategy_validation.py
```

### 代码结构
- **异步编程**: 使用asyncio处理并发
- **错误处理**: 全面的异常捕获和恢复
- **配置驱动**: 所有参数可配置
- **模块化设计**: 组件独立可测试

## ⚠️ 风险管理

### 资金安全
- **API权限**: 仅交易权限，禁用提现
- **仓位限制**: 单仓位最大1000 USD
- **日损限制**: 单日最大亏损500 USD
- **币种分散**: 最多3个并发仓位

### 技术风险
- **网络断线**: 自动重连机制
- **数据延迟**: 毫秒级WebSocket
- **系统故障**: 优雅停机和恢复
- **异常处理**: 全面错误捕获

### 策略风险
- **趋势风险**: 不适用强趋势市场
- **时间风险**: 严格时间纪律控制
- **滑点风险**: 市价单快速执行
- **相关性风险**: 避免高相关币种

## 🔧 故障排除

### 常见问题

**1. WebSocket连接失败**
```bash
# 检查网络连接
ping -c 4 stream.binance.com

# 检查API状态
curl -s https://api.binance.com/api/v3/ping
```

**2. 交易信号缺失**
```python
# 检查市场数据
python src/scripts/core/test_market_data.py

# 验证技术指标
python src/scripts/core/technical_analysis.py --symbol BTCUSDT
```

**3. 订单执行失败**
- 检查API权限和余额
- 验证交易对规则
- 确认网络稳定性

**4. 内存使用过高**
```bash
# 监控内存
watch -n 1 'ps aux | grep python | head -5'

# 清理日志
find logs/ -name "*.log" -mtime +7 -delete
```

### 紧急处理

**立即停止交易**:
```bash
# 发送停止信号
pkill -TERM -f "python main.py"

# 或使用管理接口
curl -X POST http://localhost:8080/emergency-stop
```

**强制平仓**:
```bash
# 执行紧急平仓脚本
python src/tools/emergency_close_positions.py
```

## 📈 性能优化

### 系统优化
- **多进程**: CPU密集型任务
- **连接池**: 重用HTTP连接
- **内存缓存**: 热点数据缓存
- **批量处理**: 减少API调用

### 策略优化
- **参数调优**: 基于历史数据
- **时段优化**: 识别最佳交易时间
- **币种筛选**: 选择高流动性币种
- **信号强度**: 增加置信度过滤

## 📚 相关资源

### 文档链接
- [策略详解](docs/strategy_guides/)
- [API文档](docs/api_reference.md)
- [部署指南](docs/deployment_guide.md)
- [性能报告](results/reports/)

### 外部资源
- [Binance API文档](https://binance-docs.github.io/apidocs/)
- [技术分析库](https://github.com/bukosabino/ta)
- [WebSocket协议](https://github.com/binance/binance-spot-api-docs)

## 🔄 更新和维护

### 版本控制
- 遵循语义化版本控制
- 主要版本更新需充分测试
- 保持向后兼容性

### 定期维护
- **每日**: 检查日志和性能
- **每周**: 更新市场数据
- **每月**: 策略参数优化
- **每季**: 全面系统审计

### 备份策略
- **配置备份**: 每次修改后
- **数据备份**: 每日增量
- **代码备份**: Git版本控制
- **日志归档**: 每月压缩

---

**🚨 重要提醒**: 
- 始终在纸面交易模式下充分测试
- 定期监控系统性能和交易结果
- 遵循风险管理原则，控制仓位大小
- 保持API密钥安全，定期更换

**📞 技术支持**: 遇到问题请查看日志文件，或联系系统管理员

---

**最后更新**: 2025-08-12  
**文档版本**: 1.0.0  
**系统版本**: DipMaster Trading System v1.0.0