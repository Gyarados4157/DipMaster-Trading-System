# Agent工作流快速参考指南

## 🚀 快速启动命令

### 🎯 策略开发完整流程

```bash
# Step 0: 策略目标设定
Task: strategy-orchestrator
Prompt: "开发DipMaster V4增强策略，目标胜率85%+，最大回撤<3%，适用于BTCUSDT/ETHUSDT等主流币种"

# Step 1: 数据基础建设  
Task: data-infrastructure-builder
Prompt: "基于StrategySpec收集11个主流币种2年历史数据，包含5分钟K线、成交量、资金费率数据"

# Step 2: 特征工程
Task: feature-engineering-labeler
Prompt: "生成DipMaster策略所需技术指标：RSI、布林带、成交量指标、价格动量特征，并创建15分钟未来收益标签"

# Step 3: 模型训练回测
Task: model-backtest-validator
Prompt: "训练LightGBM模型预测逢跌买入信号，使用purged时序交叉验证，进行完整回测含交易成本"

# Step 4: 组合优化
Task: portfolio-risk-optimizer
Prompt: "基于信号构建多币种组合，控制相关性<0.7，目标夏普比>1.5，最大单仓位30%"

# Step 5: 执行系统
Task: execution-microstructure-oms
Prompt: "实现智能订单执行，支持TWAP分割，最小化滑点，集成Binance API"

# Step 6: 监控告警
Task: monitoring-log-collector
Prompt: "建立实时监控系统，跟踪信号-持仓一致性，VaR违规告警，发布到Kafka"

# Step 7: 数据API
Task: dashboard-api-kafka-consumer
Prompt: "构建REST API服务，提供PnL查询、持仓状态、实时告警WebSocket"

# Step 8: 可视化面板
Task: frontend-dashboard-nextjs
Prompt: "开发实时交易监控面板，显示PnL曲线、风险指标、持仓分布、告警通知"
```

## 🔄 分阶段执行

### 阶段1：策略研发 (离线)
```bash
strategy-orchestrator → data-infrastructure-builder → feature-engineering-labeler → model-backtest-validator
```

### 阶段2：系统集成 (半实盘)
```bash
portfolio-risk-optimizer → execution-microstructure-oms → monitoring-log-collector
```

### 阶段3：生产部署 (全实盘)
```bash
dashboard-api-kafka-consumer → frontend-dashboard-nextjs
```

## 📋 常用Agent调用模板

### 🧪 策略研究
```bash
# 快速回测现有策略
Task: model-backtest-validator
Prompt: "对DipMaster V3策略进行2年历史回测，评估在当前市场环境下的表现"

# 特征重要性分析
Task: feature-engineering-labeler  
Prompt: "分析RSI、布林带、成交量等特征的预测能力，识别最重要的信号"
```

### 📊 数据更新
```bash
# 增量数据更新
Task: data-infrastructure-builder
Prompt: "更新最近30天的市场数据，补充新增交易对数据"

# 数据质量检查
Task: data-infrastructure-builder
Prompt: "检查历史数据完整性，修复缺失数据，验证数据一致性"
```

### 🎛️ 风险管理
```bash
# 风险敞口分析
Task: portfolio-risk-optimizer
Prompt: "分析当前组合风险敞口，计算VaR/ES，检查相关性集中度"

# 动态止损优化
Task: portfolio-risk-optimizer  
Prompt: "优化止损策略，平衡风险控制与收益最大化"
```

### 🔧 系统维护
```bash
# 性能监控
Task: monitoring-log-collector
Prompt: "生成系统性能报告，监控订单执行质量，识别潜在问题"

# API服务优化
Task: dashboard-api-kafka-consumer
Prompt: "优化数据查询性能，增加新的API端点，改进WebSocket稳定性"
```

## 🚨 应急处理

### 💥 策略异常
```bash
# 紧急停止
Task: execution-microstructure-oms
Prompt: "执行紧急停止程序，平仓所有持仓，停止新开仓"

# 快速诊断
Task: monitoring-log-collector
Prompt: "分析最近24小时交易日志，识别异常信号或执行问题"
```

### 📉 风险警报
```bash
# 风险评估
Task: portfolio-risk-optimizer
Prompt: "紧急风险评估，计算当前敞口，建议减仓方案"

# 实时监控
Task: monitoring-log-collector
Prompt: "启动实时风险监控，设置VaR告警阈值，监控市场异常"
```

## 💡 最佳实践

1. **顺序执行**: 按步骤顺序调用Agent，确保输入输出匹配
2. **检查点验证**: 每个阶段完成后验证输出质量
3. **版本管理**: 保存每次Agent输出的版本，便于回滚
4. **监控优先**: 优先建立监控体系，确保系统安全
5. **逐步上线**: 从纸面交易开始，逐步过渡到实盘

## 🔗 相关文件
- `CLAUDE.md` - 完整文档
- `mcp-config.json` - MCP服务配置  
- `config/dipmaster_v3_optimized.json` - 策略配置
- `main.py` - 系统入口