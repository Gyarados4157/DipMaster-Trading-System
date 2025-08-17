# DipMaster数据API服务

## 🚀 项目概述

DipMaster数据API服务是一个高性能的实时交易数据服务，专为DipMaster交易系统设计。该服务提供完整的数据管道，包括Kafka事件消费、ClickHouse时序数据存储、REST API查询和WebSocket实时推送。

### 核心特性

- **🔄 实时数据流处理**: 异步消费Kafka事件流(exec.reports.v1, risk.metrics.v1, alerts.v1, system.health.v1)
- **🗄️ 高性能时序存储**: ClickHouse优化的交易数据存储和查询
- **🌐 REST API服务**: 完整的RESTful API，支持PnL、持仓、成交、风险和性能查询
- **⚡ WebSocket实时推送**: 毫秒级实时数据推送，支持告警、PnL、持仓和健康状态
- **💾 智能缓存系统**: 多层缓存架构，<100ms响应时间
- **🛡️ 企业级安全**: API认证、限流保护和访问控制
- **📊 完整监控**: 健康检查、性能指标和告警系统

## 🏗️ 系统架构

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Kafka Topics  │───▶│ API Service  │───▶│   ClickHouse    │
│                 │    │              │    │   Database      │
│ • exec.reports  │    │ ┌──────────┐ │    │                 │
│ • risk.metrics │    │ │  Kafka   │ │    │ • exec_reports  │
│ • alerts       │    │ │ Consumer │ │    │ • risk_metrics  │
│ • system.health│    │ └──────────┘ │    │ • alerts        │
└─────────────────┘    │              │    │ • positions     │
                       │ ┌──────────┐ │    │ • fills         │
┌─────────────────┐    │ │   REST   │ │    └─────────────────┘
│  Web Clients    │◀───┤ │   API    │ │
│                 │    │ └──────────┘ │    ┌─────────────────┐
│ • Frontend      │    │              │    │   Cache Layer   │
│ • Mobile App    │    │ ┌──────────┐ │    │                 │
│ • Trading Bot   │    │ │WebSocket │ │    │ • Memory Cache  │
└─────────────────┘    │ │ Manager  │ │    │ • Redis Cache   │
                       │ └──────────┘ │    │ • Query Cache   │
                       └──────────────┘    └─────────────────┘
```

## 📋 API端点

### REST API

| 端点 | 方法 | 描述 | 响应时间 |
|------|------|------|----------|
| `/api/pnl` | GET | PnL查询和时间序列 | <50ms |
| `/api/positions` | GET | 当前持仓和历史快照 | <30ms |
| `/api/fills` | GET | 成交记录和分析 | <100ms |
| `/api/risk` | GET | 风险指标和限制状态 | <50ms |
| `/api/alerts` | GET | 告警历史和管理 | <50ms |
| `/api/performance` | GET | 策略性能分析 | <200ms |
| `/health` | GET | 系统健康检查 | <10ms |

### WebSocket端点

| 端点 | 描述 | 更新频率 |
|------|------|----------|
| `/ws/alerts` | 实时告警推送 | 事件驱动 |
| `/ws/pnl` | 实时PnL更新 | 5秒 |
| `/ws/positions` | 持仓变化推送 | 10秒 |
| `/ws/health` | 系统状态监控 | 30秒 |
| `/ws` | 通用WebSocket | 按需订阅 |

## 🚀 快速开始

### 环境要求

- Python 3.11+
- ClickHouse 23.8+
- Apache Kafka 2.8+
- Redis (可选)
- 8GB+ RAM

### 安装部署

#### 1. 本地开发环境

```bash
# 克隆项目
git clone <repository-url>
cd DipMaster-Trading-System/src/api

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
export CLICKHOUSE_HOST=localhost
export KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# 启动服务
./start.sh -d
```

#### 2. Docker部署

```bash
# 使用Docker Compose启动完整环境
docker-compose up -d

# 检查服务状态
docker-compose ps

# 查看日志
docker-compose logs -f dipmaster-api
```

#### 3. 生产环境

```bash
# 生产模式启动
./start.sh -p

# 或者使用systemd服务
sudo systemctl start dipmaster-api
sudo systemctl enable dipmaster-api
```

### 配置参数

#### 环境变量

| 变量名 | 默认值 | 描述 |
|--------|--------|------|
| `CLICKHOUSE_HOST` | localhost | ClickHouse主机 |
| `CLICKHOUSE_PORT` | 9000 | ClickHouse端口 |
| `CLICKHOUSE_DATABASE` | dipmaster | 数据库名称 |
| `KAFKA_BOOTSTRAP_SERVERS` | localhost:9092 | Kafka服务器 |
| `API_HOST` | 0.0.0.0 | API监听地址 |
| `API_PORT` | 8000 | API监听端口 |
| `API_LOG_LEVEL` | INFO | 日志级别 |

#### 性能调优

```bash
# 缓存配置
export API_ENABLE_CACHING=true
export API_CACHE_TTL=300

# 限流配置
export API_RATE_LIMIT_CALLS=1000
export API_RATE_LIMIT_PERIOD=60

# WebSocket配置
export WS_MAX_CONNECTIONS=2000
export WS_HEARTBEAT_INTERVAL=30
```

## 📊 监控和运维

### 健康检查

```bash
# 系统健康状态
curl http://localhost:8000/health

# 详细性能指标
curl http://localhost:8000/health/metrics

# WebSocket统计
curl http://localhost:8000/ws/stats
```

### 日志监控

```bash
# 查看服务日志
tail -f logs/dipmaster_api.log

# 错误日志过滤
grep ERROR logs/dipmaster_api.log

# 性能日志分析
grep "Process-Time" logs/dipmaster_api.log
```

### 性能指标

| 指标 | 目标值 | 监控方式 |
|------|--------|----------|
| API响应时间 | P95 < 200ms | Prometheus |
| WebSocket连接数 | < 1000 | 内置统计 |
| 缓存命中率 | > 80% | 缓存统计 |
| Kafka消费滞后 | < 1000 | Kafka监控 |
| 数据库连接 | < 50 | ClickHouse监控 |

## 🔧 开发指南

### 项目结构

```
src/api/
├── main.py                 # 主程序入口
├── config.py              # 配置管理
├── requirements.txt       # 依赖包
├── schemas/               # 数据模式
│   ├── kafka_events.py    # Kafka事件模式
│   └── api_responses.py   # API响应模式
├── database/              # 数据库层
│   ├── clickhouse_client.py  # ClickHouse客户端
│   ├── schema.py          # 数据库模式
│   └── models.py          # 数据模型
├── kafka/                 # Kafka服务
│   ├── consumer.py        # 消费者管理
│   └── handlers.py        # 事件处理器
├── rest/                  # REST API
│   ├── app.py            # FastAPI应用
│   ├── middleware.py     # 中间件
│   ├── dependencies.py   # 依赖注入
│   └── endpoints/        # API端点
├── websocket/            # WebSocket服务
│   ├── manager.py        # 连接管理
│   └── handlers.py       # 消息处理
└── cache/                # 缓存服务
    ├── memory_cache.py   # 内存缓存
    └── cache_manager.py  # 缓存管理
```

### 开发环境设置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 安装开发依赖
pip install -r requirements.txt
pip install pytest black isort

# 代码格式化
black .
isort .

# 运行测试
pytest tests/
```

### API开发示例

```python
from fastapi import APIRouter, Depends
from ..database import ClickHouseClient
from ..dependencies import get_db_client

router = APIRouter()

@router.get("/api/custom")
async def custom_endpoint(
    db: ClickHouseClient = Depends(get_db_client)
):
    # 查询数据
    result = await db.query_to_dataframe("SELECT * FROM custom_table")
    
    # 返回响应
    return {
        "data": result.to_dict('records'),
        "count": len(result)
    }
```

### WebSocket开发示例

```python
from ..websocket import WebSocketManager, SubscriptionType

async def handle_custom_event(ws_manager: WebSocketManager, event_data):
    message = {
        "type": "custom_update",
        "timestamp": datetime.utcnow().isoformat(),
        "data": event_data
    }
    
    await ws_manager.broadcast(SubscriptionType.ALERTS, message)
```

## 📈 性能优化

### 数据库优化

```sql
-- ClickHouse查询优化
SELECT 
    symbol,
    sum(total_pnl) as pnl
FROM exec_reports 
WHERE timestamp >= now() - INTERVAL 1 DAY
GROUP BY symbol
ORDER BY pnl DESC
LIMIT 10
```

### 缓存策略

```python
# 使用缓存装饰器
@cache_result(category='performance', ttl=300)
async def get_performance_data():
    # 耗时查询
    return expensive_query_result
```

### 批量处理

```python
# Kafka批量消费
async def process_batch(messages):
    # 批量处理100条消息
    batch_size = 100
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i + batch_size]
        await process_message_batch(batch)
```

## 🛡️ 安全配置

### API认证

```bash
# 启用API密钥认证
export API_ENABLE_AUTH=true
export API_KEY=your-secret-api-key
```

### 限流保护

```python
# 自定义限流规则
rate_limits = {
    '/api/pnl': {'calls': 100, 'period': 60},
    '/api/performance': {'calls': 10, 'period': 60}
}
```

### 数据安全

- API密钥轮换
- 网络访问控制
- 数据加密传输
- 审计日志记录

## 🚨 故障排除

### 常见问题

**1. 数据库连接失败**
```bash
# 检查ClickHouse状态
curl http://localhost:8123/ping

# 测试连接
python -c "from clickhouse_connect import get_client; print(get_client().ping())"
```

**2. Kafka消费异常**
```bash
# 检查Kafka主题
kafka-topics --list --bootstrap-server localhost:9092

# 查看消费者组状态
kafka-consumer-groups --describe --group dipmaster-api --bootstrap-server localhost:9092
```

**3. WebSocket连接问题**
```javascript
// 客户端重连机制
const ws = new WebSocket('ws://localhost:8000/ws/alerts');
ws.onclose = () => {
    setTimeout(() => connect(), 5000);
};
```

**4. 性能问题**
```bash
# 查看系统资源
htop
iotop

# 分析慢查询
grep "slow" logs/dipmaster_api.log
```

### 诊断工具

```bash
# 健康检查脚本
./start.sh --check-only

# 性能测试
ab -n 1000 -c 10 http://localhost:8000/api/pnl

# 内存分析
python -m memory_profiler main.py
```

## 📚 相关文档

- [ClickHouse文档](https://clickhouse.com/docs)
- [FastAPI文档](https://fastapi.tiangolo.com)
- [Kafka文档](https://kafka.apache.org/documentation)
- [WebSocket协议](https://tools.ietf.org/html/rfc6455)

## 🤝 贡献指南

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🆘 支持

如果您遇到问题或有建议，请：

1. 查看[FAQ](docs/FAQ.md)
2. 搜索[Issues](../../issues)
3. 创建新的[Issue](../../issues/new)
4. 联系技术支持

---

**🚀 DipMaster数据API服务 - 为高频交易而生**