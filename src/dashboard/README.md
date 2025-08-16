# DipMaster Enhanced V4 - Dashboard API

高性能实时交易仪表板API服务，为DipMaster Enhanced V4交易系统提供完整的数据服务支持。

## 功能特性

### 🚀 核心功能
- **Kafka事件消费**: 实时消费exec.reports.v1、risk.metrics.v1、alerts.v1、strategy.performance.v1事件流
- **ClickHouse时序数据库**: 高性能OLAP查询，自动分片和数据压缩
- **REST API**: 完整的PnL、仓位、成交、风险、性能数据API
- **WebSocket实时流**: 毫秒级实时数据推送
- **Redis缓存**: 智能缓存和数据预聚合
- **JWT认证**: 完整的用户认证和权限控制
- **监控告警**: 全面的性能监控和健康检查

### 📊 API端点

#### REST API
- `GET /api/v1/pnl` - PnL曲线和历史收益
- `GET /api/v1/positions` - 当前持仓和历史仓位
- `GET /api/v1/fills` - 成交记录和执行分析
- `GET /api/v1/risk` - 风险指标和VaR数据
- `GET /api/v1/performance` - 策略表现统计
- `GET /api/v1/summary` - 账户概览数据
- `GET /api/v1/health` - 系统健康状态

#### WebSocket频道
- `/ws/alerts` - 实时告警推送
- `/ws/positions` - 实时仓位更新
- `/ws/pnl` - 实时损益更新
- `/ws/risk` - 实时风险指标

### 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Dashboard                       │
└─────────────────────┬───────────────────────────────────────┘
                      │ HTTP/WebSocket
┌─────────────────────▼───────────────────────────────────────┐
│                  FastAPI Gateway                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │     JWT     │ │ Rate Limit  │ │    CORS     │          │
│  │    Auth     │ │   & Quota   │ │  & Security │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                Business Logic Layer                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │    API      │ │  WebSocket  │ │   Kafka     │          │
│  │  Handlers   │ │   Manager   │ │  Consumer   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Data Access Layer                          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │ ClickHouse  │ │    Redis    │ │  Monitoring │          │
│  │  Database   │ │    Cache    │ │   Service   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## 快速开始

### 环境要求
- Python 3.11+
- ClickHouse 23.10+
- Redis 7.2+
- Apache Kafka 2.8+
- Docker & Docker Compose (可选)

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd DipMaster-Trading-System/src/dashboard

# 安装Python依赖
pip install -r requirements.txt
```

### 配置文件

编辑 `config/dashboard_config.json`：

```json
{
  "kafka": {
    "bootstrap_servers": ["localhost:9092"],
    "consumer_group": "dipmaster_dashboard_v4"
  },
  "database": {
    "host": "localhost",
    "port": 9000,
    "database": "dipmaster"
  },
  "redis": {
    "host": "localhost",
    "port": 6379,
    "db": 0
  },
  "auth": {
    "jwt_secret_key": "your-secret-key-change-in-production"
  }
}
```

### 启动方式

#### 1. 开发模式

```bash
# 直接启动
python start_dashboard.py dev

# 或使用主模块
python main.py --config config/dashboard_config.json --reload
```

#### 2. 生产模式

```bash
# 使用启动脚本
python start_dashboard.py prod --workers 4

# 或直接使用gunicorn
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker
```

#### 3. Docker模式

```bash
# 启动所有服务
python start_dashboard.py docker

# 或直接使用docker-compose
docker-compose up -d
```

## API使用示例

### 认证

```bash
# 获取访问令牌
curl -X POST "http://localhost:8080/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# 响应
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### 查询PnL数据

```bash
curl -X GET "http://localhost:8080/api/v1/pnl?account_id=default&start_time=2024-01-01T00:00:00Z&end_time=2024-01-02T00:00:00Z" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### WebSocket连接

```javascript
// 连接WebSocket
const ws = new WebSocket('ws://localhost:8080/ws/alerts');

// 发送认证
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'YOUR_JWT_TOKEN'
  }));
};

// 接收消息
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Alert received:', data);
};
```

## 性能优化

### 缓存策略
- **热点数据**: 自动缓存最近访问的数据
- **预聚合**: 后台预计算常用指标
- **分层缓存**: 内存 + Redis双层缓存

### 查询优化
- **时间分区**: 按月分区存储历史数据
- **索引优化**: 多维度复合索引
- **批量查询**: 减少数据库往返次数

### 并发处理
- **异步IO**: 全异步非阻塞架构
- **连接池**: 数据库连接复用
- **消息队列**: 削峰填谷处理突发流量

## 监控和运维

### 健康检查

```bash
# 检查服务健康状态
curl http://localhost:8080/health

# 检查详细指标
curl http://localhost:8080/api/v1/health
```

### 监控面板

- **Grafana**: http://localhost:3000 (admin/dipmaster123)
- **Prometheus**: http://localhost:9091
- **Kafka UI**: http://localhost:8081

### 日志查看

```bash
# API日志
tail -f logs/dashboard_api.log

# 系统日志
docker-compose logs -f dashboard-api
```

## 故障排除

### 常见问题

1. **Kafka连接失败**
```bash
# 检查Kafka状态
docker-compose ps kafka
curl -f http://localhost:8081  # Kafka UI
```

2. **ClickHouse查询慢**
```bash
# 检查查询性能
SELECT query, elapsed, memory_usage FROM system.processes;
```

3. **Redis内存不足**
```bash
# 检查Redis状态
redis-cli info memory
```

4. **WebSocket连接断开**
```bash
# 检查连接状态
curl http://localhost:8080/api/v1/websocket/stats
```

### 性能调优

1. **ClickHouse优化**
   - 增加内存分配: `max_memory_usage`
   - 调整合并策略: `merge_tree`设置
   - 优化分区策略: 按业务特点分区

2. **Redis优化**
   - 设置最大内存: `maxmemory`
   - 配置淘汰策略: `maxmemory-policy`
   - 启用持久化: `appendonly yes`

3. **Kafka优化**
   - 增加分区数: 提高并行度
   - 调整批次大小: `batch.size`
   - 优化压缩: `compression.type`

## 安全配置

### 生产环境配置

1. **更改默认密钥**
```json
{
  "auth": {
    "jwt_secret_key": "production-secret-key-256-bits-long"
  }
}
```

2. **启用HTTPS**
```bash
# 使用nginx反向代理
nginx -c /etc/nginx/nginx.conf
```

3. **配置防火墙**
```bash
# 只允许必要端口
ufw allow 8080/tcp
ufw enable
```

## API文档

启动服务后访问:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

## 开发指南

### 添加新的API端点

```python
# src/dashboard/api.py
@router.get("/new-endpoint")
async def new_endpoint(
    request: Request,
    current_user: dict = Depends(get_current_user)
):
    # 实现逻辑
    return APIResponse(success=True, data=result)
```

### 添加新的Kafka事件处理

```python
# src/dashboard/kafka_consumer.py
class NewEventProcessor(EventProcessor):
    async def process_batch(self, events):
        # 处理新事件类型
        pass
```

### 添加新的缓存策略

```python
# src/dashboard/cache.py
async def cache_new_data_type(self, data):
    # 实现新的缓存逻辑
    pass
```

## 许可证

本项目使用 MIT 许可证。详见 [LICENSE](../../LICENSE) 文件。

## 支持

如有问题或建议，请通过以下方式联系：
- GitHub Issues
- 邮件支持
- 技术文档