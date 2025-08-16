"""
Redis缓存和数据聚合服务
实现热点数据缓存、数据预聚合、查询优化
"""

import asyncio
import json
import pickle
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from decimal import Decimal
import redis.asyncio as redis
from redis.asyncio import Redis
import structlog

from .config import RedisConfig

logger = structlog.get_logger(__name__)

class DecimalEncoder(json.JSONEncoder):
    """支持Decimal类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

class CacheKey:
    """缓存键生成器"""
    
    @staticmethod
    def pnl_data(account_id: str, start_time: str, end_time: str, symbols: str = "all") -> str:
        return f"pnl:{account_id}:{start_time}:{end_time}:{symbols}"
    
    @staticmethod
    def positions_latest(account_id: str) -> str:
        return f"positions:latest:{account_id}"
    
    @staticmethod
    def risk_latest(account_id: str) -> str:
        return f"risk:latest:{account_id}"
    
    @staticmethod
    def fills_data(account_id: str, start_time: str, end_time: str) -> str:
        return f"fills:{account_id}:{start_time}:{end_time}"
    
    @staticmethod
    def strategy_performance(strategy_id: str, start_time: str, end_time: str) -> str:
        return f"performance:{strategy_id}:{start_time}:{end_time}"
    
    @staticmethod
    def account_summary(account_id: str) -> str:
        return f"summary:{account_id}"
    
    @staticmethod
    def market_data(symbol: str) -> str:
        return f"market:{symbol}"
    
    @staticmethod
    def alert_data(account_id: str) -> str:
        return f"alerts:{account_id}"

class DataAggregator:
    """数据聚合器"""
    
    def __init__(self, redis_client: Redis):
        self.redis = redis_client
    
    async def aggregate_pnl_by_symbol(self, pnl_data: List[Dict]) -> Dict[str, Any]:
        """按交易对聚合PnL数据"""
        symbol_pnl = {}
        
        for item in pnl_data:
            symbol = item['symbol']
            if symbol not in symbol_pnl:
                symbol_pnl[symbol] = {
                    'realized_pnl': 0,
                    'unrealized_pnl': 0,
                    'total_pnl': 0,
                    'trade_count': 0,
                    'volume': 0
                }
            
            symbol_pnl[symbol]['realized_pnl'] += float(item.get('realized_pnl', 0))
            symbol_pnl[symbol]['unrealized_pnl'] += float(item.get('unrealized_pnl', 0))
            symbol_pnl[symbol]['total_pnl'] += float(item.get('total_pnl', 0))
            symbol_pnl[symbol]['trade_count'] += 1
            symbol_pnl[symbol]['volume'] += float(item.get('volume', 0))
        
        return symbol_pnl
    
    async def aggregate_pnl_by_time(self, pnl_data: List[Dict], interval: str = "1h") -> Dict[str, Any]:
        """按时间聚合PnL数据"""
        time_pnl = {}
        
        for item in pnl_data:
            # 解析时间戳并按间隔聚合
            timestamp = datetime.fromisoformat(item['timestamp'].replace('Z', '+00:00'))
            
            if interval == "1h":
                time_key = timestamp.replace(minute=0, second=0, microsecond=0).isoformat()
            elif interval == "1d":
                time_key = timestamp.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            else:
                time_key = timestamp.isoformat()
            
            if time_key not in time_pnl:
                time_pnl[time_key] = {
                    'realized_pnl': 0,
                    'unrealized_pnl': 0,
                    'total_pnl': 0,
                    'trade_count': 0
                }
            
            time_pnl[time_key]['realized_pnl'] += float(item.get('realized_pnl', 0))
            time_pnl[time_key]['unrealized_pnl'] += float(item.get('unrealized_pnl', 0))
            time_pnl[time_key]['total_pnl'] += float(item.get('total_pnl', 0))
            time_pnl[time_key]['trade_count'] += 1
        
        return time_pnl
    
    async def calculate_portfolio_metrics(self, positions: List[Dict], risk_data: Dict) -> Dict[str, Any]:
        """计算组合指标"""
        total_value = sum(float(pos.get('market_value', 0)) for pos in positions)
        total_pnl = sum(float(pos.get('unrealized_pnl', 0)) for pos in positions)
        
        metrics = {
            'total_portfolio_value': total_value,
            'total_unrealized_pnl': total_pnl,
            'position_count': len(positions),
            'largest_position': max((float(pos.get('market_value', 0)) for pos in positions), default=0),
            'concentration_ratio': 0,
            'diversification_score': 0
        }
        
        # 计算集中度比率
        if total_value > 0:
            largest_position_ratio = metrics['largest_position'] / total_value
            metrics['concentration_ratio'] = largest_position_ratio
            
            # 简单的分散化评分 (越分散分数越高)
            if len(positions) > 0:
                avg_position_size = total_value / len(positions)
                variance = sum((float(pos.get('market_value', 0)) - avg_position_size) ** 2 for pos in positions) / len(positions)
                metrics['diversification_score'] = max(0, 1 - (variance / (avg_position_size ** 2)))
        
        # 整合风险数据
        if risk_data:
            metrics.update({
                'var_1d': risk_data.get('var_1d'),
                'max_drawdown': risk_data.get('max_drawdown'),
                'leverage': risk_data.get('leverage'),
                'risk_score': risk_data.get('risk_score')
            })
        
        return metrics

class CacheManager:
    """缓存管理器主类"""
    
    def __init__(self, config: RedisConfig):
        self.config = config
        self.redis: Optional[Redis] = None
        self.aggregator: Optional[DataAggregator] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def initialize(self):
        """初始化Redis连接"""
        try:
            self.redis = redis.Redis(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                max_connections=self.config.max_connections,
                socket_timeout=self.config.socket_timeout,
                socket_connect_timeout=self.config.socket_connect_timeout,
                decode_responses=False  # 支持二进制数据
            )
            
            # 测试连接
            await self.redis.ping()
            
            # 初始化聚合器
            self.aggregator = DataAggregator(self.redis)
            
            logger.info("Redis缓存管理器初始化完成")
            
        except Exception as e:
            logger.error(f"Redis初始化失败: {e}")
            raise
    
    async def cache_data(self, key: str, data: Any, ttl: Optional[int] = None, 
                        data_type: str = "json") -> bool:
        """缓存数据"""
        try:
            if not self.redis:
                return False
            
            if data_type == "json":
                # JSON序列化
                serialized_data = json.dumps(data, cls=DecimalEncoder, ensure_ascii=False)
            elif data_type == "pickle":
                # Pickle序列化（支持复杂对象）
                serialized_data = pickle.dumps(data)
            else:
                # 直接存储字符串
                serialized_data = str(data)
            
            # 设置TTL
            cache_ttl = ttl or self.config.cache_ttl.get(key.split(':')[0], 300)
            
            await self.redis.setex(key, cache_ttl, serialized_data)
            logger.debug(f"缓存数据: {key}, TTL: {cache_ttl}秒")
            return True
            
        except Exception as e:
            logger.error(f"缓存数据失败 {key}: {e}")
            return False
    
    async def get_cached_data(self, key: str, data_type: str = "json") -> Optional[Any]:
        """获取缓存数据"""
        try:
            if not self.redis:
                return None
            
            cached_data = await self.redis.get(key)
            if not cached_data:
                return None
            
            if data_type == "json":
                return json.loads(cached_data)
            elif data_type == "pickle":
                return pickle.loads(cached_data)
            else:
                return cached_data.decode('utf-8')
                
        except Exception as e:
            logger.error(f"获取缓存数据失败 {key}: {e}")
            return None
    
    async def invalidate_cache(self, pattern: str) -> int:
        """删除匹配模式的缓存"""
        try:
            if not self.redis:
                return 0
            
            keys = await self.redis.keys(pattern)
            if keys:
                deleted_count = await self.redis.delete(*keys)
                logger.info(f"删除缓存: {deleted_count} 个键, 模式: {pattern}")
                return deleted_count
            return 0
            
        except Exception as e:
            logger.error(f"删除缓存失败 {pattern}: {e}")
            return 0
    
    async def cache_exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        try:
            if not self.redis:
                return False
            return bool(await self.redis.exists(key))
        except Exception as e:
            logger.error(f"检查缓存存在性失败 {key}: {e}")
            return False
    
    async def get_cache_ttl(self, key: str) -> int:
        """获取缓存剩余TTL"""
        try:
            if not self.redis:
                return -1
            return await self.redis.ttl(key)
        except Exception as e:
            logger.error(f"获取缓存TTL失败 {key}: {e}")
            return -1
    
    # 业务数据缓存方法
    
    async def update_pnl_data(self, pnl_updates: Dict[str, Any]):
        """更新PnL缓存数据"""
        try:
            # 为每个交易对更新PnL缓存
            for symbol, pnl_info in pnl_updates.items():
                key = f"pnl:symbol:{symbol}"
                await self.cache_data(key, pnl_info, ttl=60)
            
            # 更新聚合PnL数据
            if self.aggregator:
                aggregated_data = await self.aggregator.aggregate_pnl_by_symbol(
                    [{"symbol": k, **v} for k, v in pnl_updates.items()]
                )
                await self.cache_data("pnl:aggregated", aggregated_data, ttl=60)
                
        except Exception as e:
            logger.error(f"更新PnL缓存失败: {e}")
    
    async def update_fills_data(self, fills: List[Dict[str, Any]]):
        """更新成交记录缓存"""
        try:
            # 按账户ID分组缓存
            account_fills = {}
            for fill in fills:
                account_id = fill.get('account_id')
                if account_id:
                    if account_id not in account_fills:
                        account_fills[account_id] = []
                    account_fills[account_id].append(fill)
            
            # 缓存最新成交记录
            for account_id, account_fill_list in account_fills.items():
                key = f"fills:latest:{account_id}"
                # 只保留最新的100条记录
                sorted_fills = sorted(account_fill_list, key=lambda x: x['timestamp'], reverse=True)[:100]
                await self.cache_data(key, sorted_fills, ttl=30)
                
        except Exception as e:
            logger.error(f"更新成交记录缓存失败: {e}")
    
    async def update_risk_data(self, risk_data: Dict[str, Any]):
        """更新风险指标缓存"""
        try:
            for account_id, risk_info in risk_data.items():
                key = CacheKey.risk_latest(account_id)
                await self.cache_data(key, risk_info, ttl=60)
                
        except Exception as e:
            logger.error(f"更新风险指标缓存失败: {e}")
    
    async def update_strategy_performance(self, performance_data: Dict[str, Any]):
        """更新策略性能缓存"""
        try:
            for strategy_id, perf_info in performance_data.items():
                key = f"performance:latest:{strategy_id}"
                await self.cache_data(key, perf_info, ttl=300)  # 5分钟TTL
                
        except Exception as e:
            logger.error(f"更新策略性能缓存失败: {e}")
    
    async def update_alert_data(self, alerts: List[Dict[str, Any]]):
        """更新告警缓存"""
        try:
            # 按账户ID和严重性分组
            account_alerts = {}
            for alert in alerts:
                account_id = alert.get('account_id')
                if account_id:
                    if account_id not in account_alerts:
                        account_alerts[account_id] = {'CRITICAL': [], 'WARNING': [], 'INFO': []}
                    
                    severity = alert.get('severity', 'INFO')
                    if severity in account_alerts[account_id]:
                        account_alerts[account_id][severity].append(alert)
            
            # 缓存最新告警
            for account_id, alerts_by_severity in account_alerts.items():
                key = CacheKey.alert_data(account_id)
                await self.cache_data(key, alerts_by_severity, ttl=30)
                
        except Exception as e:
            logger.error(f"更新告警缓存失败: {e}")
    
    async def get_aggregated_portfolio_data(self, account_id: str) -> Optional[Dict[str, Any]]:
        """获取聚合的组合数据"""
        try:
            # 尝试从缓存获取
            cache_key = f"portfolio:aggregated:{account_id}"
            cached_data = await self.get_cached_data(cache_key)
            
            if cached_data:
                return cached_data
            
            # 如果缓存不存在，从各个缓存组装数据
            positions_key = CacheKey.positions_latest(account_id)
            risk_key = CacheKey.risk_latest(account_id)
            alerts_key = CacheKey.alert_data(account_id)
            
            positions_data = await self.get_cached_data(positions_key) or []
            risk_data = await self.get_cached_data(risk_key) or {}
            alerts_data = await self.get_cached_data(alerts_key) or {}
            
            # 计算聚合指标
            if self.aggregator:
                portfolio_metrics = await self.aggregator.calculate_portfolio_metrics(
                    positions_data, risk_data
                )
                
                aggregated_data = {
                    'account_id': account_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'positions': positions_data,
                    'risk_metrics': risk_data,
                    'alerts': alerts_data,
                    'portfolio_metrics': portfolio_metrics
                }
                
                # 缓存聚合数据
                await self.cache_data(cache_key, aggregated_data, ttl=60)
                return aggregated_data
            
            return None
            
        except Exception as e:
            logger.error(f"获取聚合组合数据失败: {e}")
            return None
    
    async def precompute_dashboard_data(self, account_ids: List[str]):
        """预计算仪表板数据"""
        try:
            for account_id in account_ids:
                # 预计算组合概览数据
                await self.get_aggregated_portfolio_data(account_id)
                
                # 预计算今日PnL数据
                today = datetime.utcnow().date()
                start_time = datetime.combine(today, datetime.min.time()).isoformat()
                end_time = datetime.utcnow().isoformat()
                
                pnl_key = CacheKey.pnl_data(account_id, start_time, end_time)
                
                # 这里可以从数据库获取数据并缓存
                # 由于这是预计算，可以后台异步执行
                
            logger.info(f"预计算仪表板数据完成，账户数: {len(account_ids)}")
            
        except Exception as e:
            logger.error(f"预计算仪表板数据失败: {e}")
    
    async def start_cleanup_task(self):
        """启动缓存清理任务"""
        self._running = True
        
        async def cleanup_expired_cache():
            while self._running:
                try:
                    # 清理过期的临时缓存
                    patterns_to_clean = [
                        "temp:*",
                        "session:*",
                        "locks:*"
                    ]
                    
                    for pattern in patterns_to_clean:
                        await self.invalidate_cache(pattern)
                    
                    # 清理内存使用率高的缓存
                    if self.redis:
                        memory_info = await self.redis.info("memory")
                        memory_usage = memory_info.get("used_memory", 0)
                        max_memory = memory_info.get("maxmemory", 0)
                        
                        if max_memory > 0 and memory_usage / max_memory > 0.8:  # 80%内存使用率
                            logger.warning("Redis内存使用率过高，开始清理缓存")
                            # 清理最老的缓存
                            await self.invalidate_cache("pnl:*")
                            await self.invalidate_cache("fills:*")
                    
                    await asyncio.sleep(3600)  # 每小时清理一次
                    
                except Exception as e:
                    logger.error(f"缓存清理任务异常: {e}")
                    await asyncio.sleep(600)  # 出错时10分钟后重试
        
        self._cleanup_task = asyncio.create_task(cleanup_expired_cache())
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        try:
            if not self.redis:
                return {"status": "unavailable"}
            
            info = await self.redis.info()
            
            stats = {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "used_memory_peak": info.get("used_memory_peak_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "expired_keys": info.get("expired_keys", 0),
                "evicted_keys": info.get("evicted_keys", 0)
            }
            
            # 计算命中率
            hits = stats["keyspace_hits"]
            misses = stats["keyspace_misses"]
            if hits + misses > 0:
                stats["hit_rate"] = hits / (hits + misses)
            else:
                stats["hit_rate"] = 0
            
            return stats
            
        except Exception as e:
            logger.error(f"获取缓存统计失败: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_health_status(self) -> Dict[str, Any]:
        """获取缓存健康状态"""
        try:
            if not self.redis:
                return {"status": "unhealthy", "reason": "Redis连接未初始化"}
            
            # 测试连接
            await self.redis.ping()
            
            # 获取基本信息
            stats = await self.get_cache_stats()
            
            return {
                "status": "healthy",
                "stats": stats
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def shutdown(self):
        """关闭缓存管理器"""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self.redis:
            await self.redis.aclose()
            self.redis = None
        
        logger.info("缓存管理器关闭完成")