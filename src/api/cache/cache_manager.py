"""
缓存管理器
=========

统一管理不同类型的缓存。
"""

import asyncio
import logging
from typing import Any, Optional, Dict, Union
from datetime import datetime, timedelta

from .memory_cache import MemoryCache, CacheKeyBuilder

logger = logging.getLogger(__name__)


class CacheManager:
    """统一缓存管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化内存缓存
        self.memory_cache = MemoryCache(
            max_size=config.get('memory_max_size', 1000),
            default_ttl=config.get('default_ttl', 300)
        )
        
        # Redis缓存（可选）
        self.redis_cache = None
        if config.get('enable_redis', False):
            try:
                from .redis_cache import RedisCache
                self.redis_cache = RedisCache(config.get('redis_config', {}))
            except ImportError:
                logger.warning("Redis缓存不可用，将只使用内存缓存")
        
        # 缓存策略配置
        self.cache_strategies = {
            # API查询缓存 - 短期缓存
            'api_query': {
                'ttl': 60,  # 1分钟
                'use_redis': False
            },
            
            # 性能数据缓存 - 中期缓存  
            'performance': {
                'ttl': 300,  # 5分钟
                'use_redis': True
            },
            
            # 风险指标缓存 - 短期缓存
            'risk_metrics': {
                'ttl': 30,  # 30秒
                'use_redis': False
            },
            
            # 持仓数据缓存 - 短期缓存
            'positions': {
                'ttl': 15,  # 15秒
                'use_redis': False
            },
            
            # PnL数据缓存 - 短期缓存
            'pnl': {
                'ttl': 10,  # 10秒
                'use_redis': False
            },
            
            # 告警数据缓存 - 很短期缓存
            'alerts': {
                'ttl': 5,  # 5秒
                'use_redis': False
            },
            
            # 静态数据缓存 - 长期缓存
            'static': {
                'ttl': 3600,  # 1小时
                'use_redis': True
            }
        }
        
        self.running = False
    
    async def start(self):
        """启动缓存管理器"""
        if self.running:
            return
        
        # 启动内存缓存
        await self.memory_cache.start()
        
        # 启动Redis缓存
        if self.redis_cache:
            await self.redis_cache.start()
        
        self.running = True
        logger.info("缓存管理器已启动")
    
    async def stop(self):
        """停止缓存管理器"""
        if not self.running:
            return
        
        self.running = False
        
        # 停止缓存服务
        await self.memory_cache.stop()
        
        if self.redis_cache:
            await self.redis_cache.stop()
        
        logger.info("缓存管理器已停止")
    
    async def get(self, key: str, category: str = 'api_query') -> Optional[Any]:
        """获取缓存值"""
        strategy = self.cache_strategies.get(category, self.cache_strategies['api_query'])
        
        # 选择缓存后端
        cache = self._get_cache_backend(strategy)
        
        try:
            return await cache.get(key)
        except Exception as e:
            logger.error(f"获取缓存失败 [{key}]: {e}")
            return None
    
    async def set(self, key: str, value: Any, category: str = 'api_query', ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        strategy = self.cache_strategies.get(category, self.cache_strategies['api_query'])
        
        if ttl is None:
            ttl = strategy['ttl']
        
        # 选择缓存后端
        cache = self._get_cache_backend(strategy)
        
        try:
            return await cache.set(key, value, ttl)
        except Exception as e:
            logger.error(f"设置缓存失败 [{key}]: {e}")
            return False
    
    async def delete(self, key: str, category: str = 'api_query') -> bool:
        """删除缓存值"""
        strategy = self.cache_strategies.get(category, self.cache_strategies['api_query'])
        
        # 选择缓存后端
        cache = self._get_cache_backend(strategy)
        
        try:
            return await cache.delete(key)
        except Exception as e:
            logger.error(f"删除缓存失败 [{key}]: {e}")
            return False
    
    async def exists(self, key: str, category: str = 'api_query') -> bool:
        """检查键是否存在"""
        strategy = self.cache_strategies.get(category, self.cache_strategies['api_query'])
        cache = self._get_cache_backend(strategy)
        
        try:
            return await cache.exists(key)
        except Exception as e:
            logger.error(f"检查缓存存在性失败 [{key}]: {e}")
            return False
    
    async def clear(self, category: Optional[str] = None):
        """清空缓存"""
        if category:
            # 清空特定分类的缓存
            pattern = f"dipmaster:{category}:*"
            
            # 清空内存缓存
            keys = await self.memory_cache.keys(pattern)
            for key in keys:
                await self.memory_cache.delete(key)
            
            # 清空Redis缓存
            if self.redis_cache:
                keys = await self.redis_cache.keys(pattern)
                for key in keys:
                    await self.redis_cache.delete(key)
        else:
            # 清空所有缓存
            await self.memory_cache.clear()
            if self.redis_cache:
                await self.redis_cache.clear()
    
    def _get_cache_backend(self, strategy: Dict[str, Any]):
        """选择缓存后端"""
        if strategy.get('use_redis', False) and self.redis_cache:
            return self.redis_cache
        return self.memory_cache
    
    # 高级缓存方法
    
    async def get_or_set(
        self, 
        key: str, 
        factory_func, 
        category: str = 'api_query',
        ttl: Optional[int] = None
    ) -> Any:
        """获取缓存值，如果不存在则调用工厂函数生成"""
        # 尝试获取缓存
        value = await self.get(key, category)
        if value is not None:
            return value
        
        # 生成新值
        try:
            if asyncio.iscoroutinefunction(factory_func):
                value = await factory_func()
            else:
                value = factory_func()
            
            # 缓存新值
            await self.set(key, value, category, ttl)
            return value
            
        except Exception as e:
            logger.error(f"工厂函数执行失败 [{key}]: {e}")
            raise
    
    async def cache_api_response(
        self, 
        endpoint: str, 
        params: Dict[str, Any], 
        response_data: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """缓存API响应"""
        cache_key = CacheKeyBuilder.build_query_key(endpoint, params)
        return await self.set(cache_key, response_data, 'api_query', ttl)
    
    async def get_cached_api_response(
        self, 
        endpoint: str, 
        params: Dict[str, Any]
    ) -> Optional[Any]:
        """获取缓存的API响应"""
        cache_key = CacheKeyBuilder.build_query_key(endpoint, params)
        return await self.get(cache_key, 'api_query')
    
    async def invalidate_pattern(self, pattern: str, category: str = 'api_query'):
        """根据模式失效缓存"""
        strategy = self.cache_strategies.get(category, self.cache_strategies['api_query'])
        cache = self._get_cache_backend(strategy)
        
        try:
            keys = await cache.keys(pattern)
            for key in keys:
                await cache.delete(key)
            
            logger.info(f"失效了 {len(keys)} 个缓存条目，模式: {pattern}")
            
        except Exception as e:
            logger.error(f"失效缓存模式失败 [{pattern}]: {e}")
    
    async def warm_up_cache(self, warm_up_func):
        """预热缓存"""
        try:
            logger.info("开始缓存预热")
            
            if asyncio.iscoroutinefunction(warm_up_func):
                await warm_up_func(self)
            else:
                warm_up_func(self)
            
            logger.info("缓存预热完成")
            
        except Exception as e:
            logger.error(f"缓存预热失败: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        stats = {
            'memory_cache': await self.memory_cache.get_stats(),
            'redis_cache': None,
            'strategies': self.cache_strategies,
            'running': self.running
        }
        
        if self.redis_cache:
            stats['redis_cache'] = await self.redis_cache.get_stats()
        
        return stats


# 缓存装饰器

def cache_result(category: str = 'api_query', ttl: Optional[int] = None, key_func=None):
    """缓存结果装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 获取缓存管理器（需要通过依赖注入）
            cache_manager = getattr(wrapper, '_cache_manager', None)
            if not cache_manager:
                # 没有缓存管理器，直接执行函数
                return await func(*args, **kwargs)
            
            # 构建缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # 默认键生成策略
                import json
                func_name = f"{func.__module__}.{func.__name__}"
                args_str = json.dumps([str(arg) for arg in args], default=str)
                kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
                cache_key = CacheKeyBuilder.build_key(
                    "func",
                    func_name, 
                    CacheKeyBuilder.hash_key(args_str + kwargs_str)
                )
            
            # 尝试从缓存获取
            cached_result = await cache_manager.get(cache_key, category)
            if cached_result is not None:
                return cached_result
            
            # 执行函数并缓存结果
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, category, ttl)
            
            return result
        
        return wrapper
    return decorator


async def cache_warm_up_function(cache_manager: CacheManager):
    """缓存预热函数示例"""
    # 预热静态数据
    static_data = {
        "supported_symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "risk_limits": {
            "max_positions": 3,
            "daily_loss_limit": -500,
            "max_drawdown": 5.0
        },
        "system_info": {
            "version": "1.0.0",
            "strategy": "dipmaster"
        }
    }
    
    for key, value in static_data.items():
        cache_key = CacheKeyBuilder.build_data_key("static", key)
        await cache_manager.set(cache_key, value, "static")
    
    logger.info("静态数据预热完成")