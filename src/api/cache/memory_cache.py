"""
内存缓存实现
===========

基于内存的高性能缓存实现。
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
import hashlib
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_access: datetime = None
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def touch(self):
        """更新访问时间"""
        self.access_count += 1
        self.last_access = datetime.utcnow()


class MemoryCache:
    """内存缓存实现"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0,
            'start_time': datetime.utcnow()
        }
        
        # 启动清理任务
        self.cleanup_task = None
        self.running = False
    
    async def start(self):
        """启动缓存服务"""
        if self.running:
            return
        
        self.running = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("内存缓存服务已启动")
    
    async def stop(self):
        """停止缓存服务"""
        if not self.running:
            return
        
        self.running = False
        
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.cache.clear()
        logger.info("内存缓存服务已停止")
    
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        entry = self.cache.get(key)
        
        if entry is None:
            self.stats['misses'] += 1
            return None
        
        if entry.is_expired():
            await self.delete(key)
            self.stats['misses'] += 1
            return None
        
        entry.touch()
        self.stats['hits'] += 1
        return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        try:
            if ttl is None:
                ttl = self.default_ttl
            
            expires_at = None
            if ttl > 0:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            
            entry = CacheEntry(
                key=key,
                value=value,
                created_at=datetime.utcnow(),
                expires_at=expires_at,
                last_access=datetime.utcnow()
            )
            
            # 检查容量限制
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_lru()
            
            self.cache[key] = entry
            self.stats['sets'] += 1
            return True
            
        except Exception as e:
            logger.error(f"设置缓存失败 [{key}]: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        if key in self.cache:
            del self.cache[key]
            self.stats['deletes'] += 1
            return True
        return False
    
    async def exists(self, key: str) -> bool:
        """检查键是否存在"""
        entry = self.cache.get(key)
        if entry and not entry.is_expired():
            return True
        return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """设置过期时间"""
        entry = self.cache.get(key)
        if entry:
            entry.expires_at = datetime.utcnow() + timedelta(seconds=ttl)
            return True
        return False
    
    async def clear(self):
        """清空缓存"""
        self.cache.clear()
        logger.info("缓存已清空")
    
    async def keys(self, pattern: str = "*") -> list:
        """获取匹配的键列表"""
        if pattern == "*":
            return list(self.cache.keys())
        
        # 简单的通配符匹配
        import fnmatch
        return [key for key in self.cache.keys() if fnmatch.fnmatch(key, pattern)]
    
    async def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_requests = self.stats['hits'] + self.stats['misses']
        hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        uptime = (datetime.utcnow() - self.stats['start_time']).total_seconds()
        
        return {
            **self.stats,
            'hit_rate_percent': hit_rate,
            'current_size': len(self.cache),
            'max_size': self.max_size,
            'uptime_seconds': uptime,
            'memory_usage_estimate': self._estimate_memory_usage()
        }
    
    def _estimate_memory_usage(self) -> int:
        """估算内存使用量"""
        try:
            import sys
            total_size = 0
            
            for entry in self.cache.values():
                total_size += sys.getsizeof(entry.key)
                total_size += sys.getsizeof(entry.value)
                total_size += sys.getsizeof(entry)
            
            return total_size
        except Exception:
            return 0
    
    async def _evict_lru(self):
        """LRU淘汰策略"""
        if not self.cache:
            return
        
        # 找到最少访问的条目
        lru_entry = min(
            self.cache.values(),
            key=lambda e: (e.last_access or e.created_at, e.access_count)
        )
        
        await self.delete(lru_entry.key)
        self.stats['evictions'] += 1
        
        logger.debug(f"LRU淘汰缓存条目: {lru_entry.key}")
    
    async def _cleanup_loop(self):
        """清理过期条目的循环任务"""
        while self.running:
            try:
                await asyncio.sleep(60)  # 每分钟清理一次
                
                if not self.running:
                    break
                
                # 清理过期条目
                expired_keys = []
                for key, entry in self.cache.items():
                    if entry.is_expired():
                        expired_keys.append(key)
                
                for key in expired_keys:
                    await self.delete(key)
                
                if expired_keys:
                    logger.debug(f"清理了 {len(expired_keys)} 个过期缓存条目")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"缓存清理异常: {e}")


class CacheKeyBuilder:
    """缓存键构建器"""
    
    @staticmethod
    def build_key(*parts: str, prefix: str = "dipmaster") -> str:
        """构建缓存键"""
        key_parts = [prefix] + list(parts)
        return ":".join(key_parts)
    
    @staticmethod
    def hash_key(key: str) -> str:
        """对键进行哈希处理"""
        return hashlib.md5(key.encode()).hexdigest()
    
    @staticmethod
    def build_query_key(endpoint: str, params: Dict[str, Any]) -> str:
        """构建查询缓存键"""
        # 对参数进行排序和序列化
        sorted_params = sorted(params.items())
        params_str = json.dumps(sorted_params, sort_keys=True, default=str)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        return CacheKeyBuilder.build_key("query", endpoint, params_hash)
    
    @staticmethod 
    def build_data_key(data_type: str, identifier: str) -> str:
        """构建数据缓存键"""
        return CacheKeyBuilder.build_key("data", data_type, identifier)


# 缓存装饰器
def cached(ttl: int = 300, key_prefix: str = "func"):
    """缓存装饰器"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # 构建缓存键
            func_name = f"{func.__module__}.{func.__name__}"
            args_str = json.dumps([str(arg) for arg in args], default=str)
            kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
            
            cache_key = CacheKeyBuilder.build_key(
                key_prefix, 
                func_name, 
                CacheKeyBuilder.hash_key(args_str + kwargs_str)
            )
            
            # 这里需要访问全局缓存实例
            # 在实际使用中，应该通过依赖注入获取缓存实例
            cache = getattr(wrapper, '_cache', None)
            if cache:
                # 尝试从缓存获取
                result = await cache.get(cache_key)
                if result is not None:
                    return result
                
                # 执行函数并缓存结果
                result = await func(*args, **kwargs)
                await cache.set(cache_key, result, ttl)
                return result
            else:
                # 没有缓存，直接执行
                return await func(*args, **kwargs)
        
        return wrapper
    return decorator