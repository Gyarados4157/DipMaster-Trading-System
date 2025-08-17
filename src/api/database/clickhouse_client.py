"""
ClickHouse异步客户端
==================

提供高性能异步数据库操作和连接池管理。
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
import json

import asyncpg
from clickhouse_connect import get_async_client
from clickhouse_connect.driver import AsyncClient
import pandas as pd

from ..config import DatabaseConfig

logger = logging.getLogger(__name__)


class ClickHouseClient:
    """ClickHouse异步客户端"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client: Optional[AsyncClient] = None
        self.connection_pool = None
        self._connected = False
        
    async def connect(self):
        """建立数据库连接"""
        try:
            self.client = await get_async_client(
                host=self.config.host,
                port=self.config.port,
                username=self.config.username,
                password=self.config.password,
                database=self.config.database,
                secure=self.config.secure,
                ca_cert=self.config.ca_cert,
                verify=self.config.verify_ssl,
                pool_size=self.config.pool_size,
                pool_timeout=self.config.pool_timeout,
                query_timeout=self.config.query_timeout,
                compress=True,  # 启用压缩
                settings={
                    'use_uncompressed_cache': 1,
                    'load_balancing': 'round_robin',
                    'max_threads': 8,
                    'max_memory_usage': '8000000000',  # 8GB
                    'prefer_localhost_replica': 1
                }
            )
            
            # 测试连接
            await self.client.ping()
            self._connected = True
            
            logger.info(f"ClickHouse连接成功: {self.config.host}:{self.config.port}")
            
            # 初始化数据库模式
            await self._initialize_schema()
            
        except Exception as e:
            logger.error(f"ClickHouse连接失败: {e}")
            raise
    
    async def disconnect(self):
        """关闭数据库连接"""
        if self.client:
            await self.client.close()
            self._connected = False
            logger.info("ClickHouse连接已关闭")
    
    async def _initialize_schema(self):
        """初始化数据库模式"""
        from .schema import DatabaseSchema
        schema = DatabaseSchema()
        
        # 创建数据库(如果不存在)
        await self.execute(f"CREATE DATABASE IF NOT EXISTS {self.config.database}")
        
        # 创建所有表
        for table_name, create_sql in schema.get_all_create_statements().items():
            try:
                await self.execute(create_sql)
                logger.info(f"表 {table_name} 创建成功")
            except Exception as e:
                logger.warning(f"表 {table_name} 创建失败: {e}")
    
    @property
    def is_connected(self) -> bool:
        """检查连接状态"""
        return self._connected and self.client is not None
    
    async def execute(self, query: str, parameters: Optional[Dict] = None) -> Any:
        """执行SQL查询"""
        if not self.is_connected:
            await self.connect()
        
        try:
            if parameters:
                result = await self.client.query(query, parameters=parameters)
            else:
                result = await self.client.query(query)
            return result
        except Exception as e:
            logger.error(f"查询执行失败: {query[:100]}... 错误: {e}")
            raise
    
    async def insert_dataframe(self, table: str, df: pd.DataFrame) -> int:
        """批量插入DataFrame数据"""
        if not self.is_connected:
            await self.connect()
        
        try:
            result = await self.client.insert_df(table, df)
            logger.debug(f"插入 {len(df)} 行数据到表 {table}")
            return len(df)
        except Exception as e:
            logger.error(f"数据插入失败到表 {table}: {e}")
            raise
    
    async def insert_batch(self, table: str, data: List[Dict], batch_size: int = 1000) -> int:
        """批量插入数据"""
        if not data:
            return 0
        
        if not self.is_connected:
            await self.connect()
        
        total_inserted = 0
        
        # 分批插入
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            
            try:
                # 转换为DataFrame
                df = pd.DataFrame(batch)
                
                # 处理特殊数据类型
                df = self._prepare_dataframe(df)
                
                await self.insert_dataframe(table, df)
                total_inserted += len(batch)
                
            except Exception as e:
                logger.error(f"批量插入失败 (batch {i//batch_size + 1}): {e}")
                # 继续处理下一批
                continue
        
        logger.info(f"成功插入 {total_inserted}/{len(data)} 行数据到表 {table}")
        return total_inserted
    
    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """准备DataFrame用于插入"""
        df = df.copy()
        
        # 处理Decimal类型
        for col in df.columns:
            if df[col].dtype == object:
                # 检查是否包含Decimal
                sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
                if isinstance(sample, Decimal):
                    df[col] = df[col].astype(float)
        
        # 处理datetime
        for col in df.columns:
            if df[col].dtype.name.startswith('datetime'):
                df[col] = pd.to_datetime(df[col], utc=True)
        
        # 处理JSON字段
        for col in ['metadata', 'context', 'metrics']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
        
        return df
    
    async def query_to_dataframe(self, query: str, parameters: Optional[Dict] = None) -> pd.DataFrame:
        """查询结果转为DataFrame"""
        if not self.is_connected:
            await self.connect()
        
        try:
            result = await self.client.query_df(query, parameters=parameters)
            return result
        except Exception as e:
            logger.error(f"查询转DataFrame失败: {e}")
            raise
    
    async def query_scalar(self, query: str, parameters: Optional[Dict] = None) -> Any:
        """查询单个值"""
        result = await self.execute(query, parameters)
        if result.result_rows:
            return result.result_rows[0][0]
        return None
    
    async def query_rows(self, query: str, parameters: Optional[Dict] = None) -> List[tuple]:
        """查询多行数据"""
        result = await self.execute(query, parameters)
        return result.result_rows if result.result_rows else []
    
    # PnL数据查询
    async def get_pnl_timeseries(
        self, 
        start_time: datetime, 
        end_time: datetime,
        symbol: Optional[str] = None,
        interval: str = '1h'
    ) -> pd.DataFrame:
        """获取PnL时间序列数据"""
        
        symbol_filter = f"AND symbol = '{symbol}'" if symbol else ""
        
        query = f"""
        SELECT 
            toStartOfInterval(timestamp, INTERVAL 1 {interval}) as time_bucket,
            {f"symbol," if symbol else ""}
            sum(total_realized_pnl) as realized_pnl,
            sum(total_unrealized_pnl) as unrealized_pnl,
            sum(total_realized_pnl + total_unrealized_pnl) as total_pnl,
            sum(total_commission) as commission,
            count(*) as trade_count
        FROM exec_reports 
        WHERE timestamp >= %(start_time)s 
          AND timestamp <= %(end_time)s
          {symbol_filter}
        GROUP BY time_bucket{', symbol' if symbol else ''}
        ORDER BY time_bucket
        """
        
        return await self.query_to_dataframe(query, {
            'start_time': start_time,
            'end_time': end_time
        })
    
    # 持仓数据查询
    async def get_current_positions(self) -> pd.DataFrame:
        """获取当前持仓"""
        query = """
        SELECT 
            symbol,
            quantity,
            avg_price,
            market_value,
            unrealized_pnl,
            realized_pnl,
            side,
            timestamp
        FROM risk_metrics 
        WHERE timestamp = (
            SELECT max(timestamp) FROM risk_metrics
        )
        AND quantity != 0
        ORDER BY abs(market_value) DESC
        """
        
        return await self.query_to_dataframe(query)
    
    # 成交数据查询  
    async def get_fills(
        self,
        start_time: datetime,
        end_time: datetime, 
        symbol: Optional[str] = None,
        page: int = 1,
        page_size: int = 100
    ) -> pd.DataFrame:
        """获取成交记录"""
        
        symbol_filter = f"AND symbol = '{symbol}'" if symbol else ""
        offset = (page - 1) * page_size
        
        query = f"""
        SELECT 
            fill_id,
            trade_id,
            order_id,
            symbol,
            side,
            quantity,
            price,
            commission,
            commission_asset,
            timestamp
        FROM fills
        WHERE timestamp >= %(start_time)s 
          AND timestamp <= %(end_time)s
          {symbol_filter}
        ORDER BY timestamp DESC
        LIMIT %(page_size)s OFFSET %(offset)s
        """
        
        return await self.query_to_dataframe(query, {
            'start_time': start_time,
            'end_time': end_time,
            'page_size': page_size,
            'offset': offset
        })
    
    # 风险指标查询
    async def get_risk_metrics(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """获取风险指标"""
        query = """
        SELECT 
            timestamp,
            total_exposure,
            var_1d,
            var_5d,
            max_drawdown,
            sharpe_ratio,
            volatility,
            daily_pnl,
            risk_score,
            current_positions
        FROM risk_metrics
        WHERE timestamp >= %(start_time)s 
          AND timestamp <= %(end_time)s
        ORDER BY timestamp DESC
        """
        
        return await self.query_to_dataframe(query, {
            'start_time': start_time,
            'end_time': end_time
        })
    
    # 告警数据查询
    async def get_alerts(
        self,
        start_time: datetime,
        end_time: datetime,
        severity: Optional[str] = None,
        resolved: Optional[bool] = None,
        page: int = 1,
        page_size: int = 100
    ) -> pd.DataFrame:
        """获取告警记录"""
        
        filters = []
        params = {
            'start_time': start_time,
            'end_time': end_time,
            'page_size': page_size,
            'offset': (page - 1) * page_size
        }
        
        if severity:
            filters.append("AND severity = %(severity)s")
            params['severity'] = severity
            
        if resolved is not None:
            filters.append("AND auto_resolved = %(resolved)s")
            params['resolved'] = resolved
        
        filter_clause = " ".join(filters)
        
        query = f"""
        SELECT 
            alert_id,
            timestamp,
            severity,
            code,
            title,
            message,
            source,
            symbol,
            auto_resolved,
            resolution_action
        FROM alerts
        WHERE timestamp >= %(start_time)s 
          AND timestamp <= %(end_time)s
          {filter_clause}
        ORDER BY timestamp DESC
        LIMIT %(page_size)s OFFSET %(offset)s
        """
        
        return await self.query_to_dataframe(query, params)
    
    # 性能分析查询
    async def get_performance_metrics(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> Dict[str, Any]:
        """获取性能指标"""
        
        # 基础统计
        stats_query = """
        SELECT 
            count(*) as total_trades,
            sum(CASE WHEN total_realized_pnl > 0 THEN 1 ELSE 0 END) as profitable_trades,
            sum(total_realized_pnl) as total_realized_pnl,
            sum(total_commission) as total_commission,
            avg(total_realized_pnl) as avg_pnl_per_trade,
            max(total_realized_pnl) as max_profit,
            min(total_realized_pnl) as max_loss
        FROM exec_reports
        WHERE timestamp >= %(start_time)s 
          AND timestamp <= %(end_time)s
        """
        
        stats_df = await self.query_to_dataframe(stats_query, {
            'start_time': start_time,
            'end_time': end_time
        })
        
        if stats_df.empty:
            return {}
        
        stats = stats_df.iloc[0].to_dict()
        
        # 计算胜率
        win_rate = (stats['profitable_trades'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
        
        # 获取风险指标
        risk_query = """
        SELECT 
            max(max_drawdown) as max_drawdown,
            avg(sharpe_ratio) as avg_sharpe_ratio,
            avg(volatility) as avg_volatility
        FROM risk_metrics
        WHERE timestamp >= %(start_time)s 
          AND timestamp <= %(end_time)s
        """
        
        risk_df = await self.query_to_dataframe(risk_query, {
            'start_time': start_time,
            'end_time': end_time
        })
        
        risk_stats = risk_df.iloc[0].to_dict() if not risk_df.empty else {}
        
        return {
            **stats,
            'win_rate': win_rate,
            **risk_stats
        }
    
    # 健康检查
    async def health_check(self) -> Dict[str, Any]:
        """数据库健康检查"""
        try:
            if not self.is_connected:
                await self.connect()
            
            # 检查连接
            start_time = datetime.utcnow()
            await self.client.ping()
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # 检查表状态
            tables_query = "SHOW TABLES"
            tables_result = await self.execute(tables_query)
            table_count = len(tables_result.result_rows)
            
            # 检查最近数据
            recent_data_query = """
            SELECT 
                'exec_reports' as table_name,
                count(*) as row_count,
                max(timestamp) as last_update
            FROM exec_reports
            WHERE timestamp >= now() - INTERVAL 1 DAY
            UNION ALL
            SELECT 
                'risk_metrics' as table_name,
                count(*) as row_count,
                max(timestamp) as last_update
            FROM risk_metrics
            WHERE timestamp >= now() - INTERVAL 1 DAY
            """
            
            recent_data_df = await self.query_to_dataframe(recent_data_query)
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'table_count': table_count,
                'recent_data': recent_data_df.to_dict('records') if not recent_data_df.empty else []
            }
            
        except Exception as e:
            logger.error(f"数据库健康检查失败: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }