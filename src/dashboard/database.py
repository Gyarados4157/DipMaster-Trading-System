"""
ClickHouse时序数据库管理器
高性能OLAP查询，数据分片和压缩策略，自动数据清理和归档
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from decimal import Decimal
import asyncio
from contextlib import asynccontextmanager
import structlog
from clickhouse_driver import Client
from clickhouse_driver.errors import Error as ClickHouseError
import pandas as pd

from .config import DatabaseConfig
from .schemas import (
    ExecReportEvent, RiskMetricsEvent, AlertEvent, 
    StrategyPerformanceEvent, PnLDataPoint, PositionSnapshot,
    FillRecord, RiskSnapshot, PerformanceSnapshot
)

logger = structlog.get_logger(__name__)

class ClickHouseConnection:
    """ClickHouse连接管理器"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self._client = None
        self._connection_pool = []
        self._pool_lock = asyncio.Lock()
    
    async def get_client(self) -> Client:
        """获取数据库客户端"""
        if not self._client:
            self._client = Client(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                secure=self.config.secure,
                verify=self.config.verify,
                settings={
                    'use_numpy': True,
                    'max_threads': 4,
                    'max_memory_usage': 2 * 1024 * 1024 * 1024  # 2GB
                }
            )
        return self._client
    
    async def execute(self, query: str, params: Optional[List] = None) -> Any:
        """执行查询"""
        client = await self.get_client()
        try:
            if params:
                return client.execute(query, params)
            else:
                return client.execute(query)
        except ClickHouseError as e:
            logger.error(f"ClickHouse查询失败: {e}")
            raise
    
    async def execute_async(self, query: str, params: Optional[List] = None) -> Any:
        """异步执行查询"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.execute, query, params
        )
    
    async def close(self):
        """关闭连接"""
        if self._client:
            self._client.disconnect()
            self._client = None

class DatabaseManager:
    """数据库管理器主类"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection = ClickHouseConnection(config)
        self._initialized = False
    
    async def initialize(self):
        """初始化数据库"""
        if self._initialized:
            return
        
        try:
            # 创建数据库
            await self._create_database()
            
            # 创建表结构
            await self._create_tables()
            
            # 创建物化视图
            await self._create_materialized_views()
            
            # 创建索引
            await self._create_indexes()
            
            self._initialized = True
            logger.info("ClickHouse数据库初始化完成")
            
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
            raise
    
    async def _create_database(self):
        """创建数据库"""
        query = f"CREATE DATABASE IF NOT EXISTS {self.config.database}"
        await self.connection.execute(query)
        logger.info(f"创建数据库: {self.config.database}")
    
    async def _create_tables(self):
        """创建所有表"""
        
        # 执行报告表
        exec_reports_ddl = """
        CREATE TABLE IF NOT EXISTS exec_reports (
            event_id String,
            timestamp DateTime64(3, 'UTC'),
            order_id String,
            client_order_id Nullable(String),
            symbol String,
            side Enum8('BUY' = 1, 'SELL' = 2),
            order_type Enum8('MARKET' = 1, 'LIMIT' = 2, 'STOP' = 3, 'STOP_LIMIT' = 4),
            order_status Enum8('NEW' = 1, 'PARTIALLY_FILLED' = 2, 'FILLED' = 3, 'CANCELED' = 4, 'REJECTED' = 5, 'EXPIRED' = 6),
            exec_type Enum8('NEW' = 1, 'PARTIAL_FILL' = 2, 'FILL' = 3, 'TRADE' = 4, 'CANCELED' = 5, 'REJECTED' = 6, 'EXPIRED' = 7),
            exec_id Nullable(String),
            quantity Decimal64(8),
            price Decimal64(8),
            last_qty Nullable(Decimal64(8)),
            last_price Nullable(Decimal64(8)),
            cum_qty Nullable(Decimal64(8)),
            avg_price Nullable(Decimal64(8)),
            commission Nullable(Decimal64(8)),
            commission_asset Nullable(String),
            realized_pnl Nullable(Decimal64(8)),
            unrealized_pnl Nullable(Decimal64(8)),
            position_qty Nullable(Decimal64(8)),
            position_side Nullable(String),
            order_time Nullable(DateTime64(3, 'UTC')),
            trade_time Nullable(DateTime64(3, 'UTC')),
            strategy_id Nullable(String),
            account_id String,
            venue Nullable(String),
            source String,
            version String
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, symbol, order_id)
        TTL timestamp + INTERVAL 1 YEAR
        SETTINGS index_granularity = 8192
        """
        
        # 风险指标表
        risk_metrics_ddl = """
        CREATE TABLE IF NOT EXISTS risk_metrics (
            event_id String,
            timestamp DateTime64(3, 'UTC'),
            account_id String,
            portfolio_value Decimal64(8),
            cash_balance Decimal64(8),
            margin_balance Nullable(Decimal64(8)),
            total_pnl Decimal64(8),
            daily_pnl Decimal64(8),
            realized_pnl Decimal64(8),
            unrealized_pnl Decimal64(8),
            var_1d Nullable(Decimal64(8)),
            var_5d Nullable(Decimal64(8)),
            max_drawdown Nullable(Decimal64(8)),
            leverage Nullable(Decimal64(8)),
            risk_score Nullable(Decimal64(8)),
            num_positions UInt32,
            position_limit Nullable(Decimal64(8)),
            loss_limit Nullable(Decimal64(8)),
            market_exposure Nullable(Decimal64(8)),
            source String,
            version String
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, account_id)
        TTL timestamp + INTERVAL 6 MONTH
        SETTINGS index_granularity = 8192
        """
        
        # PnL曲线表
        pnl_curve_ddl = """
        CREATE TABLE IF NOT EXISTS pnl_curve (
            timestamp DateTime64(3, 'UTC'),
            symbol String,
            account_id String,
            strategy_id Nullable(String),
            realized_pnl Decimal64(8),
            unrealized_pnl Decimal64(8),
            total_pnl Decimal64(8),
            cumulative_pnl Decimal64(8),
            trade_count UInt32,
            volume Decimal64(8)
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, symbol, account_id)
        TTL timestamp + INTERVAL 2 YEAR
        SETTINGS index_granularity = 8192
        """
        
        # 告警表
        alerts_ddl = """
        CREATE TABLE IF NOT EXISTS alerts (
            event_id String,
            timestamp DateTime64(3, 'UTC'),
            alert_id String,
            alert_type Enum8('RISK_LIMIT' = 1, 'POSITION_LIMIT' = 2, 'LOSS_LIMIT' = 3, 'SYSTEM_ERROR' = 4, 'CONNECTIVITY' = 5, 'MARKET_DATA' = 6, 'EXECUTION' = 7, 'STRATEGY' = 8),
            severity Enum8('INFO' = 1, 'WARNING' = 2, 'ERROR' = 3, 'CRITICAL' = 4),
            title String,
            message String,
            description Nullable(String),
            account_id Nullable(String),
            strategy_id Nullable(String),
            symbol Nullable(String),
            order_id Nullable(String),
            context Nullable(String),
            status String,
            acknowledged UInt8,
            resolved UInt8,
            threshold_value Nullable(Decimal64(8)),
            current_value Nullable(Decimal64(8)),
            source String,
            version String
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, severity, alert_type)
        TTL timestamp + INTERVAL 3 MONTH
        SETTINGS index_granularity = 8192
        """
        
        # 策略性能表
        strategy_performance_ddl = """
        CREATE TABLE IF NOT EXISTS strategy_performance (
            event_id String,
            timestamp DateTime64(3, 'UTC'),
            strategy_id String,
            strategy_name String,
            account_id String,
            total_trades UInt32,
            winning_trades UInt32,
            losing_trades UInt32,
            win_rate Decimal64(4),
            total_pnl Decimal64(8),
            gross_profit Decimal64(8),
            gross_loss Decimal64(8),
            profit_factor Decimal64(4),
            sharpe_ratio Nullable(Decimal64(4)),
            sortino_ratio Nullable(Decimal64(4)),
            calmar_ratio Nullable(Decimal64(4)),
            max_drawdown Decimal64(8),
            max_drawdown_duration Nullable(UInt32),
            avg_win Decimal64(8),
            avg_loss Decimal64(8),
            largest_win Decimal64(8),
            largest_loss Decimal64(8),
            avg_hold_time Nullable(Decimal64(4)),
            avg_win_time Nullable(Decimal64(4)),
            avg_loss_time Nullable(Decimal64(4)),
            max_capital_used Decimal64(8),
            avg_capital_used Decimal64(8),
            capital_efficiency Decimal64(4),
            source String,
            version String
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, strategy_id, account_id)
        TTL timestamp + INTERVAL 1 YEAR
        SETTINGS index_granularity = 8192
        """
        
        # 仓位详情表
        positions_ddl = """
        CREATE TABLE IF NOT EXISTS positions (
            timestamp DateTime64(3, 'UTC'),
            account_id String,
            symbol String,
            quantity Decimal64(8),
            avg_price Decimal64(8),
            market_value Decimal64(8),
            unrealized_pnl Decimal64(8),
            position_side String
        ) ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, account_id, symbol)
        TTL timestamp + INTERVAL 6 MONTH
        """
        
        tables = [
            ("exec_reports", exec_reports_ddl),
            ("risk_metrics", risk_metrics_ddl),
            ("pnl_curve", pnl_curve_ddl),
            ("alerts", alerts_ddl),
            ("strategy_performance", strategy_performance_ddl),
            ("positions", positions_ddl)
        ]
        
        for table_name, ddl in tables:
            await self.connection.execute(ddl)
            logger.info(f"创建表: {table_name}")
    
    async def _create_materialized_views(self):
        """创建物化视图用于数据聚合"""
        
        # 每日PnL聚合视图
        daily_pnl_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS daily_pnl_mv
        ENGINE = SummingMergeTree()
        PARTITION BY toYYYYMM(date)
        ORDER BY (date, symbol, account_id)
        AS SELECT
            toDate(timestamp) as date,
            symbol,
            account_id,
            sum(realized_pnl) as daily_realized_pnl,
            sum(volume) as daily_volume,
            count() as daily_trades
        FROM pnl_curve
        GROUP BY date, symbol, account_id
        """
        
        # 小时级风险指标聚合视图
        hourly_risk_view = """
        CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_risk_mv
        ENGINE = ReplacingMergeTree()
        PARTITION BY toYYYYMM(hour)
        ORDER BY (hour, account_id)
        AS SELECT
            toStartOfHour(timestamp) as hour,
            account_id,
            argMax(portfolio_value, timestamp) as portfolio_value,
            argMax(total_pnl, timestamp) as total_pnl,
            argMax(var_1d, timestamp) as var_1d,
            argMax(max_drawdown, timestamp) as max_drawdown,
            argMax(leverage, timestamp) as leverage,
            argMax(risk_score, timestamp) as risk_score
        FROM risk_metrics
        GROUP BY hour, account_id
        """
        
        views = [
            ("daily_pnl_mv", daily_pnl_view),
            ("hourly_risk_mv", hourly_risk_view)
        ]
        
        for view_name, view_ddl in views:
            try:
                await self.connection.execute(view_ddl)
                logger.info(f"创建物化视图: {view_name}")
            except Exception as e:
                logger.warning(f"创建物化视图失败 {view_name}: {e}")
    
    async def _create_indexes(self):
        """创建额外的索引优化查询性能"""
        # ClickHouse主要依赖排序键，可以创建跳数索引
        
        indexes = [
            # 在exec_reports表上创建symbol索引
            "ALTER TABLE exec_reports ADD INDEX IF NOT EXISTS idx_symbol symbol TYPE bloom_filter GRANULARITY 1",
            # 在alerts表上创建严重性索引
            "ALTER TABLE alerts ADD INDEX IF NOT EXISTS idx_severity severity TYPE set(4) GRANULARITY 1",
            # 在risk_metrics表上创建账户索引
            "ALTER TABLE risk_metrics ADD INDEX IF NOT EXISTS idx_account account_id TYPE bloom_filter GRANULARITY 1"
        ]
        
        for index_ddl in indexes:
            try:
                await self.connection.execute(index_ddl)
            except Exception as e:
                logger.warning(f"创建索引失败: {e}")
    
    # 数据插入方法
    
    async def insert_exec_reports(self, events: List[ExecReportEvent]):
        """批量插入执行报告"""
        if not events:
            return
        
        data = []
        for event in events:
            data.append([
                event.event_id,
                event.timestamp,
                event.order_id,
                event.client_order_id,
                event.symbol,
                event.side.value,
                event.order_type.value,
                event.order_status.value,
                event.exec_type.value,
                event.exec_id,
                float(event.quantity),
                float(event.price),
                float(event.last_qty) if event.last_qty else None,
                float(event.last_price) if event.last_price else None,
                float(event.cum_qty) if event.cum_qty else None,
                float(event.avg_price) if event.avg_price else None,
                float(event.commission) if event.commission else None,
                event.commission_asset,
                float(event.realized_pnl) if event.realized_pnl else None,
                float(event.unrealized_pnl) if event.unrealized_pnl else None,
                float(event.position_qty) if event.position_qty else None,
                event.position_side,
                event.order_time,
                event.trade_time,
                event.strategy_id,
                event.account_id,
                event.venue,
                event.source,
                event.version
            ])
        
        query = """
        INSERT INTO exec_reports VALUES
        """
        
        await self.connection.execute(query, data)
        logger.debug(f"插入执行报告: {len(events)}条")
    
    async def insert_risk_metrics(self, events: List[RiskMetricsEvent]):
        """批量插入风险指标"""
        if not events:
            return
        
        data = []
        for event in events:
            data.append([
                event.event_id,
                event.timestamp,
                event.account_id,
                float(event.portfolio_value),
                float(event.cash_balance),
                float(event.margin_balance) if event.margin_balance else None,
                float(event.total_pnl),
                float(event.daily_pnl),
                float(event.realized_pnl),
                float(event.unrealized_pnl),
                float(event.var_1d) if event.var_1d else None,
                float(event.var_5d) if event.var_5d else None,
                float(event.max_drawdown) if event.max_drawdown else None,
                float(event.leverage) if event.leverage else None,
                float(event.risk_score) if event.risk_score else None,
                event.num_positions,
                float(event.position_limit) if event.position_limit else None,
                float(event.loss_limit) if event.loss_limit else None,
                float(event.market_exposure) if event.market_exposure else None,
                event.source,
                event.version
            ])
        
        query = "INSERT INTO risk_metrics VALUES"
        await self.connection.execute(query, data)
        logger.debug(f"插入风险指标: {len(events)}条")
    
    async def insert_alerts(self, events: List[AlertEvent]):
        """批量插入告警"""
        if not events:
            return
        
        data = []
        for event in events:
            data.append([
                event.event_id,
                event.timestamp,
                event.alert_id,
                event.alert_type.value,
                event.severity.value,
                event.title,
                event.message,
                event.description,
                event.account_id,
                event.strategy_id,
                event.symbol,
                event.order_id,
                json.dumps(event.context) if event.context else None,
                event.status,
                1 if event.acknowledged else 0,
                1 if event.resolved else 0,
                float(event.threshold_value) if event.threshold_value else None,
                float(event.current_value) if event.current_value else None,
                event.source,
                event.version
            ])
        
        query = "INSERT INTO alerts VALUES"
        await self.connection.execute(query, data)
        logger.debug(f"插入告警: {len(events)}条")
    
    async def insert_strategy_performance(self, events: List[StrategyPerformanceEvent]):
        """批量插入策略性能"""
        if not events:
            return
        
        data = []
        for event in events:
            data.append([
                event.event_id,
                event.timestamp,
                event.strategy_id,
                event.strategy_name,
                event.account_id,
                event.total_trades,
                event.winning_trades,
                event.losing_trades,
                float(event.win_rate),
                float(event.total_pnl),
                float(event.gross_profit),
                float(event.gross_loss),
                float(event.profit_factor),
                float(event.sharpe_ratio) if event.sharpe_ratio else None,
                float(event.sortino_ratio) if event.sortino_ratio else None,
                float(event.calmar_ratio) if event.calmar_ratio else None,
                float(event.max_drawdown),
                event.max_drawdown_duration,
                float(event.avg_win),
                float(event.avg_loss),
                float(event.largest_win),
                float(event.largest_loss),
                float(event.avg_hold_time) if event.avg_hold_time else None,
                float(event.avg_win_time) if event.avg_win_time else None,
                float(event.avg_loss_time) if event.avg_loss_time else None,
                float(event.max_capital_used),
                float(event.avg_capital_used),
                float(event.capital_efficiency),
                event.source,
                event.version
            ])
        
        query = "INSERT INTO strategy_performance VALUES"
        await self.connection.execute(query, data)
        logger.debug(f"插入策略性能: {len(events)}条")
    
    # 查询方法
    
    async def get_pnl_data(self, account_id: str, start_time: datetime, 
                          end_time: datetime, symbols: Optional[List[str]] = None,
                          limit: int = 1000) -> List[PnLDataPoint]:
        """获取PnL数据"""
        where_clauses = [
            f"timestamp >= '{start_time.isoformat()}'",
            f"timestamp <= '{end_time.isoformat()}'"
        ]
        
        if account_id:
            where_clauses.append(f"account_id = '{account_id}'")
        
        if symbols:
            symbol_list = "','".join(symbols)
            where_clauses.append(f"symbol IN ('{symbol_list}')")
        
        query = f"""
        SELECT 
            timestamp,
            symbol,
            sum(realized_pnl) as realized_pnl,
            sum(unrealized_pnl) as unrealized_pnl,
            sum(total_pnl) as total_pnl,
            sum(cumulative_pnl) as cumulative_pnl
        FROM pnl_curve
        WHERE {' AND '.join(where_clauses)}
        GROUP BY timestamp, symbol
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        
        result = await self.connection.execute(query)
        return [
            PnLDataPoint(
                timestamp=row[0],
                symbol=row[1],
                realized_pnl=Decimal(str(row[2])),
                unrealized_pnl=Decimal(str(row[3])),
                total_pnl=Decimal(str(row[4])),
                cumulative_pnl=Decimal(str(row[5]))
            )
            for row in result
        ]
    
    async def get_latest_positions(self, account_id: str) -> List[PositionSnapshot]:
        """获取最新仓位快照"""
        query = f"""
        SELECT 
            timestamp,
            symbol,
            quantity,
            avg_price,
            market_value,
            unrealized_pnl,
            position_side
        FROM positions
        WHERE account_id = '{account_id}'
        AND timestamp = (
            SELECT max(timestamp) 
            FROM positions 
            WHERE account_id = '{account_id}'
        )
        ORDER BY symbol
        """
        
        result = await self.connection.execute(query)
        return [
            PositionSnapshot(
                timestamp=row[0],
                symbol=row[1],
                quantity=Decimal(str(row[2])),
                avg_price=Decimal(str(row[3])),
                market_price=Decimal(str(row[3])),  # 简化处理
                market_value=Decimal(str(row[4])),
                unrealized_pnl=Decimal(str(row[5])),
                position_side=row[6]
            )
            for row in result
        ]
    
    async def get_fills(self, account_id: str, start_time: datetime,
                       end_time: datetime, symbols: Optional[List[str]] = None,
                       limit: int = 500) -> List[FillRecord]:
        """获取成交记录"""
        where_clauses = [
            f"timestamp >= '{start_time.isoformat()}'",
            f"timestamp <= '{end_time.isoformat()}'",
            f"account_id = '{account_id}'",
            "exec_type = 'TRADE'"
        ]
        
        if symbols:
            symbol_list = "','".join(symbols)
            where_clauses.append(f"symbol IN ('{symbol_list}')")
        
        query = f"""
        SELECT 
            timestamp,
            symbol,
            side,
            quantity,
            price,
            commission,
            order_id
        FROM exec_reports
        WHERE {' AND '.join(where_clauses)}
        ORDER BY timestamp DESC
        LIMIT {limit}
        """
        
        result = await self.connection.execute(query)
        return [
            FillRecord(
                timestamp=row[0],
                symbol=row[1],
                side=row[2],
                quantity=Decimal(str(row[3])),
                price=Decimal(str(row[4])),
                commission=Decimal(str(row[5])) if row[5] else Decimal('0'),
                order_id=row[6]
            )
            for row in result
        ]
    
    async def get_risk_metrics(self, account_id: str, start_time: datetime,
                              end_time: datetime) -> List[RiskSnapshot]:
        """获取风险指标"""
        query = f"""
        SELECT 
            timestamp,
            account_id,
            portfolio_value,
            total_pnl,
            var_1d,
            max_drawdown,
            leverage,
            risk_score,
            num_positions
        FROM risk_metrics
        WHERE account_id = '{account_id}'
        AND timestamp >= '{start_time.isoformat()}'
        AND timestamp <= '{end_time.isoformat()}'
        ORDER BY timestamp DESC
        """
        
        result = await self.connection.execute(query)
        return [
            RiskSnapshot(
                timestamp=row[0],
                account_id=row[1],
                portfolio_value=Decimal(str(row[2])),
                total_pnl=Decimal(str(row[3])),
                var_1d=Decimal(str(row[4])) if row[4] else None,
                max_drawdown=Decimal(str(row[5])) if row[5] else None,
                leverage=Decimal(str(row[6])) if row[6] else None,
                risk_score=Decimal(str(row[7])) if row[7] else None,
                num_positions=row[8]
            )
            for row in result
        ]
    
    async def get_strategy_performance(self, strategy_id: str, 
                                     start_time: datetime,
                                     end_time: datetime) -> List[PerformanceSnapshot]:
        """获取策略性能"""
        query = f"""
        SELECT 
            timestamp,
            strategy_id,
            total_trades,
            win_rate,
            total_pnl,
            sharpe_ratio,
            max_drawdown,
            profit_factor
        FROM strategy_performance
        WHERE strategy_id = '{strategy_id}'
        AND timestamp >= '{start_time.isoformat()}'
        AND timestamp <= '{end_time.isoformat()}'
        ORDER BY timestamp DESC
        """
        
        result = await self.connection.execute(query)
        return [
            PerformanceSnapshot(
                timestamp=row[0],
                strategy_id=row[1],
                total_trades=row[2],
                win_rate=Decimal(str(row[3])),
                total_pnl=Decimal(str(row[4])),
                sharpe_ratio=Decimal(str(row[5])) if row[5] else None,
                max_drawdown=Decimal(str(row[6])),
                profit_factor=Decimal(str(row[7]))
            )
            for row in result
        ]
    
    async def cleanup_old_data(self):
        """清理过期数据"""
        # ClickHouse TTL会自动清理，这里可以手动清理一些临时数据
        cleanup_queries = [
            "OPTIMIZE TABLE exec_reports FINAL",
            "OPTIMIZE TABLE risk_metrics FINAL",
            "OPTIMIZE TABLE pnl_curve FINAL",
            "OPTIMIZE TABLE alerts FINAL",
            "OPTIMIZE TABLE strategy_performance FINAL"
        ]
        
        for query in cleanup_queries:
            try:
                await self.connection.execute(query)
            except Exception as e:
                logger.warning(f"数据清理失败: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """获取数据库健康状态"""
        try:
            # 检查连接
            result = await self.connection.execute("SELECT 1")
            
            # 检查表状态
            tables_query = """
            SELECT table, formatReadableSize(sum(bytes_on_disk)) as size,
                   sum(rows) as rows
            FROM system.parts
            WHERE database = %s AND active = 1
            GROUP BY table
            """
            
            tables_result = await self.connection.execute(tables_query, [self.config.database])
            
            return {
                "status": "healthy",
                "connection": "OK",
                "tables": [
                    {
                        "name": row[0],
                        "size": row[1],
                        "rows": row[2]
                    }
                    for row in tables_result
                ]
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def shutdown(self):
        """关闭数据库连接"""
        await self.connection.close()
        logger.info("数据库连接已关闭")