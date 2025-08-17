"""
ClickHouse数据库模式定义
======================

定义时序数据库的表结构和索引。
"""

from typing import Dict


class DatabaseSchema:
    """数据库模式定义"""
    
    def __init__(self):
        self.tables = {
            'exec_reports': self._exec_reports_schema(),
            'fills': self._fills_schema(), 
            'orders': self._orders_schema(),
            'positions': self._positions_schema(),
            'risk_metrics': self._risk_metrics_schema(),
            'alerts': self._alerts_schema(),
            'system_health': self._system_health_schema(),
            'pnl_curve': self._pnl_curve_schema()
        }
    
    def get_all_create_statements(self) -> Dict[str, str]:
        """获取所有表的创建语句"""
        return self.tables
    
    def _exec_reports_schema(self) -> str:
        """执行报告表"""
        return """
        CREATE TABLE IF NOT EXISTS exec_reports (
            event_id String,
            timestamp DateTime64(3, 'UTC'),
            strategy_id String DEFAULT 'dipmaster',
            account_id String,
            
            -- PnL信息
            total_realized_pnl Decimal128(8),
            total_unrealized_pnl Decimal128(8), 
            total_commission Decimal128(8),
            total_cost Decimal128(8),
            slippage Decimal128(8),
            market_impact Decimal128(8),
            
            -- 执行统计
            execution_time_ms UInt32,
            venue String DEFAULT 'binance',
            
            -- 关联数据
            order_count UInt16,
            fill_count UInt16,
            
            -- 元数据
            metadata String DEFAULT '{}',
            
            -- 创建时间
            created_at DateTime64(3, 'UTC') DEFAULT now64(3, 'UTC')
        ) 
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, strategy_id, account_id)
        TTL timestamp + INTERVAL 1 YEAR
        SETTINGS index_granularity = 8192
        """
    
    def _fills_schema(self) -> str:
        """成交记录表"""
        return """
        CREATE TABLE IF NOT EXISTS fills (
            fill_id String,
            trade_id String,
            order_id String,
            symbol String,
            side Enum8('BUY' = 1, 'SELL' = 2),
            quantity Decimal128(8),
            price Decimal128(8),
            commission Decimal128(8),
            commission_asset String,
            timestamp DateTime64(3, 'UTC'),
            
            -- 扩展信息
            strategy_id String DEFAULT 'dipmaster',
            account_id String DEFAULT '',
            venue String DEFAULT 'binance',
            
            -- 创建时间
            created_at DateTime64(3, 'UTC') DEFAULT now64(3, 'UTC')
        )
        ENGINE = MergeTree()
        PARTITION BY (toYYYYMM(timestamp), symbol)
        ORDER BY (timestamp, symbol, fill_id)
        TTL timestamp + INTERVAL 1 YEAR
        SETTINGS index_granularity = 8192
        """
    
    def _orders_schema(self) -> str:
        """订单记录表"""
        return """
        CREATE TABLE IF NOT EXISTS orders (
            order_id String,
            client_order_id String,
            symbol String,
            side Enum8('BUY' = 1, 'SELL' = 2),
            type String,
            quantity Decimal128(8),
            price Nullable(Decimal128(8)),
            stop_price Nullable(Decimal128(8)),
            status String,
            filled_quantity Decimal128(8),
            remaining_quantity Decimal128(8),
            create_time DateTime64(3, 'UTC'),
            update_time DateTime64(3, 'UTC'),
            
            -- 扩展信息
            strategy_id String DEFAULT 'dipmaster',
            account_id String DEFAULT '',
            venue String DEFAULT 'binance',
            
            -- 创建时间
            created_at DateTime64(3, 'UTC') DEFAULT now64(3, 'UTC')
        )
        ENGINE = ReplacingMergeTree(update_time)
        PARTITION BY (toYYYYMM(create_time), symbol)
        ORDER BY (create_time, symbol, order_id)
        TTL create_time + INTERVAL 1 YEAR
        SETTINGS index_granularity = 8192
        """
    
    def _positions_schema(self) -> str:
        """持仓快照表"""
        return """
        CREATE TABLE IF NOT EXISTS positions (
            symbol String,
            quantity Decimal128(8),
            avg_price Decimal128(8),
            market_value Decimal128(8),
            unrealized_pnl Decimal128(8),
            realized_pnl Decimal128(8),
            side Enum8('LONG' = 1, 'SHORT' = 2, 'FLAT' = 3),
            timestamp DateTime64(3, 'UTC'),
            
            -- 扩展信息
            strategy_id String DEFAULT 'dipmaster',
            account_id String DEFAULT '',
            venue String DEFAULT 'binance',
            
            -- 创建时间
            created_at DateTime64(3, 'UTC') DEFAULT now64(3, 'UTC')
        )
        ENGINE = ReplacingMergeTree(timestamp)
        PARTITION BY (toYYYYMM(timestamp), symbol)
        ORDER BY (timestamp, symbol, strategy_id)
        TTL timestamp + INTERVAL 6 MONTH
        SETTINGS index_granularity = 8192
        """
    
    def _risk_metrics_schema(self) -> str:
        """风险指标表"""
        return """
        CREATE TABLE IF NOT EXISTS risk_metrics (
            event_id String,
            timestamp DateTime64(3, 'UTC'),
            strategy_id String DEFAULT 'dipmaster',
            account_id String,
            
            -- 敞口风险
            total_exposure Decimal128(8),
            max_position_size Decimal128(8),
            
            -- VaR指标
            var_1d Decimal128(8),
            var_5d Decimal128(8),
            
            -- 绩效指标
            max_drawdown Decimal128(8),
            sharpe_ratio Decimal128(4),
            volatility Decimal128(4),
            
            -- 当日指标
            daily_pnl Decimal128(8),
            daily_loss_limit Decimal128(8),
            
            -- 持仓统计
            max_positions UInt16,
            current_positions UInt16,
            
            -- 风险评分
            risk_score Decimal32(2),
            
            -- JSON字段
            position_concentration String DEFAULT '{}',
            risk_metrics_detail String DEFAULT '{}',
            correlation_matrix String DEFAULT '{}',
            metadata String DEFAULT '{}',
            
            -- 创建时间
            created_at DateTime64(3, 'UTC') DEFAULT now64(3, 'UTC')
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, strategy_id, account_id)
        TTL timestamp + INTERVAL 6 MONTH
        SETTINGS index_granularity = 8192
        """
    
    def _alerts_schema(self) -> str:
        """告警记录表"""
        return """
        CREATE TABLE IF NOT EXISTS alerts (
            event_id String,
            alert_id String,
            timestamp DateTime64(3, 'UTC'),
            
            -- 告警基本信息
            severity Enum8('LOW' = 1, 'MEDIUM' = 2, 'HIGH' = 3, 'CRITICAL' = 4),
            code String,
            title String,
            message String,
            
            -- 来源信息
            source String,
            strategy_id String DEFAULT 'dipmaster',
            account_id String DEFAULT '',
            symbol String DEFAULT '',
            
            -- 处理信息
            auto_resolved Bool DEFAULT false,
            resolution_action String DEFAULT '',
            
            -- JSON字段
            context String DEFAULT '{}',
            metrics String DEFAULT '{}',
            metadata String DEFAULT '{}',
            
            -- 创建时间
            created_at DateTime64(3, 'UTC') DEFAULT now64(3, 'UTC')
        )
        ENGINE = MergeTree()
        PARTITION BY (toYYYYMM(timestamp), severity)
        ORDER BY (timestamp, severity, source)
        TTL timestamp + INTERVAL 3 MONTH
        SETTINGS index_granularity = 8192
        """
    
    def _system_health_schema(self) -> str:
        """系统健康状态表"""
        return """
        CREATE TABLE IF NOT EXISTS system_health (
            event_id String,
            timestamp DateTime64(3, 'UTC'),
            
            -- 整体状态
            overall_status Enum8('HEALTHY' = 1, 'DEGRADED' = 2, 'UNHEALTHY' = 3),
            strategy_id String DEFAULT 'dipmaster',
            
            -- 系统指标
            total_cpu_usage Decimal32(2),
            total_memory_usage Decimal32(2),
            active_connections UInt32,
            
            -- 交易指标
            total_positions UInt16,
            daily_trades UInt32,
            daily_pnl Decimal128(8),
            
            -- 延迟指标
            market_data_delay_ms UInt32,
            execution_delay_ms UInt32,
            
            -- JSON字段
            components String DEFAULT '{}',
            websocket_connections String DEFAULT '{}',
            metadata String DEFAULT '{}',
            
            -- 创建时间
            created_at DateTime64(3, 'UTC') DEFAULT now64(3, 'UTC')
        )
        ENGINE = MergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (timestamp, strategy_id)
        TTL timestamp + INTERVAL 1 MONTH
        SETTINGS index_granularity = 8192
        """
    
    def _pnl_curve_schema(self) -> str:
        """PnL曲线表 - 预聚合的时间序列数据"""
        return """
        CREATE TABLE IF NOT EXISTS pnl_curve (
            timestamp DateTime64(3, 'UTC'),
            time_bucket DateTime64(3, 'UTC'),
            interval_type Enum8('1m' = 1, '5m' = 2, '15m' = 3, '1h' = 4, '1d' = 5),
            
            -- 分组维度
            strategy_id String DEFAULT 'dipmaster',
            account_id String DEFAULT '',
            symbol String DEFAULT 'ALL',
            
            -- PnL指标
            realized_pnl Decimal128(8),
            unrealized_pnl Decimal128(8),
            total_pnl Decimal128(8),
            cumulative_pnl Decimal128(8),
            
            -- 统计指标
            trade_count UInt32,
            commission Decimal128(8),
            volume Decimal128(8),
            
            -- 风险指标
            drawdown Decimal128(8),
            var_1d Decimal128(8),
            
            -- 创建时间
            created_at DateTime64(3, 'UTC') DEFAULT now64(3, 'UTC')
        )
        ENGINE = ReplacingMergeTree(created_at)
        PARTITION BY (toYYYYMM(time_bucket), interval_type)
        ORDER BY (time_bucket, interval_type, strategy_id, symbol)
        TTL time_bucket + INTERVAL 1 YEAR
        SETTINGS index_granularity = 8192
        """
    
    def get_materialized_views(self) -> Dict[str, str]:
        """获取物化视图定义"""
        return {
            'pnl_curve_1h_mv': self._pnl_curve_hourly_mv(),
            'pnl_curve_1d_mv': self._pnl_curve_daily_mv(),
            'risk_summary_mv': self._risk_summary_mv()
        }
    
    def _pnl_curve_hourly_mv(self) -> str:
        """小时级PnL曲线物化视图"""
        return """
        CREATE MATERIALIZED VIEW IF NOT EXISTS pnl_curve_1h_mv
        TO pnl_curve
        AS SELECT
            timestamp,
            toStartOfHour(timestamp) as time_bucket,
            '1h' as interval_type,
            strategy_id,
            account_id,
            'ALL' as symbol,
            sum(total_realized_pnl) as realized_pnl,
            sum(total_unrealized_pnl) as unrealized_pnl,
            sum(total_realized_pnl + total_unrealized_pnl) as total_pnl,
            sum(sum(total_realized_pnl + total_unrealized_pnl)) OVER (
                PARTITION BY strategy_id, account_id 
                ORDER BY toStartOfHour(timestamp)
                ROWS UNBOUNDED PRECEDING
            ) as cumulative_pnl,
            sum(fill_count) as trade_count,
            sum(total_commission) as commission,
            0 as volume,
            0 as drawdown,
            0 as var_1d,
            now64(3, 'UTC') as created_at
        FROM exec_reports
        GROUP BY 
            toStartOfHour(timestamp),
            strategy_id,
            account_id,
            timestamp
        """
    
    def _pnl_curve_daily_mv(self) -> str:
        """日级PnL曲线物化视图"""
        return """
        CREATE MATERIALIZED VIEW IF NOT EXISTS pnl_curve_1d_mv  
        TO pnl_curve
        AS SELECT
            timestamp,
            toStartOfDay(timestamp) as time_bucket,
            '1d' as interval_type,
            strategy_id,
            account_id,
            'ALL' as symbol,
            sum(total_realized_pnl) as realized_pnl,
            sum(total_unrealized_pnl) as unrealized_pnl, 
            sum(total_realized_pnl + total_unrealized_pnl) as total_pnl,
            sum(sum(total_realized_pnl + total_unrealized_pnl)) OVER (
                PARTITION BY strategy_id, account_id
                ORDER BY toStartOfDay(timestamp)
                ROWS UNBOUNDED PRECEDING
            ) as cumulative_pnl,
            sum(fill_count) as trade_count,
            sum(total_commission) as commission,
            0 as volume,
            0 as drawdown,
            0 as var_1d,
            now64(3, 'UTC') as created_at
        FROM exec_reports
        GROUP BY
            toStartOfDay(timestamp),
            strategy_id, 
            account_id,
            timestamp
        """
    
    def _risk_summary_mv(self) -> str:
        """风险汇总物化视图"""
        return """
        CREATE MATERIALIZED VIEW IF NOT EXISTS risk_summary_mv
        ENGINE = AggregatingMergeTree()
        PARTITION BY toYYYYMM(timestamp)
        ORDER BY (toStartOfHour(timestamp), strategy_id)
        AS SELECT
            toStartOfHour(timestamp) as hour_bucket,
            strategy_id,
            account_id,
            maxState(risk_score) as max_risk_score,
            avgState(total_exposure) as avg_exposure,
            maxState(current_positions) as max_positions,
            minState(daily_pnl) as min_daily_pnl,
            maxState(daily_pnl) as max_daily_pnl
        FROM risk_metrics
        GROUP BY
            toStartOfHour(timestamp),
            strategy_id,
            account_id
        """