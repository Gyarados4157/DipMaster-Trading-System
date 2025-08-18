#!/usr/bin/env python3
"""
DipMaster Trading System - Automated Reporting System
自动化报告系统 - 生成日度、周度和月度运营报告

Author: DipMaster Trading System Monitoring Agent
Date: 2025-08-18
Version: 1.0.0
"""

import asyncio
import json
import time
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
import statistics
import base64
from io import BytesIO
import calendar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ReportType(Enum):
    """报告类型"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    CUSTOM = "custom"


class ReportSection(Enum):
    """报告章节"""
    EXECUTIVE_SUMMARY = "executive_summary"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    RISK_ANALYSIS = "risk_analysis"
    TRADING_ACTIVITY = "trading_activity"
    SYSTEM_HEALTH = "system_health"
    QUALITY_METRICS = "quality_metrics"
    RECOMMENDATIONS = "recommendations"
    APPENDIX = "appendix"


@dataclass
class ReportMetrics:
    """报告指标数据"""
    period_start: datetime
    period_end: datetime
    
    # 交易指标
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    
    # 性能指标
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_return: float = 0.0
    avg_holding_time: float = 0.0
    
    # 风险指标
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall: float = 0.0
    volatility: float = 0.0
    max_leverage: float = 0.0
    
    # 执行指标
    avg_slippage: float = 0.0
    avg_latency: float = 0.0
    fill_rate: float = 0.0
    execution_quality_score: float = 0.0
    
    # 系统指标
    uptime_percentage: float = 0.0
    error_rate: float = 0.0
    alerts_generated: int = 0
    critical_alerts: int = 0
    
    def calculate_derived_metrics(self) -> None:
        """计算派生指标"""
        if self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100
            self.avg_trade_return = self.realized_pnl / self.total_trades
        
        if self.gross_loss != 0:
            self.profit_factor = abs(self.gross_profit / self.gross_loss)
        
        # 计算夏普比率（简化版）
        if self.volatility > 0 and self.total_trades > 0:
            annualized_return = self.avg_trade_return * 252  # 假设252个交易日
            self.sharpe_ratio = annualized_return / self.volatility


@dataclass
class ReportChart:
    """报告图表数据"""
    chart_type: str
    title: str
    data: List[Dict[str, Any]]
    config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    
    def to_html(self) -> str:
        """转换为HTML图表"""
        # 这里应该使用实际的图表库如Plotly或Chart.js
        # 为了演示，我们返回一个简化的HTML表示
        return f"""
        <div class="chart-container">
            <h3>{self.title}</h3>
            <p>{self.description}</p>
            <div class="chart" id="chart_{id(self)}" data-type="{self.chart_type}">
                <!-- Chart data: {len(self.data)} points -->
            </div>
        </div>
        """


@dataclass
class ReportTable:
    """报告表格数据"""
    title: str
    headers: List[str]
    rows: List[List[Any]]
    description: str = ""
    
    def to_html(self) -> str:
        """转换为HTML表格"""
        html = f"""
        <div class="table-container">
            <h3>{self.title}</h3>
            <p>{self.description}</p>
            <table class="report-table">
                <thead>
                    <tr>{''.join(f'<th>{header}</th>' for header in self.headers)}</tr>
                </thead>
                <tbody>
        """
        
        for row in self.rows:
            html += f"<tr>{''.join(f'<td>{cell}</td>' for cell in row)}</tr>"
        
        html += """
                </tbody>
            </table>
        </div>
        """
        return html


@dataclass
class ReportData:
    """完整报告数据"""
    report_id: str
    report_type: ReportType
    period_start: datetime
    period_end: datetime
    generated_at: datetime
    metrics: ReportMetrics
    charts: List[ReportChart] = field(default_factory=list)
    tables: List[ReportTable] = field(default_factory=list)
    sections: Dict[ReportSection, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCalculator:
    """指标计算器"""
    
    def __init__(self):
        self.trade_history: List[Dict[str, Any]] = []
        self.risk_history: List[Dict[str, Any]] = []
        self.execution_history: List[Dict[str, Any]] = []
        self.system_events: List[Dict[str, Any]] = []
    
    def add_trade_record(self, trade: Dict[str, Any]) -> None:
        """添加交易记录"""
        self.trade_history.append(trade)
    
    def add_risk_record(self, risk_data: Dict[str, Any]) -> None:
        """添加风险记录"""
        self.risk_history.append(risk_data)
    
    def add_execution_record(self, execution: Dict[str, Any]) -> None:
        """添加执行记录"""
        self.execution_history.append(execution)
    
    def add_system_event(self, event: Dict[str, Any]) -> None:
        """添加系统事件"""
        self.system_events.append(event)
    
    def calculate_metrics(self, period_start: datetime, period_end: datetime) -> ReportMetrics:
        """计算指定期间的指标"""
        metrics = ReportMetrics(period_start=period_start, period_end=period_end)
        
        # 筛选期间内的数据
        period_trades = [
            trade for trade in self.trade_history
            if period_start <= trade.get('timestamp', datetime.min.replace(tzinfo=timezone.utc)) <= period_end
        ]
        
        period_risk = [
            risk for risk in self.risk_history
            if period_start <= risk.get('timestamp', datetime.min.replace(tzinfo=timezone.utc)) <= period_end
        ]
        
        period_executions = [
            exec for exec in self.execution_history
            if period_start <= exec.get('timestamp', datetime.min.replace(tzinfo=timezone.utc)) <= period_end
        ]
        
        # 计算交易指标
        metrics.total_trades = len(period_trades)
        
        if period_trades:
            winning_trades = [t for t in period_trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in period_trades if t.get('pnl', 0) <= 0]
            
            metrics.winning_trades = len(winning_trades)
            metrics.losing_trades = len(losing_trades)
            metrics.total_pnl = sum(t.get('pnl', 0) for t in period_trades)
            metrics.realized_pnl = sum(t.get('realized_pnl', t.get('pnl', 0)) for t in period_trades)
            
            metrics.gross_profit = sum(t.get('pnl', 0) for t in winning_trades)
            metrics.gross_loss = sum(t.get('pnl', 0) for t in losing_trades)
            
            # 计算平均持仓时间
            holding_times = [t.get('holding_minutes', 0) for t in period_trades if t.get('holding_minutes')]
            if holding_times:
                metrics.avg_holding_time = statistics.mean(holding_times)
            
            # 计算波动率
            returns = [t.get('return_pct', 0) for t in period_trades if t.get('return_pct')]
            if len(returns) > 1:
                metrics.volatility = statistics.stdev(returns)
        
        # 计算风险指标
        if period_risk:
            latest_risk = period_risk[-1]  # 取最新的风险数据
            metrics.var_95 = latest_risk.get('var_95', 0)
            metrics.var_99 = latest_risk.get('var_99', 0)
            metrics.expected_shortfall = latest_risk.get('expected_shortfall', 0)
            metrics.max_drawdown = latest_risk.get('max_drawdown', 0)
            metrics.max_leverage = latest_risk.get('max_leverage', 0)
        
        # 计算执行指标
        if period_executions:
            slippages = [e.get('slippage_bps', 0) for e in period_executions]
            latencies = [e.get('latency_ms', 0) for e in period_executions]
            
            if slippages:
                metrics.avg_slippage = statistics.mean(slippages)
            if latencies:
                metrics.avg_latency = statistics.mean(latencies)
            
            # 成交率
            filled_orders = sum(1 for e in period_executions if e.get('status') == 'FILLED')
            metrics.fill_rate = (filled_orders / len(period_executions)) * 100
        
        # 计算系统指标
        period_events = [
            event for event in self.system_events
            if period_start <= event.get('timestamp', datetime.min.replace(tzinfo=timezone.utc)) <= period_end
        ]
        
        metrics.alerts_generated = len(period_events)
        metrics.critical_alerts = sum(
            1 for event in period_events
            if event.get('severity') in ['CRITICAL', 'EMERGENCY']
        )
        
        # 计算派生指标
        metrics.calculate_derived_metrics()
        
        return metrics


class ChartGenerator:
    """图表生成器"""
    
    def __init__(self):
        pass
    
    def generate_pnl_chart(self, trade_data: List[Dict[str, Any]]) -> ReportChart:
        """生成PnL图表"""
        # 计算累积PnL
        cumulative_pnl = 0
        chart_data = []
        
        for trade in sorted(trade_data, key=lambda x: x.get('timestamp', datetime.min)):
            cumulative_pnl += trade.get('pnl', 0)
            chart_data.append({
                'timestamp': trade.get('timestamp', datetime.now()).isoformat(),
                'cumulative_pnl': cumulative_pnl,
                'trade_pnl': trade.get('pnl', 0)
            })
        
        return ReportChart(
            chart_type="line",
            title="Cumulative P&L Over Time",
            data=chart_data,
            description="Cumulative profit and loss progression throughout the reporting period"
        )
    
    def generate_drawdown_chart(self, trade_data: List[Dict[str, Any]]) -> ReportChart:
        """生成回撤图表"""
        cumulative_pnl = 0
        peak_pnl = 0
        chart_data = []
        
        for trade in sorted(trade_data, key=lambda x: x.get('timestamp', datetime.min)):
            cumulative_pnl += trade.get('pnl', 0)
            peak_pnl = max(peak_pnl, cumulative_pnl)
            drawdown = (cumulative_pnl - peak_pnl) / max(peak_pnl, 1) * 100
            
            chart_data.append({
                'timestamp': trade.get('timestamp', datetime.now()).isoformat(),
                'drawdown': drawdown,
                'cumulative_pnl': cumulative_pnl,
                'peak_pnl': peak_pnl
            })
        
        return ReportChart(
            chart_type="area",
            title="Drawdown Analysis",
            data=chart_data,
            description="Portfolio drawdown relative to previous peak value"
        )
    
    def generate_performance_distribution(self, trade_data: List[Dict[str, Any]]) -> ReportChart:
        """生成收益分布图"""
        returns = [trade.get('return_pct', 0) for trade in trade_data if trade.get('return_pct') is not None]
        
        # 创建直方图数据
        if returns:
            # 简化的分组
            bins = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
            bin_labels = ['<-10%', '-10% to -5%', '-5% to -2%', '-2% to -1%', '-1% to 0%', 
                         '0% to 1%', '1% to 2%', '2% to 5%', '>5%']
            
            distribution = [0] * len(bin_labels)
            for ret in returns:
                for i, threshold in enumerate(bins[1:], 1):
                    if ret <= threshold:
                        distribution[i-1] += 1
                        break
                else:
                    distribution[-1] += 1
            
            chart_data = [
                {'range': label, 'count': count, 'percentage': count/len(returns)*100}
                for label, count in zip(bin_labels, distribution)
            ]
        else:
            chart_data = []
        
        return ReportChart(
            chart_type="bar",
            title="Trade Return Distribution",
            data=chart_data,
            description="Distribution of individual trade returns"
        )
    
    def generate_risk_metrics_chart(self, risk_data: List[Dict[str, Any]]) -> ReportChart:
        """生成风险指标图表"""
        chart_data = []
        
        for risk_point in sorted(risk_data, key=lambda x: x.get('timestamp', datetime.min)):
            chart_data.append({
                'timestamp': risk_point.get('timestamp', datetime.now()).isoformat(),
                'var_95': risk_point.get('var_95', 0),
                'var_99': risk_point.get('var_99', 0),
                'expected_shortfall': risk_point.get('expected_shortfall', 0),
                'max_drawdown': risk_point.get('max_drawdown', 0) * 100
            })
        
        return ReportChart(
            chart_type="multi_line",
            title="Risk Metrics Evolution",
            data=chart_data,
            description="Evolution of key risk metrics over the reporting period"
        )


class TableGenerator:
    """表格生成器"""
    
    def __init__(self):
        pass
    
    def generate_trade_summary_table(self, trade_data: List[Dict[str, Any]]) -> ReportTable:
        """生成交易汇总表"""
        if not trade_data:
            return ReportTable(
                title="Trade Summary by Symbol",
                headers=["Symbol", "Trades", "Win Rate", "Total PnL", "Avg PnL"],
                rows=[["No trades in period", "", "", "", ""]],
                description="Summary of trading activity grouped by symbol"
            )
        
        # 按币种分组统计
        symbol_stats = {}
        for trade in trade_data:
            symbol = trade.get('symbol', 'UNKNOWN')
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    'trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0,
                    'pnls': []
                }
            
            stats = symbol_stats[symbol]
            stats['trades'] += 1
            pnl = trade.get('pnl', 0)
            stats['total_pnl'] += pnl
            stats['pnls'].append(pnl)
            if pnl > 0:
                stats['winning_trades'] += 1
        
        # 生成表格行
        rows = []
        for symbol, stats in symbol_stats.items():
            win_rate = (stats['winning_trades'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            avg_pnl = statistics.mean(stats['pnls']) if stats['pnls'] else 0
            
            rows.append([
                symbol,
                stats['trades'],
                f"{win_rate:.1f}%",
                f"${stats['total_pnl']:.2f}",
                f"${avg_pnl:.2f}"
            ])
        
        # 按总PnL排序
        rows.sort(key=lambda x: float(x[3].replace('$', '').replace(',', '')), reverse=True)
        
        return ReportTable(
            title="Trade Summary by Symbol",
            headers=["Symbol", "Trades", "Win Rate", "Total PnL", "Avg PnL"],
            rows=rows,
            description="Summary of trading activity grouped by symbol"
        )
    
    def generate_best_worst_trades(self, trade_data: List[Dict[str, Any]]) -> ReportTable:
        """生成最佳/最差交易表"""
        if not trade_data:
            return ReportTable(
                title="Best and Worst Trades",
                headers=["Type", "Symbol", "PnL", "Return %", "Date"],
                rows=[["No trades in period", "", "", "", ""]],
                description="Top performing and worst performing individual trades"
            )
        
        # 排序交易
        sorted_trades = sorted(trade_data, key=lambda x: x.get('pnl', 0))
        
        rows = []
        
        # 最差的5笔交易
        worst_trades = sorted_trades[:5]
        for trade in worst_trades:
            rows.append([
                "Worst",
                trade.get('symbol', ''),
                f"${trade.get('pnl', 0):.2f}",
                f"{trade.get('return_pct', 0):.2f}%",
                trade.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M')
            ])
        
        # 最佳的5笔交易
        best_trades = sorted_trades[-5:]
        best_trades.reverse()
        for trade in best_trades:
            rows.append([
                "Best",
                trade.get('symbol', ''),
                f"${trade.get('pnl', 0):.2f}",
                f"{trade.get('return_pct', 0):.2f}%",
                trade.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M')
            ])
        
        return ReportTable(
            title="Best and Worst Trades",
            headers=["Type", "Symbol", "PnL", "Return %", "Date"],
            rows=rows,
            description="Top performing and worst performing individual trades"
        )
    
    def generate_monthly_performance_table(self, trade_data: List[Dict[str, Any]], period_months: int = 12) -> ReportTable:
        """生成月度性能表"""
        monthly_stats = {}
        
        for trade in trade_data:
            timestamp = trade.get('timestamp', datetime.now())
            month_key = timestamp.strftime('%Y-%m')
            
            if month_key not in monthly_stats:
                monthly_stats[month_key] = {
                    'trades': 0,
                    'winning_trades': 0,
                    'total_pnl': 0,
                    'month_name': timestamp.strftime('%B %Y')
                }
            
            stats = monthly_stats[month_key]
            stats['trades'] += 1
            pnl = trade.get('pnl', 0)
            stats['total_pnl'] += pnl
            if pnl > 0:
                stats['winning_trades'] += 1
        
        # 生成表格行
        rows = []
        for month_key in sorted(monthly_stats.keys())[-period_months:]:
            stats = monthly_stats[month_key]
            win_rate = (stats['winning_trades'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
            
            rows.append([
                stats['month_name'],
                stats['trades'],
                stats['winning_trades'],
                f"{win_rate:.1f}%",
                f"${stats['total_pnl']:.2f}"
            ])
        
        return ReportTable(
            title="Monthly Performance Summary",
            headers=["Month", "Total Trades", "Winning Trades", "Win Rate", "Total PnL"],
            rows=rows,
            description=f"Monthly trading performance over the last {period_months} months"
        )


class ReportRenderer:
    """报告渲染器"""
    
    def __init__(self):
        self.template = self._load_template()
    
    def _load_template(self) -> str:
        """加载HTML报告模板"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DipMaster Trading System - {report_title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; background: #f5f7fa; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #ecf0f1; padding: 15px; border-radius: 6px; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
        .metric-label {{ font-size: 12px; color: #7f8c8d; text-transform: uppercase; }}
        .positive {{ color: #27ae60; }}
        .negative {{ color: #e74c3c; }}
        .chart-container {{ margin: 20px 0; padding: 20px; background: #fafbfc; border-radius: 6px; }}
        .table-container {{ margin: 20px 0; }}
        .report-table {{ width: 100%; border-collapse: collapse; }}
        .report-table th, .report-table td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        .report-table th {{ background-color: #f8f9fa; font-weight: 600; }}
        .report-table tr:hover {{ background-color: #f5f5f5; }}
        .executive-summary {{ background: #e8f4f8; padding: 20px; border-radius: 6px; margin: 20px 0; }}
        .recommendations {{ background: #fff3cd; padding: 20px; border-radius: 6px; margin: 20px 0; }}
        .footer {{ text-align: center; margin-top: 40px; color: #7f8c8d; font-size: 12px; }}
        .status-healthy {{ color: #27ae60; }}
        .status-warning {{ color: #f39c12; }}
        .status-critical {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DipMaster Trading System Report</h1>
        <div class="report-header">
            <h2>{report_title}</h2>
            <p><strong>Period:</strong> {period_start} to {period_end}</p>
            <p><strong>Generated:</strong> {generated_at}</p>
        </div>
        
        {content}
        
        <div class="footer">
            <p>Generated by DipMaster Trading System Monitoring Agent | Version 1.0.0</p>
            <p>This report contains confidential trading information. Do not distribute.</p>
        </div>
    </div>
</body>
</html>
"""
    
    def render_report(self, report_data: ReportData) -> str:
        """渲染完整报告"""
        content_sections = []
        
        # 执行摘要
        if ReportSection.EXECUTIVE_SUMMARY in report_data.sections:
            content_sections.append(f"""
            <div class="executive-summary">
                <h2>📋 Executive Summary</h2>
                {report_data.sections[ReportSection.EXECUTIVE_SUMMARY]}
            </div>
            """)
        
        # 关键指标展示
        content_sections.append(self._render_key_metrics(report_data.metrics))
        
        # 性能分析
        if ReportSection.PERFORMANCE_ANALYSIS in report_data.sections:
            content_sections.append(f"""
            <h2>📈 Performance Analysis</h2>
            {report_data.sections[ReportSection.PERFORMANCE_ANALYSIS]}
            """)
        
        # 图表展示
        for chart in report_data.charts:
            content_sections.append(chart.to_html())
        
        # 表格展示
        for table in report_data.tables:
            content_sections.append(table.to_html())
        
        # 风险分析
        if ReportSection.RISK_ANALYSIS in report_data.sections:
            content_sections.append(f"""
            <h2>⚠️ Risk Analysis</h2>
            {report_data.sections[ReportSection.RISK_ANALYSIS]}
            """)
        
        # 建议
        if ReportSection.RECOMMENDATIONS in report_data.sections:
            content_sections.append(f"""
            <div class="recommendations">
                <h2>💡 Recommendations</h2>
                {report_data.sections[ReportSection.RECOMMENDATIONS]}
            </div>
            """)
        
        # 替换模板变量
        return self.template.format(
            report_title=f"{report_data.report_type.value.title()} Report",
            period_start=report_data.period_start.strftime('%Y-%m-%d %H:%M UTC'),
            period_end=report_data.period_end.strftime('%Y-%m-%d %H:%M UTC'),
            generated_at=report_data.generated_at.strftime('%Y-%m-%d %H:%M UTC'),
            content=''.join(content_sections)
        )
    
    def _render_key_metrics(self, metrics: ReportMetrics) -> str:
        """渲染关键指标"""
        return f"""
        <h2>📊 Key Performance Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.total_pnl > 0 else 'negative'}">${metrics.total_pnl:.2f}</div>
                <div class="metric-label">Total P&L</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.win_rate >= 50 else 'negative'}">{metrics.win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.profit_factor > 1 else 'negative'}">{metrics.profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.sharpe_ratio > 1 else 'negative'}">{metrics.sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.max_drawdown < 0.1 else 'negative'}">{metrics.max_drawdown:.1%}</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics.avg_holding_time:.0f}m</div>
                <div class="metric-label">Avg Holding Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value {'positive' if metrics.avg_slippage < 10 else 'negative'}">{metrics.avg_slippage:.1f} bps</div>
                <div class="metric-label">Avg Slippage</div>
            </div>
        </div>
        """


class AutomatedReportingSystem:
    """自动化报告系统"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.metrics_calculator = MetricsCalculator()
        self.chart_generator = ChartGenerator()
        self.table_generator = TableGenerator()
        self.report_renderer = ReportRenderer()
        
        # 报告存储路径
        self.reports_dir = Path(self.config.get('reports_dir', 'reports/automated'))
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # 调度配置
        self.daily_report_time = self.config.get('daily_report_time', '06:00')  # UTC时间
        self.weekly_report_day = self.config.get('weekly_report_day', 'monday')
        self.monthly_report_day = self.config.get('monthly_report_day', 1)
        
        self.is_running = False
        self._scheduler_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """启动报告系统"""
        if self.is_running:
            return
        
        self.is_running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("📊 Automated Reporting System started")
    
    async def stop(self) -> None:
        """停止报告系统"""
        self.is_running = False
        
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        
        logger.info("📊 Automated Reporting System stopped")
    
    async def _scheduler_loop(self) -> None:
        """调度循环"""
        while self.is_running:
            try:
                current_time = datetime.now(timezone.utc)
                
                # 检查是否需要生成日报
                if await self._should_generate_daily_report(current_time):
                    await self._generate_scheduled_report(ReportType.DAILY)
                
                # 检查是否需要生成周报
                if await self._should_generate_weekly_report(current_time):
                    await self._generate_scheduled_report(ReportType.WEEKLY)
                
                # 检查是否需要生成月报
                if await self._should_generate_monthly_report(current_time):
                    await self._generate_scheduled_report(ReportType.MONTHLY)
                
                # 每小时检查一次
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Error in scheduler loop: {e}")
                await asyncio.sleep(300)  # 5分钟后重试
    
    async def _should_generate_daily_report(self, current_time: datetime) -> bool:
        """检查是否应该生成日报"""
        report_time_parts = self.daily_report_time.split(':')
        target_hour = int(report_time_parts[0])
        target_minute = int(report_time_parts[1]) if len(report_time_parts) > 1 else 0
        
        # 检查是否接近目标时间
        return (current_time.hour == target_hour and 
                current_time.minute >= target_minute and
                current_time.minute < target_minute + 10)  # 10分钟窗口
    
    async def _should_generate_weekly_report(self, current_time: datetime) -> bool:
        """检查是否应该生成周报"""
        # 简化：每周一生成
        return (current_time.weekday() == 0 and  # 周一
                current_time.hour == 7 and       # 7点
                current_time.minute < 10)         # 10分钟窗口
    
    async def _should_generate_monthly_report(self, current_time: datetime) -> bool:
        """检查是否应该生成月报"""
        # 简化：每月第一天生成
        return (current_time.day == 1 and
                current_time.hour == 8 and
                current_time.minute < 10)
    
    async def _generate_scheduled_report(self, report_type: ReportType) -> None:
        """生成定时报告"""
        try:
            current_time = datetime.now(timezone.utc)
            
            if report_type == ReportType.DAILY:
                period_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
                period_end = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            elif report_type == ReportType.WEEKLY:
                days_since_monday = current_time.weekday()
                period_start = current_time - timedelta(days=days_since_monday + 7)
                period_end = current_time - timedelta(days=days_since_monday)
            else:  # MONTHLY
                # 上个月
                if current_time.month == 1:
                    period_end = current_time.replace(year=current_time.year-1, month=12, day=1)
                else:
                    period_end = current_time.replace(month=current_time.month-1, day=1)
                
                # 上个月第一天到最后一天
                period_start = period_end
                if period_end.month == 12:
                    period_end = period_end.replace(year=period_end.year+1, month=1, day=1) - timedelta(days=1)
                else:
                    period_end = period_end.replace(month=period_end.month+1, day=1) - timedelta(days=1)
            
            report = await self.generate_report(report_type, period_start, period_end)
            filename = f"{report_type.value}_report_{period_start.strftime('%Y%m%d')}_{report.report_id}.html"
            
            await self.save_report(report, filename)
            logger.info(f"📊 Generated scheduled {report_type.value} report: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to generate scheduled {report_type.value} report: {e}")
    
    async def generate_report(
        self,
        report_type: ReportType,
        period_start: datetime,
        period_end: datetime
    ) -> ReportData:
        """生成报告"""
        report_id = f"{report_type.value}_{int(time.time())}"
        generated_at = datetime.now(timezone.utc)
        
        # 计算指标
        metrics = self.metrics_calculator.calculate_metrics(period_start, period_end)
        
        # 获取期间内的数据
        period_trades = [
            trade for trade in self.metrics_calculator.trade_history
            if period_start <= trade.get('timestamp', datetime.min.replace(tzinfo=timezone.utc)) <= period_end
        ]
        
        period_risk = [
            risk for risk in self.metrics_calculator.risk_history
            if period_start <= risk.get('timestamp', datetime.min.replace(tzinfo=timezone.utc)) <= period_end
        ]
        
        # 生成图表
        charts = []
        if period_trades:
            charts.extend([
                self.chart_generator.generate_pnl_chart(period_trades),
                self.chart_generator.generate_drawdown_chart(period_trades),
                self.chart_generator.generate_performance_distribution(period_trades)
            ])
        
        if period_risk:
            charts.append(self.chart_generator.generate_risk_metrics_chart(period_risk))
        
        # 生成表格
        tables = []
        if period_trades:
            tables.extend([
                self.table_generator.generate_trade_summary_table(period_trades),
                self.table_generator.generate_best_worst_trades(period_trades)
            ])
        
        if report_type == ReportType.MONTHLY:
            tables.append(self.table_generator.generate_monthly_performance_table(
                self.metrics_calculator.trade_history, 12
            ))
        
        # 生成文本内容
        sections = await self._generate_report_sections(metrics, period_trades, report_type)
        
        return ReportData(
            report_id=report_id,
            report_type=report_type,
            period_start=period_start,
            period_end=period_end,
            generated_at=generated_at,
            metrics=metrics,
            charts=charts,
            tables=tables,
            sections=sections,
            metadata={
                'trade_count': len(period_trades),
                'risk_data_points': len(period_risk),
                'generation_time_seconds': (datetime.now(timezone.utc) - generated_at).total_seconds()
            }
        )
    
    async def _generate_report_sections(
        self,
        metrics: ReportMetrics,
        trades: List[Dict[str, Any]],
        report_type: ReportType
    ) -> Dict[ReportSection, str]:
        """生成报告章节内容"""
        sections = {}
        
        # 执行摘要
        if metrics.total_trades > 0:
            performance_status = "strong" if metrics.win_rate > 60 and metrics.profit_factor > 1.5 else \
                               "good" if metrics.win_rate > 50 and metrics.profit_factor > 1.2 else \
                               "concerning" if metrics.win_rate < 40 or metrics.profit_factor < 0.8 else "mixed"
            
            sections[ReportSection.EXECUTIVE_SUMMARY] = f"""
            <p>During the reporting period, the DipMaster strategy executed <strong>{metrics.total_trades} trades</strong> 
            with a win rate of <strong>{metrics.win_rate:.1f}%</strong> and achieved a total P&L of 
            <strong>${metrics.total_pnl:.2f}</strong>.</p>
            
            <p>The strategy demonstrated <strong>{performance_status}</strong> performance with a profit factor of 
            <strong>{metrics.profit_factor:.2f}</strong> and Sharpe ratio of <strong>{metrics.sharpe_ratio:.2f}</strong>. 
            Maximum drawdown was contained at <strong>{metrics.max_drawdown:.1%}</strong>.</p>
            
            <p>Average holding time was <strong>{metrics.avg_holding_time:.0f} minutes</strong>, consistent with the 
            DipMaster strategy's quick turnaround approach. Execution quality remained high with average slippage 
            of <strong>{metrics.avg_slippage:.1f} basis points</strong>.</p>
            """
        else:
            sections[ReportSection.EXECUTIVE_SUMMARY] = """
            <p>No trading activity occurred during this reporting period. This may indicate market conditions 
            were not suitable for the DipMaster strategy, or system issues prevented trade execution.</p>
            """
        
        # 性能分析
        sections[ReportSection.PERFORMANCE_ANALYSIS] = await self._generate_performance_analysis(metrics, trades)
        
        # 风险分析
        sections[ReportSection.RISK_ANALYSIS] = await self._generate_risk_analysis(metrics)
        
        # 建议
        sections[ReportSection.RECOMMENDATIONS] = await self._generate_recommendations(metrics, trades, report_type)
        
        return sections
    
    async def _generate_performance_analysis(self, metrics: ReportMetrics, trades: List[Dict[str, Any]]) -> str:
        """生成性能分析内容"""
        if not trades:
            return "<p>No performance data available for analysis.</p>"
        
        # 分析交易时段分布
        hourly_distribution = {}
        for trade in trades:
            hour = trade.get('timestamp', datetime.now()).hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        
        best_hour = max(hourly_distribution, key=hourly_distribution.get) if hourly_distribution else 0
        
        return f"""
        <h3>Trading Activity Analysis</h3>
        <p>The strategy showed strongest activity during <strong>{best_hour}:00 UTC</strong> with 
        {hourly_distribution.get(best_hour, 0)} trades executed.</p>
        
        <h3>Profit Distribution</h3>
        <p>Of the {metrics.total_trades} trades executed:</p>
        <ul>
            <li><strong>{metrics.winning_trades}</strong> were profitable (gross profit: ${metrics.gross_profit:.2f})</li>
            <li><strong>{metrics.losing_trades}</strong> resulted in losses (gross loss: ${metrics.gross_loss:.2f})</li>
            <li>Average trade return: <strong>${metrics.avg_trade_return:.2f}</strong></li>
        </ul>
        
        <h3>Risk-Adjusted Performance</h3>
        <p>The strategy achieved a Sharpe ratio of <strong>{metrics.sharpe_ratio:.2f}</strong>, indicating 
        {'excellent' if metrics.sharpe_ratio > 2 else 'good' if metrics.sharpe_ratio > 1 else 'poor'} 
        risk-adjusted returns. Return volatility was <strong>{metrics.volatility:.2%}</strong>.</p>
        """
    
    async def _generate_risk_analysis(self, metrics: ReportMetrics) -> str:
        """生成风险分析内容"""
        return f"""
        <h3>Value at Risk Analysis</h3>
        <p>Current risk exposure metrics:</p>
        <ul>
            <li><strong>95% VaR:</strong> ${metrics.var_95:,.2f}</li>
            <li><strong>99% VaR:</strong> ${metrics.var_99:,.2f}</li>
            <li><strong>Expected Shortfall:</strong> ${metrics.expected_shortfall:,.2f}</li>
        </ul>
        
        <h3>Drawdown Analysis</h3>
        <p>Maximum drawdown reached <strong>{metrics.max_drawdown:.1%}</strong>, which is 
        {'within acceptable limits' if metrics.max_drawdown < 0.15 else 'elevated and requires attention'}.</p>
        
        <h3>Leverage and Exposure</h3>
        <p>Maximum leverage utilized was <strong>{metrics.max_leverage:.1f}x</strong>, 
        {'maintaining conservative risk levels' if metrics.max_leverage < 2 else 'indicating aggressive positioning'}.</p>
        
        <h3>Risk Management Effectiveness</h3>
        <p>The strategy's risk controls {'performed effectively' if metrics.max_drawdown < 0.15 else 'need review'} 
        during the reporting period, with drawdown levels 
        {'well below' if metrics.max_drawdown < 0.1 else 'at' if metrics.max_drawdown < 0.15 else 'above'} 
        target thresholds.</p>
        """
    
    async def _generate_recommendations(self, metrics: ReportMetrics, trades: List[Dict[str, Any]], report_type: ReportType) -> str:
        """生成建议内容"""
        recommendations = []
        
        # 性能建议
        if metrics.win_rate < 50:
            recommendations.append("🔍 Win rate below target (50%): Review signal quality and entry criteria")
        
        if metrics.profit_factor < 1.2:
            recommendations.append("📊 Profit factor below target (1.2): Consider adjusting position sizing or exit timing")
        
        if metrics.max_drawdown > 0.15:
            recommendations.append("⚠️ Maximum drawdown exceeded 15%: Implement stricter risk controls")
        
        if metrics.avg_slippage > 15:
            recommendations.append("⚡ Average slippage above 15 bps: Optimize execution timing or venue selection")
        
        # 系统建议
        if metrics.critical_alerts > 5:
            recommendations.append("🚨 High number of critical alerts: Investigate system stability issues")
        
        if len(trades) > 0:
            avg_holding = statistics.mean([t.get('holding_minutes', 0) for t in trades])
            if avg_holding > 180:  # DipMaster目标：60-120分钟
                recommendations.append("⏱️ Average holding time exceeded target: Review exit logic")
        
        # 市场环境建议
        if metrics.volatility > 0.05:  # 5%日波动率
            recommendations.append("🌊 High market volatility detected: Consider reducing position sizes")
        
        # 长期建议（周报/月报）
        if report_type in [ReportType.WEEKLY, ReportType.MONTHLY]:
            if metrics.sharpe_ratio < 1.5:
                recommendations.append("📈 Sharpe ratio below target: Consider strategy parameter optimization")
            
            recommendations.append("🔄 Schedule quarterly strategy review and parameter rebalancing")
        
        # 默认建议
        if not recommendations:
            recommendations.append("✅ Strategy performance is within acceptable parameters")
            recommendations.append("📋 Continue monitoring key metrics and maintain current risk controls")
        
        return "<ul>" + "".join(f"<li>{rec}</li>" for rec in recommendations) + "</ul>"
    
    async def save_report(self, report_data: ReportData, filename: str) -> str:
        """保存报告"""
        html_content = self.report_renderer.render_report(report_data)
        
        report_path = self.reports_dir / filename
        
        # 保存HTML文件
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # 同时保存JSON元数据
        metadata_path = report_path.with_suffix('.json')
        metadata = {
            'report_id': report_data.report_id,
            'report_type': report_data.report_type.value,
            'period_start': report_data.period_start.isoformat(),
            'period_end': report_data.period_end.isoformat(),
            'generated_at': report_data.generated_at.isoformat(),
            'metrics': asdict(report_data.metrics),
            'metadata': report_data.metadata
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"📊 Saved report: {report_path}")
        return str(report_path)
    
    # 公共接口方法
    
    def add_trade_data(self, trade_data: Dict[str, Any]) -> None:
        """添加交易数据"""
        self.metrics_calculator.add_trade_record(trade_data)
    
    def add_risk_data(self, risk_data: Dict[str, Any]) -> None:
        """添加风险数据"""
        self.metrics_calculator.add_risk_record(risk_data)
    
    def add_execution_data(self, execution_data: Dict[str, Any]) -> None:
        """添加执行数据"""
        self.metrics_calculator.add_execution_record(execution_data)
    
    def add_system_event(self, event_data: Dict[str, Any]) -> None:
        """添加系统事件"""
        self.metrics_calculator.add_system_event(event_data)
    
    async def generate_custom_report(
        self,
        period_start: datetime,
        period_end: datetime,
        title: str = "Custom Report"
    ) -> ReportData:
        """生成自定义报告"""
        return await self.generate_report(ReportType.CUSTOM, period_start, period_end)
    
    def get_recent_reports(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取最近的报告列表"""
        reports = []
        
        for json_file in sorted(self.reports_dir.glob('*.json'), key=os.path.getmtime, reverse=True)[:count]:
            try:
                with open(json_file, 'r') as f:
                    metadata = json.load(f)
                    reports.append(metadata)
            except Exception as e:
                logger.error(f"❌ Failed to load report metadata {json_file}: {e}")
        
        return reports


# 工厂函数
def create_reporting_system(config: Dict[str, Any] = None) -> AutomatedReportingSystem:
    """创建自动化报告系统"""
    return AutomatedReportingSystem(config)


# 演示函数
async def reporting_system_demo():
    """报告系统演示"""
    print("🚀 DipMaster Automated Reporting System Demo")
    
    # 创建报告系统
    config = {
        'reports_dir': 'demo_reports',
        'daily_report_time': '06:00',
        'weekly_report_day': 'monday'
    }
    
    reporting_system = create_reporting_system(config)
    
    try:
        # 添加模拟数据
        print("📊 Adding sample trading data...")
        
        base_time = datetime.now(timezone.utc) - timedelta(days=7)
        
        # 添加交易数据
        for i in range(20):
            pnl = (i - 10) * 5.5  # 模拟盈亏
            reporting_system.add_trade_data({
                'timestamp': base_time + timedelta(hours=i*6),
                'symbol': 'BTCUSDT' if i % 2 == 0 else 'ETHUSDT',
                'side': 'BUY',
                'quantity': 0.1,
                'pnl': pnl,
                'realized_pnl': pnl,
                'return_pct': (pnl / 1000) * 100,  # 假设1000美元基准
                'holding_minutes': 75 + (i % 60),
                'strategy': 'dipmaster'
            })
        
        # 添加风险数据
        for i in range(10):
            reporting_system.add_risk_data({
                'timestamp': base_time + timedelta(days=i),
                'var_95': 120000 + i * 5000,
                'var_99': 180000 + i * 7500,
                'expected_shortfall': 200000 + i * 10000,
                'max_drawdown': 0.08 + (i * 0.01),
                'max_leverage': 1.5
            })
        
        # 添加执行数据
        for i in range(15):
            reporting_system.add_execution_data({
                'timestamp': base_time + timedelta(hours=i*12),
                'symbol': 'BTCUSDT',
                'slippage_bps': 2.5 + (i % 10),
                'latency_ms': 45 + (i % 30),
                'status': 'FILLED'
            })
        
        print("📈 Generating weekly report...")
        
        # 生成周报
        period_start = base_time
        period_end = datetime.now(timezone.utc)
        
        report = await reporting_system.generate_report(
            ReportType.WEEKLY,
            period_start,
            period_end
        )
        
        print(f"📊 Generated report: {report.report_id}")
        print(f"   Report Type: {report.report_type.value}")
        print(f"   Period: {report.period_start.date()} to {report.period_end.date()}")
        print(f"   Total Trades: {report.metrics.total_trades}")
        print(f"   Win Rate: {report.metrics.win_rate:.1f}%")
        print(f"   Total P&L: ${report.metrics.total_pnl:.2f}")
        print(f"   Profit Factor: {report.metrics.profit_factor:.2f}")
        print(f"   Charts: {len(report.charts)}")
        print(f"   Tables: {len(report.tables)}")
        
        # 保存报告
        filename = f"demo_weekly_report_{int(time.time())}.html"
        saved_path = await reporting_system.save_report(report, filename)
        print(f"📄 Report saved to: {saved_path}")
        
        # 生成自定义日报
        print("\n📈 Generating custom daily report...")
        
        daily_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        daily_end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        daily_report = await reporting_system.generate_custom_report(
            daily_start,
            daily_end,
            "Custom Daily Performance Report"
        )
        
        daily_filename = f"demo_daily_report_{int(time.time())}.html"
        daily_path = await reporting_system.save_report(daily_report, daily_filename)
        print(f"📄 Daily report saved to: {daily_path}")
        
        # 显示报告列表
        recent_reports = reporting_system.get_recent_reports(5)
        print(f"\n📋 Recent reports ({len(recent_reports)}):")
        for report_meta in recent_reports:
            print(f"   {report_meta['report_type']} - {report_meta['period_start'][:10]} to {report_meta['period_end'][:10]}")
        
        print("✅ Demo completed successfully")
        print(f"📂 Check the '{config['reports_dir']}' directory for generated HTML reports")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await reporting_system.stop()
        print("🛑 Reporting system stopped")


if __name__ == "__main__":
    asyncio.run(reporting_system_demo())