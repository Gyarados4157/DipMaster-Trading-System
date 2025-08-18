#!/usr/bin/env python3
"""
Data Infrastructure Status Dashboard
DipMaster Trading System - 数据基础设施状态仪表板

实时显示数据基础设施的运行状态、质量指标和监控信息
"""

import sys
import time
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import argparse

class DataInfrastructureStatus:
    """数据基础设施状态显示器"""
    
    def __init__(self):
        self.data_path = Path("data/enhanced_market_data")
        self.monitoring_db = Path("data/monitoring.db")
        self.reports_path = Path("data")
        
        # TOP30符号
        self.top30_symbols = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT",
            "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "TRXUSDT", "LINKUSDT",
            "LTCUSDT", "DOTUSDT", "MATICUSDT", "UNIUSDT", "ICPUSDT",
            "NEARUSDT", "XLMUSDT", "ATOMUSDT", "VETUSDT", "FILUSDT",
            "APTUSDT", "ARBUSDT", "OPUSDT", "GRTUSDT", "MKRUSDT",
            "AAVEUSDT", "COMPUSDT", "ALGOUSDT", "TONUSDT", "INJUSDT"
        ]
        
        self.timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    def get_file_coverage_status(self) -> Dict[str, Any]:
        """获取文件覆盖状态"""
        if not self.data_path.exists():
            return {
                'status': 'no_data_directory',
                'total_files': 0,
                'expected_files': len(self.top30_symbols) * len(self.timeframes),
                'coverage_percentage': 0
            }
        
        parquet_files = list(self.data_path.glob("*.parquet"))
        expected_files = len(self.top30_symbols) * len(self.timeframes)
        
        # 按交易对统计
        symbol_coverage = {}
        for symbol in self.top30_symbols:
            symbol_files = [f for f in parquet_files if f.name.startswith(f"{symbol}_")]
            symbol_coverage[symbol] = {
                'files': len(symbol_files),
                'expected': len(self.timeframes),
                'coverage': len(symbol_files) / len(self.timeframes) * 100
            }
        
        # 按时间框架统计
        timeframe_coverage = {}
        for tf in self.timeframes:
            tf_files = [f for f in parquet_files if f"_{tf}_" in f.name]
            timeframe_coverage[tf] = {
                'files': len(tf_files),
                'expected': len(self.top30_symbols),
                'coverage': len(tf_files) / len(self.top30_symbols) * 100
            }
        
        # 文件大小统计
        total_size = sum(f.stat().st_size for f in parquet_files)
        avg_size = total_size / len(parquet_files) if parquet_files else 0
        
        return {
            'status': 'active',
            'total_files': len(parquet_files),
            'expected_files': expected_files,
            'coverage_percentage': len(parquet_files) / expected_files * 100 if expected_files > 0 else 0,
            'total_size_gb': total_size / 1024**3,
            'avg_file_size_mb': avg_size / 1024**2,
            'by_symbol': symbol_coverage,
            'by_timeframe': timeframe_coverage,
            'last_updated': self._get_latest_file_timestamp()
        }
    
    def _get_latest_file_timestamp(self) -> Optional[str]:
        """获取最新文件时间戳"""
        if not self.data_path.exists():
            return None
        
        parquet_files = list(self.data_path.glob("*.parquet"))
        if not parquet_files:
            return None
        
        latest_mtime = max(f.stat().st_mtime for f in parquet_files)
        return datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat()
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """获取质量指标"""
        if not self.monitoring_db.exists():
            return {
                'status': 'no_monitoring_data',
                'message': '监控数据库不存在'
            }
        
        try:
            conn = sqlite3.connect(str(self.monitoring_db))
            
            # 最新的平均质量分数
            cursor = conn.execute("""
                SELECT AVG(overall_score) as avg_score,
                       AVG(completeness) as avg_completeness,
                       AVG(consistency) as avg_consistency, 
                       AVG(accuracy) as avg_accuracy,
                       COUNT(*) as total_assessments
                FROM quality_metrics 
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            
            result = cursor.fetchone()
            if result and result[0] is not None:
                avg_metrics = {
                    'overall_score': result[0],
                    'completeness': result[1],
                    'consistency': result[2],
                    'accuracy': result[3],
                    'assessments_24h': result[4]
                }
            else:
                avg_metrics = {
                    'overall_score': 0,
                    'completeness': 0,
                    'consistency': 0,
                    'accuracy': 0,
                    'assessments_24h': 0
                }
            
            # 按时间框架统计质量
            cursor = conn.execute("""
                SELECT timeframe, AVG(overall_score) as avg_score
                FROM quality_metrics 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY timeframe
                ORDER BY avg_score DESC
            """)
            
            timeframe_quality = dict(cursor.fetchall())
            
            # 质量趋势 (最近24小时每小时平均)
            cursor = conn.execute("""
                SELECT strftime('%H', timestamp) as hour,
                       AVG(overall_score) as avg_score
                FROM quality_metrics 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY hour
                ORDER BY hour
            """)
            
            hourly_trend = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'status': 'active',
                'average_metrics': avg_metrics,
                'by_timeframe': timeframe_quality,
                'hourly_trend': hourly_trend,
                'last_check': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'质量指标获取失败: {e}'
            }
    
    def get_alert_status(self) -> Dict[str, Any]:
        """获取告警状态"""
        if not self.monitoring_db.exists():
            return {
                'status': 'no_monitoring_data',
                'total_alerts': 0,
                'active_alerts': 0
            }
        
        try:
            conn = sqlite3.connect(str(self.monitoring_db))
            
            # 最近24小时告警统计
            cursor = conn.execute("""
                SELECT severity, COUNT(*) as count
                FROM alerts 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY severity
            """)
            
            alerts_by_severity = dict(cursor.fetchall())
            
            # 最近的告警
            cursor = conn.execute("""
                SELECT symbol, timeframe, alert_type, severity, message, timestamp
                FROM alerts 
                WHERE timestamp > datetime('now', '-24 hours')
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            
            recent_alerts = []
            for row in cursor.fetchall():
                recent_alerts.append({
                    'symbol': row[0],
                    'timeframe': row[1],
                    'type': row[2],
                    'severity': row[3],
                    'message': row[4],
                    'timestamp': row[5]
                })
            
            # 告警类型统计
            cursor = conn.execute("""
                SELECT alert_type, COUNT(*) as count
                FROM alerts 
                WHERE timestamp > datetime('now', '-24 hours')
                GROUP BY alert_type
            """)
            
            alerts_by_type = dict(cursor.fetchall())
            
            conn.close()
            
            total_alerts = sum(alerts_by_severity.values())
            critical_alerts = alerts_by_severity.get('critical', 0) + alerts_by_severity.get('high', 0)
            
            return {
                'status': 'active',
                'total_alerts_24h': total_alerts,
                'critical_alerts': critical_alerts,
                'by_severity': alerts_by_severity,
                'by_type': alerts_by_type,
                'recent_alerts': recent_alerts,
                'last_check': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'告警状态获取失败: {e}'
            }
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统健康度"""
        coverage = self.get_file_coverage_status()
        quality = self.get_quality_metrics()
        alerts = self.get_alert_status()
        
        # 计算健康度评分
        health_score = 0
        
        # 数据覆盖度 (40%)
        if coverage['status'] == 'active':
            coverage_score = coverage['coverage_percentage'] / 100
            health_score += coverage_score * 0.4
        
        # 数据质量 (40%)
        if quality['status'] == 'active' and quality['average_metrics']['overall_score'] > 0:
            quality_score = quality['average_metrics']['overall_score']
            health_score += quality_score * 0.4
        
        # 告警状态 (20%)
        alert_penalty = 0
        if alerts['status'] == 'active':
            # 严重告警扣分
            critical_count = alerts.get('critical_alerts', 0)
            alert_penalty = min(critical_count * 0.1, 0.2)  # 最多扣20%
        
        health_score = max(0, health_score - alert_penalty)
        
        # 健康等级
        if health_score >= 0.9:
            health_level = 'excellent'
            health_color = '🟢'
        elif health_score >= 0.8:
            health_level = 'good'
            health_color = '🟢'
        elif health_score >= 0.6:
            health_level = 'fair'
            health_color = '🟡'
        elif health_score >= 0.4:
            health_level = 'poor'
            health_color = '🟠'
        else:
            health_level = 'critical'
            health_color = '🔴'
        
        return {
            'health_score': health_score,
            'health_level': health_level,
            'health_color': health_color,
            'components': {
                'data_coverage': coverage,
                'data_quality': quality,
                'alert_status': alerts
            },
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def print_status_dashboard(self):
        """打印状态仪表板"""
        health = self.get_system_health()
        
        # 标题
        print("\n" + "="*80)
        print(f"{health['health_color']} DipMaster Data Infrastructure Status Dashboard")
        print(f"📊 System Health: {health['health_level'].upper()} ({health['health_score']:.1%})")
        print(f"🕐 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # 数据覆盖状态
        coverage = health['components']['data_coverage']
        print(f"\n📁 DATA COVERAGE")
        print(f"   Files: {coverage['total_files']:,} / {coverage['expected_files']:,} ({coverage['coverage_percentage']:.1f}%)")
        print(f"   Size:  {coverage['total_size_gb']:.2f} GB")
        print(f"   Updated: {coverage.get('last_updated', 'Unknown')}")
        
        # 数据质量状态
        quality = health['components']['data_quality']
        print(f"\n📈 DATA QUALITY")
        if quality['status'] == 'active':
            metrics = quality['average_metrics']
            print(f"   Overall:     {metrics['overall_score']:.1%}")
            print(f"   Completeness: {metrics['completeness']:.1%}")
            print(f"   Consistency:  {metrics['consistency']:.1%}")
            print(f"   Accuracy:     {metrics['accuracy']:.1%}")
            print(f"   Assessments:  {metrics['assessments_24h']} (24h)")
        else:
            print(f"   Status: {quality.get('message', 'No data')}")
        
        # 告警状态
        alerts = health['components']['alert_status']
        print(f"\n🚨 ALERTS (24h)")
        if alerts['status'] == 'active':
            print(f"   Total: {alerts['total_alerts_24h']:,}")
            print(f"   Critical: {alerts['critical_alerts']:,}")
            
            if alerts['by_severity']:
                print(f"   By Severity: ", end="")
                severity_strs = [f"{sev}:{count}" for sev, count in alerts['by_severity'].items()]
                print(", ".join(severity_strs))
            
            if alerts['recent_alerts']:
                print(f"   Recent:")
                for alert in alerts['recent_alerts'][:3]:  # 显示最近3个
                    print(f"     • {alert['symbol']} {alert['timeframe']}: {alert['message'][:50]}...")
        else:
            print(f"   Status: {alerts.get('message', 'No data')}")
        
        # TOP币种状态
        if coverage['status'] == 'active' and coverage['by_symbol']:
            print(f"\n💰 TOP SYMBOLS STATUS")
            
            # 按覆盖率排序显示前10和后5
            symbol_items = list(coverage['by_symbol'].items())
            symbol_items.sort(key=lambda x: x[1]['coverage'], reverse=True)
            
            print("   Best Coverage:")
            for symbol, data in symbol_items[:5]:
                status_icon = "✅" if data['coverage'] == 100 else "⚠️" if data['coverage'] >= 50 else "❌"
                print(f"     {status_icon} {symbol}: {data['files']}/{data['expected']} ({data['coverage']:.0f}%)")
            
            if len(symbol_items) > 5:
                print("   Needs Attention:")
                for symbol, data in symbol_items[-3:]:
                    status_icon = "✅" if data['coverage'] == 100 else "⚠️" if data['coverage'] >= 50 else "❌"
                    print(f"     {status_icon} {symbol}: {data['files']}/{data['expected']} ({data['coverage']:.0f}%)")
        
        # 系统建议
        print(f"\n💡 RECOMMENDATIONS")
        if health['health_score'] >= 0.9:
            print("   🎉 System is performing excellently!")
            print("   • Continue monitoring for optimal performance")
        elif health['health_score'] >= 0.8:
            print("   ✨ System is performing well")
            print("   • Monitor for any quality degradation")
        elif health['health_score'] >= 0.6:
            print("   ⚠️  System needs attention")
            if coverage['coverage_percentage'] < 80:
                print("   • Improve data coverage by downloading missing files")
            if alerts['critical_alerts'] > 0:
                print("   • Address critical alerts immediately")
        else:
            print("   🚨 System requires immediate attention!")
            print("   • Check system logs for errors")
            print("   • Consider running initial data collection")
            print("   • Verify exchange connectivity")
        
        print("\n" + "="*80 + "\n")
    
    def print_detailed_status(self):
        """打印详细状态"""
        health = self.get_system_health()
        
        print(json.dumps(health, indent=2, default=str))
    
    def export_status_report(self) -> str:
        """导出状态报告"""
        health = self.get_system_health()
        
        report_file = f"data/infrastructure_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(health, f, indent=2, default=str)
        
        return report_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DipMaster Data Infrastructure Status Dashboard")
    parser.add_argument('--json', action='store_true', help='输出JSON格式')
    parser.add_argument('--export', action='store_true', help='导出状态报告')
    parser.add_argument('--watch', type=int, metavar='SECONDS', help='监视模式，每N秒刷新')
    
    args = parser.parse_args()
    
    status_monitor = DataInfrastructureStatus()
    
    if args.export:
        report_file = status_monitor.export_status_report()
        print(f"状态报告已导出到: {report_file}")
        return
    
    if args.watch:
        try:
            while True:
                # 清屏 (在Unix系统上)
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                
                if args.json:
                    status_monitor.print_detailed_status()
                else:
                    status_monitor.print_status_dashboard()
                
                print(f"🔄 Refreshing in {args.watch} seconds... (Ctrl+C to stop)")
                time.sleep(args.watch)
                
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped.")
            return
    
    if args.json:
        status_monitor.print_detailed_status()
    else:
        status_monitor.print_status_dashboard()

if __name__ == "__main__":
    main()