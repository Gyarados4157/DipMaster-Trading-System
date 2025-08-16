#!/usr/bin/env python3
"""
System Monitor for DipMaster Trading System
系统监控器 - 监控系统资源和健康状况

Features:
- CPU, memory, disk, and network monitoring
- Process-specific resource tracking
- WebSocket connection health monitoring
- Database connection pool monitoring
- API rate limiting and error tracking
- System alerts and threshold monitoring
"""

import os
import time
import psutil
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import asyncio
import socket

from .metrics_collector import MetricsCollector, get_metrics_collector

logger = logging.getLogger(__name__)


@dataclass
class SystemThresholds:
    """System monitoring thresholds for alerts."""
    cpu_usage_warning: float = 80.0      # %
    cpu_usage_critical: float = 95.0     # %
    memory_usage_warning: float = 85.0   # %
    memory_usage_critical: float = 95.0  # %
    disk_usage_warning: float = 85.0     # %
    disk_usage_critical: float = 95.0    # %
    network_error_rate_warning: float = 5.0    # %
    network_error_rate_critical: float = 15.0  # %
    response_time_warning: float = 1000.0      # ms
    response_time_critical: float = 5000.0     # ms


@dataclass
class ProcessInfo:
    """Process information tracking."""
    pid: int
    name: str
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_rss: int = 0  # bytes
    memory_vms: int = 0  # bytes
    num_threads: int = 0
    num_fds: int = 0  # file descriptors (Unix only)
    connections: int = 0
    create_time: float = field(default_factory=time.time)


@dataclass
class NetworkStats:
    """Network statistics tracking."""
    bytes_sent: int = 0
    bytes_recv: int = 0
    packets_sent: int = 0
    packets_recv: int = 0
    errors_in: int = 0
    errors_out: int = 0
    drops_in: int = 0
    drops_out: int = 0
    timestamp: float = field(default_factory=time.time)


class SystemMonitor:
    """
    Comprehensive system monitoring for trading system.
    
    Monitors system resources, process health, network conditions,
    and provides alerting for threshold violations.
    """
    
    def __init__(self,
                 metrics_collector: Optional[MetricsCollector] = None,
                 monitoring_interval: int = 30,  # seconds
                 thresholds: Optional[SystemThresholds] = None):
        """
        Initialize system monitor.
        
        Args:
            metrics_collector: Metrics collector instance
            monitoring_interval: Seconds between monitoring cycles
            thresholds: Alert thresholds configuration
        """
        self.metrics_collector = metrics_collector or get_metrics_collector()
        self.monitoring_interval = monitoring_interval
        self.thresholds = thresholds or SystemThresholds()
        
        # System information
        self.system_info = self._get_system_info()
        
        # Process tracking
        self._process_info: Optional[ProcessInfo] = None
        self._process = None
        
        # Historical data
        self._cpu_history: List[float] = []
        self._memory_history: List[float] = []
        self._network_history: List[NetworkStats] = []
        
        # WebSocket monitoring
        self._websocket_connections: Dict[str, Dict[str, Any]] = {}
        self._api_call_stats: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring control
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Register system metrics
        self._register_system_metrics()
        
        # Get current process info
        self._initialize_process_tracking()
        
        logger.info("🖥️  SystemMonitor initialized")
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get static system information."""
        try:
            boot_time = datetime.fromtimestamp(psutil.boot_time())
            
            return {
                'platform': os.uname().sysname if hasattr(os, 'uname') else 'Unknown',
                'hostname': socket.gethostname(),
                'cpu_count': psutil.cpu_count(),
                'cpu_count_logical': psutil.cpu_count(logical=True),
                'memory_total': psutil.virtual_memory().total,
                'boot_time': boot_time.isoformat(),
                'python_version': os.sys.version
            }
        except Exception as e:
            logger.error(f"❌ Failed to get system info: {e}")
            return {}
    
    def _register_system_metrics(self):
        """Register system monitoring metrics."""
        from .metrics_collector import MetricType
        
        system_metrics = [
            # CPU Metrics
            ('system.cpu.usage_percent', MetricType.GAUGE, 'CPU usage percentage', '%'),
            ('system.cpu.load_1m', MetricType.GAUGE, '1-minute load average', 'load'),
            ('system.cpu.load_5m', MetricType.GAUGE, '5-minute load average', 'load'),
            ('system.cpu.load_15m', MetricType.GAUGE, '15-minute load average', 'load'),
            
            # Memory Metrics
            ('system.memory.usage_percent', MetricType.GAUGE, 'Memory usage percentage', '%'),
            ('system.memory.available_mb', MetricType.GAUGE, 'Available memory', 'MB'),
            ('system.memory.used_mb', MetricType.GAUGE, 'Used memory', 'MB'),
            ('system.memory.cached_mb', MetricType.GAUGE, 'Cached memory', 'MB'),
            ('system.memory.swap_usage_percent', MetricType.GAUGE, 'Swap usage percentage', '%'),
            
            # Disk Metrics
            ('system.disk.usage_percent', MetricType.GAUGE, 'Disk usage percentage', '%'),
            ('system.disk.free_gb', MetricType.GAUGE, 'Free disk space', 'GB'),
            ('system.disk.used_gb', MetricType.GAUGE, 'Used disk space', 'GB'),
            ('system.disk.read_mb_per_sec', MetricType.GAUGE, 'Disk read rate', 'MB/s'),
            ('system.disk.write_mb_per_sec', MetricType.GAUGE, 'Disk write rate', 'MB/s'),
            
            # Network Metrics
            ('system.network.bytes_sent_per_sec', MetricType.GAUGE, 'Network bytes sent rate', 'bytes/s'),
            ('system.network.bytes_recv_per_sec', MetricType.GAUGE, 'Network bytes received rate', 'bytes/s'),
            ('system.network.packets_sent_per_sec', MetricType.GAUGE, 'Network packets sent rate', 'packets/s'),
            ('system.network.packets_recv_per_sec', MetricType.GAUGE, 'Network packets received rate', 'packets/s'),
            ('system.network.error_rate', MetricType.GAUGE, 'Network error rate', '%'),
            
            # Process Metrics
            ('process.cpu_percent', MetricType.GAUGE, 'Process CPU usage', '%'),
            ('process.memory_percent', MetricType.GAUGE, 'Process memory usage', '%'),
            ('process.memory_rss_mb', MetricType.GAUGE, 'Process RSS memory', 'MB'),
            ('process.num_threads', MetricType.GAUGE, 'Number of process threads', 'count'),
            ('process.num_connections', MetricType.GAUGE, 'Number of network connections', 'count'),
            ('process.num_file_descriptors', MetricType.GAUGE, 'Number of file descriptors', 'count'),
            
            # Application Metrics
            ('app.websocket_connections', MetricType.GAUGE, 'Active WebSocket connections', 'count'),
            ('app.api_calls_per_minute', MetricType.GAUGE, 'API calls per minute', 'calls/min'),
            ('app.api_error_rate', MetricType.GAUGE, 'API error rate', '%'),
            ('app.avg_response_time_ms', MetricType.GAUGE, 'Average API response time', 'ms'),
            
            # Health Metrics
            ('health.overall_score', MetricType.GAUGE, 'Overall system health score', 'score'),
            ('health.alerts_active', MetricType.GAUGE, 'Number of active alerts', 'count'),
            ('health.uptime_hours', MetricType.GAUGE, 'System uptime', 'hours')
        ]
        
        for name, metric_type, description, unit in system_metrics:
            self.metrics_collector.register_metric(name, metric_type, description, unit)
    
    def _initialize_process_tracking(self):
        """Initialize process tracking for current process."""
        try:
            self._process = psutil.Process()
            self._process_info = ProcessInfo(
                pid=self._process.pid,
                name=self._process.name(),
                create_time=self._process.create_time()
            )
            logger.debug(f"🔍 Process tracking initialized: PID {self._process.pid}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize process tracking: {e}")
    
    def start_monitoring(self):
        """Start system monitoring in background thread."""
        if self._monitoring_active:
            logger.warning("⚠️  System monitoring already active")
            return
        
        self._monitoring_active = True
        self._stop_event.clear()
        
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("🚀 System monitoring started")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        if not self._monitoring_active:
            return
        
        logger.info("🛑 Stopping system monitoring...")
        
        self._monitoring_active = False
        self._stop_event.set()
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=10)
        
        logger.info("✅ System monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active and not self._stop_event.is_set():
            try:
                start_time = time.time()
                
                # Collect all metrics
                self._collect_cpu_metrics()
                self._collect_memory_metrics()
                self._collect_disk_metrics()
                self._collect_network_metrics()
                self._collect_process_metrics()
                self._collect_application_metrics()
                
                # Calculate health score
                self._calculate_health_score()
                
                # Check thresholds and generate alerts
                self._check_thresholds()
                
                # Cleanup old data
                self._cleanup_historical_data()
                
                # Calculate sleep time to maintain interval
                elapsed = time.time() - start_time
                sleep_time = max(0, self.monitoring_interval - elapsed)
                
                if self._stop_event.wait(sleep_time):
                    break
                
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")
                if self._stop_event.wait(10):  # Wait 10 seconds before retry
                    break
    
    def _collect_cpu_metrics(self):
        """Collect CPU usage metrics."""
        try:
            # Current CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics_collector.set_gauge('system.cpu.usage_percent', cpu_percent)
            
            # Load averages (Unix only)
            if hasattr(os, 'getloadavg'):
                load_avg = os.getloadavg()
                self.metrics_collector.set_gauge('system.cpu.load_1m', load_avg[0])
                self.metrics_collector.set_gauge('system.cpu.load_5m', load_avg[1])
                self.metrics_collector.set_gauge('system.cpu.load_15m', load_avg[2])
            
            # Store in history
            self._cpu_history.append(cpu_percent)
            if len(self._cpu_history) > 1440:  # Keep 24 hours at 1-minute intervals
                self._cpu_history.pop(0)
                
        except Exception as e:
            logger.error(f"❌ CPU metrics collection error: {e}")
    
    def _collect_memory_metrics(self):
        """Collect memory usage metrics."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Memory metrics
            self.metrics_collector.set_gauge('system.memory.usage_percent', memory.percent)
            self.metrics_collector.set_gauge('system.memory.available_mb', memory.available / 1024 / 1024)
            self.metrics_collector.set_gauge('system.memory.used_mb', memory.used / 1024 / 1024)
            
            if hasattr(memory, 'cached'):
                self.metrics_collector.set_gauge('system.memory.cached_mb', memory.cached / 1024 / 1024)
            
            # Swap metrics
            self.metrics_collector.set_gauge('system.memory.swap_usage_percent', swap.percent)
            
            # Store in history
            self._memory_history.append(memory.percent)
            if len(self._memory_history) > 1440:
                self._memory_history.pop(0)
                
        except Exception as e:
            logger.error(f"❌ Memory metrics collection error: {e}")
    
    def _collect_disk_metrics(self):
        """Collect disk usage and I/O metrics."""
        try:
            # Disk usage for root partition
            disk_usage = psutil.disk_usage('/')
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            
            self.metrics_collector.set_gauge('system.disk.usage_percent', usage_percent)
            self.metrics_collector.set_gauge('system.disk.free_gb', disk_usage.free / 1024 / 1024 / 1024)
            self.metrics_collector.set_gauge('system.disk.used_gb', disk_usage.used / 1024 / 1024 / 1024)
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            if disk_io:
                # Calculate rates (simplified)
                current_time = time.time()
                if hasattr(self, '_last_disk_io') and hasattr(self, '_last_disk_time'):
                    time_delta = current_time - self._last_disk_time
                    if time_delta > 0:
                        read_rate = (disk_io.read_bytes - self._last_disk_io.read_bytes) / time_delta / 1024 / 1024  # MB/s
                        write_rate = (disk_io.write_bytes - self._last_disk_io.write_bytes) / time_delta / 1024 / 1024  # MB/s
                        
                        self.metrics_collector.set_gauge('system.disk.read_mb_per_sec', read_rate)
                        self.metrics_collector.set_gauge('system.disk.write_mb_per_sec', write_rate)
                
                self._last_disk_io = disk_io
                self._last_disk_time = current_time
                
        except Exception as e:
            logger.error(f"❌ Disk metrics collection error: {e}")
    
    def _collect_network_metrics(self):
        """Collect network statistics."""
        try:
            net_io = psutil.net_io_counters()
            if net_io:
                current_time = time.time()
                
                # Store current stats
                current_stats = NetworkStats(
                    bytes_sent=net_io.bytes_sent,
                    bytes_recv=net_io.bytes_recv,
                    packets_sent=net_io.packets_sent,
                    packets_recv=net_io.packets_recv,
                    errors_in=net_io.errin if hasattr(net_io, 'errin') else 0,
                    errors_out=net_io.errout if hasattr(net_io, 'errout') else 0,
                    drops_in=net_io.dropin if hasattr(net_io, 'dropin') else 0,
                    drops_out=net_io.dropout if hasattr(net_io, 'dropout') else 0,
                    timestamp=current_time
                )
                
                # Calculate rates if we have previous data
                if self._network_history:
                    last_stats = self._network_history[-1]
                    time_delta = current_time - last_stats.timestamp
                    
                    if time_delta > 0:
                        bytes_sent_rate = (current_stats.bytes_sent - last_stats.bytes_sent) / time_delta
                        bytes_recv_rate = (current_stats.bytes_recv - last_stats.bytes_recv) / time_delta
                        packets_sent_rate = (current_stats.packets_sent - last_stats.packets_sent) / time_delta
                        packets_recv_rate = (current_stats.packets_recv - last_stats.packets_recv) / time_delta
                        
                        # Error rate calculation
                        total_errors = (current_stats.errors_in + current_stats.errors_out) - (last_stats.errors_in + last_stats.errors_out)
                        total_packets = (current_stats.packets_sent + current_stats.packets_recv) - (last_stats.packets_sent + last_stats.packets_recv)
                        error_rate = (total_errors / total_packets * 100) if total_packets > 0 else 0
                        
                        # Update metrics
                        self.metrics_collector.set_gauge('system.network.bytes_sent_per_sec', bytes_sent_rate)
                        self.metrics_collector.set_gauge('system.network.bytes_recv_per_sec', bytes_recv_rate)
                        self.metrics_collector.set_gauge('system.network.packets_sent_per_sec', packets_sent_rate)
                        self.metrics_collector.set_gauge('system.network.packets_recv_per_sec', packets_recv_rate)
                        self.metrics_collector.set_gauge('system.network.error_rate', error_rate)
                
                # Store in history
                self._network_history.append(current_stats)
                if len(self._network_history) > 1440:  # Keep 24 hours
                    self._network_history.pop(0)
                    
        except Exception as e:
            logger.error(f"❌ Network metrics collection error: {e}")
    
    def _collect_process_metrics(self):
        """Collect process-specific metrics."""
        try:
            if not self._process or not self._process_info:
                return
            
            # Update process info
            with self._process.oneshot():
                self._process_info.cpu_percent = self._process.cpu_percent()
                self._process_info.memory_percent = self._process.memory_percent()
                
                memory_info = self._process.memory_info()
                self._process_info.memory_rss = memory_info.rss
                self._process_info.memory_vms = memory_info.vms
                
                self._process_info.num_threads = self._process.num_threads()
                
                # File descriptors (Unix only)
                try:
                    self._process_info.num_fds = self._process.num_fds()
                except (AttributeError, psutil.AccessDenied):
                    pass
                
                # Network connections
                try:
                    connections = self._process.connections()
                    self._process_info.connections = len(connections)
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass
            
            # Update metrics
            self.metrics_collector.set_gauge('process.cpu_percent', self._process_info.cpu_percent)
            self.metrics_collector.set_gauge('process.memory_percent', self._process_info.memory_percent)
            self.metrics_collector.set_gauge('process.memory_rss_mb', self._process_info.memory_rss / 1024 / 1024)
            self.metrics_collector.set_gauge('process.num_threads', self._process_info.num_threads)
            self.metrics_collector.set_gauge('process.num_connections', self._process_info.connections)
            
            if self._process_info.num_fds > 0:
                self.metrics_collector.set_gauge('process.num_file_descriptors', self._process_info.num_fds)
                
        except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
            logger.warning(f"⚠️  Process access error: {e}")
        except Exception as e:
            logger.error(f"❌ Process metrics collection error: {e}")
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            # WebSocket connections
            active_ws = len(self._websocket_connections)
            self.metrics_collector.set_gauge('app.websocket_connections', active_ws)
            
            # API call statistics
            current_time = time.time()
            api_calls_last_minute = 0
            total_response_times = []
            total_calls = 0
            error_calls = 0
            
            for endpoint, stats in self._api_call_stats.items():
                calls_in_window = [call for call in stats.get('calls', []) 
                                 if call['timestamp'] > current_time - 60]
                api_calls_last_minute += len(calls_in_window)
                
                for call in calls_in_window:
                    total_calls += 1
                    if call.get('response_time'):
                        total_response_times.append(call['response_time'])
                    if call.get('error'):
                        error_calls += 1
            
            # Update API metrics
            self.metrics_collector.set_gauge('app.api_calls_per_minute', api_calls_last_minute)
            
            if total_calls > 0:
                error_rate = (error_calls / total_calls) * 100
                self.metrics_collector.set_gauge('app.api_error_rate', error_rate)
            
            if total_response_times:
                avg_response_time = sum(total_response_times) / len(total_response_times)
                self.metrics_collector.set_gauge('app.avg_response_time_ms', avg_response_time)
            
            # System uptime
            if hasattr(self, 'system_info') and 'boot_time' in self.system_info:
                try:
                    boot_time = datetime.fromisoformat(self.system_info['boot_time'])
                    uptime_hours = (datetime.now() - boot_time).total_seconds() / 3600
                    self.metrics_collector.set_gauge('health.uptime_hours', uptime_hours)
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"❌ Application metrics collection error: {e}")
    
    def _calculate_health_score(self):
        """Calculate overall system health score."""
        try:
            score = 100.0  # Start with perfect score
            
            # CPU health impact
            cpu_usage = self.metrics_collector.get_latest_value('system.cpu.usage_percent') or 0
            if cpu_usage > self.thresholds.cpu_usage_critical:
                score -= 30
            elif cpu_usage > self.thresholds.cpu_usage_warning:
                score -= 15
            
            # Memory health impact
            memory_usage = self.metrics_collector.get_latest_value('system.memory.usage_percent') or 0
            if memory_usage > self.thresholds.memory_usage_critical:
                score -= 25
            elif memory_usage > self.thresholds.memory_usage_warning:
                score -= 10
            
            # Disk health impact
            disk_usage = self.metrics_collector.get_latest_value('system.disk.usage_percent') or 0
            if disk_usage > self.thresholds.disk_usage_critical:
                score -= 20
            elif disk_usage > self.thresholds.disk_usage_warning:
                score -= 10
            
            # Network health impact
            error_rate = self.metrics_collector.get_latest_value('system.network.error_rate') or 0
            if error_rate > self.thresholds.network_error_rate_critical:
                score -= 15
            elif error_rate > self.thresholds.network_error_rate_warning:
                score -= 8
            
            # API response time impact
            response_time = self.metrics_collector.get_latest_value('app.avg_response_time_ms') or 0
            if response_time > self.thresholds.response_time_critical:
                score -= 20
            elif response_time > self.thresholds.response_time_warning:
                score -= 10
            
            # Ensure score doesn't go below 0
            score = max(0, score)
            
            self.metrics_collector.set_gauge('health.overall_score', score)
            
        except Exception as e:
            logger.error(f"❌ Health score calculation error: {e}")
    
    def _check_thresholds(self):
        """Check thresholds and count active alerts."""
        try:
            active_alerts = 0
            
            # Check all threshold violations
            cpu_usage = self.metrics_collector.get_latest_value('system.cpu.usage_percent') or 0
            memory_usage = self.metrics_collector.get_latest_value('system.memory.usage_percent') or 0
            disk_usage = self.metrics_collector.get_latest_value('system.disk.usage_percent') or 0
            error_rate = self.metrics_collector.get_latest_value('system.network.error_rate') or 0
            response_time = self.metrics_collector.get_latest_value('app.avg_response_time_ms') or 0
            
            # Count critical alerts
            if cpu_usage > self.thresholds.cpu_usage_critical:
                active_alerts += 1
            if memory_usage > self.thresholds.memory_usage_critical:
                active_alerts += 1
            if disk_usage > self.thresholds.disk_usage_critical:
                active_alerts += 1
            if error_rate > self.thresholds.network_error_rate_critical:
                active_alerts += 1
            if response_time > self.thresholds.response_time_critical:
                active_alerts += 1
            
            # Count warning alerts
            if cpu_usage > self.thresholds.cpu_usage_warning:
                active_alerts += 1
            if memory_usage > self.thresholds.memory_usage_warning:
                active_alerts += 1
            if disk_usage > self.thresholds.disk_usage_warning:
                active_alerts += 1
            if error_rate > self.thresholds.network_error_rate_warning:
                active_alerts += 1
            if response_time > self.thresholds.response_time_warning:
                active_alerts += 1
            
            self.metrics_collector.set_gauge('health.alerts_active', active_alerts)
            
        except Exception as e:
            logger.error(f"❌ Threshold checking error: {e}")
    
    def _cleanup_historical_data(self):
        """Clean up old historical data."""
        # Keep only recent data to prevent memory growth
        max_history_points = 1440  # 24 hours at 1-minute intervals
        
        if len(self._cpu_history) > max_history_points:
            self._cpu_history = self._cpu_history[-max_history_points:]
        
        if len(self._memory_history) > max_history_points:
            self._memory_history = self._memory_history[-max_history_points:]
        
        if len(self._network_history) > max_history_points:
            self._network_history = self._network_history[-max_history_points:]
    
    def record_websocket_connection(self, connection_id: str, info: Dict[str, Any]):
        """Record WebSocket connection."""
        self._websocket_connections[connection_id] = {
            **info,
            'connected_at': time.time()
        }
        
        self.metrics_collector.set_gauge('app.websocket_connections', len(self._websocket_connections))
    
    def remove_websocket_connection(self, connection_id: str):
        """Remove WebSocket connection."""
        if connection_id in self._websocket_connections:
            connection_info = self._websocket_connections.pop(connection_id)
            
            # Record connection duration
            duration = time.time() - connection_info.get('connected_at', time.time())
            self.metrics_collector.record_timer('app.websocket_connection_duration', duration)
        
        self.metrics_collector.set_gauge('app.websocket_connections', len(self._websocket_connections))
    
    def record_api_call(self, endpoint: str, response_time: float, error: bool = False):
        """Record API call statistics."""
        if endpoint not in self._api_call_stats:
            self._api_call_stats[endpoint] = {'calls': []}
        
        call_info = {
            'timestamp': time.time(),
            'response_time': response_time,
            'error': error
        }
        
        self._api_call_stats[endpoint]['calls'].append(call_info)
        
        # Keep only last 1000 calls per endpoint
        if len(self._api_call_stats[endpoint]['calls']) > 1000:
            self._api_call_stats[endpoint]['calls'].pop(0)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            return {
                'timestamp': time.time(),
                'system_info': self.system_info,
                'health_score': self.metrics_collector.get_latest_value('health.overall_score'),
                'active_alerts': self.metrics_collector.get_latest_value('health.alerts_active'),
                'uptime_hours': self.metrics_collector.get_latest_value('health.uptime_hours'),
                'resources': {
                    'cpu_usage': self.metrics_collector.get_latest_value('system.cpu.usage_percent'),
                    'memory_usage': self.metrics_collector.get_latest_value('system.memory.usage_percent'),
                    'disk_usage': self.metrics_collector.get_latest_value('system.disk.usage_percent'),
                    'available_memory_mb': self.metrics_collector.get_latest_value('system.memory.available_mb'),
                    'free_disk_gb': self.metrics_collector.get_latest_value('system.disk.free_gb')
                },
                'network': {
                    'bytes_sent_per_sec': self.metrics_collector.get_latest_value('system.network.bytes_sent_per_sec'),
                    'bytes_recv_per_sec': self.metrics_collector.get_latest_value('system.network.bytes_recv_per_sec'),
                    'error_rate': self.metrics_collector.get_latest_value('system.network.error_rate')
                },
                'process': {
                    'cpu_percent': self.metrics_collector.get_latest_value('process.cpu_percent'),
                    'memory_percent': self.metrics_collector.get_latest_value('process.memory_percent'),
                    'memory_rss_mb': self.metrics_collector.get_latest_value('process.memory_rss_mb'),
                    'num_threads': self.metrics_collector.get_latest_value('process.num_threads'),
                    'num_connections': self.metrics_collector.get_latest_value('process.num_connections')
                },
                'application': {
                    'websocket_connections': self.metrics_collector.get_latest_value('app.websocket_connections'),
                    'api_calls_per_minute': self.metrics_collector.get_latest_value('app.api_calls_per_minute'),
                    'api_error_rate': self.metrics_collector.get_latest_value('app.api_error_rate'),
                    'avg_response_time_ms': self.metrics_collector.get_latest_value('app.avg_response_time_ms')
                }
            }
        except Exception as e:
            logger.error(f"❌ Failed to get system status: {e}")
            return {'error': str(e)}


def create_system_monitor(metrics_collector: Optional[MetricsCollector] = None) -> SystemMonitor:
    """Factory function to create system monitor."""
    return SystemMonitor(metrics_collector)