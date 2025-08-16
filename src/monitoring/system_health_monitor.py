#!/usr/bin/env python3
"""
System Health Monitor for DipMaster Trading System
系统健康监控器 - 综合系统健康状况监控

Features:
- API connection status monitoring
- Data feed latency tracking
- Order execution latency monitoring
- Memory and CPU usage tracking
- Network connectivity monitoring
- Database health checks
- Service dependency monitoring
- Health scoring and alerting
"""

import time
import logging
import asyncio
import psutil
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import socket
import requests
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """System component types."""
    API = "api"
    DATABASE = "database"
    WEBSOCKET = "websocket"
    NETWORK = "network"
    SYSTEM_RESOURCE = "system_resource"
    SERVICE = "service"
    DATA_FEED = "data_feed"


@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    component_type: ComponentType
    check_function: Callable
    check_interval: int  # seconds
    timeout: int  # seconds
    enabled: bool = True
    critical: bool = False  # If true, failure causes overall system to be unhealthy
    last_check_time: float = 0.0
    last_result: Optional[Dict[str, Any]] = None


@dataclass
class ComponentHealth:
    """Health status of a system component."""
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    last_check_time: float
    response_time_ms: float
    error_message: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    uptime_percentage: float = 100.0
    consecutive_failures: int = 0


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_available_gb: float
    network_io_bytes_sent: int
    network_io_bytes_recv: int
    active_connections: int
    load_average: List[float]  # 1, 5, 15 minute averages
    swap_usage_percent: float = 0.0


@dataclass
class LatencyMetrics:
    """Latency tracking for various operations."""
    api_latency_ms: float = 0.0
    database_latency_ms: float = 0.0
    websocket_latency_ms: float = 0.0
    order_execution_latency_ms: float = 0.0
    data_feed_latency_ms: float = 0.0


class SystemHealthMonitor:
    """
    Comprehensive system health monitoring.
    
    Monitors all critical system components and provides health scoring,
    alerting, and detailed diagnostics for trading system operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize system health monitor.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Health checks registry
        self.health_checks = {}
        self.component_health = {}
        
        # Metrics tracking
        self.system_metrics_history = deque(maxlen=1000)
        self.latency_metrics_history = deque(maxlen=1000)
        
        # Health scoring
        self.overall_health_score = 100.0
        self.health_score_history = deque(maxlen=1000)
        
        # Monitoring control
        self._monitoring_active = False
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        
        # Alert callbacks
        self.health_alert_callbacks = []
        
        # Service dependencies
        self.service_dependencies = {}
        
        # Network connectivity
        self.connectivity_targets = self.config.get('connectivity_targets', [
            'google.com',
            'api.binance.com',
            'cloudflare.com'
        ])
        
        # Initialize default health checks
        self._setup_default_health_checks()
        
        logger.info("🏥 SystemHealthMonitor initialized")
    
    def _setup_default_health_checks(self):
        """Setup default health checks for common components."""
        
        # System resource checks
        self.add_health_check(HealthCheck(
            name="cpu_usage",
            component_type=ComponentType.SYSTEM_RESOURCE,
            check_function=self._check_cpu_usage,
            check_interval=30,
            timeout=5,
            critical=True
        ))
        
        self.add_health_check(HealthCheck(
            name="memory_usage",
            component_type=ComponentType.SYSTEM_RESOURCE,
            check_function=self._check_memory_usage,
            check_interval=30,
            timeout=5,
            critical=True
        ))
        
        self.add_health_check(HealthCheck(
            name="disk_space",
            component_type=ComponentType.SYSTEM_RESOURCE,
            check_function=self._check_disk_space,
            check_interval=60,
            timeout=5,
            critical=False
        ))
        
        # Network connectivity check
        self.add_health_check(HealthCheck(
            name="network_connectivity",
            component_type=ComponentType.NETWORK,
            check_function=self._check_network_connectivity,
            check_interval=60,
            timeout=10,
            critical=True
        ))
        
        # API availability check (example for Binance)
        self.add_health_check(HealthCheck(
            name="binance_api",
            component_type=ComponentType.API,
            check_function=self._check_binance_api,
            check_interval=30,
            timeout=10,
            critical=True
        ))
        
        logger.info(f"📋 Setup {len(self.health_checks)} default health checks")
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check to the monitoring system."""
        self.health_checks[health_check.name] = health_check
        logger.info(f"➕ Added health check: {health_check.name}")
    
    def remove_health_check(self, check_name: str):
        """Remove a health check from monitoring."""
        if check_name in self.health_checks:
            del self.health_checks[check_name]
            if check_name in self.component_health:
                del self.component_health[check_name]
            logger.info(f"➖ Removed health check: {check_name}")
    
    def add_service_dependency(self, service_name: str, dependencies: List[str]):
        """Add service dependency mapping."""
        self.service_dependencies[service_name] = dependencies
        logger.info(f"🔗 Added dependencies for {service_name}: {dependencies}")
    
    def add_health_alert_callback(self, callback: Callable[[str, ComponentHealth], None]):
        """Add callback for health alerts."""
        self.health_alert_callbacks.append(callback)
        logger.info("📞 Added health alert callback")
    
    async def start_monitoring(self):
        """Start health monitoring."""
        if self._monitoring_active:
            logger.warning("⚠️ Health monitoring already active")
            return
        
        try:
            self._monitoring_active = True
            self._stop_event.clear()
            
            # Start monitoring thread
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()
            
            logger.info("🚀 Health monitoring started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start health monitoring: {e}")
            self._monitoring_active = False
            raise
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        if not self._monitoring_active:
            logger.warning("⚠️ Health monitoring not active")
            return
        
        try:
            self._monitoring_active = False
            self._stop_event.set()
            
            # Wait for monitoring thread
            if self._monitoring_thread and self._monitoring_thread.is_alive():
                self._monitoring_thread.join(timeout=10)
            
            logger.info("🛑 Health monitoring stopped")
            
        except Exception as e:
            logger.error(f"❌ Error stopping health monitoring: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop (runs in separate thread)."""
        logger.info("🔄 Health monitoring loop started")
        
        while not self._stop_event.is_set():
            try:
                current_time = time.time()
                
                # Run health checks
                for check_name, health_check in self.health_checks.items():
                    if not health_check.enabled:
                        continue
                    
                    # Check if it's time to run this check
                    if current_time - health_check.last_check_time >= health_check.check_interval:
                        self._run_health_check(health_check)
                
                # Collect system metrics
                self._collect_system_metrics()
                
                # Calculate overall health score
                self._calculate_health_score()
                
                # Sleep for a short interval
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                time.sleep(10)  # Longer sleep on error
    
    def _run_health_check(self, health_check: HealthCheck):
        """Run an individual health check."""
        try:
            start_time = time.time()
            health_check.last_check_time = start_time
            
            # Run the check function with timeout
            result = self._run_with_timeout(health_check.check_function, health_check.timeout)
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            # Process result
            if result and result.get('status') == 'healthy':
                status = HealthStatus.HEALTHY
                error_message = None
                consecutive_failures = 0
            elif result and result.get('status') == 'degraded':
                status = HealthStatus.DEGRADED
                error_message = result.get('error')
                consecutive_failures = self.component_health.get(health_check.name, ComponentHealth(
                    health_check.name, health_check.component_type, HealthStatus.UNKNOWN, 0, 0
                )).consecutive_failures
            else:
                status = HealthStatus.UNHEALTHY
                error_message = result.get('error', 'Check failed') if result else 'Check timeout'
                consecutive_failures = self.component_health.get(health_check.name, ComponentHealth(
                    health_check.name, health_check.component_type, HealthStatus.UNKNOWN, 0, 0
                )).consecutive_failures + 1
            
            # Update component health
            component_health = ComponentHealth(
                component_name=health_check.name,
                component_type=health_check.component_type,
                status=status,
                last_check_time=start_time,
                response_time_ms=response_time_ms,
                error_message=error_message,
                details=result.get('details', {}) if result else {},
                consecutive_failures=consecutive_failures
            )
            
            # Calculate uptime percentage
            component_health.uptime_percentage = self._calculate_uptime(health_check.name, status)
            
            self.component_health[health_check.name] = component_health
            health_check.last_result = result
            
            # Trigger alerts if needed
            if status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                self._trigger_health_alert(health_check.name, component_health)
            
            logger.debug(f"🔍 Health check {health_check.name}: {status.value} "
                        f"({response_time_ms:.1f}ms)")
            
        except Exception as e:
            logger.error(f"❌ Error running health check {health_check.name}: {e}")
            
            # Mark as unhealthy on exception
            self.component_health[health_check.name] = ComponentHealth(
                component_name=health_check.name,
                component_type=health_check.component_type,
                status=HealthStatus.CRITICAL,
                last_check_time=time.time(),
                response_time_ms=0,
                error_message=str(e),
                consecutive_failures=self.component_health.get(health_check.name, ComponentHealth(
                    health_check.name, health_check.component_type, HealthStatus.UNKNOWN, 0, 0
                )).consecutive_failures + 1
            )
    
    def _run_with_timeout(self, func: Callable, timeout: int) -> Optional[Dict[str, Any]]:
        """Run function with timeout."""
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Function timed out")
            
            # Set timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
            
            try:
                result = func()
                signal.alarm(0)  # Cancel timeout
                return result
            except TimeoutError:
                return {'status': 'unhealthy', 'error': 'Timeout'}
            
        except Exception as e:
            try:
                signal.alarm(0)  # Cancel timeout
            except:
                pass
            return {'status': 'unhealthy', 'error': str(e)}
    
    def _calculate_uptime(self, component_name: str, current_status: HealthStatus) -> float:
        """Calculate component uptime percentage."""
        # Simple implementation - in production would track over longer periods
        if component_name not in self.component_health:
            return 100.0 if current_status == HealthStatus.HEALTHY else 0.0
        
        # For now, return simplified uptime based on recent status
        consecutive_failures = self.component_health[component_name].consecutive_failures
        if consecutive_failures == 0:
            return 100.0
        elif consecutive_failures < 3:
            return 95.0
        elif consecutive_failures < 10:
            return 85.0
        else:
            return 70.0
    
    def _trigger_health_alert(self, component_name: str, component_health: ComponentHealth):
        """Trigger health alert callbacks."""
        try:
            for callback in self.health_alert_callbacks:
                callback(component_name, component_health)
        except Exception as e:
            logger.error(f"❌ Error triggering health alert: {e}")
    
    def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_available_gb = disk.free / (1024 * 1024 * 1024)
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Connection count
            connections = len(psutil.net_connections())
            
            # Load average (Unix systems)
            try:
                load_avg = list(psutil.getloadavg())
            except AttributeError:
                load_avg = [0.0, 0.0, 0.0]  # Windows doesn't have load average
            
            # Swap usage
            swap = psutil.swap_memory()
            swap_percent = swap.percent
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory_percent,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_percent,
                disk_available_gb=disk_available_gb,
                network_io_bytes_sent=network_bytes_sent,
                network_io_bytes_recv=network_bytes_recv,
                active_connections=connections,
                load_average=load_avg,
                swap_usage_percent=swap_percent
            )
            
            self.system_metrics_history.append(metrics)
            
        except Exception as e:
            logger.error(f"❌ Error collecting system metrics: {e}")
    
    def _calculate_health_score(self):
        """Calculate overall system health score."""
        try:
            if not self.component_health:
                self.overall_health_score = 100.0
                return
            
            total_score = 0.0
            total_weight = 0.0
            
            for component_name, health in self.component_health.items():
                # Get weight based on criticality
                health_check = self.health_checks.get(component_name)
                weight = 2.0 if health_check and health_check.critical else 1.0
                
                # Score based on status
                if health.status == HealthStatus.HEALTHY:
                    score = 100.0
                elif health.status == HealthStatus.DEGRADED:
                    score = 70.0
                elif health.status == HealthStatus.UNHEALTHY:
                    score = 30.0
                elif health.status == HealthStatus.CRITICAL:
                    score = 0.0
                else:
                    score = 50.0  # Unknown
                
                # Adjust score based on consecutive failures
                if health.consecutive_failures > 0:
                    failure_penalty = min(health.consecutive_failures * 10, 50)
                    score = max(score - failure_penalty, 0)
                
                total_score += score * weight
                total_weight += weight
            
            self.overall_health_score = total_score / total_weight if total_weight > 0 else 100.0
            
            # Store in history
            self.health_score_history.append({
                'timestamp': time.time(),
                'score': self.overall_health_score
            })
            
        except Exception as e:
            logger.error(f"❌ Error calculating health score: {e}")
            self.overall_health_score = 50.0  # Default to degraded on error
    
    # Default health check implementations
    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage levels."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent < 80:
                status = 'healthy'
            elif cpu_percent < 95:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            return {
                'status': status,
                'details': {
                    'cpu_percent': cpu_percent,
                    'cpu_count': psutil.cpu_count()
                }
            }
            
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage levels."""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent < 80:
                status = 'healthy'
            elif memory.percent < 95:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            return {
                'status': status,
                'details': {
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'memory_total_gb': memory.total / (1024**3)
                }
            }
            
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability."""
        try:
            disk = psutil.disk_usage('/')
            
            if disk.percent < 85:
                status = 'healthy'
            elif disk.percent < 95:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            return {
                'status': status,
                'details': {
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free / (1024**3),
                    'disk_total_gb': disk.total / (1024**3)
                }
            }
            
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def _check_network_connectivity(self) -> Dict[str, Any]:
        """Check network connectivity to external services."""
        try:
            successful_pings = 0
            total_targets = len(self.connectivity_targets)
            
            for target in self.connectivity_targets:
                try:
                    # Simple socket connection test
                    sock = socket.create_connection((target, 80), timeout=5)
                    sock.close()
                    successful_pings += 1
                except:
                    pass
            
            connectivity_ratio = successful_pings / total_targets if total_targets > 0 else 1.0
            
            if connectivity_ratio >= 0.8:
                status = 'healthy'
            elif connectivity_ratio >= 0.5:
                status = 'degraded'
            else:
                status = 'unhealthy'
            
            return {
                'status': status,
                'details': {
                    'successful_connections': successful_pings,
                    'total_targets': total_targets,
                    'connectivity_ratio': connectivity_ratio
                }
            }
            
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def _check_binance_api(self) -> Dict[str, Any]:
        """Check Binance API availability."""
        try:
            start_time = time.time()
            response = requests.get('https://api.binance.com/api/v3/ping', timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                if response_time < 1000:
                    status = 'healthy'
                elif response_time < 3000:
                    status = 'degraded'
                else:
                    status = 'unhealthy'
            else:
                status = 'unhealthy'
            
            return {
                'status': status,
                'details': {
                    'response_time_ms': response_time,
                    'status_code': response.status_code,
                    'endpoint': 'https://api.binance.com/api/v3/ping'
                }
            }
            
        except Exception as e:
            return {'status': 'unhealthy', 'error': str(e)}
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        try:
            # Component status summary
            component_status_counts = defaultdict(int)
            critical_issues = []
            degraded_components = []
            
            for name, health in self.component_health.items():
                component_status_counts[health.status.value] += 1
                
                if health.status == HealthStatus.CRITICAL:
                    critical_issues.append({
                        'component': name,
                        'error': health.error_message,
                        'consecutive_failures': health.consecutive_failures
                    })
                elif health.status == HealthStatus.DEGRADED:
                    degraded_components.append({
                        'component': name,
                        'response_time_ms': health.response_time_ms
                    })
            
            # Latest system metrics
            latest_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
            
            # Overall status
            if self.overall_health_score >= 90:
                overall_status = HealthStatus.HEALTHY
            elif self.overall_health_score >= 70:
                overall_status = HealthStatus.DEGRADED
            elif self.overall_health_score >= 30:
                overall_status = HealthStatus.UNHEALTHY
            else:
                overall_status = HealthStatus.CRITICAL
            
            return {
                'timestamp': time.time(),
                'overall_status': overall_status.value,
                'overall_score': self.overall_health_score,
                'component_summary': {
                    'total_components': len(self.component_health),
                    'status_breakdown': dict(component_status_counts),
                    'critical_issues': critical_issues,
                    'degraded_components': degraded_components
                },
                'system_resources': {
                    'cpu_usage_percent': latest_metrics.cpu_usage_percent if latest_metrics else 0,
                    'memory_usage_percent': latest_metrics.memory_usage_percent if latest_metrics else 0,
                    'disk_usage_percent': latest_metrics.disk_usage_percent if latest_metrics else 0,
                    'active_connections': latest_metrics.active_connections if latest_metrics else 0,
                    'load_average': latest_metrics.load_average if latest_metrics else [0, 0, 0]
                },
                'monitoring_status': {
                    'active': self._monitoring_active,
                    'checks_configured': len(self.health_checks),
                    'checks_enabled': len([c for c in self.health_checks.values() if c.enabled])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get health summary: {e}")
            return {
                'error': str(e),
                'overall_status': 'unknown',
                'overall_score': 0
            }
    
    def get_component_details(self, component_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific component."""
        try:
            if component_name not in self.component_health:
                return None
            
            health = self.component_health[component_name]
            health_check = self.health_checks.get(component_name)
            
            return {
                'component_name': component_name,
                'component_type': health.component_type.value,
                'status': health.status.value,
                'last_check_time': health.last_check_time,
                'response_time_ms': health.response_time_ms,
                'uptime_percentage': health.uptime_percentage,
                'consecutive_failures': health.consecutive_failures,
                'error_message': health.error_message,
                'details': health.details,
                'check_configuration': {
                    'enabled': health_check.enabled if health_check else False,
                    'critical': health_check.critical if health_check else False,
                    'check_interval': health_check.check_interval if health_check else 0,
                    'timeout': health_check.timeout if health_check else 0
                } if health_check else None
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting component details: {e}")
            return None
    
    def export_health_report(self) -> Dict[str, Any]:
        """Export comprehensive health report."""
        try:
            # Component details
            component_details = {}
            for name in self.component_health:
                component_details[name] = self.get_component_details(name)
            
            # System metrics trend (last hour)
            recent_metrics = [m for m in self.system_metrics_history 
                            if time.time() - m.timestamp <= 3600]
            
            # Health score trend
            recent_scores = [s for s in self.health_score_history 
                           if time.time() - s['timestamp'] <= 3600]
            
            return {
                'timestamp': time.time(),
                'report_type': 'system_health',
                'summary': self.get_health_summary(),
                'component_details': component_details,
                'system_metrics_trend': {
                    'data_points': len(recent_metrics),
                    'avg_cpu_usage': statistics.mean([m.cpu_usage_percent for m in recent_metrics]) if recent_metrics else 0,
                    'avg_memory_usage': statistics.mean([m.memory_usage_percent for m in recent_metrics]) if recent_metrics else 0,
                    'peak_cpu_usage': max([m.cpu_usage_percent for m in recent_metrics], default=0),
                    'peak_memory_usage': max([m.memory_usage_percent for m in recent_metrics], default=0)
                },
                'health_score_trend': {
                    'data_points': len(recent_scores),
                    'current_score': self.overall_health_score,
                    'avg_score': statistics.mean([s['score'] for s in recent_scores]) if recent_scores else 0,
                    'min_score': min([s['score'] for s in recent_scores], default=0)
                },
                'service_dependencies': self.service_dependencies,
                'recommendations': self._generate_health_recommendations()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to export health report: {e}")
            return {'error': str(e)}
    
    def _generate_health_recommendations(self) -> List[str]:
        """Generate health improvement recommendations."""
        recommendations = []
        
        try:
            # Check for critical issues
            critical_components = [name for name, health in self.component_health.items() 
                                 if health.status == HealthStatus.CRITICAL]
            
            if critical_components:
                recommendations.append(f"CRITICAL: Investigate failed components: {', '.join(critical_components)}")
            
            # Check system resources
            if self.system_metrics_history:
                latest = self.system_metrics_history[-1]
                
                if latest.cpu_usage_percent > 90:
                    recommendations.append("High CPU usage detected: Consider scaling or optimizing processes")
                
                if latest.memory_usage_percent > 90:
                    recommendations.append("High memory usage: Review memory leaks and consider increasing capacity")
                
                if latest.disk_usage_percent > 90:
                    recommendations.append("Low disk space: Clean up logs and temporary files")
                
                if latest.active_connections > 1000:
                    recommendations.append("High connection count: Monitor for connection leaks")
            
            # Check degraded components
            degraded_components = [name for name, health in self.component_health.items() 
                                 if health.status == HealthStatus.DEGRADED]
            
            if degraded_components:
                recommendations.append(f"Monitor degraded components: {', '.join(degraded_components)}")
            
            # Check consecutive failures
            failing_components = [name for name, health in self.component_health.items() 
                                if health.consecutive_failures > 3]
            
            if failing_components:
                recommendations.append(f"Investigate persistent failures: {', '.join(failing_components)}")
            
            # Overall health score
            if self.overall_health_score < 70:
                recommendations.append("Overall system health is degraded: Review all components and dependencies")
            
            if not recommendations:
                recommendations.append("System health is good: Continue monitoring")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            return ["Error generating recommendations"]


# Factory function
def create_system_health_monitor(config: Dict[str, Any]) -> SystemHealthMonitor:
    """Create and configure system health monitor."""
    return SystemHealthMonitor(config=config.get('health_config', {}))