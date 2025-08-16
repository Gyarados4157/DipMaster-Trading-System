#!/usr/bin/env python3
"""
Metrics Collector for DipMaster Trading System
指标收集器 - 实时收集和聚合系统指标

Features:
- Real-time metrics collection
- Multiple metric types (counter, gauge, histogram, timer)
- Efficient storage and aggregation
- Thread-safe operations
- Configurable retention policies
- Export capabilities for external systems
"""

import time
import threading
from collections import defaultdict, deque
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from enum import Enum
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics supported."""
    COUNTER = "counter"      # Monotonically increasing values
    GAUGE = "gauge"         # Point-in-time values
    HISTOGRAM = "histogram" # Distribution of values
    TIMER = "timer"        # Duration measurements
    RATE = "rate"          # Rate calculations


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'timestamp': self.timestamp,
            'value': self.value,
            'tags': self.tags
        }


@dataclass
class MetricSeries:
    """Time series data for a metric."""
    name: str
    metric_type: MetricType
    description: str = ""
    unit: str = ""
    points: deque = field(default_factory=lambda: deque(maxlen=10000))
    tags: Dict[str, str] = field(default_factory=dict)
    
    def add_point(self, value: Union[int, float], tags: Optional[Dict[str, str]] = None):
        """Add a new data point."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            tags=tags or {}
        )
        self.points.append(point)
    
    def get_latest_value(self) -> Optional[Union[int, float]]:
        """Get the most recent value."""
        if self.points:
            return self.points[-1].value
        return None
    
    def get_values_since(self, since: float) -> List[MetricPoint]:
        """Get all values since a specific timestamp."""
        return [point for point in self.points if point.timestamp >= since]
    
    def calculate_rate(self, window_seconds: int = 60) -> float:
        """Calculate rate of change over time window."""
        if len(self.points) < 2:
            return 0.0
        
        now = time.time()
        recent_points = [p for p in self.points if p.timestamp >= now - window_seconds]
        
        if len(recent_points) < 2:
            return 0.0
        
        # For counters, calculate rate of increase
        if self.metric_type == MetricType.COUNTER:
            start_value = recent_points[0].value
            end_value = recent_points[-1].value
            time_diff = recent_points[-1].timestamp - recent_points[0].timestamp
            
            if time_diff > 0:
                return (end_value - start_value) / time_diff
        
        return 0.0
    
    def calculate_percentiles(self, percentiles: List[float] = None) -> Dict[float, float]:
        """Calculate percentiles for the metric values."""
        if not self.points:
            return {}
        
        percentiles = percentiles or [50.0, 90.0, 95.0, 99.0]
        values = [p.value for p in self.points]
        
        result = {}
        for p in percentiles:
            try:
                result[p] = statistics.quantiles(values, n=100)[int(p)-1]
            except (statistics.StatisticsError, IndexError):
                result[p] = 0.0
        
        return result


class MetricsCollector:
    """
    Thread-safe metrics collection system.
    
    Collects, stores, and provides access to various types of metrics
    with support for real-time queries and historical analysis.
    """
    
    def __init__(self, 
                 retention_hours: int = 24,
                 cleanup_interval: int = 300):  # 5 minutes
        """
        Initialize metrics collector.
        
        Args:
            retention_hours: Hours to retain metric data
            cleanup_interval: Seconds between cleanup operations
        """
        self.retention_hours = retention_hours
        self.cleanup_interval = cleanup_interval
        
        # Thread-safe storage
        self._lock = threading.RLock()
        self._metrics: Dict[str, MetricSeries] = {}
        
        # Aggregation functions for different metric types
        self._aggregation_functions: Dict[str, List[Callable]] = defaultdict(list)
        
        # Background cleanup
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        # Start cleanup thread
        self._start_cleanup_thread()
        
        logger.info("📊 MetricsCollector initialized")
    
    def _start_cleanup_thread(self):
        """Start background thread for data cleanup."""
        def cleanup_worker():
            while not self._stop_cleanup.wait(self.cleanup_interval):
                try:
                    self._cleanup_old_data()
                except Exception as e:
                    logger.error(f"❌ Metrics cleanup error: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_old_data(self):
        """Remove old metric data points."""
        cutoff_time = time.time() - (self.retention_hours * 3600)
        
        with self._lock:
            for metric_name, series in self._metrics.items():
                # Remove old points
                while (series.points and 
                       series.points[0].timestamp < cutoff_time):
                    series.points.popleft()
    
    def register_metric(self,
                       name: str,
                       metric_type: MetricType,
                       description: str = "",
                       unit: str = "",
                       tags: Optional[Dict[str, str]] = None) -> None:
        """
        Register a new metric.
        
        Args:
            name: Metric name (must be unique)
            metric_type: Type of metric
            description: Human-readable description
            unit: Unit of measurement
            tags: Default tags for this metric
        """
        with self._lock:
            if name in self._metrics:
                logger.warning(f"⚠️  Metric {name} already registered")
                return
            
            self._metrics[name] = MetricSeries(
                name=name,
                metric_type=metric_type,
                description=description,
                unit=unit,
                tags=tags or {}
            )
            
            logger.debug(f"📈 Registered metric: {name} ({metric_type.value})")
    
    def record_value(self,
                    name: str,
                    value: Union[int, float],
                    tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Additional tags for this data point
        """
        with self._lock:
            if name not in self._metrics:
                # Auto-register as gauge if not exists
                self.register_metric(name, MetricType.GAUGE)
            
            self._metrics[name].add_point(value, tags)
    
    def increment_counter(self,
                         name: str,
                         value: Union[int, float] = 1,
                         tags: Optional[Dict[str, str]] = None) -> None:
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            value: Value to add (default: 1)
            tags: Additional tags
        """
        with self._lock:
            if name not in self._metrics:
                self.register_metric(name, MetricType.COUNTER, 
                                   description=f"Counter: {name}")
            
            current_value = self._metrics[name].get_latest_value() or 0
            self._metrics[name].add_point(current_value + value, tags)
    
    def set_gauge(self,
                 name: str,
                 value: Union[int, float],
                 tags: Optional[Dict[str, str]] = None) -> None:
        """
        Set a gauge metric value.
        
        Args:
            name: Gauge name
            value: Current value
            tags: Additional tags
        """
        with self._lock:
            if name not in self._metrics:
                self.register_metric(name, MetricType.GAUGE,
                                   description=f"Gauge: {name}")
            
            self._metrics[name].add_point(value, tags)
    
    def record_timer(self,
                    name: str,
                    duration_seconds: float,
                    tags: Optional[Dict[str, str]] = None) -> None:
        """
        Record a timer metric.
        
        Args:
            name: Timer name
            duration_seconds: Duration in seconds
            tags: Additional tags
        """
        with self._lock:
            if name not in self._metrics:
                self.register_metric(name, MetricType.TIMER,
                                   description=f"Timer: {name}",
                                   unit="seconds")
            
            self._metrics[name].add_point(duration_seconds, tags)
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """Get a metric series by name."""
        with self._lock:
            return self._metrics.get(name)
    
    def get_latest_value(self, name: str) -> Optional[Union[int, float]]:
        """Get the latest value for a metric."""
        series = self.get_metric(name)
        return series.get_latest_value() if series else None
    
    def get_metric_names(self) -> List[str]:
        """Get list of all registered metric names."""
        with self._lock:
            return list(self._metrics.keys())
    
    def get_metrics_summary(self, 
                           window_seconds: int = 300) -> Dict[str, Any]:
        """
        Get summary statistics for all metrics.
        
        Args:
            window_seconds: Time window for calculations
            
        Returns:
            Dictionary with metric summaries
        """
        summary = {}
        now = time.time()
        
        with self._lock:
            for name, series in self._metrics.items():
                recent_points = series.get_values_since(now - window_seconds)
                
                if not recent_points:
                    continue
                
                values = [p.value for p in recent_points]
                
                metric_summary = {
                    'name': name,
                    'type': series.metric_type.value,
                    'description': series.description,
                    'unit': series.unit,
                    'latest_value': series.get_latest_value(),
                    'count': len(values),
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0,
                    'avg': statistics.mean(values) if values else 0,
                    'rate': series.calculate_rate(window_seconds),
                    'percentiles': series.calculate_percentiles()
                }
                
                summary[name] = metric_summary
        
        return summary
    
    def export_metrics(self, 
                      format_type: str = "json",
                      start_time: Optional[float] = None,
                      end_time: Optional[float] = None) -> str:
        """
        Export metrics in specified format.
        
        Args:
            format_type: Export format (json, prometheus)
            start_time: Start time filter
            end_time: End time filter
            
        Returns:
            Formatted metrics string
        """
        if format_type.lower() == "json":
            return self._export_json(start_time, end_time)
        elif format_type.lower() == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_json(self, 
                    start_time: Optional[float] = None,
                    end_time: Optional[float] = None) -> str:
        """Export metrics in JSON format."""
        export_data = {
            'timestamp': time.time(),
            'retention_hours': self.retention_hours,
            'metrics': {}
        }
        
        with self._lock:
            for name, series in self._metrics.items():
                points = series.points
                
                # Apply time filters if specified
                if start_time or end_time:
                    points = [p for p in points
                             if (not start_time or p.timestamp >= start_time) and
                                (not end_time or p.timestamp <= end_time)]
                
                export_data['metrics'][name] = {
                    'type': series.metric_type.value,
                    'description': series.description,
                    'unit': series.unit,
                    'tags': series.tags,
                    'points': [p.to_dict() for p in points]
                }
        
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            for name, series in self._metrics.items():
                latest_value = series.get_latest_value()
                if latest_value is None:
                    continue
                
                # Prometheus metric name (replace invalid chars)
                prom_name = name.replace('-', '_').replace('.', '_')
                
                # Add HELP and TYPE lines
                if series.description:
                    lines.append(f"# HELP {prom_name} {series.description}")
                
                metric_type = "gauge"  # Default for Prometheus
                if series.metric_type == MetricType.COUNTER:
                    metric_type = "counter"
                
                lines.append(f"# TYPE {prom_name} {metric_type}")
                
                # Add metric value with tags
                tag_str = ""
                if series.tags:
                    tag_pairs = [f'{k}="{v}"' for k, v in series.tags.items()]
                    tag_str = f"{{{','.join(tag_pairs)}}}"
                
                lines.append(f"{prom_name}{tag_str} {latest_value}")
        
        return "\n".join(lines)
    
    def create_timer_context(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Create a context manager for timing operations."""
        return TimerContext(self, name, tags)
    
    def shutdown(self):
        """Shutdown the metrics collector."""
        logger.info("🛑 Shutting down metrics collector...")
        
        # Stop cleanup thread
        self._stop_cleanup.set()
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        logger.info("✅ Metrics collector shutdown complete")


class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.name, duration, self.tags)


# Global metrics collector instance
_global_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = MetricsCollector()
    return _global_collector


def set_metrics_collector(collector: MetricsCollector):
    """Set global metrics collector instance."""
    global _global_collector
    _global_collector = collector


# Convenience functions for common operations
def record_metric(name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None):
    """Record a metric value using global collector."""
    get_metrics_collector().record_value(name, value, tags)


def increment(name: str, value: Union[int, float] = 1, tags: Optional[Dict[str, str]] = None):
    """Increment a counter using global collector."""
    get_metrics_collector().increment_counter(name, value, tags)


def set_gauge(name: str, value: Union[int, float], tags: Optional[Dict[str, str]] = None):
    """Set a gauge value using global collector."""
    get_metrics_collector().set_gauge(name, value, tags)


def timer(name: str, tags: Optional[Dict[str, str]] = None):
    """Create a timer context using global collector."""
    return get_metrics_collector().create_timer_context(name, tags)