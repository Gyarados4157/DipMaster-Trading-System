#!/usr/bin/env python3
"""
Structured Logger for DipMaster Trading System
结构化日志管理器 - 专业交易系统日志管理

Features:
- Structured JSON logging with consistent schema
- Log rotation and compression
- Error tracking and debugging support
- Audit logging for compliance
- Performance logging and profiling
- Correlation ID tracking
- Multiple output formats and destinations
- Log aggregation and searching
"""

import time
import logging
import json
import os
import gzip
import threading
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime, timezone
from pathlib import Path
import uuid
import traceback
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import asyncio
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Log level enumeration."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    AUDIT = "AUDIT"
    PERFORMANCE = "PERFORMANCE"


class LogCategory(Enum):
    """Log category enumeration."""
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    EXECUTION = "execution"
    DATA = "data"
    API = "api"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MONITORING = "monitoring"
    COMPLIANCE = "compliance"


@dataclass
class LogContext:
    """Log context information."""
    correlation_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    component: Optional[str] = None
    environment: str = "production"


@dataclass
class StructuredLogEntry:
    """Structured log entry schema."""
    timestamp: str
    level: str
    category: str
    message: str
    component: str
    correlation_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    environment: str = "production"
    
    # Technical details
    file_name: Optional[str] = None
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    thread_id: Optional[int] = None
    process_id: Optional[int] = None
    
    # Context data
    data: Dict[str, Any] = None
    error: Optional[Dict[str, Any]] = None
    duration_ms: Optional[float] = None
    
    # Trading specific
    symbol: Optional[str] = None
    trade_id: Optional[str] = None
    order_id: Optional[str] = None
    strategy: Optional[str] = None
    
    # Performance metrics
    cpu_usage: Optional[float] = None
    memory_usage: Optional[float] = None
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.data is None:
            self.data = {}


class PerformanceTimer:
    """Context manager for performance timing."""
    
    def __init__(self, logger_instance, operation_name: str, context: LogContext):
        self.logger = logger_instance
        self.operation_name = operation_name
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log_performance(
            f"Started {self.operation_name}",
            self.context,
            data={'operation': self.operation_name, 'stage': 'start'}
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.time() - self.start_time) * 1000 if self.start_time else 0
        
        if exc_type is None:
            self.logger.log_performance(
                f"Completed {self.operation_name}",
                self.context,
                duration_ms=duration_ms,
                data={'operation': self.operation_name, 'stage': 'complete', 'success': True}
            )
        else:
            self.logger.log_error(
                f"Failed {self.operation_name}: {exc_val}",
                self.context,
                duration_ms=duration_ms,
                data={'operation': self.operation_name, 'stage': 'error'},
                exception=exc_val
            )


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        try:
            # Extract structured data if available
            if hasattr(record, 'structured_data'):
                return json.dumps(record.structured_data, ensure_ascii=False, separators=(',', ':'))
            else:
                # Fallback for standard log records
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                    'level': record.levelname,
                    'category': 'system',
                    'message': record.getMessage(),
                    'component': record.name,
                    'correlation_id': str(uuid.uuid4()),
                    'file_name': record.filename,
                    'line_number': record.lineno,
                    'function_name': record.funcName,
                    'thread_id': record.thread,
                    'process_id': record.process
                }
                return json.dumps(log_entry, ensure_ascii=False, separators=(',', ':'))
        except Exception as e:
            # Fallback to standard formatting
            return f'{{"timestamp": "{datetime.now(timezone.utc).isoformat()}", "level": "ERROR", "message": "Log formatting error: {str(e)}", "original_message": "{record.getMessage()}"}}'


class LogAggregator:
    """Aggregate and analyze log entries."""
    
    def __init__(self, retention_hours: int = 24):
        self.retention_hours = retention_hours
        self.log_entries = deque(maxlen=100000)
        self.error_counts = defaultdict(int)
        self.category_counts = defaultdict(int)
        self.component_counts = defaultdict(int)
        self.level_counts = defaultdict(int)
    
    def add_log_entry(self, log_entry: StructuredLogEntry):
        """Add log entry to aggregation."""
        try:
            # Store entry
            self.log_entries.append(log_entry)
            
            # Update counters
            self.level_counts[log_entry.level] += 1
            self.category_counts[log_entry.category] += 1
            self.component_counts[log_entry.component] += 1
            
            # Track errors specifically
            if log_entry.level in ['ERROR', 'CRITICAL']:
                error_key = f"{log_entry.component}:{log_entry.message[:100]}"
                self.error_counts[error_key] += 1
            
            # Cleanup old entries
            self._cleanup_old_entries()
            
        except Exception as e:
            print(f"Error adding log entry to aggregator: {e}")
    
    def _cleanup_old_entries(self):
        """Remove old log entries based on retention policy."""
        try:
            current_time = time.time()
            retention_seconds = self.retention_hours * 3600
            
            # Remove old entries (simple approach for deque)
            while (self.log_entries and 
                   current_time - datetime.fromisoformat(self.log_entries[0].timestamp.replace('Z', '+00:00')).timestamp() > retention_seconds):
                self.log_entries.popleft()
                
        except Exception as e:
            print(f"Error during log cleanup: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get aggregated log statistics."""
        try:
            current_time = time.time()
            
            # Recent entries (last hour)
            recent_entries = [
                entry for entry in self.log_entries
                if current_time - datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00')).timestamp() <= 3600
            ]
            
            # Error rate calculation
            total_recent = len(recent_entries)
            error_recent = len([entry for entry in recent_entries if entry.level in ['ERROR', 'CRITICAL']])
            error_rate = (error_recent / total_recent * 100) if total_recent > 0 else 0
            
            return {
                'total_entries': len(self.log_entries),
                'recent_entries_1h': total_recent,
                'error_rate_percent': error_rate,
                'level_distribution': dict(self.level_counts),
                'category_distribution': dict(self.category_counts),
                'component_distribution': dict(self.component_counts),
                'top_errors': dict(list(self.error_counts.items())[:10]),
                'retention_hours': self.retention_hours
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def search_logs(self, 
                   level: Optional[str] = None,
                   category: Optional[str] = None,
                   component: Optional[str] = None,
                   message_contains: Optional[str] = None,
                   since_minutes: int = 60,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """Search log entries with filters."""
        try:
            current_time = time.time()
            since_timestamp = current_time - (since_minutes * 60)
            
            results = []
            
            for entry in reversed(self.log_entries):  # Most recent first
                try:
                    entry_time = datetime.fromisoformat(entry.timestamp.replace('Z', '+00:00')).timestamp()
                    if entry_time < since_timestamp:
                        break  # Entries are ordered by time
                    
                    # Apply filters
                    if level and entry.level != level:
                        continue
                    if category and entry.category != category:
                        continue
                    if component and entry.component != component:
                        continue
                    if message_contains and message_contains.lower() not in entry.message.lower():
                        continue
                    
                    results.append(asdict(entry))
                    
                    if len(results) >= limit:
                        break
                        
                except Exception as e:
                    continue  # Skip malformed entries
            
            return results
            
        except Exception as e:
            return [{'error': str(e)}]


class StructuredLogger:
    """
    Professional structured logging system for trading applications.
    
    Provides comprehensive logging with structured JSON output, correlation
    tracking, performance monitoring, and audit trails.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 log_directory: str = "logs",
                 component_name: str = "dipmaster"):
        """
        Initialize structured logger.
        
        Args:
            config: Logger configuration
            log_directory: Directory for log files
            component_name: Default component name
        """
        self.config = config or {}
        self.log_directory = Path(log_directory)
        self.component_name = component_name
        
        # Create log directory
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # Context tracking
        self.default_context = LogContext(
            correlation_id=str(uuid.uuid4()),
            environment=self.config.get('environment', 'production'),
            component=component_name
        )
        
        # Log aggregation
        self.aggregator = LogAggregator(
            retention_hours=self.config.get('aggregation_retention_hours', 24)
        )
        
        # Setup loggers
        self._setup_loggers()
        
        # Performance tracking
        self.performance_metrics = deque(maxlen=10000)
        
        logger.info(f"📝 StructuredLogger initialized for {component_name}")
    
    def _setup_loggers(self):
        """Setup different logger instances for different purposes."""
        # Main application logger
        self.app_logger = self._create_logger(
            'dipmaster.app',
            self.log_directory / 'dipmaster_app.log',
            LogLevel.INFO
        )
        
        # Error logger with separate file
        self.error_logger = self._create_logger(
            'dipmaster.error',
            self.log_directory / 'dipmaster_error.log',
            LogLevel.ERROR
        )
        
        # Audit logger for compliance
        self.audit_logger = self._create_logger(
            'dipmaster.audit',
            self.log_directory / 'dipmaster_audit.log',
            LogLevel.INFO,
            timed_rotation=True
        )
        
        # Performance logger
        self.performance_logger = self._create_logger(
            'dipmaster.performance',
            self.log_directory / 'dipmaster_performance.log',
            LogLevel.INFO
        )
        
        # Trading activities logger
        self.trading_logger = self._create_logger(
            'dipmaster.trading',
            self.log_directory / 'dipmaster_trading.log',
            LogLevel.INFO
        )
    
    def _create_logger(self, 
                      logger_name: str, 
                      log_file: Path, 
                      level: LogLevel,
                      timed_rotation: bool = False) -> logging.Logger:
        """Create a configured logger instance."""
        
        logger_instance = logging.getLogger(logger_name)
        logger_instance.setLevel(getattr(logging, level.value))
        
        # Remove existing handlers
        logger_instance.handlers.clear()
        
        # File handler with rotation
        if timed_rotation:
            handler = TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=self.config.get('backup_count', 30),
                encoding='utf-8'
            )
        else:
            handler = RotatingFileHandler(
                log_file,
                maxBytes=self.config.get('max_bytes', 100 * 1024 * 1024),  # 100MB
                backupCount=self.config.get('backup_count', 10),
                encoding='utf-8'
            )
        
        # Set JSON formatter
        handler.setFormatter(JsonFormatter())
        logger_instance.addHandler(handler)
        
        # Console handler for development
        if self.config.get('console_output', False):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(JsonFormatter())
            logger_instance.addHandler(console_handler)
        
        # Prevent propagation to root logger
        logger_instance.propagate = False
        
        return logger_instance
    
    def create_context(self, 
                      correlation_id: Optional[str] = None,
                      session_id: Optional[str] = None,
                      user_id: Optional[str] = None,
                      request_id: Optional[str] = None,
                      component: Optional[str] = None) -> LogContext:
        """Create a new log context."""
        return LogContext(
            correlation_id=correlation_id or str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            request_id=request_id,
            component=component or self.component_name,
            environment=self.config.get('environment', 'production')
        )
    
    def _create_log_entry(self,
                         level: LogLevel,
                         category: LogCategory,
                         message: str,
                         context: LogContext,
                         data: Optional[Dict[str, Any]] = None,
                         duration_ms: Optional[float] = None,
                         exception: Optional[Exception] = None,
                         **kwargs) -> StructuredLogEntry:
        """Create a structured log entry."""
        
        import inspect
        import threading
        import os
        
        # Get caller information
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back  # Skip this method and the calling log method
        
        file_name = os.path.basename(caller_frame.f_code.co_filename) if caller_frame else None
        line_number = caller_frame.f_lineno if caller_frame else None
        function_name = caller_frame.f_code.co_name if caller_frame else None
        
        # Error information
        error_info = None
        if exception:
            error_info = {
                'type': type(exception).__name__,
                'message': str(exception),
                'traceback': traceback.format_exc() if level in [LogLevel.ERROR, LogLevel.CRITICAL] else None
            }
        
        # Create log entry
        log_entry = StructuredLogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level.value,
            category=category.value,
            message=message,
            component=context.component or self.component_name,
            correlation_id=context.correlation_id,
            session_id=context.session_id,
            user_id=context.user_id,
            request_id=context.request_id,
            trace_id=context.trace_id,
            environment=context.environment,
            file_name=file_name,
            line_number=line_number,
            function_name=function_name,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            data=data or {},
            error=error_info,
            duration_ms=duration_ms,
            **kwargs
        )
        
        return log_entry
    
    def _write_log(self, log_entry: StructuredLogEntry, target_logger: logging.Logger):
        """Write log entry to specified logger."""
        try:
            # Create log record
            log_record = logging.LogRecord(
                name=target_logger.name,
                level=getattr(logging, log_entry.level),
                pathname=log_entry.file_name or '',
                lineno=log_entry.line_number or 0,
                msg=log_entry.message,
                args=(),
                exc_info=None
            )
            
            # Attach structured data
            log_record.structured_data = asdict(log_entry)
            
            # Write to logger
            target_logger.handle(log_record)
            
            # Add to aggregator
            self.aggregator.add_log_entry(log_entry)
            
        except Exception as e:
            # Fallback logging
            print(f"Failed to write structured log: {e}")
            print(f"Original message: {log_entry.message}")
    
    def log_info(self, 
                message: str, 
                context: Optional[LogContext] = None,
                category: LogCategory = LogCategory.SYSTEM,
                data: Optional[Dict[str, Any]] = None,
                **kwargs):
        """Log info level message."""
        context = context or self.default_context
        log_entry = self._create_log_entry(
            LogLevel.INFO, category, message, context, data, **kwargs
        )
        self._write_log(log_entry, self.app_logger)
    
    def log_warning(self, 
                   message: str, 
                   context: Optional[LogContext] = None,
                   category: LogCategory = LogCategory.SYSTEM,
                   data: Optional[Dict[str, Any]] = None,
                   **kwargs):
        """Log warning level message."""
        context = context or self.default_context
        log_entry = self._create_log_entry(
            LogLevel.WARNING, category, message, context, data, **kwargs
        )
        self._write_log(log_entry, self.app_logger)
    
    def log_error(self, 
                 message: str, 
                 context: Optional[LogContext] = None,
                 category: LogCategory = LogCategory.SYSTEM,
                 data: Optional[Dict[str, Any]] = None,
                 exception: Optional[Exception] = None,
                 **kwargs):
        """Log error level message."""
        context = context or self.default_context
        log_entry = self._create_log_entry(
            LogLevel.ERROR, category, message, context, data, exception=exception, **kwargs
        )
        self._write_log(log_entry, self.error_logger)
        self._write_log(log_entry, self.app_logger)  # Also log to main log
    
    def log_critical(self, 
                    message: str, 
                    context: Optional[LogContext] = None,
                    category: LogCategory = LogCategory.SYSTEM,
                    data: Optional[Dict[str, Any]] = None,
                    exception: Optional[Exception] = None,
                    **kwargs):
        """Log critical level message."""
        context = context or self.default_context
        log_entry = self._create_log_entry(
            LogLevel.CRITICAL, category, message, context, data, exception=exception, **kwargs
        )
        self._write_log(log_entry, self.error_logger)
        self._write_log(log_entry, self.app_logger)
    
    def log_audit(self, 
                 message: str, 
                 context: Optional[LogContext] = None,
                 data: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """Log audit trail message."""
        context = context or self.default_context
        log_entry = self._create_log_entry(
            LogLevel.AUDIT, LogCategory.COMPLIANCE, message, context, data, **kwargs
        )
        self._write_log(log_entry, self.audit_logger)
    
    def log_performance(self, 
                       message: str, 
                       context: Optional[LogContext] = None,
                       duration_ms: Optional[float] = None,
                       data: Optional[Dict[str, Any]] = None,
                       **kwargs):
        """Log performance metrics."""
        context = context or self.default_context
        
        # Add system metrics if available
        try:
            import psutil
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
        except:
            cpu_usage = None
            memory_usage = None
        
        log_entry = self._create_log_entry(
            LogLevel.PERFORMANCE, LogCategory.PERFORMANCE, message, context, 
            data, duration_ms, cpu_usage=cpu_usage, memory_usage=memory_usage, **kwargs
        )
        self._write_log(log_entry, self.performance_logger)
        
        # Track performance metrics
        if duration_ms is not None:
            self.performance_metrics.append({
                'timestamp': time.time(),
                'operation': data.get('operation', 'unknown') if data else 'unknown',
                'duration_ms': duration_ms,
                'component': context.component
            })
    
    def log_trading(self, 
                   message: str, 
                   context: Optional[LogContext] = None,
                   data: Optional[Dict[str, Any]] = None,
                   symbol: Optional[str] = None,
                   trade_id: Optional[str] = None,
                   order_id: Optional[str] = None,
                   strategy: Optional[str] = None,
                   **kwargs):
        """Log trading-specific message."""
        context = context or self.default_context
        log_entry = self._create_log_entry(
            LogLevel.INFO, LogCategory.TRADING, message, context, data,
            symbol=symbol, trade_id=trade_id, order_id=order_id, strategy=strategy, **kwargs
        )
        self._write_log(log_entry, self.trading_logger)
        self._write_log(log_entry, self.app_logger)  # Also log to main log
    
    def timer(self, operation_name: str, context: Optional[LogContext] = None) -> PerformanceTimer:
        """Create a performance timer context manager."""
        context = context or self.default_context
        return PerformanceTimer(self, operation_name, context)
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics."""
        try:
            # Aggregator statistics
            agg_stats = self.aggregator.get_statistics()
            
            # Performance statistics
            recent_perf = [m for m in self.performance_metrics 
                          if time.time() - m['timestamp'] <= 3600]  # Last hour
            
            perf_stats = {}
            if recent_perf:
                durations = [m['duration_ms'] for m in recent_perf]
                perf_stats = {
                    'total_operations': len(recent_perf),
                    'avg_duration_ms': sum(durations) / len(durations),
                    'max_duration_ms': max(durations),
                    'min_duration_ms': min(durations),
                    'operations_by_component': defaultdict(int)
                }
                
                for metric in recent_perf:
                    perf_stats['operations_by_component'][metric['component']] += 1
                
                perf_stats['operations_by_component'] = dict(perf_stats['operations_by_component'])
            
            return {
                'timestamp': time.time(),
                'aggregation': agg_stats,
                'performance': perf_stats,
                'log_files': {
                    'app_log': str(self.log_directory / 'dipmaster_app.log'),
                    'error_log': str(self.log_directory / 'dipmaster_error.log'),
                    'audit_log': str(self.log_directory / 'dipmaster_audit.log'),
                    'performance_log': str(self.log_directory / 'dipmaster_performance.log'),
                    'trading_log': str(self.log_directory / 'dipmaster_trading.log')
                }
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def search_logs(self, **kwargs) -> List[Dict[str, Any]]:
        """Search log entries using aggregator."""
        return self.aggregator.search_logs(**kwargs)
    
    def export_logs(self, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   categories: Optional[List[str]] = None,
                   format_type: str = 'json') -> str:
        """Export logs for external analysis."""
        try:
            # Default time range (last 24 hours)
            if end_time is None:
                end_time = datetime.now(timezone.utc)
            if start_time is None:
                start_time = end_time - timedelta(hours=24)
            
            # Filter logs by time range and categories
            filtered_logs = []
            
            for log_entry in self.aggregator.log_entries:
                try:
                    log_time = datetime.fromisoformat(log_entry.timestamp.replace('Z', '+00:00'))
                    
                    if start_time <= log_time <= end_time:
                        if categories is None or log_entry.category in categories:
                            filtered_logs.append(asdict(log_entry))
                except:
                    continue
            
            # Export in requested format
            if format_type == 'json':
                return json.dumps(filtered_logs, indent=2, ensure_ascii=False)
            elif format_type == 'ndjson':
                return '\n'.join(json.dumps(log, ensure_ascii=False) for log in filtered_logs)
            else:
                return json.dumps({'error': f'Unsupported format: {format_type}'})
                
        except Exception as e:
            return json.dumps({'error': str(e)})
    
    def compress_old_logs(self, days_old: int = 7):
        """Compress old log files to save space."""
        try:
            cutoff_time = time.time() - (days_old * 24 * 3600)
            
            for log_file in self.log_directory.glob('*.log.*'):
                if log_file.stat().st_mtime < cutoff_time and not log_file.name.endswith('.gz'):
                    # Compress the file
                    compressed_file = log_file.with_suffix(log_file.suffix + '.gz')
                    
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(compressed_file, 'wb') as f_out:
                            f_out.writelines(f_in)
                    
                    # Remove original file
                    log_file.unlink()
                    
                    logger.info(f"Compressed old log file: {log_file} -> {compressed_file}")
            
        except Exception as e:
            logger.error(f"Error compressing old logs: {e}")


# Global logger instance
_global_structured_logger: Optional[StructuredLogger] = None


def get_logger(component_name: str = "dipmaster") -> StructuredLogger:
    """Get global structured logger instance."""
    global _global_structured_logger
    
    if _global_structured_logger is None:
        _global_structured_logger = StructuredLogger(component_name=component_name)
    
    return _global_structured_logger


def initialize_logger(config: Dict[str, Any], component_name: str = "dipmaster") -> StructuredLogger:
    """Initialize global structured logger with configuration."""
    global _global_structured_logger
    
    _global_structured_logger = StructuredLogger(
        config=config,
        log_directory=config.get('log_directory', 'logs'),
        component_name=component_name
    )
    
    return _global_structured_logger


# Factory function
def create_structured_logger(config: Dict[str, Any]) -> StructuredLogger:
    """Create and configure structured logger."""
    return StructuredLogger(
        config=config.get('logging_config', {}),
        log_directory=config.get('log_directory', 'logs'),
        component_name=config.get('component_name', 'dipmaster')
    )