#!/usr/bin/env python3
"""
Structured Logging System for DipMaster Trading System
结构化日志系统 - 专业级金融系统日志管理

Features:
- Multi-level structured logging with JSON format
- Automatic log rotation and archival
- Key metrics extraction and aggregation
- Trading-specific log enrichment
- Performance-optimized async logging
- Centralized log correlation and tracing
"""

import asyncio
import json
import time
import logging
import logging.handlers
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import threading
import gzip
import os
from pathlib import Path
import uuid
import hashlib
from collections import defaultdict, deque
import sys

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Extended log levels for trading systems."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARN = 30
    ERROR = 40
    CRITICAL = 50
    ALERT = 60      # Trading-specific alert level
    EMERGENCY = 70  # System emergency level


class LogCategory(Enum):
    """Log categories for trading system components."""
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"
    EXECUTION = "execution"
    MARKET_DATA = "market_data"
    STRATEGY = "strategy"
    MONITORING = "monitoring"
    SECURITY = "security"
    PERFORMANCE = "performance"
    COMPLIANCE = "compliance"


@dataclass
class LogContext:
    """Log context for correlation and tracing."""
    session_id: str
    correlation_id: Optional[str] = None
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    strategy: Optional[str] = None
    symbol: Optional[str] = None
    trade_id: Optional[str] = None
    order_id: Optional[str] = None


@dataclass
class TradingLogEntry:
    """Structured trading log entry."""
    timestamp: str
    level: str
    category: str
    component: str
    message: str
    context: LogContext
    data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    exception: Optional[Dict[str, Any]] = None
    performance: Optional[Dict[str, float]] = None


class AsyncLogHandler(logging.Handler):
    """High-performance async log handler."""
    
    def __init__(self, log_processor: 'StructuredLogProcessor'):
        super().__init__()
        self.log_processor = log_processor
        self.log_queue = asyncio.Queue(maxsize=10000)
        self.processing_task = None
        self.is_processing = False
    
    def emit(self, record):
        """Emit log record to async queue."""
        try:
            # Convert log record to structured format
            structured_record = self._convert_record(record)
            
            # Add to queue (non-blocking)
            try:
                self.log_queue.put_nowait(structured_record)
            except asyncio.QueueFull:
                # Drop oldest records if queue is full
                try:
                    self.log_queue.get_nowait()
                    self.log_queue.put_nowait(structured_record)
                except asyncio.QueueEmpty:
                    pass
                    
        except Exception as e:
            # Fallback to stderr to avoid infinite recursion
            print(f"Error in AsyncLogHandler: {e}", file=sys.stderr)
    
    def _convert_record(self, record) -> TradingLogEntry:
        """Convert logging record to structured format."""
        # Extract context from record
        context = getattr(record, 'context', LogContext(session_id=str(uuid.uuid4())))
        
        # Extract additional data
        data = getattr(record, 'data', {})
        metrics = getattr(record, 'metrics', {})
        tags = getattr(record, 'tags', {})
        
        # Handle exception info
        exception_info = None
        if record.exc_info:
            exception_info = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': self.format(record) if record.exc_info[2] else None
            }
        
        # Performance metrics
        performance = getattr(record, 'performance', None)
        
        return TradingLogEntry(
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            level=record.levelname,
            category=getattr(record, 'category', LogCategory.SYSTEM.value),
            component=getattr(record, 'component', record.name),
            message=record.getMessage(),
            context=context,
            data=data,
            metrics=metrics,
            tags=tags,
            exception=exception_info,
            performance=performance
        )
    
    async def start_processing(self):
        """Start async log processing."""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_task = asyncio.create_task(self._process_logs())
    
    async def stop_processing(self):
        """Stop async log processing."""
        if not self.is_processing:
            return
        
        self.is_processing = False
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
        
        # Process remaining logs
        while not self.log_queue.empty():
            try:
                record = self.log_queue.get_nowait()
                await self.log_processor.process_log(record)
            except asyncio.QueueEmpty:
                break
    
    async def _process_logs(self):
        """Process logs from queue."""
        while self.is_processing:
            try:
                # Get log record with timeout
                record = await asyncio.wait_for(self.log_queue.get(), timeout=1.0)
                await self.log_processor.process_log(record)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Error processing log: {e}", file=sys.stderr)


class LogRotationManager:
    """Manages log file rotation and archival."""
    
    def __init__(self, 
                 log_dir: Path,
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 max_files: int = 30,
                 compression: bool = True):
        self.log_dir = log_dir
        self.max_file_size = max_file_size
        self.max_files = max_files
        self.compression = compression
        self.current_files = {}
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def get_log_file(self, category: str, date_str: str) -> Path:
        """Get current log file for category and date."""
        filename = f"{category}_{date_str}.log"
        filepath = self.log_dir / filename
        
        # Check if rotation is needed
        if filepath.exists() and filepath.stat().st_size > self.max_file_size:
            self._rotate_file(filepath)
        
        return filepath
    
    def _rotate_file(self, filepath: Path):
        """Rotate log file when it exceeds size limit."""
        base_name = filepath.stem
        extension = filepath.suffix
        
        # Find next rotation number
        rotation_num = 1
        while True:
            rotated_name = f"{base_name}.{rotation_num}{extension}"
            rotated_path = filepath.parent / rotated_name
            
            if not rotated_path.exists():
                break
            rotation_num += 1
        
        # Rename current file
        filepath.rename(rotated_path)
        
        # Compress if enabled
        if self.compression:
            self._compress_file(rotated_path)
        
        # Clean up old files
        self._cleanup_old_files(base_name, extension)
    
    def _compress_file(self, filepath: Path):
        """Compress log file using gzip."""
        try:
            compressed_path = filepath.with_suffix(filepath.suffix + '.gz')
            
            with open(filepath, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Remove original file
            filepath.unlink()
            
        except Exception as e:
            logger.error(f"Failed to compress {filepath}: {e}")
    
    def _cleanup_old_files(self, base_name: str, extension: str):
        """Remove old log files exceeding max_files limit."""
        try:
            # Find all rotated files
            pattern = f"{base_name}.*{extension}*"
            rotated_files = list(self.log_dir.glob(pattern))
            
            # Sort by modification time (oldest first)
            rotated_files.sort(key=lambda p: p.stat().st_mtime)
            
            # Remove excess files
            while len(rotated_files) > self.max_files:
                old_file = rotated_files.pop(0)
                old_file.unlink()
                
        except Exception as e:
            logger.error(f"Failed to cleanup old files: {e}")


class MetricsExtractor:
    """Extracts and aggregates metrics from log entries."""
    
    def __init__(self):
        self.metrics_cache = defaultdict(lambda: defaultdict(list))
        self.aggregated_metrics = defaultdict(dict)
        self.last_aggregation = time.time()
        self.aggregation_interval = 60  # 1 minute
    
    def extract_metrics(self, log_entry: TradingLogEntry):
        """Extract metrics from log entry."""
        timestamp = time.time()
        
        # Extract performance metrics
        if log_entry.performance:
            for metric_name, value in log_entry.performance.items():
                key = f"performance.{log_entry.component}.{metric_name}"
                self.metrics_cache[key]['values'].append(value)
                self.metrics_cache[key]['timestamps'].append(timestamp)
        
        # Extract custom metrics
        if log_entry.metrics:
            for metric_name, value in log_entry.metrics.items():
                key = f"custom.{log_entry.category}.{metric_name}"
                self.metrics_cache[key]['values'].append(value)
                self.metrics_cache[key]['timestamps'].append(timestamp)
        
        # Extract trading-specific metrics
        if log_entry.category == LogCategory.TRADING.value:
            self._extract_trading_metrics(log_entry, timestamp)
        elif log_entry.category == LogCategory.RISK.value:
            self._extract_risk_metrics(log_entry, timestamp)
        elif log_entry.category == LogCategory.EXECUTION.value:
            self._extract_execution_metrics(log_entry, timestamp)
        
        # Periodic aggregation
        if timestamp - self.last_aggregation > self.aggregation_interval:
            self._aggregate_metrics()
            self.last_aggregation = timestamp
    
    def _extract_trading_metrics(self, log_entry: TradingLogEntry, timestamp: float):
        """Extract trading-specific metrics."""
        data = log_entry.data
        
        # Trade counts
        if 'trade_opened' in data:
            self.metrics_cache['trading.trades_opened']['count'].append(1)
            self.metrics_cache['trading.trades_opened']['timestamps'].append(timestamp)
        
        if 'trade_closed' in data:
            self.metrics_cache['trading.trades_closed']['count'].append(1)
            self.metrics_cache['trading.trades_closed']['timestamps'].append(timestamp)
            
            # PnL tracking
            if 'pnl' in data:
                self.metrics_cache['trading.pnl']['values'].append(data['pnl'])
                self.metrics_cache['trading.pnl']['timestamps'].append(timestamp)
        
        # Signal metrics
        if 'signal_generated' in data:
            self.metrics_cache['trading.signals_generated']['count'].append(1)
            self.metrics_cache['trading.signals_generated']['timestamps'].append(timestamp)
    
    def _extract_risk_metrics(self, log_entry: TradingLogEntry, timestamp: float):
        """Extract risk-specific metrics."""
        data = log_entry.data
        
        # Risk violations
        if 'risk_violation' in data:
            violation_type = data.get('violation_type', 'unknown')
            key = f"risk.violations.{violation_type}"
            self.metrics_cache[key]['count'].append(1)
            self.metrics_cache[key]['timestamps'].append(timestamp)
        
        # Portfolio metrics
        for metric in ['var', 'expected_shortfall', 'max_drawdown', 'leverage']:
            if metric in data:
                key = f"risk.{metric}"
                self.metrics_cache[key]['values'].append(data[metric])
                self.metrics_cache[key]['timestamps'].append(timestamp)
    
    def _extract_execution_metrics(self, log_entry: TradingLogEntry, timestamp: float):
        """Extract execution-specific metrics."""
        data = log_entry.data
        
        # Execution quality
        if 'slippage_bps' in data:
            self.metrics_cache['execution.slippage']['values'].append(data['slippage_bps'])
            self.metrics_cache['execution.slippage']['timestamps'].append(timestamp)
        
        if 'latency_ms' in data:
            self.metrics_cache['execution.latency']['values'].append(data['latency_ms'])
            self.metrics_cache['execution.latency']['timestamps'].append(timestamp)
        
        # Order status
        if 'order_status' in data:
            status = data['order_status']
            key = f"execution.orders.{status}"
            self.metrics_cache[key]['count'].append(1)
            self.metrics_cache[key]['timestamps'].append(timestamp)
    
    def _aggregate_metrics(self):
        """Aggregate collected metrics."""
        current_time = time.time()
        window_start = current_time - self.aggregation_interval
        
        for metric_name, metric_data in self.metrics_cache.items():
            # Filter recent data
            recent_indices = [
                i for i, ts in enumerate(metric_data.get('timestamps', []))
                if ts >= window_start
            ]
            
            if not recent_indices:
                continue
            
            aggregated = {}
            
            # Aggregate values
            if 'values' in metric_data:
                recent_values = [metric_data['values'][i] for i in recent_indices]
                if recent_values:
                    aggregated.update({
                        'count': len(recent_values),
                        'sum': sum(recent_values),
                        'avg': sum(recent_values) / len(recent_values),
                        'min': min(recent_values),
                        'max': max(recent_values)
                    })
            
            # Aggregate counts
            if 'count' in metric_data:
                recent_counts = [metric_data['count'][i] for i in recent_indices]
                aggregated['total_count'] = sum(recent_counts)
            
            if aggregated:
                self.aggregated_metrics[metric_name] = aggregated
    
    def get_aggregated_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get current aggregated metrics."""
        return dict(self.aggregated_metrics)
    
    def get_metric_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metric summary for specified time window."""
        cutoff_time = time.time() - (hours * 3600)
        summary = defaultdict(lambda: defaultdict(list))
        
        for metric_name, metric_data in self.metrics_cache.items():
            timestamps = metric_data.get('timestamps', [])
            
            # Filter by time window
            recent_indices = [
                i for i, ts in enumerate(timestamps)
                if ts >= cutoff_time
            ]
            
            if recent_indices:
                if 'values' in metric_data:
                    recent_values = [metric_data['values'][i] for i in recent_indices]
                    summary[metric_name]['values'] = recent_values
                
                if 'count' in metric_data:
                    recent_counts = [metric_data['count'][i] for i in recent_indices]
                    summary[metric_name]['total_count'] = sum(recent_counts)
        
        return dict(summary)


class StructuredLogProcessor:
    """Main processor for structured logging system."""
    
    def __init__(self,
                 log_dir: Union[str, Path] = "logs",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize structured log processor.
        
        Args:
            log_dir: Directory for log files
            config: Configuration parameters
        """
        self.log_dir = Path(log_dir)
        self.config = config or {}
        
        # Components
        self.rotation_manager = LogRotationManager(
            log_dir=self.log_dir,
            max_file_size=self.config.get('max_file_size', 100 * 1024 * 1024),
            max_files=self.config.get('max_files', 30),
            compression=self.config.get('compression', True)
        )
        
        self.metrics_extractor = MetricsExtractor()
        
        # File handles cache
        self.file_handles = {}
        self.file_locks = defaultdict(threading.Lock)
        
        # Processing statistics
        self.stats = {
            'logs_processed': 0,
            'logs_written': 0,
            'metrics_extracted': 0,
            'errors': 0
        }
        
        logger.info(f"📝 StructuredLogProcessor initialized with log_dir: {self.log_dir}")
    
    async def process_log(self, log_entry: TradingLogEntry):
        """Process a single log entry."""
        try:
            # Extract metrics
            self.metrics_extractor.extract_metrics(log_entry)
            self.stats['metrics_extracted'] += 1
            
            # Write to file
            await self._write_log_entry(log_entry)
            self.stats['logs_written'] += 1
            
            # Update statistics
            self.stats['logs_processed'] += 1
            
        except Exception as e:
            self.stats['errors'] += 1
            # Fallback logging to stderr
            print(f"Error processing log entry: {e}", file=sys.stderr)
    
    async def _write_log_entry(self, log_entry: TradingLogEntry):
        """Write log entry to appropriate file."""
        # Determine file based on category and date
        date_str = datetime.now().strftime('%Y%m%d')
        log_file = self.rotation_manager.get_log_file(log_entry.category, date_str)
        
        # Convert to JSON
        log_json = json.dumps(asdict(log_entry), separators=(',', ':'))
        
        # Write to file (thread-safe)
        file_key = str(log_file)
        with self.file_locks[file_key]:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(log_json + '\n')
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self.stats,
            'aggregated_metrics': self.metrics_extractor.get_aggregated_metrics(),
            'file_handles_count': len(self.file_handles)
        }
    
    def close(self):
        """Close all file handles."""
        for handle in self.file_handles.values():
            try:
                handle.close()
            except:
                pass
        self.file_handles.clear()


class StructuredLogger:
    """High-level structured logger for trading system."""
    
    def __init__(self,
                 name: str,
                 log_dir: Union[str, Path] = "logs",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize structured logger.
        
        Args:
            name: Logger name
            log_dir: Log directory
            config: Configuration
        """
        self.name = name
        self.config = config or {}
        
        # Create log processor
        self.processor = StructuredLogProcessor(log_dir, config)
        
        # Create async handler
        self.async_handler = AsyncLogHandler(self.processor)
        
        # Setup Python logger
        self.logger = logging.getLogger(name)
        self.logger.addHandler(self.async_handler)
        self.logger.setLevel(logging.DEBUG)
        
        # Default context
        self.default_context = LogContext(
            session_id=str(uuid.uuid4()),
            component=name
        )
    
    async def start(self):
        """Start async logging."""
        await self.async_handler.start_processing()
    
    async def stop(self):
        """Stop async logging."""
        await self.async_handler.stop_processing()
        self.processor.close()
    
    def _log(self,
             level: int,
             message: str,
             category: LogCategory = LogCategory.SYSTEM,
             context: Optional[LogContext] = None,
             data: Optional[Dict[str, Any]] = None,
             metrics: Optional[Dict[str, float]] = None,
             tags: Optional[Dict[str, str]] = None,
             performance: Optional[Dict[str, float]] = None,
             exc_info: bool = False):
        """Internal logging method."""
        
        # Use default context if none provided
        log_context = context or self.default_context
        
        # Create log record
        extra = {
            'category': category.value,
            'component': self.name,
            'context': log_context,
            'data': data or {},
            'metrics': metrics or {},
            'tags': tags or {},
            'performance': performance
        }
        
        self.logger.log(level, message, extra=extra, exc_info=exc_info)
    
    # Convenience methods for different log levels
    def trace(self, message: str, **kwargs):
        """Log trace message."""
        self._log(LogLevel.TRACE.value, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG.value, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO.value, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARN.value, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR.value, message, exc_info=True, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL.value, message, exc_info=True, **kwargs)
    
    def alert(self, message: str, **kwargs):
        """Log trading alert."""
        self._log(LogLevel.ALERT.value, message, category=LogCategory.TRADING, **kwargs)
    
    def emergency(self, message: str, **kwargs):
        """Log emergency message."""
        self._log(LogLevel.EMERGENCY.value, message, exc_info=True, **kwargs)
    
    # Trading-specific logging methods
    def log_trade_entry(self,
                       symbol: str,
                       side: str,
                       quantity: float,
                       price: float,
                       signal_id: str,
                       strategy: str,
                       context: Optional[LogContext] = None,
                       **kwargs):
        """Log trade entry."""
        log_context = context or LogContext(
            session_id=self.default_context.session_id,
            strategy=strategy,
            symbol=symbol,
            trade_id=signal_id
        )
        
        data = {
            'trade_opened': True,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'signal_id': signal_id,
            'strategy': strategy,
            **kwargs
        }
        
        self._log(
            LogLevel.INFO.value,
            f"Trade entry: {side} {quantity} {symbol} @ {price}",
            category=LogCategory.TRADING,
            context=log_context,
            data=data
        )
    
    def log_trade_exit(self,
                      symbol: str,
                      side: str,
                      quantity: float,
                      exit_price: float,
                      entry_price: float,
                      pnl: float,
                      holding_minutes: int,
                      trade_id: str,
                      strategy: str,
                      context: Optional[LogContext] = None,
                      **kwargs):
        """Log trade exit."""
        log_context = context or LogContext(
            session_id=self.default_context.session_id,
            strategy=strategy,
            symbol=symbol,
            trade_id=trade_id
        )
        
        data = {
            'trade_closed': True,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'holding_minutes': holding_minutes,
            'trade_id': trade_id,
            'strategy': strategy,
            **kwargs
        }
        
        metrics = {
            'pnl': pnl,
            'holding_time': holding_minutes,
            'return_pct': (pnl / (entry_price * quantity)) * 100 if entry_price > 0 else 0
        }
        
        self._log(
            LogLevel.INFO.value,
            f"Trade exit: {side} {quantity} {symbol} @ {exit_price}, PnL: {pnl:.2f}",
            category=LogCategory.TRADING,
            context=log_context,
            data=data,
            metrics=metrics
        )
    
    def log_signal_generated(self,
                           signal_id: str,
                           symbol: str,
                           signal_type: str,
                           confidence: float,
                           price: float,
                           strategy: str,
                           technical_indicators: Dict[str, float],
                           context: Optional[LogContext] = None,
                           **kwargs):
        """Log signal generation."""
        log_context = context or LogContext(
            session_id=self.default_context.session_id,
            strategy=strategy,
            symbol=symbol,
            correlation_id=signal_id
        )
        
        data = {
            'signal_generated': True,
            'signal_id': signal_id,
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': confidence,
            'price': price,
            'strategy': strategy,
            'technical_indicators': technical_indicators,
            **kwargs
        }
        
        metrics = {
            'confidence': confidence,
            'rsi': technical_indicators.get('rsi', 0),
            'volume_ratio': technical_indicators.get('volume_ratio', 0)
        }
        
        self._log(
            LogLevel.INFO.value,
            f"Signal generated: {signal_type} {symbol} @ {price}, confidence: {confidence:.2f}",
            category=LogCategory.STRATEGY,
            context=log_context,
            data=data,
            metrics=metrics
        )
    
    def log_execution(self,
                     execution_id: str,
                     symbol: str,
                     side: str,
                     quantity: float,
                     price: float,
                     slippage_bps: float,
                     latency_ms: float,
                     venue: str,
                     status: str,
                     context: Optional[LogContext] = None,
                     **kwargs):
        """Log order execution."""
        log_context = context or LogContext(
            session_id=self.default_context.session_id,
            symbol=symbol,
            order_id=execution_id
        )
        
        data = {
            'execution_id': execution_id,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'slippage_bps': slippage_bps,
            'latency_ms': latency_ms,
            'venue': venue,
            'order_status': status,
            **kwargs
        }
        
        metrics = {
            'slippage_bps': slippage_bps,
            'latency_ms': latency_ms
        }
        
        performance = {
            'execution_latency': latency_ms,
            'slippage': slippage_bps
        }
        
        self._log(
            LogLevel.INFO.value,
            f"Execution: {side} {quantity} {symbol} @ {price}, slippage: {slippage_bps:.1f}bps",
            category=LogCategory.EXECUTION,
            context=log_context,
            data=data,
            metrics=metrics,
            performance=performance
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logging statistics."""
        return self.processor.get_processing_stats()


# Factory function
def create_structured_logger(name: str,
                            log_dir: Union[str, Path] = "logs",
                            config: Optional[Dict[str, Any]] = None) -> StructuredLogger:
    """Create and configure structured logger."""
    return StructuredLogger(name, log_dir, config)