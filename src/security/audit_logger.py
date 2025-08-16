#!/usr/bin/env python3
"""
Security Audit Logger for DipMaster Trading System
安全审计日志记录器 - 记录所有安全敏感操作

Features:
- Comprehensive security event logging
- Tamper-evident log files with checksums
- Real-time security monitoring and alerting
- Log rotation and long-term retention
- Compliance-ready audit trails
"""

import os
import json
import time
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import queue

logger = logging.getLogger(__name__)


class SecurityAuditLogger:
    """
    Enterprise security audit logging system.
    
    Provides tamper-evident logging of all security-sensitive operations
    with real-time monitoring and compliance features.
    """
    
    def __init__(self, 
                 log_dir: Optional[str] = None,
                 enable_real_time_alerts: bool = True,
                 max_log_size: int = 50 * 1024 * 1024):  # 50MB
        """
        Initialize security audit logger.
        
        Args:
            log_dir: Directory for security logs
            enable_real_time_alerts: Enable real-time security alerts
            max_log_size: Maximum log file size before rotation
        """
        self.log_dir = Path(log_dir or "logs/security")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_real_time_alerts = enable_real_time_alerts
        self.max_log_size = max_log_size
        
        # Current log file
        self.current_log_file = self.log_dir / f"security_audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # Thread-safe logging
        self._lock = threading.RLock()
        self._log_queue = queue.Queue()
        
        # Security metrics
        self.session_start_time = datetime.now()
        self.total_events = 0
        self.critical_events = 0
        self.failed_operations = 0
        
        # Start background log processor
        self._stop_event = threading.Event()
        self._log_thread = threading.Thread(target=self._log_processor, daemon=True)
        self._log_thread.start()
        
        logger.info(f"🔍 SecurityAuditLogger initialized: {self.log_dir}")
        
        # Log system startup
        self.log_system_event('AUDIT_SYSTEM_START', {
            'log_dir': str(self.log_dir),
            'pid': os.getpid(),
            'session_id': self._generate_session_id()
        })
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        return f"audit_{int(time.time())}_{os.getpid()}"
    
    def _log_processor(self):
        """Background thread to process log queue."""
        while not self._stop_event.is_set():
            try:
                # Get log entry from queue with timeout
                log_entry = self._log_queue.get(timeout=1.0)
                self._write_log_entry(log_entry)
                self._log_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Log processor error: {e}")
    
    def _write_log_entry(self, log_entry: Dict[str, Any]):
        """Write log entry to file with integrity protection."""
        try:
            with self._lock:
                # Check if log rotation is needed
                if self.current_log_file.exists() and self.current_log_file.stat().st_size > self.max_log_size:
                    self._rotate_log_file()
                
                # Add integrity checksum
                log_entry['checksum'] = self._calculate_entry_checksum(log_entry)
                
                # Write to log file
                with open(self.current_log_file, 'a') as f:
                    json.dump(log_entry, f, separators=(',', ':'))
                    f.write('\n')
                
                # Update metrics
                self.total_events += 1
                if log_entry.get('severity') == 'CRITICAL':
                    self.critical_events += 1
                if log_entry.get('result') == 'FAILURE':
                    self.failed_operations += 1
                
                # Real-time alerting
                if self.enable_real_time_alerts:
                    self._check_security_alerts(log_entry)
                
        except Exception as e:
            logger.error(f"❌ Failed to write audit log: {e}")
    
    def _calculate_entry_checksum(self, log_entry: Dict[str, Any]) -> str:
        """Calculate SHA-256 checksum of log entry for integrity."""
        # Create a copy without the checksum field
        entry_copy = {k: v for k, v in log_entry.items() if k != 'checksum'}
        entry_json = json.dumps(entry_copy, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(entry_json.encode()).hexdigest()[:16]  # First 16 chars
    
    def _rotate_log_file(self):
        """Rotate current log file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            rotated_file = self.log_dir / f"security_audit_{timestamp}.jsonl"
            
            # Move current log file
            if self.current_log_file.exists():
                self.current_log_file.rename(rotated_file)
                logger.info(f"📋 Rotated security log: {rotated_file}")
            
            # Create new log file
            self.current_log_file = self.log_dir / f"security_audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
        except Exception as e:
            logger.error(f"❌ Log rotation failed: {e}")
    
    def _check_security_alerts(self, log_entry: Dict[str, Any]):
        """Check for security conditions that require alerts."""
        alert_conditions = [
            log_entry.get('severity') == 'CRITICAL',
            log_entry.get('event_type') in ['KEY_ACCESS_ERROR', 'INVALID_ACCESS_ATTEMPT'],
            log_entry.get('result') == 'FAILURE' and 'authentication' in log_entry.get('operation', '').lower(),
            self.failed_operations > 5 and (time.time() - self.session_start_time.timestamp()) < 300  # 5 failures in 5 minutes
        ]
        
        if any(alert_conditions):
            self._send_security_alert(log_entry)
    
    def _send_security_alert(self, log_entry: Dict[str, Any]):
        """Send real-time security alert."""
        # In production, this would send to monitoring systems
        # For now, just log at critical level
        logger.critical(f"🚨 SECURITY ALERT: {log_entry['event_type']} - {log_entry.get('description', '')}")
    
    def log_key_operation(self, 
                         operation: str, 
                         key_id: str, 
                         exchange: str = 'unknown',
                         error_message: Optional[str] = None,
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Log API key related operations.
        
        Args:
            operation: Operation type (KEY_CREATE, KEY_ACCESS, KEY_DELETE, etc.)
            key_id: API key identifier
            exchange: Exchange name
            error_message: Error message if operation failed
            metadata: Additional metadata
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'KEY_OPERATION',
            'operation': operation,
            'key_id': key_id,
            'exchange': exchange,
            'result': 'FAILURE' if error_message else 'SUCCESS',
            'error_message': error_message,
            'severity': 'CRITICAL' if error_message else 'INFO',
            'user_id': os.getenv('USER', 'system'),
            'process_id': os.getpid(),
            'metadata': metadata or {}
        }
        
        self._log_queue.put(log_entry)
    
    def log_access_attempt(self,
                          resource: str,
                          access_type: str,
                          result: str,
                          user_id: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None):
        """
        Log access attempts to protected resources.
        
        Args:
            resource: Resource being accessed
            access_type: Type of access (READ, WRITE, DELETE, etc.)
            result: Result (SUCCESS, FAILURE, DENIED)
            user_id: User identifier
            ip_address: Client IP address
            metadata: Additional metadata
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'ACCESS_ATTEMPT',
            'resource': resource,
            'access_type': access_type,
            'result': result,
            'user_id': user_id or os.getenv('USER', 'unknown'),
            'ip_address': ip_address,
            'severity': 'CRITICAL' if result == 'FAILURE' else 'INFO',
            'process_id': os.getpid(),
            'metadata': metadata or {}
        }
        
        self._log_queue.put(log_entry)
    
    def log_system_event(self,
                        event_type: str,
                        details: Dict[str, Any],
                        severity: str = 'INFO'):
        """
        Log system-level security events.
        
        Args:
            event_type: Type of system event
            details: Event details
            severity: Event severity (INFO, WARNING, CRITICAL)
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'process_id': os.getpid(),
            'user_id': os.getenv('USER', 'system')
        }
        
        self._log_queue.put(log_entry)
    
    def log_trading_operation(self,
                             operation: str,
                             symbol: str,
                             result: str,
                             details: Optional[Dict[str, Any]] = None):
        """
        Log trading operations for compliance.
        
        Args:
            operation: Trading operation (BUY, SELL, CANCEL, etc.)
            symbol: Trading symbol
            result: Operation result
            details: Additional details
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': 'TRADING_OPERATION',
            'operation': operation,
            'symbol': symbol,
            'result': result,
            'severity': 'WARNING' if result == 'FAILURE' else 'INFO',
            'details': details or {},
            'process_id': os.getpid()
        }
        
        self._log_queue.put(log_entry)
    
    def get_audit_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get audit summary for specified time period.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Summary of audit events
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Read recent log files
            events = []
            for log_file in self.log_dir.glob("security_audit_*.jsonl"):
                try:
                    with open(log_file, 'r') as f:
                        for line in f:
                            event = json.loads(line.strip())
                            event_time = datetime.fromisoformat(event['timestamp'])
                            if event_time >= cutoff_time:
                                events.append(event)
                except Exception as e:
                    logger.error(f"❌ Error reading log file {log_file}: {e}")
                    continue
            
            # Analyze events
            summary = {
                'period_hours': hours,
                'total_events': len(events),
                'critical_events': len([e for e in events if e.get('severity') == 'CRITICAL']),
                'failed_operations': len([e for e in events if e.get('result') == 'FAILURE']),
                'key_operations': len([e for e in events if e.get('event_type') == 'KEY_OPERATION']),
                'access_attempts': len([e for e in events if e.get('event_type') == 'ACCESS_ATTEMPT']),
                'trading_operations': len([e for e in events if e.get('event_type') == 'TRADING_OPERATION']),
                'unique_users': len(set(e.get('user_id') for e in events if e.get('user_id'))),
                'event_types': {},
                'failure_reasons': {}
            }
            
            # Count event types
            for event in events:
                event_type = event.get('event_type', 'UNKNOWN')
                summary['event_types'][event_type] = summary['event_types'].get(event_type, 0) + 1
                
                if event.get('result') == 'FAILURE' and event.get('error_message'):
                    error = event['error_message'][:100]  # First 100 chars
                    summary['failure_reasons'][error] = summary['failure_reasons'].get(error, 0) + 1
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Failed to generate audit summary: {e}")
            return {'error': str(e)}
    
    def verify_log_integrity(self, log_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify integrity of audit logs.
        
        Args:
            log_file: Specific log file to verify (optional)
            
        Returns:
            Integrity verification results
        """
        try:
            files_to_check = [Path(log_file)] if log_file else list(self.log_dir.glob("security_audit_*.jsonl"))
            
            results = {
                'total_files': len(files_to_check),
                'verified_files': 0,
                'corrupted_files': 0,
                'total_entries': 0,
                'corrupted_entries': 0,
                'details': {}
            }
            
            for log_file in files_to_check:
                file_result = {
                    'total_entries': 0,
                    'valid_entries': 0,
                    'corrupted_entries': 0,
                    'errors': []
                }
                
                try:
                    with open(log_file, 'r') as f:
                        for line_num, line in enumerate(f, 1):
                            try:
                                entry = json.loads(line.strip())
                                file_result['total_entries'] += 1
                                
                                # Verify checksum if present
                                if 'checksum' in entry:
                                    stored_checksum = entry['checksum']
                                    calculated_checksum = self._calculate_entry_checksum(entry)
                                    
                                    if stored_checksum == calculated_checksum:
                                        file_result['valid_entries'] += 1
                                    else:
                                        file_result['corrupted_entries'] += 1
                                        file_result['errors'].append(f"Line {line_num}: Checksum mismatch")
                                else:
                                    file_result['valid_entries'] += 1  # No checksum to verify
                                    
                            except json.JSONDecodeError as e:
                                file_result['corrupted_entries'] += 1
                                file_result['errors'].append(f"Line {line_num}: JSON decode error")
                    
                    results['details'][str(log_file)] = file_result
                    results['total_entries'] += file_result['total_entries']
                    results['corrupted_entries'] += file_result['corrupted_entries']
                    
                    if file_result['corrupted_entries'] == 0:
                        results['verified_files'] += 1
                    else:
                        results['corrupted_files'] += 1
                        
                except Exception as e:
                    results['corrupted_files'] += 1
                    results['details'][str(log_file)] = {'error': str(e)}
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Log integrity verification failed: {e}")
            return {'error': str(e)}
    
    def cleanup_old_logs(self, days: int = 90) -> int:
        """
        Clean up old audit logs.
        
        Args:
            days: Number of days to retain logs
            
        Returns:
            Number of files removed
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            removed_count = 0
            
            for log_file in self.log_dir.glob("security_audit_*.jsonl"):
                try:
                    # Get file creation time from filename or file stats
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    
                    if file_time < cutoff_date:
                        log_file.unlink()
                        removed_count += 1
                        logger.info(f"🗑️  Removed old audit log: {log_file}")
                        
                except Exception as e:
                    logger.error(f"❌ Error removing log file {log_file}: {e}")
            
            return removed_count
            
        except Exception as e:
            logger.error(f"❌ Log cleanup failed: {e}")
            return 0
    
    def shutdown(self):
        """Shutdown audit logger gracefully."""
        logger.info("🛑 Shutting down security audit logger...")
        
        # Stop background thread
        self._stop_event.set()
        
        # Process remaining queue items
        while not self._log_queue.empty():
            try:
                log_entry = self._log_queue.get_nowait()
                self._write_log_entry(log_entry)
            except queue.Empty:
                break
        
        # Wait for thread to finish
        if self._log_thread.is_alive():
            self._log_thread.join(timeout=5)
        
        # Log shutdown
        self.log_system_event('AUDIT_SYSTEM_SHUTDOWN', {
            'total_events': self.total_events,
            'critical_events': self.critical_events,
            'failed_operations': self.failed_operations,
            'uptime_seconds': int((datetime.now() - self.session_start_time).total_seconds())
        })


def create_audit_logger(log_dir: Optional[str] = None) -> SecurityAuditLogger:
    """Factory function to create security audit logger."""
    return SecurityAuditLogger(log_dir)