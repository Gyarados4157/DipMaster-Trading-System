#!/usr/bin/env python3
"""
Access Control System for DipMaster Trading System
访问控制系统 - 基于角色的权限管理(RBAC)

Features:
- Role-based access control (RBAC)
- Fine-grained permission management
- Session management and authentication
- IP-based access restrictions
- Time-based access policies
"""

import os
import json
import time
import hashlib
import secrets
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading

from .audit_logger import SecurityAuditLogger

logger = logging.getLogger(__name__)


class Permission(Enum):
    """System permissions enumeration."""
    
    # API Key Management
    KEY_READ = "key:read"
    KEY_WRITE = "key:write"
    KEY_DELETE = "key:delete"
    KEY_ROTATE = "key:rotate"
    
    # Trading Operations
    TRADE_EXECUTE = "trade:execute"
    TRADE_VIEW = "trade:view"
    TRADE_CANCEL = "trade:cancel"
    
    # System Configuration
    CONFIG_READ = "config:read"
    CONFIG_WRITE = "config:write"
    CONFIG_DEPLOY = "config:deploy"
    
    # Dashboard and Monitoring
    DASHBOARD_VIEW = "dashboard:view"
    DASHBOARD_ADMIN = "dashboard:admin"
    
    # Security and Audit
    SECURITY_ADMIN = "security:admin"
    AUDIT_VIEW = "audit:view"
    
    # System Administration
    SYSTEM_ADMIN = "system:admin"
    USER_ADMIN = "user:admin"


class Role(Enum):
    """Predefined system roles."""
    
    TRADER = "trader"           # Can execute trades and view positions
    ANALYST = "analyst"         # Read-only access to trading data
    OPERATOR = "operator"       # System operation and monitoring
    ADMIN = "admin"            # Full system administration
    SECURITY_OFFICER = "security_officer"  # Security and audit functions


# Role-Permission Mapping
ROLE_PERMISSIONS = {
    Role.TRADER: {
        Permission.TRADE_EXECUTE,
        Permission.TRADE_VIEW, 
        Permission.TRADE_CANCEL,
        Permission.DASHBOARD_VIEW,
        Permission.CONFIG_READ
    },
    Role.ANALYST: {
        Permission.TRADE_VIEW,
        Permission.DASHBOARD_VIEW,
        Permission.CONFIG_READ,
        Permission.AUDIT_VIEW
    },
    Role.OPERATOR: {
        Permission.TRADE_VIEW,
        Permission.DASHBOARD_VIEW,
        Permission.DASHBOARD_ADMIN,
        Permission.CONFIG_READ,
        Permission.CONFIG_WRITE
    },
    Role.SECURITY_OFFICER: {
        Permission.KEY_READ,
        Permission.KEY_ROTATE,
        Permission.SECURITY_ADMIN,
        Permission.AUDIT_VIEW,
        Permission.CONFIG_READ
    },
    Role.ADMIN: set(Permission)  # All permissions
}


class AccessSession:
    """Represents an authenticated user session."""
    
    def __init__(self, user_id: str, roles: Set[Role], ip_address: Optional[str] = None):
        self.user_id = user_id
        self.roles = roles
        self.ip_address = ip_address
        self.session_id = secrets.token_urlsafe(32)
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.permissions = self._calculate_permissions()
    
    def _calculate_permissions(self) -> Set[Permission]:
        """Calculate effective permissions from roles."""
        permissions = set()
        for role in self.roles:
            permissions.update(ROLE_PERMISSIONS.get(role, set()))
        return permissions
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if session has specific permission."""
        return permission in self.permissions
    
    def is_expired(self, timeout_minutes: int = 480) -> bool:  # 8 hours default
        """Check if session has expired."""
        return (datetime.now() - self.last_activity).total_seconds() > (timeout_minutes * 60)
    
    def refresh_activity(self):
        """Update last activity time."""
        self.last_activity = datetime.now()


class AccessController:
    """
    Role-based access control system.
    
    Provides comprehensive access control with user authentication,
    role management, and permission checking.
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 audit_logger: Optional[SecurityAuditLogger] = None,
                 session_timeout: int = 480):  # 8 hours
        """
        Initialize access controller.
        
        Args:
            config_path: Path to access control configuration
            audit_logger: Security audit logger instance
            session_timeout: Session timeout in minutes
        """
        self.config_path = Path(config_path or "config/access_control.json")
        self.audit_logger = audit_logger or SecurityAuditLogger()
        self.session_timeout = session_timeout
        
        # Thread-safe operations
        self._lock = threading.RLock()
        
        # Active sessions
        self.active_sessions: Dict[str, AccessSession] = {}
        
        # User database (in production, use external auth system)
        self.users: Dict[str, Dict[str, Any]] = {}
        
        # IP-based restrictions
        self.ip_whitelist: Set[str] = set()
        self.ip_blacklist: Set[str] = set()
        
        # Load configuration
        self._load_config()
        
        # Start session cleanup thread
        self._cleanup_thread = threading.Thread(target=self._session_cleanup_worker, daemon=True)
        self._cleanup_thread.start()
        
        logger.info("🔐 AccessController initialized")
    
    def _load_config(self):
        """Load access control configuration."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Load users
                for user_id, user_data in config.get('users', {}).items():
                    roles = {Role(r) for r in user_data.get('roles', [])}
                    self.users[user_id] = {
                        'password_hash': user_data.get('password_hash'),
                        'roles': roles,
                        'enabled': user_data.get('enabled', True),
                        'ip_restrictions': user_data.get('ip_restrictions', []),
                        'created_at': user_data.get('created_at'),
                        'last_login': user_data.get('last_login')
                    }
                
                # Load IP restrictions
                self.ip_whitelist = set(config.get('ip_whitelist', []))
                self.ip_blacklist = set(config.get('ip_blacklist', []))
                
                logger.info(f"📋 Loaded access control config: {len(self.users)} users")
            else:
                # Create default configuration
                self._create_default_config()
                
        except Exception as e:
            logger.error(f"❌ Failed to load access control config: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default access control configuration."""
        try:
            # Create default admin user
            admin_password = os.getenv('DIPMASTER_ADMIN_PASSWORD', 'admin123!')
            admin_password_hash = self._hash_password(admin_password)
            
            default_config = {
                'version': '1.0',
                'users': {
                    'admin': {
                        'password_hash': admin_password_hash,
                        'roles': ['admin'],
                        'enabled': True,
                        'ip_restrictions': [],
                        'created_at': datetime.now().isoformat()
                    },
                    'trader': {
                        'password_hash': self._hash_password('trader123!'),
                        'roles': ['trader'],
                        'enabled': True,
                        'ip_restrictions': [],
                        'created_at': datetime.now().isoformat()
                    }
                },
                'ip_whitelist': [],
                'ip_blacklist': [],
                'settings': {
                    'session_timeout_minutes': self.session_timeout,
                    'max_failed_attempts': 5,
                    'lockout_duration_minutes': 30
                }
            }
            
            # Save configuration
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            
            # Set secure permissions
            os.chmod(self.config_path, 0o600)
            
            logger.info("📁 Created default access control configuration")
            
            # Reload configuration
            self._load_config()
            
        except Exception as e:
            logger.error(f"❌ Failed to create default config: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt."""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash."""
        try:
            salt, stored_hash = password_hash.split(':', 1)
            computed_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return secrets.compare_digest(computed_hash.hex(), stored_hash)
        except Exception:
            return False
    
    def _session_cleanup_worker(self):
        """Background worker to cleanup expired sessions."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"❌ Session cleanup error: {e}")
    
    def _cleanup_expired_sessions(self):
        """Remove expired sessions."""
        with self._lock:
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                if session.is_expired(self.session_timeout):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                session = self.active_sessions.pop(session_id)
                self.audit_logger.log_access_attempt(
                    'session', 'EXPIRE', 'SUCCESS',
                    user_id=session.user_id,
                    metadata={'session_id': session_id}
                )
                logger.info(f"⏰ Expired session for user: {session.user_id}")
    
    def authenticate(self, user_id: str, password: str, ip_address: Optional[str] = None) -> Optional[str]:
        """
        Authenticate user and create session.
        
        Args:
            user_id: User identifier
            password: User password
            ip_address: Client IP address
            
        Returns:
            Session ID if successful, None if failed
        """
        try:
            # Check if user exists and is enabled
            if user_id not in self.users:
                self.audit_logger.log_access_attempt(
                    'authentication', 'LOGIN', 'FAILURE',
                    user_id=user_id, ip_address=ip_address,
                    metadata={'reason': 'user_not_found'}
                )
                return None
            
            user = self.users[user_id]
            if not user.get('enabled', False):
                self.audit_logger.log_access_attempt(
                    'authentication', 'LOGIN', 'FAILURE',
                    user_id=user_id, ip_address=ip_address,
                    metadata={'reason': 'user_disabled'}
                )
                return None
            
            # Check IP restrictions
            if not self._check_ip_access(ip_address, user.get('ip_restrictions', [])):
                self.audit_logger.log_access_attempt(
                    'authentication', 'LOGIN', 'FAILURE',
                    user_id=user_id, ip_address=ip_address,
                    metadata={'reason': 'ip_restricted'}
                )
                return None
            
            # Verify password
            if not self._verify_password(password, user['password_hash']):
                self.audit_logger.log_access_attempt(
                    'authentication', 'LOGIN', 'FAILURE',
                    user_id=user_id, ip_address=ip_address,
                    metadata={'reason': 'invalid_password'}
                )
                return None
            
            # Create session
            with self._lock:
                session = AccessSession(user_id, user['roles'], ip_address)
                self.active_sessions[session.session_id] = session
                
                # Update user last login
                self.users[user_id]['last_login'] = datetime.now().isoformat()
            
            self.audit_logger.log_access_attempt(
                'authentication', 'LOGIN', 'SUCCESS',
                user_id=user_id, ip_address=ip_address,
                metadata={'session_id': session.session_id}
            )
            
            logger.info(f"✅ User authenticated: {user_id}")
            return session.session_id
            
        except Exception as e:
            logger.error(f"❌ Authentication error for {user_id}: {e}")
            self.audit_logger.log_access_attempt(
                'authentication', 'LOGIN', 'FAILURE',
                user_id=user_id, ip_address=ip_address,
                metadata={'reason': 'system_error', 'error': str(e)}
            )
            return None
    
    def _check_ip_access(self, ip_address: Optional[str], user_restrictions: List[str]) -> bool:
        """Check if IP address is allowed access."""
        if not ip_address:
            return True  # Allow if no IP provided (local access)
        
        # Check blacklist first
        if ip_address in self.ip_blacklist:
            return False
        
        # Check global whitelist (if configured)
        if self.ip_whitelist and ip_address not in self.ip_whitelist:
            return False
        
        # Check user-specific restrictions
        if user_restrictions and ip_address not in user_restrictions:
            return False
        
        return True
    
    def check_permission(self, session_id: str, permission: Permission, 
                        resource: Optional[str] = None) -> bool:
        """
        Check if session has permission for specific operation.
        
        Args:
            session_id: Session identifier
            permission: Required permission
            resource: Optional resource identifier
            
        Returns:
            True if permission granted
        """
        try:
            with self._lock:
                # Get session
                session = self.active_sessions.get(session_id)
                if not session:
                    self.audit_logger.log_access_attempt(
                        resource or 'unknown', 'CHECK_PERMISSION', 'FAILURE',
                        metadata={'reason': 'session_not_found', 'permission': permission.value}
                    )
                    return False
                
                # Check if session expired
                if session.is_expired(self.session_timeout):
                    self.active_sessions.pop(session_id, None)
                    self.audit_logger.log_access_attempt(
                        resource or 'unknown', 'CHECK_PERMISSION', 'FAILURE',
                        user_id=session.user_id,
                        metadata={'reason': 'session_expired', 'permission': permission.value}
                    )
                    return False
                
                # Update activity
                session.refresh_activity()
                
                # Check permission
                has_permission = session.has_permission(permission)
                
                result = 'SUCCESS' if has_permission else 'DENIED'
                self.audit_logger.log_access_attempt(
                    resource or 'unknown', 'CHECK_PERMISSION', result,
                    user_id=session.user_id,
                    metadata={'permission': permission.value, 'session_id': session_id}
                )
                
                return has_permission
                
        except Exception as e:
            logger.error(f"❌ Permission check error: {e}")
            self.audit_logger.log_access_attempt(
                resource or 'unknown', 'CHECK_PERMISSION', 'FAILURE',
                metadata={'reason': 'system_error', 'error': str(e)}
            )
            return False
    
    def logout(self, session_id: str):
        """Logout user and destroy session."""
        try:
            with self._lock:
                session = self.active_sessions.pop(session_id, None)
                if session:
                    self.audit_logger.log_access_attempt(
                        'authentication', 'LOGOUT', 'SUCCESS',
                        user_id=session.user_id,
                        metadata={'session_id': session_id}
                    )
                    logger.info(f"👋 User logged out: {session.user_id}")
                    return True
                return False
                
        except Exception as e:
            logger.error(f"❌ Logout error: {e}")
            return False
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        try:
            session = self.active_sessions.get(session_id)
            if not session or session.is_expired(self.session_timeout):
                return None
            
            return {
                'user_id': session.user_id,
                'roles': [role.value for role in session.roles],
                'permissions': [perm.value for perm in session.permissions],
                'ip_address': session.ip_address,
                'created_at': session.created_at.isoformat(),
                'last_activity': session.last_activity.isoformat(),
                'session_id': session.session_id
            }
            
        except Exception as e:
            logger.error(f"❌ Session info error: {e}")
            return None
    
    def create_user(self, user_id: str, password: str, roles: List[Role],
                   enabled: bool = True, ip_restrictions: List[str] = None) -> bool:
        """
        Create new user.
        
        Args:
            user_id: User identifier
            password: User password
            roles: User roles
            enabled: User enabled status
            ip_restrictions: IP address restrictions
            
        Returns:
            True if user created successfully
        """
        try:
            if user_id in self.users:
                logger.warning(f"⚠️  User already exists: {user_id}")
                return False
            
            password_hash = self._hash_password(password)
            
            self.users[user_id] = {
                'password_hash': password_hash,
                'roles': set(roles),
                'enabled': enabled,
                'ip_restrictions': ip_restrictions or [],
                'created_at': datetime.now().isoformat(),
                'last_login': None
            }
            
            self.audit_logger.log_system_event(
                'USER_CREATE',
                {'user_id': user_id, 'roles': [r.value for r in roles]},
                'INFO'
            )
            
            logger.info(f"👤 Created user: {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ User creation error: {e}")
            return False
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        sessions = []
        with self._lock:
            for session_id, session in self.active_sessions.items():
                if not session.is_expired(self.session_timeout):
                    sessions.append({
                        'session_id': session_id,
                        'user_id': session.user_id,
                        'ip_address': session.ip_address,
                        'created_at': session.created_at.isoformat(),
                        'last_activity': session.last_activity.isoformat()
                    })
        return sessions


# Decorator for permission checking
def require_permission(permission: Permission, resource: str = None):
    """Decorator to check permissions before function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Extract session_id from kwargs or first arg
            session_id = kwargs.get('session_id') or (args[0] if args else None)
            
            if not session_id:
                raise PermissionError("No session ID provided")
            
            # Get access controller from global context or create one
            # In production, inject this dependency properly
            controller = AccessController()
            
            if not controller.check_permission(session_id, permission, resource):
                raise PermissionError(f"Permission denied: {permission.value}")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def create_access_controller(config_path: Optional[str] = None) -> AccessController:
    """Factory function to create access controller."""
    return AccessController(config_path)