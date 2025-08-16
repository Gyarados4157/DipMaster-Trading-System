#!/usr/bin/env python3
"""
API Key Manager for DipMaster Trading System
API密钥管理器 - 安全的密钥存储和生命周期管理

Features:
- Encrypted storage of API keys
- Key rotation and lifecycle management  
- Multiple exchange support
- Secure key retrieval with access logging
- Backup and recovery capabilities
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging

from .crypto_manager import CryptoManager
from .audit_logger import SecurityAuditLogger

logger = logging.getLogger(__name__)


class ApiKeyManager:
    """
    Secure API key management system with encryption and rotation.
    
    Provides enterprise-grade API key storage, retrieval, and lifecycle
    management with full audit logging and security controls.
    """
    
    def __init__(self, 
                 storage_path: Optional[str] = None,
                 crypto_manager: Optional[CryptoManager] = None,
                 audit_logger: Optional[SecurityAuditLogger] = None):
        """
        Initialize API key manager.
        
        Args:
            storage_path: Path to encrypted key storage file
            crypto_manager: Cryptographic manager instance
            audit_logger: Security audit logger instance
        """
        self.storage_path = Path(storage_path or "config/encrypted_keys.json")
        self.crypto_manager = crypto_manager or CryptoManager()
        self.audit_logger = audit_logger or SecurityAuditLogger()
        
        # Create storage directory if not exists
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread lock for concurrent access
        self._lock = threading.RLock()
        
        # Cache for decrypted keys (memory only, with TTL)
        self._key_cache = {}
        self._cache_ttl = 300  # 5 minutes
        
        # Initialize storage file if not exists
        self._initialize_storage()
        
        logger.info(f"🔐 ApiKeyManager initialized with storage: {self.storage_path}")
    
    def _initialize_storage(self):
        """Initialize encrypted storage file if it doesn't exist."""
        if not self.storage_path.exists():
            initial_data = {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'last_modified': datetime.now().isoformat(),
                'keys': {},
                'metadata': {
                    'key_count': 0,
                    'last_rotation': None,
                    'backup_count': 0
                }
            }
            self._save_storage(initial_data)
            logger.info("📁 Initialized new encrypted key storage")
    
    def _load_storage(self) -> Dict[str, Any]:
        """Load and decrypt storage file."""
        try:
            with open(self.storage_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self._initialize_storage()
            return self._load_storage()
        except Exception as e:
            logger.error(f"❌ Failed to load key storage: {e}")
            raise
    
    def _save_storage(self, data: Dict[str, Any]):
        """Save storage file."""
        try:
            data['last_modified'] = datetime.now().isoformat()
            
            # Atomic write using temporary file
            temp_path = self.storage_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            # Atomic move
            temp_path.replace(self.storage_path)
            
            # Set secure file permissions (owner read/write only)
            os.chmod(self.storage_path, 0o600)
            
        except Exception as e:
            logger.error(f"❌ Failed to save key storage: {e}")
            raise
    
    def store_api_key(self, 
                      key_id: str,
                      api_key: str, 
                      api_secret: str,
                      exchange: str = "binance",
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store API key with encryption.
        
        Args:
            key_id: Unique identifier for the key pair
            api_key: API key to store
            api_secret: API secret to store
            exchange: Exchange name
            metadata: Additional metadata
            
        Returns:
            True if successfully stored
        """
        with self._lock:
            try:
                # Validate inputs
                if not key_id or not api_key or not api_secret:
                    raise ValueError("key_id, api_key, and api_secret are required")
                
                # Encrypt credentials
                encrypted_creds = self.crypto_manager.encrypt_api_credentials(
                    api_key, api_secret, exchange
                )
                
                # Load current storage
                storage = self._load_storage()
                
                # Prepare key entry
                key_entry = {
                    'encrypted_credentials': encrypted_creds,
                    'exchange': exchange,
                    'created_at': datetime.now().isoformat(),
                    'last_used': None,
                    'use_count': 0,
                    'metadata': metadata or {},
                    'status': 'active',
                    'expires_at': None
                }
                
                # Check if key already exists
                if key_id in storage['keys']:
                    logger.warning(f"⚠️  Overwriting existing key: {key_id}")
                    self.audit_logger.log_key_operation('KEY_UPDATE', key_id, exchange)
                else:
                    self.audit_logger.log_key_operation('KEY_CREATE', key_id, exchange)
                
                # Store encrypted key
                storage['keys'][key_id] = key_entry
                storage['metadata']['key_count'] = len(storage['keys'])
                
                # Save storage
                self._save_storage(storage)
                
                # Clear cache for this key
                self._invalidate_cache(key_id)
                
                logger.info(f"🔐 Successfully stored API key: {key_id} ({exchange})")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to store API key {key_id}: {e}")
                self.audit_logger.log_key_operation('KEY_STORE_ERROR', key_id, exchange, str(e))
                return False
    
    def get_api_key(self, key_id: str) -> Optional[Tuple[str, str]]:
        """
        Retrieve and decrypt API key.
        
        Args:
            key_id: Key identifier
            
        Returns:
            Tuple of (api_key, api_secret) or None if not found
        """
        with self._lock:
            try:
                # Check cache first
                cached_key = self._get_from_cache(key_id)
                if cached_key:
                    return cached_key
                
                # Load storage
                storage = self._load_storage()
                
                if key_id not in storage['keys']:
                    logger.warning(f"⚠️  API key not found: {key_id}")
                    return None
                
                key_entry = storage['keys'][key_id]
                
                # Check if key is active
                if key_entry.get('status') != 'active':
                    logger.warning(f"⚠️  API key is not active: {key_id}")
                    return None
                
                # Check expiration
                if key_entry.get('expires_at'):
                    expires_at = datetime.fromisoformat(key_entry['expires_at'])
                    if datetime.now() > expires_at:
                        logger.warning(f"⚠️  API key has expired: {key_id}")
                        return None
                
                # Decrypt credentials
                encrypted_creds = key_entry['encrypted_credentials']
                api_key, api_secret = self.crypto_manager.decrypt_api_credentials(encrypted_creds)
                
                # Update usage statistics
                key_entry['last_used'] = datetime.now().isoformat()
                key_entry['use_count'] = key_entry.get('use_count', 0) + 1
                self._save_storage(storage)
                
                # Cache the decrypted key
                self._cache_key(key_id, (api_key, api_secret))
                
                # Log access
                exchange = key_entry.get('exchange', 'unknown')
                self.audit_logger.log_key_operation('KEY_ACCESS', key_id, exchange)
                
                logger.info(f"🔓 Retrieved API key: {key_id}")
                return (api_key, api_secret)
                
            except Exception as e:
                logger.error(f"❌ Failed to retrieve API key {key_id}: {e}")
                self.audit_logger.log_key_operation('KEY_ACCESS_ERROR', key_id, 'unknown', str(e))
                return None
    
    def list_keys(self, exchange: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List stored API keys (metadata only).
        
        Args:
            exchange: Filter by exchange (optional)
            
        Returns:
            List of key metadata
        """
        try:
            storage = self._load_storage()
            keys = []
            
            for key_id, key_entry in storage['keys'].items():
                if exchange and key_entry.get('exchange') != exchange:
                    continue
                
                # Return metadata only (no encrypted data)
                key_info = {
                    'key_id': key_id,
                    'exchange': key_entry.get('exchange'),
                    'status': key_entry.get('status'),
                    'created_at': key_entry.get('created_at'),
                    'last_used': key_entry.get('last_used'),
                    'use_count': key_entry.get('use_count', 0),
                    'expires_at': key_entry.get('expires_at'),
                    'metadata': key_entry.get('metadata', {})
                }
                keys.append(key_info)
            
            return keys
            
        except Exception as e:
            logger.error(f"❌ Failed to list keys: {e}")
            return []
    
    def delete_key(self, key_id: str) -> bool:
        """
        Delete API key.
        
        Args:
            key_id: Key identifier
            
        Returns:
            True if successfully deleted
        """
        with self._lock:
            try:
                storage = self._load_storage()
                
                if key_id not in storage['keys']:
                    logger.warning(f"⚠️  Key not found for deletion: {key_id}")
                    return False
                
                # Get exchange info for logging
                exchange = storage['keys'][key_id].get('exchange', 'unknown')
                
                # Remove key
                del storage['keys'][key_id]
                storage['metadata']['key_count'] = len(storage['keys'])
                
                # Save storage
                self._save_storage(storage)
                
                # Clear cache
                self._invalidate_cache(key_id)
                
                # Log deletion
                self.audit_logger.log_key_operation('KEY_DELETE', key_id, exchange)
                
                logger.info(f"🗑️  Deleted API key: {key_id}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to delete key {key_id}: {e}")
                return False
    
    def rotate_key(self, key_id: str, new_api_key: str, new_api_secret: str) -> bool:
        """
        Rotate API key with new credentials.
        
        Args:
            key_id: Key identifier
            new_api_key: New API key
            new_api_secret: New API secret
            
        Returns:
            True if successfully rotated
        """
        with self._lock:
            try:
                storage = self._load_storage()
                
                if key_id not in storage['keys']:
                    logger.error(f"❌ Key not found for rotation: {key_id}")
                    return False
                
                key_entry = storage['keys'][key_id]
                exchange = key_entry.get('exchange', 'binance')
                
                # Encrypt new credentials
                encrypted_creds = self.crypto_manager.encrypt_api_credentials(
                    new_api_key, new_api_secret, exchange
                )
                
                # Update key entry
                key_entry['encrypted_credentials'] = encrypted_creds
                key_entry['rotated_at'] = datetime.now().isoformat()
                key_entry['rotation_count'] = key_entry.get('rotation_count', 0) + 1
                
                # Save storage
                self._save_storage(storage)
                
                # Clear cache
                self._invalidate_cache(key_id)
                
                # Log rotation
                self.audit_logger.log_key_operation('KEY_ROTATE', key_id, exchange)
                
                logger.info(f"🔄 Rotated API key: {key_id}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to rotate key {key_id}: {e}")
                return False
    
    def _get_from_cache(self, key_id: str) -> Optional[Tuple[str, str]]:
        """Get key from memory cache if valid."""
        if key_id in self._key_cache:
            cached_data, cached_time = self._key_cache[key_id]
            if time.time() - cached_time < self._cache_ttl:
                return cached_data
            else:
                # Cache expired
                del self._key_cache[key_id]
        return None
    
    def _cache_key(self, key_id: str, credentials: Tuple[str, str]):
        """Cache decrypted key in memory."""
        self._key_cache[key_id] = (credentials, time.time())
    
    def _invalidate_cache(self, key_id: str):
        """Remove key from cache."""
        if key_id in self._key_cache:
            del self._key_cache[key_id]
    
    def backup_keys(self, backup_path: Optional[str] = None) -> bool:
        """
        Create backup of encrypted keys.
        
        Args:
            backup_path: Backup file path
            
        Returns:
            True if backup successful
        """
        try:
            backup_path = backup_path or f"backups/keys_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            backup_file = Path(backup_path)
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy encrypted storage file
            storage = self._load_storage()
            
            # Add backup metadata
            backup_data = {
                **storage,
                'backup_info': {
                    'created_at': datetime.now().isoformat(),
                    'original_path': str(self.storage_path),
                    'backup_version': '1.0'
                }
            }
            
            with open(backup_file, 'w') as f:
                json.dump(backup_data, f, indent=2)
            
            logger.info(f"💾 API keys backed up to: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False


def create_key_manager(storage_path: Optional[str] = None) -> ApiKeyManager:
    """Factory function to create API key manager."""
    return ApiKeyManager(storage_path)