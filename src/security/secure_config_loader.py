#!/usr/bin/env python3
"""
Secure Configuration Loader for DipMaster Trading System
安全配置加载器 - 集成加密密钥管理的配置系统

Features:
- Encrypted API key integration
- Environment variable substitution
- Configuration validation and schema checking
- Secure defaults and best practices
- Runtime configuration updates
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

from .key_manager import ApiKeyManager
from .crypto_manager import CryptoManager
from .audit_logger import SecurityAuditLogger

logger = logging.getLogger(__name__)


class SecureConfigLoader:
    """
    Secure configuration loader with encrypted key support.
    
    Integrates with the key management system to provide secure
    configuration loading with encrypted API credentials.
    """
    
    def __init__(self, 
                 key_manager: Optional[ApiKeyManager] = None,
                 audit_logger: Optional[SecurityAuditLogger] = None):
        """
        Initialize secure config loader.
        
        Args:
            key_manager: API key manager instance
            audit_logger: Security audit logger instance
        """
        self.key_manager = key_manager or ApiKeyManager()
        self.audit_logger = audit_logger or SecurityAuditLogger()
        
        # Configuration schema for validation
        self.config_schema = self._get_config_schema()
        
        logger.info("🔧 SecureConfigLoader initialized")
    
    def _get_config_schema(self) -> Dict[str, Any]:
        """Define configuration schema for validation."""
        return {
            'required_sections': ['api', 'trading', 'risk_management'],
            'required_fields': {
                'api': ['key_id'],  # Changed from api_key/api_secret to key_id
                'trading': ['symbols', 'paper_trading'],
                'risk_management': ['max_daily_trades', 'emergency_stop_loss']
            },
            'field_types': {
                'trading.paper_trading': bool,
                'trading.max_positions': int,
                'trading.max_position_size': (int, float),
                'api.timeout': int,
                'risk_management.max_daily_trades': int,
                'websocket.enabled': bool,
                'dashboard.enabled': bool,
                'performance.cache_enabled': bool
            },
            'secure_fields': ['api.key_id', 'api.api_key', 'api.api_secret']
        }
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load and process configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Processed configuration dictionary
        """
        try:
            config_path = Path(config_path)
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            # Load raw configuration
            with open(config_path, 'r', encoding='utf-8') as f:
                raw_config = json.load(f)
            
            # Process configuration
            processed_config = self._process_config(raw_config)
            
            # Validate configuration
            self._validate_config(processed_config)
            
            # Log configuration load
            self.audit_logger.log_system_event(
                'CONFIG_LOADED',
                {
                    'config_path': str(config_path),
                    'sections': list(processed_config.keys()),
                    'api_key_method': self._get_api_key_method(processed_config)
                }
            )
            
            logger.info(f"✅ Configuration loaded successfully: {config_path}")
            return processed_config
            
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            self.audit_logger.log_system_event(
                'CONFIG_LOAD_ERROR',
                {'config_path': str(config_path), 'error': str(e)},
                'CRITICAL'
            )
            raise
    
    def _process_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw configuration with security enhancements.
        
        Args:
            config: Raw configuration dictionary
            
        Returns:
            Processed configuration
        """
        processed_config = self._deep_copy_dict(config)
        
        # Process environment variable substitutions
        processed_config = self._substitute_environment_variables(processed_config)
        
        # Process encrypted API keys
        processed_config = self._process_api_keys(processed_config)
        
        # Apply secure defaults
        processed_config = self._apply_secure_defaults(processed_config)
        
        # Remove sensitive fields from logs
        self._sanitize_for_logging(processed_config)
        
        return processed_config
    
    def _substitute_environment_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values."""
        def substitute_value(value):
            if isinstance(value, str):
                # Pattern: ${VARIABLE_NAME} or ${VARIABLE_NAME:default_value}
                pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
                
                def replace_env_var(match):
                    var_name = match.group(1)
                    default_value = match.group(2) if match.group(2) is not None else ''
                    return os.getenv(var_name, default_value)
                
                return re.sub(pattern, replace_env_var, value)
            elif isinstance(value, dict):
                return {k: substitute_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [substitute_value(item) for item in value]
            else:
                return value
        
        return substitute_value(config)
    
    def _process_api_keys(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process API key configuration with encrypted storage support."""
        try:
            api_config = config.get('api', {})
            
            # Check if using encrypted key storage (new method)
            if 'key_id' in api_config:
                key_id = api_config['key_id']
                
                if not key_id:
                    raise ValueError("key_id is specified but empty")
                
                # Retrieve encrypted credentials
                credentials = self.key_manager.get_api_key(key_id)
                if not credentials:
                    raise ValueError(f"API key not found: {key_id}")
                
                api_key, api_secret = credentials
                
                # Update config with decrypted credentials
                api_config['api_key'] = api_key
                api_config['api_secret'] = api_secret
                
                # Log successful key retrieval (without revealing keys)
                self.audit_logger.log_key_operation(
                    'KEY_CONFIG_LOAD', key_id,
                    api_config.get('exchange', 'binance')
                )
                
                logger.info(f"🔓 Loaded encrypted API key: {key_id}")
            
            # Legacy support: direct API key in config (less secure)
            elif 'api_key' in api_config and 'api_secret' in api_config:
                api_key = api_config.get('api_key', '')
                api_secret = api_config.get('api_secret', '')
                
                if not api_key or not api_secret:
                    raise ValueError("API key and secret are required")
                
                # Warn about insecure configuration
                logger.warning("⚠️  Using direct API keys in config - consider using encrypted storage")
                
                self.audit_logger.log_system_event(
                    'INSECURE_API_CONFIG',
                    {'method': 'direct_config'},
                    'WARNING'
                )
            
            else:
                raise ValueError("No valid API key configuration found. Specify 'key_id' or 'api_key'+'api_secret'")
            
            return config
            
        except Exception as e:
            logger.error(f"❌ Failed to process API keys: {e}")
            self.audit_logger.log_system_event(
                'API_KEY_PROCESS_ERROR',
                {'error': str(e)},
                'CRITICAL'
            )
            raise
    
    def _apply_secure_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply secure defaults to configuration."""
        secure_defaults = {
            'api': {
                'testnet': False,
                'timeout': 30,
                'rate_limit_buffer': 0.1
            },
            'trading': {
                'paper_trading': True,  # Safe default
                'max_positions': 3,
                'max_position_size': 1000,
                'daily_loss_limit': -500,
                'auto_start': False  # Require manual start
            },
            'risk_management': {
                'enable_risk_limits': True,
                'emergency_stop_loss': -0.05,
                'max_drawdown': -0.20,
                'cool_down_minutes': 15
            },
            'websocket': {
                'reconnect_attempts': 5,
                'reconnect_delay': 5,
                'timeout': 10
            },
            'logging': {
                'level': 'INFO',
                'file_enabled': True,
                'console_enabled': True
            },
            'performance': {
                'cache_enabled': True,
                'cache_ttl': 300,
                'max_workers': 4
            }
        }
        
        # Apply defaults recursively
        def apply_defaults(target: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
            for key, default_value in defaults.items():
                if key not in target:
                    target[key] = default_value
                elif isinstance(default_value, dict) and isinstance(target[key], dict):
                    apply_defaults(target[key], default_value)
            return target
        
        return apply_defaults(config, secure_defaults)
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration against schema."""
        try:
            # Check required sections
            for section in self.config_schema['required_sections']:
                if section not in config:
                    raise ValueError(f"Missing required configuration section: {section}")
            
            # Check required fields
            for section, fields in self.config_schema['required_fields'].items():
                section_config = config.get(section, {})
                for field in fields:
                    if field not in section_config:
                        raise ValueError(f"Missing required field: {section}.{field}")
            
            # Check field types
            for field_path, expected_type in self.config_schema['field_types'].items():
                value = self._get_nested_value(config, field_path)
                if value is not None and not isinstance(value, expected_type):
                    raise ValueError(f"Invalid type for {field_path}: expected {expected_type}, got {type(value)}")
            
            # Validate trading configuration
            self._validate_trading_config(config.get('trading', {}))
            
            # Validate risk management
            self._validate_risk_config(config.get('risk_management', {}))
            
            logger.info("✅ Configuration validation passed")
            
        except Exception as e:
            logger.error(f"❌ Configuration validation failed: {e}")
            raise
    
    def _validate_trading_config(self, trading_config: Dict[str, Any]):
        """Validate trading-specific configuration."""
        # Check symbols list
        symbols = trading_config.get('symbols', [])
        if not isinstance(symbols, list) or len(symbols) == 0:
            raise ValueError("At least one trading symbol must be specified")
        
        # Validate position limits
        max_positions = trading_config.get('max_positions', 0)
        if max_positions <= 0 or max_positions > 10:
            raise ValueError("max_positions must be between 1 and 10")
        
        # Validate position size
        max_position_size = trading_config.get('max_position_size', 0)
        if max_position_size <= 0 or max_position_size > 10000:
            raise ValueError("max_position_size must be between 1 and 10000 USD")
    
    def _validate_risk_config(self, risk_config: Dict[str, Any]):
        """Validate risk management configuration."""
        # Check stop loss
        emergency_stop = risk_config.get('emergency_stop_loss', 0)
        if emergency_stop >= 0 or emergency_stop < -0.50:
            raise ValueError("emergency_stop_loss must be between -50% and 0%")
        
        # Check max drawdown
        max_drawdown = risk_config.get('max_drawdown', 0)
        if max_drawdown >= 0 or max_drawdown < -1.0:
            raise ValueError("max_drawdown must be between -100% and 0%")
    
    def _get_api_key_method(self, config: Dict[str, Any]) -> str:
        """Determine API key configuration method."""
        api_config = config.get('api', {})
        
        if 'key_id' in api_config:
            return 'encrypted_storage'
        elif 'api_key' in api_config:
            return 'direct_config'
        else:
            return 'unknown'
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested dictionary value by dot-notation path."""
        keys = path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        
        return value
    
    def _deep_copy_dict(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """Deep copy dictionary."""
        if isinstance(d, dict):
            return {k: self._deep_copy_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [self._deep_copy_dict(item) for item in d]
        else:
            return d
    
    def _sanitize_for_logging(self, config: Dict[str, Any]):
        """Remove sensitive fields from config for logging."""
        sensitive_fields = ['api_key', 'api_secret', 'password', 'token', 'secret']
        
        def sanitize_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if any(sensitive in key.lower() for sensitive in sensitive_fields):
                        obj[key] = "***REDACTED***"
                    else:
                        sanitize_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    sanitize_recursive(item)
        
        sanitize_recursive(config)
    
    def create_secure_config_template(self, output_path: str) -> bool:
        """
        Create a secure configuration template.
        
        Args:
            output_path: Path to save configuration template
            
        Returns:
            True if template created successfully
        """
        try:
            template = {
                "_comment": "DipMaster Trading System - Secure Configuration Template",
                "_security_notice": "This template uses encrypted API key storage for maximum security",
                "_version": "2.0.0",
                
                "api": {
                    "_comment": "API Configuration - Uses encrypted key storage",
                    "key_id": "binance-production",  # Reference to encrypted key
                    "exchange": "binance",
                    "testnet": False,
                    "timeout": 30,
                    "rate_limit_buffer": 0.1,
                    "_legacy_note": "For legacy support, you can still use api_key/api_secret directly (not recommended)"
                },
                
                "trading": {
                    "_comment": "Trading Configuration",
                    "symbols": [
                        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT",
                        "ADAUSDT", "XRPUSDT", "DOGEUSDT"
                    ],
                    "paper_trading": True,
                    "max_positions": 3,
                    "max_position_size": 1000,
                    "daily_loss_limit": -500,
                    "trading_enabled": True,
                    "auto_start": False
                },
                
                "strategy": {
                    "_comment": "DipMaster strategy parameters",
                    "rsi_entry_range": [30, 50],
                    "rsi_period": 14,
                    "dip_threshold": 0.002,
                    "volume_multiplier": 1.5,
                    "max_holding_minutes": 180,
                    "target_profit": 0.008,
                    "boundary_slots": [15, 30, 45, 60]
                },
                
                "risk_management": {
                    "_comment": "Enhanced risk management",
                    "max_daily_trades": 50,
                    "max_consecutive_losses": 5,
                    "emergency_stop_loss": -0.05,
                    "max_drawdown": -0.20,
                    "enable_risk_limits": True,
                    "position_size_type": "fixed"
                },
                
                "security": {
                    "_comment": "Security configuration",
                    "encrypted_keys": True,
                    "audit_logging": True,
                    "access_control": True,
                    "session_timeout": 480
                },
                
                "monitoring": {
                    "_comment": "Monitoring and alerting",
                    "dashboard_enabled": True,
                    "real_time_alerts": True,
                    "performance_metrics": True,
                    "audit_alerts": True
                }
            }
            
            # Save template
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(template, f, indent=2)
            
            logger.info(f"✅ Secure configuration template created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create config template: {e}")
            return False


def load_secure_config(config_path: Union[str, Path],
                      key_manager: Optional[ApiKeyManager] = None) -> Dict[str, Any]:
    """
    Convenience function to load secure configuration.
    
    Args:
        config_path: Path to configuration file
        key_manager: Optional key manager instance
        
    Returns:
        Processed configuration dictionary
    """
    loader = SecureConfigLoader(key_manager)
    return loader.load_config(config_path)


def create_secure_config_template(output_path: str) -> bool:
    """
    Convenience function to create secure configuration template.
    
    Args:
        output_path: Path to save template
        
    Returns:
        True if successful
    """
    loader = SecureConfigLoader()
    return loader.create_secure_config_template(output_path)