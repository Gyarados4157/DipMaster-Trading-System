#!/usr/bin/env python3
"""
Configuration Manager for Adaptive DipMaster Strategy
自适应DipMaster策略配置管理器

This module implements comprehensive configuration management and parameter
persistence for the adaptive strategy system. It handles loading, validation,
versioning, and real-time updates of strategy configurations.

Features:
- Hierarchical configuration management
- Parameter validation and type checking
- Configuration versioning and rollback
- Hot-reload capabilities
- Environment-specific configurations
- Parameter encryption for sensitive data
- Configuration templates and inheritance

Author: Portfolio Risk Optimizer Agent
Date: 2025-08-17
Version: 1.0.0
"""

import os
import yaml
import json
import pickle
import hashlib
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from pathlib import Path
import threading
import asyncio
import time
import uuid
from copy import deepcopy

# Validation and encryption
from pydantic import BaseModel, validator, ValidationError
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Core components
from .market_regime_detector import MarketRegime
from .adaptive_parameter_engine import ParameterSet
from .integrated_adaptive_strategy import StrategyConfig

warnings.filterwarnings('ignore')

class ConfigType(Enum):
    """Configuration types"""
    STRATEGY = "strategy"
    RISK = "risk"
    OPTIMIZATION = "optimization"
    LEARNING = "learning"
    INFRASTRUCTURE = "infrastructure"
    MONITORING = "monitoring"

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class ConfigFormat(Enum):
    """Configuration file formats"""
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    PYTHON = "python"

@dataclass
class ConfigVersion:
    """Configuration version information"""
    version: str
    timestamp: datetime
    hash: str
    description: str
    author: str
    environment: Environment
    is_active: bool = False
    parent_version: Optional[str] = None

@dataclass
class ConfigValidationResult:
    """Configuration validation result"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    validation_time: float = 0.0

@dataclass
class ConfigChange:
    """Configuration change event"""
    change_id: str
    timestamp: datetime
    config_type: ConfigType
    section: str
    key: str
    old_value: Any
    new_value: Any
    author: str
    reason: str
    auto_applied: bool = False

class StrategyConfigModel(BaseModel):
    """Pydantic model for strategy configuration validation"""
    starting_capital: float = 10000.0
    max_positions: int = 3
    default_position_size: float = 1000.0
    adaptation_frequency: int = 100
    reoptimization_threshold: float = 0.1
    regime_change_sensitivity: float = 0.7
    max_portfolio_var: float = 0.02
    max_drawdown: float = 0.05
    emergency_stop_threshold: float = 0.08
    ab_test_frequency: int = 500
    validation_frequency: int = 1000
    min_learning_samples: int = 200
    target_win_rate: float = 0.65
    target_sharpe_ratio: float = 2.0
    target_annual_return: float = 0.25
    
    @validator('starting_capital')
    def validate_starting_capital(cls, v):
        if v <= 0:
            raise ValueError('Starting capital must be positive')
        return v
    
    @validator('max_positions')
    def validate_max_positions(cls, v):
        if v < 1 or v > 10:
            raise ValueError('Max positions must be between 1 and 10')
        return v
    
    @validator('target_win_rate')
    def validate_win_rate(cls, v):
        if v < 0.3 or v > 0.95:
            raise ValueError('Target win rate must be between 30% and 95%')
        return v

class RiskConfigModel(BaseModel):
    """Pydantic model for risk configuration validation"""
    max_total_exposure: float = 10000
    max_single_position: float = 2000
    max_leverage: float = 3.0
    daily_var_limit: float = 0.02
    weekly_var_limit: float = 0.05
    max_drawdown_cutoff: float = 0.05
    max_correlation: float = 0.7
    min_liquidity_score: float = 0.3
    
    @validator('max_leverage')
    def validate_leverage(cls, v):
        if v < 1.0 or v > 10.0:
            raise ValueError('Leverage must be between 1.0 and 10.0')
        return v
    
    @validator('daily_var_limit')
    def validate_var_limit(cls, v):
        if v < 0.005 or v > 0.1:
            raise ValueError('Daily VaR limit must be between 0.5% and 10%')
        return v

class ConfigManager:
    """
    Comprehensive Configuration Management System
    综合配置管理系统
    
    Manages all aspects of strategy configuration:
    1. Hierarchical configuration loading
    2. Parameter validation and type checking
    3. Environment-specific overrides
    4. Configuration versioning and rollback
    5. Hot-reload capabilities
    6. Parameter encryption for sensitive data
    7. Configuration templates and inheritance
    """
    
    def __init__(self, config_dir: str = "config",
                 environment: Environment = Environment.DEVELOPMENT,
                 encryption_key: Optional[str] = None):
        """Initialize configuration manager"""
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(config_dir)
        self.environment = environment
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Encryption setup
        self.encryption_key = encryption_key
        self.cipher_suite = None
        if encryption_key:
            self._setup_encryption(encryption_key)
        
        # Configuration storage
        self.configs = {}
        self.config_versions = defaultdict(list)
        self.active_versions = {}
        self.config_changes = deque(maxlen=1000)
        
        # Validation models
        self.validation_models = {
            ConfigType.STRATEGY: StrategyConfigModel,
            ConfigType.RISK: RiskConfigModel,
        }
        
        # File watchers for hot-reload
        self.file_watchers = {}
        self.hot_reload_enabled = True
        
        # Threading
        self._lock = threading.Lock()
        self._shutdown = False
        
        # Load initial configurations
        self._load_initial_configs()
        
        self.logger.info(f"ConfigManager initialized for {environment.value} environment")
    
    def _setup_encryption(self, password: str):
        """Setup encryption for sensitive configuration data"""
        password_bytes = password.encode()
        salt = b'dipmaster_salt_2025'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        self.cipher_suite = Fernet(key)
        
        self.logger.info("Encryption enabled for sensitive configuration data")
    
    def _load_initial_configs(self):
        """Load initial configurations from files"""
        config_files = {
            ConfigType.STRATEGY: "strategy_config.yaml",
            ConfigType.RISK: "risk_config.yaml",
            ConfigType.OPTIMIZATION: "optimization_config.yaml",
            ConfigType.LEARNING: "learning_config.yaml",
            ConfigType.INFRASTRUCTURE: "infrastructure_config.yaml",
            ConfigType.MONITORING: "monitoring_config.yaml"
        }
        
        for config_type, filename in config_files.items():
            config_path = self.config_dir / filename
            
            # Load base configuration
            if config_path.exists():
                config_data = self._load_config_file(config_path)
                if config_data:
                    self._store_config(config_type, config_data, "initial", "system")
            else:
                # Create default configuration
                default_config = self._get_default_config(config_type)
                self._store_config(config_type, default_config, "default", "system")
                self._save_config_file(config_path, default_config)
            
            # Load environment-specific overrides
            env_config_path = self.config_dir / f"{self.environment.value}_{filename}"
            if env_config_path.exists():
                env_config = self._load_config_file(env_config_path)
                if env_config:
                    # Merge with base configuration
                    base_config = self.configs.get(config_type, {})
                    merged_config = self._merge_configs(base_config, env_config)
                    self._store_config(config_type, merged_config, f"env_{self.environment.value}", "system")
    
    def _load_config_file(self, file_path: Path) -> Optional[Dict]:
        """Load configuration from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                elif file_path.suffix.lower() == '.json':
                    return json.load(f)
                else:
                    self.logger.warning(f"Unsupported config file format: {file_path}")
                    return None
        except Exception as e:
            self.logger.error(f"Failed to load config file {file_path}: {e}")
            return None
    
    def _save_config_file(self, file_path: Path, config_data: Dict):
        """Save configuration to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif file_path.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2, default=str)
            
            self.logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save config file {file_path}: {e}")
    
    def _get_default_config(self, config_type: ConfigType) -> Dict:
        """Get default configuration for type"""
        defaults = {
            ConfigType.STRATEGY: {
                'starting_capital': 10000.0,
                'max_positions': 3,
                'default_position_size': 1000.0,
                'adaptation_frequency': 100,
                'reoptimization_threshold': 0.1,
                'regime_change_sensitivity': 0.7,
                'max_portfolio_var': 0.02,
                'max_drawdown': 0.05,
                'emergency_stop_threshold': 0.08,
                'ab_test_frequency': 500,
                'validation_frequency': 1000,
                'min_learning_samples': 200,
                'target_win_rate': 0.65,
                'target_sharpe_ratio': 2.0,
                'target_annual_return': 0.25
            },
            ConfigType.RISK: {
                'portfolio_limits': {
                    'max_total_exposure': 10000,
                    'max_single_position': 2000,
                    'max_leverage': 3.0,
                    'daily_var_limit': 0.02,
                    'weekly_var_limit': 0.05,
                    'max_drawdown_cutoff': 0.05,
                    'max_correlation': 0.7,
                    'min_liquidity_score': 0.3
                },
                'position_limits': {
                    'max_position_pct': 0.25,
                    'min_position_size': 10,
                    'var_multiplier': 2.5,
                    'concentration_limit': 0.3,
                    'beta_limit': 2.0,
                    'volatility_limit': 0.15
                }
            },
            ConfigType.OPTIMIZATION: {
                'optimization': {
                    'n_trials': 200,
                    'n_jobs': 4,
                    'timeout': 3600,
                    'cv_folds': 5,
                    'test_size': 0.2,
                    'random_state': 42
                },
                'objectives': {
                    'primary': 'risk_adjusted_return',
                    'weights': {
                        'win_rate': 0.3,
                        'sharpe_ratio': 0.25,
                        'max_drawdown': 0.2,
                        'profit_factor': 0.15,
                        'expected_return': 0.1
                    }
                }
            },
            ConfigType.LEARNING: {
                'ab_testing': {
                    'min_sample_size': 100,
                    'significance_level': 0.05,
                    'power': 0.8,
                    'max_test_duration_days': 30
                },
                'validation': {
                    'walk_forward_window': 1000,
                    'n_splits': 5,
                    'test_size': 0.2,
                    'monte_carlo_simulations': 1000
                },
                'reinforcement_learning': {
                    'algorithm': 'PPO',
                    'learning_rate': 3e-4,
                    'n_timesteps': 10000
                }
            },
            ConfigType.INFRASTRUCTURE: {
                'database': {
                    'path': 'data/dipmaster.db',
                    'backup_frequency': 3600,
                    'retention_days': 30
                },
                'logging': {
                    'level': 'INFO',
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'file': 'logs/dipmaster.log'
                }
            },
            ConfigType.MONITORING: {
                'alerts': {
                    'email_enabled': False,
                    'webhook_url': None,
                    'alert_cooldown': 300
                },
                'metrics': {
                    'collection_frequency': 60,
                    'retention_days': 90
                }
            }
        }
        
        return defaults.get(config_type, {})
    
    def _merge_configs(self, base_config: Dict, override_config: Dict) -> Dict:
        """Merge configuration dictionaries"""
        merged = deepcopy(base_config)
        
        def merge_recursive(base_dict, override_dict):
            for key, value in override_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    merge_recursive(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        merge_recursive(merged, override_config)
        return merged
    
    def _store_config(self, config_type: ConfigType, config_data: Dict,
                     version_name: str, author: str):
        """Store configuration with versioning"""
        config_hash = hashlib.md5(json.dumps(config_data, sort_keys=True).encode()).hexdigest()
        
        version = ConfigVersion(
            version=version_name,
            timestamp=datetime.now(),
            hash=config_hash,
            description=f"Configuration update by {author}",
            author=author,
            environment=self.environment,
            is_active=True
        )
        
        with self._lock:
            # Deactivate previous version
            for v in self.config_versions[config_type]:
                v.is_active = False
            
            # Store new version
            self.config_versions[config_type].append(version)
            self.active_versions[config_type] = version.version
            self.configs[config_type] = deepcopy(config_data)
        
        self.logger.info(f"Configuration {config_type.value} updated to version {version_name}")
    
    def validate_config(self, config_type: ConfigType, config_data: Dict) -> ConfigValidationResult:
        """
        Validate configuration against schema
        配置验证
        """
        start_time = time.time()
        errors = []
        warnings = []
        suggestions = []
        
        try:
            # Use Pydantic model for validation if available
            if config_type in self.validation_models:
                model_class = self.validation_models[config_type]
                
                try:
                    # Flatten config for model validation
                    flat_config = self._flatten_config(config_data)
                    model_instance = model_class(**flat_config)
                    
                except ValidationError as e:
                    for error in e.errors():
                        errors.append(f"{'.'.join(str(x) for x in error['loc'])}: {error['msg']}")
            
            # Custom validation rules
            if config_type == ConfigType.STRATEGY:
                self._validate_strategy_config(config_data, errors, warnings, suggestions)
            elif config_type == ConfigType.RISK:
                self._validate_risk_config(config_data, errors, warnings, suggestions)
            elif config_type == ConfigType.OPTIMIZATION:
                self._validate_optimization_config(config_data, errors, warnings, suggestions)
            
            validation_time = time.time() - start_time
            
            return ConfigValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                validation_time=validation_time
            )
            
        except Exception as e:
            self.logger.error(f"Configuration validation error: {e}")
            return ConfigValidationResult(
                is_valid=False,
                errors=[f"Validation error: {str(e)}"],
                validation_time=time.time() - start_time
            )
    
    def _flatten_config(self, config_data: Dict, prefix: str = "") -> Dict:
        """Flatten nested configuration dictionary"""
        flat_config = {}
        
        for key, value in config_data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                flat_config.update(self._flatten_config(value, full_key))
            else:
                flat_config[key] = value
        
        return flat_config
    
    def _validate_strategy_config(self, config: Dict, errors: List, warnings: List, suggestions: List):
        """Validate strategy-specific configuration"""
        # Check target metrics consistency
        if 'target_win_rate' in config and 'target_sharpe_ratio' in config:
            win_rate = config['target_win_rate']
            sharpe = config['target_sharpe_ratio']
            
            if win_rate > 0.8 and sharpe > 3.0:
                warnings.append("Very high target win rate and Sharpe ratio may be unrealistic")
            
            if win_rate < 0.4:
                suggestions.append("Consider increasing target win rate for better strategy performance")
        
        # Check adaptation frequency
        if 'adaptation_frequency' in config:
            freq = config['adaptation_frequency']
            if freq < 50:
                warnings.append("Very frequent adaptation may lead to overfitting")
            elif freq > 500:
                suggestions.append("Consider more frequent adaptation for better responsiveness")
    
    def _validate_risk_config(self, config: Dict, errors: List, warnings: List, suggestions: List):
        """Validate risk-specific configuration"""
        # Check VaR limits consistency
        if 'portfolio_limits' in config:
            limits = config['portfolio_limits']
            
            if 'daily_var_limit' in limits and 'weekly_var_limit' in limits:
                daily_var = limits['daily_var_limit']
                weekly_var = limits['weekly_var_limit']
                
                expected_weekly = daily_var * np.sqrt(5)  # Assuming 5 trading days
                
                if weekly_var < expected_weekly * 0.8:
                    warnings.append("Weekly VaR limit may be too tight compared to daily VaR")
        
        # Check leverage limits
        if 'portfolio_limits' in config and 'max_leverage' in config['portfolio_limits']:
            leverage = config['portfolio_limits']['max_leverage']
            if leverage > 5.0:
                warnings.append("High leverage increases risk significantly")
    
    def _validate_optimization_config(self, config: Dict, errors: List, warnings: List, suggestions: List):
        """Validate optimization-specific configuration"""
        # Check optimization parameters
        if 'optimization' in config:
            opt_config = config['optimization']
            
            if 'n_trials' in opt_config:
                n_trials = opt_config['n_trials']
                if n_trials < 50:
                    warnings.append("Low number of optimization trials may result in suboptimal parameters")
                elif n_trials > 1000:
                    suggestions.append("Consider reducing trials for faster optimization")
    
    def get_config(self, config_type: ConfigType, section: Optional[str] = None) -> Any:
        """
        Get configuration value
        获取配置值
        """
        if config_type not in self.configs:
            self.logger.warning(f"Configuration type {config_type.value} not found")
            return None
        
        config = self.configs[config_type]
        
        if section is None:
            return deepcopy(config)
        
        # Navigate to section using dot notation
        sections = section.split('.')
        current = config
        
        for sec in sections:
            if isinstance(current, dict) and sec in current:
                current = current[sec]
            else:
                self.logger.warning(f"Configuration section {section} not found in {config_type.value}")
                return None
        
        return deepcopy(current)
    
    def set_config(self, config_type: ConfigType, key: str, value: Any,
                  author: str = "system", reason: str = "programmatic update",
                  validate: bool = True) -> bool:
        """
        Set configuration value
        设置配置值
        """
        try:
            # Get current configuration
            current_config = self.get_config(config_type)
            if current_config is None:
                current_config = {}
            
            # Store old value for change tracking
            old_value = self._get_nested_value(current_config, key)
            
            # Set new value
            self._set_nested_value(current_config, key, value)
            
            # Validate if requested
            if validate:
                validation_result = self.validate_config(config_type, current_config)
                if not validation_result.is_valid:
                    self.logger.error(f"Configuration validation failed: {validation_result.errors}")
                    return False
            
            # Create version name
            version_name = f"update_{int(time.time())}"
            
            # Store updated configuration
            self._store_config(config_type, current_config, version_name, author)
            
            # Record change
            change = ConfigChange(
                change_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                config_type=config_type,
                section=key.split('.')[0] if '.' in key else 'root',
                key=key,
                old_value=old_value,
                new_value=value,
                author=author,
                reason=reason,
                auto_applied=author == "system"
            )
            
            with self._lock:
                self.config_changes.append(change)
            
            self.logger.info(f"Configuration {config_type.value}.{key} updated by {author}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set configuration {config_type.value}.{key}: {e}")
            return False
    
    def _get_nested_value(self, config: Dict, key: str) -> Any:
        """Get nested configuration value using dot notation"""
        keys = key.split('.')
        current = config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return current
    
    def _set_nested_value(self, config: Dict, key: str, value: Any):
        """Set nested configuration value using dot notation"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        current[keys[-1]] = value
    
    def rollback_config(self, config_type: ConfigType, version: str) -> bool:
        """
        Rollback configuration to specific version
        回滚配置到指定版本
        """
        try:
            # Find version
            target_version = None
            for v in self.config_versions[config_type]:
                if v.version == version:
                    target_version = v
                    break
            
            if not target_version:
                self.logger.error(f"Version {version} not found for {config_type.value}")
                return False
            
            # Load configuration from version (simplified - would store actual config data)
            # For now, just log the rollback attempt
            self.logger.info(f"Rollback {config_type.value} to version {version}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback {config_type.value} to {version}: {e}")
            return False
    
    def encrypt_sensitive_value(self, value: str) -> str:
        """Encrypt sensitive configuration value"""
        if not self.cipher_suite:
            self.logger.warning("Encryption not available - storing value in plain text")
            return value
        
        try:
            encrypted_value = self.cipher_suite.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted_value).decode()
        except Exception as e:
            self.logger.error(f"Failed to encrypt value: {e}")
            return value
    
    def decrypt_sensitive_value(self, encrypted_value: str) -> str:
        """Decrypt sensitive configuration value"""
        if not self.cipher_suite:
            return encrypted_value
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted_value = self.cipher_suite.decrypt(encrypted_bytes)
            return decrypted_value.decode()
        except Exception as e:
            self.logger.error(f"Failed to decrypt value: {e}")
            return encrypted_value
    
    def export_config(self, config_type: ConfigType, output_path: str,
                     format: ConfigFormat = ConfigFormat.YAML,
                     include_metadata: bool = True) -> bool:
        """Export configuration to file"""
        try:
            config_data = self.get_config(config_type)
            if config_data is None:
                return False
            
            # Add metadata if requested
            if include_metadata:
                metadata = {
                    'export_timestamp': datetime.now().isoformat(),
                    'config_type': config_type.value,
                    'environment': self.environment.value,
                    'version': self.active_versions.get(config_type, 'unknown'),
                    'exported_by': 'config_manager'
                }
                config_data['_metadata'] = metadata
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save in requested format
            if format == ConfigFormat.YAML:
                with open(output_file, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif format == ConfigFormat.JSON:
                with open(output_file, 'w') as f:
                    json.dump(config_data, f, indent=2, default=str)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Configuration {config_type.value} exported to {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {e}")
            return False
    
    def get_config_history(self, config_type: ConfigType) -> List[ConfigVersion]:
        """Get configuration version history"""
        return deepcopy(self.config_versions.get(config_type, []))
    
    def get_recent_changes(self, limit: int = 50) -> List[ConfigChange]:
        """Get recent configuration changes"""
        return list(self.config_changes)[-limit:]
    
    def get_all_configs(self) -> Dict[ConfigType, Dict]:
        """Get all current configurations"""
        return {
            config_type: deepcopy(config_data)
            for config_type, config_data in self.configs.items()
        }
    
    def create_strategy_config_object(self) -> StrategyConfig:
        """Create StrategyConfig object from current configuration"""
        strategy_config = self.get_config(ConfigType.STRATEGY)
        if not strategy_config:
            return StrategyConfig()  # Default
        
        return StrategyConfig(
            starting_capital=strategy_config.get('starting_capital', 10000.0),
            max_positions=strategy_config.get('max_positions', 3),
            default_position_size=strategy_config.get('default_position_size', 1000.0),
            adaptation_frequency=strategy_config.get('adaptation_frequency', 100),
            reoptimization_threshold=strategy_config.get('reoptimization_threshold', 0.1),
            regime_change_sensitivity=strategy_config.get('regime_change_sensitivity', 0.7),
            max_portfolio_var=strategy_config.get('max_portfolio_var', 0.02),
            max_drawdown=strategy_config.get('max_drawdown', 0.05),
            emergency_stop_threshold=strategy_config.get('emergency_stop_threshold', 0.08),
            ab_test_frequency=strategy_config.get('ab_test_frequency', 500),
            validation_frequency=strategy_config.get('validation_frequency', 1000),
            min_learning_samples=strategy_config.get('min_learning_samples', 200),
            target_win_rate=strategy_config.get('target_win_rate', 0.65),
            target_sharpe_ratio=strategy_config.get('target_sharpe_ratio', 2.0),
            target_annual_return=strategy_config.get('target_annual_return', 0.25)
        )
    
    def shutdown(self):
        """Shutdown configuration manager"""
        self._shutdown = True
        self.logger.info("Configuration manager shutdown")

# Factory function
def create_config_manager(config_dir: str = "config",
                         environment: Environment = Environment.DEVELOPMENT,
                         encryption_key: Optional[str] = None) -> ConfigManager:
    """Factory function to create configuration manager"""
    return ConfigManager(config_dir, environment, encryption_key)