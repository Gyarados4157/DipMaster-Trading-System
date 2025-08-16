#!/usr/bin/env python3
"""
DipMaster Trading System - Unified Configuration Loader
统一配置管理系统，支持环境继承、变量替换和配置验证

Author: DipMaster Development Team
Date: 2025-08-16
Version: 4.0.0
"""

import os
import re
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from enum import Enum
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

class ConfigEnvironment(str, Enum):
    """配置环境枚举"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

@dataclass
class ConfigMetadata:
    """配置元数据"""
    name: str
    version: str
    environment: ConfigEnvironment
    loaded_at: str
    source_files: List[str] = field(default_factory=list)
    validation_errors: List[str] = field(default_factory=list)

class ConfigValidationError(Exception):
    """配置验证错误"""
    pass

class ConfigLoader:
    """统一配置加载器"""
    
    def __init__(self, config_root: Optional[Path] = None):
        """
        初始化配置加载器
        
        Args:
            config_root: 配置根目录，默认为当前目录下的config/unified_config
        """
        if config_root is None:
            config_root = Path(__file__).parent
        self.config_root = Path(config_root)
        self.environment = self._detect_environment()
        self.loaded_configs = {}
        self._env_var_pattern = re.compile(r'\$\{([^}]+)\}')
        
    def _detect_environment(self) -> ConfigEnvironment:
        """自动检测运行环境"""
        env = os.getenv('DIPMASTER_ENV', 'development').lower()
        try:
            return ConfigEnvironment(env)
        except ValueError:
            logger.warning(f"Unknown environment '{env}', defaulting to development")
            return ConfigEnvironment.DEVELOPMENT
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """加载YAML文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {e}")
            return {}
    
    def _substitute_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """递归替换环境变量"""
        if isinstance(config, dict):
            return {k: self._substitute_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_env_var_string(config)
        else:
            return config
    
    def _substitute_env_var_string(self, value: str) -> Union[str, int, float, bool]:
        """替换字符串中的环境变量"""
        def replace_var(match):
            var_expr = match.group(1)
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
            else:
                var_name, default_value = var_expr, None
                
            env_value = os.getenv(var_name.strip(), default_value)
            if env_value is None:
                raise ConfigValidationError(f"Environment variable {var_name} not found and no default provided")
            
            # Try to convert to appropriate type
            return self._convert_type(env_value)
        
        # Replace all environment variable references
        if self._env_var_pattern.search(value):
            try:
                result = self._env_var_pattern.sub(replace_var, value)
                # If the entire string was a single env var, return the converted type
                if value.startswith('${') and value.endswith('}') and value.count('${') == 1:
                    return result
                return str(result)  # Otherwise return as string
            except Exception as e:
                logger.warning(f"Failed to substitute environment variables in '{value}': {e}")
                return value
        return value
    
    def _convert_type(self, value: str) -> Union[str, int, float, bool]:
        """将字符串转换为适当的类型"""
        if not isinstance(value, str):
            return value
            
        # Boolean conversion
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Number conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        return value
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并配置字典"""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def load_global_config(self) -> Dict[str, Any]:
        """加载全局配置"""
        global_path = self.config_root / "global.yaml"
        if not global_path.exists():
            raise ConfigValidationError(f"Global config file not found: {global_path}")
        
        config = self._load_yaml_file(global_path)
        return self._substitute_env_vars(config)
    
    def load_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """加载策略配置"""
        # 首先加载基础策略配置
        base_strategy_path = self.config_root / "strategy" / "base_strategy.yaml"
        base_config = {}
        if base_strategy_path.exists():
            base_config = self._load_yaml_file(base_strategy_path)
        
        # 然后加载特定策略配置
        strategy_path = self.config_root / "strategy" / f"{strategy_name}.yaml"
        if not strategy_path.exists():
            raise ConfigValidationError(f"Strategy config file not found: {strategy_path}")
        
        strategy_config = self._load_yaml_file(strategy_path)
        
        # 合并配置
        merged_config = self._merge_configs(base_config, strategy_config)
        return self._substitute_env_vars(merged_config)
    
    def load_environment_config(self) -> Dict[str, Any]:
        """加载环境特定配置"""
        env_path = self.config_root / "environments" / f"{self.environment.value}.yaml"
        if not env_path.exists():
            logger.warning(f"Environment config file not found: {env_path}")
            return {}
        
        config = self._load_yaml_file(env_path)
        return self._substitute_env_vars(config)
    
    def load_complete_config(self, strategy_name: str = "dipmaster_v4") -> Dict[str, Any]:
        """
        加载完整配置，按以下优先级合并：
        1. 全局配置 (global.yaml)
        2. 基础策略配置 (base_strategy.yaml)
        3. 特定策略配置 (strategy_name.yaml)
        4. 环境特定配置 (environment.yaml)
        """
        logger.info(f"Loading complete configuration for strategy '{strategy_name}' in '{self.environment.value}' environment")
        
        # 按优先级加载配置
        configs = []
        source_files = []
        
        try:
            # 1. 全局配置
            global_config = self.load_global_config()
            configs.append(global_config)
            source_files.append("global.yaml")
            logger.debug("Loaded global configuration")
            
            # 2. 策略配置（包含基础策略继承）
            strategy_config = self.load_strategy_config(strategy_name)
            configs.append(strategy_config)
            source_files.extend(["base_strategy.yaml", f"{strategy_name}.yaml"])
            logger.debug(f"Loaded strategy configuration: {strategy_name}")
            
            # 3. 环境配置
            env_config = self.load_environment_config()
            if env_config:
                configs.append(env_config)
                source_files.append(f"{self.environment.value}.yaml")
                logger.debug(f"Loaded environment configuration: {self.environment.value}")
            
            # 合并所有配置
            merged_config = {}
            for config in configs:
                merged_config = self._merge_configs(merged_config, config)
            
            # 验证配置
            validation_errors = self._validate_config(merged_config)
            if validation_errors:
                logger.warning(f"Configuration validation warnings: {validation_errors}")
            
            # 添加元数据
            metadata = ConfigMetadata(
                name=strategy_name,
                version=merged_config.get('strategy', {}).get('version', '1.0.0'),
                environment=self.environment,
                loaded_at=self._get_timestamp(),
                source_files=source_files,
                validation_errors=validation_errors
            )
            
            merged_config['_metadata'] = metadata.__dict__
            
            logger.info(f"Configuration loaded successfully from {len(source_files)} files")
            return merged_config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigValidationError(f"Configuration loading failed: {e}")
    
    def _validate_config(self, config: Dict[str, Any]) -> List[str]:
        """验证配置的基本结构和必需字段"""
        errors = []
        
        # 检查必需的顶级字段
        required_fields = ['system', 'logging', 'data', 'market_data']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # 验证系统配置
        if 'system' in config:
            system_config = config['system']
            if 'name' not in system_config:
                errors.append("Missing system.name")
            if 'version' not in system_config:
                errors.append("Missing system.version")
        
        # 验证环境特定设置
        if self.environment == ConfigEnvironment.PRODUCTION:
            # 生产环境验证
            if config.get('development', {}).get('debug', False):
                errors.append("Debug mode should be disabled in production")
            
            database_url = config.get('database', {}).get('url', '')
            if database_url.startswith('sqlite://'):
                errors.append("SQLite should not be used in production")
        
        return errors
    
    def _get_timestamp(self) -> str:
        """获取当前时间戳"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def save_resolved_config(self, config: Dict[str, Any], output_path: Optional[Path] = None) -> Path:
        """保存解析后的配置到文件"""
        if output_path is None:
            output_path = self.config_root / f"resolved_{self.environment.value}_config.yaml"
        
        # 移除元数据中不能序列化的内容
        config_copy = deepcopy(config)
        if '_metadata' in config_copy:
            metadata = config_copy['_metadata']
            if 'environment' in metadata:
                metadata['environment'] = metadata['environment']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_copy, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        logger.info(f"Resolved configuration saved to: {output_path}")
        return output_path
    
    def get_config_value(self, config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
        """
        使用点号分隔的路径获取配置值
        例如: get_config_value(config, "api.fastapi.port", 8000)
        """
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set_config_value(self, config: Dict[str, Any], key_path: str, value: Any) -> None:
        """
        使用点号分隔的路径设置配置值
        例如: set_config_value(config, "api.fastapi.port", 8080)
        """
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value


def create_config_loader(config_root: Optional[Path] = None) -> ConfigLoader:
    """创建配置加载器实例"""
    return ConfigLoader(config_root)


def load_config(strategy_name: str = "dipmaster_v4", config_root: Optional[Path] = None) -> Dict[str, Any]:
    """
    便捷函数：加载完整配置
    
    Args:
        strategy_name: 策略名称
        config_root: 配置根目录
        
    Returns:
        完整的合并配置字典
    """
    loader = create_config_loader(config_root)
    return loader.load_complete_config(strategy_name)


# Example usage and testing
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Set environment for testing
        os.environ['DIPMASTER_ENV'] = 'development'
        os.environ['LOG_LEVEL'] = 'DEBUG'
        
        # Load configuration
        loader = create_config_loader()
        config = loader.load_complete_config("dipmaster_v4")
        
        # Print summary
        print(f"Configuration loaded successfully!")
        print(f"Environment: {config['_metadata']['environment']}")
        print(f"Strategy: {config['_metadata']['name']} v{config['_metadata']['version']}")
        print(f"Source files: {config['_metadata']['source_files']}")
        
        # Test config value access
        api_port = loader.get_config_value(config, "api.fastapi.port", 8000)
        print(f"API Port: {api_port}")
        
        # Save resolved config
        output_path = loader.save_resolved_config(config)
        print(f"Resolved config saved to: {output_path}")
        
    except Exception as e:
        print(f"Configuration loading failed: {e}")
        sys.exit(1)