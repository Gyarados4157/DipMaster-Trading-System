#!/usr/bin/env python3
"""
DipMaster Trading System - Configuration Validator
配置验证工具，确保配置的完整性、一致性和安全性

Author: DipMaster Development Team
Date: 2025-08-16
Version: 4.0.0
"""

import re
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class ValidationLevel(str, Enum):
    """验证级别"""
    ERROR = "ERROR"      # 阻止系统运行的错误
    WARNING = "WARNING"  # 应该修复但不阻止运行的警告
    INFO = "INFO"        # 信息性建议

@dataclass
class ValidationResult:
    """验证结果"""
    level: ValidationLevel
    message: str
    path: str
    suggestion: Optional[str] = None

class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        
    def validate_complete_config(self, config: Dict[str, Any]) -> List[ValidationResult]:
        """验证完整配置"""
        self.results = []
        
        # 基础结构验证
        self._validate_structure(config)
        
        # 系统配置验证
        self._validate_system_config(config.get('system', {}))
        
        # 数据配置验证
        self._validate_data_config(config.get('data', {}))
        
        # 市场数据配置验证
        self._validate_market_data_config(config.get('market_data', {}))
        
        # 特征工程配置验证
        self._validate_features_config(config.get('features', {}))
        
        # 机器学习配置验证
        self._validate_ml_config(config.get('models', {}))
        
        # 风险管理配置验证
        self._validate_risk_config(config.get('risk', {}))
        
        # 执行配置验证
        self._validate_execution_config(config.get('execution', {}))
        
        # API配置验证
        self._validate_api_config(config.get('api', {}))
        
        # 监控配置验证
        self._validate_monitoring_config(config.get('monitoring', {}))
        
        # 环境特定验证
        environment = config.get('_metadata', {}).get('environment', 'development')
        self._validate_environment_specific(config, environment)
        
        # 安全配置验证
        self._validate_security_config(config.get('security', {}), environment)
        
        return self.results
    
    def _add_result(self, level: ValidationLevel, message: str, path: str, suggestion: str = None):
        """添加验证结果"""
        self.results.append(ValidationResult(level, message, path, suggestion))
    
    def _validate_structure(self, config: Dict[str, Any]):
        """验证基础配置结构"""
        required_sections = [
            'system', 'logging', 'data', 'market_data', 'features', 
            'models', 'risk', 'execution', 'api', 'monitoring'
        ]
        
        for section in required_sections:
            if section not in config:
                self._add_result(
                    ValidationLevel.ERROR,
                    f"Missing required configuration section: {section}",
                    f"root.{section}",
                    f"Add {section} section to configuration"
                )
    
    def _validate_system_config(self, system_config: Dict[str, Any]):
        """验证系统配置"""
        path_prefix = "system"
        
        # 必需字段
        required_fields = ['name', 'version', 'environment']
        for field in required_fields:
            if field not in system_config:
                self._add_result(
                    ValidationLevel.ERROR,
                    f"Missing required system field: {field}",
                    f"{path_prefix}.{field}",
                    f"Add {field} to system configuration"
                )
        
        # 版本格式验证
        if 'version' in system_config:
            version = system_config['version']
            if not re.match(r'^\d+\.\d+\.\d+$', version):
                self._add_result(
                    ValidationLevel.WARNING,
                    f"Version format should be X.Y.Z, got: {version}",
                    f"{path_prefix}.version",
                    "Use semantic versioning format (e.g., 4.0.0)"
                )
        
        # 超时设置验证
        timeouts = system_config.get('timeouts', {})
        if timeouts:
            self._validate_timeout_values(timeouts, f"{path_prefix}.timeouts")
        
        # 资源限制验证
        limits = system_config.get('limits', {})
        if limits:
            self._validate_resource_limits(limits, f"{path_prefix}.limits")
    
    def _validate_timeout_values(self, timeouts: Dict[str, Any], path_prefix: str):
        """验证超时设置"""
        reasonable_timeouts = {
            'api_request': (1, 300),  # 1秒到5分钟
            'websocket_connect': (1, 60),  # 1秒到1分钟
            'database_query': (1, 300),  # 1秒到5分钟
            'model_inference': (0.1, 30)  # 0.1秒到30秒
        }
        
        for timeout_name, value in timeouts.items():
            if timeout_name in reasonable_timeouts:
                min_val, max_val = reasonable_timeouts[timeout_name]
                if not isinstance(value, (int, float)) or value < min_val or value > max_val:
                    self._add_result(
                        ValidationLevel.WARNING,
                        f"Timeout {timeout_name} value {value} outside reasonable range ({min_val}-{max_val})",
                        f"{path_prefix}.{timeout_name}",
                        f"Set {timeout_name} between {min_val} and {max_val} seconds"
                    )
    
    def _validate_resource_limits(self, limits: Dict[str, Any], path_prefix: str):
        """验证资源限制"""
        if 'max_memory_usage_mb' in limits:
            memory_limit = limits['max_memory_usage_mb']
            if memory_limit < 1024:  # Less than 1GB
                self._add_result(
                    ValidationLevel.WARNING,
                    f"Memory limit {memory_limit}MB might be too low for ML operations",
                    f"{path_prefix}.max_memory_usage_mb",
                    "Consider setting at least 2048MB for stable operation"
                )
        
        if 'max_cpu_usage_percent' in limits:
            cpu_limit = limits['max_cpu_usage_percent']
            if cpu_limit > 95:
                self._add_result(
                    ValidationLevel.WARNING,
                    f"CPU limit {cpu_limit}% is very high and may affect system stability",
                    f"{path_prefix}.max_cpu_usage_percent",
                    "Consider setting CPU limit to 80-90% for stability"
                )
    
    def _validate_data_config(self, data_config: Dict[str, Any]):
        """验证数据配置"""
        path_prefix = "data"
        
        # 存储路径验证
        storage = data_config.get('storage', {})
        if storage:
            for path_type, path_value in storage.items():
                if isinstance(path_value, str):
                    path_obj = Path(path_value)
                    # 检查路径是否包含潜在的安全风险
                    if '..' in str(path_obj):
                        self._add_result(
                            ValidationLevel.ERROR,
                            f"Path traversal detected in {path_type}: {path_value}",
                            f"{path_prefix}.storage.{path_type}",
                            "Use absolute paths without .. components"
                        )
        
        # 数据质量阈值验证
        quality = data_config.get('quality', {})
        if quality:
            self._validate_quality_thresholds(quality, f"{path_prefix}.quality")
        
        # 数据保留策略验证
        retention = data_config.get('retention', {})
        if retention:
            self._validate_retention_policies(retention, f"{path_prefix}.retention")
    
    def _validate_quality_thresholds(self, quality: Dict[str, Any], path_prefix: str):
        """验证数据质量阈值"""
        if 'min_data_completeness' in quality:
            completeness = quality['min_data_completeness']
            if not 0 <= completeness <= 1:
                self._add_result(
                    ValidationLevel.ERROR,
                    f"Data completeness must be between 0 and 1, got: {completeness}",
                    f"{path_prefix}.min_data_completeness",
                    "Set completeness between 0.0 and 1.0"
                )
            elif completeness < 0.8:
                self._add_result(
                    ValidationLevel.WARNING,
                    f"Data completeness threshold {completeness} is low, may affect model quality",
                    f"{path_prefix}.min_data_completeness",
                    "Consider setting completeness to at least 0.9 for production"
                )
    
    def _validate_retention_policies(self, retention: Dict[str, Any], path_prefix: str):
        """验证数据保留策略"""
        reasonable_retention = {
            'raw_market_data': (30, 3650),  # 30 days to 10 years
            'processed_features': (7, 730),  # 7 days to 2 years
            'model_predictions': (1, 365),  # 1 day to 1 year
            'execution_logs': (1, 90)  # 1 day to 3 months
        }
        
        for data_type, days in retention.items():
            if data_type in reasonable_retention:
                min_days, max_days = reasonable_retention[data_type]
                if not isinstance(days, int) or days < min_days or days > max_days:
                    self._add_result(
                        ValidationLevel.WARNING,
                        f"Retention period for {data_type} ({days} days) outside reasonable range",
                        f"{path_prefix}.{data_type}",
                        f"Set retention between {min_days} and {max_days} days"
                    )
    
    def _validate_market_data_config(self, market_data_config: Dict[str, Any]):
        """验证市场数据配置"""
        path_prefix = "market_data"
        
        # 交易所验证
        primary_exchange = market_data_config.get('primary_exchange')
        if primary_exchange:
            supported_exchanges = ['binance', 'okx', 'coinbase', 'kraken', 'bitfinex']
            if primary_exchange not in supported_exchanges:
                self._add_result(
                    ValidationLevel.WARNING,
                    f"Primary exchange {primary_exchange} may not be fully supported",
                    f"{path_prefix}.primary_exchange",
                    f"Consider using one of: {', '.join(supported_exchanges)}"
                )
        
        # 符号验证
        symbols = market_data_config.get('collection', {}).get('symbols', [])
        if symbols:
            self._validate_trading_symbols(symbols, f"{path_prefix}.collection.symbols")
        
        # 实时数据配置验证
        realtime = market_data_config.get('realtime', {})
        if realtime:
            self._validate_realtime_config(realtime, f"{path_prefix}.realtime")
    
    def _validate_trading_symbols(self, symbols: List[str], path_prefix: str):
        """验证交易符号"""
        valid_pattern = re.compile(r'^[A-Z]{3,10}USDT?$')
        
        for symbol in symbols:
            if not valid_pattern.match(symbol):
                self._add_result(
                    ValidationLevel.WARNING,
                    f"Symbol {symbol} does not match expected pattern",
                    f"{path_prefix}",
                    "Use format like BTCUSDT, ETHUSDT"
                )
        
        # 检查重复符号
        if len(symbols) != len(set(symbols)):
            duplicates = [s for s in symbols if symbols.count(s) > 1]
            self._add_result(
                ValidationLevel.WARNING,
                f"Duplicate symbols found: {set(duplicates)}",
                f"{path_prefix}",
                "Remove duplicate symbols"
            )
    
    def _validate_realtime_config(self, realtime: Dict[str, Any], path_prefix: str):
        """验证实时数据配置"""
        # WebSocket URL验证
        websocket_url = realtime.get('websocket_url')
        if websocket_url and not websocket_url.startswith(('ws://', 'wss://')):
            self._add_result(
                ValidationLevel.ERROR,
                f"Invalid WebSocket URL format: {websocket_url}",
                f"{path_prefix}.websocket_url",
                "Use ws:// or wss:// protocol"
            )
        
        # 缓冲区大小验证
        buffer_size = realtime.get('buffer_size')
        if buffer_size and buffer_size < 100:
            self._add_result(
                ValidationLevel.WARNING,
                f"Buffer size {buffer_size} may be too small for high-frequency data",
                f"{path_prefix}.buffer_size",
                "Consider setting buffer size to at least 1000"
            )
    
    def _validate_features_config(self, features_config: Dict[str, Any]):
        """验证特征工程配置"""
        path_prefix = "features"
        
        # 技术指标参数验证
        technical = features_config.get('technical', {})
        if technical:
            self._validate_technical_indicators(technical, f"{path_prefix}.technical")
    
    def _validate_technical_indicators(self, technical: Dict[str, Any], path_prefix: str):
        """验证技术指标参数"""
        # RSI周期验证
        rsi_periods = technical.get('rsi_periods', [])
        for period in rsi_periods:
            if not isinstance(period, int) or period < 2 or period > 100:
                self._add_result(
                    ValidationLevel.WARNING,
                    f"RSI period {period} outside typical range (2-100)",
                    f"{path_prefix}.rsi_periods",
                    "Use RSI periods between 14-30 for most strategies"
                )
        
        # 移动平均周期验证
        ma_periods = technical.get('ma_periods', [])
        for period in ma_periods:
            if not isinstance(period, int) or period < 2:
                self._add_result(
                    ValidationLevel.ERROR,
                    f"Invalid MA period: {period}",
                    f"{path_prefix}.ma_periods",
                    "MA periods must be positive integers >= 2"
                )
    
    def _validate_ml_config(self, ml_config: Dict[str, Any]):
        """验证机器学习配置"""
        path_prefix = "models"
        
        # 集成模型验证
        ensemble = ml_config.get('ensemble', {})
        if ensemble:
            models = ensemble.get('models', [])
            weights = ensemble.get('weights', [])
            
            if len(models) != len(weights):
                self._add_result(
                    ValidationLevel.ERROR,
                    f"Number of models ({len(models)}) doesn't match weights ({len(weights)})",
                    f"{path_prefix}.ensemble",
                    "Ensure each model has a corresponding weight"
                )
            
            if weights and abs(sum(weights) - 1.0) > 0.01:
                self._add_result(
                    ValidationLevel.WARNING,
                    f"Ensemble weights sum to {sum(weights)}, should sum to 1.0",
                    f"{path_prefix}.ensemble.weights",
                    "Adjust weights to sum to 1.0"
                )
    
    def _validate_risk_config(self, risk_config: Dict[str, Any]):
        """验证风险管理配置"""
        path_prefix = "risk"
        
        # 仓位大小验证
        position_sizing = risk_config.get('position_sizing', {})
        if position_sizing:
            default_size = position_sizing.get('default_size_usd')
            max_size = position_sizing.get('max_position_size_usd')
            
            if default_size and max_size and default_size > max_size:
                self._add_result(
                    ValidationLevel.ERROR,
                    f"Default position size ({default_size}) exceeds maximum ({max_size})",
                    f"{path_prefix}.position_sizing",
                    "Set default_size_usd <= max_position_size_usd"
                )
        
        # 风险限制验证
        limits = risk_config.get('limits', {})
        if limits:
            self._validate_risk_limits(limits, f"{path_prefix}.limits")
    
    def _validate_risk_limits(self, limits: Dict[str, Any], path_prefix: str):
        """验证风险限制"""
        # 检查损失限制的合理性
        daily_loss = limits.get('max_daily_loss_usd')
        weekly_loss = limits.get('max_weekly_loss_usd') 
        monthly_loss = limits.get('max_monthly_loss_usd')
        
        if daily_loss and weekly_loss and daily_loss * 7 < weekly_loss:
            self._add_result(
                ValidationLevel.WARNING,
                f"Weekly loss limit ({weekly_loss}) allows more risk than 7x daily limit",
                f"{path_prefix}.max_weekly_loss_usd",
                "Consider setting weekly limit to 5-7x daily limit"
            )
    
    def _validate_execution_config(self, execution_config: Dict[str, Any]):
        """验证执行配置"""
        path_prefix = "execution"
        
        orders = execution_config.get('orders', {})
        if orders:
            slippage_tolerance = orders.get('slippage_tolerance')
            if slippage_tolerance and slippage_tolerance > 0.01:  # 1%
                self._add_result(
                    ValidationLevel.WARNING,
                    f"Slippage tolerance {slippage_tolerance} is high (>1%)",
                    f"{path_prefix}.orders.slippage_tolerance",
                    "Consider reducing slippage tolerance for better execution"
                )
    
    def _validate_api_config(self, api_config: Dict[str, Any]):
        """验证API配置"""
        path_prefix = "api"
        
        fastapi = api_config.get('fastapi', {})
        if fastapi:
            port = fastapi.get('port')
            if port and (port < 1024 or port > 65535):
                self._add_result(
                    ValidationLevel.WARNING,
                    f"API port {port} outside standard range (1024-65535)",
                    f"{path_prefix}.fastapi.port",
                    "Use ports between 8000-9000 for development"
                )
            
            workers = fastapi.get('workers', 1)
            if workers > 8:
                self._add_result(
                    ValidationLevel.WARNING,
                    f"High number of workers ({workers}) may cause resource contention",
                    f"{path_prefix}.fastapi.workers",
                    "Consider 2-4 workers for most setups"
                )
    
    def _validate_monitoring_config(self, monitoring_config: Dict[str, Any]):
        """验证监控配置"""
        path_prefix = "monitoring"
        
        alerts = monitoring_config.get('alerts', {})
        if alerts and alerts.get('enabled'):
            channels = alerts.get('channels', [])
            if not channels:
                self._add_result(
                    ValidationLevel.WARNING,
                    "Alerts enabled but no channels configured",
                    f"{path_prefix}.alerts.channels",
                    "Configure at least one alert channel (email, webhook, etc.)"
                )
    
    def _validate_environment_specific(self, config: Dict[str, Any], environment: str):
        """环境特定验证"""
        if environment == 'production':
            self._validate_production_config(config)
        elif environment == 'development':
            self._validate_development_config(config)
        elif environment == 'testing':
            self._validate_testing_config(config)
    
    def _validate_production_config(self, config: Dict[str, Any]):
        """生产环境特定验证"""
        # 调试模式检查
        if config.get('development', {}).get('debug', False):
            self._add_result(
                ValidationLevel.ERROR,
                "Debug mode must be disabled in production",
                "development.debug",
                "Set debug: false in production environment"
            )
        
        # 数据库检查
        database_url = config.get('database', {}).get('url', '')
        if 'sqlite' in database_url.lower():
            self._add_result(
                ValidationLevel.ERROR,
                "SQLite should not be used in production",
                "database.url",
                "Use PostgreSQL or other production database"
            )
        
        # 风险限制检查
        daily_loss = config.get('risk', {}).get('limits', {}).get('max_daily_loss_usd')
        if daily_loss and daily_loss > 10000:  # $10k
            self._add_result(
                ValidationLevel.WARNING,
                f"Daily loss limit ${daily_loss} is very high for production",
                "risk.limits.max_daily_loss_usd",
                "Consider lower daily loss limits for production safety"
            )
    
    def _validate_development_config(self, config: Dict[str, Any]):
        """开发环境特定验证"""
        # 开发环境应该有调试功能
        if not config.get('development', {}).get('debug', False):
            self._add_result(
                ValidationLevel.INFO,
                "Debug mode disabled in development environment",
                "development.debug",
                "Enable debug mode for easier development"
            )
    
    def _validate_testing_config(self, config: Dict[str, Any]):
        """测试环境特定验证"""
        # 测试环境应该使用模拟数据
        mock_mode = config.get('market_data', {}).get('realtime', {}).get('mock_mode', False)
        if not mock_mode:
            self._add_result(
                ValidationLevel.WARNING,
                "Real market data in testing environment",
                "market_data.realtime.mock_mode",
                "Enable mock_mode for consistent testing"
            )
    
    def _validate_security_config(self, security_config: Dict[str, Any], environment: str):
        """验证安全配置"""
        path_prefix = "security"
        
        if environment == 'production':
            # 生产环境安全检查
            jwt_config = security_config.get('jwt', {})
            if jwt_config:
                secret_key = jwt_config.get('secret_key', '')
                if len(secret_key) < 32:
                    self._add_result(
                        ValidationLevel.ERROR,
                        "JWT secret key too short for production",
                        f"{path_prefix}.jwt.secret_key",
                        "Use at least 32 character random string for JWT secret"
                    )
                
                algorithm = jwt_config.get('algorithm', 'HS256')
                if algorithm.startswith('HS') and environment == 'production':
                    self._add_result(
                        ValidationLevel.WARNING,
                        "Consider using RS256 (asymmetric) instead of HS256 in production",
                        f"{path_prefix}.jwt.algorithm",
                        "Use RS256 for better security in production"
                    )
    
    def generate_report(self) -> str:
        """生成验证报告"""
        if not self.results:
            return "✅ Configuration validation passed - no issues found!"
        
        # 按级别分组
        errors = [r for r in self.results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in self.results if r.level == ValidationLevel.WARNING]
        infos = [r for r in self.results if r.level == ValidationLevel.INFO]
        
        report = []
        report.append("=" * 80)
        report.append("DipMaster Configuration Validation Report")
        report.append("=" * 80)
        report.append("")
        
        # 摘要
        report.append(f"📊 Summary:")
        report.append(f"   Errors: {len(errors)}")
        report.append(f"   Warnings: {len(warnings)}")
        report.append(f"   Info: {len(infos)}")
        report.append("")
        
        # 错误详情
        if errors:
            report.append("❌ ERRORS (Must Fix):")
            for error in errors:
                report.append(f"   • {error.message}")
                report.append(f"     Path: {error.path}")
                if error.suggestion:
                    report.append(f"     Fix: {error.suggestion}")
                report.append("")
        
        # 警告详情
        if warnings:
            report.append("⚠️  WARNINGS (Should Fix):")
            for warning in warnings:
                report.append(f"   • {warning.message}")
                report.append(f"     Path: {warning.path}")
                if warning.suggestion:
                    report.append(f"     Suggestion: {warning.suggestion}")
                report.append("")
        
        # 信息详情
        if infos:
            report.append("ℹ️  INFO (Consider):")
            for info in infos:
                report.append(f"   • {info.message}")
                report.append(f"     Path: {info.path}")
                if info.suggestion:
                    report.append(f"     Suggestion: {info.suggestion}")
                report.append("")
        
        return "\n".join(report)
    
    def has_errors(self) -> bool:
        """检查是否有错误"""
        return any(r.level == ValidationLevel.ERROR for r in self.results)
    
    def has_warnings(self) -> bool:
        """检查是否有警告"""
        return any(r.level == ValidationLevel.WARNING for r in self.results)


def validate_config_file(config_path: Path) -> List[ValidationResult]:
    """验证单个配置文件"""
    validator = ConfigValidator()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return validator.validate_complete_config(config or {})
    except Exception as e:
        return [ValidationResult(
            ValidationLevel.ERROR,
            f"Failed to load config file: {e}",
            str(config_path),
            "Check file format and syntax"
        )]


# Example usage
if __name__ == "__main__":
    import sys
    from config_loader import load_config
    
    try:
        # Load configuration
        config = load_config("dipmaster_v4")
        
        # Validate configuration
        validator = ConfigValidator()
        results = validator.validate_complete_config(config)
        
        # Generate and print report
        report = validator.generate_report()
        print(report)
        
        # Exit with appropriate code
        if validator.has_errors():
            print("❌ Configuration validation failed with errors!")
            sys.exit(1)
        elif validator.has_warnings():
            print("⚠️  Configuration validation completed with warnings.")
            sys.exit(0)
        else:
            print("✅ Configuration validation passed!")
            sys.exit(0)
            
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)