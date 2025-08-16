"""
Common Type Definitions for DipMaster Trading System
通用类型定义 - 系统中使用的基础类型

This module defines common types used throughout the trading system.
"""

from typing import Dict, Any, Optional, Union, List, Tuple, Callable, TypeVar, Generic
from datetime import datetime
from decimal import Decimal

# Basic value types
Timestamp = Union[float, int, datetime]
Price = Union[float, Decimal]
Quantity = Union[float, Decimal] 
Percentage = float
Volume = Union[float, Decimal]

# Numeric types
NumericValue = Union[int, float, Decimal]
OptionalNumeric = Optional[NumericValue]

# String types  
Symbol = str
Currency = str
Exchange = str
Identifier = str

# Dictionary types
ConfigDict = Dict[str, Any]
JsonDict = Dict[str, Any]
OptionalDict = Optional[Dict[str, Any]]
StringDict = Dict[str, str]
MetadataDict = Dict[str, Union[str, int, float, bool]]

# List and tuple types
StringList = List[str]
NumericList = List[NumericValue]
OptionalStringList = Optional[List[str]]

# Function types
CallbackFunction = Callable[..., Any]
AsyncCallbackFunction = Callable[..., Any]  # Should be Awaitable but keeping simple
ValidationFunction = Callable[[Any], bool]

# Generic types
T = TypeVar('T')
K = TypeVar('K') 
V = TypeVar('V')

# Result types for operations
class OperationResult(Generic[T]):
    """Generic result wrapper for operations."""
    
    def __init__(self, success: bool, data: Optional[T] = None, error: Optional[str] = None):
        self.success = success
        self.data = data
        self.error = error
    
    def is_success(self) -> bool:
        return self.success
    
    def is_error(self) -> bool:
        return not self.success
    
    def get_data(self) -> Optional[T]:
        return self.data
    
    def get_error(self) -> Optional[str]:
        return self.error

# Success and error result aliases
Success = OperationResult[T]
Error = OperationResult[None]

# File and path types
FilePath = Union[str, 'Path']  # Avoid circular import with pathlib
DirectoryPath = Union[str, 'Path']

# Network and API types
IpAddress = str
Port = int
Url = str
HttpMethod = str
HttpHeaders = Dict[str, str]
HttpParams = Dict[str, Union[str, int, float, bool]]

# Time-related types
TimeWindow = Tuple[datetime, datetime]
Duration = Union[int, float]  # Usually in seconds
TimeInterval = Union[int, float]  # Usually in seconds

# Data validation types
ValidationErrors = Dict[str, List[str]]
ValidationResult = Tuple[bool, Optional[ValidationErrors]]

# Event types
EventName = str
EventData = JsonDict
EventHandler = Callable[[EventData], None]
AsyncEventHandler = Callable[[EventData], Any]  # Should be Awaitable

# Logging types
LogLevel = str
LogMessage = str
LogContext = Dict[str, Any]

# Configuration types
EnvironmentName = str
ConfigSection = str
ConfigValue = Union[str, int, float, bool, List[Any], Dict[str, Any]]

# Database types (if used)
DatabaseUrl = str
TableName = str
ColumnName = str
QueryParams = Dict[str, Any]
DatabaseRow = Dict[str, Any]
DatabaseRows = List[DatabaseRow]

# Cache types
CacheKey = str
CacheValue = Any
CacheTtl = Union[int, float]  # Usually in seconds

# Serialization types
SerializedData = Union[str, bytes]
SerializationFormat = str  # 'json', 'pickle', etc.

# Rate limiting types  
RateLimit = int  # requests per time period
RateLimitPeriod = int  # time period in seconds

# Pagination types
PageNumber = int
PageSize = int
TotalCount = int
PaginationInfo = Tuple[PageNumber, PageSize, TotalCount]

# Status types
Status = str
StatusCode = Union[int, str]
HealthStatus = str  # 'healthy', 'degraded', 'unhealthy'

# Resource types
ResourceId = str
ResourceType = str
ResourceName = str

# Version types
Version = str
VersionNumber = Tuple[int, int, int]  # major, minor, patch

# Geographic types (if needed for compliance/regulations)
CountryCode = str
RegionCode = str
TimezoneStr = str

# Financial types
CurrencyCode = str  # 'USD', 'EUR', etc.
CurrencyPair = str  # 'BTCUSDT', etc.
MarketPrice = Price
OrderValue = Price
ProfitLoss = Union[Price, Percentage]
Balance = Price

# Risk management types
RiskLevel = str  # 'low', 'medium', 'high', 'critical'
RiskScore = float  # Usually 0-1 or 0-100
ExposureAmount = Price
DrawdownAmount = Union[Price, Percentage]

# Performance types
LatencyMs = float
ThroughputRps = float  # Requests per second
SuccessRate = Percentage
ErrorRate = Percentage

# Monitoring types
MetricValue = NumericValue
MetricTimestamp = Timestamp
AlertThreshold = NumericValue
AlertMessage = str

# Security types
SecretValue = str  # For passwords, API keys, etc. 
HashedValue = str
EncryptedValue = str
PublicKey = str
PrivateKey = str

# Trading-specific financial types
Leverage = Union[int, float]
MarginRequirement = Price
PositionSize = Price
Commission = Price
Slippage = Price
Spread = Price

# Time-based types for trading
TradingHours = Tuple[int, int]  # (start_hour, end_hour)
TradingSession = str  # 'london', 'new_york', 'tokyo', etc.
MarketState = str  # 'open', 'closed', 'pre_market', 'after_hours'

# Strategy types
StrategyName = str
StrategyVersion = str
StrategyParameters = ConfigDict
StrategySignal = JsonDict

# Backtesting types
BacktestPeriod = TimeWindow
BacktestResults = JsonDict
PerformanceMetrics = JsonDict