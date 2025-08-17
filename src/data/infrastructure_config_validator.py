"""
Data Infrastructure Configuration and Test Validation System
数据基础设施配置和测试验证系统 - 确保系统完整性和性能
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import time
import psutil
import platform
import subprocess
import sys
import importlib
from dataclasses import dataclass, asdict
from enum import Enum
import yaml
import sqlite3
import redis
import warnings

warnings.filterwarnings('ignore')

class ValidationLevel(Enum):
    """验证级别"""
    BASIC = "basic"          # 基础验证
    STANDARD = "standard"    # 标准验证
    COMPREHENSIVE = "comprehensive"  # 全面验证
    PERFORMANCE = "performance"      # 性能验证

class ComponentStatus(Enum):
    """组件状态"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    SKIP = "skip"
    NOT_TESTED = "not_tested"

@dataclass
class ValidationResult:
    """验证结果"""
    component: str
    test_name: str
    status: ComponentStatus
    message: str
    execution_time_ms: float
    details: Dict[str, Any] = None

@dataclass
class SystemRequirements:
    """系统要求"""
    min_python_version: str = "3.9.0"
    min_memory_gb: int = 8
    min_disk_space_gb: int = 50
    required_packages: List[str] = None
    optional_packages: List[str] = None
    external_services: List[str] = None

@dataclass
class PerformanceBenchmark:
    """性能基准"""
    operation: str
    expected_time_ms: float
    max_time_ms: float
    min_throughput: float
    description: str

class InfrastructureConfigValidator:
    """数据基础设施配置验证器"""
    
    def __init__(self, config_path: str = None):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 配置路径
        self.config_path = Path(config_path) if config_path else Path("config/infrastructure.yaml")
        
        # 验证结果
        self.validation_results = []
        self.system_info = {}
        
        # 性能基准
        self.performance_benchmarks = [
            PerformanceBenchmark("pandas_read_parquet_100k", 500, 2000, 200000, "读取10万行Parquet文件"),
            PerformanceBenchmark("pandas_write_parquet_100k", 800, 3000, 125000, "写入10万行Parquet文件"),
            PerformanceBenchmark("numpy_computation_1m", 100, 500, 10000000, "100万数据点numpy计算"),
            PerformanceBenchmark("redis_roundtrip", 5, 20, 1000, "Redis读写往返"),
            PerformanceBenchmark("sqlite_query_10k", 50, 200, 200000, "SQLite查询1万条记录")
        ]
        
        # 系统要求
        self.system_requirements = SystemRequirements(
            min_python_version="3.9.0",
            min_memory_gb=8,
            min_disk_space_gb=50,
            required_packages=[
                "pandas>=2.0.0",
                "numpy>=1.21.0",
                "pyarrow>=10.0.0",
                "polars>=0.18.0",
                "duckdb>=0.8.0",
                "redis>=4.0.0",
                "ccxt>=4.0.0",
                "sqlite3",
                "asyncio",
                "aiofiles",
                "psutil>=5.8.0",
                "pyyaml>=6.0.0",
                "zstandard>=0.19.0",
                "lz4>=4.0.0"
            ],
            optional_packages=[
                "zarr>=2.12.0",
                "h5py>=3.7.0",
                "git-python>=1.0.3",
                "semver>=2.13.0",
                "websocket-client>=1.4.0"
            ],
            external_services=[
                "redis-server",
                "git"
            ]
        )
        
        # 测试数据
        self.test_data = None
        self.generate_test_data()
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler('logs/infrastructure_validator.log'),
                logging.StreamHandler()
            ]
        )
    
    def generate_test_data(self):
        """生成测试数据"""
        try:
            # 生成示例市场数据
            dates = pd.date_range('2024-01-01', '2024-12-31', freq='5min')
            self.test_data = pd.DataFrame({
                'timestamp': dates,
                'open': np.random.uniform(45000, 55000, len(dates)),
                'high': np.random.uniform(50000, 60000, len(dates)),
                'low': np.random.uniform(40000, 50000, len(dates)),
                'close': np.random.uniform(45000, 55000, len(dates)),
                'volume': np.random.uniform(1000, 100000, len(dates))
            })
            self.test_data.set_index('timestamp', inplace=True)
            
            # 确保OHLC逻辑正确
            for i in range(len(self.test_data)):
                row = self.test_data.iloc[i]
                self.test_data.iloc[i, self.test_data.columns.get_loc('high')] = max(row['open'], row['close'], row['high'])
                self.test_data.iloc[i, self.test_data.columns.get_loc('low')] = min(row['open'], row['close'], row['low'])
            
        except Exception as e:
            self.logger.error(f"生成测试数据失败: {e}")
            self.test_data = pd.DataFrame()
    
    async def validate_full_infrastructure(self, level: ValidationLevel = ValidationLevel.STANDARD) -> Dict:
        """全面验证基础设施"""
        self.logger.info(f"开始 {level.value} 级别的基础设施验证")
        start_time = time.time()
        
        # 清空之前的结果
        self.validation_results = []
        
        # 收集系统信息
        await self.collect_system_info()
        
        # 验证系统要求
        await self.validate_system_requirements()
        
        # 验证Python包
        await self.validate_python_packages()
        
        # 验证外部服务
        await self.validate_external_services()
        
        # 验证核心功能
        await self.validate_core_functionality()
        
        # 验证数据处理
        await self.validate_data_processing()
        
        # 验证存储系统
        await self.validate_storage_systems()
        
        # 验证网络连接
        await self.validate_network_connectivity()
        
        if level in [ValidationLevel.COMPREHENSIVE, ValidationLevel.PERFORMANCE]:
            # 性能基准测试
            await self.run_performance_benchmarks()
            
            # 压力测试
            await self.run_stress_tests()
            
            # 并发测试
            await self.run_concurrency_tests()
        
        # 生成验证报告
        total_time = (time.time() - start_time) * 1000
        report = self.generate_validation_report(total_time)
        
        self.logger.info(f"基础设施验证完成，耗时: {total_time:.2f}ms")
        
        return report
    
    async def collect_system_info(self):
        """收集系统信息"""
        try:
            self.system_info = {
                'platform': {
                    'system': platform.system(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor()
                },
                'python': {
                    'version': sys.version,
                    'executable': sys.executable,
                    'path': sys.path[:3]  # 只显示前3个路径
                },
                'memory': {
                    'total_gb': psutil.virtual_memory().total / (1024**3),
                    'available_gb': psutil.virtual_memory().available / (1024**3),
                    'percent_used': psutil.virtual_memory().percent
                },
                'disk': {
                    'total_gb': psutil.disk_usage('/').total / (1024**3),
                    'free_gb': psutil.disk_usage('/').free / (1024**3),
                    'percent_used': (psutil.disk_usage('/').total - psutil.disk_usage('/').free) / psutil.disk_usage('/').total * 100
                },
                'cpu': {
                    'count': psutil.cpu_count(),
                    'frequency_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else 'unknown',
                    'usage_percent': psutil.cpu_percent(interval=1)
                }
            }
            
            self.add_result("system_info", "collect_info", ComponentStatus.PASS, 
                          "系统信息收集完成", 0, self.system_info)
            
        except Exception as e:
            self.add_result("system_info", "collect_info", ComponentStatus.FAIL,
                          f"系统信息收集失败: {e}", 0)
    
    async def validate_system_requirements(self):
        """验证系统要求"""
        try:
            # Python版本检查
            current_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            if self.version_compare(current_python, self.system_requirements.min_python_version) >= 0:
                self.add_result("system", "python_version", ComponentStatus.PASS,
                              f"Python版本 {current_python} 满足要求", 0)
            else:
                self.add_result("system", "python_version", ComponentStatus.FAIL,
                              f"Python版本 {current_python} 低于要求 {self.system_requirements.min_python_version}", 0)
            
            # 内存检查
            memory_gb = self.system_info['memory']['total_gb']
            if memory_gb >= self.system_requirements.min_memory_gb:
                self.add_result("system", "memory", ComponentStatus.PASS,
                              f"内存 {memory_gb:.1f}GB 满足要求", 0)
            else:
                self.add_result("system", "memory", ComponentStatus.WARNING,
                              f"内存 {memory_gb:.1f}GB 低于推荐 {self.system_requirements.min_memory_gb}GB", 0)
            
            # 磁盘空间检查
            disk_free_gb = self.system_info['disk']['free_gb']
            if disk_free_gb >= self.system_requirements.min_disk_space_gb:
                self.add_result("system", "disk_space", ComponentStatus.PASS,
                              f"可用磁盘空间 {disk_free_gb:.1f}GB 满足要求", 0)
            else:
                self.add_result("system", "disk_space", ComponentStatus.WARNING,
                              f"可用磁盘空间 {disk_free_gb:.1f}GB 低于推荐 {self.system_requirements.min_disk_space_gb}GB", 0)
            
        except Exception as e:
            self.add_result("system", "requirements", ComponentStatus.FAIL,
                          f"系统要求验证失败: {e}", 0)
    
    async def validate_python_packages(self):
        """验证Python包"""
        try:
            # 检查必需包
            for package in self.system_requirements.required_packages:
                await self.check_package(package, required=True)
            
            # 检查可选包
            for package in self.system_requirements.optional_packages:
                await self.check_package(package, required=False)
                
        except Exception as e:
            self.add_result("packages", "validation", ComponentStatus.FAIL,
                          f"包验证失败: {e}", 0)
    
    async def check_package(self, package_spec: str, required: bool = True):
        """检查单个包"""
        start_time = time.time()
        
        try:
            # 解析包名和版本要求
            if ">=" in package_spec:
                package_name, min_version = package_spec.split(">=")
            else:
                package_name = package_spec
                min_version = None
            
            # 尝试导入包
            try:
                module = importlib.import_module(package_name)
                
                # 检查版本（如果指定）
                if min_version and hasattr(module, '__version__'):
                    current_version = module.__version__
                    if self.version_compare(current_version, min_version) >= 0:
                        status = ComponentStatus.PASS
                        message = f"{package_name} {current_version} 已安装且满足版本要求"
                    else:
                        status = ComponentStatus.WARNING if not required else ComponentStatus.FAIL
                        message = f"{package_name} {current_version} 版本过低，要求 >= {min_version}"
                else:
                    status = ComponentStatus.PASS
                    message = f"{package_name} 已安装"
                    
            except ImportError:
                status = ComponentStatus.FAIL if required else ComponentStatus.WARNING
                message = f"{package_name} 未安装"
            
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("packages", package_name, status, message, elapsed_time)
            
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("packages", package_name, ComponentStatus.FAIL,
                          f"包检查失败: {e}", elapsed_time)
    
    async def validate_external_services(self):
        """验证外部服务"""
        try:
            # Redis检查
            await self.check_redis()
            
            # Git检查
            await self.check_git()
            
        except Exception as e:
            self.add_result("services", "validation", ComponentStatus.FAIL,
                          f"外部服务验证失败: {e}", 0)
    
    async def check_redis(self):
        """检查Redis服务"""
        start_time = time.time()
        
        try:
            redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
            redis_client.ping()
            
            # 测试基本操作
            test_key = "test_infrastructure_validation"
            redis_client.set(test_key, "test_value", ex=60)
            value = redis_client.get(test_key)
            redis_client.delete(test_key)
            
            if value == "test_value":
                elapsed_time = (time.time() - start_time) * 1000
                self.add_result("services", "redis", ComponentStatus.PASS,
                              f"Redis服务正常，响应时间: {elapsed_time:.1f}ms", elapsed_time)
            else:
                elapsed_time = (time.time() - start_time) * 1000
                self.add_result("services", "redis", ComponentStatus.FAIL,
                              "Redis读写测试失败", elapsed_time)
                
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("services", "redis", ComponentStatus.WARNING,
                          f"Redis连接失败: {e}", elapsed_time)
    
    async def check_git(self):
        """检查Git"""
        start_time = time.time()
        
        try:
            result = subprocess.run(['git', '--version'], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                version_info = result.stdout.strip()
                elapsed_time = (time.time() - start_time) * 1000
                self.add_result("services", "git", ComponentStatus.PASS,
                              f"Git可用: {version_info}", elapsed_time)
            else:
                elapsed_time = (time.time() - start_time) * 1000
                self.add_result("services", "git", ComponentStatus.FAIL,
                              "Git命令执行失败", elapsed_time)
                
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("services", "git", ComponentStatus.WARNING,
                          f"Git检查失败: {e}", elapsed_time)
    
    async def validate_core_functionality(self):
        """验证核心功能"""
        try:
            # 测试数据基础设施模块导入
            await self.test_module_imports()
            
            # 测试基本数据操作
            await self.test_basic_data_operations()
            
        except Exception as e:
            self.add_result("core", "functionality", ComponentStatus.FAIL,
                          f"核心功能验证失败: {e}", 0)
    
    async def test_module_imports(self):
        """测试模块导入"""
        modules_to_test = [
            "src.data.advanced_data_infrastructure",
            "src.data.realtime_quality_monitor",
            "src.data.high_performance_storage",
            "src.data.bundle_version_manager"
        ]
        
        for module_name in modules_to_test:
            start_time = time.time()
            try:
                # 添加项目根目录到路径
                sys.path.insert(0, str(Path.cwd()))
                
                importlib.import_module(module_name)
                elapsed_time = (time.time() - start_time) * 1000
                self.add_result("modules", module_name.split('.')[-1], ComponentStatus.PASS,
                              f"模块导入成功", elapsed_time)
                
            except Exception as e:
                elapsed_time = (time.time() - start_time) * 1000
                self.add_result("modules", module_name.split('.')[-1], ComponentStatus.FAIL,
                              f"模块导入失败: {e}", elapsed_time)
    
    async def test_basic_data_operations(self):
        """测试基本数据操作"""
        if self.test_data.empty:
            self.add_result("data_ops", "basic_operations", ComponentStatus.SKIP,
                          "测试数据为空，跳过基本数据操作测试", 0)
            return
        
        start_time = time.time()
        
        try:
            # Pandas基本操作
            result1 = self.test_data.describe()
            result2 = self.test_data.resample('1H').mean()
            result3 = self.test_data['close'].rolling(20).mean()
            
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("data_ops", "pandas_operations", ComponentStatus.PASS,
                          f"Pandas基本操作正常，耗时: {elapsed_time:.1f}ms", elapsed_time)
            
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("data_ops", "pandas_operations", ComponentStatus.FAIL,
                          f"Pandas操作失败: {e}", elapsed_time)
    
    async def validate_data_processing(self):
        """验证数据处理"""
        try:
            # 测试数据质量检查
            await self.test_data_quality_checks()
            
            # 测试数据转换
            await self.test_data_transformations()
            
        except Exception as e:
            self.add_result("data_processing", "validation", ComponentStatus.FAIL,
                          f"数据处理验证失败: {e}", 0)
    
    async def test_data_quality_checks(self):
        """测试数据质量检查"""
        if self.test_data.empty:
            self.add_result("data_quality", "checks", ComponentStatus.SKIP,
                          "测试数据为空，跳过数据质量检查", 0)
            return
        
        start_time = time.time()
        
        try:
            # 完整性检查
            missing_count = self.test_data.isnull().sum().sum()
            
            # 一致性检查
            consistency_violations = 0
            consistency_violations += (self.test_data['high'] < self.test_data[['open', 'close']].max(axis=1)).sum()
            consistency_violations += (self.test_data['low'] > self.test_data[['open', 'close']].min(axis=1)).sum()
            
            # 有效性检查
            invalid_count = (self.test_data <= 0).sum().sum()
            
            elapsed_time = (time.time() - start_time) * 1000
            
            if missing_count == 0 and consistency_violations == 0 and invalid_count == 0:
                self.add_result("data_quality", "checks", ComponentStatus.PASS,
                              f"数据质量检查通过，耗时: {elapsed_time:.1f}ms", elapsed_time)
            else:
                self.add_result("data_quality", "checks", ComponentStatus.WARNING,
                              f"数据质量问题: 缺失值{missing_count}, 一致性违规{consistency_violations}, 无效值{invalid_count}",
                              elapsed_time)
            
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("data_quality", "checks", ComponentStatus.FAIL,
                          f"数据质量检查失败: {e}", elapsed_time)
    
    async def test_data_transformations(self):
        """测试数据转换"""
        if self.test_data.empty:
            self.add_result("data_transform", "transformations", ComponentStatus.SKIP,
                          "测试数据为空，跳过数据转换测试", 0)
            return
        
        start_time = time.time()
        
        try:
            # 技术指标计算
            sma_20 = self.test_data['close'].rolling(20).mean()
            rsi = self.calculate_rsi(self.test_data['close'], 14)
            
            # 数据重采样
            hourly_data = self.test_data.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("data_transform", "transformations", ComponentStatus.PASS,
                          f"数据转换测试通过，耗时: {elapsed_time:.1f}ms", elapsed_time)
            
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("data_transform", "transformations", ComponentStatus.FAIL,
                          f"数据转换测试失败: {e}", elapsed_time)
    
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def validate_storage_systems(self):
        """验证存储系统"""
        try:
            # 测试文件系统读写
            await self.test_filesystem_operations()
            
            # 测试数据库操作
            await self.test_database_operations()
            
        except Exception as e:
            self.add_result("storage", "validation", ComponentStatus.FAIL,
                          f"存储系统验证失败: {e}", 0)
    
    async def test_filesystem_operations(self):
        """测试文件系统操作"""
        start_time = time.time()
        
        try:
            test_dir = Path("temp_test")
            test_dir.mkdir(exist_ok=True)
            
            # 测试Parquet读写
            if not self.test_data.empty:
                parquet_file = test_dir / "test.parquet"
                self.test_data.to_parquet(parquet_file)
                loaded_data = pd.read_parquet(parquet_file)
                
                # 验证数据完整性
                if len(loaded_data) == len(self.test_data):
                    elapsed_time = (time.time() - start_time) * 1000
                    self.add_result("storage", "parquet_io", ComponentStatus.PASS,
                                  f"Parquet读写测试通过，耗时: {elapsed_time:.1f}ms", elapsed_time)
                else:
                    elapsed_time = (time.time() - start_time) * 1000
                    self.add_result("storage", "parquet_io", ComponentStatus.FAIL,
                                  "Parquet读写数据不一致", elapsed_time)
                
                # 清理
                parquet_file.unlink(missing_ok=True)
            
            test_dir.rmdir()
            
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("storage", "filesystem", ComponentStatus.FAIL,
                          f"文件系统操作失败: {e}", elapsed_time)
    
    async def test_database_operations(self):
        """测试数据库操作"""
        start_time = time.time()
        
        try:
            # SQLite测试
            db_path = "temp_test.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 创建测试表
            cursor.execute("""
                CREATE TABLE test_table (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    value REAL
                )
            """)
            
            # 插入测试数据
            test_records = [
                (1, "2024-01-01T00:00:00", 100.5),
                (2, "2024-01-01T01:00:00", 101.2),
                (3, "2024-01-01T02:00:00", 99.8)
            ]
            
            cursor.executemany("""
                INSERT INTO test_table (id, timestamp, value) VALUES (?, ?, ?)
            """, test_records)
            
            # 查询数据
            cursor.execute("SELECT COUNT(*) FROM test_table")
            count = cursor.fetchone()[0]
            
            conn.commit()
            conn.close()
            
            # 清理
            Path(db_path).unlink(missing_ok=True)
            
            if count == len(test_records):
                elapsed_time = (time.time() - start_time) * 1000
                self.add_result("storage", "sqlite", ComponentStatus.PASS,
                              f"SQLite操作测试通过，耗时: {elapsed_time:.1f}ms", elapsed_time)
            else:
                elapsed_time = (time.time() - start_time) * 1000
                self.add_result("storage", "sqlite", ComponentStatus.FAIL,
                              "SQLite数据不一致", elapsed_time)
            
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("storage", "database", ComponentStatus.FAIL,
                          f"数据库操作失败: {e}", elapsed_time)
    
    async def validate_network_connectivity(self):
        """验证网络连接"""
        try:
            # 测试HTTP连接
            await self.test_http_connectivity()
            
            # 测试WebSocket连接（模拟）
            await self.test_websocket_simulation()
            
        except Exception as e:
            self.add_result("network", "validation", ComponentStatus.FAIL,
                          f"网络连接验证失败: {e}", 0)
    
    async def test_http_connectivity(self):
        """测试HTTP连接"""
        start_time = time.time()
        
        try:
            import urllib.request
            import urllib.error
            
            # 测试连接到Binance API
            test_url = "https://api.binance.com/api/v3/ping"
            
            with urllib.request.urlopen(test_url, timeout=10) as response:
                if response.status == 200:
                    elapsed_time = (time.time() - start_time) * 1000
                    self.add_result("network", "http_binance", ComponentStatus.PASS,
                                  f"Binance API连接正常，耗时: {elapsed_time:.1f}ms", elapsed_time)
                else:
                    elapsed_time = (time.time() - start_time) * 1000
                    self.add_result("network", "http_binance", ComponentStatus.WARNING,
                                  f"Binance API响应异常: {response.status}", elapsed_time)
            
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("network", "http", ComponentStatus.WARNING,
                          f"HTTP连接测试失败: {e}", elapsed_time)
    
    async def test_websocket_simulation(self):
        """测试WebSocket模拟"""
        start_time = time.time()
        
        try:
            # 模拟WebSocket连接逻辑
            import socket
            
            # 测试TCP连接到Binance WebSocket端口
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            
            result = sock.connect_ex(('stream.binance.com', 9443))
            sock.close()
            
            elapsed_time = (time.time() - start_time) * 1000
            
            if result == 0:
                self.add_result("network", "websocket", ComponentStatus.PASS,
                              f"WebSocket端口连接正常，耗时: {elapsed_time:.1f}ms", elapsed_time)
            else:
                self.add_result("network", "websocket", ComponentStatus.WARNING,
                              f"WebSocket端口连接失败: {result}", elapsed_time)
            
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("network", "websocket", ComponentStatus.WARNING,
                          f"WebSocket连接测试失败: {e}", elapsed_time)
    
    async def run_performance_benchmarks(self):
        """运行性能基准测试"""
        try:
            for benchmark in self.performance_benchmarks:
                await self.run_single_benchmark(benchmark)
                
        except Exception as e:
            self.add_result("performance", "benchmarks", ComponentStatus.FAIL,
                          f"性能基准测试失败: {e}", 0)
    
    async def run_single_benchmark(self, benchmark: PerformanceBenchmark):
        """运行单个性能基准"""
        start_time = time.time()
        
        try:
            if benchmark.operation == "pandas_read_parquet_100k":
                # 准备测试数据
                if not self.test_data.empty:
                    test_file = Path("temp_benchmark.parquet")
                    sample_data = self.test_data.head(100000) if len(self.test_data) > 100000 else self.test_data
                    sample_data.to_parquet(test_file)
                    
                    # 执行基准测试
                    bench_start = time.time()
                    pd.read_parquet(test_file)
                    bench_time = (time.time() - bench_start) * 1000
                    
                    # 清理
                    test_file.unlink(missing_ok=True)
                    
                    self.evaluate_benchmark_result(benchmark, bench_time, len(sample_data))
                else:
                    self.add_result("performance", benchmark.operation, ComponentStatus.SKIP,
                                  "测试数据不足", 0)
            
            elif benchmark.operation == "numpy_computation_1m":
                # NumPy计算基准
                data = np.random.random(1000000)
                
                bench_start = time.time()
                result = np.mean(data) + np.std(data) + np.sum(data)
                bench_time = (time.time() - bench_start) * 1000
                
                self.evaluate_benchmark_result(benchmark, bench_time, 1000000)
            
            elif benchmark.operation == "redis_roundtrip":
                # Redis往返基准
                try:
                    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
                    
                    bench_start = time.time()
                    redis_client.set("benchmark_key", "benchmark_value")
                    value = redis_client.get("benchmark_key")
                    redis_client.delete("benchmark_key")
                    bench_time = (time.time() - bench_start) * 1000
                    
                    self.evaluate_benchmark_result(benchmark, bench_time, 1)
                    
                except Exception as e:
                    self.add_result("performance", benchmark.operation, ComponentStatus.SKIP,
                                  f"Redis不可用: {e}", 0)
            
            else:
                self.add_result("performance", benchmark.operation, ComponentStatus.SKIP,
                              "基准测试未实现", 0)
            
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("performance", benchmark.operation, ComponentStatus.FAIL,
                          f"基准测试失败: {e}", elapsed_time)
    
    def evaluate_benchmark_result(self, benchmark: PerformanceBenchmark, actual_time_ms: float, items_processed: int):
        """评估基准测试结果"""
        throughput = items_processed / (actual_time_ms / 1000) if actual_time_ms > 0 else 0
        
        if actual_time_ms <= benchmark.expected_time_ms:
            status = ComponentStatus.PASS
            message = f"性能优秀: {actual_time_ms:.1f}ms (预期: {benchmark.expected_time_ms:.1f}ms), 吞吐量: {throughput:.0f}/s"
        elif actual_time_ms <= benchmark.max_time_ms:
            status = ComponentStatus.WARNING
            message = f"性能可接受: {actual_time_ms:.1f}ms (最大: {benchmark.max_time_ms:.1f}ms), 吞吐量: {throughput:.0f}/s"
        else:
            status = ComponentStatus.FAIL
            message = f"性能不达标: {actual_time_ms:.1f}ms (最大: {benchmark.max_time_ms:.1f}ms), 吞吐量: {throughput:.0f}/s"
        
        self.add_result("performance", benchmark.operation, status, message, actual_time_ms,
                       {"throughput": throughput, "items_processed": items_processed})
    
    async def run_stress_tests(self):
        """运行压力测试"""
        try:
            # 内存压力测试
            await self.test_memory_stress()
            
            # CPU压力测试
            await self.test_cpu_stress()
            
        except Exception as e:
            self.add_result("stress", "tests", ComponentStatus.FAIL,
                          f"压力测试失败: {e}", 0)
    
    async def test_memory_stress(self):
        """内存压力测试"""
        start_time = time.time()
        
        try:
            # 创建大量数据测试内存使用
            large_arrays = []
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            for i in range(10):
                arr = np.random.random((100000, 10))
                large_arrays.append(arr)
            
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # 清理
            del large_arrays
            
            elapsed_time = (time.time() - start_time) * 1000
            
            if memory_increase < 1000:  # 1GB
                self.add_result("stress", "memory", ComponentStatus.PASS,
                              f"内存压力测试通过，峰值增加: {memory_increase:.1f}MB", elapsed_time)
            else:
                self.add_result("stress", "memory", ComponentStatus.WARNING,
                              f"内存使用较高，峰值增加: {memory_increase:.1f}MB", elapsed_time)
            
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("stress", "memory", ComponentStatus.FAIL,
                          f"内存压力测试失败: {e}", elapsed_time)
    
    async def test_cpu_stress(self):
        """CPU压力测试"""
        start_time = time.time()
        
        try:
            # CPU密集型计算
            initial_cpu = psutil.cpu_percent(interval=1)
            
            # 计算密集型任务
            result = 0
            for i in range(1000000):
                result += np.sqrt(i)
            
            final_cpu = psutil.cpu_percent(interval=1)
            
            elapsed_time = (time.time() - start_time) * 1000
            
            self.add_result("stress", "cpu", ComponentStatus.PASS,
                          f"CPU压力测试完成，CPU使用: {initial_cpu:.1f}% -> {final_cpu:.1f}%", elapsed_time,
                          {"cpu_before": initial_cpu, "cpu_after": final_cpu})
            
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("stress", "cpu", ComponentStatus.FAIL,
                          f"CPU压力测试失败: {e}", elapsed_time)
    
    async def run_concurrency_tests(self):
        """运行并发测试"""
        try:
            # 并发数据处理测试
            await self.test_concurrent_data_processing()
            
        except Exception as e:
            self.add_result("concurrency", "tests", ComponentStatus.FAIL,
                          f"并发测试失败: {e}", 0)
    
    async def test_concurrent_data_processing(self):
        """测试并发数据处理"""
        start_time = time.time()
        
        try:
            if self.test_data.empty:
                self.add_result("concurrency", "data_processing", ComponentStatus.SKIP,
                              "测试数据为空，跳过并发测试", 0)
                return
            
            # 创建并发任务
            async def process_data_chunk(chunk):
                return chunk.describe()
            
            # 分割数据
            chunk_size = len(self.test_data) // 4
            chunks = [
                self.test_data.iloc[i:i+chunk_size] 
                for i in range(0, len(self.test_data), chunk_size)
            ]
            
            # 并发执行
            tasks = [process_data_chunk(chunk) for chunk in chunks if not chunk.empty]
            results = await asyncio.gather(*tasks)
            
            elapsed_time = (time.time() - start_time) * 1000
            
            if len(results) == len(tasks):
                self.add_result("concurrency", "data_processing", ComponentStatus.PASS,
                              f"并发数据处理测试通过，处理了{len(chunks)}个数据块", elapsed_time)
            else:
                self.add_result("concurrency", "data_processing", ComponentStatus.FAIL,
                              "并发数据处理结果不完整", elapsed_time)
            
        except Exception as e:
            elapsed_time = (time.time() - start_time) * 1000
            self.add_result("concurrency", "data_processing", ComponentStatus.FAIL,
                          f"并发数据处理测试失败: {e}", elapsed_time)
    
    def add_result(self, component: str, test_name: str, status: ComponentStatus, 
                   message: str, execution_time_ms: float, details: Dict = None):
        """添加验证结果"""
        result = ValidationResult(
            component=component,
            test_name=test_name,
            status=status,
            message=message,
            execution_time_ms=execution_time_ms,
            details=details
        )
        self.validation_results.append(result)
    
    def generate_validation_report(self, total_time_ms: float) -> Dict:
        """生成验证报告"""
        # 统计结果
        status_counts = {status: 0 for status in ComponentStatus}
        for result in self.validation_results:
            status_counts[result.status] += 1
        
        # 按组件分组
        components = {}
        for result in self.validation_results:
            if result.component not in components:
                components[result.component] = []
            components[result.component].append(asdict(result))
        
        # 计算总体状态
        if status_counts[ComponentStatus.FAIL] > 0:
            overall_status = "FAIL"
        elif status_counts[ComponentStatus.WARNING] > 0:
            overall_status = "WARNING"
        else:
            overall_status = "PASS"
        
        # 性能统计
        performance_stats = {}
        for result in self.validation_results:
            if result.component == "performance":
                performance_stats[result.test_name] = {
                    'status': result.status.value,
                    'time_ms': result.execution_time_ms,
                    'details': result.details
                }
        
        report = {
            'validation_summary': {
                'overall_status': overall_status,
                'total_tests': len(self.validation_results),
                'passed': status_counts[ComponentStatus.PASS],
                'failed': status_counts[ComponentStatus.FAIL],
                'warnings': status_counts[ComponentStatus.WARNING],
                'skipped': status_counts[ComponentStatus.SKIP],
                'total_time_ms': total_time_ms,
                'timestamp': datetime.now().isoformat()
            },
            'system_information': self.system_info,
            'test_results_by_component': components,
            'performance_benchmarks': performance_stats,
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # 分析失败的测试
        failed_tests = [r for r in self.validation_results if r.status == ComponentStatus.FAIL]
        warning_tests = [r for r in self.validation_results if r.status == ComponentStatus.WARNING]
        
        if failed_tests:
            recommendations.append(f"发现 {len(failed_tests)} 个严重问题需要立即解决")
            
            # 包相关问题
            failed_packages = [r for r in failed_tests if r.component == "packages"]
            if failed_packages:
                recommendations.append("安装缺失的Python包: pip install -r requirements.txt")
        
        if warning_tests:
            recommendations.append(f"发现 {len(warning_tests)} 个警告需要关注")
            
            # 系统资源警告
            memory_warnings = [r for r in warning_tests if "memory" in r.test_name.lower()]
            if memory_warnings:
                recommendations.append("考虑增加系统内存以获得更好的性能")
            
            # 网络警告
            network_warnings = [r for r in warning_tests if r.component == "network"]
            if network_warnings:
                recommendations.append("检查网络连接设置，确保可以访问交易所API")
        
        # 性能建议
        slow_performance = [r for r in self.validation_results 
                          if r.component == "performance" and r.execution_time_ms > 1000]
        if slow_performance:
            recommendations.append("某些操作性能较慢，考虑优化硬件配置或代码")
        
        if not recommendations:
            recommendations.append("系统配置良好，所有测试均通过")
        
        return recommendations
    
    def version_compare(self, version1: str, version2: str) -> int:
        """比较版本号"""
        try:
            v1_parts = [int(x) for x in version1.split('.')]
            v2_parts = [int(x) for x in version2.split('.')]
            
            # 补齐长度
            max_len = max(len(v1_parts), len(v2_parts))
            v1_parts.extend([0] * (max_len - len(v1_parts)))
            v2_parts.extend([0] * (max_len - len(v2_parts)))
            
            for v1, v2 in zip(v1_parts, v2_parts):
                if v1 > v2:
                    return 1
                elif v1 < v2:
                    return -1
            
            return 0
            
        except Exception:
            return 0
    
    async def save_report(self, report: Dict, filename: str = None):
        """保存验证报告"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"infrastructure_validation_report_{timestamp}.json"
        
        report_path = Path("reports") / filename
        report_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        self.logger.info(f"验证报告已保存: {report_path}")
        return str(report_path)

# 使用示例
async def main():
    """基础设施验证演示"""
    
    # 创建验证器
    validator = InfrastructureConfigValidator()
    
    # 运行全面验证
    print("开始基础设施验证...")
    report = await validator.validate_full_infrastructure(ValidationLevel.COMPREHENSIVE)
    
    # 显示摘要
    summary = report['validation_summary']
    print(f"\n=== 验证摘要 ===")
    print(f"总体状态: {summary['overall_status']}")
    print(f"总测试数: {summary['total_tests']}")
    print(f"通过: {summary['passed']}")
    print(f"失败: {summary['failed']}")
    print(f"警告: {summary['warnings']}")
    print(f"跳过: {summary['skipped']}")
    print(f"总耗时: {summary['total_time_ms']:.2f}ms")
    
    # 显示建议
    print(f"\n=== 改进建议 ===")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    # 保存完整报告
    report_path = await validator.save_report(report)
    print(f"\n完整报告已保存: {report_path}")

if __name__ == "__main__":
    asyncio.run(main())