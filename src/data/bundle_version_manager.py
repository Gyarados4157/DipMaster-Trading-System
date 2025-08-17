"""
MarketDataBundle Version Management System
市场数据包版本管理系统 - 支持完整的数据生命周期管理
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import hashlib
import shutil
import zipfile
import tarfile
import sqlite3
from dataclasses import dataclass, asdict
from enum import Enum
import git
import semver
import pickle
import yaml
from collections import defaultdict, OrderedDict
import threading
import time
import warnings

warnings.filterwarnings('ignore')

class VersionStatus(Enum):
    """版本状态"""
    ACTIVE = "active"           # 活跃版本
    ARCHIVED = "archived"       # 已归档
    DEPRECATED = "deprecated"   # 已弃用
    FAILED = "failed"          # 创建失败
    TESTING = "testing"        # 测试中

class ConflictResolution(Enum):
    """冲突解决策略"""
    OVERWRITE = "overwrite"     # 覆盖
    MERGE = "merge"            # 合并
    SKIP = "skip"              # 跳过
    ERROR = "error"            # 报错

class BackupType(Enum):
    """备份类型"""
    FULL = "full"              # 全量备份
    INCREMENTAL = "incremental" # 增量备份
    DIFFERENTIAL = "differential" # 差异备份

@dataclass
class BundleVersion:
    """数据包版本信息"""
    version_id: str
    semantic_version: str  # 如 "1.2.3"
    timestamp: datetime
    description: str
    author: str
    status: VersionStatus
    parent_version: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None
    quality_score: float = 0.0
    file_hash: str = ""
    file_size_mb: float = 0.0
    symbols: List[str] = None
    exchanges: List[str] = None
    data_sources: Dict[str, Any] = None
    performance_metrics: Dict[str, Any] = None

@dataclass
class VersionDiff:
    """版本差异"""
    added_symbols: List[str]
    removed_symbols: List[str]
    modified_symbols: List[str]
    quality_changes: Dict[str, float]
    size_change_mb: float
    record_count_change: int

@dataclass
class BackupConfig:
    """备份配置"""
    backup_type: BackupType
    retention_days: int
    compression_enabled: bool
    encryption_enabled: bool
    remote_backup_enabled: bool
    backup_schedule: str  # cron格式

class BundleVersionManager:
    """数据包版本管理器"""
    
    def __init__(self, workspace_path: str = "data/bundle_versions"):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(exist_ok=True, parents=True)
        
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # 版本存储路径
        self.versions_path = self.workspace_path / "versions"
        self.versions_path.mkdir(exist_ok=True, parents=True)
        
        # 备份路径
        self.backups_path = self.workspace_path / "backups"
        self.backups_path.mkdir(exist_ok=True, parents=True)
        
        # 临时路径
        self.temp_path = self.workspace_path / "temp"
        self.temp_path.mkdir(exist_ok=True, parents=True)
        
        # 数据库初始化
        self.db_path = self.workspace_path / "version_registry.db"
        self.init_database()
        
        # Git仓库初始化（用于版本控制）
        self.git_repo_path = self.workspace_path / "git_repo"
        self.init_git_repository()
        
        # 版本缓存
        self.version_cache = OrderedDict()
        self.cache_lock = threading.RLock()
        
        # 当前活跃版本
        self.current_version = None
        self.load_current_version()
        
        # 备份配置
        self.backup_config = BackupConfig(
            backup_type=BackupType.INCREMENTAL,
            retention_days=90,
            compression_enabled=True,
            encryption_enabled=False,
            remote_backup_enabled=False,
            backup_schedule="0 2 * * *"  # 每天凌晨2点
        )
        
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s',
            handlers=[
                logging.FileHandler('logs/bundle_version_manager.log'),
                logging.StreamHandler()
            ]
        )
    
    def init_database(self):
        """初始化版本数据库"""
        self.db_conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.db_lock = threading.Lock()
        
        cursor = self.db_conn.cursor()
        
        # 版本表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bundle_versions (
                version_id TEXT PRIMARY KEY,
                semantic_version TEXT UNIQUE,
                timestamp TEXT,
                description TEXT,
                author TEXT,
                status TEXT,
                parent_version TEXT,
                tags TEXT,
                metadata TEXT,
                quality_score REAL,
                file_hash TEXT,
                file_size_mb REAL,
                symbols TEXT,
                exchanges TEXT,
                data_sources TEXT,
                performance_metrics TEXT
            )
        """)
        
        # 版本依赖表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS version_dependencies (
                version_id TEXT,
                dependency_id TEXT,
                dependency_type TEXT,
                PRIMARY KEY (version_id, dependency_id)
            )
        """)
        
        # 版本变更日志表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS version_changelog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT,
                operation TEXT,
                timestamp TEXT,
                details TEXT
            )
        """)
        
        # 备份记录表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backup_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version_id TEXT,
                backup_type TEXT,
                backup_path TEXT,
                timestamp TEXT,
                file_size_mb REAL,
                status TEXT
            )
        """)
        
        self.db_conn.commit()
    
    def init_git_repository(self):
        """初始化Git仓库"""
        try:
            if not (self.git_repo_path / ".git").exists():
                self.git_repo_path.mkdir(exist_ok=True, parents=True)
                self.git_repo = git.Repo.init(self.git_repo_path)
                
                # 初始提交
                gitignore_content = """
# Temporary files
*.tmp
*.temp

# Large data files
*.parquet
*.hdf5
*.zarr/

# Cache files
__pycache__/
*.pyc
*.pyo
                """
                
                gitignore_path = self.git_repo_path / ".gitignore"
                with open(gitignore_path, 'w', encoding='utf-8') as f:
                    f.write(gitignore_content)
                
                self.git_repo.index.add([".gitignore"])
                self.git_repo.index.commit("Initial commit")
                
                self.logger.info("Git仓库初始化完成")
            else:
                self.git_repo = git.Repo(self.git_repo_path)
                
        except Exception as e:
            self.logger.warning(f"Git仓库初始化失败: {e}")
            self.git_repo = None
    
    def load_current_version(self):
        """加载当前活跃版本"""
        try:
            current_file = self.workspace_path / "current_version.txt"
            if current_file.exists():
                self.current_version = current_file.read_text(encoding='utf-8').strip()
            else:
                # 查找最新的活跃版本
                with self.db_lock:
                    cursor = self.db_conn.cursor()
                    cursor.execute("""
                        SELECT version_id FROM bundle_versions 
                        WHERE status = 'active' 
                        ORDER BY timestamp DESC 
                        LIMIT 1
                    """)
                    result = cursor.fetchone()
                    if result:
                        self.current_version = result[0]
                        self.set_current_version(self.current_version)
                        
        except Exception as e:
            self.logger.error(f"加载当前版本失败: {e}")
    
    def set_current_version(self, version_id: str):
        """设置当前版本"""
        current_file = self.workspace_path / "current_version.txt"
        with open(current_file, 'w', encoding='utf-8') as f:
            f.write(version_id)
        self.current_version = version_id
    
    async def create_version(self,
                           bundle_data: Dict,
                           description: str,
                           author: str = "system",
                           tags: List[str] = None,
                           parent_version: str = None) -> str:
        """创建新版本"""
        try:
            # 生成版本ID和语义版本
            version_id = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 自动确定语义版本
            semantic_version = await self.determine_semantic_version(bundle_data, parent_version)
            
            # 计算文件哈希
            file_hash = self.calculate_bundle_hash(bundle_data)
            
            # 创建版本对象
            version = BundleVersion(
                version_id=version_id,
                semantic_version=semantic_version,
                timestamp=datetime.now(),
                description=description,
                author=author,
                status=VersionStatus.TESTING,
                parent_version=parent_version or self.current_version,
                tags=tags or [],
                metadata=bundle_data.get('metadata', {}),
                quality_score=bundle_data.get('metadata', {}).get('data_quality_score', 0.0),
                file_hash=file_hash,
                symbols=bundle_data.get('metadata', {}).get('symbols', []),
                exchanges=bundle_data.get('metadata', {}).get('exchanges', []),
                data_sources=bundle_data.get('data_sources', {}),
                performance_metrics=bundle_data.get('performance_benchmarks', {})
            )
            
            # 创建版本目录
            version_dir = self.versions_path / version_id
            version_dir.mkdir(exist_ok=True, parents=True)
            
            # 保存数据包
            await self.save_bundle_data(version_dir, bundle_data)
            
            # 计算文件大小
            version.file_size_mb = self.calculate_directory_size(version_dir)
            
            # 保存版本到数据库
            await self.save_version_to_db(version)
            
            # Git提交
            if self.git_repo:
                await self.commit_to_git(version)
            
            # 添加到缓存
            with self.cache_lock:
                self.version_cache[version_id] = version
                if len(self.version_cache) > 100:  # 限制缓存大小
                    self.version_cache.popitem(last=False)
            
            # 记录变更日志
            await self.log_version_operation(version_id, "CREATE", {
                "description": description,
                "author": author,
                "parent_version": parent_version
            })
            
            self.logger.info(f"版本创建成功: {version_id} ({semantic_version})")
            
            return version_id
            
        except Exception as e:
            self.logger.error(f"创建版本失败: {e}")
            raise
    
    async def determine_semantic_version(self, bundle_data: Dict, parent_version: str = None) -> str:
        """确定语义版本号"""
        try:
            if not parent_version:
                return "1.0.0"  # 首个版本
            
            # 获取父版本信息
            parent_version_info = await self.get_version(parent_version)
            if not parent_version_info:
                return "1.0.0"
            
            parent_semantic = parent_version_info.semantic_version
            
            # 分析变更类型
            diff = await self.compare_with_parent(bundle_data, parent_version)
            
            # 根据变更确定版本号增量
            if self.is_major_change(diff):
                # 主版本号增加（破坏性变更）
                return semver.bump_major(parent_semantic)
            elif self.is_minor_change(diff):
                # 次版本号增加（新功能）
                return semver.bump_minor(parent_semantic)
            else:
                # 补丁版本号增加（修复）
                return semver.bump_patch(parent_semantic)
                
        except Exception as e:
            self.logger.error(f"确定语义版本失败: {e}")
            return "1.0.0"
    
    def is_major_change(self, diff: VersionDiff) -> bool:
        """判断是否为主要变更"""
        # 移除了交易对或交易所
        if diff.removed_symbols or (hasattr(diff, 'removed_exchanges') and diff.removed_exchanges):
            return True
        
        # 质量分数大幅下降
        if any(change < -0.1 for change in diff.quality_changes.values()):
            return True
        
        return False
    
    def is_minor_change(self, diff: VersionDiff) -> bool:
        """判断是否为次要变更"""
        # 添加了新的交易对
        if diff.added_symbols:
            return True
        
        # 数据量显著增加
        if diff.size_change_mb > 100 or diff.record_count_change > 100000:
            return True
        
        return False
    
    async def compare_with_parent(self, bundle_data: Dict, parent_version: str) -> VersionDiff:
        """与父版本比较"""
        try:
            parent_data = await self.load_bundle_data(parent_version)
            
            if not parent_data:
                return VersionDiff([], [], [], {}, 0, 0)
            
            # 符号比较
            current_symbols = set(bundle_data.get('metadata', {}).get('symbols', []))
            parent_symbols = set(parent_data.get('metadata', {}).get('symbols', []))
            
            added_symbols = list(current_symbols - parent_symbols)
            removed_symbols = list(parent_symbols - current_symbols)
            common_symbols = list(current_symbols & parent_symbols)
            
            # 质量变化
            quality_changes = {}
            current_quality = bundle_data.get('metadata', {}).get('data_quality_score', 0)
            parent_quality = parent_data.get('metadata', {}).get('data_quality_score', 0)
            quality_changes['overall'] = current_quality - parent_quality
            
            # 大小变化
            current_size = bundle_data.get('metadata', {}).get('total_size_mb', 0)
            parent_size = parent_data.get('metadata', {}).get('total_size_mb', 0)
            size_change_mb = current_size - parent_size
            
            # 记录数变化
            current_records = bundle_data.get('metadata', {}).get('total_records', 0)
            parent_records = parent_data.get('metadata', {}).get('total_records', 0)
            record_count_change = current_records - parent_records
            
            return VersionDiff(
                added_symbols=added_symbols,
                removed_symbols=removed_symbols,
                modified_symbols=common_symbols,  # 简化处理
                quality_changes=quality_changes,
                size_change_mb=size_change_mb,
                record_count_change=record_count_change
            )
            
        except Exception as e:
            self.logger.error(f"版本比较失败: {e}")
            return VersionDiff([], [], [], {}, 0, 0)
    
    def calculate_bundle_hash(self, bundle_data: Dict) -> str:
        """计算数据包哈希"""
        try:
            # 使用关键字段计算哈希
            hash_data = {
                'symbols': sorted(bundle_data.get('metadata', {}).get('symbols', [])),
                'exchanges': sorted(bundle_data.get('metadata', {}).get('exchanges', [])),
                'quality_score': bundle_data.get('metadata', {}).get('data_quality_score', 0),
                'data_sources_keys': sorted(bundle_data.get('data_sources', {}).keys())
            }
            
            hash_string = json.dumps(hash_data, sort_keys=True)
            return hashlib.sha256(hash_string.encode()).hexdigest()[:16]
            
        except Exception as e:
            self.logger.error(f"计算哈希失败: {e}")
            return ""
    
    async def save_bundle_data(self, version_dir: Path, bundle_data: Dict):
        """保存数据包数据"""
        try:
            # 保存主要配置文件
            bundle_file = version_dir / "bundle.json"
            with open(bundle_file, 'w', encoding='utf-8') as f:
                json.dump(bundle_data, f, ensure_ascii=False, indent=2, default=str)
            
            # 保存YAML格式（便于人类阅读）
            yaml_file = version_dir / "bundle.yaml"
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(bundle_data, f, default_flow_style=False, allow_unicode=True)
            
            # 创建符号链接到实际数据文件（避免重复存储）
            data_sources = bundle_data.get('data_sources', {})
            if 'historical' in data_sources:
                for timeframe, symbols_data in data_sources['historical'].items():
                    if isinstance(symbols_data, dict):
                        for symbol, file_info in symbols_data.items():
                            if isinstance(file_info, dict) and 'file_path' in file_info:
                                self.create_data_link(version_dir, file_info['file_path'], f"{symbol}_{timeframe}")
            
        except Exception as e:
            self.logger.error(f"保存数据包失败: {e}")
            raise
    
    def create_data_link(self, version_dir: Path, source_path: str, link_name: str):
        """创建数据文件链接"""
        try:
            source_file = Path(source_path)
            if source_file.exists():
                links_dir = version_dir / "data_links"
                links_dir.mkdir(exist_ok=True, parents=True)
                
                link_file = links_dir / f"{link_name}.link"
                with open(link_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'source_path': str(source_file),
                        'created_at': datetime.now().isoformat(),
                        'file_size_mb': source_file.stat().st_size / 1024 / 1024
                    }, f, indent=2)
                    
        except Exception as e:
            self.logger.warning(f"创建数据链接失败: {e}")
    
    def calculate_directory_size(self, directory: Path) -> float:
        """计算目录大小（MB）"""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / 1024 / 1024
    
    async def save_version_to_db(self, version: BundleVersion):
        """保存版本到数据库"""
        with self.db_lock:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO bundle_versions 
                (version_id, semantic_version, timestamp, description, author, status, 
                 parent_version, tags, metadata, quality_score, file_hash, file_size_mb,
                 symbols, exchanges, data_sources, performance_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                version.version_id,
                version.semantic_version,
                version.timestamp.isoformat(),
                version.description,
                version.author,
                version.status.value,
                version.parent_version,
                json.dumps(version.tags or []),
                json.dumps(version.metadata or {}),
                version.quality_score,
                version.file_hash,
                version.file_size_mb,
                json.dumps(version.symbols or []),
                json.dumps(version.exchanges or []),
                json.dumps(version.data_sources or {}),
                json.dumps(version.performance_metrics or {})
            ))
            self.db_conn.commit()
    
    async def commit_to_git(self, version: BundleVersion):
        """提交到Git"""
        try:
            if not self.git_repo:
                return
            
            # 复制版本文件到Git仓库
            version_dir = self.versions_path / version.version_id
            git_version_dir = self.git_repo_path / "versions" / version.version_id
            git_version_dir.mkdir(exist_ok=True, parents=True)
            
            # 只复制配置文件，不复制大型数据文件
            for file_name in ["bundle.json", "bundle.yaml"]:
                source_file = version_dir / file_name
                if source_file.exists():
                    target_file = git_version_dir / file_name
                    shutil.copy2(source_file, target_file)
            
            # Git提交
            self.git_repo.index.add([str(git_version_dir)])
            commit_message = f"Add version {version.semantic_version}: {version.description}"
            self.git_repo.index.commit(commit_message)
            
            # 创建标签
            tag_name = f"v{version.semantic_version}"
            self.git_repo.create_tag(tag_name, message=f"Version {version.semantic_version}")
            
        except Exception as e:
            self.logger.warning(f"Git提交失败: {e}")
    
    async def log_version_operation(self, version_id: str, operation: str, details: Dict):
        """记录版本操作日志"""
        with self.db_lock:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO version_changelog (version_id, operation, timestamp, details)
                VALUES (?, ?, ?, ?)
            """, (
                version_id,
                operation,
                datetime.now().isoformat(),
                json.dumps(details)
            ))
            self.db_conn.commit()
    
    async def get_version(self, version_id: str) -> Optional[BundleVersion]:
        """获取版本信息"""
        # 先检查缓存
        with self.cache_lock:
            if version_id in self.version_cache:
                return self.version_cache[version_id]
        
        # 从数据库查询
        with self.db_lock:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT * FROM bundle_versions WHERE version_id = ?
            """, (version_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # 构造版本对象
            version = BundleVersion(
                version_id=row[0],
                semantic_version=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                description=row[3],
                author=row[4],
                status=VersionStatus(row[5]),
                parent_version=row[6],
                tags=json.loads(row[7]) if row[7] else [],
                metadata=json.loads(row[8]) if row[8] else {},
                quality_score=row[9],
                file_hash=row[10],
                file_size_mb=row[11],
                symbols=json.loads(row[12]) if row[12] else [],
                exchanges=json.loads(row[13]) if row[13] else [],
                data_sources=json.loads(row[14]) if row[14] else {},
                performance_metrics=json.loads(row[15]) if row[15] else {}
            )
            
            # 添加到缓存
            with self.cache_lock:
                self.version_cache[version_id] = version
            
            return version
    
    async def load_bundle_data(self, version_id: str) -> Optional[Dict]:
        """加载数据包数据"""
        try:
            version_dir = self.versions_path / version_id
            bundle_file = version_dir / "bundle.json"
            
            if not bundle_file.exists():
                return None
            
            with open(bundle_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"加载数据包失败: {e}")
            return None
    
    async def activate_version(self, version_id: str) -> bool:
        """激活版本"""
        try:
            version = await self.get_version(version_id)
            if not version:
                self.logger.error(f"版本不存在: {version_id}")
                return False
            
            # 更新状态
            version.status = VersionStatus.ACTIVE
            await self.save_version_to_db(version)
            
            # 设置为当前版本
            self.set_current_version(version_id)
            
            # 记录操作
            await self.log_version_operation(version_id, "ACTIVATE", {"timestamp": datetime.now().isoformat()})
            
            self.logger.info(f"版本已激活: {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"激活版本失败: {e}")
            return False
    
    async def rollback_to_version(self, version_id: str) -> bool:
        """回滚到指定版本"""
        try:
            version = await self.get_version(version_id)
            if not version:
                self.logger.error(f"版本不存在: {version_id}")
                return False
            
            # 创建回滚备份
            current_backup = await self.create_backup(self.current_version, BackupType.FULL)
            
            # 执行回滚
            success = await self.activate_version(version_id)
            
            if success:
                # 记录回滚操作
                await self.log_version_operation(version_id, "ROLLBACK", {
                    "from_version": self.current_version,
                    "backup_created": current_backup,
                    "timestamp": datetime.now().isoformat()
                })
                
                self.logger.info(f"成功回滚到版本: {version_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"回滚版本失败: {e}")
            return False
    
    async def create_backup(self, version_id: str, backup_type: BackupType) -> str:
        """创建备份"""
        try:
            if not version_id:
                return ""
            
            backup_id = f"backup_{version_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir = self.backups_path / backup_id
            backup_dir.mkdir(exist_ok=True, parents=True)
            
            # 备份版本数据
            source_dir = self.versions_path / version_id
            if source_dir.exists():
                shutil.copytree(source_dir, backup_dir / "version_data", dirs_exist_ok=True)
            
            # 创建备份元数据
            backup_metadata = {
                'backup_id': backup_id,
                'version_id': version_id,
                'backup_type': backup_type.value,
                'created_at': datetime.now().isoformat(),
                'backup_size_mb': self.calculate_directory_size(backup_dir)
            }
            
            metadata_file = backup_dir / "backup_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(backup_metadata, f, indent=2)
            
            # 压缩备份
            if self.backup_config.compression_enabled:
                await self.compress_backup(backup_dir)
            
            # 记录备份
            with self.db_lock:
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    INSERT INTO backup_records 
                    (version_id, backup_type, backup_path, timestamp, file_size_mb, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    version_id,
                    backup_type.value,
                    str(backup_dir),
                    datetime.now().isoformat(),
                    backup_metadata['backup_size_mb'],
                    'completed'
                ))
                self.db_conn.commit()
            
            self.logger.info(f"备份创建完成: {backup_id}")
            return backup_id
            
        except Exception as e:
            self.logger.error(f"创建备份失败: {e}")
            return ""
    
    async def compress_backup(self, backup_dir: Path):
        """压缩备份"""
        try:
            archive_path = backup_dir.parent / f"{backup_dir.name}.tar.gz"
            
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(backup_dir, arcname=backup_dir.name)
            
            # 删除原目录
            shutil.rmtree(backup_dir)
            
        except Exception as e:
            self.logger.error(f"压缩备份失败: {e}")
    
    async def cleanup_old_versions(self, retention_days: int = 90):
        """清理旧版本"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            with self.db_lock:
                cursor = self.db_conn.cursor()
                
                # 查找过期版本
                cursor.execute("""
                    SELECT version_id FROM bundle_versions 
                    WHERE timestamp < ? AND status != 'active'
                """, (cutoff_date.isoformat(),))
                
                old_versions = [row[0] for row in cursor.fetchall()]
            
            for version_id in old_versions:
                await self.archive_version(version_id)
            
            self.logger.info(f"清理了 {len(old_versions)} 个旧版本")
            
        except Exception as e:
            self.logger.error(f"清理旧版本失败: {e}")
    
    async def archive_version(self, version_id: str):
        """归档版本"""
        try:
            # 创建备份
            backup_id = await self.create_backup(version_id, BackupType.FULL)
            
            # 更新版本状态
            version = await self.get_version(version_id)
            if version:
                version.status = VersionStatus.ARCHIVED
                await self.save_version_to_db(version)
            
            # 删除版本目录
            version_dir = self.versions_path / version_id
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            # 记录操作
            await self.log_version_operation(version_id, "ARCHIVE", {
                "backup_id": backup_id,
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info(f"版本已归档: {version_id}")
            
        except Exception as e:
            self.logger.error(f"归档版本失败: {e}")
    
    async def list_versions(self, 
                          status: VersionStatus = None,
                          limit: int = 50,
                          offset: int = 0) -> List[BundleVersion]:
        """列出版本"""
        try:
            with self.db_lock:
                cursor = self.db_conn.cursor()
                
                if status:
                    cursor.execute("""
                        SELECT * FROM bundle_versions 
                        WHERE status = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ? OFFSET ?
                    """, (status.value, limit, offset))
                else:
                    cursor.execute("""
                        SELECT * FROM bundle_versions 
                        ORDER BY timestamp DESC 
                        LIMIT ? OFFSET ?
                    """, (limit, offset))
                
                versions = []
                for row in cursor.fetchall():
                    version = BundleVersion(
                        version_id=row[0],
                        semantic_version=row[1],
                        timestamp=datetime.fromisoformat(row[2]),
                        description=row[3],
                        author=row[4],
                        status=VersionStatus(row[5]),
                        parent_version=row[6],
                        tags=json.loads(row[7]) if row[7] else [],
                        metadata=json.loads(row[8]) if row[8] else {},
                        quality_score=row[9],
                        file_hash=row[10],
                        file_size_mb=row[11],
                        symbols=json.loads(row[12]) if row[12] else [],
                        exchanges=json.loads(row[13]) if row[13] else [],
                        data_sources=json.loads(row[14]) if row[14] else {},
                        performance_metrics=json.loads(row[15]) if row[15] else {}
                    )
                    versions.append(version)
                
                return versions
                
        except Exception as e:
            self.logger.error(f"列出版本失败: {e}")
            return []
    
    async def compare_versions(self, version1_id: str, version2_id: str) -> Dict:
        """比较两个版本"""
        try:
            v1_data = await self.load_bundle_data(version1_id)
            v2_data = await self.load_bundle_data(version2_id)
            
            if not v1_data or not v2_data:
                return {"error": "版本数据不存在"}
            
            # 详细比较
            comparison = {
                'version1': version1_id,
                'version2': version2_id,
                'symbols': {
                    'v1_only': list(set(v1_data.get('metadata', {}).get('symbols', [])) - 
                                  set(v2_data.get('metadata', {}).get('symbols', []))),
                    'v2_only': list(set(v2_data.get('metadata', {}).get('symbols', [])) - 
                                  set(v1_data.get('metadata', {}).get('symbols', []))),
                    'common': list(set(v1_data.get('metadata', {}).get('symbols', [])) & 
                                 set(v2_data.get('metadata', {}).get('symbols', [])))
                },
                'quality_scores': {
                    'v1': v1_data.get('metadata', {}).get('data_quality_score', 0),
                    'v2': v2_data.get('metadata', {}).get('data_quality_score', 0),
                    'difference': v2_data.get('metadata', {}).get('data_quality_score', 0) - 
                                v1_data.get('metadata', {}).get('data_quality_score', 0)
                },
                'size_comparison': {
                    'v1_mb': v1_data.get('metadata', {}).get('total_size_mb', 0),
                    'v2_mb': v2_data.get('metadata', {}).get('total_size_mb', 0),
                    'difference_mb': v2_data.get('metadata', {}).get('total_size_mb', 0) - 
                                   v1_data.get('metadata', {}).get('total_size_mb', 0)
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"版本比较失败: {e}")
            return {"error": str(e)}
    
    def get_version_statistics(self) -> Dict:
        """获取版本统计信息"""
        try:
            with self.db_lock:
                cursor = self.db_conn.cursor()
                
                # 总版本数
                cursor.execute("SELECT COUNT(*) FROM bundle_versions")
                total_versions = cursor.fetchone()[0]
                
                # 按状态统计
                cursor.execute("""
                    SELECT status, COUNT(*) FROM bundle_versions 
                    GROUP BY status
                """)
                status_counts = dict(cursor.fetchall())
                
                # 最新版本
                cursor.execute("""
                    SELECT version_id, semantic_version FROM bundle_versions 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                latest_version = cursor.fetchone()
                
                # 总存储大小
                cursor.execute("SELECT SUM(file_size_mb) FROM bundle_versions")
                total_size_mb = cursor.fetchone()[0] or 0
                
                # 平均质量分数
                cursor.execute("SELECT AVG(quality_score) FROM bundle_versions")
                avg_quality = cursor.fetchone()[0] or 0
                
                return {
                    'total_versions': total_versions,
                    'status_distribution': status_counts,
                    'latest_version': {
                        'id': latest_version[0] if latest_version else None,
                        'semantic_version': latest_version[1] if latest_version else None
                    },
                    'current_version': self.current_version,
                    'total_storage_mb': total_size_mb,
                    'average_quality_score': avg_quality,
                    'workspace_path': str(self.workspace_path)
                }
                
        except Exception as e:
            self.logger.error(f"获取统计信息失败: {e}")
            return {}

# 使用示例
async def main():
    """版本管理系统演示"""
    
    # 初始化版本管理器
    version_manager = BundleVersionManager()
    
    # 模拟数据包
    sample_bundle = {
        "version": "2025-08-17T10:30:00Z",
        "metadata": {
            "bundle_id": "dipmaster_sample_20250817_103000",
            "strategy_name": "DipMaster_Sample",
            "description": "示例数据包",
            "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
            "exchanges": ["binance"],
            "data_quality_score": 0.995,
            "total_size_mb": 150.5,
            "total_records": 500000
        },
        "data_sources": {
            "historical": {
                "5m": {
                    "BTCUSDT": {"file_path": "data/BTCUSDT_5m.parquet"},
                    "ETHUSDT": {"file_path": "data/ETHUSDT_5m.parquet"}
                }
            }
        },
        "performance_benchmarks": {
            "data_access_latency_ms": 25,
            "query_throughput_ops": 2000
        }
    }
    
    # 创建版本
    version_id = await version_manager.create_version(
        sample_bundle,
        description="初始版本",
        author="demo_user",
        tags=["demo", "sample"]
    )
    
    print(f"创建版本: {version_id}")
    
    # 激活版本
    await version_manager.activate_version(version_id)
    print(f"激活版本: {version_id}")
    
    # 创建第二个版本
    sample_bundle['metadata']['symbols'].append("ADAUSDT")
    sample_bundle['metadata']['data_quality_score'] = 0.997
    
    version_id_2 = await version_manager.create_version(
        sample_bundle,
        description="添加ADA支持",
        author="demo_user",
        tags=["demo", "ada"],
        parent_version=version_id
    )
    
    print(f"创建版本2: {version_id_2}")
    
    # 列出版本
    versions = await version_manager.list_versions(limit=10)
    print(f"\n版本列表:")
    for v in versions:
        print(f"- {v.version_id} ({v.semantic_version}): {v.description} [{v.status.value}]")
    
    # 比较版本
    comparison = await version_manager.compare_versions(version_id, version_id_2)
    print(f"\n版本比较:")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    
    # 统计信息
    stats = version_manager.get_version_statistics()
    print(f"\n统计信息:")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    # 创建备份
    backup_id = await version_manager.create_backup(version_id_2, BackupType.FULL)
    print(f"\n创建备份: {backup_id}")

if __name__ == "__main__":
    asyncio.run(main())