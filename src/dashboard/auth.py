"""
JWT认证和权限控制系统
实现用户认证、API密钥管理、角色权限控制、请求频率限制
"""

import jwt
import hashlib
import secrets
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog
from enum import Enum

from .config import AuthConfig

logger = structlog.get_logger(__name__)

class Permission(str, Enum):
    """权限枚举"""
    READ = "read"
    WRITE = "write" 
    ADMIN = "admin"
    DELETE = "delete"

class UserRole(str, Enum):
    """用户角色枚举"""
    ADMIN = "admin"
    TRADER = "trader"
    VIEWER = "viewer"
    API_CLIENT = "api_client"

class APIKey:
    """API密钥模型"""
    def __init__(self, key_id: str, key_hash: str, user_id: str, 
                 permissions: List[str], expires_at: Optional[datetime] = None,
                 rate_limit: int = 1000):
        self.key_id = key_id
        self.key_hash = key_hash
        self.user_id = user_id
        self.permissions = permissions
        self.expires_at = expires_at
        self.rate_limit = rate_limit
        self.created_at = datetime.utcnow()
        self.last_used = None
        self.usage_count = 0

class User:
    """用户模型"""
    def __init__(self, user_id: str, username: str, role: UserRole,
                 permissions: List[str], account_ids: List[str] = None):
        self.user_id = user_id
        self.username = username
        self.role = role
        self.permissions = permissions
        self.account_ids = account_ids or []
        self.created_at = datetime.utcnow()
        self.last_login = None
        self.is_active = True

class TokenManager:
    """JWT令牌管理器"""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.secret_key = config.jwt_secret_key
        self.algorithm = config.jwt_algorithm
        self.expiration_hours = config.jwt_expiration_hours
    
    def create_access_token(self, user: User, expires_delta: Optional[timedelta] = None) -> str:
        """创建访问令牌"""
        try:
            if expires_delta:
                expire = datetime.utcnow() + expires_delta
            else:
                expire = datetime.utcnow() + timedelta(hours=self.expiration_hours)
            
            payload = {
                "sub": user.user_id,
                "username": user.username,
                "role": user.role.value,
                "permissions": user.permissions,
                "account_ids": user.account_ids,
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": "access_token"
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"创建访问令牌: 用户={user.username}, 过期时间={expire}")
            return token
            
        except Exception as e:
            logger.error(f"创建访问令牌失败: {e}")
            raise
    
    def create_refresh_token(self, user: User) -> str:
        """创建刷新令牌"""
        try:
            expire = datetime.utcnow() + timedelta(days=30)  # 刷新令牌30天有效期
            
            payload = {
                "sub": user.user_id,
                "exp": expire,
                "iat": datetime.utcnow(),
                "type": "refresh_token"
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            return token
            
        except Exception as e:
            logger.error(f"创建刷新令牌失败: {e}")
            raise
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """验证令牌"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="令牌已过期")
        except jwt.JWTError as e:
            logger.warning(f"令牌验证失败: {e}")
            raise HTTPException(status_code=401, detail="无效的令牌")
    
    def refresh_access_token(self, refresh_token: str, user_store: 'UserStore') -> str:
        """刷新访问令牌"""
        try:
            payload = self.verify_token(refresh_token)
            
            if payload.get("type") != "refresh_token":
                raise HTTPException(status_code=401, detail="无效的刷新令牌")
            
            user_id = payload.get("sub")
            user = user_store.get_user(user_id)
            
            if not user or not user.is_active:
                raise HTTPException(status_code=401, detail="用户不存在或已禁用")
            
            return self.create_access_token(user)
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"刷新令牌失败: {e}")
            raise HTTPException(status_code=401, detail="刷新令牌失败")

class APIKeyManager:
    """API密钥管理器"""
    
    def __init__(self):
        self.api_keys: Dict[str, APIKey] = {}
    
    def generate_api_key(self, user_id: str, permissions: List[str],
                        expires_at: Optional[datetime] = None,
                        rate_limit: int = 1000) -> tuple[str, str]:
        """生成API密钥"""
        try:
            # 生成密钥
            key = secrets.token_urlsafe(32)
            key_id = secrets.token_hex(16)
            
            # 哈希密钥用于存储
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            
            # 创建API密钥对象
            api_key = APIKey(
                key_id=key_id,
                key_hash=key_hash,
                user_id=user_id,
                permissions=permissions,
                expires_at=expires_at,
                rate_limit=rate_limit
            )
            
            self.api_keys[key_id] = api_key
            
            logger.info(f"生成API密钥: 用户={user_id}, 密钥ID={key_id}")
            return key, key_id
            
        except Exception as e:
            logger.error(f"生成API密钥失败: {e}")
            raise
    
    def verify_api_key(self, api_key: str) -> Optional[APIKey]:
        """验证API密钥"""
        try:
            key_hash = hashlib.sha256(api_key.encode()).hexdigest()
            
            for api_key_obj in self.api_keys.values():
                if api_key_obj.key_hash == key_hash:
                    # 检查是否过期
                    if api_key_obj.expires_at and datetime.utcnow() > api_key_obj.expires_at:
                        logger.warning(f"API密钥已过期: {api_key_obj.key_id}")
                        return None
                    
                    # 更新使用记录
                    api_key_obj.last_used = datetime.utcnow()
                    api_key_obj.usage_count += 1
                    
                    return api_key_obj
            
            return None
            
        except Exception as e:
            logger.error(f"验证API密钥失败: {e}")
            return None
    
    def revoke_api_key(self, key_id: str) -> bool:
        """撤销API密钥"""
        try:
            if key_id in self.api_keys:
                del self.api_keys[key_id]
                logger.info(f"撤销API密钥: {key_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"撤销API密钥失败: {e}")
            return False
    
    def list_user_api_keys(self, user_id: str) -> List[Dict[str, Any]]:
        """列出用户的API密钥"""
        user_keys = []
        
        for api_key in self.api_keys.values():
            if api_key.user_id == user_id:
                user_keys.append({
                    "key_id": api_key.key_id,
                    "permissions": api_key.permissions,
                    "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                    "created_at": api_key.created_at.isoformat(),
                    "last_used": api_key.last_used.isoformat() if api_key.last_used else None,
                    "usage_count": api_key.usage_count,
                    "rate_limit": api_key.rate_limit
                })
        
        return user_keys

class UserStore:
    """用户存储管理器"""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.users: Dict[str, User] = {}
        self._init_default_users()
    
    def _init_default_users(self):
        """初始化默认用户"""
        # 创建默认管理员用户
        admin_user = User(
            user_id="admin",
            username="admin",
            role=UserRole.ADMIN,
            permissions=self.config.roles.get("admin", ["read", "write", "admin"]),
            account_ids=["*"]  # 管理员可以访问所有账户
        )
        self.users["admin"] = admin_user
        
        # 创建默认交易员用户
        trader_user = User(
            user_id="trader",
            username="trader",
            role=UserRole.TRADER,
            permissions=self.config.roles.get("trader", ["read", "write"]),
            account_ids=["default"]
        )
        self.users["trader"] = trader_user
        
        # 创建默认查看者用户
        viewer_user = User(
            user_id="viewer",
            username="viewer",
            role=UserRole.VIEWER,
            permissions=self.config.roles.get("viewer", ["read"]),
            account_ids=["default"]
        )
        self.users["viewer"] = viewer_user
        
        logger.info("初始化默认用户完成")
    
    def create_user(self, username: str, role: UserRole, 
                   account_ids: List[str] = None) -> User:
        """创建用户"""
        user_id = secrets.token_hex(16)
        permissions = self.config.roles.get(role.value, ["read"])
        
        user = User(
            user_id=user_id,
            username=username,
            role=role,
            permissions=permissions,
            account_ids=account_ids or []
        )
        
        self.users[user_id] = user
        logger.info(f"创建用户: {username}, 角色: {role.value}")
        return user
    
    def get_user(self, user_id: str) -> Optional[User]:
        """获取用户"""
        return self.users.get(user_id)
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """通过用户名获取用户"""
        for user in self.users.values():
            if user.username == username:
                return user
        return None
    
    def update_user_permissions(self, user_id: str, permissions: List[str]) -> bool:
        """更新用户权限"""
        user = self.get_user(user_id)
        if user:
            user.permissions = permissions
            logger.info(f"更新用户权限: {user.username}, 权限: {permissions}")
            return True
        return False
    
    def deactivate_user(self, user_id: str) -> bool:
        """停用用户"""
        user = self.get_user(user_id)
        if user:
            user.is_active = False
            logger.info(f"停用用户: {user.username}")
            return True
        return False

class RateLimiter:
    """频率限制器"""
    
    def __init__(self):
        self.request_counts: Dict[str, Dict[str, int]] = {}
        self.windows: Dict[str, datetime] = {}
    
    async def check_rate_limit(self, identifier: str, limit: int, 
                             window_seconds: int = 3600) -> bool:
        """检查频率限制"""
        current_time = datetime.utcnow()
        window_key = f"{identifier}:{window_seconds}"
        
        # 检查时间窗口是否需要重置
        if (window_key not in self.windows or 
            (current_time - self.windows[window_key]).seconds >= window_seconds):
            self.windows[window_key] = current_time
            self.request_counts[window_key] = 0
        
        # 检查请求计数
        current_count = self.request_counts.get(window_key, 0)
        if current_count >= limit:
            return False
        
        # 增加请求计数
        self.request_counts[window_key] = current_count + 1
        return True

class AuthManager:
    """认证管理器主类"""
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.token_manager = TokenManager(config)
        self.api_key_manager = APIKeyManager()
        self.user_store = UserStore(config)
        self.rate_limiter = RateLimiter()
        self.security = HTTPBearer()
    
    async def authenticate_request(self, credentials: HTTPAuthorizationCredentials) -> User:
        """认证请求"""
        token = credentials.credentials
        
        try:
            # 首先尝试JWT令牌认证
            payload = self.token_manager.verify_token(token)
            user_id = payload.get("sub")
            
            user = self.user_store.get_user(user_id)
            if not user or not user.is_active:
                raise HTTPException(status_code=401, detail="用户不存在或已禁用")
            
            # 更新最后登录时间
            user.last_login = datetime.utcnow()
            return user
            
        except HTTPException:
            # JWT认证失败，尝试API密钥认证
            api_key_obj = self.api_key_manager.verify_api_key(token)
            if api_key_obj:
                user = self.user_store.get_user(api_key_obj.user_id)
                if user and user.is_active:
                    # 检查API密钥的频率限制
                    rate_limit_ok = await self.rate_limiter.check_rate_limit(
                        f"api_key:{api_key_obj.key_id}",
                        api_key_obj.rate_limit
                    )
                    
                    if not rate_limit_ok:
                        raise HTTPException(status_code=429, detail="API密钥频率限制")
                    
                    return user
            
            raise HTTPException(status_code=401, detail="认证失败")
    
    def check_permission(self, user: User, required_permission: str) -> bool:
        """检查用户权限"""
        if required_permission in user.permissions:
            return True
        
        # 管理员权限可以访问所有资源
        if "admin" in user.permissions:
            return True
        
        return False
    
    def check_account_access(self, user: User, account_id: str) -> bool:
        """检查账户访问权限"""
        # 管理员可以访问所有账户
        if "*" in user.account_ids or "admin" in user.permissions:
            return True
        
        return account_id in user.account_ids
    
    async def login(self, username: str, password: str = None) -> Dict[str, str]:
        """用户登录（简化版，实际项目中需要密码验证）"""
        user = self.user_store.get_user_by_username(username)
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        
        # 这里应该验证密码，为了演示省略密码验证
        access_token = self.token_manager.create_access_token(user)
        refresh_token = self.token_manager.create_refresh_token(user)
        
        user.last_login = datetime.utcnow()
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": self.config.jwt_expiration_hours * 3600
        }
    
    async def refresh_token(self, refresh_token: str) -> Dict[str, str]:
        """刷新访问令牌"""
        access_token = self.token_manager.refresh_access_token(refresh_token, self.user_store)
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": self.config.jwt_expiration_hours * 3600
        }
    
    async def create_api_key(self, user_id: str, permissions: List[str],
                           expires_at: Optional[datetime] = None,
                           rate_limit: int = 1000) -> Dict[str, str]:
        """创建API密钥"""
        user = self.user_store.get_user(user_id)
        if not user:
            raise HTTPException(status_code=404, detail="用户不存在")
        
        # 验证权限范围
        for permission in permissions:
            if permission not in user.permissions and "admin" not in user.permissions:
                raise HTTPException(status_code=403, detail=f"权限不足: {permission}")
        
        api_key, key_id = self.api_key_manager.generate_api_key(
            user_id, permissions, expires_at, rate_limit
        )
        
        return {
            "api_key": api_key,
            "key_id": key_id,
            "permissions": permissions,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "rate_limit": rate_limit
        }

# FastAPI依赖函数

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> User:
    """获取当前用户（FastAPI依赖）"""
    from .main import dashboard_service
    
    if not dashboard_service or not dashboard_service.auth_manager:
        raise HTTPException(status_code=503, detail="认证服务不可用")
    
    return await dashboard_service.auth_manager.authenticate_request(credentials)

async def require_permission(user: User, permission: str):
    """要求特定权限（FastAPI依赖）"""
    from .main import dashboard_service
    
    if not dashboard_service or not dashboard_service.auth_manager:
        raise HTTPException(status_code=503, detail="认证服务不可用")
    
    if not dashboard_service.auth_manager.check_permission(user, permission):
        raise HTTPException(status_code=403, detail=f"权限不足: {permission}")

async def require_account_access(user: User, account_id: str):
    """要求账户访问权限（FastAPI依赖）"""
    from .main import dashboard_service
    
    if not dashboard_service or not dashboard_service.auth_manager:
        raise HTTPException(status_code=503, detail="认证服务不可用")
    
    if not dashboard_service.auth_manager.check_account_access(user, account_id):
        raise HTTPException(status_code=403, detail=f"无权限访问账户: {account_id}")

async def verify_websocket_token(token: str, auth_manager: AuthManager) -> Optional[Dict[str, Any]]:
    """验证WebSocket令牌"""
    try:
        payload = auth_manager.token_manager.verify_token(token)
        user_id = payload.get("sub")
        
        user = auth_manager.user_store.get_user(user_id)
        if not user or not user.is_active:
            return None
        
        return {
            "user_id": user_id,
            "username": user.username,
            "permissions": user.permissions,
            "account_ids": user.account_ids
        }
        
    except Exception as e:
        logger.warning(f"WebSocket令牌验证失败: {e}")
        return None