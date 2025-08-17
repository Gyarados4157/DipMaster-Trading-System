"""
REST API模块
===========

提供高性能的REST API端点。
"""

from .app import create_app
from .endpoints import *

__all__ = [
    "create_app"
]