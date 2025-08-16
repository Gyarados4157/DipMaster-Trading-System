"""
DipMaster Strategy Validation Package
严格的策略验证系统，解决过拟合问题

Author: DipMaster Trading Team
Date: 2025-08-15
Version: 1.0.0 (Anti-Overfitting Edition)
"""

from .data_splitter import DataSplitter, DataSplitConfig
from .statistical_validator import StatisticalValidator
from .walk_forward_analyzer import WalkForwardAnalyzer, WalkForwardConfig
from .overfitting_detector_v2 import OverfittingDetectorV2
from .multi_asset_validator import MultiAssetValidator
from .comprehensive_validator import ComprehensiveValidator, ValidationConfig

__all__ = [
    'DataSplitter',
    'DataSplitConfig',
    'StatisticalValidator', 
    'WalkForwardAnalyzer',
    'WalkForwardConfig',
    'OverfittingDetectorV2',
    'MultiAssetValidator',
    'ComprehensiveValidator',
    'ValidationConfig'
]