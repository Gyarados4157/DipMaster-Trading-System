#!/usr/bin/env python3
"""
DipMaster Analysis Configuration
DipMaster 分析配置文件

This file contains all configuration settings and paths for the analysis.
"""

import os
from pathlib import Path

# 基础路径配置
BASE_DIR = Path(__file__).parent.absolute()
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'
SCRIPTS_DIR = BASE_DIR / 'scripts'
TOOLS_DIR = BASE_DIR / 'tools'
DOCS_DIR = BASE_DIR / 'docs'

# 数据路径
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MARKET_DATA_DIR = DATA_DIR / 'market_data'

# 结果路径
REPORTS_DIR = RESULTS_DIR / 'reports'
CHARTS_DIR = RESULTS_DIR / 'charts'
ANALYSIS_DIR = RESULTS_DIR / 'analysis'

# 原始数据文件
DIPMASTER_CSV_FILE = RAW_DATA_DIR / 'all-trades-Binance-DipMaster AI-2025-08-12T04-51-41.csv'

# 确保所有目录存在
for directory in [DATA_DIR, RESULTS_DIR, SCRIPTS_DIR, TOOLS_DIR, DOCS_DIR,
                  RAW_DATA_DIR, PROCESSED_DATA_DIR, MARKET_DATA_DIR,
                  REPORTS_DIR, CHARTS_DIR, ANALYSIS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# 分析配置
ANALYSIS_CONFIG = {
    'timeframes': ['1m', '5m', '15m', '1h', '4h'],
    'technical_indicators': {
        'rsi_period': 14,
        'ma_periods': [5, 10, 20, 50],
        'bb_period': 20,
        'bb_std': 2,
        'volume_ma_period': 20
    },
    'strategy_rules': {
        'entry_rsi_range': (30, 50),
        'exit_rsi_range': (35, 55),
        'max_holding_minutes': 180,
        'min_holding_minutes': 15,
        'leverage': 10,
        'preferred_exit_slots': [1, 3]  # 15-29分钟和45-59分钟
    },
    'validation': {
        'lookback_periods': 30,
        'min_signal_strength': 0.67,
        'confidence_threshold': 0.75
    }
}

# 交易对配置
TRADING_PAIRS = [
    'XRP/USDT:USDT',
    'DOGE/USDT:USDT', 
    'ICP/USDT:USDT',
    'IOTA/USDT:USDT',
    'SOL/USDT:USDT',
    'SUI/USDT:USDT',
    'ALGO/USDT:USDT',
    'BNB/USDT:USDT',
    'ADA/USDT:USDT'
]

# 币安符号映射
SYMBOL_MAPPING = {
    'XRP/USDT:USDT': 'XRPUSDT',
    'DOGE/USDT:USDT': 'DOGEUSDT',
    'ICP/USDT:USDT': 'ICPUSDT',
    'IOTA/USDT:USDT': 'IOTAUSDT',
    'SOL/USDT:USDT': 'SOLUSDT',
    'SUI/USDT:USDT': 'SUIUSDT',
    'ALGO/USDT:USDT': 'ALGOUSDT',
    'BNB/USDT:USDT': 'BNBUSDT',
    'ADA/USDT:USDT': 'ADAUSDT'
}

print(f"📁 DipMaster Analysis Configuration Loaded")
print(f"   Base Directory: {BASE_DIR}")
print(f"   Data Directory: {DATA_DIR}")
print(f"   Results Directory: {RESULTS_DIR}")