#!/usr/bin/env python3
"""
DipMaster Trading System - Main Entry Point
主程序入口点 - 实盘交易系统

Author: DipMaster Trading Team
Date: 2025-08-13
Version: 3.0.0
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any
import argparse

from src.core.trading_engine import DipMasterTradingEngine
from src.core.dipmaster_live import DipMasterLiveStrategy
from src.dashboard.monitor_dashboard import DashboardServer


def setup_logging(log_level: str = "INFO") -> None:
    """设置日志配置"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"dipmaster_{datetime.now().strftime('%Y%m%d')}.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 DipMaster Trading System v3.0.0 启动")
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置文件的有效性"""
    required_keys = ['api', 'trading', 'risk_management']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置文件缺少必要部分: {key}")
    
    # 验证API配置
    api_config = config['api']
    if 'api_key' not in api_config or 'api_secret' not in api_config:
        raise ValueError("缺少API密钥配置")
    
    # 验证交易配置
    trading_config = config['trading']
    if 'symbols' not in trading_config:
        raise ValueError("缺少交易对配置")
    
    return True


async def run_trading_engine(config: Dict[str, Any], dashboard: bool = True) -> None:
    """运行交易引擎"""
    logger = logging.getLogger(__name__)
    
    try:
        # 创建交易引擎
        engine = DipMasterTradingEngine(config)
        
        # 启动仪表板（如果需要）
        dashboard_server = None
        if dashboard:
            dashboard_server = DashboardServer(port=config.get('dashboard', {}).get('port', 8080))
            await dashboard_server.start()
            logger.info("📊 监控仪表板已启动")
        
        # 启动交易引擎
        logger.info("⚡ 启动DipMaster交易引擎...")
        await engine.start()
        
        # 保持运行
        try:
            while engine.running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("🛑 接收到停止信号，正在关闭系统...")
        
        # 优雅关闭
        await engine.stop()
        if dashboard_server:
            await dashboard_server.stop()
            
        logger.info("✅ DipMaster交易系统已安全关闭")
        
    except Exception as e:
        logger.error(f"❌ 交易引擎运行错误: {e}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DipMaster Trading System v3.0.0")
    parser.add_argument('--config', '-c', default='config/dipmaster_v3_optimized.json',
                      help='配置文件路径 (默认: config/dipmaster_v3_optimized.json)')
    parser.add_argument('--log-level', default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='日志级别 (默认: INFO)')
    parser.add_argument('--paper', action='store_true',
                      help='纸面交易模式 (用于测试)')
    parser.add_argument('--no-dashboard', action='store_true',
                      help='禁用监控仪表板')
    
    args = parser.parse_args()
    
    try:
        # 设置日志
        from datetime import datetime
        logger = setup_logging(args.log_level)
        
        # 加载并验证配置
        logger.info(f"📋 加载配置文件: {args.config}")
        config = load_config(args.config)
        validate_config(config)
        
        # 纸面交易模式
        if args.paper:
            config['trading']['paper_trading'] = True
            logger.info("📄 启用纸面交易模式")
        
        # 显示系统信息
        logger.info("=" * 60)
        logger.info("🎯 DipMaster Trading System v3.0.0")
        logger.info("=" * 60)
        logger.info(f"💼 交易模式: {'纸面交易' if config.get('trading', {}).get('paper_trading', False) else '实盘交易'}")
        logger.info(f"📊 监控面板: {'禁用' if args.no_dashboard else '启用'}")
        logger.info(f"🔧 配置文件: {args.config}")
        logger.info(f"📈 交易对: {', '.join(config.get('trading', {}).get('symbols', []))}")
        logger.info("=" * 60)
        
        # 启动交易系统
        asyncio.run(run_trading_engine(config, dashboard=not args.no_dashboard))
        
    except Exception as e:
        logger.error(f"❌ 系统启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()