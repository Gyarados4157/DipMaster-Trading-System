#!/usr/bin/env python3
"""
Expand Symbol Pool - 扩展币种池数据下载
================================

快速下载脚本：为Ultra优化系统扩展币种数据
避开BTC和ETH，专注优质中小盘币种

Author: DipMaster Ultra Team  
Date: 2025-08-15
"""

import asyncio
import logging
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from src.tools.ultra_symbol_data_manager import UltraSymbolDataManager

async def main():
    """主函数 - 扩展币种池"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Symbol Pool Expansion")
    
    # 创建数据管理器
    manager = UltraSymbolDataManager("data/market_data")
    
    # 初始化币种池
    manager.initialize_symbol_pool()
    
    # 下载新币种数据
    logger.info("📥 Starting batch download of new symbols...")
    success_count, fail_count = manager.download_all_symbols()
    
    # 生成质量报告
    quality_report = manager.get_quality_report()
    
    # 输出结果
    logger.info("🎉 Symbol Pool Expansion Complete!")
    logger.info("="*50)
    logger.info(f"✅ Successfully downloaded: {success_count}")
    logger.info(f"❌ Failed downloads: {fail_count}")
    logger.info(f"📊 Average quality score: {quality_report.get('平均质量评分', 0):.1f}")
    
    # 推荐高质量币种
    recommended = manager.get_recommended_symbols(min_quality=75)
    logger.info(f"🎯 Recommended high-quality symbols: {len(recommended)}")
    for symbol in recommended:
        info = manager.symbol_info.get(symbol, {})
        quality = manager.data_quality.get(symbol, {})
        logger.info(f"  • {symbol} [Tier {getattr(info, 'tier', 'N/A')}] - "
                   f"Quality: {getattr(quality, 'quality_score', 0):.1f}")
    
    logger.info("✅ Ready for Ultra Optimization Validation!")

if __name__ == "__main__":
    asyncio.run(main())