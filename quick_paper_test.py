#!/usr/bin/env python3
"""
DipMaster快速纸面交易测试 - 运行2分钟验证系统
"""

import asyncio
import sys
from run_paper_trading import PaperTradingRunner
from test_paper_trading import setup_logging

async def quick_test():
    """2分钟快速测试"""
    
    print("🧪 开始2分钟快速纸面交易测试...")
    
    # 设置日志
    logger = setup_logging("INFO")
    
    try:
        # 创建运行器
        runner = PaperTradingRunner("config/paper_trading_config.json")
        
        # 运行2分钟
        await runner.run(max_duration_hours=0.033)  # 2分钟 = 0.033小时
        
        print("✅ 快速测试完成！系统运行正常")
        return True
        
    except Exception as e:
        print(f"❌ 快速测试失败: {e}")
        return False

def main():
    try:
        success = asyncio.run(quick_test())
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 测试被中断")

if __name__ == "__main__":
    main()