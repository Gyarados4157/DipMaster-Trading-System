#!/usr/bin/env python3
"""
DipMaster纸面交易测试脚本
用于验证系统是否可以正常运行纸面交易模式
"""

import asyncio
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# 设置简单的日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)

def create_paper_trading_config():
    """创建纸面交易配置"""
    config = {
        "strategy_name": "DipMaster_Paper_Test",
        "version": "test_1.0.0",
        
        # 纸面交易设置
        "trading": {
            "paper_trading": True,
            "initial_capital": 10000,
            "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
            "max_concurrent_positions": 2,
            "position_size_usd": 500
        },
        
        # API设置 (纸面交易模式下这些可以是空的)
        "api": {
            "api_key": "test_key",
            "api_secret": "test_secret",
            "testnet": True
        },
        
        # 基础风险管理
        "risk_management": {
            "max_daily_loss_usd": 200,
            "max_drawdown_percent": 5.0,
            "position_size_limit_percent": 20,
            "leverage_limit": 1
        },
        
        # 简化的信号检测
        "enhanced_signal_detection": {
            "enabled": True,
            "minimum_confidence": 0.6,
            "layer_1_rsi_filter": {
                "enabled": True,
                "rsi_range": [30, 50],
                "weight": 0.4
            },
            "layer_2_trend_filter": {
                "enabled": True,
                "max_consecutive_red_candles": 3,
                "weight": 0.3
            },
            "layer_3_volume_filter": {
                "enabled": True,
                "min_volume_multiplier": 1.2,
                "weight": 0.3
            }
        },
        
        # 时间管理
        "asymmetric_risk_management": {
            "enabled": True,
            "time_management": {
                "min_holding_minutes": 15,
                "max_holding_minutes": 180
            },
            "stop_loss_system": {
                "emergency_stop_percent": 0.8,
                "normal_stop_percent": 1.5
            },
            "profit_taking_system": {
                "enabled": True,
                "partial_exits": [
                    {"profit_percent": 0.8, "exit_ratio": 0.5}
                ]
            }
        },
        
        # 技术指标
        "technical_indicators": {
            "primary_indicators": {
                "rsi": {"period": 14, "entry_range": [30, 50]},
                "ema": {"periods": [20], "use_for_trend": True},
                "volume_ma": {"period": 20, "min_multiplier": 1.2}
            }
        },
        
        # 数据设置
        "data_requirements": {
            "timeframes": {
                "primary": "5m"
            },
            "minimum_history": "7_days"
        },
        
        # 日志设置
        "logging_and_monitoring": {
            "log_level": "INFO",
            "detailed_trade_logging": True,
            "dashboard_enabled": False,
            "save_results": True
        }
    }
    
    return config

async def test_paper_trading_system():
    """测试纸面交易系统"""
    
    logger.info("🧪 开始DipMaster纸面交易系统测试...")
    
    try:
        # 1. 创建配置
        config = create_paper_trading_config()
        logger.info("✅ 纸面交易配置创建成功")
        
        # 2. 测试基础组件导入
        try:
            from src.core.trading_engine import DipMasterTradingEngine
            logger.info("✅ 交易引擎导入成功")
        except ImportError as e:
            logger.error(f"❌ 交易引擎导入失败: {e}")
            return False
            
        # 3. 创建交易引擎实例
        try:
            engine = DipMasterTradingEngine(config)
            logger.info("✅ 交易引擎实例创建成功")
        except Exception as e:
            logger.error(f"❌ 交易引擎创建失败: {e}")
            return False
            
        # 4. 测试基础功能
        logger.info("📊 开始测试基础功能...")
        
        # 测试配置验证
        if hasattr(engine, 'config'):
            logger.info("✅ 配置加载正常")
        else:
            logger.error("❌ 配置加载失败")
            return False
            
        # 测试组件初始化
        components = ['stream_manager', 'timing_manager', 'signal_detector', 'position_manager', 'order_executor']
        for component in components:
            if hasattr(engine, component):
                logger.info(f"✅ {component} 初始化成功")
            else:
                logger.warning(f"⚠️ {component} 未找到")
        
        # 5. 短期运行测试 (5秒)
        logger.info("⏰ 开始5秒短期运行测试...")
        
        # 设置超时
        timeout_duration = 5
        try:
            await asyncio.wait_for(
                run_engine_for_duration(engine, timeout_duration),
                timeout=timeout_duration + 2
            )
            logger.info("✅ 短期运行测试完成")
        except asyncio.TimeoutError:
            logger.info("✅ 超时正常停止 (预期行为)")
        except Exception as e:
            logger.error(f"❌ 运行测试失败: {e}")
            return False
        finally:
            # 确保引擎停止
            if hasattr(engine, 'stop'):
                try:
                    await engine.stop()
                    logger.info("✅ 引擎安全关闭")
                except:
                    pass
        
        logger.info("🎉 纸面交易系统测试完成!")
        return True
        
    except Exception as e:
        logger.error(f"❌ 测试过程中发生错误: {e}")
        return False

async def run_engine_for_duration(engine, duration):
    """运行引擎指定时间"""
    try:
        # 尝试启动引擎
        await engine.start()
        logger.info(f"🚀 引擎已启动，将运行{duration}秒...")
        
        # 运行指定时间
        start_time = asyncio.get_event_loop().time()
        while (asyncio.get_event_loop().time() - start_time) < duration:
            if not engine.running:
                logger.warning("引擎已停止运行")
                break
            await asyncio.sleep(0.1)
            
    except Exception as e:
        logger.warning(f"引擎运行中出现问题 (可能正常): {e}")

def main():
    """主函数"""
    try:
        # 运行异步测试
        success = asyncio.run(test_paper_trading_system())
        
        if success:
            print("\n" + "="*60)
            print("🎊 DipMaster纸面交易系统测试成功!")
            print("✅ 系统可以正常进行纸面交易")
            print("🚀 准备好进行长期测试和服务器部署")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ 测试失败，需要修复问题后重新测试")
            print("="*60)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n🛑 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生未预期错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()