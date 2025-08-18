#!/usr/bin/env python3
"""
DipMaster纸面交易运行脚本
用于启动长期纸面交易测试，适合在服务器上运行一周
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
import argparse
import os

# 设置日志
def setup_logging(log_level="INFO"):
    """设置详细日志"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 创建日志文件名
    log_filename = f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = log_dir / log_filename
    
    # 配置日志格式
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 日志文件: {log_path}")
    return logger

class PaperTradingRunner:
    """纸面交易运行器"""
    
    def __init__(self, config_path="config/paper_trading_config.json"):
        self.config_path = config_path
        self.engine = None
        self.running = False
        self.start_time = None
        self.stats = {
            'start_time': None,
            'total_signals': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0
        }
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _signal_handler(self, signum, frame):
        """处理停止信号"""
        self.logger.info(f"📧 接收到信号 {signum}, 准备优雅停机...")
        self.running = False
    
    def load_config(self):
        """加载配置文件"""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 确保纸面交易模式
        config['trading']['paper_trading'] = True
        
        self.logger.info(f"✅ 配置加载成功: {config_file}")
        return config
    
    async def initialize_engine(self):
        """初始化交易引擎"""
        try:
            # 加载配置
            config = self.load_config()
            
            # 导入交易引擎
            from src.core.trading_engine import DipMasterTradingEngine
            
            # 创建引擎
            self.engine = DipMasterTradingEngine(config)
            
            self.logger.info("✅ 交易引擎初始化成功")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 交易引擎初始化失败: {e}")
            return False
    
    def print_startup_banner(self):
        """打印启动横幅"""
        banner = """
╔══════════════════════════════════════════════════════════════╗
║                    🎯 DipMaster 纸面交易系统                    ║
║                                                              ║
║  📊 模式: 纸面交易 (无实际资金风险)                              ║
║  💰 初始资金: $10,000                                         ║
║  🎯 目标: 验证策略有效性                                        ║
║  ⏱️  建议运行时间: 1周                                          ║
║                                                              ║
║  🛑 停止方式: Ctrl+C 优雅停机                                  ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)
    
    def print_current_stats(self):
        """打印当前统计"""
        if not self.start_time:
            return
        
        runtime = datetime.now() - self.start_time
        win_rate = (self.stats['winning_trades'] / max(self.stats['total_trades'], 1)) * 100
        
        stats_info = f"""
📈 运行统计 (运行时长: {str(runtime).split('.')[0]})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💹 总信号数: {self.stats['total_signals']}
🎯 总交易数: {self.stats['total_trades']}
✅ 盈利交易: {self.stats['winning_trades']} ({win_rate:.1f}%)
❌ 亏损交易: {self.stats['losing_trades']}
💰 总盈亏: ${self.stats['total_pnl']:.2f}
📉 最大回撤: {self.stats['max_drawdown']:.2f}%
📊 当前回撤: {self.stats['current_drawdown']:.2f}%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        self.logger.info(stats_info)
    
    async def run(self, max_duration_hours=168):  # 默认7天 = 168小时
        """运行纸面交易"""
        
        self.print_startup_banner()
        
        # 初始化引擎
        if not await self.initialize_engine():
            return False
        
        self.running = True
        self.start_time = datetime.now()
        self.stats['start_time'] = self.start_time
        
        try:
            self.logger.info(f"🚀 开始纸面交易 (最大运行时间: {max_duration_hours}小时)")
            
            # 启动引擎
            await self.engine.start()
            
            # 主运行循环
            last_stats_time = datetime.now()
            stats_interval = timedelta(minutes=30)  # 每30分钟打印统计
            
            while self.running:
                # 检查最大运行时间
                if datetime.now() - self.start_time > timedelta(hours=max_duration_hours):
                    self.logger.info(f"⏰ 达到最大运行时间 {max_duration_hours} 小时，准备停止")
                    break
                
                # 检查引擎状态
                if not self.engine.running:
                    self.logger.warning("⚠️ 引擎已停止，尝试重新启动...")
                    try:
                        await self.engine.start()
                        await asyncio.sleep(5)
                    except Exception as e:
                        self.logger.error(f"❌ 重启引擎失败: {e}")
                        break
                
                # 定期打印统计
                if datetime.now() - last_stats_time >= stats_interval:
                    self.print_current_stats()
                    last_stats_time = datetime.now()
                
                # 更新统计 (这里应该从引擎获取真实数据)
                await self.update_stats()
                
                await asyncio.sleep(10)  # 每10秒检查一次
            
        except KeyboardInterrupt:
            self.logger.info("🛑 接收到停止信号...")
        except Exception as e:
            self.logger.error(f"❌ 运行过程中发生错误: {e}")
        finally:
            # 优雅停机
            if self.engine:
                try:
                    await self.engine.stop()
                    self.logger.info("✅ 引擎已安全关闭")
                except Exception as e:
                    self.logger.error(f"⚠️ 引擎关闭时出现问题: {e}")
            
            # 打印最终统计
            self.print_final_summary()
    
    async def update_stats(self):
        """更新统计数据 (模拟)"""
        # 这里应该从实际引擎获取数据
        # 目前使用模拟数据演示
        pass
    
    def print_final_summary(self):
        """打印最终总结"""
        if not self.start_time:
            return
        
        runtime = datetime.now() - self.start_time
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    📊 纸面交易完成总结                          ║
╠══════════════════════════════════════════════════════════════╣
║  ⏱️  总运行时间: {str(runtime).split('.')[0]}                    ║
║  💹 总信号数: {self.stats['total_signals']}                     ║
║  🎯 总交易数: {self.stats['total_trades']}                      ║
║  ✅ 盈利交易: {self.stats['winning_trades']}                    ║
║  ❌ 亏损交易: {self.stats['losing_trades']}                     ║
║  💰 总盈亏: ${self.stats['total_pnl']:.2f}                      ║
║  📉 最大回撤: {self.stats['max_drawdown']:.2f}%                 ║
╠══════════════════════════════════════════════════════════════╣
║  📝 日志位置: logs/paper_trading_*.log                        ║
║  📈 结果保存: results/paper_trading_*.json                    ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(summary)
        self.logger.info("🎉 纸面交易会话结束")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DipMaster纸面交易系统")
    parser.add_argument('--config', '-c', 
                       default='config/paper_trading_config.json',
                       help='配置文件路径')
    parser.add_argument('--hours', '-t', type=int, default=168,
                       help='最大运行小时数 (默认: 168小时 = 1周)')
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='日志级别')
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level)
    
    try:
        # 创建运行器
        runner = PaperTradingRunner(args.config)
        
        # 运行纸面交易
        asyncio.run(runner.run(max_duration_hours=args.hours))
        
    except KeyboardInterrupt:
        logger.info("🛑 程序被用户中断")
    except Exception as e:
        logger.error(f"❌ 程序运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()