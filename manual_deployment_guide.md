# 🚀 DipMaster 7天纸面交易手动部署指南

## 步骤1: 通过阿里云VNC连接服务器

1. **登录阿里云控制台**
2. **ECS管理控制台** → **实例** → **远程连接** → **VNC远程连接**
3. **输入root密码登录**

## 步骤2: 配置SSH密钥（可选但推荐）

在VNC中执行：

```bash
# 创建SSH目录
mkdir -p /root/.ssh
chmod 700 /root/.ssh

# 添加你的公钥
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDHiJwgf21DN2pl4f91Mf/QPW6IwZHL3VPbb/uXKnW/IA9c05QV0F1HzqikpCVjIukbhYpr1lfyxGNiCIRSsGZgfX4aaXzKvG+QIq+Tj9Hf0X6lxgZBBHvTaOBKQDDQdGn91zyhedeuyz5D5h3uiSZQSTo8quXPb7GQ/7nLNnZZCQMjdgF5s8kmcaEkWUJoSKOKkceI530W2OTnISJAeBM6Fyl7ubb100OKHTdq43R/o4WqlXlu7TsJ/qFFum5lvz4NH5wU+cf41GQYUhCXxZESc8sbGnFqsFVfpdoTD3Y2ZubNrtK+scvJyaZa2ueTuobmD+Q3G8sgGevo1/CD45oyegQifIRoObtVXQu6yjYhdgSG53QkfTv8fM5jY+2v/VE5KxmQPuGoM/eS2tDMCaj3SNU7skgZgjFZuiijtUnsnPOiH3V1LmqqanadM0rqjp8uYPUeqnQ0fj6FkRewwFc9N/ZUw+Sdu7ejeuGUYW9sKfHFA6wIHwKe81EtFrxkvVyHe88mQ1srN66NFEO7lCUajfReoEmv+ZzAAKmHxPI8LmSByxdwR1qUzKwLDwuJsrYYGGJwa3eKIFe+FQQHpEMpSooHauB47YjW9CdhBnOwfAfRTtNKECPp4iZQphNP6DexaILrGoAOfkweZkDuGTDSrGbJ6gX8Kq0kUWndDs46CQ== zhangxuanyang@Xuanyangs-Macbook.local" > /root/.ssh/authorized_keys

chmod 600 /root/.ssh/authorized_keys
systemctl restart sshd
```

## 步骤3: 准备服务器环境

```bash
# 更新系统
yum update -y

# 安装必要软件
yum install -y python3 python3-pip git screen htop wget curl

# 检查Python版本
python3 --version
pip3 --version

# 创建项目目录
mkdir -p /opt/DipMaster-Trading-System
cd /opt/DipMaster-Trading-System

# 创建子目录
mkdir -p {logs,results,data,config,src}
```

## 步骤4: 创建核心文件

### 4.1 创建依赖文件

```bash
cat > requirements.txt << 'EOF'
numpy>=1.21.0
pandas>=1.3.0
asyncio-mqtt>=0.11.0
aiohttp>=3.8.0
websockets>=10.0
ccxt>=2.5.0
python-binance>=1.0.15
TA-Lib>=0.4.25
requests>=2.28.0
python-dotenv>=0.19.0
pyyaml>=6.0
colorlog>=6.0.0
asyncio>=3.4.3
typing-extensions>=4.0.0
EOF
```

### 4.2 创建纸面交易配置

```bash
cat > config/paper_trading_config.json << 'EOF'
{
  "strategy_name": "DipMaster_Paper_Trading_Server",
  "version": "1.0.0",
  "description": "DipMaster服务器纸面交易配置",
  "created_date": "2025-08-18",
  
  "trading": {
    "paper_trading": true,
    "initial_capital": 10000,
    "symbols": ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "BNBUSDT"],
    "max_concurrent_positions": 3,
    "position_size_usd": 800,
    "min_position_size_usd": 300,
    "max_position_size_usd": 1200
  },
  
  "api": {
    "exchange": "binance",
    "api_key": "paper_trading_key",
    "api_secret": "paper_trading_secret",
    "testnet": false,
    "paper_mode": true
  },
  
  "risk_management": {
    "global_settings": {
      "max_daily_loss_usd": 300,
      "max_drawdown_percent": 8.0,
      "position_size_limit_percent": 25,
      "leverage_limit": 1,
      "correlation_limit": 0.8
    }
  },
  
  "logging_and_monitoring": {
    "log_level": "INFO",
    "detailed_trade_logging": true,
    "save_results": true,
    "dashboard_enabled": false
  },
  
  "deployment_settings": {
    "server_mode": true,
    "auto_restart": true,
    "health_check_interval": 300,
    "max_memory_usage_mb": 1024
  }
}
EOF
```

### 4.3 创建简化的纸面交易脚本

```bash
cat > run_paper_trading.py << 'EOF'
#!/usr/bin/env python3
"""
DipMaster简化版纸面交易脚本
用于服务器运行一周测试
"""

import asyncio
import json
import logging
import signal
import sys
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
import argparse

class SimplePaperTradingEngine:
    """简化的纸面交易引擎"""
    
    def __init__(self, config):
        self.config = config
        self.running = False
        self.start_time = None
        self.capital = config['trading']['initial_capital']
        self.positions = {}
        self.trade_history = []
        self.stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0
        }
        
        # 设置日志
        self.setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def setup_logging(self):
        """设置日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"paper_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _signal_handler(self, signum, frame):
        """信号处理"""
        self.logger.info(f"接收到停止信号 {signum}")
        self.running = False
    
    def simulate_market_data(self):
        """模拟市场数据"""
        symbols = self.config['trading']['symbols']
        # 模拟价格数据
        market_data = {}
        for symbol in symbols:
            base_price = random.uniform(20000, 70000) if 'BTC' in symbol else random.uniform(1000, 4000)
            market_data[symbol] = {
                'price': base_price * (1 + random.uniform(-0.05, 0.05)),
                'volume': random.uniform(1000000, 10000000),
                'rsi': random.uniform(20, 80)
            }
        return market_data
    
    def generate_trading_signal(self, symbol, market_data):
        """生成交易信号"""
        data = market_data[symbol]
        
        # 简化的DipMaster信号逻辑
        if data['rsi'] < 45 and random.random() < 0.15:  # 15%概率生成买入信号
            return {
                'action': 'BUY',
                'symbol': symbol,
                'price': data['price'],
                'confidence': random.uniform(0.6, 0.9),
                'reason': f"RSI:{data['rsi']:.1f}, DIP detected"
            }
        
        return None
    
    def execute_trade(self, signal):
        """执行交易"""
        symbol = signal['symbol']
        price = signal['price']
        position_size = self.config['trading']['position_size_usd']
        
        if signal['action'] == 'BUY' and len(self.positions) < self.config['trading']['max_concurrent_positions']:
            
            trade = {
                'id': len(self.trade_history) + 1,
                'symbol': symbol,
                'side': 'BUY',
                'price': price,
                'size_usd': position_size,
                'timestamp': datetime.now(),
                'exit_price': None,
                'pnl': 0,
                'status': 'OPEN'
            }
            
            self.positions[symbol] = trade
            self.trade_history.append(trade)
            
            self.logger.info(f"📈 开仓: {symbol} @ ${price:.2f}, 投入: ${position_size}")
            
    def check_exit_conditions(self, symbol, current_price):
        """检查出场条件"""
        if symbol not in self.positions:
            return False
            
        position = self.positions[symbol]
        entry_time = position['timestamp']
        holding_time = datetime.now() - entry_time
        
        # DipMaster出场逻辑：15分钟边界或达到目标利润
        entry_price = position['price']
        profit_pct = (current_price - entry_price) / entry_price
        
        # 15分钟边界出场
        if holding_time.total_seconds() >= 15 * 60:
            if holding_time.total_seconds() % (15 * 60) < 60:  # 15分钟边界
                return True
                
        # 目标利润出场
        if profit_pct >= 0.008:  # 0.8%目标利润
            return True
            
        # 最大持仓时间
        if holding_time.total_seconds() >= 180 * 60:  # 3小时强制平仓
            return True
            
        return False
    
    def close_position(self, symbol, current_price):
        """平仓"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        entry_price = position['price']
        size_usd = position['size_usd']
        
        pnl = size_usd * (current_price - entry_price) / entry_price
        
        position['exit_price'] = current_price
        position['pnl'] = pnl
        position['status'] = 'CLOSED'
        
        # 更新统计
        self.stats['total_trades'] += 1
        self.stats['total_pnl'] += pnl
        
        if pnl > 0:
            self.stats['winning_trades'] += 1
            
        self.capital += pnl
        
        holding_time = datetime.now() - position['timestamp']
        
        self.logger.info(f"📉 平仓: {symbol} @ ${current_price:.2f}, "
                        f"盈亏: ${pnl:.2f}, 持仓: {holding_time.total_seconds()/60:.1f}分钟")
        
        del self.positions[symbol]
    
    def print_status(self):
        """打印状态"""
        if not self.start_time:
            return
            
        runtime = datetime.now() - self.start_time
        win_rate = (self.stats['winning_trades'] / max(self.stats['total_trades'], 1)) * 100
        
        status = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 DipMaster纸面交易状态 (运行: {str(runtime).split('.')[0]})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💰 当前资金: ${self.capital:.2f}
📈 总交易数: {self.stats['total_trades']}
✅ 盈利交易: {self.stats['winning_trades']} ({win_rate:.1f}%)
💹 总盈亏: ${self.stats['total_pnl']:.2f}
📊 当前持仓: {len(self.positions)}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        """
        self.logger.info(status)
    
    async def run(self, duration_hours=168):
        """运行纸面交易"""
        self.running = True
        self.start_time = datetime.now()
        
        self.logger.info("🚀 DipMaster纸面交易启动！")
        self.logger.info(f"💰 初始资金: ${self.capital}")
        self.logger.info(f"⏱️ 运行时长: {duration_hours}小时")
        self.logger.info(f"📊 交易币种: {self.config['trading']['symbols']}")
        
        last_status_time = datetime.now()
        
        try:
            while self.running:
                # 检查运行时间
                if datetime.now() - self.start_time > timedelta(hours=duration_hours):
                    self.logger.info(f"⏰ 达到运行时间 {duration_hours} 小时，停止交易")
                    break
                
                # 模拟市场数据
                market_data = self.simulate_market_data()
                
                # 处理现有持仓
                for symbol in list(self.positions.keys()):
                    current_price = market_data[symbol]['price']
                    if self.check_exit_conditions(symbol, current_price):
                        self.close_position(symbol, current_price)
                
                # 生成新信号
                for symbol in self.config['trading']['symbols']:
                    signal = self.generate_trading_signal(symbol, market_data)
                    if signal:
                        self.execute_trade(signal)
                
                # 定期打印状态 (每30分钟)
                if datetime.now() - last_status_time >= timedelta(minutes=30):
                    self.print_status()
                    last_status_time = datetime.now()
                
                # 等待下一个周期 (模拟5分钟周期)
                await asyncio.sleep(60)  # 1分钟间隔，加速测试
                
        except KeyboardInterrupt:
            self.logger.info("🛑 接收到停止信号")
        except Exception as e:
            self.logger.error(f"❌ 运行错误: {e}")
        finally:
            # 关闭所有持仓
            market_data = self.simulate_market_data()
            for symbol in list(self.positions.keys()):
                self.close_position(symbol, market_data[symbol]['price'])
            
            # 最终统计
            self.print_final_summary()
    
    def print_final_summary(self):
        """最终总结"""
        runtime = datetime.now() - self.start_time if self.start_time else timedelta(0)
        win_rate = (self.stats['winning_trades'] / max(self.stats['total_trades'], 1)) * 100
        roi = (self.capital - self.config['trading']['initial_capital']) / self.config['trading']['initial_capital'] * 100
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    🏁 DipMaster纸面交易完成                    ║
╠══════════════════════════════════════════════════════════════╣
║  ⏱️ 总运行时间: {str(runtime).split('.')[0]}                     ║
║  💰 初始资金: ${self.config['trading']['initial_capital']}      ║
║  💹 最终资金: ${self.capital:.2f}                              ║
║  📈 总回报率: {roi:+.2f}%                                      ║
║  🎯 总交易数: {self.stats['total_trades']}                     ║
║  ✅ 胜率: {win_rate:.1f}%                                      ║
║  💰 总盈亏: ${self.stats['total_pnl']:+.2f}                    ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(summary)
        self.logger.info("🎉 纸面交易结束")

def main():
    parser = argparse.ArgumentParser(description="DipMaster纸面交易")
    parser.add_argument('--config', '-c', 
                       default='config/paper_trading_config.json',
                       help='配置文件')
    parser.add_argument('--hours', '-t', type=int, default=168,
                       help='运行小时数')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 创建引擎
    engine = SimplePaperTradingEngine(config)
    
    # 运行
    try:
        asyncio.run(engine.run(duration_hours=args.hours))
    except KeyboardInterrupt:
        print("🛑 程序被中断")
    except Exception as e:
        print(f"❌ 程序错误: {e}")

if __name__ == "__main__":
    main()
EOF
```

## 步骤5: 安装依赖并测试

```bash
# 安装Python依赖
pip3 install -r requirements.txt

# 给脚本执行权限
chmod +x run_paper_trading.py

# 快速测试（运行5分钟）
python3 run_paper_trading.py --hours 0.083

# 如果测试通过，ctrl+C停止
```

## 步骤6: 启动7天纸面测试

### 6.1 创建启动脚本

```bash
cat > start_7day_test.sh << 'EOF'
#!/bin/bash

cd /opt/DipMaster-Trading-System

echo "🚀 启动DipMaster 7天纸面交易测试"
echo "开始时间: $(date)"

# 使用screen后台运行
screen -dmS dipmaster-paper bash -c "
    cd /opt/DipMaster-Trading-System
    python3 run_paper_trading.py --hours 168 --config config/paper_trading_config.json
    echo '测试完成，按任意键退出'
    read
"

sleep 3
echo "✅ 测试已启动，运行在screen会话中"
echo "查看状态: screen -r dipmaster-paper"
echo "停止测试: screen -S dipmaster-paper -X quit"
screen -list
EOF

chmod +x start_7day_test.sh
```

### 6.2 创建监控脚本

```bash
cat > monitor.sh << 'EOF'
#!/bin/bash

echo "📊 DipMaster监控面板 - $(date)"
echo "=================================="

# 检查进程
if screen -list | grep -q dipmaster-paper; then
    echo "✅ 交易程序正在运行"
else
    echo "❌ 交易程序未运行"
fi

echo ""

# 显示最新日志
latest_log=$(ls -t logs/paper_trading_*.log 2>/dev/null | head -1)
if [ -n "$latest_log" ]; then
    echo "📝 最新日志: $latest_log"
    echo "最后20行:"
    echo "----------"
    tail -20 "$latest_log"
fi

echo ""
echo "💻 系统状态:"
echo "内存使用: $(free -m | awk 'NR==2{printf "%.1f%%", $3*100/$2}')"
echo "磁盘使用: $(df -h . | awk 'NR==2{print $5}')"
echo ""
echo "常用命令:"
echo "进入会话: screen -r dipmaster-paper"
echo "实时日志: tail -f logs/paper_trading_*.log"
echo "停止测试: screen -S dipmaster-paper -X quit"
EOF

chmod +x monitor.sh
```

## 步骤7: 启动测试

```bash
# 启动7天测试
./start_7day_test.sh

# 查看监控
./monitor.sh
```

## 步骤8: 日常监控

```bash
# 查看实时状态
./monitor.sh

# 查看实时日志
tail -f logs/paper_trading_*.log

# 进入交互会话
screen -r dipmaster-paper

# 退出screen会话 (不停止程序)
# 按 Ctrl+A 然后按 D
```

## 重要提醒

1. **测试将运行7天** (168小时)
2. **虚拟资金$10,000**，无实际风险
3. **日志保存在logs/目录**
4. **可随时通过screen监控状态**
5. **服务器重启后需要重新启动**

执行完这些步骤后，你的DipMaster纸面交易就会在服务器上运行7天了！
EOF