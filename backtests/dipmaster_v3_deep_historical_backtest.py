#!/usr/bin/env python3
"""
DipMaster V3 深度历史回测
2年期完整回测，重点验证DIP策略复刻和大额亏损分析

目标：
1. 验证策略是否能完整复刻DipMaster AI的逢跌买入操作
2. 进行2年期深度回测
3. 重点分析大额亏损情况和风险控制
4. 生成详细的性能和风险报告
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# 添加src路径
sys.path.append('src')
sys.path.append('src/core')

# 导入V3组件
try:
    from src.core.comprehensive_backtest_v3 import ComprehensiveBacktestV3, BacktestConfig, BacktestMetrics
    from src.core.enhanced_signal_detector import EnhancedSignalDetector
    from src.core.asymmetric_risk_manager import AsymmetricRiskManager  
    from src.core.volatility_adaptive_sizing import VolatilityAdaptiveSizing
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'dipmaster_v3_deep_backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class DipMasterDeepBacktest:
    """DipMaster V3 深度历史回测器"""
    
    def __init__(self):
        # 指定的9个币对
        self.target_symbols = [
            'XRPUSDT', 'DOGEUSDT', 'ICPUSDT', 'IOTAUSDT', 
            'SOLUSDT', 'SUIUSDT', 'ALGOUSDT', 'BNBUSDT', 'ADAUSDT'
        ]
        
        # 数据映射（优先使用最长时间数据）
        self.data_files = {
            'XRPUSDT': 'XRPUSDT_5m_2years.csv',       # 2年数据 (新)
            'DOGEUSDT': 'DOGEUSDT_5m_2years.csv',     # 2年数据 (新)
            'ICPUSDT': 'ICPUSDT_5m_2years.csv',       # 2年数据
            'IOTAUSDT': 'IOTAUSDT_5m_2years.csv',     # 2年数据 (新)
            'SOLUSDT': 'SOLUSDT_5m_2years.csv',       # 2年数据 (新)
            'ADAUSDT': 'ADAUSDT_5m_2years.csv',       # 2年数据 (新)
            'SUIUSDT': 'SUIUSDT_5m_2years.csv',       # 2年数据 (新)
            'ALGOUSDT': 'ALGOUSDT_5m_2years.csv',     # 2年数据 (新)
            'BNBUSDT': 'BNBUSDT_5m_2years.csv'        # 2年数据 (新)
        }
        
        self.data_path = Path("data/market_data")
        self.results_path = Path("results/deep_backtest")
        self.results_path.mkdir(exist_ok=True)
        
        # 深度分析配置
        self.analysis_config = {
            'max_drawdown_alert': 0.05,      # 5%回撤警报
            'large_loss_threshold': 100,     # 单笔大额亏损阈值（USD）
            'consecutive_loss_limit': 5,     # 连续亏损笔数限制
            'daily_loss_limit': 300,         # 日亏损限制（USD）
            'monthly_loss_limit': 1000       # 月亏损限制（USD）
        }
        
    def load_and_prepare_data(self) -> Dict[str, pd.DataFrame]:
        """加载和准备市场数据"""
        logger.info("🔄 开始加载市场数据...")
        
        market_data = {}
        
        for symbol in self.target_symbols:
            if symbol in self.data_files:
                file_path = self.data_path / self.data_files[symbol]
                
                if file_path.exists():
                    logger.info(f"📊 加载 {symbol} 数据: {file_path}")
                    
                    try:
                        df = pd.read_csv(file_path)
                        
                        # 数据标准化处理
                        if 'datetime' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['datetime'])
                        else:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        # 确保必要的列存在
                        required_cols = ['open', 'high', 'low', 'close', 'volume']
                        if all(col in df.columns for col in required_cols):
                            df = df[['timestamp'] + required_cols].copy()
                            df.set_index('timestamp', inplace=True)
                            df.sort_index(inplace=True)
                            
                            # 数据质量检查
                            df = self._clean_data(df)
                            
                            if len(df) > 1000:  # 至少1000条数据
                                market_data[symbol] = df
                                logger.info(f"✅ {symbol}: {len(df)}条数据, 时间范围: {df.index[0]} 到 {df.index[-1]}")
                            else:
                                logger.warning(f"⚠️ {symbol}: 数据不足，跳过")
                        else:
                            logger.error(f"❌ {symbol}: 缺少必要列")
                            
                    except Exception as e:
                        logger.error(f"❌ 加载{symbol}数据失败: {e}")
                else:
                    logger.warning(f"⚠️ 文件不存在: {file_path}")
            else:
                logger.warning(f"⚠️ 未配置{symbol}的数据文件")
        
        logger.info(f"✅ 成功加载 {len(market_data)} 个币种数据")
        return market_data
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理数据"""
        # 移除空值
        df = df.dropna()
        
        # 移除价格为0或负数的数据
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            df = df[df[col] > 0]
        
        # 移除异常价格（价格跳变超过50%）
        for col in price_cols:
            price_change = df[col].pct_change().abs()
            df = df[price_change < 0.5]
        
        # 确保OHLC逻辑正确
        df = df[(df['high'] >= df['low']) & 
                (df['high'] >= df['open']) & 
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) & 
                (df['low'] <= df['close'])]
        
        return df
    
    def analyze_dip_replication(self, backtest_results: List, market_data: Dict) -> Dict:
        """分析DIP策略复刻效果"""
        logger.info("🔍 分析DIP策略复刻效果...")
        
        dip_analysis = {
            'total_entries': 0,
            'dip_entries': 0,
            'dip_entry_rate': 0.0,
            'price_below_ma20_rate': 0.0,
            'rsi_in_range_rate': 0.0,
            'volume_surge_rate': 0.0,
            'boundary_exit_rate': 0.0,
            'avg_holding_minutes': 0.0,
            'dip_characteristics': {}
        }
        
        if not backtest_results:
            return dip_analysis
        
        # 分析每笔交易的DIP特征
        dip_entries = 0
        ma20_below_count = 0
        rsi_range_count = 0
        volume_surge_count = 0
        boundary_exits = 0
        total_holding_time = 0
        
        for trade in backtest_results:
            dip_analysis['total_entries'] += 1
            
            # 获取入场时的市场数据
            symbol = trade.symbol
            entry_time = trade.entry_time
            
            if symbol in market_data:
                df = market_data[symbol]
                
                # 找到入场时间点的数据
                entry_data = df[df.index <= entry_time]
                if len(entry_data) >= 20:
                    current_data = entry_data.iloc[-1]
                    ma20 = entry_data['close'].rolling(20).mean().iloc[-1]
                    
                    # 检查DIP特征
                    # 1. 价格低于MA20
                    if current_data['close'] < ma20:
                        ma20_below_count += 1
                    
                    # 2. 检查RSI（简化计算）
                    if len(entry_data) >= 14:
                        rsi = self._calculate_simple_rsi(entry_data['close'], 14)
                        if 30 <= rsi <= 50:  # DipMaster的RSI范围
                            rsi_range_count += 1
                    
                    # 3. 检查成交量放大
                    if len(entry_data) >= 10:
                        vol_ma = entry_data['volume'].rolling(10).mean().iloc[-2]
                        current_vol = current_data['volume']
                        if current_vol > vol_ma * 1.5:  # 1.5倍成交量
                            volume_surge_count += 1
                    
                    # 4. 检查是否为逢跌买入
                    if current_data['close'] < current_data['open']:
                        dip_entries += 1
            
            # 检查出场特征
            if hasattr(trade, 'exit_reason'):
                if 'boundary' in trade.exit_reason.lower():
                    boundary_exits += 1
            
            # 累计持仓时间
            if hasattr(trade, 'holding_minutes'):
                total_holding_time += trade.holding_minutes
        
        # 计算比率
        total = dip_analysis['total_entries']
        if total > 0:
            dip_analysis['dip_entries'] = dip_entries
            dip_analysis['dip_entry_rate'] = dip_entries / total * 100
            dip_analysis['price_below_ma20_rate'] = ma20_below_count / total * 100
            dip_analysis['rsi_in_range_rate'] = rsi_range_count / total * 100
            dip_analysis['volume_surge_rate'] = volume_surge_count / total * 100
            dip_analysis['boundary_exit_rate'] = boundary_exits / total * 100
            dip_analysis['avg_holding_minutes'] = total_holding_time / total
        
        logger.info(f"✅ DIP策略分析完成:")
        logger.info(f"   逢跌买入率: {dip_analysis['dip_entry_rate']:.1f}%")
        logger.info(f"   MA20下方率: {dip_analysis['price_below_ma20_rate']:.1f}%")
        logger.info(f"   边界出场率: {dip_analysis['boundary_exit_rate']:.1f}%")
        
        return dip_analysis
    
    def analyze_large_losses(self, backtest_results: List) -> Dict:
        """深度分析大额亏损情况"""
        logger.info("🔍 深度分析大额亏损情况...")
        
        loss_analysis = {
            'total_trades': len(backtest_results),
            'losing_trades': 0,
            'large_losses': [],
            'consecutive_losses': [],
            'max_consecutive_losses': 0,
            'daily_losses': {},
            'monthly_losses': {},
            'worst_periods': [],
            'loss_distribution': {},
            'risk_metrics': {}
        }
        
        if not backtest_results:
            return loss_analysis
        
        # 分析每笔交易
        consecutive_count = 0
        max_consecutive = 0
        current_streak = []
        
        for i, trade in enumerate(backtest_results):
            pnl_usd = trade.pnl_usd if hasattr(trade, 'pnl_usd') else 0
            
            if pnl_usd < 0:
                loss_analysis['losing_trades'] += 1
                consecutive_count += 1
                current_streak.append(trade)
                
                # 检查是否为大额亏损
                if abs(pnl_usd) >= self.analysis_config['large_loss_threshold']:
                    loss_analysis['large_losses'].append({
                        'trade_id': i + 1,
                        'symbol': trade.symbol,
                        'loss_usd': pnl_usd,
                        'loss_percent': trade.pnl_percent if hasattr(trade, 'pnl_percent') else 0,
                        'entry_time': trade.entry_time,
                        'exit_time': trade.exit_time,
                        'holding_minutes': trade.holding_minutes if hasattr(trade, 'holding_minutes') else 0
                    })
                
                # 记录日损失
                trade_date = trade.entry_time.date()
                if trade_date not in loss_analysis['daily_losses']:
                    loss_analysis['daily_losses'][trade_date] = 0
                loss_analysis['daily_losses'][trade_date] += pnl_usd
                
                # 记录月损失
                month_key = f"{trade_date.year}-{trade_date.month:02d}"
                if month_key not in loss_analysis['monthly_losses']:
                    loss_analysis['monthly_losses'][month_key] = 0
                loss_analysis['monthly_losses'][month_key] += pnl_usd
                
            else:
                # 盈利交易，重置连续亏损计数
                if consecutive_count > 0:
                    loss_analysis['consecutive_losses'].append({
                        'count': consecutive_count,
                        'trades': current_streak.copy(),
                        'total_loss': sum(t.pnl_usd for t in current_streak),
                        'period': f"{current_streak[0].entry_time} 到 {current_streak[-1].exit_time}"
                    })
                    max_consecutive = max(max_consecutive, consecutive_count)
                    consecutive_count = 0
                    current_streak = []
        
        # 处理最后的连续亏损
        if consecutive_count > 0:
            loss_analysis['consecutive_losses'].append({
                'count': consecutive_count,
                'trades': current_streak.copy(),
                'total_loss': sum(t.pnl_usd for t in current_streak),
                'period': f"{current_streak[0].entry_time} 到 {current_streak[-1].exit_time}"
            })
            max_consecutive = max(max_consecutive, consecutive_count)
        
        loss_analysis['max_consecutive_losses'] = max_consecutive
        
        # 找出最糟糕的时期
        worst_days = sorted(loss_analysis['daily_losses'].items(), key=lambda x: x[1])[:5]
        worst_months = sorted(loss_analysis['monthly_losses'].items(), key=lambda x: x[1])[:3]
        
        loss_analysis['worst_periods'] = {
            'worst_days': [{'date': str(d), 'loss_usd': l} for d, l in worst_days],
            'worst_months': [{'month': m, 'loss_usd': l} for m, l in worst_months]
        }
        
        # 亏损分布统计
        all_losses = [t.pnl_usd for t in backtest_results if t.pnl_usd < 0]
        if all_losses:
            loss_analysis['loss_distribution'] = {
                'min_loss': min(all_losses),
                'max_loss': max(all_losses),
                'avg_loss': np.mean(all_losses),
                'median_loss': np.median(all_losses),
                'std_loss': np.std(all_losses),
                'percentiles': {
                    '95th': np.percentile(all_losses, 5),  # 最糟糕的5%
                    '90th': np.percentile(all_losses, 10),
                    '75th': np.percentile(all_losses, 25)
                }
            }
        
        # 风险指标
        total_pnl = sum(t.pnl_usd for t in backtest_results)
        losing_trades_pnl = sum(t.pnl_usd for t in backtest_results if t.pnl_usd < 0)
        
        loss_analysis['risk_metrics'] = {
            'win_rate': (loss_analysis['total_trades'] - loss_analysis['losing_trades']) / loss_analysis['total_trades'] * 100,
            'loss_rate': loss_analysis['losing_trades'] / loss_analysis['total_trades'] * 100,
            'avg_loss_per_losing_trade': losing_trades_pnl / loss_analysis['losing_trades'] if loss_analysis['losing_trades'] > 0 else 0,
            'total_losses_usd': losing_trades_pnl,
            'largest_drawdown_trade': min(all_losses) if all_losses else 0,
            'risk_of_ruin': self._calculate_risk_of_ruin(backtest_results)
        }
        
        logger.info(f"✅ 亏损分析完成:")
        logger.info(f"   总交易数: {loss_analysis['total_trades']}")
        logger.info(f"   亏损交易数: {loss_analysis['losing_trades']}")
        logger.info(f"   大额亏损数: {len(loss_analysis['large_losses'])}")
        logger.info(f"   最大连续亏损: {loss_analysis['max_consecutive_losses']}笔")
        
        return loss_analysis
    
    def _calculate_simple_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """简化RSI计算"""
        if len(prices) < period + 1:
            return 50  # 默认中性值
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    def _calculate_risk_of_ruin(self, trades: List) -> float:
        """计算破产风险"""
        if not trades:
            return 0.0
        
        wins = [t.pnl_usd for t in trades if t.pnl_usd > 0]
        losses = [abs(t.pnl_usd) for t in trades if t.pnl_usd < 0]
        
        if not wins or not losses:
            return 0.0
        
        win_rate = len(wins) / len(trades)
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)
        
        if avg_win <= avg_loss:
            return 100.0  # 如果平均亏损大于等于平均盈利，破产风险极高
        
        # 简化的破产风险计算
        advantage = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        if advantage <= 0:
            return 100.0
        
        # Kelly公式
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        
        if kelly <= 0:
            return 100.0
        
        # 简化的破产风险估算
        return max(0, min(100, (1 - kelly) * 100))
    
    def run_comprehensive_backtest(self) -> Dict:
        """运行综合深度回测"""
        logger.info("🚀 开始DipMaster V3深度历史回测...")
        
        # 加载数据
        market_data = self.load_and_prepare_data()
        if not market_data:
            logger.error("❌ 没有可用的市场数据")
            return {}
        
        # 确定回测时间范围
        all_start_dates = [df.index[0] for df in market_data.values()]
        all_end_dates = [df.index[-1] for df in market_data.values()]
        
        # 使用所有数据的重叠时间段
        backtest_start = max(all_start_dates)
        backtest_end = min(all_end_dates)
        
        logger.info(f"📅 回测时间范围: {backtest_start} 到 {backtest_end}")
        logger.info(f"📊 回测币种: {list(market_data.keys())}")
        
        # 配置回测参数
        config = BacktestConfig(
            start_date=backtest_start.strftime("%Y-%m-%d"),
            end_date=backtest_end.strftime("%Y-%m-%d"),
            initial_capital=10000,
            symbols=list(market_data.keys()),
            commission_rate=0.0004,  # 0.04%手续费
            slippage_bps=2.0,        # 2BP滑点
            max_positions=3,
            use_enhanced_signals=True,
            use_asymmetric_risk=True,
            use_adaptive_sizing=True,
            use_symbol_scoring=False,  # 简化测试
            use_time_filtering=False   # 简化测试
        )
        
        # 创建回测实例
        backtest = ComprehensiveBacktestV3(config)
        
        # 加载数据到回测器
        for symbol, df in market_data.items():
            # 过滤时间范围
            filtered_df = df[(df.index >= backtest_start) & (df.index <= backtest_end)]
            if len(filtered_df) > 100:
                backtest.price_data[symbol] = filtered_df
                logger.info(f"✅ {symbol}: 加载{len(filtered_df)}条数据")
        
        # 运行回测
        try:
            logger.info("⏳ 正在运行深度回测，这可能需要几分钟...")
            metrics = backtest.run_backtest()
            
            # 深度分析
            logger.info("🔍 开始深度分析...")
            
            # DIP策略复刻分析
            dip_analysis = self.analyze_dip_replication(backtest.trade_history, market_data)
            
            # 大额亏损分析
            loss_analysis = self.analyze_large_losses(backtest.trade_history)
            
            # 生成综合报告
            comprehensive_report = {
                'backtest_info': {
                    'strategy_version': 'DipMaster V3 Deep Backtest',
                    'start_date': config.start_date,
                    'end_date': config.end_date,
                    'symbols_tested': config.symbols,
                    'total_days': (backtest_end - backtest_start).days,
                    'data_points': sum(len(df) for df in backtest.price_data.values())
                },
                'performance_metrics': {
                    'total_trades': metrics.total_trades,
                    'winning_trades': metrics.winning_trades,
                    'losing_trades': metrics.losing_trades,
                    'win_rate': metrics.win_rate,
                    'total_return': metrics.total_return,
                    'profit_factor': metrics.profit_factor,
                    'max_drawdown_percent': metrics.max_drawdown_percent,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'avg_holding_minutes': metrics.avg_holding_minutes
                },
                'dip_strategy_analysis': dip_analysis,
                'loss_risk_analysis': loss_analysis,
                'risk_assessment': self._generate_risk_assessment(metrics, loss_analysis),
                'timestamp': datetime.now().isoformat()
            }
            
            # 保存详细报告
            report_file = self.results_path / f"dipmaster_v3_deep_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📄 详细报告已保存: {report_file}")
            
            return comprehensive_report
            
        except Exception as e:
            logger.error(f"❌ 回测执行失败: {e}")
            return {}
    
    def _generate_risk_assessment(self, metrics, loss_analysis: Dict) -> Dict:
        """生成风险评估"""
        risk_level = "LOW"
        warnings = []
        recommendations = []
        
        # 检查各项风险指标
        if metrics.max_drawdown_percent > 5:
            risk_level = "HIGH"
            warnings.append(f"最大回撤过高: {metrics.max_drawdown_percent:.1f}%")
        elif metrics.max_drawdown_percent > 3:
            risk_level = "MEDIUM"
            warnings.append(f"回撤偏高: {metrics.max_drawdown_percent:.1f}%")
        
        if loss_analysis['max_consecutive_losses'] > 5:
            risk_level = "HIGH"
            warnings.append(f"连续亏损过多: {loss_analysis['max_consecutive_losses']}笔")
        
        if len(loss_analysis['large_losses']) > 0:
            warnings.append(f"发现{len(loss_analysis['large_losses'])}笔大额亏损")
        
        if metrics.win_rate < 70:
            warnings.append(f"胜率偏低: {metrics.win_rate:.1f}%")
        
        # 生成建议
        if risk_level == "HIGH":
            recommendations.extend([
                "建议降低仓位大小",
                "加强风险控制参数",
                "考虑增加止损机制"
            ])
        elif risk_level == "MEDIUM":
            recommendations.extend([
                "监控风险指标变化",
                "考虑优化入场条件"
            ])
        else:
            recommendations.append("风险控制良好，可以考虑实盘测试")
        
        return {
            'risk_level': risk_level,
            'warnings': warnings,
            'recommendations': recommendations,
            'overall_assessment': self._get_overall_assessment(metrics, loss_analysis)
        }
    
    def _get_overall_assessment(self, metrics, loss_analysis: Dict) -> str:
        """综合评估"""
        if metrics.win_rate >= 75 and metrics.max_drawdown_percent <= 3 and loss_analysis['max_consecutive_losses'] <= 3:
            return "策略表现优秀，风险控制良好，建议进入纸面交易测试阶段"
        elif metrics.win_rate >= 65 and metrics.max_drawdown_percent <= 5:
            return "策略表现良好，但需要关注风险控制，建议进一步优化"
        else:
            return "策略存在较大风险，需要重新优化参数或调整策略逻辑"

def main():
    """主函数"""
    print("🎯 DipMaster V3 深度历史回测")
    print("=" * 60)
    
    # 创建回测器
    backtest_runner = DipMasterDeepBacktest()
    
    # 运行深度回测
    results = backtest_runner.run_comprehensive_backtest()
    
    if results:
        print("\n📊 回测结果摘要:")
        print("=" * 60)
        
        perf = results.get('performance_metrics', {})
        dip = results.get('dip_strategy_analysis', {})
        loss = results.get('loss_risk_analysis', {})
        risk = results.get('risk_assessment', {})
        
        print(f"总交易数: {perf.get('total_trades', 0)}")
        print(f"胜率: {perf.get('win_rate', 0):.1f}%")
        print(f"总收益: {perf.get('total_return', 0):.1f}%")
        print(f"最大回撤: {perf.get('max_drawdown_percent', 0):.1f}%")
        print(f"夏普率: {perf.get('sharpe_ratio', 0):.2f}")
        
        print(f"\n🎯 DIP策略复刻效果:")
        print(f"逢跌买入率: {dip.get('dip_entry_rate', 0):.1f}%")
        print(f"MA20下方率: {dip.get('price_below_ma20_rate', 0):.1f}%")
        print(f"边界出场率: {dip.get('boundary_exit_rate', 0):.1f}%")
        
        print(f"\n⚠️ 风险分析:")
        print(f"亏损交易数: {loss.get('losing_trades', 0)}")
        print(f"大额亏损数: {len(loss.get('large_losses', []))}")
        print(f"最大连续亏损: {loss.get('max_consecutive_losses', 0)}笔")
        print(f"风险等级: {risk.get('risk_level', 'UNKNOWN')}")
        
        print(f"\n📝 综合评估:")
        print(f"{risk.get('overall_assessment', 'N/A')}")
        
        if risk.get('warnings'):
            print(f"\n⚠️ 风险警告:")
            for warning in risk.get('warnings', []):
                print(f"  - {warning}")
        
        if risk.get('recommendations'):
            print(f"\n💡 建议:")
            for rec in risk.get('recommendations', []):
                print(f"  - {rec}")
        
        print("\n🎉 深度回测完成!")
        return 0
    else:
        print("❌ 回测失败")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)