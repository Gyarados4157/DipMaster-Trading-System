#!/usr/bin/env python3
"""
Ultra Optimization Validator - 超级优化验证系统
==============================================

集成验证：
1. 短期优化效果验证：参数调整、止损优化、时间过滤、置信度提升
2. 中期优化效果验证：市场适应性、相关性控制、执行优化
3. 扩展币种池验证：20+优质非BTC/ETH标的
4. 综合性能评估：胜率、夏普率、回撤、稳定性

目标：
- 胜率从55%提升至75%+
- 评分从40.8提升至80+
- 风险等级从HIGH降至LOW

Author: DipMaster Ultra Team
Date: 2025-08-15
Version: 1.0.0
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# 导入我们的超级优化模块
from src.core.ultra_optimized_dipmaster import (
    UltraSignalGenerator, UltraSignalConfig, UltraSymbolPool,
    UltraRiskManager, MarketRegime
)
from src.tools.ultra_symbol_data_manager import UltraSymbolDataManager
from src.validation.comprehensive_validator import ComprehensiveValidator

logger = logging.getLogger(__name__)


@dataclass
class UltraValidationResults:
    """超级验证结果"""
    # 基础性能指标
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl_per_trade: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    
    # 优化效果对比
    baseline_win_rate: float = 55.0    # 原始胜率
    improved_win_rate: float = 0.0     # 优化后胜率
    win_rate_improvement: float = 0.0   # 胜率提升
    
    # 信号质量分析
    total_signals_generated: int = 0
    signals_filtered_out: int = 0
    signal_filter_rate: float = 0.0
    grade_a_signals: int = 0
    grade_b_signals: int = 0
    
    # 市场状态分析
    regime_performance: Dict[str, Dict] = None
    
    # 风险管理效果
    emergency_stops: int = 0
    trailing_stops: int = 0
    profit_takes: int = 0
    avg_holding_time_profit: float = 0.0
    avg_holding_time_loss: float = 0.0
    
    # 币种池表现
    symbol_performance: Dict[str, Dict] = None
    tier_1_performance: Dict = None
    tier_2_performance: Dict = None
    tier_3_performance: Dict = None
    
    # 综合评分
    technical_score: float = 0.0
    risk_management_score: float = 0.0
    diversification_score: float = 0.0
    execution_score: float = 0.0
    overall_score: float = 0.0
    risk_level: str = "UNKNOWN"
    
    def __post_init__(self):
        if self.regime_performance is None:
            self.regime_performance = {}
        if self.symbol_performance is None:
            self.symbol_performance = {}
        if self.tier_1_performance is None:
            self.tier_1_performance = {}
        if self.tier_2_performance is None:
            self.tier_2_performance = {}
        if self.tier_3_performance is None:
            self.tier_3_performance = {}


class UltraOptimizationValidator:
    """超级优化验证器"""
    
    def __init__(self, data_dir: str = "data/market_data"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path("results/ultra_validation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.signal_config = UltraSignalConfig()
        self.signal_generator = UltraSignalGenerator(self.signal_config)
        self.risk_manager = UltraRiskManager()
        self.symbol_pool = UltraSymbolPool()
        
        # 数据管理
        self.data_manager = UltraSymbolDataManager(str(self.data_dir))
        self.market_data: Dict[str, pd.DataFrame] = {}
        
        # 验证配置
        self.validation_start_date = "2023-08-14"
        self.validation_end_date = "2025-08-13"
        self.initial_capital = 10000
        self.position_size = 1000  # 固定仓位大小
        
        # 结果存储
        self.validation_results = UltraValidationResults()
        self.trade_history: List[Dict] = []
        self.signal_history: List[Dict] = []
        
    async def run_ultra_validation(self) -> UltraValidationResults:
        """运行超级优化验证"""
        logger.info("🚀 Starting Ultra Optimization Validation")
        
        try:
            # Phase 1: 准备数据
            await self._prepare_data()
            
            # Phase 2: 信号生成和过滤验证
            await self._validate_signal_optimization()
            
            # Phase 3: 风险管理验证
            await self._validate_risk_management()
            
            # Phase 4: 市场状态适应性验证
            await self._validate_market_adaptation()
            
            # Phase 5: 币种池多样化验证
            await self._validate_symbol_diversification()
            
            # Phase 6: 综合性能评估
            await self._calculate_comprehensive_score()
            
            # Phase 7: 生成报告
            await self._generate_validation_report()
            
            logger.info("✅ Ultra Optimization Validation Completed")
            return self.validation_results
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise
            
    async def _prepare_data(self):
        """准备数据"""
        logger.info("📊 Preparing market data...")
        
        # 获取所有可用的币种数据
        all_symbols = self.symbol_pool.current_symbols.copy()
        
        # 尝试添加新下载的币种
        for symbol in self.symbol_pool.target_symbols:
            data_file = self.data_dir / f"{symbol}_5m_2years.csv"
            if data_file.exists():
                all_symbols.append(symbol)
                
        logger.info(f"📈 Found data for {len(all_symbols)} symbols")
        
        # 加载数据
        loaded_count = 0
        for symbol in all_symbols:
            data_file = self.data_dir / f"{symbol}_5m_2years.csv"
            if data_file.exists():
                try:
                    df = pd.read_csv(data_file)
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    
                    # 过滤到验证期间
                    start_date = pd.to_datetime(self.validation_start_date)
                    end_date = pd.to_datetime(self.validation_end_date)
                    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
                    
                    if len(df) > 1000:  # 确保有足够数据
                        self.market_data[symbol] = df
                        loaded_count += 1
                        
                except Exception as e:
                    logger.error(f"Error loading {symbol}: {e}")
                    
        logger.info(f"✅ Successfully loaded {loaded_count} symbol datasets")
        
    async def _validate_signal_optimization(self):
        """验证信号优化效果"""
        logger.info("🔍 Validating signal optimization...")
        
        total_signals = 0
        filtered_signals = 0
        grade_distribution = {"A": 0, "B": 0, "C": 0, "D": 0}
        
        for symbol, df in self.market_data.items():
            logger.debug(f"Processing signals for {symbol}...")
            
            # 滑动窗口生成信号
            window_size = 100
            for i in range(window_size, len(df), 5):  # 每5个数据点检查一次
                window_df = df.iloc[i-window_size:i+1].copy()
                
                total_signals += 1
                signal = self.signal_generator.generate_ultra_signal(symbol, window_df)
                
                if signal:
                    self.signal_history.append(signal)
                    grade_distribution[signal["grade"]] += 1
                    
                    # 模拟交易执行
                    await self._simulate_trade_execution(signal, df, i)
                else:
                    filtered_signals += 1
                    
        # 更新结果
        self.validation_results.total_signals_generated = total_signals
        self.validation_results.signals_filtered_out = filtered_signals
        self.validation_results.signal_filter_rate = (filtered_signals / total_signals * 100) if total_signals > 0 else 0
        self.validation_results.grade_a_signals = grade_distribution["A"]
        self.validation_results.grade_b_signals = grade_distribution["B"]
        
        logger.info(f"📊 Signal Analysis Complete:")
        logger.info(f"  • Total Signals: {total_signals}")
        logger.info(f"  • Filter Rate: {self.validation_results.signal_filter_rate:.1f}%")
        logger.info(f"  • Grade A: {grade_distribution['A']}")
        logger.info(f"  • Grade B: {grade_distribution['B']}")
        
    async def _simulate_trade_execution(self, signal: Dict, df: pd.DataFrame, entry_index: int):
        """模拟交易执行"""
        entry_time = df.iloc[entry_index]['timestamp']
        entry_price = signal["price"]
        symbol = signal["symbol"]
        
        # 寻找出场点
        max_holding_minutes = signal["max_holding_minutes"]
        stop_loss_price = signal["stop_loss_price"]
        take_profit_levels = signal["take_profit_levels"]
        
        exit_index = None
        exit_reason = "TIME_LIMIT"
        exit_price = entry_price
        
        # 从入场点开始寻找出场
        for j in range(entry_index + 1, min(entry_index + max_holding_minutes//5 + 1, len(df))):
            current_row = df.iloc[j]
            current_price = current_row['close']
            current_time = current_row['timestamp']
            holding_minutes = (current_time - entry_time).total_seconds() / 60
            
            # 止损检查
            if current_price <= stop_loss_price:
                exit_index = j
                exit_price = current_price
                exit_reason = "STOP_LOSS"
                break
                
            # 止盈检查
            for level in take_profit_levels:
                if current_price >= level["price"]:
                    exit_index = j
                    exit_price = current_price
                    exit_reason = f"TAKE_PROFIT_L{level['level']}"
                    break
                    
            if exit_index:
                break
                
            # 时间止损
            if holding_minutes >= max_holding_minutes:
                exit_index = j
                exit_price = current_price
                exit_reason = "TIME_LIMIT"
                break
                
        # 记录交易
        if exit_index:
            exit_time = df.iloc[exit_index]['timestamp']
            holding_minutes = (exit_time - entry_time).total_seconds() / 60
            pnl = (exit_price - entry_price) / entry_price * self.position_size
            pnl_percent = (exit_price - entry_price) / entry_price * 100
            
            trade = {
                "symbol": symbol,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "holding_minutes": holding_minutes,
                "pnl_usd": pnl,
                "pnl_percent": pnl_percent,
                "exit_reason": exit_reason,
                "signal_grade": signal["grade"],
                "signal_confidence": signal["confidence"],
                "market_regime": signal["regime"],
                "is_winner": pnl > 0
            }
            
            self.trade_history.append(trade)
            
    async def _validate_risk_management(self):
        """验证风险管理效果"""
        logger.info("🛡️ Validating risk management...")
        
        if not self.trade_history:
            logger.warning("No trades to analyze for risk management")
            return
            
        trades_df = pd.DataFrame(self.trade_history)
        
        # 基础性能指标
        self.validation_results.total_trades = len(trades_df)
        self.validation_results.win_rate = (trades_df['is_winner'].sum() / len(trades_df)) * 100
        self.validation_results.total_pnl = trades_df['pnl_usd'].sum()
        self.validation_results.avg_pnl_per_trade = trades_df['pnl_usd'].mean()
        
        # 胜率改善
        self.validation_results.improved_win_rate = self.validation_results.win_rate
        self.validation_results.win_rate_improvement = self.validation_results.improved_win_rate - self.validation_results.baseline_win_rate
        
        # 夏普比率
        if trades_df['pnl_percent'].std() > 0:
            self.validation_results.sharpe_ratio = trades_df['pnl_percent'].mean() / trades_df['pnl_percent'].std()
        
        # 最大回撤
        cumulative_pnl = trades_df['pnl_usd'].cumsum()
        running_max = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - running_max
        self.validation_results.max_drawdown = drawdown.min()
        
        # 出场原因分析
        exit_reasons = trades_df['exit_reason'].value_counts()
        self.validation_results.emergency_stops = exit_reasons.get('STOP_LOSS', 0)
        self.validation_results.profit_takes = sum(exit_reasons.get(reason, 0) for reason in exit_reasons.index if 'TAKE_PROFIT' in reason)
        
        # 持仓时间分析
        winning_trades = trades_df[trades_df['is_winner']]
        losing_trades = trades_df[~trades_df['is_winner']]
        
        if len(winning_trades) > 0:
            self.validation_results.avg_holding_time_profit = winning_trades['holding_minutes'].mean()
        if len(losing_trades) > 0:
            self.validation_results.avg_holding_time_loss = losing_trades['holding_minutes'].mean()
            
        logger.info(f"🎯 Risk Management Results:")
        logger.info(f"  • Win Rate: {self.validation_results.win_rate:.1f}% (Target: 75%+)")
        logger.info(f"  • Win Rate Improvement: {self.validation_results.win_rate_improvement:+.1f}%")
        logger.info(f"  • Sharpe Ratio: {self.validation_results.sharpe_ratio:.2f}")
        logger.info(f"  • Max Drawdown: ${self.validation_results.max_drawdown:.2f}")
        
    async def _validate_market_adaptation(self):
        """验证市场状态适应性"""
        logger.info("🔄 Validating market adaptation...")
        
        if not self.trade_history:
            return
            
        trades_df = pd.DataFrame(self.trade_history)
        
        # 按市场状态分组分析
        regime_groups = trades_df.groupby('market_regime')
        
        for regime, group in regime_groups:
            performance = {
                "trade_count": len(group),
                "win_rate": (group['is_winner'].sum() / len(group)) * 100,
                "avg_pnl": group['pnl_usd'].mean(),
                "total_pnl": group['pnl_usd'].sum(),
                "avg_holding_time": group['holding_minutes'].mean()
            }
            self.validation_results.regime_performance[regime] = performance
            
        logger.info("📊 Market Regime Performance:")
        for regime, perf in self.validation_results.regime_performance.items():
            logger.info(f"  • {regime}: {perf['trade_count']} trades, {perf['win_rate']:.1f}% win rate")
            
    async def _validate_symbol_diversification(self):
        """验证币种池多样化效果"""
        logger.info("🌐 Validating symbol diversification...")
        
        if not self.trade_history:
            return
            
        trades_df = pd.DataFrame(self.trade_history)
        
        # 按币种分析
        symbol_groups = trades_df.groupby('symbol')
        
        for symbol, group in symbol_groups:
            performance = {
                "trade_count": len(group),
                "win_rate": (group['is_winner'].sum() / len(group)) * 100,
                "avg_pnl": group['pnl_usd'].mean(),
                "total_pnl": group['pnl_usd'].sum(),
                "sharpe": group['pnl_percent'].mean() / group['pnl_percent'].std() if group['pnl_percent'].std() > 0 else 0
            }
            self.validation_results.symbol_performance[symbol] = performance
            
        # 按Tier分析
        tier_performance = {"tier_1": [], "tier_2": [], "tier_3": []}
        
        for symbol, perf in self.validation_results.symbol_performance.items():
            if symbol in self.symbol_pool.tier_1_symbols:
                tier_performance["tier_1"].append(perf)
            elif symbol in self.symbol_pool.tier_2_symbols:
                tier_performance["tier_2"].append(perf)
            else:
                tier_performance["tier_3"].append(perf)
                
        # 计算Tier级别统计
        for tier, perfs in tier_performance.items():
            if perfs:
                tier_stats = {
                    "symbol_count": len(perfs),
                    "avg_win_rate": np.mean([p['win_rate'] for p in perfs]),
                    "total_trades": sum([p['trade_count'] for p in perfs]),
                    "total_pnl": sum([p['total_pnl'] for p in perfs])
                }
                setattr(self.validation_results, f"{tier}_performance", tier_stats)
                
        logger.info("🎯 Symbol Diversification Results:")
        logger.info(f"  • Active Symbols: {len(self.validation_results.symbol_performance)}")
        logger.info(f"  • Tier 1 Avg Win Rate: {getattr(self.validation_results, 'tier_1_performance', {}).get('avg_win_rate', 0):.1f}%")
        logger.info(f"  • Tier 2 Avg Win Rate: {getattr(self.validation_results, 'tier_2_performance', {}).get('avg_win_rate', 0):.1f}%")
        
    async def _calculate_comprehensive_score(self):
        """计算综合评分"""
        logger.info("📊 Calculating comprehensive score...")
        
        # 技术分析评分 (30%)
        tech_score = 0
        if self.validation_results.win_rate >= 75:
            tech_score = 90
        elif self.validation_results.win_rate >= 65:
            tech_score = 75
        elif self.validation_results.win_rate >= 55:
            tech_score = 60
        else:
            tech_score = 40
            
        # 信号过滤质量加分
        if self.validation_results.signal_filter_rate >= 70:
            tech_score += 5
        if self.validation_results.grade_a_signals / max(self.validation_results.total_signals_generated - self.validation_results.signals_filtered_out, 1) >= 0.3:
            tech_score += 5
            
        self.validation_results.technical_score = min(tech_score, 100)
        
        # 风险管理评分 (25%)
        risk_score = 50
        if self.validation_results.sharpe_ratio >= 1.5:
            risk_score += 25
        elif self.validation_results.sharpe_ratio >= 1.0:
            risk_score += 15
        elif self.validation_results.sharpe_ratio >= 0.5:
            risk_score += 10
            
        if abs(self.validation_results.max_drawdown) <= 500:  # 小于$500回撤
            risk_score += 15
        elif abs(self.validation_results.max_drawdown) <= 1000:
            risk_score += 10
        elif abs(self.validation_results.max_drawdown) <= 2000:
            risk_score += 5
            
        # 持仓时间优化加分
        if self.validation_results.avg_holding_time_loss > 0 and self.validation_results.avg_holding_time_profit > 0:
            time_ratio = self.validation_results.avg_holding_time_profit / self.validation_results.avg_holding_time_loss
            if time_ratio >= 1.5:  # 盈利持仓时间长于亏损
                risk_score += 10
                
        self.validation_results.risk_management_score = min(risk_score, 100)
        
        # 多样化评分 (20%)
        diversification_score = len(self.validation_results.symbol_performance) * 5  # 每个活跃币种5分
        diversification_score = min(diversification_score, 100)
        self.validation_results.diversification_score = diversification_score
        
        # 执行效果评分 (15%)
        execution_score = 70  # 基础分
        if self.validation_results.win_rate_improvement > 0:
            execution_score += min(self.validation_results.win_rate_improvement * 2, 30)
        self.validation_results.execution_score = min(execution_score, 100)
        
        # 市场适应性评分 (10%)
        adaptation_score = 60
        if self.validation_results.regime_performance:
            good_regimes = ['sideways', 'low_vol', 'accumulation']
            good_performance = [perf for regime, perf in self.validation_results.regime_performance.items() 
                              if any(good in regime.lower() for good in good_regimes)]
            if good_performance:
                avg_good_win_rate = np.mean([p['win_rate'] for p in good_performance])
                if avg_good_win_rate >= 75:
                    adaptation_score += 25
                elif avg_good_win_rate >= 65:
                    adaptation_score += 15
                    
        # 综合评分
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]
        scores = [
            self.validation_results.technical_score,
            self.validation_results.risk_management_score,
            self.validation_results.diversification_score,
            self.validation_results.execution_score,
            adaptation_score
        ]
        
        self.validation_results.overall_score = sum(w * s for w, s in zip(weights, scores))
        
        # 风险等级评估
        if self.validation_results.overall_score >= 80:
            self.validation_results.risk_level = "LOW"
        elif self.validation_results.overall_score >= 60:
            self.validation_results.risk_level = "MEDIUM"
        else:
            self.validation_results.risk_level = "HIGH"
            
        logger.info("🎯 Comprehensive Scoring Complete:")
        logger.info(f"  • Technical Score: {self.validation_results.technical_score:.1f}/100")
        logger.info(f"  • Risk Management: {self.validation_results.risk_management_score:.1f}/100")
        logger.info(f"  • Diversification: {self.validation_results.diversification_score:.1f}/100")
        logger.info(f"  • Execution: {self.validation_results.execution_score:.1f}/100")
        logger.info(f"  • Overall Score: {self.validation_results.overall_score:.1f}/100")
        logger.info(f"  • Risk Level: {self.validation_results.risk_level}")
        
    async def _generate_validation_report(self):
        """生成验证报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON结果
        results_file = self.results_dir / f"ultra_validation_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(asdict(self.validation_results), f, indent=2, default=str)
            
        # 交易历史
        if self.trade_history:
            trades_file = self.results_dir / f"ultra_trades_{timestamp}.csv"
            trades_df = pd.DataFrame(self.trade_history)
            trades_df.to_csv(trades_file, index=False)
            
        # Markdown报告
        report_file = self.results_dir / f"ULTRA_VALIDATION_REPORT_{timestamp}.md"
        await self._create_markdown_report(report_file)
        
        logger.info(f"📋 Validation report generated:")
        logger.info(f"  • Results: {results_file}")
        logger.info(f"  • Trades: {trades_file if self.trade_history else 'N/A'}")
        logger.info(f"  • Report: {report_file}")
        
    async def _create_markdown_report(self, report_file: Path):
        """创建Markdown报告"""
        content = f"""# 🚀 DipMaster Ultra Optimization Validation Report

## 📋 验证概览

**验证时间**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}  
**验证期间**: {self.validation_start_date} 至 {self.validation_end_date}  
**综合评分**: {self.validation_results.overall_score:.1f}/100  
**风险等级**: {self.validation_results.risk_level}  
**验证结果**: {'✅ 通过' if self.validation_results.overall_score >= 70 else '❌ 未通过'}  

## 🎯 优化目标达成情况

### 短期优化效果
| 优化项目 | 目标 | 实际结果 | 达成状态 |
|---------|------|----------|----------|
| 胜率提升 | 75%+ | {self.validation_results.win_rate:.1f}% | {'✅' if self.validation_results.win_rate >= 75 else '❌'} |
| 胜率改善 | +20% | {self.validation_results.win_rate_improvement:+.1f}% | {'✅' if self.validation_results.win_rate_improvement >= 20 else '❌'} |
| 信号过滤率 | 70%+ | {self.validation_results.signal_filter_rate:.1f}% | {'✅' if self.validation_results.signal_filter_rate >= 70 else '❌'} |
| A级信号比例 | 30%+ | {(self.validation_results.grade_a_signals / max(self.validation_results.total_signals_generated - self.validation_results.signals_filtered_out, 1) * 100):.1f}% | {'✅' if (self.validation_results.grade_a_signals / max(self.validation_results.total_signals_generated - self.validation_results.signals_filtered_out, 1)) >= 0.3 else '❌'} |

### 中期优化效果
| 优化项目 | 评分 | 状态 |
|---------|------|------|
| 技术分析优化 | {self.validation_results.technical_score:.1f}/100 | {'✅ 优秀' if self.validation_results.technical_score >= 85 else '✅ 良好' if self.validation_results.technical_score >= 70 else '❌ 需改进'} |
| 风险管理增强 | {self.validation_results.risk_management_score:.1f}/100 | {'✅ 优秀' if self.validation_results.risk_management_score >= 85 else '✅ 良好' if self.validation_results.risk_management_score >= 70 else '❌ 需改进'} |
| 币种多样化 | {self.validation_results.diversification_score:.1f}/100 | {'✅ 优秀' if self.validation_results.diversification_score >= 85 else '✅ 良好' if self.validation_results.diversification_score >= 70 else '❌ 需改进'} |
| 执行效果 | {self.validation_results.execution_score:.1f}/100 | {'✅ 优秀' if self.validation_results.execution_score >= 85 else '✅ 良好' if self.validation_results.execution_score >= 70 else '❌ 需改进'} |

## 📊 详细性能分析

### 核心指标
- **交易总数**: {self.validation_results.total_trades}
- **胜率**: {self.validation_results.win_rate:.2f}%
- **总盈亏**: ${self.validation_results.total_pnl:.2f}
- **平均每笔盈亏**: ${self.validation_results.avg_pnl_per_trade:.2f}
- **夏普比率**: {self.validation_results.sharpe_ratio:.2f}
- **最大回撤**: ${self.validation_results.max_drawdown:.2f}

### 信号质量分析
- **总信号数**: {self.validation_results.total_signals_generated}
- **过滤信号数**: {self.validation_results.signals_filtered_out}
- **信号过滤率**: {self.validation_results.signal_filter_rate:.1f}%
- **A级信号**: {self.validation_results.grade_a_signals}
- **B级信号**: {self.validation_results.grade_b_signals}

### 风险管理效果
- **紧急止损次数**: {self.validation_results.emergency_stops}
- **止盈次数**: {self.validation_results.profit_takes}
- **盈利平均持仓时间**: {self.validation_results.avg_holding_time_profit:.1f}分钟
- **亏损平均持仓时间**: {self.validation_results.avg_holding_time_loss:.1f}分钟

### 市场状态表现
"""
        
        # 添加市场状态表现
        if self.validation_results.regime_performance:
            content += "| 市场状态 | 交易数 | 胜率 | 平均盈亏 | 总盈亏 |\n"
            content += "|---------|-------|------|----------|--------|\n"
            for regime, perf in self.validation_results.regime_performance.items():
                content += f"| {regime} | {perf['trade_count']} | {perf['win_rate']:.1f}% | ${perf['avg_pnl']:.2f} | ${perf['total_pnl']:.2f} |\n"
                
        # 添加币种表现
        content += "\n### 币种表现排名\n"
        if self.validation_results.symbol_performance:
            sorted_symbols = sorted(
                self.validation_results.symbol_performance.items(),
                key=lambda x: x[1]['win_rate'],
                reverse=True
            )
            
            content += "| 币种 | 交易数 | 胜率 | 夏普比率 | 总盈亏 |\n"
            content += "|------|-------|------|----------|--------|\n"
            for symbol, perf in sorted_symbols[:15]:  # Top 15
                content += f"| {symbol} | {perf['trade_count']} | {perf['win_rate']:.1f}% | {perf['sharpe']:.2f} | ${perf['total_pnl']:.2f} |\n"
                
        # 结论和建议
        content += f"""
## 🎯 验证结论

### 总体评估
{'✅ **验证通过**' if self.validation_results.overall_score >= 70 else '❌ **验证未通过**'}

综合评分 **{self.validation_results.overall_score:.1f}/100**，风险等级 **{self.validation_results.risk_level}**

### 优化效果总结
1. **信号质量**: {'大幅提升' if self.validation_results.signal_filter_rate >= 70 else '有所提升'}，过滤率达到{self.validation_results.signal_filter_rate:.1f}%
2. **胜率改善**: {'显著提升' if self.validation_results.win_rate_improvement >= 10 else '小幅提升'}{self.validation_results.win_rate_improvement:+.1f}%，达到{self.validation_results.win_rate:.1f}%
3. **风险控制**: {'优秀' if self.validation_results.risk_management_score >= 80 else '良好' if self.validation_results.risk_management_score >= 60 else '需改进'}
4. **币种多样化**: 活跃交易{len(self.validation_results.symbol_performance)}个币种

### 下一步建议
"""
        
        if self.validation_results.overall_score >= 80:
            content += """
🚀 **建议实盘部署**
- 系统已达到优秀水平，可考虑小资金实盘测试
- 建议初始资金不超过总资金的10%
- 实盘运行1个月后评估是否扩大规模
"""
        elif self.validation_results.overall_score >= 70:
            content += """
✅ **建议谨慎实盘**
- 系统达到可接受水平，建议进一步优化后实盘
- 重点关注胜率和风险管理指标
- 可进行更长期的历史回测验证
"""
        else:
            content += """
❌ **建议继续优化**
- 系统尚未达到实盘要求，需要进一步优化
- 重点改进信号质量和风险控制
- 建议重新审视策略参数和过滤条件
"""
            
        content += f"""
---

**📝 报告生成**: {datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")}  
**🔍 验证框架**: DipMaster Ultra Optimization System v1.0.0  
**⚠️ 重要提醒**: 本报告基于历史数据验证，实盘交易仍存在风险，请谨慎决策
"""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(content)


async def main():
    """主验证函数"""
    logger.info("🚀 Starting Ultra Optimization Validation")
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 创建验证器
        validator = UltraOptimizationValidator()
        
        # 运行验证
        results = await validator.run_ultra_validation()
        
        # 输出关键结果
        logger.info("🎉 Ultra Optimization Validation Completed!")
        logger.info("="*60)
        logger.info(f"📊 Overall Score: {results.overall_score:.1f}/100")
        logger.info(f"⚠️  Risk Level: {results.risk_level}")
        logger.info(f"🎯 Win Rate: {results.win_rate:.1f}% (Improvement: {results.win_rate_improvement:+.1f}%)")
        logger.info(f"💰 Total PnL: ${results.total_pnl:.2f}")
        logger.info(f"📈 Sharpe Ratio: {results.sharpe_ratio:.2f}")
        logger.info(f"📉 Max Drawdown: ${results.max_drawdown:.2f}")
        logger.info("="*60)
        
        if results.overall_score >= 80:
            logger.info("✅ EXCELLENT - Ready for live trading!")
        elif results.overall_score >= 70:
            logger.info("✅ GOOD - Consider cautious live trading")
        else:
            logger.info("❌ NEEDS IMPROVEMENT - Continue optimization")
            
        return results
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())