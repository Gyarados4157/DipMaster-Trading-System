#!/usr/bin/env python3
"""
Edge Analyzer - Phase 0 of Overfitting Optimization
确定DipMaster策略核心边界是否仍然存在

关键问题:
1. 基本假设: "RSI逢跌买入加密货币可获利" 是否仍然成立?
2. 失败模式分析: 71.2%亏损交易的根本原因
3. 市场演变影响: 2020-2025年市场结构变化对策略的影响
4. 竞争分析: 类似策略增加是否削弱了边界效应

Author: DipMaster Optimization Team
Date: 2025-08-15
Version: 1.0.0
"""

import sys
import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EdgeDiagnosis:
    """边界诊断结果"""
    edge_exists: bool
    confidence_score: float  # 0-1
    primary_failure_mode: str
    market_evolution_impact: str
    recommendation: str
    supporting_evidence: Dict[str, Any]


class EdgeAnalyzer:
    """边界存在性分析器"""
    
    def __init__(self):
        self.core_hypothesis = "RSI逢跌买入加密货币可获利"
        
        # 可能的失败模式
        self.failure_modes = [
            "edge_degraded_over_time",
            "stop_losses_too_tight_for_volatility", 
            "position_sizing_inappropriate",
            "market_microstructure_changed",
            "algorithmic_competition_increased",
            "crypto_maturation_reduced_inefficiencies",
            "correlation_breakdown_during_stress"
        ]
        
        # 市场时期定义
        self.market_periods = {
            'early_bull': ('2020-01-01', '2021-12-31'),
            'bear_crash': ('2022-01-01', '2022-12-31'), 
            'recovery': ('2023-01-01', '2024-12-31')
        }
        
    def load_historical_data(self, symbol: str = "ICPUSDT") -> pd.DataFrame:
        """加载历史数据用于边界分析"""
        data_file = f"data/market_data/{symbol}_5m_2years.csv"
        
        logger.info(f"🔍 Loading data for edge analysis: {symbol}")
        
        try:
            df = pd.read_csv(data_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            logger.info(f"✅ Data loaded: {len(df)} records from {df.index[0]} to {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Data loading failed: {e}")
            return pd.DataFrame()
    
    def calculate_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算基础技术指标"""
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 价格变化
        df['is_dip'] = df['close'] < df['open']
        df['price_change_pct'] = df['close'].pct_change()
        
        return df
    
    def test_minimal_strategy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """测试最简策略 - 只有3个参数"""
        
        logger.info("🧪 Testing minimal 3-parameter strategy...")
        
        # 超简参数 - 无优化
        RSI_THRESHOLD = 40
        POSITION_SIZE_PCT = 0.05  # 5% per trade
        MAX_HOLD_MINUTES = 60
        
        df = self.calculate_basic_indicators(df)
        
        trades = []
        current_position = None
        capital = 10000
        
        for i in range(len(df)):
            row = df.iloc[i]
            current_time = df.index[i]
            
            # 跳过空值
            if pd.isna(row['rsi']):
                continue
            
            # 出场检查
            if current_position:
                holding_minutes = (current_time - current_position['entry_time']).total_seconds() / 60
                pnl_pct = ((row['close'] - current_position['entry_price']) / current_position['entry_price']) * 100
                
                should_exit = False
                exit_reason = ""
                
                # 时间出场
                if holding_minutes >= MAX_HOLD_MINUTES:
                    should_exit = True
                    exit_reason = "time_exit"
                
                if should_exit:
                    pnl_usd = (row['close'] - current_position['entry_price']) * current_position['quantity']
                    commission = abs(pnl_usd) * 0.0008
                    net_pnl = pnl_usd - commission
                    
                    trades.append({
                        'entry_time': current_position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': current_position['entry_price'],
                        'exit_price': row['close'],
                        'holding_minutes': holding_minutes,
                        'pnl_usd': net_pnl,
                        'pnl_percent': pnl_pct,
                        'exit_reason': exit_reason,
                        'win': net_pnl > 0
                    })
                    
                    capital += net_pnl
                    current_position = None
            
            # 入场检查 - 最简条件
            if not current_position:
                if (row['rsi'] <= RSI_THRESHOLD and 
                    row['is_dip']):
                    
                    position_value = capital * POSITION_SIZE_PCT
                    current_position = {
                        'entry_time': current_time,
                        'entry_price': row['close'],
                        'quantity': position_value / row['close']
                    }
        
        # 分析结果
        if trades:
            wins = [t for t in trades if t['win']]
            win_rate = len(wins) / len(trades)
            
            total_return_pct = (capital - 10000) / 10000 * 100
            avg_holding = np.mean([t['holding_minutes'] for t in trades])
            
            win_pnl = [t['pnl_percent'] for t in trades if t['win']]
            loss_pnl = [t['pnl_percent'] for t in trades if not t['win']]
            
            avg_win = np.mean(win_pnl) if win_pnl else 0
            avg_loss = np.mean(loss_pnl) if loss_pnl else 0
            
        else:
            win_rate = 0
            total_return_pct = 0
            avg_holding = 0
            avg_win = 0
            avg_loss = 0
        
        results = {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'total_return_pct': total_return_pct,
            'final_capital': capital,
            'avg_holding_minutes': avg_holding,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'trades': trades
        }
        
        logger.info(f"📊 Minimal strategy results: {len(trades)} trades, {win_rate:.1%} win rate, {total_return_pct:+.1f}% return")
        
        return results
    
    def analyze_failure_patterns(self, trades: List[Dict]) -> Dict[str, Any]:
        """分析失败模式"""
        
        if not trades:
            return {'failure_mode': 'no_trades', 'analysis': {}}
        
        losing_trades = [t for t in trades if not t['win']]
        winning_trades = [t for t in trades if t['win']]
        
        logger.info(f"🔬 Analyzing failure patterns: {len(losing_trades)} losses vs {len(winning_trades)} wins")
        
        failure_analysis = {}
        
        # 1. 时间聚集分析
        if losing_trades:
            loss_times = [t['entry_time'] for t in losing_trades]
            loss_hours = [t.hour for t in loss_times]
            failure_analysis['time_clustering'] = {
                'worst_hours': sorted(set(loss_hours), key=loss_hours.count, reverse=True)[:3],
                'hour_distribution': {h: loss_hours.count(h) for h in set(loss_hours)}
            }
        
        # 2. 持仓时间分析
        if losing_trades and winning_trades:
            avg_loss_hold = np.mean([t['holding_minutes'] for t in losing_trades])
            avg_win_hold = np.mean([t['holding_minutes'] for t in winning_trades])
            
            failure_analysis['holding_time'] = {
                'avg_loss_minutes': avg_loss_hold,
                'avg_win_minutes': avg_win_hold,
                'hold_time_disparity': avg_loss_hold / avg_win_hold if avg_win_hold > 0 else float('inf')
            }
        
        # 3. 损失幅度分析
        if losing_trades:
            loss_magnitudes = [abs(t['pnl_percent']) for t in losing_trades]
            failure_analysis['loss_magnitude'] = {
                'avg_loss_pct': np.mean(loss_magnitudes),
                'max_loss_pct': max(loss_magnitudes),
                'loss_std': np.std(loss_magnitudes)
            }
        
        # 4. 市场条件关联
        # 简化版本：基于时间的市场条件推断
        market_condition_losses = {}
        for trade in losing_trades:
            month = trade['entry_time'].month
            if month in [12, 1, 2]:  # 通常低流动性
                season = 'low_liquidity'
            elif month in [3, 4, 5]:  # 春季反弹
                season = 'spring_rally'
            elif month in [6, 7, 8]:  # 夏季低迷
                season = 'summer_doldrums'  
            else:  # 秋季波动
                season = 'autumn_volatility'
                
            market_condition_losses[season] = market_condition_losses.get(season, 0) + 1
        
        failure_analysis['market_conditions'] = market_condition_losses
        
        return failure_analysis
    
    def assess_market_evolution_impact(self, df: pd.DataFrame) -> Dict[str, Any]:
        """评估市场演变影响"""
        
        logger.info("📈 Assessing market evolution impact...")
        
        evolution_analysis = {}
        
        # 按年份分组测试
        df['year'] = df.index.year
        yearly_performance = {}
        
        for year in df['year'].unique():
            year_data = df[df['year'] == year]
            if len(year_data) < 1000:  # 跳过数据不足的年份
                continue
                
            year_results = self.test_minimal_strategy(year_data.copy())
            yearly_performance[year] = {
                'win_rate': year_results['win_rate'],
                'total_return': year_results['total_return_pct'],
                'total_trades': year_results['total_trades'],
                'avg_holding': year_results['avg_holding_minutes']
            }
        
        evolution_analysis['yearly_performance'] = yearly_performance
        
        # 趋势分析
        if len(yearly_performance) >= 2:
            years = sorted(yearly_performance.keys())
            win_rates = [yearly_performance[y]['win_rate'] for y in years]
            
            # 简单线性趋势
            if len(win_rates) >= 3:
                trend_slope = (win_rates[-1] - win_rates[0]) / len(win_rates)
                evolution_analysis['performance_trend'] = {
                    'slope': trend_slope,
                    'direction': 'improving' if trend_slope > 0.02 else 'declining' if trend_slope < -0.02 else 'stable',
                    'volatility': np.std(win_rates)
                }
        
        # 波动率变化
        monthly_volatility = df.groupby(df.index.to_period('M'))['close'].apply(lambda x: x.pct_change().std())
        evolution_analysis['market_volatility_trend'] = {
            'avg_monthly_vol': monthly_volatility.mean(),
            'vol_trend': 'increasing' if monthly_volatility.iloc[-6:].mean() > monthly_volatility.iloc[:6].mean() else 'decreasing'
        }
        
        return evolution_analysis
    
    def diagnose_edge_existence(self, symbol: str = "ICPUSDT") -> EdgeDiagnosis:
        """综合边界存在性诊断"""
        
        logger.info("🚨 Starting comprehensive edge existence diagnosis...")
        
        # 加载数据
        df = self.load_historical_data(symbol)
        if df.empty:
            return EdgeDiagnosis(
                edge_exists=False,
                confidence_score=0.0,
                primary_failure_mode="data_unavailable",
                market_evolution_impact="unknown",
                recommendation="Fix data issues before proceeding",
                supporting_evidence={}
            )
        
        # 1. 最简策略测试
        minimal_results = self.test_minimal_strategy(df.copy())
        
        # 2. 失败模式分析
        failure_analysis = self.analyze_failure_patterns(minimal_results['trades'])
        
        # 3. 市场演变影响
        evolution_impact = self.assess_market_evolution_impact(df.copy())
        
        # 4. 边界存在性评估
        edge_exists = False
        confidence_score = 0.0
        primary_failure_mode = ""
        recommendation = ""
        
        # 评估标准
        win_rate = minimal_results['win_rate']
        total_return = minimal_results['total_return_pct']
        trade_count = minimal_results['total_trades']
        
        # 边界存在性判断
        if win_rate >= 0.55 and total_return > 5 and trade_count >= 50:
            edge_exists = True
            confidence_score = 0.8
            recommendation = "Edge exists and is strong. Proceed with optimization."
            
        elif win_rate >= 0.50 and total_return > 0 and trade_count >= 30:
            edge_exists = True
            confidence_score = 0.6
            recommendation = "Edge exists but weak. Proceed with extreme caution."
            
        elif win_rate >= 0.45 and trade_count >= 20:
            edge_exists = False
            confidence_score = 0.3
            primary_failure_mode = "marginal_performance"
            recommendation = "Edge questionable. Consider strategy pivot."
            
        else:
            edge_exists = False
            confidence_score = 0.1
            primary_failure_mode = "poor_performance"
            recommendation = "No viable edge detected. STOP optimization and pivot to new strategy."
        
        # 识别主要失败模式
        if not primary_failure_mode and not edge_exists:
            if failure_analysis.get('holding_time', {}).get('hold_time_disparity', 1) > 3:
                primary_failure_mode = "excessive_holding_time"
            elif failure_analysis.get('loss_magnitude', {}).get('avg_loss_pct', 0) > 2:
                primary_failure_mode = "large_loss_magnitude"
            else:
                primary_failure_mode = "general_underperformance"
        
        # 市场演变影响评估
        evolution_direction = evolution_impact.get('performance_trend', {}).get('direction', 'unknown')
        if evolution_direction == 'declining':
            confidence_score *= 0.8
            market_evolution_impact = "negative_trend_detected"
        elif evolution_direction == 'improving':
            market_evolution_impact = "positive_trend_detected"
        else:
            market_evolution_impact = "stable_performance"
        
        diagnosis = EdgeDiagnosis(
            edge_exists=edge_exists,
            confidence_score=confidence_score,
            primary_failure_mode=primary_failure_mode,
            market_evolution_impact=market_evolution_impact,
            recommendation=recommendation,
            supporting_evidence={
                'minimal_strategy_results': minimal_results,
                'failure_analysis': failure_analysis,
                'evolution_analysis': evolution_impact
            }
        )
        
        logger.info(f"✅ Edge diagnosis complete: Edge exists = {edge_exists}, Confidence = {confidence_score:.1%}")
        logger.info(f"📋 Recommendation: {recommendation}")
        
        return diagnosis


def main():
    """主函数 - 执行边界分析"""
    
    print("🔍 DipMaster Edge Existence Analysis (Phase 0)")
    print("=" * 80)
    
    analyzer = EdgeAnalyzer()
    
    # 执行诊断
    diagnosis = analyzer.diagnose_edge_existence("ICPUSDT")
    
    # 显示结果
    print(f"\n🎯 DIAGNOSIS RESULTS:")
    print(f"Edge Exists: {'✅ YES' if diagnosis.edge_exists else '❌ NO'}")
    print(f"Confidence: {diagnosis.confidence_score:.1%}")
    
    if diagnosis.primary_failure_mode:
        print(f"Primary Issue: {diagnosis.primary_failure_mode}")
    
    print(f"Market Evolution: {diagnosis.market_evolution_impact}")
    print(f"\n💡 RECOMMENDATION:")
    print(f"{diagnosis.recommendation}")
    
    # 保存诊断结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"edge_diagnosis_{timestamp}.json"
    
    # 转换为可JSON序列化的格式
    diagnosis_dict = {
        'edge_exists': diagnosis.edge_exists,
        'confidence_score': diagnosis.confidence_score,
        'primary_failure_mode': diagnosis.primary_failure_mode,
        'market_evolution_impact': diagnosis.market_evolution_impact,
        'recommendation': diagnosis.recommendation,
        'analysis_timestamp': datetime.now().isoformat(),
        'supporting_evidence': diagnosis.supporting_evidence
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(diagnosis_dict, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n📁 Detailed analysis saved to: {filename}")
    
    # 关键指标摘要
    if diagnosis.supporting_evidence:
        minimal_results = diagnosis.supporting_evidence.get('minimal_strategy_results', {})
        print(f"\n📊 KEY METRICS (Minimal 3-Parameter Strategy):")
        print(f"Total Trades: {minimal_results.get('total_trades', 0)}")
        print(f"Win Rate: {minimal_results.get('win_rate', 0):.1%}")
        print(f"Total Return: {minimal_results.get('total_return_pct', 0):+.1f}%")
        print(f"Avg Holding: {minimal_results.get('avg_holding_minutes', 0):.0f} minutes")
    
    print("\n" + "="*80)
    
    if not diagnosis.edge_exists:
        print("🚨 CRITICAL: No viable edge detected. Consider strategy pivot.")
    else:
        print(f"✅ Edge confirmed. Proceed to Phase 1 with confidence: {diagnosis.confidence_score:.1%}")


if __name__ == "__main__":
    main()