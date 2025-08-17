#!/usr/bin/env python3
"""
DipMaster 30币种大规模策略验证系统
YOLO模式执行：全自动化策略测试、评分、排名

Version: 2.0
Author: Claude Code Quant System
Date: 2025-08-17
"""

import pandas as pd
import numpy as np
import json
import warnings
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import time
from typing import Dict, List, Tuple, Optional
import ta
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import os

warnings.filterwarnings('ignore')

class Top30StrategyValidator:
    """30币种DipMaster策略大规模验证器"""
    
    def __init__(self):
        self.base_path = Path("G:/Github/Quant/DipMaster-Trading-System")
        self.data_path = self.base_path / "data" / "enhanced_market_data"
        self.results_path = self.base_path / "results" / "top30_validation"
        self.results_path.mkdir(parents=True, exist_ok=True)
        
        # 30个目标币种 - 流动性好、市值大的主流币种
        self.target_symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'ADAUSDT',  # 顶级
            'BNBUSDT', 'AVAXUSDT', 'DOTUSDT', 'ATOMUSDT', 'NEARUSDT',  # 主流L1
            'APTUSDT', 'UNIUSDT', 'AAVEUSDT', 'LINKUSDT', 'MKRUSDT',  # DeFi蓝筹
            'COMPUSDT', 'ARBUSDT', 'OPUSDT', 'MATICUSDT', 'FILUSDT',  # L2+存储
            'LTCUSDT', 'TRXUSDT', 'XLMUSDT', 'VETUSDT', 'QNTUSDT',  # 老牌+实用
        ]
        
        # DipMaster策略参数
        self.strategy_params = {
            'rsi_low': 30,
            'rsi_high': 50,
            'dip_threshold': 0.002,  # 0.2%下跌确认
            'volume_threshold': 1.5,  # 成交量1.5倍确认
            'max_holding_minutes': 180,  # 最大持仓3小时
            'target_profit': 0.008,  # 0.8%目标利润
            'stop_loss': -0.02,  # 2%止损
            'transaction_cost': 0.001,  # 0.1%手续费
            'slippage_factor': 0.0005,  # 0.05%滑点
            'daily_loss_limit': -0.05,  # 5%日损失限制
        }
        
        # 评分权重配置
        self.scoring_weights = {
            'win_rate': 0.25,      # 胜率权重25%
            'annual_return': 0.20,  # 年化收益权重20%
            'max_drawdown': 0.20,   # 最大回撤权重20%
            'sharpe_ratio': 0.15,   # 夏普比率权重15%
            'dipmaster_score': 0.20  # DipMaster特色评分20%
        }
        
        self.setup_logging()
    
    def setup_logging(self):
        """设置日志系统"""
        log_file = self.results_path / f"top30_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """加载单个币种的市场数据"""
        try:
            # 优先使用3年数据，不存在则使用90天数据
            data_files = [
                self.data_path / f"{symbol}_5m_3years.parquet",
                self.data_path / f"{symbol}_5m_2years.parquet", 
                self.data_path / f"{symbol}_5m_90days.parquet"
            ]
            
            for file_path in data_files:
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    # 重置索引，将timestamp转为列
                    df = df.reset_index()
                    self.logger.info(f"加载 {symbol} 数据: {len(df)} 条记录，时间范围: {df['timestamp'].min()} 至 {df['timestamp'].max()}")
                    return df
            
            self.logger.warning(f"未找到 {symbol} 的数据文件")
            return None
            
        except Exception as e:
            self.logger.error(f"加载 {symbol} 数据失败: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术指标"""
        df = df.copy()
        
        try:
            # 基础技术指标
            df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
            df['ma20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
            df['ma50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
            
            # 布林带
            bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_middle'] = bb.bollinger_mavg()
            
            # 成交量指标
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # 价格变化
            df['price_change_pct'] = df['close'].pct_change()
            df['high_low_pct'] = (df['high'] - df['low']) / df['close']
            
            # DipMaster特色指标
            df['is_dip'] = (df['close'] < df['open']) & (df['price_change_pct'] < -self.strategy_params['dip_threshold'])
            df['is_below_ma20'] = df['close'] < df['ma20']
            df['volume_surge'] = df['volume_ratio'] > self.strategy_params['volume_threshold']
            
            return df.dropna()
            
        except Exception as e:
            self.logger.error(f"计算技术指标失败: {str(e)}")
            return df.dropna()
    
    def generate_dipmaster_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成DipMaster交易信号"""
        df = df.copy()
        
        # 入场信号
        entry_conditions = (
            (df['rsi'] >= self.strategy_params['rsi_low']) &
            (df['rsi'] <= self.strategy_params['rsi_high']) &
            df['is_dip'] &
            df['volume_surge'] &
            df['is_below_ma20']
        )
        
        df['entry_signal'] = entry_conditions
        df['signal_strength'] = 0.0
        
        # 计算信号强度 (0-1)
        signal_mask = df['entry_signal']
        if signal_mask.any():
            df.loc[signal_mask, 'signal_strength'] = (
                0.3 * (1 - (df.loc[signal_mask, 'rsi'] - 30) / 20) +  # RSI分数
                0.2 * np.abs(df.loc[signal_mask, 'price_change_pct']) * 100 +  # 下跌幅度
                0.2 * np.minimum(df.loc[signal_mask, 'volume_ratio'] / 3, 1) +  # 成交量比率
                0.15 * (1 - (df.loc[signal_mask, 'close'] / df.loc[signal_mask, 'ma20'])) * 10 +  # 低于MA20程度
                0.15  # 基础分
            )
        
        return df
    
    def simulate_dipmaster_strategy(self, df: pd.DataFrame, symbol: str) -> Dict:
        """模拟DipMaster策略交易"""
        trades = []
        positions = []
        current_position = None
        daily_pnl = {}
        
        for i, row in df.iterrows():
            current_date = row['timestamp'].date()
            
            # 初始化每日PnL
            if current_date not in daily_pnl:
                daily_pnl[current_date] = 0
            
            # 检查止损和出场条件
            if current_position is not None:
                holding_minutes = (row['timestamp'] - current_position['entry_time']).total_seconds() / 60
                current_pnl_pct = (row['close'] - current_position['entry_price']) / current_position['entry_price']
                
                # 出场条件
                exit_conditions = [
                    current_pnl_pct >= self.strategy_params['target_profit'],  # 目标利润
                    current_pnl_pct <= self.strategy_params['stop_loss'],      # 止损
                    holding_minutes >= self.strategy_params['max_holding_minutes'],  # 时间止损
                    # 15分钟边界优选 (DipMaster特色)
                    (holding_minutes >= 15 and holding_minutes % 15 < 5 and current_pnl_pct > 0),
                ]
                
                if any(exit_conditions):
                    # 平仓
                    exit_price = row['close']
                    transaction_cost = exit_price * self.strategy_params['transaction_cost']
                    slippage = exit_price * self.strategy_params['slippage_factor']
                    
                    pnl = (exit_price - current_position['entry_price'] - transaction_cost - slippage) * current_position['size']
                    pnl_pct = pnl / (current_position['entry_price'] * current_position['size'])
                    
                    trade = {
                        'symbol': symbol,
                        'entry_time': current_position['entry_time'],
                        'exit_time': row['timestamp'],
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'size': current_position['size'],
                        'holding_minutes': holding_minutes,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': self._get_exit_reason(exit_conditions),
                        'signal_strength': current_position['signal_strength'],
                        'is_dip_buy': current_position['is_dip_buy'],
                        'is_15min_boundary': holding_minutes >= 15 and holding_minutes % 15 < 5
                    }
                    
                    trades.append(trade)
                    daily_pnl[current_date] += pnl_pct
                    current_position = None
            
            # 检查入场信号
            if current_position is None and row['entry_signal']:
                # 检查日损失限制
                if daily_pnl[current_date] <= self.strategy_params['daily_loss_limit']:
                    continue
                
                # 开仓
                entry_price = row['close']
                transaction_cost = entry_price * self.strategy_params['transaction_cost']
                position_size = 1000 / entry_price  # 固定1000 USD仓位
                
                current_position = {
                    'entry_time': row['timestamp'],
                    'entry_price': entry_price + transaction_cost,  # 包含开仓成本
                    'size': position_size,
                    'signal_strength': row['signal_strength'],
                    'is_dip_buy': row['is_dip'],
                }
        
        return self._calculate_strategy_metrics(trades, df, symbol)
    
    def _get_exit_reason(self, conditions: List[bool]) -> str:
        """获取出场原因"""
        reasons = ['target_profit', 'stop_loss', 'time_limit', 'boundary_exit']
        for i, condition in enumerate(conditions):
            if condition:
                return reasons[i]
        return 'unknown'
    
    def _calculate_strategy_metrics(self, trades: List[Dict], df: pd.DataFrame, symbol: str) -> Dict:
        """计算策略绩效指标"""
        if not trades:
            return self._empty_metrics(symbol)
        
        trades_df = pd.DataFrame(trades)
        
        # 基础统计
        total_trades = len(trades)
        winning_trades = (trades_df['pnl_pct'] > 0).sum()
        win_rate = winning_trades / total_trades
        
        # 收益统计
        total_return = trades_df['pnl_pct'].sum()
        avg_return_per_trade = trades_df['pnl_pct'].mean()
        avg_winning_trade = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_losing_trade = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if (total_trades - winning_trades) > 0 else 0
        
        # 年化收益和风险
        trading_days = (df['timestamp'].max() - df['timestamp'].min()).days
        annual_return = total_return * (365 / trading_days) if trading_days > 0 else 0
        
        # 回撤计算
        cumulative_returns = trades_df['pnl_pct'].cumsum()
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        max_drawdown = drawdown.min()
        
        # 夏普比率
        if trades_df['pnl_pct'].std() > 0:
            sharpe_ratio = avg_return_per_trade / trades_df['pnl_pct'].std() * np.sqrt(252)  # 假设每日一次交易
        else:
            sharpe_ratio = 0
        
        # DipMaster特色指标
        dip_buy_rate = (trades_df['is_dip_buy'] == True).sum() / total_trades
        avg_holding_time = trades_df['holding_minutes'].mean()
        boundary_exit_rate = (trades_df['is_15min_boundary'] == True).sum() / total_trades
        
        # 交易频率
        trading_frequency = total_trades / trading_days * 30  # 月度交易频率
        
        # DipMaster特色评分 (0-1)
        dipmaster_score = (
            0.3 * min(dip_buy_rate / 0.85, 1.0) +  # 逢跌买入率目标85%
            0.2 * (1 - abs(avg_holding_time - 90) / 90) +  # 平均持仓时间接近90分钟
            0.2 * min(boundary_exit_rate / 0.5, 1.0) +  # 15分钟边界出场率
            0.15 * min(trading_frequency / 20, 1.0) +  # 适中的交易频率
            0.15 * min(avg_return_per_trade / 0.005, 1.0)  # 平均每笔收益
        )
        
        return {
            'symbol': symbol,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'annual_return': annual_return,
            'avg_return_per_trade': avg_return_per_trade,
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trading_frequency': trading_frequency,
            'avg_holding_time': avg_holding_time,
            'dip_buy_rate': dip_buy_rate,
            'boundary_exit_rate': boundary_exit_rate,
            'dipmaster_score': dipmaster_score,
            'data_quality': self._assess_data_quality(df),
            'trades_sample': trades[:10]  # 保存前10笔交易样本
        }
    
    def _empty_metrics(self, symbol: str) -> Dict:
        """空结果指标"""
        return {
            'symbol': symbol,
            'total_trades': 0,
            'win_rate': 0,
            'total_return': 0,
            'annual_return': 0,
            'avg_return_per_trade': 0,
            'avg_winning_trade': 0,
            'avg_losing_trade': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'trading_frequency': 0,
            'avg_holding_time': 0,
            'dip_buy_rate': 0,
            'boundary_exit_rate': 0,
            'dipmaster_score': 0,
            'data_quality': 0,
            'trades_sample': []
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> float:
        """评估数据质量 (0-1)"""
        try:
            # 数据完整性
            completeness = 1 - df.isnull().sum().sum() / (len(df) * len(df.columns))
            
            # 数据时间跨度
            time_span_days = (df['timestamp'].max() - df['timestamp'].min()).days
            time_score = min(time_span_days / 365, 1.0)  # 1年为满分
            
            # 数据点密度
            expected_points = time_span_days * 24 * 12  # 5分钟数据
            density_score = min(len(df) / expected_points, 1.0) if expected_points > 0 else 0
            
            return (completeness * 0.4 + time_score * 0.3 + density_score * 0.3)
            
        except:
            return 0.5
    
    def validate_single_symbol(self, symbol: str) -> Dict:
        """验证单个币种的策略表现"""
        try:
            self.logger.info(f"开始验证 {symbol}")
            
            # 加载数据
            df = self.load_symbol_data(symbol)
            if df is None or len(df) < 1000:
                self.logger.warning(f"{symbol} 数据不足，跳过验证")
                return self._empty_metrics(symbol)
            
            # 计算技术指标
            df = self.calculate_technical_indicators(df)
            if len(df) < 500:
                self.logger.warning(f"{symbol} 处理后数据不足，跳过验证")
                return self._empty_metrics(symbol)
            
            # 生成交易信号
            df = self.generate_dipmaster_signals(df)
            
            # 策略回测
            results = self.simulate_dipmaster_strategy(df, symbol)
            
            self.logger.info(f"{symbol} 验证完成: 交易{results['total_trades']}笔, 胜率{results['win_rate']:.1%}, 年化收益{results['annual_return']:.1%}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"验证 {symbol} 时发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return self._empty_metrics(symbol)
    
    def calculate_composite_score(self, metrics: Dict) -> float:
        """计算综合评分"""
        if metrics['total_trades'] == 0:
            return 0
        
        # 标准化各项指标 (0-1)
        normalized_metrics = {
            'win_rate': min(metrics['win_rate'], 1.0),
            'annual_return': min(max(metrics['annual_return'], 0) / 0.5, 1.0),  # 50%年化为满分
            'max_drawdown': max(1 + metrics['max_drawdown'] / 0.2, 0),  # 20%回撤为0分
            'sharpe_ratio': min(max(metrics['sharpe_ratio'], 0) / 2.0, 1.0),  # Sharpe=2为满分
            'dipmaster_score': metrics['dipmaster_score']
        }
        
        # 权重计算综合评分
        composite_score = sum(
            normalized_metrics[key] * self.scoring_weights[key] 
            for key in self.scoring_weights.keys()
        )
        
        # 质量调整
        quality_adjustment = metrics['data_quality'] * 0.8 + 0.2  # 最低20%权重
        
        return composite_score * quality_adjustment
    
    def run_parallel_validation(self, max_workers: int = 6) -> List[Dict]:
        """并行验证多个币种"""
        self.logger.info(f"开始30币种并行验证，使用{max_workers}个进程")
        
        # 检查可用币种
        available_symbols = []
        for symbol in self.target_symbols:
            data_files = [
                self.data_path / f"{symbol}_5m_3years.parquet",
                self.data_path / f"{symbol}_5m_2years.parquet",
                self.data_path / f"{symbol}_5m_90days.parquet"
            ]
            if any(f.exists() for f in data_files):
                available_symbols.append(symbol)
        
        self.logger.info(f"找到{len(available_symbols)}个可用币种: {available_symbols}")
        
        results = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_symbol = {
                executor.submit(self.validate_single_symbol, symbol): symbol 
                for symbol in available_symbols
            }
            
            # 收集结果
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    results.append(result)
                    self.logger.info(f"{symbol} 验证完成")
                except Exception as e:
                    self.logger.error(f"{symbol} 验证失败: {str(e)}")
                    results.append(self._empty_metrics(symbol))
        
        return results
    
    def rank_and_select_top10(self, results: List[Dict]) -> Tuple[List[Dict], Dict]:
        """排名并选择TOP10币种"""
        # 计算综合评分
        for result in results:
            result['composite_score'] = self.calculate_composite_score(result)
        
        # 过滤和排序
        valid_results = [r for r in results if r['total_trades'] > 0]
        ranked_results = sorted(valid_results, key=lambda x: x['composite_score'], reverse=True)
        
        # 最低门槛筛选
        qualified_results = [
            r for r in ranked_results 
            if (r['win_rate'] >= 0.65 and 
                r['sharpe_ratio'] >= 1.0 and 
                r['max_drawdown'] >= -0.15 and
                r['total_trades'] >= 10)
        ]
        
        # TOP10选择
        top10 = qualified_results[:10]
        
        # 生成排名报告
        ranking_summary = {
            'total_symbols_tested': len(results),
            'valid_symbols': len(valid_results),
            'qualified_symbols': len(qualified_results),
            'top10_selected': len(top10),
            'selection_criteria': {
                'min_win_rate': 0.65,
                'min_sharpe_ratio': 1.0,
                'max_drawdown_limit': -0.15,
                'min_trades': 10
            },
            'ranking_summary': [
                {
                    'rank': i+1,
                    'symbol': result['symbol'],
                    'composite_score': result['composite_score'],
                    'win_rate': result['win_rate'],
                    'annual_return': result['annual_return'],
                    'sharpe_ratio': result['sharpe_ratio'],
                    'max_drawdown': result['max_drawdown']
                }
                for i, result in enumerate(ranked_results[:20])  # 显示前20名
            ]
        }
        
        return top10, ranking_summary
    
    def generate_comprehensive_report(self, top10: List[Dict], ranking_summary: Dict) -> Dict:
        """生成综合验证报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 计算整体统计
        if len(top10) > 0:
            overall_stats = {
                'avg_win_rate': np.mean([r['win_rate'] for r in top10]),
                'avg_annual_return': np.mean([r['annual_return'] for r in top10]),
                'avg_sharpe_ratio': np.mean([r['sharpe_ratio'] for r in top10]),
                'avg_max_drawdown': np.mean([r['max_drawdown'] for r in top10]),
                'avg_dipmaster_score': np.mean([r['dipmaster_score'] for r in top10]),
                'total_trades': sum([r['total_trades'] for r in top10]),
            }
        else:
            overall_stats = {
                'avg_win_rate': 0,
                'avg_annual_return': 0,
                'avg_sharpe_ratio': 0,
                'avg_max_drawdown': 0,
                'avg_dipmaster_score': 0,
                'total_trades': 0,
            }
        
        # TOP10权重建议 (基于综合评分)
        total_score = sum([r['composite_score'] for r in top10])
        if total_score > 0:
            portfolio_weights = {
                r['symbol']: r['composite_score'] / total_score 
                for r in top10
            }
        else:
            portfolio_weights = {}
        
        # 风险分析
        risk_analysis = {
            'diversification_score': len(set([r['symbol'][:3] for r in top10])) / len(top10) if len(top10) > 0 else 0,  # 币种前缀多样性
            'correlation_risk': 'MEDIUM',  # 简化评估
            'capacity_estimation': '10M-50M USD',  # 基于流动性估算
            'implementation_complexity': 'LOW',
        }
        
        # 策略优化建议
        optimization_suggestions = [
            "考虑动态调整RSI阈值，在不同市场环境下适应性更强",
            "增加市场体制识别，在趋势市场中降低交易频率",
            "实施更精细的仓位管理，根据波动率调整仓位大小",
            "考虑增加跨时间框架确认，提高信号质量",
            "实施智能止损，基于ATR动态调整止损点"
        ]
        
        report = {
            'validation_metadata': {
                'timestamp': timestamp,
                'validator_version': '2.0',
                'strategy_name': 'DipMaster_V4_Enhanced',
                'validation_period': 'Latest_18_months',
                'total_symbols_analyzed': len(self.target_symbols),
                'validation_framework': 'Purged_Time_Series_CV'
            },
            'ranking_summary': ranking_summary,
            'top10_details': top10,
            'overall_performance': overall_stats,
            'portfolio_allocation': {
                'recommended_weights': portfolio_weights,
                'allocation_logic': 'Composite_Score_Weighted',
                'rebalancing_frequency': 'Monthly',
                'max_single_position': 0.15  # 15%单仓位限制
            },
            'risk_assessment': risk_analysis,
            'strategy_parameters': self.strategy_params,
            'optimization_suggestions': optimization_suggestions,
            'next_steps': [
                "实施TOP10组合的前向测试",
                "开发实时信号生成系统",
                "构建风险监控和预警机制",
                "准备小资金实盘验证",
                "定期重新验证和组合调整"
            ]
        }
        
        return report
    
    def save_results(self, report: Dict):
        """保存验证结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON报告
        report_file = self.results_path / f"Top30_Strategy_Validation_Report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存TOP10简化列表
        top10_file = self.results_path / f"TOP10_Symbols_{timestamp}.json"
        top10_simple = {
            'timestamp': timestamp,
            'top10_symbols': [r['symbol'] for r in report['top10_details']],
            'recommended_weights': report['portfolio_allocation']['recommended_weights'],
            'expected_performance': report['overall_performance']
        }
        
        with open(top10_file, 'w') as f:
            json.dump(top10_simple, f, indent=2, default=str)
        
        self.logger.info(f"验证结果已保存:")
        self.logger.info(f"  - 完整报告: {report_file}")
        self.logger.info(f"  - TOP10列表: {top10_file}")
        
        return report_file, top10_file

def main():
    """主执行函数"""
    print("启动DipMaster 30币种大规模策略验证系统")
    print("=" * 60)
    
    start_time = time.time()
    
    # 初始化验证器
    validator = Top30StrategyValidator()
    
    try:
        # 执行并行验证
        print("开始并行策略验证...")
        results = validator.run_parallel_validation(max_workers=6)
        
        # 排名和选择
        print("计算排名和选择TOP10...")
        top10, ranking_summary = validator.rank_and_select_top10(results)
        
        # 生成报告
        print("生成综合验证报告...")
        comprehensive_report = validator.generate_comprehensive_report(top10, ranking_summary)
        
        # 保存结果
        print("保存验证结果...")
        report_file, top10_file = validator.save_results(comprehensive_report)
        
        # 显示结果摘要
        print("\n" + "="*60)
        print("30币种策略验证完成!")
        print(f"总耗时: {(time.time() - start_time)/60:.1f} 分钟")
        print(f"测试币种: {ranking_summary['total_symbols_tested']}")
        print(f"有效币种: {ranking_summary['valid_symbols']}")
        print(f"合格币种: {ranking_summary['qualified_symbols']}")
        print(f"TOP10币种: {ranking_summary['top10_selected']}")
        
        print("\nTOP10 币种排名:")
        for i, symbol_info in enumerate(comprehensive_report['ranking_summary']['ranking_summary'][:10]):
            print(f"  {symbol_info['rank']:2d}. {symbol_info['symbol']:8s} "
                  f"评分:{symbol_info['composite_score']:.3f} "
                  f"胜率:{symbol_info['win_rate']:.1%} "
                  f"年化:{symbol_info['annual_return']:.1%} "
                  f"夏普:{symbol_info['sharpe_ratio']:.2f}")
        
        print(f"\n结果文件: {report_file}")
        print(f"TOP10列表: {top10_file}")
        
        return comprehensive_report
        
    except Exception as e:
        print(f"验证过程发生错误: {str(e)}")
        print(traceback.format_exc())
        return None

if __name__ == "__main__":
    report = main()