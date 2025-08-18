#!/usr/bin/env python3
"""
DipMaster持续组合风险管理系统
Continuous Portfolio Risk Management System

核心功能：
1. 持续信号处理和组合优化
2. 实时风险监控和控制
3. 动态权重调整和再平衡
4. Kelly公式优化的仓位管理
5. Beta中性和波动率控制
6. VaR/ES实时监控

作者: DipMaster Trading System
版本: V1.0.0 - Continuous Portfolio Risk Management
"""

import pandas as pd
import numpy as np
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import time
import schedule

# 数值计算
import cvxpy as cp
from scipy.optimize import minimize
from scipy import stats
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler

# 导入现有组件
import sys
sys.path.append('/Users/zhangxuanyang/Desktop/Quant/DipMaster-Trading-System')

from src.core.portfolio_risk_optimizer import PortfolioRiskOptimizer, PortfolioConstraints, RiskMetrics
from src.monitoring.metrics_collector import MetricsCollector

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ContinuousPortfolioRiskManager')

@dataclass
class ContinuousRiskConfig:
    """持续风险管理配置"""
    # 组合优化参数
    base_capital: float = 100000
    rebalance_frequency: str = "hourly"  # hourly, daily, weekly
    min_signal_confidence: float = 0.60
    min_expected_return: float = 0.005
    
    # 风险约束
    max_portfolio_beta: float = 0.10
    max_portfolio_volatility: float = 0.18
    max_single_position: float = 0.20
    max_total_leverage: float = 3.0
    max_var_95: float = 0.03
    max_es_95: float = 0.04
    max_drawdown: float = 0.03
    
    # Kelly优化参数
    kelly_fraction: float = 0.25  # 保守Kelly比例
    min_kelly_weight: float = 0.01
    max_kelly_weight: float = 0.25
    
    # 相关性和分散化
    max_correlation_threshold: float = 0.70
    min_diversification_ratio: float = 1.20
    
    # 时间控制
    position_hold_time_limit: int = 180  # 分钟
    force_rebalance_time: int = 240  # 分钟
    
    # 数据路径
    signal_data_path: str = "results/basic_ml_pipeline/"
    output_path: str = "results/continuous_risk_management/"

@dataclass  
class PortfolioPosition:
    """组合仓位"""
    symbol: str
    weight: float
    dollar_amount: float
    entry_time: datetime
    signal_strength: float
    confidence: float
    expected_return: float
    kelly_weight: float
    risk_contribution: float

@dataclass
class RiskAlert:
    """风险告警"""
    alert_type: str
    priority: str  # HIGH, MEDIUM, LOW
    description: str
    current_value: float
    threshold: float
    timestamp: datetime
    
class ContinuousPortfolioRiskManager:
    """持续组合风险管理器"""
    
    def __init__(self, config: ContinuousRiskConfig):
        self.config = config
        self.is_running = False
        
        # 初始化组件
        self.portfolio_optimizer = PortfolioRiskOptimizer(
            base_capital=config.base_capital,
            constraints=PortfolioConstraints(
                target_beta=0.0,
                beta_tolerance=config.max_portfolio_beta,
                max_position=config.max_single_position,
                max_leverage=config.max_total_leverage,
                target_volatility=config.max_portfolio_volatility,
                max_var_95=config.max_var_95
            )
        )
        
        # 状态管理
        self.current_positions: Dict[str, PortfolioPosition] = {}
        self.portfolio_history: List[Dict] = []
        self.risk_alerts: List[RiskAlert] = []
        self.performance_metrics = {}
        
        # 风险监控
        self.risk_monitor_data = {
            'beta_history': [],
            'volatility_history': [], 
            'var_history': [],
            'drawdown_history': [],
            'correlation_matrix': None
        }
        
        # 创建输出目录
        Path(config.output_path).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Continuous Portfolio Risk Manager initialized")
        logger.info(f"Base Capital: ${config.base_capital:,.2f}")
        logger.info(f"Rebalance Frequency: {config.rebalance_frequency}")
        logger.info(f"Output Path: {config.output_path}")

    async def load_latest_signals(self) -> pd.DataFrame:
        """加载最新的Alpha信号"""
        try:
            signal_path = Path(self.config.signal_data_path)
            
            # 查找最新的信号文件
            signal_files = list(signal_path.glob("signals_*.csv"))
            if not signal_files:
                logger.warning("No signal files found")
                return pd.DataFrame()
            
            latest_file = max(signal_files, key=lambda x: x.stat().st_mtime)
            signals_df = pd.read_csv(latest_file)
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])
            
            # 过滤信号质量
            filtered_signals = signals_df[
                (signals_df['confidence'] >= self.config.min_signal_confidence) &
                (signals_df['predicted_return'] >= self.config.min_expected_return)
            ].copy()
            
            logger.info(f"Loaded {len(signals_df)} signals, filtered to {len(filtered_signals)}")
            return filtered_signals
            
        except Exception as e:
            logger.error(f"Error loading signals: {e}")
            return pd.DataFrame()

    def calculate_kelly_optimal_weights(self, signals_df: pd.DataFrame) -> Dict[str, float]:
        """使用Kelly公式计算最优权重"""
        kelly_weights = {}
        
        for _, signal in signals_df.iterrows():
            symbol = signal['symbol']
            expected_return = signal['predicted_return']
            confidence = signal['confidence']
            
            # Kelly公式参数
            win_prob = confidence
            loss_prob = 1 - win_prob
            
            # 假设亏损比例（保守估计）
            avg_win = expected_return
            avg_loss = -expected_return * 0.5  # 假设亏损是盈利的一半
            
            if win_prob > 0.5 and avg_win > 0:
                # Kelly公式: f* = (bp - q) / b
                # 其中 b = 赔率, p = 胜率, q = 败率
                kelly_fraction = (win_prob * avg_win + loss_prob * avg_loss) / (avg_win * avg_loss) if avg_loss != 0 else 0
                
                # 应用Kelly缩放因子
                kelly_weight = kelly_fraction * self.config.kelly_fraction
                
                # 限制权重范围
                kelly_weight = max(self.config.min_kelly_weight, 
                                 min(kelly_weight, self.config.max_kelly_weight))
                
                kelly_weights[symbol] = kelly_weight
            else:
                kelly_weights[symbol] = 0
                
        return kelly_weights

    def check_correlation_constraints(self, signals_df: pd.DataFrame) -> Dict[str, float]:
        """检查相关性约束并调整权重"""
        symbols = signals_df['symbol'].unique()
        
        if len(symbols) <= 1:
            return {}
            
        # 模拟相关性矩阵（实际应从市场数据计算）
        np.random.seed(42)
        corr_matrix = np.random.uniform(0.3, 0.8, (len(symbols), len(symbols)))
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        np.fill_diagonal(corr_matrix, 1.0)
        
        # 识别高相关性对
        high_corr_pairs = []
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                if corr_matrix[i, j] > self.config.max_correlation_threshold:
                    high_corr_pairs.append((symbols[i], symbols[j], corr_matrix[i, j]))
        
        correlation_adjustments = {}
        for symbol1, symbol2, corr in high_corr_pairs:
            # 对高相关性资产降权
            adjustment_factor = 1 - (corr - self.config.max_correlation_threshold) * 2
            correlation_adjustments[symbol1] = adjustment_factor
            correlation_adjustments[symbol2] = adjustment_factor
            
            logger.warning(f"High correlation detected: {symbol1}-{symbol2} = {corr:.3f}")
        
        return correlation_adjustments

    async def optimize_portfolio(self, signals_df: pd.DataFrame) -> Tuple[Dict[str, PortfolioPosition], Dict]:
        """执行组合优化"""
        if signals_df.empty:
            return {}, {}
        
        # 计算Kelly权重
        kelly_weights = self.calculate_kelly_optimal_weights(signals_df)
        
        # 相关性调整
        correlation_adjustments = self.check_correlation_constraints(signals_df)
        
        # 应用相关性调整
        adjusted_weights = {}
        for symbol, weight in kelly_weights.items():
            adjustment = correlation_adjustments.get(symbol, 1.0)
            adjusted_weights[symbol] = weight * adjustment
        
        # 标准化权重
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            # 确保总权重不超过最大杠杆
            if total_weight > self.config.max_total_leverage:
                scaling_factor = self.config.max_total_leverage / total_weight
                adjusted_weights = {k: v * scaling_factor for k, v in adjusted_weights.items()}
        
        # 创建仓位对象
        positions = {}
        for symbol, weight in adjusted_weights.items():
            if abs(weight) < self.config.min_kelly_weight:
                continue
                
            signal_info = signals_df[signals_df['symbol'] == symbol].iloc[0]
            
            position = PortfolioPosition(
                symbol=symbol,
                weight=weight,
                dollar_amount=weight * self.config.base_capital,
                entry_time=datetime.now(),
                signal_strength=signal_info['signal'],
                confidence=signal_info['confidence'],
                expected_return=signal_info['predicted_return'],
                kelly_weight=kelly_weights.get(symbol, 0),
                risk_contribution=0  # 稍后计算
            )
            positions[symbol] = position
        
        # 优化信息
        optimization_info = {
            'total_positions': len(positions),
            'gross_exposure': sum(abs(p.weight) for p in positions.values()),
            'net_exposure': sum(p.weight for p in positions.values()),
            'kelly_total': sum(kelly_weights.values()),
            'correlation_adjustments': len(correlation_adjustments),
            'optimization_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Portfolio optimized: {len(positions)} positions, "
                   f"gross exposure: {optimization_info['gross_exposure']:.3f}")
        
        return positions, optimization_info

    def calculate_real_time_risk_metrics(self, positions: Dict[str, PortfolioPosition]) -> Dict:
        """计算实时风险指标"""
        if not positions:
            return {}
        
        # 提取权重向量
        symbols = list(positions.keys())
        weights = np.array([positions[s].weight for s in symbols])
        
        # 模拟协方差矩阵（实际应使用历史收益率数据）
        np.random.seed(42)
        n_assets = len(symbols)
        random_matrix = np.random.randn(n_assets, n_assets)
        cov_matrix = (random_matrix @ random_matrix.T) / n_assets * 0.02  # 年化协方差
        
        # 组合风险指标
        portfolio_var = weights.T @ cov_matrix @ weights
        portfolio_vol = np.sqrt(portfolio_var)
        
        # VaR和ES计算（正态假设）
        daily_vol = portfolio_vol / np.sqrt(252)
        var_95 = 1.645 * daily_vol  # 95% VaR
        var_99 = 2.33 * daily_vol   # 99% VaR
        es_95 = var_95 * 1.28       # 95% Expected Shortfall
        
        # Beta计算（简化：假设市场beta = 1）
        portfolio_beta = np.sum(weights)
        
        # 多样化比率
        individual_vols = np.sqrt(np.diag(cov_matrix))
        diversification_ratio = portfolio_vol / np.sum(weights * individual_vols) if np.sum(weights * individual_vols) > 0 else 1
        
        risk_metrics = {
            'portfolio_volatility': portfolio_vol,
            'portfolio_beta': portfolio_beta,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'diversification_ratio': diversification_ratio,
            'max_position_weight': max(abs(w) for w in weights),
            'calculation_timestamp': datetime.now().isoformat()
        }
        
        return risk_metrics

    def check_risk_limits(self, positions: Dict[str, PortfolioPosition], risk_metrics: Dict) -> List[RiskAlert]:
        """检查风险限制并生成告警"""
        alerts = []
        current_time = datetime.now()
        
        # Beta风险检查
        if abs(risk_metrics.get('portfolio_beta', 0)) > self.config.max_portfolio_beta:
            alerts.append(RiskAlert(
                alert_type='BETA_VIOLATION',
                priority='HIGH',
                description=f"Portfolio beta {risk_metrics['portfolio_beta']:.3f} exceeds limit {self.config.max_portfolio_beta}",
                current_value=abs(risk_metrics['portfolio_beta']),
                threshold=self.config.max_portfolio_beta,
                timestamp=current_time
            ))
        
        # 波动率检查
        if risk_metrics.get('portfolio_volatility', 0) > self.config.max_portfolio_volatility:
            alerts.append(RiskAlert(
                alert_type='VOLATILITY_VIOLATION',
                priority='HIGH',
                description=f"Portfolio volatility {risk_metrics['portfolio_volatility']:.3f} exceeds limit {self.config.max_portfolio_volatility}",
                current_value=risk_metrics['portfolio_volatility'],
                threshold=self.config.max_portfolio_volatility,
                timestamp=current_time
            ))
        
        # VaR检查
        if risk_metrics.get('var_95', 0) > self.config.max_var_95:
            alerts.append(RiskAlert(
                alert_type='VAR_VIOLATION',
                priority='HIGH',
                description=f"VaR(95%) {risk_metrics['var_95']:.3f} exceeds limit {self.config.max_var_95}",
                current_value=risk_metrics['var_95'],
                threshold=self.config.max_var_95,
                timestamp=current_time
            ))
        
        # 单仓位检查
        max_position = risk_metrics.get('max_position_weight', 0)
        if max_position > self.config.max_single_position:
            alerts.append(RiskAlert(
                alert_type='POSITION_SIZE_VIOLATION',
                priority='MEDIUM',
                description=f"Max position size {max_position:.3f} exceeds limit {self.config.max_single_position}",
                current_value=max_position,
                threshold=self.config.max_single_position,
                timestamp=current_time
            ))
        
        # 持仓时间检查
        for symbol, position in positions.items():
            hold_time = (current_time - position.entry_time).total_seconds() / 60
            if hold_time > self.config.position_hold_time_limit:
                alerts.append(RiskAlert(
                    alert_type='HOLDING_TIME_VIOLATION',
                    priority='MEDIUM',
                    description=f"{symbol} held for {hold_time:.0f} minutes, exceeds limit {self.config.position_hold_time_limit}",
                    current_value=hold_time,
                    threshold=self.config.position_hold_time_limit,
                    timestamp=current_time
                ))
        
        # 杠杆检查
        total_leverage = sum(abs(p.weight) for p in positions.values())
        if total_leverage > self.config.max_total_leverage:
            alerts.append(RiskAlert(
                alert_type='LEVERAGE_VIOLATION',
                priority='HIGH',
                description=f"Total leverage {total_leverage:.3f} exceeds limit {self.config.max_total_leverage}",
                current_value=total_leverage,
                threshold=self.config.max_total_leverage,
                timestamp=current_time
            ))
        
        return alerts

    async def execute_rebalancing(self, new_positions: Dict[str, PortfolioPosition]) -> Dict:
        """执行组合再平衡"""
        rebalancing_info = {
            'timestamp': datetime.now().isoformat(),
            'old_positions': len(self.current_positions),
            'new_positions': len(new_positions),
            'position_changes': [],
            'turnover': 0.0
        }
        
        # 计算持仓变化
        all_symbols = set(list(self.current_positions.keys()) + list(new_positions.keys()))
        
        total_turnover = 0
        for symbol in all_symbols:
            old_weight = self.current_positions.get(symbol, PortfolioPosition('', 0, 0, datetime.now(), 0, 0, 0, 0, 0)).weight
            new_weight = new_positions.get(symbol, PortfolioPosition('', 0, 0, datetime.now(), 0, 0, 0, 0, 0)).weight
            
            weight_change = abs(new_weight - old_weight)
            total_turnover += weight_change
            
            if weight_change > 0.001:  # 只记录显著变化
                rebalancing_info['position_changes'].append({
                    'symbol': symbol,
                    'old_weight': old_weight,
                    'new_weight': new_weight,
                    'change': new_weight - old_weight
                })
        
        rebalancing_info['turnover'] = total_turnover
        
        # 更新当前持仓
        self.current_positions = new_positions.copy()
        
        logger.info(f"Rebalancing executed: turnover={total_turnover:.3f}, "
                   f"{len(rebalancing_info['position_changes'])} position changes")
        
        return rebalancing_info

    def save_portfolio_snapshot(self, positions: Dict[str, PortfolioPosition], 
                               risk_metrics: Dict, alerts: List[RiskAlert], 
                               rebalancing_info: Dict):
        """保存组合快照"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        snapshot = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'positions_count': len(positions),
                'base_capital': self.config.base_capital
            },
            'positions': [asdict(pos) for pos in positions.values()],
            'risk_metrics': risk_metrics,
            'alerts': [asdict(alert) for alert in alerts],
            'rebalancing_info': rebalancing_info,
            'configuration': asdict(self.config)
        }
        
        # 保存当前快照
        snapshot_file = f"{self.config.output_path}/portfolio_snapshot_{timestamp}.json"
        with open(snapshot_file, 'w') as f:
            json.dump(snapshot, f, indent=2, default=str)
        
        # 添加到历史记录
        self.portfolio_history.append(snapshot)
        
        # 只保留最近的100个快照
        if len(self.portfolio_history) > 100:
            self.portfolio_history = self.portfolio_history[-100:]
        
        logger.info(f"Portfolio snapshot saved: {snapshot_file}")

    async def continuous_optimization_cycle(self):
        """持续优化周期"""
        logger.info("Starting continuous optimization cycle...")
        
        try:
            # 1. 加载最新信号
            signals_df = await self.load_latest_signals()
            if signals_df.empty:
                logger.warning("No signals available for optimization")
                return
            
            # 2. 执行组合优化
            new_positions, optimization_info = await self.optimize_portfolio(signals_df)
            
            # 3. 计算风险指标
            risk_metrics = self.calculate_real_time_risk_metrics(new_positions)
            
            # 4. 检查风险限制
            alerts = self.check_risk_limits(new_positions, risk_metrics)
            
            # 5. 处理风险告警
            if alerts:
                for alert in alerts:
                    logger.warning(f"Risk Alert: {alert.alert_type} - {alert.description}")
                    
                    # 高优先级告警需要立即处理
                    if alert.priority == 'HIGH':
                        # 在这里可以实现自动风险缓解措施
                        pass
            
            # 6. 执行再平衡
            rebalancing_info = await self.execute_rebalancing(new_positions)
            
            # 7. 保存快照
            self.save_portfolio_snapshot(new_positions, risk_metrics, alerts, rebalancing_info)
            
            # 8. 更新告警列表
            self.risk_alerts.extend(alerts)
            
            # 只保留最近的50个告警
            if len(self.risk_alerts) > 50:
                self.risk_alerts = self.risk_alerts[-50:]
            
            logger.info(f"Optimization cycle completed successfully. "
                       f"Positions: {len(new_positions)}, Alerts: {len(alerts)}")
            
        except Exception as e:
            logger.error(f"Error in optimization cycle: {e}")
            import traceback
            traceback.print_exc()

    async def start_continuous_management(self):
        """启动持续管理"""
        logger.info("Starting Continuous Portfolio Risk Management...")
        self.is_running = True
        
        # 根据频率设置调度
        if self.config.rebalance_frequency == "hourly":
            schedule.every().hour.do(lambda: asyncio.create_task(self.continuous_optimization_cycle()))
        elif self.config.rebalance_frequency == "daily":
            schedule.every().day.at("09:00").do(lambda: asyncio.create_task(self.continuous_optimization_cycle()))
        elif self.config.rebalance_frequency == "weekly":
            schedule.every().monday.at("09:00").do(lambda: asyncio.create_task(self.continuous_optimization_cycle()))
        
        # 初始优化
        await self.continuous_optimization_cycle()
        
        # 持续运行
        try:
            while self.is_running:
                schedule.run_pending()
                await asyncio.sleep(60)  # 每分钟检查一次调度
        except KeyboardInterrupt:
            logger.info("Continuous management stopped by user")
        except Exception as e:
            logger.error(f"Error in continuous management: {e}")
        finally:
            self.is_running = False

    def stop_continuous_management(self):
        """停止持续管理"""
        logger.info("Stopping continuous portfolio risk management...")
        self.is_running = False

    def get_current_portfolio_summary(self) -> Dict:
        """获取当前组合摘要"""
        if not self.current_positions:
            return {'status': 'NO_POSITIONS'}
        
        total_value = sum(pos.dollar_amount for pos in self.current_positions.values())
        total_weight = sum(abs(pos.weight) for pos in self.current_positions.values())
        avg_confidence = np.mean([pos.confidence for pos in self.current_positions.values()])
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'positions_count': len(self.current_positions),
            'total_dollar_value': total_value,
            'total_weight': total_weight,
            'average_confidence': avg_confidence,
            'positions_details': [
                {
                    'symbol': pos.symbol,
                    'weight': pos.weight,
                    'dollar_amount': pos.dollar_amount,
                    'confidence': pos.confidence,
                    'expected_return': pos.expected_return,
                    'hold_time_minutes': (datetime.now() - pos.entry_time).total_seconds() / 60
                }
                for pos in self.current_positions.values()
            ],
            'recent_alerts_count': len([a for a in self.risk_alerts if (datetime.now() - a.timestamp).total_seconds() < 3600])
        }
        
        return summary

async def main():
    """主执行函数"""
    print("🚀 DipMaster Continuous Portfolio Risk Management System")
    print("=" * 70)
    
    # 配置系统
    config = ContinuousRiskConfig(
        base_capital=100000,
        rebalance_frequency="hourly",
        max_portfolio_beta=0.10,
        max_portfolio_volatility=0.18,
        max_single_position=0.20,
        max_total_leverage=3.0,
        max_var_95=0.03,
        kelly_fraction=0.25
    )
    
    # 创建管理器
    risk_manager = ContinuousPortfolioRiskManager(config)
    
    print(f"\n📊 System Configuration:")
    print(f"   Base Capital: ${config.base_capital:,.2f}")
    print(f"   Rebalance Frequency: {config.rebalance_frequency}")
    print(f"   Max Portfolio Beta: {config.max_portfolio_beta}")
    print(f"   Max Portfolio Volatility: {config.max_portfolio_volatility:.1%}")
    print(f"   Max Single Position: {config.max_single_position:.1%}")
    print(f"   Max Total Leverage: {config.max_total_leverage}x")
    print(f"   Max VaR (95%): {config.max_var_95:.1%}")
    
    # 执行一次优化周期作为演示
    print(f"\n🎯 Executing Initial Optimization Cycle...")
    await risk_manager.continuous_optimization_cycle()
    
    # 显示当前组合摘要
    summary = risk_manager.get_current_portfolio_summary()
    print(f"\n📈 Current Portfolio Summary:")
    print(f"   Positions: {summary.get('positions_count', 0)}")
    print(f"   Total Value: ${summary.get('total_dollar_value', 0):,.2f}")
    print(f"   Total Weight: {summary.get('total_weight', 0):.3f}")
    print(f"   Average Confidence: {summary.get('average_confidence', 0):.3f}")
    print(f"   Recent Alerts: {summary.get('recent_alerts_count', 0)}")
    
    if summary.get('positions_details'):
        print(f"\n   Top Positions:")
        for pos in sorted(summary['positions_details'], key=lambda x: abs(x['weight']), reverse=True)[:5]:
            print(f"     {pos['symbol']}: {pos['weight']:.3f} "
                  f"(${pos['dollar_amount']:,.0f}, conf: {pos['confidence']:.3f})")
    
    print(f"\n✅ Continuous Portfolio Risk Management System Ready!")
    print(f"💾 Results saved to: {config.output_path}")
    
    # 可选：启动持续管理（注释掉以避免无限运行）
    # await risk_manager.start_continuous_management()

if __name__ == "__main__":
    asyncio.run(main())