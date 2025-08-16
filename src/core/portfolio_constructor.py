"""
DipMaster Enhanced V4 - 组合构建主程序
整合Alpha信号、组合优化、风险管理和仓位调整的完整解决方案

作者: DipMaster Trading System
版本: 4.0.0
创建时间: 2025-08-16
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio_optimizer import PortfolioOptimizer, create_portfolio_optimizer
from risk_manager import RealTimeRiskManager, create_risk_manager
from dynamic_position_sizer import DynamicPositionSizer, create_position_sizer

class PortfolioConstructor:
    """
    DipMaster V4 组合构建器
    
    核心流程:
    1. 读取Alpha信号和市场数据
    2. 动态仓位大小优化
    3. 多币种组合优化
    4. 实时风险管理
    5. 生成目标组合和风险报告
    """
    
    def __init__(self, config_path: str):
        """
        初始化组合构建器
        
        参数:
        - config_path: 策略配置文件路径
        """
        
        self.config_path = config_path
        self.config = self._load_config()
        
        # 初始化各个组件
        self.optimizer = create_portfolio_optimizer(config_path)
        self.risk_manager = create_risk_manager(self.config.get('risk_management', {}))
        self.position_sizer = create_position_sizer(self.config.get('position_sizing', {}))
        
        # 数据路径
        self.data_dir = self.config.get('data_directory', 'G:/Github/Quant/DipMaster-Trading-System')
        self.results_dir = os.path.join(self.data_dir, 'results', 'portfolio_construction')
        os.makedirs(self.results_dir, exist_ok=True)
        
    def construct_portfolio(self, 
                           alpha_signal_path: Optional[str] = None,
                           current_positions: Optional[Dict[str, float]] = None,
                           available_capital: float = 10000) -> Dict[str, Any]:
        """
        构建完整的目标组合
        
        参数:
        - alpha_signal_path: Alpha信号文件路径 (可选，使用最新)
        - current_positions: 当前仓位 {symbol: usd_value}
        - available_capital: 可用资金
        
        返回:
        - 完整的组合构建结果
        """
        
        print("Starting DipMaster V4 Portfolio Construction...")
        
        try:
            # 1. 加载Alpha信号
            alpha_signals = self._load_alpha_signals(alpha_signal_path)
            print(f"[OK] Loaded Alpha signals: {len(alpha_signals)} records")
            
            # 2. 加载市场数据
            market_data = self._load_market_data()
            print(f"[OK] Loaded market data: {len(market_data)} rows")
            
            # 3. 数据预处理和验证
            processed_data = self._preprocess_data(alpha_signals, market_data)
            print("[OK] Data preprocessing completed")
            
            # 4. 动态仓位大小计算
            position_sizes = self.position_sizer.calculate_optimal_sizes(
                alpha_signals=alpha_signals,
                market_data=market_data,
                current_positions=current_positions,
                available_capital=available_capital
            )
            print(f"[OK] Position sizing completed: {len(position_sizes)} positions")
            
            # 5. 组合优化
            portfolio_weights = self._convert_sizes_to_weights(position_sizes)
            target_portfolio = self.optimizer.optimize_portfolio(
                alpha_signals=alpha_signals,
                market_data=market_data,
                current_positions=current_positions
            )
            print("[OK] Portfolio optimization completed")
            
            # 6. 风险分析
            risk_report = self.risk_manager.generate_risk_report(
                portfolio_weights=portfolio_weights,
                market_data=market_data,
                returns_data=processed_data['returns']
            )
            print("[OK] Risk analysis completed")
            
            # 7. 整合最终结果
            final_result = self._integrate_results(
                target_portfolio=target_portfolio,
                position_sizes=position_sizes,
                risk_report=risk_report,
                alpha_signals=alpha_signals,
                available_capital=available_capital
            )
            
            # 8. 保存结果
            self._save_results(final_result)
            print("[OK] Results saved successfully")
            
            return final_result
            
        except Exception as e:
            print(f"[ERROR] Portfolio construction failed: {str(e)}")
            raise
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_alpha_signals(self, signal_path: Optional[str] = None) -> pd.DataFrame:
        """加载Alpha信号数据"""
        
        if signal_path is None:
            # 查找最新的Alpha信号文件
            ml_results_dir = os.path.join(self.data_dir, 'results', 'ml_pipeline')
            signal_files = [f for f in os.listdir(ml_results_dir) if f.startswith('AlphaSignal_') and f.endswith('.parquet')]
            
            if not signal_files:
                raise FileNotFoundError("未找到Alpha信号文件")
            
            # 选择最新的文件
            latest_signal_file = sorted(signal_files)[-1]
            signal_path = os.path.join(ml_results_dir, latest_signal_file)
        
        return pd.read_parquet(signal_path)
    
    def _load_market_data(self) -> pd.DataFrame:
        """加载市场数据"""
        
        market_data_dir = os.path.join(self.data_dir, 'data', 'market_data')
        
        # 获取配置中的交易对
        symbols = self.config.get('universe', ['BTCUSDT'])
        
        all_data = []
        
        for symbol in symbols:
            # 尝试多种文件格式
            file_patterns = [
                f"{symbol}_5m_2years.parquet",
                f"{symbol}_5m_2years.csv"
            ]
            
            for pattern in file_patterns:
                file_path = os.path.join(market_data_dir, pattern)
                if os.path.exists(file_path):
                    if pattern.endswith('.parquet'):
                        df = pd.read_parquet(file_path)
                    else:
                        df = pd.read_csv(file_path)
                    
                    # 标准化列名
                    df['symbol'] = symbol
                    if 'timestamp' not in df.columns and 'time' in df.columns:
                        df['timestamp'] = df['time']
                    
                    # 选择最近的数据 (最近1000条记录)
                    df = df.tail(1000)
                    all_data.append(df[['timestamp', 'symbol', 'close', 'volume']])
                    break
        
        if not all_data:
            raise FileNotFoundError("未找到任何市场数据文件")
        
        # 合并所有数据
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # 数据清理
        combined_data = combined_data.dropna()
        combined_data = combined_data.sort_values(['symbol', 'timestamp'])
        
        return combined_data
    
    def _preprocess_data(self, alpha_signals: pd.DataFrame, 
                        market_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """数据预处理"""
        
        # 确保timestamp为数值类型
        if 'timestamp' in market_data.columns:
            market_data['timestamp'] = pd.to_numeric(market_data['timestamp'], errors='coerce')
        
        # 去除重复的时间戳记录 (保留最后一个)
        market_data = market_data.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        
        # 确保有足够的数据
        if len(market_data) < 10:
            raise ValueError("Market data insufficient after deduplication")
        
        # 透视表格式转换
        try:
            prices = market_data.pivot(index='timestamp', columns='symbol', values='close')
            volumes = market_data.pivot(index='timestamp', columns='symbol', values='volume')
        except ValueError as e:
            print(f"Pivot error: {e}")
            # 回退方案：使用groupby
            prices = market_data.groupby(['timestamp', 'symbol'])['close'].last().unstack('symbol')
            volumes = market_data.groupby(['timestamp', 'symbol'])['volume'].last().unstack('symbol')
        
        # 计算收益率
        returns = prices.pct_change().dropna()
        
        # 确保时间索引排序 (数值排序)
        if len(returns) > 0:
            returns = returns.sort_index()
        if len(prices) > 0:
            prices = prices.sort_index()
        if len(volumes) > 0:
            volumes = volumes.sort_index()
        
        print(f"[DEBUG] Data shapes - Prices: {prices.shape}, Returns: {returns.shape}")
        
        return {
            'prices': prices,
            'volumes': volumes,
            'returns': returns
        }
    
    def _convert_sizes_to_weights(self, position_sizes: List) -> Dict[str, float]:
        """将仓位大小转换为权重"""
        
        weights = {}
        for position in position_sizes:
            if hasattr(position, 'symbol') and hasattr(position, 'final_size'):
                weights[position.symbol] = position.final_size
        
        return weights
    
    def _integrate_results(self, **kwargs) -> Dict[str, Any]:
        """整合所有结果"""
        
        target_portfolio = kwargs['target_portfolio']
        position_sizes = kwargs['position_sizes']
        risk_report = kwargs['risk_report']
        alpha_signals = kwargs['alpha_signals']
        available_capital = kwargs['available_capital']
        
        # 构建最终的目标组合
        final_weights = []
        position_details = []
        
        for position in position_sizes:
            if position.final_size > 0.001:  # 过滤掉极小仓位
                final_weights.append({
                    "symbol": position.symbol,
                    "w": float(position.final_size),
                    "usd_size": float(position.final_size * available_capital),
                    "kelly_fraction": float(position.kelly_fraction),
                    "confidence_adj": float(position.confidence_adjustment),
                    "volatility_adj": float(position.volatility_adjustment)
                })
                
                position_details.append({
                    "symbol": position.symbol,
                    "reasoning": position.reasoning,
                    "risk_budget": float(position.risk_budget_used),
                    "final_size_pct": float(position.final_size)
                })
        
        # 计算组合统计
        total_weight = sum(w["w"] for w in final_weights)
        total_usd = sum(w["usd_size"] for w in final_weights)
        
        # 信号统计
        latest_signals = alpha_signals.sort_values('timestamp').groupby('symbol').tail(1)
        signal_summary = {
            "total_signals": len(latest_signals),
            "avg_score": float(latest_signals['score'].mean()),
            "avg_confidence": float(latest_signals['confidence'].mean()),
            "signal_range": {
                "min_score": float(latest_signals['score'].min()),
                "max_score": float(latest_signals['score'].max())
            }
        }
        
        # 构建完整结果
        integrated_result = {
            "timestamp": datetime.now().isoformat(),
            "strategy_name": "DipMaster_Enhanced_V4",
            "construction_metadata": {
                "available_capital": available_capital,
                "total_weight": total_weight,
                "total_allocation": total_usd,
                "capital_utilization": total_usd / available_capital,
                "num_positions": len(final_weights)
            },
            "target_portfolio": {
                "ts": datetime.now().isoformat(),
                "weights": final_weights,
                "leverage": total_weight,
                "risk": target_portfolio.get('risk', {}),
                "venue_allocation": target_portfolio.get('venue_allocation', {"binance": 1.0}),
                "risk_attribution": target_portfolio.get('risk_attribution', {}),
                "constraints_status": target_portfolio.get('constraints_status', {})
            },
            "position_sizing_analysis": {
                "methodology": "Kelly + Volatility Targeting + Confidence Adjustment",
                "position_details": position_details,
                "sizing_summary": {
                    "largest_position": max((p.final_size for p in position_sizes), default=0),
                    "smallest_position": min((p.final_size for p in position_sizes if p.final_size > 0), default=0),
                    "avg_position_size": np.mean([p.final_size for p in position_sizes]),
                    "position_count": len([p for p in position_sizes if p.final_size > 0.001])
                }
            },
            "risk_analysis": risk_report,
            "signal_analysis": signal_summary,
            "performance_expectations": {
                "target_win_rate": "85%+",
                "target_sharpe": ">2.0",
                "max_drawdown_limit": "3%",
                "expected_monthly_return": "12-20%",
                "risk_adjusted_score": self._calculate_risk_score(risk_report)
            },
            "implementation_guidance": {
                "execution_order": self._generate_execution_order(final_weights),
                "risk_monitoring_alerts": self._extract_risk_alerts(risk_report),
                "rebalance_triggers": [
                    "信号置信度<0.7持续24小时",
                    "组合VaR>1.5%日度",
                    "最大回撤>2%",
                    "单币种权重>30%"
                ]
            }
        }
        
        return integrated_result
    
    def _calculate_risk_score(self, risk_report: Dict[str, Any]) -> float:
        """计算综合风险评分 (0-100)"""
        
        try:
            risk_metrics = risk_report.get('risk_metrics', {})
            
            # 各项风险指标评分
            var_score = max(0, 100 - (risk_metrics.get('VaR_95_daily', 0) / 0.02) * 50)  # VaR < 2%
            vol_score = max(0, 100 - (risk_metrics.get('annualized_volatility', 0) / 0.20) * 50)  # 波动率 < 20%
            beta_score = max(0, 100 - abs(risk_metrics.get('portfolio_beta', 0)) * 100)  # Beta接近0
            dd_score = max(0, 100 - (risk_metrics.get('max_drawdown', 0) / 0.03) * 50)  # 回撤 < 3%
            
            # 加权平均
            risk_score = (var_score * 0.3 + vol_score * 0.25 + beta_score * 0.25 + dd_score * 0.20)
            
            return min(100, max(0, risk_score))
            
        except Exception:
            return 50.0  # 默认中等风险
    
    def _generate_execution_order(self, weights: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """生成执行顺序建议"""
        
        # 按权重大小排序，大仓位优先
        sorted_weights = sorted(weights, key=lambda x: x['w'], reverse=True)
        
        execution_order = []
        for i, weight in enumerate(sorted_weights):
            execution_order.append({
                "order": i + 1,
                "symbol": weight['symbol'],
                "weight": weight['w'],
                "usd_size": weight['usd_size'],
                "priority": "HIGH" if weight['w'] > 0.15 else "MEDIUM" if weight['w'] > 0.05 else "LOW",
                "execution_method": "TWAP" if weight['usd_size'] > 1000 else "MARKET"
            })
        
        return execution_order
    
    def _extract_risk_alerts(self, risk_report: Dict[str, Any]) -> List[str]:
        """提取风险预警"""
        
        try:
            monitoring = risk_report.get('risk_monitoring', {})
            alerts = monitoring.get('alerts', [])
            
            return [alert.get('message', '') for alert in alerts]
            
        except Exception:
            return []
    
    def _save_results(self, result: Dict[str, Any]):
        """保存结果到文件"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存TargetPortfolio.json
        target_portfolio_path = os.path.join(self.results_dir, f"TargetPortfolio_{timestamp}.json")
        with open(target_portfolio_path, 'w', encoding='utf-8') as f:
            json.dump(result['target_portfolio'], f, indent=2, ensure_ascii=False)
        
        # 保存完整分析报告
        full_report_path = os.path.join(self.results_dir, f"PortfolioConstruction_Report_{timestamp}.json")
        with open(full_report_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # 保存风险报告
        risk_report_path = os.path.join(self.results_dir, f"RiskReport_{timestamp}.json")
        with open(risk_report_path, 'w', encoding='utf-8') as f:
            json.dump(result['risk_analysis'], f, indent=2, ensure_ascii=False)
        
        print(f"Files saved to:")
        print(f"   Target Portfolio: {target_portfolio_path}")
        print(f"   Full Report: {full_report_path}")
        print(f"   Risk Report: {risk_report_path}")


def main():
    """主函数 - 演示组合构建"""
    
    # 配置文件路径
    config_path = "G:/Github/Quant/DipMaster-Trading-System/config/dipmaster_enhanced_v4_spec.json"
    
    # 创建组合构建器
    constructor = PortfolioConstructor(config_path)
    
    # 模拟当前仓位 (可选)
    current_positions = {
        "BTCUSDT": 2000,  # $2000 in BTC
        "ETHUSDT": 1500   # $1500 in ETH
    }
    
    # 可用资金
    available_capital = 10000  # $10,000
    
    # 构建组合
    result = constructor.construct_portfolio(
        current_positions=current_positions,
        available_capital=available_capital
    )
    
    # 打印关键结果
    print("\n" + "="*60)
    print("DipMaster V4 Portfolio Construction Complete!")
    print("="*60)
    
    target_portfolio = result['target_portfolio']
    print(f"\nPortfolio Weights ({len(target_portfolio['weights'])} positions):")
    for weight in target_portfolio['weights']:
        print(f"   {weight['symbol']}: {weight['w']:.2%} (${weight['usd_size']:.0f})")
    
    print(f"\nRisk Metrics:")
    risk_metrics = target_portfolio.get('risk', {})
    print(f"   Annualized Volatility: {risk_metrics.get('ann_vol', 0):.1%}")
    print(f"   Portfolio Beta: {risk_metrics.get('beta', 0):.3f}")
    print(f"   Daily VaR(95%): {risk_metrics.get('VaR_95', 0):.2%}")
    print(f"   Sharpe Ratio: {risk_metrics.get('sharpe', 0):.2f}")
    
    construction_meta = result['construction_metadata']
    print(f"\nCapital Allocation:")
    print(f"   Total Allocated: ${construction_meta['total_allocation']:.0f}")
    print(f"   Capital Utilization: {construction_meta['capital_utilization']:.1%}")
    print(f"   Leverage: {construction_meta['total_weight']:.2f}x")
    
    risk_score = result['performance_expectations']['risk_adjusted_score']
    print(f"\nRisk Score: {risk_score:.0f}/100")
    
    print("\nPortfolio construction completed! Check generated JSON files for details.")


if __name__ == "__main__":
    main()