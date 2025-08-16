"""
DipMaster Enhanced V4 - Comprehensive Backtesting Engine
Implements realistic trading simulation with proper cost modeling and risk management
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class TradeRecord:
    """Individual trade record"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    entry_price: float
    exit_price: float
    position_size: float
    signal_strength: float
    profit_loss: float
    profit_loss_pct: float
    holding_minutes: int
    transaction_costs: float
    slippage_costs: float
    total_costs: float
    net_profit_loss: float
    win: bool

@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_holding_minutes: float
    total_costs: float
    cost_adjusted_return: float

class EnhancedBacktester:
    """
    Comprehensive backtesting engine for DipMaster Enhanced V4
    """
    
    def __init__(self, config: Dict = None):
        self.config = self._load_config(config)
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = None
        
    def _load_config(self, config: Dict) -> Dict:
        """Load backtesting configuration"""
        default_config = {
            'initial_capital': 10000,
            'position_sizing': {
                'method': 'fixed_amount',  # 'fixed_amount', 'percent_capital', 'kelly'
                'amount': 1000,  # USD per trade
                'max_positions': 3,
                'max_position_pct': 0.33
            },
            'transaction_costs': {
                'commission_rate': 0.001,  # 0.1% per trade
                'slippage_model': 'adaptive',  # 'fixed', 'adaptive'
                'fixed_slippage': 0.0005,  # 0.05%
                'market_impact_factor': 0.0001
            },
            'risk_management': {
                'max_daily_loss': 0.05,  # 5% of capital
                'max_portfolio_heat': 0.15,  # 15% of capital at risk
                'stop_loss': None,  # No stop loss (time-based exits only)
                'max_holding_minutes': 180
            },
            'signal_filtering': {
                'min_signal_strength': 0.0,
                'confidence_threshold': 0.5
            },
            'timing': {
                'entry_delay_minutes': 1,  # 1 bar delay for signal processing
                'exit_boundaries': [15, 30, 45, 60],  # 15-minute boundaries
                'preferred_exit_windows': [(15, 29), (45, 59)]
            }
        }
        
        if config:
            default_config.update(config)
            
        return default_config
    
    def calculate_slippage(self, price: float, volume: float, position_size: float) -> float:
        """Calculate realistic slippage based on market conditions"""
        if self.config['transaction_costs']['slippage_model'] == 'fixed':
            return self.config['transaction_costs']['fixed_slippage']
        
        # Adaptive slippage model
        base_slippage = self.config['transaction_costs']['fixed_slippage']
        market_impact = self.config['transaction_costs']['market_impact_factor']
        
        # Volume-based adjustment
        volume_factor = min(position_size / (volume * price), 0.01)  # Cap at 1%
        
        # Price volatility adjustment (simplified)
        volatility_factor = 0.0001  # Would be calculated from price data in practice
        
        total_slippage = base_slippage + (market_impact * volume_factor) + volatility_factor
        return min(total_slippage, 0.01)  # Cap at 1%
    
    def calculate_position_size(self, signal_strength: float, current_capital: float, 
                              price: float, existing_positions: int) -> float:
        """Calculate position size based on configuration and risk management"""
        method = self.config['position_sizing']['method']
        
        if existing_positions >= self.config['position_sizing']['max_positions']:
            return 0
        
        if method == 'fixed_amount':
            base_amount = self.config['position_sizing']['amount']
        elif method == 'percent_capital':
            pct = self.config['position_sizing']['max_position_pct']
            base_amount = current_capital * pct
        elif method == 'kelly':
            # Simplified Kelly criterion
            win_rate = 0.67  # From historical analysis
            avg_win = 0.008  # 0.8%
            avg_loss = 0.004  # 0.4%
            kelly_f = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            base_amount = current_capital * max(0, min(kelly_f * signal_strength, 0.25))
        
        # Adjust for signal strength
        adjusted_amount = base_amount * (0.5 + 0.5 * signal_strength)
        
        # Convert to shares
        shares = adjusted_amount / price
        
        return shares
    
    def find_exit_time(self, entry_time: datetime, data: pd.DataFrame) -> Tuple[datetime, float]:
        """Find optimal exit time based on 15-minute boundary rules"""
        max_holding = self.config['risk_management']['max_holding_minutes']
        boundaries = self.config['timing']['exit_boundaries']
        preferred_windows = self.config['timing']['preferred_exit_windows']
        
        # Get available data after entry
        future_data = data[data.index > entry_time].copy()
        if len(future_data) == 0:
            return entry_time, 0.0
        
        entry_minute = entry_time.minute
        
        # Find next preferred exit windows
        exit_candidates = []
        
        for i, row in future_data.iterrows():
            minutes_held = (i - entry_time).total_seconds() / 60
            current_minute = i.minute
            
            # Check if we're in a 15-minute boundary
            if current_minute in boundaries:
                # Check if in preferred window
                in_preferred = any(start <= current_minute <= end for start, end in preferred_windows)
                priority = 1 if in_preferred else 2
                
                exit_candidates.append({
                    'time': i,
                    'price': row['close'],
                    'minutes_held': minutes_held,
                    'priority': priority
                })
            
            # Force exit at max holding time
            if minutes_held >= max_holding:
                exit_candidates.append({
                    'time': i,
                    'price': row['close'],
                    'minutes_held': minutes_held,
                    'priority': 3  # Forced exit
                })
                break
        
        if not exit_candidates:
            # No exit found, use last available data
            last_idx = future_data.index[-1]
            last_price = future_data.iloc[-1]['close']
            return last_idx, last_price
        
        # Select best exit (prefer early exits in preferred windows)
        best_exit = min(exit_candidates, key=lambda x: (x['priority'], x['minutes_held']))
        return best_exit['time'], best_exit['price']
    
    def run_backtest(self, data: pd.DataFrame, signals: pd.Series, 
                    prices: pd.DataFrame = None) -> Dict:
        """Run comprehensive backtest"""
        print("Running enhanced backtest...")
        
        # Initialize
        if prices is None:
            # Assume data contains OHLCV
            prices = data[['open', 'high', 'low', 'close', 'volume']]
        
        current_capital = self.config['initial_capital']
        positions = {}  # symbol -> position info
        daily_pnl = []
        
        # Get unique symbols
        symbols = data['symbol'].unique() if 'symbol' in data.columns else ['DEFAULT']
        
        print(f"Backtesting {len(symbols)} symbols...")
        print(f"Signal coverage: {(signals > 0).sum()} / {len(signals)} ({(signals > 0).mean()*100:.1f}%)")
        
        # Process each signal
        signal_count = 0
        for timestamp, signal_strength in signals.items():
            if signal_strength <= self.config['signal_filtering']['min_signal_strength']:
                continue
                
            signal_count += 1
            if signal_count % 1000 == 0:
                print(f"  Processed {signal_count} signals...")
            
            # Get market data at signal time
            try:
                current_data = data.loc[timestamp]
                if isinstance(current_data, pd.Series):
                    symbol = current_data.get('symbol', 'DEFAULT')
                    current_price = current_data.get('close', current_data.get('price', 0))
                    current_volume = current_data.get('volume', 0)
                else:
                    # Multiple symbols at same timestamp
                    continue
            except (KeyError, IndexError):
                continue
            
            if current_price <= 0:
                continue
            
            # Check position limits
            active_positions = len([p for p in positions.values() if p['active']])
            if active_positions >= self.config['position_sizing']['max_positions']:
                continue
            
            # Calculate position size
            position_size = self.calculate_position_size(
                signal_strength, current_capital, current_price, active_positions
            )
            
            if position_size <= 0:
                continue
            
            # Entry execution with delay
            entry_delay = self.config['timing']['entry_delay_minutes']
            entry_time = timestamp + timedelta(minutes=entry_delay)
            
            # Get entry price (next available price)
            try:
                future_data = data[data.index >= entry_time]
                if len(future_data) == 0:
                    continue
                    
                entry_data = future_data.iloc[0]
                entry_price = entry_data.get('open', entry_data.get('close', current_price))
                entry_symbol = entry_data.get('symbol', symbol)
            except (KeyError, IndexError):
                continue
            
            # Calculate transaction costs
            commission = position_size * entry_price * self.config['transaction_costs']['commission_rate']
            slippage_rate = self.calculate_slippage(entry_price, current_volume, position_size * entry_price)
            slippage = position_size * entry_price * slippage_rate
            
            # Find exit time and price
            symbol_data = data[data['symbol'] == entry_symbol] if 'symbol' in data.columns else data
            exit_time, exit_price = self.find_exit_time(entry_time, symbol_data)
            
            if exit_price <= 0:
                continue
            
            # Calculate holding time
            holding_minutes = (exit_time - entry_time).total_seconds() / 60
            
            # Calculate P&L
            gross_pnl = (exit_price - entry_price) * position_size
            gross_pnl_pct = (exit_price - entry_price) / entry_price
            
            # Exit costs
            exit_commission = position_size * exit_price * self.config['transaction_costs']['commission_rate']
            exit_slippage = position_size * exit_price * slippage_rate
            
            total_costs = commission + slippage + exit_commission + exit_slippage
            net_pnl = gross_pnl - total_costs
            
            # Update capital
            current_capital += net_pnl
            
            # Create trade record
            trade = TradeRecord(
                entry_time=entry_time,
                exit_time=exit_time,
                symbol=entry_symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position_size,
                signal_strength=signal_strength,
                profit_loss=gross_pnl,
                profit_loss_pct=gross_pnl_pct,
                holding_minutes=holding_minutes,
                transaction_costs=commission + exit_commission,
                slippage_costs=slippage + exit_slippage,
                total_costs=total_costs,
                net_profit_loss=net_pnl,
                win=net_pnl > 0
            )
            
            self.trades.append(trade)
            
            # Track equity curve
            self.equity_curve.append({
                'timestamp': exit_time,
                'capital': current_capital,
                'trade_pnl': net_pnl,
                'cumulative_return': (current_capital / self.config['initial_capital'] - 1)
            })
        
        print(f"Backtest completed: {len(self.trades)} trades executed")
        
        # Calculate performance metrics
        self.performance_metrics = self.calculate_performance_metrics()
        
        return {
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'performance_metrics': self.performance_metrics,
            'final_capital': current_capital
        }
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return None
        
        # Basic trade statistics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.win)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L statistics
        total_gross_pnl = sum(t.profit_loss for t in self.trades)
        total_net_pnl = sum(t.net_profit_loss for t in self.trades)
        total_costs = sum(t.total_costs for t in self.trades)
        
        final_capital = self.config['initial_capital'] + total_net_pnl
        total_return_pct = total_net_pnl / self.config['initial_capital']
        
        # Win/Loss analysis
        wins = [t.net_profit_loss for t in self.trades if t.win]
        losses = [t.net_profit_loss for t in self.trades if not t.win]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')
        
        # Risk metrics
        if self.equity_curve:
            equity_series = pd.Series([e['capital'] for e in self.equity_curve])
            returns = equity_series.pct_change().dropna()
            
            # Sharpe ratio (assuming daily returns, 252 trading days)
            if len(returns) > 1 and returns.std() > 0:
                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            else:
                sharpe_ratio = 0
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 1 and negative_returns.std() > 0:
                sortino_ratio = (returns.mean() * 252) / (negative_returns.std() * np.sqrt(252))
            else:
                sortino_ratio = 0
            
            # Maximum drawdown
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown = abs(drawdown.min())
            max_drawdown_pct = max_drawdown
            
            # Calmar ratio
            calmar_ratio = total_return_pct / max_drawdown if max_drawdown > 0 else 0
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0
            max_drawdown = max_drawdown_pct = 0
        
        # Timing analysis
        avg_holding_minutes = np.mean([t.holding_minutes for t in self.trades])
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_net_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_holding_minutes=avg_holding_minutes,
            total_costs=total_costs,
            cost_adjusted_return=total_return_pct
        )
    
    def analyze_performance_by_regime(self, data: pd.DataFrame) -> Dict:
        """Analyze performance across different market regimes"""
        if not self.trades:
            return {}
        
        # Define regimes based on volatility and trend
        regime_analysis = {}
        
        # Volatility regimes
        if 'volatility' in data.columns:
            vol_median = data['volatility'].median()
            high_vol_trades = [t for t in self.trades if 
                             data.loc[t.entry_time:t.exit_time]['volatility'].mean() > vol_median]
            low_vol_trades = [t for t in self.trades if 
                            data.loc[t.entry_time:t.exit_time]['volatility'].mean() <= vol_median]
            
            regime_analysis['volatility'] = {
                'high_vol': {
                    'trades': len(high_vol_trades),
                    'win_rate': sum(1 for t in high_vol_trades if t.win) / len(high_vol_trades) if high_vol_trades else 0,
                    'avg_return': np.mean([t.net_profit_loss for t in high_vol_trades]) if high_vol_trades else 0
                },
                'low_vol': {
                    'trades': len(low_vol_trades),
                    'win_rate': sum(1 for t in low_vol_trades if t.win) / len(low_vol_trades) if low_vol_trades else 0,
                    'avg_return': np.mean([t.net_profit_loss for t in low_vol_trades]) if low_vol_trades else 0
                }
            }
        
        # Time-of-day analysis
        hour_performance = {}
        for trade in self.trades:
            hour = trade.entry_time.hour
            if hour not in hour_performance:
                hour_performance[hour] = []
            hour_performance[hour].append(trade)
        
        hourly_stats = {}
        for hour, trades in hour_performance.items():
            hourly_stats[hour] = {
                'trades': len(trades),
                'win_rate': sum(1 for t in trades if t.win) / len(trades),
                'avg_return': np.mean([t.net_profit_loss for t in trades])
            }
        
        regime_analysis['hourly'] = hourly_stats
        
        return regime_analysis
    
    def generate_detailed_report(self) -> str:
        """Generate comprehensive backtest report"""
        if not self.performance_metrics:
            return "No backtest results available"
        
        metrics = self.performance_metrics
        
        report = []
        report.append("=" * 80)
        report.append("DIPMASTER ENHANCED V4 - COMPREHENSIVE BACKTEST REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Initial Capital: ${self.config['initial_capital']:,.2f}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Trades: {metrics.total_trades}")
        report.append(f"Win Rate: {metrics.win_rate:.4f} ({metrics.win_rate*100:.2f}%)")
        report.append(f"Total Return: ${metrics.total_return:,.2f} ({metrics.total_return_pct*100:.2f}%)")
        report.append(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        report.append(f"Max Drawdown: {metrics.max_drawdown_pct*100:.2f}%")
        report.append(f"Profit Factor: {metrics.profit_factor:.3f}")
        report.append("")
        
        # Target Achievement
        target_win_rate = 0.85
        target_sharpe = 2.0
        target_max_dd = 0.03
        target_profit_factor = 1.8
        
        report.append("TARGET ACHIEVEMENT")
        report.append("-" * 40)
        report.append(f"Win Rate Target: {target_win_rate*100:.0f}% | Achieved: {metrics.win_rate*100:.1f}% | {'✅' if metrics.win_rate >= target_win_rate else '❌'}")
        report.append(f"Sharpe Target: {target_sharpe:.1f} | Achieved: {metrics.sharpe_ratio:.2f} | {'✅' if metrics.sharpe_ratio >= target_sharpe else '❌'}")
        report.append(f"Max DD Target: <{target_max_dd*100:.0f}% | Achieved: {metrics.max_drawdown_pct*100:.1f}% | {'✅' if metrics.max_drawdown_pct <= target_max_dd else '❌'}")
        report.append(f"Profit Factor Target: {target_profit_factor:.1f} | Achieved: {metrics.profit_factor:.2f} | {'✅' if metrics.profit_factor >= target_profit_factor else '❌'}")
        report.append("")
        
        # Detailed Metrics
        report.append("DETAILED PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append(f"Winning Trades: {metrics.winning_trades}")
        report.append(f"Losing Trades: {metrics.losing_trades}")
        report.append(f"Average Win: ${metrics.avg_win:.2f}")
        report.append(f"Average Loss: ${metrics.avg_loss:.2f}")
        report.append(f"Average Holding Time: {metrics.avg_holding_minutes:.1f} minutes")
        report.append(f"Sortino Ratio: {metrics.sortino_ratio:.3f}")
        report.append(f"Calmar Ratio: {metrics.calmar_ratio:.3f}")
        report.append(f"Total Transaction Costs: ${metrics.total_costs:.2f}")
        report.append("")
        
        # Risk Analysis
        report.append("RISK ANALYSIS")
        report.append("-" * 40)
        consecutive_losses = 0
        max_consecutive_losses = 0
        current_streak = 0
        
        for trade in self.trades:
            if trade.win:
                current_streak = 0
            else:
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
        
        report.append(f"Maximum Consecutive Losses: {max_consecutive_losses}")
        report.append(f"Largest Single Loss: ${min([t.net_profit_loss for t in self.trades]):.2f}")
        report.append(f"Cost Impact: {(metrics.total_costs / abs(metrics.total_return))*100:.1f}% of gross P&L")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS FOR 85%+ WIN RATE")
        report.append("-" * 40)
        
        gap_to_target = (target_win_rate - metrics.win_rate) * 100
        if gap_to_target > 0:
            report.append(f"Gap to 85% target: {gap_to_target:.1f} percentage points")
            report.append("")
            report.append("Suggested improvements:")
            report.append("1. Implement regime-aware signal filtering")
            report.append("2. Increase minimum signal confidence threshold")
            report.append("3. Add multi-timeframe confirmation")
            report.append("4. Optimize exit timing based on market microstructure")
            report.append("5. Implement dynamic position sizing based on signal quality")
        else:
            report.append("🎉 TARGET ACHIEVED! Consider increasing target or reducing risk.")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def plot_results(self, save_path: str = None):
        """Generate comprehensive performance plots"""
        if not self.trades or not self.equity_curve:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DipMaster Enhanced V4 - Backtest Results', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        equity_df = pd.DataFrame(self.equity_curve)
        axes[0, 0].plot(equity_df['timestamp'], equity_df['capital'])
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Capital ($)')
        axes[0, 0].grid(True)
        
        # 2. Trade P&L Distribution
        pnl_data = [t.net_profit_loss for t in self.trades]
        axes[0, 1].hist(pnl_data, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Trade P&L Distribution')
        axes[0, 1].set_xlabel('P&L ($)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Cumulative Returns
        axes[1, 0].plot(equity_df['timestamp'], equity_df['cumulative_return'] * 100)
        axes[1, 0].set_title('Cumulative Returns')
        axes[1, 0].set_ylabel('Return (%)')
        axes[1, 0].grid(True)
        
        # 4. Rolling Win Rate
        window_size = max(50, len(self.trades) // 20)
        rolling_wins = pd.Series([t.win for t in self.trades]).rolling(window_size).mean()
        axes[1, 1].plot(rolling_wins * 100)
        axes[1, 1].axhline(85, color='red', linestyle='--', alpha=0.7, label='Target 85%')
        axes[1, 1].axhline(rolling_wins.iloc[-1] * 100, color='green', linestyle='-', alpha=0.7, 
                          label=f'Current: {rolling_wins.iloc[-1]*100:.1f}%')
        axes[1, 1].set_title(f'Rolling Win Rate ({window_size} trades)')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plots saved to: {save_path}")
        
        return fig


def main():
    """Main backtesting execution"""
    print("DipMaster Enhanced V4 - Comprehensive Backtesting")
    print("=" * 60)
    
    # This would be called with actual model predictions
    # For now, this is a template
    pass


if __name__ == "__main__":
    main()