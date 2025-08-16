"""
DipMaster Enhanced V4 - Advanced Backtesting Framework
Implements rigorous backtesting with realistic cost modeling and risk management.
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import warnings

# Plotting and visualization
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Statistical libraries
from scipy import stats
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings('ignore')

@dataclass
class TradingCosts:
    """Trading cost configuration"""
    commission_rate: float = 0.0002  # 0.02% commission
    slippage_base: float = 0.0001    # 0.01% base slippage
    slippage_impact: float = 0.5     # Market impact factor
    funding_rate_8h: float = 0.0001  # 0.01% every 8 hours
    min_commission: float = 0.1      # Minimum commission USD

@dataclass
class RiskLimits:
    """Risk management limits"""
    max_position_size: float = 1000   # USD per position
    max_daily_positions: int = 10     # Maximum positions per day
    max_concurrent_positions: int = 3  # Maximum concurrent positions
    daily_loss_limit: float = -500    # Daily loss limit USD
    max_drawdown_limit: float = -0.05 # Maximum drawdown 5%
    position_correlation_limit: float = 0.7  # Max correlation between positions

@dataclass
class Position:
    """Individual trading position"""
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: float
    side: str  # 'long' or 'short'
    stop_loss: float
    take_profit: float
    entry_signal_strength: float
    position_id: str
    
    # Filled on exit
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    realized_pnl: Optional[float] = None
    commission_paid: Optional[float] = None
    funding_paid: Optional[float] = None
    slippage_paid: Optional[float] = None
    holding_minutes: Optional[int] = None

@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    avg_holding_time: float
    
    # Risk metrics
    var_95: float
    cvar_95: float
    volatility: float
    skewness: float
    kurtosis: float
    
    # Cost analysis
    total_commission: float
    total_slippage: float
    total_funding: float
    total_costs: float
    
    # Statistical significance
    t_statistic: float
    p_value: float
    information_ratio: float
    
    # Additional metrics
    positions: List[Position]
    equity_curve: pd.Series
    daily_returns: pd.Series
    drawdown_series: pd.Series

class AdvancedBacktester:
    """
    Advanced backtesting engine with realistic execution simulation
    """
    
    def __init__(self,
                 initial_capital: float = 10000,
                 costs: TradingCosts = None,
                 risk_limits: RiskLimits = None):
        
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.costs = costs or TradingCosts()
        self.risk_limits = risk_limits or RiskLimits()
        
        # State tracking
        self.positions = []
        self.open_positions = {}
        self.daily_pnl = {}
        self.equity_curve = []
        self.trade_log = []
        
        # Performance tracking
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('AdvancedBacktester')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calculate_slippage(self, price: float, volume: float, volatility: float) -> float:
        """
        Calculate market impact slippage based on volume and volatility
        """
        base_slippage = self.costs.slippage_base * price
        
        # Volume impact (simple square root model)
        volume_impact = np.sqrt(volume / 1000000) * 0.0001 * price
        
        # Volatility impact
        volatility_impact = volatility * 0.001 * price
        
        total_slippage = base_slippage + volume_impact + volatility_impact
        return min(total_slippage, 0.002 * price)  # Cap at 0.2%
    
    def calculate_funding_cost(self, position: Position, current_time: datetime) -> float:
        """
        Calculate funding costs for perpetual futures
        """
        holding_hours = (current_time - position.entry_time).total_seconds() / 3600
        funding_periods = int(holding_hours / 8)  # Every 8 hours
        
        if funding_periods > 0:
            position_value = abs(position.quantity * position.entry_price)
            funding_cost = funding_periods * self.costs.funding_rate_8h * position_value
            return funding_cost
        
        return 0.0
    
    def check_risk_limits(self, symbol: str, signal_strength: float, current_time: datetime) -> bool:
        """
        Check if new position would violate risk limits
        """
        # Check daily position limit
        today = current_time.date()
        today_positions = sum(1 for p in self.positions if p.entry_time.date() == today)
        if today_positions >= self.risk_limits.max_daily_positions:
            return False
        
        # Check concurrent position limit
        if len(self.open_positions) >= self.risk_limits.max_concurrent_positions:
            return False
        
        # Check daily loss limit
        daily_pnl = self.daily_pnl.get(today, 0.0)
        if daily_pnl <= self.risk_limits.daily_loss_limit:
            return False
        
        # Check drawdown limit
        current_drawdown = (self.current_capital - self.peak_equity) / self.peak_equity
        if current_drawdown <= self.risk_limits.max_drawdown_limit:
            return False
        
        # Check correlation with existing positions
        if len(self.open_positions) > 0:
            # Simplified correlation check - avoid same symbol
            if symbol in self.open_positions:
                return False
        
        return True
    
    def calculate_position_size(self, price: float, signal_strength: float, volatility: float) -> float:
        """
        Calculate position size based on risk and signal strength
        """
        # Base position size
        base_size = self.risk_limits.max_position_size
        
        # Scale by signal strength
        signal_scaled_size = base_size * signal_strength
        
        # Scale by inverse volatility (lower vol = larger size)
        vol_scaled_size = signal_scaled_size * min(2.0, 0.02 / max(volatility, 0.005))
        
        # Calculate quantity
        quantity = vol_scaled_size / price
        
        # Ensure we don't exceed capital limits
        max_quantity = (self.current_capital * 0.3) / price  # Max 30% of capital per position
        quantity = min(quantity, max_quantity)
        
        return max(quantity, 0.001)  # Minimum position size
    
    def open_position(self,
                     symbol: str,
                     entry_time: datetime,
                     entry_price: float,
                     signal_strength: float,
                     market_data: pd.Series) -> Optional[Position]:
        """
        Open a new trading position
        """
        # Check risk limits
        if not self.check_risk_limits(symbol, signal_strength, entry_time):
            return None
        
        # Calculate position size
        volatility = market_data.get('volatility_20', 0.02)
        volume = market_data.get('volume', 1000000)
        quantity = self.calculate_position_size(entry_price, signal_strength, volatility)
        
        # Calculate slippage
        slippage = self.calculate_slippage(entry_price, volume, volatility)
        actual_entry_price = entry_price + slippage
        
        # Calculate commission
        position_value = quantity * actual_entry_price
        commission = max(position_value * self.costs.commission_rate, self.costs.min_commission)
        
        # Set stop loss and take profit
        stop_loss = actual_entry_price * 0.996  # 0.4% stop loss
        take_profit = actual_entry_price * 1.008  # 0.8% take profit
        
        # Create position
        position = Position(
            symbol=symbol,
            entry_time=entry_time,
            entry_price=actual_entry_price,
            quantity=quantity,
            side='long',
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_signal_strength=signal_strength,
            position_id=f"{symbol}_{entry_time.strftime('%Y%m%d_%H%M%S')}",
            commission_paid=commission,
            slippage_paid=slippage
        )
        
        # Update capital
        self.current_capital -= commission
        
        # Track position
        self.open_positions[symbol] = position
        self.positions.append(position)
        
        self.logger.info(f"Opened position: {symbol} at {actual_entry_price:.4f}, qty: {quantity:.4f}")
        
        return position
    
    def close_position(self,
                      symbol: str,
                      exit_time: datetime,
                      exit_price: float,
                      exit_reason: str,
                      market_data: pd.Series) -> Optional[Position]:
        """
        Close an existing position
        """
        if symbol not in self.open_positions:
            return None
        
        position = self.open_positions[symbol]
        
        # Calculate slippage on exit
        volatility = market_data.get('volatility_20', 0.02)
        volume = market_data.get('volume', 1000000)
        exit_slippage = self.calculate_slippage(exit_price, volume, volatility)
        actual_exit_price = exit_price - exit_slippage  # Adverse slippage on exit
        
        # Calculate commission
        position_value = position.quantity * actual_exit_price
        exit_commission = max(position_value * self.costs.commission_rate, self.costs.min_commission)
        
        # Calculate funding costs
        funding_cost = self.calculate_funding_cost(position, exit_time)
        
        # Calculate P&L
        gross_pnl = position.quantity * (actual_exit_price - position.entry_price)
        net_pnl = gross_pnl - exit_commission - funding_cost
        
        # Update position
        position.exit_time = exit_time
        position.exit_price = actual_exit_price
        position.exit_reason = exit_reason
        position.realized_pnl = net_pnl
        position.commission_paid += exit_commission
        position.funding_paid = funding_cost
        position.slippage_paid += exit_slippage
        position.holding_minutes = int((exit_time - position.entry_time).total_seconds() / 60)
        
        # Update capital
        self.current_capital += position_value - exit_commission
        
        # Update daily P&L tracking
        today = exit_time.date()
        if today not in self.daily_pnl:
            self.daily_pnl[today] = 0.0
        self.daily_pnl[today] += net_pnl
        
        # Update peak equity and drawdown
        if self.current_capital > self.peak_equity:
            self.peak_equity = self.current_capital
        
        current_drawdown = (self.current_capital - self.peak_equity) / self.peak_equity
        if current_drawdown < self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Remove from open positions
        del self.open_positions[symbol]
        
        # Add to trade log
        self.trade_log.append({
            'symbol': symbol,
            'entry_time': position.entry_time,
            'exit_time': exit_time,
            'entry_price': position.entry_price,
            'exit_price': actual_exit_price,
            'quantity': position.quantity,
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'commission': position.commission_paid,
            'slippage': position.slippage_paid,
            'funding': funding_cost,
            'holding_minutes': position.holding_minutes,
            'exit_reason': exit_reason,
            'signal_strength': position.entry_signal_strength
        })
        
        self.logger.info(f"Closed position: {symbol} at {actual_exit_price:.4f}, P&L: {net_pnl:.2f}")
        
        return position
    
    def update_equity_curve(self, timestamp: datetime):
        """Update equity curve with current portfolio value"""
        # Calculate unrealized P&L from open positions
        unrealized_pnl = 0.0
        # Note: Would need current market prices to calculate unrealized P&L
        
        total_equity = self.current_capital + unrealized_pnl
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': total_equity,
            'realized_pnl': self.current_capital - self.initial_capital,
            'unrealized_pnl': unrealized_pnl
        })
    
    def run_backtest(self,
                    signals_df: pd.DataFrame,
                    market_data: pd.DataFrame) -> BacktestResult:
        """
        Run complete backtest with signals and market data
        """
        self.logger.info("Starting backtest execution")
        
        # Ensure data is sorted by time
        signals_df = signals_df.sort_values('timestamp')
        market_data = market_data.sort_index()
        
        # Track positions by time boundaries (15-minute intervals)
        position_check_times = {}
        
        for _, signal_row in signals_df.iterrows():
            timestamp = signal_row['timestamp']
            symbol = signal_row.get('symbol', 'BTCUSDT')  # Default symbol if not specified
            signal_strength = signal_row['confidence']
            
            # Get market data for this timestamp
            try:
                market_row = market_data.loc[timestamp]
                entry_price = market_row['close']
            except KeyError:
                # Use nearest available data point
                nearest_idx = market_data.index.get_indexer([timestamp], method='nearest')[0]
                market_row = market_data.iloc[nearest_idx]
                entry_price = market_row['close']
            
            # Check if we should open a position
            if signal_row['signal'] == 1 and signal_strength >= 0.6:
                position = self.open_position(
                    symbol=symbol,
                    entry_time=timestamp,
                    entry_price=entry_price,
                    signal_strength=signal_strength,
                    market_data=market_row
                )
                
                if position:
                    # Schedule position check at 15-minute boundary
                    next_boundary = self._get_next_15min_boundary(timestamp)
                    if next_boundary not in position_check_times:
                        position_check_times[next_boundary] = []
                    position_check_times[next_boundary].append(symbol)
            
            # Update equity curve
            self.update_equity_curve(timestamp)
        
        # Process position exits at 15-minute boundaries
        for check_time, symbols_to_check in sorted(position_check_times.items()):
            for symbol in symbols_to_check:
                if symbol in self.open_positions:
                    # Get market data for exit
                    try:
                        market_row = market_data.loc[check_time]
                        exit_price = market_row['close']
                    except KeyError:
                        nearest_idx = market_data.index.get_indexer([check_time], method='nearest')[0]
                        market_row = market_data.iloc[nearest_idx]
                        exit_price = market_row['close']
                    
                    # Check exit conditions
                    position = self.open_positions[symbol]
                    
                    # Time-based exit (15-minute boundary)
                    if (check_time - position.entry_time).total_seconds() >= 15 * 60:
                        self.close_position(
                            symbol=symbol,
                            exit_time=check_time,
                            exit_price=exit_price,
                            exit_reason='15min_boundary',
                            market_data=market_row
                        )
                    
                    # Stop loss / take profit
                    elif exit_price <= position.stop_loss:
                        self.close_position(
                            symbol=symbol,
                            exit_time=check_time,
                            exit_price=exit_price,
                            exit_reason='stop_loss',
                            market_data=market_row
                        )
                    
                    elif exit_price >= position.take_profit:
                        self.close_position(
                            symbol=symbol,
                            exit_time=check_time,
                            exit_price=exit_price,
                            exit_reason='take_profit',
                            market_data=market_row
                        )
        
        # Close any remaining open positions
        final_time = signals_df['timestamp'].max()
        for symbol in list(self.open_positions.keys()):
            try:
                market_row = market_data.loc[final_time]
                exit_price = market_row['close']
            except KeyError:
                exit_price = self.open_positions[symbol].entry_price
                market_row = pd.Series({'volatility_20': 0.02, 'volume': 1000000})
            
            self.close_position(
                symbol=symbol,
                exit_time=final_time,
                exit_price=exit_price,
                exit_reason='end_of_test',
                market_data=market_row
            )
        
        # Calculate final results
        return self._calculate_results()
    
    def _get_next_15min_boundary(self, timestamp: datetime) -> datetime:
        """Get next 15-minute boundary for position exit"""
        minutes = timestamp.minute
        next_boundary_minute = ((minutes // 15) + 1) * 15
        
        if next_boundary_minute >= 60:
            return timestamp.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        else:
            return timestamp.replace(minute=next_boundary_minute, second=0, microsecond=0)
    
    def _calculate_results(self) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        
        if not self.trade_log:
            # Return empty results if no trades
            return BacktestResult(
                total_return=0.0, annual_return=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                calmar_ratio=0.0, max_drawdown=0.0, win_rate=0.0, profit_factor=0.0,
                total_trades=0, winning_trades=0, losing_trades=0, avg_win=0.0, avg_loss=0.0,
                avg_holding_time=0.0, var_95=0.0, cvar_95=0.0, volatility=0.0,
                skewness=0.0, kurtosis=0.0, total_commission=0.0, total_slippage=0.0,
                total_funding=0.0, total_costs=0.0, t_statistic=0.0, p_value=1.0,
                information_ratio=0.0, positions=[], equity_curve=pd.Series(),
                daily_returns=pd.Series(), drawdown_series=pd.Series()
            )
        
        trades_df = pd.DataFrame(self.trade_log)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic performance metrics
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital
        
        # Calculate daily returns
        equity_df['date'] = equity_df['timestamp'].dt.date
        daily_equity = equity_df.groupby('date')['equity'].last()
        daily_returns = daily_equity.pct_change().dropna()
        
        # Annualized metrics
        trading_days = len(daily_returns)
        if trading_days > 0:
            annual_return = (1 + total_return) ** (252 / trading_days) - 1
            volatility = daily_returns.std() * np.sqrt(252)
            
            if volatility > 0:
                sharpe_ratio = (annual_return - 0.02) / volatility  # Assuming 2% risk-free rate
            else:
                sharpe_ratio = 0.0
        else:
            annual_return = 0.0
            volatility = 0.0
            sharpe_ratio = 0.0
        
        # Downside deviation for Sortino ratio
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 0:
            downside_deviation = negative_returns.std() * np.sqrt(252)
            sortino_ratio = (annual_return - 0.02) / downside_deviation if downside_deviation > 0 else 0.0
        else:
            sortino_ratio = sharpe_ratio
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(self.max_drawdown) if self.max_drawdown != 0 else 0.0
        
        # Trade statistics
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
        win_rate = winning_trades / len(trades_df) if len(trades_df) > 0 else 0.0
        
        wins = trades_df[trades_df['net_pnl'] > 0]['net_pnl']
        losses = trades_df[trades_df['net_pnl'] <= 0]['net_pnl']
        
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = losses.mean() if len(losses) > 0 else 0.0
        
        # Profit factor
        total_wins = wins.sum() if len(wins) > 0 else 0.0
        total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
        
        # Risk metrics
        if len(daily_returns) > 0:
            var_95 = np.percentile(daily_returns, 5)
            cvar_95 = daily_returns[daily_returns <= var_95].mean()
            skewness = stats.skew(daily_returns)
            kurtosis = stats.kurtosis(daily_returns)
        else:
            var_95 = cvar_95 = skewness = kurtosis = 0.0
        
        # Cost analysis
        total_commission = trades_df['commission'].sum()
        total_slippage = trades_df['slippage'].sum()
        total_funding = trades_df['funding'].sum()
        total_costs = total_commission + total_slippage + total_funding
        
        # Statistical significance
        if len(daily_returns) > 1:
            t_stat, p_value = stats.ttest_1samp(daily_returns, 0)
            information_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0.0
        else:
            t_stat = p_value = information_ratio = 0.0
        
        # Average holding time
        avg_holding_time = trades_df['holding_minutes'].mean() if len(trades_df) > 0 else 0.0
        
        # Create equity and drawdown series
        equity_series = pd.Series(
            [eq['equity'] for eq in self.equity_curve],
            index=[eq['timestamp'] for eq in self.equity_curve]
        )
        
        # Calculate drawdown series
        running_max = equity_series.expanding().max()
        drawdown_series = (equity_series - running_max) / running_max
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=self.max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades_df),
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_holding_time=avg_holding_time,
            var_95=var_95,
            cvar_95=cvar_95,
            volatility=volatility,
            skewness=skewness,
            kurtosis=kurtosis,
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_funding=total_funding,
            total_costs=total_costs,
            t_statistic=t_stat,
            p_value=p_value,
            information_ratio=information_ratio,
            positions=self.positions,
            equity_curve=equity_series,
            daily_returns=daily_returns,
            drawdown_series=drawdown_series
        )
    
    def generate_report(self, result: BacktestResult, output_path: str):
        """Generate comprehensive HTML backtest report"""
        
        # Create HTML report
        html_content = self._create_html_report(result)
        
        # Save HTML report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Backtest report saved to {output_path}")
    
    def _create_html_report(self, result: BacktestResult) -> str:
        """Create comprehensive HTML report"""
        
        # Generate charts
        equity_chart = self._create_equity_chart(result)
        drawdown_chart = self._create_drawdown_chart(result)
        returns_dist_chart = self._create_returns_distribution_chart(result)
        monthly_returns_chart = self._create_monthly_returns_chart(result)
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DipMaster Enhanced V4 - Backtest Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric-card {{ border: 1px solid #ddd; padding: 15px; border-radius: 5px; }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2e7d32; }}
                .metric-label {{ font-size: 14px; color: #666; }}
                .chart-container {{ margin: 30px 0; }}
                .summary {{ background: #f5f5f5; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>DipMaster Enhanced V4 - Backtest Report</h1>
            <div class="summary">
                <h2>Executive Summary</h2>
                <p><strong>Strategy Performance:</strong> The DipMaster Enhanced V4 strategy achieved a {result.total_return:.2%} total return 
                with a {result.win_rate:.1%} win rate over {result.total_trades} trades.</p>
                <p><strong>Risk-Adjusted Returns:</strong> Sharpe ratio of {result.sharpe_ratio:.2f} with maximum drawdown of {result.max_drawdown:.2%}.</p>
                <p><strong>Statistical Significance:</strong> T-statistic of {result.t_statistic:.2f} (p-value: {result.p_value:.4f})</p>
            </div>
            
            <h2>Key Performance Metrics</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{result.total_return:.2%}</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result.annual_return:.2%}</div>
                    <div class="metric-label">Annual Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result.sharpe_ratio:.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result.win_rate:.1%}</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result.max_drawdown:.2%}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result.profit_factor:.2f}</div>
                    <div class="metric-label">Profit Factor</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result.total_trades}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{result.avg_holding_time:.0f}m</div>
                    <div class="metric-label">Avg Holding Time</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>Equity Curve</h2>
                <div id="equity-chart">{equity_chart}</div>
            </div>
            
            <div class="chart-container">
                <h2>Drawdown Analysis</h2>
                <div id="drawdown-chart">{drawdown_chart}</div>
            </div>
            
            <div class="chart-container">
                <h2>Returns Distribution</h2>
                <div id="returns-dist-chart">{returns_dist_chart}</div>
            </div>
            
            <div class="chart-container">
                <h2>Monthly Returns Heatmap</h2>
                <div id="monthly-returns-chart">{monthly_returns_chart}</div>
            </div>
            
            <h2>Risk Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Description</th></tr>
                <tr><td>Volatility</td><td>{result.volatility:.2%}</td><td>Annualized volatility</td></tr>
                <tr><td>VaR (95%)</td><td>{result.var_95:.2%}</td><td>Value at Risk (95% confidence)</td></tr>
                <tr><td>CVaR (95%)</td><td>{result.cvar_95:.2%}</td><td>Conditional Value at Risk</td></tr>
                <tr><td>Skewness</td><td>{result.skewness:.2f}</td><td>Return distribution skewness</td></tr>
                <tr><td>Kurtosis</td><td>{result.kurtosis:.2f}</td><td>Return distribution kurtosis</td></tr>
                <tr><td>Sortino Ratio</td><td>{result.sortino_ratio:.2f}</td><td>Downside risk-adjusted return</td></tr>
                <tr><td>Calmar Ratio</td><td>{result.calmar_ratio:.2f}</td><td>Return/Max Drawdown ratio</td></tr>
            </table>
            
            <h2>Trading Statistics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{result.total_trades}</td></tr>
                <tr><td>Winning Trades</td><td>{result.winning_trades}</td></tr>
                <tr><td>Losing Trades</td><td>{result.losing_trades}</td></tr>
                <tr><td>Win Rate</td><td>{result.win_rate:.1%}</td></tr>
                <tr><td>Average Win</td><td>${result.avg_win:.2f}</td></tr>
                <tr><td>Average Loss</td><td>${result.avg_loss:.2f}</td></tr>
                <tr><td>Profit Factor</td><td>{result.profit_factor:.2f}</td></tr>
                <tr><td>Average Holding Time</td><td>{result.avg_holding_time:.0f} minutes</td></tr>
            </table>
            
            <h2>Cost Analysis</h2>
            <table>
                <tr><th>Cost Type</th><th>Total Cost</th><th>% of Capital</th></tr>
                <tr><td>Commission</td><td>${result.total_commission:.2f}</td><td>{result.total_commission/10000:.2%}</td></tr>
                <tr><td>Slippage</td><td>${result.total_slippage:.2f}</td><td>{result.total_slippage/10000:.2%}</td></tr>
                <tr><td>Funding</td><td>${result.total_funding:.2f}</td><td>{result.total_funding/10000:.2%}</td></tr>
                <tr><td>Total Costs</td><td>${result.total_costs:.2f}</td><td>{result.total_costs/10000:.2%}</td></tr>
            </table>
            
            <div class="summary">
                <h2>Strategy Assessment</h2>
                <p><strong>Performance Target Achievement:</strong></p>
                <ul>
                    <li>Win Rate Target (≥85%): {'✅ ACHIEVED' if result.win_rate >= 0.85 else '❌ NOT ACHIEVED'} ({result.win_rate:.1%})</li>
                    <li>Sharpe Ratio Target (≥2.0): {'✅ ACHIEVED' if result.sharpe_ratio >= 2.0 else '❌ NOT ACHIEVED'} ({result.sharpe_ratio:.2f})</li>
                    <li>Max Drawdown Target (≤3%): {'✅ ACHIEVED' if abs(result.max_drawdown) <= 0.03 else '❌ NOT ACHIEVED'} ({result.max_drawdown:.2%})</li>
                    <li>Profit Factor Target (≥1.8): {'✅ ACHIEVED' if result.profit_factor >= 1.8 else '❌ NOT ACHIEVED'} ({result.profit_factor:.2f})</li>
                </ul>
                
                <p><strong>Statistical Significance:</strong></p>
                <p>T-statistic: {result.t_statistic:.2f}, P-value: {result.p_value:.4f}</p>
                <p>{'✅ Statistically significant' if result.p_value < 0.05 else '❌ Not statistically significant'} at 95% confidence level</p>
            </div>
            
            <p><em>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
        </body>
        </html>
        """
        
        return html_template
    
    def _create_equity_chart(self, result: BacktestResult) -> str:
        """Create equity curve chart"""
        if result.equity_curve.empty:
            return "<p>No equity data available</p>"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.equity_curve.index,
            y=result.equity_curve.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Portfolio Equity Curve',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=400,
            showlegend=True
        )
        
        return fig.to_html(include_plotlyjs=False, div_id="equity-chart")
    
    def _create_drawdown_chart(self, result: BacktestResult) -> str:
        """Create drawdown chart"""
        if result.drawdown_series.empty:
            return "<p>No drawdown data available</p>"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=result.drawdown_series.index,
            y=result.drawdown_series.values * 100,
            mode='lines',
            name='Drawdown %',
            fill='tonexty',
            line=dict(color='red', width=1)
        ))
        
        fig.update_layout(
            title='Portfolio Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            height=300,
            showlegend=True
        )
        
        return fig.to_html(include_plotlyjs=False, div_id="drawdown-chart")
    
    def _create_returns_distribution_chart(self, result: BacktestResult) -> str:
        """Create returns distribution chart"""
        if result.daily_returns.empty:
            return "<p>No returns data available</p>"
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=result.daily_returns * 100,
            nbinsx=50,
            name='Daily Returns',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='Daily Returns Distribution',
            xaxis_title='Daily Return (%)',
            yaxis_title='Frequency',
            height=400
        )
        
        return fig.to_html(include_plotlyjs=False, div_id="returns-dist-chart")
    
    def _create_monthly_returns_chart(self, result: BacktestResult) -> str:
        """Create monthly returns heatmap"""
        if result.daily_returns.empty:
            return "<p>No returns data available</p>"
        
        # Calculate monthly returns
        monthly_returns = result.daily_returns.groupby([
            result.daily_returns.index.year,
            result.daily_returns.index.month
        ]).apply(lambda x: (1 + x).prod() - 1)
        
        if len(monthly_returns) == 0:
            return "<p>Insufficient data for monthly analysis</p>"
        
        # Create pivot table for heatmap
        monthly_df = monthly_returns.reset_index()
        monthly_df.columns = ['Year', 'Month', 'Return']
        
        pivot_table = monthly_df.pivot(index='Year', columns='Month', values='Return') * 100
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=[f'Month {i}' for i in pivot_table.columns],
            y=pivot_table.index,
            colorscale='RdYlGn',
            zmid=0
        ))
        
        fig.update_layout(
            title='Monthly Returns Heatmap (%)',
            height=400
        )
        
        return fig.to_html(include_plotlyjs=False, div_id="monthly-returns-chart")