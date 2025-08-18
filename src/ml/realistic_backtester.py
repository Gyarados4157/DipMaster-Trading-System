"""
Realistic Backtesting Engine with Comprehensive Cost Modeling
Implements rigorous backtesting with realistic trading costs and constraints.
"""

import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, NamedTuple
from dataclasses import dataclass
import logging
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

warnings.filterwarnings('ignore')

@dataclass
class TradingCosts:
    """Trading cost configuration"""
    maker_fee: float = 0.0002  # 0.02% maker fee
    taker_fee: float = 0.0004  # 0.04% taker fee
    slippage_base: float = 0.0001  # 0.01% base slippage
    slippage_impact: float = 0.00005  # Price impact coefficient
    funding_rate_8h: float = 0.0001  # 0.01% every 8 hours
    withdrawal_fee: float = 0.0  # For spot trading
    
@dataclass
class RiskLimits:
    """Risk management limits"""
    max_position_size: float = 1000.0  # USD
    max_leverage: float = 1.0  # No leverage for spot
    max_daily_trades: int = 50
    max_concurrent_positions: int = 3
    daily_loss_limit: float = -500.0  # USD
    max_drawdown_limit: float = -0.10  # 10%
    position_timeout_minutes: int = 180  # 3 hours max hold

@dataclass  
class Position:
    """Trading position"""
    symbol: str
    entry_time: datetime
    entry_price: float
    size_usd: float
    direction: int  # 1 for long, -1 for short
    entry_fee: float
    funding_paid: float = 0.0
    is_active: bool = True
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_fee: float = 0.0
    pnl: float = 0.0
    
class BacktestResult(NamedTuple):
    """Backtest results container"""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    total_trades: int
    total_fees: float
    total_slippage: float
    total_funding: float
    final_capital: float
    equity_curve: pd.Series
    trade_history: pd.DataFrame
    monthly_returns: pd.Series
    statistical_significance: Dict[str, float]

class RealisticBacktester:
    """
    Comprehensive backtesting engine with realistic trading simulation
    """
    
    def __init__(self,
                 initial_capital: float = 10000.0,
                 costs: TradingCosts = None,
                 risk_limits: RiskLimits = None,
                 data_frequency_minutes: int = 5):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting capital in USD
            costs: Trading cost configuration
            risk_limits: Risk management limits
            data_frequency_minutes: Data frequency in minutes
        """
        
        self.initial_capital = initial_capital
        self.costs = costs or TradingCosts()
        self.risk_limits = risk_limits or RiskLimits()
        self.data_frequency_minutes = data_frequency_minutes
        
        # State variables
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.equity_curve: List[float] = [initial_capital]
        self.trade_log: List[Dict] = []
        self.daily_pnl: Dict[str, float] = {}
        
        # Performance tracking
        self.total_fees_paid = 0.0
        self.total_slippage_paid = 0.0
        self.total_funding_paid = 0.0
        
        # Setup logging
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('RealisticBacktester')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def _calculate_slippage(self, price: float, volume: float, volatility: float = 0.02) -> float:
        """
        Calculate realistic slippage based on market conditions
        
        Args:
            price: Market price
            volume: Trade volume in USD
            volatility: Current market volatility
            
        Returns:
            Slippage amount in USD
        """
        # Base slippage
        base_slippage = self.costs.slippage_base * volume
        
        # Volume impact (larger trades have higher slippage)
        volume_impact = self.costs.slippage_impact * (volume / 1000) ** 0.5
        
        # Volatility impact (higher volatility = higher slippage)
        volatility_impact = volatility * 0.1 * volume
        
        total_slippage = base_slippage + volume_impact + volatility_impact
        
        return total_slippage
    
    def _calculate_funding_cost(self, position: Position, current_time: datetime) -> float:
        """
        Calculate funding cost for position (8-hour cycles)
        
        Args:
            position: Trading position
            current_time: Current timestamp
            
        Returns:
            Funding cost in USD
        """
        if not position.is_active:
            return 0.0
        
        # Calculate 8-hour periods since position opened
        time_held = current_time - position.entry_time
        funding_periods = int(time_held.total_seconds() / (8 * 3600))
        
        # Calculate funding for new periods only
        already_paid_periods = int(position.funding_paid / (self.costs.funding_rate_8h * position.size_usd))
        new_funding_periods = max(0, funding_periods - already_paid_periods)
        
        funding_cost = new_funding_periods * self.costs.funding_rate_8h * position.size_usd
        
        return funding_cost
    
    def _check_risk_limits(self, current_time: datetime) -> bool:
        """Check if risk limits are violated"""
        
        # Daily loss limit
        current_date = current_time.date()
        if current_date in self.daily_pnl:
            if self.daily_pnl[current_date] <= self.risk_limits.daily_loss_limit:
                self.logger.warning(f"Daily loss limit hit: {self.daily_pnl[current_date]}")
                return False
        
        # Maximum drawdown
        if len(self.equity_curve) > 1:
            peak = max(self.equity_curve)
            current_equity = self.equity_curve[-1]
            drawdown = (current_equity - peak) / peak
            
            if drawdown <= self.risk_limits.max_drawdown_limit:
                self.logger.warning(f"Max drawdown limit hit: {drawdown:.2%}")
                return False
        
        # Position timeout
        positions_to_close = []
        for symbol, position in self.positions.items():
            time_held = current_time - position.entry_time
            if time_held.total_seconds() > self.risk_limits.position_timeout_minutes * 60:
                positions_to_close.append(symbol)
        
        return True
    
    def _open_position(self, 
                      symbol: str,
                      timestamp: datetime,
                      price: float,
                      size_usd: float,
                      direction: int,
                      volatility: float = 0.02,
                      confidence: float = 0.5) -> bool:
        """
        Open a new trading position
        
        Args:
            symbol: Trading symbol
            timestamp: Entry timestamp
            price: Entry price
            size_usd: Position size in USD
            direction: 1 for long, -1 for short
            volatility: Current market volatility
            confidence: Signal confidence (affects position sizing)
            
        Returns:
            True if position opened successfully
        """
        
        # Risk checks
        if not self._check_risk_limits(timestamp):
            return False
        
        # Position count limit
        if len(self.positions) >= self.risk_limits.max_concurrent_positions:
            self.logger.debug(f"Max concurrent positions limit reached")
            return False
        
        # Adjust position size based on confidence
        adjusted_size = size_usd * confidence
        adjusted_size = min(adjusted_size, self.risk_limits.max_position_size)\n        adjusted_size = min(adjusted_size, self.current_capital * 0.3)  # Max 30% of capital per trade\n        \n        if adjusted_size < 10:  # Minimum position size\n            return False\n        \n        # Calculate entry costs\n        entry_fee = adjusted_size * self.costs.taker_fee  # Assume market order\n        slippage = self._calculate_slippage(price, adjusted_size, volatility)\n        \n        total_entry_cost = entry_fee + slippage\n        \n        # Check if enough capital\n        if self.current_capital < adjusted_size + total_entry_cost:\n            self.logger.debug(f\"Insufficient capital for position\")\n            return False\n        \n        # Create position\n        position = Position(\n            symbol=symbol,\n            entry_time=timestamp,\n            entry_price=price,\n            size_usd=adjusted_size,\n            direction=direction,\n            entry_fee=entry_fee\n        )\n        \n        # Update capital and tracking\n        self.current_capital -= (adjusted_size + total_entry_cost)\n        self.total_fees_paid += entry_fee\n        self.total_slippage_paid += slippage\n        \n        # Store position\n        self.positions[symbol] = position\n        \n        # Log trade\n        self.trade_log.append({\n            'timestamp': timestamp,\n            'symbol': symbol,\n            'action': 'OPEN',\n            'price': price,\n            'size_usd': adjusted_size,\n            'direction': direction,\n            'fee': entry_fee,\n            'slippage': slippage,\n            'confidence': confidence\n        })\n        \n        self.logger.debug(f\"Opened {symbol} position: ${adjusted_size:.2f} at {price}\")\n        \n        return True\n    \n    def _close_position(self,\n                       symbol: str,\n                       timestamp: datetime,\n                       price: float,\n                       reason: str = \"SIGNAL\",\n                       volatility: float = 0.02) -> bool:\n        \"\"\"Close an existing position\"\"\"\n        \n        if symbol not in self.positions:\n            return False\n        \n        position = self.positions[symbol]\n        if not position.is_active:\n            return False\n        \n        # Calculate exit costs\n        exit_fee = position.size_usd * self.costs.taker_fee\n        slippage = self._calculate_slippage(price, position.size_usd, volatility)\n        funding_cost = self._calculate_funding_cost(position, timestamp)\n        \n        total_exit_cost = exit_fee + slippage + funding_cost\n        \n        # Calculate P&L\n        price_pnl = position.size_usd * (price - position.entry_price) / position.entry_price * position.direction\n        total_pnl = price_pnl - position.entry_fee - total_exit_cost\n        \n        # Update position\n        position.exit_time = timestamp\n        position.exit_price = price\n        position.exit_fee = exit_fee\n        position.funding_paid += funding_cost\n        position.pnl = total_pnl\n        position.is_active = False\n        \n        # Update capital and tracking\n        self.current_capital += position.size_usd + total_pnl\n        self.total_fees_paid += exit_fee\n        self.total_slippage_paid += slippage\n        self.total_funding_paid += funding_cost\n        \n        # Update daily P&L\n        current_date = timestamp.date()\n        if current_date not in self.daily_pnl:\n            self.daily_pnl[current_date] = 0.0\n        self.daily_pnl[current_date] += total_pnl\n        \n        # Move to closed positions\n        self.closed_positions.append(position)\n        del self.positions[symbol]\n        \n        # Log trade\n        self.trade_log.append({\n            'timestamp': timestamp,\n            'symbol': symbol,\n            'action': 'CLOSE',\n            'price': price,\n            'size_usd': position.size_usd,\n            'direction': position.direction,\n            'fee': exit_fee,\n            'slippage': slippage,\n            'funding': funding_cost,\n            'pnl': total_pnl,\n            'reason': reason,\n            'hold_time_minutes': (timestamp - position.entry_time).total_seconds() / 60\n        })\n        \n        self.logger.debug(f\"Closed {symbol} position: P&L=${total_pnl:.2f} ({reason})\")\n        \n        return True\n    \n    def _update_equity_curve(self, timestamp: datetime, market_prices: Dict[str, float]):\n        \"\"\"Update equity curve with current portfolio value\"\"\"\n        \n        # Start with current cash\n        total_equity = self.current_capital\n        \n        # Add unrealized P&L from open positions\n        for symbol, position in self.positions.items():\n            if symbol in market_prices and position.is_active:\n                current_price = market_prices[symbol]\n                unrealized_pnl = (\n                    position.size_usd * \n                    (current_price - position.entry_price) / position.entry_price * \n                    position.direction\n                )\n                # Subtract estimated exit costs\n                unrealized_pnl -= position.size_usd * self.costs.taker_fee\n                unrealized_pnl -= self._calculate_slippage(current_price, position.size_usd)\n                \n                total_equity += unrealized_pnl\n        \n        self.equity_curve.append(total_equity)\n    \n    def run_backtest(self, \n                    signals_df: pd.DataFrame,\n                    market_data: pd.DataFrame) -> BacktestResult:\n        \"\"\"\n        Run comprehensive backtest\n        \n        Args:\n            signals_df: DataFrame with columns ['timestamp', 'symbol', 'signal', 'confidence', 'predicted_return']\n            market_data: DataFrame with OHLCV data and timestamp index\n            \n        Returns:\n            BacktestResult object with comprehensive results\n        \"\"\"\n        \n        self.logger.info(\"Starting backtest simulation\")\n        self.logger.info(f\"Initial capital: ${self.initial_capital:,.2f}\")\n        self.logger.info(f\"Signals: {len(signals_df)}\")\n        self.logger.info(f\"Market data: {len(market_data)} bars\")\n        \n        # Ensure data is sorted by time\n        market_data = market_data.sort_index()\n        signals_df = signals_df.sort_values('timestamp')\n        \n        # Track processed signals\n        processed_signals = 0\n        skipped_signals = 0\n        \n        # Main simulation loop\n        for idx, row in market_data.iterrows():\n            current_time = idx\n            current_prices = {\n                'BTCUSDT': row['close']  # Assume single symbol for now\n            }\n            \n            # Update funding costs for open positions\n            for symbol, position in list(self.positions.items()):\n                funding_cost = self._calculate_funding_cost(position, current_time)\n                if funding_cost > 0:\n                    self.total_funding_paid += funding_cost\n                    position.funding_paid += funding_cost\n            \n            # Check for exit signals (position timeout, stop loss, take profit)\n            positions_to_close = []\n            \n            for symbol, position in self.positions.items():\n                if symbol in current_prices:\n                    current_price = current_prices[symbol]\n                    \n                    # Position timeout\n                    hold_time = current_time - position.entry_time\n                    if hold_time.total_seconds() > self.risk_limits.position_timeout_minutes * 60:\n                        positions_to_close.append((symbol, \"TIMEOUT\"))\n                        continue\n                    \n                    # Stop loss (2% loss)\n                    unrealized_return = ((current_price - position.entry_price) / position.entry_price) * position.direction\n                    if unrealized_return < -0.02:\n                        positions_to_close.append((symbol, \"STOP_LOSS\"))\n                        continue\n                    \n                    # Take profit (0.8% gain) - DipMaster target\n                    if unrealized_return > 0.008:\n                        positions_to_close.append((symbol, \"TAKE_PROFIT\"))\n                        continue\n                    \n                    # Time-based exit (15-minute boundary logic)\n                    hold_minutes = hold_time.total_seconds() / 60\n                    if hold_minutes >= 15:  # Check for 15-minute boundary exit\n                        minute_in_hour = current_time.minute\n                        if minute_in_hour in [15, 30, 45, 0]:  # 15-minute boundaries\n                            positions_to_close.append((symbol, \"BOUNDARY_EXIT\"))\n            \n            # Close positions marked for exit\n            for symbol, reason in positions_to_close:\n                if symbol in current_prices:\n                    self._close_position(\n                        symbol, current_time, current_prices[symbol], reason,\n                        volatility=row.get('volatility_20', 0.02)\n                    )\n            \n            # Check for new entry signals\n            current_signals = signals_df[\n                (signals_df['timestamp'] >= current_time - timedelta(minutes=self.data_frequency_minutes)) &\n                (signals_df['timestamp'] <= current_time)\n            ]\n            \n            for _, signal_row in current_signals.iterrows():\n                symbol = signal_row.get('symbol', 'BTCUSDT')\n                signal = signal_row.get('signal', signal_row.get('score', 0))\n                confidence = signal_row.get('confidence', abs(signal))\n                \n                # Only take long signals (DipMaster is long-only)\n                if signal > 0.5 and symbol not in self.positions:\n                    \n                    # Calculate position size based on confidence and risk management\n                    base_position_size = min(\n                        self.risk_limits.max_position_size,\n                        self.current_capital * 0.2  # Max 20% per position\n                    )\n                    \n                    success = self._open_position(\n                        symbol=symbol,\n                        timestamp=current_time,\n                        price=current_prices.get(symbol, row['close']),\n                        size_usd=base_position_size,\n                        direction=1,  # Long only\n                        volatility=row.get('volatility_20', 0.02),\n                        confidence=confidence\n                    )\n                    \n                    if success:\n                        processed_signals += 1\n                    else:\n                        skipped_signals += 1\n            \n            # Update equity curve\n            self._update_equity_curve(current_time, current_prices)\n            \n            # Risk checks\n            if not self._check_risk_limits(current_time):\n                self.logger.warning(\"Risk limits violated, stopping backtest\")\n                break\n        \n        # Close any remaining positions\n        final_prices = {\n            'BTCUSDT': market_data.iloc[-1]['close']\n        }\n        \n        for symbol in list(self.positions.keys()):\n            if symbol in final_prices:\n                self._close_position(\n                    symbol, market_data.index[-1], final_prices[symbol], \"END_OF_BACKTEST\"\n                )\n        \n        self.logger.info(f\"Backtest complete: {processed_signals} signals processed, {skipped_signals} skipped\")\n        self.logger.info(f\"Total trades: {len(self.closed_positions)}\")\n        \n        # Generate results\n        return self._generate_results(market_data.index)\n    \n    def _generate_results(self, timestamps: pd.DatetimeIndex) -> BacktestResult:\n        \"\"\"Generate comprehensive backtest results\"\"\"\n        \n        # Create equity curve series\n        equity_series = pd.Series(\n            self.equity_curve,\n            index=timestamps[:len(self.equity_curve)]\n        )\n        \n        # Calculate returns\n        returns = equity_series.pct_change().dropna()\n        \n        # Basic performance metrics\n        total_return = (self.current_capital - self.initial_capital) / self.initial_capital\n        \n        # Risk-adjusted metrics\n        if len(returns) > 1 and returns.std() > 0:\n            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252 * 24 * 12)  # 5-min to annual\n            \n            # Sortino ratio (only downside volatility)\n            negative_returns = returns[returns < 0]\n            if len(negative_returns) > 1:\n                sortino_ratio = returns.mean() / negative_returns.std() * np.sqrt(252 * 24 * 12)\n            else:\n                sortino_ratio = sharpe_ratio\n        else:\n            sharpe_ratio = 0.0\n            sortino_ratio = 0.0\n        \n        # Maximum drawdown\n        rolling_max = equity_series.cummax()\n        drawdowns = (equity_series - rolling_max) / rolling_max\n        max_drawdown = drawdowns.min()\n        \n        # Calmar ratio\n        calmar_ratio = (total_return * 252) / abs(max_drawdown) if max_drawdown != 0 else 0\n        \n        # Trade statistics\n        if self.closed_positions:\n            trade_returns = [pos.pnl / pos.size_usd for pos in self.closed_positions]\n            winning_trades = [ret for ret in trade_returns if ret > 0]\n            losing_trades = [ret for ret in trade_returns if ret < 0]\n            \n            win_rate = len(winning_trades) / len(trade_returns)\n            avg_trade_return = np.mean(trade_returns)\n            avg_win = np.mean(winning_trades) if winning_trades else 0\n            avg_loss = np.mean(losing_trades) if losing_trades else 0\n            \n            # Profit factor\n            total_wins = sum(pos.pnl for pos in self.closed_positions if pos.pnl > 0)\n            total_losses = abs(sum(pos.pnl for pos in self.closed_positions if pos.pnl < 0))\n            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')\n            \n        else:\n            win_rate = 0.0\n            avg_trade_return = 0.0\n            avg_win = 0.0\n            avg_loss = 0.0\n            profit_factor = 0.0\n        \n        # Statistical significance\n        statistical_significance = {}\n        if len(returns) > 30:\n            # T-test for returns being significantly different from zero\n            t_stat, p_value = stats.ttest_1samp(returns, 0)\n            statistical_significance = {\n                't_statistic': t_stat,\n                'p_value': p_value,\n                'is_significant': p_value < 0.05,\n                'confidence_interval': stats.t.interval(\n                    0.95, len(returns)-1, \n                    loc=returns.mean(), \n                    scale=returns.sem()\n                )\n            }\n        \n        # Monthly returns\n        monthly_returns = equity_series.resample('M').last().pct_change().dropna()\n        \n        # Trade history DataFrame\n        trade_history = pd.DataFrame(self.trade_log)\n        \n        return BacktestResult(\n            total_return=total_return,\n            sharpe_ratio=sharpe_ratio,\n            sortino_ratio=sortino_ratio,\n            max_drawdown=max_drawdown,\n            calmar_ratio=calmar_ratio,\n            win_rate=win_rate,\n            profit_factor=profit_factor,\n            avg_trade_return=avg_trade_return,\n            avg_win=avg_win,\n            avg_loss=avg_loss,\n            total_trades=len(self.closed_positions),\n            total_fees=self.total_fees_paid,\n            total_slippage=self.total_slippage_paid,\n            total_funding=self.total_funding_paid,\n            final_capital=self.current_capital,\n            equity_curve=equity_series,\n            trade_history=trade_history,\n            monthly_returns=monthly_returns,\n            statistical_significance=statistical_significance\n        )\n    \n    def generate_html_report(self, result: BacktestResult, filepath: str = None) -> str:\n        \"\"\"Generate comprehensive HTML backtest report\"\"\"\n        \n        if filepath is None:\n            filepath = f\"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html\"\n        \n        # Create subplots\n        fig = make_subplots(\n            rows=3, cols=2,\n            subplot_titles=(\n                'Equity Curve', 'Monthly Returns',\n                'Drawdown', 'Trade Distribution',\n                'Rolling Sharpe (30-day)', 'Cumulative Returns'\n            ),\n            specs=[[{\"colspan\": 2}, None],\n                   [{}, {}],\n                   [{}, {}]]\n        )\n        \n        # Equity curve\n        fig.add_trace(\n            go.Scatter(\n                x=result.equity_curve.index,\n                y=result.equity_curve.values,\n                mode='lines',\n                name='Portfolio Value',\n                line=dict(color='blue', width=2)\n            ),\n            row=1, col=1\n        )\n        \n        # Monthly returns\n        fig.add_trace(\n            go.Bar(\n                x=result.monthly_returns.index,\n                y=result.monthly_returns.values,\n                name='Monthly Returns',\n                marker=dict(\n                    color=['green' if x > 0 else 'red' for x in result.monthly_returns.values]\n                )\n            ),\n            row=2, col=1\n        )\n        \n        # Drawdown\n        rolling_max = result.equity_curve.cummax()\n        drawdowns = (result.equity_curve - rolling_max) / rolling_max\n        \n        fig.add_trace(\n            go.Scatter(\n                x=drawdowns.index,\n                y=drawdowns.values,\n                mode='lines',\n                name='Drawdown',\n                fill='tonexty',\n                line=dict(color='red', width=1)\n            ),\n            row=2, col=2\n        )\n        \n        # Trade P&L distribution\n        if len(self.closed_positions) > 0:\n            trade_pnls = [pos.pnl for pos in self.closed_positions]\n            fig.add_trace(\n                go.Histogram(\n                    x=trade_pnls,\n                    name='Trade P&L',\n                    nbinsx=20,\n                    marker=dict(color='lightblue')\n                ),\n                row=3, col=1\n            )\n        \n        # Rolling Sharpe ratio\n        returns = result.equity_curve.pct_change().dropna()\n        rolling_sharpe = returns.rolling(30).mean() / returns.rolling(30).std() * np.sqrt(252 * 24 * 12)\n        \n        fig.add_trace(\n            go.Scatter(\n                x=rolling_sharpe.index,\n                y=rolling_sharpe.values,\n                mode='lines',\n                name='30-day Sharpe',\n                line=dict(color='orange', width=2)\n            ),\n            row=3, col=2\n        )\n        \n        # Update layout\n        fig.update_layout(\n            height=1200,\n            title_text=f\"DipMaster Strategy Backtest Report\",\n            showlegend=True\n        )\n        \n        # Create HTML report\n        html_content = f\"\"\"\n        <!DOCTYPE html>\n        <html>\n        <head>\n            <title>DipMaster Backtest Report</title>\n            <style>\n                body {{ font-family: Arial, sans-serif; margin: 20px; }}\n                .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}\n                .metric-label {{ font-weight: bold; color: #333; }}\n                .metric-value {{ font-size: 1.2em; color: #007acc; }}\n                .positive {{ color: #28a745; }}\n                .negative {{ color: #dc3545; }}\n                table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}\n                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}\n                th {{ background-color: #f2f2f2; }}\n            </style>\n        </head>\n        <body>\n            <h1>DipMaster Strategy Backtest Report</h1>\n            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n            \n            <h2>Performance Summary</h2>\n            <div class=\"metric\">\n                <div class=\"metric-label\">Total Return</div>\n                <div class=\"metric-value {'positive' if result.total_return > 0 else 'negative'}\">{result.total_return:.2%}</div>\n            </div>\n            <div class=\"metric\">\n                <div class=\"metric-label\">Sharpe Ratio</div>\n                <div class=\"metric-value\">{result.sharpe_ratio:.2f}</div>\n            </div>\n            <div class=\"metric\">\n                <div class=\"metric-label\">Win Rate</div>\n                <div class=\"metric-value\">{result.win_rate:.1%}</div>\n            </div>\n            <div class=\"metric\">\n                <div class=\"metric-label\">Max Drawdown</div>\n                <div class=\"metric-value negative\">{result.max_drawdown:.2%}</div>\n            </div>\n            <div class=\"metric\">\n                <div class=\"metric-label\">Profit Factor</div>\n                <div class=\"metric-value\">{result.profit_factor:.2f}</div>\n            </div>\n            <div class=\"metric\">\n                <div class=\"metric-label\">Total Trades</div>\n                <div class=\"metric-value\">{result.total_trades}</div>\n            </div>\n            \n            <h2>Cost Breakdown</h2>\n            <table>\n                <tr><th>Cost Type</th><th>Amount (USD)</th><th>% of Initial Capital</th></tr>\n                <tr><td>Trading Fees</td><td>{result.total_fees:.2f}</td><td>{result.total_fees/self.initial_capital:.2%}</td></tr>\n                <tr><td>Slippage</td><td>{result.total_slippage:.2f}</td><td>{result.total_slippage/self.initial_capital:.2%}</td></tr>\n                <tr><td>Funding</td><td>{result.total_funding:.2f}</td><td>{result.total_funding/self.initial_capital:.2%}</td></tr>\n                <tr><td><strong>Total Costs</strong></td><td><strong>{result.total_fees + result.total_slippage + result.total_funding:.2f}</strong></td><td><strong>{(result.total_fees + result.total_slippage + result.total_funding)/self.initial_capital:.2%}</strong></td></tr>\n            </table>\n            \n            <h2>Statistical Analysis</h2>\n            <p><strong>Statistical Significance:</strong> {'Yes' if result.statistical_significance.get('is_significant', False) else 'No'} (p-value: {result.statistical_significance.get('p_value', 'N/A'):.4f})</p>\n            \n            <div>{fig.to_html(include_plotlyjs='cdn')}</div>\n            \n            <h2>Risk Metrics</h2>\n            <table>\n                <tr><th>Metric</th><th>Value</th></tr>\n                <tr><td>Sortino Ratio</td><td>{result.sortino_ratio:.2f}</td></tr>\n                <tr><td>Calmar Ratio</td><td>{result.calmar_ratio:.2f}</td></tr>\n                <tr><td>Average Win</td><td>{result.avg_win:.2%}</td></tr>\n                <tr><td>Average Loss</td><td>{result.avg_loss:.2%}</td></tr>\n                <tr><td>Average Trade</td><td>{result.avg_trade_return:.2%}</td></tr>\n            </table>\n        </body>\n        </html>\n        \"\"\"\n        \n        with open(filepath, 'w') as f:\n            f.write(html_content)\n        \n        self.logger.info(f\"HTML report generated: {filepath}\")\n        return filepath