import asyncio
import logging
from datetime import datetime
from typing import Dict, List
from rich.console import Console
from rich.table import Table
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

logger = logging.getLogger(__name__)


class RealTimeDashboard:
    """Real-time monitoring dashboard for DipMaster trading"""
    
    def __init__(self, engine):
        self.engine = engine
        self.console = Console()
        self.layout = Layout()
        self.running = False
        
    def create_layout(self):
        """Create dashboard layout"""
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        
        self.layout["main"].split_row(
            Layout(name="positions", ratio=2),
            Layout(name="stats")
        )
        
        self.layout["stats"].split(
            Layout(name="performance"),
            Layout(name="signals")
        )
        
    def generate_header(self) -> Panel:
        """Generate header panel"""
        header_text = Text(
            f"🤖 DipMaster Real-Time Trading Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            style="bold magenta"
        )
        
        # Add boundary countdown
        seconds_to_boundary = self.engine.timing_manager.seconds_to_boundary()
        slot = self.engine.timing_manager.get_current_slot()
        
        countdown_text = Text(
            f"⏰ Next 15-min boundary in: {seconds_to_boundary}s | Current Slot: {slot.name}",
            style="cyan"
        )
        
        combined_text = Text()
        combined_text.append(header_text)
        combined_text.append("\n")
        combined_text.append(countdown_text)
        
        return Panel(combined_text, style="bold blue")
        
    def generate_positions_table(self) -> Table:
        """Generate positions table"""
        table = Table(title="📊 Open Positions", show_header=True, header_style="bold cyan")
        
        table.add_column("Symbol", style="yellow")
        table.add_column("Entry Price", justify="right")
        table.add_column("Current Price", justify="right")
        table.add_column("Quantity", justify="right")
        table.add_column("PnL", justify="right")
        table.add_column("PnL %", justify="right")
        table.add_column("Hold Time", justify="right")
        table.add_column("Status")
        
        for position in self.engine.position_manager.get_open_positions():
            current_price = self.engine.current_prices.get(position.symbol, position.entry_price)
            pnl = (current_price - position.entry_price) * position.quantity
            pnl_percent = (current_price - position.entry_price) / position.entry_price * 100
            holding_time = (datetime.now() - position.entry_time).total_seconds() / 60
            
            pnl_style = "green" if pnl >= 0 else "red"
            
            table.add_row(
                position.symbol,
                f"${position.entry_price:.4f}",
                f"${current_price:.4f}",
                f"{position.quantity:.4f}",
                f"${pnl:.2f}",
                f"{pnl_percent:.2f}%",
                f"{holding_time:.0f}m",
                position.status.value,
                style=pnl_style if pnl != 0 else None
            )
            
        if not self.engine.position_manager.get_open_positions():
            table.add_row("No open positions", "", "", "", "", "", "", "")
            
        return table
        
    def generate_performance_panel(self) -> Panel:
        """Generate performance statistics panel"""
        stats = self.engine.position_manager.get_performance_stats()
        exposure = self.engine.position_manager.calculate_exposure()
        
        content = f"""
[bold cyan]Performance Statistics[/bold cyan]
─────────────────────
Total Trades: {stats['total_trades']}
Win Rate: {stats['win_rate']:.1f}%
Total PnL: ${stats['total_pnl']:.2f}
Avg Profit: ${stats['avg_profit']:.2f}
Avg Loss: ${stats['avg_loss']:.2f}
Profit Factor: {stats['profit_factor']:.2f}
Avg Hold Time: {stats['avg_holding_time']:.1f}m

[bold cyan]Current Exposure[/bold cyan]
─────────────────────
Open Positions: {exposure['total_positions']}
Total Value: ${exposure['total_value']:.2f}
"""
        
        return Panel(content, title="📈 Performance", style="green")
        
    def generate_signals_panel(self) -> Panel:
        """Generate recent signals panel"""
        recent_signals = self.engine.signal_history[-10:] if self.engine.signal_history else []
        
        content = "[bold cyan]Recent Signals[/bold cyan]\n"
        content += "─────────────────────\n"
        
        for signal_data in reversed(recent_signals):
            signal = signal_data['signal']
            time_str = signal.timestamp.strftime('%H:%M:%S')
            
            emoji = "🟢" if signal.action == 'buy' else "🔴"
            content += f"{emoji} {time_str} - {signal.symbol}\n"
            content += f"   {signal.signal_type.value}: {signal.confidence:.2f}\n"
            content += f"   {signal.reason}\n\n"
            
        if not recent_signals:
            content += "No recent signals\n"
            
        return Panel(content, title="📡 Signals", style="yellow")
        
    def generate_footer(self) -> Panel:
        """Generate footer panel"""
        risk_ok = self.engine.check_risk_limits()
        risk_status = "✅ OK" if risk_ok else "⚠️ LIMITS EXCEEDED"
        
        mode = "💰 LIVE" if not self.engine.config.get('paper_trading', True) else "📝 PAPER"
        
        footer_text = f"Status: {'🟢 Running' if self.engine.running else '🔴 Stopped'} | " \
                     f"Risk: {risk_status} | " \
                     f"Mode: {mode} | " \
                     f"Symbols: {', '.join(self.engine.symbols)}"
                     
        return Panel(footer_text, style="bold blue")
        
    def update_display(self):
        """Update the entire display"""
        self.layout["header"].update(self.generate_header())
        self.layout["positions"].update(self.generate_positions_table())
        self.layout["performance"].update(self.generate_performance_panel())
        self.layout["signals"].update(self.generate_signals_panel())
        self.layout["footer"].update(self.generate_footer())
        
        return self.layout
        
    async def run(self):
        """Run the dashboard"""
        self.running = True
        self.create_layout()
        
        with Live(self.update_display(), refresh_per_second=1, screen=True) as live:
            while self.running:
                try:
                    live.update(self.update_display())
                    await asyncio.sleep(1)
                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    logger.error(f"Dashboard error: {e}")
                    await asyncio.sleep(1)
                    
    def stop(self):
        """Stop the dashboard"""
        self.running = False


class SimpleDashboard:
    """Simple text-based dashboard for basic monitoring"""
    
    def __init__(self, engine):
        self.engine = engine
        self.running = False
        
    async def run(self):
        """Run simple dashboard"""
        self.running = True
        
        while self.running:
            try:
                self.print_status()
                await asyncio.sleep(5)
            except KeyboardInterrupt:
                self.running = False
                break
                
    def print_status(self):
        """Print simple status update"""
        stats = self.engine.position_manager.get_performance_stats()
        exposure = self.engine.position_manager.calculate_exposure()
        seconds_to_boundary = self.engine.timing_manager.seconds_to_boundary()
        
        print("\n" + "="*50)
        print(f"DipMaster Trading Status - {datetime.now().strftime('%H:%M:%S')}")
        print("="*50)
        print(f"Open Positions: {exposure['total_positions']}")
        print(f"Total Exposure: ${exposure['total_value']:.2f}")
        print(f"Today's PnL: ${stats['total_pnl']:.2f}")
        print(f"Win Rate: {stats['win_rate']:.1f}%")
        print(f"Next Boundary: {seconds_to_boundary}s")
        print("="*50)
        
        # Show open positions
        if self.engine.position_manager.get_open_positions():
            print("\nOpen Positions:")
            for position in self.engine.position_manager.get_open_positions():
                current_price = self.engine.current_prices.get(position.symbol, position.entry_price)
                pnl = (current_price - position.entry_price) * position.quantity
                pnl_percent = (current_price - position.entry_price) / position.entry_price * 100
                print(f"  {position.symbol}: ${pnl:.2f} ({pnl_percent:.2f}%)")
                
    def stop(self):
        """Stop the dashboard"""
        self.running = False