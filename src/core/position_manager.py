import logging
import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


class PositionStatus(Enum):
    """Position status types"""
    PENDING = "pending"
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class Position:
    """Trading position data structure"""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    entry_time: datetime
    quantity: float
    status: PositionStatus
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    holding_minutes: Optional[float] = None
    entry_reason: Optional[str] = None
    exit_reason: Optional[str] = None
    entry_indicators: Optional[Dict] = None
    exit_indicators: Optional[Dict] = None
    leverage: int = 1  # Default leverage
    position_size_usd: Optional[float] = None  # Position size in USD
    

class PositionManager:
    """Manages trading positions and tracks performance"""
    
    def __init__(self, data_dir: str = "data/positions", strategy_mode: str = "optimized"):
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.position_counter = 0
        self.strategy_mode = strategy_mode  # "original" or "optimized"
        
        # Original strategy parameters
        if strategy_mode == "original":
            self.base_position_usd = 1000  # Fixed position size
            self.leverage = 10  # 10x leverage
            self.max_concurrent_positions = 8
            self.use_stop_loss = False
            self.use_take_profit = False
        else:
            self.base_position_usd = None  # Dynamic sizing
            self.leverage = 1
            self.max_concurrent_positions = 3
            self.use_stop_loss = True
            self.use_take_profit = True
            
        self.daily_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'total_volume': 0.0
        }
        
    def create_position(self, symbol: str, entry_price: float, quantity: float, 
                       reason: str = "", entry_reason: str = None, indicators: Dict = None,
                       leverage: int = None, position_size_usd: float = None) -> Position:
        """Create a new position"""
        self.position_counter += 1
        position_id = f"{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.position_counter}"
        
        # Support both 'reason' and 'entry_reason' for backward compatibility
        actual_reason = reason if reason else (entry_reason if entry_reason else "")
        
        # Use strategy defaults if not specified
        if leverage is None:
            leverage = self.leverage
        if position_size_usd is None and self.strategy_mode == "original":
            position_size_usd = self.base_position_usd
            
        position = Position(
            id=position_id,
            symbol=symbol,
            side='buy',
            entry_price=entry_price,
            entry_time=datetime.now(),
            quantity=quantity,
            status=PositionStatus.OPEN,
            entry_reason=actual_reason,
            entry_indicators=indicators,
            leverage=leverage,
            position_size_usd=position_size_usd
        )
        
        self.positions[position_id] = position
        
        # Log with leverage info for original strategy
        if self.strategy_mode == "original":
            logger.info(f"📈 Opened position {position_id}: {symbol} @ {entry_price} "
                       f"[{leverage}x leverage, ${position_size_usd:.0f}]")
        else:
            logger.info(f"📈 Opened position {position_id}: {symbol} @ {entry_price}")
        
        return position
        
    def close_position(self, position_id: str, exit_price: float, 
                       exit_reason: str = "", indicators: Dict = None) -> Optional[Position]:
        """Close an existing position"""
        if position_id not in self.positions:
            logger.error(f"Position {position_id} not found")
            return None
            
        position = self.positions[position_id]
        position.status = PositionStatus.CLOSED
        position.exit_price = exit_price
        position.exit_time = datetime.now()
        position.exit_reason = exit_reason
        position.exit_indicators = indicators
        
        # Calculate PnL
        position.pnl = (exit_price - position.entry_price) * position.quantity
        position.pnl_percent = (exit_price - position.entry_price) / position.entry_price * 100
        position.holding_minutes = (position.exit_time - position.entry_time).total_seconds() / 60
        
        # Update statistics
        self.daily_stats['total_trades'] += 1
        self.daily_stats['total_pnl'] += position.pnl
        self.daily_stats['total_volume'] += position.quantity * exit_price
        
        if position.pnl > 0:
            self.daily_stats['winning_trades'] += 1
            logger.info(f"✅ Closed {position_id} with profit: ${position.pnl:.2f} ({position.pnl_percent:.2f}%)")
        else:
            self.daily_stats['losing_trades'] += 1
            logger.info(f"❌ Closed {position_id} with loss: ${position.pnl:.2f} ({position.pnl_percent:.2f}%)")
            
        # Move to closed positions
        self.closed_positions.append(position)
        del self.positions[position_id]
        
        # Save position data
        self._save_position(position)
        
        return position
        
    def get_open_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get all open positions or for specific symbol"""
        if symbol:
            return [p for p in self.positions.values() if p.symbol == symbol]
        return list(self.positions.values())
        
    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get first open position for symbol"""
        for position in self.positions.values():
            if position.symbol == symbol and position.status == PositionStatus.OPEN:
                return position
        return None
        
    def calculate_exposure(self) -> Dict:
        """Calculate current market exposure"""
        exposure = {
            'total_positions': len(self.positions),
            'total_value': 0.0,
            'by_symbol': {}
        }
        
        for position in self.positions.values():
            value = position.entry_price * position.quantity
            exposure['total_value'] += value
            
            if position.symbol not in exposure['by_symbol']:
                exposure['by_symbol'][position.symbol] = 0
            exposure['by_symbol'][position.symbol] += value
            
        return exposure
        
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.closed_positions:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'avg_holding_time': 0
            }
            
        wins = [p for p in self.closed_positions if p.pnl > 0]
        losses = [p for p in self.closed_positions if p.pnl <= 0]
        
        total_profit = sum(p.pnl for p in wins) if wins else 0
        total_loss = abs(sum(p.pnl for p in losses)) if losses else 0
        
        stats = {
            'total_trades': len(self.closed_positions),
            'win_rate': len(wins) / len(self.closed_positions) * 100,
            'avg_profit': total_profit / len(wins) if wins else 0,
            'avg_loss': total_loss / len(losses) if losses else 0,
            'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
            'avg_holding_time': sum(p.holding_minutes for p in self.closed_positions) / len(self.closed_positions),
            'total_pnl': sum(p.pnl for p in self.closed_positions),
            'best_trade': max(self.closed_positions, key=lambda p: p.pnl).pnl if self.closed_positions else 0,
            'worst_trade': min(self.closed_positions, key=lambda p: p.pnl).pnl if self.closed_positions else 0
        }
        
        return stats
        
    def _save_position(self, position: Position):
        """Save position data to file"""
        try:
            filename = self.data_dir / f"position_{position.id}.json"
            position_dict = asdict(position)
            
            # Convert datetime objects to strings
            for key in ['entry_time', 'exit_time']:
                if position_dict[key]:
                    position_dict[key] = position_dict[key].isoformat()
                    
            # Convert enum to string
            position_dict['status'] = position.status.value
            
            with open(filename, 'w') as f:
                json.dump(position_dict, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save position: {e}")
            
    def load_positions(self):
        """Load positions from files"""
        try:
            for file in self.data_dir.glob("position_*.json"):
                with open(file, 'r') as f:
                    data = json.load(f)
                    
                # Convert strings back to datetime
                for key in ['entry_time', 'exit_time']:
                    if data[key]:
                        data[key] = datetime.fromisoformat(data[key])
                        
                # Convert string to enum
                data['status'] = PositionStatus(data['status'])
                
                position = Position(**data)
                
                if position.status == PositionStatus.CLOSED:
                    self.closed_positions.append(position)
                else:
                    self.positions[position.id] = position
                    
        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
            
    def can_open_position(self) -> bool:
        """Check if we can open a new position based on strategy limits"""
        open_positions = len([p for p in self.positions.values() 
                             if p.status == PositionStatus.OPEN])
        return open_positions < self.max_concurrent_positions
        
    def get_position_size(self, symbol: str, price: float) -> Dict:
        """Calculate position size based on strategy mode"""
        if self.strategy_mode == "original":
            # Fixed position sizing for original strategy
            quantity = self.base_position_usd / price
            return {
                'quantity': quantity,
                'size_usd': self.base_position_usd,
                'leverage': self.leverage,
                'effective_size': self.base_position_usd * self.leverage
            }
        else:
            # Dynamic sizing for optimized strategy (implement as needed)
            return {
                'quantity': 0,
                'size_usd': 0,
                'leverage': 1,
                'effective_size': 0
            }
            
    async def monitor_positions(self, price_updates: Dict[str, float]):
        """Monitor open positions with current prices"""
        for position in list(self.positions.values()):
            if position.symbol in price_updates:
                current_price = price_updates[position.symbol]
                # Account for leverage in PnL calculation
                leverage = position.leverage if hasattr(position, 'leverage') else 1
                unrealized_pnl = (current_price - position.entry_price) * position.quantity * leverage
                unrealized_pnl_percent = (current_price - position.entry_price) / position.entry_price * 100 * leverage
                
                logger.debug(f"Position {position.id}: Unrealized PnL: ${unrealized_pnl:.2f} ({unrealized_pnl_percent:.2f}%)")