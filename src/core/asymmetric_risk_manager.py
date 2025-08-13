"""
Asymmetric Risk Management System - Phase 2 Optimization
非对称风险管理：让亏损快走，让利润奔跑

核心理念：
- 亏损0.5%立即止损，不等15分钟边界
- 盈利1%以上延长持仓至90分钟
- 分批止盈：0.8%, 1.5%, 2.5%
- 追踪止损：盈利后0.3%追踪

目标：将盈亏比从0.58提升至1.5-2.0
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class ExitReason(Enum):
    """出场原因"""
    EMERGENCY_STOP = "emergency_stop"      # 0.5%紧急止损
    NORMAL_STOP = "normal_stop"            # 1%正常止损
    TRAILING_STOP = "trailing_stop"        # 追踪止损
    PARTIAL_PROFIT = "partial_profit"      # 分批止盈
    FINAL_PROFIT = "final_profit"          # 最终止盈
    BOUNDARY_PROFIT = "boundary_profit"    # 边界盈利出场
    BOUNDARY_NEUTRAL = "boundary_neutral"  # 边界中性出场
    MAX_TIME = "max_time"                  # 最大时间


@dataclass
class RiskLevels:
    """风险等级配置"""
    emergency_stop: float = 0.005     # 0.5%紧急止损
    normal_stop: float = 0.01         # 1.0%正常止损
    trailing_distance: float = 0.003  # 0.3%追踪止损距离
    
    # 分批止盈等级
    profit_levels: List[Tuple[float, float]] = None
    
    def __post_init__(self):
        if self.profit_levels is None:
            self.profit_levels = [
                (0.008, 0.25),  # 0.8%时减仓25%
                (0.015, 0.35),  # 1.5%时减仓35%
                (0.025, 0.4)    # 2.5%时减仓剩余40%
            ]


@dataclass
class Position:
    """增强持仓信息"""
    symbol: str
    entry_time: datetime
    entry_price: float
    quantity: float
    remaining_quantity: float
    stop_loss: float
    trailing_stop: Optional[float] = None
    highest_price: float = 0.0
    profit_levels_hit: List[int] = None
    total_realized_pnl: float = 0.0
    
    def __post_init__(self):
        if self.profit_levels_hit is None:
            self.profit_levels_hit = []
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
    
    @property
    def holding_minutes(self) -> float:
        """持仓时间（分钟）"""
        return (datetime.now() - self.entry_time).total_seconds() / 60
    
    def get_unrealized_pnl_percent(self, current_price: float) -> float:
        """未实现盈亏百分比"""
        return (current_price - self.entry_price) / self.entry_price * 100
    
    def get_total_pnl_percent(self, current_price: float) -> float:
        """总盈亏百分比（包含已实现）"""
        unrealized = self.get_unrealized_pnl_percent(current_price)
        realized_percent = self.total_realized_pnl / (self.entry_price * self.quantity) * 100
        return unrealized + realized_percent


class AsymmetricRiskManager:
    """非对称风险管理器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.risk_levels = RiskLevels()
        self.positions: Dict[str, Position] = {}
        
        # === 时间管理参数 ===
        self.min_holding_minutes = 15      # 最小持仓时间
        self.profit_extension_minutes = 90  # 盈利时延长持仓
        self.loss_cut_minutes = 30         # 亏损时快速止损
        self.max_holding_minutes = 180     # 绝对最大持仓
        
        # === 15分钟边界参数 ===
        self.boundary_minutes = [15, 30, 45, 60]
        self.profit_boundary_threshold = 0.002  # 盈利0.2%以上才在边界出场
        
        # 统计数据
        self.trade_stats = {
            'total_exits': 0,
            'emergency_stops': 0,
            'trailing_stops': 0,
            'profit_takes': 0,
            'avg_hold_time_profit': 0,
            'avg_hold_time_loss': 0
        }
        
    def create_position(self, symbol: str, entry_price: float, 
                       quantity: float, atr: float = 0.01) -> Position:
        """创建持仓"""
        # 动态止损基于ATR
        emergency_stop = max(self.risk_levels.emergency_stop, atr * 0.5)
        normal_stop = max(self.risk_levels.normal_stop, atr * 1.0)
        
        position = Position(
            symbol=symbol,
            entry_time=datetime.now(),
            entry_price=entry_price,
            quantity=quantity,
            remaining_quantity=quantity,
            stop_loss=entry_price * (1 - emergency_stop),  # 初始紧急止损
            highest_price=entry_price
        )
        
        self.positions[symbol] = position
        
        logger.info(f"Created position: {symbol} @ {entry_price:.4f}, "
                   f"Emergency stop: {position.stop_loss:.4f}")
        
        return position
        
    def update_position(self, symbol: str, current_price: float) -> List[Dict]:
        """更新持仓并检查出场信号"""
        if symbol not in self.positions:
            return []
            
        position = self.positions[symbol]
        exit_signals = []
        
        # 更新最高价
        if current_price > position.highest_price:
            position.highest_price = current_price
            
        # 更新追踪止损
        self._update_trailing_stop(position, current_price)
        
        # 检查各种出场条件
        exit_signals.extend(self._check_stop_loss(position, current_price))
        exit_signals.extend(self._check_profit_taking(position, current_price))
        exit_signals.extend(self._check_boundary_exit(position, current_price))
        exit_signals.extend(self._check_time_exit(position, current_price))
        
        return exit_signals
        
    def _update_trailing_stop(self, position: Position, current_price: float):
        """更新追踪止损"""
        pnl_percent = position.get_unrealized_pnl_percent(current_price)
        
        # 盈利1%后启用追踪止损
        if pnl_percent > 1.0:
            trailing_price = current_price * (1 - self.risk_levels.trailing_distance)
            
            if position.trailing_stop is None:
                position.trailing_stop = trailing_price
                logger.debug(f"{position.symbol}: Trailing stop activated at {trailing_price:.4f}")
            elif trailing_price > position.trailing_stop:
                position.trailing_stop = trailing_price
                logger.debug(f"{position.symbol}: Trailing stop updated to {trailing_price:.4f}")
                
    def _check_stop_loss(self, position: Position, current_price: float) -> List[Dict]:
        """检查止损"""
        signals = []
        
        # 紧急止损 - 0.5%
        emergency_threshold = position.entry_price * (1 - self.risk_levels.emergency_stop)
        if current_price <= emergency_threshold:
            signals.append({
                'symbol': position.symbol,
                'action': 'SELL',
                'quantity_ratio': 1.0,  # 全部平仓
                'price': current_price,
                'reason': ExitReason.EMERGENCY_STOP,
                'priority': 'URGENT'
            })
            return signals  # 紧急止损后直接返回
            
        # 追踪止损
        if position.trailing_stop and current_price <= position.trailing_stop:
            signals.append({
                'symbol': position.symbol,
                'action': 'SELL',
                'quantity_ratio': 1.0,
                'price': current_price,
                'reason': ExitReason.TRAILING_STOP,
                'priority': 'HIGH'
            })
            
        return signals
        
    def _check_profit_taking(self, position: Position, current_price: float) -> List[Dict]:
        """检查分批止盈"""
        signals = []
        pnl_percent = position.get_unrealized_pnl_percent(current_price) / 100
        
        for i, (profit_level, exit_ratio) in enumerate(self.risk_levels.profit_levels):
            if i in position.profit_levels_hit:
                continue  # 已经触发过的等级
                
            if pnl_percent >= profit_level:
                position.profit_levels_hit.append(i)
                
                signals.append({
                    'symbol': position.symbol,
                    'action': 'SELL',
                    'quantity_ratio': exit_ratio,
                    'price': current_price,
                    'reason': ExitReason.PARTIAL_PROFIT,
                    'priority': 'MEDIUM',
                    'profit_level': i + 1
                })
                
                logger.info(f"{position.symbol}: Profit level {i+1} hit at {pnl_percent*100:.2f}%, "
                           f"selling {exit_ratio*100:.0f}%")
                
        return signals
        
    def _check_boundary_exit(self, position: Position, current_price: float) -> List[Dict]:
        """检查15分钟边界出场（仅适用于微盈微亏）"""
        signals = []
        
        current_time = datetime.now()
        minute = current_time.minute
        holding_time = position.holding_minutes
        pnl_percent = position.get_unrealized_pnl_percent(current_price)
        
        # 必须持仓至少15分钟
        if holding_time < self.min_holding_minutes:
            return signals
            
        # 检查是否在边界时间点
        is_boundary = any(abs(minute - b) <= 1 for b in self.boundary_minutes) or minute >= 59
        
        if not is_boundary:
            return signals
            
        # 非对称边界逻辑
        if pnl_percent < -0.5:  # 亏损0.5%以上，不等边界，应该已经止损了
            return signals
        elif pnl_percent > 0.2:  # 盈利0.2%以上，边界出场
            # 如果已经分批止盈，只出场剩余仓位
            signals.append({
                'symbol': position.symbol,
                'action': 'SELL', 
                'quantity_ratio': 1.0,
                'price': current_price,
                'reason': ExitReason.BOUNDARY_PROFIT,
                'priority': 'MEDIUM'
            })
        elif -0.2 <= pnl_percent <= 0.2:  # 微盈微亏区间
            # 根据持仓时间决定
            if holding_time >= 45:  # 持仓超过45分钟
                signals.append({
                    'symbol': position.symbol,
                    'action': 'SELL',
                    'quantity_ratio': 1.0,
                    'price': current_price,
                    'reason': ExitReason.BOUNDARY_NEUTRAL,
                    'priority': 'LOW'
                })
                
        return signals
        
    def _check_time_exit(self, position: Position, current_price: float) -> List[Dict]:
        """检查时间出场"""
        signals = []
        pnl_percent = position.get_unrealized_pnl_percent(current_price)
        holding_time = position.holding_minutes
        
        # 盈利时的时间管理
        if pnl_percent > 0:
            if holding_time >= self.profit_extension_minutes:  # 盈利时90分钟
                signals.append({
                    'symbol': position.symbol,
                    'action': 'SELL',
                    'quantity_ratio': 1.0,
                    'price': current_price,
                    'reason': ExitReason.FINAL_PROFIT,
                    'priority': 'MEDIUM'
                })
        else:
            # 亏损时的时间管理
            if holding_time >= self.loss_cut_minutes:  # 亏损时30分钟
                signals.append({
                    'symbol': position.symbol,
                    'action': 'SELL',
                    'quantity_ratio': 1.0,
                    'price': current_price,
                    'reason': ExitReason.NORMAL_STOP,
                    'priority': 'HIGH'
                })
                
        # 绝对最大时间
        if holding_time >= self.max_holding_minutes:
            signals.append({
                'symbol': position.symbol,
                'action': 'SELL',
                'quantity_ratio': 1.0,
                'price': current_price,
                'reason': ExitReason.MAX_TIME,
                'priority': 'URGENT'
            })
            
        return signals
        
    def execute_exit(self, symbol: str, exit_signal: Dict) -> Optional[Dict]:
        """执行出场"""
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        quantity_to_sell = position.remaining_quantity * exit_signal['quantity_ratio']
        
        if quantity_to_sell <= 0:
            return None
            
        # 计算这部分的盈亏
        pnl = (exit_signal['price'] - position.entry_price) * quantity_to_sell
        pnl_percent = pnl / (position.entry_price * quantity_to_sell) * 100
        
        # 更新持仓
        position.remaining_quantity -= quantity_to_sell
        position.total_realized_pnl += pnl
        
        # 更新统计
        self._update_trade_stats(exit_signal['reason'], position.holding_minutes, pnl_percent > 0)
        
        trade_result = {
            'symbol': symbol,
            'exit_time': datetime.now(),
            'exit_price': exit_signal['price'],
            'quantity': quantity_to_sell,
            'pnl_usd': pnl,
            'pnl_percent': pnl_percent,
            'holding_minutes': position.holding_minutes,
            'exit_reason': exit_signal['reason'].value,
            'is_partial': position.remaining_quantity > 0
        }
        
        # 如果全部平仓，删除持仓记录
        if position.remaining_quantity <= 0.01:  # 考虑精度问题
            del self.positions[symbol]
            trade_result['is_final'] = True
            
        logger.info(f"Exit executed: {symbol} - {exit_signal['reason'].value}, "
                   f"PnL: {pnl_percent:.2f}%, Holding: {position.holding_minutes:.1f}min")
        
        return trade_result
        
    def _update_trade_stats(self, reason: ExitReason, holding_time: float, is_profit: bool):
        """更新交易统计"""
        self.trade_stats['total_exits'] += 1
        
        if reason == ExitReason.EMERGENCY_STOP:
            self.trade_stats['emergency_stops'] += 1
        elif reason == ExitReason.TRAILING_STOP:
            self.trade_stats['trailing_stops'] += 1
        elif reason in [ExitReason.PARTIAL_PROFIT, ExitReason.FINAL_PROFIT]:
            self.trade_stats['profit_takes'] += 1
            
        # 更新平均持仓时间
        if is_profit:
            current_avg = self.trade_stats['avg_hold_time_profit']
            new_count = self.trade_stats['profit_takes']
            self.trade_stats['avg_hold_time_profit'] = (current_avg * (new_count - 1) + holding_time) / new_count
        else:
            loss_count = self.trade_stats['total_exits'] - self.trade_stats['profit_takes']
            if loss_count > 0:
                current_avg = self.trade_stats['avg_hold_time_loss']
                self.trade_stats['avg_hold_time_loss'] = (current_avg * (loss_count - 1) + holding_time) / loss_count
                
    def get_position_status(self, symbol: str, current_price: float) -> Optional[Dict]:
        """获取持仓状态"""
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        
        return {
            'symbol': symbol,
            'entry_price': position.entry_price,
            'current_price': current_price,
            'unrealized_pnl_percent': position.get_unrealized_pnl_percent(current_price),
            'total_pnl_percent': position.get_total_pnl_percent(current_price),
            'holding_minutes': position.holding_minutes,
            'stop_loss': position.stop_loss,
            'trailing_stop': position.trailing_stop,
            'remaining_quantity': position.remaining_quantity,
            'profit_levels_hit': len(position.profit_levels_hit)
        }
        
    def get_risk_metrics(self) -> Dict:
        """获取风险指标"""
        total_exits = self.trade_stats['total_exits']
        
        if total_exits == 0:
            return {'message': 'No trades completed yet'}
            
        return {
            'total_exits': total_exits,
            'emergency_stop_rate': self.trade_stats['emergency_stops'] / total_exits * 100,
            'trailing_stop_rate': self.trade_stats['trailing_stops'] / total_exits * 100,
            'profit_take_rate': self.trade_stats['profit_takes'] / total_exits * 100,
            'avg_hold_time_profit': self.trade_stats['avg_hold_time_profit'],
            'avg_hold_time_loss': self.trade_stats['avg_hold_time_loss'],
            'asymmetric_ratio': (self.trade_stats['avg_hold_time_profit'] / 
                               max(self.trade_stats['avg_hold_time_loss'], 1))
        }