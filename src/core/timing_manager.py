import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Optional, List, Dict
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TimeSlot(Enum):
    """15-minute time slots within an hour"""
    SLOT_1 = 1  # 00-14 minutes
    SLOT_2 = 2  # 15-29 minutes  
    SLOT_3 = 3  # 30-44 minutes
    SLOT_4 = 4  # 45-59 minutes


@dataclass
class BoundaryEvent:
    """Event triggered at 15-minute boundaries"""
    timestamp: datetime
    slot: TimeSlot
    next_boundary: datetime
    seconds_to_next: int


class BoundaryTimingManager:
    """Manages 15-minute boundary timing for DipMaster strategy"""
    
    def __init__(self, strategy_mode: str = "optimized"):
        self.callbacks: List[Callable] = []
        self.running = False
        self.current_slot: Optional[TimeSlot] = None
        self.positions_by_slot: Dict[TimeSlot, List] = {slot: [] for slot in TimeSlot}
        self.strategy_mode = strategy_mode  # "original" or "optimized"
        
        # Original strategy slot exit weights (based on real data)
        self.original_slot_weights = {
            TimeSlot.SLOT_1: 0.19,   # 00-14min: 19.0%
            TimeSlot.SLOT_2: 0.262,  # 15-29min: 26.2%
            TimeSlot.SLOT_3: 0.25,   # 30-44min: 25.0%
            TimeSlot.SLOT_4: 0.298   # 45-59min: 29.8%
        }
        
    def add_boundary_callback(self, callback: Callable):
        """Add callback for boundary events"""
        self.callbacks.append(callback)
        
    def get_current_slot(self) -> TimeSlot:
        """Get current 15-minute slot"""
        minute = datetime.now().minute
        if minute < 15:
            return TimeSlot.SLOT_1
        elif minute < 30:
            return TimeSlot.SLOT_2
        elif minute < 45:
            return TimeSlot.SLOT_3
        else:
            return TimeSlot.SLOT_4
            
    def get_next_boundary(self) -> datetime:
        """Calculate next 15-minute boundary"""
        now = datetime.now()
        minute = now.minute
        
        # Calculate minutes to next boundary
        next_boundary_minute = ((minute // 15) + 1) * 15
        
        if next_boundary_minute >= 60:
            # Next boundary is in the next hour
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            return next_hour
        else:
            return now.replace(minute=next_boundary_minute, second=0, microsecond=0)
            
    def seconds_to_boundary(self) -> int:
        """Calculate seconds until next 15-minute boundary"""
        next_boundary = self.get_next_boundary()
        return int((next_boundary - datetime.now()).total_seconds())
        
    def get_next_boundary_time(self) -> datetime:
        """Alias for get_next_boundary for API compatibility"""
        return self.get_next_boundary()
    
    def is_near_boundary(self, seconds_threshold: int = 30, minutes_ahead: int = None) -> bool:
        """Check if we're near a 15-minute boundary"""
        # Support both seconds_threshold and minutes_ahead parameters
        if minutes_ahead is not None:
            seconds_threshold = minutes_ahead * 60
        return self.seconds_to_boundary() <= seconds_threshold
        
    async def start(self):
        """Start the timing manager"""
        self.running = True
        logger.info("Starting 15-minute boundary timing manager")
        
        while self.running:
            try:
                # Calculate time to next boundary
                seconds_to_wait = self.seconds_to_boundary()
                
                # Update current slot
                self.current_slot = self.get_current_slot()
                
                # Log countdown
                if seconds_to_wait <= 60:
                    logger.info(f"⏰ {seconds_to_wait}s to next 15-min boundary")
                    
                # Wait strategically
                if seconds_to_wait > 60:
                    await asyncio.sleep(seconds_to_wait - 60)
                elif seconds_to_wait > 10:
                    await asyncio.sleep(10)
                else:
                    await asyncio.sleep(1)
                    
                # Trigger boundary event if we're at the boundary
                if seconds_to_wait <= 1:
                    await self._trigger_boundary_event()
                    # Wait for the boundary to pass
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"Timing manager error: {e}")
                await asyncio.sleep(1)
                
    async def _trigger_boundary_event(self):
        """Trigger callbacks for boundary event"""
        event = BoundaryEvent(
            timestamp=datetime.now(),
            slot=self.get_current_slot(),
            next_boundary=self.get_next_boundary(),
            seconds_to_next=self.seconds_to_boundary()
        )
        
        logger.info(f"🔔 15-minute boundary reached! Slot: {event.slot.name}")
        
        for callback in self.callbacks:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Boundary callback error: {e}")
                
    def register_position(self, position_id: str, entry_time: datetime):
        """Register a position with its entry slot"""
        minute = entry_time.minute
        if minute < 15:
            slot = TimeSlot.SLOT_1
        elif minute < 30:
            slot = TimeSlot.SLOT_2
        elif minute < 45:
            slot = TimeSlot.SLOT_3
        else:
            slot = TimeSlot.SLOT_4
            
        self.positions_by_slot[slot].append({
            'id': position_id,
            'entry_time': entry_time,
            'slot': slot
        })
        
    def get_preferred_exit_slots(self, entry_slot: TimeSlot) -> List[TimeSlot]:
        """Get preferred exit slots based on entry slot"""
        # Based on analysis: exits typically 1-3 slots after entry
        slot_number = entry_slot.value
        preferred = []
        
        for i in [1, 2, 3]:
            exit_slot_num = (slot_number + i - 1) % 4 + 1
            preferred.append(TimeSlot(exit_slot_num))
            
        return preferred
    
    def should_exit_original_strategy(self, position_entry_time: datetime, 
                                     current_pnl_percent: float = 0) -> bool:
        """
        Original strategy exit logic - 100% boundary compliance
        Returns True if position should exit at current boundary
        """
        if self.strategy_mode != "original":
            return False
            
        now = datetime.now()
        holding_minutes = (now - position_entry_time).total_seconds() / 60
        
        # Minimum holding time check
        if holding_minutes < 15:
            return False
            
        # Maximum holding time - force exit
        if holding_minutes >= 180:
            return True
            
        # Check if we're at a boundary (within 1 minute tolerance)
        minute = now.minute
        at_boundary = any(abs(minute - b) <= 1 for b in [0, 15, 30, 45]) or minute >= 59
        
        if not at_boundary:
            return False
            
        # Determine exit probability based on slot and holding time
        current_slot = self.get_current_slot()
        slot_weight = self.original_slot_weights.get(current_slot, 0.25)
        
        # Increase exit probability with holding time
        time_factor = min(holding_minutes / 72.65, 2.0)  # 72.65 is target avg
        
        # Higher exit probability if profitable
        profit_factor = 1.2 if current_pnl_percent > 0 else 1.0
        
        # Calculate final exit probability
        exit_probability = min(slot_weight * time_factor * profit_factor, 1.0)
        
        # For original strategy, we want deterministic behavior
        # Exit if holding time exceeds target or profitable
        if holding_minutes >= 72.65 or current_pnl_percent > 0.5:
            return True
            
        # Otherwise use slot-based logic
        if holding_minutes >= 30 and current_slot in [TimeSlot.SLOT_2, TimeSlot.SLOT_4]:
            return True  # Preferred exit slots
            
        return holding_minutes >= 45  # Exit after 45 minutes minimum
        
    async def stop(self):
        """Stop the timing manager"""
        self.running = False
        logger.info("Stopped timing manager")


class PrecisionTimer:
    """High-precision timer for critical timing operations"""
    
    def __init__(self):
        self.scheduled_tasks = []
        
    async def schedule_at_boundary(self, callback: Callable, lead_time_seconds: int = 1):
        """Schedule task to run just before boundary"""
        next_boundary = self.get_next_15min_boundary()
        target_time = next_boundary - timedelta(seconds=lead_time_seconds)
        
        wait_seconds = (target_time - datetime.now()).total_seconds()
        if wait_seconds > 0:
            await asyncio.sleep(wait_seconds)
            await callback()
            
    @staticmethod
    def get_next_15min_boundary() -> datetime:
        """Get exact next 15-minute boundary"""
        now = datetime.now()
        minute = now.minute
        next_minute = ((minute // 15) + 1) * 15
        
        if next_minute >= 60:
            return now.replace(hour=(now.hour + 1) % 24, minute=0, second=0, microsecond=0)
        else:
            return now.replace(minute=next_minute, second=0, microsecond=0)