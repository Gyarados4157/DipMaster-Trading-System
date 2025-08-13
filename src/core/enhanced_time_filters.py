"""
Enhanced Time-Based Filter System - Phase 5 Optimization  
增强时间过滤系统：基于时间模式优化交易时机

核心功能：
1. 交易时段分析：亚洲/欧洲/美洲时段
2. 星期效应：周一到周日的表现差异
3. 小时级模式：24小时内的最佳交易窗口
4. 市场开闭盘效应：开盘/收盘前后的行为
5. 节假日过滤：特殊日期的风险控制
6. 成交量时间模式：基于成交量的时间权重
7. 波动率时间分布：不同时间的波动特征

目标：识别DipMaster策略的最优交易时间窗口
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, time, date
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import pytz
from collections import defaultdict
import calendar

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """交易时段"""
    ASIAN = "asian"           # 亚洲时段 (UTC 0-8)
    EUROPEAN = "european"     # 欧洲时段 (UTC 7-15) 
    AMERICAN = "american"     # 美洲时段 (UTC 14-22)
    OVERLAP_AE = "asian_european"     # 亚欧重叠 (UTC 7-8)
    OVERLAP_EA = "european_american"  # 欧美重叠 (UTC 14-15)
    OFF_PEAK = "off_peak"    # 非高峰时段


class DayOfWeek(Enum):
    """星期"""
    MONDAY = 0
    TUESDAY = 1  
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


@dataclass
class TimeWindowScore:
    """时间窗口评分"""
    window_name: str
    win_rate: float = 0.0
    avg_profit: float = 0.0
    trade_count: int = 0
    avg_holding_time: float = 0.0
    volatility: float = 0.0
    volume_ratio: float = 1.0
    score: float = 0.0
    confidence: float = 0.0
    
    @property
    def grade(self) -> str:
        """评分等级"""
        if self.score >= 80:
            return "A"
        elif self.score >= 65:
            return "B"
        elif self.score >= 50:
            return "C"
        else:
            return "D"


@dataclass 
class TimePattern:
    """时间模式分析结果"""
    best_sessions: List[TradingSession]
    best_days: List[DayOfWeek]
    best_hours: List[int]
    worst_sessions: List[TradingSession]
    worst_days: List[DayOfWeek]
    worst_hours: List[int]
    session_scores: Dict[TradingSession, TimeWindowScore]
    day_scores: Dict[DayOfWeek, TimeWindowScore]
    hour_scores: Dict[int, TimeWindowScore]
    overall_recommendation: str


class EnhancedTimeFilter:
    """增强时间过滤器"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # === 时区设置 ===
        self.timezone = pytz.UTC
        self.local_timezone = pytz.timezone(self.config.get('local_timezone', 'UTC'))
        
        # === 时段定义 (UTC时间) ===
        self.session_hours = {
            TradingSession.ASIAN: (0, 8),
            TradingSession.EUROPEAN: (7, 15),
            TradingSession.AMERICAN: (14, 22),
            TradingSession.OVERLAP_AE: (7, 8),
            TradingSession.OVERLAP_EA: (14, 15),
            TradingSession.OFF_PEAK: (22, 24)  # 包括22-24和0-7的非重叠部分
        }
        
        # === 分析参数 ===
        self.min_trades_for_analysis = 10      # 最少交易次数
        self.analysis_lookback_days = 90       # 分析回看天数
        self.confidence_threshold = 0.6        # 置信度阈值
        
        # === 过滤阈值 ===
        self.min_session_score = 50            # 最低时段分数
        self.min_day_score = 45                # 最低日期分数
        self.min_hour_score = 40               # 最低小时分数
        
        # === 特殊日期 ===
        self.market_holidays = self._load_market_holidays()
        self.pre_holiday_days = set()           # 节前交易日
        self.post_holiday_days = set()          # 节后交易日
        
        # === 数据存储 ===
        self.trade_history: List[Dict] = []
        self.time_patterns: TimePattern = None
        self.last_analysis_time: datetime = datetime.now()
        self.session_cache: Dict = {}
        
        # === 实时时间状态 ===
        self.current_session_score = 100
        self.current_day_score = 100
        self.current_hour_score = 100
        
    def _load_market_holidays(self) -> Set[date]:
        """加载市场假期"""
        # 主要的国际假期（影响加密货币市场）
        holidays = set()
        current_year = datetime.now().year
        
        # 新年
        holidays.add(date(current_year, 1, 1))
        
        # 圣诞节
        holidays.add(date(current_year, 12, 25))
        
        # 感恩节 (美国 - 11月第四个周四)
        thanksgiving = self._get_nth_weekday(current_year, 11, 3, 4)  # 第4个周四
        holidays.add(thanksgiving)
        
        # 独立日 (美国)
        holidays.add(date(current_year, 7, 4))
        
        # 劳动节 (美国 - 9月第一个周一)
        labor_day = self._get_nth_weekday(current_year, 9, 0, 1)  # 第1个周一
        holidays.add(labor_day)
        
        return holidays
        
    def _get_nth_weekday(self, year: int, month: int, weekday: int, n: int) -> date:
        """获取某月第n个星期几"""
        first_day = date(year, month, 1)
        first_weekday = first_day.weekday()
        
        # 计算第一个目标星期几
        days_ahead = weekday - first_weekday
        if days_ahead < 0:
            days_ahead += 7
            
        # 第n个目标星期几
        target_day = 1 + days_ahead + (n - 1) * 7
        return date(year, month, target_day)
        
    def update_trade_history(self, trade_result: Dict):
        """更新交易历史"""
        # 确保有时间戳
        if 'timestamp' not in trade_result:
            trade_result['timestamp'] = datetime.now()
        elif isinstance(trade_result['timestamp'], str):
            trade_result['timestamp'] = pd.to_datetime(trade_result['timestamp'])
            
        self.trade_history.append(trade_result)
        
        # 限制历史长度
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-800:]
            
        # 触发重新分析
        if len(self.trade_history) % 10 == 0:  # 每10笔交易重新分析
            self.analyze_time_patterns()
            
    def get_trading_session(self, dt: datetime) -> TradingSession:
        """获取交易时段"""
        # 转换为UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=self.timezone)
        elif dt.tzinfo != self.timezone:
            dt = dt.astimezone(self.timezone)
            
        hour = dt.hour
        
        # 检查重叠时段
        if 7 <= hour < 8:
            return TradingSession.OVERLAP_AE
        elif 14 <= hour < 15:
            return TradingSession.OVERLAP_EA
        elif 0 <= hour < 7:
            return TradingSession.ASIAN
        elif 8 <= hour < 14:
            return TradingSession.EUROPEAN
        elif 15 <= hour < 22:
            return TradingSession.AMERICAN
        else:  # 22-24
            return TradingSession.OFF_PEAK
            
    def is_market_holiday(self, dt: datetime) -> bool:
        """检查是否为市场假期"""
        return dt.date() in self.market_holidays
        
    def is_pre_post_holiday(self, dt: datetime) -> Tuple[bool, bool]:
        """检查是否为节前/节后交易日"""
        current_date = dt.date()
        
        # 检查节前（假期前一个工作日）
        is_pre_holiday = False
        for holiday in self.market_holidays:
            # 假期前1-3天内的工作日
            for days_before in range(1, 4):
                check_date = holiday - timedelta(days=days_before)
                if check_date == current_date and check_date.weekday() < 5:  # 工作日
                    is_pre_holiday = True
                    break
                    
        # 检查节后（假期后一个工作日）
        is_post_holiday = False  
        for holiday in self.market_holidays:
            # 假期后1-3天内的工作日
            for days_after in range(1, 4):
                check_date = holiday + timedelta(days=days_after)
                if check_date == current_date and check_date.weekday() < 5:  # 工作日
                    is_post_holiday = True
                    break
                    
        return is_pre_holiday, is_post_holiday
        
    def analyze_time_patterns(self) -> TimePattern:
        """分析时间模式"""
        if len(self.trade_history) < self.min_trades_for_analysis:
            logger.warning(f"交易历史不足，需要至少{self.min_trades_for_analysis}笔交易")
            return TimePattern([], [], [], [], [], [], {}, {}, {}, "数据不足")
            
        # 过滤最近的交易
        cutoff_date = datetime.now() - timedelta(days=self.analysis_lookback_days)
        recent_trades = [
            t for t in self.trade_history
            if t['timestamp'] >= cutoff_date
        ]
        
        if len(recent_trades) < self.min_trades_for_analysis:
            recent_trades = self.trade_history[-self.min_trades_for_analysis:]
            
        # 分析各个时间维度
        session_scores = self._analyze_sessions(recent_trades)
        day_scores = self._analyze_days(recent_trades)  
        hour_scores = self._analyze_hours(recent_trades)
        
        # 排序找出最佳和最差时间段
        best_sessions = sorted(session_scores.items(), key=lambda x: x[1].score, reverse=True)
        best_days = sorted(day_scores.items(), key=lambda x: x[1].score, reverse=True)
        best_hours = sorted(hour_scores.items(), key=lambda x: x[1].score, reverse=True)
        
        # 生成建议
        recommendation = self._generate_recommendation(session_scores, day_scores, hour_scores)
        
        self.time_patterns = TimePattern(
            best_sessions=[s[0] for s in best_sessions[:2]],
            best_days=[d[0] for d in best_days[:3]],
            best_hours=[h[0] for h in best_hours[:6]],
            worst_sessions=[s[0] for s in best_sessions[-2:]],
            worst_days=[d[0] for d in best_days[-2:]],
            worst_hours=[h[0] for h in best_hours[-6:]],
            session_scores={s[0]: s[1] for s in best_sessions},
            day_scores={d[0]: d[1] for d in best_days},
            hour_scores={h[0]: h[1] for h in best_hours},
            overall_recommendation=recommendation
        )
        
        self.last_analysis_time = datetime.now()
        logger.info(f"时间模式分析完成，基于{len(recent_trades)}笔交易")
        
        return self.time_patterns
        
    def _analyze_sessions(self, trades: List[Dict]) -> Dict[TradingSession, TimeWindowScore]:
        """分析交易时段表现"""
        session_data = defaultdict(list)
        
        # 按时段分组
        for trade in trades:
            session = self.get_trading_session(trade['timestamp'])
            session_data[session].append(trade)
            
        session_scores = {}
        
        for session, session_trades in session_data.items():
            if len(session_trades) < 3:  # 至少3笔交易
                continue
                
            # 计算指标
            wins = [t for t in session_trades if t.get('pnl_percent', 0) > 0]
            win_rate = len(wins) / len(session_trades) * 100
            
            avg_profit = np.mean([t.get('pnl_percent', 0) for t in session_trades])
            avg_holding = np.mean([t.get('holding_minutes', 0) for t in session_trades])
            
            profit_std = np.std([t.get('pnl_percent', 0) for t in session_trades])
            volatility = profit_std if profit_std > 0 else 1.0
            
            # 综合评分
            score = (
                win_rate * 0.4 +                    # 胜率权重40%
                max(0, avg_profit * 10) * 0.3 +     # 平均盈利权重30%
                max(0, (90 - avg_holding) / 90 * 100) * 0.2 +  # 持仓时间权重20%
                max(0, (2 - volatility) / 2 * 100) * 0.1       # 稳定性权重10%
            )
            
            session_scores[session] = TimeWindowScore(
                window_name=session.value,
                win_rate=win_rate,
                avg_profit=avg_profit,
                trade_count=len(session_trades),
                avg_holding_time=avg_holding,
                volatility=volatility,
                score=score,
                confidence=min(1.0, len(session_trades) / 20)  # 基于样本量的置信度
            )
            
        return session_scores
        
    def _analyze_days(self, trades: List[Dict]) -> Dict[DayOfWeek, TimeWindowScore]:
        """分析星期表现"""
        day_data = defaultdict(list)
        
        # 按星期分组
        for trade in trades:
            day_of_week = DayOfWeek(trade['timestamp'].weekday())
            day_data[day_of_week].append(trade)
            
        day_scores = {}
        
        for day, day_trades in day_data.items():
            if len(day_trades) < 2:
                continue
                
            wins = [t for t in day_trades if t.get('pnl_percent', 0) > 0]
            win_rate = len(wins) / len(day_trades) * 100
            
            avg_profit = np.mean([t.get('pnl_percent', 0) for t in day_trades])
            avg_holding = np.mean([t.get('holding_minutes', 0) for t in day_trades])
            
            profit_std = np.std([t.get('pnl_percent', 0) for t in day_trades])
            volatility = profit_std if profit_std > 0 else 1.0
            
            # 综合评分（与时段评分公式相同）
            score = (
                win_rate * 0.4 +
                max(0, avg_profit * 10) * 0.3 +
                max(0, (90 - avg_holding) / 90 * 100) * 0.2 +
                max(0, (2 - volatility) / 2 * 100) * 0.1
            )
            
            day_scores[day] = TimeWindowScore(
                window_name=day.name,
                win_rate=win_rate,
                avg_profit=avg_profit,
                trade_count=len(day_trades),
                avg_holding_time=avg_holding,
                volatility=volatility,
                score=score,
                confidence=min(1.0, len(day_trades) / 15)
            )
            
        return day_scores
        
    def _analyze_hours(self, trades: List[Dict]) -> Dict[int, TimeWindowScore]:
        """分析小时表现"""
        hour_data = defaultdict(list)
        
        # 按小时分组（UTC）
        for trade in trades:
            hour = trade['timestamp'].hour
            hour_data[hour].append(trade)
            
        hour_scores = {}
        
        for hour, hour_trades in hour_data.items():
            if len(hour_trades) < 2:
                continue
                
            wins = [t for t in hour_trades if t.get('pnl_percent', 0) > 0]
            win_rate = len(wins) / len(hour_trades) * 100 if hour_trades else 0
            
            avg_profit = np.mean([t.get('pnl_percent', 0) for t in hour_trades])
            avg_holding = np.mean([t.get('holding_minutes', 0) for t in hour_trades])
            
            profit_std = np.std([t.get('pnl_percent', 0) for t in hour_trades])
            volatility = profit_std if profit_std > 0 else 1.0
            
            score = (
                win_rate * 0.4 +
                max(0, avg_profit * 10) * 0.3 +
                max(0, (90 - avg_holding) / 90 * 100) * 0.2 +
                max(0, (2 - volatility) / 2 * 100) * 0.1
            )
            
            hour_scores[hour] = TimeWindowScore(
                window_name=f"{hour:02d}:00-{hour+1:02d}:00 UTC",
                win_rate=win_rate,
                avg_profit=avg_profit,
                trade_count=len(hour_trades),
                avg_holding_time=avg_holding,
                volatility=volatility,
                score=score,
                confidence=min(1.0, len(hour_trades) / 8)
            )
            
        return hour_scores
        
    def _generate_recommendation(self, session_scores: Dict, day_scores: Dict, hour_scores: Dict) -> str:
        """生成交易建议"""
        recommendations = []
        
        # 最佳时段建议
        if session_scores:
            best_session = max(session_scores.items(), key=lambda x: x[1].score)
            if best_session[1].score >= 60:
                recommendations.append(f"最佳交易时段：{best_session[0].value}")
                
        # 最佳日期建议  
        if day_scores:
            best_day = max(day_scores.items(), key=lambda x: x[1].score)
            if best_day[1].score >= 55:
                recommendations.append(f"最佳交易日：{best_day[0].name}")
                
        # 最佳小时建议
        if hour_scores:
            top_hours = sorted(hour_scores.items(), key=lambda x: x[1].score, reverse=True)[:3]
            good_hours = [h[0] for h in top_hours if h[1].score >= 50]
            if good_hours:
                recommendations.append(f"最佳交易小时：{good_hours}")
                
        if not recommendations:
            return "暂无明显时间偏好，建议继续收集数据"
            
        return "; ".join(recommendations)
        
    def should_trade_now(self, current_time: datetime = None) -> Tuple[bool, float, str]:
        """判断当前时间是否适合交易"""
        if current_time is None:
            current_time = datetime.now()
            
        # 确保时间模式已分析
        if self.time_patterns is None:
            if len(self.trade_history) >= self.min_trades_for_analysis:
                self.analyze_time_patterns()
            else:
                return True, 80.0, "数据不足，默认允许交易"
                
        # 假期检查
        if self.is_market_holiday(current_time):
            return False, 0.0, "市场假期，暂停交易"
            
        is_pre_holiday, is_post_holiday = self.is_pre_post_holiday(current_time)
        holiday_penalty = 0
        if is_pre_holiday:
            holiday_penalty = 20
        elif is_post_holiday:
            holiday_penalty = 10
            
        # 获取当前时间各维度评分
        current_session = self.get_trading_session(current_time)
        current_day = DayOfWeek(current_time.weekday())
        current_hour = current_time.hour
        
        # 计算评分
        session_score = 60  # 默认分数
        day_score = 60
        hour_score = 60
        
        if self.time_patterns:
            if current_session in self.time_patterns.session_scores:
                session_score = self.time_patterns.session_scores[current_session].score
                
            if current_day in self.time_patterns.day_scores:
                day_score = self.time_patterns.day_scores[current_day].score
                
            if current_hour in self.time_patterns.hour_scores:
                hour_score = self.time_patterns.hour_scores[current_hour].score
                
        # 综合评分
        total_score = (session_score * 0.4 + day_score * 0.3 + hour_score * 0.3) - holiday_penalty
        
        # 决策逻辑
        if total_score >= 65:
            decision = True
            reason = f"优质交易时间 [时段:{session_score:.0f} 日期:{day_score:.0f} 小时:{hour_score:.0f}]"
        elif total_score >= 45:
            decision = True
            reason = f"一般交易时间 [时段:{session_score:.0f} 日期:{day_score:.0f} 小时:{hour_score:.0f}]"
        else:
            decision = False
            reason = f"不建议交易时间 [时段:{session_score:.0f} 日期:{day_score:.0f} 小时:{hour_score:.0f}]"
            
        if holiday_penalty > 0:
            reason += f" (节假日影响-{holiday_penalty}分)"
            
        return decision, total_score, reason
        
    def get_time_analysis_report(self) -> Dict:
        """获取时间分析报告"""
        if self.time_patterns is None:
            return {'message': 'No time pattern analysis available'}
            
        report = {
            'analysis_date': self.last_analysis_time.isoformat(),
            'trade_count': len(self.trade_history),
            'best_sessions': [s.value for s in self.time_patterns.best_sessions],
            'best_days': [d.name for d in self.time_patterns.best_days],
            'best_hours': self.time_patterns.best_hours,
            'worst_sessions': [s.value for s in self.time_patterns.worst_sessions],
            'worst_days': [d.name for d in self.time_patterns.worst_days],
            'worst_hours': self.time_patterns.worst_hours,
            'recommendation': self.time_patterns.overall_recommendation
        }
        
        # 详细分数
        report['session_details'] = {
            s.value: {
                'score': score.score,
                'win_rate': score.win_rate,
                'avg_profit': score.avg_profit,
                'trade_count': score.trade_count,
                'grade': score.grade
            }
            for s, score in self.time_patterns.session_scores.items()
        }
        
        report['day_details'] = {
            d.name: {
                'score': score.score,
                'win_rate': score.win_rate,
                'avg_profit': score.avg_profit,
                'trade_count': score.trade_count,
                'grade': score.grade
            }
            for d, score in self.time_patterns.day_scores.items()
        }
        
        return report
        
    def get_current_time_score(self) -> Dict:
        """获取当前时间评分"""
        now = datetime.now()
        should_trade, score, reason = self.should_trade_now(now)
        
        return {
            'current_time': now.isoformat(),
            'should_trade': should_trade,
            'score': score,
            'reason': reason,
            'session': self.get_trading_session(now).value,
            'day_of_week': DayOfWeek(now.weekday()).name,
            'hour_utc': now.hour,
            'is_holiday': self.is_market_holiday(now),
            'is_pre_holiday': self.is_pre_post_holiday(now)[0],
            'is_post_holiday': self.is_pre_post_holiday(now)[1]
        }