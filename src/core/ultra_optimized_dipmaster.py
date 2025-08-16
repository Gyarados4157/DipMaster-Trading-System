#!/usr/bin/env python3
"""
Ultra-Optimized DipMaster Strategy - 同步实施短期+中期优化
=====================================================

核心改进：
1. 短期优化：信号参数收紧、止损优化、时间过滤增强、置信度提升
2. 中期优化：市场状态自适应、实时相关性控制、执行优化、币种池扩展
3. 新币种池：避开BTC/ETH，选择20+个优质标的

目标：将胜率从55%提升至75%+，评分从40.8提升至80+

Author: DipMaster Ultra Team
Date: 2025-08-15
Version: 2.0.0 (Ultra Edition)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import ta
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """增强市场状态识别"""
    BULL_TREND = "bull_trend"         # 强牛市 - 谨慎交易
    BEAR_TREND = "bear_trend"         # 强熊市 - 暂停交易  
    SIDEWAYS = "sideways"             # 横盘震荡 - 最佳环境
    HIGH_VOLATILITY = "high_vol"      # 高波动 - 降低仓位
    LOW_VOLATILITY = "low_vol"        # 低波动 - 增加仓位
    ACCUMULATION = "accumulation"     # 积累阶段 - 适合抄底
    DISTRIBUTION = "distribution"     # 分发阶段 - 避免交易
    RECOVERY = "recovery"             # 反弹恢复 - 谨慎乐观


@dataclass
class UltraSignalConfig:
    """超级优化信号配置"""
    # === 短期优化：信号参数收紧 ===
    rsi_optimal_range: Tuple[int, int] = (38, 45)    # 从35-42收紧至38-45
    rsi_acceptable_range: Tuple[int, int] = (30, 50) # 可接受范围
    
    volume_min_multiplier: float = 2.0               # 从1.5提升至2.0
    volume_optimal_multiplier: float = 3.0           # 最优成交量倍数
    
    # === 风险管理优化 ===
    emergency_stop_loss: float = 0.003               # 从0.5%收紧至0.3%
    quick_profit_target: float = 0.005               # 快速止盈0.5%
    trailing_stop_distance: float = 0.002            # 追踪止损0.2%
    
    # === 时间过滤增强 ===
    forbidden_hours: List[int] = field(default_factory=lambda: [0, 1, 12, 13, 18, 23])  # 增加禁用时段
    optimal_hours: List[int] = field(default_factory=lambda: [3, 8, 10, 14, 20, 21])     # 最佳时段
    
    # === 置信度提升 ===
    min_signal_confidence: float = 0.65              # 从0.5提升至0.65
    min_grade_threshold: str = "B"                    # 最低B级信号
    
    # === 市场状态过滤 ===
    allowed_regimes: Set[MarketRegime] = field(default_factory=lambda: {
        MarketRegime.SIDEWAYS,
        MarketRegime.LOW_VOLATILITY,
        MarketRegime.ACCUMULATION,
        MarketRegime.RECOVERY
    })


@dataclass  
class UltraSymbolPool:
    """扩展的优质币种池（排除BTC/ETH）"""
    # 当前已有数据的币种
    current_symbols: List[str] = field(default_factory=lambda: [
        "DOGEUSDT", "IOTAUSDT", "SOLUSDT", "SUIUSDT", 
        "ALGOUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "ICPUSDT"
    ])
    
    # 新增高流动性币种（需要下载数据）
    target_symbols: List[str] = field(default_factory=lambda: [
        "MATICUSDT", "DOTUSDT", "AVAXUSDT", "LINKUSDT", "UNIUSDT",
        "NEARUSDT", "FTMUSDT", "ATOMUSDT", "VETUSDT", "XLMUSDT",
        "HBARUSDT", "SANDUSDT", "MANAUSDT", "CHZUSDT", "ENJUSDT",
        "GALAUSDT", "AXSUSDT", "FLOWUSDT", "ARUSDT", "IMXUSDT"
    ])
    
    # 币种质量评级
    tier_1_symbols: Set[str] = field(default_factory=lambda: {
        "BNBUSDT", "ADAUSDT", "SOLUSDT", "MATICUSDT", "DOTUSDT", 
        "AVAXUSDT", "LINKUSDT", "NEARUSDT", "ATOMUSDT"
    })
    
    tier_2_symbols: Set[str] = field(default_factory=lambda: {
        "XRPUSDT", "DOGEUSDT", "UNIUSDT", "VETUSDT", "XLMUSDT",
        "SUIUSDT", "ICPUSDT", "ALGOUSDT", "FTMUSDT"
    })

    @property
    def all_symbols(self) -> List[str]:
        """所有币种池"""
        return self.current_symbols + self.target_symbols


class UltraMarketRegimeDetector:
    """超级市场状态检测器"""
    
    def __init__(self):
        self.regime_history = deque(maxlen=20)  # 保存最近20次检测结果
        self.regime_weights = {
            MarketRegime.SIDEWAYS: 1.0,
            MarketRegime.LOW_VOLATILITY: 0.9,
            MarketRegime.ACCUMULATION: 1.1,
            MarketRegime.RECOVERY: 0.8,
            MarketRegime.HIGH_VOLATILITY: 0.6,
            MarketRegime.BULL_TREND: 0.4,
            MarketRegime.BEAR_TREND: 0.0,
            MarketRegime.DISTRIBUTION: 0.2
        }
        
    def detect_regime(self, df: pd.DataFrame, symbol: str) -> Tuple[MarketRegime, float]:
        """检测市场状态"""
        if len(df) < 50:
            return MarketRegime.SIDEWAYS, 0.8
            
        # === 趋势分析 ===
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        
        trend_strength = (ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1]
        trend_consistency = (ema_20.iloc[-10:] > ema_50.iloc[-10:]).mean()
        
        # === 波动率分析 ===
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1]
        avg_volatility = returns.rolling(50).std().mean()
        vol_ratio = volatility / avg_volatility if avg_volatility > 0 else 1
        
        # === 成交量分析 ===
        volume_trend = df['volume'].rolling(10).mean().iloc[-1] / df['volume'].rolling(30).mean().iloc[-1]
        
        # === 价格位置分析 ===
        price_range_20 = df['high'].rolling(20).max() - df['low'].rolling(20).min()
        current_position = (df['close'].iloc[-1] - df['low'].rolling(20).min().iloc[-1]) / price_range_20.iloc[-1]
        
        # === 状态判断逻辑 ===
        regime = MarketRegime.SIDEWAYS  # 默认
        confidence = 0.5
        
        # 强趋势检测
        if abs(trend_strength) > 0.03 and trend_consistency > 0.8:
            if trend_strength > 0:
                regime = MarketRegime.BULL_TREND
                confidence = min(trend_strength * 20, 0.95)
            else:
                regime = MarketRegime.BEAR_TREND
                confidence = min(abs(trend_strength) * 20, 0.95)
                
        # 波动率状态
        elif vol_ratio > 2.0:
            regime = MarketRegime.HIGH_VOLATILITY
            confidence = min(vol_ratio / 3, 0.9)
        elif vol_ratio < 0.5:
            regime = MarketRegime.LOW_VOLATILITY
            confidence = 0.85
            
        # 积累/分发阶段
        elif volume_trend > 1.5 and current_position < 0.3:
            regime = MarketRegime.ACCUMULATION
            confidence = 0.8
        elif volume_trend > 1.5 and current_position > 0.7:
            regime = MarketRegime.DISTRIBUTION
            confidence = 0.75
            
        # 恢复阶段
        elif len(self.regime_history) > 0 and self.regime_history[-1] == MarketRegime.BEAR_TREND:
            if trend_strength > -0.01:  # 从熊市开始恢复
                regime = MarketRegime.RECOVERY
                confidence = 0.7
                
        # 横盘震荡（最佳环境）
        elif abs(trend_strength) < 0.02 and 0.4 <= vol_ratio <= 1.5:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.85
            
        # 更新历史记录
        self.regime_history.append(regime)
        
        return regime, confidence
        
    def get_regime_multiplier(self, regime: MarketRegime) -> float:
        """获取状态权重系数"""
        return self.regime_weights.get(regime, 0.5)


class UltraSignalGenerator:
    """超级信号生成器 - 集成所有优化"""
    
    def __init__(self, config: UltraSignalConfig = None):
        self.config = config or UltraSignalConfig()
        self.regime_detector = UltraMarketRegimeDetector()
        
        # 信号历史和过滤
        self.recent_signals: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
        self.correlation_matrix = pd.DataFrame()
        self.symbol_performance = defaultdict(lambda: {"trades": [], "avg_win_rate": 0.5})
        
        # 实时监控数据
        self.signal_stats = {
            "total_generated": 0,
            "filtered_out": 0,
            "grade_A_signals": 0,
            "grade_B_signals": 0,
            "market_regime_blocks": 0,
            "correlation_blocks": 0,
            "time_filter_blocks": 0
        }
        
    def generate_ultra_signal(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """生成超级优化信号"""
        if len(df) < 100:  # 需要足够的历史数据
            return None
            
        self.signal_stats["total_generated"] += 1
        
        # === Phase 1: 市场状态检测 ===
        regime, regime_confidence = self.regime_detector.detect_regime(df, symbol)
        
        if regime not in self.config.allowed_regimes:
            self.signal_stats["market_regime_blocks"] += 1
            logger.debug(f"{symbol}: Market regime {regime.value} not suitable - skipped")
            return None
            
        # === Phase 2: 核心技术分析 ===
        tech_analysis = self._enhanced_technical_analysis(df)
        if not tech_analysis["basic_conditions_met"]:
            self.signal_stats["filtered_out"] += 1
            return None
            
        # === Phase 3: 多重过滤系统 ===
        filters_result = self._apply_multi_layer_filters(symbol, df, tech_analysis)
        if not filters_result["passed"]:
            self.signal_stats["filtered_out"] += 1
            return None
            
        # === Phase 4: 时间和相关性过滤 ===
        if not self._time_and_correlation_filter(symbol):
            return None
            
        # === Phase 5: 置信度计算和评级 ===
        confidence_data = self._calculate_ultra_confidence(
            tech_analysis, filters_result, regime_confidence
        )
        
        final_confidence = confidence_data["total_confidence"]
        if final_confidence < self.config.min_signal_confidence:
            self.signal_stats["filtered_out"] += 1
            return None
            
        signal_grade = self._get_signal_grade(final_confidence)
        if signal_grade not in ["A", "B"] and self.config.min_grade_threshold == "B":
            self.signal_stats["filtered_out"] += 1
            return None
            
        # === Phase 6: 动态风险参数 ===
        risk_params = self._calculate_dynamic_risk_params(
            tech_analysis, regime, final_confidence
        )
        
        # === Phase 7: 构建最终信号 ===
        signal = {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "price": df['close'].iloc[-1],
            "signal_type": "ULTRA_DIP_BUY",
            
            # 置信度和评级
            "confidence": final_confidence,
            "grade": signal_grade,
            "regime": regime.value,
            "regime_confidence": regime_confidence,
            
            # 技术指标
            "rsi": tech_analysis["rsi"],
            "volume_ratio": tech_analysis["volume_ratio"],
            "bb_position": tech_analysis["bb_position"],
            
            # 风险管理参数
            "stop_loss_price": risk_params["stop_loss_price"],
            "take_profit_levels": risk_params["take_profit_levels"],
            "max_holding_minutes": risk_params["max_holding_minutes"],
            
            # 调试信息
            "filters_passed": filters_result["filters_passed"],
            "confidence_breakdown": confidence_data["breakdown"],
            "market_analysis": {
                "trend_strength": tech_analysis.get("trend_strength", 0),
                "momentum_score": tech_analysis.get("momentum_score", 0),
                "structure_score": tech_analysis.get("structure_score", 0)
            }
        }
        
        # 更新统计
        if signal_grade == "A":
            self.signal_stats["grade_A_signals"] += 1
        elif signal_grade == "B":
            self.signal_stats["grade_B_signals"] += 1
            
        # 记录信号历史
        self.recent_signals[symbol].append({
            "timestamp": datetime.now(),
            "confidence": final_confidence,
            "grade": signal_grade,
            "regime": regime.value
        })
        
        logger.info(f"🎯 Ultra Signal Generated: {symbol} @ {df['close'].iloc[-1]:.4f} "
                   f"[Grade: {signal_grade}, Confidence: {final_confidence:.2f}, Regime: {regime.value}]")
        
        return signal
        
    def _enhanced_technical_analysis(self, df: pd.DataFrame) -> Dict:
        """增强技术分析"""
        current_row = df.iloc[-1]
        
        # === RSI分析 ===
        rsi_series = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        rsi = rsi_series.iloc[-1]
        
        # === 成交量分析 ===
        volume_ma = df['volume'].rolling(20).mean()
        volume_ratio = current_row['volume'] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
        
        # === 价格结构分析 ===
        # 布林带位置
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_lower = bb_middle - (bb_std * 2)
        bb_upper = bb_middle + (bb_std * 2)
        bb_position = (current_row['close'] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        
        # 趋势强度
        ema_10 = df['close'].ewm(span=10).mean()
        ema_20 = df['close'].ewm(span=20).mean()
        trend_strength = (ema_10.iloc[-1] - ema_20.iloc[-1]) / ema_20.iloc[-1]
        
        # === 动量分析 ===
        # 检查下跌动量是否在减缓
        recent_changes = df['close'].pct_change().tail(5)
        momentum_score = 0
        if len(recent_changes) >= 3:
            if recent_changes.iloc[-1] > recent_changes.iloc[-2]:  # 最新变化较小（下跌减缓）
                momentum_score += 0.3
            if recent_changes.iloc[-2] > recent_changes.iloc[-3]:  # 倒数第二次也在减缓
                momentum_score += 0.3
            if recent_changes.mean() < 0 and recent_changes.std() < 0.02:  # 整体下跌但波动减少
                momentum_score += 0.4
                
        # === 基础条件检查 ===
        basic_conditions_met = all([
            current_row['close'] < current_row['open'],  # 当前K线下跌
            rsi >= self.config.rsi_acceptable_range[0],  # RSI不能过低
            rsi <= self.config.rsi_acceptable_range[1],  # RSI不能过高
            volume_ratio >= 1.0,  # 成交量至少正常
            bb_position <= 0.3,   # 价格在布林带下方区域
        ])
        
        return {
            "rsi": rsi,
            "volume_ratio": volume_ratio,
            "bb_position": bb_position,
            "trend_strength": trend_strength,
            "momentum_score": momentum_score,
            "basic_conditions_met": basic_conditions_met,
            "current_price": current_row['close'],
            "candle_change": (current_row['close'] - current_row['open']) / current_row['open']
        }
        
    def _apply_multi_layer_filters(self, symbol: str, df: pd.DataFrame, tech_analysis: Dict) -> Dict:
        """应用多层过滤系统"""
        filters_passed = []
        total_score = 0
        
        rsi = tech_analysis["rsi"]
        volume_ratio = tech_analysis["volume_ratio"]
        bb_position = tech_analysis["bb_position"]
        momentum_score = tech_analysis["momentum_score"]
        
        # === Filter 1: RSI精确区间 ===
        rsi_score = 0
        if self.config.rsi_optimal_range[0] <= rsi <= self.config.rsi_optimal_range[1]:
            rsi_score = 30  # 最优RSI区间
            filters_passed.append(f"RSI_OPTIMAL_{rsi:.1f}")
        elif self.config.rsi_acceptable_range[0] <= rsi <= self.config.rsi_acceptable_range[1]:
            # 在可接受范围内，距离最优区间越近分数越高
            distance_to_optimal = min(
                abs(rsi - self.config.rsi_optimal_range[0]),
                abs(rsi - self.config.rsi_optimal_range[1])
            )
            rsi_score = max(15, 30 - distance_to_optimal * 2)
            filters_passed.append(f"RSI_ACCEPTABLE_{rsi:.1f}")
        else:
            rsi_score = 5  # 不在理想区间但不直接拒绝
            filters_passed.append(f"RSI_POOR_{rsi:.1f}")
            
        total_score += rsi_score
        
        # === Filter 2: 成交量确认 ===
        volume_score = 0
        if volume_ratio >= self.config.volume_optimal_multiplier:
            volume_score = 25  # 成交量激增
            filters_passed.append(f"VOLUME_SURGE_{volume_ratio:.1f}x")
        elif volume_ratio >= self.config.volume_min_multiplier:
            volume_score = 20  # 满足最小要求
            filters_passed.append(f"VOLUME_OK_{volume_ratio:.1f}x")
        elif volume_ratio >= 1.5:
            volume_score = 15  # 一般放量
            filters_passed.append(f"VOLUME_FAIR_{volume_ratio:.1f}x")
        else:
            volume_score = 5   # 成交量不足但不直接拒绝
            filters_passed.append(f"VOLUME_LOW_{volume_ratio:.1f}x")
            
        total_score += volume_score
        
        # === Filter 3: 布林带位置 ===
        bb_score = 0
        if bb_position <= 0.15:  # 非常接近下轨
            bb_score = 20
            filters_passed.append("BB_NEAR_LOWER")
        elif bb_position <= 0.3:  # 在下轨区域
            bb_score = 15
            filters_passed.append("BB_LOWER_ZONE")
        else:
            bb_score = 5
            filters_passed.append("BB_NOT_IDEAL")
            
        total_score += bb_score
        
        # === Filter 4: 动量确认 ===
        momentum_filter_score = momentum_score * 15  # 转换为评分
        total_score += momentum_filter_score
        if momentum_score > 0.7:
            filters_passed.append("MOMENTUM_STRONG")
        elif momentum_score > 0.4:
            filters_passed.append("MOMENTUM_OK")
        else:
            filters_passed.append("MOMENTUM_WEAK")
            
        # === Filter 5: 趋势过滤 ===
        trend_strength = tech_analysis["trend_strength"]
        trend_score = 0
        if -0.015 <= trend_strength <= 0.005:  # 轻微下跌到微涨最佳
            trend_score = 10
            filters_passed.append("TREND_IDEAL")
        elif trend_strength > -0.025:  # 不要过于弱势
            trend_score = 8
            filters_passed.append("TREND_OK")
        else:
            trend_score = 3
            filters_passed.append("TREND_WEAK")
            
        total_score += trend_score
        
        # 判断是否通过（总分100，60分及格）
        passed = total_score >= 60
        
        return {
            "passed": passed,
            "total_score": total_score,
            "filters_passed": filters_passed,
            "individual_scores": {
                "rsi_score": rsi_score,
                "volume_score": volume_score,
                "bb_score": bb_score,
                "momentum_score": momentum_filter_score,
                "trend_score": trend_score
            }
        }
        
    def _time_and_correlation_filter(self, symbol: str) -> bool:
        """时间和相关性过滤"""
        current_hour = datetime.now().hour
        
        # 时间过滤
        if current_hour in self.config.forbidden_hours:
            self.signal_stats["time_filter_blocks"] += 1
            return False
            
        # 相关性过滤（简化版本，实际需要实时价格数据）
        # 这里先跳过，实盘时可以集成实时相关性检查
        
        return True
        
    def _calculate_ultra_confidence(self, tech_analysis: Dict, filters_result: Dict, 
                                  regime_confidence: float) -> Dict:
        """计算综合置信度"""
        # 基础技术分析置信度（40%权重）
        tech_confidence = filters_result["total_score"] / 100 * 0.4
        
        # 市场状态置信度（30%权重）
        regime_weight = regime_confidence * 0.3
        
        # 历史表现调整（20%权重）
        symbol = tech_analysis.get("symbol", "UNKNOWN")
        symbol_perf = self.symbol_performance.get(symbol, {"avg_win_rate": 0.5})
        historical_weight = symbol_perf["avg_win_rate"] * 0.2
        
        # 信号强度调整（10%权重）
        signal_strength = min(tech_analysis.get("momentum_score", 0) * 2, 1.0) * 0.1
        
        total_confidence = tech_confidence + regime_weight + historical_weight + signal_strength
        total_confidence = max(0, min(total_confidence, 1.0))  # 限制在0-1之间
        
        return {
            "total_confidence": total_confidence,
            "breakdown": {
                "technical": tech_confidence,
                "regime": regime_weight,
                "historical": historical_weight,
                "strength": signal_strength
            }
        }
        
    def _get_signal_grade(self, confidence: float) -> str:
        """获取信号等级"""
        if confidence >= 0.8:
            return "A+"
        elif confidence >= 0.75:
            return "A"
        elif confidence >= 0.65:
            return "B"
        elif confidence >= 0.5:
            return "C"
        else:
            return "D"
            
    def _calculate_dynamic_risk_params(self, tech_analysis: Dict, regime: MarketRegime, 
                                     confidence: float) -> Dict:
        """计算动态风险参数"""
        current_price = tech_analysis["current_price"]
        
        # 基础止损根据置信度调整
        base_stop_loss = self.config.emergency_stop_loss
        if confidence > 0.8:
            base_stop_loss *= 0.8  # 高置信度可以适当放宽止损
        elif confidence < 0.65:
            base_stop_loss *= 1.2  # 低置信度收紧止损
            
        # 根据市场状态调整
        regime_multiplier = self.regime_detector.get_regime_multiplier(regime)
        adjusted_stop_loss = base_stop_loss * (2 - regime_multiplier)  # 恶劣环境收紧止损
        
        stop_loss_price = current_price * (1 - adjusted_stop_loss)
        
        # 分级止盈设置
        take_profit_levels = [
            {"level": 1, "profit_pct": 0.005, "exit_ratio": 0.3, "price": current_price * 1.005},  # 0.5%
            {"level": 2, "profit_pct": 0.012, "exit_ratio": 0.4, "price": current_price * 1.012},  # 1.2%
            {"level": 3, "profit_pct": 0.025, "exit_ratio": 0.3, "price": current_price * 1.025},  # 2.5%
        ]
        
        # 动态持仓时间
        max_holding = 60  # 基础60分钟
        if regime in [MarketRegime.SIDEWAYS, MarketRegime.LOW_VOLATILITY]:
            max_holding = 90  # 良好环境延长
        elif regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.BULL_TREND]:
            max_holding = 30  # 高风险环境缩短
            
        return {
            "stop_loss_price": stop_loss_price,
            "take_profit_levels": take_profit_levels,
            "max_holding_minutes": max_holding,
            "trailing_stop_distance": self.config.trailing_stop_distance
        }
        
    def get_optimization_stats(self) -> Dict:
        """获取优化统计数据"""
        total = self.signal_stats["total_generated"]
        if total == 0:
            return {"message": "No signals generated yet"}
            
        return {
            "总信号数": total,
            "过滤率": f"{self.signal_stats['filtered_out'] / total * 100:.1f}%",
            "A级信号比例": f"{self.signal_stats['grade_A_signals'] / max(total - self.signal_stats['filtered_out'], 1) * 100:.1f}%",
            "B级信号比例": f"{self.signal_stats['grade_B_signals'] / max(total - self.signal_stats['filtered_out'], 1) * 100:.1f}%",
            "市场状态过滤": self.signal_stats["market_regime_blocks"],
            "时间过滤": self.signal_stats["time_filter_blocks"],
            "相关性过滤": self.signal_stats["correlation_blocks"]
        }


class UltraRiskManager:
    """超级风险管理器 - 实时相关性控制"""
    
    def __init__(self):
        self.positions: Dict[str, Dict] = {}
        self.correlation_threshold = 0.6  # 相关性阈值
        self.max_portfolio_risk = 0.15    # 组合最大风险15%
        self.max_single_position = 0.05   # 单个仓位最大5%
        
        # 动态调整参数
        self.risk_budget_used = 0.0
        self.correlation_penalty_factor = 1.5
        
    def calculate_position_size(self, signal: Dict, portfolio_value: float, 
                              active_symbols: Set[str]) -> float:
        """基于风险的动态仓位计算"""
        
        # 基础仓位大小
        base_position_size = portfolio_value * 0.03  # 3%基础仓位
        
        # 根据信号质量调整
        confidence_multiplier = signal["confidence"] / 0.7  # 标准化到0.7基准
        quality_adjusted_size = base_position_size * confidence_multiplier
        
        # 相关性调整
        correlation_penalty = self._calculate_correlation_penalty(signal["symbol"], active_symbols)
        final_size = quality_adjusted_size * (1 - correlation_penalty)
        
        # 风险限制
        max_allowed = min(
            portfolio_value * self.max_single_position,
            portfolio_value * (self.max_portfolio_risk - self.risk_budget_used)
        )
        
        return min(final_size, max_allowed)
        
    def _calculate_correlation_penalty(self, symbol: str, active_symbols: Set[str]) -> float:
        """计算相关性惩罚"""
        if len(active_symbols) == 0:
            return 0.0
            
        # 简化版本：基于币种类别的相关性估算
        # 实盘中应该使用实时价格数据计算真实相关性
        
        symbol_category = self._get_symbol_category(symbol)
        penalty = 0.0
        
        for active_symbol in active_symbols:
            if self._get_symbol_category(active_symbol) == symbol_category:
                penalty += 0.2  # 同类别币种增加20%惩罚
                
        return min(penalty, 0.6)  # 最大60%惩罚
        
    def _get_symbol_category(self, symbol: str) -> str:
        """获取币种类别（简化分类）"""
        defi_coins = {"UNIUSDT", "LINKUSDT", "AAVEUSDT", "COMPUSDT"}
        layer1_coins = {"ADAUSDT", "SOLUSDT", "AVAXUSDT", "NEARUSDT", "DOTUSDT", "ATOMUSDT"}
        gaming_coins = {"AXSUSDT", "SANDUSDT", "MANAUSDT", "ENJUSDT", "GALAUSDT"}
        
        if symbol in defi_coins:
            return "DEFI"
        elif symbol in layer1_coins:
            return "LAYER1"
        elif symbol in gaming_coins:
            return "GAMING"
        else:
            return "OTHER"


# === 快速测试函数 ===

def test_ultra_optimization():
    """测试超级优化系统"""
    logger.info("🚀 Testing Ultra-Optimized DipMaster Strategy")
    
    # 创建配置
    config = UltraSignalConfig()
    signal_generator = UltraSignalGenerator(config)
    risk_manager = UltraRiskManager()
    symbol_pool = UltraSymbolPool()
    
    logger.info("✅ Ultra-Optimized Components Initialized")
    logger.info("📊 Configuration Summary:")
    logger.info(f"  • RSI Range: {config.rsi_optimal_range}")
    logger.info(f"  • Volume Multiplier: {config.volume_min_multiplier}")
    logger.info(f"  • Emergency Stop: {config.emergency_stop_loss*100:.1f}%")
    logger.info(f"  • Min Confidence: {config.min_signal_confidence}")
    logger.info(f"  • Symbol Pool Size: {len(symbol_pool.all_symbols)}")
    logger.info(f"  • Tier 1 Symbols: {len(symbol_pool.tier_1_symbols)}")
    
    return {
        "signal_generator": signal_generator,
        "risk_manager": risk_manager,
        "symbol_pool": symbol_pool,
        "config": config
    }

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 运行测试
    components = test_ultra_optimization()
    logger.info("🎉 Ultra-Optimized DipMaster Strategy Ready!")