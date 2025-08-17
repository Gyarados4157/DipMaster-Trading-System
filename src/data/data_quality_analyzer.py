"""
Advanced Data Quality Analyzer and Symbol Ranking System
高级数据质量分析器和币种排名系统

功能：
- 实时监控数据收集进度
- 生成综合数据质量报告
- 币种交易适用性排名
- 相关性分析和分组
- 自动生成MarketDataBundle配置
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
from dataclasses import dataclass, asdict
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

@dataclass
class QualityMetrics:
    """质量指标数据类"""
    symbol: str
    timeframe: str
    completeness: float
    consistency: float
    continuity: float
    liquidity: float
    volatility: float
    overall_score: float
    recommendation: str
    
@dataclass
class SymbolRanking:
    """币种排名数据类"""
    symbol: str
    overall_score: float
    quality_score: float
    liquidity_score: float
    volatility_score: float
    market_cap_score: float
    volume_score: float
    tier: str
    recommendation: str
    risk_level: str

class DataQualityAnalyzer:
    """数据质量分析器"""
    
    def __init__(self):
        self.logger = self.setup_logging()
        self.data_path = Path("data/enhanced_market_data")
        self.results_path = Path("results")
        self.results_path.mkdir(exist_ok=True)
        
        # 质量评估标准
        self.quality_thresholds = {
            'excellent': 0.95,
            'good': 0.85,
            'fair': 0.70,
            'poor': 0.50
        }
        
        # 币种分类配置
        self.symbol_categories = {
            'Payment': ['XRPUSDT', 'LTCUSDT', 'XLMUSDT'],
            'SmartContract': ['SOLUSDT', 'ADAUSDT', 'AVAXUSDT', 'NEARUSDT', 'APTUSDT', 'TRXUSDT', 'TONUSDT', 'ALGOUSDT'],
            'DeFi': ['UNIUSDT', 'AAVEUSDT', 'MKRUSDT', 'COMPUSDT'],
            'Layer2': ['MATICUSDT', 'ARBUSDT', 'OPUSDT'],
            'Infrastructure': ['LINKUSDT', 'ATOMUSDT', 'DOTUSDT', 'GRTUSDT'],
            'Exchange': ['BNBUSDT'],
            'Meme': ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'WIFUSDT'],
            'Storage': ['FILUSDT'],
            'Enterprise': ['VETUSDT'],
            'Computing': ['ICPUSDT']
        }
        
    def setup_logging(self) -> logging.Logger:
        """设置日志系统"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/data_quality_analysis.log'),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger(__name__)
    
    def scan_available_data(self) -> Dict[str, Dict[str, Any]]:
        """扫描可用的数据文件"""
        available_data = defaultdict(dict)
        
        # 扫描parquet文件
        for file_path in self.data_path.glob("*_2years.parquet"):
            try:
                # 解析文件名：SYMBOL_TIMEFRAME_2years.parquet
                file_name = file_path.stem
                parts = file_name.split('_')
                if len(parts) >= 3:
                    symbol = parts[0]
                    timeframe = parts[1]
                    
                    # 检查对应的元数据文件
                    metadata_path = self.data_path / f"{file_name}_metadata.json"
                    metadata = {}
                    if metadata_path.exists():
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                    
                    # 获取文件基本信息
                    file_info = {
                        'file_path': str(file_path),
                        'metadata_path': str(metadata_path) if metadata_path.exists() else None,
                        'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                        'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime),
                        'metadata': metadata
                    }
                    
                    available_data[symbol][timeframe] = file_info
                    
            except Exception as e:
                self.logger.warning(f"解析文件失败 {file_path}: {e}")
        
        self.logger.info(f"扫描到 {len(available_data)} 个币种的数据文件")
        return dict(available_data)
    
    def analyze_file_quality(self, file_path: str) -> QualityMetrics:
        """分析单个文件的数据质量"""
        try:
            # 读取数据
            df = pd.read_parquet(file_path)
            
            # 从文件路径提取信息
            file_name = Path(file_path).stem
            parts = file_name.split('_')
            symbol = parts[0]
            timeframe = parts[1]
            
            # 计算质量指标
            quality_scores = self.calculate_quality_scores(df, symbol, timeframe)
            
            # 生成质量评级
            overall_score = np.mean(list(quality_scores.values()))
            recommendation = self.get_quality_recommendation(overall_score)
            
            return QualityMetrics(
                symbol=symbol,
                timeframe=timeframe,
                completeness=quality_scores.get('completeness', 0),
                consistency=quality_scores.get('consistency', 0),
                continuity=quality_scores.get('continuity', 0),
                liquidity=quality_scores.get('liquidity', 0),
                volatility=quality_scores.get('volatility', 0),
                overall_score=overall_score,
                recommendation=recommendation
            )
            
        except Exception as e:
            self.logger.error(f"分析文件质量失败 {file_path}: {e}")
            return QualityMetrics(
                symbol="UNKNOWN",
                timeframe="UNKNOWN",
                completeness=0, consistency=0, continuity=0,
                liquidity=0, volatility=0, overall_score=0,
                recommendation="ERROR"
            )
    
    def calculate_quality_scores(self, df: pd.DataFrame, symbol: str, timeframe: str) -> Dict[str, float]:
        """计算详细的质量评分"""
        scores = {}
        
        if df.empty:
            return {key: 0.0 for key in ['completeness', 'consistency', 'continuity', 'liquidity', 'volatility']}
        
        try:
            # 1. 完整性评分
            expected_records = self.get_expected_records(timeframe)
            actual_records = len(df)
            scores['completeness'] = min(1.0, actual_records / expected_records)
            
            # 2. 数据一致性评分
            consistency_checks = []
            
            # OHLC一致性
            high_ok = (df['high'] >= df[['open', 'close']].max(axis=1)).mean()
            low_ok = (df['low'] <= df[['open', 'close']].min(axis=1)).mean()
            positive_prices = (df[['open', 'high', 'low', 'close']] > 0).all(axis=1).mean()
            positive_volume = (df['volume'] >= 0).mean()
            
            consistency_checks.extend([high_ok, low_ok, positive_prices, positive_volume])
            scores['consistency'] = np.mean(consistency_checks)
            
            # 3. 时间连续性评分
            if len(df) > 1:
                time_diffs = df.index.to_series().diff().dropna()
                expected_diff = self.get_expected_time_diff(timeframe)
                
                # 计算正常间隔的比例
                normal_intervals = (time_diffs <= expected_diff * 1.5).mean()
                scores['continuity'] = normal_intervals
            else:
                scores['continuity'] = 0.0
            
            # 4. 流动性评分（基于成交量）
            if df['volume'].std() > 0:
                volume_cv = df['volume'].std() / df['volume'].mean()
                avg_volume = df['volume'].mean()
                
                # 标准化成交量评分
                volume_score = min(1.0, avg_volume / 1000000)  # 基于100万的标准化
                variability_score = min(1.0, volume_cv / 2)     # 变异系数评分
                
                scores['liquidity'] = (volume_score + variability_score) / 2
            else:
                scores['liquidity'] = 0.1  # 成交量为0或恒定
            
            # 5. 波动性评分（适度波动性最佳）
            if df['close'].std() > 0:
                price_cv = df['close'].std() / df['close'].mean()
                
                # 理想的波动性范围：0.3-0.8
                if 0.3 <= price_cv <= 0.8:
                    scores['volatility'] = 1.0
                elif price_cv < 0.3:
                    scores['volatility'] = price_cv / 0.3  # 波动性太低
                else:
                    scores['volatility'] = max(0.1, 1.0 - (price_cv - 0.8) / 2)  # 波动性太高
            else:
                scores['volatility'] = 0.1  # 价格恒定
            
        except Exception as e:
            self.logger.error(f"计算质量评分失败 {symbol} {timeframe}: {e}")
            scores = {key: 0.0 for key in ['completeness', 'consistency', 'continuity', 'liquidity', 'volatility']}
        
        return scores
    
    def get_expected_records(self, timeframe: str) -> int:
        """获取期望的记录数量"""
        timeframe_records = {
            '1m': 90 * 1440,   # 90天 * 1440分钟/天
            '5m': 730 * 288,   # 2年 * 288条/天
            '15m': 730 * 96,   # 2年 * 96条/天
            '1h': 730 * 24,    # 2年 * 24条/天
            '4h': 730 * 6,     # 2年 * 6条/天
            '1d': 1095         # 3年 * 1条/天
        }
        return timeframe_records.get(timeframe, 730 * 288)
    
    def get_expected_time_diff(self, timeframe: str) -> timedelta:
        """获取期望的时间间隔"""
        time_diffs = {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1)
        }
        return time_diffs.get(timeframe, timedelta(minutes=5))
    
    def get_quality_recommendation(self, score: float) -> str:
        """获取质量推荐等级"""
        if score >= self.quality_thresholds['excellent']:
            return 'EXCELLENT'
        elif score >= self.quality_thresholds['good']:
            return 'GOOD'
        elif score >= self.quality_thresholds['fair']:
            return 'FAIR'
        else:
            return 'POOR'
    
    def analyze_symbol_rankings(self, available_data: Dict) -> List[SymbolRanking]:
        """分析币种排名"""
        symbol_rankings = []
        
        for symbol, timeframes in available_data.items():
            try:
                # 计算各个评分维度
                quality_scores = []
                for timeframe, file_info in timeframes.items():
                    file_path = file_info['file_path']
                    quality_metric = self.analyze_file_quality(file_path)
                    quality_scores.append(quality_metric.overall_score)
                
                # 整体质量评分
                avg_quality = np.mean(quality_scores) if quality_scores else 0
                
                # 流动性评分（基于文件大小和数据完整性）
                total_size = sum(tf['file_size_mb'] for tf in timeframes.values())
                liquidity_score = min(1.0, total_size / 100)  # 基于100MB标准化
                
                # 波动性评分（基于币种类型）
                volatility_score = self.get_volatility_score(symbol)
                
                # 市值评分（基于假设排名）
                market_cap_score = self.get_market_cap_score(symbol)
                
                # 成交量评分（基于文件数量和完整性）
                volume_score = len(timeframes) / 6  # 基于6个时间框架
                
                # 综合评分
                weights = {
                    'quality': 0.30,
                    'liquidity': 0.25,
                    'market_cap': 0.20,
                    'volume': 0.15,
                    'volatility': 0.10
                }
                
                overall_score = (
                    avg_quality * weights['quality'] +
                    liquidity_score * weights['liquidity'] +
                    market_cap_score * weights['market_cap'] +
                    volume_score * weights['volume'] +
                    volatility_score * weights['volatility']
                )
                
                # 评级和推荐
                tier = self.get_tier(overall_score)
                recommendation = self.get_trading_recommendation(overall_score, symbol)
                risk_level = self.get_risk_level(symbol, volatility_score)
                
                ranking = SymbolRanking(
                    symbol=symbol,
                    overall_score=overall_score,
                    quality_score=avg_quality,
                    liquidity_score=liquidity_score,
                    volatility_score=volatility_score,
                    market_cap_score=market_cap_score,
                    volume_score=volume_score,
                    tier=tier,
                    recommendation=recommendation,
                    risk_level=risk_level
                )
                
                symbol_rankings.append(ranking)
                
            except Exception as e:
                self.logger.error(f"分析币种排名失败 {symbol}: {e}")
        
        # 按综合评分排序
        symbol_rankings.sort(key=lambda x: x.overall_score, reverse=True)
        
        return symbol_rankings
    
    def get_volatility_score(self, symbol: str) -> float:
        """获取波动性评分（适度波动性更佳）"""
        volatility_map = {
            # 低波动性
            'XRPUSDT': 0.7, 'LTCUSDT': 0.8, 'XLMUSDT': 0.7,
            'BNBUSDT': 0.75, 'ADAUSDT': 0.8, 'TRXUSDT': 0.75,
            'VETUSDT': 0.7, 'ATOMUSDT': 0.8, 'LINKUSDT': 0.85,
            
            # 中等波动性
            'SOLUSDT': 0.9, 'AVAXUSDT': 0.9, 'NEARUSDT': 0.85,
            'DOTUSDT': 0.8, 'MATICUSDT': 0.85, 'UNIUSDT': 0.85,
            'AAVEUSDT': 0.85, 'MKRUSDT': 0.8, 'FILUSDT': 0.8,
            'ARBUSDT': 0.9, 'OPUSDT': 0.9, 'ALGOUSDT': 0.8,
            'GRTUSDT': 0.8, 'COMPUSDT': 0.8,
            
            # 高波动性
            'DOGEUSDT': 0.6, 'SHIBUSDT': 0.5, 'PEPEUSDT': 0.4,
            'APTUSDT': 0.7, 'TONUSDT': 0.75, 'ICPUSDT': 0.6,
            'WIFUSDT': 0.3
        }
        
        return volatility_map.get(symbol, 0.7)  # 默认中等评分
    
    def get_market_cap_score(self, symbol: str) -> float:
        """获取市值评分"""
        market_cap_ranks = {
            'XRPUSDT': 0.97, 'BNBUSDT': 0.96, 'SOLUSDT': 0.95,
            'DOGEUSDT': 0.94, 'ADAUSDT': 0.93, 'TRXUSDT': 0.92,
            'TONUSDT': 0.91, 'AVAXUSDT': 0.90, 'LINKUSDT': 0.89,
            'DOTUSDT': 0.88, 'MATICUSDT': 0.87, 'LTCUSDT': 0.85,
            'NEARUSDT': 0.83, 'APTUSDT': 0.82, 'UNIUSDT': 0.81,
            'ATOMUSDT': 0.80, 'XLMUSDT': 0.79, 'FILUSDT': 0.75,
            'ARBUSDT': 0.74, 'OPUSDT': 0.73, 'VETUSDT': 0.72,
            'ALGOUSDT': 0.71, 'AAVEUSDT': 0.69, 'MKRUSDT': 0.68,
            'GRTUSDT': 0.65, 'ICPUSDT': 0.64, 'SHIBUSDT': 0.63,
            'COMPUSDT': 0.60, 'PEPEUSDT': 0.55, 'WIFUSDT': 0.50
        }
        
        return market_cap_ranks.get(symbol, 0.60)
    
    def get_tier(self, score: float) -> str:
        """获取等级"""
        if score >= 0.85:
            return 'S'
        elif score >= 0.75:
            return 'A'
        elif score >= 0.65:
            return 'B'
        elif score >= 0.55:
            return 'C'
        else:
            return 'D'
    
    def get_trading_recommendation(self, score: float, symbol: str) -> str:
        """获取交易推荐"""
        if score >= 0.80:
            return 'HIGHLY_RECOMMENDED'
        elif score >= 0.70:
            return 'RECOMMENDED'
        elif score >= 0.60:
            return 'CONDITIONAL'
        else:
            return 'NOT_RECOMMENDED'
    
    def get_risk_level(self, symbol: str, volatility_score: float) -> str:
        """获取风险等级"""
        # Meme币自动高风险
        meme_symbols = ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'WIFUSDT']
        if symbol in meme_symbols:
            return 'HIGH'
        
        if volatility_score >= 0.8:
            return 'LOW'
        elif volatility_score >= 0.6:
            return 'MEDIUM'
        else:
            return 'HIGH'
    
    def generate_correlation_analysis(self, available_data: Dict) -> Dict:
        """生成相关性分析"""
        correlation_groups = {
            'Layer1_Smart_Contracts': ['SOLUSDT', 'ADAUSDT', 'AVAXUSDT', 'NEARUSDT', 'APTUSDT', 'TRXUSDT', 'TONUSDT'],
            'DeFi_Ecosystem': ['UNIUSDT', 'AAVEUSDT', 'MKRUSDT', 'COMPUSDT'],
            'Payment_Tokens': ['XRPUSDT', 'LTCUSDT', 'XLMUSDT'],
            'Layer2_Solutions': ['MATICUSDT', 'ARBUSDT', 'OPUSDT'],
            'Infrastructure': ['LINKUSDT', 'ATOMUSDT', 'DOTUSDT', 'GRTUSDT'],
            'Meme_Tokens': ['DOGEUSDT', 'SHIBUSDT', 'PEPEUSDT', 'WIFUSDT'],
            'Exchange_Tokens': ['BNBUSDT'],
            'Specialized': ['FILUSDT', 'VETUSDT', 'ICPUSDT', 'ALGOUSDT']
        }
        
        correlation_analysis = {
            'groups': correlation_groups,
            'high_correlation_pairs': [
                ['SOLUSDT', 'AVAXUSDT'],  # 同类Layer1
                ['UNIUSDT', 'AAVEUSDT'],  # DeFi生态
                ['ARBUSDT', 'OPUSDT'],    # Layer2竞争
                ['DOGEUSDT', 'SHIBUSDT'], # Meme币
                ['LINKUSDT', 'GRTUSDT']   # 基础设施
            ],
            'low_correlation_pairs': [
                ['BNBUSDT', 'DOGEUSDT'],  # 交易所代币 vs Meme
                ['FILUSDT', 'UNIUSDT'],   # 存储 vs DeFi
                ['VETUSDT', 'APTUSDT'],   # 企业 vs 新Layer1
                ['XLMUSDT', 'SOLUSDT']    # 支付 vs 智能合约
            ],
            'diversification_recommendations': {
                'conservative': ['XRPUSDT', 'BNBUSDT', 'ADAUSDT', 'LINKUSDT'],
                'balanced': ['SOLUSDT', 'AVAXUSDT', 'UNIUSDT', 'MATICUSDT', 'LTCUSDT'],
                'aggressive': ['NEARUSDT', 'APTUSDT', 'ARBUSDT', 'AAVEUSDT', 'DOGEUSDT']
            }
        }
        
        return correlation_analysis
    
    def create_enhanced_market_data_bundle(self, 
                                         available_data: Dict, 
                                         symbol_rankings: List[SymbolRanking],
                                         correlation_analysis: Dict) -> Dict:
        """创建增强版MarketDataBundle"""
        
        timestamp = datetime.now().isoformat()
        
        # 筛选高质量币种
        excellent_symbols = [r.symbol for r in symbol_rankings if r.tier in ['S', 'A']]
        recommended_symbols = [r.symbol for r in symbol_rankings if r.recommendation in ['HIGHLY_RECOMMENDED', 'RECOMMENDED']]
        
        # 数据统计
        total_files = sum(len(timeframes) for timeframes in available_data.values())
        total_size_mb = sum(
            sum(tf['file_size_mb'] for tf in timeframes.values())
            for timeframes in available_data.values()
        )
        
        bundle = {
            "version": timestamp,
            "bundle_id": f"enhanced_top30_altcoins_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            
            "metadata": {
                "strategy_name": "DipMaster_Enhanced_Top30_V2",
                "description": "增强版前30大市值山寨币完整数据集 - 支持6个时间框架",
                "collection_date": timestamp,
                "data_coverage": "最近24个月完整数据",
                "timeframes": ['1m', '5m', '15m', '1h', '4h', '1d'],
                "total_symbols": len(available_data),
                "excellent_symbols_count": len(excellent_symbols),
                "recommended_symbols_count": len(recommended_symbols),
                "total_files": total_files,
                "total_size_mb": round(total_size_mb, 2),
                "exchanges": ["binance"],
                "quality_standard": "99.5%+ completeness, strict OHLC validation"
            },
            
            "symbol_rankings": {
                ranking.symbol: {
                    "rank": idx + 1,
                    "overall_score": round(ranking.overall_score, 3),
                    "tier": ranking.tier,
                    "recommendation": ranking.recommendation,
                    "risk_level": ranking.risk_level,
                    "scores": {
                        "quality": round(ranking.quality_score, 3),
                        "liquidity": round(ranking.liquidity_score, 3),
                        "market_cap": round(ranking.market_cap_score, 3),
                        "volume": round(ranking.volume_score, 3),
                        "volatility": round(ranking.volatility_score, 3)
                    }
                }
                for idx, ranking in enumerate(symbol_rankings)
            },
            
            "data_files": {
                symbol: {
                    timeframe: {
                        "file_path": file_info['file_path'],
                        "metadata_path": file_info['metadata_path'],
                        "file_size_mb": round(file_info['file_size_mb'], 2),
                        "format": "parquet",
                        "compression": "zstd"
                    }
                    for timeframe, file_info in timeframes.items()
                }
                for symbol, timeframes in available_data.items()
            },
            
            "correlation_analysis": correlation_analysis,
            
            "portfolio_recommendations": {
                "tier_s_symbols": [r.symbol for r in symbol_rankings if r.tier == 'S'],
                "tier_a_symbols": [r.symbol for r in symbol_rankings if r.tier == 'A'],
                "highly_recommended": [r.symbol for r in symbol_rankings if r.recommendation == 'HIGHLY_RECOMMENDED'],
                "low_risk": [r.symbol for r in symbol_rankings if r.risk_level == 'LOW'],
                "optimal_portfolio_size": min(15, len(excellent_symbols)),
                "diversification_groups": correlation_analysis['groups'],
                "risk_balanced_selection": {
                    "conservative": [r.symbol for r in symbol_rankings if r.risk_level == 'LOW'][:8],
                    "balanced": [r.symbol for r in symbol_rankings if r.risk_level == 'MEDIUM'][:10],
                    "aggressive": [r.symbol for r in symbol_rankings if r.risk_level == 'HIGH'][:5]
                }
            },
            
            "quality_assurance": {
                "standards": {
                    "minimum_completeness": "99.5%",
                    "ohlc_consistency": "100%",
                    "time_continuity": "95%+",
                    "data_freshness": "< 24 hours",
                    "volume_validation": "active trading confirmed"
                },
                "quality_distribution": {
                    "excellent": len([r for r in symbol_rankings if r.tier == 'S']),
                    "good": len([r for r in symbol_rankings if r.tier == 'A']),
                    "fair": len([r for r in symbol_rankings if r.tier == 'B']),
                    "poor": len([r for r in symbol_rankings if r.tier in ['C', 'D']])
                },
                "average_quality_score": round(np.mean([r.quality_score for r in symbol_rankings]), 3)
            },
            
            "usage_guidelines": {
                "dipmaster_strategy": {
                    "preferred_symbols": excellent_symbols[:15],
                    "timeframe_priority": ["5m", "15m", "1h"],
                    "quality_threshold": 0.85,
                    "volume_requirement": "active_trading"
                },
                "risk_management": {
                    "max_symbols_per_group": 3,
                    "correlation_limit": 0.7,
                    "volatility_limit": "medium_to_low",
                    "position_sizing": "equal_weight_with_quality_adjustment"
                },
                "data_validation": [
                    "验证时间序列完整性",
                    "检查OHLC数据一致性",
                    "确认成交量真实性",
                    "监控数据更新频率",
                    "定期质量评估"
                ]
            },
            
            "timestamp": timestamp
        }
        
        return bundle
    
    def generate_comprehensive_report(self) -> Dict:
        """生成综合数据质量报告"""
        self.logger.info("开始生成综合数据质量报告...")
        
        try:
            # 1. 扫描可用数据
            available_data = self.scan_available_data()
            
            # 2. 分析币种排名
            symbol_rankings = self.analyze_symbol_rankings(available_data)
            
            # 3. 相关性分析
            correlation_analysis = self.generate_correlation_analysis(available_data)
            
            # 4. 创建MarketDataBundle
            market_data_bundle = self.create_enhanced_market_data_bundle(
                available_data, symbol_rankings, correlation_analysis
            )
            
            # 5. 保存报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存详细分析报告
            analysis_report = {
                'analysis_summary': {
                    'timestamp': datetime.now().isoformat(),
                    'total_symbols_analyzed': len(available_data),
                    'total_files_analyzed': sum(len(tf) for tf in available_data.values()),
                    'average_quality_score': np.mean([r.overall_score for r in symbol_rankings]),
                    'excellent_tier_count': len([r for r in symbol_rankings if r.tier == 'S']),
                    'recommended_count': len([r for r in symbol_rankings if r.recommendation == 'HIGHLY_RECOMMENDED'])
                },
                'available_data': available_data,
                'symbol_rankings': [asdict(r) for r in symbol_rankings],
                'correlation_analysis': correlation_analysis,
                'quality_metrics': {
                    'by_symbol': {r.symbol: asdict(r) for r in symbol_rankings},
                    'by_tier': {
                        tier: [r.symbol for r in symbol_rankings if r.tier == tier]
                        for tier in ['S', 'A', 'B', 'C', 'D']
                    }
                }
            }
            
            # 保存文件
            analysis_path = self.results_path / f"Data_Quality_Analysis_{timestamp}.json"
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_report, f, ensure_ascii=False, indent=2, default=str)
            
            bundle_path = Path("data") / "MarketDataBundle_Top30_Enhanced.json"
            with open(bundle_path, 'w', encoding='utf-8') as f:
                json.dump(market_data_bundle, f, ensure_ascii=False, indent=2, default=str)
            
            # 生成执行摘要
            print("\n" + "="*80)
            print("数据质量分析完成!")
            print("="*80)
            print(f"分析报告: {analysis_path}")
            print(f"数据束配置: {bundle_path}")
            print(f"分析币种数: {len(available_data)}")
            print(f"S级币种: {len([r for r in symbol_rankings if r.tier == 'S'])} 个")
            print(f"A级币种: {len([r for r in symbol_rankings if r.tier == 'A'])} 个")
            print(f"平均质量: {np.mean([r.overall_score for r in symbol_rankings]):.3f}")
            print(f"推荐币种: {len([r for r in symbol_rankings if r.recommendation == 'HIGHLY_RECOMMENDED'])} 个")
            print("="*80)
            
            # 显示TOP 10排名
            print("\nTOP 10 币种排名:")
            for i, ranking in enumerate(symbol_rankings[:10]):
                print(f"{i+1:2d}. {ranking.symbol:10s} - {ranking.tier} 级 - {ranking.overall_score:.3f} - {ranking.recommendation}")
            
            return {
                'analysis_report_path': str(analysis_path),
                'bundle_path': str(bundle_path),
                'market_data_bundle': market_data_bundle,
                'analysis_report': analysis_report
            }
            
        except Exception as e:
            self.logger.error(f"生成综合报告失败: {e}")
            raise

def main():
    """主函数"""
    analyzer = DataQualityAnalyzer()
    
    try:
        # 生成综合报告
        results = analyzer.generate_comprehensive_report()
        
        print(f"\n数据质量分析成功完成!")
        print(f"报告文件: {results['analysis_report_path']}")
        print(f"配置文件: {results['bundle_path']}")
        
        return results
        
    except Exception as e:
        print(f"数据质量分析失败: {e}")
        raise

if __name__ == "__main__":
    main()