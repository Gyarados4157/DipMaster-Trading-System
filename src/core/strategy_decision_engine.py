#!/usr/bin/env python3
"""
Strategy Decision Engine - Automated Go/No-Go Decision Logic
自动化决策引擎：基于验证结果自动决定策略是否继续/部署/放弃

决策逻辑:
1. Phase 0 (Edge Analysis): 边界是否存在？
2. Phase 2 (Fast Validation): 策略是否鲁棒？
3. Parameter Sensitivity: 参数是否稳定？
4. Final Decision: 部署/保守部署/放弃

避免人为偏见和沉没成本谬误的自动化决策系统

Author: DipMaster Optimization Team
Date: 2025-08-15
Version: 1.0.0
"""

import sys
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import json

# 导入分析组件
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from edge_analyzer import EdgeAnalyzer, EdgeDiagnosis
from simplified_dipmaster import SimplifiedDipMaster
from fast_validator import FastValidator, RobustnessReport

logger = logging.getLogger(__name__)


class StrategyDecision(Enum):
    """策略决策结果"""
    DEPLOY_FULL = "deploy_full"                    # 完整部署
    DEPLOY_CONSERVATIVE = "deploy_conservative"    # 保守部署
    DEPLOY_MINIMAL = "deploy_minimal"              # 最小化部署
    REJECT_PIVOT = "reject_pivot"                  # 拒绝并转向新策略
    REJECT_OPTIMIZE = "reject_optimize"            # 拒绝但可尝试重新优化
    HALT_INVESTIGATE = "halt_investigate"          # 停止并深入调查


@dataclass
class DecisionContext:
    """决策上下文"""
    edge_diagnosis: EdgeDiagnosis
    robustness_report: RobustnessReport
    sensitivity_analysis: Dict[str, Any]
    strategy_params: Dict[str, float]
    analysis_timestamp: datetime


@dataclass
class StrategyRecommendation:
    """策略建议"""
    decision: StrategyDecision
    confidence: float  # 0-1
    reasoning: List[str]
    risk_warnings: List[str]
    deployment_parameters: Dict[str, Any]
    monitoring_requirements: List[str]
    next_review_date: datetime


class StrategyDecisionEngine:
    """策略决策引擎"""
    
    def __init__(self):
        # 决策阈值配置
        self.decision_thresholds = {
            # Edge存在性阈值
            'edge_high_confidence': 0.80,
            'edge_medium_confidence': 0.60,
            'edge_minimum_viable': 0.40,
            
            # 鲁棒性阈值
            'robustness_high': 0.80,
            'robustness_medium': 0.60,
            'robustness_minimum': 0.40,
            
            # 参数稳定性阈值
            'param_stability_excellent': 0.15,    # ±15%变化
            'param_stability_acceptable': 0.25,   # ±25%变化
            'param_stability_poor': 0.40,         # ±40%变化
            
            # 性能阈值
            'win_rate_excellent': 0.60,
            'win_rate_good': 0.55,
            'win_rate_minimum': 0.50,
            
            'return_excellent': 15.0,   # %
            'return_good': 8.0,         # %
            'return_minimum': 2.0,      # %
            
            'drawdown_excellent': 8.0,  # %
            'drawdown_acceptable': 15.0, # %
            'drawdown_maximum': 25.0     # %
        }
        
        # 部署参数配置
        self.deployment_configs = {
            'full': {
                'max_position_size': 0.05,      # 5% per trade
                'max_concurrent_positions': 3,
                'daily_loss_limit': 0.10,       # 10% daily loss limit
                'monitoring_frequency': 'daily'
            },
            'conservative': {
                'max_position_size': 0.03,      # 3% per trade
                'max_concurrent_positions': 2,
                'daily_loss_limit': 0.05,       # 5% daily loss limit
                'monitoring_frequency': 'daily'
            },
            'minimal': {
                'max_position_size': 0.01,      # 1% per trade
                'max_concurrent_positions': 1,
                'daily_loss_limit': 0.02,       # 2% daily loss limit
                'monitoring_frequency': 'hourly'
            }
        }
    
    def analyze_edge_quality(self, edge_diagnosis: EdgeDiagnosis) -> Tuple[str, float, List[str]]:
        """分析边界质量"""
        
        reasoning = []
        warnings = []
        
        confidence = edge_diagnosis.confidence_score
        
        if not edge_diagnosis.edge_exists:
            quality = "none"
            reasoning.append("No viable edge detected in fundamental analysis")
            warnings.append("Strategy lacks basic profitability requirements")
            
        elif confidence >= self.decision_thresholds['edge_high_confidence']:
            quality = "excellent"
            reasoning.append(f"Strong edge confirmed with {confidence:.1%} confidence")
            
        elif confidence >= self.decision_thresholds['edge_medium_confidence']:
            quality = "good"
            reasoning.append(f"Moderate edge detected with {confidence:.1%} confidence")
            warnings.append("Edge strength is moderate - monitor performance closely")
            
        elif confidence >= self.decision_thresholds['edge_minimum_viable']:
            quality = "marginal"
            reasoning.append(f"Marginal edge detected with {confidence:.1%} confidence")
            warnings.append("Edge is weak - consider conservative deployment only")
            
        else:
            quality = "poor"
            reasoning.append(f"Poor edge quality with only {confidence:.1%} confidence")
            warnings.append("Edge quality too low for reliable trading")
        
        # 市场演化影响分析
        if edge_diagnosis.market_evolution_impact == "negative_trend_detected":
            warnings.append("Performance trend is declining over time")
        
        # 失败模式分析
        if edge_diagnosis.primary_failure_mode:
            warnings.append(f"Primary failure mode: {edge_diagnosis.primary_failure_mode}")
        
        return quality, confidence, reasoning, warnings
    
    def analyze_robustness_quality(self, report: RobustnessReport) -> Tuple[str, float, List[str]]:
        """分析鲁棒性质量"""
        
        reasoning = []
        warnings = []
        
        if not report.strategy_robust:
            quality = "poor"
            reasoning.append("Strategy failed robustness tests across market conditions")
            warnings.append("Inconsistent performance across different market scenarios")
            return quality, report.confidence_score, reasoning, warnings
        
        confidence = report.confidence_score
        consistency = report.consistency_score
        
        if (confidence >= self.decision_thresholds['robustness_high'] and 
            consistency >= 0.80):
            quality = "excellent"
            reasoning.append(f"Excellent robustness: {confidence:.1%} confidence, {consistency:.1%} consistency")
            
        elif (confidence >= self.decision_thresholds['robustness_medium'] and 
              consistency >= 0.65):
            quality = "good"
            reasoning.append(f"Good robustness: {confidence:.1%} confidence, {consistency:.1%} consistency")
            warnings.append("Some performance variation across market conditions")
            
        elif confidence >= self.decision_thresholds['robustness_minimum']:
            quality = "marginal"
            reasoning.append(f"Marginal robustness: {confidence:.1%} confidence, {consistency:.1%} consistency")
            warnings.append("Significant performance variation detected")
            
        else:
            quality = "poor"
            reasoning.append(f"Poor robustness: {confidence:.1%} confidence, {consistency:.1%} consistency")
            warnings.append("High performance inconsistency across market conditions")
        
        # 具体性能分析
        avg_perf = report.average_performance
        if avg_perf['avg_win_rate'] < self.decision_thresholds['win_rate_minimum']:
            warnings.append(f"Average win rate {avg_perf['avg_win_rate']:.1%} below minimum threshold")
        
        if avg_perf['avg_max_drawdown'] > self.decision_thresholds['drawdown_maximum']:
            warnings.append(f"Average max drawdown {avg_perf['avg_max_drawdown']:.1f}% exceeds limit")
        
        return quality, confidence, reasoning, warnings
    
    def analyze_parameter_stability(self, sensitivity_analysis: Dict[str, Any]) -> Tuple[str, List[str]]:
        """分析参数稳定性"""
        
        reasoning = []
        warnings = []
        
        if not sensitivity_analysis.get('parameter_robust', False):
            stability = "poor"
            sensitive_params = sensitivity_analysis.get('sensitive_parameters', [])
            reasoning.append(f"Parameters are sensitive: {', '.join(sensitive_params)}")
            warnings.append("Strategy may be overfit to specific parameter values")
            return stability, reasoning, warnings
        
        max_sensitivity = sensitivity_analysis.get('max_win_rate_sensitivity', 0)
        
        if max_sensitivity <= self.decision_thresholds['param_stability_excellent']:
            stability = "excellent"
            reasoning.append(f"Excellent parameter stability: max variation {max_sensitivity:.1%}")
            
        elif max_sensitivity <= self.decision_thresholds['param_stability_acceptable']:
            stability = "good"
            reasoning.append(f"Good parameter stability: max variation {max_sensitivity:.1%}")
            warnings.append("Some parameter sensitivity detected - monitor performance")
            
        else:
            stability = "marginal"
            reasoning.append(f"Marginal parameter stability: max variation {max_sensitivity:.1%}")
            warnings.append("High parameter sensitivity may indicate overfitting")
        
        return stability, reasoning, warnings
    
    def make_deployment_decision(self, context: DecisionContext) -> StrategyRecommendation:
        """做出部署决策"""
        
        logger.info("🎯 Making deployment decision based on comprehensive analysis...")
        
        # 分析各个维度
        edge_quality, edge_confidence, edge_reasoning, edge_warnings = self.analyze_edge_quality(context.edge_diagnosis)
        robustness_quality, robustness_confidence, robustness_reasoning, robustness_warnings = self.analyze_robustness_quality(context.robustness_report)
        param_stability, param_reasoning, param_warnings = self.analyze_parameter_stability(context.sensitivity_analysis)
        
        # 综合所有分析结果
        all_reasoning = edge_reasoning + robustness_reasoning + param_reasoning
        all_warnings = edge_warnings + robustness_warnings + param_warnings
        
        # 决策矩阵
        decision = self._determine_decision(edge_quality, robustness_quality, param_stability)
        
        # 计算综合置信度
        overall_confidence = (edge_confidence * 0.4 + robustness_confidence * 0.4 + 
                             (1.0 if param_stability in ['excellent', 'good'] else 0.5) * 0.2)
        
        # 获取部署参数
        deployment_params = self._get_deployment_parameters(decision)
        
        # 确定监控要求
        monitoring_requirements = self._determine_monitoring_requirements(decision, all_warnings)
        
        # 下次审查时间
        if decision in [StrategyDecision.DEPLOY_FULL, StrategyDecision.DEPLOY_CONSERVATIVE]:
            next_review = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=30)
        elif decision == StrategyDecision.DEPLOY_MINIMAL:
            next_review = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=7)
        else:
            next_review = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=90)
        
        recommendation = StrategyRecommendation(
            decision=decision,
            confidence=overall_confidence,
            reasoning=all_reasoning,
            risk_warnings=all_warnings,
            deployment_parameters=deployment_params,
            monitoring_requirements=monitoring_requirements,
            next_review_date=next_review
        )
        
        logger.info(f"🎯 Decision: {decision.value.upper()}")
        logger.info(f"📊 Confidence: {overall_confidence:.1%}")
        logger.info(f"🔍 Key reasoning: {len(all_reasoning)} factors, {len(all_warnings)} warnings")
        
        return recommendation
    
    def _determine_decision(self, edge_quality: str, robustness_quality: str, param_stability: str) -> StrategyDecision:
        """基于质量评估确定决策"""
        
        # 决策矩阵 - 保守的决策逻辑
        
        # 如果没有边界，直接拒绝
        if edge_quality == "none":
            return StrategyDecision.REJECT_PIVOT
        
        # 如果边界质量差，拒绝
        if edge_quality == "poor":
            if robustness_quality in ["excellent", "good"]:
                return StrategyDecision.REJECT_OPTIMIZE  # 可能是参数问题
            else:
                return StrategyDecision.REJECT_PIVOT     # 根本问题
        
        # 如果鲁棒性差，拒绝
        if robustness_quality == "poor":
            return StrategyDecision.REJECT_OPTIMIZE
        
        # 参数不稳定的策略，最多保守部署
        if param_stability == "poor":
            if edge_quality == "excellent" and robustness_quality == "excellent":
                return StrategyDecision.DEPLOY_MINIMAL   # 高度监控
            else:
                return StrategyDecision.REJECT_OPTIMIZE
        
        # 基于组合质量决策
        if (edge_quality == "excellent" and robustness_quality == "excellent" and 
            param_stability in ["excellent", "good"]):
            return StrategyDecision.DEPLOY_FULL
        
        elif (edge_quality in ["excellent", "good"] and robustness_quality in ["excellent", "good"] and 
              param_stability in ["excellent", "good", "marginal"]):
            return StrategyDecision.DEPLOY_CONSERVATIVE
        
        elif (edge_quality in ["good", "marginal"] and robustness_quality in ["good", "marginal"] and 
              param_stability != "poor"):
            return StrategyDecision.DEPLOY_MINIMAL
        
        else:
            return StrategyDecision.REJECT_OPTIMIZE
    
    def _get_deployment_parameters(self, decision: StrategyDecision) -> Dict[str, Any]:
        """获取部署参数"""
        
        if decision == StrategyDecision.DEPLOY_FULL:
            return self.deployment_configs['full'].copy()
        elif decision == StrategyDecision.DEPLOY_CONSERVATIVE:
            return self.deployment_configs['conservative'].copy()
        elif decision == StrategyDecision.DEPLOY_MINIMAL:
            return self.deployment_configs['minimal'].copy()
        else:
            return {}
    
    def _determine_monitoring_requirements(self, decision: StrategyDecision, warnings: List[str]) -> List[str]:
        """确定监控要求"""
        
        requirements = []
        
        if decision in [StrategyDecision.DEPLOY_FULL, StrategyDecision.DEPLOY_CONSERVATIVE, StrategyDecision.DEPLOY_MINIMAL]:
            requirements.append("Daily performance review")
            requirements.append("Weekly parameter stability check")
            requirements.append("Monthly comprehensive analysis")
        
        if decision == StrategyDecision.DEPLOY_MINIMAL:
            requirements.append("Real-time position monitoring")
            requirements.append("Immediate alert on 5% daily loss")
        
        # 基于警告添加特定监控
        for warning in warnings:
            if "declining" in warning.lower():
                requirements.append("Weekly trend analysis")
            if "sensitive" in warning.lower() or "overfit" in warning.lower():
                requirements.append("Parameter drift monitoring")
            if "drawdown" in warning.lower():
                requirements.append("Enhanced risk monitoring")
        
        return list(set(requirements))  # 去重
    
    def run_full_decision_process(self, symbol: str = "ICPUSDT") -> StrategyRecommendation:
        """运行完整决策流程"""
        
        logger.info("🚀 Starting comprehensive strategy decision process...")
        
        # Phase 0: Edge Analysis
        logger.info("📍 Phase 0: Edge Existence Analysis")
        edge_analyzer = EdgeAnalyzer()
        edge_diagnosis = edge_analyzer.diagnose_edge_existence(symbol)
        
        # 如果没有边界，直接停止
        if not edge_diagnosis.edge_exists:
            logger.warning("❌ No edge detected - stopping analysis")
            return StrategyRecommendation(
                decision=StrategyDecision.REJECT_PIVOT,
                confidence=0.0,
                reasoning=["No viable trading edge detected in fundamental analysis"],
                risk_warnings=["Strategy lacks basic profitability requirements"],
                deployment_parameters={},
                monitoring_requirements=[],
                next_review_date=datetime.now() + timedelta(days=90)
            )
        
        # Phase 1: Strategy Simplification (already done by using SimplifiedDipMaster)
        logger.info("📍 Phase 1: Strategy Simplification (using 3-parameter strategy)")
        strategy = SimplifiedDipMaster()
        
        # Phase 2: Fast Validation
        logger.info("📍 Phase 2: Fast Robustness Validation")
        validator = FastValidator()
        robustness_report = validator.run_3x3x3_validation(strategy)
        
        # Phase 2.5: Parameter Sensitivity
        logger.info("📍 Phase 2.5: Parameter Sensitivity Analysis")
        sensitivity_analysis = validator.test_parameter_sensitivity(strategy)
        
        # Create decision context
        context = DecisionContext(
            edge_diagnosis=edge_diagnosis,
            robustness_report=robustness_report,
            sensitivity_analysis=sensitivity_analysis,
            strategy_params={
                'rsi_threshold': strategy.rsi_threshold,
                'take_profit_pct': strategy.take_profit_pct,
                'stop_loss_pct': strategy.stop_loss_pct
            },
            analysis_timestamp=datetime.now()
        )
        
        # Make final decision
        logger.info("📍 Phase 3: Final Decision Making")
        recommendation = self.make_deployment_decision(context)
        
        # Save decision record
        self._save_decision_record(context, recommendation)
        
        return recommendation
    
    def _save_decision_record(self, context: DecisionContext, recommendation: StrategyRecommendation):
        """保存决策记录"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_decision_{timestamp}.json"
        
        decision_record = {
            'decision_summary': {
                'decision': recommendation.decision.value,
                'confidence': recommendation.confidence,
                'timestamp': context.analysis_timestamp.isoformat()
            },
            'analysis_results': {
                'edge_exists': context.edge_diagnosis.edge_exists,
                'edge_confidence': context.edge_diagnosis.confidence_score,
                'strategy_robust': context.robustness_report.strategy_robust,
                'robustness_confidence': context.robustness_report.confidence_score,
                'parameter_robust': context.sensitivity_analysis.get('parameter_robust', False)
            },
            'reasoning': recommendation.reasoning,
            'risk_warnings': recommendation.risk_warnings,
            'deployment_parameters': recommendation.deployment_parameters,
            'monitoring_requirements': recommendation.monitoring_requirements,
            'next_review_date': recommendation.next_review_date.isoformat(),
            'strategy_parameters': context.strategy_params
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(decision_record, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"📁 Decision record saved to: {filename}")


def main():
    """主函数 - 执行完整决策流程"""
    
    print("🎯 Strategy Decision Engine - Comprehensive Analysis & Decision")
    print("="*80)
    
    # 创建决策引擎
    decision_engine = StrategyDecisionEngine()
    
    print("🔍 DECISION PROCESS:")
    print("   Phase 0: Edge existence validation")
    print("   Phase 1: Strategy simplification (3 parameters)")  
    print("   Phase 2: Fast robustness validation (3x3x3 matrix)")
    print("   Phase 3: Automated decision making")
    
    # 运行完整决策流程
    try:
        recommendation = decision_engine.run_full_decision_process("ICPUSDT")
        
        # 显示最终结果
        print(f"\n🎯 FINAL DECISION: {recommendation.decision.value.upper().replace('_', ' ')}")
        print(f"📊 Confidence: {recommendation.confidence:.1%}")
        
        print(f"\n🧠 KEY REASONING:")
        for i, reason in enumerate(recommendation.reasoning, 1):
            print(f"   {i}. {reason}")
        
        if recommendation.risk_warnings:
            print(f"\n⚠️ RISK WARNINGS:")
            for i, warning in enumerate(recommendation.risk_warnings, 1):
                print(f"   {i}. {warning}")
        
        if recommendation.deployment_parameters:
            print(f"\n⚙️ DEPLOYMENT CONFIGURATION:")
            for key, value in recommendation.deployment_parameters.items():
                print(f"   {key}: {value}")
        
        if recommendation.monitoring_requirements:
            print(f"\n📊 MONITORING REQUIREMENTS:")
            for req in recommendation.monitoring_requirements:
                print(f"   • {req}")
        
        print(f"\n📅 Next Review: {recommendation.next_review_date.strftime('%Y-%m-%d')}")
        
        # 行动建议
        if recommendation.decision in [StrategyDecision.DEPLOY_FULL, StrategyDecision.DEPLOY_CONSERVATIVE, StrategyDecision.DEPLOY_MINIMAL]:
            print(f"\n🚀 ACTION: Proceed with deployment using specified parameters")
        elif recommendation.decision == StrategyDecision.REJECT_OPTIMIZE:
            print(f"\n🔄 ACTION: Reject current approach but consider re-optimization")
        else:
            print(f"\n🛑 ACTION: Reject strategy and pivot to new approach")
        
    except Exception as e:
        logger.error(f"❌ Decision process failed: {e}")
        print(f"\n❌ DECISION PROCESS FAILED: {e}")
        print("🔧 Please check data availability and system configuration")


if __name__ == "__main__":
    main()