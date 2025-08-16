#!/usr/bin/env python3
"""
Run Ultra Optimization - 一键运行超级优化系统
==========================================

完整流程：
1. 扩展币种池（下载新数据）
2. 运行超级优化验证
3. 生成完整报告
4. 提供部署建议

同步实施短期和中期优化，目标：
- 胜率从55%提升至75%+
- 评分从40.8提升至80+
- 风险等级从HIGH降至LOW

Author: DipMaster Ultra Team
Date: 2025-08-15
Version: 1.0.0
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
import json

# 设置项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入我们的模块
from src.tools.ultra_symbol_data_manager import UltraSymbolDataManager
from ultra_optimization_validator import UltraOptimizationValidator

logger = logging.getLogger(__name__)


class UltraOptimizationRunner:
    """超级优化运行器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data" / "market_data"
        self.results_dir = self.project_root / "results" / "ultra_optimization"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 运行配置
        self.download_new_data = True
        self.run_full_validation = True
        self.generate_charts = True
        
    async def run_complete_optimization(self):
        """运行完整的超级优化流程"""
        start_time = datetime.now()
        
        logger.info("🚀 Starting Complete Ultra Optimization Process")
        logger.info("="*60)
        logger.info("📋 Process Overview:")
        logger.info("  1. 扩展币种池（下载20+新币种数据）")
        logger.info("  2. 短期优化验证（信号参数、风险管理）")
        logger.info("  3. 中期优化验证（市场适应、相关性控制）")
        logger.info("  4. 综合性能评估")
        logger.info("  5. 生成完整报告和部署建议")
        logger.info("="*60)
        
        results = {}
        
        try:
            # === Phase 1: 扩展币种池 ===
            if self.download_new_data:
                logger.info("📥 Phase 1: Expanding Symbol Pool...")
                expansion_results = await self._expand_symbol_pool()
                results["symbol_expansion"] = expansion_results
            else:
                logger.info("⏭️  Phase 1: Skipped (using existing data)")
                
            # === Phase 2: 超级优化验证 ===
            logger.info("🔍 Phase 2: Running Ultra Optimization Validation...")
            validation_results = await self._run_validation()
            results["validation"] = validation_results
            
            # === Phase 3: 结果分析和报告 ===
            logger.info("📊 Phase 3: Generating Analysis and Reports...")
            analysis_results = await self._generate_analysis(results)
            results["analysis"] = analysis_results
            
            # === Phase 4: 部署建议 ===
            logger.info("🎯 Phase 4: Generating Deployment Recommendations...")
            deployment_advice = await self._generate_deployment_advice(results)
            results["deployment"] = deployment_advice
            
            # === 保存完整结果 ===
            await self._save_complete_results(results, start_time)
            
            # === 输出总结 ===
            self._print_final_summary(results, start_time)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Ultra optimization process failed: {e}")
            raise
            
    async def _expand_symbol_pool(self):
        """扩展币种池"""
        try:
            manager = UltraSymbolDataManager(str(self.data_dir))
            
            # 初始化币种池
            manager.initialize_symbol_pool()
            
            # 下载新币种数据
            success_count, fail_count = manager.download_all_symbols()
            
            # 质量评估
            quality_report = manager.get_quality_report()
            
            # 推荐币种
            recommended_symbols = manager.get_recommended_symbols(min_quality=70)
            
            return {
                "successful_downloads": success_count,
                "failed_downloads": fail_count,
                "quality_report": quality_report,
                "recommended_symbols": recommended_symbols,
                "total_symbols_available": len(manager.symbol_info)
            }
            
        except Exception as e:
            logger.error(f"Symbol pool expansion failed: {e}")
            return {"error": str(e)}
            
    async def _run_validation(self):
        """运行超级优化验证"""
        try:
            validator = UltraOptimizationValidator(str(self.data_dir))
            results = await validator.run_ultra_validation()
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"error": str(e)}
            
    async def _generate_analysis(self, results):
        """生成深度分析"""
        analysis = {
            "optimization_success": False,
            "key_improvements": [],
            "remaining_issues": [],
            "performance_grade": "F"
        }
        
        try:
            validation = results.get("validation")
            if not validation or hasattr(validation, 'error'):
                return analysis
                
            # 成功标准
            win_rate_target = 75.0
            score_target = 80.0
            improvement_target = 15.0
            
            # 评估优化成功度
            win_rate_achieved = getattr(validation, 'win_rate', 0)
            score_achieved = getattr(validation, 'overall_score', 0)
            improvement_achieved = getattr(validation, 'win_rate_improvement', 0)
            
            success_criteria = [
                win_rate_achieved >= win_rate_target,
                score_achieved >= score_target,
                improvement_achieved >= improvement_target
            ]
            
            analysis["optimization_success"] = sum(success_criteria) >= 2  # 至少满足2个条件
            
            # 记录改进
            if win_rate_achieved >= win_rate_target:
                analysis["key_improvements"].append(f"✅ 胜率达标: {win_rate_achieved:.1f}% (目标: {win_rate_target}%)")
            else:
                analysis["remaining_issues"].append(f"❌ 胜率未达标: {win_rate_achieved:.1f}% < {win_rate_target}%")
                
            if score_achieved >= score_target:
                analysis["key_improvements"].append(f"✅ 综合评分达标: {score_achieved:.1f} (目标: {score_target})")
            else:
                analysis["remaining_issues"].append(f"❌ 综合评分未达标: {score_achieved:.1f} < {score_target}")
                
            if improvement_achieved >= improvement_target:
                analysis["key_improvements"].append(f"✅ 胜率改善显著: {improvement_achieved:+.1f}% (目标: +{improvement_target}%)")
            else:
                analysis["remaining_issues"].append(f"❌ 胜率改善不足: {improvement_achieved:+.1f}% < +{improvement_target}%")
                
            # 性能评级
            if score_achieved >= 90:
                analysis["performance_grade"] = "A+"
            elif score_achieved >= 85:
                analysis["performance_grade"] = "A"
            elif score_achieved >= 80:
                analysis["performance_grade"] = "B"
            elif score_achieved >= 70:
                analysis["performance_grade"] = "C"
            elif score_achieved >= 60:
                analysis["performance_grade"] = "D"
            else:
                analysis["performance_grade"] = "F"
                
            # 具体分析
            analysis.update({
                "win_rate_analysis": {
                    "achieved": win_rate_achieved,
                    "target": win_rate_target,
                    "improvement": improvement_achieved,
                    "status": "✅ 达标" if win_rate_achieved >= win_rate_target else "❌ 未达标"
                },
                "risk_analysis": {
                    "risk_level": getattr(validation, 'risk_level', 'UNKNOWN'),
                    "sharpe_ratio": getattr(validation, 'sharpe_ratio', 0),
                    "max_drawdown": getattr(validation, 'max_drawdown', 0)
                },
                "signal_quality": {
                    "filter_rate": getattr(validation, 'signal_filter_rate', 0),
                    "grade_a_signals": getattr(validation, 'grade_a_signals', 0),
                    "total_signals": getattr(validation, 'total_signals_generated', 0)
                }
            })
            
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis generation failed: {e}")
            analysis["error"] = str(e)
            return analysis
            
    async def _generate_deployment_advice(self, results):
        """生成部署建议"""
        advice = {
            "deployment_recommendation": "❌ 不建议部署",
            "confidence_level": "低",
            "suggested_actions": [],
            "risk_warnings": []
        }
        
        try:
            validation = results.get("validation")
            analysis = results.get("analysis", {})
            
            if not validation or hasattr(validation, 'error'):
                advice["risk_warnings"].append("验证过程失败，无法提供可靠建议")
                return advice
                
            score = getattr(validation, 'overall_score', 0)
            risk_level = getattr(validation, 'risk_level', 'HIGH')
            win_rate = getattr(validation, 'win_rate', 0)
            
            # 部署建议逻辑
            if score >= 85 and risk_level == "LOW" and win_rate >= 75:
                advice.update({
                    "deployment_recommendation": "🚀 强烈建议部署",
                    "confidence_level": "高",
                    "suggested_actions": [
                        "立即开始小规模实盘测试（初始资金≤总资金10%）",
                        "设置严格的风险控制参数",
                        "进行1-2周的实盘验证",
                        "根据实盘表现逐步扩大规模"
                    ]
                })
            elif score >= 75 and win_rate >= 65:
                advice.update({
                    "deployment_recommendation": "✅ 谨慎建议部署",
                    "confidence_level": "中等",
                    "suggested_actions": [
                        "进行更长期的历史回测验证",
                        "小规模实盘测试（初始资金≤总资金5%）",
                        "密切监控实盘表现",
                        "优先选择Tier 1币种交易"
                    ]
                })
            elif score >= 60:
                advice.update({
                    "deployment_recommendation": "⚠️ 需要改进后部署",
                    "confidence_level": "低",
                    "suggested_actions": [
                        "继续优化信号质量和风险管理",
                        "扩大训练数据集",
                        "调整策略参数",
                        "进行更严格的过拟合检测"
                    ]
                })
            else:
                advice.update({
                    "deployment_recommendation": "❌ 不建议部署",
                    "confidence_level": "极低",
                    "suggested_actions": [
                        "重新设计策略核心逻辑",
                        "增加更多技术指标",
                        "改进数据质量",
                        "寻求专业策略建议"
                    ]
                })
                
            # 风险警告
            if risk_level == "HIGH":
                advice["risk_warnings"].append("⚠️ 系统风险等级较高，需要加强风险控制")
            if win_rate < 60:
                advice["risk_warnings"].append("⚠️ 胜率过低，可能导致长期亏损")
            if getattr(validation, 'max_drawdown', 0) < -1000:
                advice["risk_warnings"].append("⚠️ 最大回撤过大，需要改进止损策略")
                
            return advice
            
        except Exception as e:
            logger.error(f"Deployment advice generation failed: {e}")
            advice["error"] = str(e)
            return advice
            
    async def _save_complete_results(self, results, start_time):
        """保存完整结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        duration = datetime.now() - start_time
        
        # 完整结果JSON
        complete_results = {
            "meta": {
                "timestamp": timestamp,
                "duration_seconds": duration.total_seconds(),
                "process_version": "Ultra Optimization v1.0.0"
            },
            "results": results
        }
        
        results_file = self.results_dir / f"ultra_optimization_complete_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(complete_results, f, indent=2, default=str)
            
        logger.info(f"💾 Complete results saved to: {results_file}")
        
    def _print_final_summary(self, results, start_time):
        """打印最终总结"""
        duration = datetime.now() - start_time
        
        logger.info("🎉 Ultra Optimization Process Completed!")
        logger.info("="*60)
        logger.info(f"⏱️  Total Duration: {duration}")
        
        # 数据扩展结果
        expansion = results.get("symbol_expansion", {})
        if "error" not in expansion:
            logger.info(f"📥 Symbol Pool Expansion:")
            logger.info(f"  • Downloaded: {expansion.get('successful_downloads', 0)} symbols")
            logger.info(f"  • Failed: {expansion.get('failed_downloads', 0)} symbols")
            logger.info(f"  • Total Available: {expansion.get('total_symbols_available', 0)} symbols")
            
        # 验证结果
        validation = results.get("validation")
        if validation and not hasattr(validation, 'error'):
            logger.info(f"🎯 Validation Results:")
            logger.info(f"  • Overall Score: {getattr(validation, 'overall_score', 0):.1f}/100")
            logger.info(f"  • Win Rate: {getattr(validation, 'win_rate', 0):.1f}%")
            logger.info(f"  • Win Rate Improvement: {getattr(validation, 'win_rate_improvement', 0):+.1f}%")
            logger.info(f"  • Risk Level: {getattr(validation, 'risk_level', 'UNKNOWN')}")
            logger.info(f"  • Total Trades: {getattr(validation, 'total_trades', 0)}")
            
        # 分析结果
        analysis = results.get("analysis", {})
        if "error" not in analysis:
            logger.info(f"📊 Performance Analysis:")
            logger.info(f"  • Performance Grade: {analysis.get('performance_grade', 'N/A')}")
            logger.info(f"  • Optimization Success: {'✅' if analysis.get('optimization_success', False) else '❌'}")
            
        # 部署建议
        deployment = results.get("deployment", {})
        if "error" not in deployment:
            logger.info(f"🚀 Deployment Advice:")
            logger.info(f"  • Recommendation: {deployment.get('deployment_recommendation', 'N/A')}")
            logger.info(f"  • Confidence: {deployment.get('confidence_level', 'N/A')}")
            
        logger.info("="*60)
        
        # 关键建议
        if analysis.get("optimization_success", False):
            logger.info("🎉 优化成功！系统性能显著提升，可考虑实盘部署！")
        else:
            logger.info("⚠️ 优化效果有限，建议继续改进后再考虑实盘部署")
            
        logger.info("📋 详细报告已生成，请查看results/ultra_validation/目录")


async def main():
    """主函数"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ultra_optimization.log')
        ]
    )
    
    logger.info("🌟 DipMaster Ultra Optimization System")
    logger.info("=====================================")
    logger.info("同步实施短期和中期优化，目标：")
    logger.info("• 胜率从55%提升至75%+")
    logger.info("• 评分从40.8提升至80+")  
    logger.info("• 风险等级从HIGH降至LOW")
    logger.info("• 扩展币种池至20+优质标的（避开BTC/ETH）")
    logger.info("=====================================")
    
    try:
        # 创建运行器
        runner = UltraOptimizationRunner()
        
        # 运行完整流程
        results = await runner.run_complete_optimization()
        
        return results
        
    except KeyboardInterrupt:
        logger.info("⏹️  Process interrupted by user")
        return None
    except Exception as e:
        logger.error(f"❌ Process failed: {e}")
        raise


if __name__ == "__main__":
    # 运行主程序
    results = asyncio.run(main())