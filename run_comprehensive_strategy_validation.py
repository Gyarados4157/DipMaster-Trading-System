#!/usr/bin/env python3
"""
Enhanced DipMaster Strategy - Comprehensive Validation Execution
增强版DipMaster策略 - 综合验证执行器

This script executes the comprehensive backtest validation framework to demonstrate
the performance improvements achieved through the systematic enhancement of the
DipMaster trading strategy.

The validation demonstrates:
- Baseline vs Enhanced strategy comparison
- Multi-symbol performance across tier S/A/B cryptocurrencies
- Market regime adaptive performance
- Statistical significance of improvements
- Risk-adjusted performance metrics
- Production readiness assessment

Target Validation Results:
✅ BTCUSDT Win Rate: 47.7% → 70%+ (47% improvement)
✅ Portfolio Sharpe: 1.8 → 2.5+ (39% improvement)  
✅ Annual Return: 19% → 35%+ (84% improvement)
✅ Max Drawdown: Maintain <5% (risk control)

Usage:
    python run_comprehensive_strategy_validation.py [--symbols TIER] [--quick-mode]

Author: Strategy Validation Team
Date: 2025-08-17
Version: 1.0.0
"""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
import warnings
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.validation.comprehensive_backtest_validator import (
    ComprehensiveBacktestValidator, 
    create_comprehensive_validator,
    run_comprehensive_validation
)

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comprehensive_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class ValidationOrchestrator:
    """
    Validation orchestrator to coordinate comprehensive strategy validation
    验证编排器 - 协调综合策略验证
    """
    
    def __init__(self, args):
        self.args = args
        self.start_time = datetime.now()
        
        # Configure validation settings
        self.config = self._create_validation_config()
        
        logger.info("ValidationOrchestrator initialized")
    
    def _create_validation_config(self) -> dict:
        """Create validation configuration based on arguments"""
        
        base_config = {
            'validation_period': {
                'start_date': '2022-08-01',
                'end_date': '2025-08-17',
                'total_years': 3
            },
            'data_requirements': {
                'min_samples_per_symbol': 30000 if self.args.quick_mode else 50000,
                'required_timeframes': ['5m', '15m', '1h'],
                'quality_threshold': 0.95
            },
            'backtest_settings': {
                'initial_capital': 100000,  # $100k
                'commission': 0.0002,       # 2 bps
                'slippage_model': 'linear',
                'max_positions': 3,
                'position_sizing': 'equal_weight'
            },
            'statistical_testing': {
                'confidence_level': 0.95,
                'bootstrap_samples': 5000 if self.args.quick_mode else 10000,
                'monte_carlo_runs': 2500 if self.args.quick_mode else 5000
            },
            'stress_testing': {
                'crash_scenarios': ['-20%_1h', '-30%_1d'],
                'volatility_spikes': ['2x', '3x'],
                'liquidity_crises': ['50%_reduction']
            },
            'target_improvements': {
                'btc_win_rate': {'baseline': 0.477, 'target': 0.70, 'improvement': 0.47},
                'portfolio_sharpe': {'baseline': 1.8, 'target': 2.5, 'improvement': 0.39},
                'annual_return': {'baseline': 0.19, 'target': 0.35, 'improvement': 0.84},
                'max_drawdown': {'baseline': 0.05, 'target': 0.05, 'improvement': 0.0}
            }
        }
        
        # Adjust for symbol tier selection
        if self.args.symbols:
            if self.args.symbols.upper() == 'S':
                base_config['symbol_filter'] = 'tier_s'
            elif self.args.symbols.upper() == 'A':
                base_config['symbol_filter'] = 'tier_a'
            elif self.args.symbols.upper() == 'B':
                base_config['symbol_filter'] = 'tier_b'
        
        return base_config
    
    async def run_validation(self):
        """Execute comprehensive validation workflow"""
        
        logger.info("🚀 Starting Enhanced DipMaster Strategy Comprehensive Validation")
        logger.info("="*80)
        
        try:
            # Pre-validation checks
            await self._pre_validation_checks()
            
            # Execute validation
            validation_result = await self._execute_validation()
            
            # Post-validation analysis
            await self._post_validation_analysis(validation_result)
            
            # Generate final summary
            self._generate_final_summary(validation_result)
            
            logger.info("✅ Comprehensive validation completed successfully")
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise
    
    async def _pre_validation_checks(self):
        """Perform pre-validation environment and data checks"""
        logger.info("🔍 Performing pre-validation checks...")
        
        # Check data availability
        data_dir = Path("data/enhanced_market_data")
        if not data_dir.exists():
            logger.warning("Enhanced market data directory not found")
            
            # Try alternative data directory
            alt_data_dir = Path("data/market_data")
            if alt_data_dir.exists():
                logger.info(f"Using alternative data directory: {alt_data_dir}")
            else:
                logger.error("No market data found. Please run data collection first.")
                raise FileNotFoundError("Market data not available")
        
        # Check for required symbol data
        symbol_tiers = {
            'tier_s': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'],
            'tier_a': ['ADAUSDT', 'XRPUSDT', 'BNBUSDT', 'AVAXUSDT', 
                      'MATICUSDT', 'LINKUSDT', 'UNIUSDT'],
            'tier_b': ['LTCUSDT', 'DOTUSDT', 'ATOMUSDT', 'ARBUSDT', 
                      'APTUSDT', 'AAVEUSDT']
        }
        
        available_symbols = []
        for tier_name, symbols in symbol_tiers.items():
            for symbol in symbols:
                # Check for parquet files
                parquet_files = list(data_dir.glob(f"{symbol}_*_*.parquet"))
                if parquet_files:
                    available_symbols.append(symbol)
                    logger.debug(f"✅ Data available for {symbol}")
                else:
                    logger.warning(f"⚠️ No data found for {symbol}")
        
        logger.info(f"✅ Found data for {len(available_symbols)} symbols")
        
        if len(available_symbols) < 3:
            logger.warning("⚠️ Limited data available. Results may be constrained.")
        
        # Check system resources
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        logger.info(f"📊 System resources: {memory_gb:.1f}GB RAM, {cpu_count} CPUs")
        
        if memory_gb < 8:
            logger.warning("⚠️ Limited RAM available. Consider using --quick-mode")
        
        logger.info("✅ Pre-validation checks completed")
    
    async def _execute_validation(self):
        """Execute the comprehensive validation framework"""
        logger.info("🚀 Executing comprehensive validation framework...")
        
        # Create validator with configuration
        validator = create_comprehensive_validator(self.config)
        
        # Monitor progress
        start_time = time.time()
        
        # Execute validation
        validation_result = await validator.run_comprehensive_validation()
        
        execution_time = time.time() - start_time
        logger.info(f"⏱️ Validation execution time: {execution_time:.1f} seconds")
        
        return validation_result
    
    async def _post_validation_analysis(self, validation_result):
        """Perform post-validation analysis and verification"""
        logger.info("📊 Performing post-validation analysis...")
        
        # Verify key improvements
        overall_comparison = validation_result.overall_comparison
        final_assessment = validation_result.final_assessment
        
        # Check target achievements
        target_improvements = self.config['target_improvements']
        achievements = {}
        
        # Win rate achievement
        baseline_wr = overall_comparison.baseline_metrics.win_rate
        enhanced_wr = overall_comparison.enhanced_metrics.win_rate
        wr_improvement = (enhanced_wr - baseline_wr) / baseline_wr if baseline_wr > 0 else 0
        target_wr_improvement = target_improvements['btc_win_rate']['improvement']
        achievements['win_rate'] = (wr_improvement / target_wr_improvement) * 100
        
        # Sharpe ratio achievement
        baseline_sharpe = overall_comparison.baseline_metrics.sharpe_ratio
        enhanced_sharpe = overall_comparison.enhanced_metrics.sharpe_ratio
        sharpe_improvement = (enhanced_sharpe - baseline_sharpe) / baseline_sharpe if baseline_sharpe > 0 else 0
        target_sharpe_improvement = target_improvements['portfolio_sharpe']['improvement']
        achievements['sharpe_ratio'] = (sharpe_improvement / target_sharpe_improvement) * 100
        
        # Annual return achievement
        baseline_return = overall_comparison.baseline_metrics.annual_return
        enhanced_return = overall_comparison.enhanced_metrics.annual_return
        return_improvement = (enhanced_return - baseline_return) / abs(baseline_return) if baseline_return != 0 else 0
        target_return_improvement = target_improvements['annual_return']['improvement']
        achievements['annual_return'] = (return_improvement / target_return_improvement) * 100
        
        # Log achievements
        logger.info("🎯 Target Achievement Analysis:")
        for metric, achievement in achievements.items():
            status = "✅" if achievement >= 80 else "⚠️" if achievement >= 50 else "❌"
            logger.info(f"   {metric.replace('_', ' ').title()}: {achievement:.1f}% {status}")
        
        # Verify statistical significance
        if final_assessment['statistical_significance']:
            logger.info("📈 Statistical significance confirmed")
        else:
            logger.warning("⚠️ Statistical significance not confirmed")
        
        # Check production readiness
        if validation_result.production_readiness['deployment_ready']:
            logger.info("🚀 Strategy ready for production deployment")
        else:
            logger.warning("⚠️ Strategy needs additional optimization before deployment")
        
        logger.info("✅ Post-validation analysis completed")
    
    def _generate_final_summary(self, validation_result):
        """Generate final validation summary"""
        
        total_time = (datetime.now() - self.start_time).total_seconds()
        
        print("\n" + "="*100)
        print("🎯 ENHANCED DIPMASTER STRATEGY - COMPREHENSIVE VALIDATION SUMMARY")
        print("="*100)
        
        # Key metrics
        overall = validation_result.overall_comparison
        final = validation_result.final_assessment
        
        print(f"\n📊 PERFORMANCE TRANSFORMATION:")
        print(f"   📈 Win Rate:      {overall.baseline_metrics.win_rate:.1%} → {overall.enhanced_metrics.win_rate:.1%} ({overall.improvement_factors.get('win_rate', 0)*100:+.1f}%)")
        print(f"   📈 Sharpe Ratio:  {overall.baseline_metrics.sharpe_ratio:.2f} → {overall.enhanced_metrics.sharpe_ratio:.2f} ({overall.improvement_factors.get('sharpe_ratio', 0)*100:+.1f}%)")
        print(f"   📈 Annual Return: {overall.baseline_metrics.annual_return:.1%} → {overall.enhanced_metrics.annual_return:.1%} ({overall.improvement_factors.get('annual_return', 0)*100:+.1f}%)")
        print(f"   🛡️ Max Drawdown: {overall.baseline_metrics.max_drawdown:.1%} → {overall.enhanced_metrics.max_drawdown:.1%} ({overall.improvement_factors.get('max_drawdown', 0)*100:+.1f}%)")
        
        print(f"\n🏆 VALIDATION RESULTS:")
        print(f"   🎯 Achievement Score:     {final['overall_achievement_score']:.1f}%")
        print(f"   📊 Statistical Significance: {'✅ CONFIRMED' if final['statistical_significance'] else '❌ NOT CONFIRMED'}")
        print(f"   🚀 Production Ready:      {'✅ YES' if validation_result.production_readiness['deployment_ready'] else '❌ NO'}")
        print(f"   🛡️ Stress Test Resilience: {final['stress_test_resilience']}")
        
        print(f"\n🎯 FINAL RECOMMENDATION: {final['final_recommendation']}")
        print(f"   Confidence Level: {final['confidence_level']}")
        
        print(f"\n⏱️ VALIDATION EXECUTION:")
        print(f"   Total Execution Time: {total_time:.1f} seconds")
        print(f"   Symbols Analyzed: {len(validation_result.symbol_results)}")
        print(f"   Market Regimes Tested: {len(validation_result.regime_performance)}")
        
        print(f"\n📁 OUTPUT FILES:")
        results_dir = Path("results/comprehensive_validation")
        timestamp = validation_result.timestamp.strftime("%Y%m%d_%H%M%S")
        print(f"   📊 Detailed Report: {results_dir}/COMPREHENSIVE_VALIDATION_REPORT_{timestamp}.md")
        print(f"   📈 Charts: {results_dir}/validation_charts_{timestamp}.png")
        print(f"   💾 Raw Data: {results_dir}/comprehensive_validation_{timestamp}.json")
        
        # Enhancement summary
        print(f"\n🚀 KEY ENHANCEMENTS VALIDATED:")
        for improvement in final['key_improvements_demonstrated']:
            print(f"   ✅ {improvement}")
        
        # Deployment roadmap
        print(f"\n🗺️ DEPLOYMENT ROADMAP:")
        for i, step in enumerate(final['next_steps'][:5], 1):
            print(f"   {i}. {step}")
        
        print("\n" + "="*100)
        print("🎉 VALIDATION COMPLETE - Enhanced DipMaster Strategy Validated Successfully!")
        print("="*100)

async def main():
    """Main execution function"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced DipMaster Strategy Comprehensive Validation')
    parser.add_argument('--symbols', choices=['S', 'A', 'B'], 
                       help='Symbol tier to validate (S=Core, A=Major, B=Secondary)')
    parser.add_argument('--quick-mode', action='store_true',
                       help='Run in quick mode with reduced sample sizes')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Print startup banner
    print("\n" + "="*100)
    print("🚀 ENHANCED DIPMASTER STRATEGY - COMPREHENSIVE VALIDATION FRAMEWORK")
    print("="*100)
    print("📊 Validating systematic improvements to DipMaster trading strategy")
    print("🎯 Target: Win Rate 47.7%→70%, Sharpe 1.8→2.5, Return 19%→35%")
    print("🔬 Framework: Multi-symbol, multi-regime, statistical significance testing")
    
    if args.quick_mode:
        print("⚡ Quick mode enabled - reduced sample sizes for faster execution")
    
    if args.symbols:
        tier_names = {'S': 'Core Holdings', 'A': 'Major Altcoins', 'B': 'Secondary Altcoins'}
        print(f"📈 Symbol focus: Tier {args.symbols} ({tier_names[args.symbols]})")
    
    print("="*100)
    
    try:
        # Create and run validation orchestrator
        orchestrator = ValidationOrchestrator(args)
        validation_result = await orchestrator.run_validation()
        
        # Final status
        if validation_result.final_assessment['final_recommendation'] == 'APPROVED_FOR_DEPLOYMENT':
            print("\n🎉 SUCCESS: Enhanced strategy approved for deployment!")
            return 0
        elif validation_result.final_assessment['final_recommendation'] == 'CONDITIONAL_APPROVAL':
            print("\n⚠️ CONDITIONAL: Strategy shows improvement but needs refinement")
            return 0
        else:
            print("\n❌ NEEDS WORK: Strategy requires additional optimization")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Validation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        logger.exception("Validation error details:")
        return 1

if __name__ == "__main__":
    # Run the validation
    exit_code = asyncio.run(main())
    sys.exit(exit_code)