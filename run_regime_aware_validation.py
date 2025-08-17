#!/usr/bin/env python3
"""
Regime-Aware DipMaster Strategy Validation Runner
市场体制感知DipMaster策略验证运行器

This script demonstrates the complete regime-aware strategy implementation
and validates the performance improvements against baseline DipMaster.

Target: Improve BTCUSDT win rate from 47.7% to 65%+

Usage:
    python run_regime_aware_validation.py
    python run_regime_aware_validation.py --symbol BTCUSDT --quick

Author: Strategy Orchestrator
Date: 2025-08-17
Version: 1.0.0
"""

import sys
import os
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to Python path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.validation.regime_aware_validator import RegimeAwareValidator, ValidationConfig
from src.core.market_regime_detector import MarketRegimeDetector, create_regime_detector
from src.data.regime_aware_feature_engineering import RegimeAwareFeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('regime_validation.log')
    ]
)

logger = logging.getLogger(__name__)

def find_symbol_data_files() -> dict:
    """Find available symbol data files"""
    data_dir = Path(__file__).parent / 'data' / 'enhanced_market_data'
    
    if not data_dir.exists():
        logger.warning(f"Enhanced market data directory not found: {data_dir}")
        # Try alternative locations
        alt_dirs = [
            Path(__file__).parent / 'data' / 'market_data',
            Path(__file__).parent / 'data'
        ]
        
        for alt_dir in alt_dirs:
            if alt_dir.exists():
                data_dir = alt_dir
                break
    
    symbol_files = {}
    
    # Look for 5-minute data files (preferred for DipMaster)
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT', 
                   'BNBUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'DOTUSDT']:
        
        # Try different file patterns
        patterns = [
            f"{symbol}_5m_2years.parquet",
            f"{symbol}_5m_3years.parquet", 
            f"{symbol}_5m_2years.csv",
            f"{symbol}_5m.parquet",
            f"{symbol}_5m.csv"
        ]
        
        for pattern in patterns:
            file_path = data_dir / pattern
            if file_path.exists():
                symbol_files[symbol] = str(file_path)
                logger.info(f"Found data for {symbol}: {file_path}")
                break
    
    if not symbol_files:
        logger.error(f"No symbol data files found in {data_dir}")
        logger.info("Available files:")
        if data_dir.exists():
            for f in sorted(data_dir.iterdir()):
                if f.is_file() and (f.suffix in ['.parquet', '.csv']):
                    logger.info(f"  {f.name}")
    
    return symbol_files

def run_quick_regime_demo(symbol: str = 'BTCUSDT') -> dict:
    """Run a quick demonstration of regime detection"""
    logger.info(f"Running quick regime detection demo for {symbol}")
    
    # Find data file
    symbol_files = find_symbol_data_files()
    
    if symbol not in symbol_files:
        logger.error(f"No data file found for {symbol}")
        return {}
    
    try:
        # Load data
        data_path = symbol_files[symbol]
        if data_path.endswith('.parquet'):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        
        logger.info(f"Loaded {len(df)} rows of data for {symbol}")
        
        # Initialize regime detector
        regime_detector = create_regime_detector()
        
        # Analyze regime for recent data (last 1000 bars)
        recent_df = df.tail(1000).reset_index(drop=True)
        regime_signal = regime_detector.identify_regime(recent_df, symbol)
        
        # Get adaptive parameters
        adaptive_params = regime_detector.get_adaptive_parameters(regime_signal.regime, symbol)
        
        # Analyze historical regime distribution
        regime_history = []
        
        # Sample every 100 bars to get regime distribution
        for i in range(500, len(df), 100):
            window_df = df.iloc[:i]
            regime = regime_detector.identify_regime(window_df, symbol)
            regime_history.append({
                'timestamp': df.iloc[i].get('timestamp', i),
                'regime': regime.regime.value,
                'confidence': regime.confidence
            })
        
        regime_df = pd.DataFrame(regime_history)
        regime_distribution = regime_df['regime'].value_counts(normalize=True).to_dict()
        
        demo_results = {
            'symbol': symbol,
            'data_points': len(df),
            'current_regime': {
                'regime': regime_signal.regime.value,
                'confidence': regime_signal.confidence,
                'stability': regime_signal.stability_score,
                'description': regime_detector.get_regime_description(regime_signal.regime)
            },
            'adaptive_parameters': adaptive_params,
            'historical_regime_distribution': regime_distribution,
            'regime_transitions': len(regime_df),
            'should_trade': regime_detector.should_trade_in_regime(regime_signal.regime, regime_signal.confidence)
        }
        
        logger.info(f"Current regime for {symbol}: {regime_signal.regime.value} "
                   f"(confidence: {regime_signal.confidence:.2f})")
        logger.info(f"Regime distribution: {regime_distribution}")
        
        return demo_results
        
    except Exception as e:
        logger.error(f"Error in regime demo: {str(e)}")
        return {'error': str(e)}

def run_btc_focused_validation() -> dict:
    """Run focused validation on BTCUSDT (our best performer)"""
    logger.info("Running focused validation on BTCUSDT")
    
    symbol_files = find_symbol_data_files()
    
    if 'BTCUSDT' not in symbol_files:
        logger.error("BTCUSDT data not found")
        return {}
    
    try:
        # Create validator with BTCUSDT focus
        validator = RegimeAwareValidator(ValidationConfig(
            test_symbols=['BTCUSDT'],
            validation_period=('2023-01-01', '2025-08-17')
        ))
        
        # Run validation
        result = validator.validate_symbol('BTCUSDT', symbol_files['BTCUSDT'])
        
        # Create results summary
        validation_results = {
            'symbol_results': {'BTCUSDT': result},
            'validation_summary': {
                'total_symbols': 1,
                'successful_validations': 1 if result.validation_passed else 0,
                'success_rate': 1.0 if result.validation_passed else 0.0
            }
        }
        
        # Export report
        output_dir = Path(__file__).parent / 'results' / 'regime_validation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = validator.export_validation_report(validation_results, str(output_dir))
        
        # Log key results
        baseline_wr = result.baseline_performance.get('win_rate', 0)
        regime_wr = result.regime_aware_performance.get('win_rate', 0)
        
        logger.info(f"BTCUSDT Validation Results:")
        logger.info(f"  Baseline win rate: {baseline_wr:.1%}")
        logger.info(f"  Regime-aware win rate: {regime_wr:.1%}")
        logger.info(f"  Improvement: {(regime_wr - baseline_wr) / baseline_wr:.1%}" if baseline_wr > 0 else "  No baseline trades")
        logger.info(f"  Target achieved (65%): {'YES' if regime_wr >= 0.65 else 'NO'}")
        logger.info(f"  Validation passed: {'YES' if result.validation_passed else 'NO'}")
        logger.info(f"  Report saved: {report_path}")
        
        return {
            'btc_result': result,
            'report_path': report_path,
            'validation_summary': validation_results
        }
        
    except Exception as e:
        logger.error(f"Error in BTC validation: {str(e)}")
        return {'error': str(e)}

def run_multi_symbol_validation(max_symbols: int = 5) -> dict:
    """Run validation on multiple symbols"""
    logger.info(f"Running multi-symbol validation (max {max_symbols} symbols)")
    
    symbol_files = find_symbol_data_files()
    
    if not symbol_files:
        logger.error("No symbol data files found")
        return {}
    
    # Limit to max_symbols for performance
    selected_symbols = dict(list(symbol_files.items())[:max_symbols])
    logger.info(f"Selected symbols: {list(selected_symbols.keys())}")
    
    try:
        # Create validator
        validator = RegimeAwareValidator(ValidationConfig(
            test_symbols=list(selected_symbols.keys()),
            validation_period=('2023-01-01', '2025-08-17')
        ))
        
        # Run validation
        validation_results = validator.validate_multiple_symbols(selected_symbols, max_workers=2)
        
        # Export report
        output_dir = Path(__file__).parent / 'results' / 'regime_validation'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = validator.export_validation_report(validation_results, str(output_dir))
        
        # Log summary results
        summary = validation_results.get('validation_summary', {})
        aggregate = validation_results.get('aggregate_analysis', {})
        
        logger.info(f"Multi-Symbol Validation Results:")
        logger.info(f"  Symbols tested: {summary.get('total_symbols', 0)}")
        logger.info(f"  Successful validations: {summary.get('successful_validations', 0)}")
        logger.info(f"  Success rate: {summary.get('success_rate', 0):.1%}")
        
        if aggregate:
            baseline_agg = aggregate.get('baseline_aggregate', {})
            regime_agg = aggregate.get('regime_aware_aggregate', {})
            
            logger.info(f"  Aggregate baseline win rate: {baseline_agg.get('aggregate_win_rate', 0):.1%}")
            logger.info(f"  Aggregate regime-aware win rate: {regime_agg.get('aggregate_win_rate', 0):.1%}")
            
            target_achievement = aggregate.get('target_achievement', {})
            logger.info(f"  65% target achieved: {'YES' if target_achievement.get('win_rate_target_65pct', False) else 'NO'}")
            logger.info(f"  Improvement over baseline: {'YES' if target_achievement.get('improvement_over_baseline', False) else 'NO'}")
        
        logger.info(f"  Report saved: {report_path}")
        
        return {
            'validation_results': validation_results,
            'report_path': report_path
        }
        
    except Exception as e:
        logger.error(f"Error in multi-symbol validation: {str(e)}")
        return {'error': str(e)}

def demonstrate_regime_features(symbol: str = 'BTCUSDT') -> dict:
    """Demonstrate regime-aware feature engineering"""
    logger.info(f"Demonstrating regime-aware features for {symbol}")
    
    symbol_files = find_symbol_data_files()
    
    if symbol not in symbol_files:
        logger.error(f"No data file found for {symbol}")
        return {}
    
    try:
        # Initialize feature engineer
        feature_engineer = RegimeAwareFeatureEngineer()
        
        # Process symbol data
        result = feature_engineer.process_symbol_data(symbol, symbol_files[symbol])
        
        if 'features_df' in result:
            features_df = result['features_df']
            
            # Analyze generated features
            feature_cols = [col for col in features_df.columns 
                          if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Regime analysis
            if 'regime' in features_df.columns:
                regime_dist = features_df['regime'].value_counts(normalize=True).to_dict()
                regime_confidence_avg = features_df['regime_confidence'].mean()
            else:
                regime_dist = {}
                regime_confidence_avg = 0
            
            # Feature importance (simple correlation with targets)
            feature_importance = {}
            if 'adaptive_target' in features_df.columns and 'target' in features_df.columns:
                for col in feature_cols[:20]:  # Top 20 features
                    if features_df[col].dtype in ['float64', 'int64']:
                        corr_adaptive = features_df[col].corr(features_df['adaptive_target'])
                        corr_baseline = features_df[col].corr(features_df['target'])
                        feature_importance[col] = {
                            'adaptive_correlation': corr_adaptive,
                            'baseline_correlation': corr_baseline
                        }
            
            demo_results = {
                'symbol': symbol,
                'total_features': len(feature_cols),
                'data_points': len(features_df),
                'regime_distribution': regime_dist,
                'avg_regime_confidence': regime_confidence_avg,
                'adaptive_target_rate': features_df.get('adaptive_target', pd.Series([0])).mean(),
                'baseline_target_rate': features_df.get('target', pd.Series([0])).mean(),
                'top_features': dict(list(feature_importance.items())[:10]),
                'processing_time': result.get('processing_time', 0)
            }
            
            logger.info(f"Generated {len(feature_cols)} features for {symbol}")
            logger.info(f"Regime distribution: {regime_dist}")
            logger.info(f"Adaptive target rate: {demo_results['adaptive_target_rate']:.1%}")
            logger.info(f"Baseline target rate: {demo_results['baseline_target_rate']:.1%}")
            
            return demo_results
        else:
            logger.error(f"Feature engineering failed for {symbol}")
            return {'error': 'Feature engineering failed'}
        
    except Exception as e:
        logger.error(f"Error in feature demonstration: {str(e)}")
        return {'error': str(e)}

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Regime-Aware DipMaster Strategy Validation')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', 
                       help='Symbol to analyze (default: BTCUSDT)')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick regime detection demo only')
    parser.add_argument('--btc-only', action='store_true', 
                       help='Run validation on BTCUSDT only')
    parser.add_argument('--max-symbols', type=int, default=5,
                       help='Maximum symbols for multi-symbol validation')
    parser.add_argument('--features-demo', action='store_true',
                       help='Demonstrate regime-aware feature engineering')
    
    args = parser.parse_args()
    
    logger.info("Starting Regime-Aware DipMaster Strategy Validation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Create results directory
    results_dir = Path(__file__).parent / 'results' / 'regime_validation'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_summary = {
        'validation_timestamp': datetime.now().isoformat(),
        'arguments': vars(args),
        'results': {}
    }
    
    try:
        if args.quick:
            # Quick regime detection demo
            logger.info("=== Running Quick Regime Detection Demo ===")
            demo_result = run_quick_regime_demo(args.symbol)
            results_summary['results']['regime_demo'] = demo_result
            
        elif args.features_demo:
            # Features demonstration
            logger.info("=== Running Feature Engineering Demo ===")
            features_result = demonstrate_regime_features(args.symbol)
            results_summary['results']['features_demo'] = features_result
            
        elif args.btc_only:
            # BTC-focused validation
            logger.info("=== Running BTCUSDT Focused Validation ===")
            btc_result = run_btc_focused_validation()
            results_summary['results']['btc_validation'] = btc_result
            
        else:
            # Full multi-symbol validation
            logger.info("=== Running Multi-Symbol Validation ===")
            multi_result = run_multi_symbol_validation(args.max_symbols)
            results_summary['results']['multi_symbol_validation'] = multi_result
            
            # Also run quick demos
            logger.info("=== Running Additional Demos ===")
            demo_result = run_quick_regime_demo(args.symbol)
            features_result = demonstrate_regime_features(args.symbol)
            
            results_summary['results']['regime_demo'] = demo_result
            results_summary['results']['features_demo'] = features_result
        
        # Save comprehensive results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_file = results_dir / f"validation_summary_{timestamp}.json"
        
        with open(summary_file, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        logger.info(f"Validation complete! Results saved to {summary_file}")
        
        # Print key conclusions
        print("\n" + "="*60)
        print("REGIME-AWARE DIPMASTER VALIDATION SUMMARY")
        print("="*60)
        
        if 'btc_validation' in results_summary['results']:
            btc_result = results_summary['results']['btc_validation'].get('btc_result')
            if btc_result and hasattr(btc_result, 'baseline_performance'):
                baseline_wr = btc_result.baseline_performance.get('win_rate', 0)
                regime_wr = btc_result.regime_aware_performance.get('win_rate', 0)
                print(f"BTCUSDT Performance:")
                print(f"  Baseline Win Rate: {baseline_wr:.1%}")
                print(f"  Regime-Aware Win Rate: {regime_wr:.1%}")
                print(f"  Target (65%) Achieved: {'✓' if regime_wr >= 0.65 else '✗'}")
                print(f"  Improvement: {'+' if regime_wr > baseline_wr else ''}{(regime_wr - baseline_wr) / baseline_wr:.1%}" if baseline_wr > 0 else "  No baseline comparison")
        
        if 'multi_symbol_validation' in results_summary['results']:
            multi_result = results_summary['results']['multi_symbol_validation']
            if 'validation_results' in multi_result:
                summary = multi_result['validation_results'].get('validation_summary', {})
                print(f"\nMulti-Symbol Results:")
                print(f"  Symbols Tested: {summary.get('total_symbols', 0)}")
                print(f"  Successful Validations: {summary.get('successful_validations', 0)}")
                print(f"  Success Rate: {summary.get('success_rate', 0):.1%}")
        
        if 'regime_demo' in results_summary['results']:
            demo = results_summary['results']['regime_demo']
            if 'current_regime' in demo:
                regime_info = demo['current_regime']
                print(f"\nCurrent Market Regime for {demo.get('symbol', 'N/A')}:")
                print(f"  Regime: {regime_info.get('regime', 'Unknown')}")
                print(f"  Confidence: {regime_info.get('confidence', 0):.1%}")
                print(f"  Should Trade: {'✓' if demo.get('should_trade', False) else '✗'}")
        
        print("\n" + "="*60)
        print(f"Detailed results: {summary_file}")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        print(f"\nValidation failed: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())