#!/usr/bin/env python3
"""
Generate comprehensive feature validation report
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime

def validate_no_leakage(df):
    """Validate that no future information is leaked into features"""
    validation_results = {
        'leakage_detected': False,
        'suspicious_features': [],
        'validation_passed': True
    }
    
    # Get feature columns (excluding targets)
    feature_cols = [col for col in df.columns 
                   if not any(x in col for x in ['target', 'future_', 'hits_', 'return_class'])
                   and col not in ['timestamp', 'symbol']]
    
    # Check correlations with main target
    target_col = 'target_return'
    if target_col in df.columns:
        target_data = df[target_col].dropna()
        
        for feature_col in feature_cols:
            if df[feature_col].dtype in ['int64', 'float64']:
                feature_data = df[feature_col].dropna()
                
                # Align data
                aligned_data = pd.concat([feature_data, target_data], axis=1, join='inner')
                if len(aligned_data) > 100:
                    corr = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                    
                    # Flag suspiciously high correlations
                    if abs(corr) > 0.7:  # Threshold for suspicion
                        validation_results['suspicious_features'].append({
                            'feature': feature_col,
                            'correlation': float(corr),
                            'reason': 'High correlation with future return'
                        })
    
    # Check for future-looking features (basic temporal validation)
    for feature_col in feature_cols:
        if any(keyword in feature_col.lower() for keyword in ['future', 'next', 'ahead', 'forward']):
            validation_results['suspicious_features'].append({
                'feature': feature_col,
                'reason': 'Feature name suggests future information'
            })
    
    if len(validation_results['suspicious_features']) > 0:
        validation_results['leakage_detected'] = True
        validation_results['validation_passed'] = False
    
    return validation_results

def calculate_feature_importance(df):
    """Calculate feature importance using mutual information"""
    try:
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.preprocessing import LabelEncoder
        
        # Get features and target
        feature_cols = [col for col in df.columns 
                       if not any(x in col for x in ['target', 'future_', 'hits_', 'return_class'])
                       and col not in ['timestamp', 'symbol']]
        
        # Prepare data
        X = df[feature_cols].fillna(0)
        y = df['target_return'].fillna(0)
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
        return feature_importance.to_dict('records')
        
    except Exception as e:
        print(f"Feature importance calculation failed: {e}")
        return []

def analyze_signal_performance(df):
    """Analyze DipMaster signal performance across different conditions"""
    
    analysis = {}
    
    # Overall signal performance
    high_signals = df[df['high_signal'] == 1]
    medium_signals = df[df['medium_signal'] == 1]
    
    analysis['signal_performance'] = {
        'high_signal_count': int(len(high_signals)),
        'high_signal_win_rate': float(high_signals['dipmaster_win'].mean()) if len(high_signals) > 0 else 0.0,
        'medium_signal_count': int(len(medium_signals)),
        'medium_signal_win_rate': float(medium_signals['dipmaster_win'].mean()) if len(medium_signals) > 0 else 0.0
    }
    
    # Performance by symbol
    symbol_performance = {}
    for symbol in df['symbol'].unique():
        symbol_data = df[df['symbol'] == symbol]
        symbol_high_signals = symbol_data[symbol_data['high_signal'] == 1]
        
        symbol_performance[symbol] = {
            'total_samples': int(len(symbol_data)),
            'high_signals': int(len(symbol_high_signals)),
            'high_signal_win_rate': float(symbol_high_signals['dipmaster_win'].mean()) if len(symbol_high_signals) > 0 else 0.0,
            'overall_win_rate': float(symbol_data['dipmaster_win'].mean())
        }
    
    analysis['symbol_performance'] = symbol_performance
    
    # Performance by time of day
    hourly_performance = {}
    for hour in range(24):
        hour_data = df[df['hour'] == hour]
        hour_signals = hour_data[hour_data['high_signal'] == 1]
        
        if len(hour_data) > 100:  # Minimum sample size
            hourly_performance[str(hour)] = {
                'samples': int(len(hour_data)),
                'signals': int(len(hour_signals)),
                'win_rate': float(hour_signals['dipmaster_win'].mean()) if len(hour_signals) > 0 else 0.0
            }
    
    analysis['hourly_performance'] = hourly_performance
    
    # Performance at 15-minute boundaries
    boundary_data = df[df['minute_boundary'] == 1]
    boundary_signals = boundary_data[boundary_data['high_signal'] == 1]
    
    analysis['boundary_performance'] = {
        'boundary_samples': int(len(boundary_data)),
        'boundary_signals': int(len(boundary_signals)),
        'boundary_win_rate': float(boundary_signals['dipmaster_win'].mean()) if len(boundary_signals) > 0 else 0.0
    }
    
    return analysis

def generate_validation_report():
    """Generate comprehensive validation report"""
    
    # Load data
    df = pd.read_parquet('data/Enhanced_Features_V5_20250817_143704.parquet')
    
    print("Generating Feature Validation Report...")
    
    # Basic information
    report = {
        'metadata': {
            'report_generated': datetime.now().isoformat(),
            'data_file': 'Enhanced_Features_V5_20250817_143704.parquet',
            'total_samples': int(len(df)),
            'symbols': list(df['symbol'].unique()),
            'time_range': {
                'start': str(df['timestamp'].min()),
                'end': str(df['timestamp'].max())
            }
        }
    }
    
    # Data leakage validation
    print("Validating data leakage...")
    leakage_validation = validate_no_leakage(df)
    report['leakage_validation'] = leakage_validation
    
    # Feature importance
    print("Calculating feature importance...")
    feature_importance = calculate_feature_importance(df)
    report['feature_importance'] = feature_importance[:20]  # Top 20 features
    
    # Signal performance analysis
    print("Analyzing signal performance...")
    signal_analysis = analyze_signal_performance(df)
    report['signal_analysis'] = signal_analysis
    
    # Data quality metrics
    print("Calculating data quality metrics...")
    feature_cols = [col for col in df.columns 
                   if not any(x in col for x in ['target', 'future_', 'hits_'])
                   and col not in ['timestamp', 'symbol']]
    
    quality_metrics = {}
    for col in feature_cols:
        if df[col].dtype in ['int64', 'float64']:
            quality_metrics[col] = {
                'nan_percentage': float(df[col].isnull().sum() / len(df)),
                'unique_values': int(df[col].nunique()),
                'zero_percentage': float((df[col] == 0).sum() / len(df)),
                'infinite_values': int(np.isinf(df[col]).sum()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
    
    report['data_quality'] = quality_metrics
    
    # Performance summary
    report['performance_summary'] = {
        'overall_win_rate': float(df['dipmaster_win'].mean()),
        'high_signal_win_rate': float(df[df['high_signal'] == 1]['dipmaster_win'].mean()) if len(df[df['high_signal'] == 1]) > 0 else 0.0,
        'target_0_6_percent_hit_rate': float(df['target_0.6%'].mean()),
        'positive_return_rate': float(df['target_binary'].mean()),
        'signal_coverage': {
            'high_signal_percentage': float(df['high_signal'].mean()),
            'medium_signal_percentage': float(df['medium_signal'].mean()),
            'total_signal_percentage': float((df['high_signal'] | df['medium_signal']).mean())
        }
    }
    
    # Save report
    report_file = f"data/Feature_Validation_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"Validation report saved to: {report_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("FEATURE VALIDATION SUMMARY")
    print("=" * 60)
    
    print(f"Total Samples: {report['metadata']['total_samples']:,}")
    print(f"Symbols: {len(report['metadata']['symbols'])}")
    print(f"Time Range: {report['metadata']['time_range']['start']} to {report['metadata']['time_range']['end']}")
    
    print(f"\nData Leakage Validation:")
    print(f"  Status: {'FAILED' if leakage_validation['leakage_detected'] else 'PASSED'}")
    if leakage_validation['suspicious_features']:
        print(f"  Suspicious features: {len(leakage_validation['suspicious_features'])}")
        for feature in leakage_validation['suspicious_features'][:5]:
            print(f"    - {feature['feature']}: {feature['reason']}")
    
    print(f"\nDipMaster Performance:")
    perf = report['performance_summary']
    print(f"  Overall win rate: {perf['overall_win_rate']:.1%}")
    print(f"  High signal win rate: {perf['high_signal_win_rate']:.1%}")
    print(f"  0.6% profit hit rate: {perf['target_0_6_percent_hit_rate']:.1%}")
    print(f"  Signal coverage: {perf['signal_coverage']['total_signal_percentage']:.1%}")
    
    print(f"\nTop 10 Most Important Features:")
    for i, feature in enumerate(feature_importance[:10]):
        print(f"  {i+1:2d}. {feature['feature']}: {feature['importance']:.4f}")
    
    print(f"\nBest Performing Symbols:")
    for symbol, performance in signal_analysis['symbol_performance'].items():
        print(f"  {symbol}: {performance['high_signal_win_rate']:.1%} win rate ({performance['high_signals']} signals)")
    
    print("=" * 60)
    
    return report_file

if __name__ == "__main__":
    generate_validation_report()