#!/usr/bin/env python3
"""
Run Top30 Enhanced Feature Engineering Pipeline
运行30币种增强特征工程管道
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings
import time

# Import our enhanced feature engineering system
from enhanced_feature_engineering_top30_v6 import Top30EnhancedFeatureEngineer, Top30FeatureConfig

warnings.filterwarnings('ignore')

def setup_logging():
    """Setup logging configuration"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f'top30_enhanced_features_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def validate_data_availability():
    """Validate that all 30 symbols have required data"""
    logger = logging.getLogger(__name__)
    data_dir = Path('data/enhanced_market_data')
    
    required_symbols = [
        'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT', 'BNBUSDT', 
        'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT', 'LTCUSDT', 'DOTUSDT',
        'ATOMUSDT', 'FILUSDT', 'NEARUSDT', 'ARBUSDT', 'OPUSDT', 'APTUSDT',
        'AAVEUSDT', 'COMPUSDT', 'MKRUSDT', 'QNTUSDT', 'TRXUSDT', 'VETUSDT',
        'XLMUSDT', 'SHIBUSDT', 'DOGEUSDT', 'TONUSDT', 'PEPEUSDT', 'INJUSDT'
    ]
    
    available_symbols = []
    missing_symbols = []
    
    for symbol in required_symbols:
        file_path = data_dir / f"{symbol}_5m_90days.parquet"
        if file_path.exists():
            try:
                df = pd.read_parquet(file_path)
                if len(df) > 1000:  # Minimum data requirement
                    available_symbols.append(symbol)
                    logger.info(f"✅ {symbol}: {len(df)} rows")
                else:
                    missing_symbols.append(f"{symbol} (insufficient data: {len(df)} rows)")
                    logger.warning(f"⚠️ {symbol}: Insufficient data ({len(df)} rows)")
            except Exception as e:
                missing_symbols.append(f"{symbol} (read error: {e})")
                logger.error(f"❌ {symbol}: Read error - {e}")
        else:
            missing_symbols.append(f"{symbol} (file not found)")
            logger.error(f"❌ {symbol}: File not found")
    
    logger.info(f"Data validation complete: {len(available_symbols)}/30 symbols available")
    
    if missing_symbols:
        logger.warning("Missing symbols:")
        for missing in missing_symbols:
            logger.warning(f"  - {missing}")
    
    return available_symbols, missing_symbols

def save_enhanced_features(all_data: dict, results: dict, timestamp: str):
    """Save enhanced features to files"""
    logger = logging.getLogger(__name__)
    
    try:
        # Create output directory
        output_dir = Path('data')
        output_dir.mkdir(exist_ok=True)
        
        # Combine all symbol data into single dataframe
        combined_data = []
        for symbol, df in all_data.items():
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            combined_data.append(df_copy)
        
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Save main feature file
            feature_file = output_dir / f'Enhanced_Features_Top30_V6_{timestamp}.parquet'
            combined_df.to_parquet(feature_file, index=False)
            logger.info(f"💾 Main features saved: {feature_file}")
            
            # Create feature metadata
            feature_metadata = {
                'file_info': {
                    'filename': feature_file.name,
                    'creation_date': datetime.now().isoformat(),
                    'total_rows': len(combined_df),
                    'total_symbols': len(all_data),
                    'total_features': len(combined_df.columns) - 1,  # Exclude symbol column
                    'file_size_mb': feature_file.stat().st_size / (1024 * 1024)
                },
                'symbols': list(all_data.keys()),
                'feature_categories': results.get('feature_categories', {}),
                'processing_times': results.get('processing_times', {}),
                'validation_results': results.get('validation_results', {}),
                'errors': results.get('errors', [])
            }
            
            # Save metadata
            metadata_file = output_dir / f'FeatureSet_Top30_V6_{timestamp}.json'
            with open(metadata_file, 'w') as f:
                json.dump(feature_metadata, f, indent=2, default=str)
            logger.info(f"📋 Metadata saved: {metadata_file}")
            
            # Create feature importance analysis
            feature_importance = analyze_feature_importance(combined_df)
            importance_file = output_dir / f'Feature_Importance_Analysis_Top30_{timestamp}.json'
            with open(importance_file, 'w') as f:
                json.dump(feature_importance, f, indent=2, default=str)
            logger.info(f"📊 Feature importance saved: {importance_file}")
            
            # Save individual symbol data (optional, for debugging)
            symbol_dir = output_dir / 'symbol_features'
            symbol_dir.mkdir(exist_ok=True)
            
            for symbol, df in all_data.items():
                symbol_file = symbol_dir / f'{symbol}_features_{timestamp}.parquet'
                df.to_parquet(symbol_file, index=False)
            
            logger.info(f"💾 Individual symbol files saved to: {symbol_dir}")
            
            return {
                'feature_file': str(feature_file),
                'metadata_file': str(metadata_file),
                'importance_file': str(importance_file),
                'combined_rows': len(combined_df),
                'feature_count': len(combined_df.columns) - 1
            }
        else:
            logger.error("No data to save")
            return None
            
    except Exception as e:
        logger.error(f"Failed to save enhanced features: {e}")
        return None

def analyze_feature_importance(df: pd.DataFrame) -> dict:
    """Analyze feature importance and correlations"""
    logger = logging.getLogger(__name__)
    
    try:
        # Get feature columns (exclude target columns and identifiers)
        feature_cols = [col for col in df.columns 
                       if not col.startswith('target') 
                       and not col.startswith('future_')
                       and not col.startswith('optimal_')
                       and col not in ['symbol', 'timestamp']]
        
        # Basic statistics
        feature_stats = {}
        for col in feature_cols:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                stats_dict = {
                    'mean': float(df[col].mean()) if not pd.isna(df[col].mean()) else 0,
                    'std': float(df[col].std()) if not pd.isna(df[col].std()) else 0,
                    'min': float(df[col].min()) if not pd.isna(df[col].min()) else 0,
                    'max': float(df[col].max()) if not pd.isna(df[col].max()) else 0,
                    'null_pct': float(df[col].isnull().sum() / len(df)),
                    'unique_values': int(df[col].nunique())
                }
                feature_stats[col] = stats_dict
        
        # Feature categories
        categories = {
            'symbol_specific': [col for col in feature_cols if any(x in col.lower() for x in ['adaptive', 'symbol', 'tier'])],
            'cross_symbol': [col for col in feature_cols if any(x in col.lower() for x in ['market_', 'sector_', 'correlation', 'rank'])],
            'microstructure': [col for col in feature_cols if any(x in col.lower() for x in ['vwap', 'flow', 'pressure', 'spread'])],
            'technical': [col for col in feature_cols if any(x in col.lower() for x in ['rsi', 'macd', 'momentum', 'sma', 'ema'])],
            'regime': [col for col in feature_cols if any(x in col.lower() for x in ['regime', 'vol_', 'trend_'])],
            'time_based': [col for col in feature_cols if any(x in col.lower() for x in ['hour', 'session', 'weekend'])],
            'volume': [col for col in feature_cols if 'volume' in col.lower() and 'vwap' not in col.lower()],
            'volatility': [col for col in feature_cols if any(x in col.lower() for x in ['vol_', 'volatility', 'atr'])],
            'pattern': [col for col in feature_cols if any(x in col.lower() for x in ['pattern', 'hammer', 'doji', 'wick'])],
            'other': []
        }
        
        # Assign uncategorized features
        categorized = set()
        for cat_features in categories.values():
            categorized.update(cat_features)
        
        categories['other'] = [col for col in feature_cols if col not in categorized]
        
        # Category statistics
        category_stats = {}
        for category, cols in categories.items():
            category_stats[category] = {
                'count': len(cols),
                'percentage': len(cols) / len(feature_cols) * 100 if feature_cols else 0
            }
        
        # High correlation pairs (potential redundancy)
        high_corr_pairs = []
        if len(feature_cols) > 1:
            try:
                # Sample data for correlation analysis (to avoid memory issues)
                sample_size = min(10000, len(df))
                sample_df = df[feature_cols].sample(n=sample_size, random_state=42) if len(df) > sample_size else df[feature_cols]
                
                # Calculate correlation matrix
                corr_matrix = sample_df.corr()
                
                # Find high correlation pairs
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.8 and not pd.isna(corr_val):
                            high_corr_pairs.append({
                                'feature_1': corr_matrix.columns[i],
                                'feature_2': corr_matrix.columns[j],
                                'correlation': float(corr_val)
                            })
                
                # Sort by absolute correlation
                high_corr_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
                high_corr_pairs = high_corr_pairs[:20]  # Top 20 pairs
                
            except Exception as e:
                logger.warning(f"Correlation analysis failed: {e}")
                high_corr_pairs = []
        
        return {
            'total_features': len(feature_cols),
            'feature_statistics': feature_stats,
            'category_breakdown': category_stats,
            'categories': {k: v for k, v in categories.items() if v},  # Only non-empty categories
            'high_correlation_pairs': high_corr_pairs,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Feature importance analysis failed: {e}")
        return {'error': str(e)}

def main():
    """Main execution function"""
    print("DipMaster Top30 Enhanced Feature Engineering V6")
    print("=" * 60)
    
    # Setup logging
    logger = setup_logging()
    
    # Validate data availability
    print("\n📊 Validating data availability...")
    available_symbols, missing_symbols = validate_data_availability()
    
    if len(available_symbols) < 25:
        print(f"❌ Insufficient symbols available: {len(available_symbols)}/30")
        print("Please ensure all required data files are downloaded.")
        return
    
    print(f"✅ Data validation passed: {len(available_symbols)}/30 symbols available")
    
    # Initialize feature engineer
    print("\n🔧 Initializing Top30 Enhanced Feature Engineer...")
    config = Top30FeatureConfig()
    
    # Update config to use only available symbols
    config.symbols = available_symbols
    
    engineer = Top30EnhancedFeatureEngineer(config)
    
    # Generate features
    print(f"\n⚙️ Generating enhanced features for {len(available_symbols)} symbols...")
    print("This may take several minutes...")
    
    start_time = time.time()
    
    try:
        all_data, results = engineer.generate_top30_features()
        
        processing_time = time.time() - start_time
        
        if not all_data:
            print("❌ Feature generation failed!")
            logger.error("No data returned from feature generation")
            return
        
        print(f"✅ Feature generation completed in {processing_time:.1f}s")
        print(f"📈 Processed {len(all_data)} symbols")
        print(f"🎯 Generated {results.get('total_features', 0)} features")
        
        # Print feature breakdown
        if 'feature_categories' in results:
            print("\n📊 Feature Categories:")
            for category, count in results['feature_categories'].items():
                print(f"  - {category}: {count}")
        
        # Print processing times
        if 'processing_times' in results:
            print("\n⏱️ Processing Times:")
            for step, duration in results['processing_times'].items():
                print(f"  - {step}: {duration:.1f}s")
        
        # Print any errors
        if results.get('errors'):
            print(f"\n⚠️ Errors encountered: {len(results['errors'])}")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
        
        # Save results
        print("\n💾 Saving enhanced features...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        save_results = save_enhanced_features(all_data, results, timestamp)
        
        if save_results:
            print("✅ Features saved successfully!")
            print(f"📁 Main file: {save_results['feature_file']}")
            print(f"📋 Metadata: {save_results['metadata_file']}")
            print(f"📊 Feature analysis: {save_results['importance_file']}")
            print(f"📈 Total rows: {save_results['combined_rows']:,}")
            print(f"🎯 Total features: {save_results['feature_count']}")
        else:
            print("❌ Failed to save features!")
            
        print("\n🎉 Top30 Enhanced Feature Engineering completed!")
        print(f"⏱️ Total runtime: {time.time() - start_time:.1f}s")
        
    except Exception as e:
        logger.error(f"Feature generation failed: {e}")
        print(f"❌ Error: {e}")
        return

if __name__ == "__main__":
    main()