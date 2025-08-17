#!/usr/bin/env python3
"""
DipMaster Enhanced Feature Generation V5 - Complete Pipeline
增强版特征生成V5 - 完整管道

This script runs the complete enhanced feature engineering pipeline and generates:
1. FeatureSet.json - Complete feature set configuration
2. features.parquet - Processed feature data with labels
3. Validation reports and quality metrics

Author: DipMaster Quant Team
Date: 2025-08-17
Version: 5.0.0
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import time
import warnings

# Import our enhanced feature engineering module
from src.data.enhanced_feature_engineering_v5 import UltraEnhancedDipMasterFeatureEngineer, FeatureConfig

warnings.filterwarnings('ignore')

class EnhancedFeatureGenerationPipeline:
    """Complete pipeline for enhanced feature generation"""
    
    def __init__(self, data_dir: str = "data/enhanced_market_data"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("data")
        self.logger = self._setup_logging()
        
        # Enhanced symbols list - 25 top crypto pairs
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT',
            'BNBUSDT', 'AVAXUSDT', 'MATICUSDT', 'LINKUSDT', 'UNIUSDT',
            'LTCUSDT', 'DOTUSDT', 'ATOMUSDT', 'FILUSDT', 'NEARUSDT',
            'ARBUSDT', 'OPUSDT', 'APTUSDT', 'AAVEUSDT', 'COMPUSDT',
            'MKRUSDT', 'QNTUSDT', 'TRXUSDT', 'VETUSDT', 'XLMUSDT'
        ]
        
        # Initialize feature engineer
        config = FeatureConfig(
            symbols=self.symbols,
            timeframes=['5m', '15m', '1h'],
            prediction_horizons=[1, 3, 6, 12, 24, 36],
            profit_targets=[0.003, 0.006, 0.008, 0.012, 0.015, 0.020],
            stop_loss=0.004,
            max_holding_periods=36,
            rsi_periods=[7, 14, 21, 30, 50],
            ma_periods=[5, 10, 20, 50, 100, 200],
            volatility_periods=[5, 10, 20, 50],
            enable_cross_asset=True,
            enable_microstructure=True,
            enable_advanced_labels=True
        )
        
        self.feature_engineer = UltraEnhancedDipMasterFeatureEngineer(config)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data for all symbols"""
        self.logger.info("Loading market data for enhanced feature generation...")
        
        market_data = {}
        loaded_symbols = []
        
        for symbol in self.symbols:
            try:
                # Try to load 5-minute data (primary timeframe)
                file_pattern = f"{symbol}_5m_90days.parquet"
                file_path = self.data_dir / file_pattern
                
                if file_path.exists():
                    df = pd.read_parquet(file_path)
                    
                    # Ensure required columns exist
                    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_cols):
                        # Sort by timestamp
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        
                        # Ensure timestamp is datetime
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        
                        # Filter out invalid data
                        df = df[df['close'] > 0].copy()
                        df = df[df['volume'] >= 0].copy()
                        
                        if len(df) >= 1000:  # Minimum 1000 data points
                            market_data[symbol] = df
                            loaded_symbols.append(symbol)
                            self.logger.info(f"Loaded {symbol}: {len(df)} rows")
                        else:
                            self.logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                    else:
                        self.logger.warning(f"Missing required columns for {symbol}")
                else:
                    self.logger.warning(f"Data file not found: {file_path}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load data for {symbol}: {e}")
        
        self.logger.info(f"Successfully loaded {len(market_data)} symbols: {loaded_symbols}")
        return market_data
    
    def generate_features(self, market_data: Dict[str, pd.DataFrame]) -> tuple:
        """Generate enhanced features for all symbols"""
        self.logger.info("Starting enhanced feature generation...")
        
        # Generate comprehensive features
        processed_data, feature_stats = self.feature_engineer.generate_comprehensive_features(market_data)
        
        return processed_data, feature_stats
    
    def combine_symbol_data(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine data from all symbols into single dataset"""
        self.logger.info("Combining data from all symbols...")
        
        combined_dfs = []
        
        for symbol, df in processed_data.items():
            # Add symbol identifier
            df_copy = df.copy()
            df_copy['symbol'] = symbol
            
            # Ensure consistent columns across symbols
            combined_dfs.append(df_copy)
        
        if not combined_dfs:
            raise ValueError("No processed data to combine")
        
        # Combine all dataframes
        combined_df = pd.concat(combined_dfs, ignore_index=True, sort=False)
        
        # Sort by timestamp and symbol
        combined_df = combined_df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        
        self.logger.info(f"Combined dataset: {len(combined_df)} rows, {len(combined_df.columns)} columns")
        
        return combined_df
    
    def create_train_val_test_splits(self, combined_df: pd.DataFrame) -> Dict[str, Any]:
        """Create proper time-series splits for training/validation/testing"""
        self.logger.info("Creating train/validation/test splits...")
        
        # Sort by timestamp to ensure proper time ordering
        combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate split points (70% train, 15% validation, 15% test)
        total_samples = len(combined_df)
        train_end = int(total_samples * 0.70)
        val_end = int(total_samples * 0.85)
        
        # Create splits ensuring no data leakage
        train_df = combined_df.iloc[:train_end].copy()
        val_df = combined_df.iloc[train_end:val_end].copy()
        test_df = combined_df.iloc[val_end:].copy()
        
        # Get split timestamps for reference
        train_start = train_df['timestamp'].min()
        train_end_ts = train_df['timestamp'].max()
        val_start = val_df['timestamp'].min()
        val_end_ts = val_df['timestamp'].max()
        test_start = test_df['timestamp'].min()
        test_end_ts = test_df['timestamp'].max()
        
        split_info = {
            'train': {
                'start_date': str(train_start),
                'end_date': str(train_end_ts),
                'samples': len(train_df)
            },
            'validation': {
                'start_date': str(val_start),
                'end_date': str(val_end_ts),
                'samples': len(val_df)
            },
            'test': {
                'start_date': str(test_start),
                'end_date': str(test_end_ts),
                'samples': len(test_df)
            }
        }
        
        self.logger.info(f"Train split: {len(train_df)} samples ({train_start} to {train_end_ts})")
        self.logger.info(f"Validation split: {len(val_df)} samples ({val_start} to {val_end_ts})")
        self.logger.info(f"Test split: {len(test_df)} samples ({test_start} to {test_end_ts})")
        
        return {
            'combined_data': combined_df,
            'train_data': train_df,
            'validation_data': val_df,
            'test_data': test_df,
            'split_info': split_info
        }
    
    def analyze_feature_quality(self, combined_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature quality and generate quality metrics"""
        self.logger.info("Analyzing feature quality...")
        
        # Separate features and labels
        feature_cols = [col for col in combined_df.columns 
                       if not any(x in col for x in ['target', 'future_', 'hits_', 'time_to_', 'optimal_', 'mfe_', 'mae_'])
                       and col not in ['timestamp', 'symbol']]
        
        label_cols = [col for col in combined_df.columns 
                     if any(x in col for x in ['target', 'future_', 'hits_', 'time_to_', 'optimal_', 'mfe_', 'mae_'])]
        
        # Feature quality metrics
        feature_quality = {}
        
        for col in feature_cols:
            if combined_df[col].dtype in ['int64', 'float64']:
                quality_metrics = {
                    'nan_percentage': combined_df[col].isnull().sum() / len(combined_df),
                    'unique_values': combined_df[col].nunique(),
                    'min_value': float(combined_df[col].min()),
                    'max_value': float(combined_df[col].max()),
                    'mean_value': float(combined_df[col].mean()),
                    'std_value': float(combined_df[col].std()),
                    'zero_percentage': (combined_df[col] == 0).sum() / len(combined_df)
                }
                feature_quality[col] = quality_metrics
        
        # Label distribution analysis
        label_analysis = {}
        
        # Main target labels
        if 'target_binary' in combined_df.columns:
            target_binary = combined_df['target_binary'].dropna()
            label_analysis['target_binary'] = {
                'positive_rate': float(target_binary.mean()),
                'samples': int(len(target_binary)),
                'distribution': target_binary.value_counts().to_dict()
            }
        
        if 'dipmaster_win' in combined_df.columns:
            dipmaster_win = combined_df['dipmaster_win'].dropna()
            label_analysis['dipmaster_win'] = {
                'win_rate': float(dipmaster_win.mean()),
                'samples': int(len(dipmaster_win)),
                'distribution': dipmaster_win.value_counts().to_dict()
            }
        
        if 'return_class' in combined_df.columns:
            return_class = combined_df['return_class'].dropna()
            label_analysis['return_class'] = {
                'class_distribution': return_class.value_counts().to_dict(),
                'samples': int(len(return_class))
            }
        
        # Feature correlation analysis
        feature_data = combined_df[feature_cols].select_dtypes(include=[np.number])
        correlation_matrix = feature_data.corr()
        
        # Find highly correlated features
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.9:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        
        quality_summary = {
            'total_features': len(feature_cols),
            'total_labels': len(label_cols),
            'total_samples': len(combined_df),
            'symbols_count': combined_df['symbol'].nunique(),
            'time_range': {
                'start': str(combined_df['timestamp'].min()),
                'end': str(combined_df['timestamp'].max())
            },
            'feature_quality': feature_quality,
            'label_analysis': label_analysis,
            'high_correlation_pairs': high_corr_pairs,
            'memory_usage_mb': combined_df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return quality_summary
    
    def create_feature_set_config(self, combined_df: pd.DataFrame, 
                                 feature_stats: Dict[str, Any], 
                                 quality_analysis: Dict[str, Any],
                                 split_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive FeatureSet configuration"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get feature names by category
        feature_cols = [col for col in combined_df.columns 
                       if not any(x in col for x in ['target', 'future_', 'hits_', 'time_to_', 'optimal_', 'mfe_', 'mae_'])
                       and col not in ['timestamp', 'symbol']]
        
        label_cols = [col for col in combined_df.columns 
                     if any(x in col for x in ['target', 'future_', 'hits_', 'time_to_', 'optimal_', 'mfe_', 'mae_'])]
        
        feature_set_config = {
            "metadata": {
                "version": "5.0.0-UltraEnhanced",
                "strategy_name": "DipMaster_Enhanced_V5",
                "created_timestamp": timestamp,
                "description": "Ultra-enhanced feature set with 50+ technical indicators, microstructure features, and cross-asset signals",
                "data_source": "Binance 5-minute OHLCV data"
            },
            
            "symbols": list(combined_df['symbol'].unique()),
            
            "time_range": {
                "start_date": str(combined_df['timestamp'].min()),
                "end_date": str(combined_df['timestamp'].max()),
                "total_periods": len(combined_df),
                "timeframe": "5min"
            },
            
            "data_splits": split_info,
            
            "feature_engineering": {
                "pipeline_version": "V5-UltraEnhanced",
                "total_features": len(feature_cols),
                "total_labels": len(label_cols),
                "enhancement_components": [
                    "extended_technical_indicators_50+",
                    "enhanced_market_microstructure",
                    "cross_asset_correlation_features",
                    "market_regime_detection",
                    "feature_interactions_polynomial",
                    "optimized_15min_boundary_labels",
                    "comprehensive_validation_system"
                ],
                "feature_categories": feature_stats.get('feature_categories', {}),
                "no_leakage_validation": True
            },
            
            "technical_indicators": {
                "rsi_variants": [7, 14, 21, 30, 50],
                "macd_variants": ["12_26_9", "5_35_5", "8_21_5"],
                "stochastic_variants": ["14_3", "9_3", "21_5"],
                "moving_averages": [5, 10, 20, 50, 100, 200],
                "bollinger_bands": ["20_2.0", "20_1.5", "50_2.0"],
                "volatility_indicators": ["atr", "historical_vol", "garch_proxy"],
                "momentum_indicators": ["williams_r", "cci", "adx", "roc"],
                "volume_indicators": ["obv", "vpt", "volume_sma", "volume_ratio"]
            },
            
            "microstructure_features": {
                "order_flow_indicators": ["bid_ask_spread_proxy", "order_flow_imbalance", "depth_imbalance"],
                "liquidity_proxies": ["price_impact", "effective_spread", "vwap_deviations"],
                "intraday_patterns": ["hourly_volatility", "session_effects", "time_based_volume"],
                "transaction_cost_proxies": ["slippage_estimate", "market_impact"]
            },
            
            "cross_asset_features": {
                "relative_strength": ["1h", "4h", "1d", "3d"],
                "correlation_features": ["market_correlation", "btc_correlation", "eth_correlation"],
                "ranking_features": ["daily_rank", "weekly_rank", "volatility_rank"],
                "momentum_divergence": ["1h_divergence", "4h_divergence"],
                "sector_rotation": ["leadership_signals", "rotation_strength"],
                "market_sentiment": ["risk_on_off", "beta_analysis"]
            },
            
            "regime_detection": {
                "volatility_regimes": ["low_vol", "normal_vol", "high_vol"],
                "trend_regimes": ["strong_uptrend", "sideways", "strong_downtrend"],
                "liquidity_regimes": ["high_liquidity", "normal_liquidity", "low_liquidity"],
                "market_stress": ["drawdown_analysis", "stress_indicators"]
            },
            
            "label_specifications": {
                "primary_targets": {
                    "target_return": "12-period (1h) future return",
                    "target_binary": "Positive return binary classification",
                    "dipmaster_win": "DipMaster strategy win condition",
                    "target_risk_adjusted": "Risk-adjusted return (Sharpe-like)"
                },
                "profit_targets": [0.3, 0.6, 0.8, 1.2, 1.5, 2.0],
                "time_horizons": [1, 3, 6, 12, 24, 36],
                "boundary_optimization": "15-minute boundary exits preferred",
                "risk_management": {
                    "stop_loss": 0.4,
                    "max_holding_periods": 36,
                    "risk_adjustment": "volatility_normalized"
                }
            },
            
            "quality_metrics": quality_analysis,
            
            "validation_results": feature_stats.get('validation_results', {}),
            
            "performance_expectations": {
                "target_win_rate": "85%+",
                "target_sharpe": "2.0+",
                "max_drawdown": "3%",
                "avg_holding_time": "60-90 minutes",
                "boundary_compliance": "100%"
            },
            
            "usage_guidelines": {
                "train_validation_split": "70/15/15 time-based",
                "feature_selection": "Use mutual information + correlation filtering",
                "model_types": ["LightGBM", "XGBoost", "CatBoost", "Ensemble"],
                "cross_validation": "Purged time series CV",
                "hyperparameter_tuning": "Bayesian optimization recommended"
            },
            
            "file_specifications": {
                "feature_data_file": f"Enhanced_Features_V5_{timestamp}.parquet",
                "train_file": f"train_features_{timestamp}.parquet",
                "validation_file": f"validation_features_{timestamp}.parquet",
                "test_file": f"test_features_{timestamp}.parquet"
            }
        }
        
        return feature_set_config
    
    def save_results(self, combined_df: pd.DataFrame, feature_set_config: Dict[str, Any], 
                    split_data: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save paths
        save_paths = {}
        
        # 1. Save main feature set configuration
        config_path = self.output_dir / f"Enhanced_FeatureSet_V5_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(feature_set_config, f, indent=2, default=str)
        save_paths['feature_set_config'] = str(config_path)
        
        # 2. Save complete feature data
        features_path = self.output_dir / f"Enhanced_Features_V5_{timestamp}.parquet"
        combined_df.to_parquet(features_path, compression='snappy', index=False)
        save_paths['complete_features'] = str(features_path)
        
        # 3. Save split datasets
        train_path = self.output_dir / f"train_features_V5_{timestamp}.parquet"
        split_data['train_data'].to_parquet(train_path, compression='snappy', index=False)
        save_paths['train_features'] = str(train_path)
        
        val_path = self.output_dir / f"validation_features_V5_{timestamp}.parquet"
        split_data['validation_data'].to_parquet(val_path, compression='snappy', index=False)
        save_paths['validation_features'] = str(val_path)
        
        test_path = self.output_dir / f"test_features_V5_{timestamp}.parquet"
        split_data['test_data'].to_parquet(test_path, compression='snappy', index=False)
        save_paths['test_features'] = str(test_path)
        
        # 4. Save feature importance analysis (if available)
        feature_cols = [col for col in combined_df.columns 
                       if not any(x in col for x in ['target', 'future_', 'hits_', 'time_to_', 'optimal_', 'mfe_', 'mae_'])
                       and col not in ['timestamp', 'symbol']]
        
        if 'target_binary' in combined_df.columns and len(feature_cols) > 0:
            try:
                from sklearn.feature_selection import mutual_info_classif
                from sklearn.preprocessing import LabelEncoder
                
                # Prepare data for feature importance
                X = combined_df[feature_cols].fillna(0)
                y = combined_df['target_binary'].fillna(0)
                
                # Calculate mutual information
                mi_scores = mutual_info_classif(X, y, random_state=42)
                feature_importance = pd.DataFrame({
                    'feature': feature_cols,
                    'mutual_info_score': mi_scores
                }).sort_values('mutual_info_score', ascending=False)
                
                importance_path = self.output_dir / f"Feature_Importance_V5_{timestamp}.json"
                feature_importance.to_json(importance_path, orient='records', indent=2)
                save_paths['feature_importance'] = str(importance_path)
                
            except Exception as e:
                self.logger.warning(f"Could not calculate feature importance: {e}")
        
        self.logger.info("Results saved successfully:")
        for key, path in save_paths.items():
            self.logger.info(f"  {key}: {path}")
        
        return save_paths
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Run the complete enhanced feature generation pipeline"""
        start_time = time.time()
        
        try:
            self.logger.info("🚀 Starting DipMaster Enhanced Feature Generation V5 Pipeline")
            
            # Step 1: Load market data
            market_data = self.load_market_data()
            if not market_data:
                raise ValueError("No market data loaded")
            
            # Step 2: Generate enhanced features
            processed_data, feature_stats = self.generate_features(market_data)
            
            # Step 3: Combine symbol data
            combined_df = self.combine_symbol_data(processed_data)
            
            # Step 4: Create train/val/test splits
            split_data = self.create_train_val_test_splits(combined_df)
            
            # Step 5: Analyze feature quality
            quality_analysis = self.analyze_feature_quality(combined_df)
            
            # Step 6: Create feature set configuration
            feature_set_config = self.create_feature_set_config(
                combined_df, feature_stats, quality_analysis, split_data['split_info']
            )
            
            # Step 7: Save all results
            save_paths = self.save_results(combined_df, feature_set_config, split_data)
            
            # Final summary
            total_time = time.time() - start_time
            
            summary = {
                'pipeline_status': 'SUCCESS',
                'total_runtime_seconds': total_time,
                'symbols_processed': len(processed_data),
                'total_samples': len(combined_df),
                'total_features': len([col for col in combined_df.columns 
                                     if not any(x in col for x in ['target', 'future_', 'hits_'])
                                     and col not in ['timestamp', 'symbol']]),
                'total_labels': len([col for col in combined_df.columns 
                                   if any(x in col for x in ['target', 'future_', 'hits_'])]),
                'saved_files': save_paths,
                'feature_stats': feature_stats,
                'quality_metrics': quality_analysis
            }
            
            self.logger.info("✅ Enhanced Feature Generation V5 Pipeline Completed Successfully!")
            self.logger.info(f"⏱️  Total Runtime: {total_time:.1f} seconds")
            self.logger.info(f"📊 Generated {summary['total_features']} features and {summary['total_labels']} labels")
            self.logger.info(f"💾 Processed {summary['total_samples']} samples across {summary['symbols_processed']} symbols")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline failed: {e}")
            return {
                'pipeline_status': 'FAILED',
                'error': str(e),
                'total_runtime_seconds': time.time() - start_time
            }

def main():
    """Main execution function"""
    print("DipMaster Enhanced Feature Generation V5")
    print("=" * 50)
    
    # Initialize and run pipeline
    pipeline = EnhancedFeatureGenerationPipeline()
    results = pipeline.run_complete_pipeline()
    
    # Print summary
    print("\n" + "=" * 50)
    print("PIPELINE SUMMARY")
    print("=" * 50)
    
    if results['pipeline_status'] == 'SUCCESS':
        print(f"✅ Status: {results['pipeline_status']}")
        print(f"⏱️  Runtime: {results['total_runtime_seconds']:.1f} seconds")
        print(f"📊 Features: {results['total_features']}")
        print(f"🏷️  Labels: {results['total_labels']}")
        print(f"💾 Samples: {results['total_samples']:,}")
        print(f"🪙 Symbols: {results['symbols_processed']}")
        
        print("\n📁 Generated Files:")
        for file_type, path in results['saved_files'].items():
            print(f"   {file_type}: {path}")
        
        # Show feature categories
        if 'feature_categories' in results['feature_stats']:
            print("\n🏗️  Feature Categories:")
            for category, count in results['feature_stats']['feature_categories'].items():
                print(f"   {category}: {count}")
        
        # Show quality metrics
        if 'label_analysis' in results['quality_metrics']:
            print("\n📈 Label Quality:")
            for label, metrics in results['quality_metrics']['label_analysis'].items():
                if 'win_rate' in metrics:
                    print(f"   {label}: {metrics['win_rate']:.1%} win rate ({metrics['samples']} samples)")
                elif 'positive_rate' in metrics:
                    print(f"   {label}: {metrics['positive_rate']:.1%} positive rate ({metrics['samples']} samples)")
        
    else:
        print(f"❌ Status: {results['pipeline_status']}")
        print(f"💥 Error: {results.get('error', 'Unknown error')}")
    
    print("=" * 50)

if __name__ == "__main__":
    main()