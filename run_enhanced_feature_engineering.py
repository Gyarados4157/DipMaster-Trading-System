#!/usr/bin/env python3
"""
Enhanced Feature Engineering Runner for DipMaster V4
增强特征工程运行器 - 应用到25个币种数据

This script applies the enhanced feature engineering pipeline to the complete
25-symbol dataset, generating comprehensive features for the DipMaster strategy
optimization targeting 85%+ win rate.

Author: DipMaster Quant Team
Date: 2025-08-16
Version: 4.0.0-Enhanced
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.enhanced_feature_engineering_v4 import EnhancedDipMasterFeatureEngineer

warnings.filterwarnings('ignore')

class EnhancedFeatureRunner:
    """Enhanced Feature Engineering Runner"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Enhanced 25-symbol configuration
        self.config = {
            'symbols': [
                'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT', 'XRPUSDT',
                'BNBUSDT', 'DOGEUSDT', 'SUIUSDT', 'ICPUSDT', 'ALGOUSDT', 
                'IOTAUSDT', 'TRXUSDT', 'LINKUSDT', 'AVAXUSDT', 'MATICUSDT', 
                'UNIUSDT', 'LTCUSDT', 'DOTUSDT', 'ATOMUSDT', 'FILUSDT', 
                'NEARUSDT', 'ARBUSDT', 'OPUSDT', 'APTUSDT', 'QNTUSDT'
            ],
            'timeframes': ['5m', '15m', '1h'],
            'data_path': 'data/enhanced_market_data',
            'output_path': 'data',
            'min_samples': 1000,  # Minimum samples required per symbol
        }
        
        self.engineer = EnhancedDipMasterFeatureEngineer(self.config)
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/enhanced_feature_engineering.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_symbol_data(self, symbol: str) -> pd.DataFrame:
        """Load data for a specific symbol"""
        try:
            data_path = Path(self.config['data_path'])
            
            # Try to load 5m data (primary timeframe)
            file_patterns = [
                f"{symbol}_5m_90days.parquet",
                f"{symbol}_5m_3years.parquet",
            ]
            
            for pattern in file_patterns:
                file_path = data_path / pattern
                if file_path.exists():
                    self.logger.info(f"Loading {symbol} data from {file_path}")
                    df = pd.read_parquet(file_path)
                    
                    # Ensure required columns exist
                    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in df.columns for col in required_cols):
                        # Convert timestamp if needed
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df = df.sort_values('timestamp').reset_index(drop=True)
                        
                        # Remove duplicates
                        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
                        
                        self.logger.info(f"Loaded {len(df)} samples for {symbol}")
                        return df
            
            self.logger.warning(f"No suitable data file found for {symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Failed to load data for {symbol}: {e}")
            return pd.DataFrame()
    
    def load_all_symbol_data(self) -> Dict[str, pd.DataFrame]:
        """Load data for all symbols"""
        self.logger.info("Loading data for all symbols...")
        
        data_dict = {}
        successful_loads = 0
        
        for symbol in self.config['symbols']:
            df = self.load_symbol_data(symbol)
            
            if len(df) >= self.config['min_samples']:
                data_dict[symbol] = df
                successful_loads += 1
                self.logger.info(f"✓ {symbol}: {len(df)} samples loaded")
            else:
                self.logger.warning(f"✗ {symbol}: Insufficient data ({len(df)} samples)")
        
        self.logger.info(f"Successfully loaded {successful_loads}/{len(self.config['symbols'])} symbols")
        return data_dict
    
    def generate_enhanced_features(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Generate enhanced features for all symbols"""
        try:
            self.logger.info("Starting enhanced feature generation...")
            
            # Process enhanced features
            enhanced_data = self.engineer.process_enhanced_features(data_dict)
            
            # Combine all symbol data
            combined_features = []
            
            for symbol, df in enhanced_data.items():
                if len(df) > 0:
                    # Add symbol identifier
                    df['symbol'] = symbol
                    
                    # Remove rows with NaN targets
                    df = df.dropna(subset=['target_return', 'target_binary'])
                    
                    if len(df) > 0:
                        combined_features.append(df)
                        self.logger.info(f"{symbol}: {len(df)} samples with features")
            
            if combined_features:
                # Combine all dataframes
                final_features = pd.concat(combined_features, ignore_index=True)
                self.logger.info(f"Combined features: {len(final_features)} total samples, {len(final_features.columns)} features")
                return final_features
            else:
                self.logger.error("No valid feature data generated")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Feature generation failed: {e}")
            return pd.DataFrame()
    
    def analyze_feature_quality(self, features_df: pd.DataFrame) -> Dict:
        """Analyze feature quality and generate report"""
        try:
            self.logger.info("Analyzing feature quality...")
            
            # Basic statistics
            total_samples = len(features_df)
            total_features = len(features_df.columns)
            symbols = features_df['symbol'].nunique() if 'symbol' in features_df.columns else 0
            
            # Feature categories
            feature_categories = {
                'dipmaster_core': [col for col in features_df.columns if any(x in col for x in ['rsi_', 'bb_', 'ma_', 'dip', 'signal'])],
                'microstructure': [col for col in features_df.columns if any(x in col for x in ['price_impact', 'order_flow', 'vwap', 'liquidity'])],
                'cross_symbol': [col for col in features_df.columns if any(x in col for x in ['relative_strength', 'correlation', 'momentum_divergence'])],
                'ml_features': [col for col in features_df.columns if any(x in col for x in ['pca_', 'interact_', 'skew', 'kurtosis'])],
                'advanced_signals': [col for col in features_df.columns if any(x in col for x in ['dipmaster_v4', 'signal_confidence', 'regime_'])],
                'labels': [col for col in features_df.columns if any(x in col for x in ['target', 'future_return', 'is_profitable', 'hits_'])]
            }
            
            # Missing value analysis
            missing_values = features_df.isnull().sum()
            missing_pct = (missing_values / len(features_df) * 100).round(2)
            
            # Feature correlations with targets
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            target_cols = [col for col in numeric_cols if col.startswith('target')]
            
            correlations = {}
            if 'target_binary' in features_df.columns:
                feature_cols = [col for col in numeric_cols if not any(x in col for x in ['target', 'future_return'])]
                corr_with_target = features_df[feature_cols + ['target_binary']].corr()['target_binary'].abs().sort_values(ascending=False)
                correlations['top_features'] = corr_with_target.head(20).to_dict()
            
            # Signal quality analysis
            signal_quality = {}
            if 'dipmaster_v4_final_signal' in features_df.columns:
                signal_col = 'dipmaster_v4_final_signal'
                signal_quality['mean_signal_strength'] = features_df[signal_col].mean()
                signal_quality['signal_distribution'] = features_df[signal_col].describe().to_dict()
                
                # Signal vs performance analysis
                if 'target_binary' in features_df.columns:
                    high_signal = features_df[features_df[signal_col] > 0.7]
                    if len(high_signal) > 0:
                        signal_quality['high_signal_win_rate'] = high_signal['target_binary'].mean()
                        signal_quality['high_signal_samples'] = len(high_signal)
            
            # Cross-symbol analysis
            cross_symbol_analysis = {}
            if 'symbol' in features_df.columns:
                symbol_performance = features_df.groupby('symbol').agg({
                    'target_binary': ['count', 'mean'],
                    'target_return': 'mean'
                }).round(4)
                
                cross_symbol_analysis['symbol_performance'] = symbol_performance.to_dict()
                cross_symbol_analysis['best_symbols'] = symbol_performance[('target_binary', 'mean')].nlargest(5).to_dict()
                cross_symbol_analysis['worst_symbols'] = symbol_performance[('target_binary', 'mean')].nsmallest(5).to_dict()
            
            # Quality report
            quality_report = {
                'basic_statistics': {
                    'total_samples': int(total_samples),
                    'total_features': int(total_features),
                    'symbols_count': int(symbols),
                    'date_range': {
                        'start': str(features_df['timestamp'].min()) if 'timestamp' in features_df.columns else 'N/A',
                        'end': str(features_df['timestamp'].max()) if 'timestamp' in features_df.columns else 'N/A'
                    },
                    'memory_usage_mb': float(features_df.memory_usage(deep=True).sum() / 1024 / 1024)
                },
                'feature_categories': {k: len(v) for k, v in feature_categories.items()},
                'data_quality': {
                    'missing_values_count': int(missing_values.sum()),
                    'features_with_missing': int((missing_values > 0).sum()),
                    'worst_missing_features': missing_pct[missing_pct > 5].to_dict()
                },
                'feature_importance': correlations,
                'signal_quality': signal_quality,
                'cross_symbol_analysis': cross_symbol_analysis,
                'enhanced_features_summary': {
                    'multi_timeframe_rsi': len([col for col in features_df.columns if 'rsi_' in col and 'convergence' in col]),
                    'microstructure_features': len(feature_categories['microstructure']),
                    'cross_symbol_features': len(feature_categories['cross_symbol']),
                    'ml_derived_features': len(feature_categories['ml_features']),
                    'advanced_signals': len(feature_categories['advanced_signals'])
                }
            }
            
            return quality_report
            
        except Exception as e:
            self.logger.error(f"Feature quality analysis failed: {e}")
            return {}
    
    def save_enhanced_features(self, features_df: pd.DataFrame) -> Tuple[str, str, str]:
        """Save enhanced features and generate reports"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save main feature dataset
            features_file = f"data/Enhanced_Features_25symbols_{timestamp}.parquet"
            features_df.to_parquet(features_file, compression='snappy')
            self.logger.info(f"Enhanced features saved: {features_file}")
            
            # Generate quality report
            quality_report = self.analyze_feature_quality(features_df)
            
            # Save feature configuration
            feature_config = {
                'version': '4.0.0-Enhanced',
                'strategy_name': 'DipMaster_Enhanced_V4',
                'created_timestamp': timestamp,
                'symbols': list(features_df['symbol'].unique()) if 'symbol' in features_df.columns else [],
                'feature_engineering': {
                    'pipeline_type': 'enhanced_multi_layer',
                    'total_features': len(features_df.columns),
                    'enhancement_components': [
                        'multi_timeframe_rsi_convergence',
                        'dynamic_rsi_thresholds',
                        'enhanced_dip_detection',
                        'cross_symbol_relative_strength',
                        'market_microstructure',
                        'advanced_ml_features',
                        'multi_layer_signal_scoring',
                        'enhanced_risk_adjusted_labels',
                        'survival_analysis_labels',
                        'pca_dimensionality_reduction'
                    ]
                },
                'target_specification': {
                    'primary_target': 'target_binary',
                    'return_target': 'target_return',
                    'risk_adjusted_target': 'target_risk_adjusted',
                    'prediction_horizon': '1_hour',
                    'profit_targets': [0.6, 1.2, 2.0],
                    'stop_loss': 0.4
                },
                'quality_metrics': quality_report
            }
            
            config_file = f"data/Enhanced_FeatureSet_V4_{timestamp}.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(feature_config, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Feature configuration saved: {config_file}")
            
            # Generate comprehensive analysis report
            analysis_report = self.generate_comprehensive_analysis(features_df, quality_report)
            
            report_file = f"data/Feature_Quality_Report_{timestamp}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_report, f, indent=2, ensure_ascii=False, default=str)
            self.logger.info(f"Quality analysis report saved: {report_file}")
            
            return features_file, config_file, report_file
            
        except Exception as e:
            self.logger.error(f"Failed to save enhanced features: {e}")
            return "", "", ""
    
    def generate_comprehensive_analysis(self, features_df: pd.DataFrame, quality_report: Dict) -> Dict:
        """Generate comprehensive feature analysis"""
        try:
            # Feature importance analysis
            feature_importance = {}
            if 'target_binary' in features_df.columns:
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                feature_cols = [col for col in numeric_cols if not any(x in col for x in ['target', 'future_return', 'timestamp'])]
                
                # Calculate correlations
                correlations = features_df[feature_cols + ['target_binary']].corr()['target_binary'].abs()
                feature_importance['correlation_ranking'] = correlations.sort_values(ascending=False).head(30).to_dict()
                
                # DipMaster signal analysis
                dipmaster_features = [col for col in feature_cols if 'dipmaster' in col]
                if dipmaster_features:
                    dipmaster_corr = correlations[dipmaster_features].sort_values(ascending=False)
                    feature_importance['dipmaster_signal_ranking'] = dipmaster_corr.to_dict()
            
            # Signal strength distribution analysis
            signal_analysis = {}
            if 'dipmaster_v4_final_signal' in features_df.columns:
                signal_col = 'dipmaster_v4_final_signal'
                
                # Signal quartile analysis
                quartiles = features_df[signal_col].quantile([0.25, 0.5, 0.75, 0.95])
                signal_analysis['signal_quartiles'] = quartiles.to_dict()
                
                # Performance by signal strength
                if 'target_binary' in features_df.columns:
                    signal_bins = pd.cut(features_df[signal_col], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                    performance_by_signal = features_df.groupby(signal_bins)['target_binary'].agg(['count', 'mean']).round(4)
                    signal_analysis['performance_by_signal_strength'] = performance_by_signal.to_dict()
            
            # Cross-symbol insights
            cross_symbol_insights = {}
            if 'symbol' in features_df.columns:
                symbol_stats = features_df.groupby('symbol').agg({
                    'target_binary': ['count', 'mean', 'std'],
                    'target_return': ['mean', 'std'],
                    'dipmaster_v4_final_signal': 'mean' if 'dipmaster_v4_final_signal' in features_df.columns else lambda x: 0
                }).round(4)
                
                cross_symbol_insights['symbol_statistics'] = symbol_stats.to_dict()
                
                # Symbol correlation matrix
                if 'relative_strength_1d' in features_df.columns:
                    symbol_pivot = features_df.pivot_table(
                        values='relative_strength_1d', 
                        index='timestamp', 
                        columns='symbol'
                    )
                    symbol_corr = symbol_pivot.corr()
                    cross_symbol_insights['symbol_correlations'] = symbol_corr.to_dict()
            
            # Feature validation insights
            validation_insights = {
                'feature_stability': self.analyze_feature_stability(features_df),
                'temporal_consistency': self.analyze_temporal_patterns(features_df),
                'leakage_detection': self.detect_potential_leakage(features_df)
            }
            
            return {
                'analysis_timestamp': datetime.now().isoformat(),
                'feature_importance_analysis': feature_importance,
                'signal_strength_analysis': signal_analysis,
                'cross_symbol_insights': cross_symbol_insights,
                'validation_insights': validation_insights,
                'enhancement_summary': {
                    'total_enhancements': 10,
                    'key_innovations': [
                        'Multi-timeframe RSI convergence with dynamic thresholds',
                        'Cross-symbol relative strength and momentum analysis',
                        'Market microstructure order flow features',
                        'PCA-based dimensionality reduction',
                        'Multi-layer signal scoring system',
                        'Enhanced risk-adjusted labels',
                        'Survival analysis for optimal timing',
                        'Advanced feature interactions',
                        'Regime-aware signal filtering',
                        'Comprehensive quality validation'
                    ]
                },
                'performance_expectations': {
                    'target_win_rate': '85%+',
                    'expected_sharpe': '2.0+',
                    'max_drawdown': '<3%',
                    'feature_quality_score': quality_report.get('quality_metrics', {}).get('overall_score', 'N/A')
                }
            }
            
        except Exception as e:
            self.logger.error(f"Comprehensive analysis failed: {e}")
            return {}
    
    def analyze_feature_stability(self, features_df: pd.DataFrame) -> Dict:
        """Analyze feature stability across time"""
        try:
            if 'timestamp' not in features_df.columns:
                return {}
            
            # Split data into time periods
            features_df['timestamp'] = pd.to_datetime(features_df['timestamp'])
            features_df = features_df.sort_values('timestamp')
            
            n_periods = 5
            period_size = len(features_df) // n_periods
            
            stability_metrics = {}
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in numeric_cols if not any(x in col for x in ['target', 'future_return', 'timestamp'])]
            
            # Calculate mean and std for each period
            for i in range(n_periods):
                start_idx = i * period_size
                end_idx = (i + 1) * period_size if i < n_periods - 1 else len(features_df)
                period_data = features_df.iloc[start_idx:end_idx]
                
                period_stats = period_data[feature_cols].agg(['mean', 'std']).round(4)
                stability_metrics[f'period_{i+1}'] = period_stats.to_dict()
            
            return stability_metrics
            
        except Exception as e:
            self.logger.error(f"Feature stability analysis failed: {e}")
            return {}
    
    def analyze_temporal_patterns(self, features_df: pd.DataFrame) -> Dict:
        """Analyze temporal patterns in features"""
        try:
            temporal_analysis = {}
            
            if 'timestamp' in features_df.columns:
                features_df['hour'] = pd.to_datetime(features_df['timestamp']).dt.hour
                features_df['day_of_week'] = pd.to_datetime(features_df['timestamp']).dt.dayofweek
                
                # Hourly patterns
                if 'target_binary' in features_df.columns:
                    hourly_performance = features_df.groupby('hour')['target_binary'].agg(['count', 'mean']).round(4)
                    temporal_analysis['hourly_patterns'] = hourly_performance.to_dict()
                
                # Day of week patterns
                if 'target_binary' in features_df.columns:
                    daily_performance = features_df.groupby('day_of_week')['target_binary'].agg(['count', 'mean']).round(4)
                    temporal_analysis['daily_patterns'] = daily_performance.to_dict()
            
            return temporal_analysis
            
        except Exception as e:
            self.logger.error(f"Temporal analysis failed: {e}")
            return {}
    
    def detect_potential_leakage(self, features_df: pd.DataFrame) -> Dict:
        """Detect potential data leakage"""
        try:
            leakage_analysis = {}
            
            # Check for suspiciously high correlations with targets
            if 'target_binary' in features_df.columns:
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                feature_cols = [col for col in numeric_cols if not any(x in col for x in ['target', 'future_return'])]
                
                correlations = features_df[feature_cols + ['target_binary']].corr()['target_binary'].abs()
                suspicious_features = correlations[correlations > 0.95].drop('target_binary', errors='ignore')
                
                leakage_analysis['suspicious_high_correlation'] = suspicious_features.to_dict()
                leakage_analysis['max_correlation'] = float(correlations.max())
                leakage_analysis['features_above_90_pct'] = len(correlations[correlations > 0.9])
            
            return leakage_analysis
            
        except Exception as e:
            self.logger.error(f"Leakage detection failed: {e}")
            return {}
    
    def run_enhanced_feature_engineering(self):
        """Run the complete enhanced feature engineering pipeline"""
        try:
            self.logger.info("=" * 80)
            self.logger.info("ENHANCED DIPMASTER V4 FEATURE ENGINEERING PIPELINE")
            self.logger.info("=" * 80)
            self.logger.info(f"Target: 85%+ Win Rate with 25-Symbol Enhanced Features")
            self.logger.info(f"Symbols: {len(self.config['symbols'])}")
            self.logger.info(f"Enhancements: 10 Major Feature Engineering Innovations")
            
            # Step 1: Load all symbol data
            self.logger.info("\nStep 1: Loading 25-symbol market data...")
            data_dict = self.load_all_symbol_data()
            
            if len(data_dict) < 5:
                self.logger.error(f"Insufficient symbol data loaded: {len(data_dict)}")
                return False
            
            # Step 2: Generate enhanced features
            self.logger.info(f"\nStep 2: Generating enhanced features for {len(data_dict)} symbols...")
            enhanced_features = self.generate_enhanced_features(data_dict)
            
            if len(enhanced_features) == 0:
                self.logger.error("Enhanced feature generation failed")
                return False
            
            # Step 3: Save results
            self.logger.info(f"\nStep 3: Saving enhanced features and analysis...")
            features_file, config_file, report_file = self.save_enhanced_features(enhanced_features)
            
            if features_file:
                self.logger.info("\n" + "=" * 80)
                self.logger.info("ENHANCED FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
                self.logger.info("=" * 80)
                self.logger.info(f"Enhanced Features: {features_file}")
                self.logger.info(f"Configuration: {config_file}")
                self.logger.info(f"Quality Report: {report_file}")
                self.logger.info(f"Total Samples: {len(enhanced_features):,}")
                self.logger.info(f"Total Features: {len(enhanced_features.columns)}")
                self.logger.info(f"Symbols Processed: {enhanced_features['symbol'].nunique() if 'symbol' in enhanced_features.columns else 0}")
                
                # Display key feature categories
                feature_categories = {
                    'DipMaster Core': len([col for col in enhanced_features.columns if any(x in col for x in ['rsi_', 'bb_', 'ma_', 'dip'])]),
                    'Microstructure': len([col for col in enhanced_features.columns if any(x in col for x in ['price_impact', 'order_flow', 'vwap'])]),
                    'Cross-Symbol': len([col for col in enhanced_features.columns if any(x in col for x in ['relative_strength', 'correlation'])]),
                    'ML Features': len([col for col in enhanced_features.columns if any(x in col for x in ['pca_', 'interact_'])]),
                    'Advanced Signals': len([col for col in enhanced_features.columns if 'dipmaster_v4' in col]),
                    'Enhanced Labels': len([col for col in enhanced_features.columns if any(x in col for x in ['target', 'future_return'])])
                }
                
                self.logger.info("\nFeature Categories:")
                for category, count in feature_categories.items():
                    self.logger.info(f"  {category}: {count} features")
                
                return True
            else:
                self.logger.error("Failed to save enhanced features")
                return False
                
        except Exception as e:
            self.logger.error(f"Enhanced feature engineering pipeline failed: {e}")
            return False

def main():
    """Main execution function"""
    try:
        # Ensure directories exist
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Run enhanced feature engineering
        runner = EnhancedFeatureRunner()
        success = runner.run_enhanced_feature_engineering()
        
        if success:
            print("\n🎯 Enhanced Feature Engineering Completed Successfully!")
            print("📊 Ready for DipMaster V4 Strategy Optimization")
            print("🚀 Target: 85%+ Win Rate with Enhanced Features")
        else:
            print("\n❌ Enhanced Feature Engineering Failed")
            print("📋 Please check logs for details")
        
        return success
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)