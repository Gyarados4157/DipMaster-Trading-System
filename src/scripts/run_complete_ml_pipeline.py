"""
DipMaster Enhanced V4 - Complete ML Training and Backtesting Pipeline
Comprehensive script to train models, validate performance, and generate production-ready systems
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import json
import joblib
from typing import Dict, List, Tuple

# Import ML libraries directly
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna
import matplotlib.pyplot as plt
import seaborn as sns

class CompleteMLPipeline:
    """
    Complete ML pipeline combining training, validation, and backtesting
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.ml_pipeline = None
        self.backtester = None
        self.results = {}
        
    def _load_config(self, config_path: str) -> Dict:
        """Load complete pipeline configuration"""
        default_config = {
            'data': {
                'features_path': 'data/Enhanced_Features_25symbols_20250816_223904.parquet',
                'metadata_path': 'data/Enhanced_FeatureSet_V4_20250816_223904.json'
            },
            'ml_training': {
                'target_variable': 'target_binary',
                'validation_split': 0.3,
                'cv_folds': 5,
                'optimization_trials': 50,  # Reduced for faster execution
                'top_features': 25
            },
            'backtesting': {
                'initial_capital': 10000,
                'position_size': 1000,
                'max_positions': 3,
                'transaction_costs': 0.001,
                'slippage': 0.0005
            },
            'output': {
                'results_dir': 'results/complete_pipeline',
                'save_models': True,
                'generate_reports': True,
                'create_plots': True
            },
            'targets': {
                'win_rate': 0.85,
                'sharpe_ratio': 2.0,
                'max_drawdown': 0.03,
                'profit_factor': 1.8
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                # Deep merge configurations
                for key, value in loaded_config.items():
                    if isinstance(value, dict) and key in default_config:
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
        
        return default_config
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Load and prepare all data for training and backtesting"""
        print("Loading and preparing enhanced features dataset...")
        
        # Load main dataset
        features_path = self.config['data']['features_path']
        df = pd.read_parquet(features_path)
        
        print(f"Loaded dataset: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Handle missing values
        print(f"Missing values before cleaning: {df.isnull().sum().sum()}")
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
        
        # Create proper datetime index for backtesting
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Creating datetime index...")
            # Assuming 5-minute intervals starting from May 1, 2024
            base_date = datetime(2024, 5, 1)
            df.index = pd.date_range(
                start=base_date,
                periods=len(df),
                freq='5T'
            )
        
        # Separate features, targets, and price data
        target_col = self.config['ml_training']['target_variable']
        
        # Identify feature columns (exclude targets and metadata)
        exclude_cols = {
            'symbol', target_col, 'target_return', 'target_risk_adjusted',
            'open', 'high', 'low', 'close', 'volume'
        }
        
        # Add all target columns to exclusion
        target_cols = [col for col in df.columns if col.startswith('hits_target') or col.startswith('future_return')]
        exclude_cols.update(target_cols)
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        print(f"Feature columns identified: {len(feature_cols)}")
        
        # Create feature matrix and target vector
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Create price data for backtesting
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        available_price_cols = [col for col in price_cols if col in df.columns]
        
        if available_price_cols:
            price_data = df[['symbol'] + available_price_cols].copy()
        else:
            # Create synthetic price data if not available
            print("Warning: No price data found, creating synthetic data")
            price_data = pd.DataFrame({
                'symbol': df.get('symbol', 'DEFAULT'),
                'open': 100 * (1 + np.random.normal(0, 0.001, len(df))),
                'high': 100 * (1 + np.random.normal(0.001, 0.001, len(df))),
                'low': 100 * (1 + np.random.normal(-0.001, 0.001, len(df))),
                'close': 100 * (1 + np.random.normal(0, 0.001, len(df))),
                'volume': np.random.exponential(1000, len(df))
            }, index=df.index)
        
        print(f"Feature matrix: {X.shape}")
        print(f"Target variable: {y.name}, distribution: {y.value_counts().to_dict()}")
        print(f"Price data: {price_data.shape}")
        
        return X, y, price_data
    
    def run_ml_training(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Run comprehensive ML training pipeline"""
        print("\n" + "="*60)
        print("MACHINE LEARNING TRAINING PHASE")
        print("="*60)
        
        # Initialize ML pipeline
        ml_config = {
            'target_variable': self.config['ml_training']['target_variable'],
            'feature_selection': {
                'top_k_features': self.config['ml_training']['top_features'],
                'importance_threshold': 0.001
            },
            'validation': {
                'cv_folds': self.config['ml_training']['cv_folds'],
                'test_size': self.config['ml_training']['validation_split']
            },
            'optimization': {
                'n_trials': self.config['ml_training']['optimization_trials'],
                'direction': 'maximize'
            }
        }
        
        self.ml_pipeline = DipMasterMLPipeline()
        self.ml_pipeline.config.update(ml_config)
        
        # Feature selection
        print("\nPhase 1: Feature Selection")
        selected_features = self.ml_pipeline.feature_selection(X, y)
        X_selected = X[selected_features]
        
        print(f"Selected {len(selected_features)} features for training")
        print("Top 10 selected features:")
        for i, feat in enumerate(selected_features[:10]):
            print(f"  {i+1:2d}. {feat}")
        
        # Model training
        print("\nPhase 2: Model Training and Optimization")
        models = self.ml_pipeline.train_models(X_selected, y)
        
        print(f"Trained {len(models)} models successfully")
        
        # Model evaluation
        print("\nPhase 3: Model Evaluation")
        evaluation_results = self.ml_pipeline.evaluate_models()
        
        # Feature importance analysis
        print("\nPhase 4: Feature Importance Analysis")
        importance_analysis = self.ml_pipeline.analyze_feature_importance()
        
        # Save models and results
        results_dir = self.config['output']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        if self.config['output']['save_models']:
            print(f"\nSaving models to {results_dir}")
            self.ml_pipeline.save_models(results_dir)
        
        # Generate ML performance report
        ml_report = self.ml_pipeline.generate_performance_report()
        print("\n" + ml_report)
        
        # Save report
        report_path = os.path.join(results_dir, 'ml_training_report.txt')
        with open(report_path, 'w') as f:
            f.write(ml_report)
        
        return {
            'models': models,
            'selected_features': selected_features,
            'evaluation_results': evaluation_results,
            'importance_analysis': importance_analysis,
            'performance_report': ml_report
        }
    
    def generate_model_signals(self, X: pd.DataFrame, price_data: pd.DataFrame) -> pd.Series:
        """Generate trading signals using trained models"""
        print("\n" + "="*60)
        print("SIGNAL GENERATION PHASE")
        print("="*60)
        
        if not self.ml_pipeline or not self.ml_pipeline.models:
            raise ValueError("Models must be trained before generating signals")
        
        # Use the best performing model (ensemble typically)
        best_model_name = 'ensemble'
        if best_model_name not in self.ml_pipeline.models:
            best_model_name = list(self.ml_pipeline.models.keys())[0]
        
        best_model = self.ml_pipeline.models[best_model_name]
        print(f"Using {best_model_name} model for signal generation")
        
        # Get selected features
        selected_features = self.results['ml_training']['selected_features']
        X_selected = X[selected_features]
        
        # Scale features using saved scaler
        scaler = self.ml_pipeline.scalers['robust']
        X_scaled = pd.DataFrame(
            scaler.transform(X_selected),
            columns=X_selected.columns,
            index=X_selected.index
        )
        
        # Generate predictions
        print("Generating probability predictions...")
        signal_probabilities = best_model.predict_proba(X_scaled)[:, 1]
        
        # Convert to signal strengths (0-1 scale)
        signals = pd.Series(signal_probabilities, index=X_scaled.index)
        
        print(f"Generated {len(signals)} signals")
        print(f"Signal distribution:")
        print(f"  Mean: {signals.mean():.4f}")
        print(f"  Std:  {signals.std():.4f}")
        print(f"  Min:  {signals.min():.4f}")
        print(f"  Max:  {signals.max():.4f}")
        
        # Analyze signal quality
        signal_stats = {}
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            high_signals = signals[signals >= threshold]
            signal_stats[threshold] = {
                'count': len(high_signals),
                'percentage': len(high_signals) / len(signals) * 100
            }
            print(f"  Signals >= {threshold}: {len(high_signals)} ({len(high_signals)/len(signals)*100:.1f}%)")
        
        return signals
    
    def run_comprehensive_backtest(self, signals: pd.Series, price_data: pd.DataFrame) -> Dict:
        """Run comprehensive backtesting with the generated signals"""
        print("\n" + "="*60)
        print("COMPREHENSIVE BACKTESTING PHASE")
        print("="*60)
        
        # Initialize backtester
        backtest_config = {
            'initial_capital': self.config['backtesting']['initial_capital'],
            'position_sizing': {
                'method': 'fixed_amount',
                'amount': self.config['backtesting']['position_size'],
                'max_positions': self.config['backtesting']['max_positions']
            },
            'transaction_costs': {
                'commission_rate': self.config['backtesting']['transaction_costs'],
                'slippage_model': 'adaptive',
                'fixed_slippage': self.config['backtesting']['slippage']
            },
            'signal_filtering': {
                'min_signal_strength': 0.5  # Only trade high-confidence signals
            }
        }
        
        self.backtester = EnhancedBacktester(backtest_config)
        
        # Run backtest
        print("Running backtest simulation...")
        backtest_results = self.backtester.run_backtest(price_data, signals)
        
        # Analyze performance by regime
        print("Analyzing regime-specific performance...")
        regime_analysis = self.backtester.analyze_performance_by_regime(price_data)
        
        # Generate comprehensive report
        backtest_report = self.backtester.generate_detailed_report()
        print("\n" + backtest_report)
        
        # Save backtest results
        results_dir = self.config['output']['results_dir']
        
        # Save detailed backtest report
        backtest_report_path = os.path.join(results_dir, 'backtest_report.txt')
        with open(backtest_report_path, 'w') as f:
            f.write(backtest_report)
        
        # Save trade records
        if self.backtester.trades:
            trades_df = pd.DataFrame([
                {
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'symbol': t.symbol,
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'position_size': t.position_size,
                    'signal_strength': t.signal_strength,
                    'profit_loss': t.profit_loss,
                    'profit_loss_pct': t.profit_loss_pct,
                    'holding_minutes': t.holding_minutes,
                    'total_costs': t.total_costs,
                    'net_profit_loss': t.net_profit_loss,
                    'win': t.win
                }
                for t in self.backtester.trades
            ])
            
            trades_path = os.path.join(results_dir, 'trade_records.csv')
            trades_df.to_csv(trades_path, index=False)
            print(f"Trade records saved to: {trades_path}")
        
        # Generate performance plots
        if self.config['output']['create_plots']:
            plots_path = os.path.join(results_dir, 'performance_plots.png')
            self.backtester.plot_results(plots_path)
        
        return {
            'backtest_results': backtest_results,
            'regime_analysis': regime_analysis,
            'performance_metrics': self.backtester.performance_metrics,
            'backtest_report': backtest_report
        }
    
    def analyze_target_achievement(self) -> Dict:
        """Analyze how close we are to achieving the 85%+ win rate target"""
        print("\n" + "="*60)
        print("TARGET ACHIEVEMENT ANALYSIS")
        print("="*60)
        
        if not self.backtester or not self.backtester.performance_metrics:
            print("No backtest results available for analysis")
            return {}
        
        metrics = self.backtester.performance_metrics
        targets = self.config['targets']
        
        analysis = {
            'current_performance': {
                'win_rate': metrics.win_rate,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown_pct,
                'profit_factor': metrics.profit_factor
            },
            'target_gaps': {
                'win_rate_gap': targets['win_rate'] - metrics.win_rate,
                'sharpe_gap': targets['sharpe_ratio'] - metrics.sharpe_ratio,
                'drawdown_excess': metrics.max_drawdown_pct - targets['max_drawdown'],
                'profit_factor_gap': targets['profit_factor'] - metrics.profit_factor
            },
            'achievement_status': {
                'win_rate_achieved': metrics.win_rate >= targets['win_rate'],
                'sharpe_achieved': metrics.sharpe_ratio >= targets['sharpe_ratio'],
                'drawdown_achieved': metrics.max_drawdown_pct <= targets['max_drawdown'],
                'profit_factor_achieved': metrics.profit_factor >= targets['profit_factor']
            }
        }
        
        # Calculate overall score
        achievements = list(analysis['achievement_status'].values())
        overall_score = sum(achievements) / len(achievements)
        analysis['overall_achievement'] = overall_score
        
        print(f"Overall Target Achievement: {overall_score*100:.1f}%")
        print(f"Targets Met: {sum(achievements)}/{len(achievements)}")
        print()
        
        # Detailed gap analysis
        print("DETAILED GAP ANALYSIS:")
        print(f"Win Rate: {metrics.win_rate*100:.2f}% (Target: {targets['win_rate']*100:.0f}%) - Gap: {analysis['target_gaps']['win_rate_gap']*100:+.2f}%")
        print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f} (Target: {targets['sharpe_ratio']:.1f}) - Gap: {analysis['target_gaps']['sharpe_gap']:+.3f}")
        print(f"Max Drawdown: {metrics.max_drawdown_pct*100:.2f}% (Target: <{targets['max_drawdown']*100:.0f}%) - Excess: {analysis['target_gaps']['drawdown_excess']*100:+.2f}%")
        print(f"Profit Factor: {metrics.profit_factor:.3f} (Target: {targets['profit_factor']:.1f}) - Gap: {analysis['target_gaps']['profit_factor_gap']:+.3f}")
        
        # Recommendations
        recommendations = []
        
        if not analysis['achievement_status']['win_rate_achieved']:
            gap_pct = analysis['target_gaps']['win_rate_gap'] * 100
            recommendations.append(f"Win Rate Improvement: Need {gap_pct:.1f}% improvement")
            recommendations.append("- Increase signal confidence threshold")
            recommendations.append("- Implement regime filtering")
            recommendations.append("- Add multi-timeframe confirmation")
        
        if not analysis['achievement_status']['sharpe_achieved']:
            recommendations.append("Sharpe Ratio Improvement:")
            recommendations.append("- Optimize position sizing")
            recommendations.append("- Improve risk-adjusted returns")
            recommendations.append("- Reduce return volatility")
        
        if not analysis['achievement_status']['drawdown_achieved']:
            recommendations.append("Drawdown Reduction:")
            recommendations.append("- Implement stricter stop losses")
            recommendations.append("- Reduce position correlations")
            recommendations.append("- Add volatility-based position sizing")
        
        if not analysis['achievement_status']['profit_factor_achieved']:
            recommendations.append("Profit Factor Enhancement:")
            recommendations.append("- Optimize exit timing")
            recommendations.append("- Improve winning trade size")
            recommendations.append("- Reduce losing trade impact")
        
        analysis['recommendations'] = recommendations
        
        if recommendations:
            print("\nRECOMMENDATIONS FOR IMPROVEMENT:")
            for rec in recommendations:
                if rec.startswith('-'):
                    print(f"  {rec}")
                else:
                    print(f"\n{rec}")
        
        return analysis
    
    def run_complete_pipeline(self) -> Dict:
        """Run the complete ML training and backtesting pipeline"""
        start_time = datetime.now()
        
        print("DipMaster Enhanced V4 - Complete ML Pipeline")
        print("=" * 80)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Target win rate: {self.config['targets']['win_rate']*100:.0f}%")
        print(f"Results directory: {self.config['output']['results_dir']}")
        print()
        
        try:
            # Phase 1: Data Preparation
            X, y, price_data = self.prepare_data()
            
            # Phase 2: ML Training
            ml_results = self.run_ml_training(X, y)
            self.results['ml_training'] = ml_results
            
            # Phase 3: Signal Generation
            signals = self.generate_model_signals(X, price_data)
            self.results['signals'] = signals
            
            # Phase 4: Comprehensive Backtesting
            backtest_results = self.run_comprehensive_backtest(signals, price_data)
            self.results['backtesting'] = backtest_results
            
            # Phase 5: Target Achievement Analysis
            target_analysis = self.analyze_target_achievement()
            self.results['target_analysis'] = target_analysis
            
            # Generate final summary
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            print("\n" + "="*80)
            print("PIPELINE EXECUTION COMPLETED")
            print("="*80)
            print(f"Execution time: {execution_time}")
            print(f"Total trades: {len(self.backtester.trades) if self.backtester else 0}")
            
            if self.backtester and self.backtester.performance_metrics:
                metrics = self.backtester.performance_metrics
                print(f"Final win rate: {metrics.win_rate*100:.2f}%")
                print(f"Target achievement: {target_analysis.get('overall_achievement', 0)*100:.1f}%")
            
            # Save complete results
            self.save_complete_results()
            
            return self.results
            
        except Exception as e:
            print(f"Pipeline execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def save_complete_results(self):
        """Save all results to disk"""
        results_dir = self.config['output']['results_dir']
        os.makedirs(results_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(results_dir, 'pipeline_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        
        # Save target analysis
        if 'target_analysis' in self.results:
            analysis_path = os.path.join(results_dir, 'target_analysis.json')
            with open(analysis_path, 'w') as f:
                json.dump(self.results['target_analysis'], f, indent=2, default=str)
        
        # Save signals
        if 'signals' in self.results:
            signals_path = os.path.join(results_dir, 'generated_signals.csv')
            self.results['signals'].to_csv(signals_path)
        
        print(f"All results saved to: {results_dir}")


def main():
    """Main execution function"""
    # Initialize and run complete pipeline
    pipeline = CompleteMLPipeline()
    
    # Run the complete pipeline
    results = pipeline.run_complete_pipeline()
    
    # Print final summary
    if 'error' not in results:
        print("\n" + "="*80)
        print("SUCCESS: Complete pipeline executed successfully!")
        print("="*80)
        
        if 'target_analysis' in results:
            analysis = results['target_analysis']
            achievement_pct = analysis.get('overall_achievement', 0) * 100
            print(f"Target Achievement: {achievement_pct:.1f}%")
            
            if analysis.get('achievement_status', {}).get('win_rate_achieved', False):
                print("🎉 85%+ WIN RATE TARGET ACHIEVED!")
            else:
                current_wr = analysis.get('current_performance', {}).get('win_rate', 0) * 100
                print(f"Current win rate: {current_wr:.2f}% - Target: 85.0%")
                print("See recommendations in the detailed reports.")
    else:
        print("Pipeline execution failed. Check error messages above.")
    
    return results


if __name__ == "__main__":
    results = main()