"""
DipMaster Enhanced V4 - Comprehensive ML Training Pipeline
Implements rigorous time-series model training with proper validation methodology
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime, timedelta
import joblib
import pickle

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Statistical and optimization
import optuna
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Feature importance
import shap

class PurgedTimeSeriesSplit:
    """
    Time series cross-validation with purging and embargo
    Prevents lookahead bias by excluding observations around validation set
    """
    
    def __init__(self, n_splits: int = 5, embargo_hours: int = 2, purge_hours: int = 1):
        self.n_splits = n_splits
        self.embargo_periods = embargo_hours * 12  # Convert to 5-minute periods
        self.purge_periods = purge_hours * 12
        
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
        """Generate purged time-series splits"""
        indices = np.arange(len(X))
        n_samples = len(indices)
        
        # Calculate split sizes
        test_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Define test period
            test_start = i * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
                
            # Define train period (everything before test, minus embargo)
            train_end = test_start - self.embargo_periods
            
            if train_end <= 0:
                continue
                
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]
            
            # Apply purging around test set
            purge_start = test_start - self.purge_periods
            purge_end = test_end + self.purge_periods
            
            # Remove purged indices from training set
            train_indices = train_indices[train_indices < purge_start]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


class DipMasterMLPipeline:
    """
    Comprehensive ML training pipeline for DipMaster Enhanced V4
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.models = {}
        self.feature_importance = {}
        self.validation_results = {}
        self.scalers = {}
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration settings"""
        default_config = {
            'target_variable': 'target_binary',
            'feature_selection': {
                'top_k_features': 30,
                'importance_threshold': 0.01
            },
            'model_params': {
                'lgb': {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 20,
                    'random_state': 42,
                    'verbosity': -1
                },
                'xgb': {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'random_state': 42,
                    'verbosity': 0
                }
            },
            'validation': {
                'cv_folds': 5,
                'embargo_hours': 2,
                'purge_hours': 1,
                'test_size': 0.3
            },
            'optimization': {
                'n_trials': 100,
                'direction': 'maximize',
                'metric': 'f1_score'
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
                default_config.update(loaded_config)
                
        return default_config
    
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare training data"""
        print("Loading enhanced features dataset...")
        df = pd.read_parquet(data_path)
        
        # Handle missing values
        print(f"Handling missing values: {df.isnull().sum().sum()} total missing")
        df = df.fillna(method='ffill').fillna(0)
        
        # Create proper datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            # Assuming 5-minute intervals starting from a base date
            base_date = datetime(2024, 5, 1)  # Adjust based on actual data
            df.index = pd.date_range(
                start=base_date, 
                periods=len(df), 
                freq='5T'
            )
        
        # Extract features and targets
        target_col = self.config['target_variable']
        feature_cols = [col for col in df.columns if col not in [
            'symbol', target_col, 'target_return', 'target_risk_adjusted'
        ] and not col.startswith('hits_target') and not col.startswith('future_return')]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        print(f"Dataset shape: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"Features selected: {len(feature_cols)}")
        
        return X, y
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select top features based on importance"""
        print("Performing feature selection...")
        
        # Quick LightGBM for feature importance
        lgb_model = lgb.LGBMClassifier(**self.config['model_params']['lgb'])
        lgb_model.fit(X, y)
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Select top features
        top_k = self.config['feature_selection']['top_k_features']
        threshold = self.config['feature_selection']['importance_threshold']
        
        selected_features = importance_df[
            (importance_df['importance'] >= threshold) & 
            (importance_df.index < top_k)
        ]['feature'].tolist()
        
        print(f"Selected {len(selected_features)} features")
        print("Top 10 features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i+1}. {row['feature']}: {row['importance']:.4f}")
            
        self.feature_importance['initial_selection'] = importance_df
        return selected_features
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, 
                                model_type: str) -> Dict:
        """Optimize hyperparameters using Optuna"""
        print(f"Optimizing {model_type} hyperparameters...")
        
        def objective(trial):
            if model_type == 'lgb':
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                    'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                    'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0, 10),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 10),
                    'random_state': 42,
                    'verbosity': -1
                }
                model = lgb.LGBMClassifier(**params)
                
            elif model_type == 'xgb':
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                    'random_state': 42,
                    'verbosity': 0
                }
                model = xgb.XGBClassifier(**params)
            
            # Time series cross-validation
            cv_scores = []
            tscv = PurgedTimeSeriesSplit(
                n_splits=self.config['validation']['cv_folds'],
                embargo_hours=self.config['validation']['embargo_hours']
            )
            
            for train_idx, val_idx in tscv.split(X):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                
                # Use F1 score as optimization metric
                score = f1_score(y_val_fold, y_pred)
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        # Run optimization
        study = optuna.create_study(
            direction=self.config['optimization']['direction'],
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=self.config['optimization']['n_trials'])
        
        print(f"Best {model_type} score: {study.best_value:.4f}")
        print(f"Best {model_type} params: {study.best_params}")
        
        return study.best_params
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train ensemble of optimized models"""
        print("Training optimized models...")
        
        # Split data temporally
        test_size = self.config['validation']['test_size']
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Feature scaling
        self.scalers['robust'] = RobustScaler()
        X_train_scaled = pd.DataFrame(
            self.scalers['robust'].fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scalers['robust'].transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        models = {}
        
        # 1. Optimize and train LightGBM
        lgb_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'lgb')
        lgb_params.update(self.config['model_params']['lgb'])
        models['lgb'] = lgb.LGBMClassifier(**lgb_params)
        models['lgb'].fit(X_train_scaled, y_train)
        
        # 2. Optimize and train XGBoost
        xgb_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'xgb')
        xgb_params.update(self.config['model_params']['xgb'])
        models['xgb'] = xgb.XGBClassifier(**xgb_params)
        models['xgb'].fit(X_train_scaled, y_train)
        
        # 3. Create ensemble
        models['ensemble'] = VotingClassifier(
            estimators=[
                ('lgb', models['lgb']),
                ('xgb', models['xgb'])
            ],
            voting='soft'
        )
        models['ensemble'].fit(X_train_scaled, y_train)
        
        # Store models and data splits
        self.models = models
        self.train_data = (X_train_scaled, y_train)
        self.test_data = (X_test_scaled, y_test)
        
        return models
    
    def evaluate_models(self) -> Dict:
        """Comprehensive model evaluation"""
        print("Evaluating models...")
        
        X_train, y_train = self.train_data
        X_test, y_test = self.test_data
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_train_proba = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else None
            y_test_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Metrics
            train_metrics = self._calculate_metrics(y_train, y_train_pred, y_train_proba)
            test_metrics = self._calculate_metrics(y_test, y_test_pred, y_test_proba)
            
            results[name] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'predictions': {
                    'train': y_train_pred,
                    'test': y_test_pred,
                    'train_proba': y_train_proba,
                    'test_proba': y_test_proba
                }
            }
            
            # Print key metrics
            print(f"  Train - Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            print(f"  Test  - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        self.validation_results = results
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_proba=None) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1': f1_score(y_true, y_pred, average='weighted'),
            'win_rate': accuracy_score(y_true, y_pred)  # Same as accuracy for binary
        }
        
        if y_proba is not None:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        
        return metrics
    
    def analyze_feature_importance(self) -> Dict:
        """Analyze feature importance using SHAP"""
        print("Analyzing feature importance with SHAP...")
        
        X_test, y_test = self.test_data
        importance_analysis = {}
        
        for name, model in self.models.items():
            if name == 'ensemble':
                continue  # Skip ensemble for SHAP analysis
                
            print(f"Analyzing {name}...")
            
            # SHAP analysis
            if name == 'lgb':
                explainer = shap.TreeExplainer(model)
            elif name == 'xgb':
                explainer = shap.TreeExplainer(model)
            else:
                continue
                
            # Calculate SHAP values on a sample (too expensive for full dataset)
            sample_size = min(1000, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification
            
            # Feature importance summary
            feature_importance = pd.DataFrame({
                'feature': X_sample.columns,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)
            
            importance_analysis[name] = {
                'feature_importance': feature_importance,
                'shap_values': shap_values,
                'sample_data': X_sample
            }
        
        self.feature_importance.update(importance_analysis)
        return importance_analysis
    
    def save_models(self, save_dir: str) -> None:
        """Save trained models and results"""
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving models to {save_dir}...")
        
        # Save models
        for name, model in self.models.items():
            model_path = os.path.join(save_dir, f'{name}_model.pkl')
            joblib.dump(model, model_path)
            print(f"  Saved {name} model")
        
        # Save scalers
        scaler_path = os.path.join(save_dir, 'scalers.pkl')
        joblib.dump(self.scalers, scaler_path)
        
        # Save results
        results_path = os.path.join(save_dir, 'validation_results.json')
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for name, result in self.validation_results.items():
            json_results[name] = {
                'train_metrics': result['train_metrics'],
                'test_metrics': result['test_metrics']
            }
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save feature importance
        importance_path = os.path.join(save_dir, 'feature_importance.pkl')
        joblib.dump(self.feature_importance, importance_path)
        
        print("All models and results saved successfully!")
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("=" * 80)
        report.append("DIPMASTER ENHANCED V4 - MODEL PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model Performance Summary
        report.append("MODEL PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        for name, result in self.validation_results.items():
            test_metrics = result['test_metrics']
            report.append(f"\n{name.upper()} MODEL:")
            report.append(f"  Win Rate (Accuracy): {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
            report.append(f"  Precision: {test_metrics['precision']:.4f}")
            report.append(f"  Recall: {test_metrics['recall']:.4f}")
            report.append(f"  F1 Score: {test_metrics['f1']:.4f}")
            if 'auc' in test_metrics:
                report.append(f"  AUC: {test_metrics['auc']:.4f}")
        
        # Best Model
        best_model = max(self.validation_results.items(), 
                        key=lambda x: x[1]['test_metrics']['accuracy'])
        best_name, best_result = best_model
        best_accuracy = best_result['test_metrics']['accuracy']
        
        report.append(f"\nBEST MODEL: {best_name.upper()}")
        report.append(f"Win Rate: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        # Target Achievement
        target_win_rate = 0.85
        report.append(f"\nTARGET ACHIEVEMENT:")
        report.append(f"  Target Win Rate: {target_win_rate*100:.1f}%")
        report.append(f"  Current Best: {best_accuracy*100:.2f}%")
        report.append(f"  Gap: {(target_win_rate - best_accuracy)*100:.2f}%")
        
        if best_accuracy >= target_win_rate:
            report.append("  ✅ TARGET ACHIEVED!")
        else:
            report.append("  ❌ Target not yet achieved")
        
        # Feature Importance Top 10
        if self.feature_importance:
            report.append(f"\nTOP 10 FEATURES ({best_name.upper()}):")
            report.append("-" * 40)
            if best_name in self.feature_importance:
                top_features = self.feature_importance[best_name]['feature_importance'].head(10)
                for i, (_, row) in enumerate(top_features.iterrows()):
                    report.append(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.6f}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)


def main():
    """Main training pipeline execution"""
    print("DipMaster Enhanced V4 - ML Training Pipeline")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = DipMasterMLPipeline()
    
    # Load data
    data_path = "data/Enhanced_Features_25symbols_20250816_223904.parquet"
    X, y = pipeline.load_data(data_path)
    
    # Feature selection
    selected_features = pipeline.feature_selection(X, y)
    X_selected = X[selected_features]
    
    # Train models
    models = pipeline.train_models(X_selected, y)
    
    # Evaluate models
    results = pipeline.evaluate_models()
    
    # Analyze feature importance
    importance = pipeline.analyze_feature_importance()
    
    # Save everything
    save_dir = "results/model_training"
    pipeline.save_models(save_dir)
    
    # Generate report
    report = pipeline.generate_performance_report()
    print("\n" + report)
    
    # Save report
    report_path = f"{save_dir}/performance_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\nTraining completed! Results saved to: {save_dir}")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()