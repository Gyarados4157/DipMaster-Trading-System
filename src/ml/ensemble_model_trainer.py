"""
Ensemble Model Training Pipeline with Rigorous Hyperparameter Optimization
Implements comprehensive training framework for multiple ML models.
"""

import numpy as np
import pandas as pd
import json
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import logging
import joblib

# Model imports
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score, classification_report)

# Hyperparameter optimization
import optuna
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback

# Custom modules
from time_series_validator import PurgedWalkForwardCV, RobustTimeSeriesValidator

warnings.filterwarnings('ignore')

class AdvancedFeatureProcessor:
    """
    Advanced feature processing with selection and engineering
    """
    
    def __init__(self, 
                 scaling_method: str = 'robust',
                 feature_selection: bool = True,
                 max_features: int = 50,
                 correlation_threshold: float = 0.95):
        """
        Initialize feature processor
        
        Args:
            scaling_method: 'standard', 'robust', or 'none'
            feature_selection: Whether to perform feature selection
            max_features: Maximum number of features to select
            correlation_threshold: Remove features with correlation > threshold
        """
        self.scaling_method = scaling_method
        self.feature_selection = feature_selection
        self.max_features = max_features
        self.correlation_threshold = correlation_threshold
        
        # Initialize processors
        self.scaler = None
        self.feature_selector = None
        self.selected_features_ = None
        self.correlation_matrix_ = None
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series = None) -> np.ndarray:
        """
        Fit processors and transform features
        
        Args:
            X: Input features
            y: Target variable
            
        Returns:
            Transformed feature matrix
        """
        X_processed = X.copy()
        
        # 1. Remove features with high correlation
        X_processed = self._remove_correlated_features(X_processed)
        
        # 2. Feature selection
        if self.feature_selection and y is not None:
            X_processed = self._select_features(X_processed, y)
        
        # 3. Scaling
        X_processed = self._scale_features(X_processed, fit=True)
        
        return X_processed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform features using fitted processors
        
        Args:
            X: Input features
            
        Returns:
            Transformed feature matrix
        """
        X_processed = X.copy()
        
        # Apply same transformations in same order
        if self.correlation_matrix_ is not None:
            # Keep only features that were not removed due to correlation
            remaining_features = [col for col in X_processed.columns 
                                if col in self.selected_features_]
            X_processed = X_processed[remaining_features]
        
        if self.feature_selector is not None:
            X_processed = pd.DataFrame(
                self.feature_selector.transform(X_processed),
                index=X_processed.index,
                columns=self.feature_selector.get_feature_names_out()
            )
        
        X_processed = self._scale_features(X_processed, fit=False)
        
        return X_processed
    
    def _remove_correlated_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with high correlation"""
        correlation_matrix = X.corr().abs()
        self.correlation_matrix_ = correlation_matrix
        
        # Find features to remove
        upper_triangle = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        high_corr_pairs = np.where((correlation_matrix > self.correlation_threshold) & ~upper_triangle)
        
        features_to_remove = set()
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            # Remove feature with lower variance
            if X.iloc[:, i].var() < X.iloc[:, j].var():
                features_to_remove.add(X.columns[i])
            else:
                features_to_remove.add(X.columns[j])
        
        # Keep features not marked for removal
        features_to_keep = [col for col in X.columns if col not in features_to_remove]
        self.selected_features_ = features_to_keep
        
        return X[features_to_keep]
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Perform statistical feature selection"""
        n_features_to_select = min(self.max_features, len(X.columns))
        
        self.feature_selector = SelectKBest(
            score_func=f_classif,
            k=n_features_to_select
        )
        
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_feature_names = self.feature_selector.get_feature_names_out()
        
        return pd.DataFrame(
            X_selected,
            index=X.index,
            columns=selected_feature_names
        )
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Scale features using specified method"""
        if self.scaling_method == 'none':
            return X.values
            
        if fit:
            if self.scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif self.scaling_method == 'robust':
                self.scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {self.scaling_method}")
            
            X_scaled = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Call fit_transform first.")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled


class ModelTrainer:
    """
    Individual model trainer with hyperparameter optimization
    """
    
    def __init__(self, model_name: str, cv_strategy: PurgedWalkForwardCV,
                 n_trials: int = 100, timeout: int = 600):
        """
        Initialize model trainer
        
        Args:
            model_name: Name of model to train
            cv_strategy: Cross-validation strategy
            n_trials: Number of Optuna trials
            timeout: Timeout in seconds for optimization
        """
        self.model_name = model_name
        self.cv_strategy = cv_strategy
        self.n_trials = n_trials
        self.timeout = timeout
        
        # Results storage
        self.best_params_ = None
        self.best_score_ = None
        self.best_model_ = None
        self.cv_results_ = {}
        self.feature_importance_ = None
        
    def _create_model(self, params: dict = None):
        """Create model instance with parameters"""
        if params is None:
            params = {}
            
        if self.model_name == 'lightgbm':
            return lgb.LGBMClassifier(
                **params,
                random_state=42,
                n_jobs=1,  # Avoid conflicts with Optuna
                verbose=-1
            )
        elif self.model_name == 'xgboost':
            return xgb.XGBClassifier(
                **params,
                random_state=42,
                n_jobs=1,
                verbosity=0
            )
        elif self.model_name == 'random_forest':
            return RandomForestClassifier(
                **params,
                random_state=42,
                n_jobs=1
            )
        elif self.model_name == 'logistic_regression':
            return LogisticRegression(
                **params,
                random_state=42,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
    
    def _get_param_space(self, trial: optuna.Trial) -> dict:
        """Define hyperparameter search space"""
        
        if self.model_name == 'lightgbm':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            }
            
        elif self.model_name == 'xgboost':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            }
            
        elif self.model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            }
            
        elif self.model_name == 'logistic_regression':
            return {
                'C': trial.suggest_float('C', 0.001, 100.0, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0) if trial.params.get('penalty') == 'elasticnet' else None
            }
    
    def _objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series) -> float:
        """Objective function for hyperparameter optimization"""
        
        params = self._get_param_space(trial)
        
        # Create model with trial parameters
        model = self._create_model(params)
        
        # Cross-validation scores
        cv_scores = []
        
        for train_idx, val_idx in self.cv_strategy.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_val)[:, 1]
            else:
                y_pred = model.predict(X_val)
            
            # Calculate AUC score
            score = roc_auc_score(y_val, y_pred)
            cv_scores.append(score)
            
            # Pruning for tree-based models
            if len(cv_scores) >= 2:  # Prune after 2 folds
                intermediate_score = np.mean(cv_scores)
                trial.report(intermediate_score, len(cv_scores) - 1)
                
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
        
        return np.mean(cv_scores)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train model with hyperparameter optimization
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Training results dictionary
        """
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        
        # Optimize hyperparameters
        study.optimize(
            lambda trial: self._objective(trial, X, y),
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=1
        )
        
        # Store best parameters and score
        self.best_params_ = study.best_params
        self.best_score_ = study.best_value
        
        # Train final model on full training set
        self.best_model_ = self._create_model(self.best_params_)
        self.best_model_.fit(X, y)
        
        # Calculate feature importance
        if hasattr(self.best_model_, 'feature_importances_'):
            self.feature_importance_ = dict(zip(
                X.columns, self.best_model_.feature_importances_
            ))
        
        # Store CV results
        self.cv_results_ = {
            'best_score': self.best_score_,
            'best_params': self.best_params_,
            'n_trials': len(study.trials),
            'optimization_time': sum(t.duration.total_seconds() for t in study.trials if t.duration)
        }
        
        return self.cv_results_


class EnsembleModelTrainer:
    """
    Complete ensemble training pipeline
    """
    
    def __init__(self, 
                 models: List[str] = None,
                 cv_splits: int = 5,
                 embargo_hours: int = 24,
                 optimization_trials: int = 100,
                 feature_processing: bool = True):
        """
        Initialize ensemble trainer
        
        Args:
            models: List of model names to train
            cv_splits: Number of CV splits
            embargo_hours: Embargo period for CV
            optimization_trials: Number of hyperparameter trials per model
            feature_processing: Whether to apply feature processing
        """
        
        if models is None:
            models = ['lightgbm', 'xgboost', 'random_forest', 'logistic_regression']
        
        self.models = models
        self.cv_splits = cv_splits
        self.embargo_hours = embargo_hours
        self.optimization_trials = optimization_trials
        self.feature_processing = feature_processing
        
        # Initialize components
        self.cv_strategy = PurgedWalkForwardCV(
            n_splits=cv_splits,
            embargo_hours=embargo_hours
        )
        
        if feature_processing:
            self.feature_processor = AdvancedFeatureProcessor()
        else:
            self.feature_processor = None
        
        # Results storage
        self.trained_models_ = {}
        self.training_results_ = {}
        self.ensemble_weights_ = {}
        self.feature_importance_ = {}
        
        # Setup logging
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('EnsembleModelTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train complete ensemble of models
        
        Args:
            X: Training features with datetime index
            y: Training targets
            
        Returns:
            Dictionary with trained models and results
        """
        
        self.logger.info("Starting ensemble training")
        self.logger.info(f"Training models: {self.models}")
        self.logger.info(f"Training samples: {len(X)}")
        self.logger.info(f"Feature count: {X.shape[1]}")
        
        # Feature processing
        if self.feature_processor is not None:
            self.logger.info("Applying feature processing")
            X_processed = self.feature_processor.fit_transform(X, y)
            X_processed = pd.DataFrame(
                X_processed, 
                index=X.index,
                columns=[f"feature_{i}" for i in range(X_processed.shape[1])]
            )
        else:
            X_processed = X.copy()
        
        # Train individual models
        for model_name in self.models:
            self.logger.info(f"Training {model_name}")
            
            trainer = ModelTrainer(
                model_name=model_name,
                cv_strategy=self.cv_strategy,
                n_trials=self.optimization_trials
            )
            
            # Train model
            training_result = trainer.train(X_processed, y)
            
            # Store results
            self.trained_models_[model_name] = trainer.best_model_
            self.training_results_[model_name] = training_result
            
            if trainer.feature_importance_:
                self.feature_importance_[model_name] = trainer.feature_importance_
            
            self.logger.info(f"{model_name} - CV Score: {training_result['best_score']:.4f}")
        
        # Calculate ensemble weights based on CV performance
        self._calculate_ensemble_weights()
        
        # Create ensemble prediction function
        ensemble_predictor = self._create_ensemble_predictor(X_processed)
        
        self.logger.info("Ensemble training completed")
        self.logger.info(f"Ensemble weights: {self.ensemble_weights_}")
        
        return {
            'models': self.trained_models_,
            'ensemble_predict': ensemble_predictor,
            'feature_processor': self.feature_processor,
            'training_results': self.training_results_,
            'ensemble_weights': self.ensemble_weights_,
            'feature_importance': self.feature_importance_
        }
    
    def _calculate_ensemble_weights(self):
        """Calculate ensemble weights based on CV performance"""
        
        # Get CV scores
        cv_scores = {name: result['best_score'] 
                    for name, result in self.training_results_.items()}
        
        # Simple performance-based weighting
        total_score = sum(cv_scores.values())
        self.ensemble_weights_ = {name: score / total_score 
                                for name, score in cv_scores.items()}
        
        # Optional: Apply softmax for more concentrated weights
        scores_array = np.array(list(cv_scores.values()))
        softmax_weights = np.exp(scores_array * 5) / np.sum(np.exp(scores_array * 5))  # Temperature = 5
        
        for i, name in enumerate(cv_scores.keys()):
            self.ensemble_weights_[name] = softmax_weights[i]
    
    def _create_ensemble_predictor(self, X_sample: pd.DataFrame):
        """Create ensemble prediction function"""
        
        def ensemble_predict(X_new: pd.DataFrame) -> np.ndarray:
            """
            Make ensemble predictions on new data
            
            Args:
                X_new: New feature data
                
            Returns:
                Array of prediction probabilities
            """
            # Process features
            if self.feature_processor is not None:
                X_processed = self.feature_processor.transform(X_new)
                X_processed = pd.DataFrame(
                    X_processed,
                    index=X_new.index,
                    columns=[f"feature_{i}" for i in range(X_processed.shape[1])]
                )
            else:
                X_processed = X_new.copy()
            
            # Collect predictions from each model
            predictions = {}
            for model_name, model in self.trained_models_.items():
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_processed)[:, 1]
                else:
                    pred = model.predict(X_processed)
                predictions[model_name] = pred
            
            # Weighted ensemble
            ensemble_pred = np.zeros(len(X_processed))
            for model_name, weight in self.ensemble_weights_.items():
                ensemble_pred += weight * predictions[model_name]
            
            return ensemble_pred
        
        return ensemble_predict
    
    def evaluate_ensemble(self, ensemble_predict_func, X_test: pd.DataFrame, 
                         y_test: pd.Series) -> dict:
        """
        Evaluate ensemble performance on test set
        
        Args:
            ensemble_predict_func: Ensemble prediction function
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Evaluation metrics dictionary
        """
        
        # Generate predictions
        y_pred_proba = ensemble_predict_func(X_test)
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'auc_score': roc_auc_score(y_test, y_pred_proba),
            'accuracy': accuracy_score(y_test, y_pred_binary),
            'precision': precision_score(y_test, y_pred_binary),
            'recall': recall_score(y_test, y_pred_binary),
            'f1_score': f1_score(y_test, y_pred_binary),
            'n_samples': len(y_test),
            'positive_rate': y_test.mean(),
            'prediction_mean': y_pred_proba.mean(),
            'prediction_std': y_pred_proba.std()
        }
        
        # Additional analysis
        metrics['classification_report'] = classification_report(
            y_test, y_pred_binary, output_dict=True
        )
        
        return metrics
    
    def save_ensemble(self, filepath: str):
        """Save trained ensemble to disk"""
        
        ensemble_data = {
            'models': self.trained_models_,
            'feature_processor': self.feature_processor,
            'ensemble_weights': self.ensemble_weights_,
            'training_results': self.training_results_,
            'feature_importance': self.feature_importance_,
            'model_names': self.models,
            'training_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(ensemble_data, filepath)
        self.logger.info(f"Ensemble saved to {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath: str):
        """Load trained ensemble from disk"""
        
        ensemble_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(models=ensemble_data['model_names'])
        
        # Restore trained components
        instance.trained_models_ = ensemble_data['models']
        instance.feature_processor = ensemble_data['feature_processor']
        instance.ensemble_weights_ = ensemble_data['ensemble_weights']
        instance.training_results_ = ensemble_data['training_results']
        instance.feature_importance_ = ensemble_data['feature_importance']
        
        return instance