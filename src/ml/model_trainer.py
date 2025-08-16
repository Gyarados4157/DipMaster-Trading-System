"""
DipMaster Enhanced V4 - Advanced ML Model Training Framework
Implements comprehensive model training with time-series validation and ensemble learning.
"""

import numpy as np
import pandas as pd
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler

# Deep learning
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. LSTM models will be disabled.")

warnings.filterwarnings('ignore')

@dataclass
class ModelConfig:
    """Configuration for model training"""
    model_type: str
    hyperparams: Dict[str, Any]
    use_feature_selection: bool = True
    feature_selection_k: int = 30
    early_stopping_rounds: int = 100
    cv_folds: int = 5
    embargo_hours: int = 2

@dataclass
class ValidationResult:
    """Results from model validation"""
    model_name: str
    cv_scores: List[float]
    best_score: float
    feature_importance: Dict[str, float]
    confusion_matrix: np.ndarray
    classification_report: str
    hyperparams: Dict[str, Any]

class PurgedTimeSeriesSplit:
    """
    Time Series Cross-Validation with Purging and Embargo
    Prevents data leakage by adding gaps between train/test sets
    """
    
    def __init__(self, n_splits: int = 5, embargo_hours: int = 2, test_size_hours: int = 48):
        self.n_splits = n_splits
        self.embargo_hours = embargo_hours
        self.test_size_hours = test_size_hours
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate time-series splits with purging and embargo"""
        
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex for time-based splitting")
        
        n_samples = len(X)
        embargo_samples = int(self.embargo_hours * 12)  # 5-minute bars
        test_samples = int(self.test_size_hours * 12)   # 5-minute bars
        
        # Calculate split points
        total_test_samples = self.n_splits * test_samples
        total_embargo_samples = self.n_splits * embargo_samples
        train_samples = n_samples - total_test_samples - total_embargo_samples
        
        if train_samples <= 0:
            raise ValueError("Not enough data for time series split")
        
        splits = []
        
        for i in range(self.n_splits):
            # Calculate indices
            test_start = train_samples + i * (test_samples + embargo_samples)
            test_end = test_start + test_samples
            
            if test_end > n_samples:
                break
            
            # Train indices (up to embargo start)
            train_end = test_start - embargo_samples
            train_indices = np.arange(0, train_end)
            
            # Test indices
            test_indices = np.arange(test_start, test_end)
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        return splits

class AdvancedModelTrainer:
    """
    Advanced ML model trainer for DipMaster Enhanced V4
    Implements ensemble learning with rigorous time-series validation
    """
    
    def __init__(self, config_path: str = None):
        self.logger = self._setup_logging()
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.validation_results = {}
        
        # Default model configurations
        self.model_configs = {
            'lgbm_classifier': ModelConfig(
                model_type='lgbm_classifier',
                hyperparams={
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 20,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    'verbose': -1,
                    'n_estimators': 1000
                }
            ),
            'lgbm_regressor': ModelConfig(
                model_type='lgbm_regressor',
                hyperparams={
                    'objective': 'regression',
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 31,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 5,
                    'min_child_samples': 20,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    'verbose': -1,
                    'n_estimators': 1000
                }
            ),
            'xgb_classifier': ModelConfig(
                model_type='xgb_classifier',
                hyperparams={
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': 6,
                    'learning_rate': 0.05,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'min_child_weight': 1,
                    'reg_alpha': 0.1,
                    'reg_lambda': 0.1,
                    'random_state': 42,
                    'n_estimators': 1000
                }
            ),
            'random_forest': ModelConfig(
                model_type='random_forest',
                hyperparams={
                    'n_estimators': 200,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': -1
                }
            ),
            'logistic_regression': ModelConfig(
                model_type='logistic_regression',
                hyperparams={
                    'C': 1.0,
                    'penalty': 'elasticnet',
                    'l1_ratio': 0.5,
                    'solver': 'saga',
                    'max_iter': 1000,
                    'random_state': 42
                }
            )
        }
        
        if TORCH_AVAILABLE:
            self.model_configs['lstm'] = ModelConfig(
                model_type='lstm',
                hyperparams={
                    'hidden_size': 64,
                    'num_layers': 2,
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                    'batch_size': 256,
                    'epochs': 100,
                    'sequence_length': 60
                }
            )
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ModelTrainer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def optimize_hyperparameters(self, 
                                model_name: str,
                                X_train: pd.DataFrame, 
                                y_train: pd.Series,
                                cv_splitter: PurgedTimeSeriesSplit,
                                n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna with time-series CV
        """
        self.logger.info(f"Starting hyperparameter optimization for {model_name}")
        
        def objective(trial):
            if model_name == 'lgbm_classifier':
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
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                    'random_state': 42,
                    'verbose': -1,
                    'n_estimators': 1000
                }
                
                scores = []
                for train_idx, val_idx in cv_splitter.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model = lgb.LGBMClassifier(**params)
                    model.fit(X_tr, y_tr, 
                             eval_set=[(X_val, y_val)],
                             callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                    
                    pred = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, pred)
                    scores.append(score)
                
                return np.mean(scores)
            
            elif model_name == 'xgb_classifier':
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.4, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                    'random_state': 42,
                    'n_estimators': 1000
                }
                
                scores = []
                for train_idx, val_idx in cv_splitter.split(X_train):
                    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    model = xgb.XGBClassifier(**params)
                    model.fit(X_tr, y_tr, 
                             eval_set=[(X_val, y_val)],
                             early_stopping_rounds=50,
                             verbose=False)
                    
                    pred = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, pred)
                    scores.append(score)
                
                return np.mean(scores)
            
            # Add more model optimizations as needed
            return 0.5
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials, timeout=3600)  # 1 hour timeout
        
        self.logger.info(f"Best score for {model_name}: {study.best_value:.4f}")
        self.logger.info(f"Best params for {model_name}: {study.best_params}")
        
        return study.best_params
    
    def prepare_features(self, 
                        X: pd.DataFrame, 
                        y: pd.Series,
                        model_name: str,
                        fit_scalers: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features with scaling and selection
        """
        self.logger.info(f"Preparing features for {model_name}")
        
        # Handle missing values
        X_clean = X.fillna(method='ffill').fillna(method='bfill')
        
        # Feature scaling for models that need it
        scale_models = ['logistic_regression', 'lstm']
        if model_name in scale_models:
            if fit_scalers:
                scaler = RobustScaler()
                X_scaled = pd.DataFrame(
                    scaler.fit_transform(X_clean),
                    columns=X_clean.columns,
                    index=X_clean.index
                )
                self.scalers[model_name] = scaler
            else:
                scaler = self.scalers.get(model_name)
                if scaler is None:
                    raise ValueError(f"No fitted scaler found for {model_name}")
                X_scaled = pd.DataFrame(
                    scaler.transform(X_clean),
                    columns=X_clean.columns,
                    index=X_clean.index
                )
        else:
            X_scaled = X_clean
        
        # Feature selection
        config = self.model_configs.get(model_name)
        if config and config.use_feature_selection and fit_scalers:
            # Use mutual information for feature selection
            selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(config.feature_selection_k, len(X_scaled.columns))
            )
            X_selected = pd.DataFrame(
                selector.fit_transform(X_scaled, y),
                columns=X_scaled.columns[selector.get_support()],
                index=X_scaled.index
            )
            self.feature_selectors[model_name] = selector
        elif not fit_scalers and model_name in self.feature_selectors:
            selector = self.feature_selectors[model_name]
            X_selected = pd.DataFrame(
                selector.transform(X_scaled),
                columns=X_scaled.columns[selector.get_support()],
                index=X_scaled.index
            )
        else:
            X_selected = X_scaled
        
        self.logger.info(f"Feature preparation complete. Shape: {X_selected.shape}")
        return X_selected, y
    
    def train_single_model(self,
                          model_name: str,
                          X_train: pd.DataFrame,
                          y_train: pd.Series,
                          optimize_hyperparams: bool = True) -> Any:
        """
        Train a single model with optional hyperparameter optimization
        """
        self.logger.info(f"Training {model_name} model")
        
        # Prepare features
        X_prepared, y_prepared = self.prepare_features(X_train, y_train, model_name, fit_scalers=True)
        
        # Setup cross-validation
        cv_splitter = PurgedTimeSeriesSplit(
            n_splits=self.model_configs[model_name].cv_folds,
            embargo_hours=self.model_configs[model_name].embargo_hours
        )
        
        # Optimize hyperparameters if requested
        if optimize_hyperparams:
            best_params = self.optimize_hyperparameters(
                model_name, X_prepared, y_prepared, cv_splitter
            )
            # Update config with best params
            self.model_configs[model_name].hyperparams.update(best_params)
        
        # Train final model
        config = self.model_configs[model_name]
        
        if model_name == 'lgbm_classifier':
            model = lgb.LGBMClassifier(**config.hyperparams)
            model.fit(X_prepared, y_prepared)
            
        elif model_name == 'lgbm_regressor':
            model = lgb.LGBMRegressor(**config.hyperparams)
            model.fit(X_prepared, y_prepared)
            
        elif model_name == 'xgb_classifier':
            model = xgb.XGBClassifier(**config.hyperparams)
            model.fit(X_prepared, y_prepared)
            
        elif model_name == 'random_forest':
            model = RandomForestClassifier(**config.hyperparams)
            model.fit(X_prepared, y_prepared)
            
        elif model_name == 'logistic_regression':
            model = LogisticRegression(**config.hyperparams)
            model.fit(X_prepared, y_prepared)
            
        elif model_name == 'lstm' and TORCH_AVAILABLE:
            model = self._train_lstm_model(X_prepared, y_prepared, config.hyperparams)
        
        else:
            raise ValueError(f"Unknown model type: {model_name}")
        
        # Store model
        self.models[model_name] = model
        
        # Validate model
        validation_result = self._validate_model(
            model_name, model, X_prepared, y_prepared, cv_splitter
        )
        self.validation_results[model_name] = validation_result
        
        self.logger.info(f"Training complete for {model_name}. CV Score: {validation_result.best_score:.4f}")
        
        return model
    
    def _validate_model(self,
                       model_name: str,
                       model: Any,
                       X: pd.DataFrame,
                       y: pd.Series,
                       cv_splitter: PurgedTimeSeriesSplit) -> ValidationResult:
        """
        Validate model using time-series cross-validation
        """
        scores = []
        all_y_true = []
        all_y_pred = []
        
        for train_idx, val_idx in cv_splitter.split(X):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]
            
            # Clone and train model for this fold
            if model_name.startswith('lgbm'):
                fold_model = lgb.LGBMClassifier(**self.model_configs[model_name].hyperparams)
                fold_model.fit(X_train_cv, y_train_cv)
                y_pred_proba = fold_model.predict_proba(X_val_cv)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
                
            elif model_name.startswith('xgb'):
                fold_model = xgb.XGBClassifier(**self.model_configs[model_name].hyperparams)
                fold_model.fit(X_train_cv, y_train_cv)
                y_pred_proba = fold_model.predict_proba(X_val_cv)[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
                
            else:
                fold_model = model.__class__(**self.model_configs[model_name].hyperparams)
                fold_model.fit(X_train_cv, y_train_cv)
                if hasattr(fold_model, 'predict_proba'):
                    y_pred_proba = fold_model.predict_proba(X_val_cv)[:, 1]
                    y_pred = (y_pred_proba > 0.5).astype(int)
                else:
                    y_pred = fold_model.predict(X_val_cv)
                    y_pred_proba = y_pred
            
            score = roc_auc_score(y_val_cv, y_pred_proba)
            scores.append(score)
            
            all_y_true.extend(y_val_cv.values)
            all_y_pred.extend(y_pred)
        
        # Feature importance
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            for feature, importance in zip(X.columns, model.feature_importances_):
                feature_importance[feature] = float(importance)
        elif hasattr(model, 'coef_'):
            for feature, coef in zip(X.columns, model.coef_[0]):
                feature_importance[feature] = float(abs(coef))
        
        # Generate classification report
        conf_matrix = confusion_matrix(all_y_true, all_y_pred)
        class_report = classification_report(all_y_true, all_y_pred)
        
        return ValidationResult(
            model_name=model_name,
            cv_scores=scores,
            best_score=np.mean(scores),
            feature_importance=feature_importance,
            confusion_matrix=conf_matrix,
            classification_report=class_report,
            hyperparams=self.model_configs[model_name].hyperparams
        )
    
    def _train_lstm_model(self, X: pd.DataFrame, y: pd.Series, hyperparams: Dict) -> Any:
        """
        Train LSTM model for time-series prediction
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available for LSTM training")
        
        # Prepare sequences
        seq_length = hyperparams['sequence_length']
        X_sequences, y_sequences = self._create_sequences(X.values, y.values, seq_length)
        
        # Create data loaders
        dataset = TensorDataset(
            torch.FloatTensor(X_sequences),
            torch.FloatTensor(y_sequences)
        )
        dataloader = DataLoader(dataset, batch_size=hyperparams['batch_size'], shuffle=False)
        
        # Define LSTM model
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  dropout=dropout, batch_first=True)
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                output = self.fc(lstm_out[:, -1, :])  # Use last time step
                return self.sigmoid(output)
        
        # Initialize model
        model = LSTMModel(
            input_size=X.shape[1],
            hidden_size=hyperparams['hidden_size'],
            num_layers=hyperparams['num_layers'],
            dropout=hyperparams['dropout']
        )
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
        
        # Training loop
        model.train()
        for epoch in range(hyperparams['epochs']):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                self.logger.info(f"LSTM Epoch {epoch}, Loss: {total_loss/len(dataloader):.4f}")
        
        return model
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X_sequences = []
        y_sequences = []
        
        for i in range(seq_length, len(X)):
            X_sequences.append(X[i-seq_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def train_ensemble(self,
                      X_train: pd.DataFrame,
                      y_train: pd.Series,
                      model_names: List[str] = None) -> Dict[str, Any]:
        """
        Train ensemble of models
        """
        if model_names is None:
            model_names = ['lgbm_classifier', 'xgb_classifier', 'random_forest']
        
        self.logger.info(f"Training ensemble with models: {model_names}")
        
        # Train individual models
        ensemble_models = {}
        for model_name in model_names:
            model = self.train_single_model(model_name, X_train, y_train, optimize_hyperparams=True)
            ensemble_models[model_name] = model
        
        # Create ensemble predictions function
        def ensemble_predict(X_test: pd.DataFrame) -> np.ndarray:
            predictions = []
            
            for model_name, model in ensemble_models.items():
                X_prepared, _ = self.prepare_features(X_test, None, model_name, fit_scalers=False)
                
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_prepared)[:, 1]
                else:
                    pred = model.predict(X_prepared)
                
                predictions.append(pred)
            
            # Average predictions
            return np.mean(predictions, axis=0)
        
        ensemble_models['ensemble_predict'] = ensemble_predict
        return ensemble_models
    
    def generate_alpha_signals(self,
                             X_test: pd.DataFrame,
                             ensemble_models: Dict[str, Any],
                             confidence_threshold: float = 0.6) -> pd.DataFrame:
        """
        Generate trading signals from ensemble models
        """
        self.logger.info("Generating alpha signals")
        
        # Get ensemble predictions
        predictions = ensemble_models['ensemble_predict'](X_test)
        
        # Calculate signal strength
        signals_df = pd.DataFrame({
            'timestamp': X_test.index,
            'score': predictions,
            'confidence': np.abs(predictions - 0.5) * 2,  # Confidence from 0 to 1
            'predicted_return': predictions * 0.015,  # Scale to expected return
            'signal': (predictions > confidence_threshold).astype(int)
        })
        
        signals_df = signals_df[signals_df['confidence'] >= 0.2]  # Filter low confidence
        
        return signals_df
    
    def save_models(self, save_dir: str):
        """Save trained models and preprocessing objects"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save models (implementation depends on model type)
        for model_name, model in self.models.items():
            if model_name.startswith('lgbm'):
                model.booster_.save_model(str(save_path / f"{model_name}.txt"))
            elif model_name.startswith('xgb'):
                model.save_model(str(save_path / f"{model_name}.json"))
            # Add other model saving logic
        
        # Save preprocessing objects
        import pickle
        with open(save_path / "scalers.pkl", 'wb') as f:
            pickle.dump(self.scalers, f)
        
        with open(save_path / "feature_selectors.pkl", 'wb') as f:
            pickle.dump(self.feature_selectors, f)
        
        # Save validation results
        with open(save_path / "validation_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            results_dict = {}
            for name, result in self.validation_results.items():
                results_dict[name] = {
                    'model_name': result.model_name,
                    'cv_scores': result.cv_scores,
                    'best_score': result.best_score,
                    'feature_importance': result.feature_importance,
                    'confusion_matrix': result.confusion_matrix.tolist(),
                    'classification_report': result.classification_report,
                    'hyperparams': result.hyperparams
                }
            json.dump(results_dict, f, indent=2)
        
        self.logger.info(f"Models saved to {save_path}")