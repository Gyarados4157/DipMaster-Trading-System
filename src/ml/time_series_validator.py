"""
Time Series Cross-Validation with Purged Walk-Forward Analysis
Implements rigorous time-series validation to prevent data leakage.
"""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, List
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PurgedWalkForwardCV:
    """
    Purged Walk-Forward Cross-Validation for time series data.
    
    This implementation ensures:
    1. No data leakage between train/test periods
    2. Embargo periods to prevent label leakage
    3. Purging of overlapping samples
    4. Realistic out-of-sample testing
    """
    
    def __init__(self, 
                 n_splits: int = 5,
                 embargo_hours: int = 24,
                 min_train_samples: int = 1000,
                 min_test_samples: int = 100,
                 gap_hours: int = 4):
        """
        Initialize Purged Walk-Forward CV
        
        Args:
            n_splits: Number of walk-forward splits
            embargo_hours: Hours to embargo after training period
            min_train_samples: Minimum samples required in training set
            min_test_samples: Minimum samples required in test set
            gap_hours: Gap between train and test to prevent overlap
        """
        self.n_splits = n_splits
        self.embargo_hours = embargo_hours
        self.min_train_samples = min_train_samples
        self.min_test_samples = min_test_samples
        self.gap_hours = gap_hours
        
    def split(self, X: pd.DataFrame, y: pd.Series = None, 
              groups: pd.Series = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test splits for walk-forward analysis
        
        Args:
            X: Feature dataframe with datetime index
            y: Target series (optional)
            groups: Group labels (optional)
            
        Yields:
            train_idx, test_idx: Arrays of indices for train/test sets
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex")
            
        # Sort by time to ensure temporal order
        X = X.sort_index()
        total_samples = len(X)
        
        # Calculate split sizes
        base_train_size = total_samples // (self.n_splits + 1)
        test_size = total_samples // (self.n_splits * 2)  # Smaller test sets
        
        for split_idx in range(self.n_splits):
            # Calculate training period
            train_start_idx = 0
            train_end_idx = base_train_size + (split_idx * test_size)
            
            # Ensure minimum training samples
            if train_end_idx - train_start_idx < self.min_train_samples:
                train_end_idx = train_start_idx + self.min_train_samples
                
            # Calculate test period with gap and embargo
            gap_samples = self._hours_to_samples(X, self.gap_hours)
            embargo_samples = self._hours_to_samples(X, self.embargo_hours)
            
            test_start_idx = train_end_idx + gap_samples + embargo_samples
            test_end_idx = min(test_start_idx + test_size, total_samples)
            
            # Ensure minimum test samples
            if test_end_idx - test_start_idx < self.min_test_samples:
                if test_start_idx + self.min_test_samples > total_samples:
                    break  # Not enough data for this split
                test_end_idx = test_start_idx + self.min_test_samples
                
            # Create indices
            train_indices = np.arange(train_start_idx, train_end_idx)
            test_indices = np.arange(test_start_idx, test_end_idx)
            
            # Additional purging based on overlapping information
            train_indices, test_indices = self._purge_overlapping_samples(
                X, train_indices, test_indices
            )
            
            yield train_indices, test_indices
    
    def _hours_to_samples(self, X: pd.DataFrame, hours: int) -> int:
        """Convert hours to number of samples based on data frequency"""
        if len(X) < 2:
            return 0
            
        # Estimate frequency from first few timestamps
        time_diffs = X.index[1:6] - X.index[0:5]
        avg_freq = time_diffs.mean()
        
        samples_per_hour = timedelta(hours=1) / avg_freq
        return max(1, int(hours * samples_per_hour))
    
    def _purge_overlapping_samples(self, X: pd.DataFrame, 
                                 train_indices: np.ndarray, 
                                 test_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Purge samples that might have overlapping information
        """
        # For trading data, we need to be careful about:
        # 1. Label leakage (future returns calculated from overlapping periods)
        # 2. Feature overlap (rolling statistics computed on overlapping windows)
        
        train_times = X.index[train_indices]
        test_times = X.index[test_indices]
        
        # Remove training samples that are too close to test period
        purge_threshold = timedelta(hours=self.embargo_hours)
        train_cutoff = test_times.min() - purge_threshold
        
        valid_train_mask = train_times <= train_cutoff
        purged_train_indices = train_indices[valid_train_mask]
        
        # Test indices remain unchanged (they are already properly separated)
        return purged_train_indices, test_indices
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return number of splits"""
        return self.n_splits


class RobustTimeSeriesValidator:
    """
    Comprehensive time series validation with multiple methodologies
    """
    
    def __init__(self, 
                 primary_cv: PurgedWalkForwardCV = None,
                 bootstrap_samples: int = 1000,
                 confidence_level: float = 0.95):
        """
        Initialize comprehensive validator
        
        Args:
            primary_cv: Primary CV strategy
            bootstrap_samples: Number of bootstrap samples for CI
            confidence_level: Confidence level for statistical tests
        """
        self.primary_cv = primary_cv or PurgedWalkForwardCV()
        self.bootstrap_samples = bootstrap_samples
        self.confidence_level = confidence_level
        
    def validate_model(self, model, X: pd.DataFrame, y: pd.Series,
                      scoring_func=None, fit_params: dict = None) -> dict:
        """
        Perform comprehensive model validation
        
        Args:
            model: ML model with fit/predict interface
            X: Feature dataframe with datetime index
            y: Target series
            scoring_func: Custom scoring function
            fit_params: Additional parameters for model.fit()
            
        Returns:
            Validation results dictionary
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
        from sklearn.base import clone
        
        if scoring_func is None:
            scoring_func = roc_auc_score
            
        if fit_params is None:
            fit_params = {}
        
        results = {
            'cv_scores': [],
            'train_scores': [],
            'oos_periods': [],
            'feature_importance': [],
            'prediction_stats': []
        }
        
        # Walk-forward validation
        for fold_idx, (train_idx, test_idx) in enumerate(self.primary_cv.split(X, y)):
            # Clone model for this fold
            fold_model = clone(model)
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            fold_model.fit(X_train, y_train, **fit_params)
            
            # Generate predictions
            if hasattr(fold_model, 'predict_proba'):
                y_pred = fold_model.predict_proba(X_test)[:, 1]
                y_train_pred = fold_model.predict_proba(X_train)[:, 1]
            else:
                y_pred = fold_model.predict(X_test)
                y_train_pred = fold_model.predict(X_train)
            
            # Calculate scores
            test_score = scoring_func(y_test, y_pred)
            train_score = scoring_func(y_train, y_train_pred)
            
            results['cv_scores'].append(test_score)
            results['train_scores'].append(train_score)
            
            # Store period information
            results['oos_periods'].append({
                'train_start': X_train.index.min(),
                'train_end': X_train.index.max(),
                'test_start': X_test.index.min(),
                'test_end': X_test.index.max(),
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            })
            
            # Feature importance (if available)
            if hasattr(fold_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': fold_model.feature_importances_
                }).sort_values('importance', ascending=False)
                results['feature_importance'].append(importance_df)
            
            # Prediction statistics
            results['prediction_stats'].append({
                'test_mean_pred': np.mean(y_pred),
                'test_std_pred': np.std(y_pred),
                'test_positive_rate': np.mean(y_test),
                'prediction_range': [np.min(y_pred), np.max(y_pred)]
            })
        
        # Aggregate results
        results['mean_cv_score'] = np.mean(results['cv_scores'])
        results['std_cv_score'] = np.std(results['cv_scores'])
        results['mean_train_score'] = np.mean(results['train_scores'])
        
        # Calculate overfitting metrics
        results['overfitting_ratio'] = results['mean_train_score'] / results['mean_cv_score']
        results['performance_stability'] = 1 - results['std_cv_score'] / results['mean_cv_score']
        
        # Statistical significance test
        results['t_statistic'] = (results['mean_cv_score'] * np.sqrt(len(results['cv_scores']))) / results['std_cv_score']
        
        # Confidence intervals
        scores_array = np.array(results['cv_scores'])
        ci_alpha = 1 - self.confidence_level
        results['confidence_interval'] = [
            np.percentile(scores_array, 100 * ci_alpha / 2),
            np.percentile(scores_array, 100 * (1 - ci_alpha / 2))
        ]
        
        return results
    
    def bootstrap_validation(self, model, X: pd.DataFrame, y: pd.Series,
                           test_size: float = 0.2) -> dict:
        """
        Bootstrap validation for robust performance estimates
        """
        from sklearn.base import clone
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        
        bootstrap_scores = []
        
        for _ in range(self.bootstrap_samples):
            # Random temporal split while preserving time order
            split_point = np.random.uniform(0.6, 0.8)  # Random split between 60-80%
            split_idx = int(len(X) * split_point)
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            # Skip if test set is too small
            if len(X_test) < 50:
                continue
                
            # Train and evaluate
            boot_model = clone(model)
            boot_model.fit(X_train, y_train)
            
            if hasattr(boot_model, 'predict_proba'):
                y_pred = boot_model.predict_proba(X_test)[:, 1]
            else:
                y_pred = boot_model.predict(X_test)
                
            score = roc_auc_score(y_test, y_pred)
            bootstrap_scores.append(score)
        
        return {
            'bootstrap_mean': np.mean(bootstrap_scores),
            'bootstrap_std': np.std(bootstrap_scores),
            'bootstrap_ci': [
                np.percentile(bootstrap_scores, 2.5),
                np.percentile(bootstrap_scores, 97.5)
            ],
            'bootstrap_scores': bootstrap_scores
        }


class ModelStabilityAnalyzer:
    """
    Analyze model stability across different market conditions
    """
    
    def __init__(self):
        self.regime_columns = ['volatility_20', 'volume_ratio', 'trend_short']
        
    def analyze_regime_stability(self, model, X: pd.DataFrame, y: pd.Series,
                                predictions: np.ndarray) -> dict:
        """
        Analyze model performance across different market regimes
        """
        from sklearn.metrics import roc_auc_score, accuracy_score
        
        regime_results = {}
        
        # Volatility regimes
        if 'volatility_20' in X.columns:
            vol_median = X['volatility_20'].median()
            high_vol_mask = X['volatility_20'] > vol_median * 1.5
            low_vol_mask = X['volatility_20'] < vol_median * 0.7
            
            if high_vol_mask.sum() > 50:
                regime_results['high_volatility'] = {
                    'auc': roc_auc_score(y[high_vol_mask], predictions[high_vol_mask]),
                    'accuracy': accuracy_score(y[high_vol_mask], predictions[high_vol_mask] > 0.5),
                    'samples': high_vol_mask.sum(),
                    'positive_rate': y[high_vol_mask].mean()
                }
                
            if low_vol_mask.sum() > 50:
                regime_results['low_volatility'] = {
                    'auc': roc_auc_score(y[low_vol_mask], predictions[low_vol_mask]),
                    'accuracy': accuracy_score(y[low_vol_mask], predictions[low_vol_mask] > 0.5),
                    'samples': low_vol_mask.sum(),
                    'positive_rate': y[low_vol_mask].mean()
                }
        
        # Trend regimes
        if 'trend_short' in X.columns:
            uptrend_mask = X['trend_short'] > 0.02
            downtrend_mask = X['trend_short'] < -0.02
            
            if uptrend_mask.sum() > 50:
                regime_results['uptrend'] = {
                    'auc': roc_auc_score(y[uptrend_mask], predictions[uptrend_mask]),
                    'accuracy': accuracy_score(y[uptrend_mask], predictions[uptrend_mask] > 0.5),
                    'samples': uptrend_mask.sum(),
                    'positive_rate': y[uptrend_mask].mean()
                }
                
            if downtrend_mask.sum() > 50:
                regime_results['downtrend'] = {
                    'auc': roc_auc_score(y[downtrend_mask], predictions[downtrend_mask]),
                    'accuracy': accuracy_score(y[downtrend_mask], predictions[downtrend_mask] > 0.5),
                    'samples': downtrend_mask.sum(),
                    'positive_rate': y[downtrend_mask].mean()
                }
        
        # Time-based regimes
        if hasattr(X.index, 'hour'):
            # Asian session (22-06 UTC)
            asian_mask = ((X.index.hour >= 22) | (X.index.hour <= 6))
            # European session (06-14 UTC)
            european_mask = ((X.index.hour >= 6) & (X.index.hour <= 14))
            # US session (14-22 UTC)
            us_mask = ((X.index.hour >= 14) & (X.index.hour <= 22))
            
            for session_name, session_mask in [('asian', asian_mask), 
                                             ('european', european_mask), 
                                             ('us', us_mask)]:
                if session_mask.sum() > 50:
                    regime_results[f'{session_name}_session'] = {
                        'auc': roc_auc_score(y[session_mask], predictions[session_mask]),
                        'accuracy': accuracy_score(y[session_mask], predictions[session_mask] > 0.5),
                        'samples': session_mask.sum(),
                        'positive_rate': y[session_mask].mean()
                    }
        
        return regime_results
    
    def detect_performance_degradation(self, scores: List[float], 
                                     window_size: int = 5) -> dict:
        """
        Detect if model performance is degrading over time
        """
        if len(scores) < window_size * 2:
            return {'degradation_detected': False, 'reason': 'insufficient_data'}
        
        # Calculate rolling averages
        scores_array = np.array(scores)
        early_avg = np.mean(scores_array[:window_size])
        late_avg = np.mean(scores_array[-window_size:])
        
        # Statistical test for degradation
        from scipy import stats
        early_scores = scores_array[:window_size]
        late_scores = scores_array[-window_size:]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(early_scores, late_scores)
        
        degradation_threshold = 0.05  # 5% performance drop
        degradation_detected = (early_avg - late_avg) > degradation_threshold and p_value < 0.05
        
        return {
            'degradation_detected': degradation_detected,
            'early_performance': early_avg,
            'late_performance': late_avg,
            'performance_drop': early_avg - late_avg,
            'p_value': p_value,
            't_statistic': t_stat,
            'recommendation': 'retrain' if degradation_detected else 'continue'
        }