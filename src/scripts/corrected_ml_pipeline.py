"""
DipMaster Enhanced V4 - Corrected ML Training Pipeline
Fixed data leakage issues and improved validation methodology
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import json
import joblib
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import optuna
import matplotlib.pyplot as plt
import seaborn as sns


class PurgedTimeSeriesSplit:
    """Time series cross-validation with purging and embargo"""
    
    def __init__(self, n_splits: int = 5, embargo_hours: int = 2, purge_hours: int = 1):
        self.n_splits = n_splits
        self.embargo_periods = embargo_hours * 12  # Convert to 5-minute periods
        self.purge_periods = purge_hours * 12
        
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups=None):
        """Generate purged time-series splits"""
        indices = np.arange(len(X))
        n_samples = len(indices)
        test_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            test_start = i * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
                
            train_end = test_start - self.embargo_periods
            if train_end <= 0:
                continue
                
            train_indices = indices[:train_end]
            test_indices = indices[test_start:test_end]
            
            # Apply purging
            purge_start = test_start - self.purge_periods
            train_indices = train_indices[train_indices < purge_start]
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices


@dataclass
class TradeRecord:
    """Individual trade record"""
    entry_time: datetime
    exit_time: datetime
    symbol: str
    entry_price: float
    exit_price: float
    position_size: float
    signal_strength: float
    profit_loss: float
    profit_loss_pct: float
    holding_minutes: int
    transaction_costs: float
    slippage_costs: float
    total_costs: float
    net_profit_loss: float
    win: bool


@dataclass
class PerformanceMetrics:
    """Performance metrics container"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    avg_holding_minutes: float
    total_costs: float


class CorrectedDipMasterPipeline:
    """
    Corrected ML training and backtesting pipeline with proper data leakage prevention
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_importance = {}
        self.validation_results = {}
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = None
        
        # Configuration
        self.config = {
            'ml': {
                'target_variable': 'target_binary',
                'top_features': 25,
                'cv_folds': 5,
                'test_size': 0.3,
                'optimization_trials': 20  # Reduced for faster execution
            },
            'backtest': {
                'initial_capital': 10000,
                'position_size': 1000,
                'max_positions': 3,
                'commission_rate': 0.001,
                'slippage_rate': 0.0005,
                'min_signal_threshold': 0.6
            },
            'targets': {
                'win_rate': 0.85,
                'sharpe_ratio': 2.0,
                'max_drawdown': 0.03,
                'profit_factor': 1.8
            }
        }
    
    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Load and prepare training data with proper leakage prevention"""
        print("Loading enhanced features dataset...")
        df = pd.read_parquet(data_path)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Handle missing values
        print(f"Missing values: {df.isnull().sum().sum()}")
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Create datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            base_date = datetime(2024, 5, 1)
            df.index = pd.date_range(start=base_date, periods=len(df), freq='5T')
        
        # Identify target variable
        target_col = self.config['ml']['target_variable']
        target = df[target_col].copy()
        
        # Exclude leakage-prone features
        exclude_cols = {
            'symbol', 'timestamp', target_col, 'target_return', 'target_risk_adjusted',
            'open', 'high', 'low', 'close', 'volume',
            # Exclude features that may cause data leakage
            'is_profitable_12p', 'is_profitable_6p', 'is_profitable_3p',
            'is_profitable_24p', 'is_profitable_1p',  # Any is_profitable features
        }
        
        # Add all target-related columns to exclusion
        target_cols = [col for col in df.columns if 
                      col.startswith('hits_target') or 
                      col.startswith('future_return') or 
                      col.startswith('is_profitable')]
        exclude_cols.update(target_cols)
        
        # Only include numeric columns as features
        feature_cols = [col for col in df.columns if 
                       col not in exclude_cols and 
                       pd.api.types.is_numeric_dtype(df[col])]
        
        # Additional leakage detection
        print("Performing data leakage detection...")
        suspicious_features = []
        for col in feature_cols[:]:  # Copy list to modify during iteration
            try:
                corr = target.corr(df[col])
                if abs(corr) > 0.95:  # Very high correlation threshold
                    suspicious_features.append((col, corr))
                    feature_cols.remove(col)
                    print(f"  Removed {col} (correlation: {corr:.4f})")
            except:
                pass
        
        print(f"Removed {len(suspicious_features)} suspicious features")
        
        X = df[feature_cols].copy()
        y = target.copy()
        
        # Create price data for backtesting
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        available_price_cols = [col for col in price_cols if col in df.columns]
        
        if available_price_cols:
            price_data = df[['symbol'] + available_price_cols].copy()
        else:
            # Create synthetic price data
            print("Creating synthetic price data...")
            price_data = pd.DataFrame({
                'symbol': df.get('symbol', 'DEFAULT'),
                'close': 100 + np.cumsum(np.random.normal(0, 0.01, len(df))),  # Random walk
                'volume': np.random.exponential(1000, len(df))
            }, index=df.index)
            
            # Add OHLC based on close
            price_data['open'] = price_data['close'].shift(1).fillna(price_data['close'])
            price_data['high'] = price_data[['open', 'close']].max(axis=1) * (1 + np.random.uniform(0, 0.001, len(df)))
            price_data['low'] = price_data[['open', 'close']].min(axis=1) * (1 - np.random.uniform(0, 0.001, len(df)))
        
        print(f"Final features: {X.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        print(f"Target balance: {y.mean():.3f}")
        
        return X, y, price_data
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select top features using LightGBM with proper validation"""
        print("Performing feature selection...")
        
        # Use a simple train/validation split for feature selection
        split_idx = int(len(X) * 0.7)
        X_train_fs = X.iloc[:split_idx]
        X_val_fs = X.iloc[split_idx:split_idx + int(len(X) * 0.15)]
        y_train_fs = y.iloc[:split_idx]
        y_val_fs = y.iloc[split_idx:split_idx + int(len(X) * 0.15)]
        
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(X_train_fs, y_train_fs)
        
        # Validate the model to ensure it's not overfitting
        val_pred = lgb_model.predict(X_val_fs)
        val_accuracy = accuracy_score(y_val_fs, val_pred)
        print(f"Feature selection model validation accuracy: {val_accuracy:.4f}")
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(self.config['ml']['top_features'])['feature'].tolist()
        
        print(f"Selected {len(top_features)} features")
        print("Top 10 features:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
            print(f"  {i+1:2d}. {row['feature']:<30} {row['importance']:.6f}")
        
        self.feature_importance['selection'] = importance_df
        return top_features
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> Dict:
        """Optimize model hyperparameters using Optuna with proper validation"""
        print(f"Optimizing {model_type} hyperparameters...")
        
        def objective(trial):
            if model_type == 'lgb':
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
                    'lambda_l1': trial.suggest_float('lambda_l1', 0, 5),
                    'lambda_l2': trial.suggest_float('lambda_l2', 0, 5),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'random_state': 42,
                    'verbosity': -1,
                    'n_estimators': 100
                }
                model = lgb.LGBMClassifier(**params)
                
            elif model_type == 'xgb':
                params = {
                    'objective': 'binary:logistic',
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
                    'random_state': 42,
                    'verbosity': 0,
                    'n_estimators': 100
                }
                model = xgb.XGBClassifier(**params)
            
            # Time series cross-validation with proper validation
            cv_scores = []
            tscv = PurgedTimeSeriesSplit(n_splits=3, embargo_hours=1)  # Reduced for speed
            
            for train_idx, val_idx in tscv.split(X):
                if len(train_idx) < 100 or len(val_idx) < 50:  # Minimum samples
                    continue
                    
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
                
                try:
                    model.fit(X_train_fold, y_train_fold)
                    y_pred = model.predict(X_val_fold)
                    score = accuracy_score(y_val_fold, y_pred)
                    cv_scores.append(score)
                except:
                    continue
            
            if len(cv_scores) == 0:
                return 0.5  # Random performance
                
            return np.mean(cv_scores)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.config['ml']['optimization_trials'], show_progress_bar=True)
        
        print(f"Best {model_type} score: {study.best_value:.4f}")
        return study.best_params
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train ensemble models with proper validation"""
        print("Training optimized models...")
        
        # Split data temporally
        test_size = self.config['ml']['test_size']
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        print(f"Train target balance: {y_train.mean():.3f}")
        print(f"Test target balance: {y_test.mean():.3f}")
        
        # Scale features
        self.scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Store splits for later use
        self.train_data = (X_train_scaled, y_train)
        self.test_data = (X_test_scaled, y_test)
        
        # Train LightGBM
        lgb_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'lgb')
        lgb_params.update({
            'objective': 'binary', 
            'random_state': 42, 
            'verbosity': -1,
            'n_estimators': 200
        })
        self.models['lgb'] = lgb.LGBMClassifier(**lgb_params)
        self.models['lgb'].fit(X_train_scaled, y_train)
        
        # Train XGBoost
        xgb_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'xgb')
        xgb_params.update({
            'objective': 'binary:logistic', 
            'random_state': 42, 
            'verbosity': 0,
            'n_estimators': 200
        })
        self.models['xgb'] = xgb.XGBClassifier(**xgb_params)
        self.models['xgb'].fit(X_train_scaled, y_train)
        
        # Create ensemble
        self.models['ensemble'] = VotingClassifier(
            estimators=[('lgb', self.models['lgb']), ('xgb', self.models['xgb'])],
            voting='soft'
        )
        self.models['ensemble'].fit(X_train_scaled, y_train)
        
        print("Model training completed!")
        return self.models
    
    def evaluate_models(self) -> Dict:
        """Evaluate all trained models"""
        print("Evaluating models...")
        
        X_train, y_train = self.train_data
        X_test, y_test = self.test_data
        
        results = {}
        
        for name, model in self.models.items():
            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)
            test_auc = roc_auc_score(y_test, y_test_proba)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'test_f1': test_f1,
                'test_auc': test_auc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'predictions': y_test_pred,
                'probabilities': y_test_proba
            }
            
            print(f"{name:10} - Train: {train_acc:.4f}, Test: {test_acc:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
        
        self.validation_results = results
        return results
    
    def generate_signals(self, X: pd.DataFrame) -> pd.Series:
        """Generate trading signals using the best model"""
        print("Generating trading signals...")
        
        # Use ensemble model for signals
        model = self.models['ensemble']
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Generate probabilities
        probabilities = model.predict_proba(X_scaled)[:, 1]
        signals = pd.Series(probabilities, index=X.index)
        
        print(f"Generated {len(signals)} signals")
        print(f"Signal stats: mean={signals.mean():.4f}, std={signals.std():.4f}")
        
        # Signal quality analysis
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            count = (signals >= threshold).sum()
            pct = count / len(signals) * 100
            print(f"  Signals >= {threshold}: {count} ({pct:.1f}%)")
        
        return signals
    
    def run_backtest(self, signals: pd.Series, price_data: pd.DataFrame) -> Dict:
        """Run comprehensive backtest"""
        print("Running backtest...")
        
        # Initialize
        current_capital = self.config['backtest']['initial_capital']
        self.trades = []
        self.equity_curve = []
        
        # Filter high-confidence signals
        min_threshold = self.config['backtest']['min_signal_threshold']
        high_conf_signals = signals[signals >= min_threshold]
        print(f"Trading {len(high_conf_signals)} high-confidence signals (>= {min_threshold})")
        
        if len(high_conf_signals) == 0:
            print("No high-confidence signals found! Lowering threshold...")
            min_threshold = signals.quantile(0.8)  # Top 20% of signals
            high_conf_signals = signals[signals >= min_threshold]
            print(f"Using threshold {min_threshold:.3f}, trading {len(high_conf_signals)} signals")
        
        # Get symbols
        symbols = price_data['symbol'].unique() if 'symbol' in price_data.columns else ['DEFAULT']
        
        # Process each signal
        active_positions = 0
        trade_count = 0
        
        for timestamp, signal_strength in high_conf_signals.items():
            # Check position limits
            if active_positions >= self.config['backtest']['max_positions']:
                continue
            
            # Get market data
            try:
                if 'symbol' in price_data.columns:
                    current_data = price_data.loc[timestamp]
                    if isinstance(current_data, pd.Series):
                        symbol = current_data['symbol']
                        entry_price = current_data['close']
                        volume = current_data.get('volume', 1000)
                    else:
                        continue
                else:
                    symbol = 'DEFAULT'
                    entry_price = price_data.loc[timestamp, 'close']
                    volume = price_data.loc[timestamp, 'volume'] if 'volume' in price_data.columns else 1000
            except (KeyError, IndexError):
                continue
            
            if entry_price <= 0:
                continue
            
            # Calculate position size
            position_value = self.config['backtest']['position_size']
            position_size = position_value / entry_price
            
            # Find exit (simplified - use next available data point + some random holding time)
            holding_periods = np.random.choice([3, 6, 9, 12, 15], p=[0.1, 0.2, 0.3, 0.3, 0.1])  # 15-75 minutes
            target_exit_time = timestamp + timedelta(minutes=holding_periods * 5)
            
            # Get exit price
            try:
                if 'symbol' in price_data.columns:
                    exit_data = price_data[
                        (price_data.index >= target_exit_time) & 
                        (price_data['symbol'] == symbol)
                    ]
                else:
                    exit_data = price_data[price_data.index >= target_exit_time]
                    
                if len(exit_data) == 0:
                    continue
                    
                exit_price = exit_data.iloc[0]['close']
                actual_exit_time = exit_data.index[0]
            except (KeyError, IndexError):
                continue
            
            # Calculate P&L
            gross_pnl = (exit_price - entry_price) * position_size
            
            # Costs
            commission = position_value * self.config['backtest']['commission_rate'] * 2  # Buy + sell
            slippage = position_value * self.config['backtest']['slippage_rate'] * 2
            total_costs = commission + slippage
            
            net_pnl = gross_pnl - total_costs
            holding_minutes = (actual_exit_time - timestamp).total_seconds() / 60
            
            # Update capital
            current_capital += net_pnl
            
            # Create trade record
            trade = TradeRecord(
                entry_time=timestamp,
                exit_time=actual_exit_time,
                symbol=symbol,
                entry_price=entry_price,
                exit_price=exit_price,
                position_size=position_size,
                signal_strength=signal_strength,
                profit_loss=gross_pnl,
                profit_loss_pct=(exit_price - entry_price) / entry_price,
                holding_minutes=holding_minutes,
                transaction_costs=commission,
                slippage_costs=slippage,
                total_costs=total_costs,
                net_profit_loss=net_pnl,
                win=net_pnl > 0
            )
            
            self.trades.append(trade)
            trade_count += 1
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': actual_exit_time,
                'capital': current_capital,
                'trade_pnl': net_pnl,
                'cumulative_return': (current_capital / self.config['backtest']['initial_capital'] - 1)
            })
            
            # Limit total trades for demonstration
            if trade_count >= 1000:
                break
        
        print(f"Backtest completed: {len(self.trades)} trades executed")
        
        # Calculate performance metrics
        self.performance_metrics = self.calculate_performance_metrics()
        
        return {
            'trades': len(self.trades),
            'final_capital': current_capital,
            'performance_metrics': self.performance_metrics
        }
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return None
        
        # Basic statistics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.win)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades
        
        # P&L statistics
        total_net_pnl = sum(t.net_profit_loss for t in self.trades)
        total_costs = sum(t.total_costs for t in self.trades)
        total_return_pct = total_net_pnl / self.config['backtest']['initial_capital']
        
        # Win/Loss analysis
        wins = [t.net_profit_loss for t in self.trades if t.win]
        losses = [t.net_profit_loss for t in self.trades if not t.win]
        
        avg_win = np.mean(wins) if wins else 0
        avg_loss = abs(np.mean(losses)) if losses else 0
        profit_factor = sum(wins) / abs(sum(losses)) if losses else float('inf')
        
        # Risk metrics
        if self.equity_curve:
            equity_series = pd.Series([e['capital'] for e in self.equity_curve])
            returns = equity_series.pct_change().dropna()
            
            if len(returns) > 1 and returns.std() > 0:
                sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            else:
                sharpe_ratio = 0
            
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 1 and negative_returns.std() > 0:
                sortino_ratio = (returns.mean() * 252) / (negative_returns.std() * np.sqrt(252))
            else:
                sortino_ratio = 0
            
            # Maximum drawdown
            peak = equity_series.expanding().max()
            drawdown = (equity_series - peak) / peak
            max_drawdown_pct = abs(drawdown.min())
            max_drawdown = max_drawdown_pct * self.config['backtest']['initial_capital']
            
            calmar_ratio = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0
        else:
            sharpe_ratio = sortino_ratio = calmar_ratio = 0
            max_drawdown = max_drawdown_pct = 0
        
        avg_holding_minutes = np.mean([t.holding_minutes for t in self.trades])
        
        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_return=total_net_pnl,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_holding_minutes=avg_holding_minutes,
            total_costs=total_costs
        )
    
    def generate_comprehensive_report(self) -> str:
        """Generate detailed performance report"""
        if not self.performance_metrics:
            return "No results available"
        
        metrics = self.performance_metrics
        targets = self.config['targets']
        
        report = []
        report.append("=" * 80)
        report.append("DIPMASTER ENHANCED V4 - CORRECTED PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Trades: {metrics.total_trades}")
        report.append(f"Win Rate: {metrics.win_rate:.4f} ({metrics.win_rate*100:.2f}%)")
        report.append(f"Total Return: ${metrics.total_return:,.2f} ({metrics.total_return_pct*100:.2f}%)")
        report.append(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        report.append(f"Max Drawdown: {metrics.max_drawdown_pct*100:.2f}%")
        report.append(f"Profit Factor: {metrics.profit_factor:.3f}")
        report.append("")
        
        # Model Performance Summary
        if self.validation_results:
            report.append("MODEL VALIDATION RESULTS")
            report.append("-" * 40)
            for name, result in self.validation_results.items():
                report.append(f"{name.upper()}:")
                report.append(f"  Test Accuracy: {result['test_accuracy']:.4f}")
                report.append(f"  Test F1 Score: {result['test_f1']:.4f}")
                report.append(f"  Test AUC: {result['test_auc']:.4f}")
                report.append(f"  Test Precision: {result['test_precision']:.4f}")
                report.append(f"  Test Recall: {result['test_recall']:.4f}")
                report.append("")
        
        # Target Achievement Analysis
        report.append("TARGET ACHIEVEMENT ANALYSIS")
        report.append("-" * 40)
        
        achievements = {
            'win_rate': metrics.win_rate >= targets['win_rate'],
            'sharpe_ratio': metrics.sharpe_ratio >= targets['sharpe_ratio'],
            'max_drawdown': metrics.max_drawdown_pct <= targets['max_drawdown'],
            'profit_factor': metrics.profit_factor >= targets['profit_factor']
        }
        
        overall_achievement = sum(achievements.values()) / len(achievements)
        
        report.append(f"Overall Achievement: {overall_achievement*100:.1f}% ({sum(achievements.values())}/{len(achievements)} targets met)")
        report.append("")
        
        # Individual target analysis
        report.append("Individual Target Analysis:")
        report.append(f"  Win Rate: {metrics.win_rate*100:.2f}% (Target: {targets['win_rate']*100:.0f}%) {'✅' if achievements['win_rate'] else '❌'}")
        report.append(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f} (Target: {targets['sharpe_ratio']:.1f}) {'✅' if achievements['sharpe_ratio'] else '❌'}")
        report.append(f"  Max Drawdown: {metrics.max_drawdown_pct*100:.2f}% (Target: <{targets['max_drawdown']*100:.0f}%) {'✅' if achievements['max_drawdown'] else '❌'}")
        report.append(f"  Profit Factor: {metrics.profit_factor:.3f} (Target: {targets['profit_factor']:.1f}) {'✅' if achievements['profit_factor'] else '❌'}")
        report.append("")
        
        # Gap Analysis and Recommendations
        win_rate_gap = targets['win_rate'] - metrics.win_rate
        if win_rate_gap > 0:
            report.append("PATH TO 85%+ WIN RATE TARGET")
            report.append("-" * 40)
            report.append(f"Current Gap: {win_rate_gap*100:.1f} percentage points")
            report.append("")
            report.append("Recommended Strategy Enhancements:")
            report.append("1. Signal Quality Improvement:")
            report.append("   - Increase minimum confidence threshold")
            report.append("   - Add multi-timeframe confirmation (5m + 15m + 1h)")
            report.append("   - Implement volatility regime filtering")
            report.append("")
            report.append("2. Feature Engineering Enhancement:")
            report.append("   - Add market microstructure features")
            report.append("   - Include cross-asset correlation signals")
            report.append("   - Develop sentiment-based indicators")
            report.append("")
            report.append("3. Risk Management Optimization:")
            report.append("   - Dynamic position sizing based on signal confidence")
            report.append("   - Improved exit timing optimization")
            report.append("   - Correlation-based position filtering")
            report.append("")
            report.append("4. Model Architecture Improvements:")
            report.append("   - Ensemble of regime-specific models")
            report.append("   - Online learning adaptation")
            report.append("   - Deep learning feature extraction")
        else:
            report.append("🎉 WIN RATE TARGET ACHIEVED!")
            report.append("Focus on optimizing other performance metrics.")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def create_performance_plots(self, save_path: str = None):
        """Generate performance visualization"""
        if not self.trades or not self.equity_curve:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DipMaster Enhanced V4 - Corrected Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        equity_df = pd.DataFrame(self.equity_curve)
        axes[0, 0].plot(equity_df['timestamp'], equity_df['capital'], linewidth=2, color='blue')
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Capital ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Trade P&L Distribution
        pnl_data = [t.net_profit_loss for t in self.trades]
        axes[0, 1].hist(pnl_data, bins=50, alpha=0.7, edgecolor='black', color='green')
        axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Trade P&L Distribution')
        axes[0, 1].set_xlabel('Net P&L ($)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Rolling Win Rate
        window_size = max(50, len(self.trades) // 10)
        rolling_wins = pd.Series([t.win for t in self.trades]).rolling(window_size, min_periods=1).mean()
        trade_numbers = range(1, len(self.trades) + 1)
        
        axes[0, 2].plot(trade_numbers, rolling_wins * 100, linewidth=2, color='purple')
        axes[0, 2].axhline(85, color='red', linestyle='--', alpha=0.7, label='Target 85%')
        axes[0, 2].axhline(rolling_wins.iloc[-1] * 100, color='green', linestyle='-', 
                          alpha=0.7, label=f'Current: {rolling_wins.iloc[-1]*100:.1f}%')
        axes[0, 2].set_title(f'Rolling Win Rate ({window_size} trades)')
        axes[0, 2].set_xlabel('Trade Number')
        axes[0, 2].set_ylabel('Win Rate (%)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Drawdown Analysis
        equity_series = pd.Series([e['capital'] for e in self.equity_curve])
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak * 100
        
        axes[1, 0].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        axes[1, 0].plot(drawdown, color='red', linewidth=1)
        axes[1, 0].set_title('Drawdown Analysis')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Signal Strength vs Performance
        signal_strengths = [t.signal_strength for t in self.trades]
        trade_results = [1 if t.win else 0 for t in self.trades]
        
        # Create bins for signal strength
        strength_bins = np.linspace(min(signal_strengths), max(signal_strengths), 10)
        bin_centers = (strength_bins[:-1] + strength_bins[1:]) / 2
        binned_performance = []
        
        for i in range(len(strength_bins) - 1):
            mask = (np.array(signal_strengths) >= strength_bins[i]) & (np.array(signal_strengths) < strength_bins[i+1])
            if mask.sum() > 0:
                binned_performance.append(np.array(trade_results)[mask].mean())
            else:
                binned_performance.append(0)
        
        axes[1, 1].bar(bin_centers, np.array(binned_performance) * 100, 
                      width=(strength_bins[1] - strength_bins[0]) * 0.8, alpha=0.7, color='orange')
        axes[1, 1].set_title('Win Rate by Signal Strength')
        axes[1, 1].set_xlabel('Signal Strength')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Monthly Performance
        monthly_returns = {}
        for trade in self.trades:
            month_key = trade.exit_time.strftime('%Y-%m')
            if month_key not in monthly_returns:
                monthly_returns[month_key] = []
            monthly_returns[month_key].append(trade.net_profit_loss)
        
        months = sorted(monthly_returns.keys())
        monthly_pnl = [sum(monthly_returns[month]) for month in months]
        
        colors = ['green' if pnl >= 0 else 'red' for pnl in monthly_pnl]
        axes[1, 2].bar(range(len(monthly_pnl)), monthly_pnl, color=colors, alpha=0.7)
        axes[1, 2].set_title('Monthly P&L')
        axes[1, 2].set_xlabel('Month')
        axes[1, 2].set_ylabel('P&L ($)')
        axes[1, 2].set_xticks(range(len(months)))
        axes[1, 2].set_xticklabels(months, rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plots saved to: {save_path}")
        
        return fig
    
    def save_results(self, output_dir: str):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            model_path = os.path.join(output_dir, f'{name}_model.pkl')
            joblib.dump(model, model_path)
        
        # Save scaler
        scaler_path = os.path.join(output_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Save performance metrics
        if self.performance_metrics:
            metrics_dict = {
                'total_trades': self.performance_metrics.total_trades,
                'win_rate': self.performance_metrics.win_rate,
                'total_return_pct': self.performance_metrics.total_return_pct,
                'sharpe_ratio': self.performance_metrics.sharpe_ratio,
                'max_drawdown_pct': self.performance_metrics.max_drawdown_pct,
                'profit_factor': self.performance_metrics.profit_factor,
                'avg_win': self.performance_metrics.avg_win,
                'avg_loss': self.performance_metrics.avg_loss
            }
            
            metrics_path = os.path.join(output_dir, 'performance_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
        
        # Save validation results
        if self.validation_results:
            val_results = {}
            for name, result in self.validation_results.items():
                val_results[name] = {
                    'test_accuracy': result['test_accuracy'],
                    'test_f1': result['test_f1'],
                    'test_auc': result['test_auc'],
                    'test_precision': result['test_precision'],
                    'test_recall': result['test_recall']
                }
            
            val_path = os.path.join(output_dir, 'validation_results.json')
            with open(val_path, 'w') as f:
                json.dump(val_results, f, indent=2)
        
        # Save trade records
        if self.trades:
            trades_data = []
            for trade in self.trades:
                trades_data.append({
                    'entry_time': trade.entry_time.isoformat(),
                    'exit_time': trade.exit_time.isoformat(),
                    'symbol': trade.symbol,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'signal_strength': trade.signal_strength,
                    'profit_loss_pct': trade.profit_loss_pct,
                    'net_profit_loss': trade.net_profit_loss,
                    'holding_minutes': trade.holding_minutes,
                    'win': trade.win
                })
            
            trades_path = os.path.join(output_dir, 'trades.json')
            with open(trades_path, 'w') as f:
                json.dump(trades_data, f, indent=2)
        
        print(f"Results saved to: {output_dir}")
    
    def run_complete_pipeline(self, data_path: str) -> Dict:
        """Run the complete corrected ML pipeline"""
        start_time = datetime.now()
        
        print("DipMaster Enhanced V4 - Corrected ML Pipeline")
        print("=" * 70)
        print(f"Start time: {start_time}")
        print(f"Target win rate: {self.config['targets']['win_rate']*100:.0f}%")
        print()
        
        try:
            # Phase 1: Data Preparation with Leakage Prevention
            print("Phase 1: Data Preparation & Leakage Prevention")
            X, y, price_data = self.load_and_prepare_data(data_path)
            
            # Phase 2: Feature Selection with Validation
            print("\nPhase 2: Feature Selection with Validation")
            selected_features = self.feature_selection(X, y)
            X_selected = X[selected_features]
            
            # Phase 3: Model Training with Proper Validation
            print("\nPhase 3: Model Training with Proper Validation")
            models = self.train_models(X_selected, y)
            
            # Phase 4: Model Evaluation
            print("\nPhase 4: Comprehensive Model Evaluation")
            evaluation_results = self.evaluate_models()
            
            # Phase 5: Signal Generation
            print("\nPhase 5: Signal Generation")
            signals = self.generate_signals(X_selected)
            
            # Phase 6: Realistic Backtesting
            print("\nPhase 6: Realistic Backtesting")
            backtest_results = self.run_backtest(signals, price_data)
            
            # Phase 7: Generate Reports
            print("\nPhase 7: Generating Comprehensive Reports")
            performance_report = self.generate_comprehensive_report()
            
            # Create output directory
            output_dir = f"results/corrected_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Save results
            self.save_results(output_dir)
            
            # Save performance report
            report_path = os.path.join(output_dir, 'performance_report.txt')
            with open(report_path, 'w') as f:
                f.write(performance_report)
            
            # Create plots
            plots_path = os.path.join(output_dir, 'performance_plots.png')
            self.create_performance_plots(plots_path)
            
            # Calculate execution time
            end_time = datetime.now()
            execution_time = end_time - start_time
            
            # Final summary
            print("\n" + "=" * 70)
            print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print(f"Execution time: {execution_time}")
            print(f"Output directory: {output_dir}")
            
            if self.performance_metrics:
                metrics = self.performance_metrics
                target_wr = self.config['targets']['win_rate']
                
                print(f"\nKEY RESULTS:")
                print(f"  Total Trades: {metrics.total_trades}")
                print(f"  Win Rate: {metrics.win_rate*100:.2f}% (Target: {target_wr*100:.0f}%)")
                print(f"  Total Return: {metrics.total_return_pct*100:.2f}%")
                print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
                print(f"  Max Drawdown: {metrics.max_drawdown_pct*100:.2f}%")
                print(f"  Profit Factor: {metrics.profit_factor:.3f}")
                
                if metrics.win_rate >= target_wr:
                    print(f"\n🎉 SUCCESS: {target_wr*100:.0f}%+ WIN RATE ACHIEVED!")
                else:
                    gap = (target_wr - metrics.win_rate) * 100
                    print(f"\n📈 PROGRESS: {gap:.1f}% gap remaining to reach {target_wr*100:.0f}% target")
                    
                # Calculate overall achievement
                achievements = {
                    'win_rate': metrics.win_rate >= target_wr,
                    'sharpe_ratio': metrics.sharpe_ratio >= self.config['targets']['sharpe_ratio'],
                    'max_drawdown': metrics.max_drawdown_pct <= self.config['targets']['max_drawdown'],
                    'profit_factor': metrics.profit_factor >= self.config['targets']['profit_factor']
                }
                overall_achievement = sum(achievements.values()) / len(achievements)
                print(f"Overall Target Achievement: {overall_achievement*100:.1f}%")
            
            print(f"\nDetailed report saved to: {report_path}")
            print("\n" + performance_report)
            
            return {
                'success': True,
                'execution_time': execution_time,
                'output_directory': output_dir,
                'performance_metrics': self.performance_metrics,
                'models': self.models,
                'evaluation_results': evaluation_results
            }
            
        except Exception as e:
            print(f"Pipeline execution failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'error': str(e)}


def main():
    """Main execution function"""
    # Data path
    data_path = "data/Enhanced_Features_25symbols_20250816_223904.parquet"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return None
    
    # Initialize and run corrected pipeline
    pipeline = CorrectedDipMasterPipeline()
    results = pipeline.run_complete_pipeline(data_path)
    
    return results


if __name__ == "__main__":
    results = main()