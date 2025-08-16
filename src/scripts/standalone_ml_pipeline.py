"""
DipMaster Enhanced V4 - Standalone ML Training and Backtesting Pipeline
Complete implementation without external module dependencies
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


class StandaloneDipMasterPipeline:
    """
    Complete standalone ML training and backtesting pipeline
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
                'optimization_trials': 30  # Reduced for faster execution
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
        """Load and prepare training data"""
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
        
        # Identify features and targets
        target_col = self.config['ml']['target_variable']
        exclude_cols = {
            'symbol', 'timestamp', target_col, 'target_return', 'target_risk_adjusted',
            'open', 'high', 'low', 'close', 'volume'
        }
        
        # Add target columns to exclusion
        target_cols = [col for col in df.columns if 
                      col.startswith('hits_target') or col.startswith('future_return')]
        exclude_cols.update(target_cols)
        
        # Only include numeric columns as features
        feature_cols = [col for col in df.columns if 
                       col not in exclude_cols and 
                       pd.api.types.is_numeric_dtype(df[col])]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
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
                'open': 100 * (1 + np.random.normal(0, 0.001, len(df))),
                'high': 100 * (1 + np.random.normal(0.001, 0.001, len(df))),
                'low': 100 * (1 + np.random.normal(-0.001, 0.001, len(df))),
                'close': 100 * (1 + np.random.normal(0, 0.001, len(df))),
                'volume': np.random.exponential(1000, len(df))
            }, index=df.index)
        
        print(f"Features: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
        return X, y, price_data
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select top features using LightGBM"""
        print("Performing feature selection...")
        
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=100,
            random_state=42,
            verbosity=-1
        )
        lgb_model.fit(X, y)
        
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
        """Optimize model hyperparameters using Optuna"""
        print(f"Optimizing {model_type} hyperparameters...")
        
        def objective(trial):
            if model_type == 'lgb':
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
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
            
            # Cross-validation
            cv_scores = []
            tscv = PurgedTimeSeriesSplit(n_splits=self.config['ml']['cv_folds'])
            
            for train_idx, val_idx in tscv.split(X):
                X_train_fold = X.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_train_fold = y.iloc[train_idx]
                y_val_fold = y.iloc[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                y_pred = model.predict(X_val_fold)
                score = accuracy_score(y_val_fold, y_pred)  # Use accuracy as primary metric
                cv_scores.append(score)
            
            return np.mean(cv_scores)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=self.config['ml']['optimization_trials'], show_progress_bar=True)
        
        print(f"Best {model_type} score: {study.best_value:.4f}")
        return study.best_params
    
    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train ensemble models"""
        print("Training optimized models...")
        
        # Split data
        test_size = self.config['ml']['test_size']
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
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
        lgb_params.update({'objective': 'binary', 'random_state': 42, 'verbosity': -1})
        self.models['lgb'] = lgb.LGBMClassifier(**lgb_params)
        self.models['lgb'].fit(X_train_scaled, y_train)
        
        # Train XGBoost
        xgb_params = self.optimize_hyperparameters(X_train_scaled, y_train, 'xgb')
        xgb_params.update({'objective': 'binary:logistic', 'random_state': 42, 'verbosity': 0})
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
            
            results[name] = {
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'test_f1': test_f1,
                'test_auc': test_auc,
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
        
        # Get symbols
        symbols = price_data['symbol'].unique() if 'symbol' in price_data.columns else ['DEFAULT']
        
        # Process each signal
        active_positions = 0
        
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
                    else:
                        continue
                else:
                    symbol = 'DEFAULT'
                    entry_price = price_data.loc[timestamp, 'close']
            except (KeyError, IndexError):
                continue
            
            if entry_price <= 0:
                continue
            
            # Calculate position size
            position_value = self.config['backtest']['position_size']
            position_size = position_value / entry_price
            
            # Find exit (simplified - use next 15-minute boundary)
            entry_minute = timestamp.minute
            target_exit_minutes = [15, 30, 45, 0]  # 15-minute boundaries
            
            # Find next boundary
            next_boundaries = [m for m in target_exit_minutes if m > entry_minute]
            if not next_boundaries:
                target_minute = target_exit_minutes[0]  # Next hour
                exit_time = timestamp.replace(minute=target_minute) + timedelta(hours=1)
            else:
                target_minute = next_boundaries[0]
                exit_time = timestamp.replace(minute=target_minute)
            
            # Get exit price
            try:
                if 'symbol' in price_data.columns:
                    exit_data = price_data[
                        (price_data.index >= exit_time) & 
                        (price_data['symbol'] == symbol)
                    ]
                else:
                    exit_data = price_data[price_data.index >= exit_time]
                    
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
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': actual_exit_time,
                'capital': current_capital,
                'trade_pnl': net_pnl,
                'cumulative_return': (current_capital / self.config['backtest']['initial_capital'] - 1)
            })
        
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
        report.append("DIPMASTER ENHANCED V4 - COMPREHENSIVE PERFORMANCE REPORT")
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
        
        # Detailed Performance Metrics
        report.append("DETAILED METRICS")
        report.append("-" * 40)
        report.append(f"Winning Trades: {metrics.winning_trades}")
        report.append(f"Losing Trades: {metrics.losing_trades}")
        report.append(f"Average Win: ${metrics.avg_win:.2f}")
        report.append(f"Average Loss: ${metrics.avg_loss:.2f}")
        report.append(f"Average Holding Time: {metrics.avg_holding_minutes:.1f} minutes")
        report.append(f"Sortino Ratio: {metrics.sortino_ratio:.3f}")
        report.append(f"Calmar Ratio: {metrics.calmar_ratio:.3f}")
        report.append(f"Total Transaction Costs: ${metrics.total_costs:.2f}")
        report.append("")
        
        # Improvement Recommendations
        if not achievements['win_rate']:
            gap = (targets['win_rate'] - metrics.win_rate) * 100
            report.append("RECOMMENDATIONS FOR 85%+ WIN RATE TARGET")
            report.append("-" * 40)
            report.append(f"Current gap: {gap:.1f} percentage points")
            report.append("")
            report.append("Strategic improvements:")
            report.append("1. Increase signal confidence threshold to 0.7+")
            report.append("2. Implement regime-aware filtering (volatility, trend)")
            report.append("3. Add multi-timeframe confirmation (5m + 15m + 1h)")
            report.append("4. Optimize exit timing with market microstructure")
            report.append("5. Dynamic position sizing based on signal quality")
            report.append("6. Add correlation filtering to avoid similar trades")
            report.append("7. Implement sector/momentum rotation filters")
        else:
            report.append("🎉 WIN RATE TARGET ACHIEVED!")
            report.append("Consider optimizing other metrics or increasing targets.")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def create_performance_plots(self, save_path: str = None):
        """Generate performance visualization"""
        if not self.trades or not self.equity_curve:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DipMaster Enhanced V4 - Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Equity Curve
        equity_df = pd.DataFrame(self.equity_curve)
        axes[0, 0].plot(equity_df['timestamp'], equity_df['capital'], linewidth=2)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_ylabel('Capital ($)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Trade P&L Distribution
        pnl_data = [t.net_profit_loss for t in self.trades]
        axes[0, 1].hist(pnl_data, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Trade P&L Distribution')
        axes[0, 1].set_xlabel('Net P&L ($)')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. Rolling Win Rate
        window_size = max(20, len(self.trades) // 10)
        rolling_wins = pd.Series([t.win for t in self.trades]).rolling(window_size, min_periods=1).mean()
        trade_numbers = range(1, len(self.trades) + 1)
        
        axes[1, 0].plot(trade_numbers, rolling_wins * 100, linewidth=2)
        axes[1, 0].axhline(85, color='red', linestyle='--', alpha=0.7, label='Target 85%')
        axes[1, 0].axhline(rolling_wins.iloc[-1] * 100, color='green', linestyle='-', 
                          alpha=0.7, label=f'Current: {rolling_wins.iloc[-1]*100:.1f}%')
        axes[1, 0].set_title(f'Rolling Win Rate ({window_size} trades)')
        axes[1, 0].set_xlabel('Trade Number')
        axes[1, 0].set_ylabel('Win Rate (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Signal Strength vs. Performance
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
                      width=(strength_bins[1] - strength_bins[0]) * 0.8, alpha=0.7)
        axes[1, 1].set_title('Win Rate by Signal Strength')
        axes[1, 1].set_xlabel('Signal Strength')
        axes[1, 1].set_ylabel('Win Rate (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
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
                'profit_factor': self.performance_metrics.profit_factor
            }
            
            metrics_path = os.path.join(output_dir, 'performance_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics_dict, f, indent=2)
        
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
                    'net_profit_loss': trade.net_profit_loss,
                    'win': trade.win
                })
            
            trades_path = os.path.join(output_dir, 'trades.json')
            with open(trades_path, 'w') as f:
                json.dump(trades_data, f, indent=2)
        
        print(f"Results saved to: {output_dir}")
    
    def run_complete_pipeline(self, data_path: str) -> Dict:
        """Run the complete ML pipeline"""
        start_time = datetime.now()
        
        print("DipMaster Enhanced V4 - Standalone ML Pipeline")
        print("=" * 70)
        print(f"Start time: {start_time}")
        print(f"Target win rate: {self.config['targets']['win_rate']*100:.0f}%")
        print()
        
        try:
            # Phase 1: Data Preparation
            print("Phase 1: Data Preparation")
            X, y, price_data = self.load_and_prepare_data(data_path)
            
            # Phase 2: Feature Selection
            print("\nPhase 2: Feature Selection")
            selected_features = self.feature_selection(X, y)
            X_selected = X[selected_features]
            
            # Phase 3: Model Training
            print("\nPhase 3: Model Training")
            models = self.train_models(X_selected, y)
            
            # Phase 4: Model Evaluation
            print("\nPhase 4: Model Evaluation")
            evaluation_results = self.evaluate_models()
            
            # Phase 5: Signal Generation
            print("\nPhase 5: Signal Generation")
            signals = self.generate_signals(X_selected)
            
            # Phase 6: Backtesting
            print("\nPhase 6: Backtesting")
            backtest_results = self.run_backtest(signals, price_data)
            
            # Phase 7: Generate Reports
            print("\nPhase 7: Generating Reports")
            performance_report = self.generate_comprehensive_report()
            
            # Create output directory
            output_dir = f"results/standalone_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
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
            print("PIPELINE EXECUTION COMPLETED")
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
                
                if metrics.win_rate >= target_wr:
                    print(f"\n🎉 SUCCESS: {target_wr*100:.0f}%+ WIN RATE ACHIEVED!")
                else:
                    gap = (target_wr - metrics.win_rate) * 100
                    print(f"\n📈 PROGRESS: {gap:.1f}% gap remaining to reach {target_wr*100:.0f}% target")
            
            print(f"\nDetailed report saved to: {report_path}")
            print(performance_report)
            
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
    
    # Initialize and run pipeline
    pipeline = StandaloneDipMasterPipeline()
    results = pipeline.run_complete_pipeline(data_path)
    
    return results


if __name__ == "__main__":
    results = main()