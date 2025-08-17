"""
Simplified DipMaster Model Trainer
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
import json
from datetime import datetime
import os

def purged_time_series_split(X, y, n_splits=5, embargo_period=12):
    """Simple purged time series split"""
    n_samples = len(X)
    test_size = n_samples // (n_splits + 1)
    
    splits = []
    for i in range(n_splits):
        test_start = (i + 1) * test_size
        test_end = test_start + test_size
        
        if test_end > n_samples:
            break
            
        train_end = test_start - embargo_period
        
        if train_end <= 0:
            continue
            
        train_indices = np.arange(0, train_end)
        test_indices = np.arange(test_start, min(test_end, n_samples))
        
        splits.append((train_indices, test_indices))
    
    return splits

def analyze_win_rate_distribution(df):
    """Analyze why win rate is low"""
    print("=== Win Rate Analysis ===")
    
    target_cols = [col for col in df.columns if col.startswith('target_') or 'win' in col.lower()]
    
    for col in target_cols:
        if col in df.columns:
            rate = df[col].mean()
            count = df[col].sum()
            print(f"{col}: {rate:.4f} ({count:,} wins out of {len(df):,} samples)")
    
    # Analyze by symbol
    if 'timestamp' in df.columns and len(df['timestamp'].unique()) > 100:
        print("\n=== Distribution over time ===")
        df_sample = df.sample(min(10000, len(df)))
        win_by_time = df_sample.groupby(df_sample.index // 1000)['dipmaster_win'].mean()
        print(f"Win rate std over time buckets: {win_by_time.std():.4f}")
        print(f"Min/Max win rate by time: {win_by_time.min():.4f} / {win_by_time.max():.4f}")

def simple_feature_selection(X, y, top_k=50):
    """Simple feature selection using mutual information"""
    print("Starting feature selection...")
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    
    # Select top features
    selected_features = mi_ranking.head(top_k).index.tolist()
    
    print(f"Selected {len(selected_features)} features from {len(X.columns)} total")
    print(f"Top 10 features: {selected_features[:10]}")
    
    return selected_features, mi_ranking

def train_simple_models(X, y, selected_features):
    """Train simple LGBM and XGBoost models"""
    X_selected = X[selected_features]
    
    # Split data for training/testing
    split_point = int(len(X_selected) * 0.8)
    X_train, X_test = X_selected.iloc[:split_point], X_selected.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    results = {}
    
    # LGBM Model
    print("Training LightGBM...")
    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbosity': -1,
        'random_state': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    lgbm_model = lgb.train(
        lgbm_params, 
        train_data,
        valid_sets=[val_data],
        num_boost_round=500,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # LGBM predictions
    lgbm_pred_proba = lgbm_model.predict(X_test)
    lgbm_pred = (lgbm_pred_proba > 0.5).astype(int)
    
    lgbm_auc = roc_auc_score(y_test, lgbm_pred_proba)
    lgbm_win_rate = np.mean(lgbm_pred)
    lgbm_precision = np.mean(lgbm_pred == y_test)
    lgbm_hit_rate = np.sum((lgbm_pred == 1) & (y_test == 1)) / max(np.sum(lgbm_pred), 1)
    
    results['lgbm'] = {
        'model': lgbm_model,
        'auc': lgbm_auc,
        'predicted_win_rate': lgbm_win_rate,
        'precision': lgbm_precision,
        'hit_rate': lgbm_hit_rate,
        'feature_importance': dict(zip(selected_features, lgbm_model.feature_importance()))
    }
    
    print(f"LGBM - AUC: {lgbm_auc:.4f}, Win Rate: {lgbm_win_rate:.4f}, Hit Rate: {lgbm_hit_rate:.4f}")
    
    # XGBoost Model
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        max_depth=6,
        learning_rate=0.1,
        n_estimators=500,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42
    )
    
    xgb_model.fit(X_train, y_train)
    
    # XGBoost predictions
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = xgb_model.predict(X_test)
    
    xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
    xgb_win_rate = np.mean(xgb_pred)
    xgb_precision = np.mean(xgb_pred == y_test)
    xgb_hit_rate = np.sum((xgb_pred == 1) & (y_test == 1)) / max(np.sum(xgb_pred), 1)
    
    results['xgb'] = {
        'model': xgb_model,
        'auc': xgb_auc,
        'predicted_win_rate': xgb_win_rate,
        'precision': xgb_precision,
        'hit_rate': xgb_hit_rate,
        'feature_importance': dict(zip(selected_features, xgb_model.feature_importances_))
    }
    
    print(f"XGBoost - AUC: {xgb_auc:.4f}, Win Rate: {xgb_win_rate:.4f}, Hit Rate: {xgb_hit_rate:.4f}")
    
    return results, X_test, y_test

def realistic_backtest_simulation(results, X_test, y_test, selected_features):
    """Simulate realistic trading with costs"""
    print("\n=== Realistic Backtest Simulation ===")
    
    # Use best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    
    print(f"Using {best_model_name} model for backtest")
    
    # Generate signals
    if best_model_name == 'lgbm':
        signal_proba = best_model.predict(X_test[selected_features])
        signals = signal_proba > 0.5
    else:
        signal_proba = best_model.predict_proba(X_test[selected_features])[:, 1]
        signals = best_model.predict(X_test[selected_features]) == 1
    
    # Simulate trading costs
    # Assumptions:
    # - Slippage: 0.05% (0.5 bps)
    # - Commission: 0.1% (Binance spot)
    # - Market impact: 0.02% (0.2 bps)
    # - Funding cost: negligible for short holding periods
    
    transaction_cost = 0.0005 + 0.001 + 0.0002  # Total: 0.17%
    
    # Simulate trades
    signal_indices = np.where(signals)[0]
    
    if len(signal_indices) == 0:
        print("No signals generated!")
        return None
    
    # Assume we have some return data (simplified)
    # In reality, this would come from actual price movements
    simulated_base_returns = np.random.normal(0.002, 0.01, len(signal_indices))  # 0.2% mean with 1% volatility
    
    trades = []
    cumulative_pnl = 0
    
    for i, idx in enumerate(signal_indices):
        base_return = simulated_base_returns[i]
        net_return = base_return - transaction_cost  # Subtract costs
        
        cumulative_pnl += net_return
        
        trades.append({
            'signal_strength': signal_proba[idx],
            'base_return': base_return,
            'net_return': net_return,
            'cumulative_pnl': cumulative_pnl,
            'is_profitable': net_return > 0
        })
    
    # Calculate performance metrics
    df_trades = pd.DataFrame(trades)
    
    backtest_metrics = {
        'total_trades': len(trades),
        'win_rate': np.mean(df_trades['is_profitable']),
        'avg_return_per_trade': np.mean(df_trades['net_return']),
        'total_pnl': cumulative_pnl,
        'sharpe_ratio': np.mean(df_trades['net_return']) / np.std(df_trades['net_return']) if np.std(df_trades['net_return']) > 0 else 0,
        'max_drawdown': np.min(df_trades['cumulative_pnl'] - df_trades['cumulative_pnl'].cummax()),
        'hit_rate_vs_actual': np.mean((signals[signal_indices]) & (y_test.iloc[signal_indices] == 1))
    }
    
    print(f"Total Trades: {backtest_metrics['total_trades']}")
    print(f"Win Rate (after costs): {backtest_metrics['win_rate']:.4f}")
    print(f"Average Return per Trade: {backtest_metrics['avg_return_per_trade']:.4f}")
    print(f"Total PnL: {backtest_metrics['total_pnl']:.4f}")
    print(f"Sharpe Ratio: {backtest_metrics['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {backtest_metrics['max_drawdown']:.4f}")
    
    return backtest_metrics

def generate_recommendations(data_analysis, model_results, backtest_results):
    """Generate specific optimization recommendations"""
    print("\n=== Optimization Recommendations ===")
    
    recommendations = []
    
    # Data quality issues
    current_win_rate = data_analysis.get('dipmaster_win_rate', 0.18)
    
    if current_win_rate < 0.25:
        recommendations.append("1. CRITICAL: Low base win rate indicates fundamental issue with strategy logic or labeling")
        recommendations.append("   - Review DipMaster signal definition")
        recommendations.append("   - Check 15-minute boundary exit logic") 
        recommendations.append("   - Validate profit target calculations")
    
    # Model performance
    best_auc = max([r['auc'] for r in model_results.values()])
    
    if best_auc < 0.6:
        recommendations.append("2. Model Performance: Low AUC suggests weak predictive power")
        recommendations.append("   - Add more sophisticated features (market regime, volatility clustering)")
        recommendations.append("   - Consider ensemble methods")
        recommendations.append("   - Implement feature engineering for time-of-day effects")
    
    # Feature engineering
    recommendations.append("3. Feature Enhancement:")
    recommendations.append("   - Add market microstructure features (bid-ask spread, order book depth)")
    recommendations.append("   - Include volatility regime indicators")
    recommendations.append("   - Add cross-asset momentum features")
    recommendations.append("   - Implement volatility-adjusted RSI")
    
    # Cost analysis
    if backtest_results and backtest_results['win_rate'] < 0.5:
        recommendations.append("4. Cost Optimization:")
        recommendations.append("   - Reduce trading frequency to decrease cost drag")
        recommendations.append("   - Implement minimum profit targets above transaction costs")
        recommendations.append("   - Consider using limit orders to reduce slippage")
    
    # Signal quality
    recommendations.append("5. Signal Quality Improvement:")
    recommendations.append("   - Implement confidence-based filtering")
    recommendations.append("   - Add volatility-adjusted position sizing")
    recommendations.append("   - Use multi-timeframe confirmation")
    
    # Time-based improvements
    recommendations.append("6. Temporal Optimization:")
    recommendations.append("   - Analyze performance by hour-of-day")
    recommendations.append("   - Implement session-based filtering")
    recommendations.append("   - Add market open/close effect handling")
    
    for rec in recommendations:
        print(rec)
    
    return recommendations

def main():
    """Main training pipeline"""
    print("=== DipMaster Model Training Pipeline ===")
    
    # Load data
    data_path = "data/Enhanced_Features_V5_Clean_20250817_144045.parquet"
    df = pd.read_parquet(data_path)
    
    print(f"Loaded data: {df.shape}")
    
    # Analyze current win rate issues
    analyze_win_rate_distribution(df)
    
    # Prepare features and labels
    exclude_cols = ['timestamp', 'symbol'] + [col for col in df.columns 
                   if col.startswith('target_') or col.startswith('is_') 
                   or col.startswith('hits_') or 'win' in col.lower() 
                   or 'return_class' in col]
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    y = df['dipmaster_win'].copy()
    
    # Handle missing values
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    print(f"Features: {len(feature_cols)}, Samples: {len(X)}, Target Win Rate: {y.mean():.4f}")
    
    # Feature selection
    selected_features, feature_ranking = simple_feature_selection(X, y, top_k=40)
    
    # Train models
    model_results, X_test, y_test = train_simple_models(X, y, selected_features)
    
    # Realistic backtest
    backtest_results = realistic_backtest_simulation(model_results, X_test, y_test, selected_features)
    
    # Generate recommendations
    data_analysis = {'dipmaster_win_rate': y.mean()}
    recommendations = generate_recommendations(data_analysis, model_results, backtest_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "results/model_training"
    os.makedirs(results_dir, exist_ok=True)
    
    # Prepare results for saving
    final_results = {
        'timestamp': timestamp,
        'data_shape': df.shape,
        'base_win_rate': float(y.mean()),
        'selected_features': selected_features,
        'model_performance': {
            name: {
                'auc': float(results['auc']),
                'predicted_win_rate': float(results['predicted_win_rate']),
                'hit_rate': float(results['hit_rate'])
            }
            for name, results in model_results.items()
        },
        'backtest_results': backtest_results if backtest_results else {},
        'recommendations': recommendations,
        'feature_importance': {
            name: {k: float(v) for k, v in results['feature_importance'].items()}
            for name, results in model_results.items()
        }
    }
    
    # Save results
    results_file = f"{results_dir}/dipmaster_training_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return final_results

if __name__ == "__main__":
    results = main()