"""
Simplified DipMaster V4 ML Pipeline - Quick execution version
"""

import numpy as np
import pandas as pd
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

# ML libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

def run_ml_pipeline():
    """Run simplified ML pipeline"""
    
    print("=== DipMaster Enhanced V4 ML Pipeline ===")
    
    # 1. Load data
    print("Loading data...")
    data_path = "G:/Github/Quant/DipMaster-Trading-System/data/dipmaster_v4_features_20250816_175605.parquet"
    df = pd.read_parquet(data_path)
    
    print(f"Loaded {len(df)} samples with {df.shape[1]} columns")
    
    # 2. Prepare features and labels
    print("Preparing features and labels...")
    
    # Define feature columns
    feature_cols = [
        'rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_squeeze',
        'volume_ma', 'volume_ratio', 'volume_spike',
        'volatility_10', 'volatility_20', 'volatility_50',
        'momentum_5', 'momentum_10', 'momentum_20',
        'ma_15', 'ma_60', 'trend_short', 'trend_long', 'trend_alignment',
        'order_flow_imbalance', 'dipmaster_signal_strength',
        'open', 'high', 'low', 'close', 'volume'
    ]
    
    # Filter available features
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features].copy()
    
    # Create labels
    if 'is_profitable_15m' in df.columns:
        y = df['is_profitable_15m'].astype(int)
    elif 'future_return_15m' in df.columns:
        y = (df['future_return_15m'] > 0.006).astype(int)  # 0.6% profit target
    else:
        raise ValueError("No suitable label column found")
    
    # Handle missing values
    X = X.fillna(method='ffill').fillna(method='bfill')
    valid_mask = ~y.isna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"Using {len(available_features)} features")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # 3. Time-based split
    print("Creating time-based splits...")
    n_samples = len(X)
    train_end = int(n_samples * 0.6)
    val_end = int(n_samples * 0.8)
    
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    
    print(f"Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # 4. Feature selection
    print("Selecting top features...")
    selector = SelectKBest(score_func=mutual_info_classif, k=20)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X.columns[selector.get_support()].tolist()
    print(f"Selected features: {selected_features}")
    
    # 5. Train models
    print("Training models...")
    
    models = {}
    results = {}
    
    # LightGBM
    print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        learning_rate=0.1,
        num_leaves=31,
        random_state=42,
        verbose=-1
    )
    lgb_model.fit(X_train_selected, y_train)
    models['lgbm'] = lgb_model
    
    # XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    xgb_model.fit(X_train_selected, y_train)
    models['xgb'] = xgb_model
    
    # Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_selected, y_train)
    models['rf'] = rf_model
    
    # 6. Validate models
    print("Validating models...")
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_val_selected)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        auc = roc_auc_score(y_val, y_pred_proba)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        
        results[name] = {
            'auc': auc,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
        
        print(f"{name}: AUC={auc:.4f}, Accuracy={accuracy:.4f}, Precision={precision:.4f}")
    
    # 7. Ensemble prediction
    print("Creating ensemble...")
    ensemble_pred = np.mean([
        models['lgbm'].predict_proba(X_test_selected)[:, 1],
        models['xgb'].predict_proba(X_test_selected)[:, 1],
        models['rf'].predict_proba(X_test_selected)[:, 1]
    ], axis=0)
    
    # 8. Generate signals
    print("Generating trading signals...")
    signals_df = pd.DataFrame({
        'timestamp': X_test.index,
        'score': ensemble_pred,
        'confidence': np.abs(ensemble_pred - 0.5) * 2,
        'predicted_return': ensemble_pred * 0.015,
        'signal': (ensemble_pred > 0.6).astype(int)
    })
    
    # Filter high confidence signals
    high_conf_signals = signals_df[signals_df['confidence'] >= 0.2]
    
    print(f"Generated {len(signals_df)} total signals")
    print(f"High confidence signals: {len(high_conf_signals)}")
    print(f"Signal distribution: {signals_df['signal'].value_counts().to_dict()}")
    
    # 9. Simple backtest
    print("Running simple backtest...")
    
    # Get market data for backtest
    market_data = X_test[['open', 'high', 'low', 'close', 'volume']].copy()
    
    # Simple backtest simulation
    trades = []
    capital = 10000
    position_size = 1000
    
    for _, signal in high_conf_signals.iterrows():
        if signal['signal'] == 1:
            timestamp = signal['timestamp']
            
            # Get entry price
            try:
                entry_price = market_data.loc[timestamp, 'close']
                
                # Simple exit after 15 minutes (3 bars at 5-minute intervals)
                exit_time = timestamp + timedelta(minutes=15)
                
                # Find nearest exit price
                exit_prices = market_data[market_data.index >= exit_time]
                if len(exit_prices) > 0:
                    exit_price = exit_prices.iloc[0]['close']
                    
                    # Calculate P&L
                    quantity = position_size / entry_price
                    gross_pnl = quantity * (exit_price - entry_price)
                    commission = position_size * 0.0004  # 0.04% total commission
                    net_pnl = gross_pnl - commission
                    
                    trades.append({
                        'entry_time': timestamp,
                        'exit_time': exit_prices.index[0],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'quantity': quantity,
                        'gross_pnl': gross_pnl,
                        'net_pnl': net_pnl,
                        'return_pct': net_pnl / position_size
                    })
                    
            except (KeyError, IndexError):
                continue
    
    # Calculate backtest results
    if trades:
        trades_df = pd.DataFrame(trades)
        
        total_return = trades_df['net_pnl'].sum() / capital
        win_rate = (trades_df['net_pnl'] > 0).mean()
        avg_return = trades_df['return_pct'].mean()
        std_return = trades_df['return_pct'].std()
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
        losing_trades = len(trades_df[trades_df['net_pnl'] <= 0])
        
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl'] <= 0]['net_pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 else 0
        
        # Performance assessment
        performance_targets = {
            'win_rate_target': 0.85,
            'sharpe_target': 2.0,
            'profit_factor_target': 1.8
        }
        
        target_achievement = {
            'win_rate_achieved': win_rate >= performance_targets['win_rate_target'],
            'sharpe_achieved': sharpe_ratio >= performance_targets['sharpe_target'],
            'profit_factor_achieved': profit_factor >= performance_targets['profit_factor_target']
        }
        
        all_targets_met = all(target_achievement.values())
        
        # Create output directory
        output_dir = Path("G:/Github/Quant/DipMaster-Trading-System/results/ml_pipeline")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save alpha signals
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        alpha_signal = {
            "signal_uri": str(output_dir / f"AlphaSignal_DipMasterV4_{timestamp}.parquet"),
            "schema": ["timestamp", "symbol", "score", "confidence", "predicted_return"],
            "model_version": "DipMaster_Enhanced_V4_1.0.0",
            "retrain_policy": "weekly",
            "feature_importance": dict(zip(selected_features, models['lgbm'].feature_importances_)),
            "validation_metrics": results,
            "generation_metadata": {
                "generated_timestamp": datetime.now().isoformat(),
                "total_signals": len(signals_df),
                "confident_signals": len(high_conf_signals),
                "signal_period": {
                    "start": signals_df['timestamp'].min().isoformat(),
                    "end": signals_df['timestamp'].max().isoformat()
                }
            }
        }
        
        # Save files
        with open(output_dir / f"AlphaSignal_DipMasterV4_{timestamp}.json", 'w') as f:
            json.dump(alpha_signal, f, indent=2, default=str)
        
        signals_df['symbol'] = 'BTCUSDT'
        signals_df.to_parquet(output_dir / f"AlphaSignal_DipMasterV4_{timestamp}.parquet")
        
        # Final summary
        summary = {
            'pipeline_status': 'COMPLETED',
            'targets_achieved': all_targets_met,
            'target_details': target_achievement,
            'performance_metrics': {
                'total_return': total_return,
                'win_rate': win_rate,
                'sharpe_ratio': sharpe_ratio,
                'profit_factor': profit_factor,
                'total_trades': len(trades_df),
                'avg_return_per_trade': avg_return
            },
            'model_performance': results,
            'feature_importance': dict(zip(selected_features, models['lgbm'].feature_importances_)),
            'output_files': {
                'alpha_signals': str(output_dir / f"AlphaSignal_DipMasterV4_{timestamp}.json")
            }
        }
        
        # Save summary
        with open(output_dir / f"pipeline_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print results
        print("\n=== BACKTEST RESULTS ===")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Average Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        
        print("\n=== PERFORMANCE TARGET ACHIEVEMENT ===")
        print(f"All Targets Met: {all_targets_met}")
        print(f"Win Rate Target (≥85%): {'✅ ACHIEVED' if target_achievement['win_rate_achieved'] else '❌ NOT ACHIEVED'} ({win_rate:.1%})")
        print(f"Sharpe Ratio Target (≥2.0): {'✅ ACHIEVED' if target_achievement['sharpe_achieved'] else '❌ NOT ACHIEVED'} ({sharpe_ratio:.2f})")
        print(f"Profit Factor Target (≥1.8): {'✅ ACHIEVED' if target_achievement['profit_factor_achieved'] else '❌ NOT ACHIEVED'} ({profit_factor:.2f})")
        
        print(f"\nResults saved to: {output_dir}")
        
        return summary
        
    else:
        print("No trades executed in backtest")
        return None

if __name__ == "__main__":
    results = run_ml_pipeline()