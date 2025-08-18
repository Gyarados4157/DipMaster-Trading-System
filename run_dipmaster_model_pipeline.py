#!/usr/bin/env python3
"""
DipMaster Model Training and Backtesting Pipeline
Creates AlphaSignal and BacktestReport using basic ML models.
"""

import pandas as pd
import numpy as np
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Standard ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
import joblib

warnings.filterwarnings('ignore')


def engineer_dipmaster_features(df):
    """Engineer basic DipMaster features"""
    
    data = df.copy()
    
    # Basic price features
    data['returns'] = data['close'].pct_change()
    data['hl_spread'] = (data['high'] - data['low']) / data['close']
    data['oc_spread'] = (data['close'] - data['open']) / data['open']
    
    # Moving averages
    data['sma_5'] = data['close'].rolling(5).mean()
    data['sma_20'] = data['close'].rolling(20).mean()
    data['price_vs_sma_20'] = (data['close'] - data['sma_20']) / data['sma_20']
    
    # RSI approximation
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_window = 20
    data['bb_middle'] = data['close'].rolling(bb_window).mean()
    bb_std = data['close'].rolling(bb_window).std()
    data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
    data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
    data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    # Volume features
    data['volume_ma_5'] = data['volume'].rolling(5).mean()
    data['volume_ratio'] = data['volume'] / data['volume_ma_5']
    
    # Volatility
    data['volatility'] = data['returns'].rolling(20).std()
    
    # DipMaster specific features
    data['is_dip'] = (data['returns'] < -0.002).astype(int)
    data['rsi_dip_zone'] = ((data['rsi'] >= 25) & (data['rsi'] <= 50)).astype(int)
    data['volume_spike'] = (data['volume_ratio'] > 1.3).astype(int)
    
    # Signal strength
    data['signal_strength'] = (
        data['rsi_dip_zone'] * 0.4 +
        data['is_dip'] * 0.3 +
        data['volume_spike'] * 0.2 +
        (data['bb_position'] < 0.3).astype(int) * 0.1
    )
    
    # Time features
    if isinstance(data.index, pd.DatetimeIndex):
        data['hour'] = data.index.hour
        data['day_of_week'] = data.index.dayofweek
    
    # Labels - profitable in 15 minutes (3 periods)
    data['future_return'] = data['close'].shift(-3) / data['close'] - 1
    data['target'] = (data['future_return'] > 0.008).astype(int)  # 0.8% target
    
    return data.dropna()


def train_ensemble_models(X, y):
    """Train ensemble of models"""
    
    # Time-based split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Feature selection
    selector = SelectKBest(f_classif, k=min(15, len(X.columns)))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Scale for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Train models
    models = {
        'random_forest': RandomForestClassifier(n_estimators=50, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Use scaled data for logistic regression
        if name == 'logistic_regression':
            X_train_final, X_test_final = X_train_scaled, X_test_scaled
        else:
            X_train_final, X_test_final = X_train_selected, X_test_selected
        
        # Train
        model.fit(X_train_final, y_train)
        
        # Predict
        y_pred = model.predict_proba(X_test_final)[:, 1]
        
        # Evaluate
        auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred > 0.5)
        
        results[name] = {'auc': auc, 'accuracy': accuracy}
        trained_models[name] = {
            'model': model,
            'scaler': scaler if name == 'logistic_regression' else None
        }
        
        print(f"  {name} - AUC: {auc:.4f}, Accuracy: {accuracy:.1%}")
    
    # Ensemble predictions
    rf_pred = trained_models['random_forest']['model'].predict_proba(X_test_selected)[:, 1]
    gb_pred = trained_models['gradient_boosting']['model'].predict_proba(X_test_selected)[:, 1]
    lr_pred = trained_models['logistic_regression']['model'].predict_proba(X_test_scaled)[:, 1]
    
    ensemble_pred = (rf_pred * 0.4 + gb_pred * 0.4 + lr_pred * 0.2)
    ensemble_auc = roc_auc_score(y_test, ensemble_pred)
    
    print(f"Ensemble AUC: {ensemble_auc:.4f}")
    
    return {
        'models': trained_models,
        'selector': selector,
        'scaler': scaler,
        'results': results,
        'test_data': {'X': X_test, 'y': y_test, 'predictions': ensemble_pred}
    }


def run_backtest(signals_df, market_data):
    """Simple backtest"""
    
    capital = 10000
    positions = []
    equity_curve = [capital]
    
    for _, signal in signals_df.iterrows():
        timestamp = signal['timestamp']
        confidence = signal['confidence']
        
        if confidence > 0.5:
            # Find market data
            try:
                market_row = market_data.loc[market_data.index >= timestamp].iloc[0]
                entry_price = market_row['close']
                
                # Position sizing
                position_size = min(1000, capital * 0.2)
                
                if position_size >= 100:
                    # Exit after 15 minutes
                    exit_time = timestamp + timedelta(minutes=15)
                    exit_data = market_data.loc[market_data.index >= exit_time]
                    
                    if len(exit_data) > 0:
                        exit_price = exit_data.iloc[0]['close']
                        returns = (exit_price - entry_price) / entry_price
                        pnl = position_size * returns
                        
                        # Trading costs
                        pnl -= position_size * 0.0004
                        
                        capital += pnl
                        equity_curve.append(capital)
                        
                        positions.append({
                            'entry_time': timestamp,
                            'returns': returns,
                            'pnl': pnl,
                            'confidence': confidence
                        })
            except:
                continue
    
    # Calculate metrics
    if positions:
        trade_returns = [p['returns'] for p in positions]
        winning_trades = [r for r in trade_returns if r > 0]
        
        win_rate = len(winning_trades) / len(trade_returns)
        total_return = (capital - 10000) / 10000
        
        if len(trade_returns) > 1:
            sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.cummax()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Profit factor
        wins = sum(p['pnl'] for p in positions if p['pnl'] > 0)
        losses = abs(sum(p['pnl'] for p in positions if p['pnl'] < 0))
        profit_factor = wins / losses if losses > 0 else float('inf')
    else:
        win_rate = 0
        total_return = 0
        sharpe_ratio = 0
        max_drawdown = 0
        profit_factor = 0
    
    return {
        'total_return': total_return,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'total_trades': len(positions),
        'final_capital': capital
    }


def main():
    """Run complete pipeline"""
    
    print("\n" + "="*50)
    print("DipMaster Model Training & Backtesting Pipeline")
    print("="*50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load data
    data_file = "data/market_data/SOLUSDT_5m_2years.csv"
    
    if not Path(data_file).exists():
        print(f"Data file not found: {data_file}")
        available_files = list(Path("data/market_data").glob("*.csv"))
        if available_files:
            print("Available files:")
            for f in available_files[:5]:
                print(f"  {f}")
            data_file = str(available_files[0])
            print(f"Using: {data_file}")
        else:
            print("No data files found!")
            return
    
    print(f"Loading data from: {data_file}")
    
    # Load and process data
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    df = df[['open', 'high', 'low', 'close', 'volume']].dropna().sort_index()
    
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    
    # Feature engineering
    print("\nEngineering features...")
    processed_data = engineer_dipmaster_features(df)
    print(f"Processed data: {processed_data.shape}")
    
    # Prepare ML data
    feature_cols = [col for col in processed_data.columns 
                   if col not in ['future_return', 'target'] and 
                   not col.startswith('bb_') and len(col) < 20]
    
    X = processed_data[feature_cols]
    y = processed_data['target']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Train models
    print("\nTraining models...")
    training_results = train_ensemble_models(X, y)
    
    # Generate signals
    print("\nGenerating signals...")
    test_predictions = training_results['test_data']['predictions']
    test_timestamps = training_results['test_data']['X'].index
    
    signals_df = pd.DataFrame({
        'timestamp': test_timestamps,
        'signal': test_predictions,
        'confidence': test_predictions,
        'predicted_return': test_predictions * 0.01
    })
    
    # Filter by confidence
    signals_df = signals_df[signals_df['confidence'] >= 0.6]
    print(f"Generated {len(signals_df)} high-confidence signals")
    
    # Run backtest
    print("\nRunning backtest...")
    backtest_results = run_backtest(signals_df, processed_data)
    
    # Display results
    print("\n" + "="*30)
    print("RESULTS SUMMARY")
    print("="*30)
    print(f"Total Return: {backtest_results['total_return']:.1%}")
    print(f"Win Rate: {backtest_results['win_rate']:.1%}")
    print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {backtest_results['max_drawdown']:.1%}")
    print(f"Profit Factor: {backtest_results['profit_factor']:.2f}")
    print(f"Total Trades: {backtest_results['total_trades']}")
    print(f"Final Capital: ${backtest_results['final_capital']:.2f}")
    
    # Performance targets
    targets = {
        'win_rate_achieved': backtest_results['win_rate'] >= 0.70,
        'sharpe_achieved': backtest_results['sharpe_ratio'] >= 1.0,
        'drawdown_ok': abs(backtest_results['max_drawdown']) <= 0.10,
        'return_positive': backtest_results['total_return'] > 0
    }
    
    all_targets = all(targets.values())
    print(f"\nTargets Achieved: {all_targets}")
    for target, achieved in targets.items():
        status = "✓" if achieved else "✗"
        print(f"  {status} {target}: {achieved}")
    
    # Save results
    output_dir = Path("results/basic_ml_pipeline")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save AlphaSignal
    alpha_signal = {
        "signal_uri": f"results/basic_ml_pipeline/signals_{timestamp}.csv",
        "schema": ["timestamp", "signal", "confidence", "predicted_return"],
        "model_version": "DipMaster_Basic_v1.0.0",
        "validation_metrics": {
            name: result for name, result in training_results['results'].items()
        },
        "backtest_performance": backtest_results,
        "retrain_policy": "weekly",
        "generation_timestamp": datetime.now().isoformat(),
        "targets_achieved": all_targets,
        "production_ready": all_targets
    }
    
    alpha_file = output_dir / f"AlphaSignal_{timestamp}.json"
    with open(alpha_file, 'w') as f:
        json.dump(alpha_signal, f, indent=2, default=str)
    
    # Save signals
    signals_file = output_dir / f"signals_{timestamp}.csv"
    signals_df.to_csv(signals_file, index=False)
    
    # Save models
    models_file = output_dir / f"models_{timestamp}.joblib"
    joblib.dump(training_results, models_file)
    
    # Create backtest report
    backtest_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DipMaster Backtest Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                  border: 1px solid #ddd; border-radius: 5px; }}
        .positive {{ color: green; }}
        .negative {{ color: red; }}
    </style>
</head>
<body>
    <h1>DipMaster Strategy Backtest Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Performance Summary</h2>
    <div class="metric">
        <strong>Total Return:</strong> 
        <span class="{'positive' if backtest_results['total_return'] > 0 else 'negative'}">
            {backtest_results['total_return']:.1%}
        </span>
    </div>
    <div class="metric">
        <strong>Win Rate:</strong> {backtest_results['win_rate']:.1%}
    </div>
    <div class="metric">
        <strong>Sharpe Ratio:</strong> {backtest_results['sharpe_ratio']:.2f}
    </div>
    <div class="metric">
        <strong>Max Drawdown:</strong> 
        <span class="negative">{backtest_results['max_drawdown']:.1%}</span>
    </div>
    <div class="metric">
        <strong>Profit Factor:</strong> {backtest_results['profit_factor']:.2f}
    </div>
    <div class="metric">
        <strong>Total Trades:</strong> {backtest_results['total_trades']}
    </div>
    
    <h2>Model Performance</h2>
    <ul>
"""
    
    for name, result in training_results['results'].items():
        backtest_report += f"<li><strong>{name}:</strong> AUC {result['auc']:.4f}, Accuracy {result['accuracy']:.1%}</li>"
    
    backtest_report += f"""
    </ul>
    
    <h2>Strategy Assessment</h2>
    <p><strong>Targets Achieved:</strong> {'Yes' if all_targets else 'No'}</p>
    <p><strong>Production Ready:</strong> {'Yes' if all_targets else 'Needs Improvement'}</p>
    
    <h2>Data Summary</h2>
    <p><strong>Total Samples:</strong> {len(processed_data):,}</p>
    <p><strong>Features:</strong> {len(feature_cols)}</p>
    <p><strong>Signals Generated:</strong> {len(signals_df)}</p>
    <p><strong>Date Range:</strong> {df.index.min()} to {df.index.max()}</p>
</body>
</html>
"""
    
    report_file = output_dir / f"BacktestReport_{timestamp}.html"
    with open(report_file, 'w') as f:
        f.write(backtest_report)
    
    print(f"\n" + "="*30)
    print("FILES GENERATED")
    print("="*30)
    print(f"AlphaSignal: {alpha_file}")
    print(f"Signals CSV: {signals_file}")
    print(f"Models: {models_file}")
    print(f"Backtest Report: {report_file}")
    
    print(f"\nPipeline completed {'successfully' if all_targets else 'with mixed results'}!")
    
    return {
        'status': 'SUCCESS' if all_targets else 'PARTIAL',
        'alpha_signal_path': str(alpha_file),
        'backtest_report_path': str(report_file),
        'performance': backtest_results,
        'targets_achieved': all_targets
    }


if __name__ == "__main__":
    main()