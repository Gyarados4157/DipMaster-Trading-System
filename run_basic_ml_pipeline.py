#!/usr/bin/env python3
"""
Basic ML Pipeline for DipMaster Strategy
Uses only scikit-learn and standard libraries for maximum compatibility.
"""

import sys
import os
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score, 
                           recall_score, f1_score, classification_report)
from sklearn.base import clone
import joblib

warnings.filterwarnings('ignore')

class BasicDipMasterFeatureEngineer:
    """Basic feature engineering using only pandas and numpy"""
    
    def __init__(self):
        self.logger = logging.getLogger('BasicFeatureEngineer')
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer basic technical features"""
        
        self.logger.info("Engineering basic features")
        data = df.copy()
        
        # Basic price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        data['hl_spread'] = (data['high'] - data['low']) / data['close']
        data['oc_spread'] = (data['close'] - data['open']) / data['open']
        
        # Simple technical indicators
        # Moving averages
        for window in [5, 10, 20, 50]:
            data[f'sma_{window}'] = data['close'].rolling(window).mean()
            data[f'price_vs_sma_{window}'] = (data['close'] - data[f'sma_{window}']) / data[f'sma_{window}']
        
        # RSI approximation
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands approximation
        bb_window = 20
        data['bb_middle'] = data['close'].rolling(bb_window).mean()
        bb_std = data['close'].rolling(bb_window).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volume features
        for window in [5, 10, 20]:
            data[f'volume_ma_{window}'] = data['volume'].rolling(window).mean()
            data[f'volume_ratio_{window}'] = data['volume'] / data[f'volume_ma_{window}']
        
        # Volatility
        for window in [10, 20, 50]:
            data[f'volatility_{window}'] = data['returns'].rolling(window).std() * np.sqrt(288)
        
        # Momentum
        for window in [5, 10, 20]:
            data[f'momentum_{window}'] = data['close'] / data['close'].shift(window) - 1
        
        # DipMaster specific features
        data['is_dip'] = (data['returns'] < -0.002).astype(int)
        data['rsi_dip_zone'] = ((data['rsi'] >= 25) & (data['rsi'] <= 50)).astype(int)
        data['volume_spike'] = (data['volume_ratio_5'] > 1.3).astype(int)
        
        # Combined signal
        data['dipmaster_signal_strength'] = (
            data['rsi_dip_zone'] * 0.4 +
            data['is_dip'] * 0.3 +
            data['volume_spike'] * 0.2 +
            (data['bb_position'] < 0.3).astype(int) * 0.1
        )
        
        # Time features
        if isinstance(data.index, pd.DatetimeIndex):
            data['hour'] = data.index.hour
            data['day_of_week'] = data.index.dayofweek
            data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
        
        # Labels - future returns
        data['future_return_15m'] = data['close'].shift(-3) / data['close'] - 1  # 3 periods = 15 min
        data['profitable_15m'] = (data['future_return_15m'] > 0.008).astype(int)  # 0.8% target
        
        # Primary label
        data['dipmaster_primary_label'] = data['profitable_15m']
        
        # Drop NaN
        data = data.dropna()
        
        self.logger.info(f"Feature engineering complete: {data.shape}")
        return data

class BasicBacktester:
    """Basic backtesting engine"""
    
    def __init__(self, initial_capital=10000):\n        self.initial_capital = initial_capital\n        self.logger = logging.getLogger('BasicBacktester')\n    \n    def run_backtest(self, signals_df: pd.DataFrame, market_data: pd.DataFrame) -> dict:\n        \"\"\"Run basic backtest\"\"\"\n        \n        self.logger.info(\"Running basic backtest\")\n        \n        capital = self.initial_capital\n        positions = []\n        equity_curve = [capital]\n        \n        # Simple backtest logic\n        for idx, signal_row in signals_df.iterrows():\n            timestamp = signal_row['timestamp']\n            signal = signal_row['signal']\n            confidence = signal_row['confidence']\n            \n            if signal > 0.5:  # Buy signal\n                # Find corresponding market data\n                market_row = market_data.loc[market_data.index >= timestamp].iloc[0] if len(market_data.loc[market_data.index >= timestamp]) > 0 else None\n                \n                if market_row is not None:\n                    entry_price = market_row['close']\n                    position_size = min(1000, capital * 0.2)  # Max 20% per position\n                    \n                    if position_size >= 100:  # Minimum position size\n                        # Hold for 15 minutes (3 periods)\n                        exit_time = timestamp + timedelta(minutes=15)\n                        exit_data = market_data.loc[market_data.index >= exit_time]\n                        \n                        if len(exit_data) > 0:\n                            exit_price = exit_data.iloc[0]['close']\n                            \n                            # Calculate P&L\n                            returns = (exit_price - entry_price) / entry_price\n                            pnl = position_size * returns\n                            \n                            # Trading costs (0.04% total)\n                            costs = position_size * 0.0004\n                            pnl -= costs\n                            \n                            capital += pnl\n                            equity_curve.append(capital)\n                            \n                            positions.append({\n                                'entry_time': timestamp,\n                                'exit_time': exit_time,\n                                'entry_price': entry_price,\n                                'exit_price': exit_price,\n                                'returns': returns,\n                                'pnl': pnl,\n                                'confidence': confidence\n                            })\n        \n        # Calculate metrics\n        if positions:\n            trade_returns = [pos['returns'] for pos in positions]\n            winning_trades = [r for r in trade_returns if r > 0]\n            \n            total_return = (capital - self.initial_capital) / self.initial_capital\n            win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0\n            \n            # Sharpe ratio approximation\n            if len(trade_returns) > 1:\n                sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252)\n            else:\n                sharpe_ratio = 0\n            \n            # Max drawdown\n            equity_series = pd.Series(equity_curve)\n            rolling_max = equity_series.cummax()\n            drawdowns = (equity_series - rolling_max) / rolling_max\n            max_drawdown = drawdowns.min()\n            \n            # Profit factor\n            total_wins = sum(pos['pnl'] for pos in positions if pos['pnl'] > 0)\n            total_losses = abs(sum(pos['pnl'] for pos in positions if pos['pnl'] < 0))\n            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')\n            \n        else:\n            total_return = 0\n            win_rate = 0\n            sharpe_ratio = 0\n            max_drawdown = 0\n            profit_factor = 0\n        \n        results = {\n            'total_return': total_return,\n            'win_rate': win_rate,\n            'sharpe_ratio': sharpe_ratio,\n            'max_drawdown': max_drawdown,\n            'profit_factor': profit_factor,\n            'total_trades': len(positions),\n            'final_capital': capital,\n            'equity_curve': equity_curve,\n            'positions': positions\n        }\n        \n        self.logger.info(f\"Backtest complete: {len(positions)} trades\")\n        return results\n\nclass BasicMLPipeline:\n    \"\"\"Basic ML pipeline using scikit-learn\"\"\"\n    \n    def __init__(self, output_dir=\"results/basic_ml_pipeline\"):\n        self.output_dir = Path(output_dir)\n        self.output_dir.mkdir(parents=True, exist_ok=True)\n        \n        self.feature_engineer = BasicDipMasterFeatureEngineer()\n        self.backtester = BasicBacktester()\n        \n        # Setup logging\n        self.logger = self._setup_logging()\n        \n        # Results storage\n        self.processed_data = None\n        self.trained_models = {}\n        self.validation_results = {}\n    \n    def _setup_logging(self):\n        logger = logging.getLogger('BasicMLPipeline')\n        logger.setLevel(logging.INFO)\n        \n        if not logger.handlers:\n            # Console handler\n            console_handler = logging.StreamHandler()\n            console_formatter = logging.Formatter(\n                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'\n            )\n            console_handler.setFormatter(console_formatter)\n            logger.addHandler(console_handler)\n        \n        return logger\n    \n    def load_and_process_data(self, data_path: str) -> pd.DataFrame:\n        \"\"\"Load and process market data\"\"\"\n        \n        self.logger.info(f\"Loading data from {data_path}\")\n        \n        # Load data\n        if data_path.endswith('.csv'):\n            df = pd.read_csv(data_path)\n            df['timestamp'] = pd.to_datetime(df['timestamp'])\n            df = df.set_index('timestamp')\n        else:\n            df = pd.read_parquet(data_path)\n        \n        # Select required columns\n        required_columns = ['open', 'high', 'low', 'close', 'volume']\n        df = df[required_columns]\n        \n        # Basic cleaning\n        df = df.dropna().sort_index()\n        \n        # Engineer features\n        self.processed_data = self.feature_engineer.engineer_features(df)\n        \n        self.logger.info(f\"Processed data shape: {self.processed_data.shape}\")\n        return self.processed_data\n    \n    def train_models(self, train_ratio=0.8):\n        \"\"\"Train ensemble of basic models\"\"\"\n        \n        if self.processed_data is None:\n            raise ValueError(\"Data not processed. Call load_and_process_data first.\")\n        \n        self.logger.info(\"Training models\")\n        \n        # Prepare data\n        label_cols = ['future_return_15m', 'profitable_15m', 'dipmaster_primary_label']\n        feature_cols = [col for col in self.processed_data.columns if col not in label_cols]\n        \n        X = self.processed_data[feature_cols]\n        y = self.processed_data['dipmaster_primary_label']\n        \n        # Time-based split\n        split_idx = int(len(X) * train_ratio)\n        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]\n        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]\n        \n        # Feature selection\n        selector = SelectKBest(f_classif, k=min(20, len(feature_cols)))\n        X_train_selected = selector.fit_transform(X_train, y_train)\n        X_test_selected = selector.transform(X_test)\n        \n        # Scale features\n        scaler = StandardScaler()\n        X_train_scaled = scaler.fit_transform(X_train_selected)\n        X_test_scaled = scaler.transform(X_test_selected)\n        \n        # Train models\n        models = {\n            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),\n            'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),\n            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000)\n        }\n        \n        # Cross-validation\n        cv = TimeSeriesSplit(n_splits=3)\n        \n        for name, model in models.items():\n            self.logger.info(f\"Training {name}\")\n            \n            # Use appropriate data (scaled for logistic regression)\n            if name == 'logistic_regression':\n                X_train_final = X_train_scaled\n                X_test_final = X_test_scaled\n            else:\n                X_train_final = X_train_selected\n                X_test_final = X_test_selected\n            \n            # Cross-validation\n            cv_scores = cross_val_score(model, X_train_final, y_train, cv=cv, scoring='roc_auc')\n            \n            # Train final model\n            model.fit(X_train_final, y_train)\n            \n            # Test predictions\n            if hasattr(model, 'predict_proba'):\n                y_pred_proba = model.predict_proba(X_test_final)[:, 1]\n            else:\n                y_pred_proba = model.predict(X_test_final)\n            \n            test_auc = roc_auc_score(y_test, y_pred_proba)\n            test_accuracy = accuracy_score(y_test, y_pred_proba > 0.5)\n            \n            self.trained_models[name] = {\n                'model': model,\n                'selector': selector if name != 'ensemble' else None,\n                'scaler': scaler if name == 'logistic_regression' else None\n            }\n            \n            self.validation_results[name] = {\n                'cv_score_mean': cv_scores.mean(),\n                'cv_score_std': cv_scores.std(),\n                'test_auc': test_auc,\n                'test_accuracy': test_accuracy\n            }\n            \n            self.logger.info(f\"{name} - CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\")\n            self.logger.info(f\"{name} - Test AUC: {test_auc:.4f}\")\n        \n        # Create ensemble\n        self.logger.info(\"Creating ensemble\")\n        ensemble_predictions = np.mean([\n            self.trained_models['random_forest']['model'].predict_proba(X_test_selected)[:, 1] * 0.4,\n            self.trained_models['gradient_boosting']['model'].predict_proba(X_test_selected)[:, 1] * 0.4,\n            self.trained_models['logistic_regression']['model'].predict_proba(X_test_scaled)[:, 1] * 0.2\n        ], axis=0)\n        \n        ensemble_auc = roc_auc_score(y_test, ensemble_predictions)\n        self.validation_results['ensemble'] = {\n            'test_auc': ensemble_auc,\n            'test_accuracy': accuracy_score(y_test, ensemble_predictions > 0.5)\n        }\n        \n        self.logger.info(f\"Ensemble - Test AUC: {ensemble_auc:.4f}\")\n        \n        # Store test data for signal generation\n        self.test_data = {\n            'X': X_test,\n            'y': y_test,\n            'predictions': ensemble_predictions,\n            'timestamps': X_test.index\n        }\n        \n        return self.validation_results\n    \n    def generate_signals(self, confidence_threshold=0.6):\n        \"\"\"Generate trading signals\"\"\"\n        \n        if not hasattr(self, 'test_data'):\n            raise ValueError(\"Models not trained. Call train_models first.\")\n        \n        self.logger.info(\"Generating trading signals\")\n        \n        predictions = self.test_data['predictions']\n        timestamps = self.test_data['timestamps']\n        \n        # Create signals DataFrame\n        signals_df = pd.DataFrame({\n            'timestamp': timestamps,\n            'signal': predictions,\n            'confidence': predictions,\n            'predicted_return': predictions * 0.012  # Scale to expected return\n        })\n        \n        # Filter by confidence\n        signals_df = signals_df[signals_df['confidence'] >= confidence_threshold]\n        \n        self.logger.info(f\"Generated {len(signals_df)} signals\")\n        return signals_df\n    \n    def run_backtest(self, signals_df):\n        \"\"\"Run backtest on signals\"\"\"\n        \n        market_data = self.processed_data[['open', 'high', 'low', 'close', 'volume']]\n        results = self.backtester.run_backtest(signals_df, market_data)\n        \n        return results\n    \n    def run_complete_pipeline(self, data_path: str):\n        \"\"\"Run complete pipeline\"\"\"\n        \n        self.logger.info(\"=== Starting Basic ML Pipeline ===\")\n        \n        start_time = datetime.now()\n        \n        try:\n            # 1. Load and process data\n            self.load_and_process_data(data_path)\n            \n            # 2. Train models\n            validation_results = self.train_models()\n            \n            # 3. Generate signals\n            signals_df = self.generate_signals()\n            \n            # 4. Run backtest\n            backtest_results = self.run_backtest(signals_df)\n            \n            # 5. Create summary\n            execution_time = (datetime.now() - start_time).total_seconds()\n            \n            # Performance targets\n            targets = {\n                'win_rate_achieved': backtest_results['win_rate'] >= 0.75,\n                'sharpe_achieved': backtest_results['sharpe_ratio'] >= 1.0,\n                'drawdown_ok': abs(backtest_results['max_drawdown']) <= 0.08,\n                'profit_factor_achieved': backtest_results['profit_factor'] >= 1.2\n            }\n            \n            all_targets_met = all(targets.values())\n            \n            summary = {\n                'pipeline_status': 'COMPLETED',\n                'execution_time_seconds': execution_time,\n                'targets_achieved': all_targets_met,\n                'target_details': targets,\n                'performance_metrics': {\n                    'total_return': backtest_results['total_return'],\n                    'win_rate': backtest_results['win_rate'],\n                    'sharpe_ratio': backtest_results['sharpe_ratio'],\n                    'max_drawdown': backtest_results['max_drawdown'],\n                    'profit_factor': backtest_results['profit_factor'],\n                    'total_trades': backtest_results['total_trades']\n                },\n                'model_performance': validation_results,\n                'data_statistics': {\n                    'total_samples': len(self.processed_data),\n                    'features_engineered': self.processed_data.shape[1],\n                    'signals_generated': len(signals_df)\n                }\n            }\n            \n            # Save results\n            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n            \n            # Save summary\n            summary_file = self.output_dir / f\"pipeline_summary_{timestamp}.json\"\n            with open(summary_file, 'w') as f:\n                json.dump(summary, f, indent=2, default=str)\n            \n            # Save signals\n            signals_file = self.output_dir / f\"signals_{timestamp}.csv\"\n            signals_df.to_csv(signals_file)\n            \n            # Save models\n            models_file = self.output_dir / f\"models_{timestamp}.joblib\"\n            joblib.dump(self.trained_models, models_file)\n            \n            self.logger.info(\"=== Pipeline Completed ===\")\n            self.logger.info(f\"Targets achieved: {all_targets_met}\")\n            self.logger.info(f\"Win rate: {backtest_results['win_rate']:.1%}\")\n            self.logger.info(f\"Sharpe ratio: {backtest_results['sharpe_ratio']:.2f}\")\n            self.logger.info(f\"Max drawdown: {backtest_results['max_drawdown']:.2%}\")\n            \n            return summary\n            \n        except Exception as e:\n            self.logger.error(f\"Pipeline failed: {str(e)}\")\n            raise\n\ndef main():\n    \"\"\"Main execution\"\"\"\n    \n    print(\"\\nDipMaster Basic ML Pipeline\")\n    print(\"==========================\\n\")\n    \n    # Setup logging\n    logging.basicConfig(\n        level=logging.INFO,\n        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'\n    )\n    \n    # Check for data file\n    data_path = \"data/market_data/SOLUSDT_5m_2years.csv\"\n    \n    if not os.path.exists(data_path):\n        print(f\"Data file not found: {data_path}\")\n        print(\"Available files:\")\n        data_dir = Path(\"data/market_data\")\n        if data_dir.exists():\n            for f in data_dir.glob(\"*.csv\"):\n                print(f\"  {f}\")\n        return\n    \n    # Run pipeline\n    pipeline = BasicMLPipeline()\n    \n    try:\n        results = pipeline.run_complete_pipeline(data_path)\n        \n        print(\"\\n=== RESULTS ===\")\n        print(f\"Status: {results['pipeline_status']}\")\n        print(f\"All targets achieved: {results['targets_achieved']}\")\n        print(f\"\\nPerformance:\")\n        for metric, value in results['performance_metrics'].items():\n            if isinstance(value, float):\n                if 'rate' in metric or 'return' in metric:\n                    print(f\"  {metric}: {value:.1%}\")\n                else:\n                    print(f\"  {metric}: {value:.2f}\")\n            else:\n                print(f\"  {metric}: {value}\")\n        \n        print(f\"\\nExecution time: {results['execution_time_seconds']:.1f} seconds\")\n        \n        # Create AlphaSignal format\n        alpha_signal = {\n            \"signal_uri\": f\"results/basic_ml_pipeline/signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv\",\n            \"schema\": [\"timestamp\", \"signal\", \"confidence\", \"predicted_return\"],\n            \"model_version\": \"DipMaster_Basic_v1.0.0\",\n            \"validation_metrics\": results['model_performance'],\n            \"backtest_performance\": results['performance_metrics'],\n            \"retrain_policy\": \"weekly\"\n        }\n        \n        alpha_file = f\"results/basic_ml_pipeline/AlphaSignal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json\"\n        with open(alpha_file, 'w') as f:\n            json.dump(alpha_signal, f, indent=2, default=str)\n        \n        print(f\"\\nAlphaSignal saved to: {alpha_file}\")\n        print(\"Pipeline completed successfully!\")\n        \n        return results\n        \n    except Exception as e:\n        print(f\"\\nError: {str(e)}\")\n        return None\n\nif __name__ == \"__main__\":\n    main()