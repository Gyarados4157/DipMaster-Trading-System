"""
DipMaster Enhanced V4 - Complete ML Training and Backtesting Pipeline
Implements the full machine learning pipeline with rigorous validation.
"""

import numpy as np
import pandas as pd
import json
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
import pickle

# Import our custom modules
from model_trainer import AdvancedModelTrainer, PurgedTimeSeriesSplit
from advanced_backtester import AdvancedBacktester, TradingCosts, RiskLimits, BacktestResult

warnings.filterwarnings('ignore')

class DipMasterV4MLPipeline:
    """
    Complete ML pipeline for DipMaster Enhanced V4 strategy
    """
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "results/ml_pipeline",
                 random_state: int = 42):
        
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
        
        # Initialize components
        self.trainer = AdvancedModelTrainer()
        self.backtester = None
        
        # Data containers
        self.raw_data = None
        self.features = None
        self.labels = None
        self.train_data = None
        self.validation_data = None
        self.test_data = None
        
        # Model containers
        self.ensemble_models = None
        self.validation_results = {}
        self.backtest_results = {}
        
        # Setup logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('DipMasterV4MLPipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # File handler
            log_file = self.output_dir / f"ml_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare the feature dataset"""
        self.logger.info(f"Loading data from {self.data_path}")
        
        if self.data_path.suffix == '.parquet':
            self.raw_data = pd.read_parquet(self.data_path)
        elif self.data_path.suffix == '.csv':
            self.raw_data = pd.read_csv(self.data_path, index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        
        self.logger.info(f"Loaded {len(self.raw_data)} samples with {self.raw_data.shape[1]} columns")
        
        # Ensure datetime index
        if not isinstance(self.raw_data.index, pd.DatetimeIndex):
            self.raw_data.index = pd.to_datetime(self.raw_data.index)
        
        # Sort by timestamp
        self.raw_data = self.raw_data.sort_index()
        
        # Handle missing values
        missing_pct = (self.raw_data.isnull().sum() / len(self.raw_data)) * 100
        if missing_pct.max() > 0:
            self.logger.warning(f"Missing values detected. Max missing: {missing_pct.max():.2f}%")
            self.raw_data = self.raw_data.fillna(method='ffill').fillna(method='bfill')
        
        return self.raw_data
    
    def prepare_features_and_labels(self):
        """Prepare features and labels for ML training"""
        self.logger.info("Preparing features and labels")
        
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Define feature columns (exclude labels)
        label_columns = [
            'future_return_15m', 'is_profitable_15m',
            'future_return_30m', 'is_profitable_30m', 
            'future_return_60m', 'is_profitable_60m',
            'hits_target_0.6%', 'hits_target_1.2%', 'hits_stop_loss'
        ]
        
        # Core features for DipMaster strategy
        core_features = [
            'rsi', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_position', 'bb_squeeze',
            'volume_ma', 'volume_ratio', 'volume_spike',
            'volatility_10', 'volatility_20', 'volatility_50',
            'momentum_5', 'momentum_10', 'momentum_20',
            'ma_15', 'ma_60', 'trend_short', 'trend_long', 'trend_alignment',
            'order_flow_imbalance', 'dipmaster_signal_strength'
        ]
        
        # Price features
        price_features = ['open', 'high', 'low', 'close', 'volume']
        
        # Additional technical features (if available)
        additional_features = [col for col in self.raw_data.columns 
                             if col not in label_columns + core_features + price_features
                             and not col.startswith('future_') and not col.startswith('hits_')]
        
        # Combine all features
        feature_columns = core_features + price_features + additional_features
        feature_columns = [col for col in feature_columns if col in self.raw_data.columns]
        
        self.features = self.raw_data[feature_columns].copy()
        
        # Primary label: profitable in 15 minutes
        primary_label = 'is_profitable_15m'
        if primary_label not in self.raw_data.columns:
            # Create label from future returns if not available
            if 'future_return_15m' in self.raw_data.columns:
                self.labels = (self.raw_data['future_return_15m'] > 0.006).astype(int)  # 0.6% profit target
            else:
                raise ValueError("No suitable label column found")
        else:
            self.labels = self.raw_data[primary_label].astype(int)
        
        # Remove rows with NaN labels
        valid_mask = ~self.labels.isna()
        self.features = self.features[valid_mask]
        self.labels = self.labels[valid_mask]
        
        self.logger.info(f"Prepared {len(self.features)} samples with {len(feature_columns)} features")
        self.logger.info(f"Label distribution: {self.labels.value_counts().to_dict()}")
        
        # Add some derived features for better predictions
        self._engineer_additional_features()
        
    def _engineer_additional_features(self):
        """Engineer additional features for better model performance"""
        self.logger.info("Engineering additional features")
        
        # RSI-based features
        if 'rsi' in self.features.columns:
            self.features['rsi_oversold'] = (self.features['rsi'] < 30).astype(int)
            self.features['rsi_dip_zone'] = ((self.features['rsi'] >= 25) & (self.features['rsi'] <= 45)).astype(int)
            self.features['rsi_momentum'] = self.features['rsi'].diff().fillna(0)\n        \n        # Price-based features  \n        if all(col in self.features.columns for col in ['open', 'close']):\n            self.features['price_change'] = (self.features['close'] - self.features['open']) / self.features['open']\n            self.features['is_dip'] = (self.features['price_change'] < -0.002).astype(int)  # 0.2% dip\n        \n        # Volume features\n        if all(col in self.features.columns for col in ['volume', 'volume_ma']):\n            self.features['volume_surge'] = (self.features['volume'] > self.features['volume_ma'] * 1.5).astype(int)\n        \n        # Volatility regime\n        if 'volatility_20' in self.features.columns:\n            vol_median = self.features['volatility_20'].rolling(100).median()\n            self.features['high_vol_regime'] = (self.features['volatility_20'] > vol_median * 1.2).astype(int)\n        \n        # Bollinger Band features\n        if all(col in self.features.columns for col in ['bb_position', 'bb_squeeze']):\n            self.features['bb_oversold'] = (self.features['bb_position'] < 0.2).astype(int)\n            self.features['bb_squeeze_exit'] = (self.features['bb_squeeze'] == 0).astype(int)\n        \n        # Time-based features\n        self.features['hour'] = self.features.index.hour\n        self.features['day_of_week'] = self.features.index.dayofweek\n        self.features['is_weekend'] = (self.features['day_of_week'] >= 5).astype(int)\n        \n        # Rolling statistics\n        for window in [5, 10, 20]:\n            if 'close' in self.features.columns:\n                self.features[f'price_std_{window}'] = self.features['close'].rolling(window).std()\n                self.features[f'price_skew_{window}'] = self.features['close'].rolling(window).skew()\n        \n        # Fill any new NaN values\n        self.features = self.features.fillna(method='ffill').fillna(method='bfill')\n        \n        self.logger.info(f\"Final feature set: {len(self.features.columns)} features\")\n    \n    def create_time_splits(self, \n                          train_pct: float = 0.6,\n                          validation_pct: float = 0.2,\n                          test_pct: float = 0.2):\n        \"\"\"Create time-based train/validation/test splits\"\"\"\n        \n        if abs(train_pct + validation_pct + test_pct - 1.0) > 1e-6:\n            raise ValueError(\"Split percentages must sum to 1.0\")\n        \n        n_samples = len(self.features)\n        train_end = int(n_samples * train_pct)\n        val_end = int(n_samples * (train_pct + validation_pct))\n        \n        # Create splits with time ordering\n        self.train_data = {\n            'X': self.features.iloc[:train_end].copy(),\n            'y': self.labels.iloc[:train_end].copy()\n        }\n        \n        self.validation_data = {\n            'X': self.features.iloc[train_end:val_end].copy(),\n            'y': self.labels.iloc[train_end:val_end].copy()\n        }\n        \n        self.test_data = {\n            'X': self.features.iloc[val_end:].copy(),\n            'y': self.labels.iloc[val_end:].copy()\n        }\n        \n        self.logger.info(f\"Data splits created:\")\n        self.logger.info(f\"  Train: {len(self.train_data['X'])} samples ({train_end})\")\n        self.logger.info(f\"  Validation: {len(self.validation_data['X'])} samples ({train_end}-{val_end})\")\n        self.logger.info(f\"  Test: {len(self.test_data['X'])} samples ({val_end}-{n_samples})\")\n        \n        # Log time ranges\n        self.logger.info(f\"  Train period: {self.train_data['X'].index.min()} to {self.train_data['X'].index.max()}\")\n        self.logger.info(f\"  Validation period: {self.validation_data['X'].index.min()} to {self.validation_data['X'].index.max()}\")\n        self.logger.info(f\"  Test period: {self.test_data['X'].index.min()} to {self.test_data['X'].index.max()}\")\n    \n    def train_models(self, \n                    model_names: List[str] = None,\n                    optimize_hyperparams: bool = True) -> Dict[str, Any]:\n        \"\"\"Train ensemble of models\"\"\"\n        \n        if model_names is None:\n            model_names = ['lgbm_classifier', 'xgb_classifier', 'random_forest', 'logistic_regression']\n        \n        self.logger.info(f\"Training models: {model_names}\")\n        \n        if self.train_data is None:\n            raise ValueError(\"Training data not prepared. Call create_time_splits() first.\")\n        \n        # Train ensemble\n        self.ensemble_models = self.trainer.train_ensemble(\n            X_train=self.train_data['X'],\n            y_train=self.train_data['y'],\n            model_names=model_names\n        )\n        \n        # Store validation results\n        self.validation_results = self.trainer.validation_results.copy()\n        \n        # Log training results\n        for model_name, result in self.validation_results.items():\n            self.logger.info(f\"{model_name} CV Score: {result.best_score:.4f} ± {np.std(result.cv_scores):.4f}\")\n        \n        return self.ensemble_models\n    \n    def validate_models(self) -> Dict[str, float]:\n        \"\"\"Validate models on out-of-sample validation set\"\"\"\n        self.logger.info(\"Validating models on validation set\")\n        \n        if self.ensemble_models is None:\n            raise ValueError(\"Models not trained. Call train_models() first.\")\n        \n        validation_scores = {}\n        \n        # Test each individual model\n        for model_name, model in self.ensemble_models.items():\n            if model_name == 'ensemble_predict':\n                continue\n                \n            X_val_prepared, _ = self.trainer.prepare_features(\n                self.validation_data['X'], None, model_name, fit_scalers=False\n            )\n            \n            if hasattr(model, 'predict_proba'):\n                y_pred_proba = model.predict_proba(X_val_prepared)[:, 1]\n            else:\n                y_pred_proba = model.predict(X_val_prepared)\n            \n            from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score\n            \n            auc_score = roc_auc_score(self.validation_data['y'], y_pred_proba)\n            y_pred_binary = (y_pred_proba > 0.5).astype(int)\n            accuracy = accuracy_score(self.validation_data['y'], y_pred_binary)\n            precision = precision_score(self.validation_data['y'], y_pred_binary)\n            recall = recall_score(self.validation_data['y'], y_pred_binary)\n            \n            validation_scores[model_name] = {\n                'auc': auc_score,\n                'accuracy': accuracy,\n                'precision': precision,\n                'recall': recall\n            }\n            \n            self.logger.info(f\"{model_name} Validation - AUC: {auc_score:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}\")\n        \n        # Test ensemble\n        ensemble_pred = self.ensemble_models['ensemble_predict'](self.validation_data['X'])\n        ensemble_auc = roc_auc_score(self.validation_data['y'], ensemble_pred)\n        ensemble_binary = (ensemble_pred > 0.5).astype(int)\n        ensemble_accuracy = accuracy_score(self.validation_data['y'], ensemble_binary)\n        ensemble_precision = precision_score(self.validation_data['y'], ensemble_binary)\n        \n        validation_scores['ensemble'] = {\n            'auc': ensemble_auc,\n            'accuracy': ensemble_accuracy,\n            'precision': ensemble_precision,\n            'recall': recall_score(self.validation_data['y'], ensemble_binary)\n        }\n        \n        self.logger.info(f\"Ensemble Validation - AUC: {ensemble_auc:.4f}, Accuracy: {ensemble_accuracy:.4f}, Precision: {ensemble_precision:.4f}\")\n        \n        return validation_scores\n    \n    def stress_test_models(self) -> Dict[str, Any]:\n        \"\"\"Perform stress tests on different market conditions\"\"\"\n        self.logger.info(\"Performing stress tests\")\n        \n        stress_results = {}\n        \n        # Test on different volatility regimes\n        if 'volatility_20' in self.validation_data['X'].columns:\n            vol_median = self.validation_data['X']['volatility_20'].median()\n            \n            # High volatility periods\n            high_vol_mask = self.validation_data['X']['volatility_20'] > vol_median * 1.5\n            if high_vol_mask.sum() > 100:  # Ensure sufficient samples\n                X_high_vol = self.validation_data['X'][high_vol_mask]\n                y_high_vol = self.validation_data['y'][high_vol_mask]\n                \n                ensemble_pred = self.ensemble_models['ensemble_predict'](X_high_vol)\n                stress_results['high_volatility'] = {\n                    'auc': roc_auc_score(y_high_vol, ensemble_pred),\n                    'samples': len(X_high_vol),\n                    'positive_rate': y_high_vol.mean()\n                }\n            \n            # Low volatility periods\n            low_vol_mask = self.validation_data['X']['volatility_20'] < vol_median * 0.5\n            if low_vol_mask.sum() > 100:\n                X_low_vol = self.validation_data['X'][low_vol_mask]\n                y_low_vol = self.validation_data['y'][low_vol_mask]\n                \n                ensemble_pred = self.ensemble_models['ensemble_predict'](X_low_vol)\n                stress_results['low_volatility'] = {\n                    'auc': roc_auc_score(y_low_vol, ensemble_pred),\n                    'samples': len(X_low_vol),\n                    'positive_rate': y_low_vol.mean()\n                }\n        \n        # Test on different time periods (weekend vs weekday)\n        if 'day_of_week' in self.validation_data['X'].columns:\n            weekend_mask = self.validation_data['X']['day_of_week'] >= 5\n            if weekend_mask.sum() > 50:\n                X_weekend = self.validation_data['X'][weekend_mask]\n                y_weekend = self.validation_data['y'][weekend_mask]\n                \n                ensemble_pred = self.ensemble_models['ensemble_predict'](X_weekend)\n                stress_results['weekend'] = {\n                    'auc': roc_auc_score(y_weekend, ensemble_pred),\n                    'samples': len(X_weekend),\n                    'positive_rate': y_weekend.mean()\n                }\n        \n        self.logger.info(f\"Stress test results: {stress_results}\")\n        return stress_results\n    \n    def generate_trading_signals(self, confidence_threshold: float = 0.6) -> pd.DataFrame:\n        \"\"\"Generate trading signals for backtesting\"\"\"\n        self.logger.info(f\"Generating trading signals (confidence >= {confidence_threshold})\")\n        \n        if self.ensemble_models is None:\n            raise ValueError(\"Models not trained. Call train_models() first.\")\n        \n        # Generate signals on test set\n        signals_df = self.trainer.generate_alpha_signals(\n            X_test=self.test_data['X'],\n            ensemble_models=self.ensemble_models,\n            confidence_threshold=confidence_threshold\n        )\n        \n        # Add symbol information (assuming single symbol for now)\n        signals_df['symbol'] = 'BTCUSDT'  # Default symbol\n        \n        self.logger.info(f\"Generated {len(signals_df)} trading signals\")\n        self.logger.info(f\"Signal distribution: {signals_df['signal'].value_counts().to_dict()}\")\n        \n        return signals_df\n    \n    def run_backtest(self, \n                    signals_df: pd.DataFrame,\n                    initial_capital: float = 10000) -> BacktestResult:\n        \"\"\"Run comprehensive backtest\"\"\"\n        self.logger.info(\"Running backtest\")\n        \n        # Setup backtester with realistic costs\n        costs = TradingCosts(\n            commission_rate=0.0004,  # 0.04% total (0.02% each side)\n            slippage_base=0.0001,    # 0.01% base slippage\n            funding_rate_8h=0.0001   # 0.01% funding every 8 hours\n        )\n        \n        risk_limits = RiskLimits(\n            max_position_size=1000,\n            max_concurrent_positions=3,\n            daily_loss_limit=-500\n        )\n        \n        self.backtester = AdvancedBacktester(\n            initial_capital=initial_capital,\n            costs=costs,\n            risk_limits=risk_limits\n        )\n        \n        # Prepare market data for backtesting\n        market_data = self.test_data['X'][['open', 'high', 'low', 'close', 'volume']].copy()\n        \n        # Add required columns if missing\n        if 'volatility_20' not in market_data.columns:\n            market_data['volatility_20'] = market_data['close'].pct_change().rolling(20).std()\n        \n        # Run backtest\n        result = self.backtester.run_backtest(signals_df, market_data)\n        \n        self.backtest_results['main'] = result\n        \n        # Log key results\n        self.logger.info(f\"Backtest Results:\")\n        self.logger.info(f\"  Total Return: {result.total_return:.2%}\")\n        self.logger.info(f\"  Win Rate: {result.win_rate:.1%}\")\n        self.logger.info(f\"  Sharpe Ratio: {result.sharpe_ratio:.2f}\")\n        self.logger.info(f\"  Max Drawdown: {result.max_drawdown:.2%}\")\n        self.logger.info(f\"  Total Trades: {result.total_trades}\")\n        \n        return result\n    \n    def save_alpha_signals(self, signals_df: pd.DataFrame, filename: str = None):\n        \"\"\"Save alpha signals in standard format\"\"\"\n        \n        if filename is None:\n            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n            filename = f\"AlphaSignal_DipMasterV4_{timestamp}.json\"\n        \n        output_path = self.output_dir / filename\n        \n        # Prepare alpha signal format\n        alpha_signal = {\n            \"signal_uri\": str(output_path.with_suffix('.parquet')),\n            \"schema\": [\"timestamp\", \"symbol\", \"score\", \"confidence\", \"predicted_return\"],\n            \"model_version\": \"DipMaster_Enhanced_V4_1.0.0\",\n            \"retrain_policy\": \"weekly\",\n            \"feature_importance\": {},\n            \"validation_metrics\": {}\n        }\n        \n        # Add feature importance from best model\n        if self.validation_results:\n            best_model = max(self.validation_results.keys(), \n                           key=lambda x: self.validation_results[x].best_score)\n            alpha_signal[\"feature_importance\"] = self.validation_results[best_model].feature_importance\n            \n            # Add validation metrics\n            for model_name, result in self.validation_results.items():\n                alpha_signal[\"validation_metrics\"][model_name] = {\n                    \"cv_score\": result.best_score,\n                    \"cv_std\": float(np.std(result.cv_scores)),\n                    \"hyperparams\": result.hyperparams\n                }\n        \n        # Add generation metadata\n        alpha_signal[\"generation_metadata\"] = {\n            \"generated_timestamp\": datetime.now().isoformat(),\n            \"total_signals\": len(signals_df),\n            \"confident_signals\": len(signals_df[signals_df['confidence'] >= 0.6]),\n            \"signal_period\": {\n                \"start\": signals_df['timestamp'].min().isoformat(),\n                \"end\": signals_df['timestamp'].max().isoformat()\n            },\n            \"model_performance\": {\n                \"expected_win_rate\": \"85%+\",\n                \"target_sharpe\": \"2.0+\",\n                \"max_drawdown_limit\": \"3%\"\n            }\n        }\n        \n        # Save JSON\n        with open(output_path, 'w') as f:\n            json.dump(alpha_signal, f, indent=2, default=str)\n        \n        # Save signals as parquet\n        signals_df.to_parquet(output_path.with_suffix('.parquet'))\n        \n        self.logger.info(f\"Alpha signals saved to {output_path}\")\n        \n        return output_path\n    \n    def generate_backtest_report(self, result: BacktestResult, filename: str = None):\n        \"\"\"Generate comprehensive backtest report\"\"\"\n        \n        if filename is None:\n            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n            filename = f\"BacktestReport_DipMasterV4_{timestamp}.html\"\n        \n        output_path = self.output_dir / filename\n        \n        self.backtester.generate_report(result, str(output_path))\n        \n        self.logger.info(f\"Backtest report saved to {output_path}\")\n        \n        return output_path\n    \n    def run_complete_pipeline(self, \n                             confidence_threshold: float = 0.6,\n                             initial_capital: float = 10000) -> Dict[str, Any]:\n        \"\"\"Run the complete ML pipeline\"\"\"\n        \n        self.logger.info(\"=== Starting DipMaster Enhanced V4 ML Pipeline ===\")\n        \n        # 1. Load and prepare data\n        self.load_data()\n        self.prepare_features_and_labels()\n        self.create_time_splits()\n        \n        # 2. Train models\n        self.train_models()\n        \n        # 3. Validate models\n        validation_scores = self.validate_models()\n        \n        # 4. Stress test\n        stress_results = self.stress_test_models()\n        \n        # 5. Generate signals\n        signals_df = self.generate_trading_signals(confidence_threshold)\n        \n        # 6. Run backtest\n        backtest_result = self.run_backtest(signals_df, initial_capital)\n        \n        # 7. Save results\n        alpha_signal_path = self.save_alpha_signals(signals_df)\n        report_path = self.generate_backtest_report(backtest_result)\n        \n        # 8. Performance assessment\n        performance_targets = {\n            'win_rate_target': 0.85,\n            'sharpe_target': 2.0,\n            'max_drawdown_limit': 0.03,\n            'profit_factor_target': 1.8\n        }\n        \n        target_achievement = {\n            'win_rate_achieved': backtest_result.win_rate >= performance_targets['win_rate_target'],\n            'sharpe_achieved': backtest_result.sharpe_ratio >= performance_targets['sharpe_target'],\n            'drawdown_ok': abs(backtest_result.max_drawdown) <= performance_targets['max_drawdown_limit'],\n            'profit_factor_achieved': backtest_result.profit_factor >= performance_targets['profit_factor_target']\n        }\n        \n        all_targets_met = all(target_achievement.values())\n        \n        # Final summary\n        summary = {\n            'pipeline_status': 'COMPLETED',\n            'targets_achieved': all_targets_met,\n            'target_details': target_achievement,\n            'performance_metrics': {\n                'total_return': backtest_result.total_return,\n                'win_rate': backtest_result.win_rate,\n                'sharpe_ratio': backtest_result.sharpe_ratio,\n                'max_drawdown': backtest_result.max_drawdown,\n                'profit_factor': backtest_result.profit_factor,\n                'total_trades': backtest_result.total_trades,\n                'statistical_significance': backtest_result.p_value < 0.05\n            },\n            'model_performance': validation_scores,\n            'stress_test_results': stress_results,\n            'output_files': {\n                'alpha_signals': str(alpha_signal_path),\n                'backtest_report': str(report_path)\n            }\n        }\n        \n        # Save summary\n        summary_path = self.output_dir / f\"pipeline_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json\"\n        with open(summary_path, 'w') as f:\n            json.dump(summary, f, indent=2, default=str)\n        \n        self.logger.info(\"=== Pipeline Completed ===\")\n        self.logger.info(f\"All targets achieved: {all_targets_met}\")\n        self.logger.info(f\"Summary saved to: {summary_path}\")\n        \n        return summary\n\n\ndef main():\n    \"\"\"Main execution function\"\"\"\n    \n    # Configure pipeline\n    data_path = \"G:/Github/Quant/DipMaster-Trading-System/data/dipmaster_v4_features_20250816_175605.parquet\"\n    output_dir = \"G:/Github/Quant/DipMaster-Trading-System/results/ml_pipeline\"\n    \n    # Initialize pipeline\n    pipeline = DipMasterV4MLPipeline(\n        data_path=data_path,\n        output_dir=output_dir\n    )\n    \n    # Run complete pipeline\n    results = pipeline.run_complete_pipeline(\n        confidence_threshold=0.6,\n        initial_capital=10000\n    )\n    \n    print(\"\\n=== FINAL RESULTS ===\")\n    print(f\"Targets Achieved: {results['targets_achieved']}\")\n    print(f\"Win Rate: {results['performance_metrics']['win_rate']:.1%}\")\n    print(f\"Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.2f}\")\n    print(f\"Max Drawdown: {results['performance_metrics']['max_drawdown']:.2%}\")\n    print(f\"Total Trades: {results['performance_metrics']['total_trades']}\")\n    \n    return results\n\n\nif __name__ == \"__main__\":\n    main()"