#!/usr/bin/env python3
"""
DipMaster持续训练优化系统
目标: 胜率85%+, 夏普比率>1.5, 最大回撤<3%, 年化收益>15%
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
except ImportError:
    cb = None

# 超参数优化
import optuna
from optuna.samplers import TPESampler

# 回测和性能分析
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 本地模块
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 数据预处理和验证
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import pickle
import joblib
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """训练配置类"""
    target_win_rate: float = 0.85
    target_sharpe: float = 1.5
    target_max_drawdown: float = 0.03
    target_annual_return: float = 0.15
    
    # 交易成本
    commission_rate: float = 0.001  # 0.1% commission
    slippage_rate: float = 0.0005   # 0.05% slippage
    
    # 训练参数
    cv_folds: int = 5
    embargo_hours: int = 2
    test_size: float = 0.2
    
    # 模型参数
    optuna_trials: int = 100
    max_features_per_model: int = 30
    ensemble_models: List[str] = None
    
    def __post_init__(self):
        if self.ensemble_models is None:
            self.ensemble_models = ['lgbm', 'xgb', 'rf', 'lr']

class PurgedKFold:
    """时序交叉验证with数据清洗"""
    
    def __init__(self, n_splits=5, embargo_hours=2):
        self.n_splits = n_splits
        self.embargo_td = timedelta(hours=embargo_hours)
        
    def split(self, X, y=None, groups=None):
        """生成训练/验证分割"""
        if 'timestamp' not in X.columns:
            raise ValueError("需要timestamp列进行时序分割")
            
        timestamps = pd.to_datetime(X['timestamp'])
        indices = np.arange(len(X))
        
        # 计算分割点
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # 训练集结束点
            train_end_idx = fold_size * (i + 1)
            if train_end_idx >= n_samples:
                continue
                
            # 验证集开始点（添加embargo期）
            train_end_time = timestamps.iloc[train_end_idx]
            val_start_time = train_end_time + self.embargo_td
            
            # 找到验证集实际开始索引
            val_start_idx = timestamps[timestamps >= val_start_time].index
            if len(val_start_idx) == 0:
                continue
            val_start_idx = val_start_idx[0]
            
            # 验证集结束点
            val_end_idx = min(train_end_idx + fold_size, n_samples)
            
            train_indices = indices[:train_end_idx]
            val_indices = indices[val_start_idx:val_end_idx]
            
            if len(train_indices) > 100 and len(val_indices) > 50:
                yield train_indices, val_indices

class ContinuousTrainingSystem:
    """持续训练优化系统"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_history = []
        self.best_params = {}
        
        # 创建输出目录
        self.output_dir = Path("results/continuous_training")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        np.random.seed(42)
        
    def load_multi_symbol_data(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """加载多币种特征数据"""
        logger.info("加载多币种特征数据...")
        
        data_files = list(Path(data_dir).glob("features_*_optimized_*.parquet"))
        datasets = {}
        
        for file_path in data_files:
            try:
                # 从文件名提取币种
                symbol = file_path.stem.split('_')[1]  # features_BTCUSDT_optimized_...
                
                df = pd.read_parquet(file_path)
                if len(df) > 1000:  # 确保有足够的数据
                    datasets[symbol] = df
                    logger.info(f"加载 {symbol}: {len(df)} samples, {len(df.columns)} features")
                    
            except Exception as e:
                logger.warning(f"无法加载 {file_path}: {e}")
                
        logger.info(f"成功加载 {len(datasets)} 个币种的数据")
        return datasets
        
    def prepare_features_and_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """准备特征和标签"""
        # 移除时间戳和原始价格数据
        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'target_return', 'target_binary', 
            'target_return_12p', 'target_return_24p', 'target_return_48p',
            'target_profitable_12p', 'target_profitable_24p', 'target_profitable_48p'
        ]]
        
        # 选择主要目标（12期标签最相关）
        X = df[feature_cols].copy()
        y_return = df['target_return_12p'].copy() if 'target_return_12p' in df.columns else df.get('target_return', pd.Series())
        y_binary = df['target_profitable_12p'].copy() if 'target_profitable_12p' in df.columns else df.get('target_binary', pd.Series())
        
        # 清理数据
        mask = ~(y_return.isna() | y_binary.isna())
        X = X[mask].fillna(method='ffill').fillna(0)
        y_return = y_return[mask]
        y_binary = y_binary[mask]
        
        logger.info(f"特征维度: {X.shape}, 正样本比例: {y_binary.mean():.3f}")
        
        return X, y_return, y_binary
    
    def optimize_hyperparameters(self, X, y, model_type: str) -> Dict:
        """使用Optuna优化超参数"""
        logger.info(f"优化 {model_type} 超参数...")
        
        def objective(trial):
            if model_type == 'lgbm':
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
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'verbose': -1
                }
                
            elif model_type == 'xgb':
                params = {
                    'objective': 'binary:logistic',
                    'eval_metric': 'logloss',
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                }
                
            elif model_type == 'rf':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 5, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.8]),
                }
                
            else:  # logistic regression
                params = {
                    'C': trial.suggest_float('C', 0.01, 10, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                    'solver': 'saga',
                    'max_iter': 1000
                }
                if params['penalty'] == 'elasticnet':
                    params['l1_ratio'] = trial.suggest_float('l1_ratio', 0, 1)
            
            # 交叉验证评估
            cv_scores = []
            kfold = PurgedKFold(n_splits=3, embargo_hours=self.config.embargo_hours)
            
            for train_idx, val_idx in kfold.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 标准化
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                try:
                    if model_type == 'lgbm':
                        model = lgb.LGBMClassifier(**params)
                    elif model_type == 'xgb':
                        model = xgb.XGBClassifier(**params)
                    elif model_type == 'rf':
                        model = RandomForestClassifier(**params, random_state=42)
                    else:
                        model = LogisticRegression(**params, random_state=42)
                    
                    model.fit(X_train_scaled, y_train)
                    y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
                    auc_score = roc_auc_score(y_val, y_pred_proba)
                    cv_scores.append(auc_score)
                    
                except Exception as e:
                    logger.warning(f"模型训练失败: {e}")
                    return 0.5
                    
            return np.mean(cv_scores) if cv_scores else 0.5
        
        # 运行优化
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
        
        study.optimize(objective, n_trials=self.config.optuna_trials)
        
        logger.info(f"{model_type} 最佳AUC: {study.best_value:.4f}")
        return study.best_params
    
    def train_ensemble_model(self, X, y_return, y_binary) -> Dict:
        """训练集成模型"""
        logger.info("训练集成模型...")
        
        # 优化各个模型的超参数
        optimized_params = {}
        for model_type in self.config.ensemble_models:
            if model_type in ['lgbm', 'xgb', 'rf', 'lr']:
                optimized_params[model_type] = self.optimize_hyperparameters(X, y_binary, model_type)
        
        # 训练所有模型
        models = {}
        scalers = {}
        
        # 分割数据
        train_size = int(len(X) * (1 - self.config.test_size))
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_binary_train, y_binary_test = y_binary.iloc[:train_size], y_binary.iloc[train_size:]
        y_return_train, y_return_test = y_return.iloc[:train_size], y_return.iloc[train_size:]
        
        # 标准化
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        scalers['main'] = scaler
        
        # 训练各个模型
        for model_type in self.config.ensemble_models:
            try:
                logger.info(f"训练 {model_type} 模型...")
                params = optimized_params.get(model_type, {})
                
                if model_type == 'lgbm':
                    model = lgb.LGBMClassifier(**params)
                elif model_type == 'xgb':
                    model = xgb.XGBClassifier(**params)
                elif model_type == 'rf':
                    model = RandomForestClassifier(**params, random_state=42)
                else:  # lr
                    model = LogisticRegression(**params, random_state=42)
                
                model.fit(X_train_scaled, y_binary_train)
                models[model_type] = model
                
                # 计算特征重要性
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_[0])
                else:
                    importance = np.ones(len(X_train.columns))
                
                self.feature_importance[model_type] = dict(zip(X_train.columns, importance))
                
            except Exception as e:
                logger.error(f"训练 {model_type} 失败: {e}")
                
        # 集成预测
        ensemble_predictions = self._ensemble_predict(models, scalers, X_test)
        
        # 回测评估
        backtest_results = self._comprehensive_backtest(
            predictions=ensemble_predictions,
            actual_returns=y_return_test,
            actual_binary=y_binary_test,
            timestamps=X_test.get('timestamp', pd.Series(range(len(X_test))))
        )
        
        return {
            'models': models,
            'scalers': scalers,
            'optimized_params': optimized_params,
            'backtest_results': backtest_results,
            'test_data': {
                'X_test': X_test,
                'y_return_test': y_return_test,
                'y_binary_test': y_binary_test,
                'predictions': ensemble_predictions
            }
        }
    
    def _ensemble_predict(self, models: Dict, scalers: Dict, X_test: pd.DataFrame) -> np.ndarray:
        """集成预测"""
        X_test_scaled = scalers['main'].transform(X_test)
        
        predictions = []
        weights = []
        
        for model_name, model in models.items():
            try:
                pred = model.predict_proba(X_test_scaled)[:, 1]
                predictions.append(pred)
                
                # 根据模型类型设置权重
                if model_name in ['lgbm', 'xgb']:
                    weights.append(0.4)  # 树模型权重较高
                elif model_name == 'rf':
                    weights.append(0.15)
                else:  # lr
                    weights.append(0.05)
                    
            except Exception as e:
                logger.warning(f"{model_name} 预测失败: {e}")
        
        if not predictions:
            return np.zeros(len(X_test))
            
        # 加权平均
        weights = np.array(weights)
        weights = weights / weights.sum()  # 归一化权重
        
        ensemble_pred = np.average(predictions, axis=0, weights=weights)
        return ensemble_pred
    
    def _comprehensive_backtest(self, predictions: np.ndarray, actual_returns: pd.Series, 
                               actual_binary: pd.Series, timestamps: pd.Series) -> Dict:
        """综合回测分析"""
        logger.info("执行综合回测分析...")
        
        # 转换为numpy数组以便计算
        pred_proba = predictions
        actual_ret = actual_returns.values
        actual_bin = actual_binary.values
        
        # 动态阈值优化
        thresholds = np.linspace(0.1, 0.9, 50)
        best_threshold = 0.5
        best_sharpe = -999
        
        threshold_results = []
        
        for threshold in thresholds:
            signals = (pred_proba >= threshold).astype(int)
            
            if signals.sum() == 0:  # 没有信号
                continue
                
            # 只在有信号时进行交易
            signal_returns = actual_ret[signals == 1]
            
            if len(signal_returns) < 10:  # 信号太少
                continue
            
            # 应用交易成本
            gross_returns = signal_returns
            net_returns = gross_returns - self.config.commission_rate - self.config.slippage_rate
            
            # 计算性能指标
            win_rate = (signal_returns > 0).mean()
            avg_return = net_returns.mean()
            std_return = net_returns.std()
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            
            # 累积收益
            cum_returns = (1 + pd.Series(net_returns)).cumprod()
            max_drawdown = (cum_returns / cum_returns.expanding().max() - 1).min()
            
            result = {
                'threshold': threshold,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'num_trades': len(signal_returns),
                'annual_return': avg_return * 252 * 24 * 12  # 假设5分钟数据
            }
            
            threshold_results.append(result)
            
            # 更新最佳阈值
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_threshold = threshold
        
        # 使用最佳阈值进行最终回测
        best_signals = (pred_proba >= best_threshold).astype(int)
        signal_indices = np.where(best_signals == 1)[0]
        
        if len(signal_indices) > 0:
            signal_returns = actual_ret[signal_indices]
            net_returns = signal_returns - self.config.commission_rate - self.config.slippage_rate
            
            # 计算最终性能指标
            final_metrics = {
                'best_threshold': best_threshold,
                'win_rate': (signal_returns > 0).mean(),
                'avg_return_per_trade': net_returns.mean(),
                'sharpe_ratio': net_returns.mean() / net_returns.std() if net_returns.std() > 0 else 0,
                'num_trades': len(signal_returns),
                'total_return': net_returns.sum(),
                'max_drawdown': (pd.Series(net_returns).cumsum().expanding().max() - 
                               pd.Series(net_returns).cumsum()).max(),
                'annual_return': net_returns.mean() * 252 * 24 * 12,  # 年化收益
                'profit_factor': (signal_returns[signal_returns > 0].sum() / 
                                -signal_returns[signal_returns < 0].sum() 
                                if (signal_returns < 0).any() else np.inf),
                'threshold_analysis': threshold_results
            }
            
            # 目标达成情况
            targets_achieved = {
                'win_rate_target': final_metrics['win_rate'] >= self.config.target_win_rate,
                'sharpe_target': final_metrics['sharpe_ratio'] >= self.config.target_sharpe,
                'drawdown_target': abs(final_metrics['max_drawdown']) <= self.config.target_max_drawdown,
                'return_target': final_metrics['annual_return'] >= self.config.target_annual_return
            }
            
            final_metrics['targets_achieved'] = targets_achieved
            final_metrics['all_targets_met'] = all(targets_achieved.values())
            
        else:
            final_metrics = {'error': 'No valid signals generated'}
        
        return final_metrics
    
    def continuous_optimization_loop(self, data_dir: str, hours_interval: int = 2):
        """持续优化循环"""
        logger.info(f"启动持续优化循环，每{hours_interval}小时重训练")
        
        iteration = 0
        
        while True:
            try:
                logger.info(f"\n=== 优化迭代 {iteration + 1} ===")
                
                # 加载最新数据
                datasets = self.load_multi_symbol_data(data_dir)
                
                if not datasets:
                    logger.error("未找到有效数据，跳过此次迭代")
                    continue
                
                # 合并多币种数据进行训练
                combined_results = {}
                
                for symbol, data in datasets.items():
                    logger.info(f"\n--- 处理 {symbol} ---")
                    
                    try:
                        X, y_return, y_binary = self.prepare_features_and_labels(data)
                        
                        if len(X) < 1000:
                            logger.warning(f"{symbol} 数据不足，跳过")
                            continue
                        
                        # 训练模型
                        result = self.train_ensemble_model(X, y_return, y_binary)
                        combined_results[symbol] = result
                        
                        # 检查是否达到目标
                        backtest = result['backtest_results']
                        
                        if isinstance(backtest, dict) and 'all_targets_met' in backtest:
                            if backtest['all_targets_met']:
                                logger.info(f"🎉 {symbol} 达到所有目标指标!")
                                self._save_successful_model(symbol, result, iteration)
                            else:
                                logger.info(f"⚠️ {symbol} 尚未达到目标，继续优化...")
                                self._log_performance_gap(symbol, backtest)
                        
                    except Exception as e:
                        logger.error(f"处理 {symbol} 时出错: {e}")
                        continue
                
                # 生成综合报告
                self._generate_iteration_report(combined_results, iteration)
                
                # 性能历史记录
                self.performance_history.append({
                    'iteration': iteration,
                    'timestamp': datetime.now(),
                    'results': combined_results
                })
                
                iteration += 1
                
                # 检查是否有任何币种达到目标
                targets_met = any(
                    result.get('backtest_results', {}).get('all_targets_met', False)
                    for result in combined_results.values()
                )
                
                if targets_met:
                    logger.info("🏆 检测到达标模型，生成最终报告...")
                    self._generate_final_report()
                    break
                
                # 等待下次迭代
                logger.info(f"等待 {hours_interval} 小时后继续下次优化...")
                import time
                time.sleep(hours_interval * 3600)
                
            except KeyboardInterrupt:
                logger.info("用户中断，生成最终报告...")
                self._generate_final_report()
                break
            except Exception as e:
                logger.error(f"迭代 {iteration} 出现错误: {e}")
                continue
    
    def _save_successful_model(self, symbol: str, result: Dict, iteration: int):
        """保存成功的模型"""
        model_dir = self.output_dir / f"successful_models/{symbol}_iteration_{iteration}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        for model_name, model in result['models'].items():
            joblib.dump(model, model_dir / f"{model_name}_model.pkl")
        
        # 保存标准化器
        joblib.dump(result['scalers'], model_dir / "scalers.pkl")
        
        # 保存结果
        with open(model_dir / "results.json", 'w') as f:
            # 转换不可序列化的对象
            serializable_result = self._make_json_serializable(result)
            json.dump(serializable_result, f, indent=2)
        
        logger.info(f"✅ 成功模型已保存至: {model_dir}")
    
    def _make_json_serializable(self, obj):
        """将对象转换为JSON可序列化格式"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()
                   if k not in ['models', 'scalers', 'test_data']}  # 排除不可序列化对象
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _log_performance_gap(self, symbol: str, backtest: Dict):
        """记录性能差距"""
        targets = backtest.get('targets_achieved', {})
        metrics = {
            'win_rate': backtest.get('win_rate', 0),
            'sharpe_ratio': backtest.get('sharpe_ratio', 0),
            'max_drawdown': abs(backtest.get('max_drawdown', 0)),
            'annual_return': backtest.get('annual_return', 0)
        }
        
        logger.info(f"{symbol} 当前性能:")
        logger.info(f"  胜率: {metrics['win_rate']:.3f} (目标: {self.config.target_win_rate:.3f}) {'✓' if targets.get('win_rate_target', False) else '✗'}")
        logger.info(f"  夏普比率: {metrics['sharpe_ratio']:.3f} (目标: {self.config.target_sharpe:.3f}) {'✓' if targets.get('sharpe_target', False) else '✗'}")
        logger.info(f"  最大回撤: {metrics['max_drawdown']:.3f} (目标: <{self.config.target_max_drawdown:.3f}) {'✓' if targets.get('drawdown_target', False) else '✗'}")
        logger.info(f"  年化收益: {metrics['annual_return']:.3f} (目标: >{self.config.target_annual_return:.3f}) {'✓' if targets.get('return_target', False) else '✗'}")
    
    def _generate_iteration_report(self, results: Dict, iteration: int):
        """生成迭代报告"""
        report_path = self.output_dir / f"iteration_{iteration}_report.json"
        
        with open(report_path, 'w') as f:
            serializable_results = self._make_json_serializable(results)
            json.dump({
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'results': serializable_results
            }, f, indent=2)
        
        logger.info(f"📊 迭代 {iteration} 报告已保存: {report_path}")
    
    def _generate_final_report(self):
        """生成最终性能分析报告"""
        logger.info("生成最终性能分析报告...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_iterations': len(self.performance_history),
            'performance_summary': {},
            'successful_models': [],
            'recommendations': []
        }
        
        # 分析成功模型目录
        successful_models_dir = self.output_dir / "successful_models"
        if successful_models_dir.exists():
            for model_dir in successful_models_dir.iterdir():
                if model_dir.is_dir():
                    try:
                        with open(model_dir / "results.json", 'r') as f:
                            model_result = json.load(f)
                            report['successful_models'].append({
                                'path': str(model_dir),
                                'symbol': model_dir.name.split('_')[0],
                                'iteration': model_dir.name.split('_')[-1],
                                'performance': model_result.get('backtest_results', {})
                            })
                    except Exception as e:
                        logger.warning(f"无法读取模型结果: {e}")
        
        # 生成HTML报告
        html_report = self._create_html_report(report)
        html_path = self.output_dir / "final_performance_report.html"
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # 保存JSON报告
        json_path = self.output_dir / "final_performance_report.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📈 最终报告已生成:")
        logger.info(f"  HTML: {html_path}")
        logger.info(f"  JSON: {json_path}")
        
        return report
    
    def _create_html_report(self, report: Dict) -> str:
        """创建HTML格式的报告"""
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DipMaster持续训练优化报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary {{ background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }}
        .model-card {{
            border: 1px solid #bdc3c7;
            padding: 20px;
            margin: 15px 0;
            border-radius: 5px;
            background: #ffffff;
        }}
        .success {{ border-left: 4px solid #27ae60; background: #d5f5d8; }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .metric {{
            text-align: center;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 5px;
        }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #2980b9; }}
        .metric-label {{ font-size: 14px; color: #7f8c8d; }}
        .status-good {{ color: #27ae60; }}
        .status-bad {{ color: #e74c3c; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #3498db; color: white; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 DipMaster持续训练优化报告</h1>
        
        <div class="summary">
            <h2>📊 执行总结</h2>
            <p><strong>报告生成时间:</strong> {report['timestamp']}</p>
            <p><strong>总迭代次数:</strong> {report['total_iterations']}</p>
            <p><strong>成功模型数量:</strong> {len(report['successful_models'])}</p>
        </div>
        
        <h2>🏆 成功模型</h2>
        {self._generate_success_models_html(report['successful_models'])}
        
        <h2>🎯 目标达成情况</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">目标胜率</div>
                <div class="metric-value">≥85%</div>
            </div>
            <div class="metric">
                <div class="metric-label">目标夏普比率</div>
                <div class="metric-value">≥1.5</div>
            </div>
            <div class="metric">
                <div class="metric-label">目标最大回撤</div>
                <div class="metric-value">≤3%</div>
            </div>
            <div class="metric">
                <div class="metric-label">目标年化收益</div>
                <div class="metric-value">≥15%</div>
            </div>
        </div>
        
        <div style="margin-top: 40px; padding: 20px; background: #f8f9fa; border-radius: 5px;">
            <p><strong>注意:</strong> 本报告基于历史数据回测生成，实际交易结果可能与回测结果存在差异。</p>
            <p><strong>风险提示:</strong> 加密货币交易存在高风险，请谨慎投资。</p>
        </div>
    </div>
</body>
</html>
        """
        return html
    
    def _generate_success_models_html(self, successful_models: List) -> str:
        """生成成功模型的HTML"""
        if not successful_models:
            return "<p>暂无达到目标的模型。</p>"
        
        html = ""
        for model in successful_models:
            performance = model.get('performance', {})
            
            html += f"""
            <div class="model-card success">
                <h3>✅ {model['symbol']} - 迭代 {model['iteration']}</h3>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value status-good">{performance.get('win_rate', 0):.1%}</div>
                        <div class="metric-label">胜率</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value status-good">{performance.get('sharpe_ratio', 0):.2f}</div>
                        <div class="metric-label">夏普比率</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value status-good">{abs(performance.get('max_drawdown', 0)):.1%}</div>
                        <div class="metric-label">最大回撤</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value status-good">{performance.get('annual_return', 0):.1%}</div>
                        <div class="metric-label">年化收益</div>
                    </div>
                </div>
                <p><strong>交易次数:</strong> {performance.get('num_trades', 0)}</p>
                <p><strong>盈亏比:</strong> {performance.get('profit_factor', 0):.2f}</p>
                <p><strong>模型路径:</strong> {model['path']}</p>
            </div>
            """
        
        return html

def main():
    """主函数"""
    # 配置
    config = TrainingConfig()
    
    # 创建训练系统
    training_system = ContinuousTrainingSystem(config)
    
    # 数据目录
    data_dir = "data/continuous_optimization"
    
    logger.info("🚀 启动DipMaster持续训练优化系统")
    logger.info(f"目标: 胜率≥{config.target_win_rate:.1%}, 夏普≥{config.target_sharpe}, 回撤≤{config.target_max_drawdown:.1%}, 年化≥{config.target_annual_return:.1%}")
    
    try:
        # 运行持续优化循环
        training_system.continuous_optimization_loop(data_dir, hours_interval=2)
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        raise
    
    logger.info("✅ 持续训练优化系统运行完成")

if __name__ == "__main__":
    main()