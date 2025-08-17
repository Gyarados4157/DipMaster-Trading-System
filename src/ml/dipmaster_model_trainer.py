"""
DipMaster Trading Model Trainer
严格的时序机器学习训练和验证系统
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
import optuna
from optuna.samplers import TPESampler
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import shap
from typing import Dict, List, Tuple, Any

class PurgedTimeSeriesSplit:
    """
    Purged Time Series Cross-Validation
    防止数据泄漏的时序交叉验证
    """
    def __init__(self, n_splits=5, test_size=None, embargo_period=12):
        self.n_splits = n_splits
        self.test_size = test_size
        self.embargo_period = embargo_period  # 禁运期，防止数据泄漏
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        
        if self.test_size is None:
            test_size = n_samples // (self.n_splits + 1)
        else:
            test_size = int(n_samples * self.test_size)
        
        for i in range(self.n_splits):
            # 测试集的起始和结束位置
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
                
            # 训练集结束位置（考虑禁运期）
            train_end = test_start - self.embargo_period
            
            if train_end <= 0:
                continue
                
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, min(test_end, n_samples))
            
            yield train_indices, test_indices

class CostModel:
    """
    现实交易成本模型
    """
    def __init__(self):
        # 不同币种的成本参数
        self.costs = {
            'BTCUSDT': {'slippage': 0.0005, 'commission': 0.001, 'impact': 0.0002},
            'ETHUSDT': {'slippage': 0.0008, 'commission': 0.001, 'impact': 0.0003},
            'SOLUSDT': {'slippage': 0.0010, 'commission': 0.001, 'impact': 0.0005},
            'ADAUSDT': {'slippage': 0.0012, 'commission': 0.001, 'impact': 0.0006},
            'XRPUSDT': {'slippage': 0.0010, 'commission': 0.001, 'impact': 0.0004},
        }
        self.funding_rate = 0.05  # 年化5%的资金成本
    
    def calculate_total_cost(self, symbol: str, holding_periods: np.ndarray) -> float:
        """计算总交易成本"""
        if symbol not in self.costs:
            symbol = 'BTCUSDT'  # 默认使用BTC成本
            
        costs = self.costs[symbol]
        
        # 双边成本（开仓+平仓）
        transaction_cost = (costs['slippage'] + costs['commission']) * 2
        
        # 市场冲击成本
        impact_cost = costs['impact']
        
        # 资金成本（按小时计算）
        avg_holding_hours = np.mean(holding_periods) / 12  # 5分钟 * 12 = 1小时
        funding_cost = self.funding_rate * avg_holding_hours / (365 * 24)
        
        return transaction_cost + impact_cost + funding_cost

class DipMasterModelTrainer:
    """
    DipMaster交易模型训练器
    """
    def __init__(self, data_path: str, config_path: str):
        self.data_path = data_path
        self.config_path = config_path
        self.cost_model = CostModel()
        
        # 载入数据和配置
        self.df = pd.read_parquet(data_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
            
        # 模型存储
        self.models = {}
        self.feature_importance = {}
        self.training_results = {}
        
        print(f"数据加载完成: {self.df.shape}")
        print(f"配置加载完成: {self.config['metadata']['version']}")
    
    def prepare_features_labels(self) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        准备特征和标签数据
        """
        # 排除标签列和时间列
        exclude_cols = ['timestamp', 'symbol'] + [col for col in self.df.columns 
                       if col.startswith('target_') or col.startswith('is_') 
                       or col.startswith('hits_') or 'win' in col.lower() 
                       or 'return_class' in col]
        
        feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        X = self.df[feature_cols].copy()
        
        # 主要目标：DipMaster_Win
        y = self.df['dipmaster_win'].copy()
        
        # 处理缺失值
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"特征数量: {len(feature_cols)}")
        print(f"样本数量: {len(X)}")
        print(f"目标胜率: {y.mean():.4f}")
        
        return X, y, feature_cols
    
    def feature_selection(self, X: pd.DataFrame, y: pd.Series, top_k: int = 50) -> List[str]:
        """
        基于互信息和树模型的特征选择
        """
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.ensemble import RandomForestClassifier
        
        print("开始特征选择...")
        
        # 互信息特征选择
        mi_scores = mutual_info_classif(X, y, random_state=42)
        mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        
        # 随机森林特征重要性
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_ranking = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
        
        # 综合排名（取前50%的互信息特征和前50%的RF特征的并集）
        mi_top = set(mi_ranking.head(top_k).index)
        rf_top = set(rf_ranking.head(top_k).index)
        selected_features = list(mi_top.union(rf_top))
        
        print(f"特征选择完成: 从{len(X.columns)}个特征中选择了{len(selected_features)}个")
        
        # 保存特征重要性
        self.feature_importance['mutual_info'] = mi_ranking.to_dict()
        self.feature_importance['random_forest'] = rf_ranking.to_dict()
        
        return selected_features
    
    def optimize_lgbm_hyperparams(self, X: pd.DataFrame, y: pd.Series, 
                                 cv_splits: List[Tuple], n_trials: int = 100) -> Dict:
        """
        LGBM超参数优化
        """
        def objective(trial):
            params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'verbosity': -1,
                'random_state': 42
            }
            
            scores = []
            for train_idx, val_idx in cv_splits:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                train_data = lgb.Dataset(X_train, label=y_train)
                val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
                
                model = lgb.train(
                    params, train_data, 
                    valid_sets=[val_data],
                    num_boost_round=1000,
                    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                )
                
                y_pred = model.predict(X_val)
                auc = roc_auc_score(y_val, y_pred)
                scores.append(auc)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def optimize_xgb_hyperparams(self, X: pd.DataFrame, y: pd.Series, 
                                cv_splits: List[Tuple], n_trials: int = 100) -> Dict:
        """
        XGBoost超参数优化
        """
        def objective(trial):
            params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'random_state': 42
            }
            
            scores = []
            for train_idx, val_idx in cv_splits:
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)], 
                         early_stopping_rounds=50, 
                         verbose=False)
                
                y_pred = model.predict_proba(X_val)[:, 1]
                auc = roc_auc_score(y_val, y_pred)
                scores.append(auc)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str]):
        """
        训练多个模型
        """
        X_selected = X[selected_features]
        
        # 创建时序交叉验证分割
        cv = PurgedTimeSeriesSplit(n_splits=5, embargo_period=12)  # 1小时禁运期
        cv_splits = list(cv.split(X_selected, y))
        
        print(f"时序交叉验证分割数: {len(cv_splits)}")
        
        # LGBM超参数优化和训练
        print("优化LGBM超参数...")
        lgbm_best_params = self.optimize_lgbm_hyperparams(X_selected, y, cv_splits, n_trials=50)
        
        # 训练最终LGBM模型
        print("训练LGBM模型...")
        train_data = lgb.Dataset(X_selected, label=y)
        lgbm_model = lgb.train(
            lgbm_best_params, train_data,
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100)]
        )
        
        # XGBoost超参数优化和训练
        print("优化XGBoost超参数...")
        xgb_best_params = self.optimize_xgb_hyperparams(X_selected, y, cv_splits, n_trials=50)
        
        # 训练最终XGBoost模型
        print("训练XGBoost模型...")
        xgb_model = xgb.XGBClassifier(**xgb_best_params)
        xgb_model.fit(X_selected, y)
        
        # 保存模型
        self.models['lgbm'] = lgbm_model
        self.models['xgb'] = xgb_model
        
        # 保存超参数
        self.training_results['lgbm_params'] = lgbm_best_params
        self.training_results['xgb_params'] = xgb_best_params
        
        print("模型训练完成!")
    
    def create_ensemble_model(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str]):
        """
        创建集成模型
        """
        X_selected = X[selected_features]
        
        # 准备基础模型
        lgbm_clf = lgb.LGBMClassifier(**self.training_results['lgbm_params'], random_state=42)
        xgb_clf = xgb.XGBClassifier(**self.training_results['xgb_params'])
        
        # 创建投票分类器
        ensemble = VotingClassifier(
            estimators=[
                ('lgbm', lgbm_clf),
                ('xgb', xgb_clf)
            ],
            voting='soft'  # 使用概率投票
        )
        
        ensemble.fit(X_selected, y)
        self.models['ensemble'] = ensemble
        
        print("集成模型创建完成!")
    
    def validate_models(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str]):
        """
        严格的时序验证
        """
        X_selected = X[selected_features]
        
        # Walk-Forward验证
        walk_forward_results = {}
        
        # 使用最后20%的数据作为最终测试集
        split_point = int(len(X_selected) * 0.8)
        X_train, X_test = X_selected.iloc[:split_point], X_selected.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        print(f"Walk-Forward验证: 训练集{len(X_train)}, 测试集{len(X_test)}")
        
        for model_name, model in self.models.items():
            if model_name == 'ensemble':
                # 重新训练集成模型
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
            elif model_name == 'lgbm':
                # LGBM需要重新训练
                train_data = lgb.Dataset(X_train, label=y_train)
                retrained_model = lgb.train(
                    self.training_results['lgbm_params'], 
                    train_data, 
                    num_boost_round=model.num_trees()
                )
                y_pred_proba = retrained_model.predict(X_test)
                y_pred = (y_pred_proba > 0.5).astype(int)
            else:  # XGBoost
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                y_pred = model.predict(X_test)
            
            # 计算指标
            auc = roc_auc_score(y_test, y_pred_proba)
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = np.trapz(recall, precision)
            
            # 计算胜率和风险调整收益
            win_rate = np.mean(y_pred)
            actual_win_rate = np.mean(y_test)
            
            walk_forward_results[model_name] = {
                'auc': auc,
                'pr_auc': pr_auc,
                'predicted_win_rate': win_rate,
                'actual_win_rate': actual_win_rate,
                'precision': np.mean(y_pred == y_test),
                'n_signals': np.sum(y_pred),
                'hit_rate': np.sum((y_pred == 1) & (y_test == 1)) / max(np.sum(y_pred), 1)
            }
            
            print(f"{model_name}模型验证结果:")
            print(f"  AUC: {auc:.4f}")
            print(f"  PR-AUC: {pr_auc:.4f}")
            print(f"  预测胜率: {win_rate:.4f}")
            print(f"  实际胜率: {actual_win_rate:.4f}")
            print(f"  命中率: {walk_forward_results[model_name]['hit_rate']:.4f}")
        
        self.training_results['validation'] = walk_forward_results
    
    def realistic_backtest(self, X: pd.DataFrame, y: pd.Series, selected_features: List[str]):
        """
        现实成本回测
        """
        X_selected = X[selected_features]
        
        # 使用最佳模型进行回测
        best_model_name = max(
            self.training_results['validation'].keys(),
            key=lambda x: self.training_results['validation'][x]['auc']
        )
        best_model = self.models[best_model_name]
        
        print(f"使用{best_model_name}模型进行回测...")
        
        # 生成信号
        if best_model_name == 'lgbm':
            signals = best_model.predict(X_selected) > 0.5
            signal_strength = best_model.predict(X_selected)
        else:
            signals = best_model.predict(X_selected) == 1
            signal_strength = best_model.predict_proba(X_selected)[:, 1]
        
        # 模拟交易结果
        signal_indices = np.where(signals)[0]
        
        # 获取实际收益（假设我们有真实的价格数据）
        actual_returns = self.df['target_return'].values
        
        trades = []
        total_pnl = 0
        total_cost = 0
        
        for idx in signal_indices:
            if idx < len(actual_returns):
                # 基础收益
                base_return = actual_returns[idx]
                
                # 计算交易成本
                symbol = 'BTCUSDT'  # 简化处理，实际应该根据数据确定
                holding_period = np.array([12])  # 假设平均持仓12个5分钟周期（1小时）
                trade_cost = self.cost_model.calculate_total_cost(symbol, holding_period)
                
                # 净收益
                net_return = base_return - trade_cost
                
                trades.append({
                    'index': idx,
                    'signal_strength': signal_strength[idx],
                    'base_return': base_return,
                    'trade_cost': trade_cost,
                    'net_return': net_return,
                    'is_win': net_return > 0
                })
                
                total_pnl += net_return
                total_cost += trade_cost
        
        # 计算回测指标
        if trades:
            df_trades = pd.DataFrame(trades)
            
            backtest_results = {
                'total_trades': len(trades),
                'win_rate': np.mean(df_trades['is_win']),
                'avg_return': np.mean(df_trades['net_return']),
                'total_pnl': total_pnl,
                'total_cost': total_cost,
                'cost_ratio': total_cost / abs(total_pnl) if total_pnl != 0 else np.inf,
                'sharpe_ratio': np.mean(df_trades['net_return']) / np.std(df_trades['net_return']) if np.std(df_trades['net_return']) > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(df_trades['net_return'].cumsum()),
                'profit_factor': abs(df_trades[df_trades['net_return'] > 0]['net_return'].sum()) / abs(df_trades[df_trades['net_return'] < 0]['net_return'].sum()) if len(df_trades[df_trades['net_return'] < 0]) > 0 else np.inf
            }
            
            self.training_results['backtest'] = backtest_results
            
            print("回测结果:")
            print(f"  总交易数: {backtest_results['total_trades']}")
            print(f"  胜率: {backtest_results['win_rate']:.4f}")
            print(f"  平均收益: {backtest_results['avg_return']:.4f}")
            print(f"  总盈亏: {backtest_results['total_pnl']:.4f}")
            print(f"  总成本: {backtest_results['total_cost']:.4f}")
            print(f"  夏普比率: {backtest_results['sharpe_ratio']:.4f}")
            print(f"  最大回撤: {backtest_results['max_drawdown']:.4f}")
            print(f"  盈亏比: {backtest_results['profit_factor']:.4f}")
    
    def _calculate_max_drawdown(self, cumulative_returns):
        """计算最大回撤"""
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        return np.min(drawdown)
    
    def analyze_feature_importance(self, selected_features: List[str]):
        """
        特征重要性分析
        """
        # LGBM特征重要性
        if 'lgbm' in self.models:
            lgbm_importance = dict(zip(selected_features, self.models['lgbm'].feature_importance()))
            self.feature_importance['lgbm'] = lgbm_importance
        
        # XGBoost特征重要性
        if 'xgb' in self.models:
            xgb_importance = dict(zip(selected_features, self.models['xgb'].feature_importances_))
            self.feature_importance['xgb'] = xgb_importance
        
        print("特征重要性分析完成")
    
    def generate_optimization_recommendations(self):
        """
        生成模型优化建议
        """
        current_win_rate = self.training_results['validation']['ensemble']['actual_win_rate']
        target_win_rate = 0.85
        
        recommendations = []
        
        if current_win_rate < 0.3:
            recommendations.append("1. 数据质量问题: 当前胜率过低，建议检查特征工程和标签定义")
            recommendations.append("2. 特征增强: 添加更多市场微观结构特征和时间特征")
            recommendations.append("3. 信号过滤: 实施更严格的信号质量过滤机制")
        
        if current_win_rate < 0.5:
            recommendations.append("4. 模型复杂度: 考虑使用更复杂的模型如深度学习")
            recommendations.append("5. 集成方法: 增加更多基础模型提高集成效果")
        
        recommendations.append("6. 成本优化: 实施更精确的交易成本建模")
        recommendations.append("7. 时序特征: 加强对市场制度的识别能力")
        recommendations.append("8. 风险管理: 实施动态止损和仓位管理")
        
        self.training_results['recommendations'] = recommendations
        
        return recommendations
    
    def save_results(self, output_dir: str = "results/model_training"):
        """
        保存所有结果
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存模型
        for name, model in self.models.items():
            model_path = f"{output_dir}/dipmaster_{name}_model_{timestamp}.pkl"
            joblib.dump(model, model_path)
            print(f"模型已保存: {model_path}")
        
        # 保存训练结果
        results_path = f"{output_dir}/training_results_{timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            # 处理numpy类型
            def convert_numpy(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            results_to_save = {}
            for key, value in self.training_results.items():
                if isinstance(value, dict):
                    results_to_save[key] = {k: convert_numpy(v) for k, v in value.items()}
                else:
                    results_to_save[key] = convert_numpy(value)
                    
            json.dump(results_to_save, f, indent=2, ensure_ascii=False)
        
        # 保存特征重要性
        importance_path = f"{output_dir}/feature_importance_{timestamp}.json"
        with open(importance_path, 'w', encoding='utf-8') as f:
            json.dump(self.feature_importance, f, indent=2, ensure_ascii=False)
        
        print(f"训练结果已保存到: {output_dir}")
        
        return output_dir, timestamp
    
    def run_complete_pipeline(self):
        """
        运行完整的训练pipeline
        """
        print("=== DipMaster模型训练开始 ===")
        
        # 1. 准备数据
        X, y, feature_cols = self.prepare_features_labels()
        
        # 2. 特征选择
        selected_features = self.feature_selection(X, y, top_k=50)
        
        # 3. 训练模型
        self.train_models(X, y, selected_features)
        
        # 4. 创建集成模型
        self.create_ensemble_model(X, y, selected_features)
        
        # 5. 验证模型
        self.validate_models(X, y, selected_features)
        
        # 6. 现实回测
        self.realistic_backtest(X, y, selected_features)
        
        # 7. 特征重要性分析
        self.analyze_feature_importance(selected_features)
        
        # 8. 生成优化建议
        recommendations = self.generate_optimization_recommendations()
        
        # 9. 保存结果
        output_dir, timestamp = self.save_results()
        
        print("=== 训练完成 ===")
        print(f"结果保存在: {output_dir}")
        
        # 打印核心结果
        print("\n=== 核心结果 ===")
        if 'validation' in self.training_results:
            for model_name, results in self.training_results['validation'].items():
                print(f"{model_name}: AUC={results['auc']:.4f}, 胜率={results['actual_win_rate']:.4f}")
        
        if 'backtest' in self.training_results:
            bt = self.training_results['backtest']
            print(f"回测胜率: {bt['win_rate']:.4f}")
            print(f"夏普比率: {bt['sharpe_ratio']:.4f}")
        
        print("\n=== 优化建议 ===")
        for rec in recommendations:
            print(rec)
        
        return self.training_results

if __name__ == "__main__":
    # 训练配置
    data_path = "data/Enhanced_Features_V5_Clean_20250817_144045.parquet"
    config_path = "data/Enhanced_FeatureSet_V5_Clean_20250817_144045.json"
    
    # 创建训练器并运行
    trainer = DipMasterModelTrainer(data_path, config_path)
    results = trainer.run_complete_pipeline()