"""
Clean DipMaster Model Trainer - No Data Leakage
严格防止数据泄漏的DipMaster模型训练器
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
import json
from datetime import datetime
import os

class CleanDipMasterTrainer:
    """
    清理版DipMaster训练器 - 严格防止数据泄漏
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.forbidden_keywords = [
            'future_return', 'target_', 'is_profitable', 'hits_',
            'forward_', 'next_', 'ahead_', 'future_'
        ]
        
    def load_and_clean_data(self):
        """加载数据并移除所有潜在的数据泄漏特征"""
        print("Loading and cleaning data...")
        
        df = pd.read_parquet(self.data_path)
        print(f"Original data shape: {df.shape}")
        
        # 识别并移除数据泄漏特征
        leakage_features = []
        for col in df.columns:
            for keyword in self.forbidden_keywords:
                if keyword in col.lower():
                    leakage_features.append(col)
                    break
        
        print(f"Identified leakage features ({len(leakage_features)}): {leakage_features}")
        
        # 保留的标签列
        label_cols = ['dipmaster_win']  # 只保留主要目标
        
        # 特征列 = 所有列 - 泄漏特征 - 标签列 - 元数据列
        meta_cols = ['timestamp', 'symbol']
        feature_cols = [col for col in df.columns 
                       if col not in leakage_features 
                       and col not in label_cols 
                       and col not in meta_cols]
        
        print(f"Clean feature count: {len(feature_cols)}")
        print(f"Sample features: {feature_cols[:10]}")
        
        # 构建清理后的数据集
        X = df[feature_cols].copy()
        y = df['dipmaster_win'].copy()
        
        # 处理缺失值
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 基础统计
        print(f"\nData Summary:")
        print(f"Samples: {len(X):,}")
        print(f"Features: {len(feature_cols)}")
        print(f"Target win rate: {y.mean():.4f}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_cols
    
    def create_time_aware_features(self, df):
        """创建时间感知特征（不包含未来信息）"""
        print("Creating time-aware features...")
        
        # 确保有timestamp列
        if 'timestamp' not in df.columns:
            print("Warning: No timestamp column found")
            return df
        
        df = df.copy()
        
        # 基础时间特征
        if 'hour' not in df.columns:
            df['hour'] = df.index % 288 // 12  # 假设5分钟数据，一天288个点
        
        if 'minute_in_hour' not in df.columns:
            df['minute_in_hour'] = (df.index % 12) * 5  # 5分钟间隔
        
        # 滞后特征（历史信息）
        price_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in price_cols:
            if col in df.columns:
                # 1-3期滞后
                for lag in [1, 2, 3]:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
                # 滚动统计（过去N期）
                for window in [5, 10, 20]:
                    df[f'{col}_sma_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_std_{window}'] = df[col].rolling(window).std()
        
        # 价格变化特征
        if 'close' in df.columns:
            df['return_1'] = df['close'].pct_change(1)
            df['return_3'] = df['close'].pct_change(3)
            df['return_5'] = df['close'].pct_change(5)
            
            # 波动率特征
            df['volatility_5'] = df['return_1'].rolling(5).std()
            df['volatility_20'] = df['return_1'].rolling(20).std()
        
        # 成交量特征
        if 'volume' in df.columns:
            df['volume_ratio_5'] = df['volume'] / df['volume'].rolling(5).mean()
            df['volume_ratio_20'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # 移除原始价格列以避免多重共线性
        cols_to_remove = [col for col in price_cols if col in df.columns]
        enhanced_df = df.drop(columns=cols_to_remove)
        
        # 移除包含NaN的行
        enhanced_df = enhanced_df.dropna()
        
        print(f"Enhanced features count: {enhanced_df.shape[1]}")
        return enhanced_df
    
    def feature_selection_clean(self, X, y, top_k=30):
        """清理版特征选择"""
        print(f"Starting clean feature selection from {X.shape[1]} features...")
        
        # 移除方差过低的特征
        var_threshold = 0.01
        feature_vars = X.var()
        low_var_features = feature_vars[feature_vars < var_threshold].index
        X_filtered = X.drop(columns=low_var_features)
        print(f"Removed {len(low_var_features)} low-variance features")
        
        # 互信息特征选择
        mi_scores = mutual_info_classif(X_filtered, y, random_state=42)
        mi_ranking = pd.Series(mi_scores, index=X_filtered.columns).sort_values(ascending=False)
        
        # 选择top_k特征
        selected_features = mi_ranking.head(top_k).index.tolist()
        
        print(f"Selected {len(selected_features)} features:")
        for i, (feature, score) in enumerate(mi_ranking.head(10).items()):
            print(f"  {i+1}. {feature}: {score:.4f}")
        
        return selected_features, mi_ranking
    
    def time_series_split_clean(self, X, y, n_splits=5):
        """清理版时序分割"""
        print(f"Creating {n_splits} time series splits...")
        
        n_samples = len(X)
        test_size = n_samples // (n_splits + 1)
        embargo_period = 12  # 1小时间隔，防止数据泄漏
        
        splits = []
        for i in range(n_splits):
            test_start = (i + 1) * test_size
            test_end = test_start + test_size
            
            if test_end > n_samples:
                break
            
            # 训练集结束于测试集开始前的embargo_period
            train_end = test_start - embargo_period
            
            if train_end <= test_size:  # 确保有足够的训练数据
                continue
            
            train_indices = np.arange(0, train_end)
            test_indices = np.arange(test_start, min(test_end, n_samples))
            
            splits.append((train_indices, test_indices))
            print(f"  Split {len(splits)}: Train[0:{train_end}], Test[{test_start}:{min(test_end, n_samples)}]")
        
        return splits
    
    def train_clean_models(self, X, y, selected_features):
        """训练清理版模型"""
        X_selected = X[selected_features]
        
        # 时序分割
        splits = self.time_series_split_clean(X_selected, y)
        
        if len(splits) == 0:
            print("Error: No valid splits created")
            return None
        
        # 最终训练/测试分割
        split_point = int(len(X_selected) * 0.8)
        X_train, X_test = X_selected.iloc[:split_point], X_selected.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        print(f"Final split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        results = {}
        
        # LightGBM训练
        print("\nTraining LightGBM...")
        lgbm_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,  # 较低学习率
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 50,  # 增加最小样本数
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'max_depth': 6,  # 限制深度
            'verbosity': -1,
            'random_state': 42
        }
        
        # 交叉验证
        cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(splits):
            X_cv_train, X_cv_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
            y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]
            
            train_data = lgb.Dataset(X_cv_train, label=y_cv_train)
            val_data = lgb.Dataset(X_cv_val, label=y_cv_val, reference=train_data)
            
            model = lgb.train(
                lgbm_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=200,  # 减少轮数
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
            )
            
            y_pred = model.predict(X_cv_val)
            auc = roc_auc_score(y_cv_val, y_pred)
            cv_scores.append(auc)
            print(f"  Fold {fold+1} AUC: {auc:.4f}")
        
        print(f"Cross-validation AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # 最终模型训练
        train_data = lgb.Dataset(X_train, label=y_train)
        lgbm_model = lgb.train(
            lgbm_params,
            train_data,
            num_boost_round=200,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        # 测试集评估
        lgbm_pred_proba = lgbm_model.predict(X_test)
        lgbm_pred = (lgbm_pred_proba > 0.5).astype(int)
        
        lgbm_auc = roc_auc_score(y_test, lgbm_pred_proba)
        lgbm_accuracy = np.mean(lgbm_pred == y_test)
        
        results['lgbm'] = {
            'model': lgbm_model,
            'cv_auc_mean': float(np.mean(cv_scores)),
            'cv_auc_std': float(np.std(cv_scores)),
            'test_auc': float(lgbm_auc),
            'test_accuracy': float(lgbm_accuracy),
            'predicted_win_rate': float(np.mean(lgbm_pred)),
            'feature_importance': dict(zip(selected_features, lgbm_model.feature_importance()))
        }
        
        print(f"LGBM Test Results:")
        print(f"  AUC: {lgbm_auc:.4f}")
        print(f"  Accuracy: {lgbm_accuracy:.4f}")
        print(f"  Predicted Win Rate: {np.mean(lgbm_pred):.4f}")
        
        # XGBoost训练
        print("\nTraining XGBoost...")
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            max_depth=5,  # 限制深度
            learning_rate=0.05,  # 较低学习率
            n_estimators=200,  # 减少估计器数量
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_weight=10,  # 增加最小权重
            random_state=42
        )
        
        xgb_model.fit(X_train, y_train)
        
        xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        xgb_pred = xgb_model.predict(X_test)
        
        xgb_auc = roc_auc_score(y_test, xgb_pred_proba)
        xgb_accuracy = np.mean(xgb_pred == y_test)
        
        results['xgb'] = {
            'model': xgb_model,
            'test_auc': float(xgb_auc),
            'test_accuracy': float(xgb_accuracy),
            'predicted_win_rate': float(np.mean(xgb_pred)),
            'feature_importance': dict(zip(selected_features, xgb_model.feature_importances_))
        }
        
        print(f"XGBoost Test Results:")
        print(f"  AUC: {xgb_auc:.4f}")
        print(f"  Accuracy: {xgb_accuracy:.4f}")
        print(f"  Predicted Win Rate: {np.mean(xgb_pred):.4f}")
        
        return results, X_test, y_test
    
    def realistic_performance_assessment(self, results, X_test, y_test):
        """现实性能评估"""
        print("\n=== Realistic Performance Assessment ===")
        
        # 选择最佳模型
        best_model_name = 'lgbm' if results['lgbm']['test_auc'] > results['xgb']['test_auc'] else 'xgb'
        best_model = results[best_model_name]['model']
        
        print(f"Using {best_model_name} for assessment")
        
        # 生成预测
        if best_model_name == 'lgbm':
            pred_proba = best_model.predict(X_test)
            predictions = (pred_proba > 0.5).astype(int)
        else:
            pred_proba = best_model.predict_proba(X_test)[:, 1]
            predictions = best_model.predict(X_test)
        
        # 不同置信度阈值的性能
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        threshold_results = {}
        
        for threshold in thresholds:
            high_conf_mask = pred_proba > threshold
            
            if np.sum(high_conf_mask) > 0:
                precision = np.mean(y_test[high_conf_mask])
                signal_count = np.sum(high_conf_mask)
                
                threshold_results[threshold] = {
                    'signal_count': int(signal_count),
                    'signal_rate': float(signal_count / len(y_test)),
                    'precision': float(precision),
                    'total_samples': int(len(y_test))
                }
                
                print(f"Threshold {threshold}: {signal_count} signals ({signal_count/len(y_test):.2%}), Precision: {precision:.4f}")
            else:
                threshold_results[threshold] = {
                    'signal_count': 0,
                    'signal_rate': 0.0,
                    'precision': 0.0,
                    'total_samples': int(len(y_test))
                }
        
        # 成本影响分析
        transaction_cost = 0.0017  # 0.17% 双边成本
        
        cost_analysis = {}
        for threshold, stats in threshold_results.items():
            if stats['signal_count'] > 0:
                # 假设每个信号平均收益0.1%（保守估计）
                gross_return_per_trade = 0.001
                net_return_per_trade = gross_return_per_trade - transaction_cost
                
                expected_return = stats['precision'] * net_return_per_trade - (1 - stats['precision']) * transaction_cost
                
                cost_analysis[threshold] = {
                    'gross_return_per_trade': gross_return_per_trade,
                    'net_return_per_trade': net_return_per_trade,
                    'expected_return': float(expected_return),
                    'breakeven_precision': float(transaction_cost / (gross_return_per_trade + transaction_cost))
                }
        
        return {
            'best_model': best_model_name,
            'threshold_analysis': threshold_results,
            'cost_analysis': cost_analysis,
            'base_performance': {
                'test_auc': float(results[best_model_name]['test_auc']),
                'test_accuracy': float(results[best_model_name]['test_accuracy']),
                'baseline_win_rate': float(y_test.mean())
            }
        }
    
    def generate_clean_recommendations(self, performance_results):
        """生成清理版建议"""
        print("\n=== Clean Model Recommendations ===")
        
        recommendations = []
        
        base_auc = performance_results['base_performance']['test_auc']
        baseline_win_rate = performance_results['base_performance']['baseline_win_rate']
        
        # AUC评估
        if base_auc < 0.6:
            recommendations.append("CRITICAL: AUC < 0.6 indicates weak predictive power")
            recommendations.append("   - Add more sophisticated features")
            recommendations.append("   - Consider ensemble methods")
        elif base_auc < 0.7:
            recommendations.append("MODERATE: AUC suggests modest predictive ability")
            recommendations.append("   - Feature engineering needed")
        else:
            recommendations.append("GOOD: AUC indicates reasonable predictive power")
        
        # 胜率分析
        if baseline_win_rate < 0.3:
            recommendations.append("CRITICAL: Very low base win rate")
            recommendations.append("   - Fundamental strategy review needed")
            recommendations.append("   - Consider different market conditions")
        
        # 成本分析
        best_threshold = None
        best_expected_return = -float('inf')
        
        for threshold, analysis in performance_results['cost_analysis'].items():
            if analysis['expected_return'] > best_expected_return:
                best_expected_return = analysis['expected_return']
                best_threshold = threshold
        
        if best_expected_return > 0:
            recommendations.append(f"PROFITABLE: Optimal threshold {best_threshold} with expected return {best_expected_return:.4f}")
        else:
            recommendations.append("UNPROFITABLE: No threshold yields positive expected returns")
            recommendations.append("   - Reduce trading costs or improve precision")
        
        # 具体改进建议
        recommendations.extend([
            "",
            "SPECIFIC IMPROVEMENTS:",
            "1. Feature Engineering:",
            "   - Add volatility regime indicators",
            "   - Include market microstructure features",
            "   - Implement cross-asset momentum",
            "",
            "2. Model Enhancement:",
            "   - Try ensemble methods",
            "   - Implement feature interactions",
            "   - Add temporal embeddings",
            "",
            "3. Execution Optimization:",
            "   - Use limit orders to reduce slippage",
            "   - Implement TWAP execution",
            "   - Add liquidity detection",
            "",
            "4. Risk Management:",
            "   - Add position sizing rules",
            "   - Implement stop-loss logic",
            "   - Monitor correlation risk"
        ])
        
        for rec in recommendations:
            print(rec)
        
        return recommendations
    
    def save_clean_results(self, results, performance_results, recommendations):
        """保存清理版结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = "results/model_training"
        os.makedirs(results_dir, exist_ok=True)
        
        # 准备保存的结果
        clean_results = {
            'metadata': {
                'timestamp': timestamp,
                'version': 'clean_v1.0',
                'data_leakage_removed': True,
                'description': 'Clean DipMaster model without data leakage'
            },
            'model_performance': {
                name: {
                    'test_auc': float(model_results.get('test_auc', 0)),
                    'test_accuracy': float(model_results.get('test_accuracy', 0)),
                    'predicted_win_rate': float(model_results.get('predicted_win_rate', 0)),
                    'cv_auc_mean': float(model_results.get('cv_auc_mean', 0)),
                    'cv_auc_std': float(model_results.get('cv_auc_std', 0))
                }
                for name, model_results in results.items()
                if isinstance(model_results, dict)
            },
            'performance_analysis': performance_results,
            'recommendations': recommendations,
            'feature_importance': {
                name: {k: float(v) for k, v in model_results['feature_importance'].items()}
                for name, model_results in results.items()
                if 'feature_importance' in model_results
            }
        }
        
        # 保存结果
        results_file = f"{results_dir}/clean_dipmaster_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"\nClean results saved to: {results_file}")
        return results_file
    
    def run_clean_pipeline(self):
        """运行清理版完整pipeline"""
        print("=== Clean DipMaster Training Pipeline ===")
        
        # 1. 加载并清理数据
        X, y, feature_cols = self.load_and_clean_data()
        
        # 2. 创建时间感知特征
        enhanced_df = self.create_time_aware_features(
            pd.concat([X, y], axis=1)
        )
        
        # 分离特征和标签
        y_enhanced = enhanced_df['dipmaster_win']
        X_enhanced = enhanced_df.drop(columns=['dipmaster_win'])
        
        print(f"Enhanced dataset: {X_enhanced.shape[0]} samples, {X_enhanced.shape[1]} features")
        
        # 3. 特征选择
        selected_features, feature_ranking = self.feature_selection_clean(X_enhanced, y_enhanced)
        
        # 4. 训练模型
        results, X_test, y_test = self.train_clean_models(X_enhanced, y_enhanced, selected_features)
        
        if results is None:
            print("Training failed!")
            return None
        
        # 5. 性能评估
        performance_results = self.realistic_performance_assessment(results, X_test, y_test)
        
        # 6. 生成建议
        recommendations = self.generate_clean_recommendations(performance_results)
        
        # 7. 保存结果
        results_file = self.save_clean_results(results, performance_results, recommendations)
        
        print("\n=== Pipeline Complete ===")
        return {
            'results': results,
            'performance': performance_results,
            'recommendations': recommendations,
            'results_file': results_file
        }

def main():
    """主函数"""
    data_path = "data/Enhanced_Features_V5_Clean_20250817_144045.parquet"
    
    trainer = CleanDipMasterTrainer(data_path)
    final_results = trainer.run_clean_pipeline()
    
    if final_results:
        print(f"\nTraining completed successfully!")
        print(f"Results saved to: {final_results['results_file']}")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()