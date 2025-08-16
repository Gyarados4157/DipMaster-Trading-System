"""
DipMaster Enhanced V4 特征使用示例
展示如何加载和使用生成的特征集进行机器学习建模
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

class DipMasterFeatureLoader:
    """DipMaster特征加载器"""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.features_df = None
        self.metadata = None
        
    def load_latest_features(self):
        """加载最新的特征文件"""
        # 查找最新的特征文件
        feature_files = list(self.data_path.glob("dipmaster_v4_features_*.parquet"))
        metadata_files = list(self.data_path.glob("FeatureSet_*.json"))
        
        if not feature_files:
            raise FileNotFoundError("No feature files found!")
        
        # 加载最新文件
        latest_feature_file = max(feature_files, key=lambda x: x.stat().st_mtime)
        latest_metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
        
        print(f"Loading features from: {latest_feature_file.name}")
        self.features_df = pd.read_parquet(latest_feature_file)
        
        with open(latest_metadata_file, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {len(self.features_df)} samples with {len(self.features_df.columns)} features")
        return self.features_df, self.metadata
    
    def get_feature_categories(self):
        """获取特征分类"""
        if self.metadata:
            return self.metadata['feature_categories']
        return {}
    
    def prepare_modeling_data(self, target_label='is_profitable_15m'):
        """准备建模数据"""
        if self.features_df is None:
            raise ValueError("Features not loaded. Call load_latest_features() first.")
        
        # 选择核心特征 (根据重要性排序)
        core_features = [
            'rsi', 'dipmaster_signal_strength', 'bb_position',
            'momentum_20', 'momentum_10', 'momentum_5',
            'volume_ratio', 'volatility_20', 'trend_alignment',
            'order_flow_imbalance', 'hour', 'bb_squeeze',
            'volatility_10', 'turnover_rate', 'price_impact'
        ]
        
        # 过滤存在的特征
        available_features = [f for f in core_features if f in self.features_df.columns]
        
        print(f"Using {len(available_features)} features for modeling:")
        for i, feature in enumerate(available_features, 1):
            print(f"  {i:2d}. {feature}")
        
        # 准备特征和标签
        X = self.features_df[available_features].copy()
        y = self.features_df[target_label].copy()
        
        # 移除缺失值
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask].reset_index(drop=True)
        y_clean = y[mask].reset_index(drop=True)
        timestamps = self.features_df[mask]['timestamp'].reset_index(drop=True)
        
        print(f"Clean data: {len(X_clean)} samples ({mask.mean():.1%} of total)")
        
        return X_clean, y_clean, timestamps, available_features

class DipMasterModelTrainer:
    """DipMaster模型训练器"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        
    def train_model(self, X, y, timestamps, test_size=0.2):
        """训练模型 - 使用时间序列分割"""
        
        # 按时间排序
        sort_idx = timestamps.argsort()
        X_sorted = X.iloc[sort_idx].reset_index(drop=True)
        y_sorted = y.iloc[sort_idx].reset_index(drop=True)
        timestamps_sorted = timestamps.iloc[sort_idx].reset_index(drop=True)
        
        # 时间分割
        split_idx = int(len(X_sorted) * (1 - test_size))
        
        X_train = X_sorted.iloc[:split_idx]
        y_train = y_sorted.iloc[:split_idx]
        X_test = X_sorted.iloc[split_idx:]
        y_test = y_sorted.iloc[split_idx:]
        
        train_start = timestamps_sorted.iloc[0]
        train_end = timestamps_sorted.iloc[split_idx-1]
        test_start = timestamps_sorted.iloc[split_idx]
        test_end = timestamps_sorted.iloc[-1]
        
        print(f"\nTime-based split:")
        print(f"Training period: {train_start} to {train_end}")
        print(f"Testing period:  {test_start} to {test_end}")
        print(f"Training samples: {len(X_train):,}")
        print(f"Testing samples:  {len(X_test):,}")
        
        # 特征标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 训练随机森林模型
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=1000,
            min_samples_leaf=500,
            random_state=42,
            n_jobs=-1
        )
        
        print("\nTraining Random Forest model...")
        self.model.fit(X_train_scaled, y_train)
        
        # 预测和评估
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE EVALUATION")
        print("="*60)
        
        print("\nTraining Set Performance:")
        print(classification_report(y_train, y_train_pred))
        
        print("\nTest Set Performance:")
        print(classification_report(y_test, y_test_pred))
        
        # 混淆矩阵
        print("\nTest Set Confusion Matrix:")
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        
        # 计算策略指标
        test_accuracy = (y_test_pred == y_test).mean()
        test_precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        test_recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        
        print(f"\nKey Metrics:")
        print(f"Test Accuracy:  {test_accuracy:.3f} ({test_accuracy:.1%})")
        print(f"Test Precision: {test_precision:.3f} ({test_precision:.1%})")
        print(f"Test Recall:    {test_recall:.3f} ({test_recall:.1%})")
        
        return {
            'model': self.model,
            'scaler': self.scaler,
            'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test,
            'y_train_pred': y_train_pred,
            'y_test_pred': y_test_pred,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall
        }
    
    def get_feature_importance(self, X, feature_names):
        """获取特征重要性"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE RANKING")
        print("="*50)
        
        for i, row in importance_df.iterrows():
            print(f"{len(importance_df) - i:2d}. {row['feature']:25s} {row['importance']:.4f}")
        
        return importance_df

def analyze_signal_quality(features_df, predictions):
    """分析信号质量"""
    
    # 计算DipMaster信号强度分布
    signal_strength = features_df['dipmaster_signal_strength']
    
    print("\n" + "="*50)
    print("DIPMASTER SIGNAL ANALYSIS")
    print("="*50)
    
    print(f"Signal Strength Statistics:")
    print(f"  Mean: {signal_strength.mean():.3f}")
    print(f"  Std:  {signal_strength.std():.3f}")
    print(f"  Min:  {signal_strength.min():.3f}")
    print(f"  Max:  {signal_strength.max():.3f}")
    
    # 按信号强度分层分析
    high_signal = signal_strength > 0.7
    medium_signal = (signal_strength > 0.4) & (signal_strength <= 0.7)
    low_signal = signal_strength <= 0.4
    
    print(f"\nSignal Distribution:")
    print(f"  High Signal (>0.7):   {high_signal.mean():.1%}")
    print(f"  Medium Signal (0.4-0.7): {medium_signal.mean():.1%}")
    print(f"  Low Signal (<=0.4):    {low_signal.mean():.1%}")
    
    # 按时间分析信号分布
    features_df['hour'] = pd.to_datetime(features_df['timestamp']).dt.hour
    hourly_signal = features_df.groupby('hour')['dipmaster_signal_strength'].mean()
    
    print(f"\nHourly Signal Strength (Top 5):")
    top_hours = hourly_signal.sort_values(ascending=False).head()
    for hour, strength in top_hours.items():
        print(f"  Hour {hour:2d}: {strength:.3f}")

def main():
    """主函数 - 特征使用示例"""
    print("DipMaster Enhanced V4 Feature Usage Example")
    print("="*60)
    
    # 1. 加载特征
    data_path = "G:/Github/Quant/DipMaster-Trading-System/data"
    loader = DipMasterFeatureLoader(data_path)
    
    try:
        features_df, metadata = loader.load_latest_features()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # 2. 准备建模数据
    print(f"\nPreparing data for modeling...")
    X, y, timestamps, feature_names = loader.prepare_modeling_data(target_label='is_profitable_15m')
    
    print(f"\nTarget distribution:")
    print(f"  Profitable (1): {y.mean():.1%}")
    print(f"  Unprofitable (0): {(1-y.mean()):.1%}")
    
    # 3. 训练模型
    trainer = DipMasterModelTrainer()
    results = trainer.train_model(X, y, timestamps, test_size=0.2)
    
    # 4. 特征重要性分析
    importance_df = trainer.get_feature_importance(X, feature_names)
    
    # 5. 信号质量分析
    analyze_signal_quality(features_df, results['y_test_pred'])
    
    # 6. 实际策略信号生成示例
    print("\n" + "="*50)
    print("STRATEGY SIGNAL GENERATION EXAMPLE")
    print("="*50)
    
    # 使用模型预测生成交易信号
    X_scaled = trainer.scaler.transform(X)
    probabilities = results['model'].predict_proba(X_scaled)[:, 1]  # 获取正类概率
    
    # 设置信号阈值
    signal_threshold = 0.6  # 60%概率阈值
    trading_signals = probabilities > signal_threshold
    
    print(f"Signal generation with {signal_threshold:.0%} probability threshold:")
    print(f"  Total signals: {trading_signals.sum():,} ({trading_signals.mean():.2%} of samples)")
    
    # 如果有信号，计算信号质量
    if trading_signals.sum() > 0:
        signal_accuracy = y[trading_signals].mean()
        print(f"  Signal accuracy: {signal_accuracy:.1%}")
        
        # 不同时间段的信号表现
        signal_df = pd.DataFrame({
            'timestamp': timestamps,
            'signal': trading_signals,
            'actual': y,
            'hour': pd.to_datetime(timestamps).dt.hour
        })
        
        hourly_performance = signal_df[signal_df['signal']].groupby('hour')['actual'].agg(['count', 'mean']).round(3)
        hourly_performance = hourly_performance[hourly_performance['count'] >= 10]  # 至少10个信号
        
        print(f"\nBest performing hours (with ≥10 signals):")
        if len(hourly_performance) > 0:
            top_hours = hourly_performance.sort_values('mean', ascending=False).head()
            for hour, row in top_hours.iterrows():
                print(f"  Hour {hour:2d}: {row['mean']:.1%} accuracy ({int(row['count'])} signals)")
    
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✓ Loaded {len(features_df):,} samples with {len(features_df.columns)} features")
    print(f"✓ Used {len(feature_names)} core features for modeling")
    print(f"✓ Achieved {results['test_accuracy']:.1%} test accuracy")
    print(f"✓ Generated {trading_signals.sum():,} trading signals")
    print(f"")
    print(f"The feature set is ready for DipMaster Enhanced V4 strategy implementation!")
    print(f"Consider further optimization based on feature importance and signal analysis.")

if __name__ == "__main__":
    main()