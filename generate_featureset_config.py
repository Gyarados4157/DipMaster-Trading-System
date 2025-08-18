#!/usr/bin/env python3
"""
Generate FeatureSet Configuration for SuperDip Pin Bar Strategy
为超跌接针策略生成FeatureSet配置文件
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_features(feature_file: str) -> dict:
    """Analyze features from a parquet file"""
    try:
        df = pd.read_parquet(feature_file)
        
        # Basic statistics
        analysis = {
            'record_count': int(len(df)),
            'feature_count': int(len(df.columns)),
            'data_range': {
                'start_date': df.index.min().isoformat(),
                'end_date': df.index.max().isoformat()
            },
            'missing_data_percentage': float(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100),
            'feature_categories': {
                'core_technical': [],
                'pin_bar_pattern': [],
                'volume_profile': [],
                'momentum': [],
                'volatility': [],
                'labels': []
            }
        }
        
        # Categorize features
        for col in df.columns:
            if any(x in col.lower() for x in ['rsi', 'ma_', 'bb_', 'bollinger']):
                analysis['feature_categories']['core_technical'].append(col)
            elif any(x in col.lower() for x in ['pin_bar', 'wick', 'body', 'recovery']):
                analysis['feature_categories']['pin_bar_pattern'].append(col)
            elif any(x in col.lower() for x in ['volume', 'vpt']):
                analysis['feature_categories']['volume_profile'].append(col)
            elif any(x in col.lower() for x in ['momentum', 'roc', 'macd']):
                analysis['feature_categories']['momentum'].append(col)
            elif any(x in col.lower() for x in ['volatility', 'atr']):
                analysis['feature_categories']['volatility'].append(col)
            elif any(x in col.lower() for x in ['forward', 'win_', 'risk_adj', 'mfe', 'mae']):
                analysis['feature_categories']['labels'].append(col)
        
        # Sample pin bar statistics
        if any('is_pin_bar' in col for col in df.columns):
            pin_bar_col = [col for col in df.columns if 'is_pin_bar' in col][0]
            pin_bar_rate = float(df[pin_bar_col].mean())
            analysis['pin_bar_detection_rate'] = pin_bar_rate
        
        # Sample RSI oversold statistics
        rsi_oversold_cols = [col for col in df.columns if 'rsi' in col and 'oversold' in col]
        if rsi_oversold_cols:
            oversold_rates = {}
            for col in rsi_oversold_cols:
                oversold_rates[col] = float(df[col].mean())
            analysis['oversold_detection_rates'] = oversold_rates
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error analyzing {feature_file}: {e}")
        return {}

def generate_feature_set_config():
    """Generate comprehensive FeatureSet configuration"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Find all feature files
    feature_files = list(Path("data").glob("features_*_superdip_pinbar_*.parquet"))
    
    if not feature_files:
        logger.error("No feature files found")
        return
    
    logger.info(f"Found {len(feature_files)} feature files")
    
    # Analyze each file
    symbols_analysis = {}
    all_feature_categories = {
        'core_technical': set(),
        'pin_bar_pattern': set(),
        'volume_profile': set(),
        'momentum': set(),
        'volatility': set(),
        'labels': set()
    }
    
    for file_path in feature_files:
        # Extract symbol from filename
        symbol = file_path.name.split('_')[1]  # features_SYMBOL_superdip_pinbar_timestamp.parquet
        
        logger.info(f"Analyzing {symbol}...")
        analysis = analyze_features(str(file_path))
        
        if analysis:
            symbols_analysis[symbol] = {
                'file_path': str(file_path),
                'analysis': analysis
            }
            
            # Aggregate feature categories
            for category, features in analysis['feature_categories'].items():
                all_feature_categories[category].update(features)
    
    # Convert sets to lists for JSON serialization
    for category in all_feature_categories:
        all_feature_categories[category] = list(all_feature_categories[category])
    
    # Create comprehensive FeatureSet configuration
    feature_set_config = {
        "version": datetime.now().isoformat(),
        "feature_set_id": f"superdip_pinbar_complete_{timestamp}",
        "strategy_name": "SuperDip_PinBar_Complete_Strategy",
        "description": "超跌接针反转策略完整特征集 - 包含核心技术指标、接针形态识别、多时间框架特征和前向标签，专门用于捕捉市场超跌反转机会",
        
        "metadata": {
            "creation_date": datetime.now().isoformat(),
            "feature_engineer": "SuperDipPinBarDemo",
            "version": "1.0.0-complete",
            "total_symbols": len(symbols_analysis),
            "symbols_processed": list(symbols_analysis.keys()),
            "total_features": sum(len(cat) for cat in all_feature_categories.values()),
            "avg_records_per_symbol": int(np.mean([data['analysis']['record_count'] for data in symbols_analysis.values()])) if symbols_analysis else 0
        },
        
        "strategy_overview": {
            "objective": "识别超跌市场中的接针反转形态，实现高胜率的短期交易机会",
            "key_signals": [
                "RSI(14,7,21) 超跌区域识别 (RSI < 30)",
                "价格低于MA20的偏离度计算",
                "布林带下轨突破确认超跌状态", 
                "接针形态识别：下影线长度 > 2倍实体",
                "成交量放大确认有效突破",
                "4小时前向收益目标：0.8%-2.5%"
            ],
            "timeframe": "5分钟主图，多时间框架确认",
            "holding_period": "典型持仓4小时，最大240分钟"
        },
        
        "feature_engineering_details": {
            "core_technical_indicators": {
                "description": "核心技术指标用于超跌识别",
                "features": {
                    "rsi_multi_period": "RSI(7,14,21) 多周期超跌识别",
                    "moving_averages": "MA(10,20,50) 价格位置和偏离度",
                    "bollinger_bands": "布林带位置和超跌区域判断",
                    "volume_analysis": "成交量相对强度和放大倍数检测"
                },
                "feature_count": len(all_feature_categories['core_technical'])
            },
            
            "pin_bar_pattern_detection": {
                "description": "接针形态特征提取和识别",
                "features": {
                    "candlestick_ratios": "下影线/实体比率、上影线比率、实体比率",
                    "body_position": "蜡烛实体在K线中的位置",
                    "price_recovery": "从最低点的价格恢复程度",
                    "pattern_strength": "接针形态强度评分(0-1)",
                    "enhanced_detection": "结合成交量确认的增强接针识别"
                },
                "detection_criteria": {
                    "lower_wick_ratio": "> 50%",
                    "body_ratio": "< 30%", 
                    "upper_wick_ratio": "< 20%",
                    "volume_confirmation": "> 1.2x平均成交量"
                },
                "feature_count": len(all_feature_categories['pin_bar_pattern'])
            },
            
            "volume_microstructure": {
                "description": "成交量微观结构分析",
                "features": {
                    "volume_relative_strength": "多周期成交量相对强度",
                    "volume_spikes": "成交量异常放大检测",
                    "volume_price_trend": "量价趋势指标(VPT)"
                },
                "feature_count": len(all_feature_categories['volume_profile'])
            },
            
            "momentum_indicators": {
                "description": "价格动量和趋势确认",
                "features": {
                    "price_momentum": "多周期价格动量(3,5,10,15分钟)",
                    "rate_of_change": "变化率指标(ROC)",
                    "macd_analysis": "MACD及其信号线、柱状图"
                },
                "feature_count": len(all_feature_categories['momentum'])
            },
            
            "volatility_measures": {
                "description": "波动率和风险度量",
                "features": {
                    "rolling_volatility": "滚动波动率(10,20,30周期)",
                    "average_true_range": "平均真实波幅(ATR)"
                },
                "feature_count": len(all_feature_categories['volatility'])
            }
        },
        
        "target_definitions": {
            "forward_returns": {
                "type": "regression",
                "description": "4小时前向收益率",
                "horizon": "240分钟(48个5分钟周期)",
                "calculation": "(future_price / current_price) - 1"
            },
            "profit_achievement_labels": {
                "type": "classification", 
                "targets": [
                    {"target": 0.008, "description": "0.8%利润目标"},
                    {"target": 0.015, "description": "1.5%利润目标"},
                    {"target": 0.025, "description": "2.5%利润目标"}
                ],
                "stop_loss": 0.006,
                "logic": "在指定时间内达到利润目标且未触发止损"
            },
            "risk_metrics": {
                "mfe": "最大有利价格偏移",
                "mae": "最大不利价格偏移", 
                "risk_adjusted_return": "波动率调整收益"
            }
        },
        
        "feature_categories": dict(all_feature_categories),
        
        "data_files": {
            symbol: data['file_path'] for symbol, data in symbols_analysis.items()
        },
        
        "symbols_analysis": {
            symbol: {
                "file_path": data['file_path'],
                "record_count": data['analysis']['record_count'],
                "feature_count": data['analysis']['feature_count'],
                "data_range": data['analysis']['data_range'],
                "missing_data_percentage": data['analysis']['missing_data_percentage'],
                "pin_bar_detection_rate": data['analysis'].get('pin_bar_detection_rate', 0),
                "oversold_detection_rates": data['analysis'].get('oversold_detection_rates', {})
            }
            for symbol, data in symbols_analysis.items()
        },
        
        "model_development_guidelines": {
            "recommended_algorithms": [
                {
                    "algorithm": "LightGBM",
                    "reason": "处理表格数据效果优秀，特征重要性解释性强",
                    "hyperparameters": {
                        "objective": "binary/regression",
                        "metric": "auc/rmse", 
                        "num_leaves": 31,
                        "learning_rate": 0.05,
                        "feature_fraction": 0.8
                    }
                },
                {
                    "algorithm": "XGBoost", 
                    "reason": "强大的梯度提升，适合金融时间序列",
                    "hyperparameters": {
                        "max_depth": 6,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8
                    }
                },
                {
                    "algorithm": "Random Forest",
                    "reason": "稳定的基线模型，特征重要性分析",
                    "hyperparameters": {
                        "n_estimators": 200,
                        "max_depth": 10,
                        "min_samples_split": 20
                    }
                }
            ],
            
            "feature_selection_strategy": {
                "method": "多阶段特征选择",
                "steps": [
                    "1. 移除高缺失值特征 (>5%)",
                    "2. 移除高相关性特征 (>0.95)",
                    "3. 基于mutual_info_regression筛选TOP50",
                    "4. 递归特征消除(RFE)优化到TOP30",
                    "5. 前向特征选择验证最终特征集"
                ]
            },
            
            "cross_validation_strategy": {
                "method": "时序交叉验证",
                "splits": 5,
                "train_ratio": 0.7,
                "validation_ratio": 0.15,
                "test_ratio": 0.15,
                "gap": "24小时间隔避免数据泄漏",
                "purging": "删除重叠时间窗口"
            },
            
            "preprocessing_pipeline": {
                "missing_values": "前向填充 + 后向填充",
                "outliers": "RobustScaler处理异常值",
                "scaling": "StandardScaler标准化",
                "feature_engineering": "多项式特征 + 交互特征"
            }
        },
        
        "performance_benchmarks": {
            "target_metrics": {
                "classification": {
                    "accuracy": "> 65%",
                    "precision": "> 70%", 
                    "recall": "> 60%",
                    "f1_score": "> 65%",
                    "auc_roc": "> 0.75"
                },
                "regression": {
                    "rmse": "< 0.02",
                    "mae": "< 0.015",
                    "r2_score": "> 0.3"
                },
                "trading_metrics": {
                    "win_rate": "> 60%",
                    "sharpe_ratio": "> 1.5",
                    "max_drawdown": "< 10%",
                    "profit_factor": "> 2.0"
                }
            },
            
            "expected_signal_characteristics": {
                "daily_signals": "5-15个交易信号",
                "signal_quality": "高质量信号，低假阳性",
                "market_conditions": "适合震荡和弱势市场",
                "risk_profile": "中低风险，适合稳健交易"
            }
        },
        
        "deployment_considerations": {
            "data_requirements": {
                "update_frequency": "5分钟实时更新",
                "lookback_period": "至少200个数据点",
                "data_quality": "OHLCV数据完整性>99%"
            },
            
            "computational_requirements": {
                "feature_calculation_time": "< 1秒",
                "model_inference_time": "< 100毫秒",
                "memory_usage": "< 500MB",
                "cpu_requirements": "2核以上"
            },
            
            "monitoring_alerts": [
                "特征分布偏移检测",
                "模型性能衰减监控",
                "数据质量异常告警",
                "交易信号频率监控"
            ]
        },
        
        "risk_management": {
            "position_sizing": {
                "max_position_size": "账户资金的2%",
                "correlation_limit": "相关品种总敞口<10%",
                "single_symbol_limit": "单个品种<5%"
            },
            
            "stop_loss_strategy": {
                "initial_stop_loss": "0.6%",
                "trailing_stop": "价格有利移动后激活",
                "time_stop": "最大持仓240分钟"
            },
            
            "risk_controls": [
                "日内最大亏损限制",
                "连续亏损次数控制",
                "市场波动率调整",
                "流动性风险管理"
            ]
        },
        
        "usage_examples": {
            "feature_loading": '''
# 加载特征数据
import pandas as pd
df = pd.read_parquet("features_BTCUSDT_superdip_pinbar.parquet")
print(f"特征数量: {len(df.columns)}, 数据条数: {len(df)}")
            ''',
            
            "signal_generation": '''
# 生成交易信号
def generate_signals(df):
    signals = (
        (df['BTCUSDT_rsi_14'] < 30) &  # RSI超跌
        (df['BTCUSDT_price_below_ma_20'] == 1) &  # 价格低于MA20
        (df['BTCUSDT_bb_oversold'] == 1) &  # 布林带超跌
        (df['BTCUSDT_is_enhanced_pin_bar'] == 1) &  # 接针形态
        (df['BTCUSDT_volume_ratio_20'] > 1.5)  # 成交量放大
    )
    return signals
            ''',
            
            "model_training": '''
# 模型训练示例
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit

# 特征选择
feature_cols = [col for col in df.columns if not col.startswith('BTCUSDT_win_')]
target_col = 'BTCUSDT_win_80bp_4h'

X, y = df[feature_cols], df[target_col]

# 时序交叉验证
tscv = TimeSeriesSplit(n_splits=5)
model = LGBMClassifier(objective='binary', metric='auc')

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"验证集AUC: {score:.3f}")
            '''
        }
    }
    
    # Save configuration
    config_file = f"data/FeatureSet_SuperDip_PinBar_Complete_{timestamp}.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(feature_set_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n{'='*60}")
    logger.info("SuperDip Pin Bar Complete FeatureSet Generated!")
    logger.info(f"{'='*60}")
    logger.info(f"Configuration file: {config_file}")
    logger.info(f"Total symbols: {len(symbols_analysis)}")
    logger.info(f"Total features: {sum(len(cat) for cat in all_feature_categories.values())}")
    logger.info(f"Feature categories: {', '.join(all_feature_categories.keys())}")
    
    print("\nFeature Summary:")
    for category, features in all_feature_categories.items():
        print(f"  {category}: {len(features)} features")
    
    print("\nSymbol Analysis:")
    for symbol, analysis in symbols_analysis.items():
        data = analysis['analysis']
        print(f"  {symbol}: {data['feature_count']} features, {data['record_count']} records")
        if 'pin_bar_detection_rate' in data:
            print(f"    Pin Bar Rate: {data['pin_bar_detection_rate']:.2%}")
    
    return config_file

if __name__ == "__main__":
    config_file = generate_feature_set_config()
    print(f"\nFeatureSet configuration saved to: {config_file}")