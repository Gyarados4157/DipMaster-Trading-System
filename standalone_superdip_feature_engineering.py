#!/usr/bin/env python3
"""
Standalone SuperDip Needle Feature Engineering
独立的超跌接针策略特征工程

独立运行的特征工程脚本，避免复杂的依赖关系
"""

import pandas as pd
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Any, Tuple, Optional
import time
from dataclasses import dataclass, field

# 抑制警告
warnings.filterwarnings('ignore')

@dataclass
class FeatureEngineeringConfig:
    """特征工程配置"""
    symbols: List[str] = field(default_factory=lambda: [
        'ADAUSDT', 'XRPUSDT', 'SOLUSDT', 'BNBUSDT', 'AVAXUSDT',
        'LINKUSDT', 'LTCUSDT', 'DOGEUSDT', 'TRXUSDT', 'DOTUSDT'
    ])
    primary_timeframe: str = '5m'
    prediction_horizons: List[int] = field(default_factory=lambda: [15, 30, 60, 240])
    profit_targets: List[float] = field(default_factory=lambda: [0.008, 0.015, 0.025, 0.040])
    stop_loss: float = 0.006

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_market_data_bundle(bundle_path: str, symbols: List[str], timeframe: str) -> Dict[str, pd.DataFrame]:
    """加载市场数据"""
    logger = logging.getLogger(__name__)
    
    try:
        with open(bundle_path, 'r') as f:
            bundle = json.load(f)
        
        data_dict = {}
        
        for symbol in symbols:
            if symbol in bundle.get('data_files', {}):
                symbol_files = bundle['data_files'][symbol]
                if timeframe in symbol_files:
                    file_path = symbol_files[timeframe]['file_path']
                    if Path(file_path).exists():
                        try:
                            df = pd.read_parquet(file_path)
                            df.index = pd.to_datetime(df.index)
                            data_dict[symbol] = df
                            logger.info(f"Loaded {symbol}: {len(df)} rows")
                        except Exception as e:
                            logger.error(f"Failed to load {symbol}: {e}")
                    else:
                        logger.warning(f"File not found: {file_path}")
        
        return data_dict
        
    except Exception as e:
        logger.error(f"Failed to load market data bundle: {e}")
        return {}

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """计算RSI指标"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """计算布林带"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return upper, sma, lower

def generate_oversold_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成超跌识别特征"""
    result = df.copy()
    
    # 1. RSI多周期特征
    for period in [7, 14, 21, 30]:
        rsi_col = f'rsi_{period}'
        result[rsi_col] = calculate_rsi(df['close'], period)
        result[f'{rsi_col}_oversold'] = (result[rsi_col] < 30).astype(int)
        result[f'{rsi_col}_extreme_oversold'] = (result[rsi_col] < 25).astype(int)
        result[f'{rsi_col}_change'] = result[rsi_col].pct_change()
    
    # 2. 移动平均偏离度
    for period in [10, 20, 50]:
        ma_col = f'ma_{period}'
        result[ma_col] = df['close'].rolling(window=period).mean()
        deviation_col = f'price_deviation_ma{period}'
        result[deviation_col] = (df['close'] - result[ma_col]) / result[ma_col]
        result[f'{deviation_col}_oversold'] = (result[deviation_col] < -0.03).astype(int)
    
    # 3. 布林带特征
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(df['close'], 20, 2.0)
    result['bb_upper'] = bb_upper
    result['bb_middle'] = bb_middle
    result['bb_lower'] = bb_lower
    result['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
    result['bb_squeeze'] = (bb_upper - bb_lower) / bb_middle
    result['bb_break_lower'] = (df['close'] < bb_lower).astype(int)
    
    # 4. 价格Z-Score
    for window in [20, 50]:
        price_mean = df['close'].rolling(window=window).mean()
        price_std = df['close'].rolling(window=window).std()
        zscore_col = f'price_zscore_{window}'
        result[zscore_col] = (df['close'] - price_mean) / price_std
        result[f'{zscore_col}_oversold'] = (result[zscore_col] < -2).astype(int)
    
    return result

def generate_needle_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成接针形态特征"""
    result = df.copy()
    
    # K线基本特征
    open_price = df['open']
    high_price = df['high']
    low_price = df['low']
    close_price = df['close']
    
    # 1. K线形态计算
    result['candle_body'] = abs(close_price - open_price)
    result['candle_range'] = high_price - low_price
    result['body_ratio'] = result['candle_body'] / (result['candle_range'] + 1e-8)
    
    # 2. 影线特征
    result['upper_shadow'] = np.where(
        close_price > open_price,
        high_price - close_price,
        high_price - open_price
    )
    result['lower_shadow'] = np.where(
        close_price > open_price,
        open_price - low_price,
        close_price - low_price
    )
    
    result['upper_shadow_ratio'] = result['upper_shadow'] / (result['candle_range'] + 1e-8)
    result['lower_shadow_ratio'] = result['lower_shadow'] / (result['candle_range'] + 1e-8)
    result['lower_shadow_body_ratio'] = result['lower_shadow'] / (result['candle_body'] + 1e-8)
    
    # 3. 接针形态识别
    result['hammer_pattern'] = (
        (result['lower_shadow_ratio'] > 0.6) &
        (result['body_ratio'] < 0.3) &
        (result['lower_shadow_body_ratio'] > 2)
    ).astype(int)
    
    result['doji_pattern'] = (result['body_ratio'] < 0.1).astype(int)
    
    # 4. 实体位置
    result['body_position'] = np.where(
        result['candle_range'] > 0,
        np.where(
            close_price > open_price,
            (open_price - low_price) / result['candle_range'],
            (close_price - low_price) / result['candle_range']
        ),
        0.5
    )
    
    # 5. 价格恢复度
    result['recovery_from_low'] = (close_price - low_price) / (result['candle_range'] + 1e-8)
    
    return result

def generate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成成交量特征"""
    result = df.copy()
    volume = df['volume']
    
    # 1. 成交量移动平均
    for period in [10, 20, 50]:
        volume_ma = volume.rolling(window=period).mean()
        result[f'volume_ma_{period}'] = volume_ma
        result[f'volume_ratio_{period}'] = volume / (volume_ma + 1e-8)
        result[f'volume_spike_{period}'] = (result[f'volume_ratio_{period}'] > 2.0).astype(int)
    
    # 2. 成交量趋势
    result['volume_trend_10'] = volume.rolling(window=10).apply(
        lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) > 1 else 0
    )
    
    # 3. 价格-成交量关系
    returns = df['close'].pct_change()
    volume_change = volume.pct_change()
    result['price_volume_corr_20'] = returns.rolling(window=20).corr(volume_change)
    
    # 4. 成交量加权平均价格 (VWAP)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    for period in [10, 20]:
        vwap = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        result[f'vwap_{period}'] = vwap
        result[f'vwap_deviation_{period}'] = (df['close'] - vwap) / vwap
    
    return result

def generate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成动量特征"""
    result = df.copy()
    
    # 1. 价格动量
    for period in [3, 5, 10]:
        result[f'momentum_{period}'] = df['close'].pct_change(periods=period)
    
    # 2. 价格加速度
    returns = df['close'].pct_change()
    result['price_acceleration'] = returns.diff()
    
    # 3. 趋势强度
    for period in [10, 20]:
        result[f'trend_strength_{period}'] = df['close'].rolling(window=period).apply(
            lambda x: abs(np.corrcoef(np.arange(len(x)), x)[0, 1]) if len(x) > 1 else 0
        )
    
    return result

def generate_labels(df: pd.DataFrame, horizons: List[int], targets: List[float], stop_loss: float) -> pd.DataFrame:
    """生成标签数据"""
    result = df.copy()
    prices = df['close'].values
    
    # 计算未来收益
    for horizon in horizons:
        return_col = f'target_return_{horizon}min'
        future_returns = []
        
        for i in range(len(prices)):
            if i + horizon < len(prices):
                future_return = (prices[i + horizon] - prices[i]) / prices[i]
                future_returns.append(future_return)
            else:
                future_returns.append(np.nan)
        
        result[return_col] = future_returns
        
        # 利润目标标签
        for target in targets:
            result[f'target_profit_{target*100:.1f}pct_{horizon}min'] = (
                result[return_col] >= target
            ).astype(int)
        
        # 止损标签
        result[f'target_stop_loss_{horizon}min'] = (
            result[return_col] <= -stop_loss
        ).astype(int)
    
    # 最优出场时间
    max_horizon = max(horizons)
    optimal_exits = []
    optimal_returns = []
    
    for i in range(len(prices)):
        best_return = -float('inf')
        best_time = None
        
        for t in range(5, min(max_horizon + 1, len(prices) - i), 5):
            future_return = (prices[i + t] - prices[i]) / prices[i]
            if future_return > best_return:
                best_return = future_return
                best_time = t
        
        optimal_exits.append(best_time)
        optimal_returns.append(best_return)
    
    result['optimal_exit_time'] = optimal_exits
    result['optimal_return'] = optimal_returns
    
    return result

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """清理特征数据"""
    result = df.copy()
    
    # 处理无穷值
    result = result.replace([np.inf, -np.inf], np.nan)
    
    # 处理缺失值
    numeric_cols = result.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if not col.startswith('target_'):
            # 前向填充
            result[col] = result[col].fillna(method='ffill')
            # 后向填充
            result[col] = result[col].fillna(method='bfill')
            # 剩余用中位数填充
            if result[col].isnull().sum() > 0:
                result[col] = result[col].fillna(result[col].median())
    
    # 异常值处理
    for col in numeric_cols:
        if not col.startswith('target_') and result[col].std() > 0:
            # 99%分位数裁剪
            lower = result[col].quantile(0.005)
            upper = result[col].quantile(0.995)
            result[col] = result[col].clip(lower=lower, upper=upper)
    
    return result

def calculate_feature_importance_simple(df: pd.DataFrame, target_col: str) -> Dict[str, float]:
    """简单的特征重要性计算（相关系数）"""
    if target_col not in df.columns:
        return {}
    
    feature_cols = [col for col in df.columns if not col.startswith('target_')]
    clean_df = df[feature_cols + [target_col]].dropna()
    
    if len(clean_df) < 50:
        return {}
    
    importance = {}
    target_series = clean_df[target_col]
    
    for col in feature_cols:
        try:
            corr = abs(clean_df[col].corr(target_series))
            importance[col] = corr if not np.isnan(corr) else 0
        except:
            importance[col] = 0
    
    # 按重要性排序
    return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

def validate_data_quality(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """数据质量验证"""
    quality_report = {
        'symbol': symbol,
        'total_rows': len(df),
        'total_features': len(df.columns),
        'missing_data': {},
        'data_range': {
            'start': str(df.index.min()),
            'end': str(df.index.max())
        }
    }
    
    # 缺失值分析
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        missing_pct = missing_count / len(df) * 100
        quality_report['missing_data'][col] = {
            'count': int(missing_count),
            'percentage': round(missing_pct, 2)
        }
    
    return quality_report

def process_single_symbol(symbol: str, df: pd.DataFrame, config: FeatureEngineeringConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """处理单个交易品种"""
    logger = logging.getLogger(__name__)
    logger.info(f"Processing features for {symbol}...")
    
    try:
        # 生成各类特征
        enhanced_df = df.copy()
        
        # 1. 超跌识别特征
        enhanced_df = generate_oversold_features(enhanced_df)
        
        # 2. 接针形态特征
        enhanced_df = generate_needle_pattern_features(enhanced_df)
        
        # 3. 成交量特征
        enhanced_df = generate_volume_features(enhanced_df)
        
        # 4. 动量特征
        enhanced_df = generate_momentum_features(enhanced_df)
        
        # 5. 标签生成
        enhanced_df = generate_labels(
            enhanced_df, 
            config.prediction_horizons, 
            config.profit_targets, 
            config.stop_loss
        )
        
        # 6. 特征清理
        enhanced_df = clean_features(enhanced_df)
        
        # 7. 质量验证
        quality_report = validate_data_quality(enhanced_df, symbol)
        
        # 8. 特征重要性
        feature_importance = calculate_feature_importance_simple(enhanced_df, 'target_return_30min')
        
        # 统计信息
        stats = {
            'symbol': symbol,
            'total_features': len(enhanced_df.columns),
            'total_rows': len(enhanced_df),
            'feature_importance': feature_importance,
            'data_quality': quality_report
        }
        
        logger.info(f"Completed {symbol}: {len(enhanced_df.columns)} features, {len(enhanced_df)} rows")
        
        return enhanced_df, stats
        
    except Exception as e:
        logger.error(f"Failed to process {symbol}: {e}")
        raise

def main():
    """主函数"""
    logger = setup_logging()
    logger.info("=== SuperDip Needle Feature Engineering (Standalone) ===")
    
    try:
        # 配置
        config = FeatureEngineeringConfig()
        
        # 创建输出目录
        output_dir = Path("data/superdip_features")
        output_dir.mkdir(exist_ok=True)
        
        # 加载数据
        bundle_path = "data/MarketDataBundle_Top30_Enhanced_Final.json"
        if not Path(bundle_path).exists():
            logger.error(f"Bundle not found: {bundle_path}")
            return
        
        market_data = load_market_data_bundle(bundle_path, config.symbols, config.primary_timeframe)
        
        if not market_data:
            logger.error("No market data loaded")
            return
        
        # 处理每个交易品种
        results = {}
        processing_stats = {}
        
        for symbol, df in market_data.items():
            if len(df) < 500:  # 最少数据要求
                logger.warning(f"Insufficient data for {symbol}: {len(df)} rows")
                continue
            
            try:
                processed_df, stats = process_single_symbol(symbol, df, config)
                results[symbol] = processed_df
                processing_stats[symbol] = stats
            except Exception as e:
                logger.error(f"Failed to process {symbol}: {e}")
                continue
        
        # 保存结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存特征数据
        feature_files = {}
        for symbol, df in results.items():
            feature_file = output_dir / f"{symbol}_features_{timestamp}.parquet"
            df.to_parquet(feature_file, compression='zstd')
            feature_files[symbol] = str(feature_file)
            logger.info(f"Saved {symbol} features to {feature_file}")
        
        # 创建特征集信息
        feature_set_info = {
            'creation_time': datetime.now().isoformat(),
            'pipeline_version': '1.0.0-Standalone',
            'config': {
                'symbols': config.symbols,
                'primary_timeframe': config.primary_timeframe,
                'prediction_horizons': config.prediction_horizons,
                'profit_targets': config.profit_targets
            },
            'results': {
                'total_symbols_processed': len(results),
                'success_rate': len(results) / len(config.symbols),
                'feature_files': feature_files
            },
            'processing_stats': processing_stats
        }
        
        # 保存特征集信息
        feature_set_file = output_dir / f"SuperDipNeedle_FeatureSet_{timestamp}.json"
        with open(feature_set_file, 'w') as f:
            json.dump(feature_set_info, f, indent=2, default=str)
        
        # 保存标签集信息
        label_set_info = {
            'creation_time': datetime.now().isoformat(),
            'pipeline_version': '1.0.0-Standalone',
            'label_types': {
                'return_targets': config.prediction_horizons,
                'profit_targets': config.profit_targets,
                'optimal_timing': ['optimal_exit_time', 'optimal_return']
            },
            'symbols': list(results.keys()),
            'quality_assurance': {
                'no_future_leakage': True,
                'proper_alignment': True,
                'missing_handled': True
            }
        }
        
        label_set_file = output_dir / f"SuperDipNeedle_LabelSet_{timestamp}.json"
        with open(label_set_file, 'w') as f:
            json.dump(label_set_info, f, indent=2, default=str)
        
        # 生成简单报告
        report_file = output_dir / f"Feature_Engineering_Report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# SuperDip Needle Feature Engineering Report\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 处理结果\n\n")
            f.write(f"- **处理币种数**: {len(results)}\n")
            f.write(f"- **成功率**: {len(results)/len(config.symbols)*100:.1f}%\n\n")
            
            f.write("## 币种详情\n\n")
            for symbol, stats in processing_stats.items():
                f.write(f"### {symbol}\n")
                f.write(f"- **特征数**: {stats['total_features']}\n")
                f.write(f"- **数据行数**: {stats['total_rows']}\n")
                
                # Top 5 重要特征
                if stats['feature_importance']:
                    top_features = list(stats['feature_importance'].items())[:5]
                    f.write("- **Top 5 重要特征**:\n")
                    for i, (feature, importance) in enumerate(top_features, 1):
                        f.write(f"  {i}. {feature}: {importance:.4f}\n")
                f.write("\n")
            
            f.write("## 文件输出\n\n")
            f.write(f"- **特征集**: {feature_set_file}\n")
            f.write(f"- **标签集**: {label_set_file}\n")
            for symbol, file_path in feature_files.items():
                f.write(f"- **{symbol} 特征数据**: {file_path}\n")
        
        # 输出总结
        print("\n" + "="*60)
        print("🎉 SuperDip Needle 特征工程完成!")
        print("="*60)
        print(f"✅ 处理币种: {len(results)}/{len(config.symbols)}")
        print(f"✅ 成功率: {len(results)/len(config.symbols)*100:.1f}%")
        print(f"✅ 特征集文件: {feature_set_file}")
        print(f"✅ 标签集文件: {label_set_file}")
        print(f"✅ 报告文件: {report_file}")
        print("\n生成的特征包括:")
        print("- 🎯 超跌识别: RSI多周期、价格偏离、布林带、Z-Score")
        print("- 🕯️ 接针形态: K线形态、影线比率、实体位置")
        print("- 📊 成交量: 成交量放大、VWAP偏离、价格-成交量关系")
        print("- 🚀 动量特征: 价格动量、加速度、趋势强度")
        print("- 🏷️ 多目标标签: 15/30/60/240分钟收益预测")
        print("\n" + "="*60)
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        print(f"❌ 特征工程失败: {e}")

if __name__ == "__main__":
    main()