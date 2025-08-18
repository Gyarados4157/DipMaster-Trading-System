#!/usr/bin/env python3
"""
DipMaster持续训练系统演示
使用当前优化的特征数据运行单次迭代，验证整个系统流程
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import logging

warnings.filterwarnings('ignore')

# 添加项目路径
sys.path.append('src')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """运行演示"""
    logger.info("🚀 启动DipMaster持续训练系统演示")
    
    try:
        # 检查数据是否存在
        data_dir = "data/continuous_optimization"
        if not os.path.exists(data_dir):
            logger.error(f"❌ 数据目录不存在: {data_dir}")
            return
        
        # 检查特征文件
        feature_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        if not feature_files:
            logger.error(f"❌ 未找到特征文件在目录: {data_dir}")
            return
        
        logger.info(f"✅ 发现 {len(feature_files)} 个特征文件:")
        for file in feature_files:
            logger.info(f"  - {file}")
        
        # 运行持续训练系统
        logger.info("📊 启动持续训练编排器...")
        
        # 导入主程序
        from run_continuous_model_training import ContinuousModelTrainingOrchestrator
        
        # 创建演示配置
        demo_config = {
            "training_interval_hours": 0.1,  # 6分钟间隔（演示用）
            "data_dir": data_dir,
            "max_iterations": 2,  # 只运行2次迭代
            "early_stopping_patience": 1
        }
        
        # 创建编排器
        orchestrator = ContinuousModelTrainingOrchestrator()
        orchestrator.config.update(demo_config)
        
        # 运行单次迭代演示
        logger.info("🔄 开始演示迭代...")
        result = orchestrator.run_single_iteration()
        
        # 显示结果
        if result.get('success', True):
            logger.info("✅ 演示迭代成功完成!")
            
            # 显示性能摘要
            perf_summary = result.get('performance_summary', {})
            if perf_summary:
                logger.info("📈 性能摘要:")
                logger.info(f"  平均胜率: {perf_summary.get('avg_win_rate', 0):.1%}")
                logger.info(f"  平均夏普比率: {perf_summary.get('avg_sharpe_ratio', 0):.2f}")
                logger.info(f"  平均最大回撤: {perf_summary.get('avg_max_drawdown', 0):.1%}")
                logger.info(f"  平均年化收益: {perf_summary.get('avg_annual_return', 0):.1%}")
            
            # 显示处理的币种
            symbols_processed = result.get('symbols_processed', 0)
            targets_achieved = result.get('targets_achieved_count', 0)
            
            logger.info(f"📊 处理统计:")
            logger.info(f"  处理币种数: {symbols_processed}")
            logger.info(f"  达标模型数: {targets_achieved}")
            
            if targets_achieved > 0:
                logger.info("🎉 发现达标模型!")
            else:
                logger.info("⚠️ 尚无达标模型，系统会继续优化")
                
        else:
            logger.error(f"❌ 演示迭代失败: {result.get('error', 'Unknown error')}")
        
        logger.info("📋 检查输出文件...")
        results_dir = "results/continuous_training"
        if os.path.exists(results_dir):
            files = os.listdir(results_dir)
            logger.info(f"✅ 生成了 {len(files)} 个输出文件")
            for file in sorted(files)[-5:]:  # 显示最新的5个文件
                logger.info(f"  📄 {file}")
        
        logger.info("🎯 系统演示完成!")
        logger.info("💡 提示:")
        logger.info("  - 使用 'python run_continuous_model_training.py --single-run' 运行单次完整迭代")
        logger.info("  - 使用 'python run_continuous_model_training.py' 运行持续训练循环")
        logger.info("  - 检查 results/continuous_training/ 目录查看详细结果")
        
    except ImportError as e:
        logger.error(f"❌ 模块导入失败: {e}")
        logger.info("💡 确保已安装所有依赖库:")
        logger.info("  pip install -r requirements.txt")
        
    except Exception as e:
        logger.error(f"❌ 演示运行失败: {e}")
        import traceback
        logger.error(traceback.format_exc())

def quick_system_check():
    """快速系统检查"""
    logger.info("🔍 执行系统检查...")
    
    # 检查必要的库
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'lightgbm', 
        'xgboost', 'optuna', 'matplotlib', 'seaborn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            logger.error(f"❌ {package}")
    
    if missing_packages:
        logger.error(f"缺少依赖包: {missing_packages}")
        logger.info("请运行: pip install " + " ".join(missing_packages))
        return False
    
    # 检查数据文件
    data_dir = "data/continuous_optimization"
    if os.path.exists(data_dir):
        feature_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
        logger.info(f"✅ 数据目录存在，包含 {len(feature_files)} 个特征文件")
        
        if feature_files:
            # 检查一个文件的内容
            sample_file = os.path.join(data_dir, feature_files[0])
            try:
                df = pd.read_parquet(sample_file)
                logger.info(f"✅ 样本数据: {df.shape[0]} 行, {df.shape[1]} 列")
                
                # 检查是否有目标标签
                target_cols = [col for col in df.columns if 'target' in col.lower()]
                if target_cols:
                    logger.info(f"✅ 发现目标列: {target_cols}")
                else:
                    logger.warning("⚠️ 未发现目标列")
                    
            except Exception as e:
                logger.error(f"❌ 无法读取数据文件: {e}")
                return False
        else:
            logger.error("❌ 数据目录为空")
            return False
    else:
        logger.error(f"❌ 数据目录不存在: {data_dir}")
        return False
    
    logger.info("✅ 系统检查完成")
    return True

if __name__ == "__main__":
    # 先进行系统检查
    if quick_system_check():
        # 运行演示
        main()
    else:
        logger.error("❌ 系统检查失败，无法运行演示")
        sys.exit(1)