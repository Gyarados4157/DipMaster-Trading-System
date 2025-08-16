#!/usr/bin/env python3
"""
反过拟合验证主程序
Anti-Overfitting Validation Main Program

运行完整的策略验证流程，解决过拟合问题

Author: DipMaster Trading Team
Date: 2025-08-15
Version: 1.0.0
"""

import sys
import logging
from pathlib import Path
import pandas as pd
from typing import Dict

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.validation.comprehensive_validator import ComprehensiveValidator, ValidationConfig
from src.core.simple_dipmaster_strategy import SimpleDipMasterStrategy

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/anti_overfitting_validation.log')
        ]
    )

def load_market_data() -> Dict[str, pd.DataFrame]:
    """加载市场数据"""
    logger = logging.getLogger(__name__)
    logger.info("加载市场数据...")
    
    data_dir = Path("data/market_data")
    market_data = {}
    
    # 标准测试币种
    symbols = ['BTCUSDT', 'ADAUSDT', 'ALGOUSDT', 'BNBUSDT', 'DOGEUSDT',
               'ICPUSDT', 'IOTAUSDT', 'SOLUSDT', 'SUIUSDT', 'XRPUSDT']
    
    for symbol in symbols:
        file_path = data_dir / f"{symbol}_5m_2years.csv"
        
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                market_data[symbol] = df
                logger.info(f"加载 {symbol}: {len(df)} 条记录")
            except Exception as e:
                logger.error(f"加载 {symbol} 失败: {e}")
        else:
            logger.warning(f"数据文件不存在: {file_path}")
    
    logger.info(f"成功加载 {len(market_data)} 个币种的数据")
    return market_data

def main():
    """主程序"""
    print("🚀 DipMaster反过拟合验证系统启动")
    print("="*60)
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # 创建日志目录
        Path("logs").mkdir(exist_ok=True)
        
        logger.info("开始反过拟合验证流程...")
        
        # 1. 加载市场数据
        print("📊 加载市场数据...")
        market_data = load_market_data()
        
        if not market_data:
            print("❌ 没有可用的市场数据")
            return
        
        # 2. 配置验证参数
        print("⚙️ 配置验证参数...")
        validation_config = ValidationConfig(
            train_ratio=0.60,
            val_ratio=0.20,
            test_ratio=0.20,
            significance_level=0.05,
            monte_carlo_simulations=1000,  # 为了演示，减少模拟次数
            min_overall_score=70
        )
        
        # 3. 初始化验证器
        print("🔧 初始化综合验证器...")
        validator = ComprehensiveValidator(validation_config)
        
        # 4. 运行完整验证
        print("🔍 开始完整策略验证...")
        print("注意：这个过程可能需要几分钟时间...")
        
        validation_result = validator.run_full_validation(
            market_data=market_data,
            strategy_class=SimpleDipMasterStrategy
        )
        
        # 5. 显示结果
        print("\n" + "="*60)
        print("🎯 验证完成！")
        print("="*60)
        print(f"总体评分: {validation_result.overall_score:.1f}/100")
        print(f"风险等级: {validation_result.risk_level}")
        print(f"验证状态: {'✅ 通过' if validation_result.validation_passed else '❌ 未通过'}")
        print("\n" + validation_result.final_decision)
        
        if validation_result.warnings:
            print("\n⚠️ 警告信息:")
            for warning in validation_result.warnings:
                print(f"  - {warning}")
        
        if validation_result.recommendations:
            print("\n💡 建议措施:")
            for rec in validation_result.recommendations:
                print(f"  - {rec}")
        
        print("\n📁 详细结果保存在: results/comprehensive_validation/")
        print("="*60)
        
        # 6. 生成对比分析
        print("\n📊 验证结果分析:")
        print("-" * 40)
        
        # 显示各组件得分
        component_scores = validation_result.component_results
        
        components = [
            ("数据质量", "data_splitting"),
            ("策略回测", "strategy_backtest"), 
            ("统计验证", "statistical_validation"),
            ("Walk-Forward", "walk_forward"),
            ("过拟合检测", "overfitting_detection"),
            ("多资产验证", "multi_asset_validation")
        ]
        
        for name, key in components:
            result = component_scores.get(key, {})
            status = "✅ 成功" if 'error' not in result else "❌ 失败"
            print(f"{name:12}: {status}")
        
        print("-" * 40)
        
        # 7. 最终建议
        if validation_result.validation_passed:
            print("🎉 恭喜！策略已通过反过拟合验证")
            print("💼 可以考虑谨慎的实盘交易")
            print("📈 建议从小额资金开始")
        else:
            print("⚠️ 策略未通过验证")
            print("🛑 禁止实盘交易")
            print("🔧 请根据建议改进策略")
        
        print("\n" + "="*60)
        logger.info("反过拟合验证流程完成")
        
    except Exception as e:
        logger.error(f"验证流程出错: {e}")
        print(f"❌ 验证过程中出现错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)