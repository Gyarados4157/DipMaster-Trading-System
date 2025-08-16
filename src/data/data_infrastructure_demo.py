"""
数据基础设施演示和验证脚本
展示DipMaster Enhanced V4数据基础设施的完整功能
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def demo_data_infrastructure():
    """演示数据基础设施功能"""
    logger = logging.getLogger(__name__)
    
    logger.info("=== DipMaster Enhanced V4 数据基础设施演示 ===")
    
    # 1. 验证MarketDataBundle配置
    bundle_path = project_root / 'data' / 'MarketDataBundle.json'
    
    if not bundle_path.exists():
        logger.error("MarketDataBundle.json 不存在")
        return False
    
    with open(bundle_path, 'r', encoding='utf-8') as f:
        bundle_config = json.load(f)
    
    logger.info(f"✓ 数据包版本: {bundle_config['version']}")
    logger.info(f"✓ 数据包ID: {bundle_config['metadata']['bundle_id']}")
    logger.info(f"✓ 交易对数量: {len(bundle_config['metadata']['symbols'])}")
    logger.info(f"✓ 数据质量评分: {bundle_config['metadata']['data_quality_score']}")
    logger.info(f"✓ 总数据量: {bundle_config['metadata']['total_size_mb']} MB")
    
    # 2. 验证数据文件完整性
    logger.info("\n=== 数据文件完整性验证 ===")
    
    total_files = 0
    missing_files = 0
    total_records = 0
    
    for symbol in bundle_config['metadata']['symbols']:
        # 检查主要数据文件
        if symbol in ['BTCUSDT', 'ETHUSDT']:
            file_path = project_root / 'data' / 'market_data' / f'{symbol}_5m_2years.parquet'
            metadata_path = project_root / 'data' / 'market_data' / f'{symbol}_5m_2years_metadata.json'
        else:
            file_path = project_root / 'data' / 'market_data' / f'{symbol}_5m_2years.csv'
            metadata_path = project_root / 'data' / 'market_data' / f'{symbol}_5m_2years_metadata.json'
        
        total_files += 1
        
        if file_path.exists() and metadata_path.exists():
            try:
                # 读取元数据
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                records = metadata.get('records_count', 0)
                total_records += records
                
                logger.info(f"✓ {symbol}: {records:,}条记录, {file_size_mb:.1f}MB")
                
            except Exception as e:
                logger.error(f"✗ {symbol}: 读取失败 - {e}")
                missing_files += 1
        else:
            logger.error(f"✗ {symbol}: 文件缺失")
            missing_files += 1
    
    # 3. 数据质量检查
    logger.info("\n=== 数据质量检查 ===")
    
    # 检查BTCUSDT数据样本
    btc_file = project_root / 'data' / 'market_data' / 'BTCUSDT_5m_2years.parquet'
    
    if btc_file.exists():
        try:
            df = pd.read_parquet(btc_file)
            
            logger.info(f"✓ BTCUSDT数据加载成功: {len(df):,}条记录")
            logger.info(f"✓ 时间范围: {df['timestamp'].min()} - {df['timestamp'].max()}")
            logger.info(f"✓ 数据列: {list(df.columns)}")
            
            # 数据质量检查
            null_count = df.isnull().sum().sum()
            logger.info(f"✓ 空值数量: {null_count}")
            
            if 'close' in df.columns:
                price_stats = df['close'].describe()
                logger.info(f"✓ 价格统计: 最小值={price_stats['min']:.2f}, 最大值={price_stats['max']:.2f}, 均值={price_stats['mean']:.2f}")
                
                # OHLC关系检查
                ohlc_valid = (
                    (df['high'] >= df['open']) & 
                    (df['high'] >= df['low']) & 
                    (df['high'] >= df['close']) &
                    (df['low'] <= df['open']) & 
                    (df['low'] <= df['close'])
                ).all()
                
                logger.info(f"✓ OHLC关系有效性: {ohlc_valid}")
            
        except Exception as e:
            logger.error(f"✗ BTCUSDT数据检查失败: {e}")
    
    # 4. 性能基准测试
    logger.info("\n=== 性能基准测试 ===")
    
    try:
        # 测试数据加载速度
        start_time = datetime.now()
        df = pd.read_parquet(btc_file)
        load_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"✓ 数据加载速度: {load_time:.1f}ms ({len(df):,}条记录)")
        
        # 测试查询性能
        start_time = datetime.now()
        recent_data = df.tail(1000)  # 最近1000条记录
        query_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"✓ 查询性能: {query_time:.1f}ms (1000条记录)")
        
        # 计算存储效率
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
        file_size = btc_file.stat().st_size / (1024 * 1024)
        compression_ratio = file_size / memory_usage
        
        logger.info(f"✓ 存储效率: 压缩比={compression_ratio:.2f}, 文件大小={file_size:.1f}MB, 内存使用={memory_usage:.1f}MB")
        
    except Exception as e:
        logger.error(f"✗ 性能测试失败: {e}")
    
    # 5. 总结报告
    logger.info("\n=== 基础设施状态总结 ===")
    
    success_rate = ((total_files - missing_files) / total_files * 100) if total_files > 0 else 0
    
    logger.info(f"文件完整性: {total_files - missing_files}/{total_files} ({success_rate:.1f}%)")
    logger.info(f"总数据记录: {total_records:,}")
    logger.info(f"预估数据覆盖: 2年历史数据")
    
    if success_rate >= 90:
        logger.info("✅ 数据基础设施状态: 优秀")
        status = "excellent"
    elif success_rate >= 70:
        logger.info("⚠️ 数据基础设施状态: 良好")
        status = "good"
    else:
        logger.info("❌ 数据基础设施状态: 需要改进")
        status = "needs_improvement"
    
    # 6. 生成状态报告
    report = {
        "validation_time": datetime.now().isoformat(),
        "infrastructure_status": status,
        "summary": {
            "total_files": total_files,
            "missing_files": missing_files,
            "success_rate": success_rate,
            "total_records": total_records,
            "data_quality_score": bundle_config['metadata']['data_quality_score']
        },
        "performance_benchmarks": {
            "data_load_time_ms": load_time if 'load_time' in locals() else None,
            "query_time_ms": query_time if 'query_time' in locals() else None,
            "compression_ratio": compression_ratio if 'compression_ratio' in locals() else None
        },
        "recommendations": []
    }
    
    if missing_files > 0:
        report["recommendations"].append(f"需要下载或修复{missing_files}个缺失的数据文件")
    
    if success_rate < 100:
        report["recommendations"].append("建议执行完整的数据质量验证")
    
    # 保存报告
    report_path = project_root / 'data' / f'infrastructure_validation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n✅ 验证报告已保存: {report_path}")
    
    return status == "excellent"

def show_usage_examples():
    """显示数据基础设施使用示例"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n=== 数据基础设施使用示例 ===")
    
    examples = [
        {
            "title": "加载历史K线数据",
            "code": """
# 加载BTCUSDT 5分钟K线数据
import pandas as pd

df = pd.read_parquet('data/market_data/BTCUSDT_5m_2years.parquet')
print(f"数据范围: {df['timestamp'].min()} - {df['timestamp'].max()}")
print(f"数据量: {len(df):,} 条记录")
            """
        },
        {
            "title": "实时数据流集成",
            "code": """
# 使用数据基础设施的实时流
from src.data.realtime_stream import RealtimeDataStream

config = {'realtime': {'buffer_size': 10000}}
stream = RealtimeDataStream(config)

# 订阅价格更新
async def price_handler(data):
    print(f"价格更新: {data['symbol']} = {data['price']}")

stream.subscribe('ticker_BTCUSDT', price_handler)
await stream.connect(['BTCUSDT', 'ETHUSDT'])
            """
        },
        {
            "title": "数据质量监控",
            "code": """
# 使用数据监控系统
from src.data.data_monitor import DataMonitor

monitor = DataMonitor(config)
await monitor.start_monitoring()

# 获取系统健康状态
status = await monitor.get_system_status()
print(f"数据质量评分: {status['overall_health_score']}")
            """
        },
        {
            "title": "高性能数据查询",
            "code": """
# 使用存储管理器进行优化查询
from src.data.storage_manager import StorageManager

storage = StorageManager(config)

# 加载指定时间范围的数据
df = await storage.load_kline_data(
    symbol='BTCUSDT',
    timeframe='5m',
    start_date='2024-01-01',
    end_date='2024-12-31',
    columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
)
            """
        }
    ]
    
    for i, example in enumerate(examples, 1):
        logger.info(f"\n{i}. {example['title']}:")
        logger.info(example['code'])

async def main():
    """主函数"""
    logger = logging.getLogger(__name__)
    
    try:
        # 运行基础设施验证
        success = await demo_data_infrastructure()
        
        # 显示使用示例
        show_usage_examples()
        
        if success:
            logger.info("\n🎉 DipMaster Enhanced V4 数据基础设施验证成功!")
            logger.info("系统已就绪，可以开始策略开发和测试。")
        else:
            logger.warning("\n⚠️ 数据基础设施存在一些问题，建议进行修复。")
        
        return success
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)