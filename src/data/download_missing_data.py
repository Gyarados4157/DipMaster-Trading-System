"""
下载缺失的BTCUSDT和ETHUSDT历史数据
为DipMaster Enhanced V4策略补充完整的数据集
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.data.market_data_manager import MarketDataManager, MarketDataSpec
from src.data.data_downloader import DataDownloader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download_missing_data.log'),
        logging.StreamHandler()
    ]
)

async def download_missing_symbols():
    """下载缺失的BTCUSDT和ETHUSDT数据"""
    logger = logging.getLogger(__name__)
    
    # 配置信息
    config = {
        'data_root': str(project_root / 'data'),
        'api_keys': {},  # 使用公共接口
        'chunk_size': 1000,
        'max_concurrent_downloads': 3,
        'retry_attempts': 3
    }
    
    # 缺失的交易对
    missing_symbols = ['BTCUSDT', 'ETHUSDT']
    
    # 时间范围（最近2年）
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)
    
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    logger.info(f"开始下载缺失数据: {missing_symbols}")
    logger.info(f"时间范围: {start_date_str} 到 {end_date_str}")
    
    try:
        # 初始化下载器
        downloader = DataDownloader(config)
        await downloader.initialize()
        
        # 批量下载
        results = await downloader.download_missing_symbols(
            missing_symbols=missing_symbols,
            timeframes=['5m'],
            start_date=start_date_str,
            end_date=end_date_str
        )
        
        # 处理结果
        success_count = 0
        total_records = 0
        
        for symbol, symbol_results in results.items():
            logger.info(f"\n=== {symbol} 下载结果 ===")
            
            for timeframe, result in symbol_results.items():
                if result.get('status') == 'success':
                    success_count += 1
                    records = result.get('records_count', 0)
                    total_records += records
                    file_path = result.get('file_path', '')
                    
                    logger.info(f"✓ {timeframe}: {records}条记录, 文件: {file_path}")
                else:
                    logger.error(f"✗ {timeframe}: {result.get('error', '未知错误')}")
        
        logger.info(f"\n=== 下载完成 ===")
        logger.info(f"成功: {success_count}, 总记录: {total_records}")
        
        # 生成下载报告
        await generate_download_report(results, start_date_str, end_date_str)
        
        # 清理资源
        await downloader.cleanup()
        
        return results
        
    except Exception as e:
        logger.error(f"数据下载失败: {e}")
        raise

async def generate_download_report(results, start_date, end_date):
    """生成下载报告"""
    logger = logging.getLogger(__name__)
    
    report = {
        'download_time': datetime.now().isoformat(),
        'date_range': {
            'start': start_date,
            'end': end_date
        },
        'symbols': [],
        'summary': {
            'total_symbols': len(results),
            'successful_downloads': 0,
            'failed_downloads': 0,
            'total_records': 0,
            'total_size_mb': 0
        }
    }
    
    for symbol, symbol_results in results.items():
        symbol_report = {
            'symbol': symbol,
            'timeframes': {},
            'total_records': 0,
            'status': 'success'
        }
        
        for timeframe, result in symbol_results.items():
            if result.get('status') == 'success':
                records = result.get('records_count', 0)
                file_size = result.get('file_size', 0)
                
                symbol_report['timeframes'][timeframe] = {
                    'status': 'success',
                    'records': records,
                    'file_size_bytes': file_size,
                    'file_path': result.get('file_path', '')
                }
                
                symbol_report['total_records'] += records
                report['summary']['total_records'] += records
                report['summary']['total_size_mb'] += file_size / (1024 * 1024)
                report['summary']['successful_downloads'] += 1
                
            else:
                symbol_report['timeframes'][timeframe] = {
                    'status': 'failed',
                    'error': result.get('error', '未知错误')
                }
                symbol_report['status'] = 'partial' if symbol_report['status'] == 'success' else 'failed'
                report['summary']['failed_downloads'] += 1
        
        report['symbols'].append(symbol_report)
    
    # 保存报告
    report_path = project_root / 'data' / 'market_data' / f'download_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"下载报告已保存: {report_path}")

async def verify_downloaded_data():
    """验证下载的数据质量"""
    logger = logging.getLogger(__name__)
    logger.info("开始验证下载的数据质量")
    
    # 检查文件是否存在
    data_dir = project_root / 'data' / 'market_data'
    
    expected_files = [
        'BTCUSDT_5m_2years.parquet',
        'ETHUSDT_5m_2years.parquet'
    ]
    
    verification_results = {}
    
    for filename in expected_files:
        file_path = data_dir / filename
        symbol = filename.split('_')[0]
        
        if file_path.exists():
            try:
                # 读取并验证数据
                import pandas as pd
                df = pd.read_parquet(file_path)
                
                verification_results[symbol] = {
                    'file_exists': True,
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                    'record_count': len(df),
                    'date_range': {
                        'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else 'N/A',
                        'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else 'N/A'
                    },
                    'columns': list(df.columns),
                    'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'quality_checks': {
                        'no_null_values': df.isnull().sum().sum() == 0,
                        'positive_prices': (df[['open', 'high', 'low', 'close']] > 0).all().all() if all(col in df.columns for col in ['open', 'high', 'low', 'close']) else 'N/A',
                        'ohlc_relationships': 'N/A'
                    }
                }
                
                # OHLC关系检查
                if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    ohlc_valid = (
                        (df['high'] >= df['open']) & 
                        (df['high'] >= df['low']) & 
                        (df['high'] >= df['close']) &
                        (df['low'] <= df['open']) & 
                        (df['low'] <= df['close'])
                    ).all()
                    verification_results[symbol]['quality_checks']['ohlc_relationships'] = ohlc_valid
                
                logger.info(f"✓ {symbol}: {len(df)}条记录, 文件大小: {file_path.stat().st_size / (1024 * 1024):.1f}MB")
                
            except Exception as e:
                verification_results[symbol] = {
                    'file_exists': True,
                    'error': str(e)
                }
                logger.error(f"✗ {symbol}: 数据验证失败 - {e}")
        else:
            verification_results[symbol] = {
                'file_exists': False,
                'error': 'File not found'
            }
            logger.error(f"✗ {symbol}: 文件不存在")
    
    # 保存验证报告
    verification_report_path = project_root / 'data' / 'market_data' / f'verification_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    
    with open(verification_report_path, 'w', encoding='utf-8') as f:
        json.dump(verification_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"验证报告已保存: {verification_report_path}")
    
    return verification_results

async def main():
    """主函数"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("=== DipMaster Enhanced V4 - 缺失数据下载 ===")
        
        # 1. 下载缺失数据
        download_results = await download_missing_symbols()
        
        # 2. 验证数据质量
        verification_results = await verify_downloaded_data()
        
        # 3. 汇总结果
        logger.info("\n=== 最终结果汇总 ===")
        
        for symbol in ['BTCUSDT', 'ETHUSDT']:
            download_status = download_results.get(symbol, {}).get('5m', {}).get('status', 'failed')
            verification_status = verification_results.get(symbol, {}).get('file_exists', False)
            
            if download_status == 'success' and verification_status:
                logger.info(f"✓ {symbol}: 下载并验证成功")
            else:
                logger.error(f"✗ {symbol}: 下载或验证失败")
        
        logger.info("=== 数据下载任务完成 ===")
        
    except Exception as e:
        logger.error(f"任务执行失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # 运行下载任务
    success = asyncio.run(main())
    
    if success:
        print("\n数据下载任务成功完成!")
        print("请检查 data/market_data/ 目录查看下载的文件")
    else:
        print("\n数据下载任务失败!")
        print("请查看日志文件 download_missing_data.log 了解详情")
        exit(1)