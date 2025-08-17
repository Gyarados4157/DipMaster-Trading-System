"""
数据收集完成度统计和验证
"""

import os
import json
from pathlib import Path
from datetime import datetime
import pandas as pd

def analyze_data_collection():
    """分析数据收集完成度"""
    
    data_path = Path("data/enhanced_market_data")
    
    # 统计文件
    all_files = list(data_path.glob("*.parquet"))
    metadata_files = list(data_path.glob("*_metadata.json"))
    
    # 按币种分组
    symbols = set()
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    symbol_data = {}
    
    for file in all_files:
        if "_2years.parquet" in file.name:
            parts = file.name.split('_')
            if len(parts) >= 3:
                symbol = parts[0]
                timeframe = parts[1]
                
                if symbol not in symbol_data:
                    symbol_data[symbol] = {}
                
                symbol_data[symbol][timeframe] = {
                    'file': file.name,
                    'size_mb': file.stat().st_size / (1024 * 1024)
                }
                symbols.add(symbol)
    
    # 生成统计报告
    report = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'total_symbols': len(symbols),
            'total_files': len([f for f in all_files if "_2years.parquet" in f.name]),
            'total_size_mb': sum(f.stat().st_size for f in all_files if "_2years.parquet" in f.name) / (1024 * 1024),
            'symbols_list': sorted(list(symbols))
        },
        'completion_status': {}
    }
    
    # 分析每个币种的完成度
    for symbol in sorted(symbols):
        completed_timeframes = list(symbol_data[symbol].keys())
        completion_rate = len(completed_timeframes) / len(timeframes)
        
        report['completion_status'][symbol] = {
            'completed_timeframes': completed_timeframes,
            'missing_timeframes': [tf for tf in timeframes if tf not in completed_timeframes],
            'completion_rate': completion_rate,
            'total_size_mb': sum(tf_data['size_mb'] for tf_data in symbol_data[symbol].values()),
            'status': 'COMPLETE' if completion_rate == 1.0 else 
                     'PARTIAL' if completion_rate >= 0.5 else 'INCOMPLETE'
        }
    
    # 按完成度分类
    complete_symbols = [s for s, data in report['completion_status'].items() if data['status'] == 'COMPLETE']
    partial_symbols = [s for s, data in report['completion_status'].items() if data['status'] == 'PARTIAL']
    incomplete_symbols = [s for s, data in report['completion_status'].items() if data['status'] == 'INCOMPLETE']
    
    report['summary'].update({
        'complete_symbols': len(complete_symbols),
        'partial_symbols': len(partial_symbols),
        'incomplete_symbols': len(incomplete_symbols),
        'complete_symbols_list': complete_symbols,
        'partial_symbols_list': partial_symbols,
        'incomplete_symbols_list': incomplete_symbols
    })
    
    # 计算总体成功率
    total_expected_files = len(symbols) * len(timeframes)
    total_actual_files = sum(len(data['completed_timeframes']) for data in report['completion_status'].values())
    overall_success_rate = total_actual_files / total_expected_files if total_expected_files > 0 else 0
    
    report['summary']['overall_success_rate'] = overall_success_rate
    report['summary']['total_expected_files'] = total_expected_files
    report['summary']['total_actual_files'] = total_actual_files
    
    return report

def main():
    """主函数"""
    print("分析数据收集完成度...")
    
    report = analyze_data_collection()
    
    # 保存报告
    report_path = Path("data") / f"Data_Collection_Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)
    
    # 输出摘要
    summary = report['summary']
    
    print("\n" + "="*80)
    print("前30山寨币数据收集完成度报告")
    print("="*80)
    print(f"总币种数: {summary['total_symbols']}")
    print(f"总文件数: {summary['total_actual_files']}/{summary['total_expected_files']}")
    print(f"总数据量: {summary['total_size_mb']:.1f} MB")
    print(f"整体成功率: {summary['overall_success_rate']*100:.1f}%")
    print(f"完全成功: {summary['complete_symbols']} 币种")
    print(f"部分成功: {summary['partial_symbols']} 币种")
    print(f"未完成: {summary['incomplete_symbols']} 币种")
    print("="*80)
    
    # 显示完全成功的币种
    if summary['complete_symbols_list']:
        print(f"\n完全成功的币种 ({summary['complete_symbols']} 个):")
        for symbol in summary['complete_symbols_list']:
            data = report['completion_status'][symbol]
            print(f"  {symbol}: {data['total_size_mb']:.1f} MB")
    
    # 显示部分成功的币种
    if summary['partial_symbols_list']:
        print(f"\n部分成功的币种 ({summary['partial_symbols']} 个):")
        for symbol in summary['partial_symbols_list']:
            data = report['completion_status'][symbol]
            print(f"  {symbol}: {len(data['completed_timeframes'])}/6 时间框架, 缺失: {data['missing_timeframes']}")
    
    # 显示未完成的币种
    if summary['incomplete_symbols_list']:
        print(f"\n未完成的币种 ({summary['incomplete_symbols']} 个):")
        for symbol in summary['incomplete_symbols_list']:
            data = report['completion_status'][symbol]
            print(f"  {symbol}: {len(data['completed_timeframes'])}/6 时间框架")
    
    print(f"\n详细报告已保存至: {report_path}")
    print("="*80)
    
    return report

if __name__ == "__main__":
    main()