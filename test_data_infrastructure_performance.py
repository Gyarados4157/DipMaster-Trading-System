#!/usr/bin/env python3
"""
数据基础设施性能测试 - Data Infrastructure Performance Testing
验证DipMaster Trading System数据基础设施的性能和质量

Test Categories:
- 数据访问性能测试
- 质量监控功能测试  
- API接口性能测试
- 实时流处理测试
- 存储效率测试
"""

import asyncio
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging
import sys
import statistics
from typing import Dict, List, Any, Tuple
import concurrent.futures
import requests
import websockets
import traceback
import psutil
import gc

# 导入测试模块
sys.path.append(str(Path(__file__).parent))
from src.data.professional_data_infrastructure import ProfessionalDataInfrastructure
from src.data.data_quality_monitor import DataQualityMonitor
from src.data.realtime_data_stream import DataStreamManager

class PerformanceTestSuite:
    """性能测试套件"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.test_results = {
            'test_suite': 'Data Infrastructure Performance',
            'test_start': datetime.now(timezone.utc).isoformat(),
            'test_environment': self._get_test_environment(),
            'tests': {}
        }
        
        # 测试配置
        self.test_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'SOLUSDT', 'XRPUSDT']
        self.test_timeframes = ['1m', '5m', '15m', '1h']
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('performance_test.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger(__name__)
        
    def _get_test_environment(self) -> Dict[str, Any]:
        """获取测试环境信息"""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'disk_free_gb': psutil.disk_usage('.').free / 1024**3
        }
        
    def measure_time(self, func):
        """时间测量装饰器"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            return result, (end_time - start_time)
        return wrapper
        
    async def test_data_loading_performance(self) -> Dict[str, Any]:
        """测试数据加载性能"""
        self.logger.info("开始数据加载性能测试...")
        
        results = {
            'test_name': 'data_loading_performance',
            'status': 'running',
            'metrics': {}
        }
        
        try:
            # 初始化基础设施
            infrastructure = ProfessionalDataInfrastructure()
            
            loading_times = []
            record_counts = []
            memory_usage = []
            
            # 测试不同币种和时间框架的加载性能
            for symbol in self.test_symbols:
                for timeframe in self.test_timeframes:
                    try:
                        # 记录内存使用
                        memory_before = psutil.Process().memory_info().rss / 1024**2
                        
                        # 测量加载时间
                        start_time = time.time()
                        df = infrastructure.get_data(symbol, timeframe)
                        end_time = time.time()
                        
                        if not df.empty:
                            loading_time = end_time - start_time
                            record_count = len(df)
                            
                            loading_times.append(loading_time)
                            record_counts.append(record_count)
                            
                            # 计算加载速度
                            records_per_second = record_count / loading_time if loading_time > 0 else 0
                            
                            memory_after = psutil.Process().memory_info().rss / 1024**2
                            memory_delta = memory_after - memory_before
                            memory_usage.append(memory_delta)
                            
                            self.logger.info(f"{symbol} {timeframe}: {record_count} 条记录, "
                                           f"{loading_time:.3f}s, {records_per_second:.0f} 记录/秒")
                            
                    except Exception as e:
                        self.logger.warning(f"加载 {symbol} {timeframe} 失败: {e}")
                        
            # 计算统计指标
            if loading_times:
                results['metrics'] = {
                    'total_tests': len(loading_times),
                    'avg_loading_time_s': statistics.mean(loading_times),
                    'median_loading_time_s': statistics.median(loading_times),
                    'p95_loading_time_s': np.percentile(loading_times, 95),
                    'p99_loading_time_s': np.percentile(loading_times, 99),
                    'avg_records_per_second': statistics.mean(
                        [count / time for count, time in zip(record_counts, loading_times) if time > 0]
                    ),
                    'total_records_loaded': sum(record_counts),
                    'avg_memory_usage_mb': statistics.mean(memory_usage) if memory_usage else 0,
                    'max_memory_usage_mb': max(memory_usage) if memory_usage else 0
                }
                
                results['status'] = 'passed'
                
                # 性能判断
                avg_loading_time = results['metrics']['avg_loading_time_s']
                if avg_loading_time > 5.0:
                    results['status'] = 'warning'
                    results['warning'] = f"平均加载时间过长: {avg_loading_time:.3f}s"
                elif avg_loading_time > 10.0:
                    results['status'] = 'failed'
                    results['error'] = f"加载性能不达标: {avg_loading_time:.3f}s"
                    
            else:
                results['status'] = 'failed'
                results['error'] = '未能加载任何数据'
                
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            self.logger.error(f"数据加载性能测试失败: {e}")
            
        return results
        
    async def test_quality_monitoring_performance(self) -> Dict[str, Any]:
        """测试质量监控性能"""
        self.logger.info("开始质量监控性能测试...")
        
        results = {
            'test_name': 'quality_monitoring_performance',
            'status': 'running',
            'metrics': {}
        }
        
        try:
            # 初始化质量监控
            quality_monitor = DataQualityMonitor({
                'auto_repair': True,
                'db_path': 'data/test_quality_monitor.db'
            })
            
            # 初始化基础设施
            infrastructure = ProfessionalDataInfrastructure()
            
            assessment_times = []
            quality_scores = []
            
            # 测试质量评估性能
            for symbol in self.test_symbols[:3]:  # 限制测试数量
                try:
                    # 加载数据
                    df = infrastructure.get_data(symbol, '5m')
                    
                    if not df.empty:
                        # 测量质量评估时间
                        start_time = time.time()
                        metrics = quality_monitor.assess_data_quality(df, symbol, '5m')
                        end_time = time.time()
                        
                        assessment_time = end_time - start_time
                        assessment_times.append(assessment_time)
                        quality_scores.append(metrics.overall_score)
                        
                        # 计算评估速度
                        records_per_second = len(df) / assessment_time if assessment_time > 0 else 0
                        
                        self.logger.info(f"{symbol} 质量评估: {len(df)} 条记录, "
                                       f"{assessment_time:.3f}s, 评分: {metrics.overall_score:.3f}")
                        
                except Exception as e:
                    self.logger.warning(f"质量评估 {symbol} 失败: {e}")
                    
            # 测试报告生成性能
            report_start_time = time.time()
            try:
                report = quality_monitor.get_quality_report(days=1)
                report_generation_time = time.time() - report_start_time
            except Exception as e:
                report_generation_time = -1
                self.logger.warning(f"报告生成失败: {e}")
                
            # 计算统计指标
            if assessment_times:
                results['metrics'] = {
                    'total_assessments': len(assessment_times),
                    'avg_assessment_time_s': statistics.mean(assessment_times),
                    'median_assessment_time_s': statistics.median(assessment_times),
                    'max_assessment_time_s': max(assessment_times),
                    'avg_quality_score': statistics.mean(quality_scores),
                    'min_quality_score': min(quality_scores),
                    'report_generation_time_s': report_generation_time
                }
                
                results['status'] = 'passed'
                
                # 性能判断
                avg_assessment_time = results['metrics']['avg_assessment_time_s']
                if avg_assessment_time > 2.0:
                    results['status'] = 'warning'
                    results['warning'] = f"质量评估时间过长: {avg_assessment_time:.3f}s"
                elif avg_assessment_time > 5.0:
                    results['status'] = 'failed'
                    results['error'] = f"质量监控性能不达标: {avg_assessment_time:.3f}s"
                    
            else:
                results['status'] = 'failed'
                results['error'] = '未能完成任何质量评估'
                
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            self.logger.error(f"质量监控性能测试失败: {e}")
            
        return results
        
    async def test_concurrent_access_performance(self) -> Dict[str, Any]:
        """测试并发访问性能"""
        self.logger.info("开始并发访问性能测试...")
        
        results = {
            'test_name': 'concurrent_access_performance',
            'status': 'running',
            'metrics': {}
        }
        
        try:
            infrastructure = ProfessionalDataInfrastructure()
            
            # 并发测试参数
            concurrent_levels = [1, 5, 10, 20]
            concurrent_results = {}
            
            for concurrency in concurrent_levels:
                self.logger.info(f"测试并发级别: {concurrency}")
                
                # 准备并发任务
                async def load_data_task(symbol, timeframe):
                    try:
                        start_time = time.time()
                        df = infrastructure.get_data(symbol, timeframe)
                        end_time = time.time()
                        return {
                            'success': True,
                            'duration': end_time - start_time,
                            'record_count': len(df) if not df.empty else 0
                        }
                    except Exception as e:
                        return {
                            'success': False,
                            'error': str(e),
                            'duration': 0,
                            'record_count': 0
                        }
                
                # 创建并发任务
                tasks = []
                test_combinations = []
                for i in range(concurrency):
                    symbol = self.test_symbols[i % len(self.test_symbols)]
                    timeframe = self.test_timeframes[i % len(self.test_timeframes)]
                    test_combinations.append((symbol, timeframe))
                    tasks.append(load_data_task(symbol, timeframe))
                
                # 执行并发测试
                start_time = time.time()
                task_results = await asyncio.gather(*tasks, return_exceptions=True)
                total_duration = time.time() - start_time
                
                # 统计结果
                successful_tasks = [r for r in task_results if isinstance(r, dict) and r.get('success', False)]
                failed_tasks = len(task_results) - len(successful_tasks)
                
                if successful_tasks:
                    avg_duration = statistics.mean([r['duration'] for r in successful_tasks])
                    total_records = sum([r['record_count'] for r in successful_tasks])
                    throughput = len(successful_tasks) / total_duration if total_duration > 0 else 0
                    
                    concurrent_results[concurrency] = {
                        'total_tasks': len(task_results),
                        'successful_tasks': len(successful_tasks),
                        'failed_tasks': failed_tasks,
                        'success_rate': len(successful_tasks) / len(task_results),
                        'total_duration_s': total_duration,
                        'avg_task_duration_s': avg_duration,
                        'throughput_tasks_per_s': throughput,
                        'total_records_loaded': total_records
                    }
                else:
                    concurrent_results[concurrency] = {
                        'total_tasks': len(task_results),
                        'successful_tasks': 0,
                        'failed_tasks': failed_tasks,
                        'success_rate': 0,
                        'error': '所有任务失败'
                    }
                    
            results['metrics'] = {
                'concurrent_test_results': concurrent_results,
                'max_successful_concurrency': max([
                    level for level, result in concurrent_results.items()
                    if result.get('success_rate', 0) > 0.8
                ], default=0),
                'peak_throughput_tasks_per_s': max([
                    result.get('throughput_tasks_per_s', 0)
                    for result in concurrent_results.values()
                ], default=0)
            }
            
            results['status'] = 'passed'
            
            # 性能判断
            max_concurrency = results['metrics']['max_successful_concurrency']
            if max_concurrency < 5:
                results['status'] = 'warning'
                results['warning'] = f"并发处理能力较低: {max_concurrency}"
            elif max_concurrency < 2:
                results['status'] = 'failed'
                results['error'] = f"并发性能不达标: {max_concurrency}"
                
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            self.logger.error(f"并发访问性能测试失败: {e}")
            
        return results
        
    async def test_memory_usage(self) -> Dict[str, Any]:
        """测试内存使用情况"""
        self.logger.info("开始内存使用测试...")
        
        results = {
            'test_name': 'memory_usage',
            'status': 'running',
            'metrics': {}
        }
        
        try:
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024**2  # MB
            
            infrastructure = ProfessionalDataInfrastructure()
            memory_samples = [initial_memory]
            
            # 加载大量数据并监控内存
            for symbol in self.test_symbols:
                for timeframe in ['5m', '15m']:
                    try:
                        df = infrastructure.get_data(symbol, timeframe)
                        current_memory = process.memory_info().rss / 1024**2
                        memory_samples.append(current_memory)
                        
                        self.logger.debug(f"加载 {symbol} {timeframe} 后内存: {current_memory:.1f}MB")
                        
                    except Exception as e:
                        self.logger.warning(f"内存测试加载 {symbol} {timeframe} 失败: {e}")
                        
            # 强制垃圾回收
            gc.collect()
            final_memory = process.memory_info().rss / 1024**2
            memory_samples.append(final_memory)
            
            # 计算内存统计
            results['metrics'] = {
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': max(memory_samples),
                'final_memory_mb': final_memory,
                'memory_increase_mb': final_memory - initial_memory,
                'memory_samples_count': len(memory_samples),
                'avg_memory_mb': statistics.mean(memory_samples),
                'memory_efficiency_score': min(1.0, 500 / max(memory_samples)) if memory_samples else 0
            }
            
            results['status'] = 'passed'
            
            # 内存使用判断
            peak_memory = results['metrics']['peak_memory_mb']
            if peak_memory > 1000:  # 1GB
                results['status'] = 'warning'
                results['warning'] = f"内存使用较高: {peak_memory:.1f}MB"
            elif peak_memory > 2000:  # 2GB
                results['status'] = 'failed'
                results['error'] = f"内存使用过高: {peak_memory:.1f}MB"
                
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            self.logger.error(f"内存使用测试失败: {e}")
            
        return results
        
    async def test_storage_efficiency(self) -> Dict[str, Any]:
        """测试存储效率"""
        self.logger.info("开始存储效率测试...")
        
        results = {
            'test_name': 'storage_efficiency',
            'status': 'running',
            'metrics': {}
        }
        
        try:
            # 检查数据存储路径
            storage_path = Path("data/professional_storage")
            
            if not storage_path.exists():
                results['status'] = 'skipped'
                results['reason'] = '存储路径不存在'
                return results
                
            # 计算存储统计
            total_size = 0
            file_count = 0
            file_sizes = []
            
            for file_path in storage_path.rglob("*.parquet"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    file_count += 1
                    file_sizes.append(file_size)
                    
            # 估算原始数据大小（基于记录数）
            estimated_raw_size = 0
            infrastructure = ProfessionalDataInfrastructure()
            
            for symbol in self.test_symbols[:3]:  # 限制测试范围
                try:
                    df = infrastructure.get_data(symbol, '5m')
                    if not df.empty:
                        # 估算：每条记录约40字节(5个float64 = 8*5=40)
                        estimated_raw_size += len(df) * 40
                except Exception:
                    continue
                    
            # 计算压缩比
            compression_ratio = total_size / estimated_raw_size if estimated_raw_size > 0 else 0
            
            results['metrics'] = {
                'total_storage_size_mb': total_size / 1024**2,
                'file_count': file_count,
                'avg_file_size_mb': statistics.mean(file_sizes) / 1024**2 if file_sizes else 0,
                'median_file_size_mb': statistics.median(file_sizes) / 1024**2 if file_sizes else 0,
                'estimated_raw_size_mb': estimated_raw_size / 1024**2,
                'compression_ratio': compression_ratio,
                'storage_efficiency_score': min(1.0, 0.5 / compression_ratio) if compression_ratio > 0 else 0
            }
            
            results['status'] = 'passed'
            
            # 存储效率判断
            if compression_ratio > 0.3:  # 压缩比过低
                results['status'] = 'warning'
                results['warning'] = f"压缩效率较低: {compression_ratio:.3f}"
            elif compression_ratio > 0.5:
                results['status'] = 'failed'
                results['error'] = f"存储效率不达标: {compression_ratio:.3f}"
                
        except Exception as e:
            results['status'] = 'failed'
            results['error'] = str(e)
            self.logger.error(f"存储效率测试失败: {e}")
            
        return results
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有性能测试"""
        self.logger.info("=" * 80)
        self.logger.info("开始DipMaster数据基础设施性能测试套件")
        self.logger.info("=" * 80)
        
        # 测试列表
        test_functions = [
            self.test_data_loading_performance,
            self.test_quality_monitoring_performance,
            self.test_concurrent_access_performance,
            self.test_memory_usage,
            self.test_storage_efficiency
        ]
        
        # 执行测试
        for test_func in test_functions:
            try:
                test_result = await test_func()
                self.test_results['tests'][test_result['test_name']] = test_result
                
                # 输出测试结果
                status = test_result['status']
                status_symbol = "✅" if status == 'passed' else "⚠️" if status == 'warning' else "❌"
                self.logger.info(f"{status_symbol} {test_result['test_name']}: {status.upper()}")
                
                if 'error' in test_result:
                    self.logger.error(f"   错误: {test_result['error']}")
                elif 'warning' in test_result:
                    self.logger.warning(f"   警告: {test_result['warning']}")
                    
            except Exception as e:
                self.logger.error(f"测试 {test_func.__name__} 异常: {e}")
                self.test_results['tests'][test_func.__name__] = {
                    'test_name': test_func.__name__,
                    'status': 'error',
                    'error': str(e)
                }
                
        # 生成测试摘要
        total_tests = len(self.test_results['tests'])
        passed_tests = sum(1 for t in self.test_results['tests'].values() if t['status'] == 'passed')
        warning_tests = sum(1 for t in self.test_results['tests'].values() if t['status'] == 'warning')
        failed_tests = sum(1 for t in self.test_results['tests'].values() if t['status'] in ['failed', 'error'])
        
        self.test_results.update({
            'test_end': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'warning_tests': warning_tests,
                'failed_tests': failed_tests,
                'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
                'overall_status': 'passed' if failed_tests == 0 else 'warning' if passed_tests > failed_tests else 'failed'
            }
        })
        
        # 保存测试结果
        results_path = Path("data/performance_test_results.json")
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, ensure_ascii=False, indent=2, default=str)
            
        self.logger.info("=" * 80)
        self.logger.info("性能测试完成")
        self.logger.info(f"总计: {total_tests} 个测试")
        self.logger.info(f"通过: {passed_tests}, 警告: {warning_tests}, 失败: {failed_tests}")
        self.logger.info(f"成功率: {passed_tests/total_tests:.1%}")
        self.logger.info(f"结果保存: {results_path}")
        self.logger.info("=" * 80)
        
        return self.test_results

async def main():
    """主函数"""
    try:
        test_suite = PerformanceTestSuite()
        results = await test_suite.run_all_tests()
        
        # 输出关键指标
        print("\n📊 关键性能指标:")
        
        if 'data_loading_performance' in results['tests']:
            loading_metrics = results['tests']['data_loading_performance'].get('metrics', {})
            if loading_metrics:
                print(f"   数据加载速度: {loading_metrics.get('avg_records_per_second', 0):.0f} 记录/秒")
                print(f"   平均响应时间: {loading_metrics.get('avg_loading_time_s', 0):.3f} 秒")
                
        if 'concurrent_access_performance' in results['tests']:
            concurrent_metrics = results['tests']['concurrent_access_performance'].get('metrics', {})
            if concurrent_metrics:
                print(f"   最大并发处理: {concurrent_metrics.get('max_successful_concurrency', 0)} 个任务")
                print(f"   峰值吞吐量: {concurrent_metrics.get('peak_throughput_tasks_per_s', 0):.1f} 任务/秒")
                
        if 'memory_usage' in results['tests']:
            memory_metrics = results['tests']['memory_usage'].get('metrics', {})
            if memory_metrics:
                print(f"   峰值内存使用: {memory_metrics.get('peak_memory_mb', 0):.1f} MB")
                print(f"   内存效率评分: {memory_metrics.get('memory_efficiency_score', 0):.3f}")
                
        return results['summary']['overall_status'] == 'passed'
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
        return False
    except Exception as e:
        print(f"\n测试异常: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)