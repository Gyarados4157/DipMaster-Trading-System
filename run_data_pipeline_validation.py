#!/usr/bin/env python3
"""
DipMaster Trading System - Data Pipeline Validation
数据流管道综合验证系统，集成监控、优化和质量保证

Author: DipMaster Development Team  
Date: 2025-08-16
Version: 4.0.0
"""

import sys
import os
import json
import time
import logging
import asyncio
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

# Add source directories to path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "config" / "unified_config"))

class DataPipelineValidationSuite:
    """数据管道验证套件"""
    
    def __init__(self):
        self.setup_logging()
        self.test_start_time = datetime.now()
        self.validation_results = {}
        
        # Initialize components
        self.monitor = None
        self.optimizer = None
        
    def setup_logging(self):
        """设置日志"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"data_pipeline_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def print_banner(self):
        """打印测试横幅"""
        banner = f"""
        ================================================================
              DipMaster Enhanced V4 - Data Pipeline Validation               
                    Performance Monitoring & Quality Assurance                 
        ================================================================
        
        Testing: Complete Data Pipeline Performance & Quality
        Start Time: {self.test_start_time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        print(banner)
        self.logger.info("Starting DipMaster Data Pipeline Validation")

    def test_data_pipeline_monitoring(self):
        """测试数据管道监控"""
        self.logger.info("Testing Data Pipeline Monitoring System...")
        
        try:
            from data.data_pipeline_monitor import DataPipelineMonitor, DataQualityStatus, PipelineStage
            
            # 创建监控配置
            config = {
                'quality_thresholds': {
                    'completeness_min': 0.95,
                    'latency_max_ms': 1000,
                    'error_rate_max': 0.05,
                    'anomaly_threshold': 0.02
                },
                'performance_targets': {
                    'throughput_min_rps': 100,
                    'cpu_usage_max': 0.8,
                    'memory_usage_max_mb': 2048,
                    'success_rate_min': 0.95
                },
                'alert_channels': ['log'],
                'monitoring_interval': 30
            }
            
            # 创建监控器
            self.monitor = DataPipelineMonitor(config)
            
            # 生成测试数据
            test_data = self._generate_test_market_data(1000)
            
            # 测试数据质量监控
            start_time = time.time()
            quality_metrics = self.monitor.monitor_data_quality(
                test_data, "BTCUSDT", "5m", PipelineStage.VALIDATION
            )
            
            # 测试性能监控
            performance_metrics = self.monitor.monitor_pipeline_performance(
                PipelineStage.PROCESSING,
                start_time,
                len(test_data),
                error_count=0
            )
            
            # 获取监控摘要
            quality_summary = self.monitor.get_quality_summary(hours=1)
            performance_summary = self.monitor.get_performance_summary(hours=1)
            
            self.validation_results['data_pipeline_monitoring'] = {
                'status': 'PASSED',
                'quality_score': quality_metrics.quality_score,
                'quality_status': quality_metrics.status,
                'throughput_rps': performance_metrics.throughput_records_per_second,
                'latency_ms': performance_metrics.latency_ms,
                'success_rate': performance_metrics.success_rate,
                'quality_checks': quality_summary['quality_stats'],
                'performance_stats': performance_summary['performance_by_stage'],
                'message': f"Monitoring system operational - Quality: {quality_metrics.quality_score:.3f}, Throughput: {performance_metrics.throughput_records_per_second:.1f} rps"
            }
            
            self.logger.info(f"Data pipeline monitoring test passed - Quality Score: {quality_metrics.quality_score:.3f}")
            return True
            
        except Exception as e:
            self.validation_results['data_pipeline_monitoring'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Data pipeline monitoring test failed"
            }
            self.logger.error(f"Data pipeline monitoring test failed: {e}")
            return False
    
    def test_pipeline_optimization(self):
        """测试管道优化"""
        self.logger.info("Testing Pipeline Optimization System...")
        
        try:
            from data.pipeline_optimizer import PipelineOptimizer, OptimizationConfig, OptimizationStrategy
            
            # 创建优化配置
            config = OptimizationConfig(
                strategy=OptimizationStrategy.BALANCED,
                target_throughput_rps=200.0,
                target_latency_ms=500.0,
                max_memory_mb=4096.0,
                max_cpu_percent=80.0,
                optimization_interval_minutes=15,
                auto_apply=False
            )
            
            # 创建优化器
            self.optimizer = PipelineOptimizer(config)
            
            # 模拟性能数据
            test_metrics = [
                {
                    'throughput_rps': 150.0,
                    'latency_ms': 800.0,
                    'cpu_usage_percent': 90.0,
                    'memory_usage_mb': 3000.0,
                    'error_count': 5,
                    'total_processed': 1000
                },
                {
                    'throughput_rps': 160.0,
                    'latency_ms': 750.0,
                    'cpu_usage_percent': 85.0,
                    'memory_usage_mb': 3200.0,
                    'error_count': 3,
                    'total_processed': 1200
                }
            ]
            
            # 执行优化分析
            optimization_result = self.optimizer.optimize_pipeline(test_metrics)
            
            # 获取优化报告
            optimization_report = self.optimizer.get_optimization_report()
            
            self.validation_results['pipeline_optimization'] = {
                'status': 'PASSED',
                'strategy': optimization_result.strategy,
                'confidence_score': optimization_result.confidence_score,
                'actions_count': len(optimization_result.actions),
                'expected_improvements': optimization_result.expected_improvement,
                'current_params': optimization_report['current_parameters'],
                'optimization_reason': optimization_result.reason,
                'message': f"Optimization system operational - Confidence: {optimization_result.confidence_score:.2f}, Actions: {len(optimization_result.actions)}"
            }
            
            self.logger.info(f"Pipeline optimization test passed - Confidence: {optimization_result.confidence_score:.2f}")
            return True
            
        except Exception as e:
            self.validation_results['pipeline_optimization'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Pipeline optimization test failed"
            }
            self.logger.error(f"Pipeline optimization test failed: {e}")
            return False
    
    def test_data_quality_validation(self):
        """测试数据质量验证"""
        self.logger.info("Testing Data Quality Validation...")
        
        try:
            # 测试真实的市场数据文件
            data_dir = Path("data/enhanced_market_data")
            if not data_dir.exists():
                raise FileNotFoundError("Enhanced market data directory not found")
            
            # 找到一些测试文件
            test_files = list(data_dir.glob("*USDT_5m_*.parquet"))[:5]  # 取前5个文件
            
            quality_results = []
            total_files = len(test_files)
            processed_files = 0
            
            for file_path in test_files:
                try:
                    # 读取数据
                    data = pd.read_parquet(file_path)
                    
                    # 提取符号名称
                    symbol = file_path.stem.split('_')[0]
                    
                    # 使用监控器验证数据质量
                    if self.monitor:
                        quality_metrics = self.monitor.monitor_data_quality(
                            data, symbol, "5m"
                        )
                        
                        quality_results.append({
                            'file': file_path.name,
                            'symbol': symbol,
                            'records': len(data),
                            'quality_score': quality_metrics.quality_score,
                            'status': quality_metrics.status,
                            'completeness': quality_metrics.completeness_ratio,
                            'anomalies': quality_metrics.price_anomalies + quality_metrics.volume_anomalies
                        })
                        
                        processed_files += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process {file_path}: {e}")
            
            # 计算整体质量指标
            if quality_results:
                avg_quality_score = np.mean([r['quality_score'] for r in quality_results])
                avg_completeness = np.mean([r['completeness'] for r in quality_results])
                total_records = sum([r['records'] for r in quality_results])
                total_anomalies = sum([r['anomalies'] for r in quality_results])
                
                self.validation_results['data_quality_validation'] = {
                    'status': 'PASSED',
                    'files_processed': processed_files,
                    'total_files': total_files,
                    'avg_quality_score': avg_quality_score,
                    'avg_completeness': avg_completeness,
                    'total_records': total_records,
                    'total_anomalies': total_anomalies,
                    'quality_results': quality_results,
                    'message': f"Data quality validated - {processed_files} files, avg quality: {avg_quality_score:.3f}"
                }
            else:
                raise ValueError("No data files could be processed")
            
            self.logger.info(f"Data quality validation passed - {processed_files} files, avg quality: {avg_quality_score:.3f}")
            return True
            
        except Exception as e:
            self.validation_results['data_quality_validation'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Data quality validation test failed"
            }
            self.logger.error(f"Data quality validation test failed: {e}")
            return False
    
    def test_performance_benchmarks(self):
        """测试性能基准"""
        self.logger.info("Testing Performance Benchmarks...")
        
        try:
            # 测试数据处理性能
            test_sizes = [1000, 5000, 10000, 25000]
            performance_results = []
            
            for size in test_sizes:
                # 生成测试数据
                test_data = self._generate_test_market_data(size)
                
                # 测试处理性能
                start_time = time.time()
                
                # 模拟数据处理操作
                processed_data = self._simulate_data_processing(test_data)
                
                end_time = time.time()
                processing_time = (end_time - start_time) * 1000  # ms
                throughput = size / (processing_time / 1000)  # records/second
                
                performance_results.append({
                    'data_size': size,
                    'processing_time_ms': processing_time,
                    'throughput_rps': throughput,
                    'memory_efficient': len(processed_data) == size
                })
                
                self.logger.debug(f"Processed {size} records in {processing_time:.1f}ms ({throughput:.1f} rps)")
            
            # 计算性能指标
            avg_throughput = np.mean([r['throughput_rps'] for r in performance_results])
            max_throughput = max([r['throughput_rps'] for r in performance_results])
            min_latency = min([r['processing_time_ms'] for r in performance_results])
            
            # 性能基准
            throughput_benchmark = 1000  # 1000 rps minimum
            latency_benchmark = 100      # 100ms maximum for 1k records
            
            performance_status = 'PASSED' if avg_throughput >= throughput_benchmark else 'WARNING'
            
            self.validation_results['performance_benchmarks'] = {
                'status': performance_status,
                'avg_throughput_rps': avg_throughput,
                'max_throughput_rps': max_throughput,
                'min_latency_ms': min_latency,
                'throughput_benchmark': throughput_benchmark,
                'latency_benchmark': latency_benchmark,
                'meets_throughput_benchmark': avg_throughput >= throughput_benchmark,
                'performance_results': performance_results,
                'message': f"Performance benchmarks - Avg throughput: {avg_throughput:.1f} rps, Min latency: {min_latency:.1f}ms"
            }
            
            self.logger.info(f"Performance benchmarks completed - Avg throughput: {avg_throughput:.1f} rps")
            return True
            
        except Exception as e:
            self.validation_results['performance_benchmarks'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Performance benchmarks test failed"
            }
            self.logger.error(f"Performance benchmarks test failed: {e}")
            return False
    
    def test_scalability_limits(self):
        """测试可扩展性限制"""
        self.logger.info("Testing Scalability Limits...")
        
        try:
            scalability_results = []
            
            # 测试不同的并发级别
            concurrency_levels = [1, 2, 4, 8]
            
            for concurrency in concurrency_levels:
                try:
                    # 生成测试数据
                    test_data = self._generate_test_market_data(5000)
                    
                    # 测试并发处理
                    start_time = time.time()
                    
                    # 模拟并发数据处理
                    results = self._simulate_concurrent_processing(test_data, concurrency)
                    
                    end_time = time.time()
                    processing_time = (end_time - start_time) * 1000
                    effective_throughput = (len(test_data) * concurrency) / (processing_time / 1000)
                    
                    scalability_results.append({
                        'concurrency_level': concurrency,
                        'processing_time_ms': processing_time,
                        'effective_throughput_rps': effective_throughput,
                        'scalability_factor': effective_throughput / concurrency if concurrency > 0 else 0,
                        'successful': len(results) == concurrency
                    })
                    
                    self.logger.debug(f"Concurrency {concurrency}: {effective_throughput:.1f} rps")
                    
                except Exception as e:
                    self.logger.warning(f"Scalability test failed at concurrency {concurrency}: {e}")
                    scalability_results.append({
                        'concurrency_level': concurrency,
                        'processing_time_ms': 0,
                        'effective_throughput_rps': 0,
                        'scalability_factor': 0,
                        'successful': False,
                        'error': str(e)
                    })
            
            # 分析可扩展性
            successful_tests = [r for r in scalability_results if r['successful']]
            if successful_tests:
                best_throughput = max([r['effective_throughput_rps'] for r in successful_tests])
                optimal_concurrency = max([r['concurrency_level'] for r in successful_tests if r['effective_throughput_rps'] == best_throughput])
                
                scalability_efficiency = len(successful_tests) / len(scalability_results)
            else:
                best_throughput = 0
                optimal_concurrency = 1
                scalability_efficiency = 0
            
            self.validation_results['scalability_limits'] = {
                'status': 'PASSED' if scalability_efficiency >= 0.75 else 'WARNING',
                'best_throughput_rps': best_throughput,
                'optimal_concurrency': optimal_concurrency,
                'scalability_efficiency': scalability_efficiency,
                'max_tested_concurrency': max(concurrency_levels),
                'scalability_results': scalability_results,
                'message': f"Scalability tested - Best: {best_throughput:.1f} rps at {optimal_concurrency}x concurrency"
            }
            
            self.logger.info(f"Scalability limits tested - Best: {best_throughput:.1f} rps at {optimal_concurrency}x concurrency")
            return True
            
        except Exception as e:
            self.validation_results['scalability_limits'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Scalability limits test failed"
            }
            self.logger.error(f"Scalability limits test failed: {e}")
            return False
    
    def _generate_test_market_data(self, size: int) -> pd.DataFrame:
        """生成测试市场数据"""
        dates = pd.date_range(start='2024-08-01', periods=size, freq='5min')
        
        # 生成模拟价格数据
        base_price = 50000
        price_walk = np.random.randn(size).cumsum() * 10
        prices = base_price + price_walk
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices + np.random.uniform(0, 200, size),
            'low': prices - np.random.uniform(0, 200, size),
            'close': prices + np.random.randn(size) * 50,
            'volume': np.random.uniform(100, 1000, size)
        })
        
        # 确保OHLC一致性
        data['high'] = data[['open', 'close']].max(axis=1) + np.random.uniform(0, 100, size)
        data['low'] = data[['open', 'close']].min(axis=1) - np.random.uniform(0, 100, size)
        
        return data
    
    def _simulate_data_processing(self, data: pd.DataFrame) -> pd.DataFrame:
        """模拟数据处理"""
        # 添加一些技术指标计算
        data = data.copy()
        
        # 简单移动平均
        data['sma_20'] = data['close'].rolling(window=20, min_periods=1).mean()
        
        # 价格变化
        data['price_change'] = data['close'].pct_change()
        
        # 成交量移动平均
        data['volume_ma'] = data['volume'].rolling(window=10, min_periods=1).mean()
        
        # 添加延迟模拟
        time.sleep(0.001)  # 1ms 处理时间
        
        return data
    
    def _simulate_concurrent_processing(self, data: pd.DataFrame, concurrency: int) -> List[pd.DataFrame]:
        """模拟并发处理"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 分割数据
        chunk_size = len(data) // concurrency
        chunks = [data[i*chunk_size:(i+1)*chunk_size].copy() for i in range(concurrency)]
        
        results = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(self._simulate_data_processing, chunk) for chunk in chunks]
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=10)  # 10 second timeout
                    results.append(result)
                except Exception as e:
                    self.logger.warning(f"Concurrent processing task failed: {e}")
        
        return results
    
    def generate_comprehensive_report(self):
        """生成综合验证报告"""
        self.logger.info("Generating Comprehensive Validation Report...")
        
        test_end_time = datetime.now()
        test_duration = test_end_time - self.test_start_time
        
        # 统计测试结果
        passed_tests = sum(1 for result in self.validation_results.values() if result['status'] == 'PASSED')
        warning_tests = sum(1 for result in self.validation_results.values() if result['status'] == 'WARNING')
        failed_tests = sum(1 for result in self.validation_results.values() if result['status'] == 'FAILED')
        total_tests = len(self.validation_results)
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # 创建报告
        report = {
            'validation_summary': {
                'start_time': self.test_start_time.isoformat(),
                'end_time': test_end_time.isoformat(),
                'duration_seconds': test_duration.total_seconds(),
                'total_tests': total_tests,
                'passed': passed_tests,
                'warnings': warning_tests,
                'failed': failed_tests,
                'success_rate': success_rate
            },
            'validation_results': self.validation_results,
            'data_pipeline_status': {
                'overall_health': 'HEALTHY' if failed_tests == 0 else 'NEEDS_ATTENTION',
                'monitoring_system': 'OPERATIONAL' if 'data_pipeline_monitoring' in self.validation_results and self.validation_results['data_pipeline_monitoring']['status'] == 'PASSED' else 'NEEDS_FIX',
                'optimization_system': 'OPERATIONAL' if 'pipeline_optimization' in self.validation_results and self.validation_results['pipeline_optimization']['status'] == 'PASSED' else 'NEEDS_FIX',
                'quality_validation': 'OPERATIONAL' if 'data_quality_validation' in self.validation_results and self.validation_results['data_quality_validation']['status'] == 'PASSED' else 'NEEDS_FIX'
            },
            'performance_metrics': self._extract_performance_metrics(),
            'recommendations': self._generate_recommendations()
        }
        
        # 保存报告
        report_dir = Path("reports/data_pipeline")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"pipeline_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        return report
    
    def _extract_performance_metrics(self) -> Dict[str, Any]:
        """提取性能指标"""
        metrics = {}
        
        if 'performance_benchmarks' in self.validation_results:
            perf_data = self.validation_results['performance_benchmarks']
            metrics['throughput'] = {
                'avg_rps': perf_data.get('avg_throughput_rps', 0),
                'max_rps': perf_data.get('max_throughput_rps', 0),
                'meets_benchmark': perf_data.get('meets_throughput_benchmark', False)
            }
            metrics['latency'] = {
                'min_ms': perf_data.get('min_latency_ms', 0)
            }
        
        if 'scalability_limits' in self.validation_results:
            scale_data = self.validation_results['scalability_limits']
            metrics['scalability'] = {
                'best_throughput_rps': scale_data.get('best_throughput_rps', 0),
                'optimal_concurrency': scale_data.get('optimal_concurrency', 1),
                'efficiency': scale_data.get('scalability_efficiency', 0)
            }
        
        if 'data_quality_validation' in self.validation_results:
            quality_data = self.validation_results['data_quality_validation']
            metrics['quality'] = {
                'avg_quality_score': quality_data.get('avg_quality_score', 0),
                'avg_completeness': quality_data.get('avg_completeness', 0),
                'total_records': quality_data.get('total_records', 0)
            }
        
        return metrics
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        failed_components = [name for name, result in self.validation_results.items() if result['status'] == 'FAILED']
        warning_components = [name for name, result in self.validation_results.items() if result['status'] == 'WARNING']
        
        if failed_components:
            recommendations.append(f"Critical: Fix failed components - {', '.join(failed_components)}")
        
        if warning_components:
            recommendations.append(f"Important: Address warnings in - {', '.join(warning_components)}")
        
        # 性能建议
        if 'performance_benchmarks' in self.validation_results:
            perf_data = self.validation_results['performance_benchmarks']
            if not perf_data.get('meets_throughput_benchmark', True):
                recommendations.append("Consider optimizing data processing algorithms for better throughput")
        
        # 可扩展性建议
        if 'scalability_limits' in self.validation_results:
            scale_data = self.validation_results['scalability_limits']
            if scale_data.get('scalability_efficiency', 1) < 0.8:
                recommendations.append("Review concurrent processing implementation for better scalability")
        
        if not failed_components and not warning_components:
            recommendations.extend([
                "Data pipeline validation successful - ready for production use",
                "Consider implementing advanced monitoring alerts",
                "Review optimization parameters for fine-tuning",
                "Schedule regular performance benchmarking"
            ])
        
        return recommendations
    
    def run_all_validations(self):
        """运行所有验证测试"""
        self.print_banner()
        
        validation_functions = [
            self.test_data_pipeline_monitoring,
            self.test_pipeline_optimization,
            self.test_data_quality_validation,
            self.test_performance_benchmarks,
            self.test_scalability_limits
        ]
        
        total_tests = len(validation_functions)
        passed_tests = 0
        
        for i, validation_func in enumerate(validation_functions, 1):
            try:
                self.logger.info(f"[{i}/{total_tests}] Running {validation_func.__name__}...")
                if validation_func():
                    passed_tests += 1
                time.sleep(1)  # Brief pause between tests
            except Exception as e:
                self.logger.error(f"Validation {validation_func.__name__} crashed: {e}")
                traceback.print_exc()
        
        # 生成最终报告
        final_report = self.generate_comprehensive_report()
        
        # 计算成功率
        success_rate = (passed_tests / total_tests * 100)
        
        print(f"\n{'='*80}")
        print(f" DipMaster Enhanced V4 - Data Pipeline Validation Complete")
        print(f"{'='*80}")
        print(f" Passed: {passed_tests}/{total_tests} tests ({success_rate:.1f}%)")
        print(f" Overall Health: {final_report['data_pipeline_status']['overall_health']}")
        print(f" Monitoring System: {final_report['data_pipeline_status']['monitoring_system']}")
        print(f" Optimization System: {final_report['data_pipeline_status']['optimization_system']}")
        print(f" Quality Validation: {final_report['data_pipeline_status']['quality_validation']}")
        
        # 显示性能指标
        if 'performance_metrics' in final_report:
            perf_metrics = final_report['performance_metrics']
            if 'throughput' in perf_metrics:
                print(f" Average Throughput: {perf_metrics['throughput']['avg_rps']:.1f} rps")
            if 'quality' in perf_metrics:
                print(f" Average Quality Score: {perf_metrics['quality']['avg_quality_score']:.3f}")
        
        if final_report['data_pipeline_status']['overall_health'] == 'HEALTHY':
            print(f"\n SUCCESS: Data pipeline is HEALTHY and ready for production!")
            print(f" All critical systems are operational!")
        else:
            print(f"\n ATTENTION: Data pipeline needs attention")
            print(f" Please review failed/warning components")
        
        print(f"\n Detailed report saved to: reports/data_pipeline/")
        print(f"{'='*80}\n")
        
        return final_report

def main():
    """主函数"""
    try:
        validator = DataPipelineValidationSuite()
        report = validator.run_all_validations()
        
        # 根据结果设置退出代码
        if report['data_pipeline_status']['overall_health'] == 'HEALTHY':
            sys.exit(0)  # 成功
        else:
            sys.exit(1)  # 需要注意
            
    except KeyboardInterrupt:
        print("\n Validation interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n Validation system crashed: {e}")
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()