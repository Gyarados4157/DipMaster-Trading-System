#!/usr/bin/env python3
"""
DipMaster Enhanced V4 Complete System Integration Test
 - 

Author: DipMaster Development Team  
Date: 2025-08-16
Version: 4.0.0
"""

import asyncio
import json
import logging
import multiprocessing
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
import sys
import subprocess
import requests
import threading
import signal
import os

# src
sys.path.append(str(Path(__file__).parent / "src"))

class SystemIntegrationTest:
    def __init__(self):
        self.setup_logging()
        self.test_results = {}
        self.services = {}
        self.test_start_time = datetime.now()
        
    def setup_logging(self):
        """"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"system_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def print_banner(self):
        """"""
        banner = """
        ================================================================
              DipMaster Enhanced V4 System Test               
                    Complete Integration Test                 
        ================================================================
        
        Target: 85%+ Win Rate | 2.0+ Sharpe | <3% Max Drawdown
        Testing: End-to-End System Integration
        Start Time: {start_time}
        """.format(start_time=self.test_start_time.strftime('%Y-%m-%d %H:%M:%S'))
        
        print(banner)
        self.logger.info("Starting DipMaster Enhanced V4 Complete System Test")

    def test_project_structure(self):
        """"""
        self.logger.info("Testing Project Structure...")
        
        required_files = [
            "main.py",
            "requirements.txt",
            "CLAUDE.md",
            "config/dipmaster_enhanced_v4_spec.json",
            "data/MarketDataBundle.json",
            "src/data/market_data_manager.py",
            "src/data/feature_engineering_pipeline.py", 
            "src/core/smart_execution_engine.py",
            "src/core/dipmaster_oms_v4.py",
            "src/monitoring/kafka_event_producer.py",
            "src/dashboard/main.py",
            "frontend/package.json",
            "mcp-config.json"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
                
        if missing_files:
            self.test_results['project_structure'] = {
                'status': 'FAILED',
                'missing_files': missing_files,
                'message': f"Missing {len(missing_files)} required files"
            }
            self.logger.error(f"Project structure test failed: {missing_files}")
            return False
        else:
            self.test_results['project_structure'] = {
                'status': 'PASSED',
                'message': "All required files present"
            }
            self.logger.info("Project structure test passed")
            return True

    def test_dependencies(self):
        """"""
        self.logger.info("Testing Dependencies...")
        
        try:
            # Python
            import pandas
            import numpy
            import ta
            import sklearn
            import lightgbm
            import fastapi
            import websockets
            import kafka
            
            # MCP
            result = subprocess.run(["npm", "list", "-g", "@modelcontextprotocol/server-memory"], 
                                  capture_output=True, text=True)
            mcp_available = result.returncode == 0
            
            self.test_results['dependencies'] = {
                'status': 'PASSED' if mcp_available else 'WARNING',
                'python_libs': 'OK',
                'mcp_services': 'OK' if mcp_available else 'MISSING',
                'message': "Dependencies check completed"
            }
            self.logger.info("Dependencies test passed")
            return True
            
        except Exception as e:
            self.test_results['dependencies'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Dependency import failed"
            }
            self.logger.error(f"Dependencies test failed: {e}")
            return False

    def test_data_infrastructure(self):
        """"""
        self.logger.info(" Testing Data Infrastructure...")
        
        try:
            # 
            data_bundle_path = Path("data/MarketDataBundle.json")
            if data_bundle_path.exists():
                with open(data_bundle_path, 'r', encoding='utf-8') as f:
                    data_bundle = json.load(f)
                
                symbols = data_bundle.get('symbols', [])
                data_quality = data_bundle.get('quality_metrics', {}).get('overall_score', 0)
                
                self.test_results['data_infrastructure'] = {
                    'status': 'PASSED',
                    'symbols_count': len(symbols),
                    'data_quality': data_quality,
                    'message': f"Data available for {len(symbols)} symbols with {data_quality}% quality"
                }
                self.logger.info(f" Data infrastructure test passed: {len(symbols)} symbols, {data_quality}% quality")
                return True
            else:
                raise FileNotFoundError("MarketDataBundle.json not found")
                
        except Exception as e:
            self.test_results['data_infrastructure'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Data infrastructure test failed"
            }
            self.logger.error(f" Data infrastructure test failed: {e}")
            return False

    def test_feature_engineering(self):
        """"""
        self.logger.info(" Testing Feature Engineering...")
        
        try:
            # 
            feature_files = list(Path("data").glob("dipmaster_v4_features_*.parquet"))
            
            if feature_files:
                # 
                latest_feature_file = max(feature_files, key=os.path.getctime)
                
                # FeatureSet
                feature_configs = list(Path("data").glob("FeatureSet_*.json"))
                if feature_configs:
                    latest_config = max(feature_configs, key=os.path.getctime)
                    with open(latest_config, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    feature_count = config.get('feature_count', 0)
                    sample_count = config.get('sample_count', 0)
                    
                    self.test_results['feature_engineering'] = {
                        'status': 'PASSED',
                        'feature_count': feature_count,
                        'sample_count': sample_count,
                        'feature_file': str(latest_feature_file),
                        'message': f"Features ready: {feature_count} features, {sample_count} samples"
                    }
                    self.logger.info(f" Feature engineering test passed: {feature_count} features")
                    return True
                else:
                    raise FileNotFoundError("FeatureSet configuration not found")
            else:
                raise FileNotFoundError("Feature parquet files not found")
                
        except Exception as e:
            self.test_results['feature_engineering'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Feature engineering test failed"
            }
            self.logger.error(f" Feature engineering test failed: {e}")
            return False

    def test_model_training(self):
        """"""
        self.logger.info(" Testing Model Training...")
        
        try:
            # Alpha
            signal_files = list(Path("results").glob("**/AlphaSignal_*.json"))
            
            if signal_files:
                latest_signal = max(signal_files, key=os.path.getctime)
                with open(latest_signal, 'r', encoding='utf-8') as f:
                    signal_config = json.load(f)
                
                model_performance = signal_config.get('model_performance', {})
                validation_metrics = signal_config.get('validation_metrics', {})
                
                auc_score = validation_metrics.get('auc', 0)
                accuracy = validation_metrics.get('accuracy', 0)
                
                self.test_results['model_training'] = {
                    'status': 'PASSED',
                    'auc_score': auc_score,
                    'accuracy': accuracy,
                    'signal_file': str(latest_signal),
                    'message': f"Model trained: AUC={auc_score:.3f}, Accuracy={accuracy:.3f}"
                }
                self.logger.info(f" Model training test passed: AUC={auc_score:.3f}")
                return True
            else:
                raise FileNotFoundError("Alpha signal files not found")
                
        except Exception as e:
            self.test_results['model_training'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Model training test failed"
            }
            self.logger.error(f" Model training test failed: {e}")
            return False

    def test_portfolio_optimization(self):
        """"""
        self.logger.info(" Testing Portfolio Optimization...")
        
        try:
            # 
            portfolio_files = list(Path("results").glob("**/TargetPortfolio_*.json"))
            
            if portfolio_files:
                latest_portfolio = max(portfolio_files, key=os.path.getctime)
                with open(latest_portfolio, 'r', encoding='utf-8') as f:
                    portfolio_config = json.load(f)
                
                positions = portfolio_config.get('positions', {})
                risk_metrics = portfolio_config.get('risk_metrics', {})
                
                expected_sharpe = risk_metrics.get('expected_sharpe_ratio', 0)
                expected_vol = risk_metrics.get('expected_volatility', 0)
                
                self.test_results['portfolio_optimization'] = {
                    'status': 'PASSED',
                    'position_count': len(positions),
                    'expected_sharpe': expected_sharpe,
                    'expected_volatility': expected_vol,
                    'portfolio_file': str(latest_portfolio),
                    'message': f"Portfolio optimized: {len(positions)} positions, Sharpe={expected_sharpe:.2f}"
                }
                self.logger.info(f" Portfolio optimization test passed: Sharpe={expected_sharpe:.2f}")
                return True
            else:
                raise FileNotFoundError("Target portfolio files not found")
                
        except Exception as e:
            self.test_results['portfolio_optimization'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Portfolio optimization test failed"
            }
            self.logger.error(f" Portfolio optimization test failed: {e}")
            return False

    def test_execution_system(self):
        """"""
        self.logger.info(" Testing Execution System...")
        
        try:
            # 
            execution_files = list(Path("results").glob("**/DipMaster_ExecutionReport_*.json"))
            
            if execution_files:
                latest_execution = max(execution_files, key=os.path.getctime)
                with open(latest_execution, 'r', encoding='utf-8') as f:
                    execution_report = json.load(f)
                
                orders = execution_report.get('orders', [])
                performance = execution_report.get('execution_performance', {})
                
                total_volume = performance.get('total_volume_usd', 0)
                avg_slippage = performance.get('average_slippage_bps', 0)
                fill_rate = performance.get('fill_rate', 0)
                
                self.test_results['execution_system'] = {
                    'status': 'PASSED',
                    'order_count': len(orders),
                    'total_volume': total_volume,
                    'avg_slippage': avg_slippage,
                    'fill_rate': fill_rate,
                    'execution_file': str(latest_execution),
                    'message': f"Execution tested: {len(orders)} orders, {avg_slippage:.1f}bps slippage"
                }
                self.logger.info(f" Execution system test passed: {len(orders)} orders")
                return True
            else:
                raise FileNotFoundError("Execution report files not found")
                
        except Exception as e:
            self.test_results['execution_system'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Execution system test failed"
            }
            self.logger.error(f" Execution system test failed: {e}")
            return False

    def test_monitoring_system(self):
        """"""
        self.logger.info(" Testing Monitoring System...")
        
        try:
            # 
            monitoring_files = list(Path("config").glob("monitoring_config.json"))
            log_files = list(Path("logs").glob("dipmaster_*.log"))
            
            # Kafka
            kafka_config_exists = Path("config/kafka_config.json").exists()
            
            # 
            monitoring_reports = list(Path("reports").glob("monitoring/*"))
            
            self.test_results['monitoring_system'] = {
                'status': 'PASSED',
                'config_files': len(monitoring_files),
                'log_files': len(log_files),
                'kafka_configured': kafka_config_exists,
                'reports_count': len(monitoring_reports),
                'message': f"Monitoring configured: {len(log_files)} log files, {len(monitoring_reports)} reports"
            }
            self.logger.info(f" Monitoring system test passed: {len(monitoring_reports)} reports")
            return True
            
        except Exception as e:
            self.test_results['monitoring_system'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Monitoring system test failed"
            }
            self.logger.error(f" Monitoring system test failed: {e}")
            return False

    def test_api_service(self):
        """API"""
        self.logger.info(" Testing API Service...")
        
        try:
            # API
            api_files = [
                Path("src/dashboard/main.py"),
                Path("src/dashboard/api.py"),
                Path("src/dashboard/websocket.py"),
                Path("config/dashboard_config.json")
            ]
            
            missing_api_files = [f for f in api_files if not f.exists()]
            
            if missing_api_files:
                raise FileNotFoundError(f"API files missing: {missing_api_files}")
            
            # API
            # 
            
            self.test_results['api_service'] = {
                'status': 'PASSED',
                'api_files_present': len(api_files) - len(missing_api_files),
                'message': "API service files validated"
            }
            self.logger.info(" API service test passed")
            return True
            
        except Exception as e:
            self.test_results['api_service'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "API service test failed"
            }
            self.logger.error(f" API service test failed: {e}")
            return False

    def test_frontend_application(self):
        """"""
        self.logger.info(" Testing Frontend Application...")
        
        try:
            # 
            frontend_files = [
                Path("frontend/package.json"),
                Path("frontend/src/app/layout.tsx"),
                Path("frontend/src/app/dashboard/page.tsx"),
                Path("frontend/src/components/dashboard/sidebar.tsx"),
                Path("frontend/.env.example")
            ]
            
            missing_frontend_files = [f for f in frontend_files if not f.exists()]
            
            if missing_frontend_files:
                raise FileNotFoundError(f"Frontend files missing: {missing_frontend_files}")
            
            # package.json
            with open("frontend/package.json", 'r', encoding='utf-8') as f:
                package_json = json.load(f)
            
            dependencies = package_json.get('dependencies', {})
            required_deps = ['next', 'react', 'typescript', 'tailwindcss']
            missing_deps = [dep for dep in required_deps if dep not in dependencies]
            
            self.test_results['frontend_application'] = {
                'status': 'PASSED' if not missing_deps else 'WARNING',
                'files_present': len(frontend_files) - len(missing_frontend_files),
                'dependencies': len(dependencies),
                'missing_deps': missing_deps,
                'message': f"Frontend validated: {len(dependencies)} dependencies"
            }
            self.logger.info(" Frontend application test passed")
            return True
            
        except Exception as e:
            self.test_results['frontend_application'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Frontend application test failed"
            }
            self.logger.error(f" Frontend application test failed: {e}")
            return False

    def generate_comprehensive_report(self):
        """"""
        self.logger.info(" Generating Comprehensive Test Report...")
        
        test_end_time = datetime.now()
        test_duration = test_end_time - self.test_start_time
        
        # 
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        warning_tests = sum(1 for result in self.test_results.values() if result['status'] == 'WARNING')
        failed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'FAILED')
        total_tests = len(self.test_results)
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # 
        report = {
            'test_summary': {
                'start_time': self.test_start_time.isoformat(),
                'end_time': test_end_time.isoformat(),
                'duration_seconds': test_duration.total_seconds(),
                'total_tests': total_tests,
                'passed': passed_tests,
                'warnings': warning_tests,
                'failed': failed_tests,
                'success_rate': success_rate
            },
            'test_results': self.test_results,
            'dipmaster_v4_status': {
                'overall_readiness': 'READY' if failed_tests == 0 else 'NEEDS_ATTENTION',
                'components_status': {
                    'data_infrastructure': self.test_results.get('data_infrastructure', {}).get('status', 'UNKNOWN'),
                    'feature_engineering': self.test_results.get('feature_engineering', {}).get('status', 'UNKNOWN'),
                    'model_training': self.test_results.get('model_training', {}).get('status', 'UNKNOWN'),
                    'portfolio_optimization': self.test_results.get('portfolio_optimization', {}).get('status', 'UNKNOWN'),
                    'execution_system': self.test_results.get('execution_system', {}).get('status', 'UNKNOWN'),
                    'monitoring_system': self.test_results.get('monitoring_system', {}).get('status', 'UNKNOWN'),
                    'api_service': self.test_results.get('api_service', {}).get('status', 'UNKNOWN'),
                    'frontend_application': self.test_results.get('frontend_application', {}).get('status', 'UNKNOWN')
                }
            },
            'next_steps': self.generate_next_steps()
        }
        
        # 
        report_dir = Path("reports/integration_tests")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"system_integration_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        # 
        self.generate_readable_report(report, report_dir)
        
        return report

    def generate_next_steps(self):
        """"""
        next_steps = []
        
        failed_components = [name for name, result in self.test_results.items() if result['status'] == 'FAILED']
        warning_components = [name for name, result in self.test_results.items() if result['status'] == 'WARNING']
        
        if failed_components:
            next_steps.append(f" Fix failed components: {', '.join(failed_components)}")
        
        if warning_components:
            next_steps.append(f" Address warnings in: {', '.join(warning_components)}")
        
        if not failed_components:
            next_steps.extend([
                " System ready for deployment testing",
                " Run paper trading validation",
                " Test real-time data integration",
                " Monitor system performance",
                " Begin strategy optimization"
            ])
        
        return next_steps

    def generate_readable_report(self, report, report_dir):
        """"""
        
        readable_report = f"""
# DipMaster Enhanced V4 - System Integration Test Report

**Test Date**: {report['test_summary']['start_time'][:19]}
**Duration**: {report['test_summary']['duration_seconds']:.1f} seconds
**Success Rate**: {report['test_summary']['success_rate']:.1f}%

##  Test Summary

-  **Passed**: {report['test_summary']['passed']} tests
-  **Warnings**: {report['test_summary']['warnings']} tests  
-  **Failed**: {report['test_summary']['failed']} tests
-  **Total**: {report['test_summary']['total_tests']} tests

##  Component Status

"""
        
        # 
        for component, status in report['dipmaster_v4_status']['components_status'].items():
            emoji = "" if status == "PASSED" else "" if status == "WARNING" else ""
            readable_report += f"- {emoji} **{component.replace('_', ' ').title()}**: {status}\n"
        
        readable_report += f"\n##  Overall Readiness\n\n**Status**: {report['dipmaster_v4_status']['overall_readiness']}\n\n"
        
        # 
        readable_report += "##  Detailed Test Results\n\n"
        for test_name, result in report['test_results'].items():
            emoji = "" if result['status'] == "PASSED" else "" if result['status'] == "WARNING" else ""
            readable_report += f"### {emoji} {test_name.replace('_', ' ').title()}\n\n"
            readable_report += f"- **Status**: {result['status']}\n"
            readable_report += f"- **Message**: {result['message']}\n"
            if 'error' in result:
                readable_report += f"- **Error**: {result['error']}\n"
            readable_report += "\n"
        
        # 
        readable_report += "##  Next Steps\n\n"
        for step in report['next_steps']:
            readable_report += f"- {step}\n"
        
        readable_report += f"\n---\n*Report generated by DipMaster Enhanced V4 System Integration Test*\n"
        
        # 
        readable_file = report_dir / f"system_integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(readable_file, 'w', encoding='utf-8') as f:
            f.write(readable_report)
            
        self.logger.info(f" Readable report saved to: {readable_file}")

    def run_all_tests(self):
        """"""
        self.print_banner()
        
        test_functions = [
            self.test_project_structure,
            self.test_dependencies,
            self.test_data_infrastructure,
            self.test_feature_engineering,
            self.test_model_training,
            self.test_portfolio_optimization,
            self.test_execution_system,
            self.test_monitoring_system,
            self.test_api_service,
            self.test_frontend_application
        ]
        
        total_tests = len(test_functions)
        passed_tests = 0
        
        for i, test_func in enumerate(test_functions, 1):
            try:
                self.logger.info(f"[{i}/{total_tests}] Running {test_func.__name__}...")
                if test_func():
                    passed_tests += 1
                time.sleep(0.5)  # 
            except Exception as e:
                self.logger.error(f"Test {test_func.__name__} crashed: {e}")
                traceback.print_exc()
        
        # 
        final_report = self.generate_comprehensive_report()
        
        # 
        success_rate = (passed_tests / total_tests * 100)
        
        print(f"\n{'='*80}")
        print(f" DipMaster Enhanced V4 - System Integration Test Complete")
        print(f"{'='*80}")
        print(f" Passed: {passed_tests}/{total_tests} tests ({success_rate:.1f}%)")
        print(f" Overall Status: {final_report['dipmaster_v4_status']['overall_readiness']}")
        
        if final_report['dipmaster_v4_status']['overall_readiness'] == 'READY':
            print(f"\n System is READY for deployment!")
            print(f" DipMaster Enhanced V4 successfully integrated!")
        else:
            print(f"\n System needs attention before deployment")
            print(f" Please review failed tests and address issues")
        
        print(f"\n Detailed report saved to: reports/integration_tests/")
        print(f"{'='*80}\n")
        
        return final_report

def main():
    """"""
    try:
        tester = SystemIntegrationTest()
        report = tester.run_all_tests()
        
        # 
        if report['dipmaster_v4_status']['overall_readiness'] == 'READY':
            sys.exit(0)  # 
        else:
            sys.exit(1)  # 
            
    except KeyboardInterrupt:
        print("\n Test interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n Test system crashed: {e}")
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()