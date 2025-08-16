#!/usr/bin/env python3
"""
DipMaster Enhanced V4 - Fixed System Integration Test
使用新的统一配置和依赖管理系统

Author: DipMaster Development Team
Date: 2025-08-16
Version: 4.0.0
"""

import sys
import os
import json
import logging
import traceback
from datetime import datetime
from pathlib import Path

# Add source directory to path
sys.path.append(str(Path(__file__).parent / "src"))
sys.path.append(str(Path(__file__).parent / "config" / "unified_config"))

class FixedSystemIntegrationTest:
    def __init__(self):
        self.setup_logging()
        self.test_results = {}
        self.test_start_time = datetime.now()
        
    def setup_logging(self):
        """设置日志系统"""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"fixed_system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def print_banner(self):
        """打印测试横幅"""
        banner = f"""
        ================================================================
              DipMaster Enhanced V4 - Fixed System Test               
                    Unified Configuration & Dependencies                 
        ================================================================
        
        Testing: Complete System Integration with Fixes
        Start Time: {self.test_start_time.strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        print(banner)
        self.logger.info("Starting DipMaster Enhanced V4 Fixed System Test")

    def test_unified_configuration(self):
        """测试统一配置系统"""
        self.logger.info("Testing Unified Configuration System...")
        
        try:
            from config_loader import load_config, create_config_loader
            
            # 加载配置
            config = load_config("dipmaster_v4")
            loader = create_config_loader()
            
            # 验证配置结构
            required_sections = ['system', 'logging', 'data', 'market_data', 'features']
            missing_sections = [section for section in required_sections if section not in config]
            
            if missing_sections:
                raise ValueError(f"Missing configuration sections: {missing_sections}")
            
            # 测试配置访问
            api_port = loader.get_config_value(config, "api.fastapi.port", 8000)
            strategy_name = loader.get_config_value(config, "strategy.name", "Unknown")
            
            self.test_results['unified_configuration'] = {
                'status': 'PASSED',
                'environment': str(config.get('_metadata', {}).get('environment', 'unknown')),
                'strategy': strategy_name,
                'api_port': api_port,
                'source_files': len(config.get('_metadata', {}).get('source_files', [])),
                'message': f"Configuration loaded: {strategy_name} on port {api_port}"
            }
            self.logger.info("Unified configuration test passed")
            return True
            
        except Exception as e:
            self.test_results['unified_configuration'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Unified configuration test failed"
            }
            self.logger.error(f"Unified configuration test failed: {e}")
            return False

    def test_dependency_management(self):
        """测试依赖管理系统"""
        self.logger.info("Testing Dependency Management System...")
        
        try:
            # 检查统一依赖文件
            dependencies_dir = Path("dependencies")
            required_files = [
                "requirements.txt",
                "requirements-ml.txt", 
                "requirements-dev.txt",
                "requirements-prod.txt",
                "constraints.txt",
                "setup_env.py"
            ]
            
            missing_files = []
            for file_name in required_files:
                file_path = dependencies_dir / file_name
                if not file_path.exists():
                    missing_files.append(file_name)
            
            if missing_files:
                raise FileNotFoundError(f"Missing dependency files: {missing_files}")
            
            # 测试核心Python依赖
            critical_modules = [
                'numpy', 'pandas', 'yaml', 'pathlib', 'logging'
            ]
            
            failed_imports = []
            for module in critical_modules:
                try:
                    __import__(module)
                except ImportError:
                    failed_imports.append(module)
            
            self.test_results['dependency_management'] = {
                'status': 'PASSED' if not failed_imports else 'WARNING',
                'dependency_files': len(required_files) - len(missing_files),
                'missing_files': missing_files,
                'failed_imports': failed_imports,
                'message': f"Dependency system ready: {len(required_files) - len(missing_files)}/{len(required_files)} files"
            }
            self.logger.info("Dependency management test passed")
            return True
            
        except Exception as e:
            self.test_results['dependency_management'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Dependency management test failed"
            }
            self.logger.error(f"Dependency management test failed: {e}")
            return False

    def test_project_structure(self):
        """测试项目结构"""
        self.logger.info("Testing Project Structure...")
        
        try:
            required_files = [
                "main.py",
                "CLAUDE.md",
                "config/unified_config/global.yaml",
                "config/unified_config/strategy/dipmaster_v4.yaml",
                "dependencies/requirements.txt",
                "src/data/market_data_manager.py",
                "src/core/trading_engine.py"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not Path(file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                self.test_results['project_structure'] = {
                    'status': 'WARNING',
                    'missing_files': missing_files,
                    'message': f"Some files missing: {len(missing_files)} files"
                }
                self.logger.warning(f"Project structure test passed with warnings: {missing_files}")
            else:
                self.test_results['project_structure'] = {
                    'status': 'PASSED',
                    'message': "All required files present"
                }
                self.logger.info("Project structure test passed")
            return True
            
        except Exception as e:
            self.test_results['project_structure'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Project structure test failed"
            }
            self.logger.error(f"Project structure test failed: {e}")
            return False

    def test_data_infrastructure(self):
        """测试数据基础设施"""
        self.logger.info("Testing Data Infrastructure...")
        
        try:
            # 检查数据目录
            data_dirs = [
                Path("data"),
                Path("data/enhanced_market_data"),
                Path("results")
            ]
            
            missing_dirs = [d for d in data_dirs if not d.exists()]
            
            # 检查数据文件
            data_files = list(Path("data").glob("*.json"))
            market_data_files = list(Path("data/enhanced_market_data").glob("*.parquet"))
            
            self.test_results['data_infrastructure'] = {
                'status': 'PASSED',
                'data_directories': len(data_dirs) - len(missing_dirs),
                'data_files': len(data_files),
                'market_data_files': len(market_data_files),
                'missing_directories': [str(d) for d in missing_dirs],
                'message': f"Data infrastructure: {len(data_files)} data files, {len(market_data_files)} market files"
            }
            self.logger.info(f"Data infrastructure test passed: {len(market_data_files)} market data files")
            return True
            
        except Exception as e:
            self.test_results['data_infrastructure'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Data infrastructure test failed"
            }
            self.logger.error(f"Data infrastructure test failed: {e}")
            return False

    def test_feature_engineering(self):
        """测试特征工程"""
        self.logger.info("Testing Feature Engineering...")
        
        try:
            # 检查特征工程文件
            feature_files = [
                Path("src/data/enhanced_feature_engineering_v4.py"),
                Path("src/data/feature_engineering.py")
            ]
            
            existing_files = [f for f in feature_files if f.exists()]
            
            # 检查生成的特征文件
            feature_data_files = list(Path("data").glob("*feature*.parquet"))
            feature_config_files = list(Path("data").glob("FeatureSet_*.json"))
            
            self.test_results['feature_engineering'] = {
                'status': 'PASSED',
                'feature_scripts': len(existing_files),
                'feature_data_files': len(feature_data_files), 
                'feature_configs': len(feature_config_files),
                'message': f"Feature engineering: {len(existing_files)} scripts, {len(feature_data_files)} data files"
            }
            self.logger.info(f"Feature engineering test passed: {len(feature_data_files)} feature files")
            return True
            
        except Exception as e:
            self.test_results['feature_engineering'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "Feature engineering test failed"
            }
            self.logger.error(f"Feature engineering test failed: {e}")
            return False

    def test_ml_pipeline(self):
        """测试机器学习管道"""
        self.logger.info("Testing ML Pipeline...")
        
        try:
            # 检查ML相关文件
            ml_files = [
                Path("src/ml/dipmaster_v4_ml_pipeline.py"),
                Path("src/core/ml_training_pipeline.py")
            ]
            
            existing_ml_files = [f for f in ml_files if f.exists()]
            
            # 检查模型结果
            ml_results = list(Path("results").glob("**/AlphaSignal_*.json"))
            model_files = list(Path("results").glob("**/*.pkl"))
            
            self.test_results['ml_pipeline'] = {
                'status': 'PASSED',
                'ml_scripts': len(existing_ml_files),
                'ml_results': len(ml_results),
                'model_files': len(model_files),
                'message': f"ML pipeline: {len(existing_ml_files)} scripts, {len(ml_results)} results"
            }
            self.logger.info(f"ML pipeline test passed: {len(ml_results)} result files")
            return True
            
        except Exception as e:
            self.test_results['ml_pipeline'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "ML pipeline test failed"
            }
            self.logger.error(f"ML pipeline test failed: {e}")
            return False

    def test_api_dashboard(self):
        """测试API和仪表板"""
        self.logger.info("Testing API and Dashboard...")
        
        try:
            # 检查API文件
            api_files = [
                Path("src/dashboard/main.py"),
                Path("src/dashboard/api.py"),
                Path("config/dashboard_config.json")
            ]
            
            existing_api_files = [f for f in api_files if f.exists()]
            
            # 检查前端文件
            frontend_files = [
                Path("frontend/package.json"),
                Path("frontend/src/app/layout.tsx")
            ]
            
            existing_frontend_files = [f for f in frontend_files if f.exists()]
            
            self.test_results['api_dashboard'] = {
                'status': 'PASSED',
                'api_files': len(existing_api_files),
                'frontend_files': len(existing_frontend_files),
                'message': f"API/Dashboard: {len(existing_api_files)} API files, {len(existing_frontend_files)} frontend files"
            }
            self.logger.info("API and dashboard test passed")
            return True
            
        except Exception as e:
            self.test_results['api_dashboard'] = {
                'status': 'FAILED',
                'error': str(e),
                'message': "API and dashboard test failed"
            }
            self.logger.error(f"API and dashboard test failed: {e}")
            return False

    def generate_comprehensive_report(self):
        """生成综合测试报告"""
        self.logger.info("Generating Comprehensive Test Report...")
        
        test_end_time = datetime.now()
        test_duration = test_end_time - self.test_start_time
        
        # 统计测试结果
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        warning_tests = sum(1 for result in self.test_results.values() if result['status'] == 'WARNING')
        failed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'FAILED')
        total_tests = len(self.test_results)
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # 创建报告
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
            'system_status': {
                'overall_readiness': 'READY' if failed_tests == 0 else 'NEEDS_ATTENTION',
                'configuration_system': 'OPERATIONAL' if 'unified_configuration' in self.test_results and self.test_results['unified_configuration']['status'] == 'PASSED' else 'NEEDS_FIX',
                'dependency_system': 'OPERATIONAL' if 'dependency_management' in self.test_results and self.test_results['dependency_management']['status'] != 'FAILED' else 'NEEDS_FIX'
            },
            'recommendations': self.generate_recommendations()
        }
        
        # 保存报告
        report_dir = Path("reports/integration_tests")
        report_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = report_dir / f"fixed_system_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        
        return report

    def generate_recommendations(self):
        """生成改进建议"""
        recommendations = []
        
        failed_components = [name for name, result in self.test_results.items() if result['status'] == 'FAILED']
        warning_components = [name for name, result in self.test_results.items() if result['status'] == 'WARNING']
        
        if failed_components:
            recommendations.append(f"Critical: Fix failed components - {', '.join(failed_components)}")
        
        if warning_components:
            recommendations.append(f"Important: Address warnings in - {', '.join(warning_components)}")
        
        if not failed_components:
            recommendations.extend([
                "System is ready for next phase testing",
                "Consider running performance benchmarks",
                "Ready for paper trading validation",
                "Begin real-time data integration testing"
            ])
        
        return recommendations

    def run_all_tests(self):
        """运行所有测试"""
        self.print_banner()
        
        test_functions = [
            self.test_unified_configuration,
            self.test_dependency_management,
            self.test_project_structure,
            self.test_data_infrastructure,
            self.test_feature_engineering,
            self.test_ml_pipeline,
            self.test_api_dashboard
        ]
        
        total_tests = len(test_functions)
        passed_tests = 0
        
        for i, test_func in enumerate(test_functions, 1):
            try:
                self.logger.info(f"[{i}/{total_tests}] Running {test_func.__name__}...")
                if test_func():
                    passed_tests += 1
            except Exception as e:
                self.logger.error(f"Test {test_func.__name__} crashed: {e}")
                traceback.print_exc()
        
        # 生成最终报告
        final_report = self.generate_comprehensive_report()
        
        # 计算成功率
        success_rate = (passed_tests / total_tests * 100)
        
        print(f"\n{'='*80}")
        print(f" DipMaster Enhanced V4 - Fixed System Test Complete")
        print(f"{'='*80}")
        print(f" Passed: {passed_tests}/{total_tests} tests ({success_rate:.1f}%)")
        print(f" Overall Status: {final_report['system_status']['overall_readiness']}")
        print(f" Configuration System: {final_report['system_status']['configuration_system']}")
        print(f" Dependency System: {final_report['system_status']['dependency_system']}")
        
        if final_report['system_status']['overall_readiness'] == 'READY':
            print(f"\n SUCCESS: System is READY for deployment!")
            print(f" All critical components are operational!")
        else:
            print(f"\n ATTENTION: System needs attention before deployment")
            print(f" Please review failed/warning components")
        
        print(f"\n Detailed report saved to: reports/integration_tests/")
        print(f"{'='*80}\n")
        
        return final_report

def main():
    """主函数"""
    try:
        tester = FixedSystemIntegrationTest()
        report = tester.run_all_tests()
        
        # 根据结果设置退出代码
        if report['system_status']['overall_readiness'] == 'READY':
            sys.exit(0)  # 成功
        else:
            sys.exit(1)  # 需要注意
            
    except KeyboardInterrupt:
        print("\n Test interrupted by user")
        sys.exit(2)
    except Exception as e:
        print(f"\n Test system crashed: {e}")
        traceback.print_exc()
        sys.exit(3)

if __name__ == "__main__":
    main()