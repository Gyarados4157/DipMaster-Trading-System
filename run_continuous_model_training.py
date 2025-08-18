#!/usr/bin/env python3
"""
DipMaster持续模型训练主程序
整合所有优化组件，实现持续训练和验证循环
目标：达到胜率85%+, 夏普比率>1.5, 最大回撤<3%, 年化收益>15%
"""

import os
import sys
import time
import json
import asyncio
import numpy as np
import pandas as pd
import warnings
import schedule
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# 导入自定义模块
from ml.continuous_training_system import ContinuousTrainingSystem, TrainingConfig
from validation.enhanced_time_series_validator import EnhancedTimeSeriesValidator, ValidationConfig
from core.signal_optimization_engine import SignalOptimizationEngine, SignalConfig
from ml.realistic_backtester import RealisticBacktester, TradingCosts, BacktestConfig

warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('continuous_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousModelTrainingOrchestrator:
    """持续模型训练编排器"""
    
    def __init__(self, config_path: str = None):
        """初始化编排器"""
        self.config = self._load_config(config_path)
        self.is_running = False
        self.iteration_count = 0
        self.best_performance = {}
        self.performance_history = []
        
        # 初始化组件
        self._initialize_components()
        
        # 创建输出目录
        self.output_dir = Path("results/continuous_training")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 目标性能指标
        self.target_metrics = {
            'win_rate': 0.85,      # 85%胜率
            'sharpe_ratio': 1.5,   # 夏普比率>1.5
            'max_drawdown': 0.03,  # 最大回撤<3%
            'annual_return': 0.15  # 年化收益>15%
        }
        
        logger.info("🚀 DipMaster持续训练编排器已初始化")
    
    def _load_config(self, config_path: str) -> Dict:
        """加载配置文件"""
        default_config = {
            "training_interval_hours": 2,
            "data_dir": "data/continuous_optimization",
            "max_iterations": 100,
            "early_stopping_patience": 10,
            "performance_threshold": {
                "min_improvement": 0.01,
                "lookback_iterations": 5
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                default_config.update(file_config)
        
        return default_config
    
    def _initialize_components(self):
        """初始化所有组件"""
        # 训练系统配置
        training_config = TrainingConfig(
            target_win_rate=self.target_metrics['win_rate'],
            target_sharpe=self.target_metrics['sharpe_ratio'],
            target_max_drawdown=self.target_metrics['max_drawdown'],
            target_annual_return=self.target_metrics['annual_return']
        )
        
        # 验证系统配置
        validation_config = ValidationConfig(
            n_splits=5,
            embargo_hours=2,
            walk_forward_steps=10
        )
        
        # 信号优化配置
        signal_config = SignalConfig(
            base_threshold=0.5,
            min_signal_strength=0.6,
            max_daily_signals=10
        )
        
        # 回测配置
        costs = TradingCosts(
            maker_fee=0.0010,
            taker_fee=0.0010,
            slippage_base=0.0005
        )
        backtest_config = BacktestConfig(
            initial_capital=10000,
            max_position_size=0.1,
            stop_loss_ratio=0.05,
            take_profit_ratio=0.02
        )
        
        # 创建组件实例
        self.training_system = ContinuousTrainingSystem(training_config)
        self.validator = EnhancedTimeSeriesValidator(validation_config)
        self.signal_optimizer = SignalOptimizationEngine(signal_config)
        self.backtester = RealisticBacktester(costs, backtest_config)
        
        logger.info("✅ 所有组件已初始化")
    
    def run_single_iteration(self) -> Dict:
        """运行单次训练迭代"""
        iteration_start_time = datetime.now()
        self.iteration_count += 1
        
        logger.info(f"\n{'='*50}")
        logger.info(f"🔄 开始第 {self.iteration_count} 次训练迭代")
        logger.info(f"{'='*50}")
        
        try:
            # 1. 加载最新数据
            logger.info("📊 加载最新特征数据...")
            datasets = self.training_system.load_multi_symbol_data(
                self.config["data_dir"]
            )
            
            if not datasets:
                logger.error("❌ 未找到有效数据")
                return {'success': False, 'error': 'No data available'}
            
            # 2. 对每个币种进行训练和验证
            symbol_results = {}
            
            for symbol, data in datasets.items():
                logger.info(f"\n🪙 处理 {symbol}...")
                
                try:
                    # 准备特征和标签
                    X, y_return, y_binary = self.training_system.prepare_features_and_labels(data)
                    
                    if len(X) < 1000:
                        logger.warning(f"⚠️ {symbol} 数据不足 ({len(X)} 样本)")
                        continue
                    
                    # 2.1 训练集成模型
                    logger.info(f"🤖 训练 {symbol} 集成模型...")
                    training_result = self.training_system.train_ensemble_model(X, y_return, y_binary)
                    
                    # 2.2 增强验证
                    logger.info(f"🔍 执行 {symbol} 增强验证...")
                    
                    # 创建模型工厂函数
                    def create_model_factory():
                        import lightgbm as lgb
                        return lambda: lgb.LGBMClassifier(
                            objective='binary',
                            metric='binary_logloss',
                            boosting_type='gbdt',
                            num_leaves=31,
                            learning_rate=0.05,
                            verbose=-1
                        )
                    
                    model_factory = create_model_factory()
                    validation_result = self.validator.comprehensive_validation(
                        X, y_binary, model_factory
                    )
                    
                    # 2.3 信号优化
                    logger.info(f"⚡ 优化 {symbol} 信号...")
                    
                    predictions = training_result['test_data']['predictions']
                    market_data = data.copy()
                    
                    optimized_signals = self.signal_optimizer.generate_optimized_signals(
                        predictions, X, market_data
                    )
                    
                    # 2.4 现实化回测
                    logger.info(f"📈 执行 {symbol} 现实化回测...")
                    
                    if optimized_signals:
                        # 创建信号DataFrame
                        signals_df = pd.DataFrame(optimized_signals)
                        
                        # 准备市场数据
                        market_data_dict = {symbol: market_data}
                        
                        # 运行回测
                        backtest_result = self.backtester.run_backtest(
                            signals_df, market_data_dict
                        )
                    else:
                        backtest_result = {'error': 'No signals generated'}
                    
                    # 2.5 评估性能
                    symbol_performance = self._evaluate_symbol_performance(
                        symbol, training_result, validation_result, backtest_result
                    )
                    
                    symbol_results[symbol] = {
                        'training_result': training_result,
                        'validation_result': validation_result,
                        'backtest_result': backtest_result,
                        'performance': symbol_performance,
                        'signals_count': len(optimized_signals) if optimized_signals else 0
                    }
                    
                    # 检查是否达到目标
                    if symbol_performance.get('targets_achieved', False):
                        logger.info(f"🎉 {symbol} 达到所有目标指标!")
                        self._save_champion_model(symbol, symbol_results[symbol])
                    
                except Exception as e:
                    logger.error(f"❌ 处理 {symbol} 时出错: {e}")
                    continue
            
            # 3. 生成迭代总结
            iteration_summary = self._generate_iteration_summary(
                symbol_results, iteration_start_time
            )
            
            # 4. 保存结果
            self._save_iteration_results(iteration_summary)
            
            # 5. 检查全局目标达成
            if self._check_global_targets_achieved(symbol_results):
                logger.info("🏆 全局目标达成!")
                iteration_summary['global_success'] = True
                return iteration_summary
            
            logger.info(f"✅ 第 {self.iteration_count} 次迭代完成")
            return iteration_summary
            
        except Exception as e:
            logger.error(f"❌ 迭代 {self.iteration_count} 失败: {e}")
            return {
                'success': False, 
                'error': str(e),
                'iteration': self.iteration_count
            }
    
    def _evaluate_symbol_performance(self, symbol: str, training_result: Dict, 
                                   validation_result: Dict, backtest_result: Dict) -> Dict:
        """评估币种性能"""
        
        performance = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'targets_achieved': False
        }
        
        # 从回测结果中提取性能指标
        if 'performance_metrics' in backtest_result and 'error' not in backtest_result:
            metrics = backtest_result['performance_metrics']
            
            performance.update({
                'win_rate': metrics.get('win_rate', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown': abs(metrics.get('max_drawdown', 0)),
                'annual_return': metrics.get('annual_return', 0),
                'total_trades': metrics.get('total_trades', 0),
                'profit_factor': metrics.get('profit_factor', 0),
                'total_return': metrics.get('total_return', 0)
            })
            
            # 检查目标达成情况
            targets_met = {
                'win_rate': performance['win_rate'] >= self.target_metrics['win_rate'],
                'sharpe_ratio': performance['sharpe_ratio'] >= self.target_metrics['sharpe_ratio'],
                'max_drawdown': performance['max_drawdown'] <= self.target_metrics['max_drawdown'],
                'annual_return': performance['annual_return'] >= self.target_metrics['annual_return']
            }
            
            performance['targets_met'] = targets_met
            performance['targets_achieved'] = all(targets_met.values())
            
        else:
            performance['error'] = backtest_result.get('error', 'Unknown backtest error')
        
        # 添加验证结果
        if 'error' not in validation_result:
            performance['validation_stable'] = validation_result.get(
                'stability_analysis', {}
            ).get('overall_stability_score', 0) > 0.7
            
        return performance
    
    def _generate_iteration_summary(self, symbol_results: Dict, start_time: datetime) -> Dict:
        """生成迭代总结"""
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        summary = {
            'iteration': self.iteration_count,
            'timestamp': datetime.now().isoformat(),
            'processing_time_seconds': processing_time,
            'symbols_processed': len(symbol_results),
            'results': symbol_results,
            'performance_summary': {},
            'targets_achieved_count': 0
        }
        
        # 统计性能
        if symbol_results:
            performances = []
            targets_achieved = 0
            
            for symbol, result in symbol_results.items():
                perf = result.get('performance', {})
                if 'error' not in perf:
                    performances.append(perf)
                    if perf.get('targets_achieved', False):
                        targets_achieved += 1
            
            if performances:
                summary['performance_summary'] = {
                    'avg_win_rate': np.mean([p.get('win_rate', 0) for p in performances]),
                    'avg_sharpe_ratio': np.mean([p.get('sharpe_ratio', 0) for p in performances]),
                    'avg_max_drawdown': np.mean([p.get('max_drawdown', 0) for p in performances]),
                    'avg_annual_return': np.mean([p.get('annual_return', 0) for p in performances]),
                    'best_performer': max(performances, key=lambda x: x.get('sharpe_ratio', 0))['symbol'] if performances else None
                }
                
            summary['targets_achieved_count'] = targets_achieved
        
        # 更新性能历史
        self.performance_history.append(summary)
        
        return summary
    
    def _save_champion_model(self, symbol: str, result: Dict):
        """保存冠军模型"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        champion_dir = self.output_dir / f"champion_models/{symbol}_{timestamp}"
        champion_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        if 'training_result' in result and 'models' in result['training_result']:
            import joblib
            
            models = result['training_result']['models']
            for model_name, model in models.items():
                model_path = champion_dir / f"{model_name}_model.pkl"
                joblib.dump(model, model_path)
        
        # 保存结果摘要
        summary_path = champion_dir / "champion_summary.json"
        with open(summary_path, 'w') as f:
            # 清理不可序列化的对象
            clean_result = self._clean_for_json(result)
            json.dump(clean_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"🏆 冠军模型已保存: {champion_dir}")
    
    def _save_iteration_results(self, summary: Dict):
        """保存迭代结果"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = self.output_dir / f"iteration_{self.iteration_count}_{timestamp}.json"
        
        with open(results_path, 'w') as f:
            clean_summary = self._clean_for_json(summary)
            json.dump(clean_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 迭代结果已保存: {results_path}")
    
    def _clean_for_json(self, obj):
        """清理对象以便JSON序列化"""
        if isinstance(obj, dict):
            return {k: self._clean_for_json(v) for k, v in obj.items() 
                   if k not in ['models', 'scalers', 'test_data', 'fold_results']}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, (np.ndarray, pd.Series, pd.DataFrame)):
            return f"<{type(obj).__name__} shape={getattr(obj, 'shape', len(obj))}>"
        elif isinstance(obj, (np.int64, np.float64, np.int32, np.float32)):
            return obj.item()
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def _check_global_targets_achieved(self, symbol_results: Dict) -> bool:
        """检查全局目标是否达成"""
        
        targets_achieved_count = 0
        total_symbols = len(symbol_results)
        
        for symbol, result in symbol_results.items():
            perf = result.get('performance', {})
            if perf.get('targets_achieved', False):
                targets_achieved_count += 1
        
        # 如果至少50%的币种达到目标，或有任何币种表现优异
        min_required = max(1, total_symbols // 2)
        
        if targets_achieved_count >= min_required:
            return True
        
        # 检查是否有表现特别优异的币种
        for symbol, result in symbol_results.items():
            perf = result.get('performance', {})
            if (perf.get('win_rate', 0) > 0.9 and 
                perf.get('sharpe_ratio', 0) > 2.0):
                logger.info(f"🌟 发现超级表现者: {symbol}")
                return True
        
        return False
    
    def run_continuous_loop(self):
        """运行持续训练循环"""
        logger.info("🚀 启动持续训练循环...")
        logger.info(f"🎯 目标指标: 胜率≥{self.target_metrics['win_rate']:.0%}, "
                   f"夏普≥{self.target_metrics['sharpe_ratio']}, "
                   f"回撤≤{self.target_metrics['max_drawdown']:.0%}, "
                   f"年化≥{self.target_metrics['annual_return']:.0%}")
        
        self.is_running = True
        consecutive_failures = 0
        max_failures = 5
        
        try:
            while self.is_running:
                
                # 检查最大迭代数
                if self.iteration_count >= self.config.get("max_iterations", 100):
                    logger.info("🔚 达到最大迭代数，停止训练")
                    break
                
                # 运行单次迭代
                result = self.run_single_iteration()
                
                if result.get('success', True):
                    consecutive_failures = 0
                    
                    # 检查是否达到全局目标
                    if result.get('global_success', False):
                        logger.info("🎉 达到全局目标，训练成功完成!")
                        self._generate_final_report()
                        break
                    
                    # 检查早停条件
                    if self._should_early_stop():
                        logger.info("⏹️ 满足早停条件，停止训练")
                        break
                    
                else:
                    consecutive_failures += 1
                    logger.warning(f"⚠️ 连续失败次数: {consecutive_failures}/{max_failures}")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("❌ 连续失败次数过多，停止训练")
                        break
                
                # 等待下次迭代
                wait_hours = self.config.get("training_interval_hours", 2)
                logger.info(f"⏰ 等待 {wait_hours} 小时后继续下次迭代...")
                
                # 使用非阻塞等待，以便可以优雅停止
                for _ in range(wait_hours * 60):  # 分钟级检查
                    if not self.is_running:
                        break
                    time.sleep(60)  # 等待1分钟
        
        except KeyboardInterrupt:
            logger.info("🛑 用户中断，正在优雅停止...")
        except Exception as e:
            logger.error(f"❌ 持续训练循环出错: {e}")
        finally:
            self.is_running = False
            logger.info("📋 生成最终报告...")
            self._generate_final_report()
    
    def _should_early_stop(self) -> bool:
        """检查是否应该早停"""
        
        patience = self.config.get("early_stopping_patience", 10)
        min_improvement = self.config.get("performance_threshold", {}).get("min_improvement", 0.01)
        
        if len(self.performance_history) < patience:
            return False
        
        # 检查最近几次迭代是否有改善
        recent_performances = self.performance_history[-patience:]
        
        # 使用平均夏普比率作为主要指标
        recent_sharpe_ratios = [
            p.get('performance_summary', {}).get('avg_sharpe_ratio', 0)
            for p in recent_performances
        ]
        
        if not recent_sharpe_ratios:
            return False
        
        # 检查是否有持续改善
        best_recent = max(recent_sharpe_ratios)
        if len(self.performance_history) > patience:
            historical_best = max(
                p.get('performance_summary', {}).get('avg_sharpe_ratio', 0)
                for p in self.performance_history[:-patience]
            )
            
            improvement = best_recent - historical_best
            if improvement < min_improvement:
                logger.info(f"📉 性能改善不足: {improvement:.4f} < {min_improvement}")
                return True
        
        return False
    
    def _generate_final_report(self):
        """生成最终训练报告"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"final_training_report_{timestamp}.json"
        
        # 查找所有冠军模型
        champion_models_dir = self.output_dir / "champion_models"
        champion_models = []
        
        if champion_models_dir.exists():
            for champion_dir in champion_models_dir.iterdir():
                if champion_dir.is_dir():
                    summary_file = champion_dir / "champion_summary.json"
                    if summary_file.exists():
                        try:
                            with open(summary_file, 'r') as f:
                                champion_data = json.load(f)
                                champion_data['model_path'] = str(champion_dir)
                                champion_models.append(champion_data)
                        except Exception as e:
                            logger.warning(f"无法读取冠军模型摘要: {e}")
        
        # 生成综合报告
        final_report = {
            'training_summary': {
                'start_time': self.performance_history[0]['timestamp'] if self.performance_history else None,
                'end_time': datetime.now().isoformat(),
                'total_iterations': self.iteration_count,
                'champion_models_found': len(champion_models),
                'target_metrics': self.target_metrics
            },
            'champion_models': champion_models,
            'performance_evolution': [
                {
                    'iteration': p['iteration'],
                    'timestamp': p['timestamp'],
                    'avg_performance': p.get('performance_summary', {}),
                    'targets_achieved_count': p.get('targets_achieved_count', 0)
                }
                for p in self.performance_history
            ],
            'final_recommendations': self._generate_final_recommendations(champion_models)
        }
        
        # 保存报告
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        # 生成HTML报告
        html_path = self._generate_html_report(final_report, timestamp)
        
        logger.info(f"📊 最终报告已生成:")
        logger.info(f"  JSON: {report_path}")
        logger.info(f"  HTML: {html_path}")
        
        return final_report
    
    def _generate_final_recommendations(self, champion_models: List[Dict]) -> List[str]:
        """生成最终建议"""
        recommendations = []
        
        if not champion_models:
            recommendations.append("❌ 未找到达标的冠军模型，建议：")
            recommendations.append("  1. 增加训练数据量或改善数据质量")
            recommendations.append("  2. 调整特征工程策略")
            recommendations.append("  3. 尝试不同的模型架构")
            recommendations.append("  4. 优化风险管理参数")
        else:
            recommendations.append(f"✅ 发现 {len(champion_models)} 个冠军模型:")
            
            # 分析最佳表现者
            best_model = max(champion_models, 
                           key=lambda x: x.get('performance', {}).get('sharpe_ratio', 0))
            
            recommendations.append(f"🏆 最佳模型表现:")
            perf = best_model.get('performance', {})
            recommendations.append(f"  - 胜率: {perf.get('win_rate', 0):.1%}")
            recommendations.append(f"  - 夏普比率: {perf.get('sharpe_ratio', 0):.2f}")
            recommendations.append(f"  - 最大回撤: {perf.get('max_drawdown', 0):.1%}")
            recommendations.append(f"  - 年化收益: {perf.get('annual_return', 0):.1%}")
            
            recommendations.append("💡 部署建议:")
            recommendations.append("  1. 使用冠军模型进行纸面交易验证")
            recommendations.append("  2. 实施严格的风险控制")
            recommendations.append("  3. 监控模型表现并定期重训练")
            recommendations.append("  4. 考虑多模型集成策略")
        
        return recommendations
    
    def _generate_html_report(self, report_data: Dict, timestamp: str) -> str:
        """生成HTML格式的最终报告"""
        
        html_path = self.output_dir / f"final_training_report_{timestamp}.html"
        
        # 创建性能演化图表数据
        performance_data = report_data.get('performance_evolution', [])
        iterations = [p['iteration'] for p in performance_data]
        avg_win_rates = [p.get('avg_performance', {}).get('avg_win_rate', 0) for p in performance_data]
        avg_sharpe_ratios = [p.get('avg_performance', {}).get('avg_sharpe_ratio', 0) for p in performance_data]
        
        html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DipMaster持续训练最终报告</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }}
        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .card h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
        }}
        .metric-label {{
            font-weight: 600;
        }}
        .metric-value {{
            color: #27ae60;
        }}
        .chart-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .champion-models {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        .champion-model {{
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 15px 0;
            background: #f8fff8;
        }}
        .recommendations {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .recommendations ul {{
            list-style-type: none;
            padding-left: 0;
        }}
        .recommendations li {{
            margin: 8px 0;
            padding: 5px 0;
        }}
        .success {{ color: #27ae60; }}
        .warning {{ color: #f39c12; }}
        .error {{ color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 DipMaster持续训练最终报告</h1>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="summary-cards">
        <div class="card">
            <h3>📊 训练统计</h3>
            <div class="metric">
                <span class="metric-label">总迭代次数:</span>
                <span class="metric-value">{report_data['training_summary']['total_iterations']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">冠军模型数:</span>
                <span class="metric-value">{report_data['training_summary']['champion_models_found']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">训练状态:</span>
                <span class="metric-value {'success' if report_data['training_summary']['champion_models_found'] > 0 else 'warning'}">
                    {'✅ 成功' if report_data['training_summary']['champion_models_found'] > 0 else '⚠️ 部分成功'}
                </span>
            </div>
        </div>
        
        <div class="card">
            <h3>🎯 目标指标</h3>
            <div class="metric">
                <span class="metric-label">目标胜率:</span>
                <span class="metric-value">{report_data['training_summary']['target_metrics']['win_rate']:.0%}</span>
            </div>
            <div class="metric">
                <span class="metric-label">目标夏普比率:</span>
                <span class="metric-value">{report_data['training_summary']['target_metrics']['sharpe_ratio']}</span>
            </div>
            <div class="metric">
                <span class="metric-label">最大回撤限制:</span>
                <span class="metric-value">{report_data['training_summary']['target_metrics']['max_drawdown']:.0%}</span>
            </div>
            <div class="metric">
                <span class="metric-label">目标年化收益:</span>
                <span class="metric-value">{report_data['training_summary']['target_metrics']['annual_return']:.0%}</span>
            </div>
        </div>
    </div>
    
    <div class="chart-container">
        <h3>📈 性能演化趋势</h3>
        <div id="performanceChart"></div>
    </div>
    
    <div class="champion-models">
        <h3>🏆 冠军模型列表</h3>
        {self._generate_champion_models_html(report_data.get('champion_models', []))}
    </div>
    
    <div class="recommendations">
        <h3>💡 最终建议</h3>
        <ul>
            {''.join(f'<li>{rec}</li>' for rec in report_data.get('final_recommendations', []))}
        </ul>
    </div>
    
    <script>
        // 创建性能演化图表
        var trace1 = {{
            x: {iterations},
            y: {avg_win_rates},
            type: 'scatter',
            mode: 'lines+markers',
            name: '平均胜率',
            yaxis: 'y'
        }};
        
        var trace2 = {{
            x: {iterations},
            y: {avg_sharpe_ratios},
            type: 'scatter',
            mode: 'lines+markers',
            name: '平均夏普比率',
            yaxis: 'y2'
        }};
        
        var layout = {{
            title: '训练迭代性能趋势',
            xaxis: {{ title: '迭代次数' }},
            yaxis: {{ 
                title: '胜率',
                side: 'left'
            }},
            yaxis2: {{
                title: '夏普比率',
                side: 'right',
                overlaying: 'y'
            }}
        }};
        
        Plotly.newPlot('performanceChart', [trace1, trace2], layout);
    </script>
</body>
</html>
        """
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(html_path)
    
    def _generate_champion_models_html(self, champion_models: List[Dict]) -> str:
        """生成冠军模型HTML"""
        if not champion_models:
            return "<p>暂无冠军模型达到目标指标。</p>"
        
        html = ""
        for i, model in enumerate(champion_models):
            perf = model.get('performance', {})
            
            html += f"""
            <div class="champion-model">
                <h4>🥇 冠军模型 #{i+1}: {perf.get('symbol', 'Unknown')}</h4>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                    <div class="metric">
                        <span class="metric-label">胜率:</span>
                        <span class="metric-value">{perf.get('win_rate', 0):.1%}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">夏普比率:</span>
                        <span class="metric-value">{perf.get('sharpe_ratio', 0):.2f}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">最大回撤:</span>
                        <span class="metric-value">{perf.get('max_drawdown', 0):.1%}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">年化收益:</span>
                        <span class="metric-value">{perf.get('annual_return', 0):.1%}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">交易次数:</span>
                        <span class="metric-value">{perf.get('total_trades', 0)}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">盈亏比:</span>
                        <span class="metric-value">{perf.get('profit_factor', 0):.2f}</span>
                    </div>
                </div>
                <p><strong>模型路径:</strong> {model.get('model_path', 'N/A')}</p>
            </div>
            """
        
        return html
    
    def stop(self):
        """停止持续训练"""
        logger.info("🛑 收到停止信号...")
        self.is_running = False

def main():
    """主程序入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='DipMaster持续模型训练系统')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--single-run', action='store_true', help='只运行一次迭代')
    
    args = parser.parse_args()
    
    # 创建编排器
    orchestrator = ContinuousModelTrainingOrchestrator(args.config)
    
    try:
        if args.single_run:
            # 单次运行模式
            logger.info("🔄 单次迭代模式")
            result = orchestrator.run_single_iteration()
            
            if result.get('global_success', False):
                logger.info("🎉 单次运行成功达到目标!")
            else:
                logger.info("📊 单次运行完成，可查看结果进行分析")
        
        else:
            # 持续运行模式
            logger.info("🔁 持续训练模式")
            orchestrator.run_continuous_loop()
    
    except KeyboardInterrupt:
        logger.info("🛑 用户中断程序")
        orchestrator.stop()
    
    except Exception as e:
        logger.error(f"❌ 程序运行出错: {e}")
        raise
    
    finally:
        logger.info("👋 程序退出")

if __name__ == "__main__":
    main()