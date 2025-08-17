#!/usr/bin/env python3
"""
DipMaster Trading System v1.0.0 - Production Entry Point
生产环境主启动脚本
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.trading_engine import TradingEngine
from src.core.market_regime_detector import MarketRegimeDetector
from src.core.adaptive_parameter_engine import AdaptiveParameterEngine
from src.core.risk_control_manager import RiskControlManager
from src.monitoring.dipmaster_strategy_monitor import DipMasterStrategyMonitor

def setup_logging(log_level="INFO", log_dir="logs"):
    """Setup production logging"""
    Path(log_dir).mkdir(exist_ok=True)
    
    # Main log file
    log_file = Path(log_dir) / f"dipmaster_production_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("DipMaster")
    logger.info(f"DipMaster Trading System v1.0.0 - Production Mode")
    logger.info(f"Log file: {log_file}")
    
    return logger

def load_config(config_path):
    """Load and validate configuration"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required sections
    required_sections = ['trading', 'risk_management', 'strategy', 'exchange']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    return config

def validate_environment():
    """Validate production environment requirements"""
    errors = []
    
    # Check API credentials
    if not os.getenv('BINANCE_API_KEY'):
        errors.append("BINANCE_API_KEY environment variable not set")
    
    if not os.getenv('BINANCE_API_SECRET'):
        errors.append("BINANCE_API_SECRET environment variable not set")
    
    # Check required directories
    required_dirs = ['data', 'logs', 'config']
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            Path(dir_name).mkdir(exist_ok=True)
    
    if errors:
        raise RuntimeError("Environment validation failed:\n" + "\n".join(f"- {error}" for error in errors))

def create_production_engine(config, logger):
    """Create and configure production trading engine"""
    
    # Initialize components
    regime_detector = MarketRegimeDetector()
    parameter_engine = AdaptiveParameterEngine()
    risk_manager = RiskControlManager(config['risk_management'])
    
    # Create trading engine
    engine = TradingEngine(
        config=config,
        regime_detector=regime_detector,
        parameter_engine=parameter_engine,
        risk_manager=risk_manager,
        logger=logger
    )
    
    # Setup monitoring
    monitor = DipMasterStrategyMonitor(config)
    engine.add_monitor(monitor)
    
    return engine

def pre_flight_checks(config, engine, logger):
    """Perform pre-flight safety checks"""
    logger.info("Performing pre-flight checks...")
    
    checks_passed = 0
    total_checks = 8
    
    # Check 1: API Connection
    try:
        if engine.test_api_connection():
            logger.info("✓ API connection successful")
            checks_passed += 1
        else:
            logger.error("✗ API connection failed")
    except Exception as e:
        logger.error(f"✗ API connection error: {e}")
    
    # Check 2: Account Balance
    try:
        balance = engine.get_account_balance()
        min_balance = config['trading']['initial_capital'] * 0.1  # 10% minimum
        if balance >= min_balance:
            logger.info(f"✓ Account balance sufficient: {balance:.2f} USDT")
            checks_passed += 1
        else:
            logger.error(f"✗ Account balance too low: {balance:.2f} USDT (minimum: {min_balance:.2f})")
    except Exception as e:
        logger.error(f"✗ Balance check error: {e}")
    
    # Check 3: Trading Symbols
    try:
        symbols = config['trading']['symbols']
        valid_symbols = engine.validate_symbols(symbols)
        if len(valid_symbols) == len(symbols):
            logger.info(f"✓ All trading symbols valid: {symbols}")
            checks_passed += 1
        else:
            logger.error(f"✗ Invalid symbols found: {set(symbols) - set(valid_symbols)}")
    except Exception as e:
        logger.error(f"✗ Symbol validation error: {e}")
    
    # Check 4: Risk Limits
    try:
        risk_config = config['risk_management']
        if (risk_config['daily_loss_limit_usd'] > 0 and 
            risk_config['max_position_size_usd'] > 0):
            logger.info("✓ Risk limits configured")
            checks_passed += 1
        else:
            logger.error("✗ Risk limits not properly configured")
    except Exception as e:
        logger.error(f"✗ Risk check error: {e}")
    
    # Check 5: Data Availability
    try:
        if engine.check_data_availability():
            logger.info("✓ Market data available")
            checks_passed += 1
        else:
            logger.error("✗ Market data not available")
    except Exception as e:
        logger.error(f"✗ Data check error: {e}")
    
    # Check 6: Strategy Configuration
    try:
        strategy_config = config['strategy']
        if all(key in strategy_config for key in ['entry_params', 'exit_params']):
            logger.info("✓ Strategy configuration valid")
            checks_passed += 1
        else:
            logger.error("✗ Strategy configuration incomplete")
    except Exception as e:
        logger.error(f"✗ Strategy check error: {e}")
    
    # Check 7: Database Connection
    try:
        if engine.test_database_connection():
            logger.info("✓ Database connection successful")
            checks_passed += 1
        else:
            logger.error("✗ Database connection failed")
    except Exception as e:
        logger.error(f"✗ Database check error: {e}")
    
    # Check 8: Emergency Stop Mechanism
    try:
        if engine.test_emergency_stop():
            logger.info("✓ Emergency stop mechanism ready")
            checks_passed += 1
        else:
            logger.error("✗ Emergency stop mechanism not ready")
    except Exception as e:
        logger.error(f"✗ Emergency stop check error: {e}")
    
    # Final assessment
    check_ratio = checks_passed / total_checks
    logger.info(f"Pre-flight checks completed: {checks_passed}/{total_checks} ({check_ratio:.1%})")
    
    if check_ratio < 0.8:
        raise RuntimeError(f"Pre-flight checks failed. Only {checks_passed}/{total_checks} checks passed. Minimum 80% required for production.")
    
    return True

def main():
    """Main production entry point"""
    parser = argparse.ArgumentParser(description="DipMaster Trading System v1.0.0")
    parser.add_argument('--config', default='config/production_500usdt.json', 
                       help='Configuration file path')
    parser.add_argument('--paper', action='store_true', 
                       help='Run in paper trading mode')
    parser.add_argument('--log-level', default='INFO', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip pre-flight checks (not recommended)')
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        logger = setup_logging(args.log_level)
        
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Override for paper trading
        if args.paper:
            config['trading']['mode'] = 'paper'
            config['safety_features']['paper_trading_mode'] = True
            logger.info("*** PAPER TRADING MODE ENABLED ***")
        
        # Validate environment
        logger.info("Validating environment...")
        validate_environment()
        
        # Create trading engine
        logger.info("Initializing trading engine...")
        engine = create_production_engine(config, logger)
        
        # Pre-flight checks
        if not args.skip_checks:
            if not pre_flight_checks(config, engine, logger):
                logger.error("Pre-flight checks failed. Aborting startup.")
                return 1
        else:
            logger.warning("Skipping pre-flight checks (not recommended for production)")
        
        # Display configuration summary
        logger.info("Configuration Summary:")
        logger.info(f"  Mode: {'Paper Trading' if args.paper else 'LIVE TRADING'}")
        logger.info(f"  Capital: {config['trading']['initial_capital']} USDT")
        logger.info(f"  Symbols: {config['trading']['symbols']}")
        logger.info(f"  Max Position: {config['trading']['max_position_size_usd']} USDT")
        logger.info(f"  Daily Loss Limit: {config['risk_management']['daily_loss_limit_usd']} USDT")
        
        # Final confirmation for live trading
        if not args.paper:
            logger.warning("*** LIVE TRADING MODE ***")
            logger.warning("This will trade with real money!")
            logger.warning("Press Ctrl+C within 10 seconds to abort...")
            
            import time
            for i in range(10, 0, -1):
                print(f"Starting in {i} seconds...", end='\r')
                time.sleep(1)
            print("\nStarting live trading...")
        
        # Start trading engine
        logger.info("Starting DipMaster Trading System...")
        engine.start()
        
        # Keep running until interrupted
        try:
            engine.run()
        except KeyboardInterrupt:
            logger.info("Shutdown signal received...")
        except Exception as e:
            logger.error(f"Trading engine error: {e}")
            raise
        finally:
            logger.info("Stopping trading engine...")
            engine.stop()
            logger.info("DipMaster Trading System stopped.")
    
    except Exception as e:
        print(f"Fatal error: {e}")
        if 'logger' in locals():
            logger.error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())