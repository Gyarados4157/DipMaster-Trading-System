#!/usr/bin/env python3
"""
DipMaster Trading System - Secure Main Entry Point
主程序入口点 - 集成企业级安全系统

Features:
- Encrypted API key management
- Access control and authentication  
- Comprehensive security audit logging
- Secure configuration loading
- Enhanced error handling and monitoring

Author: DipMaster Trading Team
Date: 2025-08-13
Version: 4.0.0 - Security Enhanced
"""

import asyncio
import json
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
from datetime import datetime

# Import security modules
from src.security.secure_config_loader import SecureConfigLoader
from src.security.key_manager import ApiKeyManager
from src.security.access_control import AccessController, Permission
from src.security.audit_logger import SecurityAuditLogger

# Import core modules
from src.core.trading_engine import DipMasterTradingEngine
from src.core.dipmaster_live import DipMasterLiveStrategy
from src.dashboard.monitor_dashboard import DashboardServer


class SecureDipMasterSystem:
    """
    Secure DipMaster Trading System with enterprise security.
    
    Integrates all security components for a production-ready
    trading system with comprehensive access control and audit logging.
    """
    
    def __init__(self):
        """Initialize secure trading system."""
        self.config = None
        self.trading_engine = None
        self.dashboard_server = None
        
        # Security components
        self.key_manager = None
        self.access_controller = None  
        self.audit_logger = None
        self.config_loader = None
        
        # System state
        self.running = False
        self.session_id = None
        
        # Setup logging
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """Setup comprehensive logging system."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    log_dir / f"dipmaster_secure_{datetime.now().strftime('%Y%m%d')}.log"
                ),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Set library log levels
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
    
    def initialize_security_system(self) -> bool:
        """Initialize all security components."""
        try:
            self.logger.info("🔒 Initializing security system...")
            
            # Initialize audit logger first
            self.audit_logger = SecurityAuditLogger(
                log_dir="logs/security",
                enable_real_time_alerts=True
            )
            
            # Initialize key manager
            self.key_manager = ApiKeyManager(
                storage_path="config/encrypted_keys.json",
                audit_logger=self.audit_logger
            )
            
            # Initialize access controller
            self.access_controller = AccessController(
                config_path="config/access_control.json",
                audit_logger=self.audit_logger
            )
            
            # Initialize secure config loader
            self.config_loader = SecureConfigLoader(
                key_manager=self.key_manager,
                audit_logger=self.audit_logger
            )
            
            self.logger.info("✅ Security system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize security system: {e}")
            return False
    
    def authenticate_user(self, user_id: Optional[str] = None, 
                         password: Optional[str] = None) -> bool:
        """
        Authenticate user for system access.
        
        Args:
            user_id: User identifier (if None, will prompt)
            password: User password (if None, will prompt securely)
            
        Returns:
            True if authentication successful
        """
        try:
            # For automated/service mode, try environment variables
            if not user_id:
                user_id = os.getenv('DIPMASTER_USER_ID')
            if not password:
                password = os.getenv('DIPMASTER_USER_PASSWORD')
            
            # Interactive authentication if needed
            if not user_id or not password:
                if sys.stdin.isatty():  # Interactive mode
                    import getpass
                    user_id = user_id or input("User ID: ").strip()
                    password = password or getpass.getpass("Password: ")
                else:
                    # Non-interactive mode - require environment variables
                    raise ValueError(
                        "Non-interactive mode requires DIPMASTER_USER_ID and "
                        "DIPMASTER_USER_PASSWORD environment variables"
                    )
            
            # Authenticate with access controller
            self.session_id = self.access_controller.authenticate(
                user_id, password, self._get_client_ip()
            )
            
            if self.session_id:
                session_info = self.access_controller.get_session_info(self.session_id)
                self.logger.info(f"✅ User authenticated: {user_id}")
                self.logger.info(f"📋 User roles: {session_info['roles']}")
                return True
            else:
                self.logger.error("❌ Authentication failed")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Authentication error: {e}")
            return False
    
    def check_permission(self, permission: Permission, resource: str = None) -> bool:
        """Check if current session has required permission."""
        if not self.session_id:
            self.logger.error("❌ No active session for permission check")
            return False
        
        return self.access_controller.check_permission(
            self.session_id, permission, resource
        )
    
    def load_configuration(self, config_path: str) -> bool:
        """
        Load and validate configuration with security checks.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            True if configuration loaded successfully
        """
        try:
            # Check permission to read configuration
            if not self.check_permission(Permission.CONFIG_READ, config_path):
                raise PermissionError("Insufficient permissions to read configuration")
            
            # Load secure configuration
            self.config = self.config_loader.load_config(config_path)
            
            # Additional security validations
            self._validate_security_config()
            
            self.logger.info(f"✅ Configuration loaded successfully: {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            return False
    
    def _validate_security_config(self):
        """Validate security-related configuration settings."""
        # Check if using secure API key method
        api_config = self.config.get('api', {})
        if 'key_id' not in api_config:
            self.logger.warning("⚠️  Using legacy API key configuration - consider upgrading")
        
        # Validate paper trading in production
        trading_config = self.config.get('trading', {})
        if not trading_config.get('paper_trading', True):
            if not self.check_permission(Permission.TRADE_EXECUTE, 'live_trading'):
                raise PermissionError("Insufficient permissions for live trading")
            
            # Additional confirmation for live trading
            confirm = input("⚠️  LIVE TRADING MODE - Are you sure? (type 'CONFIRM'): ")
            if confirm != 'CONFIRM':
                raise ValueError("Live trading not confirmed")
    
    async def start_trading_engine(self) -> bool:
        """
        Start the trading engine with security controls.
        
        Returns:
            True if started successfully
        """
        try:
            # Check trading permissions
            if not self.check_permission(Permission.TRADE_EXECUTE, 'trading_engine'):
                raise PermissionError("Insufficient permissions to start trading")
            
            # Initialize trading engine
            self.trading_engine = DipMasterTradingEngine(self.config)
            
            # Start engine
            await self.trading_engine.start()
            self.running = True
            
            # Log trading start
            self.audit_logger.log_trading_operation(
                'ENGINE_START', 
                'SYSTEM',
                'SUCCESS',
                {
                    'config_file': 'secure',
                    'paper_trading': self.config.get('trading', {}).get('paper_trading', True),
                    'symbols': self.config.get('trading', {}).get('symbols', [])
                }
            )
            
            self.logger.info("⚡ Trading engine started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start trading engine: {e}")
            self.audit_logger.log_trading_operation(
                'ENGINE_START', 'SYSTEM', 'FAILURE', {'error': str(e)}
            )
            return False
    
    async def start_dashboard(self, port: int = 8080) -> bool:
        """
        Start monitoring dashboard with access control.
        
        Args:
            port: Dashboard port number
            
        Returns:
            True if started successfully
        """
        try:
            # Check dashboard permissions
            if not self.check_permission(Permission.DASHBOARD_VIEW, 'monitoring'):
                self.logger.warning("⚠️  No dashboard permissions - skipping dashboard startup")
                return False
            
            # Initialize dashboard with security integration
            self.dashboard_server = DashboardServer(
                port=port,
                access_controller=self.access_controller,
                audit_logger=self.audit_logger
            )
            
            await self.dashboard_server.start()
            
            self.logger.info(f"📊 Monitoring dashboard started on port {port}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start dashboard: {e}")
            return False
    
    async def run_main_loop(self):
        """Main system loop with monitoring and health checks."""
        try:
            self.logger.info("🚀 Starting main system loop")
            
            # Health check interval (5 minutes)
            health_check_interval = 300
            last_health_check = 0
            
            while self.running:
                current_time = asyncio.get_event_loop().time()
                
                # Periodic health checks
                if current_time - last_health_check >= health_check_interval:
                    await self._perform_health_check()
                    last_health_check = current_time
                
                # Check if session is still valid
                if not self._is_session_valid():
                    self.logger.error("❌ Session expired or invalid - shutting down")
                    break
                
                # Main loop sleep
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Shutdown signal received")
        except Exception as e:
            self.logger.error(f"❌ Main loop error: {e}")
        finally:
            await self.shutdown()
    
    async def _perform_health_check(self):
        """Perform periodic health check."""
        try:
            health_status = {
                'timestamp': datetime.now().isoformat(),
                'trading_engine_running': self.trading_engine and self.trading_engine.running,
                'dashboard_running': self.dashboard_server is not None,
                'session_valid': self._is_session_valid(),
                'memory_usage': self._get_memory_usage(),
                'active_positions': self._get_active_positions_count()
            }
            
            self.audit_logger.log_system_event(
                'HEALTH_CHECK',
                health_status,
                'INFO'
            )
            
            # Log critical issues
            if not health_status['session_valid']:
                self.logger.critical("🚨 Session validation failed in health check")
            
            if not health_status['trading_engine_running'] and self.trading_engine:
                self.logger.critical("🚨 Trading engine stopped unexpectedly")
                
        except Exception as e:
            self.logger.error(f"❌ Health check failed: {e}")
    
    def _is_session_valid(self) -> bool:
        """Check if current session is still valid."""
        if not self.session_id:
            return False
        
        session_info = self.access_controller.get_session_info(self.session_id)
        return session_info is not None
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0
    
    def _get_active_positions_count(self) -> int:
        """Get number of active trading positions."""
        try:
            if self.trading_engine and hasattr(self.trading_engine, 'position_manager'):
                return len(self.trading_engine.position_manager.active_positions)
            return 0
        except Exception:
            return 0
    
    def _get_client_ip(self) -> Optional[str]:
        """Get client IP address for audit logging."""
        # In a real deployment, this would get the actual client IP
        return os.getenv('CLIENT_IP', '127.0.0.1')
    
    async def shutdown(self):
        """Graceful system shutdown with security cleanup."""
        try:
            self.logger.info("🛑 Starting graceful shutdown...")
            
            # Stop trading engine
            if self.trading_engine:
                await self.trading_engine.stop()
                self.logger.info("✅ Trading engine stopped")
            
            # Stop dashboard
            if self.dashboard_server:
                await self.dashboard_server.stop()
                self.logger.info("✅ Dashboard stopped")
            
            # Log system shutdown
            if self.audit_logger:
                self.audit_logger.log_system_event(
                    'SYSTEM_SHUTDOWN',
                    {'graceful': True, 'session_id': self.session_id},
                    'INFO'
                )
                
                # Shutdown audit logger
                self.audit_logger.shutdown()
            
            # Logout session
            if self.session_id and self.access_controller:
                self.access_controller.logout(self.session_id)
            
            self.running = False
            self.logger.info("✅ System shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")


def create_argument_parser():
    """Create enhanced command line argument parser."""
    parser = argparse.ArgumentParser(
        description="DipMaster Trading System v4.0.0 - Security Enhanced",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Security Features:
  - Encrypted API key storage
  - Role-based access control  
  - Comprehensive audit logging
  - Session management
  - Real-time security monitoring

Examples:
  # Interactive mode with authentication
  python main_secure.py --config config/dipmaster_secure.json
  
  # Automated mode with environment variables
  export DIPMASTER_USER_ID=trader
  export DIPMASTER_USER_PASSWORD=secure_password
  python main_secure.py --config config/dipmaster_secure.json --no-interactive
  
  # Paper trading mode
  python main_secure.py --config config/dipmaster_secure.json --paper
        """
    )
    
    # Basic arguments
    parser.add_argument('--config', '-c', 
                       default='config/dipmaster_secure.json',
                       help='Configuration file path')
    
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    
    # Trading options
    parser.add_argument('--paper', action='store_true',
                       help='Force paper trading mode')
    
    parser.add_argument('--no-dashboard', action='store_true',
                       help='Disable monitoring dashboard')
    
    # Authentication options
    parser.add_argument('--user-id',
                       help='User ID for authentication')
    
    parser.add_argument('--no-interactive', action='store_true',
                       help='Non-interactive mode (requires env vars)')
    
    # Security options
    parser.add_argument('--create-user',
                       help='Create new user (admin only)')
    
    parser.add_argument('--security-audit', action='store_true',
                       help='Generate security audit report')
    
    return parser


async def main():
    """Enhanced main function with comprehensive security."""
    try:
        # Parse arguments
        parser = create_argument_parser()
        args = parser.parse_args()
        
        # Initialize secure system
        system = SecureDipMasterSystem()
        
        # Display startup banner
        print("\n" + "=" * 80)
        print("🔒 DipMaster Trading System v4.0.0 - Security Enhanced")
        print("=" * 80)
        
        # Initialize security components
        if not system.initialize_security_system():
            print("❌ Failed to initialize security system")
            sys.exit(1)
        
        # Handle special operations
        if args.security_audit:
            # Generate security audit report
            summary = system.audit_logger.get_audit_summary(24)
            print(json.dumps(summary, indent=2))
            return
        
        if args.create_user:
            # Create user functionality (admin only)
            print("👤 User creation functionality would be implemented here")
            return
        
        # Authenticate user
        print("🔐 Authentication required...")
        if not system.authenticate_user(args.user_id):
            print("❌ Authentication failed")
            sys.exit(1)
        
        # Load configuration
        print(f"📋 Loading configuration: {args.config}")
        if not system.load_configuration(args.config):
            print("❌ Failed to load configuration")
            sys.exit(1)
        
        # Apply command line overrides
        if args.paper:
            system.config['trading']['paper_trading'] = True
            print("📄 Paper trading mode enabled")
        
        # Display system information
        trading_mode = "Paper Trading" if system.config.get('trading', {}).get('paper_trading', False) else "LIVE TRADING"
        dashboard_status = "Disabled" if args.no_dashboard else "Enabled"
        symbols = system.config.get('trading', {}).get('symbols', [])
        
        print(f"💼 Trading Mode: {trading_mode}")
        print(f"📊 Dashboard: {dashboard_status}")
        print(f"📈 Trading Pairs: {len(symbols)}")
        print(f"🔧 Config File: {args.config}")
        print("=" * 80)
        
        # Start dashboard (if enabled)
        if not args.no_dashboard:
            dashboard_port = system.config.get('dashboard', {}).get('port', 8080)
            await system.start_dashboard(dashboard_port)
        
        # Start trading engine
        print("⚡ Starting trading engine...")
        if not await system.start_trading_engine():
            print("❌ Failed to start trading engine")
            sys.exit(1)
        
        print("✅ System startup completed successfully")
        print("🚀 DipMaster Trading System is now running...")
        print("Press Ctrl+C to stop the system")
        
        # Run main loop
        await system.run_main_loop()
        
    except KeyboardInterrupt:
        print("\n👋 Shutdown initiated by user")
    except Exception as e:
        print(f"❌ Critical system error: {e}")
        logging.error(f"Critical error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Set event loop policy for Windows compatibility
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run main function
    asyncio.run(main())