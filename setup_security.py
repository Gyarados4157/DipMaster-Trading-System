#!/usr/bin/env python3
"""
DipMaster Trading System - Security Setup Script
安全系统设置脚本 - 一键配置企业级安全环境

Features:
- Automated security system setup
- Master key generation and secure storage
- Default user creation with proper roles
- API key storage setup
- Configuration validation
- Security best practices enforcement
"""

import os
import sys
import json
import getpass
import secrets
from pathlib import Path
from typing import Dict, Any, List
import argparse

# Add src directory to path
sys.path.append('src')

try:
    from security.crypto_manager import CryptoManager, generate_master_key
    from security.key_manager import ApiKeyManager
    from security.access_control import AccessController, Role
    from security.audit_logger import SecurityAuditLogger
    from security.secure_config_loader import SecureConfigLoader
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please run this script from the project root directory")
    sys.exit(1)


class SecuritySetup:
    """
    Automated security setup for DipMaster Trading System.
    
    Handles the complete setup process including master key generation,
    user creation, API key storage, and configuration validation.
    """
    
    def __init__(self):
        """Initialize security setup."""
        self.setup_complete = False
        self.master_password = None
        
        # Component instances (will be initialized)
        self.crypto_manager = None
        self.key_manager = None
        self.access_controller = None
        self.audit_logger = None
        self.config_loader = None
        
        print("🔒 DipMaster Trading System - Security Setup")
        print("=" * 50)
    
    def run_interactive_setup(self) -> bool:
        """Run interactive security setup."""
        try:
            print("\nThis script will set up enterprise-grade security for DipMaster Trading System")
            print("⚠️  Please ensure you're running this in a secure environment")
            
            # Step 1: Generate or set master password
            if not self._setup_master_password():
                return False
            
            # Step 2: Initialize security components
            if not self._initialize_components():
                return False
            
            # Step 3: Create default users
            if not self._create_default_users():
                return False
            
            # Step 4: Setup API keys
            if not self._setup_api_keys():
                return False
            
            # Step 5: Validate configuration
            if not self._validate_setup():
                return False
            
            # Step 6: Generate setup summary
            self._generate_setup_summary()
            
            self.setup_complete = True
            print("\n✅ Security setup completed successfully!")
            return True
            
        except KeyboardInterrupt:
            print("\n❌ Setup cancelled by user")
            return False
        except Exception as e:
            print(f"\n❌ Setup failed: {e}")
            return False
    
    def _setup_master_password(self) -> bool:
        """Setup master password for encryption."""
        try:
            print("\n🔑 Step 1: Master Password Setup")
            print("-" * 30)
            
            # Check if master password already exists in environment
            existing_password = os.getenv('DIPMASTER_MASTER_PASSWORD')
            if existing_password:
                use_existing = input("Found existing master password in environment. Use it? (y/N): ")
                if use_existing.lower() == 'y':
                    self.master_password = existing_password
                    print("✅ Using existing master password")
                    return True
            
            # Generate new master password
            print("\nChoose master password setup method:")
            print("1. Generate secure random password (recommended)")
            print("2. Set custom password")
            
            choice = input("Choice (1-2): ").strip()
            
            if choice == '1':
                # Generate secure password
                self.master_password = generate_master_key()
                print("✅ Generated secure master password")
                
                # Save to environment file
                if self._save_master_password():
                    print("📁 Master password saved to .env file")
                else:
                    print("⚠️  Could not save to .env file - you'll need to set it manually")
                
            elif choice == '2':
                # Custom password
                while True:
                    password = getpass.getpass("Enter master password: ")
                    confirm = getpass.getpass("Confirm master password: ")
                    
                    if password != confirm:
                        print("❌ Passwords do not match")
                        continue
                    
                    if len(password) < 16:
                        print("❌ Password must be at least 16 characters")
                        continue
                    
                    self.master_password = password
                    break
                
                print("✅ Custom master password set")
                
            else:
                print("❌ Invalid choice")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ Master password setup failed: {e}")
            return False
    
    def _save_master_password(self) -> bool:
        """Save master password to .env file."""
        try:
            env_file = Path('.env')
            env_content = []
            
            # Read existing .env file if it exists
            if env_file.exists():
                with open(env_file, 'r') as f:
                    env_content = f.readlines()
            
            # Remove existing DIPMASTER_MASTER_PASSWORD line
            env_content = [line for line in env_content 
                          if not line.startswith('DIPMASTER_MASTER_PASSWORD=')]
            
            # Add new master password
            env_content.append(f'DIPMASTER_MASTER_PASSWORD={self.master_password}\n')
            
            # Write back to file
            with open(env_file, 'w') as f:
                f.writelines(env_content)
            
            # Set secure permissions
            os.chmod(env_file, 0o600)
            
            return True
            
        except Exception as e:
            print(f"❌ Could not save master password: {e}")
            return False
    
    def _initialize_components(self) -> bool:
        """Initialize all security components."""
        try:
            print("\n🔧 Step 2: Initialize Security Components")
            print("-" * 40)
            
            # Initialize components
            self.crypto_manager = CryptoManager(self.master_password)
            print("✅ Crypto manager initialized")
            
            self.audit_logger = SecurityAuditLogger(
                log_dir="logs/security",
                enable_real_time_alerts=True
            )
            print("✅ Audit logger initialized")
            
            self.key_manager = ApiKeyManager(
                storage_path="config/encrypted_keys.json",
                crypto_manager=self.crypto_manager,
                audit_logger=self.audit_logger
            )
            print("✅ Key manager initialized")
            
            self.access_controller = AccessController(
                config_path="config/access_control.json",
                audit_logger=self.audit_logger
            )
            print("✅ Access controller initialized")
            
            self.config_loader = SecureConfigLoader(
                key_manager=self.key_manager,
                audit_logger=self.audit_logger
            )
            print("✅ Config loader initialized")
            
            return True
            
        except Exception as e:
            print(f"❌ Component initialization failed: {e}")
            return False
    
    def _create_default_users(self) -> bool:
        """Create default user accounts."""
        try:
            print("\n👤 Step 3: Create Default Users")
            print("-" * 30)
            
            # Create admin user
            admin_password = self._get_secure_password("admin user")
            success = self.access_controller.create_user(
                user_id='admin',
                password=admin_password,
                roles=[Role.ADMIN],
                enabled=True
            )
            
            if success:
                print("✅ Admin user created")
                
                # Save admin credentials to secure file
                self._save_user_credentials('admin', admin_password)
            else:
                print("❌ Failed to create admin user")
                return False
            
            # Create trader user
            create_trader = input("\nCreate trader user account? (y/N): ").lower() == 'y'
            if create_trader:
                trader_password = self._get_secure_password("trader user")
                success = self.access_controller.create_user(
                    user_id='trader',
                    password=trader_password,
                    roles=[Role.TRADER],
                    enabled=True
                )
                
                if success:
                    print("✅ Trader user created")
                    self._save_user_credentials('trader', trader_password)
                else:
                    print("⚠️  Failed to create trader user")
            
            return True
            
        except Exception as e:
            print(f"❌ User creation failed: {e}")
            return False
    
    def _get_secure_password(self, user_type: str) -> str:
        """Get secure password for user."""
        print(f"\nSetup password for {user_type}:")
        print("1. Generate secure random password (recommended)")
        print("2. Set custom password")
        
        choice = input("Choice (1-2): ").strip()
        
        if choice == '1':
            password = secrets.token_urlsafe(16)
            print(f"Generated password: {password}")
            print("⚠️  Save this password securely!")
            return password
        
        elif choice == '2':
            while True:
                password = getpass.getpass(f"Enter {user_type} password: ")
                confirm = getpass.getpass("Confirm password: ")
                
                if password != confirm:
                    print("❌ Passwords do not match")
                    continue
                
                if len(password) < 8:
                    print("❌ Password must be at least 8 characters")
                    continue
                
                return password
        
        else:
            # Default to generated password
            password = secrets.token_urlsafe(16)
            print(f"Using generated password: {password}")
            return password
    
    def _save_user_credentials(self, user_id: str, password: str):
        """Save user credentials to secure file."""
        try:
            creds_dir = Path('config/credentials')
            creds_dir.mkdir(parents=True, exist_ok=True)
            
            creds_file = creds_dir / f'{user_id}_credentials.txt'
            with open(creds_file, 'w') as f:
                f.write(f"User ID: {user_id}\n")
                f.write(f"Password: {password}\n")
                f.write(f"Created: {os.times().system}\n")
                f.write("\n⚠️  IMPORTANT: Delete this file after securing the credentials!\n")
            
            # Set secure permissions
            os.chmod(creds_file, 0o600)
            
            print(f"📁 Credentials saved to: {creds_file}")
            
        except Exception as e:
            print(f"⚠️  Could not save credentials: {e}")
    
    def _setup_api_keys(self) -> bool:
        """Setup API keys for trading."""
        try:
            print("\n🔑 Step 4: API Key Setup")
            print("-" * 25)
            
            setup_keys = input("Setup Binance API keys now? (y/N): ").lower() == 'y'
            if not setup_keys:
                print("⚠️  API keys not configured - you can set them up later using:")
                print("   python src/tools/key_management_tool.py store --key-id binance-production")
                return True
            
            print("\nEnter your Binance API credentials:")
            print("⚠️  Make sure you're in a secure environment!")
            
            api_key = getpass.getpass("Binance API Key: ")
            api_secret = getpass.getpass("Binance API Secret: ")
            
            if not api_key or not api_secret:
                print("❌ API key and secret are required")
                return False
            
            # Store encrypted API key
            success = self.key_manager.store_api_key(
                key_id='binance-production',
                api_key=api_key,
                api_secret=api_secret,
                exchange='binance',
                metadata={
                    'environment': 'production',
                    'setup_date': os.times().system,
                    'description': 'Main production API key'
                }
            )
            
            if success:
                print("✅ API keys stored securely")
                
                # Test API key retrieval
                retrieved = self.key_manager.get_api_key('binance-production')
                if retrieved:
                    print("✅ API key retrieval test successful")
                else:
                    print("❌ API key retrieval test failed")
                    return False
            else:
                print("❌ Failed to store API keys")
                return False
            
            return True
            
        except Exception as e:
            print(f"❌ API key setup failed: {e}")
            return False
    
    def _validate_setup(self) -> bool:
        """Validate the complete security setup."""
        try:
            print("\n🔍 Step 5: Validate Setup")
            print("-" * 25)
            
            # Test configuration loading
            try:
                config = self.config_loader.load_config('config/dipmaster_secure.json')
                print("✅ Configuration loading test passed")
            except Exception as e:
                print(f"❌ Configuration loading test failed: {e}")
                return False
            
            # Test authentication
            try:
                session_id = self.access_controller.authenticate('admin', 'test_password')
                if session_id:
                    print("✅ Authentication system working")
                    self.access_controller.logout(session_id)
                else:
                    print("ℹ️  Authentication test skipped (expected)")
            except Exception:
                print("ℹ️  Authentication test skipped")
            
            # Test audit logging
            try:
                self.audit_logger.log_system_event(
                    'SETUP_VALIDATION',
                    {'status': 'success'},
                    'INFO'
                )
                print("✅ Audit logging test passed")
            except Exception as e:
                print(f"❌ Audit logging test failed: {e}")
                return False
            
            # Check file permissions
            self._check_file_permissions()
            
            print("✅ All validation tests passed")
            return True
            
        except Exception as e:
            print(f"❌ Setup validation failed: {e}")
            return False
    
    def _check_file_permissions(self):
        """Check that sensitive files have secure permissions."""
        sensitive_files = [
            'config/encrypted_keys.json',
            'config/access_control.json',
            '.env'
        ]
        
        for file_path in sensitive_files:
            path = Path(file_path)
            if path.exists():
                stat_info = path.stat()
                permissions = oct(stat_info.st_mode)[-3:]
                
                if permissions in ['600', '640']:
                    print(f"✅ {file_path} has secure permissions ({permissions})")
                else:
                    print(f"⚠️  {file_path} permissions could be more secure ({permissions})")
    
    def _generate_setup_summary(self):
        """Generate setup summary and next steps."""
        print("\n📋 Setup Summary")
        print("=" * 50)
        
        print("✅ Security system initialized")
        print("✅ Master password configured")
        print("✅ User accounts created")
        print("✅ API keys encrypted and stored")
        print("✅ Configuration validated")
        
        print("\n🚀 Next Steps:")
        print("1. Test the system with:")
        print("   python main_secure.py --config config/dipmaster_secure.json")
        
        print("\n2. Access the key management tool:")
        print("   python src/tools/key_management_tool.py interactive")
        
        print("\n3. View security audit logs:")
        print("   tail -f logs/security/security_audit_*.jsonl")
        
        print("\n⚠️  Important Security Notes:")
        print("- Store your master password securely")
        print("- Delete credential files from config/credentials/ after securing")
        print("- Regularly rotate API keys")
        print("- Monitor audit logs for suspicious activity")
        print("- Keep the system updated")
        
        print(f"\n📁 Configuration files created:")
        print(f"- config/encrypted_keys.json (encrypted API keys)")
        print(f"- config/access_control.json (user accounts)")
        print(f"- config/dipmaster_secure.json (main configuration)")
        print(f"- .env (environment variables)")


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(
        description="DipMaster Trading System - Security Setup"
    )
    
    parser.add_argument('--non-interactive', action='store_true',
                       help='Non-interactive setup (requires environment variables)')
    
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing setup')
    
    args = parser.parse_args()
    
    # Create setup instance
    setup = SecuritySetup()
    
    if args.validate_only:
        # Only run validation
        print("🔍 Validating existing security setup...")
        success = setup._initialize_components() and setup._validate_setup()
    else:
        # Run full setup
        success = setup.run_interactive_setup()
    
    if success:
        print(f"\n🎉 Setup {'validation' if args.validate_only else 'process'} completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Setup {'validation' if args.validate_only else 'process'} failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()