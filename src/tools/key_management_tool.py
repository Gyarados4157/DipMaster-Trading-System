#!/usr/bin/env python3
"""
API Key Management Tool for DipMaster Trading System
API密钥管理工具 - 命令行工具用于安全管理API密钥

Features:
- Interactive key management CLI
- Secure key storage and retrieval
- Key rotation and lifecycle management
- Backup and restore capabilities
- Security audit and monitoring
"""

import os
import sys
import json
import getpass
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from security.key_manager import ApiKeyManager
from security.crypto_manager import CryptoManager
from security.audit_logger import SecurityAuditLogger

logger = logging.getLogger(__name__)


class KeyManagementTool:
    """
    Command-line tool for API key management.
    
    Provides interactive interface for secure API key operations
    including storage, retrieval, rotation, and backup.
    """
    
    def __init__(self):
        """Initialize key management tool."""
        self.key_manager = None
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def initialize_key_manager(self, storage_path: Optional[str] = None) -> bool:
        """Initialize key manager with proper authentication."""
        try:
            print("🔐 Initializing DipMaster Key Management System")
            
            # Get master password
            master_password = os.getenv('DIPMASTER_MASTER_PASSWORD')
            if not master_password:
                print("Master password not found in environment.")
                master_password = getpass.getpass("Enter master password: ")
            
            # Initialize components
            crypto_manager = CryptoManager(master_password)
            audit_logger = SecurityAuditLogger()
            self.key_manager = ApiKeyManager(storage_path, crypto_manager, audit_logger)
            
            print("✅ Key management system initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize key manager: {e}")
            return False
    
    def store_key(self, args):
        """Store new API key."""
        try:
            key_id = args.key_id or input("Enter key ID: ").strip()
            exchange = args.exchange or input("Enter exchange (default: binance): ").strip() or "binance"
            
            # Get API credentials securely
            if args.api_key and args.api_secret:
                api_key = args.api_key
                api_secret = args.api_secret
            else:
                print("Enter API credentials:")
                api_key = getpass.getpass("API Key: ")
                api_secret = getpass.getpass("API Secret: ")
            
            # Validate inputs
            if not key_id or not api_key or not api_secret:
                print("❌ All fields are required")
                return False
            
            # Additional metadata
            metadata = {}
            if args.description:
                metadata['description'] = args.description
            if args.environment:
                metadata['environment'] = args.environment
            
            # Store key
            success = self.key_manager.store_api_key(
                key_id, api_key, api_secret, exchange, metadata
            )
            
            if success:
                print(f"✅ API key '{key_id}' stored successfully")
            else:
                print("❌ Failed to store API key")
            
            return success
            
        except Exception as e:
            print(f"❌ Error storing key: {e}")
            return False
    
    def retrieve_key(self, args):
        """Retrieve API key."""
        try:
            key_id = args.key_id or input("Enter key ID: ").strip()
            
            if not key_id:
                print("❌ Key ID is required")
                return False
            
            # Retrieve key
            credentials = self.key_manager.get_api_key(key_id)
            
            if credentials:
                api_key, api_secret = credentials
                
                if args.show_secret:
                    print(f"API Key: {api_key}")
                    print(f"API Secret: {api_secret}")
                else:
                    print(f"API Key: {api_key}")
                    print(f"API Secret: {'*' * (len(api_secret) - 8) + api_secret[-8:]}")
                
                return True
            else:
                print(f"❌ API key '{key_id}' not found")
                return False
                
        except Exception as e:
            print(f"❌ Error retrieving key: {e}")
            return False
    
    def list_keys(self, args):
        """List stored API keys."""
        try:
            exchange_filter = args.exchange
            keys = self.key_manager.list_keys(exchange_filter)
            
            if not keys:
                print("No API keys found")
                return True
            
            print(f"\n📋 Found {len(keys)} API key(s):")
            print("-" * 80)
            
            for key_info in keys:
                print(f"Key ID: {key_info['key_id']}")
                print(f"Exchange: {key_info['exchange']}")
                print(f"Status: {key_info['status']}")
                print(f"Created: {key_info['created_at']}")
                print(f"Last Used: {key_info['last_used'] or 'Never'}")
                print(f"Use Count: {key_info['use_count']}")
                
                if key_info.get('metadata'):
                    print(f"Metadata: {json.dumps(key_info['metadata'], indent=2)}")
                
                print("-" * 80)
            
            return True
            
        except Exception as e:
            print(f"❌ Error listing keys: {e}")
            return False
    
    def delete_key(self, args):
        """Delete API key."""
        try:
            key_id = args.key_id or input("Enter key ID to delete: ").strip()
            
            if not key_id:
                print("❌ Key ID is required")
                return False
            
            # Confirmation
            if not args.force:
                confirm = input(f"Are you sure you want to delete key '{key_id}'? (y/N): ")
                if confirm.lower() != 'y':
                    print("Operation cancelled")
                    return False
            
            # Delete key
            success = self.key_manager.delete_key(key_id)
            
            if success:
                print(f"✅ API key '{key_id}' deleted successfully")
            else:
                print(f"❌ Failed to delete API key '{key_id}'")
            
            return success
            
        except Exception as e:
            print(f"❌ Error deleting key: {e}")
            return False
    
    def rotate_key(self, args):
        """Rotate API key with new credentials."""
        try:
            key_id = args.key_id or input("Enter key ID to rotate: ").strip()
            
            if not key_id:
                print("❌ Key ID is required")
                return False
            
            # Get new credentials
            if args.new_api_key and args.new_api_secret:
                new_api_key = args.new_api_key
                new_api_secret = args.new_api_secret
            else:
                print("Enter new API credentials:")
                new_api_key = getpass.getpass("New API Key: ")
                new_api_secret = getpass.getpass("New API Secret: ")
            
            if not new_api_key or not new_api_secret:
                print("❌ New credentials are required")
                return False
            
            # Rotate key
            success = self.key_manager.rotate_key(key_id, new_api_key, new_api_secret)
            
            if success:
                print(f"✅ API key '{key_id}' rotated successfully")
            else:
                print(f"❌ Failed to rotate API key '{key_id}'")
            
            return success
            
        except Exception as e:
            print(f"❌ Error rotating key: {e}")
            return False
    
    def backup_keys(self, args):
        """Create backup of encrypted keys."""
        try:
            backup_path = args.backup_path
            if not backup_path:
                from datetime import datetime
                backup_path = f"backups/dipmaster_keys_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            success = self.key_manager.backup_keys(backup_path)
            
            if success:
                print(f"✅ Keys backed up to: {backup_path}")
            else:
                print("❌ Backup failed")
            
            return success
            
        except Exception as e:
            print(f"❌ Error creating backup: {e}")
            return False
    
    def generate_master_key(self, args):
        """Generate new master key for production use."""
        try:
            from security.crypto_manager import generate_master_key
            
            master_key = generate_master_key()
            
            print("🔑 Generated new master key:")
            print(f"DIPMASTER_MASTER_PASSWORD={master_key}")
            print("\n⚠️  IMPORTANT:")
            print("- Store this key securely (e.g., in a password manager)")
            print("- Set it as an environment variable before using the system")
            print("- Never share this key or commit it to version control")
            
            if args.save_to_file:
                key_file = Path("master_key.txt")
                with open(key_file, 'w') as f:
                    f.write(master_key)
                os.chmod(key_file, 0o600)  # Owner read only
                print(f"📁 Master key saved to: {key_file}")
                print("   Remember to delete this file after securing the key!")
            
            return True
            
        except Exception as e:
            print(f"❌ Error generating master key: {e}")
            return False
    
    def health_check(self, args):
        """Perform health check on key management system."""
        try:
            print("🔍 Performing health check...")
            
            # Check storage file exists and is readable
            storage_path = Path(self.key_manager.storage_path)
            if storage_path.exists():
                print("✅ Storage file exists")
            else:
                print("❌ Storage file not found")
                return False
            
            # Check file permissions
            stat_info = storage_path.stat()
            if oct(stat_info.st_mode)[-3:] == '600':
                print("✅ Storage file has secure permissions")
            else:
                print("⚠️  Storage file permissions may be insecure")
            
            # Test encryption/decryption
            test_data = "test_data_for_health_check"
            try:
                encrypted = self.key_manager.crypto_manager.encrypt_data(test_data)
                decrypted = self.key_manager.crypto_manager.decrypt_data(encrypted)
                if decrypted == test_data:
                    print("✅ Encryption/decryption working")
                else:
                    print("❌ Encryption/decryption failed")
                    return False
            except Exception as e:
                print(f"❌ Encryption test failed: {e}")
                return False
            
            # Get key count
            keys = self.key_manager.list_keys()
            print(f"📊 Total keys stored: {len(keys)}")
            
            print("✅ Health check completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def interactive_mode(self):
        """Run in interactive mode."""
        print("\n" + "="*60)
        print("🔐 DipMaster API Key Management Tool - Interactive Mode")
        print("="*60)
        
        while True:
            try:
                print("\nAvailable commands:")
                print("1. Store new API key")
                print("2. Retrieve API key") 
                print("3. List all keys")
                print("4. Delete API key")
                print("5. Rotate API key")
                print("6. Create backup")
                print("7. Health check")
                print("8. Exit")
                
                choice = input("\nSelect command (1-8): ").strip()
                
                if choice == '1':
                    args = argparse.Namespace(
                        key_id=None, api_key=None, api_secret=None,
                        exchange=None, description=None, environment=None
                    )
                    self.store_key(args)
                
                elif choice == '2':
                    args = argparse.Namespace(key_id=None, show_secret=False)
                    show_secret = input("Show full secret? (y/N): ").lower() == 'y'
                    args.show_secret = show_secret
                    self.retrieve_key(args)
                
                elif choice == '3':
                    args = argparse.Namespace(exchange=None)
                    self.list_keys(args)
                
                elif choice == '4':
                    args = argparse.Namespace(key_id=None, force=False)
                    self.delete_key(args)
                
                elif choice == '5':
                    args = argparse.Namespace(
                        key_id=None, new_api_key=None, new_api_secret=None
                    )
                    self.rotate_key(args)
                
                elif choice == '6':
                    args = argparse.Namespace(backup_path=None)
                    self.backup_keys(args)
                
                elif choice == '7':
                    args = argparse.Namespace()
                    self.health_check(args)
                
                elif choice == '8':
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid choice. Please select 1-8.")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_command(self, args):
        """Run specific command based on arguments."""
        # Initialize key manager
        if not self.initialize_key_manager(args.storage_path):
            return False
        
        # Execute command
        command_map = {
            'store': self.store_key,
            'get': self.retrieve_key,
            'list': self.list_keys,
            'delete': self.delete_key,
            'rotate': self.rotate_key,
            'backup': self.backup_keys,
            'generate-key': self.generate_master_key,
            'health': self.health_check,
            'interactive': lambda args: self.interactive_mode()
        }
        
        command_func = command_map.get(args.command)
        if command_func:
            return command_func(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            return False


def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="DipMaster API Key Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python key_management_tool.py interactive
  
  # Store new API key
  python key_management_tool.py store --key-id binance-main --exchange binance
  
  # Retrieve API key
  python key_management_tool.py get --key-id binance-main
  
  # List all keys
  python key_management_tool.py list
  
  # Create backup
  python key_management_tool.py backup
  
  # Generate master key
  python key_management_tool.py generate-key
        """
    )
    
    parser.add_argument('command', 
                       choices=['store', 'get', 'list', 'delete', 'rotate', 
                               'backup', 'generate-key', 'health', 'interactive'],
                       help='Command to execute')
    
    parser.add_argument('--storage-path', 
                       help='Path to encrypted key storage file')
    
    # Store command arguments
    parser.add_argument('--key-id', 
                       help='API key identifier')
    parser.add_argument('--api-key', 
                       help='API key (use interactive mode for security)')
    parser.add_argument('--api-secret', 
                       help='API secret (use interactive mode for security)')
    parser.add_argument('--exchange', 
                       help='Exchange name (default: binance)')
    parser.add_argument('--description', 
                       help='Key description')
    parser.add_argument('--environment', 
                       help='Environment (production, staging, test)')
    
    # Retrieve command arguments
    parser.add_argument('--show-secret', action='store_true',
                       help='Show full API secret (default: masked)')
    
    # Rotate command arguments
    parser.add_argument('--new-api-key', 
                       help='New API key for rotation')
    parser.add_argument('--new-api-secret', 
                       help='New API secret for rotation')
    
    # Delete command arguments
    parser.add_argument('--force', action='store_true',
                       help='Skip confirmation prompt')
    
    # Backup command arguments
    parser.add_argument('--backup-path', 
                       help='Backup file path')
    
    # Generate key command arguments
    parser.add_argument('--save-to-file', action='store_true',
                       help='Save generated key to file (use with caution)')
    
    return parser


def main():
    """Main entry point."""
    try:
        parser = create_argument_parser()
        args = parser.parse_args()
        
        tool = KeyManagementTool()
        success = tool.run_command(args)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n👋 Operation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()