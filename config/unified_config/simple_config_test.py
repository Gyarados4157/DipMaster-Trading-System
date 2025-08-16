#!/usr/bin/env python3
"""
Simple Configuration Test - No Unicode Characters
"""

import sys
import os
from pathlib import Path

# Add the config directory to path
sys.path.append(str(Path(__file__).parent))

def test_config_system():
    """Test the unified configuration system"""
    try:
        print("Testing unified configuration system...")
        
        # Test config loader
        from config_loader import load_config, create_config_loader
        
        print("1. Loading configuration...")
        config = load_config("dipmaster_v4")
        
        print("2. Configuration loaded successfully!")
        metadata = config.get('_metadata', {})
        print(f"   Environment: {metadata.get('environment', 'unknown')}")
        print(f"   Strategy: {metadata.get('name', 'unknown')} v{metadata.get('version', 'unknown')}")
        print(f"   Source files: {len(metadata.get('source_files', []))}")
        
        # Test key configuration values
        print("\n3. Testing configuration access...")
        loader = create_config_loader()
        
        api_port = loader.get_config_value(config, "api.fastapi.port", 8000)
        log_level = loader.get_config_value(config, "logging.level", "INFO")
        strategy_name = loader.get_config_value(config, "strategy.name", "Unknown")
        
        print(f"   API Port: {api_port}")
        print(f"   Log Level: {log_level}")
        print(f"   Strategy Name: {strategy_name}")
        
        # Test environment variables
        print("\n4. Testing environment variable substitution...")
        os.environ['TEST_VALUE'] = 'test_success'
        test_config = {"test": "${TEST_VALUE:default}"}
        resolved = loader._substitute_env_vars(test_config)
        print(f"   Environment variable test: {resolved.get('test', 'failed')}")
        
        print("\n5. Configuration system test PASSED!")
        return True
        
    except Exception as e:
        print(f"\nConfiguration system test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("\nTesting dependencies...")
    
    critical_modules = [
        'yaml', 'pathlib', 'logging', 'os', 'sys', 're', 'json'
    ]
    
    failed_imports = []
    
    for module in critical_modules:
        try:
            __import__(module)
            print(f"   {module}: OK")
        except ImportError as e:
            print(f"   {module}: FAILED - {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nMissing dependencies: {failed_imports}")
        return False
    else:
        print("\nAll dependencies available!")
        return True

def main():
    print("="*60)
    print("DipMaster Configuration System Test")
    print("="*60)
    
    # Test dependencies first
    deps_ok = test_dependencies()
    if not deps_ok:
        print("\nDependency test failed!")
        return 1
    
    # Test configuration system
    config_ok = test_config_system()
    if not config_ok:
        print("\nConfiguration test failed!")
        return 1
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED - Configuration system ready!")
    print("="*60)
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)