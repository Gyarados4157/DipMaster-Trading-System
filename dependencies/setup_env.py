#!/usr/bin/env python3
"""
DipMaster Trading System - Environment Setup Script
统一虚拟环境创建和依赖安装工具

Author: DipMaster Development Team
Date: 2025-08-16
Version: 4.0.0
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import platform
import venv
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentSetup:
    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent
        self.dependencies_dir = self.project_root / "dependencies"
        self.venv_dir = self.project_root / "venv"
        
    def check_python_version(self):
        """检查Python版本是否符合要求"""
        version = sys.version_info
        if version < (3, 11):
            raise RuntimeError(f"Python 3.11+ required, got {version.major}.{version.minor}")
        if version >= (3, 13):
            logger.warning(f"Python {version.major}.{version.minor} not fully tested, recommended: 3.11-3.12")
        logger.info(f"Python version {version.major}.{version.minor}.{version.micro} OK")
        
    def create_virtual_environment(self, force: bool = False):
        """创建虚拟环境"""
        if self.venv_dir.exists() and not force:
            logger.info(f"Virtual environment already exists at {self.venv_dir}")
            return
            
        if force and self.venv_dir.exists():
            logger.info("Removing existing virtual environment...")
            import shutil
            shutil.rmtree(self.venv_dir)
            
        logger.info(f"Creating virtual environment at {self.venv_dir}")
        venv.create(self.venv_dir, with_pip=True)
        
    def get_pip_command(self):
        """获取pip命令路径"""
        if platform.system() == "Windows":
            return self.venv_dir / "Scripts" / "pip.exe"
        else:
            return self.venv_dir / "bin" / "pip"
            
    def get_python_command(self):
        """获取Python命令路径"""
        if platform.system() == "Windows":
            return self.venv_dir / "Scripts" / "python.exe"
        else:
            return self.venv_dir / "bin" / "python"
            
    def upgrade_pip(self):
        """升级pip到最新版本"""
        pip_cmd = self.get_pip_command()
        logger.info("Upgrading pip...")
        subprocess.run([str(pip_cmd), "install", "--upgrade", "pip"], check=True)
        
    def install_dependencies(self, env_type: str = "dev"):
        """安装指定环境的依赖"""
        pip_cmd = self.get_pip_command()
        constraints_file = self.dependencies_dir / "constraints.txt"
        
        env_files = {
            "core": [self.dependencies_dir / "requirements.txt"],
            "ml": [self.dependencies_dir / "requirements-ml.txt"],
            "dev": [self.dependencies_dir / "requirements-dev.txt"],
            "prod": [self.dependencies_dir / "requirements-prod.txt"]
        }
        
        if env_type not in env_files:
            raise ValueError(f"Unknown environment type: {env_type}")
            
        logger.info(f"Installing {env_type} dependencies...")
        
        for req_file in env_files[env_type]:
            if not req_file.exists():
                logger.error(f"Requirements file not found: {req_file}")
                continue
                
            cmd = [
                str(pip_cmd), "install", 
                "-r", str(req_file),
                "-c", str(constraints_file)
            ]
            
            logger.info(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            
    def verify_installation(self):
        """验证安装是否成功"""
        python_cmd = self.get_python_command()
        
        critical_packages = [
            "numpy", "pandas", "fastapi", "pydantic", 
            "ta", "python-binance", "websockets"
        ]
        
        logger.info("Verifying critical package installation...")
        for package in critical_packages:
            try:
                subprocess.run([
                    str(python_cmd), "-c", f"import {package}; print(f'{package}: OK')"
                ], check=True, capture_output=True)
                logger.info(f"✓ {package}")
            except subprocess.CalledProcessError:
                logger.error(f"✗ {package} - FAILED")
                
    def generate_activation_script(self):
        """生成环境激活脚本"""
        if platform.system() == "Windows":
            script_content = f"""@echo off
REM DipMaster Trading System - Environment Activation
echo Activating DipMaster Trading System environment...
call "{self.venv_dir}\\Scripts\\activate.bat"
echo Environment activated. Python: 
"{self.venv_dir}\\Scripts\\python.exe" --version
echo.
echo Available commands:
echo   python main.py --help          - Run trading system
echo   python run_complete_system_test.py - Run system tests  
echo   jupyter lab                    - Start Jupyter Lab
echo.
"""
            script_file = self.project_root / "activate_env.bat"
        else:
            script_content = f"""#!/bin/bash
# DipMaster Trading System - Environment Activation
echo "Activating DipMaster Trading System environment..."
source "{self.venv_dir}/bin/activate"
echo "Environment activated. Python: $(python --version)"
echo ""
echo "Available commands:"
echo "  python main.py --help          - Run trading system"
echo "  python run_complete_system_test.py - Run system tests"
echo "  jupyter lab                    - Start Jupyter Lab"
echo ""
"""
            script_file = self.project_root / "activate_env.sh"
            
        with open(script_file, 'w') as f:
            f.write(script_content)
            
        if not platform.system() == "Windows":
            os.chmod(script_file, 0o755)
            
        logger.info(f"Activation script created: {script_file}")
        
    def setup_complete_environment(self, env_type: str = "dev", force: bool = False):
        """完整环境设置流程"""
        try:
            logger.info("Starting DipMaster Trading System environment setup...")
            
            # 检查Python版本
            self.check_python_version()
            
            # 创建虚拟环境
            self.create_virtual_environment(force=force)
            
            # 升级pip
            self.upgrade_pip()
            
            # 安装依赖
            self.install_dependencies(env_type)
            
            # 验证安装
            self.verify_installation()
            
            # 生成激活脚本
            self.generate_activation_script()
            
            logger.info("✅ Environment setup completed successfully!")
            logger.info(f"To activate: {'activate_env.bat' if platform.system() == 'Windows' else 'source activate_env.sh'}")
            
        except Exception as e:
            logger.error(f"❌ Environment setup failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="DipMaster Trading System Environment Setup")
    parser.add_argument(
        "--env-type", 
        choices=["core", "ml", "dev", "prod"], 
        default="dev",
        help="Environment type to setup (default: dev)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Force recreation of virtual environment"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        help="Project root directory (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    setup = EnvironmentSetup(project_root=args.project_root)
    setup.setup_complete_environment(env_type=args.env_type, force=args.force)

if __name__ == "__main__":
    main()