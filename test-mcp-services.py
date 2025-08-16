#!/usr/bin/env python3
"""
测试MCP服务连接和功能
Test MCP Services Connectivity and Functionality
"""

import asyncio
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

def test_sqlite_database():
    """测试SQLite数据库连接"""
    print("测试SQLite数据库...")
    
    db_path = Path("data/dipmaster.db")
    db_path.parent.mkdir(exist_ok=True)
    
    try:
        # 创建测试数据库和表
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 创建测试表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_trades (
                id INTEGER PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 插入测试数据
        cursor.execute("""
            INSERT INTO test_trades (symbol, side, price, quantity)
            VALUES (?, ?, ?, ?)
        """, ("BTCUSDT", "BUY", 45000.0, 0.001))
        
        conn.commit()
        
        # 查询测试
        cursor.execute("SELECT COUNT(*) FROM test_trades")
        count = cursor.fetchone()[0]
        
        conn.close()
        
        print(f"SQLite数据库测试成功! 记录数: {count}")
        return True
        
    except Exception as e:
        print(f"SQLite数据库测试失败: {e}")
        return False

def test_npm_packages():
    """测试NPM包安装"""
    print("测试NPM包...")
    
    packages = [
        "@modelcontextprotocol/server-memory",
        "@modelcontextprotocol/server-filesystem", 
        "@modelcontextprotocol/server-sequential-thinking"
    ]
    
    results = {}
    
    for package in packages:
        try:
            result = subprocess.run(
                ["npm", "list", "-g", package],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"{package} 已安装")
                results[package] = True
            else:
                print(f"{package} 未找到")
                results[package] = False
                
        except Exception as e:
            print(f"检查 {package} 时出错: {e}")
            results[package] = False
    
    return all(results.values())

def test_mcp_server_startup():
    """测试MCP服务器启动"""
    print("测试MCP服务器启动...")
    
    tests = []
    
    # 测试内存服务器
    try:
        proc = subprocess.Popen(
            ["npx", "@modelcontextprotocol/server-memory"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 等待启动
        try:
            stdout, stderr = proc.communicate(timeout=3)
            if proc.returncode == 0 or "running on stdio" in stdout:
                print("Memory服务器启动成功")
                tests.append(True)
            else:
                print(f"Memory服务器启动失败: {stderr}")
                tests.append(False)
        except subprocess.TimeoutExpired:
            proc.kill()
            print("Memory服务器启动成功 (超时但正常)")
            tests.append(True)
            
    except Exception as e:
        print(f"Memory服务器测试失败: {e}")
        tests.append(False)
    
    # 测试文件系统服务器
    try:
        proc = subprocess.Popen(
            ["npx", "@modelcontextprotocol/server-filesystem", "data"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        try:
            stdout, stderr = proc.communicate(timeout=3)
            if proc.returncode == 0 or "Filesystem" in stdout:
                print("Filesystem服务器启动成功")
                tests.append(True)
            else:
                print(f"Filesystem服务器启动失败: {stderr}")
                tests.append(False)
        except subprocess.TimeoutExpired:
            proc.kill()
            print("Filesystem服务器启动成功 (超时但正常)")
            tests.append(True)
            
    except Exception as e:
        print(f"Filesystem服务器测试失败: {e}")
        tests.append(False)
    
    return all(tests)

def generate_test_report():
    """生成测试报告"""
    print("\n" + "="*60)
    print("DipMaster Trading System - MCP服务测试报告")
    print("="*60)
    
    # 执行所有测试
    sqlite_ok = test_sqlite_database()
    npm_ok = test_npm_packages()
    server_ok = test_mcp_server_startup()
    
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    print(f"SQLite数据库: {'通过' if sqlite_ok else '失败'}")
    print(f"NPM包安装: {'通过' if npm_ok else '失败'}")
    print(f"MCP服务器: {'通过' if server_ok else '失败'}")
    
    overall_status = sqlite_ok and npm_ok and server_ok
    print(f"\n总体状态: {'所有测试通过' if overall_status else '部分测试失败'}")
    
    if overall_status:
        print("\nMCP服务配置完成! 可以开始使用DipMaster Trading System")
        print("\n使用说明:")
        print("1. 在Claude Code中连接MCP服务")
        print("2. 使用 'sqlite' 服务查询交易数据")
        print("3. 使用 'memory' 服务缓存数据")
        print("4. 使用 'filesystem' 服务管理文件")
        print("5. 使用 'sequential-thinking' 进行策略分析")
    else:
        print("\n建议修复以下问题:")
        if not sqlite_ok:
            print("- 检查SQLite数据库权限和路径")
        if not npm_ok:
            print("- 重新安装缺失的NPM包")
        if not server_ok:
            print("- 检查Node.js环境和MCP服务器配置")
    
    print("\n配置文件: mcp-config.json")
    print("测试脚本: test-mcp-services.py")
    print("="*60)

if __name__ == "__main__":
    generate_test_report()