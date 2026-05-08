#!/usr/bin/env python
"""
SubStation AI 客户端快速启动脚本

用法:
    python launch_client.py

或者直接运行:
    python client_app.py
"""

import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """检查依赖是否已安装"""
    required_packages = ['PIL', 'requests', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 缺少以下依赖:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print("\n正在安装依赖...")
        
        try:
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install", 
                "-r", 
                str(Path(__file__).parent / "client_requirements.txt")
            ])
            print("✅ 依赖安装成功!")
        except subprocess.CalledProcessError as e:
            print(f"❌ 依赖安装失败: {e}")
            print("\n请手动运行: pip install -r service/client_requirements.txt")
            sys.exit(1)


def check_service_running(base_url="http://localhost:8000"):
    """检查推理服务是否正在运行"""
    import requests
    
    try:
        response = requests.get(f"{base_url}/docs", timeout=2)
        if response.status_code == 200:
            return True
    except requests.exceptions.ConnectionError:
        pass
    
    return False


def main():
    """主函数"""
    print("=" * 60)
    print("  SubStation AI 实例分割客户端")
    print("=" * 60)
    print()
    
    # 检查依赖
    print("📦 检查依赖...")
    check_dependencies()
    
    # 检查服务状态
    print("🔍 检查推理服务状态...")
    if check_service_running():
        print("✅ 推理服务正在运行")
    else:
        print("⚠️  推理服务未运行")
        print("   请先启动服务: python -m uvicorn service.app:app --host 0.0.0.0 --port 8000")
        print()
        
        choice = input("是否继续启动客户端? (y/n): ")
        if choice.lower() != 'y':
            print("已取消启动")
            sys.exit(0)
    
    print()
    print("🚀 启动客户端...")
    print("=" * 60)
    print()
    
    # 导入并启动应用
    try:
        from client_app import main as app_main
        app_main()
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
