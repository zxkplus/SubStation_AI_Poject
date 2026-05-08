"""
SubStation AI 服务端安装和启动脚本

使用方法:
    python service/install_and_start.py
    
或直接手动执行:
    1. python -m pip install uvicorn[standard] fastapi pydantic
    2. python -m uvicorn service.app:app --port 8000
"""

import sys
import subprocess
import importlib


def check_and_install_package(package_name, import_name=None):
    """检查并安装包"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"️  {package_name} 未安装，正在安装...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name
            ])
            print(f"✅ {package_name} 安装成功")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {package_name} 安装失败: {e}")
            return False


def main():
    print("=" * 60)
    print("  SubStation AI 服务端安装检查")
    print("=" * 60)
    print()
    
    # 检查必需的包
    packages = [
        ("uvicorn", "uvicorn"),
        ("fastapi", "fastapi"),
        ("pydantic", "pydantic"),
    ]
    
    all_installed = True
    for package_name, import_name in packages:
        if not check_and_install_package(package_name, import_name):
            all_installed = False
    
    if not all_installed:
        print("\n❌ 部分包安装失败，请手动运行:")
        print("   python -m pip install uvicorn[standard] fastapi pydantic")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ 所有依赖已安装完成!")
    print("=" * 60)
    print()
    print("现在可以启动服务:")
    print()
    print("  python -m uvicorn service.app:app --host 0.0.0.0 --port 8000")
    print()
    print("或者双击运行:")
    print("  service/start_server.bat")
    print()
    print("启动后访问:")
    print("  📄 API文档: http://localhost:8000/docs")
    print("  📖 ReDoc:   http://localhost:8000/redoc")
    print()
    
    # 询问是否立即启动
    choice = input("是否立即启动服务? (y/n): ")
    if choice.lower() == 'y':
        print("\n🚀 启动服务...")
        print("按 Ctrl+C 停止服务\n")
        
        try:
            subprocess.run([
                sys.executable, "-m", "uvicorn",
                "service.app:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ])
        except KeyboardInterrupt:
            print("\n\n✅ 服务已停止")
        except Exception as e:
            print(f"\n❌ 启动失败: {e}")
            sys.exit(1)
    else:
        print("\n👋 稍后手动启动服务即可")


if __name__ == "__main__":
    main()
