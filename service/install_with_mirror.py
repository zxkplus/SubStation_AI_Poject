"""
SubStation AI 服务端安装和启动脚本（使用国内镜像源）

解决 ProxyError 问题，使用清华镜像源加速安装

使用方法:
    python service/install_with_mirror.py
"""

import sys
import subprocess
import importlib


# 国内镜像源列表
MIRRORS = [
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple/",
    "https://pypi.mirrors.ustc.edu.cn/simple/",
    "https://mirror.baidu.com/pypi/simple/",
]


def check_and_install_package(package_name, import_name=None):
    """检查并安装包，使用国内镜像源"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"⚠️  {package_name} 未安装，正在使用国内镜像源安装...")
        
        # 尝试多个镜像源
        for mirror in MIRRORS:
            print(f"   尝试镜像源: {mirror}")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    package_name,
                    "-i", mirror,
                    "--trusted-host", mirror.split("//")[1].split("/")[0]
                ], timeout=60)
                print(f"✅ {package_name} 安装成功")
                return True
            except subprocess.TimeoutExpired:
                print(f"   ⏱️  超时，尝试下一个镜像源...")
                continue
            except subprocess.CalledProcessError as e:
                print(f"   ❌ 安装失败，尝试下一个镜像源...")
                continue
        
        print(f"❌ {package_name} 所有镜像源都失败")
        return False


def main():
    print("=" * 60)
    print("  SubStation AI 服务端安装检查（国内镜像源）")
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
        print("\n" + "=" * 60)
        print(" 部分包安装失败")
        print("=" * 60)
        print("\n请尝试以下方法:")
        print("1. 检查网络连接")
        print("2. 暂时关闭代理")
        print("3. 手动运行: python -m pip install uvicorn fastapi pydantic")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ 所有依赖已安装完成!")
    print("=" * 60)
    print()
    
    # 启动服务
    print("🚀 现在启动推理服务...")
    print("服务将在 http://localhost:8000 启动")
    print("API文档: http://localhost:8000/docs")
    print("按 Ctrl+C 停止服务")
    print()
    
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


if __name__ == "__main__":
    main()
