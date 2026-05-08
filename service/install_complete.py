"""
SubStation AI 服务端完整安装脚本

安装所有必需的依赖，包括 ultralytics（YOLO）

使用方法:
    python service/install_complete.py
"""

import sys
import subprocess
import importlib


# 国内镜像源
MIRRORS = [
    "https://pypi.tuna.tsinghua.edu.cn/simple",
    "https://mirrors.aliyun.com/pypi/simple/",
]


def install_package(package_name, import_name=None):
    """使用国内镜像源安装包"""
    if import_name is None:
        import_name = package_name
    
    # 先检查是否已安装
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} 已安装")
        return True
    except ImportError:
        print(f"⚠️  安装 {package_name}...")
        
        for mirror in MIRRORS:
            try:
                print(f"   镜像源: {mirror}")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    package_name,
                    "-i", mirror,
                    "--trusted-host", mirror.split("//")[1].split("/")[0],
                    "--timeout", "120"
                ], timeout=180)
                print(f"✅ {package_name} 安装成功")
                return True
            except subprocess.TimeoutExpired:
                print(f"   ⏱️  超时")
                continue
            except subprocess.CalledProcessError:
                print(f"   ❌ 失败，尝试下一个")
                continue
        
        print(f"❌ {package_name} 安装失败")
        return False


def main():
    print("=" * 60)
    print("  SubStation AI 服务端完整安装")
    print("=" * 60)
    print()
    
    # 安装顺序：先基础依赖，再核心库
    packages = [
        ("uvicorn", "uvicorn"),
        ("fastapi", "fastapi"),
        ("pydantic", "pydantic"),
        ("numpy", "numpy"),
        ("Pillow", "PIL"),
        ("opencv-python", "cv2"),
        ("ultralytics", "ultralytics"),  # 核心：YOLO
        ("requests", "requests"),
    ]
    
    print("开始安装依赖包...\n")
    
    failed = []
    for package_name, import_name in packages:
        if not install_package(package_name, import_name):
            failed.append(package_name)
        print()
    
    if failed:
        print("\n" + "=" * 60)
        print("⚠️  以下包安装失败:")
        for pkg in failed:
            print(f"   - {pkg}")
        print("=" * 60)
        print("\n建议:")
        print("1. 检查网络连接")
        print("2. 关闭代理 VPN")
        print("3. 手动安装失败的包")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ 所有依赖安装完成！")
    print("=" * 60)
    print()
    print("现在可以启动服务:")
    print()
    print("  python -m uvicorn service.app:app --port 8000")
    print()
    print("或运行客户端:")
    print("  cd service")
    print("  python launch_client.py")
    print()


if __name__ == "__main__":
    main()
