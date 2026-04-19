"""
运行所有测试的主入口
"""

import sys
import subprocess
from pathlib import Path


def run_test_file(test_file_path):
    """运行单个测试文件"""
    print(f"\n{'=' * 60}")
    print(f"运行测试: {test_file_path.name}")
    print(f"{'=' * 60}")

    result = subprocess.run(
        [sys.executable, str(test_file_path)],
        capture_output=False
    )

    return result.returncode == 0


def run_pytest():
    """使用pytest运行所有测试"""
    print(f"\n{'=' * 60}")
    print("使用Pytest运行所有测试")
    print(f"{'=' * 60}")

    # 检查pytest是否安装
    try:
        import pytest
    except ImportError:
        print("Pytest未安装，正在安装...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "pytest"],
            check=True
        )
        import pytest

    # 运行pytest
    tests_dir = Path(__file__).parent
    result = pytest.main([
        str(tests_dir),
        "-v",  # 详细输出
        "--tb=short",  # 简短的traceback
    ])

    return result == 0


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("变电站设备分割训练系统 - 测试套件")
    print("=" * 60)

    # 获取tests目录
    tests_dir = Path(__file__).parent

    if not tests_dir.exists():
        print(f"错误: tests目录不存在: {tests_dir}")
        return 1

    # 查找所有测试文件
    test_files = list(tests_dir.glob("test_*.py"))

    if len(test_files) == 0:
        print("错误: 未找到测试文件")
        return 1

    print(f"\n找到测试文件: {len(test_files)}")
    for test_file in test_files:
        print(f"  - {test_file.name}")

    # 询问用户使用哪种方式运行
    print("\n请选择运行方式:")
    print("1. 使用Pytest运行所有测试（推荐）")
    print("2. 逐个运行测试文件")
    print("3. 运行特定测试文件")

    try:
        choice = input("\n请输入选项 (1/2/3): ").strip()

        if choice == "1":
            # 使用pytest
            success = run_pytest()
        elif choice == "2":
            # 逐个运行
            success_count = 0
            for test_file in test_files:
                if run_test_file(test_file):
                    success_count += 1

            print("\n" + "=" * 60)
            print(f"测试完成: {success_count}/{len(test_files)} 通过")
            print("=" * 60)

            success = success_count == len(test_files)
        elif choice == "3":
            # 选择特定文件
            print("\n可用测试文件:")
            for i, test_file in enumerate(test_files, 1):
                print(f"  {i}. {test_file.name}")

            file_choice = input("\n请输入文件编号: ").strip()
            try:
                file_idx = int(file_choice) - 1
                if 0 <= file_idx < len(test_files):
                    success = run_test_file(test_files[file_idx])
                else:
                    print("错误: 无效的文件编号")
                    success = False
            except ValueError:
                print("错误: 请输入有效的数字")
                success = False
        else:
            print("错误: 无效的选项")
            success = False

    except KeyboardInterrupt:
        print("\n\n测试被用户中断")
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
