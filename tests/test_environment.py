"""
环境配置检查测试
验证训练环境是否正确配置
"""

import sys
import subprocess
from pathlib import Path


def test_python_version():
    """测试Python版本"""
    print("\n[测试] Python版本检查...")
    version = sys.version_info
    assert version.major == 3, f"Python主版本应为3，当前为{version.major}"
    assert version.minor >= 8, f"Python子版本应>= 8，当前为{version.minor}"
    print(f"✓ Python版本: {version.major}.{version.minor}.{version.micro}")


def test_ultralytics_installed():
    """测试Ultralytics是否安装"""
    print("\n[测试] Ultralytics安装检查...")
    try:
        import ultralytics
        print(f"✓ Ultralytics已安装，版本: {ultralytics.__version__}")
    except ImportError:
        assert False, "Ultralytics未安装，请运行: pip install ultralytics"


def test_torch_installed():
    """测试PyTorch是否安装"""
    print("\n[测试] PyTorch安装检查...")
    try:
        import torch
        print(f"✓ PyTorch已安装，版本: {torch.__version__}")

        # 检查CUDA可用性
        if torch.cuda.is_available():
            print(f"✓ CUDA可用: {torch.version.cuda}")
            print(f"✓ GPU数量: {torch.cuda.device_count()}")
            print(f"✓ 当前GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠ CUDA不可用，将使用CPU模式")
    except ImportError:
        assert False, "PyTorch未安装，请运行: pip install torch torchvision"


def test_opencv_installed():
    """测试OpenCV是否安装"""
    print("\n[测试] OpenCV安装检查...")
    try:
        import cv2
        print(f"✓ OpenCV已安装，版本: {cv2.__version__}")
    except ImportError:
        assert False, "OpenCV未安装，请运行: pip install opencv-python"


def test_numpy_installed():
    """测试NumPy是否安装"""
    print("\n[测试] NumPy安装检查...")
    try:
        import numpy as np
        print(f"✓ NumPy已安装，版本: {np.__version__}")
    except ImportError:
        assert False, "NumPy未安装，请运行: pip install numpy"


def test_matplotlib_installed():
    """测试Matplotlib是否安装"""
    print("\n[测试] Matplotlib安装检查...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # 使用非交互式后端
        import matplotlib.pyplot as plt
        print(f"✓ Matplotlib已安装，版本: {matplotlib.__version__}")
    except ImportError:
        assert False, "Matplotlib未安装，请运行: pip install matplotlib"


def test_pillow_installed():
    """测试Pillow是否安装"""
    print("\n[测试] Pillow安装检查...")
    try:
        from PIL import Image
        print(f"✓ Pillow已安装，版本: {Image.__version__}")
    except ImportError:
        assert False, "Pillow未安装，请运行: pip install pillow"


def test_yaml_installed():
    """测试PyYAML是否安装"""
    print("\n[测试] PyYAML安装检查...")
    try:
        import yaml
        print(f"✓ PyYAML已安装")
    except ImportError:
        assert False, "PyYAML未安装，请运行: pip install pyyaml"


def test_scripts_directory_exists():
    """测试scripts目录是否存在"""
    print("\n[测试] scripts目录检查...")
    scripts_dir = Path(__file__).parent.parent / "scripts"
    assert scripts_dir.exists(), f"scripts目录不存在: {scripts_dir}"
    print(f"✓ scripts目录存在: {scripts_dir}")


def test_trainers_directory_exists():
    """测试trainers目录是否存在"""
    print("\n[测试] trainers目录检查...")
    trainers_dir = Path(__file__).parent.parent / "scripts" / "trainers"
    assert trainers_dir.exists(), f"trainers目录不存在: {trainers_dir}"
    print(f"✓ trainers目录存在: {trainers_dir}")


def test_trainers_modules_import():
    """测试训练器模块是否可以导入"""
    print("\n[测试] 训练器模块导入检查...")

    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

    try:
        from trainers.base_trainer import BaseTrainer
        print("✓ base_trainer导入成功")
    except ImportError as e:
        assert False, f"base_trainer导入失败: {e}"

    try:
        from trainers.yolov8_trainer import YOLOv8Trainer
        print("✓ yolov8_trainer导入成功")
    except ImportError as e:
        assert False, f"yolov8_trainer导入失败: {e}"

    try:
        from trainers.yolov26_trainer import YOLO26Trainer
        print("✓ yolov26_trainer导入成功")
    except ImportError as e:
        assert False, f"yolov26_trainer导入失败: {e}"

    try:
        from trainers.yolov6_trainer import YOLOv6Trainer
        print("✓ yolov6_trainer导入成功")
    except ImportError as e:
        # YOLOv6可能未安装，这是可选的
        print(f"⚠ yolov6_trainer导入失败（可选）: {e}")


def test_train_yolo_script_exists():
    """测试train_yolo.py脚本是否存在"""
    print("\n[测试] train_yolo.py脚本检查...")
    train_script = Path(__file__).parent.parent / "scripts" / "train_yolo.py"
    assert train_script.exists(), f"train_yolo.py不存在: {train_script}"
    print(f"✓ train_yolo.py存在: {train_script}")


def test_data_loader_module_exists():
    """测试data_loader模块是否存在"""
    print("\n[测试] data_loader模块检查...")
    data_loader = Path(__file__).parent.parent / "scripts" / "data_loader.py"
    assert data_loader.exists(), f"data_loader.py不存在: {data_loader}"
    print(f"✓ data_loader.py存在: {data_loader}")


def test_config_files_exist():
    """测试配置文件是否存在"""
    print("\n[测试] 配置文件检查...")

    configs_dir = Path(__file__).parent.parent / "train_configs"

    required_configs = [
        "data_template.yaml",
        "yolov8_seg_config.yaml",
        "yolov26_seg_config.yaml",
        "yolov6_seg_config.yaml",
    ]

    for config_file in required_configs:
        config_path = configs_dir / config_file
        if config_path.exists():
            print(f"✓ {config_file}存在")
        else:
            print(f"⚠ {config_file}不存在（可选）")


def test_gpu_memory():
    """测试GPU内存（如果可用）"""
    print("\n[测试] GPU内存检查...")
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory
            total_memory_gb = total_memory / (1024 ** 3)
            print(f"✓ GPU总内存: {total_memory_gb:.2f} GB")

            # 检查至少有8GB内存
            assert total_memory_gb >= 6, f"GPU内存不足，建议至少6GB，当前{total_memory_gb:.2f}GB"
        else:
            print("⚠ 无GPU可用，跳过内存检查")
    except Exception as e:
        print(f"⚠ GPU内存检查失败: {e}")


def test_disk_space():
    """测试磁盘空间"""
    print("\n[测试] 磁盘空间检查...")
    import shutil

    project_dir = Path(__file__).parent.parent
    usage = shutil.disk_usage(project_dir)

    free_gb = usage.free / (1024 ** 3)
    total_gb = usage.total / (1024 ** 3)

    print(f"✓ 磁盘总空间: {total_gb:.2f} GB")
    print(f"✓ 磁盘可用空间: {free_gb:.2f} GB")

    # 至少需要20GB可用空间
    assert free_gb >= 20, f"磁盘空间不足，建议至少20GB，当前{free_gb:.2f}GB"


def run_all_environment_tests():
    """运行所有环境测试"""
    print("=" * 60)
    print("开始环境配置检查")
    print("=" * 60)

    tests = [
        test_python_version,
        test_ultralytics_installed,
        test_torch_installed,
        test_opencv_installed,
        test_numpy_installed,
        test_matplotlib_installed,
        test_pillow_installed,
        test_yaml_installed,
        test_scripts_directory_exists,
        test_trainers_directory_exists,
        test_trainers_modules_import,
        test_train_yolo_script_exists,
        test_data_loader_module_exists,
        test_config_files_exist,
        test_gpu_memory,
        test_disk_space,
    ]

    failed_tests = []

    for test_func in tests:
        try:
            test_func()
        except AssertionError as e:
            print(f"✗ {test_func.__name__} 失败: {e}")
            failed_tests.append((test_func.__name__, str(e)))
        except Exception as e:
            print(f"✗ {test_func.__name__} 出错: {e}")
            failed_tests.append((test_func.__name__, str(e)))

    print("\n" + "=" * 60)
    if failed_tests:
        print(f"测试完成，失败: {len(failed_tests)}/{len(tests)}")
        print("\n失败的测试:")
        for test_name, error in failed_tests:
            print(f"  - {test_name}: {error}")
        print("=" * 60)
        return False
    else:
        print(f"测试完成，全部通过: {len(tests)}/{len(tests)}")
        print("=" * 60)
        return True


if __name__ == "__main__":
    success = run_all_environment_tests()
    sys.exit(0 if success else 1)
