"""
数据加载和处理测试
测试数据集加载、转换和验证流程
"""

import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def test_data_loader_import():
    """测试数据加载器导入"""
    print("\n[测试] 数据加载器导入...")
    try:
        from data_loader import DatasetLoader
        print("✓ DatasetLoader导入成功")
        return DatasetLoader
    except ImportError as e:
        assert False, f"数据加载器导入失败: {e}"


def create_mock_dataset(temp_dir):
    """创建模拟数据集"""
    print("\n[设置] 创建模拟数据集...")

    # 创建类别目录
    categories = ["category1", "category2"]
    for category in categories:
        cat_dir = temp_dir / category / "images"
        cat_dir.mkdir(parents=True, exist_ok=True)

        # 创建图片
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = cat_dir / f"img_{i}.jpg"
            cv2.imwrite(str(img_path), img)

        # 创建JSON标注
        cat_dir_labels = temp_dir / category / "labels"
        cat_dir_labels.mkdir(parents=True, exist_ok=True)

        for i in range(3):
            label_path = cat_dir_labels / f"img_{i}.txt"
            # 创建YOLO格式标注: class_id x_center y_center width height
            with open(label_path, 'w') as f:
                f.write(f"0 0.5 0.5 0.3 0.4\n")

    # 创建classes.txt
    classes_file = temp_dir / "classes.txt"
    with open(classes_file, 'w') as f:
        for i, cat in enumerate(categories):
            f.write(f"{i} {cat}\n")

    print(f"✓ 模拟数据集创建完成: {temp_dir}")
    return temp_dir


def test_load_dataset():
    """测试加载数据集"""
    print("\n[测试] 加载数据集...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        create_mock_dataset(temp_path)

        from data_loader import DatasetLoader
        loader = DatasetLoader(str(temp_path))

        dataset = loader.load_dataset()

        assert len(dataset) > 0, "数据集为空"
        assert "category1" in dataset, "缺少category1"
        assert "category2" in dataset, "缺少category2"

        print(f"✓ 数据集加载成功，类别数: {len(dataset)}")

        for category, samples in dataset.items():
            print(f"  - {category}: {len(samples)} 个样本")


def test_get_category_stats():
    """测试获取类别统计"""
    print("\n[测试] 获取类别统计...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        create_mock_dataset(temp_path)

        from data_loader import DatasetLoader
        loader = DatasetLoader(str(temp_path))

        stats = loader.get_category_stats()

        assert len(stats) > 0, "统计结果为空"
        assert "category1" in stats, "缺少category1统计"
        assert stats["category1"] > 0, "category1样本数为0"

        print("✓ 类别统计获取成功")
        for category, count in stats.items():
            print(f"  - {category}: {count}")


def test_statistics_module():
    """测试统计模块"""
    print("\n[测试] 统计模块...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        create_mock_dataset(temp_path)

        from data_loader import DatasetLoader
        from statistics import DatasetStats

        loader = DatasetLoader(str(temp_path))
        dataset = loader.load_dataset()

        stats_module = DatasetStats(dataset)
        report = stats_module.calculate_stats()

        assert report is not None, "统计报告为空"
        assert "total_samples" in report, "缺少total_samples"

        print("✓ 统计模块运行成功")
        print(f"  - 总样本数: {report['total_samples']}")


def test_yaml_data_config():
    """测试YAML数据配置生成"""
    print("\n[测试] YAML数据配置生成...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dataset_dir = temp_path / "dataset"
        create_mock_dataset(dataset_dir)

        # 创建data.yaml
        data_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'train',
            'val': 'val',
            'nc': 2,
            'names': ['category1', 'category2'],
            'img_size': 640,
            'epochs': 10,
            'batch_size': 16,
        }

        config_path = temp_path / "data.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(data_config, f)

        print("✓ YAML数据配置创建成功")
        print(f"  配置文件: {config_path}")

        # 验证配置
        with open(config_path, 'r') as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config['nc'] == 2, "类别数不正确"
        assert len(loaded_config['names']) == 2, "类别名称数不正确"

        print("✓ YAML数据配置验证通过")


def test_image_loading():
    """测试图片加载"""
    print("\n[测试] 图片加载...")

    # 创建测试图片
    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as temp_dir:
        img_path = Path(temp_dir) / "test.jpg"
        cv2.imwrite(str(img_path), test_img)

        # 读取图片
        loaded_img = cv2.imread(str(img_path))

        assert loaded_img is not None, "图片加载失败"
        assert loaded_img.shape == test_img.shape, "图片形状不匹配"

        print("✓ 图片加载成功")
        print(f"  图片形状: {loaded_img.shape}")


def test_label_format():
    """测试标注格式"""
    print("\n[测试] 标注格式...")

    # 创建YOLO格式标注
    label_content = "0 0.5 0.5 0.3 0.4\n1 0.7 0.8 0.2 0.2\n"

    with tempfile.TemporaryDirectory() as temp_dir:
        label_path = Path(temp_dir) / "label.txt"
        with open(label_path, 'w') as f:
            f.write(label_content)

        # 读取标注
        with open(label_path, 'r') as f:
            lines = f.readlines()

        assert len(lines) == 2, "标注数量不正确"

        # 解析第一行
        parts = lines[0].strip().split()
        class_id = int(parts[0])
        bbox = [float(x) for x in parts[1:]]

        assert len(bbox) == 4, "边界框坐标数不正确"
        assert 0 <= bbox[0] <= 1, "x_center超出范围"
        assert 0 <= bbox[1] <= 1, "y_center超出范围"

        print("✓ 标注格式验证通过")
        print(f"  类别ID: {class_id}, 边界框: {bbox}")


def test_yolo_converter_import():
    """测试YOLO转换器导入"""
    print("\n[测试] YOLO转换器导入...")
    try:
        from yolo_converter import YOLOConverter
        print("✓ YOLOConverter导入成功")
        return YOLOConverter
    except ImportError as e:
        # 这是可选的，如果导入失败只警告
        print(f"⚠ YOLOConverter导入失败（可选）: {e}")
        return None


def test_yolo_validator_import():
    """测试YOLO验证器导入"""
    print("\n[测试] YOLO验证器导入...")
    try:
        from yolo_validator import YOLOValidator
        print("✓ YOLOValidator导入成功")
        return YOLOValidator
    except ImportError as e:
        # 这是可选的，如果导入失败只警告
        print(f"⚠ YOLOValidator导入失败（可选）: {e}")
        return None


def run_all_data_tests():
    """运行所有数据测试"""
    print("=" * 60)
    print("开始数据处理测试")
    print("=" * 60)

    tests = [
        test_data_loader_import,
        test_load_dataset,
        test_get_category_stats,
        test_statistics_module,
        test_yaml_data_config,
        test_image_loading,
        test_label_format,
        test_yolo_converter_import,
        test_yolo_validator_import,
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
    success = run_all_data_tests()
    sys.exit(0 if success else 1)
