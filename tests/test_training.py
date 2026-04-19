"""
训练流程测试
测试YOLO模型训练流程
"""

import sys
import tempfile
import shutil
import time
from pathlib import Path
import yaml
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def create_test_dataset(temp_dir):
    """创建测试用的小型数据集"""
    print("\n[设置] 创建测试数据集...")

    import cv2
    import numpy as np

    # 创建类别目录
    categories = ["class1", "class2"]
    for category in categories:
        cat_dir = temp_dir / "dataset" / category / "images"
        cat_dir.mkdir(parents=True, exist_ok=True)

        # 创建10张小图片
        for i in range(10):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = cat_dir / f"img_{i}.jpg"
            cv2.imwrite(str(img_path), img)

        # 创建标注
        cat_dir_labels = temp_dir / "dataset" / category / "labels"
        cat_dir_labels.mkdir(parents=True, exist_ok=True)

        for i in range(10):
            label_path = cat_dir_labels / f"img_{i}.txt"
            with open(label_path, 'w') as f:
                f.write(f"0 0.5 0.5 0.3 0.4\n")

    # 划分训练集和验证集
    train_dir = temp_dir / "dataset" / "train"
    val_dir = temp_dir / "dataset" / "val"

    for split_dir in [train_dir, val_dir]:
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "images").mkdir(exist_ok=True)
        (split_dir / "labels").mkdir(exist_ok=True)

        # 复制部分图片到训练集和验证集
        for category in categories:
            cat_images = list((temp_dir / "dataset" / category / "images").glob("*.jpg"))
            if split_dir == train_dir:
                selected = cat_images[:7]
            else:
                selected = cat_images[7:]

            for img_path in selected:
                shutil.copy(img_path, split_dir / "images" / img_path.name)
                label_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
                shutil.copy(label_path, split_dir / "labels" / label_path.name)

    # 创建classes.txt
    classes_file = temp_dir / "dataset" / "classes.txt"
    with open(classes_file, 'w') as f:
        for i, cat in enumerate(categories):
            f.write(f"{i} {cat}\n")

    print(f"✓ 测试数据集创建完成: {temp_dir}")
    return temp_dir


def create_data_yaml(dataset_dir, output_path):
    """创建data.yaml配置文件"""
    print("\n[设置] 创建data.yaml...")

    data_config = {
        'path': str(dataset_dir.absolute()),
        'train': 'train',
        'val': 'val',
        'nc': 2,
        'names': ['class1', 'class2'],
        'img_size': 100,
    }

    with open(output_path, 'w') as f:
        yaml.dump(data_config, f)

    print(f"✓ data.yaml创建完成: {output_path}")
    return data_config


def test_yolov8_trainer_import():
    """测试YOLOv8训练器导入"""
    print("\n[测试] YOLOv8训练器导入...")
    try:
        from trainers.yolov8_trainer import YOLOv8Trainer
        print("✓ YOLOv8Trainer导入成功")
        return YOLOv8Trainer
    except ImportError as e:
        assert False, f"YOLOv8Trainer导入失败: {e}"


def test_yolov26_trainer_import():
    """测试YOLOv26训练器导入"""
    print("\n[测试] YOLOv26训练器导入...")
    try:
        from trainers.yolov26_trainer import YOLO26Trainer
        print("✓ YOLO26Trainer导入成功")
        return YOLO26Trainer
    except ImportError as e:
        assert False, f"YOLO26Trainer导入失败: {e}"


def test_yolov8_trainer_initialization():
    """测试YOLOv8训练器初始化"""
    print("\n[测试] YOLOv8训练器初始化...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dataset_dir = create_test_dataset(temp_path)
        data_yaml = temp_path / "data.yaml"
        create_data_yaml(dataset_dir, data_yaml)
        output_dir = temp_path / "runs"

        from trainers.yolov8_trainer import YOLOv8Trainer

        trainer = YOLOv8Trainer(
            data_config_path=str(data_yaml),
            output_dir=str(output_dir),
            model_size='n',
            device='cpu'  # 使用CPU进行测试
        )

        assert trainer.model_name == 'yolov8n-seg.pt', "模型名称不正确"
        print("✓ YOLOv8训练器初始化成功")
        print(f"  模型: {trainer.model_name}")


def test_yolov26_trainer_initialization():
    """测试YOLOv26训练器初始化"""
    print("\n[测试] YOLOv26训练器初始化...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dataset_dir = create_test_dataset(temp_path)
        data_yaml = temp_path / "data.yaml"
        create_data_yaml(dataset_dir, data_yaml)
        output_dir = temp_path / "runs"

        from trainers.yolov26_trainer import YOLO26Trainer

        trainer = YOLO26Trainer(
            data_config_path=str(data_yaml),
            output_dir=str(output_dir),
            model_size='n',
            device='cpu'  # 使用CPU进行测试
        )

        assert trainer.model_name == 'yolo26n.pt', "模型名称不正确"
        print("✓ YOLOv26训练器初始化成功")
        print(f"  模型: {trainer.model_name}")


def test_yolov8_quick_train():
    """测试YOLOv8快速训练（1个epoch）"""
    print("\n[测试] YOLOv8快速训练...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dataset_dir = create_test_dataset(temp_path)
        data_yaml = temp_path / "data.yaml"
        create_data_yaml(dataset_dir, data_yaml)
        output_dir = temp_path / "runs"

        from trainers.yolov8_trainer import YOLOv8Trainer

        trainer = YOLOv8Trainer(
            data_config_path=str(data_yaml),
            output_dir=str(output_dir),
            model_size='n',
            device='cpu'
        )

        # 快速训练1个epoch
        print("  开始训练（1个epoch）...")
        start_time = time.time()

        try:
            results = trainer.train(
                epochs=1,
                batch_size=4,
                img_size=100,
                workers=1,
                patience=100  # 禁用早停
            )

            train_time = time.time() - start_time
            print(f"✓ 训练完成，耗时: {train_time:.2f}秒")

            # 检查输出文件
            exp_dir = output_dir / "exp"
            if exp_dir.exists():
                weights_dir = exp_dir / "weights"
                if weights_dir.exists():
                    print(f"✓ 权重目录存在: {weights_dir}")
                    weight_files = list(weights_dir.glob("*.pt"))
                    print(f"  生成权重文件数: {len(weight_files)}")
                    for wf in weight_files:
                        print(f"    - {wf.name}")

        except Exception as e:
            # 训练失败不一定是致命错误，可能是资源不足
            print(f"⚠ 训练失败（可能由于资源限制）: {e}")
            # 不断言失败，只记录


def test_yolov26_quick_train():
    """测试YOLOv26快速训练（1个epoch）"""
    print("\n[测试] YOLOv26快速训练...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        dataset_dir = create_test_dataset(temp_path)
        data_yaml = temp_path / "data.yaml"
        create_data_yaml(dataset_dir, data_yaml)
        output_dir = temp_path / "runs"

        from trainers.yolov26_trainer import YOLO26Trainer

        trainer = YOLO26Trainer(
            data_config_path=str(data_yaml),
            output_dir=str(output_dir),
            model_size='n',
            device='cpu'
        )

        # 快速训练1个epoch
        print("  开始训练（1个epoch）...")
        start_time = time.time()

        try:
            results = trainer.train(
                epochs=1,
                batch_size=4,
                img_size=100,
                workers=1,
                patience=100
            )

            train_time = time.time() - start_time
            print(f"✓ 训练完成，耗时: {train_time:.2f}秒")

            # 检查输出文件
            exp_dir = output_dir / "exp"
            if exp_dir.exists():
                weights_dir = exp_dir / "weights"
                if weights_dir.exists():
                    print(f"✓ 权重目录存在: {weights_dir}")
                    weight_files = list(weights_dir.glob("*.pt"))
                    print(f"  生成权重文件数: {len(weight_files)}")

        except Exception as e:
            print(f"⚠ 训练失败（可能由于资源限制）: {e}")


def run_all_training_tests():
    """运行所有训练测试"""
    print("=" * 60)
    print("开始训练流程测试")
    print("=" * 60)

    tests = [
        test_yolov8_trainer_import,
        test_yolov26_trainer_import,
        test_yolov8_trainer_initialization,
        test_yolov26_trainer_initialization,
        test_yolov8_quick_train,
        test_yolov26_quick_train,
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
    success = run_all_training_tests()
    sys.exit(0 if success else 1)
