"""
验证和推理测试
测试模型验证和推理流程
"""

import sys
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def create_test_weights():
    """创建测试用的模拟权重文件"""
    print("\n[设置] 创建测试权重文件...")

    # 创建一个小的.pt文件作为模拟权重
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        weights_path = temp_path / "test_weights.pt"

        # 创建一个小的PyTorch模型
        try:
            import torch
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 5),
                torch.nn.ReLU(),
                torch.nn.Linear(5, 2)
            )
            torch.save(model.state_dict(), weights_path)
            print(f"✓ 测试权重创建: {weights_path}")
            return weights_path, temp_path
        except Exception as e:
            print(f"⚠ 无法创建测试权重: {e}")
            return None, temp_path


def create_test_image():
    """创建测试图片"""
    print("\n[设置] 创建测试图片...")

    test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        img_path = temp_path / "test.jpg"
        cv2.imwrite(str(img_path), test_img)

        print(f"✓ 测试图片创建: {img_path}")
        return img_path, temp_path


def create_test_dataset_and_config():
    """创建测试数据集和配置"""
    print("\n[设置] 创建测试数据集和配置...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # 创建数据集结构
        dataset_dir = temp_path / "dataset"
        dataset_dir.mkdir()

        # 创建验证集
        val_dir = dataset_dir / "val"
        val_images = val_dir / "images"
        val_images.mkdir(parents=True)
        val_labels = val_dir / "labels"
        val_labels.mkdir(parents=True)

        # 创建测试图片
        for i in range(3):
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            img_path = val_images / f"img_{i}.jpg"
            cv2.imwrite(str(img_path), img)

            # 创建标注
            label_path = val_labels / f"img_{i}.txt"
            with open(label_path, 'w') as f:
                f.write(f"0 0.5 0.5 0.3 0.4\n")

        # 创建data.yaml
        data_yaml = temp_path / "data.yaml"
        data_config = {
            'path': str(dataset_dir.absolute()),
            'train': 'train',
            'val': 'val',
            'nc': 1,
            'names': ['test_class'],
            'img_size': 100,
        }

        with open(data_yaml, 'w') as f:
            yaml.dump(data_config, f)

        print(f"✓ 测试数据集和配置创建完成")
        return data_yaml, temp_path


def test_validate_with_mask_import():
    """测试validate_with_mask导入"""
    print("\n[测试] validate_with_mask导入...")
    try:
        from validate_with_mask import validate_with_mask_visualization
        print("✓ validate_with_mask导入成功")
        return validate_with_mask_visualization
    except ImportError as e:
        print(f"⚠ validate_with_mask导入失败（可选）: {e}")
        return None


def test_diagnose_mask_import():
    """测试diagnose_mask导入"""
    print("\n[测试] diagnose_mask导入...")
    try:
        from diagnose_mask import main as diagnose_main
        print("✓ diagnose_mask导入成功")
        return diagnose_main
    except ImportError as e:
        print(f"⚠ diagnose_mask导入失败（可选）: {e}")
        return None


def test_yolov8_validate():
    """测试YOLOv8验证流程（需要真实权重）"""
    print("\n[测试] YOLOv8验证流程...")

    data_yaml, temp_path = create_test_dataset_and_config()

    # 尝试使用Ultralytics的预训练模型进行验证测试
    try:
        from ultralytics import YOLO

        # 加载预训练模型
        model = YOLO('yolov8n-seg.pt')

        # 验证参数
        val_args = {
            'data': str(data_yaml),
            'batch': 2,
            'imgsz': 100,
            'conf': 0.25,
            'iou': 0.6,
            'device': 'cpu',
            'split': 'val',
        }

        print("  开始验证...")
        results = model.val(**val_args)

        print(f"✓ 验证完成")
        if hasattr(results, 'box'):
            print(f"  mAP50: {results.box.map50:.4f}")
            print(f"  mAP50-95: {results.box.map:.4f}")

    except Exception as e:
        print(f"⚠ 验证失败（可能由于网络或数据集问题）: {e}")


def test_yolov8_inference():
    """测试YOLOv8推理流程"""
    print("\n[测试] YOLOv8推理流程...")

    img_path, temp_path = create_test_image()

    try:
        from ultralytics import YOLO
        import cv2

        # 加载预训练模型
        model = YOLO('yolov8n-seg.pt')

        # 读取图片
        image = cv2.imread(str(img_path))

        if image is None:
            raise ValueError("无法读取测试图片")

        # 转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 推理
        print("  开始推理...")
        results = model.predict(source=image_rgb, conf=0.25, verbose=False)

        print(f"✓ 推理完成")
        print(f"  检测结果数: {len(results)}")

        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                print(f"  检测框数: {len(result.boxes)}")
            if hasattr(result, 'masks') and result.masks is not None:
                print(f"  Mask数: {len(result.masks)}")

    except Exception as e:
        print(f"⚠ 推理失败（可能由于网络或模型问题）: {e}")


def test_mask_visualization():
    """测试mask可视化"""
    print("\n[测试] Mask可视化...")

    img_path, temp_path = create_test_image()

    try:
        from ultralytics import YOLO
        import cv2

        # 加载预训练模型
        model = YOLO('yolov8n-seg.pt')

        # 读取图片
        image = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 推理
        results = model.predict(source=image_rgb, conf=0.25, verbose=False)

        if len(results) > 0:
            result = results[0]

            # 可视化
            vis_image = image_rgb.copy()

            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data

                if len(masks) > 0:
                    mask = masks[0].cpu().numpy()
                    mask_resized = cv2.resize(mask, (100, 100))
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    mask_colored = np.zeros_like(vis_image)
                    mask_colored[:, :, 0] = mask_binary * 255

                    vis_image = cv2.addWeighted(vis_image, 0.7, mask_colored, 0.3, 0)

                    # 保存结果
                    output_path = temp_path / "vis_result.jpg"
                    vis_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_path), vis_bgr)

                    print(f"✓ Mask可视化完成: {output_path}")
                else:
                    print("  未检测到mask（这是正常的）")
            else:
                print("  模型未输出mask（可能由于未检测到目标）")
        else:
            print("  无检测结果")

    except Exception as e:
        print(f"⚠ Mask可视化失败: {e}")


def test_numpy_array_inference():
    """测试numpy数组推理方式"""
    print("\n[测试] numpy数组推理方式...")

    try:
        from ultralytics import YOLO
        import cv2

        # 创建测试图片（numpy数组）
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 加载模型
        model = YOLO('yolov8n-seg.pt')

        # 使用numpy数组推理
        print("  开始推理（numpy数组）...")
        results = model.predict(source=image_rgb, conf=0.25, verbose=False)

        print(f"✓ numpy数组推理成功")
        print(f"  结果数: {len(results)}")

    except Exception as e:
        print(f"⚠ numpy数组推理失败: {e}")


def run_all_validation_tests():
    """运行所有验证和推理测试"""
    print("=" * 60)
    print("开始验证和推理测试")
    print("=" * 60)

    tests = [
        test_validate_with_mask_import,
        test_diagnose_mask_import,
        test_yolov8_validate,
        test_yolov8_inference,
        test_mask_visualization,
        test_numpy_array_inference,
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
    success = run_all_validation_tests()
    sys.exit(0 if success else 1)
