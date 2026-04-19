"""
简化的mask可视化测试脚本
用于快速验证模型能否输出mask
"""

import sys
import cv2
import numpy as np
from pathlib import Path

from ultralytics import YOLO
import yaml

def main():
    weights_path = sys.argv[1] if len(sys.argv) > 1 else "path/to/weights.pt"
    data_config_path = sys.argv[2] if len(sys.argv) > 2 else "path/to/data.yaml"

    print("=" * 60)
    print("简化Mask可视化测试")
    print("=" * 60)

    # 加载模型
    print(f"\n1. 加载模型: {weights_path}")
    model = YOLO(weights_path)
    print(f"   模型任务: {model.task}")

    # 加载数据配置
    print(f"\n2. 加载数据配置: {data_config_path}")
    with open(data_config_path, 'r') as f:
        data_config = yaml.safe_load(f)

    # 获取验证集图片
    dataset_root = Path(data_config['path'])
    val_images_dir = dataset_root / data_config['val'] / 'images'
    image_files = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))

    if len(image_files) == 0:
        print(f"   错误: 未找到验证图片")
        return

    print(f"   验证集图片数量: {len(image_files)}")

    # 选择第一张图片进行测试
    test_image = image_files[0]
    print(f"\n3. 测试图片: {test_image.name}")

    # 读取图片
    image = cv2.imread(str(test_image))
    if image is None:
        print(f"   错误: 无法读取图片")
        return

    print(f"   图片尺寸: {image.shape}")

    # 运行推理 - 直接传递numpy数组
    print(f"\n4. 运行推理...")
    try:
        # 转换为RGB格式（Ultralytics期望RGB）
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 直接传递numpy数组
        results = model.predict(
            source=image_rgb,
            conf=0.25,
            verbose=True
        )

        if not results or len(results) == 0:
            print("   未返回结果")
            return

        result = results[0]
        print(f"   推理成功")

        # 检查结果
        print(f"\n5. 检查结果:")
        print(f"   结果类型: {type(result)}")
        print(f"   结果属性: {[attr for attr in dir(result) if not attr.startswith('_')]}")

        # 检查boxes
        if hasattr(result, 'boxes') and result.boxes is not None:
            print(f"\n   检测框:")
            print(f"     数量: {len(result.boxes)}")
            if len(result.boxes) > 0:
                print(f"     第一个框的置信度: {result.boxes.conf[0]:.4f}")
                print(f"     第一个框的类别: {int(result.boxes.cls[0])}")

        # 检查masks
        if hasattr(result, 'masks'):
            print(f"\n   Masks:")
            if result.masks is None:
                print(f"     状态: None（这是问题所在！）")
            else:
                print(f"     状态: 存在")
                print(f"     数量: {len(result.masks)}")
                print(f"     数据形状: {result.masks.data.shape}")
                print(f"     数据类型: {result.masks.data.dtype}")

                # 可视化第一个mask
                if len(result.masks) > 0:
                    print(f"\n6. 可视化第一个mask...")
                    mask = result.masks.data[0].cpu().numpy()

                    # 创建可视化
                    h, w = image.shape[:2]
                    mask_resized = cv2.resize(mask, (w, h))

                    # 创建彩色mask
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    mask_colored = np.zeros_like(image)
                    mask_colored[:, :, 0] = mask_binary * 255  # 红色

                    # 叠加到原图
                    vis_image = cv2.addWeighted(image, 0.7, mask_colored, 0.3, 0)

                    # 绘制边界框
                    if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                        box = result.boxes.xyxy[0].cpu().numpy().astype(int)
                        cv2.rectangle(vis_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

                    # 保存结果
                    output_path = '/tmp/mask_test_result.jpg'
                    cv2.imwrite(output_path, vis_image)
                    print(f"   保存结果: {output_path}")
                    print(f"\n   ✓ Mask可视化成功！请查看 /tmp/mask_test_result.jpg")
        else:
            print(f"\n   Masks: 结果中没有masks属性")

    except Exception as e:
        print(f"\n   推理失败: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
