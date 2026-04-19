"""
自定义YOLO验证脚本，支持mask可视化（修复版）
用于检查训练模型的mask预测效果
修复了proto参数传递错误的问题
"""

import argparse
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 添加scripts目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from ultralytics import YOLO
import yaml

# 设置matplotlib后端
import matplotlib
matplotlib.use('Agg')


def load_data_config(data_config_path):
    """加载data.yaml配置"""
    with open(data_config_path, 'r') as f:
        return yaml.safe_load(f)


def visualize_predictions(
    model,
    image_path,
    class_names,
    conf_threshold=0.25,
    output_path=None,
    show_bbox=True,
    show_mask=True,
    mask_alpha=0.5
):
    """
    可视化模型的预测结果（包含mask）- 修复版

    Args:
        model: YOLO模型
        image_path: 图片路径
        class_names: 类别名称列表
        conf_threshold: 置信度阈值
        output_path: 输出图片路径
        show_bbox: 是否显示边界框
        show_mask: 是否显示mask
        mask_alpha: mask透明度
    """
    # 读取图片（使用cv2，格式为BGR）
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    # 转换为RGB格式（Ultralytics期望RGB）
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 运行推理 - 直接传递numpy数组
    try:
        # 注意：OpenCV读取的是BGR格式，需要转换为RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 方法1: 直接传递numpy数组（推荐，避免路径问题）
        results = model.predict(
            source=image_rgb,
            conf=conf_threshold,
            verbose=False,
            device=model.device if hasattr(model, 'device') else 'cpu',
            imgsz=640
        )

        # 方法2: 如果方法1失败，尝试使用路径
        if not results or len(results) == 0:
            print(f"方法1失败，尝试方法2（使用路径）...")
            results = model.predict(
                source=str(image_path),
                conf=conf_threshold,
                verbose=False
            )

    except Exception as e:
        print(f"推理失败: {e}")
        import traceback
        traceback.print_exc()
        return

    if not results or len(results) == 0:
        print(f"未检测到任何目标: {image_path}")
        return

    result = results[0]

    # 创建可视化画布
    vis_image = image_rgb.copy()

    # 生成颜色（为每个类别分配不同颜色）
    colors = [
        (255, 0, 0),      # 红色
        (0, 255, 0),      # 绿色
        (0, 0, 255),      # 蓝色
        (255, 255, 0),    # 黄色
        (255, 0, 255),    # 品红
        (0, 255, 255),    # 青色
        (128, 0, 128),    # 紫色
        (255, 165, 0),    # 橙色
        (0, 128, 128),    # 深青
        (128, 128, 0),    # 橄榄色
    ]

    # 处理每个检测
    try:
        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes

            for i in range(len(boxes)):
                # 获取置信度和类别
                conf = float(boxes.conf[i].cpu().item())
                class_id = int(boxes.cls[i].cpu().item())

                if conf < conf_threshold:
                    continue

                class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                color = colors[class_id % len(colors)]

                # 绘制边界框
                if show_bbox:
                    box = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                    # 绘制标签
                    label = f"{class_name} {conf:.2f}"
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        vis_image,
                        (x1, y1 - label_height - baseline - 5),
                        (x1 + label_width, y1),
                        color,
                        -1
                    )
                    cv2.putText(
                        vis_image,
                        label,
                        (x1, y1 - baseline - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )

                # 绘制mask - 修复版
                if show_mask and hasattr(result, 'masks') and result.masks is not None:
                    masks = result.masks.data  # 直接访问data属性

                    if masks is not None and len(masks) > 0:
                        if i < len(masks):
                            mask = masks[i].cpu().numpy()

                            # resize mask到图片尺寸
                            h, w = vis_image.shape[:2]
                            mask_resized = cv2.resize(mask, (w, h))

                            # 创建彩色mask
                            mask_binary = (mask_resized > 0.5).astype(np.uint8)
                            mask_colored = np.zeros_like(vis_image)

                            # 应用颜色
                            for c in range(3):
                                mask_colored[:, :, c] = mask_binary * color[c]

                            # 叠加mask
                            vis_image = cv2.addWeighted(
                                vis_image,
                                1.0,
                                mask_colored.astype(np.uint8),
                                mask_alpha,
                                0
                            )
        else:
            print(f"  未检测到目标（置信度阈值: {conf_threshold}）")

    except Exception as e:
        print(f"  可视化时出错: {e}")
        import traceback
        traceback.print_exc()

    # 保存或显示结果
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_path), vis_image_bgr)
        print(f"  保存可视化结果: {output_path}")
    else:
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.title(f'Predictions - {Path(image_path).name}')
        plt.tight_layout()
        plt.savefig('/tmp/prediction.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("  保存可视化结果: /tmp/prediction.png")


def validate_with_mask_visualization(
    weights_path,
    data_config_path,
    output_dir='./validation_vis',
    num_samples=10,
    conf_threshold=0.25,
    show_bbox=True,
    show_mask=True,
    mask_alpha=0.5,
    img_size=640
):
    """
    验证模型并可视化mask预测

    Args:
        weights_path: 模型权重路径
        data_config_path: 数据集配置路径
        output_dir: 输出目录
        num_samples: 可视化样本数量
        conf_threshold: 置信度阈值
        show_bbox: 是否显示边界框
        show_mask: 是否显示mask
        mask_alpha: mask透明度
        img_size: 输入图片尺寸
    """
    print("=" * 60)
    print("YOLO模型验证 - Mask可视化（修复版）")
    print("=" * 60)

    # 加载数据配置
    print(f"加载数据配置: {data_config_path}")
    data_config = load_data_config(data_config_path)

    # 获取类别名称
    if 'names' in data_config:
        if isinstance(data_config['names'], list):
            class_names = data_config['names']
        else:
            class_names = [data_config['names'][i] for i in sorted(data_config['names'].keys())]
    else:
        class_names = [f"Class_{i}" for i in range(data_config.get('nc', 2))]

    print(f"类别数量: {len(class_names)}")
    print(f"类别名称: {class_names}")

    # 加载模型
    print(f"\n加载模型: {weights_path}")
    try:
        model = YOLO(str(weights_path))
        print(f"  模型任务: {model.task}")
        print(f"  模型设备: {model.device if hasattr(model, 'device') else '未知'}")
    except Exception as e:
        print(f"  加载模型失败: {e}")
        return

    # 获取验证集路径
    dataset_root = Path(data_config['path'])
    val_images_dir = dataset_root / data_config['val'] / 'images'

    if not val_images_dir.exists():
        print(f"错误: 验证集目录不存在: {val_images_dir}")
        return

    # 获取验证图片列表
    image_files = list(val_images_dir.glob('*.jpg')) + list(val_images_dir.glob('*.png'))

    if len(image_files) == 0:
        print(f"错误: 验证集目录中没有找到图片: {val_images_dir}")
        return

    print(f"验证集图片数量: {len(image_files)}")

    # 随机选择样本
    import random
    random.seed(42)
    selected_images = random.sample(image_files, min(num_samples, len(image_files)))

    print(f"\n选择 {len(selected_images)} 张图片进行可视化...")

    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 可视化每个样本
    success_count = 0
    for i, image_path in enumerate(selected_images, 1):
        print(f"\n[{i}/{len(selected_images)}] 处理: {image_path.name}")

        output_path = output_dir / f"pred_{image_path.name}"

        try:
            visualize_predictions(
                model,
                image_path,
                class_names,
                conf_threshold=conf_threshold,
                output_path=output_path,
                show_bbox=show_bbox,
                show_mask=show_mask,
                mask_alpha=mask_alpha
            )
            success_count += 1
        except Exception as e:
            print(f"  处理失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print(f"可视化完成！成功: {success_count}/{len(selected_images)}")
    print(f"结果保存在: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='YOLO模型验证 - Mask可视化（修复版）')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--data', type=str, required=True,
                        help='数据集配置文件路径 (data.yaml)')
    parser.add_argument('--output_dir', type=str, default='./validation_vis',
                        help='输出目录')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='可视化样本数量')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='置信度阈值')
    parser.add_argument('--img_size', type=int, default=640,
                        help='输入图片尺寸')
    parser.add_argument('--no_bbox', action='store_true',
                        help='不显示边界框')
    parser.add_argument('--no_mask', action='store_true',
                        help='不显示mask')
    parser.add_argument('--mask_alpha', type=float, default=0.5,
                        help='mask透明度 (0-1)')

    args = parser.parse_args()

    validate_with_mask_visualization(
        weights_path=args.weights,
        data_config_path=args.data,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        conf_threshold=args.conf,
        show_bbox=not args.no_bbox,
        show_mask=not args.no_mask,
        mask_alpha=args.mask_alpha,
        img_size=args.img_size
    )


if __name__ == '__main__':
    main()
