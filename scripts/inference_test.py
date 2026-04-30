"""
推理测试脚本
用于处理单张图片或整个文件夹的图片，完成实例分割并展示结果
"""

import argparse
import os
import sys
from pathlib import Path

# 添加项目根目录到模块搜索路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from service.inference import YOLOMaskService


def visualize_segmentation_result(image, detections, class_names=None):
    """
    可视化分割结果
    
    Args:
        image: 原始图像 (RGB格式)
        detections: 检测结果列表
        class_names: 类别名称列表
    """
    # 复制图像以便绘制
    vis_image = image.copy()
    
    for detection in detections:
        # 获取边界框和类别信息
        bbox = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class_id']
        contours = detection.get('contours', [])
        
        # 绘制边界框
        cv2.rectangle(vis_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        
        # 显示类别和置信度
        label = f"Class {class_id}: {confidence:.2f}"
        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {confidence:.2f}"
        
        cv2.putText(vis_image, label, (int(bbox[0]), int(bbox[1]) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 绘制分割轮廓
        for contour_info in contours:
            points = np.array(contour_info['points'], dtype=np.int32)
            if len(points) > 0:
                cv2.polylines(vis_image, [points], True, (255, 0, 0), 2)
                
                # 填充分割区域
                mask = np.zeros(vis_image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)
                
                # 创建彩色掩码层
                color_mask = np.zeros_like(vis_image)
                color_mask[:, :, 1] = 255  # 绿色通道
                masked_region = np.where(mask == 255)
                vis_image[masked_region[0], masked_region[1], :] = (
                    0.7 * vis_image[masked_region[0], masked_region[1], :].astype(np.float32) + 
                    0.3 * color_mask[masked_region[0], masked_region[1], :].astype(np.float32)
                ).astype(np.uint8)
    
    return vis_image


def process_single_image(image_path, model_service, class_names=None):
    """
    处理单张图片的分割任务
    
    Args:
        image_path: 图片路径
        model_service: 模型服务实例
        class_names: 类别名称列表
    """
    # 读取图像
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"无法读取图片: {image_path}")
        return
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 执行预测
    result = model_service.predict(image_rgb)
    
    # 获取检测结果
    all_detections = []
    for res in result['results']:
        all_detections.extend(res['detections'])
    
    # 可视化结果
    vis_image = visualize_segmentation_result(image_rgb, all_detections, class_names)
    
    # 显示结果
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plt.title("原始图像")
    plt.imshow(image_rgb)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("分割结果")
    plt.imshow(vis_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"处理图片: {image_path}")
    print(f"检测到 {len(all_detections)} 个目标")


def process_image_folder(folder_path, model_service, class_names=None):
    """
    处理文件夹中的所有图片
    
    Args:
        folder_path: 图片文件夹路径
        model_service: 模型服务实例
        class_names: 类别名称列表
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_paths = []
    
    # 收集所有图片文件
    for ext in image_extensions:
        image_paths.extend(list(Path(folder_path).rglob(f'*{ext}')))
        image_paths.extend(list(Path(folder_path).rglob(f'*{ext.upper()}')))
    
    if not image_paths:
        print(f"在文件夹 {folder_path} 中未找到任何图片文件")
        return
    
    print(f"找到 {len(image_paths)} 张图片")
    
    for idx, image_path in enumerate(image_paths):
        print(f"\n处理第 {idx+1}/{len(image_paths)} 张图片: {image_path.name}")
        
        # 读取图像
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            print(f"无法读取图片: {image_path}")
            continue
        
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # 执行预测
        result = model_service.predict(image_rgb)
        
        # 获取检测结果
        all_detections = []
        for res in result['results']:
            all_detections.extend(res['detections'])
        
        # 可视化结果
        vis_image = visualize_segmentation_result(image_rgb, all_detections, class_names)
        
        # 显示结果
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.title(f"原始图像 - {image_path.name}")
        plt.imshow(image_rgb)
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title(f"分割结果 - 检测到 {len(all_detections)} 个目标")
        plt.imshow(vis_image)
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print(f"检测到 {len(all_detections)} 个目标")


def load_class_names_from_yaml(data_yaml_path):
    """
    从yaml配置文件中加载类别名称
    """
    import yaml
    
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 获取类别名称列表
        if 'names' in data:
            names_dict = data['names']
            if isinstance(names_dict, dict):
                # 如果是字典格式 {0: 'class1', 1: 'class2', ...}
                class_names = [''] * len(names_dict)
                for k, v in names_dict.items():
                    class_names[int(k)] = v
                return class_names
            elif isinstance(names_dict, list):
                # 如果是列表格式 ['class1', 'class2', ...]
                return names_dict
        
        return None
    except Exception as e:
        print(f"加载类别名称失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="YOLO实例分割推理测试")
    parser.add_argument("--weights", type=str, required=True, help="模型权重文件路径")
    parser.add_argument("--image", type=str, help="单张图片路径")
    parser.add_argument("--folder", type=str, help="图片文件夹路径")
    parser.add_argument("--device", type=str, default="0", help="推理设备 (默认: 0)")
    parser.add_argument("--conf-threshold", type=float, default=0.25, help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--img-size", type=int, default=640, help="输入图像尺寸 (默认: 640)")
    parser.add_argument("--data-config", type=str, help="数据配置文件路径，用于加载类别名称")
    
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        print("请指定 --image 或 --folder 参数")
        return
    
    # 加载类别名称
    class_names = None
    if args.data_config:
        class_names = load_class_names_from_yaml(args.data_config)
    
    # 初始化模型服务
    print("初始化模型服务...")
    model_service = YOLOMaskService(
        weights_path=args.weights,
        device=args.device,
        conf_threshold=args.conf_threshold,
        img_size=args.img_size
    )
    
    if args.image:
        image_path = Path(args.image)
        if image_path.is_dir():
            # 如果image参数实际上是一个目录，则按目录处理
            print(f"检测到 {args.image} 是一个目录，将按目录处理...")
            process_image_folder(image_path, model_service, class_names)
        else:
            # 处理单张图片
            if not image_path.exists():
                print(f"图片不存在: {image_path}")
                return
            process_single_image(image_path, model_service, class_names)
    
    elif args.folder:
        # 处理图片文件夹
        folder_path = Path(args.folder)
        if not folder_path.exists():
            print(f"文件夹不存在: {folder_path}")
            return
        process_image_folder(folder_path, model_service, class_names)


if __name__ == "__main__":
    main()