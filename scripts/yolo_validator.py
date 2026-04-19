"""
YOLO数据集验证脚本
将mask绘制回裁剪后的图片，验证转换逻辑是否正确
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import random
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


class YOLOValidator:
    """YOLO数据集验证器"""
    
    def __init__(self, yolo_dataset_path: str):
        """
        初始化验证器
        
        Args:
            yolo_dataset_path: YOLO数据集路径
        """
        self.yolo_path = Path(yolo_dataset_path)
        
        # 加载类别映射
        self.classes = {}
        self.class_mapping = {}
        self._load_classes()
        
        # 支持的图片格式
        self.image_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    def _load_classes(self):
        """加载classes.txt文件"""
        classes_file = self.yolo_path / "classes.txt"
        if not classes_file.exists():
            raise FileNotFoundError(f"找不到classes.txt文件: {classes_file}")
        
        with open(classes_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    class_id = int(parts[0])
                    class_name = parts[1]
                    self.classes[class_id] = class_name
                    self.class_mapping[class_name] = class_id
        
        print(f"加载了 {len(self.classes)} 个类别")
    
    def load_dataset_structure(self) -> Dict[str, Dict]:
        """
        加载数据集结构
        
        Returns:
            {类别名: {'images': [], 'labels': []}}
        """
        dataset_structure = defaultdict(lambda: {'images': [], 'labels': []})
        
        # 遍历所有类别目录
        for category_dir in self.yolo_path.iterdir():
            if not category_dir.is_dir() or category_dir.name == 'classes.txt':
                continue
            
            category_name = category_dir.name
            images_dir = category_dir / 'images'
            labels_dir = category_dir / 'labels'
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            # 加载图片和标签
            for img_file in images_dir.iterdir():
                if img_file.suffix.lower() in self.image_formats:
                    label_file = labels_dir / (img_file.stem + '.txt')
                    if label_file.exists():
                        dataset_structure[category_name]['images'].append(str(img_file))
                        dataset_structure[category_name]['labels'].append(str(label_file))
        
        return dict(dataset_structure)
    
    def parse_yolo_label(self, label_path: str) -> List[Tuple[int, List[float]]]:
        """
        解析YOLO标签文件（支持polygon格式）
        
        Args:
            label_path: 标签文件路径
            
        Returns:
            [(class_id, [x1, y1, x2, y2, ...]), ...] - 归一化坐标
        """
        annotations = []
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                
                class_id = int(parts[0])
                coords = [float(p) for p in parts[1:]]
                
                # 如果是5个坐标，是边界框格式；如果是更多，是polygon格式
                if len(coords) == 4:
                    # 边界框格式：x_center, y_center, width, height
                    # 转换为polygon（4个点）
                    x_center, y_center, width, height = coords
                    x1 = x_center - width / 2
                    y1 = y_center - height / 2
                    x2 = x_center + width / 2
                    y2 = y_center + height / 2
                    polygon_coords = [x1, y1, x2, y1, x2, y2, x1, y2]
                    annotations.append((class_id, polygon_coords))
                else:
                    # Polygon格式：x1, y1, x2, y2, ...
                    annotations.append((class_id, coords))
        
        return annotations
    
    def yolo_polygon_to_mask(self,
                           coords: List[float],
                           img_width: int,
                           img_height: int) -> np.ndarray:
        """
        将YOLO polygon格式转换为mask
        
        Args:
            coords: 归一化的polygon坐标 [x1, y1, x2, y2, ...]
            img_width: 图片宽度
            img_height: 图片高度
            
        Returns:
            二值mask数组
        """
        # 将归一化坐标转换为像素坐标
        points = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * img_width)
            y = int(coords[i+1] * img_height)
            points.append([x, y])
        
        # 创建mask
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        if len(points) >= 3:
            points_np = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [points_np], 255)
        
        return mask
    
    def yolo_to_bbox(self, 
                     x_center: float, 
                     y_center: float, 
                     width: float, 
                     height: float, 
                     img_width: int, 
                     img_height: int) -> Tuple[int, int, int, int]:
        """
        将YOLO格式转换为像素坐标的边界框（用于绘制标签文字）
        
        Args:
            x_center: 中心点X坐标（归一化）
            y_center: 中心点Y坐标（归一化）
            width: 宽度（归一化）
            height: 高度（归一化）
            img_width: 图片宽度
            img_height: 图片高度
            
        Returns:
            (x, y, w, h) - 左上角坐标和宽高
        """
        x = int((x_center - width / 2) * img_width)
        y = int((y_center - height / 2) * img_height)
        w = int(width * img_width)
        h = int(height * img_height)
        
        return x, y, w, h
    
    def draw_mask_on_image(self,
                          img: np.ndarray,
                          class_id: int,
                          mask: np.ndarray) -> np.ndarray:
        """
        在图片上绘制mask和类别名称
        
        Args:
            img: 原始图片
            class_id: 类别ID
            mask: 二值mask
            
        Returns:
            绘制后的图片
        """
        result = img.copy()
        
        # 颜色（基于类别ID）
        colors = [
            (0, 0, 255),      # 红色
            (0, 255, 0),      # 绿色
            (255, 0, 0),      # 蓝色
            (0, 255, 255),    # 黄色
            (255, 0, 255),    # 品红
            (255, 255, 0),    # 青色
        ]
        color = colors[class_id % len(colors)]
        
        # 创建彩色mask
        color_mask = np.zeros_like(result)
        color_mask[mask > 0] = color
        
        # 半透明叠加mask
        alpha = 0.5
        result = cv2.addWeighted(result, 1 - alpha, color_mask, alpha, 0)
        
        # 绘制mask轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
        
        # 绘制类别名称（在mask中心）
        if contours:
            # 计算mask中心点
            M = cv2.moments(contours[0])
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                
                class_name = self.classes.get(class_id, f"class_{class_id}")
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 2
                
                # 计算文本大小
                (text_w, text_h), _ = cv2.getTextSize(class_name, font, font_scale, font_thickness)
                
                # 绘制文本背景
                cv2.rectangle(result,
                             (cx - text_w // 2 - 5, cy - text_h // 2 - 5),
                             (cx + text_w // 2 + 5, cy + text_h // 2 + 5),
                             (0, 0, 0), -1)
                
                # 绘制文本
                cv2.putText(result, class_name,
                          (cx - text_w // 2, cy + text_h // 2),
                          font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        return result
    
    def validate_and_visualize(
        self,
        samples_per_class: int = 10,
        output_path: str = "./validation_output"
    ):
        """
        验证数据集并生成可视化图片
        
        Args:
            samples_per_class: 每个类别随机选择的样本数量
            output_path: 输出目录路径
        """
        print("=" * 60)
        print("YOLO数据集验证与可视化")
        print("=" * 60)
        
        # 加载数据集结构
        dataset_structure = self.load_dataset_structure()
        
        if not dataset_structure:
            print("错误: 数据集为空或结构不正确")
            return
        
        print(f"\n发现 {len(dataset_structure)} 个类别")
        print(f"每类随机选择 {samples_per_class} 张图片")
        print()
        
        # 创建输出目录
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 为每个类别创建子目录
        total_validated = 0
        
        # 验证每个类别
        for category_name, data in dataset_structure.items():
            print(f"处理类别: {category_name}")
            
            images = data['images']
            labels = data['labels']
            
            # 随机采样
            if len(images) > samples_per_class:
                indices = random.sample(range(len(images)), samples_per_class)
                selected_images = [images[i] for i in indices]
                selected_labels = [labels[i] for i in indices]
            else:
                selected_images = images
                selected_labels = labels
            
            print(f"  选择了 {len(selected_images)} 张图片")
            
            # 为该类别创建子目录
            category_output = output_path / category_name
            category_output.mkdir(parents=True, exist_ok=True)
            
            # 验证并保存每张图片
            validated_count = self._validate_and_save_images(
                category_name,
                selected_images,
                selected_labels,
                category_output
            )
            
            print(f"  ✓ 已保存 {validated_count} 张验证图片")
            total_validated += validated_count
        
        # 生成汇总报告
        self._generate_summary_report(dataset_structure, output_path)
        
        print(f"\n总计验证了 {total_validated} 张图片")
        print(f"所有验证图片已保存到: {output_path}")
        print("=" * 60)
    
    def _validate_and_save_images(
        self,
        category_name: str,
        images: List[str],
        labels: List[str],
        output_dir: Path
    ) -> int:
        """
        验证并保存每张图片（每张单独保存）
        
        Args:
            category_name: 类别名称
            images: 图片路径列表
            labels: 标签路径列表
            output_dir: 输出目录
            
        Returns:
            成功验证的图片数量
        """
        validated_count = 0
        
        for idx, (img_path, label_path) in enumerate(zip(images, labels)):
            try:
                # 读取图片
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img_height, img_width = img.shape[:2]
                
                # 解析标签
                annotations = self.parse_yolo_label(label_path)
                
                # 绘制所有mask
                for class_id, coords in annotations:
                    # 转换为mask
                    mask = self.yolo_polygon_to_mask(coords, img_width, img_height)
                    # 绘制mask
                    img = self.draw_mask_on_image(img, class_id, mask)
                
                # 保存验证图片
                img_name = Path(img_path).stem
                output_path = output_dir / f"{img_name}_validated.jpg"
                cv2.imwrite(str(output_path), img)
                
                validated_count += 1
                
            except Exception as e:
                print(f"    处理失败 {Path(img_path).name}: {str(e)}")
        
        return validated_count
    
    def _generate_summary_report(
        self,
        dataset_structure: Dict,
        output_path: Path
    ):
        """生成汇总报告"""
        report_path = output_path / "validation_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("YOLO数据集验证报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"数据集路径: {self.yolo_path}\n")
            f.write(f"输出路径: {output_path}\n\n")
            
            f.write("【类别统计】\n")
            f.write(f"{'类别名称':<20} {'样本数量':>10}\n")
            f.write("-" * 30 + "\n")
            
            total_samples = 0
            for category, data in sorted(dataset_structure.items()):
                count = len(data['images'])
                total_samples += count
                f.write(f"{category:<20} {count:>10}\n")
            
            f.write("-" * 30 + "\n")
            f.write(f"{'总计':<20} {total_samples:>10}\n\n")
            
            f.write("【类别映射】\n")
            f.write(f"{'类别ID':<10} {'类别名称':<20}\n")
            f.write("-" * 30 + "\n")
            
            for class_id, class_name in sorted(self.classes.items()):
                f.write(f"{class_id:<10} {class_name:<20}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        print(f"  已生成验证报告: {report_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='YOLO数据集验证与可视化工具'
    )
    
    parser.add_argument(
        '--yolo_path',
        type=str,
        required=True,
        help='YOLO数据集路径'
    )
    
    parser.add_argument(
        '--samples_per_class',
        type=int,
        default=10,
        help='每个类别随机选择的图片数量（默认10）'
    )
    
    parser.add_argument(
        '--output_path',
        type=str,
        default="./validation_output",
        help='验证图片输出目录'
    )
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = YOLOValidator(args.yolo_path)
    
    # 执行验证和可视化
    validator.validate_and_visualize(
        samples_per_class=args.samples_per_class,
        output_path=args.output_path
    )


if __name__ == '__main__':
    main()
