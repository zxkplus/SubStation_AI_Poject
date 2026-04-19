"""
可视化模块
负责将分割标注（mask）叠加到原图上，并提供交互式展示
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt


class MaskVisualizer:
    """Mask可视化器"""
    
    # 常用颜色表（用于不同类别的mask）
    COLORS = [
        (255, 0, 0),      # 红色
        (0, 255, 0),      # 绿色
        (0, 0, 255),      # 蓝色
        (255, 255, 0),    # 黄色
        (255, 0, 255),    # 品红
        (0, 255, 255),    # 青色
        (255, 128, 0),    # 橙色
        (128, 0, 255),    # 紫色
    ]
    
    def __init__(self, alpha: float = 0.5):
        """
        初始化可视化器
        
        Args:
            alpha: mask叠加的透明度（0-1）
        """
        self.alpha = alpha
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            BGR格式的numpy数组
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法加载图片: {image_path}")
        return img
    
    def parse_json_mask(self, json_data: dict, img_height: int, img_width: int) -> Tuple[List[np.ndarray], List[str]]:
        """
        解析JSON标注中的mask信息
        
        Args:
            json_data: JSON标注数据
            img_height: 实际图片高度
            img_width: 实际图片宽度
            
        Returns:
            (mask列表, 标签列表)
            mask列表：每个mask是一个二值numpy数组
            标签列表：每个mask对应的类别名称
        """
        masks = []
        labels = []
        
        # 尝试多种常见的mask存储格式
        # 格式1: COCO格式的RLE编码
        if 'segmentation' in json_data:
            segmentation = json_data['segmentation']
            category_name = json_data.get('category_name', json_data.get('label', 'mask'))
            if isinstance(segmentation, list):
                # Polygon格式
                for idx, polygon in enumerate(segmentation):
                    # 优先使用JSON中的尺寸，否则使用实际图片尺寸
                    height = json_data.get('image_height', img_height)
                    width = json_data.get('image_width', img_width)
                    mask = self._polygon_to_mask(polygon, height, width)
                    if mask is not None:
                        masks.append(mask)
                        # 如果有多个polygon，添加序号
                        if len(segmentation) > 1:
                            labels.append(f"{category_name}_{idx+1}")
                        else:
                            labels.append(category_name)
        
        # 格式2: 直接的mask数据（base64编码或二进制）
        elif 'mask' in json_data:
            mask_data = json_data['mask']
            category_name = json_data.get('category_name', 'mask')
            if isinstance(mask_data, list):
                # 直接的mask数组
                mask = np.array(mask_data, dtype=np.uint8)
                if len(mask.shape) == 2:
                    masks.append(mask)
                    labels.append(category_name)
        
        # 格式3: LabelMe格式（包含shapes）
        elif 'shapes' in json_data:
            for shape in json_data['shapes']:
                if shape['shape_type'] == 'polygon':
                    points = shape['points']
                    label = shape.get('label', 'mask')
                    # 优先使用JSON中的尺寸，否则使用实际图片尺寸
                    image_height = json_data.get('imageHeight', img_height)
                    image_width = json_data.get('imageWidth', img_width)
                    mask = self._polygon_to_mask(points, image_height, image_width)
                    if mask is not None:
                        masks.append(mask)
                        labels.append(label)
        
        return masks, labels
    
    def _polygon_to_mask(self, polygon: List, height: int, width: int) -> Optional[np.ndarray]:
        """
        将多边形转换为mask
        
        Args:
            polygon: 多边形坐标列表 [[x1, y1], [x2, y2], ...]
            height: 图像高度
            width: 图像宽度
            
        Returns:
            二值mask数组
        """
        try:
            # 转换为numpy数组
            points = np.array(polygon, dtype=np.int32)
            
            # 如果是嵌套的 [[x, y], [x, y]] 格式，直接使用
            if len(points.shape) == 2 and points.shape[1] == 2:
                points = points.reshape((-1, 1, 2))
            else:
                # 如果是 [x1, y1, x2, y2, ...] 格式，重塑
                points = points.reshape((-1, 2)).reshape((-1, 1, 2))
            
            # 如果没有提供高度和宽度，根据多边形坐标推断
            if height == 0 or width == 0:
                max_x = int(np.max(points[:, 0, 0]))
                max_y = int(np.max(points[:, 0, 1]))
                width = max(max_x, 1)
                height = max(max_y, 1)
            
            # 创建mask
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # 填充多边形
            cv2.fillPoly(mask, [points], 255)
            
            return mask
        except Exception as e:
            print(f"转换多边形到mask失败: {str(e)}")
            return None
    
    def visualize_single_sample(self, img_path: str, json_path: str) -> np.ndarray:
        """
        可视化单个样本，将mask叠加到原图
        
        Args:
            img_path: 图片路径
            json_path: JSON标注路径
            
        Returns:
            叠加了mask的图片
        """
        # 加载图片
        img = self.load_image(img_path)
        img_height, img_width = img.shape[:2]
        
        # 加载JSON标注
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 解析mask
        masks, labels = self.parse_json_mask(json_data, img_height, img_width)
        
        if not masks:
            print(f"警告: {json_path} 中没有找到有效的mask数据")
            return img
        
        # 创建彩色mask
        color_mask = np.zeros_like(img)
        for i, mask in enumerate(masks):
            color = self.COLORS[i % len(self.COLORS)]
            # 确保mask尺寸与图片匹配
            if mask.shape[:2] != img.shape[:2]:
                # 调整mask尺寸
                mask_resized = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
                color_mask[mask_resized > 0] = color
            else:
                color_mask[mask > 0] = color
        
        # 叠加mask到原图
        result = cv2.addWeighted(img, 1 - self.alpha, color_mask, self.alpha, 0)
        
        # 绘制轮廓和标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        
        for i, (mask, label) in enumerate(zip(masks, labels)):
            # 调整mask尺寸
            if mask.shape[:2] != img.shape[:2]:
                mask = cv2.resize(mask, (img_width, img_height), interpolation=cv2.INTER_NEAREST)
            
            # 绘制轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (255, 255, 255), 2)
            
            # 绘制标签
            if contours:
                # 找到轮廓的中心点
                cnt = contours[0]
                M = cv2.moments(cnt)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    # 计算文本大小
                    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                    
                    # 绘制文本背景（半透明黑色）
                    text_bg = np.zeros_like(result)
                    cv2.rectangle(text_bg, 
                                (cx - text_w//2 - 5, cy - text_h//2 - 5),
                                (cx + text_w//2 + 5, cy + text_h//2 + 5),
                                (0, 0, 0), -1)
                    result = cv2.addWeighted(result, 0.7, text_bg, 0.3, 0)
                    
                    # 绘制文本
                    cv2.putText(result, label, 
                              (cx - text_w//2, cy + text_h//2),
                              font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
        return result
    
    def show_interactive(self, samples_dict: Dict[str, List[Tuple[str, str]]], output_dir: str = "./visualization_output"):
        """
        保存可视化结果到文件（替代GUI弹窗）

        Args:
            samples_dict: {类别名: [(img_path, json_path), ...]}
            output_dir: 输出目录路径
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 统计信息
        total_samples = sum(len(samples) for samples in samples_dict.values())
        
        print(f"\n开始生成可视化结果...")
        print(f"输出目录: {output_dir}")
        
        # 为每个样本生成可视化图片
        sample_idx = 0
        for category, samples in samples_dict.items():
            print(f"\n处理类别: {category}")
            
            for img_path, json_path in samples:
                try:
                    # 可视化单个样本
                    result_img = self.visualize_single_sample(img_path, json_path)
                    
                    # 转换为RGB
                    result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                    
                    # 生成输出文件名
                    img_name = Path(img_path).stem
                    output_filename = f"{sample_idx+1:03d}_{category}_{img_name}_visualized.png"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # 保存图片
                    plt.figure(figsize=(12, 8))
                    plt.imshow(result_rgb)
                    plt.title(f"{category} - {img_name}", fontsize=12)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(output_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    
                    sample_idx += 1
                    print(f"  ✓ 已保存: {output_filename}")
                    
                except Exception as e:
                    print(f"  ✗ 处理失败 {Path(img_path).name}: {str(e)}")
        
        # 生成汇总报告
        print(f"\n可视化完成!")
        print(f"总共处理了 {sample_idx} 个样本")
        print(f"所有图片已保存到: {output_dir}")
        print("\n提示: 在支持GUI的环境中，可以使用matplotlib的交互式查看器查看这些图片")
