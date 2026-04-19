"""
YOLO数据集转换模块
将分割标注数据转换为YOLO格式，支持：
1. 根据mask外接框裁剪图片
2. 保留mask轮廓信息，转换为YOLO polygon格式
3. 生成符合YOLO规范的标注文件
4. 多线程并行处理加速转换
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import random


class YOLOConverter:
    """YOLO数据集转换器（支持多线程并行处理）"""
    
    def __init__(self, input_dataset_path: str, output_dataset_path: str):
        """
        初始化转换器
        
        Args:
            input_dataset_path: 输入数据集路径（原始数据）
            output_dataset_path: 输出数据集路径（YOLO格式）
        """
        self.input_path = Path(input_dataset_path)
        self.output_path = Path(output_dataset_path)
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # 类别映射：{类别名: class_id}
        self.class_mapping: Dict[str, int] = {}
        self.next_class_id = 0
        
        # 线程锁
        self._lock = threading.Lock()
        
        # 统计信息
        self.stats = {
            'total_images': 0,
            'converted_images': 0,
            'skipped_images': 0,
            'total_masks': 0,
            'class_distribution': defaultdict(int),
            'image_sizes': defaultdict(list)  # {类别名: [(width, height), ...]}
        }
    
    def _get_class_id(self, class_name: str) -> int:
        """
        获取类别的数字ID，自动创建映射（线程安全）
        
        Args:
            class_name: 类别名称
            
        Returns:
            类别ID（整数）
        """
        with self._lock:
            if class_name not in self.class_mapping:
                self.class_mapping[class_name] = self.next_class_id
                self.next_class_id += 1
            return self.class_mapping[class_name]
    
    def _update_stats(self, converted: int = 0, skipped: int = 0, masks: int = 0,
                      class_dist: Dict[str, int] = None, size_dist: Dict[str, Tuple[int, int]] = None):
        """更新统计信息（线程安全）"""
        with self._lock:
            self.stats['converted_images'] += converted
            self.stats['skipped_images'] += skipped
            self.stats['total_masks'] += masks
            if class_dist:
                for cls, count in class_dist.items():
                    self.stats['class_distribution'][cls] += count
            if size_dist:
                for cls, size in size_dist.items():
                    self.stats['image_sizes'][cls].append(size)
    
    def load_dataset_structure(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        加载数据集目录结构
        
        Returns:
            {类别名: [(图片路径, json路径), ...]}
        """
        dataset_structure = defaultdict(list)
        
        # 遍历第一层目录（类别）
        for category_dir in self.input_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            # 遍历第二层目录（图片和JSON文件）
            for file_path in category_dir.iterdir():
                if file_path.suffix.lower() in self.supported_image_formats:
                    json_path = file_path.with_suffix('.json')
                    if json_path.exists():
                        dataset_structure[category_dir.name].append(
                            (str(file_path), str(json_path))
                        )
        
        return dict(dataset_structure)
    
    def parse_json_polygons(self, json_data: dict, img_height: int = 0, img_width: int = 0) -> List[Tuple[List[List[float]], str]]:
        """
        解析JSON标注中的polygon信息和标签（直接返回坐标，不转mask）
        
        Args:
            json_data: JSON标注数据
            img_height: 图片高度（用于确定坐标范围）
            img_width: 图片宽度（用于确定坐标范围）
            
        Returns:
            [(polygon, label), ...] - polygon格式: [[x1, y1], [x2, y2], ...]
        """
        results = []
        
        # 格式1: COCO格式（segmentation）
        if 'segmentation' in json_data:
            segmentation = json_data['segmentation']
            category_name = json_data.get('category_name', json_data.get('label', 'object'))
            
            if isinstance(segmentation, list):
                for idx, polygon in enumerate(segmentation):
                    # polygon格式可能是 [x1, y1, x2, y2, ...] 或 [[x1, y1], [x2, y2], ...]
                    if len(polygon) > 0 and isinstance(polygon[0], list):
                        # [[x1, y1], [x2, y2], ...] 格式
                        results.append((polygon, category_name if len(segmentation) == 1 else f"{category_name}_{idx+1}"))
                    else:
                        # [x1, y1, x2, y2, ...] 格式，转为 [[x1, y1], [x2, y2], ...]
                        converted = [[polygon[i], polygon[i+1]] for i in range(0, len(polygon), 2)]
                        results.append((converted, category_name if len(segmentation) == 1 else f"{category_name}_{idx+1}"))
        
        # 格式2: LabelMe格式（shapes）
        elif 'shapes' in json_data:
            for shape in json_data['shapes']:
                if shape['shape_type'] == 'polygon':
                    points = shape['points']
                    label = shape.get('label', 'object')
                    # 确保是 [[x1, y1], [x2, y2], ...] 格式
                    if len(points) > 0 and isinstance(points[0], list):
                        results.append((points, label))
                    else:
                        # [x1, y1, x2, y2, ...] 格式
                        converted = [[points[i], points[i+1]] for i in range(0, len(points), 2)]
                        results.append((converted, label))
        
        return results
    
    def get_mask_bbox(self, polygon: List[List[float]]) -> Tuple[int, int, int, int]:
        """
        获取polygon的外接矩形边界框
        
        Args:
            polygon: 多边形坐标 [[x1, y1], [x2, y2], ...]
            
        Returns:
            (x, y, w, h) - 左上角坐标和宽高
        """
        # 找所有坐标的最小最大值
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        
        x_min = int(min(x_coords))
        y_min = int(min(y_coords))
        x_max = int(max(x_coords))
        y_max = int(max(y_coords))
        
        return (x_min, y_min, x_max - x_min + 1, y_max - y_min + 1)
    
    def crop_and_convert(
        self, 
        img_path: str, 
        json_path: str,
        expand_ratio: float = 0.0,
        min_size: int = 32
    ) -> List[Tuple[np.ndarray, str, List[List[float]], Tuple[int, int, int, int]]]:
        """
        裁剪图片并转换标注（保留polygon坐标）
        
        Args:
            img_path: 图片路径
            json_path: JSON标注路径
            expand_ratio: 边界框扩展比例（0-1），增加裁剪区域
            min_size: 最小裁剪尺寸，低于此尺寸的mask将被跳过
            
        Returns:
            [(裁剪图片, 类别名, 原始polygon坐标, 原图坐标下的bbox), ...]
        """
        # 加载图片
        img = cv2.imread(img_path)
        if img is None:
            return []
        
        img_height, img_width = img.shape[:2]
        
        # 加载JSON标注
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 解析polygons（传入图片尺寸）
        polygons_with_labels = self.parse_json_polygons(json_data, img_height, img_width)
        
        results = []
        
        for polygon, label in polygons_with_labels:
            # 获取外接矩形
            x, y, w, h = self.get_mask_bbox(polygon)
            
            # 检查最小尺寸
            if w < min_size or h < min_size:
                continue
            
            # 扩展边界框
            if expand_ratio > 0:
                expand_w = int(w * expand_ratio)
                expand_h = int(h * expand_ratio)
                x = max(0, x - expand_w)
                y = max(0, y - expand_h)
                w = min(img_width - x, w + 2 * expand_w)
                h = min(img_height - y, h + 2 * expand_h)
            
            # 裁剪图片
            cropped_img = img[y:y+h, x:x+w]
            
            # 保存原始polygon坐标和bbox
            results.append((cropped_img, label, polygon, (x, y, w, h)))
        
        return results
    
    def polygon_to_yolo_format(
        self,
        original_polygon: List[List[float]],
        crop_bbox: Tuple[int, int, int, int],
        img_width: int,
        img_height: int,
        class_name: str
    ) -> str:
        """
        将polygon转换为YOLO polygon格式（归一化坐标）
        
        Args:
            original_polygon: 原始polygon坐标 [[x1, y1], [x2, y2], ...]
            crop_bbox: 裁剪边界框 (x, y, w, h)
            img_width: 裁剪后图片宽度
            img_height: 裁剪后图片高度
            class_name: 类别名称
            
        Returns:
            YOLO polygon格式的标注行: class_id x1 y1 x2 y2 ...
        """
        crop_x, crop_y, crop_w, crop_h = crop_bbox
        
        # 转换polygon坐标到裁剪后的小图坐标系
        cropped_polygon = []
        for point in original_polygon:
            new_x = point[0] - crop_x
            new_y = point[1] - crop_y
            # 归一化到0-1
            norm_x = new_x / img_width
            norm_y = new_y / img_height
            # 确保在0-1范围内
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))
            cropped_polygon.append(norm_x)
            cropped_polygon.append(norm_y)
        
        class_id = self._get_class_id(class_name)
        
        # 转换为字符串格式: class_id x1 y1 x2 y2 ...
        polygon_str = ' '.join([f"{coord:.6f}" for coord in cropped_polygon])
        
        return f"{class_id} {polygon_str}"
    
    def _process_single_image(
        self,
        img_path: str,
        json_path: str,
        category_img_dir: Path,
        category_label_dir: Path,
        expand_ratio: float,
        min_size: int,
        base_name_prefix: str = ""
    ) -> Tuple[int, int, int, Dict[str, int], List[str]]:
        """
        处理单张图片（供多线程调用）
        
        Returns:
            (converted_count, skipped_count, masks_count, class_dist, processed_names)
        """
        converted = 0
        skipped = 0
        masks = 0
        class_dist = defaultdict(int)
        processed_names = []
        
        # 裁剪并转换
        cropped_results = self.crop_and_convert(
            img_path, json_path,
            expand_ratio=expand_ratio,
            min_size=min_size
        )
        
        if not cropped_results:
            skipped = 1
            return converted, skipped, masks, dict(class_dist), processed_names
        
        base_name = Path(img_path).stem
        
        # 保存每个裁剪结果
        for idx, (cropped_img, label, polygon, bbox) in enumerate(cropped_results):
            converted += 1
            masks += 1
            class_dist[label] += 1
            
            # 生成唯一文件名
            output_name = f"{base_name_prefix}{base_name}_{idx+1}"
            processed_names.append(output_name)
            
            # 保存裁剪图片
            img_output_path = category_img_dir / f"{output_name}.jpg"
            cv2.imwrite(str(img_output_path), cropped_img)
            
            # 获取裁剪后图片尺寸
            crop_width = cropped_img.shape[1]
            crop_height = cropped_img.shape[0]
            
            # 生成YOLO polygon标注（直接从polygon转换，不经过mask图像）
            yolo_label = self.polygon_to_yolo_format(
                polygon, bbox,
                crop_width, crop_height,
                label
            )
            
            # 保存标注文件
            label_output_path = category_label_dir / f"{output_name}.txt"
            with open(label_output_path, 'w') as f:
                f.write(yolo_label + '\n')
            
            # 记录尺寸信息
            with self._lock:
                self.stats['image_sizes'][label].append((crop_width, crop_height))
        
        return converted, skipped, masks, dict(class_dist), processed_names
    
    def _process_category(
        self,
        category: str,
        samples: List[Tuple[str, str]],
        category_output_path: Path,
        output_images_subdir: str,
        output_labels_subdir: str,
        expand_ratio: float,
        min_size: int,
        num_workers: int
    ) -> Dict:
        """处理单个类别的所有样本（供多线程调用）"""
        # 创建输出子目录
        category_img_dir = category_output_path / output_images_subdir
        category_label_dir = category_output_path / output_labels_subdir
        category_img_dir.mkdir(parents=True, exist_ok=True)
        category_label_dir.mkdir(parents=True, exist_ok=True)
        
        category_stats = {
            'total': len(samples),
            'converted': 0,
            'skipped': 0,
            'masks': 0,
            'class_dist': defaultdict(int)
        }
        
        # 使用线程池并行处理该类别内的图片
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for img_path, json_path in samples:
                future = executor.submit(
                    self._process_single_image,
                    img_path, json_path,
                    category_img_dir, category_label_dir,
                    expand_ratio, min_size,
                    f"{category}_"
                )
                futures.append((Path(img_path).name, future))
            
            # 收集结果
            for img_name, future in futures:
                try:
                    converted, skipped, masks, class_dist, names = future.result()
                    category_stats['converted'] += converted
                    category_stats['skipped'] += skipped
                    category_stats['masks'] += masks
                    for cls, cnt in class_dist.items():
                        category_stats['class_dist'][cls] += cnt
                except Exception as e:
                    print(f"    处理失败 {img_name}: {str(e)}")
                    category_stats['skipped'] += 1
        
        return category_stats
    
    def convert_dataset(
        self,
        samples_per_class: Optional[int] = None,
        expand_ratio: float = 0.0,
        min_size: int = 32,
        output_images_subdir: str = "images",
        output_labels_subdir: str = "labels",
        preserve_category_structure: bool = True,
        num_workers: int = 8
    ):
        """
        转换整个数据集（多线程并行处理）
        
        Args:
            samples_per_class: 每个类别采样的样本数量，None表示全部
            expand_ratio: 边界框扩展比例
            min_size: 最小裁剪尺寸
            output_images_subdir: 输出图片子目录名
            output_labels_subdir: 输出标签子目录名
            preserve_category_structure: 是否保留类别目录结构（默认为True）
            num_workers: 并行处理的线程数（默认为8）
        """
        print("=" * 60)
        print("YOLO数据集转换（多线程并行）")
        print("=" * 60)
        print(f"并行线程数: {num_workers}")
        print(f"标注格式: YOLO Polygon (保留轮廓信息)")
        print()
        
        # 创建根目录
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载数据集结构
        dataset_structure = self.load_dataset_structure()
        
        print(f"输入数据集: {self.input_path}")
        print(f"输出数据集: {self.output_path}")
        print(f"发现 {len(dataset_structure)} 个类别")
        print(f"保留类别目录: {preserve_category_structure}")
        print()
        
        # 重置统计信息
        self.stats = {
            'total_images': 0,
            'converted_images': 0,
            'skipped_images': 0,
            'total_masks': 0,
            'class_distribution': defaultdict(int),
            'image_sizes': defaultdict(list)
        }
        
        # 准备每个类别的数据
        categories_data = []
        for category, samples in dataset_structure.items():
            # 采样
            if samples_per_class is not None and len(samples) > samples_per_class:
                samples = random.sample(samples, samples_per_class)
            
            # 确定输出目录
            if preserve_category_structure:
                category_output_path = self.output_path / category
            else:
                category_output_path = self.output_path
            
            categories_data.append({
                'category': category,
                'samples': samples,
                'output_path': category_output_path
            })
            
            self.stats['total_images'] += len(samples)
        
        print(f"总计待处理图片: {self.stats['total_images']}")
        print()
        
        # 使用线程池并行处理所有类别
        with ThreadPoolExecutor(max_workers=min(num_workers, len(categories_data))) as executor:
            futures = []
            for data in categories_data:
                future = executor.submit(
                    self._process_category,
                    data['category'],
                    data['samples'],
                    data['output_path'],
                    output_images_subdir,
                    output_labels_subdir,
                    expand_ratio,
                    min_size,
                    num_workers
                )
                futures.append((data['category'], future))
            
            # 收集结果
            for category, future in futures:
                try:
                    cat_stats = future.result()
                    
                    # 更新全局统计
                    self._update_stats(
                        converted=cat_stats['converted'],
                        skipped=cat_stats['skipped'],
                        masks=cat_stats['masks'],
                        class_dist=cat_stats['class_dist']
                    )
                    
                    print(f"  [{category}] 完成: {cat_stats['converted']} 个裁剪图 "
                          f"(原始: {cat_stats['total']}, "
                          f"跳过: {cat_stats['skipped']})")
                    
                except Exception as e:
                    print(f"  [{category}] 处理失败: {str(e)}")
        
        # 保存类别映射
        self._save_class_mapping()
        
        # 打印统计报告
        self._print_report()
    
    def _save_class_mapping(self):
        """保存类别映射文件"""
        mapping_path = self.output_path / "classes.txt"
        
        # 按class_id排序
        with self._lock:
            sorted_classes = sorted(self.class_mapping.items(), key=lambda x: x[1])
        
        with open(mapping_path, 'w', encoding='utf-8') as f:
            for class_name, class_id in sorted_classes:
                f.write(f"{class_id} {class_name}\n")
    
    def _print_report(self):
        """打印转换报告"""
        print("\n" + "=" * 60)
        print("转换报告")
        print("=" * 60)
        
        print(f"\n【总体统计】")
        print(f"  原始图片数: {self.stats['total_images']}")
        print(f"  生成裁剪图: {self.stats['converted_images']}")
        print(f"  跳过图片数: {self.stats['skipped_images']}")
        print(f"  总目标数: {self.stats['total_masks']}")
        
        print(f"\n【类别分布】")
        for class_name, count in sorted(self.stats['class_distribution'].items(), key=lambda x: -x[1]):
            print(f"  {class_name}: {count}")
        
        print(f"\n【图像尺寸分布】")
        for class_name, sizes in sorted(self.stats['image_sizes'].items()):
            if sizes:
                widths = [w for w, h in sizes]
                heights = [h for w, h in sizes]
                min_w, max_w = min(widths), max(widths)
                min_h, max_h = min(heights), max(heights)
                avg_w = sum(widths) / len(widths)
                avg_h = sum(heights) / len(heights)
                print(f"  {class_name}:")
                print(f"    数量: {len(sizes)}")
                print(f"    宽度: min={min_w}, max={max_w}, avg={avg_w:.1f}")
                print(f"    高度: min={min_h}, max={max_h}, avg={avg_h:.1f}")
        
        print(f"\n【输出目录结构】")
        print(f"  根目录: {self.output_path}")
        print(f"  类别文件: {self.output_path / 'classes.txt'}")
        print(f"\n  按类别分目录结构:")
        for category in self.stats['class_distribution'].keys():
            category_path = self.output_path / category
            print(f"    {category}/")
            print(f"      images/  # 裁剪图片")
            print(f"      labels/  # YOLO polygon标注文件")
        
        print("\n" + "=" * 60)
