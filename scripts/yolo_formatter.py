"""
YOLO格式转换模块
将分割标注数据转换为YOLO格式，支持：
1. JSON标注转换为YOLO txt格式
2. 生成类别映射文件
3. 数据集划分（训练/验证/测试）
4. 生成YOLO配置文件
5. 多线程并行处理加速转换
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading
import random
import yaml


class YOLOFormatter:
    """YOLO格式转换器（支持多线程并行处理）"""
    
    def __init__(self, input_dataset_path: str, output_dataset_path: str, ignore_classes: List[str] = None, class_mapping_file: str = None):
        """
        初始化转换器

        Args:
            input_dataset_path: 输入数据集路径（原始数据）
            output_dataset_path: 输出数据集路径（YOLO格式）
            ignore_classes: 要忽略的类别名称列表
            class_mapping_file: 类别映射文件路径，用于将中文类别名映射为英文类别名
        """
        self.input_path = Path(input_dataset_path)
        self.output_path = Path(output_dataset_path)
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        self.ignore_classes = set(ignore_classes) if ignore_classes else set()
        self.class_name_mapping = self._load_class_mapping(class_mapping_file) if class_mapping_file else {}

        # 线程锁
        self._lock = threading.Lock()

        # 统计信息
        self.stats = {
            'total_images': 0,
            'converted_images': 0,
            'skipped_images': 0,
            'total_annotations': 0,
            'class_distribution': defaultdict(int)
        }

        # 类别映射（名称到ID）
        self.class_mapping = {}  # {class_name: class_id}
        self.reverse_class_mapping = {}  # {class_id: class_name}
        self.next_class_id = 0

    def _load_class_mapping(self, mapping_file: str) -> Dict[str, str]:
        """加载类别映射文件（格式: 中文名:英文名，每行一个映射）"""
        mapping = {}
        try:
            with open(mapping_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if ':' in line:
                            src_name, dst_name = line.split(':', 1)
                            src_name = src_name.strip()
                            dst_name = dst_name.strip()
                            if src_name and dst_name:
                                mapping[src_name] = dst_name
        except Exception as e:
            print(f"警告: 加载类别映射文件失败: {str(e)}")
        return mapping

    def _map_class_name(self, class_name: str) -> str:
        """映射类别名，如果存在映射则返回映射后的名称，否则返回原名称"""
        return self.class_name_mapping.get(class_name, class_name)

    def _update_stats(self, converted: int = 0, skipped: int = 0, annotations: int = 0,
                      class_dist: Dict[str, int] = None):
        """更新统计信息（线程安全）"""
        with self._lock:
            self.stats['converted_images'] += converted
            self.stats['skipped_images'] += skipped
            self.stats['total_annotations'] += annotations
            if class_dist:
                for cls, count in class_dist.items():
                    self.stats['class_distribution'][cls] += count
    
    def _get_class_id(self, class_name: str) -> Optional[int]:
        """获取或分配类别ID，如果类别被忽略则返回None（线程安全）"""
        if class_name in self.ignore_classes:
            return None

        # 快速路径：如果已存在，直接返回（避免不必要的锁竞争）
        if class_name in self.class_mapping:
            return self.class_mapping[class_name]

        with self._lock:
            # 双重检查：锁内再次检查，防止其他线程已添加
            if class_name not in self.class_mapping:
                self.class_mapping[class_name] = self.next_class_id
                self.reverse_class_mapping[self.next_class_id] = class_name
                self.next_class_id += 1
            return self.class_mapping[class_name]
    
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
            
            # 如果类别被忽略，跳过整个目录
            if category_dir.name in self.ignore_classes:
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
    
    def parse_json_polygons(self, json_data: dict, img_height: int, img_width: int) -> List[Tuple[List[List[float]], str]]:
        """
        解析JSON标注中的polygon信息和标签
        
        Args:
            json_data: JSON标注数据
            img_height: 图片高度
            img_width: 图片宽度
            
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
    
    def polygon_to_yolo_format(self, polygon: List[List[float]], img_width: int, img_height: int) -> List[float]:
        """
        将polygon坐标转换为YOLO格式（归一化坐标）
        
        Args:
            polygon: 多边形坐标 [[x1, y1], [x2, y2], ...]
            img_width: 图片宽度
            img_height: 图片高度
            
        Returns:
            [x1, y1, x2, y2, ...] - 归一化坐标
        """
        yolo_coords = []
        for x, y in polygon:
            norm_x = x / img_width
            norm_y = y / img_height
            yolo_coords.extend([norm_x, norm_y])
        return yolo_coords
    
    def convert_to_yolo_format(
        self,
        img_path: str,
        json_path: str
    ) -> Optional[Tuple[str, List[str]]]:
        """
        转换单个样本为YOLO格式
        
        Args:
            img_path: 图片路径
            json_path: JSON标注路径
            
        Returns:
            (图片路径, [YOLO格式标注行, ...]) 或 None
        """
        # 加载图片获取尺寸
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        img_height, img_width = img.shape[:2]
        
        # 加载JSON标注
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        # 解析polygons
        polygons_with_labels = self.parse_json_polygons(json_data, img_height, img_width)
        
        if not polygons_with_labels:
            return None
        
        # 转换为YOLO格式，过滤被忽略的类别
        yolo_annotations = []
        class_dist = defaultdict(int)
        
        for polygon, label in polygons_with_labels:
            # 应用类别名映射（如中文→英文）
            mapped_label = self._map_class_name(label)

            # 检查标签是否被忽略
            if mapped_label in self.ignore_classes:
                continue

            class_id = self._get_class_id(mapped_label)
            if class_id is None:  # 类别被忽略
                continue

            yolo_coords = self.polygon_to_yolo_format(polygon, img_width, img_height)

            # 构造YOLO格式字符串
            yolo_line = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in yolo_coords])
            yolo_annotations.append(yolo_line)
            class_dist[mapped_label] += 1
        
        # 如果所有标注都被过滤掉了，返回None
        if not yolo_annotations:
            return None
            
        # 更新统计
        self._update_stats(annotations=len(yolo_annotations), class_dist=class_dist)
        
        return img_path, yolo_annotations
    
    def _process_single_sample(self, img_path: str, json_path: str, output_labels_dir: Path) -> Optional[Tuple[str, str]]:
        """
        处理单个样本
        
        Returns:
            (图片相对路径, 标签相对路径) 或 None
        """
        result = self.convert_to_yolo_format(img_path, json_path)
        if result is None:
            return None
        
        img_path_str, yolo_annotations = result
        
        # 生成标签文件
        img_filename = Path(img_path_str).stem
        label_filename = img_filename + '.txt'
        label_path = output_labels_dir / label_filename
        
        # 写入YOLO格式标注
        with open(label_path, 'w', encoding='utf-8') as f:
            for annotation in yolo_annotations:
                f.write(annotation + '\n')
        
        # 返回相对路径
        rel_img_path = Path(img_path_str).relative_to(self.input_path)
        rel_label_path = label_path.relative_to(self.output_path)
        
        return str(rel_img_path), str(rel_label_path)
    
    def split_dataset(self, all_samples: List[Tuple[str, str]], train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, List[Tuple[str, str]]]:
        """
        划分数据集为训练集、验证集和测试集
        
        Args:
            all_samples: 所有样本列表 [(img_path, label_path), ...]
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            {'train': [...], 'val': [...], 'test': [...]}
        """
        # 打乱样本
        shuffled_samples = all_samples[:]
        random.shuffle(shuffled_samples)
        
        total_count = len(shuffled_samples)
        train_count = int(total_count * train_ratio)
        val_count = int(total_count * val_ratio)
        
        train_samples = shuffled_samples[:train_count]
        val_samples = shuffled_samples[train_count:train_count + val_count]
        test_samples = shuffled_samples[train_count + val_count:]
        
        return {
            'train': train_samples,
            'val': val_samples,
            'test': test_samples
        }
    
    def save_class_mapping(self):
        """保存类别映射文件"""
        classes_file = self.output_path / 'classes.txt'
        with open(classes_file, 'w', encoding='utf-8') as f:
            for class_id in sorted(self.reverse_class_mapping.keys()):
                class_name = self.reverse_class_mapping[class_id]
                f.write(f"{class_id} {class_name}\n")
    
    def save_data_config(self, dataset_split: Dict[str, List[Tuple[str, str]]]):
        """保存数据集配置文件"""
        # 生成数据集配置 - 符合训练脚本期望的路径格式
        data_config = {
            'path': str(self.output_path.absolute()),  # 数据集根目录
            'train': 'train',  # 相对于path的路径，直接指向train目录
            'val': 'val',
            'test': 'test' if dataset_split.get('test') else 'val',
            'nc': len(self.reverse_class_mapping),
            'names': [self.reverse_class_mapping[i] for i in sorted(self.reverse_class_mapping.keys())]
        }
        
        # 保存配置文件
        config_path = self.output_path / 'data.yaml'
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data_config, f, default_flow_style=False, allow_unicode=True)
    
    def copy_and_organize_files(self, dataset_split: Dict[str, List[Tuple[str, str]]]):
        """复制并组织文件到对应目录 - 采用训练脚本期望的结构"""
        import shutil
        
        # 为每个分割创建包含images和labels的子目录
        # 结构: dataset_path/train/images, dataset_path/train/labels, 等等
        for split_name, samples in dataset_split.items():
            split_img_dir = self.output_path / split_name / 'images'
            split_lbl_dir = self.output_path / split_name / 'labels'
            
            split_img_dir.mkdir(parents=True, exist_ok=True)
            split_lbl_dir.mkdir(exist_ok=True)
            
            # 对于当前split，复制其所有图片和标签
            for img_rel_path, lbl_rel_path in samples:
                # 复制图片
                src_img_path = self.input_path / img_rel_path
                dst_img_name = Path(img_rel_path).name
                dst_img_path = split_img_dir / dst_img_name
                
                # 复制图片文件
                shutil.copy2(src_img_path, dst_img_path)
                
                # 复制标签
                src_lbl_path = self.output_path / lbl_rel_path
                dst_lbl_name = Path(lbl_rel_path).name
                dst_lbl_path = split_lbl_dir / dst_lbl_name
                
                # 确保标签文件已存在后再复制
                if src_lbl_path.exists():
                    shutil.copy2(src_lbl_path, dst_lbl_path)
    
    def format_dataset(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        num_workers: int = 8,
        ignore_classes: List[str] = None
    ):
        """
        格式化整个数据集为YOLO格式
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            num_workers: 并行处理的线程数
            ignore_classes: 要忽略的类别名称列表
        """
        # 更新忽略类别设置
        if ignore_classes is not None:
            self.ignore_classes = set(ignore_classes)
        
        print("=" * 60)
        print("YOLO格式转换（多线程并行）")
        print("=" * 60)
        print(f"并行线程数: {num_workers}")
        print(f"标注格式: YOLO Polygon (保留轮廓信息)")
        print(f"训练集比例: {train_ratio}")
        print(f"验证集比例: {val_ratio}")
        print(f"测试集比例: {1 - train_ratio - val_ratio}")
        if self.ignore_classes:
            print(f"忽略类别: {', '.join(sorted(self.ignore_classes))}")
        print()
        
        # 创建根目录
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载数据集结构
        dataset_structure = self.load_dataset_structure()
        
        print(f"输入数据集: {self.input_path}")
        print(f"输出数据集: {self.output_path}")
        print(f"发现 {len(dataset_structure)} 个类别")
        print()
        
        # 重置统计信息
        self.stats = {
            'total_images': 0,
            'converted_images': 0,
            'skipped_images': 0,
            'total_annotations': 0,
            'class_distribution': defaultdict(int)
        }
        
        # 收集所有样本
        all_samples = []
        for category, samples in dataset_structure.items():
            self.stats['total_images'] += len(samples)
            for img_path, json_path in samples:
                all_samples.append((img_path, json_path))
        
        print(f"总计待处理图片: {len(all_samples)}")
        print()
        
        # 创建标签输出目录
        output_labels_dir = self.output_path / 'labels_temp'
        output_labels_dir.mkdir(exist_ok=True)
        
        # 使用线程池处理所有样本
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for img_path, json_path in all_samples:
                future = executor.submit(
                    self._process_single_sample,
                    img_path, json_path, output_labels_dir
                )
                futures.append(future)
            
            # 收集结果
            processed_samples = []
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    if result is not None:
                        processed_samples.append(result)
                        self.stats['converted_images'] += 1
                    else:
                        self.stats['skipped_images'] += 1
                except Exception as e:
                    print(f"  处理失败: {all_samples[i][0]} - {str(e)}")
                    self.stats['skipped_images'] += 1
        
        print(f"转换完成: {self.stats['converted_images']} 个图片，跳过: {self.stats['skipped_images']} 个")
        print()
        
        # 划分数据集
        print("划分数据集...")
        dataset_split = self.split_dataset(processed_samples, train_ratio, val_ratio)
        
        for split_name, samples in dataset_split.items():
            print(f"  {split_name}: {len(samples)} 个样本")
        
        print()
        
        # 组织文件
        print("组织文件...")
        self.copy_and_organize_files(dataset_split)
        print("文件组织完成")
        print()
        
        # 保存类别映射
        print("保存类别映射...")
        self.save_class_mapping()
        print("类别映射保存完成")
        print()
        
        # 保存数据集配置
        print("生成数据集配置...")
        self.save_data_config(dataset_split)
        print("数据集配置生成完成")
        print()
        
        # 清理临时标签目录
        import shutil
        shutil.rmtree(output_labels_dir, ignore_errors=True)
        
        # 打印统计报告
        self._print_report()
    
    def _print_report(self):
        """打印转换报告"""
        print("\n" + "=" * 60)
        print("YOLO格式转换报告")
        print("=" * 60)
        
        print(f"\n【总体统计】")
        print(f"  原始图片数: {self.stats['total_images']}")
        print(f"  成功转换: {self.stats['converted_images']}")
        print(f"  跳过图片: {self.stats['skipped_images']}")
        print(f"  总标注数: {self.stats['total_annotations']}")
        
        print(f"\n【类别分布】")
        for class_name, count in sorted(self.stats['class_distribution'].items(), key=lambda x: -x[1]):
            class_id = self.class_mapping[class_name]
            print(f"  {class_id} - {class_name}: {count}")
        
        print(f"\n【输出目录结构】")
        print(f"  {self.output_path}/")
        print(f"  ├── classes.txt          # 类别映射文件")
        print(f"  ├── data.yaml            # YOLO数据集配置文件")
        print(f"  ├── train/")
        print(f"  │   ├── images/          # 训练集图片")
        print(f"  │   └── labels/          # 训练集标签")
        print(f"  ├── val/")
        print(f"  │   ├── images/          # 验证集图片")
        print(f"  │   └── labels/          # 验证集标签")
        print(f"  └── test/")
        print(f"      ├── images/          # 测试集图片")
        print(f"      └── labels/          # 测试集标签")
        
        print("\n" + "=" * 60)