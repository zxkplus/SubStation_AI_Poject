"""
数据加载器模块
负责遍历数据集目录结构，加载图片和对应的JSON标注文件
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class DatasetLoader:
    """变电站设备分割数据集加载器"""
    
    def __init__(self, dataset_path: str):
        """
        初始化数据集加载器
        
        Args:
            dataset_path: 数据集根目录路径
        """
        self.dataset_path = Path(dataset_path)
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"数据集目录不存在: {dataset_path}")
        
        self.dataset_structure: Dict[str, List[Tuple[str, str]]] = {}
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    def load_dataset(self) -> Dict[str, List[Tuple[str, str]]]:
        """
        加载数据集，遍历目录结构
        
        Returns:
            字典结构: {类别名: [(图片路径, json路径), ...]}
        """
        self.dataset_structure.clear()
        
        # 遍历第一层目录（类别）
        for category_dir in self.dataset_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            category_name = category_dir.name
            self.dataset_structure[category_name] = []
            
            # 遍历第二层目录（图片和JSON文件）
            image_files = []
            
            for file_path in category_dir.iterdir():
                if file_path.suffix.lower() in self.supported_image_formats:
                    image_files.append(file_path)
            
            # 为每个图片查找对应的JSON文件
            for img_path in image_files:
                json_path = img_path.with_suffix('.json')
                
                if json_path.exists():
                    self.dataset_structure[category_name].append(
                        (str(img_path), str(json_path))
                    )
                else:
                    print(f"警告: 图片 {img_path.name} 缺少对应的JSON标注文件")
        
        return self.dataset_structure
    
    def get_categories(self) -> List[str]:
        """
        获取所有类别名称
        
        Returns:
            类别名称列表
        """
        return list(self.dataset_structure.keys())
    
    def get_category_samples(self, category: str) -> List[Tuple[str, str]]:
        """
        获取指定类别的所有样本
        
        Args:
            category: 类别名称
            
        Returns:
            该类别的样本列表 [(img_path, json_path), ...]
        """
        return self.dataset_structure.get(category, [])
    
    def get_total_samples(self) -> int:
        """
        获取数据集总样本数
        
        Returns:
            所有类别的样本总数
        """
        return sum(len(samples) for samples in self.dataset_structure.values())
    
    def get_category_stats(self) -> Dict[str, int]:
        """
        获取每个类别的样本数量统计
        
        Returns:
            {类别名: 样本数量}
        """
        return {
            category: len(samples)
            for category, samples in self.dataset_structure.items()
        }
    
    def load_json_annotation(self, json_path: str) -> dict:
        """
        加载JSON标注文件
        
        Args:
            json_path: JSON文件路径
            
        Returns:
            标注数据字典
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"加载JSON文件失败 {json_path}: {str(e)}")
            return {}
    
    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """
        验证数据集的完整性
        
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        if not self.dataset_structure:
            errors.append("数据集为空或目录结构不正确")
            return False, errors
        
        for category, samples in self.dataset_structure.items():
            if not samples:
                errors.append(f"类别 '{category}' 没有有效的样本")
        
        return len(errors) == 0, errors
