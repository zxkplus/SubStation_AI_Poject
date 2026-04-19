"""
统计模块
负责计算数据集的统计信息，生成分析报告
"""

import json
from typing import Dict, List, Tuple
from data_loader import DatasetLoader


class DatasetStats:
    """数据集统计器"""
    
    def __init__(self, data_loader: DatasetLoader):
        """
        初始化统计器
        
        Args:
            data_loader: 数据加载器实例
        """
        self.loader = data_loader
        self.stats = {}
    
    def calculate_stats(self) -> Dict:
        """
        计算数据集统计信息
        
        Returns:
            统计信息字典
        """
        # 基础统计
        self.stats['total_samples'] = self.loader.get_total_samples()
        self.stats['total_categories'] = len(self.loader.get_categories())
        self.stats['category_stats'] = self.loader.get_category_stats()
        
        # 计算每个类别的占比
        for category, count in self.stats['category_stats'].items():
            ratio = count / self.stats['total_samples'] if self.stats['total_samples'] > 0 else 0
            self.stats['category_stats'][category] = {
                'count': count,
                'ratio': ratio
            }
        
        return self.stats
    
    def generate_report(self) -> str:
        """
        生成统计报告
        
        Returns:
            格式化的统计报告字符串
        """
        if not self.stats:
            self.calculate_stats()
        
        lines = []
        lines.append("=" * 60)
        lines.append("变电站设备分割数据集统计报告")
        lines.append("=" * 60)
        lines.append("")
        
        # 总体统计
        lines.append("【总体统计】")
        lines.append(f"  类别数量: {self.stats['total_categories']}")
        lines.append(f"  样本总数: {self.stats['total_samples']}")
        lines.append("")
        
        # 类别分布
        lines.append("【类别分布】")
        lines.append(f"  {'类别':<20} {'数量':>8} {'占比':>10}")
        lines.append("  " + "-" * 40)
        
        # 按数量排序
        sorted_categories = sorted(
            self.stats['category_stats'].items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )
        
        for category, info in sorted_categories:
            count = info['count']
            ratio = info['ratio'] * 100
            lines.append(f"  {category:<20} {count:>8} {ratio:>9.2f}%")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def get_samples_per_category(self, num_samples: int) -> Dict[str, List[Tuple[str, str]]]:
        """
        从每个类别随机获取指定数量的样本
        
        Args:
            num_samples: 每个类别要获取的样本数量
            
        Returns:
            {类别名: [(img_path, json_path), ...]}
        """
        import random
        
        samples_dict = {}
        for category in self.loader.get_categories():
            all_samples = self.loader.get_category_samples(category)
            
            if len(all_samples) <= num_samples:
                selected = all_samples
            else:
                selected = random.sample(all_samples, num_samples)
            
            samples_dict[category] = selected
        
        return samples_dict
    
    def save_report(self, output_path: str):
        """
        将统计报告保存到文件
        
        Args:
            output_path: 输出文件路径
        """
        report = self.generate_report()
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"统计报告已保存到: {output_path}")
