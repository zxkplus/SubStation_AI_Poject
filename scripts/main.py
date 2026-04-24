"""
主入口脚本
整合数据加载、统计、可视化和YOLO格式转换功能
"""

import sys
import os
import argparse

# 添加scripts目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import DatasetLoader
from statistics import DatasetStats
from visualization import MaskVisualizer
from yolo_converter import DatasetConverter


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='变电站设备分割数据集处理工具集'
    )
    
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='数据集根目录路径'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['stats', 'visualize', 'full', 'yolo', 'convert'],
        default='full',
        help='运行模式: stats(统计), visualize(可视化), full(统计+可视化), yolo(YOLO格式转换), convert(数据集转换，图片和标注文件在同一目录)'
    )
    
    parser.add_argument(
        '--samples_per_class',
        type=int,
        default=2,
        help='每个类别随机选择的样本数量（用于可视化或采样转换），convert模式默认100'
    )
    
    parser.add_argument(
        '--output_report',
        type=str,
        default=None,
        help='统计报告输出路径（可选）'
    )
    
    parser.add_argument(
        '--output_visualization',
        type=str,
        default="./visualization_output",
        help='可视化图片输出目录'
    )
    
    # 数据集转换专用参数
    parser.add_argument(
        '--output_yolo_path',
        type=str,
        default=None,
        help='数据集转换输出路径（用于yolo/convert模式）'
    )
    
    parser.add_argument(
        '--expand_ratio',
        type=float,
        default=0.0,
        help='裁剪边界框扩展比例（0-1），增加裁剪区域'
    )
    
    parser.add_argument(
        '--min_size',
        type=int,
        default=32,
        help='最小裁剪尺寸，低于此尺寸的目标将被跳过'
    )
    
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='并行处理的线程数（用于yolo模式）'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("变电站设备分割数据集处理工具集")
    print("=" * 60)
    print(f"数据集路径: {args.dataset_path}")
    print(f"运行模式: {args.mode}")
    if args.mode in ['yolo', 'convert']:
        print(f"输出路径: {args.output_yolo_path or '默认路径'}")
    print()
    
    try:
        if args.mode in ['yolo', 'convert']:
            # 数据集转换模式
            from yolo_converter import DatasetConverter
            
            if not args.output_yolo_path:
                # 默认输出到输入目录的同级converted_dataset文件夹
                parent_dir = os.path.dirname(os.path.abspath(args.dataset_path))
                args.output_yolo_path = os.path.join(parent_dir, 'converted_dataset')
            
            converter = DatasetConverter(args.dataset_path, args.output_yolo_path)
            
            # 根据模式设置参数
            if args.mode == 'convert':
                # convert模式：不创建子目录，每个类别转换100张
                converter.convert_dataset(
                    samples_per_class=100,
                    expand_ratio=args.expand_ratio,
                    min_size=args.min_size,
                    output_images_subdir="",  # 不创建images子目录
                    output_labels_subdir="",  # 不创建labels子目录
                    num_workers=args.num_workers
                )
            else:
                # yolo模式：创建子目录，使用指定的samples_per_class
                converter.convert_dataset(
                    samples_per_class=args.samples_per_class,
                    expand_ratio=args.expand_ratio,
                    min_size=args.min_size,
                    num_workers=args.num_workers
                )
        
        else:
            # 统计/可视化模式
            print(f"每类采样数: {args.samples_per_class}")
            print()
            
            # 1. 加载数据集
            print("【步骤1】加载数据集...")
            loader = DatasetLoader(args.dataset_path)
            loader.load_dataset()
            
            # 验证数据集
            is_valid, errors = loader.validate_dataset()
            if not is_valid:
                print("数据集验证失败:")
                for error in errors:
                    print(f"  - {error}")
                return 1
            
            print(f"✓ 数据集加载成功")
            print(f"  - 发现 {len(loader.get_categories())} 个类别")
            print(f"  - 总共 {loader.get_total_samples()} 个样本")
            print()
            
            # 初始化统计器（用于后续随机采样）
            stats = DatasetStats(loader)
            
            # 2. 统计分析
            if args.mode in ['stats', 'full']:
                print("【步骤2】统计分析...")
                stats.calculate_stats()
                
                # 打印报告
                report = stats.generate_report()
                print(report)
                
                # 保存报告
                if args.output_report:
                    stats.save_report(args.output_report)
                print("✓ 统计分析完成")
                print()
            
            # 3. 可视化
            if args.mode in ['visualize', 'full']:
                print("【步骤3】可视化标注...")
                
                # 随机选择样本
                samples_dict = stats.get_samples_per_category(args.samples_per_class)
                
                print(f"✓ 选择了 {len(samples_dict)} 个类别的样本进行可视化")
                print(f"✓ 正在生成可视化图片到: {args.output_visualization}")
                print()
                
                # 创建可视化器并展示
                visualizer = MaskVisualizer(alpha=0.5)
                visualizer.show_interactive(samples_dict, args.output_visualization)
                
                print("✓ 可视化完成")
            
            print()
            print("=" * 60)
            print("所有任务完成!")
            print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
