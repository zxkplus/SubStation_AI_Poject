"""
YOLO训练统一入口脚本
支持多个YOLO版本：YOLOv6, YOLO26等
"""

import sys
import argparse
import logging
import yaml
from pathlib import Path
from typing import Dict, Any

# 添加scripts目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from trainers.base_trainer import BaseTrainer
from trainers.yolov6_trainer import YOLOv6Trainer
from trainers.yolov26_trainer import YOLO26Trainer
from trainers.yolov8_trainer import YOLOv8Trainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 支持的YOLO版本映射
TRAINER_REGISTRY = {
    'yolov6': YOLOv6Trainer,
    'yolo26': YOLO26Trainer,
    'yolov8': YOLOv8Trainer,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_data_yaml(
    dataset_path: str,
    output_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2
):
    """
    自动生成data.yaml配置文件

    Args:
        dataset_path: YOLO格式数据集路径
        output_path: 输出配置文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    dataset_path = Path(dataset_path)
    classes_file = dataset_path / 'classes.txt'

    # 读取类别信息
    if not classes_file.exists():
        raise FileNotFoundError(f"找不到classes.txt文件: {classes_file}")

    with open(classes_file, 'r', encoding='utf-8') as f:
        classes = [line.strip().split(' ', 1)[1] for line in f if line.strip()]

    # 构建配置
    config = {
        'path': str(dataset_path.absolute()),
        'train': 'train',  # 训练集会由prepare_dataset函数创建
        'val': 'val',      # 验证集会由prepare_dataset函数创建
        'nc': len(classes),
        'names': {i: name for i, name in enumerate(classes)},
        'img_size': 640,
        'epochs': 300,
        'batch_size': 32,
        'workers': 8,
    }

    # 保存配置
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"数据集配置文件已生成: {output_path}")
    logger.info(f"类别数量: {len(classes)}")
    logger.info(f"类别列表: {classes}")

    return output_path


def prepare_dataset(
    dataset_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2
):
    """
    准备训练数据集，划分训练集和验证集

    Args:
        dataset_path: YOLO格式数据集路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    import shutil
    import random

    dataset_path = Path(dataset_path)
    train_dir = dataset_path / 'train'
    val_dir = dataset_path / 'val'

    # 清理并创建目录
    for dir_path in [train_dir, val_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True)
        (dir_path / 'images').mkdir()
        (dir_path / 'labels').mkdir()

    # 获取所有类别的图片
    all_images = []
    for category_dir in dataset_path.iterdir():
        if category_dir.is_dir() and category_dir.name not in ['train', 'val', 'test']:
            images_dir = category_dir / 'images'
            labels_dir = category_dir / 'labels'

            if images_dir.exists() and labels_dir.exists():
                for img_file in images_dir.glob('*.jpg'):
                    label_file = labels_dir / (img_file.stem + '.txt')
                    if label_file.exists():
                        all_images.append({
                            'img_path': img_file,
                            'label_path': label_file,
                            'category': category_dir.name
                        })

    if not all_images:
        raise ValueError(f"数据集中没有找到有效的图片和标注: {dataset_path}")

    logger.info(f"总共找到 {len(all_images)} 张图片")

    # 随机打乱并划分
    random.shuffle(all_images)
    split_idx = int(len(all_images) * train_ratio)

    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]

    logger.info(f"训练集: {len(train_images)} 张, 验证集: {len(val_images)} 张")

    # 复制文件到训练集和验证集
    def copy_images(images, target_dir):
        for item in images:
            # 复制图片
            shutil.copy2(item['img_path'], target_dir / 'images' / item['img_path'].name)
            # 复制标注
            shutil.copy2(item['label_path'], target_dir / 'labels' / item['label_path'].name)

    copy_images(train_images, train_dir)
    copy_images(val_images, val_dir)

    logger.info("数据集划分完成")


def main():
    parser = argparse.ArgumentParser(description='YOLO训练脚本')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'val', 'test', 'export'],
                        help='运行模式')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='YOLO格式数据集路径')
    parser.add_argument('--output_dir', type=str, default='./runs/train',
                        help='输出目录')
    parser.add_argument('--yolo_version', type=str, default='yolov6',
                        choices=['yolov6', 'yolo26', 'yolov8'],
                        help='YOLO版本')
    parser.add_argument('--data_config', type=str, default=None,
                        help='数据集配置文件路径（如果未提供将自动生成）')
    parser.add_argument('--model_config', type=str, default=None,
                        help='模型配置文件路径')
    parser.add_argument('--weights', type=str, default=None,
                        help='模型权重路径（用于验证/测试/导出）')
    parser.add_argument('--epochs', type=int, default=300,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--img_size', type=int, default=640,
                        help='输入图片尺寸')
    parser.add_argument('--device', type=str, default='0',
                        help='设备ID，0表示GPU0，-1表示CPU')
    parser.add_argument('--workers', type=int, default=8,
                        help='数据加载线程数')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='训练集比例')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的checkpoint路径')
    parser.add_argument('--export_format', type=str, default='onnx',
                        choices=['onnx', 'torchscript', 'engine', 'coreml', 'tflite', 'pb'],
                        help='导出格式')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称')
    parser.add_argument('--model_size', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO26模型尺寸（仅适用于yolo26）')

    args = parser.parse_args()

    try:
        # 检查YOLO版本
        if args.yolo_version not in TRAINER_REGISTRY:
            raise ValueError(f"不支持的YOLO版本: {args.yolo_version}")

        # 准备数据集配置
        if args.data_config is None:
            args.data_config = str(
                Path(args.output_dir) / 'data.yaml'
            )

        # 如果data.yaml不存在，自动生成
        if not Path(args.data_config).exists():
            logger.info("数据集配置文件不存在，正在准备数据集...")
            prepare_dataset(args.dataset_path, args.train_ratio, 1 - args.train_ratio)
            generate_data_yaml(args.dataset_path, args.data_config, args.train_ratio, 1 - args.train_ratio)

        # 加载模型配置（如果有）
        model_config = None
        if args.model_config:
            model_config = load_config(args.model_config)

        # 创建训练器
        trainer_class = TRAINER_REGISTRY[args.yolo_version]
        trainer_kwargs = {
            'data_config_path': args.data_config,
            'output_dir': args.output_dir,
            'model_config': model_config,
            'device': args.device
        }

        # YOLO26需要额外的model_size参数
        if args.yolo_version == 'yolo26':
            trainer_kwargs['model_size'] = args.model_size

        trainer = trainer_class(**trainer_kwargs)

        # 根据模式执行对应操作
        if args.mode == 'train':
            trainer.train(
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=args.img_size,
                resume=args.resume,
                workers=args.workers,
                name=args.name
            )
        elif args.mode == 'val':
            if args.weights is None:
                raise ValueError("验证模式需要指定--weights参数")
            trainer.validate(
                weights_path=args.weights,
                batch_size=args.batch_size,
                img_size=args.img_size
            )
        elif args.mode == 'test':
            if args.weights is None:
                raise ValueError("测试模式需要指定--weights参数")
            trainer.test(
                weights_path=args.weights,
                batch_size=args.batch_size,
                img_size=args.img_size
            )
        elif args.mode == 'export':
            if args.weights is None:
                raise ValueError("导出模式需要指定--weights参数")
            trainer.export(
                weights_path=args.weights,
                format=args.export_format,
                img_size=args.img_size
            )

    except Exception as e:
        logger.error(f"训练失败: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
