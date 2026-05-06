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
    val_ratio: float = 0.2,
    ignore_classes: list = None
):
    """
    自动生成data.yaml配置文件

    Args:
        dataset_path: YOLO格式数据集路径
        output_path: 输出配置文件路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        ignore_classes: 要忽略的类别列表
    """
    if ignore_classes is None:
        ignore_classes = []
    
    dataset_path = Path(dataset_path)
    classes_file = dataset_path / 'classes.txt'

    # 读取类别信息
    if not classes_file.exists():
        raise FileNotFoundError(f"找不到classes.txt文件: {classes_file}")

    with open(classes_file, 'r', encoding='utf-8') as f:
        original_classes = [line.strip().split(' ', 1)[1] for line in f if line.strip()]
    
    # 过滤要忽略的类别
    filtered_classes = [c for c in original_classes if c not in ignore_classes]
    
    if len(filtered_classes) == 0:
        raise ValueError(f"所有类别都被忽略！原类别: {original_classes}, 忽略: {ignore_classes}")
    
    if len(filtered_classes) != len(original_classes):
        logger.info(f"已过滤类别: {set(original_classes) - set(filtered_classes)}")
        logger.info(f"剩余类别: {filtered_classes}")

    # 检查是否存在train/val/test目录结构
    train_dir = dataset_path / 'train'
    val_dir = dataset_path / 'val'
    test_dir = dataset_path / 'test'
    
    has_standard_structure = train_dir.exists() and val_dir.exists()

    # 构建配置
    config = {
        'path': str(dataset_path.absolute()),
        'nc': len(filtered_classes),
        'names': {i: name for i, name in enumerate(filtered_classes)},
        'img_size': 640,
        'epochs': 300,
        'batch_size': 32,
        'workers': 8,
    }
    
    # 根据数据集结构设置路径
    if has_standard_structure:
        config['train'] = 'train'  # 相对于path的路径
        config['val'] = 'val'
        if test_dir.exists():
            config['test'] = 'test'
    else:
        # 旧格式，按类别组织
        config['train'] = 'train'  # 训练集会由prepare_dataset函数创建
        config['val'] = 'val'      # 验证集会由prepare_dataset函数创建

    # 保存配置
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"数据集配置文件已生成: {output_path}")
    logger.info(f"类别数量: {len(filtered_classes)}")
    logger.info(f"类别列表: {filtered_classes}")

    return output_path


def filter_label_file(label_path, class_mapping, ignore_classes):
    """
    过滤标注文件，移除指定的类别
    
    Args:
        label_path: 标注文件路径
        class_mapping: 类别名称到ID的映射字典
        ignore_classes: 要忽略的类别名称列表
        
    Returns:
        bool: 如果文件中有有效标注返回True，否则返回False
    """
    if not label_path.exists():
        return False
    
    # 创建反向映射：ID到类别名称
    id_to_class = {v: k for k, v in class_mapping.items()}
    
    # 读取原始标注
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 过滤掉要忽略的类别
    filtered_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) < 5:
            continue
            
        try:
            class_id = int(parts[0])
            class_name = id_to_class.get(class_id)
            if class_name is None or class_name not in ignore_classes:
                filtered_lines.append(line)
        except (ValueError, KeyError):
            # 如果无法解析类别ID，保留原行（可能是格式问题）
            filtered_lines.append(line)
    
    # 如果没有有效标注，删除文件并返回False
    if not filtered_lines:
        label_path.unlink(missing_ok=True)
        return False
    
    # 写回过滤后的标注
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(filtered_lines) + '\n')
    
    return True


def prepare_dataset(
    dataset_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    ignore_classes: list = None
):
    """
    准备训练数据集，划分训练集和验证集
    支持三种数据集格式：
    1. 按类别组织的格式：dataset/class_name/images/, dataset/class_name/labels/
    2. 标准YOLO格式：dataset/images/train/, dataset/labels/train/
    3. 由yolo_formatter.py生成的格式：dataset/train/images/, dataset/val/images/, dataset/test/images/

    Args:
        dataset_path: YOLO格式数据集路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        ignore_classes: 要忽略的类别列表
    """
    if ignore_classes is None:
        ignore_classes = []
        
    import shutil
    import random

    dataset_path = Path(dataset_path)
    
    # 读取类别映射（用于过滤标注）
    class_mapping = {}
    classes_file = dataset_path / 'classes.txt'
    if classes_file.exists():
        with open(classes_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        class_id, class_name = parts
                        class_mapping[class_name] = int(class_id)
    
    # 检查是否已经是标准YOLO格式（存在images/train和images/val目录）
    standard_train_img_dir = dataset_path / 'images' / 'train'
    standard_val_img_dir = dataset_path / 'images' / 'val'
    standard_test_img_dir = dataset_path / 'images' / 'test'
    
    if standard_train_img_dir.exists() and standard_val_img_dir.exists():
        logger.info("检测到标准YOLO格式数据集，正在复制到正确位置...")
        
        # 如果数据已经在正确位置，直接返回
        actual_train_img_dir = dataset_path / 'train' / 'images'
        actual_val_img_dir = dataset_path / 'val' / 'images'
        
        if actual_train_img_dir.exists() and actual_val_img_dir.exists():
            logger.info("数据已在正确位置，无需移动")
            return
        
        # 创建标准目录结构
        train_img_dir = dataset_path / 'train' / 'images'
        train_lbl_dir = dataset_path / 'train' / 'labels'
        val_img_dir = dataset_path / 'val' / 'images'
        val_lbl_dir = dataset_path / 'val' / 'labels'
        test_img_dir = dataset_path / 'test' / 'images'
        test_lbl_dir = dataset_path / 'test' / 'labels'
        
        # 创建目录
        train_img_dir.mkdir(parents=True, exist_ok=True)
        train_lbl_dir.mkdir(parents=True, exist_ok=True)
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_lbl_dir.mkdir(parents=True, exist_ok=True)
        test_img_dir.mkdir(parents=True, exist_ok=True)
        test_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查images和labels目录是否存在
        images_train_dir = dataset_path / 'images' / 'train'
        images_val_dir = dataset_path / 'images' / 'val'
        labels_train_dir = dataset_path / 'labels' / 'train'
        labels_val_dir = dataset_path / 'labels' / 'val'
        
        # 复制训练集并过滤标注
        valid_train_files = 0
        for img_file in images_train_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                label_file = labels_train_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    # 复制图片
                    shutil.copy2(img_file, train_img_dir / img_file.name)
                    # 复制并过滤标注
                    temp_label_path = train_lbl_dir / label_file.name
                    shutil.copy2(label_file, temp_label_path)
                    if filter_label_file(temp_label_path, class_mapping, ignore_classes):
                        valid_train_files += 1
                    else:
                        # 如果过滤后没有有效标注，删除对应的图片
                        (train_img_dir / img_file.name).unlink(missing_ok=True)
        
        # 复制验证集并过滤标注
        valid_val_files = 0
        for img_file in images_val_dir.glob('*'):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                label_file = labels_val_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    # 复制图片
                    shutil.copy2(img_file, val_img_dir / img_file.name)
                    # 复制并过滤标注
                    temp_label_path = val_lbl_dir / label_file.name
                    shutil.copy2(label_file, temp_label_path)
                    if filter_label_file(temp_label_path, class_mapping, ignore_classes):
                        valid_val_files += 1
                    else:
                        # 如果过滤后没有有效标注，删除对应的图片
                        (val_img_dir / img_file.name).unlink(missing_ok=True)
        
        # 复制测试集（如果存在）并过滤标注
        images_test_dir = dataset_path / 'images' / 'test'
        labels_test_dir = dataset_path / 'labels' / 'test'
        if images_test_dir.exists() and labels_test_dir.exists():
            valid_test_files = 0
            for img_file in images_test_dir.glob('*'):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    label_file = labels_test_dir / (img_file.stem + '.txt')
                    if label_file.exists():
                        # 复制图片
                        shutil.copy2(img_file, test_img_dir / img_file.name)
                        # 复制并过滤标注
                        temp_label_path = test_lbl_dir / label_file.name
                        shutil.copy2(label_file, temp_label_path)
                        if filter_label_file(temp_label_path, class_mapping, ignore_classes):
                            valid_test_files += 1
                        else:
                            # 如果过滤后没有有效标注，删除对应的图片
                            (test_img_dir / img_file.name).unlink(missing_ok=True)
            logger.info(f"测试集过滤完成: {valid_test_files} 张有效图片")
        
        logger.info(f"数据集类别过滤完成，训练集: {valid_train_files} 张, 验证集: {valid_val_files} 张")
        return

    # 检查是否有由yolo_formatter.py生成的数据集结构（train/val/test目录分别包含images和labels）
    train_img_dir = dataset_path / 'train' / 'images'
    train_lbl_dir = dataset_path / 'train' / 'labels'
    val_img_dir = dataset_path / 'val' / 'images'
    val_lbl_dir = dataset_path / 'val' / 'labels'
    test_img_dir = dataset_path / 'test' / 'images'
    test_lbl_dir = dataset_path / 'test' / 'labels'

    # 如果已经存在train/val/test结构，还需要检查是否有实际的图像文件
    if (train_img_dir.exists() and train_lbl_dir.exists() and 
        val_img_dir.exists() and val_lbl_dir.exists()):
        # 额外检查目录中是否有图像文件
        train_img_count = len(list(train_img_dir.glob('*.[jJ][pP][gG]')) + 
                              list(train_img_dir.glob('*.[pP][nN][gG]')) + 
                              list(train_img_dir.glob('*.[jJ][pP][eE][gG]')) + 
                              list(train_img_dir.glob('*.[bB][mM][pP]')))
        val_img_count = len(list(val_img_dir.glob('*.[jJ][pP][gG]')) + 
                            list(val_img_dir.glob('*.[pP][nN][gG]')) + 
                            list(val_img_dir.glob('*.[jJ][pP][eE][gG]')) + 
                            list(val_img_dir.glob('*.[bB][mM][pP]')))
        
        if train_img_count > 0 and val_img_count > 0:
            logger.info("检测到由yolo_formatter.py生成的数据集格式，正在过滤指定类别...")
            
            # 过滤训练集标注
            valid_train_files = 0
            for label_file in train_lbl_dir.glob('*.txt'):
                img_file = train_img_dir / (label_file.stem + label_file.suffix.replace('.txt', '.jpg'))
                if not img_file.exists():
                    img_file = train_img_dir / (label_file.stem + '.png')
                if not img_file.exists():
                    img_file = train_img_dir / (label_file.stem + '.jpeg')
                if not img_file.exists():
                    img_file = train_img_dir / (label_file.stem + '.bmp')
                
                if filter_label_file(label_file, class_mapping, ignore_classes):
                    valid_train_files += 1
                else:
                    # 如果过滤后没有有效标注，删除对应的图片
                    img_file.unlink(missing_ok=True)
                    label_file.unlink(missing_ok=True)
            
            # 过滤验证集标注
            valid_val_files = 0
            for label_file in val_lbl_dir.glob('*.txt'):
                img_file = val_img_dir / (label_file.stem + label_file.suffix.replace('.txt', '.jpg'))
                if not img_file.exists():
                    img_file = val_img_dir / (label_file.stem + '.png')
                if not img_file.exists():
                    img_file = val_img_dir / (label_file.stem + '.jpeg')
                if not img_file.exists():
                    img_file = val_img_dir / (label_file.stem + '.bmp')
                
                if filter_label_file(label_file, class_mapping, ignore_classes):
                    valid_val_files += 1
                else:
                    # 如果过滤后没有有效标注，删除对应的图片
                    img_file.unlink(missing_ok=True)
                    label_file.unlink(missing_ok=True)
            
            # 过滤测试集标注（如果存在）
            if test_img_dir.exists() and test_lbl_dir.exists():
                valid_test_files = 0
                for label_file in test_lbl_dir.glob('*.txt'):
                    img_file = test_img_dir / (label_file.stem + label_file.suffix.replace('.txt', '.jpg'))
                    if not img_file.exists():
                        img_file = test_img_dir / (label_file.stem + '.png')
                    if not img_file.exists():
                        img_file = test_img_dir / (label_file.stem + '.jpeg')
                    if not img_file.exists():
                        img_file = test_img_dir / (label_file.stem + '.bmp')
                    
                    if filter_label_file(label_file, class_mapping, ignore_classes):
                        valid_test_files += 1
                    else:
                        # 如果过滤后没有有效标注，删除对应的图片
                        img_file.unlink(missing_ok=True)
                        label_file.unlink(missing_ok=True)
                logger.info(f"测试集过滤完成: {valid_test_files} 张有效图片")
            
            logger.info(f"数据集类别过滤完成，训练集: {valid_train_files} 张, 验证集: {valid_val_files} 张")
            return
        else:
            logger.error(f"检测到数据集目录结构，但缺少图像文件: train图片数={train_img_count}, val图片数={val_img_count}")
            raise FileNotFoundError(f"数据集中没有找到有效的图片文件: "
                                    f"train目录({train_img_dir})或val目录({val_img_dir})中没有图像文件")

    # 如果不是标准格式，则按照原有逻辑处理按类别组织的格式
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
            # 如果当前类别在忽略列表中，跳过
            if category_dir.name in ignore_classes:
                continue
                
            images_dir = category_dir / 'images'
            labels_dir = category_dir / 'labels'

            if images_dir.exists() and labels_dir.exists():
                for img_file in images_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        label_file = labels_dir / (img_file.stem + '.txt')
                        if label_file.exists():
                            all_images.append({
                                'img_path': img_file,
                                'label_path': label_file,
                                'category': category_dir.name
                            })

    if not all_images:
        # 提供更详细的错误信息，说明期望的数据集格式
        error_msg = (
            f"数据集中没有找到有效的图片和标注: {dataset_path}\n"
            f"请确保您的数据集满足以下条件:\n"
            f"1. 存在标准YOLO格式: {dataset_path}/images/train/ 和 {dataset_path}/images/val/\n"
            f"2. 或者由yolo_formatter.py生成的格式: {dataset_path}/train/images/ 和 {dataset_path}/val/images/\n"
            f"3. 或者按类别组织的格式: \n"
            f"   - {dataset_path}/class_name/images/*.jpg\n"
            f"   - {dataset_path}/class_name/labels/*.txt\n"
            f"   并且jpg图片文件与txt标签文件需一一对应\n"
            f"4. 同时还需要一个classes.txt文件在数据集根目录下"
        )
        raise ValueError(error_msg)

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
    parser.add_argument('--img_size', type=int, default=None,  # 修改默认值为None，以便从配置文件读取
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
    parser.add_argument('--optimizer', type=str, default='auto',
                        choices=['auto', 'SGD', 'Adam', 'AdamW', 'RMSProp'],
                        help='优化器类型')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='初始学习率')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='动量参数')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='权重衰减（L2正则化）')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='预热轮数')
    parser.add_argument('--warmup_momentum', type=float, default=0.8,
                        help='预热阶段动量')
    parser.add_argument('--warmup_bias_lr', type=float, default=0.1,
                        help='预热阶段偏置学习率')
    parser.add_argument('--patience', type=int, default=50,
                        help='早停耐心值')
    parser.add_argument('--ignore_classes', type=str, nargs='*', default=[],
                        help='要忽略的类别名称列表（多个类别用空格分隔）')

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
            prepare_dataset(args.dataset_path, args.train_ratio, 1 - args.train_ratio, args.ignore_classes)
            generate_data_yaml(args.dataset_path, args.data_config, args.train_ratio, 1 - args.train_ratio, args.ignore_classes)

        # 加载模型配置（如果有）
        model_config = None
        if args.model_config:
            model_config = load_config(args.model_config)

        # 从模型配置中提取训练参数（如果命令行未指定）
        if model_config:
            training_config = model_config.get('training', {})
            augmentation_config = model_config.get('augmentation', {})
            
            # 只有当命令行未指定时，才从配置文件获取这些参数
            if args.img_size is None:
                args.img_size = training_config.get('img_size', 640)
            if args.epochs == 300:  # 检查是否使用默认值
                args.epochs = training_config.get('epochs', 300)
            if args.batch_size == 32:  # 检查是否使用默认值
                args.batch_size = training_config.get('batch_size', 32)
            if args.optimizer == 'auto':  # 检查是否使用默认值
                args.optimizer = training_config.get('optimizer', 'auto')
            if args.lr0 == 0.01:  # 检查是否使用默认值
                args.lr0 = training_config.get('lr0', 0.01)
            if args.momentum == 0.937:  # 检查是否使用默认值
                args.momentum = training_config.get('momentum', 0.937)
            if args.weight_decay == 0.0005:  # 检查是否使用默认值
                args.weight_decay = training_config.get('weight_decay', 0.0005)
            if args.warmup_epochs == 3:  # 检查是否使用默认值
                args.warmup_epochs = training_config.get('warmup_epochs', 3)
            if args.warmup_momentum == 0.8:  # 检查是否使用默认值
                args.warmup_momentum = training_config.get('warmup_momentum', 0.8)
            if args.warmup_bias_lr == 0.1:  # 检查是否使用默认值
                args.warmup_bias_lr = training_config.get('warmup_bias_lr', 0.1)
            if args.patience == 50:  # 检查是否使用默认值
                args.patience = training_config.get('patience', 50)

            # 从配置文件获取数据增强参数
            # 将增强参数存储到args中，以便后续传递给训练器
            args.hsv_h = augmentation_config.get('hsv_h', 0.015)
            args.hsv_s = augmentation_config.get('hsv_s', 0.7)
            args.hsv_v = augmentation_config.get('hsv_v', 0.4)
            args.degrees = augmentation_config.get('degrees', 0.0)
            args.translate = augmentation_config.get('translate', 0.1)
            args.scale = augmentation_config.get('scale', 0.5)
            args.shear = augmentation_config.get('shear', 0.0)
            args.perspective = augmentation_config.get('perspective', 0.0)
            args.flipud = augmentation_config.get('flipud', 0.0)
            args.fliplr = augmentation_config.get('fliplr', 0.5)
            args.mosaic = augmentation_config.get('mosaic', 1.0)
            args.mixup = augmentation_config.get('mixup', 0.0)
            args.copy_paste = augmentation_config.get('copy_paste', 0.0)

            # 从配置文件获取损失函数权重参数
            loss_config = model_config.get('loss', {})
            args.box_loss = loss_config.get('box', 7.5)
            args.cls_loss = loss_config.get('cls', 0.5)
            args.dfl_loss = loss_config.get('dfl', 1.5)
            args.mask_loss = loss_config.get('mask', 1.0)

            # 从配置文件获取验证参数
            validation_config = model_config.get('validation', {})
            args.conf_thres = validation_config.get('conf_thres', 0.001)
            args.iou_thres = validation_config.get('iou_thres', 0.6)
            args.max_det = validation_config.get('max_det', 300)

        # 如果命令行和配置文件都没有设置img_size，则使用默认值640
        if args.img_size is None:
            args.img_size = 640

        logger.info(f"使用参数: epochs={args.epochs}, batch_size={args.batch_size}, img_size={args.img_size}")

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
                name=args.name,
                optimizer=args.optimizer,
                lr0=args.lr0,
                patience=args.patience,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                warmup_epochs=args.warmup_epochs,
                warmup_momentum=args.warmup_momentum,
                warmup_bias_lr=args.warmup_bias_lr,
                hsv_h=args.hsv_h,
                hsv_s=args.hsv_s,
                hsv_v=args.hsv_v,
                degrees=args.degrees,
                translate=args.translate,
                scale=args.scale,
                shear=args.shear,
                perspective=args.perspective,
                flipud=args.flipud,
                fliplr=args.fliplr,
                mosaic=args.mosaic,
                mixup=args.mixup,
                copy_paste=args.copy_paste,
                box=args.box_loss,
                cls=args.cls_loss,
                dfl=args.dfl_loss,
                pose=args.mask_loss,  # ultralytics中mask损失可能使用pose参数
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                max_det=args.max_det
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
