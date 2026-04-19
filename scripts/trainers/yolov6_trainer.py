"""
YOLOv6实例分割训练器
"""

import sys
import yaml
import subprocess
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainers.base_trainer import BaseTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv6Trainer(BaseTrainer):
    """YOLOv6实例分割训练器"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_yolov6_installed()

    def check_yolov6_installed(self):
        """检查YOLOv6是否已安装"""
        try:
            import yolov6
            logger.info(f"YOLOv6已安装，版本: {yolov6.__version__ if hasattr(yolov6, '__version__') else 'unknown'}")
        except ImportError:
            logger.warning("YOLOv6未安装，正在安装...")
            subprocess.run(
                ['pip', 'install', '-q', 'git+https://github.com/meituan/YOLOv6.git@main'],
                check=True
            )
            logger.info("YOLOv6安装完成")

    def train(
        self,
        epochs: int = 300,
        batch_size: int = 32,
        img_size: int = 640,
        resume: Optional[str] = None,
        workers: int = 8,
        conf_thres: float = 0.001,
        iou_thres: float = 0.7,
        device: str = None,
        name: str = 'exp',
        **kwargs
    ):
        """
        训练YOLOv6实例分割模型

        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 输入图片尺寸
            resume: 恢复训练的checkpoint路径
            workers: 数据加载线程数
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
            device: 设备ID，默认使用初始化时的设置
            name: 实验名称
            **kwargs: 其他参数
        """
        device = device or self.device

        # 构建训练命令
        cmd = [
            'python',
            '-m', 'yolov6.train',
            '--data', str(self.data_config_path),
            '--batch-size', str(batch_size),
            '--epochs', str(epochs),
            '--img-size', str(img_size),
            '--device', device,
            '--workers', str(workers),
            '--conf-thres', str(conf_thres),
            '--iou-thres', str(iou_thres),
            '--project', str(self.output_dir),
            '--name', name,
            '--task', 'seg'  # 实例分割任务
        ]

        # 如果指定了恢复训练
        if resume:
            cmd.extend(['--resume', resume])

        # 添加额外参数
        if 'lr' in kwargs:
            cmd.extend(['--lr', str(kwargs['lr'])])
        if 'optimizer' in kwargs:
            cmd.extend(['--optimizer', kwargs['optimizer']])

        logger.info("开始训练YOLOv6实例分割模型...")
        logger.info(f"命令: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            logger.info("训练完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"训练失败: {e}")
            raise

    def validate(
        self,
        weights_path: str,
        batch_size: int = 32,
        img_size: int = 640,
        conf_thres: float = 0.001,
        iou_thres: float = 0.65,
        device: str = None,
        **kwargs
    ):
        """
        验证YOLOv6模型

        Args:
            weights_path: 模型权重路径
            batch_size: 批次大小
            img_size: 输入图片尺寸
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
            device: 设备ID
            **kwargs: 其他参数
        """
        device = device or self.device

        cmd = [
            'python',
            '-m', 'yolov6.val',
            '--data', str(self.data_config_path),
            '--weights', weights_path,
            '--batch-size', str(batch_size),
            '--img-size', str(img_size),
            '--device', device,
            '--conf-thres', str(conf_thres),
            '--iou-thres', str(iou_thres),
            '--task', 'seg'
        ]

        logger.info("开始验证模型...")
        logger.info(f"命令: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            logger.info("验证完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"验证失败: {e}")
            raise

    def test(
        self,
        weights_path: str,
        batch_size: int = 32,
        img_size: int = 640,
        conf_thres: float = 0.001,
        iou_thres: float = 0.65,
        device: str = None,
        **kwargs
    ):
        """
        测试YOLOv6模型

        Args:
            weights_path: 模型权重路径
            batch_size: 批次大小
            img_size: 输入图片尺寸
            conf_thres: 置信度阈值
            iou_thres: IOU阈值
            device: 设备ID
            **kwargs: 其他参数
        """
        device = device or self.device

        # 测试和验证使用相同的命令，但使用test数据集
        cmd = [
            'python',
            '-m', 'yolov6.val',
            '--data', str(self.data_config_path),
            '--weights', weights_path,
            '--batch-size', str(batch_size),
            '--img-size', str(img_size),
            '--device', device,
            '--conf-thres', str(conf_thres),
            '--iou-thres', str(iou_thres),
            '--task', 'seg',
            '--test'  # 使用测试集
        ]

        logger.info("开始测试模型...")
        logger.info(f"命令: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            logger.info("测试完成")
        except subprocess.CalledProcessError as e:
            logger.error(f"测试失败: {e}")
            raise

    def export(
        self,
        weights_path: str,
        format: str = 'onnx',
        img_size: int = 640,
        batch_size: int = 1,
        opset: int = 12,
        simplify: bool = True,
        **kwargs
    ):
        """
        导出YOLOv6模型

        Args:
            weights_path: 模型权重路径
            format: 导出格式 (onnx, torchscript)
            img_size: 输入图片尺寸
            batch_size: 批次大小
            opset: ONNX opset版本
            simplify: 是否简化ONNX模型
            **kwargs: 其他参数
        """
        output_dir = self.output_dir / 'exported_models'
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            'python',
            '-m', 'yolov6.deploy',
            '--weights', weights_path,
            '--img-size', str(img_size),
            '--batch-size', str(batch_size),
            '--device', self.device,
            '--half',  # 使用FP16
        ]

        if format == 'onnx':
            cmd.extend([
                '--include', 'onnx',
                '--opset', str(opset),
                '--simplify', str(simplify).lower(),
            ])
        elif format == 'torchscript':
            cmd.extend(['--include', 'torchscript'])
        else:
            raise ValueError(f"不支持的导出格式: {format}")

        logger.info("开始导出模型...")
        logger.info(f"命令: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
            logger.info(f"模型导出完成，保存到: {output_dir}")
        except subprocess.CalledProcessError as e:
            logger.error(f"导出失败: {e}")
            raise
