"""
Ultralytics YOLO26实例分割训练器
使用Ultralytics Python API
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from trainers.base_trainer import BaseTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLO26Trainer(BaseTrainer):
    """Ultralytics YOLO26实例分割训练器"""

    # 支持的模型尺寸
    MODEL_SIZES = {
        'n': 'yolo26n.pt',
        's': 'yolo26s.pt',
        'm': 'yolo26m.pt',
        'l': 'yolo26l.pt',
        'x': 'yolo26x.pt'
    }

    def __init__(self, *args, model_size: str = 's', **kwargs):
        """
        初始化YOLO26训练器

        Args:
            model_size: 模型尺寸 (n/s/m/l/x)，默认's'
            **kwargs: 其他参数传递给BaseTrainer
        """
        super().__init__(*args, **kwargs)

        if model_size not in self.MODEL_SIZES:
            raise ValueError(f"不支持的模型尺寸: {model_size}，可选: {list(self.MODEL_SIZES.keys())}")

        self.model_size = model_size
        self.model_name = self.MODEL_SIZES[model_size]

        logger.info(f"YOLO26训练器初始化，模型尺寸: {model_size} ({self.model_name})")

        self.check_ultralytics_installed()

    def check_ultralytics_installed(self):
        """检查Ultralytics是否已安装"""
        try:
            import ultralytics
            logger.info(f"Ultralytics已安装，版本: {ultralytics.__version__}")
        except ImportError:
            logger.warning("Ultralytics未安装，正在安装...")
            subprocess = __import__('subprocess')
            subprocess.run(
                ['pip', 'install', '-q', 'ultralytics>=8.4.0'],
                check=True
            )
            logger.info("Ultralytics安装完成")

    def _load_model(self, weights_path: Optional[str] = None):
        """
        加载YOLO26模型

        Args:
            weights_path: 权重路径，如果为None则使用预训练模型
        """
        from ultralytics import YOLO

        if weights_path and Path(weights_path).exists():
            logger.info(f"加载自定义权重: {weights_path}")
            model = YOLO(weights_path)
        else:
            logger.info(f"加载预训练模型: {self.model_name}")
            model = YOLO(self.model_name)

        return model

    def train(
        self,
        epochs: int = 300,
        batch_size: int = 32,
        img_size: int = 640,
        resume: Optional[str] = None,
        workers: int = 8,
        device: str = None,
        name: str = 'exp',
        optimizer: str = 'auto',
        lr0: float = 0.01,
        patience: int = 50,
        **kwargs
    ):
        """
        训练YOLO26实例分割模型

        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 输入图片尺寸
            resume: 恢复训练的checkpoint路径
            workers: 数据加载线程数
            device: 设备ID，默认使用初始化时的设置
            name: 实验名称
            optimizer: 优化器 (SGD, Adam, AdamW, auto)
            lr0: 初始学习率
            patience: 早停耐心值
            **kwargs: 其他参数
        """
        device = device or self.device

        logger.info("开始训练YOLO26实例分割模型...")
        logger.info(f"数据集配置: {self.data_config_path}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"模型: {self.model_name}")
        logger.info(f"Epochs: {epochs}, Batch Size: {batch_size}, Image Size: {img_size}")

        try:
            # 加载模型
            model = self._load_model(weights_path=resume)

            # 训练参数
            train_args = {
                'data': str(self.data_config_path),
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': img_size,
                'workers': workers,
                'device': device,
                'project': str(self.output_dir),
                'name': name,
                'exist_ok': True,
                'verbose': True,
                'optimizer': optimizer,
                'lr0': lr0,
                'patience': patience,
                'task': 'segment',  # 实例分割任务
            }

            # 添加额外参数
            if 'mosaic' in kwargs:
                train_args['mosaic'] = kwargs['mosaic']
            if 'mixup' in kwargs:
                train_args['mixup'] = kwargs['mixup']
            if 'hsv_h' in kwargs:
                train_args['hsv_h'] = kwargs['hsv_h']
            if 'hsv_s' in kwargs:
                train_args['hsv_s'] = kwargs['hsv_s']
            if 'hsv_v' in kwargs:
                train_args['hsv_v'] = kwargs['hsv_v']
            if 'degrees' in kwargs:
                train_args['degrees'] = kwargs['degrees']
            if 'translate' in kwargs:
                train_args['translate'] = kwargs['translate']
            if 'scale' in kwargs:
                train_args['scale'] = kwargs['scale']
            if 'fliplr' in kwargs:
                train_args['fliplr'] = kwargs['fliplr']
            if 'flipud' in kwargs:
                train_args['flipud'] = kwargs['flipud']
            if 'mosaic' in kwargs:
                train_args['mosaic'] = kwargs['mosaic']
            if 'pretrained' in kwargs:
                train_args['pretrained'] = kwargs['pretrained']

            # 开始训练
            results = model.train(**train_args)

            logger.info("训练完成")
            logger.info(f"最佳模型保存在: {self.output_dir / name / 'weights' / 'best.pt'}")

            return results

        except Exception as e:
            logger.error(f"训练失败: {e}", exc_info=True)
            raise

    def validate(
        self,
        weights_path: str,
        batch_size: int = 32,
        img_size: int = 640,
        conf: float = 0.001,
        iou: float = 0.6,
        device: str = None,
        **kwargs
    ):
        """
        验证YOLO26模型

        Args:
            weights_path: 模型权重路径
            batch_size: 批次大小
            img_size: 输入图片尺寸
            conf: 置信度阈值
            iou: IOU阈值
            device: 设备ID
            **kwargs: 其他参数
        """
        device = device or self.device

        logger.info("开始验证YOLO26模型...")
        logger.info(f"权重: {weights_path}")

        try:
            # 加载模型
            model = self._load_model(weights_path=weights_path)

            # 验证参数
            val_args = {
                'data': str(self.data_config_path),
                'batch': batch_size,
                'imgsz': img_size,
                'conf': conf,
                'iou': iou,
                'device': device,
                'split': 'val',  # 使用验证集
            }

            # 开始验证
            results = model.val(**val_args)

            logger.info("验证完成")
            logger.info(f"mAP50: {results.box.map50:.4f}")
            logger.info(f"mAP50-95: {results.box.map:.4f}")

            return results

        except Exception as e:
            logger.error(f"验证失败: {e}", exc_info=True)
            raise

    def test(
        self,
        weights_path: str,
        batch_size: int = 32,
        img_size: int = 640,
        conf: float = 0.001,
        iou: float = 0.6,
        device: str = None,
        **kwargs
    ):
        """
        测试YOLO26模型

        Args:
            weights_path: 模型权重路径
            batch_size: 批次大小
            img_size: 输入图片尺寸
            conf: 置信度阈值
            iou: IOU阈值
            device: 设备ID
            **kwargs: 其他参数
        """
        device = device or self.device

        logger.info("开始测试YOLO26模型...")
        logger.info(f"权重: {weights_path}")

        try:
            # 加载模型
            model = self._load_model(weights_path=weights_path)

            # 测试参数（使用测试集）
            test_args = {
                'data': str(self.data_config_path),
                'batch': batch_size,
                'imgsz': img_size,
                'conf': conf,
                'iou': iou,
                'device': device,
                'split': 'test',  # 使用测试集
            }

            # 开始测试
            results = model.val(**test_args)

            logger.info("测试完成")
            logger.info(f"mAP50: {results.box.map50:.4f}")
            logger.info(f"mAP50-95: {results.box.map:.4f}")

            return results

        except Exception as e:
            logger.error(f"测试失败: {e}", exc_info=True)
            raise

    def export(
        self,
        weights_path: str,
        format: str = 'onnx',
        img_size: int = 640,
        batch_size: int = 1,
        half: bool = False,
        dynamic: bool = False,
        simplify: bool = True,
        opset: int = 12,
        **kwargs
    ):
        """
        导出YOLO26模型

        Args:
            weights_path: 模型权重路径
            format: 导出格式 (onnx, torchscript, engine, coreml, tflite, pb)
            img_size: 输入图片尺寸
            batch_size: 批次大小
            half: 是否使用FP16
            dynamic: 是否动态输入尺寸
            simplify: 是否简化ONNX模型
            opset: ONNX opset版本
            **kwargs: 其他参数
        """
        output_dir = self.output_dir / 'exported_models'
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("开始导出YOLO26模型...")
        logger.info(f"权重: {weights_path}")
        logger.info(f"格式: {format}")

        try:
            # 加载模型
            model = self._load_model(weights_path=weights_path)

            # 导出参数
            export_args = {
                'format': format,
                'imgsz': img_size,
                'batch': batch_size,
                'half': half,
                'dynamic': dynamic,
                'simplify': simplify,
                'opset': opset,
            }

            # 特定格式的额外参数
            if format == 'engine':
                export_args['workspace'] = kwargs.get('workspace', 4)
            elif format == 'tflite':
                export_args['int8'] = kwargs.get('int8', False)

            # 开始导出
            results = model.export(**export_args)

            logger.info(f"模型导出完成: {results}")
            logger.info(f"导出文件保存在: {output_dir}")

            return results

        except Exception as e:
            logger.error(f"导出失败: {e}", exc_info=True)
            raise
