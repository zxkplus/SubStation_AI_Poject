"""
基础训练器接口
定义所有YOLO版本训练器的通用接口
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """YOLO训练器基类"""

    def __init__(
        self,
        data_config_path: str,
        output_dir: str,
        model_config: Optional[Dict[str, Any]] = None,
        device: str = '0',
        **kwargs
    ):
        """
        初始化训练器

        Args:
            data_config_path: 数据集配置文件路径
            output_dir: 输出目录
            model_config: 模型配置字典
            device: 设备ID，'0'表示GPU0，'-1'表示CPU
            **kwargs: 其他参数
        """
        self.data_config_path = Path(data_config_path)
        self.output_dir = Path(output_dir)
        self.model_config = model_config or {}
        self.device = device
        self.kwargs = kwargs

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"初始化训练器: {self.__class__.__name__}")
        logger.info(f"数据集配置: {self.data_config_path}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"设备: {device}")

    @abstractmethod
    def train(
        self,
        epochs: int = 300,
        batch_size: int = 32,
        img_size: int = 640,
        resume: Optional[str] = None,
        workers: int = 8,
        **kwargs
    ):
        """
        训练模型

        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 输入图片尺寸
            resume: 恢复训练的checkpoint路径
            workers: 数据加载线程数
            **kwargs: 其他训练参数
        """
        pass

    @abstractmethod
    def validate(
        self,
        weights_path: str,
        batch_size: int = 32,
        img_size: int = 640,
        **kwargs
    ):
        """
        验证模型

        Args:
            weights_path: 模型权重路径
            batch_size: 批次大小
            img_size: 输入图片尺寸
            **kwargs: 其他验证参数
        """
        pass

    @abstractmethod
    def test(
        self,
        weights_path: str,
        batch_size: int = 32,
        img_size: int = 640,
        **kwargs
    ):
        """
        测试模型

        Args:
            weights_path: 模型权重路径
            batch_size: 批次大小
            img_size: 输入图片尺寸
            **kwargs: 其他测试参数
        """
        pass

    @abstractmethod
    def export(
        self,
        weights_path: str,
        format: str = 'onnx',
        **kwargs
    ):
        """
        导出模型

        Args:
            weights_path: 模型权重路径
            format: 导出格式 (onnx, torchscript, engine)
            **kwargs: 其他导出参数
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息

        Returns:
            模型信息字典
        """
        return {
            'trainer_name': self.__class__.__name__,
            'data_config': str(self.data_config_path),
            'output_dir': str(self.output_dir),
            'device': self.device
        }
