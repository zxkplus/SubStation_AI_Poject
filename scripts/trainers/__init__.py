"""
Trainers包初始化
"""

from .base_trainer import BaseTrainer
from .yolov6_trainer import YOLOv6Trainer
from .yolov26_trainer import YOLO26Trainer
from .yolov8_trainer import YOLOv8Trainer

__all__ = ['BaseTrainer', 'YOLOv6Trainer', 'YOLO26Trainer', 'YOLOv8Trainer']
