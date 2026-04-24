"""
线程安全的日志模块
支持等级控制、时间打印、异常捕获等功能
"""

import logging
import logging.handlers
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class ThreadSafeLogger:
    """线程安全的日志记录器"""

    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, name: str = "app", log_dir: str = "logs", prefix: str = "app"):
        with cls._lock:
            if name not in cls._instances:
                cls._instances[name] = super().__new__(cls)
            return cls._instances[name]

    def __init__(self, name: str = "app", log_dir: str = "logs", prefix: str = "app"):
        if hasattr(self, '_initialized'):
            return

        self.name = name
        self.log_dir = Path(log_dir)
        self.prefix = prefix
        self._lock = threading.RLock()  # 递归锁，支持同一线程多次获取

        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 设置日志器
        self.logger = logging.getLogger(name)
        if self.logger.handlers:
            # 如果已经配置过，直接返回
            self._initialized = True
            return

        self.logger.setLevel(logging.DEBUG)

        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(threadName)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # 文件处理器 - 按日期分割
        today = datetime.now().strftime('%Y-%m-%d')
        log_file = self.log_dir / f"{prefix}_{today}.log"

        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file,
            when='midnight',
            interval=1,
            backupCount=30  # 保留30天的日志
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self._initialized = True

    def _log(self, level: int, message: str, *args, **kwargs):
        """内部日志记录方法"""
        with self._lock:
            self.logger.log(level, message, *args, **kwargs)

    def debug(self, message: str):
        """记录DEBUG级别日志"""
        self._log(logging.DEBUG, message)

    def info(self, message: str):
        """记录INFO级别日志"""
        self._log(logging.INFO, message)

    def warning(self, message: str):
        """记录WARNING级别日志"""
        self._log(logging.WARNING, message)

    def error(self, message: str):
        """记录ERROR级别日志"""
        self._log(logging.ERROR, message)

    def critical(self, message: str):
        """记录CRITICAL级别日志"""
        self._log(logging.CRITICAL, message)

    def exception(self, message: str = "Exception occurred"):
        """记录异常信息，包括堆栈跟踪"""
        with self._lock:
            self.logger.exception(message)

    def log_exception(self, exc: Exception, message: Optional[str] = None):
        """专门记录异常的便捷方法"""
        import traceback

        with self._lock:
            if message:
                self.error(f"{message}: {str(exc)}")
            else:
                self.error(f"Exception: {str(exc)}")

            # 记录完整的堆栈跟踪
            self.error("Full traceback:")
            for line in traceback.format_exception(type(exc), exc, exc.__traceback__):
                self.error(line.rstrip())

    def set_level(self, level: str):
        """设置日志级别"""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        if level.upper() not in level_map:
            raise ValueError(f"Invalid log level: {level}. Must be one of {list(level_map.keys())}")

        with self._lock:
            self.logger.setLevel(level_map[level.upper()])

            # 更新所有处理器的级别
            for handler in self.logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    # 控制台只显示INFO及以上
                    handler.setLevel(logging.INFO)
                else:
                    # 文件显示所有级别
                    handler.setLevel(level_map[level.upper()])

    def get_level(self) -> str:
        """获取当前日志级别"""
        level_map = {
            logging.DEBUG: 'DEBUG',
            logging.INFO: 'INFO',
            logging.WARNING: 'WARNING',
            logging.ERROR: 'ERROR',
            logging.CRITICAL: 'CRITICAL'
        }
        return level_map.get(self.logger.level, 'UNKNOWN')


# 全局日志记录器实例
logger = ThreadSafeLogger()


def get_logger(name: str = "app", log_dir: str = "logs", prefix: str = "app") -> ThreadSafeLogger:
    """获取日志记录器实例"""
    return ThreadSafeLogger(name=name, log_dir=log_dir, prefix=prefix)


def setup_global_exception_handler(logger_instance: Optional[ThreadSafeLogger] = None):
    """设置全局异常处理器，捕获未处理的异常"""

    def exception_handler(exc_type, exc_value, exc_traceback):
        """全局异常处理器"""
        if issubclass(exc_type, KeyboardInterrupt):
            # 忽略键盘中断
            return

        logger_to_use = logger_instance or logger

        import traceback
        import sys

        # 记录异常信息
        logger_to_use.critical("未处理的异常发生!")
        logger_to_use.critical(f"异常类型: {exc_type.__name__}")
        logger_to_use.critical(f"异常信息: {exc_value}")

        # 获取异常发生的文件和行号
        tb = traceback.extract_tb(exc_traceback)
        if tb:
            filename, line_number, func_name, text = tb[-1]
            logger_to_use.critical(f"发生位置: {filename}:{line_number} 在函数 {func_name}")
            if text:
                logger_to_use.critical(f"代码行: {text}")

        # 记录完整的堆栈跟踪
        logger_to_use.critical("完整堆栈跟踪:")
        for line in traceback.format_exception(exc_type, exc_value, exc_traceback):
            logger_to_use.critical(line.rstrip())

        # 确保日志写入
        logging.shutdown()

    # 设置全局异常处理器
    sys.excepthook = exception_handler


# 在模块导入时设置全局异常处理器
setup_global_exception_handler()