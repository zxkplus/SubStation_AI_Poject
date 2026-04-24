#!/usr/bin/env python3
"""
日志模块使用示例
演示如何使用线程安全的日志记录器
"""

import sys
import time
import threading
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from service.logger import get_logger, setup_global_exception_handler


def demo_basic_logging():
    """演示基础日志功能"""
    print("=== 基础日志功能演示 ===")

    # 获取日志记录器
    logger = get_logger(name="demo", log_dir="demo_logs", prefix="demo")

    logger.info("开始演示基础日志功能")
    logger.debug("这是一条调试信息")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")

    print("基础日志演示完成，请查看 demo_logs/demo_YYYY-MM-DD.log 文件")


def demo_exception_logging():
    """演示异常日志记录"""
    print("\n=== 异常日志记录演示 ===")

    logger = get_logger(name="exception_demo", log_dir="demo_logs", prefix="exception")

    logger.info("开始演示异常日志记录")

    try:
        # 故意制造一个异常
        result = 1 / 0
    except ZeroDivisionError as e:
        logger.log_exception(e, "除零错误示例")

    # 使用 exception 方法记录当前异常
    try:
        raise ValueError("这是一个测试异常")
    except ValueError:
        logger.exception("使用 exception 方法记录异常")

    print("异常日志演示完成")


def demo_thread_safety():
    """演示线程安全性"""
    print("\n=== 线程安全演示 ===")

    logger = get_logger(name="thread_demo", log_dir="demo_logs", prefix="thread")

    def worker(thread_id):
        """工作线程函数"""
        for i in range(5):
            logger.info(f"线程 {thread_id}: 处理任务 {i}")
            time.sleep(0.1)

    logger.info("启动多线程日志记录测试")

    # 创建多个线程
    threads = []
    for i in range(3):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    # 等待所有线程完成
    for t in threads:
        t.join()

    logger.info("多线程日志记录测试完成")
    print("线程安全演示完成")


def demo_log_levels():
    """演示日志级别控制"""
    print("\n=== 日志级别控制演示 ===")

    logger = get_logger(name="level_demo", log_dir="demo_logs", prefix="level")

    print(f"当前日志级别: {logger.get_level()}")

    # 设置为 ERROR 级别
    logger.set_level("ERROR")
    print(f"设置日志级别为 ERROR: {logger.get_level()}")

    logger.debug("这条调试信息不会显示")
    logger.info("这条信息信息不会显示")
    logger.warning("这条警告信息不会显示")
    logger.error("这条错误信息会显示")
    logger.critical("这条严重错误信息会显示")

    # 设置为 DEBUG 级别
    logger.set_level("DEBUG")
    print(f"设置日志级别为 DEBUG: {logger.get_level()}")

    logger.debug("现在这条调试信息会显示")
    logger.info("现在这条信息信息会显示")

    print("日志级别控制演示完成")


def demo_global_exception_handler():
    """演示全局异常处理器"""
    print("\n=== 全局异常处理器演示 ===")

    # 设置全局异常处理器
    logger = get_logger(name="global_demo", log_dir="demo_logs", prefix="global")
    setup_global_exception_handler(logger)

    logger.info("全局异常处理器已设置")

    # 模拟未捕获的异常
    def cause_unhandled_exception():
        time.sleep(1)  # 让主线程先输出
        raise RuntimeError("这是一个未捕获的异常示例")

    # 在单独的线程中产生异常，这样不会影响主程序
    exception_thread = threading.Thread(target=cause_unhandled_exception)
    exception_thread.start()
    exception_thread.join()

    logger.info("全局异常处理器演示完成")
    print("全局异常处理器演示完成（异常已被记录到日志）")


def main():
    """主函数"""
    print("SubStation AI 日志模块演示")
    print("=" * 50)

    # 演示各个功能
    demo_basic_logging()
    demo_exception_logging()
    demo_thread_safety()
    demo_log_levels()
    demo_global_exception_handler()

    print("\n" + "=" * 50)
    print("所有演示完成！")
    print("请查看 demo_logs 目录中的日志文件")


if __name__ == "__main__":
    main()