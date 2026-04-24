# SubStation AI 日志模块

这是一个线程安全的日志记录模块，专为 SubStation AI 项目设计。

## 功能特性

- ✅ **线程安全**：使用递归锁确保多线程环境下的安全日志记录
- ✅ **等级控制**：支持 DEBUG、INFO、WARNING、ERROR、CRITICAL 五个级别
- ✅ **时间格式化**：自动记录时间戳，格式为 `YYYY-MM-DD HH:MM:SS`
- ✅ **文件名格式化**：按 `前缀_日期.log` 格式命名，如 `service_2026-04-24.log`
- ✅ **异常捕获**：自动捕获未处理的异常并记录到日志
- ✅ **堆栈跟踪**：记录异常发生的位置、行号和完整堆栈信息

## 基本使用

```python
from service.logger import get_logger

# 获取日志记录器
logger = get_logger(name="my_service", log_dir="logs", prefix="service")

# 记录不同级别的日志
logger.debug("调试信息")
logger.info("普通信息")
logger.warning("警告信息")
logger.error("错误信息")
logger.critical("严重错误")
```

## 高级功能

### 异常记录

```python
try:
    # 可能出错的代码
    result = risky_operation()
except Exception as e:
    logger.log_exception(e, "操作失败")
```

### 日志级别设置

```python
# 设置日志级别
logger.set_level("ERROR")  # 只显示 ERROR 及以上的日志

# 获取当前级别
current_level = logger.get_level()
```

### 全局异常处理器

```python
from service.logger import setup_global_exception_handler

# 设置全局异常处理器（推荐在程序启动时设置）
logger = get_logger(name="app", log_dir="logs", prefix="app")
setup_global_exception_handler(logger)
```

## 日志格式

日志文件的格式如下：

```
2026-04-24 11:12:04 [INFO] MainThread - 这是一条信息
2026-04-24 11:12:04 [ERROR] Thread-1 - 发生错误
```

包含：
- 时间戳（年-月-日 时:分:秒）
- 日志级别
- 线程名称
- 日志消息

## 配置参数

### get_logger 参数

- `name`: 日志记录器名称，用于单例模式
- `log_dir`: 日志文件存放目录，默认为 "logs"
- `prefix`: 日志文件名前缀，默认为 "app"

### 日志级别

- `DEBUG`: 调试信息
- `INFO`: 普通信息
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误

## 运行演示

```bash
cd /path/to/project
python service/demo_logger.py
```

这将演示所有功能并生成示例日志文件。

## 集成到服务

日志模块已经集成到 FastAPI 服务中：

```python
from service.logger import get_logger

logger = get_logger(name="service", log_dir="logs", prefix="service")

@app.post("/infer")
def infer(request: InferenceRequest):
    logger.info(f"收到推理请求: {len(request.rois)} 个ROI")
    # ... 处理逻辑
    logger.info(f"推理完成: 检测到 {detections_count} 个目标")
```

## 注意事项

1. 日志文件会按日期自动分割，每天创建一个新文件
2. 默认保留最近30天的日志文件
3. 控制台只显示 INFO 级别及以上的日志
4. 文件记录所有级别的日志（可通过 `set_level` 控制）
5. 全局异常处理器会捕获所有未处理的异常