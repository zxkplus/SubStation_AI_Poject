# SubStation AI 推理服务安装指南

## ❌ 问题描述

运行以下命令时出现错误：
```bash
python -m uvicorn service.app:app --port 8000
```

错误信息：
```
No module named uvicorn
```

## 🔍 原因分析

`uvicorn` 是一个 ASGI 服务器，用于运行 FastAPI 应用。该包未安装在你的 Python 环境（Ultralytics conda 环境）中。

## ✅ 解决方案

### 方案 1：使用自动安装脚本（推荐）

在项目根目录运行：

```bash
cd d:\zengxinke\workspace\SubStation_AI_Poject
python service/install_and_start.py
```

这个脚本会：
1. 自动检查并安装必需的包（uvicorn, fastapi, pydantic）
2. 验证安装是否成功
3. 询问是否立即启动服务

### 方案 2：手动安装（Windows 批处理）

双击运行：
```
service/start_server.bat
```

或者在命令行运行：
```bash
cd d:\zengxinke\workspace\SubStation_AI_Poject\service
start_server.bat
```

### 方案 3：手动安装依赖

在你的 `Ultralytics` conda 环境中执行：

```bash
# 安装核心依赖
python -m pip install uvicorn[standard] fastapi pydantic

# 或者安装完整依赖（推荐）
python -m pip install -r requirements.txt
```

### 方案 4：使用 conda 安装

```bash
# 使用 conda 安装（如果 pip 不可用）
conda install -c conda-forge uvicorn fastapi pydantic
```

## 🚀 启动服务

安装完成后，启动推理服务：

```bash
# 基本启动
python -m uvicorn service.app:app --host 0.0.0.0 --port 8000

# 开发模式（自动重载）
python -m uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload

# 使用启动脚本
python service/run.py
```

## ✅ 验证服务

服务启动后，在浏览器中访问：

- **API 文档**: http://localhost:8000/docs
- **ReDoc 文档**: http://localhost:8000/redoc

如果能看到 Swagger UI 界面，说明服务启动成功！

## 📋 完整依赖列表

服务端所需的核心依赖：

| 包名 | 版本要求 | 用途 |
|------|----------|------|
| fastapi | >=0.104.0 | Web 框架 |
| uvicorn | >=0.24.0 | ASGI 服务器 |
| pydantic | >=2.5.0 | 数据验证 |
| ultralytics | >=8.4.0 | YOLO 模型 |
| torch | >=2.0.0 | 深度学习框架 |
| torchvision | >=0.15.0 | 图像处理 |
| Pillow | >=9.0.0 | 图像加载 |
| numpy | >=1.19.0 | 数值计算 |

## 🔄 完整启动流程

### 步骤 1: 安装依赖
```bash
cd d:\zengxinke\workspace\SubStation_AI_Poject
python service/install_and_start.py
```

### 步骤 2: 启动服务（终端 1）
```bash
python -m uvicorn service.app:app --port 8000
```

**保持此终端运行**

### 步骤 3: 启动客户端（终端 2）
```bash
cd service
python launch_client.py
```

### 步骤 4: 使用客户端
1. 选择图片
2. 划定 ROI 区域
3. 执行推理
4. 查看结果

## ❓ 常见问题

### Q1: pip 命令找不到？

**A**: 使用 `python -m pip` 替代 `pip`：
```bash
python -m pip install uvicorn fastapi pydantic
```

### Q2: 安装速度慢或失败？

**A**: 使用国内镜像源：
```bash
python -m pip install uvicorn fastapi pydantic -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q3: uvicorn[standard] 安装失败？

**A**: 尝试不带 extra 的安装：
```bash
python -m pip install uvicorn fastapi pydantic
```

### Q4: 端口 8000 已被占用？

**A**: 使用其他端口：
```bash
python -m uvicorn service.app:app --port 8080
```

然后在客户端中修改服务器地址为 `http://localhost:8080`

### Q5: 如何在后台运行服务？

**A**: Windows 可以使用 `start` 命令：
```bash
start /B python -m uvicorn service.app:app --port 8000
```

Linux/Mac：
```bash
nohup python -m uvicorn service.app:app --port 8000 &
```

## 📝 更新日志

### 2026-05-08
- ✅ 添加 requirements.txt
- ✅ 创建自动安装脚本
- ✅ 创建 Windows 启动脚本
- ✅ 添加详细安装文档

## 🔗 相关文档

- [CLIENT_README.md](CLIENT_README.md) - 客户端使用指南
- [README.md](README.md) - 服务模块说明
- [../QUICKSTART_CLIENT.md](../QUICKSTART_CLIENT.md) - 快速开始指南
