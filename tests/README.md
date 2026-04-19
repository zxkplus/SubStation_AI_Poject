# 测试文档

## 概述

本测试套件使用pytest模块，用于检查环境配置、测试训练、验证和推理流程是否正确。

## 测试分类

### 1. 环境配置测试 (`test_environment.py`)

测试项目环境和依赖是否正确配置。

**测试内容**：
- Python版本检查（>= 3.8）
- Ultralytics安装检查
- PyTorch安装检查
- CUDA可用性检查
- OpenCV、NumPy、Matplotlib等库安装检查
- scripts和trainers目录存在性检查
- 训练器模块导入检查
- GPU内存检查
- 磁盘空间检查

**运行方式**：
```bash
# 单独运行环境测试
python tests/test_environment.py

# 或使用pytest
pytest tests/test_environment.py -v
```

### 2. 数据处理测试 (`test_data.py`)

测试数据加载、转换和验证功能。

**测试内容**：
- 数据加载器导入和功能测试
- 数据集加载测试
- 类别统计测试
- 统计模块测试
- YAML配置文件生成和验证
- 图片加载测试
- 标注格式验证
- YOLO转换器和验证器导入测试

**运行方式**：
```bash
python tests/test_data.py
pytest tests/test_data.py -v
```

### 3. 训练流程测试 (`test_training.py`)

测试YOLO模型训练流程。

**测试内容**：
- YOLOv8/YOLOv26训练器导入测试
- 训练器初始化测试
- 快速训练测试（1个epoch）
- 输出文件验证

**注意事项**：
- 使用小型测试数据集
- 训练1个epoch用于快速验证
- 使用CPU模式（测试环境）
- 可能需要较长时间（几分钟）

**运行方式**：
```bash
python tests/test_training.py
pytest tests/test_training.py -v
```

### 4. 验证和推理测试 (`test_validation.py`)

测试模型验证和推理流程。

**测试内容**：
- validate_with_mask导入测试
- diagnose_mask导入测试
- YOLOv8验证流程测试
- YOLOv8推理流程测试
- Mask可视化测试
- numpy数组推理方式测试

**注意事项**：
- 需要网络连接（下载预训练模型）
- 使用Ultralytics预训练模型测试
- 可能出现失败（网络或模型问题）

**运行方式**：
```bash
python tests/test_validation.py
pytest tests/test_validation.py -v
```

## 安装依赖

```bash
pip install pytest
```

## 运行测试

### 方式1: 使用主脚本（推荐）

```bash
python tests/run_tests.py
```

交互式菜单：
1. 使用Pytest运行所有测试（推荐）
2. 逐个运行测试文件
3. 运行特定测试文件

### 方式2: 使用Pytest命令

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试文件
pytest tests/test_environment.py -v

# 运行特定测试函数
pytest tests/test_training.py::test_yolov8_trainer_import -v

# 显示详细输出
pytest tests/ -vv

# 只运行失败的测试
pytest tests/ --lf

# 显示print输出
pytest tests/ -s
```

### 方式3: 直接运行测试文件

```bash
python tests/test_environment.py
python tests/test_data.py
python tests/test_training.py
python tests/test_validation.py
```

## 测试输出

### 成功输出

```
============================================================
开始环境配置检查
============================================================

[测试] Python版本检查...
✓ Python版本: 3.10.0

[测试] Ultralytics安装检查...
✓ Ultralytics已安装，版本: 8.0.0

...

============================================================
测试完成，全部通过: 15/15
============================================================
```

### 失败输出

```
============================================================
测试完成，失败: 2/15
============================================================

失败的测试:
  - test_ultralytics_installed: Ultralytics未安装，请运行: pip install ultralytics
  - test_torch_installed: PyTorch未安装，请运行: pip install torch torchvision
============================================================
```

## 常见问题

### Q1: 测试失败，提示Ultralytics未安装

**A**: 安装Ultralytics
```bash
pip install ultralytics
```

### Q2: 测试失败，提示PyTorch未安装

**A**: 安装PyTorch
```bash
pip install torch torchvision
```

### Q3: 训练测试失败，提示内存不足

**A**: 训练测试使用小型数据集，但如果系统内存不足，可以：
1. 关闭其他程序
2. 减少batch_size（修改测试代码）
3. 跳过训练测试

### Q4: 验证测试失败，提示网络错误

**A**: 验证测试需要下载预训练模型。如果网络不可用：
1. 检查网络连接
2. 手动下载模型到缓存目录
3. 跳过验证测试

### Q5: 如何只运行快速测试？

**A**: 只运行环境测试和数据测试
```bash
python tests/test_environment.py
python tests/test_data.py
```

跳过耗时的训练和验证测试。

### Q6: 测试输出太多，如何简化？

**A**: 使用pytest的输出控制
```bash
# 只显示失败的测试
pytest tests/ --tb=line

# 只显示概要
pytest tests/ -q

# 不显示print输出
pytest tests/ --capture=no
```

## 测试最佳实践

### 1. 训练前运行环境测试

```bash
python tests/test_environment.py
```

确保环境配置正确后再开始训练。

### 2. 数据集处理前运行数据测试

```bash
python tests/test_data.py
```

确保数据加载和转换功能正常。

### 3. 定期运行所有测试

```bash
python tests/run_tests.py
```

确保整个系统工作正常。

### 4. 修改代码后运行相关测试

如果修改了训练器代码：
```bash
pytest tests/test_training.py -v
```

如果修改了验证代码：
```bash
pytest tests/test_validation.py -v
```

## 持续集成

测试可以集成到CI/CD流程中：

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install pytest ultralytics torch torchvision opencv-python numpy matplotlib pillow pyyaml
      - name: Run tests
        run: pytest tests/ -v
```

## 测试覆盖率

要查看测试覆盖率：

```bash
# 安装pytest-cov
pip install pytest-cov

# 运行测试并生成覆盖率报告
pytest tests/ --cov=scripts --cov-report=html

# 查看报告
open htmlcov/index.html
```

## 编写新测试

添加新测试时：

1. 在对应的测试文件中添加测试函数
2. 函数名以`test_`开头
3. 使用`assert`进行断言
4. 添加适当的打印信息

示例：
```python
def test_new_feature():
    print("\n[测试] 新功能测试...")

    # 测试代码
    result = some_function()

    # 断言
    assert result is not None, "结果不应为空"
    assert result > 0, "结果应大于0"

    print("✓ 新功能测试通过")
```

## 调试测试

如果测试失败：

1. 使用详细输出
```bash
pytest tests/test_file.py -vv -s
```

2. 在测试函数中添加调试信息
```python
def test_something():
    print(f"调试信息: {variable}")
    # 测试代码
```

3. 使用Python调试器
```bash
pytest tests/test_file.py::test_function --pdb
```

## 性能测试

测试运行时间：
```bash
pytest tests/ --durations=10
```

显示最慢的10个测试。

## 参考资源

- [Pytest官方文档](https://docs.pytest.org/)
- [PyTorch测试最佳实践](https://pytorch.org/tutorials/recipes/recipes/testing_recipe.html)
