"""
客户端模块导入测试

验证所有客户端组件是否可以正确导入和初始化
"""

import sys
from pathlib import Path

# 添加 service 目录到路径
service_dir = Path(__file__).parent
sys.path.insert(0, str(service_dir))


def test_imports():
    """测试导入"""
    print("=" * 60)
    print("测试模块导入")
    print("=" * 60)
    
    try:
        from client_app import InferenceClient, ROIDrawer, ResultRenderer, SubstationClientApp
        print("✅ client_app 模块导入成功")
        
        from client_app import tk, Image, ImageTk, ImageDraw
        print("✅ 依赖库导入成功 (tkinter, PIL)")
        
        import requests
        print("✅ requests 库导入成功")
        
        import numpy as np
        print("✅ numpy 库导入成功")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入失败: {e}")
        return False


def test_inference_client():
    """测试推理客户端"""
    print("\n" + "=" * 60)
    print("测试 InferenceClient")
    print("=" * 60)
    
    try:
        from client_app import InferenceClient
        
        # 创建客户端实例
        client = InferenceClient(base_url="http://localhost:8000")
        print(f"✅ InferenceClient 创建成功")
        print(f"   Base URL: {client.base_url}")
        print(f"   Infer URL: {client.infer_url}")
        
        return True
        
    except Exception as e:
        print(f"❌ InferenceClient 测试失败: {e}")
        return False


def test_result_renderer():
    """测试结果渲染器"""
    print("\n" + "=" * 60)
    print("测试 ResultRenderer")
    print("=" * 60)
    
    try:
        from client_app import ResultRenderer
        from PIL import Image
        
        # 创建测试图片
        test_image = Image.new('RGB', (100, 100), color='white')
        
        # 创建模拟推理结果
        mock_result = {
            "image_width": 100,
            "image_height": 100,
            "results": [
                {
                    "roi": {"x1": 10, "y1": 10, "x2": 90, "y2": 90},
                    "detections": [
                        {
                            "bbox": [20, 20, 80, 80],
                            "confidence": 0.9,
                            "class_id": 0,
                            "contours": [
                                {
                                    "points": [
                                        [20, 20], [80, 20], 
                                        [80, 80], [20, 80]
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        # 绘制结果
        result_image = ResultRenderer.draw_results_on_image(test_image, mock_result)
        print(f"✅ ResultRenderer 工作正常")
        print(f"   输入尺寸: {test_image.size}")
        print(f"   输出尺寸: {result_image.size}")
        
        return True
        
    except Exception as e:
        print(f"❌ ResultRenderer 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dependencies():
    """测试依赖包版本"""
    print("\n" + "=" * 60)
    print("检查依赖包版本")
    print("=" * 60)
    
    packages = {
        'PIL': 'Pillow',
        'requests': 'requests',
        'numpy': 'numpy',
    }
    
    for import_name, package_name in packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package_name}: {version}")
        except ImportError:
            print(f"❌ {package_name}: 未安装")
            return False
    
    return True


def main():
    """运行所有测试"""
    print("\nSubStation AI 客户端模块测试\n")
    
    results = []
    
    # 运行测试
    results.append(("依赖包版本", test_dependencies()))
    results.append(("模块导入", test_imports()))
    results.append(("InferenceClient", test_inference_client()))
    results.append(("ResultRenderer", test_result_renderer()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name:20s} {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 所有测试通过！客户端已准备就绪。")
        print("\n现在可以运行:")
        print("  python client_app.py")
        print("或")
        print("  python launch_client.py")
    else:
        print("⚠️  部分测试失败，请检查错误信息并安装缺失的依赖。")
        print("\n安装依赖:")
        print("  pip install -r client_requirements.txt")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
