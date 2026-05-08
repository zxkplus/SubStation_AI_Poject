"""简单的导入测试"""
import sys
sys.path.insert(0, 'd:/zengxinke/workspace/SubStation_AI_Poject/service')

try:
    from client_app import InferenceClient
    print("✅ 导入成功")
    client = InferenceClient()
    print("✅ 实例化成功")
    print(f"Base URL: {client.base_url}")
except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()
