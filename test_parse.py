#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from statistic.dataset_tools import DatasetFilterTool

def test_parse_json_labels():
    # 创建测试JSON数据
    test_data = {
        "full_sensitivity": 0, 
        "scene_type": "室外", 
        "focus_type": "常规", 
        "image_width": 1920, 
        "image_height": 1080, 
        "point_name": "1#主变压器---1#主变35kV侧避雷器---1#主变35kV侧避雷器B相避雷器引线接头外观.jpg", 
        "rois": [{
            "sensitivity": 0.8, 
            "points": [{"x": 1007, "y": 405}, {"x": 1004, "y": 428}], 
            "name": "线路线缆"
        }]
    }
    
    # 保存到临时文件
    with open('test_temp.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)
    
    try:
        # 测试解析
        tool = DatasetFilterTool()
        labels = tool.parse_json_labels('test_temp.json')
        print(f"解析结果: {labels}")
        print(f"标签数量: {len(labels)}")
        
        if labels:
            print("✅ 解析成功！")
        else:
            print("❌ 解析失败，返回空列表")
            
    finally:
        # 清理临时文件
        import os
        if os.path.exists('test_temp.json'):
            os.remove('test_temp.json')

if __name__ == "__main__":
    test_parse_json_labels()