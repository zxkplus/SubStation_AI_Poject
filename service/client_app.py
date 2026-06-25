"""
SubStation AI 实例分割客户端应用

功能：
1. 选择图片并显示
2. 在图片上划定ROI区域
3. 调用推理服务接口
4. 展示实例分割结果（mask轮廓和边界框）
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import base64
import io
import requests
import json
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path


class InferenceClient:
    """推理服务客户端"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.infer_url = f"{base_url}/infer"
    
    def predict(self,
                image_path: str,
                roi: dict,
                weights_path: str = "yolov8n-seg.pt",
                conf_threshold: float = 0.25,
                img_size: int = 640,
                device: str = "cpu",
                retina_masks: bool = False) -> dict:
        """
        调用推理接口
        
        Args:
            image_path: 图片路径
            roi: ROI区域 {"x1": int, "y1": int, "x2": int, "y2": int}
            weights_path: 模型权重路径
            conf_threshold: 置信度阈值
            img_size: 图像尺寸
            device: 设备 (cpu/cuda)
            
        Returns:
            推理结果字典
        """
        # 读取并编码图片
        with open(image_path, 'rb') as f:
            image_bytes = f.read()
        
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # 构建请求数据
        request_data = {
            "image_base64": image_base64,
            "rois": [roi],
            "weights_path": weights_path,
            "conf_threshold": conf_threshold,
            "img_size": img_size,
            "device": device,
            "retina_masks": retina_masks,
        }
        
        # 发送请求
        try:
            response = requests.post(
                self.infer_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"推理请求失败: {str(e)}")


class ROIDrawer:
    """ROI区域绘制器"""
    
    def __init__(self, canvas: tk.Canvas):
        self.canvas = canvas
        self.start_x = None
        self.start_y = None
        self.current_rect = None
        self.roi = None  # 最终ROI坐标
        
    def start_drawing(self, event):
        """开始绘制ROI"""
        self.start_x = event.x
        self.start_y = event.y
        if self.current_rect:
            self.canvas.delete(self.current_rect)
        self.current_rect = self.canvas.create_rectangle(
            self.start_x, self.start_y, self.start_x, self.start_y,
            outline='red', width=2
        )
        
    def update_drawing(self, event):
        """更新绘制"""
        if self.current_rect:
            self.canvas.coords(
                self.current_rect,
                self.start_x, self.start_y, event.x, event.y
            )
            
    def finish_drawing(self, event):
        """完成绘制，保存ROI坐标"""
        if self.start_x is not None:
            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)
            
            # 确保ROI有效
            if x2 - x1 > 10 and y2 - y1 > 10:
                self.roi = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                return True
        return False
    
    def clear(self):
        """清除绘制的ROI"""
        if self.current_rect:
            self.canvas.delete(self.current_rect)
            self.current_rect = None
        self.roi = None


class ResultRenderer:
    """结果渲染器"""
    
    @staticmethod
    def draw_results_on_image(pil_image: Image.Image, 
                             inference_result: dict,
                             class_colors: Optional[dict] = None) -> Image.Image:
        """
        在图片上绘制推理结果
        
        Args:
            pil_image: PIL图像对象
            inference_result: 推理结果
            class_colors: 类别颜色映射
            
        Returns:
            绘制后的PIL图像
        """
        if class_colors is None:
            # 默认颜色表
            class_colors = {
                0: (255, 0, 0),     # 红色
                1: (0, 255, 0),     # 绿色
                2: (0, 0, 255),     # 蓝色
                3: (255, 255, 0),   # 黄色
                4: (255, 0, 255),   # 紫色
                5: (0, 255, 255),   # 青色
            }
        
        # 创建可绘制的副本
        result_image = pil_image.copy()
        draw = ImageDraw.Draw(result_image, 'RGBA')
        
        # 遍历所有ROI结果
        for roi_result in inference_result.get('results', []):
            detections = roi_result.get('detections', [])
            
            for detection in detections:
                class_id = detection['class_id']
                confidence = detection['confidence']
                bbox = detection['bbox']
                contours = detection['contours']
                
                # 获取颜色
                color = class_colors.get(class_id, (255, 255, 255))
                color_with_alpha = (*color, 128)  # 半透明
                
                # 绘制mask轮廓
                for contour in contours:
                    points = [(p[0], p[1]) for p in contour['points']]
                    if len(points) > 2:
                        # 填充mask区域
                        draw.polygon(points, fill=color_with_alpha)
                        # 绘制轮廓线
                        draw.line(points + [points[0]], fill=color, width=2)
                
                # 绘制边界框
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # 绘制标签
                label = f"Class {class_id}: {confidence:.2f}"
                # 简单的文本绘制（需要更复杂的实现来支持中文）
                text_bbox = draw.textbbox((x1, y1 - 20), label)
                draw.rectangle(text_bbox, fill=(*color, 200))
                draw.text((x1, y1 - 20), label, fill=(255, 255, 255))
        
        return result_image


class SubstationClientApp:
    """主应用程序"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("SubStation AI 实例分割客户端")
        self.root.geometry("1200x800")
        
        # 初始化组件
        self.client = InferenceClient()
        self.image_path = None
        self.original_image = None
        self.display_image = None
        self.result_image = None
        self.image_scale = 1.0  # 添加缩放比例属性
        self.image_offset_x = 0  # 添加图片在Canvas中的X偏移
        self.image_offset_y = 0  # 添加图片在Canvas中的Y偏移
        
        # 创建UI
        self._create_ui()
        
    def _create_ui(self):
        """创建用户界面"""
        # 顶部控制面板
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)
        
        # 服务器地址
        ttk.Label(control_frame, text="服务器地址:").pack(side=tk.LEFT, padx=5)
        self.server_url_var = tk.StringVar(value="http://localhost:8000")
        server_entry = ttk.Entry(control_frame, textvariable=self.server_url_var, width=30)
        server_entry.pack(side=tk.LEFT, padx=5)
        
        # 选择图片按钮
        select_btn = ttk.Button(control_frame, text="选择图片", command=self._select_image)
        select_btn.pack(side=tk.LEFT, padx=5)
        
        # 推理按钮
        self.infer_btn = ttk.Button(control_frame, text="执行推理", command=self._run_inference, state=tk.DISABLED)
        self.infer_btn.pack(side=tk.LEFT, padx=5)
        
        # 清除ROI按钮
        clear_btn = ttk.Button(control_frame, text="清除ROI", command=self._clear_roi)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 参数设置框架
        param_frame = ttk.LabelFrame(self.root, text="推理参数", padding="10")
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 模型权重
        ttk.Label(param_frame, text="模型权重:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.weights_var = tk.StringVar(value="yolov8n-seg.pt")
        ttk.Entry(param_frame, textvariable=self.weights_var, width=30).grid(row=0, column=1, padx=5, pady=2)
        
        # 置信度阈值
        ttk.Label(param_frame, text="置信度阈值:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=2)
        self.conf_var = tk.DoubleVar(value=0.25)
        ttk.Spinbox(param_frame, from_=0.0, to=1.0, increment=0.05, 
                   textvariable=self.conf_var, width=10).grid(row=0, column=3, padx=5, pady=2)
        
        # 图像尺寸
        ttk.Label(param_frame, text="图像尺寸:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.img_size_var = tk.IntVar(value=640)
        ttk.Spinbox(param_frame, from_=320, to=1920, increment=32, 
                   textvariable=self.img_size_var, width=10).grid(row=1, column=1, padx=5, pady=2)
        
        # 设备选择
        ttk.Label(param_frame, text="设备:").grid(row=1, column=2, sticky=tk.W, padx=5, pady=2)
        self.device_var = tk.StringVar(value="cpu")
        device_combo = ttk.Combobox(param_frame, textvariable=self.device_var,
                                   values=["cpu", "cuda"], width=10, state="readonly")
        device_combo.grid(row=1, column=3, padx=5, pady=2)

        # retina_masks 高分辨率掩码
        self.retina_masks_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text="高分辨率掩码 (retina_masks)",
                        variable=self.retina_masks_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # 图像显示区域
        image_frame = ttk.Frame(self.root)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建Canvas用于显示图片和绘制ROI
        self.canvas = tk.Canvas(image_frame, bg='gray', cursor='crosshair')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 绑定鼠标事件
        self.roi_drawer = ROIDrawer(self.canvas)
        self.canvas.bind("<ButtonPress-1>", self.roi_drawer.start_drawing)
        self.canvas.bind("<B1-Motion>", self.roi_drawer.update_drawing)
        self.canvas.bind("<ButtonRelease-1>", self._on_roi_complete)
        
        # 状态栏
        self.status_var = tk.StringVar(value="请选择一张图片开始")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
    def _select_image(self):
        """选择图片"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.image_path = file_path
            self._load_and_display_image(file_path)
            self.infer_btn.config(state=tk.NORMAL)
            self.status_var.set(f"已加载图片: {Path(file_path).name}")
            
    def _load_and_display_image(self, image_path: str):
        """加载并显示图片"""
        try:
            # 加载原始图片
            self.original_image = Image.open(image_path)
            
            # 调整图片大小以适应窗口
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width < 10 or canvas_height < 10:
                # Canvas还未初始化，使用默认值
                canvas_width = 1180
                canvas_height = 650
            
            # 计算缩放比例
            img_width, img_height = self.original_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            resized_image = self.original_image.resize((new_width, new_height), Image.LANCZOS)
            self.display_image = resized_image
            
            # 计算图片在Canvas中的偏移量（因为是居中显示）
            self.image_offset_x = (canvas_width - new_width) // 2
            self.image_offset_y = (canvas_height - new_height) // 2
            self.image_scale = scale
            
            # 转换为Tkinter格式
            self.tk_image = ImageTk.PhotoImage(resized_image)
            
            # 在Canvas中显示
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2, 
                canvas_height // 2, 
                image=self.tk_image, 
                anchor=tk.CENTER
            )
            
            # 重置ROI绘制器
            self.roi_drawer.clear()
            
        except Exception as e:
            messagebox.showerror("错误", f"加载图片失败: {str(e)}")
            
    def _on_roi_complete(self, event):
        """ROI绘制完成回调"""
        if self.roi_drawer.finish_drawing(event):
            # 将Canvas坐标转换为原始图片坐标
            if self.original_image and self.image_scale > 0:
                # Canvas坐标 -> 缩放后图片坐标 -> 原始图片坐标
                orig_x1 = max(0, (self.roi_drawer.roi['x1'] - self.image_offset_x) / self.image_scale)
                orig_y1 = max(0, (self.roi_drawer.roi['y1'] - self.image_offset_y) / self.image_scale)
                orig_x2 = min(self.original_image.width, (self.roi_drawer.roi['x2'] - self.image_offset_x) / self.image_scale)
                orig_y2 = min(self.original_image.height, (self.roi_drawer.roi['y2'] - self.image_offset_y) / self.image_scale)
                
                # 确保坐标有效
                if orig_x2 > orig_x1 and orig_y2 > orig_y1:
                    self.roi_drawer.roi = {
                        "x1": int(orig_x1),
                        "y1": int(orig_y1), 
                        "x2": int(orig_x2),
                        "y2": int(orig_y2)
                    }
                    self.status_var.set(f"ROI已划定: ({self.roi_drawer.roi['x1']}, {self.roi_drawer.roi['y1']}) - "
                                      f"({self.roi_drawer.roi['x2']}, {self.roi_drawer.roi['y2']})")
                else:
                    self.roi_drawer.clear()
                    self.status_var.set("ROI区域无效，请重新划定")
            else:
                self.status_var.set("ROI区域太小，请重新划定")
        else:
            self.status_var.set("ROI区域太小，请重新划定")
            
    def _clear_roi(self):
        """清除ROI"""
        self.roi_drawer.clear()
        self.status_var.set("ROI已清除")
        
    def _run_inference(self):
        """执行推理"""
        if not self.image_path:
            messagebox.showwarning("警告", "请先选择一张图片")
            return
        
        if not self.roi_drawer.roi:
            messagebox.showwarning("警告", "请先划定ROI区域")
            return
        
        # 更新客户端URL
        self.client.base_url = self.server_url_var.get()
        self.client.infer_url = f"{self.client.base_url}/infer"
        
        # 显示进度
        self.status_var.set("正在执行推理...")
        self.root.update()
        self.infer_btn.config(state=tk.DISABLED)
        
        try:
            # 调用推理接口
            result = self.client.predict(
                image_path=self.image_path,
                roi=self.roi_drawer.roi,
                weights_path=self.weights_var.get(),
                conf_threshold=self.conf_var.get(),
                img_size=self.img_size_var.get(),
                device=self.device_var.get(),
                retina_masks=self.retina_masks_var.get(),
            )
            
            # 绘制结果
            self._display_results(result)
            
            self.status_var.set(f"推理完成! 检测到 {sum(len(r['detections']) for r in result['results'])} 个目标")
            
        except Exception as e:
            messagebox.showerror("推理错误", f"推理失败: {str(e)}")
            self.status_var.set("推理失败")
        finally:
            self.infer_btn.config(state=tk.NORMAL)
            
    def _display_results(self, inference_result: dict):
        """显示推理结果"""
        if not self.original_image:
            return
        
        try:
            # 在图片上绘制结果
            result_image = ResultRenderer.draw_results_on_image(
                self.original_image, 
                inference_result
            )
            
            # 调整大小以匹配显示
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            img_width, img_height = result_image.size
            scale = min(canvas_width / img_width, canvas_height / img_height, 1.0)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            resized_result = result_image.resize((new_width, new_height), Image.LANCZOS)
            
            # 显示结果
            self.tk_result_image = ImageTk.PhotoImage(resized_result)
            self.canvas.delete("all")
            self.canvas.create_image(
                canvas_width // 2, 
                canvas_height // 2, 
                image=self.tk_result_image, 
                anchor=tk.CENTER
            )
            
        except Exception as e:
            messagebox.showerror("显示错误", f"显示结果失败: {str(e)}")


def main():
    """主函数"""
    root = tk.Tk()
    app = SubstationClientApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
