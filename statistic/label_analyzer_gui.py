#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集标签分析可视化工具
基于 count_labels.py 的功能，提供图形用户界面进行数据集标签统计和分析
"""

import os
import sys
import json
import threading
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端支持GUI

# 配置Matplotlib中文字体支持
def configure_matplotlib_chinese():
    """配置Matplotlib支持中文显示

    通过扫描 matplotlib 字体管理器中的实际可用字体来匹配合适的中文字体，
    避免因字体名称不匹配导致中文显示为方框。
    """
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    # 清除 matplotlib 字体缓存，确保能发现系统新安装的字体
    try:
        fm._load_fontmanager(try_read_cache=False)
    except Exception:
        pass

    # 从 matplotlib 已注册的字体中，筛选出可能支持中文的字体
    CHINESE_KEYWORDS = [
        'CJK', 'Hei', 'Song', 'Ming', 'Kai', 'Fang', 'Yuan',
        'UKai', 'UMing', 'WenQuanYi', 'SimHei', 'SimSun',
        'YaHei', 'NSimSun', 'STSong', 'KaiTi', 'Microsoft',
        'Noto Sans', 'Noto Serif',
    ]

    available_cjk = []
    for f in fm.fontManager.ttflist:
        name = f.name
        if any(kw in name for kw in CHINESE_KEYWORDS):
            available_cjk.append(name)

    # 去重，保持顺序
    seen = set()
    available_cjk = [f for f in available_cjk if not (f in seen or seen.add(f))]

    # 按优先级排序：首选 Linux 常见字体，然后是 Windows 系统字体
    priority = [
        'WenQuanYi Micro Hei',   # 文泉驿微米黑 (Linux)
        'WenQuanYi Zen Hei',     # 文泉驿正黑 (Linux)
        'Noto Sans CJK SC',      # Noto 简体中文 (跨平台)
        'Noto Sans CJK JP',      # Noto 日文 (含汉字，常见于 Linux)
        'Noto Serif CJK SC',     # Noto 衬线简体中文
        'AR PL UKai CN',         # 文鼎PL中楷 (Linux)
        'AR PL UMing CN',        # 文鼎PL明体 (Linux)
        'SimHei',                # 黑体 (Windows)
        'Microsoft YaHei',       # 微软雅黑 (Windows)
        'SimSun',                # 宋体 (Windows)
        'KaiTi',                 # 楷体 (Windows)
        'FangSong',              # 仿宋 (Windows)
        'STSong',                # 华文宋体 (Windows)
    ]

    # 构建最终尝试列表: 可用字体(按优先级) + 所有CJK字体 + 优先级列表(兜底)
    ordered_fonts = [f for f in priority if f in available_cjk]
    ordered_fonts += [f for f in available_cjk if f not in ordered_fonts]
    ordered_fonts += [f for f in priority if f not in ordered_fonts]

    configured_font = None
    for font in ordered_fonts:
        try:
            # 使用 font_manager.findfont 验证字体是否真正可用
            font_path = fm.findfont(fm.FontProperties(family=font), fallback_to_default=False)
            if font_path:
                plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
                plt.rcParams['axes.unicode_minus'] = False
                configured_font = font
                print(f"成功配置 Matplotlib 中文字体: {font}")
                break
        except Exception:
            continue

    if not configured_font:
        plt.rcParams['axes.unicode_minus'] = False
        print("警告: 未找到合适的中文字体，图表中文可能显示为方框")

    return configured_font

def configure_tkinter_chinese(root):
    """配置 Tkinter 默认字体支持中文显示

    为 ttk 主题设置默认中文字体，并为常见组件设定合适的字体，
    避免中文在界面中显示为方框。
    """
    import tkinter.font as tkfont

    # 获取系统中可用的 Tkinter 字体
    available_fonts = set(tkfont.families(root))

    # 按优先级选择中文字体
    tk_chinese_candidates = [
        'fangsong ti',       # 仿宋 (Linux)
        'song ti',           # 宋体 (Linux)
        'SimHei',            # 黑体 (Windows)
        'Microsoft YaHei',   # 微软雅黑 (Windows)
        'SimSun',            # 宋体 (Windows)
        'KaiTi',             # 楷体 (Windows)
        'FangSong',          # 仿宋 (Windows)
        'Noto Sans CJK SC',  # Noto 简体中文
    ]

    chosen_font = None
    for font_name in tk_chinese_candidates:
        if font_name.lower() in {f.lower() for f in available_fonts} or font_name in available_fonts:
            chosen_font = font_name
            break

    if chosen_font:
        # 设置 ttk 主题默认字体
        style = ttk.Style(root)
        default_font = (chosen_font, 10)
        style.configure('.', font=default_font)
        style.configure('TLabelframe.Label', font=(chosen_font, 10))
        style.configure('TLabel', font=(chosen_font, 10))
        style.configure('TButton', font=(chosen_font, 10))
        style.configure('TCheckbutton', font=(chosen_font, 10))
        style.configure('TNotebook.Tab', font=(chosen_font, 10))
        print(f"成功配置 Tkinter 中文字体: {chosen_font}")
        return chosen_font
    else:
        print("警告: 未找到 Tkinter 中文字体，界面中文可能显示为方框")
        return None


# 在导入matplotlib.pyplot之前配置字体
configure_matplotlib_chinese()

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import Counter, defaultdict
import orjson

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 颜色表用于不同类别
COLORS = [
    (255, 0, 0),      # 红色
    (0, 255, 0),      # 绿色
    (0, 0, 255),      # 蓝色
    (255, 255, 0),    # 黄色
    (255, 0, 255),    # 品红
    (0, 255, 255),    # 青色
    (255, 128, 0),    # 橙色
    (128, 0, 255),    # 紫色
]


class LabelAnalyzerGUI:
    """数据集标签分析可视化工具主类"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("数据集标签分析工具")
        self.root.geometry("1200x850")
        self.root.minsize(800, 650)

        # 配置 Tkinter 中文字体（必须在创建组件之前调用）
        self._chinese_font = configure_tkinter_chinese(root)

        # 初始化变量
        self.dataset_path = ""
        self.output_path = ""
        self.stats_data = None
        self.label_to_files = defaultdict(list)
        self.is_processing = False

        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        """创建GUI界面组件"""
        # 创建主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # 顶部控制面板
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # 数据集路径选择
        ttk.Label(control_frame, text="数据集路径:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.path_var = tk.StringVar()
        path_entry = ttk.Entry(control_frame, textvariable=self.path_var, width=50)
        path_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(control_frame, text="浏览...", command=self.browse_dataset).grid(row=0, column=2)
        
        # 输出路径选择
        ttk.Label(control_frame, text="输出目录:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=(5, 0))
        self.output_var = tk.StringVar()
        output_entry = ttk.Entry(control_frame, textvariable=self.output_var, width=50)
        output_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=(5, 0))
        ttk.Button(control_frame, text="浏览...", command=self.browse_output).grid(row=1, column=2, pady=(5, 0))
        
        # 参数设置区域 - 使用Notebook分页组织
        param_notebook = ttk.Notebook(control_frame)
        param_notebook.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 基本参数页面
        basic_frame = ttk.Frame(param_notebook)
        param_notebook.add(basic_frame, text="基本参数")
        basic_frame.columnconfigure(1, weight=1)
        
        ttk.Label(basic_frame, text="进程数:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.workers_var = tk.IntVar(value=os.cpu_count() or 4)
        workers_spin = ttk.Spinbox(basic_frame, from_=1, to=32, textvariable=self.workers_var, width=10)
        workers_spin.grid(row=0, column=1, sticky=tk.W, pady=2)
        ttk.Label(basic_frame, text="(并行处理标注文件的数量)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        ttk.Label(basic_frame, text="采样数量:").grid(row=1, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.sample_var = tk.IntVar(value=5)
        sample_spin = ttk.Spinbox(basic_frame, from_=1, to=50, textvariable=self.sample_var, width=10)
        sample_spin.grid(row=1, column=1, sticky=tk.W, pady=2)
        ttk.Label(basic_frame, text="(每个类别随机采样的文件数量)").grid(row=1, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # 过滤参数页面
        filter_frame = ttk.Frame(param_notebook)
        param_notebook.add(filter_frame, text="过滤参数")
        filter_frame.columnconfigure(1, weight=1)
        
        ttk.Label(filter_frame, text="忽略标签:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5), pady=2)
        self.ignore_var = tk.StringVar(value="通用-不识别")
        ignore_entry = ttk.Entry(filter_frame, textvariable=self.ignore_var, width=30)
        ignore_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 5), pady=2)
        ttk.Label(filter_frame, text="(多个标签用逗号分隔)").grid(row=0, column=2, sticky=tk.W, padx=(10, 0), pady=2)
        
        # 输出选项页面
        output_frame = ttk.Frame(param_notebook)
        param_notebook.add(output_frame, text="输出选项")
        output_frame.columnconfigure(1, weight=1)
        
        self.save_report_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(output_frame, text="保存统计报告", variable=self.save_report_var).grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(output_frame, text="(自动保存为 label_statistics_时间戳.txt)").grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        self.create_sample_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(output_frame, text="创建可视化样本", variable=self.create_sample_var).grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Label(output_frame, text="(生成带标注的可视化图片)").grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # 操作按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=(15, 0))
        self.analyze_btn = ttk.Button(button_frame, text="开始分析", command=self.start_analysis)
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.copy_btn = ttk.Button(button_frame, text="按类别整理数据集", command=self.start_copy_dataset, state=tk.DISABLED)
        self.copy_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.export_btn = ttk.Button(button_frame, text="导出报告", command=self.export_report, state=tk.DISABLED)
        self.export_btn.pack(side=tk.LEFT)
        self.clear_btn = ttk.Button(button_frame, text="清除结果", command=self.clear_results)
        self.clear_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # 进度条
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪 - 请选择数据集路径和输出目录")
        status_label = ttk.Label(control_frame, textvariable=self.status_var, foreground="blue")
        status_label.grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # 主内容区域 - 使用Notebook分页
        notebook = ttk.Notebook(main_frame)
        notebook.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 统计结果页面
        stats_frame = ttk.Frame(notebook)
        notebook.add(stats_frame, text="统计结果")
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
        
        # 图表区域
        chart_frame = ttk.Frame(stats_frame)
        chart_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        self.figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 详细信息页面
        detail_frame = ttk.Frame(notebook)
        notebook.add(detail_frame, text="详细信息")
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(0, weight=1)
        
        # 文本显示区域
        # 使用中文字体作为等宽文本的显示字体（避免中文方框乱码）
        text_font = (self._chinese_font, 10) if self._chinese_font else ("sans-serif", 10)
        self.text_area = scrolledtext.ScrolledText(detail_frame, wrap=tk.WORD, font=text_font)
        self.text_area.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.text_area.config(state=tk.DISABLED)
        
        # 文件列表页面
        files_frame = ttk.Frame(notebook)
        notebook.add(files_frame, text="文件列表")
        files_frame.columnconfigure(0, weight=1)
        files_frame.rowconfigure(0, weight=1)
        
        # 标签选择下拉框
        label_frame = ttk.Frame(files_frame)
        label_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 5))
        label_frame.columnconfigure(1, weight=1)
        
        ttk.Label(label_frame, text="选择标签:").grid(row=0, column=0, sticky=tk.W, padx=(0, 5))
        self.label_combo = ttk.Combobox(label_frame, state="readonly")
        self.label_combo.grid(row=0, column=1, sticky=(tk.W, tk.E))
        self.label_combo.bind('<<ComboboxSelected>>', self.on_label_selected)
        
        # 文件列表
        listbox_font = (self._chinese_font, 10) if self._chinese_font else ("sans-serif", 10)
        self.file_listbox = tk.Listbox(files_frame, font=listbox_font)
        self.file_listbox.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.file_listbox.configure(yscrollcommand=scrollbar.set)
        
        # 设置默认输出路径
        self.update_default_output_path()
        
    def browse_dataset(self):
        """浏览选择数据集目录"""
        directory = filedialog.askdirectory(title="选择数据集根目录")
        if directory:
            self.path_var.set(directory)
            self.dataset_path = directory
            self.update_default_output_path()
            
    def browse_output(self):
        """浏览选择输出目录"""
        directory = filedialog.askdirectory(title="选择输出目录")
        if directory:
            self.output_var.set(directory)
            self.output_path = directory
            
    def update_default_output_path(self):
        """根据数据集路径更新默认输出路径"""
        if self.path_var.get():
            dataset_dir = Path(self.path_var.get())
            default_output = dataset_dir.parent / f"{dataset_dir.name}_analysis"
            self.output_var.set(str(default_output))
            self.output_path = str(default_output)
        else:
            self.output_var.set("")
            self.output_path = ""
            
    def start_analysis(self):
        """开始分析数据集"""
        if not self.path_var.get():
            messagebox.showerror("错误", "请先选择数据集路径")
            return
            
        if not self.output_var.get():
            messagebox.showerror("错误", "请先选择输出目录")
            return
            
        if self.is_processing:
            messagebox.showinfo("提示", "正在处理中，请稍候...")
            return
            
        self.dataset_path = self.path_var.get()
        self.output_path = self.output_var.get()
        
        if not os.path.exists(self.dataset_path):
            messagebox.showerror("错误", "指定的数据集路径不存在")
            return
            
        # 创建输出目录
        os.makedirs(self.output_path, exist_ok=True)
            
        # 在后台线程中执行分析
        self.is_processing = True
        self.analyze_btn.config(state=tk.DISABLED)
        self.status_var.set("正在扫描标注文件...")
        self.progress_var.set(0)
        
        thread = threading.Thread(target=self.analyze_dataset, daemon=True)
        thread.start()
        
    def analyze_dataset(self):
        """分析数据集（在后台线程中执行）"""
        try:
            # 收集所有标注文件
            annotation_files = []
            for root, _, files in os.walk(self.dataset_path):
                for filename in files:
                    if filename.endswith(('.annotate', '.json')):
                        annotation_files.append(os.path.join(root, filename))
            
            if not annotation_files:
                self.root.after(0, lambda: messagebox.showerror("错误", "未找到标注文件"))
                self.reset_ui()
                return
                
            total_files = len(annotation_files)
            self.root.after(0, lambda: self.status_var.set(f"共发现 {total_files} 个标注文件，开始分析..."))
            
            # 多进程处理（简化版，使用单线程避免复杂性）
            annotate_counter = Counter()
            json_counter = Counter()
            total_counter = Counter()
            label_to_files = defaultdict(list)
            error_count = 0
            
            for i, filepath in enumerate(annotation_files):
                try:
                    labels, is_annotate, _ = self.process_file(filepath)
                    for label in labels:
                        if is_annotate:
                            annotate_counter[label] += 1
                        else:
                            json_counter[label] += 1
                        total_counter[label] += 1
                        label_to_files[label].append(filepath)
                    
                    # 更新进度
                    progress = (i + 1) / total_files * 100
                    self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    if i % 10 == 0:  # 每10个文件更新一次状态
                        self.root.after(0, lambda c=i+1, t=total_files: 
                                      self.status_var.set(f"已处理 {c}/{t} 个文件"))
                        
                except Exception as e:
                    error_count += 1
                    print(f"处理文件失败 {filepath}: {e}")
            
            # 保存结果
            self.stats_data = {
                'total_files': total_files,
                'annotate_counter': dict(annotate_counter),
                'json_counter': dict(json_counter),
                'total_counter': dict(total_counter),
                'error_count': error_count,
                'workers_used': self.workers_var.get(),
                'sample_count': self.sample_var.get(),
                'ignore_labels': [label.strip() for label in self.ignore_var.get().split(',') if label.strip()]
            }
            self.label_to_files = label_to_files
            
            # 自动保存报告（如果启用）
            if self.save_report_var.get():
                self.auto_save_report()
            
            # 创建可视化样本（如果启用）
            if self.create_sample_var.get():
                self.create_visual_samples()
            
            # 更新UI
            self.root.after(0, self.update_results)
            self.root.after(0, lambda: self.status_var.set(f"分析完成！结果保存到: {self.output_path}"))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"分析过程中发生错误: {str(e)}"))
            self.root.after(0, self.reset_ui)
            
    def process_file(self, filepath: str):
        """处理单个文件（复制自count_labels.py）"""
        try:
            with open(filepath, 'rb') as f:
                data = orjson.loads(f.read())

            filename = os.path.basename(filepath)
            labels = []

            if filename.endswith('.annotate'):
                for entity in data.get('entities', []):
                    label = entity.get('label')
                    if label:
                        labels.append(label)
        
            elif filename.endswith('.json'):
                for roi in data.get('rois', []):
                    name = roi.get('name')
                    if name:
                        labels.append(name)
        
            return labels, filename.endswith('.annotate'), filepath
    
        except Exception:
            return [], False, filepath
            
    def auto_save_report(self):
        """自动保存统计报告"""
        if not self.stats_data:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_path, f"label_statistics_{timestamp}.txt")
        
        # 生成报告内容
        content = "==================== 数据集标注类别统计报告 ====================\n"
        content += f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"数据集路径: {os.path.abspath(self.dataset_path)}\n"
        content += f"输出目录: {os.path.abspath(self.output_path)}\n"
        content += f"总文件数: {self.stats_data['total_files']}  (失败: {self.stats_data['error_count']})\n"
        content += f"使用的进程数: {self.stats_data['workers_used']}\n"
        content += f"采样数量: {self.stats_data['sample_count']}\n"
        content += f"忽略标签: {', '.join(self.stats_data['ignore_labels']) if self.stats_data['ignore_labels'] else '无'}\n"
        content += "=" * 60 + "\n\n"
        
        # .annotate 统计
        content += "【1. .annotate 文件统计】\n"
        annotate_counter = self.stats_data['annotate_counter']
        if annotate_counter:
            for label, count in sorted(annotate_counter.items(), key=lambda x: x[1], reverse=True):
                content += f"{label:<30} : {count:>6} 次\n"
            content += f"{'-'*50}\n小计: {sum(annotate_counter.values())} 个标签\n\n"
        else:
            content += "（未找到 .annotate 文件）\n\n"
        
        # .json 统计
        content += "【2. .json 文件统计】\n"
        json_counter = self.stats_data['json_counter']
        if json_counter:
            for name, count in sorted(json_counter.items(), key=lambda x: x[1], reverse=True):
                content += f"{name:<30} : {count:>6} 次\n"
            content += f"{'-'*50}\n小计: {sum(json_counter.values())} 个标签\n\n"
        else:
            content += "（未找到 .json 文件）\n\n"
        
        # 汇总统计
        content += "【3. 汇总统计（.annotate + .json）】\n"
        total_counter = self.stats_data['total_counter']
        if total_counter:
            for label, count in sorted(total_counter.items(), key=lambda x: x[1], reverse=True):
                content += f"{label:<30} : {count:>6} 次\n"
            content += f"{'-'*50}\n总计: {sum(total_counter.values())} 个标签\n"
        else:
            content += "（未找到任何标注文件）\n"
        
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            print(f"自动保存报告失败: {e}")
            
    def create_visual_samples(self):
        """创建可视化样本"""
        try:
            sample_dir = os.path.join(self.output_path, "visual_samples")
            os.makedirs(sample_dir, exist_ok=True)
            
            sampled_count = 0
            sample_per_class = self.sample_var.get()
            
            for label, files in self.label_to_files.items():
                if not files:
                    continue
                
                # 随机选择样本
                import random
                selected_files = random.sample(files, min(sample_per_class, len(files)))
                
                # 创建类别子文件夹
                label_dir = os.path.join(sample_dir, label)
                os.makedirs(label_dir, exist_ok=True)
                
                for ann_file in selected_files:
                    if not ann_file.endswith(('.json', '.annotate')):
                        continue
                    
                    # 找到对应的图片文件
                    image_path = self._find_image_for_annotation(ann_file)
                    if not image_path:
                        continue
                    
                    # 生成可视化图片（简化版，只复制文件）
                    import shutil
                    vis_filename = f"{Path(ann_file).stem}_sample{sampled_count}.jpg"
                    vis_path = os.path.join(label_dir, vis_filename)
                    
                    try:
                        shutil.copy2(image_path, vis_path)
                        sampled_count += 1
                    except Exception as e:
                        print(f"复制样本失败 {ann_file}: {e}")
            
            if sampled_count > 0:
                print(f"创建了 {sampled_count} 个可视化样本到 {sample_dir}")
                
        except Exception as e:
            print(f"创建可视化样本失败: {e}")
            
    def update_results(self):
        """更新结果显示"""
        if not self.stats_data:
            return
            
        # 更新图表
        self.update_chart()
        
        # 更新详细信息文本
        self.update_detail_text()
        
        # 更新文件列表下拉框
        labels = list(self.label_to_files.keys())
        self.label_combo['values'] = labels
        if labels:
            self.label_combo.set(labels[0])
            self.update_file_list(labels[0])
        
        # 启用其他按钮
        self.copy_btn.config(state=tk.NORMAL)
        self.export_btn.config(state=tk.NORMAL)
        self.reset_ui()
        
    def update_chart(self):
        """更新统计图表"""
        self.figure.clear()
        
        # 获取前15个最频繁的标签
        total_counter = Counter(self.stats_data['total_counter'])
        top_labels = total_counter.most_common(15)
        
        if not top_labels:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, '没有找到标签数据', ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
            return
            
        labels, counts = zip(*top_labels)
        
        # 创建柱状图
        ax = self.figure.add_subplot(111)
        bars = ax.bar(range(len(labels)), counts, color='steelblue')
        ax.set_xlabel('标签名称')
        ax.set_ylabel('出现次数')
        ax.set_title('标签频率统计（前15名）')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # 在柱子上显示数值
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   str(count), ha='center', va='bottom')
        
        self.figure.tight_layout()
        self.canvas.draw()
        
    def update_detail_text(self):
        """更新详细信息文本"""
        self.text_area.config(state=tk.NORMAL)
        self.text_area.delete(1.0, tk.END)
        
        # 写入统计摘要
        content = "==================== 数据集标注类别统计报告 ====================\n"
        content += f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        content += f"数据集路径: {os.path.abspath(self.dataset_path)}\n"
        content += f"输出目录: {os.path.abspath(self.output_path)}\n"
        content += f"总文件数: {self.stats_data['total_files']}  (失败: {self.stats_data['error_count']})\n"
        content += f"使用的进程数: {self.stats_data['workers_used']}\n"
        content += f"采样数量: {self.stats_data['sample_count']}\n"
        content += f"忽略标签: {', '.join(self.stats_data['ignore_labels']) if self.stats_data['ignore_labels'] else '无'}\n"
        content += "=" * 60 + "\n\n"
        
        # .annotate 统计
        content += "【1. .annotate 文件统计】\n"
        annotate_counter = self.stats_data['annotate_counter']
        if annotate_counter:
            for label, count in sorted(annotate_counter.items(), key=lambda x: x[1], reverse=True):
                content += f"{label:<30} : {count:>6} 次\n"
            content += f"{'-'*50}\n小计: {sum(annotate_counter.values())} 个标签\n\n"
        else:
            content += "（未找到 .annotate 文件）\n\n"
        
        # .json 统计
        content += "【2. .json 文件统计】\n"
        json_counter = self.stats_data['json_counter']
        if json_counter:
            for name, count in sorted(json_counter.items(), key=lambda x: x[1], reverse=True):
                content += f"{name:<30} : {count:>6} 次\n"
            content += f"{'-'*50}\n小计: {sum(json_counter.values())} 个标签\n\n"
        else:
            content += "（未找到 .json 文件）\n\n"
        
        # 汇总统计
        content += "【3. 汇总统计（.annotate + .json）】\n"
        total_counter = self.stats_data['total_counter']
        if total_counter:
            for label, count in sorted(total_counter.items(), key=lambda x: x[1], reverse=True):
                content += f"{label:<30} : {count:>6} 次\n"
            content += f"{'-'*50}\n总计: {sum(total_counter.values())} 个标签\n"
        else:
            content += "（未找到任何标注文件）\n"
        
        self.text_area.insert(tk.END, content)
        self.text_area.config(state=tk.DISABLED)
        
    def on_label_selected(self, event=None):
        """当选择标签时更新文件列表"""
        selected_label = self.label_combo.get()
        if selected_label:
            self.update_file_list(selected_label)
            
    def update_file_list(self, label):
        """更新文件列表"""
        self.file_listbox.delete(0, tk.END)
        files = self.label_to_files.get(label, [])
        for file_path in files:
            self.file_listbox.insert(tk.END, file_path)
            
    def start_copy_dataset(self):
        """开始按类别整理数据集"""
        if not self.dataset_path or not self.stats_data:
            messagebox.showerror("错误", "请先完成数据分析")
            return
            
        # 如果没有指定输出目录，使用默认的整理目录
        if not self.output_path:
            target_dir = filedialog.askdirectory(title="选择目标目录")
            if not target_dir:
                return
        else:
            target_dir = os.path.join(self.output_path, "organized_dataset")
            
        # 在后台线程中执行整理
        self.is_processing = True
        self.copy_btn.config(state=tk.DISABLED)
        self.status_var.set("正在整理数据集...")
        self.progress_var.set(0)
        
        thread = threading.Thread(
            target=self.copy_dataset_thread, 
            args=(target_dir,),
            daemon=True
        )
        thread.start()
        
    def copy_dataset_thread(self, target_dir):
        """整理数据集线程"""
        try:
            ignore_labels = [label.strip() for label in self.ignore_var.get().split(',') if label.strip()]
            if not ignore_labels:
                ignore_labels = ['通用-不识别']
                
            # 简化版整理逻辑
            copied_counts = defaultdict(int)
            skipped_count = 0
            error_count = 0
            
            target_root = Path(target_dir)
            target_root.mkdir(parents=True, exist_ok=True)
            
            # 获取所有文件
            all_files = []
            for label, files in self.label_to_files.items():
                if label not in ignore_labels:
                    all_files.extend([(label, f) for f in files])
            
            total_files = len(all_files)
            
            for i, (label, annotation_path) in enumerate(all_files):
                try:
                    # 找到对应的图片文件
                    image_path = self._find_image_for_annotation(annotation_path)
                    if image_path is None:
                        error_count += 1
                        continue
                    
                    # 创建目标目录
                    dest_dir = target_root / label
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 复制文件
                    import shutil
                    shutil.copy2(image_path, dest_dir / os.path.basename(image_path))
                    shutil.copy2(annotation_path, dest_dir / os.path.basename(annotation_path))
                    
                    copied_counts[label] += 1
                    
                    # 更新进度
                    progress = (i + 1) / total_files * 100
                    self.root.after(0, lambda p=progress: self.progress_var.set(p))
                    if i % 10 == 0:
                        self.root.after(0, lambda c=i+1, t=total_files: 
                                      self.status_var.set(f"整理中 {c}/{t}"))
                        
                except Exception as e:
                    error_count += 1
                    print(f"复制失败 {annotation_path}: {e}")
            
            # 完成整理
            self.root.after(0, lambda: self.status_var.set(f"整理完成！共复制 {sum(copied_counts.values())} 个文件到 {target_dir}"))
            self.root.after(0, lambda: messagebox.showinfo("完成", 
                f"数据集整理完成！\n"
                f"成功复制: {sum(copied_counts.values())} 个文件\n"
                f"失败: {error_count} 个文件\n"
                f"目标目录: {target_dir}"))
                
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("错误", f"整理过程中发生错误: {str(e)}"))
        finally:
            self.root.after(0, self.reset_ui)
            
    def _find_image_for_annotation(self, annotation_path: str):
        """查找标注文件对应的图片文件"""
        annotation_path = Path(annotation_path)
        ann_dir = annotation_path.parent
        ann_name = annotation_path.name
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        stems = []
        if ann_name.endswith('.annotate'):
            stems.append(ann_name[:-len('.annotate')])
        else:
            stems.append(annotation_path.stem)

        for stem in stems:
            # 如果已包含图片后缀，则直接尝试原名
            for ext in image_exts:
                if stem.lower().endswith(ext):
                    candidate = ann_dir / stem
                    if candidate.exists():
                        return str(candidate)
            # 否则尝试添加常见后缀
            for ext in image_exts:
                candidate = ann_dir / f"{stem}{ext}"
                if candidate.exists():
                    return str(candidate)

        return None
        
    def export_report(self):
        """导出统计报告"""
        if not self.stats_data:
            messagebox.showerror("错误", "请先完成数据分析")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"label_statistics_manual_{timestamp}.txt"
        file_path = filedialog.asksaveasfilename(
            title="保存统计报告",
            defaultextension=".txt",
            initialfile=default_filename,
            filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.text_area.get(1.0, tk.END))
                messagebox.showinfo("成功", f"报告已保存到:\n{file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"保存报告失败: {str(e)}")
                
    def clear_results(self):
        """清除分析结果"""
        if self.stats_data and messagebox.askyesno("确认", "确定要清除所有分析结果吗？"):
            self.stats_data = None
            self.label_to_files.clear()
            self.text_area.config(state=tk.NORMAL)
            self.text_area.delete(1.0, tk.END)
            self.text_area.config(state=tk.DISABLED)
            self.figure.clear()
            self.canvas.draw()
            self.label_combo['values'] = []
            self.label_combo.set('')
            self.file_listbox.delete(0, tk.END)
            self.copy_btn.config(state=tk.DISABLED)
            self.export_btn.config(state=tk.DISABLED)
            self.status_var.set("结果已清除 - 请选择数据集重新分析")
            
    def reset_ui(self):
        """重置UI状态"""
        self.is_processing = False
        self.analyze_btn.config(state=tk.NORMAL)
        self.copy_btn.config(state=tk.NORMAL if self.stats_data else tk.DISABLED)


def main():
    """主函数"""
    root = tk.Tk()
    app = LabelAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()