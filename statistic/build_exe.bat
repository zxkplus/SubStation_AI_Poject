@echo off
REM ============================================================
REM  LabelAnalyzer Windows EXE 本地构建脚本
REM  在 Windows 上运行此脚本，将 label_analyzer_gui.py 打包为 .exe
REM
REM  前提条件：
REM    - 已安装 Python 3.8+
REM    - Python 已添加到 PATH
REM ============================================================

echo ============================================================
echo  LabelAnalyzer - Windows EXE 构建脚本
echo ============================================================
echo.

REM 切换到脚本所在目录
cd /d "%~dp0"

echo [1/3] 安装依赖...
python -m pip install --upgrade pip
python -m pip install -r requirements_gui.txt
python -m pip install pyinstaller
echo.

echo [2/3] 使用 PyInstaller 打包...
pyinstaller ^
  --noconfirm ^
  --onefile ^
  --windowed ^
  --name "LabelAnalyzer" ^
  --hidden-import orjson ^
  --hidden-import tkinter ^
  --hidden-import tkinter.ttk ^
  --hidden-import tkinter.filedialog ^
  --hidden-import tkinter.messagebox ^
  --hidden-import tkinter.scrolledtext ^
  --hidden-import matplotlib.backends.backend_tkagg ^
  --collect-submodules matplotlib ^
  label_analyzer_gui.py

echo.
echo [3/3] 打包完成！
echo.
echo 输出文件: %~dp0dist\LabelAnalyzer.exe
echo.
echo 将 dist\LabelAnalyzer.exe 复制到任意 Windows 机器即可运行。

pause
