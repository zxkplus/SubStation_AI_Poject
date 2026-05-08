@echo off
echo ========================================
echo  SubStation AI 服务端依赖安装
echo ========================================
echo.

echo [1/3] 安装 uvicorn 和 fastapi...
python -m pip install uvicorn[standard] fastapi pydantic

echo.
echo [2/3] 验证安装...
python -c "import uvicorn; import fastapi; print('✅ 安装成功')"

echo.
echo [3/3] 启动服务...
echo.
echo 提示: 服务将在 http://localhost:8000 启动
echo       API文档: http://localhost:8000/docs
echo       按 Ctrl+C 停止服务
echo.

python -m uvicorn service.app:app --host 0.0.0.0 --port 8000 --reload
