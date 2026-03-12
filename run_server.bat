@echo off
chcp 65001 >nul
title BTC 实时分析服务器

echo ============================================================
echo   BTC/USDT 实时技术分析服务器
echo   工作目录: %~dp0
echo ============================================================
echo.

cd /d "%~dp0"

:: 检查并安装依赖
python -m pip show flask >nul 2>&1
if %errorlevel% neq 0 (
    echo [安装] 正在安装依赖...
    python -m pip install -r requirements.txt -q
)

echo [启动] 服务器正在初始化，首次加载约需 30-60 秒...
echo [访问] 请在浏览器打开: http://localhost:5000
echo.

:: 自动打开浏览器（延迟3秒等服务器启动）
start "" /b cmd /c "timeout /t 3 >nul && start http://localhost:5000"

python server.py

if %errorlevel% neq 0 (
    echo.
    echo [错误] 服务器异常退出，请检查报错信息
    pause
)
