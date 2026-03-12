@echo off
chcp 65001 >nul
title BTC/USDT 技术分析

echo ============================================================
echo   BTC/USDT 日线技术分析工具
echo   工作目录: %~dp0
echo ============================================================
echo.

cd /d "%~dp0"

python btc_analysis.py
if %errorlevel% neq 0 (
    echo.
    echo [错误] 程序运行失败，请检查依赖是否已安装:
    echo   python -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo [完成] 按任意键退出...
pause >nul
