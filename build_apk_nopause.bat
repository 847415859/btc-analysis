@echo off
chcp 65001 > nul
echo ============================================================
echo   CryptoTA Android APK 构建脚本 (No Pause - Clean Build)
echo ============================================================
echo.

:: ── JDK 17 (GraalVM) ─────────────────────────────────────────
set "JAVA_HOME=D:\Develop\Java\JDK\jdk17-22.3.0 graalvm"
set "PATH=%JAVA_HOME%\bin;%PATH%"

:: ── Android SDK ───────────────────────────────────────────────
set "ANDROID_HOME=D:\Develop\Android\sdk"
set "PATH=%ANDROID_HOME%\platform-tools;%ANDROID_HOME%\cmdline-tools\latest\bin;%PATH%"

:: ── 验证环境 ──────────────────────────────────────────────────
echo [1/4] 验证 Java 版本...
"%JAVA_HOME%\bin\java.exe" --version
if %ERRORLEVEL% neq 0 (
    echo ERROR: JDK 17 未找到，请检查 JAVA_HOME 路径
    exit /b 1
)

echo.
echo [2/4] 验证 Android SDK...
if not exist "%ANDROID_HOME%\platforms\android-34" (
    echo ERROR: Android SDK 未安装，请先运行 setup_sdk.bat
    exit /b 1
)
echo Android SDK OK: %ANDROID_HOME%

echo.
echo [3/4] 开始构建 Debug APK (Clean Build)...
cd /d "D:\trade\api\btc-analysis-android"
call gradlew.bat clean assembleDebug --stacktrace

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: 构建失败！
    exit /b 1
)

echo.
echo [4/4] 构建成功！
echo.
set APK=app\build\outputs\apk\debug\app-debug.apk
for %%F in ("%APK%") do echo APK 大小: %%~zF 字节
echo.
