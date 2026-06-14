@echo off
REM PaperLens 后端启动脚本
REM 必须用此脚本启动，确保 ocr 环境（含 mineru-open-api）在 PATH 中
chcp 65001 >nul
cd /d "E:\其他\大模型项目\PaperLens"
call "E:\MiniConda\Installation\Scripts\activate.bat" ocr
if errorlevel 1 (
    echo [错误] 激活 ocr 环境失败
    pause
    exit /b 1
)
echo [OK] 已激活 ocr 环境: %CONDA_PREFIX%
where mineru-open-api
echo.
echo === 启动 PaperLens 后端 (端口 8000) ===
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause
