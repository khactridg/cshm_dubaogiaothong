@echo off
rem Mo giao dien demo tim duong (Streamlit). Can cai_dat.bat truoc. Tat demo: dong cua so nay.
chcp 65001 >nul
cd /d "%~dp0demo"
if not exist "..\.venv\Scripts\python.exe" (
    echo Chua co moi truong .venv. Hay chay cai_dat.bat truoc.
    pause
    exit /b 1
)
set LAM_SACH=v2
set PYTHONUTF8=1
echo Dang mo demo... Trinh duyet se tu mo trang http://localhost:8501
"..\.venv\Scripts\python.exe" -m streamlit run app\app.py --browser.gatherUsageStats false
pause
