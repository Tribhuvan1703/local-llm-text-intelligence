@echo off
echo Starting Local LLM Text Intelligence...
cd /d "%~dp0"

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Creating one...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo Launching Streamlit app...
python -m streamlit run streamlit_app/app.py
pause
