@echo off
echo ========================================
echo  Virage Procedures Chatbot - Setup
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python from python.org and check "Add to PATH".
    pause
    exit /b 1
)

echo [1/4] Installing dependencies...
pip install -r "%~dp0requirements.txt"
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo [2/4] Checking for procedure documents...
if not exist "%~dp0procedures\*.docx" (
    echo WARNING: No .docx files found in the procedures folder.
    echo Please copy your procedure documents to:
    echo   %~dp0procedures\
    echo.
    echo You can still start the chatbot and add documents later.
)

echo.
echo [3/4] Building document index...
python "%~dp0document_processor.py"

echo.
echo [4/4] Starting chatbot...
echo.
echo The chatbot will open in your web browser.
echo Press Ctrl+C in this window to stop the server.
echo.
streamlit run "%~dp0app.py"

pause
