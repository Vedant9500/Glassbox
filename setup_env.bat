@echo off
echo ========================================
echo Glassbox Environment Setup
echo ========================================

REM Check if venv exists
if exist "venv" (
    echo Virtual environment already exists.
    echo To recreate, delete the 'venv' folder first.
    goto :activate
)

echo Creating virtual environment...
python -m venv venv

:activate
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Installing PyTorch with CUDA 12.1...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo Installing other dependencies...
pip install numpy matplotlib graphviz

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To activate the environment in the future, run:
echo     venv\Scripts\activate
echo.
echo Testing CUDA availability...
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

pause
