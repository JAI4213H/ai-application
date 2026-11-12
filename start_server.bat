@echo off
cd /d "%~dp0"
title MNIST Digit Recognizer

python -c "import zipfile,sys; sys.exit(0 if zipfile.is_zipfile('digit_model.keras') else 1)"
if errorlevel 1 (
    echo Model missing or invalid. Training it now...
    python model.py
    if errorlevel 1 (
        echo Training failed.
        pause
        exit /b 1
    )
)

echo Starting the MNIST server...
python app.py
pause
