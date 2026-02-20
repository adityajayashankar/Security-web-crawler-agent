@echo off
cd /d "C:\Users\adity\dataset-deplai"
echo ========== Running analyze_dataset.py ==========
python analyze_dataset.py
echo.
echo ========== Running validate_dataset.py --no-tokenizer ==========
python validate_dataset.py --no-tokenizer
pause
