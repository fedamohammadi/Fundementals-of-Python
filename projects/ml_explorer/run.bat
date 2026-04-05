@echo off
cd /d "%~dp0..\.."
echo Starting ML Explorer...
.venv\Scripts\streamlit run projects/ml_explorer/app.py
pause
