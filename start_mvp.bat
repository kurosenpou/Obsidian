@echo off
echo.
echo 🔬 Starting Obsidian AI Agent MVP...
echo.
echo Installing dependencies if needed...
pip install -r requirements_mvp.txt --quiet

echo.
echo Starting Streamlit application...
echo Open your browser to: http://localhost:8501
echo.
streamlit run app_mvp.py
