@echo off
echo Starting Bird vs Drone Detector (Streamlit)...
echo.
echo Install deps first (once):
echo   pip install -r requirements_streamlit.txt
echo.
streamlit run app.py --server.headless false --browser.gatherUsageStats false
pause
