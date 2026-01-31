REM Activate virtual environment
call venv\Scripts\activate.bat

echo Starting Flask application...
echo.
echo Access the app at: http://localhost:5000
echo Press Ctrl+C to stop
echo.

REM Run the app
python app.py
