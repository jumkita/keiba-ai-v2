@echo off
setlocal

cd /d "%~dp0"

echo ============================================================
echo  Weekly Automation (PDCA) Start
echo  Extract - Master - Train - Predict - Evaluate
echo ============================================================
echo.

python run_weekly_automation.py
set EXIT_CODE=%ERRORLEVEL%

echo.
rem Open latest HTML report
set "REPORT_DIR=%~dp0jv_data\reports"
if exist "%REPORT_DIR%" (
    for /f "delims=" %%f in ('dir /b /o-d "%REPORT_DIR%\report_*.html" 2^>nul') do (
        echo Opening report: %%f
        start "" "%REPORT_DIR%\%%f"
        goto :opened
    )
)
echo No report found.
:opened

rem Show evaluation summary path
if exist "%~dp0jv_data\reports\evaluation_summary.csv" (
    echo.
    echo Evaluation: jv_data\reports\evaluation_summary.csv
    echo PredLog:    jv_data\history\prediction_log.csv
)

echo.
echo ============================================================
echo  Git Auto-Deploy (push docs to GitHub)
echo ============================================================
git add .
git commit -m "Auto-update: Weekly Prediction %date%"
if %ERRORLEVEL% equ 0 (
    git push origin main
    if %ERRORLEVEL% equ 0 (
        echo Push complete. GitHub Pages will update.
    ) else (
        echo Push failed. Check remote settings.
    )
) else (
    echo No changes to commit.
)

echo.
echo ============================================================
echo  Done. Press any key to exit.
echo ============================================================
pause
exit /b %EXIT_CODE%
