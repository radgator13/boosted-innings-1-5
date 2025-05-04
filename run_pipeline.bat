@echo off
setlocal enabledelayedexpansion

:: === Log setup ===
set LOGFILE=logs\pipeline_log.txt
if not exist logs mkdir logs

echo =============================== > %LOGFILE%
echo ?? Starting full boosted innings pipeline... >> %LOGFILE%
echo =============================== >> %LOGFILE%
echo. >> %LOGFILE%

:: === Run each Python script ===
call :run_and_log Scrape_Fan_Graph.py
call :run_and_log get_scores_full.py
call :run_and_log predict_over_4_5.py
call :run_and_log merge_predictions.py
call :run_and_log train_model.py
call :run_and_log backfill_predict_over_4_5.py

:: === Git push (optional)
call :run_and_log_git

echo. >> %LOGFILE%
echo ? All tasks complete. >> %LOGFILE%
pause
exit /b 0

:: === Helper function to run and log ===
:run_and_log
set SCRIPT=%1
echo ?? Running: %SCRIPT% >> %LOGFILE%
python %SCRIPT% >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ? Script failed: %SCRIPT% >> %LOGFILE%
    echo ? Pipeline halted due to error in %SCRIPT%.
    pause
    exit /b 1
)
echo ? Finished: %SCRIPT% >> %LOGFILE%
echo. >> %LOGFILE%
exit /b 0

:: === Git push (optional)
:run_and_log_git
echo ?? Committing to GitHub... >> %LOGFILE%
git add . >> %LOGFILE% 2>&1
git commit -m "?? Auto-update: predictions as of %DATE% %TIME%" >> %LOGFILE% 2>&1
git push origin main >> %LOGFILE% 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo ?? Git push failed. >> %LOGFILE%
) ELSE (
    echo ? Git push complete. >> %LOGFILE%
)
exit /b 0
