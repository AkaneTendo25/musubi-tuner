@echo off
setlocal EnableExtensions

for %%I in ("%~dp0.") do set "REPO_ROOT=%%~fI"

if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=7860"
if not defined PROJECT_FILE set "PROJECT_FILE=%REPO_ROOT%\projects\runpod\project.json"

set "PYTHONPATH=%REPO_ROOT%\src;%PYTHONPATH%"

if defined VENV_PY (
  if exist "%VENV_PY%" goto run_dashboard
  echo [launch_musubi_dashboard] VENV_PY was set but does not exist: %VENV_PY%
)

set "VENV_PY=%REPO_ROOT%\ltx-2\python.exe"
if exist "%VENV_PY%" goto run_dashboard

set "VENV_PY=%REPO_ROOT%\venv\Scripts\python.exe"
if exist "%VENV_PY%" goto run_dashboard

set "VENV_PY=%REPO_ROOT%\.venv\Scripts\python.exe"
if exist "%VENV_PY%" goto run_dashboard

echo [launch_musubi_dashboard] ERROR: no local Python environment found.
echo [launch_musubi_dashboard] Checked:
echo   %REPO_ROOT%\ltx-2\python.exe
echo   %REPO_ROOT%\venv\Scripts\python.exe
echo   %REPO_ROOT%\.venv\Scripts\python.exe
echo.
echo Run launch_musubi_setup.cmd to recreate the environment if it was not moved.
pause
exit /b 1

:run_dashboard
set "DASHBOARD_URL=http://%HOST%:%PORT%"
if /I "%HOST%"=="0.0.0.0" set "DASHBOARD_URL=http://127.0.0.1:%PORT%"

echo [launch_musubi_dashboard] Starting Musubi dashboard at %DASHBOARD_URL%
if /I not "%~1"=="--help" if /I not "%~1"=="-h" start "" "%DASHBOARD_URL%"

if exist "%PROJECT_FILE%" (
  "%VENV_PY%" -m musubi_tuner.gui_dashboard --host "%HOST%" --port "%PORT%" --project "%PROJECT_FILE%" %*
) else (
  "%VENV_PY%" -m musubi_tuner.gui_dashboard --host "%HOST%" --port "%PORT%" %*
)

set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
