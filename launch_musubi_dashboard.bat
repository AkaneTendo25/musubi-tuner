@echo off
setlocal EnableExtensions

for %%I in ("%~dp0.") do set "REPO_ROOT=%%~fI"

if not defined HOST set "HOST=127.0.0.1"
if not defined PORT set "PORT=7860"
if not defined MUSUBI_DASHBOARD_VENV set "MUSUBI_DASHBOARD_VENV=%REPO_ROOT%\venv"

set "VENV_ROOT=%MUSUBI_DASHBOARD_VENV%"
if exist "%VENV_ROOT%\python.exe" (
  set "VENV_PY=%VENV_ROOT%\python.exe"
) else (
  set "VENV_PY=%VENV_ROOT%\Scripts\python.exe"
)

if not exist "%VENV_PY%" (
  echo [launch_musubi_dashboard] ERROR: Python environment not found:
  echo   %VENV_PY%
  echo Set MUSUBI_DASHBOARD_VENV to another Conda or virtual environment.
  pause
  exit /b 1
)

if not exist "%REPO_ROOT%\src\musubi_tuner\gui_dashboard\__main__.py" (
  echo [launch_musubi_dashboard] ERROR: Dashboard module not found in this checkout.
  pause
  exit /b 1
)

set "FRONTEND_DIR=%REPO_ROOT%\src\musubi_tuner\gui_dashboard\frontend"
if not exist "%FRONTEND_DIR%\dist\index.html" (
  where npm >nul 2>nul
  if errorlevel 1 (
    echo [launch_musubi_dashboard] ERROR: The dashboard frontend is not built and npm was not found.
    echo Install Node.js, then run this launcher again.
    pause
    exit /b 1
  )
  echo [launch_musubi_dashboard] Building dashboard frontend...
  pushd "%FRONTEND_DIR%"
  call npm ci --no-audit --no-fund
  if errorlevel 1 (
    popd
    echo [launch_musubi_dashboard] ERROR: Dashboard frontend dependency install failed.
    pause
    exit /b 1
  )
  call npm run build
  if errorlevel 1 (
    popd
    echo [launch_musubi_dashboard] ERROR: Dashboard frontend build failed.
    pause
    exit /b 1
  )
  popd
)

rem torch.compile on Windows needs the MSVC include and library paths from vcvars64.bat.
set "VCVARS64="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" (
  for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find VC\Auxiliary\Build\vcvars64.bat`) do if not defined VCVARS64 set "VCVARS64=%%I"
)

if defined VCVARS64 (
  call "%VCVARS64%" >nul 2>&1
  if errorlevel 1 echo [launch_musubi_dashboard] WARNING: Visual Studio C++ environment setup failed. torch.compile may not work.
) else (
  echo [launch_musubi_dashboard] WARNING: Visual Studio C++ tools were not found. torch.compile may not work.
)

set "PYTHONHOME="
set "PYTHONPATH=%REPO_ROOT%\src;%PYTHONPATH%"
set "PATH=%VENV_ROOT%\Scripts;%VENV_ROOT%;%VENV_ROOT%\Library\bin;%PATH%"
set "PYTHONNOUSERSITE=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "DASHBOARD_URL=http://%HOST%:%PORT%"
if /I "%HOST%"=="0.0.0.0" set "DASHBOARD_URL=http://127.0.0.1:%PORT%"

if /I not "%~1"=="--help" if /I not "%~1"=="-h" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri '%DASHBOARD_URL%/api/processes/status' -TimeoutSec 2; if ($r.StatusCode -ge 200 -and $r.StatusCode -lt 500) { exit 0 } } catch { exit 1 }" >nul 2>nul
  if not errorlevel 1 (
    echo [launch_musubi_dashboard] Dashboard is already running at %DASHBOARD_URL%
    start "" "%DASHBOARD_URL%"
    exit /b 0
  )
)

echo [launch_musubi_dashboard] Starting H3 dashboard at %DASHBOARD_URL%
echo [launch_musubi_dashboard] Python: %VENV_PY%
if /I not "%~1"=="--help" if /I not "%~1"=="-h" start "" "%DASHBOARD_URL%"

pushd "%REPO_ROOT%"
if defined PROJECT (
  if exist "%PROJECT%" (
    "%VENV_PY%" -m musubi_tuner.gui_dashboard --host "%HOST%" --port "%PORT%" --project "%PROJECT%" %*
  ) else (
    echo [launch_musubi_dashboard] WARNING: Project file not found: %PROJECT%
    "%VENV_PY%" -m musubi_tuner.gui_dashboard --host "%HOST%" --port "%PORT%" %*
  )
) else (
  "%VENV_PY%" -m musubi_tuner.gui_dashboard --host "%HOST%" --port "%PORT%" %*
)
set "EXIT_CODE=%ERRORLEVEL%"
popd

if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
