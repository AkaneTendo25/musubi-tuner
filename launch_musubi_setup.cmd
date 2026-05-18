@echo off
setlocal

cd /d "%~dp0"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\install.ps1" -InstallRoot "%~dp0.." -RepoDir "%~dp0" -RepoUrl "https://github.com/Ada123-a/musubi-tuner-DoKr-SinkSGD_adv-TREAD.git" -Branch "ltx-2" -Cuda "cu128" -PythonVersion "3.12" -DashboardHost "127.0.0.1" -Port 7860 %*
