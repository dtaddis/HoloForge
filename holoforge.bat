@echo off
rem HoloForge launcher: runs in the project's own .venv, whatever Python is installed globally.
rem   holoforge.bat                       open the web app (or double-click this file)
rem   holoforge.bat photo.jpg [options]   convert photos on the command line (--help for options)
rem   holoforge.bat ui^|portrait^|setup ...  the web app, Portrait and setup tools
setlocal
rem Activate the venv from this file's own folder, so it still works if the folder is moved.
set "VIRTUAL_ENV=%~dp0.venv"
if not exist "%VIRTUAL_ENV%\Scripts\holoforge.exe" (
    echo HoloForge is not installed yet. In PowerShell, run: "%~dp0install.ps1"
    if "%~1"=="" pause
    exit /b 1
)
set "PATH=%VIRTUAL_ENV%\Scripts;%PATH%"

if "%~1"=="" (
    holoforge-ui.exe
    rem A double-clicked window closes on exit; keep it open long enough to read any error.
    if errorlevel 1 pause
    exit /b
)
set "TOOL=holoforge"
if /i "%~1"=="ui" set "TOOL=holoforge-ui"
if /i "%~1"=="portrait" set "TOOL=holoforge-portrait"
if /i "%~1"=="setup" set "TOOL=holoforge-setup"
if not "%TOOL%"=="holoforge" shift

set "ARGS="
:collect
if "%~1"=="" goto run
set ARGS=%ARGS% %1
shift
goto collect
:run
%TOOL%.exe%ARGS%
