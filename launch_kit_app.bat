@echo off
REM Launch the OmniMarble Kit application
REM Requires NVIDIA Kit SDK to be installed
REM
REM Option 1: If you have Kit SDK installed via NVIDIA Hub, set KIT_PATH:
REM   set KIT_PATH=C:\Users\%USERNAME%\AppData\Local\ov\pkg\kit-sdk-106.x
REM
REM Option 2: If you cloned kit-app-template and built it:
REM   Use repo.bat launch instead
REM
REM Option 3: If Kit is on your PATH:
REM   kit.exe source\apps\omnimarble\omnimarble.kit

if defined KIT_PATH (
    echo Launching OmniMarble with Kit SDK at: %KIT_PATH%
    "%KIT_PATH%\kit.exe" "%~dp0source\apps\omnimarble\omnimarble.kit"
) else (
    echo.
    echo === OmniMarble Kit App ===
    echo.
    echo To launch, you need the NVIDIA Kit SDK. Options:
    echo.
    echo 1. Install via NVIDIA Hub ^(https://www.nvidia.com/en-us/omniverse/download/^)
    echo    Then set KIT_PATH to the kit-sdk folder and re-run this script.
    echo.
    echo 2. Clone kit-app-template and copy our source/ folder into it:
    echo    git clone https://github.com/NVIDIA-Omniverse/kit-app-template.git
    echo    Copy source\apps\omnimarble\ into kit-app-template\source\apps\
    echo    Copy source\extensions\omni.marble.coaster\ into kit-app-template\source\extensions\
    echo    Then run: repo.bat launch --app omnimarble
    echo.
    echo 3. If kit.exe is on your PATH:
    echo    kit.exe source\apps\omnimarble\omnimarble.kit
    echo.
    pause
)
