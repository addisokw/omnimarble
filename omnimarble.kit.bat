@echo off
REM Launch OmniMarble in Kit
REM Kit SDK lives in the kit-app-template build next to this repo

set KIT_BUILD=%~dp0..\kit-app-template\_build\windows-x86_64\release

if not exist "%KIT_BUILD%\kit\kit.exe" (
    echo ERROR: Kit SDK not found at %KIT_BUILD%
    echo.
    echo Run this once to build it:
    echo   cd ..\kit-app-template
    echo   repo.bat build
    echo.
    pause
    exit /b 1
)

REM kit-app-template 110.x stages apps flat as apps\omnimarble.kit; older
REM templates nested them as apps\omnimarble\omnimarble.kit. Accept both.
set KIT_APP=%KIT_BUILD%\apps\omnimarble.kit
if not exist "%KIT_APP%" set KIT_APP=%KIT_BUILD%\apps\omnimarble\omnimarble.kit

if not exist "%KIT_APP%" (
    echo ERROR: omnimarble.kit not found under %KIT_BUILD%\apps
    echo.
    echo Register the app in the kit-app-template checkout, then rebuild:
    echo   copy source\apps\omnimarble\omnimarble.kit ..\kit-app-template\source\apps\
    echo   add define_app^("omnimarble.kit"^) to premake5.lua
    echo   add the app path to [repo_precache_exts].apps in repo.toml
    echo   cd ..\kit-app-template ^&^& repo.bat build
    echo.
    pause
    exit /b 1
)

REM Our extension folder MUST come first so Kit finds our live source
REM before any cached copies in the build tree
call "%KIT_BUILD%\kit\kit.exe" "%KIT_APP%" ^
    --ext-folder "%~dp0source\extensions" ^
    --ext-folder "%KIT_BUILD%\exts" ^
    --ext-folder "%KIT_BUILD%\extscache" ^
    --ext-folder "%KIT_BUILD%\apps" ^
    %*
