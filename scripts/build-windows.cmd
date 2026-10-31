@echo off
setlocal
if defined TTS_SOURCE_DIR (pushd "%TTS_SOURCE_DIR%") else (pushd "%~dp0..")
if errorlevel 1 exit /b 1
set "TTS_BUILD_EXIT=1"
set "TTS_BUILD_MODE=build"
set "TTS_BUILD_TARGET="
if not defined TTS_CMAKE_EXE set "TTS_CMAKE_EXE=cmake"
if not defined TTS_NVCC_EXE set "TTS_NVCC_EXE=nvcc"
if not defined TTS_VSWHERE_EXE set "TTS_VSWHERE_EXE=vswhere"
if not defined TTS_PRESET set "TTS_PRESET=%TTS_WINDOWS_PRESET%"
:arguments
if "%~1"=="" goto prepare
if /I "%~1"=="--preset" (
  if "%~2"=="" goto usage
  set "TTS_PRESET=%~2"
  shift
  shift
  goto arguments
)
if /I "%~1"=="--target" (
  if "%~2"=="" goto usage
  set "TTS_BUILD_TARGET=%~2"
  shift
  shift
  goto arguments
)
if /I "%~1"=="--configure-only" (set "TTS_BUILD_MODE=configure-only") else if /I "%~1"=="--configure" (set "TTS_BUILD_MODE=configure") else if /I "%~1"=="--rebuild" (set "TTS_BUILD_MODE=rebuild") else if /I "%~1"=="--build-only" (set "TTS_BUILD_MODE=build") else goto usage
shift
goto arguments
:prepare
if not defined TTS_PRESET goto usage
rem Use an existing developer environment, or discover/import the requested one.
if defined INCLUDE if defined LIB (
  where cl >nul 2>&1
  if not errorlevel 1 goto ready
)
if not defined TTS_VCVARS (
  for /f "usebackq delims=" %%I in (`"%TTS_VSWHERE_EXE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find VC\Auxiliary\Build\vcvars64.bat`) do set "TTS_VCVARS=%%I"
)
if not defined TTS_VCVARS (
  echo Provide an MSVC developer environment, TTS_VCVARS, or vswhere on PATH.
  goto finish
)
call "%TTS_VCVARS%" >nul
if errorlevel 1 goto capture
:ready
if /I "%TTS_BUILD_MODE%"=="configure" goto configure
if /I "%TTS_BUILD_MODE%"=="configure-only" goto configure
goto build
:configure
"%TTS_CMAKE_EXE%" --version
if errorlevel 1 goto capture
rustc --version
if errorlevel 1 goto capture
"%TTS_NVCC_EXE%" --version
if errorlevel 1 goto capture
"%TTS_CMAKE_EXE%" --preset "%TTS_PRESET%"
if errorlevel 1 goto capture
if /I "%TTS_BUILD_MODE%"=="configure-only" goto capture
:build
set "TTS_CLEAN_FLAG="
if /I "%TTS_BUILD_MODE%"=="rebuild" set "TTS_CLEAN_FLAG=--clean-first"
if defined TTS_BUILD_TARGET (
  "%TTS_CMAKE_EXE%" --build --preset "%TTS_PRESET%" %TTS_CLEAN_FLAG% --target "%TTS_BUILD_TARGET%"
) else (
  "%TTS_CMAKE_EXE%" --build --preset "%TTS_PRESET%" %TTS_CLEAN_FLAG%
)
goto capture
:usage
echo Usage: build-windows.cmd --preset NAME [--configure^|--configure-only^|--build-only^|--rebuild] [--target NAME]
set "TTS_BUILD_EXIT=2"
goto finish
:capture
set "TTS_BUILD_EXIT=%errorlevel%"
:finish
popd
exit /b %TTS_BUILD_EXIT%
