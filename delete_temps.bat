@echo off
setlocal

rem Cambia directory alla posizione dello script
cd /d "%~dp0"

echo.
echo Cleaning Windows-specific CUDA/linker artifacts...
echo.

rem Elimina i file .lib, .exp, .pdb, .ilk nelle sottocartelle 'algorithms' e 'CV_ORCHESTRATOR'
rem %%i è usato qui perché è la sintassi corretta per i loop FOR nei file .bat
for /f "delims=" %%i in ('dir /S /B ".\algorithms\*.lib" ".\algorithms\*.exp" ".\algorithms\*.pdb" ".\algorithms\*.ilk" ".\CV_ORCHESTRATOR\*.lib" ".\CV_ORCHESTRATOR\*.exp" ".\CV_ORCHESTRATOR\*.pdb" ".\CV_ORCHESTRATOR\*.ilk"') do (
    echo Deleting "%%i"
    del /F /Q "%%i"
)

echo.
echo Clean process complete.
echo.

endlocal