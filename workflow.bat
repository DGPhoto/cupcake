@echo off
setlocal enabledelayedexpansion

echo ===================================
echo Cupcake Photo Culling Library
echo Test completo del flusso di lavoro
echo ===================================
echo.

:: Configurazione percorsi
set "PYTHON_EXE=python"
set "CUPCAKE_DIR=%~dp0"
set "EXAMPLE_SCRIPT=%CUPCAKE_DIR%examples\complete_workflow.py"

:: Configurazione dei parametri
set "INPUT_DIR="
set "OUTPUT_DIR=%USERPROFILE%\Pictures\CupcakeOutput"
set "PROFILE=black_and_white"
set "FORMAT=jpeg"
set "THRESHOLD=75"

echo Controllo dell'ambiente...

:: Verifica Python
%PYTHON_EXE% --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo ERRORE: Python non trovato. Assicurati che Python sia installato e nel PATH.
    goto :end
)

:: Verifica esempi
if not exist "%EXAMPLE_SCRIPT%" (
    echo ERRORE: Script di esempio non trovato in %EXAMPLE_SCRIPT%
    goto :end
)

:: Chiedi directory di input
echo.
echo Inserisci il percorso completo alla directory delle immagini:
set /p INPUT_DIR="> "

if "%INPUT_DIR%"=="" (
    echo ERRORE: Directory di input non specificata.
    goto :end
)

if not exist "%INPUT_DIR%" (
    echo ERRORE: La directory "%INPUT_DIR%" non esiste.
    goto :end
)

:: Chiedi se modificare le opzioni predefinite
echo.
echo Opzioni predefinite:
echo   Profilo di rating: %PROFILE%
echo   Formato di esportazione: %FORMAT%
echo   Soglia di selezione: %THRESHOLD%
echo   Directory di output: %OUTPUT_DIR%
echo.
echo Vuoi modificare le opzioni predefinite? (s/n)
set /p CHANGE_OPTIONS="> "

if "%CHANGE_OPTIONS%"=="s" (
    echo.
    echo Inserisci il profilo di rating:
    echo [default, black_and_white, portrait, landscape, wildlife, street, architecture, macro, night]
    set /p PROFILE="> "
    
    echo.
    echo Inserisci il formato di esportazione:
    echo [original, jpeg, tiff, png]
    set /p FORMAT="> "
    
    echo.
    echo Inserisci la soglia di selezione (0-100):
    set /p THRESHOLD="> "
    
    echo.
    echo Inserisci la directory di output:
    set /p OUTPUT_DIR="> "
)

:: Crea la directory di output se non esiste
if not exist "%OUTPUT_DIR%" (
    echo.
    echo Creazione della directory di output: %OUTPUT_DIR%
    mkdir "%OUTPUT_DIR%" 2>nul
)

:: Costruisci il comando
set "COMMAND=%PYTHON_EXE% "%EXAMPLE_SCRIPT%" --input-dir "%INPUT_DIR%" --output-dir "%OUTPUT_DIR%" --profile %PROFILE% --format %FORMAT% --threshold %THRESHOLD% --verbose"

:: Esegui lo script
echo.
echo Esecuzione del workflow Cupcake...
echo Comando: %COMMAND%
echo.
echo =======================================================================
%COMMAND%
echo =======================================================================
echo.

:: Fine
:end
echo.
echo Premi un tasto per uscire...
pause >nul
endlocal