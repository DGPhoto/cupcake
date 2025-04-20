#!/bin/bash

echo "==================================="
echo "Cupcake Photo Culling Library"
echo "Test completo del flusso di lavoro"
echo "==================================="
echo

# Configurazione percorsi
PYTHON_EXE="python3"
CUPCAKE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
EXAMPLE_SCRIPT="${CUPCAKE_DIR}/examples/complete_workflow.py"

# Configurazione dei parametri
INPUT_DIR=""
OUTPUT_DIR="${HOME}/Pictures/CupcakeOutput"
PROFILE="black_and_white"
FORMAT="jpeg"
THRESHOLD="75"

echo "Controllo dell'ambiente..."

# Verifica Python
command -v ${PYTHON_EXE} >/dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "ERRORE: Python non trovato. Assicurati che Python sia installato."
    exit 1
fi

# Verifica esempi
if [ ! -f "${EXAMPLE_SCRIPT}" ]; then
    echo "ERRORE: Script di esempio non trovato in ${EXAMPLE_SCRIPT}"
    exit 1
fi

# Chiedi directory di input
echo
echo "Inserisci il percorso completo alla directory delle immagini:"
read -p "> " INPUT_DIR

if [ -z "${INPUT_DIR}" ]; then
    echo "ERRORE: Directory di input non specificata."
    exit 1
fi

if [ ! -d "${INPUT_DIR}" ]; then
    echo "ERRORE: La directory '${INPUT_DIR}' non esiste."
    exit 1
fi

# Chiedi se modificare le opzioni predefinite
echo
echo "Opzioni predefinite:"
echo "  Profilo di rating: ${PROFILE}"
echo "  Formato di esportazione: ${FORMAT}"
echo "  Soglia di selezione: ${THRESHOLD}"
echo "  Directory di output: ${OUTPUT_DIR}"
echo
echo "Vuoi modificare le opzioni predefinite? (s/n)"
read -p "> " CHANGE_OPTIONS

if [ "${CHANGE_OPTIONS,,}" = "s" ]; then
    echo
    echo "Inserisci il profilo di rating:"
    echo "[default, black_and_white, portrait, landscape, wildlife, street, architecture, macro, night]"
    read -p "> " PROFILE
    
    echo
    echo "Inserisci il formato di esportazione:"
    echo "[original, jpeg, tiff, png]"
    read -p "> " FORMAT
    
    echo
    echo "Inserisci la soglia di selezione (0-100):"
    read -p "> " THRESHOLD
    
    echo
    echo "Inserisci la directory di output:"
    read -p "> " OUTPUT_DIR
fi

# Crea la directory di output se non esiste
if [ ! -d "${OUTPUT_DIR}" ]; then
    echo
    echo "Creazione della directory di output: ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
fi

# Costruisci il comando
COMMAND="${PYTHON_EXE} \"${EXAMPLE_SCRIPT}\" --input-dir \"${INPUT_DIR}\" --output-dir \"${OUTPUT_DIR}\" --profile ${PROFILE} --format ${FORMAT} --threshold ${THRESHOLD} --verbose"

# Esegui lo script
echo
echo "Esecuzione del workflow Cupcake..."
echo "Comando: ${COMMAND}"
echo
echo "======================================================================="
eval ${COMMAND}
echo "======================================================================="
echo

# Fine
echo
echo "Premi INVIO per uscire..."
read