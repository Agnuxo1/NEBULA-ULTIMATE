#!/bin/bash
# Script para instalar dependencias y ejecutar NEBULA

echo "Instalando dependencias para NEBULA..."
echo "======================================="

# Verificar si pip está instalado
if ! command -v pip &> /dev/null; then
    echo "Error: pip no está instalado. Por favor, instala pip primero."
    exit 1
fi

# Instalar dependencias con manejo de errores
echo "Instalando PyTorch y otras dependencias (esto puede tomar varios minutos)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu || { echo "Error al instalar PyTorch"; exit 1; }
pip install pennylane || { echo "Error al instalar PennyLane"; exit 1; }
pip install transformers || { echo "Error al instalar Transformers"; exit 1; }
pip install huggingface_hub || { echo "Error al instalar Hugging Face Hub"; exit 1; }
pip install sentence-transformers || { echo "Error al instalar Sentence Transformers"; exit 1; }
pip install networkx || { echo "Error al instalar NetworkX"; exit 1; }
pip install scipy || { echo "Error al instalar SciPy"; exit 1; }
pip install spacy || { echo "Error al instalar Spacy"; exit 1; }
pip install deap || { echo "Error al instalar DEAP"; exit 1; }
pip install faiss-cpu || { echo "Error al instalar FAISS"; exit 1; }

echo "Dependencias instaladas correctamente."
echo ""

# Crear directorio para datos
mkdir -p ./nebula_data

# Ejecutar NEBULA con parámetros predeterminados
echo "Iniciando NEBULA - Sistema de IA Autónomo"
echo "=========================================="
echo ""

python3 integration.py --base-dir ./nebula_data --learning-cycles 3 --evolution-cycles 2 --improvement-cycles 1 --save-results
