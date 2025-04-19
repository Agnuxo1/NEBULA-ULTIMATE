@echo off
echo Instalando dependencias para NEBULA...
echo =======================================

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pennylane
pip install transformers
pip install huggingface_hub
pip install sentence-transformers
pip install networkx
pip install scipy
pip install spacy
pip install deap
pip install faiss-cpu

echo Dependencias instaladas correctamente.
echo.

mkdir nebula_data

echo Iniciando NEBULA - Sistema de IA Autónomo
echo ==========================================
echo.

python integration.py --base-dir ./nebula_data --learning-cycles 3 --evolution-cycles 2 --improvement-cycles 1 --save-results
