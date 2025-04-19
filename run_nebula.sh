#!/bin/bash
# Script para ejecutar NEBULA

# Crear directorio para datos
mkdir -p ./nebula_data

# Ejecutar NEBULA con parámetros predeterminados
echo "Iniciando NEBULA - Sistema de IA Autónomo"
echo "=========================================="
echo ""

python3 integration.py --base-dir ./nebula_data --learning-cycles 3 --evolution-cycles 2 --improvement-cycles 1 --save-results
