# Archivo renombrado de logging.py a nebula_logging.py para evitar conflicto con el módulo estándar logging

"""
Configuración de logging para NEBULA.

Este módulo configura el sistema de logging utilizado por todos los componentes
 del sistema NEBULA.
"""

import logging
import sys
from pathlib import Path

from config import PARAMETERS

def setup_logging():
    """
    Configura el sistema de logging para NEBULA.
    
    Configura tanto el logging a consola como a archivo, con formatos adecuados
    y niveles de detalle configurables.
    """
    # Obtener nivel de logging de los parámetros
    log_level_str = PARAMETERS.get("LOG_LEVEL", "INFO")
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # Obtener archivo de log de los parámetros
    log_file = PARAMETERS.get("LOG_FILE", Path("./nebula.log"))
    
    # Crear formateador
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)-5.5s] [%(name)-15.15s] %(message)s"
    )
    
    # Configurar logger raíz
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Limpiar handlers existentes
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Configurar handler de consola
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    
    # Configurar handler de archivo
    try:
        # Asegurar que el directorio del archivo de log existe
        log_file.parent.mkdir(exist_ok=True, parents=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')  # Modo append
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        logging.warning(f"No se pudo configurar el handler de archivo de log: {e}")
    
    # Crear logger específico para NEBULA
    nebula_logger = logging.getLogger("NEBULA")
    nebula_logger.info(f"Sistema de logging inicializado (Nivel: {log_level_str})")
    
    return nebula_logger

# Inicializar logger al importar este módulo
logger = setup_logging()