import os
import sys
import logging
from pathlib import Path

# Configuración de directorios
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(BASE_DIR / "nebula.log")
    ]
)

logger = logging.getLogger("NEBULA")

def main():
    """Función principal para iniciar NEBULA."""
    logger.info("🚀 Iniciando NEBULA - Sistema de IA Autónomo para Aprendizaje y Automejora")
    
    try:
        # Importar componentes principales
        from core.nebula_agi import NebulaAGI
        
        # Crear instancia principal
        nebula = NebulaAGI()
        
        # Iniciar el sistema
        nebula.run()
        
    except ImportError as e:
        logger.error(f"Error al importar componentes: {e}")
        logger.info("Asegúrate de haber instalado todas las dependencias necesarias.")
        return 1
    except Exception as e:
        logger.error(f"Error al iniciar NEBULA: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
