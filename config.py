"""
Configuración global para NEBULA.

Este módulo contiene los parámetros de configuración utilizados por todos los componentes
del sistema NEBULA.
"""

import os
import torch
from pathlib import Path

# Determinar dispositivo disponible
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Directorio base
BASE_DIR = Path(__file__).resolve().parent.parent

# Parámetros del sistema
PARAMETERS = {
    # --- General ---
    "SEED": 42,
    "DEVICE": DEVICE,
    "LOG_LEVEL": "INFO",
    "LOG_FILE": BASE_DIR / "nebula.log",
    "ALLOW_INTERNET_ACCESS": True,
    
    # --- NebulaSpace & QuantumNeurons ---
    "SPACE_DIMENSIONS": (100, 100, 100),
    "INITIAL_NEURONS": 200,
    "MAX_NEURONS": 1000,
    "MIN_NEURONS": 50,
    "INPUT_DIM": 16,
    "NUM_QUBITS": 4,
    "QUANTUM_LAYERS": 2,
    "CONNECTION_PROBABILITY": 0.08,
    "MAX_CONNECTIONS": 10,
    "NEURON_INTERACTION_MODEL": "light_attenuation",
    "LIGHT_ATTENUATION_FACTOR": 0.3,
    "LIGHT_RECEIVE_FACTOR": 0.02,
    "LUMINOSITY_DECAY": 0.005,
    "NEURON_ACTIVITY_THRESHOLD": 0.1,
    "NEURON_INACTIVITY_PERIOD": 100,
    "POSITION_UPDATE_DT": 0.1,
    "STRUCTURE_UPDATE_INTERVAL": 25,
    "N_CLUSTERS": 5,
    
    # --- LLM & Knowledge ---
    "EMBEDDING_MODEL_NAME": "sentence-transformers/all-MiniLM-L6-v2",
    "GENERATION_MODEL_SMALL": "gpt2",
    "GENERATION_MODEL_LARGE": "microsoft/Phi-3-mini-4k-instruct",
    "CODEGEN_MODEL_NAME": "Salesforce/codegen-350M-mono",
    "QA_MODEL_NAME": "deepset/roberta-base-squad2",
    "IMAGE_CAPTION_MODEL_NAME": "Salesforce/blip-image-captioning-base",
    "TRUST_REMOTE_CODE_LARGE_GEN": True,
    "MODEL_CACHE_DIR": BASE_DIR / "model_cache",
    "MAX_LLM_INPUT_LENGTH": 512,
    "MAX_LLM_GENERATION_LENGTH": 150,
    "MODEL_UNLOAD_DELAY": 900,
    "KNOWLEDGE_GRAPH_FILE": BASE_DIR / "data" / "nebula_kg.graphml",
    "WIKIPEDIA_LANG": "en",
    "WIKIPEDIA_MAX_PAGES_PER_CYCLE": 2,
    "HOLOGRAPHIC_MEMORY_THRESHOLD": 0.65,
    
    # --- Evolution (DEAP) ---
    "EVOLUTION_ENABLED": True,
    "EVOLUTION_INTERVAL": 50,
    "GA_POPULATION_SIZE": 20,
    "GA_GENERATIONS": 10,
    "GA_CXPB": 0.7,
    "GA_MUTPB": 0.2,
    "FITNESS_EVAL_TASKS": 3,
    
    # --- Self-Modification ---
    "SELF_CORRECTION_ENABLED": True,
    "SELF_IMPROVEMENT_INTERVAL": 100,
    "CODE_EVALUATION_THRESHOLD": 0.5,
    "MAX_ERROR_HISTORY": 50,
    "MAX_MODIFICATION_HISTORY": 20,
    
    # --- State & Backup ---
    "STATE_FILE": BASE_DIR / "data" / "nebula_state.pkl",
    "BACKUP_INTERVAL": 3600,
    "BACKUP_DIRECTORY": BASE_DIR / "backups",
    
    # --- UI ---
    "UI_ENABLED": False,
}

# Crear directorios necesarios
PARAMETERS["MODEL_CACHE_DIR"].mkdir(exist_ok=True, parents=True)
PARAMETERS["BACKUP_DIRECTORY"].mkdir(exist_ok=True, parents=True)
(BASE_DIR / "data").mkdir(exist_ok=True, parents=True)

# Configurar variables de entorno para Hugging Face
os.environ["TRANSFORMERS_CACHE"] = str(PARAMETERS["MODEL_CACHE_DIR"].absolute())
os.environ["HF_HOME"] = str(PARAMETERS["MODEL_CACHE_DIR"].absolute())

# Configurar semillas para reproducibilidad
def set_seeds(seed=PARAMETERS["SEED"]):
    """Configura semillas para reproducibilidad."""
    import random
    import numpy as np
    import torch
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Configurar semilla para PennyLane si está disponible
    try:
        import pennylane as qml
        from pennylane import numpy as pnp
        pnp.random.seed(seed)
    except ImportError:
        pass
    
    # Configurar semilla para DEAP si está disponible
    try:
        from deap import base, creator, tools, algorithms
        random.seed(seed)
    except ImportError:
        pass

# Configurar semillas al importar este módulo
set_seeds()
