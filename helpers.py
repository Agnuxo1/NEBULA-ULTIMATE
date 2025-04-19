"""
Funciones auxiliares para NEBULA.

Este módulo contiene funciones de utilidad usadas por los diferentes componentes
del sistema NEBULA.
"""

import logging
import numpy as np
import torch
from typing import Any, Optional, Union, List

logger = logging.getLogger("NEBULA.Helpers")

def convert_to_numpy(data: Any) -> Optional[np.ndarray]:
    """
    Convierte diferentes tipos de datos a arrays de NumPy.
    
    Args:
        data: Datos a convertir (tensor de PyTorch, lista, o array de NumPy).
        
    Returns:
        Array de NumPy o None si la conversión no es posible.
    """
    if data is None:
        return None
    
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, (list, tuple)):
        return np.array(data)
    else:
        logger.warning(f"No se puede convertir tipo {type(data)} a NumPy array")
        return None

def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calcula la similitud del coseno entre dos vectores.
    
    Args:
        vec1: Primer vector.
        vec2: Segundo vector.
        
    Returns:
        Similitud del coseno (entre -1 y 1).
    """
    if vec1 is None or vec2 is None:
        return 0.0
    
    # Normalizar vectores
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)

def cosine_similarity(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
    """
    Calcula la similitud del coseno entre todas las filas de dos matrices.
    
    Args:
        matrix1: Primera matriz (shape: [n, d]).
        matrix2: Segunda matriz (shape: [m, d]).
        
    Returns:
        Matriz de similitudes (shape: [n, m]).
    """
    # Normalizar matrices
    norm1 = np.linalg.norm(matrix1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(matrix2, axis=1, keepdims=True)
    
    # Evitar división por cero
    norm1 = np.where(norm1 == 0, 1e-10, norm1)
    norm2 = np.where(norm2 == 0, 1e-10, norm2)
    
    matrix1_normalized = matrix1 / norm1
    matrix2_normalized = matrix2 / norm2
    
    # Calcular similitud
    return np.dot(matrix1_normalized, matrix2_normalized.T)

def require_component(component_name: str):
    """
    Decorador que verifica si un componente requerido está disponible.
    
    Args:
        component_name: Nombre del atributo del componente requerido.
        
    Returns:
        Decorador que verifica la disponibilidad del componente.
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            component = getattr(self, component_name, None)
            if component is None:
                logger.warning(f"Componente requerido '{component_name}' no disponible para {func.__name__}")
                return None
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def require_llm(model_key: str):
    """
    Decorador que asegura que un modelo LLM específico esté cargado.
    
    Args:
        model_key: Clave del modelo LLM requerido.
        
    Returns:
        Decorador que carga el modelo si es necesario.
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, 'load_llm_model'):
                logger.warning(f"Método load_llm_model no disponible para {func.__name__}")
                return None
            
            if not self.load_llm_model(model_key):
                logger.warning(f"No se pudo cargar el modelo LLM '{model_key}' para {func.__name__}")
                return None
            
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def safe_loop(max_retries: int = 3, delay: int = 5):
    """
    Decorador para manejo seguro de bucles con reintentos y backoff exponencial.
    
    Args:
        max_retries: Número máximo de reintentos.
        delay: Retraso inicial entre reintentos (segundos).
        
    Returns:
        Decorador para manejo seguro de bucles.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                instance = args[0] if args and hasattr(args[0], 'handle_error') else None
                try:
                    return func(*args, **kwargs)
                except KeyboardInterrupt:
                    logger.warning(f"🛑 Interrupción de usuario detectada en {func.__name__}. Intentando apagado gracioso...")
                    if instance and hasattr(instance, 'shutdown'):
                        instance.shutdown()
                    return None
                except Exception as e:
                    logger.error(f"🚨 Excepción no manejada en {func.__name__} (Intento {retries + 1}/{max_retries}): {e}", exc_info=True)
                    if instance:
                        try:
                            snippet = instance.get_relevant_code_snippet(depth=2) if hasattr(instance, 'get_relevant_code_snippet') else "No se pudo recuperar fragmento de código."
                        except:
                            snippet = "No se pudo recuperar fragmento de código."
                        instance.handle_error(f"Excepción no manejada en {func.__name__}: {e}", snippet)
                    
                    retries += 1
                    if retries < max_retries:
                        import time
                        sleep_time = delay * (2 ** (retries - 1))  # Backoff exponencial
                        logger.info(f"Reintentando en {sleep_time} segundos...")
                        time.sleep(sleep_time)
            
            logger.critical(f"Función {func.__name__} falló después de {max_retries} intentos.")
            return None
        return wrapper
    return decorator
