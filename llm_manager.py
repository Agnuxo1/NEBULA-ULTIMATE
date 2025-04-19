"""
Implementación del Gestor de Modelos de Lenguaje para NEBULA.

Esta clase gestiona la carga, uso y descarga de modelos de lenguaje grandes (LLMs)
para tareas de procesamiento de lenguaje natural.
"""

import logging
import time
import os
import gc
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from utils.config import PARAMETERS
from utils.helpers import safe_loop

logger = logging.getLogger("NEBULA.LLMManager")

class LLMManager:
    """
    Gestor de modelos de lenguaje grandes (LLMs) para NEBULA.
    
    Características:
    - Carga y descarga dinámica de modelos
    - Gestión eficiente de memoria
    - Soporte para diferentes tipos de modelos
    - Interfaz unificada para generación de texto
    """
    
    def __init__(self, device: torch.device = PARAMETERS["DEVICE"]):
        """
        Inicializa el gestor de modelos LLM.
        
        Args:
            device: Dispositivo de PyTorch para cálculos tensoriales.
        """
        logger.info("Inicializando LLMManager...")
        self.device = device
        
        # Diccionario para almacenar modelos, tokenizers y procesadores
        self.models = {}
        
        # Estado de carga de modelos
        self.load_status = {}
        
        # Registro de uso de modelos
        self.model_usage = {}
        
        # Configurar modelos disponibles
        self._configure_available_models()
        
        logger.info("LLMManager inicializado correctamente.")
    
    def _configure_available_models(self):
        """Configura los modelos disponibles y sus parámetros."""
        # Modelo de embeddings
        self.models["embedding"] = {
            "name": PARAMETERS["EMBEDDING_MODEL_NAME"],
            "type": "sentence_transformer",
            "model": None,
            "tokenizer": None,
            "processor": None,
            "pipeline": None,
            "last_used": 0,
            "memory_usage": 500,  # Estimación en MB
            "loaded": False
        }
        
        # Modelo pequeño de generación de texto
        self.models["text_generation_small"] = {
            "name": PARAMETERS["GENERATION_MODEL_SMALL"],
            "type": "causal_lm",
            "model": None,
            "tokenizer": None,
            "processor": None,
            "pipeline": None,
            "last_used": 0,
            "memory_usage": 1000,  # Estimación en MB
            "loaded": False
        }
        
        # Modelo grande de generación de texto
        self.models["text_generation_large"] = {
            "name": PARAMETERS["GENERATION_MODEL_LARGE"],
            "type": "causal_lm",
            "model": None,
            "tokenizer": None,
            "processor": None,
            "pipeline": None,
            "last_used": 0,
            "memory_usage": 3000,  # Estimación en MB
            "loaded": False
        }
        
        # Modelo de generación de código
        self.models["code_generation"] = {
            "name": PARAMETERS["CODEGEN_MODEL_NAME"],
            "type": "causal_lm",
            "model": None,
            "tokenizer": None,
            "processor": None,
            "pipeline": None,
            "last_used": 0,
            "memory_usage": 1500,  # Estimación en MB
            "loaded": False
        }
        
        # Modelo de respuesta a preguntas
        self.models["qa"] = {
            "name": PARAMETERS["QA_MODEL_NAME"],
            "type": "question_answering",
            "model": None,
            "tokenizer": None,
            "processor": None,
            "pipeline": None,
            "last_used": 0,
            "memory_usage": 800,  # Estimación en MB
            "loaded": False
        }
        
        # Modelo de descripción de imágenes
        self.models["image_captioning"] = {
            "name": PARAMETERS["IMAGE_CAPTION_MODEL_NAME"],
            "type": "image_captioning",
            "model": None,
            "tokenizer": None,
            "processor": None,
            "pipeline": None,
            "last_used": 0,
            "memory_usage": 1200,  # Estimación en MB
            "loaded": False
        }
        
        # Inicializar estado de carga
        self.load_status = {key: False for key in self.models.keys()}
    
    def is_model_loaded(self, model_key: str) -> bool:
        """
        Verifica si un modelo está cargado.
        
        Args:
            model_key: Clave del modelo a verificar.
            
        Returns:
            True si el modelo está cargado, False en caso contrario.
        """
        if model_key not in self.models:
            return False
        
        model_info = self.models[model_key]
        
        # Para SentenceTransformer, solo necesitamos el modelo
        if model_info["type"] == "sentence_transformer":
            return model_info["model"] is not None
        
        # Para otros modelos, necesitamos modelo y tokenizer/processor
        tokenizer_or_processor = model_info["tokenizer"] or model_info["processor"]
        return model_info["model"] is not None and tokenizer_or_processor is not None
    
    @safe_loop(max_retries=2, delay=2)
    def load_model(self, model_key: str, force: bool = False) -> bool:
        """
        Carga un modelo específico.
        
        Args:
            model_key: Clave del modelo a cargar.
            force: Si es True, fuerza la carga incluso si ya está cargado.
            
        Returns:
            True si el modelo se cargó correctamente, False en caso contrario.
        """
        if model_key not in self.models:
            logger.error(f"Modelo '{model_key}' no configurado.")
            return False
        
        # Verificar si ya está cargado
        if self.is_model_loaded(model_key) and not force:
            logger.debug(f"Modelo '{model_key}' ya está cargado.")
            self.models[model_key]["last_used"] = time.time()
            return True
        
        model_info = self.models[model_key]
        model_name = model_info["name"]
        model_type = model_info["type"]
        
        logger.info(f"Cargando modelo '{model_key}' ({model_name})...")
        
        try:
            # Verificar memoria disponible antes de cargar
            self._check_memory_usage(f"antes de cargar {model_key}")
            
            # Liberar memoria si es necesario
            if self._get_ram_usage() > 80 or (self.device.type == 'cuda' and self._get_gpu_usage() > 80):
                logger.warning(f"Memoria alta antes de cargar {model_key}. Intentando liberar...")
                self.unload_inactive_models()
            
            # Directorio de caché
            cache_dir = PARAMETERS["MODEL_CACHE_DIR"]
            
            # Cargar según tipo de modelo
            if model_type == "sentence_transformer":
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
                
                # Mover a dispositivo adecuado
                if self.device.type == 'cuda':
                    model.to(self.device)
                
                model_info["model"] = model
                model_info["tokenizer"] = None
                model_info["processor"] = None
                model_info["pipeline"] = None
            
            elif model_type == "causal_lm":
                from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                
                # Cargar tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir,
                    use_fast=True
                )
                
                # Cargar modelo
                trust_remote_code = PARAMETERS.get("TRUST_REMOTE_CODE_LARGE_GEN", False)
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    device_map="auto" if self.device.type == 'cuda' else None,
                    trust_remote_code=trust_remote_code
                )
                
                # Crear pipeline
                text_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if self.device.type == 'cuda' else -1
                )
                
                model_info["model"] = model
                model_info["tokenizer"] = tokenizer
                model_info["processor"] = None
                model_info["pipeline"] = text_pipeline
            
            elif model_type == "question_answering":
                from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
                
                # Cargar tokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir,
                    use_fast=True
                )
                
                # Cargar modelo
                model = AutoModelForQuestionAnswering.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    device_map="auto" if self.device.type == 'cuda' else None
                )
                
                # Crear pipeline
                qa_pipeline = pipeline(
                    "question-answering",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if self.device.type == 'cuda' else -1
                )
                
                model_info["model"] = model
                model_info["tokenizer"] = tokenizer
                model_info["processor"] = None
                model_info["pipeline"] = qa_pipeline
            
            elif model_type == "image_captioning":
                from transformers import BlipProcessor, BlipForConditionalGeneration
                
                # Cargar procesador
                processor = BlipProcessor.from_pretrained(
                    model_name,
                    cache_dir=cache_dir
                )
                
                # Cargar modelo
                model = BlipForConditionalGeneration.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    device_map="auto" if self.device.type == 'cuda' else None
                )
                
                model_info["model"] = model
                model_info["tokenizer"] = None
                model_info["processor"] = processor
                model_info["pipeline"] = None
            
            else:
                logger.error(f"Tipo de modelo no soportado: {model_type}")
                return False
            
            # Actualizar estado
            model_info["last_used"] = time.time()
            model_info["loaded"] = True
            self.load_status[model_key] = True
            
            logger.info(f"Modelo '{model_key}' cargado correctamente.")
            return True
        
        except Exception as e:
            logger.error(f"Error al cargar modelo '{model_key}': {e}", exc_info=True)
            
            # Limpiar recursos en caso de error
            if model_key in self.models:
                self.models[model_key]["model"] = None
                self.models[model_key]["tokenizer"] = None
                self.models[model_key]["processor"] = None
                self.models[model_key]["pipeline"] = None
                self.models[model_key]["loaded"] = False
                self.load_status[model_key] = False
            
            # Forzar recolección de basura
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return False
    
    def unload_model(self, model_key: str, force: bool = False) -> bool:
        """
        Descarga un modelo para liberar memoria.
        
        Args:
            model_key: Clave del modelo a descargar.
            force: Si es True, fuerza la descarga incluso si el modelo está en uso.
            
        Returns:
            True si el modelo se descargó correctamente, False en caso contrario.
        """
        if model_key not in self.models:
            logger.warning(f"Modelo '{model_key}' no configurado para descarga.")
            return False
        
        model_info = self.models[model_key]
        
        # Verificar si está cargado
        if not self.is_model_loaded(model_key):
            logger.debug(f"Modelo '{model_key}' ya está descargado.")
            return True
        
        # Verificar tiempo desde último uso
        time_since_last_use = time.time() - model_info["last_used"]
        if not force and time_since_last_use < PARAMETERS["MODEL_UNLOAD_DELAY"]:
            logger.debug(f"Modelo '{model_key}' usado recientemente ({time_since_last_use:.1f}s). Omitiendo descarga.")
            return False
        
        try:
            logger.info(f"Descargando modelo '{model_key}'...")
            
            # Limpiar referencias
            model_info["model"] = None
            model_info["tokenizer"] = None
            model_info["processor"] = None
            model_info["pipeline"] = None
            model_info["loaded"] = False
            self.load_status[model_key] = False
            
            # Forzar recolección de basura
            gc.collect()
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            logger.info(f"Modelo '{model_key}' descargado correctamente.")
            return True
        
        except Exception as e:
            logger.error(f"Error al descargar modelo '{model_key}': {e}")
            return False
    
    def unload_inactive_models(self, min_idle_time: int = None):
        """
        Descarga modelos inactivos para liberar memoria.
        
        Args:
            min_idle_time: Tiempo mínimo de inactividad en segundos para descargar.
                          Si es None, usa el valor de configuración.
        """
        if min_idle_time is None:
            min_idle_time = PARAMETERS["MODEL_UNLOAD_DELAY"]
        
        current_time = time.time()
        
        # Ordenar modelos por tiempo de inactividad (más inactivos primero)
        models_to_check = []
        for key, info in self.models.items():
            if self.is_model_loaded(key):
                idle_time = current_time - info["last_used"]
                models_to_check.append((key, idle_time))
        
        models_to_check.sort(key=lambda x: x[1], reverse=True)
        
        # Descargar modelos inactivos
        for key, idle_time in models_to_check:
            if idle_time >= min_idle_time:
                logger.info(f"Descargando modelo inactivo '{key}' (inactivo por {idle_time:.1f}s)")
                self.unload_model(key)
    
    def _check_memory_usage(self, context: str = "") -> bool:
        """
        Verifica el uso de memoria.
        
        Args:
            context: Contexto para el registro.
            
        Returns:
            True si la memoria está alta, False en caso contrario.
        """
        import psutil
        
        # Verificar memoria RAM
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Verificar memoria GPU si está disponible
        gpu_memory_percent = 0
        try:
            if self.device.type == 'cuda':
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Asum
(Content truncated due to size limit. Use line ranges to read in chunks)