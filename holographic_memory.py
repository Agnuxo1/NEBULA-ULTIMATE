"""
Implementación de la Memoria Holográfica para NEBULA.

Esta clase proporciona un sistema de memoria asociativa basado en vectores
que permite almacenar y recuperar información por similitud semántica.
"""

import logging
import time
import os
import pickle
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer

from utils.config import PARAMETERS
from utils.helpers import calculate_similarity

logger = logging.getLogger("NEBULA.HolographicMemory")

class HolographicMemory:
    """
    Memoria holográfica que almacena información como vectores y permite
    recuperación por similitud semántica.
    
    Características:
    - Almacenamiento de pares clave-valor con embeddings
    - Búsqueda eficiente por similitud mediante FAISS
    - Organización en múltiples espacios de memoria
    - Consolidación periódica de memoria
    """
    
    def __init__(self, device: torch.device = PARAMETERS["DEVICE"]):
        """
        Inicializa la memoria holográfica.
        
        Args:
            device: Dispositivo de PyTorch para cálculos tensoriales.
        """
        logger.info("Inicializando HolographicMemory...")
        self.device = device
        
        # Modelo de embeddings
        self.embedding_model = None
        self.embedding_dim = 384  # Dimensión por defecto para all-MiniLM-L6-v2
        
        # Índices FAISS para búsqueda eficiente
        self.indices = {}
        
        # Almacenamiento de memoria
        self.memory_spaces = defaultdict(list)
        self.memory_metadata = defaultdict(dict)
        
        # Contadores y estadísticas
        self.total_items = 0
        self.total_retrievals = 0
        self.successful_retrievals = 0
        self.last_consolidated = time.time()
        
        # Archivo de memoria
        self.memory_file = PARAMETERS["BACKUP_DIRECTORY"] / "holographic_memory.pkl"
        self._load_memory()
        
        logger.info(f"HolographicMemory inicializada con {self.total_items} elementos en {len(self.memory_spaces)} espacios.")
    
    def _load_embedding_model(self) -> bool:
        """
        Carga el modelo de embeddings.
        
        Returns:
            True si el modelo se cargó correctamente, False en caso contrario.
        """
        if self.embedding_model is not None:
            return True
        
        try:
            model_name = PARAMETERS["EMBEDDING_MODEL_NAME"]
            cache_dir = PARAMETERS["MODEL_CACHE_DIR"]
            
            logger.info(f"Cargando modelo de embeddings: {model_name}")
            self.embedding_model = SentenceTransformer(model_name, cache_folder=str(cache_dir))
            
            # Mover modelo a dispositivo adecuado
            if self.device.type == 'cuda':
                self.embedding_model.to(self.device)
            
            logger.info(f"Modelo de embeddings cargado correctamente.")
            return True
        except Exception as e:
            logger.error(f"Error al cargar modelo de embeddings: {e}", exc_info=True)
            return False
    
    def _load_memory(self):
        """Carga la memoria desde archivo si existe."""
        if not os.path.exists(self.memory_file):
            logger.info(f"No se encontró archivo de memoria en {self.memory_file}. Iniciando con memoria vacía.")
            return
        
        try:
            # Cargar memoria
            with open(self.memory_file, 'rb') as f:
                memory_data = pickle.load(f)
            
            self.memory_spaces = memory_data.get('memory_spaces', defaultdict(list))
            self.memory_metadata = memory_data.get('memory_metadata', defaultdict(dict))
            self.total_items = memory_data.get('total_items', 0)
            self.total_retrievals = memory_data.get('total_retrievals', 0)
            self.successful_retrievals = memory_data.get('successful_retrievals', 0)
            self.last_consolidated = memory_data.get('last_consolidated', time.time())
            
            # Reconstruir índices FAISS
            self._rebuild_indices()
            
            logger.info(f"Memoria cargada correctamente con {self.total_items} elementos en {len(self.memory_spaces)} espacios.")
        except Exception as e:
            logger.error(f"Error al cargar memoria: {e}", exc_info=True)
            # Iniciar con memoria vacía en caso de error
            self.memory_spaces = defaultdict(list)
            self.memory_metadata = defaultdict(dict)
            self.total_items = 0
    
    def save_memory(self):
        """Guarda la memoria en archivo."""
        try:
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
            
            # Preparar datos para guardado
            memory_data = {
                'memory_spaces': dict(self.memory_spaces),
                'memory_metadata': dict(self.memory_metadata),
                'total_items': self.total_items,
                'total_retrievals': self.total_retrievals,
                'successful_retrievals': self.successful_retrievals,
                'last_consolidated': self.last_consolidated,
            }
            
            # Guardar memoria
            with open(self.memory_file, 'wb') as f:
                pickle.dump(memory_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f"Memoria guardada correctamente en {self.memory_file}")
        except Exception as e:
            logger.error(f"Error al guardar memoria: {e}", exc_info=True)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Obtiene el embedding de un texto.
        
        Args:
            text: Texto a convertir en embedding.
            
        Returns:
            Vector de embedding.
        """
        if not self._load_embedding_model():
            # Devolver vector aleatorio si no se puede cargar el modelo
            logger.warning("Usando embedding aleatorio debido a error en modelo.")
            return np.random.randn(self.embedding_dim)
        
        try:
            # Normalizar texto
            text = text.strip()
            
            # Obtener embedding
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error al obtener embedding para '{text}': {e}")
            return np.random.randn(self.embedding_dim)
    
    def _rebuild_indices(self):
        """Reconstruye los índices FAISS para todos los espacios de memoria."""
        self.indices = {}
        
        for space_name, items in self.memory_spaces.items():
            if not items:
                continue
            
            try:
                # Extraer embeddings
                embeddings = np.array([item['embedding'] for item in items if 'embedding' in item])
                
                if len(embeddings) == 0:
                    continue
                
                # Crear índice FAISS
                dim = embeddings.shape[1]
                index = faiss.IndexFlatL2(dim)
                index.add(embeddings)
                
                self.indices[space_name] = index
                
                logger.debug(f"Reconstruido índice FAISS para espacio '{space_name}' con {len(embeddings)} elementos.")
            except Exception as e:
                logger.error(f"Error al reconstruir índice para espacio '{space_name}': {e}")
    
    def store(self, key: str, value: Any, space: str = "default", metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Almacena un par clave-valor en la memoria.
        
        Args:
            key: Clave o descripción del elemento.
            value: Valor a almacenar.
            space: Espacio de memoria donde almacenar.
            metadata: Metadatos adicionales.
            
        Returns:
            True si se almacenó correctamente, False en caso contrario.
        """
        try:
            # Obtener embedding de la clave
            embedding = self.get_embedding(key)
            
            # Preparar elemento de memoria
            memory_item = {
                'key': key,
                'value': value,
                'embedding': embedding,
                'timestamp': time.time(),
                'access_count': 0,
                'metadata': metadata or {}
            }
            
            # Almacenar en espacio de memoria
            self.memory_spaces[space].append(memory_item)
            
            # Actualizar índice FAISS
            if space in self.indices:
                try:
                    self.indices[space].add(np.array([embedding]))
                except Exception as e:
                    logger.error(f"Error al actualizar índice FAISS para '{space}': {e}")
                    # Reconstruir índice completo si falla la actualización
                    self._rebuild_indices()
            else:
                # Crear nuevo índice
                index = faiss.IndexFlatL2(embedding.shape[0])
                index.add(np.array([embedding]))
                self.indices[space] = index
            
            # Actualizar metadatos del espacio
            if space not in self.memory_metadata:
                self.memory_metadata[space] = {
                    'created': time.time(),
                    'item_count': 0,
                    'last_access': time.time(),
                    'description': f"Espacio de memoria: {space}"
                }
            
            self.memory_metadata[space]['item_count'] = len(self.memory_spaces[space])
            self.memory_metadata[space]['last_modified'] = time.time()
            
            # Actualizar contador global
            self.total_items += 1
            
            logger.debug(f"Almacenado elemento con clave '{key}' en espacio '{space}'")
            return True
        except Exception as e:
            logger.error(f"Error al almacenar elemento con clave '{key}': {e}")
            return False
    
    def retrieve(self, query: str, space: Optional[str] = None, top_k: int = 5, threshold: float = 0.65) -> List[Dict[str, Any]]:
        """
        Recupera elementos similares a la consulta.
        
        Args:
            query: Texto de consulta.
            space: Espacio de memoria específico (None para buscar en todos).
            top_k: Número máximo de resultados.
            threshold: Umbral mínimo de similitud.
            
        Returns:
            Lista de elementos recuperados con sus metadatos.
        """
        self.total_retrievals += 1
        
        try:
            # Obtener embedding de la consulta
            query_embedding = self.get_embedding(query)
            
            # Determinar espacios a buscar
            spaces_to_search = [space] if space else list(self.memory_spaces.keys())
            
            all_results = []
            
            for space_name in spaces_to_search:
                if space_name not in self.memory_spaces or not self.memory_spaces[space_name]:
                    continue
                
                # Actualizar metadatos del espacio
                if space_name in self.memory_metadata:
                    self.memory_metadata[space_name]['last_access'] = time.time()
                
                # Buscar usando FAISS si está disponible
                if space_name in self.indices:
                    index = self.indices[space_name]
                    items = self.memory_spaces[space_name]
                    
                    # Realizar búsqueda
                    distances, indices = index.search(np.array([query_embedding]), min(top_k, len(items)))
                    
                    # Procesar resultados
                    for i, idx in enumerate(indices[0]):
                        if idx < 0 or idx >= len(items):
                            continue
                        
                        item = items[idx]
                        distance = distances[0][i]
                        
                        # Convertir distancia L2 a similitud (aproximación)
                        # Nota: Esta es una aproximación simple, no exacta
                        max_distance = 4.0  # Valor máximo típico para distancia L2 entre embeddings normalizados
                        similarity = max(0.0, 1.0 - (distance / max_distance))
                        
                        if similarity >= threshold:
                            # Incrementar contador de acceso
                            item['access_count'] = item.get('access_count', 0) + 1
                            
                            # Añadir a resultados
                            result = item.copy()
                            result['similarity'] = similarity
                            result['space'] = space_name
                            
                            # Eliminar embedding del resultado (no necesario para el usuario)
                            if 'embedding' in result:
                                del result['embedding']
                            
                            all_results.append(result)
                else:
                    # Búsqueda manual si no hay índice FAISS
                    items = self.memory_spaces[space_name]
                    
                    for item in items:
                        if 'embedding' not in item:
                            continue
                        
                        similarity = calculate_similarity(query_embedding, item['embedding'])
                        
                        if similarity >= threshold:
                            # Incrementar contador de acceso
                            item['access_count'] = item.get('access_count', 0) + 1
                            
                            # Añadir a resultados
                            result = item.copy()
                            result['similarity'] = similarity
                            result['space'] = space_name
                            
                            # Eliminar embedding del resultado
                            if 'embedding' in result:
                                del result['embedding']
                            
                            all_results.append(result)
            
            # Ordenar por similitud
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Limitar resultados
            final_results = all_results[:top_k]
            
            if final_results:
                self.successful_retrievals += 1
            
            return final_results
        except Exception as e:
            logger.error(f"Error al recuperar elementos para consulta '{query}': {e}")
            return []
    
    def update(self, key: str, new_value: Any, space: str = "default") -> bool:
        """
        Actualiza un elemento existente en la memoria.
        
        Args:
            key: Clave exacta del elemento a actualizar.
            new_value: Nuevo valor.
            space: Espacio de memoria.
            
        Returns:
            True si se actualizó correctamente, False en caso contrario.
        """
        if space not in self.memory_spaces:
            logger.warning(f"Espacio '{space}' no existe para actualización.")
            return False
        
        try:
            # Buscar elemento por clave exacta
            for item in self.memory_spaces[space]:
                if item['key'] == key:
                    # Actualizar valor
                    item['value'] = new_value
                    item['timestamp'] = time.time()
                    
                    # Actualizar metadatos del espacio
         
(Content truncated due to size limit. Use line ranges to read in chunks)