"""
Implementación del Grafo de Conocimiento Mejorado para NEBULA.

Esta clase proporciona una representación estructurada del conocimiento
mediante un grafo donde los nodos son conceptos y las aristas son relaciones.
"""

import logging
import time
import os
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path

import numpy as np
import torch
import networkx as nx
from sentence_transformers import SentenceTransformer

from utils.config import PARAMETERS
from utils.helpers import calculate_similarity, require_llm

logger = logging.getLogger("NEBULA.KnowledgeGraph")

class EnhancedKnowledgeGraph:
    """
    Grafo de conocimiento mejorado que representa conceptos y relaciones.
    
    Características:
    - Representación mediante grafo dirigido (NetworkX)
    - Embeddings semánticos para nodos y relaciones
    - Búsqueda por similitud semántica
    - Inferencia de relaciones implícitas
    - Integración con memoria holográfica
    """
    
    def __init__(self, device: torch.device = PARAMETERS["DEVICE"]):
        """
        Inicializa el grafo de conocimiento.
        
        Args:
            device: Dispositivo de PyTorch para cálculos tensoriales.
        """
        logger.info("Inicializando EnhancedKnowledgeGraph...")
        self.device = device
        
        # Inicializar grafo dirigido
        self.graph = nx.DiGraph()
        
        # Modelo de embeddings
        self.embedding_model = None
        self.embedding_dim = 384  # Dimensión por defecto para all-MiniLM-L6-v2
        
        # Caché de embeddings
        self.node_embeddings = {}
        self.relation_embeddings = {}
        
        # Contadores y estadísticas
        self.last_modified = time.time()
        self.total_queries = 0
        self.successful_queries = 0
        
        # Cargar grafo existente si está disponible
        self.graph_file = PARAMETERS["KNOWLEDGE_GRAPH_FILE"]
        self._load_graph()
        
        logger.info(f"EnhancedKnowledgeGraph inicializado con {self.graph.number_of_nodes()} nodos y {self.graph.number_of_edges()} relaciones.")
    
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
    
    def _load_graph(self):
        """Carga el grafo desde archivo si existe."""
        if not os.path.exists(self.graph_file):
            logger.info(f"No se encontró archivo de grafo en {self.graph_file}. Iniciando con grafo vacío.")
            return
        
        try:
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(self.graph_file), exist_ok=True)
            
            # Cargar grafo
            self.graph = nx.read_graphml(self.graph_file)
            
            # Convertir atributos de nodos y aristas a tipos correctos
            for node, data in self.graph.nodes(data=True):
                # Convertir embedding de string a numpy array si existe
                if 'embedding' in data and isinstance(data['embedding'], str):
                    try:
                        data['embedding'] = np.array(json.loads(data['embedding']))
                    except:
                        data['embedding'] = None
            
            for u, v, data in self.graph.edges(data=True):
                # Convertir peso a float
                if 'weight' in data and not isinstance(data['weight'], float):
                    try:
                        data['weight'] = float(data['weight'])
                    except:
                        data['weight'] = 1.0
                
                # Convertir embedding de relación
                if 'embedding' in data and isinstance(data['embedding'], str):
                    try:
                        data['embedding'] = np.array(json.loads(data['embedding']))
                    except:
                        data['embedding'] = None
            
            # Reconstruir caché de embeddings
            self.node_embeddings = {}
            self.relation_embeddings = {}
            
            for node, data in self.graph.nodes(data=True):
                if 'embedding' in data and data['embedding'] is not None:
                    self.node_embeddings[node] = data['embedding']
            
            logger.info(f"Grafo cargado correctamente con {self.graph.number_of_nodes()} nodos y {self.graph.number_of_edges()} relaciones.")
        except Exception as e:
            logger.error(f"Error al cargar grafo: {e}", exc_info=True)
            # Iniciar con grafo vacío en caso de error
            self.graph = nx.DiGraph()
    
    def save_graph(self):
        """Guarda el grafo en archivo."""
        try:
            # Asegurar que el directorio existe
            os.makedirs(os.path.dirname(self.graph_file), exist_ok=True)
            
            # Preparar grafo para guardado (convertir numpy arrays a strings)
            graph_to_save = self.graph.copy()
            
            for node, data in graph_to_save.nodes(data=True):
                if 'embedding' in data and data['embedding'] is not None:
                    if isinstance(data['embedding'], np.ndarray):
                        data['embedding'] = json.dumps(data['embedding'].tolist())
            
            for u, v, data in graph_to_save.edges(data=True):
                if 'embedding' in data and data['embedding'] is not None:
                    if isinstance(data['embedding'], np.ndarray):
                        data['embedding'] = json.dumps(data['embedding'].tolist())
            
            # Guardar grafo
            nx.write_graphml(graph_to_save, self.graph_file)
            logger.info(f"Grafo guardado correctamente en {self.graph_file}")
        except Exception as e:
            logger.error(f"Error al guardar grafo: {e}", exc_info=True)
    
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
            text = text.strip().lower()
            
            # Obtener embedding
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error al obtener embedding para '{text}': {e}")
            return np.random.randn(self.embedding_dim)
    
    def add_node(self, concept: str, attributes: Optional[Dict[str, Any]] = None) -> bool:
        """
        Añade un nodo (concepto) al grafo.
        
        Args:
            concept: Concepto a añadir.
            attributes: Atributos adicionales del concepto.
            
        Returns:
            True si se añadió correctamente, False si ya existía o hubo error.
        """
        # Normalizar concepto
        concept = concept.strip()
        if not concept:
            logger.warning("Intento de añadir concepto vacío. Ignorando.")
            return False
        
        # Verificar si ya existe
        if self.graph.has_node(concept):
            logger.debug(f"Concepto '{concept}' ya existe en el grafo.")
            return False
        
        try:
            # Preparar atributos
            node_attrs = attributes or {}
            
            # Obtener embedding si no se proporciona
            if 'embedding' not in node_attrs:
                embedding = self.get_embedding(concept)
                node_attrs['embedding'] = embedding
                self.node_embeddings[concept] = embedding
            
            # Añadir metadatos
            node_attrs['created'] = time.time()
            node_attrs['updated'] = time.time()
            node_attrs['access_count'] = 0
            
            # Añadir nodo al grafo
            self.graph.add_node(concept, **node_attrs)
            self.last_modified = time.time()
            
            logger.debug(f"Añadido concepto '{concept}' al grafo.")
            return True
        except Exception as e:
            logger.error(f"Error al añadir concepto '{concept}': {e}")
            return False
    
    def add_relation(self, source: str, relation: str, target: str, weight: float = 1.0, attributes: Optional[Dict[str, Any]] = None) -> bool:
        """
        Añade una relación entre dos conceptos.
        
        Args:
            source: Concepto origen.
            relation: Tipo de relación.
            target: Concepto destino.
            weight: Peso de la relación.
            attributes: Atributos adicionales de la relación.
            
        Returns:
            True si se añadió correctamente, False en caso contrario.
        """
        # Normalizar conceptos
        source = source.strip()
        relation = relation.strip()
        target = target.strip()
        
        if not source or not relation or not target:
            logger.warning("Intento de añadir relación con campos vacíos. Ignorando.")
            return False
        
        try:
            # Añadir nodos si no existen
            if not self.graph.has_node(source):
                self.add_node(source)
            
            if not self.graph.has_node(target):
                self.add_node(target)
            
            # Preparar atributos
            edge_attrs = attributes or {}
            edge_attrs['relation_type'] = relation
            edge_attrs['weight'] = weight
            
            # Obtener embedding de la relación si no se proporciona
            if 'embedding' not in edge_attrs:
                relation_text = f"{source} {relation} {target}"
                embedding = self.get_embedding(relation_text)
                edge_attrs['embedding'] = embedding
                
                # Almacenar en caché de relaciones
                relation_key = (source, relation, target)
                self.relation_embeddings[relation_key] = embedding
            
            # Añadir metadatos
            edge_attrs['created'] = time.time()
            edge_attrs['updated'] = time.time()
            
            # Añadir relación al grafo
            self.graph.add_edge(source, target, **edge_attrs)
            self.last_modified = time.time()
            
            logger.debug(f"Añadida relación '{source} --[{relation}]--> {target}'")
            return True
        except Exception as e:
            logger.error(f"Error al añadir relación '{source} --[{relation}]--> {target}': {e}")
            return False
    
    def get_node(self, concept: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de un nodo.
        
        Args:
            concept: Concepto a buscar.
            
        Returns:
            Diccionario con atributos del nodo o None si no existe.
        """
        concept = concept.strip()
        if not self.graph.has_node(concept):
            return None
        
        # Incrementar contador de acceso
        self.graph.nodes[concept]['access_count'] = self.graph.nodes[concept].get('access_count', 0) + 1
        
        return dict(self.graph.nodes[concept])
    
    def get_relations(self, concept: str, relation_type: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Obtiene relaciones de un concepto.
        
        Args:
            concept: Concepto origen.
            relation_type: Tipo de relación a filtrar (opcional).
            
        Returns:
            Lista de tuplas (origen, destino, atributos).
        """
        concept = concept.strip()
        if not self.graph.has_node(concept):
            return []
        
        relations = []
        
        # Obtener relaciones salientes
        for _, target, data in self.graph.out_edges(concept, data=True):
            if relation_type is None or data.get('relation_type') == relation_type:
                relations.append((concept, target, dict(data)))
        
        return relations
    
    def get_incoming_relations(self, concept: str, relation_type: Optional[str] = None) -> List[Tuple[str, str, Dict[str, Any]]]:
        """
        Obtiene relaciones entrantes a un concepto.
        
        Args:
            concept: Concepto destino.
            relation_type: Tipo de relación a filtrar (opcional).
            
        Returns:
            Lista de tuplas (origen, destino, atributos).
        """
        concept = concept.strip()
        if not self.graph.has_node(concept):
            return []
        
        relations = []
        
        # Obtener relaciones entrantes
        for source, _, data in self.graph.in_edges(concept, data=True):
            if relation_type is None or data.get('relation_type') == relation_type:
                relations.append((source, concept, dict(data)))
        
        return relations
    
    def find_similar_concepts(self, query: str, top_k: int = 5, threshold: float = 0.6) -> List[Tuple[str, float]]:
        """
        Encuentra conceptos similares a una consulta.
        
        Args:
            query: Texto de consulta.
            top_k: Número máximo de resultados.
            threshold: Umbral mínimo de similitud.
            
        Returns:
            Lista de tuplas (concepto, similitud).
        """
        self.total_queries += 1
        
        if not self.graph.nodes:
            logger.warning("Grafo vacío, no se pueden encontrar conceptos similares.")
            return []
        
        try:
            # Obtener embedding de la consulta
            query_embedding = self.get_embedding(query)
            
            # Calcular similitud con todos los nodos
            similarities = []
            
            for node, embedding in self.node_embeddings.items():
                if embedding is not None:
                    similarity = calculate_similarity(query_embedding, embedding)
                    if similarity >= threshold:
                        similarities.append((node, similarity))
            
            # Ordenar por similitud descendente
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Limitar resultados
            results = similarities[:top_k]
            
            if results:
                self.successful_queries += 1
            
            return results
        except Exception as e:
            logger.error(f"Error al buscar conceptos similares a '{query}': {e}")
            return []
    
    def find_path(self, source: str, target: str, max_length: int = 3) -> List[List[Tuple[str, str, str]]]:
        """
        Encuentra caminos entre dos conceptos.
        
        Args:
            source: Concepto origen.
            target: Concepto destino.
            max_length: Longitud máxima del camino.
   
(Content truncated due to size limit. Use line ranges to read in chunks)