"""
Implementación del Sistema de Preguntas y Respuestas para NEBULA.

Esta clase proporciona funcionalidades avanzadas para responder preguntas
utilizando modelos de lenguaje y el conocimiento almacenado en el sistema.
"""

import logging
import time
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from utils.config import PARAMETERS
from utils.helpers import require_llm, calculate_similarity

logger = logging.getLogger("NEBULA.QASystem")

class QASystem:
    """
    Sistema de preguntas y respuestas que combina modelos de lenguaje con
    conocimiento estructurado para proporcionar respuestas precisas.
    
    Características:
    - Respuesta a preguntas basada en contexto
    - Búsqueda de información relevante en el grafo de conocimiento
    - Recuperación de información de la memoria holográfica
    - Generación de respuestas con diferentes niveles de detalle
    """
    
    def __init__(self, llm_manager=None, knowledge_graph=None, holographic_memory=None):
        """
        Inicializa el sistema de preguntas y respuestas.
        
        Args:
            llm_manager: Gestor de modelos LLM.
            knowledge_graph: Grafo de conocimiento para búsqueda de información.
            holographic_memory: Memoria holográfica para recuperación de información.
        """
        logger.info("Inicializando QASystem...")
        self.llm_manager = llm_manager
        self.knowledge_graph = knowledge_graph
        self.holographic_memory = holographic_memory
        
        # Contadores y estadísticas
        self.query_count = 0
        self.successful_queries = 0
        self.last_query_time = 0
        
        # Plantillas para prompts
        self.templates = self._initialize_templates()
        
        logger.info("QASystem inicializado correctamente.")
    
    def _initialize_templates(self) -> Dict[str, str]:
        """
        Inicializa plantillas para diferentes tipos de consultas.
        
        Returns:
            Diccionario de plantillas.
        """
        templates = {
            "basic_qa": "Responde la siguiente pregunta de manera concisa y precisa:\n\nPregunta: {question}\n\n",
            
            "context_qa": "Utiliza el siguiente contexto para responder la pregunta:\n\nContexto:\n{context}\n\nPregunta: {question}\n\nRespuesta:",
            
            "detailed_qa": "Responde la siguiente pregunta de manera detallada, incluyendo explicaciones y ejemplos cuando sea apropiado:\n\nPregunta: {question}\n\n",
            
            "factual_qa": "Responde la siguiente pregunta factual con precisión, basándote únicamente en hechos verificables:\n\nPregunta: {question}\n\n",
            
            "opinion_qa": "Proporciona una opinión equilibrada sobre la siguiente pregunta, considerando diferentes perspectivas:\n\nPregunta: {question}\n\n",
            
            "step_by_step": "Responde la siguiente pregunta paso a paso, explicando tu razonamiento:\n\nPregunta: {question}\n\n",
            
            "multi_context": "Utiliza los siguientes fragmentos de información para responder la pregunta:\n\n{contexts}\n\nPregunta: {question}\n\nRespuesta:",
        }
        
        return templates
    
    def _extract_keywords(self, question: str) -> List[str]:
        """
        Extrae palabras clave de una pregunta.
        
        Args:
            question: Pregunta de la que extraer palabras clave.
            
        Returns:
            Lista de palabras clave.
        """
        # Implementación simple de extracción de palabras clave
        # En una implementación completa, usar NER o extracción de entidades
        words = re.findall(r'\b\w+\b', question.lower())
        stopwords = {"que", "cual", "como", "donde", "quien", "por", "para", "con", "del", "al", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero", "si", "no", "en", "a", "de"}
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        
        return keywords
    
    def _search_knowledge_graph(self, keywords: List[str], max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Busca información relevante en el grafo de conocimiento.
        
        Args:
            keywords: Palabras clave para buscar.
            max_results: Número máximo de resultados.
            
        Returns:
            Lista de fragmentos de información relevante.
        """
        if not self.knowledge_graph:
            return []
        
        results = []
        
        try:
            # Buscar conceptos relacionados con cada palabra clave
            for keyword in keywords:
                similar_concepts = self.knowledge_graph.find_similar_concepts(keyword, top_k=3)
                
                for concept_name, similarity in similar_concepts:
                    if similarity > 0.6:  # Umbral de similitud
                        # Obtener información del nodo
                        node_info = self.knowledge_graph.get_node(concept_name)
                        if node_info:
                            results.append({
                                "type": "concept",
                                "name": concept_name,
                                "similarity": similarity,
                                "attributes": node_info
                            })
                        
                        # Obtener relaciones salientes
                        relations = self.knowledge_graph.get_relations(concept_name)
                        for source, target, data in relations:
                            relation_type = data.get('relation_type', 'relacionado_con')
                            results.append({
                                "type": "relation",
                                "source": source,
                                "relation": relation_type,
                                "target": target,
                                "similarity": similarity * 0.9  # Reducir similitud para relaciones
                            })
                        
                        # Obtener relaciones entrantes
                        incoming = self.knowledge_graph.get_incoming_relations(concept_name)
                        for source, target, data in incoming:
                            relation_type = data.get('relation_type', 'relacionado_con')
                            results.append({
                                "type": "relation",
                                "source": source,
                                "relation": relation_type,
                                "target": target,
                                "similarity": similarity * 0.9  # Reducir similitud para relaciones
                            })
            
            # Ordenar por similitud y eliminar duplicados
            unique_results = []
            seen = set()
            
            for result in sorted(results, key=lambda x: x["similarity"], reverse=True):
                # Crear clave única según tipo
                if result["type"] == "concept":
                    key = f"concept:{result['name']}"
                else:
                    key = f"relation:{result['source']}:{result['relation']}:{result['target']}"
                
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
            
            return unique_results[:max_results]
        
        except Exception as e:
            logger.error(f"Error al buscar en grafo de conocimiento: {e}")
            return []
    
    def _search_holographic_memory(self, question: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """
        Busca información relevante en la memoria holográfica.
        
        Args:
            question: Pregunta para buscar.
            max_results: Número máximo de resultados.
            
        Returns:
            Lista de fragmentos de información relevante.
        """
        if not self.holographic_memory:
            return []
        
        try:
            # Buscar en memoria holográfica
            memory_results = self.holographic_memory.retrieve(
                query=question,
                space=None,  # Buscar en todos los espacios
                top_k=max_results,
                threshold=PARAMETERS["HOLOGRAPHIC_MEMORY_THRESHOLD"]
            )
            
            return memory_results
        
        except Exception as e:
            logger.error(f"Error al buscar en memoria holográfica: {e}")
            return []
    
    def _format_context(self, kg_results: List[Dict[str, Any]], memory_results: List[Dict[str, Any]]) -> str:
        """
        Formatea los resultados de búsqueda como contexto para respuesta.
        
        Args:
            kg_results: Resultados del grafo de conocimiento.
            memory_results: Resultados de la memoria holográfica.
            
        Returns:
            Contexto formateado.
        """
        context = ""
        
        # Formatear resultados del grafo de conocimiento
        if kg_results:
            context += "Información del grafo de conocimiento:\n"
            
            for result in kg_results:
                if result["type"] == "concept":
                    # Formatear información de concepto
                    context += f"- Concepto: {result['name']}\n"
                    
                    # Añadir atributos relevantes si existen
                    attrs = result.get("attributes", {})
                    for key, value in attrs.items():
                        if key not in ["embedding", "created", "updated", "access_count"]:
                            context += f"  - {key}: {value}\n"
                
                elif result["type"] == "relation":
                    # Formatear relación
                    context += f"- {result['source']} {result['relation']} {result['target']}\n"
            
            context += "\n"
        
        # Formatear resultados de la memoria holográfica
        if memory_results:
            context += "Información de la memoria:\n"
            
            for result in memory_results:
                key = result.get("key", "")
                value = result.get("value", "")
                similarity = result.get("similarity", 0.0)
                
                # Formatear según tipo de valor
                if isinstance(value, str):
                    # Si el valor es texto, incluirlo directamente
                    context += f"- {key}: {value}\n"
                elif isinstance(value, dict):
                    # Si es un diccionario, formatear como pares clave-valor
                    context += f"- {key}:\n"
                    for k, v in value.items():
                        if not k.startswith("_"):  # Ignorar claves internas
                            context += f"  - {k}: {v}\n"
                else:
                    # Para otros tipos, usar representación de cadena
                    context += f"- {key}: {str(value)}\n"
            
            context += "\n"
        
        return context
    
    @require_llm("llm_manager")
    def answer(self, question: str, mode: str = "auto", model_size: str = "small") -> Dict[str, Any]:
        """
        Responde una pregunta utilizando el conocimiento disponible.
        
        Args:
            question: Pregunta a responder.
            mode: Modo de respuesta ('auto', 'basic', 'detailed', 'factual', 'opinion', 'step_by_step').
            model_size: Tamaño del modelo a usar.
            
        Returns:
            Diccionario con respuesta y metadatos.
        """
        start_time = time.time()
        self.query_count += 1
        self.last_query_time = start_time
        
        # Verificar disponibilidad de LLM Manager
        if not self.llm_manager:
            logger.error("LLM Manager no disponible para responder preguntas.")
            return {
                "answer": "Error: LLM Manager no disponible.",
                "sources": [],
                "confidence": 0.0,
                "processing_time": time.time() - start_time
            }
        
        try:
            # Extraer palabras clave
            keywords = self._extract_keywords(question)
            
            # Buscar información relevante
            kg_results = self._search_knowledge_graph(keywords)
            memory_results = self._search_holographic_memory(question)
            
            # Determinar si tenemos suficiente contexto
            has_context = bool(kg_results or memory_results)
            
            # Seleccionar plantilla según modo y contexto disponible
            if mode == "auto":
                if has_context:
                    template_key = "context_qa"
                else:
                    # Analizar tipo de pregunta para seleccionar plantilla adecuada
                    if any(word in question.lower() for word in ["cómo", "como", "pasos", "proceso"]):
                        template_key = "step_by_step"
                    elif any(word in question.lower() for word in ["qué", "que", "cuál", "cual", "quién", "quien", "dónde", "donde", "cuándo", "cuando"]):
                        template_key = "factual_qa"
                    elif any(word in question.lower() for word in ["opina", "piensas", "crees", "mejor", "peor"]):
                        template_key = "opinion_qa"
                    else:
                        template_key = "basic_qa"
            else:
                # Usar modo especificado
                template_key = f"{mode}_qa" if mode != "context" or not has_context else "context_qa"
                
                # Fallback a básico si la plantilla no existe
                if template_key not in self.templates:
                    template_key = "basic_qa"
            
            # Preparar contexto si es necesario
            context = ""
            if template_key in ["context_qa", "multi_context"] and has_context:
                context = self._format_context(kg_results, memory_results)
            
            # Preparar prompt usando plantilla
            if template_key == "context_qa":
                prompt = self.templates[template_key].format(question=question, context=context)
            elif template_key == "multi_context":
                prompt = self.templates[template_key].format(question=question, contexts=context)
            else:
                prompt = self.templates[template_key].format(question=question)
            
            # Determinar longitud máxima según modo
            if mode in ["detailed", "step_by_step"]:
                max_length = 350
            else:
                max_length = 200
            
            # Usar modelo específico de QA si está disponible y hay contexto
            if has_context and hasattr(self.llm_manager, "answer_question") and template_key == "context_qa":
                qa_result = self.llm_manager.answer_question(question, context)
                answer = qa_result["answer"]
                confidence = qa_result["score"]
            else:
                # Generar respuesta con modelo de texto
                answer = self.llm_manager.generate_text(prompt, max_length, model_size)
                
                # Estimar confianza basada en fuentes disponibles
                if has_context:
                    # Mayor confianza si tenemos fuentes
                    confidence = 0.7 + (len(kg_results) * 0.05) + (len(memory_results) * 0.05)
                    confidence = min(confidence, 0.95)  # Limitar a 0.95
                else:
                    # Menor confianza sin fuentes
                    confidence = 0.5
            
            # Preparar fuentes
            sources = []
            
            # Añadir fuentes del grafo de conocimiento
            for result in kg_results:
                if result["type"] == "conc
(Content truncated due to size limit. Use line ranges to read in chunks)