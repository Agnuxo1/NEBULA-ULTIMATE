"""
Implementación del Generador de Texto para NEBULA.

Esta clase proporciona funcionalidades avanzadas para generación de texto
utilizando modelos de lenguaje grandes (LLMs).
"""

import logging
import time
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from utils.config import PARAMETERS
from utils.helpers import require_llm

logger = logging.getLogger("NEBULA.TextGenerator")

class TextGenerator:
    """
    Generador de texto que utiliza modelos de lenguaje grandes (LLMs).
    
    Características:
    - Generación de texto con diferentes estilos y formatos
    - Continuación de texto existente
    - Resumen de textos largos
    - Generación de respuestas a preguntas
    - Integración con el grafo de conocimiento
    """
    
    def __init__(self, llm_manager=None, knowledge_graph=None):
        """
        Inicializa el generador de texto.
        
        Args:
            llm_manager: Gestor de modelos LLM.
            knowledge_graph: Grafo de conocimiento para enriquecer generaciones.
        """
        logger.info("Inicializando TextGenerator...")
        self.llm_manager = llm_manager
        self.knowledge_graph = knowledge_graph
        
        # Contadores y estadísticas
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.last_generation_time = 0
        
        # Plantillas para prompts
        self.templates = self._initialize_templates()
        
        logger.info("TextGenerator inicializado correctamente.")
    
    def _initialize_templates(self) -> Dict[str, str]:
        """
        Inicializa plantillas para diferentes tipos de generación.
        
        Returns:
            Diccionario de plantillas.
        """
        templates = {
            "continue": "Continúa el siguiente texto de manera coherente y natural:\n\n{text}\n\n",
            
            "summarize": "Resume el siguiente texto en {length} palabras, manteniendo los puntos principales:\n\n{text}\n\n",
            
            "explain": "Explica el siguiente concepto de manera {style}:\n\n{concept}\n\n",
            
            "creative": "Escribe un {format} creativo sobre {topic} en estilo {style}.\n\n",
            
            "academic": "Escribe un texto académico sobre {topic} con el siguiente formato:\n\n{outline}\n\n",
            
            "dialogue": "Escribe un diálogo entre {characters} sobre {topic}.\n\n",
            
            "question": "Responde la siguiente pregunta de manera {style}:\n\n{question}\n\n",
            
            "code_comment": "Añade comentarios detallados al siguiente código:\n\n```{language}\n{code}\n```\n\n",
            
            "code_generate": "Escribe código en {language} para {task}. Incluye comentarios explicativos.\n\n",
            
            "code_complete": "Completa el siguiente código en {language}:\n\n```{language}\n{code}\n```\n\n",
            
            "code_refactor": "Refactoriza el siguiente código en {language} para mejorar {aspect}:\n\n```{language}\n{code}\n```\n\n",
        }
        
        return templates
    
    @require_llm("llm_manager")
    def generate(self, prompt: str, max_length: int = 150, model_size: str = "small", **kwargs) -> str:
        """
        Genera texto a partir de un prompt.
        
        Args:
            prompt: Texto de entrada.
            max_length: Longitud máxima de la generación.
            model_size: Tamaño del modelo a usar ('small' o 'large').
            **kwargs: Argumentos adicionales para la generación.
            
        Returns:
            Texto generado.
        """
        start_time = time.time()
        
        # Verificar disponibilidad de LLM Manager
        if not self.llm_manager:
            logger.error("LLM Manager no disponible para generación de texto.")
            return "Error: LLM Manager no disponible."
        
        # Generar texto
        generated_text = self.llm_manager.generate_text(prompt, max_length, model_size, **kwargs)
        
        # Actualizar estadísticas
        self.generation_count += 1
        self.last_generation_time = time.time()
        
        # Estimar tokens generados (aproximación simple)
        estimated_tokens = len(generated_text.split())
        self.total_tokens_generated += estimated_tokens
        
        generation_time = time.time() - start_time
        logger.debug(f"Texto generado en {generation_time:.2f}s ({estimated_tokens} tokens estimados)")
        
        return generated_text
    
    def continue_text(self, text: str, length: int = 150, model_size: str = "small") -> str:
        """
        Continúa un texto existente.
        
        Args:
            text: Texto a continuar.
            length: Longitud aproximada de la continuación.
            model_size: Tamaño del modelo a usar.
            
        Returns:
            Continuación del texto.
        """
        # Preparar prompt usando plantilla
        prompt = self.templates["continue"].format(text=text)
        
        # Generar continuación
        continuation = self.generate(prompt, max_length=length, model_size=model_size)
        
        return continuation
    
    def summarize(self, text: str, length: str = "breve", model_size: str = "small") -> str:
        """
        Resume un texto.
        
        Args:
            text: Texto a resumir.
            length: Longitud del resumen ('breve', 'medio', 'detallado').
            model_size: Tamaño del modelo a usar.
            
        Returns:
            Resumen del texto.
        """
        # Determinar longitud en palabras según parámetro
        if length == "breve":
            word_count = "100"
        elif length == "medio":
            word_count = "250"
        elif length == "detallado":
            word_count = "500"
        else:
            word_count = length  # Usar valor literal si no es una opción predefinida
        
        # Preparar prompt usando plantilla
        prompt = self.templates["summarize"].format(text=text, length=word_count)
        
        # Generar resumen
        summary = self.generate(prompt, max_length=int(word_count) * 2, model_size=model_size)
        
        return summary
    
    def explain_concept(self, concept: str, style: str = "clara", model_size: str = "small") -> str:
        """
        Explica un concepto.
        
        Args:
            concept: Concepto a explicar.
            style: Estilo de explicación ('clara', 'detallada', 'simple', 'técnica').
            model_size: Tamaño del modelo a usar.
            
        Returns:
            Explicación del concepto.
        """
        # Enriquecer con conocimiento del grafo si está disponible
        context = ""
        if self.knowledge_graph:
            try:
                # Buscar concepto en el grafo
                similar_concepts = self.knowledge_graph.find_similar_concepts(concept, top_k=3)
                
                if similar_concepts:
                    # Recopilar información relacionada
                    for concept_name, _ in similar_concepts:
                        # Obtener relaciones salientes
                        relations = self.knowledge_graph.get_relations(concept_name)
                        for source, target, data in relations:
                            relation_type = data.get('relation_type', 'relacionado_con')
                            context += f"- {source} {relation_type} {target}\n"
                        
                        # Obtener relaciones entrantes
                        incoming = self.knowledge_graph.get_incoming_relations(concept_name)
                        for source, target, data in incoming:
                            relation_type = data.get('relation_type', 'relacionado_con')
                            context += f"- {source} {relation_type} {target}\n"
                
                if context:
                    context = f"\n\nInformación relacionada:\n{context}"
            except Exception as e:
                logger.error(f"Error al enriquecer explicación con grafo de conocimiento: {e}")
        
        # Preparar prompt usando plantilla
        prompt = self.templates["explain"].format(concept=concept, style=style) + context
        
        # Generar explicación
        explanation = self.generate(prompt, max_length=300, model_size=model_size)
        
        return explanation
    
    def creative_writing(self, format_type: str, topic: str, style: str = "narrativo", model_size: str = "large") -> str:
        """
        Genera un texto creativo.
        
        Args:
            format_type: Formato del texto ('historia', 'poema', 'ensayo', etc.).
            topic: Tema del texto.
            style: Estilo de escritura.
            model_size: Tamaño del modelo a usar.
            
        Returns:
            Texto creativo generado.
        """
        # Preparar prompt usando plantilla
        prompt = self.templates["creative"].format(format=format_type, topic=topic, style=style)
        
        # Determinar longitud según formato
        if format_type.lower() in ["poema", "haiku", "verso"]:
            max_length = 200
        elif format_type.lower() in ["historia corta", "cuento", "fábula"]:
            max_length = 500
        else:
            max_length = 300
        
        # Generar texto creativo
        creative_text = self.generate(prompt, max_length=max_length, model_size=model_size, 
                                     temperature=0.8, top_p=0.95)  # Mayor temperatura para creatividad
        
        return creative_text
    
    def academic_writing(self, topic: str, outline: str = None, model_size: str = "large") -> str:
        """
        Genera un texto académico.
        
        Args:
            topic: Tema del texto.
            outline: Esquema del texto (opcional).
            model_size: Tamaño del modelo a usar.
            
        Returns:
            Texto académico generado.
        """
        # Crear esquema por defecto si no se proporciona
        if not outline:
            outline = f"""1. Introducción
2. Antecedentes teóricos sobre {topic}
3. Análisis de aspectos principales
4. Discusión de implicaciones
5. Conclusiones"""
        
        # Preparar prompt usando plantilla
        prompt = self.templates["academic"].format(topic=topic, outline=outline)
        
        # Generar texto académico
        academic_text = self.generate(prompt, max_length=500, model_size=model_size, 
                                     temperature=0.6)  # Menor temperatura para texto más formal
        
        return academic_text
    
    def generate_dialogue(self, characters: str, topic: str, model_size: str = "small") -> str:
        """
        Genera un diálogo entre personajes.
        
        Args:
            characters: Personajes del diálogo (ej. "Juan y María").
            topic: Tema del diálogo.
            model_size: Tamaño del modelo a usar.
            
        Returns:
            Diálogo generado.
        """
        # Preparar prompt usando plantilla
        prompt = self.templates["dialogue"].format(characters=characters, topic=topic)
        
        # Generar diálogo
        dialogue = self.generate(prompt, max_length=300, model_size=model_size, 
                                temperature=0.75)  # Temperatura media para diálogo natural
        
        return dialogue
    
    def answer_question(self, question: str, style: str = "informativa", model_size: str = "small") -> str:
        """
        Responde una pregunta.
        
        Args:
            question: Pregunta a responder.
            style: Estilo de respuesta ('informativa', 'concisa', 'detallada').
            model_size: Tamaño del modelo a usar.
            
        Returns:
            Respuesta a la pregunta.
        """
        # Enriquecer con conocimiento del grafo si está disponible
        context = ""
        if self.knowledge_graph:
            try:
                # Extraer conceptos clave de la pregunta (implementación simple)
                # En una implementación completa, usar NER o extracción de entidades
                words = re.findall(r'\b\w+\b', question.lower())
                stopwords = {"que", "cual", "como", "donde", "quien", "por", "para", "con", "del", "al", "la", "el", "los", "las", "un", "una", "unos", "unas", "y", "o", "pero", "si", "no", "en", "a", "de"}
                keywords = [word for word in words if word not in stopwords and len(word) > 3]
                
                # Buscar conceptos relacionados en el grafo
                related_info = []
                for keyword in keywords:
                    similar_concepts = self.knowledge_graph.find_similar_concepts(keyword, top_k=2)
                    for concept_name, similarity in similar_concepts:
                        if similarity > 0.6:  # Umbral de similitud
                            # Obtener relaciones
                            relations = self.knowledge_graph.get_relations(concept_name)
                            for source, target, data in relations:
                                relation_type = data.get('relation_type', 'relacionado_con')
                                related_info.append(f"{source} {relation_type} {target}")
                
                if related_info:
                    context = "\n\nInformación relacionada:\n" + "\n".join(f"- {info}" for info in related_info[:5])
            except Exception as e:
                logger.error(f"Error al enriquecer respuesta con grafo de conocimiento: {e}")
        
        # Preparar prompt usando plantilla
        prompt = self.templates["question"].format(question=question, style=style) + context
        
        # Generar respuesta
        answer = self.generate(prompt, max_length=250, model_size=model_size)
        
        return answer
    
    @require_llm("llm_manager")
    def generate_code(self, language: str, task: str, model_size: str = "large") -> str:
        """
        Genera código para una tarea específica.
        
        Args:
            language: Lenguaje de programación.
            task: Descripción de la tarea.
            model_size: Tamaño del modelo a usar.
            
        Returns:
            Código generado.
        """
        # Verificar disponibilidad de LLM Manager
        if not self.llm_manager:
            logger.error("LLM Manager no disponible para generación de código.")
            return "Error: LLM Manager no disponible."
        
        # Preparar prompt usando plantilla
        prompt = self.templates["code_generate"].format(language=language, task=task)
        
        # Usar modelo específico de código si está disponible
        if hasattr(self.llm_manager, "generate_code"):
            code = self.llm_manager.generate_code(prompt, max_length=500)
        else:
            # Fallback a generación de texto normal
            code = self.llm_manager.generate_text(prompt, max_length=500, model_size=model_size, 
                                                temperature=0.4)  # Temperatura baja para código más determinista
        
        # Actualizar estadísticas
        self.generation_count += 1
        self.last_generation_time = time.time()
        
        # Estimar tokens generados (aproximación simple)
        estimated_tokens = len(code.split())
        self.total_tokens_generated += estimated_tokens
        
        return code
    
    def complete_code(self, language: str, code: str, model_size: str = "large") -> str:
        """
        Completa un fragmento de código.
        
        Args:
            language: Lenguaje de programación.
            code: Código a completar.
            model_size: Tamaño del modelo a usar.
            
        Returns:
            Código completado.
        """
        # Preparar prompt usando plantilla
        prompt = self.templates["code_complete"].format(language=language, code=code)
        
        # U
(Content truncated due to size limit. Use line ranges to read in chunks)