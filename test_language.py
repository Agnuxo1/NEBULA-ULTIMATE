"""
Módulo de pruebas para los componentes de lenguaje de NEBULA.

Este módulo contiene pruebas unitarias para verificar el funcionamiento
correcto de los componentes de procesamiento de lenguaje de NEBULA.
"""

import unittest
import sys
import os
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from language.llm_manager import LLMManager
from language.text_generator import TextGenerator
from language.qa_system import QASystem
from utils.config import PARAMETERS

class TestLLMManager(unittest.TestCase):
    """Pruebas para la clase LLMManager."""
    
    def setUp(self):
        """Configuración para las pruebas."""
        self.temp_dir = tempfile.mkdtemp()
        self.llm_manager = LLMManager(
            cache_dir=self.temp_dir,
            max_models=2
        )
    
    def tearDown(self):
        """Limpieza después de las pruebas."""
        shutil.rmtree(self.temp_dir)
        # Asegurar que se liberan los modelos
        self.llm_manager.unload_all_models()
    
    def test_initialization(self):
        """Prueba la inicialización correcta del gestor de LLM."""
        self.assertEqual(self.llm_manager.max_models, 2)
        self.assertEqual(len(self.llm_manager.loaded_models), 0)
        self.assertIsNotNone(self.llm_manager.model_configs)
    
    def test_load_unload_model(self):
        """Prueba la carga y descarga de modelos."""
        # Cargar un modelo pequeño para pruebas
        model_id = self.llm_manager.load_model("small")
        
        # Verificar que el modelo se ha cargado
        self.assertIsNotNone(model_id)
        self.assertIn(model_id, self.llm_manager.loaded_models)
        
        # Descargar el modelo
        self.llm_manager.unload_model(model_id)
        
        # Verificar que el modelo se ha descargado
        self.assertNotIn(model_id, self.llm_manager.loaded_models)
    
    def test_model_rotation(self):
        """Prueba la rotación automática de modelos."""
        # Cargar el máximo de modelos permitidos
        model_id1 = self.llm_manager.load_model("small")
        model_id2 = self.llm_manager.load_model("small")
        
        # Verificar que ambos modelos están cargados
        self.assertEqual(len(self.llm_manager.loaded_models), 2)
        
        # Cargar un modelo adicional (debería descargar el más antiguo)
        model_id3 = self.llm_manager.load_model("small")
        
        # Verificar que se mantiene el límite de modelos
        self.assertEqual(len(self.llm_manager.loaded_models), 2)
        
        # Verificar que el modelo más antiguo se ha descargado
        self.assertNotIn(model_id1, self.llm_manager.loaded_models)
        self.assertIn(model_id2, self.llm_manager.loaded_models)
        self.assertIn(model_id3, self.llm_manager.loaded_models)
    
    def test_generate_text(self):
        """Prueba la generación de texto."""
        # Generar texto con un modelo pequeño
        prompt = "Explica brevemente qué es la inteligencia artificial."
        text = self.llm_manager.generate_text(prompt, max_length=50, model_size="small")
        
        # Verificar que se ha generado texto
        self.assertIsNotNone(text)
        self.assertTrue(len(text) > 0)
        self.assertLessEqual(len(text), 100)  # Considerando que max_length es aproximado
    
    def test_generate_code(self):
        """Prueba la generación de código."""
        # Generar código con un modelo pequeño
        prompt = "Escribe una función en Python que calcule el factorial de un número."
        code = self.llm_manager.generate_code(prompt, language="python", model_size="small")
        
        # Verificar que se ha generado código
        self.assertIsNotNone(code)
        self.assertTrue(len(code) > 0)
        self.assertIn("def", code)  # Debería contener la definición de una función
    
    def test_get_embeddings(self):
        """Prueba la obtención de embeddings."""
        # Obtener embeddings para un texto
        text = "Inteligencia artificial"
        embeddings = self.llm_manager.get_embeddings(text)
        
        # Verificar que se han obtenido embeddings
        self.assertIsNotNone(embeddings)
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertTrue(embeddings.shape[0] > 0)  # Al menos una dimensión
    
    def test_calculate_similarity(self):
        """Prueba el cálculo de similitud entre textos."""
        # Calcular similitud entre textos relacionados
        text1 = "Inteligencia artificial"
        text2 = "Aprendizaje automático"
        similarity = self.llm_manager.calculate_similarity(text1, text2)
        
        # Verificar que se ha calculado la similitud
        self.assertIsNotNone(similarity)
        self.assertIsInstance(similarity, float)
        self.assertTrue(0 <= similarity <= 1)  # Similitud en rango [0, 1]


class TestTextGenerator(unittest.TestCase):
    """Pruebas para la clase TextGenerator."""
    
    def setUp(self):
        """Configuración para las pruebas."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Crear un LLMManager para el TextGenerator
        self.llm_manager = LLMManager(
            cache_dir=self.temp_dir,
            max_models=2
        )
        
        # Crear un grafo de conocimiento simulado
        class MockKnowledgeGraph:
            def search_by_similarity(self, query, top_k=5):
                return [("Concepto1", 0.9), ("Concepto2", 0.8)]
            
            def get_concept(self, concept_name):
                return {"descripción": f"Descripción de {concept_name}"}
            
            def extract_subgraph(self, concept, max_distance=2):
                G = nx.DiGraph()
                G.add_node(concept, descripción=f"Descripción de {concept}")
                G.add_node("Concepto relacionado", descripción="Descripción relacionada")
                G.add_edge(concept, "Concepto relacionado", tipo="relacionado con")
                return G
        
        self.mock_kg = MockKnowledgeGraph()
        
        # Crear el TextGenerator
        self.text_generator = TextGenerator(
            llm_manager=self.llm_manager,
            knowledge_graph=self.mock_kg
        )
    
    def tearDown(self):
        """Limpieza después de las pruebas."""
        shutil.rmtree(self.temp_dir)
        # Asegurar que se liberan los modelos
        self.llm_manager.unload_all_models()
    
    def test_initialization(self):
        """Prueba la inicialización correcta del generador de texto."""
        self.assertIsNotNone(self.text_generator.llm_manager)
        self.assertIsNotNone(self.text_generator.knowledge_graph)
    
    def test_generate_text(self):
        """Prueba la generación de texto."""
        # Generar texto
        prompt = "Escribe un párrafo sobre la inteligencia artificial."
        text = self.text_generator.generate_text(prompt, max_length=100)
        
        # Verificar que se ha generado texto
        self.assertIsNotNone(text)
        self.assertTrue(len(text) > 0)
    
    def test_continue_text(self):
        """Prueba la continuación de texto."""
        # Continuar texto
        text = "La inteligencia artificial es una rama de la informática que"
        continuation = self.text_generator.continue_text(text, max_length=50)
        
        # Verificar que se ha continuado el texto
        self.assertIsNotNone(continuation)
        self.assertTrue(len(continuation) > 0)
        self.assertTrue(continuation.startswith(text))
    
    def test_summarize_text(self):
        """Prueba la generación de resúmenes."""
        # Texto a resumir
        text = """
        La inteligencia artificial (IA) es la simulación de procesos de inteligencia humana por parte de máquinas, 
        especialmente sistemas informáticos. Estos procesos incluyen el aprendizaje (la adquisición de información 
        y reglas para el uso de la información), el razonamiento (usando las reglas para llegar a conclusiones 
        aproximadas o definitivas) y la autocorrección. Aplicaciones particulares de la IA incluyen sistemas expertos, 
        reconocimiento de voz y visión artificial.
        """
        
        # Generar resumen
        summary = self.text_generator.summarize_text(text, max_length=50)
        
        # Verificar que se ha generado un resumen
        self.assertIsNotNone(summary)
        self.assertTrue(len(summary) > 0)
        self.assertLess(len(summary), len(text))
    
    def test_explain_concept(self):
        """Prueba la explicación de conceptos."""
        # Explicar un concepto
        explanation = self.text_generator.explain_concept("Inteligencia Artificial", detail_level="basic")
        
        # Verificar que se ha generado una explicación
        self.assertIsNotNone(explanation)
        self.assertTrue(len(explanation) > 0)
    
    def test_generate_code(self):
        """Prueba la generación de código."""
        # Generar código
        task = "Crear una función que calcule el factorial de un número"
        code = self.text_generator.generate_code(task, language="python")
        
        # Verificar que se ha generado código
        self.assertIsNotNone(code)
        self.assertTrue(len(code) > 0)
        self.assertIn("def", code)  # Debería contener la definición de una función
    
    def test_refactor_code(self):
        """Prueba la refactorización de código."""
        # Código a refactorizar
        code = """
        def factorial(n):
            if n == 0:
                return 1
            else:
                return n * factorial(n-1)
        """
        
        # Refactorizar código
        refactored = self.text_generator.refactor_code(code, language="python")
        
        # Verificar que se ha refactorizado el código
        self.assertIsNotNone(refactored)
        self.assertTrue(len(refactored) > 0)
        self.assertIn("def", refactored)  # Debería contener la definición de una función


class TestQASystem(unittest.TestCase):
    """Pruebas para la clase QASystem."""
    
    def setUp(self):
        """Configuración para las pruebas."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Crear un LLMManager para el QASystem
        self.llm_manager = LLMManager(
            cache_dir=self.temp_dir,
            max_models=2
        )
        
        # Crear un grafo de conocimiento simulado
        class MockKnowledgeGraph:
            def search_by_similarity(self, query, top_k=5):
                return [("Concepto1", 0.9), ("Concepto2", 0.8)]
            
            def get_concept(self, concept_name):
                return {"descripción": f"Descripción de {concept_name}"}
            
            def find_path(self, source, target):
                return [source, "Concepto intermedio", target]
        
        self.mock_kg = MockKnowledgeGraph()
        
        # Crear una memoria holográfica simulada
        class MockHolographicMemory:
            def search_by_similarity(self, query_vector, top_k=5):
                return [
                    {"id": "item1", "similarity": 0.9, "metadata": {"texto": "Información relevante 1"}},
                    {"id": "item2", "similarity": 0.8, "metadata": {"texto": "Información relevante 2"}}
                ]
            
            def get_embeddings(self, text):
                return np.random.rand(128)
        
        self.mock_memory = MockHolographicMemory()
        
        # Crear el QASystem
        self.qa_system = QASystem(
            llm_manager=self.llm_manager,
            knowledge_graph=self.mock_kg,
            holographic_memory=self.mock_memory
        )
    
    def tearDown(self):
        """Limpieza después de las pruebas."""
        shutil.rmtree(self.temp_dir)
        # Asegurar que se liberan los modelos
        self.llm_manager.unload_all_models()
    
    def test_initialization(self):
        """Prueba la inicialización correcta del sistema de preguntas y respuestas."""
        self.assertIsNotNone(self.qa_system.llm_manager)
        self.assertIsNotNone(self.qa_system.knowledge_graph)
        self.assertIsNotNone(self.qa_system.holographic_memory)
    
    def test_answer_question(self):
        """Prueba la respuesta a preguntas."""
        # Hacer una pregunta
        question = "¿Qué es la inteligencia artificial?"
        answer = self.qa_system.answer_question(question)
        
        # Verificar que se ha generado una respuesta
        self.assertIsNotNone(answer)
        self.assertTrue(len(answer) > 0)
    
    def test_answer_with_sources(self):
        """Prueba la respuesta a preguntas con fuentes."""
        # Hacer una pregunta
        question = "¿Cuáles son las aplicaciones de la inteligencia artificial?"
        answer, sources = self.qa_system.answer_with_sources(question)
        
        # Verificar que se ha generado una respuesta con fuentes
        self.assertIsNotNone(answer)
        self.assertTrue(len(answer) > 0)
        self.assertIsNotNone(sources)
        self.assertTrue(len(sources) > 0)
    
    def test_multi_step_reasoning(self):
        """Prueba el razonamiento multi-paso."""
        # Hacer una pregunta compleja
        question = "¿Cómo se relaciona la inteligencia artificial con el aprendizaje profundo?"
        answer, steps = self.qa_system.multi_step_reasoning(question)
        
        # Verificar que se ha generado una respuesta con pasos de razonamiento
        self.assertIsNotNone(answer)
        self.assertTrue(len(answer) > 0)
        self.assertIsNotNone(steps)
        self.assertTrue(len(steps) > 0)
    
    def test_generate_questions(self):
        """Prueba la generación de preguntas."""
        # Generar preguntas sobre un tema
        topic = "Inteligencia Artificial"
        questions = self.qa_system.generate_questions(topic, count=3)
        
        # Verificar que se han generado preguntas
        self.assertIsNotNone(questions)
        self.assertTrue(len(questions) > 0)
        self.assertLessEqual(len(questions), 3)
    
    def test_evaluate_answer(self):
        """Prueba la evaluación de respuestas."""
        # Evaluar una respuesta
        question = "¿Qué es el aprendizaje profundo?"
        answer = "El aprendizaje profundo es una técnica de inteligencia artificial basada en redes neuronales."
        evaluation = self.qa_system.evaluate_answer(question, answer)
        
        # Verificar que se ha generado una evaluación
        self.assertIsNotNone(evaluation)
        self.assertIsInstance(evaluation, dict)
        self.assertIn("score", evaluation)
        self.assertTrue(0 <= evaluation["score"] <= 1)


if __name__ == '__main__':
    unittest.main()
