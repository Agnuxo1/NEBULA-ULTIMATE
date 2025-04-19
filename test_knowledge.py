"""
Módulo de pruebas para los componentes de conocimiento de NEBULA.

Este módulo contiene pruebas unitarias para verificar el funcionamiento
correcto de los componentes de gestión de conocimiento de NEBULA.
"""

import unittest
import sys
import os
import numpy as np
import torch
import networkx as nx
from pathlib import Path
import tempfile
import shutil

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge.knowledge_graph import EnhancedKnowledgeGraph
from knowledge.holographic_memory import HolographicMemory
from knowledge.optical_processor import OpticalProcessingUnit
from utils.config import PARAMETERS

class TestKnowledgeGraph(unittest.TestCase):
    """Pruebas para la clase EnhancedKnowledgeGraph."""
    
    def setUp(self):
        """Configuración para las pruebas."""
        self.temp_dir = tempfile.mkdtemp()
        self.graph = EnhancedKnowledgeGraph(storage_path=self.temp_dir)
        
        # Añadir algunos nodos y relaciones para las pruebas
        self.graph.add_concept("Inteligencia Artificial", {"tipo": "campo", "importancia": 0.9})
        self.graph.add_concept("Aprendizaje Profundo", {"tipo": "técnica", "importancia": 0.8})
        self.graph.add_concept("Redes Neuronales", {"tipo": "modelo", "importancia": 0.7})
        self.graph.add_concept("Transformers", {"tipo": "arquitectura", "importancia": 0.8})
        
        self.graph.add_relation("Inteligencia Artificial", "Aprendizaje Profundo", "incluye", 0.9)
        self.graph.add_relation("Aprendizaje Profundo", "Redes Neuronales", "utiliza", 0.8)
        self.graph.add_relation("Redes Neuronales", "Transformers", "evolucionó a", 0.7)
    
    def tearDown(self):
        """Limpieza después de las pruebas."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Prueba la inicialización correcta del grafo de conocimiento."""
        self.assertIsInstance(self.graph.graph, nx.DiGraph)
        self.assertEqual(len(self.graph.graph.nodes), 4)
        self.assertEqual(len(self.graph.graph.edges), 3)
    
    def test_add_concept(self):
        """Prueba la adición de conceptos al grafo."""
        # Añadir un nuevo concepto
        self.graph.add_concept("GPT", {"tipo": "modelo", "importancia": 0.9})
        
        # Verificar que el concepto se ha añadido
        self.assertIn("GPT", self.graph.graph.nodes)
        self.assertEqual(self.graph.graph.nodes["GPT"]["tipo"], "modelo")
        self.assertEqual(self.graph.graph.nodes["GPT"]["importancia"], 0.9)
    
    def test_add_relation(self):
        """Prueba la adición de relaciones al grafo."""
        # Añadir una nueva relación
        self.graph.add_relation("Transformers", "GPT", "base de", 0.9)
        
        # Verificar que la relación se ha añadido
        self.assertTrue(self.graph.graph.has_edge("Transformers", "GPT"))
        self.assertEqual(self.graph.graph.edges["Transformers", "GPT"]["tipo"], "base de")
        self.assertEqual(self.graph.graph.edges["Transformers", "GPT"]["peso"], 0.9)
    
    def test_get_concept(self):
        """Prueba la obtención de conceptos del grafo."""
        # Obtener un concepto existente
        concept = self.graph.get_concept("Redes Neuronales")
        
        # Verificar que se ha obtenido el concepto correcto
        self.assertIsNotNone(concept)
        self.assertEqual(concept["tipo"], "modelo")
        self.assertEqual(concept["importancia"], 0.7)
        
        # Intentar obtener un concepto inexistente
        nonexistent = self.graph.get_concept("Concepto Inexistente")
        self.assertIsNone(nonexistent)
    
    def test_get_related_concepts(self):
        """Prueba la obtención de conceptos relacionados."""
        # Obtener conceptos relacionados con Inteligencia Artificial
        related = self.graph.get_related_concepts("Inteligencia Artificial")
        
        # Verificar que se han obtenido los conceptos correctos
        self.assertEqual(len(related), 1)
        self.assertIn("Aprendizaje Profundo", [r[0] for r in related])
        
        # Obtener conceptos relacionados con Aprendizaje Profundo
        related = self.graph.get_related_concepts("Aprendizaje Profundo")
        
        # Verificar que se han obtenido los conceptos correctos
        self.assertEqual(len(related), 1)
        self.assertIn("Redes Neuronales", [r[0] for r in related])
    
    def test_find_path(self):
        """Prueba la búsqueda de caminos entre conceptos."""
        # Buscar camino entre Inteligencia Artificial y Transformers
        path = self.graph.find_path("Inteligencia Artificial", "Transformers")
        
        # Verificar que se ha encontrado un camino
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 4)  # IA -> AP -> RN -> Transformers
        self.assertEqual(path[0], "Inteligencia Artificial")
        self.assertEqual(path[-1], "Transformers")
    
    def test_search_by_similarity(self):
        """Prueba la búsqueda por similitud semántica."""
        # Buscar conceptos similares a "aprendizaje"
        results = self.graph.search_by_similarity("aprendizaje", top_k=2)
        
        # Verificar que se han encontrado resultados
        self.assertTrue(len(results) > 0)
        self.assertLessEqual(len(results), 2)
    
    def test_extract_subgraph(self):
        """Prueba la extracción de subgrafos."""
        # Extraer subgrafo centrado en Aprendizaje Profundo
        subgraph = self.graph.extract_subgraph("Aprendizaje Profundo", max_distance=1)
        
        # Verificar que el subgrafo contiene los nodos correctos
        self.assertIn("Aprendizaje Profundo", subgraph.nodes)
        self.assertIn("Inteligencia Artificial", subgraph.nodes)
        self.assertIn("Redes Neuronales", subgraph.nodes)
        self.assertNotIn("Transformers", subgraph.nodes)  # Distancia > 1
    
    def test_save_and_load(self):
        """Prueba el guardado y carga del grafo."""
        # Guardar grafo
        save_path = os.path.join(self.temp_dir, "test_graph.pkl")
        self.graph.save(save_path)
        
        # Verificar que el archivo se ha creado
        self.assertTrue(os.path.exists(save_path))
        
        # Crear un nuevo grafo y cargarlo
        new_graph = EnhancedKnowledgeGraph(storage_path=self.temp_dir)
        new_graph.load(save_path)
        
        # Verificar que el grafo cargado tiene los mismos nodos y relaciones
        self.assertEqual(len(new_graph.graph.nodes), len(self.graph.graph.nodes))
        self.assertEqual(len(new_graph.graph.edges), len(self.graph.graph.edges))
        self.assertIn("Inteligencia Artificial", new_graph.graph.nodes)
        self.assertIn("Aprendizaje Profundo", new_graph.graph.nodes)
        self.assertTrue(new_graph.graph.has_edge("Inteligencia Artificial", "Aprendizaje Profundo"))


class TestHolographicMemory(unittest.TestCase):
    """Pruebas para la clase HolographicMemory."""
    
    def setUp(self):
        """Configuración para las pruebas."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory = HolographicMemory(
            vector_dimension=128,
            storage_path=self.temp_dir
        )
        
        # Añadir algunos vectores para las pruebas
        self.memory.store("concepto1", np.random.rand(128), {"tipo": "concepto"})
        self.memory.store("concepto2", np.random.rand(128), {"tipo": "concepto"})
        self.memory.store("dato1", np.random.rand(128), {"tipo": "dato"})
        self.memory.store("dato2", np.random.rand(128), {"tipo": "dato"})
    
    def tearDown(self):
        """Limpieza después de las pruebas."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Prueba la inicialización correcta de la memoria holográfica."""
        self.assertEqual(self.memory.vector_dimension, 128)
        self.assertEqual(len(self.memory.memory_spaces), 1)  # Espacio por defecto
        self.assertEqual(len(self.memory.memory_spaces["default"]), 4)
    
    def test_store_and_retrieve(self):
        """Prueba el almacenamiento y recuperación de vectores."""
        # Almacenar un nuevo vector
        vector = np.random.rand(128)
        self.memory.store("nuevo", vector, {"tipo": "nuevo"})
        
        # Verificar que el vector se ha almacenado
        self.assertIn("nuevo", self.memory.memory_spaces["default"])
        
        # Recuperar el vector
        retrieved = self.memory.retrieve("nuevo")
        
        # Verificar que se ha recuperado el vector correcto
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["metadata"]["tipo"], "nuevo")
        np.testing.assert_array_equal(retrieved["vector"], vector)
    
    def test_search_by_similarity(self):
        """Prueba la búsqueda por similitud."""
        # Crear un vector de consulta
        query_vector = np.random.rand(128)
        
        # Buscar vectores similares
        results = self.memory.search_by_similarity(query_vector, top_k=2)
        
        # Verificar que se han encontrado resultados
        self.assertEqual(len(results), 2)
        
        # Verificar que los resultados están ordenados por similitud
        self.assertTrue(results[0]["similarity"] >= results[1]["similarity"])
    
    def test_create_memory_space(self):
        """Prueba la creación de espacios de memoria."""
        # Crear un nuevo espacio de memoria
        self.memory.create_memory_space("nuevo_espacio")
        
        # Verificar que el espacio se ha creado
        self.assertIn("nuevo_espacio", self.memory.memory_spaces)
        self.assertEqual(len(self.memory.memory_spaces["nuevo_espacio"]), 0)
        
        # Almacenar un vector en el nuevo espacio
        vector = np.random.rand(128)
        self.memory.store("item_nuevo", vector, {"tipo": "item"}, space="nuevo_espacio")
        
        # Verificar que el vector se ha almacenado en el espacio correcto
        self.assertIn("item_nuevo", self.memory.memory_spaces["nuevo_espacio"])
        self.assertNotIn("item_nuevo", self.memory.memory_spaces["default"])
    
    def test_consolidate_memory(self):
        """Prueba la consolidación de memoria."""
        # Añadir vectores similares
        base_vector = np.random.rand(128)
        
        # Añadir variaciones del vector base
        for i in range(5):
            # Crear una pequeña variación
            variation = base_vector + np.random.normal(0, 0.1, 128)
            variation = variation / np.linalg.norm(variation)  # Normalizar
            self.memory.store(f"similar{i}", variation, {"tipo": "similar"})
        
        # Contar elementos antes de consolidar
        count_before = len(self.memory.memory_spaces["default"])
        
        # Consolidar memoria
        self.memory.consolidate_memory(similarity_threshold=0.8)
        
        # Verificar que se han consolidado elementos
        count_after = len(self.memory.memory_spaces["default"])
        self.assertLess(count_after, count_before)
    
    def test_save_and_load(self):
        """Prueba el guardado y carga de la memoria."""
        # Guardar memoria
        save_path = os.path.join(self.temp_dir, "test_memory.pkl")
        self.memory.save(save_path)
        
        # Verificar que el archivo se ha creado
        self.assertTrue(os.path.exists(save_path))
        
        # Crear una nueva memoria y cargarla
        new_memory = HolographicMemory(vector_dimension=128, storage_path=self.temp_dir)
        new_memory.load(save_path)
        
        # Verificar que la memoria cargada tiene los mismos elementos
        self.assertEqual(len(new_memory.memory_spaces), len(self.memory.memory_spaces))
        self.assertEqual(len(new_memory.memory_spaces["default"]), len(self.memory.memory_spaces["default"]))
        self.assertIn("concepto1", new_memory.memory_spaces["default"])
        self.assertIn("dato1", new_memory.memory_spaces["default"])


class TestOpticalProcessingUnit(unittest.TestCase):
    """Pruebas para la clase OpticalProcessingUnit."""
    
    def setUp(self):
        """Configuración para las pruebas."""
        self.opu = OpticalProcessingUnit(dimensions=(32, 32))
    
    def test_initialization(self):
        """Prueba la inicialización correcta de la unidad de procesamiento óptico."""
        self.assertEqual(self.opu.dimensions, (32, 32))
        self.assertIsNotNone(self.opu.filters)
    
    def test_fourier_transform(self):
        """Prueba la transformada de Fourier."""
        # Crear una imagen de prueba
        image = np.random.rand(32, 32)
        
        # Aplicar transformada de Fourier
        transformed = self.opu.fourier_transform(image)
        
        # Verificar que la transformada tiene las dimensiones correctas
        self.assertEqual(transformed.shape, (32, 32))
        
        # Aplicar transformada inversa
        inverse = self.opu.inverse_fourier_transform(transformed)
        
        # Verificar que la transformada inversa recupera la imagen original
        np.testing.assert_array_almost_equal(inverse.real, image, decimal=10)
    
    def test_convolve(self):
        """Prueba la convolución."""
        # Crear una imagen y un kernel de prueba
        image = np.random.rand(32, 32)
        kernel = np.random.rand(5, 5)
        
        # Aplicar convolución
        result = self.opu.convolve(image, kernel)
        
        # Verificar que el resultado tiene las dimensiones correctas
        self.assertEqual(result.shape, (32, 32))
    
    def test_correlate(self):
        """Prueba la correlación."""
        # Crear dos imágenes de prueba
        image1 = np.random.rand(32, 32)
        image2 = np.random.rand(32, 32)
        
        # Aplicar correlación
        result = self.opu.correlate(image1, image2)
        
        # Verificar que el resultado tiene las dimensiones correctas
        self.assertEqual(result.shape, (32, 32))
    
    def test_apply_filter(self):
        """Prueba la aplicación de filtros."""
        # Crear una imagen de prueba
        image = np.random.rand(32, 32)
        
        # Aplicar filtro de paso bajo
        filtered = self.opu.apply_filter(image, "lowpass")
        
        # Verificar que el resultado tiene las dimensiones correctas
        self.assertEqual(filtered.shape, (32, 32))
        
        # Aplicar filtro de paso alto
        filtered = self.opu.apply_filter(image, "highpass")
        
        # Verificar que el resultado tiene las dimensiones correctas
        self.assertEqual(filtered.shape, (32, 32))
    
    def test_holographic_storage(self):
        """Prueba el almacenamiento holográfico."""
        # Crear imágenes de prueba
        image1 = np.random.rand(32, 32)
        image2 = np.random.rand(32, 32)
        
        # Almacenar holográficamente
        hologram = self.opu.holographic_store([image1, image2])
        
        # Verificar que el holograma tiene las dimensiones correctas
        self.assertEqual(hologram.shape, (32, 32))
        
        # Recuperar imágenes
        recovered1 = self.opu.holographic_retrieve(hologram, image1)
        recovered2 = self.opu.holographic_retrieve(hologram, image2)
        
        # Verificar que las imágenes recuperadas tienen las dimensiones correctas
        self.assertEqual(recovered1.shape, (32, 32))
        self.assertEqual(recovered2.shape, (32, 32))


if __name__ == '__main__':
    unittest.main()
