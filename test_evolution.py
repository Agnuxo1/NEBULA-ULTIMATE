"""
Módulo de pruebas para los componentes de evolución y automejora de NEBULA.

Este módulo contiene pruebas unitarias para verificar el funcionamiento
correcto de los componentes de evolución y automejora de NEBULA.
"""

import unittest
import sys
import os
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
import json

# Añadir directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evolution.evolution_engine import EvolutionEngine
from evolution.code_analyzer import CodeAnalyzer
from evolution.self_optimizer import SelfOptimizer
from utils.config import PARAMETERS

class TestEvolutionEngine(unittest.TestCase):
    """Pruebas para la clase EvolutionEngine."""
    
    def setUp(self):
        """Configuración para las pruebas."""
        self.temp_dir = tempfile.mkdtemp()
        self.engine = EvolutionEngine(
            checkpoint_dir=self.temp_dir
        )
    
    def tearDown(self):
        """Limpieza después de las pruebas."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Prueba la inicialización correcta del motor de evolución."""
        self.assertIsNotNone(self.engine.toolbox)
        self.assertEqual(self.engine.checkpoint_dir, self.temp_dir)
    
    def test_evolve_real_parameters(self):
        """Prueba la evolución de parámetros reales."""
        # Definir función de evaluación
        def eval_func(params):
            # Función simple: minimizar la suma de cuadrados
            return -sum(p**2 for p in params)
        
        # Configuración de parámetros
        param_config = [
            {"name": "param1", "min": -5.0, "max": 5.0},
            {"name": "param2", "min": -5.0, "max": 5.0}
        ]
        
        # Ejecutar evolución
        results = self.engine.evolve_parameters(
            param_type="real",
            param_config=param_config,
            eval_function=eval_func,
            pop_size=10,
            n_generations=5
        )
        
        # Verificar resultados
        self.assertIsNotNone(results)
        self.assertIn("best_params", results)
        self.assertIn("best_fitness", results)
        self.assertEqual(len(results["best_params"]), 2)
        
        # Los mejores parámetros deberían estar cerca de cero
        self.assertLess(abs(results["best_params"][0]), 1.0)
        self.assertLess(abs(results["best_params"][1]), 1.0)
    
    def test_evolve_integer_parameters(self):
        """Prueba la evolución de parámetros enteros."""
        # Definir función de evaluación
        def eval_func(params):
            # Función simple: minimizar la suma de cuadrados
            return -sum(p**2 for p in params)
        
        # Configuración de parámetros
        param_config = [
            {"name": "param1", "min": -5, "max": 5},
            {"name": "param2", "min": -5, "max": 5}
        ]
        
        # Ejecutar evolución
        results = self.engine.evolve_parameters(
            param_type="integer",
            param_config=param_config,
            eval_function=eval_func,
            pop_size=10,
            n_generations=5
        )
        
        # Verificar resultados
        self.assertIsNotNone(results)
        self.assertIn("best_params", results)
        self.assertIn("best_fitness", results)
        self.assertEqual(len(results["best_params"]), 2)
        
        # Los parámetros deben ser enteros
        self.assertTrue(isinstance(results["best_params"][0], int))
        self.assertTrue(isinstance(results["best_params"][1], int))
    
    def test_evolve_categorical_parameters(self):
        """Prueba la evolución de parámetros categóricos."""
        # Definir función de evaluación
        def eval_func(params):
            # Función simple: preferir ciertos valores
            score = 0
            if params[0] == "option2":
                score += 1
            if params[1] == "choice3":
                score += 1
            return score
        
        # Configuración de parámetros
        param_config = [
            {"name": "param1", "options": ["option1", "option2", "option3"]},
            {"name": "param2", "options": ["choice1", "choice2", "choice3"]}
        ]
        
        # Ejecutar evolución
        results = self.engine.evolve_parameters(
            param_type="categorical",
            param_config=param_config,
            eval_function=eval_func,
            pop_size=10,
            n_generations=5
        )
        
        # Verificar resultados
        self.assertIsNotNone(results)
        self.assertIn("best_params", results)
        self.assertIn("best_fitness", results)
        self.assertEqual(len(results["best_params"]), 2)
        
        # Los parámetros deben ser de las opciones disponibles
        self.assertIn(results["best_params"][0], ["option1", "option2", "option3"])
        self.assertIn(results["best_params"][1], ["choice1", "choice2", "choice3"])
    
    def test_evolve_mixed_parameters(self):
        """Prueba la evolución de parámetros mixtos."""
        # Definir función de evaluación
        def eval_func(params):
            # Función simple: combinación de criterios
            score = -params[0]**2  # Minimizar el cuadrado del parámetro real
            if params[1] == 3:  # Preferir el valor 3 para el parámetro entero
                score += 1
            if params[2] == "option2":  # Preferir "option2" para el parámetro categórico
                score += 1
            return score
        
        # Configuración de parámetros
        param_config = [
            {"name": "real_param", "type": "real", "min": -5.0, "max": 5.0},
            {"name": "int_param", "type": "integer", "min": 1, "max": 5},
            {"name": "cat_param", "type": "categorical", "options": ["option1", "option2", "option3"]}
        ]
        
        # Ejecutar evolución
        results = self.engine.evolve_parameters(
            param_type="mixed",
            param_config=param_config,
            eval_function=eval_func,
            pop_size=10,
            n_generations=5
        )
        
        # Verificar resultados
        self.assertIsNotNone(results)
        self.assertIn("best_params", results)
        self.assertIn("best_fitness", results)
        self.assertEqual(len(results["best_params"]), 3)
        
        # Verificar tipos de parámetros
        self.assertTrue(isinstance(results["best_params"][0], float))
        self.assertTrue(isinstance(results["best_params"][1], int))
        self.assertTrue(isinstance(results["best_params"][2], str))
    
    def test_checkpoint_and_resume(self):
        """Prueba el guardado y carga de checkpoints."""
        # Definir función de evaluación
        def eval_func(params):
            return -sum(p**2 for p in params)
        
        # Configuración de parámetros
        param_config = [
            {"name": "param1", "min": -5.0, "max": 5.0},
            {"name": "param2", "min": -5.0, "max": 5.0}
        ]
        
        # Ejecutar evolución con checkpoint
        self.engine.evolve_parameters(
            param_type="real",
            param_config=param_config,
            eval_function=eval_func,
            pop_size=10,
            n_generations=3,
            checkpoint_freq=1
        )
        
        # Verificar que se ha creado un archivo de checkpoint
        checkpoint_files = list(Path(self.temp_dir).glob("checkpoint_*.pkl"))
        self.assertTrue(len(checkpoint_files) > 0)
        
        # Crear un nuevo motor y continuar desde el checkpoint
        new_engine = EvolutionEngine(checkpoint_dir=self.temp_dir)
        results = new_engine.resume_from_checkpoint(
            checkpoint_path=str(checkpoint_files[0]),
            eval_function=eval_func,
            n_generations=2
        )
        
        # Verificar resultados
        self.assertIsNotNone(results)
        self.assertIn("best_params", results)
        self.assertIn("best_fitness", results)
        self.assertEqual(len(results["best_params"]), 2)


class TestCodeAnalyzer(unittest.TestCase):
    """Pruebas para la clase CodeAnalyzer."""
    
    def setUp(self):
        """Configuración para las pruebas."""
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = CodeAnalyzer()
        
        # Crear un archivo de código de prueba
        self.test_file = Path(self.temp_dir) / "test_code.py"
        with open(self.test_file, 'w') as f:
            f.write("""
# Archivo de prueba para CodeAnalyzer

def factorial(n):
    \"\"\"
    Calcula el factorial de un número.
    
    Args:
        n: Número entero positivo
        
    Returns:
        El factorial de n
    \"\"\"
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

class TestClass:
    \"\"\"Clase de prueba con algunos métodos.\"\"\"
    
    def __init__(self, value):
        self.value = value
    
    def double(self):
        # Duplica el valor
        return self.value * 2
    
    def complex_method(self, a, b, c, d, e):
        # Método con demasiados parámetros
        result = 0
        for i in range(a):
            for j in range(b):
                # Bucles anidados
                result += i * j
        return result
""")
    
    def tearDown(self):
        """Limpieza después de las pruebas."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Prueba la inicialización correcta del analizador de código."""
        self.assertIsNotNone(self.analyzer.code_smells)
        self.assertIsNotNone(self.analyzer.good_practices)
    
    def test_analyze_file(self):
        """Prueba el análisis de un archivo."""
        # Analizar el archivo de prueba
        analysis = self.analyzer.analyze_file(self.test_file)
        
        # Verificar que el análisis se ha completado correctamente
        self.assertEqual(analysis['status'], 'success')
        self.assertIn('line_counts', analysis)
        self.assertIn('classes', analysis)
        self.assertIn('functions', analysis)
        self.assertIn('code_smells', analysis)
        self.assertIn('quality_score', analysis)
        
        # Verificar conteo de líneas
        self.assertTrue(analysis['line_counts']['total'] > 0)
        
        # Verificar detección de clases
        self.assertEqual(len(analysis['classes']), 1)
        self.assertEqual(analysis['classes'][0]['name'], 'TestClass')
        
        # Verificar detección de funciones
        self.assertEqual(len(analysis['functions']), 1)
        self.assertEqual(analysis['functions'][0]['name'], 'factorial')
        
        # Verificar detección de problemas
        self.assertTrue(len(analysis['code_smells']) > 0)
        
        # Verificar puntuación de calidad
        self.assertTrue(0 <= analysis['quality_score'] <= 10)
    
    def test_generate_improvement_suggestions(self):
        """Prueba la generación de sugerencias de mejora."""
        # Analizar el archivo de prueba
        analysis = self.analyzer.analyze_file(self.test_file)
        
        # Generar sugerencias
        suggestions = self.analyzer.generate_improvement_suggestions(analysis)
        
        # Verificar que se han generado sugerencias
        self.assertEqual(suggestions['status'], 'success')
        self.assertIn('general_suggestions', suggestions)
        self.assertIn('specific_suggestions', suggestions)
        
        # Debería haber al menos una sugerencia específica
        self.assertTrue(len(suggestions['specific_suggestions']) > 0)


class TestSelfOptimizer(unittest.TestCase):
    """Pruebas para la clase SelfOptimizer."""
    
    def setUp(self):
        """Configuración para las pruebas."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Crear un EvolutionEngine para el SelfOptimizer
        self.evolution_engine = EvolutionEngine(
            checkpoint_dir=self.temp_dir
        )
        
        # Crear un CodeAnalyzer para el SelfOptimizer
        self.code_analyzer = CodeAnalyzer()
        
        # Crear un LLMManager simulado
        class MockLLMManager:
            def generate_text(self, prompt, max_length=500, model_size="small"):
                return "Texto generado para pruebas"
            
            def generate_code(self, prompt, language="python", model_size="small"):
                return "def test_function():\n    return 'Código generado para pruebas'"
        
        self.mock_llm = MockLLMManager()
        
        # Crear el SelfOptimizer
        self.optimizer = SelfOptimizer(
            evolution_engine=self.evolution_engine,
            code_analyzer=self.code_analyzer,
            llm_manager=self.mock_llm
        )
        
        # Crear un archivo de código de prueba
        self.test_file = Path(self.temp_dir) / "test_code.py"
        with open(self.test_file, 'w') as f:
            f.write("""
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

class TestClass:
    def __init__(self, value):
        self.value = value
    
    def double(self):
        return self.value * 2
""")
        
        # Crear un archivo de configuración de prueba
        self.config_file = Path(self.temp_dir) / "config.py"
        with open(self.config_file, 'w') as f:
            f.write("""
PARAMETERS = {
    "PARAM1": 10,
    "PARAM2": "valor",
    "PARAM3": True,
    "PARAM4": [1, 2, 3],
    "PARAM5": {"key": "value"}
}
""")
    
    def tearDown(self):
        """Limpieza después de las pruebas."""
        shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Prueba la inicialización correcta del optimizador automático."""
        self.assertIsNotNone(self.optimizer.evolution_engine)
        self.assertIsNotNone(self.optimizer.code_analyzer)
        self.assertIsNotNone(self.optimizer.llm_manager)
        self.assertIsNotNone(self.optimizer.safety_checks)
    
    def test_create_backup(self):
        """Prueba la creación de backups."""
        # Crear backup de un archivo
        backup_dir = self.optimizer._create_backup(self.test_file)
        
        # Verificar que se ha creado el backup
        self.assertTrue(backup_dir.exists())
        self.assertTrue((backup_dir / self.test_file.name).exists())
        
        # Verificar que se ha creado el archivo de metadatos
        self.assertTrue((backup_dir / "backup_metadata.json").exists())
    
    def test_validate_code_changes(self):
        """Prueba la validación de cambios de código."""
        # Código original
        original_code = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""
        
        # Código modificado válido
        valid_code = """
def factorial(n):
    \"\"\"Calcula el factorial de un número.\"\"\"
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""
        
        # Código modificado inválido (elimina la función)
        invalid_code = """
# La función ha sido eliminada
"""
        
        # Validar cambios válidos
        validation = self.optimizer._validate_code_changes(original_code, valid_code)
        self.assertTrue(validation["valid"])
        
        # Validar cambios inválidos
        validation = self.optimizer._validate_code_changes(original_code, invalid_code)
        self.assertFalse(validation["valid"])
    
    def test_update_configuration(self):
        """Prueba la actualización de configuración."""
        # Permitir cambios de configuración
        self.optimizer.safety_checks["allow_config_changes"] = True
        
        # Actualizaciones de configuración
        updates = {
            "PARAM1": 20,
            "PARAM2": "nuevo_valor"
        }
        
        # Actualizar configuración
        result = self.optimizer.update_configuration(updates)
        
        # Verificar que la actualización se ha completado correctamente
        self.assertEqual(result["sta
(Content truncated due to size limit. Use line ranges to read in chunks)