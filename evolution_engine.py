"""
Implementación del Motor de Evolución para NEBULA.

Esta clase proporciona mecanismos para la evolución de componentes del sistema
mediante algoritmos evolutivos y optimización.
"""

import logging
import time
import os
import random
import copy
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path

import numpy as np
from deap import base, creator, tools, algorithms

from utils.config import PARAMETERS
from utils.helpers import safe_loop

logger = logging.getLogger("NEBULA.EvolutionEngine")

class EvolutionEngine:
    """
    Motor de evolución que permite optimizar componentes del sistema
    mediante algoritmos evolutivos.
    
    Características:
    - Evolución de parámetros de configuración
    - Optimización de arquitecturas neuronales
    - Selección de algoritmos y estrategias
    - Evaluación de rendimiento de variantes
    """
    
    def __init__(self, evaluation_function: Optional[Callable] = None):
        """
        Inicializa el motor de evolución.
        
        Args:
            evaluation_function: Función para evaluar individuos.
                                Si es None, debe proporcionarse en cada ejecución.
        """
        logger.info("Inicializando EvolutionEngine...")
        
        # Función de evaluación
        self.evaluation_function = evaluation_function
        
        # Historial de evolución
        self.evolution_history = []
        
        # Contadores y estadísticas
        self.generation_count = 0
        self.total_evaluations = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Configuración de algoritmos evolutivos
        self._setup_deap()
        
        # Directorio para guardar resultados
        self.results_dir = PARAMETERS["BACKUP_DIRECTORY"] / "evolution"
        os.makedirs(self.results_dir, exist_ok=True)
        
        logger.info("EvolutionEngine inicializado correctamente.")
    
    def _setup_deap(self):
        """Configura el framework DEAP para algoritmos evolutivos."""
        try:
            # Crear tipos para fitness y individuos
            if not hasattr(creator, "FitnessMax"):
                creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            
            if not hasattr(creator, "Individual"):
                creator.create("Individual", list, fitness=creator.FitnessMax)
            
            # Crear toolbox
            self.toolbox = base.Toolbox()
            
            # Nota: Los operadores específicos se configurarán en cada ejecución
            # ya que dependen del tipo de problema
            
            logger.debug("Framework DEAP configurado correctamente.")
        except Exception as e:
            logger.error(f"Error al configurar DEAP: {e}")
    
    def _configure_for_real_values(self, param_ranges: List[Tuple[float, float]]):
        """
        Configura el toolbox para optimización de valores reales.
        
        Args:
            param_ranges: Lista de tuplas (min, max) para cada parámetro.
        """
        # Registrar funciones para crear individuos
        self.toolbox.register("attr_float", random.uniform, 0.0, 1.0)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             self.toolbox.attr_float, n=len(param_ranges))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Registrar operadores genéticos
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Guardar rangos para mapeo
        self.param_ranges = param_ranges
    
    def _configure_for_integer_values(self, param_ranges: List[Tuple[int, int]]):
        """
        Configura el toolbox para optimización de valores enteros.
        
        Args:
            param_ranges: Lista de tuplas (min, max) para cada parámetro.
        """
        # Funciones para crear atributos aleatorios dentro de rangos
        def random_int(min_val, max_val):
            return random.randint(min_val, max_val)
        
        # Registrar funciones para crear individuos
        attr_ints = []
        for i, (min_val, max_val) in enumerate(param_ranges):
            self.toolbox.register(f"attr_int_{i}", random_int, min_val, max_val)
            attr_ints.append(getattr(self.toolbox, f"attr_int_{i}"))
        
        self.toolbox.register("individual", tools.initCycle, creator.Individual, 
                             tuple(attr_ints), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Registrar operadores genéticos
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        
        # Mutación personalizada para enteros
        def mutate_int(individual, indpb):
            for i, (min_val, max_val) in enumerate(param_ranges):
                if random.random() < indpb:
                    # Mutación por desplazamiento o valor aleatorio
                    if random.random() < 0.5:
                        # Desplazamiento
                        delta = random.randint(-3, 3)
                        individual[i] = max(min_val, min(max_val, individual[i] + delta))
                    else:
                        # Valor aleatorio
                        individual[i] = random.randint(min_val, max_val)
            return individual,
        
        self.toolbox.register("mutate", mutate_int, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Guardar rangos
        self.param_ranges = param_ranges
    
    def _configure_for_categorical_values(self, categories: List[List[Any]]):
        """
        Configura el toolbox para optimización de valores categóricos.
        
        Args:
            categories: Lista de listas con opciones para cada parámetro.
        """
        # Funciones para crear atributos aleatorios dentro de categorías
        def random_category(options):
            return random.choice(options)
        
        # Registrar funciones para crear individuos
        attr_cats = []
        for i, options in enumerate(categories):
            self.toolbox.register(f"attr_cat_{i}", random_category, options)
            attr_cats.append(getattr(self.toolbox, f"attr_cat_{i}"))
        
        self.toolbox.register("individual", tools.initCycle, creator.Individual, 
                             tuple(attr_cats), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Registrar operadores genéticos
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        
        # Mutación personalizada para categorías
        def mutate_categorical(individual, indpb):
            for i, options in enumerate(categories):
                if random.random() < indpb:
                    # Seleccionar una categoría diferente
                    current = individual[i]
                    remaining = [opt for opt in options if opt != current]
                    if remaining:
                        individual[i] = random.choice(remaining)
            return individual,
        
        self.toolbox.register("mutate", mutate_categorical, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Guardar categorías
        self.categories = categories
    
    def _configure_for_mixed_values(self, param_specs: List[Dict[str, Any]]):
        """
        Configura el toolbox para optimización de valores mixtos (reales, enteros, categóricos).
        
        Args:
            param_specs: Lista de diccionarios con especificaciones para cada parámetro.
                        Cada diccionario debe tener:
                        - 'type': 'real', 'int' o 'categorical'
                        - 'range': (min, max) para tipos real e int
                        - 'options': lista de opciones para tipo categorical
        """
        # Funciones para crear atributos aleatorios según tipo
        def random_value(spec):
            if spec['type'] == 'real':
                return random.uniform(spec['range'][0], spec['range'][1])
            elif spec['type'] == 'int':
                return random.randint(spec['range'][0], spec['range'][1])
            elif spec['type'] == 'categorical':
                return random.choice(spec['options'])
            else:
                raise ValueError(f"Tipo de parámetro no soportado: {spec['type']}")
        
        # Registrar funciones para crear individuos
        attr_funcs = []
        for i, spec in enumerate(param_specs):
            self.toolbox.register(f"attr_{i}", random_value, spec)
            attr_funcs.append(getattr(self.toolbox, f"attr_{i}"))
        
        self.toolbox.register("individual", tools.initCycle, creator.Individual, 
                             tuple(attr_funcs), n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Registrar operadores genéticos
        self.toolbox.register("mate", tools.cxUniform, indpb=0.5)
        
        # Mutación personalizada para valores mixtos
        def mutate_mixed(individual, indpb):
            for i, spec in enumerate(param_specs):
                if random.random() < indpb:
                    if spec['type'] == 'real':
                        # Mutación gaussiana para reales
                        sigma = (spec['range'][1] - spec['range'][0]) * 0.1
                        individual[i] += random.gauss(0, sigma)
                        individual[i] = max(spec['range'][0], min(spec['range'][1], individual[i]))
                    elif spec['type'] == 'int':
                        # Mutación por desplazamiento o valor aleatorio para enteros
                        if random.random() < 0.5:
                            delta = random.randint(-3, 3)
                            individual[i] = max(spec['range'][0], min(spec['range'][1], individual[i] + delta))
                        else:
                            individual[i] = random.randint(spec['range'][0], spec['range'][1])
                    elif spec['type'] == 'categorical':
                        # Seleccionar una categoría diferente
                        current = individual[i]
                        remaining = [opt for opt in spec['options'] if opt != current]
                        if remaining:
                            individual[i] = random.choice(remaining)
            return individual,
        
        self.toolbox.register("mutate", mutate_mixed, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Guardar especificaciones
        self.param_specs = param_specs
    
    def _map_individual_to_params(self, individual: List[Any]) -> List[Any]:
        """
        Mapea un individuo a valores de parámetros según la configuración.
        
        Args:
            individual: Individuo a mapear.
            
        Returns:
            Lista de valores de parámetros mapeados.
        """
        # Si tenemos rangos de parámetros reales, mapear de [0,1] a rangos específicos
        if hasattr(self, 'param_ranges') and all(isinstance(r[0], float) for r in self.param_ranges):
            mapped = []
            for i, (min_val, max_val) in enumerate(self.param_ranges):
                mapped.append(min_val + individual[i] * (max_val - min_val))
            return mapped
        
        # Para enteros y categóricos, los valores ya están en el rango correcto
        return individual
    
    def _evaluate_wrapper(self, individual: List[Any]) -> Tuple[float,]:
        """
        Wrapper para la función de evaluación.
        
        Args:
            individual: Individuo a evaluar.
            
        Returns:
            Tupla con valor de fitness.
        """
        if self.evaluation_function is None:
            raise ValueError("Función de evaluación no configurada.")
        
        try:
            # Mapear individuo a parámetros
            params = self._map_individual_to_params(individual)
            
            # Evaluar
            fitness = self.evaluation_function(params)
            
            # Actualizar contador
            self.total_evaluations += 1
            
            return (fitness,)
        except Exception as e:
            logger.error(f"Error en evaluación: {e}")
            return (-1.0,)  # Valor de fitness negativo para individuos que fallan
    
    @safe_loop(max_retries=2, delay=1)
    def evolve_parameters(self, 
                         param_type: str,
                         param_config: Union[List[Tuple[float, float]], List[Tuple[int, int]], List[List[Any]], List[Dict[str, Any]]],
                         eval_function: Optional[Callable] = None,
                         pop_size: int = 50,
                         n_generations: int = 20,
                         checkpoint_freq: int = 5,
                         **kwargs) -> Dict[str, Any]:
        """
        Evoluciona parámetros mediante algoritmos genéticos.
        
        Args:
            param_type: Tipo de parámetros ('real', 'integer', 'categorical', 'mixed').
            param_config: Configuración de parámetros según tipo.
            eval_function: Función de evaluación (opcional si ya se configuró).
            pop_size: Tamaño de la población.
            n_generations: Número de generaciones.
            checkpoint_freq: Frecuencia de guardado de checkpoints.
            **kwargs: Argumentos adicionales para el algoritmo.
            
        Returns:
            Diccionario con resultados de la evolución.
        """
        start_time = time.time()
        
        # Configurar función de evaluación
        if eval_function is not None:
            self.evaluation_function = eval_function
        
        if self.evaluation_function is None:
            raise ValueError("Debe proporcionar una función de evaluación.")
        
        # Configurar toolbox según tipo de parámetros
        if param_type == 'real':
            self._configure_for_real_values(param_config)
        elif param_type == 'integer':
            self._configure_for_integer_values(param_config)
        elif param_type == 'categorical':
            self._configure_for_categorical_values(param_config)
        elif param_type == 'mixed':
            self._configure_for_mixed_values(param_config)
        else:
            raise ValueError(f"Tipo de parámetro no soportado: {param_type}")
        
        # Registrar función de evaluación
        self.toolbox.register("evaluate", self._evaluate_wrapper)
        
        # Crear población inicial
        population = self.toolbox.population(n=pop_size)
        
        # Evaluar población inicial
        fitnesses = list(map(self.toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        
        # Extraer estadísticas iniciales
        fits = [ind.fitness.values[0] for ind in population]
        best_fitness = max(fits)
        avg_fitness = sum(fits) / len(fits)
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        # Registrar estadísticas para seguimiento
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Crear objeto para guardar el hall of fame (mejores individuos)
        hof = tools.HallOfFame(10)
        
        # Ejecutar algoritmo evolutivo
        logger.info(f"Iniciando evolución con población de {pop_size
(Content truncated due to size limit. Use line ranges to read in chunks)