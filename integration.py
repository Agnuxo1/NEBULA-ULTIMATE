"""
Script de integración para NEBULA.

Este script implementa la integración de todos los componentes de NEBULA
y ejecuta un ciclo completo de aprendizaje, evolución y automejora.
"""

import os
import sys
import time
import logging
import numpy as np
import torch
from pathlib import Path
import argparse

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("nebula_integration.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("NEBULA.Integration")

# Importar componentes de NEBULA
from core.nebula_space import NebulaSpace
from core.nebula_agi import NebulaAGI
from knowledge.knowledge_graph import EnhancedKnowledgeGraph
from knowledge.holographic_memory import HolographicMemory
from knowledge.optical_processor import OpticalProcessingUnit
from language.llm_manager import LLMManager
from language.text_generator import TextGenerator
from language.qa_system import QASystem
from evolution.evolution_engine import EvolutionEngine
from evolution.code_analyzer import CodeAnalyzer
from evolution.self_optimizer import SelfOptimizer
from utils.config import PARAMETERS
from utils.helpers import safe_loop

def setup_nebula(base_dir, config=None):
    """
    Configura e inicializa todos los componentes de NEBULA.
    
    Args:
        base_dir: Directorio base para almacenamiento.
        config: Configuración personalizada (opcional).
        
    Returns:
        Instancia de NebulaAGI inicializada.
    """
    logger.info("Iniciando configuración de NEBULA...")
    
    # Crear directorios necesarios
    data_dir = Path(base_dir) / "data"
    models_dir = Path(base_dir) / "models"
    checkpoints_dir = Path(base_dir) / "checkpoints"
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Inicializar componentes
    
    # 1. Componentes del núcleo
    logger.info("Inicializando componentes del núcleo...")
    nebula_space = NebulaSpace(
        dimensions=3,
        size=1.0,
        initial_neurons=100,
        initial_clusters=10,
        initial_sectors=3
    )
    
    # 2. Componentes de conocimiento
    logger.info("Inicializando componentes de conocimiento...")
    knowledge_graph = EnhancedKnowledgeGraph(
        storage_path=str(data_dir / "knowledge_graph")
    )
    
    holographic_memory = HolographicMemory(
        vector_dimension=256,
        storage_path=str(data_dir / "holographic_memory")
    )
    
    optical_processor = OpticalProcessingUnit(
        dimensions=(64, 64)
    )
    
    # 3. Componentes de lenguaje
    logger.info("Inicializando componentes de lenguaje...")
    llm_manager = LLMManager(
        cache_dir=str(models_dir),
        max_models=3
    )
    
    text_generator = TextGenerator(
        llm_manager=llm_manager,
        knowledge_graph=knowledge_graph
    )
    
    qa_system = QASystem(
        llm_manager=llm_manager,
        knowledge_graph=knowledge_graph,
        holographic_memory=holographic_memory
    )
    
    # 4. Componentes de evolución y automejora
    logger.info("Inicializando componentes de evolución y automejora...")
    evolution_engine = EvolutionEngine(
        checkpoint_dir=str(checkpoints_dir / "evolution")
    )
    
    code_analyzer = CodeAnalyzer(
        llm_manager=llm_manager
    )
    
    self_optimizer = SelfOptimizer(
        evolution_engine=evolution_engine,
        code_analyzer=code_analyzer,
        llm_manager=llm_manager
    )
    
    # 5. Integrar todo en NebulaAGI
    logger.info("Integrando componentes en NebulaAGI...")
    nebula_agi = NebulaAGI(
        nebula_space=nebula_space,
        knowledge_graph=knowledge_graph,
        holographic_memory=holographic_memory,
        optical_processor=optical_processor,
        llm_manager=llm_manager,
        text_generator=text_generator,
        qa_system=qa_system,
        evolution_engine=evolution_engine,
        code_analyzer=code_analyzer,
        self_optimizer=self_optimizer,
        config=config
    )
    
    logger.info("Configuración de NEBULA completada.")
    return nebula_agi

def run_learning_cycle(nebula, cycles=1, learning_sources=None):
    """
    Ejecuta ciclos de aprendizaje en NEBULA.
    
    Args:
        nebula: Instancia de NebulaAGI.
        cycles: Número de ciclos a ejecutar.
        learning_sources: Fuentes de aprendizaje (opcional).
        
    Returns:
        Resultados del aprendizaje.
    """
    logger.info(f"Iniciando {cycles} ciclos de aprendizaje...")
    
    if learning_sources is None:
        # Fuentes de aprendizaje predeterminadas
        learning_sources = [
            {
                "type": "text",
                "content": """
                La inteligencia artificial (IA) es la simulación de procesos de inteligencia humana por parte de máquinas, 
                especialmente sistemas informáticos. Estos procesos incluyen el aprendizaje (la adquisición de información 
                y reglas para el uso de la información), el razonamiento (usando las reglas para llegar a conclusiones 
                aproximadas o definitivas) y la autocorrección. Aplicaciones particulares de la IA incluyen sistemas expertos, 
                reconocimiento de voz y visión artificial.
                
                El aprendizaje profundo es una forma de aprendizaje automático que utiliza redes neuronales artificiales 
                con múltiples capas (de ahí el término "profundo"). Estas redes pueden aprender representaciones jerárquicas 
                de datos, donde cada capa transforma la entrada en una representación más abstracta.
                
                Las redes neuronales cuánticas son modelos que combinan principios de la computación cuántica con redes 
                neuronales artificiales. Estos modelos aprovechan fenómenos cuánticos como la superposición y el entrelazamiento 
                para realizar cálculos que serían ineficientes o imposibles con métodos clásicos.
                """
            },
            {
                "type": "concept",
                "name": "Aprendizaje Automático",
                "properties": {
                    "definición": "Campo de la IA que permite a las máquinas aprender de los datos sin ser programadas explícitamente",
                    "importancia": 0.9,
                    "aplicaciones": ["clasificación", "regresión", "clustering", "reducción de dimensionalidad"]
                }
            },
            {
                "type": "relation",
                "source": "Inteligencia Artificial",
                "target": "Aprendizaje Automático",
                "relation_type": "incluye",
                "strength": 0.9
            }
        ]
    
    results = []
    
    for cycle in range(1, cycles + 1):
        logger.info(f"Ejecutando ciclo de aprendizaje {cycle}/{cycles}...")
        
        # Ejecutar ciclo de aprendizaje
        cycle_result = nebula.learning_cycle(learning_sources)
        
        # Guardar resultados
        results.append({
            "cycle": cycle,
            "timestamp": time.time(),
            "metrics": cycle_result
        })
        
        logger.info(f"Ciclo {cycle} completado. Métricas: {cycle_result}")
    
    logger.info(f"Completados {cycles} ciclos de aprendizaje.")
    return results

def run_evolution_cycle(nebula, cycles=1):
    """
    Ejecuta ciclos de evolución en NEBULA.
    
    Args:
        nebula: Instancia de NebulaAGI.
        cycles: Número de ciclos a ejecutar.
        
    Returns:
        Resultados de la evolución.
    """
    logger.info(f"Iniciando {cycles} ciclos de evolución...")
    
    results = []
    
    for cycle in range(1, cycles + 1):
        logger.info(f"Ejecutando ciclo de evolución {cycle}/{cycles}...")
        
        # Ejecutar ciclo de evolución
        cycle_result = nebula.evolution_cycle()
        
        # Guardar resultados
        results.append({
            "cycle": cycle,
            "timestamp": time.time(),
            "metrics": cycle_result
        })
        
        logger.info(f"Ciclo {cycle} completado. Métricas: {cycle_result}")
    
    logger.info(f"Completados {cycles} ciclos de evolución.")
    return results

def run_self_improvement_cycle(nebula, cycles=1):
    """
    Ejecuta ciclos de automejora en NEBULA.
    
    Args:
        nebula: Instancia de NebulaAGI.
        cycles: Número de ciclos a ejecutar.
        
    Returns:
        Resultados de la automejora.
    """
    logger.info(f"Iniciando {cycles} ciclos de automejora...")
    
    results = []
    
    for cycle in range(1, cycles + 1):
        logger.info(f"Ejecutando ciclo de automejora {cycle}/{cycles}...")
        
        # Ejecutar ciclo de automejora
        cycle_result = nebula.self_improvement_cycle()
        
        # Guardar resultados
        results.append({
            "cycle": cycle,
            "timestamp": time.time(),
            "metrics": cycle_result
        })
        
        logger.info(f"Ciclo {cycle} completado. Métricas: {cycle_result}")
    
    logger.info(f"Completados {cycles} ciclos de automejora.")
    return results

def evaluate_nebula(nebula):
    """
    Evalúa el rendimiento y capacidades de NEBULA.
    
    Args:
        nebula: Instancia de NebulaAGI.
        
    Returns:
        Resultados de la evaluación.
    """
    logger.info("Iniciando evaluación de NEBULA...")
    
    # Evaluar capacidades de respuesta a preguntas
    qa_questions = [
        "¿Qué es la inteligencia artificial?",
        "¿Cómo se relaciona el aprendizaje profundo con las redes neuronales?",
        "¿Qué son las redes neuronales cuánticas?"
    ]
    
    qa_results = []
    for question in qa_questions:
        answer = nebula.qa_system.answer_question(question)
        qa_results.append({
            "question": question,
            "answer": answer
        })
    
    # Evaluar capacidades de generación de texto
    text_prompts = [
        "Explica el concepto de aprendizaje automático",
        "Describe las aplicaciones de la inteligencia artificial en la medicina"
    ]
    
    text_results = []
    for prompt in text_prompts:
        text = nebula.text_generator.generate_text(prompt, max_length=200)
        text_results.append({
            "prompt": prompt,
            "text": text
        })
    
    # Evaluar grafo de conocimiento
    kg_stats = {
        "total_concepts": len(nebula.knowledge_graph.graph.nodes),
        "total_relations": len(nebula.knowledge_graph.graph.edges),
        "central_concepts": [node for node, degree in sorted(
            nebula.knowledge_graph.graph.degree(),
            key=lambda x: x[1],
            reverse=True
        )[:5]]
    }
    
    # Evaluar memoria holográfica
    memory_stats = {
        "total_items": sum(len(space) for space in nebula.holographic_memory.memory_spaces.values()),
        "memory_spaces": list(nebula.holographic_memory.memory_spaces.keys())
    }
    
    # Evaluar espacio neuronal
    space_stats = {
        "total_neurons": len(nebula.nebula_space.neurons),
        "total_clusters": len(nebula.nebula_space.clusters),
        "total_sectors": len(nebula.nebula_space.sectors),
        "total_connections": sum(len(connections) for connections in nebula.nebula_space.connections.values())
    }
    
    # Compilar resultados
    evaluation_results = {
        "timestamp": time.time(),
        "qa_evaluation": qa_results,
        "text_generation": text_results,
        "knowledge_graph": kg_stats,
        "holographic_memory": memory_stats,
        "neural_space": space_stats
    }
    
    logger.info("Evaluación de NEBULA completada.")
    return evaluation_results

def main():
    """Función principal para ejecutar la integración de NEBULA."""
    parser = argparse.ArgumentParser(description="NEBULA - Sistema de IA Autónomo")
    parser.add_argument("--base-dir", type=str, default="./nebula_data",
                        help="Directorio base para almacenamiento")
    parser.add_argument("--learning-cycles", type=int, default=3,
                        help="Número de ciclos de aprendizaje")
    parser.add_argument("--evolution-cycles", type=int, default=2,
                        help="Número de ciclos de evolución")
    parser.add_argument("--improvement-cycles", type=int, default=1,
                        help="Número de ciclos de automejora")
    parser.add_argument("--save-results", action="store_true",
                        help="Guardar resultados de la ejecución")
    
    args = parser.parse_args()
    
    logger.info("Iniciando NEBULA...")
    
    try:
        # Configurar NEBULA
        nebula = setup_nebula(args.base_dir)
        
        # Ejecutar ciclos de aprendizaje
        learning_results = run_learning_cycle(nebula, cycles=args.learning_cycles)
        
        # Ejecutar ciclos de evolución
        evolution_results = run_evolution_cycle(nebula, cycles=args.evolution_cycles)
        
        # Ejecutar ciclos de automejora
        improvement_results = run_self_improvement_cycle(nebula, cycles=args.improvement_cycles)
        
        # Evaluar NEBULA
        evaluation = evaluate_nebula(nebula)
        
        # Guardar resultados
        if args.save_results:
            results_dir = Path(args.base_dir) / "results"
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = int(time.time())
            results_file = results_dir / f"nebula_results_{timestamp}.json"
            
            import json
            with open(results_file, 'w') as f:
                json.dump({
                    "learning_results": learning_results,
                    "evolution_results": evolution_results,
                    "improvement_results": improvement_results,
                    "evaluation": evaluation
                }, f, indent=2)
            
            logger.info(f"Resultados guardados en {results_file}")
        
        logger.info("Ejecución de NEBULA completada con éxito.")
        
    except Exception as e:
        logger.error(f"Error durante la ejecución de NEBULA: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
