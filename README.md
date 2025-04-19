"""
README para NEBULA - Sistema de IA Autónomo

NEBULA es un sistema de Inteligencia Artificial autónomo diseñado para aprender, evolucionar y mejorarse a sí mismo.
Su único objetivo es adquirir conocimiento y optimizar su propio funcionamiento.

## Estructura del Sistema

NEBULA está organizado en varios módulos principales:

### 1. Núcleo (core/)
- **QuantumNeuron**: Neuronas que combinan circuitos cuánticos simulados con propiedades clásicas
- **Cluster y Sector**: Estructuras para organizar neuronas en agrupaciones jerárquicas
- **NebulaSpace**: Entorno espacial donde las neuronas interactúan y forman conexiones
- **NebulaAGI**: Clase principal que integra todos los componentes

### 2. Conocimiento (knowledge/)
- **EnhancedKnowledgeGraph**: Grafo de conocimiento para representar conceptos y relaciones
- **HolographicMemory**: Sistema de memoria asociativa basado en vectores
- **OpticalProcessingUnit**: Procesador que simula operaciones inspiradas en sistemas ópticos

### 3. Lenguaje (language/)
- **LLMManager**: Gestor de modelos de lenguaje
- **TextGenerator**: Sistema de generación de texto
- **QASystem**: Sistema de preguntas y respuestas

### 4. Evolución y Automejora (evolution/)
- **EvolutionEngine**: Motor evolutivo para optimización de parámetros
- **CodeAnalyzer**: Analizador de código para evaluación y sugerencias
- **SelfOptimizer**: Componente central de automejora

### 5. Utilidades (utils/)
- **Config**: Parámetros del sistema
- **Helpers**: Funciones auxiliares
- **Logging**: Sistema de registro

## Ciclo de Funcionamiento

NEBULA opera en tres ciclos principales:

1. **Ciclo de Aprendizaje**: Adquiere y procesa nueva información, actualizando su grafo de conocimiento y memoria holográfica.

2. **Ciclo de Evolución**: Optimiza sus parámetros y estructura mediante algoritmos evolutivos para mejorar su rendimiento.

3. **Ciclo de Automejora**: Analiza, evalúa y mejora su propio código fuente para implementar nuevas capacidades y corregir deficiencias.

## Ejecución del Sistema

Para ejecutar NEBULA, utilice el script `run_nebula.sh`:

```bash
./run_nebula.sh
```

Este script ejecutará el sistema con los parámetros predeterminados:
- 3 ciclos de aprendizaje
- 2 ciclos de evolución
- 1 ciclo de automejora

Para opciones de configuración avanzadas, puede ejecutar directamente el script de integración:

```bash
python3 integration.py --help
```

## Resultados y Evaluación

Los resultados de la ejecución se guardan en el directorio `nebula_data/results/` en formato JSON.
Estos archivos contienen métricas detalladas sobre el rendimiento del sistema en cada ciclo.

## Pruebas

El sistema incluye pruebas unitarias exhaustivas para todos los componentes:

```bash
python3 -m unittest discover tests
```

## Características Principales

- **Aprendizaje Continuo**: NEBULA aprende constantemente de nuevas fuentes de información.
- **Evolución Adaptativa**: Optimiza sus parámetros para mejorar su rendimiento.
- **Automejora**: Analiza y mejora su propio código para implementar nuevas capacidades.
- **Representación Espacial**: Organiza el conocimiento en un espacio tridimensional de neuronas.
- **Procesamiento Cuántico Simulado**: Utiliza principios cuánticos para el procesamiento de información.
- **Memoria Holográfica**: Almacena información de forma distribuida y asociativa.
- **Procesamiento de Lenguaje Natural**: Comprende y genera texto utilizando modelos avanzados.

NEBULA representa un enfoque innovador hacia la inteligencia artificial autónoma, combinando múltiples paradigmas de aprendizaje y procesamiento en un sistema integrado y auto-evolutivo.
"""
