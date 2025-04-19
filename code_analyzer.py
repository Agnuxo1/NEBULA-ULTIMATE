"""
Implementación del Analizador de Código para NEBULA.

Esta clase proporciona funcionalidades para analizar, evaluar y sugerir mejoras
en el código fuente del sistema NEBULA.
"""

import logging
import time
import os
import re
import ast
import inspect
import importlib
import pkgutil
from typing import Any, Dict, List, Optional, Tuple, Union, Set
from pathlib import Path
import json

from utils.config import PARAMETERS
from utils.helpers import safe_loop

logger = logging.getLogger("NEBULA.CodeAnalyzer")

class CodeAnalyzer:
    """
    Analizador de código que evalúa la calidad, eficiencia y estructura
    del código fuente del sistema NEBULA.
    
    Características:
    - Análisis estático de código
    - Detección de patrones y anti-patrones
    - Evaluación de complejidad y mantenibilidad
    - Sugerencias de mejora y optimización
    """
    
    def __init__(self, llm_manager=None):
        """
        Inicializa el analizador de código.
        
        Args:
            llm_manager: Gestor de modelos LLM para análisis avanzado.
        """
        logger.info("Inicializando CodeAnalyzer...")
        self.llm_manager = llm_manager
        
        # Directorio raíz del proyecto
        self.project_root = Path(PARAMETERS["PROJECT_ROOT"])
        
        # Directorio para informes
        self.reports_dir = PARAMETERS["BACKUP_DIRECTORY"] / "code_analysis"
        os.makedirs(self.reports_dir, exist_ok=True)
        
        # Historial de análisis
        self.analysis_history = []
        
        # Contadores y estadísticas
        self.total_analyses = 0
        self.last_analysis_time = 0
        
        # Patrones y reglas
        self._initialize_patterns()
        
        logger.info("CodeAnalyzer inicializado correctamente.")
    
    def _initialize_patterns(self):
        """Inicializa patrones y reglas para análisis de código."""
        # Patrones de código problemático
        self.code_smells = {
            "long_method": {
                "description": "Método demasiado largo (más de 50 líneas)",
                "severity": "medium"
            },
            "too_many_parameters": {
                "description": "Método con demasiados parámetros (más de 5)",
                "severity": "medium"
            },
            "complex_conditional": {
                "description": "Condicional demasiado complejo",
                "severity": "high"
            },
            "duplicate_code": {
                "description": "Código duplicado",
                "severity": "high"
            },
            "global_variable": {
                "description": "Uso de variables globales",
                "severity": "medium"
            },
            "nested_loops": {
                "description": "Bucles anidados profundos",
                "severity": "medium"
            },
            "large_class": {
                "description": "Clase demasiado grande (más de 300 líneas)",
                "severity": "medium"
            },
            "commented_code": {
                "description": "Código comentado",
                "severity": "low"
            },
            "magic_number": {
                "description": "Número mágico (constante sin nombre)",
                "severity": "low"
            },
            "empty_catch": {
                "description": "Bloque catch vacío",
                "severity": "high"
            }
        }
        
        # Patrones de buenas prácticas
        self.good_practices = {
            "docstring": {
                "description": "Documentación adecuada",
                "importance": "high"
            },
            "type_hints": {
                "description": "Uso de type hints",
                "importance": "medium"
            },
            "error_handling": {
                "description": "Manejo adecuado de errores",
                "importance": "high"
            },
            "modular_design": {
                "description": "Diseño modular",
                "importance": "high"
            },
            "consistent_naming": {
                "description": "Nomenclatura consistente",
                "importance": "medium"
            },
            "unit_tests": {
                "description": "Pruebas unitarias",
                "importance": "high"
            }
        }
    
    def _get_python_files(self, directory: Path) -> List[Path]:
        """
        Obtiene todos los archivos Python en un directorio y sus subdirectorios.
        
        Args:
            directory: Directorio a analizar.
            
        Returns:
            Lista de rutas a archivos Python.
        """
        python_files = []
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def _parse_file(self, file_path: Path) -> Optional[ast.Module]:
        """
        Parsea un archivo Python a un AST.
        
        Args:
            file_path: Ruta al archivo.
            
        Returns:
            AST del archivo o None si hay error.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            return ast.parse(content, filename=str(file_path))
        except Exception as e:
            logger.error(f"Error al parsear {file_path}: {e}")
            return None
    
    def _count_lines(self, file_path: Path) -> Dict[str, int]:
        """
        Cuenta líneas de código, comentarios y espacios en blanco.
        
        Args:
            file_path: Ruta al archivo.
            
        Returns:
            Diccionario con conteos.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            blank_lines = sum(1 for line in lines if line.strip() == '')
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            
            # Detectar líneas de docstring
            content = ''.join(lines)
            tree = ast.parse(content)
            
            docstring_lines = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                    if ast.get_docstring(node):
                        docstring = ast.get_docstring(node)
                        docstring_lines += len(docstring.split('\n'))
            
            code_lines = total_lines - blank_lines - comment_lines - docstring_lines
            
            return {
                'total': total_lines,
                'code': code_lines,
                'comments': comment_lines,
                'docstrings': docstring_lines,
                'blank': blank_lines
            }
        except Exception as e:
            logger.error(f"Error al contar líneas en {file_path}: {e}")
            return {
                'total': 0,
                'code': 0,
                'comments': 0,
                'docstrings': 0,
                'blank': 0
            }
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """
        Calcula la complejidad ciclomática de un nodo AST.
        
        Args:
            node: Nodo AST.
            
        Returns:
            Complejidad ciclomática.
        """
        complexity = 1  # Base complexity
        
        # Incrementar por cada estructura de control
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp) and isinstance(child.op, ast.And):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.BoolOp) and isinstance(child.op, ast.Or):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _analyze_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """
        Analiza una función o método.
        
        Args:
            node: Nodo AST de la función.
            
        Returns:
            Diccionario con análisis.
        """
        # Obtener código fuente
        source_lines = ast.get_source_segment(node.parent, node).split('\n')
        
        # Contar líneas
        total_lines = len(source_lines)
        
        # Obtener docstring
        docstring = ast.get_docstring(node)
        has_docstring = docstring is not None
        
        # Calcular complejidad
        complexity = self._calculate_complexity(node)
        
        # Analizar parámetros
        params = []
        for arg in node.args.args:
            param = {
                'name': arg.arg,
                'has_type': arg.annotation is not None
            }
            if arg.annotation:
                param['type'] = ast.unparse(arg.annotation)
            params.append(param)
        
        # Verificar return type
        has_return_type = node.returns is not None
        return_type = ast.unparse(node.returns) if node.returns else None
        
        # Detectar problemas
        issues = []
        
        # Método largo
        if total_lines > 50:
            issues.append({
                'type': 'long_method',
                'description': f"Función demasiado larga ({total_lines} líneas)",
                'severity': 'medium'
            })
        
        # Demasiados parámetros
        if len(params) > 5:
            issues.append({
                'type': 'too_many_parameters',
                'description': f"Demasiados parámetros ({len(params)})",
                'severity': 'medium'
            })
        
        # Complejidad alta
        if complexity > 10:
            issues.append({
                'type': 'high_complexity',
                'description': f"Complejidad ciclomática alta ({complexity})",
                'severity': 'high'
            })
        
        # Sin docstring
        if not has_docstring:
            issues.append({
                'type': 'missing_docstring',
                'description': "Falta docstring",
                'severity': 'medium'
            })
        
        # Detectar bucles anidados
        nested_loops = []
        for_loops = []
        
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                for_loops.append(child)
        
        # Verificar anidamiento
        for loop in for_loops:
            nested_count = 0
            for inner_loop in for_loops:
                if inner_loop != loop:
                    # Verificar si inner_loop está dentro de loop
                    if loop.lineno <= inner_loop.lineno and loop.end_lineno >= inner_loop.end_lineno:
                        nested_count += 1
            
            if nested_count > 1:
                nested_loops.append({
                    'line': loop.lineno,
                    'nested_count': nested_count
                })
        
        if nested_loops:
            issues.append({
                'type': 'nested_loops',
                'description': f"Bucles anidados detectados ({len(nested_loops)})",
                'severity': 'medium',
                'details': nested_loops
            })
        
        return {
            'name': node.name,
            'lines': total_lines,
            'complexity': complexity,
            'has_docstring': has_docstring,
            'docstring_quality': self._evaluate_docstring(docstring) if has_docstring else 0,
            'params': params,
            'param_count': len(params),
            'type_hint_coverage': sum(1 for p in params if p['has_type']) / max(1, len(params)),
            'has_return_type': has_return_type,
            'return_type': return_type,
            'issues': issues
        }
    
    def _analyze_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """
        Analiza una clase.
        
        Args:
            node: Nodo AST de la clase.
            
        Returns:
            Diccionario con análisis.
        """
        # Obtener código fuente
        source_lines = ast.get_source_segment(node.parent, node).split('\n')
        
        # Contar líneas
        total_lines = len(source_lines)
        
        # Obtener docstring
        docstring = ast.get_docstring(node)
        has_docstring = docstring is not None
        
        # Analizar métodos
        methods = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                methods.append(self._analyze_function(child))
        
        # Detectar problemas
        issues = []
        
        # Clase grande
        if total_lines > 300:
            issues.append({
                'type': 'large_class',
                'description': f"Clase demasiado grande ({total_lines} líneas)",
                'severity': 'medium'
            })
        
        # Sin docstring
        if not has_docstring:
            issues.append({
                'type': 'missing_docstring',
                'description': "Falta docstring",
                'severity': 'medium'
            })
        
        # Demasiados métodos
        if len(methods) > 20:
            issues.append({
                'type': 'too_many_methods',
                'description': f"Demasiados métodos ({len(methods)})",
                'severity': 'medium'
            })
        
        return {
            'name': node.name,
            'lines': total_lines,
            'has_docstring': has_docstring,
            'docstring_quality': self._evaluate_docstring(docstring) if has_docstring else 0,
            'methods': methods,
            'method_count': len(methods),
            'issues': issues,
            'inheritance': [ast.unparse(base) for base in node.bases] if node.bases else []
        }
    
    def _evaluate_docstring(self, docstring: Optional[str]) -> float:
        """
        Evalúa la calidad de un docstring.
        
        Args:
            docstring: Docstring a evaluar.
            
        Returns:
            Puntuación de calidad (0-1).
        """
        if not docstring:
            return 0.0
        
        score = 0.0
        max_score = 4.0
        
        # Longitud mínima
        if len(docstring) > 10:
            score += 1.0
        
        # Descripción de parámetros
        if re.search(r'Args:|Parameters:', docstring):
            score += 1.0
        
        # Descripción de retorno
        if re.search(r'Returns:|Return:', docstring):
            score += 1.0
        
        # Ejemplos o notas adicionales
        if re.search(r'Example|Note|Usage:', docstring):
            score += 1.0
        
        return score / max_score
    
    def _detect_code_smells(self, file_path: Path, tree: ast.Module) -> List[Dict[str, Any]]:
        """
        Detecta code smells en un archivo.
        
        Args:
            file_path: Ruta al archivo.
            tree: AST del archivo.
            
        Returns:
            Lista de code smells detectados.
        """
        smells = []
        
        # Leer contenido del archivo
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Detectar código comentado
        commented_code_pattern = re.compile(r'^\s*#\s*(def|class|if|for|while|return|import|from)')
        for i, line in enumerate(lines):
            if commented_code_pattern.match(line):
                smells.append({
                    'type': 'commented_code',
                    'line': i + 1,
                    'description': "Código comentado",
                    'severity': 'low'
                })
        
        # Detectar números mágicos
        magic_number_pattern = re.compile(r'\b[0-9]+\b')
        for i, line in 
(Content truncated due to size limit. Use line ranges to read in chunks)